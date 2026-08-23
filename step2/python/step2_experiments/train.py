"""Two-T4 vertical-slice gate and bounded C1-candidate training start."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
import tomllib
from typing import Any

from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs, set_seed
import torch
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup

import step2_world_py

from .data import generate_torch_batch, tensorize_rollout, world_kwargs
from .model import (
    Step2Config,
    Step2ForTrajectoryPrediction,
    assert_selected_parameter_report,
    assert_selected_profile,
    parameter_report,
)


def load_config(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def resolve_project_path(config_path: Path, relative: str) -> Path:
    root = config_path.resolve().parents[3]
    return root / relative


def build_optimizer(model: torch.nn.Module, learning_rate: float, weight_decay: float) -> AdamW:
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or "embedding" in name or "norm" in name or "queries" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
        betas=(0.9, 0.95),
        eps=1.0e-8,
    )


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: tensor.to(device, non_blocking=False) for name, tensor in batch.items()}


def model_checksum(model: torch.nn.Module) -> torch.Tensor:
    checksum = torch.zeros(3, device=next(model.parameters()).device, dtype=torch.float64)
    with torch.no_grad():
        for parameter in model.parameters():
            values = parameter.detach().to(dtype=torch.float64)
            checksum[0] += values.sum()
            checksum[1] += values.square().sum()
            checksum[2] += values.numel()
    return checksum


def aggregate_scalar(accelerator: Accelerator, value: torch.Tensor) -> float:
    return accelerator.gather(value.detach().float().reshape(1)).mean().item()


@torch.no_grad()
def teacher_forced_eval(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: dict[str, Any],
    start_index: int,
) -> dict[str, Any]:
    run = config["run"]
    world = config["world"]
    sequence_length = int(config["model"]["sequence_length"])
    batch_size = int(run["per_device_batch_size"])
    model.eval()
    records: list[torch.Tensor] = []
    for batch_index in range(int(run["eval_batches"])):
        local_start = (
            start_index
            + (batch_index * accelerator.num_processes + accelerator.process_index) * batch_size
        )
        batch, metadata = generate_torch_batch(
            seed=int(world["validation_seed"]),
            start_index=local_start,
            batch_size=batch_size,
            max_tokens=sequence_length,
            world=world,
        )
        batch = move_batch(batch, accelerator.device)
        output = model(**batch)
        action_abs = (output.action_predictions - batch["action_targets"]).abs()
        outcome_abs = (output.outcome_predictions - batch["outcome_targets"]).abs()
        for row, dimension in enumerate(metadata["dimensions"]):
            action_mask = batch["action_target_mask"][row]
            outcome_mask = batch["outcome_target_mask"][row]
            records.append(
                torch.tensor(
                    [
                        float(dimension),
                        float((action_abs[row] * action_mask).sum().item()),
                        float(action_mask.sum().item()),
                        float((outcome_abs[row] * outcome_mask).sum().item()),
                        float(outcome_mask.sum().item()),
                    ],
                    device=accelerator.device,
                )
            )
    gathered = accelerator.gather_for_metrics(torch.stack(records)).cpu()
    result: dict[str, Any] = {"by_dimension": {}}
    total_action_sum = total_action_count = total_outcome_sum = total_outcome_count = 0.0
    for dimension in range(int(world["d_min"]), int(world["d_max"]) + 1):
        rows = gathered[gathered[:, 0] == float(dimension)]
        action_sum = rows[:, 1].sum().item()
        action_count = rows[:, 2].sum().item()
        outcome_sum = rows[:, 3].sum().item()
        outcome_count = rows[:, 4].sum().item()
        result["by_dimension"][str(dimension)] = {
            "episodes": int(rows.shape[0]),
            "action_l1": action_sum / max(action_count, 1.0),
            "outcome_l1": outcome_sum / max(outcome_count, 1.0),
        }
        total_action_sum += action_sum
        total_action_count += action_count
        total_outcome_sum += outcome_sum
        total_outcome_count += outcome_count
    result["action_l1"] = total_action_sum / max(total_action_count, 1.0)
    result["outcome_l1"] = total_outcome_sum / max(total_outcome_count, 1.0)
    return result


@torch.no_grad()
def closed_loop_eval(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    seed: int,
    start_index: int,
    episodes_per_rank: int,
    use_oracle: bool = False,
) -> dict[str, Any]:
    world = config["world"]
    sequence_length = int(config["model"]["sequence_length"])
    local_start = start_index + accelerator.process_index * episodes_per_rank
    rollouts = step2_world_py.RolloutBatch(
        seed=seed,
        start_index=local_start,
        batch_size=episodes_per_rank,
        max_tokens=sequence_length,
        **world_kwargs(world),
    )
    model.eval()
    while not rollouts.all_done():
        raw = rollouts.learner_batch()
        if use_oracle:
            actions = rollouts.privileged_oracle_actions()
        else:
            tensors = move_batch(tensorize_rollout(raw, "cpu"), accelerator.device)
            output = model(**tensors)
            predictions = output.action_predictions[..., 0].detach().float().cpu()
            actions = []
            for row, dimension in enumerate(raw["dimensions"]):
                if raw["done"][row]:
                    actions.append([])
                    continue
                positions = raw["query_positions"][row][:dimension]
                actions.append([float(predictions[row, position]) for position in positions])
        rollouts.step(actions)
    summary = rollouts.summary()
    local = torch.tensor(
        [
            [
                float(summary["dimensions"][i]),
                1.0 if summary["success"][i] else 0.0,
                float(summary["terminal_error"][i]),
                float(summary["steps"][i]),
            ]
            for i in range(episodes_per_rank)
        ],
        device=accelerator.device,
    )
    gathered = accelerator.gather_for_metrics(local).cpu()
    result: dict[str, Any] = {
        "episodes": int(gathered.shape[0]),
        "success_rate": float(gathered[:, 1].mean().item()),
        "terminal_error": float(gathered[:, 2].mean().item()),
        "mean_steps": float(gathered[:, 3].mean().item()),
        "by_dimension": {},
    }
    for dimension in range(int(world["d_min"]), int(world["d_max"]) + 1):
        rows = gathered[gathered[:, 0] == float(dimension)]
        result["by_dimension"][str(dimension)] = {
            "episodes": int(rows.shape[0]),
            "success_rate": float(rows[:, 1].mean().item()) if rows.numel() else 0.0,
            "terminal_error": float(rows[:, 2].mean().item()) if rows.numel() else math.nan,
        }
    return result


def run_overfit_gate(
    accelerator: Accelerator,
    model_config: Step2Config,
    config: dict[str, Any],
) -> tuple[dict[str, Any], torch.nn.Module]:
    run = config["run"]
    world = config["world"]
    batch_size = int(run["overfit_per_device_batch_size"])
    sequence_length = int(config["model"]["sequence_length"])
    set_seed(int(run["seed"]))
    model = Step2ForTrajectoryPrediction(model_config)
    optimizer = build_optimizer(model, float(run["learning_rate"]), float(run["weight_decay"]))
    updates = int(run["overfit_updates"])
    warmup = int(run.get("overfit_warmup_updates", 0))
    if warmup < 0 or warmup >= updates:
        raise ValueError("overfit_warmup_updates must satisfy 0 <= warmup < overfit_updates")
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup,
        num_training_steps=updates,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    local_start = accelerator.process_index * batch_size
    fixed_batch, _ = generate_torch_batch(
        seed=int(world["overfit_seed"]),
        start_index=local_start,
        batch_size=batch_size,
        max_tokens=sequence_length,
        world=world,
    )
    fixed_batch = move_batch(fixed_batch, accelerator.device)
    model.eval()
    with torch.no_grad():
        initial = aggregate_scalar(accelerator, model(**fixed_batch).loss)
    trace = [initial]
    model.train()
    for update in range(updates):
        optimizer.zero_grad(set_to_none=True)
        output = model(**fixed_batch)
        accelerator.backward(output.loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(model.parameters(), float(run["max_grad_norm"]))
        optimizer.step()
        scheduler.step()
        if update in {0, 1, 3, 7, 15, 31, 63, updates - 1}:
            trace.append(aggregate_scalar(accelerator, output.loss))
    model.eval()
    with torch.no_grad():
        final = aggregate_scalar(accelerator, model(**fixed_batch).loss)
    required = float(run["overfit_required_fraction"])
    passed = math.isfinite(initial) and math.isfinite(final) and final <= required * initial
    result = {
        "initial_loss": initial,
        "final_loss": final,
        "required_final_fraction": required,
        "observed_final_fraction": final / initial,
        "trace": trace,
        "updates": updates,
        "warmup_updates": warmup,
        "passed": passed,
    }
    if not passed:
        raise RuntimeError(f"real-batch overfit gate failed: {result}")
    return result, model


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_inventory(root: Path) -> dict[str, Any]:
    """Describe a remote-only recovery payload without moving it off Kaggle."""
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "remote_path": (Path(output_root_name(root)) / root.name).as_posix(),
        "size": sum(int(item["size"]) for item in files),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "files": files,
    }


def output_root_name(checkpoint_root: Path) -> str:
    # checkpoint_root is <output-root>/checkpoints/<label>.
    return checkpoint_root.parents[1].name + "/checkpoints"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = load_config(config_path)
    run = config["run"]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    if bool(run["require_two_t4"]):
        if torch.cuda.device_count() != 2:
            raise RuntimeError(f"expected exactly two CUDA devices, found {torch.cuda.device_count()}")
        names = [torch.cuda.get_device_name(index) for index in range(2)]
        if any("T4" not in name.upper() for name in names):
            raise RuntimeError(f"expected two T4 GPUs, found {names}")

    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision=str(run["mixed_precision"]),
        gradient_accumulation_steps=int(run["gradient_accumulation_steps"]),
        kwargs_handlers=[ddp],
    )
    if accelerator.num_processes != 2:
        raise RuntimeError(f"expected two training processes, found {accelerator.num_processes}")

    model_config_path = resolve_project_path(config_path, str(config["model"]["config"]))
    model_config = Step2Config.from_project_json(model_config_path)
    assert_selected_profile(model_config)
    started = time.time()

    overfit, diagnostic_model = run_overfit_gate(accelerator, model_config, config)
    diagnostic_closed_loop = closed_loop_eval(
        accelerator,
        diagnostic_model,
        config,
        seed=int(config["world"]["overfit_seed"]),
        start_index=0,
        episodes_per_rank=int(run["overfit_per_device_batch_size"]),
    )
    del diagnostic_model
    accelerator.free_memory()
    accelerator.wait_for_everyone()
    gate_progress: dict[str, Any] = {
        "overfit_gate": overfit,
        "diagnostic_closed_loop": diagnostic_closed_loop,
        "diagnostic_weights_discarded": True,
        "fresh_blank_lineage_initialized": False,
        "resume_smoke": {"passed": False, "attempted": False},
    }
    gate_progress_path = output_root / "architecture-gate-progress.json"
    if accelerator.is_main_process:
        gate_progress_path.write_text(
            json.dumps(gate_progress, indent=2, sort_keys=True), encoding="utf-8"
        )

    # The diagnostic weights are deliberately discarded. This reset begins the
    # true blank C1-candidate lineage.
    set_seed(int(run["seed"]))
    model = Step2ForTrajectoryPrediction(model_config)
    params = parameter_report(model)
    assert_selected_parameter_report(params)
    gate_progress.update(
        {
            "fresh_blank_lineage_initialized": True,
            "parameter_report": params,
        }
    )
    if accelerator.is_main_process:
        gate_progress_path.write_text(
            json.dumps(gate_progress, indent=2, sort_keys=True), encoding="utf-8"
        )
    optimizer = build_optimizer(model, float(run["learning_rate"]), float(run["weight_decay"]))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(run["warmup_updates"]),
        num_training_steps=int(run["max_updates"]),
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    sequence_length = int(config["model"]["sequence_length"])
    batch_size = int(run["per_device_batch_size"])
    loss_trace: list[dict[str, float | int]] = []
    resume_smoke: dict[str, Any] = {"passed": False}

    model.train()
    for update in range(int(run["max_updates"])):
        local_start = (update * accelerator.num_processes + accelerator.process_index) * batch_size
        batch, _ = generate_torch_batch(
            seed=int(config["world"]["train_seed"]),
            start_index=local_start,
            batch_size=batch_size,
            max_tokens=sequence_length,
            world=config["world"],
        )
        batch = move_batch(batch, accelerator.device)
        with accelerator.accumulate(model):
            optimizer.zero_grad(set_to_none=True)
            output = model(**batch)
            accelerator.backward(output.loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), float(run["max_grad_norm"]))
            optimizer.step()
            scheduler.step()

        if update % int(run["log_every"]) == 0 or update == int(run["max_updates"]) - 1:
            loss_trace.append(
                {
                    "update": update + 1,
                    "loss": aggregate_scalar(accelerator, output.loss),
                    "action_loss": aggregate_scalar(accelerator, output.action_loss),
                    "outcome_loss": aggregate_scalar(accelerator, output.outcome_loss),
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
            )

        if update + 1 == int(run["resume_smoke_update"]):
            resume_dir = output_root / "resume-smoke"
            accelerator.save_state(str(resume_dir))
            # save_state is main-process-only for model weights. Without an
            # explicit rendezvous, another rank can inspect the directory
            # before model.safetensors is visible and incorrectly fall back to
            # the legacy pytorch_model.bin filename.
            accelerator.wait_for_everyone()
            before = model_checksum(accelerator.unwrap_model(model))
            with torch.no_grad():
                next(accelerator.unwrap_model(model).parameters()).add_(0.5)
            accelerator.load_state(str(resume_dir))
            after = model_checksum(accelerator.unwrap_model(model))
            local_pass = torch.allclose(before, after, rtol=0.0, atol=1.0e-8)
            gathered_pass = accelerator.gather(torch.tensor([int(local_pass)], device=accelerator.device))
            resume_smoke = {
                "passed": bool(gathered_pass.bool().all().item()),
                "update": update + 1,
                "checksum_before": before.detach().cpu().tolist(),
                "checksum_after": after.detach().cpu().tolist(),
            }
            if not resume_smoke["passed"]:
                raise RuntimeError(f"accelerator checkpoint restore failed: {resume_smoke}")
            gate_progress["resume_smoke"] = {**resume_smoke, "attempted": True}
            if accelerator.is_main_process:
                gate_progress_path.write_text(
                    json.dumps(gate_progress, indent=2, sort_keys=True), encoding="utf-8"
                )

    validation = teacher_forced_eval(accelerator, model, config, start_index=0)
    rollout_count = int(config["preflight"]["rollout_episodes_per_rank"])
    closed_loop = closed_loop_eval(
        accelerator,
        model,
        config,
        seed=int(config["world"]["rollout_seed"]),
        start_index=0,
        episodes_per_rank=rollout_count,
    )
    oracle_closed_loop = closed_loop_eval(
        accelerator,
        model,
        config,
        seed=int(config["world"]["rollout_seed"]),
        start_index=0,
        episodes_per_rank=rollout_count,
        use_oracle=True,
    )

    accelerator.wait_for_everyone()
    final_state_dir = output_root / "checkpoints" / str(run["checkpoint_label"]) / "accelerate-state"
    accelerator.save_state(str(final_state_dir))
    accelerator.wait_for_everyone()
    model_dir = output_root / "checkpoints" / str(run["checkpoint_label"]) / "model"
    if accelerator.is_main_process:
        model_dir.mkdir(parents=True, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(
            model_dir,
            is_main_process=True,
            save_function=accelerator.save,
            state_dict=accelerator.get_state_dict(model),
            safe_serialization=True,
        )
        weights = model_dir / "model.safetensors"
        recovery = checkpoint_inventory(output_root / "checkpoints" / str(run["checkpoint_label"]))
        result = {
            "status": "complete",
            "checkpoint_label": str(run["checkpoint_label"]),
            "checkpoint_classification": "candidate; architecture-integrated but not yet a completed developmental session",
            "diagnostic_weights_discarded_before_lineage": True,
            "overfit_gate": overfit,
            "diagnostic_closed_loop": diagnostic_closed_loop,
            "resume_smoke": resume_smoke,
            "parameter_report": params,
            "loss_trace": loss_trace,
            "validation": validation,
            "closed_loop": closed_loop,
            "oracle_closed_loop": oracle_closed_loop,
            "world_versions": step2_world_py.versions(),
            "updates": int(run["max_updates"]),
            "global_episodes": int(run["max_updates"]) * batch_size * accelerator.num_processes,
            "elapsed_seconds": time.time() - started,
            "world_size": accelerator.num_processes,
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "torch_version": torch.__version__,
            "model_sha256": sha256_file(weights),
            "recovery_artifact": recovery,
            "root_seed": int(run["seed"]),
            "completed_at_unix": time.time(),
            "architecture_gate_passed": bool(overfit["passed"] and resume_smoke["passed"]),
        }
        (output_root / "training-result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True))
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
