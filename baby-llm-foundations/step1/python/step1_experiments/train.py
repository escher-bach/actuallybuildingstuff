"""Transformers Trainer entry point for teacher-conditioned causal training.

The training budget is deliberately expressed in *nominal global input tokens*:
``world_size * per_device_sequences * context_length`` for each forward pass.
This is the only token unit that has a fixed value before a batch is read, and
therefore the only one that can be used honestly for Trainer accumulation,
``max_steps``, and standard step-based checkpointing.  Supervised action tokens
are sparse and data dependent; they are reported separately and never used to
silently change an optimizer update.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch

from .artifacts import atomic_json
from .data import BinaryShard, SequenceDataset, collate
from .standard_stack import (
    EXPECTED_PARAMETER_COUNT,
    assert_model_contract,
    create_model,
    load_tokenizer,
)


@dataclass(frozen=True)
class TrainingPlan:
    """The exact Trainer-compatible interpretation of the training config."""

    world_size: int
    per_device_sequences: int
    context_length: int
    nominal_global_input_tokens_per_microstep: int
    gradient_accumulation_steps: int
    nominal_global_input_tokens_per_update: int
    token_budget: int
    max_steps: int
    checkpoint_interval_updates: int
    checkpoint_interval_nominal_global_input_tokens: int
    checkpoint_total_limit: int

    def report(self) -> dict:
        return {
            **asdict(self),
            "token_unit": "nominal global input tokens (world_size * per_device_sequences * context_length)",
            "supervised_token_unit": "action-label tokens after the causal shift; variable and reported separately",
        }


def distributed_world_size() -> int:
    """Read the launcher contract before Trainer constructs Accelerate state."""
    value = int(os.environ.get("WORLD_SIZE", "1"))
    if value < 1:
        raise ValueError(f"WORLD_SIZE must be positive, got {value}")
    return value


def training_plan(config: dict, world_size: int | None = None) -> TrainingPlan:
    """Validate exact global-input-token accounting for Trainer/Accelerate."""
    training, world = config["training"], config["world"]
    size = distributed_world_size() if world_size is None else world_size
    if size < 1:
        raise ValueError(f"world_size must be positive, got {size}")
    per_device, context = training["microbatch_sequences"], world["context_length"]
    if per_device < 1 or context < 1:
        raise ValueError("microbatch_sequences and context_length must both be positive")
    microstep_tokens = size * per_device * context
    requested_update_tokens = training["global_tokens_per_update"]
    if requested_update_tokens < microstep_tokens:
        raise ValueError(
            "global_tokens_per_update is smaller than one global microstep: "
            f"{requested_update_tokens} < {microstep_tokens}"
        )
    if requested_update_tokens % microstep_tokens:
        raise ValueError(
            "global_tokens_per_update must be divisible by world_size * "
            "microbatch_sequences * context_length so Trainer accumulation is exact: "
            f"{requested_update_tokens} % {microstep_tokens} != 0"
        )
    accumulation = requested_update_tokens // microstep_tokens
    token_budget = training["token_budget"]
    if token_budget < 0:
        raise ValueError("token_budget cannot be negative")
    if token_budget == 0 and config["run"]["mode"] != "rlvr":
        raise ValueError("token_budget must be positive for base training")
    if token_budget and token_budget % requested_update_tokens:
        raise ValueError(
            "token_budget must be divisible by global_tokens_per_update; partial "
            "updates would make the reported global input-token budget inaccurate: "
            f"{token_budget} % {requested_update_tokens} != 0"
        )
    checkpoint_interval_updates = training["checkpoint_interval_updates"]
    checkpoint_total_limit = training["checkpoint_total_limit"]
    if checkpoint_interval_updates < 1 or checkpoint_total_limit < 1:
        raise ValueError("checkpoint_interval_updates and checkpoint_total_limit must be positive")
    return TrainingPlan(
        world_size=size,
        per_device_sequences=per_device,
        context_length=context,
        nominal_global_input_tokens_per_microstep=microstep_tokens,
        gradient_accumulation_steps=accumulation,
        nominal_global_input_tokens_per_update=requested_update_tokens,
        token_budget=token_budget,
        max_steps=token_budget // requested_update_tokens,
        checkpoint_interval_updates=checkpoint_interval_updates,
        checkpoint_interval_nominal_global_input_tokens=checkpoint_interval_updates * requested_update_tokens,
        checkpoint_total_limit=checkpoint_total_limit,
    )


def _training_arguments(
    config: dict,
    checkpoint_dir: Path,
    max_steps: int,
    plan: TrainingPlan,
    *,
    save_strategy: str = "steps",
    save_steps: int | None = None,
    save_total_limit: int | None = None,
):
    from transformers import TrainingArguments

    training = config["training"]
    return TrainingArguments(
        output_dir=str(checkpoint_dir),
        per_device_train_batch_size=plan.per_device_sequences,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        learning_rate=training["learning_rate"],
        weight_decay=training["weight_decay"],
        lr_scheduler_type="cosine",
        warmup_ratio=training["warmup_fraction"],
        max_steps=max_steps,
        fp16=torch.cuda.is_available(),
        bf16=False,
        save_strategy=save_strategy,
        save_steps=plan.checkpoint_interval_updates if save_steps is None else save_steps,
        save_total_limit=plan.checkpoint_total_limit if save_total_limit is None else save_total_limit,
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=config["run"]["root_seed"],
        data_seed=config["run"]["root_seed"],
        # This asks Trainer/Accelerate to normalize sparse causal-LM labels by
        # their global token count.  Project code neither implements a loss nor
        # rescales distributed gradients.
        average_tokens_across_devices=True,
    )


def _trainer(
    config: dict,
    run_dir: Path,
    max_steps: int,
    plan: TrainingPlan,
    *,
    train_dataset=None,
    save_strategy: str = "steps",
    save_steps: int | None = None,
    save_total_limit: int | None = None,
):
    from transformers import Trainer, set_seed

    # The process seed is intentionally identical before Accelerate/DDP wraps
    # the model.  Trainer subsequently owns sampler/rank seeding and DDP sync.
    set_seed(config["run"]["root_seed"])
    shard = BinaryShard(run_dir / "datasets" / "train.bin")
    model = create_model()
    assert_model_contract(model)
    return Trainer(
        model=model,
        args=_training_arguments(
            config, run_dir / "checkpoints", max_steps, plan,
            save_strategy=save_strategy,
            save_steps=save_steps,
            save_total_limit=save_total_limit,
        ),
        train_dataset=shard if train_dataset is None else train_dataset,
        data_collator=partial(collate, context=config["world"]["context_length"]),
        processing_class=load_tokenizer(),
    ), shard


def _wait_for_everyone(trainer) -> None:
    trainer.accelerator.wait_for_everyone()


def _write_rank_zero_json(trainer, path: Path, value: dict) -> None:
    if trainer.is_world_process_zero():
        atomic_json(path, value)
    _wait_for_everyone(trainer)


def _model_artifact(run_dir: Path) -> Path:
    return run_dir / "model"


def _save_model_artifact(trainer, run_dir: Path, config: dict, plan: TrainingPlan) -> Path:
    """Save one complete HF artifact before any rank attempts to reload it."""
    artifact = _model_artifact(run_dir)
    # Trainer owns model serialization and writes only on its world process
    # zero.  Synchronize before adding project-owned artifact metadata.
    trainer.save_model(str(artifact))
    _wait_for_everyone(trainer)
    if trainer.is_world_process_zero():
        load_tokenizer().save_pretrained(artifact)
        atomic_json(artifact / "experiment.json", {
            "initialization": "random_from_local_gpt_neox_config",
            "parameter_count": EXPECTED_PARAMETER_COUNT,
            "config_hash": config["_meta"]["hash"],
            "model_config_sha256": hashlib.sha256((artifact / "config.json").read_bytes()).hexdigest(),
            "root_seed": config["run"]["root_seed"],
            "token_accounting": plan.report(),
        })
    # A second barrier prevents nonzero ranks from racing a partial tokenizer or
    # metadata directory during the preflight reload check.
    _wait_for_everyone(trainer)
    return artifact


def _batch_on_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {name: value.to(device) for name, value in batch.items()}


def _fixed_batch_loss(model, batch: dict[str, torch.Tensor], device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        loss = model(**_batch_on_device(batch, device)).loss
    if loss is None or not torch.isfinite(loss):
        raise FloatingPointError("fixed real-batch loss is absent or non-finite")
    return float(loss.detach().float().cpu())


def _scored_token_count(batch: dict[str, torch.Tensor]) -> torch.Tensor:
    # GPT-NeoX scores labels after shifting them one position to the left.
    return (batch["labels"][:, 1:] != -100).sum()


def _one_vs_two_process_loss_diagnostic(trainer, dataset, config: dict) -> dict:
    """Record two-rank loss normalization against a one-process reference.

    Each rank receives a different, real Rust sequence.  ``Trainer.compute_loss``
    and the model's standard ``num_items_in_batch`` path are used directly; the
    reduction below only observes the DDP average that Trainer will apply to
    gradients.  No project loss or gradient scaling is introduced here.
    """
    world_size = trainer.accelerator.num_processes
    if world_size != 2:
        raise RuntimeError(f"two-process loss check requires exactly two processes, got {world_size}")
    context = config["world"]["context_length"]
    rank = trainer.args.process_index
    local_batch = collate([dataset[rank]], context)
    local_count = _scored_token_count(local_batch).to(trainer.args.device)
    global_count = trainer.accelerator.reduce(local_count, reduction="sum")
    if int(global_count.item()) <= 0:
        raise AssertionError("two-process loss check received no supervised action tokens")
    trainer.model.eval()
    with torch.no_grad():
        # Trainer multiplies this local value by the number of ranks when
        # average_tokens_across_devices is enabled; DDP then averages the
        # gradients.  Record the resulting numerical difference rather than
        # asserting a GPU-kernel-specific tolerance.
        local_effective_loss = trainer.compute_loss(
            trainer.model_wrapped,
            _batch_on_device(local_batch, trainer.args.device),
            num_items_in_batch=global_count,
        )
        two_process_effective_loss = trainer.accelerator.reduce(local_effective_loss.detach(), reduction="sum") / world_size
        reference_batch = collate([dataset[index] for index in range(world_size)], context)
        reference_loss = trainer.accelerator.unwrap_model(trainer.model_wrapped)(
            **_batch_on_device(reference_batch, trainer.args.device)
        ).loss
    if reference_loss is None or not torch.isfinite(two_process_effective_loss) or not torch.isfinite(reference_loss):
        raise FloatingPointError("loss-equivalence diagnostic produced a non-finite loss")
    return {
        "world_size": world_size,
        "global_supervised_action_tokens": int(global_count.item()),
        "one_process_reference_loss": float(reference_loss.detach().float().cpu()),
        "two_process_effective_loss": float(two_process_effective_loss.detach().float().cpu()),
        "numerical_diagnostic": _tensor_difference_report(two_process_effective_loss, reference_loss),
        "acceptance_policy": "diagnostic_only_until_empirically_calibrated",
    }


def _stop_at_step_callback(step: int):
    """Build a narrow Trainer callback used only to split save and resume."""
    from transformers import TrainerCallback

    class StopAtStep(TrainerCallback):
        def on_step_end(self, _args, state, control, **_kwargs):
            if state.global_step == step:
                control.should_training_stop = True
            return control

    return StopAtStep()


def _preflight_settings(config: dict) -> tuple[int, int]:
    settings = config["preflight"]
    checkpoint_after_updates = settings["checkpoint_after_updates"]
    resume_updates = settings["resume_updates"]
    if checkpoint_after_updates < 2 or resume_updates < 1:
        raise ValueError("preflight needs at least two diagnostic updates and one resumed update")
    return checkpoint_after_updates, resume_updates


def _tensor_difference_report(expected: torch.Tensor, actual: torch.Tensor) -> dict:
    """Describe floating-point drift without making it a serialization gate."""
    if expected.shape != actual.shape:
        raise ValueError(f"cannot compare different shapes: {tuple(expected.shape)} != {tuple(actual.shape)}")
    if not (expected.is_floating_point() and actual.is_floating_point()):
        raise ValueError("numerical diagnostic requires floating-point tensors")
    difference = (expected.detach().float().cpu() - actual.detach().float().cpu()).abs()
    return {
        "shape": list(expected.shape),
        "expected_dtype": str(expected.dtype),
        "actual_dtype": str(actual.dtype),
        "max_abs_difference": float(difference.max()),
        "mean_abs_difference": float(difference.mean()),
        "exactly_equal": bool(torch.equal(expected.detach().cpu(), actual.detach().cpu())),
    }


def _state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    """Fingerprint state names, tensor metadata, and raw tensor bytes."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def exact_state_dict_report(expected_model, actual_model) -> dict:
    """Return an exact, framework-agnostic serialization contract report.

    Model serialization is proven by matching every state-dict key, shape,
    dtype, and tensor value exactly.  Floating-point output drift is reported
    separately because GPU kernel execution need not be bit-identical even
    when the saved state is byte-for-byte identical.
    """
    expected, actual = expected_model.state_dict(), actual_model.state_dict()
    expected_keys, actual_keys = set(expected), set(actual)
    missing, unexpected = sorted(expected_keys - actual_keys), sorted(actual_keys - expected_keys)
    shape_mismatches = []
    dtype_mismatches = []
    unequal = []
    max_parameter_abs_difference = 0.0
    for name in sorted(expected_keys & actual_keys):
        left, right = expected[name].detach().cpu(), actual[name].detach().cpu()
        if tuple(left.shape) != tuple(right.shape):
            shape_mismatches.append({"name": name, "expected": list(left.shape), "actual": list(right.shape)})
            continue
        if left.dtype != right.dtype:
            dtype_mismatches.append({"name": name, "expected": str(left.dtype), "actual": str(right.dtype)})
            continue
        if not torch.equal(left, right):
            unequal.append(name)
            if left.is_floating_point():
                max_parameter_abs_difference = max(max_parameter_abs_difference, float((left.float() - right.float()).abs().max()))
    return {
        "expected_key_count": len(expected),
        "actual_key_count": len(actual),
        "expected_state_sha256": _state_dict_sha256(expected),
        "actual_state_sha256": _state_dict_sha256(actual),
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "unequal_keys": unequal,
        "max_parameter_abs_difference_for_unequal_floats": max_parameter_abs_difference,
        "exact": not (missing or unexpected or shape_mismatches or dtype_mismatches or unequal),
    }


def assert_exact_state_dict_roundtrip(expected_model, actual_model) -> dict:
    """Hard-gate model serialization on exact state, with actionable evidence."""
    report = exact_state_dict_report(expected_model, actual_model)
    if not report["exact"]:
        raise AssertionError(f"save_pretrained reload changed model state: {json.dumps(report, sort_keys=True)}")
    return report


def _per_rank_numerical_diagnostic(trainer, local: dict) -> list[dict]:
    """Preserve a small logit-drift diagnostic from every distributed rank."""
    values = torch.tensor(
        [local["max_abs_difference"], local["mean_abs_difference"], float(local["exactly_equal"])],
        device=trainer.args.device,
        dtype=torch.float64,
    )
    gathered = trainer.accelerator.gather_for_metrics(values).detach().cpu().reshape(-1, 3)
    if len(gathered) != trainer.accelerator.num_processes:
        raise AssertionError(f"expected {trainer.accelerator.num_processes} rank diagnostics, got {len(gathered)}")
    return [
        {
            "rank": rank,
            "max_abs_difference": float(row[0]),
            "mean_abs_difference": float(row[1]),
            "exactly_equal": bool(row[2]),
        }
        for rank, row in enumerate(gathered)
    ]


def _artifact_reload_diagnostics(trainer, fixed_batch: dict[str, torch.Tensor], run_dir: Path, config: dict, plan: TrainingPlan) -> dict:
    """Hard-gate state, and record only the numerical output comparison."""
    from transformers import AutoModelForCausalLM

    source_model = trainer.accelerator.unwrap_model(trainer.model_wrapped)
    source_model.eval()
    artifact = _save_model_artifact(trainer, run_dir, config, plan)
    reloaded = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True).to(trainer.args.device).eval()
    state = assert_exact_state_dict_roundtrip(source_model, reloaded)
    inputs = _batch_on_device({name: value for name, value in fixed_batch.items() if name != "labels"}, trainer.args.device)
    with torch.no_grad():
        expected = source_model(**inputs).logits
        actual = reloaded(**inputs).logits
    logit_diagnostic = _tensor_difference_report(expected, actual)
    return {
        "artifact": str(artifact),
        "exact_state_dict": state,
        "logit_numerical_diagnostic": {
            **logit_diagnostic,
            "per_rank": _per_rank_numerical_diagnostic(trainer, logit_diagnostic),
            "acceptance_policy": "diagnostic_only_until_empirically_calibrated",
        },
    }


def _run_preflight(config: dict, run_dir: Path, plan: TrainingPlan) -> None:
    """Checkpoint once, resume once, and leave an inspectable rank-zero report."""
    from transformers.trainer_utils import get_last_checkpoint

    checkpoint_after, resume_updates = _preflight_settings(config)
    total_updates = checkpoint_after + resume_updates
    source = BinaryShard(run_dir / "datasets" / "train.bin")
    try:
        # The same real sequence is intentionally repeated so the short run is
        # a deterministic label/mask diagnostic, not a capability benchmark.
        fixed_dataset = SequenceDataset([source[0]] * (plan.world_size * plan.per_device_sequences))
    finally:
        source.close()
    trainer, shard = _trainer(
        config, run_dir, total_updates, plan, train_dataset=fixed_dataset,
        save_strategy="steps", save_steps=checkpoint_after, save_total_limit=1,
    )
    resumed = None
    resumed_shard = None
    try:
        fixed_batch = collate([shard[0]], config["world"]["context_length"])
        if not (fixed_batch["labels"] == -100).any() or not (fixed_batch["labels"] != -100).any():
            raise AssertionError("real batch must contain both context and supervised labels")
        initial_loss = _fixed_batch_loss(trainer.model, fixed_batch, trainer.args.device)
        trainer.add_callback(_stop_at_step_callback(checkpoint_after))
        trainer.train()
        if trainer.state.global_step != checkpoint_after:
            raise AssertionError(f"preflight stopped at {trainer.state.global_step}, expected {checkpoint_after}")
        _wait_for_everyone(trainer)
        checkpoint_loss = _fixed_batch_loss(trainer.model, fixed_batch, trainer.args.device)
        if not checkpoint_loss < initial_loss:
            raise AssertionError(
                f"fixed real-batch overfit did not reduce loss: {initial_loss:.6f} -> {checkpoint_loss:.6f}"
            )
        equivalence = _one_vs_two_process_loss_diagnostic(trainer, shard, config)
        checkpoint_name = f"checkpoint-{checkpoint_after}"
        checkpoint_path = run_dir / "checkpoints" / checkpoint_name
        found = get_last_checkpoint(str(run_dir / "checkpoints"))
        if found is None or Path(found).resolve() != checkpoint_path.resolve():
            raise AssertionError(f"expected exactly {checkpoint_name}, found {found}")
        if sorted(path.name for path in (run_dir / "checkpoints").glob("checkpoint-*")) != [checkpoint_name]:
            raise AssertionError("preflight must create exactly one Trainer checkpoint before resume")
        _wait_for_everyone(trainer)

        # Reconstruct a standard Trainer and restore its standard checkpoint;
        # no project checkpoint loader or optimizer state is involved.
        resumed, resumed_shard = _trainer(
            config, run_dir, total_updates, plan, train_dataset=fixed_dataset,
            save_strategy="no", save_steps=checkpoint_after,
        )
        resumed.train(resume_from_checkpoint=found)
        if resumed.state.global_step != total_updates:
            raise AssertionError(f"resume ended at {resumed.state.global_step}, expected {total_updates}")
        _wait_for_everyone(resumed)
        resumed_loss = _fixed_batch_loss(resumed.model, fixed_batch, resumed.args.device)
        if not resumed_loss < initial_loss:
            raise AssertionError(
                f"fixed real-batch loss was not lower after checkpoint/resume: {initial_loss:.6f} -> {resumed_loss:.6f}"
            )
        artifact_diagnostics = _artifact_reload_diagnostics(resumed, fixed_batch, run_dir, config, plan)
        finished_ranks = resumed.accelerator.gather_for_metrics(
            torch.tensor([resumed.args.process_index], device=resumed.args.device)
        ).detach().cpu().tolist()
        if sorted(finished_ranks) != list(range(resumed.accelerator.num_processes)):
            raise AssertionError(f"not every rank reached preflight completion: {finished_ranks}")
        report = {
            "checkpoint": str(checkpoint_path),
            "fixed_real_batch_loss": {
                "initial": initial_loss,
                "after_checkpoint": checkpoint_loss,
                "after_resume": resumed_loss,
            },
            "behavioral_plumbing_diagnostic": {
                "fixed_real_batch_loss_progress": {
                    "initial": initial_loss,
                    "after_checkpoint": checkpoint_loss,
                    "after_resume": resumed_loss,
                },
                "acceptance_policy": "finite_loss_and_progress_required; not_serialization_proof",
            },
            "loss_equivalence": equivalence,
            "model_artifact": artifact_diagnostics["artifact"],
            "serialization": artifact_diagnostics,
        }
        report.update({
            "checkpoint_after_updates": checkpoint_after,
            "resume_updates": resume_updates,
            "resumed_global_step": resumed.state.global_step,
            "ranks_finished": finished_ranks,
            "token_accounting": plan.report(),
        })
        _write_rank_zero_json(resumed, run_dir / "preflight_report.json", report)
    finally:
        shard.close()
        if resumed_shard is not None:
            resumed_shard.close()


def run(resolved_config: Path, run_dir: Path, preflight_only: bool = False, resume: bool = False) -> None:
    config = json.loads(resolved_config.read_text())
    plan = training_plan(config)
    if preflight_only:
        _run_preflight(config, run_dir, plan)
        return
    trainer, dataset = _trainer(config, run_dir, plan.max_steps, plan)
    try:
        _write_rank_zero_json(trainer, run_dir / "training_plan.json", plan.report())
        from transformers.trainer_utils import get_last_checkpoint
        checkpoint = get_last_checkpoint(str(run_dir / "checkpoints")) if resume else None
        trainer.train(resume_from_checkpoint=checkpoint)
        _save_model_artifact(trainer, run_dir, config, plan)
    finally:
        dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    run(args.resolved_config, args.run_dir, args.preflight_only, args.resume)
