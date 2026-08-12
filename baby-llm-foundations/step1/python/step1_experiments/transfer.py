"""Matched Rendering B interface-calibration curves on standard HF artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import tomllib
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import torch

from .artifacts import atomic_json, sha256_file
from .data import BinaryShard, collate, generate_rust_shard
from .evaluate import EVALUATION_METRIC_NAMES, _aggregate_rows, _execute_batched
from .standard_stack import EXPECTED_PARAMETER_COUNT, assert_model_contract, create_model, load_tokenizer
from .train import _state_dict_sha256, assert_exact_state_dict_roundtrip


TRANSFER_CONTRACT = "step1_rendering_b_transfer_v1"
ARMS = ("a_trained", "init")


@dataclass(frozen=True)
class TransferPlan:
    root_seed: int
    world_size: int
    per_device_sequences: int
    context_length: int
    gradient_accumulation_steps: int
    nominal_global_input_tokens_per_update: int
    budgets_updates: tuple[int, ...]
    budgets_nominal_global_input_tokens: tuple[int, ...]
    calibration_seed: int
    calibration_episodes: int
    evaluation_seed: int
    evaluation_episodes: int

    def report(self) -> dict:
        return {
            **asdict(self),
            "budgets_updates": list(self.budgets_updates),
            "budgets_nominal_global_input_tokens": list(self.budgets_nominal_global_input_tokens),
            "token_unit": "nominal global input tokens (world_size * per_device_sequences * context_length * accumulation)",
            "calibration_cost_policy": "B-only cumulative cost; prior A training cost reported separately",
            "full_b_reference_included": True,
        }


def load_transfer_config(path: Path) -> dict:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    config["_meta"] = {"hash": hashlib.sha256(canonical).hexdigest(), "path": str(path.resolve())}
    return config


def transfer_plan(config: dict, world_size: int = 1) -> TransferPlan:
    calibration, world = config["calibration"], config["world"]
    budgets = tuple(calibration["budgets_updates"])
    if not budgets or budgets[0] != 0 or tuple(sorted(set(budgets))) != budgets:
        raise ValueError("calibration budgets must be unique, increasing, and begin at zero")
    if calibration.get("include_full_b_reference") is not True:
        raise ValueError("the final cumulative point must be declared as the full-B reference")
    microstep = world_size * calibration["per_device_sequences"] * world["context_length"]
    per_update = calibration["global_input_tokens_per_update"]
    if per_update <= 0 or per_update % microstep:
        raise ValueError("global input tokens per update must be an exact multiple of the global microstep")
    accumulation = per_update // microstep
    root_seed = config["run"]["root_seed"]
    return TransferPlan(
        root_seed=root_seed,
        world_size=world_size,
        per_device_sequences=calibration["per_device_sequences"],
        context_length=world["context_length"],
        gradient_accumulation_steps=accumulation,
        nominal_global_input_tokens_per_update=per_update,
        budgets_updates=budgets,
        budgets_nominal_global_input_tokens=tuple(step * per_update for step in budgets),
        calibration_seed=root_seed + calibration["calibration_seed_offset"],
        calibration_episodes=calibration["calibration_episodes"],
        evaluation_seed=root_seed + config["evaluation"]["seed_offset"],
        evaluation_episodes=config["evaluation"]["episodes"],
    )


def locate_dense_source(roots: list[Path], expected: dict) -> Path:
    """Locate an extracted dense bundle by its exact report identity."""
    matches: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for report_path in root.rglob("production/training_report.json"):
            try:
                report = json.loads(report_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if (
                report.get("contract") == "step1_dense_training_v1"
                and report.get("source_git_sha") == expected["git_sha"]
                and report.get("config_hash") == expected["config_hash"]
            ):
                matches.append(report_path.parents[1].resolve())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"expected exactly one matching extracted dense source, found {unique}")
    return unique[0]


def _verify_source(source_run: Path, config: dict) -> tuple[dict, Path, Path]:
    expected = config["source"]
    report = json.loads((source_run / "production" / "training_report.json").read_text())
    resolved = json.loads((source_run / "resolved_config.json").read_text())
    experiment = json.loads((source_run / "production" / "model" / "experiment.json").read_text())
    checks = {
        "dense contract": (report.get("contract"), "step1_dense_training_v1"),
        "source SHA": (report.get("source_git_sha"), expected["git_sha"]),
        "source config hash": (report.get("config_hash"), expected["config_hash"]),
        "resolved root seed": (resolved.get("run", {}).get("root_seed"), config["run"]["root_seed"]),
        "resolved rendering": (resolved.get("world", {}).get("rendering"), "a"),
        "model config SHA": (experiment.get("model_config_sha256"), expected["model_config_sha256"]),
        "parameter count": (experiment.get("parameter_count"), EXPECTED_PARAMETER_COUNT),
        "exact source serialization": (report.get("serialization", {}).get("exact_state_dict", {}).get("exact"), True),
        "reported source state SHA": (
            report.get("serialization", {}).get("exact_state_dict", {}).get("actual_state_sha256"),
            expected["model_state_sha256"],
        ),
    }
    mismatches = {name: {"actual": actual, "expected": wanted} for name, (actual, wanted) in checks.items() if actual != wanted}
    if mismatches:
        raise AssertionError(f"dense source identity mismatch: {json.dumps(mismatches, sort_keys=True)}")
    model_path = source_run / "production" / "model"
    model_config_hash = sha256_file(model_path / "config.json")
    if model_config_hash != expected["model_config_sha256"]:
        raise AssertionError("source model config bytes do not match the dense report")
    calibration_prefix = source_run / "datasets" / "transfer.bin"
    replay = source_run / "datasets" / "transfer.replay"
    if sha256_file(calibration_prefix) != expected["calibration_data_sha256"]:
        raise AssertionError("Rendering B calibration shard hash mismatch")
    if sha256_file(replay) != expected["calibration_replay_sha256"]:
        raise AssertionError("Rendering B calibration replay hash mismatch")
    return report, model_path, calibration_prefix


def _prepare_calibration_dataset(output_dir: Path, source_prefix: Path, config: dict, plan: TransferPlan) -> dict:
    """Generate the full B shard once and verify the old 1,024-episode prefix."""
    dataset_dir = output_dir / "datasets"
    prefix_binary, prefix_manifest, prefix_replay = generate_rust_shard(
        config["world"], plan.calibration_seed, 1024, plan.context_length,
        "b", dataset_dir, "calibration-prefix-1024",
    )
    expected_prefix_hash = config["source"]["calibration_data_sha256"]
    if sha256_file(prefix_binary) != expected_prefix_hash or sha256_file(source_prefix) != expected_prefix_hash:
        raise AssertionError("current generator does not reproduce the source bundle's 1,024-episode B prefix")
    binary, manifest, replay = generate_rust_shard(
        config["world"], plan.calibration_seed, plan.calibration_episodes,
        plan.context_length, "b", dataset_dir, "calibration",
    )
    return {
        "path": str(binary.resolve()),
        "data_sha256": sha256_file(binary),
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": sha256_file(manifest),
        "replay_path": str(replay.resolve()),
        "replay_sha256": sha256_file(replay),
        "prefix_equivalence": {
            "episodes": 1024,
            "expected_data_sha256": expected_prefix_hash,
            "actual_data_sha256": sha256_file(prefix_binary),
            "manifest_sha256": sha256_file(prefix_manifest),
            "replay_sha256": sha256_file(prefix_replay),
            "exact": True,
        },
    }


class _MilestoneSaves:
    """Create standard Trainer checkpoints only at predeclared curve points."""
    def __new__(cls, milestones: tuple[int, ...]):
        from transformers import TrainerCallback

        class Callback(TrainerCallback):
            def on_step_end(self, _args, state, control, **_kwargs):
                # DefaultFlowCallback first requests the uniform `save_steps=1`
                # cadence. This later callback retains only declared milestones;
                # Trainer still owns every checkpoint write and its contents.
                control.should_save = state.global_step in milestones
                return control

        return Callback()


def _trainer(model, shard: BinaryShard, workspace: Path, config: dict, plan: TransferPlan):
    from transformers import Trainer, TrainingArguments

    calibration = config["calibration"]
    args = TrainingArguments(
        output_dir=str(workspace / "checkpoints"),
        per_device_train_batch_size=plan.per_device_sequences,
        gradient_accumulation_steps=plan.gradient_accumulation_steps,
        learning_rate=calibration["learning_rate"],
        weight_decay=calibration["weight_decay"],
        lr_scheduler_type="cosine",
        warmup_ratio=calibration["warmup_fraction"],
        max_steps=max(plan.budgets_updates),
        fp16=torch.cuda.is_available(),
        bf16=False,
        save_strategy="steps",
        save_steps=1,
        save_total_limit=None,
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        seed=plan.root_seed,
        data_seed=plan.root_seed,
        average_tokens_across_devices=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=shard,
        data_collator=partial(collate, context=plan.context_length),
        processing_class=load_tokenizer(),
        callbacks=[_MilestoneSaves(tuple(step for step in plan.budgets_updates if step))],
    )
    return trainer


def _checkpoint_state_roundtrip(checkpoint: Path) -> tuple[object, dict]:
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM

    expected_state = load_file(checkpoint / "model.safetensors", device="cpu")
    model = AutoModelForCausalLM.from_pretrained(checkpoint, local_files_only=True).eval()
    actual_state = model.state_dict()
    expected_keys, actual_keys = set(expected_state), set(actual_state)
    unequal = []
    for name in sorted(expected_keys & actual_keys):
        if not torch.equal(expected_state[name], actual_state[name].detach().cpu()):
            unequal.append(name)
    report = {
        "expected_state_sha256": _state_dict_sha256(expected_state),
        "actual_state_sha256": _state_dict_sha256(actual_state),
        "missing_keys": sorted(expected_keys - actual_keys),
        "unexpected_keys": sorted(actual_keys - expected_keys),
        "unequal_keys": unequal,
    }
    report["exact"] = not (report["missing_keys"] or report["unexpected_keys"] or unequal) and report["expected_state_sha256"] == report["actual_state_sha256"]
    if not report["exact"]:
        raise AssertionError(f"checkpoint state roundtrip failed: {json.dumps(report, sort_keys=True)}")
    return model, report


def _evaluate_b(model, config: dict, plan: TransferPlan, device: torch.device) -> dict:
    model.to(device).eval()
    variants = {"irreversible": config["world"]}
    if config["evaluation"].get("include_reversible_control"):
        variants["reversible_control"] = {**config["world"], "variant": "reversible"}
    result = {}
    for label, params in variants.items():
        rows = _execute_batched(
            model, params, plan.evaluation_seed, plan.evaluation_episodes,
            "b", device,
        )
        metrics = _aggregate_rows(rows)
        for name, value in metrics.items():
            if value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise FloatingPointError(f"non-finite transfer metric {label}.{name}: {value!r}")
        result[label] = metrics
    return result


def _initial_model_artifact(workspace: Path, config: dict):
    from transformers import AutoModelForCausalLM, set_seed

    set_seed(config["run"]["root_seed"])
    model = create_model()
    assert_model_contract(model)
    artifact = workspace / "budget-0-model"
    model.save_pretrained(artifact, safe_serialization=True)
    load_tokenizer().save_pretrained(artifact)
    reloaded = AutoModelForCausalLM.from_pretrained(artifact, local_files_only=True)
    serialization = assert_exact_state_dict_roundtrip(model, reloaded)
    return reloaded, artifact, serialization


def assert_transfer_report_contract(report: dict, config: dict, plan: TransferPlan) -> None:
    expected_source = config["source"]
    if report.get("contract") != TRANSFER_CONTRACT or report.get("plan") != plan.report():
        raise AssertionError("transfer report contract or exact plan mismatch")
    source = report.get("source", {})
    for field, expected in (
        ("git_sha", expected_source["git_sha"]),
        ("config_hash", expected_source["config_hash"]),
        ("model_state_sha256", expected_source["model_state_sha256"]),
    ):
        if source.get(field) != expected:
            raise AssertionError(f"transfer source {field} mismatch")
    datasets = report.get("datasets", {})
    calibration = datasets.get("calibration", {})
    if calibration.get("prefix_equivalence", {}).get("expected_data_sha256") != expected_source["calibration_data_sha256"] or calibration.get("prefix_equivalence", {}).get("exact") is not True:
        raise AssertionError("transfer calibration prefix identity mismatch")
    if not calibration.get("data_sha256") or calibration.get("episodes") != plan.calibration_episodes:
        raise AssertionError("full transfer calibration dataset identity is incomplete")
    evaluation = datasets.get("held_out_evaluation_worlds", {})
    if evaluation.get("seed") != plan.evaluation_seed or evaluation.get("episodes") != plan.evaluation_episodes or evaluation.get("rendering") != "b":
        raise AssertionError("held-out Rendering B evaluation contract mismatch")
    if set(report.get("arms", {})) != set(ARMS):
        raise AssertionError("transfer report arm mismatch")
    required_variants = {"irreversible", "reversible_control"} if config["evaluation"].get("include_reversible_control") else {"irreversible"}
    for arm in ARMS:
        points = report["arms"][arm]["curve"]
        if report["arms"][arm].get("calibration_data_sha256") != calibration["data_sha256"]:
            raise AssertionError(f"{arm} did not use the matched calibration shard")
        if [point["budget_updates"] for point in points] != list(plan.budgets_updates):
            raise AssertionError(f"{arm} curve budget mismatch")
        for point in points:
            serialization = point["serialization"]
            if serialization.get("exact") is not True or serialization.get("expected_state_sha256") != serialization.get("actual_state_sha256"):
                raise AssertionError(f"{arm} budget {point['budget_updates']} artifact is not exact")
            if set(point.get("metrics", {})) != required_variants:
                raise AssertionError(f"{arm} budget {point['budget_updates']} evaluation variant mismatch")
            for variant, metrics in point["metrics"].items():
                if set(metrics) != set(EVALUATION_METRIC_NAMES):
                    raise AssertionError(f"{arm} budget {point['budget_updates']} {variant} metric fields mismatch")
                for metric, value in metrics.items():
                    if metric == "mean_success_excess_cost" and value is None:
                        continue
                    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                        raise FloatingPointError(f"invalid transfer metric {arm}.{point['budget_updates']}.{variant}.{metric}: {value!r}")
    paired = report.get("paired_diagnostics", [])
    if [point.get("budget_updates") for point in paired] != list(plan.budgets_updates):
        raise AssertionError("paired diagnostic update budgets are not aligned")
    if [point.get("b_calibration_nominal_global_input_tokens") for point in paired] != list(plan.budgets_nominal_global_input_tokens):
        raise AssertionError("paired diagnostic token budgets are not aligned")
    for point in paired:
        if point.get("definition") != "a_trained_minus_init; paired diagnostic only":
            raise AssertionError("paired diagnostic definition mismatch")
        if set(point.get("variants", {})) != required_variants:
            raise AssertionError("paired diagnostic variant mismatch")
        for variant, metrics in point["variants"].items():
            if set(metrics) != set(EVALUATION_METRIC_NAMES):
                raise AssertionError(f"paired diagnostic {variant} metric fields mismatch")
            for metric, value in metrics.items():
                if metric == "mean_success_excess_cost" and value is None:
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise FloatingPointError(f"invalid paired diagnostic {variant}.{metric}: {value!r}")


def run(config_path: Path, source_run: Path, output_dir: Path) -> Path:
    from accelerate import PartialState
    from transformers import AutoModelForCausalLM, set_seed

    config = load_transfer_config(config_path)
    state = PartialState()
    plan = transfer_plan(config, world_size=state.num_processes)
    if state.num_processes != 2:
        raise AssertionError(f"Rendering B transfer requires exactly two Trainer/Accelerate ranks, got {state.num_processes}")
    source_report, source_model_path, source_prefix = _verify_source(source_run, config)
    if state.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        calibration_identity = _prepare_calibration_dataset(output_dir, source_prefix, config, plan)
        atomic_json(output_dir / "calibration_dataset.json", calibration_identity)
    state.wait_for_everyone()
    calibration_identity = json.loads((output_dir / "calibration_dataset.json").read_text())
    calibration_path = Path(calibration_identity["path"])
    if sha256_file(calibration_path) != calibration_identity["data_sha256"]:
        raise AssertionError("prepared calibration dataset changed before training")
    device = state.device
    shard = BinaryShard(calibration_path)
    try:
        arms: dict[str, dict] = {}
        for arm in ARMS:
            workspace = output_dir / arm
            if arm == "a_trained":
                model = AutoModelForCausalLM.from_pretrained(source_model_path, local_files_only=True)
                if _state_dict_sha256(model.state_dict()) != config["source"]["model_state_sha256"]:
                    raise AssertionError("loaded A-trained source state hash mismatch")
                zero_artifact = source_model_path
                zero_serialization = source_report["serialization"]["exact_state_dict"]
            else:
                zero_artifact = workspace / "budget-0-model"
                zero_report_path = workspace / "budget-0-serialization.json"
                if state.is_main_process:
                    model, zero_artifact, zero_serialization = _initial_model_artifact(workspace, config)
                    atomic_json(zero_report_path, zero_serialization)
                state.wait_for_everyone()
                if not state.is_main_process:
                    model = AutoModelForCausalLM.from_pretrained(zero_artifact, local_files_only=True)
                    zero_serialization = json.loads(zero_report_path.read_text())
            curve = []
            if state.is_main_process:
                curve.append({
                    "budget_updates": 0,
                    "b_calibration_nominal_global_input_tokens": 0,
                    "artifact": str(zero_artifact.resolve()),
                    "serialization": zero_serialization,
                    "metrics": None,
                    "point_label": "zero_shot_interface_diagnostic_not_transfer",
                })
            state.wait_for_everyone()
            model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            set_seed(plan.root_seed)
            trainer = _trainer(model, shard, workspace, config, plan)
            trainer.train()
            if trainer.state.global_step != max(plan.budgets_updates):
                raise AssertionError("calibration Trainer did not reach the final declared budget")
            expected_checkpoints = [f"checkpoint-{step}" for step in plan.budgets_updates if step]
            found_checkpoints = sorted(
                (path.name for path in (workspace / "checkpoints").glob("checkpoint-*")),
                key=lambda name: int(name.split("-")[-1]),
            )
            if found_checkpoints != expected_checkpoints:
                raise AssertionError(f"{arm} standard checkpoint milestones mismatch: {found_checkpoints}")
            finished_ranks = trainer.accelerator.gather_for_metrics(
                torch.tensor([trainer.args.process_index], device=trainer.args.device)
            ).detach().cpu().tolist()
            if sorted(finished_ranks) != list(range(plan.world_size)):
                raise AssertionError(f"not every rank completed {arm} calibration: {finished_ranks}")
            trainer.accelerator.wait_for_everyone()
            model.to("cpu")
            del trainer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if state.is_main_process:
                for step in plan.budgets_updates[1:]:
                    checkpoint = workspace / "checkpoints" / f"checkpoint-{step}"
                    point_model, serialization = _checkpoint_state_roundtrip(checkpoint)
                    curve.append({
                        "budget_updates": step,
                        "b_calibration_nominal_global_input_tokens": step * plan.nominal_global_input_tokens_per_update,
                        "artifact": str(checkpoint.resolve()),
                        "serialization": serialization,
                        "metrics": None,
                        "point_label": "full_b_reference" if step == max(plan.budgets_updates) else "interface_calibrated",
                    })
                    point_model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                arms[arm] = {
                    "initialization": "dense_A_checkpoint" if arm == "a_trained" else "random_from_original_seed0_config",
                    "root_seed": plan.root_seed,
                    "ranks_finished": finished_ranks,
                    "calibration_data_sha256": calibration_identity["data_sha256"],
                    "total_executed_b_calibration_updates": max(plan.budgets_updates),
                    "curve": curve,
                }
                atomic_json(output_dir / f"{arm}_curve.json", arms[arm])
            state.wait_for_everyone()
            if not state.is_main_process:
                arms[arm] = json.loads((output_dir / f"{arm}_curve.json").read_text())
        state.wait_for_everyone()
    finally:
        shard.close()
    path = output_dir / "rendering_b_transfer_report.json"
    if not state.is_main_process:
        # Distributed training is complete. Rank zero exclusively performs the
        # long autoregressive evaluations and report write with no pending
        # collective/barrier that could time out.
        return path
    for arm in ARMS:
        for point in arms[arm]["curve"]:
            point_model, serialization = _checkpoint_state_roundtrip(Path(point["artifact"]))
            recorded = point["serialization"]
            if (
                recorded.get("exact") is not True
                or recorded.get("expected_state_sha256") != serialization["expected_state_sha256"]
                or recorded.get("actual_state_sha256") != serialization["actual_state_sha256"]
            ):
                raise AssertionError(f"{arm} budget {point['budget_updates']} artifact identity changed before evaluation")
            point["metrics"] = _evaluate_b(point_model, config, plan, device)
            point_model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    paired_deltas = []
    for a_point, init_point in zip(arms["a_trained"]["curve"], arms["init"]["curve"]):
        if a_point["budget_updates"] != init_point["budget_updates"]:
            raise AssertionError("paired transfer curves are not aligned by B calibration budget")
        variants = {}
        for variant in a_point["metrics"]:
            variants[variant] = {
                metric: (
                    None if a_point["metrics"][variant][metric] is None or init_point["metrics"][variant][metric] is None
                    else a_point["metrics"][variant][metric] - init_point["metrics"][variant][metric]
                )
                for metric in a_point["metrics"][variant]
            }
        paired_deltas.append({
            "budget_updates": a_point["budget_updates"],
            "b_calibration_nominal_global_input_tokens": a_point["b_calibration_nominal_global_input_tokens"],
            "definition": "a_trained_minus_init; paired diagnostic only",
            "variants": variants,
        })
    report = {
        "contract": TRANSFER_CONTRACT,
        "experiment_config_sha256": config["_meta"]["hash"],
        "source": {
            "run": str(source_run.resolve()),
            "git_sha": config["source"]["git_sha"],
            "config_hash": config["source"]["config_hash"],
            "model_state_sha256": config["source"]["model_state_sha256"],
            "prior_a_training_cost": source_report["token_accounting"],
        },
        "plan": plan.report(),
        "datasets": {
            "calibration": {**calibration_identity, "rendering": "b", "seed": plan.calibration_seed, "episodes": plan.calibration_episodes},
            "held_out_evaluation_worlds": {
                "rendering": "b",
                "seed": plan.evaluation_seed,
                "episodes": plan.evaluation_episodes,
                "matched_across_arms_and_budgets": True,
                "variants": ["irreversible", "reversible_control"] if config["evaluation"].get("include_reversible_control") else ["irreversible"],
            },
        },
        "arms": arms,
        "paired_diagnostics": paired_deltas,
        "scientific_acceptance_policy": "diagnostic learning curves; no capability threshold",
    }
    assert_transfer_report_contract(report, config, plan)
    atomic_json(path, report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config, args.source_run, args.output_dir))


if __name__ == "__main__":
    main()
