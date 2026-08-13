"""Independent, fully annealed Rendering-B transfer endpoints.

Each nonzero budget starts from the arm's original state and owns a complete
warmup-plus-cosine schedule whose learning rate reaches zero at that budget.
This deliberately does not reuse intermediate checkpoints from a longer run:
those checkpoints answer a training-dynamics question, not the terminal-budget
sample-efficiency question measured here.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .artifacts import atomic_json, sha256_file
from .data import BinaryShard
from .train import _state_dict_sha256
from .transfer import (
    ARMS,
    TransferPlan,
    _checkpoint_state_roundtrip,
    _evaluate_b,
    _initial_model_artifact,
    _prepare_calibration_dataset,
    _trainer,
    _verify_source,
    load_transfer_config,
    transfer_plan,
)


TERMINAL_TRANSFER_CONTRACT = "step1_rendering_b_terminal_transfer_v1"


def _validate_terminal_plan(plan: TransferPlan) -> None:
    if plan.budgets_updates[0] != 0 or len(plan.budgets_updates) < 2:
        raise ValueError("terminal transfer requires budget zero and at least one trained endpoint")


def _metric_deltas(a_metrics: dict, init_metrics: dict) -> dict:
    result = {}
    for variant, metrics in a_metrics.items():
        result[variant] = {
            metric: (
                None if value is None or init_metrics[variant][metric] is None
                else value - init_metrics[variant][metric]
            )
            for metric, value in metrics.items()
        }
    return result


def assert_terminal_transfer_report_contract(report: dict, config: dict, plan: TransferPlan) -> None:
    if report.get("contract") != TERMINAL_TRANSFER_CONTRACT:
        raise AssertionError("terminal transfer contract mismatch")
    if report.get("experiment_config_sha256") != config["_meta"]["hash"]:
        raise AssertionError("terminal transfer configuration hash mismatch")
    if report.get("plan") != plan.report():
        raise AssertionError("terminal transfer plan mismatch")
    source = report.get("source", {})
    for field in ("git_sha", "config_hash", "model_state_sha256"):
        if source.get(field) != config["source"][field]:
            raise AssertionError(f"terminal source {field} mismatch")
    calibration = report.get("datasets", {}).get("calibration", {})
    if calibration.get("data_sha256") != report.get("arms", {}).get("a_trained", {}).get("calibration_data_sha256"):
        raise AssertionError("terminal A-trained calibration identity mismatch")
    if calibration.get("data_sha256") != report.get("arms", {}).get("init", {}).get("calibration_data_sha256"):
        raise AssertionError("terminal init calibration identity mismatch")
    if calibration.get("prefix_equivalence", {}).get("exact") is not True:
        raise AssertionError("terminal calibration prefix identity mismatch")
    heldout = report.get("datasets", {}).get("held_out_evaluation_worlds", {})
    if heldout.get("seed") != plan.evaluation_seed or heldout.get("episodes") != plan.evaluation_episodes:
        raise AssertionError("terminal held-out evaluation identity mismatch")
    if report.get("schedule_policy") != {
        "type": "independent_budget_terminal_cosine",
        "restart_from_arm_initialization_per_budget": True,
        "warmup_fraction": config["calibration"]["warmup_fraction"],
        "terminal_learning_rate_upper_bound": 1e-9,
    }:
        raise AssertionError("terminal schedule policy mismatch")
    required_variants = {"irreversible", "reversible_control"} if config["evaluation"].get("include_reversible_control") else {"irreversible"}
    budgets = list(plan.budgets_updates)
    tokens = list(plan.budgets_nominal_global_input_tokens)
    if set(report.get("arms", {})) != set(ARMS):
        raise AssertionError("terminal transfer arms mismatch")
    for arm in ARMS:
        points = report["arms"][arm].get("endpoints", [])
        if [point.get("budget_updates") for point in points] != budgets:
            raise AssertionError(f"{arm} terminal budget mismatch")
        if [point.get("b_calibration_nominal_global_input_tokens") for point in points] != tokens:
            raise AssertionError(f"{arm} terminal token budget mismatch")
        for point in points:
            if set(point.get("metrics", {})) != required_variants:
                raise AssertionError(f"{arm} terminal evaluation variant mismatch")
            serialization = point.get("serialization", {})
            if serialization.get("exact") is not True or serialization.get("expected_state_sha256") != serialization.get("actual_state_sha256"):
                raise AssertionError(f"{arm} terminal artifact is not exact")
            if point["budget_updates"]:
                schedule = point.get("schedule", {})
                if schedule.get("max_steps") != point["budget_updates"] or schedule.get("lr_scheduler_type") != "cosine":
                    raise AssertionError(f"{arm} endpoint did not own its terminal cosine schedule")
                if point.get("ranks_finished") != list(range(plan.world_size)):
                    raise AssertionError(f"{arm} endpoint did not finish on every rank")
                terminal_lr = schedule.get("terminal_learning_rate")
                if not isinstance(terminal_lr, (int, float)) or terminal_lr > 1e-9:
                    raise AssertionError(f"{arm} endpoint learning rate was not annealed to numerical zero")
            for metrics in point["metrics"].values():
                if any(value is not None and (not isinstance(value, (int, float)) or not math.isfinite(value)) for value in metrics.values()):
                    raise FloatingPointError(f"non-finite {arm} terminal metric")
    paired = report.get("paired_diagnostics", [])
    if [point.get("budget_updates") for point in paired] != budgets:
        raise AssertionError("terminal paired budgets are not aligned")


def run(config_path: Path, source_run: Path, output_dir: Path) -> Path:
    from accelerate import PartialState
    from transformers import AutoModelForCausalLM, set_seed

    config = load_transfer_config(config_path)
    state = PartialState()
    plan = transfer_plan(config, world_size=state.num_processes)
    _validate_terminal_plan(plan)
    if state.num_processes != 2:
        raise AssertionError(f"terminal Rendering-B transfer requires exactly two ranks, got {state.num_processes}")
    source_report, source_model_path, source_prefix = _verify_source(source_run, config)

    if state.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        calibration_identity = _prepare_calibration_dataset(output_dir, source_prefix, config, plan)
        atomic_json(output_dir / "calibration_dataset.json", calibration_identity)
    state.wait_for_everyone()
    calibration_identity = json.loads((output_dir / "calibration_dataset.json").read_text())
    calibration_path = Path(calibration_identity["path"])
    if sha256_file(calibration_path) != calibration_identity["data_sha256"]:
        raise AssertionError("terminal calibration dataset changed before training")

    # Materialize the random arm's exact budget-zero state once. Every budget
    # reloads this artifact, just as every A-trained budget reloads its source.
    init_workspace = output_dir / "init"
    init_artifact = init_workspace / "budget-0-model"
    init_serialization_path = init_workspace / "budget-0-serialization.json"
    if state.is_main_process:
        _, init_artifact, init_serialization = _initial_model_artifact(init_workspace, config)
        atomic_json(init_serialization_path, init_serialization)
    state.wait_for_everyone()
    init_serialization = json.loads(init_serialization_path.read_text())

    shard = BinaryShard(calibration_path)
    device = state.device
    try:
        arms: dict[str, dict] = {}
        for arm in ARMS:
            zero_artifact = source_model_path if arm == "a_trained" else init_artifact
            zero_serialization = source_report["serialization"]["exact_state_dict"] if arm == "a_trained" else init_serialization
            endpoints = []
            if state.is_main_process:
                endpoints.append({
                    "budget_updates": 0,
                    "b_calibration_nominal_global_input_tokens": 0,
                    "artifact": str(zero_artifact.resolve()),
                    "serialization": zero_serialization,
                    "schedule": None,
                    "metrics": None,
                    "point_label": "zero_shot_interface_diagnostic_not_transfer",
                })
            for budget in plan.budgets_updates[1:]:
                workspace = output_dir / arm / f"budget-{budget}"
                set_seed(plan.root_seed)
                model = AutoModelForCausalLM.from_pretrained(zero_artifact, local_files_only=True)
                if arm == "a_trained" and _state_dict_sha256(model.state_dict()) != config["source"]["model_state_sha256"]:
                    raise AssertionError("terminal run loaded the wrong A-trained source")
                trainer = _trainer(
                    model, shard, workspace, config, plan,
                    max_steps=budget, save_milestones=(budget,),
                )
                trainer.train()
                if trainer.state.global_step != budget:
                    raise AssertionError(f"{arm} terminal run stopped at {trainer.state.global_step}, expected {budget}")
                terminal_lrs = trainer.lr_scheduler.get_last_lr()
                if any(abs(float(lr)) > 1e-9 for lr in terminal_lrs):
                    raise AssertionError(f"{arm} budget {budget} did not anneal to zero: {terminal_lrs}")
                finished_ranks = trainer.accelerator.gather_for_metrics(
                    torch.tensor([trainer.args.process_index], device=trainer.args.device)
                ).detach().cpu().tolist()
                if sorted(finished_ranks) != list(range(plan.world_size)):
                    raise AssertionError(f"not every rank completed {arm} budget {budget}: {finished_ranks}")
                trainer.accelerator.wait_for_everyone()
                model.to("cpu")
                del trainer, model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if state.is_main_process:
                    checkpoint = workspace / "checkpoints" / f"checkpoint-{budget}"
                    point_model, serialization = _checkpoint_state_roundtrip(checkpoint)
                    endpoints.append({
                        "budget_updates": budget,
                        "b_calibration_nominal_global_input_tokens": budget * plan.nominal_global_input_tokens_per_update,
                        "artifact": str(checkpoint.resolve()),
                        "serialization": serialization,
                        "schedule": {
                            "max_steps": budget,
                            "lr_scheduler_type": "cosine",
                            "peak_learning_rate": config["calibration"]["learning_rate"],
                            "warmup_fraction": config["calibration"]["warmup_fraction"],
                            "terminal_learning_rate": max(abs(float(lr)) for lr in terminal_lrs),
                        },
                        "ranks_finished": sorted(finished_ranks),
                        "metrics": None,
                        "point_label": "independently_annealed_terminal_endpoint",
                    })
                    point_model.to("cpu")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            if state.is_main_process:
                arms[arm] = {
                    "initialization": "dense_A_checkpoint" if arm == "a_trained" else "random_from_root_seed_config",
                    "root_seed": plan.root_seed,
                    "calibration_data_sha256": calibration_identity["data_sha256"],
                    "endpoints": endpoints,
                }
                atomic_json(output_dir / f"{arm}_terminal_endpoints.json", arms[arm])
            state.wait_for_everyone()
            if not state.is_main_process:
                arms[arm] = json.loads((output_dir / f"{arm}_terminal_endpoints.json").read_text())
        state.wait_for_everyone()
    finally:
        shard.close()

    report_path = output_dir / "rendering_b_terminal_transfer_report.json"
    if not state.is_main_process:
        return report_path

    for arm in ARMS:
        for point in arms[arm]["endpoints"]:
            point_model, serialization = _checkpoint_state_roundtrip(Path(point["artifact"]))
            recorded = point["serialization"]
            if recorded.get("actual_state_sha256") != serialization["actual_state_sha256"]:
                raise AssertionError(f"{arm} terminal artifact changed before evaluation")
            point["metrics"] = _evaluate_b(point_model, config, plan, device)
            point_model.to("cpu")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    paired = []
    for a_point, init_point in zip(arms["a_trained"]["endpoints"], arms["init"]["endpoints"]):
        paired.append({
            "budget_updates": a_point["budget_updates"],
            "b_calibration_nominal_global_input_tokens": a_point["b_calibration_nominal_global_input_tokens"],
            "definition": "a_trained_minus_init; paired terminal diagnostic only",
            "variants": _metric_deltas(a_point["metrics"], init_point["metrics"]),
        })

    report = {
        "contract": TERMINAL_TRANSFER_CONTRACT,
        "experiment_config_sha256": config["_meta"]["hash"],
        "schedule_policy": {
            "type": "independent_budget_terminal_cosine",
            "restart_from_arm_initialization_per_budget": True,
            "warmup_fraction": config["calibration"]["warmup_fraction"],
            "terminal_learning_rate_upper_bound": 1e-9,
        },
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
                "rendering": "b", "seed": plan.evaluation_seed, "episodes": plan.evaluation_episodes,
                "matched_across_arms_and_budgets": True,
                "variants": ["irreversible", "reversible_control"] if config["evaluation"].get("include_reversible_control") else ["irreversible"],
            },
        },
        "arms": arms,
        "paired_diagnostics": paired,
        "scientific_acceptance_policy": "independently annealed terminal-budget endpoints; no capability threshold",
    }
    assert_terminal_transfer_report_contract(report, config, plan)
    atomic_json(report_path, report)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(run(args.config, args.source_run, args.output_dir))


if __name__ == "__main__":
    main()
