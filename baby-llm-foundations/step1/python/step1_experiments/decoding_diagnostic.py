"""Which policy has this stage been measuring?

Every retained evaluation decodes greedily, while every RLVR rollout samples at
temperature 1.0.  Across the outcome-only arms those two quantities moved
independently — at 1e-4 sampled reward held while greedy capability fell nine
points, and in the tuned arm sampled reward rose precisely to meet an unchanged
greedy score.  Either the stage optimized one policy and scored another, or the
two agree and the null stands as reported.

This module answers that by scoring the same checkpoints on the same held-out
worlds under both decoding rules.  It trains nothing and changes no retained
metric: the greedy path is the frozen evaluator, called unchanged.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .artifacts import atomic_json
from .evaluate import EVALUATION_METRIC_NAMES, _aggregate_rows, _execute_batched, _matched_sets
from .train import _state_dict_sha256


DECODING_CONTRACT = "step1_decoding_diagnostic_v1"


def _lookup(report: dict, field: str):
    """Read a possibly dotted field, so nested identity can be matched."""
    value = report
    for part in field.split("."):
        if not isinstance(value, dict):
            return None
        value = value.get(part)
    return value


def locate_run(roots: list[Path], report_name: str, expected: dict) -> Path:
    """Find an attached run *root* by the identity its own report carries.

    ``report_name`` may be nested (``production/training_report.json``), so the
    root is found by climbing exactly as many levels as the name has parts.
    """
    depth = len(Path(report_name).parts)
    matches = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob(report_name):
            try:
                report = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if all(_lookup(report, field) == value for field, value in expected.items()):
                matches.append(path.parents[depth - 1].resolve())
    unique = sorted(set(matches))
    if len(unique) != 1:
        raise RuntimeError(f"expected exactly one attached run matching {expected}, found {unique}")
    return unique[0]


def resolve_checkpoints(config: dict, roots: list[Path]) -> list[dict]:
    """Turn declared model entries into verified on-disk artifact paths."""
    resolved = []
    for entry in config["models"]:
        if entry["kind"] == "dense":
            run = locate_run(roots, "production/training_report.json", {
                "contract": "step1_dense_training_v1",
                "source_git_sha": entry["git_sha"],
                "config_hash": entry["config_hash"],
            })
            artifact = run / "production" / "model"
        elif entry["kind"] == "rlvr":
            run = locate_run(roots, "rlvr_report.json", {
                "contract": "step1_rlvr_grpo_v1",
                "experiment_config_sha256": entry["config_hash"],
            })
            report = json.loads((run / "rlvr_report.json").read_text())
            artifact = run / report["plan"]["arm"] / "checkpoints" / f"checkpoint-{entry['checkpoint_updates']}"
        elif entry["kind"] == "transfer":
            # A terminal-transfer endpoint: an arm trained to a declared budget
            # on Rendering B, starting either from the dense A model or from the
            # same random initialization.
            run = locate_run(roots, "rendering_b_terminal_transfer_report.json", {
                "contract": "step1_rendering_b_terminal_transfer_v1",
                "source.git_sha": entry["source_git_sha"],
            })
            budget = entry["budget_updates"]
            artifact = run / entry["arm"] / f"budget-{budget}" / "checkpoints" / f"checkpoint-{budget}"
        elif entry["kind"] == "learner_conditioned":
            # A learner- or teacher-conditioned arm endpoint. Both regimes write
            # the same report and the same budget-N layout; `collection.policy`
            # is what separated them, and the config hash carries it.
            run = locate_run(roots, "learner_conditioned_report.json", {
                "contract": "step1_learner_conditioned_dagger_v1",
                "experiment_config_sha256": entry["config_hash"],
            })
            report = json.loads((run / "learner_conditioned_report.json").read_text())
            artifact = run / report["plan"]["arm"] / f"budget-{entry['budget_updates']}" / "model"
        else:
            raise ValueError(f"unknown model kind {entry['kind']!r}")
        if not (artifact / "config.json").is_file():
            raise RuntimeError(f"resolved artifact is not a model directory: {artifact}")
        resolved.append({**entry, "run": str(run), "artifact": str(artifact)})
    return resolved


@torch.no_grad()
def score(entry: dict, config: dict, device, shard: Path | None = None) -> dict:
    """Score one checkpoint under every declared decoding rule.

    Closed-loop scoring requires the model to emit parseable actions; when it
    cannot, every arm reads 0.0% and the metric has no resolution. The optional
    teacher-forced diagnostic is graded and needs no parseable output, so it can
    separate "cannot speak this rendering" from "knows nothing about it".
    """
    from transformers import AutoModelForCausalLM, set_seed

    model = AutoModelForCausalLM.from_pretrained(entry["artifact"], local_files_only=True).to(device).eval()
    state_sha = _state_dict_sha256(model.state_dict())
    if entry.get("model_state_sha256") and state_sha != entry["model_state_sha256"]:
        raise AssertionError(f"{entry['name']} state hash mismatch: {state_sha}")
    params, seed, count, rendering, comparison = _matched_sets(config)["validation"]
    count = config["evaluation"]["episodes"]
    nll = None
    if shard is not None:
        from .rlvr import _teacher_forced_nll

        nll = _teacher_forced_nll(model, config, shard, device)
    results = []
    for rule in (config["decoding"] if config["evaluation"].get("closed_loop", True) else []):
        for index in range(rule.get("repeats", 1)):
            # The same worlds every time; only the decoding rule and its sampling
            # seed change, so differences cannot come from the evaluation set.
            set_seed(config["run"]["root_seed"] + index)
            metrics = _aggregate_rows(_execute_batched(
                model, params, seed, count, rendering, device, temperature=rule["temperature"],
            ))
            for name, value in metrics.items():
                if value is not None and not math.isfinite(value):
                    raise FloatingPointError(f"non-finite metric {entry['name']}.{rule['name']}.{name}")
            results.append({"decoding": rule["name"], "temperature": rule["temperature"],
                            "sampling_seed_offset": index, "metrics": metrics})
    retry_tolerant = None
    if config["evaluation"].get("retry_tolerant"):
        # The frozen evaluator ends an episode at its first protocol failure.
        # The world does not: `step_attempts` leaves the state unchanged and the
        # episode running, and the teacher labels that unchanged state. Scoring a
        # regime trained on recovery with a rule that grants none charges it for
        # behaviour the world permits, so both readings are reported.
        from .learner import CollectionSettings
        from .learner_conditioned import _execute_retry_tolerant, aggregate_retry_tolerant

        settings = CollectionSettings(
            max_turns=config["evaluation"]["retry_max_turns"],
            max_action_tokens=96,
            max_consecutive_failures=config["evaluation"]["max_retries"] + 1,
            context_length=config["world"]["context_length"],
        )
        rows = _execute_retry_tolerant(
            model, params, seed, count, rendering, device, settings,
            config["evaluation"]["max_retries"],
        )
        retry_tolerant = aggregate_retry_tolerant(rows)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {**entry, "model_state_sha256": state_sha, "teacher_forced_action_nll": nll,
            "retry_tolerant": retry_tolerant, "evaluation_set": {
        "seed": seed, "episodes": count, "rendering": rendering, "comparison": comparison,
    }, "results": results}


def assert_diagnostic_contract(report: dict, config: dict) -> None:
    if report.get("contract") != DECODING_CONTRACT:
        raise AssertionError("decoding diagnostic contract mismatch")
    if report.get("experiment_config_sha256") != config["_meta"]["hash"]:
        raise AssertionError("decoding diagnostic configuration hash mismatch")
    closed_loop = config["evaluation"].get("closed_loop", True)
    expected_rules = ([(rule["name"], index) for rule in config["decoding"] for index in range(rule.get("repeats", 1))]
                      if closed_loop else [])
    if len(report.get("models", [])) != len(config["models"]):
        raise AssertionError("decoding diagnostic model count mismatch")
    for model in report["models"]:
        if [(item["decoding"], item["sampling_seed_offset"]) for item in model["results"]] != expected_rules:
            raise AssertionError(f"{model['name']} did not run every declared decoding rule")
        for item in model["results"]:
            if set(item["metrics"]) != set(EVALUATION_METRIC_NAMES):
                raise AssertionError(f"{model['name']} {item['decoding']} metric fields mismatch")
        if config["evaluation"].get("teacher_forced_nll"):
            value = model.get("teacher_forced_action_nll")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
                raise AssertionError(f"{model['name']} has no finite teacher-forced diagnostic")
        if config["evaluation"].get("retry_tolerant"):
            result = model.get("retry_tolerant")
            if not result or not isinstance(result.get("success_rate"), (int, float)):
                raise AssertionError(f"{model['name']} has no retry-tolerant score")
        # Every model must be scored on the identical world set.
        if model["evaluation_set"]["seed"] != report["models"][0]["evaluation_set"]["seed"]:
            raise AssertionError("models were scored on different evaluation worlds")


def run(resolved_config: Path, run_dir: Path, input_roots: list[Path] | None = None) -> Path:
    from .rlvr import load_config

    config = load_config(resolved_config)
    roots = input_roots or [Path("/kaggle/input")]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    shard = None
    if config["evaluation"].get("teacher_forced_nll"):
        from .rlvr import _diagnostic_shard

        # One deterministic teacher shard on the evaluated rendering's own
        # validation seeds; every model is scored against the same targets.
        shard = _diagnostic_shard(config, run_dir)
    models = [score(entry, config, device, shard) for entry in resolve_checkpoints(config, roots)]
    greedy = {model["name"]: item for model in models
              for item in model["results"] if item["temperature"] == 0.0}
    report = {
        "contract": DECODING_CONTRACT,
        "experiment_config_sha256": config["_meta"]["hash"],
        "question": (
            "does the decoding rule decide the stage's conclusion? every retained metric is greedy; "
            "every RLVR rollout sampled at temperature 1.0"
        ),
        "models": models,
        "greedy_reference": {name: item["metrics"]["success_rate"] for name, item in greedy.items()},
        "teacher_forced_reference": {model["name"]: model["teacher_forced_action_nll"] for model in models},
        "scientific_acceptance_policy": "diagnostic only; it re-scores existing checkpoints and retrains nothing",
    }
    assert_diagnostic_contract(report, config)
    path = run_dir / "decoding_diagnostic_report.json"
    atomic_json(path, report)
    atomic_json(run_dir / "analysis" / "result-report.json", report)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, action="append")
    args = parser.parse_args()
    print(run(args.resolved_config, args.run_dir, args.input_root))


if __name__ == "__main__":
    main()
