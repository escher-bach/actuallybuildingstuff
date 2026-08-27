"""Learner capacity probe: can the core represent multi-binding control at all?

This answers a learner question, not a world question. The audited `f9b1647`
candidate solves `d=1` exactly (teacher-forced action L1 `0.0000`) and then
falls off a cliff at `d=2` (`0.2582`), plateauing through `d=4` (`0.3317`).
That cliff is measured on the oracle's own path, so covariate shift does not
explain it. Two hypotheses survive and have never been separated:

1. the representation cannot carry multi-binding identity, because public key
   identity is encoded as a frozen Fourier code that cannot adapt; or
2. 35,000 updates is simply not enough budget.

The probe separates them by asking a necessary-condition question that budget
cannot confound: with world difficulty pinned at `d=4` and a fixed cohort
small enough to memorize, can the model fit it at all?

A representation that cannot fit 64 episodes it has seen thousands of times is
insufficient regardless of budget. A representation that can fit them is not
thereby proven to generalize, but it is exonerated as the cause of the cliff,
and the remaining suspects are budget and data.

Both arms differ in exactly one factor: how `key_ids` become vectors.

    sinusoid   frozen Fourier code -- the canonical `0.2.0` selected profile
    learned    probe arm, `nn.Embedding(max_keys, hidden)`

The decision rule below is predeclared. It is applied mechanically to the
measured numbers and must not be adjusted after reading them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import Trainer, set_seed

from .data import generate_torch_batch
from .model import Step2Config, Step2ForTrajectoryPrediction, parameter_report
from .train import (
    FixedCohortDataset,
    load_config,
    resolve_project_path,
    training_arguments,
)

# Predeclared thresholds on teacher-forced action L1 over the fitted cohort.
# Action targets are normalized to [-1, 1] by `action_limit`, so 0.02 is one
# percent of full range and counts as fit; 0.05 is a generous "clearly not
# fit" bar. The band between them is a deliberate no-decision zone: landing
# there means the probe was too short, not that a conclusion may be invented.
FIT_THRESHOLD = 0.02
FAIL_THRESHOLD = 0.05

ARMS = ("sinusoid", "learned")


def probe_world(world: dict[str, Any], dimension: int) -> dict[str, Any]:
    """Pin world difficulty to a single dimension so the signal is not diluted."""
    pinned = dict(world)
    pinned["d_min"] = dimension
    pinned["d_max"] = dimension
    return pinned


@torch.no_grad()
def cohort_teacher_forced_l1(
    model: Step2ForTrajectoryPrediction,
    cohort: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, float]:
    """Teacher-forced L1 on the very cohort the model was fitted to."""
    was_training = model.training
    model.eval()
    batch = {name: value.to(device) for name, value in cohort.items()}
    output = model(**batch)
    action_mask = batch["action_target_mask"].to(dtype=torch.float32)
    future_mask = batch["future_target_mask"].to(dtype=torch.float32)
    action_abs = (output.action_predictions.float() - batch["action_targets"].float()).abs()
    future_abs = (output.future_predictions.float() - batch["future_targets"].float()).abs()
    model.train(was_training)
    return {
        "action_l1": float((action_abs * action_mask).sum() / action_mask.sum().clamp_min(1.0)),
        "future_l1": float((future_abs * future_mask).sum() / future_mask.sum().clamp_min(1.0)),
        "supervised_action_positions": int(action_mask.sum().item()),
    }


def run_arm(
    arm: str,
    config: dict[str, Any],
    output_root: Path,
    *,
    use_cpu: bool = False,
) -> dict[str, Any]:
    run = config["run"]
    probe = config["probe"]
    world = probe_world(config["world"], int(probe["dimension"]))

    model_config = Step2Config.from_project_json(config["model"]["config"])
    model_config.key_embedding = arm

    cohort, metadata = generate_torch_batch(
        seed=int(world["overfit_seed"]),
        start_index=0,
        batch_size=int(probe["cohort_episodes"]),
        max_tokens=int(config["model"]["sequence_length"]),
        world=world,
    )
    observed = sorted({int(value) for value in metadata["dimensions"]})
    if observed != [int(probe["dimension"])]:
        raise RuntimeError(f"probe cohort is not pinned to one dimension: {observed}")

    updates = int(probe["updates"])
    set_seed(int(run["seed"]))
    model = Step2ForTrajectoryPrediction(model_config)
    arguments = training_arguments(
        output_dir=output_root / f"probe-{arm}",
        run=run,
        max_steps=updates,
        per_device_batch_size=int(probe["per_device_batch_size"]),
        warmup_steps=int(probe.get("warmup_updates", 0)),
        save=False,
        use_cpu=use_cpu,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=FixedCohortDataset(cohort, repeats=max(2, updates)),
    )
    device = trainer.args.device
    before = cohort_teacher_forced_l1(model, cohort, device)
    train_output = trainer.train()
    after = cohort_teacher_forced_l1(model, cohort, device)

    return {
        "arm": arm,
        "key_embedding": arm,
        "dimension": int(probe["dimension"]),
        "cohort_episodes": int(probe["cohort_episodes"]),
        "updates": updates,
        "optimizer_steps": int(trainer.state.global_step),
        "train_loss": float(train_output.metrics.get("train_loss", float("nan"))),
        "fit_before": before,
        "fit_after": after,
        "action_l1_fit": after["action_l1"],
        "parameter_report": parameter_report(model),
    }


def decide(arms: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared decision rule to the two measured arms."""
    k0 = arms["sinusoid"]["action_l1_fit"]
    k1 = arms["learned"]["action_l1_fit"]
    fit0, fit1 = k0 <= FIT_THRESHOLD, k1 <= FIT_THRESHOLD
    bad0, bad1 = k0 > FAIL_THRESHOLD, k1 > FAIL_THRESHOLD

    if bad0 and fit1:
        verdict = "key-encoding-is-the-bottleneck"
        action = "Adopt the learned key embedding, then re-run the hour-scale candidate."
    elif fit0 and fit1:
        verdict = "representation-sufficient-budget-suspect"
        action = "Do not change the architecture. Run a budget ladder at d=4 next."
    elif bad0 and bad1:
        verdict = "deeper-defect"
        action = "Halt world work. Debug the token ABI, loss, or masking before any new world."
    elif fit0 and bad1:
        verdict = "anomalous-learned-arm-worse"
        action = "Investigate init/learning-rate interaction before concluding anything."
    else:
        verdict = "inconclusive-no-decision-band"
        action = (
            "At least one arm landed between the thresholds. Extend updates and re-run; "
            "do not invent a threshold after the fact."
        )

    return {
        "verdict": verdict,
        "next_action": action,
        "action_l1_fit": {"sinusoid": k0, "learned": k1},
        "fit_threshold": FIT_THRESHOLD,
        "fail_threshold": FAIL_THRESHOLD,
        "learner_validated_as_instrument": verdict
        in ("key-encoding-is-the-bottleneck", "representation-sufficient-budget-suspect"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="STEP 2 learner capacity probe")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cpu", action="store_true", help="local reduced-scale smoke")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    config["model"] = dict(config["model"])
    config["model"]["config"] = str(
        resolve_project_path(config_path, config["model"]["config"])
    )
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    arms = {arm: run_arm(arm, config, output_root, use_cpu=args.cpu) for arm in ARMS}
    result = {
        "arms": arms,
        "decision": decide(arms),
        "world": config["world"],
        "probe": config["probe"],
    }
    (output_root / "capacity-probe.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result["decision"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
