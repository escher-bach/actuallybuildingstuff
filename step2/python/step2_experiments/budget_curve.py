"""Does the learner acquire d=3 and d=4, given more budget and more exposure?

The capacity probe settled that the representation can hold four-way binding:
a fixed d=4 cohort fits to ~0.024 teacher-forced action L1 while held-out d=4
sat at 0.3317 in `f9b1647`. What that leaves open is acquisition, and the
only way to read acquisition is as a curve rather than a point.

This run pins the world to the hard dimensions, spends a larger budget on
them, and measures held-out teacher-forced action L1 per dimension at fixed
checkpoints against an identical evaluation support every time. The shape of
the curve is the finding:

  still falling at the end   -> budget was the binding constraint
  flat well above the fit    -> budget is not the constraint, and the
                                remaining suspect is the training
                                distribution, which is expert-only

Training is pure offline behaviour cloning: `ProceduralTrajectoryDataset`
draws oracle trajectories, and the state sequence advances under the oracle's
own action, never the learner's. No learner-visited state ever enters the
training set. A flat curve here is therefore evidence about that choice.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from accelerate import Accelerator
from transformers import Trainer, TrainerCallback, set_seed

from .model import Step2Config, Step2ForTrajectoryPrediction, parameter_report
from .train import (
    ProceduralTrajectoryDataset,
    load_config,
    resolve_project_path,
    teacher_forced_eval,
    training_arguments,
)


class DimensionCurveCallback(TrainerCallback):
    """Record held-out per-dimension teacher-forced L1 on a fixed schedule.

    The evaluation support is regenerated from `validation_seed` at
    `start_index=0` every time, so every checkpoint is scored on identical
    episodes and differences are attributable to training alone.
    """

    def __init__(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        config: dict[str, Any],
        every: int,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.config = config
        self.every = every
        self.points: list[dict[str, Any]] = []

    def record(self, step: int) -> None:
        # The final explicit record can land on a step the schedule already
        # took. Recording it twice would make `points[-2]` and `points[-1]`
        # identical, and the "still improving" comparison between them would
        # then be false by construction rather than by measurement.
        if self.points and self.points[-1]["updates"] == step:
            return
        result = teacher_forced_eval(self.accelerator, self.model, self.config, start_index=0)
        self.model.train()
        self.points.append(
            {
                "updates": step,
                "action_l1": result["action_l1"],
                "future_l1": result["future_l1"],
                "by_dimension": result["by_dimension"],
            }
        )
        print(
            f"[curve] updates={step} action_l1={result['action_l1']:.4f} "
            + " ".join(
                f"d{d}={v['action_l1']:.4f}" for d, v in sorted(result["by_dimension"].items())
            ),
            flush=True,
        )

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, D102
        if state.global_step > 0 and state.global_step % self.every == 0:
            self.record(int(state.global_step))
        return control


def main() -> None:
    parser = argparse.ArgumentParser(description="STEP 2 per-dimension budget curve")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    config["model"] = dict(config["model"])
    config["model"]["config"] = str(resolve_project_path(config_path, config["model"]["config"]))

    curve = config["curve"]
    run = config["run"]
    config["world"] = dict(config["world"])
    config["world"]["d_min"] = int(curve["d_min"])
    config["world"]["d_max"] = int(curve["d_max"])

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model_config = Step2Config.from_project_json(config["model"]["config"])
    set_seed(int(run["seed"]))
    model = Step2ForTrajectoryPrediction(model_config)

    updates = int(curve["updates"])
    arguments = training_arguments(
        output_dir=output_root / "budget-curve-trainer",
        run=run,
        max_steps=updates,
        per_device_batch_size=int(run["per_device_batch_size"]),
        warmup_steps=int(run["warmup_updates"]),
        save=False,
        use_cpu=args.cpu,
    )
    trainer = Trainer(
        model=model,
        args=arguments,
        train_dataset=ProceduralTrajectoryDataset(
            seed=int(config["world"]["train_seed"]),
            max_tokens=int(config["model"]["sequence_length"]),
            world=config["world"],
        ),
    )
    accelerator = Accelerator()
    callback = DimensionCurveCallback(accelerator, model, config, int(curve["eval_every"]))
    trainer.add_callback(callback)

    callback.record(0)
    train_output = trainer.train()
    callback.record(int(trainer.state.global_step))

    points = callback.points
    first, last = points[0], points[-1]
    dimensions = sorted(last["by_dimension"])
    verdict = {
        "still_improving": {
            d: last["by_dimension"][d]["action_l1"]
            < points[-2]["by_dimension"][d]["action_l1"] - 0.005
            for d in dimensions
        },
        "total_reduction": {
            d: first["by_dimension"][d]["action_l1"] - last["by_dimension"][d]["action_l1"]
            for d in dimensions
        },
        "final": {d: last["by_dimension"][d]["action_l1"] for d in dimensions},
        "step_zero": {d: first["by_dimension"][d]["action_l1"] for d in dimensions},
    }
    result = {
        "curve": points,
        "verdict": verdict,
        "updates": updates,
        "world": config["world"],
        "train_loss": float(train_output.metrics.get("train_loss", float("nan"))),
        "parameter_report": parameter_report(model),
        "training_paradigm": "offline-behaviour-cloning-on-oracle-trajectories",
    }
    (output_root / "budget-curve.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(verdict, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
