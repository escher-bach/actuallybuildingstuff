"""DAgger arm: train on states the learner actually visits, labelled by the oracle.

`f9b1647` is the control. It trained pure offline behaviour cloning on
oracle trajectories: the supervision target was the oracle's action and the
state sequence advanced under the oracle's own action, so no learner-visited
state ever entered the training set. It fit its own distribution and degraded
off it -- closed-loop terminal error `0.3743` against a zero-action policy's
`0.3385`, error falling after the first action and then rising monotonically.
That is the compounding-covariate-shift signature of behaviour cloning.

This arm changes exactly one thing: where the training states come from.
Everything else -- world `d=1..4`, 35,000 updates, seed, learning rate, batch
size, architecture -- is held at the audited control's values, so the
comparison is paired and attributable.

The labelling stays inside the information boundary. `RolloutBatch`'s oracle
is `PublicOracle::from_public_prefix`, which reconstructs the mapping from the
public calibration prefix rather than from latent instance state. Its action
at a learner-visited state is therefore still derivable from what the learner
itself observes, which is what `WORLD-VALIDITY.md` requires of a supervised
target. The method is named "privileged" because feeding an oracle action
back as a model *input* would leak the answer; using it as a target is what
`generate_trajectory` already does for every expert trajectory.

Schedule. Rolling out an untrained policy produces states that are off
distribution in an uninformative way, so the run begins at beta=1 (pure
expert) for a warmup, then mixes a fixed fraction of on-policy episodes drawn
from a bounded buffer that is refreshed with the current policy. This is
DAgger with aggregation over a sliding window rather than over all history,
which keeps memory bounded and keeps the buffer on-policy.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import deque
from pathlib import Path
from typing import Any, Iterator

import torch
from accelerate import Accelerator
from torch.utils.data import IterableDataset
from transformers import Trainer, TrainerCallback, set_seed

import step2_world_py

from .data import MODEL_FIELDS, generate_torch_batch, tensorize_rollout, world_kwargs
from .model import Step2Config, Step2ForTrajectoryPrediction, parameter_report
from .train import (
    ProceduralTrajectoryDataset,
    closed_loop_eval,
    load_config,
    move_batch,
    resolve_project_path,
    teacher_forced_eval,
    training_arguments,
)


@torch.no_grad()
def collect_dagger_episodes(
    accelerator: Accelerator,
    model: torch.nn.Module,
    config: dict[str, Any],
    *,
    seed: int,
    start_index: int,
    episodes: int,
) -> list[dict[str, torch.Tensor]]:
    """Roll the current policy forward, then label every visited state.

    The learner's action decides which state comes next; the oracle's action
    at that state becomes the target. That inversion is the whole of DAgger.
    """
    world = config["world"]
    sequence_length = int(config["model"]["sequence_length"])
    rollouts = step2_world_py.RolloutBatch(
        seed=seed,
        start_index=start_index,
        batch_size=episodes,
        max_tokens=sequence_length,
        **world_kwargs(world),
    )
    was_training = model.training
    model.eval()
    # token position -> oracle action at the state the learner was actually in
    labels: list[dict[int, float]] = [{} for _ in range(episodes)]

    for _ in range(int(world["max_control_steps"])):
        if rollouts.all_done():
            break
        raw = rollouts.learner_batch()
        oracle_actions = rollouts.privileged_oracle_actions()
        tensors = move_batch(tensorize_rollout(raw, "cpu"), accelerator.device)
        predictions = model(**tensors).action_predictions[..., 0].detach().float().cpu()
        actions: list[list[float]] = []
        for row, dimension in enumerate(raw["dimensions"]):
            if raw["done"][row]:
                actions.append([])
                continue
            positions = raw["query_positions"][row][:dimension]
            for slot, position in enumerate(positions):
                labels[row][int(position)] = float(oracle_actions[row][slot])
            actions.append([float(predictions[row, position]) for position in positions])
        rollouts.step(actions)

    final = tensorize_rollout(rollouts.learner_batch(), "cpu")
    model.train(was_training)

    collected: list[dict[str, torch.Tensor]] = []
    for row in range(episodes):
        if not labels[row]:
            continue
        record = {name: final[name][row].clone() for name in MODEL_FIELDS}
        # Rollout tokens carry no supervision; the oracle labels are written
        # into slot zero, which is the only action slot the world ever uses.
        record["action_targets"].zero_()
        record["action_target_mask"].zero_()
        for position, value in labels[row].items():
            record["action_targets"][position, 0] = value
            record["action_target_mask"][position, 0] = 1.0
        # Future supervision is not available for rollout states, so it is
        # left masked out rather than fabricated.
        record["future_target_mask"].zero_()
        collected.append(record)
    return collected


class MixedTrajectoryDataset(IterableDataset):
    """Expert stream, with a fixed fraction drawn from the on-policy buffer."""

    def __init__(
        self,
        *,
        expert: ProceduralTrajectoryDataset,
        buffer: deque,
        fraction: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.expert = expert
        self.buffer = buffer
        self.fraction = fraction
        self.rng = random.Random(seed)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        stream = iter(self.expert)
        while True:
            if self.buffer and self.rng.random() < self.fraction:
                yield {
                    name: tensor.clone()
                    for name, tensor in self.rng.choice(self.buffer).items()
                }
            else:
                yield next(stream)


class DaggerRefreshCallback(TrainerCallback):
    """Refill the on-policy buffer with the current policy on a fixed schedule."""

    def __init__(
        self,
        accelerator: Accelerator,
        model: torch.nn.Module,
        config: dict[str, Any],
        buffer: deque,
    ) -> None:
        self.accelerator = accelerator
        self.model = model
        self.config = config
        self.buffer = buffer
        self.dagger = config["dagger"]
        self.refreshes = 0
        self.log: list[dict[str, Any]] = []

    def refresh(self, step: int) -> None:
        episodes = int(self.dagger["episodes_per_refresh"])
        collected = collect_dagger_episodes(
            self.accelerator,
            self.model,
            self.config,
            seed=int(self.config["world"]["rollout_seed"]) + 1_000_003 * (self.refreshes + 1),
            start_index=self.accelerator.process_index * episodes,
            episodes=episodes,
        )
        self.buffer.extend(collected)
        self.refreshes += 1
        entry = {"updates": step, "collected": len(collected), "buffer": len(self.buffer)}
        self.log.append(entry)
        print(f"[dagger] {entry}", flush=True)

    def on_step_end(self, args, state, control, **kwargs):  # noqa: ANN001, D102
        step = int(state.global_step)
        warmup = int(self.dagger["warmup_updates"])
        every = int(self.dagger["refresh_every"])
        if step >= warmup and (step - warmup) % every == 0:
            self.refresh(step)
        return control


def main() -> None:
    parser = argparse.ArgumentParser(description="STEP 2 DAgger arm")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_config(config_path)
    config["model"] = dict(config["model"])
    config["model"]["config"] = str(resolve_project_path(config_path, config["model"]["config"]))

    run = config["run"]
    dagger = config["dagger"]
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    model_config = Step2Config.from_project_json(config["model"]["config"])
    set_seed(int(run["seed"]))
    model = Step2ForTrajectoryPrediction(model_config)

    updates = int(dagger["updates"])
    buffer: deque = deque(maxlen=int(dagger["buffer_episodes"]))
    expert = ProceduralTrajectoryDataset(
        seed=int(config["world"]["train_seed"]),
        max_tokens=int(config["model"]["sequence_length"]),
        world=config["world"],
    )
    arguments = training_arguments(
        output_dir=output_root / "dagger-trainer",
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
        train_dataset=MixedTrajectoryDataset(
            expert=expert,
            buffer=buffer,
            fraction=float(dagger["fraction"]),
            seed=int(run["seed"]),
        ),
    )
    accelerator = Accelerator()
    callback = DaggerRefreshCallback(accelerator, model, config, buffer)
    trainer.add_callback(callback)

    rollout_episodes = int(config["preflight"]["rollout_episodes_per_rank"])
    before = {
        "teacher_forced": teacher_forced_eval(accelerator, model, config, start_index=0),
        "closed_loop": closed_loop_eval(
            accelerator,
            model,
            config,
            seed=int(config["world"]["rollout_seed"]),
            start_index=0,
            episodes_per_rank=rollout_episodes,
        ),
    }
    train_output = trainer.train()
    model.eval()
    after = {
        "teacher_forced": teacher_forced_eval(accelerator, model, config, start_index=0),
        "closed_loop": closed_loop_eval(
            accelerator,
            model,
            config,
            seed=int(config["world"]["rollout_seed"]),
            start_index=0,
            episodes_per_rank=rollout_episodes,
        ),
    }

    result = {
        "training_paradigm": "dagger-on-policy-states-labelled-by-public-prefix-oracle",
        "control_run": "step2-architecture-world-f9b1647",
        "updates": updates,
        "dagger": dict(dagger),
        "refresh_log": callback.log,
        "step_zero": before,
        "trained": after,
        "train_loss": float(train_output.metrics.get("train_loss", float("nan"))),
        "parameter_report": parameter_report(model),
        "world": config["world"],
    }
    (output_root / "dagger-result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    summary = {
        "closed_loop_terminal_error": {
            "step_zero": before["closed_loop"]["terminal_error"],
            "trained": after["closed_loop"]["terminal_error"],
        },
        "teacher_forced_action_l1": {
            "step_zero": before["teacher_forced"]["action_l1"],
            "trained": after["teacher_forced"]["action_l1"],
        },
        "by_dimension_terminal_error": {
            d: v["terminal_error"] for d, v in sorted(after["closed_loop"]["by_dimension"].items())
        },
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
