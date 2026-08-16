"""How much outcome signal the world offers a well-formed but unguided policy.

A byte-level policy that cannot emit a parseable action receives no reward, and
a group of identical rewards produces no GRPO gradient.  That is a fact about
the action surface, not about the task.  This diagnostic answers the separate
question the surface hides: if the interface were already grounded, would
outcome-only reward have anything to learn from?

It uses no model and no teacher targets — only the world's own valid-action set
— so it isolates reward density from policy quality.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from .artifacts import atomic_json
from .data import _family


POLICIES = ("commit_immediately", "uniform_valid", "probe_biased")


def _rollout(params: dict, seed: int, rng: random.Random, policy: str) -> bool:
    from world_py import Batch

    batch = Batch(_family(params), seed=seed, n_episodes=1)
    n_probe = params["n_probe"]
    while not batch.done()[0]:
        valid = batch.valid_actions()[0]
        probes = [action for action in valid if action < n_probe]
        commits = [action for action in valid if action >= n_probe]
        if policy == "commit_immediately" or not probes:
            action = rng.choice(commits or valid)
        elif policy == "probe_biased" and rng.random() < 0.7:
            action = rng.choice(probes)
        else:
            action = rng.choice(valid)
        batch.step([action])
    terminated, correct, *_rest = batch.privileged_outcomes()[0]
    return bool(terminated and correct)


def teacher_success(params: dict, seed: int) -> bool:
    from world_py import Batch

    batch = Batch(_family(params), seed=seed, n_episodes=1)
    while not batch.done()[0]:
        batch.step([min(batch.privileged_teacher_targets()[0]["preferred_actions"])])
    terminated, correct, *_rest = batch.privileged_outcomes()[0]
    return bool(terminated and correct)


def measure(params: dict, seed_base: int, episodes: int, group_size: int, groups: int, rng_seed: int = 0) -> dict:
    """Success rate per policy, and the share of groups that carry a gradient."""
    rng = random.Random(rng_seed)
    rates = {
        policy: sum(_rollout(params, seed_base + index, rng, policy) for index in range(episodes)) / episodes
        for policy in POLICIES
    }
    informative = 0
    for index in range(groups):
        seed = seed_base + 50_000 + index
        wins = sum(_rollout(params, seed, rng, "uniform_valid") for _ in range(group_size))
        informative += 0 < wins < group_size
    return {
        "episodes_per_policy": episodes,
        "success_rate": rates,
        "teacher_success_rate": sum(teacher_success(params, seed_base + index) for index in range(episodes)) / episodes,
        "group_size": group_size,
        "groups_sampled": groups,
        # This is the quantity GRPO actually consumes: a group whose rewards are
        # all equal contributes exactly zero advantage.
        "fraction_of_groups_with_reward_variance": informative / groups,
        "policy_definitions": {
            "commit_immediately": "uniform over commitments, no probing",
            "uniform_valid": "uniform over the world's valid actions",
            "probe_biased": "probes with probability 0.7 while probes remain",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True, help="any Step 1 config; only [world] is read")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--groups", type=int, default=400)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    import tomllib

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    result = measure(config["world"], config["run"]["root_seed"] + config["rollout"]["train_seed_offset"],
                     args.episodes, args.group_size, args.groups)
    if args.output:
        atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
