"""Held-out diagnostic for the renderer's raw action-order fingerprint.

The fitted policy receives only the exact reset prefix seen by the causal LM.
Training labels are recovered by trying public terminal commitment actions in
fresh evaluator-matched worlds and reading the verifier's terminal correctness
bit.  The policy is frozen before a disjoint seed range is scored.

This is deliberately not a capability ceiling.  Its purpose is to separate
two effects that a same-support optimum conflates:

* an exact raw-prefix lookup can memorize a high-cardinality renderer
  fingerprint; and
* a held-out lookup can use that fingerprint only when the exact key recurs.

The canonical baseline discards action ordering and conditions only on the
serialized ``BUDGET`` line.  No function in this module reads an ``Instance``,
an evidence table, probe costs, latent truth, ``consistent()``, teacher
targets, ``valid_actions()``, or a replay key.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import inspect
import json
import struct
import tomllib
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from .protocol import BOS
from .public_information_ceiling import (
    COMMIT_RE,
    Prefix,
    append_observation,
    available_actions,
    current_observation,
    wilson_interval,
)
from .world_config import family_from_config as _family


DIAGNOSTIC_CONTRACT = "step1_raw_fingerprint_diagnostic_v1"
BatchFactory = Callable[[dict, int, int], object]
ActionParser = Callable[[str, dict], int]


def reset_prefix(observation: str) -> Prefix:
    """The exact evaluator prefix at the first action decision."""
    return append_observation((BOS,), observation)


def budget_key(observation: str) -> str:
    """Canonical public key with raw action ordering removed completely."""
    matches = [line for line in observation.splitlines() if line.startswith("BUDGET ")]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one BUDGET line, found {len(matches)}")
    return matches[0]


def commitment_actions(observation: str) -> tuple[str, ...]:
    """Terminal candidate actions parsed only from the serialized list."""
    commitments = tuple(action for action in available_actions(observation) if COMMIT_RE.match(action))
    if not commitments:
        raise ValueError("reset observation exposes no commitment action")
    return commitments


def _commit_number(action: str) -> int:
    match = COMMIT_RE.match(action)
    if not match:
        raise ValueError(f"not a canonical commitment action: {action!r}")
    return int(match.group(1))


def _majority(counter: Counter[str]) -> str:
    if not counter:
        raise ValueError("cannot choose a majority action from an empty counter")
    return min(counter, key=lambda action: (-counter[action], _commit_number(action)))


def _policy_digest(
    raw_actions: Mapping[Prefix, str],
    budget_actions: Mapping[str, str],
    default_action: str,
) -> str:
    """Stable identity proving the mapping was fixed before held-out scoring."""
    digest = hashlib.sha256()
    digest.update(b"step1-raw-fingerprint-policy-v1\0")
    for prefix, action in sorted(raw_actions.items()):
        digest.update(struct.pack("<I", len(prefix)))
        for token in prefix:
            digest.update(struct.pack("<H", token))
        digest.update(action.encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"budget\0")
    for key, action in sorted(budget_actions.items()):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(action.encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"default\0")
    digest.update(default_action.encode("utf-8"))
    return digest.hexdigest()


class FrozenFingerprintPolicy:
    """Immutable raw lookup plus a budget-only fallback.

    ``choose`` intentionally accepts one value: the serialized prefix.  The
    object stores mappings and action strings, never a world or world metadata.
    """

    __slots__ = (
        "_raw_actions", "_budget_actions", "default_action", "policy_hash", "_sealed"
    )

    def __init__(
        self,
        raw_actions: Mapping[Prefix, str],
        budget_actions: Mapping[str, str],
        default_action: str,
    ) -> None:
        raw = dict(raw_actions)
        budget = dict(budget_actions)
        object.__setattr__(self, "_raw_actions", MappingProxyType(raw))
        object.__setattr__(self, "_budget_actions", MappingProxyType(budget))
        object.__setattr__(self, "default_action", default_action)
        object.__setattr__(self, "policy_hash", _policy_digest(raw, budget, default_action))
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("frozen fingerprint policy cannot be modified")

    def choose(self, prefix: Prefix) -> str:
        """Choose from the raw key, falling back to the public budget key."""
        observation = current_observation(prefix)
        return self._raw_actions.get(
            prefix,
            self._budget_actions.get(budget_key(observation), self.default_action),
        )

    def choose_budget_only(self, prefix: Prefix) -> str:
        """The frozen canonical baseline, independent of action ordering."""
        observation = current_observation(prefix)
        return self._budget_actions.get(budget_key(observation), self.default_action)

    def has_raw_key(self, prefix: Prefix) -> bool:
        return prefix in self._raw_actions

    @property
    def raw_key_count(self) -> int:
        return len(self._raw_actions)

    @property
    def budget_key_count(self) -> int:
        return len(self._budget_actions)


@dataclass(frozen=True)
class TrainingRecord:
    prefix: Prefix
    correct_action: str


def fit_policy(records: Sequence[TrainingRecord]) -> FrozenFingerprintPolicy:
    """Fit empirical-majority actions and return an immutable policy."""
    if not records:
        raise ValueError("cannot fit a fingerprint policy without training records")
    raw_counts: dict[Prefix, Counter[str]] = defaultdict(Counter)
    budget_counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_counts: Counter[str] = Counter()
    for record in records:
        observation = current_observation(record.prefix)
        if record.correct_action not in commitment_actions(observation):
            raise ValueError("training label is absent from its serialized AVAILABLE list")
        raw_counts[record.prefix][record.correct_action] += 1
        budget_counts[budget_key(observation)][record.correct_action] += 1
        global_counts[record.correct_action] += 1
    return FrozenFingerprintPolicy(
        {key: _majority(counts) for key, counts in raw_counts.items()},
        {key: _majority(counts) for key, counts in budget_counts.items()},
        _majority(global_counts),
    )


def _default_batch_factory(params: dict, seed: int, n_episodes: int):
    from world_py import Batch

    return Batch(_family(params), seed=seed, n_episodes=n_episodes)


def _default_action_parser(action: str, params: dict) -> int:
    from world_py import parse_action

    return int(parse_action(action, params["n_probe"], params["n_hyp"], "a"))


def _new_evaluator_world(factory: BatchFactory, params: dict, seed: int):
    """Create exactly the one-episode world used by the frozen evaluator."""
    batch = factory(params, seed, 1)
    if batch.n_episodes != 1:
        raise AssertionError("fingerprint diagnostic requires Batch(..., n_episodes=1)")
    return batch


def _initial_observation(batch) -> str:
    observations = batch.observations("a")
    if len(observations) != 1:
        raise AssertionError("one-episode batch returned the wrong observation count")
    observation = observations[0]
    if not observation.endswith("STATUS running"):
        raise AssertionError("fingerprint diagnostic expected a running reset observation")
    return observation


def _score_terminal_commit(
    batch,
    params: dict,
    expected_observation: str,
    action: str,
    parse_action: ActionParser,
) -> bool:
    """Execute one public commit and read correctness only after termination."""
    if _initial_observation(batch) != expected_observation:
        raise AssertionError("recreated seed did not reproduce the exact reset observation")
    if action not in commitment_actions(expected_observation):
        raise AssertionError("policy selected a commitment absent from the serialized observation")
    batch.step([parse_action(action, params)])
    if not batch.done()[0]:
        raise AssertionError("diagnostic is frozen to an irreversible terminal commitment world")
    outcome = batch.privileged_outcomes()[0]
    if not bool(outcome[0]):
        raise AssertionError("terminal commit did not produce a verifier outcome")
    return bool(outcome[1])


def recover_training_record(
    params: dict,
    seed: int,
    *,
    batch_factory: BatchFactory = _default_batch_factory,
    parse_action: ActionParser = _default_action_parser,
) -> TrainingRecord:
    """Recover one label using only public commits and terminal rewards."""
    first = _new_evaluator_world(batch_factory, params, seed)
    observation = _initial_observation(first)
    for index, action in enumerate(commitment_actions(observation)):
        trial = first if index == 0 else _new_evaluator_world(batch_factory, params, seed)
        if _score_terminal_commit(trial, params, observation, action, parse_action):
            return TrainingRecord(reset_prefix(observation), action)
    raise AssertionError("no serialized commitment received a correct terminal reward")


def collect_training_records(
    params: dict,
    base_seed: int,
    episodes: int,
    *,
    batch_factory: BatchFactory = _default_batch_factory,
    parse_action: ActionParser = _default_action_parser,
) -> list[TrainingRecord]:
    if episodes <= 0:
        raise ValueError("training episode count must be positive")
    return [
        recover_training_record(
            params,
            base_seed + index,
            batch_factory=batch_factory,
            parse_action=parse_action,
        )
        for index in range(episodes)
    ]


def _key_statistics(prefixes: Iterable[Prefix]) -> dict:
    counts = Counter(prefixes)
    sizes = Counter(counts.values())
    return {
        "unique_keys": len(counts),
        "singleton_keys": sizes.get(1, 0),
        "singleton_episode_rate": sizes.get(1, 0) / sum(counts.values()),
        "collision_pairs": sum(count * (count - 1) // 2 for count in counts.values()),
        "max_bucket_size": max(counts.values()),
        "bucket_size_histogram": {str(size): count for size, count in sorted(sizes.items())},
    }


def _accuracy(successes: int, episodes: int) -> dict:
    low, high = wilson_interval(successes, episodes)
    return {
        "successes": successes,
        "episodes": episodes,
        "accuracy": successes / episodes,
        "wilson_95": [low, high],
    }


def evaluate_frozen_policy(
    params: dict,
    policy: FrozenFingerprintPolicy,
    base_seed: int,
    episodes: int,
    *,
    batch_factory: BatchFactory = _default_batch_factory,
    parse_action: ActionParser = _default_action_parser,
) -> dict:
    """Score a pre-fitted policy on a disjoint evaluator-matched seed range."""
    if episodes <= 0:
        raise ValueError("test episode count must be positive")
    raw_successes = canonical_successes = covered_successes = uncovered_successes = 0
    covered = 0
    raw_prefixes: list[Prefix] = []
    budget_keys: list[str] = []

    for index in range(episodes):
        seed = base_seed + index
        raw_world = _new_evaluator_world(batch_factory, params, seed)
        observation = _initial_observation(raw_world)
        prefix = reset_prefix(observation)
        raw_prefixes.append(prefix)
        budget_keys.append(budget_key(observation))

        is_covered = policy.has_raw_key(prefix)
        raw_action = policy.choose(prefix)
        canonical_action = policy.choose_budget_only(prefix)
        raw_correct = _score_terminal_commit(
            raw_world, params, observation, raw_action, parse_action
        )
        raw_successes += raw_correct
        if is_covered:
            covered += 1
            covered_successes += raw_correct
        else:
            uncovered_successes += raw_correct

        if canonical_action == raw_action:
            canonical_correct = raw_correct
        else:
            canonical_world = _new_evaluator_world(batch_factory, params, seed)
            canonical_correct = _score_terminal_commit(
                canonical_world, params, observation, canonical_action, parse_action
            )
        canonical_successes += canonical_correct

    raw_result = _accuracy(raw_successes, episodes)
    canonical_result = _accuracy(canonical_successes, episodes)
    return {
        "seed": base_seed,
        "episodes": episodes,
        "seed_policy": "Batch(seed=base_seed + episode_index, n_episodes=1)",
        "raw_reset_key_statistics": _key_statistics(raw_prefixes),
        "canonical_budget_key_count": len(set(budget_keys)),
        "exact_raw_key": {
            "coverage_count": covered,
            "coverage_rate": covered / episodes,
            "covered_accuracy": (
                _accuracy(covered_successes, covered) if covered else None
            ),
            "uncovered_budget_fallback_accuracy": (
                _accuracy(uncovered_successes, episodes - covered)
                if covered < episodes else None
            ),
            "with_budget_fallback": raw_result,
        },
        "canonical_budget_only": canonical_result,
        "raw_minus_canonical_accuracy": raw_result["accuracy"] - canonical_result["accuracy"],
    }


def _ranges_overlap(base_a: int, count_a: int, base_b: int, count_b: int) -> bool:
    return base_a < base_b + count_b and base_b < base_a + count_a


def diagnose(
    params: dict,
    train_seed: int,
    train_episodes: int,
    test_seed: int,
    test_episodes: int,
    *,
    batch_factory: BatchFactory = _default_batch_factory,
    parse_action: ActionParser = _default_action_parser,
) -> dict:
    """Fit on one seed range, freeze, then evaluate on a disjoint range."""
    if params.get("rendering") != "a":
        raise ValueError("raw fingerprint diagnostic v1 requires Rendering A")
    if params.get("variant") != "irreversible":
        raise ValueError("terminal-reward label recovery requires the irreversible variant")
    if _ranges_overlap(train_seed, train_episodes, test_seed, test_episodes):
        raise ValueError("training and test seed ranges must be disjoint")

    records = collect_training_records(
        params,
        train_seed,
        train_episodes,
        batch_factory=batch_factory,
        parse_action=parse_action,
    )
    policy = fit_policy(records)
    # Everything below this line is held-out scoring. The immutable policy is
    # never updated, and its hash is recorded before and after as a guard.
    frozen_hash = policy.policy_hash
    held_out = evaluate_frozen_policy(
        params,
        policy,
        test_seed,
        test_episodes,
        batch_factory=batch_factory,
        parse_action=parse_action,
    )
    if policy.policy_hash != frozen_hash:
        raise AssertionError("fingerprint policy changed during held-out evaluation")

    training_prefixes = [record.prefix for record in records]
    training_raw_successes = sum(
        policy.choose(record.prefix) == record.correct_action for record in records
    )
    training_budget_successes = sum(
        policy.choose_budget_only(record.prefix) == record.correct_action for record in records
    )
    training_budget_keys = {
        budget_key(current_observation(record.prefix)) for record in records
    }
    return {
        "contract": DIAGNOSTIC_CONTRACT,
        "question": (
            "does the exact raw action-order fingerprint support a frozen held-out policy "
            "beyond a canonical public-budget baseline?"
        ),
        "policy_boundary": {
            "policy_input": "exact reset causal-LM prefix",
            "training_label": "first public Commit action receiving correct terminal verifier reward",
            "held_out_reward": "terminal correctness bit after the frozen policy commits",
            "forbidden": [
                "Instance", "evidence table", "probe costs", "truth", "consistent()",
                "teacher targets", "valid_actions()", "replay_key()",
            ],
        },
        "fit": {
            "seed": train_seed,
            "episodes": train_episodes,
            "seed_policy": "Batch(seed=base_seed + episode_index, n_episodes=1)",
            "raw_reset_key_statistics": _key_statistics(training_prefixes),
            "canonical_budget_key_count": len(training_budget_keys),
            "exact_raw_transductive_accuracy": _accuracy(
                training_raw_successes, train_episodes
            ),
            "canonical_budget_transductive_accuracy": _accuracy(
                training_budget_successes, train_episodes
            ),
        },
        "frozen_policy": {
            "sha256": frozen_hash,
            "raw_key_count": policy.raw_key_count,
            "canonical_budget_key_count": policy.budget_key_count,
            "frozen_before_test": True,
        },
        "held_out": held_out,
        "interpretation_rule": (
            "high fit accuracy with low held-out key coverage is transductive memorization, "
            "not reusable public information"
        ),
    }


def assert_dependency_free_imports() -> None:
    """Guard the laptop audit against accidental Torch/data-stack imports."""
    source = inspect.getsource(inspect.getmodule(assert_dependency_free_imports))
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import) and any(name.name == "torch" for name in node.names):
            raise AssertionError("raw fingerprint diagnostic imports Torch")
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "data" or module.endswith(".data"):
                raise AssertionError("raw fingerprint diagnostic imports the data stack")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-seed", type=int)
    parser.add_argument("--test-seed", type=int)
    parser.add_argument("--train-episodes", type=int, default=20_000)
    parser.add_argument("--test-episodes", type=int, default=20_000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with args.config.open("rb") as handle:
        config = tomllib.load(handle)
    root_seed = int(config["run"]["root_seed"])
    train_seed = args.train_seed if args.train_seed is not None else root_seed + 10_000_000
    test_seed = args.test_seed if args.test_seed is not None else root_seed + 20_000_000
    assert_dependency_free_imports()
    report = diagnose(
        config["world"],
        train_seed,
        args.train_episodes,
        test_seed,
        args.test_episodes,
    )
    document = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document + "\n", encoding="utf-8")
    print(document)


if __name__ == "__main__":
    main()
