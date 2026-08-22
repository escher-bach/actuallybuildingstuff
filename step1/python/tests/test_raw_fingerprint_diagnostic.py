from __future__ import annotations

import inspect
import unittest
from collections import Counter

from step1_experiments.raw_fingerprint_diagnostic import (
    DIAGNOSTIC_CONTRACT,
    FrozenFingerprintPolicy,
    TrainingRecord,
    assert_dependency_free_imports,
    budget_key,
    diagnose,
    fit_policy,
    reset_prefix,
)


def observation(commit_order: tuple[int, ...], budget: int = 5) -> str:
    actions = ["inspect(probe_1)"] + [f"commit(cause_{number})" for number in commit_order]
    return "\n".join(
        [
            f"BUDGET {budget}",
            "AVAILABLE " + ", ".join(actions),
            "STATUS running",
        ]
    )


class FakeBatch:
    """Small irreversible world exposing only APIs allowed by the audit."""

    n_episodes = 1

    def __init__(self, rendered: str, correct_action: str) -> None:
        self.rendered = rendered
        self.correct_action = correct_action
        self.action: str | None = None
        self.outcome_reads = 0

    def observations(self, rendering: str):
        if rendering != "a":
            raise AssertionError("test backend supports only Rendering A")
        return [self.rendered]

    def step(self, actions):
        if self.action is not None:
            raise AssertionError("terminal fake world was stepped twice")
        if len(actions) != 1:
            raise AssertionError("fake world is one episode")
        self.action = actions[0]

    def done(self):
        return [self.action is not None]

    def privileged_outcomes(self):
        if self.action is None:
            raise AssertionError("verifier was queried before termination")
        self.outcome_reads += 1
        return [(True, self.action == self.correct_action, 0, 1, False, False)]


class FakeFactory:
    def __init__(self, episodes: dict[int, tuple[str, str]]) -> None:
        self.episodes = episodes
        self.calls: list[tuple[int, int]] = []

    def __call__(self, _params: dict, seed: int, n_episodes: int):
        self.calls.append((seed, n_episodes))
        rendered, correct = self.episodes[seed]
        return FakeBatch(rendered, correct)


def identity_parser(action: str, _params: dict):
    return action


PARAMS = {
    "n_hyp": 2,
    "n_probe": 1,
    "n_evidence": 2,
    "cost_lo": 1,
    "cost_hi": 1,
    "budget_slack": 1,
    "min_depth": 1,
    "step_slack": 1,
    "variant": "irreversible",
    "rendering": "a",
}


class FrozenLookupContracts(unittest.TestCase):
    def test_policy_input_is_only_the_exact_prefix(self) -> None:
        self.assertEqual(list(inspect.signature(FrozenFingerprintPolicy.choose).parameters),
                         ["self", "prefix"])

    def test_raw_lookup_and_budget_baseline_are_distinct_and_immutable(self) -> None:
        first = reset_prefix(observation((2, 1), budget=5))
        second = reset_prefix(observation((1, 2), budget=5))
        records = [
            TrainingRecord(first, "commit(cause_2)"),
            TrainingRecord(first, "commit(cause_2)"),
            TrainingRecord(second, "commit(cause_1)"),
        ]
        policy = fit_policy(records)

        self.assertEqual(policy.choose(first), "commit(cause_2)")
        self.assertEqual(policy.choose(second), "commit(cause_1)")
        self.assertEqual(policy.choose_budget_only(first), "commit(cause_2)")
        unseen_same_budget = reset_prefix(observation((2, 1), budget=5).replace(
            "inspect(probe_1), ", ""
        ))
        self.assertEqual(policy.choose(unseen_same_budget), "commit(cause_2)")
        with self.assertRaises(TypeError):
            policy._raw_actions[first] = "commit(cause_1)"  # type: ignore[index]
        with self.assertRaises(AttributeError):
            policy.default_action = "commit(cause_1)"  # type: ignore[misc]

    def test_policy_hash_is_reproducible(self) -> None:
        prefix = reset_prefix(observation((1, 2)))
        records = [TrainingRecord(prefix, "commit(cause_1)")]
        self.assertEqual(fit_policy(records).policy_hash, fit_policy(records).policy_hash)


class HeldOutDiagnosticContracts(unittest.TestCase):
    def setUp(self) -> None:
        raw_a = observation((2, 1), budget=5)
        raw_b = observation((1, 2), budget=5)
        raw_unseen = observation((1, 2), budget=6)
        self.factory = FakeFactory(
            {
                100: (raw_a, "commit(cause_2)"),
                101: (raw_b, "commit(cause_1)"),
                200: (raw_a, "commit(cause_2)"),
                201: (raw_unseen, "commit(cause_1)"),
            }
        )

    def test_fit_is_frozen_before_disjoint_evaluator_matched_scoring(self) -> None:
        report = diagnose(
            PARAMS,
            train_seed=100,
            train_episodes=2,
            test_seed=200,
            test_episodes=2,
            batch_factory=self.factory,
            parse_action=identity_parser,
        )

        self.assertEqual(report["contract"], DIAGNOSTIC_CONTRACT)
        self.assertTrue(report["frozen_policy"]["frozen_before_test"])
        exact = report["held_out"]["exact_raw_key"]
        self.assertEqual(exact["coverage_count"], 1)
        self.assertEqual(exact["coverage_rate"], 0.5)
        self.assertEqual(exact["covered_accuracy"]["accuracy"], 1.0)
        self.assertEqual(exact["with_budget_fallback"]["accuracy"], 1.0)
        self.assertEqual(report["held_out"]["canonical_budget_only"]["accuracy"], 0.5)

        # Every construction is evaluator-matched, and held-out worlds are
        # used only for the selected raw/fallback and canonical actions—not
        # enumerated to recover their labels.
        self.assertTrue(all(n_episodes == 1 for _, n_episodes in self.factory.calls))
        self.assertTrue(all(seed in {100, 101, 200, 201} for seed, _ in self.factory.calls))
        test_calls = Counter(seed for seed, _ in self.factory.calls if seed >= 200)
        self.assertLessEqual(test_calls[200], 2)
        self.assertLessEqual(test_calls[201], 2)

    def test_overlapping_seed_ranges_are_rejected_before_world_access(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be disjoint"):
            diagnose(
                PARAMS,
                train_seed=100,
                train_episodes=3,
                test_seed=102,
                test_episodes=2,
                batch_factory=self.factory,
                parse_action=identity_parser,
            )
        self.assertEqual(self.factory.calls, [])

    def test_budget_key_contains_no_action_order(self) -> None:
        self.assertEqual(budget_key(observation((1, 2), 7)), "BUDGET 7")
        self.assertEqual(budget_key(observation((2, 1), 7)), "BUDGET 7")

    def test_module_has_no_torch_or_data_stack_import(self) -> None:
        assert_dependency_free_imports()


if __name__ == "__main__":
    unittest.main()
