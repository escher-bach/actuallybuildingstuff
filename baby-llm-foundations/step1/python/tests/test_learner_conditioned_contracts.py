"""Contracts for the learner-conditioned stage and its two configurations."""
from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from step1_experiments.evaluate import _aggregate_rows
from step1_experiments.learner_conditioned import (
    EXTENDED_METRIC_NAMES,
    LEARNER_CONDITIONED_CONTRACT,
    _aggregate_extended,
    assert_report_contract,
    dagger_plan,
    round_seeds,
)


CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "kaggle"
COLD = CONFIGS / "t4x2_learner_conditioned_cold_seed0.toml"
WARM = CONFIGS / "t4x2_learner_conditioned_warm_seed0.toml"


def _load(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


class PlanArithmetic(unittest.TestCase):
    def setUp(self) -> None:
        self.config = _load(WARM)

    def test_the_declared_budget_resolves_exactly(self) -> None:
        plan = dagger_plan(self.config, world_size=2)
        self.assertEqual(plan.sequences_per_update, 16)
        self.assertEqual(plan.max_steps, 512)
        self.assertEqual(plan.collection_episode_budget, 8192)
        self.assertEqual(plan.milestones_updates, (128, 256, 384, 512))

    def test_an_undeclared_arm_is_rejected(self) -> None:
        broken = {**self.config, "run": {**self.config["run"], "arm": "teacher_conditioned"}}
        with self.assertRaisesRegex(ValueError, "run.arm must be one of"):
            dagger_plan(broken, world_size=2)

    def test_an_episode_count_that_cannot_be_split_across_ranks_is_rejected(self) -> None:
        broken = {**self.config, "collection": {**self.config["collection"], "episodes_per_round": 2049}}
        with self.assertRaisesRegex(ValueError, "must divide across ranks"):
            dagger_plan(broken, world_size=2)


class SeedBlocks(unittest.TestCase):
    """A training world that is also an evaluation world would be undetectable."""

    def setUp(self) -> None:
        self.plan = dagger_plan(_load(WARM), world_size=2)

    def test_rounds_consume_disjoint_world_streams(self) -> None:
        blocks = [set(round_seeds(self.plan, index)) for index in range(self.plan.rounds)]
        union = set().union(*blocks)
        self.assertEqual(len(union), self.plan.collection_episode_budget)

    def test_training_worlds_never_overlap_the_evaluated_ones(self) -> None:
        world = _load(WARM)["world"]
        root = self.plan.root_seed
        evaluated = set()
        for offset, count in ((1_000_000, world["validation_episodes"]),
                              (2_000_000, world["structural_episodes"]),
                              (3_000_000, world["transfer_episodes"])):
            evaluated.update(range(root + offset, root + offset + count))
        collected = set().union(*(set(round_seeds(self.plan, i)) for i in range(self.plan.rounds)))
        self.assertEqual(collected & evaluated, set())

    def test_the_recovery_probe_deliberately_uses_the_held_out_block(self) -> None:
        """It is an evaluation, so it must sit on evaluated seeds and be untrained."""
        self.assertEqual(self.plan.recovery_probe_seed, self.plan.root_seed + 1_000_000)
        collected = set().union(*(set(round_seeds(self.plan, i)) for i in range(self.plan.rounds)))
        probe = set(range(self.plan.recovery_probe_seed,
                          self.plan.recovery_probe_seed + self.plan.recovery_probe_episodes))
        self.assertEqual(collected & probe, set())


def _row(success: bool, malformed: int = 0, invalid: int = 0, probes: int = 2,
         committed: bool = True, licensed: bool | None = True, live: int | None = 1) -> dict:
    return {
        "success": success, "spent": 3, "malformed": malformed, "invalid": invalid,
        "steps": probes + int(committed), "teacher_spent": 3, "probes": probes,
        "committed": committed,
        "licensed_at_commitment": licensed if committed else None,
        "live_hypotheses_at_commitment": live if committed else None,
    }


class ExtendedMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.rows = [
            _row(True), _row(True),
            _row(False, licensed=False, live=3),
            _row(False, malformed=1, committed=False),
            _row(False, invalid=1, committed=False),
        ]

    def test_the_field_set_is_exactly_what_the_contract_requires(self) -> None:
        self.assertEqual(set(_aggregate_extended(self.rows)), set(EXTENDED_METRIC_NAMES))

    def test_legal_conditioned_success_removes_the_interface_term(self) -> None:
        extended = _aggregate_extended(self.rows)
        # Three episodes played legally, two of them succeeded.
        self.assertAlmostEqual(extended["success_conditioned_on_legal_play"], 2 / 3)
        self.assertAlmostEqual(extended["protocol_failure_rate"], 2 / 5)

    def test_it_agrees_with_the_frozen_aggregate_it_is_derived_from(self) -> None:
        """Malformed and invalid are disjoint per episode, so the frozen rates
        determine legal-conditioned success exactly. This is what lets every
        arm already run be restated without re-scoring it."""
        frozen = _aggregate_rows(self.rows)
        legal_share = 1 - frozen["malformed_action_rate"] - frozen["invalid_action_rate"]
        self.assertAlmostEqual(
            _aggregate_extended(self.rows)["success_conditioned_on_legal_play"],
            frozen["success_rate"] / legal_share,
        )

    def test_premature_commitment_is_counted_over_commitments_not_episodes(self) -> None:
        extended = _aggregate_extended(self.rows)
        # Three episodes committed; one of them with three hypotheses still live.
        self.assertAlmostEqual(extended["commitment_rate"], 3 / 5)
        self.assertAlmostEqual(extended["premature_commitment_rate"], 1 / 3)
        self.assertAlmostEqual(extended["mean_live_hypotheses_at_commitment"], 5 / 3)

    def test_a_policy_that_never_commits_reports_undefined_not_zero(self) -> None:
        rows = [_row(False, malformed=1, committed=False)]
        extended = _aggregate_extended(rows)
        self.assertIsNone(extended["premature_commitment_rate"])
        self.assertIsNone(extended["success_conditioned_on_legal_play"])


class ReportContract(unittest.TestCase):
    def setUp(self) -> None:
        self.config = {**_load(WARM), "_meta": {"hash": "0" * 64}}
        self.plan = dagger_plan(self.config, world_size=2)
        evaluation = {
            "comparison": {"label": "x", "seed_policy": "y"},
            "metrics": dict.fromkeys(
                ("success_rate", "failure_rate", "malformed_action_rate", "invalid_action_rate",
                 "mean_spent", "mean_success_excess_cost", "mean_steps"), 0.0),
            "extended": dict.fromkeys(EXTENDED_METRIC_NAMES, 0.0),
        }
        self.report = {
            "contract": LEARNER_CONDITIONED_CONTRACT,
            "experiment_config_sha256": "0" * 64,
            "algorithm": {"objective": "supervised_cross_entropy_on_teacher_corrections",
                          "policy_gradient": False, "state_distribution": "learner_conditioned"},
            "plan": self.plan.report(),
            "budget_accounting": {
                "optimizer_updates": 512, "collection_episodes": 8192, "world_transitions": 1,
                "generated_action_tokens": 1, "supervised_correction_tokens": 1,
                "wall_clock_seconds": 1.0, "states_refused_unlicensed_commit": 0,
            },
            "milestones": [
                {"budget_updates": updates, "serialization": {"exact": True},
                 "evaluation": ({"validation": evaluation, "structural": evaluation,
                                 "rendering_b": evaluation, "reversible_control": evaluation}
                                if updates == 512 else {"validation": evaluation}),
                 "recovery_probe": {"episodes": 256}}
                for updates in (0, 128, 256, 384, 512)
            ],
            "rounds": [{} for _ in range(4)],
        }

    def test_a_complete_report_passes(self) -> None:
        assert_report_contract(self.report, self.config, self.plan)

    def test_a_stage_claiming_to_be_policy_gradient_fails(self) -> None:
        broken = {**self.report, "algorithm": {**self.report["algorithm"], "policy_gradient": True}}
        with self.assertRaisesRegex(AssertionError, "must not report itself as policy-gradient"):
            assert_report_contract(broken, self.config, self.plan)

    def test_a_report_without_the_refusal_count_fails(self) -> None:
        budget = {key: value for key, value in self.report["budget_accounting"].items()
                  if key != "states_refused_unlicensed_commit"}
        with self.assertRaisesRegex(AssertionError, "guard ran"):
            assert_report_contract({**self.report, "budget_accounting": budget}, self.config, self.plan)

    def test_a_report_missing_the_starting_policy_fails(self) -> None:
        broken = {**self.report, "milestones": self.report["milestones"][1:]}
        with self.assertRaisesRegex(AssertionError, "milestone grid mismatch"):
            assert_report_contract(broken, self.config, self.plan)

    def test_a_milestone_without_the_extended_read_fails(self) -> None:
        milestones = [dict(point) for point in self.report["milestones"]]
        milestones[1]["evaluation"] = {"validation": {
            **milestones[1]["evaluation"]["validation"], "extended": {"mean_probes": 0.0}}}
        with self.assertRaisesRegex(AssertionError, "extended fields mismatch"):
            assert_report_contract({**self.report, "milestones": milestones}, self.config, self.plan)

    def test_a_short_collection_budget_fails(self) -> None:
        budget = {**self.report["budget_accounting"], "collection_episodes": 4096}
        with self.assertRaisesRegex(AssertionError, "against a declared"):
            assert_report_contract({**self.report, "budget_accounting": budget}, self.config, self.plan)


class ArmsDifferOnlyWhereDeclared(unittest.TestCase):
    """The RUNBOOK's control discipline: name every field that differs.

    These two arms answer different questions, so unlike a one-field control
    they legitimately differ in more than one place. The point of the test is
    that the list is exhaustive and reviewed, not that it is short.
    """

    DECLARED = {("run", "name"), ("run", "arm"), ("training", "learning_rate")}

    def test_only_the_declared_fields_and_the_source_block_differ(self) -> None:
        cold, warm = _load(COLD), _load(WARM)
        self.assertNotIn("source", cold)
        self.assertIn("source", warm)
        differences = set()
        for section in (set(cold) | set(warm)) - {"source"}:
            for key in set(cold.get(section, {})) | set(warm.get(section, {})):
                if cold.get(section, {}).get(key) != warm.get(section, {}).get(key):
                    differences.add((section, key))
        self.assertEqual(differences, self.DECLARED)

    def test_both_arms_share_the_world_and_the_collection_budget(self) -> None:
        cold, warm = _load(COLD), _load(WARM)
        self.assertEqual(cold["world"], warm["world"])
        self.assertEqual(cold["collection"], warm["collection"])
        self.assertEqual(dagger_plan(cold, 2).collection_episode_budget,
                         dagger_plan(warm, 2).collection_episode_budget)


if __name__ == "__main__":
    unittest.main()
