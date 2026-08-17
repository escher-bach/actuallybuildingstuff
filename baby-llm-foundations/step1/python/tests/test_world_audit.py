"""Contracts for the CPU world audit.

Deliberately free of torch so the audit stays runnable off a GPU host: the
whole point of THEORY-PHASE.md §9 is that these numbers cost nothing.
"""
from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

from step1_experiments.world_audit import (
    AUDIT_CONTRACT,
    assert_audit_contract,
    audit,
    identification_ceiling,
    opening_liveness,
    teacher_play,
    teacher_target_leakage,
    truth_blind_ceiling,
)


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "kaggle" / "t4x2_dense_seed0.toml"


def _world() -> dict:
    with CONFIG.open("rb") as handle:
        return tomllib.load(handle)["world"]


class IdentificationBound(unittest.TestCase):
    def test_bound_is_two_to_the_k_capped_at_the_hypothesis_count(self) -> None:
        self.assertAlmostEqual(identification_ceiling(6, 0), 1 / 6)
        self.assertAlmostEqual(identification_ceiling(6, 1), 2 / 6)
        self.assertAlmostEqual(identification_ceiling(6, 2), 4 / 6)
        self.assertEqual(identification_ceiling(6, 3), 1.0)
        self.assertEqual(identification_ceiling(6, 4), 1.0)

    def test_negative_probe_count_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot be negative"):
            identification_ceiling(6, -1)


class OpeningLiveness(unittest.TestCase):
    def test_the_first_observation_rules_nothing_out(self) -> None:
        world = _world()
        result = opening_liveness(world, seed=20260811 + 1_000_000, episodes=64)
        self.assertTrue(result["equals_n_hyp_in_every_episode"])
        self.assertEqual(result["mean_live_hypotheses"], float(world["n_hyp"]))
        self.assertEqual(result["histogram"], {str(world["n_hyp"]): 64})


class TeacherPlay(unittest.TestCase):
    def test_the_demonstrator_solves_every_episode_without_guessing(self) -> None:
        result = teacher_play(_world(), seed=20260811 + 1_000_000, episodes=64)
        self.assertEqual(result["success_rate"], 1.0)
        self.assertTrue(result["never_commits_prematurely"])
        self.assertEqual(result["max_live_hypotheses_at_commitment"], 1)
        self.assertGreater(result["mean_probes"], 0.0)


class TruthBlindCeiling(unittest.TestCase):
    def test_every_episode_is_identifiable_within_budget(self) -> None:
        result = truth_blind_ceiling(_world(), seed=20260811 + 1_000_000, episodes=32)
        self.assertEqual(result["identifiable_rate"], 1.0)
        self.assertEqual(result["success_ceiling"], 1.0)


class TeacherTargetLeakage(unittest.TestCase):
    def test_the_probe_actually_reaches_states_the_teacher_never_visits(self) -> None:
        result = teacher_target_leakage(_world(), seed=20260811 + 1_000_000, episodes=64)
        self.assertGreater(result["unlicensed_states"], 0)
        self.assertLessEqual(result["commit_was_the_only_preferred_action"],
                             result["commit_proposed_at_unlicensed_state"])

    def test_the_leak_this_arm_must_not_supervise_is_present(self) -> None:
        """If this ever reads zero the collector's guard is untested, not unnecessary."""
        result = teacher_target_leakage(_world(), seed=20260811 + 1_000_000, episodes=64)
        self.assertGreater(result["commit_proposed_at_unlicensed_state"], 0)


class AuditContract(unittest.TestCase):
    def setUp(self) -> None:
        self.world = _world()
        # A cheap validation split; the contract does not depend on its size.
        self.report = audit({"run": {"root_seed": 20260811},
                             "world": {**self.world, "validation_episodes": 32}}, ceiling_episodes=16)

    def test_a_complete_audit_passes(self) -> None:
        self.assertEqual(self.report["contract"], AUDIT_CONTRACT)
        assert_audit_contract(self.report, self.world)

    def test_a_premature_teacher_commitment_fails_the_contract(self) -> None:
        broken = {**self.report, "teacher_play": {**self.report["teacher_play"],
                                                  "never_commits_prematurely": False}}
        with self.assertRaisesRegex(AssertionError, "not measuring prematurity"):
            assert_audit_contract(broken, self.world)

    def test_a_teacher_that_loses_fails_the_contract(self) -> None:
        broken = {**self.report, "teacher_play": {**self.report["teacher_play"], "success_rate": 0.99}}
        with self.assertRaisesRegex(AssertionError, "must solve every episode"):
            assert_audit_contract(broken, self.world)

    def test_a_teacher_that_never_probes_fails_the_contract(self) -> None:
        broken = {**self.report, "teacher_play": {**self.report["teacher_play"], "mean_probes": 0.0}}
        with self.assertRaisesRegex(AssertionError, "never probes"):
            assert_audit_contract(broken, self.world)

    def test_a_leakage_probe_that_reached_no_unlicensed_state_fails_the_contract(self) -> None:
        broken = {**self.report,
                  "teacher_target_leakage": {**self.report["teacher_target_leakage"], "unlicensed_states": 0}}
        with self.assertRaisesRegex(AssertionError, "checked nothing"):
            assert_audit_contract(broken, self.world)

    def test_an_impossible_ceiling_fails_the_contract(self) -> None:
        broken = {**self.report,
                  "truth_blind_ceiling": {**self.report["truth_blind_ceiling"], "success_ceiling": 1.5}}
        with self.assertRaisesRegex(AssertionError, "not a probability"):
            assert_audit_contract(broken, self.world)


if __name__ == "__main__":
    unittest.main()
