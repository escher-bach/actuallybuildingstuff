"""Tests for the pre-committed matrix expectations.

These test the SCORER, not the matrix -- there is no matrix yet. What they
establish is that each expectation can both pass and fail on constructed inputs,
because an expectation that cannot fail is not a check. That is hazard 6 in
docs/07, met twice already in the A2 machinery, and this is the place to avoid
meeting it a third time on the gate that decides whether the programme proceeds.
"""

from __future__ import annotations

import unittest

from repertoire.expectations import EXPECTATIONS, Matrix, Outcome, score

FAMILIES = [
    "junk_random",
    "junk_trivial",
    "conjunction",
    "bruner_conjunction",
    "parity_identification",
    "shj_type_vi",
    "shj_type_i",
    "probability_matching",
]


def build(diag: dict[str, float], transfers: dict[tuple[str, str], float]) -> Matrix:
    """Build a matrix from desired TRANSFER fractions, inverting the definition.

    transfer(i, j) = 1 - s[i][j] / diag[j], so s[i][j] = diag[j] * (1 - t).
    """
    s = {i: {} for i in diag}
    for i in diag:
        for j in diag:
            t = transfers.get((i, j), 0.0)
            s[i][j] = diag[j] * (1.0 - t)
    return Matrix(s, diag)


def ideal() -> Matrix:
    diag = {f: 1.0 for f in FAMILIES}
    diag["junk_random"] = 0.01
    diag["junk_trivial"] = 0.01
    t = {}
    for a, b in (
        ("conjunction", "bruner_conjunction"),
        ("parity_identification", "shj_type_vi"),
    ):
        t[(a, b)] = 0.9
        t[(b, a)] = 0.9
    t[("shj_type_i", "shj_type_vi")] = 0.5
    t[("shj_type_vi", "shj_type_i")] = 0.2
    return build(diag, t)


class TestScorerOnIdealMatrix(unittest.TestCase):
    def test_ideal_matrix_passes_the_gate(self):
        gate, lines = score(ideal())
        self.assertTrue(gate, "\n".join(lines))

    def test_every_expectation_is_testable_on_the_ideal_matrix(self):
        m = ideal()
        for e in EXPECTATIONS:
            outcome, detail = e.check(m)
            self.assertIsNot(outcome, Outcome.UNTESTABLE, f"{e.id}: {detail}")


class TestEveryExpectationCanFail(unittest.TestCase):
    """Hazard 6: a check that cannot fail proves nothing about what passes."""

    def test_junk_reading_high_fails_the_gate(self):
        m = ideal()
        m.diag["junk_random"] = 0.9
        gate, _ = score(m)
        self.assertFalse(gate)

    def test_near_duplicates_not_clustering_fails_the_gate(self):
        diag = {f: 1.0 for f in FAMILIES}
        diag["junk_random"] = diag["junk_trivial"] = 0.01
        m = build(diag, {})  # no transfer anywhere
        gate, _ = score(m)
        self.assertFalse(gate)

    def test_prerequisite_with_the_wrong_sign_is_detected(self):
        diag = {f: 1.0 for f in FAMILIES}
        diag["junk_random"] = diag["junk_trivial"] = 0.01
        m = build(diag, {("shj_type_vi", "shj_type_i"): 0.9})  # reversed
        e = next(x for x in EXPECTATIONS if x.id == "shj-prerequisite-ordering")
        outcome, _ = e.check(m)
        self.assertIs(outcome, Outcome.FAIL)

    def test_independent_pair_that_transfers_fails(self):
        m = ideal()
        m.s["shj_type_i"]["probability_matching"] = 0.1  # near-total transfer
        e = next(x for x in EXPECTATIONS if x.id == "independent-pair")
        outcome, _ = e.check(m)
        self.assertIs(outcome, Outcome.FAIL)

    def test_junk_attributed_with_capability_fails(self):
        # The prior-art failure: a fitted model assigning high mastery to
        # subjects who scored zero. Junk must transfer to nothing.
        m = ideal()
        m.s["junk_random"]["conjunction"] = 0.2  # junk "explains" a real family
        e = next(x for x in EXPECTATIONS if x.id == "junk-attributes-nothing")
        outcome, detail = e.check(m)
        self.assertIs(outcome, Outcome.FAIL, detail)

    def test_structure_vs_paradigm_prediction_can_go_either_way(self):
        e = next(x for x in EXPECTATIONS if x.id == "P-structure-beats-paradigm")

        m = ideal()  # parity twin transfers 0.9, paradigm-mate 0.5
        self.assertIs(e.check(m)[0], Outcome.PASS)

        diag = {f: 1.0 for f in FAMILIES}
        diag["junk_random"] = diag["junk_trivial"] = 0.01
        rival = build(
            diag,
            {
                ("shj_type_i", "shj_type_vi"): 0.9,  # paradigm-mate wins
                ("parity_identification", "shj_type_vi"): 0.2,
            },
        )
        self.assertIs(e.check(rival)[0], Outcome.FAIL)


class TestGateSemantics(unittest.TestCase):
    def test_non_blocking_failure_does_not_stop_the_programme(self):
        # A prediction about the world is supposed to be able to fail. Only
        # instrument checks decide the gate.
        m = ideal()
        m.s["shj_type_i"]["shj_type_vi"] = 1.0
        m.s["shj_type_vi"]["shj_type_i"] = 1.0  # prerequisite signal gone
        gate, lines = score(m)
        self.assertTrue(gate, "\n".join(lines))

    def test_missing_families_are_untestable_not_passing(self):
        # An absent family must never be scored as a pass. Silently passing on
        # missing data is how a gate stops being a gate.
        m = Matrix({}, {})
        for e in EXPECTATIONS:
            self.assertIs(e.check(m)[0], Outcome.UNTESTABLE, e.id)

    def test_blocking_expectations_cover_the_instrument_checks(self):
        blocking = {e.id for e in EXPECTATIONS if e.blocking}
        self.assertIn("junk-near-zero", blocking)
        self.assertIn("constructed-near-duplicate", blocking)
        self.assertIn("independent-pair", blocking)
        self.assertIn("junk-attributes-nothing", blocking)


if __name__ == "__main__":
    unittest.main()
