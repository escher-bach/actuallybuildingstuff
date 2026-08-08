"""The dial sweep and, mostly, the reading of it.

Task Spec section 8 step 5 is "the only step that can end the programme rather
than redirect it", so the thing under test here is not really the runner -- it is
whether a verdict can be manufactured that the data does not support.  Every
shape in the spec's table gets a synthetic curve, and so do the shapes that must
NOT produce a verdict.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from repertoire.harness.episode import EpisodeSpec
from repertoire.harness.metrics import Budget, BudgetMismatch
from repertoire.harness.model import ModelConfig
from repertoire.harness.stub import StubLookupFamily
from repertoire.harness.sweep import (
    Dial,
    SweepPoint,
    SweepResult,
    compare_dials,
    observation_dial,
    preamble_dial,
    read_curve,
    run_sweep,
)


def _budget(steps=20):
    cfg = ModelConfig(n_layer=1, n_head=2, d_model=32, d_ff=64, max_len=192)
    return Budget(steps=steps, batch_size=4, max_len=192, lr=2e-3, warmup=4,
                  model=cfg.as_dict()), cfg


class TestReadingTheCurve(unittest.TestCase):
    """Task Spec section 8 step 5's table, one synthetic curve per row."""

    def test_monotone_collapse_says_stop(self):
        curve = [(0.0, 1.0), (0.5, 0.7), (1.0, 0.4), (1.5, 0.2), (2.0, 0.05)]
        r = read_curve(curve)
        self.assertEqual(r.verdict, "monotone collapse")
        self.assertIn("STOP", r.action)
        self.assertIn("worth writing up", r.detail)
        self.assertTrue(r.confident)

    def test_interior_peak_says_relocate_the_cuts(self):
        curve = [(0.0, 0.2), (0.5, 0.9), (1.0, 0.95), (1.5, 0.5), (2.0, 0.15)]
        r = read_curve(curve)
        self.assertEqual(r.verdict, "interior peak")
        self.assertIn("relocate the cuts", r.action)
        self.assertTrue(r.confident)

    def test_flat_says_investigate(self):
        r = read_curve([(0.0, 0.5), (0.5, 0.5), (1.0, 0.51), (1.5, 0.49), (2.0, 0.5)])
        self.assertEqual(r.verdict, "flat")
        self.assertIn("investigate", r.action)

    def test_rising_says_investigate(self):
        r = read_curve([(0.0, 0.1), (0.5, 0.3), (1.0, 0.6), (1.5, 0.8), (2.0, 0.95)])
        self.assertEqual(r.verdict, "rising")
        self.assertIn("investigate", r.action)

    def test_a_peak_that_does_not_clear_the_ends_is_refused(self):
        """The shape noise produces. A bump one point above its neighbours is not
        an interior peak, and calling it one is how a null result becomes a
        finding."""
        curve = [(0.0, 0.50), (0.5, 0.52), (1.0, 0.58), (1.5, 0.55), (2.0, 0.40)]
        r = read_curve(curve)
        self.assertNotEqual(r.verdict, "interior peak")
        self.assertFalse(r.confident)

    def test_too_few_points_is_refused_rather_than_guessed(self):
        r = read_curve([(0.0, 1.0), (1.0, 0.5), (2.0, 0.1)])
        self.assertEqual(r.verdict, "insufficient")
        self.assertFalse(r.confident)

    def test_the_thresholds_are_fixed_in_the_module_not_passed_in(self):
        """Same discipline as expectations.py: the failure mode is not
        forgetting the prediction, it is reading the curve and finding the
        reading that fits. A threshold that is an argument is a threshold that
        gets tuned after the numbers arrive."""
        import inspect

        from repertoire.harness import sweep

        sig = inspect.signature(sweep.read_curve)
        self.assertEqual(list(sig.parameters), ["curve", "noise"])
        self.assertIsInstance(sweep.INTERIOR_MARGIN_ABS, float)
        self.assertIsInstance(sweep.INTERIOR_SIGMA, float)
        self.assertIsInstance(sweep.FLATNESS, float)


class TestFitToRead(unittest.TestCase):
    """The guard against reading an underpowered sweep."""

    def _result(self, **overrides):
        budget, _ = _budget(steps=100)
        base = dict(rule_entropy=0.5, notation_upper=0.1, structural_content=100.0,
                    l_final=0.6, excess_over_floor=0.1, converged=True,
                    acq_slope=-0.2, transfer=5.0, transfer_baseline=10.0)
        base.update(overrides)
        pts = [SweepPoint(dial="d", setting=float(i), seed=0, **base) for i in range(5)]
        return SweepResult(family="f", k=0, dial="d", budget=budget, points=pts)

    def test_a_healthy_sweep_is_readable(self):
        fit, why = self._result().fit_to_read()
        self.assertTrue(fit, why)

    def test_a_model_still_at_chance_is_refused(self):
        """The failure the smoke run produced: 60 steps, loss unmoved from
        chance, and a confident 'rising' verdict off pure sampling noise."""
        fit, why = self._result(l_final=4.2, excess_over_floor=3.7,
                                structural_content=10.0).fit_to_read()
        self.assertFalse(fit)
        self.assertIn("ACHIEVABLE floor", why)

    def test_learning_is_scored_against_the_achievable_floor_not_the_lower_one(self):
        """The first real T4 sweep found this.

        The lower end of the floor band conditions on the encoding, which the
        model never observes. Scoring "did it learn" against it asks the model to
        close a distance no predictor can close, and would refuse a converged run
        for missing a target that is physically out of reach.

        Here: a model sitting exactly ON the achievable floor (rule 0.5 +
        notation 1.2) must read as fully learned, even though it is 1.2 nats
        above the lower end.
        """
        at_optimal = self._result(rule_entropy=0.5, notation_upper=1.2, l_final=1.7,
                                  excess_over_floor=1.2, structural_content=100.0)
        fit, why = at_optimal.fit_to_read()
        self.assertTrue(fit, why)

    def test_mostly_unconverged_runs_are_refused(self):
        fit, why = self._result(converged=False).fit_to_read()
        self.assertFalse(fit)
        self.assertIn("falling tail", why)

    def test_an_empty_sweep_is_refused(self):
        budget, _ = _budget()
        fit, _ = SweepResult(family="f", k=0, dial="d", budget=budget).fit_to_read()
        self.assertFalse(fit)


class TestDials(unittest.TestCase):
    def test_the_two_dials_score_the_same_number_of_trials(self):
        """`compare_dials` only says anything if both sweeps score the same
        tokens per episode against the same fixed transfer target."""
        a = observation_dial(max_free=4, T=6)
        b = preamble_dial(points=5, T=6)
        self.assertEqual(a.T, b.T)
        self.assertEqual(a.transfer_target, b.transfer_target)
        for d in (a, b):
            for s in d.settings:
                self.assertEqual(d.spec(s).T, 6)

    def test_the_observation_dial_holds_supervised_tokens_constant(self):
        """The reason it is the generic dial: n_free changes the context and not
        the number of targets, so the comparison across settings is exact."""
        from repertoire.harness.episode import build_episode

        fam = StubLookupFamily(d=6)
        dial = observation_dial(max_free=4, T=4)
        counts = {
            s: build_episode(fam, 1, 0, dial.spec(s)).n_supervised for s in dial.settings
        }
        self.assertEqual(set(counts.values()), {4})

    def test_the_observation_dial_asks_nothing_of_the_family(self):
        """It must work on a family with no partial_preamble -- that is the whole
        point of having it alongside the preamble dial."""
        from repertoire.families import SHJTypeIFamily
        from repertoire.harness.episode import build_episode

        dial = observation_dial(max_free=3, T=3)
        for s in dial.settings:
            ep = build_episode(SHJTypeIFamily(), 1, 0, dial.spec(s))
            self.assertEqual(ep.n_supervised, 3)

    def test_more_free_observations_lowers_residual_entropy(self):
        """The dial has to move the x-axis, or the curve is plotted against
        something that is not being varied."""
        from repertoire.harness.entropy import measure_residual_entropy

        fam = StubLookupFamily(d=6)
        dial = observation_dial(max_free=4, T=3)
        hs = [measure_residual_entropy(fam, 1, dial.spec(s), n_episodes=24).rule
              for s in dial.settings]
        self.assertGreater(hs[0], hs[-1])
        self.assertEqual(hs, sorted(hs, reverse=True))


class TestRunner(unittest.TestCase):
    def test_a_sweep_runs_and_is_resumable(self):
        fam = StubLookupFamily(d=5)
        budget, cfg = _budget(steps=8)
        dial = Dial(name="tiny", settings=(0.0, 1.0), T=3,
                    build=lambda s, t: EpisodeSpec(T=t, n_free=int(s)))
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "s"
            first = run_sweep(fam, 1, dial, budget, cfg, seeds=(0,),
                              entropy_episodes=6, out_dir=out, device="cpu",
                              verbose=False)
            self.assertEqual(len(first.points), 2)
            self.assertTrue((out / "sweep.json").exists())

            # Second call must reuse the cache rather than retrain.
            import time

            t0 = time.time()
            second = run_sweep(fam, 1, dial, budget, cfg, seeds=(0,),
                               entropy_episodes=6, out_dir=out, device="cpu",
                               verbose=False)
            self.assertLess(time.time() - t0, 15)
            self.assertEqual(
                [p.structural_content for p in first.points],
                [p.structural_content for p in second.points],
            )

    def test_a_stale_cache_at_a_different_budget_is_discarded(self):
        """Section 4: comparing across budgets is meaningless, and a stale cache
        in a reused output directory is the quietest way to end up doing it."""
        fam = StubLookupFamily(d=5)
        budget_a, cfg = _budget(steps=8)
        budget_b, _ = _budget(steps=12)
        dial = Dial(name="tiny", settings=(0.0,), T=3,
                    build=lambda s, t: EpisodeSpec(T=t, n_free=int(s)))
        with TemporaryDirectory() as tmp:
            out = Path(tmp) / "s"
            run_sweep(fam, 1, dial, budget_a, cfg, seeds=(0,), entropy_episodes=6,
                      out_dir=out, device="cpu", verbose=False)
            again = run_sweep(fam, 1, dial, budget_b, cfg, seeds=(0,),
                              entropy_episodes=6, out_dir=out, device="cpu",
                              verbose=False)
            self.assertEqual(again.budget.fingerprint, budget_b.fingerprint)
            self.assertEqual(len(again.points), 1)

    def test_comparing_dials_across_budgets_is_refused(self):
        a_budget, _ = _budget(steps=10)
        b_budget, _ = _budget(steps=20)
        a = SweepResult(family="f", k=0, dial="a", budget=a_budget)
        b = SweepResult(family="f", k=0, dial="b", budget=b_budget)
        with self.assertRaises(BudgetMismatch):
            compare_dials(a, b)

    def test_disagreement_between_dials_is_stated_not_averaged(self):
        budget, _ = _budget()
        def mk(name, ys):
            pts = [SweepPoint(dial=name, setting=float(i), seed=0, rule_entropy=float(i),
                              notation_upper=0.0, structural_content=1.0, l_final=0.5,
                              excess_over_floor=0.05, converged=True, acq_slope=-0.1,
                              transfer=(1 - y) * 10.0, transfer_baseline=10.0)
                   for i, y in enumerate(ys)]
            return SweepResult(family="f", k=0, dial=name, budget=budget, points=pts)

        collapse = mk("a", [1.0, 0.7, 0.4, 0.2, 0.05])
        peak = mk("b", [0.2, 0.9, 0.95, 0.5, 0.15])
        out = compare_dials(collapse, peak)
        self.assertIn("DISAGREE", out)
        self.assertIn("about the parametrization", out)

        agree = compare_dials(collapse, mk("c", [1.0, 0.6, 0.3, 0.15, 0.02]))
        self.assertIn("agree", agree)


if __name__ == "__main__":
    unittest.main()
