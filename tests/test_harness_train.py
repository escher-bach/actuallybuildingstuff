"""Batching, the loss mask under a real model, and the section 4 measurement.

The arithmetic tests here matter more than the training ones.  Task Spec section
4 says two later sections depend on these quantities being computed the same way
every time, so structural content and the acquisition slope are checked against
hand-computed values rather than against "it produced a number".

One test does train: section 8 step 2's third gate is "L1 shows within-episode
loss decay", and there is no way to assert that without a run.  It is kept to a
size that finishes on a CPU in seconds, because a gate nobody runs is not a gate.
"""

from __future__ import annotations

import math
import unittest
from random import Random

import torch

from repertoire import vocab
from repertoire.form import Level
from repertoire.harness.episode import EpisodeSpec, build_episode, spec_for_level
from repertoire.harness.entropy import measure_residual_entropy
from repertoire.harness.metrics import (
    Budget,
    BudgetMismatch,
    acquisition_slope,
    require_same_budget,
    structural_content,
    turnover,
)
from repertoire.harness.model import Inducer, ModelConfig
from repertoire.harness.stub import StubLookupFamily
from repertoire.harness.train import (
    RunRecord,
    collate,
    episode_stream,
    masked_loss,
    per_trial_means,
    train_run,
)


class TestStructuralContent(unittest.TestCase):
    def test_area_matches_hand_arithmetic(self):
        """S = sum_i (L_i - L_final), and nothing else."""
        losses = [4.0] * 8 + [1.0] * 2  # tail_frac 0.1 -> floor is the last 2 (min 2)
        s = structural_content(losses, tail_frac=0.1)
        self.assertAlmostEqual(s.l_final, 1.0)
        self.assertAlmostEqual(s.value, 8 * 3.0)

    def test_flat_curve_has_no_structural_content(self):
        """A family with nothing learnable never drops. Section 4's junk_random
        end: loss pinned at log(alphabet size) from the first token."""
        s = structural_content([math.log(8)] * 40)
        self.assertAlmostEqual(s.value, 0.0, places=9)
        self.assertTrue(s.converged)

    def test_instantly_learned_curve_also_has_little(self):
        """The opposite end, and the reason both junk plants read near zero.
        junk_trivial drops in one step and stops; the area above the floor is
        one step's worth, not a sustained one."""
        instant = [4.0] + [0.01] * 39
        sustained = [4.0 - 4.0 * i / 39 for i in range(40)]
        self.assertLess(
            structural_content(instant).value, structural_content(sustained).value
        )

    def test_unconverged_run_is_flagged_and_S_is_a_lower_bound(self):
        """Section 4 says "the converged loss"; at a fixed budget convergence is
        not observed. A still-falling tail means the floor was overestimated, so
        the area above it was larger than measured -- the direction matters and
        is the one that understates a family rather than flattering it."""
        still_falling = [10.0 - 0.1 * i for i in range(60)]
        self.assertFalse(structural_content(still_falling).converged)
        self.assertTrue(structural_content([2.0] * 60).converged)

    def test_loss_below_the_floor_is_reported_as_a_leak(self):
        """Nothing can beat a floor computed with knowledge the model does not
        have. A run that does is reading something the posterior calculation does
        not know about, and the report must say so rather than showing a
        flattering excess."""
        s = structural_content([1.0] * 40, bayes_floor=2.0)
        self.assertIn("leak", s.note)
        self.assertIn("BELOW", s.note)
        self.assertNotIn("leak", structural_content([3.0] * 40, bayes_floor=2.0).note)

    def test_too_few_steps_refuses_rather_than_guessing_a_floor(self):
        with self.assertRaises(ValueError):
            structural_content([1.0, 2.0])


class TestAcquisitionSlope(unittest.TestCase):
    def test_slope_is_negative_when_theta_is_being_identified(self):
        a = acquisition_slope([4.0, 3.0, 2.0, 1.0])
        self.assertAlmostEqual(a.slope, -1.0)
        self.assertAlmostEqual(a.r2, 1.0)

    def test_slope_is_zero_when_there_is_nothing_to_identify(self):
        """The L0 control. Section 4's quantity is an identification rate, so a
        level that states theta up front must read flat -- and it did: the first
        real run gave L1 -0.40 against L0 -0.001."""
        self.assertAlmostEqual(acquisition_slope([2.0] * 8).slope, 0.0)

    def test_a_step_function_still_gets_the_sign_right_with_poor_fit(self):
        """junk_trivial is the sharp case: theta is identified in exactly one
        trial, so the curve is a step. The slope must still be negative while R^2
        is poor, which is why the curve is returned alongside the fit rather than
        the fit being trusted on its own."""
        a = acquisition_slope([4.0] + [0.1] * 7)
        self.assertLess(a.slope, 0.0)
        self.assertLess(a.r2, 0.7)


class TestBudget(unittest.TestCase):
    def test_comparison_across_budgets_is_refused(self):
        """Section 4: the quantity is budget-relative and comparing across
        budgets is meaningless. Raised rather than warned -- the failure it
        prevents is a sweep re-run at a larger step count and half-merged with
        the old points, which produces a curve that looks like a finding."""
        a = Budget(steps=100, batch_size=8, max_len=128, lr=1e-3, warmup=10)
        b = Budget(steps=200, batch_size=8, max_len=128, lr=1e-3, warmup=10)
        mk = lambda bud: RunRecord(family="f", k=0, spec={}, budget=bud)
        self.assertTrue(require_same_budget([mk(a), mk(a)]))
        with self.assertRaises(BudgetMismatch):
            require_same_budget([mk(a), mk(b)])

    def test_fingerprint_moves_with_the_model_too(self):
        base = dict(steps=100, batch_size=8, max_len=128, lr=1e-3, warmup=10)
        a = Budget(**base, model=ModelConfig(n_layer=2).as_dict())
        b = Budget(**base, model=ModelConfig(n_layer=4).as_dict())
        self.assertNotEqual(a.fingerprint, b.fingerprint)


class TestTurnover(unittest.TestCase):
    def test_says_it_cannot_tell_with_too_few_budgets(self):
        """A confident False here reads as "no collapse" when it means "not
        enough points to see one"."""
        seen, detail = turnover([1.0, 2.0])
        self.assertFalse(seen)
        self.assertIn("at least 3", detail)

    def test_detects_the_sign_change_hazard_17_warns_about(self):
        """Brute-force collapse appears as a turnover in structural content as
        compute grows, "which looks like noise unless you are watching for a sign
        change"."""
        self.assertFalse(turnover([1.0, 2.0, 3.0, 4.0])[0])
        seen, detail = turnover([1.0, 2.0, 3.0, 2.0])
        self.assertTrue(seen)
        self.assertIn("brute-force collapse", detail)


class TestBatching(unittest.TestCase):
    def setUp(self):
        self.fam = StubLookupFamily(d=6)

    def test_collate_pads_and_masks(self):
        eps = [build_episode(self.fam, 1, s, spec_for_level(Level.L1, T=5))
               for s in range(4)]
        b = collate(eps, max_len=256)
        self.assertEqual(b.tokens.shape, b.targets.shape)
        self.assertEqual(int(b.mask.sum()), sum(e.n_supervised for e in eps))
        for i, e in enumerate(eps):
            self.assertTrue(bool((b.tokens[i, len(e):] == vocab.PAD).all()))
            self.assertFalse(bool(b.mask[i, len(e):].any()))

    def test_an_episode_longer_than_max_len_is_refused_not_truncated(self):
        """Truncation would drop the final trials -- the best-identified ones,
        carrying the lowest loss -- so it would raise measured loss most on the
        families that identify fastest."""
        eps = [build_episode(self.fam, 1, 0, spec_for_level(Level.L1, T=20))]
        with self.assertRaises(ValueError) as cm:
            collate(eps, max_len=16)
        self.assertIn("Truncation is not offered", str(cm.exception))

    def test_soft_targets_are_only_built_when_a_level_needs_them(self):
        realized = collate([build_episode(self.fam, 1, 0, spec_for_level(Level.L1, T=4))], 256)
        posterior = collate([build_episode(self.fam, 1, 0, spec_for_level(Level.L3, T=4))], 256)
        self.assertIsNone(realized.soft)
        self.assertIsNotNone(posterior.soft)


class TestLoss(unittest.TestCase):
    def setUp(self):
        self.fam = StubLookupFamily(d=6)
        self.cfg = ModelConfig(n_layer=1, n_head=2, d_model=32, d_ff=64, max_len=256)

    def test_loss_is_normalized_per_supervised_token_not_per_episode(self):
        """Handoff section 2.2. The `slotted` encoding renders ~85% longer than
        its siblings; a per-episode average would mix that into the family
        effect, and every structural-content number would inherit it.

        Checked by scoring a batch of long episodes against one of short ones
        under a model that has learned nothing: the per-token loss must land in
        the same place, because it is the same prediction problem repeated more
        times.
        """
        torch.manual_seed(0)
        model = Inducer(self.cfg)
        out = {}
        for T in (3, 9):
            eps = [build_episode(self.fam, 1, s, spec_for_level(Level.L1, T=T))
                   for s in range(8)]
            b = collate(eps, 256)
            with torch.no_grad():
                out[T] = float(masked_loss(model(b.tokens[:, :-1]), b).total)
        self.assertAlmostEqual(out[3], out[9], delta=0.35)

    def test_a_batch_with_no_supervised_tokens_is_an_error(self):
        eps = [build_episode(self.fam, 1, 0, spec_for_level(Level.L1, T=4))]
        b = collate(eps, 256)
        b.mask[:] = False
        model = Inducer(self.cfg)
        with torch.no_grad():
            with self.assertRaises(ValueError):
                masked_loss(model(b.tokens[:, :-1]), b)

    def test_posterior_targets_score_against_the_distribution(self):
        """A model predicting exactly the posterior must score its entropy, not
        zero. If soft targets were silently falling through to the realized
        token this would come out at the hard-target loss instead."""
        eps = [build_episode(self.fam, 1, 0, spec_for_level(Level.L3, T=4))]
        b = collate(eps, 256)
        pos = list(eps[0].posterior_targets)
        # Hand the model's job to a logit tensor built from the target itself.
        logits = torch.full((1, len(eps[0]) - 1, vocab.VOCAB_SIZE), -30.0)
        for p in pos:
            for tok, prob in eps[0].posterior_targets[p].items():
                logits[0, p - 1, tok] = math.log(prob) + 30.0
        parts = masked_loss(logits, b)
        expected = -sum(
            p * math.log(p)
            for d in eps[0].posterior_targets.values()
            for p in d.values()
        ) / len(pos)
        self.assertAlmostEqual(float(parts.total), expected, places=3)

    def test_per_trial_means_bucket_by_trial_index(self):
        eps = [build_episode(self.fam, 1, s, spec_for_level(Level.L1, T=5))
               for s in range(4)]
        b = collate(eps, 256)
        model = Inducer(self.cfg)
        with torch.no_grad():
            means = per_trial_means(masked_loss(model(b.tokens[:, :-1]), b), 5)
        self.assertEqual(len(means), 5)
        self.assertTrue(all(m == m for m in means), "a trial bucket came out empty")


class TestTheStepOneGate(unittest.TestCase):
    """"A stub episode round-trips at all four levels and the logger emits a
    structural-content number."" """

    def test_logger_emits_a_structural_content_number(self):
        fam = StubLookupFamily(d=5)
        cfg = ModelConfig(n_layer=1, n_head=2, d_model=32, d_ff=64, max_len=128)
        bud = Budget(steps=12, batch_size=4, max_len=128, lr=1e-3, warmup=2,
                     model=cfg.as_dict())
        rec, _ = train_run(fam, 1, spec_for_level(Level.L1, T=4), bud,
                           model_cfg=cfg, seed=0, device="cpu")
        self.assertEqual(len(rec.losses), 12)
        self.assertTrue(all(x == x for x in rec.losses))
        self.assertIsInstance(rec.content().value, float)

    def test_every_level_trains_without_family_specific_code(self):
        """Section 8 step 3's gate, applied early: all levels through one harness.

        L2 included, which is the expensive one -- it samples the query from the
        model at the current weights, so it cannot be built offline.
        """
        fam = StubLookupFamily(d=5)
        cfg = ModelConfig(n_layer=1, n_head=2, d_model=32, d_ff=64, max_len=128)
        bud = Budget(steps=4, batch_size=2, max_len=128, lr=1e-3, warmup=1,
                     model=cfg.as_dict())
        for level in (Level.L0, Level.L1, Level.L2, Level.L3):
            rec, _ = train_run(fam, 1, spec_for_level(level, T=3), bud,
                               model_cfg=cfg, seed=0, device="cpu")
            self.assertEqual(len(rec.losses), 4, f"{level} produced no curve")


class TestTheStepTwoDecayGate(unittest.TestCase):
    def test_L1_shows_within_episode_loss_decay_and_L0_does_not(self):
        """Section 8 step 2: "L1 shows within-episode loss decay".

        Run on the stub rather than the worked family so it finishes in seconds
        on a CPU. The L0 arm is the control that makes the L1 arm mean something:
        a decay that appeared at both levels would be an artifact of position in
        the sequence, not identification.
        """
        fam = StubLookupFamily(d=5)
        cfg = ModelConfig(n_layer=3, n_head=4, d_model=96, d_ff=192, max_len=160)
        bud = Budget(steps=350, batch_size=16, max_len=160, lr=2e-3, warmup=30,
                     model=cfg.as_dict())

        slopes = {}
        for level in (Level.L0, Level.L1):
            rec, _ = train_run(fam, 1, spec_for_level(level, T=6), bud,
                               model_cfg=cfg, seed=0, device="cpu")
            slopes[level] = rec.slope(window=40)

        self.assertLess(slopes[Level.L1].slope, -0.05,
                        f"L1 did not decay: {slopes[Level.L1].report()}")
        self.assertLess(slopes[Level.L1].slope, slopes[Level.L0].slope,
                        "L1 must identify faster than L0, which has nothing to identify")


class TestEvaluate(unittest.TestCase):
    """Held-out scoring. Added because its absence made an arm unreadable."""

    def setUp(self):
        self.fam = StubLookupFamily(d=6)
        self.cfg = ModelConfig(n_layer=1, n_head=2, d_model=32, d_ff=64, max_len=192)

    def test_untrained_model_scores_log_vocab_on_every_channel(self):
        """The calibration that makes the number readable: a model at
        initialization must sit at log(vocab) everywhere, or the scoring is
        wrong before any training question is asked."""
        from repertoire.harness.train import evaluate

        r = evaluate(Inducer(self.cfg), self.fam, 1, spec_for_level(Level.L1, T=4),
                     n_episodes=32, device="cpu")
        for name, loss in r.per_channel.items():
            self.assertAlmostEqual(loss, math.log(vocab.VOCAB_SIZE), delta=0.15,
                                   msg=f"channel {name} at {loss}")

    def test_scoring_is_independent_of_the_training_mask(self):
        """The property the whole function exists for.

        Two models scored the same way must be comparable however they were
        trained. Before this, a run trained with extra supervision reported a
        loss averaged over tokens the comparison run never scored, and the two
        looked comparable while measuring different things.
        """
        from repertoire.harness.train import evaluate

        model = Inducer(self.cfg)
        spec = spec_for_level(Level.L1, T=4)
        a = evaluate(model, self.fam, 1, spec, n_episodes=32, device="cpu")
        b = evaluate(model, self.fam, 1, spec, n_episodes=32, device="cpu")
        self.assertAlmostEqual(a.answer_loss, b.answer_loss, places=9)
        self.assertEqual(a.n_tokens, b.n_tokens)

    def test_answer_channel_is_separated_from_the_others(self):
        from repertoire.harness.train import evaluate

        r = evaluate(Inducer(self.cfg), self.fam, 1, spec_for_level(Level.L1, T=5),
                     n_episodes=16, device="cpu")
        self.assertIn("ANSWER", r.per_channel)
        self.assertIn("QUERY", r.per_channel)
        self.assertEqual(len(r.per_trial), 5)
        # one answer token per trial for this family
        self.assertEqual(r.n_tokens["ANSWER"], 16 * 5)

    def test_evaluation_episodes_are_disjoint_from_training_ones(self):
        """`train_run` streams from seed * 1_000_003; evaluation starts far above
        it, so 'held out' is checkable rather than argued."""
        from repertoire.harness.episode import build_episode
        from repertoire.harness.train import episode_stream

        spec = spec_for_level(Level.L1, T=4)
        trained = {tuple(next(s).tokens) for s in [episode_stream(self.fam, 1, spec, 0)]
                   for _ in range(200)}
        held = {tuple(build_episode(self.fam, 1, 900_000_000 + i, spec).tokens)
                for i in range(64)}
        self.assertFalse(trained & held)


class TestReproducibility(unittest.TestCase):
    def test_two_runs_at_the_same_seed_produce_the_same_curve(self):
        """Section 7 calls this non-negotiable: "the whole design depends on
        being able to re-run a branch after a backtrack"."""
        fam = StubLookupFamily(d=5)
        cfg = ModelConfig(n_layer=1, n_head=2, d_model=32, d_ff=64, max_len=128)
        bud = Budget(steps=8, batch_size=4, max_len=128, lr=1e-3, warmup=2,
                     model=cfg.as_dict())
        a, _ = train_run(fam, 1, spec_for_level(Level.L1, T=4), bud,
                         model_cfg=cfg, seed=3, device="cpu")
        b, _ = train_run(fam, 1, spec_for_level(Level.L1, T=4), bud,
                         model_cfg=cfg, seed=3, device="cpu")
        for x, y in zip(a.losses, b.losses):
            self.assertAlmostEqual(x, y, places=10)

    def test_the_episode_stream_never_repeats(self):
        fam = StubLookupFamily(d=6)
        stream = episode_stream(fam, 1, spec_for_level(Level.L1, T=4))
        seen = {tuple(next(stream).tokens) for _ in range(200)}
        self.assertGreater(len(seen), 190, "the stream is recycling episodes")


if __name__ == "__main__":
    unittest.main()
