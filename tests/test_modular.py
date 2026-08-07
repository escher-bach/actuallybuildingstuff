"""The worked family. Task Spec section 6, and section 8 step 2's gate.

Step 2 is done when:

    the A2 permuted-alphabet check passes as a unit test;
    L1 shows within-episode loss decay;
    L2 runs with a computable q*;
    L3 targets match a brute-force enumeration of consistent theta.

Three of the four are here.  The within-episode decay is a training result and
lives in `test_harness_train.py`, because asserting it requires a run.

Beyond the gate, the tests that matter most here are the ones asserting the
family does what it *says*, not that it runs.  Hazard 3 is a plant that stopped
being a plant while type-checking perfectly, and hazard 4 is a calibration
exemplar that silently degraded into L1 while remaining genuinely stochastic.
The generalization recorded there -- "every plant now has a test asserting the
plant property rather than that the code runs" -- applies to a worked example
as much as to a plant, so `test_oracle_really_is_modular_addition` asserts the
algebra rather than the interface.
"""

from __future__ import annotations

import math
import time
import unittest
from random import Random

from repertoire import vocab
from repertoire.families.modular import (
    Answer,
    LeakyModularFamily,
    ModularHiddenPermutationFamily,
    ModTheta,
    Pair,
    Step,
)
from repertoire.form import Level, TaskFamily
from repertoire.harness.entropy import (
    belief_state,
    brute_force_answer_distribution,
    measure_residual_entropy,
)
from repertoire.harness.episode import Channel, build_episode, build_reveal, spec_for_level
from repertoire.harness.protocol import (
    answer_distribution,
    check_query_sensitivity,
    draw_answer,
)


def small() -> ModularHiddenPermutationFamily:
    """pool 7, k=0 -> |Theta| = 7560. Enumerable, which every exact test needs."""
    return ModularHiddenPermutationFamily(pool_size=7)


class TestConformance(unittest.TestCase):
    def test_is_a_section_7_family(self):
        self.assertIsInstance(small(), TaskFamily)

    def test_all_four_levels_round_trip(self):
        from repertoire.harness.train import round_trip_all_levels

        for level, summary in round_trip_all_levels(small(), k=0, T=5).items():
            self.assertNotIn("error", summary, f"{level}: {summary.get('error')}")

    def test_content_tokens_come_from_the_symbol_alphabet(self):
        """Hazard 5. A2 passing does not mean the rendering is right."""
        fam = small()
        rng = Random(0)
        theta, enc = fam.sample_theta(0, rng), fam.sample_encoding(rng)
        self.assertTrue(set(enc.symbols) <= set(vocab.SYMBOL_IDS))
        q = fam.sample_query(theta, [], rng)
        a = fam.evaluate(theta, q)
        content = {enc.token(q.a), enc.token(q.b), enc.token(a.position)}
        self.assertTrue(content <= set(vocab.SYMBOL_IDS))


class TestTheAlgebra(unittest.TestCase):
    """What the family claims to be, asserted rather than assumed."""

    def test_oracle_really_is_modular_addition(self):
        fam = small()
        rng = Random(3)
        for _ in range(40):
            theta = fam.sample_theta(0, rng)
            inv = theta.inverse()
            for a in theta.pi:
                for b in theta.pi:
                    got = fam.evaluate(theta, Pair(a, b))
                    want = theta.pi[(inv[a] + inv[b]) % theta.m]
                    self.assertEqual(got.position, want)

    def test_the_oracle_is_commutative_and_has_an_identity(self):
        """Structure the model can exploit; if it were absent the family would be
        a random table wearing modular arithmetic's name."""
        fam = small()
        rng = Random(4)
        theta = fam.sample_theta(0, rng)
        zero = theta.pi[0]
        for a in theta.pi:
            self.assertEqual(fam.evaluate(theta, Pair(a, zero)).position, a)
            for b in theta.pi:
                self.assertEqual(
                    fam.evaluate(theta, Pair(a, b)).position,
                    fam.evaluate(theta, Pair(b, a)).position,
                )

    def test_A1_generation_cost_does_not_grow_with_k(self):
        """A1: instance cost O(1) in k. Sample theta, sample x, evaluate.

        Timed rather than argued. Crude -- wall clock on a laptop -- but the
        failure it guards against is not subtle: a generator that searches or
        rejection-samples (hazard 18) shows an order of magnitude, not 20%.
        """
        fam = ModularHiddenPermutationFamily(pool_size=24)
        times = []
        for k in (0, 4, 8):
            rng = Random(0)
            t0 = time.perf_counter()
            for _ in range(2000):
                theta = fam.sample_theta(k, rng)
                fam.evaluate(theta, fam.sample_query(theta, [], rng))
            times.append(time.perf_counter() - t0)
        self.assertLess(max(times), 6 * min(times), f"generation cost moved with k: {times}")


class TestA2(unittest.TestCase):
    def test_check_passes_and_the_leaky_family_fails_it(self):
        self.assertTrue(small().permuted_alphabet_check(Random(1)))
        self.assertFalse(LeakyModularFamily(pool_size=7).permuted_alphabet_check(Random(1)))

    def test_check_survives_the_encoding_gaining_a_field(self):
        """Hazard 6, second occurrence: adding a field to `Encoding` silently
        broke this same check, because it rebuilt the encoding field-by-field and
        the new field defaulted -- then failed an A2-*compliant* family. "A check
        that breaks when its subject gains a field is worse than no check,
        because its failure is indistinguishable from a finding."

        Tested by actually giving the encoding a field, rather than by reading
        the source for `replace(`. A grep passes whether or not the code works.
        """
        import dataclasses

        from repertoire.families import modular

        @dataclasses.dataclass(frozen=True)
        class WiderEncoding(modular.Encoding):
            flourish: int = vocab.STOI["COLON"]

        class WiderFamily(ModularHiddenPermutationFamily):
            name = "mod_arith_wider"

            def sample_encoding(self, rng):
                base = super().sample_encoding(rng)
                return WiderEncoding(base.name, base.symbols, base.operator)

            def render(self, encoding, obj):
                out = super().render(encoding, obj)
                if isinstance(obj, Pair):
                    out = out + [encoding.flourish]  # uses the new field
                return out

        fam = WiderFamily(pool_size=7)
        self.assertTrue(
            fam.permuted_alphabet_check(Random(1)),
            "the A2 check lost a field when it rebuilt the permuted encoding",
        )


class TestEncodings(unittest.TestCase):
    def test_all_three_encodings_are_reachable(self):
        fam = small()
        names = {fam.sample_encoding(Random(s)).name for s in range(60)}
        self.assertEqual(names, {"infix", "prefix", "tabular"})

    def test_encodings_differ_in_more_than_punctuation(self):
        """Hazard 7: every family's encodings once produced byte-identical
        episodes and differed only in which token was the separator. A3 was
        satisfied in letter and vacuous in fact."""
        fam = small()
        rng = Random(0)
        theta = fam.sample_theta(0, rng)
        q = Pair(theta.pi[0], theta.pi[1])
        lengths = set()
        for s in range(60):
            enc = fam.sample_encoding(Random(s))
            lengths.add(len(fam.render(enc, q)))
        self.assertGreater(len(lengths), 1, "encodings render to one length: A3 is vacuous")

    def test_parse_query_inverts_render_under_every_encoding(self):
        """rho_e^-1, which section 7 does not have and L2 cannot work without."""
        fam = small()
        rng = Random(9)
        theta = fam.sample_theta(0, rng)
        for s in range(30):
            enc = fam.sample_encoding(Random(s))
            for a in theta.pi:
                for b in theta.pi:
                    q = Pair(a, b)
                    self.assertEqual(fam.parse_query(enc, fam.render(enc, q)), q)

    def test_parse_query_returns_None_for_malformed_input(self):
        """A6's case. None is not a failure -- it is the recovery lesson's cue."""
        fam = small()
        enc = fam.sample_encoding(Random(0))
        for bad in ([], [vocab.PAD], [vocab.PAD] * 4, [vocab.BOS, vocab.EOS, vocab.SEP],
                    [enc.symbols[0], enc.operator, enc.symbols[1]]):
            self.assertIsNone(fam.parse_query(enc, bad))

    def test_invalid_position_gets_a_well_formed_refusal(self):
        """A6 total: responds sensibly to invalid queries rather than raising."""
        fam = small()
        rng = Random(0)
        theta = fam.sample_theta(0, rng)
        outside = next(p for p in range(fam.pool_size) if p not in theta.pi)
        ans = fam.evaluate(theta, Pair(outside, theta.pi[0]))
        self.assertEqual(ans.position, -1)
        enc = fam.sample_encoding(rng)
        self.assertEqual(fam.render(enc, ans), [vocab.ERR])


class TestA7(unittest.TestCase):
    def test_teacher_query_is_deterministic_in_theta_and_history(self):
        """Interface finding 3: section 7 hands teacher_query no rng, and A7
        wants one-pass computability. Both require determinism."""
        fam = small()
        rng = Random(2)
        theta = fam.sample_theta(0, rng)
        history = []
        for _ in range(3):
            first = fam.teacher_query(theta, history)
            self.assertEqual(first, fam.teacher_query(theta, list(history)))
            history.append((first, fam.evaluate(theta, first)))

    def test_teacher_query_names_a_legal_query(self):
        fam = small()
        rng = Random(5)
        for _ in range(10):
            theta = fam.sample_theta(0, rng)
            history = []
            for _ in range(5):
                q = fam.teacher_query(theta, history)
                self.assertIn(q.a, theta.pi)
                self.assertIn(q.b, theta.pi)
                self.assertNotEqual(fam.evaluate(theta, q).position, -1)
                history.append((q, fam.evaluate(theta, q)))

    def test_teacher_query_actually_reduces_uncertainty_faster_than_random(self):
        """A7 says the target is a good query, not a provably optimal one -- but
        'good' has to mean something. Measured against the family's own random
        sampler on the quantity q* is supposed to move."""
        fam = small()
        teach_alive, rand_alive = [], []
        for s in range(6):
            rng = Random(100 + s)
            theta = fam.sample_theta(0, rng)
            for policy, out in (("teacher", teach_alive), ("random", rand_alive)):
                history = []
                for _ in range(3):
                    q = (fam.teacher_query(theta, history) if policy == "teacher"
                         else fam.sample_query(theta, history, rng))
                    history.append((q, fam.evaluate(theta, q)))
                out.append(belief_state(fam, 0, history).n_alive)
        self.assertLess(
            sum(teach_alive) / len(teach_alive),
            sum(rand_alive) / len(rand_alive),
            "q* leaves at least as many hypotheses alive as random queries do",
        )


class TestL3(unittest.TestCase):
    def test_posterior_matches_brute_force_enumeration(self):
        """Task Spec section 8 step 2's fourth gate, exactly as stated."""
        fam = small()
        rng = Random(7)
        for _ in range(10):
            theta = fam.sample_theta(0, rng)
            history = []
            for _ in range(4):
                q = fam.sample_query(theta, history, rng)
                mine = answer_distribution(fam, history, q, 0)
                brute = brute_force_answer_distribution(fam, 0, history, q)
                for key in set(mine) | set(brute):
                    self.assertAlmostEqual(mine.get(key, 0.0), brute.get(key, 0.0), places=9)
                history.append((q, draw_answer(fam, theta, q, rng)))

    def test_the_target_depends_on_the_query(self):
        """docs/03 finding 1. A family emitting the query-marginal as its L3
        target trains maximal uncertainty about a rule already identified."""
        sensitive, detail = check_query_sensitivity(small(), 0, Random(0))
        self.assertTrue(sensitive, detail)

    def test_the_truth_always_survives_conditioning(self):
        """A posterior that excludes the true theta is a bug that shows up as a
        model failing to learn, which is the hardest way to find it."""
        fam = small()
        rng = Random(13)
        for _ in range(8):
            theta = fam.sample_theta(0, rng)
            history = []
            for _ in range(5):
                q = fam.sample_query(theta, history, rng)
                history.append((q, fam.evaluate(theta, q)))
            bs = belief_state(fam, 0, history)
            w = dict(zip(bs.thetas, bs.weights)).get(theta, 0.0)
            self.assertGreater(w, 0.0, "the true theta was conditioned away")

    def test_prior_weight_matches_how_theta_is_actually_sampled(self):
        """Section 1: the meaning of the family is given entirely by its sampler.

        sample_theta picks m uniformly from the band and then an injection, so a
        small-m theta carries more mass per theta than a large-m one. A uniform
        prior over the enumeration would get the m-marginal wrong and every exact
        target with it.
        """
        fam = small()
        rng = Random(0)
        counts = {m: 0 for m in fam.moduli(0)}
        for _ in range(6000):
            counts[fam.sample_theta(0, rng).m] += 1
        for m in fam.moduli(0):
            empirical = counts[m] / 6000
            implied = fam.prior_weight(ModTheta(m, tuple(range(m))), 0) * math.perm(
                fam.pool_size, m
            )
            self.assertAlmostEqual(empirical, implied, delta=0.03)


class TestTheDial(unittest.TestCase):
    def test_reveal_monotonically_shrinks_the_hypothesis_space(self):
        fam = small()
        rng = Random(1)
        theta, enc = fam.sample_theta(0, rng), fam.sample_encoding(rng)
        alive = []
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            reveal = build_reveal(fam, theta, enc, frac, Random(8))
            self.assertTrue(reveal.consistent(theta))
            alive.append(belief_state(fam, 0, [], reveal).n_alive)
        self.assertEqual(alive, sorted(alive, reverse=True))
        self.assertEqual(alive[-1], 1, "a full reveal must identify theta")

    def test_reveal_monotonically_lowers_residual_entropy(self):
        """The sweep's x-axis has to move with the sweep's dial, or the curve is
        plotted against something that is not the thing being varied."""
        fam = small()
        hs = []
        for frac in (0.0, 0.5, 1.0):
            spec = spec_for_level(Level.L1, T=4)
            spec = type(spec)(T=4, reveal=frac)
            hs.append(measure_residual_entropy(fam, 0, spec, n_episodes=10).rule)
        self.assertEqual(hs, sorted(hs, reverse=True))
        self.assertAlmostEqual(hs[-1], 0.0, places=9)


class TestTrace(unittest.TestCase):
    def test_trace_steps_are_the_derivation_they_claim_to_be(self):
        fam = small()
        rng = Random(6)
        theta = fam.sample_theta(0, rng)
        inv = theta.inverse()
        q = Pair(theta.pi[2], theta.pi[3])
        steps = fam.trace(theta, q)
        self.assertEqual([s.kind for s in steps], ["decode", "decode", "add", "reduce"])
        self.assertEqual(steps[0].value, inv[q.a])
        self.assertEqual(steps[1].value, inv[q.b])
        self.assertEqual(steps[2].value, inv[q.a] + inv[q.b])
        # The reduce step must be the residue the oracle's own answer denotes.
        self.assertEqual(steps[3].value, (inv[q.a] + inv[q.b]) % theta.m)
        self.assertEqual(steps[3].value, inv[fam.evaluate(theta, q).position])

    def test_thinning_is_a_parameter_and_not_a_function_of_k(self):
        """Section 1.2 calls the thinning schedule "the least settled decision in
        the document" and says to treat it as a swept parameter. docs/05 found
        the literature it cites fades against learner competence, not task
        difficulty. A parameter can be swept; a function of k baked in cannot."""
        rng = Random(0)
        full = ModularHiddenPermutationFamily(pool_size=7, trace_detail="full")
        reduced = ModularHiddenPermutationFamily(pool_size=7, trace_detail="reduced")
        none = ModularHiddenPermutationFamily(pool_size=7, trace_detail="none")
        theta = full.sample_theta(0, rng)
        q = Pair(theta.pi[1], theta.pi[2])
        self.assertEqual(len(full.trace(theta, q)), 4)
        self.assertEqual(len(reduced.trace(theta, q)), 1)
        self.assertIsNone(none.trace(theta, q))

        # And the same at every k, which is the point: the Task Spec's default
        # would thin this automatically as k rose, making the schedule a constant
        # of the family rather than something a sweep can vary.
        wide = ModularHiddenPermutationFamily(pool_size=12, trace_detail="full")
        for k in (0, 2, 5):
            theta_k = wide.sample_theta(k, rng)
            q_k = Pair(theta_k.pi[0], theta_k.pi[1])
            self.assertEqual(len(wide.trace(theta_k, q_k)), 4)
            self.assertEqual(len(reduced.trace(theta_k, q_k)), 1)

    def test_a_modulus_the_pool_cannot_hold_is_refused_by_name(self):
        """Found by the test above rather than by inspection.

        Without the guard, k beyond the pool's reach fails as a ValueError out of
        `random.sample` reading "Sample larger than population" -- which names
        neither k, nor the pool, nor this family, and would be debugged from the
        wrong end.
        """
        fam = ModularHiddenPermutationFamily(pool_size=7)
        self.assertEqual(fam.max_k(), 1)
        with self.assertRaises(ValueError) as cm:
            fam.moduli(5)
        self.assertIn("symbol pool", str(cm.exception))

    def test_trace_tokens_are_supervised_when_emitted(self):
        """Section 1.2: "the trace is the teacher's working, loss covering the
        step tokens"."""
        fam = small()
        spec = spec_for_level(Level.L1, T=4, emit_trace=True)
        ep = build_episode(fam, 0, 0, spec)
        trace_positions = [i for i, c in enumerate(ep.channel) if c is Channel.TRACE]
        self.assertTrue(trace_positions)
        self.assertTrue(all(ep.supervised[i] for i in trace_positions))


class TestKnownLimits(unittest.TestCase):
    """The register row's own verdicts, asserted so they cannot quietly drift."""

    def test_A4_fails_at_these_sizes_and_the_row_says_so(self):
        """The row records A4 as REPAIRABLE, not PASS: at small m the oracle is
        an m x m table a learner can memorize outright. Section 6 calls that
        "instructive rather than disqualifying, and exactly what section 8 step 5
        measures". Asserted here so nobody later reads a passing test suite as
        the family being A4-clean."""
        fam = small()
        m = max(fam.moduli(0))
        self.assertLessEqual(m * m, 64, "the whole oracle fits in a 64-entry table")

    def test_enumeration_refuses_rather_than_approximating_when_too_large(self):
        big = ModularHiddenPermutationFamily(pool_size=20)
        with self.assertRaises(ValueError) as cm:
            big.enumerate_theta(3)
        self.assertIn("too large to enumerate", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
