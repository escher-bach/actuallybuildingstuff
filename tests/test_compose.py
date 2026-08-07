"""Tests for composition (Task Spec section 1.1).

Three things are being tested, and they are different in kind.

1. The machinery works.
2. **Gate 2 fires.** Section 1.1 says A5 "is precisely the condition under which
   a composite is meaningful rather than a type-checking accident" -- so a type
   check alone is explicitly not enough, and the semantic gate must refuse a
   composite that type-checks while a stage does no work.
3. **Closure is measured, not assumed.** Section 1.1 rests on it: "without
   closure you are enumerating, and enumeration cannot cover." A basis with no
   legal compositions has no closure, and until composition was implemented
   nobody could tell.
"""

from __future__ import annotations

import unittest
from random import Random

from repertoire import vocab
from repertoire.compose import (
    Composite,
    CompositionIncoherent,
    CompositionTypeError,
    closure_report,
    coherence_report,
    compose,
)
from repertoire.families import (
    ConjunctionFamily,
    ConstantTargetFamily,
    ParityIdentificationFamily,
    PermutedBitsFamily,
    ProbabilityMatchingFamily,
    RandomTargetFamily,
    SHJTypeIFamily,
)
from repertoire.families.algebraic import BitVector, Parity, ParityTheta, PermTheta
from repertoire.families.concepts import Stimulus


def parity_of_permuted() -> Composite:
    return compose(
        ParityIdentificationFamily(),
        PermutedBitsFamily(),
        types=[BitVector, BitVector],
        name="parity_of_permuted_bits",
    )


def depth3() -> Composite:
    return compose(
        ParityIdentificationFamily(),
        PermutedBitsFamily(),
        PermutedBitsFamily(),
        types=[BitVector, BitVector, BitVector],
        name="depth3_parity_perm_perm",
    )


class TestGateOneTypes(unittest.TestCase):
    def test_matching_types_compose(self):
        self.assertEqual(parity_of_permuted().name, "parity_of_permuted_bits")

    def test_mismatched_types_are_refused(self):
        with self.assertRaises(CompositionTypeError):
            compose(
                SHJTypeIFamily(),
                ParityIdentificationFamily(),
                types=[Stimulus, Stimulus],
                check_coherence=False,
            )

    def test_the_error_names_the_family_and_both_types(self):
        try:
            compose(
                ConjunctionFamily(),
                ParityIdentificationFamily(),
                types=[Stimulus, Stimulus],
                check_coherence=False,
            )
        except CompositionTypeError as e:
            msg = str(e)
            for token in ("parity_identification", "Parity", "Stimulus"):
                self.assertIn(token, msg)
        else:
            self.fail("expected CompositionTypeError")


class TestGateTwoSemanticCoherence(unittest.TestCase):
    """The gate section 1.1 actually asks for, and the one a type check misses."""

    def test_a_coherent_composite_passes(self):
        r = coherence_report(parity_of_permuted(), k=1, rng=Random(1))
        self.assertTrue(r.ok, r.failures())
        self.assertTrue(all(r.varies_with_each_theta))

    def test_a_constant_outer_stage_is_caught(self):
        # The constant family ignores its input entirely. This TYPE-CHECKS --
        # both sides speak BitVector -- and is exactly the accident section 1.1
        # warns about. Found in the wild by closure_report, not contrived.
        with self.assertRaises(CompositionIncoherent):
            compose(
                ConstantTargetFamily(),
                PermutedBitsFamily(),
                types=[BitVector, BitVector],
                name="constant_of_permuted",
            )

    def test_the_refusal_explains_itself_in_the_spec_terms(self):
        try:
            compose(
                ConstantTargetFamily(),
                PermutedBitsFamily(),
                types=[BitVector, BitVector],
            )
        except CompositionIncoherent as e:
            self.assertIn("type-checking accident", str(e))
        else:
            self.fail("expected CompositionIncoherent")

    def test_the_gate_can_be_disabled_only_deliberately(self):
        c = compose(
            ConstantTargetFamily(),
            PermutedBitsFamily(),
            types=[BitVector, BitVector],
            check_coherence=False,
        )
        self.assertIsInstance(c, Composite)

    def test_coherence_report_identifies_which_stage_is_idle(self):
        c = Composite([ConstantTargetFamily(), PermutedBitsFamily()])
        r = coherence_report(c, k=1, rng=Random(2))
        self.assertFalse(r.ok)
        self.assertTrue(any("no work" in f or "constant" in f for f in r.failures()))


class TestDepthAsTheKnob(unittest.TestCase):
    """Section 1.1: depth is the difficulty knob that does not require inventing
    a new family per level. That only holds if composites are n-ary."""

    def test_depth_three_builds_and_evaluates(self):
        c = depth3()
        self.assertEqual(len(c.stages), 3)
        rng = Random(2)
        theta = c.sample_theta(1, rng)
        self.assertEqual(len(theta.parts), 3)
        q = c.sample_query(theta, [], rng)
        self.assertIsInstance(c.evaluate(theta, q), Parity)

    def test_depth_three_is_not_depth_two(self):
        d2, d3 = parity_of_permuted(), depth3()
        rng = Random(3)
        t2 = d2.sample_theta(1, rng)
        t3 = d3.sample_theta(1, rng)
        # Share the outer stage so the only difference is the extra depth.
        t3 = type(t3)((t2.parts[0], t3.parts[1], t3.parts[2]))
        diffs = sum(
            d2.evaluate(t2, q).value != d3.evaluate(t3, q).value
            for q in (d2.sample_query(t2, [], rng) for _ in range(200))
        )
        self.assertGreater(diffs, 0)

    def test_a_composite_of_one_is_refused(self):
        with self.assertRaises(ValueError):
            Composite([ParityIdentificationFamily()])


class TestClosureOfTheBasis(unittest.TestCase):
    """Closure is a property of the BASIS, which is why no single row could see
    it was missing."""

    def test_label_returning_families_are_still_dead_ends(self):
        # If someone adds an endomorphic family, closure changes and this test
        # says so rather than the change passing unnoticed.
        for f in (ParityIdentificationFamily(), SHJTypeIFamily(), ConjunctionFamily()):
            rng = Random(1)
            theta = f.sample_theta(1, rng)
            q = f.sample_query(theta, [], rng)
            self.assertNotIsInstance(
                f.evaluate(theta, q),
                type(q),
                f"{f.name} became endomorphic -- update the closure figures in "
                "docs/02, docs/07 hazard 21 and compose.py",
            )

    def test_one_endomorphic_family_exists(self):
        f = PermutedBitsFamily()
        rng = Random(2)
        theta = f.sample_theta(1, rng)
        q = f.sample_query(theta, [], rng)
        self.assertIsInstance(f.evaluate(theta, q), type(q))

    def test_report_finds_the_legal_composition_and_the_void_one(self):
        text = closure_report(
            [
                ParityIdentificationFamily(),
                PermutedBitsFamily(),
                SHJTypeIFamily(),
                ConjunctionFamily(),
                RandomTargetFamily(),
                ConstantTargetFamily(),
                ProbabilityMatchingFamily(),
            ]
        )
        self.assertIn("parity_identification o permuted_bits", text)
        self.assertIn("VOID", text)
        self.assertIn("closure =", text)


class TestCompositeBehaviour(unittest.TestCase):
    def test_theta_is_the_tuple_of_component_thetas(self):
        c = parity_of_permuted()
        theta = c.sample_theta(1, Random(3))
        self.assertIsInstance(theta.outer, ParityTheta)
        self.assertIsInstance(theta.inner, PermTheta)

    def test_evaluate_is_outer_of_inner(self):
        c = parity_of_permuted()
        rng = Random(4)
        theta = c.sample_theta(1, rng)
        for _ in range(200):
            q = c.sample_query(theta, [], rng)
            mid = c.inner.evaluate(theta.inner, q)
            self.assertEqual(
                c.evaluate(theta, q).value, c.outer.evaluate(theta.outer, mid).value
            )

    def test_the_composite_is_not_its_outer_family(self):
        c = parity_of_permuted()
        rng = Random(5)
        theta = c.sample_theta(1, rng)
        diffs = 0
        for _ in range(200):
            q = c.sample_query(theta, [], rng)
            diffs += c.outer.evaluate(theta.outer, q).value != c.evaluate(theta, q).value
        self.assertGreater(diffs, 0)

    def test_hypothesis_space_is_the_product(self):
        # The A4 headroom argument, checked rather than asserted.
        c = parity_of_permuted()
        seen = {tuple(c.sample_theta(1, Random(s)).parts) for s in range(200)}
        self.assertGreater(len(seen), 190)

    def test_seeded_reproducibility(self):
        c = parity_of_permuted()

        def run(seed):
            rng = Random(seed)
            theta = c.sample_theta(1, rng)
            enc = c.sample_encoding(rng)
            out = []
            for _ in range(8):
                q = c.sample_query(theta, [], rng)
                out += c.render(enc, q) + c.render(enc, c.evaluate(theta, q))
            return out

        self.assertEqual(run(7), run(7))
        self.assertNotEqual(run(7), run(8))

    def test_rendered_tokens_are_in_vocabulary(self):
        c = parity_of_permuted()
        rng = Random(9)
        theta = c.sample_theta(2, rng)
        enc = c.sample_encoding(rng)
        stream = c.preamble(theta, enc) or []
        for _ in range(10):
            q = c.sample_query(theta, [], rng)
            stream += c.render(enc, q) + c.render(enc, c.evaluate(theta, q))
        for t in stream:
            self.assertIn(t, range(vocab.VOCAB_SIZE))

    def test_a2_holds_for_the_composite(self):
        self.assertTrue(parity_of_permuted().permuted_alphabet_check(Random(11)))


class TestCompositeL3(unittest.TestCase):
    def test_posterior_requires_a_pending_query(self):
        # Marginalizing over the query space is not the L3 target -- docs/03
        # finding 1, now enforced for composites too.
        with self.assertRaises(NotImplementedError):
            parity_of_permuted().posterior([], k=1)

    def test_product_posterior_is_exact_when_stages_enumerate(self):
        c = parity_of_permuted()
        rng = Random(5)
        theta = c.sample_theta(0, rng)
        history: list = []
        for _ in range(3):
            q = c.sample_query(theta, history, rng)
            history.append((q, c.evaluate(theta, q)))
        probe = c.sample_query(theta, history, rng)
        post = c.posterior(history + [(probe, None)], k=0)

        self.assertGreater(len(post), 0)
        self.assertAlmostEqual(sum(post.values()), 1.0, places=9)
        truth = c.evaluate(theta, probe)
        key = (type(truth).__name__, tuple(sorted(truth.__dict__.items())))
        self.assertGreater(
            post.get(key, 0.0), 0.0, "the true answer lost all posterior mass"
        )

    def test_posterior_refuses_when_a_stage_cannot_enumerate(self):
        # Refusing beats approximating: an approximate calibration target trains
        # miscalibration, and does it silently.
        broken = Composite([SHJTypeIFamily(), PermutedBitsFamily()])
        with self.assertRaises(NotImplementedError):
            broken.posterior([(BitVector((0, 0, 0, 0, 0, 0)), None)], k=0)


class TestHonestLimits(unittest.TestCase):
    def test_l2_is_refused_not_faked(self):
        c = parity_of_permuted()
        self.assertFalse(c.supports_L2)
        with self.assertRaises(NotImplementedError):
            c.teacher_query(c.sample_theta(1, Random(1)), [])

    def test_stochastic_stage_is_refused(self):
        c = Composite([ParityIdentificationFamily(), ProbabilityMatchingFamily()])
        rng = Random(1)
        theta = c.sample_theta(1, rng)
        with self.assertRaises(CompositionTypeError):
            c.evaluate(theta, c.inner.sample_query(theta.inner, [], rng))


if __name__ == "__main__":
    unittest.main()
