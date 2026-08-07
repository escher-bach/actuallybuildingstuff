"""Tests for parity identification.

The load-bearing test is `test_type_vi_is_the_full_subset_special_case`. The
register predicts, before measurement, that SHJ Type VI clusters with this family
rather than with its own paradigm-mate — and that prediction is only meaningful
if the two really do compute the same thing where they overlap. Prose in a row
is not evidence; this is.
"""

from __future__ import annotations

import unittest
from random import Random

from repertoire import vocab
from repertoire.families.algebraic import (
    BitVector,
    Parity,
    ParityIdentificationFamily,
    ParityTheta,
)
from repertoire.families.concepts import SHJTypeVIFamily, Stimulus
from repertoire.form import TaskFamily


class TestConformance(unittest.TestCase):
    def test_protocol(self):
        self.assertIsInstance(ParityIdentificationFamily(), TaskFamily)

    def test_tokens_and_alphabet(self):
        f = ParityIdentificationFamily()
        rng = Random(1)
        theta = f.sample_theta(1, rng)
        enc = f.sample_encoding(rng)
        for tok in enc.bit_symbols + enc.result_symbols:
            self.assertIn(tok, vocab.SYMBOL_IDS)
        stream = f.preamble(theta, enc)
        for _ in range(10):
            q = f.sample_query(theta, [], rng)
            stream += f.render(enc, q) + f.render(enc, f.evaluate(theta, q))
        for t in stream:
            self.assertIn(t, range(vocab.VOCAB_SIZE))

    def test_a2(self):
        self.assertTrue(ParityIdentificationFamily().permuted_alphabet_check(Random(4)))

    def test_seeded_reproducibility(self):
        f = ParityIdentificationFamily()

        def run(seed):
            rng = Random(seed)
            theta = f.sample_theta(1, rng)
            enc = f.sample_encoding(rng)
            out = []
            for _ in range(10):
                q = f.sample_query(theta, [], rng)
                out += f.render(enc, q) + f.render(enc, f.evaluate(theta, q))
            return out

        self.assertEqual(run(3), run(3))
        self.assertNotEqual(run(3), run(9))

    def test_three_encodings_reachable_and_one_differs_structurally(self):
        f = ParityIdentificationFamily()
        encs = {}
        for s in range(40):
            e = f.sample_encoding(Random(s))
            encs[e.name] = e
        self.assertEqual(set(encs), {"bits", "equation", "grouped"})
        # A3 must not be vacuous: at least one rendering differs in LENGTH, not
        # only in which token separates fields. This is the hazard that made the
        # first A3 leak test find zero variance to measure.
        rng = Random(5)
        theta = f.sample_theta(2, rng)
        q = f.sample_query(theta, [], rng)
        lengths = {n: len(f.render(e, q)) for n, e in encs.items()}
        self.assertGreater(max(lengths.values()), min(lengths.values()))


class TestParitySemantics(unittest.TestCase):
    def test_computes_xor_over_the_subset(self):
        f = ParityIdentificationFamily()
        rng = Random(2)
        theta = f.sample_theta(1, rng)
        for _ in range(300):
            q = f.sample_query(theta, [], rng)
            expected = theta.offset
            for i in theta.subset:
                expected ^= q.bits[i]
            self.assertEqual(f.evaluate(theta, q).value, expected)

    def test_irrelevant_coordinates_do_not_matter(self):
        f = ParityIdentificationFamily()
        rng = Random(6)
        theta = f.sample_theta(1, rng)
        outside = [i for i in range(theta.n_dims) if i not in theta.subset]
        if not outside:
            self.skipTest("sampled theta uses every coordinate")
        base = BitVector(tuple(0 for _ in range(theta.n_dims)))
        for i in outside:
            flipped = list(base.bits)
            flipped[i] = 1
            self.assertEqual(
                f.evaluate(theta, BitVector(tuple(flipped))).value,
                f.evaluate(theta, base).value,
            )

    def test_subset_is_never_empty(self):
        # An empty subset makes the answer constant -- that is the junk-trivial
        # family wearing this family's clothes, and it would pollute the matrix
        # row with a degenerate episode nobody would notice.
        f = ParityIdentificationFamily()
        rng = Random(11)
        for _ in range(200):
            self.assertGreater(len(f.sample_theta(1, rng).subset), 0)

    def test_basis_probe_identifies_one_coordinate_per_query(self):
        f = ParityIdentificationFamily()
        rng = Random(8)
        theta = f.sample_theta(1, rng)
        recovered = []
        for n in range(theta.n_dims):
            q = f.teacher_query(theta, [None] * n)
            self.assertEqual(sum(q.bits), 1)
            i = q.bits.index(1)
            # e_i's answer differs from the all-zeros answer iff i participates.
            zero = f.evaluate(theta, BitVector(tuple(0 for _ in range(theta.n_dims))))
            if f.evaluate(theta, q).value != zero.value:
                recovered.append(i)
        self.assertEqual(tuple(recovered), theta.subset)


class TestUnplantedNearDuplicate(unittest.TestCase):
    """Parity and SHJ Type VI are the same object where they overlap.

    Nobody planted this. Parity came from computational learning theory, Type VI
    from 1961 categorization psychology, and the identity fell out when both were
    written into the same form.
    """

    def test_type_vi_is_the_full_subset_special_case(self):
        parity = ParityIdentificationFamily()
        shj = SHJTypeVIFamily()
        rng = Random(21)

        d = 5
        for offset in (0, 1):
            ptheta = ParityTheta(d, tuple(range(d)), offset)
            stheta = shj.sample_theta(d - 3, rng)
            stheta = type(stheta)(d, tuple(range(d)), offset)
            for _ in range(200):
                bits = tuple(rng.randrange(2) for _ in range(d))
                self.assertEqual(
                    parity.evaluate(ptheta, BitVector(bits)).value,
                    shj.evaluate(stheta, Stimulus(bits)).label,
                    "the unplanted near-duplicate diverged",
                )

    def test_the_two_families_render_differently(self):
        # Same function, and the surfaces must NOT coincide, or the matrix is
        # handed the identity for free instead of having to see through it.
        parity, shj = ParityIdentificationFamily(), SHJTypeVIFamily()
        pnames = {parity.sample_encoding(Random(s)).name for s in range(30)}
        snames = {shj.sample_encoding(Random(s)).name for s in range(30)}
        self.assertTrue(pnames.isdisjoint(snames))


class TestPosterior(unittest.TestCase):
    def test_marginal_is_balanced(self):
        # Parity is balanced over the query space, so with no pending query the
        # answer distribution is exactly 0.5/0.5 however much is known. This is
        # the concrete case that showed posterior(history, k) cannot express the
        # L3 target without a pending query.
        f = ParityIdentificationFamily()
        post = f.posterior([], k=0)
        self.assertAlmostEqual(post[0], 0.5, places=12)

    def test_conditioned_posterior_resolves_once_identified(self):
        f = ParityIdentificationFamily()
        rng = Random(13)
        k = 0
        theta = f.sample_theta(k, rng)
        history: list[tuple] = []
        for _ in range(theta.n_dims + 2):
            q = f.teacher_query(theta, history)
            history.append((q, f.evaluate(theta, q)))
        probe = f.sample_query(theta, history, rng)
        post = f.posterior(history + [(probe, None)], k=k)
        self.assertAlmostEqual(max(post.values()), 1.0, places=9)
        self.assertEqual(max(post, key=post.get), f.evaluate(theta, probe).value)

    def test_posterior_is_a_distribution(self):
        f = ParityIdentificationFamily()
        post = f.posterior([(BitVector((1, 0, 0, 0, 0, 0)), Parity(1))], k=0)
        self.assertAlmostEqual(sum(post.values()), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
