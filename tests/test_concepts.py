"""Tests for the planted prerequisite and near-duplicate pairs.

The load-bearing tests here are the ones that check the *plant property* rather
than the code: that the near-duplicate pair really does compute the same latent
function under different surfaces, and that Type VI really is parity. If those
stop holding, the plants stop being plants and the instrument-validation gate
would pass while measuring something else.
"""

from __future__ import annotations

import unittest
from random import Random

from repertoire import vocab
from repertoire.families.concepts import (
    BrunerConjunctionFamily,
    Category,
    ConjunctionFamily,
    ConjunctionTheta,
    SHJTypeIFamily,
    SHJTypeVIFamily,
    Stimulus,
)
from repertoire.form import TaskFamily

ALL = [
    SHJTypeIFamily(),
    SHJTypeVIFamily(),
    ConjunctionFamily(),
    BrunerConjunctionFamily(),
]


def episode(family, k: int, seed: int, trials: int = 10):
    rng = Random(seed)
    theta = family.sample_theta(k, rng)
    enc = family.sample_encoding(rng)
    stream: list[int] = []
    history: list[tuple] = []
    pre = family.preamble(theta, enc)
    if pre:
        stream += pre
    for _ in range(trials):
        q = family.sample_query(theta, history, rng)
        a = family.evaluate(theta, q)
        stream += family.render(enc, q) + family.render(enc, a)
        history.append((q, a))
    return stream, theta, history


class TestConformance(unittest.TestCase):
    def test_protocol(self):
        for f in ALL:
            self.assertIsInstance(f, TaskFamily, f.name)

    def test_tokens_in_vocabulary(self):
        for f in ALL:
            for seed in (1, 2, 3):
                stream, _, _ = episode(f, k=2, seed=seed)
                for t in stream:
                    self.assertIn(t, range(vocab.VOCAB_SIZE), f.name)

    def test_content_tokens_come_from_the_symbol_alphabet(self):
        # Regression guard. The encoder originally drew "symbols" from raw
        # indices 0..N_SYMBOLS, which are the CONTROL and STRUCTURAL token ids --
        # so content rendered as PAD/BOS/EQ while every in-vocabulary check still
        # passed, because those ids are valid ids. A2 would also have looked fine.
        # Being in the vocabulary is not the same as being in the alphabet.
        for f in ALL:
            enc = f.sample_encoding(Random(3))
            for tok in enc.value_symbols + enc.category_symbols:
                self.assertIn(tok, vocab.SYMBOL_IDS, f.name)

    def test_seeded_reproducibility(self):
        for f in ALL:
            a, _, _ = episode(f, k=2, seed=11)
            b, _, _ = episode(f, k=2, seed=11)
            self.assertEqual(a, b, f.name)
            c, _, _ = episode(f, k=2, seed=12)
            self.assertNotEqual(a, c, f.name)

    def test_a2_permuted_alphabet(self):
        for f in ALL:
            self.assertTrue(f.permuted_alphabet_check(Random(7)), f.name)

    def test_a2_check_can_fail(self):
        # A family that reads meaning off a particular token must fail the check,
        # otherwise the check proves nothing about the families that pass it.
        class Leaky(SHJTypeIFamily):
            def render(self, encoding, obj):
                if isinstance(obj, Category):
                    return [vocab.sym(obj.label)]  # fixed tokens, ignores encoding
                return super().render(encoding, obj)

        self.assertFalse(Leaky().permuted_alphabet_check(Random(7)))

    def test_encoding_varies_per_episode(self):
        f = ConjunctionFamily()
        assignments = {f.sample_encoding(Random(s)).value_symbols for s in range(20)}
        self.assertGreater(len(assignments), 1)


class TestPlantProperties(unittest.TestCase):
    """These check the plants are still plants, not that the code runs."""

    def test_type_vi_is_parity(self):
        # The register predicts shj_type_vi clusters with the parity family
        # rather than with its paradigm-mate. That prediction is only meaningful
        # if it really computes parity.
        f = SHJTypeVIFamily()
        rng = Random(3)
        theta = f.sample_theta(2, rng)
        for _ in range(200):
            q = f.sample_query(theta, [], rng)
            expected = 0
            for bit in q.values:
                expected ^= bit
            self.assertEqual(f.evaluate(theta, q).label, expected ^ theta.polarity)

    def test_type_i_depends_on_exactly_one_dimension(self):
        f = SHJTypeIFamily()
        rng = Random(4)
        theta = f.sample_theta(2, rng)
        d = theta.n_dims
        base = Stimulus(tuple(0 for _ in range(d)))
        changed = []
        for i in range(d):
            flip = list(base.values)
            flip[i] = 1
            if f.evaluate(theta, Stimulus(tuple(flip))).label != f.evaluate(theta, base).label:
                changed.append(i)
        self.assertEqual(changed, [theta.relevant[0]])

    def test_near_duplicate_pair_computes_the_same_function(self):
        # The plant's whole content: same latent operation, different surface.
        # Restricted to the shared binary sub-domain, the two must agree exactly.
        plain, bruner = ConjunctionFamily(), BrunerConjunctionFamily()
        rng = Random(5)
        d = plain.dimensions(2)
        for _ in range(50):
            req = tuple(rng.choice([-1, 0, 1]) for _ in range(d))
            t = ConjunctionTheta(d, req)
            for _ in range(20):
                q = Stimulus(tuple(rng.randrange(2) for _ in range(d)))
                self.assertEqual(
                    plain.evaluate(t, q).label,
                    bruner.evaluate(t, q).label,
                    "near-duplicate pair diverged -- the plant is no longer a plant",
                )

    def test_near_duplicate_pair_has_different_surfaces(self):
        # Same function, and the renderings must NOT coincide, or the matrix
        # would be seeing identity for free rather than through a surface change.
        plain, bruner = ConjunctionFamily(), BrunerConjunctionFamily()
        self.assertEqual(plain.n_values, 2)
        self.assertEqual(bruner.n_values, 3)
        names_plain = {plain.sample_encoding(Random(s)).name for s in range(20)}
        names_bruner = {bruner.sample_encoding(Random(s)).name for s in range(20)}
        self.assertTrue(names_plain.isdisjoint(names_bruner))

    def test_conservative_focusing_probes_one_attribute_at_a_time(self):
        f = ConjunctionFamily()
        rng = Random(6)
        theta = f.sample_theta(2, rng)
        anchor = tuple(max(x, 0) for x in theta.required)
        for n in range(theta.n_dims):
            q = f.teacher_query(theta, [None] * n)
            diffs = sum(1 for a, b in zip(anchor, q.values) if a != b)
            self.assertEqual(diffs, 1)


class TestPosteriors(unittest.TestCase):
    def test_posterior_is_a_distribution(self):
        for f in ALL:
            post = f.posterior([], k=1)
            self.assertAlmostEqual(sum(post.values()), 1.0, places=9, msg=f.name)
            for p in post.values():
                self.assertGreaterEqual(p, 0.0)

    def test_type_i_posterior_concentrates_with_evidence(self):
        f = SHJTypeIFamily()
        rng = Random(9)
        theta = f.sample_theta(1, rng)
        history: list[tuple] = []
        prior = f.posterior(history, k=1)
        spread_before = abs(prior[0] - prior[1])
        for _ in range(12):
            q = f.teacher_query(theta, history)
            history.append((q, f.evaluate(theta, q)))
        after = f.posterior(history, k=1)
        spread_after = abs(after[0] - after[1])
        self.assertGreaterEqual(spread_after, spread_before)

    def test_type_vi_posterior_resolves_polarity_given_a_pending_query(self):
        # Type VI has exactly two hypotheses (the polarities), so one observation
        # settles it. But the answer distribution is only determinate once a
        # QUERY is fixed: marginalized over random queries it stays 0.5/0.5,
        # because parity is balanced. A trailing (query, None) entry is the
        # convention for "condition on this pending query".
        f = SHJTypeVIFamily()
        rng = Random(10)
        theta = f.sample_theta(1, rng)
        q1 = f.sample_query(theta, [], rng)
        history = [(q1, f.evaluate(theta, q1))]

        marginal = f.posterior(history, k=1)
        self.assertAlmostEqual(marginal[0], 0.5, places=6)

        q2 = f.sample_query(theta, history, rng)
        conditioned = f.posterior(history + [(q2, None)], k=1)
        self.assertAlmostEqual(max(conditioned.values()), 1.0, places=6)
        self.assertEqual(
            max(conditioned, key=conditioned.get), f.evaluate(theta, q2).label
        )


if __name__ == "__main__":
    unittest.main()
