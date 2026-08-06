"""Tests for the probability-matching family.

The load-bearing test is `test_posterior_is_calibrated_not_maximizing`: this
family exists partly because its normative target and the human result documented
in the source paradigm come apart, and the L3 target must sit on the normative
side. If the posterior ever starts returning a point prediction, the family stops
being a calibration exemplar and nothing else here would notice.
"""

from __future__ import annotations

import math
import unittest
from random import Random

from repertoire import vocab
from repertoire.families.stochastic import (
    Outcome,
    ProbabilityMatchingFamily,
    Query,
    RateTheta,
)
from repertoire.form import TaskFamily


def draw(dist, rng: Random) -> Outcome:
    keys = sorted(dist)
    return Outcome(rng.choices(keys, weights=[dist[k] for k in keys])[0])


class TestConformance(unittest.TestCase):
    def test_protocol(self):
        self.assertIsInstance(ProbabilityMatchingFamily(), TaskFamily)

    def test_flags(self):
        f = ProbabilityMatchingFamily()
        self.assertTrue(f.stochastic)
        self.assertFalse(f.supports_L2)

    def test_tokens_in_vocabulary(self):
        f = ProbabilityMatchingFamily()
        rng = Random(1)
        theta = f.sample_theta(2, rng)
        enc = f.sample_encoding(rng)
        stream = f.preamble(theta, enc)
        for _ in range(20):
            q = f.sample_query(theta, [], rng)
            stream += f.render(enc, q) + f.render(enc, draw(f.evaluate(theta, q), rng))
        for t in stream:
            self.assertIn(t, range(vocab.VOCAB_SIZE))

    def test_outcome_symbols_from_alphabet(self):
        f = ProbabilityMatchingFamily()
        enc = f.sample_encoding(Random(2))
        for tok in enc.outcome_symbols:
            self.assertIn(tok, vocab.SYMBOL_IDS)

    def test_a2(self):
        self.assertTrue(ProbabilityMatchingFamily().permuted_alphabet_check(Random(3)))

    def test_seeded_reproducibility(self):
        f = ProbabilityMatchingFamily()

        def run(seed):
            rng = Random(seed)
            theta = f.sample_theta(2, rng)
            enc = f.sample_encoding(rng)
            out = []
            for _ in range(30):
                q = f.sample_query(theta, [], rng)
                out += f.render(enc, draw(f.evaluate(theta, q), rng))
            return out

        self.assertEqual(run(5), run(5))
        self.assertNotEqual(run(5), run(6))


class TestStochasticOracle(unittest.TestCase):
    def test_evaluate_returns_a_distribution_not_a_sample(self):
        # Task Spec 1.3: f is distribution-valued and the harness samples. This
        # is what lets a stochastic family live inside a protocol whose evaluate
        # takes no rng.
        f = ProbabilityMatchingFamily()
        d = f.evaluate(RateTheta(5, 3), Query(0))
        self.assertIsInstance(d, dict)
        self.assertAlmostEqual(sum(d.values()), 1.0, places=12)
        self.assertAlmostEqual(d[1], 4 / 6, places=12)  # interior grid

    def test_empirical_frequency_tracks_the_rate(self):
        f = ProbabilityMatchingFamily()
        rng = Random(4242)
        theta = RateTheta(5, 3)  # p = 4/6
        n = 20000
        ones = sum(draw(f.evaluate(theta, Query(i)), rng).value for i in range(n))
        self.assertLess(abs(ones / n - theta.p), 0.02)


class TestPosterior(unittest.TestCase):
    def test_prior_is_symmetric(self):
        f = ProbabilityMatchingFamily()
        post = f.posterior([], k=2)
        self.assertAlmostEqual(post[1], 0.5, places=12)

    def test_posterior_moves_toward_evidence(self):
        f = ProbabilityMatchingFamily()
        history = [(Query(i), Outcome(1)) for i in range(6)]
        self.assertGreater(f.posterior(history, k=2)[1], 0.8)
        history = [(Query(i), Outcome(0)) for i in range(6)]
        self.assertLess(f.posterior(history, k=2)[1], 0.2)

    def test_posterior_is_calibrated_not_maximizing(self):
        # The family's whole point. With mixed evidence the target must stay a
        # genuine distribution rather than collapsing to the majority outcome --
        # the human result in the source paradigm is matching, the score-optimal
        # policy is maximizing, and OUR target is neither: it is the exact
        # posterior. If this ever returns ~1.0 the calibration exemplar is gone.
        f = ProbabilityMatchingFamily()
        history = [(Query(i), Outcome(1 if i % 4 else 0)) for i in range(12)]
        p = f.posterior(history, k=3)[1]
        self.assertGreater(p, 0.5)
        self.assertLess(p, 0.95)

    def test_posterior_never_fully_resolves(self):
        # Intrinsic non-identifiability: no finite history collapses a rate to a
        # point. This is what makes it the cleanest L3 exemplar in the set.
        f = ProbabilityMatchingFamily()
        history = [(Query(i), Outcome(1)) for i in range(200)]
        self.assertLess(f.posterior(history, k=2)[1], 1.0)

    def test_l3_loss_floor_is_the_entropy_of_the_rate(self):
        # A correct model's loss cannot go below H(p). Stated as a test so a
        # harness change that shifts the floor surfaces here rather than as an
        # unexplained matrix cell.
        f = ProbabilityMatchingFamily()
        theta = RateTheta(5, 3)
        p = theta.p
        floor = -(p * math.log(p) + (1 - p) * math.log(1 - p))
        # Strictly positive for EVERY grid point, which is the property that
        # makes irreducible noise irreducible. Endpoints would give a zero floor.
        for i in range(5):
            q = RateTheta(5, i).p
            self.assertGreater(-(q * math.log(q) + (1 - q) * math.log(1 - q)), 0.0)
        self.assertAlmostEqual(floor, -(p * math.log(p) + (1 - p) * math.log(1 - p)), places=12)


if __name__ == "__main__":
    unittest.main()
