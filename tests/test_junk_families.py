"""Tests for the two junk plants.

These matter more than their size suggests: Task Spec section 8 step 4 gates the
programme on planted junk reading near-zero, so a plant that silently stops being
junk would let the instrument-validation gate pass while measuring the wrong
thing.  The tests that check *why* each family is junk -- unlearnable vs. learned
instantly -- are the load-bearing ones.
"""

from __future__ import annotations

import math
import unittest
from random import Random

from repertoire import vocab
from repertoire.families.junk import (
    Answer,
    ConstantTargetFamily,
    ConstantTheta,
    Query,
    RandomTargetFamily,
    alphabet_size,
)
from repertoire.form import TaskFamily

PROTOCOL_METHODS = [
    "sample_theta", "sample_encoding", "sample_query", "teacher_query",
    "evaluate", "trace", "render", "preamble", "posterior",
    "permuted_alphabet_check",
]


def run_episode(family, k: int, seed: int, trials: int = 8) -> list[int]:
    """Render one episode to a flat token stream, the way a harness would."""
    rng = Random(seed)
    theta = family.sample_theta(k, rng)
    enc = family.sample_encoding(rng)
    stream: list[int] = []
    pre = family.preamble(theta, enc)
    if pre:
        stream += pre
    history: list[tuple] = []
    for _ in range(trials):
        q = family.sample_query(theta, history, rng)
        out = family.evaluate(theta, q)
        if isinstance(out, dict):  # distribution-valued: harness samples
            symbols = sorted(out)
            ans = Answer(rng.choices(symbols, weights=[out[s] for s in symbols])[0])
        else:
            ans = out
        stream += family.render(enc, q) + family.render(enc, ans)
        history.append((q, ans))
    return stream


class TestProtocolConformance(unittest.TestCase):
    def test_both_satisfy_protocol(self):
        for family in (RandomTargetFamily(), ConstantTargetFamily()):
            for m in PROTOCOL_METHODS:
                self.assertTrue(callable(getattr(family, m, None)), f"{family.name}.{m}")
            for attr in ("name", "supports_L2", "emits_trace", "stochastic"):
                self.assertTrue(hasattr(family, attr), f"{family.name}.{attr}")
            self.assertIsInstance(family, TaskFamily)

    def test_flags_are_honest(self):
        self.assertTrue(RandomTargetFamily().stochastic)
        self.assertFalse(ConstantTargetFamily(epsilon=0.0).stochastic)
        self.assertTrue(ConstantTargetFamily(epsilon=0.1).stochastic)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_identical_stream(self):
        for family in (RandomTargetFamily(), ConstantTargetFamily(epsilon=0.1)):
            a = run_episode(family, k=3, seed=42)
            b = run_episode(family, k=3, seed=42)
            self.assertEqual(a, b, family.name)

    def test_different_seed_differs(self):
        for family in (RandomTargetFamily(), ConstantTargetFamily(epsilon=0.1)):
            a = run_episode(family, k=3, seed=42)
            b = run_episode(family, k=3, seed=43)
            self.assertNotEqual(a, b, family.name)


class TestRandomFamilyIsActuallyJunk(unittest.TestCase):
    """The property that makes it a plant rather than a memorization family."""

    def test_answer_does_not_depend_on_query_or_theta(self):
        # evaluate returns the same uniform distribution for every query and
        # every theta. If this ever becomes a function of the query, the family
        # turns into a learnable lookup table and stops being junk.
        fam = RandomTargetFamily()
        rng = Random(0)
        t1, t2 = fam.sample_theta(3, rng), fam.sample_theta(3, rng)
        n = alphabet_size(3)
        base = fam.evaluate(t1, Query(0))
        for q in range(n):
            self.assertEqual(fam.evaluate(t1, Query(q)), base)
            self.assertEqual(fam.evaluate(t2, Query(q)), base)

    def test_repeated_query_answers_disagree(self):
        # Deliberately the opposite of consistency. A fixed answer per query
        # would be a table, and a table is learnable within the episode.
        fam = RandomTargetFamily()
        rng = Random(7)
        theta = fam.sample_theta(3, rng)
        dist = fam.evaluate(theta, Query(0))
        symbols = sorted(dist)
        draws = {
            rng.choices(symbols, weights=[dist[s] for s in symbols])[0]
            for _ in range(200)
        }
        self.assertGreater(len(draws), 1, "repeats agreed -- family is a lookup table")

    def test_marginal_is_uniform(self):
        fam = RandomTargetFamily()
        rng = Random(12345)
        n = alphabet_size(2)
        counts = [0] * n
        trials = 20000
        for _ in range(trials):
            theta = fam.sample_theta(2, rng)
            dist = fam.evaluate(theta, Query(rng.randrange(n)))
            symbols = sorted(dist)
            counts[rng.choices(symbols, weights=[dist[s] for s in symbols])[0]] += 1
        expected = trials / n
        for c in counts:
            self.assertLess(abs(c - expected) / expected, 0.15)

    def test_posterior_is_exactly_uniform(self):
        fam = RandomTargetFamily()
        n = alphabet_size(4)
        post = fam.posterior([], k=4)
        for s in range(n):
            self.assertAlmostEqual(post[s], 1.0 / n, places=12)
        # The L3 floor is log(n) exactly; state it as a test so a harness change
        # that shifts the floor shows up here rather than in a matrix cell.
        loss = -sum(p * math.log(p) for p in post.values())
        self.assertAlmostEqual(loss, math.log(n), places=12)

    def test_no_preamble(self):
        fam = RandomTargetFamily()
        rng = Random(1)
        theta = fam.sample_theta(3, rng)
        self.assertIsNone(fam.preamble(theta, fam.sample_encoding(rng)))


class TestConstantFamilyPosterior(unittest.TestCase):
    def test_prior_is_uniform(self):
        fam = ConstantTargetFamily(epsilon=0.1)
        n = alphabet_size(3)
        post = fam.posterior([], k=3)
        for s in range(n):
            self.assertAlmostEqual(post[s], 1.0 / n, places=12)

    def test_exact_after_one_observation_epsilon_zero(self):
        fam = ConstantTargetFamily(epsilon=0.0)
        n = alphabet_size(3)
        post = fam.posterior([(Query(0), Answer(2))], k=3)
        self.assertAlmostEqual(post[2], 1.0, places=12)
        self.assertAlmostEqual(sum(post[s] for s in range(n) if s != 2), 0.0, places=12)

    def test_exact_after_one_observation_epsilon_positive(self):
        # The shorthand "mass 1-eps on the observed symbol" is WRONG here, and
        # this test pins the exact value so the shorthand cannot creep back in.
        eps = 0.1
        fam = ConstantTargetFamily(epsilon=eps)
        n = alphabet_size(3)
        off = eps / (n - 1)
        post = fam.posterior([(Query(0), Answer(5))], k=3)

        # theta posterior after seeing 5: (1-eps) on 5, off elsewhere (normalized).
        # predictive(s) = P(theta=s)*(1-eps) + (1-P(theta=s))*off
        expected_5 = (1 - eps) * (1 - eps) + (1 - (1 - eps)) * off
        self.assertAlmostEqual(post[5], expected_5, places=12)
        self.assertNotAlmostEqual(post[5], 1 - eps, places=6)
        self.assertAlmostEqual(sum(post.values()), 1.0, places=12)
        for s in range(n):
            self.assertGreater(post[s], 0.0)

    def test_posterior_concentrates_with_evidence(self):
        fam = ConstantTargetFamily(epsilon=0.1)
        h = [(Query(0), Answer(3))]
        first = fam.posterior(h, k=3)[3]
        h += [(Query(1), Answer(3)), (Query(2), Answer(3))]
        later = fam.posterior(h, k=3)[3]
        self.assertGreater(later, first)


class TestA2(unittest.TestCase):
    def test_permuted_alphabet_check_passes(self):
        self.assertTrue(RandomTargetFamily().permuted_alphabet_check(Random(1)))
        self.assertTrue(ConstantTargetFamily(epsilon=0.0).permuted_alphabet_check(Random(2)))
        self.assertTrue(ConstantTargetFamily(epsilon=0.1).permuted_alphabet_check(Random(3)))

    def test_check_would_catch_a_violation(self):
        # A plant that passes its own A2 check proves nothing unless the check can
        # fail. Break the family's equivariance deliberately and confirm it does.
        class Broken(ConstantTargetFamily):
            def evaluate(self, theta, query):
                return Answer((theta.constant + query.symbol) % theta.alphabet_size)

        self.assertFalse(Broken(epsilon=0.0).permuted_alphabet_check(Random(4)))


class TestRendering(unittest.TestCase):
    def test_tokens_in_vocabulary(self):
        for family in (RandomTargetFamily(), ConstantTargetFamily(epsilon=0.1)):
            for seed in (1, 2, 3):
                for tok in run_episode(family, k=5, seed=seed):
                    self.assertIn(tok, range(vocab.VOCAB_SIZE), family.name)

    def test_both_encodings_reachable(self):
        fam = ConstantTargetFamily()
        seen = {fam.sample_encoding(Random(s)).name for s in range(30)}
        self.assertEqual(seen, {"infix", "arrow"})

    def test_render_rejects_unknown_objects(self):
        fam = RandomTargetFamily()
        with self.assertRaises(TypeError):
            fam.render(fam.sample_encoding(Random(0)), ConstantTheta(4, 0, 0.0))


if __name__ == "__main__":
    unittest.main()
