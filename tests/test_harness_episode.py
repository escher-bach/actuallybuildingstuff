"""Episode construction, the loss mask, and the level wrappers.

Written to the standard the rest of this repository holds: **every check has a
case that makes it fail.**  Hazard 6 records two occasions where a check could
not fail and nobody noticed, and the fix each time was a deliberately-broken
subject, so the broken subjects are here alongside the working ones.
"""

from __future__ import annotations

import math
import unittest
from random import Random

from repertoire import vocab
from repertoire.form import Level, TaskFamily
from repertoire.harness.episode import (
    Channel,
    Episode,
    EpisodeSpec,
    QuerySource,
    TargetMode,
    build_episode,
    build_reveal,
    episode_seed,
    spec_for_level,
)
from repertoire.harness.entropy import (
    belief_state,
    brute_force_answer_distribution,
    check_prior_matches_sampler,
    measure_residual_entropy,
)
from repertoire.harness.protocol import (
    POSTERIOR_ROUTE,
    ProtocolGap,
    answer_distribution,
    check_query_sensitivity,
    draw_answer,
)
from repertoire.harness.stub import Answer, LeakyStubFamily, Query, StubLookupFamily


class TestStubConformance(unittest.TestCase):
    def test_stub_is_a_section_7_family(self):
        self.assertIsInstance(StubLookupFamily(), TaskFamily)

    def test_a2_passes_and_the_leaky_stub_fails_it(self):
        # Both halves matter. A plant passing its own check proves nothing
        # unless failure is reachable (hazard 6, met twice).
        self.assertTrue(StubLookupFamily(d=6).permuted_alphabet_check(Random(0)))
        self.assertFalse(LeakyStubFamily(d=6).permuted_alphabet_check(Random(0)))

    def test_content_tokens_come_from_the_symbol_alphabet(self):
        """Hazard 5: being in the vocabulary is not being in the alphabet.

        A family rendering content as PAD/BOS passes the in-vocabulary test and
        passes A2, because equivariance under the wrong alphabet is still
        equivariance. The only thing that catches it is asserting the alphabet.
        """
        fam = StubLookupFamily(d=6)
        rng = Random(3)
        enc = fam.sample_encoding(rng)
        symbols = set(vocab.SYMBOL_IDS)
        self.assertTrue(set(enc.index_symbols) <= symbols)
        self.assertTrue(set(enc.value_symbols) <= symbols)


class TestSeeding(unittest.TestCase):
    def test_episode_is_reconstructible_from_its_seed(self):
        fam = StubLookupFamily(d=6)
        spec = spec_for_level(Level.L1, T=5)
        a = build_episode(fam, 1, 42, spec)
        b = build_episode(fam, 1, 42, spec)
        self.assertEqual(a.tokens, b.tokens)
        self.assertEqual(a.targets, b.targets)
        self.assertEqual(a.supervised, b.supervised)

    def test_different_seeds_give_different_episodes(self):
        fam = StubLookupFamily(d=6)
        spec = spec_for_level(Level.L1, T=5)
        self.assertNotEqual(
            build_episode(fam, 1, 1, spec).tokens,
            build_episode(fam, 1, 2, spec).tokens,
        )

    def test_seed_derivation_is_stable_across_processes(self):
        """Section 7 calls seeded reconstructibility non-negotiable.

        Python salts `hash()` on str per process, so a seed derived from it is
        reproducible within a run and not across runs -- which passes every test
        anyone would think to write while failing the one thing the guarantee is
        for (re-running a branch after a backtrack, in a later session).
        Pinned to a literal so a change to the derivation is a test failure.
        """
        self.assertEqual(
            episode_seed("stub_lookup", 1, "T5/r0.000000/sampled/realized/no", 42),
            9485081738189367806,
        )

    def test_level_and_k_change_the_stream(self):
        s = {
            episode_seed("f", k, spec_for_level(lv, T=4).key, 0)
            for k in (1, 2)
            for lv in (Level.L0, Level.L1, Level.L3)
        }
        self.assertEqual(len(s), 6)


class TestLossMask(unittest.TestCase):
    def setUp(self):
        self.fam = StubLookupFamily(d=6)

    def test_answers_are_supervised_and_the_preamble_never_is(self):
        for level in (Level.L0, Level.L1, Level.L3):
            ep = build_episode(self.fam, 1, 0, spec_for_level(level, T=5))
            for i, ch in enumerate(ep.channel):
                if ch is Channel.ANSWER:
                    self.assertTrue(ep.supervised[i], f"{level} answer unsupervised")
                if ch in (Channel.PREAMBLE, Channel.STRUCTURAL, Channel.ORACLE_ECHO):
                    self.assertFalse(ep.supervised[i], f"{level} supervised {ch.name}")

    def test_queries_are_supervised_only_at_L2(self):
        for level in (Level.L0, Level.L1, Level.L3):
            ep = build_episode(self.fam, 1, 0, spec_for_level(level, T=5))
            for i, ch in enumerate(ep.channel):
                if ch is Channel.QUERY:
                    self.assertFalse(ep.supervised[i])

    def test_supervised_token_count_is_one_per_trial(self):
        spec = spec_for_level(Level.L1, T=7)
        ep = build_episode(self.fam, 1, 0, spec)
        self.assertEqual(ep.n_supervised, 7)

    def test_check_rejects_a_supervised_preamble(self):
        """The mask check must be able to fail, or it is not a check."""
        ep = build_episode(self.fam, 1, 0, spec_for_level(Level.L0, T=4))
        i = ep.channel.index(Channel.PREAMBLE)
        ep.supervised[i] = True
        with self.assertRaises(AssertionError):
            ep.check()

    def test_check_rejects_an_unsupervised_answer(self):
        ep = build_episode(self.fam, 1, 0, spec_for_level(Level.L1, T=4))
        i = ep.channel.index(Channel.ANSWER)
        ep.supervised[i] = False
        with self.assertRaises(AssertionError):
            ep.check()

    def test_check_rejects_a_token_outside_the_vocabulary(self):
        ep = build_episode(self.fam, 1, 0, spec_for_level(Level.L1, T=4))
        ep.tokens[3] = vocab.VOCAB_SIZE + 5
        with self.assertRaises(AssertionError):
            ep.check()

    def test_check_rejects_misaligned_arrays(self):
        ep = build_episode(self.fam, 1, 0, spec_for_level(Level.L1, T=4))
        ep.targets.pop()
        with self.assertRaises(AssertionError):
            ep.check()


class TestLevels(unittest.TestCase):
    def setUp(self):
        self.fam = StubLookupFamily(d=6)

    def test_all_four_levels_build(self):
        from repertoire.harness.train import round_trip_all_levels

        out = round_trip_all_levels(self.fam, k=1, T=5)
        for level, summary in out.items():
            self.assertNotIn("error", summary, f"{level}: {summary.get('error')}")

    def test_L0_and_L1_differ_in_content_but_not_in_length(self):
        """The dial's endpoints must be built the same way as its interior.

        Same construction, same length, different content: that is what makes a
        difference between reveal=0.0 and reveal=0.1 a difference in what was
        revealed rather than in how the preamble was made.
        """
        l0 = build_episode(self.fam, 1, 9, spec_for_level(Level.L0, T=5))
        l1 = build_episode(self.fam, 1, 9, spec_for_level(Level.L1, T=5))
        n0 = l0.channel.count(Channel.PREAMBLE)
        n1 = l1.channel.count(Channel.PREAMBLE)
        self.assertEqual(n0, n1)
        self.assertNotEqual(l0.tokens, l1.tokens)

    def test_reveal_fraction_is_monotone_in_how_much_is_stated(self):
        fam, rng = self.fam, Random(0)
        theta = fam.sample_theta(1, rng)
        enc = fam.sample_encoding(rng)
        counts = []
        for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
            r = build_reveal(fam, theta, enc, frac, Random(4))
            counts.append(r.n_revealed)
            self.assertTrue(r.consistent(theta), "the truth must satisfy its own reveal")
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(counts[0], 0)
        self.assertEqual(counts[-1], fam.d)

    def test_a_family_without_partial_reveal_refuses_the_interior(self):
        """Loud, not rounded to the nearest endpoint.

        Silently rounding would make the sweep run four points and report a
        curve, which is the failure the sweep exists to avoid.
        """
        from repertoire.families import SHJTypeIFamily

        fam = SHJTypeIFamily()
        rng = Random(0)
        theta, enc = fam.sample_theta(1, rng), fam.sample_encoding(rng)
        with self.assertRaises(ProtocolGap):
            build_reveal(fam, theta, enc, 0.5, rng)
        # ...but the endpoints still work, so no existing family is broken.
        self.assertIsNotNone(build_reveal(fam, theta, enc, 1.0, rng).tokens)
        self.assertEqual(build_reveal(fam, theta, enc, 0.0, rng).tokens, [])

    def test_L2_requires_a_query_fn(self):
        with self.assertRaises(ValueError):
            build_episode(self.fam, 1, 0, spec_for_level(Level.L2, T=3))

    def test_L2_supervises_the_query_channel_against_q_star(self):
        fam = self.fam
        spec = EpisodeSpec(T=4, query_source=QuerySource.MODEL)

        def perfect(tokens, history, enc):
            # A model that always emits exactly q*: the query channel target
            # must then equal what it emitted, and no query is malformed.
            q = fam.teacher_query(_theta_of(fam, 1, spec, 0), history)
            return q, fam.render(enc, q)

        ep = build_episode(fam, 1, 0, spec, query_fn=perfect)
        self.assertEqual(ep.malformed_queries, 0)
        for i, ch in enumerate(ep.channel):
            if ch is Channel.QUERY:
                self.assertTrue(ep.supervised[i])
                self.assertEqual(ep.tokens[i], ep.targets[i])

    def test_L2_malformed_query_gets_ERR_and_still_supervises_the_channel(self):
        fam = self.fam

        def garbage(tokens, history, enc):
            return None, [vocab.PAD, vocab.PAD]

        ep = build_episode(fam, 1, 0, EpisodeSpec(T=4, query_source=QuerySource.MODEL),
                           query_fn=garbage)
        self.assertEqual(ep.malformed_queries, 4)
        self.assertIn(vocab.ERR, ep.tokens)
        # The whole point: a bad query still costs something on the query
        # channel, and the target there is the teacher's query.
        q_positions = [i for i, c in enumerate(ep.channel) if c is Channel.QUERY]
        self.assertTrue(all(ep.supervised[i] for i in q_positions))
        self.assertTrue(any(ep.tokens[i] != ep.targets[i] for i in q_positions))
        # No answer channel at all, because nothing legal was asked.
        self.assertNotIn(Channel.ANSWER, ep.channel)

    def test_L3_posterior_targets_are_normalized_and_supervised(self):
        ep = build_episode(self.fam, 1, 0, spec_for_level(Level.L3, T=5))
        self.assertEqual(len(ep.posterior_targets), 5)
        for pos, dist in ep.posterior_targets.items():
            self.assertTrue(ep.supervised[pos])
            self.assertAlmostEqual(sum(dist.values()), 1.0, places=9)
            for tok in dist:
                self.assertIn(tok, vocab.SYMBOL_IDS)


def _theta_of(fam, k, spec, seed):
    """Reconstruct the theta an episode will use, for building a perfect prober."""
    rng = Random(episode_seed(fam.name, k, spec.key, seed))
    return fam.sample_theta(k, rng)


class TestPosteriorRouting(unittest.TestCase):
    """Handoff section 2.1: the decision, and the guard that outlives it."""

    def test_explicit_query_route_is_used_when_available(self):
        fam = StubLookupFamily(d=6)
        answer_distribution(fam, [], Query(0), 1)
        self.assertEqual(POSTERIOR_ROUTE["StubLookupFamily"], "explicit-query")

    def test_legacy_trailing_none_route_still_works(self):
        """A family on the old signature must not break, and must be visible."""

        class Legacy(StubLookupFamily):
            name = "stub_legacy"

            def posterior(self, history, k):  # the section 7 signature
                pending = history[-1][0] if history and history[-1][1] is None else None
                return StubLookupFamily.posterior(self, history[:-1] if pending else history,
                                                  k, query=pending)

        fam = Legacy(d=6)
        got = answer_distribution(fam, [(Query(2), Answer(1))], Query(2), 1)
        self.assertEqual(POSTERIOR_ROUTE["Legacy"], "trailing-none")
        self.assertAlmostEqual(got[1], 1.0)

    def test_query_sensitivity_detects_a_family_returning_the_marginal(self):
        """docs/03 finding 1, as an executable check rather than a warning.

        The marginal-returning family is the *exact* failure described: parity is
        identified after one observation and its query-marginal is 0.5/0.5, so a
        harness using the marginal trains maximal uncertainty about a rule
        already pinned down.
        """
        fam = StubLookupFamily(d=6)
        sensitive, _ = check_query_sensitivity(fam, 1, Random(0))
        self.assertTrue(sensitive)

        class Marginal(StubLookupFamily):
            name = "stub_marginal"

            def posterior(self, history, k, query=None):
                return {0: 0.5, 1: 0.5}  # the marginal, whatever is asked

        sensitive, detail = check_query_sensitivity(Marginal(d=6), 1, Random(0))
        self.assertFalse(sensitive)
        self.assertIn("marginal", detail)


class TestExactTargets(unittest.TestCase):
    def test_family_posterior_matches_brute_force_enumeration(self):
        """Task Spec section 8 step 2's gate, stated generically.

        The two are computed differently -- the family reads its history
        directly, the enumeration weights every hypothesis by likelihood -- so
        agreement is evidence rather than tautology.
        """
        fam = StubLookupFamily(d=6)
        rng = Random(11)
        for _ in range(8):
            theta = fam.sample_theta(1, rng)
            history = []
            for _ in range(4):
                q = fam.sample_query(theta, history, rng)
                mine = answer_distribution(fam, history, q, 1)
                brute = brute_force_answer_distribution(fam, 1, history, q)
                for key in set(mine) | set(brute):
                    self.assertAlmostEqual(mine.get(key, 0.0), brute.get(key, 0.0), places=9)
                history.append((q, draw_answer(fam, theta, q, rng)))

    def test_reveal_shrinks_the_hypothesis_space(self):
        fam = StubLookupFamily(d=6)
        rng = Random(2)
        theta, enc = fam.sample_theta(1, rng), fam.sample_encoding(rng)
        sizes = []
        for frac in (0.0, 0.5, 1.0):
            reveal = build_reveal(fam, theta, enc, frac, Random(5))
            sizes.append(belief_state(fam, 1, [], reveal).n_alive)
        self.assertEqual(sizes, sorted(sizes, reverse=True))
        self.assertEqual(sizes[0], 2 ** fam.d)
        self.assertEqual(sizes[-1], 1)

    def test_residual_entropy_matches_the_closed_form(self):
        """The stub exists so the sweep's x-axis can be checked against arithmetic.

        At L1 with nothing revealed, the first query's answer is a coin flip, so
        the rule entropy is exactly log 2. If this drifts, the sweep's x-axis is
        wrong on every family, including the ones with no closed form to check it
        against.
        """
        fam = StubLookupFamily(d=6)
        rep = measure_residual_entropy(fam, 1, spec_for_level(Level.L1, T=6), n_episodes=24)
        self.assertAlmostEqual(rep.rule_per_trial[0], math.log(2), places=9)
        self.assertTrue(rep.rule_per_trial[0] > rep.rule_per_trial[-1])

    def test_L0_has_no_residual_entropy_at_all(self):
        fam = StubLookupFamily(d=6)
        rep = measure_residual_entropy(fam, 1, spec_for_level(Level.L0, T=6), n_episodes=16)
        self.assertAlmostEqual(rep.rule, 0.0, places=12)
        self.assertAlmostEqual(rep.notation_upper, 0.0, places=12)

    def test_notation_term_falls_as_the_encoding_gets_used(self):
        """The finding the floor being wrong produced.

        The encoding is a hidden latent too (section 1: e is sampled on the same
        footing as theta), so early trials owe nats to not knowing which token
        names which answer. That term is large at trial 1 and shrinks as symbols
        appear -- and if it were ignored, the model would be reported as far from
        optimal when it is not.
        """
        fam = StubLookupFamily(d=6)
        rep = measure_residual_entropy(fam, 1, spec_for_level(Level.L1, T=8), n_episodes=24)
        self.assertGreater(rep.notation_per_trial[0], rep.rule_per_trial[0])
        self.assertGreater(rep.notation_per_trial[0], rep.notation_per_trial[-1])

    def test_prior_check_accepts_the_stub_whose_sampler_is_uniform(self):
        ok, detail = check_prior_matches_sampler(StubLookupFamily(d=5), 1, Random(0))
        self.assertTrue(ok, detail)

    def test_prior_check_rejects_a_sampler_that_does_not_match(self):
        """It has to be able to fail, and here is the case that makes it.

        A family whose sample_theta is biased while its enumeration is uniform is
        exactly the silent error this catches: every exact target computed from
        the enumeration is exact for a family we are not training on.
        """

        class Biased(StubLookupFamily):
            name = "stub_biased"

            def sample_theta(self, k, rng):
                if rng.random() < 0.9:  # nearly always all-zeros
                    return type(super().sample_theta(k, rng))(tuple([0] * self.d))
                return super().sample_theta(k, rng)

        ok, detail = check_prior_matches_sampler(Biased(d=5), 1, Random(0))
        self.assertFalse(ok, detail)


class TestSpec(unittest.TestCase):
    def test_reveal_outside_the_unit_interval_is_rejected(self):
        with self.assertRaises(ValueError):
            EpisodeSpec(reveal=1.5)
        with self.assertRaises(ValueError):
            EpisodeSpec(reveal=-0.01)

    def test_spec_key_separates_target_modes(self):
        a = EpisodeSpec(T=4, target_mode=TargetMode.REALIZED)
        b = EpisodeSpec(T=4, target_mode=TargetMode.POSTERIOR)
        self.assertNotEqual(a.key, b.key)


if __name__ == "__main__":
    unittest.main()
