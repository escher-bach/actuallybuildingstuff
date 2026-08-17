"""Contracts for learner-conditioned collection.

The stub policy below exists so the three transition cases in
STANDARD-LLM-STACK-MIGRATION-PLAN.md §7.3 can be forced deterministically. A
real checkpoint reaches them only by chance, and a case reached by chance is a
case that silently stops being tested.
"""
from __future__ import annotations

import tomllib
import unittest
from pathlib import Path

import torch

from step1_experiments.data import ACTION, BOS, END_TURN, EOS, OBS, Sequence, collate, encode_bytes
from step1_experiments.learner import (
    MALFORMED_ATTEMPT,
    CollectionCounters,
    CollectionSettings,
    _supervised_example,
    _teacher_target,
    assert_collection_contract,
    collect_tranche,
)


CONFIG = Path(__file__).resolve().parents[2] / "configs" / "kaggle" / "t4x2_dense_seed0.toml"
SEED = 20260811 + 4_000_000


def _world() -> dict:
    with CONFIG.open("rb") as handle:
        return tomllib.load(handle)["world"]


def _settings(**overrides) -> CollectionSettings:
    base = {"max_turns": 6, "max_action_tokens": 32, "max_consecutive_failures": 2,
            "context_length": 2048, "temperature": 0.0}
    return CollectionSettings(**{**base, **overrides})


class StubPolicy:
    """A `generate`-compatible stand-in that replays scripted action texts.

    It emits the next scripted text for every row, so a test can drive the
    whole tranche into malformed, invalid, or terminal behaviour on demand.
    """

    def __init__(self, script: list[str]):
        self.script = script
        self.turn = 0
        self.config = type("Config", (), {"pad_token_id": 256, "max_position_embeddings": 2048})()

    def parameters(self):
        yield torch.zeros(1)

    def generate(self, *, input_ids, attention_mask, max_new_tokens, **_kwargs):
        text = self.script[min(self.turn, len(self.script) - 1)]
        self.turn += 1
        tokens = encode_bytes(text) + [END_TURN]
        if len(tokens) > max_new_tokens:
            raise AssertionError("the scripted action does not fit the allowance under test")
        row = torch.tensor(tokens, dtype=torch.long)
        rows = row.repeat(input_ids.shape[0], 1)
        return torch.cat([input_ids, rows], dim=1)


class TeacherTargetGuard(unittest.TestCase):
    N_PROBE = 5

    def test_a_licensed_commit_is_supervised(self) -> None:
        targets = {"preferred_actions": {self.N_PROBE + 2}, "licenses_commitment": True}
        self.assertEqual(_teacher_target(targets, self.N_PROBE), self.N_PROBE + 2)

    def test_an_unlicensed_commit_only_target_is_refused(self) -> None:
        targets = {"preferred_actions": {self.N_PROBE + 2}, "licenses_commitment": False}
        self.assertIsNone(_teacher_target(targets, self.N_PROBE))

    def test_the_refusal_is_what_changes_the_outcome(self) -> None:
        """Without the guard this state would supervise a commit derived from truth."""
        targets = {"preferred_actions": {self.N_PROBE + 2}, "licenses_commitment": False}
        self.assertEqual(min(targets["preferred_actions"]), self.N_PROBE + 2)
        self.assertIsNone(_teacher_target(targets, self.N_PROBE))

    def test_a_probe_tied_with_a_commit_is_preferred_without_special_casing(self) -> None:
        targets = {"preferred_actions": {3, self.N_PROBE + 1}, "licenses_commitment": False}
        self.assertEqual(_teacher_target(targets, self.N_PROBE), 3)

    def test_a_terminated_state_supervises_nothing(self) -> None:
        self.assertIsNone(_teacher_target({"preferred_actions": set(), "licenses_commitment": False}, self.N_PROBE))


class MalformedSentinel(unittest.TestCase):
    def test_the_sentinel_really_is_unparseable_in_both_renderings(self) -> None:
        from world_py import parse_action

        world = _world()
        for rendering in ("a", "b"):
            with self.assertRaises(ValueError, msg=f"the stand-in parses under rendering {rendering}"):
                parse_action(MALFORMED_ATTEMPT, world["n_probe"], world["n_hyp"], rendering)


class SupervisedLayout(unittest.TestCase):
    def test_only_the_correction_span_carries_loss(self) -> None:
        context = [BOS, OBS, *encode_bytes("state"), ACTION]
        example = _supervised_example(context, "inspect(2)")
        self.assertEqual(example.tokens[:len(context)], context)
        self.assertEqual(example.loss[:len(context)], [0] * len(context))
        self.assertEqual(example.tokens[-1], EOS)
        self.assertEqual(example.loss[-1], 0)
        supervised = [t for t, flag in zip(example.tokens, example.loss) if flag]
        self.assertEqual(supervised, encode_bytes("inspect(2)") + [END_TURN])

    def test_the_standard_label_path_masks_the_on_policy_prefix(self) -> None:
        example = _supervised_example([BOS, OBS, ACTION], "commit(1)")
        batch = collate([example], 2048)
        labels = batch["labels"][0].tolist()
        self.assertEqual(labels[:3], [-100, -100, -100])
        self.assertEqual([value for value in labels if value != -100], encode_bytes("commit(1)") + [END_TURN])


class TransitionCases(unittest.TestCase):
    """The three cases §7.3 requires, each forced rather than hoped for."""

    def _collect(self, script: list[str], **overrides):
        return collect_tranche(
            StubPolicy(script), _world(), [SEED, SEED + 1], "a",
            _settings(**overrides), torch.device("cpu"),
        )

    def test_a_malformed_attempt_leaves_the_state_unchanged_and_is_still_labelled(self) -> None:
        examples, traces, counters = self._collect([MALFORMED_ATTEMPT], max_consecutive_failures=3)
        self.assertEqual(counters.world_transitions, 0)
        self.assertGreater(counters.malformed_attempts, 0)
        self.assertGreater(counters.states_supervised, 0)
        self.assertTrue(all(trace.end_reason == "repeated_failure" for trace in traces))
        # The state never advanced, so every target is the opening state's.
        first = examples[0].tokens
        self.assertEqual(first[0], BOS)

    def test_an_out_of_range_identifier_is_malformed_at_the_parser(self) -> None:
        world = _world()
        _examples, _traces, counters = self._collect(
            [f"inspect(probe_{world['n_probe'] + 4})"], max_consecutive_failures=2,
        )
        self.assertEqual(counters.world_transitions, 0)
        self.assertEqual(counters.invalid_attempts, 0)
        self.assertEqual(counters.malformed_attempts, counters.turns)

    def test_a_well_formed_illegal_attempt_is_invalid_rather_than_malformed(self) -> None:
        """Re-probing: the parser accepts it, the world refuses it."""
        _examples, _traces, counters = self._collect(
            ["inspect(probe_1)", "inspect(probe_1)"], max_turns=2, max_consecutive_failures=2,
        )
        self.assertEqual(counters.accepted_attempts, 2)
        self.assertEqual(counters.world_transitions, 2)
        self.assertEqual(counters.invalid_attempts, 2)
        self.assertEqual(counters.malformed_attempts, 0)

    def test_a_terminal_attempt_emits_no_recovery_label(self) -> None:
        examples, traces, counters = self._collect(["commit(cause_1)"])
        self.assertEqual(counters.episodes_terminated, 2)
        self.assertEqual(counters.turns, 2)
        # One state each: the opening state. Nothing is labelled after a commit.
        self.assertEqual(counters.states_supervised, len(examples))
        self.assertTrue(all(trace.turns == 1 for trace in traces))

    def test_recovery_after_a_failed_attempt_is_counted(self) -> None:
        _examples, _traces, counters = self._collect(
            [MALFORMED_ATTEMPT, "commit(cause_1)"], max_consecutive_failures=3,
        )
        self.assertEqual(counters.malformed_attempts, 2)
        self.assertEqual(counters.failures_recovered, 2)

    def test_the_teacher_action_never_enters_the_rollout_context(self) -> None:
        """Every supervised span must be the last thing in its own example."""
        examples, _traces, _counters = self._collect([MALFORMED_ATTEMPT], max_consecutive_failures=3)
        for example in examples:
            supervised_positions = [i for i, flag in enumerate(example.loss) if flag]
            self.assertEqual(
                supervised_positions,
                list(range(supervised_positions[0], len(example.tokens) - 1)),
                "a supervised span is followed by more rollout, so a teacher action re-entered context",
            )


class DecoderAgreement(unittest.TestCase):
    """Collection must visit the states the frozen evaluator visits.

    The collector has its own decoder because it needs the raw tokens for the
    on-policy context, which the evaluator's does not return. Duplication is
    only safe if the two are held to producing the same action, so that is
    asserted against a real model rather than argued from the source.

    The claim is scoped to prefixes away from the context limit. Within one
    token of it the two allowances differ by one, because the collector
    reserves a position for the `EOS` it packs and the evaluator packs none.
    Collection refuses an episode well before that point.
    """

    @staticmethod
    def _small_model():
        """A deliberately tiny GPT-NeoX: 96 sequential decode steps on the real
        19.2M config cost over a minute, and the property under test is the
        decode path, not the architecture. The vocabulary and special ids are
        the frozen ones because those the decoders do depend on."""
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

        return GPTNeoXForCausalLM(GPTNeoXConfig(
            vocab_size=262, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=2, max_position_embeddings=2048,
            pad_token_id=256, bos_token_id=257, eos_token_id=258, use_cache=False,
        )).eval()

    def test_the_collector_and_the_evaluator_decode_the_same_action(self) -> None:
        from step1_experiments.evaluate import _decode_actions_batched
        from step1_experiments.learner import _decode_actions
        from world_py import Batch

        from step1_experiments.data import _family

        torch.manual_seed(20260811)
        model = self._small_model()
        device = torch.device("cpu")
        world = _world()
        prefixes = []
        for index in range(3):
            batch = Batch(_family(world), seed=SEED + index, n_episodes=1)
            prefixes.append([BOS, OBS, *encode_bytes(batch.observations("a")[0]), ACTION])

        evaluator = _decode_actions_batched(model, prefixes, device)
        collector = _decode_actions(model, prefixes, _settings(max_action_tokens=96), device)
        self.assertEqual([text for _tokens, text in collector], evaluator)


class CollectionContract(unittest.TestCase):
    def setUp(self) -> None:
        self.example = _supervised_example([BOS, OBS, ACTION], "inspect(1)")
        self.counters = CollectionCounters(states_supervised=1)

    def test_a_well_formed_tranche_passes(self) -> None:
        assert_collection_contract([self.example], self.counters)

    def test_an_empty_tranche_fails(self) -> None:
        with self.assertRaisesRegex(AssertionError, "no supervised examples"):
            assert_collection_contract([], self.counters)

    def test_a_miscounted_tranche_fails(self) -> None:
        with self.assertRaisesRegex(AssertionError, "but packed"):
            assert_collection_contract([self.example], CollectionCounters(states_supervised=2))

    def test_supervising_the_on_policy_prefix_fails(self) -> None:
        broken = Sequence(self.example.tokens, [1] * len(self.example.tokens), self.example.channels)
        with self.assertRaisesRegex(AssertionError, "prefix or the trailing EOS"):
            assert_collection_contract([broken], self.counters)

    def test_a_span_that_does_not_end_at_end_turn_fails(self) -> None:
        tokens = [BOS, OBS, ACTION, *encode_bytes("inspect(1)"), EOS]
        loss = [0, 0, 0] + [1] * len(encode_bytes("inspect(1)")) + [0]
        with self.assertRaisesRegex(AssertionError, "does not end at END_TURN"):
            assert_collection_contract([Sequence(tokens, loss, loss)], self.counters)


if __name__ == "__main__":
    unittest.main()
