"""Learner-conditioned collection: the teacher labels states the learner reached.

STEP-1.md §5.3's secondary generation regime, and the environment adapter
STANDARD-LLM-STACK-MIGRATION-PLAN.md §7 scopes.  The Rust boundary owns parsing
and independent per-episode transitions; this module owns only the orchestration
and the token layout.  Optimization stays in Trainer: this is DAgger-style data
collection, not policy-gradient RL.

Three properties are load-bearing and easy to lose:

**The teacher never enters the rollout context.**  At each state the learner
reaches, one supervised example is emitted -- the on-policy history so far, then
the teacher's action for that state -- and the *episode* continues on the
learner's own action instead.  A teacher action appended to the running context
would make the next state teacher-conditioned, which is the entire variable this
arm manipulates.

**A commit target is dropped where the evidence does not license it.**  `teach`
reads `inst.truth`.  On the teacher's own trajectory that never shows, because
it only commits from a state whose history already pins the answer down.  Off
that trajectory -- which is where this module deliberately operates -- it will
propose `Commit(truth)` from states where no visible evidence supports it.
Supervising that would teach exactly the premature commitment this arm exists to
reduce, so it is refused and counted.  See `step1/audit/world/` for the measured
rate.  Note that `min(preferred_actions)` already prefers a probe over a commit
when the teacher ties them, so the guard changes one thing only: it refuses the
case where every preferred action is an unlicensed commit.

**Collection decodes exactly as the evaluator does.**  Greedy, `END_TURN`-
terminated, same allowance.  The states supervised are then the states the
scored policy actually visits; `tests/test_learner_conditioned.py` asserts the
two decoders agree rather than trusting the duplication.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch

from .data import ACTION, BOS, END_TURN, EOS, OBS, PAD, Sequence, encode_bytes


# The action channel, matching data.generate_world_sequences' packed shards so a
# learner-conditioned example and a teacher-conditioned one carry the same
# channel identifier for the same kind of span.
ACTION_CHANNEL = 1

# The Rust boundary takes a `String`, so an attempt whose bytes are not UTF-8
# cannot be handed over verbatim.  This stand-in reaches the world instead, where
# it must fail to parse -- which is the correct semantics for a malformed attempt
# (state unchanged, target from the unchanged state) and is asserted in the
# tests rather than assumed.
MALFORMED_ATTEMPT = "<malformed>"


@dataclass
class CollectionCounters:
    """Budget axes STEP-1 §6 requires reported alongside optimizer updates."""

    episodes: int = 0
    turns: int = 0
    world_transitions: int = 0
    generated_action_tokens: int = 0
    observation_tokens: int = 0
    supervised_correction_tokens: int = 0
    states_supervised: int = 0
    # The guard above. A nonzero value here is the arm working, not failing.
    states_refused_unlicensed_commit: int = 0
    accepted_attempts: int = 0
    malformed_attempts: int = 0
    invalid_attempts: int = 0
    # P3: a failed attempt followed by an accepted one from the same unchanged
    # state. Recovery is invisible to the frozen evaluator, which ends an
    # episode at its first protocol failure.
    failures_recovered: int = 0
    failures_unrecovered: int = 0
    episodes_terminated: int = 0
    episodes_exhausted_turns: int = 0
    episodes_exhausted_context: int = 0
    episodes_abandoned_repeated_failure: int = 0

    def report(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class CollectionSettings:
    """Bounds that keep an unproductive policy from consuming a whole session."""

    max_turns: int
    max_action_tokens: int
    max_consecutive_failures: int
    context_length: int
    temperature: float = 0.0
    # "learner" is the arm; "teacher" is its control. Under "teacher" the world
    # advances on the teacher's own action instead of the model's, so the states
    # visited are teacher states while the packing, the masking, the seeds, the
    # optimizer, and the step size stay bit-identical. It is the only way to tell
    # a result about the state distribution apart from an artefact of this
    # pipeline, and it must never be the default.
    policy: str = "learner"

    @staticmethod
    def from_config(config: dict) -> "CollectionSettings":
        collection = config["collection"]
        settings = CollectionSettings(
            max_turns=collection["max_turns"],
            max_action_tokens=collection["max_action_tokens"],
            max_consecutive_failures=collection["max_consecutive_failures"],
            context_length=config["world"]["context_length"],
            temperature=collection.get("temperature", 0.0),
            policy=collection.get("policy", "learner"),
        )
        if settings.policy not in ("learner", "teacher"):
            raise ValueError(f"collection.policy must be 'learner' or 'teacher', got {settings.policy!r}")
        if settings.max_turns < 1 or settings.max_action_tokens < 1:
            raise ValueError("collection needs at least one turn and one action token")
        if settings.max_consecutive_failures < 1:
            raise ValueError("a policy must be allowed at least one failed attempt per state")
        if settings.max_action_tokens >= settings.context_length:
            raise ValueError("a single action may not be allowed the whole context")
        return settings


@dataclass
class EpisodeTrace:
    """Replay-sufficient record of one collected episode."""

    world_seed: int
    turns: int = 0
    supervised_states: int = 0
    refused_states: int = 0
    attempts: list[str] = field(default_factory=list)
    end_reason: str = "running"


def _teacher_target(targets: dict, n_probe: int) -> int | None:
    """The action to supervise at this state, or None if it must be refused.

    `min` over the encoded action set already prefers `Inspect` (encoded
    `0..n_probe`) over `Commit` (`n_probe..`), which is the same tie-break the
    teacher-conditioned shard generator uses.  So `min(...) >= n_probe` means
    every preferred action is a commit, and an unlicensed commit is truth.
    """
    preferred = targets["preferred_actions"]
    if not preferred:
        return None
    chosen = min(preferred)
    if chosen >= n_probe and not targets["licenses_commitment"]:
        return None
    return chosen


@torch.no_grad()
def _decode_actions(model, contexts: list[list[int]], settings: CollectionSettings,
                    device) -> list[tuple[list[int], str | None]]:
    """Batched continuation of each live episode's own history.

    Returns the raw generated tokens as well as the parsed text: the tokens are
    what goes back into the on-policy context, including when they do not decode
    to anything the parser accepts.  Rows are bucketed by their own remaining
    allowance so a near-context episode cannot shorten another's action.
    """
    results: list[tuple[list[int], str | None]] = [([], None)] * len(contexts)
    buckets: dict[int, list[int]] = {}
    for index, context in enumerate(contexts):
        allowance = min(settings.max_action_tokens, settings.context_length - len(context) - 1)
        if allowance >= 1:
            buckets.setdefault(allowance, []).append(index)
    sampling = ({"do_sample": True, "temperature": settings.temperature, "top_p": 1.0, "top_k": 0}
                if settings.temperature > 0 else {"do_sample": False})
    for allowance, rows in buckets.items():
        for start in range(0, len(rows), 32):
            group = rows[start:start + 32]
            values = [contexts[index] for index in group]
            width = max(len(value) for value in values)
            input_ids = torch.full((len(group), width), PAD, dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(group), width), dtype=torch.bool, device=device)
            for row, value in enumerate(values):
                input_ids[row, width - len(value):] = torch.tensor(value, dtype=torch.long, device=device)
                attention_mask[row, width - len(value):] = True
            generated = model.generate(
                input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=allowance,
                eos_token_id=END_TURN, pad_token_id=PAD, use_cache=True, **sampling,
            )
            for index, row in zip(group, generated):
                results[index] = _split_attempt(row[width:].tolist())
    return results


def _split_attempt(generated: list[int]) -> tuple[list[int], str | None]:
    """Cut the generation at END_TURN and decode the payload if it is bytes."""
    if END_TURN in generated:
        stop = generated.index(END_TURN)
        tokens, payload = generated[:stop + 1], generated[:stop]
    else:
        # No END_TURN inside the allowance: the attempt is unterminated and
        # therefore malformed, but the tokens still happened and stay in context.
        tokens, payload = list(generated), None
    if payload is None or any(token >= 256 for token in payload):
        return tokens, None
    try:
        return tokens, bytes(payload).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return tokens, None


def _supervised_example(context: list[int], correction: str) -> Sequence:
    """One on-policy prefix, then the teacher's action for that exact state."""
    correction_tokens = encode_bytes(correction) + [END_TURN]
    tokens = [*context, *correction_tokens, EOS]
    loss = [0] * len(context) + [1] * len(correction_tokens) + [0]
    return Sequence(tokens, loss, [0] * len(context) + [ACTION_CHANNEL] * len(correction_tokens) + [0])


@torch.no_grad()
def collect_tranche(model, params: dict, seeds: list[int], rendering: str,
                    settings: CollectionSettings, device) -> tuple[list[Sequence], list[EpisodeTrace], CollectionCounters]:
    """Play `seeds` under the declared policy and label every state it reaches."""
    from world_py import Batch, render_action

    from .data import _family

    if settings.policy not in ("learner", "teacher"):
        raise ValueError(f"unknown collection policy {settings.policy!r}")

    n_probe = params["n_probe"]
    counters = CollectionCounters()
    worlds = [Batch(_family(params), seed=seed, n_episodes=1) for seed in seeds]
    traces = [EpisodeTrace(world_seed=seed) for seed in seeds]
    histories: list[list[int]] = [[BOS] for _ in seeds]
    consecutive_failures = [0] * len(seeds)
    examples: list[Sequence] = []
    live = list(range(len(seeds)))
    counters.episodes = len(seeds)

    for _turn in range(settings.max_turns):
        if not live:
            break
        contexts, active = [], []
        for index in live:
            observation = [OBS, *encode_bytes(worlds[index].observations(rendering)[0]), ACTION]
            context = histories[index] + observation
            # The example emitted here is context + correction + EOS; refuse the
            # episode rather than truncate, which is never permitted.
            if len(context) + settings.max_action_tokens + 2 > settings.context_length:
                traces[index].end_reason = "context_exhausted"
                counters.episodes_exhausted_context += 1
                continue
            counters.observation_tokens += len(observation)
            contexts.append(context)
            active.append(index)
        if not active:
            break

        # Supervise before acting: the target belongs to this state, and the
        # learner's own action must not be able to influence it.
        for index, context in zip(active, contexts):
            targets = worlds[index].privileged_teacher_targets()[0]
            action = _teacher_target(targets, n_probe)
            if action is None:
                counters.states_refused_unlicensed_commit += 1
                traces[index].refused_states += 1
                continue
            correction = render_action(action, n_probe, params["n_hyp"], rendering)
            example = _supervised_example(context, correction)
            if len(example.tokens) > settings.context_length:
                # The reservation above is meant to make this unreachable; if it
                # ever fires the bound is wrong and silence would truncate.
                raise AssertionError(
                    f"supervised example of {len(example.tokens)} tokens exceeds the frozen context"
                )
            examples.append(example)
            counters.states_supervised += 1
            counters.supervised_correction_tokens += sum(example.loss)
            traces[index].supervised_states += 1

        if settings.policy == "teacher":
            # The control: the world advances on the teacher's own action, so
            # the next state is a teacher state. Everything downstream -- the
            # packing, the mask, the counters, the contract -- is untouched.
            attempts = []
            for index in active:
                action = _teacher_target(worlds[index].privileged_teacher_targets()[0], n_probe)
                if action is None:
                    attempts.append((encode_bytes(MALFORMED_ATTEMPT) + [END_TURN], MALFORMED_ATTEMPT))
                    continue
                text = render_action(action, n_probe, params["n_hyp"], rendering)
                attempts.append((encode_bytes(text) + [END_TURN], text))
        else:
            attempts = _decode_actions(model, contexts, settings, device)
        still_live = []
        for index, context, (tokens, text) in zip(active, contexts, attempts):
            counters.turns += 1
            counters.generated_action_tokens += len(tokens)
            traces[index].turns += 1
            histories[index] = context + tokens
            record = worlds[index].step_attempts([text if text is not None else MALFORMED_ATTEMPT], rendering)[0]
            traces[index].attempts.append(record["learner_text"])
            if text is None or record["parsed_action"] is None:
                counters.malformed_attempts += 1
                accepted = False
            elif not record["accepted"]:
                counters.invalid_attempts += 1
                accepted = False
            else:
                counters.accepted_attempts += 1
                counters.world_transitions += 1
                accepted = True
            if accepted:
                if consecutive_failures[index]:
                    counters.failures_recovered += 1
                consecutive_failures[index] = 0
                if worlds[index].done()[0]:
                    traces[index].end_reason = "terminated"
                    counters.episodes_terminated += 1
                    continue
                still_live.append(index)
                continue
            consecutive_failures[index] += 1
            if consecutive_failures[index] >= settings.max_consecutive_failures:
                counters.failures_unrecovered += 1
                traces[index].end_reason = "repeated_failure"
                counters.episodes_abandoned_repeated_failure += 1
                continue
            still_live.append(index)
        live = still_live

    for index in live:
        traces[index].end_reason = "turns_exhausted"
        counters.episodes_exhausted_turns += 1
    return examples, traces, counters


def assert_collection_contract(examples: list[Sequence], counters: CollectionCounters) -> None:
    """Refuse a tranche that would train something other than what it claims."""
    if not examples:
        raise AssertionError("collection produced no supervised examples")
    if counters.states_supervised != len(examples):
        raise AssertionError(
            f"counted {counters.states_supervised} supervised states but packed {len(examples)} examples"
        )
    for example in examples:
        if not (len(example.tokens) == len(example.loss) == len(example.channels)):
            raise AssertionError("collected example fields are misaligned")
        if not any(example.loss):
            raise AssertionError("a collected example supervises no tokens at all")
        if example.loss[0] or example.loss[-1]:
            raise AssertionError("the on-policy prefix or the trailing EOS is being supervised")
        supervised = [token for token, flag in zip(example.tokens, example.loss) if flag]
        if supervised[-1] != END_TURN:
            raise AssertionError("a supervised span does not end at END_TURN")
        if any(token >= 256 for token in supervised[:-1]):
            raise AssertionError("a supervised span contains a transport token, not action bytes")
