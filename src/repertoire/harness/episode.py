"""Episode construction, the level wrappers, and the loss mask.

Task Spec section 8 step 1.  One generator, four wrappers, and the level is a
runtime argument -- section 6 says outright that "if your implementation requires
four generators, the interface is wrong."

--------------------------------------------------------------------------
The level is not primitive here, and that is a finding
--------------------------------------------------------------------------

Section 2 names four levels.  Implementing them against section 8 step 5, which
asks for a *continuous* sweep, forces the question of what the four are made of.
They decompose into three independent settings:

    reveal        how much of theta the preamble states           [0, 1]
    query_source  who chooses x_t                     sampled | model
    target_mode   what the answer channel is scored against
                                                     realized | posterior

    L0 = (1.0, sampled, realized)      L2 = (0.0, model,   realized)
    L1 = (0.0, sampled, realized)      L3 = (0.0, sampled, posterior)

Two things fall out of writing it this way, neither visible from section 2.

**The L0-L1 dial is `reveal`, and it is continuous.**  That is the sweep, and it
is the reason `EpisodeSpec` exists rather than a `Level` enum with four branches.

**L3 conflates two separable things.**  Section 2 defines L3 as "withheld; not
identifiable within the episode", which is a statement about the *episode* -- T
is too short, or Theta too large, for the history to pin theta down.  But what
makes L3 supervision different in the loop is that the target is the exact
posterior rather than the realized token, and that is available at *every*
level.  Nothing stops an L1 episode from being scored against the exact Bayes
posterior; it is strictly better supervision than a sample from it, since it is
the same target with the sampling noise removed.  Identifiability and target mode
are orthogonal, and section 2 ties them together.

The harness keeps them separate and `Level.L3` maps to the section 2 combination,
so nothing downstream changes.  Recorded in docs/10.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from random import Random
from typing import Any, Callable, Sequence

from .. import vocab
from ..form import Level
from .protocol import (
    ProtocolGap,
    Reveal,
    SupportsPartialReveal,
    answer_distribution,
    answer_type,
    draw_answer,
)

Theta = Any
Query = Any
Answer = Any


class Channel(IntEnum):
    """What a token is, so the mask and the per-trial aggregation can be checked.

    Carried per token rather than recomputed, because the loss mask is the thing
    most likely to be wrong in a way that still trains.  A mask that supervises
    the preamble produces a model that is better at reciting rules it was handed,
    and a loss curve that looks fine.
    """

    STRUCTURAL = 0  # BOS/SEP/EOS and the harness's own framing -- never supervised
    PREAMBLE = 1  # section 7: never supervised
    QUERY = 2  # supervised only at L2, against q*
    ANSWER = 3  # supervised
    TRACE = 4  # supervised where the family emits one (section 1.2)
    ORACLE_ECHO = 5  # never supervised; see note in `_emit_trial`
    OBSERVATION = 6  # an answer given as free evidence, deliberately not scored


class QuerySource(str, Enum):
    SAMPLED = "sampled"  # family.sample_query
    MODEL = "model"  # L2: the model emits it and the oracle answers what was asked


class TargetMode(str, Enum):
    REALIZED = "realized"  # cross-entropy against the token the oracle produced
    POSTERIOR = "posterior"  # cross-entropy against the exact Bayes distribution


@dataclass(frozen=True)
class EpisodeSpec:
    """The wrapper arguments. `T` lives here because section 7 says T is ours.

    Handoff section 2.4: "Episode length T is yours, and the register has numbers
    for it. Do not let families set it."
    """

    T: int = 8  # SCORED trials; n_free comes before these
    reveal: float = 0.0
    n_free: int = 0
    query_source: QuerySource = QuerySource.SAMPLED
    target_mode: TargetMode = TargetMode.REALIZED
    emit_trace: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.reveal <= 1.0:
            raise ValueError("reveal is a fraction of theta stated, in [0, 1]")
        if self.T < 1:
            raise ValueError("T must be at least 1")
        if self.n_free < 0:
            raise ValueError("n_free is a count of unscored leading observations")

    @property
    def total_trials(self) -> int:
        return self.n_free + self.T

    @property
    def key(self) -> str:
        """Stable identity, used in seeding so two specs never collide."""
        return (
            f"T{self.T}/f{self.n_free}/r{self.reveal:.6f}/{self.query_source.value}"
            f"/{self.target_mode.value}/{'tr' if self.emit_trace else 'no'}"
        )


def spec_for_level(level: Level, T: int = 8, emit_trace: bool = False) -> EpisodeSpec:
    """Section 2's four levels as settings of the three dials above."""
    if level is Level.L0:
        return EpisodeSpec(T=T, reveal=1.0, emit_trace=emit_trace)
    if level is Level.L1:
        return EpisodeSpec(T=T, reveal=0.0, emit_trace=emit_trace)
    if level is Level.L2:
        return EpisodeSpec(
            T=T, reveal=0.0, query_source=QuerySource.MODEL, emit_trace=emit_trace
        )
    if level is Level.L3:
        return EpisodeSpec(
            T=T, reveal=0.0, target_mode=TargetMode.POSTERIOR, emit_trace=emit_trace
        )
    raise ValueError(f"unknown level {level!r}")


@dataclass
class Episode:
    """One rendered episode: tokens, and everything needed to score them.

    All four arrays are the same length as `tokens` and are aligned to it.
    `targets[i]` is what *should* stand at position i; it equals `tokens[i]`
    everywhere except the L2 query channel, where the context carries the query
    the model actually asked and the target carries the teacher's q* (section
    2.1: two channels, and the oracle answers what was asked).
    """

    tokens: list[int]
    targets: list[int]
    supervised: list[bool]
    channel: list[Channel]
    trial_index: list[int]  # -1 outside a trial; used for the acquisition slope
    posterior_targets: dict[int, dict[int, float]] = field(default_factory=dict)

    family: str = ""
    k: int = 0
    seed: int = 0
    spec_key: str = ""
    encoding: str = ""
    n_trials: int = 0
    malformed_queries: int = 0  # A6: how often the model asked something illegal

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def n_supervised(self) -> int:
        return sum(self.supervised)

    def check(self) -> None:
        """Assertions the harness runs on every episode it builds.

        Cheap, and they exist because hazard 6 says a check that cannot fail is
        worse than none: these can fail, and `tests/test_harness_episode.py`
        contains the deliberately-broken episodes that make them fail.
        """
        n = len(self.tokens)
        if not (len(self.targets) == len(self.supervised) == len(self.channel) == len(self.trial_index) == n):
            raise AssertionError("episode arrays are not aligned")
        for i, tok in enumerate(self.tokens):
            if not 0 <= tok < vocab.VOCAB_SIZE:
                raise AssertionError(f"token {tok} at {i} is outside the vocabulary")
            if not 0 <= self.targets[i] < vocab.VOCAB_SIZE:
                raise AssertionError(f"target {self.targets[i]} at {i} is outside the vocabulary")
        for i, sup in enumerate(self.supervised):
            ch = self.channel[i]
            if sup and ch in (Channel.STRUCTURAL, Channel.PREAMBLE, Channel.ORACLE_ECHO,
                              Channel.OBSERVATION):
                raise AssertionError(
                    f"position {i} is supervised on channel {ch.name}; section 7 says "
                    "preamble and oracle-echo tokens are never supervised, and a free "
                    "observation is evidence rather than a target"
                )
            if not sup and ch is Channel.ANSWER:
                raise AssertionError(f"answer token at {i} is not supervised")
        for pos in self.posterior_targets:
            if not self.supervised[pos]:
                raise AssertionError(f"posterior target at unsupervised position {pos}")


# --------------------------------------------------------------------------
# Seeding -- section 7: "every episode reconstructible from (family, k, level, seed)"
# --------------------------------------------------------------------------


def episode_seed(family_name: str, k: int, spec_key: str, seed: int) -> int:
    """Derive the episode's rng seed. Stable across processes, which matters.

    Python's `hash()` on str is salted per process, so anything built on it is
    reproducible within a run and not across runs.  Section 7 calls seeded
    reconstructibility non-negotiable -- "the whole design depends on being able
    to re-run a branch after a backtrack" -- and a within-process-only guarantee
    would satisfy every test we would think to write while failing the one thing
    it is for.
    """
    payload = f"{family_name}|{k}|{spec_key}|{seed}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")


# --------------------------------------------------------------------------
# The reveal (the dial)
# --------------------------------------------------------------------------


def build_reveal(family: Any, theta: Theta, enc: Any, fraction: float, rng: Random) -> Reveal:
    """The preamble at an arbitrary reveal fraction.

    Endpoints fall back to section 7's all-or-nothing `preamble`, so every
    existing family runs at L0 and L1 unchanged.  Anything strictly between them
    needs `partial_preamble`, and a family without it gets an exception naming
    itself rather than a silent rounding to the nearest endpoint -- rounding
    would make the sweep quietly run four points and report a curve.

    **A family that has `partial_preamble` uses it at the endpoints too**, and
    the consequence is worth stating: its L1 episodes carry a full-length
    preamble in which every slot says "unknown", rather than no preamble at all.
    That is deliberate.  Section 2 calls the levels "four settings of one dial",
    and the endpoints of a dial have to be built the same way as its interior or
    the sweep's two ends are not on the same curve as its middle.  It costs
    unsupervised context tokens at L1 and changes no supervised token, so the
    per-token measurement is unaffected; what it buys is that a difference
    between reveal=0.0 and reveal=0.1 is a difference in *what was revealed* and
    not in how the preamble was constructed.
    """
    if fraction >= 1.0 and not isinstance(family, SupportsPartialReveal):
        tokens = family.preamble(theta, enc) or []
        return Reveal(list(tokens), n_slots=1, n_revealed=1, consistent=lambda t: t == theta)
    if fraction <= 0.0 and not isinstance(family, SupportsPartialReveal):
        return Reveal([], n_slots=1, n_revealed=0, consistent=lambda t: True)
    if not isinstance(family, SupportsPartialReveal):
        raise ProtocolGap(
            f"{type(family).__name__} has no partial_preamble, so reveal={fraction} "
            "is not expressible. Task Spec section 8 step 5 needs the interior of "
            "[0, 1]; see docs/10-harness-findings.md."
        )
    return family.partial_preamble(theta, enc, fraction, rng)


# --------------------------------------------------------------------------
# Building
# --------------------------------------------------------------------------

# A model-backed query channel (L2). Given the tokens emitted so far, the
# history and the episode's encoding, return the query the model asked and the
# tokens it emitted saying so. A None query is the A6 case: the model emitted
# something that does not name a legal query, and the oracle must refuse it
# well-formedly rather than the harness crashing or guessing.
#
# The encoding is passed because it is sampled inside `build_episode` and the
# callback needs it to parse -- the model emits tokens, and only the encoding
# says what they denote.
QueryFn = Callable[[list[int], list, Any], "tuple[Query | None, list[int]]"]


def build_episode(
    family: Any,
    k: int,
    seed: int,
    spec: EpisodeSpec,
    query_fn: QueryFn | None = None,
) -> Episode:
    """Render one episode. The only episode constructor in the harness.

    `query_fn` is required when `spec.query_source is MODEL` and forbidden
    otherwise, because L2's defining property is that the query comes from the
    model -- a default would make an L2 episode that is quietly an L1 episode.
    """
    if spec.query_source is QuerySource.MODEL and query_fn is None:
        raise ValueError("L2 needs a query_fn; the model is what chooses the query")

    rng = Random(episode_seed(family.name, k, spec.key, seed))
    theta = family.sample_theta(k, rng)
    enc = family.sample_encoding(rng)

    ep = Episode(
        tokens=[], targets=[], supervised=[], channel=[], trial_index=[],
        family=family.name, k=k, seed=seed, spec_key=spec.key,
        encoding=getattr(enc, "name", "default"), n_trials=spec.total_trials,
    )

    _emit(ep, [vocab.BOS], Channel.STRUCTURAL, -1)

    reveal = build_reveal(family, theta, enc, spec.reveal, rng)
    if reveal.tokens:
        _emit(ep, list(reveal.tokens), Channel.PREAMBLE, -1)
        _emit(ep, [vocab.SEP], Channel.STRUCTURAL, -1)

    history: list = []
    for t in range(spec.total_trials):
        _emit_trial(ep, family, theta, enc, k, spec, history, rng, t, query_fn,
                    scored=t >= spec.n_free)
        if t + 1 < spec.total_trials:
            _emit(ep, [vocab.SEP], Channel.STRUCTURAL, -1)

    _emit(ep, [vocab.EOS], Channel.STRUCTURAL, -1)
    ep.check()
    return ep


def _emit(ep: Episode, tokens: Sequence[int], channel: Channel, trial: int,
          supervised: bool = False, targets: Sequence[int] | None = None) -> None:
    tgt = list(tokens) if targets is None else list(targets)
    if len(tgt) != len(tokens):
        raise ValueError(
            f"channel {channel.name}: {len(tokens)} tokens against {len(tgt)} targets. "
            "The L2 query channel needs the model's query and q* to render to the "
            "same length; see docs/10-harness-findings.md."
        )
    ep.tokens.extend(tokens)
    ep.targets.extend(tgt)
    ep.supervised.extend([supervised] * len(tokens))
    ep.channel.extend([channel] * len(tokens))
    ep.trial_index.extend([trial] * len(tokens))


def _emit_trial(ep, family, theta, enc, k, spec, history, rng, t, query_fn,
                scored: bool = True) -> None:
    """One trial: the query, then the oracle's answer, then optionally the trace.

    On ORACLE_ECHO, since section 7 names it and nothing defines it: the only
    tokens the harness emits that neither state a rule nor carry an answer are
    the ERR response to a malformed L2 query and the ASK/ANSWER turn markers.
    Those are echo -- the environment talking about the exchange rather than
    participating in it -- and they are not supervised.  The oracle's *answer*
    tokens are supervised, because section 2.1 scores them as "the model's
    prediction of rho_e(y_t)"; teacher-forced and supervised are the same
    positions in an autoregressive stream, and section 2.1's "oracle's emitted
    tokens masked" cannot mean those or L2 would have no answer channel at all.
    """
    # Free observations are always sampled, never model-chosen, whatever the
    # spec says: they are evidence handed to the model, and asking it to choose
    # queries it will not be scored on would make the dial move two things at
    # once -- how much evidence precedes the scored trial, and how good the
    # model happens to be at choosing evidence at that moment.
    if scored and spec.query_source is QuerySource.MODEL:
        query, emitted = query_fn(list(ep.tokens), list(history), enc)
        _emit(ep, [vocab.ASK], Channel.ORACLE_ECHO, t)
        teacher = family.teacher_query(theta, history)
        teacher_tokens = family.render(enc, teacher)
        if query is None:
            ep.malformed_queries += 1
            # A6: a well-formed refusal, which section 3 calls a recovery lesson.
            # The query channel is still supervised against q* -- that IS the
            # lesson, and dropping supervision here would make a bad query cost
            # nothing.
            _emit(ep, emitted, Channel.QUERY, t, supervised=True,
                  targets=_pad_to(teacher_tokens, len(emitted)))
            _emit(ep, [vocab.ERR], Channel.ORACLE_ECHO, t)
            # No history entry. The trial produced no observation, so nothing was
            # learned about theta and q* is unchanged -- a family cycling its
            # probes on len(history) correctly re-asks the same one. Appending a
            # (None, None) placeholder instead would advance every such counter
            # past a question that was never answered, and would break the
            # history contract for the eight families that do not expect it.
            return
        _emit(ep, emitted, Channel.QUERY, t, supervised=True, targets=teacher_tokens)
    else:
        query = family.sample_query(theta, history, rng)
        _emit(ep, family.render(enc, query), Channel.QUERY, t)

    answer = draw_answer(family, theta, query, rng)
    answer_tokens = family.render(enc, answer)

    if not scored:
        # A free observation: rendered into the context as evidence, and not
        # scored. This is the second dial section 8 step 5 names -- "varying how
        # many observations precede the query" -- and it is the *generic* one,
        # since it asks nothing of a family that section 7 does not already
        # provide. It also holds the supervised-token count constant across the
        # whole sweep, which the preamble dial cannot promise for every family.
        _emit(ep, answer_tokens, Channel.OBSERVATION, t)
    elif spec.target_mode is TargetMode.POSTERIOR:
        dist = _token_posterior(family, enc, history, query, k, answer_tokens)
        start = len(ep.tokens)
        _emit(ep, answer_tokens, Channel.ANSWER, t, supervised=True)
        ep.posterior_targets[start] = dist
    else:
        _emit(ep, answer_tokens, Channel.ANSWER, t, supervised=True)

    if scored and spec.emit_trace and getattr(family, "emits_trace", False):
        steps = family.trace(theta, query)
        for step in steps or []:
            _emit(ep, [vocab.STEP], Channel.STRUCTURAL, t)
            _emit(ep, family.render(enc, step), Channel.TRACE, t, supervised=True)

    history.append((query, answer))


def _pad_to(tokens: list[int], n: int) -> list[int]:
    """Line a teacher query up against a differently-sized model emission.

    Only reachable on the malformed path, where the model emitted something that
    is not a query at all and so has no length guarantee.  Truncating or padding
    with PAD is arbitrary; it is done here rather than raising because refusing
    to build the episode would mean the model can crash training by emitting
    nonsense, which is the state L2 exists to train out of.
    """
    if len(tokens) >= n:
        return tokens[:n]
    return tokens + [vocab.PAD] * (n - len(tokens))


def _token_posterior(family, enc, history, query, k, answer_tokens) -> dict[int, float]:
    """The exact L3 target, as a distribution over *token ids*.

    Restricted to single-token answers on purpose.  A distribution over a
    multi-token answer is not a per-token target, and factorizing it into one
    would require assuming independence across the answer's tokens, which is
    false in general.  Families whose answers render to several tokens get an
    exception rather than a target computed under an assumption nobody stated.
    """
    if len(answer_tokens) != 1:
        raise ProtocolGap(
            f"{type(family).__name__} renders answers to {len(answer_tokens)} tokens; "
            "the posterior target mode needs single-token answers (see the note in "
            "_token_posterior)."
        )
    dist = answer_distribution(family, history, query, k)
    ctor = answer_type(family)
    out: dict[int, float] = {}
    for key, p in dist.items():
        if p <= 0.0:
            continue
        rendered = family.render(enc, ctor(key))
        if len(rendered) != 1:
            raise ProtocolGap(
                f"{type(family).__name__} renders answer key {key!r} to "
                f"{len(rendered)} tokens; cannot be a per-token target"
            )
        out[rendered[0]] = out.get(rendered[0], 0.0) + p
    total = sum(out.values())
    if total <= 0.0:
        raise ValueError("posterior target has no mass")
    return {tok: p / total for tok, p in out.items()}
