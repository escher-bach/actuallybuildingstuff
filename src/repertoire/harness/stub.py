"""The stub family Task Spec section 8 step 1 asks for.

    "Test against a stub family with one rule and one encoding.
     Done when: a stub episode round-trips at all four levels and the logger
     emits a structural-content number."

**This is not a candidate family and must never enter the register.**  It fails
A4 outright and by construction: theta *is* a lookup table, and the shortest
solver that ignores all intended structure is the intended solver.  That is
exactly what makes it the right instrument here -- every quantity the harness
computes has a closed form against it, so a disagreement is a harness bug and
never an open question about the family.

theta is a hidden bit vector b of length d.  A query is an index i; the answer is
b[i].  Everything the harness needs is exact:

    identification    needs all d indices queried; before that, exactly the
                      unqueried coordinates are open
    posterior         0.5/0.5 on an unseen index, a point mass on a seen one --
                      and therefore *query-dependent*, which is the property
                      docs/03 finding 1 says a harness must not lose
    q*                the lowest index not yet determined
    residual entropy  (unknown coordinates / d) * log 2 nats per uniform query,
                      in closed form, so the sweep's measured x-axis can be
                      checked against arithmetic before it is trusted on a
                      family where it cannot be
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

from .. import vocab
from .protocol import Reveal


@dataclass(frozen=True)
class Encoding:
    """One encoding, per step 1's "one rule and one encoding".

    Carries permutable symbol assignments anyway, so the A2 machinery has
    something to act on: a stub that cannot fail A2 could not detect an A2
    checker that never fails either (hazard 6).
    """

    name: str
    index_symbols: tuple[int, ...]  # index -> token
    value_symbols: tuple[int, int]  # bit -> token
    separator: int


@dataclass(frozen=True)
class StubTheta:
    bits: tuple[int, ...]


@dataclass(frozen=True)
class Query:
    index: int


@dataclass(frozen=True)
class Answer:
    bit: int


class StubLookupFamily:
    """A hidden bit vector, queried one coordinate at a time."""

    name = "stub_lookup"
    answer_type = Answer
    supports_L2 = True
    emits_trace = False
    stochastic = False

    def __init__(self, d: int = 8) -> None:
        if not 1 <= d <= 32:
            raise ValueError("d must fit the symbol alphabet with room for values")
        self.d = d

    def dimensions(self, k: int) -> int:
        return self.d

    # -- section 7 ------------------------------------------------------

    def sample_theta(self, k: int, rng: Random) -> StubTheta:
        return StubTheta(tuple(rng.randrange(2) for _ in range(self.d)))

    def enumerate_theta(self, k: int) -> list[StubTheta]:
        out = []
        for mask in range(1 << self.d):
            out.append(StubTheta(tuple((mask >> i) & 1 for i in range(self.d))))
        return out

    def sample_encoding(self, rng: Random) -> Encoding:
        pool = list(vocab.SYMBOL_IDS)
        rng.shuffle(pool)
        return Encoding(
            name="lookup",
            index_symbols=tuple(pool[: self.d]),
            value_symbols=(pool[self.d], pool[self.d + 1]),
            separator=vocab.STOI["EQ"],
        )

    def sample_query(self, theta: StubTheta, history: list, rng: Random) -> Query:
        return Query(rng.randrange(self.d))

    def teacher_query(self, theta: StubTheta, history: list) -> Query:
        """The lowest coordinate not yet determined; cycles once all are known.

        Deterministic in (theta, history) because section 7 hands teacher_query
        no rng -- interface finding 3, and A7's one-pass requirement anyway.
        """
        seen = {q.index for q, a in history if q is not None and a is not None}
        for i in range(self.d):
            if i not in seen:
                return Query(i)
        return Query(len(history) % self.d)

    def evaluate(self, theta: StubTheta, query: Query) -> Answer:
        return Answer(theta.bits[query.index])

    def trace(self, theta: StubTheta, query: Query) -> list | None:
        return None

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, Query):
            return [encoding.index_symbols[obj.index], encoding.separator]
        if isinstance(obj, Answer):
            return [encoding.value_symbols[obj.bit]]
        raise TypeError(f"cannot render {type(obj).__name__}")

    def parse_query(self, encoding: Encoding, tokens: list[int]) -> Query | None:
        """rho_e^-1 on the query channel. None means malformed, which is A6's case."""
        if len(tokens) != 2 or tokens[1] != encoding.separator:
            return None
        if tokens[0] not in encoding.index_symbols:
            return None
        return Query(encoding.index_symbols.index(tokens[0]))

    def preamble(self, theta: StubTheta, encoding: Encoding) -> list[int]:
        return self.partial_preamble(theta, encoding, 1.0, Random(0)).tokens

    def partial_preamble(
        self, theta: StubTheta, encoding: Encoding, fraction: float, rng: Random
    ) -> Reveal:
        """State a fraction of the bits; unstated slots render as NEG.

        Fixed length at every fraction, which is the requirement in
        `SupportsPartialReveal`: the dial must move residual entropy without also
        moving how many tokens the model gets to look at.
        """
        n_reveal = int(round(fraction * self.d))
        revealed = sorted(rng.sample(range(self.d), n_reveal)) if n_reveal else []
        shown = {i: theta.bits[i] for i in revealed}

        # Rendered in the SAME SHAPE as a trial -- `index EQ value`, not
        # `index value`. A preamble entry is a stated example, and if it is laid
        # out differently from the trials then "find this symbol and read what
        # follows it" needs one offset in the preamble and a different one in the
        # history, for the same match. That is two circuits where one would do,
        # and it is enough to keep a 3-layer model at chance on what is otherwise
        # a match-and-copy task.
        out = [vocab.PREAMBLE]
        for i in range(self.d):
            out.append(encoding.index_symbols[i])
            out.append(encoding.separator)
            out.append(
                encoding.value_symbols[shown[i]] if i in shown else vocab.STOI["NEG"]
            )
        return Reveal(
            tokens=out,
            n_slots=self.d,
            n_revealed=n_reveal,
            consistent=lambda t: all(t.bits[i] == v for i, v in shown.items()),
        )

    def posterior(self, history: list, k: int, query: Query | None = None) -> dict[int, float]:
        """Exact, and query-dependent -- which is the whole point of the argument.

        Carries the explicit `query` parameter the harness adopted (see
        `protocol.answer_distribution`). Without a query there is no target, only
        the marginal, and this returns it labelled as such rather than passing it
        off as the L3 target.
        """
        known: dict[int, int] = {}
        for q, a in history:
            if q is None or a is None:
                continue
            known[q.index] = a.bit
        if query is None:
            return {0: 0.5, 1: 0.5}  # the marginal; NOT the training target
        if query.index in known:
            b = known[query.index]
            return {0: 1.0 if b == 0 else 0.0, 1: 1.0 if b == 1 else 0.0}
        return {0: 0.5, 1: 0.5}

    def permuted_alphabet_check(self, rng: Random) -> bool:
        """A2 under a *total* permutation of the symbol alphabet.

        Total rather than encoding-local, per interface finding 5: permuting only
        the symbols in use leaves a token rendered from a fixed constant untouched
        on both sides, so the check passes the exact leak it exists to catch.
        """
        theta = self.sample_theta(1, rng)
        enc = self.sample_encoding(rng)
        shuffled = list(vocab.SYMBOL_IDS)
        rng.shuffle(shuffled)
        perm = dict(zip(vocab.SYMBOL_IDS, shuffled))
        enc2 = replace(
            enc,
            index_symbols=tuple(perm[s] for s in enc.index_symbols),
            value_symbols=(perm[enc.value_symbols[0]], perm[enc.value_symbols[1]]),
        )
        for i in range(self.d):
            q = Query(i)
            a = self.evaluate(theta, q)
            expected = [perm.get(t, t) for t in self.render(enc, q) + self.render(enc, a)]
            if expected != self.render(enc2, q) + self.render(enc2, a):
                return False
        return True


class LeakyStubFamily(StubLookupFamily):
    """A stub that renders one token from a constant instead of the encoding.

    Exists so the A2 check has a case it must fail.  Hazard 6: "a plant passing
    its own check proves nothing unless failure is reachable", met twice already.
    """

    name = "stub_leaky"

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, Answer):
            # The leak: bit 1 always renders as the same symbol, whatever the
            # encoding says. A permutation moves it on the expected side only.
            return [vocab.sym(0)] if obj.bit else [encoding.value_symbols[0]]
        return super().render(encoding, obj)
