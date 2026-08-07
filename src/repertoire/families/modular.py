"""The worked family: modular arithmetic under a hidden alphabet permutation.

Task Spec section 6, and section 8 step 2 is the gate on it.  Written by the
harness session because step 2 is the harness's, and registered as
`register/rows/mod-arith-hidden-permutation.toml`, which was filled in before any
code existed and which this implementation is held to.

    Theta         (m, pi) -- modulus m, and pi assigning each residue a symbol
    X             ordered pairs of symbols (a, b)
    f             f((m,pi),(a,b)) = pi(pi^-1(a) + pi^-1(b) mod m)
    E             infix `a + b = c`, prefix `(+ a b) -> c`, tabular `a, b -> c`

Section 6 calls this the family to build first, and it earns that: A1 holds
trivially (evaluate, never search), A2 holds *by construction* because pi IS an
alphabet permutation, and A4 fails at small m because the whole oracle is an
m x m table.  Section 6 says that failure is "instructive rather than
disqualifying, and is exactly what section 8 step 5 measures".

--------------------------------------------------------------------------
Three construction decisions where the obvious version leaks
--------------------------------------------------------------------------

**The symbol pool is longer than m and shuffled per episode.**  The obvious
construction gives the encoding exactly m symbols, and then the *number of
distinct symbols in the pool* announces m before a single trial.  m is part of
theta and inferring it is part of the task, so that is a leak straight out of the
preamble.  Here the pool is `pool_size` symbols whatever m is, drawn fresh per
episode, and m shows up only in how many distinct symbols ever appear in
answers -- which is inference, not an artifact.

**pi is an injection into the pool, not a permutation of it.**  With a
permutation, residues would occupy pool positions 0..m-1 and the unused tail
would be identifiable by position.  An injection lets the used positions be
anywhere, which is what stops position from carrying information the symbols do
not.

**The operator token is drawn from the structural tokens, not the content
alphabet.**  The register row's A3 note suggests the opposite -- "consider adding
a rendering where the operator symbol is itself drawn from the permuted
alphabet" -- and it is a good suggestion for a *fourth* encoding, but it cannot
be the default: A2's check permutes the content alphabet, and an operator drawn
from that alphabet would move under the permutation while denoting the same
operation, which is correct behaviour that would nonetheless make the A2 check
compare two different renderings. Left as a documented gap rather than done
badly; see docs/10.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from random import Random

from .. import vocab
from ..harness.protocol import Reveal

Distribution = dict[int, float]

DEFAULT_POOL = 12


@dataclass(frozen=True)
class Encoding:
    """A3: three renderings, differing in more than punctuation.

    Hazard 7 is the standing warning here -- every family's encodings once
    produced byte-identical episodes and differed only in which token was the
    separator, so A3 was satisfied in letter and vacuous in fact.  These differ
    in token count and in argument order, and `a3_test.py` is the thing that says
    whether that is enough rather than this docstring.
    """

    name: str
    symbols: tuple[int, ...]  # pool position -> token id
    operator: int

    def token(self, position: int) -> int:
        return self.symbols[position]


ENCODING_NAMES = ("infix", "prefix", "tabular")


@dataclass(frozen=True)
class ModTheta:
    """m, and the injection pi: Z_m -> pool position."""

    m: int
    pi: tuple[int, ...]  # pi[r] = pool position denoting residue r

    def inverse(self) -> dict[int, int]:
        return {p: r for r, p in enumerate(self.pi)}


@dataclass(frozen=True)
class Pair:
    """A query: two pool positions."""

    a: int
    b: int


@dataclass(frozen=True)
class Answer:
    position: int


@dataclass(frozen=True)
class Step:
    """One line of the section 1.2 trace."""

    kind: str  # decode | add | reduce | encode
    value: int


class ModularHiddenPermutationFamily:
    """Task Spec section 6, implemented against section 7."""

    name = "mod_arith_hidden_permutation"
    answer_type = Answer
    supports_L2 = True
    emits_trace = True
    stochastic = False

    def __init__(self, pool_size: int = DEFAULT_POOL, exact_q_star_cap: int = 8000,
                 trace_detail: str = "full") -> None:
        if trace_detail not in ("full", "reduced", "none"):
            raise ValueError("trace_detail is full | reduced | none")
        self.pool_size = pool_size
        self.exact_q_star_cap = exact_q_star_cap
        self.trace_detail = trace_detail
        self._enum_cache: dict[int, list[ModTheta]] = {}

    # -- k -> the modulus band ------------------------------------------

    def moduli(self, k: int) -> tuple[int, ...]:
        """Which m are in play at difficulty k.

        A band rather than a single value, because theta is (m, pi) and a
        determined m would make half of theta not hidden.  Kept small: |Theta| is
        sum over the band of P(pool_size, m), and every exact target in this file
        is an enumeration over it.  At the default pool of 12 and k=1 that is
        about 10^7 -- which is why `enumerate_theta` refuses above a cap rather
        than running for an hour, and why the sweep runs at small pools.
        """
        base = 5 + k
        return (base, base + 1)

    def max_modulus(self, k: int) -> int:
        return max(self.moduli(k))

    # -- section 7 -------------------------------------------------------

    def sample_theta(self, k: int, rng: Random) -> ModTheta:
        m = rng.choice(self.moduli(k))
        positions = rng.sample(range(self.pool_size), m)
        return ModTheta(m=m, pi=tuple(positions))

    def enumerate_theta(self, k: int) -> list[ModTheta]:
        """Every (m, pi). Exact, and it refuses rather than approximating.

        Section 8 step 2's gate is "L3 targets match a brute-force enumeration of
        consistent theta", so the brute force has to exist somewhere even though
        it is not what runs in a training loop.  The cap is a real limit of this
        family and stating it is the point: exact L3 supervision on modular
        arithmetic is affordable at a small pool and is not affordable at
        section 6's full m <= 20.
        """
        if k in self._enum_cache:
            return self._enum_cache[k]
        total = sum(
            math.perm(self.pool_size, m) for m in self.moduli(k)
        )
        if total > 200_000:
            raise ValueError(
                f"|Theta| = {total:,} at k={k}, pool={self.pool_size}: too large to "
                "enumerate. Lower pool_size or k. This family's exact posterior is "
                "affordable only at small pools -- a real property of it, not a "
                "limitation of the harness."
            )
        out: list[ModTheta] = []
        for m in self.moduli(k):
            for pi in itertools.permutations(range(self.pool_size), m):
                out.append(ModTheta(m=m, pi=pi))
        self._enum_cache[k] = out
        return out

    def prior_weight(self, theta: ModTheta, k: int) -> float:
        """P_Theta(theta | k), up to a constant.

        Present because `entropy.check_prior_matches_sampler` exists to catch
        exactly the mismatch it would otherwise have: `sample_theta` picks m
        uniformly from the band *and then* an injection, so a theta with a small
        m is more likely per-theta than one with a large m, since there are fewer
        injections to share the mass. A uniform prior over the enumeration would
        get the m-marginal wrong, and every L3 target with it.
        """
        band = self.moduli(k)
        if theta.m not in band:
            return 0.0
        return 1.0 / (len(band) * math.perm(self.pool_size, theta.m))

    def sample_encoding(self, rng: Random) -> Encoding:
        pool = list(vocab.SYMBOL_IDS)
        rng.shuffle(pool)
        name = rng.choice(ENCODING_NAMES)
        return Encoding(
            name=name,
            symbols=tuple(pool[: self.pool_size]),
            operator=vocab.OPERATOR_IDS[0],
        )

    def sample_query(self, theta: ModTheta, history: list, rng: Random) -> Pair:
        """Both operands are symbols that denote a residue, i.e. positions in pi.

        A query naming a pool position outside pi is *legal to construct* and is
        handled by `evaluate` as an A6 error case -- but it is never sampled,
        because an episode whose observations are mostly errors teaches error
        formatting rather than modular arithmetic. The model reaches those cases
        at L2, where it chooses, which is where A6 says the recovery lesson lives.
        """
        return Pair(rng.choice(theta.pi), rng.choice(theta.pi))

    def teacher_query(self, theta: ModTheta, history: list) -> Pair:
        """A7. Exact expected-entropy-reduction where affordable, heuristic above.

        A7 accepts a documented heuristic -- "the target is a good query, not a
        provably optimal one" -- and this is both, with the switch stated rather
        than hidden: below `exact_q_star_cap` surviving hypotheses the query
        maximizing expected posterior entropy reduction is computed by
        enumeration, and above it the policy is

            probe unresolved symbols against themselves, lowest first,

        which is defensible for a specific reason: a + a = c pins 2*pi^-1(a)
        directly, so a self-pair is the highest-yield single observation about one
        unknown symbol, and cycling them covers pi in m observations.

        Deterministic in (theta, history) because section 7 hands this no rng
        (interface finding 3), and one-pass because A7 requires it.
        """
        seen = self._resolved_positions(history)
        unresolved = [p for p in theta.pi if p not in seen]
        heuristic = Pair(unresolved[0], unresolved[0]) if unresolved else Pair(
            theta.pi[len(history) % theta.m], theta.pi[(len(history) + 1) % theta.m]
        )
        try:
            survivors = self._survivors(history, self._k_hint(theta))
        except (ValueError, KeyError):
            return heuristic
        if not survivors or len(survivors) > self.exact_q_star_cap:
            return heuristic
        return self._max_info_gain(survivors, theta)

    def evaluate(self, theta: ModTheta, query: Pair) -> Answer:
        """f, and it evaluates: one inverse lookup each, one add, one reduce.

        A1 holds because nothing here searches and the cost does not move with k.
        """
        inv = theta.inverse()
        if query.a not in inv or query.b not in inv:
            # A6: total. A position outside pi denotes nothing, so the answer is
            # a refusal rather than an exception or a guess.
            return Answer(-1)
        return Answer(theta.pi[(inv[query.a] + inv[query.b]) % theta.m])

    def trace(self, theta: ModTheta, query: Pair) -> list[Step] | None:
        """Section 1.2, with the thinning schedule declared as section 1.2 requires.

        `trace_detail` is a constructor parameter and not a function of k, which
        is a deliberate departure from the Task Spec's stated default ("emit
        traces at low k, thin them as k rises"). docs/05 found that the
        literature the default cites fades guidance against demonstrated *learner
        competence*, never against a task-difficulty knob, and says fading only
        bites when load is high. The spec itself calls this "the least settled
        decision in the document" and says to treat the schedule as a swept
        parameter. A parameter is sweepable; a function of k baked in here is not.
        """
        if self.trace_detail == "none":
            return None
        inv = theta.inverse()
        if query.a not in inv or query.b not in inv:
            return None
        ra, rb = inv[query.a], inv[query.b]
        total = ra + rb
        if self.trace_detail == "reduced":
            return [Step("reduce", total % theta.m)]
        return [
            Step("decode", ra),
            Step("decode", rb),
            Step("add", total),
            Step("reduce", total % theta.m),
        ]

    # -- rendering -------------------------------------------------------

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, Pair):
            a, b = encoding.token(obj.a), encoding.token(obj.b)
            if encoding.name == "infix":  # a + b =
                return [a, encoding.operator, b, vocab.STOI["EQ"]]
            if encoding.name == "prefix":  # ( + a b ) ->
                return [vocab.STOI["LPAREN"], encoding.operator, a, b,
                        vocab.STOI["RPAREN"], vocab.STOI["ARROW"]]
            return [a, vocab.STOI["COMMA"], b, vocab.STOI["ARROW"]]  # tabular
        if isinstance(obj, Answer):
            if obj.position < 0:
                return [vocab.ERR]
            return [encoding.token(obj.position)]
        if isinstance(obj, Step):
            return vocab.number(obj.value)
        raise TypeError(f"cannot render {type(obj).__name__}")

    def parse_query(self, encoding: Encoding, tokens: list[int]) -> Pair | None:
        """rho_e^-1 on the query channel, so L2 can answer what was asked.

        Returns None for anything that is not a well-formed query under this
        encoding, which is A6's case and the harness's cue to emit ERR.
        """
        pos = {t: i for i, t in enumerate(encoding.symbols)}
        eq, arrow = vocab.STOI["EQ"], vocab.STOI["ARROW"]
        if encoding.name == "infix":
            if len(tokens) != 4 or tokens[1] != encoding.operator or tokens[3] != eq:
                return None
            a, b = tokens[0], tokens[2]
        elif encoding.name == "prefix":
            if (len(tokens) != 6 or tokens[0] != vocab.STOI["LPAREN"]
                    or tokens[1] != encoding.operator
                    or tokens[4] != vocab.STOI["RPAREN"] or tokens[5] != arrow):
                return None
            a, b = tokens[2], tokens[3]
        else:
            if len(tokens) != 4 or tokens[1] != vocab.STOI["COMMA"] or tokens[3] != arrow:
                return None
            a, b = tokens[0], tokens[2]
        if a not in pos or b not in pos:
            return None
        return Pair(pos[a], pos[b])

    # -- the preamble, and the dial --------------------------------------

    def preamble(self, theta: ModTheta, encoding: Encoding) -> list[int]:
        return self.partial_preamble(theta, encoding, 1.0, Random(0)).tokens

    def partial_preamble(
        self, theta: ModTheta, encoding: Encoding, fraction: float, rng: Random
    ) -> Reveal:
        """Section 6's `MOD 7 | MAP q->0 f->1 ...`, at an arbitrary reveal fraction.

        The dial of section 8 step 5.  m occupies one slot and each residue's
        symbol occupies one, so a theta at modulus m has m+1 slots; a slot not
        revealed renders as NEG in the same position, which keeps the preamble the
        same length at every fraction *for a given m*.

        The residual length variation across m is real and is why the sweep holds
        the band narrow: the alternative is padding the map to `pool_size` entries,
        which would announce the pool size in every episode and hand back part of
        what the shuffled pool was hiding.
        """
        n_slots = theta.m + 1
        n_reveal = int(round(fraction * n_slots))
        slots = list(range(n_slots))  # 0 = modulus, 1+r = residue r
        revealed = set(rng.sample(slots, n_reveal)) if n_reveal else set()

        out = [vocab.PREAMBLE]
        out += vocab.number(theta.m) if 0 in revealed else [vocab.STOI["NEG"]]
        out.append(vocab.STOI["PIPE"])
        for r in range(theta.m):
            out += vocab.number(r)
            out.append(vocab.STOI["ARROW"])
            out.append(encoding.token(theta.pi[r]) if (1 + r) in revealed
                       else vocab.STOI["NEG"])
            out.append(vocab.STOI["COMMA"])

        shown_m = theta.m if 0 in revealed else None
        shown_pi = {r: theta.pi[r] for r in range(theta.m) if (1 + r) in revealed}

        def consistent(t: ModTheta) -> bool:
            if shown_m is not None and t.m != shown_m:
                return False
            return all(r < t.m and t.pi[r] == p for r, p in shown_pi.items())

        return Reveal(tokens=out, n_slots=n_slots, n_revealed=n_reveal,
                      consistent=consistent)

    # -- L3 --------------------------------------------------------------

    def posterior(self, history: list, k: int, query: Pair | None = None) -> Distribution:
        """Exact Bayes over the answer, conditioned on the pending query.

        Carries the explicit `query` parameter the harness adopted (handoff
        section 2.1, resolved in `harness/protocol.answer_distribution`).  Without
        one there is no L3 target, only the marginal over the query space, and
        this returns it labelled as such -- docs/03 finding 1 is that handing the
        marginal back as the target trains maximal uncertainty about a rule
        already identified.

        Keyed by pool position, which is what `Answer` wraps.
        """
        survivors = self._survivors(history, k)
        if not survivors:
            raise ValueError("no theta consistent with the history")
        if query is None:
            counts: Distribution = {}
            for theta, w in survivors:
                for a in theta.pi:
                    for b in theta.pi:
                        p = self.evaluate(theta, Pair(a, b)).position
                        counts[p] = counts.get(p, 0.0) + w / (theta.m * theta.m)
            z = sum(counts.values())
            return {p: c / z for p, c in counts.items()}

        counts = {}
        for theta, w in survivors:
            counts[self.evaluate(theta, query).position] = (
                counts.get(self.evaluate(theta, query).position, 0.0) + w
            )
        z = sum(counts.values())
        return {p: c / z for p, c in counts.items()}

    def _survivors(self, history: list, k: int) -> list[tuple[ModTheta, float]]:
        out: list[tuple[ModTheta, float]] = []
        for theta in self.enumerate_theta(k):
            w = self.prior_weight(theta, k)
            if w <= 0:
                continue
            ok = True
            for q, a in history:
                if q is None or a is None:
                    continue
                if self.evaluate(theta, q).position != a.position:
                    ok = False
                    break
            if ok:
                out.append((theta, w))
        z = sum(w for _, w in out)
        return [(t, w / z) for t, w in out] if z > 0 else []

    def _resolved_positions(self, history: list) -> set[int]:
        seen: set[int] = set()
        for q, a in history:
            if q is None or a is None:
                continue
            seen.update((q.a, q.b))
        return seen

    def _k_hint(self, theta: ModTheta) -> int:
        """Recover k from theta, since teacher_query is not given it.

        Section 7's `teacher_query(theta, history)` has no k, and the hypothesis
        space depends on k. Recovered from theta's own modulus rather than stored
        on self, because a family holding per-episode state on itself is exactly
        the thing that makes seeded reconstructibility stop working under
        batching. Recorded in docs/10 as a signature gap.
        """
        for k in range(0, 16):
            if theta.m in self.moduli(k):
                return k
        raise KeyError(f"no k gives modulus {theta.m}")

    def _max_info_gain(self, survivors: list[tuple[ModTheta, float]], theta: ModTheta) -> Pair:
        best, best_pair = None, None
        for a in theta.pi:
            for b in theta.pi:
                outcome: dict[int, float] = {}
                for t, w in survivors:
                    p = self.evaluate(t, Pair(a, b)).position
                    outcome[p] = outcome.get(p, 0.0) + w
                # Expected posterior entropy over theta after this query, up to a
                # constant: sum_y P(y) * H(theta|y). Minimizing it maximizes the
                # expected reduction, which is what A7 names.
                h = -sum(p * math.log(p) for p in outcome.values() if p > 0)
                if best is None or h > best:
                    best, best_pair = h, Pair(a, b)
        return best_pair or Pair(theta.pi[0], theta.pi[0])

    # -- A2 ---------------------------------------------------------------

    def permuted_alphabet_check(self, rng: Random) -> bool:
        """A2 under a *total* permutation of the symbol alphabet.

        Section 6 says A2 holds here "by construction, since pi IS an alphabet
        permutation" -- which makes the check a formality, and the row says that
        is "precisely why it is the right place to test that the check itself
        works".  `LeakyModularFamily` below is that test.
        """
        k = 1
        theta = self.sample_theta(k, rng)
        enc = self.sample_encoding(rng)
        shuffled = list(vocab.SYMBOL_IDS)
        rng.shuffle(shuffled)
        perm = dict(zip(vocab.SYMBOL_IDS, shuffled))
        enc2 = replace(enc, symbols=tuple(perm[s] for s in enc.symbols))

        for _ in range(64):
            q = self.sample_query(theta, [], rng)
            a = self.evaluate(theta, q)
            expected = [perm.get(t, t) for t in self.render(enc, q) + self.render(enc, a)]
            if expected != self.render(enc2, q) + self.render(enc2, a):
                return False
        return True


class LeakyModularFamily(ModularHiddenPermutationFamily):
    """Renders one answer from a fixed symbol instead of the encoding's.

    Hazard 6: a check that cannot fail is worse than no check, met twice already
    in this repository. This exists so `permuted_alphabet_check` has a case it
    must reject, and the test suite asserts that it does.
    """

    name = "mod_arith_leaky"

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, Answer) and obj.position == 0:
            return [vocab.sym(0)]  # the leak: constant, not encoding-relative
        return super().render(encoding, obj)
