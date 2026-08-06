"""The two junk plants.

Repertoire Spec section 5 requires both, and Task Spec section 8 step 4 gates the
whole programme on them behaving as predicted: if planted junk does not read
near-zero, measurement cannot guide extraction and the method is void before any
expensive step.  They are instruments, not filler.

    RandomTargetFamily    near-zero structural content because nothing is learnable
    ConstantTargetFamily  near-zero for the OPPOSITE reason: everything is learned
                          in one trial and the curve flattens immediately

Both yield a small area under the loss curve; that they do so for opposite
reasons is the point, and a matrix that cannot tell them apart from a real family
is not measuring structure.

--------------------------------------------------------------------------
A design error worth recording, because it was nearly built in
--------------------------------------------------------------------------

The first specification for RandomTargetFamily said the answer should be a hash
of (theta, query), so that asking the same query twice returned the same answer
-- on the reasoning that an oracle contradicting itself is a bug.

That is wrong, and it would have quietly destroyed the plant.  An answer that is
a fixed function of the query is a lookup table: unpredictable on first sight,
fully learnable on second.  Within an episode a model would identify it exactly
as it identifies any other hidden rule, the loss would drop, and the family would
measure as a *memorization* family with real structural content -- while sitting
in the matrix labelled "junk, expected near-zero."  A plant that silently stops
being a plant is worse than no plant, because the instrument-validation gate
would pass while measuring the wrong thing.

The correct construction: y_t is drawn fresh each trial, independent of the
query and of theta.  Repeats disagree, and that is exactly right -- there is no
function here to be consistent with.  Loss sits at log(alphabet size) from the
first token and never moves.

This makes the family a genuine instance of Task Spec section 1.3 (a
distribution-valued oracle), which is convenient: section 1.3 asks that at least
one family in the repertoire exercise stochasticity, and notes that a model which
has never seen genuinely noisy input given a known rule will be miscalibrated on
input that is actually noisy.  The junk plant covers that at the degenerate end.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .. import vocab

# Distribution over answers: symbol index -> probability.
Distribution = dict[int, float]


def alphabet_size(k: int) -> int:
    """k -> alphabet size, shared by both plants so their floors are comparable.

    A plant's loss floor is log(alphabet size).  If the two plants disagreed on
    how k maps to alphabet size their floors would differ, and the near-zero
    areas that validate the instrument would not be measured against the same
    baseline.
    """
    return max(2, min(4 + 4 * k, vocab.N_SYMBOLS))


@dataclass(frozen=True)
class Encoding:
    """Surface form. A3 wants this sampled per episode, so both plants carry two."""

    name: str
    separator: int  # token between query and answer


ENCODINGS = (
    Encoding("infix", vocab.STOI["EQ"]),
    Encoding("arrow", vocab.STOI["ARROW"]),
)


@dataclass(frozen=True)
class Query:
    symbol: int


@dataclass(frozen=True)
class Answer:
    symbol: int


def _uniform(n: int) -> Distribution:
    return {s: 1.0 / n for s in range(n)}


# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RandomTheta:
    """Shaped like a real theta so it round-trips through the harness identically.

    It carries a seed that the oracle deliberately does not consult.  The point of
    giving junk the same shape as signal is that the harness must not be able to
    tell them apart -- if it could, the plant would be testing the harness's
    special-casing rather than the measurement.
    """

    alphabet_size: int
    seed: int


class RandomTargetFamily:
    """Targets independent of the query, of theta, and of each other.

    Expected measurement: structural content ~ 0, loss pinned at log(alphabet
    size) from the first token.
    """

    name = "junk_random"
    supports_L2 = True  # in the degenerate sense below
    emits_trace = False
    stochastic = True  # genuinely distribution-valued; see module docstring

    def sample_theta(self, k: int, rng: Random) -> RandomTheta:
        return RandomTheta(alphabet_size=alphabet_size(k), seed=rng.getrandbits(32))

    def sample_encoding(self, rng: Random) -> Encoding:
        return rng.choice(ENCODINGS)

    def sample_query(self, theta: RandomTheta, history: list, rng: Random) -> Query:
        return Query(rng.randrange(theta.alphabet_size))

    def teacher_query(self, theta: RandomTheta, history: list) -> Query:
        """A7, degenerately satisfied.

        No query is more informative than any other, because no query is
        informative at all.  That is a fact about the family and not a gap in it,
        so this returns a query rather than raising -- an L2 wrapper must still
        run over junk, otherwise the junk row is missing from exactly the level
        where the instrument most needs a floor.

        Deterministic in (theta, history) because the protocol hands
        `teacher_query` no rng: q* must be a one-pass function of what is known,
        which A7 requires anyway.
        """
        return Query(len(history) % theta.alphabet_size)

    def evaluate(self, theta: RandomTheta, query: Query) -> Distribution:
        """Distribution-valued (Task Spec section 1.3); the harness samples.

        Returning a distribution rather than drawing here is what lets a
        stochastic family stay inside a protocol whose `evaluate` takes no rng.
        Seeded reconstructibility then belongs to the harness, where it is
        already required to live.
        """
        return _uniform(theta.alphabet_size)

    def trace(self, theta: RandomTheta, query: Query) -> list | None:
        return None  # no derivation exists; there is no rule being applied

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, Query):
            return [vocab.sym(obj.symbol), encoding.separator]
        if isinstance(obj, Answer):
            return [vocab.sym(obj.symbol)]
        raise TypeError(f"cannot render {type(obj).__name__}")

    def preamble(self, theta: RandomTheta, encoding: Encoding) -> list[int] | None:
        """None: there is no rule to state.

        That is not an implementation gap -- it is the definition of this plant.
        L0 over junk is the same task as L3 over junk, which is itself a useful
        degenerate check on the level wrappers.
        """
        return None

    def posterior(self, history: list, k: int) -> Distribution:
        """Exactly uniform, always. The L3 target here is known in closed form.

        This makes the family a calibration check on the L3 machinery as well as
        a junk plant: a correct model's loss must equal log(alphabet size) to
        within numerical error, so any deviation is a bug in the harness rather
        than a fact about the model.
        """
        return _uniform(alphabet_size(k))

    def permuted_alphabet_check(self, rng: Random) -> bool:
        """A2. Passes because a uniform distribution is permutation-invariant."""
        k = 3
        n = alphabet_size(k)
        theta = self.sample_theta(k, rng)
        perm = list(range(n))
        rng.shuffle(perm)
        for q in range(n):
            base = self.evaluate(theta, Query(q))
            permuted = self.evaluate(theta, Query(perm[q]))
            for s in range(n):
                if abs(base[s] - permuted[perm[s]]) > 1e-12:
                    return False
        return True


# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConstantTheta:
    alphabet_size: int
    constant: int
    epsilon: float


class ConstantTargetFamily:
    """Constant or near-constant targets, ignoring the query.

    Expected measurement: structural content ~ 0 because the single bit of
    structure is absorbed in one trial and the curve flattens.

    theta is identifiable from one observation, so the within-episode acquisition
    slope is a step function.  That degenerate shape is useful: it is the sharpest
    available test that the slope logger measures identification rather than
    smoothness.
    """

    name = "junk_trivial"
    supports_L2 = True
    emits_trace = False

    def __init__(self, epsilon: float = 0.0) -> None:
        if not 0.0 <= epsilon < 1.0:
            raise ValueError("epsilon must be in [0, 1)")
        self.epsilon = epsilon
        # Exactly constant is deterministic; near-constant is a real stochastic
        # oracle, and the L3 posterior below is exact in both cases.
        self.stochastic = epsilon > 0.0

    def _epsilon(self, k: int) -> float:
        return self.epsilon

    def sample_theta(self, k: int, rng: Random) -> ConstantTheta:
        n = alphabet_size(k)
        return ConstantTheta(n, rng.randrange(n), self._epsilon(k))

    def sample_encoding(self, rng: Random) -> Encoding:
        return rng.choice(ENCODINGS)

    def sample_query(self, theta: ConstantTheta, history: list, rng: Random) -> Query:
        return Query(rng.randrange(theta.alphabet_size))

    def teacher_query(self, theta: ConstantTheta, history: list) -> Query:
        """Every query is maximally informative on trial 1 and useless after."""
        return Query(len(history) % theta.alphabet_size)

    def evaluate(self, theta: ConstantTheta, query: Query) -> Answer | Distribution:
        if theta.epsilon == 0.0:
            return Answer(theta.constant)
        n, eps = theta.alphabet_size, theta.epsilon
        off = eps / (n - 1)
        return {s: (1.0 - eps) if s == theta.constant else off for s in range(n)}

    def trace(self, theta: ConstantTheta, query: Query) -> list | None:
        return None

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, Query):
            return [vocab.sym(obj.symbol), encoding.separator]
        if isinstance(obj, Answer):
            return [vocab.sym(obj.symbol)]
        raise TypeError(f"cannot render {type(obj).__name__}")

    def preamble(self, theta: ConstantTheta, encoding: Encoding) -> list[int]:
        """L0: state the rule. The rule is one symbol."""
        return [vocab.PREAMBLE, vocab.sym(theta.constant), encoding.separator]

    def posterior(self, history: list, k: int) -> Distribution:
        """Exact Bayes over the next answer. Not the shorthand.

        The convenient description -- 'after one observation, mass 1-epsilon on
        the observed symbol' -- is only correct at epsilon = 0.  The true
        predictive distribution marginalizes over which constant theta is, and
        the observed symbol picks up mass from two routes: theta really is that
        symbol and the oracle did not slip, or theta is something else and the
        oracle slipped toward it.  At epsilon = 0.1 the shorthand is off by
        enough to matter for a calibration target, and a calibration plant that
        is itself miscalibrated is worse than none.
        """
        n = alphabet_size(k)
        eps = self.epsilon
        if eps == 0.0:
            for _, ans in history:
                if isinstance(ans, Answer):
                    return {s: (1.0 if s == ans.symbol else 0.0) for s in range(n)}
            return _uniform(n)

        off = eps / (n - 1)
        # Posterior over theta, uniform prior, likelihood (1-eps) on a match.
        w = [1.0] * n
        for _, ans in history:
            if not isinstance(ans, Answer):
                continue
            for c in range(n):
                w[c] *= (1.0 - eps) if ans.symbol == c else off
        total = sum(w)
        if total == 0.0:  # underflow guard; history is uninformative in the limit
            w, total = [1.0] * n, float(n)
        post_theta = [x / total for x in w]

        # Predictive over the next answer.
        return {
            s: post_theta[s] * (1.0 - eps) + (1.0 - post_theta[s]) * off
            for s in range(n)
        }

    def permuted_alphabet_check(self, rng: Random) -> bool:
        """A2, and a real check rather than a formality.

        Relabelling acts on theta as well as on the queries: the permuted family's
        constant is perm[constant].  The check is that answers computed under the
        permuted theta are the permutation of answers computed under the original.
        """
        k = 3
        n = alphabet_size(k)
        theta = self.sample_theta(k, rng)
        perm = list(range(n))
        rng.shuffle(perm)
        permuted_theta = ConstantTheta(n, perm[theta.constant], theta.epsilon)

        for q in range(n):
            base = self.evaluate(theta, Query(q))
            got = self.evaluate(permuted_theta, Query(perm[q]))
            if isinstance(base, Answer):
                if not isinstance(got, Answer) or got.symbol != perm[base.symbol]:
                    return False
            else:
                for s in range(n):
                    if abs(base[s] - got[perm[s]]) > 1e-12:
                        return False
        return True
