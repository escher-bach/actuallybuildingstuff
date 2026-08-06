"""Probability matching: the independent plant, and the real stochastic oracle.

Three jobs, which is why the simplest family in the set earns its place.

1. **The independent-pair plant**, against the SHJ concept families. Different
   Theta topology (a discretized rate vs a small finite rule set), different
   oracle behaviour (a noisy draw vs a deterministic label), different target
   (a distribution vs a point rule). Off-diagonal transfer should read near zero
   in both directions. If it does not, the instrument is finding structure where
   none was planted -- a calibration fault worth catching before the real matrix
   is read, and one that no amount of staring at the matrix afterwards would
   distinguish from a real result.

2. **The cleanest L3 exemplar available.** Every other family in the set is made
   L3 artificially, by truncating observations before identification. Here
   non-identifiability is intrinsic: no finite number of draws resolves a rate.

3. **It discharges Task Spec section 1.3.** The spec asks that at least one
   family exercise a genuinely stochastic oracle, warning that with deterministic
   f only, all uncertainty in the system is epistemic, and a model that has never
   seen noisy input given a KNOWN rule will be miscalibrated on input that is
   actually noisy. `junk-random` is stochastic but carries no signal. This family
   is stochastic *and* learnable, which is the combination the warning is about.

A contrast worth keeping rather than smoothing away: the source paradigm's own
finding is that humans do NOT maximize here -- they match the base rate, which is
suboptimal for score. Our L3 target is the exact posterior, so under our loss the
correct behaviour *is* calibration rather than maximization. The family therefore
sits exactly where the human result and the normative target come apart, which
makes it a useful reference when asking whether a model is matching because it is
calibrated or because it is confused.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .. import vocab

Distribution = dict[int, float]


@dataclass(frozen=True)
class Encoding:
    name: str
    outcome_symbols: tuple[int, int]
    separator: int


@dataclass(frozen=True)
class RateTheta:
    grid: int  # number of points on the discretized rate grid
    index: int  # theta = (index + 1) / (grid + 1) -- interior points only

    @property
    def p(self) -> float:
        """Strictly interior: never 0 and never 1.

        Excluding the endpoints is not fastidiousness, it is what makes the
        family what it claims to be. A rate of exactly 0 or 1 is deterministic,
        carries no irreducible noise, and IS fully resolvable from a finite
        history -- so a grid containing the endpoints would let the posterior
        collapse to a point and the "cleanest L3 exemplar" claim would be false
        for exactly the histories most likely to arise (a long run of one
        outcome). Caught by the test that asserts the posterior never fully
        resolves, which failed against the endpoint-inclusive grid.
        """
        return (self.index + 1) / (self.grid + 1)


@dataclass(frozen=True)
class Query:
    """There is only one option, so the query carries no choice.

    Kept as an object anyway so the family renders through the same path as every
    other. A family that special-cased its query type would be asking the harness
    to know something about it, which section 7 forbids.
    """

    tick: int


@dataclass(frozen=True)
class Outcome:
    value: int  # 0 or 1


class ProbabilityMatchingFamily:
    """A hidden Bernoulli rate on a discretized grid."""

    name = "probability_matching"
    supports_L2 = False  # see teacher_query
    emits_trace = False
    stochastic = True

    def grid_size(self, k: int) -> int:
        """k sets the grid precision.

        Note what this knob does and does not do. A finer grid makes the rate
        slower to pin down without making the task structurally harder -- it
        moves residual entropy specifically, holding the structure fixed. That is
        unusual among our families and makes this a natural candidate for the
        Task Spec section 8 step 5 dial sweep, where varying residual entropy
        continuously is exactly the requirement.
        """
        return 3 + 2 * k

    def sample_theta(self, k: int, rng: Random) -> RateTheta:
        n = self.grid_size(k)
        return RateTheta(n, rng.randrange(n))

    def sample_encoding(self, rng: Random) -> Encoding:
        pool = list(vocab.SYMBOL_IDS)
        rng.shuffle(pool)
        name, sep = rng.choice(
            [("binary", vocab.STOI["ARROW"]), ("draw", vocab.STOI["COLON"])]
        )
        return Encoding(name, (pool[0], pool[1]), sep)

    def sample_query(self, theta: RateTheta, history: list, rng: Random) -> Query:
        return Query(len(history))

    def teacher_query(self, theta: RateTheta, history: list) -> Query:
        """A7 fails here, and the row records it as a failure rather than a
        degenerate pass.

        With a single hidden rate there is no informative query to choose
        between -- every trial is equally informative. Unlike `junk-random`,
        where the same is true because nothing is informative at all, this is a
        family with real content and no query structure. Marking it L2-capable
        would put a family with no queries into the L2 column of the coverage
        grid, which would misreport what the candidate set actually spans.

        Still returns a query rather than raising, so a harness that wraps
        everything uniformly does not crash on it.
        """
        return Query(len(history))

    def evaluate(self, theta: RateTheta, query: Query) -> Distribution:
        """Distribution-valued. The harness samples; seeded reconstructibility
        stays where section 7 already requires it to live."""
        return {1: theta.p, 0: 1.0 - theta.p}

    def trace(self, theta: RateTheta, query: Query) -> list | None:
        return None

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, Query):
            return [vocab.ASK, encoding.separator]
        if isinstance(obj, Outcome):
            return [encoding.outcome_symbols[obj.value]]
        raise TypeError(f"cannot render {type(obj).__name__}")

    def preamble(self, theta: RateTheta, encoding: Encoding) -> list[int]:
        """L0: state the rate. Rendered as digits, since it is a magnitude and
        the symbol alphabet is deliberately semantically empty."""
        pct = round(100 * theta.p)
        return [vocab.PREAMBLE] + vocab.number(pct) + [encoding.separator]

    def posterior(self, history: list, k: int) -> Distribution:
        """Exact Bayes over the grid. No approximation and no conjugate shortcut
        needed -- the grid is small enough to enumerate, which is the point of
        discretizing it.
        """
        n = self.grid_size(k)
        ones = sum(1 for _, a in history if isinstance(a, Outcome) and a.value == 1)
        zeros = sum(1 for _, a in history if isinstance(a, Outcome) and a.value == 0)

        weights = []
        for i in range(n):
            p = (i + 1) / (n + 1)
            weights.append((p ** ones) * ((1.0 - p) ** zeros))
        total = sum(weights)
        if total == 0.0:
            weights, total = [1.0] * n, float(n)

        p_next = sum(w * ((i + 1) / (n + 1)) for i, w in enumerate(weights)) / total
        return {1: p_next, 0: 1.0 - p_next}

    def permuted_alphabet_check(self, rng: Random) -> bool:
        """A2. Passes without repair, and it is the only family in the set that
        does: there are no perceptual stimuli here to leak through, only two
        arbitrary outcome labels."""
        k = 2
        theta = self.sample_theta(k, rng)
        enc = self.sample_encoding(rng)

        shuffled = list(vocab.SYMBOL_IDS)
        rng.shuffle(shuffled)
        perm = dict(zip(vocab.SYMBOL_IDS, shuffled))
        enc2 = Encoding(
            enc.name,
            (perm[enc.outcome_symbols[0]], perm[enc.outcome_symbols[1]]),
            enc.separator,
        )

        for v in (0, 1):
            base = self.render(enc, Outcome(v))
            expected = [perm.get(t, t) for t in base]
            if expected != self.render(enc2, Outcome(v)):
                return False

        # The distribution must be untouched by relabelling: a rate is a fact
        # about theta, not about which token names the outcome.
        d1 = self.evaluate(theta, Query(0))
        d2 = self.evaluate(theta, Query(1))
        return d1 == d2
