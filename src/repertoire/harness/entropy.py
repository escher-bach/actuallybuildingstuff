"""Exact posteriors and residual entropy, by enumeration over Theta.

Three things need this and they are the same computation:

  * **The section 8 step 2 gate** -- "L3 targets match a brute-force enumeration
    of consistent theta".  Written generically here rather than per family, so
    the gate is one test applied to everything with an enumerable Theta instead
    of a promise repeated in each family's test file.
  * **The Bayes floor.**  A loss curve's floor is not something to guess at when
    it can be computed.  Having it turns "the model stopped improving" into "the
    model is 0.03 nats/token above optimal", and turns a *negative* excess into a
    loud signal that something is leaking.
  * **The sweep's x-axis.**  Section 8 step 5 says vary residual entropy.  The
    honest x-axis is the residual entropy actually obtained, in nats, not the
    dial setting that was supposed to produce it -- the map from one to the other
    is nonlinear and family-specific, and plotting against the dial would report
    a curve in units nobody can compare across families.

--------------------------------------------------------------------------
The prior, which is where this goes wrong quietly
--------------------------------------------------------------------------

`enumerate_theta` gives the *support* of P_Theta.  It does not give P_Theta.
Section 1 is explicit that the meaning of a family is given entirely by its
sampler, so a posterior computed against a uniform prior over the support is the
posterior of a *different family* whenever the sampler is not uniform -- and it
will look completely reasonable.

This is not hypothetical.  `ParityIdentificationFamily.sample_theta` draws a
subset size uniformly from 1..d and then a subset of that size, which puts far
more mass on small and large subsets than uniform-over-subsets does.  Its own
`posterior` enumerates uniformly.  At d=8 the implied prior on a singleton subset
is off by a factor of about 8.

`check_prior_matches_sampler` below tests it by sampling, and it is meant to
fail.  Where it fails the family should supply `prior_weight`; until it does, the
L3 target for that family is exact for a family we are not training on.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from random import Random
from typing import Any

from .protocol import ProtocolGap, Reveal, SupportsEnumeration

Theta = Any
Query = Any


def prior_weight(family: Any, theta: Theta, k: int) -> float:
    """P_Theta(theta | k) up to a constant. Uniform unless the family says otherwise."""
    fn = getattr(family, "prior_weight", None)
    return 1.0 if fn is None else float(fn(theta, k))


def answer_likelihood(family: Any, theta: Theta, query: Query) -> dict[Any, float]:
    """P(y | theta, x) as a distribution, whether or not the oracle is stochastic."""
    out = family.evaluate(theta, query)
    if isinstance(out, dict):
        return out
    return {_key(out): 1.0}


def _key(answer: Any) -> Any:
    """The distribution key an Answer corresponds to.

    Families key their distributions by the Answer's single payload field and
    return bare Answer objects when deterministic, so the two have to be brought
    into one space.  Done structurally -- one dataclass field -- rather than by
    knowing any family's field names.
    """
    fields = getattr(type(answer), "__dataclass_fields__", None)
    if fields and len(fields) == 1:
        return getattr(answer, next(iter(fields)))
    return answer


@dataclass
class BeliefState:
    """Weights over an enumerated Theta, after conditioning."""

    thetas: list[Theta]
    weights: list[float]
    k: int

    @property
    def n_alive(self) -> int:
        return sum(1 for w in self.weights if w > 1e-12)

    def entropy_of_theta(self) -> float:
        return -sum(w * math.log(w) for w in self.weights if w > 1e-12)

    def answer_distribution(self, family: Any, query: Query) -> dict[Any, float]:
        out: dict[Any, float] = {}
        for theta, w in zip(self.thetas, self.weights):
            if w <= 1e-12:
                continue
            for key, p in answer_likelihood(family, theta, query).items():
                out[key] = out.get(key, 0.0) + w * p
        total = sum(out.values())
        return {kk: v / total for kk, v in out.items()} if total > 0 else out

    def residual_entropy(self, family: Any, query: Query) -> float:
        """H(y | context, this query), in nats."""
        dist = self.answer_distribution(family, query)
        return -sum(p * math.log(p) for p in dist.values() if p > 0)


def belief_state(
    family: Any,
    k: int,
    history: list,
    reveal: Reveal | None = None,
) -> BeliefState:
    """Bayes over the enumerated Theta, given the preamble and the history.

    Weighted rather than filtered.  Filtering to "consistent" theta is only
    correct for a deterministic oracle; a stochastic family's observation makes
    some theta *less likely* rather than impossible, and a filter would discard
    exactly the families section 1.3 exists to include.
    """
    if not isinstance(family, SupportsEnumeration):
        raise ProtocolGap(
            f"{type(family).__name__} has no enumerate_theta, so no exact "
            "posterior, Bayes floor or measured residual entropy is available"
        )
    thetas = list(family.enumerate_theta(k))
    weights = [prior_weight(family, t, k) for t in thetas]

    if reveal is not None:
        weights = [w if reveal.consistent(t) else 0.0 for t, w in zip(thetas, weights)]

    for q, a in history:
        if q is None or a is None:
            continue
        key = _key(a)
        weights = [
            w * answer_likelihood(family, t, q).get(key, 0.0) if w > 0 else 0.0
            for t, w in zip(thetas, weights)
        ]

    total = sum(weights)
    if total <= 0:
        raise ValueError(
            "no theta survives the history -- the oracle produced an answer no "
            "hypothesis in enumerate_theta can explain. Either the enumeration is "
            "incomplete or the family's evaluate disagrees with it."
        )
    return BeliefState(thetas, [w / total for w in weights], k)


def brute_force_answer_distribution(
    family: Any, k: int, history: list, query: Query, reveal: Reveal | None = None
) -> dict[Any, float]:
    """The section 8 step 2 gate's reference implementation.

    A family's own `posterior` is compared against this. They are computed
    differently on purpose -- a family may factor its belief state (parity does),
    this never does -- so agreement is evidence and not tautology.
    """
    return belief_state(family, k, history, reveal).answer_distribution(family, query)


# --------------------------------------------------------------------------
# The sweep's x-axis and the Bayes floor
# --------------------------------------------------------------------------


@dataclass
class EntropyReport:
    """Residual entropy, split into the two latents an episode actually hides.

    **The split is a finding, not bookkeeping**, and it was found by the floor
    being wrong.  The first L1 run against this module measured a trial-1 loss of
    4.18 nats against a computed floor of 0.69 and looked like a model 3.5 nats
    from optimal.  It was not: the stub's answers are binary, so 0.69 is the
    entire uncertainty *about the rule*, and the other 3.5 nats are uncertainty
    about **which token denotes which answer** -- because section 1 samples the
    encoding `e` on the same footing as theta and hides it too.

    So there are two quantities and conflating them overstates the model's
    excess by whatever the notation is worth, which at trial 1 is most of it:

        rule       H(y | context, e)   exact; what the dial of section 8 step 5
                                       moves, and the sweep's x-axis
        notation   the extra nats from not knowing e; bounded, not exact

    `rule` is a genuine **lower bound** on achievable loss and `rule + notation`
    a genuine **upper bound**, so a final loss below `rule` is still a leak and
    still says so.  The band between them is honest ignorance, and it narrows
    within an episode as the encoding gets pinned down by use.

    **`theta_entropy` is a third quantity and it is not redundant.**  Task Spec
    section 2 defines the dial as the residual entropy of the *answer*, and for
    most families that tracks how determined theta is.  For a **balanced** family
    it does not, and the gap is enormous: measured on parity, H(y|context) sits
    at log 2 across the first five free observations while the surviving
    hypothesis set falls from 297 to 19.  The answer to a random query is a coin
    flip however well theta is known, right up until theta is known exactly.

    That is the same property docs/03 finding 1 is about, arriving in the x-axis
    instead of in the target.  A sweep plotted against `rule` alone would report
    that nothing changed over the half of the dial where almost everything
    changed, so both are logged and a divergence between them is a fact about the
    family rather than noise.
    """

    rule: float  # nats/answer -- exact, and the sweep's x-axis
    rule_per_trial: list[float]
    notation_upper: float  # nats/answer of encoding uncertainty, upper bound
    notation_per_trial: list[float]
    max_entropy: float
    n_episodes: int
    mean_alive_theta: float
    theta_entropy: float = 0.0  # H(theta | context), nats -- see the note below
    theta_per_trial: list[float] = field(default_factory=list)
    n_free: int = 0  # the dial settings this was measured at, carried with it
    reveal: float = 0.0

    @property
    def floor_lower(self) -> float:
        return self.rule

    @property
    def floor_upper(self) -> float:
        return self.rule + self.notation_upper

    def report(self) -> str:
        curve = " ".join(f"{x:.3f}" for x in self.rule_per_trial)
        ncurve = " ".join(f"{x:.3f}" for x in self.notation_per_trial)
        return (
            f"H(y|context,e) = {self.rule:.4f} nats  [rule]\n"
            f"    per trial: {curve}\n"
            f"  + notation <= {self.notation_upper:.4f} nats -> Bayes floor in "
            f"[{self.floor_lower:.4f}, {self.floor_upper:.4f}]\n"
            f"    per trial: {ncurve}\n"
            f"    ({self.mean_alive_theta:.1f} theta alive, max H {self.max_entropy:.3f})"
        )


def measure_residual_entropy(
    family: Any,
    k: int,
    spec,
    n_episodes: int = 64,
    seed0: int = 0,
) -> EntropyReport:
    """Exact H(y_t | context, e) per trial, plus a bound on the notation term.

    The belief state is updated incrementally across trials rather than rebuilt,
    which turns an O(T^2 |Theta|) measurement into O(T |Theta|).  That is the
    difference between this being usable on the modular family and not.

    The notation bound: after conditioning on the rule, the answer still has to
    be *named*, and a token never yet seen in the episode is -- by A2, and by the
    symbol pool being drawn uniformly -- exchangeable with every other unseen
    symbol.  So the unresolved mass costs at most log(number of unseen content
    symbols).  It is an upper bound and not exact because the two sources overlap:
    when several candidate answers are all unseen, not knowing which is which
    already absorbs the rule uncertainty rather than adding to it.
    """
    from .episode import build_reveal, episode_seed  # episode imports nothing here
    from .protocol import draw_answer

    # Measured at the SCORED trials only. The free observations of the second
    # dial are context, not targets, so their entropy is not what the loss sees
    # -- averaging them in would report an x-axis for an episode nobody trains on.
    rule_sums = [0.0] * spec.T
    notation_sums = [0.0] * spec.T
    theta_sums = [0.0] * spec.T
    alive_total = 0.0
    max_h = 0.0

    for e in range(n_episodes):
        rng = Random(episode_seed(family.name, k, spec.key, seed0 + e))
        theta = family.sample_theta(k, rng)
        enc = family.sample_encoding(rng)
        reveal = build_reveal(family, theta, enc, spec.reveal, rng)

        bs = belief_state(family, k, [], reveal)
        seen_tokens: set[int] = set(reveal.tokens)
        history: list = []

        for t in range(spec.total_trials):
            query = family.sample_query(theta, history, rng)
            seen_tokens.update(family.render(enc, query))

            if t >= spec.n_free:
                s = t - spec.n_free
                dist = bs.answer_distribution(family, query)
                rule_sums[s] += -sum(p * math.log(p) for p in dist.values() if p > 0)
                notation_sums[s] += _notation_bound(family, enc, dist, seen_tokens)
                theta_sums[s] += bs.entropy_of_theta()
                alive_total += bs.n_alive
                max_h = max(max_h, math.log(max(2, len(dist))))

            answer = draw_answer(family, theta, query, rng)
            seen_tokens.update(family.render(enc, answer))
            history.append((query, answer))
            bs = _condition(bs, family, query, answer)

    rule_per_trial = [s / n_episodes for s in rule_sums]
    notation_per_trial = [s / n_episodes for s in notation_sums]
    theta_per_trial = [s / n_episodes for s in theta_sums]
    return EntropyReport(
        rule=sum(rule_per_trial) / len(rule_per_trial),
        rule_per_trial=rule_per_trial,
        notation_upper=sum(notation_per_trial) / len(notation_per_trial),
        notation_per_trial=notation_per_trial,
        max_entropy=max_h,
        n_episodes=n_episodes,
        mean_alive_theta=alive_total / (n_episodes * spec.T),
        theta_entropy=sum(theta_per_trial) / len(theta_per_trial),
        theta_per_trial=theta_per_trial,
        n_free=spec.n_free,
        reveal=spec.reveal,
    )


def _condition(bs: BeliefState, family: Any, query: Any, answer: Any) -> BeliefState:
    """One Bayes update in place of a rebuild."""
    key = _key(answer)
    w = [
        wt * answer_likelihood(family, t, query).get(key, 0.0) if wt > 0 else 0.0
        for t, wt in zip(bs.thetas, bs.weights)
    ]
    z = sum(w)
    if z <= 0:
        raise ValueError("observation is impossible under every surviving theta")
    return BeliefState(bs.thetas, [x / z for x in w], bs.k)


def _notation_bound(family: Any, enc: Any, dist: dict, seen_tokens: set[int]) -> float:
    """Upper bound on the nats owed to not knowing which token names which answer."""
    from .. import vocab

    ctor = getattr(family, "answer_type", None) or getattr(family, "output_type", None)
    if ctor is None:
        return 0.0
    unresolved = 0.0
    for key, p in dist.items():
        if p <= 0:
            continue
        try:
            toks = family.render(enc, ctor(key))
        except Exception:
            return 0.0
        if any(t not in seen_tokens for t in toks):
            unresolved += p
    n_unseen = max(1, len(set(vocab.SYMBOL_IDS) - seen_tokens))
    return unresolved * math.log(n_unseen)


# --------------------------------------------------------------------------
# The check that is meant to fail
# --------------------------------------------------------------------------


def check_prior_matches_sampler(
    family: Any, k: int, rng: Random, n_samples: int = 4000, tol: float = 0.35
) -> tuple[bool, str]:
    """Does the assumed prior match what `sample_theta` actually draws?

    Section 1: "The meaning of the family is given entirely by its sampler."  An
    exact posterior computed against the wrong prior is exact for a family we are
    not training on, and nothing about it looks wrong.

    Compares the empirical frequency of each enumerated theta against the assumed
    prior_weight, and reports the largest relative deviation.  `tol` is a
    smoke-detector threshold and not a test statistic -- with n_samples draws over
    |Theta| cells the sampling noise alone is substantial, which is why this
    returns the offender and the size of the gap rather than a bare verdict.
    """
    if not isinstance(family, SupportsEnumeration):
        return True, "no enumerate_theta; nothing is assumed about the prior"

    thetas = list(family.enumerate_theta(k))
    if len(thetas) > 4096:
        return True, f"|Theta| = {len(thetas)}; too large to check by sampling"

    assumed = [prior_weight(family, t, k) for t in thetas]
    z = sum(assumed)
    assumed = [w / z for w in assumed]

    counts = Counter(family.sample_theta(k, rng) for _ in range(n_samples))
    unknown = sum(c for t, c in counts.items() if t not in set(thetas))
    if unknown:
        return False, (
            f"{unknown}/{n_samples} sampled theta are not in enumerate_theta -- "
            "the enumeration is not the support, so every exact target is wrong"
        )

    worst, worst_t = 0.0, None
    for t, p in zip(thetas, assumed):
        empirical = counts.get(t, 0) / n_samples
        if p <= 0:
            continue
        dev = abs(empirical - p) / p
        if dev > worst:
            worst, worst_t = dev, t
    ok = worst <= tol
    return ok, (
        f"largest relative deviation {worst:.2f} at {worst_t!r} "
        f"(|Theta|={len(thetas)}, n={n_samples}, tol={tol}). "
        + ("" if ok else "Supply prior_weight, or the exact L3 target is for a "
                         "different sampler than the one generating episodes.")
    )
