"""What the harness needs from a family beyond Task Spec section 7.

Section 7 says "the harness knows nothing else about any family", and that is the
constraint this file exists to keep honest.  Everything here is either

  * a **generic resolver** -- a way of getting something section 7 implies but
    does not expose, using no family-specific knowledge, or
  * an **optional capability protocol** -- a method a family may add, which the
    harness detects by structure and never by name.

Task Spec section 8 step 3's gate is "all three run through the same harness with
zero family-specific code in the harness".  A resolver that dispatches on
`family.name` would violate it silently, so there are none; where a capability is
missing the harness raises with the family's name in the message, which is loud.

Every gap recorded here is also recorded in `docs/10-harness-findings.md` with
what the protocol should gain.  They are findings about section 7, not
workarounds to be forgotten.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from random import Random
from typing import Any, Callable, Protocol, runtime_checkable

Theta = Any
Encoding = Any
Query = Any
Answer = Any
Distribution = dict[Any, float]


class ProtocolGap(TypeError):
    """A family does not expose something the requested mode needs.

    Deliberately an exception and not a fallback.  Task Spec section 8 step 3
    says knowing which families cannot support L2 *is a result*; a harness that
    quietly degrades a family to a level it can run turns that result into
    silence.
    """


# --------------------------------------------------------------------------
# Gap 1: constructing an Answer from a distribution key
# --------------------------------------------------------------------------


def answer_type(family: Any) -> Callable[[Any], Answer]:
    """The callable that turns a distribution key back into an Answer object.

    Section 7 says `evaluate` returns `Answer | Distribution` and section 1.3
    requires that stochastic families return the distribution so the *harness*
    samples.  Nothing in section 7 says how a sampled key becomes an Answer, so a
    stochastic family cannot in fact be rendered by a harness that knows only
    section 7.  `a3_test.py` hit the same wall and said so.

    Resolved generically, in order of how explicit the family was being:

      1. `family.answer_type`  -- the field section 7 should have
      2. `family.output_type`  -- what the algebraic families already declare
      3. a conventionally-named class in the family's own module

    Step 3 is a concession and is the reason this is a finding rather than a
    fix: it reads a module namespace, which is not part of any contract.
    """
    for attr in ("answer_type", "output_type"):
        got = getattr(family, attr, None)
        if got is not None:
            return got

    import importlib

    module = importlib.import_module(type(family).__module__)
    for candidate in ("Answer", "Outcome", "Category"):
        cls = getattr(module, candidate, None)
        if cls is not None:
            return cls

    raise ProtocolGap(
        f"{type(family).__name__} returns a distribution from evaluate() but "
        "exposes no way to build an Answer from a distribution key. Add an "
        "`answer_type` attribute (see docs/10-harness-findings.md)."
    )


def draw_answer(family: Any, theta: Theta, query: Query, rng: Random) -> Answer:
    """Evaluate the oracle, sampling if it is distribution-valued.

    Section 1.3 and interface finding 3: `evaluate` takes no rng, so a stochastic
    family returns P(Y) and sampling happens here.  That is what keeps seeded
    reconstructibility in the harness, where section 7 already requires it.
    """
    out = family.evaluate(theta, query)
    if isinstance(out, dict):
        keys = sorted(out)
        return answer_type(family)(rng.choices(keys, weights=[out[k] for k in keys])[0])
    return out


# --------------------------------------------------------------------------
# Gap 2: the L3 target needs a query -- docs/03 finding 1, handoff section 2.1
# --------------------------------------------------------------------------

# Which route each family took, so the migration state is a readable fact rather
# than something the harness silently absorbs. Keyed by family class name.
POSTERIOR_ROUTE: dict[str, str] = {}


def _posterior_takes_query(family: Any) -> bool:
    """Does this family's posterior accept an explicit pending query?

    Decided by signature inspection once, not by catching TypeError.  Catching
    would swallow a TypeError raised *inside* posterior and silently reroute to
    the other convention -- the failure would look like a convention mismatch
    rather than the bug it is.
    """
    try:
        params = inspect.signature(family.posterior).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins, C callables
        return False
    return "query" in params


def answer_distribution(
    family: Any, history: list, query: Query, k: int
) -> Distribution:
    """The L3 target: P(y | history, *this query*, k).

    **The decision handoff section 2.1 asked the harness to make, made here and
    made once.**  The harness never calls `family.posterior` anywhere else.

    Adopted: `posterior(history, k, query=...)` is the interface, and the trailing
    `(query, None)` convention is a supported migration path, not a second
    interface.  Reasons for preferring the explicit parameter:

      * The marginal and the conditional are different quantities.  Under the
        trailing-None convention which one you get depends on the *shape of an
        argument*, so any harness that trims history to a window, or appends the
        pending query at a different point, silently switches the meaning of the
        return value.  A separate parameter makes the two distinguishable at the
        call site.
      * docs/03 finding 1 is that a family guessing differently emits the wrong
        target *silently*.  A parameter a family must accept is at least a place
        where the question is asked.

    Neither route is a guarantee: a family can accept `query` and ignore it.  The
    guard that actually bites is `check_query_sensitivity` below, which is
    family-agnostic and does not care which route was taken.
    """
    if _posterior_takes_query(family):
        POSTERIOR_ROUTE[type(family).__name__] = "explicit-query"
        return family.posterior(history, k, query=query)
    POSTERIOR_ROUTE[type(family).__name__] = "trailing-none"
    return family.posterior(list(history) + [(query, None)], k)


def check_query_sensitivity(
    family: Any, k: int, rng: Random, n_queries: int = 24
) -> tuple[bool, str]:
    """Does the L3 target actually depend on the query being asked?

    The check that matters, and the reason the signature question above is less
    important than it looks.  It is family-agnostic, it works under either
    convention, and it catches the failure docs/03 finding 1 describes:

        parity is fully identified after one observation, yet its answer
        distribution marginalized over queries is exactly 0.5/0.5.

    A family returning that marginal as its L3 target trains maximal uncertainty
    about a rule already pinned down -- worst on the families with the most
    structure.

    Returns (sensitive, detail).  **Not sensitive is not automatically a bug:**
    junk_random's target is uniform whatever is asked, and that is correct.  So
    this returns a fact for a per-family test to interpret, rather than a verdict
    -- a check that cannot distinguish those two cases would be worse than none.
    """
    theta = family.sample_theta(k, rng)
    history: list = []
    for _ in range(3):
        q = family.sample_query(theta, history, rng)
        history.append((q, draw_answer(family, theta, q, rng)))

    seen: list[tuple[Any, Distribution]] = []
    for _ in range(n_queries):
        q = family.sample_query(theta, history, rng)
        seen.append((q, answer_distribution(family, history, q, k)))

    first = seen[0][1]
    for q, dist in seen[1:]:
        if any(abs(dist.get(key, 0.0) - first.get(key, 0.0)) > 1e-9 for key in set(dist) | set(first)):
            return True, f"target moves between queries {seen[0][0]!r} and {q!r}"
    return False, (
        f"{n_queries} distinct queries all give the same target "
        f"{ {kk: round(v, 4) for kk, v in first.items()} } -- either the family is "
        "genuinely query-independent (junk, calibration exemplars) or it is "
        "returning the marginal, which is docs/03 finding 1"
    )


# --------------------------------------------------------------------------
# Gap 3: L2 needs rho_e^-1, which section 7 does not have
# --------------------------------------------------------------------------


@runtime_checkable
class SupportsParse(Protocol):
    """`parse_query` -- the inverse of `render` on the query channel.

    Section 2.1 says "the oracle answers the query the model actually asked, not
    the teacher's", and calls the resulting error-recovery lesson free.  It is
    not free: it requires reading the model's emitted tokens back into a Query,
    and section 7 exposes `render` with no inverse.  Without it a harness cannot
    know what was asked, so it cannot answer what was asked.

    A6 makes the same demand from the other side -- "responds sensibly to
    malformed and invalid queries" presupposes something that can *detect*
    malformed, which is a parser.  A6 is not satisfiable under section 7 as
    written.

    Contract: return None for tokens that do not name a legal query.  None is not
    a failure -- it is the A6 case, and the harness answers it with ERR.
    """

    def parse_query(self, encoding: Encoding, tokens: list[int]) -> Query | None: ...


# --------------------------------------------------------------------------
# Gap 4: a partial reveal, which is the dial of Task Spec section 8 step 5
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Reveal:
    """A partially-stated theta: the preamble, and what it commits to.

    `consistent` is the load-bearing field.  The sweep's x-axis is the *measured*
    residual entropy H(y | context), not the nominal dial setting, and computing
    it means enumerating theta consistent with what the preamble said.  A family
    that renders a partial preamble without exposing that predicate has given the
    harness something it cannot measure.
    """

    tokens: list[int]
    n_slots: int
    n_revealed: int
    consistent: Callable[[Theta], bool]

    @property
    def fraction(self) -> float:
        return self.n_revealed / self.n_slots if self.n_slots else 1.0


@runtime_checkable
class SupportsPartialReveal(Protocol):
    """`partial_preamble` -- state a *fraction* of theta.

    Section 7's `preamble` is all-or-nothing, which makes L0 and L1 two points
    with nothing between them.  Section 8 step 5 requires exactly what is between
    them: "vary residual entropy continuously -- not four discrete levels".

    The implementation requirement that matters, and it is not obvious: the
    rendered preamble should be **the same length at every fraction**, with
    unrevealed slots rendered as an explicit unknown marker.  Otherwise the dial
    moves sequence length and supervised-token count alongside residual entropy,
    and hazard 7 (an 85% length difference between encodings) is the same failure
    arriving from the other direction -- the sweep would confound what it is
    measuring with how much context it is measuring it in.
    """

    def partial_preamble(
        self, theta: Theta, encoding: Encoding, fraction: float, rng: Random
    ) -> Reveal: ...


@runtime_checkable
class SupportsEnumeration(Protocol):
    """`enumerate_theta` -- the hypothesis space, for exact targets and entropy.

    Already present on the algebraic families for composite posteriors.  The
    harness uses it for the section 8 step 2 gate ("L3 targets match a
    brute-force enumeration of consistent theta") and for the sweep's measured
    x-axis.  Families whose Theta is too large to enumerate simply do not offer
    it, and the harness says so rather than approximating.
    """

    def enumerate_theta(self, k: int) -> list[Theta]: ...
