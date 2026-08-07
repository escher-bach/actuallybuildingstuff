"""Composition of task families -- Task Spec section 1.1, taken seriously.

    T1 o T2 is defined when the codomain of f2 lies in the query space of T1:
        theta = (theta1, theta2)
        f((theta1, theta2), x) = f1(theta1, f2(theta2, x))
    with P_Theta the product of the component samplers and rho_e shared.

Section 1.1 is emphatic that composition "should not be treated as a
convenience", and gives three separate reasons:

  1. **It is the natural instantiation of k.** Depth is the difficulty knob that
     does not require inventing a new family per level. So a composite must be
     n-ary and its depth must be driven by k -- a fixed depth-2 pair is not an
     instantiation of anything.
  2. **It is the cheapest route to A4.** A solver ignoring intended structure
     must generally enumerate the composite; the intended solver composes. That
     is a claim about the product hypothesis space, and it is checkable.
  3. **It is what lets a finite basis cover an infinite target space.** Without
     closure you are enumerating, and enumeration cannot cover.

And it names the condition that makes a composite worth having:

  > A5 (semantic coherence) is precisely the condition under which a composite
  > is meaningful rather than a **type-checking accident**.

That sentence is the design requirement this module exists to meet. A type check
is necessary and explicitly NOT sufficient: `Composite` therefore ships two
gates, and the second one is the one that matters.

--------------------------------------------------------------------------
The two gates
--------------------------------------------------------------------------

**Gate 1 -- types.** codomain(f_inner) must lie in X_outer. Cheap, mechanical,
catches the gross errors, and is the thing the spec warns is not enough.

**Gate 2 -- semantic coherence, operationalized.** A composite is a type-checking
accident when a component is not doing any work. Three failure shapes, all
computable by sampling:

  * the answer never varies with theta_inner  -> the outer family ignores what
    the inner one computed; the composite is the outer family wearing a costume
  * the answer never varies with theta_outer  -> the outer family is a constant
    or a bijection-of-nothing; composing bought nothing
  * the answer never varies with the query    -> the composite is constant

Each is exactly what happens if you compose junk with something real, or
something real with junk, and each passes the type check. `coherence_report`
below computes all three and `compose` refuses the composition on failure. This
is not a proof of meaningfulness -- it is a falsifier, and a composite that
passes has merely failed to be caught being vacuous.

--------------------------------------------------------------------------
A finding this module produced
--------------------------------------------------------------------------

**The implemented basis had essentially no composition closure.** Composition
needs codomain(f2) inside X1, and our families overwhelmingly map a structured
query to a SINGLE LABEL. A label cannot be fed back in as a query. So every
register row citing "compose with a second family" as its A4 repair was citing an
operation with no legal instance -- silently, because nothing checks that a named
repair is performable, and the gap exists at the level of the basis rather than
of any row.

`closure_report` makes that measurable rather than anecdotal.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from random import Random
from typing import Any, Protocol, Sequence, runtime_checkable

Distribution = dict[Any, float]


class CompositionTypeError(TypeError):
    """codomain(f_inner) is not inside X_outer. A design error, not a runtime one."""


class CompositionIncoherent(ValueError):
    """The composite type-checks but a component does not affect the answer.

    Separate from CompositionTypeError because the remedy is different: a type
    error means you composed the wrong pair, an incoherence means you composed a
    pair that does nothing, and the second is the one section 1.1 warns about.
    """


@runtime_checkable
class Enumerable(Protocol):
    """A family that can list its hypothesis space at a given k.

    Optional, and only families that implement it can participate in a
    composite's exact L3 target -- the product posterior has to enumerate
    Theta_1 x ... x Theta_n, and there is no way to do that from `sample_theta`
    alone. Recorded as a protocol extension rather than bolted onto TaskFamily,
    because most families will not have it and requiring it would be a lie.
    """

    def enumerate_theta(self, k: int) -> Sequence[Any]: ...


@dataclass(frozen=True)
class ComposedTheta:
    """theta = (theta_1, ..., theta_n), outermost first.

    Kept as a tuple rather than flattened so a composite's hidden parameter stays
    inspectable component-wise; the L3 posterior enumerates the product and a
    flattened theta would make that opaque.
    """

    parts: tuple  # outermost first

    @property
    def outer(self):
        return self.parts[0]

    @property
    def inner(self):
        return self.parts[-1]


# --------------------------------------------------------------------------
# Gate 2: semantic coherence
# --------------------------------------------------------------------------


@dataclass
class CoherenceReport:
    """What varying each component actually does to the answer."""

    varies_with_query: bool
    varies_with_each_theta: tuple[bool, ...]
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.varies_with_query and all(self.varies_with_each_theta)

    def failures(self) -> list[str]:
        out = []
        if not self.varies_with_query:
            out.append("answer never varies with the query -- composite is constant")
        for i, v in enumerate(self.varies_with_each_theta):
            if not v:
                pos = "outermost" if i == 0 else ("innermost" if i == len(self.varies_with_each_theta) - 1 else f"stage {i}")
                out.append(
                    f"answer never varies with theta of the {pos} component "
                    "-- that stage is doing no work, so the composite is a "
                    "type-checking accident in section 1.1's sense"
                )
        return out


def coherence_report(composite: "Composite", k: int, rng: Random, trials: int = 40) -> CoherenceReport:
    """Sample-based falsifier for 'this composite is meaningful'.

    Method: hold everything fixed but one thing, vary that thing, and see whether
    the answer ever moves. Cheap, and it catches every way we have actually
    managed to build a vacuous composite.

    It cannot prove coherence -- a component could matter only on inputs we did
    not sample. Stated plainly because a check that overclaims is worse than one
    that does not exist (hazard 6 in docs/07).
    """
    n = len(composite.stages)
    base_thetas = [composite.sample_theta(k, rng) for _ in range(trials)]
    base = base_thetas[0]

    # Does the answer move with the query, holding theta fixed?
    answers = set()
    for _ in range(trials):
        q = composite.sample_query(base, [], rng)
        try:
            answers.add(_key(composite.evaluate(base, q)))
        except CompositionTypeError:
            raise
    varies_with_query = len(answers) > 1

    # Does the answer move with each component's theta, holding the rest fixed?
    varies = []
    for i in range(n):
        seen = set()
        queries = [composite.sample_query(base, [], rng) for _ in range(8)]
        for cand in base_thetas:
            parts = list(base.parts)
            parts[i] = cand.parts[i]
            th = ComposedTheta(tuple(parts))
            for q in queries:
                seen.add((_key(q), _key(composite.evaluate(th, q))))
        # If component i matters, some query must map to two different answers
        by_query: dict = {}
        moved = False
        for qk, ak in seen:
            if qk in by_query and by_query[qk] != ak:
                moved = True
                break
            by_query[qk] = ak
        varies.append(moved)

    return CoherenceReport(
        varies_with_query=varies_with_query,
        varies_with_each_theta=tuple(varies),
        detail=f"{n} stages, k={k}, {trials} theta draws",
    )


def _key(obj) -> Any:
    """Hashable identity for an answer or query object."""
    if hasattr(obj, "__dataclass_fields__"):
        return (type(obj).__name__, tuple(sorted(obj.__dict__.items())))
    return obj


# --------------------------------------------------------------------------
# The composite
# --------------------------------------------------------------------------


class Composite:
    """T_1 o T_2 o ... o T_n, presented to the harness as an ordinary family.

    The harness must not be able to tell a composite from a primitive family --
    section 7 says it knows nothing about any family beyond the protocol. If it
    could, composition would be a harness feature rather than a property of the
    basis, and the closure argument in section 1.1 would not go through.
    """

    def __init__(self, stages: Sequence, name: str | None = None):
        if len(stages) < 2:
            raise ValueError("a composite needs at least two stages")
        self.stages = tuple(stages)  # outermost first
        self.outer = self.stages[0]
        self.inner = self.stages[-1]
        self.name = name or "_of_".join(s.name for s in self.stages)

        self.supports_L2 = False  # see teacher_query
        self.emits_trace = any(getattr(s, "emits_trace", False) for s in self.stages)
        self.stochastic = any(getattr(s, "stochastic", False) for s in self.stages)

    # ---- section 1.1 ----------------------------------------------------

    def sample_theta(self, k: int, rng: Random) -> ComposedTheta:
        """P_Theta is the product of the component samplers.

        The A4 headroom argument lives here: the composite hypothesis space is
        the product, so a solver ignoring the intended structure must enumerate
        it while the intended solver identifies each stage in turn.
        """
        return ComposedTheta(tuple(s.sample_theta(k, rng) for s in self.stages))

    def evaluate(self, theta: ComposedTheta, query):
        value = query
        for stage, th in zip(reversed(self.stages), reversed(theta.parts)):
            value = stage.evaluate(th, value)
            if isinstance(value, dict):
                raise CompositionTypeError(
                    f"{stage.name} is distribution-valued; composing through a "
                    "stochastic stage would require pushing a distribution "
                    "through the next oracle, which section 1.1 does not define"
                )
        return value

    # ---- encoding: rho_e is SHARED (section 1.1) ------------------------

    def sample_encoding(self, rng: Random):
        """One encoding per stage, but only two are ever emitted.

        The model sees queries in the innermost surface and answers in the
        outermost; intermediate values are never rendered. Sampling per stage
        anyway keeps `render` total over every object the composite can produce,
        which matters for traces (section 1.2), where the intermediates DO appear.
        """
        return tuple(s.sample_encoding(rng) for s in self.stages)

    def render(self, encoding, obj) -> list[int]:
        last_error = None
        for stage, enc in zip(self.stages, encoding):
            try:
                return stage.render(enc, obj)
            except TypeError as e:
                last_error = e
        raise TypeError(f"{self.name} cannot render {type(obj).__name__}") from last_error

    def preamble(self, theta: ComposedTheta, encoding) -> list[int] | None:
        parts: list[int] = []
        for stage, th, enc in zip(self.stages, theta.parts, encoding):
            p = stage.preamble(th, enc)
            if p:
                parts += p
        return parts or None

    # ---- protocol remainder --------------------------------------------

    def sample_query(self, theta: ComposedTheta, history: list, rng: Random):
        return self.inner.sample_query(theta.inner, history, rng)

    def teacher_query(self, theta: ComposedTheta, history: list):
        raise NotImplementedError(
            f"{self.name} does not support L2. A composite's q* is NOT the "
            "composition of its components' policies: the model can only ask in "
            "the innermost query space, while the outer stages' informative "
            "queries live in their own spaces and are reachable only through "
            "whatever the inner stages happen to map onto. Deriving a composite "
            "q* is real work -- plausibly the most interesting open item this "
            "module leaves -- and faking it would put a family with no query "
            "policy into the L2 column."
        )

    def trace(self, theta: ComposedTheta, query) -> list | None:
        """Inner derivation first, then each outer stage in turn.

        Section 1.2 note: the intermediate values are the natural place to thin
        FIRST. They are the seams, and revealing them collapses the composite
        into n independent easier problems -- a thinning schedule with a clear
        semantics rather than an arbitrary one.
        """
        steps: list = []
        value = query
        for stage, th in zip(reversed(self.stages), reversed(theta.parts)):
            s = stage.trace(th, value)
            if s is None:
                return None
            steps += list(s)
            value = stage.evaluate(th, value)
        return steps

    def posterior(self, history: list, k: int) -> Distribution:
        """Exact L3 target by enumerating the PRODUCT hypothesis space.

        Requires every stage to implement `enumerate_theta`. Where they do, this
        is exact and no approximation is involved; where they do not, it raises
        rather than approximating, because an approximate calibration target
        trains miscalibration and does so silently (hazard 4 in docs/07).

        Cost is the product of the component space sizes, which is exactly the
        thing that makes composites hard and therefore also expensive to
        calibrate. That tension is real and is not resolved here: at depth 3 with
        modest components the enumeration stops being affordable, and the honest
        options are to shrink components, cap depth for L3 specifically, or drop
        L3 for deep composites. Recorded rather than papered over.
        """
        missing = [s.name for s in self.stages if not isinstance(s, Enumerable)]
        if missing:
            raise NotImplementedError(
                f"{self.name} cannot compute an exact posterior: "
                f"{missing} do not implement enumerate_theta. Refusing to "
                "approximate -- an approximate L3 target trains miscalibration."
            )

        pending = history[-1][0] if history and history[-1][1] is None else None
        if pending is None:
            raise NotImplementedError(
                "composite posterior needs a pending query: append (query, None) "
                "to the history. Marginalizing over the query space is not the "
                "L3 target -- see docs/03 finding 1."
            )

        spaces = [list(s.enumerate_theta(k)) for s in self.stages]
        counts: dict = {}
        for combo in itertools.product(*spaces):
            th = ComposedTheta(tuple(combo))
            if not _consistent(self, th, history):
                continue
            a = _key(self.evaluate(th, pending))
            counts[a] = counts.get(a, 0.0) + 1.0

        total = sum(counts.values())
        if not total:
            return {}
        return {a: c / total for a, c in counts.items()}

    def permuted_alphabet_check(self, rng: Random) -> bool:
        """A2 for a composite reduces to A2 for its stages.

        Sound because relabelling acts on the shared alphabet and each stage is
        equivariant under it; a composition of equivariant maps is equivariant.
        Stated rather than assumed because it is the one A-check a composite
        genuinely inherits -- A3 and A4 do not work this way, and A5 explicitly
        does not, which is what the coherence gate is for.
        """
        return all(s.permuted_alphabet_check(rng) for s in self.stages)


def _consistent(composite: "Composite", theta: ComposedTheta, history) -> bool:
    for q, a in history:
        if a is None:
            continue
        if _key(composite.evaluate(theta, q)) != _key(a):
            return False
    return True


# --------------------------------------------------------------------------
# Building composites
# --------------------------------------------------------------------------


def compose(
    *stages,
    types: Sequence,
    name: str | None = None,
    check_coherence: bool = True,
    k: int = 1,
    rng: Random | None = None,
) -> Composite:
    """Build a composite, outermost stage first, through BOTH gates.

    `types[i]` is the type stage i consumes; the innermost stage's output type
    must match the next-outer stage's input type, and so on outward. The check is
    explicit rather than inferred because the failure it prevents is silent:
    Python would happily compose mismatched families and produce garbage at the
    first attribute access, or coincidentally work for one shape of input.

    `check_coherence` runs gate 2 -- section 1.1's own condition for a composite
    being meaningful rather than a type-checking accident. Defaults ON. Turning
    it off is legitimate only when deliberately constructing a vacuous composite
    to test the gate itself.
    """
    if len(stages) != len(types):
        raise ValueError("one consumed-type per stage is required")

    # Gate 1: types chain from inner to outer.
    for i in range(len(stages) - 1, 0, -1):
        produced = types[i - 1]
        consumed_by_outer = types[i - 1]
        inner_out = getattr(stages[i], "output_type", None)
        if inner_out is not None and inner_out is not consumed_by_outer:
            raise CompositionTypeError(
                f"cannot compose {stages[i - 1].name} o {stages[i].name}: "
                f"{stages[i].name} returns {inner_out.__name__} but "
                f"{stages[i - 1].name} consumes {consumed_by_outer.__name__}. "
                "Section 1.1 requires the codomain of the inner oracle to lie in "
                "the query space of the outer one."
            )
        _ = produced

    c = Composite(stages, name=name)

    if check_coherence:
        report = coherence_report(c, k=k, rng=rng or Random(0))
        if not report.ok:
            raise CompositionIncoherent(
                f"{c.name} type-checks but is not semantically coherent:\n  "
                + "\n  ".join(report.failures())
                + "\nSection 1.1: A5 is precisely the condition under which a "
                "composite is meaningful rather than a type-checking accident."
            )
    return c


# --------------------------------------------------------------------------
# Closure: how much can this basis actually compose?
# --------------------------------------------------------------------------


def closure_report(families: Sequence, k: int = 1, rng: Random | None = None) -> str:
    """Which ordered pairs compose, and which are refused, and why.

    Closure is a property of the BASIS, not of any family, which is why no row
    could see that it was missing. Section 1.1 rests on it -- "without closure
    you are enumerating, and enumeration cannot cover" -- so it deserves to be
    measured and reported like coverage, rather than assumed.
    """
    rng = rng or Random(0)
    lines = [f"composition closure over {len(families)} families (k={k})", ""]
    ok = typed_out = incoherent = 0

    for outer, inner in itertools.permutations(families, 2):
        try:
            probe_rng = Random(1)
            theta = inner.sample_theta(k, probe_rng)
            q = inner.sample_query(theta, [], probe_rng)
            produced = inner.evaluate(theta, q)
        except Exception as e:  # noqa: BLE001 - a family that cannot even be sampled
            lines.append(f"  SKIP  {outer.name} o {inner.name}: inner unsamplable ({e.__class__.__name__})")
            continue

        if isinstance(produced, dict):
            typed_out += 1
            continue
        if not isinstance(produced, type(q)):
            typed_out += 1
            continue

        c = Composite([outer, inner])
        try:
            c.evaluate(c.sample_theta(k, Random(2)), q)
        except Exception:  # noqa: BLE001
            typed_out += 1
            continue

        report = coherence_report(c, k=k, rng=rng)
        if report.ok:
            ok += 1
            lines.append(f"  OK    {outer.name} o {inner.name}")
        else:
            incoherent += 1
            lines.append(f"  VOID  {outer.name} o {inner.name}: {report.failures()[0]}")

    total = len(families) * (len(families) - 1)
    lines += [
        "",
        f"{ok} composable, {incoherent} type-check but incoherent, "
        f"{typed_out} type-incompatible, of {total} ordered pairs",
        f"closure = {ok / total:.1%} of ordered pairs" if total else "",
    ]
    return "\n".join(lines)
