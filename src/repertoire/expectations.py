"""Machine-checkable expectations for the transfer matrix.

Repertoire Spec §11 step 6: **"Predicted decomposition recorded first; plants
recover. Else stop."** That gate is only meaningful if "the plants recovered" can
be evaluated without judgement after the fact. Prose in `docs/02` states what we
expect; this file states it as assertions a script can run against a matrix.

Why bother, given the predictions are already written down: the failure mode is
not forgetting the prediction, it is **reading the matrix and finding the
prediction that fits**. Vein §2.1 supplied the cautionary case — a fitted,
converged capability model assigning 74-98% skill mastery to students who scored
zero. Nobody in that chain was careless. The defence is committing to the
comparison before seeing the numbers, and a comparison in executable form is
harder to reinterpret than one in prose.

**This module deliberately does not import the matrix or the harness.** Neither
exists. It defines the expectations and the scoring; feeding it real numbers is a
later step, and writing it now is the point -- afterwards would be too late.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Outcome(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    UNTESTABLE = "untestable"  # required families missing from the matrix


@dataclass
class Expectation:
    """One prediction, with the check that decides it."""

    id: str
    kind: str  # instrument | prediction | sanity
    statement: str
    families: tuple[str, ...]
    rationale: str
    blocking: bool = False  # does failure stop the programme per §11 step 6?

    def check(self, m: "Matrix") -> tuple[Outcome, str]:
        raise NotImplementedError


class Matrix:
    """Minimal read-only view of an all-pairs transfer matrix.

    `s[i][j]` is S(T_j | m_i) -- structural content of family j given a model
    trained on family i. `diag[j]` is S(T_j), the intrinsic content.
    """

    def __init__(self, s: dict[str, dict[str, float]], diag: dict[str, float]):
        self.s = s
        self.diag = diag

    def has(self, *families: str) -> bool:
        return all(f in self.diag for f in families)

    def transfer(self, source: str, target: str) -> float:
        """Fraction of target's intrinsic content already supplied by source.

        1.0 means training on `source` left nothing to learn; 0.0 means it
        helped not at all. Normalizing by the diagonal is what makes cells
        comparable across families with different intrinsic content -- without
        it, the family with the most structure dominates every comparison.
        """
        base = self.diag[target]
        if base <= 0:
            return 0.0
        return max(0.0, min(1.0, 1.0 - (self.s[source][target] / base)))


# --------------------------------------------------------------------------
# Instrument validation -- these gate everything (Task Spec §8 step 4)
# --------------------------------------------------------------------------


@dataclass
class JunkReadsNearZero(Expectation):
    threshold: float = 0.05

    def check(self, m):
        if not m.has(*self.families):
            return Outcome.UNTESTABLE, "junk families absent from the matrix"
        worst = max(m.diag[f] for f in self.families)
        ok = worst <= self.threshold * max(m.diag.values())
        return (
            (Outcome.PASS if ok else Outcome.FAIL),
            f"max junk diagonal {worst:.4f} vs threshold "
            f"{self.threshold * max(m.diag.values()):.4f}",
        )


@dataclass
class PairClustersTightly(Expectation):
    min_transfer: float = 0.6

    def check(self, m):
        a, b = self.families
        if not m.has(a, b):
            return Outcome.UNTESTABLE, f"{a} or {b} absent"
        fwd, rev = m.transfer(a, b), m.transfer(b, a)
        ok = min(fwd, rev) >= self.min_transfer
        return (
            (Outcome.PASS if ok else Outcome.FAIL),
            f"transfer {a}->{b}={fwd:.3f}, {b}->{a}={rev:.3f}, "
            f"need both >= {self.min_transfer}",
        )


@dataclass
class PrerequisiteHasTheRightSign(Expectation):
    min_gap: float = 0.15

    def check(self, m):
        a, b = self.families  # a prerequisite for b
        if not m.has(a, b):
            return Outcome.UNTESTABLE, f"{a} or {b} absent"
        fwd, rev = m.transfer(a, b), m.transfer(b, a)
        ok = (fwd - rev) >= self.min_gap
        return (
            (Outcome.PASS if ok else Outcome.FAIL),
            f"asymmetry {fwd:.3f} - {rev:.3f} = {fwd - rev:+.3f}, "
            f"need >= {self.min_gap}",
        )


@dataclass
class PairIsIndependent(Expectation):
    max_transfer: float = 0.15

    def check(self, m):
        a, b = self.families
        if not m.has(a, b):
            return Outcome.UNTESTABLE, f"{a} or {b} absent"
        worst = max(m.transfer(a, b), m.transfer(b, a))
        return (
            (Outcome.PASS if worst <= self.max_transfer else Outcome.FAIL),
            f"max transfer {worst:.3f}, need <= {self.max_transfer}",
        )


@dataclass
class NothingTransfersFromNothing(Expectation):
    """The sanity check vein §2.1 says was missing from the prior art.

    A family with near-zero intrinsic content must not come out attributed with
    capabilities. This is the shape of the failure where a fitted model assigned
    high skill mastery to students who scored zero: recovering the plants is
    necessary and NOT sufficient, and a method can do that while producing
    nonsense on the unplanted rows.
    """

    max_outgoing: float = 0.15

    def check(self, m):
        if not m.has(*self.families):
            return Outcome.UNTESTABLE, "junk families absent"
        offenders = []
        for junk in self.families:
            for target in m.diag:
                if target == junk:
                    continue
                t = m.transfer(junk, target)
                if t > self.max_outgoing:
                    offenders.append(f"{junk}->{target}={t:.3f}")
        return (
            (Outcome.PASS if not offenders else Outcome.FAIL),
            "no junk row transfers" if not offenders else "; ".join(offenders[:5]),
        )


@dataclass
class ClustersWithRatherThan(Expectation):
    """`families = (subject, expected_partner, rival)`.

    The subject should transfer more with the expected partner than with the
    rival. This is how a structural prediction gets scored against a
    paradigm-membership prediction without either being reinterpreted later.
    """

    margin: float = 0.1

    def check(self, m):
        subject, partner, rival = self.families
        if not m.has(subject, partner, rival):
            return Outcome.UNTESTABLE, "one of the three families is absent"
        with_partner = max(m.transfer(subject, partner), m.transfer(partner, subject))
        with_rival = max(m.transfer(subject, rival), m.transfer(rival, subject))
        ok = (with_partner - with_rival) >= self.margin
        return (
            (Outcome.PASS if ok else Outcome.FAIL),
            f"{subject} with {partner}={with_partner:.3f} vs "
            f"with {rival}={with_rival:.3f}, diff {with_partner - with_rival:+.3f}",
        )


@dataclass
class CompositeNotSuppliedByParts(Expectation):
    """P4. The programme's central mechanism, as a scored assertion.

    `families = (composite, part_a, part_b)`.

    The claim is not that intelligence IS any of these families but that it is a
    COMBINATION of them. That has an empirical consequence: training on the
    parts should not already supply the whole. If it does, composition adds
    nothing beyond its components and Task Spec section 1.1's claim that
    composition lets a finite basis cover an infinite space is false in our
    setting -- covering would be enumeration of parts after all.

    Uses the BEST of the two parts as the baseline rather than a model trained
    on both, because the matrix is all-pairs and a jointly-trained model is not
    a cell in it. That makes the test CONSERVATIVE: a real joint model would
    supply at least as much, so a residual that survives this test might not
    survive the stronger one. Stated so the result is not overread.
    """

    min_residual: float = 0.25

    def check(self, m):
        composite, a, b = self.families
        if not m.has(composite, a, b):
            return Outcome.UNTESTABLE, "composite or a component absent"
        best_part = max(m.transfer(a, composite), m.transfer(b, composite))
        residual = 1.0 - best_part
        ok = residual >= self.min_residual
        return (
            (Outcome.PASS if ok else Outcome.FAIL),
            f"best part supplies {best_part:.3f} of the composite; "
            f"residual {residual:.3f}, need >= {self.min_residual} "
            "(conservative: single-part baseline, not jointly trained)",
        )


@dataclass
class DepthIncreasesResidual(Expectation):
    """P4a. `families = (deep_composite, shallow_composite, part)`.

    If depth is a real difficulty knob rather than one in name only, a deeper
    composite should be LESS supplied by its parts than a shallower one.
    """

    def check(self, m):
        deep, shallow, part = self.families
        if not m.has(deep, shallow, part):
            return Outcome.UNTESTABLE, "a composite or the shared part is absent"
        r_deep = 1.0 - m.transfer(part, deep)
        r_shallow = 1.0 - m.transfer(part, shallow)
        ok = r_deep > r_shallow
        return (
            (Outcome.PASS if ok else Outcome.FAIL),
            f"residual deep {r_deep:.3f} vs shallow {r_shallow:.3f}; "
            "depth is a knob in name only if this does not hold",
        )


# --------------------------------------------------------------------------
# The registry. Committed before any measurement.
# --------------------------------------------------------------------------

EXPECTATIONS: list[Expectation] = [
    JunkReadsNearZero(
        id="junk-near-zero",
        kind="instrument",
        statement="Both junk plants read near-zero intrinsic structural content.",
        families=("junk_random", "junk_trivial"),
        rationale=(
            "Task Spec §8 step 4. They are junk for opposite reasons -- one is "
            "unlearnable, one is learned instantly -- and both must yield a small "
            "area under the loss curve. If either reads high, the measurement is "
            "picking up something other than structure."
        ),
        blocking=True,
    ),
    PairClustersTightly(
        id="constructed-near-duplicate",
        kind="instrument",
        statement="The constructed near-duplicate pair clusters tightly.",
        families=("conjunction", "bruner_conjunction"),
        rationale=(
            "Same latent operation, deliberately different surfaces. If these do "
            "not cluster, the instrument cannot see identity through a surface "
            "change, and the whole method of translating families across fields "
            "into one form loses its warrant."
        ),
        blocking=True,
    ),
    PairClustersTightly(
        id="unplanted-near-duplicate",
        kind="instrument",
        statement="Parity and SHJ Type VI cluster tightly.",
        families=("parity_identification", "shj_type_vi"),
        rationale=(
            "Stronger evidence than the constructed pair, because nobody planted "
            "it: excavated independently from computational learning theory and "
            "1961 categorization psychology, and shown identical in code."
        ),
        blocking=False,
    ),
    PrerequisiteHasTheRightSign(
        id="shj-prerequisite-ordering",
        kind="instrument",
        statement="SHJ Type I transfers to Type VI more than the reverse.",
        families=("shj_type_i", "shj_type_vi"),
        rationale=(
            "The externally-attested, replicated human difficulty ordering. NOTE "
            "the caveat that must travel with it: this is empirical acquisition-"
            "speed precedence, not logical containment. A flat result is readable "
            "as 'human difficulty ordering does not transfer to an inducer', "
            "which is this vein's Hazard 1 confirmed rather than the instrument "
            "failing. That ambiguity is the price of an externally-attested plant "
            "and is still a better trade than one whose answer we invented."
        ),
        blocking=False,
    ),
    PairIsIndependent(
        id="independent-pair",
        kind="instrument",
        statement="SHJ Type I and probability matching are mutually independent.",
        families=("shj_type_i", "probability_matching"),
        rationale=(
            "Different Theta topology, different oracle behaviour, different "
            "target. If they transfer, the instrument is finding structure where "
            "none was planted -- a calibration fault that afterwards would be "
            "indistinguishable from a real result."
        ),
        blocking=True,
    ),
    NothingTransfersFromNothing(
        id="junk-attributes-nothing",
        kind="sanity",
        statement="Neither junk family transfers to anything.",
        families=("junk_random", "junk_trivial"),
        rationale=(
            "The check vein §2.1 says the prior art was missing. Recovering the "
            "plants is necessary and not sufficient: a method can recover planted "
            "structure and still attribute capabilities to a family that has "
            "none. Cheap, and nobody ran it."
        ),
        blocking=True,
    ),
    CompositeNotSuppliedByParts(
        id="P4-composite-not-supplied-by-parts",
        kind="prediction",
        statement=(
            "Training on parity and on hidden permutations separately does not "
            "supply parity-of-permuted-bits."
        ),
        families=("parity_of_permuted_bits", "parity_identification", "permuted_bits"),
        rationale=(
            "The programme claims intelligence is a COMBINATION of these "
            "capacities, not any of them in isolation. If the parts supply the "
            "whole, composition adds nothing and Task Spec section 1.1's claim "
            "that composition lets a finite basis cover an infinite space is "
            "false in our setting. Failure here is a result against the "
            "programme's central mechanism, which is why it is committed now "
            "with a threshold rather than argued afterwards."
        ),
        blocking=False,
    ),
    DepthIncreasesResidual(
        id="P4a-depth-is-a-real-knob",
        kind="prediction",
        statement=(
            "A depth-3 composite is less supplied by its parts than a depth-2 one."
        ),
        families=("depth3_parity_perm_perm", "parity_of_permuted_bits", "parity_identification"),
        rationale=(
            "Section 1.1 says depth is the difficulty knob that does not require "
            "inventing a new family per level. If residual does not rise with "
            "depth, depth is a knob in name only."
        ),
        blocking=False,
    ),
    ClustersWithRatherThan(
        id="P-structure-beats-paradigm",
        kind="prediction",
        statement=(
            "SHJ Type VI clusters with parity (its structural twin) rather than "
            "with SHJ Type I (its paradigm-mate)."
        ),
        families=("shj_type_vi", "parity_identification", "shj_type_i"),
        rationale=(
            "The sharpest prediction available. Type VI IS parity; Type I is "
            "single-dimension selection. If paradigm membership beats structure, "
            "translating families across fields buys less than it appears to -- "
            "which would be a real result against our own method, and is why it "
            "is committed before measurement."
        ),
        blocking=False,
    ),
]


def score(m: Matrix) -> tuple[bool, list[str]]:
    """Run every expectation. Returns (gate_passed, report lines).

    The gate is decided by the `blocking` expectations only. Non-blocking ones
    are predictions about the world rather than about the instrument: they are
    supposed to be able to fail, and a failure there is a finding, not a stop.
    """
    lines: list[str] = []
    gate = True
    for e in EXPECTATIONS:
        outcome, detail = e.check(m)
        flag = {"pass": "PASS", "fail": "FAIL", "untestable": "N/A "}[outcome.value]
        block = " [BLOCKING]" if e.blocking else ""
        lines.append(f"{flag}{block}  {e.id}: {e.statement}")
        lines.append(f"        {detail}")
        if e.blocking and outcome is not Outcome.PASS:
            gate = False
    lines.append("")
    lines.append(
        "GATE PASSED -- proceed to read the decomposition"
        if gate
        else "GATE FAILED -- Repertoire Spec §11 step 6 says stop"
    )
    return gate, lines
