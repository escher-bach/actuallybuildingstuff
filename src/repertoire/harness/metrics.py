"""The measurement. Task Spec section 4, and nothing else computes these.

Two quantities, computed one way every time, because section 4 says two later
sections depend on them being computed the same way:

    structural content        area under the prequential loss curve above its
                              floor -- what a family teaches, in nats
    acquisition slope         within-episode per-trial loss decay -- the rate at
                              which theta gets identified, section 9's *primary*
                              metric

Torch-free on purpose.  These are arithmetic over a list of numbers, and keeping
them out of the training module means they can be tested against closed forms
and re-run over a saved log without a GPU.

--------------------------------------------------------------------------
Three decisions section 4 leaves open, made here and stated
--------------------------------------------------------------------------

**Loss is per supervised token, never per episode.**  Handoff section 2.2: the
`slotted` encoding renders ~85% longer than its siblings (209 vs 113 tokens at
k=1).  A per-episode average over episodes that differ that much in length mixes
an encoding effect into the family effect, and every structural-content number
inherits it.  So every L_i here is nats per supervised token.

**L_final is the mean of the last `tail_frac` of the run, not the converged
loss.**  Section 4 says "the converged loss"; at a fixed budget convergence is
not observed, and pretending otherwise would put an unfalsifiable quantity at
the centre of the measurement.  The tail mean is the estimate, `converged` is
reported alongside it, and when it is False the structural content is a *lower
bound* -- the floor was still falling, so the area above it was larger than
measured.  That direction is stated because it is the one that matters: an
unconverged run understates a family rather than flattering it.

**The area is a sum over steps, so a comparison across different step counts is
meaningless.**  Section 4 already says the quantity is budget-relative.  This
module refuses to compare across budgets rather than trusting the caller to
remember, because "hold the budget fixed" is precisely the kind of instruction
that survives a first analysis and not a third.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from statistics import fmean


@dataclass(frozen=True)
class Budget:
    """What section 4 requires be held fixed and reported.

    Reported in *steps and tokens*, not seconds. A T4 and a P100 at the same step
    count are the same budget for this purpose; the wall clock is not the
    quantity structural content is relative to.
    """

    steps: int
    batch_size: int
    max_len: int
    lr: float
    warmup: int
    model: dict = field(default_factory=dict)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.blake2b(payload, digest_size=6).hexdigest()

    def describe(self) -> str:
        return (
            f"{self.steps} steps x {self.batch_size} episodes x <= {self.max_len} tok"
            f"  lr={self.lr:g}  [{self.fingerprint}]"
        )


class BudgetMismatch(ValueError):
    """Raised when runs at different budgets are put in the same comparison."""


@dataclass
class StructuralContent:
    value: float  # nats per supervised token, summed over steps
    l_final: float
    l_first: float
    converged: bool
    tail_slope: float  # nats/token per step over the tail; ~0 means settled
    n_steps: int
    total_supervised_tokens: int
    bayes_floor: float | None = None  # the LOWER bound: H(y | context, encoding)
    bayes_floor_upper: float | None = None  # + the notation term
    excess_over_bayes: float | None = None
    note: str = ""

    def report(self) -> str:
        lines = [
            f"S = {self.value:.4f} nats/token-summed-over-{self.n_steps}-steps",
            f"    L_first {self.l_first:.4f} -> L_final {self.l_final:.4f}"
            f"  ({'settled' if self.converged else 'STILL FALLING -- S is a lower bound'})",
        ]
        if self.bayes_floor is not None:
            band = (
                f"[{self.bayes_floor:.4f}, {self.bayes_floor_upper:.4f}]"
                if self.bayes_floor_upper is not None
                else f"{self.bayes_floor:.4f}"
            )
            lines.append(
                f"    Bayes floor {band}; final excess over its lower end "
                f"{self.excess_over_bayes:+.4f}"
            )
            if self.bayes_floor_upper is not None and self.l_final <= self.bayes_floor_upper:
                lines.append(
                    "    final loss is inside the floor band -- the model is at "
                    "optimal to within what the encoding costs"
                )
        if self.note:
            lines.append(f"    {self.note}")
        return "\n".join(lines)


def structural_content(
    losses: list[float],
    supervised_tokens: list[int] | None = None,
    tail_frac: float = 0.1,
    bayes_floor: float | None = None,
    bayes_floor_upper: float | None = None,
    settled_tol: float | None = None,
) -> StructuralContent:
    """Section 4, steps 3 and 4: sum_i (L_i - L_final).

    `losses[i]` must be the loss on batch i measured **before** training on it.
    Section 4 is explicit about the order -- "Before, not after -- the coding
    argument requires evaluating on data not yet seen" -- and the whole
    interpretation as a codelength dies if the caller gets it backwards.  The
    trainer is the only caller and it records in that order; this function cannot
    check it, which is worth knowing rather than assuming away.
    """
    if len(losses) < 4:
        raise ValueError("too few steps to estimate a floor")

    n_tail = max(2, int(round(tail_frac * len(losses))))
    tail = losses[-n_tail:]
    l_final = fmean(tail)
    value = sum(x - l_final for x in losses)

    slope = _ols_slope(list(range(len(tail))), tail)
    # "Settled" means the tail's own trend is small relative to how far the run
    # travelled. An absolute threshold would call a flat junk family unconverged
    # and a fast-learning family converged, both wrongly.
    travelled = max(1e-9, losses[0] - l_final)
    tol = settled_tol if settled_tol is not None else 0.02 * travelled / max(1, n_tail)
    converged = abs(slope) <= tol

    excess = None if bayes_floor is None else l_final - bayes_floor
    note = ""
    if bayes_floor is not None and excess is not None and excess < -1e-3:
        # Checked against the LOWER end of the floor band on purpose. The lower
        # end conditions on the encoding, which the model does not observe, so
        # nothing can legitimately beat it -- a run that does is reading something
        # the posterior calculation does not know about.
        note = (
            f"final loss is {-excess:.4f} nats/token BELOW the lower end of the "
            "Bayes floor. Nothing can beat a floor computed with knowledge the "
            "model does not have, so this is a leak or a wrong floor. Do not read "
            "S until it is resolved."
        )

    return StructuralContent(
        value=value,
        l_final=l_final,
        l_first=losses[0],
        converged=converged,
        tail_slope=slope,
        n_steps=len(losses),
        total_supervised_tokens=sum(supervised_tokens or []),
        bayes_floor=bayes_floor,
        bayes_floor_upper=bayes_floor_upper,
        excess_over_bayes=excess,
        note=note,
    )


def conditional_structural_content(
    losses_from_pretrained: list[float], **kw
) -> StructuralContent:
    """S(T_j | m_i): section 4's conditional form, and the matrix's cells.

    Identical arithmetic on a run that started from a model already trained on
    family i rather than from random init.  Separate name because the two are
    never interchangeable in a report, and a matrix cell labelled with the
    unconditional name has happened elsewhere.
    """
    return structural_content(losses_from_pretrained, **kw)


# --------------------------------------------------------------------------
# Section 9's primary metric
# --------------------------------------------------------------------------


@dataclass
class AcquisitionSlope:
    slope: float  # nats/token per trial; negative means theta is being identified
    per_trial: list[float]
    n_trials: int
    r2: float

    def report(self) -> str:
        curve = " ".join(f"{x:.3f}" for x in self.per_trial)
        return f"acquisition slope {self.slope:+.5f}/trial (R^2 {self.r2:.3f})\n    {curve}"


def acquisition_slope(per_trial_loss: list[float]) -> AcquisitionSlope:
    """Section 4's within-episode decay, fitted by OLS over trial index.

    Section 9: "If it does not rise across a curriculum, nothing else in this
    section matters -- that is the first thing to plot."  Which makes it the
    first thing the harness logs, and it is logged per run rather than derived
    later, because it is only available while the per-trial breakdown exists.

    The fit is linear on purpose.  The true shape is not linear -- identification
    is closer to a step once the critical observation arrives -- but a slope is
    comparable across families and a curve shape is not, and the curve is
    returned alongside so the linearity assumption can be inspected rather than
    trusted.  `junk_trivial` is the sharp test: theta is identified in exactly one
    trial, so its curve is a step and its R^2 will be poor while its slope is
    still correctly negative.
    """
    xs = list(range(len(per_trial_loss)))
    slope = _ols_slope(xs, per_trial_loss)
    mean = fmean(per_trial_loss) if per_trial_loss else 0.0
    ss_tot = sum((y - mean) ** 2 for y in per_trial_loss)
    inter = mean - slope * fmean(xs) if xs else 0.0
    ss_res = sum((y - (slope * x + inter)) ** 2 for x, y in zip(xs, per_trial_loss))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0
    return AcquisitionSlope(slope=slope, per_trial=list(per_trial_loss),
                            n_trials=len(per_trial_loss), r2=r2)


def _ols_slope(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = fmean(xs), fmean(ys)
    denom = sum((x - mx) ** 2 for x in xs)
    if denom <= 1e-12:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom


# --------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------


def require_same_budget(runs: list["object"]) -> str:
    """Refuse a comparison across budgets. Section 4 says it is meaningless.

    Raises rather than warns.  The failure this prevents -- a sweep re-run at a
    larger step count and half-merged with the old points -- produces a curve
    that looks like a finding, and a warning in a log is not a defence against
    that.
    """
    prints = {getattr(r, "budget").fingerprint for r in runs}
    if len(prints) != 1:
        raise BudgetMismatch(
            "structural content is budget-relative (Task Spec section 4); these "
            f"runs span {len(prints)} budgets: {sorted(prints)}"
        )
    return prints.pop()


def turnover(values: list[float]) -> tuple[bool, str]:
    """Does structural content change sign of trend as compute grows?

    Hazard 17 and the standing hazards table: A4 cannot be certified at our
    scale, so brute-force collapse must be *measured*, and it appears as a
    turnover -- content rising with compute, then falling as memorizing the
    generator becomes affordable.  "Which looks like noise unless you are
    watching for a sign change."  This is the watching.

    Needs at least three budgets to say anything, and says so rather than
    returning False, because a confident False here would be read as "no
    collapse" when it means "not enough points to see one".
    """
    if len(values) < 3:
        return False, f"{len(values)} budget(s): a turnover needs at least 3"
    rising = [values[i + 1] > values[i] for i in range(len(values) - 1)]
    if all(rising) or not any(rising):
        return False, "monotone across budgets; no turnover"
    idx = next(i for i in range(len(rising) - 1) if rising[i] != rising[i + 1])
    return True, (
        f"trend reverses between budget {idx + 1} and {idx + 2} "
        f"({values[idx]:.4f} -> {values[idx + 1]:.4f} -> {values[idx + 2]:.4f}); "
        "hazard 17 says read this as possible brute-force collapse, not noise"
    )


def nats_to_bits(x: float) -> float:
    return x / math.log(2)
