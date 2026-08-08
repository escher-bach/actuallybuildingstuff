"""The dial sweep. Task Spec section 8 step 5 -- "the first result that matters".

    "This is the only step that can end the programme rather than redirect it."

Vary residual entropy **continuously** and measure structural content and
transfer as a curve in it.  The reading was fixed by the Task Spec before any
number existed, and `READINGS` below is that table as executable predicates:

    monotone collapse   supervision needs determinate targets    STOP
    interior peak       the programme works, the four named
                        levels are cut in the wrong places       relocate the cuts
    flat or rising      the ambiguity model is wrong             investigate

--------------------------------------------------------------------------
Four decisions, because a sweep is easy to run and easy to misread
--------------------------------------------------------------------------

**The x-axis is measured, not nominal.**  Plotting against the dial setting
would report a curve in units nobody can compare across families, because the
map from dial to residual entropy is nonlinear and family-specific.  `entropy.py`
computes H(y | context, e) exactly by enumeration, and that is the axis.

**Two dials, and running both is the control.**  Section 8 step 5 names both
"how many observations precede the query" and "what fraction of theta is stated
in the preamble".  Running one tests whether *that parametrization* collapses.
If the two curves disagree, the result is about the parametrization rather than
about residual entropy -- which is the same argument the Task Spec makes for
preferring a sweep to a four-way comparison, applied one level up.

**Structural content alone cannot answer the question, and transfer can.**  S is
the area above the run's own floor, and the floor *rises* with residual entropy
because part of the loss becomes irreducible.  So S falls with entropy for a
reason that has nothing to do with whether supervision works, and a naive sweep
would manufacture a monotone collapse out of arithmetic.  This is why step 5
says "structural content **and transfer**": transfer holds the evaluation target
fixed, so the floor does not move under it.  **Transfer is the arm that decides,
and S is reported beside it because section 4 defines it.**

**The A4 confound is real and is met with a second family.**  The worked family
fails A4 at these moduli -- section 6 says so, and the register row records it as
REPAIRABLE rather than PASS. A memorizable family could flatten the curve for a
reason unrelated to residual entropy. Parity has the opposite property (2^d truth
table, d probes), so running it as a second sweep separates "supervision above L0
does not work" from "this family was memorized".
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from .entropy import measure_residual_entropy
from .episode import EpisodeSpec
from .metrics import Budget, BudgetMismatch, acquisition_slope
from .model import ModelConfig, device_report
from .train import RunRecord, evaluate, train_run


# --------------------------------------------------------------------------
# The dials
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Dial:
    """One parametrization of residual entropy, and how to build a spec from it.

    `T` -- the number of *scored* trials -- lives on the dial and is the same for
    every setting of it, and the two dials are constructed with the same T so
    their curves can be compared. That is not tidiness: `compare_dials` only says
    anything if the two sweeps score the same number of tokens per episode
    against the same fixed transfer target.
    """

    name: str
    settings: tuple[float, ...]
    T: int
    build: Callable[[float, int], EpisodeSpec]
    requires: str = ""

    def spec(self, setting: float) -> EpisodeSpec:
        return self.build(setting, self.T)

    @property
    def transfer_target(self) -> EpisodeSpec:
        """The fixed evaluation episode: nothing revealed, nothing given free."""
        return EpisodeSpec(T=self.T)


def preamble_dial(points: int = 9, T: int = 8) -> Dial:
    """Fraction of theta stated in a fixed-length preamble. Needs partial_preamble.

    The preamble is the same length at every setting, with unrevealed slots
    rendered as an explicit unknown marker, so the dial moves residual entropy
    without also moving how many tokens the model gets to look at.
    """
    return Dial(
        name="preamble_fraction",
        settings=tuple(i / (points - 1) for i in range(points)),
        T=T,
        build=lambda s, t: EpisodeSpec(T=t, reveal=s),
        requires="partial_preamble",
    )


def observation_dial(max_free: int = 8, T: int = 4) -> Dial:
    """Free observations before the scored trials. Asks nothing of the family.

    The generic dial, and the one that holds the supervised-token count exactly
    constant across every setting -- T scored trials whatever `n_free` is. That
    makes the comparison across settings cleaner than the preamble dial can
    promise for a family whose preamble length depends on theta.
    """
    return Dial(
        name="free_observations",
        settings=tuple(float(i) for i in range(max_free + 1)),
        T=T,
        build=lambda s, t: EpisodeSpec(T=t, n_free=int(s)),
    )


# --------------------------------------------------------------------------
# Points and results
# --------------------------------------------------------------------------


@dataclass
class SweepPoint:
    dial: str
    setting: float
    seed: int
    rule_entropy: float  # the x-axis: exact H(y | context, e), nats
    notation_upper: float
    structural_content: float
    l_final: float
    excess_over_floor: float
    converged: bool
    acq_slope: float
    # H(theta | context). Logged beside rule_entropy because for a *balanced*
    # family the two come apart badly: on parity, rule_entropy sits at log 2
    # across the first five free observations while this falls 5.54 -> 2.79. A
    # curve plotted against rule_entropy alone would report that nothing changed
    # over the half of the dial where almost everything changed.
    theta_entropy: float = 0.0
    transfer: float | None = None  # S(fixed target | this model); lower is more transfer
    transfer_baseline: float | None = None  # the same, from random init
    # Held-out answer-only loss on the fixed target after the probe, in nats.
    # Not a replacement for `transfer` -- that is section 4's conditional
    # structural content and is the defined quantity -- but an interpretable
    # cross-check on it. An area under a curve can move for reasons that have
    # nothing to do with competence (a different starting loss moves it); a
    # held-out per-token loss cannot. When the two disagree, read this one first.
    transfer_eval: float | None = None
    seconds: float = 0.0

    @property
    def transfer_fraction(self) -> float | None:
        """How much of the fixed target's content this model already supplied.

        Same normalization as `expectations.Matrix.transfer`: 1.0 means training
        at this dial setting left nothing to learn on the target, 0.0 means it
        helped not at all. Normalizing by the from-scratch baseline is what makes
        settings comparable when their own floors differ.
        """
        if self.transfer is None or not self.transfer_baseline:
            return None
        return max(0.0, min(1.0, 1.0 - self.transfer / self.transfer_baseline))


@dataclass
class SweepResult:
    family: str
    k: int
    dial: str
    budget: Budget
    points: list[SweepPoint] = field(default_factory=list)
    device: dict = field(default_factory=dict)
    note: str = ""

    def by_setting(self) -> dict[float, list[SweepPoint]]:
        out: dict[float, list[SweepPoint]] = {}
        for p in self.points:
            out.setdefault(p.setting, []).append(p)
        return dict(sorted(out.items()))

    def curve(self, metric: str = "transfer_fraction") -> list[tuple[float, float]]:
        """(residual entropy, mean metric) ordered by entropy -- the curve itself."""
        pts = []
        for setting, group in self.by_setting().items():
            vals = [getattr(p, metric) for p in group]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            h = sum(p.rule_entropy for p in group) / len(group)
            pts.append((h, sum(vals) / len(vals)))
        return sorted(pts)

    def noise(self, metric: str = "transfer_fraction") -> float:
        """Seed-to-seed standard error of the metric, pooled across settings.

        This is what makes "the peak clears both ends" a claim rather than a
        description: without it the only available margin is absolute, and an
        absolute margin cannot know whether this family at this budget is noisy.
        Returns 0.0 with one seed per setting, which is honest -- one seed
        supplies no estimate of its own variability, and the reading then falls
        back to the absolute margin alone.
        """
        from statistics import fmean, pstdev

        sems = []
        for _, group in self.by_setting().items():
            vals = [getattr(p, metric) for p in group]
            vals = [v for v in vals if v is not None]
            if len(vals) >= 2:
                sems.append(pstdev(vals) / (len(vals) ** 0.5))
        return fmean(sems) if sems else 0.0

    def fit_to_read(self, min_learned: float = 0.35, max_unconverged: float = 0.5
                    ) -> tuple[bool, str]:
        """Is this sweep in a state where its curve means anything?

        **The most likely way this tool gets misused is a sweep run too small,
        whose noise is then read as a monotone collapse and used to stop the
        programme.**  Task Spec section 8 step 5 is the only step that can end
        the programme; a verdict that can be produced by an underpowered run is
        worse than no verdict, so the reading is gated on the runs being fit to
        read rather than on the caller having remembered to check.

        Two conditions, both about the runs and neither about their shape:

          * the models must have *learned something* -- final loss must have
            closed a reasonable fraction of the distance from its own starting
            loss to the Bayes floor. A model still at chance produces a transfer
            curve that is entirely sampling noise, and noise has a shape.
          * most runs must have settled. An unconverged run's S is a lower
            bound, and lower bounds of differing tightness across the dial are
            not a curve.

        **Measured against the UPPER end of the floor band**, which is the
        achievable one.  The lower end conditions on the encoding, and the model
        never observes the encoding -- so scoring "did it learn" against the
        lower end asks the model to close a distance no predictor can close, and
        would refuse a converged run for being 1.2 nats from a target that is
        physically out of reach.  The lower end keeps its own job: it is the leak
        check, where nothing may legitimately go below.

        (The first real T4 sweep was refused under the old version, and refused
        *correctly* -- the model was 0.46-1.78 nats above even the achievable
        floor. But it was refused for a reason the message stated wrongly, which
        is its own kind of failure.)
        """
        if not self.points:
            return False, "no points"

        learned = []
        for setting, group in self.by_setting().items():
            for p in group:
                start = p.structural_content / max(1, self.budget.steps) + p.l_final
                achievable = p.rule_entropy + p.notation_upper
                room = max(1e-9, start - achievable)
                learned.append(1.0 - (p.l_final - achievable) / room)
        worst = min(learned)
        n_unconv = sum(1 for p in self.points if not p.converged)
        frac_unconv = n_unconv / len(self.points)

        problems = []
        if worst < min_learned:
            problems.append(
                f"the weakest run closed only {worst:.0%} of the distance to the "
                f"ACHIEVABLE floor -- rule + notation, the one a model that cannot "
                f"see the encoding could actually reach -- and needs {min_learned:.0%}. "
                "At this budget it has learned the answer marginal and not the task, "
                "so the transfer curve measures transfer of the marginal"
            )
        if frac_unconv > max_unconverged:
            problems.append(
                f"{n_unconv}/{len(self.points)} runs still had a falling tail "
                f"(allowed {max_unconverged:.0%}); their S values are lower bounds "
                "of differing tightness, which is not a curve"
            )
        if problems:
            return False, "; ".join(problems) + ". Raise steps or lower k."
        return True, (
            f"weakest run closed {worst:.0%} to its floor, "
            f"{len(self.points) - n_unconv}/{len(self.points)} settled"
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["budget"] = asdict(self.budget)
        payload["budget_fingerprint"] = self.budget.fingerprint
        _atomic_write(path, json.dumps(payload, indent=1))

    def report(self) -> str:
        lines = [
            f"{self.family} k={self.k} dial={self.dial}",
            f"  budget: {self.budget.describe()}",
            f"  device: {self.device.get('name', '?')}",
            "",
            f"  {'H(y|ctx,e)':>11} {'H(th|ctx)':>10} {'setting':>8} {'S':>10} "
            f"{'L_final':>8} {'floor+':>8} {'over':>7} {'slope':>8} {'transfer':>9}",
        ]
        for setting, group in self.by_setting().items():
            n = len(group)
            avg = lambda f: sum(f(p) for p in group) / n
            tf = [p.transfer_fraction for p in group if p.transfer_fraction is not None]
            lines.append(
                f"  {avg(lambda p: p.rule_entropy):>11.4f} "
                f"{avg(lambda p: p.theta_entropy):>10.4f} {setting:>8.3f} "
                f"{avg(lambda p: p.structural_content):>10.3f} "
                f"{avg(lambda p: p.l_final):>8.4f} "
                f"{avg(lambda p: p.rule_entropy + p.notation_upper):>8.4f} "
                f"{avg(lambda p: p.l_final - p.rule_entropy - p.notation_upper):>7.3f} "
                f"{avg(lambda p: p.acq_slope):>8.4f} "
                + (f"{sum(tf) / len(tf):>9.3f}" if tf else f"{'-':>9}")
            )
        unconverged = [p for p in self.points if not p.converged]
        if unconverged:
            lines.append(
                f"\n  {len(unconverged)}/{len(self.points)} runs had a still-falling "
                "tail; their S is a lower bound"
            )
        if self.note:
            lines.append(f"\n  {self.note}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The reading -- fixed before any number exists
# --------------------------------------------------------------------------


@dataclass
class Reading:
    verdict: str
    action: str
    detail: str
    confident: bool


# Fixed here, once, for the same reason `expectations.py` fixes its thresholds:
# the failure mode is not forgetting the prediction, it is reading the curve and
# finding the reading that fits.
#
# `INTERIOR_MARGIN_ABS` is on the metric's own scale, which for transfer_fraction
# is [0, 1]. It is deliberately NOT a fraction of the curve's range: the range is
# set by the peak, so a margin defined against it is circular and any bump
# whatsoever clears its own margin. (Found by the test that was written to make
# this check fail, which it did.)
INTERIOR_MARGIN_ABS = 0.10
INTERIOR_SIGMA = 2.0  # ...or this many standard errors, whichever is larger
FLATNESS = 0.10


def read_curve(curve: list[tuple[float, float]], noise: float = 0.0) -> Reading:
    """Task Spec section 8 step 5's table, as a predicate over the measured curve.

    `curve` is (residual entropy, metric) ascending in entropy, and the metric
    should be the one that survives the floor moving -- transfer_fraction by
    default. Passing structural content instead will bias toward "monotone
    collapse", which is exactly the artifact this module's docstring warns about,
    so the caller has to say which it means.

    `noise` is the seed-to-seed standard error of the metric, which is data
    rather than a threshold: with it, a peak has to clear the ends by
    `INTERIOR_SIGMA` standard errors, and without it only the absolute margin
    applies. `SweepResult.noise()` computes it from the seeds already run.
    """
    if len(curve) < 4:
        return Reading("insufficient", "run more points",
                       f"{len(curve)} points; the shape is not determined", False)

    xs = [x for x, _ in curve]
    ys = [y for _, y in curve]
    span = max(ys) - min(ys)
    lo_end, hi_end = ys[0], ys[-1]

    if span <= FLATNESS * max(1e-9, abs(max(ys))):
        return Reading(
            "flat", "investigate before proceeding",
            f"range {span:.4f} over the whole sweep is within {FLATNESS:.0%} of the "
            "maximum; the ambiguity model is wrong, or the budget is too small for "
            "any of these settings to separate",
            True,
        )

    required = max(INTERIOR_MARGIN_ABS, INTERIOR_SIGMA * noise)
    peak_i = max(range(len(ys)), key=lambda i: ys[i])
    interior = 0 < peak_i < len(ys) - 1
    clears = (ys[peak_i] - lo_end >= required and ys[peak_i] - hi_end >= required)

    if interior and clears:
        return Reading(
            "interior peak", "continue; relocate the cuts from the curve",
            f"peak {ys[peak_i]:.4f} at H={xs[peak_i]:.4f} nats, clearing both ends "
            f"({lo_end:.4f}, {hi_end:.4f}) by at least {required:.4f}. The programme "
            "works and the four named levels are cut in the wrong places -- the cut "
            "belongs at this entropy, not at whichever level happens to sit near it",
            True,
        )

    descending = all(ys[i] >= ys[i + 1] - 1e-9 for i in range(len(ys) - 1))
    if descending:
        return Reading(
            "monotone collapse", "STOP -- Task Spec sections 2-9 are void",
            f"falls monotonically from {lo_end:.4f} at H={xs[0]:.4f} to {hi_end:.4f} "
            f"at H={xs[-1]:.4f}. Supervision requires determinate targets and "
            "everything above L0 needs reward. This is a real result and is worth "
            "writing up",
            True,
        )

    ascending = all(ys[i] <= ys[i + 1] + 1e-9 for i in range(len(ys) - 1))
    if ascending:
        return Reading(
            "rising", "investigate before proceeding",
            f"rises monotonically to {hi_end:.4f} at the highest entropy measured; "
            "the ambiguity model is wrong in the direction nobody predicted",
            True,
        )

    return Reading(
        "non-monotone, no clear peak", "run more seeds before reading it",
        f"peak at index {peak_i} of {len(ys)} does not clear both ends by "
        f"{required:.4f} ({ys[peak_i] - lo_end:+.4f} low, {ys[peak_i] - hi_end:+.4f} "
        "high); this is the shape noise produces",
        False,
    )


# --------------------------------------------------------------------------
# Running
# --------------------------------------------------------------------------


def run_sweep(
    family: Any,
    k: int,
    dial: Dial,
    budget: Budget,
    model_cfg: ModelConfig,
    seeds: tuple[int, ...] = (0, 1, 2),
    transfer_target: EpisodeSpec | None = None,
    transfer_steps: int | None = None,
    entropy_episodes: int = 48,
    out_dir: Path | None = None,
    device: str = "auto",
    verbose: bool = True,
    shard: tuple[int, int] = (0, 1),
) -> SweepResult:
    """Run the sweep. Resumable, because a Kaggle session does not always finish.

    `transfer_target` is the fixed evaluation episode spec -- held identical at
    every dial setting, which is what makes transfer immune to the floor moving.
    Defaults to the zero-reveal, zero-free-observation episode: "does training at
    this dial setting install the ability to infer theta at all?"

    `shard=(i, n)` runs only every n-th point, so n processes can split the sweep
    across n GPUs.  **The points are embarrassingly parallel and the parallelism
    is real**: every point trains a fresh model from scratch on its own episode
    stream, so there is no gradient to exchange, no parameter server, and no
    synchronisation of any kind -- the only shared state is a directory of
    finished results.  Two T4s halve the wall clock exactly.

    Each point's record is keyed by (dial, setting, seed), so shards never write
    the same file, and `collect` rebuilds the whole sweep from the directory
    afterwards.  Writes are atomic (temp + rename), because the one genuinely
    shared file is the transfer baseline and a half-written JSON read by the
    other shard would be a confusing failure a long way from its cause.
    """
    T = dial.T
    if transfer_target is None:
        transfer_target = dial.transfer_target
    if transfer_steps is None:
        transfer_steps = max(50, budget.steps // 5)

    probe_budget = Budget(
        steps=transfer_steps, batch_size=budget.batch_size, max_len=budget.max_len,
        lr=budget.lr, warmup=min(budget.warmup, transfer_steps // 5),
        model=budget.model,
    )

    result = SweepResult(family=family.name, k=k, dial=dial.name, budget=budget,
                         device=device_report())
    out_dir = Path(out_dir) if out_dir else None

    # The from-scratch baseline for transfer, once. Every dial setting is scored
    # against the same number, so a drifting baseline cannot manufacture a curve.
    baseline_path = out_dir / "transfer_baseline.json" if out_dir else None
    baseline = _cached_run(
        baseline_path,
        lambda: train_run(family, k, transfer_target, probe_budget,
                          model_cfg=model_cfg, seed=9999, device=device,
                          label="transfer-baseline")[0],
    )
    baseline_S = baseline.content().value
    if verbose:
        print(f"transfer baseline (from scratch): S = {baseline_S:.4f}", flush=True)

    shard_i, shard_n = shard
    plan = [(s_, sd) for s_ in dial.settings for sd in seeds]
    mine = [pt for j, pt in enumerate(plan) if j % shard_n == shard_i]
    if verbose and shard_n > 1:
        print(f"shard {shard_i + 1}/{shard_n}: {len(mine)} of {len(plan)} points",
              flush=True)

    entropy_cache: dict[float, Any] = {}
    for setting, seed in mine:
        spec = dial.spec(setting)
        if setting not in entropy_cache:
            entropy_cache[setting] = measure_residual_entropy(
                family, k, spec, n_episodes=entropy_episodes)
        ent = entropy_cache[setting]
        if True:
            tag = f"{dial.name}_{setting:.4f}_seed{seed}"
            t0 = time.time()

            # Cached as a pair. The transfer probe needs the trained *model*,
            # which is not in the saved record, so resuming has to skip both arms
            # together or not at all -- caching them separately would re-run the
            # training it just skipped reporting, which is the worst of both.
            main_path = out_dir / f"{tag}.json" if out_dir else None
            probe_path = out_dir / f"{tag}_transfer.json" if out_dir else None
            cached = _load_pair(main_path, probe_path, budget)
            if cached is not None:
                rec, probe = cached
                probe_eval = probe.held_out_answer_loss
                if verbose:
                    print(f"  {tag}: resumed from cache", flush=True)
            else:
                rec, model = train_run(
                    family, k, spec, budget, model_cfg=model_cfg, seed=seed,
                    device=device, bayes_floor=ent.floor_lower,
                    bayes_floor_upper=ent.floor_upper, label=tag,
                )
                probe, probe_model = train_run(
                    family, k, transfer_target, probe_budget, model_cfg=model_cfg,
                    seed=seed, device=device, init_state=model.state_dict(),
                    label=f"{tag}-transfer",
                )
                probe_eval = evaluate(probe_model, family, k, transfer_target,
                                      n_episodes=64, device=device).answer_loss
                probe.held_out_answer_loss = probe_eval
                if main_path:
                    rec.save(main_path)
                    probe.save(probe_path)

            content = rec.content()
            result.points.append(SweepPoint(
                dial=dial.name, setting=setting, seed=seed,
                rule_entropy=ent.rule, notation_upper=ent.notation_upper,
                theta_entropy=ent.theta_entropy,
                structural_content=content.value, l_final=content.l_final,
                excess_over_floor=content.l_final - ent.floor_lower,
                converged=content.converged,
                acq_slope=rec.slope().slope,
                transfer=probe.content().value, transfer_baseline=baseline_S,
                transfer_eval=probe_eval,
                seconds=time.time() - t0,
            ))
            if verbose:
                p = result.points[-1]
                print(
                    f"  {dial.name}={setting:.3f} seed={seed}  H={p.rule_entropy:.4f}"
                    f"  S={p.structural_content:.3f}  transfer="
                    f"{p.transfer_fraction:.3f}  ({p.seconds:.0f}s)",
                    flush=True,
                )
            if out_dir:
                name = "sweep.json" if shard_n == 1 else f"sweep.shard{shard_i}.json"
                result.save(out_dir / name)

    return result


def _atomic_write(path: Path, text: str) -> None:
    """Write via temp + rename, so a reader never sees a half-written file.

    Matters once shards run concurrently: the transfer baseline is the one file
    two processes may touch, and a partial JSON read by the other shard would
    surface as a decode error a long way from its cause.
    """
    import os

    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def collect(out_dir: Path, dial_name: str, budget: Budget, family: str, k: int
            ) -> SweepResult:
    """Rebuild a whole sweep from the per-point records a set of shards wrote.

    The per-point files are the source of truth -- keyed by (dial, setting, seed),
    so shards never collide -- and this is what turns n independently-running
    processes back into one curve.
    """
    result = SweepResult(family=family, k=k, dial=dial_name, budget=budget)
    for shard_file in sorted(out_dir.glob("sweep.shard*.json")):
        data = json.loads(shard_file.read_text())
        if data.get("budget_fingerprint") != budget.fingerprint:
            continue
        for pt in data["points"]:
            result.points.append(SweepPoint(**pt))
    seen: set[tuple[float, int]] = set()
    unique = []
    for p in sorted(result.points, key=lambda p: (p.setting, p.seed)):
        if (p.setting, p.seed) in seen:
            continue
        seen.add((p.setting, p.seed))
        unique.append(p)
    result.points = unique
    return result


def _cached_run(path: Path | None, make: Callable[[], RunRecord]) -> RunRecord:
    """Resume support for a single run: a completed one is read back."""
    loaded = _load(path)
    if loaded is not None:
        return loaded
    rec = make()
    if path:
        rec.save(path)
    return rec


def _load(path: Path | None) -> RunRecord | None:
    if not path or not path.exists():
        return None
    data = json.loads(path.read_text())
    return RunRecord(
        family=data["family"], k=data["k"], spec=data["spec"],
        budget=Budget(**data["budget"]), losses=data["losses"],
        supervised_tokens=data["supervised_tokens"],
        per_trial_history=data["per_trial_history"],
        malformed_rate=data.get("malformed_rate", []),
        device=data.get("device", {}), seconds=data.get("seconds", 0.0),
        bayes_floor=data.get("bayes_floor"),
        bayes_floor_upper=data.get("bayes_floor_upper"),
        label=data.get("label", ""),
    )


def _load_pair(
    main_path: Path | None, probe_path: Path | None, budget: Budget
) -> tuple[RunRecord, RunRecord] | None:
    """Both arms of a dial point, or nothing.

    A cached record whose budget fingerprint does not match what is being asked
    for now is discarded rather than used. Section 4 says comparing across
    budgets is meaningless, and a stale cache in an output directory reused
    between two parameter choices is the quietest available way to do it.
    """
    main, probe = _load(main_path), _load(probe_path)
    if main is None or probe is None:
        return None
    if main.budget.fingerprint != budget.fingerprint:
        return None
    return main, probe


def compare_dials(a: SweepResult, b: SweepResult, metric: str = "transfer_fraction") -> str:
    """Do the two parametrizations agree? If not, the result is about the dial.

    The same argument the Task Spec makes for a sweep over a four-way comparison,
    applied to the sweep itself: one dial tests whether *that dial* collapses.
    """
    if a.budget.fingerprint != b.budget.fingerprint:
        raise BudgetMismatch(
            "the two dials were run at different budgets, so their curves are not "
            "comparable (Task Spec section 4)"
        )
    ra = read_curve(a.curve(metric), noise=a.noise(metric))
    rb = read_curve(b.curve(metric), noise=b.noise(metric))
    agree = ra.verdict == rb.verdict
    return "\n".join([
        f"{a.dial}: {ra.verdict} -- {ra.detail}",
        f"{b.dial}: {rb.verdict} -- {rb.detail}",
        "",
        ("The two parametrizations agree, so the reading is about residual entropy."
         if agree else
         "The two parametrizations DISAGREE. The result is about the "
         "parametrization, not about residual entropy -- do not read either curve "
         "as the section 8 step 5 answer until the disagreement is explained."),
    ])
