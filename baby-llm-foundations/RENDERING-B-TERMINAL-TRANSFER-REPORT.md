# Rendering Transfer After Process Training

## Consolidated stage report: corrected seeds 0 and 1, and instrument verification

### Status

This report consolidates the results of the Step 1 representation-transfer
stage. It supersedes the scientific conclusions of
[RENDERING-B-TRANSFER-SEED0-REPORT.md](RENDERING-B-TRANSFER-SEED0-REPORT.md),
whose checkpoint-schedule confound is corrected here, and it reports the
outcome of the corrected design specified in
[RENDERING-B-TERMINAL-TRANSFER-PLAN.md](RENDERING-B-TERMINAL-TRANSFER-PLAN.md)
on two independent seeds. It also reports a verification, on the same runs,
of a training-log-based measurement instrument that removes most of the
evaluation cost of future experiments of this kind.

---

## Summary

The stage asked one question:

> Does prior training on Rendering A reduce the amount of Rendering B
> experience required to act successfully in held-out Rendering B worlds,
> relative to an identically initialized model trained on B from scratch?

Across two independent seeds, with independently annealed terminal runs at
each budget, the answer is:

1. **The transfer advantage is real, and it is an acquisition effect.** At
   the smallest calibration budget (~3M Rendering B tokens), the
   from-initialization arm has exactly zero successes on both seeds (0/1024
   and 0/1024), while the A-trained arm has nonzero closed-loop capability on
   both (195/1024 and 27/1024). Each seed is independently significant
   (seed 1: p ≈ 6×10⁻⁹ by Fisher exact; seed 0 far smaller).

2. **There is no final-performance advantage.** At the full 100M-token
   budget the arms are statistically indistinguishable on both seeds
   (41.8% vs 41.6%; 40.5% vs 42.0%). Prior A training changes where the
   capability curve starts moving, not where it ends.

3. **The magnitude of the early advantage is strongly seed-dependent**
   (+19.0 pp vs +2.6 pp at 3M). The sign is stable; the size is not.

4. **The mid-budget differences are noise.** The 30M-token diffs flip sign
   across seeds (+9.9 pp on seed 0, −5.8 pp on seed 1), and the loss-based
   instrument below reads null at that budget on both seeds.

5. **A prequential instrument computed from the training logs alone
   reproduces all of the above at zero evaluation compute**, provided the
   re-encoding cost paid in the first ~31 updates is separated out rather
   than summed in.

Point 2 is not a disappointment; it is the intended shape of the result.
The project's premise is that prior structured training installs **bias**
— a reduction in the cost of acquiring later capability — rather than
**knowledge** of the later task. A bias, by definition, pays out in the
adaptation curve's slope and vanishes at convergence. That is exactly the
measured pattern, on both seeds.

---

## 1. Why the original measurement was corrected

The original seed-0 experiment trained each arm once on a single 100M-token
cosine schedule and evaluated intermediate checkpoints. Those intermediate
checkpoints were legitimate training-dynamics observations, but they were
not terminal models optimized for their reported budgets: a checkpoint taken
mid-schedule has a high current learning rate, so learning rate and token
exposure were confounded. The dramatic non-monotonicity in the original
curves (the A-trained arm falling from 21.3% to 1.3% at 30M before
recovering) lives in that confound.

The corrected design trains, for each arm and each nonzero budget, an
**independent run whose warmup-plus-cosine schedule terminates at that
budget**, always restarting from the arm's original state. Every endpoint
owns its complete schedule and finishes with a terminal learning rate below
1e-9. Both arms consume the identical calibration shard in the identical
order and are evaluated on the same 1,024 held-out Rendering B worlds.

Original versus corrected, seed 0, irreversible condition (A-trained −
Init, percentage points):

| B tokens | Original (mid-schedule) | Corrected (terminal) |
|---:|---:|---:|
| 3M | +24.8 | +19.0 |
| 10M | +21.3 | +1.0 |
| 30M | −38.0 | +9.9 |
| 100M | +2.4 | +0.2 |

The 3M advantage survives the correction at reduced size. The 10M advantage
was mostly a schedule artifact: an annealed init run reaches 38.4% at 10M,
where the mid-schedule init checkpoint had 0.0%. The violent 30M collapse
disappears entirely.

---

## 2. Apparatus

The apparatus is unchanged from the seed-0 report except for the schedule
policy: same 19.2M-parameter GPT-NeoX model (6 layers, width 512, 8 heads,
2,048-token byte-level context, vocabulary 262), same Hugging Face
Trainer/Accelerate stack on two T4 GPUs in FP16, same 32,768-token nominal
global batch, same Rust world executor and public parser for closed-loop
evaluation. Seed 1 is a full replication with an independently trained dense
Rendering A source model, an independent root seed, and independent
calibration and held-out world seeds.

Endpoint grid per arm and seed: 92, 306, 916, and 3,052 updates
(3,014,656 / 10,027,008 / 30,015,488 / 100,007,936 nominal B input tokens),
plus a budget-0 zero-shot diagnostic.

---

## 3. Results

### 3.1 Zero-shot diagnostic

At budget 0 both arms on both seeds emit 100% malformed actions and succeed
on nothing. As before, this measures an ungrounded interface, not a failure
of process transfer; it is reported only as a diagnostic.

### 3.2 Held-out Rendering B, irreversible condition

Closed-loop success on 1,024 held-out worlds:

**Seed 0**

| B tokens | A-trained → B | Init → B | Difference |
|---:|---:|---:|---:|
| 3,014,656 | **19.0%** | 0.0% | **+19.0 pp** |
| 10,027,008 | 39.4% | 38.4% | +1.0 pp |
| 30,015,488 | 29.0% | 19.1% | +9.9 pp |
| 100,007,936 | 41.8% | 41.6% | +0.2 pp |

**Seed 1**

| B tokens | A-trained → B | Init → B | Difference |
|---:|---:|---:|---:|
| 3,014,656 | **2.6%** | 0.0% | **+2.6 pp** |
| 10,027,008 | 3.0% | 0.2% | +2.8 pp |
| 30,015,488 | 35.7% | 41.5% | −5.8 pp |
| 100,007,936 | 40.5% | 42.0% | −1.5 pp |

Two features are consistent across seeds. First, at 3M tokens the init arm
is at exactly zero on both seeds while the A-trained arm is not — the
qualitative early-transfer effect replicates. Second, both seeds converge to
the same ~40–42% band at 100M with no meaningful arm difference.

Seed 1's entire acquisition curve is shifted to larger budgets: both of its
arms remain nearly incapable at 10M (3.0% and 0.2%) and acquire between 10M
and 30M, whereas seed 0's arms acquire between 3M and 10M. The transfer
advantage is visible on both seeds at the budgets where acquisition is in
progress, and its measured size depends strongly on where the evaluation
grid happens to intersect each seed's acquisition window.

### 3.3 Action-interface behavior at 3M

The early advantage is not parseable syntax alone. At 92 updates the init
arms emit essentially only parseable-but-invalid actions:

| Seed | Arm | Success | Invalid actions | Malformed |
|---|---|---:|---:|---:|
| 0 | A-trained | 19.0% | 49.9% | 0.0% |
| 0 | Init | 0.0% | 100.0% | 0.0% |
| 1 | A-trained | 2.6% | 93.6% | 0.0% |
| 1 | Init | 0.0% | 100.0% | 0.0% |

### 3.4 Reversible control

A-trained − Init on the matched reversible control:

| B tokens | Seed 0 | Seed 1 |
|---:|---:|---:|
| 3,014,656 | +7.8 pp | +0.8 pp |
| 10,027,008 | +6.1 pp | +0.2 pp |
| 30,015,488 | −4.2 pp | −5.4 pp |
| 100,007,936 | +1.1 pp | −4.2 pp |

The control reproduces the temporal pattern of the primary condition:
positive early, no consistent late advantage. This is consistent with
general behavioral transfer across representations. It does not by itself
establish transfer of the irreversible-versus-reversible process
distinction, which remains unmeasured (see §6).

---

## 4. Interpretation: bias, not knowledge

The two-seed pattern — sign-stable early advantage, magnitude variance,
ceiling equality — is the signature predicted by the project's framing of
what prior structured training should install. Knowledge of Rendering B
would produce nonzero zero-shot capability or a persistent final advantage;
neither occurs. A transferable bias — reusable structure that lowers the
cost of acquiring the new surface form — produces exactly an earlier
acquisition onset that full training erases. The measured quantity of
interest for this and future transfer experiments is therefore the
**adaptation curve**, not any single-budget endpoint, and certainly not the
converged endpoint.

Stated as the estimand of record: prior Rendering A training reduced the
Rendering B experience required for first nonzero held-out capability on
both seeds, with no effect on converged performance.

---

## 5. Instrument verification: prequential measurement from training logs

Because both arms consume the identical calibration stream in the identical
order, the summed online training loss of each run is a prequential
codelength of the same data — the Surplus Description Length of Rendering B
from each starting point, up to a shared constant. This section verifies,
against the closed-loop results above, whether that quantity — computed
purely from the saved `trainer_state.json` logs, with no GPU work — would
have reached the same conclusions. It does, with one correction that the
data itself dictates.

### 5.1 The naive sum has the wrong sign, for an identifiable reason

Summed over a full run, the A-trained arm pays *more* codelength than init
at every budget on both seeds. The excess is paid entirely in the first ~31
updates (~1M tokens):

| Window (updates) | Seed 0: A − Init | Seed 1: A − Init |
|---|---:|---:|
| 0–31 | +85.8 | +91.8 |
| 31–92 | −7.7 | −6.0 |
| 92–306 | −0.4 | −1.0 |
| 306–916 | +3.1 | +2.3 |
| 916–3052 | −1.0 | +2.9 |

The first-step losses explain it: the A-trained model opens at loss ~11.9
(seed 0) / ~11.5 (seed 1) against the init arm's ~5.8 / ~5.5 (≈ uniform
over the 262-token vocabulary). The A-trained model is not ignorant of
Rendering B; it is *confidently wrong*, predicting Rendering A surface
forms. This spike is the re-encoding cost of the representation change,
paid up front, and it is structural: same shape, same location, both seeds.
STEP-1.md already requires the closed-loop transfer measurement to be taken
after a small, separately reported interface-calibration budget; the
codelength instrument requires exactly the same treatment.

### 5.2 With the calibration prefix separated, the instrument reproduces every conclusion

Excluding the first 31 updates, the A-trained arm accumulates less
codelength in all eight independent runs:

| Budget (updates) | Seed 0: A − Init | Seed 1: A − Init |
|---:|---:|---:|
| 92 | −3.59 | −3.17 |
| 306 | −2.26 | −1.66 |
| 916 | −6.48 | −2.43 |
| 3052 | −5.95 | −1.81 |

Sign-correct everywhere, and uniformly smaller on seed 1 — matching seed
1's weaker closed-loop transfer.

The terminal-loss intercepts (mean ± std of the last 10 logged updates)
carry the same information with usable error bars:

| Budget | Seed 0 diff | Seed 0 closed-loop | Seed 1 diff | Seed 1 closed-loop |
|---:|---:|---:|---:|---:|
| 92 | **−0.0171 (~8σ)** | +19.0 pp | **−0.0112 (~4σ)** | +2.6 pp |
| 306 | −0.0023 (noise) | +1.0 pp | +0.0006 (noise) | +2.8 pp |
| 916 | −0.0003 (noise) | +9.9 pp | +0.0008 (noise) | −5.8 pp |
| 3052 | +0.0022 (noise) | +0.2 pp | +0.0051 (noise) | −1.5 pp |

Three verifications stand out:

- **Sign and seed-ordering of the real effect.** At budget 92 the
  instrument is clean and negative on both seeds, and larger on seed 0 —
  the same ordering as the closed-loop advantage.
- **Correct null where closed-loop was unstable.** At budget 916 the
  closed-loop diff flips sign across seeds. The instrument reads null on
  both seeds at that budget. The cross-seed comparison needed two full
  seeds to identify the 30M diffs as noise; the loss log said so from each
  seed alone.
- **Localization.** The largest per-update codelength advantage sits in
  the 31–92 window on both seeds — where the closed-loop effect lives.

### 5.3 Limitation

The mapping from codelength to closed-loop behavior is steep near the
acquisition threshold: −0.011 terminal-loss difference corresponds to
+2.6 pp on seed 1, while −0.017 corresponds to +19.0 pp on seed 0. The
instrument detects, localizes, and ranks transfer; it does not calibrate
behavioral magnitude. Closed-loop evaluation remains necessary to translate
the signal into success rates — but only at the points the log flags, not
on a dense grid. The logged loss is also a per-update mean over supervised
action-label tokens, so these sums support arm-to-arm comparison on an
identical stream but are not a total-bits codelength.

### 5.4 Methodological consequence

The four-endpoint terminal grid exists because "success at budget X" cannot
be read off a single run without the schedule confound. The prequential
quantity can. For future arms and seeds of this family, the default
protocol is:

1. one continuous training run per arm on the shared stream;
2. transfer estimated from the loss logs, with the interface-calibration
   prefix reported separately;
3. closed-loop evaluation at the budgets the logs identify as
   discriminating, plus the converged endpoint.

This replaces roughly 1.43 full training budgets per arm (the terminal
grid) plus per-checkpoint interactive evaluation with one budget per arm
and a handful of targeted evaluations.

---

## 6. What this stage establishes, and what it does not

Established, on two independent seeds:

- Prior Rendering A training confers nonzero early closed-loop capability
  under a re-encoded interface at a budget where from-scratch training has
  exactly none (0-vs-nonzero, 2/2 seeds, each independently significant).
- The advantage is an acquisition-cost effect: converged performance is
  equal across arms on both seeds.
- The advantage is not reducible to parseable syntax.
- The apparent mid-budget instabilities of the original design were
  artifacts of the schedule confound and of checkpoint-grid noise.
- Prequential measurement from training logs reproduces the sign, the seed
  ordering, the localization, and the nulls of the closed-loop results on
  this family, at zero evaluation compute, once the re-encoding spike is
  separated out.

Not established:

- The magnitude distribution of the early advantage (two draws, high
  variance: +19.0 pp and +2.6 pp).
- Transfer of the irreversible-versus-reversible process distinction
  itself. The reversible control transferred similarly to the primary
  condition; the discriminating comparison (matched process-contrast
  behavior under transfer, with surface/target-shuffled controls) was not
  run in this stage.
- Generality beyond this world family, model scale, or tokenization.
- Anything about the dense-teacher versus outcome-only RLVR comparison,
  which is a separate, unrun stage of the Step 1 matrix.

### Decision: seed 2 is not run in this stage

A third seed cannot change the qualitative conclusion (the 0-vs-nonzero
sign is 2/2 with each seed independently significant) and would only add
one draw to a magnitude estimate that is already known to be high-variance.
Its cost is a full dense source-model run plus a transfer run. Under the
protocol of §5.4 a future seed costs roughly a quarter of what seeds 0 and
1 cost; if the magnitude distribution ever becomes the question, that is
the way to buy it.

---

## 7. Reproducibility record

Shared:

- Contract: `step1_rendering_b_terminal_transfer_v1`
- Schedule policy: independent per-budget warmup(2%)-plus-cosine, terminal
  learning rate ≤ 1e-9, restart from arm initialization per budget
- Budgets (updates / nominal global input tokens): 92 / 3,014,656;
  306 / 10,027,008; 916 / 30,015,488; 3,052 / 100,007,936
- Evaluation: 1,024 held-out worlds per condition, closed-loop through the
  public Rust parser and executor
- Hardware: two NVIDIA Tesla T4 GPUs per run
- All runs passed the exact operational contract: source identity,
  calibration-prefix identity, two-rank completion, terminal schedule
  ownership, and exact state-dictionary serialization.

Seed 0:

- Dense Rendering A source commit: `84f29385ed623500aa2e201c45fdcf8c2257fac0`
- Source config hash: `b91eedc2655253edd662c320753c2680cb7842484411c22130deeda31a58cb14`
- Source model-state SHA-256: `cef0ac5a4159d65eae336be58c34dfe1e3f078a8024f4b98a7ca1a78e42a9a6e`
- Root seed `20260811`; calibration seed `23260811`; held-out seed `21260811`
- Calibration-shard SHA-256: `34e92a8e5c6f483a7a83689798fbd07095e65d7c2bfbf81ab3f26bb4c8d076b9`

Seed 1:

- Dense Rendering A source commit: `e592ba12c19992a33246123fad82a6406f4bc771`
- Source config hash: `09aeed773a4eec4299e20d1bc0200adb4d939fb5078b460616d1001a5e87f367`
- Source model-state SHA-256: `837165e09c4f58fc00da53ce88d240d34871d504b40075bb1aa57b9f7f92492f`
- Root seed `20260812`; calibration seed `23260812`; held-out seed `21260812`
- Calibration-shard SHA-256: `8e33766ba09f7d478cb7bd881a8dff5bd955999968e518da67388eba61c75c31`

Machine-readable results: `rendering_b_terminal_transfer_report.json` in
each seed's result bundle. The instrument quantities in §5 are computed
from the `log_history` arrays of each endpoint's `trainer_state.json`
(per-update supervised-token mean loss): prequential sum = Σ loss over
updates; prefix-excluded sum drops updates 1–31; terminal intercept = mean
± std of the last 10 updates.

---

## 7a. Postscript: what happens to Rendering A

The stage measured its endpoints on B only. Scoring the same seed-0 endpoints
back on the 1,024 held-out **Rendering A** worlds the dense arm was scored on,
with the same greedy evaluator:

| model | A success | malformed on A |
|---|---:|---:|
| dense seed 0, before any B training | 41.1% | 0.000 |
| A-trained → B, 92 updates (~3M B tokens) | **0.0%** | **1.000** |
| A-trained → B, 306 updates | 0.0% | 1.000 |
| A-trained → B, 916 updates | 0.0% | 1.000 |
| A-trained → B, 3,052 updates (converged) | 0.0% | 1.000 |
| Init → B, 3,052 updates (A-naive control) | 0.0% | 1.000 |

**Surface forgetting is total, and it is immediate.** By the earliest measured
budget — 92 updates, roughly 3M B tokens, the same point at which the A-trained
arm first shows nonzero B capability — the model can no longer emit a single
parseable Rendering A action. It is indistinguishable from the arm that never
saw A. The five checkpoints are confirmed distinct by state hash; this is five
models forgetting, not one model scored five times.

This is the exact mirror of the stage's own zero-shot finding. Before
calibration the A-trained model was 100% malformed on B; after calibration it
is 100% malformed on A. The learner speaks one rendering at a time, and
acquiring the second overwrites the first's action vocabulary within the first
few million tokens.

### What this does and does not show

It shows the **action surface** is overwritten completely. It does not show the
process knowledge underneath is gone, and the closed-loop metric cannot: a model
that emits no parseable action never gets to demonstrate whether it would have
probed sensibly. Zero here means "cannot speak A", not "cannot do A".

The graded measurement that does separate them is teacher-forced NLL on A,
which needs no parseable output. Scored against one shard of A teacher targets
shared by every model:

| model | A-target NLL |
|---|---:|
| dense seed 0, never trained on B | **0.0853** |
| A-trained → B, 92 updates | 6.9279 |
| A-trained → B, 306 updates | 6.9241 |
| A-trained → B, 916 updates | 7.8341 |
| A-trained → B, 3,052 updates | **9.5246** |
| Init → B, 3,052 updates (A-naive) | **11.6026** |

**Something survives, and it is faint and shrinking.** The converged A-trained
endpoint sits 2.08 nats below the A-naive control — 17.9% lower, a real and
consistent gap — so prior A training remains detectable after 100M tokens of B.
It is not nothing.

But the scale matters more than the gap. Uniform prediction over the 262-token
vocabulary is ln(262) ≈ 5.57 nats. **Every B-trained model is far above that**,
between 6.9 and 11.6, and the converged endpoint is 112× worse than the 0.0853
it started from. These models are not uncertain about Rendering A; they are
confidently wrong about it, which is exactly the mirror of §5.1's finding that
an A-trained model opens on B at loss ~11.9 against an untrained model's ~5.8.

Retention also decays monotonically with B exposure — 6.93, 6.92, 7.83, 9.52 —
heading toward the A-naive floor rather than plateauing above it.

So the answer to "total, or do they still remember" is: **behaviourally total,
distributionally not quite.** What remains is a trace, not a capability. A
plausible reading is that the trace is what would make re-acquiring A cheap,
the same asymmetry this stage measured in the forward direction — but that is a
hypothesis about relearning speed, and testing it needs a relearning run, not a
static score.

One limitation on the trace itself: the NLL is computed over action-token spans,
so a residual advantage could reflect retained decision competence expressed in
the wrong vocabulary, or merely shared low-level byte structure. This
measurement cannot separate those.

---

## 8. Stage conclusion

Per STEP-1.md §14, a stage is scientifically complete when its results
distinguish one of the admissible conclusions. This stage distinguishes
one: **dense-teacher training on one rendering installs a transferable
acquisition bias — it reduces the new-representation experience needed for
useful closed-loop behavior, without changing converged performance — and
the effect is sign-stable but magnitude-variable across seeds.** The stage
additionally leaves behind a validated low-cost instrument that changes how
every subsequent comparison in this project is measured.

The open questions this stage hands to the next one are the process-
distinction transfer claim (irreversible versus reversible, with shuffled
controls) and the dense-versus-RLVR comparison — both now measurable at a
fraction of the evaluation cost paid here.
