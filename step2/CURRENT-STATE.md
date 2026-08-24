# STEP 2 Current State and Next Action

**Status:** current user-requested cold-start handoff, 2026-08-25

This is the first document a fresh agent should read. It records the current
scientific object, implementation boundary, evidence level, and next action.
Historical results and assistant-authored proposals do not override it.

## Project claim

The project asks whether a control-and-reasoning core pretrained across many
cheap procedural abstract worlds becomes a useful prior when later grounded
into real sensors, goals, and actuators. The eventual decisive comparison is an
abstract-pretrained core against a scratch core with identical downstream
adapters, real data, and optimization budget.

The core is modality-free. It consumes canonical abstract sensorimotor events;
it never consumes pixels, words, or a robot-specific fixed action vector.
Vision, language, demonstrations, and robot-specific actuator interfaces are
downstream adapters. They are not part of abstract pretraining.

## Current world and targeted capability

The current `0.2.0` world family is calibrated signed-permutation effect
control, implemented as `W_calibrated_monomial[d=1..4]`.

Each instance changes the correspondence, sign, and gain between public scalar
actuators and public observation channels. A task-independent public
calibration trajectory exposes enough experience to identify those effects.
The learner then receives an abstract observation-space goal and must act under
bounded control steps.

This world targets one atomic capability:

> **In-context system identification for goal-conditioned inverse control.**

Concretely, the learner must infer from public experience:

- which actuator affects which observation coordinate;
- the sign and gain of each effect;
- how to invert the inferred mapping to move toward a public goal;
- how to correct residual error over several control steps; and
- how to bind variable public sensor/actuator keys across dimensions and
  presentation permutations.

Horizon-one future prediction is auxiliary supervision for the same
action-effect model. The world does not target vision, language, real-robot
grounding, long-horizon planning, broad partial observability, or cross-world
transfer.

## Learner information boundary

The learner receives public schema, conditions, calibration observations and
executed actions, current observations, a distinct `Goal`, feedback, action
queries, and explicit horizon-bearing `FutureQuery` events.

The learner does not receive the hidden permutation/gain matrix, generator
index, latent RNG state, or privileged oracle actions. Rust owns generation,
transition, serialization, the public-prefix teacher/oracle, verifier, and
online rollout state. Python must not reproduce those semantics.

## Current implementation

- World, oracle, token ABI, Rust crates, Python package, and model config are
  versioned coherently as `0.2.0`.
- `Goal` and `Condition` are distinct roles; `FutureQuery` carries its horizon.
- Presentation randomness is domain-separated from task semantics.
- Static teacher trajectories and online public-oracle rollouts have exact
  parity tests.
- The core is a maintained Hugging Face `LlamaModel` body with thin
  project-owned continuous event adapters and action/future heads.
- The former visual encoder/resampler has been deleted from the core. An
  external adapter may provide aligned canonical content embeddings later.
- Transformers `Trainer` owns optimizer, scheduler, accumulation, mixed
  precision, DDP, logging, checkpointing, and resume.
- The real lineage interrupts at the configured smoke step, saves standard
  model/optimizer/scheduler state, rebuilds Trainer, and resumes without adding
  training steps.
- Masked regression is an equal-per-episode mean, so equal-size DDP rank shards
  compose to the full-batch objective without project-owned collectives.
- Evaluation reports error thresholds and quantiles, absolute and fractional
  error reduction, action cost, dimension slices, and trial-indexed
  within-encounter curves.
- Before the GPU phase, the runner evaluates a zero/scaled-public-oracle
  counterfactual band.
- The run evaluates its own step-zero weights on the held-out support before
  training and reports a paired before/after delta. The step-zero learner and
  the trained candidate share one evaluation code path, so their support is
  identical by construction rather than by two call sites agreeing.

Local verification currently passes 11 Rust tests and 22 Python tests, Rust
formatting, and `git diff --check`. The full 4,096-episode, five-policy
counterfactual band runs locally in about 7.5 seconds. The whole `train.main`
path, including the paired step-zero baseline, was additionally dry-run on the
local CPU at reduced scale; that check proves wiring only and produced no
retained evidence.

## Result of the first `0.2.0` two-T4 candidate, 2026-08-25

Run `step2-architecture-world-3e973b6` completed and passed audit
verification. Receipt and artifacts are under
`step2/audit/runs/step2-architecture-world-3e973b6/`.

**The apparatus works.** Two T4s, 256 updates in 44 seconds of training,
architecture gate passed, checkpoint/resume smoke passed, candidate retained,
19 artifacts hash-verified against the remote manifest. Two earlier attempts
failed first: `efe9dbb` on a DDP dead-parameter fault and `f3dbf51` on a
process-teardown stall. Both were apparatus failures, not evidence about the
world.

**The candidate did not learn control.** On the fixed 64-episode held-out
support, against its own step-zero weights:

| metric | step zero | trained |
| --- | --- | --- |
| closed-loop terminal error | 0.5066 | 0.6308 |
| fractional error reduction | -0.6487 | -0.9264 |
| success at error <= 0.05 | 0.047 | 0.188 |
| success at error <= 0.4 | 0.312 | 0.188 |
| teacher-forced action L1 | 0.7288 | 0.6300 |
| teacher-forced future L1 | 0.2487 | 0.1078 |

Read against the predeclared band, zero action gives terminal error `0.3385`
and the privileged oracle gives `0.0000`. Both learners are therefore worse
than doing nothing, and training moved the learner further from inaction, not
closer to the oracle. Mean action magnitude rose from `3.931` to `5.403`.

The supervised objective did improve: teacher-forced future L1 fell by 57%.
So the model learned to predict effects while getting worse at using them.
Its rising success at tight thresholds alongside worse mean error is the
signature of a policy that has acquired direction but not magnitude, and
overshoots.

**This does not license rejecting the world.** The oracle solves it perfectly
from public information, so the target capability is present and reachable.
The budget was 256 updates costing 44 seconds; it is the cheapest variable
available and has not been varied. The evaluation support is also only 64
episodes, so the success-rate cells are 3 to 12 episodes and are noisy. The
result is `passes infrastructure but not learning` under the decision rule
below, with budget unexcluded as the explanation.

**Reproducibility.** Runs `f3dbf51` and `3e973b6` produced bit-identical
metrics and an identical model `sha256`, so FP16 DDP training on two T4s is
deterministic here and future comparisons can attribute differences to the
change under test.

## Evidence level

`0.2.0` now has a complete, audited two-T4 apparatus result and one retained
candidate checkpoint. It has no source-world competence evidence and no
transfer evidence.

The retained `0.1.0` checkpoints are historical apparatus evidence. They use a
superseded ABI and architecture boundary and must not be parents of the `0.2.0`
lineage.

The counterfactual band is not learner evidence. In an unretained local
contract/timing check on the fixed 4,096-episode support, zero action had mean
terminal error about `0.3385` and `0%` success at error `<=0.05`; half-oracle
action had mean error about `0.0391` and `100%` success at that threshold. This
demonstrates that binary success alone cannot distinguish materially different
control quality. The runner must reproduce and retain the band for an actual
candidate run.

## Next action

The next scientific action is a bounded `0.2.0` **C1 candidate** on the current
world, before creating a second world. Its purpose is to answer whether the
corrected core can acquire the targeted system-identification/control primitive
through its public experience while the maintained two-T4 Trainer path works
end to end.

The checked-in candidate configuration is
`step2/configs/kaggle/t4x2_vertical_slice.toml`:

- two T4 processes with FP16;
- a disposable 128-update fixed-cohort architecture diagnostic;
- diagnostic weights discarded before lineage training;
- a fresh `0.2.0` lineage; the historical `0.1.0` checkpoint is not a parent;
- checkpoint/resume at update 4;
- 256 total candidate updates, batch 8 per device; and
- immediate teacher-forced, learner closed-loop, and public-oracle evaluation.

The fixed untrained-learner evaluation that this document required before
launch is now implemented and locally verified. It evaluates the exact weights
training starts from, on the same held-out support used after training, and
reports `untrained_baseline` plus a signed `paired_learning_delta`.

Note that the step-zero learner is not a do-nothing policy. At initialization
the tanh action head emits actions of substantial magnitude, so the paired
baseline and the zero-action member of the counterfactual band are different
references. Report the candidate against both.

Then use the sole orchestration entry point:

```text
python tools/kaggle_run.py launch --experiment architecture-world-vertical-slice
```

The source changes must first be reviewed, committed, pushed, and resolved to
the exact remotely reachable Git SHA required by the execution plan. Do not
launch directly from the current dirty worktree. A fresh agent must present the
run's purpose, bounded scope, expected cost, and stopping interpretation to the
user before causing the external launch; this handoff records the next action
but is not itself launch authorization.

## Candidate decision after the run

`c1-start-candidate` is not automatically Checkpoint 1. Report it against its
paired untrained learner and the predeclared counterfactual band using the full
metric vector.

- If it exhibits real held-out learning of the targeted primitive, the user may
  promote it to C1 and the next action becomes creation of World 2.
- If it passes infrastructure but not learning, retain it only as apparatus
  evidence and repair or reject the current world/interface before adding
  worlds.
- If execution, DDP, checkpoint/resume, or artifact auditing fails, classify it
  as an apparatus failure; it is not evidence against the world.

No automatic numeric promotion threshold has been approved. Do not invent one
after seeing the result, and do not call the candidate C1 without the user's
promotion decision.

## Authority and document map

Read in this order:

1. `step2/CURRENT-STATE.md` — current state and next action.
2. `step2/CORE-BOUNDARY.md` — accepted modality-free architecture and transfer
   boundary.
3. `step2/META-PROCESS.md` — user-directed working and compute rules.
4. `STANDARD-LLM-STACK-MIGRATION-PLAN.md` — authoritative library ownership
   boundary.
5. `EXPERIMENT-EXECUTION-PLAN.md` — authoritative Kaggle launch, retention,
   retrieval, and audit workflow.
6. `step2/WORLD-VALIDITY.md` — current validity draft; pending user review.
7. `step2/WORLD-SELECTION.md` — assistant-authored proposal, not authorization.

Files under `step2/assistant-work/2026-08-24-top-down-start/` primarily record
the historical `0.1.0` decision and audit trail. Their custom-core, visual-path,
outcome-query, direct-Accelerate, checkpoint, and next-action statements are
superseded for `0.2.0` unless repeated in the current documents above.

## Worktree warning

At the time of this handoff, the worktree is intentionally dirty and contains
the current `0.2.0` migration plus older uncommitted documentation changes. A
fresh agent must inspect `git status` rather than assuming those changes were
committed by a predecessor. In particular,
`step2/WORLD-VALIDITY.md` contains a large pre-existing rewrite that was not
created by the latest implementation slices. Preserve it and do not represent
it as accepted merely because it is present.
