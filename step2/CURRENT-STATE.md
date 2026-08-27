# STEP 2 Current State and Next Action

**Status:** current user-requested cold-start handoff, 2026-08-28

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

Three attribution entry points now exist alongside the candidate path, each
selected by the presence of its own config section and each skipping the
phases that exist to make a candidate promotable:

- `capacity_probe.py` (`[probe]`) fits a fixed cohort under two key-identity
  encodings and applies a decision table declared before the run;
- `dagger.py` (`[dagger]`) rolls the current policy forward and labels the
  states it visits with the public-prefix oracle; and
- `budget_curve.py` (`[curve]`) records held-out per-dimension teacher-forced
  L1 at fixed checkpoints against an identical support.

The pinned parameter-count and profile drift guards now exempt a declared
probe arm rather than requiring the selected numbers to be edited to run one.
The canonical `sinusoid` profile still asserts `21,257,489` exactly.

Local verification currently passes 11 Rust tests and 23 Python tests, Rust
formatting, and `git diff --check`; the audited run logs confirm the same
counts on Kaggle. The full 4,096-episode, five-policy
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

## Result of the hour-scale candidate, run `f9b1647`

Budget was raised from 256 to 35,000 updates, about 59 minutes of lineage
training on two T4s, with learning rate, seed, batch size and world held
fixed so budget is the only training factor varied. Held-out support was
widened from 64 to 512 episodes. The run completed and passed audit
verification.

**Budget was the explanation for the earlier null result.** Every paired
metric improved against the same step-zero weights:

| metric | step zero | 35,000 updates |
| --- | --- | --- |
| closed-loop terminal error | 0.5090 | 0.3743 |
| fractional error reduction | -0.6275 | -0.0326 |
| teacher-forced action L1 | 0.7263 | 0.2812 |
| teacher-forced future L1 | 0.2393 | 0.0207 |
| success at error <= 0.01 | 0.006 | 0.330 |
| success at error <= 0.05 | 0.088 | 0.330 |

**The world teaches the targeted capability, and dimension one is solved
exactly.** Per-dimension terminal error and success at `0.01`:

| d | terminal error | success <= 0.01 |
| --- | --- | --- |
| 1 | 0.3418 -> 0.0000 | 0.000 -> 1.000 |
| 2 | 0.4993 -> 0.3758 | 0.008 -> 0.219 |
| 3 | 0.5788 -> 0.5693 | 0.008 -> 0.070 |
| 4 | 0.6161 -> 0.5522 | 0.008 -> 0.031 |

Every `d=1` episode is solved to within `0.01`, and teacher-forced action L1
at `d=1` is `0.0000`. In-context system identification followed by inverse
control is therefore learnable from this world's public experience. Competence
falls off sharply as the number of actuator/channel bindings grows.

**The learner is not yet competent overall, and the defect is specific.**
Mean terminal error `0.3743` remains worse than the zero-action policy's
`0.3385`, while success at every threshold is far better than zero action's
`0.000`. The policy is bimodal: it solves a third of episodes essentially
exactly and drags its own mean with the rest. The trial curve locates the
cause:

| actions allowed | mean error | success <= 0.05 |
| --- | --- | --- |
| 0 | 0.3371 | 0.000 |
| 1 | 0.3157 | 0.100 |
| 2 | 0.3312 | 0.330 |
| 3 | 0.3579 | 0.330 |
| 4 | 0.3743 | 0.330 |

Error falls after the first action, then rises monotonically. Success
saturates by the second action and never improves. The learner makes a good
opening move and then damages every episode it has not already solved: it has
learned to identify and invert, but not to stop. Residual correction is
divergent rather than contractive.

This is the specific next target. It is a property of the learner or the
interface, not evidence that the world is wrong: the privileged oracle scores
`0.0000` terminal error and `100%` at `0.01` on every dimension.

**Correction, 2026-08-28.** The earlier reading of this run as overshoot was
wrong, and it was wrong because the comparison was against step-zero weights
rather than against the oracle. Mean action magnitude did rise from `3.931` to
`5.403` against step zero, but the same run's `oracle_closed_loop` record puts
the oracle's own normalized action L1 at `1.3203`, `2.5723`, `3.9805` and
`5.1445` for `d=1..4` against the learner's `1.3203`, `2.7331`, `5.1344` and
`5.1266`. The learner spends the oracle's action budget, exactly at `d=1` and
at `d=4`, and spreads it over roughly twice as many control steps: `3.52`,
`3.85` and `3.94` where the oracle uses `1.84`, `1.95` and `1.99`. Magnitude
is right and assignment is wrong. The defect is misdirection, not overshoot,
and any repair aimed at bounding action magnitude would be aimed at the wrong
thing.

The `3e973b6` section above reads its own result as "direction but not
magnitude, and overshoots" from the same step-zero comparison. That reading is
superseded for the same reason.

## Attribution runs, 2026-08-27

`f9b1647` left the `d>=2` failure unattributed, with budget unexcluded. Three
paired runs were then executed to decide what the failure is made of. All
three passed audit verification; artifacts are under `step2/audit/runs/`.

### Capacity probe, run `30df00c`

A fixed 64-episode cohort with the world pinned to `d=4`, fitted for 4,000
updates under two key-identity encodings differing in nothing else.

| arm | action L1 before | action L1 after | future L1 |
| --- | --- | --- | --- |
| `sinusoid` (selected profile) | 0.7003 | 0.0235 | 0.0156 |
| `learned` (`nn.Embedding`) | 0.6657 | 0.0275 | 0.0122 |

Both arms landed inside the predeclared `0.02`–`0.05` no-decision band, so
the rule returns `inconclusive-no-decision-band` and the fit bar stands
unmoved. Two findings do not depend on that threshold.

**The representation is sufficient.** Against `0.3317` held-out at `d=4`, the
same architecture reaches about `0.024` on data it can see repeatedly. A 21M
core with the frozen Fourier key code can represent four-way binding and
inversion. Capacity is not what the cliff is made of.

**The key-encoding hypothesis is falsified.** The learned table scored worse
than the frozen sinusoid. That is a between-arm comparison under identical
conditions and does not depend on the absolute threshold. The assistant had
proposed adopting learned key embeddings; the probe removed that proposal
before an hour-scale run was spent on it.

This is a necessary-condition result only. Sixty-four episodes over 4,000
updates is roughly 500 epochs, so part of the fit is memorization. It shows
the representation can hold the mapping, not that the learner can acquire it
from a stream.

### DAgger arm, run `d2a1a23`

Paired against `f9b1647` on identical world `d=1..4`, seeds
`11001`/`22001`/`44001`, 35,000 updates, `21,257,489` parameters, and the same
512-episode closed-loop and 128-episode-per-dimension teacher-forced supports.
Only the source of training states changed.

Labels come from `RolloutBatch`'s `PublicOracle::from_public_prefix` evaluated
at learner-visited states. That oracle reconstructs from the public
calibration prefix rather than latent instance state, so its action remains
derivable from what the learner observes, which is what `WORLD-VALIDITY.md`
requires of a supervised target. Beta is one for a 5,000-update warmup, then
half of each batch is drawn from a 512-episode sliding buffer refreshed every
3,000 updates; eleven refreshes ran.

| metric | step zero | DAgger | `f9b1647` BC |
| --- | --- | --- | --- |
| closed-loop terminal error | 0.5090 | 0.3073 | 0.3743 |
| mean fractional error reduction | -0.6275 | +0.1362 | -0.0326 |
| teacher-forced action L1 | 0.7263 | 0.2901 | 0.2812 |
| mean steps | 3.88 | 3.24 | — |

Against the zero-action policy's `0.3385`, this is the first learner in this
world that is better than doing nothing rather than worse, and the first with
a positive mean fractional error reduction.

| d | DAgger terminal | BC terminal | DAgger teacher-forced | BC teacher-forced | success `<=0.05` |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.0001 | 0.0000 | 0.0003 | 0.0000 | 1.000 |
| 2 | 0.3215 | 0.3758 | 0.2698 | 0.2582 | 0.289 |
| 3 | 0.4218 | 0.5693 | 0.3142 | 0.3049 | 0.047 |
| 4 | 0.4859 | 0.5522 | 0.3409 | 0.3317 | 0.023 |

**The teacher-forced column moves the wrong way at every dimension.** That
trade is the mechanism confirming itself: half of each batch is now off-path,
so on-path fit gives up a little and closed-loop robustness gains a lot. A
confound would have moved both columns together.

**It is not a solution.** Success at `0.05` is `0.047` at `d=3` and `0.023` at
`d=4`. Overall success of `0.340` is essentially `d=1` always plus a quarter
of `d=2`. The learner still spends 3.9 of 4 control steps where the oracle
spends 2.

### Budget curve, run `7fd41d7`

World pinned to `d=3..4` for 60,000 updates, with ten checkpoints scored on an
identical held-out support regenerated from `validation_seed` at
`start_index` zero.

| updates | d=3 action L1 | d=4 action L1 | d=3 future L1 |
| --- | --- | --- | --- |
| 0 | 0.7405 | 0.7093 | 0.2377 |
| 6,000 | 0.4281 | 0.4170 | 0.0249 |
| 24,000 | 0.4262 | 0.4148 | 0.0236 |
| 48,000 | 0.4269 | 0.4136 | 0.0222 |
| 60,000 | 0.4261 | 0.4112 | 0.0203 |

**Budget is not the constraint.** Ninety-eight to ninety-nine percent of the
total reduction happens in the first ten percent of the budget. The remaining
54,000 updates buy `0.002` at `d=3` and `0.006` at `d=4`, and the run declares
both dimensions not still improving.

**The forward and inverse models come apart.** Future L1 keeps declining
across the whole run while action L1 is flat from 6,000 onward. More budget
keeps improving effect prediction and does nothing for action selection.

**Pinning to the hard dimensions was worse than the mixed curriculum.** This
run plateaus at `0.4261`/`0.4112` for `d=3`/`d=4` against `f9b1647`'s
`0.3049`/`0.3317`, despite giving those dimensions more exposure. The held-out
draws are not the same episodes, because `sample_instance` consumes `d_min`
and `d_max`, so pinning changes which instances a seed produces. The gap is
nonetheless far larger than sampling noise over 128 episodes per dimension.
The signal is that the easy dimensions scaffold the hard ones. It has not been
isolated by a controlled comparison and must not yet be stated as established.

## Evidence level

`0.2.0` now has audited two-T4 apparatus results and a retained
35,000-update candidate checkpoint with partial source-world competence:
dimension one solved exactly, degrading with dimension, and a located
non-termination defect in multi-step correction. It has no transfer evidence.

The three attribution runs above change what the `d>=2` failure may be
attributed to.

| hypothesis | status | evidence |
| --- | --- | --- |
| representation or capacity | excluded | `30df00c`: fits `d=4` cohort to `0.024` against `0.3317` held-out |
| key encoding specifically | excluded | `30df00c`: learned table scored worse than frozen sinusoid |
| optimization budget | excluded | `7fd41d7`: ten times the updates buys under 1.5 percent |
| training distribution | real but partial | `d2a1a23`: closed-loop `0.3743` to `0.3073`, error reduction changes sign |

The learner is therefore a characterized instrument rather than an unknown.
It can represent the task, it saturates in about 6,000 updates, and it
responds to a distribution correction with the expected on-path for off-path
trade. A remaining `d>=2` failure is now attributable to the world and its
interface rather than to the learner. Budget is no longer an available
explanation for a null result on this world.

`c1-start-candidate` from run `f9b1647` has not been promoted to Checkpoint 1.
No numeric promotion threshold was predeclared and none may be invented after
the fact; promotion is the user's decision.

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

The prior next action, the bounded `0.2.0` C1 candidate, has been executed as
`3e973b6` and `f9b1647` and is complete. Under this document's own decision
rule the outcome is `passes infrastructure but not learning`, whose stated
consequence is to repair or reject the current world and interface before
adding worlds. The three attribution runs then removed budget and capacity as
explanations, so that repair is now the live work.

**Everything below this line is an assistant proposal and is not a user
decision.** Per `step2/META-PROCESS.md`, agent-authored text does not become a
constraint by existing in the repository.

**No further learner-side probe is proposed.** Capacity, key encoding, and
budget are excluded; distribution is measured and partial. A fourth
learner-side run would be the accumulation of runs for their own sake that
`META-PROCESS.md` names as a loss of velocity.

Three candidates follow from the attribution runs, in the order the assistant
would take them.

1. **Build the operator algebra.** `principled_developmental_worlds.md` §2
   defines the research object as an algebra of structural transformations,
   `W_{n+1} = F_i(W_n)`, over thirteen named operators. None of them exists in
   code. `FamilyConfig` has twelve knobs and every one is a magnitude inside a
   fixed law, so no setting of it produces a different world. Until a world is
   a value rather than a commit, the thousand-world goal costs one world at a
   time. This also raises the value ceiling discussed below, so it is one piece
   of work answering two open problems.

2. **Make DAgger the default collection mode.** It is a strict improvement on
   the metric that matters, it is already implemented in
   `step2/python/step2_experiments/dagger.py`, and its labels stay inside the
   information boundary. The expert-only corpus is a property of
   `generate_trajectory`, which terminates the instant the oracle succeeds and
   errors out when it fails, so no world built on that generator can contain
   recovery behaviour.

3. **Isolate the curriculum question.** `7fd41d7` suggests the easy dimensions
   scaffold the hard ones, but its supports were not matched. A controlled
   comparison would fix the evaluation support and vary only the training
   mixture.

The value ceiling of this world should be stated plainly when that work is
planned. `PublicOracle::from_public_prefix` is closed form and the calibration
prefix determines the mapping exactly, so the world teaches a function rather
than a procedure. That was the right property for a first world, because it
made two null results interpretable, and it caps what a checkpoint from this
world can carry. Partial calibration, observation noise, action cost, or
coupling would each turn the lookup into a procedure.

Any GPU run still uses the sole orchestration entry point:

```text
python tools/kaggle_run.py launch --experiment <name>
```

Source changes must first be reviewed, committed, pushed, and resolved to the
exact remotely reachable Git SHA required by the execution plan. A fresh agent
must present the run's purpose, bounded scope, expected cost, and stopping
interpretation to the user before causing the external launch; this handoff
records candidate next actions but is not itself launch authorization.

## Candidate promotion

`c1-start-candidate` from `f9b1647` has not been promoted to Checkpoint 1, and
the DAgger arm did not retain a checkpoint. No automatic numeric promotion
threshold has been approved. Do not invent one after seeing a result, and do
not call any candidate C1 without the user's promotion decision.

The DAgger arm is the strongest source-world result so far and is the natural
parent if a lineage is resumed, but it was run as an attribution probe: it
skipped the lineage, the retained checkpoint, and the counterfactual band. A
promotable DAgger candidate would have to be run again through the full
candidate path.

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

At the time of this handoff the worktree is clean apart from an untracked
`tmp/`. The `0.2.0` migration and the three attribution runs are committed on
`codex/step2-vertical-slice` and pushed. A fresh agent must still inspect
`git status` rather than assuming this remains true.

`step2/WORLD-VALIDITY.md` remains a 26-line draft pending user review. It has
four unnumbered headings, so the `§7` and `§10` cross-references in
`README.md` do not resolve; the selection and translation-bridge problems are
named in `README.md` but are not written down anywhere. Preserve the draft and
do not represent it as accepted merely because it is present.
