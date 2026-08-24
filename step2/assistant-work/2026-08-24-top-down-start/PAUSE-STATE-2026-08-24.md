# Paused State: Audit and Expanded Goal

> **Restarted and superseded as an entry point.** This remains the chronological
> audit record, including deliberately stale paused-state text. Fresh agents
> must use [`../../CURRENT-STATE.md`](../../CURRENT-STATE.md) for current status
> and next action.

**Date:** 2026-08-24  
**Status:** deliberately paused at the user's request. The pause now covers two
distinct things: an unfinished apparatus audit, and a user-directed expansion of
the project goal recorded below. No decision or completed correction is claimed
for the audit. The goal expansion is direction, not a draft.

## What the audit established before pausing

1. The Rust query order, Python query-position extraction, normalized action
   convention, and Rust executor agree by direct code inspection. A stronger
   static-trajectory/online-rollout parity test is still required.
2. `0/64` used a `1e-5` max-coordinate error threshold. A CPU counterfactual
   showed that a policy applying 75% of the current oracle action scores `0%`
   by that gate despite mean terminal error near `0.0031`; the single success
   scalar is therefore inadequate. The trained checkpoint's mean error
   (`0.3430`) is also approximately the zero-action baseline on the same
   64-instance support (`0.3500`), so the result cannot be dismissed as only a
   tolerance artifact.
3. The world generator contradicts its documented factorization claim. In
   version `0.1.0`, target signs, gain signs, and magnitude strata share
   deterministic `index` structure; every target-action coordinate within an
   instance has the same sign. Task sampling also consumes the same RNG stream
   as presentation shuffles, so changing serialization mechanics can change
   the task. Existing tests do not check these invariances.
4. The implemented payload is not the documented normalized payload: action
   tokens contain the raw value `16`. On one selected-model batch, mean initial
   component norms were approximately role `0.40`, deterministic key `13.86`,
   and payload `2.35` (maximum payload-projection norm `6.16`). The unscaled key
   encoding is therefore roughly 35 times the learned role norm at
   initialization. This is a persistent representation risk and must be fixed
   or explicitly justified before continuing the checkpoint lineage.
5. The current fixed-batch overfit gate checks only loss reduction. Its exact
   four-example diagnostic still had `0/4` strict closed-loop success, and the
   gate nevertheless passed. The gate does not yet prove that the architecture
   can turn its readouts into correct actions.
6. The current runner always initializes a fresh model after its disposable
   diagnostic. It has no external parent-checkpoint input, so the documented
   next action `TRAIN_OLD` and a many-checkpoint recursive lineage are not yet
   executable through this runner.
7. The capability documents define a useful forest, but the first-world claim
   jumps from one exposure family directly to `P0`-`P7` and repeatedly uses the
   distant GEN-1.5-like horizon as a local ranking term. It needs evidence
   levels and edge-local claims; a source world may expose several contrasts
   without one short checkpoint acquiring all of those properties.

## Expanded goal recorded during the pause

The user restated the project's central hypothesis during the pause. It is not
"can a 22M VLA be built." It is:

> Can the control/reasoning part of an agent be pretrained in many cheap
> abstract worlds, independently of pixels, language, and any particular robot,
> and then used as a prior when grounding into real sensors, goals, and
> actuators?

The architecture is therefore oriented around an **abstract action policy**.
The core is the only thing pretrained, and it never consumes a pixel or a word.
Its contract is modality-free: a declared interface of what sensors and
actuators exist, abstract observations, an abstract goal, and action queries in,
actions and predicted futures out. Perception, language, and a particular
robot's actuators are **adapters outside the core**, learned later, against a
core that is already competent.

Two consequences follow directly and are part of the direction:

- the headline result is a **relative** sample-efficiency comparison
  (abstract-pretrained core versus scratch core, identical adapters, identical
  real data, identical budget), not an absolute capability claim; and
- the goal channel is an **interface**, not a modality. The synthetic-world goal
  encoder is one implementation of it; language, goal-image, and
  demonstration-derived encoders are other implementations of the same
  interface.

### Resolved: capability and invariance are different roles

The apparent opposition in `WORLD-SELECTION.md` between capability-first and
invariance-first selection was false. They answer different questions:
capability-first and affordance-first **select content**; invariance-first
**supplies an evidence discipline** for a content claim that comes from
elsewhere. The asymmetry is not mutual dependence — a capability can be named
with no invariance machinery, and an invariance can be anchored by something
other than a capability decomposition, such as the known gap between the
constructed worlds and the eventual deployment. `WORLD-SELECTION.md` has been
edited to state this; see the worktree section below.

The practical consequence is that the capability graph is **not** reordered by
the goal expansion. Each capability node's isolating test already is that node's
invariance class. What changes is that the encoding invariances become standing
requirements at every node rather than local to `P0`, `P8`, and `P11`, because a
capability acquired in an encoding-specific form will not survive the adapter
boundary.

### Prior-work position after reviewing AdA

AdA (`Human-Timescale Adaptation in an Open-Ended Task Space`, XLand 2.0) is
important premise-level prior work rather than only a memory/curriculum
citation. It reports an estimated space of more than `10^40` possible tasks, a
sampled single-agent training pool of 25B tasks, and a main single-agent run of
about 100B environment steps. Within XLand, it shows rapid in-context control,
a structured non-linguistic goal grammar, and adaptation to hidden production
rules.

AdA holds its interface fixed: one avatar, one fixed discrete action space, one
camera, one composite observation vector, and one goal grammar. Evaluation
never leaves XLand, and the paper does not discuss transfer to other observation
or action spaces. AdA therefore supports the premise that diverse procedural
experience can produce in-context adaptive competence under a fixed interface.
It does not establish modality-independent abstract pretraining and is silent
on the representation-change half tested here.

Its Transformer sweep spans 6M-265M Transformer parameters, with larger total
agents, but it does not test this project's ~21M interface. The paper reports
effects from model size, memory, task-distribution richness, curriculum, and
distillation; it does not establish that experience or world coverage rather
than capacity is the single binding constraint. Its kickstarting result warns
that large blank-slate RL can acquire harmful early representations, while its
teacher was itself trained from scratch. These are design warnings and scaling
questions, not evidence against a blank abstract core.

### Status of the proposed change list at the pause

An architecture change list was produced during the pause and is an
**assistant-authored proposal**, not direction. It has not been accepted. Its
items, in leverage order, are: train the core on multiple interchangeable
encodings of each goal; do the same for observations; separate goal from
condition; move the visual path out of the core; make forward prediction
explicit and multi-horizon; require grounding worlds to expose both abstract
state and rendering for the same episode; write the grounding experiment down
before pretraining; and scope the blank-state contract to the core so that
adapters may later use frozen pretrained encoders.

On restart, the user authorized work on the audited restart sequence with
velocity as the priority. `step2/CORE-BOUNDARY.md` now records the adopted
minimal architecture boundary. This does not retroactively authorize candidate
world operators or a GPU training run.

Under this repository's own vocabulary, only one of those is operator-shaped and
it is not new: masked conditional dynamics are a realization of `SWITCH_RULE`,
`HIDE`, and the `I3` probing cluster, which the capability decomposition already
names as candidates. The multiple-encoding items are semantic-preserving
variations already demanded by the anti-degeneracy clause. The trials metric is
a measurement at the learner's information boundary. None of them are admitted
world operators, and none authorize a world.

## Partial edits present in the worktree

The original pause inventory incorrectly reported only two files. At restart,
the worktree also contained older uncommitted edits to `step2/README.md` and
`step2/WORLD-VALIDITY.md`, plus untracked `step2/META-PROCESS.md`, this pause
note, and `step2/WORLD-SELECTION.md`. The inventory error is itself part of the
audit record; no uncommitted document is authoritative merely because it was
present.

`step2/crates/world/src/lib.rs` was partially changed by the paused audit, and
`step2/WORLD-SELECTION.md` was edited afterwards to remove the false opposition
between capability-first and invariance-first selection. The selection edit is
assistant-authored and does not authorize a world.

The Rust edit is different: it is unfinished. The edit is intentionally versioned as `0.2.0` and currently:

- domain-separates instance, task, calibration, and presentation RNG streams;
- samples gain/task signs independently instead of deriving both from one
  stratum;
- maps boundary, feedback, and action-horizon payload fields into `[-1,1]`;
- changes the default success tolerance to `0.05`; and
- bumps world, oracle, and token ABI versions.

This is **not a completed correction**. The PyO3/config/model ABI, key-encoding
scale, metrics, stronger tests, behavior gate, old-checkpoint classification,
and documentation have not yet been updated. Do not train from this state.

`cargo check --workspace` passes. `cargo fmt --check` did not run because the
pinned local Rust toolchain lacks the `rustfmt` component.

## Safe restart order

1. Re-read this note and inspect both worktree edits.
2. **Decide the scope of `0.2.0` before completing it.** The goal expansion
   implies interface changes — goal separated from condition, the core's input
   contract declared modality-free, the visual path moved outside the core — and
   the ABI is currently open anyway. Completing `0.2.0` at the audit's original
   scope and deciding afterwards means bumping again immediately and
   invalidating anything trained in between. This is the only item on the list
   with an expiry: it is nearly free while the version is open and expensive
   once a checkpoint lineage exists. It is a user decision, not an audit repair.
3. Complete the `0.2.0` ABI coherently across Rust, PyO3, TOML, and model
   config, at whatever scope step 2 settled.
4. Add exact static-versus-online trajectory/action parity and factorization
   tests before interpreting any learner metric.
5. Correct representation scaling and distributed target normalization, then
   make fixed-cohort closed-loop behavior a disposable architecture gate.
6. Replace the one-scalar rollout report with threshold curves, quantiles,
   error reduction, action cost, and baselines. Add two things the audit showed
   were missing: a trials-indexed adaptation curve, because nothing currently
   measures improvement *within* an encounter even though in-context adaptation
   is the competence being pretrained; and a trivial-policy band evaluated
   **before** a session is scheduled rather than after it, since the zero-action
   baseline (`0.3500`) already bracketed the trained checkpoint (`0.3430`) and
   was found only by post-hoc audit.
7. Rewrite the assistant-owned capability documents around atomic properties,
   evidence levels, immediate dependency edges, and an explicitly open-ended
   hundreds/thousands-of-world process. Two additions from the pause: apply the
   encoding invariances as a standing requirement at every node rather than
   locally at `P0`, `P8`, and `P11`; and reposition AdA from an `I2`/`I3`/`I4`
   citation to premise-level prior work with its interface held fixed.
8. Run CPU tests only. A new GPU run requires a fresh explicit authorization.

## Restart progress

The first restart slice completed the expiring ABI work:

- recorded the accepted modality-free core and relative transfer claim;
- separated `Goal` from `Condition`;
- replaced `OutcomeQuery` with an explicit horizon-bearing `FutureQuery`;
- removed the visual encoder/resampler from the core and added only an optional
  canonical-content embedding input for external adapters;
- scaled deterministic key features to the model initialization scale;
- versioned the Rust crates, Python package, model config, world, oracle, and
  token ABI coherently as `0.2.0`;
- added a fail-fast Rust/model ABI check; and
- added presentation/semantic independence and exact static/online public-token
  parity tests.

The second restart slice replaced the historical direct-Accelerate training
loop with the maintained Transformers `Trainer` path. Trainer now owns optimizer,
scheduler, accumulation, mixed precision, DDP, logging, checkpointing, and
resume. The actual bounded lineage interrupts at the configured smoke step,
writes standard model/optimizer/scheduler state, rebuilds the Trainer, and
resumes to the original total-step budget. The procedural Rust stream and
teacher-forced/closed-loop evaluation remain project-owned scientific code.

Masked regression is now normalized as an equal-per-episode mean. Equal-size
rank shards therefore compose to the full-batch loss without a custom
distributed reduction, and a local integration test fixes that contract.
`rustfmt` was installed and formatting was checked. The final test counts for
the Trainer slice were 11 Rust tests and 19 Python tests, all passing.

The next audit item is also implemented locally. Closed-loop evaluation now
reports predeclared error-threshold curves, error quantiles, absolute and
fractional error reduction, action cost, dimension slices, and a trial-indexed
within-encounter curve. The Kaggle runner computes a zero/scaled-public-oracle
counterfactual band before entering the GPU phase. On the 4,096-episode local
support, zero action scores `0%` at error `0.05`, while half-oracle action scores
`100%` but retains mean terminal error about `0.0391`; this is direct evidence
that the old binary scalar would collapse materially different behavior.

Final local verification now passes 11 Rust tests and 20 Python tests, plus
`git diff --check`. No `0.2.0` GPU run or checkpoint has been launched.

