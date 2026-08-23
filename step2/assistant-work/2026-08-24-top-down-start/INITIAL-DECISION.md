# Initial STEP 2 Decision

**Date:** 2026-08-24
**Status:** selected vertical slice implemented; CPU tests completed; one
two-T4 architecture/world gate and bounded `c1-start-candidate` run authorized;
TPU remains unauthorized

## Decision in one statement

Use a **custom but narrowly paper-grounded, fully random ICRT-derived causal
trajectory transformer**: keep ICRT's standard long sensorimotor history core,
replace its fixed state/action interface with variable typed public tokens and
schema-conditioned shared continuous readouts, and preserve non-linguistic
physical demonstrations as native context. Develop it through the same
cost-aware recursive checkpoint policy at every state. At `S0`, train one
bounded monitored session on calibrated signed-permutation effect control with
`d=1..4`; the scalar case is one support slice, not the first world by itself.

## 1. End target

The north star is **GEN-1.5-like physical generalization**:

- infer a new task from one or a few physical demonstrations in context;
- compose more than one physical prompt;
- control closed loop, correct perturbations, improvise, and recover;
- transfer across goals, tools, renderings, and eventually embodiment gaps such
  as human/simulation demonstrations to robot execution; and
- adapt in very few gradient steps when context alone is insufficient.

This is a capability target, not a claim to reproduce Generalist's proprietary
architecture, dataset, scale, or results. A small first checkpoint is expected
to learn simpler dependencies. Its architecture must nevertheless make the end
behavior representable rather than forbid it through a language-only boundary,
fixed robot vector, short stateless policy, or world-specific output head.

The top-down dependency forest and falsifying probes are in
[CAPABILITY-DECOMPOSITION.md](CAPABILITY-DECOMPOSITION.md).

## 2. Recursive process

There is no special root-world mechanism. The blank checkpoint is merely the
first scheduler state:

```text
S_n = (checkpoint,
       evidence and uncertainty,
       current capability/dependency hypothesis,
       admitted worlds and measured costs,
       lineage/retention ledger,
       remaining resources).
```

At every state, select one bounded action:

```text
TRAIN_NEW | TRAIN_OLD | TRAIN_STATIC_MIXTURE
EVALUATE  | ADMIT_NEW | MIGRATE_ARCHITECTURE | STOP
```

Training creates an immutable child checkpoint. Evaluation may leave weights
unchanged while changing the next decision. The process repeats at every
checkpoint, and architecture migration is charged like any other action rather
than treated as free.

The ideal objective is expected terminal end-capability utility under the
remaining budget. The practical initial policy is a logged, human-reviewed,
short-horizon approximation using learning progress, transfer option value,
retention, uncertainty, value of information, and complete wall-clock/resource
cost. A predetermined static multi-world pretraining action remains a legal
candidate and the mandatory baseline.

The canonical policy is in
[RECURSIVE-CHECKPOINT-POLICY.md](RECURSIVE-CHECKPOINT-POLICY.md).

## 3. Model: what is standard and what is custom

### Standard paper-owned core

- random Llama-style causal trajectory decoder as in ICRT;
- 12 layers, width 384, 6 heads, head width 64;
- SwiGLU intermediate width 1024, RMSNorm, and RoPE;
- approximately 21.24M backbone parameters before adapters/readouts;
- ordinary causal self-attention and continuous prediction; and
- initial packed context budget of 2,048 trajectory tokens.

### Custom gap fill

- replace ICRT's one pooled state token and fixed action vector with variable
  typed observation/entity tokens;
- derive one action readout query per public scalar actuator from its
  encounter-local schema;
- use one shared continuous head rather than a robot-class or slot-specific
  head;
- append actual executed actions and outcomes to the causal history; and
- include a random shallow visual patch path with an eight-token-per-view
  resampler for later video worlds.

This makes the end-to-end learner custom. No reviewed paper validates the
complete join. ICRT supplies physical prompting; Octo/Astra/CrossFormer supply
readout/segmentation evidence; AnyMorph/RoboToken supply variable-interface
evidence. The join, ordering robustness, visual bottleneck, and scale transfer
are project risks with explicit ablations.

All learned tensors start random. There are no Pythia/language weights,
pretrained visual encoders, pretrained action codebooks, or frozen learned
components. Maintained PyTorch, Hugging Face, standard distributed training,
and standard artifact formats own commodity infrastructure; project code owns
only adapters/packing/readouts and the scientific world process.

The exact contract and risk ledger are in
[MODEL-AND-REPRESENTATION.md](MODEL-AND-REPRESENTATION.md).

## 4. Representation

The universal envelope is:

```text
(role, encounter, episode, event_time, local public key,
 public schema attributes, normalized payload, validity mask).
```

Initial roles include:

```text
ENCOUNTER_SCHEMA   BOUNDARY        CONDITION
OBSERVATION        ACTION_QUERY    ACTION_EXECUTED
OUTCOME_QUERY      FEEDBACK
```

Important consequences:

- a world has no atomic hidden ID; it is represented through public schema,
  task evidence, boundaries, and observed trajectory;
- goals and outcome examples use the same modality adapters as observations;
- physical prompts are complete preceding trajectories, not text surrogates;
- local channel/actuator identities are encounter-local and jointly permuted;
- one query per active scalar actuator permits variable action cardinality;
- the shared head predicts a masked 16-step continuous receding-horizon chunk;
- executed actions, not intended or teacher-only actions, enter later context;
  and
- language can later be one optional modality but is never the core boundary.

Numeric channels initially use one token per public scalar. Images use a random
shallow patch stem and eight resampled tokens per view/frame. Public morphology
or relation tokens are included only when the encounter genuinely exposes
them.

## 5. First action at `S0`

Select one bounded session on:

```text
W_calibrated_monomial[d=1..4]

x_(t+1) = clip(x_t + B*u_t, -1, 1)
B       = P*diag(b_1,...,b_d).
```

`P` and the nonzero signed gains change by process instance and remain hidden.
The public transcript provides one safe calibration pulse per actuator, in
randomized order with reset boundaries. Those `d` transitions exactly reveal
`B` from public information. A separately sampled public goal then requires
bounded multi-step closed-loop control.

This family is estimated to be learnable from blank weights because it is
fully observed, deterministic, reversible, low-dimensional, and densely
supervised. It is not degenerate: identical current states/goals require
different actions under different calibration histories. Unlike the discarded
scalar-only choice, it exercises variable cardinality, sensor-actuator binding,
nontrivial relabeling, shared readouts, and multi-coordinate composition.

The session is a static/offline inner loop unless a later action gives the
learner ownership of calibration. It creates `C1`; nothing preselects the
action at `S1`. Full semantics, admission gates, candidate comparison, and
reversal conditions are in [C0-FIRST-ACTION.md](C0-FIRST-ACTION.md).

## 6. Performance and plausibility are vectors

World performance retains action error, forward-prediction error, closed-loop
success/error/cost, context-ablation and mismatch gaps, relabeling/order gaps,
held-out structure, oracle gap, confidence, and complete cost. One scalar
cannot fully map even the scalar support, much less the `d=1..4` family.

Candidate-world plausibility is ranked only after hard validity gates and keeps
blank learnability, GEN-dependency alignment, transfer fan-out, interface
coverage, anti-shortcut quality, diagnosticity, ontology neutrality, target
density, full inner/outer cost, and uncertainty separate.

A structured world may emit online score/efficiency telemetry during training
and trigger a predeclared early reconsideration. Such a scalar is scheduler
control-plane data, not automatically learner feedback and not automatically
comparable across worlds. STEP 1's SDL remains only a candidate learning-curve
summary for investigation.

## 7. Compute and scale decision

### Primary experimentation: two T4 GPUs

All ordinary architecture, tokenizer, world-validity, transfer, and scheduler
experiments target the repository's two-T4 Kaggle path. The 21.24M core and
2,048-token context are unmeasured feasibility estimates until an authorized
preflight. Use FP16 on T4, DDP replicas, and standard gradient accumulation;
two T4s do not combine into one larger memory pool.

Logical sessions and checkpoints should be batched inside an allocation when
possible. The repository's prior observed setup/build/test path costs roughly
13–16 minutes before useful work, so launching one Kaggle job per tiny world or
decision would let outer overhead dominate.

### Possible final scale: Kaggle TPU

The user permits a later scale-up for a final run. Because the TPU queue is
reported to exceed two hours, it is not the experimentation loop. The token
ABI, architecture topology, world versions, metric definitions, and scheduler
policy must be frozen first; then one allocation should contain the logical
sessions, online monitors, checkpoints, and autonomous predeclared decisions.

Continuing the same-size model preserves the T4 checkpoint lineage. A wider or
deeper TPU model starts a new truly blank confirmation lineage unless a
separate weight-growth method is validated. The small-model protocol can be
replayed, but its world rankings and stop thresholds may not transfer with
scale. A width-768 ICRT/Octo-Base-like profile and an 8,192-token context are
reference candidates, not decisions or fit claims.

The current `EXPERIMENT-EXECUTION-PLAN.md` authorizes only the T4 implementation
path; a TPU runner requires a later explicit plan amendment and run
authorization.

## 8. Overhead rule

Static dataset training is the simplest inner loop. Compare:

```text
R_plain  = ordinary prepacked static training
R_packed = exact selected-world transcripts pre-generated and packed
R_live   = actual online/generated world pipeline.
```

Use token/FLOP accounting, Roofline, communication overlap, producer pipeline,
and work/span to identify the inner critical path. Charge checkpointing,
evaluation, disposable branches, world admission, switching, launch/queue,
and decision latency to the outer loop.

The process earns its complexity only if it beats a matched fixed multi-world
pretraining action on:

```text
end-capability utility gained / total attributable wall-clock and resources.
```

The formal model is in [OVERHEAD-MODEL.md](OVERHEAD-MODEL.md).

## 9. Decided versus still empirical

Decided for this starting proposal:

- GEN-1.5-like physical-prompt generalization is the end target and hard
  architecture gate;
- the developmental process is recursive at every checkpoint;
- the model is candidly custom, with an ICRT causal core and a narrow
  variable-interface/readout bridge;
- strict random initialization and no mandatory language/action vocabulary;
- the first action is the `d=1..4` calibrated monomial-effect family;
- source performance and world plausibility remain vectors;
- online world-native scoring is an optional monitor, not an adopted objective;
- primary experiments use two T4s; TPU is a frozen final-run option; and
- adaptive scheduling must beat a matched static-mixture baseline after full
  overhead.

Still empirical or pending authorization:

- exact two-T4 memory, batch, step time, and useful target throughput;
- whether schema-conditioned variable readouts at least match a paper-faithful
  fixed-vector ICRT control on matched fixed-cardinality slices;
- whether 2,048 tokens and the eight-slot visual resampler preserve enough
  context/detail;
- source-session budget and stopping thresholds;
- realized context use and transfer from `C1`;
- action `a1` selected from the actual `S1`;
- whether any larger TPU profile is warranted and feasible; and
- whether recursive scheduling beats fixed pretraining at equal total cost.
