# Step 1: A Minimal Generated World and Dense-Teacher Baseline

## Implementation specification

### Status

This document defines the first implementation boundary of the project. It
turns the commitments in [PROCESS-AND-RENDERING.md](PROCESS-AND-RENDERING.md),
[WORLD-BACKEND.md](WORLD-BACKEND.md), and
[TRAINING-DYNAMICS.md](TRAINING-DYNAMICS.md) into one executable experiment.

> **Post-experiment status, 2026-08-17:** STEP 1 is complete as an internal
> stepping stone. The combined evidence and the decision to stop further
> STEP 1 ablations are recorded in
> [STEP-1-SYNTHESIS.md](STEP-1-SYNTHESIS.md). Unrun cells in the original matrix
> are future options, not closure requirements, unless their outcomes would
> change the design of a later step.

Step 1 is not a general world engine, a universal process algebra, or a claim
that one synthetic family contains the foundations of intelligence. It is the
smallest implementation that can test whether dense, privileged supervision
from an executable world teaches a process distinction more efficiently and
more transferably than outcome-only verified learning from the same world.

The first implementation decision is:

> Build a generated family of typed, finite, partially observable transition
> systems in Rust. Generate teacher-conditioned token sequences as the primary
> training path. Retain a batched interactive interface so that the identical
> worlds can support evaluation, learner-conditioned traces, and an RLVR
> baseline.

---

## 1. The question Step 1 must answer

The central experimental question is:

> Does dense supervision extracted from executable worlds produce more
> transferable capability per unit of training or interaction than
> outcome-only verification on those same worlds?

The first process distinction is **evidence acquisition under irreversible
commitment**. The learner must gather observations while maintaining multiple
live alternatives and must distinguish an action that preserves future options
from one that removes them.

Success cannot mean only that a model solves familiar rendered instances. The
model must also respond to the same process under a different rendering and
respond differently when the consequence of commitment is changed.

The intended claim is deliberately narrow. Step 1 may provide evidence that a
dense teacher helps a learner acquire this process organization. It cannot by
itself establish broad intelligence, real-world transfer, or a final ontology
for later worlds.

---

## 2. Why the first world is not ad hoc

A handcrafted puzzle with randomized names is insufficient. The implementation
must instantiate a declared family

\[
\mathcal F =
(\Theta, P_\Theta, S, A, T_\theta, \Omega_\theta, G_\theta,
V_\theta, R),
\]

where:

- \(\Theta\) is the space of instance parameters;
- \(P_\Theta\) is a reproducible distribution over instances;
- \(S\) is persistent process state;
- \(A\) is a set of typed learner interventions;
- \(T_\theta\) is the action-dependent transition rule;
- \(\Omega_\theta\) selects the learner-visible observation;
- \(G_\theta\) defines continuation and termination;
- \(V_\theta\) is a privileged verifier and teacher-query interface; and
- \(R\) renders typed observations and actions as tokens.

The family is admitted because it has all of the following:

1. **A standard semantic form.** It is a finite, partially observable,
   action-dependent transition system, not a transcript generator disguised
   as an environment.
2. **Structural generation.** Instances vary in consequential structure, not
   merely in labels.
3. **A process contrast.** Irreversible and reversible variants differ in
   reachable futures while retaining closely matched surfaces.
4. **A representation contrast.** Two renderings express the same typed
   process objects.
5. **A correspondence claim.** The selected organization occurs in bounded
   forms of diagnosis and debugging: acquire evidence while keeping
   explanations live, then make or revise a commitment.
6. **A transfer consequence.** Training should reduce the experience required
   to act correctly under a held-out rendering and held-out structures.

This does not make every instance useful. It makes the source of variation,
the claimed invariant, and the falsifying controls explicit.

---

## 3. The first world family

### 3.1 Instance parameters

An instance \(\theta\) contains:

- a finite set of hypotheses \(H\);
- one latent true hypothesis \(h^* \in H\);
- a finite set of available probes \(Q\);
- a finite observation alphabet \(E\);
- an evidence table or distribution
  \(D(e \mid q,h)\);
- a cost for each probe;
- a step or cost budget;
- a commitment-consequence mode;
- optional recovery conditions; and
- an independently sampled rendering configuration.

The initial generator should use deterministic evidence tables. Stochastic
evidence may be added only after deterministic semantics, teacher targets, and
replay are verified.

### 3.2 Generation constraints

The sampler must reject or construct around degenerate instances. Every
accepted acquisition instance must satisfy:

- at least two hypotheses are initially observationally possible;
- at least two learner actions are available at some nonterminal history;
- no learner-visible identifier directly encodes \(h^*\);
- at least one probe changes the set of hypotheses consistent with the
  history;
- the true hypothesis can be identified within the budget;
- at least one generated decision depends on previous evidence rather than
  only on the current surface string; and
- the declared commitment consequence changes the set of reachable future
  histories.

Difficulty parameters may control:

- number of hypotheses and probes;
- depth of the shortest discriminating probe policy;
- overlap between evidence patterns;
- probe costs;
- redundant or dominated probes;
- ordering dependencies;
- availability of recovery; and
- budget slack.

Train, validation, and test partitions must hold out structural parameter
combinations, not only random seeds.

### 3.3 Persistent state

The minimal runtime state is:

```text
instance identity and seed
latent true hypothesis
probe history and returned evidence
currently available probes
remaining budget and accumulated cost
current or final commitment, if any
commitment/recovery status
step index
continuation or terminal status
```

The latent hypothesis and any derived set of consistent hypotheses are
privileged state. They must not automatically be rendered to the learner.

### 3.4 Typed actions

Step 1 requires only:

```text
Inspect(probe_id)
Commit(hypothesis_id)
```

`Recover` may be introduced only in generated variants where recovery has a
defined persistent cost and changes the reachable state. Malformed expressions
are parser outcomes, not additional semantic actions.

Each action schema defines:

- its typed arguments;
- a validity predicate;
- its state transition;
- its learner-visible consequence;
- its cost;
- its effect on continuation; and
- the privileged facts made available to the verifier.

### 3.5 Transition requirements

`Inspect(q)` must return evidence determined by the instance and current state,
charge its declared cost, and update persistent history. It may alter later
availability only when that alteration is an explicit generated parameter.

In the **irreversible process**, `Commit(h)` removes the unchosen alternatives
and normally terminates the episode. A wrong commitment cannot be repaired
unless the instance explicitly includes a costly recovery transition.

In the **reversible control**, the surface-corresponding action records a
provisional selection but leaves inspection and later revision reachable. The
action vocabulary and rendering should remain as similar as practical. The
changed reachable futures, not a special label announcing reversibility, carry
the semantic contrast.

Forced-prefix and learner-conditioned evaluations may place the learner after
an earlier incorrect selection. Correct continuation then differs between the
two process variants: recovery is possible in the reversible control and
impossible or explicitly costly in the irreversible process. This makes the
contrast observable without inventing a reward for committing prematurely.

### 3.6 Observation boundary

Learner-visible observations may contain only:

- evidence actually returned by completed probes;
- currently expressible actions or enough information to infer them;
- public cost or budget information when declared by the instance;
- public consequences of the learner's previous action; and
- continuation or terminal status.

The generator and teacher may additionally query:

- the latent true hypothesis;
- all counterfactual probe results;
- the hypotheses consistent with the complete history;
- valid actions and their successor states;
- whether a commitment is correct;
- minimal remaining cost;
- recoverability; and
- all acceptable next actions under the declared teacher objective.

Privileged variables must be tagged by use: verifier-only, teacher-target,
temporary scaffold, or never serialized. Designer state is not automatically a
learner target.

---

## 4. Presentation and rendering controls

The process, presentation, and token rendering must remain separate.

### Rendering A: canonical symbolic form

The home rendering should be regular and easy to audit, for example:

```text
SEEN probe_2 => mark_blue
BUDGET 3
AVAILABLE inspect(probe_1), inspect(probe_4), commit(cause_1), commit(cause_3)
ACTION inspect(probe_4)
```

### Rendering B: aligned alternate form

The alternate rendering expresses the same typed objects with different
lexicon, ordering, and layout, for example:

```text
The second check returned blue. Three units remain.
Possible moves: examine K; examine R; settle on M; settle on T.
> examine R
```

Rendering B is not allowed to add or remove information. Canonical expressions
from both renderers must parse into the same typed objects, and identical typed
actions in identical states must cause identical transitions.

Object renaming, list ordering, whitespace, trace length, and token frequency
must be audited for target leakage. The changed-process control must not be
identifiable from an explicit mode word or a renderer-specific marker.

Representation transfer should be measured after a small, separately reported
amount of Rendering B interface calibration. A zero-shot result may also be
reported, but it must not be confused with process transfer when the new action
vocabulary has never been grounded.

---

## 5. The privileged dense teacher

### 5.1 Teacher objective

The first teacher policy is lexicographic:

1. avoid an incorrect irreversible commitment;
2. reach a correct terminal commitment when possible;
3. remain within the budget;
4. minimize declared probe and recovery cost; and
5. preserve all equally optimal actions rather than fabricating a unique trace.

The teacher must return a distribution or set when several next actions are
equally legitimate.

### 5.2 Target channels

Generated state may support local targets for:

- well-formed action tokens;
- the set or distribution of valid next actions;
- the teacher-preferred next-action distribution;
- predicted public consequence of a proposed action;
- predicted next observation where it is determined;
- whether current evidence licenses commitment;
- correction after a malformed or strategically poor action;
- recovery when recovery is reachable; and
- stopping or commitment.

Step 1 must record a loss mask and target-channel identifier for every directly
supervised token span. A target must not narrate a hidden variable merely
because the generator can access it.

### 5.3 Generation regimes

The primary regime is **teacher-conditioned generation**. It produces complete
token sequences and target masks without consulting current learner weights,
so generation can be parallelized, cached, or performed ahead of training.

A secondary **learner-conditioned regime** executes the learner's actual action
and asks the teacher for local correction or continuation targets from the
resulting state. This is required for selected recovery experiments, not for
all Step 1 tokens.

Dense teacher supervision remains cross-entropy training. Interaction with an
action-dependent world does not make it policy-gradient learning.

---

## 6. The role of RLVR

RLVR is a baseline, not the organizing training method.

The same privileged verifier must be able to compute an outcome from a
trajectory and final world state, including:

- valid or invalid termination;
- correct or incorrect commitment;
- total probe and recovery cost;
- budget violations; and
- whether irreversible damage made success unreachable.

The primary comparison is:

| Condition | World access | Learning signal |
|---|---|---|
| Dense teacher | Same generated family | Local token targets from privileged state |
| Outcome-only RLVR | Same generated family | Verified trajectory outcome only |
| Learner-conditioned dense teacher | Same interactive executor | Local correction and continuation targets |
| Hybrid, optional | Same interactive executor | Dense teaching followed by outcome optimization |

RLVR must not receive privileged intermediate labels that are withheld from the
outcome-only condition. Dense teaching and RLVR should share initial-state
distributions, held-out structures, renderings, model initialization policy,
and evaluator.

No single budget makes the comparison completely fair. Results must therefore
report at least:

- model training FLOPs or an explicit proxy;
- generated world transitions;
- learner-decoded action tokens;
- total training tokens;
- wall-clock time;
- peak accelerator and CPU resources; and
- success or transfer as a function of each relevant budget.

The comparison asks whether dense traces help. It does not assume that RLVR is
the strongest eventual use of the world.

---

## 7. Implementation architecture

### 7.1 Language and boundary

The executor, sampler, verifier, canonical parser, renderers, and high-throughput
teacher-data path will be implemented in **Rust**.

Python will provide experiment configuration, training-loop integration,
analysis, and plotting. Python must not be called once per world action in the
high-throughput path. The Rust/Python interface must accept and return batches.

The initial executor is CPU-based. A GPU executor is deferred until profiling
shows that world execution or rendering, rather than model training or data
movement, is a material bottleneck and that the workload can be advanced in
large regular batches.

### 7.2 Logical components

```text
WorldFamily specification
        |
        v
instance sampler -----> structural validator
        |
        v
compiled instance
        |
        +------> batched executor <------ typed learner actions
        |              |
        |              +------> verifier / teacher queries
        |              |
        |              +------> typed observations
        |                              |
        v                              v
teacher policy                    renderer/parser
        |                              |
        +--------------+---------------+
                       v
              packed token IDs + masks
                       |
             +---------+----------+
             v                    v
       dataset shards       online batch API
```

The offline teacher path and interactive RLVR path must use the same transition
and verifier implementation.

### 7.3 Proposed repository layout

```text
step1/
  Cargo.toml
  crates/
    world-core/       typed state, actions, transitions, replay
    world-generate/   instance sampling and structural validation
    world-teacher/    teacher queries, policies, target construction
    world-render/     renderers, parsers, token-fragment compilation
    world-data/       packing, sharding, manifests
    world-python/     batched Python bindings
  python/
    train/
    eval/
    analysis/
  tests/
    fixtures/
  benchmarks/
```

Crate boundaries may be collapsed if they create ceremony before independent
responsibilities exist. The semantic boundaries above must remain testable even
if the first code layout is smaller.

### 7.4 Data representation

The hot path should use:

- integer identifiers and compact enums;
- bitsets for small hypothesis and availability sets;
- immutable compiled instance tables;
- structure-of-arrays batches where profiling supports them;
- pre-tokenized constant rendering fragments;
- deterministic per-instance and per-trajectory seeds;
- packed token-ID, loss-mask, and target-channel buffers; and
- bounded producer/consumer queues for training.

JSON may be used for human-readable manifests and debugging fixtures. It must
not be the per-step runtime representation.

The canonical debug path may initially render UTF-8 strings. The production
data path must be able to assemble token IDs without allocating and tokenizing
a complete string for every observation.

---

## 8. Throughput contract

“Approach raw text” is an engineering requirement, not an adjective.

The benchmark baseline is the project's ordinary text input pipeline on the
same machine, using the same tokenizer, sequence length, packing policy,
storage medium, and consumer batch shape.

Measure separately:

1. sampled and validated instances per second;
2. world transitions plus teacher queries per second;
3. rendered and packed target tokens per second;
4. sustained tokens delivered to a mock trainer;
5. sustained tokens delivered during actual training; and
6. learner-conditioned steps per second, reported separately from offline
   teacher generation.

The Step 1 performance gates are:

- the teacher-conditioned generator and packer sustain at least 80% of the
  measured raw-text pipeline throughput on the allocated CPU resources;
- with asynchronous generation, they sustain at least twice the target
  trainer's token consumption rate or demonstrably keep its input queue
  nonempty for a representative run;
- generation and packing add no mandatory GPU work to the offline path;
- batch generation is materially faster than crossing the Python boundary per
  episode; and
- all benchmark results include hardware, worker count, batch size, sequence
  length, rendering, and whether tokens were cached.

> **Post-measurement amendment, 2026-08-17:** The 80% ratio was an advance
> engineering estimate, not a threshold derived from the training system or the
> scientific hypothesis. The measured world-to-raw ratio was about 66%, while
> the packed-shard and DataLoader paths were sufficient to complete the actual
> two-T4 experiments. The 80% item is therefore non-blocking. Retain and report
> the measurement; optimize or reprofile only if a later run shows actual input
> starvation.

If online generation cannot meet the trainer demand but deterministic offline
generation can, packed shards are an acceptable Step 1 training path. The
interactive executor remains required for evaluation and the RLVR baseline.

No runtime language model may generate surface text inside the world pipeline.
Surface variation must come from controlled compositional renderers whose
semantics are auditable.

---

## 9. Reproducibility and auditability

Every episode must be recoverable from:

```text
world-family version
generator version
root seed
instance index
process-variant identifier
rendering identifier
teacher-policy version
tokenizer identity and hash
```

A replay record must allow an auditor to reconstruct all states, actions,
observations, privileged queries, targets, and verifier outputs. Dataset shards
must carry a manifest containing generation parameters and content hashes.

Changing rendering must not change the sampled semantic instance. Changing the
teacher policy must not change world transitions. Changing the training method
must not silently change the world distribution.

---

## 10. Required correctness tests

Before model training, the implementation must pass:

### Semantic tests

- deterministic replay from seed and action sequence;
- action validity and transition invariants;
- budget and cost conservation;
- correct terminal and continuation behavior;
- irreversible commitment removes declared futures;
- reversible control retains declared futures;
- teacher and verifier agree with executed terminal state; and
- generated acquisition instances satisfy the structural constraints.

### Representation tests

- canonical render/parse round trips;
- Rendering A and B map aligned histories to identical typed objects;
- aligned actions cause identical state transitions;
- semantic-control surfaces contain no explicit process-mode leak;
- hidden identifiers, ordering, length, and token frequency do not predict the
  target above declared tolerances; and
- token-ID assembly decodes to the canonical debug rendering where exact
  equivalence is intended.

### Teacher tests

- every supervised action is valid in its state;
- every terminal teacher commitment is correct when success is reachable;
- all equally optimal actions can be represented;
- impossible or underdetermined predictions are not assigned false point
  targets; and
- target masks never expose verifier-only fields accidentally.

### Property and differential tests

- randomized action histories preserve invariants;
- optimized batch execution agrees with a simple reference execution;
- offline and interactive execution agree for identical instances and actions;
  and
- outcome-only verifier scores agree across renderings.

---

## 11. First experimental matrix

The minimum evaluation matrix is:

| | Rendering A | Rendering B |
|---|---|---|
| Irreversible process | acquisition and in-family generalization | representation transfer |
| Reversible control | semantic discrimination | optional joint transfer control |

For each training condition, evaluate:

- unseen seeds within trained parameter combinations;
- held-out structural combinations;
- larger or deeper instances within a declared extrapolation range;
- Rendering B after a measured interface-calibration budget;
- reversible versus irreversible behavior under matched surfaces;
- forced-prefix recovery states;
- malformed-action handling; and
- success, cost, calibration, and regret relative to the privileged teacher.

At minimum compare:

1. dense teacher supervision;
2. outcome-only RLVR;
3. a surface or target-shuffled control; and
4. an untrained or ordinary-text-matched baseline appropriate to the model.

A result supports process learning only if behavior follows the process
consequence across rendering changes and changes appropriately in the semantic
control. High accuracy on Rendering A alone is insufficient.

---

## 12. Deliverables

Step 1 produces:

1. a versioned formal schema for the first world family;
2. a deterministic Rust instance sampler and structural validator;
3. a reference and batched executor;
4. irreversible and reversible process variants;
5. two aligned renderers and parsers;
6. a privileged verifier and dense teacher;
7. packed token-ID and target-mask generation;
8. deterministic dataset shard generation;
9. a batched Python interface for evaluation and RLVR;
10. correctness, leakage, and differential tests;
11. a raw-text-relative throughput benchmark; and
12. scripts and manifests for the first experimental matrix.

---

## 13. Explicit non-goals

Step 1 will not build:

- a universal world-description language;
- a large inventory of named cognitive primitives;
- natural-language realism generated by another model;
- unrestricted procedural programs;
- a curriculum-learning controller;
- a GPU simulator without a measured need;
- many unrelated task families;
- a general RL platform; or
- claims of transfer to real diagnosis or debugging before such transfer is
  measured.

Reusable abstractions are extracted after repeated implementations demonstrate
the same semantic need. They are not installed in advance merely because their
names sound general.

---

## 14. Completion criteria

Step 1 is implementation-complete when:

1. an independent reader can determine the state, actions, observations,
   transitions, privileged information, and verifier from the specification;
2. the generator produces structurally varied, valid, replayable instances;
3. learner actions determine subsequent states and observations;
4. both renderings are consequence-invariant;
5. the reversible control changes reachable futures without an explicit
   surface leak;
6. teacher targets are locally valid and do not expose undeclared privileged
   state;
7. dense-teacher and outcome-only RLVR conditions use the same executable
   worlds and evaluator;
8. the teacher-conditioned data path meets or has a documented result against
   the throughput gates;
9. the required correctness and leakage tests pass; and
10. the experimental matrix can be run from versioned configurations.

Step 1 is scientifically complete only when its results distinguish at least
one of the following conclusions:

- dense teacher traces improve process learning and transfer;
- they improve only surface acquisition;
- the process contrast is not learned;
- the verifier baseline is equally or more efficient under the measured
  budgets; or
- the world family or experiment is too weak to discriminate among these
  possibilities.

Any of these is an informative result. Building a larger generator before the
first experiment can make this distinction would not be progress on the stated
claim.

The target-shuffled control, rendering-transfer stage, and outcome-only stage
have now made the required distinction. Dense traces produced state-dependent
behavior beyond a syntax-matched chance control and demonstrated that faster
acquisition under another rendering is possible. Seed 0 is the strong witness;
seed 1 preserves the early sign at much smaller magnitude. STEP 1 therefore
meets this scientific stopping rule. The next activity is theory-building, not
an automatic second implementation stage. The reversible-trained contrast and
a grounded RLVR comparison remain interpretable future experiments, but neither
is required unless a later design decision depends on it.
