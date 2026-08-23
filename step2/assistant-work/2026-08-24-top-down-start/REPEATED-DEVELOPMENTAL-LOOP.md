# Repeated Developmental Loop and Its Cost

**Date:** 2026-08-24
**Status:** assistant-authored proposal; no scheduler implementation or training
is authorized

The canonical formal definition is
[RECURSIVE-CHECKPOINT-POLICY.md](RECURSIVE-CHECKPOINT-POLICY.md). This document
describes its initially proposed operational approximation.

## 1. The unit of optimization

The project is not a fixed pretraining phase followed by a one-time transfer
test. At every decision state `S_n`, it asks the same question again:

> What additional world experience, replay, world admission, or evaluation has
> the highest expected recursive contribution to the end capabilities from the
> learner's current state under the remaining total resource budget?

Here `S_n` contains the immutable checkpoint `C_n`, evidence and uncertainty,
the current capability hypothesis, admitted worlds and cost models, the
lineage ledger, and remaining resources. This matters because evaluation can
change the next decision without changing model weights.

That creates two coupled optimization problems:

1. **Inner-loop efficiency:** how quickly and densely the learner changes while
   training inside a selected world.
2. **Outer-loop efficiency:** how much time is spent estimating which world
   should be selected next, switching worlds, evaluating branches, and
   maintaining evidence.

Both costs must be charged from the first checkpoint. Ignoring the second would
make an elaborate adaptive curriculum appear artificially better than ordinary
pretraining.

## 2. Recursive structure

```text
S_n = checkpoint + evidence + graph + worlds + ledger + resources
  |
  +-- capability-local world frontiers
  +-- replay/sentinel candidates
  +-- cross-trunk candidates
  +-- static multi-world candidates
  +-- decision-relevant evaluations
  +-- world-admission candidates
  |
  +-- choose bounded action
  +-- observe checkpoint/evidence/cost result
  +-- construct S_(n+1)
  +-- apply the same policy again
```

Each capability level exposes a local copy of the same decision problem; it is
not a one-way training phase. Evidence can promote descendants, return an old
world, switch trunks, revise the decomposition, or admit a new world. The graph
is a forest, so an unstarted perception, task-evidence, memory, or composition
trunk may outrank a control descendant. Old worlds remain available for replay
and retention measurement, but the scheduler should operate on a small active
frontier rather than repeatedly score every historical world.

A fixed or adaptive **multi-world static dataset mixture** is also an inner-loop
training action. It is not a separate category outside this model. In a large
pretraining regime it may be more efficient than frequent single-world
switching and must be included as a baseline/candidate.

## 3. Inner-loop world cost

For primary world `w` and session budget `b`, the resource-work ledger is:

```text
Work_inner(w, b) =
    N_w * c_train
  + A_w * c_decode
  + C_world(w)
  + C_targets(w)
  + C_sync(Q_w)
  + C_data_prepare(w)
```

where:

- `N_w` is the number of training tokens;
- `A_w` is the number of actions that must be decoded from current learner
  weights rather than supplied by an offline teacher;
- `Q_w` is the number of sequential learner-world boundaries;
- `C_world` and `C_targets` are transition and supervision costs; and
- `C_data_prepare` is non-amortized generation/packing cost.

Those terms can overlap and therefore must not simply be summed as wall-clock.
Actual `T_inner` is the resource-constrained critical path under the
work/span, Roofline, communication, and producer-pipeline model in
[OVERHEAD-MODEL.md](OVERHEAD-MODEL.md). The world-specific resource-work ratio
is still useful:

```text
interaction_tax(w) =
  (A_w*c_decode + C_sync(Q_w) + C_world + C_targets + C_data_prepare)
  / (N_w*c_train)
```

This makes the user's “ten tests” example explicit. If learning a new world
requires ten learner-chosen tests before every useful target, those ten
sequential boundaries are part of its cost. They are not free because they are
scientifically meaningful.

There is a crucial throughput distinction:

- Ten **teacher-generated** calibration transitions can be generated offline,
  packed into sequences, and trained in large batches. They add context tokens
  but not ten current-model synchronization barriers.
- Ten **learner-generated** probes require current weights, action decoding,
  world response, and another model call. They pay both decode and
  synchronization costs.

Therefore a world that is meant to teach inference from supplied evidence
should use the shortest sufficient offline transcript. Learner-owned probing is
selected when active information gathering is itself the capability worth its
additional cost. This is not a semantic compromise; it distinguishes two
different candidate actions and charges each correctly.

## 4. Outer-loop and scheduling cost

For checkpoint `C_n`, the additional cost is:

```text
C_outer(n) =
    C_checkpoint
  + C_session_sentinels
  + C_adaptation_branches
  + C_world_switch
  + C_scheduler_decision
  + amortized(C_world_admission)
```

The measured scheduling-overhead fraction is:

```text
rho_n = C_outer(n) / (C_inner(n) + C_outer(n))
```

Report both wall-clock `rho_time` and model-compute `rho_compute`; a cheap CPU
audit can consume calendar time without consuming GPU FLOPs, while a matched
adaptation branch consumes both.

The total developmental throughput is not tokens per second. It is:

```text
developmental_throughput =
  estimated increase in end-capability utility
  / (C_inner + C_outer)
```

This is the quantity against which an adaptive curriculum should be compared
with a fixed-mixture pretraining baseline.

## 5. End-capability utility and the recursive action objective

Let `U(S)` be a weighted summary of current evidence about the end-capability
probe families. It should include forward learning speed and retention, not
only zero-shot source scores.

For a candidate action `a`—train on a new/old/static world mixture, admit a new
world, or evaluate—the finite-budget value is recursive:

```text
V(S, r) = max {
  U(S),
  max_(a: cost(a)<=r) E[V(F(S,a), r-cost(a))]
}.
```

The exact continuation value is unknown. The initial controller approximates
it with short receding-horizon lookahead; estimated capability fan-out and
uncovered capability value approximate the uncomputed tail. A purely myopic
source-gain-per-cost score is not the governing definition.

For a training action on world `w` or declared world mixture, a practical
initial estimator is:

```text
score(w | S_n) =
  ( learning_progress_w
    * estimated_transfer_fanout_w
    - expected_forgetting_w
    + bounded_uncertainty_bonus_w
    + uncovered_end_capability_value_w )
  / (train_cost_w + switch_cost_w + mandatory_measurement_cost_w)
```

The terms have specific meanings:

- `learning_progress_w`: recent improvement per unit cost in the targeted
  property, not raw competence;
- `estimated_transfer_fanout_w`: the current estimate that progress in `w`
  helps immediate descendants of the capability graph;
- `expected_forgetting_w`: loss on compact parent sentinels;
- `bounded_uncertainty_bonus_w`: limited exploration of plausible but
  under-measured worlds; and
- `uncovered_end_capability_value_w`: credit for opening a missing capability
  trunk rather than greedily deepening the first one; and
- the denominator includes all costs caused by selecting `w`.

The transfer-fanout estimate begins as an explicit design prior derived from the
capability graph. It is replaced or revised by occasional matched checkpoint
branches. No source paper can supply this project's coefficients in advance.

## 6. Evaluation is an action with value-of-information

An evaluation does not directly improve the main checkpoint. Its value is the
expected improvement in the *next decision* that its result permits. In standard
decision terminology:

```text
VOI(e | S_n) =
  E_e[ max_a Q(S_after_e, a) ]
  - max_a Q(S_n, a)
```

Run evaluation `e` only when its expected value of information exceeds its
cost and the opportunity cost of delaying training. Operationally, an
evaluation is justified when at least one of these is true:

- the two leading world choices overlap within current uncertainty;
- a result can promote, reject, or redesign a world;
- a cheap sentinel signals context failure or forgetting;
- a new representation/model boundary is being considered; or
- a capability level is about to change.

“More measurements would be interesting” is not sufficient.

## 7. Why a full transfer suite cannot run after every checkpoint

Suppose there are `K` candidate probes. A positive-transfer test adapts both
`C_n` and `C0` for `b_probe` updates on each probe. Ignoring setup and rollout
costs, its model-update overhead relative to a primary session of `b_session`
updates is approximately:

```text
rho_branch ~= 2 * K * b_probe / b_session
```

With five probes and each short branch using only 10% of a normal session,
`rho_branch` is already `1.0`: the transfer audit costs another entire training
session. Running it at every checkpoint would halve useful training throughput
before world generation, rollout, or switching costs are counted.

The correct scalable design is therefore tiered.

## 8. Tiered measurement cadence

### Tier 0 — piggyback telemetry, every training batch

Use signals already produced by the selected session:

- action and next-observation loss by world slice;
- valid-channel-normalized error;
- source competence and learning slope;
- tokens, gradient-bearing targets, world transitions, learner decodes, and
  synchronization boundaries per second; and
- time spent in data generation, loading, forward/backward, checkpointing, and
  idle synchronization.

When computed from tensors and counters already on the path, this has low
incremental decision overhead and supplies within-world learning progress.

### Tier 0W — world-native online evaluation during training (open option)

A world is more structured than a plain dataset: it can know its goal,
transition semantics, legal actions, resource use, terminal conditions, and a
declared public-information teacher or verifier. It can therefore return a
training-time metric vector such as:

```text
z_w(t) = [success,
          regret or error to public teacher,
          goal progress,
          correction/probe/action efficiency,
          constraint violations,
          predictive calibration,
          gain per token/update/transition/second].
```

A declared reducer can produce a scalar with uncertainty for stopping or
world-selection purposes. The components must remain logged so the scalar is
not treated as self-interpreting. This monitor can be computed on the training
trajectory, on a small held-out instance stream interleaved with training, or
asynchronously when weight staleness is bounded and recorded.

The practical possibility is a feedback-controlled training session:

```text
train up to hard budget
    while world monitor says marginal progress remains worthwhile
    and no validity/efficiency stop rule fires
-> emit checkpoint immediately when reconsideration is warranted
-> run the same recursive scheduler
```

This can replace some rigid `train -> stop -> evaluate` barriers. It does not
make evaluation free: extra rollouts, metric kernels, reductions,
synchronization, and stalls are inner-loop work. Nor does it make a
training-stream score evidence of held-out transfer. Tier 1–3 measurements
remain available for claims that online world evidence cannot identify.

STEP 1's **SDL** metric is retained as a research note here. STEP 1 used it
after declared calibration to localize acquisition in training time, so an
appropriately preserved version might summarize a world learning curve or
trigger session reconsideration. This is **not** a decision to adopt SDL or
change STEP 2's direction. Before reuse, its exact definition, calibration,
normalization across worlds, uncertainty, and relation to the capability claim
would need to be specified. See
[THEORY-PHASE.md](../../../THEORY-PHASE.md).

### Tier 1 — compact sentinels, every checkpoint

Run a small, fixed-cost set consisting of:

- current-world held-out instances and anti-shortcut interventions;
- one compact parent/retention sentinel when a parent exists; and
- at most one or two immediate frontier zero-shot probes chosen by uncertainty
  or expected decision relevance.

These are forward/rollout tests only. They do not train disposable branches and
therefore answer the repeated checkpoint question cheaply.

### Tier 2 — matched adaptation branches, only for a live decision

Train short disposable branches from only the comparator checkpoints needed by
the live claim. Use `C_n` and `C0` for positive-transfer claims; use matched
alternative lineages when deciding whether a particular intervening action was
worth taking. Limit this to the one or two candidates whose result can change
the next action. Stop early when confidence intervals or predeclared bounds
make the decision unchangeable. Record the full cost in `C_outer`.

### Tier 3 — full fan-out audit, at milestones

Run the broad transfer and retention suite when:

- closing or changing a capability level;
- changing tokenization, context semantics, or architecture;
- promoting any checkpoint as a reusable foundation for multiple lineages; or
- a sequence of cheap signals and sparse branches no longer predicts later
  outcomes.

The full suite is amortized over many sessions, not attached to each one.

No fixed percentage is “ideal” before timing and transfer slopes are measured.
The first sessions should nevertheless declare an outer-overhead budget and
report actual `rho`. If a requested audit would exceed it, the scheduler must
either show enough value-of-information to justify the exception or defer it to
a milestone batch.

## 9. Initial scheduler: standard, cheap, and corrigible

Do not begin by training a bespoke neural curriculum controller. There will be
too few checkpoint decisions to fit it, and its experiments would add the very
overhead being controlled.

Use a human-reviewed **cost-normalized learning-progress frontier policy**:

1. Restrict candidates to the current world, selected parent replay, immediate
   capability-graph children, a small set of unstarted high-priority trunks,
   and a broad static-mixture baseline.
2. Estimate per-world progress from Tier-0/Tier-1 data.
3. Weight progress by the present transfer-fanout evidence.
4. Divide by measured training, interaction, switch, and mandatory-evaluation
   cost.
5. Apply a small uncertainty bonus so an untested plausible child is not
   permanently ignored.
6. Select the highest index unless an evaluation has higher value-of-information.
7. Record the prediction and compare it with the next checkpoint's result.

This is a cost-aware variant of established automated-curriculum ideas rather
than a new scheduler architecture. Automated Curriculum Learning and
Teacher-Student Curriculum Learning use learning progress as a task-selection
signal; Prioritized Level Replay prioritizes estimated learning potential;
AdA schedules tasks near the capability frontier. The project adds the necessary
end-capability transfer weighting and explicit cost denominator because its
goal is not merely faster source-task mastery.

Once enough checkpoint decisions exist, the same logged table can support a
standard nonstationary/contextual bandit or calibrated model-based selector.
That is a later substitution, not a requirement for initial sessions.

## 10. World lifecycle keeps scheduling scalable

Every world version has one of four statuses:

1. **candidate** — semantic/information-boundary review and static generator
   audit; no scheduler arm yet;
2. **frontier** — eligible for primary training because it targets an immediate
   capability child;
3. **replay/sentinel** — retained in a compact mixture or cheap evaluation, but
   not exhaustively reconsidered each checkpoint; or
4. **archived** — provenance retained; excluded from routine scheduling unless
   a later result reopens it.

World-admission work—teacher realizability, leakage checks, oracle checks, and
generator validation—is paid once per version and amortized across all sessions
using that version. It should not be repeated merely because a new checkpoint
exists.

This makes each outer decision scale with the active frontier, approximately
`O(|W_frontier|)`, rather than all worlds ever created.

## 11. First application and all later applications

The action currently selected at `S0` uses the
`W_calibrated_monomial[d=1..4]` family. Its task-independent calibration prefix
contains one safe pulse for every active actuator, in randomized order with a
reset between pulses. The transcript is teacher-generated and can be produced
offline, so the `d` calibration transitions cost additional context
tokens—`O(d^2)` numeric payload tokens under the scalar-channel ABI—but do not
create a chain of learner-synchronized tests. The scalar member remains a unit
and sentinel slice; dimensions two through four make the checkpoint-producing
session exercise binding, variable cardinality, and nontrivial permutation.

After it produces `C1`, the scheduler applies exactly the same policy using the
new evidence and measured costs:

- continue the same support if source learning progress remains high;
- select another effect-graph support if source progress saturates and an
  adjacent sentinel is promising;
- leave the control trunk for an independent missing capability trunk or a
  broad world mixture when its cost-normalized end value is higher;
- replay if a compact parent sentinel regresses; or
- purchase a matched adaptation branch only if it can distinguish those
  choices.

Later, `REMOVE_SCAFFOLD` makes probing learner-owned. Its additional sequential
cost is then accepted only if the resulting active-identification capability
has better estimated end-capability gain per total cost than competing worlds.

The first action receives no guaranteed long-run sampling share. This is not a
special rule for `a0`: no selected world receives replay merely because it
appears in the lineage. At any state, a matched branch comparison is purchased
only when its value of information can change the recursive action. Broad
static mixtures remain ordinary candidates at every state and win whenever
their expected continuation value net of cost is higher.

## 12. Required ledger from checkpoint zero

Every session proposal and result should carry:

- source and destination checkpoint IDs;
- primary world/version/support and replay mixture;
- planned and actual tokens, updates, examples, and wall-clock;
- teacher-generated versus learner-generated action counts;
- sequential learner-world boundary count;
- data generation, training, checkpoint, evaluation, and switching time;
- Tier-0/Tier-1 progress and retention slopes;
- any Tier-2/Tier-3 branch cost and the decision it changed;
- predicted world-selection score before the session;
- realized evidence after the session; and
- cumulative `rho_time`, `rho_compute`, and developmental-throughput estimate.

Without this ledger the project could know that a curriculum eventually works
but not whether it beats the simpler fixed-pretraining alternative after its
own control overhead is included.

## 13. Primary sources

- [Automated Curriculum Learning for Neural Networks](https://proceedings.mlr.press/v70/graves17a.html)
- [Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183)
- [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)
- [Human-Timescale Adaptation in an Open-Ended Task Space](https://arxiv.org/abs/2301.07608)
- [Automatic Curriculum Graph Generation for Reinforcement Learning Agents](https://ojs.aaai.org/index.php/AAAI/article/view/10933)
- [Task Selection Policies for Multitask Learning](https://arxiv.org/abs/1907.06214)
- [Formal systems accounting used by this proposal](OVERHEAD-MODEL.md)
