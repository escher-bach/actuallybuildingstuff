# Recursive Checkpoint Policy

**Date:** 2026-08-24
**Status:** canonical process definition for this assistant-authored proposal;
no training is authorized

## 1. Correction: there is no privileged root world

The developmental object is a recursive decision process. The blank checkpoint
`C0` is merely its initial learner state. A world is not a root and is not a
permanent curriculum phase. Training on a world is one candidate transition
from the current state to a new checkpoint.

The same question is asked before **every** transition:

> Given the current checkpoint and current evidence, which bounded training or
> evaluation action has the greatest expected contribution to acquiring the
> end capabilities under the remaining resource budget?

Consequently:

- the first selected world has no guaranteed descendants;
- it receives no guaranteed replay share;
- it creates no universal world ontology;
- `C1` is not treated differently from `C27`; and
- a static multi-world pretraining run is one legal training action, not a
  process outside the recursion.

## 2. The recursive state is larger than the model checkpoint

The scheduler state at decision event `n` is

```text
S_n = (C_n, E_n, G_n, W_n, L_n, R_n)
```

where:

- `C_n` is the immutable model checkpoint;
- `E_n` is the evidence state: competence, learning curves, retention,
  uncertainties, and transfer observations;
- `G_n` is the current capability-dependency hypothesis and coverage state;
- `W_n` is the admitted world/action catalogue with measured cost models;
- `L_n` is the lineage ledger, including all prior actions and costs; and
- `R_n` is the remaining wall-clock, accelerator, interaction, storage, and
  evaluation budget.

This distinction is necessary because an evaluation may leave `C_n` unchanged
while changing `E_n` enough to alter the next choice. Therefore the recursion
is over scheduler states `S_n`, not over weights alone.

## 3. Actions and transitions

At any `S_n`, the admissible actions are:

```text
TRAIN_NEW(world or world-mixture, objective, session budget, replay mix)
TRAIN_OLD(world or world-mixture, objective, session budget)
TRAIN_STATIC_MIXTURE(predeclared materialized mixture, objective, budget)
EVALUATE(probe or disposable adaptation branch, budget)
ADMIT_NEW(world specification and validity work, budget)
MIGRATE_ARCHITECTURE(new profile, initialization/conversion rule, budget)
STOP
```

`TRAIN_NEW` and `TRAIN_OLD` have the same mechanics. The labels only record
whether the data distribution has appeared in the lineage. A static dataset is
the case in which the selected world-mixture is materialized before training
and contains no learner-conditioned online transitions.

A training action produces

```text
C_(n+1) = Train(C_n, action_n, random_seed_n)
S_(n+1) = UpdateState(S_n, C_(n+1), observations_n, costs_n).
```

An evaluation action normally produces

```text
C_(n+1) = C_n
S_(n+1) = UpdateEvidence(S_n, result_n, costs_n).
```

Disposable adaptation branches belong to evaluation: they inform the lineage
choice but are not silently promoted to the main lineage. World admission can
add new actions to `W_n`; this is how newly recognized capability gaps enter
the process while training is underway. Architecture migration is explicit
because changing depth, width, context semantics, tokenization, or heads can
break checkpoint identity and transfer claims. A larger model initialized from
blank begins a new confirmation lineage; it is not silently called `C_(n+1)`.

## 4. Recursive objective

Let `U_end(C, E)` summarize evidence about the declared end-capability vector,
including learning speed on unseen worlds and retention—not source score alone.
For remaining resource budget `r`, the ideal finite-budget policy is

```text
V(S, r) = max {
    U_end(S),
    max over admissible a with cost(a) <= r of
        Expected[ V(F(S, a, outcome), r - cost(a)) ]
}.
```

The selected action is

```text
pi*(S, r) = argmax_a Expected[V(F(S, a), r - cost(a))].
```

This is the formal reason the first world is not special: `a0` is simply
`pi*(S0, R0)`. After observing its outcome, `a1` is computed by the same rule
at the new state. A world that looks weak immediately can still be selected if
it opens high-value later actions; a world with fast source learning can be
rejected if its descendants are narrow or its total cost is high.

The exact value function is not known. The practical scheduler therefore uses
a short receding-horizon approximation:

```text
Q_h(S, a) = immediate evidence-weighted capability gain
          - total attributable cost penalty
          + Expected[max_a' Q_(h-1)(S', a')].
```

At small `h`, capability-graph fan-out and uncertainty are explicit estimates
of the uncomputed tail. They are revisable priors, not proof that a particular
world is foundational.

## 5. Capability levels are local recursive subproblems

A capability level is not a one-way phase. It defines a local target and an
eligible world set:

```text
level l:
    current evidence for capability l
    worlds that train l
    parent retention sentinels
    descendant and cross-trunk probes
    same NEW | OLD | STATIC MIXTURE | EVALUATE | ADMIT | MIGRATE choice
```

When evidence suggests the local capability is sufficient, descendant worlds
become stronger candidates. When retention fails, an old world may return.
When a different trunk has higher option value, the policy can switch trunks.
When the decomposition proves wrong, `G_n` is revised and new worlds can be
admitted. There is no irreversible “finish level, then advance” rule.

## 6. Evaluation is bought only for decision value

For an evaluation `e`, its value is the expected improvement in the recursive
choice, not the amount of information produced:

```text
VOI(e | S) = Expected[max_a Q(S_after_e, a)] - max_a Q(S, a).
```

Run `e` only if its expected value of information exceeds its complete cost and
the opportunity cost of delaying training. Cheap source/retention/frontier
sentinels can run frequently; matched transfer branches and broad audits run
only when they can change an actual action. This prevents ten tests from being
silently charged to every world transition.

### 6.1 Open possibility: world-native online evaluation

Evaluation need not always be a separate `EVALUATE` action after training. A
world has transition semantics, goals, costs, constraints, and often a
public-information teacher or oracle. It can therefore emit training-time
control-plane evidence alongside experience:

```text
z_(w,t) = Monitor_w(public trajectory prefix,
                    declared teacher/oracle outputs,
                    resource counters).
```

Possible components include success, regret to a public-information teacher,
goal-error reduction, correction efficiency, probe/action cost, constraint
violations, prediction calibration, and improvement per token, update, world
transition, or wall-clock second. A scheduler may reduce a declared vector of
such measurements to a scalar decision score with uncertainty, while retaining
the components for interpretation.

This turns a bounded training action into a monitored action:

```text
TRAIN(world mixture,
      hard resource ceiling,
      online monitor,
      predeclared stop/reconsider rule).
```

The monitor can update `E_n` during the session. If its rule fires, training
stops, the current weights are emitted as the next immutable checkpoint, and
the ordinary recursive choice resumes. Thus the design need not impose a rigid
`train full block -> stop -> run separate evaluation` barrier.

This is recorded as a **possibility, not an adopted metric or direction
change**. Several boundaries remain:

- a score used only by the scheduler is not automatically serialized as learner
  feedback;
- training loss on sampled targets is not held-out capability evidence;
- a world-local scalar is not automatically comparable across worlds;
- metric computation, reduction, additional rollout, and synchronization remain
  attributable inner-loop costs; and
- sparse independent probes are still needed when training-time measurements
  cannot distinguish memorization, leakage, shortcut use, or downstream
  transfer.

STEP 1's **SDL** statistic is a candidate item for this research queue because
it was used there to localize acquisition in training time after a declared
calibration budget. It could potentially summarize an online learning curve or
trigger reconsideration of a session. This document does not redefine SDL or
adopt it for STEP 2; its exact STEP 1 definition, calibration requirements, and
interpretive limits would need to be preserved and checked against each world's
semantics. See the [STEP 1 theory synthesis](../../../THEORY-PHASE.md).

## 7. Cost belongs inside the recursion

Every action cost includes the critical-path and resource costs it causes:

```text
cost(a) = inner training or inference cost
        + world generation / interaction cost
        + synchronization and communication cost
        + checkpoint and switching cost
        + mandatory measurement cost
        + amortized world-admission cost.
```

The detailed work/span, Roofline, communication, producer-pipeline, and
outer-cadence accounting is in [OVERHEAD-MODEL.md](OVERHEAD-MODEL.md).
The policy must be compared against long static-mixture actions under equal
total resources. If recursive scheduling does not increase end-capability gain
per total wall-clock or accelerator-hour after its own overhead, the simpler
action wins.

## 8. Initial implementation of the policy

Do not train a learned curriculum controller at the start. The initial
implementation is a logged, human-reviewed receding-horizon policy using:

- measured learning progress per total cost;
- capability coverage and estimated downstream option value;
- measured forgetting;
- a bounded uncertainty term;
- explicit value-of-information for evaluations; and
- an explicit static multi-world mixture candidate.

This uses established automated-curriculum, bandit, and metareasoning ideas
without making scheduler learning another prerequisite. Every prediction and
realized result is retained so the policy itself can later be calibrated or
replaced.

## 9. First application at the blank state

At `S0`, the current prior selects one bounded training action on
`W_calibrated_monomial[d=1..4]`. Each process has a variable-cardinality public
sensor/actuator interface, a hidden signed permutation and gains, one public
calibration transition per actuator, an independently sampled observable goal,
and dense action/next-observation targets. The scalar case remains one support
slice, not the whole training action.

Call this action `a0`, not a root or bootstrap:

```text
S0 -- a0: TRAIN_NEW(W_calibrated_monomial[d=1..4], bounded budget) --> S1
S1 -- pi(S1, R1) chooses again --> ...
```

It is currently preferred at `S0` because the blank-state estimates give it:

- high acquisition probability from random weights;
- a public, exactly realizable teacher;
- direct tests for temporal action-effect and context use;
- nontrivial tests of variable action queries, cardinality, and joint
  sensor/actuator relabeling;
- negligible world-generation and online-interaction span; and
- low commitment to language, geometry, object, or robot-specific semantics.

That does **not** preselect `a1`. At `S1`, continuing this family, moving to
coupled effects, switching to a perception/task/memory/probing world, replaying,
evaluating, selecting a broad static mixture, migrating, or rejecting this
family are all actions under the same rule.

## 10. What remains fixed and what remains revisable

The initial architecture and token interface are wider-scope priors because a
checkpoint lineage needs compatible parameters and serialized experience.
They are specified separately in
[MODEL-AND-REPRESENTATION.md](MODEL-AND-REPRESENTATION.md). They can still be
changed through an explicit migration action whose conversion, retraining,
and evaluation costs are charged to the recursion.

World choice, session length, replay share, evaluation cadence, capability
frontier, and the next action are never permanently fixed by `a0`.

## 11. Process invariant

After every decision event, the record must be sufficient to reconstruct:

```text
state before action
candidate actions considered
predicted gain, uncertainty, and full cost for each
selected action and budget
checkpoint/evidence state after action
realized cost and evidence
same recursive decision again
```

If a design document instead describes a mandatory world sequence beginning at
a privileged root, it is inconsistent with this process definition.
