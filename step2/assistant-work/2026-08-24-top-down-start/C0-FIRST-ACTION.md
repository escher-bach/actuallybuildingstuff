# First Application of the Recursive Policy at `C0`

**Date:** 2026-08-24
**Status:** Rust world/oracle vertical slice implemented; local CPU gates pass;
one two-T4 apparatus gate followed by a bounded candidate-checkpoint start is
authorized

## 1. Revised decision

Given the end-capability gate, architecture decision, validity obligations,
and a truly random checkpoint, select:

> `a0 = TRAIN_NEW(W_calibrated_monomial[d=1..4], one bounded monitored session)`

This is one procedural family of **calibrated signed-permutation effect
control**. Every process instance contains `d` public observation channels and
`d` public scalar actuators, where `d` ranges from one to four. The mapping from
actuators to observed effects changes by instance and is not serialized. A
short task-independent public calibration trajectory makes it exactly
recoverable, after which the learner must act toward a public observation-space
goal.

The scalar case is retained as a support slice, tokenizer unit test, and cheap
sentinel. It is no longer the sole checkpoint-producing world.

This is action `a0` under the same recursive policy used later. It has no root
status, guaranteed descendant, replay allocation, or claim on action `a1`.

## 2. Why the scalar-only choice was rejected

For

```text
x_(t+1) = x_t + b*u_t,
```

an unsaturated deterministic process is identified by one signed gain `b`.
One nonzero calibration pulse determines it exactly:

```text
b = (x_after - x_before) / u_pulse.
```

So the scalar world's *hidden transition law* is indeed fully summarized by
one scalar. That does not make model performance one-dimensional: action
error, goal success, context dependence, correction, and relabel robustness
are still distinct measurements. It does make the experience structurally
unrepresentative of the selected variable-cardinality token/readout interface.

A scalar-only session would fail to exercise:

- more than one schema-conditioned action query;
- binding between different sensor and actuator keys;
- variable cardinality and padding-free packing;
- independent composition of several inferred effects; and
- equivariance under a nontrivial channel permutation.

Because the architecture/token interface is a persistent bias, those omissions
outweigh the small throughput gain. The scalar slice remains useful inside a
family that also supplies the missing structure.

## 3. Formal world family

For each process instance, sample active dimension `d in {1,2,3,4}` and define

```text
x_(t+1) = clip(x_t + B*u_t, -1, 1)
B       = P * diag(b_1, ..., b_d).
```

Here:

- `x_t in [-1,1]^d` is the public observation;
- `u_t in [-u_max,u_max]^d` is the public action;
- `P` is a hidden permutation matrix;
- every signed gain satisfies
  `|b_j| in [b_min,b_max]`, with `0 < b_min < b_max`;
- `B` is a monomial matrix: every actuator affects exactly one observation
  channel and every observation channel has exactly one parent actuator;
- `g in [-1,1]^d` is the public desired observation; and
- the relevant costs are public command count and normalized action energy;
  there is no hidden reward or hidden cost term; and
- channel order, actuator order, gains, starts, goals, and calibration order
  vary by process instance.

The numeric constants are world-version parameters and must be fixed before
admission. Calibration resets use a declared neutral value `x_cal` and pulse
magnitudes satisfying, coordinatewise,

```text
|x_cal| + b_max*|u_pulse| <= 1 - calibration_margin.
```

This makes clipping impossible during calibration, so the public column
reconstruction below is exact up to declared numeric precision. Expert starts
and goals use a separate margin that prevents state clipping under the public
teacher. Arbitrary learner actions still use the declared clipping rule and
receive the resulting observation.

The generator must stratify rather than merely sample a convenient majority:
dimensions, gain signs, permutation classes, pulse signs/orders, required
action signs, one-step versus multi-step goals, and near-boundary cases receive
declared support. These factors are independently sampled except where a
declared safety constraint requires conditioning. Zero-action/terminal examples
cannot dominate the direct targets, and every evaluation stratum reports its
own count.

The family does not serialize `P`, `b`, `B`, a world-family ID, a generator
seed, or a canonical global meaning for any local channel key.

## 4. Public interface and transcript

At encounter start, the learner receives only:

- the number and encounter-local keys of active observation channels;
- the number and encounter-local keys of active actuators;
- public bounds, control period, reset behavior, and validity masks; and
- role/type information needed to distinguish observations, actions,
  conditions, and boundaries.

### 4.1 Calibration phase

For each active actuator, in randomized order:

1. reset to a declared neutral observation;
2. execute a known safe nonzero pulse on that actuator while others are zero;
3. expose the pre-observation, actual executed action, and post-observation;
4. reset before the next pulse.

The pulse sign and safe magnitude may vary but are public through the executed
action. Each transition reveals one column of `B`:

```text
B[:,j] = (x_after - x_before) / u_pulse[j].
```

Thus `d` calibration transitions are sufficient. They are generated offline
by the world/teacher and packed into the training transcript. They add
trajectory tokens but no learner-decode/world-synchronization round.

The task goal is sampled only after calibration and a fresh reset. Calibration
therefore cannot encode which outcome will later be requested.

### 4.2 Goal-directed phase

After reset, sample a start `x_0` and goal `g`. The public-prefix teacher
reconstructs `B` from the calibration trajectory and emits the unique bounded
greedy command

```text
u*_t = clip(B^(-1)*(g - x_t), -u_max, u_max).
```

Goals are sampled so some examples require several bounded commands, rather
than collapsing every rollout to a one-step lookup. The actual next
observation is appended and the policy acts again. A held-out perturbation
sentinel may alter an intermediate observation to test feedback correction;
the source version remains deterministic initially.

Each completed transition supplies:

- an action target from the public teacher;
- an optional next-observation target from the public transition prefix; and
- an eventual goal/error/success result from the public verifier.

The learner's 16-step action head uses a horizon mask for this world's
single-command targets. The world therefore does not pretend to teach
high-frequency chunk structure it does not contain.

## 5. Exact information and validity claim

The intended sufficient statistic is the public calibration trajectory, not a
privileged matrix. A world version is inadmissible unless all of the following
hold:

1. a serializer-level public-prefix oracle reconstructs every teacher action;
2. the public teacher and latent-generator teacher agree exactly on sampled
   expert support;
3. target actions are unique under the declared bounds and expert sampling
   margin;
4. changing hidden generator bookkeeping without changing the public
   transcript cannot change the target;
5. train/evaluation process instances and generator seeds are disjoint;
6. filenames, batch order, padding, schema lengths, and cache keys reveal no
   hidden family parameter;
7. jointly relabeling observation and actuator keys produces the corresponding
   relabeled transcript and action target;
8. future observations/actions cannot alter earlier readout inputs or labels;
9. teacher forcing cannot expose the current target before its query; and
10. a historyless baseline is separated by counterfactual pairs with identical
    current `x_t` and `g` but different calibration histories and required
    actions.

These specialize the repository's world-validity, correspondence, rendering,
and STEP 1 failure-derived gates. Apparatus failures block admission; they are
not interpreted as evidence that the capability is unlearnable.

### 5.1 Traceability to the repository contracts

| Repository requirement | This family's concrete realization |
|---|---|
| declared ontology | Section 3 declares `S`, `A`, `O`, transition, goal, bounds, reset, and costs/command count; no physical realism is claimed |
| learner information boundary | Section 4 lists the complete public schema and trajectory; `P`, `b`, seeds, and teacher state remain hidden |
| realizable teacher | both the calibration reconstruction and control target are computed by a serializer-level public-prefix oracle |
| public-information ceiling/baseline | oracle action and outcome errors are reported; a historyless baseline is tested on counterfactual pairs |
| principled procedural family | every variation axis and support stratum is declared before generation; train/evaluation instances and seeds are disjoint |
| semantic-preserving invariance | joint key relabeling, legal token reordering, and generator-bookkeeping perturbations preserve the corresponding public problem |
| leakage and shortcut resistance | future-prefix, target-position, filename/cache/order, padding/length, and mismatched-history controls are admission gates |
| transfer plausibility rather than source score | the claimed control/interface dependencies and disposable descendant adaptation probes are declared separately from source competence |

This table operationalizes the inherited
[world-validity](../../WORLD-VALIDITY.md) contract and this packet's explicit
capability-first selection argument without modifying or superseding the
repository contract.

## 6. Blank-state plausibility criteria

Worlds first pass hard semantic gates; invalid worlds receive no plausibility
score. Admissible candidates are then compared as a vector:

```text
Plausibility(w | C0) = [
    probability of acquisition from random weights,
    alignment with GEN-like dependency properties,
    transfer fan-out / option value,
    structural coverage of the persistent token interface,
    anti-shortcut identifiability,
    diagnostic resolution,
    ontology neutrality,
    target density,
    complete inner-loop cost,
    admission and next-decision overhead,
    uncertainty in all preceding estimates].
```

No fixed weighted sum is asserted before measurements. A reducer may rank a
particular candidate set, but hard validity gates and component values remain
visible.

`W_calibrated_monomial` is currently preferred because it combines:

- a public exact teacher and dense targets;
- deterministic, immediate, reversible effects;
- low dimensions and no perception/contact/planning prerequisite;
- genuine trajectory-context dependence;
- a nontrivial variable sensor/actuator interface;
- exact relabeling and history-mismatch falsifiers;
- goal-relative inverse control and multi-step correction; and
- offline generation compatible with static pretraining throughput.

Its main weakness is equally explicit: the effect graph is one-to-one and the
teacher supplies all calibration, so it does not yet teach coupled dynamics,
active probing, object structure, or task inference from demonstrations.

## 7. Candidate comparison at `S0`

The table is a qualitative prior, not measured evidence.

| Candidate | Blank learnability | GEN-dependency alignment | Interface coverage | Diagnostic value/cost | Decision |
|---|---|---|---|---|---|
| fixed identity reaching | very high | low: no context needed | low | low | reject |
| scalar calibrated effect only | very high | moderate | low | high but too narrow | unit/evaluation slice only |
| **calibrated monomial effects, `d=1..4`** | **high** | **high for the control/interface trunk** | **high for the initial ABI** | **high** | **select one bounded session** |
| sparse coupled effects, `d=2..4` | medium | high | high | high, but harder attribution at blank weights | nearest new-world candidate |
| active identification without supplied pulses | low-to-medium | very high | high | high scientific value, high sequential span | defer |
| object-centric visual manipulation | low at blank weights | very high | high | poor initial attribution and cost | independent later trunk |
| broad static early-world mixture | uncertain | potentially highest | broad | poor initial attribution, excellent throughput | mandatory comparator and legal action |

The selection says only that this action has the highest current expected
decision value. It does not say the family is foundational or that its nearest
variant should be selected next.

## 8. World performance is not one scalar

Every checkpoint evaluation on this family returns at least:

```text
E_world = [
    normalized action L1 error,
    next-observation/delta error,
    closed-loop goal success,
    normalized terminal goal error,
    commands and action energy to success,
    context-ablation gap,
    mismatched-calibration degradation,
    joint-relabeling equivariance gap,
    token-order robustness gap,
    held-out gain/permutation and per-trained-cardinality performance,
    out-of-support cardinality extrapolation sentinel,
    public-oracle gap,
    confidence intervals and complete cost].
```

Source loss is not a substitute for closed-loop competence. Closed-loop success
is not a substitute for context dependence: a shortcut policy must fail the
history mismatch. Equivariance is not a substitute for control. The vector is
retained so later scheduler decisions remain interpretable.

### Online world-native score possibility

During a training session, the world may compute a cheap held-out subset of
this vector from public trajectories and resource counters. A predeclared
reducer can emit a scalar with uncertainty for plateau/early-reconsideration
logic, for example a lower confidence bound on context-dependent goal success
per attributable second. That scalar is scheduler telemetry only. It neither
fully maps this world nor becomes comparable to another world's score without
an explicit cross-world utility model.

STEP 1's SDL remains a candidate curve summary for this monitor, subject to
preserving its original calibration and interpretation. It is a note, not an
adopted objective or a direction change.

## 9. Cheap sentinels versus transfer evidence

Frequent cheap sentinels answer whether this source world was learned as
intended:

- remove calibration;
- substitute calibration from another process while fixing current state and
  goal;
- change only the goal;
- jointly relabel channels/actuators;
- change legal serialization order; and
- compare to a historyless model or ablation.

They do not prove transfer. When the next action actually depends on it, buy a
matched disposable adaptation branch on one or more of:

- unseen cardinality (`d=5` or `6`);
- sparse coupled `B`;
- one-step delayed effects;
- task goal conveyed by a demonstration rather than an explicit vector; or
- a renderer/body remapping that preserves the underlying effect relation.

Positive transfer means a better success-versus-new-experience curve than the
identical architecture at `C0`, under equal update/data/compute budgets. It is
not “the source checkpoint can already solve the descendant.”

## 10. Session boundary and recursion

The bounded session is declared in updates, packed targets/tokens, wall-clock
ceiling, and monitor rules. It ends at the first of:

- its hard resource ceiling;
- a predeclared source-learning plateau;
- sufficient evidence to distinguish the live next actions;
- a world-validity, leakage, or teacher-realizability failure; or
- evidence that the architecture is not using calibration or variable keys as
  intended.

It emits immutable `C1`, source evidence, monitor traces, and full cost. The
same scheduler is then called on the actual state:

```text
S0 -- TRAIN_NEW(W_calibrated_monomial, bounded budget) --> S1
S1 -- choose NEW | OLD | STATIC MIXTURE | EVALUATE | ADMIT | STOP --> ...
```

Possible `S1` actions include more support from the same family, a coupled
effect world, a perception/task/memory/active-probing world, a static mixture,
replay for a measured retention reason, a decision-relevant evaluation, or
rejection of this family. None is preselected.

## 11. Evidence that would reverse the `S0` choice before training

Recompute the decision if static analysis or bounded preflight shows that:

- a public-prefix teacher cannot exactly reproduce targets;
- simple current-state or serialization shortcuts survive the counterfactual
  gates;
- variable action queries cannot learn the `d=1..4` family at the experimental
  scale while a paper-faithful fixed-vector ICRT baseline learns matched
  fixed-cardinality slices;
- the `O(d^2)` calibration-token growth makes target throughput materially
  worse than an equally informative packet;
- another admissible prepacked world packet gives comparable blank
  learnability and materially higher GEN-dependency coverage; or
- the selected family produces no decision-relevant separation among likely
  next actions.

The first world is important only because it is the first use of a persistent
architecture and a finite budget. It has no privileged status after evidence
arrives.
