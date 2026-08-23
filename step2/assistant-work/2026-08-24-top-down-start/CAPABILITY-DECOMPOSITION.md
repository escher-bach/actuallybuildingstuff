# Top-Down Capability Decomposition

**Date:** 2026-08-24
**Status:** assistant-authored research proposal for user review; it does not
authorize a world, implementation, or training run

## 1. End capability

The target is not generic "robot intelligence," success on one simulator, or
motor fluency on one body. In standard terminology, the target is a
**generalist, goal-conditioned, adaptive embodied policy** with both
**in-context adaptation** and **positive continual transfer**:

> Given a previously unseen controlled process, a public sensor/actuator
> interface, a non-linguistic task specification, and a bounded history of
> interaction or demonstration, the learner should infer enough about the
> current system and intended outcome to improve its behavior in context. If a
> parameter update is still required, the current checkpoint should require
> less new experience than the identical architecture at random initialization,
> while retaining useful performance from earlier worlds.

Formally, an encounter supplies:

- an unknown process instance `M_z = (S, A, O, T_z, Omega_z)`;
- a public interface schema `e` describing only exposed channels, bounds, and
  timing—not the hidden dynamics parameter `z`;
- a task specification `q`, initially an observable goal and later possibly an
  outcome example or sensorimotor demonstration;
- a public history `h_t = (e, q, o_0, a_0, f_1, o_1, ..., o_t)`, where `f`
  contains only feedback actually exposed by the world; and
- a bounded interaction and/or update budget.

The deployed policy is `pi_C(a_t | h_t)` for checkpoint `C`. A developmental
session may transform `C` into a new immutable checkpoint `C'`. The transfer
claim is comparative: on held-out descendant processes, `C` or `C'` should
improve success-versus-experience relative to the preserved random checkpoint
`C0`, not merely achieve a high source-world score.

Natural language is not part of this definition. It can later be one optional
task or observation modality, but it is neither the core ontology nor the
action boundary.

### Operational north star: GEN-1.5-like generalization

The concrete end demonstration is **GEN-1.5-like generalization**, not merely
high average reward after a long fixed curriculum. In standard terms, the
target bundle is:

- one- or few-shot **in-context imitation** from a physical demonstration;
- **compositional generalization** from more than one demonstration in context;
- closed-loop execution, perturbation correction, improvisation, and recovery;
- transfer when demonstration and deployment differ in embodiment or
  realization, including eventual human-to-robot and simulation-to-real
  prompting;
- broad goal/task and object/tool generalization; and
- one-to-ten-step-style rapid gradient adaptation when context alone is
  insufficient.

This is a directionally matched target, not a claim that this project can
replicate Generalist's proprietary data, scale, architecture, or reported
success rates. The [GEN-1.5 report](https://generalistai.com/blog/gen-1.5)
describes a large model with 30 seconds of multimodal memory, 3–12 second
physical prompts, and 100 Hz action trajectories, but does not disclose an
implementable architecture. The project must therefore preserve the
*possibility* of those behaviors while testing their simpler dependencies.

The end target is a vector, not one scalar:

```text
U_end = [unseen-task success with zero updates,
         causal benefit from the physical prompt,
         two-prompt composition,
         cross-realization/embodiment transfer,
         closed-loop robustness and recovery,
         success-versus-gradient-step curve,
         breadth across held-out capability trunks,
         retention of prior capabilities,
         complete interaction/compute cost].
```

A declared reducer may support one scheduling decision, but every component
remains logged. A model cannot compensate for zero prompt dependence or a
failed cross-embodiment gate by scoring highly on an easier source world.

This end target creates a hard architecture test: physical trajectories,
multiple demonstrations, actual executed actions/outcomes, variable public
interfaces, and non-language observations must all be representable before
training begins. The selected compromise and its custom risk are specified in
[MODEL-AND-REPRESENTATION.md](MODEL-AND-REPRESENTATION.md).

## 2. Learner information contract

The mature interface must be able to expose:

- typed observations and their local channel identities;
- typed executed actions and their local actuator identities;
- reset, episode, phase, and time boundaries;
- a task specification as goal observations, desired outcomes, demonstrations,
  or another explicitly added modality;
- public outcome, cost, or validity feedback when a world has it; and
- public embodiment or connectivity information only when that information is
  genuinely available in the intended encounter.

It must never silently expose:

- the generator seed or world-family label;
- latent state not produced by the observation function;
- the hidden transition law, camera transform, morphology, or task identity;
- a teacher-only counterfactual table;
- future observations or future actions at deployment; or
- a pretrained semantic/action codebook while calling the learner blank.

This contract is part of the capability definition. A policy that succeeds
with hidden configuration tokens has not demonstrated system identification.

## 3. The downstream capability leaves

These are the concrete capability classes the developmental program is trying
to make easier to acquire. They are deliberately broader than one robot or one
benchmark but narrow enough to generate tests.

| ID | Downstream capability | Observable success condition |
|---|---|---|
| `L1` | **Interface and embodiment adaptation** | Calibrate and control when sensor ordering, actuator ordering, gains, camera/body configuration, or channel count changes. |
| `L2` | **New goal acquisition** | Reach unseen outcome specifications and interpolate or extrapolate across goals without a task-specific head. |
| `L3` | **Physical-prompt task inference** | Infer what behavior is requested from one or a few sensorimotor demonstrations or outcome examples, without requiring text. |
| `L4` | **Robust partially observed control** | Maintain task-relevant state, correct after disturbances, and recover when effects are delayed, noisy, or temporarily hidden. |
| `L5` | **Active learning of a new process** | Choose safe, informative probes when passive history is insufficient, then exploit what was learned. |
| `L6` | **Object- and relation-sensitive interaction** | Transfer control across changing numbers, identities, and arrangements of persistent entities rather than memorizing a scene layout. |
| `L7` | **Long-horizon composition and recovery** | Sequence reusable effects/subgoals under constraints, detect failed progress, revise, and continue. |
| `L8` | **Continual capability acquisition** | Add a new world or task with positive learning-curve transfer while limiting loss on earlier capability probes. |
| `L9` | **Cross-realization physical prompting** | Use a demonstration expressed through a different renderer, simulator, body, or human interface to improve control of the deployment embodiment without a shared task label. |

No individual world can or should teach all nine. The capability structure is
a forest with several trunks that later merge. At every scheduler state, worlds
from different trunks compete under the same recursive value and cost rule;
appearance earlier in the lineage creates no special status or universal
ontology.

## 4. Dependency graph

The graph below is a design hypothesis, not a theorem extracted from any one
paper. An edge means that the parent is expected to make the child more
learnable for this learner and therefore creates a transfer claim that can be
tested.

```text
cross-cutting interface: P0 typed/variable channels + P1 temporal boundaries
        |
        +-- CONTROL/ADAPTATION TRUNK
        |     P2 action-effect -> P3 predictive state
        |     -> P5 inverse control -> P6 correction
        |     + P4 observable goals + P7 system identification
        |     -> I1 new body/dynamics ----------------------> L1, L2
        |     -> I3 informative probing --------------------> L5
        |
        +-- PERCEPTION/STRUCTURE TRUNK
        |     P8 persistent/invariant sensory factors
        |     -> P9 entity and relation binding ------------> I5 -> L6
        |     + P11 cross-realization correspondence --------> I8 -> L9
        |
        +-- TASK-EVIDENCE TRUNK
        |     P10 separate desired outcome from system facts
        |     -> I4 infer intent from examples --------------> L3
        |
        +-- MEMORY/CONTROL MERGE
        |     control trunk + hidden/delayed evidence
        |     -> I2 belief-state control --------------------> L4
        |
        +-- COMPOSITION MERGE
              control + task + memory + structure
              -> I6 composition/recovery -------------------> L7

all trained trunks + I7 update efficiency/retention --------> L8

L1 + L2 + L3 + L4 + L5 + L6 + L7 + L8 + L9
        --> generalist adaptive embodied policy
```

The important top-down conclusion is narrower: `P0`–`P7` form a
high-fan-out **control/adaptation trunk** with unusually cheap candidate actions
at blank weights. They express a reusable organization:

> bind an action to its observed consequence, infer the current action-effect
> organization from context, relate a desired observation to the current one,
> and correct from the actual result.

They are not the ancestor of every perceptual or compositional capability.
`P8`–`P10` require independent early worlds, and the repeated scheduler may
select those worlds before deepening the control trunk.

## 5. Primitive properties: decomposition, tests, and prior work

| Node | Simpler property | Required public evidence | Falsifying/isolating test | Existing-work connection | Earliest world operator |
|---|---|---|---|---|---|
| `P0` | Typed interface grounding with variable sensor/actuator cardinality | role, local channel identity, value, boundary | jointly permute channels; add masked/new channels; change serialization | CrossFormer handles heterogeneous observations/actions; AnyMorph handles varying sensor/action dimensions; RoboToken and Body Transformer use entity-level tokens | `PERMUTE_INTERFACE`, `RESIZE_INTERFACE`, `CHANGE_RENDERER` |
| `P1` | Temporal binding | ordered observation-action-outcome events | reorder or mismatch action/outcome pairs; insert delay | ICRT and Algorithm Distillation use causal trajectory histories; CrossFormer uses block-causal temporal input | `DELAY_EFFECT`, `RESET`, `NEW_EPISODE` |
| `P2` | Action-effect attribution (agency) | paired pre-state, executed action, post-state | preserve states while mismatching actions; compare action-free predictor | SMART and RPT explicitly train inverse/forward sensorimotor objectives; LiLa-WAM jointly models actions and future latent state | `RANDOMIZE_EFFECT_GRAPH` |
| `P3` | Action-conditioned predictive state | multiple transitions under the same apparent state but different actions | intervention counterfactuals and held-out action-effect combinations | SMART, RPT, LiLa-WAM, and structured world-model work | `COUPLE`, `NOISE`, `SWITCH_RULE` |
| `P4` | Goal-relative state comparison | goal and observation expressed through aligned public channels | hold state/history fixed and counterfactually change only the goal | UVFA and HER establish goal-conditioned generalization/relabeling; CrossFormer supports goal images | `SET_OBSERVABLE_GOAL` |
| `P5` | Inverse controllability | recoverable action effects plus a desired change | alter only the effect map; test whether action changes consistently | SMART's inverse dynamics objective and goal-conditioned control literature | `SUPPLY_CALIBRATION`, `MAKE_REVERSIBLE` |
| `P6` | Closed-loop correction | actual post-action observation | perturb an intermediate outcome; compare open-loop and feedback policies | Standard receding-horizon control; ICRT/robot trajectory policies execute closed loop | `PERTURB_OUTCOME`, `ALLOW_UNDO` |
| `P7` | Context-conditioned system identification | task-independent action-observation transitions from the current process | remove/mismatch calibration; hold current state and goal fixed | ICWM explicitly uses task-agnostic interaction histories to infer camera/body configuration; ICRT establishes trajectory-context conditioning | `RANDOMIZE_DYNAMICS`, `SUPPLY_CALIBRATION` |
| `P8` | Persistent and invariant sensory factors | aligned observations across time or semantics-preserving render changes | alter rendering while preserving process; break temporal identity in a control | contrastive/structured world models and object-centric representation work | `CHANGE_RENDERER`, `PERSIST`, `OCCLUDE` |
| `P9` | Entity and relation binding | multiple persistent components with independently changing attributes and relations | permute entity order/count; preserve local relations while changing global layout | C-SWM, Body Transformer, GET-Zero, and RoboToken provide different structural priors | `ADD_ENTITY`, `ADD_RELATION`, `PERMUTE_ENTITIES` |
| `P10` | Task-evidence separation | examples or desired outcomes that vary independently of system-identification context | mismatch task demonstration while holding calibration fixed, and vice versa | ICRT, AdA, and GEN-1.5-style physical prompting distinguish behavioral context from current state | `REPLACE_GOAL_WITH_DEMONSTRATION` |
| `P11` | Cross-realization correspondence | paired or structurally related behavior observed through different renderers, bodies, or domains | preserve behavior while changing renderer/body; mismatch behavior while preserving superficial appearance | GEN-1.5 reports human/simulation prompts; BPP and RoboTTT study human demonstrations; CrossFormer studies embodiment variation | `CHANGE_RENDERER`, `CHANGE_BODY`, `CROSS_DOMAIN_DEMONSTRATION` |

### Why these are properties rather than task names

The primitive properties can occur in manipulation, locomotion, navigation, tool use, or an
alien synthetic process. No node presupposes a room, object class, Cartesian
arm, language instruction, or human-like body. That is why a source checkpoint
could plausibly retain value when the later world's ontology changes.

## 6. Intermediate capabilities and what they add

| Node | Depends on | New requirement | Existing-work evidence | World transformation that isolates it |
|---|---|---|---|---|
| `I1` system/embodiment adaptation | `P0`–`P3`, `P5`–`P7` | infer a changed observation/action configuration rather than a fixed mapping | ICWM, CrossFormer, AnyMorph, GET-Zero, Body Transformer | replace supplied effect map with new cardinality, coupling, viewpoint, or public morphology |
| `I2` belief-state control | `P1`–`P3`, `P6`–`P7` | retain uncertainty-relevant history when current observation is insufficient | recurrent/meta-RL and adaptive-agent results; AdA scales attention memory | `HIDE`, `DELAY_EFFECT`, `NOISE` |
| `I3` active system identification | `I1` + `I2` | choose actions for information value before task value | ICWM's probing formulation; AdA's hypothesis-driven exploration | `REMOVE_SCAFFOLD`, `ADD_QUERY`, `ADD_PROBE_COST` |
| `I4` task inference from behavior | `P1`, `P4`, `P7`, `P10` | distinguish evidence about *what to do* from evidence about *how the process works* | ICRT trajectory prompting, GEN-1.5 physical prompting, AdA demonstration prompting | `REPLACE_GOAL_WITH_DEMONSTRATION` |
| `I5` entity/relational abstraction | `P0`–`P3`, `P8`–`P9` | bind persistent entities and model interactions independent of ordering/count | C-SWM, Body Transformer, GET-Zero, RoboToken | `ADD_ENTITY`, `ADD_RELATION`, `PERMUTE_ENTITIES` |
| `I6` composition, commitment, recovery | `P3`–`P6`, `I2` | reason over several effects, resource limits, irreversible choices, and repair | planning and hierarchical-control literature; SAC-X shows auxiliary intentions can support sparse compound tasks | `COMPOSE`, `LIMIT_RESOURCE`, `MAKE_IRREVERSIBLE`, `ALLOW_UNDO` |
| `I7` continual transfer and retention | all learned source nodes | use an old checkpoint to learn a new family faster without destructive forgetting | RoboCat's iterative data/checkpoint loop; multi-embodiment world-model work finds rehearsal useful under distribution shift | `NEW_WORLD`, `REPLAY_OLD_WORLD`, frozen transfer branches |
| `I8` cross-realization task transfer | `P0`, `P8`–`P11`, `I1`, `I4` | infer common intended physical change despite different pixels, embodiment, or action realization | GEN-1.5 target evidence; BPP behavior prompts; RoboTTT human-video context; cross-embodiment policies | paired renderer/body changes followed by held-out cross-realization prompts |

Algorithm Distillation adds an important qualification for later `I3` and `I4`
training: expert trajectories alone show a solved policy, not the process of
exploration and improvement. Once the learner owns probing, training data must
contain multi-episode learning histories—including failed or non-expert probes—
and the context must be long enough to span improvement. That requirement is
not part of the world action currently selected at `S0`, but the base sequence
architecture must leave room for it.

## 7. Capability target of the currently selected `S0` action

The control-world action currently preferred at `S0` should directly target:

- `P0` typed variable interfaces;
- `P1` causal temporal binding;
- `P2` action-effect attribution;
- `P3` compact next-observation prediction;
- `P4` observable goal conditioning;
- `P5` inverse controllability;
- `P6` closed-loop correction; and
- a scaffolded, passive form of `P7` system identification.

It should deliberately defer or hand to independent trunks:

- choosing probes (`I3`), because supplied calibration makes the information
  boundary and teacher target exact;
- partial observability (`I2`), because a blank learner should first see direct
  action consequences;
- demonstration-inferred intent (`I4`), because otherwise task identification
  and system identification are confounded;
- object/morphology ontology (`I5`), because it would force a representational
  commitment before transfer warrants it; and
- long-horizon irreversible planning (`I6`), because it weakens learnability
  and diagnostic precision at the blank state.

This is the reason the present `S0` policy selects a bounded session of
scaffolded goal-conditioned control over **variable-cardinality signed and
permuted action effects**, rather than a scalar-only world. It cheaply covers
control/adaptation ancestors and exercises the persistent schema-conditioned
readout interface while excluding most branch-specific difficulty. The scalar
case remains one support slice and unit test. The selected action receives no
permanent special head, tokenizer, codebook, replay share, or claim on the next
action.

## 8. Transfer fan-out relevant to later effect-control selections

A source score cannot validate the recursive value assigned to this world
family. When the result can change a live selection, checkpoints influenced by
it should be compared with appropriate controls, including `C0`, on disposable
adaptation branches spanning these orthogonal axes:

| Probe | First changed dependency | What positive transfer would mean |
|---|---|---|
| `T-schema` | new sensor/actuator count and unseen permutations | interface/channel organization transferred |
| `T-dynamics` | coupled or mildly nonlinear effect law | action-effect and control organization transferred beyond one equation |
| `T-memory` | one-step delay or hidden intermediate state | temporal predictive state transferred |
| `T-goal` | held-out goal geometry or multi-coordinate target | goal-relative control transferred |
| `T-task` | desired outcome conveyed by a trajectory rather than an explicit goal | sensorimotor context is usable for later task inference |

`T-task` is intentionally a far probe and may show no benefit after an early
checkpoint. This family earns continued allocation only if its advantage is not
confined to the nearest algebraic variant. The minimum persuasive pattern is positive
learning-curve transfer on `T-schema` plus at least one of `T-dynamics` or
`T-memory`, with no material negative transfer on the others. Exact statistical
thresholds belong in an experiment specification, not in this design note.

This is not a requirement to run all five branches after every session. A full
matched suite can cost as much as another training session. Every checkpoint
uses cheap current/parent/frontier sentinels; matched adaptation branches are
purchased only when their result can change the next-world decision; the full
fan-out suite is a level or representation milestone. The cost model and
selection rule are specified in
[REPEATED-DEVELOPMENTAL-LOOP.md](REPEATED-DEVELOPMENTAL-LOOP.md).

## 9. Checkpoint decisions follow the graph, not a fixed curriculum

For every training session:

```text
scheduler state S_n
    + selected bounded training or evaluation action
    -> immutable child checkpoint when training occurs
    -> evidence and complete cost update
    -> optional probes only when their value of information warrants them
    -> NEW | OLD | STATIC MIXTURE | EVALUATE | ADMIT | STOP
```

- **NEW WORLD** selects an admitted world or mixture with high recursive value;
  it need not be a child of the last world.
- **OLD WORLD** continues or replays a source when mastery, context use, or
  retention is inadequate.
- **EVALUATE** leaves the main weights fixed and may create disposable branches
  when their result can change the next action.

The graph can be revised when transfer evidence disagrees with its proposed
edges. The learner's curriculum is therefore empirically corrigible rather
than a one-time list of increasingly elaborate tasks. Crucially, the same
selection question is repeated at every checkpoint and the scheduling/evaluation
cost is included in developmental throughput rather than treated as free.

## 10. Sources that materially shaped the decomposition

- [ICRT: In-Context Imitation Learning via Next-Token Prediction](https://arxiv.org/html/2408.15980)
- [GEN-1.5: Embodied Foundation Models are One-Shot Learners](https://generalistai.com/blog/gen-1.5)
- [Behavior Prompting Policy](https://arxiv.org/html/2606.30457)
- [RoboTTT: Context Scaling for Robot Policies](https://arxiv.org/html/2607.15275)
- [Astra: Efficient Transformer Architecture for Embodied Instruction Following](https://arxiv.org/html/2408.01147)
- [In-Context World Modeling for Robotic Control](https://arxiv.org/abs/2606.26025)
- [Human-Timescale Adaptation in an Open-Ended Task Space](https://arxiv.org/abs/2301.07608)
- [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/abs/2210.14215)
- [Scaling Cross-Embodied Learning / CrossFormer](https://proceedings.mlr.press/v270/doshi25a.html)
- [AnyMorph](https://proceedings.mlr.press/v162/trabucco22b.html)
- [Body Transformer](https://proceedings.mlr.press/v270/sferrazza25a.html)
- [GET-Zero](https://arxiv.org/abs/2407.15002)
- [Transformer Transformer / RoboToken](https://arxiv.org/html/2607.25798)
- [Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html)
- [Learning by Playing / SAC-X](https://proceedings.mlr.press/v80/riedmiller18a.html)
- [Automatic Curriculum Graph Generation](https://ojs.aaai.org/index.php/AAAI/article/view/10933)
