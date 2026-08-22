# Robot-Stage World Validity

## The plausibility contract

### Status of this document

This is the STEP 2 world-validity contract anticipated by
[step2/README.md](README.md) and deferred at
[STEP-1 closure](../STEP-1-WORLD-0.1-CLOSURE.md#handoff-boundary).

It supplies the acceptance criterion for worlds in the robot stage. It
supersedes the derivation chain in
[WORLD-CORRESPONDENCE.md](../WORLD-CORRESPONDENCE.md) as the test of world
validity, and retains that document's representation obligations under a new
justification given in section 7.

It specifies no world, body, operator, goal encoding, teacher, or experiment.
Those remain unauthorized.

### What this stage is not

The robot stage is not simulation training and does not become simulation
training by using bodies, actuators, or spatial structure. No world in this
stage models a real system, and no result in this stage is a claim about
transfer to a particular real platform.

The stage is also not sensorimotor training. Motor competence on a body is not
the capability under development; section 6 makes this a measurement property
rather than a matter of intent.

---

## 1. What a world is

A world in this stage is the smallest configuration that supports an
interaction mode:

- an **agent** that controls **actuators**;
- **abstract sensors** that deliver partial observation of a persistent state;
- **abstract goal specification machinery** that conditions the agent.

The goal channel is inbound. The agent receives an abstract goal specification
and does not receive, emit, or require language anywhere in the loop.

Nothing further is required. A world needs no physics engine, no rigid-body
model, no realistic morphology, and no counterpart in any real activity. The
agent may be a dog, a lion, a blob, a disembodied hand, or a coupling of
actuators with no anatomical reading at all.

What is required is the **interaction mode**, specified in section 3.

---

## 2. The plausibility criterion

Let an **interface shape** `I` be the type of the sensor channel, the type of
the actuator channel, and the type of the goal channel — not their contents.

> **Plausibility.** A world `W` is valid if, at every reachable state, it is
> defensible that any agent performing well in `W` would also perform well in
> real activity presented through an interface of shape `I`.

Three properties of this statement are load-bearing.

**It quantifies over agents, not worlds.** `W` is never compared to a real
process, and no map from a real practice to `W` is required or meaningful. The
comparison is between competences under a fixed interface shape.

**It holds interface shape fixed and nothing else.** Physical law, morphology,
sensory modality, spatial structure, object ontology, scale, and time are
unconstrained. Absurdity is not a cost to be paid down; it is licensed.

**It is continuous, not a design-time gate.** "At every reachable state" means
a world can lose validity in a region without losing it everywhere, and that
scoring moments are themselves states subject to the criterion.

### 2.1 The criterion is negative

Plausibility is not established by argument. It is **falsified by exhibiting a
degenerate winner**: a policy that performs well in `W` and that would be
manifestly incompetent under `I`.

A world admitting a degenerate winner is invalid regardless of what it was
designed to teach, what its designer intended the winning policy to be, or how
well the intended policy also scores.

### 2.2 The dual failure

The criterion fails in the other direction when no agent with interface `I` can
perform well in `W`. The antecedent never fires, the criterion is satisfied
vacuously, and the world measures nothing.

A **vacuous world** is as invalid as one with a degenerate winner. Section 5
gives its standard cause.

---

## 3. Interaction mode

The learner's experience must be that of acting inside a persistent system
rather than exchanging messages with one. This is a constraint on the
phenomenology of the loop, not on its implementation. A world satisfying it
need not compute physics.

A world is in the interaction mode when:

1. **Consequences persist.** State accumulates across actions. Nothing resets
   between interventions.
2. **Sensing is partial and positioned.** Observation depends on where the
   agent is and what it has done, not only on what is true.
3. **Actuation is inertial.** Commands do not teleport state. There is latency
   between command and effect, and effects have extent.
4. **The world changes and does not answer.** No action retrieves a fact. All
   information arrives as a consequence of a state change.
5. **Incoherent commands act.** Malformed or infeasible actuation produces a
   consequence rather than an error, a rejection, or a no-op.
6. **There is no privileged narrator.** The world never describes its own
   state, names its own objects, labels its own events, or reports its own
   rules.

Obligation 6 is the sharpest. A world that narrates is a text adventure with
motor vocabulary, and narration is a degenerate winner by construction: the
description channel carries competence the agent never had to acquire.

---

## 4. Degenerate winners

A world must be audited against each class before a run is authorized.

| Class | The policy wins by | Detection |
| --- | --- | --- |
| **Substrate exploit** | an implementation artifact — tie-break order, index layout, numerical quirk, finite generator support | replay under permuted implementation choices that preserve semantics |
| **Channel leak** | correlation between the goal or sensor encoding and the outcome, independent of the process | score a policy given the channel with the process resampled |
| **Family memorization** | the generated family being small enough to memorize | finite-support and raw-fingerprint diagnostics retained from STEP 1 |
| **Narration exploit** | information the world volunteers that a system in the interaction mode would not | obligation 6 audit; inspect every non-consequential byte reaching the learner |
| **Body-regularity exploit** | actuator-to-effect regularity alone, ignoring the process | score a body-only policy with the process held constant |

The last is the robot-stage form of the failure that closed `world-0.1.0`.
Body regularity is a legitimate thing to acquire. The audit does not ask
whether a policy uses it; it asks whether using it *alone* clears the bar. If
the body-only band covers the measured result, the world has not measured the
process, and no change of learner, optimizer, or supervision regime can repair
that.

The bands from all five audits must be reported alongside any result. A result
inside an audit band is not a weak result; it is an uninterpretable one.

---

## 5. Realizability

The standard cause of a vacuous world is a goal or target the learner's sensor
channel cannot support.

> **Realizability.** Every target the teacher supplies must be computable from
> the learner's observation prefix, or must be declared as a distribution over
> what that prefix determines.

Privilege here means one thing and has no realist reading: information on the
far side of the learner's sensor channel in the constructed world. There is no
true state being hidden, because there is no real system being modeled. There
is a designed information boundary, and the teacher either respects it or the
world scores a competence no agent with interface `I` could hold.

A demonstrator built from world-side state is not a strong teacher. It is a
specification of a vacuous world.

---

## 6. Bodies are renderings

The presentation/rendering distinction in
[PROCESS-AND-RENDERING.md](../PROCESS-AND-RENDERING.md#3-presentation-and-rendering)
transfers directly and is the stage's main discipline.

- Two bodies are **two renderings** of one world when each can express the
  world's intervention set and each sensor suite delivers the same process
  information. Consequence invariance must hold: the same typed intervention,
  expressed through either body in the same situation, induces the same
  consequence.
- If a body's sensors carry different process information, the change is a
  **presentation** change. It yields a different developmental object and must
  be labeled as one.

A body may be called a rendering only after the actuation-completeness and
sensor-equivalence checks are recorded. The distinction is checkable and it is
the one this stage is most likely to lose.

This is also what keeps the stage out of sensorimotor training, by measurement
rather than by intent. Any capability claim must be stated across bodies. A
capability that does not survive a rendering change is motor skill on one body,
whatever it was designed to be.

The probe skeleton from
[PROCESS-AND-RENDERING.md §4](../PROCESS-AND-RENDERING.md#4-the-first-discriminating-probe)
survives the pivot with substituted axes:

| | Body A | Body B |
| --- | --- | --- |
| **Same law** | acquisition | body-transfer |
| **Changed law** | semantic discrimination control | joint control |

The changed-law control is required, not optional. Two bodies without it
cannot distinguish a policy that represents the world's process from one that
has aligned to body dynamics.

---

## 7. Obligations retained from the correspondence document

Cost and commitment, validity change, recovery distinctions, and goal and
stopping conditions were obligations in
[WORLD-CORRESPONDENCE.md §3](../WORLD-CORRESPONDENCE.md#3-correspondence-obligations)
because a real practice had them and the synthetic world had to preserve them.

That justification is gone. The obligations remain, re-grounded: a world
lacking them is too shallow to separate competence from luck, and therefore
admits degenerate winners. They are anti-degeneracy requirements now, not
fidelity requirements.

The representation obligations — faithfulness, learnability, non-leakage,
consequence invariance, gradient affordance — carry over unchanged and apply to
the goal channel and actuator channel as written.

The **translation bridge** obligation does not carry over in its original form
and is not discharged by this contract. With language removed from the loop,
the claim that acquired organization can reach a later interface needs a
replacement statement. Recorded here as an open obligation on the stage, not on
any individual world.

---

## 8. Throughput

Robot-stage worlds inherit the cost model in
[TRAINING-DYNAMICS.md](../TRAINING-DYNAMICS.md#2-a-cost-model) unchanged,
including its three regimes and the gradient-throughput definition.

Two consequences follow and constrain world construction directly.

**No integrator.** A contact-dynamics solver places world transition cost in a
regime where it dominates the cost ratio, and interaction-mode episodes already
carry sequential learner-world boundaries. A body is a typed actuation
interface with a compiled effect algebra — actuators, coupling constraints, a
typed observation map. Obligation 3 of section 3 requires that actuation *feel*
inertial, which a compiled effect algebra can deliver and which no solver is
needed to produce.

**Typed channels.** Discretizing the actuator and goal channels into typed
codebooks preserves the retained standard stack, the existing gradient-bearing
token accounting, and the Rust executable-world boundary. A codebook is not
language; it carries no lexical prior. This is a default, not a requirement,
and a world needing continuous channels must state the unmet requirement and
the throughput consequence.

---

## 9. Acceptance gates

A world is authorized for a scientific run only when all of the following are
recorded:

1. Interface shape `I`, with the type of each of the three channels.
2. The plausibility argument, stated as the defense against each degenerate
   winner class in section 4.
3. Measured bands for all five audits.
4. The interaction-mode checklist, section 3, item by item.
5. The realizability statement for every teacher target, section 5.
6. For each body, the actuation-completeness and sensor-equivalence record that
   licenses calling it a rendering rather than a presentation.
7. The changed-law control, specified before the acquisition run.
8. The throughput accounting under section 8.

Gates 3 and 5 are the two that closed `world-0.1.0`. Neither was recoverable
after the fact.

---

## 10. Deferred

This contract does not define world operators, a body ontology, a goal
encoding, a law family, a teacher policy, a curriculum, or any world. Operators
may be proposed only after independently motivated worlds exist, per the
handoff boundary at STEP 1 closure.

The translation-bridge replacement claim in section 7 is open.

**Selection is open and is not a validity question.** Plausibility is a purely
negative criterion: it rejects worlds that admit degenerate winners or that no
agent with interface `I` can satisfy. It does not rank the worlds that survive,
and a world can pass every gate in section 9 while developing nothing worth
having. Correspondence answered the selection question badly, by grounding
value in a real practice; removing it removes the bad answer and leaves the
question. A selection criterion is required before a world family is
authorized, and it must not be smuggled back in as a validity gate.
