# World Correspondence and Representation

## How a synthetic world earns the right to stand for a real activity

### Status of this document

This document concerns what synthetic worlds should mean. It does not estimate
training speed; that is the subject of [TRAINING-DYNAMICS.md](TRAINING-DYNAMICS.md).

The prior question is:

> Before a tutor selects a world and before a gradient generator teaches within
> it, how do we decide what that world corresponds to in real activity?

The answer is not a task label, primitive name, or superficial resemblance. It
is a documented argument that selected relations among state, action,
observation, consequence, cost, validity, goal, and recovery are preserved.

---

## 1. Who decides the worlds

World choice occurs at four distinct levels.

### Human practice determines the target

The project begins from a distribution over real practices \(\mu\), not a list
of abstract faculties. Examples include diagnosing a failing software service,
understanding an unfamiliar program, constructing a mathematical argument,
operating tools under incomplete information, retrieving evidence, and
maintaining constraints during a long task.

This selection is normative and historical. It says which later activities the
designed childhood is intended to prepare.

### Activity analysis proposes an abstract process

Practitioner traces, artifacts, tools, instructions, corrections, and accounts
of work are used to describe persistent state, observable state, actions,
consequences, information opportunities, costs, validity changes, uncertainty,
recovery, goal and stopping conditions, and scaffolded versus practitioner-owned
control.

This produces a practice-relative process model. It is not yet a synthetic
world and it is not an ontology of intelligence.

### World design realizes a controlled abstraction

The designer chooses which relations in that process model must survive and
constructs a synthetic world in which they can be generated, inspected, and
locally taught.

The generator samples particular hidden states, histories, and consequences
only after this abstraction exists. It does not decide what the abstraction
ought to mean.

### The tutor chooses learner-relative presentation

The tutor selects among already justified worlds or changes their revelation,
representation, guidance, and action ownership in relation to learner state.

The tutor may answer:

> Which accepted relation should this learner encounter now, and in what form?

It may not answer:

> Which arbitrary synthetic regularity should count as a counterpart of real
> practice?

Correspondence is an obligation on world design before it becomes an object of
curriculum control.

---

## 2. From a real activity to a synthetic world

### Step 1 — Delimit the practice

State the activity narrowly enough that its relevant state, actions, and goals
have recognizable meaning.

“Reasoning” is too broad. “Selecting the next diagnostic observation while
maintaining several live explanations for a service failure” is a usable
practice boundary.

The boundary identifies who acts, what they seek, what tools exist, what counts
as progress or commitment, and what time scale is being modeled.

### Step 2 — Describe the real controlled process

Represent the selected practice as

\[
W_R=(S_R,A_R,O_R,T_R,\Omega_R,G_R,C_R),
\]

where \(C_R\) records relevant action cost, risk, or commitment.

The model can be partial. Its role is to make the proposed abstraction visible,
including what it leaves out.

### Step 3 — Choose a capability-relevant abstraction

Define an abstract process

\[
Z=(S_Z,U_Z,T_Z,\Omega_Z,G_Z,C_Z).
\]

The abstract state contains distinctions claimed to matter for the activity.
For diagnosis it may retain live explanations, evidence status, available
inspections, commitments, and validity epochs—not the full physical state of a
computer system.

### Step 4 — Map the real process to the abstraction

Define maps from real histories and actions:

\[
\phi_R:H_R\to S_Z,
\qquad
\alpha_R:A_R\to U_Z.
\]

These maps state what histories and actions are being treated as equivalent.
Their omissions are part of the correspondence claim.

### Step 5 — Construct the synthetic realization

Build

\[
W_S=(S_S,A_S,O_S,T_S,\Omega_S,G_S,C_S)
\]

with maps

\[
\phi_S:H_S\to S_Z,
\qquad
\alpha_S:A_S\to U_Z.
\]

The synthetic world need not look like the real world. It must realize the
chosen abstract relations under controllable, generable dynamics.

### Step 6 — Choose and document the representation

Separate presentation from rendering. Presentation selects which process
information the learner receives; rendering expresses a fixed typed
observation or action in tokens. Give the learner a stable initial rendering
and at least one aligned alternative without silently changing the available
information or the process mechanics. Where possible, connect natural language
and later interfaces to the same typed objects through aligned translations.

The minimal ontology, rendering obligations, and required semantic control are
defined in [PROCESS-AND-RENDERING.md](PROCESS-AND-RENDERING.md).

```text
real practice
    -> practice process model
    -> capability-relevant abstraction
    -> synthetic realization
    -> learner-facing representation
    -> local gradient targets
```

Each arrow makes a different claim. A synthetic generator becomes meaningful
only through the chain as a whole.

---

## 3. Correspondence obligations

### State sufficiency

If two real histories map to the same abstract state, the abstraction claims
their difference does not matter for the chosen future decisions. The abstract
state must retain the distinctions it needs to make that claim honest.

### Affordance preservation

Relevant real actions must have abstract counterparts, and the synthetic world
must instantiate those counterparts. A supposed diagnostic world without a
choice of inspection has removed the central affordance.

### Transition preservation

Acting and abstracting should approximately commute.

For the real process:

\[
\phi_R(T_R(h,a))
\approx
T_Z(\phi_R(h),\alpha_R(a)).
\]

For the synthetic process:

\[
\phi_S(T_S(h,a))
\approx
T_Z(\phi_S(h),\alpha_S(a)).
\]

More precisely, induced distributions over next abstract states should agree to
the degree the correspondence claims.

### Observation and information preservation

Corresponding actions should reveal corresponding information. Calling two
actions `query` is insufficient if one eliminates live explanations and the
other returns only a decorative fact.

### Cost and commitment preservation

A free synthetic inspection cannot stand for a real action that consumes time,
risks damage, changes the system, or forecloses alternatives unless the omitted
cost is irrelevant to the stated activity.

Numerical costs need not match. Relevant orderings and tradeoffs must survive.

### Validity preservation

When real evidence expires, rules drift, or interventions change what was
previously true, the synthetic world must preserve that temporal validity. A
permanently fixed hidden rule cannot correspond to every changing practice.

### Goal and stopping preservation

The abstraction should retain why information is gathered and action is taken.
Otherwise it can preserve inference while erasing the question of when enough
has been done.

### Recovery preservation

If real activity distinguishes revising a belief, undoing an action, restoring
a trusted state, and trying another path, the synthetic world should not
collapse all four into one generic error token.

---

## 4. Approximate process correspondence

Exact equality is rarely appropriate for a real practice. Correspondence can
be stated approximately.

Let \(d_Z\), \(d_O\), and \(d_C\) compare abstract state transitions,
observation effects, and costs. A synthetic and real realization correspond
within declared tolerances when their mapped actions satisfy:

\[
d_Z
\left(
\phi_R(T_R(h_R,a_R)),
\phi_S(T_S(h_S,a_S))
\right)
\le\varepsilon_Z,
\]

\[
d_O
\left(
\Omega_R(h_R,a_R),
\Omega_S(h_S,a_S)
\right)
\le\varepsilon_O,
\]

and

\[
d_C
\left(
C_R(h_R,a_R),
C_S(h_S,a_S)
\right)
\le\varepsilon_C,
\]

after both sides are interpreted through the shared abstraction.

The tolerances name scope; they need not initially be numerical. Their purpose
is to prevent `corresponds` from quietly meaning `reminds us of`.

When the relation is required mainly from synthetic experience toward real
activity, **approximate process homomorphism** is the natural term. When both
processes preserve relevant distinctions and transitions in both directions,
the relationship approaches **bisimulation**.

---

## 5. Three correspondence claims

The word `correspondence` hides three distinct claims.

### Designed structural correspondence

Real and synthetic processes admit explicit maps into a shared abstraction, and
the maps preserve the stated dynamics.

### Representational correspondence

Learner-facing synthetic notation and a later natural interface express the
same abstract states and actions through an explicit bridge.

### Learned developmental correspondence

Organization acquired in the synthetic realization remains useful when the
learner encounters the natural or real realization.

The first two can be documented before building a world. The third is a claim
about learning in a particular model. A clean process diagram does not by
itself determine what organization gradient descent will construct.

---

## 6. Choosing a representation

After the abstract process is stated, choose the learner-facing representation
through five obligations.

### Semantic faithfulness

Different abstract states or actions that matter must not collapse into the
same uncontrolled token expression. A canonical renderer and parser should
recover the typed object or a typed ambiguity:

\[
d_e(r_e(z))=z.
\]

### Developmental learnability

Syntax must contain stable regularities a weight-naive model can acquire.
Arbitrary per-episode grammars force repeated interface inference instead of
world learning. This is why the project uses a stable home rendering before
broad variation.

### Compositional locality

Relations that compose in the abstract process should have local, learnable
expression. A representation that turns a simple state update into a global,
entangled rewrite obscures the organization it is supposed to teach.

### Gradient affordance

The notation should make local targets serializable. Action boundaries, state
updates, uncertainty, corrections, and commitments need typed places in the
sequence rather than one undifferentiated answer string.

### Translation bridge

The canonical form should support aligned renderings into natural language,
tool schemas, code, or other later interfaces. Alignment preserves the
abstract object; paraphrase alone does not.

No notation minimizes every cost. A concise language may be hard to acquire; a
verbose one may waste tokens; an overexplicit one may expose structure that the
learner should construct. Representation is part of developmental world design,
not an afterthought.

---

## 7. The correspondence dossier

Before calling a synthetic world a counterpart of a real activity, create a
correspondence dossier. It is a documentary and theoretical object, not an
experimental gate.

It contains:

1. Practice boundary
2. Real process description
3. Shared abstraction
4. Real history and action maps
5. Synthetic history and action maps
6. Preservation argument for dynamics, information, costs, goals, and recovery
7. Declared omissions
8. Canonical representation, home rendering, and natural bridge
9. Gradient account: which local targets privileged state permits
10. Developmental claim: which reusable learner organization the world is
    intended to cultivate

This is where “world A corresponds to activity B in representation C” becomes
an inspectable argument. Primitive names may help describe the dossier. They
cannot substitute for it.

---

## 8. Conceptual example: diagnosis

Take the practice:

> Select the next diagnostic observation while maintaining several live
> explanations for a failing service.

The real process contains hidden causes, logs, tools, interventions, changing
deployments, and recovery from bad changes. The abstract process contains live
explanations, evidence status, available inspections, commitments, and validity
epochs. A synthetic dependency system can instantiate hidden faults, probes,
interventions, costs, and validity changes without pretending to be a Linux
server.

The correspondence concerns the organization of diagnosis, not factual
knowledge of operating systems, networking, organizational escalation, or the
surface signature of real incidents.

A stable home rendering could express an abstract state as:

```text
OBSERVE metric(component_3) -> elevated
BELIEF {fault_2: possible, fault_5: possible}
AVAILABLE {inspect(component_2), revert(change_4), commit(fault_2)}
```

Aligned later renderings may use terminal output or natural incident language.
The representation changes; the abstract object is held fixed.

---

## 9. Final position

Worlds are not chosen by a controller, primitive list, or formal aesthetics.
They are chosen through an explicit chain from real practice to a
capability-relevant abstraction and then to a controlled synthetic realization.

A synthetic world earns the right to stand for an activity only to the extent
that it preserves the selected action–observation–consequence organization.
Its representation earns the right to teach that world only when it is
faithful, learnable, compositional, targetable, and bridgeable to later forms.

The central rule is:

> Correspondence precedes curriculum. A tutor may decide when a learner sees a
> world; it may not decide what that world means.
