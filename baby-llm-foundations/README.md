# A Textual Childhood for Language Models

## Foundations for a developmental theory of synthetic pretraining

### Status of this document

This is a foundations document for a new project. It states the goal of the
project and develops the theory needed to speak about that goal clearly. It is
not an experimental plan, a benchmark proposal, a catalogue of tasks, or a
commitment to a particular implementation.

Several ideas were inherited from an earlier task-repertoire project. Cleaned
versions are preserved under [legacy/](legacy/README.md), with explicit
warnings that they are historical inputs rather than current foundations. In
particular, the old candidate primitives, task-family formalism, and
epiplexity-based framework must not be read as already establishing the
developmental or real-world correspondence claims made here.

The project begins from a deliberately simple question:

> If a language model began with no learned knowledge, what kind of experience
> should it receive in order to grow into a useful general agent?

The phrase **baby language model** names this point of view. It asks us to stop
starting with a capable pretrained model and then repairing its behavior. It
asks what must be learned in the first place, in what representational world it
can be learned, and why learning it there should matter outside that world.

“Baby” is an analogy, and only an analogy. A newly initialized language model
is not a human infant. It has an architecture, tokenizer, context window,
optimizer, and training rule before it sees any data. Those are substantial
prior commitments. The useful content of the analogy is narrower: the model's
weights do not yet contain a learned world, learned notation, learned language,
or learned repertoire of action. It must acquire all of them through the
experience we provide.

---

## 1. The goal

The goal is to construct a high-throughput developmental world for a
weight-naive language model.

The world is presented through tokens. It contains persistent situations,
objects, rules, actions, observations, consequences, uncertainty, mistakes,
recovery, and the formation of reusable abstractions. It is not limited to a
finite list of tasks. It is generated algorithmically, so that the model can
encounter indefinitely many situations while the underlying structure remains
under human inspection.

The intended result is not merely a model that performs well inside a
synthetic notation. The intended result is a model whose acquired ways of
learning, acting, representing state, and organizing experience remain useful
when the surface world changes.

In one sentence:

> Build an indefinitely generable textual world in which a weight-naive model
> can acquire the reusable organization needed for real intellectual and
> agentic activity.

This sentence contains four claims that must be kept distinct:

1. A language model can interact with a world through a token interface.
2. A token interface can carry more than static text: it can carry an ongoing
   environment.
3. A synthetic environment can share relevant organization with real
   activities without resembling them superficially.
4. Experience in the synthetic environment can shape a learner in ways that
   remain useful across changes of representation and domain.

The project is principally concerned with the third and fourth claims. The
first two make the project possible; they do not make it true.

---

## 2. Why begin with the learner rather than with a task catalogue

Most synthetic-data projects begin by asking which tasks to generate. That
question quietly assumes that the useful units of learning are already known.
The result is usually a collection of categories: arithmetic, copying, stack
manipulation, deduction, tool use, planning, and so on.

A category is useful for organizing a spreadsheet. It does not by itself say
what changes in the learner.

The baby-model framing reverses the direction of explanation. It asks:

- What does the learner initially lack?
- What kind of experience would make that organization necessary?
- What must remain stable across experiences for learning to accumulate?
- What may vary without forcing the learner to begin again?
- What relationship must hold between the developmental world and later
  worlds?

This shift matters because the same visible task can support very different
learning. A child solving ten arithmetic questions may be learning addition,
the written notation for addition, the teacher's expected response format, or
a local pattern that happens to work on those ten questions. The activity name
does not identify the acquired capability.

Likewise, two activities with different names may exercise the same underlying
organization. Debugging a circuit and debugging a program may both require a
belief about hidden causes, an informative intervention, and revision after an
unexpected observation. Their surface objects differ. Their interactive
structure may be closely related.

The project therefore begins with a theory of developmental experience, not a
finite taxonomy of tasks.

---

## 3. The model's boundary is made of tokens

A text-only language model receives token identifiers and produces token
identifiers. Everything the model knows about an external system must cross
that boundary.

This does not mean that the external system itself must be textual. A camera,
browser, database, robot, theorem prover, or another model may stand outside
the boundary. What matters to the central learner is that observations arrive
in a tokenized form and actions leave in a tokenized form.

It is useful to distinguish three objects:

- **World:** whatever has state and changes over time.
- **Interface:** the mechanism that turns selected facts about the world into
  model-readable observations and model outputs into world-directed actions.
- **Learner:** the sequence model whose weights change through experience.

The interface is not the world. A terminal is an interface to a computer. A
caption is an interface to an image. A tool schema is an interface to a
service. Confusing an interface with the world makes it easy to generate the
appearance of interaction without its substance.

### 3.1 Text as a sufficient interface

Text is sufficient for an activity when the textual interface preserves the
information and actions needed for that activity.

This is a conditional statement, not a universal one. A textual description
of a picture may preserve everything needed to decide whether a traffic light
is red, while losing what is needed to align a robotic gripper with a small
object. The issue is not whether text is metaphysically capable of describing
the scene. The issue is whether the available interface supplies the right
information at the right resolution, with a way to request more when needed.

A useful textual interface is therefore often interactive. The model can ask
for a crop, inspect an object, query a property, run a command, or request a
second description. In such a system, perception is modular: another component
may perform visual or acoustic processing while the central learner reasons
over a token channel.

The project's commitment is consequently modest and strong:

> The developmental world need only meet the learner at a token boundary. It
> need not imitate every physical modality inside the learner.

---

## 4. A world is not a dataset

A dataset is a collection of records. A world is something whose next state
depends on what happens in its current state.

A static question-and-answer pair can be serialized as:

```text
question -> answer
```

An interaction has a different form:

```text
observation -> action -> consequence -> new observation -> ...
```

The difference is causal. In a dataset, the learner reads a trace that already
exists. In a world, the learner's action helps determine which trace comes to
exist.

This distinction is central to the project. A corpus of successful tool-use
transcripts may teach the surface regularities of tool use. It does not by
itself give the model experience of choosing a poor action, receiving the
actual consequence of that action, recognizing the resulting state, and
recovering from it.

### 4.1 A first formal object

Only now is a formalization useful.

Represent a developmental world as:

\[
W = (S, A, O, T, \Omega, G).
\]

Here:

- \(S\) is the space of world states;
- \(A\) is the space of actions available to the learner;
- \(O\) is the space of observations;
- \(T(s' \mid s,a)\) describes how action \(a\) changes state \(s\);
- \(\Omega(o \mid s)\) describes what the interface reveals about a state;
- \(G\) describes what counts as continuation, completion, commitment, or
  abandonment within the situation.

The model does not directly receive \(s\). It receives a token rendering of an
observation \(o\), and it emits a token rendering of an action \(a\). Because
the state is only partially visible, the model must often carry a
representation of its interaction history.

This definition is intentionally broad. A theorem-proving dialogue, a hidden
rule game, a shell session, a planning world, and a debugging process can all
be instances. Their differences live in their state, action, observation, and
transition structure rather than in separate notions of “task.”

### 4.2 A world does not determine the learning rule

The language of states, actions, and consequences can make the project sound
like reinforcement learning. That conclusion does not follow.

A world says how an interaction unfolds. A learning rule says how experience
changes the model. These are separate objects.

In this project, the intended learning rule is **dense token supervision**. At
many points in an interaction, a teacher that knows the generated world state
can supply a local target distribution for the tokens the model is currently
producing. Gradient reaches those tokens directly through cross-entropy. It
does not have to travel backward from a scalar reward assigned to the completed
trajectory.

The model may still influence what happens next. If it emits an inspection
action, the world returns the result of that inspection. If it emits a malformed
action, the world returns the consequence of that malformed action. The
resulting experience may therefore be learner-conditioned without being
reward-conditioned.

This gives the project four distinct layers:

1. **World dynamics:** what follows from an action.
2. **Presentation:** what part of the resulting world is shown.
3. **Scaffolding:** which parts of control are supplied and which are left to
   the learner.
4. **Gradient generation:** which model tokens receive which local learning
   targets.

The detailed definition is developed separately in
[GRADIENT-PARADIGM.md](GRADIENT-PARADIGM.md). Keeping it separate prevents the
ontology of developmental worlds from being silently identified with one
training algorithm.

The computational consequences are developed in
[TRAINING-DYNAMICS.md](TRAINING-DYNAMICS.md). The procedure by which synthetic
worlds are related to real activities is developed separately in
[WORLD-CORRESPONDENCE.md](WORLD-CORRESPONDENCE.md). The former concerns the
economics of gradient generation; the latter concerns what those gradients
should be about.

The immediate research boundary is defined in
[PROCESS-AND-RENDERING.md](PROCESS-AND-RENDERING.md). Before choosing an
execution substrate, the project must state one future-relevant process
distinction, control what information reaches the learner, render the same
presented process in more than one form, and construct a surface-similar
semantic control. This is the minimum needed to distinguish process learning
from interface learning.

The first concrete implementation boundary is specified in
[STEP-1.md](STEP-1.md). It selects evidence acquisition under irreversible
commitment as the first controlled process family, dense teacher traces as the
primary learning path, outcome-only RLVR as a baseline, and a high-throughput
Rust executor with a batched Python boundary.

Stage results are reported separately from the specifications that produced
them. Two stages have run: representation transfer, in
[RENDERING-B-TERMINAL-TRANSFER-REPORT.md](RENDERING-B-TERMINAL-TRANSFER-REPORT.md),
and the outcome-only verified-learning baseline, in
[RLVR-STAGE-REPORT.md](RLVR-STAGE-REPORT.md). How runs are launched, collected,
and audited is in [step1/kaggle/RUNBOOK.md](step1/kaggle/RUNBOOK.md).

The selected Step 1 executor must preserve those stated semantics, as bounded
in [WORLD-BACKEND.md](WORLD-BACKEND.md). A compact finite transition executor
implemented in Rust is selected for the first probe only; it is not the
project's general world ontology. Relational schemas remain a later candidate
described in [RELATIONAL-WORLDS.md](RELATIONAL-WORLDS.md), not the route by
which the project will discover its ontology.

### 4.3 The high-throughput condition

The developmental world must be capable of producing far more useful
experience than can be manually written.

High throughput does not mean repetition at high speed. Repeating one short
generator forever increases token count without necessarily increasing what
the learner can acquire. Throughput must include structural variation:

- new states;
- new compositions;
- new histories leading to similar states;
- new surface realizations;
- new consequences of action;
- new uncertainty about what is currently happening.

The generator should be short enough to inspect but rich enough that predicting
and acting within its output cannot be reduced to memorizing a finite table.
Its purpose is to convert computation into developmental experience.

In training terms, the scarce quantity is not raw synthetic tokens. It is
**gradient-bearing experience**: tokens for which the generated state makes a
useful local target available. A million narrated tokens followed by one final
verdict are sparse supervision. A shorter interaction in which prediction,
state maintenance, action selection, uncertainty, correction, and commitment
can each receive a local target is dense supervision.

---

## 5. The world may be an illusion

The developmental environment need not be physically real. It may be entirely
simulated, symbolic, or generated on demand.

Calling it an illusion is helpful if the analogy is not pushed too far. A
flight simulator is an illusion with respect to air, height, and physical
danger. It is not an illusion with respect to the relationship between many
control decisions and their modeled consequences. Its value lies in the
organization it preserves.

The relevant question is therefore not:

> Does the synthetic world look like the real world?

It is:

> Which relationships between observation, action, consequence, uncertainty,
> and goal are preserved?

A world rendered as a few symbols may preserve an important relationship more
cleanly than a photorealistic simulator. Conversely, a convincing textual
narrative may preserve almost nothing if it produces plausible continuations
without being governed by the learner's interventions.

This gives the project a preference for **causal fidelity over surface
fidelity**. Surface realism is useful only when the surface itself is part of
what must be learned.

### 5.1 Implementation illustration: a compiled symbolic world

The requirement of causal fidelity does not imply that the synthetic system
must dynamically implement the machinery it depicts. A world can be compiled
into extremely cheap symbolic state and lookup operations while retaining an
action-dependent interface.

For illustration, imagine an episode presented through a filesystem-like
interface. Internally, its "files" are only an indexed list of text cells. The
episode exposes a finite command vocabulary, and each command is implemented by
a small transition or observation table. A search command may retrieve
preselected cells; an ordering command may return a precomputed ordering; a
delete command may change a symbolic availability bit. No general filesystem,
search routine, sorting algorithm, or deletion mechanism need run during the
interaction.

An instance generator can begin with a canonical configuration and apply a
hidden permutation \(\pi\) to object identities or positions. Contents,
references, goals, observations, and command tables are transformed
consistently. The learner therefore encounters a newly rendered instance while
the abstract action semantics remain fixed. In effect, the per-instance world
can be compiled ahead of time:

```text
canonical symbolic process
        -> sample configuration and hidden renaming
        -> compile state, observation, and transition tables
        -> learner action
        -> table-defined consequence and new symbolic state
```

From the learner's boundary, the important fact is not whether an ordering was
computed online or stored in advance. It is whether different actions expose
the appropriate information, change persistent state, constrain later
actions, and permit relevant mistakes and recovery. Precomputation is an
implementation economy; it does not excuse a fixed transcript that ignores
the learner's intervention.

This illustration has several limits. A hidden permutation alone supplies
surface variation, not an indefinitely generable world or a correspondence to
real practice. The family must still contain meaningful variation in states,
histories, compositions, and consequences. A unique valid operation order may
be engineered when the modeled practice genuinely warrants one, but uniqueness
is not a general requirement and must not erase legitimate alternatives.

---

## 6. The correspondence problem

The project cannot be justified by analogy alone.

Suppose a synthetic activity and a real activity are both described as
“hypothesis elimination.” The shared phrase is evidence about our vocabulary,
not evidence that the learner performs the same computation in both cases.
The two activities may differ in how hypotheses are represented, how evidence
arrives, whether old hypotheses can return, what actions cost, and what happens
after a mistake.

The central theoretical problem is to state when two worlds contain the same
capability-relevant organization.

### 6.1 Shared abstract process

Let \(W_S\) be a synthetic world and \(W_R\) a real-world practice. We say that
they correspond with respect to some activity when interaction histories in
both worlds can be mapped into a shared abstract state space \(Z\), and their
actions can be mapped into a shared abstract action space \(U\).

Schematically:

\[
h_S \xrightarrow{\phi_S} z \xleftarrow{\phi_R} h_R
\]

and

\[
a_S \xrightarrow{\alpha_S} u \xleftarrow{\alpha_R} a_R.
\]

The shared state \(z\) is not a claim that the two physical states are alike.
It records only what is relevant to the activity under consideration. For
example, two states may correspond because the same three explanations remain
possible, even though one concerns a circuit and the other a software service.

The correspondence has substance when acting and translating agree. If an
action removes a possibility, exposes information, creates a commitment, or
changes what can be done next in one world, its counterpart should do the same
in the shared abstract process.

Informally, the following routes should agree:

```text
act in the concrete world, then translate the result
                     ~=
translate the situation, then apply the abstract action
```

This kind of structure-preserving map is related to a process homomorphism. If
the relationship preserves the relevant dynamics in both directions, it is
closer to a bisimulation. Those names are useful, but the underlying idea is
simple: correspondence concerns consequences under action, not resemblance
under description.

### 6.2 Correspondence is relative to a practice

No single abstraction must preserve everything.

A map adequate for choosing which diagnostic test to run may be inadequate for
physically repairing the machine. A textual visual description adequate for
planning may be inadequate for motor control. Correspondence is always stated
relative to the activities we want the learner to carry out.

This prevents an easy but damaging slide from “these worlds share a useful
structure” to “these worlds are the same.” They are not the same. A chosen
relationship between them may be the same at a particular level of
organization.

### 6.3 Internal coherence and external adequacy

Two requirements should not share one name:

- **Internal semantic coherence:** an operation used in several synthetic
  worlds has the same executable meaning in each.
- **External semantic adequacy:** that operation participates in a genuine
  correspondence with the real practices the project cares about.

Internal coherence makes a cumulative synthetic childhood possible. External
adequacy gives that childhood a reason to matter.

---

## 7. What a capability is

A capability is not a task label and not a score on one family of examples.

For this project, a capability is a reusable organization of the learner that
supports a class of related interactions. It may include:

- a way of representing relevant state;
- a way of updating that representation after observation;
- a way of selecting an action;
- a way of recognizing commitment, uncertainty, or completion;
- a way of composing these processes over time.

This definition is intentionally learner-centered. The world supplies
pressures and regularities; the capability is what the learner develops in
response.

The distinction explains why a generator does not simply “contain” the full
capability it helps produce. A generator may cheaply sample a hidden rule and
answer queries by direct evaluation. A bounded learner that sees only partial
evidence may need to construct a much richer procedure for inferring the rule.
The developmental demand can exceed the complexity of the teacher's generation
path.

---

## 8. Primitives: from names to reusable transformations

The word **primitive** is dangerous because it suggests that the basic units
have already been discovered.

A label such as `constraint propagation`, `backtracking`, or `abstraction`
may point toward a recurring pattern. It does not yet specify an operation.

Within this project, a candidate primitive should eventually denote a reusable
transformation with:

- an input state or representation;
- conditions under which it applies;
- an output state or observable effect;
- a role in longer compositions;
- a meaning that does not silently change between worlds.

This does not require every primitive to look like a line of source code.
Maintaining uncertainty, noticing a contradiction, or deciding that available
evidence is insufficient may be temporally extended organizations rather than
atomic instructions. “Primitive” means basic at the chosen explanatory level,
not physically indivisible.

### 8.1 The basis is not given in advance

The project should not begin by declaring a final set of primitives. The baby
framing makes that especially clear: development may create useful internal
units that are not transparent copies of the teacher's vocabulary.

The theoretical role of a primitive basis is to explain three things:

1. **Reuse:** why learning in one situation can matter in another.
2. **Composition:** how a finite organization can support unbounded activity.
3. **Accumulation:** why later experience can build on earlier experience
   instead of constituting another unrelated training phase.

Candidate primitives are proposals about the joints along which experience
accumulates. They are not the foundation of the theory; correspondence and
development are. The primitive vocabulary must remain subordinate to them.

### 8.2 A finite basis and an infinite childhood

A finite basis is not inherently unscalable. A small instruction set can
generate infinitely many programs. Natural language uses a finite vocabulary
and finite grammar to express unboundedly many thoughts.

The unscalable object is a finite catalogue of handcrafted families with no
generative closure between them.

The developmental world should instead be generated from an algebra of:

- state transformations;
- observation processes;
- action opportunities;
- goals and commitments;
- uncertainty structures;
- compositions;
- representational renderings.

Named families may remain useful as examples or regions of this space. They
should not be the outer boundary of the space.

---

## 9. Representation is part of development

A weight-naive model does not begin with semantics and then receive syntax as
decorative packaging. It learns through representations. Syntax determines
which regularities are visible, which compositions are local, and which
distinctions are easy to preserve.

The operational boundary between process meaning, presentation, and rendering
is fixed in [PROCESS-AND-RENDERING.md](PROCESS-AND-RENDERING.md).

This creates a tension between two sound intuitions.

The first says that a learner should not overfit to arbitrary notation. If the
same rule is rendered with different symbol names, punctuation, or ordering,
the useful organization should survive.

The second says that fluency in a stable representation is itself learned.
Teaching mathematics in one language, history in another, and science in a
third introduces a translation burden even when the underlying ideas are
unchanged. The burden is not imaginary merely because a short compiler exists
in principle. The learner still has to acquire and reliably execute that
compiler.

The project must preserve both insights.

### 9.1 Stable home, multiple views

A natural arrangement has several layers:

1. **Canonical semantic form.** A typed internal representation in which world
   states, actions, and transformations have one executable meaning.
2. **Stable home rendering.** A consistent surface language in which the model
   can acquire fluency and in which learning can accumulate globally.
3. **Aligned alternative renderings.** Other ways of expressing the same
   situations, presented with enough correspondence that translation itself
   can be learned.
4. **Open interfaces.** Later notations and tool protocols that can be related
   to the earlier representations rather than learned as unrelated worlds.

This is neither one immutable syntax nor arbitrary syntax randomization. It is
a representational ecology with continuity.

Incidental variation and structural variation should also be distinguished.
Renaming symbols while preserving a grammar is different from changing the
grammar. Changing the grammar is different again from changing which
distinctions the representation makes easy. All three may be valuable, but
they ask different things of a developing learner.

### 9.2 Local syntax and global meaning

If every world invents its own surface protocol while primitive names remain
global, then global semantics exists only in the designer's annotations. The
learner experiences repeated translation problems.

If every world uses one shared language but operations are implemented
differently underneath, shared syntax creates only the appearance of semantic
unity.

The desired condition is stronger:

> Shared operations should have shared executable meaning, and changes of
> rendering should be explicit transformations of that meaning.

One tokenizer is not enough. One vocabulary is not enough. The source of
coherence is the shared process semantics beneath the renderings.

---

## 10. Development as an expanding ecology

The baby analogy is often turned too quickly into a rigid curriculum: first
task A, then task B, then task C. Human development is not a checklist of
isolated lessons. Many competencies grow together because the same environment
repeatedly demands them in increasingly rich combinations.

It is more useful to imagine an expanding ecology of experience.

### 10.1 Interactional literacy

At the beginning, even the interface is not transparent. The learner must
acquire regularities such as:

- observations and actions occupy different roles;
- delimiters and types constrain what can follow;
- an action may receive a response;
- the world persists between turns;
- invalid actions produce structured consequences;
- earlier observations can remain relevant.

These may look like mere syntax from the perspective of a mature model. For a
weight-naive model they are part of learning what interaction is.

### 10.2 Stable transformations

The learner encounters objects and operations whose consequences remain
consistent across many situations. This supports prediction, execution, and a
sense that different episodes belong to one world rather than to unrelated
text distributions.

### 10.3 Hidden organization

Not everything relevant is stated. The learner must use repeated observations
to determine which rule, state, intention, or causal organization is currently
active.

### 10.4 Intervention

Some uncertainty cannot be resolved by passive observation alone. The learner
can act to reveal information, while recognizing that actions also change what
happens next.

### 10.5 Uncertainty and commitment

Available evidence may support several futures. The learner needs forms of
representation that preserve alternatives rather than collapsing immediately
to one answer. At other times it must commit, stop seeking information, or act
under unresolved uncertainty.

### 10.6 Change and recovery

Previously useful beliefs, plans, or partial solutions may cease to fit. The
learner must distinguish a locally surprising observation from a changed
world, an invalid action, a faulty assumption, or a failed line of work.

### 10.7 Abstraction and composition

Repeated organizations can become reusable units. Those units can then
participate in longer activities without being rediscovered from scratch.
This is the point at which a developmental world begins to support open-ended
growth rather than merely broader coverage.

### 10.8 Translation into new worlds

New representations and domains should become learnable as views on partially
familiar organization. The aim is not zero-cost transfer. Learning the new
surface is real work. The aim is that learning the surface should not require
relearning every underlying relation.

These are regions of development, not a prescribed sequence. They overlap and
support one another.

---

## 11. Four distinctions inherited from the earlier work

The project is conceptually separate from the earlier task-repertoire project,
but that work discovered four distinctions that belong in a developmental
theory. They are retained here without importing its task catalogue, level
system, or evaluation machinery.

### 11.1 Coordinates of developmental worlds

Developmental worlds should not be divided too quickly into named levels. A
single interaction can vary along several independent coordinates:

- **Revelation:** how much of the world's current organization is visible;
- **Initiative:** who chooses what happens next;
- **Validity:** how long an inferred rule or state remains applicable;
- **Action stakes:** whether an action merely obtains information or also
  changes what is being pursued;
- **Reversibility:** whether an action can be undone or requires recovery;
- **Uncertainty kind:** whether uncertainty comes from incomplete knowledge or
  genuine variation in the world;
- **Representation:** whether the interface is familiar, translated, or still
  being acquired;
- **Guidance:** which intermediate organization is supplied by a teacher or
  scaffold.

These are coordinates rather than mutually exclusive categories. A world can
occupy several positions over the course of one episode, and the same world can
be presented differently over the course of development.

If \(\mathcal{W}\) is the space of possible developmental worlds, we can think
of a descriptive map

\[
c : \mathcal{W} \to
\mathcal{R} \times \mathcal{I} \times \mathcal{V} \times
\mathcal{K} \times \mathcal{U} \times \mathcal{E} \times \mathcal{G},
\]

where the codomains describe revelation, initiative, validity, stakes,
uncertainty, representation, and guidance. The notation does not assert that
each coordinate is a single number. It records that these properties can vary
independently and should not be compressed into one ladder.

The developmental question is not “which level comes next?” It is “what ecology
of positions along these coordinates gives the learner coherent, accumulating
experience?”

### 11.2 Scaffolding and the location of agency

Agency is often attributed entirely to the model or entirely denied to it. In
practice, control is distributed across the learner, interface, teacher, and
scaffold.

A scaffold may:

- interpret the overall goal;
- divide it into subgoals;
- choose what information to expose;
- decide which tool or world to consult;
- preserve memory;
- notice that progress has stalled;
- initiate recovery;
- decide when to stop.

If the scaffold performs these operations, the overall system may behave
agentically while the model itself has learned only local completion. The
capability has not disappeared; it is located outside the learner.

Let \(K\) be a set of control functions relevant to an activity. At a moment in
development, define an ownership relation

\[
\omega_t : K \to
\{\text{world},\text{teacher},\text{scaffold},\text{learner},\text{shared}\}.
\]

This is not necessarily a clean partition. A teacher may propose a subgoal
while the learner decides whether to accept it. Memory may be partly internal
and partly external. The relation is intended to make the location of control
explicit.

Development can then include a migration of control. An operation first
supplied by the environment may later be prompted, then chosen with assistance,
then initiated by the learner. This is a more precise use of the baby analogy:
scaffolding is not the opposite of agency; it is one possible route by which
the organization of agency becomes internal.

The desired endpoint is not maximal internalization. External calculators,
memory stores, perception systems, and safety boundaries may remain desirable.
The important question is whether the allocation is deliberate or merely an
accident of the current harness.

### 11.3 World uncertainty versus representation uncertainty

A mature user of a notation can usually separate uncertainty about the problem
from uncertainty about how the problem is written. A weight-naive learner
cannot initially make that separation.

Let \(z\) denote hidden organization in the world and let \(e\) denote the
rendering or interface convention. Given token history \(h\), the learner faces
a joint uncertainty:

\[
H(Z,E \mid h)
=
H(E \mid h) + H(Z \mid E,h).
\]

The first term concerns representation: what roles the tokens play, which
strings are valid actions, and how distinctions are encoded. The second
concerns the world after interpreting the representation: which rule is
active, which state has been reached, or which explanation remains possible.

These uncertainties interact. An apparently surprising world observation may
actually reflect a misunderstood action syntax. An apparently malformed action
may reflect a correct intention expressed through the wrong interface. A model
may know the relevant algorithm while being unable to realize it in an
unfamiliar language.

This is why syntax cannot be randomized away indiscriminately. The baby model
must receive enough representational continuity to learn an interface, while
also encountering enough aligned variation to learn that the interface is not
the world itself.

### 11.4 Guidance as a learner-relative relationship

Guidance is not intrinsically helpful or obstructive. Its role depends on the
relationship among the learner's present organization, the world, and the form
of assistance.

Training wheels provide a useful analogy. They do not make balancing
unnecessary in the final activity, and removing them immediately does not teach
balancing more purely. They temporarily change which aspects of cycling the
learner can participate in. The analogy ends there: a language model's guidance
may operate through token targets, revealed state, constrained actions, or
teacher traces rather than physical support.

Let \(\kappa_t\) summarize the learner's current organization, \(w\) a region
of developmental experience, and \(g\) a form of guidance. The developmental
role of guidance is relational:

\[
R_g = R(\kappa_t,w,g).
\]

The same \(g\) may make an otherwise inaccessible activity reachable for one
learner state, organize attention for another, and replace an operation that a
more developed learner should perform itself.

Guidance includes more than written reasoning traces. It may determine:

- which state variables are visible;
- which actions are currently available;
- whether a goal is decomposed;
- whether an intermediate state is named;
- whether uncertainty is represented explicitly;
- whether correction follows immediately;
- whether the teacher supplies a local token target.

The developmental ecology should therefore contain changing relations of
guidance, not a universal rule that more or less guidance is always preferable.

---

## 12. Traces are experience, not explanations pasted onto answers

The word **trace** often means a written rationale placed before an answer.
That is only one narrow kind of trace.

For this project, the primary trace is the unfolding interaction itself:

```text
state as observed
action taken
consequence received
belief or working state carried forward
next action
```

Such a trace may contain intermediate computations, but it may also contain a
guess, an inspection, a malformed action, a correction, a return to an earlier
state, or a decision to stop.

Because the world is generated, its internal state may be available to the
teacher. This does not mean every hidden state should be narrated to the
learner. Development depends partly on what is withheld and must be inferred.
The trace is a controlled view of the process, not a dump of the generator's
implementation.

The important distinction is between:

- **world trace:** what occurred because of the interaction;
- **teacher trace:** an additional representation supplied to help organize
  that experience;
- **learner trace:** what the model emits as its own intermediate activity.

They may coincide in simple worlds. They should not be conceptually fused.

---

## 13. Natural language and formal language

Natural language is not merely a noisy renderer of a perfectly formal latent
world. It carries conventions, ambiguity, implicature, social expectations,
and historically accumulated distinctions. Those are part of real practice.

Formal languages offer different advantages. Their semantics can be explicit,
their consequences mechanically generated, and their compositional structure
controlled. They make it possible to provide vast quantities of coherent
experience without pretending that a human author labeled every uncertainty or
possible action.

The project should not choose between them as rival substrates. Their roles are
different:

- Formal representations make the developmental world's mechanics explicit.
- Natural language connects those mechanics to the representational medium in
  which much human activity is described and coordinated.
- Aligned renderings make the relationship learnable rather than assumed.

The long-term object is not a model trapped in a formal toy world, nor a model
asked to discover all structure from uncontrolled prose. It is a model that
can acquire organization in environments where consequences are clear and
then extend that organization into languages where meanings are less cleanly
delimited.

---

## 14. Existing language models and the baby model

Current frontier language models should not be treated as failed versions of a
more elegant symbolic solution. They are successful learners produced by an
enormous, historically accidental developmental corpus. Their abilities are
evidence about what large-scale next-token learning can acquire. Their
inefficiencies are evidence about the price of acquiring it from that corpus.

The baby-model project changes the educational environment, not the basic fact
that the learner is a language model.

Existing models can inform the theory because they reveal distinctions that a
designer might otherwise miss. A model may understand an algorithm yet fail to
express it in a new notation. It may reason well when a task is named but fail
to infer the task from interaction. It may use tools successfully under an
external scaffold while failing to decide for itself which observation to seek
next.

These are not reasons to define a list of defects and train against it. They
are reminders that capability, representation, interaction, and scaffolding
are different objects. The new project should preserve those distinctions from
the beginning.

---

## 15. What this project is and is not

This project is:

- a theory of developmental experience for weight-naive sequence models;
- an attempt to construct rich interactive worlds through a token interface;
- an account of how synthetic and real activities may share organization;
- an investigation of representation as part of learning, not as packaging;
- an algorithmic approach to generating unbounded experience from inspectable
  rules;
- a search for the reusable organization through which learning accumulates.

This project is not:

- a finite benchmark suite;
- a list of reasoning-task names;
- a claim that labels are capabilities;
- a claim that textual descriptions preserve every perceptual fact;
- a plan to imitate the current syntax of agent harnesses;
- a claim that one formal language already contains the joints of all useful
  activity;
- an attempt to replace language models with a hand-written symbolic agent;
- a theory in which syntax is dismissed as a negligible implementation detail;
- a rigid developmental ladder copied from an analogy with human childhood.

---

## 16. Foundational commitments

The theory can be condensed into ten commitments.

### C1 — The learner boundary

The central learner encounters its environment through tokenized observations
and tokenized actions. Other modalities may be handled by external components.

### C2 — Interaction before task labels

The basic unit of experience is a persistent action–observation process, not a
question-and-answer category.

### C3 — Causal fidelity before surface fidelity

A synthetic world is useful to the extent that it preserves relevant
consequences under intervention, not to the extent that it looks realistic.

### C4 — Correspondence before analogy

Two activities share a capability-relevant structure only when a
structure-preserving relationship can be stated between their histories,
actions, and consequences.

### C5 — Development before evaluation vocabulary

The project is organized around what a weight-naive learner must acquire, not
around names inherited from existing benchmarks or products.

### C6 — Representation is learned

Syntax, notation, action formats, and translation are parts of development.
Semantic invariance cannot be obtained by pretending these costs do not exist.

### C7 — Shared meaning requires shared semantics

Global primitive names and a shared tokenizer do not create coherence. Shared
operations require shared executable meaning beneath their renderings.

### C8 — Finite generators require generative closure

A compact inspectable system may support open-ended experience when its
operations compose over unbounded states, programs, histories, and
representations.

### C9 — Interaction does not imply reward learning

Worlds determine consequences, while a separate gradient generator supplies
dense local token targets. Learner-conditioned interaction and supervised
cross-entropy can coexist.

### C10 — The primitive basis remains subordinate

Primitives are proposals about reusable developmental organization. They must
not become labels used to justify a correspondence that has not otherwise been
explained.

---

## 17. The project in its strongest form

The strongest form of the idea is not that synthetic tasks can improve model
performance. That claim is already too weak and too tied to existing task
language.

The stronger idea is:

> A language model can have a designed childhood.

Its early world need not contain the full knowledge of human civilization. It
can instead contain dense, generative experience of persistence, consequence,
inference, intervention, uncertainty, recovery, abstraction, composition, and
translation. These experiences can be presented through a token interface
without reducing them to static prose. Their surface worlds may be artificial,
provided their organization corresponds to the real practices for which the
model is being prepared.

The project is the attempt to understand and construct that childhood.
