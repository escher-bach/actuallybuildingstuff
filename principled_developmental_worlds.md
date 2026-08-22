# Toward a Principled Algebra of Developmental Worlds

### Status of this document

This is a longer-horizon proposal about what may eventually be extracted from
successful developmental processes. It does not authorize a primitive
inventory before the first discriminating probe. The current research order is
defined in [PROCESS-AND-RENDERING.md](PROCESS-AND-RENDERING.md).

## The core problem

The main difficulty is not merely generating many synthetic environments.

The deeper problem is:

> **How can developmental worlds be generated in a principled way, rather than as an ad hoc collection of settings, tasks, or games?**

A tempting approach is:

> invent lots of settings → put objects in them → invent quests → generate successful traces.

But this quickly becomes arbitrary. The design process starts revolving around whether the learner should encounter kitchens, factories, mazes, shops, circuits, text adventures, and so on.

For a developmental synthetic-pretraining project, that is the wrong level of abstraction.

The generator should not primarily generate *worlds in the ordinary narrative sense*.

It should generate **abstract causal processes first**.

---

## 1. Start from the general world object

A developmental world can be represented as

\[
W = (S, A, O, T, \Omega, G)
\]

where:

- \(S\) is the space of world states;
- \(A\) is the space of learner actions;
- \(O\) is the space of observations;
- \(T(s' \mid s,a)\) determines how actions change state;
- \(\Omega(o \mid s)\) determines what the learner observes;
- \(G\) determines continuation, completion, commitment, or abandonment.

This is almost completely content-free.

That is useful.

Instead of deciding that the learner should inhabit a "room," "laboratory," "computer," or "shop," the generator can operate directly over these abstract objects.

For example:

```text
hidden state:
    x ∈ {A,B,C,D}

available actions:
    query(q1)
    query(q2)
    transform(t1)
    commit(A/B/C/D)

rules:
    q1 reveals one function of x
    q2 reveals another function of x
    t1 changes x according to a known/unknown mapping

goal:
    commit to the correct state
```

There is no kitchen, key, computer, or real-world ontology here.

It is simply an **interactive information process**.

---

## 2. Generate worlds using structural operators

A key idea is to treat developmental dimensions not merely as descriptive coordinates, but as **operators that generate new worlds**.

Suppose we begin with a simple fully observable deterministic process \(W_0\).

We can transform it with operators such as:

```text
HIDE(variable)
```

which changes the observation function \(\Omega\) and introduces partial observability.

```text
ADD_QUERY(variable)
```

which adds an information-gathering action.

```text
MAKE_ACTION_IRREVERSIBLE(a)
```

which introduces commitment.

```text
ADD_COST(a)
```

which makes information gathering or intervention non-free.

```text
NOISE(observation, p)
```

which introduces uncertainty that cannot necessarily be eliminated by one observation.

```text
SWITCH_RULE(condition)
```

which makes previously learned structure cease to apply under some condition.

```text
DELAY_EFFECT(a, n)
```

which separates an action from its consequence.

```text
COUPLE(x, y)
```

which causes an intervention on one part of the state to affect another.

```text
LIMIT_RESOURCE(r, k)
```

which creates planning under scarcity.

```text
ALLOW_UNDO(a)
```

or

```text
FORBID_UNDO(a)
```

which controls recovery structure.

```text
CHANGE_RENDERER(R)
```

which changes representation without changing world semantics.

```text
REMOVE_SCAFFOLD(k)
```

which transfers some control function from teacher or scaffold to the learner.

The generator can therefore be viewed as applying transformations:

\[
W_{n+1} = F_i(W_n)
\]

where each \(F_i\) comes from a **small algebra of transformations on interactive processes**.

That is much more principled than simply generating more game settings.

---

## 3. A developmental-world language

A natural implementation is a small domain-specific language whose semantic types include things like:

```text
StateVariable
LatentVariable
Observation
Action
Transition
Constraint
Goal
Resource
Event
Renderer
Scaffold
```

and whose combinators include:

```text
hide
reveal
query
intervene
compose
condition
delay
invalidate
branch
couple
constrain
randomize
undo
commit
```

A generated world might look like:

```text
state x : Finite[4]
state y : Finite[3]

hide(x)

action inspect_x:
    observe partition(x, {{0,1}, {2,3}})

action modify_y(v):
    y := v

action test:
    observe relation(x, y)

goal:
    commit(x)

constraint:
    at most 3 inspections
```

The textual environment the language model experiences is then **compiled from this semantic program**.

This suggests that the developmental space should be generated from an algebra of:

- state transformations,
- observation processes,
- action opportunities,
- goals,
- uncertainty structures,
- compositions,
- representational renderings,
- scaffolding.

Named task families can exist as examples, but they should not define the outer boundary of the space.

---

## 4. The deepest problem: where do the primitives come from?

It is not enough to invent primitives such as:

```text
hide
query
commit
recover
```

because that could simply move the ad hoc design problem down one level.

A principled system needs a criterion for admitting a primitive into the world language.

One strong criterion is:

> **A primitive should have multiple independent real-world witnesses.**

Suppose we propose the primitive:

```text
QUERY
```

We should not justify it merely by saying that querying seems useful.

Instead, identify several substantially different real-world activities:

```text
debugging:
    run diagnostic → gain evidence

science:
    perform experiment → gain evidence

research:
    search source → gain evidence

medicine:
    order test → gain evidence

tool use:
    call read-only endpoint → gain evidence
```

Now strip away the surface objects.

The proposed shared abstract process is:

\[
\text{belief state}
\xrightarrow{\text{choose observation}}
\text{evidence}
\xrightarrow{\text{update}}
\text{new belief state}
\]

The primitive is therefore not admitted because it sounds cognitively important.

It is admitted because multiple real practices can be mapped to the **same abstract transformation**.

```text
real practice A ──┐
real practice B ──┼──> same abstract transformation
real practice C ──┤
real practice D ──┘
```

That gives a concrete methodology for discovering a developmental world algebra.

---

## 5. Some knowledge of the real world is unavoidable

There probably cannot be a purely mathematical derivation of:

> "Here is the uniquely correct childhood for a general intelligence."

A POMDP-like formalism tells us what interactive worlds *can be*.

It does not tell us which structures matter for real intellectual activity.

So an empirical anchor is unavoidable.

But the anchor does not need to consist mainly of facts such as:

```text
apples are edible
keys open locks
fridges contain food
```

Instead, the project can use knowledge about the **structure of human activities**:

```text
debugging often involves:
hypotheses → tests → observations → revision

planning often involves:
state → possible actions → irreversible commitments → consequences

research often involves:
uncertainty → information acquisition → source comparison → conclusion

software work often involves:
persistent state → intervention → failure → diagnosis → recovery
```

In this view, actual-world knowledge enters at the **correspondence layer**, rather than having to be embedded directly into every synthetic environment.

That is a much cleaner role for real-world grounding.

---

## 6. Separate the process generator from the renderer

The system should have at least two levels.

```text
LEVEL 1 — developmental process generator

typed states
hidden variables
transformations
observations
dependencies
uncertainty
resources
commitments
recovery
composition

        ↓ compile

LEVEL 2 — renderer

abstract symbolic notation
different symbolic notation
simple English
fictional domain
technical domain
tool-like API
eventually real interface
```

The renderer should **not invent mechanics**.

For example, the underlying operation

```text
inspect(X)
```

might be rendered as:

```text
measure node X
```

or

```text
question witness X
```

or

```text
read record X
```

but all three should execute the same semantic operation underneath.

This keeps executable meaning separate from surface form.

That separation is essential for studying world transfer and representation transfer independently.

---

## 7. The trace problem becomes much easier

The project does not necessarily need to generate "reasoning traces" in the modern LLM sense.

The world already contains the ground-truth computation.

Suppose a generated world has:

```text
hidden state = H3

belief-compatible states after history:
{H2, H3, H5}

actions:
q1 -> partitions {H2,H3,H5} as {{H2}, {H3,H5}}
q2 -> partitions them as {{H2,H5}, {H3}}
```

The generator knows this structure exactly.

It can therefore mechanically generate supervision for things such as:

```text
current possible states
action validity
predicted action consequences
possible observations
updated possible states
goal satisfaction
whether commitment is currently justified
what actions preserve recoverability
```

No language model is required to invent a plausible explanation after the fact.

For sufficiently small generated worlds, the system can even compute the full transition graph.

Instead of declaring:

```text
THE CORRECT TRACE:
q2 → inspect → commit H3
```

the teacher may know:

```text
optimal policies = {...}
valid policies = {...}
recoverable actions = {...}
dominated actions = {...}
```

This makes it possible to supervise the learner without pretending that there is always one canonical chain of reasoning.

The primary trace is then simply the **actual unfolding interaction**:

```text
state as observed
action taken
consequence received
belief or working state carried forward
next action
```

World trace, teacher trace, and learner trace can remain separate objects.

---

## 8. The world compiler can produce both environment and teacher

A single semantic program can generate both the interactive environment and the supervision oracle.

```text
             DEVELOPMENTAL PROGRAM
                     │
              ┌──────┴──────┐
              ▼             ▼
         WORLD ENGINE      ORACLE
              │             │
model action ─┤             ├─ known latent state
              │             ├─ valid actions
              │             ├─ consequences
              │             ├─ reachable states
              │             ├─ policy sets
              │             └─ local targets
              ▼
        next observation
```

This is particularly attractive for synthetic pretraining because **the supervision falls out of generation**.

You do not need three independent systems for:

1. world generation,
2. labeling,
3. trace generation.

The semantic program can provide all three.

---

## 9. A concrete meaning of "principled"

A useful way to prevent the ontology from becoming arbitrary is to impose admission tests on every proposed primitive or world operator.

### 9.1 Semantic stability

An operation must have the same executable meaning wherever it appears.

For example, if `QUERY` means "take an action that reveals evidence without directly changing the target state," that meaning should remain stable across worlds.

### 9.2 Compositional closure

The operator must combine with other operators to produce new meaningful processes.

It should not merely encode one handcrafted task.

### 9.3 Multiple correspondence

There should be mappings from several substantially different real activities into the same abstract structure.

A primitive should be justified by recurrence across practices, not by intuition alone.

### 9.4 Transfer consequence

Ultimately, pretraining on some realizations of that structure should reduce the amount of experience required to learn held-out realizations.

For example:

```text
synthetic QUERY worlds
        ↓
transfer test on
debugging / research / diagnosis / unfamiliar synthetic renderings
```

If the expected transfer does not occur, then the proposed primitive may not be capability-relevant in the way the theory predicted.

This keeps the ontology empirically corrigible.

---

## 10. The resulting research architecture

The overall research program could look like this:

```text
                 REAL PRACTICES
             /       |       |      \
       debugging  research planning  tool use
             \       |       |      /
              \  correspondence  /
                       ↓
             ABSTRACT PROCESS MOTIFS
                       ↓
              typed world algebra
                       ↓
       ┌───────────────┼────────────────┐
       ↓               ↓                ↓
    sample           compose          mutate
   structures       structures       coordinates
       └───────────────┼────────────────┘
                       ↓
               SEMANTIC PROGRAM
                  /          \
                 /            \
          world engine      oracle
               |              |
               └─────┬────────┘
                     ↓
                 renderer
                     ↓
                  tokens
                     ↓
                     LM
```

This changes the central research question.

Instead of asking:

> **What cool synthetic environments can we make?**

the project asks:

> **Can we discover a small compositional language of interaction structures that**
>
> 1. **has precise executable semantics,**
> 2. **generates an enormous space of developmental experiences, and**
> 3. **admits defensible correspondence maps to many real intellectual activities?**

That is a much more principled scientific program.

---

## 11. Why this differs from simply building more TextWorld-like environments

A procedural text-world generator is useful, but it usually inherits its ontology from the type of environment being generated.

For example:

```text
rooms
objects
containers
doors
quests
commands
winning policies
```

Those are meaningful because they make sense for a text-adventure game.

The developmental-world project should instead make its basic generative vocabulary correspond to **reusable structures of interaction and learning**.

Examples might include:

```text
hidden state
information acquisition
state intervention
commitment
reversibility
resource limits
delayed consequences
rule change
uncertainty
recovery
composition
translation
```

The distinction is:

```text
Text-world style generation:
    generate many instances of a world ontology

Developmental generation:
    generate many world ontologies from a shared process algebra
```

The second is much closer to what synthetic pretraining requires if the goal is broad transfer.

---

## 12. The immediate next research object

The next thing to design should **not be a general world implementation or a
large operator inventory**.

The immediate object is one discriminating process-and-representation probe:

1. state one future-relevant process distinction without assuming a storage
   substrate;
2. specify exactly what the learner and the privileged teacher can observe;
3. render the same presented process in two forms;
4. construct a surface-similar control in which the crucial consequence is
   changed; and
5. state which results would distinguish process learning, surface learning,
   ontology leakage, and failure to learn.

Only distinctions required by this probe should be formalized. Candidate
operators such as `HIDE` remain provisional descriptions until their presence
changes possible histories and predicts a measurable learning contrast.

The governing decision is defined in
[PROCESS-AND-RENDERING.md](PROCESS-AND-RENDERING.md).

---

## Conclusion

The central insight is:

> **Do not begin by generating worlds. Begin by generating structured interactive processes.**

Worlds should preserve stated process distinctions under intervention.

Real-world practices should constrain the algebra through correspondence.

Renderers should provide surface diversity without changing underlying semantics.

An implemented process should eventually support:

- the environment,
- the interaction dynamics,
- the oracle,
- the local supervision,
- and multiple representations.

Then "principled world generation" stops meaning "we chose sensible-looking synthetic tasks."

It means that the claimed semantics, the learner's information, and the
surface expression are controlled separately enough that success and failure
can be attributed. A general process language may later be extracted from
repeated successful probes; it is not the prerequisite for the first one.
