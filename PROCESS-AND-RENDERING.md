# Process Meaning and Learner-Facing Representation

## The minimum foundation before implementation

### Status of this document

This document fixes the project's immediate research commitment. It does not
select a world backend, propose a complete ontology of intelligence, or define
a catalogue of developmental primitives.

The decision is:

> Begin with the smallest account of process meaning needed to distinguish
> actions and consequences, together with a controlled account of how that
> meaning reaches the learner. Choose an execution substrate only after one
> such distinction can be stated and tested without building a general system.

Relational schemas, transition graphs, programs, and sandboxes remain possible
ways to execute a process. None is part of the foundation.

---

## 1. What earns foundational status

A commitment belongs in the foundation only when:

1. every credible first experiment requires it;
2. removing it makes the experiment uninterpretable;
3. it has an observable experimental consequence;
4. it does not privilege an unnecessary implementation; and
5. postponing it would merely hide the same decision elsewhere.

This admits persistent process meaning and learner-facing representation. It
does not admit relational state, a universal operator inventory, a backend
protocol, or a final surface grammar.

---

## 2. A minimal operational ontology

Here **ontology** means the distinctions the generated experience treats as
real. It does not mean an inventory of concepts presumed to be basic to
intelligence.

An ontological distinction is admitted only when changing it changes at least
one of:

- an action available to the learner;
- the observation produced by a history or intervention;
- a persistent consequence;
- whether the process continues, completes, or fails; or
- a legitimate teacher target.

If a proposed distinction changes none of these, it is currently explanatory
decoration and should be omitted.

The minimum process description therefore records only:

- histories or states that can differ in future-relevant ways;
- learner interventions;
- observations available after those interventions;
- persistent consequences;
- continuation conditions; and
- privileged information used by the generator or teacher.

This can be written as a transition table, a small mathematical object, or a
few counterfactual traces. It need not first be encoded as predicates, objects,
fluents, or a programming language.

### Names do not establish ontology

Terms such as `QUERY`, `COMMIT`, `RECOVER`, and `BELIEF` are provisional
descriptions. A term earns a reusable semantic role only when a controlled
change to the named organization changes possible histories and predicts a
measurable difference in learning or transfer.

For example, `COMMIT` should not be admitted because commitment sounds
important. First compare two otherwise similar processes: in one an action is
reversible, while in the other it removes future alternatives. The difference
in possible histories is the semantic content. The name may come later.

### Designer state is not automatically learner content

The generator may represent distinctions that the learner is expected to
discover. Those distinctions need not be named or directly serialized for the
learner. Exposing a designer variable as a target can turn abstraction
formation into vocabulary acquisition.

Every proposed learner-visible field must therefore say whether it is:

- part of the observation;
- a temporary scaffold;
- a prediction target;
- an action expression; or
- privileged teacher information that is never shown directly.

---

## 3. Presentation and rendering

Two transformations must not be conflated.

**Presentation** selects which process information the learner receives.
Changing presentation may change the information available and therefore the
developmental experience.

**Rendering** expresses a fixed typed observation or action in tokens. A change
of rendering changes expression without silently changing process mechanics or
available information.

A renderer and parser should satisfy the following obligations.

### Faithfulness

Different process distinctions that matter to the current experiment must not
collapse accidentally. Canonical expressions should parse back to the same
typed object; controlled ambiguity must be represented as ambiguity rather
than resolved by convention.

### Learnability

The initial rendering should contain stable local regularities. Arbitrary
per-episode grammars would repeatedly test interface induction instead of the
process distinction under study.

### Non-leakage

Object names, ordering, delimiters, trace length, and privileged labels must
not reveal the intended action or hidden state except where that revelation is
an explicit scaffold.

### Consequence invariance

When two expressions render the same typed action in the same presented
situation, they must induce the same consequence. If mechanics or available
information change, the result is a different process or presentation, not
merely another rendering.

### Gradient affordance

The rendering must allow the intended local targets to be serialized without
requiring the learner to emit a fabricated unique explanation. This does not
authorize exposing every privileged state variable.

The exact grammar, delimiter scheme, vocabulary, tokenizer interaction, and
natural-language style remain deferred until a concrete probe requires them.

---

## 4. The first discriminating probe

Before selecting a backend, specify one small process contrast on paper:

1. one process with a future-relevant hidden distinction;
2. two renderings of the same presented process;
3. one surface-similar process in which the crucial consequence is changed;
4. the learner-visible transcripts and privileged teacher information; and
5. the outcomes that would distinguish process learning from surface learning.

The required comparison is:

| | Rendering A | Rendering B |
|---|---|---|
| Same process | acquisition condition | representation-transfer condition |
| Changed process | semantic discrimination control | optional joint control |

Two renderings without the changed-process control are insufficient. A learner
may align vocabulary or layout without representing the process distinction.
The control asks whether behavior follows consequences when surface cues are
held as similar as practical.

The paper specification is complete when an independent reader can determine:

- which interventions are possible at each history;
- what each intervention changes;
- what the learner can and cannot observe;
- what remains invariant across the two renderings;
- which surface shortcuts are plausible; and
- what result would count against the proposed distinction.

---

## 5. What remains deferred

The first probe does not require decisions about:

- a universal primitive or operator basis;
- relational, functional, causal, or graph-rewrite state;
- a common backend interface;
- a repository module layout;
- large domain families;
- unrestricted program generation;
- a curriculum controller; or
- claims of ontological completeness.

An execution substrate should be selected only after the probe is coherent.
The selection criterion is then practical: implement the stated distinctions
with the least semantic distortion and the least machinery. A relational
schema may win that decision. An explicit transition table may win instead.

Generalization comes later. A distinction may be proposed as a reusable
operator only after it retains one executable meaning across independently
constructed processes, survives a change of representation or substrate, and
has a measurable transfer consequence.

---

## 6. Decision rule

The project currently commits to controlled process distinctions and
controlled learner-facing representation.

It does not yet commit to the internal furniture of all developmental worlds.
The first implementation should be evidence for or against one semantic and
representational claim, not the installation of an untested ontology.
