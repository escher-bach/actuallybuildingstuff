# Execution Backends for Developmental Worlds

## A bounded implementation choice

### Status of this document

This note records what an execution backend is responsible for and limits what
may be inferred from selecting one. The first local backend choice was made in
[STEP-1.md](STEP-1.md): a compact, typed finite transition executor implemented
in Rust. STEP 1 is now closed as the failed scientific design `world-0.1.0`;
the executor remains reusable machinery, not validated world semantics. See
[STEP-1-WORLD-0.1-CLOSURE.md](STEP-1-WORLD-0.1-CLOSURE.md).

The research foundation remains defined in
[PROCESS-AND-RENDERING.md](PROCESS-AND-RENDERING.md). Step 1 specifies one
process and representation contrast clearly enough to execute; its backend
selection is local to that probe and does not establish a general world
ontology.

---

## 1. Why keep a backend boundary

A developmental process requires persistent consequences, but those
consequences can be implemented in many ways: an explicit transition table, a
relational schema, a typed program, a graph rewrite system, a probabilistic
model, or a real sandbox.

No one of these representations is currently licensed as the project's world
ontology. Selecting one too early would make the structures it expresses
conveniently appear more basic than they have earned the right to be.

The eventual backend boundary has one purpose:

> Keep the meaning of an executed process separate from the machinery used to
> store and compute it.

---

## 2. Minimum responsibilities

Whatever substrate is selected for a probe must provide only what that probe
claims to study. At minimum, it must make the following statements exact:

- which typed interventions are accepted in a presented situation;
- how an accepted or rejected intervention affects persistent process state;
- which observation follows;
- whether the process continues, completes, or fails;
- which privileged facts the teacher may query; and
- how the episode can be reproduced for inspection.

The executor, privileged teacher queries, and renderer-facing objects must
agree about these semantics. A convenient teacher label must not claim a
quantity that the executed process does not define.

These are semantic obligations, not yet a software interface. Function names,
class layouts, manifests, capability registries, compiler stages, and
conformance suites should be introduced only when repeated implementation work
requires them.

---

## 3. The first substrate decision

Step 1 made a bounded first implementation choice. The executor represents
generated finite process instances using typed
state, actions, observations, guarded transitions, termination rules, and
privileged verifier queries. It may compile these objects into compact tables,
integers, and bitsets for execution. Rust is the implementation language.

Operationally, this choice satisfied the original implementation criteria:

1. **Typed execution:** it implements the declared transitions, although the
   later audit showed that the learner/teacher information contract was
   scientifically mis-specified.
2. **Auditability:** its transitions and privileged teacher information can be
   checked directly.
3. **Economy:** it requires little machinery beyond the first discriminating
   experiment.
4. **Reversibility:** replacing it would not require redefining the process or
   its learner-facing representation.
5. **Non-leakage:** its internal factorization need not be exposed to the
   learner as the answer.

The choice is warranted because the first family is finite, generated,
discrete, and performance-sensitive. A simple reference transition
implementation and an optimized batched implementation can share the same
typed semantics. The selected representation remains an implementation of the
Step 1 process contrast, not a claim that all later worlds should be finite,
discrete, table-compiled, or written in Rust.

---

## 4. When a common protocol is earned

A shared backend protocol becomes useful only after at least two implemented
processes create repeated execution needs. At that point, extract the smallest
common contract from working probes.

Do not require a second substrate merely to demonstrate architectural
generality. A second substrate is warranted when a substantive claim is said
to survive the first substrate's representational bias.

Likewise, a process operator is substrate-general only after its consequence
can be stated without reference to one backend and realized without semantic
drift elsewhere. Until then it is local machinery.

---

## 5. Current decision

Step 1 selects a typed finite transition executor implemented in Rust, with a
simple reference path, an optimized batched CPU path, and a batched Python
boundary. This is the first backend for the first probe.

Relational execution remains a possible substrate for later processes when it
is the smallest faithful representation of their semantics. A GPU executor
likewise remains a performance option only after measurement. Neither is
required to demonstrate architectural generality in Step 1.
