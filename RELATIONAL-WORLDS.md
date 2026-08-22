# Relational Worlds as a Candidate Substrate

## Status of this document

Relational schemas are one possible way to execute a developmental process.
They are not the project's foundation, the selected first backend, or a claim
about the natural ontology of intelligence.

The prior relational-first proposal has been withdrawn because no
discriminating process-and-rendering probe has yet been specified. The current
decision procedure is defined in
[PROCESS-AND-RENDERING.md](PROCESS-AND-RENDERING.md), and the role of an
eventual executor is bounded in [WORLD-BACKEND.md](WORLD-BACKEND.md).

---

## 1. What relational execution would provide

A small relational executor could represent a finite process using:

- typed entities;
- Boolean or finite-valued facts;
- parameterized interventions with conditions and effects;
- observation rules;
- continuation conditions; and
- privileged queries over generated state.

This can make discrete, factored processes exact, generable, and inspectable.
The same internal state can supply execution and teacher information without a
separate handwritten solver.

Those are engineering advantages. They do not show that the learner should be
given relational facts, that the relevant learned organization is relational,
or that other processes should be forced into this form.

---

## 2. The main risks

Relational schemas favor discrete entities, named predicates, crisp action
boundaries, and state factored into designer-selected variables. This creates
three immediate risks.

### Ontology by convenience

Processes that fit named fluents are cheaper to construct and therefore more
likely to populate the research programme. Recurrence within that selected
population cannot by itself establish a general process ontology.

### Abstraction leakage

A fluent may directly name the abstraction the learner was intended to form.
Serializing or supervising that fluent can teach the designer's vocabulary
rather than the organization needed to infer it.

### False generality

Renaming objects or generating larger relational graphs can create many
instances without varying the consequential structure. Instance count is not
evidence of developmental breadth.

Exact execution also does not establish that a process corresponds to useful
activity or that learning transfers beyond its rendering.

---

## 3. Conditions for choosing it

Use a relational substrate for the first probe only if all of the following
hold:

1. The process contrast has already been stated without assuming relational
   storage.
2. Relational encoding preserves that contrast without introducing additional
   learner-visible structure.
3. The changed-process control and both renderings can be generated from the
   same stated semantics.
4. Privileged relational state is separated explicitly from observations,
   actions, scaffolds, and targets.
5. A relational implementation is materially simpler than an explicit finite
   transition representation for the probe.

If these conditions fail, use another substrate or revise the probe. Do not
expand the relational language merely to protect its status as the first
backend.

---

## 4. What remains deliberately unspecified

There is currently no commitment to:

- a standard relational vocabulary;
- universal operators such as `HIDE`, `QUERY`, or `COMMIT`;
- a relational program generator;
- domain families such as logistics or diagnosis;
- a capability registry or lowering system;
- multiple backends for conformance testing; or
- a repository layout for relational infrastructure.

These decisions become meaningful only after a small probe shows which
semantics and audits are actually required.

---

## 5. Decision rule

Relational schemas should be selected when they are the smallest faithful
executor for an already-defined experimental contrast. They should be
rejected or replaced when their factorization determines the claim instead of
implementing it.
