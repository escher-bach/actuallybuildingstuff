# World Selection

**Status:** assistant-authored selection proposal; not an authorization for a
world or experiment

## Definition

A world is an interactive process

\[
W=(S,A,O,T,\Omega,G,C),
\]

where \(S\) is the state space, \(A\) the actuator actions, \(O\) the sensor observations, \(T\) the transition law, \(\Omega\) the observation law, \(G\) the abstract non-linguistic goal specifications, and \(C\) the relevant costs or commitments. A world family is a deliberate distribution over such processes.

## Purpose

Worlds provide controlled developmental experience for the near-tabula-rasa robot and test whether the organization acquired from that experience makes later goal-directed processes more sample-efficient to learn.

## Selection Methods and Their Roles

These are not competing answers to one question, and they are not stages of one method. They occupy different roles.

**Capability-first and affordance-first select content.** Each names, by a different route, the organization a world family is meant to make learnable: capability-first by decomposing downward from a capability desired in the mature robot, affordance-first by expanding a reliable action-effect repertoire without naming a capability in advance. These two are genuine alternatives to each other, and either can independently determine which worlds are constructed.

**Invariance-first supplies an evidence discipline.** It states the surface changes across which a claimed organization must survive, so that success cannot be explained by a policy specific to one body, one rendering, or one set of laws. It does not by itself say which organization is worth preserving. That content comes from elsewhere: a capability decomposition is one source, and the known gap between the constructed worlds and the eventual deployment is another.

The asymmetry should not be softened into mutual dependence. A capability can be named with no invariance machinery at all; what it then lacks is any way to show it was acquired rather than memorized. An invariance can be imposed with no capability decomposition; it is anchored by whatever else supplies its content, and is ad hoc only when nothing does.

Where a capability decomposition is the content source, the two structures interleave rather than sequence: each capability node's isolating test is that node's invariance class, so the variation discipline is attached node by node rather than applied afterward.

### Capability-first

Begin with a capability desired in the mature robot and decompose it into the prerequisite capabilities required by this particular learner. Continue until reaching the first reusable prerequisite that the learner does not yet possess but could acquire from its present capabilities. Select world families in which success requires that prerequisite, then test whether acquiring it reduces the experience needed to learn capabilities above it. The developmental sequence is therefore determined by a learner-relative capability dependency graph.

### Affordance-first

Begin with relations among the robot's sensors, actuators, environmental conditions, and resulting effects, without first naming a higher cognitive capability. Construct worlds in which the robot can discover which actions produce which observable effects under which conditions, and can progressively reuse or compose those relations. Select world families that expand this reliable action-effect repertoire and whose learned relations remain useful when bodies, objects, or conditions change. Capabilities are expected to emerge from mastering structured possibilities for action rather than being chosen in advance.

### Invariance-first

Begin with an abstract controlled process whose organization is intended to survive changes in surface realization. Construct multiple world families with different states, bodies, sensors, actuator meanings, or laws, while preserving a declared mapping to that same abstract process. Select worlds whose differences prevent a surface-specific policy from succeeding across the family. Transfer between realizations is then evidence that the learner acquired the preserved organization rather than memorizing one body's actions, one rendering, or one set of world laws.

## Plausibility

A world family is plausible only if:

- good performance across its meaningful variations gives reason to expect improved learnability in future worlds;
- the attainable target is defined at the learner's information boundary; and
- no simpler policy can succeed through stable labels, serialization leakage, memorization, or privileged information without acquiring the claimed organization.

This criterion is where the two roles meet. "Meaningful variations" is an evidence-discipline term; "the claimed organization" is a content term. A family that supplies only one of them cannot be assessed against this criterion at all.
