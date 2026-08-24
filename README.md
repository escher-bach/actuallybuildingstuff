# baby-robot-foundations

Developmental worlds for embodied agents, built at the level of actuators,
abstract sensors, and abstract goal specification.

## What this project is

An agent controls actuators. It senses abstractly and partially. It is
conditioned by an abstract, non-linguistic goal specification. Worlds are
constructed rather than modeled: the body may be a dog, a lion, a blob, or a
coupling of actuators with no anatomical reading, and the environmental laws are
whatever the world declares them to be.

Worlds are accepted or rejected by the **plausibility criterion** in
[step2/WORLD-VALIDITY.md](step2/WORLD-VALIDITY.md):

> A world is valid if, at every reachable state, it is defensible that any agent
> performing well in it would also perform well in real activity presented
> through an interface of the same shape.

The criterion is negative. It is falsified by exhibiting a policy that wins in
the world and would be manifestly incompetent under that interface.

## What this project is not

**Not simulation training.** No world here models a real system, and no result
is a claim about transfer to a particular platform. Bodies, actuators, and
spatial structure do not make it sim training.

**Not sensorimotor training.** Motor competence on a body is not the capability
under development. Bodies are renderings; any capability claim must be stated
across bodies, which makes this a property of the measurement rather than a
matter of intent.

## Where to start

| Document | What it settles |
| --- | --- |
| [step2/CURRENT-STATE.md](step2/CURRENT-STATE.md) | **Start here:** current world, targeted capability, evidence level, next action, and authority map |
| [step2/CORE-BOUNDARY.md](step2/CORE-BOUNDARY.md) | Accepted modality-free core and downstream grounding boundary |
| [step2/META-PROCESS.md](step2/META-PROCESS.md) | User-directed working, velocity, compute, and experiment rules |
| [step2/WORLD-VALIDITY.md](step2/WORLD-VALIDITY.md) | The validity contract: plausibility, interaction mode, degenerate winners, acceptance gates |
| [PROCESS-AND-RENDERING.md](PROCESS-AND-RENDERING.md) | Presentation vs rendering; the discriminating probe skeleton |
| [TRAINING-DYNAMICS.md](TRAINING-DYNAMICS.md) | The cost model the worlds must live inside |
| [STEP-1-WORLD-0.1-CLOSURE.md](STEP-1-WORLD-0.1-CLOSURE.md) | Why the predecessor stage closed, and what its apparatus may be reused for |
| [WORLD-CORRESPONDENCE.md](WORLD-CORRESPONDENCE.md) | The superseded validity criterion, retained for its representation obligations |

## Open problems

1. **Selection.** Plausibility rejects invalid worlds but does not rank valid
   ones. A world can pass every acceptance gate and develop nothing worth
   having. See [step2/WORLD-VALIDITY.md §10](step2/WORLD-VALIDITY.md).
2. **Translation bridge.** With language removed from the loop, the claim that
   acquired organization can reach a later interface needs a replacement
   statement. See [step2/WORLD-VALIDITY.md §7](step2/WORLD-VALIDITY.md).

## Status

STEP 2 currently implements the `0.2.0` calibrated signed-permutation control
world and modality-free core. Local apparatus tests pass; no `0.2.0` GPU result,
retained checkpoint, source-world competence, or transfer result exists. The
next proposed action is the bounded C1 candidate described in
[step2/CURRENT-STATE.md](step2/CURRENT-STATE.md). The validity contract remains
a draft pending review.

Inherited material and its boundaries are recorded in
[PROVENANCE.md](PROVENANCE.md).
