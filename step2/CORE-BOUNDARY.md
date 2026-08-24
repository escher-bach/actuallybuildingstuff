# STEP 2 Core Boundary

**Status:** user-directed architecture boundary, accepted 2026-08-24

## Research claim

STEP 2 tests whether a control-and-reasoning core pretrained in many cheap,
procedurally generated abstract worlds becomes a useful prior when it is later
grounded into real sensors, goals, and actuators. The decisive downstream
comparison is abstract-pretrained core versus scratch core with identical
adapters, real data, and optimization budget.

## Ownership boundary

The core consumes only a canonical abstract sensorimotor sequence:

```text
schema + conditions + observations + goals + actions + feedback + queries
```

It never consumes pixels, text, or a robot-specific fixed action vector.
Perception, language, demonstration, and robot-action encoders are external
adapters. They may produce optional canonical content embeddings at token
positions, but those embeddings must be aligned to the core's abstract
representations during grounding; their mere dimensional compatibility is not
evidence of semantic alignment.

`Condition` describes facts that hold in the process. `Goal` describes the
desired outcome. They are distinct roles. `FutureQuery(key, horizon)` asks for
the future value of a public observation channel after the preceding actions.
The current world initially supervises horizon one; the ABI represents the
horizon explicitly so later multi-horizon worlds do not require another role
change.

The abstract-world checkpoint contains the core and its canonical numeric event
adapter only. Modality adapters and pretrained perception or language encoders
are later grounding components and are not part of abstract pretraining.

## Evidence boundary

World performance is not self-interpreting. A source-world result must state
which control, system-identification, memory, or prediction property it tests,
what the learner can observe, which trivial policies bracket it, and which
later transfer comparison it changes. The retained `0.1.0` checkpoint is
apparatus evidence only and is not a parent for the `0.2.0` lineage.

