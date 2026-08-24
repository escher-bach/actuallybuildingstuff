# STEP 2

**Fresh agents: start with [CURRENT-STATE.md](CURRENT-STATE.md).** It states the
current world, targeted capability, evidence level, next action, and which older
documents are historical.

STEP 2 follows the closure of `world-0.1.0` documented in
[STEP-1-WORLD-0.1-CLOSURE.md](../STEP-1-WORLD-0.1-CLOSURE.md).

The user-directed working process for this stage is recorded in
[META-PROCESS.md](META-PROCESS.md).

The user-directed modality-free learner boundary and transfer claim are
recorded in [CORE-BOUNDARY.md](CORE-BOUNDARY.md).

Its intended work is first a world-validity contract, then independently
motivated process examples from which candidate world operators may later be
extracted. It must preserve and reuse STEP 1 apparatus without inheriting STEP
1 world semantics.

The stage now targets robot-shaped worlds: an agent controlling actuators under
abstract sensors and an abstract, non-linguistic goal specification. This is
neither simulation training nor sensorimotor training.

The current `0.2.0` world targets in-context system identification for
goal-conditioned inverse control: infer a changing actuator-to-observation
permutation, sign, and gain from public calibration experience, then use that
model to reach a public abstract goal across variable dimensions.

The world-validity contract is drafted in
[WORLD-VALIDITY.md](WORLD-VALIDITY.md), pending review. It replaces
correspondence with a plausibility criterion and is not yet accepted.

The committed `0.1.0` vertical slice and its audited GPU run are historical
apparatus evidence. They do not establish source competence or transfer. ABI
`0.2.0` now has coherent Rust/Python/model contracts and a maintained
Transformers Trainer path, verified locally on CPU. It has not yet produced a
new checkpoint or two-T4 result, so no source competence or transfer claim is
attached to it.

The next proposed scientific action is a bounded `0.2.0` C1 candidate on this
world. It remains a candidate until paired pre/post evidence is reviewed and the
user explicitly promotes it; World 2 follows only after that decision.
