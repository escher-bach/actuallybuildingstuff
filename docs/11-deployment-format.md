# Deployment format — a correction

*Written 2026-08-07, **substantially rewritten the same day**. The first version argued that the repertoire should match the deployment format. That was an over-correction and this document now records both the corrected position and the error, because the error is the more instructive half.*

---

## What the first version said, and why it was wrong

The argument was: the model will be deployed inside an agentic harness, so the harness is the deployment input distribution; our clean `q → a` episodes are structurally unlike what a deployed model reads; therefore the repertoire should add a **transcript encoding** rendering episodes as tool-call trajectories, and should build a **compaction-survival** family.

**That is fitting the repertoire to a transient engineering convention**, and it contradicts two constraints the design already has.

**A3 exists precisely so that format does not matter.** Its purpose is that a family be invariant under a nontrivially varied encoding set, so the model learns the structure and not the surface. The correct response to "deployment has a different format" is **not** to match that format — it is to make `𝓔` varied enough that no format is privileged. If A3 is done properly, a novel deployment format is one more encoding the model has never had reason to depend on. **Chasing the deployment format is the opposite of what A3 asks for**, and doing it would weaken the very invariance that makes format-independence possible.

**A2's argument applies at the structural level, not only at the symbol level.** A2 forbids semantics leaking through content symbols. Tool-call syntax, schema conventions, and message-role markers are conventions — they are *knowledge*, not capability. A family trained on them is training on precisely the thing the repertoire exists to exclude. That the leak is structural rather than lexical does not make it less of a leak.

**And the specific artifacts will not survive.** "Compaction" is an artifact of current context-window economics. Tool-call JSON is a serialization choice. Harness shapes are not standard and are changing quickly. A basis built against them would be dated on arrival, and §10 condition 3 judges coverage of *established practice*, not of this year's implementations.

## The corrected position

**Format is not a target. Invariance is.** The repertoire's job is to install structure that survives an arbitrary surface, and the test of that is A3, not resemblance to any deployment.

What survives from the original argument, in abstract form:

**Episodes may be long, heterogeneous, and contain material that is irrelevant.** That is not a claim about harnesses — it is a property of any realistic deployment, and it is expressible without naming a single convention. A family whose episodes are always short, homogeneous and entirely relevant has not been tested against it. The knowledge-free version is **distractor tolerance**: interleave well-formed, plausible, clearly-delimited content drawn from a *different* θ. No tool schemas, no message roles, no JSON.

**Evidence can be withdrawn.** The abstract phenomenon is: a rule is identified from evidence, then *the evidence is removed while the rule is unchanged*, and the episode continues. That is a **reveal policy** — reveal, then un-reveal — and it sits alongside L0–L3 as a variation on what the context determines, not as a family about a particular memory-management technique. Named `evidence-withdrawal`, it is knowledge-free and would remain meaningful if every current harness disappeared.

The measured compaction result is **motivation for that abstraction, not its definition.** It is evidence that the phenomenon has teeth — belief does not survive the loss of its evidence, cleanly and by a large margin — and that is worth citing. It is not a specification.

## What actually changed

| | |
|---|---|
| Transcript encoding | **Dropped.** It was format-matching, which A3 makes unnecessary and A2's spirit forbids. The request has been withdrawn from the harness handoff |
| `compaction-survival` row | **Abstracted** to `evidence-withdrawal`, framed as a reveal policy over any base family. The compaction study is retained as motivating evidence with its status as evidence-not-definition stated |
| Distractor tolerance | **Kept**, in the knowledge-free form above |
| Malformed-response recovery | **Kept.** A6 already requires well-formed errors and Task Spec §2.1 already calls the error-recovery lesson free; making the error a *sampled event with a rate* is an abstract reveal-policy knob, not a harness detail |
| The confound argument | **Withdrawn.** "Transfer might fail for a format reason" is a real risk, but the remedy is stronger A3, not format-matching. Stated as an argument for taking A3 seriously rather than as a reason to imitate a deployment |

## What remains true from vein §2.7

The vein was worth reading and its abstract findings stand — they are in [docs/12](12-l2-has-no-counterpart.md) and none of them depend on any harness convention:

- **Nobody supervises the query channel.** All outcome-graded. That is a claim about *supervision*, not about format.
- **The scaffold decides what to work on next**, which is L2's job done outside the model. That is a claim about *where capability sits*, and it is the reason L2 is worth having regardless of what next year's scaffolds look like.
- **Partial guidance beat fully structured workflows**, corroborating the trace-thinning curve from an unrelated field.

Those are abstract. The parts I built out of them were not, and the difference is the lesson.
