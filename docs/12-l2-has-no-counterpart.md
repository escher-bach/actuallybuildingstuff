# L2 has no counterpart in practice

*From vein §2.7, the first vein drawn from contemporary practice in one of our target domains. Two findings, and they point in opposite directions.*

---

## 1. Nobody supervises the query channel

Our L2 (Task Spec §2.1) does something specific: the model emits a query, and **the query channel is supervised against a teacher policy `q*`** — cross-entropy on the model's query tokens against what a well-chosen query would have been. The oracle answers what was actually asked, so a poor query yields a genuinely uninformative response and the model must recover, with the better query visible as the target.

The review found the *shape* of this in practice — malformed tool calls do get genuinely uninformative answers, and there is a benchmark for tool misuse. **But no supervised teacher policy exists anywhere in the fetched literature.** Clarification timing, tool-recovery quality, next-action choice: all graded by **downstream outcome only**. Never by a prescribed correct next action.

That is exactly the design the Task Spec argues against, and it argues against it in advance:

> Supervising only through the final answer reintroduces exactly the long-horizon credit assignment this design exists to avoid — **that is reinforcement learning with extra steps.**

So the field is doing the thing §2.1 rejects, and our L2 is genuinely different rather than a reinvention. Two readings, and I do not think the evidence chooses between them:

- **Optimistic.** A per-token supervised query channel is a real contribution, and its absence in practice is an opportunity rather than a warning. Everyone is doing outcome-grading because that is what you can do *without* owning θ — and we own θ.
- **Pessimistic.** Nobody supervises next-action choice because it is not learnable that way, or because "the best next query" is too underdetermined to be a target. The absence would then be evidence, not an opening.

The optimistic reading is more likely correct for a specific reason: outcome-grading is what you are *forced* into when you cannot compute the counterfactual. In a real agentic task nobody knows what the best next tool call was. In our setting the generator knows θ, so `q*` is computable — the constraint that makes everyone else use outcomes does not bind us. But that is an argument, not evidence, and it should be labelled as one.

## 2. The harness decides what to work on next — which is our L2's job

The field's own framing separates the layers roughly as: **the harness structures the task and decides what to work on next; the agent decides how to take each step.** Task decomposition and guided execution sit in the scaffold.

*(Sourcing note: the crisp dichotomy as stated is not in the abstract of the paper it was attributed to. What that abstract does contain is "task decomposition, which structures a task into sub-goals, and guided execution, which reshapes local action distributions during execution" — the same split, less crisply. Recorded as a paraphrase rather than a quotation; see the verification log.)*

**That is our L2's job, done by the scaffold.** Choosing what to look at next is precisely what L2 trains, and contemporary systems take that decision away from the model and put it in the harness.

This is the sharpest statement yet of what this programme is for. Every operation the harness performs is one **the model has not learned to do** — and the harness layer is, in that reading, an inventory of missing capability. A model with L2 installed could in principle absorb part of the scaffold rather than depend on it.

**One finding cuts against over-reading this**, and it is worth keeping visible: the same source reports that *specifying only the initial steps and leaving the rest to the agent achieved a higher pass rate than fully structured workflows.* Partial guidance beat full guidance.

That is the **same shape** as the trace-thinning literature in `docs/05` — worked examples help novices and hurt experts; faded guidance beats both full guidance and none. Two unrelated fields, one curve. It is the strongest independent corroboration the review has produced for the fading result, and it arrived from contemporary agent engineering rather than from 1980s instructional design.

---

## 3. Compaction: a measured deficit, and a family that targets it

The strongest empirical result in the vein, curator-verified:

> Across seven models and 1,323 episodes, context compaction raises constraint violation from **0% to 30% (up to 59%)**. Decomposed: **when the constraint survives the summary, violation is 0%; when it is dropped, 38%.** Decay is **8.3× larger for soft organizational policies than for hard safety norms**. A training-free defence that pins ~47 tokens restores violation to 0%.

The decomposition is what matters. This is not gradual degradation over a long context — it is a **clean dissociation**. While the information is present the model obeys perfectly. Once the summary drops it, the model neither retains nor reconstructs it, one time in three.

That is a measured deficit in exactly the thing the **compaction-survival** family template (`docs/11`) manufactures: identify θ, then replace the evidence with a lossy summary and require the model to continue. And it says something about how to build it — the interesting condition is not "summarize badly" but **"summarize in a way that drops the load-bearing fact"**, because that is where the failure lives. The rate at which the summary omits what matters is the knob.

It also suggests the family should carry both variants, since the source found an 8.3× difference between them: a fact that is *directly needed* for the next answer, versus a standing constraint that only *sometimes* applies. The second decays far faster and is the harder target.

---

## 4. What the evals do not measure

Confirmed as expected: **end-to-end task success dominates, and nothing measures L1–L3-like properties** — task inference under underdetermination, calibration, recovery from a self-inflicted error.

This is Task Spec §9's "building for the eval you have," confirmed from the field rather than predicted. A repertoire targeting L1–L3 will have **no external benchmark that scores what it installs**, and the dashboard will disagree with the design for as long as that remains true. `docs/02` §4 already records this as a thing not to abandon the design over. It is now an observation rather than an expectation.
