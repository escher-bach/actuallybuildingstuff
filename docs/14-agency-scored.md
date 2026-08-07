# Agency — self-observation scored against the literature

*Resolves the four predictions sealed in [docs/13](13-agency-self-observation.md). Three confirmed, one mis-scored and still open.*

---

## S1 — CONFIRMED, with the strongest evidence in the vein

*Predicted: a substantial literature on knowing-whether-you-know, containing measured dissociations between knowing and knowing-that-you-know.*

Hampton (2001), *PNAS* 98:5359–5362, "Rhesus monkeys know when they remember" — **curator-verified** citation and paradigm. Monkeys given the option to *decline* a memory test performed better on tests they chose to take than on tests they were forced to take. The dissociation is the point: accuracy on forced trials measures memory; the *gap* between chosen and forced measures knowing-about-memory, and it is non-zero.

*(The review reports 88% vs 67% and 78% vs 69% for the two animals. Citation and paradigm are curator-verified; those specific figures are not — recorded as review-reported.)*

**A control the review did not surface, found while checking, and it is a design constraint rather than a detail.** Pigeons offered the decline option *concurrently with the test display* behaved as if they knew when they remembered — but when required to judge **before** the display appeared, they could not discriminate. So the decline-after-seeing version can be solved by reading test difficulty off the stimulus, and only the decline-before version isolates metamemory.

**That transfers directly.** An L3 family in which the model declares uncertainty *after* seeing the query is a different and much weaker task than one in which it declares *before*. If we build a sufficiency family, the declaration must precede the query, or we will be measuring difficulty-estimation and calling it calibration. This is the sharpest single thing the vein produced.

## S2 — CONFIRMED

*Predicted: at most two genuinely new primitives.*

Exactly one proposed: **`termination-gate`**. Two other candidates were considered and folded — convention-extension-by-similarity into `majority-predict`, self-detected error correction into `hypothesis-elimination`/`belief-state-reset`. The folds were argued rather than asserted.

## S3 — CONFIRMED, and it exposed a third kind of entry

*Predicted: precommitment has a literature but is a protocol, not an operation a solver composes.*

Confirmed, and the argument is better than mine was. Every `operation` in the register is computed **within** a single query/response step — narrow a belief state, walk a structure, invert a bijection. An implementation intention instead **rewrites how a later step will be decided**, spanning from the moment of commitment to the moment of triggering, with arbitrary unrelated activity in between.

So it is neither an `operation` (done *at* a step) nor a `trace-act` like `epistemic-status-tagging` (also done *at* a step) — it is **done to the rest of the episode's decision procedure**. That is a third category, and the register does not have it.

**Not adding it.** The vocabulary already had to absorb one kind-distinction (`operation` vs `trace-act`) and the saturation curve is already only readable per kind. A third kind, introduced for a single member that we agree is not a primitive, would make the curve harder to read for no measurement gain. Recorded here so that if a second member ever appears, the category is already named.

## S4 — MIS-SCORED. Still open.

*Predicted: the literature names at least one agency activity **absent** from my list of seven.*

The review marked this CONFIRMED, and the evidence it gave was that **five of my seven items have named literatures** (epistemic vigilance for deciding-what-to-check; precommitment for binding-future-judgement; help-seeking for escalation; delegation-to-automation; convention nameability). That is the *converse* claim. It shows my list maps onto existing literatures — it does not show the literature contains anything outside my list.

**So S4 is unresolved**, and it is the one prediction whose failure mode was named in advance: *if the literature only confirms my framing, that is the more worrying outcome.* That outcome is still live.

**Why this matters more than a bookkeeping slip.** S4 was the prediction designed to detect stipulation — to check whether an introspective list, written by the system under study, was *incomplete* in the way an outside literature should expose. Marking it confirmed on converse evidence would have retired the check while leaving the risk in place. The pre-registration caught it; a prose expectation almost certainly would not have.

**One genuine gap the review did find and flag**, to its credit: it searched for and found **nothing** corresponding to *"recognising an operation is missing rather than mis-named."* That is the one item on my list with no literature at all — which is interesting in the opposite direction, and worth revisiting rather than treating as a null.

**Action:** S4 stays open. Resolving it needs a targeted pass asking the inverted question — *what does the agency literature study that is not on this list?* — rather than a mapping exercise. Logged as a task.

---

## The one new primitive

**`termination-gate`** — decide whether to halt or continue, driven by a **cheap proxy signal rather than a read-out of the full belief state.**

This answers the fold question I flagged as the sharpest: *with a computable posterior, isn't "do I know enough" just a threshold on entropy?* The literature's answer is no, and it is specific: feeling-of-knowing judgements are made from **cue familiarity and retrieval fluency**, not from inspecting the retrieved target — which is exactly why the pigeon control above works. A system that read its own posterior would show no difference between judging before and after seeing the test. Animals do.

So the distinction is real and it is not a threshold on `posterior-enumeration`: the gate runs on a signal that is *cheaper than* and *dissociable from* the posterior. Recorded with a falsifier.

Note this also fits the vein §2.2 finding from the other direction. `informative-query-selection` costs O(belief state); `structure-walk-query` and `basis-probe` avoid that by exploiting structure. `termination-gate` avoids it by using a proxy. Three different escapes from the same cost, which is a pattern worth watching in the matrix.
