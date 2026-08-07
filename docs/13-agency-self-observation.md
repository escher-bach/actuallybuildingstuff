# Agency — self-observation, PRE-REGISTERED

*Written **2026-08-07, before any agency literature was read**, and committed before the vein §2.8 brief was launched. The git history is the evidence for that ordering.*

**Why this is sealed rather than simply written.** Repertoire Spec §1 names stipulation as the first failure mode: *"Write down the capabilities you believe reasoning consists of, build a family per capability, discover you built your own taxonomy. Nothing in the result is evidence about anything."* An introspective list of "what agency consists of" is exactly that. So this is not a proposal — it is a **prediction to be scored**, the same device as `docs/02`.

**The method being followed.** These activities exist right now only in *concrete* instances. They become general by observing many instances and abstracting what recurs. That is what the literature review does; this document adds one more observer — the system doing the work — with the same discipline applied.

---

## 1. What I actually did in this session that is not obviously in the vocabulary

Restricted to operations I can point at in the transcript, not to what agency "should" contain.

| # | Observed activity | Concrete instance |
|---|---|---|
| A1 | **Deciding which claims are worth checking** | Verified some numbers and not others. Triaged by whether a figure was load-bearing for a verdict — the fabricated 26.5% had already reached an A1 check, so it mattered; a background citation did not |
| A2 | **Noticing a discrepancy between a record and reality** | Marked §11 step 2 complete when §2.4 had never been read. Every downstream artifact was consistent with the error; what caught it was re-reading the claim against the actual list |
| A3 | **Binding my own future judgement** | Wrote `docs/02` append-only and fixed thresholds in `expectations.py` *before* any matrix existed — specifically so a later version of me could not read the numbers and find the prediction that fits |
| A4 | **Detecting that a novel case falls outside a convention I set** | Adding vein §2.7 broke the validator, which enforced the six-vein list. The convention underdetermined the new case and something had to notice |
| A5 | **Deciding when to escalate rather than decide** | Asked about the seal before opening agentic-harness sources, rather than judging it silently |
| A6 | **Allocating work between doing and delegating** | Literature review to subagents, curation kept in-house, with a stated reason: the vocabulary is shared state |
| A7 | **Recognising an operation is missing rather than mis-named** | `abstraction-naming` accepted because every existing primitive was eliminative or interrogative and none *constructed*. That is noticing a hole in a space, not finding a synonym |

## 2. Candidate primitives — the abstraction attempt

Fold-first, as the discipline requires. For each: what it might already be, and why I think it is not.

**`monitoring-for-discrepancy`** *(from A2, A4)* — compare a claimed or recorded state against the actual state and detect mismatch. Distinct from `hypothesis-elimination`, which cuts hypotheses against *evidence about the world*; this compares a **record against reality** and the thing that fails is bookkeeping, not belief. Also distinct from `belief-state-reset`, which fires on contradicting evidence; this fires on an *absence* of correspondence that no single observation announces. **Weakest point:** it may be `hypothesis-elimination` with the record playing the role of hypothesis.

**`commitment-before-evidence`** *(from A3)* — record a prediction, threshold or rule *before* the evidence arrives, in order to constrain your own later reasoning. Related to `epistemic-status-tagging` but not the same: tagging *annotates* a step that has happened; this *binds* steps that have not. It is a control operation whose object is your own future processing. **Weakest point:** it may not be an operation a solver composes at all — it may be a protocol, in which case it belongs where `harlow-learning-set` went, in the harness rather than the register.

**`sufficiency-judgement`** *(from A1, and the "knowing when to retrieve / knowing when you don't know" pair)* — decide whether current information suffices for the decision at hand, and therefore whether to act, to seek more, or to decline. **This is the one I expect the literature to have**, under metacognitive monitoring or feeling-of-knowing. Distinct from `informative-query-selection`, which chooses *among* candidate queries assuming you have decided to query; this decides *whether* to query at all. The two compose: sufficiency says go, selection says which. **Weakest point:** with a computable posterior, "do I know enough" is a threshold on entropy, which may make it `posterior-enumeration` plus a comparison rather than an operation.

**`convention-adherence`** *(from A4, and "following conventions / notion of standards")* — maintain consistency with a convention that is *underdetermined* by its own statement, extending it to novel cases in the way it would have been extended. Distinct from `constraint-propagation`, which derives consequences from constraints that *do* determine them; the interesting case is precisely where the convention is silent and consistency is still expected. **Weakest point:** may reduce to `abstraction-naming` — inferring the template a set of instances shares, then instantiating it.

## 3. Predictions to be scored against vein §2.8

Committed now, resolved by the literature.

**S1 — `sufficiency-judgement` is real and the literature has it.** I expect a substantial cognitive-science literature on knowing-whether-you-know, with measured *dissociations* between knowing and knowing-that-you-know. If it exists and is measured, this is the strongest candidate.

**S2 — At most two of the four survive as new entries.** The base rate says so: five of ten sources added zero, and the two that added anything added one apiece. A self-observation pass producing four new primitives would be evidence that I am generating rather than observing.

**S3 — `commitment-before-evidence` fails as a primitive and survives as a protocol.** I expect it to have a literature (precommitment, implementation intentions) and to still not be an operation a solver composes.

**S4 — The literature will name at least one agency activity that is NOT on my list.** If self-observation were sufficient, the review would be unnecessary. A list from inside the system should be *incomplete* in a way an outside literature exposes — and if it is not, that is evidence the literature is confirming my framing rather than testing it, which would be the more worrying outcome.

## 4. The honest caveat

I am the system under observation, which makes this weak evidence about agency in general and stronger evidence about *what this session did*. Two specific reasons to discount it:

- **I have read the existing vocabulary.** Anything I "observe" is contaminated by knowing what the fourteen entries are, and the folds I propose may be reconstructions rather than discoveries.
- **A transcript is a finished artifact.** §2.6's whole finding is that finished artifacts delete the control structure — the backtracks, the abandoned approaches, the things considered and dropped. My own record has the same defect: I can see what I did, not most of what I nearly did.

Which is the argument for the vein rather than against the exercise. **This list is a hypothesis about agency; the literature is the test.**
