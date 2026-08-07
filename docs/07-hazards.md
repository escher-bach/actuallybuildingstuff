# Hazard Register

*Everything that has gone wrong, nearly gone wrong, or would go wrong silently. Consolidated here because they were accumulating across six documents, and a hazard nobody can find is a hazard that recurs.*

**How to read the severity column.** The question is not how bad the outcome is but **how likely it is to pass unnoticed**. A loud failure is cheap. The expensive ones are those where the system keeps running, produces plausible numbers, and is measuring the wrong thing.

| | |
|---|---|
| 🔴 **Silent** | Would have produced confident wrong results with nothing flagging it |
| 🟠 **Quiet** | Detectable, but only if you were already looking for it |
| 🟡 **Loud** | Fails visibly; costs time, not correctness |

---

## 1. Fabricated evidence from delegated review

**🔴 Silent. Occurred.** A literature review reported a statistic — "26.5% of fault-test pairs exhibit failed error propagation" — that **does not appear in the cited paper**. Its actual findings were 0% at unit level and 11.4% at system level, pointing the *opposite* way from the argument the number was supporting. The figure had already propagated into an A1 verdict in a register row.

*How found:* spot-checking sources promoted to `verified = true`, prompted by the user asking whether the subagents could be trusted. Not by anything in the process.

*Pattern, which is narrow and worth knowing:* citations are real, titles exact, structural claims reliable. **Numbers drift.** Across four verification passes: 8 clean, 4 with drift or unconfirmed detail, 1 fabricated.

*Fix:* a subagent's `VERIFIED` label now promotes a source only to *candidate-verified*; `verified = true` requires a curator check logged in [verification-log.md](../register/verification-log.md). Every brief since carries an explicit accuracy directive naming this incident. The later reviews were visibly more careful — one caught a fetch-tool misattribution in its own source, one flagged that a paper's quoted bit-patterns didn't cross-check.

*Residual:* two canonical sources (Reiter 1987, de Kleer & Williams 1987) remain curator-unchecked, and the GDE entropy claim carries `circuit-fault-localization`'s A7 argument. That row is blocked from advancing until checked.

## 2. Seal contamination

**🟠 Quiet. Occurred twice.** Held-out literature was read during review passes: a non-primate animal cognition paper (§2.3), and a graph-colouring phase-transition paper (§2.5b).

*The instructive part:* the first brief stated the boundary but not the **verb** — it said what was out of scope, not that it must not be *fetched*. The agent read the paper and self-limited its *use*, which is not the same thing: what is read cannot be un-read, and the agent's later judgments are no longer independent of it.

*Fix:* every brief now states the seal as a prohibition on fetching. The second contamination happened anyway — through a paper whose title does not announce its sealed content — but was **self-reported**, which is the fix working in the way that matters.

*Not fixed further, deliberately:* prohibiting the fetch is the strongest instruction short of an allowlist, and an allowlist defeats the purpose of a literature search. Leakage through mis-titled sources is a residual risk to budget, not engineer away. Both logged in [docs/00 §4](00-heldout-partition.md).

## 3. A plant that stops being a plant

**🔴 Silent. Caught before shipping.** The random-junk plant was specified so the answer was a hash of `(θ, query)` — same query, same answer — on the reasoning that a self-contradicting oracle is a bug.

That's wrong. An answer that is a fixed function of the query is a **lookup table**: unpredictable once, fully learnable twice. The model would identify it within the episode like any other hidden rule, the loss would drop, and the row would have measured as a *memorization family* while sitting in the matrix labelled "junk, expect near-zero". **The instrument-validation gate would have passed while measuring the wrong thing** — the worst available failure, because it is silent and it is at the gate.

*Fix:* answers drawn fresh each trial. Repeats disagree, because there is no function to be consistent with.

*Generalized:* every plant now has a test asserting the *plant property* rather than that the code runs — that Type VI really computes parity, that the near-duplicate pair really computes the same latent function under different surfaces.

## 4. A calibration exemplar that quietly becomes L1

**🔴 Silent. Caught by a test.** The probability-matching family's rate grid included the endpoints 0 and 1. A rate of exactly 0 or 1 is **deterministic** — no irreducible noise, fully resolvable from a finite history. After a long run of one outcome the posterior collapsed to a point and the family stopped being an L3 exemplar, **on exactly the histories most likely to arise**.

It ran. It was genuinely stochastic. Only the assertion that the posterior never fully resolves distinguished it from a family that degrades into L1.

*Fix:* grid strictly interior. *Lesson:* a family can satisfy its type signature and violate its purpose.

## 5. Equivariance under the wrong alphabet is still equivariance

**🔴 Silent. Occurred.** The concept-family encoder drew "symbols" from `range(N_SYMBOLS)` — raw indices — rather than from `SYMBOL_IDS`. Since the vocabulary lays out control and structural tokens first, task content was being rendered as `PAD`, `BOS`, `EQ`.

Every id was a valid id, so the in-vocabulary test passed. **The A2 permuted-alphabet check also passed**, because the family was internally consistent about its wrong tokens.

*Lesson, stated for anyone adding a family:* **A2 passing does not mean the rendering is right. It means the rendering is consistent.** Being in the vocabulary is not the same as being in the alphabet.

*Fix:* regression test asserting content tokens come from `SYMBOL_IDS`.

## 6. A check that cannot fail

**🟠 Quiet. Occurred twice, in opposite directions.**

*First:* the A2 check permuted only the symbols the current encoding used. A family rendering some token from a fixed constant leaves it untouched on both sides of the comparison — so the check passed the exact failure it exists to catch. Fixed by using a **total** permutation, and by adding a deliberately-leaky subclass that must fail. **A plant passing its own check proves nothing unless failure is reachable.**

*Second:* adding a field to `Encoding` silently broke that same check, because it rebuilt the encoding field-by-field and the new field defaulted — then failed an A2-*compliant* family. **A check that breaks when its subject gains a field is worse than no check, because its failure is indistinguishable from a finding.** Fixed by copy-and-replace.

## 7. A3 satisfied in letter, vacuous in fact

**🔴 Silent. Occurred.** Every family's encodings produced *byte-identical episode lengths*. They differed only in which token was the separator. Rows described them as "structurally different"; they were punctuation.

A3 asks for a nontrivial `𝓔`. A family whose encodings are identical satisfies "sampled per episode" while testing nothing — and the matrix would have reported an encoding-invariance that was never at risk.

*How found:* building the A3 leak test found zero variance to measure. **The test earned its place by finding there was nothing to find.**

*Consequence of the fix:* a genuinely different rendering produces an ~85% length difference, which is a real exchangeability problem — **the harness must normalize per token**, a requirement it does not currently know it has.

## 8. High explained variance is not clean separation

**🟠 Quiet. Prior art, not yet our failure.** From the item-generation literature: a difficulty radical explained **95.41% of variance** and the likelihood-ratio test **still rejected** the constrained model at p < .001. Incidentals were leaking at 6.82% and 12.36%.

A radical explaining 95% of variance passes any eyeball test, any correlation plot, and most thresholds. *And the leak was absent at the extreme of the difficulty range and present in the middle* — so testing at maximum `k` alone would have missed it.

*Status:* our A3 test currently uses a deterministic proxy and is a structural check, not the statistical one. Tracked.

## 9. Designing for A2 does not achieve A2

**🟠 Quiet. Prior art.** A test *designed* to be culture-free had **59% of items flagged as biased** across four independent methods — while a different method on the same instrument flagged 13 of 46. Same test, different method, materially different answer.

*Two consequences:* "we built it to be invariant" is precisely the argument that failed, and **our `permuted_alphabet_check` is one method**, which the psychometric experience says will understate. A second independent A2 method is worth having.

## 10. A measured decomposition that is confidently wrong

**🔴 Silent. Prior art for our core method.** A Q-matrix is a capability decomposition recovered by measurement — §6 with humans and factor analysis in place of models and transfer. Documented failure modes: **provable non-identifiability** (distinct Q-matrices producing identical response distributions), misclassification to 44% under misspecification, and a published case of a fitted, converged model assigning **74–98% skill mastery to students who scored zero**.

*Consequence, recorded in [docs/02](02-predicted-decomposition.md):* recovering the plants is **necessary and not sufficient**. A method can recover planted structure and still produce nonsense on unplanted rows. Step 6 needs a sanity check of the kind that would have caught that.

## 11. A cited literature that is about a different axis

**🟠 Quiet. Occurred in the source spec.** Task Spec §1.2's trace-thinning default says it "is not a guess; it has a literature". The literature is real and fades guidance against **demonstrated learner competence over training**, never against a static task-difficulty knob — and states that fading only has teeth when intrinsic load is **high**, i.e. harder material is where guidance earns most.

*The claim, kept narrow:* the spec's own rationale is about reachability and may still hold. What is wrong is the citation, and the direction is questionable on the cited literature's own terms. [docs/05](05-trace-thinning.md).

## 12. Knobs that do not transfer between substrates

**🟠 Quiet.** "More simultaneous faults" makes circuit diagnosis harder (masking) and program mutation **not** harder — the coupling effect means 1st-order-adequate tests kill >99% of higher-order mutants. Same-sounding knob, opposite behaviour.

Related: a composition-depth knob **degenerates** if the primitive library commutes — brute-force cost keeps climbing while identifiability hits a fixed ceiling. The knob looks like it works while L1 and L3 targets quietly stop being learnable.

*Lesson:* measure knobs, do not reason about them. Non-monotonicity in `k` is documented in at least three literatures.

## 13. Gates that cannot currently be evaluated

**🟡 Loud, and recorded as failures rather than smoothed.**

- §11 step 3's gate is *inter-annotator agreement holds*. No reliability figure for the episode-coding scheme could be traced to a primary source. The gate cannot be evaluated, let alone passed.
- The problem-behaviour-graph notation — the closest existing data structure to what we want to generate — is LEAD-only after repeated fetch failures.
- Three §2.2 rows and most §2.3 rows sit at `lead` because their primaries are paywalled or unrenderable. **This is a library-access problem, not a literature gap**, and it is the single largest blocker on the register.

## 14. The parametrization is missing two axes

**🟠 Quiet. Found by running the gate.** Three of ten established paradigms do not fall out of the formalism, and they fail in exactly two ways: **θ is assumed sampled once and to hold** (violated by piecewise-stationary and drifting rules), and **L2 assumes querying is free** (violated wherever the query is the scored action).

*Not repaired yet, deliberately.* §8 step 5's sweep expects the four named levels to be cut in the wrong places; adding axes before it runs would be designing the answer. Both were pre-registered in docs/02 so they can be scored against the sweep. [docs/06](06-paradigm-coverage.md).

## 15. A schema that cannot express a relation it discovers

**🟡 Loud, but under-reports silently.** A register row carries exactly one `plant_role`. Parity and SHJ Type VI turn out to be the same object — an **unplanted** near-duplicate, excavated independently from computational learning theory and 1961 categorization psychology — but `shj-type-vi` already holds `dependent` in the prerequisite pair, so the second relation cannot be recorded in the schema.

It now lives in prose in both rows and is **invisible to `register.py coverage`**, which will under-report near-duplicate coverage. The finding is more valuable than the constructed near-duplicate plant — §5 says the ideal near-duplicate is "two independent reinventions found in different veins", which is exactly this — and the tool cannot see it.

*Not fixed yet:* the fix is a list of relations per row rather than a single role, which is a schema migration touching every row. Recorded so the coverage output is read with this caveat rather than trusted.

---

## Standing hazards — not yet observed, watch for

| | Hazard | Why it will be hard to see |
|---|---|---|
| 🔴 | **Brute-force collapse over training.** Families satisfying every constraint lose structural content once the learner can afford to memorize the generator. A synthetic generator is short by construction and has none of natural data's protection | Shows as a *turnover* in structural content as compute grows, which looks like noise unless you are watching for a sign change |
| 🔴 | **The loss-dashboard inversion.** Families with more learnable structure carry *higher absolute loss*. A controller doing the right thing looks wrong | Task Spec §9 names this "the easiest way for the work to be abandoned for the wrong reason" |
| 🟠 | **Measured blocks reproducing an imported taxonomy exactly.** This is to be *suspicious of*, not pleased by — more likely the candidate set was built to that taxonomy than that the taxonomy was right | It is the outcome that feels like success |
| 🟠 | **Building for the eval you have.** Every convenient benchmark measures L0-ish competence, so a repertoire targeting L1–L3 will look bad for a long time | The dashboard disagreeing with the design is the *expected* state, not evidence |
| 🟠 | **Saturation curve flattening because of what we chose to read.** The held-out veins exist to test exactly this | A flat curve is the result we want, which is why it needs an independent check |
