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

## 16. A step marked complete that was not

**🟠 Quiet. Occurred, self-caught.** §11 step 2 covers veins §2.1–§2.5. Six review passes ran — §2.2, §2.5a, §2.3, §2.5b, §2.1, §2.6 — and the README was updated to "all five veins registered". **§2.4 was never read.** The count of *passes* looked complete while the count of *veins* was not, because §2.5 was split into two passes and §2.6 was run out of order.

*Why it nearly stuck:* every downstream artifact was consistent with the error. The coverage grid had no gaps, because §2.5b filled the axes §2.4 would have. The saturation curve looked healthy. The docs/02 quarantine table had an empty §2.4 row, which was the only visible symptom, and an empty cell reads as "not filled in yet" rather than "vein not read".

*What was nearly lost:* §2.4 is the only vein that offers **A4 as a guarantee rather than an estimate** — every A4 verdict in the register currently rests on a threshold someone eyeballed, and this is the literature that could lower-bound one instead.

*Fix:* README now tracks veins, not passes. General lesson: **when work is split or reordered, the completion criterion has to be restated in terms of the original units**, or the split silently redefines what done means.

## 17. A4 cannot be certified, only measured

**🔴 Silent, and structural.** Vein §2.4 was read because the source document calls it the one vein offering **A4 as a guarantee rather than a hope**. It does not, at our scale.

Only one construction offers a genuine worst-case-to-average-case reduction, and no reviewed source states the dimension at which it becomes meaningful. Everything else rests on "no known attack." Most sharply: the planted-clique statistical–computational gap sits between roughly `2·log₂ n` and `Ω(√n)`, and **below about n ≈ 256 that interval is empty** — the gap the family's difficulty lives in does not exist at toy scale. *(Reviewer's own arithmetic from standard bounds, flagged as inference.)*

*Why this is silent rather than loud:* an A4 verdict of "holds from n ≥ 8" looks like a property. It is a **prediction**, and nothing in the register's format distinguishes the two. A family can carry a confident A4 pass, be trained on, and collapse — and the collapse appears as a *turnover* in structural content as compute grows, which reads as noise.

*Consequence:* A4 is not certifiable before training. It must be measured after, and Task Spec §9's instruction to watch for turnover is now known to be the **only** available route rather than a prudent supplement. [docs/08](08-a4-guarantee.md).

## 18. Rejection sampling as a repair

**🟠 Quiet.** A well-known forced-satisfiable instance generator achieves its property by *sampling and filtering*. That is an **A1 violation** — it couples generation cost to difficulty, which is the exact coupling A1 exists to forbid.

*Why it will recur:* our own generators construct forward from θ and do not filter. But rejection sampling is the natural repair when a family fails some *other* check — "just resample until the instance has the property we want" — and applying it would silently break the constraint the entire design rests on, while fixing the visible problem.

*Fix:* A1 checks should ask "does this filter?" explicitly rather than treating it as covered by "no search."

## 19. Sealing what sits inside what you are reading

**🟠 Quiet. Three occurrences, two on the same sub-area.** Graph-colouring phase transitions have now been contaminated twice, most recently by a paper whose title ("quiet planting") announces a technique rather than a substrate.

**Adjacency predicts contamination.** No instruction about fetching can prevent it, because the decision to fetch is made from the title. Sealing a sub-area that sits *inside* a vein being actively read is close to unenforceable.

*Lesson for any future seal:* prefer held-out items that are topically distant from what will be read — accepting that distance makes them weaker tests — over adjacent items that are strong tests but will not survive contact. Recorded in [docs/00 §4](00-heldout-partition.md).

## 20. `verified` conflates provenance with warrant

**🟠 Quiet, and it affects how every row's evidence should be read.** A source can do two different jobs, and the register's single `verified` boolean does not distinguish them:

- **Provenance** — *this family came from that literature.* Confirming Angluin 1988 treats "restricted types of propositional formulas" establishes where the conjunction family comes from. It does not establish that the anchor-and-flip algorithm works.
- **Warrant** — *this specific claim is true because the source says so.* "The radical explained 95.41% of difficulty variance" is true only on the source's authority. There is no other route to it.

The difference matters because **for a large part of this register the mathematics is self-evident and no citation warrants it.** That parity is identifiable by standard-basis probes is a fact about GF(2), not about anything Angluin wrote. A row citing a paper for it is recording lineage, not evidence — and if the citation evaporated the claim would stand.

Whereas an empirical claim — a replicated difficulty ordering, a measured bias rate, a coupling-effect percentage — has no independent route. There the citation *is* the claim, and a verification failure destroys it.

*Consequence as things stand:* `verified = true` means different things in different rows, and **the strength of a row's evidence is not readable from the flag.** A reader cannot tell whether a row would survive its citations being wrong.

*Fix, not yet applied:* a per-source `role` field — `provenance` or `warrant` — so a row can say which of its sources it would survive losing. It is a small schema change but touches every row's source blocks, and doing it mid-review would mean re-triaging sources under a distinction that did not exist when they were recorded. Recorded now so the ambiguity is known rather than discovered later.

*Interim practice, already applied:* rows promoted on 2026-08-07 state in prose exactly what their citation does and does not cover. That is the right content in the wrong place — prose is not queryable — but it is better than the flag alone.

## 21. A repair cited across the register that had no legal instance

**Silent. Found by implementing the operation.** A large fraction of the register A4 verdicts name composition as their repair -- *"fails at small n; compose with a second family, or raise n."* Task Spec section 1.1 endorses this: composition is "the cheapest route to A4" and "what lets a finite basis cover an infinite target space. Without closure you are enumerating, and enumeration cannot cover."

**The implemented basis had essentially no composition closure.** `T1 o T2` requires codomain(f2) inside X1, and our families almost uniformly map a structured query to a **single label**: bit-vector to category, stimulus to parity bit, assignment to true/false. A label cannot be fed back in as a query. So every row citing "compose with another family" was citing an operation with **no legal instance in the basis**.

*Measured:* `closure_report` puts it at **2.4% of ordered pairs** -- one legal composite over seven families.

*Why it was silent:* nothing checks that a named repair is performable. Each A4 verdict was individually reasonable, the spec endorses the repair in general, and the gap exists only at the level of the basis as a whole, which no single row can see.

*This compounds hazard 17.* A4 cannot be certified at our scale, so it must be measured -- and when measurement shows a family collapsing, composition is the constructive repair. With no closure, the only remaining repair is "raise n", which is the route that runs out.

*Fixed:* `compose.py` implements section 1.1 n-arily with depth as the knob, and `PermutedBitsFamily` is the first endomorphic family, so composition has somewhere to stand. A test asserts the label-returning families are still label-returning, so adding an endomorphic family moves the closure figure and says so.

*Residual, stated:* one endomorphic family is barely closure. The basis needs one per type it intends to compose over, and that is now a design requirement on new families rather than something to rediscover.

## 22. A type check is not the coherence check the spec asks for

**Silent, and it was in my own first implementation.** Section 1.1 says plainly: *"A5 (semantic coherence) is precisely the condition under which a composite is meaningful rather than a **type-checking accident**."* The first version of `compose.py` implemented the type check and nothing else -- that is, it implemented the accident detector and called it the gate.

*What that misses, concretely:* `junk_trivial o permuted_bits` **type-checks perfectly**. Both stages speak `BitVector`. The composite is constant, because the outer stage ignores its input entirely. It would have entered the candidate set as a legal composite and contributed a row to the matrix that measures nothing.

Found in the wild by `closure_report`, not contrived for the example.

*Fixed:* `compose()` now runs two gates, and the second one is the one that matters. Coherence is operationalized as three sampled falsifiers -- the answer must vary with the query, and with **each** stage's theta. A stage whose theta never changes the answer is doing no work, and the composite is exactly the accident the spec names.

*Honest limit, stated in the code:* sampling cannot prove coherence, only fail to catch incoherence. A composite that passes has merely not been caught being vacuous.

---

## Standing hazards — not yet observed, watch for

| | Hazard | Why it will be hard to see |
|---|---|---|
| 🔴 | **Brute-force collapse over training.** Families satisfying every constraint lose structural content once the learner can afford to memorize the generator. A synthetic generator is short by construction and has none of natural data's protection | Shows as a *turnover* in structural content as compute grows, which looks like noise unless you are watching for a sign change |
| 🔴 | **The loss-dashboard inversion.** Families with more learnable structure carry *higher absolute loss*. A controller doing the right thing looks wrong | Task Spec §9 names this "the easiest way for the work to be abandoned for the wrong reason" |
| 🟠 | **Measured blocks reproducing an imported taxonomy exactly.** This is to be *suspicious of*, not pleased by — more likely the candidate set was built to that taxonomy than that the taxonomy was right | It is the outcome that feels like success |
| 🟠 | **Building for the eval you have.** Every convenient benchmark measures L0-ish competence, so a repertoire targeting L1–L3 will look bad for a long time | The dashboard disagreeing with the design is the *expected* state, not evidence |
| 🟠 | **Saturation curve flattening because of what we chose to read.** The held-out veins exist to test exactly this | A flat curve is the result we want, which is why it needs an independent check |
