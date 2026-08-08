# Alignment against *Bias Without Knowledge* v4

*A diff between the theoretical framework in [bias-without-knowledge-v4.md](../bias-without-knowledge-v4.md) (read-only; not edited by this pass) and the work in this repository as of commit `7a65ee5`. Written to be scored, not to reassure: the sections that matter are §4 (substitutions), §5 (contradictions) and §3 (findings that should change the framework).*

**State at time of writing:** 207 tests passing, 27 register rows (0 validation problems), 9 implemented families, harness steps 1/2 complete, step 5 built and unrun.

---

## 0. The frame

The two specifications in this repo are not independent of the framework — they are its Part IV, operationalized, with a vocabulary deliberately kept free of the framework's citations. Almost nothing here contradicts Parts I–III; the disagreements are all in the build.

| | Framework | Here |
|---|---|---|
| What it is | A prospectus: derives what a bias is, then specifies the next experiment | Two specs + a harness + a register: the next experiment, built |
| Vocabulary | epiplexity $S_T$, time-bounded entropy $H_T$, SDL, $\mu$, $\beta$ | structural content, Bayes floor, transfer, coverage |
| Overlap | §II.3–§II.6, R2/R5/R6, §I.6, §IV.3, §IV.8 | Task Spec §1–§4, §8; Repertoire Spec §4, §6 |

The framework names two extraction exercises (Track B, Track B′) and four compute experiments (§IV.3). **The work has run neither extraction exercise and one of the four experiments is built but unrun.** What it has instead is a large amount of instrument engineering the framework does not specify, and most of that turns out to be load-bearing.

---

## 1. Correspondence map

| Framework | Counterpart here | Status |
|---|---|---|
| §II.3 situation generator $(\mathcal{P}, [\![\cdot]\!], \pi, \nu, \mathrm{Desc})$ | Task Spec §1 family $(\Theta, P_\Theta, \mathcal{X}, f, \mathcal{E}, \rho, k)$ | **Same object.** Work adds $k$ and promotes $e$ to a per-episode latent |
| §II.4 inference axis L0–L3 | Task Spec §2 levels | **Identical, incl. v4's two corrections** — retrieval off the ladder, calibration as the irreducible cell |
| R6 four complexity axes | Repertoire Spec §4 "complexity mode" | Verbatim: description length, serial depth, state width, input entropy |
| R2 controlled ambiguity | Task Spec §2 dial, §8 step 5 | Aligned |
| R5 traces early then withdrawn | Task Spec §1.2 | Adopted **with the index changed** — see §5.2 |
| R7 sweet spot | A4 (brute-force resistance) + hazard "brute-force collapse" | Covers the too-easy half only |
| §I.6 curriculum DAG via conditional epiplexity | Task Spec §4 $S(\mathcal{T}_j \mid m_i)$; Repertoire §6 all-pairs matrix | **Same measurement.** Work adds plants and pre-registration |
| §IV.3 step 2, ambiguity sweep | Task Spec §8 step 5 | Same experiment, same three-row gate table, **materially better instrument** — §3.1 |
| §IV.8 prequential, slope not intercept | Task Spec §4 | Aligned |
| §IV.5 Track B′ (annotate real traces) | Repertoire §2.6, §11 step 3 | Attempted; **gate not cleared** |
| §IV.4 Track B (hand-write programs for real tasks) | — | **Substituted** by literature excavation — §4.1 |
| §IV.3 step 1, ECA 15/30/54 | Task Spec §8 step 4 planted basis | **Substituted** — §4.2 |
| §IV.8 SDL, task-relative instrument | — | **Absent** — §4.3 |
| §IV.9 step 3, R5 discriminator | — | Absent; docs/05 attacks R5 from a different side |
| §IV.9 step 5, MVG vs OWT and rule 54 | Repertoire §10 condition 5 | Downstream by design, and **de-confounded better than the framework does** (§0.1) |
| §II.6 curriculum controller $\kappa$ | Task Spec §9 | **Contradicts** — §5.1 |
| §II.8 translation, aligned pairs, phase change | — | Absent entirely — §6 |
| R3 behavioural dedup, R4 minimality | — | Absent — §6 |
| §I.1–§I.8 the algebra of biases | — | No counterpart, and one consequence: §4.3 |

---

## 2. Where the work is straightforwardly aligned

Worth stating so the disagreements below are read in proportion.

- **The inference axis is adopted exactly, including v4's self-corrections.** Retrieval sits on the complexity axis as input entropy, not on the level dial. Calibration is the irreducible cell. The work reproduces the v3→v4 correction rather than the v3 ladder.
- **The sweep is on the critical path in both documents, for the same reason.** Framework §IV.9: "Step 2 is placed second deliberately… the only experiment on the list that can end the program." CONTINUE-HERE §8: "The most valuable single result available is not ours to produce… it can end the programme." Same judgment, arrived at separately, and the work is closer to executing it than the framework's own sequencing would have it be.
- **Prequential, slope not intercept.** Task Spec §4 is the framework's §IV.8 recommendation with the compute-budget-fixed clause attached.
- **The inversion warning is carried.** Framework §II.6: "a controller doing the right thing will look like it is doing the wrong thing on the dashboard." docs/02 §4 lists it as a *do-not-abandon-for* condition — which is the operational form the framework asks for and does not supply.
- **A3 and Proposition 7 converge.** docs/11 reverses a format-matching decision on the argument that "format is not a target, invariance is" — reached from A2/A3, and identical in content to the framework's transport law (re-encoding costs $K(\sigma)$, so chasing an encoding is the least significant term).
- **Pre-registration is stronger here than the framework asks for.** §IV.7 asks for inter-author agreement; the work has an append-only prediction document, nine executable assertions with thresholds fixed and a test proving each can fail, and a rule that a missing family scores `UNTESTABLE` rather than `PASS`.

---

## 3. Where the work is ahead of the framework

Six findings with no counterpart in v4, four of which would change it.

### 3.1 The programme-ending gate, as the framework states it, fires spuriously

The framework's gate: *monotone collapse in $H$ → real ceiling → stop and redesign around $H=0$.* This is the single decision in the document that terminates rather than redirects.

docs/10 §6: structural content is the area above **the run's own floor**, and the floor *rises* with residual entropy because part of the loss becomes irreducible. So $S$ falls with $H$ for a reason that has nothing to do with whether supervision works. **A naive sweep manufactures the monotone collapse out of arithmetic.** The fix is already the Task Spec's wording — "structural content *and transfer*" — because transfer holds the evaluation target fixed so the floor does not move under it. Transfer is the arm that decides; $S$ is reported beside it.

This exposure is not hypothetical for the framework: Finzi et al.'s masked-latent design, which §IV.3 explicitly mirrors, has the same shape.

### 3.2 The sweep's x-axis must be measured, and for some families it lies

The framework inherits Finzi's "sweep the number of hidden bits $h$" — a *nominal* dial. Here the map from dial to residual entropy is nonlinear and family-specific, so `entropy.py` enumerates $H(y \mid \text{context}, e)$ exactly.

Sharper, and this is the part with teeth. Measured on parity:

| free observations | $H(y \mid \text{ctx})$ | $H(\theta \mid \text{ctx})$ | $\theta$ alive |
|---|---|---|---|
| 0 | 0.693 | 5.54 | 297 |
| 4 | 0.664 | 2.79 | 19 |
| 8 | 0.279 | 0.65 | 2.1 |

For a balanced family the answer entropy is **flat at $\log 2$ across the first half of the dial while the hypothesis space collapses by a factor of fifteen.** A sweep plotted against residual answer-entropy alone reports that nothing changed across the region where almost everything changed. The framework's axis is $H(p \mid \text{context})$ — entropy over the *task* — but its named estimator and its L0–L3 conditions are written over the *answer*. Those come apart, and which one you plot decides the shape of the curve.

### 3.3 The inference axis is not one-dimensional

docs/10 §2 decomposes the four levels into three independent settings: `reveal` ∈ [0,1], `query_source` (sampled | model), `target_mode` (realized | exact posterior). L0–L3 are four corners of that cube.

Two consequences the framework does not have:

- **The L0–L1 dial is `reveal`, and it is the only continuous one.** The framework asserts a continuous sweep between rungs without noticing that the interface has no way to express the interior (fixed here with `partial_preamble`, plus a non-obvious requirement: the rendered preamble must be the *same length* at every fraction, or the dial moves sequence length alongside entropy).
- **L3 conflates two separable things.** Identifiability is a property of the episode; scoring against the exact posterior is a supervision choice, and it is available at **every** level — strictly better supervision than a sample from the same distribution. The framework's partition-by-mode-of-reduction ("it needs no removal, it yields to observation, it yields only to action, or it yields to nothing") reads as exhaustive because it silently fuses these.

### 3.4 $H_T$ is exactly computable here, and it has a second term nobody expects

We own $P_\Theta$, so the residual entropy is enumerable rather than estimated. That means every family can be placed on R7's $(H_T, S_T)$ sweet-spot table with an exact first coordinate — the framework treats both as estimation problems.

But the floor has two terms. The first L1 run measured 4.18 nats against a computed floor of 0.69 and looked 3.5 nats from optimal. It was not: the stub's answers are binary, so 0.69 nats is the entire *rule* uncertainty, and the other 3.5 nats are uncertainty about **which token denotes which answer** — because $e$ is hidden on the same footing as $\theta$ and at trial 1 the answer symbols have never appeared. The model was near-optimal; the instrument was wrong.

**This generalizes past this project.** Any per-token loss comparison across corpora with different alphabet sizes or different symbol-reuse rates carries a notation term. That is every cell of the transfer matrix — and it is also the framework's own headline quantity, bias density $S_T/|X|$ compared across OpenWebText, rule 54 and a synthetic generator.

### 3.5 L2 is not available at SFT throughput

Framework §II.4: *"Generation stays cheap at every level: because $p$ was constructed, the optimal query policy is known, so the teacher trace remains free and remains at SFT throughput. No reward model, no RL."*

Half of that survives contact. docs/10 §3: because the oracle must answer **what the model actually asked**, an L2 episode needs $T$ sequential generation steps *inside the training loop at current weights*. The loss stays local and nothing waits for a terminal reward — it is genuinely not RL — but the data distribution is **on-policy and moves as the model does**, and the cost is roughly (query width × $T$) extra forward passes per episode. Generation is free; training is not.

Plus a cold-start fact worth budgeting for: at initialization essentially every emitted query is malformed (~0.06% well-formed by chance on the stub), so L2 has a phase where the answer channel never fires and a budget that is too short spends all of it teaching query *format*.

### 3.6 Instrument bugs the framework's checks would not catch

- **Prior ≠ support.** A posterior computed against a uniform prior over the enumerated support is the exact posterior *of a different family* whenever the sampler is non-uniform, and it looks entirely reasonable. Measured on parity: **31.4× deviation**, arithmetic confirmed. Anyone building the framework's calibration cell inherits this bug class.
- **Underpowered sweeps have a shape.** A 60-step smoke run produced a confident "rising" verdict from a model that never left chance. `fit_to_read()` gates the reading on the runs (≥35% of distance to the Bayes floor closed; ≤50% of runs still falling) and prints READING REFUSED otherwise. The framework's gate has no power condition at all, on the one experiment it says can end the programme.
- **Non-identifiability of the decomposition is provable, not incidental.** The Q-matrix literature (docs/02, 2026-08-07 entry) has formal results and worked examples of *distinct* decompositions producing identical response data, misspecification misclassification up to 44%, and a published case where a fitted, converged model assigned 74–98% skill mastery to students who scored **zero**. The framework's answer to Gap 1 — "stage selection becomes a topological order on a measured graph rather than an assertion" — has no identifiability argument. Recovering plants is necessary and **not sufficient**, which is why docs/02 added a sanity check that a near-zero-content family must not come out attributed with capabilities.

---

## 4. Substitutions — where the work does a different thing under the same name

### 4.1 Track B was replaced by literature excavation, and the saturation curve now measures something else

| | Framework §IV.4 | Here |
|---|---|---|
| Material | 200–300 hand-written minimal solver programs for real tasks in the sharp $\mu$ | ~27 task families excavated from six-plus literature veins |
| Saturation x-axis | distinct primitives vs **tasks processed** | distinct primitives vs **sources processed** (14 over 11) |
| Guard | 2–3 independent authors, inter-author agreement on primitive usage | one subagent at a time, curator verification, controlled vocabulary |
| What flattening means | the basis covers practice | **this literature has stopped yielding new apparatus** |

The excavation is well argued on its own terms (Repertoire Spec §1: the route between stipulation and enumeration; external validation; better plants than invention supplies). It is not Track B, and three things follow:

- **Claim 5(a) is untested.** The framework's compactness gate — "still climbing linearly at task 300 → basis unbounded, dead" — is not what "five veins added zero" reports. Five veins adding zero is a fact about the review's coverage of prior apparatus.
- **The veins are not $\mu$.** II.0 says $\mu$ ranges over *practices*. These veins range over other researchers' apparatus, selected for properties that made them publishable in their own fields — human discriminability, worst-case query complexity, asymptotic hardness. The Repertoire Spec names this itself ("transplant rejection… expect a substantial import mortality rate"), which is the right posture, but it does not make the extracted basis $\mu$-aimed.
- **The repair loop has nothing to repair against.** Repertoire §7 carries the framework's per-failure repair ("which primitive would have made the solver program short?") verbatim, but there are no downstream task failures to run it on, because the exercise that produces them is the one that was substituted away.

**Both of the framework's extraction exercises are currently not running.** Pass 1 (primitives) substituted; pass 2 (control structure, = Track B′ = §2.6) attempted and the inter-annotator gate not cleared. The framework calls both free and puts both on the critical path for their respective axes. Note that Task Spec §5 already anticipates *why* they are two passes — "a program that solves a task need not search the way the human did" — so the work knows Track B alone is L0-biased. The gap is that neither pass exists.

### 4.2 Instrument validation is internal, not external

Framework §IV.3 first experiment: reproduce ECA rules 15/30/54, with the ordering stated in advance, because *"failing to means it is misconfigured, and every number produced afterwards would have been noise."*

Here, validation is Task Spec §8 step 4 — planted duplicates cluster, junk reads near-zero, a known prerequisite comes out with the right sign — plus a set of checks the framework does not specify (Bayes-floor leak check, `check_prior_matches_sampler`, `check_query_sensitivity`, `fit_to_read`, the A3 radical/incidental test).

Those checks are in several respects *stronger* than the ECA reproduction: they caught real bugs (§3.4, §3.6) that reproducing an ordering would not have surfaced. But every one of them is computed by this code, against these families, using this estimator. **Nothing in the project has been checked against a number produced by somebody else.** A systematic error in the prequential estimator — budget-sensitivity that differs across families, say — passes every plant and every check.

There is a cheap fix, and it is cheap enough that not doing it needs a reason: **rules 15/30/54 are a `TaskFamily` at L0.** Deterministic map, $\theta$ = the rule, evaluate-only, A1 trivial. Three more families through the existing harness buys the external anchor, makes this project's numbers commensurable with the literature it is arguing against, and simultaneously produces the rule-54 baseline that Repertoire §10 condition 5 requires. Grep confirms the strings `epiplex`, `rule 54`, `cellular` and `surplus description` appear nowhere in `src/`.

### 4.3 There is no $\mu$-relative instrument — only the task-agnostic half

Framework §IV.8 is explicit that two instruments are required and **neither substitutes for the other**: $S_T$ answers "is there anything here to learn", SDL answers "are we aimed at the right thing", and §I.7's regret theorem is why — *high $S_T$ on the wrong $\mu$ is structure you cannot use.*

The work implements the first (prequential structural content, plus the conditional form). Surplus Description Length, held-out downstream targets, anything that scores aim rather than density: absent, and not merely unbuilt. The held-out partition in docs/00 is a *reading* seal over domains and sub-literatures — Euclidean construction, SQL, incident response — none of which has a generator or a scoring path in this harness. Repertoire §10 condition 3 ("solver programs for uninspected domains are short under the basis") therefore has no instrument at all: evaluating it by hand *is* Track B on the held-out side, which is §4.1's hole seen from the other end.

Net: every number this project can currently produce is in-substrate, and the framework's central warning about them is currently unfalsifiable here.

---

## 5. Contradictions — one of the two documents is wrong

Ranked by how cheap it is to fix now versus later.

### 5.1 The controller: an ordered sequence with backtracking, against an explicit prohibition

Framework §II.6: *"Catastrophic forgetting requires that $\kappa$ emit a **distribution** over all types with mass retained on mastered ones — never a pointer into an ordered list. This is what reconciles the framework with the weak empirical record of curriculum learning: hard curricula hurt, shifting mixtures help."* The validated instantiation is ADO, which reweights subsets continuously.

Task Spec §9: *"select a family; train to mastery or stall; checkpoint; probe including stale families. On stall, restore weights to an ancestor and branch elsewhere."* Task Spec §7: batch composition *"defaulting to single-family."* Task Spec §7 again, on seeding: *"the whole design depends on being able to re-run a branch after a backtrack."*

That is a tree search over single-family stages — the shape the framework names as the failure mode — and the `staleness` signal *measures* the forgetting it produces rather than preventing it. **The framework is probably right and this is cheap to change now**, because nothing downstream is built on §9 yet. The branch/restore machinery is worth keeping as a recovery path; it should not be the mechanism.

### 5.2 The trace schedule is indexed on the wrong variable

Framework R5's rationale is *brute-force affordability*: traces early when the untraced task is out of reach, then withdrawn once the trace opens a shortcut the learner can afford — a function of compute and competence.

Task Spec §1.2: *"emit traces at low $k$, thin them as $k$ rises"* — a function of task difficulty. These come apart exactly where it matters: at high $k$ with a weak model the trace is the only thing making learning possible, and the Task Spec withdraws it there.

docs/05 found the same thing from an unrelated direction: the worked-example/fading literature fades against **learner competence**, never task difficulty, and fading only bites when load is high. Two independent arguments say competence; the spec says $k$. **docs/05 records this and the Task Spec text has not been changed.** Either re-index it or state explicitly that $k$-indexing is retained against the finding, with the reason.

### 5.3 "SFT throughput at every level" does not survive L2

§3.5 above. The framework's cost claim for the rung it calls the highest-value member of the family needs the on-policy qualification. The work is right here; the framework needs the edit.

### 5.4 The inference axis is exhaustive only under two unstated assumptions

Framework §II.4: *"the axis is therefore not 'how hard is the task' but by what means the residual entropy can be removed, and that admits an exhaustive partition."*

docs/06 ran the coverage check: 7 of 10 established paradigms fall out cleanly; the 3 that do not cluster into exactly two axes the dial does not measure —

- **Validity duration.** $\theta$ sampled once and held is load-bearing (Task Spec §1) and false for reversal learning, WCST and drifting-$\theta$ paradigms. These are not L1 ($\theta$ does not persist) and not L3 ($\theta$ *is* identifiable, just not stably).
- **Query cost.** L2 as specified makes querying free — a bad query costs only the turn. In bandits and card-sorting the query **is** the scored action, which forces an exploration/exploitation trade the current L2 cannot express.

The partition is exhaustive *given* those two assumptions, and the framework states neither. The cost is concrete: `belief-state-reset` — the primitive closest to the programme's stated priority of error recovery, which the framework calls the single situation type that would justify the programme economically — exists because of paradigms the formalism cannot express.

The work's decision not to build the repairs until after the sweep ("adding axes first would be designing the answer") is the right call and matches the framework's own reason for preferring a sweep to a three-point test.

### 5.5 Stochastic semantics: the framework asks for mass, the spec asks for one

Framework §II.7: *"Let $[\![p]\!]$ return distributions, not values, and put real mass on stochastic programs. A model that has only seen deterministic ground truth will be badly calibrated on everything humans actually ask."* Task Spec §1.3 makes it optional and asks that *"at least one family in the repertoire should exercise this"*; one does (`probability-matching`). That is a floor, not mass, and V.6 (R6's axes are stated for deterministic programs and need extending) is untouched.

---

## 6. Absent, with what it would cost

| Framework item | Cost to add | Why it matters |
|---|---|---|
| **§II.8 translation vs transfer, aligned pairs, the phase-change prediction** | Large, and out of scope until the sweep reads | The mechanism by which any of this pays off downstream. Nothing here touches it |
| **Claim 4′ cross-rendering eval** | Small | A3 varies $e$ per episode *during training*, making the model invariant by construction. Claim 4′ wants train-on-one, evaluate-on-a-held-out-rendering — which *measures* invariance. These are different experiments and the second is what licenses the cheap in-loop research loop. Hold one encoding out of $\mathcal{E}$ |
| **R5 discriminator (looped vs non-looped $S_T$ turnover)** | GPU-days | The framework's own flagged most-likely-error. docs/05 argues against R5's indexing but not against R5 |
| **R3 (dedup by extension on a probe set)** | Small | Near-duplicate *plants* are deliberate; nothing checks that two families are not *accidentally* extensionally identical, and accidental duplication is exactly what corrupts block structure |
| **R4 (minimality search)** | Medium | Only bites if description length is ever used as a difficulty label |
| **Input-entropy families / distractor tolerance** | Small | Carried as a coverage axis and as intent in docs/11; no family's knob is input entropy at small $T$ |
| **V.9 drift test** (re-extract on five-year-old material) | Small | Repertoire §7 states it verbatim; unrun. Decides whether the extraction is one-time or standing |
| **V.10 weight-space assembly** | — | Open in the framework too; correctly absent |
| **Retrieval / the knowledge phase** | — | Out of scope by design; the framework agrees it is a consequence, not a generator |

---

## 7. What this implies for the critical path

Ordered by value per unit cost.

1. **Run the sweep.** Both documents agree it is the only step that can end the programme; it is built, budget-named, with the reading pre-committed and a refusal gate. Blocked only on parity's `prior_weight` for the A4 control — one method, supplied verbatim in docs/10 §5. Everything else in both documents is downstream of this number.
2. **Add ECA rules 15/30/54 as three L0 families.** ~a day. Buys the external instrument anchor the plants structurally cannot provide (§4.2), makes this project's numbers commensurable with the literature, and produces the rule-54 baseline Repertoire §10 condition 5 needs anyway.
3. **Fix Task Spec §9's controller to emit a mixture** before anything is built against it (§5.1).
4. **Re-index the trace schedule on competence, or record the retention and its reason** (§5.2).
5. **Decide the Track B question explicitly** (§4.1): either run a scaled-down version — 30–50 hand-written solver programs over the *inspected* domains, plotting primitives against tasks — or amend the Repertoire Spec to say that its saturation curve measures literature exhaustion rather than basis completeness, and that Claim 5(a) is untested here. Right now the two readings are conflated in a number that appears in three documents.
6. **Add a held-out encoding** to make Claim 4′ measurable (§6).

### Bookkeeping found in passing

- `register/reading-log.toml:219` still references row `compaction-survival`, renamed to `evidence-withdrawal` in docs/11. `saturation` prints a warning; `validate` does not.
- README and CONTINUE-HERE both say **101 tests**; the suite runs **207**. README says the hazard register lists "fourteen things"; it lists **25**.
