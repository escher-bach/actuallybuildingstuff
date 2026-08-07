# Predicted Decomposition — PRE-REGISTERED

*Repertoire Spec §6: "Write down the decomposition you expect — including the one your sources would predict — before running the matrix, and compare afterward. A mismatch between the field's inherited taxonomy and the measured blocks is the single most valuable result this process can produce, and it is unavailable if the prediction is made after the fact."*

**First written 2026-08-06**, before any vein report had been read and with one register row in existence (the calibration row). Revised only by **dated append** in §5. Text above §5 is never edited — a prediction that gets tidied up after the fact is not a prediction.

---

## 1. Predicted blocks

The guess is that a ~30-family candidate set collapses to **four or five** blocks, not the six-to-eight the coverage axes might suggest.

| # | Block | What it is | Families expected in it |
|---|---|---|---|
| B1 | **Binding and substitution** | Hold a partial map from symbols to values across trials; apply it in either direction | hidden-permutation arithmetic, ciphers, variable binding, relabelling transfer |
| B2 | **Sequential state** | Carry a machine's state forward over steps; the answer depends on history, not just the current input | stack/register machines, plan simulation, anything with serial depth as its knob |
| B3 | **Hypothesis narrowing** | Maintain a set of candidate rules and cut it against evidence | concept learning, grammar/DFA induction, L1 identification, **and L3 calibration** |
| B4 | **Compositional decomposition** | Invert a composition — given the result of a pipeline, recover or apply the parts | composite families, parsing, corrupt/localize |

## 2. The three predictions that can actually be wrong

Blocks alone are cheap to be vaguely right about. These are the falsifiable claims.

**P1 — L3 does not form its own block; it merges into B3.** Hypothesis narrowing and posterior maintenance are the same operation read out at different temperatures. L1 asks for the argmax once the set is a singleton; L3 asks for the distribution while it is not. If the matrix puts calibration families in their own block, well-separated from identification families, P1 is wrong and calibration is a genuinely separate capability — which would be a more interesting result than the prediction.

**P2 — L2 is not a capability, it is a wrapper.** An L2 family should cluster with the L1 version of *the same underlying rule class*, not with other L2 families. If instead the L2 families cluster with each other across unrelated rule classes, then "choosing an informative query" is a transferable skill in its own right, the level dial is a capability axis after all, and the §4 coverage grid is mis-shaped — it treats level as an axis to span rather than as a factor that interacts.

**P3 — complexity mode predicts block membership better than content does.** Serial depth and description length are expected to separate families more sharply than the manipulation/state/structure-induction/selection distinction does. If true, the §4 content axis is a convenience for tracking coverage and not a decomposition, and it should not be read as one. If false — if content predicts and complexity mode does not — then the four complexity modes are one axis wearing four names.

**Prerequisite DAG.** B1 is expected to be a near-universal prerequisite: binding is what everything else manipulates, so the antisymmetric part should show large positive edges from B1 into B2, B3 and B4 and weak reverse edges. B4 is expected to sit downstream of B2 and B3 both. The sharp version: **B1 should have out-degree close to the number of blocks and in-degree ~0.** A DAG with no such near-source is evidence against the whole "prerequisite ordering" reading of the antisymmetric part, not just against this guess about which family it is.

## 3. What our sources would predict — quarantined

*Filled in as each vein is read, from that vein's own taxonomy. Kept here, separated from §1–§2, so it can be scored against the measurement rather than absorbed into our own expectation. §9's first failure mode is importing the source's taxonomy along with its tasks; this section is the quarantine that makes taking the apparatus and leaving the ontology an auditable practice rather than an intention.*

| Vein | Its own grouping | Would predict |
|---|---|---|
| §2.2 | **Oracle access pattern** (Angluin): classes grouped by which oracle types suffice — membership-only, equivalence-only, MQ+EQ, subset/superset. **Teacher–learner cooperation** (teaching-dimension line): classical vs. recursive vs. preference-based vs. no-clash teaching, ordered by how much collusion is permitted. **Computational strategy** (modern BED): myopic/greedy vs. non-myopic/amortized, implicit vs. explicit likelihood. **Proof applicability** (adaptive submodularity): whether the objective is adaptive-submodular | Blocks drawn along *hypothesis-class* lines — conjunctions with conjunctions, automata with automata — and a separate block for anything requiring an equivalence oracle. Note none of these four axes is about task content; all four are about the *analyst's* access or proof situation. If our measured blocks reproduce hypothesis-class boundaries, that is a live possibility worth taking seriously, but it is also what the citation structure of this vein would produce by default, so it is weak evidence |
| §2.1 | **Cognitive operations and ability factors** — items grouped by the processing steps a cognitive design system attributes to them; **Q-matrix / skill attribution** — items grouped by which latent skills a diagnostic model says they require, estimated from human response data | Would predict blocks along the radical structure of item models: number of rules, type of transformation. Note this is the one vein whose taxonomy is *methodologically* the closest to ours — a Q-matrix is a capability decomposition recovered by measurement — so agreement with it would be the least surprising and least informative match, and disagreement the most interesting |
| §2.3 | **Paradigm membership** — the six SHJ types are one task ordered by difficulty; concept attainment, learning sets, set-shifting and probability learning are separate literatures. Clinically, **executive-function constructs** (perseveration, set-shifting) group the card-sorting work. Decision research explains probability matching by **strategy selection** | Would predict all six SHJ types in one block, ordered I→VI. **We predict the opposite and it is testable now:** SHJ Type VI is parity, i.e. `modular-add` over a subset of dimensions, so it should sit with `parity-identification` from §2.2 rather than with its own paradigm-mate `shj-type-i`. If paradigm membership beats structure here, translating across fields is buying less than it appears to |
| §2.4 | | |
| §2.5 | **Test-suite adequacy and the coupling effect** (mutation testing): operators grouped by how well killing simple mutants predicts killing complex ones. **Diagnosis minimality** (model-based diagnosis): conflict sets, minimal hitting sets, and a preference for fewest faulty components. **Failure mode** (plan validation): inexecutable vs non-goal-reaching. **Error type** (proof-failure taxonomy): 10 categories of naturally-occurring model errors | Mutation testing predicts nothing usable — its theory is about test-suite quality, not about which capability a task exercises, which is itself worth noting. Model-based diagnosis predicts **diagnosis cardinality** (single- vs multi-fault) as the primary difficulty axis. Plan validation predicts a clean binary split. If our blocks come out as a single corrupt/localize cluster spanning all four substrates, all four of these taxonomies are wrong together, and that would be the more interesting result |
| §2.6 | | |

**The scoring rule, fixed now:** after the matrix runs, compute how well each of (a) our §1–§2 prediction and (b) each source taxonomy in this table recovers the measured blocks, by the same clustering-agreement measure, chosen before seeing the matrix. If a source's inherited taxonomy beats our prediction, that is a real finding and gets reported as one. The temptation this rule exists to block is reading the measured blocks, noticing they resemble *some* taxonomy in the table, and reporting that as convergent validation.

## 4. Abandonment conditions

*§9, "building for the eval you have": every convenient benchmark measures L0-ish competence, so a repertoire deliberately targeting L1–L3 will look bad on the dashboard for a long time. Decide in advance what would make you abandon the design rather than the dashboard.*

**Abandon the design if:**

1. **The acquisition slope does not rise across a curriculum** (Task Spec §9) even though the basis measures well by §10 conditions 1–3. That is the meta-learning quantity the whole programme exists to install. If it is flat, the basis is measuring something real that is not the thing we wanted.
2. **The dial sweep collapses monotonically** (Task Spec §8 step 5). Already stated there as programme-ending for L1–L3; restated here so it is not quietly reinterpreted as "the sweep was built wrong" when it arrives.
3. **The saturation curve is still climbing linearly at ~300 sources.** The basis is unbounded, the compactness claim (§10 condition 2) is false, and "capability is the manufacturable half" loses its main support — you cannot manufacture an unbounded basis.
4. **The measured blocks reproduce an imported taxonomy exactly.** This is the one to be suspicious *of*, not pleased by. Exact agreement is more likely to mean the candidate set was built to that taxonomy than that the taxonomy was right.

**Do NOT abandon the design for:**

- Poor scores on L0-shaped public benchmarks. That is the dashboard disagreeing with the design, which §9 says to expect and for a long time.
- **High absolute loss on the families with the most structure.** Task Spec §9 names this inversion explicitly: more learnable structure means higher absolute loss, so a controller doing the right thing looks wrong on a loss dashboard. It is called out there as "the easiest way for the work to be abandoned for the wrong reason."
- A high import mortality rate from the veins. §9 says expect it; a well-attested, decades-validated family measuring near-zero for us is transplant rejection, not evidence the vein was wrong.
- Losing to Rule 54 on structural bits per token *by itself*. Per §0.1 that comparison is confounded by the generator and the harness, and cannot be credited or blamed on this document's method alone.

## 5. Dated revisions

*(append only — date, what changed, and what prompted it)*

**2026-08-06.** §3 row for vein §2.2 filled from the query-learning review. §1–§2 unchanged; nothing in that report moved the prediction. Recording one thing it *did* surface, because it bears on P2 and I do not want to be able to claim afterward that I had it in mind:

The review found that the teacher's query cost splits cleanly in two, and not along the line the level taxonomy suggests. For some rule classes `q*` requires touching the belief state and costs O(|surviving Θ|) (`informative-query-selection`); for others the teacher walks θ's own structure or runs a fixed algebraic schedule and touches no belief state at all (`structure-walk-query`, `basis-probe`), costing O(|θ|) regardless of how much is still unknown. This is a property of the *rule class*, not of the level.

If P2 is right — L2 is a wrapper, not a capability — then this split should be invisible in the block structure, because it is a fact about the teacher's implementation rather than about what the model has to learn. If instead the matrix separates structure-walk families from belief-state families, P2 is wrong in an informative way: it would mean the model is learning something that tracks how its teacher generated the curriculum, which is a result about supervision leakage and not about capability. Both readings are on the table before the measurement, which is the point of writing this down now.

**2026-08-06, second entry — a candidate axis the four levels do not have.** §1–§2 unchanged; recording a §2.3 finding that bears on P2 and that I do not want to be able to claim afterward I had anticipated.

The four levels vary *how much the context determines the answer*. §2.3 turned up two things that vary something else and are not placeable on that dial:

1. **Query/reward entanglement.** In L2 as specified, emitting a query is free — a poor choice yields an uninformative answer and costs only the turn. In bandits, card-sorting and reversal learning, the query **is** the scored action; there is no test-then-be-scored separation. Both involve the model choosing `x ∈ X`, so both look like L2, but the cost structure is entirely different. This is a **cost axis, orthogonal to the identifiability axis** the levels are built on.
2. **Validity duration.** Piecewise-stationary θ (`wcst-reversal`) is not L1 — θ is not fixed for the episode — and not L3 — θ *is* identifiable, just not stably. It varies how long the context stays valid, which the dial does not measure.

**Why this matters for P2.** P2 says L2 is a wrapper rather than a capability. If the entanglement axis is real, P2 could be *right about our L2* and still misleading, because our L2 occupies only the free-query corner of a larger space, and the interesting agentic content may live in the entangled corner we have not built. That would not falsify P2 so much as show it was answering a smaller question than it appears to.

**Not acting on this yet, deliberately.** Task Spec §8 step 5 sweeps residual entropy and expects the named cuts to be in the wrong places; adding axes before that sweep runs would be designing the answer. Recorded here so that if the sweep does relocate the cuts, this observation is on record as prior rather than as a retrofit.

**2026-08-07 — prior art for our own method, and it is not encouraging.** §1–§2 unchanged. Vein §2.1 turned up the closest methodological precedent to §6 that exists, and its failure modes transfer directly.

A **Q-matrix** maps items to the latent skills they require. Diagnostic-classification models estimate one from response data and read a capability decomposition off it. That is our §6 procedure with human subjects in place of tiny models and factor-analytic estimation in place of transfer measurement — near enough that Repertoire Spec §6 already flags "the interpretive pitfalls of that literature apply." What the review found is more specific than that warning:

- **Non-identifiability is provable, not incidental.** There are formal results and worked numeric examples of distinct Q-matrices producing identical response distributions. Two different capability decompositions, same data, no way to choose. Our matrix is not obviously exempt: a block structure is only as identified as the transfer numbers that generate it, and we have no identifiability argument at all.
- **Misspecification degrades estimates severely**, with simulation misclassification reported as high as 44%.
- **The interpretive failure is the one to internalize.** On a well-known published dataset, a diagnostic model assigned skill-mastery probabilities of 74–98% to students who scored **zero**. The model was fitted, converged, and produced a confident, entirely wrong capability attribution. Nobody in that chain was careless.

**What this changes here.** Nothing in §1–§2 — the predictions stand. But it raises the bar for §11 step 6 in a specific way: recovering the plants is necessary and **not sufficient**. A method can recover planted structure and still produce nonsense on the unplanted rows, which is precisely what the fraction-subtraction case shows. So the gate should include at least one sanity check of the kind that would have caught it — e.g. that a family measuring near-zero structural content does not come out attributed with capabilities. Cheap, and it is the check nobody ran.

*(All figures curator-verified where reachable; see register/verification-log.md.)*


**2026-08-07 — the predictions are now executable.** §1–§2 unchanged; this records a change in *form*, not in content.

Everything above is prose, and prose about an unmeasured matrix has a known failure mode: not forgetting the prediction, but **reading the matrix and finding the prediction that fits**. §2.1 supplied the cautionary case — a fitted, converged capability model assigning 74–98% skill mastery to students who scored zero, produced by nobody being careless.

`src/repertoire/expectations.py` states each expectation as an assertion a script runs against a matrix, with thresholds fixed now. Seven entries: four instrument checks that decide the §11 step 6 gate, one sanity check, and the structure-versus-paradigm prediction. Tests establish that **every one of them can fail** on constructed input — hazard 6 in `docs/07` is a check that cannot fail, and it has already been met twice in the A2 machinery, so it seemed worth not meeting a third time on the gate that decides whether the programme proceeds.

Two design decisions worth stating because they could be quietly reversed later:

- **Only instrument checks are blocking.** The structure-versus-paradigm prediction and the SHJ prerequisite ordering are predictions *about the world*; they are supposed to be able to fail, and a failure there is a finding rather than a stop. Conflating "our instrument works" with "our guess was right" is how a gate stops being a gate.
- **A missing family scores `UNTESTABLE`, never `PASS`.** Silently passing on absent data is the other way a gate stops being a gate, and it is the more likely one, since the candidate set is incomplete.


---

**2026-08-07 — P4, and a reframing of what the matrix is for.** §1–§2 unchanged. This adds a prediction rather than revising one, and it is the sharpest claim in this document.

**The reframing.** The programme is not claiming that intelligence *is* any of these families. It is claiming intelligence is a **combination** of them. That distinction has a direct methodological consequence I had not drawn: **an all-pairs transfer matrix over families in isolation measures the wrong object.** It yields blocks of isolated capacities. If the claim is about combination, the matrix must contain **composites**, and the interesting cell is not family-to-family but *parts-to-whole*.

**P4 — composites are not supplied by their parts.**

Define, for a composite `C = T1 ∘ T2` and a model trained on both components:

    residual(C) = S(C | m_{T1,T2}) / S(C)

the fraction of the composite's structural content that survives training on its parts.

**Prediction: `residual(C)` is substantially above zero** — training on parity and on hidden permutations separately does not supply parity-of-permuted-bits. Something is required to *combine* them that neither part contains.

**If P4 is false — if `residual(C) ≈ 0` — the consequence is severe and should be stated now rather than argued away later.** It would mean the basis's power lies entirely in its parts, composition adds nothing a model does not already get from the components, and Task Spec §1.1's claim that composition "is what lets a finite basis cover an infinite target space" is false *in our setting*: covering would be a matter of enumerating parts after all. That is a result against the programme's central mechanism, and it is cheap to test — one extra row and column per composite.

**Two secondary predictions that discriminate further:**

- **P4a — depth is a real knob.** `residual` should *increase* with composition depth. If depth-3 composites are no less supplied by their parts than depth-2, then depth is a difficulty knob in name only, and §1.1's "depth is the difficulty knob that does not require inventing a new family per level" fails.
- **P4b — composites do not join their components' blocks.** If a composite clusters with its own components, combination is not a separate capability. If composites cluster *with each other* across unrelated component pairs, then combining is a transferable skill in its own right — which would be the strongest possible result for the "intelligence is combination" framing, and is the one I would bet against.

**Why this is pre-registered rather than run now:** the matrix does not exist and the harness does not exist. What can be fixed now is the comparison, and it is fixed above with a definition, a direction, and a stated consequence for being wrong. `src/repertoire/expectations.py` carries it as a scored assertion.

**One honest limitation.** Our composition closure is currently **2.4% of ordered pairs** — one legal composite over seven families. P4 tested on a single composite is a case study, not a result. Making it a result requires more endomorphic families, which is now a design requirement on the candidate set rather than an afterthought.
