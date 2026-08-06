# Repertoire Specification

*Companion to the Task Specification §5. That document says what a family must satisfy; this one says which families to build and where to find them.*

*This is a **review protocol**, not a research problem. The families we need have mostly been built already — repeatedly, in fields that never shared a vocabulary and were not building training data. The work is to find them, translate them into one interface, and measure what survives. Do not attempt to derive the repertoire from first principles; that is the failure mode this document exists to prevent.*

---

## 0. The end goal, stated once and plainly

**Purpose.** Task Specification §9 proposes replacing the bulk of text pretraining with a synthetic curriculum, on the claim that capability and knowledge are separable and capability is the manufacturable half. That claim is untestable without a set of task families to manufacture from. **This document produces that set.** Nothing here is worth doing on its own account.

**Deliverables.** Three, and the third is terminal for this document.

| # | Deliverable | Form | Consumed by |
|---|---|---|---|
| D1 | **Source register** — prior task families found in the literature, one row each, translated into the §7 interface fields, triaged against A1–A7 | spreadsheet + prose notes | D2 |
| D2 | **Candidate set** — ~30 implemented families spanning §4's four axes, including deliberate plants | code against `TaskFamily` | D3 |
| D3 | **Measured basis** — block decomposition, prerequisite DAG, pruned family list, read off the all-pairs matrix | basis + DAG + the run that produced it | Task Spec §8 step 5, §9 |

**Success is D3: a basis whose decomposition is interpretable and whose plants recover.** That is the whole of what this document is accountable for.

### 0.1 What this document does not own

Three things are load-bearing for the programme, get referenced throughout, and belong to the Task Specification. They are named here so the purpose of D1–D3 is visible, and nowhere here as deliverables.

| | Owner | Relation to this document |
|---|---|---|
| The **measurement procedure** — prequential structural content, conditional form $S(\mathcal{T}_j \mid m_i)$ | Task Spec §4 | We consume it. If it is not fixed and reported at one budget, §6 here is meaningless |
| The **instrument validation** — planted basis, ~10 families, duplicates and junk recovered | Task Spec §8 step 4 | Gates us. Our §5 supplies better plants than invention would; it does not build the instrument |
| The **thesis tests** — dial sweep, and minimum viable generator vs. a text corpus and vs. Rule 54 on structural bits per token | Task Spec §8 step 5, §9 | Downstream. A basis is the input to them, not a substitute for running them |

The last row is the one worth being blunt about. The baseline comparison is the programme's go/no-go, and it is GPU-days that also depend on the harness, the level wrappers and the controller. **Passing it is not evidence that this document's method worked, and failing it is not evidence that it did not** — a good basis can lose to Rule 54 because the generator over it was built badly, and a stipulated basis could beat Rule 54 by accident. Keep the accounting separate: D3 is judged by §10 conditions 1–4, the programme is judged by condition 5, and conflating them is how a scoping failure becomes an attribution failure.

**What is explicitly not the goal:** a taxonomy of reasoning, a defensible capability list, or a novel family. Novelty here is in the unification and the measurement, not the inventory.

---

## 1. Why this is a review problem and not an invention problem

Two failure modes, and they are opposite.

**Stipulation.** Write down the capabilities you believe reasoning consists of, build a family per capability, discover you built your own taxonomy. Nothing in the result is evidence about anything. bAbI's twenty tasks are the reference case: a stipulated capability list, cleanly executed, saturated within a few years, and the saturation taught nobody what to build next.

**Enumeration.** Build every generator you can think of, train on the union, discover which transfer by exhaustive search. Correct but uninformative, and it is the current field's method.

The route between them is **excavation**. The structure the Task Specification describes — a program with a hidden parameter, a policy for revealing it, an oracle that evaluates and never searches, a difficulty knob orthogonal to generation cost — is not new. It has been independently reinvented in at least a dozen research communities since the 1950s, each time for a local reason, each time under a different name, and never once assembled. Psychometricians built it to manufacture test items at scale. Learning theorists built it to prove query-complexity bounds. Cognitive psychologists built it to isolate concept attainment from prior knowledge. Constraint-satisfaction researchers built it to generate hard instances with known solutions. Cryptographers formalized the generation asymmetry as a primitive object. Animal-behaviour researchers built the entire L1–L3 range before anyone called it meta-learning.

> The field looks empty because it is *fragmented*, not because it is unexplored. **You cannot find this work with machine-learning keywords.** The single highest-leverage artifact in this document is §3's translation table.

So: **over-generate a deliberately redundant candidate set from the record, then let measurement collapse it.** The literature supplies variation, coverage, and — critically — external validation. The instrument supplies structure. Neither alone suffices, and the order matters: measurement cannot discover a family nobody ever proposed.

---

## 2. The veins

Six bodies of work, ordered by what they yield rather than by field. Each entry names the vein, what it supplies, and the specific hazard in importing from it. **Names and dates below are leads to verify against primary sources, not citations** — the register (D1) is where they become claims.

### 2.1 Item generation — the closest prior art, and almost nobody in ML has read it

Educational measurement has been manufacturing parametrized task families at industrial scale since the 1980s: *automatic item generation*, *item models*, *cognitive design systems* (Embretson's work on matrix-reasoning items; Irvine & Kyllonen's collection; Gierl & Lai's later production systems).

The correspondence is close enough to be uncomfortable. An **item model** is a program with slots. **Radicals** are the slot fillers that move difficulty; **incidentals** are those that do not. That is $k$ and $e$, named forty years earlier, with an empirical methodology for telling them apart — you calibrate against measured difficulty and check that incidentals do not shift it. Our A3 is their incidental-variation requirement; our A2 is the "culture-fair" testing programme, including its documented failures, which are evidence about how hard A2 is to actually achieve.

*Yields:* family templates, the radical/incidental discipline, difficulty-parametrization methodology, and validated coverage of a huge item space.
*Hazard:* their difficulty scale is **human** difficulty, tuned to discriminate between people near a population mean. An item that separates humans may be trivial for an inducer and vice versa. Import the parametrization; re-derive the knob.

### 2.2 Query learning and experimental design — L2 already has a theory

Exact learning with membership and equivalence queries (Angluin's L\*), teaching dimension and teaching sets (Goldman & Kearns), and Bayesian optimal experimental design (Lindley; MacKay) are the same object as Task Spec §2.1 viewed from three angles.

A7 asks for a computable teacher query policy $q^*$ maximizing expected information gain. That is Bayesian experimental design's central quantity, with known closed forms for some model classes and known intractability for others. **The question "which families support L2" is therefore substantially answerable from the literature before writing any code** — and §7 flags L2 attrition as the expensive-to-discover-late failure.

*Yields:* $q^*$ implementations, a map of where $q^*$ is cheap, and a principled expected episode length (teaching dimension bounds what $T$ needs to be for identification).
*Hazard:* these literatures optimize worst-case query complexity. We want a *good* query in one pass, not an optimal one; take the heuristics, not the guarantees.

### 2.3 Hidden-rule paradigms from psychology and meta-RL — L1–L3 as an existing repertoire

The Task Spec's four levels are, historically, a description of what animal- and human-learning researchers spent seventy years building apparatus for.

- **Bruner, Goodnow & Austin (1956)** distinguish the *reception* paradigm, where instances are presented to the subject, from the *selection* paradigm, where the subject chooses which instance to test. That is L1 and L2, stated in 1956, with a taxonomy of subject strategies that reads as a catalogue of query policies.
- **Harlow (1949)** learning sets — improvement in the *rate* of within-episode acquisition across episodes — is precisely the acquisition slope of Task Spec §4, and is the primary metric of §9.
- **Shepard, Hovland & Jenkins (1961)** give a complete, ordered basis over a small Boolean concept space with a measured difficulty ordering. A complete basis with a known prerequisite order is exactly the kind of object §5's planted validation needs.
- **Reversal learning, Wisconsin Card Sorting, the Daw two-step, T-mazes, bandits** — and Miconi's parametrization covering them, already noted in the Task Spec.

*Yields:* the largest single stock of L1–L3 families, plus known difficulty orderings usable as plants.
*Hazard:* these are episodic behavioural procedures with tiny stimulus sets, designed for organisms with limited working memory. Most will fail A4 outright at their published sizes. They supply the *reveal structure*; the content must be scaled underneath it.

### 2.4 Instance generation with planted structure — A1 and A4 have a hardness literature

Constraint satisfaction, SAT, and cryptography have thought harder about backward generation than anyone else, because they had to.

- **Planted solutions**: generate the answer first, then the instance around it. That *is* A1, and the literature on planted-vs-random hardness (planted SAT, hidden clique, quasigroup completion with holes, forced-satisfiable Model RB) is directly about whether planting leaks a shortcut — which is A4's question.
- **Phase transitions**: random 3-SAT's clause-to-variable ratio is a difficulty knob with a theory attached, including the finding that difficulty is non-monotone in the obvious parameter. Expect the same for our $k$.
- **One-way functions and trapdoors**: the generation asymmetry axis of §4 is the trapdoor concept. Multiply/factor is the canonical instance. This is the one vein that offers A4 as a *guarantee* rather than a hope.

*Yields:* generators, hardness arguments for A4, and hard-won warnings about knob monotonicity.
*Hazard:* asymptotic hardness says nothing about the small-$k$ regime a tiny model trains in, and planted instances are the classic case of "hard in general, easy for the planting distribution."

### 2.5 Program, proof and plan corpora — the primitive inventory, partly pre-extracted

Task Spec §5's first pass — hand-write solver programs, inventory the operations — has been run before under other headings.

- **Library learning** (DreamCoder and successors) automates it: solve tasks, compress the solutions, extract reusable abstractions, repeat. Its refactoring histories are published primitive inventories with saturation curves already plotted.
- **Grammatical inference benchmarks** (Tomita grammars, the Abbadingo DFA-learning competition) are hidden-rule families with published difficulty parametrizations.
- **Planning** (blocks world, Sokoban, the IPC domain generators, PDDL instance generators) supplies parametrized generators with plan-length knobs, and planning instances are natively backward-generable: apply a random action sequence, then ask for the plan.
- **Mutation testing and model-based diagnosis** (mutation operator catalogues; Reiter-style diagnosis) hand you the corrupt/localize schema pre-populated: decades of catalogued, empirically-validated ways to break a program, and a formal account of localization.
- **Synthetic-pretraining precedents in ML** — LIME's deduction/induction/abduction tasks, SCAN, COGS, PCFG-SET, CLUTRR, the RASP/Tracr line, the controlled-setup "physics of LLMs" work — are the nearest neighbours in our own field, and the most likely to have already tested a transfer claim we are about to re-test.

*Yields:* primitives, generators, and in the library-learning case a partial answer to §5's saturation question.
*Hazard:* these inventories are relative to a chosen DSL. Two extractions over different substrates yield different "primitives." Record the substrate with every entry.

### 2.6 Practice traces — the scarce vein, with an established methodology

Task Spec §5's second pass — control structure, not primitives — is what a text corpus teaches worst and is therefore the priority. Finished artifacts systematically delete it. This is not a novel observation; it is a measured one, and there is a methodology.

- **Protocol analysis** (Ericsson & Simon) is the validated procedure for eliciting and coding think-aloud traces, including the conditions under which verbal reports are and are not veridical. Use it; do not improvise a coding scheme.
- **Problem behaviour graphs** (Newell & Simon) are a notation for exactly what we need to record: states, operators, and *backtracks*.
- **Episode coding for mathematical problem solving** (Schoenfeld: read / analyze / explore / plan / implement / verify) is a published annotation scheme with reported inter-annotator agreement — which is the gate on §9 step 3, already instrumented.
- **Cognitive task analysis** in instructional design reports that experts omit a large fraction of their own steps when describing their work — on the order of 70% in the frequently-cited estimate. That is the quantitative form of "finished artifacts delete the control structure," and it is the argument for why this pass cannot be replaced by scraping.
- **Worked examples, completion problems, fading, and the expertise-reversal effect** (Sweller; van Merriënboer; Kalyuga) are the human-subjects version of Task Spec §1.2's trace-thinning schedule — including the empirical finding that traces help novices and *hurt* experts, which is the same shape as "emit traces at low $k$, thin them as $k$ rises." That default is not a guess; it has a literature.
- **Documented reasoning practice**: Pólya's heuristics, Lakatos's proof-and-refutation dialogue, published proof attempts, debugging sessions, agent transcripts.

*Yields:* the control structure — the substance of L1–L3 — plus annotation methodology and a defensible thinning schedule.
*Hazard:* the most expensive vein per unit yield, and the one most likely to be skipped for that reason. Budget it first (§9 step 3).

---

## 3. Translation table

The veins above are unreachable by keyword search from within ML. Search on the left; write down the right.

| Their term | Our term | Where |
|---|---|---|
| Item model / item shell | task family $\mathcal{T}$ | §2.1 |
| Radical | difficulty knob $k$ | §2.1 |
| Incidental | encoding $e$, sampled per episode (A3) | §2.1 |
| Culture-fair / reduced cultural loading | knowledge-free (A2) | §2.1 |
| Q-matrix / rule space / skill attribution | the block structure of the matrix (§5) | §2.1 |
| Membership query | L2 model-emitted query $x_t$ | §2.2 |
| Active learner / optimal design | teacher policy $q^*$ (A7) | §2.2 |
| Teaching dimension | episode length $T$ to identification | §2.2 |
| Reception paradigm | L1 | §2.3 |
| Selection paradigm | L2 | §2.3 |
| Learning set / learning to learn | acquisition slope (§4 of Task Spec) | §2.3 |
| Probability matching / partial reinforcement | L3 calibration target | §2.3 |
| Planted solution / forced satisfiable | backward generation (A1) | §2.4 |
| One-way function / trapdoor | generation asymmetry (§4 axis) | §2.4 |
| Order parameter / phase transition | $k$, with non-monotone difficulty | §2.4 |
| Library learning / abstraction refactoring | primitive extraction, §1.1 pass | §2.5 |
| Mutation operator | corrupt/localize schema instance | §2.5 |
| Think-aloud protocol / problem behaviour graph | practice trace, §1.2 pass | §2.6 |
| Faded worked example / completion problem | trace thinning schedule (§1.2) | §2.6 |

Two further search disciplines. **Search by structure, not by topic**: the query is "hidden parameter, revealed over trials, generator owns the answer," and it surfaces in venues with no shared citation graph. **Follow the apparatus, not the finding**: in behavioural literatures the reusable object is the procedure, and it is usually described in the methods section of papers whose results are irrelevant to us.

---

## 4. Coverage before elegance

The candidate set must span, and be *known* to span, four axes. Redundancy along each is intended. Track coverage on this grid **as the review proceeds** — the gaps tell you which vein to read next.

| Axis | Span | Failure if narrow | Vein most likely to fill it |
|---|---|---|---|
| **Level** | L0–L3, several families each | Everything published is L0; a repertoire that repeats this is a faster route to what exists | §2.3, §2.2 |
| **Complexity mode** | description length, serial depth, state width, input entropy | Vary only description length and reasoning transfer comes out thin | §2.4, §2.5 |
| **Content** | manipulation, state, structure induction, selection, composition | Gaps here are invisible until downstream failure | §2.5, §2.1 |
| **Generation asymmetry** | evaluate/search, execute/infer, corrupt/localize, compose/decompose | The A1-admissible directions. Each is a family *template*, not one family | §2.4, §2.5 |

The fourth axis is the generative one. **A1 says search the asymmetric direction; each asymmetry is a schema that spawns many families.** Corrupt/localize alone gives fault injection into programs, into proofs, into state machines, into plans — four families, one schema, all backward-generable, all landing on error recovery, and all with a pre-existing operator catalogue behind them (§2.5).

### 4.1 The translation step, and where imports die

A found family is not a candidate until it round-trips through the §7 interface. Triage each register row:

1. **Is $\theta$ separable from the instance?** If the source generates instances without ever naming a hidden parameter, either recover $\theta$ or reject. Most benchmark generators fail here and are repairable.
2. **A1 — is the oracle evaluate-only?** Anything requiring a solver on the generation path is rejected or inverted. Ask whether the *reverse* direction is admissible; that is usually where the family lives.
3. **A2 — permuted-alphabet check.** Human-facing tasks smuggle in semantics through content words constantly. This is the most common silent failure on import from §2.1 and §2.3.
4. **A4 — does it survive scale-up?** Small behavioural paradigms usually do not. Note the composition or size increase required.
5. **A7 — is $q^*$ computable in one pass?** Answer from §2.2 where possible, before implementing.

Record the verdict and the repair, not just the verdict. **A rejected family with a named reason is a result**; the reasons cluster, and the clusters are informative about the interface.

---

## 5. Deliberate contamination

Plant things whose answer you know. Without them the measurement is uninterpretable. The review makes better plants available than invention would: a prerequisite pair with a *published, replicated* human difficulty ordering (§2.3) is a stronger plant than one we merely find obvious.

| Plant | Purpose | Expected result |
|---|---|---|
| **Near-duplicates** | Two families, different surface, same latent operation — ideally two independent reinventions found in different veins | Cluster tightly. If not, the instrument cannot see identity |
| **Junk — random** | Targets sampled independently of input | Near-zero structural content |
| **Junk — trivial** | Constant or near-constant targets | Near-zero, for the opposite reason |
| **Known prerequisite pair** | $A$ obviously required for $B$; prefer an externally-attested ordering | Antisymmetric part has the right sign |
| **Known independent pair** | No shared latent structure | Off-diagonal near zero both ways |
| **Suspected junk** | Families you believe are useless but cannot argue away | *Genuinely unknown.* Include them — this is where you learn |

The last row is the one people skip. If every plant has a known answer, the run only validates the instrument. Include candidates you cannot defend; the point of measuring is to be told something.

**Gate:** if the known plants do not come out right, stop. Measurement cannot guide extraction and this method is void before any expensive step.

---

## 6. What the matrix yields

Train a tiny model per family. Measure all-pairs conditional structural content $S(\mathcal{T}_j \mid m_i)$ at a fixed, reported compute budget.

- **Diagonal** — intrinsic structural content. Near-zero rows are junk; drop them.
- **Block structure** — the capability decomposition. Families that substitute for each other cluster. Blocks are what you thought were separate capabilities and are not. *Note that this is a Q-matrix (§2.1) estimated by transfer instead of by factor analysis, on inducers instead of on students — the interpretive pitfalls of that literature apply.*
- **Antisymmetric part** — prerequisite ordering. $S(\mathcal{T}_j \mid m_i) \ll S(\mathcal{T}_j)$ while the reverse holds weakly means $i$ is a prerequisite that pays. Threshold into a DAG.
- **Isolated columns** — families nothing else helps with. Either genuinely primitive, or genuinely broken. Inspect individually; do not assume the flattering reading.

**Both structures are read off, not designed.** This is the one place where the literature must not be allowed to speak: the veins arrive with their own capability taxonomies attached, and those taxonomies were built for other purposes. Write down the decomposition you expect — including the one your sources would predict — before running the matrix, and compare afterward. **A mismatch between the field's inherited taxonomy and the measured blocks is the single most valuable result this process can produce**, and it is unavailable if the prediction is made after the fact.

**On sizing:** the matrix is $n^2$ tiny runs. At diagnostic scale — two layers, four heads, hidden size 16 — a hundred runs of minutes each is an afternoon. This is what licenses over-generation: the cost of a wrong candidate is one row, and the cost of a missing candidate is unbounded.

---

## 7. Iteration

The matrix is not a one-shot measurement. It is the loop's sensor, and the review is the loop's supply.

```
read a vein  →  register + translate  →  candidates  →  measure matrix  →  read decomposition
      ↑                                                                            ↓
  target the gap  ←──────  diagnose failure  ←──────────────────  downstream eval
```

**Repair, per failure.** A downstream task the model fails is not a verdict. Ask: which primitive would have made the solver program short? That names a missing family — and then ask which vein is likely to already contain it. Both questions are per-instance and have answers, as opposed to "does the basis saturate," which is a verdict with no next action.

**Saturation curve — the review's stopping rule.** Track distinct primitives against sources processed, systematic-review style. Still climbing linearly at source 300 means the basis is unbounded and the compactness thesis is in trouble. Flattening at 40–80 means the basis is in hand and further reading is redundant. This is the cheapest high-information measurement available, it needs no compute, and it is what tells you when to stop reading.

**Drift.** The basis is a fact about practice, not about mathematics. Re-run extraction on tasks and traces from five years ago and measure how far it moves. Small movement means this is a one-time exercise; large movement means it is a standing instrument and the repertoire needs a maintenance plan. Either answer is worth having, and it is cheap.

---

## 8. Held-out practice

Extracting primitives from a corpus and evaluating on that corpus is circular. The ARC-AGI-3 designers name synthetic overfitting as a primary design hazard, and note that carefully constructed datasets become susceptible as training expands to target them.

**Partition before extraction, not after.** Nominate domains that will not be inspected during §2 — different mathematical subfield, different language ecosystem, different tool stack. Nobody looks at them until evaluation. **The partition covers the literature too:** name veins, or sub-areas within veins, that stay unread until the basis is fixed.

The stronger version, if affordable: extract the basis from domain set $A$ only, then check whether solver programs for domain set $B$ come out short under it. If they do not, the basis is fitted to $A$ and generalization was never demonstrated.

---

## 9. Failure modes specific to this section

**Importing the source's taxonomy along with its tasks.** Every vein arrives with a theory of what its tasks measure — cognitive operations, query complexity classes, planning subgoals. Take the apparatus; leave the ontology. §6 is void if the blocks were named in advance by somebody else.

**Reading instead of building.** A literature review with no termination condition is an indefinite delay dressed as diligence. §7's saturation curve is the stopping rule; §10 caps the reading at weeks, not months, and step 6 gates everything after it regardless of how complete the register feels.

**Transplant rejection.** Human-calibrated difficulty, worst-case query complexity, and asymptotic hardness are all *not* the quantity we optimize. A family can be well-attested, decades-validated, and still measure near-zero for us. Expect a substantial import mortality rate and do not treat it as evidence the vein was wrong.

**Building for the eval you have.** The repertoire should target what a corpus teaches badly — recovery, calibration, task inference under underdetermination. But every convenient benchmark measures L0-ish competence, so measured progress will lag the thing being built. Expect the dashboard to disagree with the design for a long time, and decide in advance what would make you abandon the design rather than the dashboard.

**Elegance capture.** A basis derived from clean formal structure will be defensible, compact, and fitted to your taste. Coverage of established practice (§2.3, §2.1) and held-out domains (§8) are the two checks; neither is optional.

**Brute-force collapse.** Families satisfying every other constraint lose their structural content once the learner can afford to memorize or simulate the generator. A synthetic generator is short by construction, so it has none of natural data's protection. Watch for a turnover in structural content as compute grows on any family; treat it as a signal to raise composition depth, not as noise.

**L2 attrition.** A7 requires a computable teacher query policy. Some families will not admit one, and they will disproportionately be the interesting ones. §2.2 lets much of this be settled by reading rather than by building — do it early, because discovering it after the basis is fixed means redoing the register.

---

## 10. What good looks like

Not a number. Four conditions on D3, and a fifth that is not ours.

**On the basis — these judge this document:**

1. **The instrument works** — planted duplicates cluster, junk reads near-zero, known prerequisites come out with the right sign. *(Instrument owned by Task Spec §8 step 4; the plants are ours.)*
2. **The basis is compact** — primitive count flattens, and the decomposition has fewer blocks than you had candidate families.
3. **It covers established practice and held-out domains** — the paradigms independent communities converged on fall out as special cases, and solver programs for uninspected domains are short under it.
4. **It reaches above L0** — a substantial share of families are L1–L3. *Whether they survive the dial sweep is settled by Task Spec §8 step 5, not here; if the sweep collapses, this document's output is still valid for L0 and the programme is smaller than hoped.*

**On the programme — this judges the thesis, and a basis is only one of its inputs:**

5. **It beats a stupid baseline** — the minimum viable generator over the basis exceeds both a text corpus and a single cellular automaton rule on structural bits per token. Rule 54 is the right sanity check precisely because it is stupid: no composition, no binding, no retrieval, designed for nothing.

Condition 4 is the one that decides whether this was worth doing. Condition 5 is the go/no-go for everything, which is why it is stated here — but see §0.1: it is run downstream, it is confounded by the generator and the harness, and this document's method cannot be credited or blamed for its outcome on its own.

---

## 11. Order of work

| # | Step | Cost | Gate |
|---|---|---|---|
| 1 | Partition held-out domains **and held-out veins** | hours | Recorded before anything is inspected |
| 2 | Register veins §2.1–§2.5; translate into §7 interface fields; A1–A7 triage | weeks | Primitive count flattens (§7) |
| 3 | §2.6 practice traces, annotated under an existing coding scheme | person-weeks | Inter-annotator agreement holds |
| 4 | Paradigm coverage check — do §2.3's paradigms fall out as special cases? | days | They do, or the parametrization is missing something |
| 5 | Candidate set, ~30 families incl. plants | weeks | Spans all four axes of §4 |
| 6 | **Matrix, planted-basis validation first** | afternoon | Predicted decomposition recorded first; plants recover. **Else stop** |
| 7 | Read decomposition; prune; repair | days | Blocks are interpretable → **D3** |
