# Vein 2.6 — Practice Traces / Control Structure

Scope reminder: this vein extracts what a human did that the finished solution does not need — backtracks, guesses, noticed ambiguity, declared uncertainty — not further solver operations. Clinical reasoning / differential-diagnosis literature is HELD OUT and was not fetched (see §10).

---

## 1. Trace-thinning schedule — empirical grounding (PRIORITY)

**Sources**: Sweller, Ayres & Kalyuga, "The Guidance Fading Effect," ch.13 in *Cognitive Load Theory* (Springer, 2011), read in full body text pp.171–182 [VERIFIED]. Kalyuga, Ayres, Chandler & Sweller, "The Expertise Reversal Effect," *Educational Psychologist* 38(1):23–31 (2003), read in full [VERIFIED].

### 1.1 The core empirical shape

| Claim | Evidence | Status |
|---|---|---|
| Worked examples beat unguided problem solving for **novices** | Multiple studies cited in Ch.13 (Cooper & Sweller 1987; Sweller & Cooper 1985; Trafton & Reiser 1993) and the full worked-example section of the 2003 paper (p.26) | VERIFIED (body text) |
| The benefit **reverses** with rising learner expertise — this is the "expertise reversal effect," the paper's central named finding | Kalyuga, Chandler, Tuovinen & Sweller (2001) mechanical-trade apprentices: worked-examples group beat problem-solving group on lower mental-load ratings while inexperienced; "with more experience in the domain, the superiority of worked examples disappeared. Eventually, with sufficient experience, additional learning was facilitated more by problem solving" (2003 paper, p.27) | VERIFIED, quoted, p.27 |
| Abrupt worked-example→problem switch is inferior to **gradual fading** | Renkl, Atkinson, Maier & Staley (2002): "a fading out procedure was superior to an abrupt switch from worked examples to problems" (2003 paper, p.27) | VERIFIED, quoted, p.27 |
| Fading only helps when intrinsic load is high to begin with | Ch.13 "Conditions of Applicability," p.181: "high levels of intrinsic cognitive load are essential to the fading effect... It is unreasonable to expect any significant effect of fading for instructional materials that do not impose a sufficiently high level of intrinsic cognitive load" | VERIFIED, quoted, p.181 |

### 1.2 Two fading procedures, concretely (Ch.13, p.173)

Both start with a fully worked example. They differ in **which end** they strip steps from:

- **Backward fading**: task 2 = worked example minus the *last* step (learner supplies it); task 3 = minus the last *two* steps; etc. Example given in the source (algebra, `(a+b)/c = d`, solve for `a`): task 2 shows `(a+b)/c=d`, `a+b=dc`, then `a=?`.
- **Forward fading**: task 2 = worked example minus the *first* step; task 3 = minus the first two; etc. Same example: task 2 shows `(a+b)/c=d`, `a+b=?`, `a=dc-b`.

**Backward fading generally wins.** Renkl, Atkinson, Maier & Staley (2002), three experiments (physics electricity lessons, then two lab replications with probability calculation): "the backward-fading condition also was superior on a far transfer post-test... The backward-fading condition was generally more efficient than forward fading, as learners presented backward fading also required less time to study the examples. From a cognitive load perspective, the backward-fading condition whereby the learner supplies the final problem-solving step may impose a lower cognitive load than the forward-fading condition where the learner supplies the first problem-solving step, a step often a critical step in the overall solution" (p.174, VERIFIED, quoted).

Caveat, same page: Renkl, Atkinson & Große (2004) found the *position* of the faded step did not actually change how much was learned about that specific step — students learned the faded step about equally well regardless of backward/forward direction. What fading (either direction) reliably did, per a think-aloud follow-up, was reduce "unproductive learning events." Net reading: **direction of fade is a second-order efficiency knob (favor backward), not the primary mechanism** — the primary mechanism is fading itself vs. abrupt cutover, plus keeping learners actively explaining the faded step rather than silently skipping it (Renkl & Atkinson 2001: self-explanation prompts at faded steps produced "strong advantages... on both near and far transfer post-tests," p.174, VERIFIED).

### 1.3 A concrete adaptive schedule (not just "fade linearly")

Kalyuga & Sweller (2004), reproduced as Fig. 13.1 in the source (p.180, VERIFIED — figure read directly):

1. **Initial diagnostic test** (a handful of rapid "what's your first step" probes) sorts the learner into a stage.
2. **Stage 1**: two fully worked examples each followed by one practice problem; a diagnostic test gates advancement (fail → more worked examples at this stage).
3. **Stage 2**: worked examples with the *last* step to complete (one-step backward fade); gated by diagnostic test.
4. **Stage 3**: worked examples with the *last two* steps to complete; gated by diagnostic test.
5. **Stage 4**: problem-solving exercises only, no worked steps.
6. **Final diagnostic test.**

This "learner-adapted tutor" beat a fixed non-adapted tutor on both knowledge gain and efficiency (p.179–180). A further refinement, Salden, Aleven, Schwonke & Renkl (2010): **individualized adaptive fading** (rate of fade tied to each learner's own self-explanation/problem-solving performance) beat both **fixed-schedule fading** and **problem-solving-only**, on immediate *and* one-week-delayed post-tests (p.176–177, VERIFIED).

The gating signal used for "is this learner ready to have a step removed" is cheap: a **first-step diagnostic** — show a problem, ask only for the first move, for a limited time. Kalyuga & Sweller (2004) validated this against full worked-solution scores at high correlation, at up to 5× less testing time (p.178, VERIFIED). This is a genuinely actionable, low-cost proxy for "is more guidance still earning its keep."

### 1.4 Does the fade direction/rate itself depend on the learner, not just the material?

Yes — Reisslein (2005), university engineering students, three transition speeds (immediate / fast fade / slow fade): "significant interactions between levels of learner prior knowledge and the pace of transitioning. More knowledgeable learners performed significantly better in the fast and immediate transitioning groups... learners with low levels of prior knowledge... benefited more from the slow transitioning condition" (p.175, VERIFIED, quoted). So the fade **rate** is itself expertise-conditional, not a fixed curve.

### 1.5 Verdict on our spec's default ("emit traces at low k, thin them as k rises")

**The literature supports fading guidance as a function of *demonstrated learner competence*, not as a static function of a task-difficulty knob. These are not obviously the same axis, and the difference matters.** This is the central finding of this deliverable, and it complicates the spec's default rather than simply confirming it:

- Every fading study in this literature varies **learner expertise over training/practice time** (pre/post-tests, staged tutors, multi-session apprentice training) while holding the **material's difficulty roughly fixed within a stage**. None of them fade guidance as a function of *item difficulty* at a single fixed point in a learner's development.
- The literature's own gating condition (§1.1, last row) says fading only has teeth when **intrinsic cognitive load is high** — i.e., harder material is exactly where worked guidance earns the most, not where it should be thinnest. Sweller's own isolated-elements-effect material (2011 vol., not separately re-verified here but consistent with the Ch.13 argument) makes the same point: high-element-interactivity (hard) content is where novices most need support.
- Combining these: if our `k` indexes **instance difficulty** rather than **the model's trained competence at that difficulty band**, a literal "thin as k rises" schedule risks being backwards on hard instances early in training — precisely where full worked traces are earning the most, per this literature.

**Recommended repair, grounded in the same sources**: two-axis schedule, not one.
1. Density of worked trace should track **the model's own demonstrated competence at a given k-band** (an adaptive, diagnostic-gated schedule, Kalyuga & Sweller 2004 style) — measured cheaply, analogous to the "first-step" probe, e.g. by checking accuracy on validation queries at that k before thinning further.
2. Within a fixed competence level, **higher k should if anything start denser and fade more slowly** (Reisslein 2005), not start thinner.
3. Independent of both: fade gradually, not abruptly, and **prefer backward fading** when a direction must be chosen, and **keep a cheap justification token at faded steps** rather than silently deleting them (self-explanation-prompt analog).

If the project wants to keep a single static `k`-indexed schedule for implementation simplicity, the literature says the honest version is *k co-varies with the density needed*, not that low-k needs less. The safer single-axis proxy, if only one axis is affordable, is **training step / curriculum position**, with `k` left as a secondary modifier on fade *rate* rather than the primary driver of fade *amount*.

---

## 2. Coding scheme — categories and reliability

**Target scheme**: Schoenfeld's episode theory (*Mathematical Problem Solving*, Academic Press, 1985).

| Category | Definition (as stated in secondary applications) |
|---|---|
| Read | Extracting/restating given information and problem goals |
| Analyze | Constructing or recalling relevant theory, deducing relationships |
| Explore | Generating potential ideas, tentative attempts |
| Plan | Announcing next step / outlining a strategy |
| Implement | Executing the planned strategy through calculation |
| Verify | Reviewing/testing the solution for correctness |
| (later addition) Monitor | Self-monitoring / hesitation, added by later applications as a transition category — **not** one of the original six |

Status: **LEAD, not VERIFIED against the primary 1985 text.** I could not obtain Schoenfeld's book directly this session (no accessible full text found; the closest attempt, a Springer/ZDM paper on a "descriptive phase model," was behind an institutional login wall and not fetched). The six-category list above is convergent across two independent secondary sources I did fetch and read: arXiv:2509.14662 ("Understanding the Thinking Process of Reasoning Models: A Perspective from Schoenfeld's Episode Theory," HTML fetched directly) and a WebSearch-summarized pass at arXiv:2512.19995 ("Schoenfeld's Anatomy of Mathematical Reasoning by Language Models"). Both cite the same six categories and both describe the source material the same way: **"built on hundreds of hours of recorded tape of students tackling non-routine math problems while being asked to think aloud"** — I could not independently verify the hours figure, exact number of subjects, or exact problem set from primary text.

**Reliability figure: not located.** Neither fetched secondary source cites a Cohen's-kappa or percent-agreement figure from Schoenfeld's own 1985 validation. One search result (a ZDM paper, "A descriptive phase model of problem-solving processes," abstract only, not fetched — paywalled) reportedly notes "initial difficulties in coding the deductive episodes reliably, especially differentiating between Analysis and Exploration episode types, which was predicted by Schoenfeld (1992)" — this is a qualitative acknowledgment of a coding difficulty between two specific categories, not a number, and I am reporting it at LEAD confidence (search-snippet level, not fetched). **This is a gate we cannot currently clear**: the task instructions treat a reported inter-annotator agreement figure as a gate on our own coding effort, and I did not find one. Recommendation: if this scheme is adopted, run our own pilot inter-annotator pass and report our own kappa rather than importing an unverified number.

**Material it was validated on**: per both secondary sources, hundreds of hours of think-aloud protocols of students solving non-routine math problems (population size, exact hour count, and exact problem set not located).

---

## 3. Control-structure primitives

### 3.1 Reused (existing primitives that already cover part of what a trace needs)

| Existing primitive | What it covers here |
|---|---|
| `belief-state-reset` | Non-monotonic replacement of a single converged hypothesis on contradiction — covers Lakatos's *monster-adjustment* (reinterpret the counterexample) and *lemma-incorporation* (fold the exception into a revised, still-single, hypothesis) |
| `hypothesis-elimination` | Dropping a specific candidate rule from a candidate *set* upon refuting evidence — covers Lakatos's *surrender* (reject the conjecture outright) and *exception-barring* (narrow the domain, i.e. eliminate the offending instances/subdomain) |
| `belief-state-maintenance` | Ongoing multi-hypothesis tracking during L1/L3 tasks |
| `posterior-enumeration` | Explicit listing of remaining consistent hypotheses |
| `majority-predict` | The computational content underneath a "guess" (see below) when the guess is principled rather than blind |
| `structure-walk-query`, `constraint-propagation` | Forward derivation/search steps that a `verify` or `backtrack` move interrupts |

### 3.2 New

**`backtrack-splice`** — abandon an in-progress derivation/search branch after committing to it for one or more steps, and resume from the last verified-good state. Argued as genuinely new because it operates over **action/operator sequences within a single derivation**, not over hypotheses-about-theta — none of the existing 12 primitives has a "discard this branch, backjump" companion to the forward-search primitives (`structure-walk-query`, `constraint-propagation`). This is the direct analog of the Problem Behavior Graph's revisit move (§5) and of the constructive method used in the "Self-Backtracking" paper (§4, §6): take a correct prefix, splice in a plausible-but-wrong continuation for a bounded number of steps from a small structural confusion set (off-by-one, sign flip, wrong-constraint-checked — defined structurally, not by content, to stay knowledge-free/A2), mark the point of divergence, then resume the true path.

**`epistemic-status-tagging`** — an explicit, inline verbal/token-level annotation of a step's status (`derived` / `guessed` / `uncertain-among-{S}` / `assumption-unverified` / `evidence-rejected-as-malformed`), decoupled from the computation that produced the step. This single mechanism is proposed to cover **three** items from the target brainstorm list at once: *notice-ambiguity* (emit `uncertain-among-{S}` before falling back to enumeration), *declare-uncertainty* (same tag family), and *guess* (emit `guessed` when committing without derivation — the selection itself still reuses `majority-predict` or a raw prior sample; what's new is only the label). Argued as one primitive rather than three because in every case the *computation* is already covered by an existing primitive or is trivial (sample from prior); the only new content is the trace-level speech-act layered on top, which is exactly the "control structure, not a further operation" framing this vein is chartered to find.

### 3.3 Rejected folds (argued)

| Candidate | Why folded, not new |
|---|---|
| `verify` (Schoenfeld's Verify episode; explicit "double-check the answer" move) | Reduces to re-invoking the oracle/constraint-check machinery already implicit in `constraint-propagation`, gated by a control-flow decision to spend a step doing so. The *decision* to check is adequately represented as an `epistemic-status-tagging` value (`checked`) rather than a new operation. |
| Lakatos's *monster-barring* (reject the counterexample's legitimacy rather than the hypothesis) | This is not hypothesis management at all — it's a claim that the **evidence itself** is out-of-domain/invalid. Folded into `epistemic-status-tagging` as the `evidence-rejected-as-malformed` label, which also gives it a natural home against admissibility criterion A6 (L2's requirement for "sensible response to malformed queries") rather than inventing a separate primitive for it. |
| Lakatos's *exception-barring* | Folded into `hypothesis-elimination` (narrowing which instances the hypothesis is asserted to cover is the same computational move as eliminating a subset of candidate extensions). |

### 3.4 Running tally

Previous five veins: +3, 0, +1, +1, 0 → 12 total. This vein: **+2** (backtrack-splice, epistemic-status-tagging) → **14 total**. This is a real but modest yield, consistent with the instruction not to force it: most of what this vein found is not new *computation* but a new *layer* (explicit labeling/verbalization) wrapped around computations the roster already has — which is itself the expected shape for a vein whose job is "what the program did not need," not "what the program does."

---

## 4. Candidate families

### 4.1 Family: Backtrack-recovery ("spliced-error" construction)

- **Theta**: same as any existing backward-generable base family (e.g., a hidden rule over a finite symbol space).
- **P_Theta(·\|k)**: as base family; `k` additionally controls the size/subtlety of the confusion set used below.
- **Generation (backward, O(1))**: sample theta and its correct solution/derivation trace `S = (s_0 … s_n)` from the base family (already O(1) by that family's own A1 pass). Pick a splice point `i`. From a small, *structurally* defined confusion set attached to the true operator at step `i` (e.g. "swap the two operands," "off-by-one index," "apply the operator that matches the surface form but not the constraint") sample a wrong operator `o'`, apply it for `j ≤ J_max` steps to produce a false continuation, insert a distinguished `⟨backtrack⟩` marker, discard the false branch, and resume `s_i → s_{i+1} → … → s_n`.
- **Query x / oracle**: unchanged from the base family for the final-answer task; for an L2 variant, the model may additionally be asked to name the divergence point, graded against the logged splice location.
- **A1 Backward-generable**: **PASS, with an explicit caveat.** This is the central methodological question this vein was asked to resolve, and the literature gives a clean, split answer (§6 has the supporting citations): *genuine* search-simulated backtracking (Stream of Search; Search-Fail-Recover) is **not** A1-compliant — it requires running an actual solver/search process on the generation path, by construction. But the "Self-Backtracking" paper's construction — take a known-correct path, splice a synthetic wrong branch from a small error model, tag, resume — **is** A1-compliant: it is O(1) given theta and its (already-generable) correct path, plus a fixed, cheap corruption model. The honest tradeoff: this buys admissibility at the cost of only ever producing *manufactured* backtracks (a controlled distractor family), not naturalistic backtracks that would emerge from a real searcher's actual uncertainty. Whether that gap matters is an empirical question for downstream evaluation, not a generation-time one.
- **A2 Knowledge-free**: PASS if the confusion set is defined structurally (index/argument/operator perturbations) rather than by symbol content, and if the base family already passes A2.
- **A3 Encoding-varied**: PASS/inherits from base family; the backtrack marker itself should get ≥2 renderings (a structured token vs. a natural-language "wait, that's wrong because…" span).
- **A4 Brute-force-resistant**: **repairable.** Adds nothing by itself unless the *query* requires identifying where/why the error occurred; tie confusion-set size/subtlety to `k` so that distinguishing the true step from the near-miss genuinely requires modeling the rule, not pattern-matching a shortcut.
- **A5 Semantically coherent**: PASS by construction — the false branch must be a valid application of *some* rule (just the wrong one), not noise.
- **A6/A7 (L2)**: PASS if the divergence log is carried in `E`; the teacher policy for "ask about the suspicious step" is one-pass given theta + the splice log.

### 4.2 Family: Ambiguity / epistemic-status

- **Theta**: a base family deliberately run at an `k`/history point where two hypotheses `theta_1, theta_2` are both consistent with trials-so-far (this is a controlled version of what L1/L3 already does; the addition is *forcing* the trace to contain an explicit noticing act before falling back to enumeration).
- **Generation**: O(1) — pick `theta_1, theta_2` from a parametrized confusable-pair family (same shape as `hypothesis-elimination`'s existing setup) and require the emitted trace to include an `epistemic-status-tagging` span (`uncertain-among-{theta_1, theta_2}`) before the posterior-enumeration step.
- **A1–A7**: inherits its base family's triage essentially unchanged; the only addition is a required trace span, which is free at generation time since theta and the ambiguity are both already known to the generator.
- **Verdict**: **pass**, but this family is a thin methodological wrapper more than a new task shape — its main value is training the *habit* of flagging ambiguity explicitly rather than only being gradable on the final posterior, and it should probably be folded into an annotation convention on top of existing L1/L3 families rather than shipped as a standalone family.

---

## 5. Problem Behavior Graph notation

**Status: LEAD, low-to-medium confidence.** I was not able to independently open Newell & Simon's *Human Problem Solving* (1972) or a clean secondary reproduction of a PBG figure this session — Google Books, archive.org, and a specialized glossary page (University of Alberta *Dictionary of Cognitive Science*, "Problem Behavior Graph" entry) were all unreachable (archive.org fetches are blocked in this environment; the Alberta page returned a connection error; a VanLehn chapter PDF that reliably reproduces a PBG figure returned 403). What follows is reconstructed from convergent but shallow secondary description and should be verified against the primary text before being used as a literal data structure spec.

| Element | Description (convergent across secondary sources) |
|---|---|
| Node | A problem state — the subject's current knowledge configuration (e.g., in the DONALD+GERALD=ROBERT cryptarithmetic task Newell & Simon used as their worked example, which letter-digit assignments are currently committed) |
| Edge | An operator application that transforms one state into the next; labeled with the operator |
| Backtrack | Per one (unverified) secondary source: when the subject returns to an earlier state, that state is **redrawn as a new node** (a duplicate of the earlier occurrence) rather than reusing the original node position, typically placed so the duplicate visually sits below/after the earlier occurrence — i.e., the graph is time-ordered (unfolds forward), and a "return" is a *revisit shown as a new time-stamped copy*, not a literal edge back to the old node. I was **not able to independently confirm this specific claim** against primary text or a second independent source this session. |

The general framing that *is* well-supported (via Newell, Shaw & Simon's 1958 companion paper, read directly, §5 below, and general secondary consensus): the graph is a record of a **specific solver's actual search trajectory through a problem space** (states + operators), not the abstract problem space itself — i.e., it is a *trace* data structure, which is exactly the object this vein wants a template for. The primary source for the notation itself is Newell & Simon (1972), *Human Problem Solving*, Ch. 3, and the original worked example is the cryptarithmetic protocol reported in Newell's 1966 CMU technical report "Studies in Problem Solving: Subject 3 on the Crypt-Arithmetic Task DONALD + GERALD = ROBERT" (citation located via DTIC/CMU KiltHub; PDF not opened — 403).

**Recommendation**: before building anything on the "duplicate node" backtrack convention specifically, get institutional/library access to *Human Problem Solving* Ch. 3 directly rather than relying on the above.

---

## 6. Available corpora

| Corpus / method | Form | What it gives us | Confidence |
|---|---|---|---|
| Stream of Search (Gandhi et al., arXiv:2404.03683) | Countdown-game search traces generated by running real heuristic solvers (BFS/DFS-style), including genuine dead ends and backtracks, flattened to a token stream ("streams of search") | Existence proof that **forward, solver-simulated** backtrack corpora are buildable and improve search accuracy; explicitly **not** A1-compliant (requires running a solver on the generation path) | LEAD — abstract fetched directly; specific numbers (dataset size, accuracy deltas) came via search-engine synthesis, not independently confirmed against body text |
| Self-Backtracking (arXiv:2502.04404) | Countdown-game training data built by taking a correct solution prefix and **synthetically splicing** one of three constructed error types (DFS-mismatch "exploration error," inserted-invalid-equation "computational error," disallowed-operand "rule violation"), tagged with a `⟨backtrack⟩` token, then resuming the true path | The direct methodological precedent for §4.1's `backtrack-splice` family; confirms an **A1-compliant** construction exists | LEAD — HTML fetched, methodology quoted directly from the page, but not visually confirmed via a raw-PDF read |
| Search, Fail, Recover (arXiv:2607.07492) | "Diligent Learner" formulation: real search trees from actual generated continuations, validated by a task checker, converted into continue/finish/backtrack supervision with abandoned-branch summaries | Another forward-search-simulated corpus (not A1-compliant), useful as a contrast case | LEAD — abstract-level fetch |
| PRM800K / "Let's Verify Step by Step" (arXiv:2305.20050) | ~800K human step-level correctness labels (positive/negative/neutral) on GPT-4-generated MATH solutions; phase-1 data includes human-written corrections at flagged steps | Directly usable as a *verify* / *epistemic-status* supervision source — humans marking exactly where a derivation goes wrong and how it was fixed | LEAD — description via search synthesis, not independently fetched |
| LeanDojo / LEAN-GitHub (arXiv:2407.17227 and related) | Tactic-level proof states mined from real Lean/mathlib repositories, including failed-tactic states | Real (not synthetic) formal proof-attempt corpus; the raw material for a genuine "proof attempts, not proofs" family | LEAD — titles/abstracts via search, not fetched |
| "Learning to Repair Lean Proofs from Compiler Feedback" (arXiv:2602.02990) | Proof-attempt + compiler-error + repair triples | A direct, already-existing "debugging session" corpus in the target's own sense (attempt → error → fix), in a formal-proof setting | LEAD — title/abstract only |
| Chess annotation symbols (`??` blunder, `?` mistake, `?!` dubious, `!?` interesting, `!` good, `!!` brilliant) | A century-old, standardized, human-expert error/quality taxonomy applied move-by-move to annotated games (e.g. Lichess games with Stockfish evaluations used to build blunder-classification training sets) | A ready-made, cheaply-scalable label vocabulary for "this step was a mistake / this step was a strong move," directly reusable as an `epistemic-status-tagging` label set; explicitly in-scope per the prohibition list (protocol analysis applied to chess is IN SCOPE) | LEAD — standard/common knowledge, secondary source only (Wikipedia) |
| Newell's own coded cryptarithmetic protocols (CMU technical reports, 1958–66) | Small-scale, hand-coded Problem Behavior Graphs from real think-aloud sessions | The historical existence proof that human PBG-coded traces are producible at all; not usable at scale | LEAD — citation located, PDF not opened (403) |

---

## 7. Rejections (with repairs)

| Item | Verdict | Repair |
|---|---|---|
| Naturalistic (non-spliced) backtracking as a backward-generable family | **fail on A1 as stated** | Use the spliced-error construction (§4.1) as the admissible substitute; treat genuine solver-simulated backtracking (Stream of Search-style) as a separate, explicitly non-A1, off-the-generation-path data source if ever wanted, not as part of the backward-generable curriculum |
| `verify` as a standalone new primitive | **fail — redundant** | Fold into `constraint-propagation` + `epistemic-status-tagging("checked")`, per §3.3 |
| Lakatos's five named repair moves as five separate new primitives | **fail — over-fragmented** | Fold four of five into `belief-state-reset` / `hypothesis-elimination` per §3.3; keep only the genuinely uncovered move (monster-barring, i.e., rejecting evidence itself) as a label extension of `epistemic-status-tagging`, not a primitive |
| A single static `k`-indexed linear thinning schedule ("thin as k rises") taken literally | **repairable, not a clean pass** | Two-axis schedule per §1.5: fade primarily against demonstrated competence at a k-band (diagnostic-gated), with `k` modulating starting density and fade rate rather than driving thinning directly |
| Schoenfeld coding scheme adopted with an *imported* reliability figure | **fail — no figure to import** | Run our own pilot inter-annotator pass and report our own agreement statistic; do not cite an unverified number (see §2) |

---

## 8. Sources

### VERIFIED (fetched and read directly; page/section noted)

1. Ericsson, K.A. & Simon, H.A. (1980). "Verbal Reports as Data." *Psychological Review* 87(3):215–251. Read pp.215–224 (abstract, Fig.1, Table 1, Level 1/2/3 definitions, directed-probe comparison, retrospective-degradation discussion). [Sections 1 note this feeds Hazard 1, not restated at length in a numbered deliverable above but load-bearing for §1's caveats and referenced directly below in §9.]
2. Sweller, J., Ayres, P., & Kalyuga, S., "The Guidance Fading Effect," ch.13 in *Cognitive Load Theory* (Springer, 2011). Read pp.171–182, full body text including Fig.13.1.
3. Kalyuga, S., Ayres, P., Chandler, P., & Sweller, J. (2003). "The Expertise Reversal Effect." *Educational Psychologist* 38(1):23–31. Read in full, body + references.
4. Clark, R.E., Feldon, D., van Merriënboer, J.J.G., Yates, K., & Early, S. (2007). "Cognitive Task Analysis." In *Handbook of Research on Educational Communications and Technology* (3rd ed.). Read pp.577–588, full body text, including the Chao & Salvendy (1994) numbers on p.585.
5. Newell, A., Shaw, J.C., & Simon, H.A. (1958). "Elements of a Theory of Human Problem Solving." *Psychological Review* 65(3):151–166. Read pp.151–156, body text. Establishes the general information-processing framing; does **not** contain the Problem Behavior Graph notation (that is specific to the 1972 book).
6. Lakatos, I. (1963). "Proofs and Refutations (I)." *British Journal for the Philosophy of Science* 14(53):1–25. Read pp.1–5, body text — full table of contents (the named dialectical moves used in §3) plus introduction.
7. arXiv:2509.14662, "Understanding the Thinking Process of Reasoning Models: A Perspective from Schoenfeld's Episode Theory." Fetched HTML directly, read body text for category list and reliability discussion (§2).

### LEAD (found via search, or fetched only at abstract/summary level, or corroborated only by secondary sources)

8. Chao, C.-J. & Salvendy, G. (1994), *International Journal of Human-Computer Interaction* 6(3):221–233 — known only via source 4's citation of it; the 41%/53%/29% single-expert figures and 87%/88%/62% aggregated figures are one hop from primary (VERIFIED that source 4 states them; LEAD that the primary paper itself states them identically).
9. Schoenfeld, A.H. (1985). *Mathematical Problem Solving.* Not independently fetched; category list cross-corroborated by sources 7 and a WebSearch-summarized pass at arXiv:2512.19995. **No reliability figure located and this is flagged explicitly in §2, not glossed over.**
10. Newell, A. & Simon, H.A. (1972). *Human Problem Solving.* PBG notation reconstructed from shallow, partly-unverifiable secondary description (§5) — flagged as the weakest-confidence item in this report.
11. Pólya, G. (1945). *How to Solve It.* Four-step scheme corroborated by multiple consistent tertiary sources (Study.com, LibreTexts); primary text not read this session.
12. Gandhi, K. et al. "Stream of Search," arXiv:2404.03683 — abstract fetched directly; supporting numbers via search synthesis only.
13. "Search, Fail, Recover," arXiv:2607.07492 — abstract-level only.
14. "Step Back to Leap Forward: Self-Backtracking," arXiv:2502.04404 — HTML fetched, methodology quoted, not independently confirmed via raw read.
15. Lightman et al., "Let's Verify Step by Step" / PRM800K, arXiv:2305.20050 — description via search synthesis only.
16. LeanDojo / LEAN-GitHub / "Learning to Repair Lean Proofs from Compiler Feedback" — titles and abstracts via search only, not fetched.
17. Wikipedia, "Chess annotation symbols" — standard/common-knowledge secondary source.

### Could not reach (attempted, blocked)

- MIT Press book page for *Protocol Analysis* — HTTP 403.
- CMU KiltHub page for Newell's cryptarithmetic technical report — HTTP 403.
- scispace-hosted VanLehn, "Problem Solving and Cognitive Skill Acquisition" — HTTP 403.
- University of Alberta *Dictionary of Cognitive Science*, "Problem Behavior Graph" entry — connection refused (site appears down).
- web.archive.org — blocked entirely in this environment (tool-level restriction).
- Springer/ZDM, "A descriptive phase model of problem-solving processes" — redirected to an institutional login wall, not fetched.

---

## 9. Source's taxonomy (quarantined)

Per Hazard 2: the following are frameworks from the fetched sources that predict *how their own material groups*, and are used above only as an annotation/coding convenience or as raw material to mine for control-structure moves — not imported as claims about our capability decomposition, which we measure empirically.

- **Schoenfeld's six/seven episodes** (Read/Analyze/Explore/Plan/Implement/Verify[/Monitor]) is a **stage theory of human mathematical problem solving**. We use it in §2 only because the task explicitly asked for "a coding scheme we could actually use" — i.e., as a labeling convention for supervising trace segmentation — not as evidence that these are the natural joints of model capability.
- **Ericsson & Simon's Level 1/2/3 verbalization taxonomy** and their concurrent/retrospective distinction are a theory of **when a verbal report is trustworthy**, not a theory of problem-solving stages — this one is closer to a methodological tool we actually need (for judging which of our own generated "traces" should count as faithful vs. confabulated), so it is not quarantined in the same sense, but is flagged here as still being Ericsson & Simon's own framework, not derived from our data.
- **Newell & Simon's problem-space theory** (state/operator/goal-test) is close enough to generic search formalism, and close enough to primitives already in the roster (`structure-walk-query`, `constraint-propagation`), that it is treated as shared infrastructure rather than a competing ontology — not quarantined.
- **Lakatos's five-move proof-refutation taxonomy** (surrender / monster-barring / exception-barring / monster-adjustment / lemma-incorporation) is a philosophy-of-mathematics account of *how working mathematicians actually argue*, not a claim about model capability categories. Used in §3 purely as raw material for control-structure primitives (and mostly folded into existing primitives, per §3.3), not imported as an ontology.

---

## 10. Sealed encounters

- PubMed: "The use of cognitive task analysis to reveal the instructional limitations of experts in the teaching of procedural skills" — title and a one-line snippet (reporting expert step-omission percentages in a **surgical/clinical** procedural-teaching context) appeared in a WebSearch result. Not opened. The equivalent non-clinical figure used in §2/§3's grounding for expert step-omission instead comes from Chao & Salvendy's programmer-troubleshooting study (in scope), reported second-hand via source 4.
- Encountered-but-not-pursued, embedded as citations *inside* an in-scope source (Clark et al. 2007, source 4) rather than found via my own search: a neonatal-ICU nurse cue-recognition study (Crandall & Getchell-Reiter, 1993) and a neuropsychologist IQ-prediction study (Kareken & Williams, 1994). I did not fetch either directly and did not build any figure in this report on them, out of caution given their clinical/medical setting, even though the chapter citing them (source 4) is itself in-scope (educational technology, not clinical reasoning).
