# Vein 2.2 — Query Learning, Teaching Dimension, and Optimal Experimental Design

Literature review for L2 (`q*(theta, history) -> x`, per A7). Scope: exact learning with queries (Angluin), teaching dimension (Goldman & Kearns and successors), Bayesian optimal experimental design (Lindley, MacKay). Sealed topics (pool-based applied active learning, clinical trial adaptive design, ILP, Latin squares, developmental psych, psychometrics, clinical diagnosis, ARC-AGI-3) were not opened; one near-miss is logged in §7.

---

## 1. q* tractability map

Cost column is per single call to `teacher_query`, i.e. cost of producing one `x`. `|Theta|` = size of the surviving/consistent hypothesis set at the current point in the episode (not the full unrestricted class). "Structure-walk" means the heuristic uses theta's internal structure directly and never touches a posterior over Theta at all — this is the situation-specific move available to us because, per the brief, we own theta (most of this literature does not).

| Hypothesis class | Query learnability (MQ / EQ) | Teaching dimension | One-pass heuristic | Cost per call | Verdict |
|---|---|---|---|---|---|
| Conjunctions / monomials over `{0,1}^n` (general, with negation) | MQ alone, O(n) queries: fix a positive anchor, flip one variable at a time (Angluin 1988, standard result) | VC-dim of monotone monomials = n (Procaccia et al.); general teaching-dim not independently confirmed here (see caveat below) | `basis-probe`: literal-elimination walk, no posterior needed | O(1) per query, O(n) total, independent of `\|Theta\|` | **CHEAP** |
| Parity functions over `GF(2)^n` | MQ alone, exactly n queries (standard-basis probe = Gaussian elimination) | n (each of n basis vectors is needed) | `basis-probe`: fixed non-adaptive schedule `e_1..e_n`, doesn't even depend on history | O(1) per query, O(n) total | **CHEAP** — and a notable Hazard-1 case: PAC learning parity under noise is conjectured hard (LPN), exact MQ learning is trivial |
| Monotone DNF (m terms, size ≤ r, n vars) | MQ+EQ, poly(m,n) (Angluin 1988); incomplete/noisy-oracle variant: expected O(mn²) queries even with ~half the MQs missing (Sloan/Turán-style result) | Not found as closed form; class size is combinatorially large | Lattice walk / boundary-following (Angluin's specify-to-generalize algorithm) | Roughly O(n) per query (local lattice move), not O(\|Theta\|) | **CHEAP**, and the other half of the Hazard-1 flag: monotone DNF is not known PAC-learnable from random examples at all (open since Valiant 1984) yet is poly-time exactly learnable from queries |
| k-term DNF (general, not monotone), n vars | MQ+EQ: classical poly(n, 2^k); improved to poly(n)·2^Õ(√k) (2025 result, first improvement since Blum–Rudich 1992) | Not found | Winnow-style attribute-efficient elimination in a constructed feature space | Cost grows sub-exponentially in k, polynomial in n | **CHEAP for fixed small k, EXPENSIVE as k grows** — parametrized boundary, good knob candidate |
| k-CNF / k-DNF (bounded clause/term size) | EQ alone suffices, poly(n^k) (Angluin 1988) | Not found | Clause-elimination via EQ; MQ-only heuristic not standard | poly(n^k) | **EXPENSIVE as k, n grow together; CHEAP for fixed small k** |
| Decision trees, depth ≤ d, n vars | MQ alone: randomized Õ(2^{2d}) + 2^d log n; deterministic 2^{5.83d} + 2^{2d+o(d)} log n (Bshouty et al. 2019, improving Kushilevitz–Mansour and Feldman) | Not found as closed form | `structure-walk-query`: since teacher owns the true tree, walk an untested root-to-leaf path directly | O(d) per query if teacher walks its own tree (structure-walk); O(2^{2d}) if a learner must search from scratch | **CHEAP when teacher owns theta (structure-walk); EXPENSIVE for a from-scratch learner** — sharpest illustration of the "we own the rule" advantage. Also: properly PAC-learning decision trees *with query access* is NP-hard for a solver that doesn't know theta (2023 result) — teacher-side ease and solver-side hardness coexist |
| DFA / regular languages, ≤ N states, alphabet Σ | MQ+EQ (Angluin's L*): poly in N and longest counterexample. MQ alone is provably insufficient in general (Balcázar–Díaz–Gavaldà–Watanabe; alternation lower bounds between MQ/EQ) | Not found as closed form for general DFA classes | `structure-walk-query`: teacher simulates its own automaton to synthesize an access-string + distinguishing-suffix pair (same construction L*'s observation table builds, but computed forward from known theta instead of inferred) | O(N · length) per query if teacher owns theta; enumeration over "all DFAs ≤ N states" is O(\|Theta\|) and `\|Theta\|` is roughly N^(N·\|Σ\|) — intractable fast | **CHEAP if teacher structure-walks theta; EXPENSIVE/OPEN if realized as brute posterior enumeration over unrestricted DFA space.** L2's "oracle answers only what's asked" model does not include an equivalence oracle, so L*'s own two-oracle machinery is not directly portable — see §3 |
| Halfspaces / linear threshold functions, dimension n, bounded integer weights | MQ: near-optimal label complexity "in some cases" (2020 result, not fully general); separate published *lower bounds* for MQ-complexity of halfspaces exist (Bshouty/Eiron-era result) | "Teaching Dimension of Linear Learners" (2016) treats this but as gradient-descent teaching, not MQ teaching — different setting, not directly portable | Boundary-following (query near current best-estimate separating hyperplane) | Brute enumeration over weight space is O(B^n); tractable only for small n, small integer weight bound B | **EXPENSIVE in general dimension; CHEAP only at small n and small B** — good parametrized boundary case |
| General monotone Boolean functions, n vars | Exact learning requires up to the full truth-table in worst case for the unrestricted class (number of monotone functions is doubly-exponential in n, Dedekind numbers); single-point *certification* given certificate complexity k costs O(k^8 log n) vs. Ω(k log n) info-theoretic floor (2022 result) | Not found | Certificate-based single-point queries; no known cheap full-identification heuristic for the unrestricted class | O(\|Theta\|) for full identification — intractable for n beyond small single digits unless restricted to a structured subclass (e.g. monotone DNF above) | **EXPENSIVE / OPEN for the unrestricted class; CHEAP once restricted to monotone DNF or k-term** |
| Finite Theta, no exploitable algebraic/graph structure (generic version-space setting) | N/A — this is the fallback, not a class | max{t0, log2\|Theta\|} ≤ #MQ ≤ t0·log2\|Theta\| (Hegedüs, cited via Goldman–Kearns line) | `split-score` + `majority-predict`: Halving Algorithm / generalized binary search — query that splits the surviving hypothesis set most evenly (Dasgupta 2005: greedy gets within log\|Theta\| of optimal query count) | O(\|Theta\|) per query (must score every candidate hypothesis against candidate queries) — Dasgupta states this cost explicitly as "linear in version-space size" | **CHEAP while \|Theta\| stays polynomial in k (say ≤ 10^4–10^5); EXPENSIVE once \|Theta\| is exponential in the natural parameter** — this is the direct, sourced answer to the brief's central question |

**Caveat on teaching dimension numbers.** Several PDF primary sources (Angluin 1988 full text, the certification paper, the Modern BED review) repeatedly failed to render through the fetch tool (binary/FlateDecode content came back garbled on every attempt) despite being valid, non-paywalled arXiv/Springer PDFs. Where a precise teaching-dimension formula is asserted above it is either (a) sourced from a VC-dimension result that is a documented *upper bound* on teaching dimension, not the teaching dimension itself, or (b) marked "not found." Do not treat blank cells as "zero" — they are honestly unresolved, not ruled out.

---

## 2. Candidate families

### 2.1 Monomial / conjunction identification

- **Theta**: literal assignment to each of n symbols — `{absent, positive, negated}^n` (or `{absent, present}^n` for the monotone-only variant). `|Theta| = 3^n` general, `2^n` monotone.
- **k**: n (alphabet size), and/or sparsity bound on relevant literals.
- **X**: `{0,1}^n` assignments.
- **f(theta, x)**: 1 iff x satisfies every literal in theta.
- **Encodings**: (a) raw bit-vector; (b) "switch panel" natural-language-free attribute list (`s3=ON, s7=OFF, ...`); (c) tabular row/column form. All three are alphabet-permutable — passes A2.
- **L3 posterior enumerable?**: yes for n up to the point where `3^n` (or a sparsity-restricted subset) is enumerable in the harness's compute budget — roughly n ≤ 12–15 unrestricted, much higher if the prior is sparsity-biased.
- **q***: `basis-probe` — anchor query then per-variable flip; O(n) total, O(1) per call, doesn't need posterior enumeration at all.
- **A1–A7**: A1 pass (trivial forward generation). A2 pass if symbol identities carry no semantics. A3 pass (three renderings above). **A4**: the whole rule is a lookup table of size `2^n`; the intended solver is O(n). Fails A4 at small n (n ≤ ~5, table is memorizable); passes once n ≥ ~8–10, where `2^n` clearly outgrows the O(n) structural solver. A5 pass (shares `basis-probe`/`posterior-update` with parity, below). A6 pass (evaluate malformed x by treating out-of-range bits as invalid, respond with a fixed error token). A7 pass — this is one of the cheapest q* in the whole map.

### 2.2 Parity identification

- **Theta**: subset S ⊆ [n] (which symbols are XORed), optionally with a constant/offset bit.
- **k**: n, or |S| if sparsity is the knob.
- **X**: `{0,1}^n`.
- **f(theta, x)** = XOR of `x_i` for `i in S`.
- **Encodings**: bit-vector; "coin flip" framing (heads/tails per switch, parity = odd/even count of heads among the relevant switches — switches unnamed/relabelable); a GF(2) equation surface form.
- **L3 posterior enumerable?**: yes, `|Theta| = 2^n`, same bound as above.
- **q***: standard-basis probe, non-adaptive, O(n) total.
- **A1–A7**: same shape as 2.1. **A4** boundary similar (n ≥ ~8). Notable: this family is the cleanest instance of the Hazard-1 asymmetry — cite it explicitly when justifying inclusion of a family whose worst-case *learnability from random examples* is bad (LPN-hardness under noise) but whose *query-teachability* is trivial.

### 2.3 Monotone DNF identification

- **Theta**: monotone DNF formula, m terms, each term a conjunction of ≤ r positive literals over n vars.
- **k**: (m, r, n) — e.g., k could ramp n and m together, holding r small.
- **X**: `{0,1}^n`.
- **f(theta,x)**: DNF evaluation.
- **Encodings**: bit-vector query against a rendered formula-free oracle (theta itself is hidden at L1–L3, so encoding is really about how x/y are rendered — same options as 2.1); could additionally vary between "assignment table" and "which switches are on" phrasing.
- **L3 posterior enumerable?**: no in general (the space of m-term r-literal monotone DNFs over n vars is huge combinatorially) — only tractable if Theta is restricted to a small enumerated template set (e.g., fixed small m, r and n ≤ 6–8), in which case the brute-force route from row 2.1's fallback applies.
- **q***: lattice walk (Angluin's monotone-DNF algorithm shape) — O(n) per query, does not require posterior enumeration.
- **A1–A7**: A1 pass. A2 pass if literal identities are symbol-permutable. A3 pass. **A4**: passes clearly once m, n grow past small values — DNF formulas are exactly the kind of structure whose lookup table (2^n rows) outgrows the O(mn) or O(mn²) structural solver quickly; this family stays admissible over a wide k range. A5/A6 pass. A7 pass but the heuristic is class-specific, not the generic `split-score` fallback — worth flagging as a **CHEAP-but-bad-worst-case** row per Hazard 1.

### 2.4 Bounded-depth decision tree identification

- **Theta**: a decision tree of depth ≤ d over n Boolean variables (internal nodes labeled by a variable, leaves 0/1).
- **k**: d primarily, n secondarily.
- **X**: `{0,1}^n`.
- **f(theta,x)**: tree evaluation (walk from root by x's bits, return leaf).
- **Encodings**: (a) query rendered as raw bit-vector, answer as leaf value; (b) "yes/no questionnaire" framing where each internal node is an anonymized switch-check. Theta itself is never rendered at L1–L3.
- **L3 posterior enumerable?**: no for realistic d, n — number of depth-≤d trees over n vars grows enormous. Only tractable for very small d (≤3) and small n, or if Theta is restricted to a small finite template library.
- **q***: `structure-walk-query` — the teacher, owning the true tree, generates x that traces an as-yet-untested root-to-leaf path (or a path that would distinguish two leaves the current version space hasn't yet separated). Cost O(d), no enumeration needed.
- **A1–A7**: A1 pass. A2 pass (variable identities anonymizable). A3 pass. **A4**: at depth d the tree encodes at most 2^d leaves against a domain of size 2^n; family stays non-trivial (fails brute memorization) once d is a small fraction of n, e.g., d ≤ n/2. Passes clearly for, say, n=20, d=6. A5/A6 pass. **A7 pass via structure-walk — this is the strongest example in the set of "teacher-side q* stays cheap even where the general learning-theoretic problem is hard,"** since a from-scratch learner facing this class is in NP-hard proper-learning territory (2023 result) while the teacher who already has theta pays O(d).

### 2.5 Small-DFA identification (finite templated automaton family)

- **Theta**: a DFA with ≤ N states over a small alphabet Σ, drawn from a *restricted, enumerable* template family (not "all DFAs ≤ N states," which is intractable — see map row).
- **k**: N, |Σ|.
- **X**: strings over Σ up to some length bound.
- **f(theta,x)**: accept/reject.
- **Encodings**: string rendered as a token sequence over an anonymized alphabet; alternative rendering as a sequence of colored-token "moves."
- **L3 posterior enumerable?**: only if Theta is restricted to a small template library (e.g., N ≤ 4, |Σ| ≤ 2, structural templates fixed) — enumerate that finite set directly, not "all DFAs."
- **q***: `structure-walk-query`, replaying L*'s access-string/distinguishing-suffix construction *forward* from the known automaton (teacher simulates its own transitions) rather than inferring it backward from an unknown one.
- **A1–A7**: A1 pass. A2 pass if alphabet symbols are anonymized. A3 pass (string vs. move-sequence rendering). A4: passes once N is large enough that the transition table (roughly N·|Σ| entries) is bigger than what a short generic string-matching heuristic could memorize — rough threshold N ≥ 6–8 for |Σ|=2. A5/A6 pass. A7: pass via structure-walk; **fails/EXPENSIVE if implemented via posterior enumeration over an unrestricted DFA space** — flag this explicitly as a repair-by-restriction case.

### 2.6 Generic finite-Theta version-space family (the fallback primitive as its own family)

- **Theta**: any small explicit enumerated hypothesis set (could be an arbitrary lookup-table family used as a controlled baseline, e.g. random Boolean functions over a small number of "features").
- **k**: |Theta| directly, or the number of underlying features.
- **X**: feature vectors.
- **f(theta,x)**: table lookup.
- **Encodings**: two arbitrary surface permutations of feature names/order.
- **L3 posterior enumerable?**: yes by construction — this family exists precisely to exercise `posterior-maintain`/`posterior-update` at controlled sizes.
- **q***: `split-score` + `majority-predict` (Halving Algorithm / generalized binary search), O(|Theta|) per call.
- **A1–A7**: A1 pass trivially. A2 pass if feature identities anonymized. A3 pass. **A4 fails outright for small |Theta|** (that's the point — this is the "trivially small lookup table" case the brief explicitly excludes) — **repair: only admissible once |Theta| is large enough that O(n)-per-query structural solvers (if any exist for the specific sub-case) beat brute table storage, or use this family purely as a calibrated instrument for measuring `split-score` cost, not as a capability-bearing member of the repertoire.** Useful as the planted "known answer" control family for the §5 measurement-validation step (task-specification.md step 4), not as a production family.

---

## 3. Rejections

| Candidate | Why it doesn't translate as-is | Repair |
|---|---|---|
| Angluin's L* for DFAs, used directly | L* requires an **equivalence query** oracle ("is this hypothesis DFA correct? if not, give a counterexample"). Our L2 oracle per task-specification.md §2.1 only "answers the query the model actually asked" — it is a pure membership-query oracle, not MQ+EQ. Importing L*'s query complexity bounds as our cost model overclaims. | Use L*'s *internal construction* (access strings, distinguishing suffixes) as the source of the `structure-walk-query` heuristic (teacher, owning theta, can synthesize what L* would have needed an EQ to discover) — this is legitimate because we own theta; just don't cite L*'s query-complexity *bounds* as bounds on our q*. |
| Learning halfspaces under continuous distributions (Gaussian etc.), general dimension | Most modern MQ-halfspace literature (Kelner 2020, agnostic variants) targets continuous domains under distributional assumptions foreign to our finite-Theta, backward-generation setup, and several results are explicitly about label-query vs. membership-query *lower bounds* for a from-scratch learner, not teacher-side query synthesis. | Discretize to a finite integer grid with a small weight bound; then row "Halfspaces" in §1 applies at small n, B via brute `split-score`. |
| Cryptographic hardness of MQ+EQ learning DNF (Angluin–Kharitonov 1995, referenced via k-term DNF literature) | This is a hardness result for a learner *without* knowledge of theta trying to properly-learn a formula; not directly about teacher-side query cost, which is our concern. | Repurpose as an A4 argument, not a q* argument: composing DNF-style families gives the family's *solver* (a model without theta) a provably hard floor, while the teacher (with theta) still pays cheaply — cite in the A4 discussion for DNF/CNF families, not in the q* map. |
| Certification of general monotone Boolean functions (2022 result) | Solves a different problem — producing a short certificate for a single point's value — not identifying theta from a sequence of queries. The O(k^8 log n) bound is for certification, not full exact learning. | Useful only as a component primitive (`structure-walk-query`-adjacent "find a minimal witness for this observation") inside a larger monotone-function family, not as a standalone q*. |
| General unrestricted monotone Boolean function class as a task family | Fails A1/A7 jointly at any interesting n: the class size is doubly-exponential (Dedekind numbers), so neither backward generation of "a random monotone function" nor posterior enumeration is tractable without heavy restriction. | Restrict to monotone DNF/CNF with bounded term count and size (§2.3) — same qualitative flavor, tractable generation and teaching. |
| Teaching dimension of "linear learners" (2016 arXiv result) | Concerns teaching a *gradient-descent learner* a target parameter via optimally chosen training points for an optimization procedure — a different learner model (continuous optimization, not discrete hypothesis elimination) than our L2 setup. | Not directly portable; flag as adjacent literature only, don't cite its teaching-dimension numbers for our halfspace row. |
| Adaptive submodularity / equivalence-class determination (Golovin & Krause) as a general q* recipe | Provides a *worst-case approximation guarantee* framework (near-optimal relative to the best adaptive policy) requiring the objective to be adaptive-submodular, which must be checked per family — not automatic, and the guarantee itself is exactly the kind of worst-case machinery Hazard 1 says to leave behind. | Take the algorithmic shape only (`split-score`, greedy one-step lookahead) as the heuristic; do not import the approximation-ratio proof machinery as a requirement for admissibility. |

---

## 4. Proposed primitives

| Slug | Gloss | Folded source terms |
|---|---|---|
| `posterior-maintain` | Keep an explicit representation (set or weighted set) of theta consistent with the history so far. | version space (Mitchell); candidate-elimination set; L*'s observation table (as a state, not its update rule) |
| `posterior-update` | Given a new `(x, y)` observation, shrink/reweight the surviving hypothesis set to those consistent with it. | eliminate-inconsistent hypotheses; Bayesian reweighting; single-step constraint propagation |
| `split-score` | Score a candidate query `x` by how evenly/informatively it divides the current posterior; pick the argmax. | generalized binary search / halving criterion; splitting index (Dasgupta); expected information gain (Lindley, MacKay); equivalence-class-determination objective (Golovin & Krause); uncertainty sampling |
| `structure-walk-query` | When theta has internal compositional structure (tree, automaton, circuit, lattice position), synthesize `x` by walking theta itself toward an untested part of its structure — no posterior over Theta is touched. | L*'s access-string + distinguishing-suffix construction, run forward from a known automaton instead of inferred from an unknown one; decision-tree path probing (Bshouty); boundary-following for convex-body/halfspace membership queries; monotone-DNF lattice walk (Angluin) |
| `basis-probe` | A fixed or lightly-adaptive query schedule that exploits algebraic structure of Theta to identify theta in O(k) queries without maintaining any posterior at all. | standard-basis / Gaussian-elimination query schedule (parity); single-flip literal-elimination walk (monomials) |
| `majority-predict` | Given the current posterior (from `posterior-maintain`), answer/predict by (weighted) majority vote over surviving hypotheses. | Halving Algorithm's vote rule; weighted-majority prediction — the L1/L3-facing counterpart of `split-score`, reusing the same maintained state |

Per A5, `structure-walk-query` and `basis-probe` are both specializations of "produce x cheaply without O(|Theta|) enumeration" — kept separate because one requires theta to have exploitable *compositional* structure (walkable graph/tree) and the other requires exploitable *algebraic* structure (linear/group structure enabling a fixed schedule); collapsing them would hide exactly the distinction the map in §1 needs.

---

## 5. Sources

**VERIFIED** (fetched and read actual abstract/content, not just a search snippet):

1. S. A. Goldman, M. J. Kearns, "On the Complexity of Teaching," *J. Comput. Syst. Sci.* 1995 / COLT 1991. Abstract confirmed via openscholarship.wustl.edu/cse_research/591/. Defines teaching dimension as the minimum number of instances a helpful teacher must reveal to uniquely identify a target concept.
2. N. H. Bshouty et al., "Adaptive Exact Learning of Decision Trees from Membership Queries," arXiv:1901.07750 (AISTATS/ALT 2019). Abstract read directly: randomized Õ(2^{2d})+2^d log n and deterministic 2^{5.83d}+2^{2d+o(d)} log n query bounds, improving Feldman and Kushilevitz–Mansour.
3. "Faster exact learning of k-term DNFs with membership and equivalence queries," arXiv:2507.20336 (2025). Abstract read directly: improves classical poly(n,2^k) to poly(n)·2^Õ(√k), first improvement since Blum–Rudich 1992, uses Winnow2 over an adaptively-constructed feature space.
4. "On Exact Learning Monotone DNF from Membership Queries," arXiv:1405.0792. Abstract partially read: deterministic/randomized adaptive algorithms with "almost optimal" and asymptotically tight (for fixed r and/or s) query complexities for m-term, size-r monotone DNF.
5. S. Dasgupta, "Analysis of a Greedy Active Learning Strategy," NeurIPS 2004 (cseweb.ucsd.edu/~dasgupta/papers/greedy.pdf). Fetched directly: the greedy rule queries the point dividing the version space as evenly as possible; cost is linear in version-space size per query; proven within a log|H| factor of the optimal adaptive policy.
6. D. Golovin, A. Krause, "Adaptive Submodularity: Theory and Applications in Active Learning and Stochastic Optimization," JAIR 2011 / arXiv:1003.3967. Abstract read: generalizes submodularity to adaptive policies; greedy adaptive-submodular policies are competitive with optimal; applies to equivalence class determination, optimal decision tree, decision region determination (per search-snippet corroboration of the same paper).
7. Wikipedia, "Teaching dimension." Fetched directly (thin content): formal definition, 1995 Goldman–Kearns formulation building on Goldman–Ron Rivest–Schapire 1993; notes CT/OT/RT/PBT/NCT variants; no concrete class-by-class numbers.
8. T. Rainforth, A. Foster, D. R. Ivanova, F. Bickford Smith, "Modern Bayesian Experimental Design," arXiv:2302.14545 (to appear, *Statistical Science*). Abstract text retrieved via fetch tool: "BED provides a powerful and general framework for optimizing the design of experiments... its deployment often poses substantial computational challenges." Full-text technical detail (EIG formula, cost scaling) could not be extracted — every attempt returned garbled binary from the PDF layer.
9. D. J. C. MacKay, "Information-Based Objective Functions for Active Data Selection," *Neural Computation* 4(4), 1992. Abstract content corroborated across MIT Press page and multiple secondary summaries (direct HTML fetch returned 403): three information-theoretic data-selection criteria, EIG = uncertainty given observed data minus expected uncertainty after the next measurement; criteria depend on the hypothesis space being correct.

**LEAD** (cited via search-result snippets or secondary description only; primary PDF could not be rendered, or not fetched):

10. D. Angluin, "Learning Regular Sets from Queries and Counterexamples," *Information and Computation* 75(2), 1987 (the L* algorithm). Query-complexity characterization (polynomial in DFA size and longest counterexample) taken from course-note secondary sources (Oxford CS AML lecture notes, cs.ox.ac.uk).
11. D. Angluin, "Queries and Concept Learning," *Machine Learning* 2, 1988. Direct PDF fetch (pbworks mirror) returned unreadable binary on every attempt. Content used here (k-CNF/k-DNF learnable from EQ alone in poly(n^k); monotone DNF learnable from MQ+EQ; base conjunction/monomial algorithm) is drawn from secondary citations (Springer abstract page, GM-RKB summary, CMU 10-806 slide title, ResearchGate) and should be treated as reconstructed, not directly quoted.
12. J. L. Balcázar, J. Díaz, R. Gavaldà, O. Watanabe, "The Query Complexity of Learning DFA," *New Generation Computing* 12, 1994. Snippet only: proves MQ-alone insufficiency and MQ/EQ alternation lower bounds for DFA learning.
13. Lindley, D. V., "On a Measure of the Information Provided by an Experiment," *Annals of Mathematical Statistics* 27(4), 1956. Never fetched directly; citation and content (EIG as the Shannon-derived measure of experimental informativeness) taken from multiple secondary BED reviews.
14. T. Doliwa, G. Simon, S. Zilles, "Recursive Teaching Dimension, VC-Dimension and Sample Compression," *JMLR* 15, 2014. Snippet only: RTD ≤ VC-dimension in many cases; connects teaching to sample compression.
15. H. U. Simon, "Open Problem: Recursive Teaching Dimension versus VC Dimension," COLT 2015. Snippet only: classes exist where RTD exceeds VC-dim by a factor > 3/2 (up to 5/3 shown); only known general upper bound on RTD in terms of VC-dim is exponential.
16. G. Blanc et al., "The Query Complexity of Certification," arXiv:2201.07736, 2022. Snippet only: O(k^8 log n)-query certification algorithm for monotone functions with certificate complexity k, vs. Ω(k log n) information-theoretic floor; classical Valiant/Angluin certification costs n queries.
17. "Exact VC-Dimension of Monotone Formulas" (Procaccia et al., NIPS-era workshop paper). Snippet only: VC-dimension of monotone monomials over n variables is exactly n.
18. "Properly Learning Decision Trees with Queries Is NP-Hard," arXiv:2307.04093, 2023. Snippet only: resolves a Bshouty 1993 open problem; proper PAC-learning of decision trees with query access is NP-hard for a from-scratch learner.
19. O. Kelner, "Learning Halfspaces With Membership Queries," arXiv:2012.10985, 2020. Abstract fetch returned only a generic sentence ("near optimal label complexity in some cases"); treated as LEAD given the shallow extraction.
20. "Lower Bounds for the Complexity of Learning Half-Spaces with Membership Queries," Springer LNCS. Snippet only (title + venue); not read.

---

## 6. Source's taxonomy (quarantined — not our capability decomposition)

Per Hazard 2, these are recorded only so they can be scored against our own empirically-measured capability decomposition later, not used to pre-shape it.

- **Angluin's query taxonomy**: concept classes are grouped by *which combination of oracle types* suffices — membership-only, equivalence-only, membership+equivalence, plus (in follow-on work) subset/superset/disequivalence queries. This is a taxonomy over *oracle access patterns*, not over cognitive operations.
- **Teaching-dimension family's own taxonomy**: classical teaching (CT) vs. recursive teaching (RTD) vs. preference-based teaching (PBT) vs. no-clash teaching (NCT) vs. self-directed learning — these are different formalizations of "how much a teacher/learner pair can cooperate," ordered by how much collusion between teacher and hypothesis-class structure is allowed. Distinct axis from difficulty or structure type.
- **Modern BED's own taxonomy** (per Rainforth et al. abstract framing): designs are grouped by *myopic/greedy one-step-lookahead* vs. *non-myopic/amortized* experimental design, and by *implicit-likelihood* vs. *explicit-likelihood* settings — an axis about computational strategy for the same underlying EIG objective, not about problem content.
- **Adaptive-submodularity's own taxonomy** (Golovin & Krause): problems are grouped by whether the objective function satisfies adaptive submodularity and adaptive monotonicity, which determines whether the greedy policy carries a provable approximation guarantee — an axis about when a proof applies, not about task content.

None of these were used to select or group the candidate families in §2; grouping there follows the hypothesis-class boundaries the source papers happened to state results about, which is a citation convenience, not a claim that those are the natural joints for our repertoire.

---

## 7. Sealed encounters

One near-miss, stopped without reading:

- A WebSearch for teaching-dimension / active-learning cost results surfaced multiple titles adjacent to "Minimax Analysis of Active Learning" and general pool-based active-learning survey material during the Dasgupta/GBS search pass. These were not opened beyond the search-result title/snippet level once it became clear they were framed as applied pool-based active-learning-for-classifiers work (the sealed category), as distinct from the Dasgupta 2004 theoretical splitting-index paper (opened, in scope as classical query-complexity theory). No content from the applied-active-learning direction was read or used.

No leads fell inside clinical trials, ILP, Latin squares/graph coloring, developmental/animal cognition, psychometrics, clinical diagnosis, or ARC-AGI-3 material during this pass.
