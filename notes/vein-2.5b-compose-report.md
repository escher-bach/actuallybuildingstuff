# Vein 2.5b — Composition: Library Learning, Grammatical Inference, Planning

Coverage gap addressed: candidate set currently has nothing exercising the `composition`
content axis or the compose/decompose generation asymmetry.

Sealed topics (per instructions, NOT to be fetched): Inductive logic programming (Metagol,
Popper, lineage); non-primate animal cognition / developmental rule-learning; verbal/linguistic
psychometric item generation; clinical reasoning; pool-based applied active learning /
clinical-trial adaptive design; quasigroup/Latin-square completion, graph-colouring phase
transitions; SRE/incident response, SQL practice, Erlang/OTP; Euclidean construction geometry,
epsilon-delta analysis practice; ARC-AGI-3 design material. **One contamination occurred — see
Section 9.**

---

## 1. Compose/decompose families

Three families designed from the gathered evidence. All are stated with a permutation-invariant
alphabet (A2) by construction — "symbol" below always means an arbitrary opaque token, never a
meaningful identifier.

### 1.1 Abstraction-Extraction (library-refactoring family)

Directly modeled on DreamCoder's own Abstraction sleep-phase (Section 2, verified) — the
refactoring step *is* a compose/decompose family in the source material itself, not just an
analogy.

| Field | Definition |
|---|---|
| `Theta` | `(P0, g)`: a fixed finite primitive alphabet `P0` of `m` opaque unary/binary combinators (semantics fixed by a lookup table, e.g. random total functions on a small domain, so behavior is well-defined but symbol identity carries no leaked meaning), plus a hidden **abstraction** `g` = one syntax tree of depth `d_g` over `P0`. |
| `P_Theta(.\|k)` | Sample `g` (a tree over `P0` of depth `d_g`), then sample `n(k)` "surface programs" `{prog_i}`, each built by embedding one call to `g` at a random position inside a larger random tree over `P0 ∪ {g}` of total depth `k`. `k` = **composition depth of the surface programs**, `n(k)` = how many independent surface programs embed `g` (more embeddings = more evidence `g` is reusable, mirrors DreamCoder needing "a moderate number of tasks" — DreamCoder's own text: "typically only a moderate number of tasks suffices," and the system was "typically trained on 100-200 tasks," Dreaming-phase paragraph). |
| `X` (query space) | Input tuples to any surface program or to `g` directly. |
| `f` (oracle) | Evaluates a *given* program (surface program or a proposed abstraction) on a query input by executing the tree over the fixed `P0` semantics — pure evaluation, no search. Separately, the oracle can VERIFY a proposed abstraction: given a candidate subtree `s` and its claimed occurrence positions, checks equality against the actual shared subtree — an O(1)-per-check, non-search operation, matching DreamCoder's own description of storing the actual refactoring set in a version-space data structure rather than searching it exhaustively (verified quote, Section 2). |
| `E` (encodings) | (a) explicit symbolic tree/prefix-string with permuted primitive names, matching DreamCoder's own Lisp-like rendering; (b) pure input-output table per surface program (structure fully hidden) — the two representations DreamCoder itself contrasts in Fig. 1B ("solution... discovered in learned language" vs. "if expressed in initial primitives"). |
| Oracle computes | Program evaluation; abstraction-match verification. |
| L3 posterior enumerable? | Yes for small `m, d_g`: finite set of depth-`d_g` trees over `P0`. Grows combinatorially in `d_g` — DreamCoder's own worked example puts a *single* refactoring search at "~10^14 possible refactorings" for one small program pair (verified quote, Fig. 3) even though the resolved version-space has only ~10^6 nodes — i.e., raw enumeration is intractable quickly, but the compressed belief-state representation (their version-space / e-graph structure) stays tractable. This is direct, sourced evidence for `posterior-enumeration` needing a compressed representation, not flat enumeration, past small `k`. |
| `q*` | One-pass teacher: present surface programs so that `g`'s occurrences are placed at *maximally distinguishing* positions (e.g., vary the filler context per embedding while holding `g` fixed) — computable in one pass from `theta` since the teacher already knows `g`. |

**A1–A7 triage:**
- A1 (backward-generable): **pass**, as specified — generation samples `g` then embeds it; evaluation and match-verification are both O(1)-per-query. **Caveat/repair needed**: the *naive* framing "find the MOST reusable / MDL-optimal abstraction" (DreamCoder's actual objective) is a search problem — DreamCoder itself needed a bespoke version-space/e-graph data structure and a bounded evaluation-step distance (default bound 3, verified quote) to make it tractable at all. **Repair applied above**: task asks the solver to propose *a* candidate abstraction and the oracle *verifies* it against the ground truth `g` baked into `theta`; it does not ask for the optimal one. This is exactly the A1 trap the brief warns about, and the family as specified sidesteps it by construction.
- A2 (knowledge-free): pass, given opaque `P0` symbols.
- A3 (encoding-varied): pass, two genuinely different renderings given above.
- A4 (brute-force-resistant): pass, conditionally. A solver ignorant of composition must, in the worst case, test all depth-`d_g` subtrees at all positions across all surface programs — combinatorial in tree size, matching DreamCoder's own 10^14-refactorings example. A solver that exploits the fact that the SAME subtree recurs across `n(k)` independent programs can instead intersect candidate sets per program (this is literally what DreamCoder's version space buys it) — cost roughly linear in `n(k)`. Composition depth `k` (surface program depth) is the difficulty knob; **monotonicity is plausible but not verified** — DreamCoder's own Fig. 6B shows performance saturating with library size/depth rather than growing without bound, which is a *different* saturating quantity (learned-library performance across iterations, not raw difficulty vs `k`). Flag as INFERENCE, not sourced fact.
- A5 (semantically coherent): pass, `P0` semantics fixed.
- A6 (L2): pass — one evaluation per query, malformed programs (ill-typed trees) simply return a fixed error token.
- A7 (L2): pass, teacher policy given above, one-pass.

### 1.2 Composition-Chain / Function-Pipeline family

Not sourced from a single paper; constructed to exercise the reused primitives `bijection-invert`,
`modular-add`, `basis-probe`, `structure-walk-query` directly, per the brief's algebraic-structure
framing (A4: "Composition is our main tool for this").

| Field | Definition |
|---|---|
| `Theta` | `(S, P0, <f_1,...,f_k>)`: a fixed finite algebraic domain `S` (e.g. `Z_n` under addition, or bijections on an `n`-element set), a fixed library `P0` of `m` named primitive operations on `S` (each a bijection on `S`, e.g. modular-add-by-constant, or a fixed permutation), and a hidden ordered pipeline of `k` primitive-labels drawn from `P0` (with replacement). `k` = **composition depth**, the difficulty knob. |
| `P_Theta(.\|k)` | Sample `<f_1,...,f_k>` i.i.d. (or without replacement) from `P0`; `g = f_k ∘ ... ∘ f_1`. |
| `X` | Elements of `S`. |
| `f` (oracle) | Default: evaluates `g(x)` for query `x` — end-to-end only, no intermediate readout, forcing genuine decompose reasoning. (An L2 variant may expose a declared "probe" channel at one designated internal stage as part of `E`, but never as a free choice — one channel, fixed by `theta`.) |
| `E` | (a) explicit expression string with permuted primitive names, e.g. `g(x)=p7(p2(p9(x)))`; (b) pure black-box query-response access to `g`, structure fully hidden. |
| Oracle computes | `g(x)`. |
| `q*` (target) | The ordered label sequence `<f_1,...,f_k>` (or a designated single stage). |
| L3 posterior enumerable? | Yes for small `m,k` (`m^k` candidates, finite); for larger `m,k` maintain the belief state incrementally via `constraint-propagation`/`hypothesis-elimination` as probes narrow per-stage candidate sets, rather than flat enumeration. |
| `q*` teacher | `structure-walk-query`: since bijections are invertible, a one-pass teacher can choose probes that peel from the outside in (query `g` at points designed so the outermost unknown stage is isolated via `bijection-invert` on the known suffix) — computable in one pass given `theta`. |

**A1–A7 triage:**
- A1: pass — sampling and evaluation are both O(k) (apply each stage) or O(1) for a single query.
- A2: pass, given opaque primitive labels.
- A3: pass, two encodings given.
- A4: pass, with a genuine, argued non-monotonicity. Brute-force cost for a structure-ignorant solver is `O(m^k)` (must search the label sequence blind) — strictly increasing in `k`. But **INFERENCE, not sourced**: if `P0` is drawn from a commutative/abelian operation (e.g. all primitives are modular-adds), the composition COLLAPSES — `g` reduces to a single net modular-add regardless of `k`, so individual stage identity becomes non-identifiable (indistinguishable pipelines) even as raw brute-force enumeration cost keeps climbing. This means *identifiability* (the thing that actually matters for the L1/L3 targets) is **non-monotonic-adjacent**: it can degrade to a fixed low ceiling for some `P0` choices regardless of `k`, while for non-commutative `P0` (e.g. general permutations) identifiability stays informative as `k` grows. This is a design-time choice, not an emergent phase transition, but it is the same *family* of caution the brief asks to check for, and it is cheap to avoid (require non-commuting `P0`) — flagging explicitly since it is easy to get wrong by accident.
- A5: pass.
- A6: pass, single evaluation per query, malformed `x` outside `S` returns a fixed error token.
- A7: pass, `structure-walk-query` teacher above.

### 1.3 Subgoal-Segmentation (macro-plan decomposition family)

Grounded in DreamCoder's own tower-building "parametric options" result (`arch(h)`, `pyramid(h)`,
`wall(w,h)`, `bridge(w,h)`, explicitly citing Sutton/Precup/Singh 1999 options — verified, Fig.
5B) and in the planning-generator material (Section 3 below).

| Field | Definition |
|---|---|
| `Theta` | A fixed library `M` of `m` named "macro" templates, each a deterministic mapping from a small parameter tuple to a fixed-length sequence of primitive actions (mirrors DreamCoder's `arch(h)`, `bridge(w,h)` etc., which are literally parametric action sequences); a hidden ordered macro-sequence `<mu_1,...,mu_k>` with sampled parameters. |
| `P_Theta(.\|k)` | Sample `k` macros (with parameters) from `M`; concatenate their expansions into one flat primitive-action sequence — this is the planning literature's own backward-generation trick ("apply a random action sequence from a state, then ask for the plan," per the brief's framing) at one level up (random *macro* sequence, not random primitive actions). |
| `X` | States reachable by executing a prefix of the flat action sequence. |
| `f` (oracle) | Evaluates preconditions/effects of the flat action sequence deterministically (standard STRIPS-style forward simulation) — no search, since the plan is constructed forward and is valid by construction. |
| `E` | (a) flat primitive-action-sequence string; (b) sequence of intermediate states/images (as in DreamCoder's tower "copy tasks," which give both the image and block locations — verified, Results section). |
| Oracle computes | State after executing a given action prefix; validity check of a proposed action. |
| `q*` (target) | The macro-boundary segmentation of the flat sequence — i.e. which primitive-action spans correspond to which macro instance, matching the brief's own candidate "given a plan recover the subgoal structure." |
| L3 posterior enumerable? | For small `m,k`: yes, `m^k` candidate macro-sequences, prunable by which segmentations are even consistent with `M`'s fixed-length templates (`constraint-propagation` over span boundaries). |
| `q*` teacher | Not attempted here beyond noting the family is well-formed; would follow the same `structure-walk-query` pattern as 1.2. |

**A1–A7 triage:**
- A1: pass, as specified (proposing a valid segmentation, not the shortest/most-compressed one — DreamCoder's own "options" are found by *its* MDL-style compression search, which is NOT what this family asks the solver to reproduce; the family only asks for A segmentation consistent with the known macro templates, side-stepping the optimality trap the way 1.1 does).
- A2: pass, opaque macro/action names.
- A3: pass.
- A4: **repairable, not a clean pass.** If macro templates have visibly different fixed lengths, boundary recovery can be nearly free (read off length changes) — a repair is to require several macros share output length so boundaries are not locally decodable, forcing the solver to use compositional/semantic structure (state preconditions) rather than surface segmentation. Flagged as a repair, not asserted as already fixed.
- A5: pass.
- A6: pass.
- A7: not filled in above — mark **unknown**, teacher policy not designed to completion in this pass.
- Difficulty-in-`k` monotonicity: **explicitly not established** here. The planning-generator literature (Section 3) shows non-monotonicity is real for at least one planning difficulty parameter (operators-to-state-variables ratio, Rintanen 2004, verified in depth) but that is a *different* parameter (density of a randomly sampled STRIPS instance) from `k` = macro-composition depth as defined here. Do not conflate the two — flagged as a distinct open question, not answered by the cited evidence.

---

## 2. Primitive inventories (concrete)

### 2.1 DreamCoder (Ellis et al. 2020/2021, arXiv:2006.08381) — VERIFIED, full main text read (pp.1-20, direct PDF read, not the fetch-tool summarizer)

Substrate: polymorphically-typed lambda calculus, one DSL per domain. All entries below are
quoted or closely paraphrased from the paper text (Results section, Figs 1/4/5/7, "From learning
libraries to learning languages" section).

| Domain | Initial primitives (as stated) | Discovered abstractions (as stated) | Task count / split |
|---|---|---|---|
| List processing | "generic functional programming basis, including routines like `map`, `fold`, `cons`, `car`, `cdr`, etc." | "composed around 20 new library routines"; rediscovers `filter`; Fig.1B shows a 4-layer-deep chain building to `sort` (concept_4≈filter-like, concept_13≈maximum, concept_15≈nth-largest) | 218 problems (from Ellis et al. 2018 NeurIPS), 50/50 test/train, 15 I/O examples each |
| Text editing | not enumerated in fetched main text (S1.1 supplement not fetched) | learned "a single library of text-editing concepts" applied generically | 128 auto-generated train tasks; tested on 108 SyGuS-2017 problems |
| LOGO graphics | pen control + imperative control flow + arithmetic on angles/distances | Fig.4B/C: `semicircle(r)`, `circle(r)`, `spiral(dθ)`, `greek-spiral(n)`, `s-curve(r)`, `polygon(n,l)`, higher-order `radial-symmetry(n, body)` (a function taking a whole program as input) | 160 images, 50/50 split |
| Tower building | "same control flow primitives as with LOGO graphics" | Fig.5B, "parametric options" (citing Sutton/Precup/Singh 1999): `arch(h)`, `pyramid(h)`, `wall(w,h)`, `bridge(w,h)` | 107 tower "copy tasks", 50/50 split |
| Regex | not enumerated in fetched main text | regexes for phone numbers, dates, times, monetary amounts | 256 CSV columns crawled from web, 5 example strings/concept, 50/50 split |
| Symbolic regression | `+`, `×`, `÷`, continuous real parameters (fit by gradient descent, BIC-penalized) | parametric curve programs, e.g. 3-parameter program for `1.7x²−2.1x+1.8`, 2-parameter for `−2.3/(x−2.8)` | curves up to degree-4 polynomial/rational |
| Physical laws | `subtract-vectors, map, zip, cons, empty, cdr, power, fold, car, +, -, *, /, θ, 1, π` (Fig.7A) | vector-algebra layer: add-vectors → add-many-vectors → norm² → scale → inverse-square-schema → etc.; reconstructs Newton's 2nd law, parallel-resistor formula, work, magnetic force, kinetic energy, Coulomb's law | 60 physical laws/identities from AP/MCAT "cheat sheets"; **after 8 wake/sleep cycles DreamCoder learns 93% of the laws and identities in the dataset** (exact figure, read directly from Results text) |
| Recursive/1959-Lisp programming | minimal subset of 1959 Lisp primitives: `car, cdr, cons, ...` + Y-combinator | rediscovers `fold`/`unfold` ("origami programming," Gibbons 2003) as dual roots, then `map`, `filter` as fold-family variants, then `index` combining both families; example outputs: Stutter, "take every other," list-lengths, list-differences | 20 basic programming tasks; "with enough compute time (roughly five days on 64 CPUs), DreamCoder learns to solve all 20 problems" |

Other exact figures read directly from text:
- "We typically train DreamCoder on 100-200 tasks" (p.7).
- Text editing: pre-learning 3.7% solved (avg 235s search); post-learning 79.6% (avg 40s); 2017-SyGuS
  winner CVC4 solved 82.4% under competition compute (1hr, 8 CPUs/problem); DreamCoder under that
  same budget solves 84.3%.
- "Across domains, deeper libraries correlate well with solving more tasks (r = 0.79)" (p.13, exact
  quote, Pearson correlation library-depth vs. % held-out solved).
- Refactoring mechanism, quoted: "we bound the number of λ-calculus evaluation steps separating a
  program from its refactoring, giving a finite but typically astronomically large set of
  refactorings... we introduce a new data structure... combining ideas from version space algebras
  and equivalence graphs [e-graphs]... A version space with 10^6 nodes, calculated in minutes, can
  represent the 10^14 refactorings in Fig. 3." Default evaluation-step bound = 3.

**Saturation curve** (Fig.6A/6B, read directly, not fetch-summarized): Fig.6A plots % held-out
test solved against wake/sleep cycle (0-20) for six domains — full-model curves rise steeply in
the first ~5-10 cycles then visibly flatten by cycle ~15-20 in all six panels. Fig.6B re-plots %
test solved against **library size** (x-axis 0-25) and **average program depth** (x-axis
~1.0-2.75): performance rises then visibly levels off past roughly library size 15-20. This is
the paper's own closest analogue to "does abstraction discovery flatten with more tasks" — a
performance-saturation curve tied to library growth, not literally a per-iteration
new-abstraction-count plot (no such plot was found in the fetched main text; S1.1 supplement not
fetched, so a literal count-of-new-primitives-per-iteration curve could not be located and is not
claimed). The shape (rises-then-flattens) is a direct read of the figure; the *cause* is
INFERENCE.

### 2.2 DreamCoder successor — Stitch (Bowers et al., "Top-Down Synthesis for Library Learning," arXiv:2211.16605) — VERIFIED via abstract only (PDF body not fetched)

Substrate: same general library-learning setting as DreamCoder (corpus-guided top-down synthesis
of reusable library functions from a corpus of programs). From the abstract, verified: "3-4
orders of magnitude faster" than DreamCoder's deductive compression algorithm, "2 orders of
magnitude less memory," "comparable or better library quality (as measured by compressivity)."
Robust to early termination, scales to "corpora containing hundreds of complex programs." **Not
verified**: which specific domains/corpora (list processing vs. LOGO vs. physics) it was
evaluated on, and no independent primitive-inventory or saturation-curve figures were located
(PDF body not fetched — flagged rather than guessed).

### 2.3 PCFG SET (Hupkes, Dankers, Mul & Bruni, "Compositionality Decomposed," arXiv:1908.08351) — VERIFIED, body read directly (pp.12-13 of PDF)

Substrate: a string-rewriting DSL, not a program-synthesis DSL — included here because its
primitive set is exactly a compose/decompose operator algebra and is a strong candidate ANCHOR
for family 1.2/1.3 designs. Exact grammar (Figs.2-3, read directly):
- Non-terminal rules: `S → F_U S | F_B S,S`; `S → X`; `S → X X`.
- Lexical rules: `F_U → copy | reverse | shift | echo | swap | repeat`; `F_B → append | prepend |
  remove_first | remove_second`; `X →` (any of 520 distinct string-argument tokens).
- Interpretation functions (Fig.3, exact): `copy x1..xn → x1..xn`; `reverse x1..xn → xn..x1`;
  `shift x1..xn → x2..xn x1`; `swap x1..xn → xn x2..x(n-1) x1`; `repeat x1..xn → x1..xn x1..xn`;
  `echo x1..xn → x1..xn xn`; `append x,y → x y`; `prepend x,y → y x`; `remove_first x,y → y`;
  `remove_second x,y → x`.
- Grammar is recursive → arbitrary nesting depth/composition; string alphabet size set to 520;
  base corpus ~100,000 distinct input-output pairs; argument strings length-limited to 5; 85/5/10
  train/val/test split; string arguments deliberately never repeated across the corpus to
  discourage memorization (all exact, read directly).
- Depth/length distribution deliberately "naturalised" to match a parsed WMT-2017 English corpus
  (Fig.4 comparison, read directly) — i.e. difficulty/complexity is controlled via PCFG production
  probabilities, not just nesting-depth caps.
- The five compositionality tests (systematicity, productivity, substitutivity, localism,
  overgeneralisation) are the SOURCE'S OWN taxonomy — quarantined to Section 8, not adopted as our
  capability decomposition.
- **Not verified**: per-test accuracy numbers — the Results section (beyond p.14) was not fetched;
  flagged rather than guessed.

---

## 3. Difficulty parametrizations

### 3.1 Grammatical inference — Tomita grammars

VERIFIED, Table 1 read directly from Wang, Zhang, Ororbia, Xing, Liu & Giles, "An Empirical
Evaluation of Rule Extraction from Recurrent Neural Networks," arXiv:1709.10380, p.7 (citing
original source Tomita 1982a, and Giles et al. 1990 for grammar 5's exact phrasing).

| G | Exact description (Table 1, quoted) |
|---|---|
| 1 | `1*` |
| 2 | `(10)*` |
| 3 | an odd number of consecutive 1s is always followed by an even number of consecutive 0s |
| 4 | any string not containing "000" as a substring |
| 5 | even number of 0s and even number of 1s |
| 6 | the difference between the number of 0s and the number of 1s is a multiple of 3 |
| 7 | `0*1*0*1*` |

Facts, all read directly from that paper's Section 2.2: "the DFA associated with these grammars
have between three and six states"; all have alphabet `{0,1}`; "the languages are numbered 1-7
such that the difficulty of learnability increases with number" — **this ordering is the field's
convention, cited by this paper to (Wang et al., 2018), not independently derived here** —
flagged accordingly. The same paper gives an evidence-based account of WHY difficulty differs,
distinct from the raw index: "grammars 1, 2 and 7 represent the class of regular languages that
define a string set that has extremely unbalanced positive and negative strings"; "grammars 5 and
6 define the class of regular languages that have equal or a relatively balanced number of
positive and negative strings"; "grammars 3 and 4 represent the class... somewhere between the
above two cases." So the actual parametrization proposed (by the cited literature, via this
paper) is **class-balance of accepted vs. rejected strings**, not grammar index per se — index
number is a proxy, not the mechanism. Some DFAs additionally contain a "garbage state" (non-final
state, all transitions self-loop) that must be learned from negative examples specifically — a
second, orthogonal difficulty source noted in the same section.

### 3.2 Grammatical inference — Abbadingo One DFA-learning competition

VERIFIED, fetched directly from the primary competition site (abbadingo.cs.nuim.ie: homepage,
QA.html, data-sets.html — plain HTML, not compressed PDF, so fetch-tool text extraction is
reliable here).

**Generation procedure (QA.html, quoted/closely paraphrased):** "To generate a random DFA with
roughly N states start with a digraph containing 5/4 N nodes and no edges. From each node add two
outgoing edges, with the destination nodes chosen randomly from a uniform distribution over the
5/4 N nodes." Then: randomly select a start state; remove unreachable nodes; assign accept/reject
labels by fair coin flip; apply Moore's DFA-minimization algorithm; discard and regenerate unless
the resulting DFA's depth matches `(2·log2(N)) − 2`, rounded to the nearest integer. Training/test
strings are drawn without replacement from a uniform distribution "over the set of all strings not
longer than 5 plus the target DFA depth" (this — string-length cap relative to DFA depth — is the
paper's operationalization of "sparseness"/density; no separate closed-form density formula was
located beyond this sampling-distribution description).

**Numeric parameters (data-sets.html, quoted):** target DFA state counts (size parameter) used:
practice problems 61, 119, 247, 498; official problems in two further batches: {63, 138, 260,
499}, {68, 130, 262, 506}, {65, 125, 267, 519}; corresponding target depths 10, 12, 14, 16 (same
pattern for practice and official). Training-set sizes for the instance table range 1,521 to
115,000 strings across the 16 problems (size × density cross product, homepage: "the competition
consisted of sixteen problems representing the cross product of 4 values of a target size
parameter and 4 values of a training set density parameter").

**Monotonicity/non-monotonicity:** the homepage states difficulty is explicitly **not a single
total order**: "problems are ranked according to two factors that affect their difficulty: the
size of the target concept and the sparsity of the training data," and "because some pairs of
problems are of incomparable difficulty, the competition might end up with multiple winners" —
the organizers used a "two-dimensional dominance lattice" for ranking rather than a scalar
difficulty score. This is a genuine, sourced example of a difficulty space that is only
**partially ordered**, not simply non-monotonic in one parameter — worth distinguishing from
Section 3.3's finding, which IS single-parameter non-monotonicity.

### 3.3 Planning — non-monotonic difficulty (phase transitions)

VERIFIED, read directly (not fetch-tool summarized — the fetch-tool's own paraphrase of this paper
was checked against the primary PDF and found to contain unverifiable/likely-fabricated exact
quotes; only the directly-read text below is used). Source: Jussi Rintanen, "Phase Transitions in
Classical Planning: An Experimental Study," KR 2004, pp.710-714.

- Builds on Bylander (1996) "Model B": randomly-sampled STRIPS instances parametrized by `n`
  (Boolean state variables), `m` (operators), `s` (preconditions/operator), `t`
  (effects/operator), `g,g'` (goal-literal counts); ratio `c = m/n` used as the control parameter.
  Rintanen introduces refinements "Model A" and "Model C" that remove the most trivially
  insoluble instances from Bylander's sampling.
- Abstract, quoted: "as the ratio between the number of clauses and number of propositions
  approaches 4.2 from below, the probability that the formula is satisfiable increases.
  Similarly, when the ratio approaches 4.3 from above, the probability that the formula is
  satisfiable decreases... The phase transition from 1 to 0 at 4.27 coincides with the difficulty
  of testing the satisfiability of the formula... far below 4.27 and far above it [runtimes]
  decrease sharply... This is the *easy-hard-easy pattern* at the phase transition region." (This
  is background on SAT, cited by Rintanen for context — SAT phase transitions are not on the
  sealed list, only "graph-colouring phase transitions" and "quasigroup/Latin-square" are.)
- Direct finding for planning, quoted: "the computational difficulty of the problem instances in
  Model A – for all the planners experimented with – peaks when the ratio between the number m of
  operators and the number n of state variables is about 2 (assuming certain fixed values for the
  rest of the parameters)."
- "There is a transition from hard to easy instances as the ratio c of operators to state
  variables grows beyond 3... difficulty in Model A peaks at about operators-to-variables ratio 2,
  and in Model C at about ratio 2.5, at least for SP [the SAT-based planner] ... FF's runtime
  curves do not suggest the same, as the curves peak later respectively at about ratios 2.7 and
  3."
- Exact instance counts, quoted: "For model A, we produced between 350 and 608 soluble problem
  instances for each ratio of operators to state variables, and for model C between 89 and 784...
  for smaller ratios this involved testing the solubility of up to 50000 (for model A) and up to
  20000 (for model C) problem instances."
- Scaling to larger instances, quoted: "With 20 state variables FF's success rate is close to 100
  per cent but with 40 state variables it is 4.3 per cent on the hardest instances and still only
  about 90 percent at the very easy ratio of 6."
- Figures 1 and 2 (both inspected directly) plot, for problem instances with 20 state variables:
  proportion of soluble instances (a rising sigmoid in ratio) overlaid with (a) average planner
  runtime and (b) average plan length — both runtime and plan-length curves visibly **rise then
  fall** as the ratio increases through the transition region, i.e. difficulty (by both runtime
  and required plan length) is **non-monotonic in the operators/state-variables ratio**, peaking
  in the transition region and decreasing on both sides. This is the concrete, verified instance
  of the "expect non-monotonicity" pattern the brief flags, for a real planning difficulty
  parameter — but note it is a **ratio-of-generator-parameters**, not literally "plan length" or
  "composition depth" as a knob; see the explicit non-conflation flag in family 1.3 above.
- Planners used: SP (satisfiability-based, via SAT solver Siege v3), FF (Hoffmann & Nebel), LPG
  (Gerevini & Serina) — none of these are named classical domains (blocksworld/logistics/etc.);
  the problem instances are abstract randomly-sampled STRIPS structures, not blocksworld. (An
  earlier fetch-tool paraphrase of this same paper incorrectly claimed the domains were
  "Blocksworld, Logistics, and Gripper" — that claim is **not supported** by the directly-read
  text and is explicitly rejected here as an example of exactly the numbers/fact-drift risk this
  review was warned about.)

### 3.4 Planning — instance generators, general

VERIFIED (GitHub HTML page, read directly): `AI-Planning/pddl-generators` on GitHub collects
generators for 60+ IPC benchmark domains (blocksworld, sokoban, gripper, logistics, depots,
rovers, satellite, transport, etc. — full list read from repo directory listing). The repo
README states generators are runnable via included makefiles/scripts, with per-domain README
files "in the less obvious cases" — but the fetched page did not expose concrete per-domain
parameter tables (e.g. blocksworld block-count, sokoban grid-size/box-count), so those specifics
are **not verified** here and are not stated as numbers. General claim from a separate search
synthesis (LEAD, not independently fetched): instance generators typically expose
object-count-style parameters (blocks, packages, trucks, grid size) and sometimes a random-walk
length; treat as plausible but unconfirmed.

---

## 4. Prior transfer claims

### 4.1 SCAN (Lake & Baroni, "Generalization without Systematicity," ICML 2018, arXiv:1711.00350) — VERIFIED, full text read directly from PDF (pp.4-8) — **the strongest negative result found in this vein**

Task: simple compositional navigation-command → action-sequence translation (seq2seq RNNs).

- Experiment 1 (random split, sparse coverage): with 1% of commands for training (~210 examples),
  ~5% test accuracy; 2% coverage → ~54% correct; 4% coverage → ~93% correct; main random split
  (80% train, >16,700 examples) → near-ceiling. (Fig.3, read directly.)
- Experiment 2 (generalize to LONGER action sequences than seen in training, train ≤22 actions /
  test 24-48 actions): overall-best model achieves **13.8%** accuracy; best single configuration
  (GRU + attention, 1 layer, 50 hidden units, dropout 0.5) achieves **20.8%**. With an oracle that
  fixes only the early-termination decoding error: overall-best improves 13.8%→23.6%,
  top-performing model 20.8%→60.2% — i.e. even removing the trivial decoding failure mode leaves
  large gaps. Accuracy by target length (top figure) ranges **95.76% at 24 actions down to 22.8%
  at 48 actions**. (All exact numbers, quoted/read directly, Experiment 2 section + Fig.4.)
- Experiment 3 (hold out ONE primitive, "jump," seen only in isolation during training, must be
  used compositionally — e.g. "jump twice" — at test time): **overall-best model: 0.08% accuracy;
  best single configuration (LSTM+attention, 1 layer, 100 hidden, dropout 0.1): 1.2% accuracy.**
  Contrast condition, same experiment structure but holding out "turn left" instead of "jump":
  **overall-best model: 90.0%; best single configuration (GRU+attention, 1 layer, 100 hidden,
  dropout 0.1): 90.3%.** The paper's own diagnosis (quoted): "jump" only occurs with this
  primitive command in training context so the model does not generalize from it, whereas "turn
  left"'s underlying action (LTURN) appears embedded inside many other training action sequences,
  giving the model indirect exposure — i.e. the negative result is specifically about primitives
  seen ONLY in isolation, not about the compositional operation per se.
- Follow-up (Fig.5, exact numbers, read directly): giving the network a few *composed* "jump"
  examples during training (rather than zero) largely recovers the ability: 8 distinct composed
  "jump" examples → 38.3% held-out accuracy; 16 examples → 77.8%; 32 examples → 88.4%. This is a
  precise, quantified "how much composed evidence is needed before compositional generalization
  emerges" curve — directly relevant to calibrating `n(k)` in family 1.1 above.
- Experiment 4 (small-scale MT proof-of-concept, English→French): a new word ("daxy") inserted
  1,000 times in one fixed sentence frame during training; at test time, correctly translated in
  only **1 of 8** novel grammatical constructions, versus **8 of 8** for a control word ("tired")
  that had appeared in 80 distinct constructions during training. (Exact counts, read directly.)
- Paper's own conclusion (quoted): "the same networks fail spectacularly when the link between
  training and testing data is dependent on the ability to extract systematic rules." This is
  explicitly framed by the authors as a still-open problem for "neural networks capable of
  systematic compositionality," not resolved within the paper.

### 4.2 COGS (Kim & Linzen, EMNLP 2020, arXiv:2010.05465) — VERIFIED via abstract (fetch-tool text extraction of arXiv HTML abstract page; body not independently read)

Quoted numbers from the abstract: in-distribution test accuracy "near-perfect (96-99%)" for
Transformers/LSTMs; generalization-split accuracy "substantially lower (16-35%)"; "high
sensitivity to random seed (±6-8%)". 24,155 training examples (from search synthesis, treated as
LEAD unless corroborated — not independently re-verified against the PDF body).

### 4.3 PCFG SET / "Compositionality Decomposed" (Hupkes et al. 2020) — VERIFIED for task design (Section 2.3), **not verified for outcome numbers**

The five-test framework is a SOURCE TAXONOMY (quarantined, Section 8). The related-work section
(read directly, pp.4-6) summarizes prior compositionality results it is positioned against: SCAN
("sequence-to-sequence recurrent networks are still not systematic" — the authors' own paraphrase
of Lake & Baroni); lookup-table composition tests (Liška et al. 2018: "out of many models trained
with different initialisations only a very small fraction exhibits compositional behaviour, while
the vast majority does not" — quoted, a second, independent negative transfer result, though this
specific paper (Liška et al.) was only encountered as a citation inside Hupkes et al., not
independently fetched — **treat as LEAD**); logical-inference/entailment composition (Bowman et
al. 2015 — positive result for recursive nets, extended positively by later work per this
paper's summary). The paper's own quantitative results (its Results section) were **not fetched**
— no PCFG SET accuracy numbers are claimed here.

### 4.4 LIME (Wu, Rabe, Li, Ba, Grosse, Szegedy, ICML 2021, arXiv:2101.06223) — VERIFIED via abstract only

Three synthetic pretraining tasks are explicitly framed around Peirce's deduction/induction/
abduction triad and are "designed to be synthetic and devoid of mathematical knowledge to ensure
that only the fundamental reasoning biases can be learned." Quoted claim: "Models trained with
LIME significantly outperform vanilla transformers on four very different large mathematical
reasoning benchmarks," at "only a small fraction of the computation cost of the typical downstream
task." This is a **positive** transfer claim. **Not verified**: the concrete task definitions
(what deduction/induction/abduction look like operationally), the four downstream benchmark names,
and the actual improvement numbers — none were located in the abstract; PDF body not fetched.
Flagged rather than guessed.

### 4.5 CLUTRR (Sinha, Sodhani, Dong, Pineau, Hamilton, EMNLP 2019, arXiv:1908.06177) — VERIFIED via abstract only

Task: infer kinship relations from short stories by composing logical rules over a relation graph
("the parent of a parent is a grandparent" style composition). Abstract, quoted/paraphrased: tests
"systematic generalization... by evaluating on held-out combinations of logical rules" and
robustness "by adding curated noise facts." Reports "a substantial performance gap between
state-of-the-art NLU models (e.g., BERT and MAC) and a graph neural network model working with
symbolic inputs," the latter generalizing better. **Not verified**: the exact `k` (hop-count)
values used for train vs. held-out test, and the exact accuracy-gap numbers — not present in the
fetched abstract; a WebSearch-only synthesis (not independently confirmed) suggested training on
2-4 hops and testing to 10 hops — **treat this specific figure as LEAD, not verified**, since it
was not read directly from the paper.

### 4.6 Physics of Language Models, Part 3.1: Knowledge Storage and Extraction (Allen-Zhu & Li, arXiv:2309.14316) — VERIFIED via abstract (fetch-tool paraphrase of arXiv HTML abstract; hedged below)

Controlled synthetic "biography dataset" pretraining setup. Per the fetch tool's rendering of the
abstract (arXiv HTML abstract pages have been reliable elsewhere in this session, e.g. DreamCoder,
but this specific number was not cross-checked against the PDF body): knowledge that is
memorized without sufficient training-data augmentation (paraphrasing, sentence-shuffling,
translation) becomes **unextractable via question-answering, reportedly 0% accuracy, even after
instruction fine-tuning**; sufficient augmentation restores reliable extraction. This is a
**negative** transfer/generalization result (pretraining exposure to a fact is not sufficient for
that fact to be extractable; the FORM of exposure matters) that is directly relevant to any design
choice about how much encoding variation (`E`) to require per episode. Flagged as abstract-level,
not independently confirmed against primary text — the "0%" figure should be treated with
appropriate caution relative to the DreamCoder/Rintanen/Tomita/PCFG-SET numbers above, which were
read directly.

### 4.7 RASP / Tracr (Weiss, Goldberg & Yahav, "Thinking Like Transformers," ICML 2021, arXiv:2106.06981; Lindner et al., "Tracr," arXiv:2301.05062) — VERIFIED via abstract-level search synthesis only, not independently fetched-and-read

RASP is a programming language whose primitives (`select`, `aggregate`, sequence-operators)
correspond directly to transformer attention/feed-forward computation; "analyzing a RASP program
implies a maximum number of heads and layers necessary to encode a task in a transformer" (quoted
from the abstract as rendered by the fetch tool). Tracr compiles RASP programs directly into
transformer weights, producing networks with known ground-truth internal structure. This is not a
transfer-claim paper in the SCAN/COGS/CLUTRR sense — it is relevant to this vein as a possible
apparatus for constructing `f` (the oracle) or for verifying that a `structure-walk-query` teacher
policy is actually realizable by a transformer of bounded depth, rather than as a source of a
transfer finding. No prior transfer claim is asserted from this source.

---

## 5. Rejections (with repairs)

| Candidate | Verdict | Reason | Repair |
|---|---|---|---|
| "Find the MDL-optimal / most-reusable abstraction" (DreamCoder's literal objective, naively copied) | **fail on A1** | Requires search over an astronomically large refactoring set — DreamCoder itself needed a bespoke version-space/e-graph structure and a bounded evaluation-step distance to make this tractable (10^14 raw refactorings → 10^6-node version space for one small example, verified). | Ask the solver to propose/verify A valid abstraction consistent with a `theta`-supplied ground truth, not the optimal one (applied directly in family 1.1). |
| "Given a flat plan, recover the SHORTEST/optimal subgoal decomposition" | **fail on A1** | Same optimality trap — plan/goal recognition under a "most compressed" criterion is a search/optimization problem in general. | Same repair pattern: bake the true macro-sequence into `theta`, ask for a valid consistent segmentation, not the shortest one (applied in family 1.3). |
| "Given a DFA-generated string, recover the sequence of grammar rules/states used" (brief's own suggested candidate, "given a derivation recover the grammar rule applied," taken literally with a DFA/regular grammar substrate) | **fail on A4 (redundant/trivial), repairable** | For a DETERMINISTIC finite automaton, the state sequence for an accepted string is unique and computable by simply re-running the automaton — an O(1)-per-symbol simulation. The "compose" direction (string from rules) and "decompose" direction (rules from string) are computationally symmetric; there is no asymmetry to exploit, and a structure-ignorant flat solver (just simulate) keeps up trivially with any composition depth. This also substantially overlaps with plain grammatical-inference/DFA-identification content already presumably covered elsewhere in the candidate set. | Move the substrate from a DFA to an ambiguous/non-deterministic derivation source (e.g. a small ambiguous CFG, where more than one rule sequence can derive the same string) so that "recover the rule sequence actually used" is no longer trivially recoverable by simulation and a genuine compose/decompose asymmetry reappears. Not built out further here — ILP/grammar-learning-adjacent territory brushes close to the sealed ILP boundary and was deliberately not pursued past this triage note. |

---

## 6. Primitives — reuse/new decision

**Reused** (folding this vein's needs into the existing list, source term in brackets where a
folded synonym was tempting but rejected):
- `bijection-invert` — family 1.2's decompose step (peeling one pipeline stage) is exactly this;
  no new "de-compose a single application" primitive needed.
- `modular-add` — direct substrate primitive for family 1.2 when `S = Z_n`.
- `structure-walk-query` — the natural `q*` teacher for both 1.1 and 1.2 (walking the known
  composition tree/pipeline to place a maximally-informative query). Folds what might have been
  tempting to call "compositional-probe-synthesis" — rejected as a synonym.
- `basis-probe` — applicable to 1.2 when `P0` has fixed algebraic structure (e.g. a small group),
  giving a fixed probe schedule rather than adaptive querying.
- `posterior-enumeration` / `belief-state-maintenance` / `hypothesis-elimination` /
  `constraint-propagation` — all directly applicable to the L3 target and to incremental
  segmentation-narrowing in family 1.3; DreamCoder's own version-space/e-graph mechanism (Section
  2.1) is itself best read as an efficient, compressed instance of exactly these four operating
  jointly over program-refactoring space, not evidence for a new "version-space" primitive.

**New — one proposed, argued explicitly:**

`abstraction-naming` (also known in the source literature as *anti-unification* — DreamCoder's
supplement cites "version space algebras" and "equivalence graphs" as its implementation
apparatus, and Stitch's own abstract describes "syntactic pattern matching of intermediate
abstractions" for the same operation; the general operation across this literature is standardly
called anti-unification/generalization).

Definition: given `n ≥ 2` concrete structures (expressions, derivations, plans) that share a
common substructure at possibly-different positions, produce a single generalized template with
the shared part factored out and bound to a fresh name, such that each original structure is
recoverable by substituting the template's free slot(s).

Argued not to fold into the existing eleven:
- Not `hypothesis-elimination` — that cuts a belief state against a single new observation; this
  operation instead compares MULTIPLE already-fully-observed concrete structures against each
  other to find their common template. Different arity, different information flow (not
  eliminating candidates from a maintained set; constructing a new named unit from scratch).
- Not `structure-walk-query` — that synthesizes a QUERY by walking an ALREADY-KNOWN structure;
  this operation instead produces a NEW REUSABLE STRUCTURE from multiple already-answered
  instances, and touches no oracle at all.
- Not `bijection-invert` — that undoes one layer of a single known composition; this operation
  generalizes across MULTIPLE instances to find a recurring pattern, which need not be invertible
  or even a bijection.
- Not `basis-probe`/`constraint-propagation`/`posterior-enumeration`/`belief-state-maintenance` —
  none of these produce a new named/reusable unit; they narrow or enumerate a belief state over
  already-fixed hypotheses.

This is the one operation family 1.1 (the DreamCoder-grounded family) cannot be built without, and
it is the operation the brief anticipated ("an abstraction/naming operation... but only if it does
not fold into the above"). Everything else in this vein's three families reuses existing
primitives cleanly.

Running tally: previous veins +3, then 0, then +1 → 4 total before this vein. This vein: **+1**
(`abstraction-naming`), running total 5.

---

## 7. Sources

**VERIFIED** (fetched and the actual text/figures read directly, not only fetch-tool paraphrase):
- Ellis, Wong, Nye, Sablé-Meyer, Cary, Morales, Hewitt, Solar-Lezama, Tenenbaum, "DreamCoder:
  Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning,"
  arXiv:2006.08381 — full main text (pp.1-20) read directly via PDF page rendering.
- Wang, Zhang, Ororbia II, Xing, Liu, Giles, "An Empirical Evaluation of Rule Extraction from
  Recurrent Neural Networks," arXiv:1709.10380 — pp.1-8 read directly, including Table 1 (Tomita
  grammar definitions).
- Abbadingo One competition site: https://abbadingo.cs.nuim.ie/ (homepage), /QA.html,
  /data-sets.html — HTML text fetched and read.
- Rintanen, "Phase Transitions in Classical Planning: An Experimental Study," KR 2004 — pp.710-714
  read directly via PDF page rendering (this is the source that exposed a fetch-tool
  hallucination risk — see note in Section 3.3).
- Hupkes, Dankers, Mul, Bruni, "Compositionality Decomposed: How do Neural Networks Generalise?"
  arXiv:1908.08351 — pp.3-14 read directly via PDF page rendering, including the PCFG SET grammar
  and interpretation-function tables (Figs.2-3).
- Lake & Baroni, "Generalization without Systematicity," arXiv:1711.00350 — pp.4-8 read directly
  via PDF page rendering, all Experiment 2/3/4 numbers.
- AI-Planning/pddl-generators GitHub repository — HTML page fetched and read (domain list only;
  per-domain parameter tables not exposed by the fetched page).
- Rieffel, Venturelli, Do, Hen, Frank, "Parametrized Families of Hard Planning Problems from Phase
  Transitions," AAAI 2014 — **fetched by mistake; excluded from use, see Section 9.**

**LEAD** (seen only via WebSearch synthesis or an abstract-only WebFetch; body not independently
read):
- Bowers et al., "Top-Down Synthesis for Library Learning" (Stitch), arXiv:2211.16605 — abstract
  fetched directly (speed/memory numbers verified from the abstract itself); body not fetched, so
  domain/primitive-inventory claims about it are not made.
- Kim & Linzen, "COGS," arXiv:2010.05465 — abstract fetched directly (96-99%/16-35%/±6-8% numbers
  verified from the abstract); body not fetched.
- Wu, Rabe, Li, Ba, Grosse, Szegedy, "LIME," arXiv:2101.06223 — abstract fetched directly; task
  operationalization and downstream benchmark numbers NOT located, explicitly flagged as such
  rather than guessed.
- Sinha, Sodhani, Dong, Pineau, Hamilton, "CLUTRR," arXiv:1908.06177 — abstract fetched directly;
  the specific hop-count (`k`) train/test values are LEAD-only (WebSearch synthesis), not
  confirmed against the paper.
- Allen-Zhu & Li, "Physics of Language Models: Part 3.1," arXiv:2309.14316 — abstract-level
  fetch-tool paraphrase only; the "0% accuracy" figure is flagged as lower-confidence than the
  directly-read numbers elsewhere in this report.
- Weiss, Goldberg, Yahav, "Thinking Like Transformers" (RASP), arXiv:2106.06981, and Lindner et
  al., "Tracr," arXiv:2301.05062 — abstract-level search synthesis only.
- Liška et al. 2018 (lookup-table compositionality, negative result) — encountered only as a
  citation inside Hupkes et al.'s related-work section, not independently fetched.
- Bylander 1996, "Learning Boolean formulas / phase transitions in planning" — encountered only as
  a citation inside Rintanen 2004, not independently fetched.
- Generic planning-instance-generator parameter claims (object-count / random-walk-length knobs
  across blocksworld/sokoban/gripper/logistics) — WebSearch-synthesis only; the
  "Automatic Instance Generation for Classical Planning" and "Exploring Instance Generation for
  Automated Planning" papers were targeted but the fetch tool could not extract readable text
  from the first (compressed PDF) and the second 404'd — **not verified, not used for any
  numeric claim in this report.**

**Not reached / explicitly could not verify:**
- DreamCoder supplement (S1.1, full primitive tables per domain; S4.5, version-space construction
  detail) — main-text-only PDF page ranges were read; the supplement was not separately fetched,
  so the complete enumerated primitive lists (beyond what appears in the main-text figures) remain
  unconfirmed.
- PCFG SET's own Results section (per-test accuracy numbers) — not fetched (only Sections 1-4 read).
- Exact numeric IPC generator parameters per domain (blocksworld block-count ranges, sokoban
  grid/box-count ranges) — not located in any successfully-fetched source.

---

## 8. Source's taxonomy (quarantined — not adopted as our capability decomposition)

- **DreamCoder's own dual-process split**: "explicit declarative knowledge" (the learned library,
  `L`) vs. "implicit procedural knowledge" (the neural recognition model) — explicitly presented
  by the authors as inspired by dual-process cognitive-science models of human expertise (Chi,
  Feltovich, Glaser 1981; Chi, Glaser, Farr 1988, both cited by DreamCoder). This is the source's
  framework for WHY its two learning mechanisms exist, not a measured capability split.
- **DreamCoder's eight "domains"** (list processing, text editing, LOGO graphics, regex, block
  towers, symbolic regression, recursive programming, physical laws) — a task-type taxonomy
  chosen by the authors to showcase breadth, not a difficulty- or mechanism-based grouping.
- **Hupkes et al.'s five compositionality tests** (systematicity, productivity, substitutivity,
  localism, overgeneralisation) — explicitly the source's own attempt to operationalize
  linguistic/philosophical theories of compositionality (Partee 1995 quoted directly on this).
  Directly on point for Hazard 2 ("leave the ontology, take the apparatus") — the PCFG SET *task*
  (Section 2.3) is reused; the five-test grouping is not adopted here as how we would carve up
  compositional capability.
- **Bylander/Rintanen's Model A/B/C classification** of randomly-sampled STRIPS instance spaces —
  a sampling-methodology taxonomy for generating hard planning instances, not a capability
  taxonomy; kept separate from any claim about what makes OUR planning-flavored family (1.3) hard.
- **Tomita's index-order difficulty convention** (grammar 1 "easiest" through grammar 7
  "hardest") — the field's own convention; Section 3.1 already flags that the mechanism proposed
  by the secondary literature (class-balance) is a different, more defensible parametrization than
  the raw index.

---

## 9. Sealed encounters

- **Inductive logic programming lineage (Metagol, Popper)** — never searched, never opened.
  Explicitly avoided throughout, including when DreamCoder's own related-work citations pointed
  toward program-induction-adjacent territory; no ILP-specific source was fetched.
- **"Parametrized Families of Hard Planning Problems from Phase Transitions"** (Rieffel,
  Venturelli, Do, Hen, Frank, AAAI 2014) — **opened in error.** It was reached while chasing
  planning-phase-transition evidence; its abstract/intro (visible before realizing the scope)
  mentioned only "navigation and scheduling," and it was fetched and partially read (pp.1-3) before
  it became clear that its second problem family is explicitly "Planning problems from Graph
  Coloring (GC)," built directly on graph-coloring phase transitions via randomly generated
  Erdős–Rényi graphs — squarely inside the sealed "graph-colouring phase transitions" topic. Per
  instructions, this counts as contamination even though it was used only for background
  orientation. **No content from this paper (including its non-graph-coloring Hamiltonian-path
  family) is used anywhere in this report** — Section 3.3's planning non-monotonicity evidence
  relies entirely on Rintanen 2004, an independently-reached, unsealed source, instead.
- No other sealed-list titles were encountered in search results during this review (no
  quasigroup/Latin-square, animal cognition, psychometrics, clinical reasoning, active learning,
  SRE/SQL/Erlang, Euclidean/analysis, or ARC-AGI-3 titles surfaced).
