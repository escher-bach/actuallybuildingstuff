# Vein 2.3 — Hidden-rule paradigms from psychology and meta-RL

Scope: Bruner/Goodnow/Austin (1956), Harlow (1949), Shepard/Hovland/Jenkins (1961), reversal
learning / WCST / Daw two-step / bandits / Miconi's procedural meta-RL parametrization,
probability matching / partial reinforcement. Held-out boundaries (developmental/infant
paradigms, non-primate animal cognition, verbal/psychometric item generation, clinical
reasoning, ILP/pool-based active learning/adaptive trial design, Latin-square/graph-coloring,
SRE/SQL/Erlang/geometry, ARC-AGI-3) were not read — see §9.

---

## 1. Plants

### 1a. Prerequisite pair — Shepard, Hovland & Jenkins (1961) Type I vs Type VI

**Claim:** Type I (single relevant dimension) is reliably learned faster/before Type VI
(three-way parity, no dimensional simplification possible).

**Evidence strength: replicated, but a difficulty-precedence claim, not a logical-containment
claim.** SHJ 1961 (*Psychological Monographs* 75, 1–42) measured the ordering
`I < II < III = IV = V < VI` on human subjects sorting 8 stimuli (3 binary dimensions: e.g.
size, color, shape) into two 4-vs-4 categories. Nosofsky, Gluck, Palmeri & McKinley (1994,
*Memory & Cognition* 22, 352–369) is the standard replication: per a search-indexed abstract,
"the main results mirrored those of Shepard et al., with the ordering of task difficulty being
the same as in the original study," collecting block-by-block learning curves and fitting four
computational models (ALCOVE best fit). **I could not read the Nosofsky primary text
(paywalled) — the replication claim is LEAD-level, not independently checked against numbers.**
Feldman (2000, *Nature* 407) proposes Boolean complexity (shortest equivalent propositional
formula) as the general predictor; I could not access this paper directly (403) — its exact
per-type complexity values are LEAD-level, taken from a tertiary arXiv source (§7) that itself
did not quote them fully.

**Important nuance — do not oversell as "prerequisite":** nothing in SHJ shows Type VI's rule
*contains* Type I's rule as a sub-structure. The ordering is empirical acquisition-speed
precedence (an organism reliably masters Type I before Type VI), not logical nesting. That is
still usable as a curriculum-ordering plant (A obviously learned before B, externally attested,
replicated at least once), but the word "required" in the deliverable spec should be read as
"empirically precedes," not "logically presupposes."

**Type definitions (canonical, from secondary/tertiary sources, see §7):** Type I = classify by
one dimension alone. Type II = XOR of two dimensions (third irrelevant) — logically a
parity/mod-2 relation. Types III–V = "intermediate," each expressible with reference to roughly
2.5 dimensions with exception structure; empirically indistinguishable from each other in
difficulty despite being logically distinct functions. Type VI = the 3-way parity/biconditional
— no dimension can be dropped, no simplification, hardest.

### 1b. Near-duplicate pair — Bruner's conjunctive-concept "selection paradigm" + conservative
focusing vs. exact-learning of monotone conjunctions from membership queries (Angluin 1988 /
computational learning theory)

**Hypothesis was: check, don't assume. Verdict: structurally very likely the same object,
independently arrived at — but I was not able to read either primary source in full, so
evidence is MODERATE, not VERIFIED.**

What I confirmed:
- Bruner, Goodnow & Austin's **selection paradigm**: the learner picks a stimulus card from an
  array and is told only "positive" or "negative" for that specific card — this is, verbatim, a
  membership-query oracle over `Theta x X -> {+,-}`.
- Their **conservative focusing** strategy (secondary-source quote, ScienceDirect Topics
  "Cognitive Strain," via search snippet, not independently fetched — LEAD): "applicable to
  learning common-element (conjunctive) concepts under the selection procedure, both maximizes
  information gain and holds cognitive strain to a low level. It begins with a positive instance
  of the target concept... selecting instances that differ in one and only one feature from the
  initial positive instance, and noting whether the chosen instance is positive or negative."
- The standard computational-learning-theory algorithm for **exact-learning a monotone
  conjunction from membership queries** (Angluin 1988, *Machine Learning* 2:319–342; presented
  as a worked textbook example in Kearns & Vazirani 1994 ch. 1, per search-indexed description,
  not independently read — LEAD): start from a known positive instance, flip one attribute at a
  time, query the Hamming-neighbor; an attribute whose flip turns the label negative is a
  relevant literal of the conjunction, one whose flip leaves it positive is irrelevant.

These are the same algorithm: fix a positive anchor, probe Hamming-distance-1 neighbors one
attribute at a time, classify each attribute as relevant/irrelevant by whether the flip changes
the label. Bruner published this as an empirically-observed/prescribed *human* strategy in 1956;
Angluin formalized query types (including membership queries) for exact concept learning in
1988; the specific "flip one bit from a known positive" algorithm for monotone conjunctions
predates Angluin's paper in the CoLT folklore (Valiant 1984 is the origin of PAC learning of
conjunctions, though from examples rather than queries).

**Caveats, stated honestly:**
- I searched explicitly for any existing publication connecting Bruner's conservative focusing
  to CoLT membership-query learning and found none — this connection appears to be ours to
  make, not an attested one. It strengthens the "independently reinvented" framing but weakens
  the citability: there is no third party to point to.
- I did not read either A Study of Thinking chs. 4–5 in full (only the front matter/intro via
  archive.org — see §7) or Angluin 1988 / Kearns & Vazirani ch. 1 in full (only abstracts/search
  snippets). **Before treating this as load-bearing, someone should read both primary texts in
  full to confirm the algorithms are identical, not merely similar.**
- Bruner's array was `4` attributes x `3` values (81 cards: number of shapes, number of borders,
  shape kind, color — secondary-source confirmed, see §7), all values simultaneously relevant
  candidates; the classic CoLT setup is usually a Boolean hypercube (`v=2`). The two are
  isomorphic only if Bruner's "differs in one feature" is read as "differs in the value taken on
  one attribute," which is exactly how conservative focusing is described. This should still
  hold for `v>2` (attribute-efficient learning of conjunctions over larger alphabets is a known
  generalization) but I have not verified this generalization against a primary CoLT source.

**Verdict: best available near-duplicate plant in this vein, MODERATE evidence, flagged for a
primary-source read before being trusted as load-bearing.**

### 1c. Independent pair — SHJ Boolean-concept identification vs. probability matching /
partial reinforcement

**Evidence strength: obvious by construction (no search needed to establish independence, but
also no prior literature explicitly contrasts them — I found none doing so).**

SHJ-style concept identification: `Theta` = one of a small enumerable set of Boolean functions
over discrete symbols; oracle is deterministic; posterior is discrete and (with enough queries)
collapses to a point — this is an L1/L2 family. Probability matching: `Theta` = a hidden
continuous (or finely discretized) Bernoulli parameter `p`; oracle is *stochastic* even when
`theta` is fixed and fully known to the oracle; the posterior over `theta` never collapses to a
point within a finite episode — genuinely L3. Different `Theta` topology (finite/discrete vs.
continuous), different oracle behavior (deterministic label vs. noisy draw), different target
object (a point rule vs. a distribution). These are two different literatures (categorization
research vs. operant-conditioning/reinforcement-schedule research) that, as far as I found,
never cite each other for this comparison — clean independence.

---

## 2. Candidate families

| Field | SHJ six-type Boolean concept | Bruner conjunctive/disjunctive concept (general) | Harlow learning-set (discrimination) | WCST / reversal learning | Daw two-step task | Bandit (Miconi parametrization) | Probability matching / partial reinforcement |
|---|---|---|---|---|---|---|---|
| `Theta` | one of 6 canonical Boolean fns (equivalence classes) over 3 binary dims, or general d-dim generalization | a conjunction or disjunction of literals over `d` attributes x `v` values | which of N stimuli is "positive" this episode | which of `d` dimensions (WCST) or which option (reversal) is currently "correct"; changes at a changepoint | transition structure (fixed) + terminal reward-probabilities (drift per trial in classic version) | per-arm reward probability vector | a single hidden Bernoulli parameter `p` (or vector) |
| `k` enters via | number of dimensions `d` (scale-up beyond 3), or Boolean-complexity class of the target fn | `d`, `v` (attribute/value counts) | number of episodes (curriculum length), stimulus-pool size/novelty | number of dims/values, run-length-to-criterion, reversals per episode | number of stages/branches, drift variance, episode length | number of arms, horizon length | precision of `p` (discretization granularity), number of trials |
| `X` (query space) | one of `2^d` stimuli | one of `v^d` stimuli | choose one of N stimuli | classify presented card into one of the fixed key categories | choose first-stage action, then second-stage action | choose an arm | predict next outcome / choose an option |
| oracle computes | truth-table lookup for `theta` at the queried stimulus | membership test: does the instance satisfy the conjunction/disjunction | reward iff chosen == theta's positive stimulus | reward iff classification matches current active dimension's value | reward iff terminal state matches; transition sampled from fixed/drifting probs | Bernoulli draw with the queried arm's probability | Bernoulli draw with the hidden `p` |
| encoding 1 | attribute-value tuples (abstract symbols) | card array (Bruner board): number/border/shape/color as symbol IDs | arbitrary symbol pairs | abstract dimension/value tokens | abstract state-graph tokens (not spaceship/alien imagery) | labeled arms/buttons | binary outcome tokens |
| encoding 2 | natural-language object descriptions | natural-language "object has features..." descriptions | described object pairs in prose | textual rule-description-free card descriptions | textual state-transition description | slot-machine narrative | coin/urn narrative |
| posterior enumerable (L3-ready)? | yes — finite hypothesis space, exactly the ingredient L3 needs if made non-identifiable (e.g., very few queries allowed) | yes, same reasoning | not naturally — designed to BE identifiable every episode (that's the point) | yes over the small dimension set, but see §3 for the changepoint issue | no — reward probabilities are continuous and drift; would need discretization | yes if `theta` discretized; continuous otherwise | yes if `p` discretized to a grid; this is the natural L3 case |
| `q*` | conservative focusing (walk Hamming-neighbors from a known positive) | conservative focusing / focus gambling (riskier variant) | trivial: teacher already knows theta, always picks correctly | teacher always classifies correctly; no exploration needed | teacher always picks the correct branch | teacher always pulls the best arm | teacher always predicts the majority outcome (`majority-predict`) |

Full family write-ups follow the admissibility triage in §5 (rejections/repairs) rather than
being duplicated here — see per-family verdicts there for what would need to change to ship
each one.

---

## 3. Reveal structures — the L1/L2/L3 apparatus catalogue

| Paradigm | Apparatus | Closest level | Fit |
|---|---|---|---|
| Bruner **reception** paradigm | Experimenter controls the sequence of labeled instances; subject cannot choose what to see next | **L1** | clean fit |
| Bruner **selection** paradigm | Subject chooses which instance to submit for labeling; oracle answers only what was asked | **L2** | clean fit — this is close to the primary source of the L1/L2 distinction as stated in the brief |
| Bruner strategy catalogue (conservative focusing, focus gambling, successive scanning, simultaneous scanning) | Candidate `q*`/solver policies for L2, differing in cognitive-strain vs. information-gain tradeoff | **A7 source material** | direct fit — see §6 primitive mapping |
| SHJ six-type task | Classic administration is reception-style: fixed/randomized-order presentation of the 8 stimuli, repeated, with feedback after each guess | **L1** | clean fit |
| Harlow learning-set | Each single discrimination problem is L1 (stimulus pair shown, choice made, feedback given); the *real* apparatus is many L1 episodes back-to-back with freshly-sampled `theta`, scoring the **within-episode learning-rate curve across episodes** | **L1, repeated as a curriculum** | fits, but the interesting object is a *training-metric protocol*, not a single-episode level — this is literally the "primary training metric" the brief names, independently validated by 1949 apparatus |
| WCST / single or serial reversal learning | Rule is hidden, fixed for a run, then **silently changes at a changepoint** (criterion-triggered in the classic version) — the belief state must be reset, not just refined | **Does not fit cleanly** | see below |
| Daw two-step task | Transition probabilities typically fixed; **terminal reward probabilities random-walk continuously, every trial**, in the classic version — `theta` is never a single sampled-once value, it is a nonstationary stochastic process | **Does not fit cleanly** | see below |
| Bandit tasks | Choosing an arm is simultaneously the informative query AND the scored/rewarded action — no free "test, then separately get scored" split | **Does not fit cleanly** | see below |
| Probability matching / partial reinforcement | `Theta` = fixed hidden Bernoulli parameter; oracle is stochastic even given full knowledge of `theta`; no finite number of queries collapses the posterior to a point | **L3** | clean fit — the strongest clean L3 exemplar found in this vein |

**Findings that do not fit the four levels (flagged per instructions, not smoothed over):**

1. **Piecewise-stationary reception ("L1 with silent regime changes").** WCST and reversal
   learning are not L1 (theta is not fixed for the episode) and not L3 (theta *is* identifiable,
   just not stably so — it resets). The belief state must be capable of **non-monotonic reset**,
   which conflicts with the given primitive `belief-state-maintenance`'s definition
   ("monotonically-refined"). See §6 for the proposed primitive.

2. **Continuously drifting theta.** Daw's classic two-step task does not sample `theta` once per
   episode at all — reward probabilities perform a bounded random walk every trial. This breaks
   the `T = (Theta, P_Theta, X, f, E, rho, k)` formalism's implicit assumption of one `theta`
   sampled per episode. A generation-admissible version would need to either (a) discretize into
   fixed within-episode regimes (turning it into case 1 above) or (b) treat the *drift process
   itself* as the object being estimated (a very different, harder L3-like target: not "what is
   theta" but "what is theta *right now*, continuously").

3. **Entangled query/reward (exploration-exploitation coupling).** In L2 as specified, choosing
   a query is free — a bad choice just yields an uninformative answer, no cost beyond the wasted
   turn. In bandits, WCST, and reversal learning, every query **is** the scored action: there is
   no "test now, get scored later" separation. This is a genuinely different reveal structure
   from Bruner's selection paradigm, even though both involve the learner choosing `x in X`. It
   is the single clearest thing this vein contributes that isn't already implied by L0–L3: a
   **cost/entanglement axis**, orthogonal to the identifiability axis the four levels are built
   on.

---

## 4. Difficulty orderings and parametrizations — what was measured, on whom, how well it replicates

| Result | Measured on | What exactly was measured | Replication status |
|---|---|---|---|
| SHJ ordering `I<II<III=IV=V<VI` | Human adults, original 1961 study | Trials/errors to a learning criterion across the 6 types, 8-stimulus reception task | Replicated in Nosofsky et al. 1994 per search-indexed abstract (ordering preserved; richer block-by-block data collected) — **not independently verified by me against primary numbers, paywalled**. Feldman (2000) proposes Boolean complexity as a general explanatory predictor beyond the 6 canonical types — **not independently verified, 403 on primary** |
| Harlow learning-set curve | Rhesus macaques, 1949, 344 object-pair discrimination problems | Percent correct at trial 2 (of ~6 trials/problem) as a function of cumulative problem number; rises from ~50% (chance, early problems) toward ~90%+ (near one-trial learning, late problems) | This is the founding result of the "learning-to-learn" literature; a 2024 replication in wolves and dogs exists (PMC11628440, VERIFIED-fetched, confirms task structure but reports species differences — **not detailed here per the non-primate-animal-cognition seal**, only used to corroborate Harlow's original task structure) |
| Boolean-complexity account (Feldman 2000) | Reanalysis of concept-learning data, not a new experiment per the sources I could reach | Predicts subjective difficulty is proportional to the length of the shortest logically-equivalent propositional formula | Cited approvingly by later arXiv-hosted symbolic-concept-learning papers (§7); I could not confirm exact per-type complexity numbers |
| WCST perseveration/switch-cost measures | Clinical and healthy populations, from Berg 1948 onward | Errors to first category, perseverative errors, non-perseverative errors, categories completed | This is squarely descriptive/clinical-outcome territory; I deliberately did not pursue this further given the adjacency to the held-out "clinical reasoning" boundary — apparatus only was extracted (§2) |
| Probability matching frequency | Human adults, Grant/Hake/Hornseth 1951 onward | Choice-proportion for the majority option vs. its true base rate (e.g., "green" chosen ~70% of trials when green pays 70%) — the finding is that choice proportions *track* the base rate rather than the theoretically optimal all-or-none maximizing strategy | Described as one of the most-replicated phenomena in the judgment-and-decision-making literature (multiple sources, not independently re-verified numerically by me) |

**Hazard-1 note applied throughout:** every ordering above is a *human* (or primate) difficulty
ordering. None of it is assumed to transfer to a sequence model's difficulty ordering. It is
recorded here as the externally-attested prior to be tested against, per the brief's own
instruction.

---

## 5. Rejections (with repairs)

| Family | Fails | Repair |
|---|---|---|
| SHJ six-type task, as published (3 dims, 8 stimuli) | **A4**: trivially memorizable (8-item truth table) | Scale to `d` binary dimensions, `d` as knob `k`; use Feldman-style Boolean-complexity classes to define graded "type" families beyond the canonical 6, so hypothesis space and truth-table size both grow with `k` |
| SHJ / Bruner conjunctive concept, as published | **A2**: canonical stimuli use size/color/shape/border — real-world dimensions with human salience priors that a pretrained sequence model may already encode (e.g., "shape is usually more diagnostic than color" folk priors) | Anonymize to abstract symbol-dimensions (`P/Q/R/S` with values `0/1/2`), never natural perceptual categories |
| Bruner conjunctive/disjunctive concept, as published (`d=4, v=3`, 81 cards) | **A4**: hypothesis space of size roughly `(v+1)^d = 256` is brute-forceable by an automated solver holding the full instance table in memory | Scale `d, v` up; the gap the brief wants (structure-ignoring solver cost grows faster than `k` than `f`'s size) is available for free here because conservative focusing needs `O(d)` queries while brute enumeration needs `O(v^d)` — just needs `d, v` large enough to matter |
| Harlow learning-set, as a standalone single-episode family | **A4 not meaningful**: a single discrimination problem is a 1-bit hidden parameter, trivial by design — that is the *point* of the paradigm, not a flaw | Not a per-episode family at all — repair is conceptual: treat Harlow's apparatus as a **curriculum wrapper** (repeat any other admissible family across many episodes with fresh `theta`, score the within-episode learning-rate curve across the curriculum) rather than as content needing its own hardness |
| WCST, as classically administered | **A1**: the changepoint (rule switch) is triggered by the *subject's own* run of correct answers, which is not decidable at generation time without simulating the (unknown) solver | Decouple: pre-sample a fixed or randomly-drawn switch schedule (e.g., switch every `T` trials, `T` itself sampled) independent of solver performance — preserves the "silent regime change" phenomenon while restoring backward generation |
| WCST, as classically administered | **A2**: color/shape/number carry real salience asymmetries (the literature itself reports color-to-shape shifts differ in difficulty from shape-to-color) | Anonymize dimension identities to symbol tokens |
| WCST, as classically administered | **A4**: hypothesis space is 3 dimensions — trivial | Scale dimensions/values up; combine with SHJ-style graded Boolean-complexity rules for a richer target space |
| Daw two-step task, as classically administered | **A2**: spaceship/alien/planet imagery in the standard human version | Re-skin to abstract state-graph tokens |
| Daw two-step task, as classically administered | Formalism mismatch, not a strict A-failure: `theta` is not sampled once per episode (see §3, finding 2) | Either discretize drift into fixed within-episode regimes, or explicitly redefine the target as "track a moving parameter" rather than "identify a fixed one" — a genuinely different task type worth having on its own terms, not force-fit into L0–L3 |
| Bandit tasks / probability matching, at small arm-count / short horizon | **A4 marginal**: with 2 arms and few trials, frequency-counting is close to optimal and requires no real search | Increase arm count / add hierarchical structure linking arm probabilities (e.g., arms grouped by a shared latent category) to force genuine inference rather than counting |

---

## 6. Primitives

**Reused (with folded source terms):**

| Existing primitive | Source terms folded in |
|---|---|
| `hypothesis-elimination` | Bruner's **successive scanning** (serial, single-hypothesis-at-a-time version); SHJ rule identification generally |
| `belief-state-maintenance` | Bruner's **simultaneous scanning** (subject explicitly tracks the full surviving-hypothesis set) |
| `posterior-enumeration` | probability-matching / partial-reinforcement calibration target; SHJ posterior over the 6 (or `2^2^d`) candidate rules |
| `majority-predict` | the theoretically-correct point prediction in probability matching, against which the human "matching" (not maximizing) behavior is the documented deviation — worth keeping this contrast in the register: the paradigm's own finding is that humans do *not* use `majority-predict` even though it's optimal |
| `informative-query-selection` | Bruner's **conservative focusing** is close to a textbook instance of this primitive: score candidate queries (Hamming-neighbors of a known positive) by expected information gain, cost scales with belief-state size exactly as the primitive's definition states. **Focus gambling** is the same primitive under a higher risk tolerance (multi-attribute flips), not a different primitive. Bandit `q*` policies (UCB/Thompson-sampling-style, mentioned qualitatively re: Miconi framing) are also this primitive |
| `modular-add` | SHJ Type II (XOR of two dimensions) and Type VI (three-way parity/biconditional) are both literally mod-2 addition over a subset of dimensions — a clean, unexpected fold |
| `constraint-propagation` | conservative-focusing's attribute-by-attribute narrowing of the conjunction's literal set |

**New — proposed, with justification (this vein's answer is not "zero"):**

`belief-state-reset` — *detect that accumulated evidence contradicts a previously-converged
hypothesis and replace (not monotonically refine) the belief state.* Two known variants: a
**discrete** trigger (WCST, reversal learning: a run of errors after prior high confidence signals
a rule change, belief state must be reset toward a fresh prior) and a **continuous** variant (Daw
two-step's drifting reward probabilities: belief must decay/track rather than converge and hold).
Justification for treating this as genuinely new rather than a repair-tag on
`belief-state-maintenance`: the given primitive is explicitly defined as carrying a
*monotonically-refined* representation — WCST/reversal/Daw structurally require the opposite
(non-monotonic replacement), which the existing primitive's definition rules out by construction.
This is a **borderline call** — flagged honestly rather than silently folded in, and a reviewer
who prefers zero new primitives could reasonably argue for handling it as
"`hypothesis-elimination` applied at the rule-set level, restarted" instead of a new primitive.
I lean toward proposing it because the discrete/continuous pairing (WCST + Daw) shows the same
gap twice, independently, in two different paradigms in this vein — that repetition is what
pushed it from "maybe" to "worth proposing."

No other new primitives were needed. This vein's other apparatus (successive/simultaneous
scanning, conservative/gambling focusing, SHJ rule types, Harlow's win-stay/lose-shift,
probability matching's calibration target) mapped cleanly onto the existing eleven.

---

## 7. Sources

**VERIFIED** (content actually fetched and read, at least in part):
- Miconi, T. "Procedural generation of meta-reinforcement learning tasks." arXiv:2302.05583.
  Fetched via ar5iv HTML; got direct quotes of the POMDP formalism, reward-rule syntax, and the
  bandit/Harlow/T-maze/Daw-two-step/Dark-Room special cases.
- "Distilling Symbolic Priors for Concept Learning into Neural Networks." arXiv:2402.07035.
  Fetched via ar5iv; got quoted SHJ type descriptions and Rational-Rules-model error rates —
  **the specific bit-pattern table it returned did not internally cross-check cleanly against a
  canonical XOR construction when I hand-verified it; treat the exact 0/1 patterns it quoted as
  unreliable, but the qualitative type descriptions (Type I = single dimension, Type II = XOR,
  Type VI = no simplification) are corroborated elsewhere and used in this report.**
- "Complexity Measures and Concept Learning." arXiv:1406.7424. Fetched via ar5iv; got quoted
  canonical descriptions of Types I, II, VI and the "paradigm-specific order ≠ general order"
  finding re: Boolean complexity.
- Archive.org full text of Bruner, Goodnow & Austin, *A Study of Thinking* (1956),
  https://archive.org/stream/in.ernet.dli.2015.139127/2015.139127.A-Study-Of-Thinking_djvu.txt —
  fetched, but the returned excerpt covered only front matter/introduction, not chs. 4–6 where
  the strategy definitions and selection-paradigm experimental detail live. **Treat as partially
  read only.**
- PMC11628440 ("Going back to 'basics': Harlow's learning set task with wolves and dogs") —
  fetched, used only to corroborate Harlow's original 344-pair task structure and the
  trial-2-percent-correct learning-curve metric, not for its own (non-primate) findings.

**LEAD** (seen via search snippets / secondary or tertiary description only, not independently
fetched and read):
- Shepard, Hovland & Jenkins (1961), *Psychological Monographs* 75, 1–42 — original paper never
  directly read.
- Nosofsky, Gluck, Palmeri & McKinley (1994), *Memory & Cognition* 22:352–369 — paywalled
  (Springer 303 redirect blocked).
- Feldman, J. (2000), "Minimization of Boolean complexity in human concept learning," *Nature*
  407 — paywalled (403).
- Harlow, H.F. (1949), "The Formation of Learning Sets," *Psychological Review* 56:51–65 —
  original never directly read, only via secondary description.
- Bruner, Goodnow & Austin (1956) chs. 4–6 (reception/selection paradigm detail, four-strategy
  definitions, 81-card array composition) — reconstructed from search-snippet secondary sources
  (ScienceDirect Topics "Cognitive Strain," multiple tertiary summaries); primary text not
  reached for these chapters.
- Angluin, D. (1988), "Queries and Concept Learning," *Machine Learning* 2:319–342 — abstract/
  description only, not fetched in full.
- Kearns, M. & Vazirani, U. (1994), *An Introduction to Computational Learning Theory*, MIT
  Press, ch. 1 — description only via search snippets, not fetched.
- Berg, E.A. (1948), "A simple objective technique for measuring flexibility in thinking,"
  *Journal of General Psychology* 39:15–22 — citation only, WCST original not read.
- Daw, N.D., Gershman, S.J., Seymour, B., Dayan, P. & Dolan, R.J. (2011), "Model-based
  influences on humans' choices and striatal prediction errors," *Neuron* 69:1204–1215 —
  citation only, not read.
- Grant, D., Hake, H. & Hornseth, J. (1951), "Acquisition and extinction of a verbal conditioned
  response with differing percentages of reinforcement," *Journal of Experimental Psychology*
  42:1–5 — citation only, foundational probability-matching apparatus paper, not read.
- Vulkan, N. (2000), "An Economist's Perspective on Probability Matching," *Journal of Economic
  Surveys* 14:101–118 — citation and abstract only, not read; used for a clean statement of the
  binary-choice/hidden-base-rate apparatus.
- "Selection versus reception concept-attainment paradigms" (psycnet record 1975-22333-001) —
  found as a citation confirming the reception/selection terminology is an established named
  distinction in the literature independent of Bruner's own book; not read.

**Not reached at all:** WCST original administration manual/scoring detail beyond what's in
secondary sources; the wolves/dogs Harlow-replication paper's own (non-primate) results
(deliberately not extracted, see seal below).

---

## 8. Source's taxonomy (quarantined — not adopted, recorded for later scoring)

- **SHJ's own theory:** difficulty ordering explained by *selective attention to relevant
  dimensions* — Type I needs 1 dimension attended, Type II needs 2, III–V and VI need
  (functionally) all 3, differing in how attention can be distributed. This is SHJ's dimensional-
  salience account, not our measurement.
- **Feldman's own theory:** difficulty is directly proportional to Boolean complexity (shortest
  equivalent logical formula), proposed as a *general* replacement for SHJ's dimension-count
  account. Also not adopted — a rival theory to be scored against our own measurement, not
  assumed.
- **Bruner et al.'s own theory:** concept-attainment strategies trade off *information gain*
  against *cognitive strain*; they claim conservative focusing is near-optimal specifically for
  conjunctive concepts, and that disjunctive concepts resist the same strategy (ch. 6 is
  explicitly "On Disjunctive Concepts and Their Attainment," title only confirmed, contents not
  read). Quarantined: their own theory of which strategy suits which concept logic-type.
- **Harlow's own theory:** learning-set formation reflects progressive elimination of
  ineffective "error factors" (stimulus perseveration, position habits) across problems, per the
  classic account (not independently verified, general domain knowledge). Quarantined.
- **WCST/clinical framing:** perseverative errors as a measure of frontal-lobe executive
  function / cognitive flexibility. Explicitly bracketed off per Hazard 2 — we take the
  rule-shift apparatus, not this diagnostic ontology, and this framing sits adjacent to the
  held-out "clinical reasoning" boundary.
- **Daw et al.'s own theory:** arbitration between a model-based and a model-free decision
  system, with the task built to dissociate them. Quarantined — a specific dual-system theory,
  not adopted as our framework for what the task measures.
- **Probability-matching literature's own debate:** whether matching behavior is a rational
  response to residual environmental uncertainty (e.g., suspected non-stationarity, "regime
  might still change") vs. a genuine deviation from optimal maximizing. Quarantined — we use only
  the apparatus (hidden fixed Bernoulli parameter, non-identifiable within episode), not either
  side of this normative debate.

---

## 9. Sealed encounters

Respected without reading: developmental/infant rule-learning paradigms (habituation,
Marcus-lineage artificial grammar work); non-primate animal cognition (corvid, cephalopod) — I
also deliberately avoided a pigeon-SHJ-replication paper (PMC6730555, "Pigeon Category Learning:
Revisiting the Shepard, Hovland, and Jenkins (1961) Tasks") that surfaced in search results,
since pigeons fall under the sealed non-primate-animal-cognition boundary even though it
directly extends an in-scope paradigm — flagging this because it sits right on the boundary the
brief asked me to watch; verbal/linguistic psychometric item generation; clinical
reasoning/differential diagnosis; inductive logic programming and pool-based active learning
(kept separate from the explicitly-authorized Angluin/membership-query exact-learning material,
which is a different subfield within computational learning theory); clinical-trial adaptive
design; quasigroup/Latin-square completion; graph-coloring phase transitions; SRE/incident
response, SQL, Erlang/OTP, Euclidean construction geometry, epsilon-delta analysis; ARC-AGI-3
design material. None of these were searched or fetched.
