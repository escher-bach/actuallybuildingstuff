# Vein 2.4 — Planted structure: CSP / SAT / cryptography instance generation

Scope: planted-vs-random hardness (planted 3-SAT, hidden/planted clique, Model RB, forced
satisfiable instances), phase transitions and non-monotonicity in random CSP/SAT, one-way
functions/trapdoors (factoring, subset-sum, LWE). Held out and **not fetched**: quasigroup/Latin
square completion, graph-colouring phase transitions (see §9 for a **contamination** — one fetch
crossed this boundary and is disclosed there), ILP, pool-based active learning, clinical-trial
adaptive design, clinical reasoning, non-primate animal cognition, developmental rule-learning,
verbal/psychometric item generation, SRE/SQL/Erlang/geometry/epsilon-delta, ARC-AGI-3.

**Read §9 before trusting §1's "quiet planting" material** — one source used there was fetched in
full despite centering on a sealed topic. The general mechanism is reported; the sealed numbers
are not used as findings anywhere in this report.

---

## 1. Planted-vs-random: what leaks, and how

This is the deliverable the vein is organized around. Five distinct leak mechanisms found, each
with a named fix (or "no known fix" for one):

| # | Construction | What is planted | Statistical signature it leaves | Exploiting algorithm | Fix, if known |
|---|---|---|---|---|---|
| 1 | **Naive 1-hidden random k-SAT** (pick random assignment `A`, reject any clause `A` violates) | Single satisfying assignment `A` | Clauses are drawn only from the ~7/8 of k-clauses `A` satisfies (for k=3, 7 of 8 possible sign-patterns per triple survive). Achlioptas–Jia–Moore show the expected solution-density function is maximized at agreement-fraction `α > 1/2` with `A` — solutions **cluster around the plant** instead of spreading uniformly. | Simple majority/local-search heuristics; WalkSAT finds `A`-correlated regions fast. Achlioptas–Jia–Moore, body: "2-hidden formulas... [are] much harder than 1-hidden ones" for WalkSAT, and zChaff branch counts on 1-hidden formulas were "2 to 5 orders of magnitude" larger than on 2-hidden (i.e. 1-hidden is that much *easier*). | **2-hidden**: plant a complementary pair `A, Ā` (reject clauses violated by either). The bias-generating function becomes symmetric around `α=1/2`; its maximum lands exactly at `α=1/2`, so "the hidden assignments are not felt." Achlioptas–Jia–Moore 2004, JAIR/AAAI. |
| 2 | **2-hidden random k-SAT** (fix for #1) | Complementary pair `A, Ā` | Cancels the *density* bias, but does not fully anonymize the clause distribution — Jia, Moore & Strain report 2-hidden formulas remain **easy for WalkSAT** even though hard for DPLL (zChaff): a solver whose local-search dynamics are attracted along the `A`↔`Ā` axis can still exploit structure DPLL's branching does not see. | WalkSAT with a random restart still lands the basin around `A` or `Ā` a nontrivial fraction of the time (per the ar5iv-fetched body text of Jia–Moore–Strain, arXiv cs/0503044, contrasting the two methods). | **q-biased ("deceptive") hiding**: for each clause, sample how many literals (`t`) are satisfied by `A` with probability `∝ q^t`, `q<1`, instead of rejecting violators outright. At a critical `q*` the formula is *balanced* (each literal equally likely positive/negative); below `q*` it is *deceptive* — the drift term `f'(1/2) < 0` actively points solvers away from `A`. Reported hard for **both** DPLL and WalkSAT (WalkSAT exceeding `10^8` flips at `n=200, r=5.5, q=0.3`, per fetched body text). Jia, Moore & Strain, arXiv cs/0503044. |
| 3 | **Forced-satisfiable Model RB/RD** (reject any clause/constraint violating a fixed random assignment `t`) | Single assignment `t` | Same clustering defect as #1 — this is literally the naive-planting rejection-filter, re-derived independently in the CSP literature. Xu & Li state the mechanism directly: "generate a random truth assignment `t`... where any clause or constraint violating `t` will be rejected" (arXiv cs/0302001, body, VERIFIED). | Xu & Li's own experiment: forced-satisfiable instances near the threshold solved in only **11% less** mean time than genuinely random satisfiable instances of the same size in one test case (`n=30`) — i.e. only mildly easier there — but the paper does not report the >10^8-flip-scale WalkSAT gap that Achlioptas-family naive planting shows for 3-SAT, so severity is model- and parameter-dependent. **This generator is flagged for A1 as well as A4: it is explicit reject-and-resample, not O(1).** | No fix proposed in this paper; the fix from #1/#2 (multi-hidden or biased hiding) is not applied to Model RB in what was fetched. |
| 4 | **Merkle–Hellman knapsack trapdoor** (public weights = superincreasing sequence disguised by modular multiplication + permutation) | A superincreasing sequence, algebraically hidden | The disguising transform is **affine** (multiply-mod-then-permute), which preserves a specific linear/lattice relation among the public weights that a random weight set would not have. | Shamir (1982): reduce to finding a particular short vector in a lattice built from the public key; recovers the trapdoor **regardless of instance density**, because the attack targets the *algebraic disguise*, not a statistical density property. (LEAD, consistent across multiple secondary sources — Odlyzko's survey, cryptanalysis surveys — primary 1982 paper not fetched.) | None for the *disguise-as-affine-transform* approach; the scheme was abandoned. (Iterated/multiply-disguised variants were also broken — Brickell's attack — LEAD only.) |
| 5 | **Any low-density subset-sum instance** (planted or not — this is a generic leak, not specific to a trapdoor construction) | A target-sum subset | Density `d = n / log2(max weight)` below a threshold means the subset-sum lattice has an anomalously short vector encoding the solution. | Lagarias–Odlyzko (1985): solves almost all instances of density `< 0.6463` in poly time given an SVP oracle; Coster–Joux–LaMacchia–Odlyzko–Schnorr–Stern (CJLOSS) improved the bound to density `< 0.9408`. (LEAD, corroborated across 3+ independent search-indexed secondary sources with a consistent 0.9408 figure; primary papers not fetched.) | Push density **above** ~0.94 (large weights relative to `n`) to defeat this *specific* attack family — but this is a density-threshold dodge, not a structural proof of hardness; no unconditional hardness guarantee follows. |

**General mechanism note (quarantined per §9):** the statistical-physics "quiet planting" idea —
construct the planted ensemble so its low-order statistics (and, in the cavity-method framing,
its belief-propagation fixed point) coincide with the *conditioned-satisfiable* random ensemble —
is the general-purpose answer to "how do you plant without leaving footprint #1's signature." The
paper I fetched to learn this states explicitly that it does **not** apply to random SAT ("the
fixed point of the BP equation is not uniform and where our results do not apply" — quoted from
the fetch, general statement, not graph-colouring-specific) — i.e., **the trick that removes the
signature for some CSPs is not a universal solvent; SAT resists it.** This negative scope
statement is usable; the paper's worked demonstration (which numbers the technique) is not — see
§9.

**Bottom line for A4 (deliverable 1's real question):** naive single-solution planting (mechanism
#1/#3) is the textbook case of "hard in general, easy for the planting distribution" — it fails
in practice by construction, not by bad luck. The literature's own fix is not "plant more
carefully once" but "plant symmetrically" (2-hidden) or "plant adversarially against the known
attractor" (q-biased/deceptive). Both fixes are still **O(1)-per-clause direct constructions**
(no rejection sampling) once the target bias is chosen — this matters for A1, see §5.

---

## 2. Candidate families

Note up front: this vein's oracle-fit is unusually clean on the optimality axis the brief warns
about. Every family below asks "does this candidate satisfy/solve," never "is this candidate
optimal" — SAT/CSP satisfaction, subset-sum target-hitting, factoring, and LWE-consistency are
all yes/no certificates. Planted clique is a near-miss (flagged in its row).

| Family | `Θ` | `k` enters via | `X` (query space) | `f` (oracle) computes | Encoding 1 | Encoding 2 | L3 posterior enumerable? |
|---|---|---|---|---|---|---|---|
| **Planted-SAT** (2-hidden or q-biased variant) | hidden assignment pair `(A, Ā)` or biased-`A`, over `n` Boolean vars | `n` (variables) and clause/variable ratio `r` | a candidate full assignment, or a single clause to check | is the candidate consistent with the formula / does the clause hold under `A` | literal list `(x1 ∨ ¬x2 ∨ x3)` tokens | natural-language "if X then not Y or Z" constraint prose | yes at small `n` (2^n assignments); intractable already by `n≈30` |
| **Model RB (forced-satisfiable)** | random assignment `t` over `n` vars, domain size `d=n^α` | `n`, tightness `p` vs. critical `p_cr = 1-e^{-α/r}` (exact formula, Xu–Li, VERIFIED) | a candidate tuple of variable/value assignments for one constraint | is the tuple in the constraint's compatible set | constraint tables (variable IDs → allowed tuples) | prose "variable P cannot take value 3 when Q=1" | yes at small `n, d`; **generator itself uses rejection sampling — see §5 A1 flag** |
| **Planted clique** | vertex subset `S`, `\|S\|=k`, embedded in `G(n,1/2)` | `n` (graph size), `k` (clique size, `k=c·√n` by convention) | a candidate vertex subset | is the subset a clique of size `≥k` in `G` | adjacency-list / edge-token graph | prose "person A knows person B" social-network framing | yes for small `n` (enumerate `C(n,k)` subsets); **near-miss on optimality** — see note below |
| **Subset-sum w/ trapdoor** | weight vector `w`, target `T=Σ_{i∈S} w_i` for hidden `S` | `n` (items), bit-length of weights (density knob) | a candidate subset (bit vector) | does the candidate's weighted sum equal `T` | integer list + target | prose "can these items be combined to weigh exactly T" | yes for small `n` (`2^n` subsets) |
| **Factoring / modular one-way** | `(p,q)` primes, `N=pq` | bit-length of `p,q` | a candidate integer `x` | does `x` divide `N` (`N mod x == 0`) | plain integer `N` | prose "N is the product of two secret numbers" | yes for small `N` (trial division over all `x≤√N`) |
| **LWE-style noisy modular** | secret vector `s`, samples `(A,b=As+e mod q)` | dimension `n`, modulus `q`, noise bound | a candidate secret guess `s'` | is `‖A s' − b‖` within the noise bound (mod q) | matrix/vector of integers | prose "approximately-consistent linear system" | yes for small `n,q` (enumerate `s' ∈ Z_q^n`) |

**Planted clique optimality caveat:** the natural certificate ("is `x` a clique of size `≥k`")
coincides with "is `x` the planted clique" only once `k` exceeds the point where a same-size
spurious clique is vanishingly unlikely to arise by chance in `G(n,1/2)` — below that, "any
clique of size `≥k`" is a weaker ask than "the specific planted one," and the family risks
smuggling in a best-of search. Concretely this bites near the information-theoretic boundary
`k ≈ 2 log2 n` (§4); comfortably above it (`k` a constant factor times `√n`), the planted clique
is with high probability the unique largest, so the certificate and the "find the plant" target
coincide again. **Repairable, not a hard fail** — keep `k` well clear of the `2log2 n` boundary.

**`q*` (teacher policy) note, all families:** because the oracle in this vein is a pure
certifier, the trivial teacher policy is "submit `θ` itself as the query" — a one-shot win, valid
per A7 but uninteresting as an *informative-query-selection* exemplar unless the interface is
restricted to partial/probing queries (e.g., "does bit `i` of the secret equal 1," "is edge `(u,v)`
inside the clique") rather than whole-candidate submission. Worth deciding explicitly at
implementation time — this vein does not resolve it for you.

---

## 3. Non-monotonicity, concretely

| Claim | Status | Source, location |
|---|---|---|
| Random k-SAT hardness is **non-monotone** in clause/variable ratio `r=m/n`: near-empty and over-constrained formulas are both easy (few constraints to violate / trivially UNSAT), difficulty peaks near the satisfiability threshold. "Easy–hard–easy" is the standard name. | Established qualitative shape; the *specific numeric peak location* for `k=3` is not proven, only bracketed (see next rows) — a case where the shape is solid and the coordinate is not. | Mitchell, Selman & Levesque, AAAI-92 (title/abstract-level LEAD only — PDF would not parse to text; I could not independently re-derive their exact reported ratio or the `n` values they tested, so I am **not** stating a number attributed to this paper). |
| Statistical-physics ("cavity method") **prediction** for 3-SAT threshold: `α_c ≈ 4.267`. | **Prediction, not proof** — explicitly non-rigorous method. | Mezard, Parisi & Zecchina, *Science* 2002 (LEAD — PDF fetch failed to parse; number corroborated across multiple independent secondary sources with consistent value, so treated as reliably-attributed LEAD, not VERIFIED against primary text). |
| Rigorous bracket on the true (unknown) `k=3` threshold: proven lower bound `≥3.52` exists in the literature (title: "The Satisfiability Threshold of Random 3-SAT Is at Least 3.52"). | Proven lower bound only — a bracket, not the exact value. | Title/search-indexed only (LEAD; not fetched). |
| For **large `k`** (asymptotic in `k`, not `k=3`), the exact threshold is proven: `r_k = 2^k ln2 − (1/2)(1+ln2) + o_k(1)`, matching the 1RSB statistical-physics prediction up to a vanishing error term. | **Rigorous, but asymptotic in `k`** — this is a `k→∞` statement, not a finite-`k` fact, and explicitly not `k=3`. | Ding, Sly & Sun, STOC 2015 / Annals of Math. 2022 (LEAD — formula corroborated via search-indexed abstract text, not fetched from primary). |
| **Model RB has an exactly-located phase transition, proven, but strictly asymptotic:** `lim_{n→∞} Pr(Sat)=1` for `r<r_cr`, `→0` for `r>r_cr`, with `r_cr = −α/ln(1−p)` given in closed form. No finite-`n` numeric bound is proven anywhere in the paper. | Rigorous existence-and-location theorem, `n→∞` only, by the paper's own explicit statement ("as the number of variables approaches infinity"). | Xu & Li, arXiv cs/0004005, Theorems 1–2 (VERIFIED, ar5iv fetch, body). |
| Model RB/RD instances near the transition provably have **no sub-exponential tree-like resolution proof** (`2^Ω(n)` lower bound on tree-resolution size), which is the paper's own justification for "these instances are hard." | Rigorous, but again `n→∞` asymptotic ("almost surely" is explicitly defined in-paper as `n→∞` probability). No finite-`n` bound given. | Xu & Li, arXiv cs/0302001, Theorem 3 (VERIFIED, ar5iv fetch, body). |
| Planted clique is **not** an easy–hard–easy peak in the SAT sense — it is a monotone **threshold/gap** structure: larger `k` is easier (both info-theoretically and computationally), smaller `k` harder down to undetectable. The "non-monotonicity" this vein was told to expect for `k`-the-difficulty-knob does not have a planted-clique analogue in the same shape. | My own structural observation from the two threshold statements below (§4) — flagged as inference, not a quoted claim. | — |

**Where the peak sits, in what parameter:** for random k-SAT-style families, the peak sits **at
the satisfiability threshold ratio itself** (`r ≈ α_c`), not at some other parameter — difficulty
and the SAT/UNSAT phase boundary are the same coordinate. This is qualitatively solid across every
source found; the exact numeric coordinate for `k=3` remains a rigorously-open question (bracketed,
not pinned) as of the sources checked here — **figure not located to my satisfaction for a
finite-`n` empirical peak; only the physics estimate (4.267, unproven) and rigorous brackets
(≥3.52, unspecified upper bound found) are available.**

---

## 4. A4 as a guarantee

| Assumption | What is lower-bounded | Proof status | Size at which it bites |
|---|---|---|---|
| **Factoring/RSA**: no poly-time classical factoring algorithm | best known classical algorithm (GNFS) is sub-exponential, `L_N[1/3,·]`, in bit-length of `N`; trial division is `O(√N)` | **Conjecture** (widely believed, not proven — factoring's decision version is in NP∩coNP, "NP-intermediate," not known/believed NP-complete, so it is not even backed by a P≠NP-style argument) | At toy scale this guarantees **nothing**: for `N` around `10^6` (~20 bits), `√N ≈ 1000` — trial division solves it near-instantly, and GNFS's asymptotic sub-exponential advantage over brute force is a large-`N` phenomenon that has not "turned on" yet. No sourced crossover digit-count was found; this is a basic-arithmetic observation, not a cited figure. |
| **Subset-sum density (Lagarias–Odlyzko/CJLOSS)** | any instance below density `0.9408` is solvable via lattice reduction in poly time | **Proven** algorithm (conditional on SVP-oracle behavior in practice, which lattice-reduction heuristics approximate) — this is actually a proven *attack*, i.e. it proves **absence** of hardness below the density line, not presence of hardness above it | Above the density line there is **no proven hardness guarantee**, only "this specific attack doesn't apply." Subset-sum-based cryptography is in practice abandoned as a hardness assumption — this family is better read as deliverable-1 material (a leak catalogue entry) than as an A4 guarantee. |
| **Planted clique conjecture**: no poly-time algorithm detects/recovers a planted clique of size `k=o(√n)` | best known poly-time algorithms (spectral, SDP) require `k=Ω(√n)`; information-theoretic detectability holds down to `k≥(2+ε)log2 n` (VERIFIED quote, ar5iv fetch of arXiv 2402.05451, intro: "it is possible to detect the presence of a planted clique of size k ≥ (2+ε)log₂(n)") | **Conjecture**, but with real unconditional partial evidence: Jerrum (1992) proved the Metropolis process specifically fails super-polynomially below the threshold (LEAD, corroborated across 3 search-indexed sources, not fetched from primary) — this is an unconditional lower bound for *one named algorithm class*, not a general one. Later work (statistical-query lower bounds, low-degree-polynomial lower bounds, sum-of-squares lower bounds) extends "no known efficient algorithm" evidence to broader restricted classes, still not full P≠NP-strength. | **My own arithmetic, flagged as inference, not a sourced claim**: solving `2log2(n) = √n` gives crossover at `n=256` (`2log2(256)=16=√256`). Using the literal formulas as stated (info-theoretic `≥2log2 n`; computational `Ω(√n)`, treating the hidden constant as 1 for illustration only), the "hard gap" region — clique sizes that are detectable in principle but not by known poly-time algorithms — is **empty or negative below `n≈256` vertices** and only widens above it. This is illustrative given the unpinned `Ω()` constant, but the qualitative conclusion (the gap does not exist at toy graph sizes) is robust to reasonable constant choices. Real cryptographic use of this conjecture targets `k` around `n^{log^α n}` scale (2025 eprint, LEAD, title: "Using the Planted Clique Conjecture for Cryptography") — astronomically past any toy regime. |
| **LWE**: worst-case lattice problems (GapSVP, SIVP) reduce to LWE | LWE hardness, **given** the reduction | **Proven reduction** (Regev 2005, originally quantum; later work established classical reductions for polynomial moduli, per search-indexed secondary sources — not independently fetched from primary) — this is the one case in the whole vein where a real worst-case-to-average-case theorem, not just failure of known attacks, backs the hardness claim | **No sourced concrete "minimum `n` for the reduction to be meaningful" was located** — the concrete-hardness literature (fetched only as search snippets, e.g. "On the concrete hardness of Learning with Errors," Albrecht et al.) works with cryptographic-scale parameters (dimension `n` in the hundreds to low thousands, per deployed schemes like Kyber/ML-KEM) and does not, in what I found, state a lower cutoff below which the *reduction itself* is vacuous. I flag this as the vein's clearest instance of Hazard 1: the reduction is a `n→∞`-flavored statement with polynomial loss factors, and nothing in the sources locates its finite-size floor. Treat "does this reduction say anything at `n≈12`" as **unresolved, not answered**, rather than assuming it transfers. |

**Honest summary (the vein's main negative result, as instructed):** exactly one family here
(LWE) offers a hardness *reduction* rather than a hardness *conjecture-about-known-attacks*, and
it is also the one family where I could not locate any statement — proven or estimated — of the
parameter size at which that reduction starts to mean anything. Every other "guarantee" in this
vein (factoring, planted clique) reduces on inspection to "best known attacks are sub-exponential/
polynomial-with-a-gap," evidenced at cryptographic scale (hundreds+ of bits/vertices/dimensions),
and by direct arithmetic (planted clique) or by construction (factoring, trial division) that
evidence **does not survive** the shrink to toy `k`. A4 in this vein should be read as "borrowed
credibility from a regime we are not operating in," not as a transferable guarantee, unless a
family is deliberately parameterized to sit well past its own crossover point — which then
reopens the question of whether a tiny model can be trained on it at all.

---

## 5. Rejections (with repairs)

| Family / generator | Fails | Repair |
|---|---|---|
| **Naive 1-hidden planted k-SAT** (single random assignment, reject violating clauses) | **A4**: solutions cluster around the plant; WalkSAT/local search finds it fast (§1, mechanism #1) | Use 2-hidden (plant `A` and `Ā`) or q-biased/deceptive hiding (§1, mechanism #2) — both still O(1)-per-clause once the bias parameter is fixed |
| **Model RB "forced satisfiable" generator** (reject any clause/constraint violating a fixed random assignment `t`) | **A1**: explicit reject-and-resample on the generation path — Xu & Li's own description is "generate... constraints... where any [one] violating `t` will be rejected" (VERIFIED quote, arXiv cs/0302001). **Also A4**-adjacent: same clustering defect as naive 1-hidden SAT, though the one reported number (11% runtime gap, `n=30`) suggests the effect may be milder here than in 3-SAT — not independently confirmed at scale. | Same repair family as above: replace single-assignment rejection with a 2-hidden or biased-hiding scheme reformulated for the constraint-tuple setting; this has not been done in the fetched literature and would need to be worked out, not just imported |
| **Model RB proper** (not forced-satisfiable — the base phase-transition model) | Not a rejection-sampling fail (generation is direct, VERIFIED), but **its hardness/phase-transition theorems are `n→∞` asymptotic only** — no finite-`n` guarantee is proven (§3, §4) | Not really repairable within the paper's own results — treat any specific small `n` as *empirically* calibrated (run a solver, measure), not as *provably* hard by appeal to the theorem |
| **Merkle–Hellman-style trapdoor subset-sum** | **A4 fails outright, historically** — broken unconditionally by Shamir's lattice attack regardless of density (§1, mechanism #4) | No repair salvages this specific disguise mechanism; if subset-sum is wanted at all, use it purely as a §1 leak-catalogue teaching example, not as a hardness source |
| **Any subset-sum instance below CJLOSS density 0.9408** | **A4 fails** — solvable via lattice reduction regardless of whether a trapdoor was planted at all (§1, mechanism #5) | Keep density above ~0.94 to dodge *this* attack — but see §4: this is not a proof of hardness above the line, only absence of *this* attack, so treat any resulting "guarantee" as weak |
| **Planted clique below `k ≈ 2 log2 n`** (or, per §2's optimality caveat, anywhere the certificate "size-≥k clique" could match a spurious clique) | **A5/A2-adjacent**: below the information-theoretic threshold the plant is not even findable in principle, and near it "find a clique of size ≥k" stops meaning "find the plant" | Keep `k` a constant factor times `√n`, comfortably clear of `2log2 n` (§2, §4) |
| **Any of the crypto-style families (factoring, LWE, planted clique) run at literal toy scale (`n` in the tens)** | **A4 fails** in practice even though the asymptotic machinery says otherwise — see §4's whole point | No repair within this vein's literature; either accept these families as A2/A5/A1-clean but A4-unsupported at the sizes actually trained on, or find a different hardness source for that regime (out of scope for this review) |

**Generators flagged for rejection sampling specifically (A1 trap), consolidated:** naive
1-hidden planted SAT (§1 #1) and the Model RB forced-satisfiable variant (§1 #3, this table) both
explicitly reject-and-resample on the generation path. The two fixes found in the literature
(2-hidden / q-biased hiding) are, by contrast, direct O(1) constructions once the bias parameter
is chosen — this is the one place in the vein where "harder to detect" and "cheaper to generate"
turned out to coincide rather than trade off.

---

## 6. Primitives

**Zero new primitives proposed.** This is the fourth "zero new" result per the running tally.

Reasoning: this vein's actual content is almost entirely about the **generator side** of the
formalism (A1/A4 concerns — how to sample `θ` and an instance without leaking or without cheap
brute force) rather than the **solver/teacher-policy side** the primitives vocabulary tracks
("something a hand-written solver COMPUTES," per `register/primitives.toml`'s own framing).
Planting mechanisms, phase transitions, and hardness reductions are properties of the *family*,
not operations a query-driven identifier performs — so most of this report's content structurally
cannot mint a new primitive regardless of novelty.

Where solver-side operations did appear (as external "attacker" algorithms used to calibrate
hardness, not as this formalism's own L1/L2 apparatus), they reuse existing entries cleanly —
in two cases suspiciously cleanly, since the register's aliases already anticipated SAT-solver
terminology by name:

| Existing primitive | Source terms folded in |
|---|---|
| `constraint-propagation` | DPLL unit propagation on planted-SAT/Model RB clauses — the register's alias list already names "unit propagation" verbatim; this vein just supplies the concrete substrate |
| `backtrack-splice` | DPLL backjumping on SAT search trees — the alias list already names "backjump" verbatim, same situation |
| `basis-probe` | Lattice-basis reduction (LLL) as used in the Shamir and Lagarias–Odlyzko/CJLOSS attacks (§1 #4, #5), and the spectral (top-eigenvector) recovery algorithm for planted clique (§4) — both exploit a fixed linear/algebraic structure (a lattice basis; an eigenbasis) with a schedule fixed in advance and no explicit belief state, matching the gloss ("a standard basis, a group action, a linear artifact order... maintaining no belief state at all") reasonably well. Moderate-confidence fold, not a verbatim alias match like the two above. |
| `informative-query-selection` (weak fit, flagged not forced) | WalkSAT's greedy flip selection (score candidate flips by clause-violation reduction, take the best) plausibly fits; backdoor-variable selection (pin a small variable subset expected to collapse the rest to poly-time) is a *plausible but not confident* fit — a backdoor commitment is closer to "propose a partial solution to a sub-solver" than "query an oracle," and I would not treat this fold as settled. |

No family in this vein needed a belief-state-reset, an abstraction-naming step, or an
epistemic-status tag — the generator-vs-solver split above is the likely reason, not an absence
of interesting structure.

---

## 7. Sources

**VERIFIED** (fetched in full — via ar5iv HTML rendering, which parses correctly, unlike raw
PDF fetches from arxiv.org/pdf or non-arXiv hosts, which failed to parse as text in every attempt
this session):

- Achlioptas, D., Jia, H., Moore, C. "Hiding Satisfying Assignments: Two are Better than One."
  arXiv cs/0503046 (also AAAI-04 / JAIR). Body text on the 1-hidden vs. 2-hidden mechanism,
  the `f_{k,r}(α)` / `g_{k,r}(α)` bias functions, and WalkSAT/zChaff hardness comparisons quoted
  in §1.
- Jia, H., Moore, C., Strain, D. "Generating Hard Satisfiable Formulas by Hiding Solutions
  Deceptively." arXiv cs/0503044. Body text on the `q^t`-weighted clause construction, the
  `q*` balance point (paper tests `q=0.618` explicitly, per two independent fetches/searches;
  **I could not independently verify the closed-form "exactly the golden ratio" framing beyond
  the numeric test value — reporting the tested number, not asserting the closed-form identity
  as confirmed**), and hardness numbers (`n=200, r=5.5`, WalkSAT exceeding `10^8` flips at
  `q=0.3`) quoted in §1.
- Xu, K., Li, W. "Exact Phase Transitions in Random Constraint Satisfaction Problems." arXiv
  cs/0004005. Body text: Model RB generation procedure, Theorems 1–2 (exact asymptotic phase
  transition location), explicitly `n→∞` framing, quoted in §2–§4.
- Xu, K., Li, W. "Many Hard Examples in Exact Phase Transitions with Application to Generating
  Hard Satisfiable Instances." arXiv cs/0302001. Body text: Theorem 3 (`2^Ω(n)` tree-resolution
  lower bound), the forced-satisfiable rejection-sampling description, and the two numeric test
  cases (`n=30`, `n=59`) with the "11% smaller mean time" result, quoted in §1, §3, §5.
- arXiv 2402.05451 ("Low-degree phase transitions for detecting a planted clique in sublinear
  time"), introduction only. Direct quote: information-theoretic detectability at
  `k ≥ (2+ε)log2(n)`; citation (not full derivation) of the AKS spectral threshold `k=Ω(√n)`.
  Used in §4.

**LEAD** (search-snippet or secondary-source only — every direct PDF fetch attempted this session
for a non-ar5iv-hosted paper returned unparseable binary; these are reported as LEAD accordingly,
never presented as directly read):

- Mitchell, D., Selman, B., Levesque, H. "Hard and Easy Distributions of SAT Problems." AAAI-92.
  Title/abstract-level only; I explicitly declined to state their reported crossover-ratio number
  or tested `n` values since I could not read the primary text (§3).
- Mezard, M., Parisi, G., Zecchina, R. "Analytic and Algorithmic Solution of Random Satisfiability
  Problems." Science 297 (2002). `α_c ≈ 4.267` figure corroborated across multiple independent
  search-indexed secondary sources with consistent value; explicitly flagged as a non-rigorous
  cavity-method prediction, not a proof (§3).
- Ding, J., Sly, A., Sun, N. "Proof of the Satisfiability Conjecture for Large k." STOC 2015 /
  Annals of Math. 196 (2022). Formula `r_k = 2^k ln2 − (1/2)(1+ln2) + o_k(1)` from search-indexed
  abstract text (§3).
- "The Satisfiability Threshold of Random 3-SAT Is at Least 3.52" — title/search-indexed only,
  used solely to establish that a rigorous lower bound exists and is below the physics estimate
  (§3).
- Alon, N., Krivelevich, M., Sudakov, B. "Finding a Large Hidden Clique in a Random Graph."
  1998 (not on arXiv; not fetched). Threshold `k=Ω(√n)` used via the VERIFIED secondary citation
  above (arXiv 2402.05451), not read directly.
- Jerrum, M. "Large Cliques Elude the Metropolis Process." Random Structures & Algorithms 3
  (1992). Negative result quoted/paraphrased consistently across 3 independent search results;
  not fetched from primary (§4).
- "Using the Planted Clique Conjecture for Cryptography." eprint 2025/1501. Title and one-line
  description only (§4).
- Shamir, A. (1982) attack on Merkle–Hellman; Lagarias, J., Odlyzko, A. (1985) low-density
  subset-sum attack; Coster, Joux, LaMacchia, Odlyzko, Schnorr, Stern (CJLOSS) improved bound.
  All LEAD, via secondary surveys (Odlyzko's own "The Rise and Fall of Knapsack Cryptosystems"
  survey, search-indexed only, not fetched) — the `0.9408` density figure is corroborated
  consistently across independent search results (§1, §4).
- Regev, O. "On Lattices, Learning with Errors, Random Linear Codes, and Cryptography." STOC
  2005. Worst-case-to-average-case reduction claim via search-indexed secondary description; not
  fetched from primary (§4). Concrete-hardness parameter literature (Albrecht et al., "On the
  concrete hardness of Learning with Errors") likewise LEAD only.
- Williams, R., Gomes, C., Selman, B. "Backdoors to Typical Case Complexity." IJCAI 2003. PDF
  fetch failed to parse; one secondary blog post (danglingpointers.substack.com) was fetched and
  quotes the paper's Table 2 directly ("in a problem with 6783 variables, the crux... can be
  solved by only considering 12 variables") — treated as LEAD-via-secondary-quote, not VERIFIED
  against the primary PDF. No connection between backdoor size and the phase-transition region
  was found in what I could read; I do not claim one.
- Achlioptas, D., Gomes, C., Kautz, H., Selman, B. "Generating Satisfiable Problem Instances."
  AAAI/IAAI 2000 — **title noted from a search snippet only, not fetched**, because the same
  paper's primary content is quasigroup-completion-with-holes (sealed topic); the snippet's
  passing remark about random-3-SAT forced-satisfiable bias ("produces a biased sampling of
  instances with many solutions clustered around the forced solution... much easier to solve")
  is consistent with mechanism #1/#3 in §1 and is reported as corroborating LEAD, not as a
  finding requiring the sealed content.

**Not reached at all:** primary text of MSL92, AKS98, Jerrum92, Shamir82, Lagarias–Odlyzko85,
Regev05, and the IJCAI-03 backdoors paper — every one of these hit either a paywall-adjacent
non-arXiv PDF that would not parse to text, or was deliberately not fetched (Achlioptas et al.
2000, per the QWH-adjacency issue above).

---

## 8. Source's taxonomy (quarantined — not adopted)

- **Statistical-physics framing (cavity method, replica symmetry breaking, "quiet" vs. "loud"
  solutions, condensation threshold):** this is the source literature's own explanatory apparatus
  for *why* the SAT threshold sits where it does (clustering of the solution space, 1RSB).
  Quarantined — we take the *empirical shape* (easy-hard-easy, phase transition) and the
  *mechanism catalogue* (§1), not the physics ontology (order parameters, free energy, replica
  symmetry) as our own explanation.
- **Proof-complexity framing (tree-like resolution, Ben-Sasson–Wigderson size-width relation):**
  Xu & Li's own justification for "Model RB is hard" is resolution-proof-size, a specific
  complexity-theoretic notion of hardness tied to one proof system. Quarantined — our A4 is about
  brute-force solver cost, not proof-system-specific lower bounds; the two can diverge (an
  instance can be resolution-hard yet have a short non-resolution certificate).
- **Cryptographic security-reduction framing (worst-case-to-average-case, security parameter,
  negligible-advantage adversary):** this is a *design goal* (defeat all poly-time adversaries)
  distinct from our A4 (structure-ignoring solver cost must outgrow `f`'s size in `k`) — the
  crypto literature's target is a much stronger and differently-shaped guarantee than what A4
  asks for, and §4's whole conclusion is about the mismatch between the two, not an adoption of
  the crypto framing.
- **Statistical-computational-gap framing (SQ lower bounds, low-degree polynomial lower bounds,
  sum-of-squares hierarchy bounds) as evidence hierarchy for planted clique:** quarantined as the
  *source's* preferred evidentiary ladder for "how convinced should we be a conjecture is true."
  We record what each rung proves (§4) without adopting the ladder itself as our confidence
  metric.

---

## 9. Sealed encounters — including one contamination

**Titles seen in search results and correctly declined (not fetched):**
- Quasigroup With Holes (QWH) / Latin-square completion — surfaced repeatedly as the companion
  topic to forced-satisfiable SAT in Achlioptas, Gomes, Kautz, Selman (2000), "Generating
  Satisfiable Problem Instances" — title noted, primary PDF not opened (§7).
- Graph-colouring phase-transition papers generally, wherever they surfaced as search hits
  separate from the item below.
- Inductive logic programming; pool-based applied active learning; clinical-trial adaptive
  design; clinical reasoning/differential diagnosis; non-primate animal cognition; developmental
  rule-learning; verbal/linguistic psychometric item generation; SRE/incident response; SQL;
  Erlang/OTP; Euclidean construction geometry; epsilon-delta analysis; ARC-AGI-3 design material —
  none of these were searched or surfaced at all.

**CONTAMINATION — logged, not hidden:** I fetched Krzakala & Zdeborová, "Hiding Quiet Solutions
in Random Constraint Satisfaction Problems" (arXiv 0901.2130) in full via ar5iv, intending to
learn the general "quiet planting" mechanism (moment/BP-fixed-point matching between planted and
random ensembles), which is squarely in scope for deliverable 1. The paper's abstract, as it
appeared in the search snippet before I fetched it, is general-CSP-framed and does not name
graph colouring. **Its worked demonstration, which the full-text fetch returned, is centered on
graph `q`-colouring** — the fetch returned graph-colouring-specific mechanics (per-vertex random
colour assignment, edges only between differently-coloured vertices) and graph-colouring-specific
numbers (a stated threshold `c_l=16` for 5-colouring). Graph-colouring phase transitions are
explicitly sealed for this vein and flagged by the brief as "the most likely contamination in the
whole review" — which is exactly what happened.

**What I did and did not use from it:** §1 uses only the paper's general, substrate-independent
claim (planted ensembles can be built whose properties match the random ensemble, up to the
existence of the plant) and its explicit **negative** scope statement (does not apply to random
SAT) — both stated in the paper without graph-colouring specifics in the sentences quoted. The
graph-colouring-specific numbers (`c_l=16`, 5-colouring, the `c_d`/`c_l` boundary values) are
**not used anywhere in §1–§6 as findings** — they appear only in this disclosure, so a reader can
judge exactly what was touched. If this paper should have been avoided entirely rather than
partially quarantined after the fact, that is a fair criticism of this review; I chose disclosure
over silent use or silent omission of a real, relevant mechanism.
