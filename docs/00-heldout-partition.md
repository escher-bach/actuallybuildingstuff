# Held-out Partition — SEALED

*Repertoire Specification §11 step 1, §8. Recorded **2026-08-06**, before any source in §2 was inspected and before any register row was written.*

**Status: SEALED.** Nothing listed under "Held out" below may be read, cited, skimmed, or used as a lead until D3 (the measured basis) is frozen. Unsealing is a recorded event: append to §4 of this file with the date and the state of the basis at that moment. If a held-out item is inspected accidentally, record the contamination here rather than quietly reclassifying it — a contaminated held-out item is worthless as evidence but an *unrecorded* contamination is worse, because it makes the coverage claim in §10 condition 3 false without anyone knowing.

The partition serves two different tests and they are not interchangeable:

| Partition | Feeds | The test |
|---|---|---|
| **Held-out practice domains** | §10 condition 3, §8 | Are solver programs for these domains *short* under the basis extracted from the inspected domains? |
| **Held-out veins** | §7 saturation curve, §10 condition 2 | Does reading these produce *new* primitives after the curve has flattened? If yes, the flattening was an artifact of what we chose to read. |

---

## 1. Practice domains

Target domains are reasoning, mathematics, coding, agent use (Task Spec §5). Each is split. The held-out side of each split was chosen to differ in *practice character*, not merely in topic — a different subfield that is solved the same way is not a held-out domain, it is a paraphrase.

### 1.1 Mathematics

**Inspected.** Elementary number theory (modular arithmetic, divisibility, gcd); linear algebra over small finite fields; propositional and first-order logic; finite combinatorics and counting.

**Held out — primary: plane Euclidean geometry with straightedge-and-compass construction.**
Chosen because its practice character is maximally unlike the inspected set: the objects are diagrammatic rather than symbolic, the characteristic move is *constructing an auxiliary object* rather than rewriting an expression, and case analysis is forced by configuration rather than by the statement. If a basis extracted from algebraic practice makes auxiliary-construction search short, that is real evidence of coverage. If it does not, we will have learned exactly which schema is missing, which is the §7 repair question with an answer.

**Held out — secondary: elementary real analysis, epsilon–delta proof practice.**
Chosen for a different reason: it is the canonical domain in which the *control structure* (§2.6) dominates the primitive inventory — the operations are few and the difficulty is entirely in quantifier bookkeeping, backward reasoning from the goal, and choosing a witness. It is the sharpest available test of whether the L1–L3 side of the basis covers anything real.

### 1.2 Coding / language ecosystem

**Inspected.** Imperative and functional general-purpose code (Python-shaped); small typed lambda-calculus and Lisp-shaped DSLs of the kind the library-learning literature uses; stack and register machines.

**Held out — primary: SQL and relational query practice.**
Declarative, set-at-a-time, no explicit control flow, and the characteristic activity — understanding why the planner chose a join order, rewriting a query that is correct but wrong-shaped — has no counterpart in the inspected set. A basis fitted to imperative control flow should visibly fail here if it is fitted at all.

**Held out — secondary: Erlang/OTP-style concurrency and supervision.**
Reserved for the case where the SQL test comes out ambiguous. Its distinctive content is failure-as-normal-operation: restart strategies, partial failure, and recovery *as the design*, which is the §2.6 control structure made into a language feature.

### 1.3 Agent use / tool stack

**Held out — primary: production incident response on distributed systems.**
Deliberately the strongest held-out choice, and the one it is most tempting to cheat on. It is dense in precisely what this programme claims to manufacture — hypothesis formation under underdetermination, choosing the next diagnostic to run (L2), recovery from a wrong first guess, and explicit calibration ("it is probably the cache, but check the deploy first"). Holding it out means we cannot build toward it. That is the point: if the basis extracted elsewhere makes incident-response solver programs short, the coverage claim is not a coincidence of what we were aiming at.

**Inspected.** Symbolic planning domains, tool-calling and API-composition traces, text-adventure-shaped environments, local debugging of single-process programs.

### 1.4 Reasoning

Not split by held-out domain. Reasoning has no independent corpus of practice here — it is the thing the other three are instruments for, and a held-out "reasoning domain" would be a stipulation of the kind §1 of the repertoire spec forbids. Coverage of reasoning is tested through §2.3 paradigm fallout (§11 step 4) and through the held-out veins below, not through a domain partition.

---

## 2. Veins

One sub-area per vein of §2, held out. Each was chosen to be *plausibly rich* — holding out a thin sub-area proves nothing, because finding no new primitives in it is the expected outcome either way.

| Vein | Held out | Why this one |
|---|---|---|
| §2.1 Item generation | **Verbal and linguistic-aptitude item generation** — analogy items, artificial-language-learning aptitude batteries (MLAT/LLAMA lineage) | The inspected side of this vein will be figural/matrix items. Verbal item models are where A2 (knowledge-free) is hardest and most often silently violated, so this is also a live test of our §4.1 step-3 triage |
| §2.2 Query learning / design | **Sequential and adaptive experimental design in clinical trials**, and pool-based active learning for labels in ML | The inspected side is exact learning and Bayesian OED. This sub-area solves the same $q^*$ problem under different constraints (cost per query, ethics stopping rules) and would supply new query policies if the basis is missing any |
| §2.3 Hidden-rule paradigms | **Developmental and infant rule-learning paradigms** (habituation, artificial-grammar work in the Marcus lineage), and non-primate animal cognition (corvid, cephalopod) | The inspected side is adult human and primate paradigms. These use the same reveal structure with radically weaker working-memory assumptions, which is where genuinely different reveal policies would show up |
| §2.4 Planted instances | **Quasigroup / Latin-square completion with holes, and graph-colouring phase transitions** | Named as a lead in the source document, which makes it tempting; that is exactly why it is a good seal test. The inspected side keeps planted SAT, hidden clique, one-way functions |
| §2.5 Program / proof / plan corpora | **Inductive logic programming** (Metagol/Popper lineage) | The strongest held-out vein. ILP *is* structure induction with a hidden hypothesis and an evaluate-only oracle, arrived at independently. If our basis does not already cover it, the compactness claim is in trouble; if it does, that is §10 condition 3 evidence of the strongest kind |
| §2.6 Practice traces | **Clinical reasoning and differential-diagnosis expertise studies** | Protocol analysis applied in a domain with real stakes, real calibration, and a large published corpus of expert-novice differences. Pairs with the held-out incident-response domain (§1.3) — same control structure, different substrate |

**Also sealed:** ARC-AGI-3 design material beyond the single hazard note already quoted in the repertoire spec §8. It is a designed-for-generalization benchmark and reading its task inventory during extraction would fit the basis to it.

---

## 3. Rules of engagement

1. A source that turns out to sit in a held-out sub-area is dropped *the moment that is noticed*, and the drop is logged in the register with reason `SEALED`. It is not read "just to check whether it counts."
2. Secondary held-out items (§1.1, §1.2) may be promoted to primary use if the primary test comes out ambiguous. Promotion is an unsealing event and is recorded.
3. The seal binds subagents too. Every literature-review delegation carries the held-out list in its brief with an explicit instruction to stop and report if a lead falls inside it.
4. The seal does **not** bind incidental prior knowledge already in an agent's weights. That contamination is unavoidable and unmeasurable; what is controlled here is what gets *read, registered, and used as a lead* during extraction. Stated so the coverage claim in §10 is not overread.

---

## 4. Unsealing log

*(append: date, item unsealed, state of the basis at that moment, reason)*

**2026-08-06 — boundary call, not an unsealing.** Dasgupta, "Analysis of a greedy active learning strategy" (NeurIPS 2004) was read and cited during the §2.2 pass. The paper self-describes as **pool-based** active learning, and pool-based active learning for labels sits on the held-out list. §2's carve-out is explicit that "classical *theory* of query learning and of optimal experimental design is IN SCOPE" and that what is sealed is the *applied* label-efficiency literature; this is a theoretical query-complexity analysis, and the reviewing agent independently flagged the same boundary and stopped before the applied material.

**Judged in scope. The seal stands.** Recorded because a defensible boundary call is still a call, and seals erode through the ones nobody wrote down.

---

**2026-08-06 — CONTAMINATION, logged. Non-primate animal cognition (§2 vein table).**

During the §2.3 pass the reviewing agent fetched and read PMC11628440, a study applying Harlow's learning-set task to **wolves and dogs**. Non-primate animal cognition is sealed. The brief stated the boundary explicitly — "Harlow's primate learning-set work is IN SCOPE; corvid/cephalopod paradigms are not" — and the agent, having read the paper, restricted its use to corroborating Harlow's original task structure (344 object pairs, trial-2 percent-correct metric) and explicitly declined to report the paper's own cross-species findings. It separately and correctly stopped short of a pigeon SHJ-replication paper on the same grounds.

**This is still a contamination and is recorded as one.** The seal covers what gets *read*, not only what gets *used*; a self-imposed limit on use is not equivalent to not having read it, because what was read cannot be un-read and the agent's later judgments are no longer independent of it.

*Scope of damage:* narrow. One paper, in a sub-area whose value as held-out evidence is testing whether new primitives appear when non-primate paradigms are read after the basis is frozen. That test is now weakened for the learning-set corner specifically, not for the sub-area as a whole. The corvid and cephalopod literature remains untouched.

*What I am not doing:* quietly reclassifying wolves-and-dogs as in-scope to make the problem disappear. My own rule in §3 says a contaminated item is worthless as evidence but an unrecorded contamination is worse, and that applies to me as much as to a subagent.

*Change to prevent recurrence:* future briefs must state the seal boundary as an instruction about **fetching**, not about using — "do not open it" rather than "do not rely on it". The §2.3 brief said the boundary but not the verb.
