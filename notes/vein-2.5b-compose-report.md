# Vein 2.5b — Composition: Library Learning, Grammatical Inference, Planning

Status: DRAFT IN PROGRESS — being written incrementally during research.

Coverage gap addressed: candidate set currently has nothing exercising the `composition`
content axis or the compose/decompose generation asymmetry.

Sealed topics (NOT fetched, per instructions): Inductive logic programming (Metagol, Popper,
lineage); non-primate animal cognition / developmental rule-learning; verbal/linguistic
psychometric item generation; clinical reasoning; pool-based applied active learning /
clinical-trial adaptive design; quasigroup/Latin-square completion, graph-colouring phase
transitions; SRE/incident response, SQL practice, Erlang/OTP; Euclidean construction geometry,
epsilon-delta analysis practice; ARC-AGI-3 design material.

---

## 1. Compose/decompose families

(to fill)

## 2. Primitive inventories (concrete)

### DreamCoder (Ellis et al. 2020/2021, arXiv:2006.08381) — VERIFIED, read full main text (pp.1-20) from PDF

Substrate: polymorphically-typed lambda calculus, one DSL per domain. Numbers below are quoted
directly from the paper text (Results section, Figs 1/4/5/7, "From learning libraries to learning
languages" section) unless marked as inference.

| Domain | Initial primitives (as stated) | Discovered abstractions (as stated) | Task count / split |
|---|---|---|---|
| List processing | "generic functional programming basis, including routines like `map`, `fold`, `cons`, `car`, `cdr`, etc." | "composed around 20 new library routines"; rediscovers `filter`; Fig.1B shows a 4-layer-deep chain: `concept_4`=filter-like, `concept_13`=maximum, `concept_15`=nth-largest, used to build `sort` | 218 problems (from Ellis et al. 2018 NeurIPS ref), 50/50 test/train, 15 I/O examples each |
| Text editing | not enumerated in main text (S1.1 supplement, not fetched) | learned "a single library of text-editing concepts" generic across tasks | 128 auto-generated train tasks; tested on 108 SyGuS-2017 problems |
| LOGO graphics | pen control + imperative control flow + arithmetic on angles/distances | Fig.4B/C: `semicircle(r)`, `circle(r)`, `spiral(dθ)`, `greek-spiral(n)`, `s-curve(r)`, `polygon(n,l)`, and higher-order `radial-symmetry(n, body)` (a function that takes a whole program as input) | 160 images, 50/50 split |
| Tower building | "same control flow primitives as with LOGO graphics" | Fig.5B, described as "parametric options" (citing Sutton/Precup/Singh 1999 options framework): `arch(h)`, `pyramid(h)`, `wall(w,h)`, `bridge(w,h)` | 107 tower "copy tasks", 50/50 split |
| Regex | not enumerated in main text | learns regexes for phone numbers, dates, times, monetary amounts | 256 CSV columns crawled from web, 5 example strings/concept, 50/50 split |
| Symbolic regression | `+`, `×`, `÷`, and continuous real-valued parameters (fit by gradient descent, penalized via BIC) | parametric curve programs, e.g. 3-parameter program for `1.7x²−2.1x+1.8`, 2-parameter for `−2.3/(x−2.8)` | curves of up to degree-4 polynomial or rational function |
| Physical laws | `subtract-vectors, map, zip, cons, empty, cdr, power, fold, car, +, -, *, /, θ, 1, π` (Fig.7A) | vector algebra layer: add-vectors → add-many-vectors → norm² → scale → inverse-square-schema → etc.; used to reconstruct Newton's 2nd law, parallel-resistor formula, work, magnetic force, kinetic energy, Coulomb's law | 60 physical laws/identities from AP/MCAT "cheat sheets"; **after 8 wake/sleep cycles DreamCoder learns 93% of the laws and identities in the dataset** (exact figure, Results text) |
| Recursive/1959-Lisp programming | minimal subset of 1959 Lisp primitives: `car, cdr, cons, ...` plus the Y-combinator (recursion) | rediscovers `fold` and `unfold` ("origami programming", citing Gibbons 2003) as dual roots, then `map`, `filter` as fold-family variants, then `index` combining fold+unfold families; example outputs shown: Stutter, "take every other", list-lengths, list-differences | 20 basic programming tasks; "with enough compute time (roughly five days on 64 CPUs), DreamCoder learns to solve all 20 problems" |

Other exact figures read directly from text:
- "We typically train DreamCoder on 100-200 tasks" (Dreaming-phase paragraph, p.7).
- Text editing: pre-learning solves 3.7% of problems (avg 235s search); post-learning solves
  79.6% (avg 40s); best 2017-SyGuS competitor CVC4 solved 82.4% under competition compute
  (1hr, 8 CPUs/problem); DreamCoder under that same generous budget solves 84.3%.
- "Across domains, deeper libraries correlate well with solving more tasks (r = 0.79)" — Pearson
  correlation between library depth and % held-out tasks solved, stated directly in text (p.13).
- Refactoring/abstraction mechanism (this IS a compose/decompose family in itself, matches the
  brief's own framing): "Code can be refactored in infinitely many ways, we bound the number of
  λ-calculus evaluation steps separating a program from its refactoring, giving a finite but
  typically astronomically large set of refactorings... we introduce a new data structure... combining
  ideas from version space algebras and equivalence graphs [e-graphs], and we derive a dynamic
  program for its construction." A worked example: "A version space with 10^6 nodes, calculated in
  minutes, can represent the 10^14 refactorings in Fig. 3" for discovering `map` from two solved
  tasks. Default bound on evaluation-step distance = 3.

Saturation curve (Fig. 6A/6B, read directly): Fig.6A plots % held-out test solved against
wake/sleep cycle number (0-20) for six domains (text editing, LOGO graphics, list processing,
symbolic regression, tower building, generative text modeling) — the full-model curves rise
steeply in the first ~5-10 cycles and then visibly flatten/plateau by cycle ~15-20 in all six
panels (e.g. LOGO graphics and tower building approach ~100% and plateau; list processing rises
to ~90% then plateaus). Fig.6B re-plots % test solved against **library size** (x-axis 0-25) and
against **average program depth** (x-axis ~1.0-2.75): performance rises with library size and
then visibly levels off past roughly library size 15-20. This is the paper's own closest analogue
to a "does abstraction discovery flatten with more tasks" curve — it is a performance-saturation
curve tied to library growth, not literally a count of new-abstractions-per-iteration (no such
per-iteration abstraction-count plot was found in the fetched main text; S1.1/supplement not
fetched). Labeling this "saturates" is directly supported by the shape of Fig.6A/6B as printed;
the *cause* (diminishing returns of compression given a fixed task distribution) is INFERENCE.

Note: primitives are LEAD-adjacent-verified only for the fragments visible in figures (Figs 1B,
4B-C, 5B, 7A-B) and the enumerated text; the complete enumerated primitive tables live in
supplement section S1.1, which was not fetched (PDF fetch returned only main text via page-range
reads; supplement pages were not requested). Flagging this explicitly rather than guessing.



## 3. Difficulty parametrizations

(to fill)

## 4. Prior transfer claims

(to fill)

## 5. Rejections

(to fill)

## 6. Primitives — reuse/new decision

(to fill)

## 7. Sources

VERIFIED (fetched and read):
(to fill)

LEAD (seen cited only, not fetched/read):
(to fill)

## 8. Source's taxonomy (quarantined)

(to fill)

## 9. Sealed encounters (titles declined to open)

(to fill)
