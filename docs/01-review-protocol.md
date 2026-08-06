# Review Protocol

*How this review is actually run. Repertoire Spec §11 gives the order of work; this records the decisions inside it that the spec leaves open, and why.*

## Vein order, and why §2.2 goes first

§11 step 2 says "register veins §2.1–§2.5" without ordering them. The order chosen:

**§2.2 (query learning and experimental design) → §2.1 (item generation) → §2.5 (program/proof/plan corpora) → §2.3 (hidden-rule paradigms) → §2.4 (planted instances) → §2.6 (practice traces, its own step).**

The one non-obvious placement is §2.2 at the front. §9 lists **L2 attrition** as a failure mode with a specific cost profile: A7 requires a computable teacher query policy, some families will not admit one, "and they will disproportionately be the interesting ones" — and discovering this after the basis is fixed means redoing the register. §2.2 is the vein that answers this *by reading*. Doing it first produces a q\* tractability map keyed by hypothesis class, which every later vein's rows can be triaged against at the moment they are written. The alternative — registering thirty rows with `A7 = unknown` and revisiting — pays the cost the spec warns about, in full, for no gain.

§2.1 goes second because it supplies the *discipline* rather than the content: the radical/incidental distinction is our k/e distinction with forty years of methodology attached, and rows written after absorbing it are better rows.

**Deviation, 2026-08-06 — §2.5 promoted ahead of §2.1.** After the §2.2 pass the coverage grid showed four empty cells: `state` and `composition` content, `corrupt/localize` and `compose/decompose` asymmetry. §2.2 is a structure-induction vein and filled none of them, which is expected. Repertoire Spec §4 says to track coverage *as the review proceeds* because "the gaps tell you which vein to read next", and that instruction is evidence where the order above was a prior. §2.5's corrupt/localize schema fills two content gaps and two asymmetry gaps at once, and it lands on error recovery, which §5 names as the priority. §2.1 keeps its place otherwise — its yield is discipline rather than coverage, and rows written after absorbing the radical/incidental distinction will still be better rows.

The §2.5 pass is split in two briefs rather than one: corrupt/localize first, then library learning / grammatical inference / planning. Partly because the vein is large, partly because the two long agent briefs that failed so far both failed by exhausting themselves before writing anything.

§2.6 is last in reading order but is **not** allowed to become the thing that gets cut. §11 step 3 gives it its own budget line and §2.6's own hazard note says it is "the most expensive vein per unit yield, and the one most likely to be skipped for that reason." It is scheduled, not deferred.

## Delegation

Literature review is delegated to subagents, **one at a time, never in parallel**. Serial is not a throughput concession here — the primitive vocabulary is shared state, and two agents naming the same operation differently is precisely the synonym drift that corrupts the saturation curve. Each agent's primitives are folded into `register/primitives.toml` before the next launches.

Every brief carries the five elements listed in `register/README.md`. The seal (`docs/00-heldout-partition.md`) is restated in full in each brief, with its boundaries drawn explicitly — "classical query-learning theory is in scope, applied active learning is sealed" — because a boundary stated vaguely gets crossed.

Subagent output is a **report**, not register rows. Rows are written from reports by hand. The curation step is where synonym folding, `verified` demotion, and cross-vein duplicate detection happen, and it does not survive being automated away.

## What gates what

| Gate | Condition | Consequence of failing |
|---|---|---|
| §11 step 1 | Held-out partition recorded before any inspection | *Done, 2026-08-06.* Coverage claims in §10 condition 3 are unavailable without it |
| §11 step 2 | Primitive count flattens (§7 saturation curve) | Keep reading. Still linear at ~300 sources means the compactness thesis is in trouble, and that is a finding |
| §11 step 3 | Inter-annotator agreement on trace coding | Do not use an improvised coding scheme to rescue it; that is what §2.6's methodology exists to prevent |
| §11 step 4 | §2.3's paradigms fall out of the parametrization as special cases | The parametrization is missing something. Repair before proceeding |
| §11 step 5 | Candidate set spans all four §4 axes | `register.py coverage` prints the gaps; the gaps name the vein to read next |
| §11 step 6 | Predicted decomposition recorded first; plants recover | **Stop.** Measurement cannot guide extraction |

## Two things recorded before they can be rationalized

**The prediction (§6).** `docs/02-predicted-decomposition.md` holds what we expect the block structure and prerequisite DAG to be, including what our sources' own taxonomies would predict, written before the matrix runs and revised only by dated append. A mismatch between the field's inherited taxonomy and the measured blocks is the most valuable result available here, and it is unavailable if the prediction is written afterward.

**The abandonment condition (§9, "building for the eval you have").** Every convenient benchmark measures L0-ish competence, so a repertoire deliberately targeting L1–L3 will look bad on the dashboard for a long time. §9 says decide in advance what would make you abandon the design rather than the dashboard. Recorded in `docs/02` alongside the prediction, for the same reason.

## Contamination handling

If a held-out item is read by accident — by us or by a subagent — it is logged in `docs/00-heldout-partition.md` §4 as a contamination, not quietly reclassified. A contaminated held-out item is worthless as evidence; an *unrecorded* contamination is worse, because it makes a coverage claim false without anyone knowing it is false.
