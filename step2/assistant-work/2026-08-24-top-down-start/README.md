# Assistant Work: STEP 2 Top-Down Starting Point

**Date:** 2026-08-24
**Status:** architecture/world vertical slice passed on two T4s; an audited
bounded `c1-start-candidate` exists; source-world competence and transfer remain
unestablished
**Scope:** research, implementation, and authorized T4 validation; no TPU run

This folder is deliberately separate from user-authored STEP 2 contracts.

## Decision snapshot

- **End demonstration:** GEN-1.5-like generalization—one/few-shot physical
  prompting, prompt composition, closed-loop recovery, cross-realization and
  cross-embodiment transfer, and very rapid gradient adaptation. This is a
  north star, not a claim to reproduce Generalist's proprietary system.
- **Process:** the same recursive choice is made at every checkpoint among a
  new world, an old world, a static mixture, evaluation, world admission,
  architecture migration, or stopping. `C0` is different only because its
  weights are random.
- **Model:** a **custom**, fully random, ICRT-derived causal trajectory
  transformer. The standard core is 12 layers, width 384, 6 heads, about
  21.24M parameters, and a 2,048-token experimental budget. The narrow custom
  bridge supplies variable typed observations/actions and schema-conditioned
  shared continuous readouts.
- **Representation:** worlds have no hidden identity token. Public schema,
  boundaries, observations, conditions/demonstrations, action queries, actual
  executed actions, feedback, and outcomes form the trajectory. Natural
  language and a fixed robot action vector are not mandatory boundaries.
- **First action:** one bounded monitored session on
  `W_calibrated_monomial[d=1..4]`, a public-calibration, signed-permutation
  action-effect family. The scalar case is retained inside it but is too narrow
  to produce the first checkpoint alone.
- **Measurement:** world performance and candidate plausibility remain vectors.
  A world-native scalar may be emitted online for a declared scheduling
  decision, but does not fully map the world or become universally comparable.
- **Compute:** primary iteration is on two T4 GPUs. A larger or longer final
  Kaggle TPU run is allowed only after the architecture/interface, worlds,
  metrics, and scheduler are frozen; its >2-hour queue makes it unsuitable for
  experimentation. A larger model begins a new blank confirmation lineage
  unless a weight-growth method is separately validated.
- **Efficiency:** static dataset training is an inner-loop special case. The
  adaptive process must beat a matched fixed multi-world pretraining action
  after all generation, interaction, evaluation, checkpoint, launch/queue, and
  decision overhead.

## Recursive process

```text
scheduler state S_n
  = checkpoint + evidence/uncertainty + capability hypothesis
    + admitted worlds/costs + lineage/retention + remaining resources

choose bounded action
  = NEW | OLD | STATIC MIXTURE | EVALUATE | ADMIT | MIGRATE | STOP

observe checkpoint/evidence/cost result
  -> construct S_(n+1)
  -> make the same choice again
```

No first world has guaranteed descendants or permanent replay share. Online
monitoring can end a bounded session early and return to the same decision
loop; it does not replace independent validity or transfer evidence.

## Documents

- [INITIAL-DECISION.md](INITIAL-DECISION.md) — compact complete decision and
  decided-versus-empirical boundary.
- [CAPABILITY-DECOMPOSITION.md](CAPABILITY-DECOMPOSITION.md) — GEN-like target,
  capability forest, information contracts, and falsifying probes.
- [MODEL-AND-REPRESENTATION.md](MODEL-AND-REPRESENTATION.md) — candid custom
  boundary, exact small core, token ABI, action/observation scheme, scale plan,
  and novelty-risk ledger.
- [C0-FIRST-ACTION.md](C0-FIRST-ACTION.md) — scalar audit, revised `d=1..4`
  first family, plausibility criteria, performance vector, and validity gates.
- [RECURSIVE-CHECKPOINT-POLICY.md](RECURSIVE-CHECKPOINT-POLICY.md) — canonical
  recursive state, actions, transition, objective, and value of information.
- [OVERHEAD-MODEL.md](OVERHEAD-MODEL.md) — inner/outer work, span, Roofline,
  communication, pipeline, monitor, queue, and static-mixture comparison.
- [REPEATED-DEVELOPMENTAL-LOOP.md](REPEATED-DEVELOPMENTAL-LOOP.md) — operational
  scheduler, measurement tiers, world lifecycle, and ledger; subordinate to
  the canonical recursive policy where wording differs.
- [EVIDENCE-AND-ALTERNATIVES.md](EVIDENCE-AND-ALTERNATIVES.md) — primary-source
  architecture, representation, physical-prompt, curriculum, and systems
  evidence, including the exact literature gap.
- [FIRST-VERTICAL-SLICE.md](FIRST-VERTICAL-SLICE.md) — apparatus-first test,
  Rust/Python ownership boundary, local CPU evidence, GPU gate, and checkpoint
  classification.
- [GPU-VERTICAL-SLICE-RESULT.md](GPU-VERTICAL-SLICE-RESULT.md) — audited two-T4
  result, overhead measurements, retained remote artifact, scientific boundary,
  and next recursive action.

## Boundary from STEP 1

This is the STEP 2/Pivot learner, not the Pythia/natural-language STEP 1
apparatus. Pythia, byte tokenization, inherited language weights, and a
mandatory linguistic action/goal boundary are not candidates. STEP 1 still
supplies reusable lessons about public-information realizability, leakage,
teacher forcing, permutation controls, immutable artifacts, provenance, and
Kaggle execution discipline.
