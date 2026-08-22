# Provenance

## What was inherited from the predecessor project, and what was not

This repository was split from `baby-llm-foundations` after that project's
STEP 1 closed as a failed scientific design. The split is deliberate: the two
are separate research programmes with separate histories, and this repository
starts a fresh one.

## Carried over

**Theory.** `PROCESS-AND-RENDERING.md`, `TRAINING-DYNAMICS.md`,
`WORLD-CORRESPONDENCE.md`, `GRADIENT-PARADIGM.md`, `WORLD-BACKEND.md`,
`RELATIONAL-WORLDS.md`, `principled_developmental_worlds.md`,
`THEORY-PHASE.md`. These govern both programmes.
`WORLD-CORRESPONDENCE.md` is superseded as a validity criterion by
[step2/WORLD-VALIDITY.md](step2/WORLD-VALIDITY.md) but retained for its
representation obligations.

**Lessons.** `STEP-1-WORLD-0.1-CLOSURE.md` and
`STEP-1-EXPERIMENT-DESIGN-FAILURE.md` record the two coupled failures that
closed the predecessor world, both of which the validity contract now gates
against. `STEP-1.md` is retained because it documents what the reused executor
actually implements.

**Apparatus.** `step1/` — the Rust executable-world boundary, the Python
binding and harness, configurations, and the Kaggle runner. Reused under the
transfer rules in the closure document: the typed transition, parsing,
verification, replay, and batch-generation boundaries are reusable; the
`world-0.1.0` semantics inside them are frozen reference and are not inherited.

**Governance.** `AGENTS.md`, `EXPERIMENT-EXECUTION-PLAN.md`,
`STANDARD-LLM-STACK-MIGRATION-PLAN.md`.

## Deliberately not carried over

The predecessor's experiment record: `STEP-1-SYNTHESIS.md`,
`RLVR-STAGE-PLAN.md`, `RLVR-STAGE-REPORT.md`,
`RENDERING-B-TERMINAL-TRANSFER-PLAN.md`,
`RENDERING-B-TERMINAL-TRANSFER-REPORT.md`,
`RENDERING-B-TRANSFER-SEED0-REPORT.md`, `TARGET-SHUFFLED-CONTROL-REPORT.md`,
`legacy/`, `results/`, and `step1/audit/runs/`.

These are measurements of `world-0.1.0` and belong to the predecessor's record,
where they remain. They are not part of this programme's evidence.

### Dangling references

`THEORY-PHASE.md`, `STEP-1.md`, and `STEP-1-WORLD-0.1-CLOSURE.md` cite some of
the documents above. Those citations are left intact rather than rewritten:
these are historical records, and editing them to conceal the boundary would
falsify them. Follow such a reference to `baby-llm-foundations` if the
underlying measurement matters.

## Rule

Reuse means preserving a tested boundary or a standard artifact. It does not
mean carrying forward `world-0.1.0` semantics, teacher policies, targets,
success thresholds, or result claims. Anything from the predecessor that is
promoted into this programme must earn its place under the acceptance gates in
[step2/WORLD-VALIDITY.md §9](step2/WORLD-VALIDITY.md).
