# STEP 1 / world-0.1 closure

**Status:** closed as a failed scientific design, 2026-08-20

**World identifier:** `world-0.1.0`

**Scope:** the six-hypothesis, five-probe, binary-evidence STEP 1 family

**Purpose:** preserve the factual record and reusable apparatus while preventing
the invalidated interpretation from propagating into the next stage.

This is a compact internal closure record. The detailed diagnosis in
[STEP-1-EXPERIMENT-DESIGN-FAILURE.md](STEP-1-EXPERIMENT-DESIGN-FAILURE.md) is
authoritative for the failure. It supersedes the scientific completion verdict
and strongest scientific claims in
[STEP-1-SYNTHESIS.md](STEP-1-SYNTHESIS.md), but it does not erase completed runs,
their metrics, or their provenance.

## Closure decision

`world-0.1.0` is closed. It will not receive another scientific training run,
optimizer comparison, or repaired rerun. Its generator, concrete semantics,
teacher policies, and result claims are retained as a failed-design case study,
not as the foundation for a corrected world.

The failure is not a negative answer to the project's larger scientific
question. The deployed world did not validly pose the intended question at the
learner's information boundary. STEP 1 therefore closes with two outputs:

1. a documented scientific failure; and
2. a working experimental apparatus that can be reused after a new world has
   independently earned scientific validity.

## The two coupled failures

1. **The generator exposed stable cause semantics.** The binary evidence
   sampler's hypothesis-conditioned rewrite made `cause_1` and `cause_2`
   publicly special. A small observation-only policy consequently explains the
   observed 40--45% band without learning the intended per-instance
   hypothesis-elimination process.
2. **The targets were not learner-realizable.** The dense teacher used the
   latent truth and complete instance; the later "truth-blind" expert still used
   the hidden evidence table, exact untried costs, and horizon. Its 97.7% score
   was a table-aware oracle value, not a ceiling for the serialized learner
   prefix. Teacher forcing further placed privileged earlier actions in later
   prefixes, exaggerating the train/evaluation mismatch.

These failures interact: the intended policy was unavailable from public
information while the buggy generator supplied a different, learnable policy.
Changing the learner, optimizer, supervision regime, or decoder while retaining
both properties could not validate the intended mechanism.

## Scientific record

| Status | What the record now supports |
| --- | --- |
| **Invalid** | The 97.7% table-aware value as the learner ceiling; the claim that the truth-blind expert was publicly followable; teacher success as proof that the public task was almost solvable; the conclusion that demonstrator followability had been ruled out. |
| **Not established** | Dense 41.1% as acquisition of hypothesis elimination; the 40--45% band as a model capacity or optimization ceiling; the intended information-theoretic identification argument. |
| **Weakened** | Cross-rendering transfer remains a change in acquisition on the deployed shortcut-bearing family. It does not isolate transfer of the intended process. |
| **Retained as measurements** | All recorded curves and scores for the exact deployed family, including dense versus shuffled, teacher and table-aware-oracle values when labeled by their actual information states, transfer curves, structural evaluations, and RLVR results. |
| **Retained as operational findings** | Teacher-forced loss can diverge from closed-loop control; RLVR improved sampled action legality without improving greedy success in this interface; cold RLVR did not bootstrap the ungrounded byte interface; parser, executor, replay, artifact, throughput, and provenance audits worked. |

The strongest admissible scientific summary is the revised conclusion in
[the failure audit](STEP-1-EXPERIMENT-DESIGN-FAILURE.md#6-what-the-earlier-experiments-now-mean):
dense training learned a state-dependent policy for the deployed generated
family, but the evidence does not establish acquisition of the intended
hypothesis-elimination procedure.

## Apparatus retained for transfer

Reuse means preserving a tested boundary or standard artifact, not carrying
forward `world-0.1.0` semantics by default.

| Retained asset | Transfer rule |
| --- | --- |
| Hugging Face model, tokenizer, checkpoint, and Trainer/Accelerate path | Keep as the standard LLM stack, with TRL owning later standard SFT/RL algorithms. Library-owned concerns remain library-owned under [STANDARD-LLM-STACK-MIGRATION-PLAN.md](STANDARD-LLM-STACK-MIGRATION-PLAN.md#8-ownership-boundary-after-migration). |
| Rust executable-world boundary and Python binding | Reuse the typed transition, parsing, verification, replay, and batch-generation boundaries. A new world supplies new semantics; it must not inherit the old generator or targets merely to fit the interface. |
| Teacher-conditioned data pipeline | Reuse framing, masking, packing, deterministic metadata, and standard labelled-example production. Teacher policy and target realizability must be re-authorized per world. |
| Learner-conditioned collection pipeline | Reuse actual-action execution, correction records, failure handling, policy-version metadata, and bounded synchronous collection. The correcting teacher remains world-specific. |
| Closed-loop evaluator and measurement plumbing | Reuse action parsing, legality/protocol accounting, deterministic seed bands, artifact reports, and paired comparisons. Success criteria, ceilings, and semantic metrics must be newly specified. |
| Rendering and action interfaces | Reuse the separation between typed semantics and textual presentation. Freeze the concrete STEP 1 vocabularies/templates; future renderings require new nuisance and information-boundary audits. |
| RLVR environment integration | Retain as an optional standard-stack adapter and as evidence about the old interface. Do not promote the old reward, grammar, or observed behavior into a general learning claim. |
| Kaggle repository runner and audit workflow | Preserve the implementation for reuse. [EXPERIMENT-EXECUTION-PLAN.md](EXPERIMENT-EXECUTION-PLAN.md) remains authoritative for retention, retrieval, and receipts of existing STEP 1 runs; a later step must supply its own scientific launch contract. No GPU run occurs without explicit authorization. |
| CPU audit tools added after failure | Retain the public-information ceiling, finite-support, and raw-fingerprint diagnostics as regression instruments. Carry the failure audit's target-realizability and permutation checks forward as requirements to implement for a new world. Passing old tools is necessary evidence, not automatic validation. |

No speculative framework extraction is required at closure. Shared code should
remain a thin adapter around maintained libraries and stable executable-world
boundaries. Refactor only when the next independently specified world reveals a
real common interface.

## Frozen and retired world-0.1 material

The following remain reproducible historical material but are retired from
future scientific inheritance:

- the evidence-table generator and its acceptance constraints;
- the six-cause/five-probe binary world semantics and configured distributions;
- the dense teacher and table-aware truth-blind policy as demonstrators or
  learner ceilings;
- the concrete Rendering A/B vocabularies, orderings, and action grammar as
  evidence of nuisance-free semantics;
- world-specific configurations, datasets, checkpoints, reports, reference
  values, and success thresholds; and
- claims that the transfer result identifies the intended process or validates
  an open-ended developmental-world programme.

The code may be read, replayed, and used for failure regression tests. It must
not be silently corrected and presented under the same world or run identity.

## Preservation, versioning, and archival rules

1. Preserve existing run metrics, reports, configurations, code SHAs, replay
   keys, hashes, and receipts as factual records of `world-0.1.0`.
2. Do not regenerate an existing run under its old run ID, overwrite an
   accepted artifact slug, or relabel a privileged score as public. Any
   materially changed generator, observation boundary, renderer, teacher, or
   evaluator requires a new world version and new run identities.
3. Keep heavyweight checkpoints and recovery payloads on Kaggle. Keep only
   compact verified analysis artifacts and tracked receipts in the repository,
   following the retention contract in
   [EXPERIMENT-EXECUTION-PLAN.md](EXPERIMENT-EXECUTION-PLAN.md#13-filesystem-retention-and-result-bundles).
4. Backfill missing legacy receipts when this is a cheap metadata operation;
   record missing evidence explicitly. Do not rerun models merely to make the
   archive cosmetically complete.
5. Treat this closure and the failure audit as the interpretation layer over
   the earlier synthesis. Historical prose and metrics may remain for audit,
   but they may not be cited without this correction.
6. Preserve STEP 1 in place until a new validated world requires a narrow,
   tested extraction. Migration must keep standard artifact formats and the
   ownership boundary intact; copying world-specific behavior into a new step
   is not transfer.

## Handoff boundary

The next stage may define a world-validity contract and, after independently
motivated examples exist, candidate world operators. **Neither the validity
contract nor the world operators are designed in this document.** Their
requirements, semantics, correspondence claims, and acceptance gates are
explicitly deferred to STEP 2.

Until that contract exists, the proper next action is preservation and design,
not a repaired `world-0.1.0` run.
