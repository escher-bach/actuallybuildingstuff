# GPU Vertical-Slice Result and Next Recursive Decision

> **Historical `0.1.0` record.** It documents the completed prior apparatus,
> not the current `0.2.0` architecture. The current boundary removes vision
> from the core and assigns training ownership to Transformers `Trainer`; see
> [`../../CORE-BOUNDARY.md`](../../CORE-BOUNDARY.md).

**Date:** 2026-08-24
**Status:** architecture/ABI gate passed; `c1-start-candidate` retained and
audited; first developmental world is not yet mastered

## Decision

Retain the selected learner and physical-event ABI as the provisional STEP 2
starting bias, and retain `W_calibrated_monomial[d=1..4]` as the first admitted
world family. The resulting checkpoint is a valid next scheduler state, but it
is only a bounded training start—not a completed first developmental session
and not transfer evidence.

At this new state, the next preferred action is `TRAIN_OLD` with cheap
world-native monitoring, followed by the smallest evaluation that can test
calibration dependence and relabeling robustness. Do not admit the harder
coupled/active-identification world yet: held-out action error remains high and
closed-loop success is zero. This is the same recursive choice rule that will
apply after every later checkpoint; the blank root has no special privilege.

## Broader capability claim

The end target remains GEN-1.5-like physical generalization: infer tasks from
physical demonstrations, compose prompts, act and recover closed loop, transfer
across realizations/embodiments, and adapt rapidly when context is insufficient.
The first world addresses only the earliest control/interface trunk:

- public schema grounding and encounter-local key binding;
- inference of action effects from prior interventions;
- action-conditioned next-state prediction;
- variable-cardinality, goal-conditioned bounded control; and
- use of causal sensorimotor history rather than a language-only boundary.

It does not yet teach coupled effects, learner-selected probes, objects,
contact, perception, demonstration-defined tasks, or cross-embodiment transfer.

## Learner and representation that passed the apparatus gate

The end-to-end learner is custom; that gap is explicit. Its maintained core is
Hugging Face `LlamaModel` with 12 layers, width 384, 6 full-attention heads,
SwiGLU, RMSNorm, RoPE, and 21,243,648 backbone parameters. The complete random
learner has 22,147,985 trainable parameters. Project-owned code adds typed
continuous event adapters, deterministic encounter-local key encoding, shared
16-step action and scalar outcome readouts, and an untrained random visual
patch/resampler path.

The learner sees only public physical events:

```text
role + local public key + event position + 8-float normalized payload
```

Roles distinguish schema, boundaries, conditions/goals, observations, action
queries, executed actions, outcome queries, and feedback. Actions are queried
per public actuator and normalized to declared bounds; actual executed actions
and outcomes re-enter history. There is no hidden world-ID token, mandatory
natural-language tokenization, fixed robot vector, or pretrained component.

## World validity and non-degeneracy evidence

Rust owns deterministic process generation, latent transition execution,
public serialization, public-prefix oracle reconstruction, verifier controls,
and batched online rollout state. Python receives only learner tensors; the
privileged oracle is evaluator-only.

The Kaggle sweep checked 4,096 processes, exactly 1,024 at each dimension
`d=1..4`. Transcript lengths were 16–120 tokens against the fixed 192-token
batch length, so no trajectory was truncated. The public oracle reconstructed
the hidden signed permutation/gains with maximum error
`5.96e-8`, and its closed-loop control succeeded on 100% of the 64 held-out
rollouts with mean terminal error `1.86e-9`. The model's 0% result therefore
cannot be attributed to an unsolvable teacher or evaluator.

The correctness suite also establishes deterministic replay, safe non-clipping
calibration, no action-target leakage into query payloads, action application
through Rust, and a failing zero-action control. More demanding history
mismatch, joint relabeling, and token-order sentinels remain required before a
source-competence claim.

## Exact two-T4 result

The audited run is Kaggle
`aniruddhavarma/step2-architecture-world-1d965c5/1`, exact Git commit
`1d965c5fb2be8a39d88e41a8f144d9cd2f5cee31`, on two Tesla T4s.

- Disposable exact-model gate: loss `0.6356 -> 0.1297` on a fixed real Rust
  cohort (final/initial `0.204`); 126 successful FP16 optimizer steps after two
  safely skipped overflow steps.
- The diagnostic model was discarded before the lineage reset.
- Distributed save/mutate/load restored identical three-part parameter
  checksums on both ranks at update 4.
- Fresh lineage: 256 updates, 4,096 newly generated encounters, 256 successful
  global optimizer steps, scheduler epoch 256, final learning rate zero.
- Training-batch loss samples changed from `0.8558` to `0.5921`; action loss
  from `0.6915` to `0.5398`; outcome loss from `0.3286` to `0.1046`.
- Held-out teacher-forced L1: action `0.5432`, outcome `0.1046`.
- Model closed loop: 0/64 successes, mean terminal error `0.3430`.
- Oracle control: 64/64 successes.

This passes the architecture/integration claim and shows early predictive and
action-loss learning. It fails any claim of closed-loop source competence. No
claim about transfer to a future world is licensed.

## Inner- and outer-loop overhead evidence

On the actual Kaggle CPU, the Rust world/oracle/serializer path produced a
median 2,512 episodes/s and 166,032 unpadded tokens/s; existing-batch
tensorization processed the equivalent of 1,642 episodes/s. The full GPU
training process, including disposable gate, reset, 256-update start, final
evaluation, and checkpointing, reported 44.8 seconds. The enclosing remote
phases took about 201 seconds: roughly 52 seconds for install/build, 52 seconds
for correctness tests, 36 seconds for the CPU benchmark, and 62 seconds for the
GPU phase. The world producer is therefore not the observed critical path at
this scale.

Four prior remote attempts are retained because they measured real outer-loop
apparatus cost and exposed, in order: absent Cargo in the Kaggle image,
unscheduled FP16 diagnostic optimization, a missing post-save distributed
rendezvous, and per-rank scheduler stepping. The final implementation pins the
Rust toolchain, preserves gate progress, rendezvous after save, and asserts one
scheduler step per successful global optimizer update.

Static-dataset training remains an inner-loop special case in the formal cost
model. Online world-native evaluation remains an open option: a world can emit
cheap score/efficiency telemetry during training for a predeclared scheduler
decision. This run did not adopt a universal scalar or alter direction; its
performance record remains a vector.

## Retained artifact and audit

The remote-only recovery payload is
`step2-results/checkpoints/c1-start-candidate`, size 347,399,320 bytes, tree
SHA-256 `dd7fa5391f5c0f30fa61050a901a27261724322f3132a3b999a6e46c29243c9b`.
The portable model SHA-256 is
`07c9f0a5cccf1810785f96cc2a903955bb3460b92f4c16a3f871774eb25ccee3`.
It remains on Kaggle; no checkpoint was downloaded.

The compact receipt verified 17 downloaded evidence files against the remote
manifest. See the tracked audit directory
`step2/audit/runs/step2-architecture-world-1d965c5/`.

## Next-session boundary

Before another training launch, freeze a bounded `TRAIN_OLD` session with:

1. a fresh continuation optimizer/schedule while preserving these weights;
2. cheap held-out action/outcome and closed-loop telemetry inside the session;
3. one combined calibration-removal/mismatch and relabel/order sentinel at the
   decision boundary rather than ten separate launches;
4. explicit resource and plateau limits; and
5. a matched blank/static-mixture transfer comparator only when a descendant
   world decision actually requires transfer evidence.

TPU remains a later full-run option, not an experimentation loop, and no TPU run
is authorized by this result.
