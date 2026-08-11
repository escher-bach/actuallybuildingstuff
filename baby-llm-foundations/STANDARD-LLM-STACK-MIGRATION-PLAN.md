# Standard LLM Stack Migration Plan

**Status:** authoritative implementation path  
**Decision date:** 2026-08-12  
**Scope:** model, tokenizer artifact, training, checkpointing, preflight, learner-conditioned collection, fine-tuning, and RL

This document replaces the architecture, hand-written training-plumbing, and
synthetic capability-gate portions of `EXPERIMENT-EXECUTION-PLAN.md`. The world,
teacher, rendering, data-provenance, and scientific comparison specifications in
the other project documents remain in force.

## 1. Decision

The project will use a **published small-model architecture through a maintained
library implementation**, initialized from scratch:

- Architecture profile: **Pythia-70M / GPT-NeoX small-model body**.
- Implementation: `transformers.GPTNeoXForCausalLM`.
- Initialization: random weights from a repository-owned `GPTNeoXConfig`; no
  pretrained model weights are downloaded.
- Base training: Hugging Face `Trainer`, which uses Accelerate for distributed
  and mixed-precision execution.
- Fine-tuning: TRL `SFTTrainer` where its prompt/completion interface fits.
- Reinforcement learning: TRL (`GRPOTrainer`, `RLOOTrainer`, or another selected
  maintained trainer after the RL objective is frozen).
- Artifacts: standard Hugging Face `save_pretrained` directories containing the
  model config, safetensors weights, generation config, and tokenizer files.

The repository will **not** define attention, QKV projections, head reshaping,
RoPE, normalization layers, MLP blocks, residual topology, causal masking,
weight tying, DDP synchronization, AMP scaling, or checkpoint serialization.

The project is still allowed to own code that expresses the experiment itself:
the procedural world, privileged teacher, verifier, renderings, replay identity,
loss-label construction, evaluation, and the learner/world interaction adapter.

## 2. Why this model

Pythia was deliberately built as a controlled research suite for studying LLMs,
and its smallest architecture is genuinely small rather than an 8B architecture
arbitrarily shrunk for this project. The published Pythia-70M configuration has
six layers, width 512, eight attention heads, and 18,915,328 non-embedding
parameters. It is implemented directly by the maintained GPT-NeoX model in
Transformers.

This choice also remains recognizable in current research. Examples from 2026
include work using [Pythia-14M](https://arxiv.org/abs/2601.12703) and independently
trained [Pythia-70M](https://arxiv.org/abs/2605.24577) models. The original design
and research rationale are documented in the [Pythia paper](https://arxiv.org/abs/2304.01373)
and [model card](https://huggingface.co/EleutherAI/pythia-70m).

The alternatives remain rejected for this project:

| Alternative | Reason not selected |
| --- | --- |
| Hand-written PyTorch transformer | Makes ordinary transformer correctness and maintenance our responsibility. |
| A tiny configuration of an 8B-class Llama family | Technically possible, but it is not the purpose-built small research model requested here. |
| nanoGPT/minGPT | Appropriately small, but not the maintained pretraining-to-SFT-to-RL artifact ecosystem we want. |
| LitGPT | Strong pretraining/SFT option, but RL would introduce a conversion or second model stack. |
| OLMo-core or TorchTitan | Good research systems, but materially heavier than needed for a roughly 19M model on two T4s. |

## 3. Frozen model contract

The initial model configuration will be the published Pythia-70M body with the
project's 262-token vocabulary:

```python
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

config = GPTNeoXConfig(
    vocab_size=262,
    hidden_size=512,
    intermediate_size=2048,
    num_hidden_layers=6,
    num_attention_heads=8,
    max_position_embeddings=2048,
    hidden_act="gelu",
    initializer_range=0.02,
    layer_norm_eps=1e-5,
    rope_parameters={
        "rope_theta": 10_000,
        "partial_rotary_factor": 0.25,
    },
    use_parallel_residual=True,
    tie_word_embeddings=False,
    pad_token_id=256,
    bos_token_id=257,
    eos_token_id=258,
    use_cache=False,
)

model = GPTNeoXForCausalLM(config)  # random initialization
```

These values come from the published
[Pythia-70M config](https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json),
apart from the project vocabulary and special-token IDs.

Expected parameter count:

- non-embedding body: 18,915,328;
- input and untied output embeddings: `2 * 262 * 512 = 268,288`;
- total: **19,183,616 parameters**.

The project name must not call this a 70M-parameter model. It is a roughly 19.2M
model using the Pythia-70M architecture profile; Pythia's published 70M total is
mostly a consequence of its 50,304-token input and output embeddings.

Architecture changes are out of scope for Step 1. If a later experiment changes
the model body, that becomes a separately justified architecture experiment and
must not be smuggled into infrastructure work.

## 4. Standard tokenizer artifact

The byte vocabulary is experimental protocol, not a reason to maintain a Python
tokenizer implementation. Create one Hugging Face fast-tokenizer artifact using
the maintained `tokenizers` library:

- IDs `0..255`: exact UTF-8 byte values;
- `256`: PAD;
- `257`: BOS;
- `258`: EOS;
- `259`: OBS;
- `260`: ACTION;
- `261`: END_TURN.

The artifact will use the standard byte-level pre-tokenizer/decoder, a 256-entry
byte alphabet, no learned merges, and the six declared special tokens. Hugging
Face documents that its ByteLevel component represents all 256 byte values, and
`PreTrainedTokenizerFast` can wrap and save the serialized tokenizer. See the
[Tokenizers ByteLevel API](https://huggingface.co/docs/tokenizers/main/en/api/pre-tokenizers)
and [Transformers tokenizer API](https://huggingface.co/docs/transformers/main_classes/tokenizer).

This requires a small one-time artifact builder and parity tests against Rust;
it does not require a runtime tokenizer class owned by this repository. Rust may
continue to pack byte IDs directly, provided its vocabulary hash, all 256
byte-to-ID entries, all special tokens, and encoding of the canonical UTF-8
protocol corpus match the saved tokenizer artifact. Arbitrary invalid UTF-8 byte
sequences are not learner observations; if generated, they are malformed action
attempts and follow the learner-conditioned error path.

## 5. Training path

### 5.1 From-scratch base training

The model is instantiated from the local config, not loaded from a remote
checkpoint. Teacher-generated packed trajectories become ordinary pre-tokenized
causal-LM examples:

```python
labels = input_ids.clone()
labels[loss_mask == 0] = -100
```

`GPTNeoXForCausalLM` performs the causal shift and cross-entropy internally.
There is no project-owned `masked_next_token_loss`. The current TRL/Transformers
data contract explicitly supports pre-tokenized `input_ids` plus `labels`, with
`-100` marking excluded tokens; see the
[SFTTrainer data contract](https://huggingface.co/docs/trl/en/sft_trainer).

Use `transformers.Trainer` for this stage because the data is already tokenized
and the objective is causal LM with a sparse label mask. `TrainingArguments`
owns optimizer selection, scheduler, gradient accumulation, clipping, fp16,
DDP, logging, checkpoint cadence, and resume. Keep
`average_tokens_across_devices=True`, and retain fixed scored-token batch
construction where practical. Add an integration test proving that one- and
two-process loss normalization agree; do not reimplement DDP normalization.

Scientifically, this stage should be described precisely as **from-scratch
causal training on teacher-conditioned traces with action-token labels**. It is
the project's base-pretraining stage, but it is not ordinary full-text next-token
pretraining because observation tokens are context rather than targets.

### 5.2 Fine-tuning

Load the local base checkpoint with `AutoModelForCausalLM.from_pretrained` and
the tokenizer with `AutoTokenizer.from_pretrained`. Use TRL `SFTTrainer` for
prompt/completion-shaped fine-tuning. The learner-conditioned correction corpus
can also be consumed as pre-tokenized examples with `-100` context labels.

No conversion step, custom checkpoint loader, or alternate model implementation
is permitted between base training and fine-tuning.

### 5.3 Reinforcement learning

Load the same local checkpoint and tokenizer into TRL. TRL currently supplies
maintained SFT and online RL trainers and integrates with Transformers; its
[trainer index](https://huggingface.co/docs/trl/en/index) is the selection point
when the exact RL objective is frozen.

For the project's outcome-only RLVR baseline, TRL keeps the RL optimizer and
distributed trainer out of this repository. `GRPOTrainer` now supports an
`environment_factory`, multi-turn generation, environment-owned state, and
environment-owned reward; see the
[GRPO environment documentation](https://huggingface.co/docs/trl/main/grpo_trainer).
That high-level path assumes a chat template and tool-call protocol. Our byte
action protocol will therefore need either (a) a deliberately trained tool-call
surface plus a thin environment wrapper, or (b) TRL's lower-level custom rollout
hook while retaining TRL's optimization. This is an environment/protocol adapter,
not a custom RL algorithm. Whether GRPO, RLOO, or another maintained algorithm is
scientifically correct must be decided from the experimental objective, not from
implementation convenience.

## 6. Preflight is plumbing, not a transformer capability exam

The associative-retrieval gate is removed. It was trying to validate a custom
transformer's semantics; after this migration, model internals come from a
maintained implementation. Preflight must answer only whether our configuration,
data contract, runtime, and artifacts are wired correctly.

The replacement preflight is:

1. Instantiate the frozen config and assert class, config fields, parameter
   count, and random initialization provenance.
2. Exhaustively verify tokenizer parity for all 256 byte values and all special
   tokens.
3. Load one real Rust-generated batch, convert its mask to `labels == -100`, and
   run one finite forward/backward/optimizer step.
4. Run a very small fixed real-batch overfit diagnostic to catch reversed labels
   or broken masks. This is a diagnostic with a transparent loss trend, not a
   claim that the transformer must learn a synthetic research task before the
   actual run is allowed.
5. Save with `save_pretrained`, reload with `AutoModelForCausalLM`, and require
   matching evaluation logits for a fixed batch.
6. Launch the same smoke job on two T4 processes through Trainer/Accelerate and
   confirm both ranks finish, checkpoint once, and resume once.

No synthetic retrieval accuracy threshold, custom AMP retry loop, or repeated
multi-million-token preflight run remains.

## 7. Learner-conditioned dense supervision

### 7.1 Feasibility decision

**It is feasible with the selected maintained model stack.** It does not require
custom attention, generation, loss, DDP, AMP, or checkpoint code.

It does require project-specific environment orchestration because the objective
is unusual in one precise way:

1. the current model samples the action;
2. the world transitions using that actual learner action;
3. the privileged teacher labels the state the learner really reached;
4. optimization remains supervised cross-entropy on the correction tokens.

This is DAgger-like data collection: policy-induced states are labelled by an
expert, as in the original [Dataset Aggregation paper](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf).
It is not policy-gradient RL. TRL's environment support can be used for the later
outcome-RL stage, but GRPO must not be substituted for this dense supervised
objective because that would change the gradient paradigm.

### 7.2 What already exists

The Rust world already contains most of the domain logic:

- batched observations and live-episode indices;
- batched valid actions and state transitions;
- privileged teacher target sets;
- privileged terminal outcomes;
- canonical action rendering and parsing;
- `generate_learner_conditioned_attempts`;
- `pack_learner_conditioned_trajectory`;
- correct handling of malformed attempts, valid poor actions, and terminal
  irreversible mistakes.

The missing piece is exposure of the correct incremental learner-attempt
semantics through the Python binding. The current `Batch.step(Vec<i64>)` validates
the entire compact batch before changing any state. Therefore one malformed or
invalid generated action rejects the whole call, while the learner-conditioned
specification requires only that episode to remain unchanged and receive a
correction.

### 7.3 Required custom adapter

Add one narrow Rust/Python API, conceptually:

```text
Batch.step_attempts(rendered_action_texts, rendering)
    -> per-live-episode records {
         learner_text,
         parsed_action_or_error,
         accepted,
         observation_before,
         observation_after,
         preferred_corrections,
         terminal_outcome
       }
```

The method must process each live episode independently:

- malformed or invalid attempt: state unchanged, correction from that unchanged
  state;
- valid nonterminal attempt: state advances, correction from the actual successor
  state;
- terminal attempt: record outcome and emit no fictional recovery label.

Python then owns only the orchestration:

1. gather current observations for live episodes;
2. batch-generate one bounded action continuation using the standard model
   generation API;
3. pass rendered attempts to `step_attempts`;
4. append the returned actual history and teacher correction to rollout records;
5. convert correction spans to standard labels (`-100` on learner/context tokens);
6. train those records through the same maintained Trainer/model stack.

The learner action remains in context for subsequent states. The correction is a
new supervised continuation; it is not forced over the learner action's token
positions, which may have a different length. When the teacher returns multiple
equally preferred actions, the first implementation should select one
deterministically using the existing replay-stable policy while retaining the
full set in metadata. Set-valued sequence loss is explicitly deferred.

### 7.4 Complexity and recommended operating mode

| Version | Complexity | Included work |
| --- | --- | --- |
| Single-action diagnostic collector | Low to moderate | One generation, one world step, one correction record, contract tests. Roughly 1–2 engineering days. |
| Synchronous batched multi-turn collector | Moderate | Active-episode batching, bounded decoding, malformed actions, replay metadata, rollout buffer, metrics, and resume tests. Roughly 3–7 engineering days after the base stack works. |
| Asynchronous actor/learner service | High | Policy versioning, stale rollouts, queues, failure recovery, and distributed actors. Not required for Step 1. |

These are planning estimates, not commitments. The important distinction is
that the moderate custom work is an **environment/data adapter**, not a custom
LLM implementation.

Start synchronously: freeze a model snapshot, collect a bounded tranche of
learner-conditioned episodes, train on that tranche, then refresh the snapshot.
This preserves the intended learner-dependent state distribution closely enough
for selected recovery experiments without building an actor/learner system.
Record the policy checkpoint ID on every rollout. The snapshot interval becomes
an explicit experimental parameter.

Learner-conditioned collection remains selective because autoregressive world
interaction is much slower than offline teacher-trajectory generation. Most base
exposure remains teacher-conditioned; learner-conditioned tranches target
malformed actions, omissions, poor commitments, and recovery.

## 8. Ownership boundary after migration

| Concern | Owner |
| --- | --- |
| Transformer blocks, attention, RoPE, normalization, causal mask | Transformers `GPTNeoXForCausalLM` |
| Causal shift and cross-entropy | `GPTNeoXForCausalLM` using standard `labels` |
| Optimizer, scheduler, accumulation, fp16, DDP, resume | Transformers Trainer / Accelerate |
| SFT and RL algorithms | TRL |
| Model/tokenizer serialization | Hugging Face `save_pretrained` artifacts |
| Procedural world and transition rules | Project Rust crate |
| Privileged teacher and verifier | Project Rust crate |
| Rendering and action grammar | Project Rust crate |
| Which tokens are supervised | Project dataset/collator via `labels == -100` |
| Learner-conditioned world interaction | Narrow project adapter described above |
| Scientific metrics and comparisons | Project evaluation code |

This table is the guardrail against repeating the current over-engineering. A
new local implementation in a library-owned row requires an explicit decision
amending this document.

## 9. Migration sequence

### Phase 0 — Freeze the decision

- Mark this document as the authoritative implementation path.
- Discard the current unvalidated associative-retrieval/preflight retry draft.
- Pin one tested compatibility set of PyTorch, Transformers, Tokenizers,
  Accelerate, TRL, Safetensors, and Datasets for Kaggle T4 images. Do not use
  floating `latest` dependencies.

**Exit:** dependency lock and architecture config are reviewed; no training run.

### Phase 1 — Standard artifacts

- Add the frozen `GPTNeoXConfig` as a checked-in JSON artifact.
- Build and check in the 262-token fast-tokenizer artifact.
- Add exhaustive Rust/Python/Hugging Face token parity tests.
- Add a model factory of only a few lines: config load plus
  `AutoModelForCausalLM.from_config`.

**Exit:** parameter count is 19,183,616; all byte-to-ID entries and special IDs
match Rust; canonical UTF-8 protocol strings round-trip exactly.

### Phase 2 — Replace the custom model and preflight

- Remove `RMSNorm`, custom RoPE, `Attention`, `SwiGLU`, `Block`, and
  `Step1Transformer`.
- Convert existing packed masks to standard label tensors.
- Replace the custom loss with the model-returned loss.
- Implement the six-item plumbing preflight from section 6.

**Exit:** CPU smoke, single-T4 smoke, two-T4 smoke, artifact round-trip, and
resume all pass on real project data.

### Phase 3 — Move training onto Trainer

- Express current optimizer/schedule/batch settings as `TrainingArguments`.
- Use Trainer callbacks only for project metrics and time/token-budget stopping.
- Save standard model/tokenizer checkpoints and Trainer state.
- Preserve run manifests, config hashes, world replay keys, and data hashes as
  supplemental experiment metadata.

**Exit:** a short teacher-conditioned run resumes deterministically and is
loadable by Auto classes without importing project model code.

### Phase 4 — Teacher-conditioned base baseline

- Run the small real baseline first.
- Validate learning on held-out world seeds and rendering transfer.
- Separate capability metrics from infrastructure success; a poor scientific
  result is not automatically an infrastructure failure.

**Exit:** one trustworthy from-scratch base checkpoint and evaluation report.

### Phase 5 — Learner-conditioned adapter

- Add and test `step_attempts` at the Rust/Python boundary.
- Implement the single-action diagnostic collector.
- Extend it to synchronous batched multi-turn tranches.
- Add rollout-policy version, malformed-action, accepted/rejected, correction,
  and wall-time metrics.
- Train the returned standard labelled examples with the existing Trainer.

**Exit:** tests prove that actual learner actions determine successor states,
malformed attempts leave only their own state unchanged, terminal mistakes do
not receive false recovery targets, and correction labels alone carry loss.

### Phase 6 — Standard SFT and RL interoperability

- Load the same artifact into `SFTTrainer` and complete a tiny fine-tuning smoke.
- Wrap the Rust world in the selected TRL environment or rollout contract for
  outcome-only RLVR; explicitly test the byte-action parsing boundary.
- Complete a tiny RL smoke before any comparative run.

**Exit:** one model artifact moves from base training to SFT to RL with no model
conversion and no alternate transformer implementation.

## 10. Acceptance criteria

The migration is complete only when all of the following are true:

- No project file defines transformer attention, positional rotation,
  normalization, MLP blocks, or causal masks.
- The checked-in model config has `model_type: "gpt_neox"` and the frozen fields
  in section 3.
- The runtime parameter count is exactly 19,183,616.
- Training starts from random initialization and records the seed and config
  hash; no pretrained weights are fetched.
- The tokenizer is loadable by `AutoTokenizer`, the model by
  `AutoModelForCausalLM`, and both are stored in each retained checkpoint.
- A real masked batch uses only standard `labels` with `-100` exclusions.
- Single-process and two-process loss accounting agree within a stated numeric
  tolerance.
- Save/reload preserves fixed-batch logits within the dtype tolerance.
- The base checkpoint completes both an SFT smoke and an RL trainer smoke.
- Learner-conditioned tests enforce the three transition cases in section 7.3.
- No associative-retrieval capability threshold can block an experiment run.

## 11. Explicit non-goals

- No architecture research in Step 1.
- No recreation of GPT-NeoX/Pythia internals.
- No large pretrained model and no distillation from one unless later approved
  as a separate experiment.
- No asynchronous rollout service in the first learner-conditioned version.
- No set-valued custom sequence loss in the first version.
- No use of an RL trainer to imitate the learner-conditioned supervised
  objective.
- No capability benchmark presented as an infrastructure preflight.

## 12. Primary references

- [Pythia paper](https://arxiv.org/abs/2304.01373)
- [Pythia-70M model card and parameter table](https://huggingface.co/EleutherAI/pythia-70m)
- [Published Pythia-70M configuration](https://huggingface.co/EleutherAI/pythia-70m/blob/main/config.json)
- [Transformers GPT-NeoX implementation](https://huggingface.co/docs/transformers/main/en/model_doc/gpt_neox)
- [Transformers Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer)
- [Accelerate](https://huggingface.co/docs/accelerate/index)
- [TRL trainer stack](https://huggingface.co/docs/trl/en/index)
- [TRL SFT data contract](https://huggingface.co/docs/trl/en/sft_trainer)
- [TRL stateful-environment GRPO](https://huggingface.co/docs/trl/main/grpo_trainer)
- [Hugging Face Tokenizers ByteLevel API](https://huggingface.co/docs/tokenizers/main/en/api/pre-tokenizers)
- [DAgger / Dataset Aggregation](https://proceedings.mlr.press/v15/ross11a/ross11a.pdf)
