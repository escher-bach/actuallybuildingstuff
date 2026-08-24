# Model and Representation Decision

> **Historical `0.1.0` record.** It documents the completed prior apparatus,
> not the current `0.2.0` architecture. The current boundary removes vision
> from the core and assigns training ownership to Transformers `Trainer`; see
> [`../../CORE-BOUNDARY.md`](../../CORE-BOUNDARY.md).

**Date:** 2026-08-24
**Status:** selected profile implemented; local CPU integration gates pass; one
two-T4 vertical-slice run is authorized and pending

## 1. Candid classification

The selected end-to-end learner is a **custom model**.

Its temporal backbone is not novel: it is the small, randomly initialized
Llama-style causal trajectory transformer released with
[ICRT](https://arxiv.org/html/2408.15980). The custom part is the narrow bridge
needed because no published small blank model found in this review combines all
of the following:

- long non-linguistic sensorimotor trajectories usable as physical prompts;
- variable observation and actuator cardinality without a robot-class head;
- continuous closed-loop action prediction;
- a representation that can later admit vision, entities, and public
  morphology; and
- a practical experimental profile for two T4 GPUs.

Exact ICRT satisfies the first and third items but compresses every timestep to
one state token and embeds one fixed-width action vector. CrossFormer, Octo,
AnyMorph, Astra, and RoboToken provide evidence for heterogeneous tokens or
readouts, but none supplies ICRT-style physical prompting under the other
constraints. Joining these ideas is an unvalidated architectural hypothesis.
It must not be described as “ICRT” without the qualifier **ICRT-derived**.

The novelty boundary is therefore:

| Surface | Owner | Decision |
|---|---|---|
| causal trajectory backbone, RoPE, RMSNorm, SwiGLU | maintained Llama implementation; ICRT evidence | reuse |
| interleaved observation/executed-action history | ICRT/PACT evidence | reuse |
| observation/action readout tokens | Octo/Astra/CrossFormer evidence | reuse the pattern |
| variable public channel/entity representation | AnyMorph/RoboToken evidence | reuse the pattern |
| schema-conditioned variable readouts joined to the ICRT core | this project | **custom gap fill** |
| worlds, teachers, verifiers, scheduling, provenance | this project | scientific novelty layer |

No custom transformer kernel, language model, visual foundation model, action
codebook, or diffusion implementation is selected.

## 2. Hard gate imposed by the end capability

The model is accepted only if the following capabilities are structurally
representable, even though a small blank checkpoint is not expected to perform
them:

1. condition closed-loop action on one or more complete sensorimotor
   demonstrations without a gradient update;
2. place two demonstrations in context and represent their composition;
3. append actual actions and outcomes during deployment so correction and
   recovery can use the lived trajectory;
4. vary sensor count, actuator count, ordering, and embodiment metadata without
   selecting a fixed robot-class action head;
5. accept video, numeric sensors, proprioception, goals, and demonstration
   trajectories without making natural language mandatory;
6. scale physical context and model capacity without changing the public token
   contract; and
7. support later few-gradient-step adaptation from the resulting checkpoint.

These are representation/architecture gates derived from the reported
[GEN-1.5](https://generalistai.com/blog/gen-1.5) target: 3–12 second physical
prompts inside 30 seconds of multimodal memory, prompt composition, human- and
simulation-to-robot transfer, closed-loop behavior, and rapid gradient-based
adaptation. Passing the gates does **not** establish that those capabilities
will emerge. It only avoids making them impossible by construction.

## 3. Selected experimental backbone

Use a randomly initialized Llama-style decoder through maintained PyTorch and
Hugging Face components, receiving `inputs_embeds` rather than vocabulary IDs:

```text
layers                 12
hidden width           384
attention heads        6
head width             64
SwiGLU intermediate    1024
normalization          RMSNorm
temporal position      RoPE
attention              ordinary causal self-attention
experimental budget    2,048 packed trajectory tokens
```

Without vocabulary embeddings, modality adapters, or heads, this core contains
approximately 21.24M learned parameters:

```text
12 * [4*384^2 + 3*384*1024 + 2*384] + 384
    = 21,243,264 parameters.
```

The maintained Hugging Face `LlamaModel` implementation retains a minimal
unused one-entry token embedding even when every forward call supplies
`inputs_embeds`. The implemented backbone therefore reports **21,243,648**
parameters rather than deleting a library-owned field for a 384-parameter
saving. With public-event adapters, continuous heads, and the random visual
stem/resampler, the implemented end-to-end model reports **22,147,985** learned
parameters. Both counts are asserted and recorded at runtime.

The width/layer/head pattern is the released ICRT `vits` configuration. The
ICRT paper's evaluated random core was wider (12 layers at width 768); the
released 384-wide configuration is implementation evidence, not published
performance evidence. Octo-Small independently uses 12 layers, width 384, and
6 heads in a 27M robotics-pretraining model.

The first response to memory pressure is microbatch reduction, gradient
accumulation, activation checkpointing, or a shorter packed context. There is
no pre-authorized width-192 fallback: changing width changes the experimental
learner and would require a recorded architecture decision.

Ordinary causal attention is deliberate. Astra and Octo's block/trajectory
masks are relevant evidence, but importing a special mask would add a custom
kernel and packing surface before it has demonstrated decision value. All
readouts occur after the public inputs they need, so a causal decoder can
represent the required mapping. Blockwise attention remains a matched
architecture ablation, not the default.

## 4. Canonical trajectory serialization

The model receives event segments in causal order. A completed transition is
represented schematically as:

```text
ENCOUNTER_SCHEMA*
BOUNDARY / CONDITION*

OBSERVATION_t*
ACTION_QUERY_t*          -> masked continuous action targets
ACTION_EXECUTED_t*
OUTCOME_QUERY_t*         -> optional public next-observation targets
FEEDBACK_(t+1)*
OBSERVATION_(t+1)*
...
```

`*` means zero or more tokens. The action-query outputs are not serialized as
facts. The actual command applied by the world is serialized afterward as an
executed-action token. Consequently, later decisions condition on what was
really executed and observed, not on a teacher-only or model-intended action.

A physical prompt is not a special language-like object. It is one or more
complete, boundary-delimited trajectories placed before the current rollout:

```text
[demonstration episode 1]
[demonstration episode 2]
[current observation and action queries]
```

Goal observations and desired outcomes use the same modality adapters as
ordinary observations with role `CONDITION`. A visual-only human demonstration
may omit action tokens; a sensorimotor demonstration includes them. Multiple
prompt episodes and reset boundaries fit the same serialization.

There is no atomic `WORLD_ID` token. A world is represented only by its public
interface schema, task evidence, boundaries, and observed interaction history.
Hidden family labels, dynamics parameters, generator seeds, and teacher state
are forbidden.

## 5. Universal token envelope

Every token has a role and an encounter-local public address:

```text
(role,
 encounter_id,
 episode_id,
 event_time,
 local_entity_or_channel_key,
 public_schema_attributes,
 normalized_payload,
 validity_mask)
```

The learned embedding is a sum or gated combination of:

- a small learned role embedding;
- a modality-specific continuous projector;
- deterministic encodings of encounter-local keys and timestamps;
- public bounds, units/control period, and relation attributes when supplied;
  and
- spatial position for image patches or public geometry.

Local keys bind the same public item across time but carry no global actuator
meaning. Keys and serialization order are permuted across encounters. Use a
deterministic Fourier/sinusoidal key encoding rather than a learned global
slot table, so an unseen local key is representable and no fixed robot
cardinality is installed.

All bounded scalar payloads are normalized from declared public bounds to
`[-1,1]`, with an explicit validity/out-of-range indicator. The tokenizer may
not estimate a hidden world parameter and insert it as a convenience feature.

## 6. Observation tokenization

### Numeric channels and public entities

- Initially, one token represents one publicly addressable scalar channel.
- A vector-valued entity may emit one token with a fixed public field schema or
  several scalar tokens; the world contract must declare which and preserve it
  across train/evaluation.
- Optional link, joint, relation, or morphology tokens are present only when
  the intended learner would actually receive that information.
- Missing channels are masked, not replaced by a hidden embodiment label.

### Images and video

The architecture includes a random visual path from `C0`, even though the
first proposed world does not exercise it:

1. a shallow convolutional patch stem, following Octo's transformer-first
   pattern, produces patch features;
2. public 2-D patch coordinates and an encounter-local camera key are added;
3. an attention resampler emits eight visual tokens per active view and frame;
4. those tokens enter the same causal trajectory stream as numeric
   observations.

All visual parameters are random and trainable. There is no DINO, CLIP,
ImageNet, language, CrossMAE, or other pretrained encoder. Eight resampler
slots are a project choice between ICRT's one-state-token compression and
RoboTTT's larger register set; it is a material unvalidated bottleneck and is
listed in the risk ledger. Raw-patch and 1/4/16-slot variants are high-value
representation ablations before visual pretraining is committed.

Video frame rate and action control rate are represented separately by public
timestamps. Long visual prompts may be temporally subsampled, as in BPP, while
the full executed action trajectory is chunked rather than pretending that
video arrives at 100 Hz.

## 7. Action tokenization and decoding

An action space is a set of publicly addressable scalar actuators, not one
fixed robot vector and not a text vocabulary.

For every active actuator, the tokenizer creates one `ACTION_QUERY` embedding
from:

```text
(ACTION_QUERY, local actuator key, public bounds,
 control period, optional public actuator attributes)
```

A single shared continuous head maps each query hidden state to a 16-sample
receding-horizon command chunk. Shorter targets use an explicit horizon mask.
Commands are normalized to the public actuator bounds. The number of query
tokens therefore changes with actuator cardinality, while the learned head is
shared; there is no `A_max`, padding-based robot vector, or robot-class head.

After the world applies a command, an `ACTION_EXECUTED` token with the same
local key and the actual applied chunk is appended. This supports saturation,
shielding, dropped commands, intervention, and actuator failure without
rewriting history.

The 16-step chunk is the ICRT starting prior and supports receding-horizon
closed-loop use; it is not a claim that all future control rates share one
natural chunk. The public control period and horizon mask make the time
semantics explicit. If a later world requires a different action distribution,
a standard diffusion/flow or FAST-style head can be evaluated behind the same
query interface, but it is not silently installed now.

For initial low-dimensional worlds, use a masked L1 action objective. It has a
unique public teacher target and avoids fitting an action codebook to the
narrow first data. Octo's diffusion results make a distributional head the
leading later alternative once multimodal demonstrations make point regression
measurably hedge.

## 8. Temporal and boundary encoding

RoPE position IDs advance by **event segment**, not by raw token count. Tokens
inside one observation, query, or executed-action segment share the segment's
temporal position; role and local-key features distinguish them. This prevents
adding a sensor from silently changing the apparent physical time of every
later event.

Episode-local time, real-valued elapsed time, reset flags, and control period
remain explicit payload features. Demonstrations and current rollouts use one
monotonic encounter event index for causal ordering, with `BOUNDARY` tokens
marking discontinuous resets. Packed examples receive an isolation mask so no
token can attend across unrelated encounters.

## 9. Auxiliary prediction interface

For numeric worlds, an optional `OUTCOME_QUERY` is constructed from each public
observation-channel key after the executed action. A shared scalar head predicts
the next normalized observation or delta using masked L1 loss. This gives the
same experience both inverse-control and action-conditioned forward-prediction
pressure, following PACT/RPT/SMART and the world-action direction represented
by LiLa-WAM.

This query sees only the public prefix. It never receives the latent transition
law. Pixel reconstruction or latent-video prediction is **not** selected at
`C0`; avoiding a collapsing learned visual target requires a separate decision
before a visual world uses this objective.

## 10. Strict blank-state contract

At `C0`:

- every backbone, adapter, resampler, role embedding, and prediction head is
  randomly initialized;
- no language tokenizer or language weights exist in the core interface;
- no visual, action, morphology, or world codebook has been pretrained;
- normalization rules and masks are deterministic representation priors, not
  learned knowledge; and
- no dormant module may contain downloaded learned weights.

Random initialization does not mean absence of bias. Causality, componentwise
tokens, local-key binding, visual resampling, continuous action chunks, and the
chosen context are strong installed priors. They are made explicit because
they persist across the checkpoint lineage.

## 11. Two-T4 experimentation and possible TPU scale-up

The primary experimental lineage runs on two T4 GPUs. Its purpose is rapid
iteration on world validity, tokenizer behavior, context use, transfer, and
scheduler overhead. The 21.24M core and 2,048-token budget are plausibility
estimates until an authorized hardware preflight measures memory, step time,
and useful target throughput. T4 training uses FP16 rather than BF16 under the
repository execution contract.

A later Kaggle TPU run is allowed as a **final, frozen run**, not as the normal
debug loop. The greater-than-two-hour queue reported by the user is charged to
outer-loop makespan. The existing `EXPERIMENT-EXECUTION-PLAN.md` currently
defines only the T4 implementation; TPU support requires a separate approved
extension after the T4 path is valid.

Two TPU paths must not be conflated:

1. **same-size continuation:** load a compatible T4 checkpoint and use the TPU
   for a longer session; this preserves the lineage;
2. **larger confirmation lineage:** initialize a wider/deeper model from blank
   with the same token ABI, attention topology, and frozen developmental
   protocol. A possible reference scale is the 12-layer, width-768 family used
   by ICRT/Octo-Base, but its exact size and context are not decided here.

A larger model cannot be called the next checkpoint of the small model without
a separately validated weight-growth procedure. By default it starts a new
blank lineage and replays the frozen protocol. Small-model world rankings and
stopping thresholds may fail to transfer with scale; that is an explicit
scientific risk, not free evidence.

The target TPU context should be selected from measured token traces. An
8,192-token candidate would make a 30-second compressed visual/action history
representable, but no fit or throughput claim is made before a TPU compile and
memory preflight. To avoid repeated queue cost, all logical world sessions,
online monitors, immutable checkpoints, and predeclared scheduler decisions
should execute inside one final allocation where possible.

## 12. Novelty and risk ledger

| Risk introduced by this choice | Why it matters | Cheapest decisive control |
|---|---|---|
| no paper validates the complete ICRT + variable-query join | positive results or failures may arise from the join, not the worlds | compare both models on a matched fixed-`d` slice using a paper-faithful pooled-state/fixed-vector ICRT adapter; test variable cardinality separately |
| query ordering may break actuator equivariance | a causal decoder can exploit serialization order | randomize order; jointly relabel; require bounded equivariance gap |
| per-actuator shared heads may under-coordinate coupled actions | independent readouts can hedge or conflict | coupled-effect held-out probe; compare one pooled-vector head only if it can change the architecture decision |
| eight-token visual resampler may discard control-relevant detail | target physical prompting is video-heavy | raw-patch and 1/4/16-slot ablation before admitting a visual checkpoint-producing world |
| L1 chunks cannot express multimodal action distributions | broad behavior data may have several valid actions | monitor conditional target ambiguity; promote a maintained diffusion/flow head only when measured |
| 2,048 experimental tokens may not teach long-context use | GEN-like behavior depends on physical memory | context-length scaling probe and context-removal/mismatch sentinels |
| small-to-large schedule transfer may fail | TPU run could replay the wrong world policy | at least two T4-scale profiles or tranche lengths before freezing a larger-lineage policy |
| tokenizer/packing bugs can create future or hidden-state leakage | structured worlds make shortcuts easy | exact-prefix oracle, forbidden-future perturbation, packed-boundary tests, seed/ID audit |
| custom mask or kernel drift | GPU/TPU behavior can diverge | default uses ordinary maintained causal attention; any special mask is a separately tested ablation |

These controls are front-loaded because the representation is the most
persistent choice. They are still bounded: the goal is to eliminate decisive
architectural ambiguity, not attach a large benchmark suite to every later
world decision.

## 13. Admission tests before checkpoint-producing training

The following must pass on generated transcripts and an untrained model graph:

1. **prefix causality:** changing any future executed action or observation
   leaves every earlier readout input and target unchanged;
2. **packed isolation:** changing another packed encounter cannot change the
   current encounter's outputs;
3. **public realizability:** a public-prefix oracle exactly reconstructs every
   teacher target in the declared expert support;
4. **joint relabeling:** jointly permuting sensor/actuator keys permutes targets
   and detokenized actions, without changing world semantics;
5. **order robustness:** alternative valid token orders do not change the
   detokenized public transcript and are represented in training;
6. **no teacher forcing side channel:** current action targets appear only
   after the corresponding action-query positions;
7. **boundary correctness:** reset and demonstration discontinuities cannot be
   mistaken for physical transitions;
8. **blank audit:** the checkpoint and dependency manifest contain no learned
   external weights; and
9. **round trip:** tokenization and detokenization preserve bounds, masks,
   keys, timestamps, and executed actions exactly within declared precision.

World-validity gates remain those in the repository's STEP 2 contracts. These
architecture tests address apparatus correctness; passing them does not make a
world scientifically valid or prove downstream transfer.

## 14. Decision summary

The selected learner is:

> a custom, fully random, ICRT-derived small causal trajectory transformer,
> using variable typed observation/action tokens and schema-conditioned shared
> continuous readouts, with no mandatory language boundary and no fixed robot
> action vector.

It is custom because the literature leaves a real interface/physical-prompt
gap. The custom surface is kept narrow and separately falsifiable. This choice
makes GEN-1.5-like physical prompting and many-interface generalization
representable; it does not claim that the initial scale, worlds, or available
compute are sufficient to produce them.
