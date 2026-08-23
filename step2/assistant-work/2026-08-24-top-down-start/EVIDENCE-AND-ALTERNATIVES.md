# Evidence and Alternatives

**Research cutoff:** 2026-08-24
**Source policy:** primary papers, official repositories, and official research
pages. Company reports and recent arXiv preprints are target or directional
evidence, not independently reproduced results.

This document separates three choices that are easy to conflate:

1. the **recursive developmental process** applied at every checkpoint;
2. the **architecture and token ABI**, which are persistent learner priors; and
3. the **training action selected at `S0`**, which is only the first ordinary
   application of the recursive policy.

No source supplies the whole system needed here. In particular, no released,
small, blank robotics-pretraining model combines long physical prompting with
open-ended sensor and actuator cardinality. The selected end-to-end learner is
therefore **custom**. Its causal core is deliberately conventional and taken
from ICRT/Llama; the smallest project-owned surface is the variable physical
token interface and schema-conditioned readout. That join is an unvalidated
research hypothesis and is recorded as such, not disguised as standard work.

## 1. Standard vocabulary

| Project term | Standard research vocabulary |
|---|---|
| world | controlled process, environment/POMDP family, task distribution, or dataset-generating process |
| model performance in a world | competence or learning curve on held-out process instances |
| new world | new task/dynamics/domain distribution or curriculum arm |
| old world | replay, rehearsal, or revisited curriculum arm |
| learn a world | bounded policy/model update on experience from that distribution |
| physical prompt | sensorimotor trajectory demonstration / in-context demonstration |
| infer how a world works | system identification / in-context world modeling |
| learn a later capability faster | positive forward transfer / improved sample efficiency |
| decide what to train next | automated curriculum learning / task selection / metareasoning |
| repeated checkpoint choice | budgeted adaptive curriculum / sequential decision over training interventions |

“World-action model” in LiLa-WAM is that paper's model class and is not the
project's definition of a world.

## 2. End-capability evidence and decomposition

### 2.1 GEN-1.5-like generalization is the hard architectural gate

The [GEN-1.5 report](https://generalistai.com/blog/gen-1.5) describes a large
multimodal model with roughly 30 seconds of memory, multimodal physical context,
and 100 Hz action trajectories. It reports one-shot behavior from 3–12 second
physical demonstrations, compositional prompting, simulation-to-real and
human-to-robot transfer, and rapid one-to-ten-gradient-step adaptation. Its
architecture and training stack are proprietary, and the pretraining scale is
far beyond this project. These are therefore **target capabilities**, not an
implementable recipe or a comparable result.

The related [thousand-hands report](https://generalistai.com/blog/towards-machines-with-a-thousand-hands)
emphasizes transfer across many end effectors and even embodiment changes during
a task. Taken together, these reports make four properties hard gates for this
project's permanent interface:

- more than one demonstration trajectory can remain in causal context;
- executed actions and subsequent observations can be represented natively;
- sensor and actuator cardinality is not frozen to the first robot; and
- language, a fixed robot vector, and a single visual realization are not
  mandatory semantic boundaries.

The project is not claiming GEN-1.5 performance at small scale. The claim under
test is narrower: do early recursively selected worlds install prerequisites
that make this *kind* of later transfer more plausible without foreclosing it
in the architecture?

### 2.2 In-context sensorimotor task inference

[ICRT](https://arxiv.org/html/2408.15980) uses a causal transformer over
interleaved image, proprioception, and action trajectories. Demonstration
trajectories act as prompts for closed-loop behavior without test-time
fine-tuning or a required language instruction. Its data ablation is crucial:
if the current observation identifies the task, the model can ignore the
trajectory prompt. This motivates counterfactual world instances where the
same current state requires different actions under different calibration or
demonstration histories.

[Algorithm Distillation](https://arxiv.org/abs/2210.14215) establishes a
separate requirement for in-context *learning*: training context must contain
learning histories, including behavior changing across episodes. Expert-only
traces do not teach exploration and improvement when those are required at
deployment. This supports a later active-probing branch and contexts spanning
multiple attempts; it does not require putting that difficulty into `S0`.

[Behavior Prompted Policies](https://arxiv.org/html/2606.30457) and
[RoboTTT](https://arxiv.org/html/2607.15275) are more direct 2026 evidence for
behavioral or human-video prompts. They are not viable blank two-T4 starting
architectures: BPP uses a large prompt encoder plus diffusion decoder and was
trained on four high-memory GPUs, while RoboTTT builds on a pretrained GR00T
model and reports training on 16 GB200 GPUs. They inform the target, not the
initial implementation.

### 2.3 In-context system and embodiment identification

[In-Context World Modeling](https://arxiv.org/abs/2606.26025) uses
task-agnostic, self-generated action-observation probes to infer camera,
morphology, and dynamics variation before control. This motivates the later
active system-identification branch. Supplied calibration in the selected
`S0` action is an easier scaffold: it exercises public evidence and context use
while postponing exploration-policy and live-interaction overhead.

[AdA](https://arxiv.org/abs/2301.07608) combines a broad, smooth task
distribution, long attention memory, and curriculum near the learner's current
frontier to produce human-timescale adaptation. It supports the joint role of
distribution design, memory, and repeated frontier scheduling. It does not
imply one privileged root task.

### 2.4 Capability graph, not a single difficulty ladder

[Universal Value Function Approximators](https://proceedings.mlr.press/v37/schaul15.html)
and [SAC-X](https://proceedings.mlr.press/v80/riedmiller18a.html) support
goal-conditioned transfer and auxiliary intent/task structures.
[Automatic Curriculum Graph Generation](https://ojs.aaai.org/index.php/AAAI/article/view/10933)
supports representing dependencies among learnable tasks rather than assuming
one scalar order. These inform the capability forest; this project's exact
nodes and edges remain falsifiable hypotheses.

## 3. Architecture evidence and the unresolved gap

### 3.1 Candidate comparison against the actual gates

| Candidate | Useful published property | Gate it fails as an unchanged base |
|---|---|---|
| ICRT | physical trajectories in causal context; random Llama core | pools an observation into one state token and predicts one fixed-width end-effector vector |
| Octo-Small | 27M, 12x384 modular robotics transformer; shallow visual tokenizer and readouts | native window is two observation steps and does not provide executed-action physical-prompt history |
| Astra/Actra | 19.4M tested profile; causal inter-trajectory and bidirectional intra-trajectory attention; action queries | released model fixes `action_dim`, creates one query/head per dimension, and uses a custom GPT-2 fork/mask |
| CrossFormer | variable modalities and embodiment-conditioned readouts | 130M, ImageNet backbones, fixed action classes/positions, short history, and very large TPU training |
| RPT/SMART | small sensorimotor cores and predictive/inverse objectives | not the long causal deployment interface; RPT also imports a pretrained visual encoder |
| BPP/RoboTTT | direct behavior/human-video prompting | large pretrained or large multi-GPU systems, not blank two-T4 learners |
| RoboToken | continuous variable-length physical entity/action tokens; 11.6M variant | articulated-body ontology and diffusion co-design target a narrower setting than the universal process interface |

No row passes all of: strict blankness, physical prompt history, variable
interfaces, standard causal implementation, and plausible primary iteration on
two T4s. Choosing any row unchanged would silently abandon at least one hard
requirement.

### 3.2 ICRT supplies the temporal core, not the complete model

The [ICRT paper](https://arxiv.org/html/2408.15980) and
[official repository](https://github.com/Max-Fu/icrt) are the closest standard
starting point:

- causal history over observations, proprioception, and actions;
- physical trajectories as context rather than mandatory text;
- continuous sensor/action projectors; and
- a randomly initialized Llama-style transformer core.

The paper's reported main core is 12 layers at width 768, attends over as many
as 512 state/action tuples, pools visual/proprioceptive input to **one state
token**, and embeds a **fixed action vector**. The repository includes smaller
12-layer width-384 and width-192 configurations, but their existence is
implementation evidence—not a paper result for those scales.

The selected experimental core adopts the repository's 12x384 shape, but uses
maintained Llama components through continuous `inputs_embeds`. It does not
copy a private transformer implementation. Its approximately 21.2M transformer
parameters and 2,048-token initial budget are project choices subject to an
authorized memory/throughput preflight; no source proves the complete model
fits or learns on two T4s.

### 3.3 Octo supplies independent scale and modularity evidence

[Octo](https://arxiv.org/html/2405.12213) is an open robotics-pretraining
framework. Octo-Small is a 27M, 12-layer, width-384, six-head transformer.
Its shallow convolutional patch tokenizer, blockwise causal observation
attention, passive readout tokens, and modular diffusion heads are strong
design evidence. Its published interface, however, uses only a short native
history and does not make past executed actions part of a long physical prompt.
Adopting Octo unchanged would fail the target gate; extending its history and
action grammar would also be custom.

Octo's diffusion head substantially outperformed MSE and discretized heads in
its benchmark ablation. That is evidence for a maintained distributional head
when later action conditionals become multimodal, not a reason to pay that
cost in the first deterministic world family.

### 3.4 Astra supplies action-query evidence, but not a reusable base

[Astra (originally Actra)](https://arxiv.org/html/2408.01147) gives each action
dimension a learned query and uses trajectory attention: tokens within a time
segment communicate bidirectionally, while segments remain causal. Its paper
tests a 19.4M, six-layer, width-512 profile, showing that the broad size regime
is real robotics research rather than an invented scale.

The [official implementation](https://github.com/naokiyokoyama/actra) was
audited at commit
[`65e7fdf`](https://github.com/naokiyokoyama/actra/commit/65e7fdf195ca570c7f00b18423b1057d2ffdd3a2).
It hard-codes `action_dim`, allocates a fixed list of action queries and output
heads, and carries a custom GPT-2 implementation, learned absolute positions,
and trajectory mask. It is evidence for query-based readout, not a maintained
drop-in solution for variable actuator sets. Copying that stack would enlarge
both the custom and systems-risk surfaces.

### 3.5 Selected ownership boundary

The end-to-end learner is custom, with the following deliberately narrow
boundary.

**Paper-/library-owned pieces:**

- ICRT's interleaved causal trajectory formulation;
- a maintained Llama decoder core with RoPE, RMSNorm, SwiGLU, and ordinary
  causal attention;
- Octo-style shallow random visual patch embedding and bounded readout/resampler
  evidence; and
- Astra/CrossFormer-style query readout and variable-modality evidence.

**Project-owned pieces—the acknowledged gap:**

- variable typed public observation/entity tokens;
- one schema-conditioned action query per public scalar actuator;
- one shared continuous action head instead of fixed per-dimension heads;
- the encounter-local schema/ID grammar and permutation controls; and
- the exact composition of numeric, visual, executed-action, outcome, and
  feedback events in one causal ABI.

No new attention mechanism, optimizer, transformer block, or special kernel is
proposed. The novelty is the physical token ABI and its use inside recursive
world scheduling. This is still material novelty: established components do
not prove their join preserves learnability, equivariance, coordinated action,
or scale transfer. Those risks receive explicit falsification tests before the
interface is frozen.

The strict non-custom fallback is exact ICRT. It is rejected as the primary
choice because its one pooled state and fixed action vector install precisely
the embodiment boundary that the GEN-1.5-like target requires the project not
to make permanent.

## 4. Cross-embodiment tokenization evidence

The following work changes the representation from a fixed robot vector to
variable typed channel/entity tokens.

[CrossFormer](https://arxiv.org/html/2408.11812) trains a block-causal
cross-embodiment transformer over heterogeneous modalities and reports results
across 20 embodiments. Its 130M system, ImageNet-initialized backbones, fixed
modality positions/action classes, short history, and TPU-v5e-256 training make
it evidence for factoring modalities and readouts—not the selected blank core.

[AnyMorph](https://proceedings.mlr.press/v162/trabucco22b.html) represents
variable numbers of sensor and actuator dimensions and can infer morphology
without requiring an explicit body graph. It argues against both a monolithic
robot vector and mandatory public morphology metadata.

[Body Transformer](https://proceedings.mlr.press/v270/sferrazza25a.html) and
[GET-Zero](https://arxiv.org/abs/2407.15002) support per-body-part/per-joint
tokens and graph structure when morphology and connectivity are public. The
project can therefore add optional relation tokens, but must not invent or leak
such information in worlds where the learner would not receive it.

[Transformer Transformer / RoboToken](https://arxiv.org/html/2607.25798)
uses continuous, variable-length link/joint/motor/state/action tokens and
reports 11.6M and 63.6M variants. Its articulated-body ontology and fresh DiT
co-design are too specific for the universal core, but it is current evidence
that variable per-entity physical tokens can be used at small scale.

[UniAct](https://openaccess.thecvf.com/content/CVPR2025/html/Zheng_Universal_Actions_for_Enhanced_Embodied_Foundation_Models_CVPR_2025_paper.html)
learns a vector-quantized universal action space across embodiments. It uses a
large pretrained vision-language model and learned codebook, violating strict
blankness. It is a later option after a broad corpus exists, not a reason to
fit a narrow early-world codebook and declare it universal.

### Representation conclusion

Use a universal causal event envelope:

```text
(role, encounter, episode, phase, time, encounter-local entity ID, payload)
```

The initial grammar contains schema, boundary, condition, observation, action
query, executed action, outcome query, and feedback roles. Numeric values use
one continuous token per public scalar. Images use a randomly initialized
shallow convolutional patch stem and bounded attention resampler. There is no
atomic world ID, global actuator meaning, language vocabulary, mandatory body
graph, or fixed robot cardinality.

This installs a real prior: interfaces decompose into persistent local
components whose meanings can be inferred from schema and history. The prior
is justified by variable-embodiment work, but its precise grammar is custom and
must survive joint relabeling, token-order, and held-out-cardinality tests.

## 5. Action representation evidence

[FAST](https://arxiv.org/html/2501.09747) compresses smooth action chunks using
normalization, DCT, quantization, coefficient ordering, and BPE. FAST+ is
pretrained on roughly one million robot action sequences, and the authors
report major VLA training-speed gains. It establishes a later option and warns
against naive per-timestep bins, but importing FAST+ would violate blankness.

The initial action ABI is therefore continuous and language-free:

- one schema-conditioned query per public scalar actuator;
- one shared head emits a masked 16-step bounded chunk for every query;
- the environment-applicable prefix is executed; and
- actual executed actions are appended to context before outcomes.

This avoids a fixed maximum action vector and permits higher-frequency chunks
without multiplying scheduler events. The choice is custom and risky: separate
per-actuator queries may underrepresent tightly coordinated actions, and L1
regression may average multimodal targets. Joint-coordination probes and a
maintained diffusion/flow-head ablation are required before treating it as a
settled universal action representation.

## 6. Current 2026 world/action-model evidence

[LiLa-WAM](https://arxiv.org/html/2608.03701) jointly models language-free
future latent states and actions. Its roughly 0.5B published system uses a
frozen DINOv3 encoder, so it is neither blank nor an initial two-T4 core. It
supports a native action-conditioned future-observation objective and
language-free visual-temporal context.

[GEN-1](https://generalistai.com/blog/gen-1) reports pretraining from scratch on
very large physical interaction data, and an official
[essay](https://generalistai.com/blog/beyond-world-models) describes most model
parameters as trained from scratch. The undisclosed architecture and enormous
scale are not reproducible evidence for this stack. The useful lesson is only
that a physical-first recipe is compatible with the end target.

[PACT](https://arxiv.org/html/2209.11133) uses a compact 12-layer, width-128
causal transformer to jointly predict actions and future states from embodied
trajectories. It is useful scale and objective evidence, but its automotive
tokenization and pretrained perception paths are not a universal blank ABI.

## 7. Repeated scheduling and continual-learning evidence

The recursive policy goes beyond ordinary fixed pretraining, but its first
approximation can use established components:

- [Automated Curriculum Learning](https://proceedings.mlr.press/v70/graves17a.html)
  uses learning progress to allocate training tasks.
- [Teacher-Student Curriculum Learning](https://arxiv.org/abs/1707.00183)
  formalizes an adaptive teacher selecting tasks for a student.
- [Prioritized Level Replay](https://proceedings.mlr.press/v139/jiang21b.html)
  prioritizes levels using estimates of future learning potential.
- [Task Selection Policies for Multitask Learning](https://arxiv.org/abs/1907.06214)
  studies policies for selecting tasks during multitask training.
- [AdA](https://arxiv.org/abs/2301.07608) provides large-scale evidence for
  sampling near the learner's current capability frontier.

These support a cheap, logged learning-progress/frontier selector initially.
They do not supply this project's end-capability utility, capability graph,
world validity gates, costs, or value-of-information thresholds. Those are the
novel outer loop and must be empirically corrected.

[World-model-guided rehearsal](https://arxiv.org/abs/2502.19544) reports that
rehearsal can improve offline-to-online adaptation under distribution shift.
It supports keeping old experience as an available action, not a permanent
replay percentage. Replay competes with new worlds, static mixtures,
evaluation, and migration under the same recursive value/cost rule.

## 8. Systems theory used for overhead

The scheduler only has value if its overhead is compared with static
pretraining. The accounting uses established systems abstractions:

- [Roofline](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
  separates compute and memory-bandwidth lower bounds.
- [Amdahl's law](https://doi.org/10.1145/1465482.1465560) bounds gains when
  learner-world interaction or scheduler work remains serial.
- [LogP](https://dl.acm.org/doi/10.1145/155332.155333) motivates explicit
  latency/bandwidth communication terms.
- [Brent's work/span result](https://maths-people.anu.edu.au/~brent/pub/pub022.html)
  separates total work from sequential critical path.

The project applies these to three matched inner paths—ordinary static
training, exact transcripts prepacked, and live world realization—and to an
outer event DAG containing checkpoint, evaluation, branching, switching, and
admission. [OVERHEAD-MODEL.md](OVERHEAD-MODEL.md) contains the formal model.

## 9. Evidence-to-decision map

| Evidence | Decision influenced | What it does not prove |
|---|---|---|
| GEN-1.5 / thousand hands | hard target gates: physical prompts, long context, cross-realization and embodiment transfer | that a small blank model can reproduce the reported performance |
| ICRT | causal random trajectory core and physical history | that one pooled state/fixed action vector is adequate, or that the custom ABI works |
| Octo-Small | independent 12x384 scale; modular visual/readout evidence; later diffusion option | two-T4 fit for this model or long action-conditioned prompting |
| Astra | small tested action-query architecture and trajectory grouping | variable actuators or justification for copying its custom GPT-2 stack |
| RPT/SMART/PACT | small cores and predictive/inverse objectives | that masked or automotive pretraining should replace causal deployment |
| CrossFormer/AnyMorph | variable modality and actuator tokens | one universal action semantics or this project's exact grammar |
| Body Transformer/GET-Zero | optional public morphology relations | permission to leak hidden morphology |
| RoboToken | current variable-length continuous physical tokens at small scale | that its articulated-body ontology is universal |
| FAST/UniAct | later compressed/universal action representations | permission to import pretrained codebooks into a blank learner |
| BPP/RoboTTT | cross-realization behavioral prompting as a real research direction | feasibility for a blank two-T4 initial learner |
| ICWM/Algorithm Distillation | active probes and learning histories for later in-context learning | that active probing is the cheapest `S0` action |
| AdA/curriculum work | repeated frontier/task allocation | a privileged root or a free scheduler |
| Roofline/Amdahl/LogP/work-span | measurable inner and outer critical-path overhead | numerical cost before hardware measurement |

## 10. Limits and live alternatives

- No source proves that `W_calibrated_monomial[d=1..4]` is the optimal first
  training action. It is the current scheduler output under explicit target,
  validity, transfer, and cost priors.
- No source validates the ICRT-core/variable-token/action-query join. That is
  the central architecture gap; interface invariance and coordination tests can
  reject it before a costly developmental run.
- No source proves the capability dependency graph. Transfer measurements may
  revise its nodes and edges whenever the expected decision value exceeds the
  evaluation cost.
- “Two T4s plausible” is an unmeasured sizing estimate. Primary architectural,
  world, and scheduler experimentation remains on two T4s; the first authorized
  preflight must establish fit and throughput.
- A Kaggle TPU is allowed only for a final frozen run after T4 decisions and a
  separate execution-plan amendment. A wider/deeper blank TPU learner is a new
  confirmation lineage, not continuation of the smaller checkpoint; its scale
  may expose different learning and scheduling behavior.
- A long static multi-world dataset may beat frequent recursive switching once
  outer overhead is charged. It remains an explicit inner-loop action and
  baseline at every checkpoint.
- Recent ICWM, RoboToken, BPP, RoboTTT, and LiLa-WAM findings require
  replication caution; Generalist reports are proprietary company evidence.
- The strongest strict-standard fallback is exact ICRT, but it fails the
  variable-interface target gate. The strongest alternate causal base is an
  extended Octo-Small, which merely moves the custom gap to long physical
  history. The strongest first-training alternative is a prepacked multi-world
  predictive mixture. All remain explicit alternatives, not erased options.
