# Rendering Transfer After Process Training

## Seed-0 experiment report

> **Superseded.** The intermediate-checkpoint results below confound
> learning rate with token exposure (checkpoints were taken mid-schedule
> from one cosine run). The corrected two-seed results and the current
> conclusions of this stage are in
> [RENDERING-B-TERMINAL-TRANSFER-REPORT.md](RENDERING-B-TERMINAL-TRANSFER-REPORT.md).
> This report remains valid as a training-dynamics record and for its
> apparatus description.

## Summary

We tested whether prior training on one representation of a small interactive
reasoning task reduces the amount of experience required to learn a second
representation of the same task.

The experiment compared two models:

1. **A-trained → B:** a model first trained on Rendering A, then fine-tuned on
   Rendering B;
2. **Init → B:** the same model architecture, recreated from the same original
   random initialization, trained directly on Rendering B.

Both models received exactly the same Rendering B calibration data and were
evaluated on exactly the same held-out Rendering B worlds. The comparison
therefore measures whether experience in Rendering A improves the
sample-efficiency of learning Rendering B.

The principal seed-0 result is strong early transfer. After approximately
3 million Rendering B input tokens, the A-trained model succeeded on **24.9%**
of held-out interactive episodes, while the model trained from initialization
succeeded on **0.1%**. At approximately 10 million tokens, the corresponding
results were **21.3%** and **0.0%**. The model trained from initialization did
not attain comparable capability until approximately 30 million tokens.

At the full 100 million-token point, the two models largely converged:
**44.6%** success for A-trained → B and **42.2%** for Init → B. The main benefit
of prior Rendering A training was therefore faster acquisition of Rendering B,
not a substantially higher final ceiling.

These results come from one experimental seed. They are evidence of positive
representation transfer, but uncertainty across training seeds has not yet
been measured.

## What we were testing

The broader question is whether a model can learn an interactive process in a
way that is not tied entirely to its original surface form.

The task requires a model to gather evidence about a hidden hypothesis and
then act under a budget. Some actions acquire information; a commitment ends
or changes the future of the episode. Success therefore requires more than
predicting isolated labels: the model's actions alter the state from which its
next observation is generated.

The same typed observations and actions can be expressed through two aligned
renderings:

- **Rendering A** is the acquisition representation used for the initial
  dense-supervision run.
- **Rendering B** expresses the same task information and actions with a
  different lexicon, ordering, and layout.

Rendering B deliberately uses a different action vocabulary. Zero-shot
failure on B is consequently not, by itself, a failure of process transfer:
the model has never been taught how B's strings map to actions. The relevant
measurement is how quickly a model learns B after a controlled amount of B
interface calibration.

The experiment therefore asks:

> At the same cumulative amount of Rendering B training, does a model already
> trained on Rendering A act more successfully in held-out Rendering B worlds
> than an otherwise matched model learning B from its original initialization?

## Experimental apparatus

### World and interaction

Each episode is a generated, finite, partially observable decision problem.
The evaluated configuration contains:

- 6 candidate hypotheses;
- 5 available probes;
- deterministic evidence;
- non-uniform action costs and a limited budget;
- an irreversible primary condition, in which commitment removes future
  alternatives or terminates the episode;
- a matched reversible control, in which a provisional commitment can be
  revised.

During evaluation, the model generates a rendered action as text. That text is
passed through the public Rust parser and world executor. Malformed strings,
parsed-but-invalid actions, state transitions, spending, and final success are
therefore measured through actual closed-loop execution rather than inferred
from model logits.

### Model and standard training stack

The model is a randomly initialized GPT-NeoX causal language model implemented
with Hugging Face Transformers. Its frozen configuration has approximately
19.2 million parameters:

- 6 transformer layers;
- hidden width 512;
- 8 attention heads;
- context length 2,048 byte-level tokens;
- vocabulary size 262, comprising byte tokens and fixed protocol tokens.

Training used the standard Transformers `Trainer` and Accelerate distributed
path on **two NVIDIA Tesla T4 GPUs**, using FP16 execution. The global nominal
input-token batch was 32,768 tokens per optimizer update.

### Source model

The A-trained arm began from the completed dense-supervision seed-0 checkpoint.
That model had received 100,007,936 nominal input tokens in Rendering A. Before
the transfer experiment, it achieved **41.1%** success on 1,024 held-out
in-distribution Rendering A episodes.

The control arm recreated the model from the same original seed-0 random
initialization and frozen model configuration. It did not begin from an
independently sampled initialization.

### Rendering B calibration data

One deterministic shard of 32,768 Rendering B teacher trajectories was
generated. Both arms consumed the identical shard in the identical order.
The shard was separate from the held-out evaluation worlds.

Each arm was trained once as a continuous run, with standard checkpoints saved
at the following cumulative budgets:

| Checkpoint | Optimizer updates | Nominal Rendering B input tokens |
|---:|---:|---:|
| Zero-shot diagnostic | 0 | 0 |
| Calibration point 1 | 31 | 1,015,808 |
| Calibration point 2 | 92 | 3,014,656 |
| Calibration point 3 | 306 | 10,027,008 |
| Calibration point 4 | 916 | 30,015,488 |
| Full-B reference | 3,052 | 100,007,936 |

The cost of the earlier Rendering A training is reported separately; it is not
counted as Rendering B calibration experience.

### Evaluation

Every checkpoint was evaluated on the same 1,024 held-out Rendering B worlds.
The primary condition used irreversible commitment. A matched reversible
control used the same held-out seed and Rendering B interface.

The primary metric was closed-loop episode success. We also recorded malformed
action rate, invalid action rate, spending, number of steps, and excess cost
relative to the privileged teacher on successful episodes.

Infrastructure properties—including source-checkpoint identity, dataset
identity, token accounting, checkpoint discovery, two-rank completion, and
model serialization—were checked exactly. Scientific metric values were
reported as measurements and were not used as hard pass/fail thresholds.

## Results

### Held-out Rendering B: irreversible process

| B calibration tokens | A-trained → B | Init → B | Difference |
|---:|---:|---:|---:|
| 0 | 0.0% | 0.0% | 0.0 percentage points |
| 1,015,808 | 0.0% | 0.0% | 0.0 pp |
| 3,014,656 | **24.9%** | **0.1%** | **+24.8 pp** |
| 10,027,008 | **21.3%** | **0.0%** | **+21.3 pp** |
| 30,015,488 | 1.3% | 39.3% | −38.0 pp |
| 100,007,936 | **44.6%** | 42.2% | +2.4 pp |

The largest early advantage appears at roughly 3 million B tokens. The
A-trained model solved 255 of 1,024 episodes, while the model trained from
initialization solved 1 of 1,024.

The from-initialization arm first exceeded 20% success at roughly 30 million B
tokens. On this seed and this checkpoint schedule, prior A training therefore
reduced the B experience required to first reach that level by approximately
an order of magnitude.

### Action-interface behavior

The early advantage was not merely an ability to emit strings that looked like
Rendering B. At approximately 3 million tokens:

| Arm | Success | Malformed actions | Invalid actions |
|---|---:|---:|---:|
| A-trained → B | 24.9% | 4.7% | 3.3% |
| Init → B | 0.1% | 0.0% | 99.6% |

The control model had learned to produce parseable B syntax, but almost every
episode attempted an action invalid in the current world state. The A-trained
model was already producing substantially more usable closed-loop behavior.

### Matched reversible control

| B calibration tokens | A-trained → B | Init → B | Difference |
|---:|---:|---:|---:|
| 0 | 0.0% | 0.0% | 0.0 pp |
| 1,015,808 | 0.0% | 0.0% | 0.0 pp |
| 3,014,656 | **17.4%** | 0.1% | **+17.3 pp** |
| 10,027,008 | **4.3%** | 0.0% | **+4.3 pp** |
| 30,015,488 | 0.0% | 20.9% | −20.9 pp |
| 100,007,936 | **21.6%** | 18.0% | +3.6 pp |

The reversible control shows a similar early advantage for the A-trained arm.
This is consistent with useful behavior transferring across representations,
although this single comparison does not by itself establish that the model
learned the intended irreversible-versus-reversible process distinction.

## Interpretation

### Positive early transfer

The result supports the experiment's primary transfer prediction: prior
Rendering A training substantially reduced the amount of Rendering B
experience needed to obtain useful held-out behavior.

The comparison is especially informative because both arms shared the model
architecture, original initialization, B calibration data, training budget,
and held-out evaluation worlds. The relevant difference was the A-trained
arm's prior experience.

### Transfer benefit is primarily sample efficiency

At the full 100 million-token B budget, the two arms were close: 44.6% versus
42.2% success. Prior A training did not create a clearly higher final ceiling
on this seed. Instead, it moved useful B performance much earlier in the
learning curve.

This is the intended notion of representation transfer: previous experience
reduces the new experience required to act under a changed representation.

### The curve is not monotonic

The A-trained arm fell from 21.3% success at approximately 10 million tokens to
1.3% at approximately 30 million tokens, with invalid actions rising to 97.0%,
before recovering to 44.6% at the final checkpoint.

Training loss remained finite and low through this interval. The collapse
therefore highlights a meaningful failure mode: teacher-forced next-token loss
can remain healthy while closed-loop policy behavior becomes substantially
worse. Intermediate interactive evaluation is necessary; the final checkpoint
alone would hide this instability.

The precise cause of this non-monotonic behavior has not yet been isolated.
Possible explanations include policy instability under continued supervised
fine-tuning, changes in action-selection behavior that are weakly reflected in
average token loss, or single-seed checkpoint variance. These are hypotheses,
not established conclusions.

## What this experiment establishes—and what it does not

This seed-0 experiment provides evidence that:

- the model learned useful closed-loop behavior under Rendering A;
- zero-shot Rendering B failure reflected an ungrounded interface and was not
  an appropriate transfer measurement;
- after limited B calibration, the A-trained model acquired effective B
  behavior substantially faster than the matched from-initialization control;
- the advantage was not reducible to parseable syntax alone;
- full B training largely closed the gap between the two arms; and
- interactive capability can change sharply while teacher-forced loss remains
  smooth.

It does **not** yet establish that:

- the effect is stable across random seeds;
- the model has learned a general or abstract process representation;
- the reversible/irreversible distinction itself has transferred cleanly;
- the approximately tenfold sample-efficiency estimate is a population-level
  effect rather than a seed-0 observation; or
- the result transfers beyond this generated world family.

## Reproducibility record

- Transfer implementation Git commit:
  `e592ba12c19992a33246123fad82a6406f4bc771`
- Dense Rendering A source commit:
  `84f29385ed623500aa2e201c45fdcf8c2257fac0`
- Dense source configuration hash:
  `b91eedc2655253edd662c320753c2680cb7842484411c22130deeda31a58cb14`
- Dense source model-state SHA-256:
  `cef0ac5a4159d65eae336be58c34dfe1e3f078a8024f4b98a7ca1a78e42a9a6e`
- Rendering B calibration-shard SHA-256:
  `34e92a8e5c6f483a7a83689798fbd07095e65d7c2bfbf81ab3f26bb4c8d076b9`
- Root seed: `20260811`
- Calibration-world seed: `23260811`
- Held-out evaluation seed: `21260811`
- Evaluation episodes per condition and checkpoint: `1,024`
- Hardware: two NVIDIA Tesla T4 GPUs

The machine-readable result is stored in
`rendering_b_transfer_report.json`. Its exact operational contract passed,
including source identity, matched dataset identity, two-rank accounting,
checkpoint milestones, and exact state-dictionary serialization checks.

## Next step

The immediate next step is replication across the predeclared seeds. That will
show whether the early transfer advantage and intermediate behavioral collapse
are stable effects. A stronger semantic-transfer claim should additionally
compare matched irreversible and reversible learning curves with the planned
surface/target-shuffled controls.
