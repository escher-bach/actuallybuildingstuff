# Experiment Execution Plan

## STEP 2 two-T4 vertical-slice amendment — 2026-08-24

The user has explicitly authorized CPU tests and a two-T4 GPU test of the
selected STEP 2 architecture joined to the first Rust world family, followed in
the same allocation by the start of bounded first-checkpoint generation. This
amendment is the sole authorized STEP 2 launch at present. The closed STEP 1
runs below remain closed.

### Scientific and apparatus boundary

The run tests one prerequisite claim before any checkpoint is interpreted:

> the fully random ICRT-derived Llama core, physical-event token ABI,
> schema-conditioned continuous readouts, Rust public-prefix oracle, and
> `W_calibrated_monomial[d=1..4]` transcript can execute one finite
> forward/backward update, overfit a fixed real batch, save/reload, run on both
> T4 ranks, checkpoint/resume, and then train from a fresh random reset.

The fixed-batch diagnostic weights are discarded. Only the separately reset
lineage may produce `c1-start-candidate`. Passing this apparatus gate does not
establish world validity, downstream transfer, or completion of the first
developmental session. A failure before the fresh lineage starts is an
apparatus failure.

The governing scientific drafts are under
`step2/assistant-work/2026-08-24-top-down-start/`, with the inherited
information-boundary and procedural-generation requirements in
`step2/WORLD-VALIDITY.md`.

### STEP 2 implementation boundary

- Rust owns process generation, latent transition execution, the
  serializer-level public-prefix teacher/oracle, verifier, deterministic replay,
  and batched online rollout state.
- PyO3 exposes one batched learner-tensor boundary plus explicitly privileged
  validator methods. Python does not reproduce the transition or oracle.
- Hugging Face `LlamaModel` owns transformer blocks, causal masking, RoPE,
  RMSNorm, and SwiGLU. PyTorch owns continuous adapters/heads; Accelerate owns
  mixed precision, DDP, gradient synchronization, and state restore.
- All learned tensors initialize randomly. No STEP 1/Pythia, language, visual,
  or action weights are loaded.

### STEP 2 run and source contract

The sole entry point is:

```text
python tools/kaggle_run.py launch --experiment architecture-world-vertical-slice
```

It renders a launcher in an ephemeral directory and submits it with the
official Kaggle CLI requesting `NvidiaTeslaT4`. The launcher clones the
configured GitHub repository, checks out one full 40-character commit, verifies
that exact SHA, and invokes exactly once:

```text
python -m step2_experiments.runner \
  --config step2/configs/kaggle/t4x2_vertical_slice.toml \
  --output-root /kaggle/working/step2-results
```

The notebook contains no world, model, training, or evaluation code. The source
clone, dependency cache, Rust target, and wheel remain under `/tmp`; only result
artifacts are written under `/kaggle/working`.

The STEP 2 runner must execute in this order:

```text
capture_environment
  -> install_and_build
  -> Rust and Python correctness tests
  -> generated-world validity sweep
  -> Kaggle-CPU world/binding throughput
  -> exact two-T4 verification
  -> real-batch architecture overfit gate
  -> discard diagnostic weights and reinitialize
  -> checkpoint/restore smoke
  -> bounded training start
  -> teacher-forced and closed-loop evaluation
  -> retained candidate checkpoint and compact audit artifacts
```

The retained checkpoint stays on Kaggle. `collect` downloads only selected JSON
evidence and logs, verifies the audit manifest, and writes a local receipt under
`step2/audit/runs/`. TPU execution remains unauthorized by this amendment.

---

## Kaggle 2xT4 implementation specification

### Status and scope

> **Closed-run status, 2026-08-20.** No additional `world-0.1.0` experiment may
> be launched. This document remains authoritative for preserving, retrieving,
> and auditing its existing runs; its orchestration and receipt machinery may
> be transferred only after the next step supplies a new scientific contract.
> See [STEP-1-WORLD-0.1-CLOSURE.md](STEP-1-WORLD-0.1-CLOSURE.md).

This document is the sole implementation and operations contract for retained
Step 1 Kaggle runs. It governs monitoring, retention, retrieval, and audit. It
turns the experimental requirements in [STEP-1.md](STEP-1.md) into one
non-interactive Kaggle workflow; other documents may point here but must not
duplicate or redefine this workflow.

The current target is a Kaggle notebook with **two NVIDIA T4 GPUs**. A future
TPU runner may reuse the same configurations, data manifests, metrics, and
artifact schema, but TPU support is not part of this implementation. Do not add
`torch_xla`, TPU branches, or a general accelerator abstraction before the T4
run works and produces interpretable results.

The historical scientific comparison was:

1. dense teacher supervision;
2. outcome-only verified learning;
3. a target-shuffled control; and
4. an untrained baseline.

The infrastructure must not grow into a general training platform. It exists
to run this comparison reproducibly and with a fast one-command operator path.

---

## 1. The execution model

### 1.1 A committed notebook version is one run

A Kaggle **Save & Run All** version, also called a commit or batch session, runs
the notebook from top to bottom. One such committed version must correspond to
one immutable experiment run:

```text
one Git commit SHA
+ one experiment configuration hash
+ one root seed
+ one Kaggle hardware allocation
+ one exact Kaggle notebook version
= one run ID, one remote recovery payload, and one tracked audit receipt
```

The notebook must succeed when run from a fresh Kaggle batch session. It must
not depend on state created by an earlier interactive execution.

### 1.2 The notebook is a launcher, not the experiment

The notebook must do only four things:

1. define immutable launch constants;
2. clone the repository and check out an exact commit;
3. invoke the repository-owned runner once; and
4. raise an error if the runner fails, after the runner has saved a failure
   bundle.

It must not contain model classes, training loops, dataset classes, evaluation
logic, plots, or copied fragments of repository code. Fixes belong in Git;
they must never be patched manually into a live notebook.

The notebook must never call `input()`, open a widget, request a login, prompt
for a secret, wait for a button press, or assume that a person is watching it.
Weights & Biases and other services requiring interactive authentication are
disabled. Logs and metrics are local files.

The source clone, dependency caches, and build products must live under
ephemeral storage such as `/tmp/step1-runtime`, never under `/kaggle/working`.
Only declared result artifacts may be written under `/kaggle/working` because
Kaggle publishes that tree as notebook output.

### 1.3 Required Kaggle settings

The repository orchestrator must submit with the official Kaggle CLI and:

- request `NvidiaTeslaT4` with `kaggle kernels push --accelerator
  NvidiaTeslaT4`;
- enable Internet access so the public Git repository and pinned dependencies
  can be fetched;
- generate current kernel metadata in an ephemeral staging directory, using
  `machine_shape` rather than a stale checked-in accelerator field; and
- create a committed batch version, never an interactive partial execution,
  for a retained result.

The requested accelerator is not trusted as evidence. The runner must inspect
the actual allocation and fail before training unless exactly two CUDA devices
are visible and both report themselves as T4 GPUs.

The Kaggle UI is reserved for one-time account consent, exceptional debugging,
or recovery when a confirmed CLI limitation blocks the standard path. Manual
notebook upload and full-output download are not the normal workflow.

### 1.4 One-command control plane

The sole operator entry point is a thin adapter around the maintained Kaggle
CLI:

```bash
python tools/kaggle_run.py run --experiment <experiment-name>
```

It must expose only the operations needed by a personal project:

```text
run       validate -> render -> submit -> wait -> collect compact evidence
launch    validate -> render -> submit and return the immutable run reference
status    report the state of an exact submitted version
logs      show execution logs, especially on failure
collect   fetch and verify compact audit/analysis artifacts only
```

The adapter must not reimplement the Kaggle API. It shells out to pinned,
documented `kaggle kernels` commands and owns only project-specific validation,
staging, provenance, checksum verification, and receipt generation. Before
submission it must verify that the full Git SHA is reachable from the configured
remote, resolve one declared experiment, and reject concurrent submission to
the same mutable kernel. It must never push Git commits or launch GPU work
implicitly.

---

## 2. Files that must be implemented

The implementation should add only the following experiment-facing structure:

```text
step1/
  audit/
    runs/
      <run-id>/
        receipt.json              # tracked compact evidence index
        result-report.json        # tracked when scientifically relevant
        summary.json
  kaggle/
    step1_t4x2.ipynb
    experiments.toml              # experiment -> config/template/source refs
  configs/
    kaggle/
      t4x2_preflight.toml
      t4x2_dense_seed0.toml
      t4x2_dense_seed1.toml
      t4x2_dense_seed2.toml
      t4x2_rlvr_seed0.toml        # added only after dense training passes
  python/
    step1_experiments/
      __init__.py
      runner.py                   # the single public entry point
      environment.py
      data.py
      model.py
      train.py
      evaluate.py
      benchmarks.py
      artifacts.py
  requirements-kaggle.txt
tools/
  kaggle_run.py                    # sole operator entry point
```

`kernel-metadata.json` is generated in a temporary submission directory. A
single checked-in mutable metadata file is not authoritative because owner,
code filename, accelerator, and upstream sources vary by run. The experiment
registry is declarative and contains no credentials.

Modules may be combined when that makes the implementation smaller. The
semantic boundaries must remain testable, but one-file-per-concept ceremony is
not required.

The public invocation is:

```bash
python -m step1_experiments.runner \
  --config step1/configs/kaggle/t4x2_dense_seed0.toml \
  --output-root /kaggle/working/step1-results \
  --resume auto
```

`runner.py` is the only command the notebook launches. It orchestrates setup,
tests, benchmarks, shard generation, distributed training, evaluation, and
artifact packaging. The notebook must not launch `torchrun` directly.

---

## 3. Exact notebook contract

The notebook should contain one short Markdown cell and three executable
Python cells. Shell commands must be invoked with `subprocess.run` and checked
return codes rather than relying on notebook magic state.

### Cell 1: immutable launch constants

```python
from pathlib import Path
import os
import subprocess
import sys

REPO_URL = "https://github.com/escher-bach/actuallybuildingstuff.git"
GIT_COMMIT = "<FULL_40_CHARACTER_COMMIT_SHA>"
CONFIG_REL = "step1/configs/kaggle/t4x2_dense_seed0.toml"

RUNTIME = Path("/tmp/step1-runtime")
WORKING = Path("/kaggle/working")
SOURCE = RUNTIME / "actuallybuildingstuff"
PROJECT = SOURCE / "baby-llm-foundations"
OUTPUT = WORKING / "step1-results"

assert len(GIT_COMMIT) == 40 and all(c in "0123456789abcdef" for c in GIT_COMMIT)
assert not SOURCE.exists(), f"fresh batch session required; already exists: {SOURCE}"
RUNTIME.mkdir(parents=True, exist_ok=True)
OUTPUT.mkdir(parents=True, exist_ok=True)
```

The placeholder must be replaced with the full commit containing the runner.
A branch name, tag, short SHA, or `main` is not acceptable for a retained run.

### Cell 2: clone and verify the source

```python
env = os.environ.copy()
env.update({
    "GIT_TERMINAL_PROMPT": "0",
    "PYTHONUNBUFFERED": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "WANDB_MODE": "disabled",
    "TOKENIZERS_PARALLELISM": "false",
})

subprocess.run(["git", "clone", REPO_URL, str(SOURCE)], check=True, env=env)
subprocess.run(
    ["git", "-C", str(SOURCE), "checkout", "--detach", GIT_COMMIT],
    check=True,
    env=env,
)
resolved = subprocess.check_output(
    ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
    text=True,
    env=env,
).strip()
assert resolved == GIT_COMMIT, (resolved, GIT_COMMIT)
assert (PROJECT / CONFIG_REL).is_file(), PROJECT / CONFIG_REL
```

### Cell 3: run exactly one runner

```python
cmd = [
    sys.executable,
    "-m",
    "step1_experiments.runner",
    "--config", str(PROJECT / CONFIG_REL),
    "--output-root", str(OUTPUT),
    "--resume", "auto",
]

completed = subprocess.run(
    cmd,
    cwd=str(PROJECT / "step1" / "python"),
    env=env,
    check=False,
)
if completed.returncode != 0:
    raise RuntimeError(
        f"Step 1 runner failed with exit code {completed.returncode}; "
        f"download the failure bundle from {OUTPUT}"
    )
```

The runner must package results in a `finally` block. The notebook raising at
the end is intentional: Kaggle should mark a broken run as failed, while the
diagnostic files remain available in `/kaggle/working`.

---

## 4. Runner phase contract

The runner is a deterministic state machine. It records the start and end of
every phase in `phase_status.json` and skips only a phase already marked
complete with a matching Git SHA and resolved-configuration hash.

The required order is:

```text
capture_environment
    -> install_and_build
    -> correctness_tests
    -> cpu_throughput
    -> prepare_shards
    -> dataloader_throughput
    -> gpu_preflight
    -> instrument_check OR train
    -> evaluate
    -> package_results
```

### 4.1 Environment capture

Before installing or training anything, save:

- resolved Git SHA and `git status --porcelain`;
- complete resolved experiment configuration and its SHA-256 hash;
- Python, PyTorch, CUDA, cuDNN, compiler, Cargo, and package versions;
- `nvidia-smi` output and per-device name, capability, driver, and VRAM;
- CPU model, logical/physical core count, process affinity, and RAM;
- `/kaggle/working` free space and filesystem information;
- relevant Kaggle environment variables, excluding secrets; and
- the exact command line.

Never serialize tokens, passwords, cookies, or the full environment mapping.

### 4.2 Installation and Rust build

The Kaggle image's installed PyTorch is the base runtime. Do not upgrade or
downgrade it from the notebook. `requirements-kaggle.txt` contains only pinned
packages absent from the image.

The runner must:

1. check whether `cargo` and `maturin` are available;
2. install a pinned minimal Rust toolchain non-interactively only when Cargo is
   absent;
3. install the pinned Python build dependencies;
4. build the Linux `world_py` extension in release mode;
5. install the resulting wheel; and
6. import `world_py` and execute `step1/python/smoke.py`.

Every subprocess uses a timeout, checked return code, and captured log. A build
failure is fatal and is packaged like a training failure.

### 4.3 Correctness tests

Run the Rust workspace tests before any benchmark or model work:

```bash
cargo test --workspace --locked
```

Then run Python tests covering the shard reader, mask alignment, distributed
sampler, checkpoint restore, and one forward/backward optimizer step. Training
must not begin after a failed or skipped correctness test.

---

## 5. Token and model contract

### 5.1 Primary tokenizer

Use the frozen UTF-8 byte vocabulary:

- token IDs `0..255`: literal bytes;
- six fixed transport tokens: `PAD`, `BOS`, `EOS`, `OBS`, `ACTION`, and
  `END_TURN`;
- no learned BPE merges; and
- no access to typed world objects from the tokenizer.

The manifest records the complete ID table and its SHA-256 hash. Encoding and
decoding tests must cover every byte, both renderings, fragmented assembly,
and malformed byte sequences.

### 5.2 Fixed inducer

The retained experiment uses one approximately 21M-parameter conventional
causal Transformer:

- 12 layers;
- model width 384;
- 6 full-attention heads of dimension 64;
- SwiGLU intermediate width 1024;
- RoPE;
- RMSNorm and QK normalization;
- bias-free projections;
- zero dropout; and
- ordinary causal next-token loss on the declared action spans.

The block is fixed, not left to implementer preference:

```text
h = x + Attention(RMSNorm(x))
out = h + SwiGLU(RMSNorm(h))
```

Q and K receive per-head RMS normalization after projection and before RoPE.
RoPE uses base 10,000 and is applied to the full 64-dimensional head. The MLP
is the ordinary bias-free `down(SiLU(gate) * up)` form. Apply a final RMSNorm,
tie the input embedding and output projection, and use epsilon `1e-5` for every
RMS normalization. Initialize matrices from a zero-mean normal distribution
with standard deviation `0.02`; scale attention-output and MLP-down residual
projection initialization by `1/sqrt(2 * n_layers)`.

Do not add GQA, sliding-window attention, mixture-of-experts layers, learned
absolute positions, auxiliary heads, or a typed-action output head.

Do not run a model-size ladder. Parameter count, including and excluding input
and output embeddings, is computed by code and recorded.

### 5.3 Optimization and initial budget

The first dense runs use one frozen, ordinary optimization recipe:

- AdamW with learning rate `6e-4`, betas `(0.9, 0.95)`, epsilon `1e-8`;
- weight decay `0.1` on matrix weights, and zero weight decay on embeddings and
  normalization parameters;
- linear warm-up over the first 2% of input tokens;
- cosine decay to `6e-5` at the fixed token budget;
- global gradient-norm clipping at `1.0` after unscaling;
- FP16 autocast and dynamic gradient scaling; and
- no label smoothing, z-loss, dropout, or early stopping.

The initial retained dense-training budget is **100,000,000 non-padding input
byte tokens per seed**. Save and evaluate at cumulative token counts
`1M, 3M, 10M, 30M, 100M`. The instrument check is capped at `2M` tokens. A run
always follows its fixed token budget; validation performance does not stop or
extend it.

The nominal global batch is 32,768 non-padding input tokens per optimizer
update. Microbatch preflight may change the per-rank microbatch and gradient
accumulation while preserving this global target. The loss is a global sum
over supervised action-token negative log likelihood divided by the global
number of supervised action tokens. DDP ranks must all-reduce that denominator;
averaging separately normalized per-rank losses is incorrect when their action
mask counts differ.

Outcome-only budgets are frozen after the dense seed-0 run supplies measured
tokens/second, transitions/second, and optimizer FLOPs. They must include
predeclared checkpoints on all three axes rather than claiming equivalence on
only one of them.

### 5.4 Context length

Context length is not guessed. During `t4x2_preflight`, generate at least
100,000 deterministic teacher trajectories and record the byte-token length
distribution. Select the smallest power of two that exceeds the 99.9th
percentile with at least 10% headroom. No training example may be silently
truncated. An overlong trajectory is either rejected before the run under a
declared rule or causes a loud failure.

The selected value is then frozen in every main-run configuration.

---

## 6. CPU throughput on the actual Kaggle CPU

The throughput decision must be measured inside the committed Kaggle session,
before GPU training, using the CPU allocation that will actually feed the two
T4s. A local workstation measurement does not satisfy this gate.

### 6.1 Required CPU measurements

Measure separately:

1. accepted sampled instances per second;
2. world transitions plus teacher queries per second;
3. rendered and packed tokens per second;
4. packed tokens delivered through the shard reader per second;
5. Python batched calls versus per-episode calls; and
6. CPU utilization, worker count, peak RSS, and bytes written/read.

Run each case after one warm-up and for at least 30 seconds or 1,000 batches,
whichever is longer. Report median, interquartile range, and total work rather
than a single fastest iteration.

### 6.2 Raw-text comparison

The raw-text baseline must use materialized UTF-8 transcripts produced before
the timed section. Its timed path performs only:

```text
read materialized text -> byte tokenize -> pack -> deliver batch
```

The world path performs:

```text
sample -> validate -> execute teacher -> render -> byte tokenize -> pack
-> deliver batch
```

Both paths must use the same total source bytes, tokenizer, maximum sequence
length, packing policy, storage directory, consumer batch shape, and worker
count. Otherwise the ratio is not meaningful.

The original Step 1 engineering targets were:

- offline world generation and packing should reach at least 80% of the
  measured raw-text pipeline on the Kaggle CPU; and
- it should either exceed measured trainer consumption by 2x or keep the
  bounded training queue nonempty throughout a representative run.

If online CPU generation cannot feed the two T4s but deterministic offline
shards can, use offline shards. Do not respond by moving world generation onto
the GPU.

All CPU benchmark results are retained even when a target is missed.

> **Post-measurement amendment, 2026-08-17:** The 80% raw-text-relative target
> was an unvalidated planning estimate. The observed ratio of about 66% is
> non-blocking because packed generation and the measured DataLoader path were
> sufficient for the completed two-T4 runs. A throughput result blocks a later
> run only when it demonstrates an actual inability to feed the trainer through
> the selected offline or online path. Preserve the comparison for diagnosis;
> do not optimize to the old percentage for its own sake.

---

## 7. Dataset and DataLoader contract

### 7.1 Shard creation

Generate deterministic offline teacher shards under the run directory. Every
shard carries the existing world, generator, teacher, rendering, tokenizer,
seed, structural-parameter, replay, and content-hash fields.

Train, validation, structural holdout, and representation-transfer data must
be distinct by construction. Dataset creation must fail if a structural
combination assigned to validation or test appears in training.

### 7.2 Reading and batching

The Python reader must:

- memory-map or sequentially stream the existing binary shard format;
- never parse JSON in the per-token hot path;
- preserve complete-trajectory boundaries recorded by the packer;
- return token IDs as `torch.long`;
- return loss masks and target-channel IDs with identical shape;
- pad only in the collator with the declared `PAD` ID;
- use pinned host memory for CUDA transfer;
- use persistent workers after the first epoch where supported; and
- bound prefetching so it cannot exhaust Kaggle RAM.

The total DataLoader worker count is derived from the actual CPU allocation.
It is a global budget divided across ranks, not a per-rank copy of all CPU
cores. The resolved count is recorded.

### 7.3 Distributed sampling

Each packed sequence is consumed by exactly one DDP rank per epoch. The sampler
is seeded from `(root_seed, epoch)`, calls `set_epoch`, and has tests proving
that rank partitions neither overlap nor omit examples except for an explicitly
reported `drop_last` remainder.

The global token batch and optimizer update schedule must be independent of
the number of ranks:

```text
global tokens/update
= microbatch tokens/rank
* 2 ranks
* gradient accumulation steps
```

---

## 8. Dual-T4 training contract

### 8.1 Process launch

After CPU preparation, the parent runner launches:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m step1_experiments.train \
  --resolved-config <path> \
  --run-dir <path>
```

There is one process per T4. Each process sets its CUDA device from
`LOCAL_RANK` before constructing the model or allocating tensors. Use PyTorch
DistributedDataParallel, not `DataParallel`.

Only rank 0 writes manifests, metrics, plots, and checkpoints. All ranks join a
barrier before rank 0 atomically finalizes those files.

### 8.2 Precision and attention

T4 does not provide BF16 training. Use FP16 autocast with `GradScaler` and
retain FP32 optimizer states. Use PyTorch scaled-dot-product attention without
forcing a kernel unsupported by the installed PyTorch/CUDA/T4 combination.

Do not enable `torch.compile` for the first retained runs. It adds compilation
and dynamic-shape failure modes without testing the scientific claim.

### 8.3 Microbatch preflight

Before the timed run, execute complete forward, backward, optimizer, zero-grad,
checkpoint-save, checkpoint-load, and evaluation steps.

Only during this preflight, CUDA OOM may trigger bounded batch adjustment:

1. start from the configured microbatch;
2. on OOM, halve it;
3. increase gradient accumulation to preserve global tokens per update;
4. retry at most three times; and
5. write the final value to `resolved_config.json`.

Once real training begins, the batch shape is frozen. An OOM during training is
a failure, not permission to silently change the experiment.

### 8.4 Checkpointing and resume

Write checkpoints atomically by saving to a temporary file, flushing it, and
renaming it. A checkpoint contains:

- model, optimizer, scaler, and scheduler state;
- global step, tokens, transitions, and wall time;
- Python, NumPy, CPU Torch, and every CUDA RNG state;
- sampler epoch and offset;
- resolved configuration hash and Git SHA; and
- dataset manifest hashes.

Keep `latest`, the latest evaluation-best checkpoint, and periodic recovery
checkpoints. Checkpoint by elapsed time as well as optimizer steps so a slow
run does not go hours without recoverable state.

`--resume auto` may resume only a checkpoint whose Git SHA, configuration hash,
tokenizer hash, and dataset hashes match. A mismatch is fatal. Previous Kaggle
output attached under `/kaggle/input` may be used as a read-only resume source;
the runner never overwrites it and never asks the user to locate a file.

---

## 9. Tensor and numerical failure requirements

The runner must turn common tensor failures into explicit artifacts rather
than an unexplained stopped notebook.

Before every first use of a new data path, assert:

- token tensor dtype is `torch.long`;
- every token ID is in `[0, vocab_size)`;
- token, loss-mask, target-channel, and attention-mask shapes agree;
- sequence length does not exceed context length;
- each training batch contains at least one supervised action token;
- padding positions have zero loss;
- logits, loss, gradients, and optimizer statistics are finite;
- tensors used together are on the same expected device; and
- no batch is empty on either rank.

The loss implementation must test the one-token causal shift explicitly. A
target mask attached to token `t` must score the logits that predict token `t`,
not the logits produced after consuming it.

On any uncaught exception, non-finite value, CUDA error, NCCL error, dataloader
worker death, or failed assertion, rank 0 writes a failure bundle containing:

- exception class, message, and full traceback;
- last completed phase and global step;
- last batch shapes, dtypes, min/max token IDs, and mask counts;
- recent loss and gradient statistics;
- resolved config and manifest hashes;
- `nvidia-smi`, free disk, and memory summaries;
- the tail of every rank's log; and
- the last valid checkpoint path and hash.

The parent runner also watches the `torchrun` exit code. If rank 0 itself dies
before it can write diagnostics, the parent creates the failure bundle from
the rank logs, last heartbeat, environment snapshot, and last checkpoint.

Do not dump secrets or an entire dataset batch. After packaging the failure,
the runner exits nonzero. `CUDA_LAUNCH_BLOCKING=1` is allowed only in a separate
diagnostic configuration, never in throughput or retained training runs.

---

## 10. Minimal instrument check

Calibration is not a model-scale study. Run one seed on the fixed 21M model,
using no more than 2% of the planned dense-training token budget.

It has only two pass/fail tasks:

1. **variable-gap associative retrieval**, with randomized positions and gaps
   so a fixed-offset rule cannot solve it; and
2. **tiny world overfit**, on a small frozen set of valid teacher trajectories,
   proving the tokenizer, loss mask, model, optimizer, and decoder work
   together.

Thresholds are fixed in `t4x2_preflight.toml` before the run. The instrument
check does not inspect Rendering B, the structural test set, or final transfer
metrics. Do not try smaller models, tune architectural variants, or turn this
into a reported result.

If it fails, stop. Diagnose the instrument before implementing or spending
compute on outcome-only training.

---

## 11. Main run sequence

Run these as separate committed notebook versions, each using a pinned config
and Git SHA:

1. `t4x2_preflight`: environment, tests, CPU/GPU throughput, instrument check;
2. dense teacher seed 0;
3. dense teacher seeds 1 and 2 after seed 0 passes end-to-end evaluation;
4. target-shuffled control with the predeclared seed policy;
5. outcome-only seed 0 only after the dense path is scientifically readable;
6. outcome-only seeds 1 and 2; and
7. the informative surface-tokenizer diagnostic only if the primary result
   cannot separate syntax acquisition from process acquisition.

Do not spend three seeds proving that an implementation is broken. Each first
seed is an end-to-end gate for the remaining paired seeds, not a license to
tune on the final evaluation metrics.

Every trained condition uses paired initialization seeds, the same semantic
instance stream, the same evaluator, and the same model configuration. Report
results against rendered tokens, supervised tokens where applicable, world
transitions, optimizer FLOPs, and wall time. Do not pretend that one of these
budgets alone makes dense supervision and RLVR identical.

---

## 12. Evaluation produced by the runner

Evaluation is non-interactive and runs immediately after training from the
checkpoint selected by the declared rule. It includes:

- unseen seeds within trained structural combinations;
- held-out structural combinations;
- declared larger/deeper extrapolation instances;
- Rendering B over the fixed interface-calibration budget;
- reversible versus irreversible matched surfaces;
- forced-prefix recovery states;
- malformed-action rate;
- success, probe cost, regret to teacher, and action calibration; and
- throughput and dataloader-wait measurements from the actual training run.

The evaluator executes decoded actions through the Rust parser and world. It
must not score a privileged typed action selected directly from model logits
unless the run is explicitly the informative-tokenizer diagnostic.

---

## 13. Filesystem, retention, and result bundles

Ephemeral execution state and published output have separate roots:

```text
/tmp/step1-runtime/
  actuallybuildingstuff/          # exact detached source checkout
  caches/                          # dependency and compiler caches
  build/                           # disposable build products

/kaggle/working/step1-results/
  <run-id>/
    run_manifest.json
    resolved_config.json
    phase_status.json
    environment/
    logs/
      runner.log
      rank0.log
      rank1.log
      metrics.jsonl
    benchmarks/
      cpu.json
      dataloader.json
      gpu.json
    datasets/
      manifests/
    checkpoints/
    evaluation/
    plots/
    analysis/
      summary.json
      result-report.json
      provenance.json
    SUCCESS                        # or FAILURE
  <run-id>-analysis.tar.gz
  <run-id>-analysis.sha256
  <run-id>.tar.gz                  # remote recovery payload
  <run-id>.sha256
  latest-summary.json
```

The clone is read-only after checkout. No source file, `.git` directory,
Python cache, dependency cache, or Cargo target may be published beneath
`/kaggle/working`.

At exit, successful or failed, create two logical payloads:

1. The **analysis payload** contains only provenance, resolved configuration,
   phase status, compact reports, final metrics, plots, and bounded diagnostic
   logs. It must contain no model weights, optimizer state, replay shards, or
   checkpoints and must remain below 50 MiB.
2. The **recovery payload** contains the selected checkpoint and the minimum
   state needed to reproduce, resume, or feed a declared downstream run. Large
   intermediate shards and redundant checkpoints may be excluded when their
   manifests and deterministic regeneration keys are retained.

Both payloads receive SHA-256 sidecars. `latest-summary.json` remains directly
readable. The recovery payload and expanded checkpoint tree stay on Kaggle by
default; only the analysis payload is collected onto the user's device.

## 14. Submission, retrieval, and audit workflow

### 14.1 Normal path

After the user explicitly authorizes an experiment, the orchestrator must:

1. resolve the experiment registry entry and exact full Git SHA;
2. verify that SHA is reachable from the configured Git remote;
3. render the launcher and current `kernel-metadata.json` into a temporary
   staging directory;
4. include declared upstream Kaggle output slugs as `kernel_sources`, together
   with their expected content identities;
5. submit with `kaggle kernels push --accelerator NvidiaTeslaT4`;
6. capture the exact immutable `owner/slug/version`, not merely the mutable
   latest-notebook slug;
7. poll that exact version and surface logs on failure;
8. use filtered Kaggle output retrieval to fetch only `latest-summary.json`,
   the analysis payload, and their checksum sidecars;
9. verify checksums and identities before unpacking; and
10. write a compact audit receipt under `step1/audit/runs/<run-id>/`.

The full `/kaggle/working` tree must not be downloaded. The UI download button
is an exceptional recovery path, not an operator step.

### 14.2 Cloud-to-cloud dependencies

A downstream experiment consumes a heavyweight upstream result by declaring
its source notebook slug in `kernel_sources`. Kaggle metadata binds a source by
`owner/slug`, so an accepted artifact slug is immutable by project policy: do
not publish a different artifact under it. Kaggle exposes the attached output
under `/kaggle/input`; the project locator must still validate the expected Git
SHA, config hash, model-state hash, dataset hash, and report contract before
use. A directory name or an unchecked "latest" relationship is never
sufficient evidence.

Promote an artifact to a private Kaggle Dataset or Model only when notebook
output is unsuitable for repeated reuse or retention. Do not route a
checkpoint through the user's device merely to attach it to another Kaggle
run.

### 14.3 Tracked audit receipt

`receipt.json` is the durable link between Git and Kaggle. It must contain at
least:

```text
schema version
run ID and experiment name
exact Kaggle owner/slug/version and URL
Kaggle terminal status and completion time
source Git remote and full SHA
configuration path and canonical SHA-256
root seed and observed accelerator inventory
upstream Kaggle version references and expected content hashes
analysis artifact paths, sizes, and SHA-256 values
recovery artifact remote path, size, and SHA-256
scientific report path, when one exists
```

Commit the receipt, compact machine-readable report, summary, and selected
plots only after verification. Do not commit raw shards, checkpoints, notebook
cell output, complete logs, or recovery archives. A prose report without a
receipt is documented but not repository-auditable; a Kaggle status of
`COMPLETE` proves batch completion, not the scientific contract.

### 14.4 Existing-run audit backfill

The migration starts by backfilling compact evidence for the existing runs,
without downloading their heavyweight payloads. As observed on 2026-08-16,
these notebook slugs remain reachable and report `COMPLETE`:

```text
aniruddhavarma/step1-t4x2-preflight-20fd8db
aniruddhavarma/step1-t4x2-dense-seed0-84f2938
aniruddhavarma/step1-t4x2-dense-seed1-e592ba1
aniruddhavarma/step1-rendering-b-transfer-e592ba1
aniruddhavarma/step1-rendering-b-transfer-seed1-42e7502
aniruddhavarma/step1-rendering-b-terminal-seed0-b7ccae9
aniruddhavarma/step1-rendering-b-terminal-seed1-b7ccae9
aniruddhavarma/step1-rlvr-smoke-de89b52
```

For each run, discover and record the exact version, fetch only compact result
JSON/checksum files, validate them against the retained Git commit and config,
and create a receipt. Record missing or irrecoverable evidence explicitly; do
not infer it from a notebook title. The existing local failed-preflight archive
may be indexed after its matching checksum is verified, but bulk legacy output
downloads are not part of the backfill.

Backfill and orchestration implementation may proceed independently. A legacy
run may not support a new scientific claim until its receipt passes; this does
not block implementing the new control plane or launching a new explicitly
authorized smoke run that produces a receipt itself.

## 15. Acceptance criteria for the Kaggle implementation

The T4 execution path is complete when a fresh CLI-submitted version:

1. renders from a declared experiment and verifies an exact remotely reachable
   Git SHA without user input;
2. requests T4 through the official CLI and records the two actual T4 devices;
3. leaves the source clone and all build/cache debris under ephemeral storage;
4. builds and tests the Rust/Python world implementation;
5. measures the raw-text and world pipelines on the actual Kaggle CPU;
6. produces deterministic, hashed shards and reads them without Python
   per-example bottlenecks;
7. launches exactly two DDP ranks and completes a synchronized optimizer step;
8. survives the instrument check without a fixed-offset shortcut;
9. checkpoints and resumes with matching hashes;
10. converts an injected tensor-shape or non-finite failure into retained
    diagnostics;
11. completes evaluation without interactive intervention;
12. leaves verified analysis and recovery payloads with checksum sidecars;
13. downloads no checkpoint during normal collection;
14. records the exact immutable Kaggle version and writes a valid tracked audit
    receipt; and
15. can attach and identity-check an upstream Kaggle output without local
    transfer.

Only after these criteria pass should a new scientific result be accepted.

## 16. Deliberate non-goals

Do not add any of the following for the first T4 execution path:

- a general cluster launcher;
- Kubernetes, Ray, Slurm, or a parallel Kaggle API implementation;
- automatic GPU launches on every Git push;
- a multi-workflow GitHub Actions scheduler or external experiment dashboard;
- mirroring recovery archives into Git or GitHub Actions artifacts;
- TPU support;
- model-size sweeps;
- automatic architecture search;
- on-GPU world generation;
- a general-purpose tokenizer trainer;
- silent recovery that changes batch size, data, precision, or optimizer; or
- interactive notebook analysis cells.

The CLI launches. Kaggle runs. The receipt proves what happened. The compact
analysis payload is the only normal download.

## 17. Velocity-first migration sequence

> **Streams 1 and 2 are complete and the merge gate has passed.**
> `tools/kaggle_run.py` and [step1/kaggle/experiments.toml](step1/kaggle/experiments.toml)
> implement the control plane; the single rendered launcher clones to
> `/tmp/step1-runtime` and keeps `/kaggle/working` output-only; `package()`
> emits the bounded analysis payload alongside the recovery payload. The gate
> was an authorized RLVR run submitted through the CLI that produced a verified
> receipt — in the event, five of them, tracked under `step1/audit/runs/`.
>
> Operating instructions and the platform behaviours discovered while getting
> there are in [step1/kaggle/RUNBOOK.md](step1/kaggle/RUNBOOK.md). Two are worth
> promoting into this contract because they contradict reasonable assumptions:
> Kaggle derives the notebook slug from the metadata **title**, not `id`, so the
> staged title must be the slug; and `--accelerator NvidiaTeslaT4` provisions
> **T4 ×2**, confirmed by device inventory rather than by documentation, which
> is why §1.3's rule that the requested accelerator is not evidence earns its
> place.
>
> **Stream 3, audit backfill, remains outstanding.** The runs in §14.4 predate
> the payload machinery and still have no receipts.

After this contract is committed, agents may work in parallel on three bounded
streams:

1. **Control plane:** implement `tools/kaggle_run.py`, ephemeral metadata
   staging, the experiment registry, exact-version capture, polling, filtered
   collection, and receipt generation.
2. **Runtime and output hygiene:** move every launcher clone to `/tmp`, keep
   `/kaggle/working` output-only, and add the bounded analysis payload without
   changing model, trainer, data, or evaluation semantics.
3. **Audit backfill:** recover compact evidence and exact version identifiers
   for the runs in Section 14.4 and create receipts, without downloading model
   payloads.

The merge gate is one local contract-test pass followed by one explicitly
authorized RLVR smoke run through the new CLI path. The smoke must produce a
verified receipt and analysis payload. After that gate, manual upload/download
instructions are removed from active launcher text; historical notebooks remain
unchanged as execution evidence.

---

## Platform references

- Kaggle's GPU-usage documentation states that committed batch sessions run
  notebook code from top to bottom and describes API-driven non-interactive
  notebook submission: <https://www.kaggle.com/docs/efficient-gpu-usage>
- Kaggle CLI notebook commands and accelerator identifiers:
  <https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md>
- Official notebook-output download behavior:
  <https://github.com/Kaggle/kagglehub>
