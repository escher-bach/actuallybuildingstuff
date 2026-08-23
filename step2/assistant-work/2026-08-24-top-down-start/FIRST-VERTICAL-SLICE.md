# First Architecture–World Vertical Slice

**Date:** 2026-08-24
**Status:** local CPU gates passed; authorized two-T4 retry pending after
image-toolchain and diagnostic-schedule apparatus failures

## 1. Why this must precede checkpoint evidence

The first checkpoint-producing session cannot simultaneously be the first time
the architecture, token ABI, Rust oracle, distributed state, and checkpoint
format are shown to work. The first experiment is therefore one allocation
with a strict boundary:

```text
apparatus gate on disposable diagnostic weights
  -> discard every diagnostic weight
  -> reinitialize the selected model from blank
  -> bounded c1-start-candidate training
  -> retained checkpoint plus held-out/closed-loop evidence
```

Failure before the reset is an apparatus failure. Passing the apparatus gate
does not validate the world or establish transfer. The post-reset artifact is
called a **candidate**, not a completed first developmental checkpoint.

## 2. Implemented ownership boundary

- Rust owns `W_calibrated_monomial[d=1..4]`, deterministic generation, latent
  transition execution, public transcript serialization, the public-prefix
  teacher/oracle, verifier, and online rollout state.
- PyO3 exposes padded batched public tensors and an online action boundary.
  Privileged oracle actions exist only behind a method explicitly named
  `privileged_oracle_actions` for verifier controls.
- Hugging Face `LlamaModel` owns all transformer blocks, causal attention,
  RoPE, RMSNorm, and SwiGLU.
- The project adapter owns role/payload embeddings, deterministic local-key
  encoding, shared action/outcome heads, and the random visual patch/resampler
  path.
- Accelerate owns FP16, two-rank DDP, gradient synchronization, and state
  save/restore.

## 3. Local CPU evidence

The following passed before GPU launch:

- eight Rust tests covering unsafe calibration rejection, dimension
  stratification, replay determinism, no serialized action target, no
  calibration clipping, public-oracle/latent-teacher equivalence, bounded zero
  policy failure, and public-oracle closed-loop completion;
- exact selected 12-layer, width-384 model forward/backward on a real Rust
  trajectory; the maintained backbone has 21,243,648 parameters and the full
  implemented learner has 22,147,985;
- fixed-real-batch overfit with a reduced test profile;
- random visual path shape and gradient flow;
- standard `save_pretrained`/`from_pretrained` output identity;
- Accelerate real-batch optimization and exact state restore;
- model-driven and oracle-driven online rollouts through the same Rust state;
  and
- generated Kaggle launcher contract and registry-path tests.

The tests found and corrected one real checkpoint bug: deterministic Fourier
key frequencies were initially a non-persistent buffer, and Transformers'
low-memory reload path did not reconstruct that buffer. Persisting it restored
exact output identity.

The non-authoritative workstation benchmark over 4,096 generated trajectories
reported median rates of approximately 1,576 episodes/s and 104,139 unpadded
tokens/s through Rust generation, public oracle, serialization, and the PyO3
boundary. The raw result is in
[local-cpu-benchmark.json](local-cpu-benchmark.json). Kaggle's CPU measurement,
not this workstation number, determines whether the producer can feed two T4s.

## 4. Authorized two-T4 gate

The committed run uses the exact 12x384 model and must establish:

1. exactly two visible T4 GPUs and two completed Accelerate ranks;
2. finite forward/backward on real Rust batches;
3. fixed-batch loss after 128 scheduled diagnostic updates no more than 80% of
   its initial value, using a four-episode global cohort that still covers
   dimensions 1–4;
4. checkpoint mutation followed by exact Accelerate state restoration;
5. diagnostic weights discarded before lineage initialization;
6. 256 bounded post-reset updates over 4,096 generated encounters;
7. portable safetensors/config output and retained Accelerate recovery state;
8. teacher-forced action/outcome errors by dimension;
9. model and public-oracle closed-loop results; and
10. actual Kaggle CPU generation, conversion, setup, and elapsed-time evidence.

Only items 1–5 are architecture/apparatus gates. Items 6–10 characterize the
candidate checkpoint. They are not silently converted into a claim that the
first world transferred or that the developmental session is complete.

## 5. Run path

The authoritative amendment is in
[EXPERIMENT-EXECUTION-PLAN.md](../../../EXPERIMENT-EXECUTION-PLAN.md). The sole
launcher is:

```text
python tools/kaggle_run.py launch --experiment architecture-world-vertical-slice
```

The heavyweight candidate checkpoint remains in Kaggle output. Only compact
audit evidence is retrieved locally.

Kaggle's current GPU image does not guarantee Cargo. The runner therefore uses
the standard `rustup` path to install the pinned minimal Rust `1.88.0`
toolchain under `/tmp`. Its measured install/build time is setup overhead, not
world-generation throughput. The first remote attempt (`54bfa53`, version 1)
stopped at Maturin before importing the world or model because Cargo was absent;
it carries no scientific evidence about either component.

The second remote attempt (`4fd3300`, version 1) passed installation, all
correctness tests, 4,096 generated-world checks, and the CPU benchmark. Its
diagnostic loss rose from `0.8807` to `0.9686`; inspection showed that the
diagnostic used an unscheduled full learning rate even though the retained
training recipe uses warm-up. That is an apparatus/optimizer-gate failure, not
a world-validity failure. The retry aligns the disposable diagnostic with the
declared warm-up and cosine schedule; it does not relax the blank reset or
retain diagnostic weights.
