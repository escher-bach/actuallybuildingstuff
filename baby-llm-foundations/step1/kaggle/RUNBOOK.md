# Step 1 Kaggle Runbook

Operational guide for `tools/kaggle_run.py`, the sole operator entry point
defined by [EXPERIMENT-EXECUTION-PLAN.md](../../EXPERIMENT-EXECUTION-PLAN.md)
§1.4. This document records how to run an experiment and the platform
behaviours that cost real submissions to discover.

---

## 1. Running an experiment

```bash
python tools/kaggle_run.py launch  --experiment rlvr-klanchor-seed0
python tools/kaggle_run.py status  --experiment rlvr-klanchor-seed0
python tools/kaggle_run.py logs    --experiment rlvr-klanchor-seed0
python tools/kaggle_run.py collect --experiment rlvr-klanchor-seed0
```

`run` does launch → wait → collect in one blocking call. For anything longer
than a few minutes, prefer `launch` and poll with `status`, because a dropped
local shell must never be able to lose a run.

Preconditions the tool enforces before it will submit:

1. the experiment is declared in [experiments.toml](experiments.toml);
2. the launcher, config, and `step1/python` paths have no uncommitted changes;
3. the commit is reachable from a ref on the configured remote. **The tool
   never pushes Git.** Push first, yourself.

It then renders the launcher *from the pinned commit* — not from your working
tree — stages a fresh `kernel-metadata.json` in a temporary directory, submits
with `--accelerator NvidiaTeslaT4`, and records the immutable
`owner/slug/version` under `step1/audit/runs/submitted/`.

## 2. What lands where

| path | contents |
|---|---|
| `step1/audit/runs/submitted/<slug>.json` | submission reference: exact version, commit, upstream identities |
| `step1/audit/runs/<run-id>/receipt.json` | the durable Git↔Kaggle link (§14.3 of the execution plan) |
| `step1/audit/runs/<run-id>/result-report.json` | the stage's own machine-readable result |
| `step1/audit/runs/<run-id>/{summary,provenance}.json` | run identity, phase status, observed accelerators |

`collect` downloads **only** `latest-summary.json`, the analysis payload, and
its checksum sidecar; it verifies the checksum against the run's own summary
before unpacking. Checkpoints and the recovery payload stay on Kaggle. Pass
`--keep-payload <dir>` to also retain the verified archive locally — useful for
reading `logs/training_log_history.json`, which is in the payload but is not
copied into the audit directory.

## 3. Adding an experiment

Add a block to `experiments.toml` and a config under `step1/configs/kaggle/`.
If the run consumes an upstream Kaggle output, declare it twice — as
`kernel_sources` for Kaggle to mount, and as `upstream_identity` for the
project to verify — and put the same hashes in the config's `[source]` table.
A contract test asserts those agree and are full-length; it exists because a
hand-transcribed hash was once two characters short and would have failed
inside the session, after a submission.

Never edit a slug that has published an artifact. The slug embeds the commit,
so a new commit gets a new slug automatically.

## 4. Platform behaviours worth knowing

These were established by submissions that failed, not by documentation.

**Kaggle derives the notebook slug from the metadata `title`, not `id`.** A
prose title publishes the run under a slugified version of it — mutable and
commit-independent. The tool therefore sets `title` to the slug itself and
verifies the URL Kaggle returns against the one requested, failing loudly on a
mismatch.

**`--accelerator NvidiaTeslaT4` provisions T4 ×2.** The CLI documentation lists
no dual-T4 value and Kaggle's UI offers no single-T4 option; the runner's
`accelerator_check` phase confirmed two Tesla T4 devices. That check runs
before the build precisely so a mis-provisioned session fails in under a minute
instead of after ten.

**`kernels output --file-pattern` matches the path relative to
`/kaggle/working`,** not the bare filename. A pattern for `latest-summary.json`
matches nothing; it lives at `step1-results/latest-summary.json`.

**`kernels logs` returns JSON stream records,** not plain text, and the CLI
prints notebook output containing non-ASCII. On a Windows console that kills
the CLI with a `charmap` error unless `PYTHONIOENCODING=utf-8` is set, which
the tool does for every invocation.

**There is no `kernels cancel`.** Only `delete`, which removes the kernel
entirely. A wrong submission has to be allowed to finish or be abandoned.

**The image no longer ships Cargo,** so the runner's rustup fallback is the
normal path, not an exception. It costs about two minutes.

**Failed runs still publish output.** A run that raises still produces the full
analysis payload, `phase_status.json`, and `failure.json`, so `collect` works
on failures and is usually the fastest way to diagnose one. `logs` shows the
notebook-level traceback; the payload shows which phase failed and why.

## 5. Environment redirects: don't

The launcher clones into `/tmp/step1-runtime` and keeps `/kaggle/working`
output-only. That alone satisfies the hygiene contract, because everything the
build writes lives under the clone or under `~/.cache`.

Adding further redirects caused two of the three build-phase failures in this
stage: `CARGO_HOME` moved the toolchain away from a hard-coded `~/.cargo/bin`
lookup, and `CARGO_TARGET_DIR` moved the wheel away from a hard-coded
`target/wheels` lookup. Both are fixed — the build derives Cargo's location
from `CARGO_HOME` and passes `--out` to maturin — but the lesson generalises:
if a redirect is not required by the contract, it is pure risk.

## 6. Observed timings

On T4 ×2, for planning a session:

| phase | cost |
|---|---|
| clone, pip install, rustup, maturin build | 10–13 min |
| `cargo test --workspace` plus the Python suite | ~3 min |
| RLVR training, 12,224 rollout episodes | ~17 min |
| RLVR training, 48,896 rollout episodes | ~68 min |
| milestone evaluation, 512 episodes | ~2 min |
| final matched evaluation, 4 × 1,024 episodes | ~8 min |

## 7. Failure triage

1. `python tools/kaggle_run.py status --experiment <name>` — `ERROR` or
   `COMPLETE`?
2. `... logs --experiment <name>` — the notebook-level exception.
3. `... collect --experiment <name> --keep-payload <dir>` — works on failures;
   read `phase_status.json` for the failed phase and `logs/build.log` or
   `logs/rlvr.log` for the underlying subprocess output.
4. The runner truncates subprocess tails to 120 lines in its exception, and pip
   progress bars can flood that window. The full log is always in the payload.
