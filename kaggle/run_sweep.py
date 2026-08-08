"""Kaggle entry point for the dial sweep (Task Spec section 8 step 5).

Paste into a Kaggle notebook cell, or attach the repo as a dataset and run it.
GPU on: Settings -> Accelerator -> GPU T4 x2 (both are used) or P100.

    !git clone -q https://github.com/<you>/actuallybuildingstuff /kaggle/working/repo
    %run /kaggle/working/repo/kaggle/run_sweep.py

Nothing here does any science. It picks a preset from the device, runs the
sweep, and writes results somewhere that survives the session ending. All the
decisions live in `repertoire.harness.sweep`.

--------------------------------------------------------------------------
Read this before spending a session
--------------------------------------------------------------------------

**Run `check` first.** It needs no GPU and takes seconds, and it is the section 8
step 1 and 2 gates. A sweep on a family whose gates fail measures nothing.

**Run `smoke` first.** Two minutes, and it exercises every path the real sweep
uses. Its *numbers* are noise -- the harness will refuse to read them, which is
the point.

**Then run `calibrate`, and this is the default.** It trains the L0 arm alone at
a few budgets. L0 states theta outright, so its Bayes floor is exactly 0 and the
final loss IS the distance from optimal -- and if L0 does not come down, nothing
above it is measurable. The first real T4 sweep cost ~50 GPU-minutes to discover
that the model had learned the answer marginal rather than the task, at every
dial setting. Calibration is one arm instead of 27 and answers that first.

**Both GPUs are used automatically.** Every dial point trains a fresh model from
scratch on its own stream, so the points are independent: no gradient exchange,
no synchronisation, an exact 2x on "T4 x2". Not DDP -- that splits one model's
gradient, which is the wrong tool when the runs are separate and each fits on
one card.

**Run both families.** The worked family fails A4 at these moduli, so a collapse
on it alone is ambiguous between "supervision above L0 does not work" and "this
family was memorized". Parity has the opposite property. `MODE = "full"` does
both.

**A session that dies is not a session lost.** Every dial point is checkpointed
under OUT and a rerun resumes. Kaggle's /kaggle/working persists between runs of
the same notebook; anything elsewhere does not.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

# --------------------------------------------------------------------------
# Configuration -- the only part meant to be edited
# --------------------------------------------------------------------------

MODE = os.environ.get("SWEEP_MODE", "calibrate")  # check | calibrate | smoke | full
OUT = Path(os.environ.get("SWEEP_OUT", "/kaggle/working/sweeps"))
REPO = Path(__file__).resolve().parent.parent


def _setup() -> None:
    src = REPO / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))


def _preset() -> str:
    """T4 and P100 differ in what fp16 buys, so they get different widths.

    Not a performance tweak: P100 is sm_60, where fp16 has no tensor cores and
    is slower and noisier than fp32. A preset that assumed otherwise would run
    slower and measure something slightly different.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            print("no GPU visible -- falling back to the smoke preset.")
            print("On Kaggle: Settings -> Accelerator -> GPU T4 x2 or P100.")
            return "smoke"
        name = torch.cuda.get_device_name(0)
        print(f"device: {name}")
        return "p100" if "P100" in name else "t4"
    except ImportError:
        return "smoke"


def run(*args: str) -> int:
    cmd = [sys.executable, "-m", "repertoire.harness", *args]
    print(f"\n$ {' '.join(cmd)}\n", flush=True)
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    return subprocess.call(cmd, env=env, cwd=str(REPO))


def n_gpus() -> int:
    try:
        import torch

        return torch.cuda.device_count()
    except ImportError:
        return 0


def run_sharded(*args: str) -> int:
    """Split a sweep across every visible GPU, one process each.

    **The parallelism here is free and exact.** Every dial point trains a fresh
    model from scratch on its own episode stream, so there is no gradient to
    exchange and nothing to synchronise -- Kaggle's "T4 x2" gives a 2x wall-clock
    speedup with no communication at all. Each shard writes its own per-point
    records, and the last one to finish merges them.

    Not torchrun, not DDP: those exist to split *one* model's gradient across
    devices, which is the wrong tool when the runs are independent and each model
    fits comfortably on one card.
    """
    n = max(1, n_gpus())
    if n == 1:
        return run(*args)

    print(f"\nsplitting across {n} GPUs -- the dial points are independent, so this "
          f"is an exact {n}x with no communication\n", flush=True)
    env_base = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    procs = []
    for i in range(n):
        cmd = [sys.executable, "-m", "repertoire.harness", *args, "--shard", f"{i}/{n}"]
        env = dict(env_base, CUDA_VISIBLE_DEVICES=str(i))
        print(f"$ CUDA_VISIBLE_DEVICES={i} {' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(REPO)))
    return max(p.wait() for p in procs)


def main() -> int:
    _setup()
    OUT.mkdir(parents=True, exist_ok=True)

    # The gates first, always. They need no GPU and a sweep past a failing gate
    # measures nothing -- Task Spec section 8: "Do not proceed past a failing gate."
    modular_ok = run("check", "--family", "modular", "--k", "0") == 0
    if not modular_ok:
        print("\nGATE FAILED for the worked family. Stopping, per section 8.")
        return 1

    # Parity is the A4 control, and its gate is EXPECTED to fail right now: its
    # sampler does not match the uniform prior its posterior assumes (docs/10
    # section 5, measured at 31.4x). That corrupts its measured x-axis, so the
    # control arm is skipped rather than run and quietly believed.
    parity_ok = run("check", "--family", "parity", "--k", "1") == 0
    if not parity_ok:
        print("\nparity's gate fails as documented -- skipping the A4 control arm.")
        print("Fix is one method on the family; see docs/10 section 5.")

    if MODE == "check":
        return 0

    if MODE == "smoke":
        return run("sweep", "--preset", "smoke", "--family", "stub", "--k", "1",
                   "--out", str(OUT))

    preset = _preset()

    # Calibration before the sweep, always. The first T4 sweep spent ~50 GPU
    # minutes to find out the model had learned the answer marginal and not the
    # task. L0 is the diagnostic: its Bayes floor is exactly 0, so L_final IS the
    # distance from optimal, and if L0 does not come down nothing above it is
    # measurable. One arm instead of 27.
    if MODE == "calibrate":
        return run("calibrate", "--preset", preset, "--family", "modular", "--k", "0")

    # The worked family, both dials. Section 8 step 5 says "take the worked
    # family", and both dials because one dial tests only that dial.
    rc = run_sharded("sweep", "--preset", preset, "--family", "modular", "--k", "0",
                     "--dial", "both", "--out", str(OUT))

    # Parity on the generic dial: the A4 control. Parity has no partial_preamble,
    # which is exactly why the observation dial exists.
    if parity_ok:
        rc |= run_sharded("sweep", "--preset", preset, "--family", "parity",
                          "--k", "1", "--dial", "observations", "--out", str(OUT))
    else:
        print("\n" + "=" * 70)
        print("A4 CONTROL NOT RUN. Any collapse on the worked family alone is")
        print("ambiguous between 'supervision above L0 does not work' and 'this")
        print("family was memorized' -- section 6 says it fails A4 at these moduli.")
        print("Do not read a collapse as the section 8 step 5 answer without it.")

    print("\n" + "=" * 70)
    print("Saved sweeps:")
    for p in sorted(OUT.rglob("sweep.json")):
        print(f"  {p}")
    print("\nRe-read any of them without retraining:")
    print(f"  PYTHONPATH=src python -m repertoire.harness read <path>")
    print("\nDownload the whole directory from the notebook's Output tab before")
    print("the session ends, or the checkpoints go with it.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
