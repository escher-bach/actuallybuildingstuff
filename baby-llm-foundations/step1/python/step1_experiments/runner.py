"""Single public Step 1 command: phase machine, diagnostics, and packaging."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import urllib.request
from pathlib import Path

from .artifacts import RunArtifacts, atomic_json
from .benchmarks import cpu_benchmark, dataloader_benchmark
from .data import ACTION, BOS, BinaryShard, END_TURN, OBS, TOKENIZER_HASH, VOCAB_SIZE, Sequence, assert_fast_tokenizer_parity, assert_rust_tokenizer_identity, encode_bytes, generate_rust_shard
from .environment import assert_two_t4s, capture, command, git_info


PHASES = ("capture_environment", "install_and_build", "correctness_tests", "cpu_throughput", "prepare_shards", "dataloader_throughput", "gpu_preflight", "train", "evaluate")


def _repo_root() -> Path: return Path(__file__).resolve().parents[3]


def _resolved(path: Path) -> tuple[dict, str]:
    with path.open("rb") as handle: config = tomllib.load(handle)
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest(); config["_meta"] = {"config_path": str(path.resolve()), "hash": digest, "tokenizer_hash": TOKENIZER_HASH}
    return config, digest


def _run_checked(args: list[str], log: Path, cwd: Path, timeout: int = 3600) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(args) + "\n"); handle.flush()
        try:
            result = subprocess.run(args, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, check=False, env={**os.environ, "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false"})
        except subprocess.TimeoutExpired:
            handle.write(f"\nTIMEOUT after {timeout} seconds\n"); handle.flush()
            result = None
    if result is None or result.returncode:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
        tail = "\n".join(lines[-120:])
        reason = f"timed out after {timeout}s" if result is None else f"failed ({result.returncode})"
        raise RuntimeError(f"subprocess {reason}: {' '.join(args)}\nlog: {log}\n--- child log tail ---\n{tail}")


def _build(repo: Path, artifacts: RunArtifacts) -> None:
    log = artifacts.run_dir / "logs" / "build.log"; cargo = shutil.which("cargo")
    if not cargo:
        # The Kaggle image no longer ships Cargo, so this bounded, non-interactive
        # install is the normal path.  The installer is build debris and stays out
        # of the published run directory.
        with tempfile.TemporaryDirectory(prefix="rustup-") as scratch:
            installer = Path(scratch) / "rustup-init"
            urllib.request.urlretrieve("https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init", installer)
            installer.chmod(0o755)
            _run_checked([str(installer), "-y", "--profile", "minimal", "--default-toolchain", "1.85.0"], log, repo, 900)
        # rustup installs into CARGO_HOME, which the launcher points at ephemeral
        # storage; only fall back to the default when it is unset.
        cargo_home = Path(os.environ.get("CARGO_HOME") or (Path.home() / ".cargo"))
        os.environ["PATH"] = f"{cargo_home / 'bin'}{os.pathsep}{os.environ['PATH']}"
        cargo = shutil.which("cargo")
        if not cargo: raise RuntimeError(f"pinned Rust 1.85.0 installation did not provide Cargo under {cargo_home}")
    _run_checked([sys.executable, "-m", "pip", "install", "-r", str(repo / "requirements-kaggle.txt")], log, repo, 900)
    _run_checked([sys.executable, "-m", "maturin", "build", "--release", "--manifest-path", str(repo / "step1" / "crates" / "world-py" / "Cargo.toml")], log, repo / "step1", 1800)
    wheels = sorted((repo / "step1" / "target" / "wheels").glob("world_py-*.whl"))
    if not wheels: raise RuntimeError("maturin did not produce a world_py wheel")
    _run_checked([sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[-1])], log, repo, 900)
    _run_checked([sys.executable, str(repo / "step1" / "python" / "smoke.py")], log, repo, 120)


def _correctness(repo: Path, artifacts: RunArtifacts) -> None:
    log = artifacts.run_dir / "logs" / "tests.log"
    _run_checked(["cargo", "test", "--workspace", "--locked"], log, repo / "step1", 1800)
    _run_checked([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"], log, repo / "step1" / "python", 300)
    # Python contracts: byte table, labels, and distributed partitions.
    script = """
import torch
from step1_experiments.data import *
from step1_experiments.data import collate
from world_py import Batch, FamilyParams, generate_teacher_shard, parse_action, render_action
from pathlib import Path
import tempfile
assert all(decode_bytes([i]).encode() == bytes([i]) for i in range(128))
for rendering in ('a', 'b'):
    text=render_action(0, 5, 6, rendering); assert parse_action(text, 5, 6, rendering)==0
assert_rust_tokenizer_identity()
assert_fast_tokenizer_parity()
params = FamilyParams(n_hyp=6,n_probe=5,n_evidence=2,cost_lo=1,cost_hi=3,budget_slack=1,min_depth=2,step_slack=2,variant='irreversible')
batch = Batch(params, seed=20260811, n_episodes=1)
prefix = [BOS, OBS] + encode_bytes(batch.observations('a')[0]) + [ACTION]
with tempfile.TemporaryDirectory() as d:
    binary, _, _ = generate_teacher_shard(params, 20260811, 1, 'a', 2048, d, 'protocol')
    sequence = BinaryShard(Path(binary))[0]
    assert sequence.tokens[:len(prefix)] == prefix
    end = sequence.tokens.index(END_TURN)
    assert sequence.loss[end] == 1 and all(sequence.loss[i] == 1 for i in range(prefix.__len__(), end + 1))
batch=collate([Sequence([1,2,3],[0,1,0],[0,1,0])], 8); assert batch['labels'].tolist()==[[-100,2,-100]]
"""
    _run_checked([sys.executable, "-c", script], log, repo / "step1" / "python", 120)


def _prepare(config: dict, artifacts: RunArtifacts) -> dict:
    root = artifacts.run_dir / "datasets"; world = config["world"]; seed = config["run"]["root_seed"]
    assert_rust_tokenizer_identity()
    partitions = [("train", world, seed, world["train_episodes"], world["rendering"]), ("validation", world, seed + 1_000_000, world["validation_episodes"], world["rendering"])]
    structural = {**world, "n_hyp": world["n_hyp"] + 1}; transfer = {**world, "rendering": "b"}
    partitions.extend([("structural", structural, seed + 2_000_000, world["structural_episodes"], world["rendering"]), ("transfer", transfer, seed + 3_000_000, world["transfer_episodes"], "b")])
    result = {}
    for name, params, part_seed, count, rendering in partitions:
        binary, manifest, replay = generate_rust_shard(params, part_seed, count, world["context_length"], rendering, root, name)
        result[name] = {"path": str(binary), "manifest": str(manifest), "replay": str(replay), "content_hash": hashlib.sha256(binary.read_bytes()).hexdigest()}
    if result["train"]["content_hash"] in {result["validation"]["content_hash"], result["structural"]["content_hash"], result["transfer"]["content_hash"]}: raise RuntimeError("dataset partition collision")
    return result


def _accelerator_check() -> dict:
    """Record the actual allocation; the requested accelerator is not evidence."""
    import torch

    assert_two_t4s()
    return {"devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]}


def _rlvr(config: dict, artifacts: RunArtifacts) -> dict:
    """Launch the outcome-only stage; it owns its own evaluation."""
    resolved = _save_resolved(config, artifacts)
    args = ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "step1_experiments.rlvr",
            "--resolved-config", str(resolved), "--run-dir", str(artifacts.run_dir)]
    _run_checked(args, artifacts.run_dir / "logs" / "rlvr.log", _repo_root() / "step1" / "python", 24 * 3600)
    return {"log": str(artifacts.run_dir / "logs" / "rlvr.log")}


def _rlvr_report(artifacts: RunArtifacts) -> dict:
    """Promote the stage report into the collected analysis payload."""
    path = artifacts.run_dir / "rlvr_report.json"
    if not path.is_file():
        raise RuntimeError("RLVR stage finished without rlvr_report.json")
    report = json.loads(path.read_text())
    shutil.copyfile(path, artifacts.run_dir / "analysis" / "result-report.json")
    return {
        "report": str(path),
        "contract": report["contract"],
        "rollout_episodes": report["budget_accounting"]["rollout_episodes"],
        "updates_with_any_reward_variance": report["training_signal"]["updates_with_any_reward_variance"],
    }


def _torchrun(config_path: Path, artifacts: RunArtifacts, preflight: bool, resume: bool = False) -> None:
    args = ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "step1_experiments.train", "--resolved-config", str(config_path), "--run-dir", str(artifacts.run_dir)]
    if preflight: args.append("--preflight-only")
    if resume and not preflight: args.append("--resume")
    _run_checked(args, artifacts.run_dir / "logs" / ("preflight.log" if preflight else "train.log"), _repo_root() / "step1" / "python", 24 * 3600)


def _last_trainer_checkpoint(checkpoint_dir: Path) -> str | None:
    """Report the actual standard Trainer checkpoint, never the removed .pt path."""
    try:
        from transformers.trainer_utils import get_last_checkpoint
        result = get_last_checkpoint(str(checkpoint_dir))
    except (ImportError, OSError):
        return None
    return result


def run(config_path: Path, output_root: Path, resume: str) -> None:
    repo = _repo_root(); config, config_hash = _resolved(config_path); sha, _ = git_info(repo)
    config["_meta"]["source_git_sha"] = sha
    run_id = f"{config['run']['name']}-{sha[:12]}-{config_hash[:12]}"; artifacts = RunArtifacts(output_root, run_id, sha, config_hash)
    success = False
    try:
        def phase(name: str, action):
            if artifacts.complete(name): return
            artifacts.begin(name)
            try:
                details = action()
            except BaseException as error:
                artifacts.phase_failed(name, error); raise
            artifacts.finish(name, details if isinstance(details, dict) else {})
        phase("capture_environment", lambda: capture(artifacts.run_dir / "environment" / "environment.json", repo, config, sys.argv))
        if config["run"]["mode"] == "rlvr":
            # The requested accelerator is not evidence.  Fail in seconds on a
            # mis-provisioned session rather than after the Rust build.
            phase("accelerator_check", _accelerator_check)
        phase("install_and_build", lambda: _build(repo, artifacts))
        phase("correctness_tests", lambda: _correctness(repo, artifacts))
        if config["run"]["mode"] == "rlvr":
            # Outcome-only training generates its own worlds interactively: no
            # teacher shards, no shard DataLoader, and therefore none of the
            # offline data phases below.
            phase("train", lambda: _rlvr(config, artifacts))
            phase("evaluate", lambda: _rlvr_report(artifacts))
            success = True
            return
        def cpu_gate():
            result = cpu_benchmark(config, artifacts.run_dir / "benchmarks")
            # Offline Rust shards are the actual training source. Retain this
            # online-generation ratio for engineering diagnosis, but do not
            # pre-judge trainer starvation before measuring the shard
            # DataLoader and the two-T4 training path.
            result["gate_policy"] = "diagnostic_only"
            return result
        phase("cpu_throughput", cpu_gate)
        phase("prepare_shards", lambda: _prepare(config, artifacts))
        phase("dataloader_throughput", lambda: dataloader_benchmark(BinaryShard(artifacts.run_dir / "datasets" / "train.bin"), config["world"]["context_length"], artifacts.run_dir / "benchmarks"))
        def gpu_preflight():
            assert_two_t4s(); resolved_path = _save_resolved(config, artifacts)
            _torchrun(resolved_path, artifacts, True)
            diagnostic_dir = artifacts.run_dir / "diagnostic-preflight"
            report_path = diagnostic_dir / "preflight_report.json"
            if not report_path.is_file():
                raise RuntimeError("two-T4 preflight finished without preflight_report.json")
            report = json.loads(report_path.read_text())
            return {
                "model_artifact": str(diagnostic_dir / "model"),
                "trainer_checkpoint": report["checkpoint"],
                "resumed_global_step": report["resumed_global_step"],
                "preflight_report": str(report_path),
            }
        phase("gpu_preflight", gpu_preflight)
        if config["run"]["mode"] == "preflight":
            pass
        elif config["run"]["mode"] == "dense":
            def dense_train():
                _torchrun(_save_resolved(config, artifacts), artifacts, False, resume == "auto")
                report_path = artifacts.run_dir / "production" / "training_report.json"
                if not report_path.is_file():
                    raise RuntimeError("dense Trainer finished without production/training_report.json")
                report = json.loads(report_path.read_text())
                return {
                    "training_report": str(report_path),
                    "global_step": report["global_step"],
                    "trainer_checkpoint": report["last_trainer_checkpoint"],
                    "model_artifact": report["model_artifact"],
                }
            phase("train", dense_train)
            phase("evaluate", lambda: __import__("step1_experiments.evaluate", fromlist=["evaluate"]).evaluate(_save_resolved(config, artifacts), artifacts.run_dir))
        # Outcome-only RLVR is its own stage, launched directly through
        # `step1_experiments.rlvr` from step1/kaggle/step1_rlvr_*.ipynb: it
        # generates no teacher shards and its budget is expressed in rollout
        # episodes rather than packed input tokens.  See RLVR-STAGE-PLAN.md.
        else: raise RuntimeError(f"mode {config['run']['mode']!r} is not run by this runner; see RLVR-STAGE-PLAN.md")
        success = True
    except BaseException as error:
        extra = {
            "nvidia_smi": command(["nvidia-smi"], 30),
            "last_production_trainer_checkpoint": _last_trainer_checkpoint(artifacts.run_dir / "production" / "checkpoints"),
            "last_diagnostic_trainer_checkpoint": _last_trainer_checkpoint(artifacts.run_dir / "diagnostic-preflight" / "checkpoints"),
        }
        artifacts.fail(error, extra); raise
    finally:
        artifacts.begin("package_results")
        artifacts.finish("package_results", {"success": success})
        artifacts.package(success)


def _save_resolved(config: dict, artifacts: RunArtifacts) -> Path:
    path = artifacts.run_dir / "resolved_config.json"; atomic_json(path, config); return path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True); parser.add_argument("--output-root", type=Path, required=True); parser.add_argument("--resume", choices=("auto", "never"), default="auto")
    args = parser.parse_args(); run(args.config, args.output_root, args.resume)


if __name__ == "__main__": main()
