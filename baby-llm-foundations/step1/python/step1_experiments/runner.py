"""Single public Step 1 command: phase machine, diagnostics, and packaging."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

from .artifacts import RunArtifacts, atomic_json
from .benchmarks import cpu_benchmark, dataloader_benchmark
from .data import BinaryShard, TOKENIZER_HASH, VOCAB_SIZE, DistributedSequenceSampler, Sequence, generate_rust_shard
from .environment import assert_two_t4s, capture, command, git_info


PHASES = ("capture_environment", "install_and_build", "correctness_tests", "cpu_throughput", "prepare_shards", "dataloader_throughput", "gpu_preflight", "instrument_check", "train", "evaluate")


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
        result = subprocess.run(args, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout, check=False, env={**os.environ, "PYTHONUNBUFFERED": "1", "WANDB_MODE": "disabled", "TOKENIZERS_PARALLELISM": "false"})
    if result.returncode: raise RuntimeError(f"subprocess failed ({result.returncode}): {' '.join(args)}")


def _build(repo: Path, artifacts: RunArtifacts) -> None:
    log = artifacts.run_dir / "logs" / "build.log"; cargo = shutil.which("cargo")
    if not cargo:
        # Kaggle normally has Cargo. This is the bounded, non-interactive fallback.
        installer = artifacts.run_dir / "rustup-init"
        urllib.request.urlretrieve("https://static.rust-lang.org/rustup/dist/x86_64-unknown-linux-gnu/rustup-init", installer)
        installer.chmod(0o755)
        _run_checked([str(installer), "-y", "--profile", "minimal", "--default-toolchain", "1.85.0"], log, repo, 900)
        os.environ["PATH"] = f"{Path.home() / '.cargo' / 'bin'}{os.pathsep}{os.environ['PATH']}"
        cargo = shutil.which("cargo")
        if not cargo: raise RuntimeError("pinned Rust 1.85.0 installation did not provide Cargo")
    _run_checked([sys.executable, "-m", "pip", "install", "-r", str(repo / "requirements-kaggle.txt")], log, repo, 900)
    _run_checked([sys.executable, "-m", "maturin", "build", "--release", "--manifest-path", str(repo / "step1" / "crates" / "world-py" / "Cargo.toml")], log, repo / "step1", 1800)
    wheels = sorted((repo / "step1" / "target" / "wheels").glob("world_py-*.whl"))
    if not wheels: raise RuntimeError("maturin did not produce a world_py wheel")
    _run_checked([sys.executable, "-m", "pip", "install", "--force-reinstall", str(wheels[-1])], log, repo, 900)
    _run_checked([sys.executable, str(repo / "step1" / "python" / "smoke.py")], log, repo, 120)


def _correctness(repo: Path, artifacts: RunArtifacts) -> None:
    log = artifacts.run_dir / "logs" / "tests.log"
    _run_checked(["cargo", "test", "--workspace", "--locked"], log, repo / "step1", 1800)
    # Python contracts: byte table, mask shift, and distributed partitions.
    script = """
import torch
from step1_experiments.data import *
from step1_experiments.model import masked_next_token_loss
from world_py import parse_action, render_action
assert all(decode_bytes([i]).encode() == bytes([i]) for i in range(128))
for rendering in ('a', 'b'):
    text=render_action(0, 5, 6, rendering); assert parse_action(text, 5, 6, rendering)==0
seq=[Sequence([1,2,3],[0,1,0],[0,1,0]) for _ in range(11)]
d=SequenceDataset(seq); a=set(DistributedSequenceSampler(d,7,0,2)); b=set(DistributedSequenceSampler(d,7,1,2)); assert not a & b and len(a|b)==10
logits=torch.zeros(1,3,VOCAB_SIZE,requires_grad=True); total,n=masked_next_token_loss(logits,torch.tensor([[1,2,3]]),torch.tensor([[0,1,0]])); assert n.item()==1; total.backward()
"""
    _run_checked([sys.executable, "-c", script], log, repo / "step1" / "python", 120)


def _prepare(config: dict, artifacts: RunArtifacts) -> dict:
    root = artifacts.run_dir / "datasets"; world = config["world"]; seed = config["run"]["root_seed"]
    partitions = [("train", world, seed, world["train_episodes"], world["rendering"]), ("validation", world, seed + 1_000_000, world["validation_episodes"], world["rendering"])]
    structural = {**world, "n_hyp": world["n_hyp"] + 1}; transfer = {**world, "rendering": "b"}
    partitions.extend([("structural", structural, seed + 2_000_000, world["structural_episodes"], world["rendering"]), ("transfer", transfer, seed + 3_000_000, world["transfer_episodes"], "b")])
    result = {}
    for name, params, part_seed, count, rendering in partitions:
        binary, manifest, replay = generate_rust_shard(params, part_seed, count, world["context_length"], rendering, root, name)
        result[name] = {"path": str(binary), "manifest": str(manifest), "replay": str(replay), "content_hash": hashlib.sha256(binary.read_bytes()).hexdigest()}
    if result["train"]["content_hash"] in {result["validation"]["content_hash"], result["structural"]["content_hash"], result["transfer"]["content_hash"]}: raise RuntimeError("dataset partition collision")
    return result


def _torchrun(config_path: Path, artifacts: RunArtifacts, preflight: bool, resume: bool = False) -> None:
    args = ["torchrun", "--standalone", "--nproc_per_node=2", "-m", "step1_experiments.train", "--resolved-config", str(config_path), "--run-dir", str(artifacts.run_dir)]
    if preflight: args.append("--preflight-only")
    if resume and not preflight: args.append("--resume")
    _run_checked(args, artifacts.run_dir / "logs" / ("preflight.log" if preflight else "train.log"), _repo_root() / "step1" / "python", 24 * 3600)


def _instrument_torchrun(config_path: Path, artifacts: RunArtifacts) -> None:
    _run_checked(["torchrun", "--standalone", "--nproc_per_node=2", "-m", "step1_experiments.instrument", "--resolved-config", str(config_path), "--run-dir", str(artifacts.run_dir)], artifacts.run_dir / "logs" / "instrument.log", _repo_root() / "step1" / "python", 24 * 3600)


def run(config_path: Path, output_root: Path, resume: str) -> None:
    repo = _repo_root(); config, config_hash = _resolved(config_path); sha, _ = git_info(repo)
    run_id = f"{config['run']['name']}-{sha[:12]}-{config_hash[:12]}"; artifacts = RunArtifacts(output_root, run_id, sha, config_hash)
    success = False
    try:
        def phase(name: str, action):
            if artifacts.complete(name): return
            artifacts.begin(name); details = action(); artifacts.finish(name, details if isinstance(details, dict) else {})
        phase("capture_environment", lambda: capture(artifacts.run_dir / "environment" / "environment.json", repo, config, sys.argv))
        phase("install_and_build", lambda: _build(repo, artifacts))
        phase("correctness_tests", lambda: _correctness(repo, artifacts))
        def cpu_gate():
            result = cpu_benchmark(config, artifacts.run_dir / "benchmarks")
            if not result["passes_80_percent_gate"]:
                raise RuntimeError(f"world CPU pipeline reached only {result['world_to_raw_ratio']:.1%} of matched raw-text throughput")
            return result
        phase("cpu_throughput", cpu_gate)
        phase("prepare_shards", lambda: _prepare(config, artifacts))
        phase("dataloader_throughput", lambda: dataloader_benchmark(BinaryShard(artifacts.run_dir / "datasets" / "train.bin"), config["world"]["context_length"], artifacts.run_dir / "benchmarks"))
        phase("gpu_preflight", lambda: (assert_two_t4s(), _torchrun(_save_resolved(config, artifacts), artifacts, True))[1])
        if config["run"]["mode"] == "preflight": phase("instrument_check", lambda: _instrument_torchrun(_save_resolved(config, artifacts), artifacts))
        elif config["run"]["mode"] == "dense":
            phase("train", lambda: _torchrun(_save_resolved(config, artifacts), artifacts, False, resume == "auto"))
            phase("evaluate", lambda: __import__("step1_experiments.evaluate", fromlist=["evaluate"]).evaluate(_save_resolved(config, artifacts), artifacts.run_dir))
        else: raise RuntimeError("RLVR budget is intentionally unset until dense seed-0 measurements freeze it")
        success = True
    except BaseException as error:
        extra = {"nvidia_smi": command(["nvidia-smi"], 30), "last_checkpoint": str(artifacts.run_dir / "checkpoints" / "latest.pt")}
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
