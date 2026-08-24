"""Single non-interactive Kaggle runner for the STEP 2 vertical slice."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import tomllib
import traceback
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def run_logged(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    timeout: int,
) -> None:
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
        check=False,
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, flush=True)
    if completed.returncode:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode} after {time.time() - started:.1f}s: {command}"
        )


def phase_update(path: Path, phases: dict[str, Any], name: str, status: str, **extra: Any) -> None:
    entry = phases.setdefault(name, {})
    entry.update({"status": status, "timestamp": time.time(), **extra})
    path.write_text(json.dumps(phases, indent=2, sort_keys=True), encoding="utf-8")


def safe_environment() -> dict[str, str]:
    allowed = {
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_URL_BASE",
        "KAGGLE_DATA_PROXY_TOKEN",
        "CUDA_VISIBLE_DEVICES",
        "NVIDIA_VISIBLE_DEVICES",
        "PYTHONHASHSEED",
    }
    result = {}
    for name in allowed:
        if name not in os.environ:
            continue
        result[name] = "<redacted>" if "TOKEN" in name else os.environ[name]
    return result


def capture_environment(repo: Path, config_text: str, command_line: list[str]) -> dict[str, Any]:
    def checked(command: list[str]) -> str:
        return subprocess.check_output(command, cwd=repo, text=True, stderr=subprocess.STDOUT).strip()

    return {
        "git_sha": checked(["git", "rev-parse", "HEAD"]),
        "git_status": checked(["git", "status", "--porcelain"]),
        "config_sha256": sha256_text(config_text),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "command_line": command_line,
        "safe_environment": safe_environment(),
        "nvidia_smi": subprocess.run(
            ["nvidia-smi"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
        ).stdout,
        "disk_free_bytes": shutil.disk_usage("/kaggle/working").free
        if Path("/kaggle/working").exists()
        else shutil.disk_usage(repo).free,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config_text = config_path.read_text(encoding="utf-8")
    config = tomllib.loads(config_text)
    repo = config_path.parents[3]
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    logs = output_root / "logs"
    phase_path = output_root / "phase_status.json"
    phases: dict[str, Any] = {}
    status = "failed"
    error: str | None = None
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "WANDB_MODE": "disabled",
            "PYTHONPATH": str(repo / "step2" / "python")
            + os.pathsep
            + env.get("PYTHONPATH", ""),
        }
    )

    try:
        phase_update(phase_path, phases, "capture_environment", "running")
        environment = capture_environment(repo, config_text, sys.argv)
        (output_root / "environment.json").write_text(
            json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
        )
        if environment["git_status"]:
            raise RuntimeError(f"Kaggle source checkout is not clean: {environment['git_status']}")
        phase_update(phase_path, phases, "capture_environment", "complete")

        phase_update(phase_path, phases, "install_and_build", "running")
        run_logged(
            [sys.executable, "-m", "pip", "install", "-r", str(repo / "step2" / "requirements-kaggle.txt")],
            cwd=repo,
            env=env,
            log_path=logs / "pip-install.log",
            timeout=900,
        )
        toolchain = tomllib.loads((repo / "step2" / "rust-toolchain.toml").read_text(encoding="utf-8"))[
            "toolchain"
        ]["channel"]
        cargo_home = Path("/tmp/step2-cargo")
        rustup_home = Path("/tmp/step2-rustup")
        env.update(
            {
                "CARGO_HOME": str(cargo_home),
                "RUSTUP_HOME": str(rustup_home),
                "RUSTUP_TOOLCHAIN": str(toolchain),
                "PATH": str(cargo_home / "bin") + os.pathsep + env["PATH"],
            }
        )
        rustup_installer = Path("/tmp/rustup-init.sh")
        if not (cargo_home / "bin" / "cargo").is_file():
            run_logged(
                [
                    "curl",
                    "--proto",
                    "=https",
                    "--tlsv1.2",
                    "--fail",
                    "--silent",
                    "--show-error",
                    "https://sh.rustup.rs",
                    "--output",
                    str(rustup_installer),
                ],
                cwd=repo,
                env=env,
                log_path=logs / "rustup-download.log",
                timeout=300,
            )
            run_logged(
                [
                    "sh",
                    str(rustup_installer),
                    "-y",
                    "--no-modify-path",
                    "--profile",
                    "minimal",
                    "--default-toolchain",
                    str(toolchain),
                ],
                cwd=repo,
                env=env,
                log_path=logs / "rustup-install.log",
                timeout=900,
            )
        run_logged(
            [str(cargo_home / "bin" / "rustc"), "--version", "--verbose"],
            cwd=repo,
            env=env,
            log_path=logs / "rust-toolchain.log",
            timeout=60,
        )
        wheel_dir = Path("/tmp/step2-wheels") if Path("/tmp").exists() else repo / "step2" / "dist"
        wheel_dir.mkdir(parents=True, exist_ok=True)
        run_logged(
            [
                sys.executable,
                "-m",
                "maturin",
                "build",
                "--release",
                "--locked",
                "--manifest-path",
                str(repo / "step2" / "crates" / "world-py" / "Cargo.toml"),
                "--out",
                str(wheel_dir),
            ],
            cwd=repo,
            env=env,
            log_path=logs / "maturin-build.log",
            timeout=1200,
        )
        wheels = sorted(wheel_dir.glob("step2_world_py-*.whl"))
        if len(wheels) != 1:
            raise RuntimeError(f"expected one STEP 2 world wheel, found {wheels}")
        run_logged(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", str(wheels[0])],
            cwd=repo,
            env=env,
            log_path=logs / "world-wheel-install.log",
            timeout=300,
        )
        phase_update(
            phase_path,
            phases,
            "install_and_build",
            "complete",
            wheel_sha256=sha256_file(wheels[0]),
        )

        phase_update(phase_path, phases, "correctness_tests", "running")
        run_logged(
            ["cargo", "test", "--manifest-path", str(repo / "step2" / "Cargo.toml"), "--workspace", "--locked"],
            cwd=repo,
            env=env,
            log_path=logs / "rust-tests.log",
            timeout=900,
        )
        run_logged(
            [
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                str(repo / "step2" / "python" / "tests"),
                "-v",
            ],
            cwd=repo,
            env=env,
            log_path=logs / "python-tests.log",
            timeout=1200,
        )
        phase_update(phase_path, phases, "correctness_tests", "complete")

        phase_update(phase_path, phases, "world_validation", "running")
        validation_code = (
            "import json,tomllib,step2_world_py; from pathlib import Path; "
            f"c=tomllib.loads(Path(r'{config_path}').read_text()); w=c['world']; "
            "k={x:w[x] for x in ('d_min','d_max','gain_min','gain_max','action_limit','calibration_pulse','max_control_steps')}; "
            "r=step2_world_py.validate_generated_worlds(seed=int(w['validation_seed']),start_index=0,"
            "count=int(c['preflight']['validation_instances']),**k); "
            f"Path(r'{output_root / 'world-validation.json'}').write_text(json.dumps(r,indent=2,sort_keys=True))"
        )
        run_logged(
            [sys.executable, "-c", validation_code],
            cwd=repo,
            env=env,
            log_path=logs / "world-validation.log",
            timeout=300,
        )
        phase_update(phase_path, phases, "world_validation", "complete")

        phase_update(phase_path, phases, "cpu_benchmark", "running")
        run_logged(
            [
                sys.executable,
                "-m",
                "step2_experiments.benchmarks",
                "--config",
                str(config_path),
                "--output",
                str(output_root / "cpu-benchmark.json"),
            ],
            cwd=repo,
            env=env,
            log_path=logs / "cpu-benchmark.log",
            timeout=900,
        )
        phase_update(phase_path, phases, "cpu_benchmark", "complete")

        phase_update(phase_path, phases, "trivial_policy_baselines", "running")
        run_logged(
            [
                sys.executable,
                "-m",
                "step2_experiments.baselines",
                "--config",
                str(config_path),
                "--output",
                str(output_root / "trivial-policy-baselines.json"),
            ],
            cwd=repo,
            env=env,
            log_path=logs / "trivial-policy-baselines.log",
            timeout=300,
        )
        phase_update(phase_path, phases, "trivial_policy_baselines", "complete")

        phase_update(phase_path, phases, "gpu_vertical_slice", "running")
        gpu_command = [
            sys.executable,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes",
            "2",
            "--num_machines",
            "1",
            "--mixed_precision",
            str(config["run"]["mixed_precision"]),
            "--dynamo_backend",
            "no",
            "-m",
            "step2_experiments.train",
            "--config",
            str(config_path),
            "--output-root",
            str(output_root),
        ]
        run_logged(
            gpu_command,
            cwd=repo,
            env=env,
            log_path=logs / "gpu-vertical-slice.log",
            timeout=10_800,
        )
        training_result = json.loads((output_root / "training-result.json").read_text(encoding="utf-8"))
        if not training_result.get("architecture_gate_passed"):
            raise RuntimeError("GPU architecture integration gate did not pass")
        phase_update(phase_path, phases, "gpu_vertical_slice", "complete")
        status = "complete"
    except Exception as exc:  # package evidence before propagating the failure
        error = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
        print(error, file=sys.stderr, flush=True)
    finally:
        artifacts = []
        for path in sorted(output_root.rglob("*")):
            if path.is_file() and "checkpoints" not in path.parts:
                artifacts.append(
                    {
                        "path": path.relative_to(output_root).as_posix(),
                        "size": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
        manifest = {
            "status": status,
            "error": error,
            "config_sha256": sha256_text(config_text),
            "git_sha": subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo,
                text=True,
                stdout=subprocess.PIPE,
                check=False,
            ).stdout.strip(),
            "artifacts": artifacts,
        }
        (output_root / "audit-manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        summary = {
            "status": status,
            "error": error,
            "purpose": config["run"]["purpose"],
            "checkpoint_label": config["run"]["checkpoint_label"],
            "phase_status": phases,
            "git_sha": manifest["git_sha"],
            "config_sha256": manifest["config_sha256"],
        }
        if (output_root / "architecture-gate-progress.json").exists():
            summary["architecture_gate_progress"] = json.loads(
                (output_root / "architecture-gate-progress.json").read_text(encoding="utf-8")
            )
        if (output_root / "training-result.json").exists():
            result = json.loads((output_root / "training-result.json").read_text(encoding="utf-8"))
            summary.update(
                {
                    "architecture_gate_passed": result.get("architecture_gate_passed"),
                    "resume_smoke": result.get("resume_smoke"),
                    "validation": result.get("validation"),
                    "closed_loop": result.get("closed_loop"),
                    "oracle_closed_loop": result.get("oracle_closed_loop"),
                    "untrained_baseline": result.get("untrained_baseline"),
                    "paired_learning_delta": result.get("paired_learning_delta"),
                    "model_sha256": result.get("model_sha256"),
                    "recovery_artifact": result.get("recovery_artifact"),
                    "root_seed": result.get("root_seed"),
                    "accelerator_inventory": {
                        "world_size": result.get("world_size"),
                        "device_names": result.get("device_names"),
                        "torch_version": result.get("torch_version"),
                    },
                }
            )
        if (output_root / "trivial-policy-baselines.json").exists():
            summary["trivial_policy_baselines"] = json.loads(
                (output_root / "trivial-policy-baselines.json").read_text(encoding="utf-8")
            )
        (output_root / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    if status != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
