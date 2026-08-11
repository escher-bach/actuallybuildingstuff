"""Environment capture and bounded subprocess execution."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .artifacts import atomic_json

SECRET_WORDS = ("token", "secret", "password", "cookie", "key")


def command(args: list[str], timeout: int = 120, cwd: Path | None = None) -> dict[str, Any]:
    try:
        done = subprocess.run(args, cwd=cwd, text=True, capture_output=True, timeout=timeout, check=False)
        return {"command": args, "returncode": done.returncode, "stdout": done.stdout, "stderr": done.stderr}
    except Exception as exc:
        return {"command": args, "exception": repr(exc)}


def git_info(repo: Path) -> tuple[str, str]:
    sha = command(["git", "rev-parse", "HEAD"], cwd=repo)["stdout"].strip()
    status = command(["git", "status", "--porcelain"], cwd=repo)["stdout"]
    if len(sha) != 40:
        raise RuntimeError(f"could not resolve 40-character Git SHA in {repo}")
    return sha, status


def capture(path: Path, repo: Path, resolved: dict[str, Any], command_line: list[str]) -> dict[str, Any]:
    try:
        import torch
        torch_info: dict[str, Any] = {"version": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "devices": []}
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            torch_info["devices"].append({"index": index, "name": props.name, "capability": [props.major, props.minor], "vram_bytes": props.total_memory})
    except Exception as exc:
        torch_info = {"error": repr(exc)}
    disk = shutil.disk_usage("/kaggle/working" if Path("/kaggle/working").exists() else str(path.parent))
    safe_env = {k: v for k, v in os.environ.items() if k.startswith(("KAGGLE_", "CUDA_", "NCCL_")) and not any(word in k.lower() for word in SECRET_WORDS)}
    sha, status = git_info(repo)
    data = {"git_sha": sha, "git_status_porcelain": status, "resolved_config": resolved, "command_line": command_line,
            "python": sys.version, "platform": platform.platform(), "cpu_count_logical": os.cpu_count(), "disk": {"total": disk.total, "free": disk.free},
            "torch": torch_info, "nvidia_smi": command(["nvidia-smi"], timeout=30), "cargo": command(["cargo", "--version"], timeout=30),
            "compiler": command(["cc", "--version"], timeout=30), "environment": safe_env}
    atomic_json(path, data)
    return data


def assert_two_t4s() -> None:
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError(f"Kaggle T4 x2 required; found {torch.cuda.device_count()} CUDA devices")
    names = [torch.cuda.get_device_name(i) for i in range(2)]
    if not all("T4" in name.upper() for name in names):
        raise RuntimeError(f"Kaggle T4 x2 required; found {names}")
