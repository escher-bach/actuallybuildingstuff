"""Atomic, inspectable run artifacts.  No artifact is written into the clone."""
from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2) + "\n").encode())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# The analysis payload is the only artifact collected onto the user's device, so
# it carries evidence and never weights.  Directory names are matched exactly;
# the suffix list is a backstop for anything a future stage misplaces.
ANALYSIS_EXCLUDED_DIRS = frozenset({"checkpoints", "datasets", "model", "budget-0-model", ".git", "__pycache__"})
ANALYSIS_EXCLUDED_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".replay", ".tar.gz")
ANALYSIS_MAX_TOTAL_BYTES = 50 * 1024 * 1024
ANALYSIS_MAX_FILE_BYTES = 2 * 1024 * 1024


def analysis_candidates(run_dir: Path) -> list[Path]:
    """Compact evidence files, in stable order, with weights excluded."""
    selected = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir)
        if ANALYSIS_EXCLUDED_DIRS & set(relative.parts[:-1]):
            continue
        if path.name.endswith(ANALYSIS_EXCLUDED_SUFFIXES):
            continue
        selected.append(path)
    return selected


def write_analysis_archive(run_dir: Path, run_id: str, archive: Path) -> dict[str, Any]:
    """Bounded evidence payload; oversized logs are kept as truncated tails."""
    included, truncated = [], []
    temporary = archive.with_suffix(".tmp")
    with tarfile.open(temporary, "w:gz") as tar:
        for path in analysis_candidates(run_dir):
            relative = path.relative_to(run_dir)
            size = path.stat().st_size
            if size <= ANALYSIS_MAX_FILE_BYTES:
                tar.add(path, arcname=(Path(run_id) / relative).as_posix())
                included.append(str(relative))
                continue
            with path.open("rb") as handle:
                handle.seek(size - ANALYSIS_MAX_FILE_BYTES)
                tail = handle.read()
            info = tarfile.TarInfo((Path(run_id) / relative.with_name(relative.name + ".tail")).as_posix())
            info.size = len(tail)
            tar.addfile(info, io.BytesIO(tail))
            truncated.append({"file": str(relative), "original_bytes": size, "retained_bytes": len(tail)})
    os.replace(temporary, archive)
    total = archive.stat().st_size
    if total > ANALYSIS_MAX_TOTAL_BYTES:
        raise RuntimeError(f"analysis payload {total} bytes exceeds the {ANALYSIS_MAX_TOTAL_BYTES}-byte contract")
    return {"archive": archive.name, "bytes": total, "sha256": sha256_file(archive),
            "file_count": len(included), "truncated": truncated}


class RunArtifacts:
    def __init__(self, root: Path, run_id: str, git_sha: str, config_hash: str):
        self.root, self.run_id = root, run_id
        self.run_dir = root / run_id
        self.git_sha, self.config_hash = git_sha, config_hash
        self.phase_path = self.run_dir / "phase_status.json"
        self.status: dict[str, Any] = {"git_sha": git_sha, "config_hash": config_hash, "phases": {}}
        self.last_phase = "not_started"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        # Trainer state belongs only to ``production/`` or
        # ``diagnostic-preflight/``.  A top-level checkpoint directory would
        # make cross-phase resume selection ambiguous.
        for name in ("environment", "logs", "benchmarks", "datasets/manifests", "evaluation", "plots", "analysis"):
            (self.run_dir / name).mkdir(parents=True, exist_ok=True)
        if self.phase_path.exists():
            previous = json.loads(self.phase_path.read_text())
            if previous.get("git_sha") != git_sha or previous.get("config_hash") != config_hash:
                raise RuntimeError("existing run directory has mismatched Git SHA or resolved configuration")
            self.status = previous

    def complete(self, phase: str) -> bool:
        return self.status.get("phases", {}).get(phase, {}).get("state") == "complete"

    def begin(self, phase: str) -> None:
        self.last_phase = phase
        self.status.setdefault("phases", {})[phase] = {"state": "running", "started_at": utc_now()}
        atomic_json(self.phase_path, self.status)

    def finish(self, phase: str, details: dict[str, Any] | None = None) -> None:
        self.status["phases"][phase] = {"state": "complete", "finished_at": utc_now(), "details": details or {}}
        atomic_json(self.phase_path, self.status)

    def phase_failed(self, phase: str, error: BaseException) -> None:
        started = self.status.get("phases", {}).get(phase, {}).get("started_at")
        self.status["phases"][phase] = {
            "state": "failed", "started_at": started, "finished_at": utc_now(),
            "exception_class": type(error).__name__, "message": str(error),
        }
        atomic_json(self.phase_path, self.status)

    def fail(self, error: BaseException, extra: dict[str, Any] | None = None) -> None:
        completed = [name for name, value in self.status.get("phases", {}).items() if value.get("state") == "complete"]
        payload = {
            "timestamp": utc_now(), "failed_phase": self.last_phase,
            "last_completed_phase": completed[-1] if completed else "not_started",
            "exception_class": type(error).__name__, "message": str(error),
            "traceback": traceback.format_exc(), "git_sha": self.git_sha,
            "config_hash": self.config_hash, **(extra or {}),
        }
        atomic_json(self.run_dir / "failure.json", payload)
        atomic_bytes(self.run_dir / "FAILURE", b"\n")

    def provenance(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Compact identity the audit receipt is built from."""
        environment = self.run_dir / "environment" / "environment.json"
        accelerators = []
        if environment.is_file():
            try:
                accelerators = json.loads(environment.read_text()).get("torch", {}).get("devices", [])
            except (OSError, json.JSONDecodeError):
                accelerators = []
        data = {
            "run_id": self.run_id, "git_sha": self.git_sha, "config_hash": self.config_hash,
            "accelerator_inventory": accelerators, "phases": self.status.get("phases", {}),
            "captured_at": utc_now(), **(extra or {}),
        }
        atomic_json(self.run_dir / "analysis" / "provenance.json", data)
        return data

    def package(self, success: bool) -> dict[str, Any]:
        """Write the bounded analysis payload and the remote recovery payload."""
        atomic_bytes(self.run_dir / ("SUCCESS" if success else "FAILURE"), b"\n")
        identity = {"run_id": self.run_id, "success": success, "git_sha": self.git_sha,
                    "config_hash": self.config_hash, "updated_at": utc_now()}
        self.provenance({"success": success})
        # Written before archiving so the payload contains its own identity; the
        # payload checksums themselves can only live outside it.
        atomic_json(self.run_dir / "analysis" / "summary.json", identity)
        analysis = write_analysis_archive(self.run_dir, self.run_id, self.root / f"{self.run_id}-analysis.tar.gz")
        atomic_bytes(self.root / f"{self.run_id}-analysis.sha256", f"{analysis['sha256']}  {analysis['archive']}\n".encode())
        # The recovery payload stays on Kaggle; it is never collected locally.
        recovery = self.root / f"{self.run_id}.tar.gz"
        temporary = recovery.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            tar.add(self.run_dir, arcname=self.run_id)
        os.replace(temporary, recovery)
        recovery_sha = sha256_file(recovery)
        atomic_bytes(self.root / f"{self.run_id}.sha256", f"{recovery_sha}  {recovery.name}\n".encode())
        summary = {
            **identity,
            "analysis": analysis,
            "recovery": {"archive": recovery.name, "bytes": recovery.stat().st_size, "sha256": recovery_sha},
        }
        atomic_json(self.root / "latest-summary.json", summary)
        return summary
