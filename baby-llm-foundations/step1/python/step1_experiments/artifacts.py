"""Atomic, inspectable run artifacts.  No artifact is written into the clone."""
from __future__ import annotations

import hashlib
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
        for name in ("environment", "logs", "benchmarks", "datasets/manifests", "evaluation", "plots"):
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

    def package(self, success: bool) -> Path:
        atomic_bytes(self.run_dir / ("SUCCESS" if success else "FAILURE"), b"\n")
        archive = self.root / f"{self.run_id}.tar.gz"
        temporary = archive.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            tar.add(self.run_dir, arcname=self.run_id)
        os.replace(temporary, archive)
        atomic_bytes(self.root / f"{self.run_id}.sha256", f"{sha256_file(archive)}  {archive.name}\n".encode())
        atomic_json(self.root / "latest-summary.json", {
            "run_id": self.run_id, "success": success, "git_sha": self.git_sha,
            "config_hash": self.config_hash, "archive": archive.name,
            "archive_sha256": sha256_file(archive), "updated_at": utc_now(),
        })
        return archive
