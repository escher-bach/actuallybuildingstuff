"""Sole operator entry point for Step 1 Kaggle runs.

This is a thin adapter around the maintained Kaggle CLI. It shells out to
documented `kaggle kernels` commands and owns only what the project must own:
experiment resolution, commit reachability, ephemeral staging, exact-version
capture, filtered collection, checksum verification, and receipt generation.

It never pushes Git commits and never launches GPU work implicitly: a
submission happens only when the operator names an experiment on the command
line. See baby-llm-foundations/EXPERIMENT-EXECUTION-PLAN.md sections 1.4 and 14.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import tomllib
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT = REPO_ROOT / "baby-llm-foundations"
REGISTRY_PATH = PROJECT / "step1" / "kaggle" / "experiments.toml"
AUDIT_ROOT = PROJECT / "step1" / "audit" / "runs"
RECEIPT_SCHEMA_VERSION = 1
COMMIT_MARKER = "__FINAL_COMMIT_SHA__"
CONFIG_MARKER = "__CONFIG_REL__"
TERMINAL_STATUSES = ("COMPLETE", "ERROR", "CANCEL")
# Collected artifacts are compact evidence only; checkpoints stay on Kaggle.
ANALYSIS_PATTERN = r"(latest-summary\.json|.*-analysis\.tar\.gz|.*-analysis\.sha256)"
LOG_PATTERN = r".*\.log"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Experiment:
    name: str
    title: str
    slug_prefix: str
    config: str
    owner: str
    accelerator: str
    output_root: str
    launcher_template: str
    git_remote: str
    kernel_sources: tuple[str, ...]
    upstream_identity: dict

    def slug(self, commit: str) -> str:
        return f"{self.slug_prefix}-{commit[:7]}"

    def kernel(self, commit: str) -> str:
        return f"{self.owner}/{self.slug(commit)}"


def load_registry(path: Path = REGISTRY_PATH) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def resolve_experiment(name: str, registry: dict | None = None) -> Experiment:
    registry = load_registry() if registry is None else registry
    entries = registry.get("experiments", {})
    if name not in entries:
        raise SystemExit(f"unknown experiment {name!r}; declared: {', '.join(sorted(entries))}")
    entry = entries[name]
    return Experiment(
        name=name,
        title=entry["title"],
        slug_prefix=entry["slug_prefix"],
        config=entry["config"],
        owner=registry["owner"],
        accelerator=entry.get("accelerator", registry["accelerator"]),
        output_root=entry.get("output_root", registry["output_root"]),
        launcher_template=entry.get("launcher_template", registry["launcher_template"]),
        git_remote=registry["git_remote"],
        kernel_sources=tuple(entry.get("kernel_sources", ())),
        upstream_identity=entry.get("upstream_identity", {}),
    )


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    result = subprocess.run(["git", *args], cwd=cwd, text=True, capture_output=True, check=False)
    if result.returncode:
        raise SystemExit(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def kaggle(*args: str, check: bool = True, timeout: int = 900) -> str:
    executable = shutil.which("kaggle")
    if not executable:
        raise SystemExit("the official Kaggle CLI is not on PATH")
    result = subprocess.run([executable, *args], text=True, capture_output=True, timeout=timeout, check=False)
    output = (result.stdout or "") + (result.stderr or "")
    if check and result.returncode:
        raise SystemExit(f"kaggle {' '.join(args)} failed ({result.returncode}):\n{output.strip()}")
    return output


def resolve_commit(requested: str | None) -> str:
    commit = git("rev-parse", requested or "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise SystemExit(f"could not resolve a full 40-character commit from {requested or 'HEAD'!r}")
    return commit


def assert_commit_on_remote(commit: str, remote: str) -> str:
    """A retained run must be reproducible from the remote, not from this disk."""
    listing = git("ls-remote", "--heads", "--tags", remote)
    heads = [line.split()[0] for line in listing.splitlines() if line.strip()]
    for head in heads:
        probe = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, head],
            cwd=REPO_ROOT, capture_output=True, check=False,
        )
        if probe.returncode == 0:
            return head
    raise SystemExit(
        f"commit {commit} is not reachable from any ref on {remote}. "
        "Push it yourself; this tool never pushes Git commits."
    )


def assert_clean_worktree(paths: list[str]) -> None:
    dirty = [line for line in git("status", "--porcelain", "--", *paths).splitlines() if line.strip()]
    if dirty:
        raise SystemExit("refusing to submit with uncommitted changes in:\n  " + "\n  ".join(dirty))


def render_launcher(commit: str, experiment: Experiment) -> str:
    """Render the launcher from the pinned commit, never from the worktree."""
    relative = (Path("baby-llm-foundations") / experiment.launcher_template).as_posix()
    template = git("show", f"{commit}:{relative}")
    for marker, count in ((COMMIT_MARKER, 1), (CONFIG_MARKER, 1)):
        if template.count(marker) != count:
            raise SystemExit(f"launcher template must contain exactly {count} {marker} placeholder")
    document = template.replace(COMMIT_MARKER, commit).replace(CONFIG_MARKER, experiment.config)
    json.loads(document)  # fail here rather than inside a Kaggle session
    return document


def stage(commit: str, experiment: Experiment, directory: Path) -> Path:
    slug = experiment.slug(commit)
    notebook = directory / f"{slug}.ipynb"
    notebook.write_text(render_launcher(commit, experiment), encoding="utf-8")
    metadata = {
        "id": experiment.kernel(commit),
        "title": experiment.title[:50],
        "code_file": notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": experiment.accelerator,
    }
    if experiment.kernel_sources:
        metadata["kernel_sources"] = list(experiment.kernel_sources)
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return notebook


def kernel_status(kernel: str) -> str:
    output = kaggle("kernels", "status", kernel, check=False)
    match = re.search(r'status "(?:KernelWorkerStatus\.)?([A-Z_]+)"', output)
    return match.group(1) if match else "UNKNOWN"


def assert_not_running(kernel: str) -> None:
    status = kernel_status(kernel)
    if status not in ("UNKNOWN", *TERMINAL_STATUSES):
        raise SystemExit(f"{kernel} is already {status}; refusing a concurrent submission")


def submit(commit: str, experiment: Experiment) -> dict:
    kernel = experiment.kernel(commit)
    assert_not_running(kernel)
    with tempfile.TemporaryDirectory(prefix="step1-kaggle-") as directory:
        staging = Path(directory)
        stage(commit, experiment, staging)
        output = kaggle("kernels", "push", "-p", str(staging), "--accelerator", experiment.accelerator)
    version = re.search(r"version\s+(\d+)", output)
    if not version:
        raise SystemExit(f"could not read the pushed version from Kaggle output:\n{output.strip()}")
    reference = {
        "experiment": experiment.name,
        "kernel": kernel,
        "version": int(version.group(1)),
        "exact_version": f"{kernel}/{version.group(1)}",
        "url": f"https://www.kaggle.com/code/{kernel}",
        "git_sha": commit,
        "config": experiment.config,
        "accelerator_requested": experiment.accelerator,
        "kernel_sources": list(experiment.kernel_sources),
        "submitted_at": utc_now(),
    }
    print(json.dumps(reference, indent=2))
    return reference


def wait(kernel: str, poll_seconds: int = 60, timeout_seconds: int = 12 * 3600) -> str:
    deadline = time.monotonic() + timeout_seconds
    status = kernel_status(kernel)
    while status not in TERMINAL_STATUSES:
        if time.monotonic() > deadline:
            raise SystemExit(f"{kernel} did not reach a terminal status within {timeout_seconds}s (last: {status})")
        time.sleep(poll_seconds)
        status = kernel_status(kernel)
        print(f"{utc_now()} {kernel} {status}", flush=True)
    return status


def _verify_sidecar(directory: Path) -> dict:
    """Check the analysis payload against the checksum the run itself wrote."""
    sidecars = sorted(directory.glob("*-analysis.sha256"))
    archives = sorted(directory.glob("*-analysis.tar.gz"))
    if len(sidecars) != 1 or len(archives) != 1:
        raise SystemExit(f"expected exactly one analysis payload and sidecar in {directory}")
    expected = sidecars[0].read_text().split()[0]
    import hashlib

    digest = hashlib.sha256()
    with archives[0].open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(f"analysis payload checksum mismatch: {actual} != {expected}")
    return {"archive": archives[0].name, "sha256": actual, "bytes": archives[0].stat().st_size, "path": archives[0]}


def collect(reference: dict, keep_payload: Path | None = None) -> dict:
    """Fetch only compact evidence, verify it, and write a tracked receipt."""
    kernel, exact = reference["kernel"], reference["exact_version"]
    with tempfile.TemporaryDirectory(prefix="step1-collect-") as directory:
        download = Path(directory)
        kaggle("kernels", "output", exact, "-p", str(download), "--file-pattern", ANALYSIS_PATTERN, "-o", "-q")
        summary_path = download / "latest-summary.json"
        if not summary_path.is_file():
            raise SystemExit(f"{exact} produced no latest-summary.json; use `logs` to inspect the failure")
        summary = json.loads(summary_path.read_text())
        verified = _verify_sidecar(download)
        if verified["sha256"] != summary.get("analysis", {}).get("sha256"):
            raise SystemExit("analysis checksum does not match the run's own summary")
        run_id = summary["run_id"]
        audit = AUDIT_ROOT / run_id
        audit.mkdir(parents=True, exist_ok=True)
        with tarfile.open(verified["path"]) as tar:
            tar.extractall(download / "unpacked", filter="data")
        unpacked = download / "unpacked" / run_id
        for name in ("analysis/summary.json", "analysis/provenance.json", "analysis/result-report.json"):
            source = unpacked / name
            if source.is_file():
                shutil.copyfile(source, audit / Path(name).name)
        if keep_payload is not None:
            keep_payload.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(verified["path"], keep_payload / verified["archive"])
        provenance = {}
        provenance_path = audit / "provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
        receipt = {
            "schema_version": RECEIPT_SCHEMA_VERSION,
            "run_id": run_id,
            "experiment": reference["experiment"],
            "kaggle": {
                "kernel": kernel, "exact_version": exact, "url": reference["url"],
                "terminal_status": reference.get("terminal_status"),
                "completed_at": reference.get("completed_at"),
                "accelerator_requested": reference["accelerator_requested"],
            },
            "git": {"remote_url": git("remote", "get-url", "origin"), "sha": reference["git_sha"]},
            "config": {"path": reference["config"], "sha256": summary.get("config_hash")},
            "root_seed": provenance.get("root_seed"),
            "accelerator_inventory": provenance.get("accelerator_inventory", []),
            "upstream": [
                {"kernel": source, "expected_identity": reference.get("upstream_identity", {}).get(source, {})}
                for source in reference.get("kernel_sources", [])
            ],
            "analysis_artifacts": [{"name": verified["archive"], "bytes": verified["bytes"], "sha256": verified["sha256"]}],
            "recovery_artifact": {**summary.get("recovery", {}), "location": f"kaggle output of {exact}"},
            "result_report": "result-report.json" if (audit / "result-report.json").is_file() else None,
            "run_success": summary.get("success"),
            "collected_at": utc_now(),
        }
        (audit / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"receipt": str(audit / "receipt.json"), "run_id": run_id, "success": summary.get("success")}, indent=2))
    return receipt


def show_logs(kernel_or_version: str, lines: int) -> None:
    with tempfile.TemporaryDirectory(prefix="step1-logs-") as directory:
        download = Path(directory)
        kaggle("kernels", "output", kernel_or_version, "-p", str(download), "--file-pattern", LOG_PATTERN, "-o", "-q")
        logs = sorted(download.rglob("*.log"))
        if not logs:
            print(f"no execution log available for {kernel_or_version}")
            return
        for log in logs:
            print(f"===== {log.name} (last {lines} lines) =====")
            print("\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:]))


def _reference_path(experiment: Experiment, commit: str) -> Path:
    return AUDIT_ROOT / "submitted" / f"{experiment.slug(commit)}.json"


def _save_reference(reference: dict, experiment: Experiment, commit: str) -> Path:
    path = _reference_path(experiment, commit)
    path.parent.mkdir(parents=True, exist_ok=True)
    reference = {**reference, "upstream_identity": experiment.upstream_identity}
    path.write_text(json.dumps(reference, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _validate(experiment: Experiment, commit: str) -> None:
    assert_clean_worktree([
        str(Path("baby-llm-foundations") / experiment.launcher_template),
        str(Path("baby-llm-foundations") / experiment.config),
        "baby-llm-foundations/step1/python",
        "baby-llm-foundations/step1/kaggle/experiments.toml",
    ])
    assert_commit_on_remote(commit, experiment.git_remote)
    if not (PROJECT / experiment.config).is_file():
        raise SystemExit(f"missing config {experiment.config}")
    render_launcher(commit, experiment)


def command_launch(args: argparse.Namespace) -> dict:
    experiment = resolve_experiment(args.experiment)
    commit = resolve_commit(args.commit)
    _validate(experiment, commit)
    reference = submit(commit, experiment)
    print(f"reference saved to {_save_reference(reference, experiment, commit)}")
    return reference


def command_run(args: argparse.Namespace) -> None:
    reference = command_launch(args)
    experiment = resolve_experiment(args.experiment)
    status = wait(reference["kernel"], poll_seconds=args.poll_seconds)
    reference.update({"terminal_status": status, "completed_at": utc_now(),
                      "upstream_identity": experiment.upstream_identity})
    _save_reference(reference, experiment, reference["git_sha"])
    if status != "COMPLETE":
        show_logs(reference["exact_version"], args.log_lines)
        raise SystemExit(f"{reference['kernel']} finished as {status}")
    collect(reference)


def _load_reference(args: argparse.Namespace) -> dict:
    experiment = resolve_experiment(args.experiment)
    commit = resolve_commit(args.commit)
    path = _reference_path(experiment, commit)
    if not path.is_file():
        raise SystemExit(f"no submission reference at {path}; launch the experiment first")
    return json.loads(path.read_text())


def command_status(args: argparse.Namespace) -> None:
    reference = _load_reference(args)
    status = kernel_status(reference["kernel"])
    print(json.dumps({"exact_version": reference["exact_version"], "status": status, "url": reference["url"]}, indent=2))


def command_logs(args: argparse.Namespace) -> None:
    show_logs(_load_reference(args)["exact_version"], args.log_lines)


def command_collect(args: argparse.Namespace) -> None:
    reference = _load_reference(args)
    status = kernel_status(reference["kernel"])
    if status not in TERMINAL_STATUSES:
        raise SystemExit(f"{reference['kernel']} is {status}; collect only a finished run")
    experiment = resolve_experiment(args.experiment)
    reference.update({"terminal_status": status, "completed_at": utc_now(),
                      "upstream_identity": experiment.upstream_identity})
    collect(reference, keep_payload=args.keep_payload)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, handler, helptext in (
        ("run", command_run, "validate, render, submit, wait, and collect"),
        ("launch", command_launch, "validate, render, and submit; return the immutable reference"),
        ("status", command_status, "report the state of an exact submitted version"),
        ("logs", command_logs, "show execution logs, especially on failure"),
        ("collect", command_collect, "fetch and verify compact audit artifacts only"),
    ):
        sub = subparsers.add_parser(name, help=helptext)
        sub.add_argument("--experiment", required=True)
        sub.add_argument("--commit", help="defaults to HEAD")
        sub.add_argument("--poll-seconds", type=int, default=60)
        sub.add_argument("--log-lines", type=int, default=80)
        sub.add_argument("--keep-payload", type=Path, help="also retain the verified analysis archive here")
        sub.set_defaults(handler=handler)
    args = parser.parse_args(argv)
    args.handler(args)


if __name__ == "__main__":
    main()
