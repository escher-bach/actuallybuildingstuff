from __future__ import annotations

import importlib.util
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

from step1_experiments.artifacts import (
    ANALYSIS_MAX_FILE_BYTES,
    RunArtifacts,
    analysis_candidates,
    write_analysis_archive,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPO_ROOT = PROJECT_ROOT.parent
TOOL = REPO_ROOT / "tools" / "kaggle_run.py"
REGISTRY = PROJECT_ROOT / "step1/kaggle/experiments.toml"


def _load_tool():
    spec = importlib.util.spec_from_file_location("kaggle_run", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    # Registered before execution so dataclasses can resolve the module's own
    # annotations under `from __future__ import annotations`.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ExperimentRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.tool = _load_tool()
        self.registry = self.tool.load_registry(REGISTRY)

    def test_every_declared_experiment_resolves_to_existing_files(self) -> None:
        for name in self.registry["experiments"]:
            with self.subTest(experiment=name):
                experiment = self.tool.resolve_experiment(name, self.registry)
                self.assertTrue((PROJECT_ROOT / experiment.config).is_file(), experiment.config)
                self.assertTrue((PROJECT_ROOT / experiment.launcher_template).is_file())
                self.assertEqual(experiment.owner, "aniruddhavarma")

    def test_slug_is_immutable_per_commit(self) -> None:
        experiment = self.tool.resolve_experiment("rlvr-warmstart-seed0", self.registry)
        self.assertEqual(experiment.slug("a" * 40), "step1-rlvr-warmstart-seed0-aaaaaaa")
        self.assertNotEqual(experiment.slug("a" * 40), experiment.slug("b" * 40))

    def test_unknown_experiment_is_rejected(self) -> None:
        with self.assertRaises(SystemExit):
            self.tool.resolve_experiment("not-declared", self.registry)

    def test_warmstart_declares_its_upstream_with_an_expected_identity(self) -> None:
        experiment = self.tool.resolve_experiment("rlvr-warmstart-seed0", self.registry)
        self.assertEqual(experiment.kernel_sources, ("aniruddhavarma/step1-t4x2-dense-seed0-84f2938",))
        identity = experiment.upstream_identity["aniruddhavarma/step1-t4x2-dense-seed0-84f2938"]
        source = json.loads(json.dumps(identity))
        for field in ("git_sha", "config_hash", "model_state_sha256"):
            self.assertRegex(source[field], r"^[0-9a-f]{40,64}$")
        # The registry must agree with the config the run itself verifies.
        import tomllib

        with (PROJECT_ROOT / experiment.config).open("rb") as handle:
            config_source = tomllib.load(handle)["source"]
        self.assertEqual(config_source["git_sha"], source["git_sha"])
        self.assertEqual(config_source["model_state_sha256"], source["model_state_sha256"])

    def test_collection_pattern_matches_nested_evidence_and_excludes_weights(self) -> None:
        import re

        pattern = re.compile(self.tool.ANALYSIS_PATTERN)
        run = "t4x2-rlvr-warmstart-seed0-5b5e5cf5fe55-ca566773c8ef"
        # Kaggle reports paths relative to /kaggle/working, not bare filenames.
        for collected in (
            "step1-results/latest-summary.json",
            f"step1-results/{run}-analysis.tar.gz",
            f"step1-results/{run}-analysis.sha256",
        ):
            self.assertTrue(pattern.fullmatch(collected), collected)
        for excluded in (
            f"step1-results/{run}.tar.gz",
            f"step1-results/{run}.sha256",
            f"step1-results/{run}/checkpoints/checkpoint-191/model.safetensors",
        ):
            self.assertIsNone(pattern.fullmatch(excluded), excluded)

    def test_staged_metadata_is_generated_not_checked_in(self) -> None:
        experiment = self.tool.resolve_experiment("rlvr-warmstart-seed0", self.registry)
        commit = "c" * 40
        self.tool.render_launcher = lambda *_args, **_kwargs: json.dumps(
            {"cells": [], "metadata": {}, "nbformat": 4, "nbformat_minor": 5}
        )
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory)
            notebook = self.tool.stage(commit, experiment, staging)
            metadata = json.loads((staging / "kernel-metadata.json").read_text())
        self.assertEqual(notebook.name, f"{experiment.slug(commit)}.ipynb")
        self.assertEqual(metadata["id"], f"aniruddhavarma/{experiment.slug(commit)}")
        self.assertEqual(metadata["machine_shape"], "NvidiaTeslaT4")
        self.assertTrue(metadata["enable_gpu"] and metadata["enable_internet"] and metadata["is_private"])
        self.assertEqual(metadata["kernel_sources"], list(experiment.kernel_sources))
        self.assertLessEqual(len(metadata["title"]), 50)


class AnalysisPayload(unittest.TestCase):
    def _run_dir(self, root: Path) -> Path:
        artifacts = RunArtifacts(root, "run-1", "a" * 40, "b" * 64)
        run = artifacts.run_dir
        (run / "analysis").mkdir(parents=True, exist_ok=True)
        (run / "rlvr_report.json").write_text('{"contract": "step1_rlvr_grpo_v1"}')
        (run / "evaluation" / "metrics.json").write_text("{}")
        (run / "checkpoints" / "checkpoint-6").mkdir(parents=True)
        (run / "checkpoints" / "checkpoint-6" / "model.safetensors").write_bytes(b"\0" * 4096)
        (run / "outcome_only_from_dense" / "budget-0-model").mkdir(parents=True)
        (run / "outcome_only_from_dense" / "budget-0-model" / "model.safetensors").write_bytes(b"\0" * 4096)
        (run / "logs" / "runner.log").write_bytes(b"x" * (ANALYSIS_MAX_FILE_BYTES + 1024))
        return artifacts

    def test_payload_excludes_weights_and_truncates_oversized_logs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            artifacts = self._run_dir(Path(directory))
            names = [str(path.relative_to(artifacts.run_dir)) for path in analysis_candidates(artifacts.run_dir)]
            self.assertIn("rlvr_report.json", names)
            self.assertFalse([name for name in names if "safetensors" in name])
            report = write_analysis_archive(artifacts.run_dir, "run-1", Path(directory) / "run-1-analysis.tar.gz")
            with tarfile.open(Path(directory) / "run-1-analysis.tar.gz") as tar:
                members = tar.getnames()
            self.assertIn("run-1/rlvr_report.json", members)
            self.assertIn("run-1/logs/runner.log.tail", members)
            self.assertEqual(report["truncated"][0]["retained_bytes"], ANALYSIS_MAX_FILE_BYTES)

    def test_package_writes_both_payloads_and_a_readable_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifacts = self._run_dir(root)
            summary = artifacts.package(True)
            self.assertTrue((root / "run-1-analysis.tar.gz").is_file())
            self.assertTrue((root / "run-1-analysis.sha256").is_file())
            self.assertTrue((root / "run-1.tar.gz").is_file())
            self.assertTrue((root / "run-1.sha256").is_file())
            on_disk = json.loads((root / "latest-summary.json").read_text())
            self.assertEqual(on_disk, summary)
            self.assertEqual(on_disk["analysis"]["sha256"], (root / "run-1-analysis.sha256").read_text().split()[0])
            # The recovery payload carries weights; the analysis payload must not.
            self.assertLess(on_disk["analysis"]["bytes"], on_disk["recovery"]["bytes"])
            self.assertTrue((artifacts.run_dir / "analysis" / "provenance.json").is_file())
            self.assertTrue((artifacts.run_dir / "analysis" / "summary.json").is_file())


if __name__ == "__main__":
    unittest.main()
