from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from step1_experiments.transfer import _MilestoneSaves, load_transfer_config, locate_dense_source, transfer_plan
from step1_experiments.terminal_transfer import _metric_deltas, _validate_terminal_plan


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG = PROJECT_ROOT / "step1/configs/kaggle/t4x2_rendering_b_transfer_seed0.toml"
TERMINAL_CONFIGS = (
    PROJECT_ROOT / "step1/configs/kaggle/t4x2_rendering_b_terminal_seed0.toml",
    PROJECT_ROOT / "step1/configs/kaggle/t4x2_rendering_b_terminal_seed1.toml",
)
NOTEBOOK = PROJECT_ROOT / "step1/kaggle/step1_rendering_b_transfer.ipynb"
TERMINAL_NOTEBOOKS = (
    PROJECT_ROOT / "step1/kaggle/step1_rendering_b_terminal_seed0.ipynb",
    PROJECT_ROOT / "step1/kaggle/step1_rendering_b_terminal_seed1.ipynb",
)
RENDERER = PROJECT_ROOT / "step1/kaggle/render_preflight_notebook.py"


class TransferContracts(unittest.TestCase):
    def test_exact_two_rank_foundation_budget_plan(self) -> None:
        plan = transfer_plan(load_transfer_config(CONFIG), world_size=2)
        self.assertEqual(plan.budgets_updates, (0, 31, 92, 306, 916, 3052))
        self.assertEqual(plan.budgets_nominal_global_input_tokens, (0, 1_015_808, 3_014_656, 10_027_008, 30_015_488, 100_007_936))
        self.assertEqual(plan.gradient_accumulation_steps, 2)
        self.assertEqual(plan.calibration_episodes, 32768)
        self.assertEqual(plan.calibration_seed, 23260811)
        self.assertEqual(plan.evaluation_seed, 21260811)

    def test_source_locator_uses_exact_report_identity(self) -> None:
        expected = load_transfer_config(CONFIG)["source"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "nested/run"
            report = run / "production/training_report.json"
            report.parent.mkdir(parents=True)
            report.write_text(json.dumps({
                "contract": "step1_dense_training_v1",
                "source_git_sha": expected["git_sha"],
                "config_hash": expected["config_hash"],
            }))
            self.assertEqual(locate_dense_source([root], expected), run.resolve())

    def test_terminal_budget_plans_are_independent_endpoint_grid(self) -> None:
        for index, config_path in enumerate(TERMINAL_CONFIGS):
            plan = transfer_plan(load_transfer_config(config_path), world_size=2)
            _validate_terminal_plan(plan)
            self.assertEqual(plan.budgets_updates, (0, 92, 306, 916, 3052))
            self.assertEqual(
                plan.budgets_nominal_global_input_tokens,
                (0, 3_014_656, 10_027_008, 30_015_488, 100_007_936),
            )
            self.assertEqual(plan.root_seed, 20260811 + index)
            self.assertEqual(sum(plan.budgets_nominal_global_input_tokens[1:]), 143_065_088)

    def test_terminal_paired_delta_preserves_undefined_metrics(self) -> None:
        metrics = {
            "irreversible": {
                "success_rate": 0.4,
                "mean_success_excess_cost": None,
            }
        }
        baseline = {
            "irreversible": {
                "success_rate": 0.25,
                "mean_success_excess_cost": 0.1,
            }
        }
        delta = _metric_deltas(metrics, baseline)["irreversible"]
        self.assertAlmostEqual(delta["success_rate"], 0.15)
        self.assertIsNone(delta["mean_success_excess_cost"])

    def test_milestone_callback_suppresses_uniform_nonmilestone_saves(self) -> None:
        callback = _MilestoneSaves((31, 92))
        control = SimpleNamespace(should_save=True)
        callback.on_step_end(None, SimpleNamespace(global_step=30), control)
        self.assertFalse(control.should_save)
        callback.on_step_end(None, SimpleNamespace(global_step=31), control)
        self.assertTrue(control.should_save)

    def test_sha_template_has_exact_operational_checks_without_thresholds(self) -> None:
        spec = importlib.util.spec_from_file_location("transfer_renderer", RENDERER)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rendered.ipynb"
            module.render_template(NOTEBOOK.read_text(), "c" * 40, output)
            document = json.loads(output.read_text())
        bootstrap = "".join(document["cells"][1]["source"]).split("WORKING =", 1)[0]
        namespace = {}
        exec(bootstrap, namespace)
        self.assertEqual(namespace["GIT_COMMIT"], "c" * 40)
        text = "\n".join("".join(cell.get("source", [])) for cell in document["cells"])
        self.assertIn("--nproc_per_node=2", text)
        self.assertIn("[0, 31, 92, 306, 916, 3052]", text)
        self.assertIn("prefix_equivalence", text)
        self.assertIn("reversible_control", text)
        self.assertIn("expected_state_sha256", text)
        self.assertIn("zero_shot_interface_diagnostic_not_transfer", text)
        self.assertNotIn("success_rate >=", text)

    def test_terminal_notebooks_are_sha_renderable_and_launch_independent_runner(self) -> None:
        spec = importlib.util.spec_from_file_location("transfer_renderer", RENDERER)
        module = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(module)
        for notebook in TERMINAL_NOTEBOOKS:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "rendered.ipynb"
                module.render_template(notebook.read_text(), "d" * 40, output)
                document = json.loads(output.read_text(encoding="utf-8"))
            text = "\n".join("".join(cell.get("source", [])) for cell in document["cells"])
            self.assertIn("step1_experiments.terminal_transfer", text)
            self.assertIn("[0,92,306,916,3052]", text)
            self.assertIn("terminal_learning_rate", text)
            self.assertIn("--nproc_per_node=2", text)
            self.assertNotIn("success_rate >=", text)


if __name__ == "__main__":
    unittest.main()
