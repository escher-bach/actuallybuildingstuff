from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK_PATH = PROJECT_ROOT / "step1" / "kaggle" / "step1_t4x2_preflight.ipynb"
DENSE_NOTEBOOK_PATH = PROJECT_ROOT / "step1" / "kaggle" / "step1_t4x2.ipynb"
RENDERER_PATH = PROJECT_ROOT / "step1" / "kaggle" / "render_preflight_notebook.py"


def _renderer_module():
    spec = importlib.util.spec_from_file_location("render_preflight_notebook", RENDERER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the preflight notebook renderer")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class KagglePreflightNotebookContracts(unittest.TestCase):
    def _bootstrap_prefix(self, document: dict) -> str:
        bootstrap = "".join(document["cells"][1]["source"])
        # The remainder creates /kaggle/working; exercising the validation
        # prefix is enough to prove the rendered notebook will pass its guard.
        return bootstrap.split("WORKING =", maxsplit=1)[0]

    def test_rendered_notebook_accepts_immutable_sha_without_bootstrap_io(self) -> None:
        renderer = _renderer_module()
        commit = "a" * 40
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rendered.ipynb"
            renderer.render_template(NOTEBOOK_PATH.read_text(encoding="utf-8"), commit, output)
            document = json.loads(output.read_text(encoding="utf-8"))
        self.assertEqual(document["nbformat"], 4)
        self.assertEqual(len(document["cells"]), 5)
        namespace: dict[str, object] = {}
        exec(self._bootstrap_prefix(document), namespace)
        self.assertEqual(namespace["GIT_COMMIT"], commit)

    def test_unrendered_template_fails_with_pinning_instruction(self) -> None:
        document = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        with self.assertRaisesRegex(RuntimeError, "render_preflight_notebook.py"):
            exec(self._bootstrap_prefix(document), {})

    def test_renderer_rejects_templates_with_more_than_one_commit_marker(self) -> None:
        renderer = _renderer_module()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "exactly one commit placeholder"):
                renderer.render_template("__FINAL_COMMIT_SHA__ __FINAL_COMMIT_SHA__", "a" * 40, Path(directory) / "bad.ipynb")

    def test_rendered_dense_notebook_has_exact_operational_not_scientific_contract(self) -> None:
        renderer = _renderer_module()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dense.ipynb"
            renderer.render_template(DENSE_NOTEBOOK_PATH.read_text(encoding="utf-8"), "b" * 40, output)
            document = json.loads(output.read_text(encoding="utf-8"))
        bootstrap = self._bootstrap_prefix(document)
        namespace: dict[str, object] = {}
        exec(bootstrap, namespace)
        self.assertEqual(namespace["GIT_COMMIT"], "b" * 40)
        contract = "".join(document["cells"][-1]["source"])
        self.assertIn("production/training_report.json", contract)
        self.assertIn("exact_state_dict", contract)
        self.assertIn("expected_state_sha256", contract)
        self.assertIn("actual_state_sha256", contract)
        self.assertIn("expected_plan", contract)
        self.assertIn("'world_size': world_size", contract)
        self.assertIn("'gradient_accumulation_steps'", contract)
        self.assertIn("'checkpoint_interval_nominal_global_input_tokens'", contract)
        self.assertIn("report['token_accounting'] == expected_plan", contract)
        self.assertIn("print(json.dumps(metrics", contract)
        self.assertNotIn("success_rate >=", contract)


if __name__ == "__main__":
    unittest.main()
