"""Render a SHA-pinned Step 1 Kaggle notebook from its checked-in template."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


MARKER = "__FINAL_COMMIT_SHA__"
DEFAULT_NOTEBOOK_RELATIVE_PATH = Path("step1/kaggle/step1_t4x2_preflight.ipynb")


def validate_commit(commit: str) -> None:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a lowercase, full 40-character Git SHA")


def render_template(template: str, commit: str, output: Path) -> Path:
    """Render a validated template without consulting Git; used by regression tests."""
    validate_commit(commit)
    if template.count(MARKER) != 1:
        raise RuntimeError("the checked-in preflight notebook must contain exactly one commit placeholder")
    document = json.loads(template.replace(MARKER, commit))
    output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output


def render(
    commit: str,
    output: Path | None = None,
    notebook_relative_path: Path = DEFAULT_NOTEBOOK_RELATIVE_PATH,
) -> Path:
    validate_commit(commit)
    project_root = Path(__file__).resolve().parents[2]
    if notebook_relative_path.parts[:2] != ("step1", "kaggle") or notebook_relative_path.suffix != ".ipynb":
        raise ValueError("--notebook must be a checked-in step1/kaggle .ipynb template")
    repository_root = Path(subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"], text=True
    ).strip())
    template_path = project_root.relative_to(repository_root) / notebook_relative_path
    # Read the checked-in template from HEAD, so this command remains repeatable
    # even when its previous invocation has already written a pinned notebook.
    template = subprocess.check_output(
        ["git", "-C", str(repository_root), "show", f"HEAD:{template_path.as_posix()}"],
        text=True,
    )
    if output is None:
        output = project_root / notebook_relative_path
    return render_template(template, commit, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--notebook", type=Path, default=DEFAULT_NOTEBOOK_RELATIVE_PATH)
    args = parser.parse_args()
    print(render(args.commit, args.output, args.notebook))


if __name__ == "__main__":
    main()
