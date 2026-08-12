"""Render the Kaggle preflight launcher with the final immutable source SHA."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path


MARKER = "__FINAL_COMMIT_SHA__"
NOTEBOOK_RELATIVE_PATH = Path("step1/kaggle/step1_t4x2_preflight.ipynb")


def render(commit: str, output: Path | None = None) -> Path:
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a lowercase, full 40-character Git SHA")
    project_root = Path(__file__).resolve().parents[2]
    repository_root = Path(subprocess.check_output(
        ["git", "-C", str(project_root), "rev-parse", "--show-toplevel"], text=True
    ).strip())
    template_path = project_root.relative_to(repository_root) / NOTEBOOK_RELATIVE_PATH
    # Read the checked-in template from HEAD, so this command remains repeatable
    # even when its previous invocation has already written a pinned notebook.
    template = subprocess.check_output(
        ["git", "-C", str(repository_root), "show", f"HEAD:{template_path.as_posix()}"],
        text=True,
    )
    if template.count(MARKER) != 2:
        raise RuntimeError("the checked-in preflight notebook is not the expected SHA template")
    document = json.loads(template.replace(MARKER, commit))
    if output is None:
        output = project_root / NOTEBOOK_RELATIVE_PATH
    output.write_text(json.dumps(document, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    print(render(args.commit, args.output))


if __name__ == "__main__":
    main()
