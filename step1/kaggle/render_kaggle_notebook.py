"""Render a one-run Kaggle launcher without hand-editing a pinned SHA."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sha", required=True, help="full 40-character source commit SHA")
    parser.add_argument("--config", required=True, help="repository-relative TOML config path")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        parser.error("--sha must be a lowercase 40-character Git SHA")
    if not re.fullmatch(r"step1/configs/kaggle/t4x2_[a-z0-9_]+\.toml", args.config):
        parser.error("--config must be a Step 1 Kaggle config")
    template = Path(__file__).with_name("step1_t4x2.ipynb")
    notebook = json.loads(template.read_text(encoding="utf-8"))
    replacements = {"<FULL_40_CHARACTER_COMMIT_SHA>": args.sha, "step1/configs/kaggle/t4x2_dense_seed0.toml": args.config}
    for cell in notebook["cells"]:
        for index, line in enumerate(cell.get("source", [])):
            for old, new in replacements.items(): line = line.replace(old, new)
            cell["source"][index] = line
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__": main()
