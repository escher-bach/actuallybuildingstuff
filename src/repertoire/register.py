"""D1 register: load, validate, export, and measure.

Rows live as TOML files in `register/rows/`, one per source family, hand-written
and diffable.  This module turns them into `RegisterRow` objects, checks them,
and emits the three things the review actually steers on:

  * `validate`   -- which rows are not yet decidable
  * `coverage`   -- the section 4 grid, which tells you which vein to read next
  * `saturation` -- distinct primitives against sources processed, which is the
                    stopping rule (section 7).  It is the cheapest
                    high-information measurement in the document and it needs no
                    compute, so there is no excuse for not having it wired from
                    row one.

Usage:
    python -m repertoire.register validate
    python -m repertoire.register coverage
    python -m repertoire.register saturation
    python -m repertoire.register export register/register.csv
"""

from __future__ import annotations

import csv
import sys
import tomllib
from dataclasses import fields as dc_fields
from pathlib import Path

from .form import (
    Asymmetry,
    Check,
    ComplexityMode,
    Content,
    Form,
    Level,
    PlantRole,
    RegisterRow,
    Source,
    Status,
    Verdict,
    validate,
)

ROOT = Path(__file__).resolve().parents[2]
ROWS_DIR = ROOT / "register" / "rows"
READING_LOG = ROOT / "register" / "reading-log.toml"
PRIMITIVES = ROOT / "register" / "primitives.toml"

_CHECK_FIELD = {
    "theta_separable": "theta_separable",
    "A1": "a1_backward_generable",
    "A2": "a2_knowledge_free",
    "A3": "a3_encoding_varied",
    "A4": "a4_brute_force_resistant",
    "A5": "a5_semantically_coherent",
    "A6": "a6_transition_cheap_total",
    "A7": "a7_teacher_policy",
}


def _load_row(path: Path) -> RegisterRow:
    with path.open("rb") as fh:
        d = tomllib.load(fh)

    form_d = d.pop("form", {})
    known = {f.name for f in dc_fields(Form)}
    unknown = set(form_d) - known
    if unknown:
        raise ValueError(f"{path.name}: unknown form fields {sorted(unknown)}")
    form = Form(**form_d)

    checks_d = d.pop("checks", {})
    check_kwargs = {}
    for name, val in checks_d.items():
        if name not in _CHECK_FIELD:
            raise ValueError(f"{path.name}: unknown check {name!r}")
        check_kwargs[_CHECK_FIELD[name]] = Check(
            verdict=Verdict(val.get("verdict", "unknown")),
            reason=val.get("reason", ""),
            repair=val.get("repair", ""),
        )

    sources = [Source(**s) for s in d.pop("sources", [])]

    row = RegisterRow(
        id=d.pop("id"),
        title=d.pop("title", ""),
        vein=str(d.pop("vein", "")),
        their_term=d.pop("their_term", ""),
        substrate=d.pop("substrate", ""),
        sources=sources,
        form=form,
        levels=[Level(x) for x in d.pop("levels", [])],
        complexity_modes=[ComplexityMode(x) for x in d.pop("complexity_modes", [])],
        contents=[Content(x) for x in d.pop("contents", [])],
        asymmetries=[Asymmetry(x) for x in d.pop("asymmetries", [])],
        plant_role=PlantRole(d.pop("plant_role", "none")),
        plant_pair=d.pop("plant_pair", ""),
        primitives=d.pop("primitives", []),
        predicted_block=d.pop("predicted_block", ""),
        status=Status(d.pop("status", "lead")),
        notes=d.pop("notes", ""),
        **check_kwargs,
    )
    if d:
        raise ValueError(f"{path.name}: unknown top-level keys {sorted(d)}")
    if row.id != path.stem:
        raise ValueError(f"{path.name}: id {row.id!r} does not match filename")
    return row


def load_rows() -> list[RegisterRow]:
    if not ROWS_DIR.exists():
        return []
    return [_load_row(p) for p in sorted(ROWS_DIR.glob("*.toml"))]


# --------------------------------------------------------------------------


def load_primitives() -> dict[str, dict]:
    if not PRIMITIVES.exists():
        return {}
    with PRIMITIVES.open("rb") as fh:
        return tomllib.load(fh)


def cmd_validate(rows: list[RegisterRow]) -> int:
    """Row-level checks, plus the one cross-row check that matters.

    An undeclared primitive slug is a validation error and not a nit: the
    saturation curve counts distinct slugs, so a synonym coined in row 40 shows
    up as basis growth that is not there.  Declaring first forces the question
    'is this the same operation I already named?' at the moment it is cheap.
    """
    declared = load_primitives()
    bad = 0
    for r in rows:
        problems = validate(r)
        undeclared = [p for p in r.primitives if p not in declared]
        if undeclared:
            problems.append(
                f"primitives not declared in register/primitives.toml: {undeclared}"
                " -- declare it, or fold it into the slug it duplicates"
            )
        if problems:
            bad += 1
            print(f"\n{r.id}  [{r.status.value}]")
            for p in problems:
                print(f"    - {p}")
    live = [r for r in rows if r.status not in (Status.SEALED, Status.LEAD)]
    print(f"\n{len(rows)} rows ({len(live)} past lead stage), {bad} with problems")
    return 1 if bad else 0


def cmd_coverage(rows: list[RegisterRow]) -> int:
    """The section 4 grid. Gaps name the vein to read next."""
    live = [r for r in rows if r.status in (Status.TRANSLATED, Status.IMPLEMENTED)]
    print(f"coverage over {len(live)} translated/implemented rows\n")

    def tally(title, enum_cls, attr):
        print(title)
        for member in enum_cls:
            hits = [r.id for r in live if member in getattr(r, attr)]
            flag = "  <-- GAP" if not hits else ""
            print(f"  {member.value:<22} {len(hits):>3}{flag}")
            if hits:
                print(f"      {', '.join(hits)}")
        print()

    tally("LEVEL", Level, "levels")
    tally("COMPLEXITY MODE", ComplexityMode, "complexity_modes")
    tally("CONTENT", Content, "contents")
    tally("GENERATION ASYMMETRY", Asymmetry, "asymmetries")

    print("PLANTS (section 5)")
    for role in PlantRole:
        if role is PlantRole.NONE:
            continue
        hits = [r.id for r in live if r.plant_role is role]
        flag = "  <-- MISSING" if not hits else ""
        print(f"  {role.value:<18} {len(hits):>3}{flag}  {', '.join(hits)}")
    print()

    by_vein: dict[str, int] = {}
    for r in rows:
        by_vein[r.vein] = by_vein.get(r.vein, 0) + 1
    print("ROWS BY VEIN")
    for v in sorted(by_vein):
        print(f"  section {v}  {by_vein[v]:>3}")
    return 0


def cmd_saturation(rows: list[RegisterRow]) -> int:
    """Distinct primitives against sources processed -- section 7's stopping rule.

    Still climbing linearly at source 300 means the basis is unbounded.
    Flattening at 40-80 means it is in hand and further reading is redundant.
    """
    by_id = {r.id: r for r in rows}
    if not READING_LOG.exists():
        print("no reading log yet; nothing processed")
        return 0
    with READING_LOG.open("rb") as fh:
        log = tomllib.load(fh)

    seen: set[str] = set()
    print(f"{'n':>4}  {'new':>4}  {'total':>5}  source")
    for n, entry in enumerate(log.get("entry", []), start=1):
        new: set[str] = set()
        for rid in entry.get("rows", []):
            row = by_id.get(rid)
            if row is None:
                print(f"  !! reading log references unknown row {rid!r}")
                continue
            new |= set(row.primitives) - seen
        seen |= new
        print(f"{n:>4}  {len(new):>4}  {len(seen):>5}  {entry.get('source', '?')}")
        if new:
            print(f"        + {', '.join(sorted(new))}")
    print(f"\n{len(seen)} distinct primitives over {len(log.get('entry', []))} sources")
    return 0


def cmd_export(rows: list[RegisterRow], out: Path) -> int:
    """D1's spreadsheet half. Prose notes stay in the TOML."""
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = [
        "id", "title", "vein", "their_term", "status", "levels",
        "theta", "oracle", "k_semantics", "n_encodings",
        *_CHECK_FIELD.keys(),
        "complexity_modes", "contents", "asymmetries",
        "plant_role", "plant_pair", "primitives", "predicted_block",
        "sources", "substrate",
    ]
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for r in rows:
            checks = r.checks
            w.writerow([
                r.id, r.title, r.vein, r.their_term, r.status.value,
                " ".join(l.value for l in r.levels),
                r.form.theta, r.form.oracle, r.form.k_semantics, len(r.form.encodings),
                *(checks[c].verdict.value for c in _CHECK_FIELD),
                " ".join(m.value for m in r.complexity_modes),
                " ".join(c.value for c in r.contents),
                " ".join(a.value for a in r.asymmetries),
                r.plant_role.value, r.plant_pair,
                " ".join(r.primitives), r.predicted_block,
                " | ".join(s.citation for s in r.sources),
                r.substrate,
            ])
    print(f"wrote {out} ({len(rows)} rows)")
    return 0


def main(argv: list[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else "validate"
    rows = load_rows()
    if cmd == "validate":
        return cmd_validate(rows)
    if cmd == "coverage":
        return cmd_coverage(rows)
    if cmd == "saturation":
        return cmd_saturation(rows)
    if cmd == "export":
        out = Path(argv[2]) if len(argv) > 2 else ROOT / "register" / "register.csv"
        return cmd_export(rows, out)
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
