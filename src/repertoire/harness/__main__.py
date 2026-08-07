"""Command line for the harness.

    python -m repertoire.harness check     [--family stub|modular|parity]
    python -m repertoire.harness sweep     --preset smoke|t4|p100
    python -m repertoire.harness read      <sweep.json>

`check` runs the section 8 step 1 and step 2 gates and needs no GPU.  `sweep` is
section 8 step 5.  `read` applies the pre-committed curve reading to a saved
sweep without re-running anything, so the reading can be repeated by someone who
did not run it.

**Presets exist so the compute budget is a named thing rather than a set of
flags somebody typed.**  Section 4 requires the budget be held fixed across a
comparison and reported; a preset is a budget with a name, and the name goes in
the output. `smoke` finishes on a laptop CPU in about two minutes and exists so
a mistake is found before a GPU session is spent on it.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..form import Level
from .metrics import Budget
from .model import ModelConfig, device_report


PRESETS = {
    # name: (model, budget-without-model, T, seeds, dial points)
    "smoke": dict(
        model=ModelConfig(n_layer=2, n_head=2, d_model=64, d_ff=128, max_len=192),
        steps=60, batch_size=8, lr=2e-3, warmup=10, max_len=192,
        T=4, seeds=(0,), points=4, entropy_episodes=12,
    ),
    # Sized so a full sweep fits inside one Kaggle session with room to spare.
    # T4 has fp16 tensor cores; P100 does not, so the two differ in width rather
    # than in anything that changes what is measured.
    "t4": dict(
        model=ModelConfig(n_layer=4, n_head=4, d_model=256, d_ff=1024, max_len=512),
        steps=2500, batch_size=32, lr=1e-3, warmup=150, max_len=512,
        T=8, seeds=(0, 1, 2), points=9, entropy_episodes=64,
    ),
    "p100": dict(
        model=ModelConfig(n_layer=4, n_head=4, d_model=192, d_ff=768, max_len=512),
        steps=2500, batch_size=24, lr=1e-3, warmup=150, max_len=512,
        T=8, seeds=(0, 1, 2), points=9, entropy_episodes=64,
    ),
}


def _family(name: str, k: int):
    if name == "stub":
        from .stub import StubLookupFamily

        return StubLookupFamily(d=6), k
    if name == "modular":
        from ..families.modular import ModularHiddenPermutationFamily

        # pool 7 keeps |Theta| at 7560, which is what makes every exact target
        # and the measured x-axis affordable. Section 6's full m <= 20 is not.
        return ModularHiddenPermutationFamily(pool_size=7), 0
    if name == "parity":
        from ..families import ParityIdentificationFamily

        return ParityIdentificationFamily(), k
    raise SystemExit(f"unknown family {name!r}: expected stub, modular or parity")


def cmd_check(args) -> int:
    from random import Random

    from .entropy import check_prior_matches_sampler, measure_residual_entropy
    from .episode import spec_for_level
    from .protocol import check_query_sensitivity
    from .train import round_trip_all_levels

    fam, k = _family(args.family, args.k)
    print(f"{fam.name}  k={k}  device={device_report()['name']}\n")

    print("-- section 8 step 1: round-trip at all four levels")
    ok = True
    for level, summary in round_trip_all_levels(fam, k=k, T=args.T).items():
        if "error" in summary:
            ok = False
            print(f"   {level}  FAILED  {summary['error']}")
        else:
            print(f"   {level}  {summary['tokens']:>4} tokens, "
                  f"{summary['supervised']:>3} supervised, "
                  f"{summary['malformed']} malformed queries")

    print("\n-- A2, and whether the check can fail at all")
    print(f"   permuted_alphabet_check: {fam.permuted_alphabet_check(Random(0))}")

    print("\n-- L3 target sanity")
    sensitive, detail = check_query_sensitivity(fam, k, Random(0))
    print(f"   query-sensitive: {sensitive}\n     {detail}")
    prior_ok, prior_detail = check_prior_matches_sampler(fam, k, Random(0))
    print(f"   prior matches sampler: {prior_ok}\n     {prior_detail}")
    if not prior_ok:
        ok = False

    print("\n-- residual entropy, the sweep's x-axis")
    for level in (Level.L0, Level.L1):
        try:
            rep = measure_residual_entropy(fam, k, spec_for_level(level, T=args.T),
                                           n_episodes=args.entropy_episodes)
            print(f"   {level.value}: {rep.report()}")
        except Exception as exc:
            print(f"   {level.value}: unavailable -- {type(exc).__name__}: {exc}")

    print("\nOK" if ok else "\nFAILED")
    return 0 if ok else 1


def cmd_sweep(args) -> int:
    from .sweep import (
        compare_dials,
        observation_dial,
        preamble_dial,
        read_curve,
        run_sweep,
    )

    p = PRESETS[args.preset]
    fam, k = _family(args.family, args.k)
    budget = Budget(steps=p["steps"], batch_size=p["batch_size"], max_len=p["max_len"],
                    lr=p["lr"], warmup=p["warmup"], model=p["model"].as_dict())
    out = Path(args.out) / f"{fam.name}_k{k}_{args.preset}"

    print(f"sweep: {fam.name} k={k} preset={args.preset}")
    print(f"  {budget.describe()}")
    print(f"  device: {device_report()}")
    print(f"  out: {out}\n")

    dials = {
        # Same T on both, so `compare_dials` compares two curves and not two
        # different measurements that happen to be plotted the same way.
        "observations": observation_dial(max_free=p["points"] - 1, T=p["T"]),
        "preamble": preamble_dial(points=p["points"], T=p["T"]),
    }
    wanted = list(dials) if args.dial == "both" else [args.dial]

    results = {}
    for name in wanted:
        dial = dials[name]
        print(f"=== dial: {dial.name}")
        try:
            res = run_sweep(
                fam, k, dial, budget, p["model"], seeds=tuple(p["seeds"]), entropy_episodes=p["entropy_episodes"],
                out_dir=out / name, device=args.device,
            )
        except Exception as exc:
            print(f"  unavailable: {type(exc).__name__}: {exc}\n")
            continue
        results[name] = res
        print()
        print(res.report())

        fit, why = res.fit_to_read()
        if not fit:
            print(f"\n  READING REFUSED -- the sweep is not fit to read.\n    {why}")
            print("    Section 8 step 5 is the only step that can end the "
                  "programme; a verdict an underpowered run can produce is worse "
                  "than no verdict.\n")
            continue
        reading = read_curve(res.curve("transfer_fraction"),
                             noise=res.noise("transfer_fraction"))
        print(f"\n  ({why})")
        print(f"  READING: {reading.verdict}  ->  {reading.action}")
        print(f"    {reading.detail}\n")

    readable = {n: r for n, r in results.items() if r.fit_to_read()[0]}
    if len(readable) == 2:
        print("=== the two parametrizations against each other")
        print(compare_dials(*readable.values()))
    elif len(results) == 2:
        print("=== the two parametrizations cannot be compared: "
              "at least one is not fit to read")
    return 0


def cmd_read(args) -> int:
    from .sweep import read_curve

    data = json.loads(Path(args.path).read_text())
    pts = data["points"]
    by_setting: dict[float, list[dict]] = {}
    for pt in pts:
        by_setting.setdefault(pt["setting"], []).append(pt)

    curve = []
    for _, group in sorted(by_setting.items()):
        vals = [g["transfer"] / g["transfer_baseline"] for g in group
                if g.get("transfer") and g.get("transfer_baseline")]
        if not vals:
            continue
        h = sum(g["rule_entropy"] for g in group) / len(group)
        curve.append((h, max(0.0, min(1.0, 1.0 - sum(vals) / len(vals)))))
    curve.sort()

    print(f"{data['family']} k={data['k']} dial={data['dial']}")
    print(f"  budget fingerprint {data.get('budget_fingerprint')}")
    print(f"  {len(curve)} points\n")
    for h, y in curve:
        print(f"   H={h:8.4f}  transfer={y:6.3f}  {'#' * int(round(y * 40))}")
    r = read_curve(curve)
    print(f"\n  READING: {r.verdict}  ->  {r.action}")
    print(f"    {r.detail}")
    print(f"    confident: {r.confident}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m repertoire.harness")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="section 8 step 1 and 2 gates; no GPU needed")
    c.add_argument("--family", default="stub")
    c.add_argument("--k", type=int, default=1)
    c.add_argument("--T", type=int, default=6)
    c.add_argument("--entropy-episodes", type=int, default=24, dest="entropy_episodes")
    c.set_defaults(fn=cmd_check)

    s = sub.add_parser("sweep", help="section 8 step 5, the dial sweep")
    s.add_argument("--preset", default="smoke", choices=sorted(PRESETS))
    s.add_argument("--family", default="modular")
    s.add_argument("--k", type=int, default=0)
    s.add_argument("--dial", default="both",
                   choices=("both", "observations", "preamble"))
    s.add_argument("--out", default="runs")
    s.add_argument("--device", default="auto")
    s.set_defaults(fn=cmd_sweep)

    r = sub.add_parser("read", help="apply the pre-committed reading to a saved sweep")
    r.add_argument("path")
    r.set_defaults(fn=cmd_read)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
