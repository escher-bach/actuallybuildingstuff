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

    shard_i, shard_n = (int(x) for x in args.shard.split("/"))
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
                fam, k, dial, budget, p["model"], seeds=tuple(p["seeds"]),
                entropy_episodes=p["entropy_episodes"],
                out_dir=out / name, device=args.device,
                shard=(shard_i, shard_n),
            )
            if shard_n > 1:
                # This process ran only its slice. Merge in whatever the other
                # shards have finished; if they have not, the reading is refused
                # for having too few points rather than read off a partial curve.
                from .sweep import collect

                res = collect(out / name, dial.name, budget, fam.name, k)
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


def cmd_calibrate(args) -> int:
    """Find the budget at which the family is learnable at all, before sweeping.

    **The cheapest diagnostic available, and the first T4 sweep is why it exists.**
    That sweep spent ~50 GPU-minutes to discover that the model had learned the
    answer marginal and not the task -- at every dial setting, so every transfer
    number was transfer of the marginal.

    L0 is the diagnostic because it is the easiest possible version: theta is
    fully stated in the preamble, the answer is a deterministic function of what
    is written there, and the Bayes floor is exactly 0. **If L0 does not come
    down, nothing above it is measurable**, and no amount of sweeping will make
    it so. One arm at a few budgets locates the knee for the whole sweep.
    """
    from .episode import spec_for_level
    from .train import train_run

    p = PRESETS[args.preset]
    fam, k = _family(args.family, args.k)
    spec = spec_for_level(Level.L0, T=p["T"])

    print(f"calibration: {fam.name} k={k} at L0 (Bayes floor is exactly 0)")
    print(f"  model: {p['model'].as_dict()}")
    print(f"  device: {device_report()['name']}\n")
    print(f"  {'steps':>8} {'L_final':>9} {'settled':>8} {'slope/step':>11}  reading")

    for steps in [int(x) for x in args.steps.split(",")]:
        budget = Budget(steps=steps, batch_size=p["batch_size"], max_len=p["max_len"],
                        lr=p["lr"], warmup=min(p["warmup"], steps // 10),
                        model=p["model"].as_dict())
        rec, _ = train_run(fam, k, spec, budget, model_cfg=p["model"], seed=0,
                           device=args.device, bayes_floor=0.0, bayes_floor_upper=0.0,
                           label=f"calib-{steps}")
        c = rec.content()
        verdict = (
            "learned -- use this budget or above" if c.l_final < 0.15
            else "close; try the next budget up" if c.l_final < 0.5
            else "has NOT learned L0; a sweep at this budget measures the marginal"
        )
        print(f"  {steps:>8} {c.l_final:>9.4f} {str(c.converged):>8} "
              f"{c.tail_slope:>11.2e}  {verdict}")

    print("\nL0's floor is 0, so L_final IS the distance from optimal.")
    print("Sweep at the smallest budget that reaches it -- residual entropy only")
    print("makes the task harder, never easier.")
    return 0


def cmd_diagnose(args) -> int:
    from .diagnose import run_all

    dev = "cpu" if args.device == "auto" else args.device
    print(f"inducer capability probes  (device={dev})\n")
    results = run_all(device=dev, quick=args.quick)
    for r in results:
        print("  " + r.report() + "\n")
    fixed_ok, variable_ok = results[0].passed, results[1].passed
    if fixed_ok and not variable_ok:
        print("READING: the model copies by POSITION and cannot match on CONTENT.")
        print("Every family here needs content matching -- L1-L3 are defined by")
        print("in-context inference about a theta resampled every episode -- so no")
        print("family will train until this passes. Change the model, not the family.")
        print("A family that looks learnable under this fault is one whose rule is")
        print("positional (junk_trivial: copy the previous answer), which is why")
        print("that one alone reaches its optimum.")
    elif not fixed_ok:
        print("READING: the model cannot even copy at a fixed offset. Something is")
        print("wrong with the model or the optimizer before any task question.")
    elif variable_ok and not results[2].passed:
        print("READING: content matching works in isolation but not in the family's")
        print("shape. The fault is between the two -- look at rendering.")
    return 0 if variable_ok else 1


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
    s.add_argument("--shard", default="0/1",
                   help="i/n -- run every n-th point, for n GPUs. The points are "
                        "independent (a fresh model per point, no gradient shared), "
                        "so n processes give an exact n-times speedup.")
    s.set_defaults(fn=cmd_sweep)

    cal = sub.add_parser(
        "calibrate",
        help="find the budget at which the family is learnable at all. Run this "
             "BEFORE a sweep -- it is one arm instead of 27 and it is what tells "
             "you whether a sweep would measure the task or the answer marginal.")
    cal.add_argument("--preset", default="t4", choices=sorted(PRESETS))
    cal.add_argument("--family", default="modular")
    cal.add_argument("--k", type=int, default=0)
    cal.add_argument("--steps", default="2500,10000,25000,60000")
    cal.add_argument("--device", default="auto")
    cal.set_defaults(fn=cmd_calibrate)

    d = sub.add_parser(
        "diagnose",
        help="capability probes for the inducer, independent of every family. "
             "Run this when a family will not learn and you need to know whether "
             "the fault is the family, the harness, or the model.")
    d.add_argument("--device", default="auto")
    d.add_argument("--quick", action="store_true", help="short runs; indicative only")
    d.set_defaults(fn=cmd_diagnose)

    r = sub.add_parser("read", help="apply the pre-committed reading to a saved sweep")
    r.add_argument("path")
    r.set_defaults(fn=cmd_read)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
