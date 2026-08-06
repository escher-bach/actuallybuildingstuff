# Resume point — paused 2026-08-06

Repo is clean: 9 register rows validating, 18/18 tests passing.

## Pending, in order

**1. Write 4 register rows from the §2.5 report.** The report is complete at
[vein-2.5-corrupt-localize-report.md](vein-2.5-corrupt-localize-report.md); only the rows are
outstanding. Curation decisions already made:

| Row | Status to assign | Why |
|---|---|---|
| `mutant-localization` | `translated` | PIT catalogue and the Defects4J FEP study both verified |
| `circuit-fault-localization` | `translated` | Reiter verified directly; GDE partially, corroborated by arXiv 2209.09819 |
| `proof-step-localization` | `translated` | mCoq README, Alchemy, litmus-test taxonomy verified — but note `P_Theta` has to be authored by us; no validated proof-corruption catalogue exists |
| `plan-fault-localization` | `lead` | Every source (VAL, PlanBench) is snippet-corroborated only. The A4 finding is ours and stands regardless, but the row's imported claims are not yet claims |

`primitives.toml` is already updated with this vein's folded aliases — **zero new slugs**, which is
the first real evidence for §10 condition 2.

**2. Reading-log entry** for vein 2.5 (n=2), listing those 4 rows.

**3. §3 quarantine row in [docs/02](../docs/02-predicted-decomposition.md)** for vein 2.5's own
taxonomies: mutation testing's adequacy/coupling-effect framing, model-based diagnosis's
minimal-diagnosis-cardinality axis, VAL's inexecutable/non-goal-reaching binary.

## Three findings worth not losing

- **The A1 equivalence trap bites unevenly across substrates**, and the ordering is useful:
  undecidable for programs (~26.5% of real fault-test pairs never propagate to an observable
  point), cheap and decidable for formal proofs (kernel checking is not general equivalence),
  fully resolvable for plans (one forward simulation). The repair differs per substrate and
  cannot be written once.
- **The plans family fails A4 outright at any plan length under full visibility** — a linear scan
  is both the intended and the brute solver. Structurally different from the programs/proofs
  failures, which pass at large k. The only real difficulty lever is observability budget, and
  that lever only exists *because* the L1/L2 "oracle answers what was asked" structure is already
  in the design. Worth noting as a case where the interface created a knob the source literature
  had no reason to name.
- **Difficulty knobs do not transfer across substrates.** More simultaneous faults makes circuits
  harder (masking) and programs *not* harder (the coupling effect: 1st-order-adequate tests kill
  >99% of higher-order mutants). Do not assume one k means one thing.

## After that

§2.5 second half (library learning, grammatical inference, planning generators) for the
`compose/decompose` and `state` coverage gaps, then §2.3 for the plants — that vein is where the
near-duplicate partner for `conjunction-identification` (Bruner's conjunctive concept attainment)
and an externally-attested prerequisite pair (Shepard–Hovland–Jenkins) should come from.
