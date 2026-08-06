# The Register (D1)

One TOML file per source family in `rows/`. The schema is `src/repertoire/form.py`, which is also where the §7 `TaskFamily` protocol lives — deliberately, because the only question a row answers is *does this thing round-trip through that protocol*.

```bash
PYTHONPATH=src python -m repertoire.register validate
```
```bash
PYTHONPATH=src python -m repertoire.register coverage
```
```bash
PYTHONPATH=src python -m repertoire.register saturation
```
```bash
PYTHONPATH=src python -m repertoire.register export register/register.csv
```

## Row lifecycle

`lead` → `registered` → `translated` → `implemented`, with `rejected` and `sealed` as terminal branches.

| Status | Means | Validation demands |
|---|---|---|
| `lead` | a name from someone's citation list | nothing — leads are the reading queue, not claims |
| `registered` | primary source actually read | at least one `verified = true` source |
| `translated` | round-trips through §7 on paper | the whole `[form]` block, all core checks resolved, ≥2 encodings, ≥1 primitive |
| `implemented` | code exists against `TaskFamily` | same, plus the code |
| `rejected` | inadmissible, with a named failing check | the reason. A rejection without one is a dead row |
| `sealed` | falls inside the held-out partition | which held-out item, and *dropped unread* |

**`unknown` is a legal verdict only before `translated`.** It is how the register says "we have not decided yet" without pretending. It must not survive into D2, and validation enforces that.

## Three rules that are easy to skip and expensive to skip

**Verified means the curator read it — not that a subagent said so.** Repertoire Spec §2 says names and dates in it are leads to verify, not citations. A row with `verified = false` on every source is a lead no matter what else is filled in. Do not promote a row on the strength of a secondary description, and never invent a citation to fill a gap — an unverifiable row is worth less than an absent one because it looks like evidence.

A delegated review's `VERIFIED` label promotes a source only to *candidate-verified*. `verified = true` requires an independent check recorded in [verification-log.md](verification-log.md). This is not bureaucratic caution: the first spot-check pass found 5 sources clean, 3 with drifting numbers, and **1 fabricated statistic** that had already propagated into an A1 verdict. The failure mode is narrow and predictable — real paper, real topic, invented or mangled number — which is precisely why it survives a plausibility read. **Check the numbers, counts and percentages; the structural claims have been reliable.**

**Record the repair, not just the verdict.** `REPAIRABLE` is the value that carries the work. "Fails A4 at published size" is a dead end; "fails A4 at published 4×4, admissible composed with a modulus family or at 12×12" is a build note. The calibration row `mod-arith-hidden-permutation.toml` fails A4 in exactly this way and is written out in full as the worked example of how to record it.

**Declare primitives before using them.** Every slug in `primitives` must exist in `primitives.toml`. The saturation curve counts distinct slugs and is the review's stopping rule (§7); a synonym coined in row 40 reads as basis growth that is not there. Declaring first forces the question *is this the same operation I already named?* while it is still cheap to answer. When you fold a source's term into an existing slug, keep it in that slug's `aliases` — those are the evidence that the unification is real rather than asserted.

## The reading log

`reading-log.toml` is ordered by *when a source was actually processed*, and it is what the saturation curve walks. **An entry with `rows = []` is not a gap — it is a source that yielded nothing, and those entries are what make the curve flatten honestly.** Omitting them turns the stopping rule into a self-fulfilling one.

## Delegation brief

Literature review is delegated one vein at a time, never in parallel. Every brief carries:

1. **The seal.** The held-out list from `docs/00-heldout-partition.md`, with the instruction to stop and report — not read — if a lead falls inside it.
2. **The form.** The `[form]` fields are the output format. A summary of a paper is not a row; a row says what θ is, how k enters, and why the oracle does not have to search.
3. **The two hazards of the vein**, from Repertoire Spec §2, restated. Every vein has a characteristic way of producing plausible-looking useless rows and the brief names it.
4. **Leave the ontology.** Take the apparatus. The source's own taxonomy goes in `predicted_block` — where §6 can be scored against it later — and nowhere else. This is §9's first failure mode and the one that voids the most valuable result available.
5. **Negative results are results.** A vein that yields three rows and eleven rejections has done its job, and the eleven reasons cluster into something informative about the interface.
