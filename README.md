# Repertoire

Excavating a basis of procedurally-generated task families from prior literature, translating them into one interface, and measuring which of them survive.

The two specifications are the authority; everything else here serves them.

| | |
|---|---|
| [task-specification.md](task-specification.md) | What a task family is, the four levels, admissibility A1–A7, the §7 interface, the measurement |
| [repertoire-specification.md](repertoire-specification.md) | Which families to build and where to find them. **This is the work being done here** — D1 register → D2 candidate set → D3 measured basis |

## State

| Step (Repertoire Spec §11) | Status |
|---|---|
| 1 — Held-out partition, sealed before inspection | **done** — [docs/00-heldout-partition.md](docs/00-heldout-partition.md) |
| 2 — Register veins §2.1–§2.5, translate, triage | **§2.4 outstanding** — was wrongly marked done; see [hazards §16](docs/07-hazards.md). 24 rows, saturation 14 entries over 8 sources |
| 3 — §2.6 practice traces under an existing coding scheme | **read; gate NOT cleared** — no inter-annotator reliability figure traceable to a primary source, and §11 step 3's gate is exactly that. See [verification-log](register/verification-log.md) |
| 4 — Paradigm coverage check | **done** — 7 of 10 paradigms fall out cleanly; 3 do not, clustering into exactly 2 missing axes. [docs/06](docs/06-paradigm-coverage.md) |
| 5 — Candidate set, ~30 families incl. plants | 7 implemented; 5 of 6 plant roles live. 46 tests passing |
| 6 — Matrix, planted-basis validation first | not started — gated on Task Spec §8 step 4 |
| 7 — Read decomposition, prune, repair → **D3** | not started |

## Layout

```
docs/00-heldout-partition.md      SEALED. Held-out domains and veins. Do not read what is listed.
docs/01-review-protocol.md        Vein order and why; delegation rules; what gates what
docs/02-predicted-decomposition.md PRE-REGISTERED. Expected blocks + abandonment conditions
docs/03-interface-findings.md     What implementing against the §7 protocol taught about the protocol
docs/04-radical-incidental.md     The empirical test for k vs e, and what its first run found
docs/05-trace-thinning.md         Why the Task Spec's trace-thinning default may be backwards
docs/06-paradigm-coverage.md      §11 step 4: which established paradigms fall out, and the two axes missing
docs/07-hazards.md                HAZARD REGISTER. Everything that went, or nearly went, silently wrong
register/README.md                Row lifecycle and the three rules that are expensive to skip
register/rows/*.toml              D1. One prior family per file
register/primitives.toml          Controlled vocabulary. Declare before use
register/reading-log.toml         Ordered sources processed; drives the saturation curve
src/repertoire/form.py            The §7 protocol and the register schema, in one file on purpose
src/repertoire/register.py        validate / coverage / saturation / export
src/repertoire/vocab.py           One shared vocabulary, fixed before any family
src/repertoire/families/          D2. Implemented families
src/repertoire/a3_test.py         A3 encoding-leak test (run before the matrix)
src/repertoire/expectations.py    PRE-COMMITTED. The §11 step 6 gate, as assertions not prose
register/verification-log.md      Curator checks of every source promoted to verified=true
notes/vein-*-report.md            Raw subagent literature reports, before curation into rows
```

## Running the register tools

```bash
PYTHONPATH=src python -m repertoire.register validate
```

`coverage` prints the §4 grid and marks gaps — the gaps name the vein to read next. `saturation` prints distinct primitives against sources processed, which is the review's stopping rule (§7). `export` writes the D1 spreadsheet.

## Three things to know before touching this

**The seal is real.** [docs/00-heldout-partition.md](docs/00-heldout-partition.md) lists domains and sub-literatures that must not be read until the basis is frozen. Reading one does not invalidate the project, but *failing to log it* makes a coverage claim false without anyone knowing. Log the contamination.

**The prediction is pre-registered.** [docs/02-predicted-decomposition.md](docs/02-predicted-decomposition.md) above its §5 is append-only. The most valuable result available here is a mismatch between the field's inherited taxonomy and the measured blocks, and it evaporates if the prediction is written after the matrix.

**Read the hazard register before adding a family.** [docs/07-hazards.md](docs/07-hazards.md) lists fourteen things that have gone wrong, most of them silently — a plant that stopped being a plant, a calibration exemplar that degraded into L1, an A2 check that passed on content rendered as `PAD` tokens. The severity column ranks by *how likely a failure is to pass unnoticed*, not by how bad it is. Loud failures are cheap.

**Rejections are output.** A row that says "fails A4 at published 4×4, admissible at 12×12 or composed with a modulus family" is worth more than a row that says pass. The reasons cluster, and the clusters are informative about the interface itself.
