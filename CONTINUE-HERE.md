# Continue Here

*Single entry point for a fresh session picking up the **repertoire** work. Read this, then [README.md](README.md) for layout. You should not need any prior transcript.*

---

## 1. What this is, in one paragraph

Two specifications are the authority. [task-specification.md](task-specification.md) defines what a task family is — a program with a hidden parameter `θ` and a policy for revealing it — plus four levels (L0 stated / L1 identifiable / L2 model-chosen queries / L3 posterior target), admissibility constraints A1–A7, the §7 `TaskFamily` interface, and the prequential structural-content measurement. [repertoire-specification.md](repertoire-specification.md) is **the work being done here**: excavate task families from prior literature (D1 register), implement ~30 of them (D2), measure all-pairs transfer and read the capability decomposition off the matrix (D3). Novelty is in the unification and the measurement, not the inventory.

**A second session owns the harness** (Task Spec §8 steps 1, 2, 5) against [docs/09-harness-handoff.md](docs/09-harness-handoff.md). Do not build a harness here.

---

## 2. State

| | |
|---|---|
| Register rows | **27** (23 past `lead`), 0 validation problems |
| Implemented families | **9**, plus composition machinery |
| Tests | **101 passing** |
| Saturation curve | **14 vocabulary entries over 10 sources** — 7, 0, 3, 0, 1, 1, 0, 2, 0, 0. **Five veins added zero** |
| Plant roles | **6 of 6 populated** |
| Composition closure | **2.4%** of ordered pairs — one legal composite |
| Hazards documented | **25** |
| Commits | ~28 |

**§11 progress:** steps 1 (seal), 2 (all veins, incl. an added §2.7) and 4 (paradigm coverage) done. Step 3 read but **gate not cleared** — no inter-annotator reliability figure traceable to a primary source. Step 5 (~30 families) in progress. Steps 6–7 blocked on the harness.

---

## 3. The disciplines — these took the longest to learn, do not drop them

**Verify the numbers yourself.** A subagent's `VERIFIED` label makes a source *candidate*-verified only; `verified = true` in a row requires a curator check logged in [register/verification-log.md](register/verification-log.md). Across seven passes: mostly clean, several numeric drifts, **one outright fabricated statistic** that had already reached an A1 verdict. The pattern is narrow and predictable — **citations real, titles exact, structural claims reliable, numbers drift.** Check numbers, counts, percentages. Structural claims have never failed.

**The seal is a prohibition on *fetching*, not on citing.** [docs/00-heldout-partition.md](docs/00-heldout-partition.md) lists held-out domains and sub-literatures. Three contaminations occurred; **all three came from adjacency**, and the last from a paper whose title named a technique rather than its substrate. Log any contamination rather than reclassifying it — an unrecorded one makes a §10 condition-3 claim false without anyone knowing.

**Declare primitives before use.** [register/primitives.toml](register/primitives.toml) is a controlled vocabulary; undeclared slugs are a validation error. The saturation curve counts distinct slugs, so a synonym coined at row 40 reads as basis growth that isn't there. Folds that are genuine *claims* (not synonym cleanups) carry a `falsifier` saying what measurement would show the fold was wrong. Entries carry a `kind` — `operation` vs `trace-act` — and **the curve must be read per kind** or its totals mislead.

**Pre-registration is append-only.** [docs/02-predicted-decomposition.md](docs/02-predicted-decomposition.md) above its §5 is never edited. [src/repertoire/expectations.py](src/repertoire/expectations.py) carries the §11 step 6 gate as **executable assertions with thresholds already fixed** — nine of them, each with a test proving it can fail. Do not edit thresholds.

**Rejections are output.** A row saying "fails A4 at published 4×4, admissible at 12×12 or composed with a modulus family" is worth more than a row saying pass. Record the **repair**, not just the verdict.

**One subagent at a time.** Not a throughput concession — the primitive vocabulary is shared state, and two agents naming the same operation differently is exactly the drift that corrupts the curve. Fold each agent's primitives in before launching the next. Use Sonnet for literature review.

**Every delegation brief carries:** the accuracy directive (naming the fabrication incident concretely), the seal stated as a fetching prohibition with in/out boundaries drawn explicitly, an instruction to write the report file early, the existing primitive list with "zero new is a good answer," and "leave the ontology, take the apparatus" with a quarantine section for the source's own taxonomy.

---

## 4. Findings that changed the design

Read the doc if you touch the area. Each is a conclusion, not a note.

| | Finding |
|---|---|
| [docs/03](docs/03-interface-findings.md) | **`posterior(history, k)` cannot express the L3 target** — it's conditional on the query and the signature has nowhere to put one. Parity is fully identified after one observation yet its answer distribution marginalized over queries is exactly 0.5/0.5. Convention here: a trailing `(query, None)` is the pending query. The harness session decides centrally |
| [docs/04](docs/04-radical-incidental.md) | **An empirical test for `k` vs `e`**, from psychometrics' radical/incidental discipline. First run found **A3 was vacuous** — every encoding produced byte-identical episodes. Fixing it produced an ~85% length difference, so **the harness must normalize per token** |
| [docs/05](docs/05-trace-thinning.md) | **The trace-thinning default may be backwards.** §1.2 says "thin as `k` rises" and claims a literature; that literature fades against *learner competence*, never task difficulty, and says fading only bites when load is **high** |
| [docs/06](docs/06-paradigm-coverage.md) | **Two axes missing**: θ is assumed sampled-once-and-held; L2 assumes querying is free. 7 of 10 paradigms fall out cleanly, the 3 failures cluster into exactly these two. **Deferred until after the dial sweep** — adding axes first would be designing the answer |
| [docs/08](docs/08-a4-guarantee.md) | **A4 cannot be certified at our scale, only measured.** Asymptotic separations are frequently *empty* at n≈10. Every A4 verdict in the register is a **prediction, not a property** |
| [docs/11](docs/11-deployment-format.md) | **A correction, and the more instructive half.** I argued the repertoire should match the deployment format; that was an overfit. **A3 exists so format does not matter** — the remedy for a format gap is a more varied encoding set, not imitation. Records what was dropped and why |
| [docs/12](docs/12-l2-has-no-counterpart.md) | **Nobody supervises the query channel.** All outcome-graded — the design §2.1 rejects. And the field puts "decide what to work on next" in the *harness*, which is L2's job done by the scaffold |
| [docs/02](docs/02-predicted-decomposition.md) §5 | **P4**: composites should not be supplied by their parts. If `residual(C) ≈ 0`, composition adds nothing and §1.1's covering claim is false in our setting |

---

## 5. What to do next, in priority order

**1. More endomorphic families — highest value.** Closure is **2.4%** (one legal composite over seven families), which makes P4 a case study rather than a result. Composition needs `codomain(f_inner) ⊆ X_outer`, and almost every family maps a structured query to a *single label* — a dead end. `PermutedBitsFamily` is the only endomorphic one. **This is a design requirement on every new family**, and it's expensive to retrofit. See [src/repertoire/compose.py](src/repertoire/compose.py) and run `closure_report`.

**2. Implement more translated rows toward step 5's ~30.** 27 rows exist, 9 are implemented. Good candidates with complete form blocks: `decision-tree-identification`, `monotone-dnf-identification`, `mutant-localization`, `composition-chain`. **Read the row before implementing** — each carries its A-check verdicts and named repairs.

**3. `evidence-withdrawal`** ([row](register/rows/evidence-withdrawal.toml)) — a reveal policy over any base family: identify θ, then remove the evidence while θ is unchanged, and continue. The only family motivated by a *measured deficit* (0%→30% violation; survives 0% vs dropped 38%), though that study is **motivating evidence, not the definition** — see hazard 25. Its A4 repair is a hard constraint: post-withdrawal trials must be too few to re-identify θ from scratch, or it measures the base family with extra steps.

**4. Per-source `provenance | warrant` field** (hazard 20). `verified` conflates "this family came from that literature" with "this claim is true because the source says so." For self-evident mathematics no citation warrants anything; for empirical claims the citation *is* the claim. A reader currently can't tell whether a row would survive its citations being wrong.

**Blocked or withdrawn, do not start:** the matrix (needs the harness); the two missing axes (deliberately deferred until after the sweep). The transcript encoding is **withdrawn**, not blocked — see hazard 25.

---

## 6. Traps

[docs/07-hazards.md](docs/07-hazards.md) has 24, ranked by **how likely a failure is to pass unnoticed** rather than by how bad it is. Loud failures are cheap. The five worth knowing before you touch anything:

- **A plant that stops being a plant.** The junk-random family was nearly specified so its answers were a fixed function of the query — a *lookup table*, learnable within the episode. It would have measured as a memorization family while labelled junk, **at the instrument-validation gate**.
- **A family can satisfy its type signature and violate its purpose.** The calibration exemplar silently degraded into L1 because its rate grid included the deterministic endpoints. Only an assertion about the *property* caught it.
- **Equivariance under the wrong alphabet is still equivariance.** A family rendered content as `PAD`/`BOS` tokens; every id was valid, the in-vocabulary test passed, **and A2 passed too**.
- **A check that cannot fail is worse than no check.** Met twice. Always write the deliberately-broken case that must fail your check.
- **A type check is not the coherence check §1.1 asks for.** Composing junk over a real family *type-checks perfectly* and is constant — the "type-checking accident" the spec names. `compose()` runs two gates for this reason.

---

## 7. Commands

```bash
PYTHONPATH=src python -m repertoire.register validate
```

`coverage` prints the §4 grid and marks gaps — gaps name the vein to read next. `saturation` prints distinct primitives against sources processed (the review's stopping rule). `export` writes the D1 spreadsheet.

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

```bash
PYTHONPATH=src python -m repertoire.a3_test
```

Closure report: `from repertoire.compose import closure_report`.

---

## 8. Two things I would tell you if we spoke

**The most valuable single result available is not ours to produce.** Task Spec §8 step 5's dial sweep can *end the programme* — if it collapses monotonically, everything above L0 is void. It's with the harness session. Everything in this repertoire is downstream of it, and it is worth knowing early rather than being surprised.

**The register's honesty is its main asset.** Rows carry failed checks, unverified sources, blocked promotions, and named contaminations. That is not incompleteness — a rejected family with a reason is a result, and reasons cluster into something informative about the interface. Resist the urge to tidy it into something that looks finished.
