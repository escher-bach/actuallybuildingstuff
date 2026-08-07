# Harness Handoff

*For the session building the harness. This document is the brief; you should not need to reconstruct anything from the transcript that produced it.*

**You own Task Spec §8 steps 1, 2 and 5. I am continuing on the repertoire (Repertoire Spec §11 steps 5–7) in parallel, so §5 below is about not colliding.**

---

## 1. What you are building, and why it is urgent

Task Spec §8 step 5 says it plainly:

> **The dial sweep. This is the first result that matters.** […] This is the only step that can end the programme rather than redirect it. It is reachable in weeks with steps 1–2 done, and **it should not be deferred behind repertoire work.**

It has been deferred behind repertoire work. Nine literature veins, 26 register rows, 9 implemented families — **none of which is worth anything above L0 if the sweep collapses.** Your job is to make that answerable.

Build in this order, and do not skip the gates:

| Step | Deliverable | Gate (Task Spec §8) |
|---|---|---|
| 1 | Harness, no families: episode loop, four level wrappers, masking, rendering, prequential logger, seeded reproducibility | A stub episode round-trips at all four levels and the logger emits a structural-content number |
| 2 | The worked family (Task Spec §6) at all four levels | A2 permuted-alphabet check passes as a unit test; L1 shows within-episode loss decay; L2 runs with a computable `q*`; **L3 targets match a brute-force enumeration of consistent θ** |
| 5 | The dial sweep | See §6 below — this is the result |

Step 3 (two more families) and step 4 (the measurement instrument on a planted basis) are largely **already done** on my side — see §4. Do not rebuild them.

---

## 2. Requirements discovered by the repertoire work

These are not preferences. Each is a failure already found, and each would be expensive to rediscover after the sweep runs.

### 2.1 `posterior(history, k)` cannot express the L3 target — decide this first

Task Spec §2 defines the L3 target as "the target distribution over the answer token," which is necessarily **conditional on the query being asked**. The §7 signature has nowhere to put a query.

It is not cosmetic. Take parity: after one observation θ is *fully identified*, yet the answer distribution **marginalized over a random query** is exactly 0.5/0.5, because parity is balanced. A harness treating that marginal as the L3 target trains the model to be maximally uncertain about a rule it has already pinned down — **and does so worst on the families with the most structure.**

*Convention currently implemented in the families:* a trailing history entry whose answer is `None` is the pending query, and the posterior conditions on it. With no such entry, the return is the marginal — a coherent object, but **not** the training target.

**Your call to make, and make it centrally:** adopt that convention explicitly, or change the signature to `posterior(history, query, k)`. The second is cleaner. Either way it must be decided once — a family that guesses differently emits the wrong target silently. If you change the signature, say so in `docs/10-harness-findings.md` (create it) and I will update the families.

### 2.2 Normalize loss per token

The `slotted` encoding renders ~85% longer than its siblings (209 vs 113 tokens at k=1; 307 vs 163 at k=3). If episodes from one family differ that much in length on a per-episode coin flip, any per-episode loss average **mixes an encoding effect into the family effect**, and every structural-content number inherits it.

Measured by `src/repertoire/a3_test.py`. Run it; it takes seconds.

### 2.3 Stochastic families return distributions; the harness samples

`evaluate` takes no rng. That is load-bearing, not an oversight: it is the only way a stochastic family fits the protocol. `evaluate` returns `P(Y)` and **you** sample, which keeps seeded reconstructibility where §7 already requires it. `junk_random` and `probability_matching` are both built this way.

Corollary: `teacher_query` also takes no rng, so `q*` must be a deterministic function of `(θ, history)`. A7 wants one-pass computability anyway.

### 2.4 Episode length `T` is yours, and the register has numbers for it

§7 makes `T` a harness parameter. Do not let families set it. The register carries teaching-dimension bounds where the literature supplies them — e.g. the generic version-space family notes `max{t₀, log₂|Θ|} ≤ #MQ ≤ t₀·log₂|Θ|`, which is a principled way to set `T` rather than guessing.

### 2.5 WITHDRAWN — ignore any earlier request about a transcript encoding

An earlier version of this brief asked you to decide how a tool call should be tokenized, whether a tool result counts as an oracle echo for masking, and whether error responses are supervised — so that the repertoire could add a "deployment format" encoding matching an agentic harness.

**That request is withdrawn.** It was an overfit to a transient convention, and it contradicted A3, whose whole purpose is that a family be *invariant* under its encoding set rather than matched to any particular surface. Building toward a deployment format would have weakened the invariance that makes format-independence possible in the first place. Recorded as hazard 25.

**Nothing is asked of you here.** Keep the loss mask as §7 specifies it: family-rendered answers supervised, queries supervised only at L2 against `q*`, preamble and oracle-echo tokens never supervised. No special cases.

### 2.6 Two axes the parametrization is missing — do not build them yet

`docs/06` found three of ten established paradigms do not fall out of the formalism, failing in exactly two ways: **θ is assumed sampled once and to hold** (violated by piecewise-stationary and drifting rules), and **L2 assumes querying is free** (violated wherever the query is the scored action).

Both repairs are small — a validity-duration declaration on `P_Θ`, and a per-family flag on whether the query channel is scored, which is a loss-mask change. Note that under §2.5 above **both are the normal case in a harness**, not edge cases: issuing a tool call *is* the action and has cost, and the world changes under a long-running agent. That strengthens the eventual case for them considerably. **Still deliberately deferred until after your sweep**, because §8 step 5 expects the four named levels to be cut in the wrong places, and adding axes before it runs would be designing the answer. They are pre-registered in `docs/02` §5 so they can be *scored against* the sweep. If the sweep relocates the cuts, come back to them.

---

## 3. Traps, from `docs/07-hazards.md`

Read that file. The ones that will bite a harness specifically:

- **A check that cannot fail is worse than no check.** Met twice already. When you write the A2 or L3 verification, also write the deliberately-broken case that must fail it. A plant passing its own check proves nothing unless failure is reachable.
- **Being in the vocabulary is not being in the alphabet.** A family once rendered content as `PAD`/`BOS` tokens; every id was valid, the in-vocabulary test passed, and **A2 passed too** — equivariance under the wrong alphabet is still equivariance.
- **A family can satisfy its type signature and violate its purpose.** The calibration exemplar silently degraded into L1 because its rate grid included the deterministic endpoints. Only an assertion about the *property* caught it.
- **Watch for turnover, not just level.** `docs/08` established A4 cannot be certified at our scale, only measured. Brute-force collapse appears as a **sign change** in structural content as compute grows, which reads as noise unless you are watching for it.
- **The loss-dashboard inversion.** Families with more learnable structure carry *higher absolute loss*. A controller doing the right thing looks wrong. Task Spec §9 calls this "the easiest way for the work to be abandoned for the wrong reason."

---

## 4. What already exists — use it, do not rebuild it

| Path | What it is |
|---|---|
| `src/repertoire/form.py` | The §7 `TaskFamily` protocol, `@runtime_checkable`. **Shared file** — see §5 |
| `src/repertoire/vocab.py` | One shared vocabulary, fixed before any family. Provisional and yours to adopt or replace — but families must not grow private vocabularies |
| `src/repertoire/families/` | **9 implemented families**, 101 tests passing. Junk plants (both), SHJ Type I / Type VI, conjunction, Bruner conjunction, parity, permuted bits, probability matching |
| `src/repertoire/compose.py` | §1.1 composition: n-ary, two gates (types **and** semantic coherence), exact product posterior, closure report |
| `src/repertoire/expectations.py` | **Pre-committed** §11 step 6 gate as executable assertions. Feed it a matrix; do not edit the thresholds |
| `src/repertoire/a3_test.py` | A3 encoding-leak test. Swap in prequential structural content for the token-length proxy once you have it — that is task #9 |
| `register/rows/*.toml` | 26 families translated into §7 fields with A1–A7 triage. Read the row before implementing a family |
| `docs/07-hazards.md` | 22 hazards, ranked by how likely they are to pass unnoticed |

**Step 4's planted basis is effectively ready.** All six plant roles exist: junk-random, junk-trivial, the prerequisite pair (SHJ I/VI, externally attested and replicated), the constructed near-duplicate pair (conjunction / Bruner), the independent pair (SHJ I / probability matching), and one suspected-junk row. There is also an **unplanted** near-duplicate — parity and SHJ Type VI are the same object, excavated independently from computational learning theory and 1961 categorization psychology, asserted in a test.

---

## 5. Not colliding with me

I am working in `register/`, `docs/00`–`docs/08`, and `notes/`. Please treat those as mine.

**Yours:** a new `src/repertoire/harness/` package, `tests/test_harness*.py`, and `docs/10-harness-findings.md` (create it — I will read it).

**Shared, and the only real conflict risk:** `src/repertoire/form.py`, `src/repertoire/vocab.py`, and `src/repertoire/families/`. You will probably need to change `form.py` (§2.1 above). That is expected and fine — **record every protocol change in `docs/10-harness-findings.md`** with the reason, and I will bring the families into line. Do not silently edit families to match a changed protocol; tell me and I will do it, because several of them carry plant properties that a mechanical edit would break.

Everything is in git. Commit often, small commits, and say in the message which gate you are working toward.

---

## 6. The sweep itself — what makes it a result

Task Spec §8 step 5: vary residual entropy **continuously**, not as four discrete levels. Suggested parameters: how many observations precede the query, or what fraction of θ is stated in the preamble. Measure structural content and transfer **as a curve** in that parameter.

Why continuous rather than a four-way comparison: a discrete test tests whether *those four cells* were well chosen. A collapse at one hand-picked construction is ambiguous between "nothing above L0 works" and "that construction was badly built." The sweep separates them, and it *locates* the cuts rather than assuming them.

| Curve shape | Reading | Action |
|---|---|---|
| **Monotone collapse** | Supervision requires determinate targets; everything above L0 needs reward | **Stop.** §2–§9 are void, and that is a real result worth writing up |
| **Interior peak at low-but-nonzero entropy** | The programme works; the four named levels are cut in the wrong places | Continue; relocate the cuts from the curve |
| **Flat or rising** | Something unexpected; the ambiguity model is wrong | Investigate before proceeding |

**Hold compute budget fixed across every comparison and report it.** Structural content is budget-relative and comparing across budgets is meaningless (Task Spec §4).

**Also log the within-episode acquisition slope separately.** Per-trial loss falling within an L1/L2 episode as θ becomes identified is the §9 primary metric, available as a training-time scalar with no downstream evaluation. It has a 1949 precedent — Harlow's learning-set curve is the same quantity, measured on macaques, and `register/rows/harlow-learning-set.toml` records why that family is a *protocol* rather than a family.

**A request:** if the sweep collapses, say so plainly and early. It is the most valuable single result available here, it is a real finding rather than a failure, and everything I am currently building is downstream of it.
