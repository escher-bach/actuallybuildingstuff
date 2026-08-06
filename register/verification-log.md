# Verification Log

*Every source promoted to `verified = true` in a register row must be independently checked by the curator, not merely labelled VERIFIED by the subagent that found it. This file records each check and its outcome.*

**Why this exists.** Delegated literature review returns reports that look uniformly confident. The first spot-check pass (2026-08-06) found the citations were overwhelmingly real and structurally accurate — and that **specific numeric claims drift, in one case into outright fabrication**. A fabricated statistic in a register row is worse than a missing one: it propagates into an A-check verdict, which propagates into a build decision, and nothing downstream flags it.

**The rule, from now on:** a subagent's `VERIFIED` label promotes a source to *candidate-verified*. It becomes `verified = true` in a row only after a curator check recorded below. Structural claims (this paper exists, it is about X) have proven reliable; **numbers, counts, and percentages have not, and are the thing to check.**

---

## Pass 1 — 2026-08-06, sources promoted from veins §2.2 and §2.5

| Source | Claim checked | Outcome |
|---|---|---|
| arXiv:2011.10787 — Failed Error Propagation in Java | "26.5% of fault-test pairs exhibit FEP" | ❌ **FABRICATED.** The figure does not appear in the paper. Actual: **0%** unit-level FEP with real faults (across 258,372 executions), **11.4%** system-level (60 of ~526 executions, 4 bugs), and 1.6% internal / 3.7% external for mutants. The paper's own framing is that unit-level FEP prevalence is *negligible* — which points the opposite way from how the number was used |
| pitest.org mutator catalogue | "11 default + 18 optional" | ⚠️ **Partly wrong.** 11 default confirmed exactly, including the full list. But optional is **8**, plus **7 experimental** — the report merged two groups and overcounted |
| arXiv:2506.17114 — Mathematical Proof as a Litmus Test | "11-category error taxonomy" | ⚠️ **Off by one.** Paper reports **10** fine-grained error types over a 200-problem dataset. The 11th name in the report's list ("Others") appears to have been added |
| arXiv:1901.07750 — Adaptive Exact Learning of Decision Trees | Randomized Õ(2^2d) + 2^d log n; deterministic 2^5.83d + 2^{2d+o(d)} log n | ✅ **Exact**, including the improvement-over-Feldman and Kushilevitz–Mansour framing |
| arXiv:2507.20336 — Faster exact learning of k-term DNFs | poly(n)·2^Õ(√k); "first improvement since Blum–Rudich 1992" | ✅ **Exact**, including the direct quote |
| arXiv:2410.15748 — Alchemy | Symbolic mutation yields *valid* theorems; Mathlib 110k → 6M | ✅ **Exact** |
| arXiv:1405.0792 — Exact Learning Monotone DNF | MQ learning of s-term, size-r monotone DNF; "almost optimal", tight for fixed r and/or s | ✅ **Confirmed** (abstract states it qualitatively, not as a closed-form bound — the row says the same) |
| arXiv:2209.09819 — Efficient Model Based Diagnosis | Title, probing-point selection, O(n²) worst case | ✅ **Confirmed.** ⚠️ But the GDE-entropy quote the report relies on is *not* in the abstract; it was attributed to the body via an HTML mirror and remains **unconfirmed by this pass** |
| Dasgupta, "Analysis of a greedy active learning strategy" | Title, authorship, greedy-within-a-log-factor result | ✅ **Confirmed** from the PDF: Sanjoy Dasgupta, UCSD, Jan 2005, NeurIPS 2004 summary. Abstract: the greedy rule is "approximately as good as any other strategy". ⚠️ The **per-query cost "linear in version-space size"** claim was not located in the pages read and is now recorded as unconfirmed |

| Golovin & Krause, arXiv:1003.3967 — Adaptive Submodularity | Title, adaptive submodularity, greedy competitive with optimal | ✅ **Confirmed.** ⚠️ The applications the review attributed to it — equivalence class determination, optimal decision tree — are **not in the abstract**, which lists sensor placement, viral marketing and active learning. The review did hedge this as snippet-corroborated. May be in the body; **unconfirmed** |
| Goldman & Kearns, "On the Complexity of Teaching" | Title, authors, and the teaching-dimension definition | ✅ **Exact**, near-verbatim on the definition |
| mCoq, github.com/EngineeringSoftware/mcoq | live/killed definitions; mutates definitions not proof scripts | ✅ **Exact** on both, including that it mutates *definitions* rather than proof scripts — which is the distinction the proof row depends on |

**Score across the pass: 8 clean, 4 with drift or unconfirmed detail, 1 fabricated.** The failure mode is specific and predictable — real paper, real topic, invented or mangled number — and it is exactly the kind of error that survives a plausibility read.

**Not independently checked, and flagged as such:** Reiter 1987 and de Kleer & Williams 1987. Both are canonical, both were reported as partially readable by the reviewing agent, and the claims drawn from them (hitting-set characterization of diagnoses; exponential worst case; GDE's entropy-based probe selection) are textbook-standard rather than novel-sounding. The GDE entropy quote in particular is load-bearing for the `circuit-fault-localization` A7 discussion and rests on a secondary source's quotation of the body. **Check before that row is promoted past `translated`.**

### Boundary call, recorded rather than buried

Dasgupta 2004/2005 self-describes as **pool-based** active learning, and "pool-based active learning for labels in ML" is on the held-out list. The seal's own wording carves out "classical *theory* of query learning and of optimal experimental design", which this is — a theoretical query-complexity analysis, not applied label-efficiency work. The reviewing agent flagged the same boundary independently and stopped short of the applied literature. **Judged in-scope, not a contamination.** Recorded here and in `docs/00-heldout-partition.md` §4 because a boundary call that is defensible is still a call, and the ones that go unrecorded are the ones that erode a seal.

---

## Pass 2 — 2026-08-06, sources from vein §2.3

The §2.3 review was notably more careful than §2.5: it marked most of its own sources LEAD rather than VERIFIED (the psychology primaries are largely paywalled), refused to promote the near-duplicate plant past MODERATE evidence, and **caught an internal inconsistency in one of its own sources** — flagging that a paper's quoted bit-pattern table did not cross-check against a canonical XOR construction, and telling us to distrust those specific patterns while keeping the qualitative claims. That is the behaviour this log exists to encourage.

| Source | Claim checked | Outcome |
|---|---|---|
| Nosofsky, Gluck, Palmeri, McKinley (+ Glauthier), *Memory & Cognition* 22(3):352–369, 1994 | That it replicates the Shepard–Hovland–Jenkins difficulty ordering | ✅ **Confirmed** from the PubMed record. Exact title "Comparing models of rule-based classification learning: a replication and extension of Shepard, Hovland, and Jenkins (1961)", and the abstract states "our main results mirrored those of Shepard et al., with the ordering of task difficulty being the same as in the original study". Note the paper has **five** authors — the review listed four. The ordering itself (VI hardest, III=IV=V equal, then II, then I easiest) is not in the PubMed abstract; it is corroborated across the Springer listing and the `catlearn` R package, which ships the `nosof94` dataset |
| Miconi, "Procedural generation of meta-reinforcement learning tasks", arXiv:2302.05583 | Parametrized meta-RL task space covering bandits, Harlow, T-mazes, Daw two-step | ✅ **Exact**, including the list of special cases, near-verbatim |

**The prerequisite plant stands.** SHJ Type I vs Type VI is externally attested and replicated, which is what §5 asks of a planted ordering. The review's own caveat is retained and is important: this is *empirical acquisition-speed precedence*, not logical containment. Nothing shows Type VI's rule contains Type I's as a substructure. "A required for B" in the plant table should be read as "A is reliably mastered before B", and the plant tests whether our transfer measurement recovers an ordering that humans reliably show — not whether it recovers a logical nesting.

**Still LEAD, and the vein's main limitation:** Shepard/Hovland/Jenkins 1961, Harlow 1949, Feldman 2000, Berg 1948, and Bruner chs. 4–6 are all paywalled or unreachable. The §2.3 rows are correspondingly held at `lead`. This is a library-access problem, not a literature gap, and it is the same blocker already logged for Angluin 1987/1988.

---

## Consequences applied

- `mutant-localization.toml` — the 26.5% claim removed and replaced with the paper's actual figures. **This materially weakens the row's A1 argument** and the correction is recorded in the row itself, not silently swapped: unit-level FEP being ~0% means the "corruption might be invisible" hazard is much smaller in the regime we would actually train in.
- `proof-step-localization.toml` — 11 → 10 categories.
- `generic-finite-theta-versionspace.toml` — the O(|version space|) per-query cost marked as unconfirmed.
- `notes/vein-2.5-corrupt-localize-report.md` — correction header added; the report is left otherwise intact as the agent's artifact.
