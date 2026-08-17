# Learner-Conditioned Dense Supervision

## Stage report: STEP-1 §6's third condition, on the existing world family

### Status

This reports the first runs of the learner-conditioned regime specified in
[STEP-1.md](STEP-1.md) §5.3 and required as the third row of §6's comparison
table. It tests the predictions [THEORY-PHASE.md](THEORY-PHASE.md) §8 registered
before the stage was implemented: **P1**, that supervision at learner-reached
states changes the learned policy, and the on-policy half of **P3**, that
protocol failures become less persistent.

Two arms ran, both at commit `e50b566`, both on two T4s:

| arm | start | run id |
|---|---|---|
| `lc-cold-seed0` | random weights | `t4x2-lc-cold-seed0-e50b5668da2b-a86b325bfd6c` |
| `lc-warm-seed0` | dense seed-0 checkpoint `cef0ac5a…` | `t4x2-lc-warm-seed0-e50b5668da2b-c46717f07ca4` |

All results are single-seed. **There is no matched teacher-conditioned
continuation arm**, so §6 below is a list of what this stage cannot separate,
not a caveat added at the end.

---

## Summary

1. **The dense policy commits before the evidence identifies the answer 73% of
   the time**, with a mean of 2.00 hypotheses still live at the moment it
   commits. This is the first time the quantity has been measured. It makes
   THEORY-PHASE §3.1's mechanism concrete: a map fitted to the teacher's
   trajectory distribution has no reason to represent "the evidence does not yet
   determine the answer", because the teacher never occupies such a state and
   commits anyway.

2. **The 41% band is very nearly "probe until two hypotheses remain, then
   guess".** Legal-conditioned success is 45.2% against a mean of 2.004 live
   hypotheses at commitment; guessing uniformly among those would give 49.9%.

3. **P1's mechanism moved exactly as predicted, and capability fell anyway.**
   Premature commitment 73.0% → 46.5%, probes 1.97 → 2.47, hypotheses live at
   commitment 2.00 → 1.56 — every one in the predicted direction. Held-out
   success went 43.8% → 24.8%.

4. **The cause of the fall is the action interface, not the decisions.**
   Protocol failure went 3.1% → 29.0%. Legal-conditioned success fell far less
   than raw success did (45.2% → 34.9%).

5. **From a weight-naive start, learner-conditioned supervision crosses the byte
   grammar where outcome-only RLVR could not.** The cold arm reached 11.5%
   closed-loop success in 512 updates. Cold RLVR, at a comparable budget,
   produced exactly zero gradient because every rollout was unparseable. A
   teacher labels the state a malformed attempt leaves unchanged, so this arm
   had a gradient from its first update.

6. **The cold arm learned to speak one round before it learned what was legal.**
   Malformed attempts collapsed from 8,192 to 18 across a single round of 128
   updates, while invalid attempts rose from 0 to 8,174 in the same round.

7. **Conditioned on legal play, the cold arm's decisions are at chance** — 17.2%
   against a 16.8% floor, with 48% premature commitment. It acquired the
   interface and not the process.

8. **The truth-leak guard refused 1,754 targets in the warm arm and 3,525 in the
   cold one**, growing monotonically as each policy began reaching
   badly-spent states. Untreated, these would have supervised commits derived
   from `inst.truth` at states where no visible evidence supports them.

9. **P3 does not hold as stated.** Attempt failure rate fell (0.603 → 0.250
   across the warm arm's trained milestones) but recovery rate did not improve
   monotonically, and the budget-0 comparison is unusable: that policy fails so
   rarely that its recovery rate is computed over four events.

10. The interpretation in §6 that most needs testing is that the damage comes
    from training on contexts containing the learner's own illegal actions —
    contexts the evaluator never presents, because it ends an episode at the
    first protocol failure.

---

## 1. What was measured, and why the frozen metrics were not enough

THEORY-PHASE §7's rule: *a raw success rate is not a measurement of decision
quality unless protocol failure is separated out.* This stage reports, on every
evaluation set and in its own report only:

| measurement | what it separates |
|---|---|
| success conditioned on legal play | decision quality, with the interface term removed |
| protocol failure rate | the interface term itself |
| premature commitment rate | committing while the evidence licenses no commitment |
| mean hypotheses live at commitment | how far from identified those commitments were |
| probe count and its distribution | evidence gathering, the process the world was built to teach |
| recovery probe | whether a failed action stays failed (P3) |

`evaluate.py`'s frozen metric set and every existing contract are untouched, so
every number here is comparable with every number already retained. The extended
read comes from privileged *row* fields added behind a default-off flag.

Two of these needed no new run at all. Because an episode leaves the evaluator
at its first protocol failure, malformed and invalid are disjoint per episode,
and legal-conditioned success is exactly `success / (1 − malformed − invalid)`.
Every arm already run can therefore be restated. On the retained validation
payload:

| model | decoding | raw | conditioned on legal play |
|---|---|---:|---:|
| dense seed 0 | greedy | 41.1% | 42.3% |
| dense seed 0 | sampled t=1.0 | 36.9% / 34.8% | 44.2% / 42.1% |
| RLVR best shot | greedy | 41.2% | 41.9% |
| RLVR best shot | sampled t=1.0 | 42.2% / 42.0% | 43.7% / 43.7% |

The tuned RLVR arm's +6-point sampled gain is entirely the removal of illegal
actions. Conditioned on legal play it is flat at 42–44% across all six cells.
[RLVR-STAGE-REPORT.md](RLVR-STAGE-REPORT.md) states this qualitatively at its
point 10; this is the number.

---

## 2. The demonstrator is not a policy the learner can imitate

A CPU replay audit ([`step1/audit/world/`](step1/audit/world/)) ran before the
stage and changed what it was testing. It overturns the arithmetic
THEORY-PHASE §2 builds on.

| quantity | value |
|---|---|
| hypotheses live at the opening observation | 6.0, in every one of 1,024 episodes |
| teacher success | 100% |
| teacher mean probes | **2.07** (954 episodes at two, 70 at three) |
| dense model mean probes | 1.94 |
| every episode identifiable within budget | yes — truth-blind ceiling 1.0 |
| fixed truth-blind order, buying everything affordable | identifies **60.7%** |

§2 reasoned that six hypotheses under binary evidence require ⌈log₂ 6⌉ = 3
probes, so 1.94 meant the model stops before the evidence can identify the
answer, with a ceiling near 67%. Two binary probes cannot separate six
hypotheses — and the teacher does not have to. `teach` reads `inst.truth` and
needs only the true hypothesis to land in a cell of its own.

So the model is not under-probing. It matches its demonstrator's probe count
almost exactly and scores 41.1% against 100%. **The gap is which probes, not how
many.** The world is not the obstacle: every episode is identifiable inside the
budget. But the budget is tight enough that choosing wrong forecloses the
answer, which is why a fixed truth-blind order fails 39% of the time.

This is a sharper form of §3's fitted-map account than §3 states. The teacher's
trajectory distribution is *itself truth-conditioned*, so cloning it copies a
cost profile whose sufficiency came from information the learner does not have.

### 2.1 The hazard this creates, which learner conditioning creates too

Walked into states a weak policy reaches, the teacher proposes a `Commit` at
**18.6%** of states where the evidence licenses none, and in two thirds of those
it is the only preferred action — so `min(preferred_actions)` would supervise
it. That target is derived from `inst.truth`, and STEP-1 §5.2 forbids
supervising a hidden variable the generator merely happens to have.

On its own trajectory the teacher never does this: over 1,024 episodes its
maximum hypotheses-live-at-commitment is 1. That is why the existing dense
shards are clean and why no teacher-conditioned check could have found this.
The collector refuses those targets and counts them. In the runs it mattered:

| arm | round 0 | round 1 | round 2 | round 3 | total |
|---|---:|---:|---:|---:|---:|
| warm | 249 | 297 | 546 | 662 | **1,754** |
| cold | 0 | 20 | 1,036 | 2,469 | **3,525** |

The count grows as each policy starts reaching states where the budget has been
spent without identifying anything. Left in, it would have trained the model to
commit on evidence that does not determine the answer — the exact behaviour P1
predicts this arm should reduce.

---

## 3. The warm arm: P1's mechanism moves, capability falls

512 updates over 8,192 on-policy episodes, from the dense seed-0 checkpoint.
`budget-0` is that checkpoint scored under this stage's own instrument; its
teacher-forced NLL of 0.0853 reproduces the dense arm's retained value exactly,
which is the instrument check.

| updates | success | protocol failure | legal-cond. | commit rate | premature | probes | live@commit | NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 (dense) | **43.8%** | 0.031 | 45.2% | 96.9% | **73.0%** | 1.97 | 2.00 | 0.0853 |
| 128 | 4.9% | 0.830 | 28.7% | 17.0% | 66.7% | 1.99 | 1.95 | 0.1445 |
| 256 | 25.4% | 0.289 | 35.7% | 71.1% | 49.2% | 2.40 | 1.59 | 0.1121 |
| 384 | 31.1% | 0.359 | 48.5% | 64.1% | 43.3% | 2.60 | 1.55 | 0.1232 |
| 512 | 24.8% | 0.290 | 34.9% | 71.0% | 46.5% | 2.47 | 1.56 | 0.1195 |

Read the premature-commitment column and the probe column: both move in exactly
the direction P1 predicted, monotonically after the first round, and by a large
margin. Read the success column: the policy is worse at the end than at the
start, and never recovers the baseline.

The two are reconciled by the protocol-failure column. The decisions improved
and the interface collapsed, and raw success sums them.

**The first round is where the damage happens.** 128 updates take success from
43.8% to 4.9% and commit rate from 96.9% to 17.0%. Rounds 2–4 recover most of
the interface and continue improving the decision metrics, but not back to
baseline.

### 3.1 Held-out sets at the endpoint

| set | success | protocol failure | legal-cond. | premature | probes |
|---|---:|---:|---:|---:|---:|
| validation | 24.8% | 0.290 | 34.9% | 46.5% | 2.47 |
| structural (`n_hyp`+1) | 3.1% | 0.852 | 21.1% | 54.0% | 1.75 |
| reversible control | 7.0% | 0.490 | 13.8% | 44.8% | 2.53 |
| Rendering B | 0.0% | 1.000 | — | — | 0.00 |

Rendering B remains ungrounded, as in every prior arm.

---

## 4. The cold arm: the interface is crossable by supervision alone

512 updates over 8,192 on-policy episodes, from random weights.

| updates | success | protocol failure | legal-cond. | commit rate | premature | probes | NLL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0% | 1.000 | — | 0.0% | — | 0.00 | 5.697 |
| 128 | 0.0% | 1.000 | — | 0.0% | — | 1.00 | 2.981 |
| 256 | 0.0% | 1.000 | — | 0.0% | — | 1.98 | 3.158 |
| 384 | 3.3% | 0.885 | 28.8% | 11.5% | 20.3% | 2.63 | 0.191 |
| 512 | **11.5%** | 0.328 | 17.2% | 67.2% | 48.3% | 2.47 | 0.141 |

The collection stream shows the acquisition order directly:

| round | accepted | malformed | invalid | terminated |
|---|---:|---:|---:|---:|
| 0 | 0 | 8,192 | 0 | 0 |
| 1 | 2,048 | 18 | 8,174 | 0 |
| 2 | 4,037 | 5,258 | 2,935 | 0 |
| 3 | 5,844 | 4,906 | 2,346 | 298 |

Round 1 is the result. **The action grammar is acquired within one round of 128
updates** — malformed attempts fall from 8,192 to 18 — and the failures
immediately move to well-formed-but-illegal. Malformed rises again in rounds 2–3
as accepted actions carry the policy into deeper states it has not seen.

This is the contrast the arm was declared against. Outcome-only RL from the same
initialization produced *no gradient at all*: every rollout died unparseable,
every group's rewards were identical, and the parameters were byte-identical
before and after. The difference is not the budget and not the world. A verifier
scores a trajectory that has to exist first; a teacher labels the state an
unparseable attempt leaves unchanged.

But conditioned on legal play the cold arm sits at 17.2% against a 16.8% chance
floor. It learned to speak and not to decide, and 48% premature commitment is
where that shows.

---

## 5. P3: recovery

The frozen evaluator cannot measure persistence — it ends an episode at the
first protocol failure. The recovery probe reuses the collector as an
instrument on a fixed held-out block that is never trained on.

| warm arm | attempts | failed | failure rate | recovered | recovery rate |
|---:|---:|---:|---:|---:|---:|
| 0 | 773 | 10 | 0.013 | 2 | 0.500 |
| 128 | 1,413 | 852 | 0.603 | 12 | 0.055 |
| 256 | 1,100 | 299 | 0.272 | 15 | 0.185 |
| 384 | 1,202 | 310 | 0.258 | 36 | 0.375 |
| 512 | 1,130 | 283 | 0.250 | 17 | 0.212 |

**P3 does not hold as stated.** The budget-0 comparison is unusable: the dense
policy fails so rarely that its 0.500 is two recoveries out of four events. Among
the trained milestones the failure rate falls steadily while the recovery rate
rises then falls, which is not a trend. The prediction that failures become less
*persistent* is not supported; what improved is that there are fewer of them.

---

## 6. What this stage cannot separate

Stated as a list because each item is a live alternative to the reading above,
not a hedge.

**No matched teacher-conditioned continuation.** Both arms are compared to their
own starting policy under an identical instrument, which matches the measurement
and not the training. Every number in §3 is consistent with "learner-state
supervision did this" and with "512 more supervised updates at 1e-4 did this".
This is the single most valuable missing run and it is one config change: the
same 8,192 world instances, the same update budget, teacher-conditioned states.

**On-policy contexts contain the learner's own illegal actions.** STANDARD-LLM-
STACK-MIGRATION-PLAN §7.3 requires the learner's action to remain in context, so
from round 1 onwards the model is trained on histories containing thousands of
its own invalid attempts. The evaluator never presents such a context — it ends
the episode at the first failure. This is a train/test context mismatch created
by the regime itself, and it is the leading candidate for why the interface
degraded while the decisions improved. It is testable without a new idea: score
the same checkpoints on a retry-tolerant rollout, or collect with failed
attempts excised from the running context.

**The first round dominates.** 128 updates account for essentially all of the
damage. Whether that is the learning rate, the format of the first tranche
(collected from a policy that had never seen an on-policy context), or the
regime itself is not separated here.

**Single seed, one world family, one step size.** The warm arm ran at 1e-4
against the dense recipe's 6e-4, chosen so an optimization shock could not
masquerade as the effect. That choice is not itself controlled.

---

## 7. What follows

In the order their information value justifies:

1. **The matched teacher-conditioned continuation arm.** Without it §3 has two
   readings and no way to choose. Cheapest run available.
2. **A retry-tolerant evaluation** of the existing checkpoints, which tests the
   context-mismatch explanation in §6 and needs no training.
3. **Premature commitment on the other retained checkpoints.** The metric is new,
   so the RLVR arms and the transfer endpoints have never been read this way, and
   the instrument now exists. Scoring-only, and it would show whether 73% is a
   property of the dense arm or of every policy this project has trained.

The result that does not need any of them: **the dense policy commits with two
hypotheses live, three times in four.** That was invisible for the whole step
because success rate summed it with everything else.
