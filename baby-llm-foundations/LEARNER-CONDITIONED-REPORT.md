# Learner-Conditioned Dense Supervision

## Stage report: STEP-1 §6's third condition, on the existing world family

### Status

This reports the learner-conditioned regime specified in [STEP-1.md](STEP-1.md)
§5.3 and required as the third row of §6's comparison table. It tests the
predictions [THEORY-PHASE.md](THEORY-PHASE.md) §8 registered before the stage
was implemented: **P1**, that supervision at learner-reached states changes the
learned policy, and the on-policy half of **P3**.

Four runs, all single-seed, all on two T4s:

| run | start | step | states | run id |
|---|---|---|---|---|
| `lc-cold-seed0` | random | 6e-4 | learner | `t4x2-lc-cold-seed0-e50b5668da2b-a86b325bfd6c` |
| ~~`lc-warm-seed0`~~ | dense seed 0 | 1e-4 | learner | **retracted, see §0** |
| `lc-warm-lr2e5-seed0` | dense seed 0 | 2e-5 | learner | `t4x2-lc-warm-lr2e5-seed0-8557c2b801d3-134d7b9c5f7d` |
| `tc-control-lr2e5-seed0` | dense seed 0 | 2e-5 | **teacher** | `t4x2-tc-control-lr2e5-seed0-8557c2b801d3-f066b26eef05` |

The control differs from the arm in exactly one field, `collection.policy`, and
a contract test enforces that. Same packing, masking, seeds, world instances,
budget, optimizer and step size; the world advances on the teacher's action
instead of the model's.

---

## 0. Retraction

The first warm arm ran at learning rate **1e-4**. That was a mistake with
evidence already in this repository: [RLVR-STAGE-REPORT.md](RLVR-STAGE-REPORT.md)
§4.5 had tested 1e-4 on this exact checkpoint and reported it as damage — NLL
0.0853 → 0.1481, success 41.1% → 31.8%, "moved it ten times further and made it
worse". The learner-conditioned run reproduced that displacement (0.0853 →
0.1445 after one round). Its config comment claimed 1e-4 was chosen *to avoid*
an optimization shock.

Its conclusions are withdrawn and replaced by §3. The rerun at 2e-5 changes the
endpoint materially (24.8% → 33.0% success) but, as §3.2 shows, **the
first-round interface collapse reproduces at 2e-5 and is therefore not a step
size artefact.** The initial diagnosis was half right: the step size inflated
the damage; it did not create it.

What was never affected by this: the CPU audit in §2, which trains nothing; the
cold arm, which runs at the dense recipe's own rate from random weights; and the
`budget-0` measurement of the dense policy, which is a pure scoring of an
existing checkpoint.

---

## Summary

1. **The dense policy commits before the evidence identifies the answer 73.0% of
   the time**, with a mean of 2.00 hypotheses still live. First measurement of
   the quantity. It makes THEORY-PHASE §3.1's mechanism concrete.

2. **P1's mechanism is confirmed, and it is attributable to the state
   distribution.** Learner-conditioned supervision moves premature commitment
   73.0% → 42.5% and probes 1.97 → 2.59. The matched teacher-conditioned control
   — same pipeline, same everything but whose action advances the world — moves
   neither: 72.5% and 1.97, both within noise of the untrained checkpoint.

3. **P1's capability clause is not confirmed.** §8 predicted premature
   commitment would fall *and* legal-conditioned success would improve. It
   didn't: 45.2% → 45.6%, flat. The policy gathers more evidence and commits
   later, and is no more accurate for it.

4. **§2 explains why that is coherent rather than contradictory.** The teacher
   identifies in 2.07 probes because it selects them using `inst.truth`. More
   probes is not the missing ingredient; the *right* probes are. The arm moved
   the disposition and not the selection rule, and hypotheses live at commitment
   fell only 2.00 → 1.51, never to 1.

5. **The interface cost is real and is caused by learner conditioning, not by
   the step size.** Protocol failure 3.1% → 27.5% in the arm; 3.1% → 3.3% in the
   control. Raw success falls 43.8% → 33.0% almost entirely through that term.

6. **From random weights, learner-conditioned supervision crosses the byte
   grammar where outcome-only RLVR could not** — 11.5% closed-loop, against cold
   RLVR's exactly zero gradient. The grammar is acquired inside one round of 128
   updates.

7. **The truth-leak guard refused 1,835 targets in the arm and 3,525 in the cold
   run, and exactly 0 in the control** — the teacher never commits from a state
   its own history has not identified.

8. **P3 does not hold as stated.** Failures got rarer, not less persistent.

---

## 1. What was measured

THEORY-PHASE §7's rule: *a raw success rate is not a measurement of decision
quality unless protocol failure is separated out.* This stage adds, in its own
report only, legal-conditioned success, protocol failure rate, premature
commitment rate, hypotheses live at commitment, the probe distribution, and a
recovery probe. `evaluate.py`'s frozen metric set and every existing contract
are untouched, so every number here is comparable with every number retained.

Two of these needed no run at all. An episode leaves the evaluator at its first
protocol failure, so malformed and invalid are disjoint per episode and
legal-conditioned success is exactly `success / (1 − malformed − invalid)`. On
the retained validation payload:

| model | decoding | raw | conditioned on legal play |
|---|---|---:|---:|
| dense seed 0 | greedy | 41.1% | 42.3% |
| dense seed 0 | sampled t=1.0 | 36.9% / 34.8% | 44.2% / 42.1% |
| RLVR best shot | greedy | 41.2% | 41.9% |
| RLVR best shot | sampled t=1.0 | 42.2% / 42.0% | 43.7% / 43.7% |

The tuned RLVR arm's +6-point sampled gain is entirely the removal of illegal
actions; conditioned on legal play it is flat at 42–44% across all six cells.
RLVR-STAGE-REPORT §10 states this qualitatively. This is the number.

---

## 2. The demonstrator is not a policy the learner can imitate

A CPU replay audit ([`step1/audit/world/`](step1/audit/world/)) ran before the
arms and changed what they were testing.

| quantity | value |
|---|---|
| hypotheses live at the opening observation | 6.0, in every one of 1,024 episodes |
| teacher success | 100% |
| teacher mean probes | **2.07** (954 episodes at two, 70 at three) |
| dense model mean probes | 1.94 |
| every episode identifiable within budget | yes — truth-blind ceiling 1.0 |
| fixed truth-blind order buying everything affordable | identifies **60.7%** |

THEORY-PHASE §2 reasoned that six hypotheses under binary evidence need
⌈log₂ 6⌉ = 3 probes, so 1.94 meant the model stops before the evidence *can*
identify the answer, with a ceiling near 67%. Two binary probes cannot separate
six hypotheses — and the teacher does not have to. `teach` reads `inst.truth`
and needs only the true hypothesis to land in a cell of its own.

**So the model is not under-probing.** It matches its demonstrator's probe count
almost exactly and scores 41.1% against 100%. The gap is which probes, not how
many. The world is not the obstacle — every episode is identifiable inside the
budget — but the budget is tight enough that choosing wrong forecloses the
answer, which is why a fixed truth-blind order fails 39% of the time.

This is a sharper form of §3's fitted-map account than §3 states: the teacher's
trajectory distribution is *itself truth-conditioned*, so cloning it copies a
cost profile whose sufficiency came from information the learner does not have.

### 2.1 A hazard learner conditioning creates

Walked into states a weak policy reaches, the teacher proposes a `Commit` at
**18.6%** of states where the evidence licenses none, and in two thirds of those
it is the only preferred action — so `min(preferred_actions)` would supervise
it. That target is derived from `inst.truth`, and STEP-1 §5.2 forbids
supervising a hidden variable the generator merely happens to have.

On its own trajectory the teacher never does this: over 1,024 episodes its
maximum hypotheses-live-at-commitment is 1. The control confirms it end to end —
**0 refusals across all four rounds** — while the arm refused 1,835 and the cold
run 3,525, both growing as the policy began reaching badly-spent states. This
hazard is invisible to every teacher-conditioned check and is created by the
regime.

---

## 3. The controlled comparison

Both start from dense seed 0, both 512 updates over the same 8,192 world
instances at 2e-5. `budget-0` is the dense checkpoint under this stage's
instrument; its NLL of 0.0853 reproduces the retained dense value exactly, which
is the instrument check.

**Learner-conditioned arm**

| updates | success | proto-fail | legal-cond. | premature | probes | live@commit |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 43.8% | 0.031 | 45.2% | 73.0% | 1.97 | 2.00 |
| 128 | 7.2% | 0.820 | 40.2% | 66.3% | 1.98 | 1.95 |
| 256 | 29.3% | 0.365 | 46.2% | 40.6% | 2.57 | 1.51 |
| 384 | 36.3% | 0.242 | 47.9% | 40.5% | 2.65 | 1.49 |
| 512 | 33.0% | 0.275 | 45.6% | **42.5%** | **2.59** | **1.51** |

**Teacher-conditioned control**

| updates | success | proto-fail | legal-cond. | premature | probes | live@commit |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 43.8% | 0.031 | 45.2% | 73.0% | 1.97 | 2.00 |
| 128 | 43.8% | 0.031 | 45.2% | 70.0% | 1.97 | 1.97 |
| 256 | 42.0% | 0.023 | 43.0% | 75.2% | 1.98 | 1.99 |
| 384 | 44.0% | 0.018 | 44.7% | 70.4% | 1.98 | 1.97 |
| 512 | 41.6% | 0.033 | 43.0% | **72.5%** | **1.97** | **1.97** |

### 3.1 What the control establishes

The control is a near-perfect no-op on every decision measurement. That is the
right behaviour — it trains a converged policy on more of the distribution it
was already fitted to — and it is what makes the arm's numbers mean something.

Premature commitment falls 30 points in the arm and does not move in the
control. Probe count rises 0.6 in the arm and does not move in the control. Both
runs share the pipeline, the packing, the mask, the seeds, the instances, the
budget and the step size. **The difference is which states carried supervision,
which is exactly P1's claim.**

It also disposes of the hypothesis that the stage is simply broken. A broken
packing, a mis-set loss mask, a bad target, or a corrupting format would damage
the control too. The control is untouched.

### 3.2 What the control does not rescue

Protocol failure goes 3.1% → 82.0% in the arm's first round and settles at
27.5%; in the control it never leaves 2–3%. So the interface damage is caused by
learner-conditioned data specifically, at a step size the project has evidence
for, and the retracted 1e-4 run's round-1 collapse (83.0%) is essentially the
same magnitude as this one (82.0%). **The step size inflated the endpoint damage
and did not cause the collapse.**

The round-0 tranche makes this concrete. It was collected from the *dense*
policy, which played almost perfectly — 6,051 accepted actions, 0 malformed, 200
invalid, 2,005 of 2,048 episodes terminated. Training 128 updates on those 6,002
examples takes protocol failure from 3% to 82%. The control's round-0 tranche is
the same size (6,279 examples) in the same format, and is a no-op.

The only systematic difference between the two tranches is whose actions the
contexts contain. Why that is destructive to action emission is **not explained
here**, and the honest statement is that it is an open mechanism rather than a
described one.

### 3.3 Held-out sets at the endpoint

| set | arm success | control success | arm premature | control premature |
|---|---:|---:|---:|---:|
| validation | 33.0% | 41.6% | 42.5% | 72.5% |
| structural (`n_hyp`+1) | 2.6% | 1.0% | 62.3% | 67.3% |
| reversible control | 10.6% | 21.4% | 41.8% | 61.3% |
| Rendering B | 0.0% | 0.0% | — | — |

The premature-commitment reduction transfers to every set on which the arm
commits at all. Raw success does not, because protocol failure does not
transfer either.

---

## 4. The cold arm: the interface is crossable by supervision alone

512 updates over 8,192 on-policy episodes from random weights, at the dense
arm's own 6e-4.

| updates | success | proto-fail | legal-cond. | premature | probes | NLL |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.0% | 1.000 | — | — | 0.00 | 5.697 |
| 128 | 0.0% | 1.000 | — | — | 1.00 | 2.981 |
| 256 | 0.0% | 1.000 | — | — | 1.98 | 3.158 |
| 384 | 3.3% | 0.885 | 28.8% | 20.3% | 2.63 | 0.191 |
| 512 | **11.5%** | 0.328 | 17.2% | 48.3% | 2.47 | 0.141 |

| round | accepted | malformed | invalid | terminated |
|---|---:|---:|---:|---:|
| 0 | 0 | 8,192 | 0 | 0 |
| 1 | 2,048 | 18 | 8,174 | 0 |
| 2 | 4,037 | 5,258 | 2,935 | 0 |
| 3 | 5,844 | 4,906 | 2,346 | 298 |

**The action grammar is acquired inside one round** — malformed attempts fall
8,192 → 18 — and the failures immediately move to well-formed-but-illegal.

This is the contrast the arm was declared against. Outcome-only RL from the same
initialization produced no gradient at all: every rollout died unparseable and
the parameters were byte-identical before and after. The difference is not the
budget and not the world. A verifier scores a trajectory that has to exist
first; a teacher labels the state an unparseable attempt leaves unchanged.

Conditioned on legal play the cold arm sits at 17.2% against a 16.8% chance
floor. It acquired the interface, not the process.

---

## 5. P3: recovery

The frozen evaluator ends an episode at its first protocol failure, so
persistence is unobservable to it. The recovery probe reuses the collector as an
instrument on a held-out block that is never trained on. Across the arm's
milestones the attempt failure rate falls steadily while the recovery rate rises
then falls, and the `budget-0` comparison is two recoveries out of four events.
**P3 is not supported.** What improved is that there are fewer failures, not
that failures are less persistent.

---

## 6. What this stage still cannot separate

- **One seed, one world family, one step size per arm.** The control removes the
  pipeline and the state-distribution confound; it does not make this a
  two-sided effect-size estimate.
- **Why learner-conditioned contexts damage action emission.** §3.2 establishes
  that they do and that teacher-conditioned contexts at matched everything do
  not. The mechanism is open. The leading untested candidate remains the
  train/test context mismatch: the arm trains on histories containing its own
  illegal actions, which the evaluator never presents.
- **Whether the premature-commitment gain survives a fixed interface.** The two
  effects move in opposite directions and are summed by raw success.

---

## 7. What follows

1. **A retry-tolerant evaluation** of the checkpoints that already exist. It
   tests §3.2's candidate mechanism and needs no training.
2. **Collect with failed attempts excised from the running context**, one
   config-level change, which would separate "supervision at learner states"
   from "training on contexts containing one's own failures".
3. **Premature commitment on the other retained checkpoints.** The metric is
   new, so the RLVR arms and transfer endpoints have never been read this way,
   and the instrument now exists. Scoring-only.

The result that needs none of them: **the dense policy commits with two
hypotheses live, three times in four, and learner-conditioned supervision is
what moves that — 73.0% → 42.5%, against a matched control that does not move it
at all.** It cost the action interface to get there, and the accuracy the
prediction expected did not follow, because more evidence is not the same as the
right evidence.
