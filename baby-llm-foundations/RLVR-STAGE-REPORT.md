# Outcome-Only RLVR

## Stage report: the verified-outcome baseline, cold start and dense warm start

### Status

This reports the first runs of the outcome-only condition specified in
[RLVR-STAGE-PLAN.md](RLVR-STAGE-PLAN.md) and required by
[STEP-1.md](STEP-1.md) §6. Three arms were executed on the frozen apparatus: a
cold start from the dense arm's initialization policy, and two warm starts that
treat the dense checkpoint as the post-SFT model of an ordinary post-training
pipeline — outcome-only RL on top of it, without and with a KL trust region.

All results are single-seed and single-budget. §5 tests the obvious objection
that the central null is an artefact of untuned RL, by running the two knobs
that objection names. Neither rescues it: a trust region changes nothing, and a
33× larger step moves the policy ten times further and costs nine points of
held-out capability.

---

## Summary

1. **From a weight-naive start, outcome-only verification supplied exactly zero
   gradient.** Every rollout was killed by an unparseable action, every group's
   rewards were identical, the loss was exactly 0.0, and the model's parameters
   were byte-identical before and after training.

2. **This is a fact about the action surface, not about the task.** A policy
   that merely emits *well-formed* actions — sampling uniformly from the
   world's own valid-action set, with no teacher guidance at all — succeeds
   17.2% of the time, and 78% of eight-sample groups carry a usable gradient.
   The world is not reward-sparse; the byte interface is unreachable by chance.

3. **Applied to an already-competent policy, outcome-only optimization moved
   the action interface and not the decision process.** It halved the rate of
   trajectories killed by illegal actions (14.7% → 6.6% across the run) while
   leaving held-out in-distribution success statistically unchanged
   (41.1% → 41.3%).

4. **It also cost out-of-distribution robustness.** On held-out structural
   combinations success fell 4.8% → 1.5%, with malformed and invalid action
   rates both rising.

5. **A KL trust region changed nothing.** The anchored arm matches the
   unanchored one to within noise on every metric, and its KL plateaued at
   0.005 against a 0.02 penalty — the anchor never engaged. Unconstrained drift
   is not what suppressed process learning.

6. **At that step size the policy barely moved at all** — teacher-forced NLL
   0.0853 → 0.0911, about 7% relative. Cheap protocol fixes live inside that
   radius; a change of decision policy does not.

7. **A 33× larger step moved it ten times further and made it worse.** NLL rose
   74%, the anchor finally engaged, and held-out success fell 41.1% → 31.8%.
   Its own sampled reward stayed flat while the greedy policy the evaluator
   measures lost nine points.

8. Step size is therefore not the missing ingredient. Too small and nothing
   moves; large enough to move and capability degrades. Taken together the arms
   say the same thing from several directions: in this world family, verified
   outcomes have purchase on the surface layer and none yet demonstrated on the
   decision layer.

---

## 1. Apparatus

Unchanged from the dense and transfer stages: the same 19.2M-parameter GPT-NeoX
model, the same 262-token byte vocabulary, the same Rust world executor and
public parser, the same evaluator, the same held-out seeds, two T4 GPUs.

The learning rule is the only difference. Outcome-only training runs through
TRL's `GRPOTrainer` with a project-owned rollout function that drives the world
turn by turn; TRL owns the objective, group-relative advantages, clipping, and
distributed optimization. The reward is one scalar per finished trajectory from
the privileged verifier, using exactly the evaluator's success rule. Verified
spend and protocol failure are computed and logged at weight zero. No teacher
target, preferred-action set, or other intermediate label crosses into
training. The unanchored arm sets `beta = 0`, so no reference model exists at
all; the anchored arm sets `beta = 0.02` against the SFT policy it starts from,
which constrains drift without adding supervision the arm did not already
have.

---

## 2. Cold start: no gradient exists

Configuration `t4x2_rlvr_smoke.toml`, six updates, 384 rollout episodes.

| quantity | value |
|---|---:|
| rollout episodes | 384 |
| episodes ending in a malformed action | 384 (100%) |
| world transitions | 0 |
| updates with any group reward variance | 0 of 6 |
| `frac_reward_zero_std` | 1.0 on every update |
| training loss | 0.0 |
| model state SHA-256 at init / update 3 / update 6 | identical (`ed5fc86eed69…`) |

The zero is exact rather than small, and it is structural. Group-relative
advantage is `(r − mean)/(std + 1e-4)`; when every reward in a group is equal
the advantage is zero, the advantage-scaled loss is zero, and AdamW with zero
gradients and zero weight decay performs no update. The identical state hashes
confirm the arithmetic: the run was a no-op, and a longer budget would have
been a longer no-op.

The reason is the conjunction the reward requires: a trajectory pays only if it
is byte-exactly parseable *and* legal *and* correct. A randomly initialized
byte-level policy must produce a 16-character string such as
`inspect(probe_3)` by chance before any of that can be evaluated.

### 2.1 The world is not reward-sparse

The obvious confound is that outcome-only learning might be hopeless here
because success itself is rare. It is not. Sampling uniformly from the world's
own valid actions — no model, no teacher — over 3,000 episodes per policy:

| policy | success |
|---|---:|
| commit immediately at random | 16.8% |
| uniform over valid actions | **17.2%** |
| probe-biased random | 17.0% |
| privileged teacher | 100% |

and 78% of eight-sample groups on a single world have non-degenerate rewards.
Four out of five groups would carry a gradient if the learner could express
actions at all. (`step1_experiments.reward_density` reproduces this; it is CPU
only and takes about a minute.)

Note also that probing at random does not beat committing at random: the ~17%
floor is chance commitment among six hypotheses, and everything above it is
genuine use of evidence. The dense model's 41.1% is therefore real process
competence, not surface fluency.

---

## 3. Dense warm start: the interface moves, the process does not

The unanchored arm is reported first; §3.3 adds its KL-anchored pair.

Configuration `t4x2_rlvr_warmstart_seed0.toml`, starting from the verified
dense seed-0 checkpoint (state SHA-256 `cef0ac5a41…`), attached from its own
Kaggle output and identity-checked before use.

Executed budget: 191 updates, 12,224 rollout episodes, 34,602 world
transitions, 597,995 model-generated action tokens, 968 s of training wall
time. All 191 updates contained groups with reward variance, so unlike the cold
start the gradient signal existed throughout.

### 3.1 Held-out closed-loop evaluation

1,024 episodes per set, identical seeds to the dense evaluation:

| set | dense (SFT) | after GRPO | Δ |
|---|---:|---:|---:|
| validation (in-distribution, Rendering A) | 41.1% | 41.3% | +0.2 pp |
| structural (n_hyp + 1) | 4.8% | **1.5%** | **−3.3 pp** |
| rendering B (zero-shot) | 0.0% | 0.0% | 0 |
| reversible control | 15.1% | **19.7%** | **+4.6 pp** |

At n = 1024 the validation difference is well inside noise (SE ≈ 1.5 pp). The
structural and reversible differences are roughly 5σ and 4σ.

Action-interface behaviour explains the direction of both real effects:

| set | malformed (dense → RL) | invalid (dense → RL) |
|---|---|---|
| validation | 0.000 → 0.000 | 0.028 → 0.019 |
| structural | 0.297 → 0.353 | 0.361 → 0.456 |
| reversible control | 0.007 → 0.001 | 0.327 → 0.086 |

### 3.2 What the training log optimized

By sixths of the run:

| sixth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| protocol failure rate | 0.147 | 0.129 | 0.089 | 0.093 | 0.070 | 0.066 |
| verified success reward | 0.357 | 0.328 | 0.381 | 0.384 | 0.415 | 0.353 |
| fraction of degenerate groups | 0.36 | 0.35 | 0.45 | 0.44 | 0.46 | 0.46 |

Protocol failure falls monotonically and roughly halves. Success reward is flat
within noise. GRPO spent 12,224 episodes learning to stop making illegal moves,
not learning to decide better.

Cost per successful episode also did not improve: mean spend on the validation
set went 3.884 → 3.924, and excess cost over the teacher on successful episodes
went 0.439 → 0.485.

### 3.3 The KL-anchored arm: the anchor is not the explanation

Configuration `t4x2_rlvr_klanchor_seed0.toml` is identical to the arm above in
every field except `beta` (0.0 → 0.02), so the pair isolates the trust region at
matched budget, world stream, seeds, and evaluator. Under the post-training
reading — the dense checkpoint is the post-SFT model — this is the ordinary
recipe, and the anchor adds no supervision the arm did not already start with.

It changed almost nothing:

| set | dense | β = 0 | β = 0.02 |
|---|---:|---:|---:|
| validation | 41.1% | 41.3% | 41.2% |
| structural | 4.8% | 1.5% | 1.1% |
| rendering B | 0.0% | 0.0% | 0.0% |
| reversible control | 15.1% | 19.7% | 19.6% |

The two runs are nearly identical trajectory for trajectory: 4,511 successful
rollout episodes against 4,512, and 34,519 world transitions against 34,602.
Their training curves coincide:

| sixth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| protocol failure, β = 0 | 0.147 | 0.129 | 0.089 | 0.093 | 0.070 | 0.066 |
| protocol failure, β = 0.02 | 0.147 | 0.130 | 0.094 | 0.092 | 0.082 | 0.075 |
| success reward, β = 0 | 0.357 | 0.328 | 0.381 | 0.384 | 0.415 | 0.353 |
| success reward, β = 0.02 | 0.357 | 0.328 | 0.376 | 0.394 | 0.409 | 0.356 |

The anchor never bound. KL to the reference rose to 0.005 and plateaued
(mean 0.0029, final 0.0046), far below where a 0.02 penalty would meaningfully
push back. The policy was not straining against the trust region; it was barely
moving. (The clipping statistic is 0.000 throughout, but that is structural
rather than evidence: with one iteration per batch and on-policy generation the
importance ratio is identically 1, so clipping can never trigger in any arm.)

### 3.4 How far the policy moved at all

Teacher-forced action NLL on the evaluator's own validation seeds, against the
dense arm's 0.0853:

| budget | NLL | validation success |
|---:|---:|---:|
| dense (0 updates) | 0.0853 | 41.1% |
| 48 | 0.0862 | 43.9% |
| 95 | 0.0881 | 42.8% |
| 191 | 0.0911 | 41.2% |

The policy does move, monotonically away from the teacher distribution, but by
about 7% relative over the whole run. That small distributional move bought a
real behavioural change on the protocol axis — illegal-move trajectories
roughly halved — and nothing on the decision axis. (The 48- and 95-update
success figures are on 512 episodes, the 191 figure on 1,024; read the column
as flat.)

### 3.5 The larger-step arm: movement without capability

Configuration `t4x2_rlvr_bigstep_seed0.toml` differs from the anchored arm in
exactly one field: `learning_rate`, 3e-6 → 1e-4, about 33× and still 6× below
the 6e-4 the dense arm used on this same model.

The step size was indeed the constraint on *movement*:

| | dense | lr 3e-6 | lr 1e-4 |
|---|---:|---:|---:|
| teacher-forced NLL | 0.0853 | 0.0911 | **0.1481** |
| KL to reference (mean) | — | 0.0029 | **0.0311** |
| validation success | 41.1% | 41.2% | **31.8%** |
| structural | 4.8% | 1.1% | 0.4% |
| rendering B | 0.0% | 0.0% | 0.0% |
| reversible control | 15.1% | 19.6% | 23.0% |

The policy moved roughly ten times further — 74% of relative NLL against 7% —
and the anchor finally engaged, KL settling above the 0.02 penalty scale rather
than an order of magnitude below it. Held-out in-distribution capability fell by
nine points.

The milestone curve locates the decline: 44.7% at 48 updates and 43.0% at 95
(both on 512 episodes), then 31.8% at 191 (on 1,024). Teacher-forced NLL rises
monotonically throughout — 0.0973, 0.1194, 0.1481 — so the policy is moving
steadily away from the teacher, and capability follows it down after the
mid-point.

What makes this more than "the learning rate was too high" is that the arm's own
sampled objective did **not** collapse with it. Success reward by sixths reads
0.388, 0.351, 0.386, 0.381, 0.411, 0.331 — roughly flat, essentially the small-
step arm's curve — and total successful rollout episodes were 4,533 against the
small step's 4,511. Protocol failure was already low and stayed low (0.069 →
0.064), with invalid episodes falling sharply (1,179 → 509) while malformed rose
(67 → 271).

So the sampled policy held its own reward while the greedy policy it is
evaluated under lost nine points. Optimizing sampled verified outcomes moved the
distribution somewhere that pays about the same under sampling and considerably
worse under the decoding the evaluator uses.

---

## 4. Interpretation

The two arms bracket the same conclusion. At the cold start, outcome-only
reward could not reach the decision process because it could not clear the
interface. From a competent start, it cleared the interface, tidied it further,
and still did not move the process — while eroding the structural
generalization the dense teacher had installed.

This is consistent with what the transfer stage found from another direction:
in this family the binding constraint on a weight-naive learner is acquiring
the representation, not discovering the process. Dense teaching is what makes
the action space reachable at all; verified outcomes then refine behaviour
within it.

It does **not** support the stronger claim that the task is learnable only
through the teacher. §2.1 shows a well-formed random policy already collects
abundant reward. What outcome-only learning lacked at the cold start was a way
into the action space, and that is a property of the byte surface, which
STEP-1 §11 anticipated as a confound requiring the informative-tokenizer
diagnostic.

---

## 5. Is this just RL being fussy?

The central null — outcome-only optimization did not improve the decision
process — could be a fact about *this* configuration of RL rather than about
outcome-only learning in this world. The runs were deliberately untuned, so the
question is live and was tested rather than argued.

**Excluded: the missing trust region.** The KL-anchored arm (§3.3) reproduces
the unanchored arm to within noise on every metric, including the structural
degradation that looked like unconstrained drift. The anchor was not merely
unhelpful, it was never engaged: KL plateaued at 0.005 against a 0.02 penalty
and the clipping region stayed at 0.000. Drift is not what suppressed process
learning, because there was very little drift to suppress.

**Excluded: too small a step.** The two small-step arms agreed because both
barely moved — 0.005 nats of KL, 7% of NLL. The 33× arm (§3.5) confirms the
diagnosis and refutes the remedy: it moved ten times further by both measures,
and held-out success fell to 31.8%. The two step sizes bracket the question.
Any useful setting lies between them, and the space in between is where a
sweet spot would have to hide — it is not where the evidence currently points,
because nothing in either arm shows the decision process improving.

**What the larger step did reveal** is a mismatch worth naming: at 1e-4 the
sampled objective stayed flat while greedy held-out capability fell nine
points. The quantity being optimized and the quantity being reported came
apart. That is a property of the setup, not of the world, and it makes
temperature and decoding a live suspect in a way they were not before.

Still open, and untested:

- **Budget.** 191 updates is a quarter of the cold-start arm's declared budget.
  Now worth running, but at a step size between the two tried, not at either.
- **Sampling versus evaluation.** Rollouts at temperature 1.0, evaluation
  greedy. The large-step arm shows these can move in opposite directions;
  matching them, or evaluating at the sampling temperature, would say whether
  the reported collapse is a decoding artefact.
- **Group and batch shape.** Eight samples per world with 40–51% of groups
  degenerate means much of each batch contributes nothing; larger groups or
  difficulty filtering would raise effective signal per update.
- **Reward shape.** Binary terminal success gives no credit for getting closer,
  and `scale_rewards = "group"` amplifies single-success groups.
- **Sampling temperature.** Rollouts at temperature 1.0 from a policy evaluated
  greedily means the optimized distribution is not the evaluated one.

The objection that this is just RL being fussy is now partly vindicated and
partly answered. It is vindicated in that the outcome is extremely
step-size-sensitive: a 33× change swings held-out success by nine points. It is
answered in that neither setting produces process learning — one is inert, the
other is destructive — so "fussy" does not yet mean "would work if tuned". The
first half of §4, that outcome-only learning cannot bootstrap from a
weight-naive start, depends on none of this: a zero advantage is zero at any
step size.

---

## 6. What this stage establishes, and what it does not

Established:

- Outcome-only verified learning from a weight-naive start produces exactly no
  gradient on this family and this byte surface, verified by identical model
  state hashes rather than inferred from a flat metric.
- That null is attributable to the action surface, not to reward sparsity: an
  unguided well-formed policy succeeds 17.2% of the time and 78% of groups
  carry a gradient.
- Applied to a competent policy, outcome-only optimization measurably improves
  action-protocol validity and measurably degrades structural generalization,
  with no in-distribution success gain, at the budget and step size run.
- A KL trust region does not change that outcome and never engages: the
  anchored and unanchored arms coincide, so unconstrained drift is not the
  mechanism.
- The result is strongly step-size-sensitive, and not in a way that rescues it:
  at 33× the step the policy moves ten times further, the anchor engages, and
  held-out success falls from 41.1% to 31.8% while its own sampled reward stays
  flat.

Not established:

- That outcome-only learning cannot improve the decision process in this family
  under a better-chosen RL configuration. §5 now excludes the two knobs the
  objection named, but a step size between 3e-6 and 1e-4, a different sampling
  temperature, or a different group composition remain untried.
- That the large-step decline is a capability loss rather than a decoding
  artefact: sampled reward held while greedy evaluation fell, and the two were
  not measured under a common decoding rule.
- Anything about seeds beyond seed 0, or budgets beyond those run.
- The dense-versus-RLVR comparison at matched budget from a common start, which
  the cold-start null makes unrunnable as specified: an arm that never updates
  cannot be compared on a learning curve. A matched comparison needs either an
  interface-calibration prefix for the RLVR arm, reported separately, or the
  informative-tokenizer diagnostic.

---

## 7. Reproducibility record

Both runs were submitted through `tools/kaggle_run.py` against pinned commits,
and their compact evidence is tracked under `step1/audit/runs/`.

Cold start:

- Config `step1/configs/kaggle/t4x2_rlvr_smoke.toml`, root seed `20260811`
- Source commit `de89b5262c69e9eeec455b8143d0e25f9ff5100a`
- Kaggle notebook `aniruddhavarma/step1-rlvr-smoke-de89b52`
- Model state SHA-256, unchanged throughout: `ed5fc86eed691fe85f21c84c…`

Dense warm start:

- Config `step1/configs/kaggle/t4x2_rlvr_warmstart_seed0.toml`, root seed
  `20260811`, training worlds from seed band `root_seed + 4,000,000`
- Source commit `58afc55`, run ID
  `t4x2-rlvr-warmstart-seed0-58afc551cddf-ca566773c8ef`
- Kaggle version `aniruddhavarma/step1-rlvr-warmstart-seed0-58afc55/1`
- Upstream dense checkpoint `aniruddhavarma/step1-t4x2-dense-seed0-84f2938`,
  git SHA `84f29385ed623500aa2e201c45fdcf8c2257fac0`, model state
  `cef0ac5a4159d65eae336be58c34dfe1e3f078a8024f4b98a7ca1a78e42a9a6e`, verified
  inside the run before training
- Algorithm: TRL 0.29.1 `GRPOTrainer`, `loss_type="dapo"`,
  `scale_rewards="group"`, `beta=0.0`, `epsilon=0.2`, 8 generations per world,
  1 iteration per batch, reward weights `[1.0, 0.0, 0.0]`
- Report contract `step1_rlvr_grpo_v1`; receipt and result report under
  `step1/audit/runs/t4x2-rlvr-warmstart-seed0-58afc551cddf-ca566773c8ef/`

KL-anchored warm start:

- Config `step1/configs/kaggle/t4x2_rlvr_klanchor_seed0.toml`, identical to the
  unanchored arm except `beta = 0.02`, enforced by a contract test
- Source commit `9ed892f`, run ID
  `t4x2-rlvr-klanchor-seed0-9ed892fae2dd-473fc24dd1f0`
- Kaggle version `aniruddhavarma/step1-rlvr-klanchor-seed0-9ed892f/1`
- Same upstream dense checkpoint, verified inside the run
- 191 updates, 12,224 rollout episodes, 34,519 world transitions, 1,003 s
- KL to reference: mean 0.0029, final 0.0046; clipping region 0.000 throughout
- Teacher-forced action NLL at 48 / 95 / 191 updates: 0.0862 / 0.0881 / 0.0911,
  against the dense arm's 0.0853, measured on a deterministic shard of the
  evaluator's validation seeds

Larger-step warm start:

- Config `step1/configs/kaggle/t4x2_rlvr_bigstep_seed0.toml`, identical to the
  KL-anchored arm except `learning_rate = 1e-4`, enforced by a contract test
- Source commit `3f2cf50`, run ID
  `t4x2-rlvr-bigstep-seed0-3f2cf506d4e9-0af3f7860338`
- Kaggle version `aniruddhavarma/step1-rlvr-bigstep-seed0-3f2cf50/1`
- 191 updates, 12,224 rollout episodes, 35,115 world transitions, 1,013 s
- KL to reference: mean 0.0311, final 0.0265
- Teacher-forced action NLL at 48 / 95 / 191 updates: 0.0973 / 0.1194 / 0.1481
- Held-out validation success at the same milestones: 44.7% and 43.0% on 512
  episodes, 31.8% on 1,024

Reward-density control: `python -m step1_experiments.reward_density --config
step1/configs/kaggle/t4x2_rlvr_seed0.toml`, 3,000 episodes per policy, 400
groups of 8, CPU only.

---

## 8. Stage conclusion

The stage distinguishes one of STEP-1 §14's admissible conclusions in a
qualified form: **on this world family and byte surface, outcome-only verified
learning cannot bootstrap from a weight-naive start at all, and applied to a
dense-trained policy it improves the action interface without improving the
decision process.**

The first half is unconditional: a zero advantage is zero at any step size or
budget. The second half survived both attempts to explain it away. A KL trust
region reproduced it exactly and never engaged. A 33× larger step did move the
policy — ten times further by KL and NLL, with the anchor finally binding — and
held-out capability fell nine points while the sampled objective stayed flat.

Three arms now sit on the same conclusion from different directions, so the
stage's finding is not an artefact of one arbitrary setting. What remains
genuinely untried is the interval between the two step sizes, a decoding rule
shared between training and evaluation, and group composition — and the first
thing any of those should be asked to beat is the 44.7% at 48 updates, the only
point in the entire stage that looks like a gain, and one that is not
statistically distinguishable from the dense model it started from.
