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
that the central null is an artefact of untuned RL: the trust-region form of it
is now excluded, and step size is the remaining live explanation.

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

6. **The policy barely moved at all.** Teacher-forced NLL went 0.0853 → 0.0911
   over 191 updates, about 7% relative. Cheap protocol fixes live inside that
   radius; a change of decision policy plausibly does not. The declared
   learning rate of 3e-6 — against the 6e-4 the dense arm used on this same
   model — is now the leading explanation for the null, and is untested.

7. Taken together the arms say the same thing from opposite ends: in this world
   family, verified outcomes have purchase on the surface layer and little on
   the decision layer, at the step size tried.

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
push back, and the clipping-region statistic was 0.000 for the whole run. The
policy was not straining against the trust region; it was barely moving.

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

**Now the leading explanation: step size × budget.** The two arms agree because
both take very small steps. Over 191 updates the policy moved 0.005 nats of KL
and +0.006 teacher-forced NLL — about 7% relative — while its protocol
behaviour changed substantially. Cheap surface fixes are reachable inside that
distributional radius; a change of decision policy plausibly is not. The
declared learning rate, 3e-6, was taken from RLHF practice for models three
orders of magnitude larger; the dense arm trained this same 19.2M model at
6e-4. That gap, not the budget, is the most suspicious single number in the
configuration.

Still open, and untested:

- **Learning rate.** The cheapest discriminating experiment is not more
  updates but larger steps: the same 191 updates at 3e-5 or 1e-4, watching KL
  and NLL to confirm the policy actually moves further.
- **Budget.** 191 updates is a quarter of the cold-start arm's declared budget.
  Worth running after the step size is calibrated, not before — 4× of a step
  that is 100× too small is still too small.
- **Group and batch shape.** Eight samples per world with ~40% of groups
  degenerate means much of each batch contributes nothing; larger groups or
  difficulty filtering would raise effective signal per update.
- **Reward shape.** Binary terminal success gives no credit for getting closer,
  and `scale_rewards = "group"` amplifies single-success groups.
- **Sampling temperature.** Rollouts at temperature 1.0 from a policy evaluated
  greedily means the optimized distribution is not the evaluated one.

Until the learning rate is calibrated, §4's second half should be read as
"outcome-only optimization did not move the process *at this step size and
budget*", not as a general claim about RLVR. The first half — that it cannot
bootstrap from a weight-naive start — does not depend on any of these knobs,
because a zero advantage is zero at any step size.

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

Not established:

- That outcome-only learning cannot improve the decision process in this family
  under a better-chosen RL configuration. §5 excludes the trust-region
  explanation but leaves step size untested, and the measured 7% distributional
  movement makes it the leading candidate.
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
budget. The second half survived the first serious attempt to explain it away —
a KL trust region reproduced it exactly — but remains conditional on step size,
and the measured movement of the policy (0.005 nats of KL, 7% NLL) says the
steps were small. The next experiment is a learning rate calibrated to this
model rather than borrowed from RLHF practice for models a thousand times
larger; only after that does a larger budget mean anything.
