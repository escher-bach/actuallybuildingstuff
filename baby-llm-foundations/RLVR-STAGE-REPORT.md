# Outcome-Only RLVR

## Stage report: the verified-outcome baseline, cold start and dense warm start

### Status

This reports the first runs of the outcome-only condition specified in
[RLVR-STAGE-PLAN.md](RLVR-STAGE-PLAN.md) and required by
[STEP-1.md](STEP-1.md) §6. Two arms were executed on the frozen apparatus: a
cold start from the dense arm's initialization policy, and the optional hybrid
row — outcome-only optimization applied to the completed dense seed-0
checkpoint.

Both results are single-seed and, for the warm start, single-budget. The
strongest alternative explanation for the central null is stated in §5 and is
not yet excluded.

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

5. Taken together the two arms say the same thing from opposite ends: in this
   world family, verified outcomes have purchase on the surface layer and
   little on the decision layer. §5 gives the competing explanation.

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
training. There is no KL anchor (`beta = 0`), so no reference model can
reintroduce teacher information.

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

Configuration `t4x2_rlvr_warmstart_seed0.toml`, starting from the verified
dense seed-0 checkpoint (state SHA-256 `cef0ac41…`), attached from its own
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

## 5. The strongest alternative explanation

The central null — outcome-only optimization did not improve the decision
process — may be a fact about *this* configuration of RL rather than about
outcome-only learning in this world. The run was deliberately untuned, and
several ordinary knobs could each produce a flat curve:

- **Budget.** 191 updates is a quarter of the cold-start arm's declared budget
  and a small fraction of the dense arm's exposure. Policy-gradient methods
  routinely need far more.
- **No KL anchor.** `beta = 0` was chosen so the objective stays outcome-only,
  but it also leaves drift unconstrained. The structural degradation is exactly
  what an unanchored policy drifting toward its training distribution looks
  like.
- **Learning rate.** 3e-6 constant-with-warmup was declared, never tuned. A
  flat reward curve is equally consistent with too small a step.
- **Group and batch shape.** Eight samples per world with 42% of groups
  degenerate means a large share of each batch contributes nothing; larger
  groups or difficulty filtering would raise the effective signal.
- **Reward shape.** Binary terminal success gives no credit for getting closer,
  and `scale_rewards = "group"` amplifies single-success groups.
- **Sampling temperature.** Rollouts at temperature 1.0 from a policy evaluated
  greedily means the optimized distribution is not the evaluated one.

Nothing here is evidence that these explain the null — only that they are not
excluded. The discriminating experiments, cheapest first:

1. a KL-anchored arm at the same budget, to separate "RL cannot move the
   process" from "unanchored RL drifts before it can";
2. one arm at 4× budget, to test the budget explanation directly;
3. a post-RL teacher-forced NLL against the dense arm's 0.0853, which would
   show whether the policy moved at all in distributional terms;
4. a larger group size or difficulty filter, to raise effective signal per
   update.

Until at least (1) and (2) are run, §4's conclusion should be read as
"outcome-only optimization did not move the process *at this budget, without an
anchor, at this learning rate*", not as a general claim about RLVR.

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
  with no in-distribution success gain, at the budget run.

Not established:

- That outcome-only learning cannot improve the decision process in this family
  under a better-chosen RL configuration (§5).
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
  `cef0ac4159d65eae336be58c34dfe1e3f078a8024f4b98a7ca1a78e42a9a6e`, verified
  inside the run before training
- Algorithm: TRL 0.29.1 `GRPOTrainer`, `loss_type="dapo"`,
  `scale_rewards="group"`, `beta=0.0`, `epsilon=0.2`, 8 generations per world,
  1 iteration per batch, reward weights `[1.0, 0.0, 0.0]`
- Report contract `step1_rlvr_grpo_v1`; receipt and result report under
  `step1/audit/runs/t4x2-rlvr-warmstart-seed0-58afc551cddf-ca566773c8ef/`

Reward-density control: `python -m step1_experiments.reward_density --config
step1/configs/kaggle/t4x2_rlvr_seed0.toml`, 3,000 episodes per policy, 400
groups of 8, CPU only.

---

## 8. Stage conclusion

The stage distinguishes one of STEP-1 §14's admissible conclusions in a
qualified form: **on this world family and byte surface, outcome-only verified
learning cannot bootstrap from a weight-naive start at all, and applied to a
dense-trained policy at the budget run it improves the action interface without
improving the decision process.** Whether the second half survives a
KL-anchored, longer, better-tuned RL configuration is the open question this
stage hands forward, and §5 lists the experiments that would settle it.
