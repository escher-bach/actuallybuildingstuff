# Outcome-Only RLVR

## Stage report: the verified-outcome baseline across five configurations

### Status

This reports the first runs of the outcome-only condition specified in
[RLVR-STAGE-PLAN.md](RLVR-STAGE-PLAN.md) and required by
[STEP-1.md](STEP-1.md) §6. Three arms were executed on the frozen apparatus: a
cold start from the dense arm's initialization policy, and two warm starts that
treat the dense checkpoint as the post-SFT model of an ordinary post-training
pipeline — outcome-only RL on top of it, without and with a KL trust region.

All results are single-seed. §6 tests the objection that the central null is an
artefact of untuned RL by running the knobs that objection names, ending with a
deliberate best-effort configuration. None rescues it: a trust region changes
nothing, a 33× step costs nine points of held-out capability, and the tuned
configuration removes every side effect while still not improving the decision
process.

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
   moves; large enough to move and capability degrades.

9. **A deliberately tuned configuration fixed everything except the result.**
   With 2e-5 and cosine decay, groups of 16, a 4× budget, and outcome-only
   credit for reaching a verdict, degenerate groups fell to ~30%, protocol
   failure to 3–4%, the structural degradation disappeared (4.4% against the
   dense arm's 4.8%), and the sampled objective improved for the first time
   (0.359 → 0.413). Held-out success finished at 41.2% against 41.1%.

10. The pattern across four configurations spanning 33× in learning rate, two
    group sizes, two reward shapes, two budgets, and with and without an
    anchor: outcome-only optimization reliably improves the action interface
    and the agreement between sampling and decoding, and has not once moved
    held-out decision capability in this world family.

---

## 1. The arms at a glance

Every warm-start arm begins from the same dense seed-0 checkpoint and is scored
by the same evaluator on the same held-out seeds. Held-out success is on 1,024
episodes per set.

| arm | lr | β | group | updates | episodes | NLL | validation | structural | reversible |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dense seed 0 (start) | — | — | — | — | — | 0.0853 | 41.1% | 4.8% | 15.1% |
| cold start (§3) | 3e-6 | 0 | 8 | 6 | 384 | — | *no update occurred* | | |
| warm start (§4) | 3e-6 | 0 | 8 | 191 | 12,224 | — | 41.3% | 1.5% | 19.7% |
| KL anchored (§4.3) | 3e-6 | 0.02 | 8 | 191 | 12,224 | 0.0911 | 41.2% | 1.1% | 19.6% |
| larger step (§4.5) | 1e-4 | 0.02 | 8 | 191 | 12,224 | 0.1481 | 31.8% | 0.4% | 23.0% |
| best shot (§4.6) | 2e-5 cos | 0.02 | 16 | 382 | 48,896 | 0.1070 | 41.2% | 4.4% | 21.2% |

Rendering B is 0.0% for every row, including the dense start: the interface is
ungrounded and no arm changes that. The teacher-forced NLL diagnostic was added
after the first warm start, so that row has no entry. The cold-start arm's
parameters never changed, so its evaluation numbers describe the dense
initialization policy rather than a trained one.

Read down the validation column: one arm destroyed capability, none improved
it. Read down the structural column: the arms that moved the policy hardest
damaged out-of-distribution robustness most, and the tuned arm restored it.

---

## 2. Apparatus

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

## 3. Cold start: no gradient exists

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

### 3.1 The world is not reward-sparse

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

## 4. Dense warm start: the interface moves, the process does not

The unanchored arm is reported first; §4.3 to §4.6 add the three that follow it.

Configuration `t4x2_rlvr_warmstart_seed0.toml`, starting from the verified
dense seed-0 checkpoint (state SHA-256 `cef0ac5a41…`), attached from its own
Kaggle output and identity-checked before use.

Executed budget: 191 updates, 12,224 rollout episodes, 34,602 world
transitions, 597,995 model-generated action tokens, 968 s of training wall
time. All 191 updates contained groups with reward variance, so unlike the cold
start the gradient signal existed throughout.

### 4.1 Held-out closed-loop evaluation

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

### 4.2 What the training log optimized

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

### 4.3 The KL-anchored arm: the anchor is not the explanation

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

### 4.4 How far the policy moved at all

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

### 4.5 The larger-step arm: movement without capability

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

### 4.6 Best shot: everything the measurements justified, at once

Configuration `t4x2_rlvr_bestshot_seed0.toml` changes four things together, so
it is a best-effort attempt rather than a controlled pair: step size 2e-5 with
cosine decay (bracketed by the two arms above), groups of 16 instead of 8
(40–51% of groups had been degenerate), a budget of 382 updates × 128 rollouts
= 48,896 episodes (matching the dense arm's 48,832 packed sequences), and a
fourth reward term — credit for reaching a verdict at all, weighted 0.25
against success's 1.0 — to break groups in which every rollout fails but not
all fail the same way. `terminated` is a field of the verifier's terminal
report, so the condition remains outcome-only.

Every process-health measure improved, several markedly:

| by sixth | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|---:|
| degenerate groups | 0.194 | 0.335 | 0.306 | 0.296 | 0.329 | 0.335 |
| protocol failure | 0.072 | 0.042 | 0.039 | 0.034 | 0.037 | 0.040 |
| sampled success reward | 0.376 | 0.401 | 0.387 | 0.418 | 0.443 | 0.394 |
| legal termination | 0.928 | 0.958 | 0.961 | 0.966 | 0.963 | 0.960 |

Degenerate groups fell from the 40–51% of earlier arms to about 30%; protocol
failure reached 3–4%, the best of the stage; KL settled at 0.011–0.016, moving
the policy without straining the anchor; and for the first time the **sampled
objective improved**, 0.359 at the first update to 0.413 over the final decile.

Held-out capability did not follow:

| set | dense | best shot |
|---|---:|---:|
| validation | 41.1% | 41.2% |
| structural | 4.8% | 4.4% |
| rendering B | 0.0% | 0.0% |
| reversible control | 15.1% | 21.2% |

The milestone curve is flat throughout: 42.8% at 48, 95 and 191 updates (512
episodes each, +1.7 pp over dense at 0.8σ), then 41.2% at 382 on the full 1,024.
Teacher-forced NLL moves once, early — 0.0853 → 0.1057 by update 48 — and then
sits at 0.105–0.107 for the remaining 334 updates. Milestone state hashes are
all distinct, so the repeated 42.8% is a stable policy rather than a repeated
checkpoint.

What this arm *did* fix is the collateral damage. Structural generalization
lands at 4.4% against the dense arm's 4.8%, inside noise, where the three
earlier arms had driven it to 0.4–1.5%. The reversible control improves as in
every RL arm, and its invalid-action rate falls from 0.327 to 0.017.

The cleanest reading of the whole arm is **sharpening**. Sampled success rose
from 0.359 to 0.413, and greedy held-out success sat at 0.412: outcome-only
optimization pulled the sampling distribution up to what the argmax policy was
already doing, without moving the argmax policy. Fewer illegal actions, fewer
degenerate groups, better agreement between sampling and decoding — and the
same decisions.

---

## 5. Interpretation

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
through the teacher. §3.1 shows a well-formed random policy already collects
abundant reward. What outcome-only learning lacked at the cold start was a way
into the action space, and that is a property of the byte surface, which
STEP-1 §11 anticipated as a confound requiring the informative-tokenizer
diagnostic.

---

## 6. Is this just RL being fussy?

The central null — outcome-only optimization did not improve the decision
process — could be a fact about *this* configuration of RL rather than about
outcome-only learning in this world. The runs were deliberately untuned, so the
question is live and was tested rather than argued.

**Excluded: the missing trust region.** The KL-anchored arm (§4.3) reproduces
the unanchored arm to within noise on every metric, including the structural
degradation that looked like unconstrained drift. The anchor was not merely
unhelpful, it was never engaged: KL plateaued at 0.005 against a 0.02 penalty
and the clipping region stayed at 0.000. Drift is not what suppressed process
learning, because there was very little drift to suppress.

**Excluded: too small a step.** The two small-step arms agreed because both
barely moved — 0.005 nats of KL, 7% of NLL. The 33× arm (§4.5) confirms the
diagnosis and refutes the remedy: it moved ten times further by both measures,
and held-out success fell to 31.8%. The two step sizes bracket the question.
Any useful setting lies between them, and the space in between is where a
sweet spot would have to hide — it is not where the evidence currently points,
because nothing in either arm shows the decision process improving.

**Excluded: degenerate groups, budget, and reward shape.** The best-shot arm
(§4.6) addressed all three at once, and each intervention worked on its own
terms — degenerate groups fell to ~30%, the budget quadrupled to the dense
arm's episode count, and the graded reward lifted legal termination to 96%. The
sampled objective improved. Held-out capability did not.

**What the arms did reveal** is a mismatch worth naming: the quantity being
optimized and the quantity being reported are not the same. At 1e-4 sampled
reward held while greedy capability fell nine points; in the best-shot arm
sampled reward rose to meet greedy capability without lifting it. Outcome-only
RL is optimizing the sampling distribution, and in this family that appears to
be a different object from the decoded policy the evaluator scores.

Still open, and untested:

- **Sampling versus evaluation.** Rollouts at temperature 1.0, evaluation
  greedy. Two arms now show these moving independently. Evaluating at the
  sampling temperature, or rolling out nearer to greedy, would say whether the
  stage has been measuring the wrong policy all along. This is now the single
  most informative untried experiment.
- **Seeds.** Every arm is seed 0. The dense stage's own transfer results were
  strongly seed-dependent in magnitude.
- **A cold start with a grounded interface.** The comparison STEP-1 actually
  asks for remains unrunnable as specified; an RLVR arm with a declared,
  separately reported interface-calibration prefix would make it runnable.
- **Reward shape.** Binary terminal success gives no credit for getting closer,
  and `scale_rewards = "group"` amplifies single-success groups.
- **Sampling temperature.** Rollouts at temperature 1.0 from a policy evaluated
  greedily means the optimized distribution is not the evaluated one.

The objection that this is just RL being fussy is vindicated in one sense and
answered in another. It is vindicated in that the outcome is extremely
sensitive to configuration: a 33× step-size change swings held-out success by
nine points, and tuning decides whether the policy is left intact or wrecked.
It is answered in that tuning it well produces a healthy, stable, well-behaved
run whose held-out capability is indistinguishable from where it started.
"Fussy" turned out to describe the side effects, not the result.

The first half of §5, that outcome-only learning cannot bootstrap from a
weight-naive start, depends on none of this: a zero advantage is zero at any
step size.

---

## 7. What this stage establishes, and what it does not

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
- A tuned configuration — bracketed step size with decay, groups of 16, 4×
  budget, and outcome-only credit for reaching a verdict — removes the
  structural degradation entirely and improves the sampled objective, and still
  finishes at 41.2% held-out against 41.1%.

Not established:

- That outcome-only learning cannot improve the decision process in this family
  under a better-chosen RL configuration. §6 now excludes the two knobs the
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

## 8. Reproducibility record

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

Best-shot arm:

- Config `step1/configs/kaggle/t4x2_rlvr_bestshot_seed0.toml`; four declared
  changes from the KL-anchored arm, listed in §4.6
- Source commit `4975e04`, run ID
  `t4x2-rlvr-bestshot-seed0-4975e043247f-32d72e790525`
- Kaggle version `aniruddhavarma/step1-rlvr-bestshot-seed0-4975e04/1`
- 382 updates, 48,896 rollout episodes, 142,427 world transitions, 4,070 s
- 19,756 successful rollout episodes (40.4%), 1,921 invalid, 230 malformed
- KL to reference: mean 0.0138, final 0.0111; degenerate groups 30.0% mean
- Teacher-forced action NLL at 48 / 95 / 191 / 382: 0.1057 / 0.1046 / 0.1048 /
  0.1070; milestone state hashes all distinct

Reward-density control: `python -m step1_experiments.reward_density --config
step1/configs/kaggle/t4x2_rlvr_seed0.toml`, 3,000 episodes per policy, 400
groups of 8, CPU only.

---

## 9. Stage conclusion

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

Four configurations now sit on the same conclusion — spanning 33× in learning
rate, two group sizes, two reward shapes, two budgets, and with and without a
trust region — so the stage's finding is not an artefact of one arbitrary
setting. The best-tuned of them is the most informative: it fixed every side
effect, improved the objective it was given, and finished exactly where it
started on held-out capability.

What remains genuinely untried is the one thing all four arms share: rollouts
are sampled at temperature 1.0 and evaluation decodes greedily. Two arms show
those quantities moving independently, and the best-shot arm's sampled reward
rose precisely to meet its unchanged greedy score. Before concluding anything
further about what verified outcomes can teach, the stage should establish
which policy it has been measuring.
