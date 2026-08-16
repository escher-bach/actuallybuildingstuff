# Outcome-Only RLVR

## Stage plan: the verified-outcome baseline against dense teacher supervision

### Status and scope

> **Executed. Results are in [RLVR-STAGE-REPORT.md](RLVR-STAGE-REPORT.md).**
>
> This plan remains the record of the apparatus, the ownership boundary, the
> rollout protocol, the reward definition, and the reasoning behind the
> algorithm choice — all of which held. Three parts of it were overtaken by
> what the runs found, and are marked **[superseded]** below rather than
> rewritten, so the plan still shows what was predeclared:
>
> - **§7 arms and run sequence.** The planned sequence was smoke → cold start
>   seed 0 → optional hybrid. The cold-start seed-0 run was never executed:
>   the six-update smoke proved it would be a no-op, verified by identical
>   model state hashes, so spending 763 updates on it would have bought
>   nothing. The stage instead ran four warm-start configurations.
> - **§5 reward.** A fourth outcome-only term — credit for reaching a verdict,
>   weighted 0.25 — was added for the final arm to break degenerate groups. It
>   defaults to zero, so the earlier arms kept the reward defined here.
> - **§8 cold-start risk.** Written as a risk to be watched. It is now a
>   result: the predicted failure occurred exactly as described, which is why
>   the instrument declared there was worth declaring.
>
> The framing also shifted. §2 justifies `beta = 0` as keeping teacher
> information out; under the post-training reading the project settled on —
> the dense checkpoint is the post-SFT model — a KL anchor to that same policy
> is a trust region rather than supervision, and three of the four arms used
> one.

This document specifies the outcome-only RLVR condition named in
[STEP-1.md](STEP-1.md) §6 and row 2 of its first experimental matrix. It fixes
the algorithm, the reward, the rollout protocol, the budget policy, and the
run sequence before any retained run, so that the result — including a null —
is interpretable.

It does not change the world, the teacher, the renderings, the model
configuration, or the evaluator. Those remain as used by the dense arm and by
the completed representation-transfer stage
([RENDERING-B-TERMINAL-TRANSFER-REPORT.md](RENDERING-B-TERMINAL-TRANSFER-REPORT.md)).

---

## 1. The question

> At the same budget of world experience, does a learner trained only on the
> verified outcome of its own trajectories acquire the same closed-loop
> capability as a learner trained on dense teacher targets?

STEP-1 §14 admits several conclusions here, all informative: dense traces help,
they help only with surface acquisition, the verifier baseline is as efficient
or more efficient, or the family cannot discriminate. This stage is designed so
each of those is distinguishable from the others, and in particular so that a
failure of outcome-only learning can be attributed to the *signal* rather than
to a broken apparatus.

---

## 2. Algorithm selection

The requirement is an RL algorithm that (a) exists in a maintained library, per
[AGENTS.md](AGENTS.md) and
[STANDARD-LLM-STACK-MIGRATION-PLAN.md](STANDARD-LLM-STACK-MIGRATION-PLAN.md)
§5.3, (b) learns from one scalar per finished trajectory, and (c) does not
require a second trained model on a 16 GB T4.

**Selected: `trl.GRPOTrainer` (TRL 0.29.1, already pinned in
`requirements-kaggle.txt`).**

| Candidate | Why not selected |
| --- | --- |
| PPO with a learned value head | Adds a second network and a value-fitting failure mode that the group baseline avoids; nothing here needs per-token credit assignment beyond the trajectory verdict. |
| RLOO (`trl.RLOOTrainer`) | Also a group-baseline, maintained trainer, and a defensible choice; GRPO is the more widely reported default for verified-reward training and shares TRL's rollout interface. Switching is a config-level change if the group baseline ever needs revisiting. |
| A project-written policy-gradient loop | Exactly the kind of reimplementation the project has already migrated away from. |

GRPO fits the world's structure directly: one generated world instance is a
prompt, `num_generations` independently sampled trajectories through that same
instance form a group, and the advantage is each trajectory's verified outcome
minus the group mean. No value model, no reward model, no reference model
(`beta = 0`, so no KL term can reintroduce teacher information).

### 2.1 Why a custom rollout function rather than TRL's tool/environment path

TRL's `environment_factory` and tool-calling support assume a chat template and
a JSON tool-call protocol. Our learner speaks a byte protocol with `OBS`,
`ACTION`, and `END_TURN` markers and has never been taught a chat surface;
adopting one would change the interface being studied. TRL's lower-level
`rollout_func` hook keeps the optimizer, loss, advantage estimation, and
distributed training inside TRL while letting the project own exactly the piece
that is project-specific: driving the Rust world turn by turn.

This is the adapter that
[STANDARD-LLM-STACK-MIGRATION-PLAN.md](STANDARD-LLM-STACK-MIGRATION-PLAN.md)
§5.3 anticipated, and it is an environment/protocol adapter, not a custom RL
algorithm. `rollout_func` is marked experimental in TRL 0.29.1; the run report
records the library version, and the contract tests fail loudly if its return
shape changes.

---

## 3. Ownership boundary

| Concern | Owner |
| --- | --- |
| Policy-gradient objective, clipping, group advantages, KL option | `trl.GRPOTrainer` |
| Optimizer, scheduler, accumulation, fp16, DDP, checkpoints, resume | Transformers Trainer / Accelerate (through TRL) |
| Model and tokenizer artifacts | Hugging Face `save_pretrained` |
| World transitions, action parsing, verifier outcomes | project Rust crate |
| Turn-by-turn rollout, loss-mask of injected observations, reward definition, budget accounting, report contract | `step1/python/step1_experiments/rlvr.py` |
| Closed-loop evaluation | the existing shared evaluator |

---

## 4. Rollout protocol

Each dataset row is an episode identity (`step1-world/<seed>`), not learner
text. The rollout renders what the learner actually sees:

```text
prompt      : BOS OBS <observation> ACTION
completion  : <action bytes> END_TURN [ OBS <observation> ACTION <action bytes> END_TURN ]* EOS
```

- Only tokens the model emitted carry gradient. Observations injected between
  turns, and the harness `EOS`, are returned in `env_mask` as external tokens
  and are excluded from the loss by TRL.
- A malformed attempt, a parsed-but-invalid action, or a terminal transition
  ends that trajectory and no other. This is exactly the evaluator's rule, so a
  trajectory that would score as a success under evaluation is precisely the
  trajectory that receives reward 1.
- A trajectory that never reaches a verdict within the declared turn and token
  budget is closed without `EOS`, scores zero, and is counted separately as
  exhausted.
- Sampling uses TRL's defaults (temperature 1.0, no top-p or top-k
  truncation). Nothing in the rollout was tuned against the evaluator.

The declared caps (12 turns, 96 tokens per action, 1,536 completion tokens,
256 prompt tokens) sit well above measured behaviour: over 500 sampled
instances, the privileged teacher needs at most 5 turns and 1,043 completion
tokens, and an exhaustive-probing policy needs at most 1,061.

---

## 5. Reward [partly superseded — see the status note]

One scalar per finished trajectory, computed by the same privileged verifier
the evaluator uses:

- `verified_success` — 1.0 if and only if the episode terminated, the
  commitment was correct, and no malformed or invalid action occurred;
  otherwise 0.0. **This is the trained objective.**
- `verified_spend` — negated verified probe spend, registered at weight 0.0 so
  it is logged every update without entering the objective. Its weight is a
  declared config field; changing it changes the experiment.
- `verified_protocol_failure` — diagnostic only, weight 0.0.

No teacher target, preferred-action set, minimum remaining cost, or any other
privileged intermediate quantity crosses into training. STEP-1 §6 requires
this: RLVR must not receive labels the outcome-only condition is meant to lack.

Note when reading raw logs: TRL's aggregate `reward` field is the *unweighted*
sum across reward functions, while advantages use the weighted sum. The series
to read is `rewards/verified_success/mean`.

---

## 6. Budget policy

No single axis makes RLVR and dense supervision equivalent, so one axis is
matched by declaration and the rest are measured and reported.

**Matched axis — world episodes.** The dense seed-0 arm consumed 100,007,936
nominal input tokens in 2,048-token sequences, i.e. 48,832 packed trajectory
sequences. The RLVR seed-0 arm rolls out exactly 48,832 episodes: 763 updates ×
64 rollouts, where each update covers 8 distinct worlds sampled 8 times.

**Measured and reported axes.** Optimizer updates, world transitions,
model-generated action tokens, injected observation tokens, prompt tokens,
turns, wall-clock seconds, and per-rank completion. These are summed across
ranks by Accelerate and written into the run report; the dense arm's own token
accounting is carried alongside for the warm-start arm.

Note that the two arms differ in world *diversity* at the same episode count:
dense saw 32,768 distinct instances, RLVR sees 6,104 sampled eight times each.
That is a property of group-relative RL, and it is reported rather than hidden.

---

## 7. Arms and run sequence [superseded — see the status note]

1. **Smoke** (`t4x2_rlvr_smoke.toml`, 6 updates). Cargo tests, Python contract
   tests, then the full GRPO path on two T4s with milestone checkpoints and a
   short evaluation. Required before any retained run; it is plumbing, not a
   result.
2. **Seed 0, from initialization** (`t4x2_rlvr_seed0.toml`). The predeclared
   comparison: same initialization policy, same world family, same evaluator,
   same root seed as dense seed 0, training worlds drawn from a disjoint seed
   band (`root_seed + 4,000,000`).
3. **Seed 0, dense warm start** (`t4x2_rlvr_warmstart_seed0.toml`). STEP-1 §6's
   optional hybrid row, at one quarter of the episode budget, starting from the
   verified dense seed-0 checkpoint. It is a diagnostic: any capability it
   shows was partly installed by the dense teacher.

Later seeds are decided after seed 0, following the precedent set by the
transfer stage: a first seed is an end-to-end gate, not a licence to tune.

Each run is a single committed Kaggle version and does not resume: checkpoints
exist at the declared milestones for evaluation, not as a recovery grid. The
seed-0 budget is sized to finish comfortably inside one session; if a measured
run turns out not to, the budget is re-declared before rerunning rather than
patched mid-flight.

---

## 8. The cold-start risk, declared in advance [now a result — §3 of the report]

A randomly initialized 19.2M model must emit exactly parseable byte actions
before it can ever be correct. If no trajectory in a group succeeds, every
reward in that group is identical, every advantage is zero, and GRPO has
nothing to learn from — at any budget. This is the expected difficulty of
outcome-only learning from scratch, and it is the substantive content of the
comparison rather than a bug.

To keep such an outcome interpretable, the run reports TRL's own
`frac_reward_zero_std` alongside the reward series:

- `updates_with_any_reward_variance` — how many updates contained at least one
  group with a non-degenerate reward, i.e. how many updates could carry a
  gradient at all;
- `frac_reward_zero_std_mean` and the first/final reward values.

If this diagnostic reads "no variance, ever", the honest conclusion is that
outcome-only verification supplied no learning signal at this budget on this
family, and the closed-loop numbers merely confirm it. That is a STEP-1 §14
admissible conclusion, and it is cheap to establish. If it reads "variance
present", the milestone curve is the measurement of interest.

---

## 9. Evaluation

Milestones are standard Trainer checkpoints at declared updates. Each is
verified for exact state-dictionary round-trip and then evaluated with the
*same* evaluator the dense and transfer stages used:

- every milestone: held-out in-distribution Rendering A worlds
  (`milestone_episodes`, a prefix of the dense validation set's seeds);
- the terminal milestone: the full matched matrix — validation, held-out
  structural combinations, zero-shot Rendering B, and the matched reversible
  control — at the dense arm's episode counts, on the identical seeds.

Metrics are measurements. No threshold gates a run; the operational contract
(budget completion, milestone grid, artifact exactness, finite metrics) is what
is asserted.

---

## 10. What this stage can and cannot establish

It can establish, at the declared episode budget and on this family:

- whether outcome-only verified learning produces any gradient signal from a
  weight-naive start;
- whether it produces closed-loop capability, and how that capability compares
  to the dense arm's on identical held-out worlds;
- whether outcome-only optimization moves an already dense-trained policy.

It cannot establish that RLVR is weak in general, that a different reward
shaping or a longer budget would not work, or anything about seeds beyond those
run. Those limits belong in the stage report.

---

## 11. Reproducibility record

Each run writes `rlvr_report.json` under contract `step1_rlvr_grpo_v1`,
carrying the config hash, TRL version and algorithm settings, the reward
specification, initialization provenance and model-state hash, the training
seed band, measured budget axes, the training-signal summary, and every
milestone's artifact path, serialization report, and metrics. The full
`log_history` is written beside it, so the prequential instrument validated in
the transfer stage can be applied without rerunning anything.
