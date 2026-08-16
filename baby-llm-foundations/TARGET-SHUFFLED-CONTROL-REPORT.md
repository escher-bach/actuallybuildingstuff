# The Target-Shuffled Control

## Is the dense arm's capability process learning or surface regularity?

### Status

This reports the control required by [STEP-1.md](STEP-1.md) §11 item 3, which
was specified at the start of the step and run only now. It settles the
attribution question that both previous stage reports had to leave open.

---

## Summary

**The dense arm's capability is process learning.** A model trained on the
identical data stream with the state-to-action correspondence destroyed scores
**16.1%** where the dense arm scores **41.1%**, and 16.1% is the chance-
commitment floor. The 25-point gap is attributable to reading the state.

---

## 1. The control

Everything is held constant except one thing. The world executes the teacher's
action at every step, so states, observations, trajectory lengths, episode
count, token budget, optimizer recipe, root seed, and evaluator are identical
to `t4x2_dense_seed0`. Only the supervised label changes: instead of the
teacher's preferred action, the span teaches an action sampled uniformly from
those legal in that state.

The configs differ in exactly one field, enforced by a test. Training labels
alone are corrupted; every evaluation shard, including the one the
teacher-forced diagnostic scores against, keeps teacher targets. The shard
manifest and replay records declare which policy produced them, so a control
shard cannot be mistaken for a teacher shard.

The control has a **predeclared expected value**. Uniform-over-legal is exactly
the unguided policy measured independently at 17.2% closed-loop success, with
chance commitment among six hypotheses at 16.8%. If the dense arm's advantage
is genuinely the state-to-action mapping, the control should land there.

---

## 2. Result

Held-out closed-loop success, 1,024 episodes per set, identical seeds:

| set | dense | shuffled control | difference |
|---|---:|---:|---:|
| validation (in-distribution) | **41.1%** | **16.1%** | −25.0 pp |
| structural (n_hyp + 1) | 4.8% | 14.1% | +9.3 pp |
| rendering B (zero-shot) | 0.0% | 0.0% | 0 |
| reversible control | 15.1% | 16.1% | +1.0 pp |

Teacher-forced action NLL: 0.0853 for the dense arm, 0.1406 for the control,
as expected when the labels themselves carry more entropy.

### 2.1 What the control actually learned

Its behaviour is more informative than its score:

| metric, validation set | dense | control |
|---|---:|---:|
| malformed action rate | 0.000 | **0.000** |
| invalid action rate | 0.028 | **0.000** |
| mean spend | 3.88 | **0.00** |
| mean steps | 2.94 | **1.00** |

The control learned the byte-level action grammar **perfectly** — zero
malformed, zero invalid, better than the dense arm on legality — and then
committed immediately, every episode, without probing at all. Spend of exactly
zero and exactly one step per episode is the signature of a policy that emits a
well-formed commitment and stops.

Its 16.1% is therefore not a degraded version of the dense policy. It is chance
commitment among six hypotheses, dressed in perfect syntax.

### 2.2 Why the control beats the dense arm on structural generalization

The +9.3 pp on the held-out structural set is not a win, it is the same fact
seen from another angle. With seven hypotheses instead of six, chance
commitment pays 1/7 ≈ 14.3%, and the control scores 14.1% — because a policy
that ignores the state is indifferent to a change of structure. The dense arm
attempts to probe on an unfamiliar structure and fails, at 4.8%.

This is worth stating plainly: **on the structural set, the dense arm is worse
than chance commitment.** Its structural generalization was already weak, and
the control makes that legible rather than causing it.

---

## 3. What this establishes

The two conclusions STEP-1 §14 asks to be separated are now separated:

- **"Dense teacher traces improve process learning"** — supported. Twenty-five
  points of the dense arm's capability require the state-to-action
  correspondence and cannot be obtained from the same tokens without it.
- **"They improve only surface acquisition"** — refuted. Surface acquisition
  is what the control got, and it is worth 16.1%, which is chance.

It also retroactively strengthens every earlier result in the project. The
transfer stage's acquisition-bias effect and the RLVR stage's interface-only
findings were both measured against a dense arm whose competence had not been
attributed. It now has been.

---

## 4. What it does not establish

- **Nothing about the process contrast.** Whether the model learned the
  irreversible-versus-reversible distinction is still untested; that needs an
  arm trained on the reversible variant, which does not exist. This control
  tests only whether the model reads the state.
- **Nothing about the ceiling.** The dense arm captures roughly 30% of the
  headroom between chance (16.8%) and the teacher (100%). The 25-point gap is
  real process learning and is also far from mastery.
- **One seed.** Seed 0 only, as with every arm in the RLVR stage.
- A residual asymmetry: `inspect(probe_5)` and `commit(cause_2)` differ in byte
  length, so the arms see supervised-token counts differing by roughly 3%
  within the identical 100,007,936-token budget. Far too small to move 25
  points, but it is not exactly zero.

---

## 5. Reproducibility record

- Config `step1/configs/kaggle/t4x2_dense_shuffled_seed0.toml`, identical to
  `t4x2_dense_seed0.toml` except `train_target_policy`
- Source commit `adbd9d0`, run ID
  `t4x2-dense-shuffled-seed0-adbd9d0814dc-59ddcc1051b1`
- Kaggle version `aniruddhavarma/step1-dense-shuffled-seed0-adbd9d0/1`
- Root seed `20260811`; 3,052 updates; 100,007,936 nominal input tokens —
  identical to the dense arm
- Baseline: dense seed 0, commit `84f29385ed623500aa2e201c45fdcf8c2257fac0`
- Independent floor: `python -m step1_experiments.reward_density`, 3,000
  episodes per policy, CPU only
