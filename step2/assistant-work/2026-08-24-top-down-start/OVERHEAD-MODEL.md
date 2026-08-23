# Formal Inner- and Outer-Loop Overhead Model

**Date:** 2026-08-24
**Status:** assistant-authored systems model; one non-authoritative local CPU
world/binding benchmark has been performed; Kaggle CPU/T4 measurements remain
pending

## 1. Accounting boundary

Static-dataset pretraining is not outside the developmental system. It is the
simplest **inner loop**:

```text
prepacked batch -> model forward/backward -> optimizer -> next batch
```

A generated offline world adds producer and serialization stages to that inner
loop. A learner-conditioned world additionally adds sequential policy/world
rounds. The outer loop begins only when training pauses or branches to decide
what experience the current checkpoint should receive next.

For `K` recursive decision actions, including evaluations that may not change
the model weights, total wall-clock is:

```text
T_total = sum_(k=0)^(K-1) [T_action(k) + T_control(k)].
```

For a training action, `T_action=T_inner`; for an evaluation or world-admission
action it is that action's execution critical path. `T_control` is the
additional checkpoint/scheduling path required to select and dispatch the next
action. This prevents evaluations at unchanged weights from disappearing from
the ledger.

The accounting keeps three different things separate:

1. **intrinsic training work**—model FLOPs and memory traffic required by the
   chosen architecture, sequence, objectives, and batch;
2. **inner-loop realization overhead**—data production stalls, learner decode,
   world synchronization, communication not hidden by compute, and pipeline
   bubbles; and
3. **outer-loop control overhead**—checkpoint barriers, evaluation, disposable
   branches, world admission/switching, queue/launch delay, and decision latency.

All three are charged to developmental throughput. “Overhead” never means
“unimportant”; it means cost relative to a declared matched reference.

## 2. Three matched inner-loop references

For a given session, preserve three measurements.

### `R_plain`: ordinary static inner loop

A prepacked static batch format with the same model, optimizer, global batch,
precision, number of gradient-bearing targets, and update count, but without
world-specific context or online interaction. This is the ordinary-pretraining
reference.

### `R_packed`: exact-transcript static inner loop

The exact world transcripts and masks that the developmental session will use
are pre-generated and read from storage. This includes the true token length
and objectives but removes dynamic generation and learner/world synchronization.

### `R_live`: actual selected world inner loop

The real offline-generated, hybrid, or learner-conditioned pipeline.

These define two separately actionable taxes:

```text
serialization/context tax = T_packed / T_plain - 1
live-realization tax       = T_live   / T_packed - 1
total inner tax            = T_live   / T_plain  - 1
```

Without `R_packed`, longer sequences can be mistaken for a slow world engine.
Without `R_plain`, a perfectly buffered generator can be called “free” even
though its calibration tokens increase transformer work.

## 3. Static inner loop: architectural lower bound

Let a training step use sequence length `L`, hidden width `h`, MLP width `m`,
`r` transformer layers, global batch `B_g`, and `p` devices.

For a Llama-style decoder layer, counting a multiply and an add as two FLOPs,
a useful first-order forward-FLOP model is:

```text
F_layer,fwd(L) ~= 8*L*h^2 + 6*L*h*m + 4*L^2*h.
```

The first term is Q/K/V/output projection, the second is the three SwiGLU
matrices, and the third is QK-score plus attention-value work. This convention
must be recorded because counting one fused multiply-add as one operation would
halve all three coefficients. Softmax, projectors, heads, normalization,
embeddings, padding, and implementation details are added from the actual
graph. Backward and optimizer multipliers should be measured rather than
permanently assumed.

For each step, record total FLOPs `F_step` and memory traffic `M_step`. The
Roofline lower bound is:

```text
t_kernel >= max(F_step / P_sustained,
                M_step / BW_sustained).
```

`P_sustained` and `BW_sustained` are measured for the selected precision and
kernels on the actual T4 setup, not quoted peak marketing values. Operational
intensity `F_step/M_step` determines whether the step is compute- or
bandwidth-limited.

The observed device-side step is decomposed as:

```text
t_device =
    t_forward_backward
  + t_optimizer
  + t_unhidden_collective
  + t_kernel_launch_gaps
  + t_device_idle.
```

The terms must be obtained from a trace with non-overlapping critical-path
durations. Summing profiler categories that overlap would double count time.

## 4. Two-device communication term

For data parallelism with gradient payload `S_g` bytes on `p` devices, a ring
all-reduce has the latency-bandwidth approximation:

```text
t_allreduce ~= 2*(p-1)*alpha
            + 2*(p-1)/p * S_g / BW_link.
```

`alpha` is per-phase communication latency and `BW_link` is measured effective
link bandwidth. Only the portion not overlapped by backward computation enters
the step critical path:

```text
t_unhidden_collective = max(0, t_allreduce - t_overlap).
```

For the intended two-device start, `p=2`; the model still measures rather than
assumes the link behavior. Gradient accumulation changes how frequently this
term is paid. Distributed efficiency is reported as:

```text
eta_DDP = throughput_p / (p * throughput_1).
```

This is a LogP/latency-bandwidth model: communication has startup latency,
host/device overhead, a finite message-rate gap, and payload time. It is more
informative than counting only transferred bytes.

## 5. Producer/trainer pipeline

For prepacked data, the producer stages are storage read, decode/collate, and
host-to-device copy. For an offline world they additionally include world
generation, teacher target construction, rendering/token packing, and optional
write/read caching.

With bounded buffers and successful overlap, steady-state time for `K` batches
is:

```text
T_pipeline ~= T_fill
           + K * max(t_generate,
                     t_pack,
                     t_load_H2D,
                     t_device)
           + T_drain
           + T_starvation.
```

The slowest stage sets throughput. An offline world generator has essentially
zero steady-state *critical-path* overhead when it produces packed batches at
least as fast as the device consumes them and the buffer never empties. Its CPU
work and energy still exist and are reported as resource cost.

Little's-law-style buffer sizing supplies an operational check. If the device
consumes at rate `lambda` batches/s and producer latency is `W` seconds, the
ready queue needs at least approximately `lambda*W` batches, plus variance
margin, to avoid deterministic starvation. The actual starvation duration is
measured directly.

## 6. Work/span model for learner-conditioned worlds

The critical distinction for an interactive world is **work** versus **span**.

- `W` is all producer, model, and world work and can often be parallelized
  across environments.
- `S` is the longest dependency chain and cannot be removed by adding workers.

On `P` effective workers, execution is bounded in the work-span model by:

```text
T >= max(W/P, S),
```

with the usual greedy-scheduling form `O(W/P + S)` under its assumptions.

If an episode needs `q` learner-owned probes, its span contains:

```text
S_episode >= sum_(j=1)^q [
    t_policy_forward(j, B_env)
  + t_action_decode(j)
  + t_world_step(j, B_env)
  + t_render_pack(j, B_env)
  + t_round_barrier(j)
].
```

Vectorizing `B_env` environments amortizes each round over more transitions,
but it does not remove the `q` sequential rounds. This is the formal reason a
world requiring ten learner-selected tests can have much lower throughput than
one whose sufficient calibration is supplied offline.

An asynchronous actor/learner system can overlap rollout and training stages,
turning steady-state throughput back into a pipeline maximum. It does not erase
the rollout work or span; it adds policy staleness and queueing that must be
measured as part of learning quality. The initial small system should prefer a
simple synchronous or bounded-staleness path until overlap is actually needed.

## 7. Exact cost of variable trajectory tokens

For event `t`, let:

- `o_t` be active observation/entity tokens, including eight tokens per active
  visual view after resampling;
- `q^a_t` be schema-conditioned action-query tokens;
- `a_t` be actual executed-action tokens;
- `q^o_t` be optional next-observation query tokens;
- `f_t` be public feedback tokens; and
- `b_t` be condition/boundary tokens.

Let `s_w` be encounter-schema tokens. The exact packed length is:

```text
L_w = s_w + sum_t [b_t + o_t + q^a_t + a_t + q^o_t + f_t].
```

Counts are taken from the serialized mask, not inferred from a nominal world
dimension. For `W_calibrated_monomial` with `d` scalar sensors and actuators,
the `d` supplied calibration transitions each contain `O(d)` channel/action
tokens, so calibration context grows `O(d^2)` even though world simulation is
`O(d)`. This is one reason `d` is initially bounded at four.

Action chunk width does not multiply transformer token count because one query
per actuator predicts a masked 16-sample vector. It does multiply head output,
target, and loss work; record that separately. Visual resampling reduces patch
count before the temporal core, so both pre- and post-resampler costs must be
reported.

There is no “tokenization is cheap” assumption. Insert `L_w` into the
transformer FLOP model and profile the exact packed batch. At the selected
experimental profile, `L_w <= 2,048`; truncation may not cut an encounter or
silently remove the demonstration/calibration evidence needed by a target.
The offline cost of adding `q` tests is:

```text
Delta T_context(q) =
    K * [t_device(L_w(q)) - t_device(L_w(0))].
```

If those tests are learner-owned, add the unhidden online span:

```text
Delta T_tests,live(q) = Delta T_context(q)
                      + T_online_span(q)
                      - T_overlap_realized(q).
```

Also report gradient density:

```text
gradient_density = number_of_direct_target_positions / L_w,
```

and target throughput:

```text
target_throughput = direct_target_positions / T_inner.
```

This allows a longer transcript to justify itself only if its additional
context improves end-capability learning enough to offset lower target and raw
token throughput.

## 8. Complete inner-loop critical path

For session `n`, use trace-derived, mutually exclusive critical-path terms:

```text
T_inner(n) =
    T_startup_fill
  + T_device_busy
  + T_unhidden_input_stall
  + T_unhidden_world_generation_stall
  + T_unhidden_online_decode_and_world_span
  + T_unhidden_distributed_communication
  + T_inner_barriers
  + T_drain.
```

Static pretraining is the special case where world-generation and online-span
terms are zero and inputs are prepacked. It still pays device, input,
communication, launch, and pipeline costs.

Define:

```text
O_inner_abs = T_live - T_plain
omega_inner = O_inner_abs / T_plain
rho_inner   = O_inner_abs / T_live.
```

`omega` is overhead relative to the baseline; `rho` is the fraction of actual
time attributable to that overhead. Both are useful and must not be confused.

### 8.1 Training-time world monitor

The open online-evaluation option belongs to the inner critical path when it is
executed during a training action. Let `m_w` compute a world-defined metric
vector or scheduler scalar every `k_m` updates. Its attributable cost is not
the sum of all monitor timers, but the additional resource-constrained
makespan:

```text
O_monitor_abs = T_inner_with_monitor - T_inner_without_monitor
rho_monitor   = O_monitor_abs / T_inner_with_monitor.
```

For a monitor calculated from already-produced trajectories, the main terms
are metric computation, cross-rank reduction, logging, and any synchronization
that cannot overlap training. A held-out stream additionally pays inference,
world generation/rollout, input transfer, and possible weight-snapshot or
staleness cost. These cases must be reported separately.

A simple world efficiency signal may take the form

```text
eff_w(t) = change in declared world score over a window
           / attributable tokens, updates, transitions, or seconds.
```

The unit and denominator must travel with the scalar; scores with different
world semantics are not made comparable merely by having one numeric value.
The score is control-plane telemetry unless the world contract separately
declares it as learner-visible feedback.

The monitor is worthwhile when its expected saved overshoot and improved next
decision exceed its own cost:

```text
E[time/resources avoided by an earlier correct reconsideration]
    > O_monitor_abs + extra checkpoint/control events it triggers.
```

This break-even condition, rather than a fixed monitoring frequency, determines
`k_m`. STEP 1's SDL statistic is an unadopted candidate curve summary for this
role; its definition and calibration would have to be ported exactly before it
could enter the cost or stopping model.

## 9. Outer control event DAG and critical path

The control path around a recursive decision event is an event DAG, not always
a simple sum. Its nodes may include:

- flush optimizer/data state and write the immutable checkpoint;
- verify/hash/upload the checkpoint and provenance;
- run Tier-1 sentinels;
- launch and run Tier-2 disposable adaptation branches;
- perform a milestone Tier-3 fan-out audit;
- admit or generate a new world version;
- wait in an execution queue or pay Kaggle launch/setup time;
- load checkpoint/data and compile/warm kernels for the chosen next session;
- compute the scheduling score; and
- wait for a human decision when one is required.

For every outer task `j`, record duration `t_j`, resource vector `r_j`, and
dependencies. Then report:

```text
T_outer_wall = resource-constrained critical-path makespan of the event DAG
C_outer_GPU  = sum_j (GPU_count_j * GPU_time_j)
C_outer_CPU  = sum_j (CPU_core_count_j * CPU_time_j).
```

Parallel tests can reduce wall-clock makespan but do not reduce GPU-hours. On a
fixed two-T4 allocation, GPU branches are normally serial unless each uses only
one device. A result that blocks selection of the next world remains on the
critical path even if its arithmetic is small.

The correct launch constant must come from the actual STEP 2 image. As prior
apparatus evidence, the repository's completed STEP 1 two-T4 path observed
10–13 minutes for clone/install/build and about 3 minutes for tests: roughly
13–16 minutes before useful work. STEP 2 may be faster because its standard
PyTorch path does not require the same Rust build, but a five-minute estimate
must not be substituted for a measurement. Many tiny outer-loop jobs can be
dominated by setup even when their GPU kernels are short.

## 10. Outer overhead and amortization interval

Let a normal session contain `K` optimizer steps at measured mean inner-loop
time `t_step`, and let the mandatory outer work after it cost `O` seconds. The
outer fraction is:

```text
rho_outer = O / (K*t_step + O).
```

To keep this below a chosen budget `rho_star`, the checkpoint/decision interval
must satisfy:

```text
K >= O*(1-rho_star) / (rho_star*t_step).
```

This makes checkpoint frequency a calculated systems parameter. Frequent
decisions improve scheduler responsiveness but increase the serial fraction;
larger training blocks amortize control but can overshoot the capability
frontier.

For tiered evaluation:

```text
E[O] = O_checkpoint
     + O_tier1
     + Pr(tier2)*O_tier2
     + Pr(tier3)*O_tier3
     + O_switch_and_launch
     + O_decision.
```

The probabilities are estimated from actual decision history. A full audit
every `M` checkpoints contributes `O_tier3/M` to amortized per-checkpoint cost.

### 10.1 Allocation batching and the TPU queue

For accelerator allocation `j`, separate queue and setup from logical
checkpoint sessions:

```text
T_allocation(j) = Q_j + T_setup_j
                + makespan(logical sessions, monitors, checkpoints,
                           autonomous scheduler decisions)
                + T_finalize_j.
```

`Q_j` consumes wall-clock latency even when it consumes no accelerator-hours.
The user reports a greater-than-two-hour Kaggle TPU queue. Therefore the TPU is
not an economical arm for ordinary world or tokenizer experiments. Primary
experimentation remains on two T4s, and multiple logical sessions should be
executed in one T4 allocation when validity and isolation permit.

A final TPU allocation may contain many logical checkpoint transitions, but
all stop/switch rules needed while the job is unattended must be frozen and
auditable before launch. A human decision that ends the allocation and requires
a relaunch pays another `Q_TPU + T_setup_TPU` on the critical path.

Scaling width/depth is not ordinary continuation. If a larger TPU model starts
from random weights, its cost and capability evidence belong to a new blank
confirmation lineage. Replaying a small-model protocol tests scale transfer;
it does not preserve the small checkpoint. A same-size TPU continuation and a
larger blank replay must be reported separately.

## 11. Exact branch-evaluation cost

For `K_probe` candidate worlds, matched adaptation from two comparator
checkpoints, `b_k` updates per arm, and measured step time `t_k`, model-training
cost is:

```text
T_branch_train = sum_(k=1)^K_probe 2*b_k*t_k.
```

`C_n` versus `C0` tests positive transfer. A current-state action decision may
instead require `C_n` versus a matched branch that skipped or replaced the
candidate action. The comparator is selected by the decision claim; two arms
are not automatically sufficient for every causal question.

Add checkpoint loads, data setup, rollouts, and launch time for total branch
cost. Its update-only ratio to a primary session of `B` updates at `t_main` is:

```text
omega_branch = T_branch_train / (B*t_main).
```

Paired generator seeds reduce comparison variance. Sequential confidence
intervals should stop evaluation once additional episodes cannot change the
world ranking or promotion decision. This minimizes sample count without
pretending that a fixed tiny test is decisive.

## 12. Amdahl limit on the full developmental system

Let fraction `s` of total time be serial or decision-blocking outer work and
unvectorized learner/world span. Even with unbounded acceleration of the
parallel training portion, speedup is bounded by:

```text
speedup_max <= 1/s.
```

This is why reducing checkpoint launches, decision barriers, and learner-world
rounds can matter more than another small kernel optimization. The measured
serial fraction is part of the scheduler state.

## 13. Total efficiency and the fair fixed-pretraining baseline

Across sessions:

```text
rho_outer,total = sum(T_outer) / T_total
rho_inner,total = sum(O_inner_abs) / T_total
eta_useful_train = sum(T_plain) / T_total.
```

The end objective remains developmental, so also report:

```text
D_throughput = [U_end(S_K) - U_end(S_0)] / T_total.
```

The adaptive process earns its extra machinery only if:

```text
D_throughput_adaptive > D_throughput_fixed_mixture,
```

or, equivalently, it reaches the same declared end-capability utility in less
total wall-clock/compute after all inner and outer costs are included.

The fair fixed-mixture baseline is one predetermined static-data inner action
using the same architecture, total resource budget, ordinary checkpointing, and
final evaluation. It has no checkpoint-conditioned world-selection barrier.
It is also a legal candidate inside the recursive policy at every state. This
comparison directly answers whether adaptive control beats standard
pretraining rather than merely producing a better final model after spending
more control-plane compute.

## 14. Measurements required from the first authorized preflight

No numerical overhead can be honestly supplied before hardware timing, but the
model makes every term measurable. The first authorized preflight should time:

1. `R_plain` prepacked static training;
2. `R_packed` exact world-transcript training;
3. buffered offline world generation plus training;
4. vectorized learner-conditioned rollout at several `q` and `B_env` values;
5. one- versus two-device throughput and unhidden all-reduce;
6. checkpoint write, verify, load, compile/warm-up, and Kaggle launch/setup;
7. Tier-1 rollout cost and one minimal matched adaptation branch; and
8. producer queue depth and starvation time.

The T4 preflight comes first. A later TPU preflight additionally measures queue,
XLA compile, device topology, checkpoint-format conversion, mask/numerical
parity, and end-to-end makespan. It requires an explicit extension to the
repository execution plan; TPU plumbing is not part of the current T4
implementation contract.

Use standard device traces and profiler events to construct the critical path;
do not infer overlap by summing subsystem timers. The result is a fitted cost
table that the checkpoint scheduler uses immediately.

## 15. Computer-architecture sources for the overhead model

- [Roofline: An Insightful Visual Performance Model for Multicore Architectures](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.html)
- [Amdahl, Validity of the Single Processor Approach](https://doi.org/10.1145/1465482.1465560)
- [LogP: Towards a Realistic Model of Parallel Computation](https://dl.acm.org/doi/10.1145/155332.155333)
- [Brent, The Parallel Evaluation of General Arithmetic Expressions](https://maths-people.anu.edu.au/~brent/pub/pub022.html)
