# STEP 1 experiment-design failure

**Status:** post-experiment audit, 2026-08-20  
**Scope:** the six-hypothesis, five-probe, binary-evidence STEP 1 family  
**Effect:** the recorded run metrics remain factual, but the main process-learning
interpretation and the previously reported 97.7% learner ceiling do not survive
the information-boundary audit.

## Executive finding

The approximately 40–45% band is not good evidence for a capacity ceiling, an
optimization ceiling, or successful acquisition of the intended
hypothesis-elimination procedure. Two coupled design mistakes produced a
different experiment:

1. **The learner and the benchmark experts were given different decision
   problems.** The learner receives a serialized interaction prefix. The dense
   teacher reads the latent truth and the complete instance. The later expert
   called “truth-blind” omits the truth but still reads the complete evidence
   table, every probe cost, and the step limit. Its 97.7% value is therefore a
   table-aware oracle value, not the ceiling of a policy restricted to the
   learner’s bytes.
2. **The evidence sampler accidentally gives two hypothesis labels stable
   public meanings.** Under the configured binary alphabet, `cause_1` always
   returns amber and `cause_2` always returns blue. The other four causes remain
   exchangeable. A very small policy using only the exact serialized prefix
   scores **483/1,024 = 47.17%** on the evaluator’s validation worlds. Its
   evaluator-matched 100,000-world replication scores **46.254%**. Its idealized
   two- and three-probe values are 41.67% and 45.83%, directly accounting for
   the observed band.

A literal optimizer on the fixed 1,024 raw prefixes reaches **100%**, but this
is not the public population ceiling: 1,020 prefixes are singleton instance
fingerprints. That result is an exact transductive optimum and a demonstration
of a renderer defect, not evidence of generalization.

These are not two competing explanations. The hidden-table mismatch prevents
the intended per-instance elimination policy, while the generator bug supplies
a simpler cross-instance shortcut that remains learnable. Repeated training
experiments changed the optimizer, supervision regime, or decoder while
preserving both properties, so convergence to the same band is expected.

The concise design rule that was missed is:

> An expert used as a learner ceiling must act on the learner’s information
> state. A privileged expert may still supply scaffolding, but its value is not
> the learner’s ceiling, and its instance-specific actions are not automatically
> realizable behavior-cloning targets.

## 1. Intended experiment and actual estimand

STEP 1 intended to test whether dense supervision from an executable world
teaches a reusable process distinction. The specification correctly separates
the learner observation from privileged teacher queries in
[STEP-1.md:220](STEP-1.md#L220): returned evidence and public budget/action
information may be visible, while truth, counterfactual probe results, the
consistent-hypothesis set, and minimal remaining cost are privileged.

Let $X_t$ be the exact causal-LM prefix available at decision time and let
$Z$ contain the unrendered instance fields. The relevant ceiling is

$$
\pi_{\mathrm{public}}^*(a\mid X_t)
=
\arg\max_a \mathbb{E}[R\mid X_t,a],
$$

where the expectation marginalizes all hidden instances compatible with the
prefix. The later audit instead computed a policy of the form

$$
\pi_{\mathrm{table}}^*(a\mid X_t,D,c,L),
$$

where $D$ is the complete counterfactual evidence table, $c$ is the complete
probe-cost vector, and $L$ is the step limit. Removing only $h^*$, the
truth, does not make these two policies equivalent.

This distinction matters even when both policies are called Bayes-optimal. They
are optimal in different information states.

## 2. Exact information-boundary audit

### 2.1 What the model actually receives

The renderer’s `ObservableView` contains only remaining budget, returned
history, current commitment/status, and the currently available actions
([render.rs:301](step1/crates/world/src/render.rs#L301)). Rendering A serializes
those fields as `SEEN`, `BUDGET`, `AVAILABLE`, and `STATUS` lines
([render.rs:449](step1/crates/world/src/render.rs#L449)). It does not serialize a
counterfactual table, an untried probe’s exact cost, or the instance’s step
limit.

The model input is the complete framed prefix, not just the latest observation:

```text
BOS (OBS observation ACTION action END_TURN)*
```

The packer documents and implements that protocol at
[trajectory.rs:314](step1/crates/world/src/trajectory.rs#L314), and the evaluator
extends the same prefix with the model’s own previous actions at
[evaluate.py:151](step1/python/step1_experiments/evaluate.py#L151). Consequently,
the public policy may use earlier observations and budget deltas, but it still
does not receive the unobserved counterfactuals.

### 2.2 What each decision maker receives

| Quantity | Learner prefix | Dense teacher | “Truth-blind” expert |
|---|---:|---:|---:|
| Returned `(probe, evidence)` history | yes | yes | yes |
| Previous actions | yes | yes | yes |
| Remaining total budget | yes | yes | yes |
| Currently available action strings | yes | yes | yes |
| Complete evidence table $D(e\mid q,h)$ | **no** | **yes** | **yes** |
| Exact cost of every untried probe | **no** | **yes** | **yes** |
| Instance step limit | **no** | **yes** | **yes** |
| Consistent-hypothesis set | **no** | derived from table | derived from table |
| Latent truth $h^*$ | **no** | **yes** | no |
| Counterfactual partition induced by each probe | **no** | **yes** | **yes** |

The current `consistent()` implementation makes the mismatch explicit. It
iterates over every hypothesis and calls `inst.evidence_of(q, h)` for every
returned observation ([teacher.rs:83](step1/crates/world/src/teacher.rs#L83)).
It is derived from history **and the table**, not from the serialized history
alone. Two worlds can render the same history while producing different
consistent sets because their hidden tables differ.

The dense teacher then centers its probe search on `inst.truth`, reads
counterfactual evidence and exact costs, and reserves against `inst.step_limit`
([teacher.rs:441](step1/crates/world/src/teacher.rs#L441)). That privilege was
declared by the original design. The mistake was not the existence of a
privileged teacher; it was failing to establish whether its local actions were
recoverable from $X_t$, and later comparing the learner against an oracle with
a different $X_t$.

The later `truth_blind_optimal_action` implementation, introduced in commit
`f7d55b6`, states that it “may read the evidence table but NOT `inst.truth`.”
Its option builder reads `inst.probe_cost[q]` and partitions live hypotheses by
calling `inst.evidence_of(q, h)`. It also branches on `inst.step_limit`. The code
therefore correctly implements a **table-aware, truth-blind** expert. The report
incorrectly promoted that expert to “a learner can actually follow.”

### 2.3 Teacher forcing adds a second information mismatch

Offline data generation gets a public observation, asks for privileged teacher
targets, executes one teacher action, and appends that action to the sequence
([data.py:140](step1/python/step1_experiments/data.py#L140)). At later training
steps, previous teacher actions are in the model context. During evaluation,
the previous actions are the model’s own.

The teacher’s early actions depend on hidden instance fields, so teacher-forced
prefixes can carry information that the learner’s self-generated prefixes do
not. This makes a low teacher-forced action NLL compatible with weak closed-loop
control. The measured 0.0853 NLL versus 41.1% success
([THEORY-PHASE.md:101](THEORY-PHASE.md#L101)) is therefore not puzzling evidence
of a decoding failure; it is also the expected signature of an information and
state-distribution mismatch.

## 3. The generator’s stable-label shortcut

### 3.1 The exact bug

Each instance samples a new table in
[generate.rs:194](step1/crates/world/src/generate.rs#L194):

```rust
let mut v = (rng.next() % p.n_evidence as u64) as u16;
// Never literally hand `h` back as its own evidence value.
if v == h && p.n_evidence > 1 {
    v = (v + 1) % p.n_evidence;
}
```

The production configuration sets six hypotheses and two evidence values
([t4x2_dense_seed0.toml:8](step1/configs/kaggle/t4x2_dense_seed0.toml#L8)).
Therefore:

- for internal hypothesis `h = 0`, a draw of 0 is changed to 1 and a draw of 1
  remains 1, so the result is always evidence 1;
- for `h = 1`, a draw of 1 is changed to 0 and a draw of 0 remains 0, so the
  result is always evidence 0;
- for `h = 2,3,4,5`, neither binary evidence value equals `h`, so the draw is
  left unchanged.

The renderer maps hypothesis 0 to `cause_1`, hypothesis 1 to `cause_2`, evidence
0 to blue, and evidence 1 to amber
([render.rs:146](step1/crates/world/src/render.rs#L146),
[render.rs:160](step1/crates/world/src/render.rs#L160)). The resulting public
semantics are:

| Public hypothesis | Evidence under every probe |
|---|---|
| `cause_1` | always `mark_amber` |
| `cause_2` | always `mark_blue` |
| `cause_3`–`cause_6` | stochastic binary columns, exchangeable under the intended nuisance-free presentation |

The comment intended to prevent a literal identity encoding. Instead, the
conditional rewrite made the evidence distribution depend persistently on the
numeric hypothesis identifier. Equality in an independently sampled table cell
would not itself have leaked the truth; changing the draw conditional on `h`
did.

### 3.2 Why the existing leakage guard missed it

`TruthLeakingProbe` rejects only a probe row that is injective across all
hypotheses ([generate.rs:223](step1/crates/world/src/generate.rs#L223)). With two
evidence values and six hypotheses, an injective row is impossible by the
pigeonhole principle. The configured check is therefore incapable of detecting
this distributional label leak.

The renderer tests address a different property: two otherwise identical
instances differing only in `truth` render identically before a probe, and the
position of the true commit action is approximately uniform. Neither test asks
whether $P(E\mid H=h)$ changes under a hypothesis-label permutation. That is
the property the sampler violates.

### 3.3 The 40–45% calculation

Under the idealized unconditioned sampler, let $H$ be uniform over six labels
and take $k\geq1$ probes. A public policy can use the following rule:

1. all evidence amber: commit `cause_1`;
2. all evidence blue: commit `cause_2`;
3. mixed evidence: `cause_1` and `cause_2` are impossible, so guess one of
   `cause_3`–`cause_6`.

The two special labels contribute (2/6=1/3) success. For an ordinary label,
the probability of a mixed length-$k$ binary history is
(1-2^{1-k}), after which one of four exchangeable labels is guessed. Hence

$$
\begin{aligned}
V_k
&= \frac{2}{6}
 + \frac{4}{6}\left(1-2^{1-k}\right)\frac14 \\
&= \frac12 - \frac{1}{3\cdot2^k}.
\end{aligned}
$$

| Probes $k$ | Idealized public success |
|---:|---:|
| 0 | 16.67% |
| 1 | 33.33% |
| 2 | **41.67%** |
| 3 | **45.83%** |
| 4 | 47.92% |
| 5 | 48.96% |

Structural rejection and budget exhaustion perturb these exact percentages,
so they are an explanatory calculation, not substitutes for executing retained
worlds. Their alignment is nevertheless diagnostic: the dense model takes
about 1.94 probes and scores 41.1%
([THEORY-PHASE.md:105](THEORY-PHASE.md#L105)), almost exactly the two-probe shortcut
regime.

This does not by itself prove which internal rule the neural model implements.
It proves that the benchmark exposes a much simpler public rule whose expected
performance lands in the same band. A per-hypothesis confusion audit would be
the direct model-level confirmation.

## 4. Serialized-observation-only execution

The CPU audit in
[public_information_ceiling.py](step1/python/step1_experiments/public_information_ceiling.py)
implements a policy with one input: the exact token prefix framed as the
evaluator frames it. It:

- decodes only the current rendered observation from that prefix;
- obtains candidate actions only from the serialized `AVAILABLE` line;
- emits a rendered action string through the public parser;
- never receives `Instance`, the table, `probe_cost`, truth, `consistent()`,
  teacher targets, `valid_actions()`, or a replay key; and
- uses the privileged verifier only after termination for scoring.

The policy inspects an available probe until evidence becomes mixed, no probe
is affordable, or its uniformly safe four-probe cap is reached. It then applies
the public rule above. The cap follows from the accepted-family constraints:
`min_depth >= 2` rules out every one-probe identifying set, while generation
adds one irreversible-commit step and two slack steps, so every retained
instance has `step_limit >= 5`
([generate.rs:517](step1/crates/world/src/generate.rs#L517)). Four inspections
followed by a commit therefore cannot lose solely by crossing an unrendered
step limit. The policy boundary and forbidden inputs are recorded in code at
[public_information_ceiling.py:91](step1/python/step1_experiments/public_information_ceiling.py#L91)
and
[public_information_ceiling.py:247](step1/python/step1_experiments/public_information_ceiling.py#L247).

The retained evaluator constructs one independently seeded one-episode `Batch`
per row, not one accepted-instance stream. The audit matches that rule exactly:

```text
Batch(seed = validation_seed + episode_index, n_episodes = 1)
```

On `validation_seed = 21260811` and 1,024 episodes:

| Public policy | Success | Wilson 95% interval | Mean spend | Mean steps |
|---|---:|---:|---:|---:|
| first serialized available probe | **483/1,024 = 47.17%** | 44.13–50.23% | 4.914 | 3.595 |
| lowest numeric available probe | **485/1,024 = 47.36%** | 44.32–50.43% | 4.959 | 3.611 |

The first row is the primary result because it uses no numeric preference beyond
the serializer’s offered order. The second is a tie-breaking sensitivity check.
Both are constructive, generalizable policies frozen independently of the
validation truths. They establish that the learner-visible bytes themselves
support performance near the observed band through the unintended shortcut.

On 100,000 additional evaluator-matched worlds beginning at seed 31,260,811,
the primary policy scores **46,254/100,000 = 46.254%** (Wilson 95% interval
45.945–46.563%), with mean spend 4.899 and mean 3.592 steps. This is the retained
population estimate. A paired 20,000-world probe-selection ablation is tied
within noise: 45.91% for first-rendered versus 45.95% for lowest numeric.
The complete retained measurements are in
[public-information-policy-audit-seed0.json](step1/audit/world/public-information-policy-audit-seed0.json).

They do **not** establish that 47.17% is the mathematical optimum of the raw
serialized channel. That distinction is essential for the next section.

## 5. Why 50% is not a raw-string upper bound

Under a nuisance-free presentation, `cause_3`–`cause_6` are exchangeable after
the two special labels are separated. The corresponding semantic evidence
channel has a 50% upper bound:

$$
\frac{2}{6}\cdot1 + \frac{4}{6}\cdot\frac14 = \frac12.
$$

The deployed raw serializer, however, is not nuisance-free. Its action-order
seed folds in:

- the root seed and instance index;
- the full evidence table; and
- the complete probe-cost vector.

See [render.rs:356](step1/crates/world/src/render.rs#L356). The resulting
permutations are printed in every `AVAILABLE` line. Although the ordering
function deliberately excludes `truth`, it is a lossy fingerprint of hidden
instance state. Direct truth-independence is weaker than conditional
independence once returned evidence is also observed.

This makes a naive “exact serialized-observation oracle” particularly easy to
misinterpret. On the 1,024 validation reset observations:

- there are **1,022 distinct raw strings**;
- 1,020 of those strings are singleton buckets, so only four episodes share a
  reset string with any other episode;
- a majority-truth lookup built and scored on that same support gets
  **1,023/1,024 = 99.90%**.

The exact finite-support dynamic program in
[finite_support_dp.py](step1/python/step1_experiments/finite_support_dp.py)
goes one step further. It groups particles only by the exact causal-LM prefix,
gets legal strings only from `AVAILABLE`, recreates and replays only public
actions, and reads only terminal correctness. It never calls `Instance`, the
table, costs, truth, `consistent()`, `valid_actions()`, teacher targets, or a
replay key. For each exact-prefix bucket (B_x), it computes the integer number
of successes by backward induction:

\[
N^*(B_x)=\max_a\left[
\sum_{i\in B_x:\,a\text{ terminates }i}r_i(a)
+\sum_{x'}N^*(B_{x'}(B_x,a))
\right],
\]

where the successor buckets are formed only by equality of the next exact
serialized prefix. On this declared 1,024-world support it obtains
**1,024/1,024 = 100%**. The policy immediately commits on 1,022 worlds; the one
different-truth collision pair is split by `inspect(probe_1)` and then commits
to `cause_1` or `cause_5`. Its retained policy has 1,024 decision nodes, two
edges, maximum depth one, and SHA-256
`69e15e26bc6b06b8d2dcc546eaeaf5ddead06d4e0151c53dbe48234a2520ce03`.

Both 99.90% and 100% are deliberately **transductive diagnostics**, not
legitimate population ceilings. They memorize near-unique validation
fingerprints and have no claim on new seeds. This demonstrates why grouping a
fixed validation support, reading its terminal rewards, and calling the result
the “real public-information ceiling” answers the wrong statistical question
even though the runtime lookup formally accepts only serialized observations.

The disjoint-support check in
[raw_fingerprint_diagnostic.py](step1/python/step1_experiments/raw_fingerprint_diagnostic.py)
recovers fit labels only by trying public terminal commitments, freezes the
lookup, and then scores a disjoint evaluator-matched seed band. With 20,000 fit
and 20,000 test worlds:

- the exact raw lookup gets **97.78%** on its own fit support;
- only **5.11%** of held-out prefixes match a fit key;
- accuracy on those matched held-out keys is **16.73%**, consistent with
  six-way chance;
- raw lookup plus canonical budget fallback gets **16.82%**, versus **16.85%**
  for the canonical budget-only baseline.

Thus the raw ordering is a severe memorization hazard but shows no reusable
reset-time truth signal in this diagnostic. This does not prove that every
possible later-prefix policy has zero ordering signal, so it is not used to
turn the canonical 50% bound into a raw-string bound. It does show that the
generalizing 40–45% shortcut is the biased evidence distribution, not a
learned reset-order lookup. The retained raw-order result is in
[raw-fingerprint-diagnostic-seed0.json](step1/audit/world/raw-fingerprint-diagnostic-seed0.json),
and the exact finite-support result is in
[transductive-empirical-optimum-seed0.json](step1/audit/world/transductive-empirical-optimum-seed0.json).

Accordingly:

- 50% is a justified upper bound only for a canonicalized evidence channel, or
  under an explicitly demonstrated prior in which the raw action permutation
  is independent nuisance;
- 47.17% is a measured lower bound from a valid prefix-only policy on the exact
  evaluator cohort, with a 46.254% estimate on 100,000 additional worlds; and
- a scientifically meaningful population optimum of the current **raw**
  serialized channel has not been established. It requires a declared seed
  prior, a frozen policy fit on disjoint support, and out-of-sample evaluation
  of whatever information the fingerprint carries.

The corrected experiment should remove the fingerprint rather than invest in
an oracle that learns to invert it.

## 6. What the earlier experiments now mean

### 6.1 Conclusions invalidated or materially weakened

| Earlier conclusion | Revised status |
|---|---|
| The 97.7% “truth-blind” value is the learner ceiling. | **Invalid.** It is the value of a table-aware agent with exact costs and horizon. |
| The truth-blind expert is “an expert a learner can actually follow.” | **Invalid.** Its action is not generally a function of the learner prefix. |
| Failure to imitate that expert falsifies demonstrator followability as the bottleneck. | **Invalid.** The replacement demonstrator still used unavailable instance information. |
| Teacher success at 100% proves the public task is nearly solvable. | **Invalid.** It proves an augmented-information task is solvable. |
| Dense 41.1% is evidence of hypothesis-elimination process competence. | **Not established.** It is consistent with the much simpler stable-label shortcut. |
| The remaining gap is primarily premature commitment, capacity, or optimization. | **Not identified.** Those may be secondary effects, but the world/interface error must be removed first. |
| Two or three probes provide 66.7% or 100% identification from binary information alone. | **Invalid for the learner boundary.** That argument silently assumes knowledge of how observations map to hypothesis labels. |
| Cross-rendering gains establish transfer of the intended process. | **Weakened.** They establish adaptation benefits on the actual shortcut-bearing family unless the process is separately isolated. |

The later truth-blind arms—37.1% for dense imitation and 35.1% for
learner-conditioned imitation—remain correctly measured. What changes is their
interpretation: they show that a 19M model did not clone a table-conditioned
policy from table-free inputs, not that a genuinely public demonstrator was
unhelpful.

### 6.2 Findings that remain valid

| Finding | What remains warranted |
|---|---|
| Dense 41.1%, shuffled 16.1%, and all recorded run curves | The measurements and audited provenance remain valid for the deployed family. |
| Dense beats the target-shuffled control | The model uses state-correlated information beyond grammar. It does not identify which information or prove table-based elimination. |
| Teacher-forced NLL can disagree with closed-loop behavior | Strengthened by this audit; prior privileged actions make the mismatch more interpretable. |
| RLVR reduced illegal sampled actions without moving greedy success | Valid for the actual policy and interface. It remains an action-reliability result. |
| Cold RLVR could not bootstrap an ungrounded byte interface | Valid for that training/evaluation setup. |
| Seven-hypothesis and reversible evaluations expose severe brittleness | Valid descriptions of this learned policy’s extrapolation and legality failures. |
| Parser, executor, replay, artifact, and throughput audits | Operationally valid; they do not depend on the scientific interpretation of the evidence channel. |
| Teacher 100% and table-aware truth-blind 97.7% | Valid values for their respective privileged-information MDPs, provided they are labeled that way. |

The strongest defensible revision to the synthesis is therefore:

> Dense training learned a state-dependent policy for the deployed generated
> family. The current evidence does not establish acquisition of the intended
> hypothesis-elimination procedure because the learner lacked its required
> instance semantics and the generator exposed a simpler stable-label cue.

## 7. Corrected experiment design

There are two coherent tasks to choose from. They should not be blended.

### Option A: test active hypothesis elimination

Make the counterfactual model learner-available. For each episode, serialize or
otherwise ground:

- the relation between hypotheses, probes, and possible evidence;
- each probe’s declared cost;
- the usable step horizon; and
- stable meanings for every hypothesis identifier.

The table can be rendered as a compact preamble, supplied through a standard
structured schema, or learned across episodes only if the same latent mapping
persists long enough to be identified. A new independent table cannot be both
hidden and expected to support instance-specific counterfactual planning.

### Option B: test privileged policy distillation

Keep the table hidden, but define the objective honestly as learning the Bayes
policy after marginalizing that table. Measure the conditional entropy of
teacher targets given the public prefix, use distributional/set-valued targets
where the action is underdetermined, and compare against an observation-bound
Bayes policy rather than the table-aware teacher.

For the project’s stated novelty, Option A is the cleaner next experiment.

### Required code corrections

1. Remove the hypothesis-conditioned rewrite in `sample_evidence_table`.
   Evidence draws must not depend on the numeric hypothesis ID unless stable
   label semantics are an explicit part of the task.
2. Make action order a function of public state only—preferably a canonical
   order—or sample an independent presentation permutation that is not derived
   from hidden table or cost fields.
3. If arbitrary surface renaming is required, apply independent, recorded
   permutations to semantic identifiers after generation. Do not alter the
   semantic distribution conditional on their numeric IDs.
4. Expose untried costs and the horizon if the target expert uses them.

## 8. Pre-GPU gates

No further GPU run should be launched until all gates below pass on CPU. This
is in addition to the authorization and audit-receipt requirements in
[EXPERIMENT-EXECUTION-PLAN.md](EXPERIMENT-EXECUTION-PLAN.md).

### Gate 1 — one-input public oracle

The policy callable must have the effective interface:

```text
choose(exact_serialized_prefix) -> rendered_action
```

The policy may know declared family constants. It must not receive a world
object, replay key, instance index, table, cost vector, truth, teacher targets,
or `consistent()` result. The harness may parse and execute its returned string
and consult the verifier only after termination.

### Gate 2 — evaluator-matched, out-of-sample ceiling

- Freeze the policy before examining validation outcomes.
- Derive/tune it on a disjoint seed band.
- Use the evaluator’s exact one-seed-per-episode construction.
- Report success, confidence intervals, cost, steps, and protocol failures.
- Replicate on at least one additional disjoint seed band.

A finite-support lookup scored on the support that created it is explicitly
forbidden as a ceiling estimate.

### Gate 3 — label-permutation equivariance

For arbitrary permutations of hypothesis labels and evidence-symbol labels:

- transform the instance and rendered action names consistently;
- verify the accepted-instance distribution is unchanged up to that
  permutation;
- verify policy value is unchanged; and
- audit empirical $P(E=e\mid H=h)$ across labels.

The current sampler must fail this gate, ensuring the test actually detects the
known defect.

### Gate 4 — nuisance-free rendering

Run a paired raw-versus-canonical rendering audit. Canonicalization must remove
all dependence of list order on evidence tables, costs, seed, and index. Values
should agree within sampling error. If they do not, the surface still carries a
hidden-state side channel.

### Gate 5 — target realizability

Canonicalize declared nuisance fields, group states by exact public prefix, and
measure:

- the number of different expert targets per group;
- $H(A_{\mathrm{expert}}\mid X)$;
- disagreement between public and privileged experts; and
- whether every claimed deterministic target is actually a function of $X$.

Near-unique raw strings do not satisfy this gate; they must first be stripped of
instance fingerprints. If targets remain genuinely set-valued, train against
the set or conditional distribution instead of selecting an arbitrary point.

### Gate 6 — teacher-forcing side-channel control

Compare:

1. ordinary teacher-conditioned prefixes;
2. prefixes whose earlier actions come from the public policy; and
3. learner-conditioned prefixes with evaluator-matched failure handling.

The comparison must isolate whether low NLL depends on prior privileged actions
that disappear during closed-loop play.

### Gate 7 — decisive CPU ablation matrix

Before training, execute public and privileged policies in this matrix:

| Evidence sampler | Table/cost/horizon | Action order |
|---|---|---|
| current biased sampler | hidden | raw fingerprinted |
| unbiased sampler | hidden | canonical |
| unbiased sampler | visible | canonical |
| current biased sampler | visible | canonical |

Expected diagnostic outcomes:

- removing the sampler bias while keeping a fresh table hidden should drive the
  semantic public policy toward label-symmetric chance;
- revealing the table and decision-relevant costs should restore a high public
  ceiling; and
- raw versus canonical ordering should quantify any residual fingerprint gain.

Only the unbiased, nuisance-free condition whose public oracle clears the
declared success threshold should proceed to model training.

## 9. Decision and ownership boundary

The current STEP 1 artifacts should be retained and labeled as results from
`world-0.1.0`, not silently regenerated. Their provenance remains useful for
understanding shortcut learning, teacher forcing, action legality, and
representation acquisition.

The next world version should amend the scientific contract before any new
training:

1. decide whether the evidence table is visible, stable, or marginalized;
2. remove identifier-conditioned evidence sampling;
3. remove hidden-state-derived presentation ordering;
4. compute and retain an observation-bound ceiling receipt; and
5. state separately the values of public, table-aware, and truth-aware experts.

That separation restores the experiment’s central causal question. Without it,
another optimizer or another 100M-token run would measure adaptation to the same
mis-specified information channel rather than the intended developmental
process.

## 10. CPU reproduction

From `step1/python`, the retained audits are reproduced with:

```text
python -m step1_experiments.public_information_ceiling --config ../configs/kaggle/t4x2_dense_seed0.toml
python -m step1_experiments.finite_support_dp --config ../configs/kaggle/t4x2_dense_seed0.toml
python -m step1_experiments.raw_fingerprint_diagnostic --config ../configs/kaggle/t4x2_dense_seed0.toml
python -m unittest tests.test_public_information_ceiling tests.test_finite_support_dp tests.test_raw_fingerprint_diagnostic
```

The policy audits require only the compiled world extension and the Python
standard library; their framing constants live in the dependency-free
[protocol.py](step1/python/step1_experiments/protocol.py), and the shared Rust
family adapter lives in
[world_config.py](step1/python/step1_experiments/world_config.py). No GPU
experiment was launched for this audit.
