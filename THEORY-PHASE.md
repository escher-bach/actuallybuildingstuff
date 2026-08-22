# The Theory Phase

## Why STEP 1 came out the way it did

### Status of this document

> **Superseded for experimental direction, 2026-08-20.** The later
> [information-boundary audit](STEP-1-EXPERIMENT-DESIGN-FAILURE.md) invalidated
> the premise that the STEP 1 numbers measured the intended public decision
> problem. This remains a historical record of hypotheses formed before that
> audit. Its recommendation to continue on the existing family is withdrawn;
> see [STEP-1-WORLD-0.1-CLOSURE.md](STEP-1-WORLD-0.1-CLOSURE.md).

This opens the theory phase required by
[STEP-1-SYNTHESIS.md](STEP-1-SYNTHESIS.md#9-the-next-stage-is-a-theory-phase).
It is an explanatory document. It does not specify a world, authorize a run, or
revise [README.md](README.md).

It is written against the six questions the synthesis set for this phase, but it
does not answer them in order, because they are not independent. Four of them
follow from one mechanism.

Nothing here requires a GPU. The measurements it proposes in §9 are CPU replay
audits over shards and checkpoints that already exist.

### Decision after review

This document has done the main job assigned to the theory phase: it turns the
STEP 1 results into discriminating predictions. The next experiment should test
the learner-conditioned explanation on the existing world family. The byte
tokenizer, rendering contract, model family, and world parameters remain fixed
for that comparison.

That choice is deliberately narrower than several proposals below. The project
has built only one world family, so it does not yet know enough to redesign a
tokenizer or declare the protocol structure that every later world must share.
Those are cross-world design questions. They are deferred until evidence from
more than one world makes their requirements concrete.

The strongest language in this document should also not be mistaken for
evidence. The current results support the teacher-state/learner-state
distribution mismatch as a leading explanation; they do not establish it as a
single cause of every STEP 1 result. Sections 2--7 contain a mixture of measured
facts, proposed interpretations, and design rules. Section 8 is the contract
that separates them: when a prediction fails, the corresponding interpretation
must be weakened or abandoned.

---

## 0. Why a theory phase at all

The project has a foundations document, a gradient paradigm, a correspondence
procedure, a process/rendering boundary, and a world-algebra proposal. That is a
great deal of writing. It is nonetheless not yet a theory, in the only sense
that matters here.

A vocabulary lets you *describe* an outcome after you have seen it. A useful
theory also rules out at least some outcomes before they are observed.

Test the current foundations against STEP 1's actual results. Dense training
reached 41.1%. Suppose it had reached 85%, or 20%. Suppose transfer had been
zero-cost, or absent. Suppose RLVR had improved the decision policy, or degraded
it. In every one of those cases the foundations would have supplied a fluent
description — acquisition bias, interface cost, representation uncertainty,
insufficient correspondence, learner-relative guidance. The vocabulary is broad
enough that these results alone do not choose among its explanations. Concrete
predictions are needed to make that choice.

So the deliverable of this phase is not more distinctions. It is:

> A mechanism that explains the STEP 1 numbers, and that would have been wrong
> if some of them had come out differently.

The phase ends when the project can state a small number of predictions whose
failure would require abandoning the explanation, not relabelling it.

There is a second, blunter reason to do this now. STEP 1's headline metric was
closed-loop success rate. Two quantities sitting in the retained reports were
never used, and both of them change what the result means. A theory phase that
began by designing the next world would have carried the misreading forward.

---

## 1. The evidence, reassembled

The synthesis built its account on success rates. The table below adds the
protocol-behaviour columns and the generator constants, which are where the
explanatory content actually is. All figures are from
[TARGET-SHUFFLED-CONTROL-REPORT.md](TARGET-SHUFFLED-CONTROL-REPORT.md),
[RLVR-STAGE-REPORT.md](RLVR-STAGE-REPORT.md), and
[`step1/configs/kaggle/`](step1/configs/kaggle/).

### 1.1 The dense model, by evaluation set

| set | success | malformed | invalid | note |
|---|---:|---:|---:|---|
| validation (Rendering A, in-distribution) | 41.1% | 0.000 | 0.028 | chance floor 16.8% |
| structural (`n_hyp` + 1) | 4.8% | **0.297** | **0.361** | chance floor 14.3% |
| reversible control | 15.1% | 0.007 | **0.327** | chance floor 16.8% |
| Rendering B (zero-shot) | 0.0% | — | — | never grounded |

### 1.2 The dense model's own play, on validation

| quantity | value |
|---|---:|
| mean steps per episode | 2.94 |
| mean spend | 3.88 |
| teacher-forced action NLL | 0.0853 |

The target-shuffled control fixes the units. It commits immediately, and records
mean steps 1.00, mean spend 0.00. So a commitment is one step costing nothing,
and spend is probe cost only. The dense model therefore takes **about 1.94
probes and then commits**, at a mean probe cost of 2.0 against a declared range
of 1 to 3.

### 1.3 The generator constants

Every config in the step — all three dense seeds, the shuffled control, both
transfer seeds, both terminal-transfer seeds, all five RLVR arms, the decoding
diagnostic — declares the same world:

```toml
n_hyp = 6
n_probe = 5
n_evidence = 2
cost_lo = 1
cost_hi = 3
budget_slack = 1
min_depth = 2
step_slack = 2
```

Not one of these varied, in any arm, at any point in the step. `n_evidence = 2`
means each probe returns one of two symbols.

### 1.4 The facts an explanation must cover

1. In-distribution success sits at 41.1% against a 100% teacher.
2. On one extra hypothesis, success falls to 4.8% — *below* the 14.3% chance
   floor — and the model emits malformed strings 29.7% of the time.
3. On the reversible control, one third of its actions are well-formed but
   illegal.
4. Outcome-only RL from random weights produced exactly zero gradient.
5. Outcome-only RL after dense training moved sampled action validity and moved
   nothing else: greedy success 41.1% → 41.2%, spend 3.88 → 3.92, steps
   unchanged at 2.94.
6. Rendering B is 0% zero-shot from both arms; prior A training moves the onset
   of B competence earlier; both arms converge to the same 40–42% band.
7. After B training, A behaviour is 0% with 100% malformed A actions, while
   A-target NLL stays better than an A-naive control (9.52 against 11.60).

---

## 2. The first explanation: the model commits before the evidence can identify the answer

Take fact 1 together with §1.2 and §1.3.

Evidence is binary and deterministic. There are six hypotheses. Distinguishing
six possibilities with binary measurements requires at least
⌈log₂ 6⌉ = **3 probes**. This is an information bound, not a claim about
policies: no probe selection, however clever, extracts more than one bit per
probe.

The model takes about **1.94** probes.

If a model performs *k* binary probes and then commits, its success probability
cannot exceed min(2ᵏ, 6) / 6, achieved only with perfectly balanced partitions
and an optimal commitment in every cell:

| probes *k* | best attainable success |
|---:|---:|
| 0 | 16.7% |
| 1 | 33.3% |
| 2 | 66.7% |
| 3 | 100% |

At a mean of 1.94 probes, the ceiling is around 67%, and cannot exceed about
71% under any distribution of *k* with that mean. The model scores 41.1%.

Subject to the opening-state caveat in §2.2, this offers a provisional
decomposition of the band:

> Roughly half of the gap between 41.1% and the teacher's 100% may be the model
> stopping before the evidence *can* determine the answer. The other half is
> imperfect probe choice and imperfect readout of the evidence it did gather.

Neither half is a capacity limit and neither is a limit of the world's
difficulty. The teacher reaches 100% inside the same budget with the same
actions.

### 2.1 What this does to the headline result

The synthesis reports that the dense arm "captures about 29% of the available
headroom above chance". That measures against a ceiling the model's own
behaviour excludes. Measured against what its observed probing permits — about
67% — it captures closer to half. The model is a better *reader* of evidence
than the headline implies, and a much worse *gatherer* of it.

That inversion matters, because gathering is what the world was built to teach.
STEP 1 selected "evidence acquisition under irreversible commitment" as its
process distinction. The dense model learned to acquire some evidence and then
commit early. Premature commitment is not a peripheral inefficiency here; it is
a failure at the exact joint the world was designed around, and success rate
alone concealed it.

### 2.2 The caveat, and the measurement that settles it

The bound above assumes all six hypotheses are live when the episode begins.
STEP-1 §3.2 requires only that *at least two* are initially possible. If the
opening observation typically leaves three or four hypotheses live, two probes
may suffice and the model is not under-probing at all.

The measured 16.8% floor for random commitment does not settle this, because
that sampler commits uniformly over all six regardless of liveness.

Two privileged-verifier queries over existing validation shards resolve it
completely:

- the mean number of hypotheses consistent with the *initial* observation;
- the teacher's own mean probe count on the same episodes.

Both branches are worth knowing:

- **If the world is as hard as `n_hyp = 6` suggests,** the model under-probes,
  and §2 stands as the primary account of the band.
- **If the opening observation already narrows the field,** then the world's
  inferential demand is materially lower than the specification implies, 16.8%
  is the wrong chance baseline for a model that reads the opening state, and
  every "headroom above chance" figure in the step needs restating.

There is no reading on which this measurement is uninformative. It is the
single highest-value thing the project can currently do, and it is a CPU replay
pass.

---

## 3. The second explanation: dense imitation installed a map over the teacher's own distribution

§2 addresses the in-distribution band. Facts 2 through 7 are all
*off*-distribution, and they share one mechanism.

The leading hypothesis:

> What dense teacher traces installed is a conditional next-action map fitted to
> the distribution of states the teacher visits — not a procedure defined over
> the world's state space.

A procedure may retain useful structure when the state is unfamiliar. A fitted
map can instead fail sharply outside its training support. STEP 1's results are
consistent with the latter, but behavior alone does not prove this internal
description.

Read the evidence as tests of that distinction.

**Fact 2 — one extra hypothesis breaks the grammar.** The 29.7% malformed rate is
important and the synthesis does not mention it. A robust action grammar would
remain well formed when there are six hypotheses or seven. The observed failure
therefore shows that surface competence was entangled with the training
distribution. It does not by itself identify whether `n_hyp = 6` was literally
encoded as a constant, whether unfamiliar observations disrupted decoding, or
whether another correlated feature caused the failure.

Below-chance success follows directly. Ignorance degrades to chance;
confident extrapolation off a fitted support degrades below it. The shuffled
control, which reads nothing, sits at 14.1% ≈ 1/7 precisely because it has no
support to fall off.

**Fact 3 — a third of actions illegal on the reversible control.** The
reversible variant is rendered near-identically and is *strictly easier*: it
permits revision. A learner carrying a procedure for gathering evidence would do
at least as well there. Instead it plays illegally a third of the time.

This has a direct methodological consequence. The 15.1% reversible score was
read across the step as evidence about semantic discrimination. It is not: it is
roughly one third protocol failure. The tuned RLVR arm makes this explicit —
cutting reversible invalid actions from 0.327 to 0.086 raised reversible success
from 15.1% to 21.2% with no change to the decision policy. **The reversible
measurement was mostly measuring interface collapse**, and the synthesis's
"semantic discrimination remains untested" is right for a stronger reason than
it states: not merely that the reversible-trained arm is missing, but that the
existing reversible number does not measure what its name says.

**Fact 4 — cold RLVR produced exactly zero.** Random weights emitted malformed
actions in every retained rollout, so this run produced no reward variance and
no gradient. This is consistent with the support account, but it establishes a
narrower fact: the tested outcome-only procedure could not cross the current
interface from this initialization.

**Fact 5 — warm RLVR moved measured validity and not the measured decision
policy.** If the policy had a readily adjustable "keep probing" disposition, an
outcome reward is a plausible signal for it, and the world is not reward-sparse
— 78% of groups carry variance. Validity is the direction that moved
consistently in the retained metrics. This is evidence for the fitted-map
account, not proof that the model has no internal direction capable of changing
probe behavior. Consistent with the account, the arm that moved the policy
hardest raised structural malformed rates from 0.297 to 0.353.

**Facts 6 and 7 — transfer and forgetting.** The fitted-map account is
consistent with zero zero-shot transfer, reduced refitting cost, the same
terminal band, and behavioral overwriting with a residual NLL trace. It did not
predict those results in advance, so they are supporting observations rather
than independent confirmations. A matched learner-conditioned experiment can
test the part of the account that makes a new prediction.

### 3.1 Why §2 and §3 may share a mechanism

They may be the same failure at two scales. A map fitted to the teacher's
trajectory distribution has no reason to represent "the evidence does not yet
determine the answer" — that proposition is never contrastively supervised,
because the teacher never occupies a state where it is true and commits anyway.
Premature commitment in-distribution is a direct candidate consequence of
imitating an optimal demonstrator with no supervision at the states the learner
itself produces. Grammar collapse out-of-distribution may have the same source,
but the current experiment does not isolate it.

This is a known property of behaviour cloning, and naming it is not a criticism
of the step. It is the attribution the synthesis deferred to this phase.

---

## 4. Consequence: instance volume inside an orbit is augmentation, not world content

The synthesis names procedural generation as the main theoretical bottleneck and
asks what counts as a new world rather than another instance. §1.3 makes the
answer concrete.

STEP 1 generated 32,768 training episodes per arm. Every one had six
hypotheses, five probes, binary evidence, costs in [1,3], and identical slack
parameters. What varied was which hypothesis was true, which evidence table was
sampled, and the identities attached to each — that is, the variation was almost
entirely a relabelling.

This suggests an operational criterion for the present family:

> **Two instances belong to the same world when a relabelling of identifiers
> carries the teacher policy of one onto the teacher policy of the other.**
> Variation within such an orbit is data augmentation. Variation that changes
> the quotient — the abstract process the labels are attached to — is world
> content.

And a hypothesis suggested by STEP 1:

> **A parameter held constant across the whole training distribution gives the
> learner no direct evidence that its policy should vary with that parameter;
> extrapolation at its boundary may therefore fail sharply.**

`n_hyp = 6` was free to hardcode, and the learner had no training evidence that
its behavior should depend smoothly on hypothesis count. This makes the
`n_hyp = 7` failure unsurprising, without proving how the constant was represented
internally.

Two things follow.

**The structural evaluation was an extrapolation test, not an interpolation
test.** The 4.8% result is valid evidence that this trained policy does not
extend to `n_hyp + 1`. It should not, by itself, be cited as evidence that the
learner cannot acquire the process or generalize within a structurally varied
training distribution. The training variation and held-out set were not
commensurable for that stronger claim.

A cleaner future design would vary selected structural parameters during
training and hold out combinations inside the varied ranges. That would measure
interpolation over known dimensions separately from the existing extrapolation
test. It is not part of the next learner-conditioned comparison.

**More instances do not by themselves demonstrate structural competence.** README §5.1's
hidden-permutation illustration is precisely an orbit-sampling scheme, and it is
explicitly flagged there as supplying surface variation rather than an
indefinitely generable world. STEP 1 is the empirical confirmation of that
warning. Additional samples from the same orbit add coverage and optimization
signal, but do not introduce evidence about a dimension that never moves.

This sharpens the question the synthesis asked. It is not "which variations
force reusable organization?" in the abstract. It is:

> Which parameters must *vary during training* for the learned policy to be a
> function of them at all — and which of those are the ones that carry the
> correspondence claim?

Anything held fixed is not demonstrated to the learner as a variable. That is a
useful design warning, not a universal claim about what a model can infer from
its inductive biases.

---

## 5. Consequence: syntax is three layers, and STEP 1 fused two of them

The synthesis asks which parts of syntax are stable infrastructure, which are
learnable content, and which are developmental targets. The evidence separates
them by how they behave under change.

**L1 — Protocol.** The token-level contract: turn structure, delimiters, where
an action goes, what a malformed emission looks like. Facts 2, 3 and 4 all show
this gating everything else. It should remain fixed and versioned for controlled
comparisons within STEP 1. What can be shared across later worlds cannot yet be
decided from a single world family.

**L2 — Lexicon.** The surface names for typed objects. This is legitimate
learnable content, and Rendering A/B is a legitimate instrument for it. But fact
7 shows that *sequential* exposure teaches overwriting, not invariance — and it
could not have taught anything else, because nothing in a sequential loss ever
requires the two renderings to share a representation.

**L3 — Notation as content.** Where the representation determines which
distinctions are cheap to express. STEP 1 never touched this, and it is the only
one of the three that is a genuine developmental target.

One plausible interpretation is that **STEP 1's byte tokenizer made L1 and L2
harder to separate.** With little lexical prior, protocol and lexicon enter the
same string-completion problem. The synthesis reads byte-level tokenization as
making a real cost "visible in an unusually severe form"; it may also have
contributed to the coupling it revealed.

That is a hypothesis, not a conclusion. It is not an actionable tokenizer
proposal now: changing the tokenizer would confound the selected
learner-conditioned comparison, and no second world yet exists from which to
infer a general lexical or protocol structure. The retention result licenses
only the narrower conclusion that semantic and interface competence were
entangled under this representational choice and training schedule.

---

## 6. Consequence: correspondence needs matched controls, not absolute transfer

STEP 1 ran many measurements. Exactly one of them was decisive: the
target-shuffled control, which held the entire data stream fixed and destroyed
one relation. Every absolute number in the step was ambiguous until that control
existed, and the control's own report says so.

Generalize this into the phase's methodological commitment:

> **Whenever a causal or correspondence claim is practical to isolate, prefer a
> control that differs in the claimed variable over an absolute score against a
> floor.**

Applied to correspondence, this fixes the weakest link in the chain from
[WORLD-CORRESPONDENCE.md](WORLD-CORRESPONDENCE.md) and the transfer-consequence
admission test in
[principled_developmental_worlds.md](principled_developmental_worlds.md#94-transfer-consequence).
That test currently reads: training on world W should reduce the experience
needed to learn practice P. STEP 1 shows why that is not enough — a reduced
acquisition cost is exactly what byte-level statistical overlap would also
produce, and the synthesis concedes it cannot presently distinguish the two.

The test should be differential:

> World W corresponds to practice P when training on W reduces the acquisition
> cost of P **more than** training on W′ does, where W′ matches W in surface
> form, token statistics, episode length and action vocabulary, and differs only
> by breaking the claimed abstract relation.

W′ is the target-shuffled control lifted to the correspondence layer. STEP 1
supports this as a stronger test than absolute transfer. Whether every later
world admits a surface-matched control is a design question, not something this
single family settles.

---

## 7. Consequence: four axes, four instruments, and success rate is not decision quality

The synthesis asks that acquisition cost, retention, semantic discrimination and
final capability be measured separately. The step provides the empirical
argument for *why* — each of these moved while the others stayed still.

| axis | instrument | STEP 1 evidence that it is independent |
|---|---|---|
| interface availability | malformed and invalid rate, reported per evaluation set | 0.000 malformed on validation and 0.297 on structural — one model, one checkpoint |
| acquisition cost | adaptation-curve onset; SDL after declared calibration | prior A training moved onset; terminal band unmoved |
| retention | closed-loop score on the prior rendering, *plus* NLL trace | behaviour 0%, distributional trace survives — total dissociation |
| decision quality | success conditioned on well-formed legal play | RLVR moved sampled validity with greedy decisions unchanged |

The operational rule that follows is narrow and immediately binding:

> **A raw success rate is not a measurement of decision quality unless protocol
> failure is separated out.** Report success, and success conditioned on legal
> play, and the protocol failure rate, on every evaluation set.

Had this rule been in force during STEP 1, the reversible number would not have
been read as evidence about semantics, and the structural number would not have
been read as evidence about process generalization. Both were dominated by a
term nobody was looking at.

---

## 8. What the hypotheses predict

The account in §2–§3 is useful only where it can be wrong. These predictions
are deliberately separated so that a result is not credited to two changes at
once.

**P1 — Learner-conditioned supervision changes the learned policy.** At a
matched training budget on the existing world family, adding supervision at
states reached by the learner should reduce premature commitments and improve
closed-loop success conditioned on legal play. Teacher-forced NLL need not
improve with it. This is the selected next experiment.

*Evidence against the hypothesis:* the matched learner-conditioned arm reaches
the same probe-count distribution, legal-conditioned success, and raw success
band as the teacher-conditioned control. That would show that teacher/learner
state-distribution mismatch is not the main explanation of the band. It would
not, by itself, prove whether capacity, optimization, or world difficulty is
responsible.

**P2 — Capacity is not the first-order limitation.** Scaling the model under the
unchanged teacher-conditioned regime should move the band less than changing
the supervised state distribution at matched conditions.

*Evidence against the hypothesis:* scale moves the policy metrics materially
more than learner-conditioned supervision. This comparison is deferred; it is
not needed to run P1.

**P3 — Learner conditioning improves on-policy protocol recovery.** Within the
existing six-hypothesis family, malformed or invalid learner actions should
become less persistent after their resulting unchanged states and corrections
are included in training.

This prediction does **not** extend to the unseen `n_hyp = 7` structural set.
Learner conditioning without structural variation supplies no examples of that
change, so a lower structural malformed rate would be welcome evidence, not a
required result. The earlier version of P3 overclaimed this implication.

**P4 — Interleaving may reduce rendering-specific forgetting.** At matched
exposure, an interleaved A/B schedule should retain more closed-loop A competence
than sequential A-then-B training.

This does not predict zero-shot competence on a third rendering C and does not
identify the tokenizer as the cause. Both claims require knowledge about
rendering and protocol structure that one world family cannot provide. This
experiment and any tokenizer change are deferred.

**P5 — Structural variation is necessary evidence for structural
interpolation.** If selected parameters such as `n_hyp` and `n_probe` vary in a
future training distribution, held-out combinations inside those ranges should
perform better than comparable extrapolation beyond a held-fixed value.

*Evidence against the sufficiency of this account:* adding well-covered
variation fails to improve interpolation or malformed rates. That would not
make the existing structural evaluation meaningless; it would show that varying
the parameters is not enough for this learner to organize them compositionally.

P1 and the on-policy part of P3 form one controlled experiment. P2, P4, and P5
change capacity, schedule, or world distribution and must remain separate.

---

## 9. A cheap audit that sharpens the selected experiment

One CPU replay audit over existing shards and the retained dense seed-0
checkpoint would sharpen the interpretation and primary metrics of P1. It can be
done while the learner-conditioned comparison is specified; it is not a
different experimental direction and need not delay implementation.

For the validation set already used:

1. Mean number of hypotheses consistent with the **initial** observation. This
   tests the assumption behind §2.2 and determines whether the existing
   "headroom above chance" figures need restating.
2. The **teacher's** mean probe count and mean spend, which STEP 1 reports for
   the model but never for the teacher.
3. For each failed episode, whether the evidence the model gathered **determined
   h\*** at the moment it committed. This splits the residual gap cleanly into
   premature commitment, poor probe selection, and misreading of gathered
   evidence — three different failures with three different fixes, currently
   summed into one number.
4. Per-decision agreement with the teacher, separated by whether the prefix is
   on or off the teacher's own trajectory. This is the direct measurement of the
   §3 mechanism.

None of this needs a GPU, an authorization, a tokenizer change, or a new world.
It is arithmetic over artefacts the project already retains. The learner-
conditioned run remains subject to the explicit GPU authorization required by
[EXPERIMENT-EXECUTION-PLAN.md](EXPERIMENT-EXECUTION-PLAN.md).

---

## 10. What this does not explain

Stated so the phase does not quietly assume it has finished.

- **Whether the learner could represent the irreversible/reversible distinction
  at all.** §3 explains why the existing measurement cannot tell us. It does not
  predict the answer. That still needs a reversible-trained arm, and §7's
  conditioned metric before it is interpretable.
- **What organization actually transferred between renderings.** §3 says the
  transferable object is whatever is encoding-invariant in a fitted map. That is
  a shape, not an identification. SDL localizes it in training time; nothing yet
  identifies it.
- **Whether any of this corresponds to a real practice.** §6 supplies a test.
  No world in the project has passed it, because no world has been asked to.
- **Why seed 1's transfer effect was an order of magnitude smaller than seed
  0's.** Two seeds cannot distinguish a heavy-tailed effect from a fragile one,
  and the account here predicts nothing about the variance.

The first three are the substance of the phase's remaining work. The fourth is
a question about statistics, and should be resolved by more seeds or dropped,
not theorized about.

---

## Summary

The leading mechanism is that dense imitation of an optimal teacher fits the
teacher's state distribution and leaves learner-induced states poorly
supervised. On the current reading of the world, premature commitment may
account for a substantial part of the in-distribution gap. The replay audit in
§9 tests the assumptions behind that decomposition.

Off-distribution, the policy also loses protocol competence. Those failures are
consistent with the same support account, but the present evidence does not show
that one mechanism causes the structural, reversible, transfer, forgetting, and
RL results. Claims that it does are interpretations to test, not findings.

Four narrower lessons remain useful. Fixed parameters provide no training
evidence about how behavior should vary with them. Protocol failure must be
reported separately from decision quality. Correspondence claims benefit from
matched broken-relation controls. And the byte tokenizer is one possible
contributor to interface coupling, not a component to redesign before the
project has evidence from another world.

P1 is the selected next experiment, with the on-policy protocol prediction in
P3 measured in the same run. P2, P4, and P5 are separate future tests. The CPU
audit sharpens P1 without changing its direction; no GPU run is authorized by
this document.
