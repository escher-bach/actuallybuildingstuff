# STEP 1 in Context

## Evidence synthesis and completion review

**Review date:** 2026-08-17  
**Scope:** the STEP 1 specification, foundational documents, stage reports,
standard-stack migration, execution workflow, and retained audit records.

This document interprets the evidence. It does not replace
[STEP-1.md](STEP-1.md), the stage reports, or the operational authority of
[EXPERIMENT-EXECUTION-PLAN.md](EXPERIMENT-EXECUTION-PLAN.md).

## Executive verdict

STEP 1 has produced a real and coherent result. Dense teacher traces taught the
model more than an action syntax: on held-out worlds, destroying only the
state-to-action correspondence reduced success from **41.1% to 16.1%**, the
chance-commitment floor. Prior training under one rendering also demonstrated
that useful behavior under a second rendering can sometimes be acquired much
earlier. Seed 0 is the strong witness; seed 1 preserves the sign at the first
budget but shows how seed-sensitive the magnitude is. Outcome-only RLVR, by
contrast, could not bootstrap through the
ungrounded byte-action interface and, after dense training, improved the
reliability with which the model expressed an existing policy without improving
the underlying evidence-gathering decisions.

The most useful combined interpretation is:

> Dense world supervision installed a state-dependent behavioral organization
> that made related behavior cheaper to acquire under a new representation.
> It did not install representation-independent knowledge, robust structural
> generalization, or a demonstrated understanding of irreversibility.

This is enough to satisfy the scientific completion rule in
[STEP-1.md](STEP-1.md#14-completion-criteria): the results distinguish process
learning from surface acquisition. The original matrix contains further
possible experiments, but completing a matrix is not the purpose of STEP 1.
Its purpose is to decide whether the approach has enough reality to support a
next step. It does.

The appropriate project status is therefore:

**Complete as a stepping stone. No further STEP 1 experiment is required before
moving on. The next activity is theory-building, not an automatic STEP 2 run.**

This is an internal research result, not a publication claim. The appropriate
standard is therefore not exhaustive ablation. It is whether another experiment
could change what the project builds next. The remaining reversible-arm,
grounded-RLVR, seed, and retention questions would refine the boundary of the
result, but none currently changes the next decision.

## What was actually tested

The world is a generated, finite, partially observable decision process. An
episode contains six candidate hypotheses, five costly probes, deterministic
evidence, and a limited budget. The learner must inspect evidence and eventually
commit to a hypothesis. Its text action is parsed and executed by the Rust
world, so action choice changes the subsequent state and observation.

Three different questions must be kept separate:

1. **Surface fluency:** can the model emit a parseable and legal action string?
2. **Within-family process behavior:** does the model use the observed state to
   decide whether and what to inspect or commit?
3. **Semantic process discrimination:** does the model behave differently when
   commitment is irreversible rather than reversible, and does that distinction
   transfer across renderings?

The experiments answer the first two. They do not yet answer the third. This
distinction resolves the apparent tension between the target-shuffled report's
statement that “process learning” is established and the transfer report's
statement that transfer of the process distinction remains unmeasured.

## The result stack

| Question | Best evidence | What it establishes | Important limit |
|---|---|---|---|
| Did dense training learn more than syntax? | Dense **41.1%** versus target-shuffled **16.1%** on 1,024 paired held-out worlds | About 25 percentage points require the state-to-action mapping. The shuffled model learned perfect grammar and then committed immediately at chance. | One training seed; this tests state use, not the irreversible/reversible distinction. |
| How much competence was learned? | Chance is about **16.8%**, dense is **41.1%**, teacher is **100%** | Dense training captures about 29% of the available headroom above chance. This is meaningful but far from mastery. | The dense model scores **4.8%** on the held-out structural set, below the seven-way chance floor of about **14.3%**. |
| Did prior process training help under a new rendering? | At about 3M Rendering B tokens, A-trained versus init is **19.0% vs 0.0%** on seed 0 and **2.6% vs 0.0%** on seed 1 | Prior A experience advances the onset of useful B behavior on both seeds. The sign replicates. | The magnitude is seed-sensitive, and the reversible control shows a similar early pattern, so transfer of the specific commitment semantics is not established. |
| Is this zero-shot invariance or a higher final ceiling? | Both arms are 0% zero-shot; at 100M B tokens they converge near **40–42%** | Transfer is an **acquisition bias**: less new experience is needed. It is not pre-existing knowledge of B and does not raise the converged ceiling. | The interface must be calibrated before the latent advantage becomes behaviorally available. |
| Is the old surface retained? | After about 3M B tokens, A closed-loop success is **0%** with 100% malformed A actions | The action vocabulary is overwritten quickly and completely at the behavioral level. | A-target NLL remains better than an A-naive control after 100M B tokens (**9.52 vs 11.60**), so a weak distributional trace survives; whether that trace speeds relearning is untested. |
| Can outcome-only RLVR learn from random weights? | Every cold-start rollout is malformed; group advantages, loss, and parameter change are exactly zero | The ungrounded byte action surface prevents any reward gradient from reaching the task policy. | The world itself is not reward-sparse: a well-formed unguided policy succeeds **17.2%** of the time and 78% of sampled groups contain reward variance. |
| What does RLVR add after dense training? | Best-shot sampled success **42.1% vs 35.9%**, but greedy success **41.2% vs 41.1%**, with unchanged spend and steps | RLVR removes sampling-induced illegal actions. It improves reliability of expression, not the evidence-gathering decisions measured here. | A grounded comparison could characterize a different regime—RL after supervised interface acquisition—but is unnecessary unless a later step proposes RLVR as its primary learner. |
| Is generation fast enough to use? | Audited control-run provenance records **2.06M world tokens/s** and **2.87M tokens/s** through the DataLoader; completed two-T4 runs consumed the resulting packed shards successfully. | Yes. The pipeline is operationally sufficient for this model and experiment. | The old 80%-of-raw-text target was a planning estimate with no principled derivation. Its measured 66% ratio is not a scientific failure or a blocker. Reprofile only if a later scale makes input throughput limiting. |

The important result is not any one percentage. It is the triangulation across
controls:

- dense training produces state-conditioned behavior;
- that behavior changes the early adaptation curve under a new surface;
- outcome-only optimization mainly repairs protocol reliability; and
- syntax can be perfect while process behavior remains at chance.

Together, these results locate the learned benefit between mere formatting and
general abstract knowledge.

## The nine consequences to preserve at closure

These are not nine independent conclusions. They form one diagnosis of the
current programme and define what the foundations must now explain.

### 1. The approximately 40% band is a problem, not a victory condition

At the 100M-token terminal budget, the successful dense and transfer arms occupy
roughly the same **40–42%** band. This is meaningful relative to the
approximately **16.8%** chance-commitment floor, but it captures only about 29%
of the available headroom to the teacher's 100%. Calling this “convergence” is
an operational description of the measured budget, not proof of an asymptotic
ceiling.

RLVR did not repair that limitation. The best tuned arm improved sampled
success from **35.9% to 42.1%** by suppressing illegal sampled actions, while
greedy success remained **41.1% versus 41.2%** and evidence spending and episode
length were unchanged. In this regime RL sharpened the expression of the policy
that dense training had already installed; it did not discover the missing
decision procedure. The unexplained 40% band is therefore evidence about the
training experience and learned policy, not merely about decoding noise. STEP 1
does not identify whether the remaining failure comes primarily from the
world's inferential demands, teacher-trajectory coverage and closed-loop
distribution shift, model capacity, or optimization. That attribution belongs
to the theory phase.

### 2. Syntax is part of the developmental object

The experiments expose two different representation failures that should not be
collapsed:

- **Never-grounded syntax:** zero-shot Rendering B failure is expected because
  the learner has never been shown how B strings denote typed actions.
- **Previously grounded but forgotten syntax:** after learning A and then B,
  closed-loop A behavior falls to 0% with 100% malformed A actions. That is a
  retention failure, not an ordinary zero-shot result. The better A-target NLL
  relative to an A-naive model shows that a weak trace remains, but a trace is
  not usable competence.

The byte-level tokenizer makes this visible in an unusually severe form: it
provides a universal lossless token boundary but gives the model almost no
world-specific lexical prior. STEP 1 gives no warrant for treating syntax as a
secondary nuisance: semantic competence and usable interface competence are
distinct necessary achievements. Replacing the byte tokenizer with a custom
tokenizer for every world could make an isolated result easier while defeating
the cumulative goal. A developmental learner must retain tasks and interface
competence across worlds, so tokenizer and rendering design must be shared,
stable, and measured across time.

This leaves open a real design possibility: some worlds **could** primarily
train syntax, interface induction, translation, and retention. They should not
be declared necessary in advance, and they should not be mistaken for semantic
worlds. Their value would have to be shown by cheaper later acquisition or
better retention under new renderings.

### 3. The transfer claim should remain deliberately modest

The statement of record is:

> **Faster transfer is possible.** The seed 0 result validates the possibility.

Seed 0 supplies a large early witness: **19.0% versus 0.0%** after about 3M B
tokens. Seed 1 supplies a smaller same-sign observation, **2.6% versus 0.0%**,
and therefore warns against treating the seed-0 effect size as dependable. The
shared endpoint near 40–42% and the lack of zero-shot B competence mean that the
result is about acquisition cost, not representation invariance, remembered B
knowledge, or a higher capability ceiling.

### 4. Surplus Description Length is a usable instrument—with calibration

The legacy bias framework's task-relative acquisition measure, Surplus
Description Length (SDL), survived contact with the experiment after one
important correction. The initial approximately 1M-token loss spike is the cost
of re-encoding from A into B; naively summing it gives the wrong sign. A
post-calibration cumulative surplus measure and the terminal-loss intercept both
detect the early transfer effect, localize it to the right training window, and
rank seed 0 above seed 1 without additional closed-loop evaluations.

SDL is therefore available as an instrument or controller feature, not yet as a
standalone behavioral metric. It can cheaply detect and localize likely
transfer; selected closed-loop evaluation is still required to translate its
signal into task success. The calibration boundary must be declared rather than
chosen after seeing the curve.

### 5. Procedural generation is now the main theoretical bottleneck

STEP 1 procedurally generated many instances of one hand-designed world family.
That is not the same achievement as generating many consequentially different
worlds. The approximately 40% in-family band and **4.8%** structural score say
that instance volume alone did not create mastery or broad structural transfer.

The difficult question is no longer whether code can emit unlimited valid
episodes. It is which variations force reusable organization, how those
variations compose, why they correspond to later practices, and how a learner's
progress should change what is generated next. The existing foundation already
states the right warning: generate structured interactive processes, not merely
more instances of a chosen ontology. STEP 1 makes that warning empirical.

### 6. Cold RL failure is surprising but coherent

Outcome-only RLVR from random weights produced malformed actions in every
rollout, hence zero within-group reward variance, zero advantages, zero loss,
and zero parameter change. This does not show that the world reward is
intrinsically too sparse: a well-formed unguided policy succeeds about 17.2% of
the time and usually creates usable reward variation. It shows that reward
cannot train a policy through an interface the learner cannot yet speak.

The language-pretraining analogy is direct but bounded: a weight-naive language
model also begins without the representational grounding needed for sparse
outcomes to teach much. Dense prediction or explicit interface grounding can
create the policy support on which later outcome optimization operates. The
experiment establishes this division of labor for the present byte-action
world; it does not establish a universal ordering of supervised learning and RL.

### 7. Same endpoint plus faster acquisition is a bias-like result

The legacy distinction is useful here: knowledge changes what is already
available, while bias changes the cost of acquiring something from new data.
The measured pattern—no zero-shot B success, earlier acquisition after A, and
the same terminal band—is therefore bias-like rather than knowledge-like.

This is an interpretation of behavior, not an identification of the internal
bias. The transferred object could mix decision structure, sequence priors, or
low-level byte regularities. SDL gives a task-relative way to measure the
acquisition saving; it does not by itself say which mechanism produced it.

### 8. The Kaggle CLI flow is retained infrastructure

The repository-driven Kaggle flow—render, submit, monitor, retain remotely,
retrieve compact artifacts, and verify a tracked audit receipt—worked smoothly
and should be preserved as the default execution path. Future theory may change
the worlds, curriculum, renderers, or metrics; it should not casually replace
this operational contract. As required by
[EXPERIMENT-EXECUTION-PLAN.md](EXPERIMENT-EXECUTION-PLAN.md), heavyweight
artifacts remain remote, compact evidence is retrieved, and no GPU experiment
is launched without explicit authorization.

### 9. The next stage is a theory phase

The next task is not to choose a larger generator, add a curriculum controller,
or launch another tuning run. It is to reconcile the foundations, the legacy
bias/SDL account, and the experimental results well enough to say what a next
experiment would discriminate.

The theory phase should explain at least:

1. what the 40% band says about the information and policy induced by the
   teacher traces;
2. which parts of syntax should be stable infrastructure, learnable content,
   or explicit developmental targets;
3. what kind of generated variation counts as a new world rather than another
   instance;
4. how candidate worlds are justified by correspondence to later practices;
5. how acquisition cost, retention, semantic discrimination, and final
   capability should be measured separately; and
6. which result would falsify the proposed explanation rather than merely
   motivate another tuning pass.

This is intentionally a slow step. Its output should be a revised theory and a
small set of discriminating experimental questions, not code produced to keep
the experiment pipeline busy.

## What this means for the foundations

The foundational project asks whether designed synthetic experience can install
reusable organization in a weight-naive language model. STEP 1 supports some of
that architecture and leaves the strongest claims untouched.

### Directly supported

- **Representation is learned, not decorative** ([README.md](README.md#c6--representation-is-learned)).
  The cold-start RLVR failure, the need for B calibration, immediate A-surface
  forgetting, and the seed-sensitive acquisition thresholds all show that the
  action language is a substantive part of development.
- **Interaction does not imply reward learning**
  ([README.md](README.md#c9--interaction-does-not-imply-reward-learning)). Dense
  cross-entropy targets extracted from an executable world produced the measured
  decision competence. The world dynamics and the gradient generator were in
  fact separable.
- **A world is more than a static dataset.** Closed-loop evaluation exposed
  malformed actions, illegal actions, probe cost, state-dependent choices, and
  failures that teacher-forced NLL alone concealed.
- **Shared executable semantics can underwrite multiple renderings.** The Rust
  tests verify aligned typed actions and consequence-invariant transitions even
  though the learner must separately acquire each token interface.
- **The useful transfer estimand is learning cost.** The transfer result appears
  in the onset and slope of adaptation, not in zero-shot behavior or final
  accuracy. This is the concrete STEP 1 form of the broader “bias rather than
  knowledge” thesis.

### Partially supported

- **Reusable process organization.** The target-shuffled control establishes
  within-family state use, and the B curves establish some reusable acquisition
  bias. They do not identify what internal organization transferred. The effect
  could combine decision structure, sequence priors, and low-level byte reuse.
- **Causal fidelity before surface fidelity.** The synthetic world has explicit,
  tested consequences under action. That establishes internal causal coherence,
  not correspondence to diagnosis, debugging, or another real practice.
- **High-throughput developmental experience.** The standard training path,
  packed shards, and two-T4 runs are operational. The generator and DataLoader
  were fast enough to execute the programme. The earlier 80%-of-raw-text number
  was an ungrounded engineering estimate, not a theoretical threshold; missing
  it has no bearing on the scientific result.

### Not yet tested

- **Correspondence to real activity**
  ([README.md](README.md#c4--correspondence-before-analogy)). No experiment maps
  the learned behavior into real diagnosis, debugging, tool use, or another
  target practice.
- **The selected semantic distinction.** No reversible-trained arm exists, so
  there is no evidence that the learner represents the difference between
  reversible and irreversible commitment.
- **Open-ended developmental accumulation.** STEP 1 uses one narrow world family.
  It does not test a compositional world algebra, an expanding ecology, primitive
  formation, or accumulation across many capabilities.
- **The strongest “designed childhood” claim.** The evidence is a successful
  first probe of one mechanism, not a demonstration of a general developmental
  programme.

## Completion decision

### Apparatus and implementation: complete for continued work

The declared world, teacher, verifier, renderers, tokenizer, standard model,
training path, evaluator, learner-action boundary, deterministic artifacts, and
versioned configurations exist. The current Rust workspace passes **68 tests**,
including replay, leakage, alignment, irreversible/reversible futures, teacher
validity, and learner-conditioned correction cases. Successful audited Kaggle
runs also record two T4 devices, distributed training, resume, exact
serialization, evaluation, and artifact packaging.

Two pieces of housekeeping can be completed when the later workflow needs them:

- The standard-stack plan names an `SFTTrainer` interoperability smoke that has
  not been retained. Run it when an SFT stage actually needs that path; it does
  not bear on the STEP 1 result.
- Some older source runs predate the receipt workflow. Backfill them where cheap
  to preserve provenance, but do not mistake audit housekeeping for a reason to
  rerun a scientific experiment.

### Scientific purpose: met

The target-shuffled control distinguishes the first two admissible STEP 1
conclusions: dense traces improve state-dependent process behavior, not only
surface acquisition. The transfer stage independently shows that the benefit
can reduce later acquisition cost under a new rendering.

The literal strongest version of the opening question—learning the specific
irreversibility distinction more efficiently than a grounded outcome-only
learner—was not measured. But STEP 1 has answered the decision underneath it:
dense executable-world supervision can produce genuine state-dependent behavior
from random weights, some of that training changes later acquisition under a new
surface, and outcome-only reward is not a viable bootstrap through this token
interface. That is enough to close this probe rather than continuing to
interrogate it. It is not yet enough to specify the next developmental world;
that requires the theory phase above.

### Why the remaining experiments are not closure requirements

Completed elements include dense training, held-out in-family evaluation,
structural evaluation, two-seed rendering transfer, malformed/invalid action
measurement, five RLVR configurations, a decoding diagnostic, and the
target-shuffled control.

The unrun experiments have clear meanings, but low present decision value:

- **A reversible-trained arm** would test whether the learner distinguishes a
  commitment that removes future options from one that permits revision, rather
  than merely learning generic evidence gathering. That matters if a later
  world treats recoverability as a central varying semantic feature. It does not
  change the current conclusion that dense teaching learned more than syntax or
  the decision to close this probe. Carry the contrast forward and
  test it where it becomes consequential; do not keep STEP 1 open for it.
- **A grounded dense-versus-RLVR comparison** would first give RLVR enough
  supervised calibration to speak the action language, then compare learning
  rules. That is no longer weight-naive outcome-only learning; it is RL after an
  interface curriculum. The five RLVR configurations and decoding diagnostic
  already show what RL buys in the regime actually reached: a sharper, more
  reliable sampled policy with unchanged decisions. Run the grounded comparison
  only if a future step seriously proposes outcome-only RL as its primary
  acquisition method.
- **More seeds and ablations** would estimate effect sizes and failure
  frequencies more precisely. They are appropriate for a publication or a
  deployment decision. Here the target-shuffled control is decisive, transfer's
  qualitative sign replicated on both tested seeds, and the tuned RLVR arm
  addressed the plausible objections without changing the decision. More runs
  would narrow uncertainty without changing direction.
- **The weak structural score** is a design input, not an invitation to tune this
  family until it generalizes. It says the theory phase must explain what would
  count as consequential structural variation and composition before another
  family is built.

### Audit and workflow: sufficient, with optional backfill

The new one-command Kaggle workflow has produced receipts for the shuffled
control, the successful warm-start RLVR arms, the decoding diagnostic, and the
retention/forgetting diagnostics under
[`step1/audit/runs/`](step1/audit/runs/). Those receipts bind exact Kaggle
versions, Git SHAs, configuration hashes, hardware, artifacts, and checksums.

The older dense seed runs and Rendering B terminal-transfer runs still lack
their own tracked receipts. Later audited diagnostics identity-check some of
their checkpoint hashes, and the reports retain source commits, configuration
hashes, model-state hashes, data hashes, seeds, and hardware. For internal
continuation that is enough evidence to use the qualitative result. Backfill is
worth doing if it is a cheap metadata operation; it is not a reason to rerun the
models or delay the next step.

## The strongest defensible internal conclusion

A concise formulation to carry into the next step is:

> In a generated partially observable decision world, dense supervision from a
> privileged teacher trained a randomly initialized 19.2M-parameter language
> model to make state-dependent evidence-gathering decisions above a
> syntax-matched chance control. It also demonstrated that faster acquisition
> under a second action rendering is possible: seed 0 is the strong witness,
> while seed 1 preserves the early sign at much smaller magnitude. Zero-shot and
> terminal performance remained unchanged.
> Outcome-only RL could not bootstrap through the ungrounded byte interface and,
> after dense training, improved sampled action validity without improving the
> measured decision policy.

Claims that should not yet be made:

- that the learner acquired a representation-independent process;
- that it learned the difference between reversible and irreversible
  commitment;
- that dense supervision is generally more sample- or compute-efficient than a
  properly grounded outcome-only learner;
- that the effect generalizes beyond this world family, model scale, tokenizer,
  or two transfer seeds;
- that any real-world correspondence has been demonstrated; or
- that STEP 1 validates an open-ended designed childhood.

## What STEP 1 buys the project

STEP 1 was not meant to produce a publishable benchmark result. It was meant to
buy confidence, distinctions, infrastructure, and direction for the work that
follows. It has bought all four.

1. **Confidence that the basic mechanism is real.** A randomly initialized
   standard language model can learn nontrivial, state-dependent closed-loop
   behavior from dense targets generated by an executable world. The decisive
   shuffled control shows this is not merely action grammar.
2. **Evidence for the “bias, not knowledge” account of development.** Earlier
   structured experience did not make the model know the second rendering or
   raise its final ceiling. It made useful behavior under that rendering begin
   earlier. The developmental object to measure is therefore the adaptation
   curve—the cost of later learning—not just zero-shot accuracy.
3. **A clear role for representation.** The token action surface is part of what
   must be learned. It can block reward learning completely, can be overwritten
   while a faint trace remains, and can hide transferred organization until a
   small amount of grounding makes it expressible. Stable home renderings,
   explicit bridges, replay, and syntax-focused worlds are now hypotheses to
   compare; semantic invariance alone cannot be assumed to erase interface cost.
4. **A clear role for outcome optimization.** In this setting RLVR is useful as
   a reliability or post-training instrument. It makes a sampled policy express
   what it already knows more cleanly. It is not the bootstrap mechanism for a
   weight-naive token agent. That is a constraint for the theory phase; STEP 1
   does not need to keep re-proving the same division of labor.
5. **A measurement discipline.** Closed-loop execution matters; teacher-forced
   loss can look smooth while behavior collapses. Surface-matched shuffled
   controls can separate grammar from state use. Adaptation curves are more
   informative than final endpoints. Prequential training logs can locate
   transfer cheaply before spending on dense evaluation grids.
6. **A theory problem for world generation.** The dense model's poor structural
   generalization says that one small generated family is not enough. Before a
   next world is built, the project must say what consequential structural
   variation, composition, and accumulation across experiences mean—not tune
   this family until every metric is polished or assume that more variety is
   automatically useful.
7. **Reusable machinery.** The project now has a tested Rust world boundary,
   aligned renderings, privileged teacher and verifier, deterministic replay,
   standard Hugging Face artifacts, dense and RL paths, closed-loop evaluation,
   and a workable remote execution and evidence workflow. The next experiment
   starts from an apparatus rather than from infrastructure uncertainty.

The stopping rule is now simple:

> Do not run another STEP 1 experiment unless its possible outcomes would change
> the design of the next step.

Under that rule, STEP 1 is done. Its contribution is not that a small model
solved a toy world. It is that the project can now distinguish interface
acquisition, state-dependent decision behavior, transfer as reduced adaptation
cost, and outcome-based reliability—and knows that the next uncertainty lies in
the theory of developmental structure, syntax, retention, and world generation,
not in another ablation of the first probe.
