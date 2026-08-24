# Robot-Stage World Plausibility

**Status:** user-directed principles with assistant-authored wording; pending
final review

## The plausibility goal

The plausibility question is whether an agent's performance across constructed tasks, where the tasks satisfy properties that arise naturally from the chosen world ontology, gives us reason to expect that what the agent learns will make the actual downstream tasks more learnable. It is a claim about confidence in transfer, not about zero-shot performance: if an agent learns policies that perform well across meaningfully varied settings—for example, for robots on Mars and Venus—we should ask how much more confident that makes us that the agent can subsequently learn to perform well on Earth.

## Scope of the robot stage

The robot stage is neither simulation training nor sensorimotor training. It makes no claim about a real system or about transfer to a particular robot. Its worlds therefore require no physics, realistic morphology, spatial realism, or real-world counterpart. The worlds given to the baby robot may be highly alien if learning and performing in them gives us justified confidence that the relevant properties will transfer by making the actual downstream tasks more learnable.

## What failed in STEP 1

STEP 1 did not validly test whether the learner acquired the intended hypothesis-elimination process. The generated worlds accidentally gave some hypothesis labels stable observable meanings and exposed other instance fingerprints, so a simple shortcut policy could explain the learner's performance. At the same time, the teacher and the reported expert ceiling used hidden information—the latent truth or counterfactual evidence table, unobserved costs, and horizon—that the learner did not receive. The learner was therefore trained toward decisions that were not generally determined by its own observations, while the world offered an easier unintended regularity. The measured results remain results for that generated family, but they do not establish learning or transfer of the intended process.

## Resulting decisions

**Clear information boundaries.** Every world must state what information is available to the learner at each decision, what remains hidden in the world, and what additional information is available only to the teacher or verifier. A performance ceiling or learnable target must be defined at the learner's information boundary. Scores from agents with privileged information may still be useful, but they must be identified as privileged scores and cannot be used as evidence of what the learner could infer or learn from its own experience.

**The teacher's role.** The teacher supplies supervision that helps the learner acquire a policy or property through the learner's available experience; it is not a substitute input channel for facts hidden from the learner. A teacher may use privileged state to construct scaffolding, feedback, or outcomes, but the supervised target must be achievable from the learner's information: a definite action when the observations determine one, or an appropriate set or distribution when they do not. Training must not silently require the learner to reproduce distinctions that exist only in the teacher's view of the world.

**Anti-degeneracy tests.** Before performance is interpreted as evidence for a claimed learned property, the world must be tested for simpler ways of obtaining the same score. These tests should include public-information baselines, semantic-preserving permutations of labels and presentation order, checks for generator and serialization leakage, and evaluation on procedurally generated instances disjoint from those used to fit the policy. Their purpose is narrow: to show that the measured performance depends on the task property under study rather than memorization, naming conventions, ordering, or another accidental shortcut. They support the transfer claim but are not themselves the plausibility criterion.

**Principled procedural generation from day one.** A world is designed from the beginning as a generative family rather than as a fixed collection of hand-authored cases. Its generator follows the declared world ontology: each source of variation has a stated meaning, its distribution is deliberate, and changes that should be semantically irrelevant are generated independently of the underlying task. Training and evaluation use disjoint generated instances, while tests verify the intended invariances and detect stable identifiers or finite-support memorization. Procedural generation is therefore part of the scientific definition of the world from its first version, not an augmentation added after a small world has already shaped the learner or the claims.
