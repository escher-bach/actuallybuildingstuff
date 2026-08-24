# STEP 2 Meta Process

**Status:** user-directed working agreement

## Purpose and scope

STEP 2 is a world-generation stage. Every task performed at this point must directly contribute to proposing, constructing, testing, or improving worlds for the baby robot. Training systems, evaluations, interfaces, and analysis are supporting work for world generation; they are not independent project goals.

This process is an internal way of working and will not be published in this form. Its primary measure is not whether the process looks scientifically viable to an outsider. Its measures are velocity—how quickly useful worlds and decisive evidence are produced—and the user's satisfaction that the work is answering the intended question.

## Direction and authority

Project direction comes from requirements, principles, and decisions written by the user. Documents or paragraphs produced by an agent are drafts and do not become constraints merely because they exist in the repository. They may record context or propose an option, but the process must not adhere to them unless the user wrote or explicitly approved their content. In particular, an agent-authored specification cannot silently authorize work, forbid a world, add acceptance gates, or expand the project's goals.

## World-generation principles

World generation begins from strong, explicit principles rather than from an improvised collection of tasks. A world must arise from a declared ontology, be procedurally generated from its first version, expose a clear information boundary to the learner, and state the role and information access of the teacher and verifier. Its supervision must teach something available through the learner's experience, and anti-degeneracy tests must check that success does not come from memorization, stable names, presentation order, serialization leakage, or other unintended shortcuts. The reason for constructing and testing a world is always its prospective contribution to downstream learnability and confidence in transfer.

The worlds may be highly alien. They do not need to simulate real physics, reproduce a real robot, or resemble the downstream setting. Their design is justified by the properties they make learnable and by the evidence their variation can provide about transfer to later tasks.

## Engineering and the location of innovation

Training uses Kaggle. No GPU training run is launched without the user's explicit authorization. Kaggle is the execution environment for training and retained experiment artifacts, not a source of scientific direction.

Use the local CPU when a test is expected to complete within ten minutes. If
the local CPU estimate exceeds ten minutes, use Kaggle and include its launch
and setup overhead in the expected time to a usable result. The user authorized
this threshold rule on 2026-08-24; it is the standing execution choice for
tests within the current STEP 2 scope. A materially different or open-ended GPU
training run still requires its own explicit scope and stopping conditions.

Standard model architectures, maintained libraries, standard training algorithms, and standard artifact formats are used wherever they already solve the problem. Engineering quality means minimizing hand-rolled infrastructure, keeping adapters thin, and leaving model training, checkpointing, optimization, logging, and other commodity concerns with their established libraries.

Innovation belongs where this project actually needs it.

## Experiment discipline and velocity

Before any experiment, the user must be told the current state, the experiment's purpose, why it is needed now, and what project decision its result will change. The proposed run must also make its relevant cost and scope clear. If those points cannot be stated plainly, the experiment is not justified: it consumes time without increasing velocity.

After an experiment, the user must receive a clear account of what happened, what was learned about the world or interface, and what decision follows. An apparatus failure must be distinguished from evidence about the world itself. Experiments are valuable only insofar as they remove uncertainty, reject a bad direction, or advance the construction of a useful world; accumulating runs, metrics, or process for their own sake is a loss of velocity.
