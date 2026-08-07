# The harness is the deployment distribution

*A design-level finding, recorded 2026-08-07. It does not invalidate anything built so far, and it adds a requirement the whole design has been silent about.*

## The claim

The model this curriculum trains **will be deployed inside an agentic harness**. Coding, reasoning, tool use — that is where it runs. So the harness is not merely a source of primitives to excavate; it is the **input distribution at deployment**.

Everything in the register so far manufactures episodes of one shape: an optional preamble, then alternating query/answer pairs, one family per episode, nothing else in context. Clean, homogeneous, self-contained.

A deployed model sees none of that. It sees a system prompt, tool schemas, prior turns, tool results, file contents, error strings, truncated output, permission denials, subagent reports, and periodically a lossy summary standing in for everything that no longer fits.

## Why this is not a presentation detail

Task Spec §9's claim under test is that **capability and knowledge are separable, capability is the manufacturable half, and a model with the capability half installed acquires the other faster.**

If capability is installed in a format structurally unlike the deployment format, transfer can fail **for a format reason** — and the failure is indistinguishable, from the outside, from the capability claim being false. The programme would record a negative result on its central thesis when what actually happened is that a capability was installed in a shape the deployment environment never presents.

That is the risk. It is not that our episodes are unrealistic; it is that **unrealism here is confounded with the thing being measured.**

## What this changes

### 1. A3 acquires a second job

A3 has been read as an *invariance* requirement: sample the encoding per episode, and the family must be invariant under it. That is still true.

But if the harness is the deployment format, then **at least one encoding in `𝓔` should be the deployment format** — an episode rendered as a tool-call transcript, with schemas, results, and errors, rather than as `q → a` pairs. A3 stops being only an invariance test and becomes partly a **format-coverage** requirement.

This is cheap in principle: `𝓔` is already per-episode and already sampled. It is not cheap in practice, because our current encodings vary punctuation and slot-tagging, and a transcript rendering is a different order of change. It is also exactly what the A3 leak test (`docs/04`) exists to check — a transcript encoding will differ enormously in length and structure from an infix one, and that difference must be shown not to move difficulty, or it is a radical wearing an incidental's clothes.

### 2. The two deferred axes are not hypothetical — they are the deployment condition

`docs/06` found the parametrization missing two axes and deferred both until after the dial sweep. Under this reframing both are *the normal case in a harness*, not edge cases:

- **Query cost / entanglement.** In our L2, issuing a query is free — a bad one costs the turn and nothing else. In a harness, **issuing a tool call is the action**: it costs latency and tokens, it may have side effects, and it is sometimes irreversible. A model trained where probing is free has been trained in the one regime deployment never offers.
- **Validity duration.** Our θ is sampled once and holds. In a harness the world changes under the agent — files are edited, state moves, earlier observations go stale.

They stay deferred, because §8 step 5 expects the named levels to be cut in the wrong places and adding axes before the sweep would be designing the answer. But the justification for eventually adding them is now much stronger than "three paradigms did not fit."

### 3. Three family templates this motivates that nothing in the register covers

Recorded as design proposals, not built. Each is backward-generable, which is what makes them admissible at all.

**Compaction survival.** Run an L1 episode until θ is identified. Then **replace the trial history with a lossy summary** and continue querying. θ is unchanged; what changed is that the evidence for it is now second-hand. The model must either have carried the identification or reconstruct it from the summary. Generation is O(1) — we own θ and we write the summary. This is the closest thing to a direct manufacture of "survive your own context being compacted", and no existing family touches it.

**Distractor tolerance.** Interleave the episode with segments that are well-formed, plausible, and irrelevant — trials from a *different* family at a different θ, clearly delimited. The task is unchanged; the context is no longer homogeneous. Backward-generable trivially. Tests whether identification survives content that must be ignored rather than used.

**Malformed-response recovery.** The oracle returns errors, truncations and permission-denials at a controlled rate, as *first-class responses* rather than as failures. A6 already requires that malformed queries get a well-formed error, and Task Spec §2.1 already calls the error-recovery lesson "free" — this makes the error a sampled event rather than an accident. The rate is a knob.

## What I am not doing

**Not rebuilding the existing families.** They are correct for what they measure, and the format question is orthogonal to whether parity-is-parity or whether the plants recover.

**Not adding a transcript encoding yet.** It interacts with the harness session's format decisions — loss masking over tool-result segments, whether oracle-echo tokens are masked, how a tool call is tokenized — and doing it before those are settled would mean doing it twice. Flagged in the handoff instead.

**Not treating any of this as established.** The vein §2.7 review is running and is now asked specifically for what a harnessed model's context actually contains and what fraction of a trajectory is unhelpful content it must ignore. The templates above are motivated by the reframing, not yet by measured evidence about the deployment distribution.

## The honest summary

The register has been optimizing a family's *content* — does it hide a rule, is it backward-generable, does it resist brute force — and has said almost nothing about the *shape of the sequence a deployed model reads*. Both matter, and only one has been under review. This does not undo the work; it names a second axis the work has not been scored on, before the scoring happens rather than after.
