# Trace thinning — the default may be backwards

*From vein §2.6. This is a finding about the Task Specification, not about the repertoire, and it is recorded here because it is the kind of thing that gets lost between documents.*

## What the Task Spec says

§1.2 sets the default: **"emit traces at low `k`, thin them as `k` rises."** The rationale given is that at low `k` the untraced task may be unreachable and the trace is the only thing making learning possible, while at high `k` the trace is a shortcut. The spec then adds — and this is the sentence that sent a reviewer to check — that this default **"is not a guess; it has a literature"**, naming worked examples, completion problems, fading, and the expertise-reversal effect. It also flags the whole question as *"the least settled decision in the document"* and says to treat the schedule as a swept parameter.

## What the literature actually says

The literature is real and says something adjacent but **not the same**. Verified from full body text of the guidance-fading chapter and the expertise-reversal paper, with page-level quotes in `notes/vein-2.6-report.md`:

- **Worked examples beat unguided problem solving for novices**, and the benefit **reverses** as expertise rises. That reversal is the named effect, and it is indexed by *learner expertise*, not by task difficulty.
- **Gradual fading beats an abrupt switch**, replicated across three experiments.
- **Backward fading** (strip the last step first) generally beats forward fading — though a later study found the faded step is learned about equally well either way, so direction is a second-order efficiency knob rather than the mechanism.
- **The fade *rate* is itself expertise-conditional**: more knowledgeable learners did better with fast or immediate transitions, less knowledgeable ones with slow transitions.
- **Fading only has teeth when intrinsic load is high.** Quoted from the source: high intrinsic cognitive load is *essential* to the fading effect, and no significant effect should be expected for material that does not impose it.

## The problem

**Every study in this literature fades guidance against demonstrated learner competence over training time, holding material difficulty roughly fixed. None fades guidance as a function of item difficulty at a fixed point in a learner's development.** Those are different axes, and the spec's default collapses them.

Worse, the last bullet points the opposite way. Harder material is where worked guidance earns *most*, not least. So if our `k` indexes **instance difficulty** rather than **the model's competence at that difficulty band**, then "thin as `k` rises" is plausibly **backwards on hard instances early in training** — removing support exactly where this literature says support is doing the most work.

To be careful about what is and is not being claimed: the spec's rationale is not obviously wrong, because its argument is about reachability rather than cognitive load, and an inducer is not a human learner. What is wrong is the citation. **The default is a guess after all — a reasonable one, but the literature named in support of it is about a different axis.** Given the spec itself calls this its least settled decision, that is worth knowing before it hardens into an assumption.

## Recommended repair — two axes, not one

1. **Trace density should track the model's demonstrated competence at a given `k`-band**, diagnostic-gated, rather than being a fixed function of `k`. The staged-tutor design in the source literature is a working template: worked examples → last step removed → last two removed → unguided, with a cheap test gating each advancement. Individualized adaptive fading beat both fixed-schedule fading and no fading, on immediate and delayed post-tests.
2. **Within a fixed competence level, higher `k` should if anything start denser and fade more slowly** — the reverse of the current default's direction.
3. **Fade gradually, prefer backward fading, and keep a cheap justification token at faded steps** rather than deleting them silently. Self-explanation prompts at faded steps produced strong advantages on near and far transfer.

**The cheap gating signal is worth stealing directly.** The source validated a *first-step diagnostic* — show a problem, ask only for the first move, time-limited — against full worked-solution scores at high correlation and up to 5× less testing time. Our analogue is nearly free: accuracy on a handful of validation queries at that `k` before thinning further. That is a controller signal in the sense of Task Spec §9, computed from things the training loop already emits.

**If only one axis is affordable**, the honest single-axis proxy is **training step / curriculum position**, with `k` as a secondary modifier on fade *rate* rather than the primary driver of fade *amount*.

## What this does not settle

The spec is right that this should be swept rather than fixed. Nothing above replaces the sweep; it changes what should be swept — a two-parameter schedule over (competence, `k`) rather than a one-parameter schedule over `k`. And the transfer of a human-cognitive-load result to an inducer is exactly the Hazard 1 this review has flagged in every behavioural vein: **import the parametrization, re-derive the knob.** The claim here is narrow and I want it to stay narrow: *the literature cited in support of the current default does not support it*, and the default's direction is questionable on that literature's own terms.
