# Paradigm Coverage Check

*Repertoire Spec §11 step 4. **Gate: "Do §2.3's paradigms fall out as special cases? They do, or the parametrization is missing something."** This is §10 condition 3's first test — validation is coverage of established practice, and a parametrization is credible when paradigms people independently converged on fall out of it rather than being bolted on.*

**Verdict: seven of ten fall out cleanly. Three do not, and they fail in two distinct ways that point at two missing axes.** Per the gate's own wording, the parametrization is missing something. What follows is what, and how much it costs.

---

## The check

For each paradigm: can it be written as `T = (Θ, P_Θ, X, f, 𝓔, ρ, k)` with a level wrapper, without special-casing the harness?

| Paradigm | θ | Level | Falls out? |
|---|---|---|---|
| **Reception paradigm** (instances presented) | the concept | L1 | ✅ Cleanly. This *is* L1 — the paradigm and the level are the same object |
| **Selection paradigm** (subject picks the instance) | the concept | L2 | ✅ Cleanly. Membership query = model-emitted `x_t`; oracle answers what was asked |
| **SHJ six types** | which Boolean function over d dimensions | L1 | ✅ Cleanly. Implemented; Types I and VI are live families |
| **Conjunctive concept attainment** | the conjunction | L1/L2 | ✅ Cleanly. Implemented, and its strategy catalogue supplies `q*` |
| **Probability matching** | a hidden Bernoulli rate | L3 | ✅ Cleanly, and it is the only family whose non-identifiability is intrinsic rather than imposed by truncation |
| **T-maze / discrimination** | which stimulus is positive | L1 | ✅ Cleanly, though trivially — one bit |
| **Harlow learning sets** | which stimulus is positive, resampled per problem | L1 × many | ⚠️ **Not a family.** The content is the across-episode acquisition curve, not the episode. Falls out of the *harness*, not the parametrization — and is Task Spec §4's own primary metric with a 1949 precedent |
| **Reversal learning / WCST** | the rule, **plus when it silently changes** | — | ❌ **Does not fit.** See A below |
| **Daw two-step** | reward probabilities that **drift every trial** | — | ❌ **Does not fit.** See A below |
| **Bandits** | per-arm rates | — | ❌ **Does not fit.** See B below |

---

## A — The parametrization assumes θ is sampled once and holds

Task Spec §1 states it directly: *"θ persists across t, resampled between episodes. The episode, not the pair, is the unit of training."* That is load-bearing — it is what makes L1 identification meaningful.

Two paradigms violate it in different degrees:

- **Piecewise-stationary θ** (reversal learning, WCST): θ is fixed for a run, then silently replaced. This is not L1, because θ does not persist across the episode. It is not L3 either, because θ *is* identifiable — just not stably. Our level dial measures *how much the context determines the answer*; this varies **how long the context stays valid**, which the dial does not measure.
- **Continuously drifting θ** (Daw two-step): θ is never sampled once at all. It is a stochastic process, and the target is "what is θ *right now*", which is a different question from "what is θ".

**Cost of the gap:** this is where error recovery lives. `belief-state-reset` — the register's twelfth primitive and the one closest to the programme's stated priority — exists *because of* these paradigms, and the formalism cannot express the family that motivated it. The `wcst-reversal` row is currently written by folding the switch schedule into θ, which restores A1 but is a workaround: it makes θ an object with internal temporal structure that the level wrappers have no way to read.

**Minimal repair.** One additional per-family declaration: a **validity duration** — how long a sampled θ holds. Constant (every family today), piecewise with a pre-sampled schedule (reversal), or a declared drift process (Daw). The harness needs it only to know when its identification assumptions reset. This is smaller than a new level: it is a property of `P_Θ`, not of the reveal policy.

## B — L2 assumes querying is free

In L2 as specified, emitting a query costs only the turn: a bad query yields an uninformative answer and the episode continues. Task Spec §2.1 makes this explicit and treats it as a feature — *"the error-recovery lesson, and it is free."*

In bandits, reversal learning and WCST, **the query is the scored action.** There is no test-then-be-scored separation. Both involve the model choosing `x ∈ X`, so both look like L2, but the cost structure is entirely different and the resulting behaviour is different: free querying rewards pure information gain, while entangled querying forces an exploration/exploitation trade the current L2 has no way to express.

**Cost of the gap:** our L2 occupies only the free-query corner of a larger space. Prediction P2 in `docs/02` — *L2 is a wrapper, not a capability* — may be right about our L2 and still misleading, because the agentic content people usually mean may live in the corner we have not built.

**Minimal repair.** A per-family flag stating whether the query channel is scored. Left unset it is today's behaviour. Set, the loss includes the oracle's response rather than masking it — which is a change to the loss mask, an object §7 already defines, not a new mechanism.

---

## What I am not doing, and why

**Not implementing either repair now.** Task Spec §8 step 5 sweeps residual entropy continuously and expects the four named levels to be cut in the wrong places; it is the only step that can end the programme rather than redirect it. Adding axes *before* that sweep would be designing the answer — and both gaps above were recorded in `docs/02` §5 as dated predictions precisely so they could be scored against the sweep rather than absorbed into it.

**Not treating this as a failed gate either.** The gate says the paradigms fall out *or the parametrization is missing something*, and the second branch is a legitimate outcome with a named consequence. Seven of ten falling out cleanly — including the two that supply L1 and L2 their historical definitions — is real coverage. The three failures are informative rather than fatal, they cluster into exactly two axes rather than scattering, and both repairs are small and local.

**The honest summary:** the four levels are a good parametrization of *how much the context determines the answer*, and the paradigms that vary that fall out of it. The paradigms that do not fall out vary something else — how long the context stays valid, and what querying costs. Neither is a defect in the level dial. Both are axes the dial was never claiming to cover, and the value of running this check was finding that they are only two, and that they are orthogonal to it.

---

## Consequence for the candidate set

`wcst-reversal` stays in the register with its workaround documented, and does **not** advance to implemented until the validity-duration declaration exists — implementing the workaround would bake it in. Recorded as a task.

Nothing else changes. The bandit family was never registered, and on this analysis should not be until the query-cost flag exists: a bandit written under today's L2 would be a bandit with the exploration/exploitation trade removed, which is the only interesting thing about it.
