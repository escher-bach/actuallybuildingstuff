# Harness Findings

*Every protocol decision the harness made, and everything implementing Task Spec §7 taught about §7. [docs/09](09-harness-handoff.md) §5 asks that protocol changes be recorded here rather than pushed into families; this is that record. Nothing in `families/` has been edited — the one new file, [`modular.py`](../src/repertoire/families/modular.py), is the §6 worked family, which is §8 step 2's deliverable.*

**State:** §8 steps 1 and 2 complete, both gates met. Step 5 is built and runnable; **it has not been run at a budget whose result means anything**, and the harness refuses to read one that has not been.

---

## 0. What was asked, and the answer

Handoff §2.1: *"Your call to make, and make it centrally: adopt that convention explicitly, or change the signature to `posterior(history, query, k)`."*

**Adopted: `posterior(self, history, k, query=None)`.** The trailing `(query, None)` convention is supported as a migration path and is not a second interface.

- **No family needs to change.** The harness never calls `family.posterior` directly; it calls `protocol.answer_distribution`, which inspects the signature once and routes. Which route each family took is recorded in `protocol.POSTERIOR_ROUTE`, so the migration state is a readable fact rather than something absorbed silently.
- Routing is by `inspect.signature`, not by catching `TypeError`. Catching would swallow a `TypeError` raised *inside* `posterior` and reroute to the other convention, making a bug look like a convention mismatch.
- **The argument matters less than I expected, and this is the useful part.** A family can accept `query` and ignore it, so the signature is not itself a guarantee. The guard that bites is `protocol.check_query_sensitivity` — family-agnostic, works under either convention, and detects exactly the docs/03 finding 1 failure: a family whose L3 target does not move when the query does. It returns a *fact* rather than a verdict, because `junk_random`'s target genuinely is query-independent and a check that could not tell those apart would be worse than none.

**If you want to migrate a family**, add `query=None` to its `posterior` signature and condition on it. Parity is the one where it matters most: it is the family docs/03 uses as the example, and its current implementation is correct under the trailing-`None` convention.

---

## 1. Three further gaps in §7, found by implementing against it

Each is in `harness/protocol.py` with the same reasoning. None is worked around by family-specific code — §8 step 3's gate is "zero family-specific code in the harness", so where a capability is missing the harness raises with the family's name in the message.

### 1.1 A stochastic family cannot be rendered

§1.3 requires distribution-valued `evaluate`, and interface finding 3 established that this is the *only* way a stochastic family fits a protocol whose `evaluate` takes no rng. But nothing in §7 says how a sampled distribution key becomes an `Answer` object, and rendering needs one. `a3_test.py` hit the same wall and said so.

Resolved generically by looking for `family.answer_type`, then `family.output_type`, then a conventionally-named class in the family's module. **The third step is a concession** — it reads a module namespace, which is not part of any contract.

*What §7 should gain:* an `answer_type` field. One line per family.

### 1.2 L2 needs ρ_e⁻¹, and A6 is not satisfiable without it

§2.1: *"The oracle answers the query the model actually asked, not the teacher's"*, and calls the resulting error-recovery lesson free. **It is not free.** Reading the model's emitted tokens back into a `Query` requires an inverse of `render`, and §7 has no such method. Without it a harness cannot know what was asked, so it cannot answer what was asked.

A6 makes the same demand from the other side: *"responds sensibly to malformed and invalid queries"* presupposes something that can *detect* malformed, which is a parser. **A6 is not satisfiable under §7 as written.**

*What §7 should gain:* `parse_query(encoding, tokens) -> Query | None`, where `None` is the A6 case rather than a failure. Implemented on the stub and on the worked family.

### 1.3 The L0–L1 interior is inexpressible

§7's `preamble` is all-or-nothing, which makes L0 and L1 two points with nothing between them. §8 step 5 requires exactly what is between them. Added as an optional `partial_preamble(theta, encoding, fraction, rng) -> Reveal`.

The `Reveal` carries a `consistent(theta)` predicate as well as tokens, because the sweep's x-axis is *measured* residual entropy, and measuring it means enumerating θ consistent with what the preamble said.

**One implementation requirement that is not obvious:** the rendered preamble should be the *same length at every fraction*, with unrevealed slots rendered as an explicit unknown marker. Otherwise the dial moves sequence length alongside residual entropy — hazard 7's 85% length difference arriving from the other direction.

*Consequence for existing families:* none. Endpoints fall back to §7's `preamble`, so all nine run at L0 and L1 unchanged. A family without `partial_preamble` asked for an interior fraction gets a `ProtocolGap` naming itself — loud, because silently rounding to the nearest endpoint would make the sweep run four points and report a curve.

### 1.4 A signature gap worth noting but not repairing

`teacher_query(theta, history)` is not given `k`, but the hypothesis space depends on `k`. The worked family recovers it from θ's own modulus. Storing it on `self` would be worse — a family holding per-episode state stops being seeded-reconstructible under batching.

---

## 2. The level is not primitive, and L3 conflates two things

Implementing §2's four levels against §8 step 5's demand for a *continuous* sweep forces the question of what the four are made of. They decompose into three independent settings:

| | |
|---|---|
| `reveal` | how much of θ the preamble states, in [0, 1] |
| `query_source` | who chooses x_t — sampled, or the model (L2) |
| `target_mode` | what the answer channel is scored against — the realized token, or the exact posterior |

with `L0 = (1.0, sampled, realized)`, `L1 = (0.0, sampled, realized)`, `L2 = (0.0, model, realized)`, `L3 = (0.0, sampled, posterior)`. `spec_for_level` still produces the four, so nothing downstream changes.

Two things fall out.

**The L0–L1 dial is `reveal`, and it is continuous.** That is the sweep.

**§2's L3 ties together two separable things.** §2 defines L3 as *"withheld; not identifiable within the episode"* — a statement about the *episode*: T too short, or Θ too large, for the history to pin θ down. But what makes L3 supervision different *in the loop* is that the target is the exact posterior rather than a realized sample, and **that is available at every level.** Nothing stops an L1 episode being scored against the exact Bayes posterior; it is strictly better supervision than a sample from it, being the same target with the sampling noise removed.

Identifiability and target mode are orthogonal. The harness keeps them apart. Worth knowing for the repertoire: a row claiming "L3" may mean either, and the two have different floors — the posterior target's floor is 0, the realized target's is H(y | context).

---

## 3. L2 cannot be built offline, and that is architectural

§2 says L2 *"stays inside the interface… One token stream. No environment, no reward, no rollout to terminal."* All true. None of it makes L2 constructible as a static dataset.

§2.1 also requires that the oracle answer **what the model actually asked**, and what the model would ask is only knowable by sampling from the model. So an L2 episode needs T sequential generation steps *inside the training loop, at the current weights*.

This is **not** RL — the loss stays local, per token, nothing waits for a terminal reward, exactly as §2.1 designs it. But two consequences follow that are worth knowing before budgeting a run:

- L2 costs roughly (query width × T) extra forward passes per episode.
- The data distribution is **on-policy** and moves as the model does. L0/L1/L3 episodes are a fixed distribution; L2 episodes are not.

Neither is on the sweep's critical path, so this is recorded rather than optimized. Uncached generation; KV caching is the obvious fix if L2 ever becomes a bottleneck.

**Observed:** at initialization essentially every model query is malformed — the stub's queries need 2 specific tokens out of 103, so well-formed emission is ~0.06% by chance. L2 therefore has a cold-start phase during which the answer channel never fires and the model learns query *format* from the q\* channel alone. The design handles it correctly (a malformed query still costs on the query channel, which is the recovery lesson), but a budget that is too short will spend all of it on format.

**A decision §7 left open:** on a malformed query, **no history entry is appended.** The trial produced no observation, so nothing was learned about θ and q\* is unchanged — a family cycling its probes on `len(history)` correctly re-asks the same one. A `(None, None)` placeholder would advance every such counter past a question that was never answered, and would break the history contract for the eight families that do not expect it.

**"Oracle echo", since §7 names it and nothing defines it:** the ERR response and the ASK/ANSWER turn markers — the environment talking *about* the exchange rather than participating in it. Never supervised. The oracle's *answer* tokens **are** supervised, because §2.1 scores them as "the model's prediction of ρ_e(y_t)"; teacher-forced and supervised are the same positions in an autoregressive stream, so §2.1's "oracle's emitted tokens masked" cannot mean those or L2 would have no answer channel at all.

---

## 4. The Bayes floor has two terms, and at trial 1 the notation is worth more than the rule

**Found by the floor being wrong, not by a test.** The first L1 run measured a trial-1 loss of 4.18 nats against a computed floor of 0.69 and looked like a model 3.5 nats from optimal.

It was not. The stub's answers are binary, so 0.69 nats is the *entire* uncertainty about the rule. The other 3.5 nats are uncertainty about **which token denotes which answer** — because §1 samples the encoding `e` on the same footing as θ and hides it too, and at trial 1 the answer symbols have never appeared.

The model was already near-optimal. The instrument was wrong.

Reported now as a **band**:

| | |
|---|---|
| lower | exact `H(y \| context, e)` — conditions on the encoding, which the model does not observe, so nothing can legitimately beat it |
| upper | + a bounded notation term: unresolved answer mass × log(unseen content symbols) |

The leak check fires against the **lower** end, where a run finishing below it is reading something the posterior calculation does not know about.

**Why this matters beyond the arithmetic.** It affects any per-token loss comparison across families with different alphabet sizes or different rates of symbol reuse — which is every cell of the §6 transfer matrix. The rule term is what the dial moves; the notation term is roughly constant across a sweep, so curve *shape* survives, but absolute "distance from optimal" claims do not without it.

The notation term is an **upper bound, not exact**: when several candidate answers are all unseen, not knowing which is which already absorbs the rule uncertainty rather than adding to it. Exactness would need the joint posterior over (θ, e), which is not enumerable.

---

## 5. `enumerate_theta` is the support, not the prior — and parity's sampler does not match it

§1: *"The meaning of the family is given entirely by its sampler."* A posterior computed against a uniform prior over the enumerated support is the exact posterior **of a different family** whenever the sampler is not uniform, and it looks entirely reasonable.

**This is not hypothetical.** `ParityIdentificationFamily.sample_theta` draws a subset *size* uniformly from 1..d and then a subset of that size, which puts far more mass on small and large subsets than uniform-over-subsets does. Its own `posterior` enumerates uniformly. At d=8 the implied prior on a singleton subset is off by roughly a factor of 8.

`entropy.check_prior_matches_sampler` tests this by sampling and is meant to fail. The worked family carries an explicit `prior_weight` for exactly this reason (its band makes small-m θ individually more likely, since fewer injections share the mass).

**Measured, not argued.** `python -m repertoire.harness check --family parity` reports a largest relative deviation of **31.4×**, at `ParityTheta(n_dims=8, subset=(0..7), offset=0)` — the full subset. The arithmetic matches exactly: `sample_theta` picks size uniformly from 1..8, so P(full subset) = ⅛ × 1 = 0.125, against 1/510 = 0.00196 under uniform-over-support. Ratio 32.

**For the repertoire:** any family with both `enumerate_theta` and a non-uniform `sample_theta` should supply `prior_weight(theta, k)`. Until it does, its exact L3 target is exact for a sampler other than the one generating its episodes. Parity is the confirmed case; the other eight have not been checked.

The fix for parity is one method, and it is yours to make rather than mine because its `posterior` would need to change alongside and that method carries the unplanted-near-duplicate assertion:

```python
def prior_weight(self, theta, k):
    d = self.dimensions(k)
    # sample_theta draws size uniformly from 1..d, then a subset of that size,
    # then an offset -- so mass per theta depends on how many subsets share its size.
    return 1.0 / (d * math.comb(d, len(theta.subset)) * 2)
```

**This currently blocks parity as the sweep's A4 control**, because the measured x-axis is computed from the same prior. The modular sweep does not depend on it.

---

## 6. The sweep, and what it will and will not answer

Built in `harness/sweep.py`. Four decisions, all in the module docstring, the important ones here.

**The x-axis is measured, not nominal.** The map from dial setting to residual entropy is nonlinear and family-specific, so plotting against the dial reports a curve in units nobody can compare across families. `entropy.py` computes `H(y | context, e)` exactly by enumeration.

**Structural content alone cannot answer the question; transfer can.** S is the area above the run's *own* floor, and the floor **rises** with residual entropy because part of the loss becomes irreducible. So S falls with entropy for a reason that has nothing to do with whether supervision works — **a naive sweep would manufacture a monotone collapse out of arithmetic.** This is why §8 step 5 says "structural content **and** transfer": transfer holds the evaluation target fixed, so the floor does not move under it. Transfer is the arm that decides; S is reported beside it because §4 defines it.

**Two dials, and running both is the control.** §8 step 5 names both parametrizations. Running one tests whether *that parametrization* collapses. `compare_dials` states a disagreement rather than averaging it — same argument the Task Spec makes for preferring a sweep to a four-way comparison, one level up.

- `free_observations` — unscored observations before the scored trials. **Generic**: asks nothing beyond §7, works on all nine existing families, and holds the supervised-token count *exactly* constant across settings.
- `preamble_fraction` — needs `partial_preamble`; available on the stub and the worked family.

**The A4 confound is real.** The worked family fails A4 at these moduli — §6 says so and the register row records REPAIRABLE, not PASS. A memorizable family could flatten the curve for a reason unrelated to residual entropy. Parity has the opposite property (2^d table, d probes), so `--family parity` on the generic dial separates "supervision above L0 does not work" from "this family was memorized". **Run both before reading any collapse** — and see §5: parity's control role is blocked until its prior is supplied.

### 6.1 For a balanced family the x-axis barely moves while everything changes

The dial's x-axis is `H(y | context)`, because that is how §2 defines the levels. Measured on parity across the observation dial:

| free observations | `H(y \| context)` | `H(θ \| context)` | θ alive |
|---|---|---|---|
| 0 | 0.693 | 5.54 | 297 |
| 2 | 0.693 | 4.16 | 74 |
| 4 | 0.664 | 2.79 | 19 |
| 6 | 0.539 | 1.55 | 5.4 |
| 8 | 0.279 | 0.65 | 2.1 |

**The answer entropy is flat at log 2 over the first half of the dial while the hypothesis space collapses by a factor of fifteen.** Parity is balanced, so the answer to a random query is a coin flip however well θ is known, right up until θ is known exactly.

This is docs/03 finding 1 arriving in the *x-axis* rather than in the target, and it has a direct consequence: a sweep on a balanced family, plotted against residual entropy alone, reports that nothing changed across the region where almost everything changed. `H(θ | context)` is now logged on every sweep point beside it. A divergence between the two is a fact about the family, not noise — and it says which families are poor instruments for this particular dial. The worked family does not have the problem (`H(y)` falls 1.51 → 0.01 smoothly), which is a further reason it is the right primary.

**The reading is pre-committed**, in `expectations.py`'s style: `read_curve` takes only the curve and a measured noise estimate, with thresholds fixed as module constants. A test asserts they are not arguments — a threshold that is an argument is a threshold that gets tuned after the numbers arrive.

### The guard that matters most

**A sweep run too small produces noise, and noise has a shape.** The smoke run produced a confident "rising" verdict from 60 steps in which the model never left chance. §8 step 5 is the only step that can *end the programme*, so a verdict an underpowered run can produce is worse than no verdict.

`SweepResult.fit_to_read()` gates the reading on the runs, not the shape: models must have closed ≥35% of the distance from their starting loss to the Bayes floor, and ≤50% of runs may still have a falling tail. Below that the CLI prints **READING REFUSED** and the reason.

---

## 7. Compute

Development is CPU-only; the sweep targets a Kaggle T4 or P100; §9's eventual run is a TPU. Nothing here uses a custom kernel, a CDN dependency, or a device-specific numeric path. `model.device_report()` turns autocast on only at sm_70+ — **P100 is sm_60, where fp16 has no tensor cores and is slower and noisier than fp32**, so "cuda" alone is not enough to justify it.

```bash
PYTHONPATH=src python -m repertoire.harness check --family modular
```

```bash
PYTHONPATH=src python -m repertoire.harness sweep --preset t4 --family modular
```

Presets (`smoke`, `t4`, `p100`) exist so the budget is a *named* thing rather than flags somebody typed — §4 requires it be held fixed and reported, and the name travels with the output. `smoke` finishes on a laptop CPU in about two minutes and exists so a mistake is found before a GPU session is spent on it. Sweeps checkpoint per point and resume, because a Kaggle session does not always finish; a cached record at a different budget fingerprint is discarded rather than used.

---

## 8. What I have not done

- **The sweep has not been run for real.** Everything is in place and verified at smoke scale; the result does not exist yet.
- **`compose.py` is not wired into the sweep or the matrix.** Composites round-trip through `build_episode` only if they satisfy the same protocol; not checked.
- **The A3 leak test still uses the token-length proxy.** Handoff §4 task #9 is to swap in prequential structural content, which now exists. Not done.
- **§8 steps 3 and 4** are the repertoire session's, and step 4's planted basis is reported ready. The harness can run it; `expectations.score` takes a matrix and nothing builds one yet.
- **The two missing axes (docs/06)** remain deliberately unbuilt, per handoff §2.6.

## 9. One request back

`register/rows/mod-arith-hidden-permutation.toml` is at `status = "translated"` and now has an implementation, so it can go to `implemented`. Two things in it are worth a second look against the code:

- Its A3 note suggests an encoding where the operator symbol is drawn from the permuted alphabet. I did **not** do that, and the reason is not laziness: A2's check permutes the content alphabet, so an operator drawn from it would move under the permutation while denoting the same operation, and the check would compare two different renderings. It is a good idea for a fourth encoding but needs the A2 check to distinguish operator symbols from content symbols first.
- Its `episode_length` field says T "should be swept around" the identification bound rather than pinned to it. The sweep does not currently vary T — it varies what precedes the scored trials at fixed T. Varying T as well is a third dial and is not built.
