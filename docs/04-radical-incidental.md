# Telling `k` from `e`

*From vein §2.1. The Task Spec asserts a distinction between the difficulty knob `k` and the per-episode encoding `e` but supplies no way to check that a given slot is one rather than the other. Educational measurement named the same distinction forty years earlier — **radicals** move difficulty, **incidentals** do not — and, unlike us, built an empirical test for it. This is that test, and it is the most directly usable thing any vein has produced.*

**Why we need it.** A3 requires that `e` be sampled per episode and that the family be invariant under it. We have been *asserting* invariance from the construction — "these three renderings are just notation" — with no procedure that could contradict us. If a supposed incidental actually moves difficulty, then `e` is secretly part of `k`, episodes drawn with different encodings are not exchangeable, and every structural-content number computed over them mixes two effects. Nothing downstream would flag it.

---

## The procedure

1. Group items into **families**: sets that share the hypothesized radical value and differ only in supposed incidentals.
2. Fit an **unconstrained** item-response model in which every item gets its own difficulty parameter.
3. Fit a **constrained** model in which item difficulty is a function of the radical alone — items in the same family are forced to share a difficulty.
4. **Likelihood-ratio test** between the two. A significant result means the constrained model is worse, i.e. something inside a family is moving difficulty, i.e. an "incidental" is doing radical work.
5. Separately report the **variance in difficulty explained by the radical**, which answers a different question — how much the radical accounts for — and must not be read as answering step 4.

## The result that makes the procedure worth adopting

A figural-memory item-generation study (N = 208) found the radical explained **95.41% of the variance** in item difficulty (adjusted R² = 0.939) — and the likelihood-ratio test **still rejected** the constrained model, χ²(6) = 24.21, p < .001. Within-family differences reached 6.82% (p < .05) in the low-load family and 12.36% (p < .001) in the medium-load family. *(Curator-verified 2026-08-07 against the source; every figure above was read directly.)*

**That combination is the whole lesson.** A radical explaining 95% of difficulty variance would pass any eyeball test, any correlation plot, and most reasonable thresholds — and its incidentals were still leaking. High explained variance and clean separation are different claims, and only the second one is what A3 actually asserts.

Note also *where* the leak was: the high-load family showed no significant variation (−1.48%, n.s.), while the middle of the range did. **A leak can be invisible at the extremes of the difficulty range and present in the middle**, so testing at maximum `k` alone would have missed it.

---

## What we do with it

**Adapt, don't transplant.** Their difficulty is human difficulty measured by item-response theory on a subject sample; ours is an inducer's loss. The analogue is direct and cheap:

- **"Item family"** → episodes drawn from one family at one `k`, differing only in the encoding `e` sampled per episode.
- **"Item difficulty"** → the per-family prequential structural content (Task Spec §4), or the within-episode acquisition slope.
- **The test** → does a model in which loss depends only on `(family, k)` fit as well as one that also lets the encoding index matter? Same shape of comparison; the model class is ours to choose, and the comparison is the point rather than the specific statistic.

**Where this belongs in the build order.** It is an A3 unit test that every family should pass before entering the matrix, and it is cheap relative to a matrix run. Adding it after the fact would mean the block structure had already been read off possibly-contaminated numbers.

**Note what it costs.** Running this needs several encodings per family and enough episodes per encoding to estimate a difficulty — which is exactly the extra sampling A3 already requires. So the marginal cost is the statistic, not the data.

**Recorded as a task, not done yet.** The families implemented so far have 2–3 encodings each and have never been tested this way. On the evidence above, the prior should be that at least one of them leaks.

---

## A second, harder lesson from the same vein: A2 is not achievable by construction

The Cattell Culture Fair Intelligence Test was *designed* to be content-free — figural items, deliberately stripped of cultural material, on the same reasoning we use when we say our symbol alphabet is semantically empty. Administered across American, Nigerian and Indian samples and analysed with four independent item-bias methods (a modified chi-square procedure, an item-difficulty p-value method, a one-parameter Rasch model, and a Cochran chi-square), **59% of items were flagged as biased**, with high inter-method agreement. *(Curator-verified 2026-08-07.)*

And a separate Rasch-based cross-cultural analysis of the same instrument found a much more modest 13 of 46 items biased. **Different methods, different answers, same test.**

Two consequences for us, and they point in the same direction:

1. **Designing for A2 does not achieve A2.** A deliberately content-free figural test still leaked at scale. Our families are better placed — our symbols really are arbitrary tokens and relabelling really is an exact symmetry of the generator — but "we built it to be invariant" is the same argument Cattell made, and it was not sufficient.
2. **Our A2 check is one method.** `permuted_alphabet_check` verifies equivariance under relabelling. It does not detect a difficulty asymmetry between *particular* alphabet choices, nor leakage through structure rather than symbols — and we already found one case where it passed on a family rendering its content as `PAD` and `BOS` tokens, because equivariance under the wrong alphabet is still equivariance. The psychometric experience is that bias detection is method-dependent and that a single method understates. **A second, independent A2 method is worth having**, and the radical/incidental test above is a natural candidate for it: an alphabet choice that moves difficulty is a radical wearing an incidental's clothes.


---

## First run, 2026-08-07 — two findings, neither of them a leak

Implemented as `src/repertoire/a3_test.py` and run over all seven implemented families. The real difficulty proxy is the prequential structural content of Task Spec §4, which needs the harness; this uses **episode token length** as a placeholder, on the reasoning that it is difficulty-adjacent and is exactly the kind of quantity an encoding can move without anyone noticing.

**Finding 1 — A3 was vacuous, and nothing else would have said so.** On the first run every family showed *zero variance across encodings*: identical episode lengths, every encoding, every family. Our encodings differed only in which token served as the separator and in the per-episode symbol assignment. Rows described them as "three renderings, structurally different"; they were three renderings differing by punctuation.

A3 asks for a nontrivial `𝓔`. A family whose encodings are byte-identical in length and structure satisfies the letter of "sampled per episode" while testing nothing — and the transfer matrix would have reported encoding-invariance that was never at risk. **The test earned its place by finding that there was nothing to find.**

**Finding 2 — after strengthening, the effect is large.** Adding a `slotted` rendering (each value tagged with its position) to the concept families produced an immediate, large effect: **209 tokens vs 113 at k=1, and 307 vs 163 at k=3 — the slotted encoding is roughly 85% longer.**

That is a real exchangeability problem, not a cosmetic one. If episodes drawn from one family differ by 85% in length depending on a per-episode coin flip, then any per-episode loss average mixes an encoding effect into the family effect, and the §4 structural-content numbers that feed the matrix inherit it. **The harness must normalize** — per-token loss, or equal token budgets per episode — and this is a requirement it does not currently know it has.

**A limitation to state plainly.** With token length as the proxy, the quantity is *deterministic given the encoding*, so within-encoding variance is zero and the F-ratio degenerates to infinity. This is therefore not a statistical test as implemented; it is an exact structural check that answers "do the encodings differ in the proxy at all". The likelihood-ratio form of the source procedure only becomes meaningful with a stochastic difficulty proxy, i.e. once loss replaces length. That swap is the remaining work; the scaffolding and the two findings above did not need it.

**A third thing, found in passing.** Adding a field to `Encoding` silently broke the A2 permuted-alphabet check, because the check rebuilt the encoding field by field and the new field defaulted. It then failed an A2-compliant family, which looks exactly like a finding. Now fixed to copy-and-replace. Worth recording as a general hazard: **a check that breaks when the thing it checks gains a field is worse than no check**, because its failure is indistinguishable from a real result.
