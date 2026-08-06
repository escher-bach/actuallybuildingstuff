# Interface Findings

*Things learned by implementing against the Task Spec §7 protocol that are not visible from reading it. The protocol had never been implemented against before this work; friction found here is a result about the interface, and it belongs to the harness owner (Task Spec §8 step 1), not to the repertoire.*

Each entry: what broke, why it matters, what was done here, and what the harness has to decide.

---

## 1. `posterior(history, k)` cannot express the L3 target

**Severity: this one changes the training signal.**

Task Spec §2 defines the L3 target as "the target distribution over the answer token" — necessarily conditional on the query being asked. The §7 signature takes only `history` and `k`. There is nowhere to put the pending query.

It is not a cosmetic gap. Take the parity family (`shj_type_vi`): after one observation θ is *fully identified*, and yet the answer distribution **marginalized over a random query** is exactly 0.5/0.5, because parity is balanced over the query space. A harness treating that marginal as the L3 target would train the model to be maximally uncertain about a rule it has already pinned down. The calibration target would be teaching miscalibration — and it would do so most severely on exactly the families with the most structure, which is the hardest case to notice.

**Convention adopted here:** a trailing history entry whose answer is `None` is the pending query, and the posterior conditions on it. With no such entry the return is the marginal over the query space — a coherent object (it is what you should predict *before* seeing the query) but **not** the L3 training target.

**Harness decision needed:** either adopt this convention explicitly, or change the signature to `posterior(history, query, k)`. The second is cleaner. Either way it must be decided once and centrally, because a family that guesses differently will silently emit the wrong target.

## 2. The protocol was not runtime-checkable

`TaskFamily` is a `Protocol`, so `isinstance` raised `TypeError` and nothing could assert conformance. Fixed by adding `@runtime_checkable`. The check is shallow — it verifies methods exist, not that signatures or return types match — so it catches a family that forgot `posterior`, not one whose `posterior` returns the wrong shape. Treat it as a smoke test; the contract is enforced by per-family unit tests.

## 3. `evaluate` and `teacher_query` take no rng, which is load-bearing

Neither method receives a random source. Two consequences, both good once seen:

- **Stochastic oracles must return distributions, not samples.** §1.3 allows `f` to be distribution-valued, and that turns out to be the *only* way a stochastic family fits the protocol: `evaluate` returns `P(Y)` and the harness samples. This keeps seeded reconstructibility where §7 already requires it to live. Both `junk-random` and `probability-matching` are built this way.
- **`q*` must be a deterministic function of `(θ, history)`.** A7 asks for one-pass computability and this enforces it. Where a family has no informative query to choose — `junk-random`, where nothing is informative — the honest implementation returns a deterministic query cycled by history length rather than raising, so an L2 wrapper still runs over junk and the instrument keeps its floor at that level.

## 4. Being in the vocabulary is not the same as being in the alphabet

A live bug, caught by a test that was checking the wrong thing. The concept-family encoder drew "symbols" from `range(N_SYMBOLS)` — raw indices — rather than from `SYMBOL_IDS`, the actual token ids. Since the vocabulary lays out control and structural tokens first, the encoder was rendering task content as `PAD`, `BOS`, `EQ`, and so on.

Every id was a valid id, so the in-vocabulary test passed. **The A2 permuted-alphabet check also passed**, because the family was internally consistent about its wrong tokens — which is the alarming part: A2 verifies equivariance, and equivariance under the wrong alphabet is still equivariance.

Guarded now by a test asserting content tokens come from `SYMBOL_IDS`. The general lesson for anyone adding a family: **A2 passing does not mean the rendering is right.** It means the rendering is consistent.

## 5. The A2 check needs a *total* alphabet permutation

The first implementation permuted only the symbols the current encoding used. That cannot catch the failure it exists to catch: a family that renders some token from a fixed constant instead of from the encoding leaves that token untouched on both sides of the comparison, so the check passes.

With a total permutation of the symbol alphabet, the constant moves on the expected side and does not move on the actual side, and the leak shows. Verified by a deliberately-leaky subclass in the test suite that must fail the check — **a plant that passes its own A2 check proves nothing unless failure is reachable**, and the same goes for every family.

Structural tokens (separators, `ARROW`) are deliberately left fixed: they are notation, not content, and A2 is a claim about content symbols.

## 6. A rejection should not require a verified source

Not a protocol finding but a validator one, recorded here because it came from the same pressure. The register originally demanded a verified primary source for any row past `lead`, including `rejected`. That makes it *harder to reject a family than to accept one*, which is backwards — and rejection reasons are usually structural, readable off the form itself ("a one-bit hidden parameter cannot resist brute force") rather than off anyone's citation. What a rejection owes is its reason.
