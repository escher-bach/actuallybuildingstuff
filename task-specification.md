# Task Specification

*Sections 0–4 define what a task family is and what makes one admissible. Sections 5–9 are the build. Read §6 (worked example) early — it is faster than the formalism it instantiates.*

**Delegation note.** §5 is not an implementation task. It is the open research problem and it is not delegable; everything else is. What is delegable about §5 is the *harness* that will eventually consume its output, specified in §8 step 4. If you find yourself trying to derive the repertoire from this document, stop and escalate — the document deliberately does not contain it.

---

## 0. The system

An **inducer**: sequence → sequence, trained by per-token gradient against a target sequence.

Everything below must serialize to that interface. This is the only hard constraint; the rest are consequences of it.

---

## 1. Task family

A task family is not a set of input–output pairs and not a distribution over them. It is **a program with a hidden parameter and a policy for revealing it.**

$$\mathcal{T} = (\Theta,\; P_\Theta,\; \mathcal{X},\; f,\; \mathcal{E},\; \rho,\; k)$$

| | |
|---|---|
| $\Theta$ | space of rules; $\theta$ is the hidden parameter of an episode |
| $P_\Theta(\cdot \mid k)$ | rule sampler, conditioned on difficulty knob $k$ |
| $\mathcal{X}$ | query space |
| $f : \Theta \times \mathcal{X} \to \mathcal{Y}$ | the oracle — **evaluates, never searches** |
| $\mathcal{E},\ \rho_e$ | encoding space; $\rho_e$ renders objects to tokens |
| $k$ | difficulty knob |

The meaning of the family is given entirely by its sampler:

```
θ ~ P_Θ(· | k)          # the rule for this episode. Hidden by default.
e ~ P_E                  # the notation for this episode. Hidden by default.

for t = 1..T:
    x_t ← query          # sampled, or emitted by the model (L2)
    y_t = f(θ, x_t)      # forward evaluation against known θ
    emit ρ_e(x_t), ρ_e(y_t)

loss: cross-entropy on ρ_e(y_t) tokens only
```

Three properties, all load-bearing:

- **$\theta$ is sampled before any query.** Generation therefore runs backward from ground truth; the oracle is never asked to solve anything. This is what keeps cost flat in $k$.
- **$\theta$ persists across $t$, resampled between episodes.** The episode, not the pair, is the unit of training. Without this nothing above L0 is expressible.
- **$e$ is sampled on the same footing as $\theta$.** Notation is a per-episode latent, not a formatting decision made once by the implementer.

### 1.1 Composition

$\mathcal{T}_1 \circ \mathcal{T}_2$ is defined when the codomain of $f_2$ lies in the query space of $\mathcal{T}_1$:

$$\theta = (\theta_1, \theta_2), \qquad f\big((\theta_1,\theta_2), x\big) = f_1\big(\theta_1,\; f_2(\theta_2, x)\big)$$

with $P_\Theta$ the product of the component samplers and $\rho_e$ shared.

Composition matters for three separate reasons and should not be treated as a convenience:

- **It is the natural instantiation of $k$.** Depth is the difficulty knob that does not require inventing a new family per level.
- **It is the cheapest route to A4.** A solver that ignores intended structure must generally enumerate the composite; the intended solver composes.
- **It is what lets a finite basis cover an infinite target space.** Without closure you are enumerating, and enumeration cannot cover.

A5 (semantic coherence) is precisely the condition under which a composite is meaningful rather than a type-checking accident.

### 1.2 Traces

$f$ may optionally emit a derivation: $f^+(\theta, x) \to (y, \text{steps})$. This costs nothing — we own $\theta$ and can always run it — and where a family supports it, the trace is the teacher's working, loss covering the step tokens.

**Traces are not free of consequence, and the decision is per-family.** They install sequential state maintenance, which is the substance of reasoning as opposed to lookup. But a trace can also open an affordable brute-force path, and there is evidence that when it does, structural extraction *drops* — the model learns to simulate rather than to compress. This interacts directly with A4.

**Default: emit traces at low $k$, thin them as $k$ rises.** Rationale: at low $k$ the untraced task may be unreachable and the trace is the only thing making learning possible; at high $k$ the trace is the shortcut. Where a family declares traces, it must declare the thinning schedule alongside.

This is the least settled decision in the document. Treat the schedule as a swept parameter, not a constant.

### 1.3 Stochastic oracles

$f$ may be distribution-valued: $f(\theta, x) \to P(\mathcal{Y})$, with the emitted $y_t$ a sample and the L3 target the marginal over both $\theta$-uncertainty and oracle noise.

**This is optional but should not be forgotten.** With deterministic $f$ only, all uncertainty in the system is epistemic — about $\theta$ — and a model that has never seen genuinely noisy input given a known rule will be miscalibrated on input that is actually noisy. At least one family in the repertoire should exercise this.

---

## 2. Levels

Not four kinds of task. **Four settings of one dial**: the relation between what the loss demands and what the context determines. Formally, the residual entropy $H(y_t \mid \text{context})$ and the mode by which it can be reduced.

| | Reveal policy | $y_t$ determined by | Installs |
|---|---|---|---|
| **L0** | $\theta, e$ stated in preamble | $x_t$ alone | execution |
| **L1** | withheld; identifiable from prior $(x,y)$ pairs | context, after some $t^*$ | task inference, in-context learning |
| **L2** | withheld; $x_t$ **chosen by the model**, oracle appends result | informativeness of the model's own queries | agency, probing, error recovery |
| **L3** | withheld; not identifiable within the episode | irreducibly a distribution | calibration |

- **L2 stays inside the interface.** Model emits a query, generator appends the oracle's response, loss is masked off the oracle's tokens. One token stream. No environment, no reward, no rollout to terminal — see §2.1 for how gradient reaches the query.
- **L3 targets are exact.** We own $P_\Theta$, so the Bayes posterior over $y_t$ is computable. Calibration is a per-token cross-entropy target, not a human judgement. **This is unavailable for scraped text at any budget** — not expensive, structurally impossible, since nobody can label what a writer should have been uncertain about.
- **One generator serves all four.** The level is a wrapper argument, not a separate build.

### 2.1 L2 supervision — model queries, teacher-scored

The model emits $x_t$. Because we own $\theta$, we can compute what a well-chosen query would have been, and supervise against it.

```
for t = 1..T:
    x_t   ← model emits                       # self-selected
    x*_t  = q*(θ, history)                    # teacher's query, per A7
    y_t   = f(θ, x_t)                         # oracle answers what was ASKED
    emit ρ_e(x_t), ρ_e(y_t)

loss: cross-entropy on ρ_e(x_t) against ρ_e(x*_t)     # query channel
    + cross-entropy on the model's prediction of ρ_e(y_t)   # answer channel
    ( oracle's emitted tokens masked )
```

Two channels, both local, every token carrying gradient. Nothing waits for a terminal reward.

**Why not the alternatives.** Supervising only through the final answer reintroduces exactly the long-horizon credit assignment this design exists to avoid — that is reinforcement learning with extra steps. Teacher-forcing the query keeps gradient local but means the model never chooses during training, making it behaviour cloning of a prober; whether agency survives the switch to self-selection at evaluation is then an open empirical question rather than a design property.

**The oracle answers the query the model actually asked, not the teacher's.** This is what makes error recovery trainable: a poor query yields a genuinely uninformative response, the episode continues, and the model must recover from its own mistake with the teacher's better query visible as the target.

---

## 3. Admissibility constraints

**A1 — Backward-generable.** Instance cost $O(1)$ in $k$. Sample $\theta$, sample $x$, evaluate $f$. No search, no rejection sampling, no external solver on the generation path.

> *Why it binds:* forward-generate-then-verify couples generation cost to difficulty. Difficulty is where pedagogical value lives. A controller that dwells at the frontier would then spend its budget generating precisely the tokens it most wants, and throw them away on every backtrack.

> *A1 does not mean easy.* Generation cost and solution cost are independent axes: multiply/factor, execute/infer-the-program, inject-fault/localize-fault. The instruction is to search the asymmetric direction.

**A2 — Knowledge-free.** Semantics invariant under consistent permutation of the symbol alphabet. Checkable by running the generator with a permuted alphabet and diffing the answers.

**A3 — Encoding-varied.** $\rho_e$ sampled per episode from a nontrivial $\mathcal{E}$; the family invariant under $e$ in the sense of A2.

**A4 — Brute-force-resistant.** The shortest solver that does not use the intended structure must grow in $k$ strictly faster than $|f|$. Absent this, structural content collapses as soon as memorizing the lookup becomes affordable — and a synthetic generator is short by construction, so it has none of natural data's protection here.

**A5 — Semantically coherent.** Primitives shared across families denote the same operation. This is what makes a sequence of families *one curriculum* rather than $n$ unrelated pretrainings, and it is the only thing a controller has to exploit when ordering them.

**A6 — (L2 only) Transition cheap and total.** *Cheap*: one evaluation; no simulation, no optimality claim. *Total*: responds sensibly to malformed and invalid queries — the model will produce many, and a well-formed error is itself a recovery lesson.

**A7 — (L2 only) Computable teacher query policy.** The family must supply $q^*(\theta, \text{history}) \to x$, computable in one pass given $\theta$ — typically the query maximizing expected information gain about $\theta$ under the current posterior.

> *This is a real filter, not a formality.* For some families choosing the most informative query is harder than answering one, and those families admit L0, L1 and L3 but **not** L2. Determine this early: which families support L2 constrains the repertoire, and finding out late is expensive.

> Exact information-gain maximization is not required. A defensible heuristic policy is acceptable provided it is computable in one pass and documented as such — the target is a good query, not a provably optimal one.

### Not constraints — these are tasks

- **Encoding transfer** ("solve $N_2$ given a key") is an L1 family whose $\theta$ is the relabelling.
- **Error recovery** and **calibration** are reveal policies (L2, L3), not desiderata.

Anything that looks like a design requirement on the task set should first be checked for whether it is better stated as a member of the task set.

### Known limit

The oracle certifies **a** solution, not an optimal one. Families whose difficulty is "find the best" rather than "find one" fall outside the interface unless optimality is engineered in by construction.

---

## 4. The measurement

Two sections below depend on quantities that must be computed the same way every time. Both reduce to one procedure.

**Structural content, prequential estimate.**

1. Train a model on a stream of episodes from the family.
2. At each step, before training on a batch, record the loss on that batch. (Before, not after — the coding argument requires evaluating on data not yet seen.)
3. Let $L_i$ be that sequence and $L_{\text{final}}$ the converged loss.
4. **Structural content $\approx \sum_i (L_i - L_{\text{final}})$** — the area under the loss curve above its floor.

Interpretation: a family the model absorbs a great deal from shows a large sustained drop. Data with no learnable structure never drops, and trivially simple data drops instantly and stops. Both yield small areas, for opposite reasons.

**Conditional structural content $S(\mathcal{T}_j \mid m_i)$** — the same procedure, starting from a model already trained on family $i$ rather than from random initialization. This is the quantity §5 and §9 both consume.

**Compute budget must be held fixed** across any comparison, and reported. The quantity is budget-relative and comparing across budgets is meaningless.

**Within-episode acquisition slope.** Distinct from the above and separately logged: within a single L1/L2 episode, per-trial loss falls as $\theta$ becomes identified. The slope of that decay is the model's **acquisition rate** on that family — available as a training-time scalar with no downstream evaluation. See §9.

---

## 5. Repertoire — the open problem

*Not delegable. Stated here because the rest of the document is conditional on it.*

The constraints filter. They do not select. **Selection is the open problem, and it is the only open problem that is not plumbing.**

Target domains: reasoning, mathematics, coding, agent use. Required: families capturing the structure of those practices with minimal knowledge content, procedurally generable, with composition closing over the basis.

**Priority is the complement of what a text corpus teaches well** — recovery from error, calibration, task inference under underdetermination. That is L1–L3, and it is where every published procedural generator is absent.

**Extraction runs in two passes with different outputs:**

- *Primitives* — hand-write solver programs for real tasks; inventory the operations composed. Biases toward L0, because a program that solves a task need not search the way the human did.
- *Control structure* — record what the human did that the program did not need: backtracks, guesses, noticed ambiguity, declared uncertainty. Traces, not artifacts. Proof attempts not proofs; debugging sessions not repositories.

**Validation is coverage of established practice**, not elegance: a parametrization is credible when paradigms people independently converged on fall out of it as special cases. Held-out practice domains are mandatory — extracting primitives from a corpus and evaluating on that same corpus is circular.

**Do not stipulate the capability list.** Over-generate a redundant candidate basis, deliberately including suspected duplicates and suspected junk. Train tiny models per family. Measure all-pairs conditional structural content (§4).

- The **block structure** of that matrix is the capability decomposition.
- The **antisymmetric part** is the prerequisite ordering, thresholded into a DAG.

Both are read off, not designed. Run it first on a basis whose answer is known; if planted duplicates and junk are not recovered, measurement cannot guide extraction and the method fails before any expensive step.

---

## 6. Worked example — modular arithmetic under a hidden permutation

*One family, all four levels, to make the formalism concrete. Build this first. Token strings are illustrative; the exact surface form is the implementer's choice, subject to A3.*

**The family.**

| | |
|---|---|
| $\Theta$ | $(m, \pi)$ — modulus $m \in \{5,\ldots,20\}$, permutation $\pi$ of the symbol alphabet |
| $P_\Theta(\cdot \mid k)$ | $k$ controls $m$ and alphabet size; larger $k$ means larger $m$ |
| $\mathcal{X}$ | ordered pairs of symbols |
| $f$ | $f((m,\pi), (a,b)) = \pi\big(\pi^{-1}(a) + \pi^{-1}(b) \bmod m\big)$ |
| $\mathcal{E}$ | infix `a + b = c`, prefix `(+ a b) -> c`, tabular `a, b -> c` |

Why this family: it satisfies A1 trivially (evaluate, never search); A2 by construction, since $\pi$ *is* an alphabet permutation and the semantics are permutation-defined; A7 is computable, because posterior updates over $(m,\pi)$ are cheap. It fails A4 at small $m$ — the lookup table is memorizable — which is instructive rather than disqualifying, and is exactly what §8 step 5 measures. Compose it with a second family, or raise $m$, to recover A4.

**L0 — rule stated.**

```
MOD 7 | MAP q→0 f→1 z→2 k→3 w→4 h→5 t→6
q + f = f
z + k = w
f + w = h                    ← loss on this token
```

**L1 — rule withheld, identifiable.**

```
q + f = f
z + k = w
f + w = h
k + k = h
z + z = w
f + z = k                    ← loss on every answer token
```

Per-trial loss is high early and falls as $(m,\pi)$ becomes identified. That decay is the acquisition slope of §4.

**L2 — model chooses the queries.**

```
? f + f          →  z        model asked; oracle answered; teacher would have asked f + f  ✓
? q + h          →  h        model asked; teacher would have asked z + k                   ✗
? z + k          →  w
```

Query channel supervised against $q^*$; answer channel supervised as prediction of the oracle's response; oracle's emitted tokens masked. Here $q^*$ is the pair maximizing expected posterior entropy reduction over $(m, \pi)$ — computable by enumeration at these sizes. Note the second line: the model asked a poor question, got a genuinely uninformative answer, and must proceed from there. **That is the error-recovery lesson, and it is free.**

**L3 — episode ends before identification.**

```
q + f = f
z + k = w
f + z = ?                    ← target is the posterior, not a point
```

With two observations the posterior over $(m,\pi)$ is not concentrated. The target distribution over the answer token is computed exactly by enumerating $\Theta$ consistent with the history under $P_\Theta$. Cross-entropy against that distribution — a calibration target with no human in the loop.

**What the example demonstrates:** one $\Theta$, one $f$, one $\rho$, four wrappers. The level is a runtime argument. If your implementation requires four generators, the interface is wrong.

---

## 7. Interface contract

Every family exposes this. The harness knows nothing else about any family.

```python
class TaskFamily(Protocol):
    name: str
    supports_L2: bool          # False if A7 cannot be satisfied
    emits_trace: bool          # §1.2
    stochastic: bool           # §1.3

    def sample_theta(self, k: int, rng) -> Theta: ...
    def sample_encoding(self, rng) -> Encoding: ...

    def sample_query(self, theta, history, rng) -> Query: ...
    def teacher_query(self, theta, history) -> Query: ...      # A7; L2 only
    def evaluate(self, theta, query) -> Answer | Distribution: ...
    def trace(self, theta, query) -> list[Step] | None: ...    # §1.2

    def render(self, encoding, obj) -> list[int]: ...
    def preamble(self, theta, encoding) -> list[int] | None: ...   # L0 only
    def posterior(self, history, k) -> Distribution: ...           # L3 target

    def permuted_alphabet_check(self, rng) -> bool: ...        # A2 unit test
```

**Conventions the harness fixes, so families cannot disagree:**

- **Episode length** $T$ — a harness parameter, not a family parameter. Log it.
- **Loss mask** — a per-token boolean emitted alongside tokens. Family-rendered *answers* are supervised; family-rendered *queries* are supervised only at L2, against $q^*$; preamble and oracle-echo tokens are never supervised.
- **Tokenizer** — one shared vocabulary across all families, fixed before any family is written. A5 is unenforceable otherwise.
- **Batch composition** — episodes never split across batch boundaries. Whether a batch mixes families is a controller decision (§9), defaulting to single-family.
- **Seeding** — every episode reconstructible from `(family, k, level, seed)`. Non-negotiable: the whole design depends on being able to re-run a branch after a backtrack.

---

## 8. Build order

Each step has a stated completion condition. Do not proceed past a failing gate.

**1 — Harness, no families.** Episode loop, four level wrappers, masking, rendering, prequential logger (§4), seeded reproducibility. Test against a stub family with one rule and one encoding.
*Done when:* a stub episode round-trips at all four levels and the logger emits a structural-content number.

**2 — The worked family (§6), all four levels.**
*Done when:* the A2 permuted-alphabet check passes as a unit test; L1 shows within-episode loss decay; L2 runs with a computable $q^*$; L3 targets match a brute-force enumeration of consistent $\theta$.

**3 — Two more families of different character.** Suggested: one with state (stack or register machine), one with structure induction (grammar or sequence rule). One of the two should exercise §1.3 stochasticity, and at least one should be *tried* at L2 and allowed to fail A7 — knowing which families cannot support L2 is a result.
*Done when:* all three run through the same harness with zero family-specific code in the harness, and a composite $\mathcal{T}_1 \circ \mathcal{T}_2$ (§1.1) runs.

**4 — Measurement instrument, validated on a planted basis.** Assemble ~10 families where the answer is known in advance: include two near-duplicates and two deliberate junk families (e.g. one with random targets, one trivially constant). Run all-pairs conditional structural content.
*Done when:* the duplicates cluster, the junk families show near-zero structural content, and the prerequisite direction between an obviously-prerequisite pair comes out with the right sign. **If this fails, stop — measurement cannot guide extraction and §5's method is void.**

**5 — The dial sweep. This is the first result that matters.** Take the worked family. Vary residual entropy continuously — not four discrete levels, but a smooth sweep, e.g. by varying how many observations precede the query, or what fraction of $\theta$ is stated in the preamble. Measure structural content and transfer as a curve in that parameter.

| Curve shape | Reading | Action |
|---|---|---|
| Monotone collapse | Supervision requires determinate targets; everything above L0 needs reward | **Stop.** §2–§9 are void, and that is a real result worth writing up |
| Interior peak at low-but-nonzero entropy | The programme works; the four named levels are cut in the wrong places | Continue; relocate the cuts from the curve |
| Flat or rising | Something unexpected; the ambiguity model is wrong | Investigate before proceeding |

*Why a sweep and not a four-way comparison:* a discrete test tests whether *these particular four cells* were well chosen. A collapse at one hand-chosen construction would be ambiguous between "nothing above L0 works" and "that construction was badly built." The sweep separates them, and it locates the cuts rather than assuming them.

This is the only step that can end the programme rather than redirect it. It is reachable in weeks with steps 1–2 done, and it should not be deferred behind repertoire work.

---

## 9. Downstream: bias induction

*Conditional on §5 and on step 5 above. Do not build this first.*

Replace the bulk of text pretraining with synthetic curriculum over the families above. Knowledge goes to retrieval or a small terminal text stage.

**The claim under test:** capability and knowledge are separable; capability is the manufacturable half; a model with the capability half installed acquires the knowledge half faster than one trained from scratch.

**The primary metric is the within-episode acquisition slope** (§4). Per-trial loss falling within an L1/L2 episode as $\theta$ becomes identified *is* the meta-learning quantity this whole programme exists to install, readable during training with no downstream evaluation. **If it does not rise across a curriculum, nothing else in this section matters** — that is the first thing to plot, and the first thing to check when results look ambiguous.

**Controller signals** — all scalars the training loop already emits, all semantically meaningful because the generator knows $\theta$:

| signal | meaning |
|---|---|
| acquisition slope | within-episode identification rate — **primary** |
| solve rate | absolute competence |
| slope | rate of structural absorption across training; prequential estimate (§4) |
| saturation | slope flat, solve rate high → mastered |
| staleness | competence drop on a mastered family → measured forgetting |

**Loop:** select a family; train to mastery or stall; checkpoint; probe including stale families. On stall, restore weights to an ancestor and branch elsewhere. The controller consumes scalars only and never touches tokens, so throughput remains pretraining's.

**Gain, measurable before the terminal run:** the structural-content drop on a small text sample given the synthetic model, relative to random initialization. That is the payoff in bits, without reading the corpus.

**Watch for the inversion.** Families with more learnable structure carry *higher* absolute loss. A controller doing the right thing will look wrong on a loss dashboard. This is the easiest way for the work to be abandoned for the wrong reason.

> **Dependency direction.** The controller, the information-theoretic machinery, and any self-generated task extension are all conditional on §5. None is worth building against a repertoire that installs the wrong structure.
