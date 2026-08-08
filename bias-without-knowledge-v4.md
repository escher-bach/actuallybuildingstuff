# Bias Without Knowledge

### A prospectus for synthetic pretraining, derived from first principles

*Version 4. v1 derived the framework independently of epiplexity and is retained separately as a self-contained statement of the skeleton. v2 rebuilt it on Finzi et al. v3 repositioned it against the established L0 literature. v4 corrects an inconsistency in v3: the identifiability ladder was presented as derived when its rungs were abducted from current agent capabilities, which is the error II.0 exists to prevent. Corrections are marked where they occur.*

---

## The claim in one paragraph

Pretraining does two things at once: it teaches a model facts about the world, and it installs the inductive biases that make later finetuning and reinforcement learning possible. These two products separate cleanly under a compression criterion. **Bias is what pays for itself across instances — reusable structure that earns its place in a model by shortening every future description. Knowledge is what doesn't — idiosyncratic facts that cost as much to store in the model as to leave in the data.** Only the second is information-theoretically expensive: it can come only from the world, and no algorithm can manufacture it. Current practice buys both together at the price of the expensive one, and accepts whatever biases happen to fall out of a corpus assembled for other reasons. **This prospectus argues that the two should be factorized, derives what an inductive bias formally is, shows that the resulting theory determines its own next experiment, and specifies that experiment.**

The immediate next step is small and specified in Part IV: reproduce a published measurement on cellular automata, on consumer hardware, in days.

---

## What is new here, and what is conceded

Synthetic pretraining at L0 — deterministic input→output tasks with the loss on outputs — is no longer speculative. Jiang et al. (2026) show that front-loading 0.1–0.3% procedural data outperforms standard pretraining on natural language, code and informal mathematics, letting models reach the same loss on 55/67/86% of the tokens, validated to 1.3B parameters and 10.5B tokens, with gains persisting through downstream fine-tuning. Needle-in-a-haystack accuracy moves from 10% to 98% under Dyck pretraining. **This document cedes L0 entirely.** Anything it proposes that reduces to "generate deterministic task data and train on it first" is already done, better resourced, and published.

Two gaps remain, and they are what this document is for.

**Gap 1 — there is no selection principle.** LIME (Wu et al., 2021) is the only prior work with a principled basis for choosing tasks: Peirce's trichotomy of deduction, induction and abduction, imported from logic and instantiated as templates. One may dispute the carve; one cannot call it arbitrary. Everything since is organized by provenance rather than by principle. Jiang et al.'s categories — sequence transformations, memory operations, formal languages, cellular automata — are a bibliography, and the paper is candid about this: their headline finding, that $k$-Dyck aids context recall while ECA rule 110 aids reversed addition, was *discovered* by running the cross product, not predicted. Their contribution is measurement, and measurement without a generating principle does not tell you what to try next.

**Gap 2 — every published generator has a determinate target.** Sequence transformations, Dyck languages, stack simulation, cellular automata: all deterministic maps with the loss on output tokens. No latent task inferable from demonstrations, no oracle the model must query, no recovery from injected fault, and nothing that trains a model to hedge when the task is underdetermined.

Accordingly:

> **The contribution claimed here is (a) a coordinate system for situations whose axes are derived and whose gradations are measured rather than asserted, and (b) the extension of procedural pretraining above $H(p \mid \text{query}) = 0$ — that is, above determinate targets.**

Part I derives the coordinate system, Part II specifies it, and §IV.3 gives the measurements that populate it. **The single load-bearing risk is that the extension in (b) does not work** — that procedural pretraining succeeds *because* its targets are determinate, and that raising residual entropy destroys the dense verifiable signal the approach depends on. That risk is V.1, and it is probed by a sweep costing a handful of small runs (§IV.3).

---

## 0. Why the entanglement matters

Pretraining is usually described as compression: a predictor is a compressor and vice versa, and a larger model compresses a corpus better. That description is accurate and unhelpful, because it says nothing about *what the compression is for*. Downstream, we do not use the pretrained model directly. We finetune it, we run RL on it, we prompt it in context. The pretrained checkpoint is not a hypothesis about the world; it is a **learner**, and its value is how cheaply it acquires the things we later ask of it.

So pretraining has two products:

| Product | Definition | Compression test | Where it can come from |
|---|---|---|---|
| **Knowledge** | Reduces error on tasks whose answers were in the corpus | Storing it in the model saves as much as it costs — no net gain | Only the world |
| **Bias** | Reduces the *cost of acquiring* tasks not in the corpus | Pays for itself across every instance that reuses it | Any sufficiently structured process |

The asymmetry is real but is **not symmetric in how well it is established**, and v1 of this document overstated one half of it. Stated carefully:

> **Knowledge side (theorem).** Let a corpus $D_n$ be produced by a generator program $G$ plus randomness. For any real-world target $f$, $\;I(D_n; f) \le K(G) + O(\log n)$. By the data processing inequality, running an algorithm cannot create information *about the world*. A synthetic corpus of any size carries at most the world-information of its generator. **Synthetic data can never supply knowledge.**

> **Structure side (partly proved, mostly empirical).** For a computationally bounded observer the ceiling does not hold. Finzi et al. (2026) prove that a deterministic map can dramatically increase *time-bounded entropy* — a CSPRNG output has near-maximal $H_{\text{Poly}}$ despite a short generator. For *structural* content the corresponding statement is weaker: they prove only that high-epiplexity variables exist assuming one-way functions, at $S_{\text{Poly}}(X_n) = \Omega(\log n)$, and note this is far short of the power-law scaling natural data exhibits. That computation *creates structure* is presented as a conjecture — the failure of a "Limited Epiplexity Increase Property" — supported by experiment rather than proof.

So the foundation is a theorem on one side and a well-evidenced conjecture on the other. This is not a defect that better work will fix: Chaitin incompleteness means unbounded-compute sophistication above a few thousand bits is provably un-exhibitable, so **the field can only be empirical.** Plan accordingly.

The resulting proposal is therefore not "replace pretraining." It is:

> **Factorize pretraining into a bias phase and a knowledge phase.** The bias phase is synthetic, compute-scalable, and bounded only by generator description length. The knowledge phase is real corpora or retrieval, and is information-bounded and irreducible.

The empirical bet underneath this — *the bias content of web pretraining, measured in bits, is orders of magnitude smaller than its information content* — is **no longer a bet.** Finzi et al. measure the decomposition directly on OpenWebText, Lichess, and CIFAR-5M: in all cases the structural component is a tiny fraction of the total, with over 99% of CIFAR-5M's information being random. Language carries the most structural content, images the least, which is their explanation for why text pretraining transfers broadly and image pretraining does not. The central premise of this document is now a measured ratio.

Separately, there is evidence that concentrating that content pays. Lee, Han, Kumar & Agrawal (2026) report that 164M tokens of neural-cellular-automata rollouts used as a pre-pretraining stage improved downstream language modeling by up to 6% and accelerated convergence by up to 1.6×, outperforming a pre-pretraining stage of 1.6B Common Crawl tokens — roughly a 10× ratio, obtained by installing the simplest bias in the family (predicting dynamics). An existence proof, not a result about reasoning.

---

# Part I — What an inductive bias is

Everything below is derived without reference to neural networks. The point of the derivation is that it ends by dissolving a question we would otherwise have argued about on aesthetic grounds.

## I.1 The primitive problem

Fix a domain $X$, codomain $Y$, and let $\mathcal{F} \subseteq Y^X$ be the possible targets. Data is a finite partial observation $D = ((x_i, y_i))_{i \le n}$.

> **Fact 0 (Underdetermination).** If $\mathcal{F} = Y^X$, then for any $D$, any unobserved $x$, and any $y \in Y$, some $f \in \mathcal{F}$ agrees with $D$ and has $f(x) = y$.

A learner's output at an unseen point is therefore *not a function of the data*. It is a function of the data **and something else**. Inductive bias is the name of that something else. This is forced, not chosen, and it already fixes the shape of the answer: bias is whatever must be added to data to make prediction a function.

## I.2 Four traditions, one object

Four fields answer this apparently differently.

- **Mitchell (1980), logical.** Bias is a minimal set of assumptions $B$ with $B \wedge D \wedge x \vdash L(D)(x)$ — the premises that turn induction into deduction. A $\{0,1\}$-valued weighting on $\mathcal{F}$.
- **Bayesian.** Bias is a prior $\pi$. A $[0,1]$-valued weighting.
- **Statistical learning theory.** Bias is a hypothesis class $\mathcal{H}$, a structure $\mathcal{H}_1 \subseteq \mathcal{H}_2 \subseteq \cdots$ under SRM, or a regularizer $c: \mathcal{F} \to \mathbb{R}_+$. A real-valued weighting.
- **Algorithmic information theory.** Bias is the choice of universal machine $U$; $K_U(f)$ is complexity relative to it. A $[0,\infty]$-valued weighting.

These are one object in different units. Mitchell's is a Bayesian prior valued in $\{0,1\}$. SRM's penalty is a prior via $\pi \propto 2^{-c}$ — which is why SRM penalties always look like codelengths. $K_U$ induces Solomonoff's $M(f) = 2^{-K_U(f)}$. The common currency is **cost in bits**, and the only constraint is that costs be realizable as codelengths.

> **Definition 1.** A **bias** on $\mathcal{F}$ is a function $\beta : \mathcal{F} \to [0,\infty]$ satisfying
> $$\sum_{f \in \mathcal{F}} 2^{-\beta(f)} \;\le\; 1.$$

By Kraft–McMillan this is exactly a prefix code on $\mathcal{F}$, and equivalently a semi-measure $\pi_\beta = 2^{-\beta}$. Three interchangeable readings:

| Reading | Makes obvious |
|---|---|
| **Cost function** | Optimization, regret |
| **Code** | Composition, transport, description length |
| **Semi-measure** | Inference, mixture, conditioning |

The code reading is the one to hold in mind: **a bias is a language for describing targets, and it favors whatever it can say briefly.**

*Why sub-probability?* Partly computability — Solomonoff's $M$ is only a semi-measure because machines may not halt. More importantly, it is what makes the algebra of §I.5 close: probability measures are not constructively closed under restriction, semi-measures are. The $\le$ is load-bearing.

> **Proposition 1 (No Free Lunch = Kraft).** Since $\sum_f 2^{-\beta(f)} \le 1$, lowering $\beta$ on one region necessarily raises it elsewhere.

NFL is not a deep fact about learning. It is the observation that a bias has one unit of belief to spend. Every claim about inductive bias is a claim about **where the mass went**.

## I.3 What is observable about a bias

Definition 1 is intensional. Two different weightings may be behaviorally identical, so we need the extensional content. A bias induces a marginal on data,
$$M_\beta(D) \;=\; \sum_f 2^{-\beta(f)} P_f(D), \qquad M_\beta(y \mid x, D) = \frac{M_\beta(D \cup (x,y))}{M_\beta(D)}.$$

> **Proposition 2 (Extensionality).** All predictive behavior of $\beta$ is determined by $M_\beta$. Hence $\beta_1 \equiv \beta_2$ iff $M_{\beta_1} = M_{\beta_2}$. The map $\beta \mapsto M_\beta$ is many-to-one; its fibers are the behavioral equivalence classes.

**The only observable of a bias is the codelength it assigns to data,** $-\log M_\beta(D)$. Anything that cannot be phrased as a claim about prequential codelength on some data distribution is not a claim about the bias. This is not a philosophical remark — it is the specification of the measuring instrument, and it is why the evaluation methodology in Part IV looks the way it does.

## I.4 The second resource: compute, and the budget-indexed code

So far, only *identification*: how many bits of evidence single out $f$. But Solomonoff's $M$ is uncomputable and every real learner is search-limited. Levin's $Kt(f) = \min_p\{|p| + \log \mathrm{time}(p)\}$ points the right way but combines the two resources by an arbitrary rule. **Epiplexity** (Finzi et al., 2026) does it properly, and this document adopts their construction and notation wholesale.

> **Definition (epiplexity and time-bounded entropy).** Fix a time bound $T$ and let $\mathcal{P}_T$ be the programs that both sample and evaluate probabilities within $T$ steps. Let
> $$P^\star = \arg\min_{P \in \mathcal{P}_T}\big\{\, |P| + \mathbb{E}[\log 1/P(X)] \,\big\}.$$
> Then $S_T(X) := |P^\star|$ is the **epiplexity** — the structural content a $T$-bounded observer can extract — and $H_T(X) := \mathbb{E}[\log 1/P^\star(X)]$ is the **time-bounded entropy** — the residual that must be stored rather than derived. Write $\mathrm{MDL}_T = S_T + H_T$.

Three things make this the right object rather than a variant of resource-bounded Kolmogorov complexity. It is grounded in **cryptographic indistinguishability**, not just clocked programs: a CSPRNG has near-maximal $H_T$ and near-constant $S_T$ for polynomial observers, which resource-bounded $K$ cannot express since the generator is short and fast. The bound covers **both finding and running** the program, so a naive time-bounded sophistication collapses to $O(1)$ while this does not. And the split is **observer-relative**: the same object is structure to one budget and noise to another.

**This replaces v1's two-coordinate story.** A bias is not a point $(\beta, \tau)$ on a Pareto curve but a one-parameter family $S_T$ indexed by budget. The conservation asymmetry survives in altered form — $H_T$ behaves like a resource you cannot argue your way out of, while what counts as extractable structure moves with $T$ — but the frontier is now a single measurable curve rather than two quantities to be traded off by hand.

Their scaling analysis matters for planning: $S_T$ typically **grows** with compute and with data, $H_T$ falls, and for a fixed test set the asymptote is $S_\infty = \tfrac{\beta}{1-\beta} D_0^{\beta} D^{1-\beta}$ — capped by *dataset size*, not compute. So there is no generic decay in the value of structural extraction as models grow. (v1 worried there was; that worry was wrong. A sharper, synthetic-specific version replaces it in V.3.)

**Bias density** falls out as a usable quantity: $S_T/|X|$, structural bits per token. Since smaller $\beta$ and larger $D_0$ give higher epiplexity, and $\beta$ is a **loss-curve exponent**, a corpus's bias density can be estimated from its scaling behaviour without any downstream evaluation. Slower loss improvement means more absorbed per token.

## I.5 Why the two products separate, and what that implies for retrieval

The two-part code makes the factorization mechanical rather than aspirational. Consider what the MDL minimizer does with a list of idiosyncratic world facts: memorizing $n$ of them costs about $n$ bits in $|P|$ and saves about $n$ bits in the residual. No net gain, so they stay in $H_T$. Reusable structure is different — it is paid for once and redeemed by every instance that invokes it, so it is amortized into the program.

> **Bias is what pays for itself across instances. Knowledge is what doesn't.** The two-part code separates them by compression gain, not by subject matter.

This exposes something worth stating plainly. Real language models *do* memorize facts into weights, because over-parameterization removes the pressure not to. **The entanglement this document objects to is precisely the gap between what pretraining does and what a compression-optimal learner at the same budget would have done.** The retrieval half of the proposal is therefore not a convenience or an engineering shortcut: it is putting $H_T$ where the two-part code says it belongs — in an external store — so that the weights carry $S_T$ alone.

## I.6 Operations on biases

Each operation is derived; the content is its *cost*.

**Restriction.** For $\mathcal{G} \subseteq \mathcal{F}$: $\beta|_\mathcal{G}(f) = \beta(f)$ on $\mathcal{G}$, $\infty$ off it. Kraft survives since the sum only drops — this is where sub-probability earns its keep, as renormalizing would require $Z_\mathcal{G} = \sum_\mathcal{G} 2^{-\beta}$, which is not computable in general. Renormalized restriction is conditioning, at $\beta(f) + \log Z_\mathcal{G}$.

> **Proposition 4.** Restricting to a set of prior mass $2^{-k}$ buys exactly $k$ bits, uniformly on every survivor.

That is the exact worth of a hypothesis class, and it recovers SLT: $\log|\mathcal{H}| = k$ gives $k$-bit codes and $\sqrt{k/n}$ bounds.

**Mixture ($\oplus$).** $\beta_1 \oplus_\lambda \beta_2 = -\log(\lambda 2^{-\beta_1} + (1-\lambda)2^{-\beta_2}) \le \min_i \beta_i + \log(1/\lambda)$.

> **Proposition 5.** Mixing $N$ biases costs $\log N$ bits and inherits the best of them everywhere.

Hedging is logarithmically cheap. This is why universal priors exist: mix over every machine and pay the index. It is also why "include more kinds of structure" is a weak design principle — it is cheap *because* it is uncommitted, and commitment is what buys anything.

**Composition / relativization ($\circ$).** A bias is a language, so a target can be described *in terms of* something already described:
$$(\beta_2 \circ \beta_1)(f) \;=\; \min_g \big[\beta_1(g) + \beta_2^{(g)}(f)\big], \qquad K(f) \le K(g) + K(f \mid g) + O(\log).$$

> **Proposition 6 (Chain rule).** Layered biases compose additively.

This is the formal basis of curriculum, abstraction, and library learning. Installing $g$ first saves exactly $\beta(f) - \beta(f \mid g)$ — the mutual information between abstraction and task. Layering pays iff that mutual information is large; layering an irrelevant $g$ costs $\beta(g)$ for nothing.

**This proposition was inert in v1 and v2 because the saving looked unmeasurable. It is not.** Finzi et al. define *conditional* epiplexity $S_T(Y \mid X)$, and note explicitly that conditioning on a deterministic string — a trained model $m$ — lets one ask what additional data is most useful given that model, a direction they flag and do not pursue. That is the chain rule with an estimator:

> **Curriculum as a measured DAG.** For content types $\{t_1,\dots,t_n\}$, train a small model $m_i$ on each and measure $S_T(t_j \mid m_i)$ for every pair. The matrix is a weighted directed graph over types, where a large drop from $S_T(t_j)$ to $S_T(t_j \mid m_i)$ means $t_i$ is a prerequisite that pays. **Stage selection becomes a topological order on a measured graph rather than an assertion.**

At the scale Jiang et al. use for diagnostics — two layers, four heads, hidden size 16 — a ten-type matrix is a hundred tiny runs. This is the rigorous basis for stage selection that Gap 1 identified as missing, and it is days of compute (§IV.3).

**Transport ($\sigma_*$).** For computable $\sigma : \mathcal{F} \to \mathcal{F}'$, define $2^{-\sigma_*\beta(f')} = \sum_{f \in \sigma^{-1}(f')} 2^{-\beta(f)}$.

> **Proposition 7.** $\sigma_*\beta(\sigma f) \le \beta(f)$, and re-encoding along $\sigma$ costs at most $K(\sigma)$ bits.

Pushforward never increases complexity — mass accumulates on fibers. A bias moved into a new representation is preserved up to **the description length of the translation**. This is the formal core of every transfer claim: transfer is cheap exactly when the encoding map is simple. It is also the invariance theorem in disguise, since simulation is transport along a compiler.

**Assembly (weight-space composition).** *(New in v3, and the algebra above does not model it.)* Biases can be combined not only by mixing data but by assembling trained components. Jiang et al. assemble the attention layers of a model trained on Set with the MLP layers of a model trained on ECA rule 110; the composite beats every single-source model across four tasks, 89.3 average against a 57.4 baseline, where each single source is weak somewhere. More striking, **selective transfer can beat full transfer** — attention-only gives an 80-percentage-point improvement over full-model transfer on one pair — which means a fully trained model carries *anti-transferable* structure alongside the useful kind.

Proposition 5 does not describe this. Mixture costs $\log N$ and inherits the best of its components everywhere; assembly is empirically superadditive in some pairs and negative in others, and it is architecture-indexed rather than function-indexed. **This is a genuine hole in the algebra.** Two consequences meanwhile hold regardless: a bias is not a scalar attached to a model but something localized in parts of it, and "transfer everything" is a default that costs accuracy.

**Ordering.** Two orders, and their relationship is the interesting part.

- *Intrinsic:* $\beta_1 \preceq \beta_2$ iff $\beta_1 \le \beta_2 + c$. A preorder — and by the invariance theorem it **collapses at the top**: all universal biases are equivalent.
- *Extrinsic:* fix a task measure $\mu$ on $\mathcal{F}$; then $\beta_1 \preceq_\mu \beta_2$ iff $\mathbb{E}_{f\sim\mu}[\beta_1] \le \mathbb{E}_{f\sim\mu}[\beta_2]$. Total — but only exists once $\mu$ is named.

The order requiring no external input is useless at the top; the order that discriminates requires an exogenous $\mu$. **"Which bias is better" is not an intrinsic question.** At finite data this bites hard: two biases are distinguishable at sample size $n$ only if they differ by $\Omega(n)$ bits, so at any finite budget the invariance-theorem constants *are* the subject matter.

## I.7 The regret theorem

Evaluating $\preceq_\mu$ by Gibbs' inequality gives the result the whole derivation points at.

> **Theorem (Regret of a bias).**
> $$\mathbb{E}_{f \sim \mu}[\beta(f)] \;=\; H(\mu) \;+\; D_{\mathrm{KL}}\!\big(\mu \,\big\|\, 2^{-\beta}\big).$$

Three consequences:

1. The excess cost of a bias is exactly its **KL divergence from the task measure**.
2. The floor is $H(\mu)$ — irreducible. No bias beats the entropy of the task distribution.
3. The optimal bias **is** $\mu$ itself.

Therefore: **bias design is density estimation on task space.** Not metaphorically — the objective is a KL projection of the task distribution onto the set of biases one can actually construct.

This closes independently with SLT. PAC-Bayes bounds the generalization gap by $\sqrt{(D_{\mathrm{KL}}(Q\|P) + \log(1/\delta))/2n}$, whose complexity term is literally KL-to-prior. The frequentist and algorithmic routes arrive at the same functional, which is evidence the object is real rather than an artifact of one formalism.

## I.8 The space of biases

| Face | Structure |
|---|---|
| **Set** | Semi-measures on $\mathcal{F}$ ≅ prefix codes ≅ Kraft-satisfying cost functions |
| **Algebra** | The log-probability semiring $([0,\infty], \mathrm{lse}, +)$: $\oplus$ and $\circ$ are its addition and multiplication. Zero-temperature limit is tropical $(\min,+)$ — the MAP/shortest-description approximation |
| **Geometry** | Convex; extreme points are dogmatic point masses, so every bias is a mixture of dogmas. Dually flat (Amari) with KL as canonical divergence; design is $m$-projection with a Pythagorean loss decomposition |
| **Order** | Preorder collapsing at the top under invariance; totalized only by choosing $\mu$ |
| **Category** | Fibred over domains-and-computable-re-encodings, with $\sigma_*$ as transport and $K(\sigma)$ its cost. Isomorphism = bidirectional translation with both compilers short |

Mixture and composition are not two arbitrary combinators; they are addition and multiplication of a semiring. If one object must be named, take the computational face:

> **A bias is an interpreter. The space of biases is the space of partial computable interpreters, quotiented by behavioral equivalence, ordered by compilation cost, graded by the compute needed to run them.**

A code is a machine; composition is running one interpreter atop another; transport is compiling; invariance is universality; the grading $T$ is runtime. Every proposition above reads naturally in this language.

## I.9 What Part I has already decided

**(a) The design problem is a projection.** Minimize $D_{\mathrm{KL}}(\mu \| 2^{-\beta})$ over the reachable set of biases. Both arguments are now the things to argue about — what is $\mu$, and what is reachable. Nothing else is a design question.

**(b) Two numbers determine feasibility.** $H(\mu)$, the irreducible floor; and the reachable set's distance from $\mu$. If $H(\mu)$ is large, no compact bias helps regardless of cleverness.

**(c) Substrate and bias are the same question.** A substrate is an interpreter; an interpreter is a code; a code is a bias. So the substrate question is not *which formalism is convenient* but: **whose induced semi-measure has small KL to the task measure, and what does its $S_T$ curve look like?** Notational convenience is $K(\sigma)$ — real, bounded, and by Proposition 7 the *smallest* term in play.

Point (c) is the reason this document exists. Most substrate debates — Lisp versus a stack language versus a sequent calculus — are arguments about the least significant term. What determines $\beta$ is the **primitive basis**: which operations get unit cost. That is orthogonal to syntax. The same basis in Lisp or Forth yields nearly the same bias; two different bases in the same Lisp yield entirely different ones.

---

# Part II — Design consequences

Part I fixed the objective. This part derives the shape of the system that pursues it. Nothing here is a preference; each item is forced by something above.

## II.0 Two things to fix before any design: what $\mu$ is, and what "type" means

**The anti-Platonic constraint.** The formalism of Part I invites a specific error, and it should be blocked explicitly rather than left to good sense.

> $\mu$ is a distribution over **practices**, not over natural kinds. Competition mathematics is a genre with conventions. Debugging is a craft shaped by contingent tooling. Retrieval as we mean it is shaped by how documentation happens to get written. **There is no primitive set that mathematics "really" has — only the set that the way humans currently do mathematics has.** The basis must therefore be read off practice, cannot be derived, and will drift as practice drifts.

This is not a hedge; it is a positive claim with a named rival. Jiang et al. explain their transfer to vision by invoking the **Platonic Representation Hypothesis** (Huh et al.) — modality-agnostic mechanisms that any competent learner converges on. The two views make different predictions. The Platonist expects one fixed basis to work across domains and to keep working; the practice-relative view expects the basis to shift with the tasks and to need periodic re-extraction. That is not a philosophical preference: **it decides whether Track B (§IV.4) is a one-time exercise or a standing instrument.**

It also disciplines R7. Rule 54 is a good generator because of properties of the *automaton*. Whether class-IV dynamics is the right bias depends on whether human mathematical practice has class-IV-like structure — a question about practice, not about automata. The theory constantly tempts this substitution.

**Axes are derived; gradations are not.** *(Corrected in v4.)* v3 exempted one axis from the constraint above, claiming the inference-form levels were derived from the theory while only the content primitives had to be read off practice. That was wrong, and the tell was visible in v3's own table: the biases those levels installed read *execution, in-context learning, agency, retrieval* — a list of current LLM product categories. No information-theoretic argument produces that list. The levels were abducted from what agents are currently asked to do, then fitted to an axis.

The correct distinction is one level down:

| | **Axis** | **Gradation along it** |
|---|---|---|
| What it is | The dimension itself | Where the useful cuts fall |
| Content | Which operations a task composes | *Which* primitives — arithmetic, binding, selection |
| Inference | Residual task entropy $H(p \mid \text{context})$ and the mode by which it can be reduced | *Which* levels are worth distinguishing |
| Status | Derived. The collision problem of §II.4 forces the inference axis; composition forces the content axis | **Empirical on both. Read off practice** |
| How to answer | — | Track B for content (§IV.4); Track B′ for inference (§IV.5) |

Temperature is a derived physical quantity; *cold, warm, hot* is a practice-relative coarse-graining of it. v3 had those two things fused on the inference axis. The anti-Platonic constraint governs **every gradation in this document**, including the ladder in §II.4 — which is more consistent than v3, not less, since it removes a rule with an exemption carved out for the author's favourite construction.

**The type split.** v1 and v2 also used "situation type" in two incompatible senses. Content type and inference form are orthogonal, and their ordering questions differ:

| | **Content type** | **Inference form** |
|---|---|---|
| Ordering question | Which prerequisites pay? | How much can be withheld before signal collapses? |
| How to answer | Measured DAG via conditional epiplexity (§I.6) | The continuous ambiguity sweep (§IV.3, V.1) |
| Prior art | LIME by principle; everything since by provenance | **Empty. Nothing above $H=0$ exists.** |

The product of the two axes, together with the four complexity axes of R6, is the coordinate system this document offers in place of a taxonomy.

## II.1 Sharpness, and why it is more than scope reduction

The system need not know everything a general LLM knows. It targets a sharp $\mu$: **mathematics, reasoning, coding, retrieval, agentic tool use.** By the regret theorem this directly attacks $H(\mu)$, the only quantity that no bias can improve. Narrowing $\mu$ lowers the floor.

But sharpness does more than shrink the target. It converts the knowledge/bias factorization from an aspiration into a **generation-time filter**.

> **Definition (context-closure).** A situation is *context-closed* if $H(y \mid \text{context}) = 0$ — the target is a fully specified function of what is present in the context, requiring no facts from outside it.

Web pretraining is overwhelmingly *not* context-closed: predicting the next token of an encyclopedia article requires knowing things. **That non-closure is precisely the mechanism by which knowledge and bias become entangled.** So "generate only context-closed situations" enforces the central thesis mechanically, at the point of data generation, rather than hoping for it after the fact.

## II.2 Retrieval is not a bias you add — it is a consequence

If every situation is context-closed and contexts are permitted to be large and mostly irrelevant, the model *must* learn selection-from-context, because that is the only route to correctness. Retrieval falls out of context-closure plus large contexts. It requires no separate generator.

That a first-class capability arrives as a corollary of a rule introduced for another reason is weak evidence the frame is right.

## II.3 The situation generator

Do not build a set of (instruction, output) pairs. Build a **prior over programs** and generate episodes.

> **Definition (situation generator).** A program space $\mathcal{P}$ with semantics $[\![\cdot]\!] : \mathcal{P} \to (X \to Y)$; a prior $\pi$ over $\mathcal{P}$; an input distribution $\nu$; a description channel $\mathrm{Desc}(\cdot \mid p)$.
>
> **Episode:** sample $p \sim \pi$; sample $x_1,\dots,x_{k+1} \sim \nu(\cdot \mid p)$; set $y_i = [\![p]\!](x_i)$; optionally sample $d \sim \mathrm{Desc}(\cdot \mid p)$; serialize
> $$e = (d,\; x_1,y_1,\; \dots,\; x_k,y_k,\; x_{k+1}), \qquad \text{target } y_{k+1}.$$

> **Proposition 8 (Amortization).** The log-loss-optimal predictor of $y_{k+1}$ is the posterior predictive
> $$\int_{\mathcal{P}} [\![p]\!](x_{k+1}) \; d\pi\big(p \mid d, x_{1:k}, y_{1:k}\big).$$

Training to optimality on such a corpus **is** amortized Bayesian program induction with prior $\pi$. The generator's prior is the installed inductive bias by construction, not by hope. This is the object Part I called $\beta$, now realized as something one can actually run: $\pi(p) \propto 2^{-|p|}$ is computable, resource-bounded Solomonoff.

The practical significance is that it replaces an ill-posed question ("what situations should we generate?") with a well-posed one ("what prior do we want?").

*Caveats, stated honestly.* Convergence is never reached, capacity is finite, and the architecture determines which $\pi$ are amortizable at all; the failure to converge is itself a source of real inductive bias, and it is architecture-specific (V.1).

**One caveat is a correction to v1, and it cuts in our favour.** As stated, Proposition 8 implies the learner recovers $\pi$ and no more — you cannot exceed your generator. Finzi et al.'s Paradox 3 shows this is false for a bounded learner: in their masked-latent experiments the program required to *predict* is strictly larger than the program required to *generate*, and strategies appear in the learner that were never present in the generating process. The proposition describes the unbounded optimum; a compute-limited learner, unable to brute-force the inverse, must construct machinery the generator never had. **Read Proposition 8 as a lower bound on what gets installed, not a ceiling.**

## II.4 The inference axis — the document's central proposal

*Everything published sits at $H = 0$. This section is the contribution. The axis is derived; the rungs below are a coarse-graining and are not.*

The classic collision problem — the input `"abc"` mapping to `"abc"` under repetition and to `"d"` under continuation — appears to demand that every instruction encode its task type. It does not. The correct requirement is that the task be identifiable **from the episode**, not from the query.

Define residual task entropy $H(p \mid \text{context})$:

- $H(p \mid \text{query alone}) = 0$: the model learns execution and never learns task inference. No in-context learning bias is installed.
- $H(p \mid \text{full episode}) \gg 0$: the target is a mixture, gradients are noise, and destructive interference results.

> **Invariant.** $H(p \mid \text{query})$ large; $H(p \mid \text{episode})$ small.

Under this invariant the collision is *desirable*: residual entropy is the pressure that forces construction of a task-inference mechanism. Train a mixture — fully-described episodes teach semantics, demonstration-only episodes teach inference, partially-described episodes teach the combination. The mixing ratio is a real hyperparameter, and ICL quality is expected to be non-monotone in it.

Making the entropy reducible only by *acting* yields agency. The axis is therefore not "how hard is the task" but **by what means the residual entropy can be removed**, and that admits an exhaustive partition: it needs no removal, it yields to observation, it yields only to action, or it yields to nothing available. Three modes plus the complement.

| Cell | Condition | Form | Bias installed | Literature |
|---|---|---|---|---|
| **L0** | $H(p \mid \text{query}) = 0$ | $(x) \to y$, all present | Execution | Established to 1.3B. Ceded |
| **L1** | Reducible by observing | $(\text{demos}, x) \to y$, latent task | Task inference, ICL | Untried |
| **L2** | Reducible only by acting | $(x, \mathcal{O}) \to \text{trace} \to y$ | Agency, tool use | Untried |
| **L3** | Irreducible within budget | $H(p \mid \text{all obtainable}) > 0$ | **Calibration — hedging, and knowing when you don't know** | Untried |

Two corrections to v3's version of this table, in opposite directions, which is the main reason to believe the axis is real rather than retrofitted.

**Retrieval was removed.** v3's fourth rung was "large context, small $T$," which is not a claim about residual task entropy at all — it is input entropy with low serial time, and that is already R6's fourth complexity axis. Retrieval was promoted onto this ladder because retrieval is a capability people care about, not because it lives on this dimension. It has been returned to R6. (The §II.2 argument that retrieval falls out of context-closure plus large contexts is unaffected; it never depended on the ladder.)

**Calibration was added.** The partition demands a cell for entropy that no observation and no affordable action can remove, and the correct behaviour there is calibrated hedging rather than a confident answer. This never appeared in v3 — not because the theory excluded it, but because "knowing when you don't know" is not a product category the way tool use is. It is arguably the bias whose absence is most conspicuous in deployed models. **A retrofitted axis regenerates only the cells it was fitted to; this one produced one that was missing and rejected one that was present.**

*Standing caveat.* Four cells remain a coarse-graining of a continuous quantity, chosen because each has a recognizable name. Under II.0 that choice is empirical and provisional. §IV.3's sweep is designed to locate the useful cuts rather than to assume these ones.

Sequence transformations, Dyck languages, stack simulation and cellular automata all sit at L0: a deterministic map with the loss on output tokens. The nearest thing to L1 in the literature is Finzi et al.'s masked-latent *diagnostic*, built to study induction rather than to install it.

Generation stays cheap at every level: because $p$ was constructed, the optimal query policy is known, so the teacher trace remains free and remains at SFT throughput. No reward model, no RL.

**The highest-value member of this family is nearly free to produce: inject faults into the oracle and have the teacher trace recover.** Error recovery is the bias that most separates usable agents from unusable ones; it is nearly absent from natural corpora, because people do not publish their failed attempts. It costs a random-error injector and a recovery-aware teacher. If any single situation type justifies this program economically, it is that one.

## II.5 Design rules

Each follows from Proposition 8 rather than from taste.

**R1 — Support, read off practice.** If $\mu$ has mass where $\pi$ has none, no bias is installed there, at any scale. Take real downstream tasks, write programs solving them, verify those programs are short under $\pi$. If they are not, the primitives are wrong. *This is the only principled way to choose primitives, and it is the step that ad-hoc systems skip.* Under II.0 it is also the step that **cannot be shortcut by reasoning about what mathematics or programming essentially are** — the content axis is empirical by the theory's own account, and a basis defended on elegance rather than usage is a basis fitted to the author.

**R2 — Controlled ambiguity.** As §II.4. **Now partially confirmed.** Finzi et al.'s induction experiments mask $h$ bits of a latent variable and measure the structure extracted: epiplexity is maximized at *intermediate* $h$, and is low both at $h=0$ (task fully specified) and at $h=8$ (task fully hidden). That is the interior optimum this rule predicts, measured directly. What remains open is where the optimum sits for the substrate here, not whether one exists.

**R3 — Behavioral diversity.** Deduplicate by $[\![p]\!]$ on a probe input set. Uniform sampling over syntax collapses extensionally; without this you generate ten thousand programs that are all the identity.

**R4 — Minimality.** After sampling $p$, search within budget for the shortest $p'$ with $[\![p']\!] = [\![p]\!]$; train on $p'$. Otherwise description-length labels are noise and any curriculum sorts on a corrupted difficulty signal.

**R5 — Traces: install reachability, then withdraw them.** *This is the least settled rule in the document. Read the flag before acting on it.*

The safe half is unchanged from v1: small-step operational semantics gives $\langle p, x \rangle \to s_1 \to \cdots \to y$ for free and correct by construction, no reward model and no RL, and training on traces installs sequential state maintenance.

The unsafe half is the assumption that more trace is always better. Two results from Finzi et al. point the other way.

- **Chess ordering.** The *harder* factorization (board-then-moves, which requires inferring the intermediate moves) yields higher $S_T$ *and* better OOD transfer than the easy forward direction — matching accuracy on puzzles, significantly better on centipawn evaluation — despite worse in-distribution loss. Difficulty of inversion is what builds the transferable representation.
- **ECA emergence.** A looped model that unrolls intermediate states is compared against a non-looped model predicting the final state directly. Past a compute threshold the looped model wins on MDL and **epiplexity abruptly drops**: it has found the cheap brute-force simulation. The non-looped model, denied that path, keeps learning emergent structure — glider species, collision rules.

Reading: **a trace can substitute for structure rather than install it**, whenever it opens a brute-force path that is affordable within budget. The proposed schedule is traces early, when the untraced task is out of reach and the alternative is no learning at all, then thinned or withdrawn to deny the shortcut and force structural extraction.

> **⚠ Flag — treat this as the document's most likely error.** This inverts the most standard practice in the area, and the strong form rests substantially on one experiment that the authors themselves describe as an uncommon situation, where the brute-force solution happened to be accessible and where more compute revealed a *simpler* underlying structure. They explicitly expect the opposite for natural data at ordinary compute budgets. The chess result is independent and points the same way, which is the only reason the rule appears here at all; but chess demonstrates that *hard factorizations* build structure, which is not quite the same claim as *traces destroy it*.
>
> **Discriminator.** Run one task traced and untraced across a compute sweep and watch the traced arm for an $S_T$ turnover. If $S_T$ rises monotonically in both arms, this rule is wrong and R5 reverts to "traces always." That experiment is small, and it should be run before any generator design commits to a trace schedule.

**R6 — Complexity is four-dimensional.** Conflating the axes is the standard error in synthetic-data work.

| Axis | Quantity | Bias installed | Target domain |
|---|---|---|---|
| Description length | $\lvert p \rvert$ | Compositional generalization | Coding |
| Serial time | $T(p,x)$ | Depth of chained inference | Reasoning |
| State width | Live bindings | Working memory / attention capacity | Long context |
| Input entropy | $H(x)$ with $T$ small | Selection from large context | **Retrieval** — returned here from the inference axis in v4, where it never belonged |

Most work varies only $|p|$ and then finds reasoning transfer thin. These are four independent knobs and a curriculum should schedule them separately.

**R7 — Aim for the sweet spot; both failure modes are real and adjacent.** A generator can fail in two opposite directions, and the boundary is narrow.

| Generator | $H_T$ | $S_T$ | Outcome |
|---|---|---|---|
| Too simple | low | low | Nothing to learn; loss saturates immediately |
| **Hard but partly comprehensible** | medium | **high** | The target |
| Hard to invert (crypto-like) | maximal | ~0 | Looks like noise; loss never moves |

Finzi et al. give named instances at minimal separation. Elementary cellular automata rules 15, 30 and 54 are near-identical programs: rule 15 (class II, periodic, simple inverse) produces low $H_T$ and low $S_T$; rule 30 (class III, believed one-way) produces maximal $H_T$ and no $S_T$; rule 54 (class IV) produces medium $H_T$ and high $S_T$, with loss that continues to fall with compute. **The same compute, spent on a very similar program, yields drastically different structural output.** Independently, Zhang et al. (2024) found that downstream task performance benefits most from training on class IV rules.

Two consequences. First, generator choice is high-variance and cannot be settled by taste — it has to be measured, and $S_T$ is the instrument. Second, Finzi et al.'s own guidance for synthetic data follows from Theorem 12's asymmetry between a function and its inverse: **prefer functions without simple, efficiently computable inverses** — but not so far that the inverse becomes unlearnable, which is rule 30.

## II.6 The curriculum controller

The controller $\kappa$ was underspecified in v1. It now has both an objective and a validated instantiation.

**Objective.** Gate on *learning progress*, not absolute performance: mastered and impossible tasks both yield zero gradient, so the quantity to maximize is the rate of structural absorption. Under the prequential estimator this is exactly the epiplexity extraction rate — the area under the loss curve above the final loss, accumulated per token.

**Instantiation.** Adaptive Data Optimization (Jiang et al., 2025) dynamically upweights data subsets whose training loss is falling fastest. Finzi et al. show this approximately maximizes the prequential epiplexity estimate, and that at 1.3B parameters over 125B tokens it achieves both higher measured epiplexity and better downstream performance than uniform sampling, despite not being optimized for either.

**The consequence that matters, and it is counterintuitive.** Subsets with more learnable structure carry *higher* absolute loss, so maximizing epiplexity means **accepting a worse training loss curve**. A controller doing the right thing will look like it is doing the wrong thing on the dashboard. This is §IV.5's slope-not-intercept principle in operational form, and it is the single easiest way for this program to be abandoned for the wrong reason.

**Structure, not sequence.** Catastrophic forgetting requires that $\kappa$ emit a *distribution* over all types with mass retained on mastered ones — never a pointer into an ordered list. This is what reconciles the framework with the weak empirical record of curriculum learning: hard curricula hurt, shifting mixtures help.

**Competence gating is validated, at least on difficulty.** Jiang et al. gate their procedural pretraining on accuracy: input sequences start at length 2 or 4 and increase by 2 once 99% accuracy is reached, up to length 20. That is the competence-gated scheme this program originally proposed, working — though over difficulty within a type rather than across types.

**Three controllers, not one.** Given the II.0 split, $\kappa$ decomposes:

| Axis | Ordering principle | Basis |
|---|---|---|
| Within a content type | Competence gating on difficulty | Validated (Jiang et al.) |
| Across content types | Topological order on the conditional-epiplexity DAG | Measured (§I.6, §IV.3) |
| Across inference forms | L0 first, then raise identifiability | Conjectured — L0 is the only rung with dense verifiable signal |

Only the third is guesswork, and it is guesswork with a reason: L0 supplies a deterministic target that can be checked, which is plausibly why the whole approach works. Raising identifiability spends that. See V.1.

## II.7 Two warnings about the substrate

**Serialization is part of $\pi$.** Tokenization and surface syntax are where unintended bias enters through the back door. Abstracting away architecture does not abstract away the serializer.

**Determinism is itself a bias, and possibly a bad one.** Pure symbolic programs install a bias toward *exact execution*, which may actively harm a model that must later handle ambiguous, underdetermined natural-language input. Let $[\![p]\!]$ return distributions, not values, and put real mass on stochastic programs. A model that has only seen deterministic ground truth will be badly calibrated on everything humans actually ask.

## II.8 Translation, not transfer

Transfer is a hope about representations. **Translation is a structural claim**: text and the substrate are two *encodings* of a shared latent space $Z$; the synthetic phase installs $\pi$ over $Z$ and the operations on it; the text phase learns only the surface map $\sigma : \text{NL} \to Z$. By Proposition 7 that costs $K(\sigma)$ — and learning a surface map is a dramatically lower-complexity problem than learning a reasoning system.

The framing carries an obligation that transfer does not:

> **To obtain translation rather than transfer, train on aligned pairs** — the same latent $p$ rendered in the formal substrate *and* in natural language.

Since $p$ is constructed, render it as a program, as a proof, and as an English word problem, and train on all renderings of the same underlying object. Without alignment one is back to hoping, having surrendered the theory's main advantage.

This makes the two hypotheses empirically distinguishable, which is worth a great deal:

| | Transfer predicts | Translation predicts |
|---|---|---|
| Shape of gain | Smooth decay with domain distance | **Phase change** when aligned pairs are introduced |
| Text-stage tokens | Somewhat fewer than baseline | **Dramatically** fewer |

*Guard:* aligned pairs invite learning a shallow template-to-template map that bypasses semantics. Defenses are stochastic rendering (many surface forms per $p$, so no surface form is diagnostic) and held-out compositions in the NL rendering specifically.

## II.9 What to reuse from classical AI, and what to discard

Twentieth-century formal systems for reasoning are a real asset here, provided the right half is taken. The discriminator is clean:

**Reuse systems that specify *how to search or derive*, because those produce traces (R5).**

| System | What it supplies |
|---|---|
| Resolution, natural deduction | Unlimited (problem, proof) pairs at controllable depth; proof length is a clean $T$-axis; the proof *is* the teacher trace |
| Unification, term rewriting | Variable binding with native small-step semantics — plausibly the most conspicuous hole in transformer inductive bias |
| STRIPS, situation calculus | Long-horizon state with optimal plans as ground truth; the frame problem is the formalization of "what stays the same across a state update," the core of L2 |
| SLD resolution | Backtracking traces — try, fail, undo, retry. This is debugging |
| Hindley–Milner inference | A coding bias with a soundness theorem attached |
| Production systems (OPS5, SOAR, ACT-R) | A forty-year formalization of harness-like patterns; ACT-R's declarative/procedural split is this document's knowledge/bias factorization under another name |

**Discard systems that specify *what is true*** — frames, semantic networks, expert systems. CYC is the canonical trap: it is a knowledge project, and knowledge is the half that is information-theoretically expensive and not hand-buildable. **Classical AI failed at knowledge acquisition; it did not fail at search.** The search algorithms all still work.

The anti-ad-hoc property comes along with them: these calculi have soundness and completeness theorems, so $\pi$ inherits semantics, and R1's support condition becomes *checkable* — "is this task expressible as a derivation in this calculus?" has an answer rather than a vibe.

---

# Part III — Claims, stated so they can die

A research program is only as good as the experiments that could end it. Each claim below is paired with its falsifier.

**Claim 1.** *Pretraining's function is to install inductive biases enabling finetuning and RL — biases that emerge incidentally from patterns of use across a corpus.*
→ **Weakened from the original.** Architecture, tokenizer, optimizer, and parameterization all inject bias for free and none of it comes from training. Softmax attention has a retrieval bias; positional encoding a locality bias; BPE a morphological prior. The defensible version: pretraining installs the **residual** bias that architecture does not supply. Attributing all bias to data is the most common error in this area.

**Claim 2.** *A structured corpus at matched token density outperforms natural text.*
→ **Largely settled in our favour, and not by us.** Jiang et al. demonstrate token substitution at 1.3B across three domains, with the shuffling control confirming that structure rather than token statistics carries the effect. What remains open for this program is not whether structured data beats text per token, but whether it does so **above L0**.
→ **Falsifier:** matched-token web vs. synthetic at 100M–1B parameters, on two axes — $S_T$ per token for structural density, Surplus Description Length on held-out targets for aim. **Slope, not intercept** (§IV.8). Operationalized as step 5 of §IV.9, with rule 54 as the sanity baseline. If synthetic beats neither, this is dead, and it is dead for a few thousand dollars.

**Claim 3.** *Current synthetic approaches install the simplest bias in the family: predicting dynamics.*
→ **Too coarse; revised.** Different procedural sources install *different specific skills*, not one generic one: $k$-Dyck improves context recall, ECA rule 110 improves reversed addition, Union and Delete improve multiplication. The primitive→capability map is partially measured already. Two consequences: the space is finer-grained than "dynamics prediction," and the map was found by exhaustive search rather than predicted — which is exactly Gap 1. The live question is whether the *harder* biases (binding, backtracking, task inference, recovery) can be installed the same way, and that is a question about inference form, not content.

**Claim 4′.** *Transfer only matters if base skills are learnable **and encoding-invariant.***
→ The unqualified version is toothless: a skill can be perfectly learnable in a form welded to the substrate's surface syntax, so in-substrate evaluation goes green and transfer is zero.
→ **Test, and it is cheap:** train a skill under one rendering, evaluate under a *different rendering within the synthetic world*. Surviving re-encoding inside the synthetic world predicts surviving translation to text.
→ **This claim is what makes the program tractable**, because it licenses a decoupled research loop: iterate on $\pi$ entirely inside the substrate using cheap in-substrate and cross-rendering evaluations, and run the expensive text stage only occasionally to check calibration. Encoding-invariance is the in-loop proxy for transfer. Almost all of the research becomes cheap.

**Claim 5.** *The biases needed for reasoning, coding, and mathematics are small enough to specify in a fixed, human-written generator of roughly 500 lines.*
→ Two refinements make this survivable and testable.

  **(a) The bound is on the primitive basis, not the situation family.** A short generator with unbounded compositional depth already produces unboundedly many situation types. What is finite is the primitive set. Saturation therefore occurs only if downstream tasks require primitives outside the basis, and the real question is:

  > **Basis completeness.** Is the primitive set $B$ sufficient that every $f$ in the sharp $\mu$ has a short program over $B$?

  This is far more tractable than "does it saturate," because it is diagnosable per-failure — take a downstream failure and ask which missing primitive would have made the program short. A repair loop, not a verdict.

  **(b) Measurable form.** There exists $G$ with $|G| \le 500$ lines whose bias density $S_T(X)/|X|$ on the sharp $\mu$ exceeds web text's by an order of magnitude at matched token count, with the corresponding reduction in Surplus Description Length on held-out downstream tasks. Both quantities now have published estimators. The NCA result establishes ≈10× for the most trivial bias in the family.

  **(c) The generator-size worry is weaker than it looks.** Finzi et al.'s induction experiments show that in both the easy and hard settings, *the program needed to predict is larger than the program needed to generate*: the induction strategy was never present in the data-generating process, yet is learned from data produced by it. A short generator does not cap the size of what the learner acquires. This is Paradox 3, and it substantially defuses the objection that a 500-line file can only install 500 lines' worth of bias.

### On open-endedness, and a reversal

An earlier version of this argument held that a fixed generator must saturate, and that co-evolution — a proposer model rewarded for producing situations at the learner's frontier — was therefore mandatory rather than optional.

**That argument is withdrawn.** A co-evolved generator's objective has to come from somewhere; it ends up encoding human judgment regardless, and now that judgment is buried inside a learned system where it can be neither inspected nor falsified. Trading a testable surface for an untestable one at the *start* of a research program is bad strategy. The saturation argument was also implicitly assuming $\mu$ = everything humans do; under sharpness it is much weaker.

**Build the fixed generator. Treat saturation as a measurement.** The types-versus-$S_T$ curve adjudicates it, and either outcome is a win: it plateaus and the co-evolution program has been *earned from data*, or it does not and a large research detour has been avoided.

The reversal has since been strengthened from an argument into evidence: see Claim 5(c) above, and V.2.

### On the bitter lesson

The standard objection deserves a better answer than the cultural one. Sutton's claim is not "never build in structure"; it is "build in methods that scale with compute." **A synthetic generator is a machine that converts compute into data**, which is maximally compliant. The failure mode is precise: a generator whose effective description length is bounded will saturate in *bias content* no matter how many tokens it emits — data volume scales, bias does not. That is exactly the quantity the saturation curve measures, which is why the measurement matters more than the argument.

---

# Part IV — The immediate program

**Corrected from v1.** v1 claimed the first step required no GPUs, on the grounds that hand-measuring program lengths estimates the whole objective. Under epiplexity that is false, and the counterexample is in the paper this version depends on: ECA rules 15, 30 and 54 have near-identical program lengths and drastically different extractable structure. **Description length under a basis is a poor predictor of what a bounded learner will extract.** So the program now has two tracks, and the compute-bearing one is the more urgent of the two.

## IV.1 Why not go substrate-hunting now

Three obstructions, one of them new.

**The syntax axis is the low-order term.** Lisp, Forth, a combinator calculus and a sequent calculus are mutually intertranslatable by compilers of a few hundred lines, so the choice among them is a $K(\sigma)$ question — bounded, small, and by Proposition 7 the least significant quantity in play. What determines the bias is the primitive basis, which is orthogonal to syntax.

**The ordering does not yet exist.** The intrinsic order collapses at the top by invariance, and the extrinsic order $\preceq_\mu$ exists only once $\mu$ is named. With no estimate of $\mu$, any substrate argument is aesthetics wearing a proof's clothing.

**There is no instrument yet.** *(New.)* Substrates are evaluated by the structure they induce, which means $S_T$, which means training runs and an estimation pipeline that does not currently exist here. Choosing a substrate before the instrument is built means choosing it by the proxy that rules 15/30/54 prove unreliable.

## IV.2 Two tracks

| | Track A — build the instrument | Track B — extract the basis |
|---|---|---|
| **Question** | Can we measure structure at all? | What primitives does $\mu$ actually use? |
| **Cost** | Small: consumer GPUs, days | None: people and a spreadsheet |
| **Blocks** | Every generator decision downstream | Substrate commitment |
| **Risk if skipped** | Design by unreliable proxy | Design by taste |

They are independent and should run in parallel. Track A is on the critical path; Track B is not, but it is free.

## IV.3 Track A — stand up the instrument (the actual next step)

Four experiments, in order. The first builds the instrument; the second can end the program; the third and fourth resolve R5 and Gap 1.

**First: reproduce the ECA epiplexity measurement.** This is the concrete first action, and it is chosen because it is the only cheap task that is simultaneously an instrument, a validation, and a known answer.

*Setup*, from Finzi et al. §5.1 and Appendix C.1, which give enough detail to replicate directly. Predict $Y$ from $X$, where $X$ is a 64-cell initial state (sampled by evolving a uniform state 1000 burn-in steps to clear transients) and $Y$ is $X$ evolved 48 steps under ECA rule $F$. Rules 15, 30 and 54. GPT-2 architecture with Adam; widths $\{16,32,64,128,256,512\}$ and depths $\{1,2,4,6,9\}$; batch 1536 sequences; base learning rate 0.03 with 100 warm-up steps, transferred across sizes by $\mu$P and CompleteP; EMA timescale 50 steps; test set $D = 100$M tokens counting $Y$ only. Estimate by **prequential coding** — area under the training loss curve above the final loss — not requential, which is 2–10× slower and unnecessary for a pipeline check. Take the lower convex hull of the $|P| + \mathbb{E}\log 1/P(X)$ versus compute curves and retain the median point per training run to get the Pareto frontier. Reference implementation: `github.com/shikaiqiu/epiplexity`.

*Success criterion, stated in advance:* rule 15 low on both $H_T$ and $S_T$; rule 30 near-maximal $H_T$ with $S_T \approx 0$; rule 54 medium $H_T$ with high $S_T$ and loss still falling at the compute ceiling. Reproducing that ordering means the instrument works. Failing to means it is misconfigured, and every number produced afterwards would have been noise.

*Scale:* their smaller experiments ran on a handful of consumer GPUs. This is days, not months.

**Second: run the R5 discriminator, which Finzi et al. have already specified for you.** Appendix C.8: ECA rule 54 over $t = 64$ steps, comparing a non-looped model predicting the final state directly against $\ell$-loop models predicting intermediate states $(X^{(\Delta)}, \ldots, X^{(t)})$ with $\Delta = t/\ell$. Widths $\{16,32,64,128\}$, depths $\{1,2,4,8,16,32\}$, loops $\ell \in \{1,2,4,8,16\}$; base learning rate 0.06, batch 147,456 tokens, EMA 50. Watch for the compute threshold beyond which the looped model wins on MDL and $S_T$ **drops**. If that turnover reproduces, the R5 inversion is real for at least this case and the trace schedule is worth designing around. If $S_T$ rises monotonically in both arms, R5 reverts to "traces always" and §II.5 needs rewriting.

**Third: the ambiguity sweep, which decides whether there is a contribution at all.** *(v3 specified a three-point L0/L1/L2 comparison. That was a mistake — see below.)*

Take one content type already validated at L0. Stack is ideal: it involves state tracking, and Jiang et al. give the generator. Construct a family of variants that **vary $H(p \mid \text{query})$ continuously**, by withholding an increasing fraction of the task specification and supplying demonstrations instead — mirroring Finzi et al.'s masked-latent design, which sweeps the number of hidden bits $h$ rather than comparing two conditions. Measure $S_T$ and downstream transfer as a curve in $H$.

**Why continuous rather than three-point.** v3's version tested *my* discretization. Under II.0 that discretization is a guess, so a collapse at my particular L1 construction would have been ambiguous between "nothing above $H=0$ works" and "that construction was badly chosen" — and v3's gate would have written off the program on the weaker reading. The sweep is strictly better in three ways: it does not depend on the rungs being right; it directly tests R2's predicted interior optimum rather than assuming one; and it is the design that has already been shown to resolve, since Finzi et al. find epiplexity peaking at intermediate $h$ and low at both ends.

**Gate, restated.**

| Curve shape | Reading | Action |
|---|---|---|
| Monotone collapse in $H$ | Real ceiling. The approach depends on determinate targets | Stop; redesign around $H = 0$ |
| Interior peak at low-but-nonzero $H$ | The program works; the rungs are misplaced | Continue; relocate the cuts from the curve |
| Flat or rising | R2 is wrong and something more interesting is happening | Investigate before proceeding |

This experiment now *locates* the gradations rather than testing a guess about them, which retro-justifies running it second: it produces the coordinate system's cuts on the inference axis, and everything downstream is indexed by those cuts.

**Fourth: the curriculum DAG.** Train a small model on each of $n$ content types and measure conditional epiplexity $S_T(t_j \mid m_i)$ for all pairs (§I.6). At diagnostic scale — two layers, four heads, hidden size 16 — ten types is a hundred runs of minutes each. The output is a weighted prerequisite graph, and its topological order is the stage curriculum. This is the answer to Gap 1, it is cheap, and Finzi et al. flagged conditional epiplexity as unexplored.

## IV.4 Track B — the corpus exercise, with corrected claims

Hand-write minimal programs solving real target tasks: 200–300 spanning the sharp $\mu$ — competition mathematics, real debugging traces, algorithm implementation, multi-hop retrieval, agentic tool sequences.

The circularity (a substrate is needed to write in) dissolves the same way as in v1: **use a deliberately over-general scratch substrate as a measuring instrument, not as a candidate** — untyped lambda with a permissive pile of primitives, or restricted Python. The point is to record which operations are reached for. The basis is extracted from usage, which is R1 executed rather than asserted.

**What changed in v2 is the claim, not the exercise.** v1 asserted this estimates $\mathbb{E}_{f\sim\mu}[\beta(f)]$, the entire regret objective. It does not. It estimates the *support and basis* of $\mu$ — necessary, irreplaceable by compute, genuinely free — but the objective's value requires $S_T$ and therefore Track A.

**What changes in v3 is its standing.** Under II.0 this is not preliminary spadework but the only admissible route to the content axis, since that axis is a fact about practice rather than about mathematics. It follows that the exercise may need repeating as practice drifts — if the tooling of debugging changes, the primitives of debugging change. Whether it is one-time or standing is precisely the Platonist/practice-relative disagreement, and it is decidable: **re-run the extraction on a corpus of tasks from five years ago and see whether the basis moves.**

## IV.5 Track B′ — extracting the inference axis from real traces

*(New in v4.)* Track B extracts content primitives by hand-writing programs for real tasks. II.0 now requires the same treatment for the inference axis, since its gradations are equally empirical. Same methodology, different dimension.

**Material.** Real traces of the practices in $\mu$: actual debugging sessions, agent transcripts, how retrieval is genuinely used rather than how it is benchmarked, worked solutions with their false starts intact where those survive.

**Annotation.** For each episode, segment by how the task specification was resolved:

- stated up front in the instruction,
- inferred from context or prior examples without acting,
- discovered only by taking an action and observing the result,
- never resolved — the practitioner proceeded under residual uncertainty, and how they hedged.

**Output.** An empirical distribution over the inference axis: where the mass actually sits in the practices we care about. That is what tells you where to spend generator effort, and it is the only non-question-begging way to place the cuts. If real debugging turns out to be 80% act-to-discover, an L0-heavy generator is misaimed no matter how well it scores.

**Guard, as in Track B.** Two or three independent annotators on an overlapping subset, with inter-annotator agreement reported. Divergence means the categories are the annotator's rather than the practice's, which is exactly the failure II.0 warns about.

**Relation to the sweep.** Track B′ says where practice sits on the axis; §IV.3's sweep says where the model can learn. The generator should target the intersection. Neither alone determines the design.

## IV.6 What each track can and cannot tell you

| Measurement | Track | How | Decides |
|---|---|---|---|
| **Primitive saturation** | B | Distinct primitives used vs. tasks processed | Claim 5(a). Still climbing at task 300 → basis unbounded, dead. Flattens at 40–80 → basis in hand |
| **Basis completeness** | B | Every long, awkward program names a missing primitive | Per-instance repair loop rather than a verdict |
| **Algorithmic content of $\mu$** | B | Compress the program corpus | Upper-bounds $\mu$'s content *at unbounded compute*. Related to but not equal to $H(\mu)$ |
| **Instrument validity** | A | Rules 15/30/54 ordering | Whether any downstream number means anything |
| **Trace schedule** | A | Looped vs non-looped $S_T$ turnover | R5, the document's flagged error |
| **Bias density** | A | $S_T/\lvert X \rvert$ on generated corpora | Claim 5(b), the central empirical claim |
| **Aim** | A+B | Surplus Description Length on held-out targets | Whether the structure is the *right* structure |

**Basis ranking by summed program length — dropped.** v1 offered this as "$\preceq_\mu$, computed, ten candidates in a day." Rules 15/30/54 show program length and extractable structure can diverge maximally, so this ranking is unreliable for the question it was meant to answer. Use it to check basis *completeness*, never to predict extractability.

## IV.7 Methodological guard

The person writing the programs is themselves a bias, and one risks measuring the author rather than $\mu$. **Fix:** two or three independent authors on an overlapping subset; measure inter-author agreement on primitive usage. Convergence is evidence the basis is a property of the task distribution; divergence is a warning that taste is being extracted. This costs almost nothing and is the difference between a measurement and a projection of one's own priors.

## IV.8 Evaluation methodology, for everything downstream

Current pretraining evaluation measures $\mathrm{risk}(\theta_0, f)$ — zero-shot loss, the **intercept** of the adaptation curve. Two checkpoints with identical validation loss can have wildly different acquisition costs. The theory says to measure the **slope**. Finzi et al. put it directly: loss captures only the residual unpredictability, corresponding to $H_T$, and says nothing about how much reusable structure was internalized to reach that loss.

**Two instruments, not one, and they answer different questions.**

| Instrument | Measures | Question answered | Cost |
|---|---|---|---|
| **Epiplexity $S_T$** | Structural content extracted, task-agnostic | Is there anything here to learn? | Prequential: one training run. Requential: 2–10× more |
| **Surplus Description Length** | Acquisition cost of a specific $f$ | Are we aimed at the right thing? | One adaptation curve per target task |

The task-relative quantity v1 wrote as $C_{\theta_0,\mathcal{A}}(f)$ already has a name: **Surplus Description Length** (Whitney et al., 2020) — the summed online loss of a training algorithm on a downstream task, used to evaluate representations by learning efficiency rather than final accuracy. Use the existing name. Finzi et al. explicitly contrast the two: SDL is task-relative, epiplexity is task-agnostic. **Neither substitutes for the other**, and §I.7's regret theorem is why: high $S_T$ on the wrong $\mu$ is structure you cannot use.

**Practical recommendation.** Prequential coding — area under the loss curve above the final loss — for the in-loop iteration and for ranking generators, since it needs only a training run and correlates well with the rigorous estimate. Requential coding, which encodes the training run explicitly as cumulative teacher–student KL, when a number rather than a ranking is needed. **Slope, not intercept.** If nothing else here is adopted, adopt this.

## IV.9 Sequencing, with decision points

| # | Step | Cost | Gate — proceed only if |
|---|---|---|---|
| 1 | Reproduce ECA rules 15/30/54 | GPU-days | The three-way ordering reproduces |
| 2 | **Ambiguity sweep on Stack** | a handful of small runs | The curve is not a monotone collapse in $H$. **If it is, stop and redesign** |
| 3 | R5 discriminator (looped vs non-looped) | GPU-days | — resolves R5 either way |
| 4 | Curriculum DAG over content types | ~100 tiny runs | Prerequisite structure is non-trivial, i.e. some off-diagonal drops are large |
| 1′ | Corpus exercise, Track B (parallel, no compute) | person-weeks | Primitive count flattens by ~task 300 |
| 1″ | Trace annotation, Track B′ (parallel, no compute) | person-weeks | Inter-annotator agreement holds |
| 5 | **Minimum viable generator** | GPU-days | See the bar below |
| 6 | Commit to a substrate | — | Now a measurement, not an argument |
| 7 | In-substrate loop on $\pi$, cross-rendering eval as transfer proxy | ongoing | Claim 4′ licenses the cheap loop |
| 8 | Aligned-pair text stage | larger | Test the phase-change prediction of §II.8 |

**Step 2 is placed second deliberately.** It is a handful of small runs, it is the only experiment on the list that can end the program rather than adjust it, and it also produces the cuts on the inference axis that everything after it is indexed by. Everything from step 3 onward assumes the answer is favourable; discovering otherwise at step 5 would waste the substrate work, and discovering it at step 8 would waste everything.

**Step 3 is the real go/no-go, and it has a brutal, clean bar.** Take the basis from Track B, write the smallest generator that produces episodes over it — tens of lines, not 500 — emit ~100M tokens, and measure $S_T$ at matched compute against two baselines: **OpenWebText** at matched tokens, and **ECA rule 54**.

> If a purpose-built generator over a basis extracted from real mathematics, code and retrieval cannot beat a one-dimensional cellular automaton on structural bits per token, the thesis that reasoning biases are compactly specifiable is in serious trouble.

Rule 54 is the right baseline precisely because it is stupid. It has no notion of composition, binding, or retrieval, and it was not designed for anything. Beating OpenWebText is the headline claim; beating rule 54 is the sanity condition, and it is the cheaper of the two to check.


# Part V — Open problems

Stated plainly rather than hidden.

**V.1 Whether anything above $H=0$ is learnable in this regime — the crux.** *(Outranks everything below.)* Every published result in procedural pretraining sits at $H(p \mid \text{query}) = 0$: deterministic target, verifiable output, dense per-token signal, no credit assignment. It is entirely possible that the approach works **because** of those properties. Raising $H$ replaces a determinate target with a posterior over tasks; raising it far enough introduces action selection with deferred payoff. If the signal degrades into something requiring RL, the SFT-throughput argument motivating the entire design collapses, and with it the case for choosing this over reward-based pretraining.

Nothing in the theory rules this out. Proposition 8 says the log-loss optimum is the posterior predictive; it says nothing about whether gradient descent reaches it at feasible scale when the posterior is broad. Finzi et al.'s masked-latent result cuts both ways — models do induct on hidden bits and epiplexity peaks at intermediate masking, which is encouraging, but the hard setting costs compute exponential in the number of hidden bits, which is precisely the degradation this risk predicts. Their setup was diagnostic, not an attempt to install a bias.

**Probe: the ambiguity sweep, §IV.3, step 2.** *(Restated in v4.)* v3 specified a three-point comparison and would have terminated the program on a collapse at one hand-chosen construction. Since the constructions are empirical guesses under II.0, the gate is now **monotone collapse across a continuous sweep in $H$**, which distinguishes a real ceiling from badly placed cuts. This remains the only listed risk that can terminate rather than redirect, and it is close to the cheapest thing on the list. Retired and answered items follow.

**V.2 The amortizability gap — retired to a measurement protocol.** v1 listed this as the framework's central unmodelled obstruction: the reachable set was defined by what one can *construct*, but the system trains an amortizer, and short code under a basis does not imply gradient descent can install it. Redefining the cost as epiplexity dissolves the theoretical gap by construction — a structure a compute-limited learner cannot extract simply does not count toward $S_T$ — at the price of turning it into an estimation problem requiring training runs. That is a good trade: estimation problems have published estimators and code. What remains genuinely open is *predictive* rather than descriptive — no theory says which $\pi$ are amortizable by which architectures in advance, and that theory would be valuable well beyond this program.

**V.3 Saturation — largely answered, and favourably.** Finzi et al.'s induction results show the program required to predict exceeding the program required to generate, in both easy and hard settings, with strategies learned that were never present in the generating process. A short generator does not cap the learner. The residual question is not whether a fixed generator saturates in principle but where the practical plateau sits for a particular basis, which the types-versus-$S_T$ curve measures directly.

**V.4 The brute-force ceiling — replaces v1's asymptotic-decay worry.** v1 raised the concern that structural extraction loses value as compute grows. That was wrong: $S_T$ typically grows with compute and with data, and the asymptote is capped by dataset size rather than compute. The correct worry is sharper and specific to synthetic data. Epiplexity collapses when a brute-force solution becomes affordable — this is exactly what happens in the ECA emergence experiment. **A synthetic generator is short by construction, so it is precisely the case where brute force is cheapest.** Natural data's generator is physics, which Finzi et al. note remains inaccessible to any physically realizable observer, so natural corpora keep their structural content for all practical purposes. Synthetic corpora have no such protection: once the learner can afford to simulate the generator, extraction stops. The design response is to make the brute-force path expensive relative to budget — composition depth, hard-to-invert steps per R7, minimality per R4 — and to treat an $S_T$ turnover as the go/no-go signal.

**V.5 The transfer surface.** Symbolic bias may fail to cross to natural language because the serialization surface is too alien. Prediction: transfer requires *interleaving* the synthetic and text phases rather than sequencing them. The NCA result being a sequential "pre-pre-training" stage shows sequencing works at all; how much better interleaving does is a one-variable ablation.

**V.6 Stochastic semantics.** §II.7 argues for distribution-valued $[\![p]\!]$. Epiplexity is defined over probabilistic models throughout, so the framework accommodates this natively — but the complexity axes of R6 were stated for deterministic programs and need extending.

**V.7 The ambiguity optimum — partly answered.** R2 predicted non-monotonicity in $H(p \mid \text{query})$ with an interior optimum; Finzi et al.'s masked-latent experiments confirm exactly this shape. Open: where the optimum sits for this substrate, and whether it coincides with the optimum for downstream transfer rather than for $S_T$ alone.

**V.8 The R5 inversion.** Flagged in place. The most likely *error* in this document, as distinct from V.1, which is the most likely *failure*.

**V.9 The gradations may not be stable.** *(New in v4.)* II.0 asserts that cuts on both axes are practice-relative and will drift. If they drift *fast* — faster than a generator can be built and a model trained — then the whole extraction methodology is chasing a moving target and the Platonist would be pragmatically right even if philosophically wrong. Test: re-run Track B and B′ extraction on tasks and traces from five years ago and measure how far the basis and the inference-mass distribution move. Cheap, and it decides whether Tracks B and B′ are one-time or standing.

**V.10 Weight-space assembly is unmodelled.** §I.6 records the operation and its empirical behaviour but has no cost law for it. Since assembly is architecture-indexed rather than function-indexed, it may not admit one in the same currency as the rest of the algebra — which would be a real limit on how far the bias-as-code picture extends.

---

# Related work

**Read first.** Finzi, Qiu, Jiang, Izmailov, Kolter & Wilson (2026), *From Entropy to Epiplexity* (arXiv:2601.03220) — the formal spine of this version; §I.4 onward depends on it. Achille & Soatto (2025), *AI agents as universal task solvers* — argues that the role of past data is to reduce the *time* to solve new tasks rather than to reduce uncertainty, with the speedup characterized by shared algorithmic information between past data and future tasks. That is this document's thesis, published, and it should be read before anything here is written up.

**The anchor.** Jiang, Shinnick, van den Hengel, Saratchandran & Teney (2026), *Procedural Pretraining: Warming Up Language Models with Abstract Data* (arXiv:2601.21725). The strongest L0 result, sharing this document's framing explicitly. Read alongside its companion, Shinnick et al. (2026), *Can You Learn to See Without Images?*, which transfers the same procedural data to vision transformers — the strongest available evidence for Claim 4′, and the result their Platonist reading explains and II.0 disputes.

**The only principled predecessor.** *LIME* (Wu et al., 2021) — task families derived from Peirce's deduction/induction/abduction rather than assembled by provenance. Whatever one thinks of the trichotomy, this is the sole prior attempt at a *selection principle*, and Gap 1 exists because nothing since has offered one. Read it for the method, not the results.

**Other L0 prior art.** Hu et al. (2025) on formal languages imparting linguistic biases and outperforming natural language per token; Wu et al. (2022), simpler synthetic tasks; Lindemann et al. (2024), SIP, injecting structural bias by simulation; Bloem (2025), universal pre-training by iterated random computation; Papadimitriou & Jurafsky, non-linguistic transfer to language; Lee, Han, Kumar & Agrawal (2026), the NCA result; Zhang et al. (2024), *Intelligence at the Edge of Chaos* — class IV rules transfer best, corroborating R7. Nakamura et al. (2024) is the extreme case: a single fractal image with augmentations approaching ImageNet-pretrained performance.

**Amortized induction.** Müller et al. (2021), *Transformers can do Bayesian inference* — Proposition 8 implemented. Grau-Moya et al. (2024), *Learning universal predictors* — training transformers toward Solomonoff induction, Proposition 8's explicit ambition. Both are prior art for the mechanism this document assumes.

**To evaluate, not to adopt.** *Reasoning Core* (arXiv:2603.02208), a procedural data generation suite for symbolic pre- and post-training. The right question is not whether it supersedes this framework but **which cells of the II.0 coordinate system its tasks occupy** — the prediction from its description is broad on content type and uniformly L0 on inference form. That evaluation is only meaningful once the coordinate system is fixed, which is why it appears here and not in Part IV.

**Mechanism.** Xie et al., in-context learning as implicit Bayesian inference — Proposition 8 for natural corpora. Chan et al. on data distributional properties driving ICL emergence — R2 discovered empirically.

**Instruments.** Whitney et al. (2020), *Surplus Description Length* — the task-relative instrument of §IV.5. Blier & Ollivier, *The Description Length of Deep Learning Models*, and Zhang et al. (2020) on information transfer — the prequential lineage. Goldblum, Finzi, Rowan & Wilson on No Free Lunch, Kolmogorov complexity and inductive bias — directly Part I, by the same group.

**Controllers.** Jiang et al. (2025), *Adaptive Data Optimization* — §II.6's instantiation. Graves et al., *Automated Curriculum Learning for Neural Networks* — the learning-progress objective. Schmidhuber's POWERPLAY and the speed prior — open-endedness held in reserve pending V.2, and a computable prior weighting both program length and runtime, an ancestor of the budget-indexed code.

**The rival to II.0.** Huh et al. (2024), *The Platonic Representation Hypothesis* — the explanation the field reaches for when procedural bias crosses modalities, and the position the anti-Platonic constraint denies. Teney et al. (2024), *Neural Redshift*, and (2025) on whether the simplicity bias is always wanted, for the architecture-side biases that Claim 1's correction concedes.

**Foundations.** Mitchell (1980) on bias as deductive closure; Solomonoff on universal induction and the invariance theorem; Levin's $Kt$; Koppel's sophistication and Bennett's logical depth as the unbounded-compute ancestors of epiplexity, together with the Chaitin-incompleteness obstruction that motivates replacing them; Amari for the information geometry of §I.8; McAllester and the PAC-Bayes line for the independent derivation of the KL functional; Håstad et al. and the pseudoentropy literature for the cryptographic grounding.

---

# Changes from earlier versions

### Changes from v1

Recorded rather than silently absorbed, since v1 is retained and two of its claims are known errors.

| Change | Where | Why |
|---|---|---|
| Information ceiling split into an asymmetric pair | §0 | v1 claimed a theorem on both sides. Only the entropy side is proved; the structural side is conjecture plus evidence |
| Central premise upgraded from bet to measurement | §0 | Structural share of natural corpora measured directly |
| Two-coordinate $(\beta,\tau)$ frontier replaced by budget-indexed $S_T$ | §I.4 | Levin's $Kt$ combines the resources by an arbitrary rule; the compute-limited two-part code does not |
| Pays-for-itself criterion and the MDL-minimizer gap added | §I.5 | Makes the factorization mechanical, and supplies the real justification for retrieval |
| **R5 inverted** | §II.5 | Traces can substitute for structure. Flagged as the most likely error here |
| R7 (sweet spot) added with named instances | §II.5 | Generator choice is high-variance and must be measured |
| Curriculum controller specified | §II.6 | Was underdetermined; now has an objective and a validated instantiation |
| Asymptotic-decay risk replaced by the brute-force ceiling | V.3 | v1's worry was wrong; the correct one is sharper and synthetic-specific |
| Amortizability gap retired; saturation largely answered; ambiguity optimum partly answered | V.1, V.2, V.6 | Three of v1's five open problems moved |
| $C_{\theta_0,\mathcal{A}}$ renamed to Surplus Description Length | §IV.8 | The quantity already had a name |
| **Part IV rewritten around two tracks** | Part IV | v1 claimed hand-measured program length estimates the objective. Rules 15/30/54 refute this: near-identical lengths, drastically different $S_T$ |
| Basis ranking by summed program length dropped | §IV.5 | Same reason — unreliable for predicting extractability |
| Proposition 8 reread as a lower bound | §II.3 | v1 implied the learner cannot exceed its generator; Paradox 3 shows it can |
| Concrete first action and go/no-go bar specified | §IV.3, §IV.7 | Beat rule 54 on structural bits per token, or the compactness thesis is in trouble |

### Changes from v2

| Change | Where | Why |
|---|---|---|
| Contribution restated; L0 explicitly ceded | front matter | Jiang et al. establish L0 at 1.3B. Claiming it would be false |
| Gap 1 (no selection principle) named, with LIME as sole predecessor | front matter | The largest unclaimed gap in the area |
| Anti-Platonic constraint added as a governing rule | §II.0 | $\mu$ ranges over practices; the content axis is empirical by the theory's own account |
| "Type" split into content type vs inference form | §II.0, throughout | Two orthogonal axes with different curriculum questions were being conflated |
| Identifiability ladder promoted to the central proposal | §II.4 | It is the part nobody has tried |
| Weight-space assembly added as a fifth operation | §I.6 | Selective transfer beats full transfer; the mixture law does not model this |
| Chain rule made measurable via conditional epiplexity | §I.6, §IV.3 | Turns stage selection from assertion into a measured DAG. Answers Gap 1 |
| Controller split into three | §II.6 | One per ordering axis; only the third is guesswork |
| Claim 2 downgraded from open to largely settled; Claim 3 refined to per-skill | Part III | Jiang et al. settle the first and refute the coarseness of the second |
| **L1-learnability promoted to V.1** | Part V | The only risk that can end the program rather than redirect it |
| L1 probe inserted as step 2 of the sequence | §IV.3, §IV.8 | Three runs, and it gates everything after it |

### Changes from v3

| Change | Where | Why |
|---|---|---|
| **Axis/gradation distinction introduced; anti-Platonic constraint extended to both axes** | §II.0 | v3 exempted the inference levels as "derived." They were abducted from current agent capabilities — the tell being that their installed biases listed today's product categories |
| Ladder retitled the inference axis; rungs marked as coarse-graining | §II.4 | The axis is derived; four named cells are not |
| **Retrieval demoted off the inference axis** | §II.4, R6 | "Large context, small $T$" is input entropy, already R6's fourth complexity axis. It was on the ladder because it is a capability people care about |
| **Calibration added as the irreducible cell** | §II.4 | The partition demands a cell for entropy no action can remove. Absent from v3 because hedging is not a product category |
| L1 probe replaced by a continuous ambiguity sweep | §IV.3, V.1 | A three-point test tests the author's discretization. The sweep locates the cuts instead of assuming them, and mirrors Finzi et al.'s masked-latent design |
| Gate restated as *monotone collapse* rather than *failure at L1* | §IV.3, §IV.9, V.1 | v3 would have terminated the program on a collapse at one hand-chosen construction |
| **Track B′ added** | §IV.5 | The inference axis needs the same empirical extraction as the content axis: annotate real traces for how task specification actually gets resolved |
| Gradation-drift risk added | V.9 | If cuts drift faster than a model can be trained, the Platonist is pragmatically right |

---

# Notation

| Symbol | Meaning |
|---|---|
| $\mathcal{F}$ | Space of possible target functions |
| $\mu$ | Task measure — the distribution over targets we care about |
| $\beta(f)$ | Description cost of $f$ in bits; a bias is the whole function $\beta$ |
| $T$ | Compute budget, covering both finding and running a program |
| $S_T(X)$ | Epiplexity — structural content extractable at budget $T$ |
| $H_T(X)$ | Time-bounded entropy — residual that must be stored, not derived |
| $H(\mu)$ | Entropy of the task measure; the irreducible floor |
| $\mathcal{A}$ | Adaptation operator (SFT, RL, or in-context learning) |
| $\mathrm{SDL}(f)$ | Surplus Description Length — acquisition cost of $f$ from a checkpoint |
| $\kappa$ | Curriculum controller; a distribution over situation types |
| $H(p \mid \text{context})$ | Residual task entropy — the inference axis |
| L0–L3 | Coarse-graining of that axis by mode of entropy reduction: none needed, observe, act, irreducible |
| $\mathcal{P}, [\![\cdot]\!], \pi$ | Program space, semantics, prior over programs |
| $\sigma$, $K(\sigma)$ | Re-encoding map and its description length |
| $p$ | A program; the latent task variable of an episode |

---

## Summary of the position

1. Pretraining produces knowledge and bias together. Bias is what pays for itself across instances; knowledge is what doesn't. Only knowledge is information-theoretically expensive; only bias is what finetuning consumes. The ratio has now been measured, and the structural share of natural corpora is tiny.
2. A bias is a prefix code on target space. The space of biases is the space of interpreters, quotiented by behavior, ordered by compilation cost, graded by runtime — and the grading is not optional, because the whole enterprise exists only for bounded observers.
3. Bias design is density estimation on task space: minimize $D_{\mathrm{KL}}(\mu \| 2^{-\beta})$. Epiplexity measures whether there is structure; only a $\mu$-relative quantity measures whether it is the right structure. Keep both.
4. Substrate and bias are the same object, so "which language" is the smallest term in the problem and "which primitives" is the largest.
5. The primitive basis should be *measured*, not chosen — by hand-writing minimal programs for real tasks and reading off what gets used.
6. That measurement requires no compute, produces the go/no-go number, and either kills or confirms the program's central empirical claim.

7. Procedural pretraining with determinate targets is established and is conceded. What is new is a coordinate system whose axes are derived and whose gradations are measured, and the extension above determinate targets — either this program's contribution or its ceiling, decided by one cheap sweep.
8. The axes can be derived. **Where to cut them cannot.** Both the primitive basis and the useful levels of residual entropy are facts about the practices being modelled, and both require their own extraction exercise before any generator is written.
9. The next action is **not** a spreadsheet. v1 said it was; that was wrong, because description length under a basis turns out to be a poor predictor of extractable structure. The next action is to reproduce a published cellular-automaton measurement on consumer GPUs, verify the instrument against a known answer, and only then design anything. The spreadsheet runs in parallel, and it is still free.
