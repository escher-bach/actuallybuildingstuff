# Dense Gradient Generation in Developmental Worlds

## A formal note on how interaction becomes learning

### Status of this document

This note defines the learning paradigm assumed by the baby-model project. It
does not prescribe a particular world, curriculum, model architecture, or
implementation. Its purpose is to prevent the language of worlds and actions
from silently turning the project into reinforcement learning from verifiable
rewards.

The central claim is:

> Interactive experience can be learner-conditioned while its learning signal
> remains dense, local, and token-supervised.

The world determines consequences. A separate gradient-generating teacher
determines which token distributions should shape the learner at each point in
the resulting interaction.

---

## 1. Why “environment” does not imply reinforcement learning

An environment is a source of stateful consequences. Reinforcement learning is
a family of ways to change a policy using reward.

These are logically independent.

A learner may act in an environment while learning through:

- supervised imitation;
- next-token prediction;
- state prediction;
- posterior prediction;
- correction against a teacher distribution;
- reinforcement from scalar outcomes;
- or mixtures of these.

The baby-model project uses environments because some experience exists only
when the learner's outputs affect what it observes next. It does not follow
that a completed trajectory must be reduced to a reward or that credit must be
assigned through policy gradients.

The relevant analogy is an interactive tutor, not a game scoreboard. A tutor
can respond to the move the student actually made while also providing a local
correction at that move. The response preserves the reality of the student's
choice; the correction prevents every lesson from being compressed into
“eventually succeeded” or “eventually failed.”

---

## 2. The five objects

The full developmental system contains five separable objects.

### 2.1 Learner

The learner is an autoregressive sequence model with parameters \(\theta\):

\[
p_\theta(x_i \mid x_{<i}).
\]

Its external boundary consists of tokens. Some tokens represent observations,
some represent its actions or predictions, and some may represent teacher
guidance or intermediate state.

### 2.2 World

The world is

\[
W=(S,A,O,T,\Omega,G),
\]

as defined in the foundations document. It determines which observation
follows an action and how the hidden state changes.

### 2.3 Renderer and parser

The renderer converts world objects into tokens:

\[
r_e : O \cup A \cup Z \to V^*,
\]

where \(e\) denotes the current representation and \(V\) is the token
vocabulary.

The parser converts a model-emitted token sequence into an action or a typed
error:

\[
d_e : V^* \to A \cup \{\mathrm{Error}\}.
\]

Keeping these functions explicit distinguishes a malformed expression from a
well-formed but poor action.

### 2.4 Presentation and scaffolding policy

A presentation policy \(P\) decides which aspects of the generated state are
shown and which control functions are supplied externally. It may reveal an
intermediate state, constrain available actions, name a subgoal, preserve a
memory, or withhold information.

The presentation policy changes the learner's experience without changing the
underlying world's semantics.

### 2.5 Gradient generator

The gradient generator \(\Gamma\) has privileged access to generated state,
history, and the intended developmental organization. It produces local token
targets:

\[
\Gamma(s_t,h_t,P)
\to
\{(i,q_i,m_i,c_i)\}.
\]

For each supervised token position \(i\):

- \(q_i\) is a target distribution over vocabulary tokens;
- \(m_i \ge 0\) is its loss weight;
- \(c_i\) identifies the supervised channel, such as action, prediction,
  belief, uncertainty, correction, or commitment.

The target may be a point mass when one token is intended, a distribution when
several continuations are acceptable, or an exact predictive distribution when
the world itself is uncertain.

The gradient generator is part of the teacher, not part of the world. A rock
falls according to the world's dynamics; the gradient generator decides which
representation of that consequence becomes a learning target.

---

## 3. The transcript

An interaction produces a transcript

\[
\tau=(x_1,x_2,\ldots,x_n).
\]

Each token also has a role. A minimal role set is:

- `OBSERVE`: supplied by the world or interface;
- `ACT`: emitted by the learner and parsed as an action;
- `PREDICT`: emitted by the learner as a prediction about the world;
- `STATE`: emitted or supplied as an intermediate representation;
- `GUIDE`: supplied by a teacher or scaffold;
- `ERROR`: supplied after an invalid or uninterpretable emission.

Roles are semantic channels, not necessarily literal tokens in the model's
context. An implementation may encode them through message positions, masks,
or typed delimiters.

Not every transcript token must be supervised. Environment observations may be
input-only. The model may also be asked to predict selected observation tokens
when learning the world's regularities is part of the developmental content.
The loss mask is determined by \(\Gamma\), not by the mere fact that a token
appears in the sequence.

---

## 4. Local token loss

Given transcript \(\tau\) and teacher targets from \(\Gamma\), the basic loss is

\[
\mathcal{L}(\theta;\tau)
=
\sum_{i=1}^{n}
m_i\,
\mathrm{CE}
\left(q_i,
p_\theta(\cdot\mid x_{<i})
\right).
\]

Equivalently, up to the entropy of the teacher distribution,

\[
\mathcal{L}(\theta;\tau)
=
\sum_i m_i
\mathrm{KL}
\left(
q_i
\;\|\;
p_\theta(\cdot\mid x_{<i})
\right).
\]

The parameter update is ordinary gradient descent:

\[
\theta' = \theta - \eta\nabla_\theta\mathcal{L}(\theta;\tau).
\]

The formal simplicity is intentional. The novelty is not a new optimizer. It
is the construction of worlds in which useful teacher distributions are
available densely and cheaply across long interactive histories.

### 4.1 Locality

A target is local when it supervises the model's behavior at the point where
that behavior is expressed.

Examples include:

- the next token of a well-formed action;
- a distribution over acceptable queries in the current state;
- the predicted consequence of the chosen action;
- an updated belief representation after a new observation;
- an uncertainty expression before commitment;
- a recovery action after the learner's earlier action changed the state.

The consequence of an action may unfold over many later tokens, but the teacher
need not wait for terminal success to say what organization was locally
appropriate.

### 4.2 Distributional targets

Dense supervision does not require pretending that every state has one uniquely
correct action.

If \(A^*(s,h)\) is a set of acceptable actions, the teacher may define a
distribution over that set. If actions express different legitimate tradeoffs,
the distribution can preserve those tradeoffs. If the teacher knows only a
partial ordering, the loss can be applied to the common prefix or to a typed
choice rather than fabricating an exact trajectory.

This matters because forcing one arbitrary teacher trace into a point target
would convert underdetermination in the world into false certainty in the
gradient.

---

## 5. Teacher-conditioned and learner-conditioned interaction

Dense token supervision supports two forms of episode generation.

### 5.1 Teacher-conditioned episodes

The teacher chooses the actions, the world returns their consequences, and the
model learns from the resulting sequence. This is ordinary supervised sequence
learning over generated interaction traces.

It is highly parallelizable because the trajectories can be generated without
consulting the current model.

### 5.2 Learner-conditioned episodes

The model emits an action \(a_t\). The world transitions using that actual
action:

\[
s_{t+1} \sim T(\cdot\mid s_t,a_t).
\]

The teacher simultaneously supplies a local target distribution for the
model's action or relevant internal channel. Crucially, the world does not
replace the learner's action with the teacher's preferred action. The next
observation is the consequence of what the learner actually did.

This produces on-policy data in the descriptive sense that the learner affects
the states it visits. The optimization remains supervised cross-entropy. No
policy-gradient estimator is required.

The distinction is useful:

- Teacher-conditioned interaction provides broad, cheap exposure.
- Learner-conditioned interaction provides experience of the learner's own
  omissions, malformed actions, commitments, and recoveries.

They are two sources of developmental experience within one gradient paradigm,
not rival learning theories.

---

## 6. Dense token supervision

“Dense” has three meanings here.

### 6.1 Temporal density

Learning targets occur throughout an interaction rather than only at its end.

For a transcript of \(n\) tokens, define gradient-bearing token density

\[
\delta(\tau)
=
\frac{1}{n}
\sum_{i=1}^{n}
\mathbf{1}[m_i>0].
\]

This is only a descriptive quantity. Two supervised tokens can carry very
different developmental content. Its role is to distinguish a token-rich but
gradient-sparse trajectory from one in which supervision is distributed across
the unfolding process.

### 6.2 Functional density

One generated situation can provide targets for several kinds of organization:

- parsing and valid expression;
- state prediction;
- action selection;
- belief maintenance;
- uncertainty;
- error recognition;
- recovery;
- stopping or commitment.

These are not necessarily separate output heads. They may be typed regions of
one sequence. Functional density means the generated state supports multiple
locally meaningful learning events.

### 6.3 Credit locality

The teacher distribution applies directly to the token decision being shaped.
The learning system does not infer the value of that token solely from a scalar
assigned after many subsequent decisions.

Dense supervision therefore concerns the topology of credit, not merely the
number of labels.

---

## 7. What high throughput means

Raw token generation rate is not the project's relevant throughput.

Define the gradient-bearing volume of a transcript as

\[
B(\tau)=\sum_i \mathbf{1}[m_i>0],
\]

or, when weights represent comparable units,

\[
B_m(\tau)=\sum_i m_i.
\]

Let \(C(\tau)\) be the computational cost of generating the world trajectory,
constructing targets, and performing the required model passes. An operational
supervised-throughput quantity is

\[
\mathcal{H}
=
\frac{\mathbb{E}[B(\tau)]}
{\mathbb{E}[C(\tau)]}.
\]

This quantity is not offered as a universal measure of learning value. It
states what the phrase **high throughput** means operationally in this project:
many locally supervised model decisions per unit of generation and training
cost.

High throughput has four sources:

1. **Cheap world evolution.** The generator knows the state and can compute
   consequences without expensive search through the learner's problem.
2. **Privileged target construction.** Because the generator owns the hidden
   state, it can construct belief, prediction, action, and uncertainty targets
   that would be unavailable in scraped text.
3. **Reuse of generated state.** One trajectory can support several
   gradient-bearing channels and alternative renderings.
4. **Local credit.** Each supervised decision uses direct token loss rather
   than repeated rollouts to estimate a distant reward's contribution.

Learner-conditioned interaction requires sequential model calls and is
therefore less parallel than teacher-conditioned generation. The formalism does
not hide that cost. High throughput comes from using learner conditioning where
its distinctive experience is needed while retaining dense local targets at
the states it produces.

---

## 8. Contrast with RLVR

The contrast is easiest to state by following the learning signal.

### Reinforcement learning from verifiable rewards

A trajectory \(\tau\) receives a scalar or small set of outcome rewards:

\[
R(\tau).
\]

The optimizer increases the probability of sampled token sequences according
to an advantage or return derived from that reward. Credit for an early token
is mediated by its statistical relationship to later success.

### Dense gradient generation

The generated state supplies token-level teacher distributions:

\[
q_i(\cdot\mid s_t,h_t)
\]

at many positions. The optimizer directly reduces their cross-entropy against
the model distribution.

The distinction is not that one system has actions and the other does not.
Both may. It is not that one system is on-policy and the other must be offline.
Both may visit learner-conditioned states. The distinction is the form and
location of the learning signal:

```text
RLVR:       sampled trajectory -> outcome reward -> assigned credit

This work:  generated state -> local token distribution -> direct gradient
```

Outcome information may still appear inside a developmental world as an
observation. Success and failure are things the learner should understand. They
do not have to be the sole source of gradient.

---

## 9. Gradient generation as the real product of the world

It is tempting to describe a synthetic generator as a machine that produces
data. For this project, that description is incomplete.

The generator produces a structured relation among:

- hidden world state;
- learner-visible history;
- possible actions;
- actual consequences;
- teacher-accessible local target distributions.

The serialized transcript is one view of this relation. The gradient generator
turns selected parts of it into learning pressure.

The real product is therefore not a large file of synthetic text. It is a
repeatable process for generating useful gradients at many points in an
interaction.

This reframes the central design question:

> What worlds make the desired organization locally teachable through tokens?

That question joins the developmental theory to the actual optimization regime
without reducing development to reward maximization.

---

## 10. Compact definition

A **dense gradient-generating developmental system** is a tuple

\[
\mathcal{D}=(W,R,P,\Gamma,M),
\]

where:

- \(W\) is a stateful world;
- \(R=(r_e,d_e)\) renders observations and parses actions;
- \(P\) controls revelation and scaffolding;
- \(\Gamma\) maps privileged generated state to weighted local token target
  distributions;
- \(M=p_\theta\) is the token learner.

Interaction may be teacher-conditioned or learner-conditioned. In either case,
model parameters change primarily through

\[
\nabla_\theta
\sum_i m_i\,
\mathrm{CE}
\left(q_i,p_\theta(\cdot\mid x_{<i})\right).
\]

The paradigm is high-throughput when the system can generate many diverse,
locally supervised token decisions per unit of computation while preserving
the persistent causal organization of an interactive world.

