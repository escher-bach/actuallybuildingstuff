# Training Dynamics of Dense Gradient Generation

## Throughput, interaction cost, and the meaning of “faster”

### Status of this document

This document follows the current gradient paradigm. It concerns the economics
of the developmental system:

\[
\mathcal D=(W,R,P,\Gamma,M).
\]

It asks how dense gradient generation changes training cost relative to ordinary
text pretraining. It does **not** decide whether a world is worth training on.
That is the separate question of [world correspondence](WORLD-CORRESPONDENCE.md).

---

## 1. The short answer

The system is not automatically faster than text pretraining.

- A teacher-generated transcript can be trained at approximately ordinary
  sequence-pretraining speed after the transcript and targets have been made.
- A learner-conditioned interaction is slower because the current model must
  emit actions before the world can reveal what happens next.
- Dense local supervision can still make the system faster **to a selected
  capability** if it produces more useful developmental change per unit of
  compute than ordinary text does.

Three quantities must remain separate:

| Quantity | Question |
|---|---|
| Raw token throughput | How many tokens does the hardware process per second? |
| Gradient throughput | How many directly supervised token decisions occur per unit cost? |
| Developmental throughput | How much reusable learner organization changes per unit cost? |

Text pretraining can be very fast in the first sense and dense in the purely
numerical sense that loss applies to almost every token. The developmental
system does not claim to exceed that label density. Its claim is narrower:

> Privileged generated state permits deliberately placed, local gradients for
> selected relations—state, action, uncertainty, recovery, and commitment—that
> uncontrolled text supplies only diffusely.

---

## 2. A cost model

Let:

- \(N\) be the number of transcript tokens processed in training;
- \(A\) be learner-generated action tokens decoded using current model weights;
- \(Q\) be the number of sequential learner–world boundaries;
- \(c_{\mathrm{train}}\) be training cost per transcript token;
- \(c_{\mathrm{decode}}\) be autoregressive decode cost per action token;
- \(C_W\) be world transition, parsing, and rendering cost;
- \(C_\Gamma\) be target-construction cost;
- \(C_{\mathrm{sync}}(Q)\) be synchronization and utilization cost.

Ordinary text pretraining costs approximately

\[
C_{\mathrm{text}}=N c_{\mathrm{train}}.
\]

Dense gradient generation costs approximately

\[
C_{\mathcal D}
=
N c_{\mathrm{train}}
+A c_{\mathrm{decode}}
+C_W+C_\Gamma+C_{\mathrm{sync}}(Q).
\]

Its raw cost ratio is therefore

\[
\alpha
=
\frac{C_{\mathcal D}}{C_{\mathrm{text}}}
=
1
+\frac{A c_{\mathrm{decode}}}{N c_{\mathrm{train}}}
+\frac{C_W+C_\Gamma+C_{\mathrm{sync}}(Q)}{N c_{\mathrm{train}}}.
\]

This equation does not predict one universal slowdown. It names the terms that
must be paid.

### 2.1 Idealized model-compute overhead

For a transformer, a forward-plus-backward training token costs roughly a few
times a forward-only decoding token. If

\[
c_{\mathrm{train}}\approx3c_{\mathrm{decode}},
\]

then idealized additional model FLOPs are about

\[
1+\frac{A}{3N}.
\]

If 20% of transcript tokens are current-model action tokens, the lower-bound
overhead is about 7%. If every token must be decoded online, it is about 33%.

These are not wall-clock predictions. Decoding is less efficiently batched than
training, and learner–world boundaries create waiting and coordination. With
effective decode inefficiency \(\lambda\ge1\):

\[
\alpha
\approx
1+\lambda\frac{A}{3N}
+\frac{C_W+C_\Gamma+C_{\mathrm{sync}}(Q)}{N c_{\mathrm{train}}}.
\]

The number of sequential turns and batch fragmentation can matter more to
wall-clock time than raw FLOP ratio.

---

## 3. Three generation regimes

### Teacher-conditioned generation

The teacher chooses a trajectory, the world evaluates it, and the full
transcript can be generated without consulting current learner weights. This
permits parallel generation, caching, separate world workers, and ordinary
batched cross-entropy training. Raw training speed can remain close to text
pretraining once the data pipeline is provisioned.

### Learner-conditioned generation

The model emits an action, the parser interprets it, and the world transitions
using the action actually taken:

\[
s_{t+1}\sim T(\cdot\mid s_t,a_t).
\]

This produces states caused by the learner's choices, misunderstandings, and
malformed actions. It cannot be fully precomputed, and multi-turn episodes add
sequential model calls. Raw throughput therefore falls.

The optimization remains local cross-entropy. A world responding to actual
learner action does not make the update policy-gradient learning.

### Hybrid generation

Most exposure can be teacher-conditioned, while learner-conditioned regions
are used where action ownership changes the developmental object. A teacher can
establish a situation, invite one diagnostic action, return its real
consequence, and then continue with generated material or another learner
choice.

If \(p\) is the fraction of cost spent in learner-conditioned regions:

\[
\alpha_{\mathrm{mix}}
\approx
(1-p)\alpha_T+p\alpha_L.
\]

Interactivity belongs where it creates otherwise unavailable experience, not
everywhere merely because the system has worlds and actions.

---

## 4. Dense supervision

For transcript \(\tau\), let \(m_i\) mark whether token position \(i\) carries
a direct teacher target. The gradient-bearing token count is

\[
B(\tau)=\sum_i\mathbf1[m_i>0].
\]

An operational gradient throughput is

\[
\mathcal H
=
\frac{\mathbb E[B(\tau)]}{\mathbb E[C(\tau)]}.
\]

This distinguishes a long trajectory whose learning signal appears only at a
terminal answer from one where local distributions can supervise valid action
formation, predicted consequences, state updates, belief updates, uncertainty,
recovery, and commitment.

Dense supervision has three aspects:

1. Temporal density: targets occur throughout an interaction.
2. Functional density: one generated state supports several target types.
3. Credit locality: targets apply directly to the token decision being shaped.

Targets may be point masses, distributions over acceptable actions, or
predictive distributions for genuinely uncertain worlds. Dense supervision must
not turn legitimate underdetermination into a falsely unique teacher trace.

---

## 5. When the system is faster overall

Let \(g_{\mathrm{text}}\) and \(g_{\mathcal D}\) be useful developmental change
per token, and \(v_{\mathrm{text}}\) and \(v_{\mathcal D}\) raw throughput. For
a fixed amount \(K\) of intended learner organization:

\[
T_{\mathrm{text}}
=
\frac{K}{v_{\mathrm{text}}g_{\mathrm{text}}},
\qquad
T_{\mathcal D}
=
\frac{K}{v_{\mathcal D}g_{\mathcal D}}.
\]

The developmental system is faster when

\[
\frac{g_{\mathcal D}}{g_{\mathrm{text}}}
>
\frac{v_{\mathrm{text}}}{v_{\mathcal D}}.
\]

The gain in useful learning per token must exceed raw-throughput loss. The
right-hand side is primarily engineering. The left-hand side depends on world
choice, rendering, guidance, target construction, and correspondence to later
activity. It cannot be claimed merely because a generator produces clean labels.

---

## 6. Controller overhead

A developmental controller may select world region, hidden-state complexity,
revelation, rendering, guidance, action ownership, conditioning regime, and
target channels. Its own overhead can be small when it uses aggregate learner
information rather than inspecting every token.

That does not make the full system cost-equivalent to offline pretraining. The
distinct cost of learner-conditioned action remains visible in \(A\), \(Q\), and
\(C_{\mathrm{sync}}\).

---

## 7. Final position

1. Raw training is not generally faster than text pretraining.
2. Teacher-conditioned material can approach ordinary training throughput.
3. Learner-conditioned interaction is slower in raw wall-clock terms.
4. Dense local targets can make the system faster to a selected capability if
   useful developmental gain exceeds raw-throughput cost.
5. No throughput argument establishes usefulness; that requires world
   correspondence.

High throughput means:

> Many locally meaningful gradient-bearing decisions per unit total cost, with
> online interaction reserved for places where it creates otherwise unavailable
> developmental experience.

