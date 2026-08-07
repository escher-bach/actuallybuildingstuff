# Vein 2.8 — Agency, Metacognition, and Self-Regulation

Scope: metacognition/metamemory (knowing-that-you-know), information foraging /
search-termination, convention formation under underdetermination, behavioural
error monitoring without external feedback, goal management (prospective
memory, goal shielding, task-switching, implementation intentions, interruption
resumption).

HELD OUT and not fetched, vein-specific: sense-of-agency neuroscience and the
comparator model (intentional binding, efference copy, felt-agency neural
basis). Behavioural error-monitoring (ERN, post-error slowing, conflict
monitoring) is IN SCOPE and treated separately from that seal — see §11.

HELD OUT, standing partition: clinical reasoning/differential diagnosis;
metacognition as clinical deficit (anosognosia, psychiatric insight);
developmental/infant paradigms; non-primate animal cognition (primate
metamemory IS in scope; corvid/cephalopod is NOT); production incident
response/SRE/SQL/Erlang; construction geometry/epsilon-delta; ILP; pool-based
active learning; clinical-trial design; verbal/linguistic-aptitude
psychometrics; Latin-square/graph-colouring; ARC-AGI-3 material.

Reading key: **VERIFIED** = fetched and read (tier noted: abstract / body /
tertiary-secondary). **LEAD** = seen only in a WebSearch snippet or synthesis,
not independently fetched — treat any number under LEAD as unconfirmed.

---

## 1. The four predictions

| # | Prediction | Verdict | Basis |
|---|---|---|---|
| **S1** | Substantial literature on knowing-whether-you-know, containing *measured dissociations* between knowing and knowing-that-you-know. | **CONFIRMED** | Nelson & Narens' monitoring/control split explicitly permits dissociation by construction (LEAD-tier characterization, see caveat §2); Koriat (1993, VERIFIED abstract) concludes FOK is *not* direct trace-access, i.e. is computed from something other than the fact it's supposed to report; Hampton (2001, VERIFIED body) gives a **quantified** primate dissociation (§2); TOT is a qualitative dissociation by definition. See §2 for the full evidence table. |
| **S2** | At most two genuinely new primitives emerge from this vein. | **CONFIRMED** — exactly **one** proposed (`termination-gate`), two other candidates considered and folded into existing entries. | §7. |
| **S3** | Precommitment/implementation-intentions has a literature but is **not** a composable operation — it's a protocol. | **CONFIRMED** | Gollwitzer's framing (LEAD tier — WebSearch synthesis of the 1999 paper, not independently fetched) is consistently described as *delegating control of a future decision to a situational cue*, i.e. restructuring the episode's control flow, not emitting a value mid-trace. Structural argument in §1.3 below is mine, built on that characterization. |
| **S4** | The literature names at least one agency activity absent from the given list of seven. | **CONFIRMED**, with a gap flagged | Found named literatures for 5 of 7 listed candidate activities (epistemic vigilance / deciding what to check; precommitment / binding future judgement; help-seeking / escalation, adult-framed only; delegation to automation / doing-vs-delegating; convention nameability as a partial proxy for detecting out-of-convention cases). **Found nothing for "recognising an operation is missing rather than mis-named"** — flagged explicitly per the brief's instruction, not glossed over. See §1.4. |

### 1.1 S1 in full — see §2, the dedicated section.

### 1.2 S2 in full — see §7.

### 1.3 S3 — why precommitment is a protocol, not an operation

Gollwitzer (1999, "Implementation intentions: strong effects of simple
plans," *American Psychologist* 54:493–503) frames an implementation
intention as an if-then binding ("Whenever situation X arises, I will
initiate response Y!") that **delegates control of a future decision to a
situational cue in advance**, so that when the cue fires the response runs
without being re-decided (paraphrase of WebSearch-synthesized direct quotes,
not independently fetched full text — LEAD tier for the exact wording, but
the mechanism description is corroborated across three independent secondary
sources returned by search).

This does not fit the shape of any primitive in the register. Every existing
`operation` entry is something computed **within** a single query/response
step of a trace: narrow a belief state, walk a structure, invert a bijection.
An implementation intention instead **rewrites how a later step will be
decided at all** — it spans from the moment of commitment to the moment of
triggering, potentially with an arbitrary amount of unrelated activity
between them (this is exactly the resumption-lag structure Altmann & Trafton
study, §5-adjacent). Composing it into a solution trace the way
`hypothesis-elimination` or `constraint-propagation` compose would require
the register to represent "adopt a standing rule that overrides case-by-case
computation for the rest of the episode" — a control-flow modification, not a
value computation. That is a different *kind* of entry than either
`operation` or the existing `trace-act` (`epistemic-status-tagging`), which
both describe things done **at** a step, not things done **to the rest of the
episode's decision procedure**.

Quantified effect (LEAD, not independently verified — see §9): Gollwitzer &
Sheeran's 2006 meta-analysis is reported across multiple secondary sources as
d = .65 across 94 independent tests and >8,000 participants for goal
attainment, d = .61 specifically for overcoming failure-to-get-started. I did
not independently fetch this meta-analysis or a source quoting its numbers
verbatim from within the paper itself; the number is consistent across two
independent secondary characterizations but is reported here as LEAD per the
accuracy requirement.

**Recommendation**: do not add precommitment as a primitive. If the register
ever wants to capture "bind a future decision now, execute without
re-deciding later," it needs a third `kind` (something like `protocol`,
episode-level control-flow), not a new entry under `operation`.

### 1.4 S4 — activities named, evidence tier, and the one gap

| Listed activity | Named in literature? | Source, tier |
|---|---|---|
| Deciding which claims are worth checking | YES | Sperber et al. 2010, "Epistemic Vigilance" — trust-vs-scrutiny as "two ends of one graded setting." **LEAD** (two direct-fetch attempts 403'd; description from WebSearch synthesis of definitional passages). |
| Noticing a discrepancy between a record and reality | Partially | Proofreading/error-detection literature's "Discrepancy-Attribution hypothesis," a two-stage model of conscious error detection. **LEAD** (search-snippet only, source paper not fetched). |
| Binding one's own future judgement | YES | Same literature as S3: Schelling (self-command, precommitment), Elster 1979 *Ulysses and the Sirens*. **LEAD** (search synthesis, no primary fetched). |
| Detecting a novel case falls outside a convention | Partially | The ad-hoc-convention literature (§4) studies *extension* to novel cases, and "nameability" (a priori naming consensus) is the closest operationalized proxy for "how confident should I be this case is covered" — but no source found that separately, explicitly studies *detecting out-of-coverage-ness* as its own act, distinct from just extending anyway. |
| Deciding when to escalate rather than decide | YES, adult-framed sources only | "Metacognitive control processes in question answering: help seeking and withholding answers" (*Metacognition and Learning*, 2021) and Karabenick's adult self-regulated-learning help-seeking literature. **LEAD** (snippet only). Nelson-Le Gall's foundational executive/instrumental distinction is **not used as a load-bearing citation** here — see §11, it is classroom/children's research and adjacent to the developmental seal; a fetch of it was attempted, failed before any content was read, and is logged as a near-miss rather than a source. |
| Allocating work between doing and delegating | YES | Lee & See 2004, "Trust in Automation: Designing for Appropriate Reliance." **VERIFIED abstract** (direct quote: "trust influences reliance on automation... particularly guiding reliance when complexity and unanticipated situations make a complete understanding of the automation impractical"). Body paywalled. |
| Recognising an operation is missing rather than mis-named | **NOT FOUND** | Searched under this framing specifically; nothing in this vein's search surfaced literature targeting this exact distinction. Reported as a real gap, not a search failure — flagged per the brief's explicit instruction. |

---

## 2. Knowing when you do not know — the dissociation evidence

| Finding | Numbers | Location | Tier |
|---|---|---|---|
| Nelson & Narens (1990) monitoring/control framework: meta-level holds a model of the object-level; "monitoring" and "control" are separate information flows that can dissociate (monitor without controlling, etc.) | — (framework, not a number) | Characterization from WebSearch synthesis across multiple secondary pages; the primary chapter and a ScienceDirect abstract page both 403'd on direct fetch | **LEAD** |
| Koriat (1993) accessibility model: FOK is computed from the *accessibility of partial retrieval cues*, not from privileged access to the target itself | Direct quote: "results challenge the view that FOK is based on a direct, privileged access to an internal monitor" | Abstract, *Psychological Review* 100(4):609–639 | **VERIFIED, abstract** |
| Koriat (1995) "Dissociating knowing and the feeling of knowing: further evidence for the accessibility model" — title and existence confirmed; full text not read (corrupted PDF on fetch) | — | — | **LEAD** (title/existence only) |
| Brown & McNeill (1966) tip-of-the-tongue: participants in TOT states could report the target's first letter, syllable count, and stress pattern at better-than-chance accuracy *before* successful recall | No percentage located in fetched text — **figure not located** | Wikipedia's synthesis of the primary study | **VERIFIED** (tertiary source only — Wikipedia, not the 1966 paper itself) |
| Shimamura & Squire (1986): amnesic patients (excluding Korsakoff's-syndrome subtype) gave FOK predictions as accurate as controls despite severely impaired recall itself | — | — | **LEAD**, and flagged as adjacent to the clinical-population seal (not itself sealed — this is not anosognosia/insight research — but a discretionary decision was made not to fetch the primary patient-population paper; treated as background only, not load-bearing |
| **Hampton (2001)**, rhesus monkeys, delayed-matching-to-sample with a "decline the test" option: accuracy on freely-chosen vs. forced trials | M1: **88% vs. 67%**; M2: **78% vs. 69%** | Discussion | **VERIFIED, body** |
| Hampton (2001): uncertainty/decline-response rate rose with memory delay | M1 declined **85%** of long-delay trials vs. **43%** of short-delay trials (initial trials) | Experiment 3 | **VERIFIED, body** |
| Hampton (2001) sample size | 2 male rhesus macaques, 7.5 yrs old at study start | General Methods | **VERIFIED, body** |

**Reading**: S1 is confirmed on two independent grounds. (1) A genuine
*quantified* dissociation exists and is the strongest single number in this
section: Hampton's monkeys are systematically **more accurate when they
themselves chose to be tested than when forced** (88 vs. 67%, 78 vs. 69%) —
i.e., the animal's own opt-in/opt-out choice carries information about its
memory state beyond what raw performance on forced trials would predict, and
this is a real primate result, in scope per the seal. (2) The *mechanism*
literature (Koriat) independently explains *why* dissociations should be
expected at all: if FOK is computed via a cheap accessibility proxy rather
than by reading the actual memory trace, then miscalibration (FOK high,
recall wrong; FOK low, recall correct) is the predicted default, not an
anomaly — this is a structural/mechanistic argument, not itself a number, and
is reported as such.

**What was NOT found with a number, and is explicitly not claimed**: no
fetched source gave a percentage for how often Brown & McNeill's TOT
participants correctly guessed partial letters (a number I recall existing in
the general literature, but I did not read it in anything I fetched, so it is
not reported).

---

## 3. Stop/seek decisions — what governs terminating a search

| Finding | Numbers | Location | Tier |
|---|---|---|---|
| Pirolli & Card, information foraging theory: patch-leaving modeled via Marginal Value Theorem — leave a patch when its instantaneous rate of gain falls below the environment's average rate of gain across patches | No numeric threshold extractable from the fetched PDF; the *shape* of the rule (threshold on rate-of-gain, informed by travel cost and patch richness) is confirmed | Body (PDF text extraction was partial; treat as **paraphrase**, not quotation) | **VERIFIED, body, paraphrase** |
| "Information scent" — imperfect proximal cues (link text, citations) standing in for the true value of unseen information, used to decide whether to keep following a path | — (definitional) | Body | **VERIFIED, body, paraphrase** |
| **Linear threshold model of human optimal-stopping** (ticket-buying task, sequential, no recall of declined options, 200 trials/experiment, real payoffs; Experiment 3 replicates on 60 real Amazon products): humans lose **6%** of optimal earnings; mean stop position 4.7 vs. optimal 5.2; acceptance rate at position 9 is **28% [95% CI 26–29%]** vs. optimal 50% | See left | Results/Methods | **VERIFIED, body** |
| Same study: 3-parameter linear threshold model `t(i+1) = t(i) + δ·i`, acceptance probability `θᵢ = 1/(1+exp{β(xᵢ−tᵢ)})`, fits human behaviour far better than 10-parameter unrestricted models | 3 vs. 10 free parameters | Methods/Model | **VERIFIED, body** |
| Same study: environment shape reverses which direction people err — in a left-skewed ("scarce") environment people search *longer* than optimal; in a right-skewed ("plentiful") environment they stop *earlier* than optimal | Directional finding, magnitude not separately quoted | Experiment 2 | **VERIFIED, body** |
| Classic "37% rule" / secretary-problem framing as the textbook optimal answer against which the above is compared | 37% of the sequence as the optimal look-then-leap boundary, for the idealized full-information-free secretary problem | Generic exposition (not a primary empirical study) | **LEAD** |

**Reading**: humans do not compute the Bayes-optimal stopping boundary; they
run a *cheap, position-linear threshold heuristic* that is measurably
suboptimal (6% loss) and whose error *direction depends on the shape of the
value distribution*, not just its mean. This is the empirical backbone for
the new-primitive argument in §7: the "should I keep searching" decision, as
actually implemented, is a comparison against a threshold computed from a
cheap signal (position, or accessibility in the metacognition literature),
not a readout of a fully-enumerated posterior.

---

## 4. Conventions under underdetermination

| Finding | Numbers | Location | Tier |
|---|---|---|---|
| Lewis (1969): convention = "everyone conforms to R; everyone expects everyone else to conform to R," self-perpetuating, and *arbitrary* (an alternative regularity could have served) | Direct quote of the definitional passage | Body (Stanford Encyclopedia of Philosophy, "Convention" entry) | **VERIFIED, body** |
| SEP entry: **no discussion found of how conventions extend to novel/unprecedented cases** — Millikan's "weight of precedent" (§6.3 of the entry) is the nearest adjacent idea, and even that doesn't address extension systematically | — | Body | **VERIFIED, body** (an absence, not a number — reported as such) |
| Ad hoc conventions generalize to new referents (arXiv 2509.05566): 3-phase dyadic reference-game study, tangrams organized into "threads" of 3 visually-similar images across pre/interactive/post phases | N = **302** participants | Abstract + body | **VERIFIED, body** |
| Same study: description similarity for undiscussed same-thread tangrams rose from pre- to post-phase | b = **0.03** [95% CI 0.02–0.03] | Results §2.2 | **VERIFIED, body** |
| Same study: effect stronger for threads whose interactive target was repeated across blocks | Bayes factor = **15.53**, posterior probability **0.94** | Results §2.2 | **VERIFIED, body** |
| Same study: generalization strength as a function of visual similarity — best fit is quadratic, consistent with Shepard's Universal Law of Generalization (nonlinear decay) | quadratic term b = **0.25** [CI 0.12–0.38]; linear term b = **1.84** [CI 1.17–1.96] | Results §2.3 | **VERIFIED, body** |
| Same study: "nameability" (a priori naming consensus) raises baseline alignment but does **not** significantly affect the *change* in alignment (i.e., doesn't change how much generalization occurs) | baseline effect b = 0.13 [CI 0.10–0.16]; change effect b = 0.02 [CI −0.04–0.07] | Results §2.4 | **VERIFIED, body** |
| Semantic uncertainty guides extension of conventions to new referents (arXiv 2305.06539): two dyadic studies on the same nameability question, distinct target transfer | N = **240** | Abstract | **VERIFIED, abstract only** (body fetch 404'd) |

**Reading**: this is the cleanest, best-quantified section of the whole vein.
The literature directly answers the brief's priority question — **people
extend a convention past its explicitly-established boundary using a
similarity/decay function to the nearest already-conventionalized case
(Shepard's law), not by treating the convention as inapplicable outside its
literal precedent set.** Lewis's own philosophical theory (the founding
text) is silent on exactly this point; the empirical psycholinguistics
literature is where the underdetermination question actually gets studied
and measured.

---

## 5. Error monitoring without feedback

| Finding | Numbers | Location | Tier |
|---|---|---|---|
| Gehring, Goss, Coles, Meyer & Donchin (1993), "A Neural System for Error Detection and Compensation": ERN is enhanced when accuracy is emphasized, diminished when speed is emphasized, and related to attempts to compensate for the error | Direct quote of abstract; **no microvolt/ms numbers were in the fetched abstract itself** | Abstract, *Psychological Science* 4(6):385–390 | **VERIFIED, abstract** |
| ERN amplitude ~ −5 to −10 μV in adults, peak ~100ms post-EMG-onset on error trials, correlated with force reduction / correction probability / next-trial slowing | −5 to −10 μV, ~100ms | — | **LEAD** (WebSearch synthesis; the direct-fetched abstract did **not** contain these numbers — flagged explicitly per the accuracy requirement, do not treat as confirmed) |
| Post-error slowing (PES) occurs after **self-committed** errors but not after errors that are externally inserted/induced | Citing 3 studies within the review: De Bruijn et al. 2004b; Logan & Crump 2010; Steinhauser & Kiesel 2011 | Body | **VERIFIED, body** (of the citing review; the 3 underlying studies themselves not independently fetched) |
| PES magnitude, flanker task | 200ms RSI: **M = 59ms** (SEM 21); 3000ms RSI: **M = −1ms** (SEM 9) | Experiment 1 of the fetched review | **VERIFIED, body** |
| PES magnitude, longitudinal (4.6-month retest) | Session 1: **2.8%** RT increase; Session 2: **1.5%**; test-retest Spearman ρ = **0.54**, p = **.04** | Experiment 2 of the fetched review | **VERIFIED, body** |
| Error rates in the same studies | 9–10% (Exp 1); 16.8% and 14.9% (Exp 2, two sessions) | Body | **VERIFIED, body** |
| Conflict monitoring theory (Botvinick, Braver, Barch, Carter & Cohen, 2001): control demand evaluated by monitoring for *response conflict*, formalized via two linked computational models, with the anterior cingulate cortex as the proposed substrate | — | Abstract | **VERIFIED, abstract** |

**Reading**: PES is the field's cleanest instance of the exact capability
named in the brief — **detecting your own error with no external signal at
all**, isolated experimentally by comparing self-committed vs.
externally-inserted errors and finding the slowing effect requires the
former. The *comparator-model* framing here is behavioural (compare intended
response vs. executed response / evolving correct-response activation), and
is treated as distinct from the sealed sense-of-agency comparator literature
(intentional binding, efference copy) per the brief's own carve-out — see
§11 for how that boundary was held.

---

## 6. Candidate families

### 6.1 Coverage-gated meta-query (motivated by §2 — FOK/JOL as a proxy signal, not a mind-read)

- **Theta**: a partial function over a finite symbol domain — a value table
  with an explicit, *stipulated* "covered" subset and an "uncovered" subset
  (both are facts about theta, not about any solver).
- **P_Theta(·|k)**: sample table size, coverage density, and a set of
  near-miss "distractor" keys (present-looking but uncovered) controlled by
  `k`.
- **X**: two query types — (a) value query "what is f(j)?", (b) meta-query
  "is j covered — do you know f(j), yes/no?"
- **f (oracle)**: for (a), return the true value if covered, else a fixed
  UNDEFINED token; for (b), return the covered/uncovered fact directly from
  theta. **Both are pure facts about theta — no inspection of the solver's
  internal state is required**, which is exactly what keeps this admissible
  where a literal replica of human FOK-grading (grade whether the solver's
  *stated confidence* matches its *own* true retrieval-accessibility) would
  not be (see §8).
- **Encodings**: (1) an explicit table rendered with visible holes; (2) a
  generative rule (e.g. modular arithmetic) whose domain is restricted to a
  subrange, so recognizing "uncovered" requires recognizing an input falls
  outside the rule's stated domain rather than reading a blank cell.
- **Posterior enumerable**: yes, at L1/L3 — over candidate coverage patterns
  consistent with the trials seen so far.
- **q\* (L2)**: nearest fit is `structure-walk-query` toward the coverage
  boundary (unexercised keys near known covered/uncovered ones).
- **A1–A7**:

| Criterion | Verdict | Note |
|---|---|---|
| A1 | **PASS** | Oracle reads theta's coverage set, never the solver's confidence — the naive version of this family (grade calibration against the solver's *true* internal uncertainty) would fail A1; this repair (make coverage a theta-level fact) is what makes it pass. |
| A2 | PASS if symbol-permutable | — |
| A3 | PASS | table vs. domain-restricted-rule encodings |
| A4 | repairable | tie distractor density / near-miss subtlety to `k` |
| A5 | PASS | — |
| A6/A7 | plausible at L2 | teacher policy walks the coverage boundary, one-pass given theta |

**Verdict: PASS.**

### 6.2 Optimal-stopping / threshold-vs-precomputed-policy (motivated by §3)

- **Theta**: a value-generating process (distribution family + parameters),
  a per-step cost, and a horizon — small enough that the *true* optimal
  stopping policy can be computed once via backward induction **at
  generation time**, then frozen as part of theta.
- **k**: horizon length, distribution skew, state-space size for the DP.
- **X**: at each position, "accept/continue" given the sequence so far.
- **f**: compare the solver's accept/continue choice to the precomputed
  optimal policy for that state. The DP runs once per episode at generation
  time, not once per solver query — this is what keeps it O(1) in the
  A1 sense (repairable/needs an explicit bound: DP cost must stay sublinear
  in whatever budget dimension `k` scales).
- **Encodings**: (1) numeric price/value stream (ticket-buying framing);
  (2) narrative "candidate interview" framing (classic secretary problem) —
  same math, different surface.
- **Posterior enumerable**: yes, over the small (position, best-so-far)
  state space.
- **q\***: not applicable in the query-selection sense — this is a
  stop/go family, not an information-seeking one.
- **A1–A7**:

| Criterion | Verdict | Note |
|---|---|---|
| A1 | repairable | DP must be precomputed once at generation time and must stay cheap as `k` grows — needs an explicit bound, not automatic |
| A4 | **flagged risk, repairable** | PMC7293628's own finding is the warning sign here: a **3-parameter linear heuristic gets within 6% of optimal** — any family built naively risks a cheap heuristic nearly solving it. Repair: make the *gap* between the naive linear heuristic and the true optimal threshold widen with `k` (e.g. adversarially-shaped, multimodal value distributions where linear thresholds diverge further as horizon grows). |
| A2, A3, A5 | PASS | numeric vs. narrative encodings; symbol-permutable if values are |

**Verdict: PASS/repairable**, with the A4 risk stated concretely rather than
asserted.

### 6.3 Convention extension by stipulated decay kernel (motivated by §4)

- **Theta**: a labeling/description function defined explicitly only on a
  "precedent set" of points in a similarity space, **plus a stipulated
  extension rule** (e.g. an explicit similarity-weighted kernel with
  parameters sampled per episode) that is the *ground truth* for any point
  outside the precedent set.
- **k**: precedent-set density, similarity-space dimensionality, kernel
  sharpness.
- **X**: "what is the correct label for point p" (p may be inside or outside
  the precedent set).
- **f**: apply the stipulated kernel directly — theta-only, no reference to
  what a real population of people would actually say.
- **Encodings**: (1) abstract symbol space with an explicit numeric metric;
  (2) natural-language-flavored items (tangram-style) with an explicit
  graph/taxonomy distance standing in for similarity.
- **Posterior enumerable**: yes, over kernel parameters consistent with
  observed precedent-label pairs, at L1/L3.
- **A1–A7**:

| Criterion | Verdict | Note |
|---|---|---|
| A1 | **repairable, not pass-as-is** | The real empirical result (§4) is a **population-level statistical regularity** (a fitted Shepard's-law curve with specific regression coefficients), not a uniquely defined per-instance ground truth. Grading against "what humans on average would say" would require the oracle to know a population's aggregate behavior — a different but related failure to the solver-mind-reading case A1 warns about. The repair is to *stipulate* the kernel as part of theta (sampled, known, arbitrary) rather than trying to match the empirical human curve, exactly as done in §6.1 for coverage. |
| A2–A5 | PASS given the repair | — |

**Verdict: PASS/repairable**, repair is essential and stated plainly rather
than glossed over.

### 6.4 Self-consistency check against own prior output (motivated by §5) — offered as a family, not evidence for a new primitive

- **Theta**: any existing deterministic derivation rule (reuse any prior
  vein's family, e.g. modular arithmetic).
- **Encoding twist**: with probability tied to `k`, the interface *silently
  corrupts* the solver's own just-computed intermediate value before it is
  fed forward; the solver must, before finalizing, compare its own last
  correctly-computed value against what the interface recorded and flag/fix
  any mismatch.
- **f**: the oracle always knows the true derivation and the corruption log
  (both generation-time facts) — never needs to inspect the solver's actual
  internal state, so this is A1-admissible by construction.
- **Verdict**: **PASS**, but explicitly **not evidence for a new
  primitive** — see §7, this reduces to `hypothesis-elimination` /
  `belief-state-reset` applied to a self-generated observation instead of an
  externally-supplied one, which is a source-of-evidence distinction, not an
  operational one (the same reasoning vein 2.7 used for belief-state drift).

### 6.5 Goal-management families — explicit non-family, folds into vein 2.7's 6.1

Prospective memory / goal-shielding / interruption-resumption all share the
shape "maintain or correctly re-trigger a commitment under intervening
distraction," which is the same shape as vein 2.7's
`constraint-persistence-under-distraction` family (6.1 in that report),
imported here from a different literature as independent corroboration
rather than proposed as a new family type. Not triaged separately to avoid
manufacturing a distinct family where none is needed.

---

## 7. Primitives

### 7.1 The sharpest fold question, argued both ways

**Claim under test**: with a computable posterior, "do I know enough?" is
just `posterior-enumeration` plus a threshold comparison — not a new
operation.

**The case FOR folding (no new primitive needed)**: A threshold comparison
on an already-enumerated posterior is, mechanically, nothing more than
`posterior-enumeration` followed by an argmax-adjacent comparison — the same
kind of composition already implicit in `majority-predict` (which is
explicitly *not* folded into `posterior-enumeration` only because of a
standing, undecided prediction about L1/L3 unification — see that entry's
own falsifier). If stopping is just "compare the posterior's concentration
to a number," it is arguably a derived quantity, not a primitive.

**The case AGAINST folding (a new primitive is needed)**: every piece of
empirical evidence gathered in this vein about how stopping/monitoring is
*actually* computed says it is **not** read off a full enumerated posterior.
Three independent literatures converge on a **cheap-proxy threshold**
instead of a true posterior read-out:

1. Koriat's accessibility model (§2): FOK is computed from the *quantity of
   retrieved partial cues*, not from the target itself — structurally
   parasitic on the retrieval attempt, not on an enumerated hypothesis set.
2. The linear threshold model of optimal stopping (§3, VERIFIED body): the
   best-fitting human model is a **3-parameter position-linear threshold**,
   explicitly contrasted against (and losing 6% of value relative to) the
   true Bayes-optimal policy, which *would* require the full state space.
   Humans use the cheap proxy, not the correct enumeration.
3. Epistemic vigilance (§1.4, LEAD): scrutiny depth is described as a
   graded setting driven by source-trust cues and surface plausibility, not
   full evaluation of the claim's actual truth-value.

This is the **same shape of argument** the register already accepted once:
`basis-probe` and `structure-walk-query` are kept **separate** from
`informative-query-selection` specifically because they *don't touch the
belief state* and so scale differently (with the rule's size or a fixed
schedule, not with `|belief state|`). A stopping rule driven by a cheap
proxy signal has exactly that property relative to a stopping rule computed
from the true posterior. And unlike a query-selection primitive, a stopping
decision produces a **halt-vs-continue** control signal over the query loop
itself, possibly with a null ("insufficient evidence") answer attached — a
type of output no existing entry produces (`majority-predict` and
`posterior-enumeration` produce answers; the three query primitives produce
the next query).

**Resolution**: propose the primitive, with the fold argument recorded as a
live hypothesis rather than settled, exactly as the register's falsifier
convention asks.

### 7.2 New primitive proposed

```
[termination-gate]
gloss = "Compare a monitoring signal against a threshold to decide whether
to continue querying/searching or to halt and commit to an answer -- 
including the null answer 'insufficient evidence, do not answer.' The
monitoring signal may be the belief state itself, but in every measured
instance found in this vein it is instead a CHEAP PROXY for it (retrieval
accessibility, sequence position, source-trust) that does not require
touching the full belief state."
kind = "operation"
first_seen = "vein-2.8 (proposed, not yet assigned a register row)"
aliases = [
    "feeling-of-knowing-gated retrieval termination (Koriat 1993)",
    "give-up time / patch-leaving threshold (Pirolli & Card, Marginal Value Theorem)",
    "linear threshold model of optimal stopping (secretary-problem framing)",
    "epistemic-vigilance scrutiny-depth setting (Sperber et al. 2010)",
    "judgment-of-learning-gated restudy allocation (Nelson & Dunlosky)",
]
falsifier = """
Argued at length in vein-2.8 section 7.1. Accepted as new because it
produces a halt/continue control decision over the query loop itself, a
type no existing primitive produces, AND because every measured instance of
it found in this vein is driven by a proxy signal cheaper than the true
belief state -- mirroring exactly the cost-scaling argument that already
separates basis-probe/structure-walk-query from informative-query-selection.

Fold into posterior-enumeration + comparison if a family is ever built
where the stop signal is provably read directly off the enumerated
posterior with no proxy/heuristic step -- in that case this entry would be
a derived comparison, not an operation, and should be demoted. As of this
vein no such family was found; every source measured a proxy, never a true
posterior read-out.
"""
```

### 7.3 Two candidates considered and explicitly folded (not proposed)

| Candidate | Motivating finding | Why folded, not new |
|---|---|---|
| Convention-extension-by-similarity | §4's Shepard's-law generalization result | Stretches `majority-predict`'s "belief state" to a similarity-weighted neighborhood of precedent, which is a real stretch — flagged as such rather than silently absorbed — but does not produce a new *kind* of output (still a point answer), unlike `termination-gate`. Fold accepted provisionally; split if a family is ever built where this needs to interoperate with a discrete belief state and the two behaviors diverge. |
| Self-detected error correction (PES/ERN, §5) | Self-committed-only slowing effect | Folds into `hypothesis-elimination` / `belief-state-reset`: the operation is "cut/replace belief given a new observation," and the *source* of the observation (self-generated mismatch vs. externally supplied) does not change what the operation computes — the same reasoning vein 2.7 used to keep harness-induced belief drift from becoming a new primitive. |

**Running tally for this vein: +1** (`termination-gate`), consistent with
S2's prior of "at most two."

---

## 8. Rejections

| Candidate paradigm | Why rejected | Repair, if any |
|---|---|---|
| Grading a solver's stated confidence against its *own true* internal retrieval-accessibility state (a literal replica of Koriat's FOK mechanism-matching, rather than outcome-calibration) | **A1 fails**: requires the oracle to inspect the solver's actual internal process, not a fact about theta | Grade calibration against ground-truth *correctness* instead (outcome, not mechanism) — this is what §6.1 does |
| ERN/comparator-model as a literal neural/EEG signal to reproduce | Not a task-family paradigm at all outside a physiological subject — meaningless for a text-based solver | Behavioural manifestation (self-committed-only post-error slowing) is admissible; captured in §6.4 |
| Goal-shielding's reaction-time semantic-priming methodology (measuring accessibility of competing goals via lexical-decision RT) | **A1 fails**: requires reading the solver's internal activation state via a proxy with no analog for a symbolic/text solver graded on correctness | Structural insight (activating a goal suppresses accessibility of competitors) already covered without the RT-measurement method, via §6.5's fold into vein 2.7's distraction-resistance family |
| Delegation decision graded against the solver's *genuine* subjective self-confidence | **A1 fails** unless self-confidence is a stipulated, theta-derived quantity | Operationalize "confidence" as an explicit entropy/count over a computable belief state, not a true subjective report |
| Real epistemic vigilance (susceptibility to an actually-persuasive false claim) | **A1 fails**: requires knowing the solver's real trust disposition/susceptibility, not theta | Stipulate source-trust and claim-plausibility as theta-derived numbers the oracle already knows |
| Convention extension graded against real population-level human generalization curves | Related-but-distinct A1 failure: oracle would need to know an aggregate human population's behavior rather than the solver's mind, but is still an external fact not derivable from theta alone | §6.3's repair: stipulate the kernel as theta |
| Real end-to-end delegation/tool-use benchmarks as admissible families | Same conclusion as vein 2.7 §6.3: requires either live execution or hand-authored ground truth, not O(1) generation | None offered; explicit negative case, consistent with prior vein |

---

## 9. Sources

**VERIFIED** (fetched and read; tier noted per item):

| Source | Tier | Key content used |
|---|---|---|
| Koriat (1993), "How do we know that we know?", *Psychological Review* 100(4) | Abstract (PubMed) | Accessibility model; direct quote on non-direct-access conclusion |
| Hampton (2001), "Rhesus monkeys know when they remember," PNAS 98:5359–5362 | Body/Results (via PMC mirror) | 88/67%, 78/69% chosen-vs-forced accuracy; 85/43% delay-dependent decline rate; N=2 |
| Wikipedia, "Tip of the tongue" | Body (tertiary source) | Brown & McNeill 1966 paradigm, qualitative better-than-chance partial recall; no percentages found |
| "Post-Error Adjustments" review, PMC3173829 | Body | PES magnitudes (59ms/−1ms; 2.8%/1.5%), self-committed-vs-inserted distinction, error rates, conflict-monitoring-theory mechanism |
| Gehring, Goss, Coles, Meyer & Donchin (1993), *Psychological Science* 4(6):385–390 | Abstract (digitalcommons.usf.edu) | Direct quote; accuracy-emphasis modulation of ERN; no numeric ERN amplitude in this fetch |
| Botvinick, Braver, Barch, Carter & Cohen (2001), "Conflict monitoring and cognitive control," *Psychological Review* 108 | Abstract | Conflict-monitoring hypothesis, two linked computational models |
| Stanford Encyclopedia of Philosophy, "Convention" | Body | Lewis's definition (direct quote), arbitrariness, absence of novel-case-extension treatment |
| "Ad hoc conventions generalize to new referents," arXiv 2509.05566 | Abstract + body | N=302, threads design, b=0.03 pre/post shift, BF=15.53, quadratic/linear Shepard's-law fit, nameability-baseline-vs-change split |
| "Semantic uncertainty guides the extension of conventions to new referents," arXiv 2305.06539 | Abstract only (body 404'd) | N=240, nameability construct, KiloGram tangram dataset |
| Pirolli & Card, Information Foraging (UIR technical report PDF) | Body, paraphrase (exact quotes not extractable) | Patch/MVT give-up model, information-scent definition |
| "A linear threshold model for optimal stopping behavior," PMC7293628 | Body | 6% loss vs. optimal, 4.7 vs 5.2 stop position, 28% vs 50% position-9 acceptance, 3-vs-10-parameter model comparison, skew-direction reversal |
| Shah, Friedman & Kruglanski (2002), "Forgetting all else," *JPSP* 83(6) | Abstract (academia.edu) | Six-study goal-shielding paradigm, moderators list, same-purpose vs. facilitating-goal asymmetry |
| Lee & See (2004), "Trust in Automation," *Human Factors* 46(1) | Abstract (body paywalled) | Direct quote on trust-governs-reliance framing |

**LEAD** (WebSearch snippet or secondary synthesis only; not independently
fetched — flagged wherever cited above):

- Nelson & Narens (1990) framework — characterization only, primary chapter and a ScienceDirect abstract page both inaccessible (403)
- Koriat (1995), "Dissociating knowing and the feeling of knowing" — title/existence only, full text corrupted on fetch
- Shimamura & Squire (1986) amnesia FOK dissociation — snippet only, deliberately not fetched (clinical-population caution, see §2)
- ERN amplitude/timing numbers (−5 to −10 μV, ~100ms) — WebSearch synthesis; explicitly NOT present in the one abstract I directly fetched
- Gollwitzer (1999) implementation-intentions mechanism quotes — WebSearch synthesis of the primary paper, not independently fetched
- Gollwitzer & Sheeran (2006) meta-analysis, d=.65/.61 — secondary-source synthesis only
- Altmann & Trafton, resumption lag / "Memory for Goals" — PDF fetch failed (corrupted); characterization from WebSearch synthesis only
- Sperber et al. (2010), "Epistemic Vigilance" — two direct-fetch attempts 403'd; characterization from WebSearch synthesis of definitional passages
- Proofreading "Discrepancy-Attribution hypothesis" / two-stage error-detection model — snippet only
- Karabenick adult help-seeking literature; "Metacognitive control processes in question answering" (2021) — snippets only
- Schelling/Elster precommitment (Ulysses contracts) — snippets only, no primary text fetched
- Monsell (2003) task-switching review — connection failed on the one PDF host tried; no numeric switch-cost figure obtained from any source, and none is reported

---

## 10. Source's taxonomy (quarantined — not adopted as our decomposition)

- **Nelson & Narens' monitoring/control, meta-level/object-level split**, and their named taxonomy of judgment types (ease-of-learning, judgment-of-learning, feeling-of-knowing, retrospective confidence) — LEAD-tier, quarantined regardless.
- **Koriat's accessibility-heuristic taxonomy** (cue-familiarity heuristic vs. accessibility heuristic as two contributing mechanisms to FOK).
- **Information foraging theory's patch/scent/diet vocabulary**, imported from optimal foraging biology.
- **Conflict-monitoring theory's ACC-as-substrate framing** and the general monitoring-vs-control executive-function split invoked throughout this literature — this is precisely the "System 1/System 2, executive function" ontology Hazard 2 warns against; not adopted.
- **Lee & See's trust-in-automation dimensions** (performance, process, purpose) — not independently verified beyond the abstract, quarantined in case fetched later.

None of the above shaped §1's or §7's decomposition. The candidate families
in §6 and the primitive in §7 were built from the *measured, quantified*
findings (Hampton's numbers, the linear-threshold-model's numbers, the
convention-extension regression coefficients, the PES magnitudes), not from
any of the field's own named constructs.

---

## 11. Sealed encounters

- **Sense-of-agency/comparator-model seal, held throughout.** No search in
  this vein used the terms "intentional binding," "efference copy," or
  "sense of agency." Where the general ERN/error-monitoring searches
  surfaced adjacent titles, they were left unopened and are logged here:
  - "Neurocognitive Mechanisms of Error-Based Motor Learning" (PMC3817858) —
    title surfaced in an ERN WebSearch result list; not opened. Motor-learning
    framing plausibly touches predicted-vs-actual sensory consequence
    (efference-copy-adjacent); treated as inside the boundary rather than
    risked.
  - "Machine Learning to Detect Anxiety Disorders from Error-Related
    Negativity and EEG Signals" (arXiv 2410.00028) — surfaced in the same
    search; not opened, doubly out of scope (clinical/disorder framing on
    top of the ERN-neuroscience adjacency).
- **Developmental-seal near-miss, self-caught.** Nelson-Le Gall's
  foundational help-seeking framework (executive vs. instrumental
  help-seeking) originates in research on **children's** classroom
  behavior — developmental/educational psychology on child subjects. An
  ERIC document (ED275741) attributed to Nelson-Le Gall was fetched; the
  fetch failed (corrupted PDF) before any content was read. On reflection
  this source sits on the wrong side of the standing developmental-paradigms
  seal and was not pursued further or cited as load-bearing; §1.4's
  "escalation" claim instead relies on adult-framed sources only (Karabenick,
  the 2021 *Metacognition and Learning* paper), both LEAD-tier.
- **Clinical-population discretionary caution.** Shimamura & Squire (1986),
  an amnesic-patient FOK study, is not on the explicit seal list (it is
  neither anosognosia/insight research nor clinical-reasoning research) but
  sits close enough to the clinical-deficit boundary that it was deliberately
  left at LEAD tier (search-snippet characterization only) rather than
  fetched and read in full, out of caution rather than a hard rule.
- No ARC-AGI-3, SRE/incident-response, SQL, Erlang, construction-geometry,
  ILP, active-learning, clinical-trial, verbal-psychometric, or
  Latin-square/graph-colouring material surfaced in any search this session.
