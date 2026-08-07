# Vein 2.7 — Agentic Harness Engineering and Reasoning-Model Scaffolds

Scope: harness architecture (orchestration layer around an LLM agent) is in scope.
HELD OUT and not fetched: (a) production incident response / SRE / on-call / distributed-systems
debugging as an application domain, (b) ARC-AGI-3 design material, plus the standing partition
(SQL/relational query, Erlang/OTP, construction geometry, epsilon-delta analysis, ILP, pool-based
active learning, clinical-trial design, clinical reasoning, animal cognition, developmental
rule-learning, verbal-aptitude psychometrics, Latin-square/graph-colouring). See §10.

Reading key: **VERIFIED** = fetched and read (abstract/body/README stated per item). **LEAD** =
seen only in a WebSearch snippet or a secondary write-up; not independently fetched; treat the
claim as unconfirmed. Every vendor/blog performance claim is flagged inline, not just in §8.

---

## 1. Convergent primitive set

The field's own division of labor, stated most crisply in the one source that formalizes it
(Wang et al., "Harnesses for Inference-Time Alignment over Execution Trajectories," arXiv
2605.21516, VERIFIED body): **"the harness decides what to work on next and which trajectories
to favor while working on it, while the agent decides how to take each step."** Everything below
is organized against that split, with independent corroboration noted where I could assess it
(Hazard 1).

| Operation | Harness or model | Evidence | Independence |
|---|---|---|---|
| Outer tool-dispatch loop | **Harness**, by definition | AI-Harnesses README (VERIFIED): a harness "manages the tool-dispatch loop, permissions, context, and lifecycle of an agent" — this is the definition, not a finding | N/A — definitional |
| Context compaction / summarization | **Harness** | Anthropic context-engineering post (VERIFIED): built-in, preserves "architectural decisions, unresolved bugs, and implementation details," discards "redundant tool outputs." Pi (VERIFIED, pi.dev): ships compaction as a *built-in* default even though it deliberately omits subagents, plan mode, and permission popups as non-core. AHE (2604.25850, VERIFIED): "long-term memory" is one of 7 editable harness component types | Plausibly real, not copied: three independently-built systems (Anthropic's own harness, Factory's, OpenAI's — see §4) all hit the same fixed-context-window constraint and converged on lossy summarization as the answer; Pi's minimal-by-default author still kept it in the core set while cutting other features |
| Task decomposition (goal → ordered subgoals) | **Harness** (per 2605.21516's explicit formalization) | "outer timescale... structures a task into sub-goals"; harness "requests latent progress" and "grants the agent at most M_t low-level steps" per stage (VERIFIED body) | Single formal source; convergence with the informal "Planning & Task Decomposition" category in ai-boost/awesome-harness-engineering (VERIFIED README) is weaker evidence — that's a curated list, not an independent measurement |
| Guided execution (reweighting the model's local action distribution) | **Harness acting on the model**, not a model-performed operation | 2605.21516 (VERIFIED body): guidance is a multiplicative reweighting `W_{t,λ}` of the base sampling distribution; the paper never specifies where the reweighting signal comes from (teacher policy vs. heuristic vs. self-critique) — this is a gap in the source, not filled by me | Single source |
| Retry after tool failure, within a bounded budget | **Harness meters it; model chooses what to retry** | 2605.21516 (VERIFIED): harness grants "at most M_t low-level steps" per stage. ToolMisuseBench (2604.01508, VERIFIED): "explicit step, call, and retry budgets" as the benchmark's own control variable | Two independent formalizations (different author groups) converge on "harness owns the retry budget as a hard resource limit" |
| Subagent delegation / context isolation | **Harness** (mechanism); model decides *when* to delegate if given the tool | Anthropic (VERIFIED): "specialized sub-agents can handle focused tasks with clean context windows," returning summaries "often 1,000-2,000 tokens" to the parent. AHE (VERIFIED): "sub-agent configuration" is one of 7 editable components | Weak-to-moderate: Pi explicitly does **not** ship this as core ("Spawn Pi instances via tmux, or build your own" — VERIFIED, pi.dev) — direct counter-evidence that this is a design choice, not a forced convergence |
| Permission / authorization gating | **Harness**, but often implemented as soft in-context text rather than a hard gate | ai-boost/awesome-harness-engineering has a dedicated "Permissions & Authorization" category (VERIFIED). Pi again omits this as core ("No permission popups: run in a container, or build your own" — VERIFIED). Governance Decay (2606.22528, VERIFIED abstract) shows that when the "gate" is just in-context policy text, it is not durable — see §4 | Same Pi counter-evidence as above; also see the governance-decay finding that even where this exists, it is fragile infrastructure, not a robust joint |
| Long-term memory read/write | **Harness owns the channel; model authors content** | AHE ablation (VERIFIED, cross-checked against Table 3 twice): memory-only ablation is the single largest lever, +5.6pp of the full system's +7.3pp gain (75.3% vs. 69.7% baseline vs. 77.0% full) — memory outweighs tools (+3.3pp), middleware (+2.2pp), and the system prompt, which is *negative* (−2.3pp, 67.4%) | Single source, but the number is a controlled ablation, not a described intuition — stronger evidence type per Hazard 1 |
| Decision self-declaration + later outcome verification | **Split**: harness supplies the falsifiability structure; model supplies the prediction content | AHE's "decision observability" pillar (VERIFIED): "pairs every edit with a self-declared prediction, later verified against the next round's task-level outcomes" | Single source |
| Skill selection, binding to authority/context, evidence capture | **Harness**, with "interpreted by a stochastic agent" as the one explicit model-entry-point | Harnessing Agent Skills (2606.20631, VERIFIED abstract): skill-in-use "arises... when the artefact is selected for a run, bound to context and authority constraints, interpreted by a stochastic agent, and recorded as run evidence" | Cross-instantiated across 8 systems per the paper's own claim (VERIFIED abstract level only — the 8 systems and the 10 patterns themselves were not enumerated in what I fetched; LEAD for specifics) |
| Belief-state drift under harness intervention | **Caused by the harness, experienced by the model as its own (uninspected) belief state** | Harness-Induced Belief Divergence (2607.04528, VERIFIED abstract): "blocked actions, compressed repairs, selective verification, and cost-aware evidence pruning" change an agent's beliefs about progress/risk/success "even when the task, environment, and base LLM are fixed," and these interventions "often preserve terminal success while changing the beliefs that drive later decisions" | Single source; the finding itself is about a harness *causing* a model-side phenomenon (belief tracking) to go wrong silently — see §4 |

**Structural observation, not itself a table row:** most of what this vein turned up (dispatch
loop, compaction, retry-budget metering, permission gating, guided execution) is infrastructure
performed *on* the model's input/output stream, not computation the model performs. That is
expected and is itself the finding: a primitive vocabulary is about what a model computes, and a
harness by definition exists to do things the model does not compute for itself. See §2.

---

## 2. Fold or new

**No new primitives.** Everything found with a genuine model-side computational counterpart folds
into the existing vocabulary. The rest (the majority of §1's rows) has no primitive correspondence
at all, for the structural reason stated above — not because the search came up empty, but because
those operations are constitutively harness-side.

| Existing primitive | What in this vein folds into it |
|---|---|
| `backtrack-splice` | Retry-after-tool-failure (abandon the failed call, resume from last good state, within the harness-metered budget — 2605.21516, ToolMisuseBench). Direct cross-domain instantiation of the same operation defined in vein 2.6 for derivation branches. |
| `epistemic-status-tagging` | AHE's decision-observability pillar: "self-declared prediction, later verified against... outcomes" is a real deployed instance of tagging a step as a falsifiable, resolvable claim. Also the deployed mitigation for governance decay — "Constraint Pinning" (2606.22528, VERIFIED abstract) — is functionally "tag this span as protected/must-persist," i.e., a special epistemic-status value that survives lossy transformation. |
| `belief-state-maintenance` / `belief-state-reset` | Harness-Induced Belief Divergence's whole subject: the model's own tracked belief (progress/risk/success) is what gets corrupted by compaction/pruning. The corruption is *silent* — no contradiction signal triggers a reset, which is a genuinely different stressor than what `belief-state-reset` currently models (reset triggered by *evidence*), but it's a stressor on the same object, not a new object. Filed as failure-mode elaboration (§4), not a new primitive. |
| `informative-query-selection` | The nearest fit for "should I take this action / ask this now," but see §3 — real systems do **not** implement the scoring/teacher-policy half of this primitive; they implement only the object it would score (a candidate query/action) and grade it by downstream outcome. Also covers a cost-weighted variant considered and rejected as new — see §7. |
| `structure-walk-query` | Task decomposition of a hierarchical/compositional theta into ordered subgoals, when the hierarchy is itself part of theta (not discovered by search) — walking a known structure to emit the next sub-goal is the same operation already in the roster. The granularity-mismatch failure mode from 2605.21516 (over-decomposition) sharpens *how* to schedule this walk against k, but doesn't change what the operation is. |

Running tally for this vein: **+0**. Argued explicitly, not just reported, per the brief's own
framing that a genuine zero from the first contemporary-practice vein would be significant: the
zero here is not "we didn't look hard enough," it's "the operations that are model-side are
already in the vocabulary, and the operations that are new to this domain are not model-side by
construction." That is itself the headline finding of this vein.

---

## 3. The L2 correspondence

**Structurally: yes, partially. As a supervised teacher policy: no evidence found.**

- **The shape matches.** A malformed or badly-scoped tool call gets back exactly what was asked
  for — an error, an empty result, a schema-validation failure — not a helpful inference of intent.
  ToolMisuseBench (2604.01508, VERIFIED) formalizes exactly this: fault categories (schema drift,
  timeout, authorization failure, rate limit, adversarial error rewriting) over 6,800 tasks, with
  recovery measured *after* the uninformative/failed response. This is a real-world instance of
  "the oracle answers what was actually asked, and a bad query yields a genuinely uninformative
  answer."
- **The supervision does not.** Nowhere in the fetched sources is there a computable `q*(theta,
  history) → x` that scores or prescribes the *correct* next query/action independent of running
  the trajectory to completion:
  - Recovery in ToolMisuseBench is graded against **final task completion within budget**, not
    against a prescribed correct next action (VERIFIED, stated explicitly in the fetched text).
  - "Guided execution" in 2605.21516 reweights the action distribution via an unspecified `ψ`
    (VERIFIED: the paper "never specifies the source of ψ" in what I could extract) — the paper
    stops exactly at the point where a teacher policy would need to be defined.
  - Ask Early, Ask Late, Ask Right (2605.07937, VERIFIED abstract, full text): clarification
    timing is evaluated purely by **downstream pass@3** via a "forced-injection framework," not by
    grading whether a given clarifying question was itself correct. Its own empirical finding
    underlines the gap: across "300 unscripted sessions," **no current frontier model asks within
    the empirically optimal window** — strategies range from over-asking (52% of sessions) to
    never asking — i.e., practice has not even converged on *when* to query well, let alone built
    a supervised oracle for *what* to query.
  - AHE's decision-observability pillar (§1, §2) is the closest analog to a "graded query," but it
    grades the harness-evolution agent's own predictions post hoc against outcomes — calibration
    of self-reported confidence, not supervision of the query/action itself.

**Conclusion, stated once:** the L2 oracle-answers-what-was-asked mechanic has a genuine real-world
counterpart (malformed tool calls), but the teacher-policy half of L2 — a computable, one-pass
`q*` that says what *should* have been asked — is not something the field has built or validated.
If we build it, we would not be duplicating existing infrastructure; we would be filling a
documented gap.

---

## 4. Documented failure modes

| Failure mode | Source | What was measured (numbers, with location) |
|---|---|---|
| **Governance decay**: compaction silently drops in-context safety/policy constraints | ConstraintRot benchmark, 2606.22528 (VERIFIED, abstract, exact figures) | "Across 1,323 episodes, violation rises from 0% with the policy in full context to 30% after compaction, reaching 59% for some models; when the constraint survives the summary, violation remains 0%, but when it is dropped, violation reaches 38%," across seven model families. An adversarial "Compaction-Eviction Attack" (content engineered to bias the summarizer into omitting a legitimate policy) "defeat[s] every evaluated model." Mitigation ("Constraint Pinning," quarantine the constraint from lossy compaction) "restores violation to 0%" in-benchmark. |
| **Compaction preserves surface obedience, loses task-critical state** | Factory.ai production compression eval (VERIFIED, table read directly) | 36,611 production SWE-agent messages; three compaction methods (Factory's own, Anthropic's, OpenAI's) scored on a 5-point rubric. "Artifact Trail" (which files were created/modified/examined) scored **2.19–2.45/5.0 for all three**, while "Instruction Following" scored **4.92–4.99/5.0** for all three — i.e., compacted agents keep obeying instructions while losing track of what they actually touched. |
| **Silent belief-state corruption that preserves terminal success** | Harness-Induced Belief Divergence, 2607.04528 (VERIFIED abstract) | Harness interventions ("blocked actions, compressed repairs, selective verification, cost-aware evidence pruning") change the model's tracked beliefs about progress/risk/success with task, environment, and base model held fixed, and this "often preserve[s] terminal success" — meaning outcome-only evals would not catch it. No numeric rate given in the fetched abstract; body not fetched (budget). |
| **Over-decomposition**: subgoal grain finer than the agent's controllable-progress window | 2605.21516 (VERIFIED body) | Formal bound: success probability decomposes into an irreducible execution cost plus a "granularity penalty" that only softens with retry budget `M` via `−log M`, not eliminates. Empirically on Terminal-Bench 2 (VERIFIED, Figure 3a per source description): "pass rate first rises and then declines with sub-goal count, peaking at six steps." |
| **Over-pruning**: guidance removes trajectories the agent needs to stay recoverable | 2605.21516 (VERIFIED body) | Defined as excessive restriction of the action space via guidance; formalized alongside over-decomposition in the same bound. No separate isolated number extracted beyond the formal statement. |
| **Hallucinated execution**: guidance concentrates probability on locally-attractive, evidence-violating trajectories | 2605.21516 (VERIFIED body) | Plotly hallucination-rate case study: "Under aligned guidance, hallucination stays near 0.35 regardless of count; under misaligned guidance it rises monotonically from 0.20 to nearly 0.90" as sub-goal count increases (Figure 3b per source). |
| **Regression blindness** in self-evolving harnesses: good at fixing, bad at foreseeing what an edit will break | AHE, 2604.25850 (VERIFIED, Section 4.4.2 per source) | "33.7% fix-precision but only 11.8% regression-precision," producing a non-monotone evolution curve across the ten iterations. |
| **Near-zero recoverability for certain fault classes even with schema-aware handling** | ToolMisuseBench, 2604.01508 (VERIFIED, Table II per source) | Recovery scores: Timeout and Schema-drift recovery reach only ≈0.50 with the best (schema-aware) baseline; **Authorization and Rate-limit recovery are 0.000 for all baselines** tested. |
| **Cascading, hard-to-attribute production regression from three independently-shipped harness/config changes** | Anthropic, "An update on recent Claude Code quality reports" (anthropic.com/engineering/april-23-postmortem), VERIFIED via direct fetch, cross-corroborated by InfoQ and VentureBeat secondary reporting (LEAD-tier corroboration only, not independently re-fetched from Anthropic for the secondary detail) | Three compounding changes over ~six weeks in early 2026: (1) default reasoning effort HIGH→MEDIUM (Mar 4, reverted Apr 7) — degraded planning-before-coding, increased retries, collapsed tool-call depth; (2) a caching bug that was meant to clear stale reasoning-thinking history once per idle session but instead "cleared it on every turn for the rest of the session" (shipped Mar 26, fixed Apr 10) — "made it past multiple human and automated code reviews"; (3) a verbosity-cap addition to the system prompt (Apr 16, reverted Apr 20) causing a measured "3% performance drop" in broader evals. **Note:** this is a distinct document from Anthropic's earlier, unrelated 2025 postmortem about inference-infrastructure bugs — do not conflate the two. |

---

## 5. What the evals measure

**Confirmed: overwhelmingly end-to-end task success. L1–L3-like properties are essentially absent.**

| Benchmark | What it reports | End-to-end or process-level? |
|---|---|---|
| AARRI-Bench / AARR (2606.07462, VERIFIED abstract) | Single success rate — best config (Mini-SWE-Agent + Claude Opus 4.7) "achieves only 68.3% success rate" (abstract, VERIFIED) | End-to-end. Framed around holistic judgment ("field sensitivity, research ethics, nuanced scientific judgment") but scored as pass/fail per task, not as a calibration or inference metric. |
| Terminal-Bench 2 (used by AHE, 2605.21516) | pass@1 | End-to-end |
| SWE-bench-verified (used by AHE) | aggregate success | End-to-end |
| ToolMisuseBench (2604.01508) | success, invalid-call behavior, policy violations, **recovery quality**, budgeted efficiency | Closest to process-level of anything found — but "recovery quality" is graded against eventual task completion within budget, not against a prescribed correct next action (§3). Still outcome-anchored, just finer-grained. |
| ConstraintRot (2606.22528) | constraint-violation rate over a horizon | Process-level in a real sense — measures whether a constraint is *maintained*, not just whether the final answer is right. Closest thing found to testing persistence of a belief/rule under adversarial pressure, which is adjacent in spirit to our belief-state-maintenance, but it is not framed as a posterior over a hidden theta and does not test task-inference under underdetermination. |
| Ask Early, Ask Late, Ask Right (2605.07937) | value-of-clarification curves (pass@3 vs. trajectory position) + real-model over/under-asking rates | Diagnostic of a behavioral pattern (when models ask), not a calibration or posterior metric. No L1-style "identify theta from trials so far" framing. |
| Architectural Design Decisions in AI Agent Harnesses (2604.18071) | descriptive/structural (design-dimension prevalence across 70 systems), not a capability score at all | N/A — not a capability eval |

**None of the fetched evals measure calibration over a hidden rule, or task-inference under
genuine underdetermination, in anything like our L1/L3 sense.** The closest approaches are (a)
ConstraintRot's persistence-under-pressure framing and (b) the clarification-timing literature's
value-of-information curves — both process-level, neither posterior-shaped. This confirms the
brief's expectation rather than refuting it.

---

## 6. Candidate families

Per the brief, most agentic tasks are expected to fail A1 hard. Two families below use the
harness-*research* findings (not real tool execution) as their generative substrate, which keeps
them backward-generable; a third is offered as the explicit negative case.

### 6.1 Constraint-persistence-under-distraction (motivated by Governance Decay/ConstraintRot)

- **Theta**: a hidden constraint/rule `C` (arbitrary symbol-space predicate).
- **P_Theta(·|k)**: samples `C` plus a stream of `k`-controlled distractor content, some of which
  are near-miss "lookalike updates" to `C` (structurally similar but wrong).
- **X / f**: at a point after the distractor stream, ask whether a given action is permitted under
  `C`, or what `C` currently says; `f` evaluates directly against the planted, unmodified `C`.
- **A1 Backward-generable**: **PASS.** No summarizer/solver runs on the generation path — we do
  not simulate lossy compaction, we plant `C` and score against it directly; the "compaction"
  stressor is *encoded as distraction density*, not executed as a real lossy transform. O(1): sample
  `C`, sample distractors, evaluate.
- **A2 Knowledge-free**: PASS if `C` and distractors are symbol-permutable.
- **A3 Encoding-varied**: PASS — vary how `C` is stated (explicit rule vs. embedded in a worked
  example) and how distractors masquerade as updates.
- **A4 Brute-force-resistant**: repairable — tie decoy subtlety/density to `k` so a shortcut
  ("ignore everything after the first mention") stops working as `k` grows.
- **A5**: PASS by construction.
- **Verdict: PASS**, and notably one of the few genuinely admissible families this vein produced —
  because it abstracts the *stressor* (interference) rather than the *mechanism* (a real lossy
  summarizer), it sidesteps the A1 failure that sinks most of the rest of this domain.

### 6.2 Decomposition-granularity matching (motivated by 2605.21516's over-decomposition finding)

- **Theta**: a hierarchical/compositional rule with an explicit tree/DAG structure and a defined
  "natural grain" at each level (this must be *part of* theta, not discovered).
- **Query**: decompose a top-level goal into a sequence of sub-goals; score against whether
  requested progress-per-step falls within theta's own defined controllable-window band.
- **A1**: PASS if the hierarchy is explicit in theta — walking a known structure to emit the
  canonical decomposition at a given grain is `structure-walk-query`, O(1).
- **A4**: fails unless graded specifically against the correct grain band tied to `k` (not "any
  valid decomposition") — repairable, and this is the family's whole point: it operationalizes
  what "good decomposition" means using the paper's own empirical finding (grain matching a
  controllable-progress window) rather than an arbitrary aesthetic.
- **Verdict**: PASS/repairable; folds into `structure-walk-query`, not a new family type, but gives
  a literature-grounded definition of "correct" granularity for any decomposition-flavored family.

### 6.3 Real tool-use / real-environment agentic task — explicit negative

- E.g. Terminal-Bench 2, SWE-bench-verified, AARRI-Bench-style "act as a real researcher" tasks.
- **A1: FAILS, hard, no repair available while remaining "real."** Grading requires either (a)
  executing the task against a live sandboxed environment (a solver/search process on the grading
  path, by construction) or (b) hand-authoring ground truth per instance (not O(1) in difficulty).
  Every real-tool-use benchmark fetched in this vein (Terminal-Bench 2, SWE-bench-verified,
  AARRI-Bench) is graded exactly this way. The only "repair" is to abstract away from real tools
  into a symbolic/schema-only version — which is family 6.1/6.2, a genuine narrowing of scope, not
  a fix for the original task shape.

---

## 7. Rejections

| Candidate | Why rejected |
|---|---|
| "Guided execution / action reweighting" as a new primitive | It's a harness operation performed *on* the model's sampling distribution, not a computation the model performs. No primitive correspondence by construction (§1, §2). |
| "Cost-gated action selection" (retry budgets, irreversibility-weighted choice) as a new primitive distinct from `informative-query-selection` | Same operation (score candidates, pick the best-scoring one) with a different scoring function (cost-weighted vs. information-shrinkage-weighted); folded as a parametrization, not a new structural operation. |
| Adopting real end-to-end agentic benchmarks (Terminal-Bench 2, SWE-bench, AARRI-Bench) as admissible task families | Fail A1 hard; see §6.3. |
| Importing a symptom taxonomy (e.g. the MAST failure-mode categories referenced in passing search results, LEAD-only, not fetched) as our primitive vocabulary | Wrong ontological level even if fetched — a catalogue of *ways trajectories fail* is not a catalogue of *operations a model performs*; would not have passed Hazard 2 regardless of content. |
| The field's own architectural taxonomies (four-layer skill architecture, seven-component AHE breakdown, five-pattern 70-system study, categorical-architecture triple) as our capability decomposition | Explicitly rejected per Hazard 2; quarantined in §9 instead. |

---

## 8. Sources

**VERIFIED** (fetched and read; tier noted):

| Source | Tier fetched | Key content used |
|---|---|---|
| arXiv 2604.25850, "Agentic Harness Engineering..." | Abstract + body (HTML, incl. Table 3 confirmed on a second, targeted fetch) | 7 editable components, ablation numbers, regression blindness |
| arXiv 2605.21516, "Harnesses for Inference-Time Alignment over Execution Trajectories" | Abstract + substantial body (HTML) | Two-mechanism formalization, harness/model split quote, three named failure modes, benchmarks |
| arXiv 2606.20631, "Harnessing Agent Skills..." | Abstract only | Skill-in-use definition, four-layer reference architecture; 10 patterns / 8 systems NOT enumerated (LEAD for those specifics) |
| arXiv 2606.07462, "Act As a Real Researcher..." / AARRI-Bench | Abstract | 68.3% top score, framing |
| pi.dev | Site body | Built-in vs. extension-territory feature split |
| github.com/ai-boost/awesome-harness-engineering | README | Category taxonomy (quarantined, §9) |
| github.com/danielrosehill/AI-Harnesses | README | Harness/framework/backend definitions, 21-project catalogue |
| arXiv 2605.12239, "Harness Engineering as Categorical Architecture" | Abstract | Formal (G, Know, Phi) treatment — quarantined, §9 |
| arXiv 2604.18071, "Architectural Design Decisions in AI Agent Harnesses" | Abstract | 70-system study, 5 design dimensions, 5 patterns (named at high level only) |
| anthropic.com/engineering/april-23-postmortem | Full body, direct fetch; cross-corroborated by InfoQ and VentureBeat news coverage (secondary, not independently re-verified from Anthropic) | Three-bug cascading regression, §4 |
| arXiv 2605.07937, "Ask Early, Ask Late, Ask Right..." | Abstract, full (second fetch got clean text after a corrupted first PDF pass) | Clarification-timing value curves, over/under-asking rates |
| anthropic.com/engineering/effective-context-engineering-for-ai-agents | Full body | Context components, compaction goals, sub-agent token-budget claim (1,000–2,000 tokens), note-taking |
| factory.ai/news/evaluating-compression | Full body incl. table | 36,611-message eval, artifact-trail vs. instruction-following scores |
| arXiv 2606.22528, "Governance Decay..." / ConstraintRot | Abstract (body PDF unreadable in fetch — abstract carried all cited numbers) | 0%→30%→59% violation figures, 1,323 episodes, 7 model families |
| github.com/Piebald-AI/claude-code-system-prompts | README | 27 tool descriptions, per-prompt token counts (used for deployment-distribution section) |
| arXiv 2607.04528, "Measuring Harness-Induced Belief Divergence..." | Abstract | Belief-drift mechanism and causal factors |
| arXiv 2604.01508, "ToolMisuseBench..." | Abstract + body (HTML) | 6,800-task fault taxonomy, Table II recovery numbers |
| arXiv 2605.26297, "Agentic AI Workload Characteristics" | Abstract | Context-reuse/caching finding, read→write temporal structure (deployment-distribution section) |

**LEAD** (WebSearch snippet or secondary source only; not independently fetched, flagged wherever
cited above and in §11):

- BFCL "12% malformed JSON on complex schemas" (from a WebSearch summary of a vendor blog, spheron.network) — **not fetched, treat as an unverified vendor-adjacent claim.**
- OpenEnv "more than half of errors from malformed tool arguments" (huggingface.co blog, WebSearch snippet only)
- TravelBench per-model error-rate figures (WebSearch snippet only)
- MAST taxonomy "FM-1.5 Unaware of Termination Conditions" (WebSearch snippet only)
- GitHub blog, "Multi-Agent Workflows Often Fail..." (title/URL seen, not fetched)
- alies.dev, "Stop Wasting 89% of Your AI Agent's Tokens on CLI Noise" — **fetched and explicitly downgraded**, not a LEAD-by-omission: methodology check (§11) found this is one author's informal analysis using their own unaudited tool, not a rigorous study. Reported only with that caveat attached.
- A DEV Community post claiming "67% of tokens... completely waste" — WebSearch snippet only, same genre as above, not fetched, not to be treated as a figure.
- Various general "context rot," "risk-based confirmation," "irreversible action guardrail" claims from WebSearch snippets across multiple vendor/consultancy blogs (Cloudzy, Latitude, harness-engineering.ai, MindStudio, etc.) — genre-consistent with each other but none independently fetched; reported in §11 as a convergent *pattern description*, explicitly not as a measured finding.

---

## 9. Source's taxonomy (quarantined — not adopted as our decomposition)

- **ai-boost/awesome-harness-engineering's category list** (VERIFIED): Agent Loop; Planning & Task
  Decomposition; Context Delivery & Compaction; Tool Design; Skills & MCP; Permissions &
  Authorization; Memory & State; Task Runners & Orchestration; Verification & CI Integration;
  Observability & Tracing; Debugging & DX; Human-in-the-Loop.
- **Harnessing Agent Skills' four-layer reference architecture** (VERIFIED, abstract): Supply
  Chain, Mediation, Execution Control, Evidence & Feedback.
- **AHE's seven editable component types** (VERIFIED, body): system prompt, tool description, tool
  implementation, middleware, skill, sub-agent configuration, long-term memory.
- **Architectural Design Decisions' five recurring design dimensions and five architectural
  patterns** (VERIFIED, abstract): dimensions = subagent architecture, context management, tool
  systems, safety mechanisms, orchestration; patterns = lightweight tools / balanced CLI
  frameworks / multi-agent orchestrators / enterprise systems / scenario-verticalized projects.
- **Categorical Architecture's formal mapping** (VERIFIED, abstract): Memory ↔ coalgebraic state,
  Skills ↔ operad-composed objects, Protocols ↔ syntactic wiring `G`, full Harness ↔ Architecture
  triple `(G, Know, Phi)`.

None of the above shaped §1's decomposition, per Hazard 2 — §1 is organized around the harness/
model split found in the empirical literature (2605.21516), not around any of these field-native
groupings.

---

## 10. Sealed encounters

- **"A Five-Plane Reference Architecture for Runtime Governance of Production AI Agents"** (arXiv
  2606.12320) — title surfaced in a WebSearch result for irreversible-action/guardrail patterns.
  Not opened: "runtime governance of production agents" sits close enough to the
  incident-response/SRE application-domain seal that I treated it as inside the boundary rather
  than risk it.
- No ARC-AGI-3 material surfaced in any search this session.
- No other titles were withheld; the SRE/on-call/incident-response seal was otherwise not
  approached — none of the fetched sources concerned an agent *performing* incident response,
  on-call work, or distributed-systems debugging as its task domain. (Two adjacent but distinct and
  unopened titles, "The Controllability Trap: A Governance Framework for Military AI Agents" and a
  BenchLM.ai latency blog post, were simply not relevant enough to pursue — not sealed, just unused.)

---

## 11. The deployment input distribution

*(Added mid-review at the coordinator's request: the model we train will itself run inside a
harness, so the harness is the deployment input distribution. Reported as concretely as sources
allow; §11.7 closes with my own judgment, labeled as such.)*

### 11.1 Context composition

No fetched source gives a single canonical "system prompt is X%, tools are Y%, history is Z%"
breakdown — I looked for this specifically and did not find it; **figure not located**. What is
VERIFIED:

- Anthropic's own framing (VERIFIED, effective-context-engineering post): an agent's context is
  built from "system instructions, tools, Model Context Protocol (MCP), external data, message
  history" — all competing for one finite budget. No ratios given in that post.
- Concrete, real per-artifact token counts exist at the individual-prompt level (Piebald-AI's
  extracted Claude Code prompts, VERIFIED README): Claude Code ships **27 builtin tool
  descriptions**; individual catalogued prompts/documents in that same repo range from near-zero
  up to **47,353 tokens** for the single largest entry (a workshop-artifact HTML template — not
  the core system prompt), with e.g. the Explore sub-agent prompt at 871 tokens, Plan mode at 715
  tokens, a security-monitor prompt at 12,085 tokens for its first part alone. These are
  individual document sizes, not a breakdown of what occupies a live session's context at once —
  I want to be explicit that I am not aggregating these into an implied "the system prompt is N
  tokens" claim; the repo catalogues many separate prompts used at different points, not one fixed
  preamble.
- Sub-agent returns: Anthropic states (VERIFIED, direct quote) that sub-agent summaries handed
  back to the parent context are "often 1,000-2,000 tokens" — this is the one closest thing to a
  standing ratio I could verify: whatever a sub-agent's own working context contained, the parent
  sees a compressed proxy at roughly this fixed size, not the underlying trace.
- Structural (not proportional) finding, VERIFIED: arXiv 2605.26297 ("Agentic AI Workload
  Characteristics," abstract) reports that with context caching, "most input tokens are reused
  across turns" (i.e., agent contexts grow by appending, with the prefix re-sent/re-cached each
  turn rather than freshly composed), and that tool use has "a clear temporal structure, with
  agents shifting from read/explore behavior early in execution to execute/write behavior later."
  Both are read from the abstract, not the body.

### 11.2 What arrives that is not clean

This was the coordinator's flagged highest-value number, and **I could not locate a rigorous,
methodologically-disclosed figure for "what fraction of a trajectory's tokens are unhelpful."**
What I found instead:

- **Rigorous but indirect (VERIFIED)**: ToolMisuseBench (2604.01508) puts a number on *how much of
  certain failure classes the agent can recover from*, which is a proxy, not a direct token-noise
  ratio: recovery reaches only ≈0.50 for timeout/schema-drift faults even with the best
  schema-aware baseline, and is **0.000 for authorization and rate-limit faults, for every
  baseline tested**, across 6,800 tasks.
- **Not rigorous, explicitly downgraded (fetched and checked)**: a blog post (alies.dev) claims
  "89.2% of measured input tokens were noise" over "~2,900 CLI commands," but on inspection this
  is one author's own unaudited tool (rtk-ai) with an undisclosed noise-classification method — I
  am reporting the existence of the claim and its methodology gap, not the number as fact.
- **Not fetched at all (LEAD only)**: a "67% of tokens are waste" figure from a different blog
  post, and per-model tool-call error rates from a WebSearch snippet on "TravelBench" (reported
  range 3–7% for one model family vs. 33–49% for another, which — even if accurate — would itself
  demonstrate the ratio is heavily model-dependent, not a fixed property of "a trajectory").

**Honest summary**: the *existence* of substantial non-informative content in real trajectories is
well evidenced (near-zero recoverability for some fault classes; multiple independent
practitioner reports converge on "tool output is often mostly noise"); the *magnitude* is not
something I can respons­ibly state as a number, because every source that offers one is either an
informal single-author measurement or a snippet I did not verify.

### 11.3 Context compaction / summarization

Covered in depth in §4 (Governance Decay, Factory.ai eval) — summarizing the deployment-relevant
shape here:

- Compaction is near-universal even in a deliberately minimal harness (Pi ships it as a default,
  VERIFIED) and is explicitly lossy by design (Anthropic's own stated intent is to keep
  "architectural decisions, unresolved bugs, and implementation details" and discard "redundant
  tool outputs" — VERIFIED quote — i.e., the vendor's own target is selective retention, not full
  fidelity).
- Measured against that intent, real compaction under-delivers specifically on **state the model
  needs to keep acting correctly**, not on general fluency: Factory's 36,611-message eval
  (VERIFIED) shows instruction-following holds up (4.92–4.99/5.0 across all three tested
  summarizers) while artifact-tracking — which files were touched — collapses (2.19–2.45/5.0).
  This is a self-consistent gap: the model still *behaves* compliantly after compaction while no
  longer reliably knowing *what it has already done*.
- Under adversarial pressure the same mechanism is exploitable: ConstraintRot's Compaction-Eviction
  Attack (VERIFIED, §4) defeats every evaluated model at getting a legitimate policy constraint
  omitted from the summary.
- What the model must reconstruct from a lossy summary, per the sources: which files/artifacts
  it touched, any policy/constraint stated earlier in the session, and (per Governance Decay's
  framing) permission/authorization state — none of these are guaranteed to survive, and the
  model is not signaled when they don't.

### 11.4 Output format constraints

- Claude Code alone exposes 27 distinct tool schemas the model must address correctly (VERIFIED,
  Piebald-AI README) — a nontrivial selection problem before any argument-filling problem starts.
- ToolMisuseBench (VERIFIED) formalizes the failure taxonomy for the argument/schema side: schema
  drift, timeout, rate limit, authorization, adversarial error rewriting — and its own abstract
  states plainly that "overall success remains limited under the released authorization and hard
  failure settings," i.e., current tool-calling agents are not close to solved on this axis even
  in a controlled, offline benchmark.
- Beyond that, hard per-model malformed-call rates (the BFCL/OpenEnv/TravelBench figures) are
  LEAD-only in this review — seen in search snippets, not independently fetched, and not to be
  treated as verified.

### 11.5 Interleaving (is query separate from action?)

**Confirmed: the separation does not hold in deployment the way it does in our L2.** In a
harness, issuing a tool call *is* the action — there is no free, side-effect-less channel for
"just asking." Evidence:

- 2605.21516 (VERIFIED): the harness meters retries as a hard resource ("at most M_t low-level
  steps" per stage) — action attempts, including failed/exploratory ones, consume a shared,
  harness-tracked budget, not a separate query allowance.
- Permission/authorization gating exists specifically to distinguish lower-risk (read/inspect) from
  higher-risk (write/execute/irreversible) actions — this is the field's own closest analog to
  "some actions have consequence, some don't," and it is architecture-level (a gating layer), not
  a property of a separate cost-free query channel. **Caveat**: the specific claims about
  risk-tiered confirmation strategies and "verification is most valuable for irreversible actions"
  come from WebSearch snippets across several vendor/consultancy blogs, not from a fetched primary
  study — I'm reporting this as a convergent *pattern description* across secondary sources, not
  as a measured finding, and flagging it as such rather than omitting it, since the coordinator
  asked specifically about this axis.
- No fetched source described a harness with a genuinely free, unlimited, consequence-free
  "ask a clarifying question" action distinct from "take a tool action" — clarification in
  2605.07937 is itself one more action competing for the same trajectory-position budget (its
  entire finding is that clarification value *decays with trajectory position*, which only makes
  sense if asking has an opportunity cost, i.e., is not free).

### 11.6 Long-horizon state

| What | Carried by harness or model | Source |
|---|---|---|
| Cross-session durable rules/identity (CLAUDE.md-style static memory) | **Harness** loads it every session | General WebSearch synthesis of Claude Code architecture write-ups — **LEAD only**, not independently fetched from a primary doc; the underlying mechanism (a loaded static file) is plausible and consistent with Anthropic's own VERIFIED note-taking description, but I did not verify this specific CLAUDE.md/memory-tool split against a primary source myself. |
| Within-session working notes ("structured note-taking... persisted to memory outside of the context window... pulled back into the context window at later times") | **Harness provides the channel; model authors the content** | Anthropic context-engineering post, VERIFIED, direct quote |
| Sub-agent working trace | **Harness discards it**; parent only sees the ~1,000–2,000-token summary | Anthropic, VERIFIED, direct quote |
| Long-term memory as an editable harness component, and its measured value | **Harness** | AHE ablation (VERIFIED): memory-only ablation contributes +5.6pp of the full system's +7.3pp pass@1 gain — the single largest lever measured, larger than tools (+3.3pp) or middleware (+2.2pp), while the system prompt alone is *negative* (−2.3pp) |
| Task-tracking scratchpad (todo-list-style) | **Harness provides the persistence/rendering; model authors content** | WebSearch synthesis of Claude Code's TodoWrite tool — **LEAD only**, not independently fetched from a primary doc |

The one hard, controlled number here (AHE's ablation) is the most important: in a real measured
system, **what the harness carries for the model (memory) mattered more to task success than what
was told to the model in prose (the system prompt, which actively hurt on its own)**. That is
direct evidence that harness-carried state is not a minor convenience — it is, empirically, doing
more work than instruction-following in at least this one measured system.

### 11.7 My judgment (labeled as mine, not sourced)

Three differences, ranked by how much I'd expect each to break transfer from a clean
alternating-query/answer curriculum to a harnessed deployment:

1. **Compaction-induced silent state loss is the biggest risk.** Our episodes are never
   interrupted; a deployed model routinely has its own history rewritten mid-task by a lossy
   process that — per the two hardest numbers in this section (Factory's 2.19–2.45/5.0 artifact
   tracking, ConstraintRot's 30–59% post-compaction violation rate) — specifically loses the kind
   of state our `belief-state-maintenance` primitive is meant to track, *without* a contradiction
   signal to trigger `belief-state-reset`. A model that has only ever practiced tracking belief
   state under contradicting evidence has not practiced noticing "my history was silently
   truncated and I don't know what I lost" — that is a different trigger condition from anything
   currently in the roster, and it's the failure mode with the most independent, numeric
   corroboration in this vein.
2. **The query channel is not free or separable from consequence.** Our L2 treats emitting a query
   as cheap and reversible; in deployment, emitting a query *is* an action, competes for a shared
   retry/step budget (2605.21516), and is sometimes irreversible (motivating the permission-gating
   layer, §11.5). A curriculum that always lets the model "just ask" without modeling that asking
   has a cost and sometimes a consequence never rehearses the joint decision (what to ask, and
   whether asking is warranted given its cost) that a harnessed deployment actually requires.
3. **Harness-carried state is doing real work our clean episodes never require trusting.** The
   AHE ablation (§11.6) shows memory — content the model authors but does not necessarily see in
   full, mediated by the harness's own retrieval/injection choices — outweighs prompt-level
   instruction. Our episodes never require the model to act on a compressed, harness-selected
   proxy for its own past reasoning (a sub-agent summary, a note file) rather than the reasoning
   itself; if a real deployment routes state through such proxies as much as this one measured
   system suggests, a model that has only ever reasoned over its own unmediated context has not
   practiced the "trust and use a compressed proxy of yourself" skill that harnessed deployment
   seems to lean on heavily.

I rank these in this order specifically because (1) and (3) both have hard, sourced numbers behind
them (the Factory/ConstraintRot pair, and the AHE ablation), while (2) is well-argued but rests
more on formal/structural evidence (the retry-budget metering) than on a single measured statistic
— still real, but slightly less load-bearing as stated evidence than the other two.
