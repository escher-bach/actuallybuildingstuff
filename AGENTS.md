# Project Agent Instructions

## Standardize everything outside the novelty layer

The user does not want hand-rolled implementations of standard functionality.
Recreating a maintained solution is extra correctness, integration, and
maintenance burden, not valuable project work.

- Before implementing infrastructure or a common capability, identify the
  established library, framework feature, protocol, artifact format, or tool
  that already owns it. Use that solution by default and pin compatible
  versions where reproducibility matters.
- Keep project-owned code focused on the novelty layer: the scientific idea,
  domain semantics, experimental protocol, world/teacher/verifier behavior,
  data provenance, evaluation, and the narrow adapters needed to connect those
  pieces to standard systems.
- Prefer thin adapters around maintained APIs over forks, parallel stacks, or
  local reimplementations. Preserve standard artifact and interoperability
  paths end to end.
- Custom implementations are justified only when they directly express project
  novelty or when an explicit requirement cannot be met by a suitable
  maintained solution. In that case, state the unmet requirement, alternatives
  considered, and the smallest custom surface before building it.
- When existing project code duplicates a standard maintained solution, prefer
  migration and deletion over extending the duplicate.

For the current model/training migration, treat
`STANDARD-LLM-STACK-MIGRATION-PLAN.md` as the authoritative ownership boundary.
Changes that move a library-owned concern back into project code require an
explicit user decision amending that plan.

## Semantic and abstraction-level reporting

Lead every research or engineering report with the broader structure it bears
on: the worlds and processes being designed, the capabilities they are meant to
develop, the learner information available, and the downstream transfer claim.
Never present a notebook metric, threshold, benchmark, or failure as
self-interpreting. State the theoretical object it operationalizes, the claim
or boundary it tests, and the project-level decision it changes; if it changes
none, say so explicitly. Distinguish apparatus failures (execution, plumbing,
measurement, or reproducibility) from world-validity and scientific failures
(invalid semantics, information boundaries, targets, controls, or
interpretations). Treat supervised fine-tuning and reinforcement learning as
downstream word/token-based interactive developmental processes inside this
same framework, not as detached technical stages. Implementation, throughput,
hardware, and notebook details are subordinate: surface them when they
constrain semantic design, learner experience, measurement validity, or the
feasibility of a project-level choice, and otherwise keep them in supporting
evidence.

## Kaggle experiment execution

Treat `EXPERIMENT-EXECUTION-PLAN.md` as the sole authoritative workflow for
launching, monitoring, retaining, retrieving, and auditing Kaggle experiments.

- Do not ask the user to upload notebooks or download complete notebook
  outputs manually when the repository orchestration path can do the work.
- Do not launch a GPU run without explicit user authorization.
- Keep source clones, dependency caches, and build products outside
  `/kaggle/working`; that directory is reserved for declared result artifacts.
- Keep heavyweight checkpoints and recovery payloads on Kaggle. Attach them to
  downstream runs through Kaggle sources rather than routing them through the
  user's device.
- Download only compact audit and analysis artifacts unless the user explicitly
  requests a checkpoint or recovery payload.
- Before relying on a result, verify its tracked audit receipt as specified by
  `EXPERIMENT-EXECUTION-PLAN.md`.
