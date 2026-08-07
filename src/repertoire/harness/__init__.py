"""The harness. Task Spec section 8 steps 1, 2 and 5.

Owned by the harness session; see `docs/09-harness-handoff.md` for the brief and
`docs/10-harness-findings.md` for every protocol decision made here.

    protocol.py   what the harness needs beyond section 7, and why
    episode.py    episode construction, the level wrappers, the loss mask
    stub.py       the step 1 stub family (NOT a register candidate)
    entropy.py    exact posteriors, Bayes floors, measured residual entropy
    model.py      a small decoder-only inducer (section 0)
    train.py      the prequential loop (section 4)
    metrics.py    structural content and acquisition slope
    sweep.py      section 8 step 5, the dial sweep

Import cost: `episode`, `protocol`, `entropy` and `metrics` are torch-free, so
the register tooling and most tests run without it.  `model`, `train` and `sweep`
need torch and are not imported here.
"""

from .episode import (
    Channel,
    Episode,
    EpisodeSpec,
    QuerySource,
    TargetMode,
    build_episode,
    episode_seed,
    spec_for_level,
)
from .metrics import Budget, acquisition_slope, structural_content
from .protocol import ProtocolGap, Reveal, answer_distribution, check_query_sensitivity

__all__ = [
    "Channel",
    "Episode",
    "EpisodeSpec",
    "QuerySource",
    "TargetMode",
    "build_episode",
    "episode_seed",
    "spec_for_level",
    "Budget",
    "acquisition_slope",
    "structural_content",
    "ProtocolGap",
    "Reveal",
    "answer_distribution",
    "check_query_sensitivity",
]
