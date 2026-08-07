"""The prequential training loop. Task Spec section 4 and section 8 step 1.

    "At each step, before training on a batch, record the loss on that batch.
     (Before, not after -- the coding argument requires evaluating on data not
     yet seen.)"

A procedural generator makes that requirement almost free, and it is worth
stating why, because it is a genuine advantage of this whole design over
measuring the same quantity on a corpus: **there is no epoch.**  Every batch is
freshly sampled from an unbounded stream with a seed never used before, so
"data not yet seen" is not an approximation maintained by a held-out split -- it
is a property of the sampler.  The prequential estimate here is the thing itself
rather than a proxy for it.

Numerically the loop is ordinary: AdamW, linear warmup into cosine decay, no
gradient accumulation, no distributed anything.  It has to survive being killed
mid-run on a Kaggle session, so it checkpoints.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from random import Random
from typing import Any, Iterator

import torch
import torch.nn.functional as F

from .. import vocab
from ..form import Level
from .episode import (
    Channel,
    Episode,
    EpisodeSpec,
    QuerySource,
    build_episode,
    spec_for_level,
)
from .metrics import (
    AcquisitionSlope,
    Budget,
    StructuralContent,
    acquisition_slope,
    structural_content,
)
from .model import Inducer, ModelConfig, device_report, pick_device
from .protocol import ProtocolGap, SupportsParse


# --------------------------------------------------------------------------
# Batching
# --------------------------------------------------------------------------


@dataclass
class Batch:
    tokens: torch.Tensor  # (B, L)
    targets: torch.Tensor  # (B, L)
    mask: torch.Tensor  # (B, L) bool -- is this position a supervised target
    trial: torch.Tensor  # (B, L) long -- trial index, -1 outside a trial
    soft: torch.Tensor | None  # (B, L, V) -- exact posterior targets, or None

    def to(self, device: torch.device) -> "Batch":
        return Batch(
            self.tokens.to(device), self.targets.to(device), self.mask.to(device),
            self.trial.to(device), None if self.soft is None else self.soft.to(device),
        )


def collate(episodes: list[Episode], max_len: int) -> Batch:
    """Pad a list of episodes into aligned tensors.

    Section 7: "episodes never split across batch boundaries."  Whole episodes go
    in; the padding is right-side and masked out, so a short episode contributes
    exactly its own supervised tokens and no more.  An episode longer than
    `max_len` raises -- truncating would silently drop the final trials, which are
    the ones carrying the most-identified theta and therefore the lowest loss, so
    the effect would be to *raise* measured loss on exactly the families that
    identify fastest.
    """
    longest = max(len(e) for e in episodes)
    if longest > max_len:
        raise ValueError(
            f"episode of {longest} tokens exceeds max_len {max_len}; raise max_len "
            "or lower T. Truncation is not offered -- it would drop the "
            "best-identified trials and bias the loss upward."
        )
    b, L = len(episodes), longest
    tokens = torch.full((b, L), vocab.PAD, dtype=torch.long)
    targets = torch.full((b, L), vocab.PAD, dtype=torch.long)
    mask = torch.zeros((b, L), dtype=torch.bool)
    trial = torch.full((b, L), -1, dtype=torch.long)

    need_soft = any(e.posterior_targets for e in episodes)
    soft = torch.zeros((b, L, vocab.VOCAB_SIZE)) if need_soft else None

    for i, ep in enumerate(episodes):
        n = len(ep)
        tokens[i, :n] = torch.tensor(ep.tokens, dtype=torch.long)
        targets[i, :n] = torch.tensor(ep.targets, dtype=torch.long)
        mask[i, :n] = torch.tensor(ep.supervised, dtype=torch.bool)
        trial[i, :n] = torch.tensor(ep.trial_index, dtype=torch.long)
        if soft is not None:
            for pos, dist in ep.posterior_targets.items():
                for tok, p in dist.items():
                    soft[i, pos, tok] = p
    return Batch(tokens, targets, mask, trial, soft)


def episode_stream(
    family: Any, k: int, spec: EpisodeSpec, seed0: int = 0, query_fn=None
) -> Iterator[Episode]:
    """An unbounded stream of never-repeated episodes.

    The seed is the episode's identity (section 7), so the stream is a counter
    and any episode in it is reconstructible from its index alone.
    """
    i = seed0
    while True:
        yield build_episode(family, k, i, spec, query_fn=query_fn)
        i += 1


# --------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------


@dataclass
class LossParts:
    total: torch.Tensor  # scalar, mean over supervised tokens
    per_token: torch.Tensor  # (B, L-1)
    mask: torch.Tensor  # (B, L-1)
    trial: torch.Tensor  # (B, L-1)


def masked_loss(logits: torch.Tensor, batch: Batch) -> LossParts:
    """Per-token cross-entropy, normalized per supervised token.

    Handoff section 2.2, restated because it is the single easiest thing to get
    wrong here: normalize per **token**, never per episode.  Encodings differing
    ~85% in length mean a per-episode average carries an encoding effect into
    every structural-content number downstream, and nothing in the resulting
    curve would look unusual.

    Where an exact posterior target exists the token is scored against the
    distribution; elsewhere against the realized token.  Both are cross-entropy
    in nats over the same vocabulary, so they are on one scale -- but they do not
    have the same floor (the posterior target's floor is 0, the realized target's
    is H(y|context)), which is why `TargetMode` is recorded in every run and the
    sweep does not mix them.
    """
    tgt = batch.targets[:, 1:]
    m = batch.mask[:, 1:]
    logp = F.log_softmax(logits.float(), dim=-1)

    hard = -logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    if batch.soft is None:
        per_token = hard
    else:
        s = batch.soft[:, 1:]
        has_soft = s.sum(-1) > 0
        soft_loss = -(s * logp).sum(-1)
        per_token = torch.where(has_soft, soft_loss, hard)

    denom = m.sum()
    if denom == 0:
        raise ValueError("batch has no supervised tokens; the loss mask is wrong")
    total = (per_token * m).sum() / denom
    return LossParts(total, per_token, m, batch.trial[:, 1:])


def per_trial_means(parts: LossParts, T: int) -> list[float]:
    """Loss by trial index -- the input to the acquisition slope (section 4)."""
    out: list[float] = []
    per_token = parts.per_token.detach()
    for t in range(T):
        sel = parts.mask & (parts.trial == t)
        n = sel.sum()
        out.append(float((per_token * sel).sum() / n) if n > 0 else float("nan"))
    return out


# --------------------------------------------------------------------------
# L2: the model chooses the query
# --------------------------------------------------------------------------


def model_query_fn(model: Inducer, family: Any, k: int, device, rng: Random,
                   temperature: float = 1.0):
    """Build the L2 query channel: sample the model's query, parse it, hand it over.

    **This is the finding L2 forces, and it is architectural.**  Section 2 says
    L2 "stays inside the interface... One token stream. No environment, no
    reward, no rollout to terminal."  All true, and none of it makes L2
    constructible offline: section 2.1 also says "the oracle answers the query the
    model actually asked", and what the model would ask is only knowable by
    *sampling from the model*.  So an L2 episode cannot be built into a static
    dataset the way L0, L1 and L3 can -- it needs T sequential generation steps
    inside the training loop, at the current weights.

    That is not RL: the loss stays local, per token, with no credit assigned
    across trials, exactly as section 2.1 designs it.  But it does mean L2 costs
    roughly (query length x T) forward passes per episode on top of the training
    step, and that the data distribution is on-policy and therefore moves as the
    model does.  Both are consequences worth knowing before budgeting a run.

    Uncached deliberately: with T=8 and two-token queries this is 16 short
    forwards, KV caching would be the optimization to reach for if L2 ever sits
    on a critical path, and it does not sit on the sweep's.
    """
    if not isinstance(family, SupportsParse):
        raise ProtocolGap(
            f"{type(family).__name__} has no parse_query, so the harness cannot "
            "know what the model asked and cannot answer what was asked "
            "(Task Spec section 2.1). See docs/10-harness-findings.md."
        )

    probe_rng = Random(rng.getrandbits(32))
    theta_probe = family.sample_theta(k, probe_rng)
    width_cache: dict[int, int] = {}

    def query_width(encoding: Any) -> int:
        """How many tokens a query occupies under this encoding.

        Constant per (family, encoding, k) for every implemented family, but
        asserted rather than assumed: a variable-length query would misalign the
        query channel against q*, and a misaligned channel trains the model to
        emit q*'s second token in the first position -- which looks like slow
        learning, not like a bug.
        """
        key = id(encoding)
        if key not in width_cache:
            widths = {
                len(family.render(encoding, family.sample_query(theta_probe, [], probe_rng)))
                for _ in range(8)
            }
            if len(widths) != 1:
                raise ProtocolGap(
                    f"{type(family).__name__} renders queries at varying widths "
                    f"{widths}; the L2 query channel needs a fixed width to align "
                    "against q*."
                )
            width_cache[key] = widths.pop()
        return width_cache[key]

    def fn(tokens: list[int], history: list, encoding: Any):
        width = query_width(encoding)
        emitted: list[int] = []
        ctx = list(tokens) + [vocab.ASK]
        model.eval()
        with torch.no_grad():
            for _ in range(width):
                x = torch.tensor([ctx[-model.cfg.max_len:]], dtype=torch.long, device=device)
                logits = model(x)[0, -1].float()
                probs = torch.softmax(logits / max(1e-6, temperature), dim=-1)
                tok = int(torch.multinomial(probs, 1, generator=None).item())
                emitted.append(tok)
                ctx.append(tok)
        model.train()
        return family.parse_query(encoding, emitted), emitted

    return fn


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


@dataclass
class RunRecord:
    family: str
    k: int
    spec: dict
    budget: Budget
    losses: list[float] = field(default_factory=list)
    supervised_tokens: list[int] = field(default_factory=list)
    per_trial_history: list[list[float]] = field(default_factory=list)
    malformed_rate: list[float] = field(default_factory=list)
    device: dict = field(default_factory=dict)
    seconds: float = 0.0
    bayes_floor: float | None = None  # lower end: H(y | context, encoding)
    bayes_floor_upper: float | None = None
    label: str = ""

    def content(self) -> StructuralContent:
        return structural_content(
            self.losses, self.supervised_tokens,
            bayes_floor=self.bayes_floor, bayes_floor_upper=self.bayes_floor_upper,
        )

    def slope(self, window: int = 20) -> AcquisitionSlope:
        """Acquisition slope over the last `window` steps.

        Late in the run rather than averaged over it: the quantity section 9 wants
        is how fast the *trained* model identifies theta within an episode, and
        averaging over training mixes that with how bad it was at the start.
        """
        tail = self.per_trial_history[-window:]
        if not tail:
            return acquisition_slope([])
        T = len(tail[0])
        cols = []
        for t in range(T):
            vals = [row[t] for row in tail if row[t] == row[t]]  # drop NaN
            cols.append(sum(vals) / len(vals) if vals else float("nan"))
        return acquisition_slope([c for c in cols if c == c])

    def to_json(self) -> dict:
        d = asdict(self)
        d["budget"] = asdict(self.budget)
        d["budget_fingerprint"] = self.budget.fingerprint
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=1))


def train_run(
    family: Any,
    k: int,
    spec: EpisodeSpec,
    budget: Budget,
    model_cfg: ModelConfig | None = None,
    seed: int = 0,
    device: str = "auto",
    init_state: dict | None = None,
    bayes_floor: float | None = None,
    bayes_floor_upper: float | None = None,
    label: str = "",
    log_every: int = 0,
) -> tuple[RunRecord, Inducer]:
    """One prequential run at a fixed budget. The unit of every measurement here.

    `init_state` is what makes section 4's *conditional* structural content
    possible: pass the state dict of a model already trained on family i and the
    returned record is S(T_j | m_i) rather than S(T_j).  Same function, same
    budget, one argument different -- which is what keeps the matrix's diagonal
    and off-diagonal comparable.
    """
    torch.manual_seed(seed)
    dev = pick_device(device)
    cfg = model_cfg or ModelConfig(max_len=budget.max_len)
    if cfg.max_len != budget.max_len:
        raise ValueError("model max_len and budget max_len must agree")

    model = Inducer(cfg).to(dev)
    if init_state is not None:
        model.load_state_dict(init_state)

    query_fn = None
    if spec.query_source is QuerySource.MODEL:
        query_fn = model_query_fn(model, family, k, dev, Random(seed))

    stream = episode_stream(family, k, spec, seed0=seed * 1_000_003, query_fn=query_fn)
    opt = torch.optim.AdamW(model.parameters(), lr=budget.lr, betas=(0.9, 0.95),
                            weight_decay=0.1)

    rec = RunRecord(
        family=family.name, k=k, spec=asdict(spec), budget=budget,
        device=device_report(), bayes_floor=bayes_floor,
        bayes_floor_upper=bayes_floor_upper, label=label,
    )
    rec.spec["query_source"] = spec.query_source.value
    rec.spec["target_mode"] = spec.target_mode.value

    t0 = time.time()
    model.train()
    for step in range(budget.steps):
        episodes = [next(stream) for _ in range(budget.batch_size)]
        batch = collate(episodes, budget.max_len).to(dev)

        for g in opt.param_groups:
            g["lr"] = _lr_at(step, budget)

        logits = model(batch.tokens[:, :-1])
        parts = masked_loss(logits, batch)

        # Recorded BEFORE the update: section 4 is explicit, and this is the only
        # place in the harness where the order is load-bearing.
        rec.losses.append(float(parts.total.detach()))
        rec.supervised_tokens.append(int(parts.mask.sum()))
        rec.per_trial_history.append(per_trial_means(parts, spec.total_trials))
        rec.malformed_rate.append(
            sum(e.malformed_queries for e in episodes) / (len(episodes) * spec.T)
        )

        opt.zero_grad(set_to_none=True)
        parts.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if log_every and (step % log_every == 0 or step == budget.steps - 1):
            print(f"  step {step:>5}/{budget.steps}  loss {rec.losses[-1]:.4f}", flush=True)

    rec.seconds = time.time() - t0
    return rec, model


def _lr_at(step: int, budget: Budget) -> float:
    import math

    if step < budget.warmup:
        return budget.lr * (step + 1) / max(1, budget.warmup)
    p = (step - budget.warmup) / max(1, budget.steps - budget.warmup)
    return budget.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def round_trip_all_levels(family: Any, k: int = 1, T: int = 6, seed: int = 0) -> dict:
    """Section 8 step 1's gate, minus the training: does an episode build at L0-L3?

    Returns a per-level summary so a failure names the level rather than the
    first one tried.  Used by the harness tests and by `python -m
    repertoire.harness` as a smoke check on any family.
    """
    out: dict[str, Any] = {}
    for level in (Level.L0, Level.L1, Level.L2, Level.L3):
        spec = spec_for_level(level, T=T)
        try:
            if level is Level.L2:
                cfg = ModelConfig(n_layer=1, d_model=32, d_ff=64, n_head=2, max_len=512)
                m = Inducer(cfg)
                qf = model_query_fn(m, family, k, torch.device("cpu"), Random(seed))
                ep = build_episode(family, k, seed, spec, query_fn=qf)
            else:
                ep = build_episode(family, k, seed, spec)
            out[level.value] = {
                "tokens": len(ep),
                "supervised": ep.n_supervised,
                "channels": {c.name: ep.channel.count(c) for c in Channel if c in ep.channel},
                "malformed": ep.malformed_queries,
            }
        except Exception as exc:  # recorded, not raised: a level failing IS a result
            out[level.value] = {"error": f"{type(exc).__name__}: {exc}"}
    return out
