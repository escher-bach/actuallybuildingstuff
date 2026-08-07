"""A small decoder-only transformer. Deliberately unremarkable.

Task Spec section 0: "An inducer: sequence -> sequence, trained by per-token
gradient against a target sequence."  That is the whole requirement, and the
model is the least interesting part of this programme -- everything measured
here is a property of the *families*, read through a fixed inducer.  So this is
a plain pre-LN transformer with nothing clever in it.

Two constraints shaped it anyway:

**It has to run on whatever is free.**  Development is CPU-only; the sweep runs
on a Kaggle T4 or P100; the eventual section 9 run is a TPU.  So: no custom
kernels, no flash-attention import, no bf16 assumption (P100 has no bf16 and no
usable fp16 tensor cores), nothing that changes numerics between devices.
`scaled_dot_product_attention` is used because it is in core torch and falls
back cleanly everywhere.

**The vocabulary is 103 tokens.**  That is small enough that the embedding and
the output head are rounding errors, so almost all the compute is in the blocks,
and a model with the same layer count is comparable across families without
worrying about how much of it is a lookup table.  It also means dense soft
targets over the full vocabulary are affordable, which is what makes the L3
posterior target cheap to score.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import vocab


@dataclass(frozen=True)
class ModelConfig:
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 1024
    max_len: int = 512
    dropout: float = 0.0
    vocab_size: int = vocab.VOCAB_SIZE
    tie_weights: bool = True

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def n_params_estimate(self) -> int:
        emb = self.vocab_size * self.d_model + self.max_len * self.d_model
        block = 4 * self.d_model**2 + 2 * self.d_model * self.d_ff
        head = 0 if self.tie_weights else self.vocab_size * self.d_model
        return emb + self.n_layer * block + head


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).split(c, dim=2)
        q = q.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(b, t, self.n_head, self.d_head).transpose(1, 2)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        a = a.transpose(1, 2).contiguous().view(b, t, c)
        x = x + self.drop(self.proj(a))
        h = self.ln2(x)
        x = x + self.drop(self.fc2(F.gelu(self.fc1(h))))
        return x


class Inducer(nn.Module):
    """Section 0's inducer. Nothing here knows what a task family is."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok.weight
        self.apply(self._init)
        for name, p in self.named_parameters():
            # Scaled init on residual projections; standard, and it matters more
            # than usual here because runs are short and a bad start eats a
            # visible slice of a fixed budget.
            if name.endswith("proj.weight") or name.endswith("fc2.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        if t > self.cfg.max_len:
            raise ValueError(f"sequence of {t} exceeds max_len {self.cfg.max_len}")
        pos = torch.arange(t, device=idx.device)
        x = self.drop(self.tok(idx) + self.pos(pos)[None, :, :])
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def n_params(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


def device_report() -> dict:
    """What we are actually running on. Recorded with every run.

    Section 4: the compute budget must be held fixed across any comparison *and
    reported*. Device is part of that -- a T4 run and a P100 run at the same step
    count are the same budget in steps and not in seconds, and only one of those
    is the quantity the measurement is relative to.
    """
    info = {"torch": torch.__version__, "device": "cpu", "name": "cpu", "amp": False}
    if torch.cuda.is_available():
        cap = torch.cuda.get_device_capability()
        info.update(
            device="cuda",
            name=torch.cuda.get_device_name(0),
            capability=f"{cap[0]}.{cap[1]}",
            # fp16 tensor cores land at sm_70+. P100 is sm_60: fp16 there is
            # slower than fp32 and noisier, so autocast stays off rather than
            # being switched on because the device says "cuda".
            amp=cap[0] >= 7,
        )
    return info


def pick_device(prefer: str = "auto") -> torch.device:
    if prefer != "auto":
        return torch.device(prefer)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
