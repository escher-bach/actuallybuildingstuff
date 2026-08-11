"""The fixed 12x384 causal Transformer used by retained Step 1 runs."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope(x: torch.Tensor, base: int) -> torch.Tensor:
    # x: [batch, heads, sequence, head_dim], full head dimension rotates.
    length, dim = x.shape[-2:]
    inv = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device, dtype=torch.float32) / dim))
    angles = torch.outer(torch.arange(length, device=x.device, dtype=torch.float32), inv)
    cos, sin = angles.cos().to(x.dtype)[None, None], angles.sin().to(x.dtype)[None, None]
    even, odd = x[..., 0::2], x[..., 1::2]
    return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, width: int, heads: int, rope_base: int):
        super().__init__()
        assert width % heads == 0
        self.heads, self.head_dim, self.rope_base = heads, width // heads, rope_base
        self.qkv = nn.Linear(width, width * 3, bias=False)
        self.out = nn.Linear(width, width, bias=False)
        self.q_norm, self.k_norm = RMSNorm(self.head_dim), RMSNorm(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        def heads(t: torch.Tensor) -> torch.Tensor:
            return t.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = heads(q), heads(k), heads(v)
        q, k = rope(self.q_norm(q), self.rope_base), rope(self.k_norm(k), self.rope_base)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).contiguous().view(batch, length, -1))


class SwiGLU(nn.Module):
    def __init__(self, width: int, intermediate: int):
        super().__init__()
        self.gate, self.up, self.down = nn.Linear(width, intermediate, bias=False), nn.Linear(width, intermediate, bias=False), nn.Linear(intermediate, width, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, width: int, heads: int, intermediate: int, rope_base: int):
        super().__init__()
        self.attn_norm, self.attn = RMSNorm(width), Attention(width, heads, rope_base)
        self.mlp_norm, self.mlp = RMSNorm(width), SwiGLU(width, intermediate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x))
        return h + self.mlp(self.mlp_norm(h))


class Step1Transformer(nn.Module):
    def __init__(self, vocab_size: int = 262, layers: int = 12, width: int = 384, heads: int = 6, mlp_width: int = 1024, rope_base: int = 10000):
        super().__init__()
        self.vocab_size, self.context_length = vocab_size, None
        self.embedding = nn.Embedding(vocab_size, width)
        self.blocks = nn.ModuleList([Block(width, heads, mlp_width, rope_base) for _ in range(layers)])
        self.norm = RMSNorm(width)
        self.output = nn.Linear(width, vocab_size, bias=False)
        self.output.weight = self.embedding.weight
        self._initialize(layers)

    def _initialize(self, layers: int) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        scale = 1 / math.sqrt(2 * layers)
        for block in self.blocks:
            block.attn.out.weight.data.mul_(scale)
            block.mlp.down.weight.data.mul_(scale)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embedding(tokens)
        for block in self.blocks:
            hidden = block(hidden)
        return self.output(self.norm(hidden))

    def parameter_counts(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        embeddings = self.embedding.weight.numel()
        return {"total": total, "embedding_tied": embeddings, "excluding_embeddings": total - embeddings}


def masked_next_token_loss(logits: torch.Tensor, tokens: torch.Tensor, loss_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Sum then globally-normalize by mask count; mask[t] scores logits[t-1]."""
    if logits.shape[:2] != tokens.shape or tokens.shape != loss_mask.shape:
        raise AssertionError("logits, tokens, and loss mask shapes do not agree")
    targets, scored = tokens[:, 1:], loss_mask[:, 1:].bool()
    count = scored.sum()
    if count.item() == 0:
        raise AssertionError("batch has no supervised action tokens")
    flat = F.cross_entropy(logits[:, :-1].float().flatten(0, 1), targets.flatten(), reduction="none").view_as(targets)
    return (flat * scored).sum(), count
