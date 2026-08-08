"""Capability probes for the inducer, independent of every family.

Run this when a family will not learn and you need to know whether the fault is
the family, the harness, or the model.  Each probe strips away one more layer of
this repository until nothing is left but the model and an optimizer.

**Written because a plateau cost most of a day to attribute.**  The worked family
sat at 1.7255 nats against a floor of 0 through 30k steps.  Four hypotheses were
tried and all four were wrong -- an inverted preamble, a mismatched preamble
shape, a sparse loss mask, and a small answer alphabet.  Three were real defects
and none of them was the cause.  What settled it was deleting the harness: the
same failure reproduces in forty lines of plain PyTorch, and the canonical
induction task then reaches zero loss at *the same learning rate that fails the
lookup*.

The probes, in the order they discriminate:

    canonical_induction   Repeat a random sequence; score the copy. Every
                          position of the second copy needs the same circuit, so
                          supervision is dense. If this fails, the model or the
                          optimizer is at fault and no family will train.

    key_value_lookup      Key-value pairs, then one query. Structurally the same
                          circuit, but only ONE supervised token per example
                          requires it. If canonical passes and this fails, the
                          scarce resource is not capacity or learning rate -- it
                          is *how many targets require the circuit*.

That distinction is the whole point.  A model can be entirely capable of a
circuit and never build it, if the loss gives it too few reasons to.  Task Spec
section 7 supervises answer tokens only, so in this design the number of reasons
per episode is exactly the number of scored trials -- which makes T, a harness
parameter, load-bearing in a way section 7 does not hint at.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import vocab
from .model import Inducer, ModelConfig


@dataclass
class ProbeResult:
    name: str
    held_out: float
    chance: float
    floor: float  # what a correct circuit achieves
    passed: bool
    detail: str

    def report(self) -> str:
        return (f"{'PASS' if self.passed else 'FAIL'}  {self.name}: "
                f"{self.held_out:.4f} nats (chance {self.chance:.4f}, "
                f"a correct circuit reaches {self.floor:.4f})\n      {self.detail}")


def _optimizer(model, lr):
    return torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                             weight_decay=0.1)


def canonical_induction(
    length: int = 16, steps: int = 2500, batch: int = 32, lr: float = 2e-3,
    n_layer: int = 3, d_model: int = 128, device: str = "cpu", seed: int = 0,
) -> ProbeResult:
    """Repeat a random sequence and score the copy. The textbook induction task.

    Dense supervision: every one of `length` positions in the second copy is
    predictable only by attending to the first copy, so each is a reason to build
    the circuit.  A model with an induction head scores ~0; one without scores
    log(alphabet).
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    syms = torch.tensor(vocab.SYMBOL_IDS, device=dev)
    chance = math.log(len(syms))
    cfg = ModelConfig(n_layer=n_layer, n_head=4, d_model=d_model,
                      d_ff=2 * d_model, max_len=4 * length)
    model = Inducer(cfg).to(dev)
    opt = _optimizer(model, lr)

    def make(n):
        s = syms[torch.randint(len(syms), (n, length), device=dev)]
        return torch.cat([s, s], dim=1)

    for _ in range(steps):
        x = make(batch)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(
            logits[:, length - 1:].reshape(-1, logits.size(-1)),
            x[:, length:].reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        x = make(256)
        lg = model(x[:, :-1])
        v = float(F.cross_entropy(lg[:, length - 1:].reshape(-1, lg.size(-1)),
                                  x[:, length:].reshape(-1)))
    return ProbeResult(
        "canonical induction", v, chance, 0.0, v < 0.5,
        f"{length} supervised tokens per example, all needing the same circuit. "
        + ("The model and optimizer are capable; look further up the stack."
           if v < 0.5 else
           "The model or the optimizer cannot build an induction head here. No "
           "family will train until this passes -- change the model, not the family."),
    )


def key_value_lookup(
    n_keys: int = 6, n_values: int = 8, steps: int = 4000, batch: int = 32,
    lr: float = 2e-3, n_layer: int = 3, d_model: int = 128,
    device: str = "cpu", seed: int = 0,
) -> ProbeResult:
    """Key-value pairs, then one query. Same circuit, one supervised token.

    The floor reported is the best a predictor that *ignores the query* can do --
    uniform over the distinct values present -- because that is the number a
    plateau has to beat before anything can be called learning.  Sitting at it is
    the signature seen on every family in this repository.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    syms = torch.tensor(vocab.SYMBOL_IDS, device=dev)
    cfg = ModelConfig(n_layer=n_layer, n_head=4, d_model=d_model,
                      d_ff=2 * d_model, max_len=4 * n_keys + 16)
    model = Inducer(cfg).to(dev)
    opt = _optimizer(model, lr)
    sep, eq, bos = vocab.SEP, vocab.STOI["EQ"], vocab.BOS

    def make(n):
        seqs, tgts = [], []
        for _ in range(n):
            perm = syms[torch.randperm(len(syms), device=dev)]
            keys = perm[:n_keys].tolist()
            vals = perm[n_keys:n_keys + n_values].tolist()
            assign = [vals[int(torch.randint(n_values, (1,)))] for _ in range(n_keys)]
            s = [bos]
            for kk, vv in zip(keys, assign):
                s += [kk, vv]
            j = int(torch.randint(n_keys, (1,)))
            seqs.append(s + [sep, keys[j], eq])
            tgts.append(assign[j])
        return (torch.tensor(seqs, device=dev), torch.tensor(tgts, device=dev))

    for _ in range(steps):
        x, y = make(batch)
        loss = F.cross_entropy(model(x)[:, -1, :], y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    with torch.no_grad():
        x, y = make(256)
        v = float(F.cross_entropy(model(x)[:, -1, :], y))

    # Best query-ignoring predictor: the empirical distribution of the n_keys
    # assigned values. Estimated rather than derived; it is a baseline, not a
    # gate, and an estimate with its method stated beats a closed form nobody
    # checks.
    import random
    from collections import Counter

    rng = random.Random(0)
    tot = 0.0
    for _ in range(20000):
        c = Counter(rng.randrange(n_values) for _ in range(n_keys))
        tot += -sum((x_ / n_keys) * math.log(x_ / n_keys) for x_ in c.values())
    no_lookup = tot / 20000

    return ProbeResult(
        "key-value lookup (1 target)", v, math.log(n_values), 0.0,
        v < no_lookup - 0.2,
        f"ONE supervised token per example needs the circuit. Best query-ignoring "
        f"predictor scores {no_lookup:.4f}. "
        + ("Beats it, so the query is being read."
           if v < no_lookup - 0.2 else
           "Does not beat it: the query is being ignored. With canonical passing, "
           "the scarce resource is the NUMBER OF TARGETS that require the circuit, "
           "not capacity or learning rate."),
    )


def run_all(device: str = "cpu", quick: bool = False) -> list[ProbeResult]:
    steps = (600, 900) if quick else (2500, 4000)
    return [
        canonical_induction(steps=steps[0], device=device),
        key_value_lookup(steps=steps[1], device=device),
    ]
