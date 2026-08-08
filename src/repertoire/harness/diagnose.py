"""Capability probes for the inducer, independent of every family.

Run this when a family will not learn and you need to know whether the fault is
the family, the harness, or the model.  Each probe strips away one more layer of
this repository until nothing is left but the model and an optimizer.

**Written because a plateau cost most of a day to attribute.**  The worked family
sat at 1.7255 nats against a floor of 0 through 30k steps.  Five hypotheses were
tried and all five were wrong -- an inverted preamble, a mismatched preamble
shape, a sparse loss mask, a small answer alphabet, and too few supervised
targets per episode.  Three were real defects, now fixed, and none was the cause.

What settled it was deleting the harness: the same failure reproduces in forty
lines of plain PyTorch, so nothing in this repository is responsible.

**And then the control itself was wrong**, which is the part worth remembering.
A fixed-length `seq ++ seq` copy task scored 0.0000 and was read as "the model
can build an induction head, so look elsewhere".  It cannot.  That task is
solvable by attending to a *fixed position offset*, and once the offset is made
variable the same model only gets partway.  The false negative sent the search
up the stack for hours.  A capability probe with a shortcut in it is worse than
none, because it certifies a capability that is absent.

The probes, in the order they discriminate:

    sequence_copy(fixed)     seq ++ seq. The match is always the same distance
                             back, so a POSITIONAL rule solves it. Passing says
                             nothing about content matching -- reading it as if
                             it did is the mistake that cost a day here.

    sequence_copy(variable)  A random-length filler between the copies, so the
                             distance varies and only content matching works.
                             THIS is the capability every family needs.

    key_value_lookup         Key-value pairs, then one query -- the same
                             capability in the shape the families use it.

**Why this matters for the whole programme.**  L1, L2 and L3 are *defined* by
requiring in-context inference about a theta resampled every episode, so every
family above L0 needs content matching.  A model that copies by position and not
by content will sit at "uniform over the answer symbols this episode names" on
all of them, forever, while looking like it is training.  The one family that
escapes is `junk_trivial`, whose rule IS positional -- copy the previous answer --
and it reaches its optimum exactly.  That pattern (everything flat except the
positional family) is the signature to watch for.
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


def sequence_copy(
    length: int = 12, steps: int = 2500, batch: int = 32, lr: float = 2e-3,
    n_layer: int = 3, d_model: int = 128, device: str = "cpu", seed: int = 0,
    variable_offset: bool = True,
) -> ProbeResult:
    """Repeat a random sequence and score the copy.

    **`variable_offset` is the whole probe, and getting it wrong wasted a day.**

    With `variable_offset=False` the input is `seq ++ seq` at a fixed length, so
    the match always sits exactly `length` positions back.  That is solvable by
    "attend to position -length" -- a *positional* rule needing no content
    matching at all -- and this model solves it to 0.0000.  Read as evidence of
    an induction head, which is what happened here, it is worse than no evidence:
    it certifies a capability the model does not have.

    With `variable_offset=True` a filler of random length sits between the
    copies, so the distance to the match changes from batch to batch and no
    fixed offset works.  Only content matching solves it.  That is the circuit
    every family in this repository needs, because L1-L3 are *defined* by
    requiring in-context inference about a theta resampled every episode.

    Run both.  The pair is the diagnostic: fixed solving while variable fails
    localizes the fault precisely, and either one alone is misleading.
    """
    torch.manual_seed(seed)
    dev = torch.device(device)
    syms = torch.tensor(vocab.SYMBOL_IDS, device=dev)
    chance = math.log(len(syms))
    cfg = ModelConfig(n_layer=n_layer, n_head=4, d_model=d_model,
                      d_ff=2 * d_model, max_len=6 * length)
    model = Inducer(cfg).to(dev)
    opt = _optimizer(model, lr)

    def make(n):
        # One filler length per batch: sequences stay rectangular, and across
        # batches the offset varies, which is what kills the positional route.
        f = int(torch.randint(1, length + 1, (1,))) if variable_offset else 0
        s = syms[torch.randint(len(syms), (n, length), device=dev)]
        fill = syms[torch.randint(len(syms), (n, f), device=dev)] if f else s[:, :0]
        return torch.cat([s, fill, s], dim=1), length + f

    for _ in range(steps):
        x, start = make(batch)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(
            logits[:, start - 1:].reshape(-1, logits.size(-1)),
            x[:, start:].reshape(-1),
        )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    model.eval()
    tot = 0.0
    with torch.no_grad():
        for _ in range(8):
            x, start = make(128)
            lg = model(x[:, :-1])
            tot += float(F.cross_entropy(lg[:, start - 1:].reshape(-1, lg.size(-1)),
                                         x[:, start:].reshape(-1)))
    v = tot / 8
    kind = "variable offset (content matching)" if variable_offset else \
           "fixed offset (positional copy)"
    return ProbeResult(
        f"sequence copy, {kind}", v, chance, 0.0, v < 0.5,
        ("Solved." if v < 0.5 else "NOT solved.") + (
            " Content matching works; look further up the stack."
            if variable_offset and v < 0.5 else
            " The model cannot match on content at this size, only on position. "
            "Every family here needs content matching, so none will train until "
            "this passes -- change the model, not the family."
            if variable_offset else
            " Positional copying only; this says nothing about content matching."
        ),
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
    """The three probes, in the order that localizes the fault.

    Fixed-offset copy first because it is the shortcut: if it fails, nothing else
    is worth running. Variable-offset second because it is the capability every
    family actually needs. Key-value last because it is the same capability in
    the shape the families use it.
    """
    steps = (600, 900, 900) if quick else (2500, 2500, 4000)
    return [
        sequence_copy(steps=steps[0], device=device, variable_offset=False),
        sequence_copy(steps=steps[1], device=device, variable_offset=True),
        key_value_lookup(steps=steps[2], device=device),
    ]
