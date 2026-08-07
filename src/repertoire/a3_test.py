"""A3 encoding-leak test: does the sampled encoding move difficulty?

Adapted from educational measurement's radical/incidental discriminator
(docs/04-radical-incidental.md). A3 requires that the per-episode encoding `e` be
something the family is invariant under. We have been asserting that from
construction -- "these renderings are just notation" -- with no procedure that
could contradict us.

The source procedure: fit an unconstrained item-difficulty model, fit one
constrained so difficulty depends on the hypothesized radical alone, and
likelihood-ratio test the difference. In their worked case the radical explained
95.41% of difficulty variance and the constrained model was STILL rejected at
p < .001. High explained variance and clean separation are different claims.

The analogue here:

    "item family"     -> episodes from one family at one k
    "radical"         -> (family, k)
    "incidental"      -> the encoding sampled per episode
    "item difficulty" -> a per-episode difficulty proxy

WHAT THIS DOES NOT DO. It does not train models. The honest difficulty proxy is
the prequential structural content of Task Spec section 4, which needs the
harness. What this module provides is the *statistic* and a cheap structural
proxy that can run today, so that the test exists before the matrix rather than
after it. Swap in the real proxy when the harness lands -- the comparison is the
part that matters, not the specific measure.

The cheap proxy used here is **episode token length**, which is a genuine
difficulty-adjacent quantity (longer renderings mean more tokens to predict and
more opportunity for the loss to differ) and, more importantly, is a quantity an
encoding can plausibly move without anyone noticing. If encodings differ
systematically in length, they are not exchangeable, and any per-episode loss
average silently mixes an encoding effect into a family effect.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from random import Random
from statistics import fmean, pstdev


@dataclass
class LeakResult:
    family: str
    k: int
    encodings: dict[str, float]  # encoding name -> mean difficulty proxy
    counts: dict[str, int]
    grand_mean: float
    between_var: float
    within_var: float
    f_stat: float
    leaked: bool
    note: str = ""

    def report(self) -> str:
        lines = [
            f"{self.family}  k={self.k}  "
            f"{'LEAK' if self.leaked else 'ok'}  F={self.f_stat:.3f}"
        ]
        for name in sorted(self.encodings):
            spread = self.encodings[name] - self.grand_mean
            pct = 100 * spread / self.grand_mean if self.grand_mean else 0.0
            lines.append(
                f"    {name:<12} n={self.counts[name]:<5} "
                f"mean={self.encodings[name]:.2f}  ({pct:+.2f}%)"
            )
        if self.note:
            lines.append(f"    {self.note}")
        return "\n".join(lines)


def episode_length(family, k: int, seed: int, trials: int = 12) -> tuple[str, int]:
    """Render one episode; return (encoding name, token count).

    Deliberately does the same thing a harness would: sample theta, sample an
    encoding, run T trials, render everything. If a family's renderings differ in
    length this is where it shows.
    """
    rng = Random(seed)
    theta = family.sample_theta(k, rng)
    enc = family.sample_encoding(rng)

    tokens = 0
    pre = family.preamble(theta, enc)
    if pre:
        tokens += len(pre)

    history: list[tuple] = []
    for _ in range(trials):
        q = family.sample_query(theta, history, rng)
        out = family.evaluate(theta, q)
        if isinstance(out, dict):  # distribution-valued; harness samples
            keys = sorted(out)
            answer_cls = _answer_class(family)
            ans = answer_cls(rng.choices(keys, weights=[out[s] for s in keys])[0])
        else:
            ans = out
        tokens += len(family.render(enc, q)) + len(family.render(enc, ans))
        history.append((q, ans))

    return getattr(enc, "name", "default"), tokens


def _answer_class(family):
    """Find the answer type a distribution-valued family expects for rendering.

    Families that return distributions do not tell us what to wrap a sample in.
    That is a small interface gap -- section 7 says `evaluate` returns
    `Answer | Distribution` but never says how a sampled answer is constructed --
    so we look it up per module rather than special-casing family names, which
    the harness is forbidden from doing.
    """
    mod = type(family).__module__
    import importlib

    m = importlib.import_module(mod)
    for candidate in ("Answer", "Outcome", "Category"):
        cls = getattr(m, candidate, None)
        if cls is not None:
            return cls
    raise TypeError(f"{mod} exposes no answer class for a sampled distribution")


def test_encoding_leak(
    family, k: int = 2, n_episodes: int = 400, alpha_f: float = 4.0
) -> LeakResult:
    """One-way comparison of the difficulty proxy across sampled encodings.

    Returns an F-like ratio of between-encoding to within-encoding variance. A
    large ratio means the encoding is moving difficulty, i.e. it is a radical
    wearing an incidental's clothes, i.e. A3 is violated in fact whatever the
    row's verdict says.

    `alpha_f` is a threshold, not a p-value. Being honest about that: with a
    proper difficulty proxy this should become a likelihood-ratio test against a
    stated null, as in the source procedure. The threshold here is a smoke
    detector, and a family that trips it deserves inspection rather than a
    verdict.
    """
    by_enc: dict[str, list[float]] = {}
    for seed in range(n_episodes):
        name, length = episode_length(family, k, seed)
        by_enc.setdefault(name, []).append(float(length))

    if len(by_enc) < 2:
        return LeakResult(
            family=family.name, k=k, encodings={n: fmean(v) for n, v in by_enc.items()},
            counts={n: len(v) for n, v in by_enc.items()},
            grand_mean=fmean([x for v in by_enc.values() for x in v]),
            between_var=0.0, within_var=0.0, f_stat=0.0, leaked=False,
            note="only one encoding sampled -- A3 is not being exercised at all",
        )

    all_vals = [x for v in by_enc.values() for x in v]
    grand = fmean(all_vals)

    between = sum(len(v) * (fmean(v) - grand) ** 2 for v in by_enc.values())
    between /= max(1, len(by_enc) - 1)

    within = sum(pstdev(v) ** 2 * len(v) for v in by_enc.values())
    within /= max(1, len(all_vals) - len(by_enc))

    f_stat = between / within if within > 0 else (math.inf if between > 0 else 0.0)

    return LeakResult(
        family=family.name,
        k=k,
        encodings={n: fmean(v) for n, v in by_enc.items()},
        counts={n: len(v) for n, v in by_enc.items()},
        grand_mean=grand,
        between_var=between,
        within_var=within,
        f_stat=f_stat,
        leaked=f_stat > alpha_f,
        note="" if within > 0 else "zero within-encoding variance: the proxy is degenerate here",
    )


def main() -> int:
    from .families import (
        BrunerConjunctionFamily,
        ConjunctionFamily,
        ConstantTargetFamily,
        ProbabilityMatchingFamily,
        RandomTargetFamily,
        SHJTypeIFamily,
        SHJTypeVIFamily,
    )

    families = [
        SHJTypeIFamily(), SHJTypeVIFamily(), ConjunctionFamily(),
        BrunerConjunctionFamily(), RandomTargetFamily(),
        ConstantTargetFamily(epsilon=0.1), ProbabilityMatchingFamily(),
    ]

    print("A3 encoding-leak test -- proxy: episode token length\n")
    leaks = 0
    for fam in families:
        for k in (1, 3):
            res = test_encoding_leak(fam, k=k)
            print(res.report())
            leaks += res.leaked
    print(f"\n{leaks} family/k combinations show an encoding effect")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
