# A4 cannot be guaranteed at the sizes we train at

*From vein §2.4. This vein was read specifically because the source document calls it "the one vein that offers A4 as a **guarantee** rather than a hope." It does not — not at our scale — and that is worth writing down properly, because it changes what an A4 verdict in the register can mean.*

## What we were hoping for

Every A4 verdict in the register is currently an **estimate**: "fails at small n, holds from n ≥ 8–10, measure it." Those thresholds were reasoned about, not derived. The hope for this vein was a family whose brute-force cost is *lower-bounded* — where A4 holds by theorem rather than by our judgement, and one row in the register could anchor the rest.

## What is actually available

**Only one construction offers a genuine worst-case-to-average-case reduction** — a guarantee that solving random instances is as hard as solving the hardest instance — rather than the much weaker "the best attack anyone has found is slow." Everything else in the vein rests on the absence of a known algorithm.

And the reduction comes with a size question nobody in the reviewed sources answered: **at what dimension does it become meaningful?** A hardness assumption calibrated for cryptographic parameters says nothing about a family a two-layer model trains on.

**The sharpest illustration is planted clique.** The statistical–computational gap — the range where a solution exists information-theoretically but no efficient algorithm is known to find it — sits between roughly `2·log₂ n` and `Ω(√n)`. Below about n ≈ 256 vertices **that interval is empty.** The gap the family's difficulty is supposed to live in does not exist at toy scale. *(This arithmetic is the reviewer's own inference from the two bounds, flagged as such rather than sourced; the bounds themselves are standard. It should be checked before being leaned on, but the qualitative point — that asymptotic separations can be vacuous at small n — does not depend on the exact number.)*

## The consequence, stated plainly

**A4 stays an estimate, for every row, and no amount of further reading changes that.** This is not a gap in our diligence or in the literature. It is a fact about what asymptotic mathematics can say about instances small enough for a tiny model: the theorems are about `n → ∞`, we operate at `n ≈ 10`, and the separations they establish are frequently *empty* in that regime.

The honest position: A4 is not a property we can certify before training. **It is a property we must measure after.**

Task Spec §9 already anticipated this without quite saying it was forced:

> *"Watch for a turnover in structural content as compute grows on any family; treat it as a signal to raise composition depth, not as noise."*

That is the measurement route, and this vein establishes that it is the **only** route available at our scale, not merely a prudent supplement to a guarantee we might otherwise have had. Which in turn raises the stakes on the standing hazard in `docs/07`: brute-force collapse shows up as a *sign change* in structural content as compute grows, and a sign change looks like noise unless you are watching for it.

## What the vein does give us, and it is worth having

Not a guarantee — a **catalogue of ways planting leaks**, which is directly actionable because each leak is a checkable property of a generator rather than a theorem about a limit.

| Leak | Mechanism | Fix |
|---|---|---|
| **Naive single-solution planting** | Forbidding only the clauses the planted assignment violates leaves solutions *clustered around the plant*; local-search solvers fall into that attractor and find it fast | Plant **two** assignments — the assignment and its complement — so the two attractors largely cancel. Verified: Achlioptas, Jia & Moore, "Hiding Satisfying Assignments: Two are Better than One", JAIR 24 (2005) |
| **Biased planting as an alternative fix** | Same problem, different remedy: skew the clause distribution so the plant is not statistically distinguished | A second published route to the same end, tested at finite size rather than asymptotically |
| **Rejection-sampling generators** | A well-known forced-satisfiable generator achieves satisfiability by *sampling and filtering* | **This is an A1 violation, not an A4 one.** Rejection sampling couples generation cost to difficulty, which is precisely what A1 forbids. Flagged below |
| **Disguised-trapdoor constructions** | A classic public-key construction hid an easy instance behind an affine disguise; the disguise was stripped by a lattice attack — **unconditionally**, not under an assumption | A cautionary case: "no known attack" became "broken" without the underlying problem changing |
| **Low-density instances** | Below a density threshold, a whole class of instances falls to a generic lattice attack regardless of how the planting was done | Density is a parameter that must be checked, not assumed |

**The general shape is one lesson repeated:** planting leaves a *statistical signature*, and the attack is to detect the signature rather than to solve the problem. That is exactly the failure mode A4 exists to prevent, and it is the thing to look for in our own generators — a solver that ignores the intended structure but detects how we generated it.

## Two consequences applied

**1. Rejection sampling is now a named A1 failure, not a judgement call.** Any generator that samples and filters to achieve a property couples generation cost to difficulty. The register's A1 checks should ask "does this filter?" explicitly. Our own generators do not — everything is constructed forward from θ — but the trap is easy to fall into when repairing a family that fails some other check, and the repair would silently break the constraint the whole design rests on.

**2. Our A4 thresholds should be written as predictions, not properties.** A row saying "holds from n ≥ 8" is making a claim that the dial sweep can falsify. That is fine and even good — but it should be *read* as a prediction awaiting measurement, and the register's phrasing ("estimate only; measure it") is now known to be the strongest available form rather than a hedge.

## What this vein contributed to the vocabulary: nothing

Fourth vein in a row to add zero primitives, and for a structural reason worth recording: **this vein's content is almost entirely generator-side.** It is about A1 and A4 — how instances are made and whether making them that way leaks — not about what a solver composes. A vein that never discusses solving cannot mint a solver primitive, so the zero here is weaker evidence for compactness than the zeros from §2.5 and §2.1, which *did* discuss solving and still added nothing.

Worth stating because a saturation curve that counts all zeros equally would overstate its own case.
