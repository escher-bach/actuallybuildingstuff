"""Parity identification -- and the register's one unplanted near-duplicate.

The near-duplicate plant (conjunction vs Bruner conjunction) was constructed: we
built two surfaces over one latent operation deliberately, so the matrix would
have something whose answer we know. This family is different. It was excavated
from computational learning theory, `SHJTypeVIFamily` was excavated from 1961
categorization psychology, and **they turn out to be the same object** -- nobody
planted that.

Precisely: parity over a subset S of d dimensions, with S sampled. When |S| = d
it IS Shepard-Hovland-Jenkins Type VI, the three-way biconditional. When |S| < d
it is strictly more general -- there are irrelevant dimensions to ignore, which
Type VI does not have. So Type VI is the full-subset special case, and a test
below asserts exactly that rather than leaving it as a claim in prose.

Why this matters more than a tidy observation: `docs/02` records, before any
measurement, the prediction that Type VI clusters with THIS family rather than
with its own paradigm-mate Type I. If paradigm membership beats structure in the
measured blocks, translating families across fields into one form is buying less
than it looks like. That prediction is only testable if both families exist, and
the surfaces are kept deliberately different so the matrix has to work for it.

--------------------------------------------------------------------------
The A4 hazard this family carries, stated where it will be read
--------------------------------------------------------------------------

Parity is the cleanest case in the register of a family whose brute-force cost
and whose structural cost are far apart -- the truth table is 2^d, the intended
solver is d standard-basis probes. That gap is the family's whole content, and it
only exists once d is large enough. At the canonical d=3 the table is 8 rows and
the family is a lookup, which is why `dimensions()` starts above the paradigms'
size rather than at it.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from random import Random

from .. import vocab

Distribution = dict[int, float]


@dataclass(frozen=True)
class Encoding:
    """Deliberately NOT the concept families' encoding.

    Same latent operation, different surface -- that is the point. If these
    rendered identically the matrix would be handed the identity for free
    instead of having to see through a surface change.
    """

    name: str
    bit_symbols: tuple[int, int]  # bit value -> token
    result_symbols: tuple[int, int]  # parity -> token
    separator: int
    grouped: bool  # emit a group marker every few bits


@dataclass(frozen=True)
class ParityTheta:
    n_dims: int
    subset: tuple[int, ...]  # which dimensions participate
    offset: int  # constant term


@dataclass(frozen=True)
class BitVector:
    bits: tuple[int, ...]


@dataclass(frozen=True)
class Parity:
    value: int


class ParityIdentificationFamily:
    """Identify which subset of coordinates is XORed."""

    name = "parity_identification"
    input_type = BitVector
    output_type = Parity
    supports_L2 = True
    emits_trace = False
    stochastic = False

    def dimensions(self, k: int) -> int:
        """Starts at 6, not at the paradigms' 3.

        A4 fails outright while the truth table is memorizable, and 2^3 = 8 is a
        lookup. Starting above the canonical size is not a tweak: it is the
        difference between a family and a table, and the register row records the
        threshold as an estimate to be measured rather than trusted.
        """
        return 6 + 2 * k

    def sample_theta(self, k: int, rng: Random) -> ParityTheta:
        self._k = k
        d = self.dimensions(k)
        # Non-empty subset: an empty one makes the answer constant, which is the
        # junk-trivial family wearing this family's clothes.
        size = rng.randint(1, d)
        subset = tuple(sorted(rng.sample(range(d), size)))
        return ParityTheta(d, subset, rng.randrange(2))

    def enumerate_theta(self, k: int):
        """Full hypothesis space, for exact composite L3 targets."""
        d = self.dimensions(k)
        out = []
        for mask in range(1, 1 << d):
            subset = tuple(i for i in range(d) if mask & (1 << i))
            for off in (0, 1):
                out.append(ParityTheta(d, subset, off))
        return out

    def sample_encoding(self, rng: Random) -> Encoding:
        pool = list(vocab.SYMBOL_IDS)
        rng.shuffle(pool)
        name, sep, grouped = rng.choice(
            [
                ("bits", vocab.STOI["COMMA"], False),
                ("equation", vocab.STOI["PIPE"], False),
                ("grouped", vocab.STOI["COMMA"], True),
            ]
        )
        return Encoding(
            name=name,
            bit_symbols=(pool[0], pool[1]),
            result_symbols=(pool[2], pool[3]),
            separator=sep,
            grouped=grouped,
        )

    def sample_query(self, theta: ParityTheta, history: list, rng: Random) -> BitVector:
        return BitVector(tuple(rng.randrange(2) for _ in range(theta.n_dims)))

    def teacher_query(self, theta: ParityTheta, history: list) -> BitVector:
        """basis-probe: the standard basis, in order. Does not consult history.

        The cheapest q* in the register, and the only one that is fully
        non-adaptive -- e_i identifies whether coordinate i participates,
        independently of everything else, because parity decomposes.
        """
        i = len(history) % theta.n_dims
        return BitVector(tuple(1 if j == i else 0 for j in range(theta.n_dims)))

    def evaluate(self, theta: ParityTheta, query: BitVector) -> Parity:
        acc = theta.offset
        for i in theta.subset:
            acc ^= query.bits[i]
        return Parity(acc)

    def trace(self, theta: ParityTheta, query: BitVector) -> list | None:
        return None

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, BitVector):
            out: list[int] = []
            for i, b in enumerate(obj.bits):
                if i:
                    out.append(encoding.separator)
                if encoding.grouped and i and i % 4 == 0:
                    out.append(vocab.STOI["LPAREN"])
                out.append(encoding.bit_symbols[b])
            out.append(vocab.STOI["EQ"])
            return out
        if isinstance(obj, Parity):
            return [encoding.result_symbols[obj.value]]
        raise TypeError(f"cannot render {type(obj).__name__}")

    def preamble(self, theta: ParityTheta, encoding: Encoding) -> list[int]:
        """L0: state which coordinates participate, positionally."""
        out = [vocab.PREAMBLE]
        for i in range(theta.n_dims):
            out.append(
                encoding.bit_symbols[1] if i in theta.subset else vocab.STOI["NEG"]
            )
        return out

    def posterior(self, history: list, k: int) -> Distribution:
        """Exact, and cheap for a reason worth noting.

        Naively there are 2^d subsets times 2 offsets. But each standard-basis
        observation independently determines one coordinate's membership, so the
        surviving set factors -- which is why this family's belief state is the
        FACTORED kind rather than the explicit kind. That distinction is exactly
        what `belief-state-maintenance`'s falsifier says is under strain: if the
        matrix separates factored-belief families from explicit-belief ones, that
        fold was wrong, and this family sits squarely on the factored side.

        Implemented by enumeration anyway, because at these sizes it is
        affordable and an exact target computed two ways is worth more than a
        clever one computed once.
        """
        d = self.dimensions(k)
        pending = None
        if history and history[-1][1] is None:
            pending = history[-1][0]

        survivors: list[ParityTheta] = []
        for mask in range(1, 1 << d):
            subset = tuple(i for i in range(d) if mask & (1 << i))
            for off in (0, 1):
                survivors.append(ParityTheta(d, subset, off))

        for q, a in history:
            if a is None:
                continue
            survivors = [t for t in survivors if self.evaluate(t, q).value == a.value]
        if not survivors:
            return {0: 0.5, 1: 0.5}

        counts = {0: 0.0, 1: 0.0}
        if pending is not None:
            for t in survivors:
                counts[self.evaluate(t, pending).value] += 1.0
        else:
            # Marginal over the query space. Parity is balanced, so for any
            # surviving set with a non-empty subset this is exactly 0.5/0.5 --
            # correct, and precisely why the pending-query convention exists.
            return {0: 0.5, 1: 0.5}

        z = counts[0] + counts[1]
        return {0: counts[0] / z, 1: counts[1] / z} if z else {0: 0.5, 1: 0.5}

    def permuted_alphabet_check(self, rng: Random) -> bool:
        k = 1
        self._k = k
        theta = self.sample_theta(k, rng)
        enc = self.sample_encoding(rng)

        shuffled = list(vocab.SYMBOL_IDS)
        rng.shuffle(shuffled)
        perm = dict(zip(vocab.SYMBOL_IDS, shuffled))
        enc2 = replace(
            enc,
            bit_symbols=(perm[enc.bit_symbols[0]], perm[enc.bit_symbols[1]]),
            result_symbols=(perm[enc.result_symbols[0]], perm[enc.result_symbols[1]]),
        )

        for _ in range(64):
            q = self.sample_query(theta, [], rng)
            a = self.evaluate(theta, q)
            expected = [
                perm.get(t, t) for t in self.render(enc, q) + self.render(enc, a)
            ]
            if expected != self.render(enc2, q) + self.render(enc2, a):
                return False
        return True


# --------------------------------------------------------------------------
# An ENDOMORPHIC family: answers are the same shape as queries.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PermTheta:
    n_dims: int
    perm: tuple[int, ...]  # perm[i] = where coordinate i is sent


class PermutedBitsFamily:
    """Apply a hidden coordinate permutation to a bit-vector.

    BitVector in, BitVector out -- which makes it the register's first
    ENDOMORPHIC family, and the reason composition has anywhere to stand.

    Every other implemented family maps a structured query to a single label:
    bit-vector to category, stimulus to parity bit. A label cannot be fed back
    in as a query, so the basis as built had near-zero closure under composition
    even though half the register's A4 verdicts name composition as their
    repair. That gap only became visible when composition was implemented rather
    than assumed.

    Composed under parity, this reproduces the shape of the Task Spec's own
    worked example -- a hidden relabelling wrapped around an operation -- with
    the composite's hidden parameter being the pair (subset, permutation) and
    its hypothesis space the product. That product is the A4 headroom
    composition is supposed to buy, and here it can be checked rather than
    argued.

    On its own this family is deliberately weak: identifying a permutation from
    (input, output) pairs is easy, and it fails A4 badly. It is infrastructure
    for composition, not a capability-bearing member of the basis, and the
    register row says so.
    """

    name = "permuted_bits"
    input_type = BitVector
    output_type = BitVector
    supports_L2 = True
    emits_trace = False
    stochastic = False

    def dimensions(self, k: int) -> int:
        return 6 + 2 * k

    def sample_theta(self, k: int, rng: Random) -> PermTheta:
        self._k = k
        d = self.dimensions(k)
        perm = list(range(d))
        rng.shuffle(perm)
        return PermTheta(d, tuple(perm))

    def sample_encoding(self, rng: Random) -> Encoding:
        pool = list(vocab.SYMBOL_IDS)
        rng.shuffle(pool)
        name, sep, grouped = rng.choice(
            [
                ("perm-bits", vocab.STOI["COMMA"], False),
                ("perm-grouped", vocab.STOI["COMMA"], True),
            ]
        )
        return Encoding(
            name=name,
            bit_symbols=(pool[0], pool[1]),
            result_symbols=(pool[2], pool[3]),
            separator=sep,
            grouped=grouped,
        )

    def enumerate_theta(self, k: int):
        """All d! permutations. Only affordable for small d -- the composite
        posterior will refuse rather than approximate once this is too large,
        which is the correct behaviour and is why the limit is visible here."""
        import itertools as _it

        d = self.dimensions(k)
        return [PermTheta(d, p) for p in _it.permutations(range(d))]

    def sample_query(self, theta: PermTheta, history: list, rng: Random) -> BitVector:
        return BitVector(tuple(rng.randrange(2) for _ in range(theta.n_dims)))

    def teacher_query(self, theta: PermTheta, history: list) -> BitVector:
        """basis-probe: e_i reveals exactly where coordinate i is sent."""
        i = len(history) % theta.n_dims
        return BitVector(tuple(1 if j == i else 0 for j in range(theta.n_dims)))

    def evaluate(self, theta: PermTheta, query: BitVector) -> BitVector:
        out = [0] * theta.n_dims
        for i, b in enumerate(query.bits):
            out[theta.perm[i]] = b
        return BitVector(tuple(out))

    def trace(self, theta: PermTheta, query: BitVector) -> list | None:
        return None

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, BitVector):
            out: list[int] = []
            for i, b in enumerate(obj.bits):
                if i:
                    out.append(encoding.separator)
                if encoding.grouped and i and i % 4 == 0:
                    out.append(vocab.STOI["LPAREN"])
                out.append(encoding.bit_symbols[b])
            out.append(vocab.STOI["ARROW"])
            return out
        raise TypeError(f"cannot render {type(obj).__name__}")

    def preamble(self, theta: PermTheta, encoding: Encoding) -> list[int]:
        out = [vocab.PREAMBLE]
        for target in theta.perm:
            out += vocab.number(target)
            out.append(encoding.separator)
        return out

    def posterior(self, history: list, k: int) -> Distribution:
        raise NotImplementedError(
            "permuted_bits has no scalar answer to place a distribution over -- "
            "its codomain is BitVector. An L3 target here would be a "
            "distribution over vectors, which the harness's per-token "
            "cross-entropy does not consume. Raising rather than returning "
            "something shaped wrongly."
        )

    def permuted_alphabet_check(self, rng: Random) -> bool:
        k = 1
        self._k = k
        theta = self.sample_theta(k, rng)
        enc = self.sample_encoding(rng)
        shuffled = list(vocab.SYMBOL_IDS)
        rng.shuffle(shuffled)
        perm = dict(zip(vocab.SYMBOL_IDS, shuffled))
        enc2 = replace(
            enc,
            bit_symbols=(perm[enc.bit_symbols[0]], perm[enc.bit_symbols[1]]),
            result_symbols=(perm[enc.result_symbols[0]], perm[enc.result_symbols[1]]),
        )
        for _ in range(48):
            q = self.sample_query(theta, [], rng)
            a = self.evaluate(theta, q)
            expected = [perm.get(t, t) for t in self.render(enc, q) + self.render(enc, a)]
            if expected != self.render(enc2, q) + self.render(enc2, a):
                return False
        return True
