"""Boolean concept families: the planted prerequisite and near-duplicate pairs.

Task Spec section 8 step 4 gates the whole programme on planted structure being
recovered by the transfer matrix. The two junk plants cover the near-zero end.
These four cover the informative plants:

    SHJTypeI    <-- prerequisite pair -->   SHJTypeVI
    Conjunction <-- near-duplicate  -->     BrunerConjunction

The prerequisite pair carries an externally-attested human difficulty ordering
(Type I reliably learned before Type VI, replicated). The near-duplicate pair is
two independent reinventions of one algorithm -- a 1956 description of human
concept attainment and the membership-query learning of conjunctions -- given
deliberately different surfaces here so that the matrix has to see through the
surface to find the identity.

--------------------------------------------------------------------------
Why the encoding carries the symbol assignment
--------------------------------------------------------------------------

A2 says semantics must be invariant under consistent permutation of the symbol
alphabet, and the check is supposed to be a real test rather than a restatement
of how the family was written. So these families are defined entirely over
*indices* -- dimension i, value j, category 0/1 -- and the Encoding holds the
map from indices to actual tokens, resampled per episode.

That does three things at once. A3 gets satisfied for real, because the surface
genuinely varies per episode rather than by switching a separator. A2 becomes
checkable, because permuting the encoding's assignment must permute the rendered
answers and nothing else. And the model cannot learn a fixed association between
a token and a role, which is the leak that the source paradigms have -- their
stimuli were colours and shapes, and the literature itself reports that shifts
between particular dimensions differ in difficulty, which is direct evidence the
dimensions were never interchangeable as published.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

from .. import vocab

Distribution = dict[int, float]


@dataclass(frozen=True)
class Encoding:
    """Per-episode surface. Holds the index->token assignment, so A2 is testable."""

    name: str
    separator: int
    value_symbols: tuple[int, ...]  # value index -> token
    category_symbols: tuple[int, int]  # category 0/1 -> token
    dimension_order: tuple[int, ...]  # permutation applied when rendering


@dataclass(frozen=True)
class Stimulus:
    values: tuple[int, ...]  # one value index per dimension


@dataclass(frozen=True)
class Category:
    label: int  # 0 or 1


def _sample_encoding(rng: Random, d: int, v: int, name: str, separator: int) -> Encoding:
    # Draw actual symbol TOKEN IDS, not indices 0..N_SYMBOLS. The raw range
    # collides with the control and structural tokens, so an encoding built from
    # it would render content as PAD/BOS/EQ and still pass an in-vocabulary
    # check -- silently, since every id is technically valid.
    pool = list(vocab.SYMBOL_IDS)
    rng.shuffle(pool)
    order = list(range(d))
    rng.shuffle(order)
    return Encoding(
        name=name,
        separator=separator,
        value_symbols=tuple(pool[:v]),
        category_symbols=(pool[v], pool[v + 1]),
        dimension_order=tuple(order),
    )


class _BooleanConceptFamily:
    """Shared machinery. Subclasses supply theta sampling and evaluation."""

    name = "boolean_concept"
    supports_L2 = True
    emits_trace = False
    stochastic = False

    n_values = 2

    def dimensions(self, k: int) -> int:
        """k -> number of dimensions.

        Starts at the paradigms' canonical 3 and grows. The published size fails
        A4 outright -- 8 stimuli is a truth table -- so k exists precisely to
        move these families out of the regime they were designed for, which was
        organisms with limited working memory.
        """
        return 3 + k

    # ---- encodings -------------------------------------------------------

    def sample_encoding(self, rng: Random) -> Encoding:
        d, v = self.dimensions(getattr(self, "_k", 1)), self.n_values
        name, sep = rng.choice(
            [("tuple", vocab.STOI["COMMA"]), ("prose", vocab.STOI["COLON"])]
        )
        return _sample_encoding(rng, d, v, name, sep)

    def render(self, encoding: Encoding, obj: object) -> list[int]:
        if isinstance(obj, Stimulus):
            out: list[int] = []
            for pos, dim in enumerate(encoding.dimension_order):
                if pos:
                    out.append(encoding.separator)
                out.append(encoding.value_symbols[obj.values[dim]])
            out.append(vocab.STOI["ARROW"])
            return out
        if isinstance(obj, Category):
            return [encoding.category_symbols[obj.label]]
        raise TypeError(f"cannot render {type(obj).__name__}")

    # ---- protocol --------------------------------------------------------

    def sample_query(self, theta, history: list, rng: Random) -> Stimulus:
        return Stimulus(
            tuple(rng.randrange(self.n_values) for _ in range(theta.n_dims))
        )

    def trace(self, theta, query: Stimulus) -> list | None:
        return None

    def preamble(self, theta, encoding: Encoding) -> list[int] | None:
        """L0: state the rule as the relevant dimensions, in encoded order."""
        out = [vocab.PREAMBLE]
        for pos, dim in enumerate(encoding.dimension_order):
            if self._is_relevant(theta, dim):
                out.append(encoding.value_symbols[0])
            else:
                out.append(vocab.STOI["NEG"])
        return out

    def _is_relevant(self, theta, dim: int) -> bool:
        raise NotImplementedError

    def permuted_alphabet_check(self, rng: Random) -> bool:
        """A2: apply a TOTAL permutation of the symbol alphabet and require
        the rendered episode to permute with it, exactly.

        The permutation must be total, not a map over the symbols this encoding
        happens to use. A partial map cannot catch the failure it exists to
        catch: a family that renders some token from a fixed constant rather than
        from the encoding would leave that token untouched on both sides of the
        comparison and pass. With a total permutation the constant moves on the
        expected side and does not move on the actual side, so the leak shows.

        Structural tokens (separators, ARROW) are outside the symbol range and
        are deliberately left fixed -- they are notation, not content, and A2 is
        a claim about content symbols.
        """
        k = 1
        self._k = k
        theta = self.sample_theta(k, rng)
        enc = self.sample_encoding(rng)

        shuffled = list(vocab.SYMBOL_IDS)
        rng.shuffle(shuffled)
        perm = dict(zip(vocab.SYMBOL_IDS, shuffled))

        enc2 = Encoding(
            name=enc.name,
            separator=enc.separator,
            value_symbols=tuple(perm[s] for s in enc.value_symbols),
            category_symbols=(
                perm[enc.category_symbols[0]],
                perm[enc.category_symbols[1]],
            ),
            dimension_order=enc.dimension_order,
        )

        for _ in range(64):
            q = self.sample_query(theta, [], rng)
            a = self.evaluate(theta, q)
            expected = [perm.get(t, t) for t in self.render(enc, q) + self.render(enc, a)]
            actual = self.render(enc2, q) + self.render(enc2, a)
            if expected != actual:
                return False
        return True


# --------------------------------------------------------------------------
# The prerequisite pair
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SHJTheta:
    n_dims: int
    relevant: tuple[int, ...]  # participating dimensions
    polarity: int


class SHJTypeIFamily(_BooleanConceptFamily):
    """Type I: one dimension decides. The easy end of the attested ordering."""

    name = "shj_type_i"

    def sample_theta(self, k: int, rng: Random) -> SHJTheta:
        self._k = k
        d = self.dimensions(k)
        return SHJTheta(d, (rng.randrange(d),), rng.randrange(2))

    def _is_relevant(self, theta: SHJTheta, dim: int) -> bool:
        return dim in theta.relevant

    def evaluate(self, theta: SHJTheta, query: Stimulus) -> Category:
        return Category(query.values[theta.relevant[0]] ^ theta.polarity)

    def teacher_query(self, theta: SHJTheta, history: list) -> Stimulus:
        """basis-probe: vary one dimension per trial from a fixed anchor.

        Deterministic in (theta, history) because the protocol gives no rng, and
        A7 wants one pass anyway.
        """
        d = theta.n_dims
        i = len(history) % d
        return Stimulus(tuple(1 if j == i else 0 for j in range(d)))

    def posterior(self, history: list, k: int) -> Distribution:
        """Exact, by enumerating the 2d consistent hypotheses."""
        d = self.dimensions(k)
        survivors = [
            SHJTheta(d, (i,), p) for i in range(d) for p in range(2)
        ]
        for q, a in history:
            if a is None:  # the pending query carries no evidence yet
                continue
            survivors = [t for t in survivors if self.evaluate(t, q).label == a.label]
        if not survivors:
            return {0: 0.5, 1: 0.5}
        return _predictive(self, survivors, history)


class SHJTypeVIFamily(_BooleanConceptFamily):
    """Type VI: parity over all dimensions. No dimension droppable.

    Structurally this IS the parity family from the query-learning vein -- mod-2
    addition over a participating subset -- reached from 1961 categorization
    psychology instead of from computational learning theory. The register
    predicts, before the matrix runs, that it clusters with parity rather than
    with its own paradigm-mate SHJTypeI. That prediction is the reason both are
    implemented rather than just one.
    """

    name = "shj_type_vi"

    def sample_theta(self, k: int, rng: Random) -> SHJTheta:
        self._k = k
        d = self.dimensions(k)
        return SHJTheta(d, tuple(range(d)), rng.randrange(2))

    def _is_relevant(self, theta: SHJTheta, dim: int) -> bool:
        return True

    def evaluate(self, theta: SHJTheta, query: Stimulus) -> Category:
        acc = 0
        for i in theta.relevant:
            acc ^= query.values[i]
        return Category(acc ^ theta.polarity)

    def teacher_query(self, theta: SHJTheta, history: list) -> Stimulus:
        """basis-probe on the standard basis -- the parity schedule exactly."""
        d = theta.n_dims
        i = len(history) % d
        return Stimulus(tuple(1 if j == i else 0 for j in range(d)))

    def posterior(self, history: list, k: int) -> Distribution:
        d = self.dimensions(k)
        survivors = [SHJTheta(d, tuple(range(d)), p) for p in range(2)]
        for q, a in history:
            if a is None:  # the pending query carries no evidence yet
                continue
            survivors = [t for t in survivors if self.evaluate(t, q).label == a.label]
        if not survivors:
            return {0: 0.5, 1: 0.5}
        return _predictive(self, survivors, history)


# --------------------------------------------------------------------------
# The near-duplicate pair
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConjunctionTheta:
    n_dims: int
    required: tuple[int, ...]  # value required per dimension; -1 means don't care


class ConjunctionFamily(_BooleanConceptFamily):
    """Membership in a conjunction over binary dimensions.

    The computational-learning-theory surface: bit vectors, terse rendering.
    """

    name = "conjunction"
    n_values = 2

    def sample_theta(self, k: int, rng: Random) -> ConjunctionTheta:
        self._k = k
        d = self.dimensions(k)
        req = tuple(
            rng.randrange(self.n_values) if rng.random() < 0.5 else -1 for _ in range(d)
        )
        return ConjunctionTheta(d, req)

    def _is_relevant(self, theta: ConjunctionTheta, dim: int) -> bool:
        return theta.required[dim] >= 0

    def evaluate(self, theta: ConjunctionTheta, query: Stimulus) -> Category:
        for i, need in enumerate(theta.required):
            if need >= 0 and query.values[i] != need:
                return Category(0)
        return Category(1)

    def teacher_query(self, theta: ConjunctionTheta, history: list) -> Stimulus:
        """Conservative focusing: anchor on a positive, flip one attribute.

        This is the 1956 human strategy and the membership-query algorithm at the
        same time. That both surfaces produce this identical method is the whole
        content of the near-duplicate plant.
        """
        anchor = tuple(max(x, 0) for x in theta.required)
        d = theta.n_dims
        i = len(history) % d
        flipped = list(anchor)
        flipped[i] = (flipped[i] + 1) % self.n_values
        return Stimulus(tuple(flipped))

    def posterior(self, history: list, k: int) -> Distribution:
        d = self.dimensions(k)
        survivors = _enumerate_conjunctions(d, self.n_values)
        for q, a in history:
            if a is None:  # the pending query carries no evidence yet
                continue
            survivors = [t for t in survivors if self.evaluate(t, q).label == a.label]
        if not survivors:
            return {0: 0.5, 1: 0.5}
        return _predictive(self, survivors, history)


class BrunerConjunctionFamily(ConjunctionFamily):
    """The same latent operation, wearing the 1956 card-array surface.

    Three values per attribute instead of two, and a different rendering. If the
    transfer matrix cannot see that this and `ConjunctionFamily` are one object,
    it cannot see identity through a surface change -- and the whole method of
    translating families across fields into one form loses its warrant. This is
    the single most diagnostic pair in the candidate set.
    """

    name = "bruner_conjunction"
    n_values = 3

    def sample_encoding(self, rng: Random) -> Encoding:
        d, v = self.dimensions(getattr(self, "_k", 1)), self.n_values
        name, sep = rng.choice(
            [("card", vocab.STOI["PIPE"]), ("listing", vocab.STOI["SEP"])]
        )
        return _sample_encoding(rng, d, v, name, sep)


# --------------------------------------------------------------------------


def _enumerate_conjunctions(d: int, v: int) -> list[ConjunctionTheta]:
    out: list[ConjunctionTheta] = []
    total = (v + 1) ** d
    for code in range(total):
        req = []
        x = code
        for _ in range(d):
            x, r = divmod(x, v + 1)
            req.append(r - 1)
        out.append(ConjunctionTheta(d, tuple(req)))
    return out


def _predictive(family, survivors: list, history: list) -> Distribution:
    """Distribution over the next answer. The exact L3 target.

    Uniform over survivors: our priors are uniform over Theta by construction, so
    the Bayes reweighting would be the identity and saying so is cheaper than
    computing it.

    --------------------------------------------------------------------------
    An interface finding, surfaced by implementing this
    --------------------------------------------------------------------------

    Task Spec section 2 says the L3 target is "the target distribution over the
    answer token" -- necessarily conditional on the query being asked. But the
    section 7 signature is `posterior(history, k)` with no query argument, so as
    written the protocol cannot express a query-conditional target.

    It matters. Take Type VI: after one observation theta is fully identified,
    yet the answer distribution *marginalized over a random query* is exactly
    0.5/0.5, because parity is balanced. A harness reading that as the L3 target
    would train the model to be maximally uncertain about a rule it has already
    pinned down -- the calibration target would be teaching miscalibration.

    Convention adopted here, and it needs to reach the harness: **a trailing
    history entry whose answer is None is the pending query**, and the posterior
    is conditioned on it. With no such entry the return is the marginal over the
    query space, which is a coherent object (it is what you should predict before
    seeing the query) but is NOT the L3 training target.
    """
    if not survivors:
        return {0: 0.5, 1: 0.5}

    pending = None
    if history and history[-1][1] is None:
        pending = history[-1][0]

    counts = {0: 0.0, 1: 0.0}
    if pending is not None:
        for t in survivors:
            counts[family.evaluate(t, pending).label] += 1.0
    else:
        d = survivors[0].n_dims
        n_values = family.n_values
        total = n_values ** d
        step = max(1, total // 256)
        for code in range(0, total, step):
            vals = []
            x = code
            for _ in range(d):
                x, r = divmod(x, n_values)
                vals.append(r)
            q = Stimulus(tuple(vals))
            for t in survivors:
                counts[family.evaluate(t, q).label] += 1.0

    z = counts[0] + counts[1]
    return {0: counts[0] / z, 1: counts[1] / z} if z else {0: 0.5, 1: 0.5}
