"""One shared vocabulary, fixed before any family is written.

Task Spec section 7: "Tokenizer -- one shared vocabulary across all families,
fixed before any family is written. A5 is unenforceable otherwise."

That last clause is the whole reason this file exists and is the reason it is
being written now rather than when it is convenient.  A5 says primitives shared
across families denote the same operation.  If each family brought its own
vocabulary, two families using "the same" symbol would not share a token, the
transfer matrix in section 6 would be measuring tokenizer overlap as much as
structure, and the block decomposition read off it would be an artifact.

PROVISIONAL.  Strictly this belongs to the harness (Task Spec section 8 step 1),
not to the repertoire.  It is here because D2 families cannot be written without
it, and it is small enough that the harness can adopt or replace it cheaply.
What must NOT happen is families quietly growing private vocabularies while
waiting for the harness to decide.

Design notes:

* `SYM_0..SYM_63` are the content alphabet and are **semantically empty**.  A2
  requires that a consistent permutation of them changes nothing, so nothing in
  this file may attach meaning to any particular one, and no family may either.
  They are deliberately not letters: `a`, `b`, `x` carry conventions -- `x` is an
  unknown, `n` is a count -- and those conventions are exactly the knowledge
  leak A2 exists to exclude.
* Digits are separate from symbols.  A family that wants number-shaped answers
  uses digits; a family that wants opaque labels uses symbols.  Conflating them
  hands the model a free bijection between labels and magnitudes.
* Structural tokens are shared across families on purpose.  `EQ` means "the
  answer follows" everywhere, so an encoding learned in one family is not
  re-learned in the next.  This is A5 at the surface level.
"""

from __future__ import annotations

N_SYMBOLS = 64
N_DIGITS = 10

_CONTROL = [
    "PAD",
    "BOS",
    "EOS",
    "SEP",       # between trials within an episode
    "PREAMBLE",  # opens an L0 rule statement; never supervised
    "ASK",       # opens a model-emitted query (L2)
    "ANSWER",    # opens an oracle response
    "STEP",      # opens one line of a trace (section 1.2)
    "ERR",       # oracle's well-formed refusal of a malformed query (A6)
    "UNK",
]

_STRUCTURAL = [
    "EQ",        # =
    "ARROW",     # ->
    "COMMA",
    "LPAREN",
    "RPAREN",
    "PIPE",
    "QMARK",
    "COLON",
    "NEG",
    "TRUE",
    "FALSE",
]

# Operator slots are abstract. A family binds OP_0 to whatever its operation is
# and declares the binding in its encoding; nothing here says OP_0 is addition.
# This is what lets two families share the token for "the binary operation" while
# denoting different things, and lets one family permute its operator symbol as
# part of the encoding (A3).
_OPERATORS = [f"OP_{i}" for i in range(8)]

TOKENS: list[str] = (
    _CONTROL
    + _STRUCTURAL
    + _OPERATORS
    + [f"DIGIT_{i}" for i in range(N_DIGITS)]
    + [f"SYM_{i}" for i in range(N_SYMBOLS)]
)

STOI: dict[str, int] = {t: i for i, t in enumerate(TOKENS)}
ITOS: list[str] = list(TOKENS)
VOCAB_SIZE = len(TOKENS)

# Convenience handles
PAD, BOS, EOS, SEP = (STOI[t] for t in ("PAD", "BOS", "EOS", "SEP"))
PREAMBLE, ASK, ANSWER, STEP, ERR, UNK = (
    STOI[t] for t in ("PREAMBLE", "ASK", "ANSWER", "STEP", "ERR", "UNK")
)

SYMBOL_IDS: list[int] = [STOI[f"SYM_{i}"] for i in range(N_SYMBOLS)]
DIGIT_IDS: list[int] = [STOI[f"DIGIT_{i}"] for i in range(N_DIGITS)]
OPERATOR_IDS: list[int] = [STOI[f"OP_{i}"] for i in range(8)]


def sym(i: int) -> int:
    """Token id of content symbol i. Semantically empty by contract."""
    return SYMBOL_IDS[i]


def digit(i: int) -> int:
    return DIGIT_IDS[i]


def decode(ids: list[int]) -> str:
    """Human-readable rendering. Debugging only -- never on the training path."""
    return " ".join(ITOS[i] if 0 <= i < VOCAB_SIZE else "?" for i in ids)


def number(n: int) -> list[int]:
    """Render a non-negative integer as digit tokens, most significant first."""
    if n < 0:
        raise ValueError("vocabulary has no sign token; encode sign in the family")
    if n == 0:
        return [DIGIT_IDS[0]]
    out: list[int] = []
    while n:
        n, r = divmod(n, 10)
        out.append(DIGIT_IDS[r])
    return out[::-1]
