"""The form.

Two things live here and they are deliberately in one file:

  1. `TaskFamily` -- the interface from Task Specification section 7, verbatim.
     Every family the harness ever sees is this and nothing else.

  2. `RegisterRow` -- the D1 register schema, whose fields are the section 7
     interface fields plus triage.

They are together because the register exists to answer one question per source:
*does this thing round-trip through the protocol above it?*  A row that cannot
name theta, or names an oracle that has to search, is not a candidate -- it is a
rejection with a reason, and Repertoire Spec section 4.1 says the reason is a
result.  Keeping the schema in the same file as the protocol is the cheapest
available guard against the register drifting into a bibliography.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable

# --------------------------------------------------------------------------
# Task Specification section 7 -- the interface contract
# --------------------------------------------------------------------------

Theta = Any
Encoding = Any
Query = Any
Answer = Any
Step = Any
Distribution = Any


class Level(str, Enum):
    """Task Spec section 2. Four settings of one dial, not four kinds of task.

    The dial is the residual entropy H(y_t | context) and the mode by which it
    can be reduced.  Task Spec section 8 step 5 sweeps this continuously; the
    four names are cut points on that sweep, and the sweep is allowed to move
    them.
    """

    L0 = "L0"  # theta, e stated in preamble          -> execution
    L1 = "L1"  # withheld, identifiable from history  -> task inference
    L2 = "L2"  # withheld, model chooses the query    -> agency, recovery
    L3 = "L3"  # withheld, not identifiable           -> calibration


@runtime_checkable
class TaskFamily(Protocol):
    """Task Spec section 7. The harness knows nothing else about any family.

    Runtime-checkable so a family can be asserted to conform without importing
    it into a harness first.  The check is shallow -- it verifies that the
    methods exist, not that their signatures or return types match -- so it
    catches a family that forgot `posterior` and not one whose `posterior`
    returns the wrong thing.  Treat it as a smoke test, not a contract; the
    contract is enforced by the per-family unit tests (A2 check, L3 targets
    against brute-force enumeration, seeded round-trip).
    """

    name: str
    supports_L2: bool  # False if A7 cannot be satisfied
    emits_trace: bool  # section 1.2
    stochastic: bool  # section 1.3

    def sample_theta(self, k: int, rng) -> Theta: ...
    def sample_encoding(self, rng) -> Encoding: ...

    def sample_query(self, theta: Theta, history: Sequence, rng) -> Query: ...
    def teacher_query(self, theta: Theta, history: Sequence) -> Query: ...  # A7; L2 only
    def evaluate(self, theta: Theta, query: Query) -> Answer | Distribution: ...
    def trace(self, theta: Theta, query: Query) -> list[Step] | None: ...  # section 1.2

    def render(self, encoding: Encoding, obj: Any) -> list[int]: ...
    def preamble(self, theta: Theta, encoding: Encoding) -> list[int] | None: ...  # L0
    def posterior(self, history: Sequence, k: int) -> Distribution: ...  # L3 target

    def permuted_alphabet_check(self, rng) -> bool: ...  # A2 unit test


# --------------------------------------------------------------------------
# Triage vocabulary
# --------------------------------------------------------------------------


class Verdict(str, Enum):
    """Outcome of one admissibility check on one source family.

    REPAIRABLE is the load-bearing value.  Repertoire Spec section 4.1 expects
    most benchmark generators to fail 'is theta separable' and to be repairable;
    a register that only ever says PASS/FAIL is being filled in by someone who
    stopped thinking about the repair.
    """

    PASS = "pass"
    FAIL = "fail"
    REPAIRABLE = "repairable"  # fails as published, admissible after a named change
    UNKNOWN = "unknown"  # not yet determined -- must not survive to D2


class Status(str, Enum):
    LEAD = "lead"  # named, not yet verified against a primary source
    REGISTERED = "registered"  # source verified, form fields filled
    TRANSLATED = "translated"  # round-trips through TaskFamily on paper
    IMPLEMENTED = "implemented"  # code exists (D2)
    REJECTED = "rejected"  # with a reason, which is a result
    SEALED = "sealed"  # falls inside the held-out partition; dropped unread


class ComplexityMode(str, Enum):
    """Repertoire Spec section 4, axis 2."""

    DESCRIPTION_LENGTH = "description_length"
    SERIAL_DEPTH = "serial_depth"
    STATE_WIDTH = "state_width"
    INPUT_ENTROPY = "input_entropy"


class Content(str, Enum):
    """Repertoire Spec section 4, axis 3."""

    MANIPULATION = "manipulation"
    STATE = "state"
    STRUCTURE_INDUCTION = "structure_induction"
    SELECTION = "selection"
    COMPOSITION = "composition"


class Asymmetry(str, Enum):
    """Repertoire Spec section 4, axis 4 -- the generative axis.

    Each value is a family *template*, not a family.  A1 says search the
    asymmetric direction; these are the directions found so far.
    """

    EVALUATE_SEARCH = "evaluate/search"
    EXECUTE_INFER = "execute/infer"
    CORRUPT_LOCALIZE = "corrupt/localize"
    COMPOSE_DECOMPOSE = "compose/decompose"


class PlantRole(str, Enum):
    """Repertoire Spec section 5. Deliberate contamination."""

    NONE = "none"
    NEAR_DUPLICATE = "near_duplicate"
    JUNK_RANDOM = "junk_random"
    JUNK_TRIVIAL = "junk_trivial"
    PREREQUISITE = "prerequisite"  # the 'A' of an attested A-before-B pair
    DEPENDENT = "dependent"  # the 'B'
    INDEPENDENT = "independent"
    SUSPECTED_JUNK = "suspected_junk"  # answer genuinely unknown -- the useful row


# --------------------------------------------------------------------------
# Register schema
# --------------------------------------------------------------------------


@dataclass
class Check:
    """One admissibility check, its verdict, and -- if it failed -- the repair.

    `repair` is not optional politeness.  'Rejected, small stimulus set' is a
    dead row; 'fails A4 at published size 4x4, admissible composed with a
    modulus family or at 12x12' is a candidate with a build note.
    """

    verdict: Verdict = Verdict.UNKNOWN
    reason: str = ""
    repair: str = ""

    def is_resolved(self) -> bool:
        return self.verdict is not Verdict.UNKNOWN


@dataclass
class Source:
    citation: str
    year: int | None = None
    kind: str = ""  # paper | book | benchmark | code | thesis | competition
    url: str = ""
    verified: bool = False  # False => this is a lead, not a claim (section 2 preamble)


@dataclass
class Form:
    """The section 1 / section 7 fields, as prose, before any code exists.

    Filling this in *is* the translation step.  If a field cannot be filled from
    the source, that is the finding -- write what is missing, not a guess.
    """

    theta: str = ""  # what the hidden parameter is
    p_theta: str = ""  # the sampler, and how k enters it
    query_space: str = ""  # X
    oracle: str = ""  # f, stated so that it evaluates and never searches
    encodings: list[str] = field(default_factory=list)  # E; A3 wants a nontrivial set
    k_semantics: str = ""  # what k moves, and whether difficulty is monotone in it
    teacher_query: str = ""  # A7: q*, and its per-call cost
    posterior: str = ""  # L3: is the Bayes posterior enumerable?
    trace: str = ""  # section 1.2: what a step looks like, and the thinning schedule
    stochastic: bool = False  # section 1.3
    episode_length: str = ""  # T to identification; teaching dimension if known


@dataclass
class RegisterRow:
    """One prior task family, translated. D1 is a table of these."""

    id: str
    title: str
    vein: str  # "2.1".."2.6"
    their_term: str = ""  # what the source calls it (section 3 translation table)
    substrate: str = ""  # section 2.5 hazard: primitives are relative to a DSL
    sources: list[Source] = field(default_factory=list)

    form: Form = field(default_factory=Form)

    # --- triage, Repertoire Spec section 4.1 ---
    theta_separable: Check = field(default_factory=Check)
    a1_backward_generable: Check = field(default_factory=Check)
    a2_knowledge_free: Check = field(default_factory=Check)
    a3_encoding_varied: Check = field(default_factory=Check)
    a4_brute_force_resistant: Check = field(default_factory=Check)
    a5_semantically_coherent: Check = field(default_factory=Check)
    a6_transition_cheap_total: Check = field(default_factory=Check)  # L2 only
    a7_teacher_policy: Check = field(default_factory=Check)  # L2 only

    levels: list[Level] = field(default_factory=list)

    # --- section 4 coverage grid ---
    complexity_modes: list[ComplexityMode] = field(default_factory=list)
    contents: list[Content] = field(default_factory=list)
    asymmetries: list[Asymmetry] = field(default_factory=list)

    # --- section 5 / section 6 ---
    plant_role: PlantRole = PlantRole.NONE
    plant_pair: str = ""  # id of the other member, for duplicate/prereq/independent
    primitives: list[str] = field(default_factory=list)  # feeds the saturation curve
    predicted_block: str = ""  # what the SOURCE's taxonomy would predict (section 6)

    status: Status = Status.LEAD
    notes: str = ""

    # ---- derived ----

    @property
    def checks(self) -> dict[str, Check]:
        return {
            "theta_separable": self.theta_separable,
            "A1": self.a1_backward_generable,
            "A2": self.a2_knowledge_free,
            "A3": self.a3_encoding_varied,
            "A4": self.a4_brute_force_resistant,
            "A5": self.a5_semantically_coherent,
            "A6": self.a6_transition_cheap_total,
            "A7": self.a7_teacher_policy,
        }

    @property
    def supports_L2(self) -> bool:
        return Level.L2 in self.levels


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------

# Checks that every row must resolve regardless of which levels it targets.
_CORE_CHECKS = ("theta_separable", "A1", "A2", "A3", "A4", "A5")
_L2_CHECKS = ("A6", "A7")


def validate(row: RegisterRow) -> list[str]:
    """Return a list of problems. Empty list means the row is admissible as a row.

    This does not judge whether the family is any good -- that is section 6's job
    and it is measured, not argued.  It judges whether the row is *decidable*:
    whether someone reading it could implement or reject the family without
    going back to the source.
    """
    problems: list[str] = []
    r = row

    if not r.id or " " in r.id:
        problems.append("id must be a non-empty slug with no spaces")
    # "0" means not excavated from a vein: the Task Spec section 6 worked family,
    # and the constructed plants. Everything else must name where it came from.
    #
    # "2.7" is an ADDITION to the source document's six veins -- agentic harness
    # engineering. Justified in docs/01: Task Spec section 5 names agent use as a
    # target domain, every other vein is historical or academic, and section 10
    # condition 3 judges the basis on coverage of established practice. Adding a
    # vein is a deviation and the validator should not have accepted it silently,
    # which is why it did not.
    if r.vein not in {"0", "2.1", "2.2", "2.3", "2.4", "2.5", "2.6", "2.7"}:
        problems.append(
            f"vein {r.vein!r} is not one of section 2.1-2.6, the added 2.7, or '0'"
        )

    if r.status is Status.SEALED:
        # A sealed row is a deliberate non-read. It carries a reason and nothing else.
        if not r.notes:
            problems.append("sealed row must say which held-out item it fell inside")
        return problems

    if r.status is Status.LEAD:
        # Leads are allowed to be empty; they are the reading queue, not claims.
        return problems

    if r.status is Status.REJECTED:
        # A rejection does NOT require a verified source. Requiring one would make
        # it harder to reject a family than to accept it, which is backwards --
        # and the reasons that matter are usually structural ("a one-bit hidden
        # parameter cannot resist brute force"), readable off the form itself
        # rather than off anyone's citation. What a rejection owes is the reason.
        failed = [n for n, c in r.checks.items() if c.verdict is Verdict.FAIL]
        if not failed:
            problems.append("rejected row records no failing check")
        for n in failed:
            if not r.checks[n].reason:
                problems.append(f"{n} failed with no reason -- a rejection needs one")
        return problems

    # From REGISTERED onward the source must be a claim, not a lead.
    if not any(s.verified for s in r.sources):
        problems.append("no verified primary source (section 2: names and dates are leads)")

    # --- translated / implemented rows carry the full form ---
    f = r.form
    for name in ("theta", "p_theta", "query_space", "oracle", "k_semantics"):
        if not getattr(f, name):
            problems.append(f"form.{name} is empty -- the family does not round-trip yet")
    if len(f.encodings) < 2:
        problems.append("A3 needs a nontrivial encoding set; fewer than 2 renderings listed")

    for name in _CORE_CHECKS:
        c = r.checks[name]
        if not c.is_resolved():
            problems.append(f"{name} unresolved -- UNKNOWN must not survive to D2")
        if c.verdict is Verdict.REPAIRABLE and not c.repair:
            problems.append(f"{name} marked repairable with no named repair")
        if c.verdict is Verdict.FAIL and not c.reason:
            problems.append(f"{name} failed with no reason")

    if Level.L2 in r.levels:
        for name in _L2_CHECKS:
            c = r.checks[name]
            if c.verdict in (Verdict.FAIL, Verdict.UNKNOWN):
                problems.append(
                    f"row claims L2 but {name} is {c.verdict.value}"
                    " -- section 9 flags L2 attrition as expensive to find late"
                )
        if not f.teacher_query:
            problems.append("row claims L2 but names no q* (A7)")
    if Level.L3 in r.levels and not f.posterior:
        problems.append("row claims L3 but does not say how the posterior is computed")

    if not r.levels:
        problems.append("no levels claimed")
    # Junk plants must contribute NO primitives. This is not a formality: a junk
    # family that has an operation worth naming is not junk, and if it reaches
    # the matrix labelled as junk the instrument-validation gate (Task Spec
    # section 8 step 4) passes while measuring the wrong thing.
    if r.plant_role in (PlantRole.JUNK_RANDOM, PlantRole.JUNK_TRIVIAL):
        if r.primitives:
            problems.append(
                f"{r.plant_role.value} lists primitives {r.primitives}"
                " -- if it has an operation worth naming it is not junk"
            )
    elif not r.primitives:
        problems.append("no primitives listed -- the saturation curve (section 7) reads this")
    if r.plant_role in (
        PlantRole.NEAR_DUPLICATE,
        PlantRole.PREREQUISITE,
        PlantRole.DEPENDENT,
        PlantRole.INDEPENDENT,
    ) and not r.plant_pair:
        problems.append(f"plant_role {r.plant_role.value} needs plant_pair")

    return problems
