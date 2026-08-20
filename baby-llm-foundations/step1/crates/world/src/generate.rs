//! Deterministic instance sampler and structural validator for the world
//! family in `lib.rs` (STEP-1.md 3.1, 3.2). No RNG crate: a hand-rolled
//! splitmix64 (Vigna) drives everything, seeded solely by `(seed, index)`.
//!
//! `structural_check` is the primary deliverable: it enforces every
//! admissibility constraint in STEP-1 3.2 on an arbitrary `Instance`,
//! including hand-built ones. `sample` builds one candidate deterministically
//! from `(seed, index)` and defers to `structural_check` (plus one knob
//! `structural_check` cannot see, `FamilyParams::min_depth`) to accept or
//! reject it. There is no internal retry loop: one `(seed, index)` yields at
//! most one candidate, and rejection just means the caller tries the next
//! `index`.
//!
//! `budget`, `step_limit`, and the enforced `min_depth` are computed from the
//! *worst case over every hypothesis*, never from `Instance::truth` alone.
//! `truth` is privileged, but `budget`/`step_limit` are learner-visible
//! before a single probe is inspected (STEP-1 3.6 permits "public cost or
//! budget information"); if either were derived from the truth-specific
//! identifying set, a learner could read `h*` off the budget alone without
//! ever executing an `Inspect` -- the same leak STEP-1 3.2's "no
//! learner-visible identifier directly encodes h*" forbids for the evidence
//! table, just moved to a different field. See `per_hypothesis_identifying_sets`.

use crate::{reset, valid_actions, EvidenceId, HypId, Instance, ProbeId, Variant};

/// Exact identifying-set search (`identifying_set`) is exponential in
/// `n_hyp` (2^(n_hyp-1) coverage masks, linear in `n_probe`), and is now run
/// twice (cost-weighted and unit-weighted) for *every* hypothesis to make
/// `budget`/`step_limit` truth-blind (`per_hypothesis_identifying_sets`).
/// This bounds the per-hypothesis search space to at most 2^19 = 524288
/// masks, so the total work is bounded by
/// `O(n_hyp * 2^(n_hyp-1) * n_probe)`. `sample` and `structural_check` both
/// reject with `Reject::SearchTooLarge` above this rather than run an
/// unbounded exact search.
const MAX_HYP_FOR_EXACT_SEARCH: u16 = 20;

/// STEP-1 3.2's literal floor ("at least one generated decision depends on
/// previous evidence rather than only on the current surface string"): the
/// shortest identifying probe set -- for the *easiest* hypothesis, so this
/// cannot leak `truth` either -- must take at least two probes. `Instance`
/// carries no `min_depth` field (that knob lives only in `FamilyParams`), so
/// `structural_check` -- which must work on any hand-built `Instance` -- can
/// only enforce this fixed baseline. `sample` additionally enforces the
/// (possibly stricter) `FamilyParams::min_depth` before this baseline ever
/// runs; see `sample`'s doc comment.
const MIN_DEPTH_FLOOR: usize = 2;

/// Difficulty knobs for one structural family (STEP-1 3.2). Every field here
/// is read by `sample` or by the `assign_split` partition below; a field is
/// present only because something in this file uses it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FamilyParams {
    pub n_hyp: u16,
    pub n_probe: u16,
    /// Size of the evidence alphabet probes draw values from.
    pub n_evidence: u16,
    pub cost_lo: i32,
    pub cost_hi: i32,
    /// Budget granted above the worst-case (over every hypothesis)
    /// minimum-cost identifying probe set.
    pub budget_slack: i32,
    /// Required size of the shortest discriminating probe set, checked
    /// against the *easiest* hypothesis so it cannot leak `truth`. Use
    /// `>= 2` to force genuine evidence accumulation (STEP-1 3.2).
    pub min_depth: u16,
    /// Extra actions granted above the worst-case minimum win length. This
    /// is what makes recovery reachable: STEP-1 3.5's forced-prefix
    /// evaluation places a `Reversible` learner after an earlier wrong
    /// provisional commit, and with zero slack the exact-minimum
    /// `step_limit` leaves no room for that wasted commit action, making
    /// recovery structurally unreachable even though `Reversible` is
    /// supposed to allow it.
    pub step_slack: u16,
    pub variant: Variant,
}

/// Why `sample` or `structural_check` rejected a candidate. Doubles as the
/// structural validator's error type, so every variant carries enough
/// payload to debug the rejection without re-deriving it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Reject {
    /// STEP-1 3.2: "at least two hypotheses are initially observationally
    /// possible."
    TooFewHypotheses { n_hyp: u16 },
    /// `n_hyp` exceeds `MAX_HYP_FOR_EXACT_SEARCH`; the exact identifying-set
    /// search this module relies on is not run.
    SearchTooLarge { n_hyp: u16 },
    /// STEP-1 3.2: "at least two learner actions are available at some
    /// nonterminal history." Checked at the initial (reset) history, which
    /// is representative here: `Commit(h)` is unconditionally listed for
    /// every `h` at every nonterminal history (see `lib.rs::valid_actions`),
    /// so if two actions are reachable at reset, they remain reachable at
    /// every later nonterminal history too.
    NoBranchingAction,
    /// Every probe's evidence row is constant across hypotheses: no probe
    /// changes the set of hypotheses consistent with the history.
    NoDiscriminatingProbe,
    /// Some probe's evidence row is injective in `h` (all `n_hyp` values
    /// distinct): observing that probe alone hands over the true hypothesis
    /// outright, regardless of which hypothesis happens to be true. This
    /// also catches a literal `evidence_of(q,h) == h` identity table, which
    /// is a special case of an injective row.
    TruthLeakingProbe { probe: ProbeId },
    /// Hypothesis `h` has no identifying probe set at all, at any cost. This
    /// is checked for *every* hypothesis, not only `inst.truth`: see
    /// `per_hypothesis_identifying_sets`.
    NotIdentifiable { h: HypId },
    /// Some hypothesis is identifiable, but the most expensive
    /// minimum-cost identifying set across all hypotheses (`hardest_hyp`,
    /// costing `min_cost`) exceeds the declared budget. Checking every
    /// hypothesis, not just `inst.truth`, is what keeps `budget` from
    /// encoding which hypothesis is actually true.
    NotIdentifiableWithinBudget {
        hardest_hyp: HypId,
        min_cost: i32,
        budget: i32,
    },
    /// Even the worst-case (largest) minimum-cost identifying probe set
    /// across all hypotheses (`hardest_hyp`), plus the variant's commit
    /// overhead, needs more actions than `step_limit` allows. See
    /// `sample`'s doc comment for the exact arithmetic.
    NotWinnableWithinStepLimit {
        hardest_hyp: HypId,
        required: u16,
        step_limit: u16,
    },
    /// The shortest discriminating probe set, minimized over *every*
    /// hypothesis (unit-cost search), is smaller than the required minimum:
    /// `min_depth` when checked by `sample`, `MIN_DEPTH_FLOOR` when checked
    /// by `structural_check` directly.
    DepthTooShallow { shortest: usize, min_depth: u16 },
    /// `FamilyParams` was internally inconsistent, or the assembled
    /// `Instance` failed `Instance::validate`'s shape checks.
    InvalidParams(String),
}

/// splitmix64 (Vigna 2015): a 6-line, allocation-free PRNG. Not
/// cryptographic; chosen only because it is small, fast, and has no external
/// dependency.
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(state: u64) -> Self {
        SplitMix64(state)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// The stream is a function of `(seed, index)` only, matching STEP-1 9's
/// reproducibility requirement.
fn seeded_rng(seed: u64, index: u64) -> SplitMix64 {
    let mut warm = SplitMix64::new(seed);
    let a = warm.next();
    SplitMix64::new(a ^ index.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31))
}

fn validate_params(p: &FamilyParams) -> Result<(), Reject> {
    if p.n_hyp < 2 {
        return Err(Reject::TooFewHypotheses { n_hyp: p.n_hyp });
    }
    if p.n_hyp > MAX_HYP_FOR_EXACT_SEARCH {
        return Err(Reject::SearchTooLarge { n_hyp: p.n_hyp });
    }
    if p.n_probe == 0 || p.n_probe > 64 {
        return Err(Reject::InvalidParams(format!(
            "n_probe {} must be in 1..=64 (Instance's probed bitset is 64 bits wide)",
            p.n_probe
        )));
    }
    if p.n_evidence == 0 {
        return Err(Reject::InvalidParams("n_evidence must be >= 1".to_string()));
    }
    if p.cost_lo < 0 || p.cost_hi < p.cost_lo {
        return Err(Reject::InvalidParams(format!(
            "cost range [{}, {}] is invalid",
            p.cost_lo, p.cost_hi
        )));
    }
    if p.budget_slack < 0 {
        return Err(Reject::InvalidParams(
            "budget_slack must be >= 0".to_string(),
        ));
    }
    Ok(())
}

fn sample_evidence_table(p: &FamilyParams, rng: &mut SplitMix64) -> Vec<EvidenceId> {
    let mut table = Vec::with_capacity(p.n_probe as usize * p.n_hyp as usize);
    for _q in 0..p.n_probe {
        for h in 0..p.n_hyp {
            let mut v = (rng.next() % p.n_evidence as u64) as u16;
            // Never literally hand `h` back as its own evidence value.
            if v == h && p.n_evidence > 1 {
                v = (v + 1) % p.n_evidence;
            }
            table.push(v);
        }
    }
    table
}

fn sample_costs(p: &FamilyParams, rng: &mut SplitMix64) -> Vec<i32> {
    let span = (p.cost_hi - p.cost_lo + 1) as u64;
    (0..p.n_probe)
        .map(|_| p.cost_lo + (rng.next() % span) as i32)
        .collect()
}

fn any_probe_varies(n_probe: u16, evidence: &[EvidenceId], n_hyp: u16) -> bool {
    (0..n_probe).any(|q| {
        let base = evidence[q as usize * n_hyp as usize];
        (1..n_hyp).any(|h| evidence[q as usize * n_hyp as usize + h as usize] != base)
    })
}

fn injective_probe(n_probe: u16, evidence: &[EvidenceId], n_hyp: u16) -> Option<ProbeId> {
    for q in 0..n_probe {
        let mut seen = std::collections::HashSet::with_capacity(n_hyp as usize);
        let injective = (0..n_hyp).all(|h| seen.insert(evidence[q as usize * n_hyp as usize + h as usize]));
        if injective {
            return Some(q);
        }
    }
    None
}

/// Exact minimum-weight probe set that distinguishes the given hypothesis
/// `pivot` from every other hypothesis (weighted set cover over the
/// `n_hyp - 1` other hypotheses). `pivot` need not be `Instance::truth` --
/// `per_hypothesis_identifying_sets` below calls this once per hypothesis,
/// treating each in turn as if it were the truth, precisely so that no
/// single call's result is truth-specific.
///
/// Exact via DP over `2^(n_hyp-1)` coverage masks: `dp[mask]` is the
/// minimum weight to have distinguished every hypothesis flagged in `mask`;
/// since `mask | coverage(q) >= mask` always (OR only sets bits), scanning
/// masks in increasing numeric order finalizes each `dp[mask]` before it is
/// used, so a single forward pass suffices. Cost: `O(2^(n_hyp-1) *
/// n_probe)`; callers must have already bounded
/// `n_hyp <= MAX_HYP_FOR_EXACT_SEARCH`. Returns `None` if no subset of
/// probes identifies `pivot` at all.
fn identifying_set(
    n_probe: u16,
    evidence: &[EvidenceId],
    n_hyp: u16,
    pivot: HypId,
    weight: impl Fn(ProbeId) -> i64,
) -> Option<(i64, Vec<ProbeId>)> {
    let others: Vec<HypId> = (0..n_hyp).filter(|&h| h != pivot).collect();
    let k = others.len();
    if k == 0 {
        return Some((0, Vec::new()));
    }

    let mut coverage = vec![0u32; n_probe as usize];
    for q in 0..n_probe {
        let e_pivot = evidence[q as usize * n_hyp as usize + pivot as usize];
        let mut bits = 0u32;
        for (i, &h) in others.iter().enumerate() {
            let e_h = evidence[q as usize * n_hyp as usize + h as usize];
            if e_h != e_pivot {
                bits |= 1u32 << i;
            }
        }
        coverage[q as usize] = bits;
    }

    let full = ((1u64 << k) - 1) as u32;
    let size = 1usize << k;
    let mut dp: Vec<i64> = vec![i64::MAX; size];
    let mut parent: Vec<Option<(u32, ProbeId)>> = vec![None; size];
    dp[0] = 0;

    for mask in 0..size {
        if dp[mask] == i64::MAX {
            continue;
        }
        for q in 0..n_probe {
            let new_mask = (mask as u32 | coverage[q as usize]) as usize;
            let new_cost = dp[mask] + weight(q);
            if new_cost < dp[new_mask] {
                dp[new_mask] = new_cost;
                parent[new_mask] = Some((mask as u32, q));
            }
        }
    }

    let full_idx = full as usize;
    if dp[full_idx] == i64::MAX {
        return None;
    }
    let mut set = Vec::new();
    let mut cur = full_idx;
    while cur != 0 {
        let (prev, q) = parent[cur].expect("reachable mask must have a recorded parent");
        set.push(q);
        cur = prev as usize;
    }
    set.sort_unstable();
    Some((dp[full_idx], set))
}

/// Per-hypothesis identifying-set summary: for hypothesis `h`, the
/// minimum-cost probe set that would identify `h` if `h` were the truth
/// (`cost_set`, `cost`), and separately the cardinality of the
/// minimum-*cardinality* (unit-weight) identifying set for `h` (`shortest`
/// -- may differ from `cost_set.len()`, since the cheapest set need not be
/// the smallest one).
struct HypSummary {
    cost_set: Vec<ProbeId>,
    cost: i64,
    shortest: usize,
}

/// Computes a `HypSummary` for *every* hypothesis in `0..n_hyp`. `budget`,
/// `step_limit`, and the enforced `min_depth` are all derived from the
/// worst case (`budget`/`step_limit`: max; `min_depth`: min) across this
/// whole vector -- never from a single distinguished hypothesis -- so that
/// they are functions of the evidence table and probe costs alone and carry
/// no information about which hypothesis is `Instance::truth`. Returns
/// `Err(h)` for the first hypothesis with no identifying set at all: the
/// max/min these summaries feed are only meaningful once every hypothesis
/// is identifiable.
fn per_hypothesis_identifying_sets(
    n_probe: u16,
    evidence: &[EvidenceId],
    n_hyp: u16,
    cost: impl Fn(ProbeId) -> i64,
) -> Result<Vec<HypSummary>, HypId> {
    let mut out = Vec::with_capacity(n_hyp as usize);
    for h in 0..n_hyp {
        let Some((c, cost_set)) = identifying_set(n_probe, evidence, n_hyp, h, &cost) else {
            return Err(h);
        };
        let (_, unit_set) = identifying_set(n_probe, evidence, n_hyp, h, |_| 1i64)
            .expect("reachability already confirmed by the cost-weighted search above");
        out.push(HypSummary {
            cost_set,
            cost: c,
            shortest: unit_set.len(),
        });
    }
    Ok(out)
}

/// The `Irreversible`/`Reversible` step overhead beyond the identifying
/// set's own inspects: `Irreversible` needs one terminating `Commit`;
/// `Reversible` needs two (a provisional `Commit`, then a repeat to
/// confirm -- see `lib.rs::step`'s termination rule). `pub(crate)`: also
/// used by `teacher::teach` to reserve step-budget room for the eventual
/// commit(s) when deciding how many probes it can still afford to propose.
pub(crate) fn commit_overhead(variant: Variant) -> u16 {
    match variant {
        Variant::Irreversible => 1,
        Variant::Reversible => 2,
    }
}

/// Validates `inst` against every structural constraint in STEP-1 3.2,
/// independent of any `FamilyParams` (this takes only a compiled
/// `Instance`, so it works equally on sampler output and hand-built
/// instances). Order matters only for which `Reject` is reported first when
/// several constraints are violated; the checks are otherwise independent.
///
/// One knob is intentionally not enforced here: `FamilyParams::min_depth`
/// may demand a stricter shortest-discriminating-set size than the fixed
/// `MIN_DEPTH_FLOOR` this function checks, because `Instance` carries no
/// `min_depth` field to check against. `sample` enforces the stricter bound
/// itself before an `Instance` is even assembled.
///
/// Deliberate design choice for identifiability, budget, and step_limit:
/// this function requires EVERY hypothesis -- not only `inst.truth` -- to be
/// identifiable within `budget` and winnable within `step_limit`.
/// `inst.truth` is never read below. The alternative (checking only
/// `inst.truth`) would make `structural_check` itself accept or reject an
/// otherwise-identical instance differently depending on the hidden truth
/// value, which is exactly the kind of truth-dependence this function
/// exists to rule out: `budget` and `step_limit` are learner-visible before
/// any evidence is gathered (STEP-1 3.6), so their validity must be a
/// property of the evidence table and declared costs alone.
pub fn structural_check(inst: &Instance) -> Result<(), Reject> {
    if inst.n_hyp < 2 {
        return Err(Reject::TooFewHypotheses { n_hyp: inst.n_hyp });
    }
    if inst.n_hyp > MAX_HYP_FOR_EXACT_SEARCH {
        return Err(Reject::SearchTooLarge { n_hyp: inst.n_hyp });
    }

    let st0 = reset(inst);
    if valid_actions(inst, &st0).len() < 2 {
        return Err(Reject::NoBranchingAction);
    }

    if !any_probe_varies(inst.n_probe, &inst.evidence, inst.n_hyp) {
        return Err(Reject::NoDiscriminatingProbe);
    }

    if let Some(q) = injective_probe(inst.n_probe, &inst.evidence, inst.n_hyp) {
        return Err(Reject::TruthLeakingProbe { probe: q });
    }

    let summaries = per_hypothesis_identifying_sets(
        inst.n_probe,
        &inst.evidence,
        inst.n_hyp,
        |q| inst.probe_cost[q as usize] as i64,
    )
    .map_err(|h| Reject::NotIdentifiable { h })?;

    let shortest = summaries.iter().map(|s| s.shortest).min().unwrap();
    if shortest < MIN_DEPTH_FLOOR {
        return Err(Reject::DepthTooShallow {
            shortest,
            min_depth: MIN_DEPTH_FLOOR as u16,
        });
    }

    let (hardest_cost_hyp, hardest_cost) = summaries
        .iter()
        .enumerate()
        .map(|(h, s)| (h as u16, s.cost))
        .max_by_key(|&(_, c)| c)
        .unwrap();
    if hardest_cost > inst.budget as i64 {
        return Err(Reject::NotIdentifiableWithinBudget {
            hardest_hyp: hardest_cost_hyp,
            min_cost: hardest_cost.min(i32::MAX as i64) as i32,
            budget: inst.budget,
        });
    }

    let overhead = commit_overhead(inst.variant);
    let (hardest_len_hyp, hardest_len) = summaries
        .iter()
        .enumerate()
        .map(|(h, s)| (h as u16, s.cost_set.len()))
        .max_by_key(|&(_, l)| l)
        .unwrap();
    let required = hardest_len as u16 + overhead;
    if required > inst.step_limit {
        return Err(Reject::NotWinnableWithinStepLimit {
            hardest_hyp: hardest_len_hyp,
            required,
            step_limit: inst.step_limit,
        });
    }

    Ok(())
}

/// Deterministically builds one candidate `Instance` from `(seed, index)`
/// and nothing else; same inputs always yield a byte-identical `Instance`
/// (STEP-1 9). Returns `Err(Reject::..)` naming the failed constraint if the
/// candidate is degenerate -- this is expected and normal; callers should
/// advance `index` and try again. There is no internal retry loop: exactly
/// one evidence table, one truth, and one cost vector are drawn per call.
///
/// `budget` and `step_limit` are constructed, not guessed, and -- critically
/// -- constructed WITHOUT ever consulting the sampled `truth`:
/// `per_hypothesis_identifying_sets` computes, for every hypothesis `h`, the
/// minimum-cost probe set `S(h)` that would identify `h` if `h` were the
/// truth. Then:
///
/// - `budget = max_h cost(S(h)) + p.budget_slack`
/// - `step_limit = max_h |S(h)| + commit_overhead(variant) + p.step_slack`
///
/// Both are therefore the same value no matter which hypothesis the RNG
/// happens to draw as `truth` a few lines later -- `budget` and
/// `step_limit` are computed first, from the evidence table and costs
/// alone, and `truth` is drawn independently and used only to fill
/// `Instance::truth`. This is why the winnability end-to-end tests below
/// (which use `identifying_set` for `inst.truth` specifically, one
/// particular hypothesis among the `max` that sized `step_limit`) still
/// terminate well within `step_limit`: `S(truth)`'s own cost and length are
/// each `<=` the worst-case values `budget`/`step_limit` were sized from.
/// `step_slack` additionally banks unused actions so a `Reversible` learner
/// can afford one wrong provisional `Commit` before recovering -- see the
/// recovery test.
pub fn sample(p: &FamilyParams, seed: u64, index: u64) -> Result<Instance, Reject> {
    validate_params(p)?;

    let mut rng = seeded_rng(seed, index);
    let evidence = sample_evidence_table(p, &mut rng);
    let truth = (rng.next() % p.n_hyp as u64) as u16;
    let probe_cost = sample_costs(p, &mut rng);

    if !any_probe_varies(p.n_probe, &evidence, p.n_hyp) {
        return Err(Reject::NoDiscriminatingProbe);
    }
    if let Some(q) = injective_probe(p.n_probe, &evidence, p.n_hyp) {
        return Err(Reject::TruthLeakingProbe { probe: q });
    }

    let summaries =
        per_hypothesis_identifying_sets(p.n_probe, &evidence, p.n_hyp, |q| probe_cost[q as usize] as i64)
            .map_err(|h| Reject::NotIdentifiable { h })?;

    let shortest = summaries.iter().map(|s| s.shortest).min().unwrap();
    if shortest < p.min_depth as usize {
        return Err(Reject::DepthTooShallow {
            shortest,
            min_depth: p.min_depth,
        });
    }

    let max_cost = summaries.iter().map(|s| s.cost).max().unwrap();
    let budget = (max_cost + p.budget_slack as i64) as i32;

    let max_len = summaries.iter().map(|s| s.cost_set.len()).max().unwrap();
    let step_limit = max_len as u16 + commit_overhead(p.variant) + p.step_slack;

    let inst = Instance {
        n_hyp: p.n_hyp,
        n_probe: p.n_probe,
        evidence,
        probe_cost,
        budget,
        step_limit,
        truth,
        variant: p.variant,
        seed,
        index,
    };

    inst.validate().map_err(Reject::InvalidParams)?;
    structural_check(&inst)?;
    Ok(inst)
}

/// Which partition a *structural family* belongs to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    Train,
    Val,
    Test,
}

/// Assigns `p` to a partition using only its structural shape -- `n_hyp`,
/// `n_probe`, `n_evidence`, `min_depth`, `variant` -- never a seed (this
/// function does not even take one) and never the purely quantitative
/// cost/budget knobs (`cost_lo`, `cost_hi`, `budget_slack`, `step_slack`).
/// STEP-1 3.2's closing paragraph requires holding out *structural
/// parameter combinations*, not random seeds: two `FamilyParams` with the
/// same shape always land in the same split no matter their seed, index, or
/// cost/budget knobs; a combination assigned to `Val`/`Test` is held out of
/// `Train` unconditionally, not with some probability tied to which seeds
/// happen to get sampled.
pub fn assign_split(p: &FamilyParams) -> Split {
    let mut rng = SplitMix64::new(0xD1B5_4A32_D192_ED03);
    let fields = [
        p.n_hyp as u64,
        p.n_probe as u64,
        p.n_evidence as u64,
        p.min_depth as u64,
        match p.variant {
            Variant::Irreversible => 0,
            Variant::Reversible => 1,
        },
    ];
    for field in fields {
        rng.0 ^= field;
        rng.next();
    }
    match rng.next() % 20 {
        0..=13 => Split::Train, // 70%
        14..=16 => Split::Val,  // 15%
        _ => Split::Test,       // 15%
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{step, Action, EndReason, Status};

    fn params(variant: Variant) -> FamilyParams {
        FamilyParams {
            n_hyp: 6,
            n_probe: 5,
            // Binary evidence. A probe creates an accidental depth-1
            // shortcut for hypothesis h iff h's value on that probe is
            // UNIQUE among all n_hyp hypotheses (no other hypothesis shares
            // it) -- not, as it might seem, only when all *other*
            // hypotheses share one common value. With few evidence values
            // relative to n_hyp, collisions are forced and singletons get
            // rare: with n_evidence=2 and n_hyp=6, a probe's 6 binary draws
            // avoid any singleton value with probability 1 - P(exactly one
            // 0) - P(exactly one 1) = 1 - 6/64 - 6/64 = 81.25%, which is why
            // this pair of values was chosen over a larger evidence alphabet
            // (a larger alphabet makes singletons, and so accidental
            // min_depth violations, sharply *more* likely, not less).
            // n_evidence < n_hyp also makes an injective (truth-leaking) row
            // structurally impossible by pigeonhole.
            n_evidence: 2,
            cost_lo: 1,
            cost_hi: 3,
            budget_slack: 1,
            min_depth: 2,
            step_slack: 2,
            variant,
        }
    }

    fn assert_same_instance(a: &Instance, b: &Instance) {
        assert_eq!(a.n_hyp, b.n_hyp);
        assert_eq!(a.n_probe, b.n_probe);
        assert_eq!(a.evidence, b.evidence);
        assert_eq!(a.probe_cost, b.probe_cost);
        assert_eq!(a.budget, b.budget);
        assert_eq!(a.step_limit, b.step_limit);
        assert_eq!(a.truth, b.truth);
        assert_eq!(a.variant, b.variant);
        assert_eq!(a.seed, b.seed);
        assert_eq!(a.index, b.index);
    }

    /// Collects `n` accepted instances by scanning `index` upward.
    /// Rejection is expected and normal (see `sample`'s doc comment); this
    /// just demonstrates the intended caller pattern.
    fn sample_n(p: &FamilyParams, seed: u64, n: usize) -> Vec<Instance> {
        let mut out = Vec::with_capacity(n);
        let mut index = 0u64;
        while out.len() < n && index < 20_000 {
            if let Ok(inst) = sample(p, seed, index) {
                out.push(inst);
            }
            index += 1;
        }
        assert_eq!(
            out.len(),
            n,
            "did not find {n} accepted instances within the attempt budget"
        );
        out
    }

    /// Does `P(E | H = h)` change under a permutation of hypothesis LABELS?
    ///
    /// It must not. If two labels carry stable public meanings across
    /// instances, a policy can score well by reading the label rather than by
    /// eliminating hypotheses, and the whole family stops measuring what it
    /// was built to measure.
    #[test]
    fn evidence_distribution_is_not_conditioned_on_the_hypothesis_label() {
        let p = FamilyParams {
            n_hyp: 6, n_probe: 5, n_evidence: 2, cost_lo: 1, cost_hi: 3,
            budget_slack: 1, min_depth: 2, step_slack: 2, variant: Variant::Irreversible,
        };
        let mut seen: Vec<std::collections::HashSet<u16>> = vec![Default::default(); 6];
        let mut index = 0u64;
        let mut found = 0usize;
        while found < 400 && index < 40_000 {
            if let Ok(inst) = sample(&p, 909, index) {
                for h in 0..inst.n_hyp {
                    for q in 0..inst.n_probe {
                        seen[h as usize].insert(inst.evidence_of(q, h));
                    }
                }
                found += 1;
            }
            index += 1;
        }
        let constant: Vec<u16> = (0..6u16).filter(|&h| seen[h as usize].len() == 1).collect();
        assert!(
            constant.is_empty(),
            "hypotheses {constant:?} return a CONSTANT evidence value across {found} instances              and all probes: {seen:?}. Their public labels therefore mean something fixed, and a              policy can exploit that without eliminating anything."
        );
    }

    #[test]
    fn sample_is_deterministic_and_varies_by_index() {
        let p = params(Variant::Irreversible);
        let a1 = sample(&p, 7, 3);
        let a2 = sample(&p, 7, 3);
        match (&a1, &a2) {
            (Ok(x), Ok(y)) => assert_same_instance(x, y),
            (Err(x), Err(y)) => assert_eq!(x, y),
            _ => panic!("sample(p, 7, 3) must be deterministic across calls"),
        }

        let batch = sample_n(&p, 7, 6);
        let all_identical = batch
            .windows(2)
            .all(|w| w[0].truth == w[1].truth && w[0].evidence == w[1].evidence);
        assert!(
            !all_identical,
            "different index must not always produce an identical instance"
        );
    }

    #[test]
    fn sampled_instances_pass_validate_and_structural_check() {
        for variant in [Variant::Irreversible, Variant::Reversible] {
            let p = params(variant);
            for inst in sample_n(&p, 99, 15) {
                assert!(inst.validate().is_ok(), "Instance::validate failed: {inst:?}");
                assert!(
                    structural_check(&inst).is_ok(),
                    "structural_check failed on sampler output: {inst:?}"
                );
            }
        }
    }

    #[test]
    fn rejection_is_deterministic() {
        let p = params(Variant::Irreversible);
        let mut found = None;
        for index in 0..5000u64 {
            if let Err(e) = sample(&p, 55, index) {
                found = Some((index, e));
                break;
            }
        }
        let (index, first_err) =
            found.expect("expected at least one rejected index in the scanned range");
        let second_err = sample(&p, 55, index).unwrap_err();
        assert_eq!(first_err, second_err);
    }

    #[test]
    fn irreversible_winnable_end_to_end_batch() {
        let p = params(Variant::Irreversible);
        for inst in sample_n(&p, 123, 20) {
            let (_, ident_set) = identifying_set(
                inst.n_probe,
                &inst.evidence,
                inst.n_hyp,
                inst.truth,
                |q| inst.probe_cost[q as usize] as i64,
            )
            .expect("sampled instance must be identifiable");

            let mut st = reset(&inst);
            for &q in &ident_set {
                step(&inst, &mut st, Action::Inspect(q)).unwrap();
            }
            step(&inst, &mut st, Action::Commit(inst.truth)).unwrap();
            assert_eq!(
                st.status,
                Status::Terminated {
                    correct: true,
                    reason: EndReason::Committed
                },
                "instance {inst:?} did not win via its own identifying set {ident_set:?}"
            );
        }
    }

    #[test]
    fn reversible_winnable_end_to_end_batch_with_confirm() {
        let p = params(Variant::Reversible);
        for inst in sample_n(&p, 456, 20) {
            let (_, ident_set) = identifying_set(
                inst.n_probe,
                &inst.evidence,
                inst.n_hyp,
                inst.truth,
                |q| inst.probe_cost[q as usize] as i64,
            )
            .expect("sampled instance must be identifiable");

            let mut st = reset(&inst);
            for &q in &ident_set {
                step(&inst, &mut st, Action::Inspect(q)).unwrap();
            }
            step(&inst, &mut st, Action::Commit(inst.truth)).unwrap(); // provisional
            assert_eq!(st.status, Status::Running, "provisional commit must not terminate");
            step(&inst, &mut st, Action::Commit(inst.truth)).unwrap(); // confirm
            assert_eq!(
                st.status,
                Status::Terminated {
                    correct: true,
                    reason: EndReason::Committed
                },
                "instance {inst:?} did not win via ident set {ident_set:?} + confirm"
            );
        }
    }

    /// The leak test. Builds instances via `sample`, then -- for each --
    /// recomputes what a truth-specific rule (`budget = cost(S(h)) +
    /// slack`) would have produced for every possible value of `h`, and
    /// asserts the instance's *actual* `budget`/`step_limit` equal the
    /// worst case (max) across all of them, not the value tied to any one
    /// hypothesis (in particular, not tied to `inst.truth` specifically).
    /// `saw_genuine_variation` confirms the per-hypothesis costs actually
    /// differ across hypotheses for at least one sampled instance, so this
    /// is not a vacuous check: if `sample` reverted to
    /// `budget = cost(S(inst.truth)) + slack`, `inst.budget` would equal
    /// `old_rule_budgets[inst.truth]`, not the max, and this test would
    /// fail on the first instance where those differ.
    #[test]
    fn budget_and_step_limit_are_truth_blind() {
        let p = params(Variant::Irreversible);
        let mut saw_genuine_variation = false;
        for inst in sample_n(&p, 321, 25) {
            let mut per_hyp_cost = Vec::new();
            let mut per_hyp_len = Vec::new();
            for h in 0..inst.n_hyp {
                let (cost, set) = identifying_set(
                    inst.n_probe,
                    &inst.evidence,
                    inst.n_hyp,
                    h,
                    |q| inst.probe_cost[q as usize] as i64,
                )
                .expect("structural_check already guarantees every hypothesis is identifiable");
                per_hyp_cost.push(cost);
                per_hyp_len.push(set.len());
            }

            let old_rule_budgets: Vec<i32> = per_hyp_cost
                .iter()
                .map(|&c| (c + p.budget_slack as i64) as i32)
                .collect();
            if old_rule_budgets.iter().any(|&b| b != old_rule_budgets[0]) {
                saw_genuine_variation = true;
            }

            let expected_budget = (*per_hyp_cost.iter().max().unwrap() + p.budget_slack as i64) as i32;
            assert_eq!(
                inst.budget, expected_budget,
                "budget must equal the worst-case (max over all hypotheses) cost, not a \
                 truth-specific one"
            );

            let expected_step_limit =
                *per_hyp_len.iter().max().unwrap() as u16 + commit_overhead(Variant::Irreversible) + p.step_slack;
            assert_eq!(
                inst.step_limit, expected_step_limit,
                "step_limit must equal the worst-case (max over all hypotheses) length, not a \
                 truth-specific one"
            );
        }
        assert!(
            saw_genuine_variation,
            "test params never produced per-hypothesis cost variation across a sampled batch; \
             strengthen params to make this test meaningful"
        );
    }

    /// The single most important test in this file: the reversible/
    /// irreversible contrast that Step 1 exists to measure. Drives a forced
    /// prefix that commits to a WRONG hypothesis first (STEP-1 3.5), then
    /// checks that a `Reversible` learner can still gather evidence and
    /// commit correctly within `step_limit` (recovery is reachable), while
    /// the matching `Irreversible` instance terminates immediately and
    /// incorrectly on that same wrong commit (recovery is impossible).
    #[test]
    fn reversible_recovers_from_wrong_commit_but_irreversible_does_not() {
        let p = params(Variant::Reversible);
        for inst in sample_n(&p, 789, 15) {
            let wrong = if inst.truth == 0 { 1 } else { 0 };
            let (_, ident_for_truth) = identifying_set(
                inst.n_probe,
                &inst.evidence,
                inst.n_hyp,
                inst.truth,
                |q| inst.probe_cost[q as usize] as i64,
            )
            .expect("sampled instance must be identifiable");

            // Reversible: a wrong provisional commit first, then still
            // gather the evidence needed to identify truth and confirm
            // correctly -- this is exactly the recovery step_slack exists
            // to fund.
            let mut st = reset(&inst);
            step(&inst, &mut st, Action::Commit(wrong)).unwrap();
            assert_eq!(
                st.status,
                Status::Running,
                "a non-repeating commit must stay Running in Reversible"
            );
            for &q in &ident_for_truth {
                step(&inst, &mut st, Action::Inspect(q)).unwrap();
            }
            step(&inst, &mut st, Action::Commit(inst.truth)).unwrap(); // replaces the wrong provisional
            assert_eq!(st.status, Status::Running);
            step(&inst, &mut st, Action::Commit(inst.truth)).unwrap(); // confirm
            assert_eq!(
                st.status,
                Status::Terminated {
                    correct: true,
                    reason: EndReason::Committed
                },
                "Reversible must recover from an earlier wrong commit within step_limit"
            );

            // The matching Irreversible instance (identical evidence, costs,
            // budget; only the variant differs) cannot recover: the very
            // same wrong commit ends the episode immediately, incorrectly.
            let irrev = Instance {
                variant: Variant::Irreversible,
                ..inst.clone()
            };
            let mut st2 = reset(&irrev);
            step(&irrev, &mut st2, Action::Commit(wrong)).unwrap();
            assert_eq!(
                st2.status,
                Status::Terminated {
                    correct: false,
                    reason: EndReason::Committed
                },
                "Irreversible must terminate immediately and incorrectly on the wrong commit, \
                 with no recovery"
            );
        }
    }

    #[test]
    fn structural_check_rejects_constant_evidence_table() {
        let inst = Instance {
            n_hyp: 2,
            n_probe: 2,
            evidence: vec![5, 5, 5, 5], // both probes constant across both hyps
            probe_cost: vec![1, 1],
            budget: 10,
            step_limit: 10,
            truth: 0,
            variant: Variant::Irreversible,
            seed: 0,
            index: 0,
        };
        assert_eq!(structural_check(&inst), Err(Reject::NoDiscriminatingProbe));
    }

    #[test]
    fn structural_check_rejects_identity_table_that_encodes_truth() {
        let inst = Instance {
            n_hyp: 3,
            n_probe: 2,
            // evidence_of(q, h) == h for both probes: reading either probe
            // hands over the true hypothesis directly.
            evidence: vec![0, 1, 2, 0, 1, 2],
            probe_cost: vec![1, 1],
            budget: 10,
            step_limit: 10,
            truth: 1,
            variant: Variant::Irreversible,
            seed: 0,
            index: 0,
        };
        assert_eq!(
            structural_check(&inst),
            Err(Reject::TruthLeakingProbe { probe: 0 })
        );
    }

    /// A symmetric 4-hypothesis, 3-probe design where each probe splits the
    /// hypotheses 2-vs-2 (`[0,0,1,1]`, `[0,1,0,1]`, `[0,1,1,0]`): every
    /// pairwise combination of two probes identifies EVERY hypothesis
    /// (verified by hand), so every hypothesis has shortest depth 2
    /// (satisfies `min_depth`/`MIN_DEPTH_FLOOR`) and the same minimum
    /// identifying cost -- cheapest pair is probes 0+1, cost 1+2=3 -- for
    /// every hypothesis alike (by the design's symmetry). This isolates the
    /// budget/step_limit checks from the depth check below.
    fn symmetric_four_hyp_instance(budget: i32, step_limit: u16, variant: Variant) -> Instance {
        Instance {
            n_hyp: 4,
            n_probe: 3,
            evidence: vec![
                0, 0, 1, 1, // probe0: {h0,h1} vs {h2,h3}
                0, 1, 0, 1, // probe1: {h0,h2} vs {h1,h3}
                0, 1, 1, 0, // probe2: {h0,h3} vs {h1,h2}
            ],
            probe_cost: vec![1, 2, 4],
            budget,
            step_limit,
            truth: 0,
            variant,
            seed: 0,
            index: 0,
        }
    }

    #[test]
    fn structural_check_rejects_budget_too_small_to_identify() {
        let inst = symmetric_four_hyp_instance(2, 10, Variant::Irreversible);
        assert!(matches!(
            structural_check(&inst),
            Err(Reject::NotIdentifiableWithinBudget {
                min_cost: 3,
                budget: 2,
                ..
            })
        ));
    }

    #[test]
    fn structural_check_rejects_step_limit_too_small_to_win() {
        // budget=10 is sufficient (min cost 3 for every hypothesis), but
        // step_limit=1 is far too small: needs 2 inspects + 1 commit = 3.
        let inst = symmetric_four_hyp_instance(10, 1, Variant::Irreversible);
        assert!(matches!(
            structural_check(&inst),
            Err(Reject::NotWinnableWithinStepLimit {
                required: 3,
                step_limit: 1,
                ..
            })
        ));
    }

    #[test]
    fn split_is_structural_not_seed_based() {
        let mut p1 = params(Variant::Irreversible);
        let mut p2 = p1;
        // Vary only non-structural knobs: cost range, budget/step slack. A
        // reimplementation that hashed the whole `FamilyParams` (or, worse,
        // hashed a seed) rather than just the structural shape would fail
        // this.
        p2.cost_lo = 999;
        p2.cost_hi = 1000;
        p2.budget_slack = 42;
        p2.step_slack = 42;
        assert_eq!(assign_split(&p1), assign_split(&p2));

        // The partition function takes no seed at all -- reproducing this
        // decision from a seed hash is not even expressible against this
        // signature. Sweeping the *structural* knob `n_hyp` must reach both
        // Train and a held-out split, i.e. the function isn't constant.
        let mut saw_train = false;
        let mut saw_held_out = false;
        for n_hyp in 2..MAX_HYP_FOR_EXACT_SEARCH {
            p1.n_hyp = n_hyp;
            match assign_split(&p1) {
                Split::Train => saw_train = true,
                Split::Val | Split::Test => saw_held_out = true,
            }
        }
        assert!(
            saw_train && saw_held_out,
            "assign_split must actually partition structural space, not be constant"
        );
    }

    #[test]
    fn split_assignment_is_deterministic() {
        let p = params(Variant::Reversible);
        assert_eq!(assign_split(&p), assign_split(&p));
    }
}
