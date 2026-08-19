//! Step 1 world executor, third slice: the privileged verifier and the dense
//! teacher (STEP-1.md sections 3.6, 5, 6, 10). Reuses `lib.rs`'s frozen API
//! and `generate.rs`'s sampler for tests; does not touch `State`, does not
//! render, does not pack tokens.
//!
//! # The privileged boundary (STEP-1 3.6)
//!
//! This module is the ONLY place besides `Instance::evidence_of`'s call
//! inside `step` that is allowed to read `Instance::truth`. Its functions
//! return supervision/verification *outputs* that may be truth-derived --
//! that is the entire point of a privileged teacher -- but nothing in here
//! ever mutates `State` or feeds back into `valid_actions`. Every function
//! below takes `&Instance` and `&State` and returns a plain value; none of
//! them execute `step`.
//!
//! # STEP-1 5.2 target channels implemented
//!
//! `TeacherTargets` carries exactly three channels, each exactly computable
//! from `(inst, st)`:
//!
//! - `valid_actions`: STEP-1 5.2's "set ... of valid next actions" --
//!   delegates to `crate::valid_actions` verbatim, so it is truth-blind by
//!   construction. Bundled here so a consumer of `TeacherTargets` can
//!   interpret `preferred_actions` as a subset without a second call.
//! - `preferred_actions`: STEP-1 5.2's "teacher-preferred next-action
//!   ... distribution", represented as the SET of every action tied for
//!   optimal under the STEP-1 5.1 lexicographic policy (point 5 of 5.1
//!   forbids collapsing ties to one pick).
//! - `licenses_commitment`: STEP-1 5.2's "whether current evidence licenses
//!   commitment", true iff `consistent(inst, st)` is a singleton.
//!
//! Channels from STEP-1 5.2 deliberately NOT implemented here, because this
//! module cannot compute them exactly without guessing (STEP-1 5.2's closing
//! line: "a target must not narrate a hidden variable merely because the
//! generator can access it", and its own text: "impossible or
//! underdetermined predictions are not assigned false point targets"):
//!
//! - "well-formed action tokens": there is no renderer in this crate slice
//!   (STEP-1 7.1 puts rendering in `world-render`); nothing to compute yet.
//! - "predicted public consequence of a proposed action" / "predicted next
//!   observation": `teach` does not take a proposed action as input, only a
//!   state, so there is no single "the" action to predict a consequence
//!   for. The one case that IS exactly determined -- `evidence_of(q, truth)`
//!   for an about-to-be-inspected probe -- is deliberately left out of
//!   `TeacherTargets` too: publishing it as a labeled channel would hand the
//!   learner the answer to an action it hasn't taken yet, which is exactly
//!   the "narrate a hidden variable" failure mode this file exists to
//!   avoid, not a legitimate consequence prediction.
//! - "correction after a malformed or strategically poor action": `teach`
//!   is the forward-looking policy target, not a reactive corrector;
//!   STEP-1 5.3 puts that in the learner-conditioned regime, out of this
//!   slice's scope.
//! - "recovery when recovery is reachable": for this world family, recovery
//!   is just "keep gathering evidence and commit correctly", which
//!   `preferred_actions` already expresses; there is no separate recovery
//!   action type to target (STEP-1 3.4: `Recover` is out of scope for
//!   Step 1's acquisition family).
//!
//! `min_remaining_cost` and `outcome` (Part A) are verifier-only functions,
//! not `TeacherTargets` fields -- STEP-1 3.6 lists them as things "the
//! generator and teacher may additionally query", not as things that must be
//! bundled into every dense-teacher training example.

use crate::generate;
use crate::{valid_actions, Action, EndReason, HypId, Instance, ProbeId, State, Status, Variant};

// ---------------------------------------------------------------------
// Part A: the privileged verifier
// ---------------------------------------------------------------------

/// Bitset over `HypId` (bit `h` set iff hypothesis `h` is consistent with
/// `st.history`): `h` is consistent iff `evidence_of(q, h) == e` for every
/// `(q, e)` in `st.history`. Derived from history alone -- two states with
/// identical history always produce identical output regardless of
/// `inst.truth` -- but still lives here (not in `lib.rs`) because it is a
/// privileged summary the learner is never handed directly (STEP-1 3.6's
/// second list: "the hypotheses consistent with the complete history").
///
/// Assumes `inst.n_hyp <= 64`, mirroring `Instance::validate`'s existing
/// assumption that `n_probe <= 64` for `State::probed`; `Instance` carries
/// no such check for `n_hyp` today, so this is a documented precondition
/// rather than an enforced one.
pub fn consistent(inst: &Instance, st: &State) -> u64 {
    debug_assert!(
        inst.n_hyp <= 64,
        "consistent's u64 bitset cannot represent n_hyp > 64"
    );
    let full: u64 = if inst.n_hyp == 64 {
        u64::MAX
    } else {
        (1u64 << inst.n_hyp) - 1
    };
    let mut mask = full;
    for h in 0..inst.n_hyp {
        let bit = 1u64 << h;
        if mask & bit == 0 {
            continue;
        }
        let ruled_out = st
            .history
            .iter()
            .any(|&(q, e)| inst.evidence_of(q, h) != e);
        if ruled_out {
            mask &= !bit;
        }
    }
    mask
}

// ---------------------------------------------------------------------
// The truth-blind optimum
// ---------------------------------------------------------------------

/// Bayes-optimal action for an expert that may read the evidence table but
/// NOT `inst.truth`.
///
/// The dense teacher identifies inside ~2 binary probes because it only has to
/// isolate the hypothesis it already knows; a learner must separate whatever
/// the evidence leaves live. Imitating the teacher's cost profile therefore
/// caps a learner at a budget that is only sufficient with the answer in hand.
/// This policy is what a learner can actually follow: it reaches ~97.7% on the
/// STEP-1 family using ~2.63 probes.
///
/// `consistent` is derived from `st.history` alone, so every quantity this
/// policy branches on is available to something that never sees `truth`.
/// Backward induction over `(probed, consistent, spent, steps)` under a uniform
/// prior on the consistent set: commit now for `1/|consistent|`, or buy an
/// affordable unprobed probe and average over the evidence it may return.
pub fn truth_blind_optimal_action(inst: &Instance, st: &State) -> Action {
    let cons = consistent(inst, st);
    let fallback = Action::Commit(cons.trailing_zeros() as HypId);
    let live = cons.count_ones();
    if live <= 1 || st.step + 1 >= inst.step_limit {
        return fallback;
    }
    let mut memo = std::collections::HashMap::new();
    let mut best = 1.0 / live as f64;
    let mut chosen = fallback;
    for (q, subsets, cost) in tb_options(inst, st.probed, cons, st.spent) {
        let mut value = 0.0;
        for subset in subsets.values() {
            let weight = subset.count_ones() as f64 / live as f64;
            value += weight
                * tb_value(inst, st.probed | (1u64 << q), *subset, st.spent + cost, st.step + 1, &mut memo);
        }
        if value > best {
            best = value;
            chosen = Action::Inspect(q);
        }
    }
    chosen
}

/// Success probability of [`truth_blind_optimal_action`] played to the end,
/// and the probes it expects to buy.
pub fn truth_blind_optimal_value(inst: &Instance, st: &State) -> (f64, f64) {
    let cons = consistent(inst, st);
    let mut memo = std::collections::HashMap::new();
    let mut probe_memo = std::collections::HashMap::new();
    (
        tb_value(inst, st.probed, cons, st.spent, st.step, &mut memo),
        tb_probes(inst, st.probed, cons, st.spent, st.step, &mut memo, &mut probe_memo),
    )
}

type TbBuckets = std::collections::HashMap<u16, u64>;

/// Every affordable unprobed probe that actually splits `cons`, with the
/// partition it induces. A probe that separates nothing can only waste budget.
fn tb_options(inst: &Instance, probed: u64, cons: u64, spent: i32) -> Vec<(ProbeId, TbBuckets, i32)> {
    let mut out = Vec::new();
    for q in 0..inst.n_probe {
        if probed & (1u64 << q) != 0 {
            continue;
        }
        let cost = inst.probe_cost[q as usize];
        if spent + cost > inst.budget {
            continue;
        }
        let mut buckets: TbBuckets = std::collections::HashMap::new();
        for h in 0..inst.n_hyp {
            if cons & (1u64 << h) == 0 {
                continue;
            }
            *buckets.entry(inst.evidence_of(q, h) as u16).or_insert(0) |= 1u64 << h;
        }
        if buckets.len() >= 2 {
            out.push((q, buckets, cost));
        }
    }
    out
}

fn tb_value(
    inst: &Instance,
    probed: u64,
    cons: u64,
    spent: i32,
    steps: u16,
    memo: &mut std::collections::HashMap<(u64, u64, i32, u16), f64>,
) -> f64 {
    let live = cons.count_ones();
    if live <= 1 {
        return 1.0;
    }
    if steps + 1 >= inst.step_limit {
        return 1.0 / live as f64;
    }
    let key = (probed, cons, spent, steps);
    if let Some(&cached) = memo.get(&key) {
        return cached;
    }
    let mut best = 1.0 / live as f64;
    for (q, subsets, cost) in tb_options(inst, probed, cons, spent) {
        let mut value = 0.0;
        for subset in subsets.values() {
            let weight = subset.count_ones() as f64 / live as f64;
            value += weight * tb_value(inst, probed | (1u64 << q), *subset, spent + cost, steps + 1, memo);
        }
        if value > best {
            best = value;
        }
    }
    memo.insert(key, best);
    best
}

fn tb_probes(
    inst: &Instance,
    probed: u64,
    cons: u64,
    spent: i32,
    steps: u16,
    memo: &mut std::collections::HashMap<(u64, u64, i32, u16), f64>,
    probe_memo: &mut std::collections::HashMap<(u64, u64, i32, u16), f64>,
) -> f64 {
    let live = cons.count_ones();
    if live <= 1 || steps + 1 >= inst.step_limit {
        return 0.0;
    }
    let key = (probed, cons, spent, steps);
    if let Some(&cached) = probe_memo.get(&key) {
        return cached;
    }
    let mut best = 1.0 / live as f64;
    let mut chosen: Option<(ProbeId, TbBuckets, i32)> = None;
    for (q, subsets, cost) in tb_options(inst, probed, cons, spent) {
        let mut value = 0.0;
        for subset in subsets.values() {
            let weight = subset.count_ones() as f64 / live as f64;
            value += weight * tb_value(inst, probed | (1u64 << q), *subset, spent + cost, steps + 1, memo);
        }
        if value > best {
            best = value;
            chosen = Some((q, subsets, cost));
        }
    }
    let result = match chosen {
        None => 0.0,
        Some((q, subsets, cost)) => {
            let mut total = 1.0;
            for subset in subsets.values() {
                let weight = subset.count_ones() as f64 / live as f64;
                total += weight
                    * tb_probes(inst, probed | (1u64 << q), *subset, spent + cost, steps + 1, memo, probe_memo);
            }
            total
        }
    };
    probe_memo.insert(key, result);
    result
}

/// For hypothesis `pivot`, the set of hypotheses in `cons` (excluding
/// `pivot` itself) that a probe must still be separated from.
fn others_of(inst: &Instance, cons: u64, pivot: HypId) -> Vec<HypId> {
    (0..inst.n_hyp)
        .filter(|&h| h != pivot && cons & (1u64 << h) != 0)
        .collect()
}

/// For every unprobed probe `q`, its cost and a bitmask (indices into
/// `others`) of which members of `others` it separates from `pivot`
/// (`evidence_of(q, h) != evidence_of(q, pivot)`). Only unprobed probes are
/// candidates: an already-probed probe's evidence is already folded into
/// `cons`/history and cannot be bought again (`step` rejects
/// `AlreadyProbed`).
fn probe_candidates(
    inst: &Instance,
    st: &State,
    others: &[HypId],
    pivot: HypId,
) -> Vec<(ProbeId, i32, u32)> {
    let mut out = Vec::new();
    for q in 0..inst.n_probe {
        if st.probed & (1u64 << q) != 0 {
            continue;
        }
        let e_pivot = inst.evidence_of(q, pivot);
        let mut bits = 0u32;
        for (j, &h) in others.iter().enumerate() {
            if inst.evidence_of(q, h) != e_pivot {
                bits |= 1u32 << j;
            }
        }
        out.push((q, inst.probe_cost[q as usize], bits));
    }
    out
}

/// Among identifying sets built from `candidates` (each `(ProbeId, cost,
/// coverage-bitmask-over-`others`)`) of AT MOST `max_size` DISTINCT probes,
/// the lexicographically best `(cost, cardinality)` pair: minimum cost
/// first, and -- among every subset tied at that minimum cost -- minimum
/// cardinality. `None` if no subset of at most `max_size` candidates
/// reaches full coverage at all.
///
/// This is where `teach`'s step-budget-vs-cost conflict (see its doc
/// comment) is actually resolved: `max_size` is a hard filter applied
/// BEFORE cost is minimized, so a cheaper-but-longer set that would not fit
/// `max_size` is excluded from the search entirely, never merely
/// deprioritized. Cost is then minimized among what remains, and
/// cardinality is the tie-break within that.
///
/// 0/1-knapsack-with-a-mask-dimension DP: `dp[count][mask]` = min cost of a
/// DISTINCT subset of exactly `count` of the folded-in candidates reaching
/// `mask`. Candidates are folded in one at a time with `count` iterated in
/// decreasing order per candidate, the standard trick that stops a single
/// candidate contributing to more than one transition per pass -- so it can
/// never be picked twice, matching `step`'s `AlreadyProbed` rule.
/// `O(candidates.len() * max_size * 2^k)`.
fn min_cost_then_cardinality(
    candidates: &[(ProbeId, i32, u32)],
    k: usize,
    max_size: usize,
) -> Option<(i32, usize)> {
    let full: u32 = if k == 0 { 0 } else { ((1u64 << k) - 1) as u32 };
    if full == 0 {
        return Some((0, 0));
    }
    let cap = max_size.min(candidates.len()).min(k);
    let size = 1usize << k;
    let mut dp: Vec<Vec<i64>> = vec![vec![i64::MAX; size]; cap + 1];
    dp[0][0] = 0;
    for &(_, cost, cov) in candidates {
        for count in (0..cap).rev() {
            for mask in 0..size {
                if dp[count][mask] == i64::MAX {
                    continue;
                }
                let nm = (mask as u32 | cov) as usize;
                let nc = dp[count][mask] + cost as i64;
                if nc < dp[count + 1][nm] {
                    dp[count + 1][nm] = nc;
                }
            }
        }
    }
    let full_idx = full as usize;
    let mut best: Option<(i64, usize)> = None;
    for (count, row) in dp.iter().enumerate() {
        let c = row[full_idx];
        if c == i64::MAX {
            continue;
        }
        best = match best {
            None => Some((c, count)),
            Some((bc, bs)) if c < bc || (c == bc && count < bs) => Some((c, count)),
            Some(prev) => Some(prev),
        };
    }
    best.map(|(c, s)| (c as i32, s))
}

/// Whether some DISTINCT subset of `candidates`, of size exactly
/// `exact_count`, completes coverage from `start_mask` to the full `k`-bit
/// mask at cost exactly `target_cost`. Same DP shape as
/// `min_cost_then_cardinality`, seeded at an arbitrary starting mask instead
/// of empty coverage. `teach` uses this to test whether a specific probe
/// `q` sits on some `(cost*, cardinality*)`-optimal completion: seed
/// `start_mask = coverage(q)`, `exact_count = cardinality* - 1`,
/// `target_cost = cost* - cost(q)`, and pass every OTHER candidate (`q`
/// itself excluded, so it cannot be picked a second time here on top of the
/// forced first pick).
fn reaches_full_at(
    candidates: &[(ProbeId, i32, u32)],
    k: usize,
    start_mask: u32,
    exact_count: usize,
    target_cost: i32,
) -> bool {
    let full: u32 = if k == 0 { 0 } else { ((1u64 << k) - 1) as u32 };
    if start_mask & full == full {
        return exact_count == 0 && target_cost == 0;
    }
    if target_cost < 0 {
        return false;
    }
    let size = 1usize << k;
    let mut dp: Vec<Vec<i64>> = vec![vec![i64::MAX; size]; exact_count + 1];
    dp[0][start_mask as usize] = 0;
    for &(_, cost, cov) in candidates {
        for count in (0..exact_count).rev() {
            for mask in 0..size {
                if dp[count][mask] == i64::MAX {
                    continue;
                }
                let nm = (mask as u32 | cov) as usize;
                let nc = dp[count][mask] + cost as i64;
                if nc < dp[count + 1][nm] {
                    dp[count + 1][nm] = nc;
                }
            }
        }
    }
    dp[exact_count][full as usize] == target_cost as i64
}

/// The cheapest additional probe cost that would reduce `consistent(inst,
/// st)` to the singleton `{inst.truth}` from here, or `None` if that is not
/// reachable within the remaining budget (`inst.budget - st.spent`) --
/// either because no subset of the still-unprobed probes can finish
/// separating `inst.truth` from every other currently-consistent hypothesis
/// at all, or because the cheapest such subset costs more than what
/// remains. This is a COST-only query, deliberately unconstrained by
/// `step_limit` (unlike `teach`, which must respect both): its contract
/// (STEP-1 3.6's "minimal remaining cost") names only the budget. Reads
/// `inst.truth`: privileged verifier output, not a learner-visible
/// quantity, and not derived into anything `TeacherTargets` exposes.
pub fn min_remaining_cost(inst: &Instance, st: &State) -> Option<i32> {
    let cons = consistent(inst, st);
    if cons.count_ones() <= 1 {
        return Some(0);
    }
    let others = others_of(inst, cons, inst.truth);
    let candidates = probe_candidates(inst, st, &others, inst.truth);
    let k = others.len();
    // `max_size: k` is "unconstrained" here (never more than k probes are
    // ever needed to cover k bits, so this cap never actually binds) --
    // see `min_cost_then_cardinality`'s doc comment.
    let (cost, _cardinality) = min_cost_then_cardinality(&candidates, k, k)?;
    let remaining_budget = inst.budget - st.spent;
    if cost > remaining_budget {
        None
    } else {
        Some(cost)
    }
}

/// The RLVR-baseline scoring of a trajectory (STEP-1 6), computable from
/// `st` alone (`step` already resolved correctness against `inst.truth` at
/// termination time into `st.status`; this function does not re-read
/// `inst.truth`). This is deliberately the ONLY signal the outcome-only RLVR
/// condition receives, so it carries nothing finer-grained than: did it
/// finish, was it right, what did it spend/take, did it overrun its budget,
/// and did an irreversible wrong commitment make success impossible.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Outcome {
    /// `true` iff `st.status` is `Terminated` (either `EndReason`).
    pub terminated: bool,
    /// `st.status`'s `correct` flag; `false` while still `Running` (nothing
    /// has been scored yet).
    pub correct: bool,
    pub spent: i32,
    pub steps: u16,
    /// `st.spent > inst.budget`. `step` never actually allows this (an
    /// unaffordable `Inspect` is rejected before any mutation -- see
    /// `lib.rs`'s `Unaffordable` handling), so this is a defensive
    /// consistency check on the trajectory rather than a reachable state
    /// under normal execution.
    pub budget_violation: bool,
    /// `true` iff the episode ended via a `Commit` under `Variant::
    /// Irreversible` and that commit was wrong. This is the narrow "an
    /// irreversible wrong commitment has made success impossible" case
    /// (STEP-1 6): it does not fire for `StepLimit` endings (running out of
    /// steps/budget is a different failure mode, not "irreversible
    /// damage"), and it does not fire for `Reversible`, where a wrong
    /// `Commit` is provisional and recoverable by construction.
    pub unreachable: bool,
}

pub fn outcome(inst: &Instance, st: &State) -> Outcome {
    let (terminated, correct) = match st.status {
        Status::Terminated { correct, .. } => (true, correct),
        Status::Running => (false, false),
    };
    let budget_violation = st.spent > inst.budget;
    let unreachable = inst.variant == Variant::Irreversible
        && matches!(
            st.status,
            Status::Terminated {
                correct: false,
                reason: EndReason::Committed
            }
        );
    Outcome {
        terminated,
        correct,
        spent: st.spent,
        steps: st.step,
        budget_violation,
        unreachable,
    }
}

// ---------------------------------------------------------------------
// Part B: the dense teacher
// ---------------------------------------------------------------------

/// `Commit(h)` would immediately and incorrectly terminate the episode from
/// `st`: always true for a wrong `h` under `Variant::Irreversible` (every
/// `Commit` terminates there); true for a wrong `h` under `Variant::
/// Reversible` only when `h` repeats the current provisional `st.commitment`
/// (that specific repeat is the one that confirms and terminates -- see
/// `lib.rs::step`). `Commit(inst.truth)` is never flagged, regardless of
/// variant or provisional state.
fn would_be_wrong_terminal_commit(inst: &Instance, st: &State, a: Action) -> bool {
    match a {
        Action::Inspect(_) => false,
        Action::Commit(h) => {
            if h == inst.truth {
                return false;
            }
            match inst.variant {
                Variant::Irreversible => true,
                Variant::Reversible => st.commitment == Some(h),
            }
        }
    }
}

/// Local supervision target for one state, STEP-1 5.2. See the module doc
/// comment for which STEP-1 5.2 channels this carries and which are
/// deliberately omitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TeacherTargets {
    /// Every learner-legal action from `st` (`crate::valid_actions`
    /// verbatim); empty once terminated.
    pub valid_actions: Vec<Action>,
    /// Every action tied for optimal under the STEP-1 5.1 lexicographic
    /// policy (avoid an incorrect irreversible commitment; reach a correct
    /// terminal commitment when reachable; stay within budget; minimize
    /// declared probe cost; preserve every tie). Always a subset of
    /// `valid_actions`. Empty only when `valid_actions` is empty
    /// (terminated).
    pub preferred_actions: Vec<Action>,
    /// `true` iff `consistent(inst, st)` is a singleton, i.e. the observed
    /// history alone pins down a unique hypothesis.
    pub licenses_commitment: bool,
}

/// The lexicographic dense-teacher policy (STEP-1 5.1). Reads `inst.truth`
/// (this is the privileged teacher, not the learner-visible path); its
/// OUTPUT may be truth-derived, but it never mutates `st` and is never fed
/// into `valid_actions` or `State`.
///
/// - If already terminated (`valid_actions` empty), there is nothing to
///   propose.
/// - If `consistent(inst, st)` is already a singleton, that hypothesis is
///   `inst.truth` by construction (truth is always in `consistent`'s output,
///   and a singleton has only one member), so the unique optimal action is
///   `Commit(that hypothesis)`. This is correct whether it is the first
///   `Commit` (which, under `Variant::Reversible`, only sets the
///   provisional selection and stays `Running`) or a repeat (which
///   confirms and terminates): `teach` does not special-case which, because
///   `lib.rs::step`'s own transition rule already handles it -- calling
///   `teach` again from the resulting state naturally proposes the same
///   `Commit` a second time, which is then the terminating confirm.
/// - Otherwise, `teach` needs a probe plan: which unprobed probes to
///   recommend inspecting toward identifying `inst.truth`.
///
/// # The cost-vs-cardinality conflict, and which one wins
///
/// STEP-1 5.1's priorities are (3) remain within the budget, (4) minimize
/// declared probe cost. Two different budgets are live simultaneously here:
/// `inst.budget` (a cost budget) and `inst.step_limit` (a step-count
/// budget -- "remain within the budget" applies to it too, it is still a
/// budget). `generate::sample` sizes these from two DIFFERENT optimization
/// criteria over the identifying-set search (see its doc comment):
/// `budget` from the worst-case MINIMUM-COST identifying set's cost, but
/// `step_limit` from that SAME cost-minimal set's CARDINALITY (not from the
/// minimum-CARDINALITY set). Those two sets need not coincide -- an
/// instance can have a cheap identifying set that needs many probes, and a
/// pricier one that needs few -- so a teacher that only minimizes cost can
/// legitimately need more steps than `step_limit` budgeted for, even while
/// staying inside `inst.budget` the whole time. Left unfixed, this
/// surfaces as a rare, unexplained "teacher loses" case that would look
/// like a learner-side defect during training and would only actually be a
/// teacher-side one.
///
/// The resolution: `step_limit` feasibility is a hard FILTER applied
/// BEFORE cost is minimized (`min_cost_then_cardinality`'s `max_size`
/// parameter), not merely a tie-break preference. Concretely: only probe
/// sets of cardinality at most `(remaining steps - commit overhead)` are
/// even considered; cost is minimized among THOSE; cardinality is the
/// tie-break among cost-ties. So if the unconstrained cost-minimal set does
/// not fit the remaining steps but a slightly costlier, shorter set does,
/// `teach` proposes the shorter one -- priority 3 (step-budget feasibility)
/// outranks priority 4 (cost) exactly when they actually conflict, and
/// otherwise (the common case) cost still wins as STEP-1 5.1 orders it.
///
/// Every unprobed probe that lies on SOME `(cost*, cardinality*)`-optimal
/// completion (and fits the remaining cost budget) is proposed, tied --
/// point 5 of STEP-1 5.1 is not relaxed by the tie-break at point 4.
pub fn teach(inst: &Instance, st: &State) -> TeacherTargets {
    let valid = valid_actions(inst, st);
    let cons = consistent(inst, st);
    let licenses_commitment = cons.count_ones() == 1;

    if valid.is_empty() {
        return TeacherTargets {
            valid_actions: valid,
            preferred_actions: Vec::new(),
            licenses_commitment,
        };
    }

    if licenses_commitment {
        let h = cons.trailing_zeros() as HypId;
        return TeacherTargets {
            valid_actions: valid,
            preferred_actions: vec![Action::Commit(h)],
            licenses_commitment: true,
        };
    }

    let others = others_of(inst, cons, inst.truth);
    let candidates = probe_candidates(inst, st, &others, inst.truth);
    let k = others.len();
    let remaining_budget = inst.budget - st.spent;

    // Reserve room for the eventual commit(s) (1 for Irreversible, 2 for
    // Reversible -- provisional then confirm) so the probe plan itself
    // never eats the steps needed to actually terminate.
    let overhead = generate::commit_overhead(inst.variant) as i64;
    let remaining_steps = inst.step_limit as i64 - st.step as i64;
    let max_probes: usize = (remaining_steps - overhead).max(0) as usize;

    let mut preferred: Vec<Action> = Vec::new();
    if let Some((cost_star, size_star)) = min_cost_then_cardinality(&candidates, k, max_probes) {
        if cost_star <= remaining_budget {
            for &(q, cost, cov) in &candidates {
                if cost > cost_star {
                    continue; // a single probe can never cost more than the whole optimal set
                }
                let rest: Vec<(ProbeId, i32, u32)> = candidates
                    .iter()
                    .copied()
                    .filter(|&(id, _, _)| id != q)
                    .collect();
                if reaches_full_at(&rest, k, cov, size_star - 1, cost_star - cost) {
                    preferred.push(Action::Inspect(q));
                }
            }
        }
    }

    if preferred.is_empty() {
        // Full identification is not reachable from here within BOTH
        // budgets at once (structurally, or on cost, or on steps). This
        // should not arise on any trajectory that has always followed
        // `teach`'s own recommendations from a sampler-produced instance:
        // by construction every action `teach` proposes stays on a
        // `(cost*, cardinality*)`-optimal completion path for the
        // then-current state, so neither remaining budget ever falls
        // short of what `teach` itself still needs. Handled defensively
        // rather than asserting a preference this function cannot justify
        // (STEP-1 5.2: "impossible or underdetermined predictions are not
        // assigned false point targets"): fall back to every valid action
        // priority 1 still permits.
        preferred = valid
            .iter()
            .copied()
            .filter(|&a| !would_be_wrong_terminal_commit(inst, st, a))
            .collect();
    }

    TeacherTargets {
        valid_actions: valid,
        preferred_actions: preferred,
        licenses_commitment: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{sample, FamilyParams};
    use crate::{reset, step};

    fn small_params(variant: Variant) -> FamilyParams {
        FamilyParams {
            n_hyp: 6,
            n_probe: 5,
            n_evidence: 2,
            cost_lo: 1,
            cost_hi: 3,
            budget_slack: 1,
            min_depth: 2,
            step_slack: 2,
            variant,
        }
    }

    fn sample_n(p: &FamilyParams, seed: u64, n: usize) -> Vec<Instance> {
        let mut out = Vec::with_capacity(n);
        let mut index = 0u64;
        while out.len() < n && index < 20_000 {
            if let Ok(inst) = sample(p, seed, index) {
                out.push(inst);
            }
            index += 1;
        }
        assert_eq!(out.len(), n, "did not find {n} accepted instances");
        out
    }

    /// Two hypotheses, two probes, symmetric evidence: hyp0 -> {a,b}=(0,0),
    /// hyp1 -> (1,1). Each probe alone fully identifies either hypothesis at
    /// the same cost, so they are exchangeable -- the anti-unique-trace
    /// case.
    fn symmetric_instance(variant: Variant) -> Instance {
        Instance {
            n_hyp: 2,
            n_probe: 2,
            // probe0: hyp0->0, hyp1->1 ; probe1: hyp0->0, hyp1->1
            evidence: vec![0, 1, 0, 1],
            probe_cost: vec![2, 2],
            budget: 10,
            step_limit: 10,
            truth: 0,
            variant,
            seed: 0,
            index: 0,
        }
    }

    #[test]
    fn consistent_contains_truth_and_shrinks_monotonically() {
        for inst in sample_n(&small_params(Variant::Irreversible), 11, 10) {
            let mut st = reset(&inst);
            let mut prev = consistent(&inst, &st);
            assert_eq!(prev & (1u64 << inst.truth), 1u64 << inst.truth);
            for q in 0..inst.n_probe {
                if inst.probe_cost[q as usize] > inst.budget - st.spent {
                    continue;
                }
                step(&inst, &mut st, Action::Inspect(q)).unwrap();
                let now = consistent(&inst, &st);
                // Monotonic shrink: `now` is a subset of `prev`.
                assert_eq!(now & !prev, 0, "consistent set must never grow");
                assert_eq!(
                    now & (1u64 << inst.truth),
                    1u64 << inst.truth,
                    "truth must always remain consistent"
                );
                prev = now;
            }
        }
    }

    /// The opening state rules nothing out, so the information bound on
    /// probe count applies to every episode rather than to some of them.
    ///
    /// This is load-bearing outside the crate: it is the assumption behind
    /// reading a probe count as under- or over-probing at all. The second
    /// half keeps the first from being vacuous -- a family whose probes
    /// never separated anything would satisfy "all live at reset" and make
    /// the measurement meaningless.
    #[test]
    fn nothing_is_ruled_out_before_the_first_probe() {
        let mut any_probe_separated = false;
        for inst in sample_n(&small_params(Variant::Irreversible), 23, 32) {
            let mut st = reset(&inst);
            assert_eq!(
                consistent(&inst, &st).count_ones(),
                u32::from(inst.n_hyp),
                "the opening observation carries no evidence, so every hypothesis must be live"
            );
            for q in 0..inst.n_probe {
                if inst.probe_cost[q as usize] > inst.budget - st.spent {
                    continue;
                }
                let before = consistent(&inst, &st).count_ones();
                step(&inst, &mut st, Action::Inspect(q)).unwrap();
                any_probe_separated |= consistent(&inst, &st).count_ones() < before;
            }
        }
        assert!(
            any_probe_separated,
            "no probe ever shrank the consistent set; the reset invariant above would be vacuous"
        );
    }

    #[test]
    fn min_remaining_cost_reaches_zero_once_licensed() {
        let inst = symmetric_instance(Variant::Irreversible);
        let mut st = reset(&inst);
        assert!(min_remaining_cost(&inst, &st).is_some());
        assert_ne!(min_remaining_cost(&inst, &st), Some(0));
        step(&inst, &mut st, Action::Inspect(0)).unwrap();
        assert_eq!(min_remaining_cost(&inst, &st), Some(0));
        assert_eq!(consistent(&inst, &st).count_ones(), 1);
    }

    /// Drives real sampled episodes with `teach`'s own recommendations and
    /// checks, at every step, that every proposed action is valid.
    #[test]
    fn every_proposed_action_is_valid() {
        for variant in [Variant::Irreversible, Variant::Reversible] {
            for inst in sample_n(&small_params(variant), 22, 12) {
                let mut st = reset(&inst);
                let mut guard = 0;
                loop {
                    guard += 1;
                    assert!(guard < 1000, "runaway episode for {inst:?}");
                    let targets = teach(&inst, &st);
                    let real_valid = valid_actions(&inst, &st);
                    for &a in &targets.preferred_actions {
                        assert!(
                            real_valid.contains(&a),
                            "teacher proposed {a:?} not in valid_actions for {inst:?}"
                        );
                    }
                    if real_valid.is_empty() {
                        break;
                    }
                    let a = targets.preferred_actions[0];
                    step(&inst, &mut st, a).unwrap();
                }
            }
        }
    }

    /// The central teacher test: following `teach`'s own recommendations to
    /// termination must always win, on both variants.
    #[test]
    fn teacher_play_always_wins() {
        for variant in [Variant::Irreversible, Variant::Reversible] {
            for inst in sample_n(&small_params(variant), 33, 25) {
                let mut st = reset(&inst);
                let mut guard = 0;
                loop {
                    guard += 1;
                    assert!(guard < 1000, "runaway episode for {inst:?}");
                    let targets = teach(&inst, &st);
                    if targets.preferred_actions.is_empty() {
                        break; // terminated
                    }
                    // Priority 1: never an incorrect irreversible commitment.
                    for &a in &targets.preferred_actions {
                        assert!(
                            !would_be_wrong_terminal_commit(&inst, &st, a),
                            "teacher proposed a wrong terminating commit {a:?} for {inst:?}"
                        );
                    }
                    let a = targets.preferred_actions[0];
                    step(&inst, &mut st, a).unwrap();
                }
                let out = outcome(&inst, &st);
                assert!(
                    out.terminated,
                    "teacher play did not terminate for {inst:?}: {st:?}"
                );
                assert!(
                    out.correct,
                    "teacher play did not win for {inst:?}: final state {st:?}"
                );
            }
        }
    }

    /// Regression stress test for the cost-minimal-vs-cardinality-minimal
    /// step-budget conflict `teach`'s doc comment describes:
    /// `generate::sample` sizes `budget` from the worst-case MINIMUM-COST
    /// identifying set's cost, but `step_limit` from that same set's
    /// CARDINALITY -- not from the minimum-CARDINALITY set -- so a teacher
    /// that only minimizes cost can legitimately need more steps than
    /// `step_limit` allows. `step_slack: 0` removes the safety margin that
    /// could otherwise paper over the gap, and a wide `cost_lo`/`cost_hi`
    /// spread makes the cost-minimal and cardinality-minimal identifying
    /// sets more likely to actually diverge (a single expensive probe
    /// competing against several cheap ones). Several seeds and a larger
    /// family (`n_hyp: 10, n_probe: 8`, comfortably inside
    /// `MAX_HYP_FOR_EXACT_SEARCH == 20`) widen the search for any
    /// unwinnable instance. If `teach` only minimized cost (the pre-fix
    /// behavior), this test fails.
    #[test]
    fn teacher_wins_stress_wide_cost_spread_zero_slack() {
        let seeds: [u64; 5] = [101, 202, 303, 404, 505];
        let instances_per_seed = 15;
        let mut total = 0usize;
        for variant in [Variant::Irreversible, Variant::Reversible] {
            let p = FamilyParams {
                n_hyp: 10,
                n_probe: 8,
                n_evidence: 3,
                cost_lo: 1,
                cost_hi: 25,
                budget_slack: 1,
                min_depth: 2,
                step_slack: 0,
                variant,
            };
            for &seed in &seeds {
                for inst in sample_n(&p, seed, instances_per_seed) {
                    total += 1;
                    let mut st = reset(&inst);
                    let mut guard = 0;
                    loop {
                        guard += 1;
                        assert!(guard < 1000, "runaway episode for {inst:?}");
                        let targets = teach(&inst, &st);
                        if targets.preferred_actions.is_empty() {
                            break; // terminated
                        }
                        for &a in &targets.preferred_actions {
                            assert!(
                                !would_be_wrong_terminal_commit(&inst, &st, a),
                                "teacher proposed a wrong terminating commit {a:?} for {inst:?}"
                            );
                        }
                        let a = targets.preferred_actions[0];
                        step(&inst, &mut st, a).unwrap();
                    }
                    let out = outcome(&inst, &st);
                    assert!(
                        out.terminated,
                        "teacher play did not terminate: seed={} index={} variant={:?}\n{inst:?}\nfinal state: {st:?}",
                        inst.seed, inst.index, inst.variant
                    );
                    assert!(
                        out.correct,
                        "teacher play LOST: seed={} index={} variant={:?}\n{inst:?}\nfinal state: {st:?}",
                        inst.seed, inst.index, inst.variant
                    );
                }
            }
        }
        assert_eq!(total, seeds.len() * instances_per_seed * 2);
        eprintln!(
            "teacher_wins_stress_wide_cost_spread_zero_slack: exercised {total} instances \
             ({} seeds x {instances_per_seed} instances x 2 variants)",
            seeds.len()
        );
    }

    /// Anti-unique-trace test: with two exchangeable probes, both must
    /// appear in `preferred_actions` at reset.
    #[test]
    fn ties_are_all_represented() {
        let inst = symmetric_instance(Variant::Irreversible);
        let st = reset(&inst);
        let targets = teach(&inst, &st);
        assert!(!targets.licenses_commitment);
        assert!(targets.preferred_actions.contains(&Action::Inspect(0)));
        assert!(targets.preferred_actions.contains(&Action::Inspect(1)));
        assert_eq!(targets.preferred_actions.len(), 2);
    }

    #[test]
    fn licensed_commitment_targets_the_unique_consistent_hypothesis() {
        let inst = symmetric_instance(Variant::Irreversible);
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Inspect(0)).unwrap(); // evidence 0 -> truth 0
        let targets = teach(&inst, &st);
        assert!(targets.licenses_commitment);
        assert_eq!(targets.preferred_actions, vec![Action::Commit(0)]);
    }

    #[test]
    fn teacher_never_proposes_incorrect_irreversible_commitment() {
        let inst = symmetric_instance(Variant::Irreversible);
        let st = reset(&inst); // not yet licensed
        let targets = teach(&inst, &st);
        assert!(
            targets
                .preferred_actions
                .iter()
                .all(|a| !matches!(a, Action::Commit(_))),
            "teacher must not propose any commit before evidence licenses one"
        );
    }

    /// The reversible confirm must actually be reached, not stalled on
    /// forever: following `teach` twice in a row after licensing terminates.
    #[test]
    fn reversible_confirm_is_reached_not_stalled() {
        let inst = symmetric_instance(Variant::Reversible);
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Inspect(0)).unwrap(); // licenses hyp 0
        let first = teach(&inst, &st);
        assert!(first.licenses_commitment);
        assert_eq!(first.preferred_actions, vec![Action::Commit(0)]);
        step(&inst, &mut st, Action::Commit(0)).unwrap(); // provisional, stays Running
        assert_eq!(st.status, Status::Running);

        let second = teach(&inst, &st);
        assert_eq!(second.preferred_actions, vec![Action::Commit(0)]);
        step(&inst, &mut st, Action::Commit(0)).unwrap(); // repeat -> confirm
        assert_eq!(
            st.status,
            Status::Terminated {
                correct: true,
                reason: EndReason::Committed
            }
        );
    }

    #[test]
    fn outcome_agrees_with_terminal_status() {
        let inst = symmetric_instance(Variant::Irreversible);
        let mut st = reset(&inst);
        let running = outcome(&inst, &st);
        assert!(!running.terminated);
        assert!(!running.unreachable);

        step(&inst, &mut st, Action::Commit(1)).unwrap(); // wrong (truth = 0), terminates
        let out = outcome(&inst, &st);
        match st.status {
            Status::Terminated { correct, .. } => assert_eq!(out.correct, correct),
            Status::Running => panic!("expected terminated"),
        }
        assert!(out.terminated);
        assert!(!out.correct);
        assert!(out.unreachable, "wrong irreversible commit must be flagged unreachable");
        assert!(!out.budget_violation);
    }

    #[test]
    fn outcome_does_not_flag_unreachable_for_reversible_wrong_provisional() {
        let inst = symmetric_instance(Variant::Reversible);
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Commit(1)).unwrap(); // wrong provisional, recoverable
        assert_eq!(st.status, Status::Running);
        let out = outcome(&inst, &st);
        assert!(!out.terminated);
        assert!(!out.unreachable);
    }

    #[test]
    fn outcome_agrees_with_executor_across_sampled_batch() {
        for variant in [Variant::Irreversible, Variant::Reversible] {
            for inst in sample_n(&small_params(variant), 44, 10) {
                let mut st = reset(&inst);
                let mut guard = 0;
                loop {
                    guard += 1;
                    assert!(guard < 1000);
                    let targets = teach(&inst, &st);
                    if targets.preferred_actions.is_empty() {
                        break;
                    }
                    step(&inst, &mut st, targets.preferred_actions[0]).unwrap();
                }
                let out = outcome(&inst, &st);
                match st.status {
                    Status::Terminated { correct, .. } => {
                        assert!(out.terminated);
                        assert_eq!(out.correct, correct);
                    }
                    Status::Running => assert!(!out.terminated),
                }
                assert_eq!(out.spent, st.spent);
                assert_eq!(out.steps, st.step);
            }
        }
    }
}
