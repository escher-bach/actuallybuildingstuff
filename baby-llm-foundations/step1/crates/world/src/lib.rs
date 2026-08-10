//! Step 1 world executor, first slice: a single acquisition-under-commitment
//! world (STEP-1.md sections 3, 7.1, 7.4, 9, 10). Hand-built `Instance`s only
//! — no sampler, no teacher, no renderer, no bindings. Std only.
//!
//! Observation boundary (STEP-1.md 3.6): `valid_actions` never reads
//! `Instance::truth`. Evidence reaches the learner only through
//! `State::history`, populated exclusively by executed `Inspect` actions.

pub mod generate;
pub mod render;
pub mod teacher;

/// Identifies a hypothesis within an `Instance`. Values are `0..n_hyp`.
pub type HypId = u16;
/// Identifies a probe within an `Instance`. Values are `0..n_probe`.
pub type ProbeId = u16;
/// Identifies a point in the (finite, instance-declared) evidence alphabet.
pub type EvidenceId = u16;

/// Process variant: whether `Commit` removes future options immediately.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Variant {
    /// `Commit(h)` terminates the episode immediately.
    Irreversible,
    /// `Commit(h)` records a provisional selection unless it repeats the
    /// current provisional selection, in which case it confirms and
    /// terminates. See `step` for the exact rule.
    Reversible,
}

/// A learner-typed intervention.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Action {
    /// Inspect probe `q`: pay its cost, learn the evidence it returns for
    /// the latent truth.
    Inspect(ProbeId),
    /// Commit to hypothesis `h`. Effect depends on `Variant`; see `step`.
    Commit(HypId),
}

/// An immutable compiled instance of the world family. All fields are
/// produced by generation/compilation; this crate does not build them from
/// raw parameters — a later `world-generate` agent owns that.
#[derive(Debug, Clone)]
pub struct Instance {
    pub n_hyp: u16,
    pub n_probe: u16,
    /// Dense evidence table indexed `[probe * n_hyp + hyp]`. Deterministic:
    /// `evidence_of(q, h)` always returns the same value for a compiled
    /// instance.
    pub evidence: Vec<EvidenceId>,
    /// Cost to inspect probe `q`, indexed by `ProbeId`.
    pub probe_cost: Vec<i32>,
    /// Total cost budget available across the episode. Gates `Inspect`
    /// affordability only — it has no other effect on continuation.
    pub budget: i32,
    /// Maximum number of executed actions (`Inspect` or `Commit`) before the
    /// episode is forced to end. Bounds episode length independently of
    /// `budget`, since `Commit` is free and a `Reversible` non-confirming
    /// `Commit` never terminates on its own (see `step`). Must be nonzero.
    pub step_limit: u16,
    /// The latent true hypothesis. PRIVILEGED: must never be reachable from
    /// `valid_actions` or any learner-visible path except through evidence
    /// actually returned by executed probes.
    pub truth: HypId,
    pub variant: Variant,
    pub seed: u64,
    pub index: u64,
}

impl Instance {
    /// Deterministic evidence probe `q` would return for hypothesis `h`.
    /// Callers are responsible for `q < n_probe` and `h < n_hyp`; this is an
    /// unchecked hot-path index, matching STEP-1 7.4 (no allocation, no
    /// validation on the per-step path).
    pub fn evidence_of(&self, q: ProbeId, h: HypId) -> EvidenceId {
        self.evidence[q as usize * self.n_hyp as usize + h as usize]
    }

    /// Checks internal shape only: table length, truth in range, costs
    /// non-negative, and that `n_probe` fits the 64-bit `probed` bitset used
    /// by `State`. Does NOT check degeneracy (distinguishability,
    /// reachability of truth within budget, etc.) — that is structural
    /// validation owned by a later generator/validator agent.
    pub fn validate(&self) -> Result<(), String> {
        if self.n_probe > 64 {
            return Err(format!(
                "n_probe {} exceeds the 64-bit probed-bitset capacity",
                self.n_probe
            ));
        }
        let expected_len = self.n_probe as usize * self.n_hyp as usize;
        if self.evidence.len() != expected_len {
            return Err(format!(
                "evidence table has length {}, expected n_probe*n_hyp = {}",
                self.evidence.len(),
                expected_len
            ));
        }
        if self.probe_cost.len() != self.n_probe as usize {
            return Err(format!(
                "probe_cost has length {}, expected n_probe = {}",
                self.probe_cost.len(),
                self.n_probe
            ));
        }
        if self.truth >= self.n_hyp {
            return Err(format!(
                "truth {} is out of range for n_hyp {}",
                self.truth, self.n_hyp
            ));
        }
        if self.probe_cost.iter().any(|&c| c < 0) {
            return Err("probe_cost contains a negative cost".to_string());
        }
        if self.step_limit == 0 {
            return Err("step_limit must be nonzero".to_string());
        }
        Ok(())
    }
}

/// Why an episode ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndReason {
    /// A `Commit` terminated the episode (irreversible commit, or a
    /// reversible confirm).
    Committed,
    /// `step` reached `Instance::step_limit` after an executed action
    /// without a terminating `Commit`. Scored against the provisional
    /// `commitment` if one is on record, else incorrect. See `step`.
    StepLimit,
}

/// Terminal/continuation status of an episode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Running,
    Terminated { correct: bool, reason: EndReason },
}

/// Persistent, learner-visible-plus-bookkeeping runtime state (STEP-1
/// 3.3). Does not itself store `truth`; correctness is computed by `step`
/// from `Instance::truth` at termination time and only the boolean result is
/// recorded here.
#[derive(Debug, Clone, PartialEq)]
pub struct State {
    /// `(probe, evidence)` pairs in the order probes were inspected. The
    /// only channel through which truth-derived information reaches the
    /// learner.
    pub history: Vec<(ProbeId, EvidenceId)>,
    /// Bitset of probes already inspected (bit `q` set iff probe `q` has
    /// been used). Requires `n_probe <= 64`.
    pub probed: u64,
    pub spent: i32,
    pub step: u16,
    /// Provisional selection in `Reversible`; the final selection once
    /// `Terminated` via `Committed` in either variant.
    pub commitment: Option<HypId>,
    pub status: Status,
}

/// Rejected-action outcomes. Rejection never mutates `State`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepError {
    OutOfRange,
    AlreadyProbed,
    Unaffordable,
    /// The episode was already `Terminated`; acting on it is an error, not
    /// a no-op.
    Terminated,
}

/// Fresh `Running` state for `inst`, before any action.
pub fn reset(inst: &Instance) -> State {
    debug_assert!(
        inst.step_limit > 0,
        "step_limit must be nonzero (see Instance::validate)"
    );
    State {
        history: Vec::new(),
        probed: 0,
        spent: 0,
        step: 0,
        commitment: None,
        status: Status::Running,
    }
}

/// Learner-visible legal actions from `st`. Never reads `inst.truth` (STEP-1
/// 3.6 observation boundary). Empty once `Terminated` (acting on a terminal
/// state is an error elsewhere, not silently a no-op here).
///
/// `Commit(h)` is always listed for every `h` in `0..n_hyp`: committing has
/// no declared cost and is never blocked by budget, so it is always
/// affordable. `Inspect(q)` is listed only for unprobed `q` whose cost fits
/// the remaining budget.
pub fn valid_actions(inst: &Instance, st: &State) -> Vec<Action> {
    if !matches!(st.status, Status::Running) {
        return Vec::new();
    }
    let remaining = inst.budget - st.spent;
    let mut actions = Vec::new();
    for q in 0..inst.n_probe {
        if st.probed & (1u64 << q) == 0 && inst.probe_cost[q as usize] <= remaining {
            actions.push(Action::Inspect(q));
        }
    }
    for h in 0..inst.n_hyp {
        actions.push(Action::Commit(h));
    }
    actions
}

/// Applies `a` to `st` in place under `inst`'s transition rule.
///
/// # Termination rule (documented here because STEP-1 leaves the exact
/// consistency condition to the implementer)
///
/// `Commit(h)`:
/// - `Irreversible`: always terminates immediately; `correct = (h ==
///   truth)`, `reason = Committed`. Removes all future options — any
///   subsequent `step` call returns `StepError::Terminated`.
/// - `Reversible`: if `st.commitment == Some(h)` (repeating the current
///   provisional choice), this **confirms**: terminates exactly as above.
///   Otherwise it replaces (or sets) the provisional `commitment` and stays
///   `Running` — inspection and later revision remain reachable.
///
/// `Commit` is always affordable (zero cost, no budget gate) and, in
/// `Reversible`, a non-repeating `Commit` never terminates on its own — so
/// `budget` and probe affordability cannot be what ends an episode. Budget
/// does exactly one job: gating whether a given `Inspect` is affordable.
/// Running out of affordable probes just means no further evidence can be
/// bought; the learner still has to commit, and `Commit` (with `h ==` the
/// current provisional selection, or immediately in `Irreversible`) is how
/// that happens. Without a separate cap, a `Reversible` learner that never
/// repeats a `Commit` (e.g. alternating between two hypotheses) would run
/// forever and overflow `State::step` (`u16`). `Instance::step_limit` (STEP-1
/// 3.1's "step or cost budget") exists for exactly this: after the action
/// above, if the episode is still `Running` and `st.step >= inst.step_limit`,
/// it ends now — scored against the provisional `commitment` if one is on
/// record (`reason = StepLimit`, `correct = (h == truth)`), else incorrect
/// with no commitment to score.
///
/// So an episode ends in exactly two ways: an explicit terminating `Commit`,
/// or hitting `step_limit`. Nothing else auto-terminates it.
///
/// `Inspect(q)`: invalid (no state change) if `q` is out of range, already
/// probed, or unaffordable under the remaining budget, in that check order.
/// Otherwise charges `probe_cost[q]`, appends `(q, evidence_of(q, truth))`
/// to `history`, sets the `probed` bit, and increments `step`. `step` is
/// incremented for every executed action (both `Inspect` and `Commit`) as a
/// general turn counter — the spec's transition-semantics prose only spells
/// this out for `Inspect`; treating it as a generic action counter is the
/// choice made here.
///
/// Acting on a `Terminated` state always returns `Err(StepError::Terminated)`
/// before any other check.
pub fn step(inst: &Instance, st: &mut State, a: Action) -> Result<(), StepError> {
    if !matches!(st.status, Status::Running) {
        return Err(StepError::Terminated);
    }

    match a {
        Action::Inspect(q) => {
            if q as usize >= inst.n_probe as usize {
                return Err(StepError::OutOfRange);
            }
            if st.probed & (1u64 << q) != 0 {
                return Err(StepError::AlreadyProbed);
            }
            let cost = inst.probe_cost[q as usize];
            let remaining = inst.budget - st.spent;
            if cost > remaining {
                return Err(StepError::Unaffordable);
            }
            st.spent += cost;
            let e = inst.evidence_of(q, inst.truth);
            st.history.push((q, e));
            st.probed |= 1u64 << q;
            st.step += 1;
        }
        Action::Commit(h) => {
            if h as usize >= inst.n_hyp as usize {
                return Err(StepError::OutOfRange);
            }
            st.step += 1;
            match inst.variant {
                Variant::Irreversible => {
                    st.commitment = Some(h);
                    let correct = h == inst.truth;
                    st.status = Status::Terminated {
                        correct,
                        reason: EndReason::Committed,
                    };
                    return Ok(());
                }
                Variant::Reversible => {
                    if st.commitment == Some(h) {
                        let correct = h == inst.truth;
                        st.status = Status::Terminated {
                            correct,
                            reason: EndReason::Committed,
                        };
                        return Ok(());
                    } else {
                        st.commitment = Some(h);
                    }
                }
            }
        }
    }

    if matches!(st.status, Status::Running) && st.step >= inst.step_limit {
        st.status = match st.commitment {
            Some(h) => Status::Terminated {
                correct: h == inst.truth,
                reason: EndReason::StepLimit,
            },
            None => Status::Terminated {
                correct: false,
                reason: EndReason::StepLimit,
            },
        };
    }

    Ok(())
}

/// Deterministically reconstructs the `State` reached by executing `actions`
/// in order from a fresh `reset`. Returns the first error encountered
/// (subsequent actions are not attempted), matching `step`'s behavior.
pub fn replay(inst: &Instance, actions: &[Action]) -> Result<State, StepError> {
    let mut st = reset(inst);
    for &a in actions {
        step(inst, &mut st, a)?;
    }
    Ok(st)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 2 hypotheses, 2 probes, both discriminating, budget covers both.
    fn small_instance(variant: Variant) -> Instance {
        Instance {
            n_hyp: 2,
            n_probe: 2,
            // probe 0: hyp0 -> 10, hyp1 -> 11
            // probe 1: hyp0 -> 20, hyp1 -> 21
            evidence: vec![10, 11, 20, 21],
            probe_cost: vec![1, 1],
            budget: 5,
            step_limit: 10,
            truth: 0,
            variant,
            seed: 42,
            index: 0,
        }
    }

    #[test]
    fn instance_validates() {
        assert!(small_instance(Variant::Irreversible).validate().is_ok());
    }

    #[test]
    fn validate_rejects_zero_step_limit() {
        let inst = Instance {
            step_limit: 0,
            ..small_instance(Variant::Irreversible)
        };
        assert!(inst.validate().is_err());
    }

    #[test]
    fn deterministic_replay() {
        let inst = small_instance(Variant::Reversible);
        let actions = [
            Action::Inspect(0),
            Action::Inspect(1),
            Action::Commit(0),
            Action::Commit(0), // confirm
        ];
        let st1 = replay(&inst, &actions).unwrap();
        let st2 = replay(&inst, &actions).unwrap();
        assert_eq!(st1, st2);
        assert_eq!(
            st1.status,
            Status::Terminated {
                correct: true,
                reason: EndReason::Committed
            }
        );
        assert_eq!(st1.history, vec![(0, 10), (1, 20)]);
    }

    #[test]
    fn cost_budget_conservation() {
        let inst = small_instance(Variant::Reversible);
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Inspect(0)).unwrap();
        assert_eq!(st.spent, 1);
        assert!(st.spent <= inst.budget);
        step(&inst, &mut st, Action::Inspect(1)).unwrap();
        assert_eq!(st.spent, 2);
        assert!(st.spent <= inst.budget);
        let expected: i32 = st
            .history
            .iter()
            .map(|&(q, _)| inst.probe_cost[q as usize])
            .sum();
        assert_eq!(st.spent, expected);
    }

    #[test]
    fn irreversible_commit_removes_futures() {
        let inst = small_instance(Variant::Irreversible);
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Inspect(0)).unwrap();
        step(&inst, &mut st, Action::Commit(1)).unwrap(); // wrong (truth = 0)
        assert_eq!(
            st.status,
            Status::Terminated {
                correct: false,
                reason: EndReason::Committed
            }
        );
        // Any further action errors, including another probe or commit.
        assert_eq!(
            step(&inst, &mut st, Action::Inspect(1)),
            Err(StepError::Terminated)
        );
        assert_eq!(
            step(&inst, &mut st, Action::Commit(0)),
            Err(StepError::Terminated)
        );
        assert_eq!(valid_actions(&inst, &st), Vec::new());
    }

    #[test]
    fn reversible_retains_futures_after_wrong_commit() {
        let inst = small_instance(Variant::Reversible);
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Inspect(0)).unwrap();
        step(&inst, &mut st, Action::Commit(1)).unwrap(); // wrong provisional, truth = 0
        assert_eq!(st.status, Status::Running);
        assert_eq!(st.commitment, Some(1));
        // Inspection still reachable.
        step(&inst, &mut st, Action::Inspect(1)).unwrap();
        assert_eq!(st.status, Status::Running);
        // A different commit replaces the provisional selection without terminating.
        step(&inst, &mut st, Action::Commit(0)).unwrap();
        assert_eq!(st.status, Status::Running);
        assert_eq!(st.commitment, Some(0));
    }

    #[test]
    fn reversible_confirm_terminates_and_scores() {
        let inst = small_instance(Variant::Reversible);
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Commit(0)).unwrap(); // provisional, truth = 0
        assert_eq!(st.status, Status::Running);
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
    fn invalid_actions_rejected() {
        let inst = small_instance(Variant::Reversible);

        let mut st = reset(&inst);
        assert_eq!(
            step(&inst, &mut st, Action::Inspect(2)),
            Err(StepError::OutOfRange)
        );
        assert_eq!(
            step(&inst, &mut st, Action::Commit(2)),
            Err(StepError::OutOfRange)
        );

        step(&inst, &mut st, Action::Inspect(0)).unwrap();
        assert_eq!(
            step(&inst, &mut st, Action::Inspect(0)),
            Err(StepError::AlreadyProbed)
        );

        // Being priced out of every remaining probe does NOT end the episode
        // — Commit is always available, so Unaffordable is rejected while
        // the state stays Running and unmutated (DEFECT 1 fix).
        let tight = Instance {
            budget: 1,
            ..small_instance(Variant::Reversible)
        };
        let mut st3 = reset(&tight);
        step(&tight, &mut st3, Action::Inspect(1)).unwrap(); // spends the entire budget
        assert_eq!(st3.status, Status::Running);
        assert_eq!(
            step(&tight, &mut st3, Action::Inspect(0)),
            Err(StepError::Unaffordable)
        );
        assert_eq!(st3.status, Status::Running);
        assert_eq!(st3.spent, 1);
    }

    #[test]
    fn unaffordable_probe_rejected_while_running() {
        // Two probes, second one costs more than remaining budget after the
        // first, but a third cheap probe keeps the episode Running so we can
        // observe Unaffordable rather than any auto-termination.
        let inst = Instance {
            n_hyp: 2,
            n_probe: 3,
            evidence: vec![10, 11, 20, 21, 30, 31],
            probe_cost: vec![1, 10, 1],
            budget: 2,
            step_limit: 10,
            truth: 0,
            variant: Variant::Reversible,
            seed: 1,
            index: 0,
        };
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Inspect(0)).unwrap(); // spent 1, remaining 1
        assert_eq!(st.status, Status::Running); // probe 2 (cost 1) still affordable
        assert_eq!(
            step(&inst, &mut st, Action::Inspect(1)),
            Err(StepError::Unaffordable)
        );
        assert_eq!(st.status, Status::Running);
        assert_eq!(st.spent, 1);
    }

    /// DEFECT 1 regression: `budget: 2`, `probe_cost: [1, 10]`. After
    /// inspecting probe 0, probe 1 is priced out — but the episode must stay
    /// Running with `Commit` reachable, not auto-terminate before the
    /// learner can act on the evidence it already has.
    #[test]
    fn priced_out_of_probing_leaves_commit_reachable() {
        let inst = Instance {
            n_hyp: 2,
            n_probe: 2,
            evidence: vec![10, 11, 20, 21],
            probe_cost: vec![1, 10],
            budget: 2,
            step_limit: 100,
            truth: 0,
            variant: Variant::Irreversible,
            seed: 7,
            index: 0,
        };
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Inspect(0)).unwrap(); // spends 1, remaining 1 < probe 1's cost 10
        assert_eq!(st.status, Status::Running);
        assert!(valid_actions(&inst, &st).contains(&Action::Commit(0)));
        assert_eq!(
            step(&inst, &mut st, Action::Inspect(1)),
            Err(StepError::Unaffordable)
        );
        assert_eq!(st.status, Status::Running); // still not terminated
        step(&inst, &mut st, Action::Commit(0)).unwrap(); // truth = 0
        assert_eq!(
            st.status,
            Status::Terminated {
                correct: true,
                reason: EndReason::Committed
            }
        );
    }

    #[test]
    fn step_limit_scores_provisional_commitment() {
        let inst = Instance {
            step_limit: 1,
            ..small_instance(Variant::Reversible)
        };
        let mut st = reset(&inst);
        // Provisional commit only; step reaches step_limit before it could
        // ever be confirmed by a second identical Commit call.
        step(&inst, &mut st, Action::Commit(0)).unwrap(); // truth = 0
        assert_eq!(
            st.status,
            Status::Terminated {
                correct: true,
                reason: EndReason::StepLimit
            }
        );
    }

    #[test]
    fn step_limit_without_commitment_scores_incorrect() {
        let inst = Instance {
            step_limit: 1,
            ..small_instance(Variant::Reversible)
        };
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Inspect(0)).unwrap(); // no commitment ever recorded
        assert_eq!(
            st.status,
            Status::Terminated {
                correct: false,
                reason: EndReason::StepLimit
            }
        );
    }

    /// DEFECT 2 regression: a `Reversible` learner that never repeats a
    /// `Commit` would otherwise run forever (and overflow `State::step`).
    /// `step_limit` forces it to end.
    #[test]
    fn reversible_commit_thrash_terminates_at_step_limit() {
        let inst = Instance {
            step_limit: 3,
            ..small_instance(Variant::Reversible)
        };
        let mut st = reset(&inst);
        step(&inst, &mut st, Action::Commit(1)).unwrap(); // provisional 1, step 1
        assert_eq!(st.status, Status::Running);
        step(&inst, &mut st, Action::Commit(0)).unwrap(); // provisional 0, step 2
        assert_eq!(st.status, Status::Running);
        step(&inst, &mut st, Action::Commit(1)).unwrap(); // provisional 1 again (not a confirm), step 3 == step_limit
        assert_eq!(st.step, 3);
        assert_eq!(
            st.status,
            Status::Terminated {
                correct: false, // truth = 0, last provisional commitment was 1
                reason: EndReason::StepLimit
            }
        );
        // Confirms the cap actually stopped it: further action errors.
        assert_eq!(
            step(&inst, &mut st, Action::Commit(0)),
            Err(StepError::Terminated)
        );
    }

    #[test]
    fn valid_actions_never_reads_truth_and_matches_reachable_set() {
        let inst = small_instance(Variant::Reversible);
        let st = reset(&inst);
        let actions = valid_actions(&inst, &st);
        assert_eq!(actions.len(), 4); // Inspect(0), Inspect(1), Commit(0), Commit(1)
        assert!(actions.contains(&Action::Inspect(0)));
        assert!(actions.contains(&Action::Inspect(1)));
        assert!(actions.contains(&Action::Commit(0)));
        assert!(actions.contains(&Action::Commit(1)));
    }
}
