//! Teacher-conditioned trajectories and packed token training examples.
//!
//! The module is downstream of the world executor, renderer, and privileged
//! teacher. It never derives a target from hidden state itself.

use crate::render::{render_action, render_observation, Rendering};
use crate::teacher::{consistent, teach, truth_blind_optimal_action};
use crate::{reset, step, valid_actions, Action, Instance, State, Status};

/// Fragment-level encoder used by Rust packing. Production tokenizers should
/// implement this trait; the reference implementation below is lossless.
pub trait FragmentTokenizer {
    fn encode_fragment(&self, text: &str) -> Vec<u32>;
}

/// Frozen transport vocabulary shared with `step1_experiments.data`.
pub const PAD_ID: u32 = 256;
pub const BOS_ID: u32 = 257;
pub const EOS_ID: u32 = 258;
pub const OBS_ID: u32 = 259;
pub const ACTION_ID: u32 = 260;
pub const END_TURN_ID: u32 = 261;

/// Dependency-free, lossless reference tokenizer: one UTF-8 byte per token.
/// It is for tests and debug fixtures, not a final model vocabulary.
#[derive(Debug, Clone, Copy, Default)]
pub struct ByteTokenizer;

impl FragmentTokenizer for ByteTokenizer {
    fn encode_fragment(&self, text: &str) -> Vec<u32> {
        text.as_bytes().iter().map(|&byte| byte as u32).collect()
    }
}

impl ByteTokenizer {
    pub fn decode(ids: &[u32]) -> Option<String> {
        let bytes: Option<Vec<u8>> = ids.iter().map(|&id| u8::try_from(id).ok()).collect();
        String::from_utf8(bytes?).ok()
    }
}

/// Meaning of a directly supervised token. Zero is intentionally no-loss.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetChannel {
    None = 0,
    TeacherPreferredAction = 1,
    TeacherCorrectionAction = 2,
}

/// An actual learner attempt. Parser-level failures retain their original
/// surface text and deliberately never become semantic world actions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LearnerAttempt {
    Action(Action),
    Malformed(String),
}

/// One learner action together with the teacher target available after that
/// action was accepted or rejected. This is the selected online regime: a
/// caller executes its own decoded actions, while Rust derives local recovery
/// labels from the resulting state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CorrectionStep {
    pub observation_before: String,
    pub learner_attempt: LearnerAttempt,
    /// `false` means the executor rejected `learner_action` and state is
    /// unchanged. The error itself is intentionally not rendered as a hidden
    /// teacher fact; the action spelling and public state are enough context.
    pub accepted: bool,
    pub observation_after: String,
    /// Empty after a terminal action: no false correction target is invented.
    pub preferred_corrections: Vec<Action>,
    pub selected_correction: Option<Action>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct LearnerConditionedTrajectory {
    pub rendering: Rendering,
    pub steps: Vec<CorrectionStep>,
    pub final_state: State,
}

impl TargetChannel {
    pub fn id(self) -> u8 {
        self as u8
    }
}

/// One state-action item. Observation is learner context; action fields are
/// teacher labels. The complete tie set is retained for set-valued losses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrajectoryStep {
    pub observation: String,
    pub preferred_actions: Vec<Action>,
    /// The action the world executes, always the teacher's.
    pub selected_action: Action,
    /// The action that receives the loss. Equal to `selected_action` under the
    /// teacher policy; under `TargetPolicy::RandomValid` it is a different
    /// legal action, which is what makes the target-shuffled control a control.
    pub supervised_action: Action,
    pub licenses_commitment: bool,
}

/// Which action the supervised span teaches. The executed action is the
/// teacher's in both cases, so the state distribution, observation sequence,
/// trajectory length, and supervision density are held constant and only the
/// state-to-action correspondence changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TargetPolicy {
    TeacherPreferred,
    /// Uniform over the actions legal in the state, which is exactly the
    /// unguided policy whose closed-loop success is measured independently.
    RandomValid,
    /// The Bayes-optimal expert that reads the evidence table but not `truth`.
    /// Unlike the two above it drives the *rollout* as well as the label, so
    /// the demonstrated trajectory is one a learner could actually follow:
    /// the dense teacher's ~2 probes are only sufficient with the answer in
    /// hand, and imitating that cost profile caps the learner.
    TruthBlindOptimal,
}

impl TargetPolicy {
    pub fn name(self) -> &'static str {
        match self {
            TargetPolicy::TeacherPreferred => "teacher_preferred",
            TargetPolicy::RandomValid => "random_valid_target_shuffled",
            TargetPolicy::TruthBlindOptimal => "truth_blind_optimal",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "teacher_preferred" => Some(TargetPolicy::TeacherPreferred),
            "random_valid_target_shuffled" => Some(TargetPolicy::RandomValid),
            "truth_blind_optimal" => Some(TargetPolicy::TruthBlindOptimal),
            _ => None,
        }
    }
}

/// A complete offline rollout produced without consulting learner weights.
#[derive(Debug, Clone, PartialEq)]
pub struct TeacherTrajectory {
    pub rendering: Rendering,
    pub steps: Vec<TrajectoryStep>,
    pub final_state: State,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TrajectoryError {
    EmptyTeacherTarget { step: u16 },
    TeacherActionRejected { step: u16 },
    DidNotTerminate { max_steps: u16 },
}

/// Build the primary teacher-conditioned data regime. Ties are selected using
/// a deterministic instance-keyed rotation while the full acceptable set is
/// retained in the trajectory metadata.
pub fn generate_teacher_trajectory(
    inst: &Instance,
    rendering: Rendering,
) -> Result<TeacherTrajectory, TrajectoryError> {
    generate_trajectory_with_targets(inst, rendering, TargetPolicy::TeacherPreferred)
}

/// The teacher rollout, with the supervised label chosen by `policy`.
pub fn generate_trajectory_with_targets(
    inst: &Instance,
    rendering: Rendering,
    policy: TargetPolicy,
) -> Result<TeacherTrajectory, TrajectoryError> {
    let mut state = reset(inst);
    let mut steps = Vec::new();
    let max_steps = inst.step_limit;

    while matches!(state.status, Status::Running) {
        if state.step >= max_steps {
            return Err(TrajectoryError::DidNotTerminate { max_steps });
        }
        // The truth-blind expert never consults `teach`, so nothing
        // truth-derived reaches the record it produces.
        let item = if matches!(policy, TargetPolicy::TruthBlindOptimal) {
            let action = truth_blind_optimal_action(inst, &state);
            TrajectoryStep {
                observation: render_observation(inst, &state, rendering),
                preferred_actions: vec![action],
                selected_action: action,
                supervised_action: action,
                licenses_commitment: consistent(inst, &state).count_ones() == 1,
            }
        } else {
            let targets = teach(inst, &state);
            if targets.preferred_actions.is_empty() {
                return Err(TrajectoryError::EmptyTeacherTarget { step: state.step });
            }
            let selected_index =
                representative_index(inst, state.step, targets.preferred_actions.len());
            let selected_action = targets.preferred_actions[selected_index];
            let supervised_action = match policy {
                TargetPolicy::TeacherPreferred => selected_action,
                TargetPolicy::RandomValid => {
                    let legal = valid_actions(inst, &state);
                    debug_assert!(!legal.is_empty());
                    legal[shuffled_index(inst, state.step, legal.len())]
                }
                TargetPolicy::TruthBlindOptimal => unreachable!(),
            };
            TrajectoryStep {
                observation: render_observation(inst, &state, rendering),
                preferred_actions: targets.preferred_actions,
                selected_action,
                supervised_action,
                licenses_commitment: targets.licenses_commitment,
            }
        };
        let selected_action = item.selected_action;
        step(inst, &mut state, selected_action)
            .map_err(|_| TrajectoryError::TeacherActionRejected { step: state.step })?;
        steps.push(item);
    }

    Ok(TeacherTrajectory {
        rendering,
        steps,
        final_state: state,
    })
}

/// Deterministic, replayable, and salted apart from tie-breaking so the
/// corrupted label cannot correlate with the teacher's own selection rule.
fn shuffled_index(inst: &Instance, step_index: u16, len: usize) -> usize {
    debug_assert!(len > 0);
    let mut value = inst.seed.rotate_left(23)
        ^ inst.index.rotate_left(5)
        ^ (step_index as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ 0x5851_f42d_4c95_7f2d;
    value ^= value >> 33;
    value = value.wrapping_mul(0xff51_afd7_ed55_8ccd);
    value ^= value >> 33;
    value = value.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    value ^= value >> 33;
    (value as usize) % len
}

fn representative_index(inst: &Instance, step_index: u16, len: usize) -> usize {
    debug_assert!(len > 0);
    let mut value = inst.seed ^ inst.index.rotate_left(17) ^ (step_index as u64).rotate_left(41);
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^= value >> 31;
    (value as usize) % len
}

/// Builds correction/recovery examples for a supplied learner action prefix.
/// The learner actions, rather than teacher actions, drive state changes. An
/// invalid typed action produces a correction from the unchanged state;
/// strategically poor but valid actions produce a correction from the real
/// successor state. No label is emitted after a terminal action.
pub fn generate_learner_conditioned_trajectory(
    inst: &Instance,
    rendering: Rendering,
    learner_actions: &[Action],
) -> LearnerConditionedTrajectory {
    let attempts: Vec<LearnerAttempt> = learner_actions
        .iter()
        .copied()
        .map(LearnerAttempt::Action)
        .collect();
    generate_learner_conditioned_attempts(inst, rendering, &attempts)
}

/// As [`generate_learner_conditioned_trajectory`], but also accepts raw
/// parser failures. A malformed surface leaves state unchanged and gets a
/// correction target from precisely that unchanged state.
pub fn generate_learner_conditioned_attempts(
    inst: &Instance,
    rendering: Rendering,
    learner_attempts: &[LearnerAttempt],
) -> LearnerConditionedTrajectory {
    let mut state = reset(inst);
    let mut steps = Vec::with_capacity(learner_attempts.len());

    for learner_attempt in learner_attempts {
        if !matches!(state.status, Status::Running) {
            break;
        }
        let observation_before = render_observation(inst, &state, rendering);
        let accepted = match learner_attempt {
            LearnerAttempt::Action(action) => step(inst, &mut state, *action).is_ok(),
            LearnerAttempt::Malformed(_) => false,
        };
        let observation_after = render_observation(inst, &state, rendering);
        let targets = if matches!(state.status, Status::Running) {
            teach(inst, &state)
        } else {
            crate::teacher::TeacherTargets {
                valid_actions: Vec::new(),
                preferred_actions: Vec::new(),
                licenses_commitment: false,
            }
        };
        let selected_correction = (!targets.preferred_actions.is_empty()).then(|| {
            let index = representative_index(inst, state.step, targets.preferred_actions.len());
            targets.preferred_actions[index]
        });
        steps.push(CorrectionStep {
            observation_before,
            learner_attempt: learner_attempt.clone(),
            accepted,
            observation_after,
            preferred_corrections: targets.preferred_actions,
            selected_correction,
        });
    }

    LearnerConditionedTrajectory {
        rendering,
        steps,
        final_state: state,
    }
}

/// Parallel buffers supplied directly to training. Only action-spelling
/// tokens receive a nonzero mask/channel; observations remain context.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedTrajectory {
    pub token_ids: Vec<u32>,
    pub loss_mask: Vec<u8>,
    pub target_channel_ids: Vec<u8>,
    /// Exact fragments passed to the tokenizer, retained for audit tests.
    pub debug_transcript: String,
}

/// Packs the exact model protocol used by Python evaluation:
/// `BOS (OBS observation ACTION rendered-action END_TURN)* EOS`.
/// Action bytes and the end-of-turn delimiter are directly supervised.
pub fn pack_teacher_trajectory<T: FragmentTokenizer>(
    trajectory: &TeacherTrajectory,
    tokenizer: &T,
) -> PackedTrajectory {
    let mut packed = PackedTrajectory {
        token_ids: Vec::new(),
        loss_mask: Vec::new(),
        target_channel_ids: Vec::new(),
        debug_transcript: String::new(),
    };
    append_special(&mut packed, BOS_ID, TargetChannel::None);
    for item in &trajectory.steps {
        append_special(&mut packed, OBS_ID, TargetChannel::None);
        append_fragment(&mut packed, tokenizer, &item.observation, TargetChannel::None);
        append_special(&mut packed, ACTION_ID, TargetChannel::None);
        let action = render_action(item.supervised_action, trajectory.rendering);
        append_fragment(
            &mut packed,
            tokenizer,
            &action,
            TargetChannel::TeacherPreferredAction,
        );
        append_special(
            &mut packed,
            END_TURN_ID,
            TargetChannel::TeacherPreferredAction,
        );
    }
    append_special(&mut packed, EOS_ID, TargetChannel::None);
    packed
}

/// Packs learner-conditioned correction traces. The learner action and both
/// observations are context. Only a nonterminal teacher correction gets a
/// direct target span, so an irreversible terminal mistake cannot be made to
/// look recoverable in the data.
pub fn pack_learner_conditioned_trajectory<T: FragmentTokenizer>(
    trajectory: &LearnerConditionedTrajectory,
    tokenizer: &T,
) -> PackedTrajectory {
    let mut packed = PackedTrajectory {
        token_ids: Vec::new(),
        loss_mask: Vec::new(),
        target_channel_ids: Vec::new(),
        debug_transcript: String::new(),
    };
    for item in &trajectory.steps {
        append_fragment(
            &mut packed,
            tokenizer,
            &item.observation_before,
            TargetChannel::None,
        );
        append_fragment(
            &mut packed,
            tokenizer,
            "\nLEARNER_ACTION ",
            TargetChannel::None,
        );
        append_fragment(
            &mut packed,
            tokenizer,
            &learner_attempt_text(&item.learner_attempt, trajectory.rendering),
            TargetChannel::None,
        );
        append_fragment(&mut packed, tokenizer, "\nRESULT ", TargetChannel::None);
        append_fragment(
            &mut packed,
            tokenizer,
            if item.accepted {
                "accepted\n"
            } else {
                "rejected\n"
            },
            TargetChannel::None,
        );
        append_fragment(
            &mut packed,
            tokenizer,
            &item.observation_after,
            TargetChannel::None,
        );
        if let Some(correction) = item.selected_correction {
            append_fragment(&mut packed, tokenizer, "\nCORRECTION ", TargetChannel::None);
            append_fragment(
                &mut packed,
                tokenizer,
                &render_action(correction, trajectory.rendering),
                TargetChannel::TeacherCorrectionAction,
            );
        }
        append_fragment(&mut packed, tokenizer, "\n", TargetChannel::None);
    }
    packed
}

fn learner_attempt_text(attempt: &LearnerAttempt, rendering: Rendering) -> String {
    match attempt {
        LearnerAttempt::Action(action) => render_action(*action, rendering),
        LearnerAttempt::Malformed(text) => text.clone(),
    }
}

fn append_fragment<T: FragmentTokenizer>(
    packed: &mut PackedTrajectory,
    tokenizer: &T,
    fragment: &str,
    channel: TargetChannel,
) {
    let ids = tokenizer.encode_fragment(fragment);
    let count = ids.len();
    packed.token_ids.extend(ids);
    packed
        .loss_mask
        .extend(std::iter::repeat(u8::from(channel != TargetChannel::None)).take(count));
    packed
        .target_channel_ids
        .extend(std::iter::repeat(channel.id()).take(count));
    packed.debug_transcript.push_str(fragment);
}

fn append_special(packed: &mut PackedTrajectory, id: u32, channel: TargetChannel) {
    packed.token_ids.push(id);
    packed.loss_mask.push(u8::from(channel != TargetChannel::None));
    packed.target_channel_ids.push(channel.id());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generate::{sample, FamilyParams};
    use crate::render::parse_action;
    use crate::replay;
    use crate::teacher::outcome;
    use crate::Variant;

    fn params(variant: Variant) -> FamilyParams {
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

    fn instance(variant: Variant) -> Instance {
        (0..10_000)
            .find_map(|index| sample(&params(variant), 20260811, index).ok())
            .expect("test parameters must produce an accepted instance")
    }

    /// The truth-blind expert is the one a learner could actually follow, so
    /// what matters is that its own play wins nearly always -- imitating a
    /// demonstrator that loses is pointless. It is deliberately NOT 100%: it
    /// cannot read the answer, so a residual guess remains.
    #[test]
    fn truth_blind_expert_wins_nearly_always_and_probes_more_than_the_teacher() {
        let params = FamilyParams {
            n_hyp: 6, n_probe: 5, n_evidence: 2, cost_lo: 1, cost_hi: 3,
            budget_slack: 1, min_depth: 2, step_slack: 2, variant: Variant::Irreversible,
        };
        let mut wins = 0usize;
        let mut total = 0usize;
        let mut blind_probes = 0usize;
        let mut teacher_probes = 0usize;
        let mut index = 0u64;
        while total < 200 && index < 20_000 {
            if let Ok(inst) = sample(&params, 4242, index) {
                let blind = generate_trajectory_with_targets(
                    &inst, Rendering::A, TargetPolicy::TruthBlindOptimal,
                )
                .expect("the truth-blind expert must terminate inside the step limit");
                let teacher = generate_teacher_trajectory(&inst, Rendering::A).unwrap();
                blind_probes += blind
                    .steps
                    .iter()
                    .filter(|s| matches!(s.selected_action, Action::Inspect(_)))
                    .count();
                teacher_probes += teacher
                    .steps
                    .iter()
                    .filter(|s| matches!(s.selected_action, Action::Inspect(_)))
                    .count();
                if matches!(
                    blind.final_state.status,
                    Status::Terminated { correct: true, .. }
                ) {
                    wins += 1;
                }
                total += 1;
            }
            index += 1;
        }
        assert_eq!(total, 200);
        let rate = wins as f64 / total as f64;
        assert!(rate > 0.90, "truth-blind expert won only {rate:.3} of episodes");
        // It must pay for not knowing the answer, or it is secretly reading it.
        assert!(
            blind_probes > teacher_probes,
            "truth-blind expert used {blind_probes} probes against the teacher's {teacher_probes}"
        );
    }

    #[test]
    fn teacher_trajectory_is_deterministic_and_wins() {
        for variant in [Variant::Irreversible, Variant::Reversible] {
            let inst = instance(variant);
            let first = generate_teacher_trajectory(&inst, Rendering::A).unwrap();
            let second = generate_teacher_trajectory(&inst, Rendering::A).unwrap();
            assert_eq!(first, second);
            assert!(outcome(&inst, &first.final_state).correct);
            assert!(matches!(
                first.final_state.status,
                Status::Terminated { .. }
            ));
            for item in &first.steps {
                assert!(item.preferred_actions.contains(&item.selected_action));
            }
        }
    }

    #[test]
    fn byte_packing_uses_shared_transport_protocol_and_masks_action_spans() {
        let trajectory =
            generate_teacher_trajectory(&instance(Variant::Irreversible), Rendering::B).unwrap();
        let packed = pack_teacher_trajectory(&trajectory, &ByteTokenizer);
        assert_eq!(packed.token_ids.first(), Some(&BOS_ID));
        assert_eq!(packed.token_ids.last(), Some(&EOS_ID));
        assert_eq!(packed.token_ids.iter().filter(|&&id| id == OBS_ID).count(), trajectory.steps.len());
        assert_eq!(packed.token_ids.iter().filter(|&&id| id == ACTION_ID).count(), trajectory.steps.len());
        assert_eq!(packed.token_ids.iter().filter(|&&id| id == END_TURN_ID).count(), trajectory.steps.len());
        assert_eq!(packed.token_ids.len(), packed.loss_mask.len());
        assert_eq!(packed.token_ids.len(), packed.target_channel_ids.len());
        assert!(packed.loss_mask.iter().any(|&mask| mask == 1));

        let expected_actions: String = trajectory
            .steps
            .iter()
            .map(|item| render_action(item.selected_action, Rendering::B))
            .collect();
        let action_ids: Vec<u32> = packed
            .token_ids
            .iter()
            .zip(&packed.loss_mask)
            .filter_map(|(&id, &mask)| (mask == 1 && id < 256).then_some(id))
            .collect();
        assert_eq!(
            ByteTokenizer::decode(&action_ids).as_deref(),
            Some(expected_actions.as_str())
        );
        assert!(packed
            .token_ids
            .iter()
            .zip(&packed.loss_mask)
            .filter(|(id, _)| **id == END_TURN_ID)
            .all(|(_, &mask)| mask == 1));
        for (&mask, &channel) in packed.loss_mask.iter().zip(&packed.target_channel_ids) {
            assert_eq!(
                mask == 1,
                channel == TargetChannel::TeacherPreferredAction.id()
            );
        }
    }

    #[test]
    fn transcript_contains_no_teacher_metadata_or_truth_label() {
        let inst = instance(Variant::Irreversible);
        let packed = pack_teacher_trajectory(
            &generate_teacher_trajectory(&inst, Rendering::A).unwrap(),
            &ByteTokenizer,
        );
        assert!(!packed.debug_transcript.contains("licenses_commitment"));
        assert!(!packed.debug_transcript.contains("preferred_actions"));
        assert!(!packed.debug_transcript.contains("truth"));
        assert!(packed.target_channel_ids.iter().all(|&id| {
            id == TargetChannel::None.id() || id == TargetChannel::TeacherPreferredAction.id()
        }));
    }

    #[test]
    fn learner_conditioned_invalid_action_gets_correction_without_state_change() {
        let inst = instance(Variant::Irreversible);
        let trace = generate_learner_conditioned_trajectory(
            &inst,
            Rendering::A,
            &[Action::Inspect(inst.n_probe)],
        );
        assert_eq!(trace.steps.len(), 1);
        let item = &trace.steps[0];
        assert!(!item.accepted);
        assert_eq!(item.observation_before, item.observation_after);
        assert!(item.selected_correction.is_some());
        assert!(item
            .preferred_corrections
            .contains(&item.selected_correction.unwrap()));
    }

    #[test]
    fn parser_level_malformed_attempt_is_preserved_and_corrected() {
        let inst = instance(Variant::Irreversible);
        let trace = generate_learner_conditioned_attempts(
            &inst,
            Rendering::B,
            &[LearnerAttempt::Malformed("settle maybe?".to_string())],
        );
        let item = &trace.steps[0];
        assert!(!item.accepted);
        assert_eq!(item.observation_before, item.observation_after);
        assert!(item.selected_correction.is_some());
        let packed = pack_learner_conditioned_trajectory(&trace, &ByteTokenizer);
        assert!(packed.debug_transcript.contains("settle maybe?"));
        assert!(packed
            .target_channel_ids
            .iter()
            .any(|&id| id == TargetChannel::TeacherCorrectionAction.id()));
    }

    #[test]
    fn wrong_commit_has_no_false_recovery_label_when_irreversible() {
        let inst = instance(Variant::Irreversible);
        let wrong = Action::Commit((inst.truth + 1) % inst.n_hyp);
        let trace = generate_learner_conditioned_trajectory(&inst, Rendering::A, &[wrong]);
        assert!(trace.steps[0].accepted);
        assert!(trace.steps[0].selected_correction.is_none());
        assert!(matches!(
            trace.final_state.status,
            Status::Terminated { correct: false, .. }
        ));
        let packed = pack_learner_conditioned_trajectory(&trace, &ByteTokenizer);
        assert!(!packed
            .target_channel_ids
            .iter()
            .any(|&id| id == TargetChannel::TeacherCorrectionAction.id()));
    }

    #[test]
    fn wrong_commit_has_recovery_label_when_reversible() {
        let inst = instance(Variant::Reversible);
        let wrong = Action::Commit((inst.truth + 1) % inst.n_hyp);
        let trace = generate_learner_conditioned_trajectory(&inst, Rendering::B, &[wrong]);
        assert!(trace.steps[0].accepted);
        assert!(matches!(trace.final_state.status, Status::Running));
        assert!(trace.steps[0].selected_correction.is_some());
        let packed = pack_learner_conditioned_trajectory(&trace, &ByteTokenizer);
        assert!(packed
            .target_channel_ids
            .iter()
            .any(|&id| id == TargetChannel::TeacherCorrectionAction.id()));
    }

    #[test]
    fn cross_rendering_keeps_teacher_actions_and_transitions_aligned() {
        let inst = instance(Variant::Reversible);
        let a = generate_teacher_trajectory(&inst, Rendering::A).unwrap();
        let b = generate_teacher_trajectory(&inst, Rendering::B).unwrap();
        assert_eq!(
            a.steps
                .iter()
                .map(|s| s.selected_action)
                .collect::<Vec<_>>(),
            b.steps
                .iter()
                .map(|s| s.selected_action)
                .collect::<Vec<_>>()
        );
        assert_eq!(a.final_state, b.final_state);
        let actions: Vec<Action> = a.steps.iter().map(|s| s.selected_action).collect();
        assert_eq!(replay(&inst, &actions).unwrap(), a.final_state);
        for item in &a.steps {
            let text = render_action(item.selected_action, Rendering::A);
            assert_eq!(parse_action(&text, Rendering::A), Ok(item.selected_action));
        }
        for item in &b.steps {
            let text = render_action(item.selected_action, Rendering::B);
            assert_eq!(parse_action(&text, Rendering::B), Ok(item.selected_action));
        }
    }

    #[test]
    fn randomized_action_histories_preserve_executor_and_mask_invariants() {
        let inst = instance(Variant::Reversible);
        let mut seed = 17u64;
        for _case in 0..64 {
            let mut actions = Vec::new();
            for _ in 0..12 {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                let raw = (seed >> 32) as u16;
                actions.push(if raw & 1 == 0 {
                    Action::Inspect(raw % (inst.n_probe + 2))
                } else {
                    Action::Commit(raw % (inst.n_hyp + 2))
                });
            }
            let trace = generate_learner_conditioned_trajectory(&inst, Rendering::A, &actions);
            let packed = pack_learner_conditioned_trajectory(&trace, &ByteTokenizer);
            assert_eq!(packed.token_ids.len(), packed.loss_mask.len());
            assert_eq!(packed.token_ids.len(), packed.target_channel_ids.len());
            assert!(trace.final_state.spent <= inst.budget);
            assert!(trace.final_state.history.len() <= inst.n_probe as usize);
            for item in &trace.steps {
                if let Some(action) = item.selected_correction {
                    assert!(item.preferred_corrections.contains(&action));
                }
            }
        }
    }
}
