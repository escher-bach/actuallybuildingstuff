//! Batched Python boundary for the Step 1 world executor.
//!
//! `observations` and `valid_actions` are learner-visible.  Methods prefixed
//! `privileged_` intentionally cross the teacher/verifier boundary and must
//! never be used as learner inputs.

use pyo3::exceptions::{PyIndexError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PySet};
use std::path::Path;
use world::data::{generate_dataset_shard, write_dataset_shard, ShardSpec, TokenizerIdentity};
use world::generate::{sample, FamilyParams};
use world::render::{parse_action as parse_rendered_action, render_action as render_rendered_action, render_observation, Rendering};
use world::teacher::{consistent, outcome, teach};
use world::trajectory::{ByteTokenizer, TargetPolicy};
use world::{reset, step, valid_actions, Action, Instance, State, StepError, Variant};

const WORLD_FAMILY_VERSION: &str = "world-0.1.0";
const GENERATOR_VERSION: &str = "world-generate-0.1.0";
const TEACHER_POLICY_VERSION: &str = "world-teacher-0.1.0";
const MAX_REJECTION_ATTEMPTS: u64 = 1_000_000;

/// Python configuration mirroring `world::generate::FamilyParams`.
#[pyclass(name = "FamilyParams")]
#[derive(Clone)]
struct PyFamilyParams {
    inner: FamilyParams,
}

#[pymethods]
impl PyFamilyParams {
    #[new]
    #[pyo3(signature = (*, n_hyp, n_probe, n_evidence, cost_lo, cost_hi, budget_slack, min_depth, step_slack, variant))]
    fn new(
        n_hyp: u16,
        n_probe: u16,
        n_evidence: u16,
        cost_lo: i32,
        cost_hi: i32,
        budget_slack: i32,
        min_depth: u16,
        step_slack: u16,
        variant: &str,
    ) -> PyResult<Self> {
        Ok(Self {
            inner: FamilyParams {
                n_hyp,
                n_probe,
                n_evidence,
                cost_lo,
                cost_hi,
                budget_slack,
                min_depth,
                step_slack,
                variant: parse_variant(variant)?,
            },
        })
    }

    #[getter]
    fn n_hyp(&self) -> u16 {
        self.inner.n_hyp
    }
    #[getter]
    fn n_probe(&self) -> u16 {
        self.inner.n_probe
    }
    #[getter]
    fn n_evidence(&self) -> u16 {
        self.inner.n_evidence
    }
    #[getter]
    fn cost_lo(&self) -> i32 {
        self.inner.cost_lo
    }
    #[getter]
    fn cost_hi(&self) -> i32 {
        self.inner.cost_hi
    }
    #[getter]
    fn budget_slack(&self) -> i32 {
        self.inner.budget_slack
    }
    #[getter]
    fn min_depth(&self) -> u16 {
        self.inner.min_depth
    }
    #[getter]
    fn step_slack(&self) -> u16 {
        self.inner.step_slack
    }
    #[getter]
    fn variant(&self) -> &'static str {
        variant_name(self.inner.variant)
    }
}

/// A batch of independently sampled episodes. There is deliberately no
/// single-episode Python API: every observation, target, transition, and
/// outcome call handles the complete batch.
#[pyclass(name = "Batch")]
struct PyBatch {
    instances: Vec<Instance>,
    states: Vec<State>,
    seed: u64,
}

#[pymethods]
impl PyBatch {
    #[new]
    fn new(params: &PyFamilyParams, seed: u64, n_episodes: usize) -> PyResult<Self> {
        let mut instances = Vec::with_capacity(n_episodes);
        let mut index = 0u64;
        while instances.len() < n_episodes && index < MAX_REJECTION_ATTEMPTS {
            if let Ok(instance) = sample(&params.inner, seed, index) {
                instances.push(instance);
            }
            index += 1;
        }
        if instances.len() != n_episodes {
            return Err(PyValueError::new_err(format!(
                "sampled {} of {n_episodes} valid episodes after {MAX_REJECTION_ATTEMPTS} rejection attempts",
                instances.len()
            )));
        }
        let states = instances.iter().map(reset).collect();
        Ok(Self {
            instances,
            states,
            seed,
        })
    }

    /// Learner-visible observations for every episode. Never includes truth,
    /// teacher targets, or verifier data.
    fn observations(&self, rendering: &str) -> PyResult<Vec<String>> {
        let rendering = parse_rendering(rendering)?;
        Ok(self
            .instances
            .iter()
            .zip(&self.states)
            .map(|(instance, state)| render_observation(instance, state, rendering))
            .collect())
    }

    /// Learner-visible encoded actions for every episode. Never reads truth.
    fn valid_actions(&self) -> Vec<Vec<u32>> {
        self.instances
            .iter()
            .zip(&self.states)
            .map(|(instance, state)| {
                valid_actions(instance, state)
                    .into_iter()
                    .map(|action| encode_action(action, instance.n_probe))
                    .collect()
            })
            .collect()
    }

    /// Applies one encoded action for each live episode in one batch call.
    ///
    /// This is a *compact* action vector: `actions[j]` is applied to the
    /// `j`th entry of `live_episode_indices()`, not to original episode `j`.
    /// Live episode indices are always in ascending original batch order.
    /// `0..n_probe` encodes `Inspect(q)`; `n_probe + h` encodes `Commit(h)`.
    /// The supplied action list is checked completely before any state changes.
    fn step(&mut self, actions: Vec<i64>) -> PyResult<()> {
        let live: Vec<usize> = self
            .states
            .iter()
            .enumerate()
            .filter_map(|(i, state)| {
                (!matches!(state.status, world::Status::Terminated { .. })).then_some(i)
            })
            .collect();
        if actions.len() != live.len() {
            return Err(PyValueError::new_err(format!(
                "expected one action for each of {} live episodes, got {}",
                live.len(),
                actions.len()
            )));
        }

        let mut decoded = Vec::with_capacity(live.len());
        for (&episode, &encoded) in live.iter().zip(&actions) {
            let instance = &self.instances[episode];
            let action = decode_action(encoded, instance.n_probe, instance.n_hyp)
                .map_err(|reason| PyValueError::new_err(format!("episode {episode}: {reason}")))?;
            let mut checked = self.states[episode].clone();
            step(instance, &mut checked, action).map_err(|error| {
                PyValueError::new_err(format!("episode {episode}: {}", step_error_name(error)))
            })?;
            decoded.push(action);
        }
        for (&episode, action) in live.iter().zip(decoded) {
            step(&self.instances[episode], &mut self.states[episode], action)
                .expect("actions were validated against an unchanged state");
        }
        Ok(())
    }

    /// Apply rendered learner attempts independently and return the actual
    /// per-episode transition record needed by dense correction collection.
    /// A malformed or invalid attempt affects only its own episode; it leaves
    /// that state unchanged and receives targets from that unchanged state.
    fn step_attempts(&mut self, py: Python<'_>, attempts: Vec<String>, rendering: &str) -> PyResult<Vec<Py<PyAny>>> {
        let rendering = parse_rendering(rendering)?;
        let live = self.live_episode_indices();
        if attempts.len() != live.len() {
            return Err(PyValueError::new_err(format!(
                "expected one rendered attempt for each of {} live episodes, got {}",
                live.len(), attempts.len()
            )));
        }
        let mut records = Vec::with_capacity(live.len());
        for (episode, learner_text) in live.into_iter().zip(attempts) {
            let instance = &self.instances[episode];
            let observation_before = render_observation(instance, &self.states[episode], rendering);
            let parsed = parse_rendered_action(&learner_text, rendering).ok().and_then(|action| match action {
                Action::Inspect(q) if q < instance.n_probe => Some(action),
                Action::Commit(h) if h < instance.n_hyp => Some(action),
                _ => None,
            });
            let parsed_encoded = parsed.map(|action| encode_action(action, instance.n_probe));
            let accepted = if let Some(action) = parsed {
                let mut successor = self.states[episode].clone();
                if step(instance, &mut successor, action).is_ok() {
                    self.states[episode] = successor;
                    true
                } else {
                    false
                }
            } else {
                false
            };
            let observation_after = render_observation(instance, &self.states[episode], rendering);
            let targets = teach(instance, &self.states[episode]);
            let outcome = outcome(instance, &self.states[episode]);
            let item = PyDict::new(py);
            item.set_item("episode_index", episode)?;
            item.set_item("learner_text", learner_text)?;
            item.set_item("parsed_action", parsed_encoded)?;
            item.set_item("accepted", accepted)?;
            item.set_item("observation_before", observation_before)?;
            item.set_item("observation_after", observation_after)?;
            item.set_item("preferred_corrections", PySet::new(py, targets.preferred_actions.into_iter().map(|a| encode_action(a, instance.n_probe)))?)?;
            item.set_item("terminal_outcome", if outcome.terminated { Some((outcome.correct, outcome.spent, outcome.steps, outcome.budget_violation, outcome.unreachable)) } else { None })?;
            records.push(item.into_any().unbind());
        }
        Ok(records)
    }

    /// Original batch indices of currently live episodes, in the exact order
    /// accepted by `step`. All other batched return values stay indexed by the
    /// original batch position and include terminated episodes.
    fn live_episode_indices(&self) -> Vec<usize> {
        self.states
            .iter()
            .enumerate()
            .filter_map(|(i, state)| {
                (!matches!(state.status, world::Status::Terminated { .. })).then_some(i)
            })
            .collect()
    }

    /// PRIVILEGED teacher targets for every episode. `preferred_actions` is a
    /// Python `set`, preserving all tied optimal actions rather than choosing one.
    fn privileged_teacher_targets(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        let mut result = Vec::with_capacity(self.instances.len());
        for (instance, state) in self.instances.iter().zip(&self.states) {
            let targets = teach(instance, state);
            let item = PyDict::new(py);
            let preferred: Vec<u32> = targets
                .preferred_actions
                .into_iter()
                .map(|action| encode_action(action, instance.n_probe))
                .collect();
            let valid: Vec<u32> = targets
                .valid_actions
                .into_iter()
                .map(|action| encode_action(action, instance.n_probe))
                .collect();
            item.set_item("preferred_actions", PySet::new(py, preferred)?)?;
            item.set_item("valid_actions", valid)?;
            item.set_item("licenses_commitment", targets.licenses_commitment)?;
            result.push(item.into_any().unbind());
        }
        Ok(result)
    }

    /// PRIVILEGED count of hypotheses still consistent with each episode's
    /// history (STEP-1 3.6's "the hypotheses consistent with the complete
    /// history"). `licenses_commitment` is the same quantity thresholded at
    /// one; the count separates "committed with two live" from "committed
    /// with six live", which is the resolution the premature-commitment
    /// measurement needs. Evaluation and audit only: it must never be
    /// rendered, tokenized, or supervised.
    fn privileged_consistent_counts(&self) -> Vec<u32> {
        self.instances
            .iter()
            .zip(&self.states)
            .map(|(instance, state)| consistent(instance, state).count_ones())
            .collect()
    }

    /// PRIVILEGED verifier outcomes for every episode; this is the
    /// outcome-only RLVR signal, not a learner observation.
    fn privileged_outcomes(&self) -> Vec<(bool, bool, i32, u16, bool, bool)> {
        self.instances
            .iter()
            .zip(&self.states)
            .map(|(instance, state)| {
                let value = outcome(instance, state);
                (
                    value.terminated,
                    value.correct,
                    value.spent,
                    value.steps,
                    value.budget_violation,
                    value.unreachable,
                )
            })
            .collect()
    }

    /// Completion state for every episode.
    fn done(&self) -> Vec<bool> {
        self.states
            .iter()
            .map(|state| matches!(state.status, world::Status::Terminated { .. }))
            .collect()
    }

    /// Reproducibility identity for episode `i`. `rendering` defaults to A so
    /// `replay_key(i)` is complete, while callers using B can name it exactly.
    #[pyo3(signature = (i, rendering = "a"))]
    fn replay_key(&self, py: Python<'_>, i: usize, rendering: &str) -> PyResult<Py<PyAny>> {
        let instance = self
            .instances
            .get(i)
            .ok_or_else(|| PyIndexError::new_err(format!("episode index {i} is out of range")))?;
        let item = PyDict::new(py);
        item.set_item("world_family_version", WORLD_FAMILY_VERSION)?;
        item.set_item("generator_version", GENERATOR_VERSION)?;
        item.set_item("root_seed", self.seed)?;
        item.set_item("instance_index", instance.index)?;
        item.set_item("variant", variant_name(instance.variant))?;
        item.set_item("rendering", rendering_name(parse_rendering(rendering)?))?;
        item.set_item("teacher_policy_version", TEACHER_POLICY_VERSION)?;
        Ok(item.into_any().unbind())
    }

    #[getter]
    fn n_episodes(&self) -> usize {
        self.instances.len()
    }
}

fn parse_variant(value: &str) -> PyResult<Variant> {
    match value.to_ascii_lowercase().as_str() {
        "irreversible" => Ok(Variant::Irreversible),
        "reversible" => Ok(Variant::Reversible),
        _ => Err(PyValueError::new_err(
            "variant must be 'irreversible' or 'reversible'",
        )),
    }
}

fn variant_name(value: Variant) -> &'static str {
    match value {
        Variant::Irreversible => "irreversible",
        Variant::Reversible => "reversible",
    }
}

fn parse_rendering(value: &str) -> PyResult<Rendering> {
    match value.to_ascii_lowercase().as_str() {
        "a" => Ok(Rendering::A),
        "b" => Ok(Rendering::B),
        _ => Err(PyValueError::new_err("rendering must be 'a' or 'b'")),
    }
}

fn rendering_name(value: Rendering) -> &'static str {
    match value {
        Rendering::A => "a",
        Rendering::B => "b",
    }
}

fn encode_action(action: Action, n_probe: u16) -> u32 {
    match action {
        Action::Inspect(q) => q as u32,
        Action::Commit(h) => n_probe as u32 + h as u32,
    }
}

fn decode_action(encoded: i64, n_probe: u16, n_hyp: u16) -> Result<Action, &'static str> {
    if encoded < 0 {
        return Err("action encoding must be non-negative");
    }
    let encoded = encoded as u64;
    if encoded < n_probe as u64 {
        return Ok(Action::Inspect(encoded as u16));
    }
    let h = encoded - n_probe as u64;
    if h < n_hyp as u64 {
        Ok(Action::Commit(h as u16))
    } else {
        Err("action encoding is out of range")
    }
}

fn step_error_name(error: StepError) -> &'static str {
    match error {
        StepError::OutOfRange => "action is out of range",
        StepError::AlreadyProbed => "probe was already inspected",
        StepError::Unaffordable => "probe is unaffordable",
        StepError::Terminated => "episode is already terminated",
    }
}

/// Canonical public action rendering. The model is trained on this text, not
/// the private integer action encoding used by the batched executor.
#[pyfunction]
fn render_action(encoded: i64, n_probe: u16, n_hyp: u16, rendering: &str) -> PyResult<String> {
    let action = decode_action(encoded, n_probe, n_hyp).map_err(PyValueError::new_err)?;
    Ok(render_rendered_action(action, parse_rendering(rendering)?))
}

/// Parser boundary used by Python evaluation. It returns the executor's
/// compact encoding only after the Rust canonical parser has accepted it.
#[pyfunction]
fn parse_action(text: &str, n_probe: u16, n_hyp: u16, rendering: &str) -> PyResult<i64> {
    let action = parse_rendered_action(text, parse_rendering(rendering)?)
        .map_err(|error| PyValueError::new_err(format!("malformed action: {error:?}")))?;
    match action {
        Action::Inspect(q) if q < n_probe => Ok(q as i64),
        Action::Commit(h) if h < n_hyp => Ok(n_probe as i64 + h as i64),
        _ => Err(PyValueError::new_err("action is outside this instance's declared vocabulary")),
    }
}

/// Produces the existing Rust binary shard format so the measured DataLoader
/// and the training loop consume identical mmap-able payloads.
#[pyfunction]
#[pyo3(signature = (params, seed, episode_count, rendering, max_sequence_tokens, directory, stem, target_policy = "teacher_preferred"))]
fn generate_teacher_shard(
    params: &PyFamilyParams,
    seed: u64,
    episode_count: usize,
    rendering: &str,
    max_sequence_tokens: usize,
    directory: &str,
    stem: &str,
    target_policy: &str,
) -> PyResult<(String, String, String)> {
    let rendering = parse_rendering(rendering)?;
    // The control is opt-in and named; a shard can never become a control by
    // accident, and its manifest and replay records record which it is.
    let target_policy = TargetPolicy::parse(target_policy).ok_or_else(|| {
        PyValueError::new_err("target_policy must be 'teacher_preferred' or 'random_valid_target_shuffled'")
    })?;
    let spec = ShardSpec {
        params: params.inner,
        root_seed: seed,
        first_instance_index: 0,
        episode_count,
        rendering,
        max_sequence_tokens,
        tokenizer: TokenizerIdentity::byte_utf8(),
        target_policy,
    };
    let shard = generate_dataset_shard(&spec, &ByteTokenizer)
        .map_err(|error| PyValueError::new_err(format!("shard generation failed: {error:?}")))?;
    let paths = write_dataset_shard(Path::new(directory), stem, &shard)
        .map_err(|error| PyValueError::new_err(format!("shard write failed: {error:?}")))?;
    Ok((
        paths.0.display().to_string(),
        paths.1.display().to_string(),
        paths.2.display().to_string(),
    ))
}

#[pyfunction]
fn tokenizer_identity() -> (String, String, String) {
    let identity = TokenizerIdentity::byte_utf8();
    (identity.name, identity.revision, identity.vocabulary_hash)
}

#[pymodule]
fn world_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFamilyParams>()?;
    m.add_class::<PyBatch>()?;
    m.add_function(wrap_pyfunction!(render_action, m)?)?;
    m.add_function(wrap_pyfunction!(parse_action, m)?)?;
    m.add_function(wrap_pyfunction!(generate_teacher_shard, m)?)?;
    m.add_function(wrap_pyfunction!(tokenizer_identity, m)?)?;
    Ok(())
}
