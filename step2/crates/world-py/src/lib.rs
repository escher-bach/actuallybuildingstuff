//! Thin batched Python boundary for the STEP 2 Rust world.
//!
//! Learner tensors contain only public event fields. Methods prefixed
//! `privileged_` are validator/evaluator surfaces and must never be passed to
//! the model as inputs.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};
use step2_world::{
    generate_trajectory, FamilyConfig, LearningToken, Role, RolloutEpisode, ACTION_HORIZON,
    ORACLE_VERSION, PAYLOAD_DIM, TOKEN_ABI_VERSION, WORLD_VERSION,
};

fn py_config(
    d_min: usize,
    d_max: usize,
    gain_min: f32,
    gain_max: f32,
    action_limit: f32,
    calibration_pulse: f32,
    max_control_steps: usize,
) -> PyResult<FamilyConfig> {
    let cfg = FamilyConfig {
        d_min,
        d_max,
        gain_min,
        gain_max,
        action_limit,
        calibration_pulse,
        max_control_steps,
        ..FamilyConfig::default()
    };
    cfg.validate().map_err(PyValueError::new_err)?;
    Ok(cfg)
}

#[derive(Default)]
struct PaddedBatch {
    role_ids: Vec<Vec<u8>>,
    key_ids: Vec<Vec<i64>>,
    position_ids: Vec<Vec<i64>>,
    payloads: Vec<Vec<Vec<f32>>>,
    attention_mask: Vec<Vec<i64>>,
    action_targets: Vec<Vec<Vec<f32>>>,
    action_target_mask: Vec<Vec<Vec<f32>>>,
    outcome_targets: Vec<Vec<f32>>,
    outcome_target_mask: Vec<Vec<f32>>,
    lengths: Vec<usize>,
}

fn pad_records(records: &[&[LearningToken]], max_tokens: usize) -> Result<PaddedBatch, String> {
    let mut batch = PaddedBatch::default();
    for record in records {
        if record.len() > max_tokens {
            return Err(format!(
                "trajectory length {} exceeds max_tokens {}; truncation is forbidden",
                record.len(),
                max_tokens
            ));
        }
        let mut roles = vec![Role::Pad as u8; max_tokens];
        let mut keys = vec![0i64; max_tokens];
        let mut positions = vec![0i64; max_tokens];
        let mut payloads = vec![vec![0.0f32; PAYLOAD_DIM]; max_tokens];
        let mut attention = vec![0i64; max_tokens];
        let mut action_targets = vec![vec![0.0f32; ACTION_HORIZON]; max_tokens];
        let mut action_mask = vec![vec![0.0f32; ACTION_HORIZON]; max_tokens];
        let mut outcome_targets = vec![0.0f32; max_tokens];
        let mut outcome_mask = vec![0.0f32; max_tokens];
        for (position, token) in record.iter().enumerate() {
            roles[position] = token.public.role as u8;
            keys[position] = token.public.key as i64;
            positions[position] = token.public.event as i64;
            payloads[position].copy_from_slice(&token.public.payload);
            attention[position] = 1;
            action_targets[position].copy_from_slice(&token.supervision.action_target);
            for h in 0..ACTION_HORIZON {
                action_mask[position][h] = if token.supervision.action_mask[h] {
                    1.0
                } else {
                    0.0
                };
            }
            outcome_targets[position] = token.supervision.outcome_target;
            outcome_mask[position] = if token.supervision.outcome_mask {
                1.0
            } else {
                0.0
            };
        }
        batch.role_ids.push(roles);
        batch.key_ids.push(keys);
        batch.position_ids.push(positions);
        batch.payloads.push(payloads);
        batch.attention_mask.push(attention);
        batch.action_targets.push(action_targets);
        batch.action_target_mask.push(action_mask);
        batch.outcome_targets.push(outcome_targets);
        batch.outcome_target_mask.push(outcome_mask);
        batch.lengths.push(record.len());
    }
    Ok(batch)
}

fn padded_to_dict<'py>(py: Python<'py>, batch: PaddedBatch) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    output.set_item("role_ids", batch.role_ids)?;
    output.set_item("key_ids", batch.key_ids)?;
    output.set_item("position_ids", batch.position_ids)?;
    output.set_item("payloads", batch.payloads)?;
    output.set_item("attention_mask", batch.attention_mask)?;
    output.set_item("action_targets", batch.action_targets)?;
    output.set_item("action_target_mask", batch.action_target_mask)?;
    output.set_item("outcome_targets", batch.outcome_targets)?;
    output.set_item("outcome_target_mask", batch.outcome_target_mask)?;
    output.set_item("lengths", batch.lengths)?;
    Ok(output)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    *, seed, start_index, batch_size, max_tokens,
    d_min=1, d_max=4, gain_min=0.75, gain_max=1.25,
    action_limit=0.20, calibration_pulse=0.10, max_control_steps=4
))]
fn generate_training_batch(
    py: Python<'_>,
    seed: u64,
    start_index: u64,
    batch_size: usize,
    max_tokens: usize,
    d_min: usize,
    d_max: usize,
    gain_min: f32,
    gain_max: f32,
    action_limit: f32,
    calibration_pulse: f32,
    max_control_steps: usize,
) -> PyResult<Py<PyAny>> {
    if batch_size == 0 || max_tokens == 0 {
        return Err(PyValueError::new_err(
            "batch_size and max_tokens must be nonzero",
        ));
    }
    let cfg = py_config(
        d_min,
        d_max,
        gain_min,
        gain_max,
        action_limit,
        calibration_pulse,
        max_control_steps,
    )?;
    let trajectories = (0..batch_size)
        .map(|offset| generate_trajectory(&cfg, seed, start_index + offset as u64))
        .collect::<Result<Vec<_>, _>>()
        .map_err(PyValueError::new_err)?;
    let records: Vec<&[LearningToken]> = trajectories.iter().map(|t| t.tokens.as_slice()).collect();
    let padded = pad_records(&records, max_tokens).map_err(PyValueError::new_err)?;
    let output = padded_to_dict(py, padded)?;
    output.set_item(
        "dimensions",
        trajectories.iter().map(|t| t.d).collect::<Vec<_>>(),
    )?;
    output.set_item(
        "indices",
        trajectories.iter().map(|t| t.index).collect::<Vec<_>>(),
    )?;
    output.set_item(
        "control_steps",
        trajectories
            .iter()
            .map(|t| t.control_steps)
            .collect::<Vec<_>>(),
    )?;
    output.set_item("world_version", WORLD_VERSION)?;
    output.set_item("oracle_version", ORACLE_VERSION)?;
    output.set_item("token_abi_version", TOKEN_ABI_VERSION)?;
    Ok(output.into_any().unbind())
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    *, seed, start_index, count,
    d_min=1, d_max=4, gain_min=0.75, gain_max=1.25,
    action_limit=0.20, calibration_pulse=0.10, max_control_steps=4
))]
fn validate_generated_worlds(
    py: Python<'_>,
    seed: u64,
    start_index: u64,
    count: usize,
    d_min: usize,
    d_max: usize,
    gain_min: f32,
    gain_max: f32,
    action_limit: f32,
    calibration_pulse: f32,
    max_control_steps: usize,
) -> PyResult<Py<PyAny>> {
    let cfg = py_config(
        d_min,
        d_max,
        gain_min,
        gain_max,
        action_limit,
        calibration_pulse,
        max_control_steps,
    )?;
    if count == 0 {
        return Err(PyValueError::new_err("count must be nonzero"));
    }
    let mut dimension_counts = vec![0usize; d_max + 1];
    let mut max_length = 0usize;
    let mut min_length = usize::MAX;
    let mut max_oracle_error = 0.0f32;
    let mut action_targets = 0usize;
    let mut outcome_targets = 0usize;
    for offset in 0..count {
        let trajectory = generate_trajectory(&cfg, seed, start_index + offset as u64)
            .map_err(PyValueError::new_err)?;
        dimension_counts[trajectory.d] += 1;
        max_length = max_length.max(trajectory.tokens.len());
        min_length = min_length.min(trajectory.tokens.len());
        max_oracle_error = max_oracle_error.max(trajectory.oracle_reconstruction_error);
        for token in &trajectory.tokens {
            action_targets += token.supervision.action_mask.iter().filter(|&&v| v).count();
            outcome_targets += usize::from(token.supervision.outcome_mask);
        }
    }
    let output = PyDict::new(py);
    output.set_item("count", count)?;
    output.set_item("dimension_counts", dimension_counts)?;
    output.set_item("min_length", min_length)?;
    output.set_item("max_length", max_length)?;
    output.set_item("max_oracle_error", max_oracle_error)?;
    output.set_item("action_targets", action_targets)?;
    output.set_item("outcome_targets", outcome_targets)?;
    output.set_item("world_version", WORLD_VERSION)?;
    output.set_item("oracle_version", ORACLE_VERSION)?;
    output.set_item("token_abi_version", TOKEN_ABI_VERSION)?;
    Ok(output.into_any().unbind())
}

#[pyclass(name = "RolloutBatch")]
struct PyRolloutBatch {
    cfg: FamilyConfig,
    episodes: Vec<RolloutEpisode>,
    max_tokens: usize,
}

#[pymethods]
impl PyRolloutBatch {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (
        *, seed, start_index, batch_size, max_tokens,
        d_min=1, d_max=4, gain_min=0.75, gain_max=1.25,
        action_limit=0.20, calibration_pulse=0.10, max_control_steps=4
    ))]
    fn new(
        seed: u64,
        start_index: u64,
        batch_size: usize,
        max_tokens: usize,
        d_min: usize,
        d_max: usize,
        gain_min: f32,
        gain_max: f32,
        action_limit: f32,
        calibration_pulse: f32,
        max_control_steps: usize,
    ) -> PyResult<Self> {
        let cfg = py_config(
            d_min,
            d_max,
            gain_min,
            gain_max,
            action_limit,
            calibration_pulse,
            max_control_steps,
        )?;
        let episodes = (0..batch_size)
            .map(|offset| RolloutEpisode::new(&cfg, seed, start_index + offset as u64))
            .collect::<Result<Vec<_>, _>>()
            .map_err(PyValueError::new_err)?;
        if episodes
            .iter()
            .any(|episode| episode.tokens.len() > max_tokens)
        {
            return Err(PyValueError::new_err(
                "initial rollout prefix exceeds max_tokens",
            ));
        }
        Ok(Self {
            cfg,
            episodes,
            max_tokens,
        })
    }

    fn learner_batch(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let records: Vec<&[LearningToken]> =
            self.episodes.iter().map(|e| e.tokens.as_slice()).collect();
        let padded = pad_records(&records, self.max_tokens).map_err(PyValueError::new_err)?;
        let output = padded_to_dict(py, padded)?;
        let mut positions = vec![vec![-1i64; self.cfg.d_max]; self.episodes.len()];
        let mut keys = vec![vec![-1i64; self.cfg.d_max]; self.episodes.len()];
        for (row, episode) in self.episodes.iter().enumerate() {
            if episode.done {
                continue;
            }
            let current = episode.current_query_positions();
            for (slot, (&position, &key)) in current.iter().zip(&episode.query_order).enumerate() {
                positions[row][slot] = position as i64;
                keys[row][slot] = key as i64;
            }
        }
        output.set_item("query_positions", positions)?;
        output.set_item("query_keys", keys)?;
        output.set_item(
            "dimensions",
            self.episodes
                .iter()
                .map(|e| e.instance.d)
                .collect::<Vec<_>>(),
        )?;
        output.set_item(
            "done",
            self.episodes.iter().map(|e| e.done).collect::<Vec<_>>(),
        )?;
        Ok(output.into_any().unbind())
    }

    /// Apply one normalized action list in the current public query order for
    /// every episode. Completed episodes must receive an empty list.
    fn step(&mut self, actions: Vec<Vec<f32>>) -> PyResult<()> {
        if actions.len() != self.episodes.len() {
            return Err(PyValueError::new_err(
                "expected one action list per rollout episode",
            ));
        }
        for (episode, action) in self.episodes.iter_mut().zip(actions) {
            if episode.done {
                if !action.is_empty() {
                    return Err(PyValueError::new_err(
                        "completed episode received a nonempty action",
                    ));
                }
                continue;
            }
            episode
                .step_normalized(&self.cfg, &action)
                .map_err(PyValueError::new_err)?;
            if episode.tokens.len() > self.max_tokens {
                return Err(PyValueError::new_err(format!(
                    "rollout grew to {} tokens beyond max_tokens {}; truncation is forbidden",
                    episode.tokens.len(),
                    self.max_tokens
                )));
            }
        }
        Ok(())
    }

    fn all_done(&self) -> bool {
        self.episodes.iter().all(|episode| episode.done)
    }

    fn summary(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let output = PyDict::new(py);
        let success: Vec<bool> = self.episodes.iter().map(|e| e.success).collect();
        let errors: Vec<f32> = self.episodes.iter().map(|e| e.terminal_error()).collect();
        let steps: Vec<usize> = self.episodes.iter().map(|e| e.steps).collect();
        output.set_item("success", success)?;
        output.set_item("terminal_error", errors)?;
        output.set_item("steps", steps)?;
        output.set_item(
            "dimensions",
            self.episodes
                .iter()
                .map(|e| e.instance.d)
                .collect::<Vec<_>>(),
        )?;
        Ok(output.into_any().unbind())
    }

    /// Validation-only oracle actions, normalized and ordered exactly like the
    /// public action queries. Never use this method to construct model inputs.
    fn privileged_oracle_actions(&self) -> PyResult<Vec<Vec<f32>>> {
        self.episodes
            .iter()
            .map(|episode| {
                if episode.done {
                    return Ok(Vec::new());
                }
                let action = episode.oracle.action(&episode.x, &episode.goal)?;
                Ok(episode
                    .query_order
                    .iter()
                    .map(|&j| action[j] / self.cfg.action_limit)
                    .collect())
            })
            .collect::<Result<Vec<_>, String>>()
            .map_err(PyValueError::new_err)
    }
}

#[pyfunction]
fn versions(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let output = PyDict::new(py);
    output.set_item("world", WORLD_VERSION)?;
    output.set_item("oracle", ORACLE_VERSION)?;
    output.set_item("token_abi", TOKEN_ABI_VERSION)?;
    output.set_item("role_count", Role::COUNT)?;
    output.set_item("payload_dim", PAYLOAD_DIM)?;
    output.set_item("action_horizon", ACTION_HORIZON)?;
    Ok(output.into_any().unbind())
}

#[pymodule]
fn step2_world_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(generate_training_batch, module)?)?;
    module.add_function(wrap_pyfunction!(validate_generated_worlds, module)?)?;
    module.add_function(wrap_pyfunction!(versions, module)?)?;
    module.add_class::<PyRolloutBatch>()?;
    Ok(())
}
