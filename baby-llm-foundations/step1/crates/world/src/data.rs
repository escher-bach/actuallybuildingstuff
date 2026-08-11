//! Deterministic sequence packing, replay records, and self-describing shards.
//!
//! The on-disk format intentionally uses only `std`: a compact binary payload
//! for trainer buffers plus UTF-8 manifest and replay sidecars. This keeps the
//! first data path auditable and portable before choosing a larger dataset
//! container format.

use std::fs;
use std::path::{Path, PathBuf};

use crate::generate::{sample, FamilyParams};
use crate::render::Rendering;
use crate::trajectory::{
    generate_teacher_trajectory, pack_teacher_trajectory, FragmentTokenizer, PackedTrajectory,
    TeacherTrajectory, TrajectoryError,
};
use crate::{Action, Instance, Variant};

pub const WORLD_FAMILY_VERSION: &str = "world-0.1.0";
pub const GENERATOR_VERSION: &str = "world-generate-0.1.0";
pub const TEACHER_POLICY_VERSION: &str = "world-teacher-0.1.0";
pub const SHARD_FORMAT_VERSION: u32 = 1;
const MAX_REJECTION_ATTEMPTS: u64 = 1_000_000;

/// Identity of the tokenizer that produced `token_ids`. A training run must
/// reject a shard when any of these fields differ from its configured tokenizer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerIdentity {
    pub name: String,
    pub revision: String,
    pub vocabulary_hash: String,
}

impl TokenizerIdentity {
    pub fn byte_utf8() -> Self {
        Self {
            name: "byte-utf8-transport".to_string(),
            revision: "v1".to_string(),
            // Hash of IDs 0..255 plus fixed PAD/BOS/EOS/OBS/ACTION/END_TURN.
            vocabulary_hash: "dcb84350b969fbcbec931b50af29309af4956c606f02fa5a40950c6cd36e104e".to_string(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardSpec {
    pub params: FamilyParams,
    pub root_seed: u64,
    pub first_instance_index: u64,
    pub episode_count: usize,
    pub rendering: Rendering,
    /// A trajectory is never split. A new sequence starts when adding the
    /// next complete trajectory would exceed this capacity.
    pub max_sequence_tokens: usize,
    pub tokenizer: TokenizerIdentity,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReplayRecord {
    pub world_family_version: String,
    pub generator_version: String,
    pub teacher_policy_version: String,
    pub root_seed: u64,
    pub instance_index: u64,
    pub params: FamilyParams,
    pub rendering: Rendering,
    pub selected_actions: Vec<Action>,
    pub tokenizer: TokenizerIdentity,
}

/// Half-open range for one source trajectory within a packed sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExampleRange {
    pub replay_record_index: usize,
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PackedSequence {
    pub token_ids: Vec<u32>,
    pub loss_mask: Vec<u8>,
    pub target_channel_ids: Vec<u8>,
    pub examples: Vec<ExampleRange>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetManifest {
    pub format_version: u32,
    pub world_family_version: String,
    pub generator_version: String,
    pub teacher_policy_version: String,
    pub tokenizer: TokenizerIdentity,
    pub root_seed: u64,
    pub first_instance_index: u64,
    pub last_examined_instance_index: u64,
    pub rendering: Rendering,
    pub params: FamilyParams,
    pub episode_count: usize,
    pub sequence_count: usize,
    pub token_count: usize,
    pub data_sha256: String,
    pub replay_sha256: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DatasetShard {
    pub sequences: Vec<PackedSequence>,
    pub replay_records: Vec<ReplayRecord>,
    pub manifest: DatasetManifest,
}

#[derive(Debug)]
pub enum DatasetError {
    SamplingExhausted {
        requested: usize,
    },
    Trajectory(TrajectoryError),
    TrajectoryTooLong {
        tokens: usize,
        max_sequence_tokens: usize,
    },
    ReplayMismatch {
        instance_index: u64,
    },
    Io(std::io::Error),
}

impl From<TrajectoryError> for DatasetError {
    fn from(value: TrajectoryError) -> Self {
        Self::Trajectory(value)
    }
}

impl From<std::io::Error> for DatasetError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value)
    }
}

/// Samples accepted instances deterministically, generates teacher rollouts,
/// and greedily packs complete examples in source order. There is no random
/// shuffle or dependency on filesystem ordering.
pub fn generate_dataset_shard<T: FragmentTokenizer>(
    spec: &ShardSpec,
    tokenizer: &T,
) -> Result<DatasetShard, DatasetError> {
    let mut records = Vec::with_capacity(spec.episode_count);
    let mut packed_examples = Vec::with_capacity(spec.episode_count);
    let mut index = spec.first_instance_index;
    let stop = index.saturating_add(MAX_REJECTION_ATTEMPTS);

    while records.len() < spec.episode_count && index < stop {
        if let Ok(instance) = sample(&spec.params, spec.root_seed, index) {
            let trajectory = generate_teacher_trajectory(&instance, spec.rendering)?;
            let packed = pack_teacher_trajectory(&trajectory, tokenizer);
            if packed.token_ids.len() > spec.max_sequence_tokens {
                return Err(DatasetError::TrajectoryTooLong {
                    tokens: packed.token_ids.len(),
                    max_sequence_tokens: spec.max_sequence_tokens,
                });
            }
            records.push(replay_record(
                &spec.params,
                spec.root_seed,
                &instance,
                &trajectory,
                spec.tokenizer.clone(),
            ));
            packed_examples.push(packed);
        }
        index += 1;
    }
    if records.len() != spec.episode_count {
        return Err(DatasetError::SamplingExhausted {
            requested: spec.episode_count,
        });
    }

    let sequences = pack_sequences(packed_examples, spec.max_sequence_tokens);
    let binary = encode_shard_binary(&sequences);
    let replay_text = encode_replay_records(&records);
    let manifest = DatasetManifest {
        format_version: SHARD_FORMAT_VERSION,
        world_family_version: WORLD_FAMILY_VERSION.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        teacher_policy_version: TEACHER_POLICY_VERSION.to_string(),
        tokenizer: spec.tokenizer.clone(),
        root_seed: spec.root_seed,
        first_instance_index: spec.first_instance_index,
        last_examined_instance_index: index.saturating_sub(1),
        rendering: spec.rendering,
        params: spec.params,
        episode_count: records.len(),
        sequence_count: sequences.len(),
        token_count: sequences.iter().map(|s| s.token_ids.len()).sum(),
        data_sha256: sha256_hex(&binary),
        replay_sha256: sha256_hex(replay_text.as_bytes()),
    };
    Ok(DatasetShard {
        sequences,
        replay_records: records,
        manifest,
    })
}

fn replay_record(
    params: &FamilyParams,
    root_seed: u64,
    instance: &Instance,
    trajectory: &TeacherTrajectory,
    tokenizer: TokenizerIdentity,
) -> ReplayRecord {
    ReplayRecord {
        world_family_version: WORLD_FAMILY_VERSION.to_string(),
        generator_version: GENERATOR_VERSION.to_string(),
        teacher_policy_version: TEACHER_POLICY_VERSION.to_string(),
        root_seed,
        instance_index: instance.index,
        params: *params,
        rendering: trajectory.rendering,
        selected_actions: trajectory
            .steps
            .iter()
            .map(|step| step.selected_action)
            .collect(),
        tokenizer,
    }
}

fn pack_sequences(examples: Vec<PackedTrajectory>, max_tokens: usize) -> Vec<PackedSequence> {
    let mut sequences = Vec::new();
    let mut current = empty_sequence();
    for (record_index, example) in examples.into_iter().enumerate() {
        if !current.token_ids.is_empty()
            && current.token_ids.len() + example.token_ids.len() > max_tokens
        {
            sequences.push(current);
            current = empty_sequence();
        }
        let start = current.token_ids.len();
        let end = start + example.token_ids.len();
        current.token_ids.extend(example.token_ids);
        current.loss_mask.extend(example.loss_mask);
        current
            .target_channel_ids
            .extend(example.target_channel_ids);
        current.examples.push(ExampleRange {
            replay_record_index: record_index,
            start,
            end,
        });
    }
    if !current.token_ids.is_empty() {
        sequences.push(current);
    }
    sequences
}

fn empty_sequence() -> PackedSequence {
    PackedSequence {
        token_ids: Vec::new(),
        loss_mask: Vec::new(),
        target_channel_ids: Vec::new(),
        examples: Vec::new(),
    }
}

/// Reconstructs and validates an entire teacher rollout using only its replay
/// record. It does not need the original shard buffers or any learner state.
pub fn replay_record_matches(record: &ReplayRecord) -> Result<bool, DatasetError> {
    if record.world_family_version != WORLD_FAMILY_VERSION
        || record.generator_version != GENERATOR_VERSION
        || record.teacher_policy_version != TEACHER_POLICY_VERSION
    {
        return Ok(false);
    }
    let instance =
        sample(&record.params, record.root_seed, record.instance_index).map_err(|_| {
            DatasetError::ReplayMismatch {
                instance_index: record.instance_index,
            }
        })?;
    let trajectory = generate_teacher_trajectory(&instance, record.rendering)?;
    Ok(trajectory
        .steps
        .iter()
        .map(|s| s.selected_action)
        .eq(record.selected_actions.iter().copied()))
}

/// Writes three deterministic files: `<stem>.bin`, `<stem>.manifest`, and
/// `<stem>.replay`. Existing files of the same stem are intentionally replaced
/// as one explicit shard-generation operation.
pub fn write_dataset_shard(
    directory: &Path,
    stem: &str,
    shard: &DatasetShard,
) -> Result<(PathBuf, PathBuf, PathBuf), DatasetError> {
    fs::create_dir_all(directory)?;
    let binary_path = directory.join(format!("{stem}.bin"));
    let manifest_path = directory.join(format!("{stem}.manifest"));
    let replay_path = directory.join(format!("{stem}.replay"));
    fs::write(&binary_path, encode_shard_binary(&shard.sequences))?;
    fs::write(&manifest_path, encode_manifest(&shard.manifest))?;
    fs::write(&replay_path, encode_replay_records(&shard.replay_records))?;
    Ok((binary_path, manifest_path, replay_path))
}

/// Stable binary layout for trainer buffers. All integers are little-endian.
pub fn encode_shard_binary(sequences: &[PackedSequence]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"BLMSHRD1");
    put_u32(&mut out, SHARD_FORMAT_VERSION);
    put_u64(&mut out, sequences.len() as u64);
    for sequence in sequences {
        assert_eq!(sequence.token_ids.len(), sequence.loss_mask.len());
        assert_eq!(sequence.token_ids.len(), sequence.target_channel_ids.len());
        put_u64(&mut out, sequence.token_ids.len() as u64);
        put_u64(&mut out, sequence.examples.len() as u64);
        for &id in &sequence.token_ids {
            put_u32(&mut out, id);
        }
        out.extend_from_slice(&sequence.loss_mask);
        out.extend_from_slice(&sequence.target_channel_ids);
        for range in &sequence.examples {
            put_u64(&mut out, range.replay_record_index as u64);
            put_u64(&mut out, range.start as u64);
            put_u64(&mut out, range.end as u64);
        }
    }
    out
}

pub fn encode_manifest(manifest: &DatasetManifest) -> String {
    let p = &manifest.params;
    format!(
        "format_version={}\nworld_family_version={}\ngenerator_version={}\nteacher_policy_version={}\ntokenizer_name={}\ntokenizer_revision={}\ntokenizer_vocabulary_hash={}\nroot_seed={}\nfirst_instance_index={}\nlast_examined_instance_index={}\nrendering={}\nvariant={}\nn_hyp={}\nn_probe={}\nn_evidence={}\ncost_lo={}\ncost_hi={}\nbudget_slack={}\nmin_depth={}\nstep_slack={}\nepisode_count={}\nsequence_count={}\ntoken_count={}\ndata_sha256={}\nreplay_sha256={}\n",
        manifest.format_version, manifest.world_family_version, manifest.generator_version,
        manifest.teacher_policy_version, manifest.tokenizer.name, manifest.tokenizer.revision,
        manifest.tokenizer.vocabulary_hash, manifest.root_seed, manifest.first_instance_index,
        manifest.last_examined_instance_index, rendering_name(manifest.rendering), variant_name(p.variant),
        p.n_hyp, p.n_probe, p.n_evidence, p.cost_lo, p.cost_hi, p.budget_slack, p.min_depth,
        p.step_slack, manifest.episode_count, manifest.sequence_count, manifest.token_count,
        manifest.data_sha256, manifest.replay_sha256,
    )
}

pub fn encode_replay_records(records: &[ReplayRecord]) -> String {
    let mut output = String::from("world_family_version\tgenerator_version\tteacher_policy_version\troot_seed\tinstance_index\trendering\tvariant\tn_hyp\tn_probe\tn_evidence\tcost_lo\tcost_hi\tbudget_slack\tmin_depth\tstep_slack\ttokenizer_name\ttokenizer_revision\ttokenizer_vocabulary_hash\tactions\n");
    for record in records {
        let p = &record.params;
        let actions = record
            .selected_actions
            .iter()
            .map(action_name)
            .collect::<Vec<_>>()
            .join(",");
        output.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            record.world_family_version,
            record.generator_version,
            record.teacher_policy_version,
            record.root_seed,
            record.instance_index,
            rendering_name(record.rendering),
            variant_name(p.variant),
            p.n_hyp,
            p.n_probe,
            p.n_evidence,
            p.cost_lo,
            p.cost_hi,
            p.budget_slack,
            p.min_depth,
            p.step_slack,
            record.tokenizer.name,
            record.tokenizer.revision,
            record.tokenizer.vocabulary_hash,
            actions,
        ));
    }
    output
}

fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}
fn rendering_name(rendering: Rendering) -> &'static str {
    match rendering {
        Rendering::A => "a",
        Rendering::B => "b",
    }
}
fn variant_name(variant: Variant) -> &'static str {
    match variant {
        Variant::Irreversible => "irreversible",
        Variant::Reversible => "reversible",
    }
}
fn action_name(action: &Action) -> String {
    match action {
        Action::Inspect(q) => format!("inspect:{q}"),
        Action::Commit(h) => format!("commit:{h}"),
    }
}

/// Minimal dependency-free SHA-256 implementation used for content-addressed
/// manifest fields. The output is standard lowercase hexadecimal SHA-256.
pub fn sha256_hex(input: &[u8]) -> String {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut bytes = input.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    bytes.push(0x80);
    while bytes.len() % 64 != 56 {
        bytes.push(0);
    }
    bytes.extend_from_slice(&bit_len.to_be_bytes());
    let mut h = [
        0x6a09e667u32,
        0xbb67ae85,
        0x3c6ef372,
        0xa54ff53a,
        0x510e527f,
        0x9b05688c,
        0x1f83d9ab,
        0x5be0cd19,
    ];
    for chunk in bytes.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes(word.try_into().unwrap());
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let (mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh) =
            (h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let t1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let t2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(t1);
            d = c;
            c = b;
            b = a;
            a = t1.wrapping_add(t2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    h.iter().map(|word| format!("{word:08x}")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trajectory::ByteTokenizer;

    fn spec() -> ShardSpec {
        ShardSpec {
            params: FamilyParams {
                n_hyp: 6,
                n_probe: 5,
                n_evidence: 2,
                cost_lo: 1,
                cost_hi: 3,
                budget_slack: 1,
                min_depth: 2,
                step_slack: 2,
                variant: Variant::Reversible,
            },
            root_seed: 20260811,
            first_instance_index: 0,
            episode_count: 8,
            rendering: Rendering::A,
            max_sequence_tokens: 20_000,
            tokenizer: TokenizerIdentity::byte_utf8(),
        }
    }

    #[test]
    fn sha256_matches_standard_test_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn generation_and_packing_are_deterministic_and_whole_example() {
        let one = generate_dataset_shard(&spec(), &ByteTokenizer).unwrap();
        let two = generate_dataset_shard(&spec(), &ByteTokenizer).unwrap();
        assert_eq!(one, two);
        assert_eq!(one.manifest.episode_count, one.replay_records.len());
        assert_eq!(
            one.manifest.data_sha256,
            sha256_hex(&encode_shard_binary(&one.sequences))
        );
        for sequence in &one.sequences {
            assert_eq!(sequence.token_ids.len(), sequence.loss_mask.len());
            assert_eq!(sequence.token_ids.len(), sequence.target_channel_ids.len());
            for range in &sequence.examples {
                assert!(range.start < range.end);
                assert!(range.end <= sequence.token_ids.len());
                assert!(range.replay_record_index < one.replay_records.len());
            }
        }
    }

    #[test]
    fn deterministic_packing_never_splits_a_trajectory() {
        let full = generate_dataset_shard(&spec(), &ByteTokenizer).unwrap();
        let longest = full.sequences[0]
            .examples
            .iter()
            .map(|range| range.end - range.start)
            .max()
            .unwrap();
        let mut constrained = spec();
        constrained.max_sequence_tokens = longest;
        let shard = generate_dataset_shard(&constrained, &ByteTokenizer).unwrap();
        assert!(shard.sequences.len() > 1);
        for sequence in &shard.sequences {
            for range in &sequence.examples {
                assert!(range.start < range.end);
                assert!(range.end <= sequence.token_ids.len());
            }
        }
    }

    #[test]
    fn every_complete_replay_record_reconstructs_its_trajectory() {
        let shard = generate_dataset_shard(&spec(), &ByteTokenizer).unwrap();
        for record in &shard.replay_records {
            assert!(replay_record_matches(record).unwrap());
        }
    }

    #[test]
    fn manifest_and_replay_are_auditable_without_literal_truth_field() {
        let shard = generate_dataset_shard(&spec(), &ByteTokenizer).unwrap();
        let manifest = encode_manifest(&shard.manifest);
        let replay = encode_replay_records(&shard.replay_records);
        assert!(manifest.contains("tokenizer_vocabulary_hash="));
        assert!(manifest.contains("data_sha256="));
        assert!(manifest.contains("replay_sha256="));
        assert!(!manifest.contains("truth="));
        assert!(!replay.contains("truth"));
        assert!(
            replay.starts_with("world_family_version\tgenerator_version\tteacher_policy_version\t")
        );
        assert_eq!(shard.manifest.replay_sha256, sha256_hex(replay.as_bytes()));
    }

    #[test]
    fn shard_writer_emits_hashed_payload_and_sidecars() {
        let shard = generate_dataset_shard(&spec(), &ByteTokenizer).unwrap();
        let dir = std::env::temp_dir().join(format!("baby-llm-shard-{}", std::process::id()));
        let (data, manifest, replay) = write_dataset_shard(&dir, "teacher", &shard).unwrap();
        assert_eq!(
            sha256_hex(&fs::read(&data).unwrap()),
            shard.manifest.data_sha256
        );
        assert_eq!(
            sha256_hex(&fs::read(&replay).unwrap()),
            shard.manifest.replay_sha256
        );
        assert!(fs::read_to_string(manifest)
            .unwrap()
            .contains("format_version=1"));
    }
}
