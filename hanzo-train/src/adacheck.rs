//! AdaCheck: adaptive redundancy-aware distributed training checkpoints.
//!
//! Clean-room implementation inspired by redundancy-aware checkpointing
//! (AdaCheck, FAST '26) for distributed training in `hanzo-train`:
//!
//! 1. **Content-addressed sub-tensor chunking**: model weights, frozen buffers,
//!    and per-worker optimizer state shards (`m`, `v`, rounding `carry`) are
//!    partitioned into fixed-size pages and addressed by domain-separated SHA-256
//!    digests ([`ShardDigest`]) that map directly to S3/VFS object keys.
//! 2. **Multi-level redundancy detection**:
//!    - **Zero suppression**: all-zero chunks (common in cold optimizer states
//!      and carries) emit zero payload bytes.
//!    - **Intra-step cross-worker deduplication**: replicated weights, frozen
//!      parameters, and shared ZeRO-group shards across workers share one object.
//!    - **Inter-step temporal deduplication**: unchanged chunks across sequential
//!      checkpoint steps reference the prior digest at zero payload cost.
//! 3. **Bit-exact adaptive delta encoding**:
//!    - Compares raw chunk bytes against the step's prior block using two
//!      lossless byte-space encoders: sparse zero-run XOR RLE (ideal for sparse
//!      MoE/embedding/LoRA updates) and IEEE-754 byte-lane transposed XOR RLE
//!      (ideal for small dense `f32`/`bf16` updates where sign and upper exponent
//!      bytes remain unchanged).
//!    - Selects the smallest representation when its encoded length beats
//!      [`AdaCheckConfig::delta_ratio_threshold`], falling back to a fresh
//!      keyframe chunk whenever delta savings vanish or the chain reaches
//!      [`AdaCheckConfig::max_delta_chain`].
//! 4. **Verified bit-exact reconstruction**: every restored block is checked
//!    against its original length and SHA-256 digest, preserving `-0.0`,
//!    subnormals, and NaN bit patterns identically.

use crate::cluster::coordinator::Inner;
use crate::cluster::net::Params;
use anyhow::{ensure, Context, Result};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Instant;

const DIGEST_DOMAIN: &[u8] = b"hanzo.train.adacheck.chunk.v1\0";
/// Minimum zero-byte gap required to split a literal XOR run (header = 8 bytes).
const MIN_ZERO_RUN_SPLIT: usize = 9;

/// Semantic role of a tensor or optimizer buffer in a distributed checkpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum ShardRole {
    /// Trainable model parameters (`theta`).
    Weight,
    /// Frozen model parameters shared identically across workers/steps.
    FrozenWeight,
    /// Optimizer first-moment buffer (`m`).
    OptimizerFirstMoment,
    /// Optimizer second-moment buffer (`v`).
    OptimizerSecondMoment,
    /// Mixed-precision rounding residual (`carry`) or master-weight residual.
    OptimizerCarry,
    /// Accumulated gradient buffer.
    Gradient,
    /// Auxiliary checkpoint metadata or custom shard.
    Other,
}

/// Element representation used to choose byte-lane transposition stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum TensorDType {
    F32,
    F16,
    BF16,
    I64,
    I32,
    U8,
    Bytes,
}

impl TensorDType {
    /// Element width in bytes; `1` indicates unaligned/raw byte stream.
    #[inline]
    pub const fn element_bytes(self) -> usize {
        match self {
            Self::F32 | Self::I32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::I64 => 8,
            Self::U8 | Self::Bytes => 1,
        }
    }
}

/// Content-addressed SHA-256 digest of an uncompressed shard block.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ShardDigest([u8; 32]);

impl ShardDigest {
    /// Compute the domain-separated, length-prefixed SHA-256 digest of `bytes`.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(DIGEST_DOMAIN);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
        Self(h.finalize().into())
    }

    /// Raw 32-byte digest.
    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lowercase 64-character hexadecimal representation.
    pub fn hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for b in self.0 {
            out.push(char::from_digit((b >> 4) as u32, 16).expect("hex nibble"));
            out.push(char::from_digit((b & 0x0f) as u32, 16).expect("hex nibble"));
        }
        out
    }

    /// Canonical S3/VFS object publication key for this content-addressed object.
    pub fn object_key(&self) -> String {
        let h = self.hex();
        format!("adacheck/chunks/{}/{}.bin", &h[..2], h)
    }
}

impl std::fmt::Display for ShardDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.hex())
    }
}

impl std::fmt::Debug for ShardDigest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ShardDigest({})", &self.hex()[..12])
    }
}

/// Configuration controlling AdaCheck chunking, deduplication, and delta encoding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdaCheckConfig {
    /// Target sub-tensor page size in bytes (must be > 0).
    pub chunk_bytes: usize,
    /// Maximum sequential delta hops before emitting a fresh full keyframe chunk.
    pub max_delta_chain: usize,
    /// Emit a delta only if `delta_bytes <= delta_ratio_threshold * raw_bytes`.
    pub delta_ratio_threshold: f64,
    /// Evaluate byte-lane transposed XOR+RLE for multi-byte dtypes.
    pub enable_byte_lane_xor: bool,
    /// Deduplicate identical chunks across workers within the same step.
    pub dedup_across_workers: bool,
    /// Deduplicate unchanged chunks against prior steps.
    pub dedup_across_steps: bool,
}

impl Default for AdaCheckConfig {
    fn default() -> Self {
        Self {
            chunk_bytes: 4096,
            max_delta_chain: 8,
            delta_ratio_threshold: 0.85,
            enable_byte_lane_xor: true,
            dedup_across_workers: true,
            dedup_across_steps: true,
        }
    }
}

impl AdaCheckConfig {
    fn validate(&self) -> Result<()> {
        ensure!(self.chunk_bytes > 0, "chunk_bytes must be > 0");
        ensure!(
            self.delta_ratio_threshold.is_finite()
                && self.delta_ratio_threshold > 0.0
                && self.delta_ratio_threshold <= 1.0,
            "delta_ratio_threshold must be in (0.0, 1.0], got {}",
            self.delta_ratio_threshold
        );
        Ok(())
    }
}

/// A borrowed tensor or optimizer shard presented for checkpointing.
#[derive(Debug, Clone)]
pub struct TensorInput<'a> {
    pub name: &'a str,
    pub role: ShardRole,
    pub dtype: TensorDType,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
}

/// One distributed worker's complete tensor/optimizer state at a checkpoint step.
#[derive(Debug, Clone)]
pub struct WorkerCheckpointInput<'a> {
    pub worker_id: &'a str,
    pub tensors: Vec<TensorInput<'a>>,
}

/// Reconstructed tensor restored from an AdaCheck checkpoint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconstructedTensor {
    pub name: String,
    pub role: ShardRole,
    pub dtype: TensorDType,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

impl ReconstructedTensor {
    /// Interpret reconstructed raw little-endian bytes as `f32` elements.
    pub fn as_f32_le(&self) -> Result<Vec<f32>> {
        ensure!(
            self.bytes.len().is_multiple_of(4),
            "tensor {} has {} bytes, not divisible by 4",
            self.name,
            self.bytes.len()
        );
        Ok(self
            .bytes
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
            .collect())
    }
}

/// Source classification when a chunk is deduplicated without writing payload bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DedupSource {
    /// Matched another worker's chunk within the same checkpoint step.
    IntraStepCrossWorker,
    /// Matched an earlier chunk on the same worker within the same step.
    IntraStepSameWorker,
    /// Matched a chunk from a prior checkpoint step.
    InterStep,
}

/// Storage encoding of one sub-tensor block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockEncoding {
    /// Block is identically all zero bytes; zero object bytes stored.
    Zero,
    /// Exact content match with an existing block (`origin_step` locates non-full refs).
    Ref {
        digest: ShardDigest,
        source: DedupSource,
        origin_step: u64,
    },
    /// Full uncompressed raw block stored in [`ChunkStore`] under `digest`.
    Full { digest: ShardDigest },
    /// Lossless sparse zero-run RLE over `current XOR base`.
    XorSparseDelta {
        base_step: u64,
        base_digest: ShardDigest,
        delta_digest: ShardDigest,
        chain_depth: usize,
    },
    /// Lossless byte-lane transposed sparse zero-run RLE over `current XOR base`.
    BytePlaneXorDelta {
        base_step: u64,
        base_digest: ShardDigest,
        delta_digest: ShardDigest,
        elem_width: usize,
        chain_depth: usize,
    },
}

/// Descriptor for one fixed-size block inside a checkpointed tensor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockRef {
    pub index: usize,
    pub raw_len: usize,
    pub expected_digest: ShardDigest,
    pub encoding: BlockEncoding,
}

/// Descriptor for one checkpointed tensor or optimizer shard.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorRecord {
    pub name: String,
    pub role: ShardRole,
    pub dtype: TensorDType,
    pub shape: Vec<usize>,
    pub raw_bytes: usize,
    pub blocks: Vec<BlockRef>,
}

/// Per-worker checkpoint manifest entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkerManifest {
    pub worker_id: String,
    pub tensors: Vec<TensorRecord>,
}

/// Redundancy and byte-reduction statistics for a single checkpoint step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CheckpointMetrics {
    pub step: u64,
    pub total_raw_bytes: u64,
    pub new_objects_written: usize,
    pub emitted_payload_bytes: u64,
    pub manifest_bytes: u64,
    pub total_checkpoint_bytes: u64,
    pub zero_saved_bytes: u64,
    pub intra_step_dedup_bytes: u64,
    pub inter_step_dedup_bytes: u64,
    pub delta_raw_bytes: u64,
    pub delta_payload_bytes: u64,
    pub delta_saved_bytes: u64,
    pub full_chunk_bytes: u64,
    pub max_chain_depth_seen: usize,
    /// `emitted_payload_bytes / total_raw_bytes` (0.0 if `total_raw_bytes == 0`).
    pub payload_ratio: f64,
    /// `100.0 * (1.0 - payload_ratio)`.
    pub byte_reduction_pct: f64,
    /// Measured encoding throughput in MiB/s.
    pub encode_mib_per_s: f64,
}

/// Complete serializable manifest for one distributed checkpoint step.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepCheckpoint {
    pub step: u64,
    pub workers: BTreeMap<String, WorkerManifest>,
    pub metrics: CheckpointMetrics,
}

impl StepCheckpoint {
    /// Serialize this manifest to canonical JSON bytes for S3/VFS metadata upload.
    pub fn to_json_bytes(&self) -> Result<Vec<u8>> {
        Ok(serde_json::to_vec(self)?)
    }
}

/// Non-mutating analysis summary produced by [`RedundancyDetector`].
#[derive(Debug, Clone, PartialEq)]
pub struct RedundancyReport {
    pub step: u64,
    pub total_chunks: usize,
    pub unique_chunks: usize,
    pub total_raw_bytes: u64,
    pub zero_bytes: u64,
    pub intra_step_duplicate_bytes: u64,
    pub inter_step_duplicate_bytes: u64,
    pub net_unique_novel_bytes: u64,
    pub raw_redundancy_pct: f64,
}

/// In-memory content-addressed object catalog mirroring S3/VFS immutable chunk storage.
#[derive(Debug, Default, Clone)]
pub struct ChunkStore {
    objects: BTreeMap<ShardDigest, Vec<u8>>,
    total_puts: u64,
    dedup_hits: u64,
}

impl ChunkStore {
    /// Create an empty content-addressed chunk store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Store `bytes` under `digest` if absent. Returns `true` if newly inserted.
    pub fn put_if_absent(&mut self, digest: ShardDigest, bytes: Vec<u8>) -> bool {
        self.total_puts += 1;
        match self.objects.entry(digest) {
            std::collections::btree_map::Entry::Vacant(e) => {
                e.insert(bytes);
                true
            }
            std::collections::btree_map::Entry::Occupied(_) => {
                self.dedup_hits += 1;
                false
            }
        }
    }

    /// Look up raw object bytes by digest.
    pub fn get(&self, digest: &ShardDigest) -> Option<&[u8]> {
        self.objects.get(digest).map(Vec::as_slice)
    }

    /// Number of unique immutable objects currently stored.
    pub fn object_count(&self) -> usize {
        self.objects.len()
    }

    /// Total `put_if_absent` calls received by this store.
    pub fn total_puts(&self) -> u64 {
        self.total_puts
    }

    /// Number of `put_if_absent` calls deduplicated against existing objects.
    pub fn dedup_hits(&self) -> u64 {
        self.dedup_hits
    }

    /// Total payload bytes stored across all unique immutable objects.
    pub fn stored_payload_bytes(&self) -> u64 {
        self.objects.values().map(|v| v.len() as u64).sum()
    }

    /// Export `(s3_object_key, payload)` pairs sorted by digest for object upload.
    pub fn objects_for_publication(&self) -> Vec<(String, &[u8])> {
        self.objects
            .iter()
            .map(|(d, b)| (d.object_key(), b.as_slice()))
            .collect()
    }

    /// Mutably corrupt an object in tests to verify tamper/corruption rejection.
    #[cfg(test)]
    pub(crate) fn corrupt_object_for_test(
        &mut self,
        digest: &ShardDigest,
        byte_idx: usize,
    ) -> bool {
        if let Some(buf) = self.objects.get_mut(digest) {
            if let Some(b) = buf.get_mut(byte_idx) {
                *b ^= 0x01;
                return true;
            }
        }
        false
    }
}

/// Stateless/incremental detector that measures exact chunk-level redundancy across workers and steps.
#[derive(Debug, Clone)]
pub struct RedundancyDetector {
    chunk_bytes: usize,
    steps_recorded: u64,
    known_history: HashSet<ShardDigest>,
}

impl RedundancyDetector {
    /// Build a detector using `chunk_bytes` fixed page granularity.
    pub fn new(chunk_bytes: usize) -> Result<Self> {
        ensure!(chunk_bytes > 0, "chunk_bytes must be > 0");
        Ok(Self {
            chunk_bytes,
            steps_recorded: 0,
            known_history: HashSet::new(),
        })
    }

    /// Analyze redundancy for `step` across `workers` without mutating internal history.
    pub fn inspect_step(
        &self,
        step: u64,
        workers: &[WorkerCheckpointInput<'_>],
    ) -> RedundancyReport {
        let mut total_chunks = 0usize;
        let mut total_raw_bytes = 0u64;
        let mut zero_bytes = 0u64;
        let mut intra_dup_bytes = 0u64;
        let mut inter_dup_bytes = 0u64;
        let mut step_seen = HashSet::new();

        for w in workers {
            for t in &w.tensors {
                for block in t.bytes.chunks(self.chunk_bytes) {
                    total_chunks += 1;
                    let len = block.len() as u64;
                    total_raw_bytes += len;
                    if is_all_zero(block) {
                        zero_bytes += len;
                        continue;
                    }
                    let d = ShardDigest::of_bytes(block);
                    if self.known_history.contains(&d) {
                        inter_dup_bytes += len;
                        step_seen.insert(d);
                    } else if !step_seen.insert(d) {
                        intra_dup_bytes += len;
                    }
                }
            }
        }

        let novel = total_raw_bytes
            .saturating_sub(zero_bytes)
            .saturating_sub(intra_dup_bytes)
            .saturating_sub(inter_dup_bytes);
        let redundant = total_raw_bytes.saturating_sub(novel);
        let raw_redundancy_pct = if total_raw_bytes == 0 {
            0.0
        } else {
            100.0 * (redundant as f64) / (total_raw_bytes as f64)
        };

        RedundancyReport {
            step,
            total_chunks,
            unique_chunks: step_seen.len(),
            total_raw_bytes,
            zero_bytes,
            intra_step_duplicate_bytes: intra_dup_bytes,
            inter_step_duplicate_bytes: inter_dup_bytes,
            net_unique_novel_bytes: novel,
            raw_redundancy_pct,
        }
    }

    /// Advance historical digest knowledge after committing a step.
    pub fn record_step(&mut self, workers: &[WorkerCheckpointInput<'_>]) -> RedundancyReport {
        let step_num = self.steps_recorded;
        let report = self.inspect_step(step_num, workers);
        for w in workers {
            for t in &w.tensors {
                for block in t.bytes.chunks(self.chunk_bytes) {
                    if !is_all_zero(block) {
                        self.known_history.insert(ShardDigest::of_bytes(block));
                    }
                }
            }
        }
        self.steps_recorded += 1;
        report
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct BlockLocationKey {
    worker_id: String,
    tensor_name: String,
    role: ShardRole,
    block_idx: usize,
}

#[derive(Debug, Clone)]
struct DigestOrigin {
    step: u64,
    worker_id: String,
    tensor_name: String,
    role: ShardRole,
    block_idx: usize,
    chain_depth: usize,
}

#[derive(Debug, Clone)]
struct PrevBlockState {
    step: u64,
    raw_digest: ShardDigest,
    chain_depth: usize,
    bytes: Vec<u8>,
}

/// Complete AdaCheck repository combining a content-addressed [`ChunkStore`] and step manifests.
///
/// Tracks cross-worker and inter-step redundancy, encodes adaptive bit-exact delta
/// checkpoints, and reconstructs any worker's full tensor/optimizer snapshot with
/// SHA-256 verification.
pub struct AdaCheckRepository {
    config: AdaCheckConfig,
    store: ChunkStore,
    manifests: BTreeMap<u64, StepCheckpoint>,
    /// Historical content-addressed digests mapped to their originating block location & depth.
    digest_origin: HashMap<ShardDigest, DigestOrigin>,
    /// Last committed raw block bytes per `(worker, tensor, role, block_idx)` for O(1) delta encode.
    prev_blocks: HashMap<BlockLocationKey, PrevBlockState>,
}

impl AdaCheckRepository {
    /// Create a new AdaCheck checkpoint repository with `config`.
    pub fn new(config: AdaCheckConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            store: ChunkStore::new(),
            manifests: BTreeMap::new(),
            digest_origin: HashMap::new(),
            prev_blocks: HashMap::new(),
        })
    }

    /// Borrow the underlying content-addressed object store.
    pub fn store(&self) -> &ChunkStore {
        &self.store
    }

    /// Borrow stored step manifests.
    pub fn manifests(&self) -> &BTreeMap<u64, StepCheckpoint> {
        &self.manifests
    }

    /// Borrow the mutable store in unit tests for tamper detection tests.
    #[cfg(test)]
    pub(crate) fn store_mut(&mut self) -> &mut ChunkStore {
        &mut self.store
    }

    /// Commit one distributed checkpoint step across `workers`.
    ///
    /// Returns the step manifest containing per-worker tensor encodings and measured
    /// redundancy / byte-reduction metrics.
    pub fn commit_step(
        &mut self,
        step: u64,
        workers: &[WorkerCheckpointInput<'_>],
    ) -> Result<StepCheckpoint> {
        ensure!(
            !self.manifests.contains_key(&step),
            "checkpoint step {step} already exists"
        );
        let t0 = Instant::now();

        let mut total_raw_bytes = 0u64;
        let mut new_objects_written = 0usize;
        let mut emitted_payload_bytes = 0u64;
        let mut zero_saved_bytes = 0u64;
        let mut intra_step_dedup_bytes = 0u64;
        let mut inter_step_dedup_bytes = 0u64;
        let mut delta_raw_bytes = 0u64;
        let mut delta_payload_bytes = 0u64;
        let mut delta_saved_bytes = 0u64;
        let mut full_chunk_bytes = 0u64;
        let mut max_chain_depth_seen = 0usize;

        // Tracks chunks first introduced during this step -> (originating worker_id, chain_depth).
        let mut step_seen: HashMap<ShardDigest, (String, usize)> = HashMap::new();
        let mut worker_manifests: BTreeMap<String, WorkerManifest> = BTreeMap::new();
        let mut pending_prev_updates: Vec<(BlockLocationKey, PrevBlockState)> = Vec::new();
        let mut pending_origin_updates: Vec<(ShardDigest, DigestOrigin)> = Vec::new();

        for worker in workers {
            ensure!(
                !worker_manifests.contains_key(worker.worker_id),
                "duplicate worker_id {} in step {step}",
                worker.worker_id
            );
            let mut tensor_records = Vec::with_capacity(worker.tensors.len());

            for tensor in &worker.tensors {
                let raw_bytes = tensor.bytes.len();
                total_raw_bytes += raw_bytes as u64;

                // Pre-hash chunks in parallel for multi-page tensors.
                let raw_blocks: Vec<&[u8]> = tensor.bytes.chunks(self.config.chunk_bytes).collect();
                let block_meta: Vec<(bool, ShardDigest)> = if raw_blocks.len() > 4 {
                    raw_blocks
                        .par_iter()
                        .map(|b| {
                            let zero = is_all_zero(b);
                            let digest = ShardDigest::of_bytes(b);
                            (zero, digest)
                        })
                        .collect()
                } else {
                    raw_blocks
                        .iter()
                        .map(|b| (is_all_zero(b), ShardDigest::of_bytes(b)))
                        .collect()
                };

                let elem_width = tensor.dtype.element_bytes();
                let mut block_refs = Vec::with_capacity(raw_blocks.len());

                for (block_idx, (&block, &(is_zero, expected_digest))) in
                    raw_blocks.iter().zip(block_meta.iter()).enumerate()
                {
                    let len = block.len() as u64;
                    let loc = BlockLocationKey {
                        worker_id: worker.worker_id.to_string(),
                        tensor_name: tensor.name.to_string(),
                        role: tensor.role,
                        block_idx,
                    };

                    let (encoding, new_depth) = if is_zero {
                        zero_saved_bytes += len;
                        (BlockEncoding::Zero, 0usize)
                    } else if self.config.dedup_across_steps
                        && self.digest_origin.contains_key(&expected_digest)
                    {
                        let origin = &self.digest_origin[&expected_digest];
                        inter_step_dedup_bytes += len;
                        (
                            BlockEncoding::Ref {
                                digest: expected_digest,
                                source: DedupSource::InterStep,
                                origin_step: origin.step,
                            },
                            origin.chain_depth,
                        )
                    } else if self.config.dedup_across_workers
                        && step_seen.contains_key(&expected_digest)
                    {
                        let (first_worker, first_depth) = &step_seen[&expected_digest];
                        let source = if first_worker == worker.worker_id {
                            DedupSource::IntraStepSameWorker
                        } else {
                            DedupSource::IntraStepCrossWorker
                        };
                        intra_step_dedup_bytes += len;
                        (
                            BlockEncoding::Ref {
                                digest: expected_digest,
                                source,
                                origin_step: step,
                            },
                            *first_depth,
                        )
                    } else {
                        // Candidate adaptive delta against prior step of the same shard block.
                        let maybe_delta = self
                            .prev_blocks
                            .get(&loc)
                            .filter(|prev| {
                                prev.chain_depth < self.config.max_delta_chain
                                    && prev.bytes.len() == block.len()
                            })
                            .and_then(|prev| {
                                select_best_delta(
                                    block,
                                    &prev.bytes,
                                    elem_width,
                                    self.config.enable_byte_lane_xor,
                                    self.config.delta_ratio_threshold,
                                )
                                .map(|(kind, delta_bytes)| (prev, kind, delta_bytes))
                            });

                        if let Some((prev, kind, delta_bytes)) = maybe_delta {
                            let delta_len = delta_bytes.len() as u64;
                            let delta_digest = ShardDigest::of_bytes(&delta_bytes);
                            let next_depth = prev.chain_depth + 1;
                            max_chain_depth_seen = max_chain_depth_seen.max(next_depth);

                            if self.store.put_if_absent(delta_digest, delta_bytes) {
                                new_objects_written += 1;
                                emitted_payload_bytes += delta_len;
                            }
                            delta_raw_bytes += len;
                            delta_payload_bytes += delta_len;
                            delta_saved_bytes += len.saturating_sub(delta_len);

                            let enc = match kind {
                                DeltaKind::XorSparse => BlockEncoding::XorSparseDelta {
                                    base_step: prev.step,
                                    base_digest: prev.raw_digest,
                                    delta_digest,
                                    chain_depth: next_depth,
                                },
                                DeltaKind::BytePlaneXor => BlockEncoding::BytePlaneXorDelta {
                                    base_step: prev.step,
                                    base_digest: prev.raw_digest,
                                    delta_digest,
                                    elem_width,
                                    chain_depth: next_depth,
                                },
                            };
                            step_seen
                                .entry(expected_digest)
                                .or_insert_with(|| (worker.worker_id.to_string(), next_depth));
                            pending_origin_updates.push((
                                expected_digest,
                                DigestOrigin {
                                    step,
                                    worker_id: worker.worker_id.to_string(),
                                    tensor_name: tensor.name.to_string(),
                                    role: tensor.role,
                                    block_idx,
                                    chain_depth: next_depth,
                                },
                            ));
                            (enc, next_depth)
                        } else {
                            if self.store.put_if_absent(expected_digest, block.to_vec()) {
                                new_objects_written += 1;
                                emitted_payload_bytes += len;
                            }
                            full_chunk_bytes += len;
                            step_seen
                                .entry(expected_digest)
                                .or_insert_with(|| (worker.worker_id.to_string(), 0));
                            pending_origin_updates.push((
                                expected_digest,
                                DigestOrigin {
                                    step,
                                    worker_id: worker.worker_id.to_string(),
                                    tensor_name: tensor.name.to_string(),
                                    role: tensor.role,
                                    block_idx,
                                    chain_depth: 0,
                                },
                            ));
                            (
                                BlockEncoding::Full {
                                    digest: expected_digest,
                                },
                                0usize,
                            )
                        }
                    };

                    pending_prev_updates.push((
                        loc,
                        PrevBlockState {
                            step,
                            raw_digest: expected_digest,
                            chain_depth: new_depth,
                            bytes: block.to_vec(),
                        },
                    ));

                    block_refs.push(BlockRef {
                        index: block_idx,
                        raw_len: block.len(),
                        expected_digest,
                        encoding,
                    });
                }

                tensor_records.push(TensorRecord {
                    name: tensor.name.to_string(),
                    role: tensor.role,
                    dtype: tensor.dtype,
                    shape: tensor.shape.to_vec(),
                    raw_bytes,
                    blocks: block_refs,
                });
            }

            worker_manifests.insert(
                worker.worker_id.to_string(),
                WorkerManifest {
                    worker_id: worker.worker_id.to_string(),
                    tensors: tensor_records,
                },
            );
        }

        for (k, v) in pending_prev_updates {
            self.prev_blocks.insert(k, v);
        }
        for (d, loc) in pending_origin_updates {
            self.digest_origin.entry(d).or_insert(loc);
        }

        let elapsed_secs = t0.elapsed().as_secs_f64().max(1e-9);
        let encode_mib_per_s = (total_raw_bytes as f64 / (1024.0 * 1024.0)) / elapsed_secs;
        let payload_ratio = if total_raw_bytes == 0 {
            0.0
        } else {
            emitted_payload_bytes as f64 / total_raw_bytes as f64
        };
        let byte_reduction_pct = if total_raw_bytes == 0 {
            0.0
        } else {
            100.0 * (1.0 - payload_ratio)
        };

        let mut ckpt = StepCheckpoint {
            step,
            workers: worker_manifests,
            metrics: CheckpointMetrics {
                step,
                total_raw_bytes,
                new_objects_written,
                emitted_payload_bytes,
                manifest_bytes: 0,
                total_checkpoint_bytes: emitted_payload_bytes,
                zero_saved_bytes,
                intra_step_dedup_bytes,
                inter_step_dedup_bytes,
                delta_raw_bytes,
                delta_payload_bytes,
                delta_saved_bytes,
                full_chunk_bytes,
                max_chain_depth_seen,
                payload_ratio,
                byte_reduction_pct,
                encode_mib_per_s,
            },
        };
        let manifest_bytes = ckpt.to_json_bytes()?.len() as u64;
        ckpt.metrics.manifest_bytes = manifest_bytes;
        ckpt.metrics.total_checkpoint_bytes = emitted_payload_bytes + manifest_bytes;
        self.manifests.insert(step, ckpt.clone());
        Ok(ckpt)
    }

    /// Reconstruct every tensor/optimizer shard for `worker_id` at checkpoint `step`,
    /// verifying SHA-256 digests on every base object and reconstructed block.
    pub fn reconstruct_worker(
        &self,
        step: u64,
        worker_id: &str,
    ) -> Result<Vec<ReconstructedTensor>> {
        let ckpt = self
            .manifests
            .get(&step)
            .with_context(|| format!("missing checkpoint step {step}"))?;
        let worker = ckpt
            .workers
            .get(worker_id)
            .with_context(|| format!("worker {worker_id} absent from step {step}"))?;

        let mut memo: HashMap<ShardDigest, Vec<u8>> = HashMap::new();
        let mut active_stack: HashSet<(u64, BlockLocationKey)> = HashSet::new();
        let mut restored = Vec::with_capacity(worker.tensors.len());

        for tensor in &worker.tensors {
            let mut bytes = Vec::with_capacity(tensor.raw_bytes);
            for block in &tensor.blocks {
                let loc = BlockLocationKey {
                    worker_id: worker_id.to_string(),
                    tensor_name: tensor.name.clone(),
                    role: tensor.role,
                    block_idx: block.index,
                };
                let decoded =
                    self.decode_block_ref(step, &loc, block, &mut memo, &mut active_stack)?;
                bytes.extend_from_slice(&decoded);
            }
            ensure!(
                bytes.len() == tensor.raw_bytes,
                "tensor {} length mismatch: decoded {} != expected {}",
                tensor.name,
                bytes.len(),
                tensor.raw_bytes
            );
            restored.push(ReconstructedTensor {
                name: tensor.name.clone(),
                role: tensor.role,
                dtype: tensor.dtype,
                shape: tensor.shape.clone(),
                bytes,
            });
        }

        Ok(restored)
    }

    fn resolve_digest_bytes(
        &self,
        digest: ShardDigest,
        origin_step: u64,
        memo: &mut HashMap<ShardDigest, Vec<u8>>,
        active_stack: &mut HashSet<(u64, BlockLocationKey)>,
    ) -> Result<Vec<u8>> {
        if let Some(cached) = memo.get(&digest) {
            return Ok(cached.clone());
        }
        if let Some(raw) = self.store.get(&digest) {
            ensure!(
                ShardDigest::of_bytes(raw) == digest,
                "corrupted object in ChunkStore for digest {digest}"
            );
            let out = raw.to_vec();
            memo.insert(digest, out.clone());
            return Ok(out);
        }
        let loc = self.digest_origin.get(&digest).with_context(|| {
            format!("missing base object {digest} referenced at step {origin_step}")
        })?;
        let ckpt = self
            .manifests
            .get(&loc.step)
            .with_context(|| format!("missing origin manifest step {} for {digest}", loc.step))?;
        let wm = ckpt
            .workers
            .get(&loc.worker_id)
            .with_context(|| format!("missing worker {} in step {}", loc.worker_id, loc.step))?;
        let tr = wm
            .tensors
            .iter()
            .find(|t| t.name == loc.tensor_name && t.role == loc.role)
            .with_context(|| format!("missing tensor {} in step {}", loc.tensor_name, loc.step))?;
        let br = tr.blocks.get(loc.block_idx).with_context(|| {
            format!(
                "missing block {} in {} step {}",
                loc.block_idx, loc.tensor_name, loc.step
            )
        })?;
        let loc_key = BlockLocationKey {
            worker_id: loc.worker_id.clone(),
            tensor_name: loc.tensor_name.clone(),
            role: loc.role,
            block_idx: loc.block_idx,
        };
        self.decode_block_ref(loc.step, &loc_key, br, memo, active_stack)
    }

    fn decode_block_ref(
        &self,
        step: u64,
        loc: &BlockLocationKey,
        block: &BlockRef,
        memo: &mut HashMap<ShardDigest, Vec<u8>>,
        active_stack: &mut HashSet<(u64, BlockLocationKey)>,
    ) -> Result<Vec<u8>> {
        if let Some(cached) = memo.get(&block.expected_digest) {
            return Ok(cached.clone());
        }
        let stack_key = (step, loc.clone());
        ensure!(
            active_stack.insert(stack_key.clone()),
            "cyclic AdaCheck delta reference detected at step {step} digest {}",
            block.expected_digest
        );

        let decoded_res: Result<Vec<u8>> = (|| match &block.encoding {
            BlockEncoding::Zero => Ok(vec![0u8; block.raw_len]),
            BlockEncoding::Ref {
                digest,
                origin_step,
                ..
            } => self.resolve_digest_bytes(*digest, *origin_step, memo, active_stack),
            BlockEncoding::Full { digest } => {
                let raw = self
                    .store
                    .get(digest)
                    .with_context(|| format!("missing full chunk {digest} at step {step}"))?;
                Ok(raw.to_vec())
            }
            BlockEncoding::XorSparseDelta {
                base_step,
                base_digest,
                delta_digest,
                ..
            } => {
                let base =
                    self.resolve_digest_bytes(*base_digest, *base_step, memo, active_stack)?;
                let ops = self.store.get(delta_digest).with_context(|| {
                    format!("missing delta chunk {delta_digest} at step {step}")
                })?;
                ensure!(
                    ShardDigest::of_bytes(ops) == *delta_digest,
                    "corrupted delta chunk {delta_digest} at step {step}"
                );
                decode_xor_sparse(&base, ops)
            }
            BlockEncoding::BytePlaneXorDelta {
                base_step,
                base_digest,
                delta_digest,
                elem_width,
                ..
            } => {
                let base =
                    self.resolve_digest_bytes(*base_digest, *base_step, memo, active_stack)?;
                let ops = self.store.get(delta_digest).with_context(|| {
                    format!("missing delta chunk {delta_digest} at step {step}")
                })?;
                ensure!(
                    ShardDigest::of_bytes(ops) == *delta_digest,
                    "corrupted delta chunk {delta_digest} at step {step}"
                );
                decode_byte_plane_xor(&base, ops, *elem_width)
            }
        })();

        active_stack.remove(&stack_key);
        let decoded = decoded_res?;
        ensure!(
            decoded.len() == block.raw_len,
            "block {} decoded length {} != expected {} at step {step}",
            block.index,
            decoded.len(),
            block.raw_len
        );
        let actual_digest = ShardDigest::of_bytes(&decoded);
        ensure!(
            actual_digest == block.expected_digest,
            "checksum mismatch restoring block {} at step {step}: got {actual_digest}, expected {}",
            block.index,
            block.expected_digest
        );
        memo.insert(block.expected_digest, decoded.clone());
        Ok(decoded)
    }
}

/// Convert `hanzo-train`'s DiLoCo coordinator state (`Params`, `theta`, and `Inner` worker
/// optimizer buffers) into [`WorkerCheckpointInput`] shards for [`AdaCheckRepository`].
pub fn commit_cluster_checkpoint(
    repo: &mut AdaCheckRepository,
    round: u64,
    params: &Params,
    theta: &[f32],
    inner: &BTreeMap<String, Inner>,
) -> Result<StepCheckpoint> {
    ensure!(
        theta.len() == params.len,
        "theta length {} != params.len {}",
        theta.len(),
        params.len
    );
    let param_bytes: Vec<Vec<u8>> = (0..params.names.len())
        .map(|i| pack_f32_le(&theta[params.span(i)]))
        .collect();

    let mut owned_inner: Vec<(String, Vec<u8>)> = Vec::with_capacity(inner.len());
    for (worker_name, state) in inner {
        ensure!(
            state.data.len() == 12 * params.len,
            "worker {worker_name} optimizer state length {} != {}",
            state.data.len(),
            12 * params.len
        );
        owned_inner.push((worker_name.clone(), state.data.as_ref().clone()));
    }

    let mut inputs = Vec::with_capacity(owned_inner.len().max(1));
    if owned_inner.is_empty() {
        let tensors = params
            .names
            .iter()
            .zip(params.shapes.iter().zip(&param_bytes))
            .map(|(name, (shape, bytes))| TensorInput {
                name,
                role: ShardRole::Weight,
                dtype: TensorDType::F32,
                shape,
                bytes,
            })
            .collect();
        inputs.push(WorkerCheckpointInput {
            worker_id: "coordinator",
            tensors,
        });
    } else {
        let span_bytes = 4 * params.len;
        for (worker_name, raw) in &owned_inner {
            let mut tensors: Vec<TensorInput<'_>> = params
                .names
                .iter()
                .zip(params.shapes.iter().zip(&param_bytes))
                .map(|(name, (shape, bytes))| TensorInput {
                    name,
                    role: ShardRole::Weight,
                    dtype: TensorDType::F32,
                    shape,
                    bytes,
                })
                .collect();
            tensors.push(TensorInput {
                name: "adam.m",
                role: ShardRole::OptimizerFirstMoment,
                dtype: TensorDType::F32,
                shape: std::slice::from_ref(&params.len),
                bytes: &raw[0..span_bytes],
            });
            tensors.push(TensorInput {
                name: "adam.v",
                role: ShardRole::OptimizerSecondMoment,
                dtype: TensorDType::F32,
                shape: std::slice::from_ref(&params.len),
                bytes: &raw[span_bytes..2 * span_bytes],
            });
            tensors.push(TensorInput {
                name: "adam.carry",
                role: ShardRole::OptimizerCarry,
                dtype: TensorDType::F32,
                shape: std::slice::from_ref(&params.len),
                bytes: &raw[2 * span_bytes..3 * span_bytes],
            });
            inputs.push(WorkerCheckpointInput {
                worker_id: worker_name,
                tensors,
            });
        }
    }

    repo.commit_step(round, &inputs)
}

/// Pack `f32` values into exact little-endian IEEE-754 bit patterns.
pub fn pack_f32_le(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for &v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

#[inline]
fn is_all_zero(bytes: &[u8]) -> bool {
    bytes.iter().all(|&b| b == 0)
}

#[derive(Debug, Clone, Copy)]
enum DeltaKind {
    XorSparse,
    BytePlaneXor,
}

fn select_best_delta(
    current: &[u8],
    base: &[u8],
    elem_width: usize,
    enable_byte_lane: bool,
    ratio_threshold: f64,
) -> Option<(DeltaKind, Vec<u8>)> {
    let sparse = encode_xor_sparse(current, base);
    let mut best = (DeltaKind::XorSparse, sparse);

    if enable_byte_lane && elem_width > 1 && current.len().is_multiple_of(elem_width) {
        let plane = encode_byte_plane_xor(current, base, elem_width);
        if plane.len() < best.1.len() {
            best = (DeltaKind::BytePlaneXor, plane);
        }
    }

    let max_allowed = (current.len() as f64 * ratio_threshold) as usize;
    if best.1.len() <= max_allowed {
        Some(best)
    } else {
        None
    }
}

fn encode_sparse_xor_buffer(xor_buf: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let n = xor_buf.len();
    let mut i = 0usize;

    while i < n {
        let skip_start = i;
        while i < n && xor_buf[i] == 0 {
            i += 1;
        }
        if i == n {
            break;
        }
        let skip = (i - skip_start) as u32;
        let lit_start = i;
        let mut lit_end = i + 1;
        i += 1;

        while i < n {
            if xor_buf[i] != 0 {
                i += 1;
                lit_end = i;
            } else {
                let z_start = i;
                while i < n && xor_buf[i] == 0 && (i - z_start) < MIN_ZERO_RUN_SPLIT {
                    i += 1;
                }
                if i == n || (i - z_start) >= MIN_ZERO_RUN_SPLIT {
                    break;
                }
                // Short zero gap bridged inside the literal run.
                lit_end = i;
            }
        }

        let lit_len = (lit_end - lit_start) as u32;
        out.extend_from_slice(&skip.to_le_bytes());
        out.extend_from_slice(&lit_len.to_le_bytes());
        out.extend_from_slice(&xor_buf[lit_start..lit_end]);
        i = lit_end;
    }

    out
}

fn encode_xor_sparse(current: &[u8], base: &[u8]) -> Vec<u8> {
    debug_assert_eq!(current.len(), base.len());
    let xor_buf: Vec<u8> = current.iter().zip(base).map(|(&a, &b)| a ^ b).collect();
    encode_sparse_xor_buffer(&xor_buf)
}

fn decode_sparse_runs_into(target: &mut [u8], ops: &[u8]) -> Result<()> {
    let mut cursor = 0usize;
    let mut pos = 0usize;
    while cursor < ops.len() {
        ensure!(cursor + 8 <= ops.len(), "truncated sparse delta header");
        let skip = u32::from_le_bytes([
            ops[cursor],
            ops[cursor + 1],
            ops[cursor + 2],
            ops[cursor + 3],
        ]) as usize;
        let lit_len = u32::from_le_bytes([
            ops[cursor + 4],
            ops[cursor + 5],
            ops[cursor + 6],
            ops[cursor + 7],
        ]) as usize;
        cursor += 8;
        ensure!(
            cursor + lit_len <= ops.len(),
            "truncated sparse delta literal"
        );
        pos = pos
            .checked_add(skip)
            .context("sparse delta offset overflow")?;
        ensure!(
            pos + lit_len <= target.len(),
            "sparse delta writes past block boundary"
        );
        for (dst, &d) in target[pos..pos + lit_len]
            .iter_mut()
            .zip(&ops[cursor..cursor + lit_len])
        {
            *dst ^= d;
        }
        pos += lit_len;
        cursor += lit_len;
    }
    Ok(())
}

fn decode_xor_sparse(base: &[u8], ops: &[u8]) -> Result<Vec<u8>> {
    let mut out = base.to_vec();
    decode_sparse_runs_into(&mut out, ops)?;
    Ok(out)
}

fn encode_byte_plane_xor(current: &[u8], base: &[u8], elem_width: usize) -> Vec<u8> {
    debug_assert_eq!(current.len(), base.len());
    debug_assert!(elem_width > 1 && current.len().is_multiple_of(elem_width));
    let n_elems = current.len() / elem_width;
    let mut transposed = vec![0u8; current.len()];
    for (e, (cur_el, base_el)) in current
        .chunks_exact(elem_width)
        .zip(base.chunks_exact(elem_width))
        .enumerate()
    {
        for (lane, (&c, &b)) in cur_el.iter().zip(base_el).enumerate() {
            transposed[lane * n_elems + e] = c ^ b;
        }
    }
    encode_sparse_xor_buffer(&transposed)
}

fn decode_byte_plane_xor(base: &[u8], ops: &[u8], elem_width: usize) -> Result<Vec<u8>> {
    ensure!(
        elem_width > 1 && base.len().is_multiple_of(elem_width),
        "invalid byte-plane elem_width {elem_width} for {} bytes",
        base.len()
    );
    let n_elems = base.len() / elem_width;
    let mut transposed = vec![0u8; base.len()];
    decode_sparse_runs_into(&mut transposed, ops)?;

    let mut out = base.to_vec();
    for (e, out_el) in out.chunks_exact_mut(elem_width).enumerate() {
        for (lane, dst) in out_el.iter_mut().enumerate() {
            *dst ^= transposed[lane * n_elems + e];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn assert_bit_exact(restored: &[ReconstructedTensor], original: &[TensorInput<'_>]) {
        assert_eq!(restored.len(), original.len());
        for (r, o) in restored.iter().zip(original.iter()) {
            assert_eq!(r.name, o.name);
            assert_eq!(r.role, o.role);
            assert_eq!(r.dtype, o.dtype);
            assert_eq!(r.shape, o.shape);
            assert_eq!(
                r.bytes, o.bytes,
                "tensor {} failed bit-exact reconstruction",
                o.name
            );
        }
    }

    #[test]
    fn cross_worker_redundancy_and_bit_exact_special_ieee754_bits() -> Result<()> {
        // 4 DDP/ZeRO workers sharing identical model weights & frozen embeddings, plus
        // distinct per-worker optimizer state shards and IEEE-754 edge bit patterns
        // (-0.0, subnormals, quiet NaNs).
        let cfg = AdaCheckConfig {
            chunk_bytes: 1024,
            ..AdaCheckConfig::default()
        };
        let mut detector = RedundancyDetector::new(cfg.chunk_bytes)?;
        let mut repo = AdaCheckRepository::new(cfg)?;

        let mut shared_weights_f32: Vec<f32> = (0..4096)
            .map(|i| ((i as f32) * 0.0137).sin() * 0.25)
            .collect();
        shared_weights_f32[0] = 0.0;
        shared_weights_f32[1] = -0.0;
        shared_weights_f32[2] = f32::from_bits(0x0000_0007); // subnormal
        shared_weights_f32[3] = f32::from_bits(0x7fc0_1234); // payload-tagged NaN
        let shared_weights = pack_f32_le(&shared_weights_f32);
        let frozen_embed: Vec<u8> = (0..16_384).map(|i| ((i * 31) % 251) as u8).collect();
        let zero_carry = vec![0u8; 8192];

        let worker_m: Vec<Vec<u8>> = (0..4)
            .map(|w| {
                let vals: Vec<f32> = (0..1024)
                    .map(|i| ((i + w * 17) as f32) * 1e-4 + if i == 0 { -0.0 } else { 0.0 })
                    .collect();
                pack_f32_le(&vals)
            })
            .collect();

        let workers: Vec<WorkerCheckpointInput<'_>> = (0..4)
            .map(|w| WorkerCheckpointInput {
                worker_id: match w {
                    0 => "rank-0",
                    1 => "rank-1",
                    2 => "rank-2",
                    _ => "rank-3",
                },
                tensors: vec![
                    TensorInput {
                        name: "layers.0.attn.q_proj",
                        role: ShardRole::Weight,
                        dtype: TensorDType::F32,
                        shape: &[64, 64],
                        bytes: &shared_weights,
                    },
                    TensorInput {
                        name: "embed_tokens.weight",
                        role: ShardRole::FrozenWeight,
                        dtype: TensorDType::BF16,
                        shape: &[128, 64],
                        bytes: &frozen_embed,
                    },
                    TensorInput {
                        name: "adam.m",
                        role: ShardRole::OptimizerFirstMoment,
                        dtype: TensorDType::F32,
                        shape: &[1024],
                        bytes: &worker_m[w],
                    },
                    TensorInput {
                        name: "adam.carry",
                        role: ShardRole::OptimizerCarry,
                        dtype: TensorDType::F32,
                        shape: &[2048],
                        bytes: &zero_carry,
                    },
                ],
            })
            .collect();

        let detected = detector.record_step(&workers);
        let ckpt = repo.commit_step(0, &workers)?;

        println!(
            "[adacheck step0] raw={}B emitted={}B intra_dedup={}B zero_saved={}B reduction={:.2}% det_redundancy={:.2}%",
            ckpt.metrics.total_raw_bytes,
            ckpt.metrics.emitted_payload_bytes,
            ckpt.metrics.intra_step_dedup_bytes,
            ckpt.metrics.zero_saved_bytes,
            ckpt.metrics.byte_reduction_pct,
            detected.raw_redundancy_pct
        );

        assert_eq!(
            ckpt.metrics.total_raw_bytes,
            4 * (16384 + 16384 + 4096 + 8192)
        );
        // Ranks 1..=3 share rank-0's weight + frozen_embed bytes (3 * 32768 = 98304 B).
        assert_eq!(ckpt.metrics.intra_step_dedup_bytes, 3 * (16384 + 16384));
        assert_eq!(ckpt.metrics.zero_saved_bytes, 4 * 8192);
        assert!(ckpt.metrics.byte_reduction_pct > 70.0);
        assert_eq!(
            repo.store().stored_payload_bytes(),
            ckpt.metrics.emitted_payload_bytes
        );
        let pub_objects = repo.store().objects_for_publication();
        assert_eq!(pub_objects.len(), repo.store().object_count());
        assert!(pub_objects
            .iter()
            .all(|(key, _)| key.starts_with("adacheck/chunks/") && key.ends_with(".bin")));
        assert!(repo.store().total_puts() >= pub_objects.len() as u64);

        for w in &workers {
            let restored = repo.reconstruct_worker(0, w.worker_id)?;
            assert_bit_exact(&restored, &w.tensors);
            // Verify IEEE-754 bit patterns of special values explicitly.
            let w0 = restored[0].as_f32_le()?;
            assert_eq!(w0[0].to_bits(), 0.0f32.to_bits());
            assert_eq!(w0[1].to_bits(), (-0.0f32).to_bits());
            assert_eq!(w0[2].to_bits(), 0x0000_0007);
            assert_eq!(w0[3].to_bits(), 0x7fc0_1234);
        }
        Ok(())
    }

    #[test]
    fn sequential_steps_measure_temporal_dedup_and_delta_byte_reduction() -> Result<()> {
        let cfg = AdaCheckConfig {
            chunk_bytes: 4096,
            max_delta_chain: 8,
            delta_ratio_threshold: 0.85,
            enable_byte_lane_xor: true,
            ..AdaCheckConfig::default()
        };
        let mut repo = AdaCheckRepository::new(cfg)?;

        // Simulate 4 workers over 5 sequential steps:
        // - frozen weights unchanged across all steps and all workers
        // - sparse MoE / embedding table: only 2% of elements change per step
        // - dense small-step F32 weights: tiny gradient steps of ~1e-5
        // - dense F32 optimizer moments: small exponential moving average updates
        let frozen_buf: Vec<u8> = (0..32_768).map(|i| ((i * 19 + 3) % 251) as u8).collect();
        let mut moe_table: Vec<f32> = (0..8192).map(|i| (i as f32 * 0.01).cos()).collect();
        let mut dense_w: Vec<f32> = (0..4096).map(|i| 1.0 + (i as f32 * 0.005).sin()).collect();
        let mut worker_v: Vec<Vec<f32>> = (0..4)
            .map(|r| {
                (0..2048)
                    .map(|i| 0.01 + ((i + r * 31) as f32) * 1e-4)
                    .collect()
            })
            .collect();

        let mut total_raw = 0u64;
        let mut total_emitted = 0u64;
        type StepWorkerBuffers = (Vec<u8>, Vec<u8>, Vec<Vec<u8>>);
        let mut history_inputs: Vec<Vec<StepWorkerBuffers>> = Vec::new();

        for step in 0..5u64 {
            if step > 0 {
                // Sparse update: touch 64 rows out of 8192 (~0.78% of MoE table).
                for idx in (step as usize * 13..8192).step_by(128) {
                    moe_table[idx] += 0.0025 * step as f32;
                }
                // Small dense update on weights (~low mantissa bits change, sign/exponent stable).
                for (i, w) in dense_w.iter_mut().enumerate() {
                    *w += 1e-6 * ((i % 7) as f32 - 3.0);
                }
                // Small optimizer second-moment updates per worker.
                for (r, vbuf) in worker_v.iter_mut().enumerate() {
                    for (i, v) in vbuf.iter_mut().enumerate() {
                        if (i + r + step as usize).is_multiple_of(8) {
                            *v += 1e-7 * (step as f32);
                        }
                    }
                }
            }

            let moe_bytes = pack_f32_le(&moe_table);
            let dense_bytes = pack_f32_le(&dense_w);
            let v_bytes: Vec<Vec<u8>> = worker_v.iter().map(|v| pack_f32_le(v)).collect();

            let workers: Vec<WorkerCheckpointInput<'_>> = (0..4)
                .map(|r| WorkerCheckpointInput {
                    worker_id: match r {
                        0 => "w0",
                        1 => "w1",
                        2 => "w2",
                        _ => "w3",
                    },
                    tensors: vec![
                        TensorInput {
                            name: "frozen.embed",
                            role: ShardRole::FrozenWeight,
                            dtype: TensorDType::BF16,
                            shape: &[256, 64],
                            bytes: &frozen_buf,
                        },
                        TensorInput {
                            name: "moe.experts",
                            role: ShardRole::Weight,
                            dtype: TensorDType::F32,
                            shape: &[128, 64],
                            bytes: &moe_bytes,
                        },
                        TensorInput {
                            name: "attn.o_proj",
                            role: ShardRole::Weight,
                            dtype: TensorDType::F32,
                            shape: &[64, 64],
                            bytes: &dense_bytes,
                        },
                        TensorInput {
                            name: "adam.v",
                            role: ShardRole::OptimizerSecondMoment,
                            dtype: TensorDType::F32,
                            shape: &[2048],
                            bytes: &v_bytes[r],
                        },
                    ],
                })
                .collect();

            let ckpt = repo.commit_step(step, &workers)?;
            total_raw += ckpt.metrics.total_raw_bytes;
            total_emitted += ckpt.metrics.emitted_payload_bytes;

            println!(
                "[adacheck seq step={step}] raw={} emitted={} inter_dedup={} intra_dedup={} delta_saved={} reduction={:.2}% throughput={:.1} MiB/s",
                ckpt.metrics.total_raw_bytes,
                ckpt.metrics.emitted_payload_bytes,
                ckpt.metrics.inter_step_dedup_bytes,
                ckpt.metrics.intra_step_dedup_bytes,
                ckpt.metrics.delta_saved_bytes,
                ckpt.metrics.byte_reduction_pct,
                ckpt.metrics.encode_mib_per_s
            );

            if step > 0 {
                assert!(ckpt.metrics.inter_step_dedup_bytes > 0);
                assert!(ckpt.metrics.delta_saved_bytes > 0);
                assert!(
                    ckpt.metrics.byte_reduction_pct >= 85.0,
                    "expected >=85% reduction on step {step}, got {:.2}%",
                    ckpt.metrics.byte_reduction_pct
                );
            }

            history_inputs.push(vec![(moe_bytes, dense_bytes, v_bytes)]);
        }

        let overall_reduction = 100.0 * (1.0 - (total_emitted as f64 / total_raw as f64));
        println!(
            "[adacheck 5-step total] raw={total_raw}B stored={}B reduction={overall_reduction:.2}% objects={}",
            repo.store().stored_payload_bytes(),
            repo.store().object_count()
        );
        assert_eq!(repo.store().stored_payload_bytes(), total_emitted);
        assert!(overall_reduction > 85.0);

        // Reconstruct EVERY worker at EVERY step and verify exact byte equality.
        let restore_t0 = Instant::now();
        for (step_idx, step_data) in history_inputs.iter().enumerate() {
            let (moe_bytes, dense_bytes, v_bytes) = &step_data[0];
            for (r, wid) in ["w0", "w1", "w2", "w3"].iter().enumerate() {
                let restored = repo.reconstruct_worker(step_idx as u64, wid)?;
                assert_eq!(restored[0].bytes, frozen_buf);
                assert_eq!(restored[1].bytes, *moe_bytes);
                assert_eq!(restored[2].bytes, *dense_bytes);
                assert_eq!(restored[3].bytes, v_bytes[r]);
            }
        }
        let restore_mib_per_s =
            (total_raw as f64 / (1024.0 * 1024.0)) / restore_t0.elapsed().as_secs_f64().max(1e-9);
        println!("[adacheck 5-step restore] bit-exact=true restore_throughput={restore_mib_per_s:.1} MiB/s");
        Ok(())
    }

    #[test]
    fn delta_chain_depth_is_bounded_and_resets_to_keyframe() -> Result<()> {
        let cfg = AdaCheckConfig {
            chunk_bytes: 2048,
            max_delta_chain: 3,
            delta_ratio_threshold: 0.90,
            enable_byte_lane_xor: true,
            ..AdaCheckConfig::default()
        };
        let mut repo = AdaCheckRepository::new(cfg)?;
        let mut data: Vec<f32> = (0..1024).map(|i| 2.0 + (i as f32) * 0.01).collect();
        let mut snapshots = Vec::new();

        for step in 0..9u64 {
            if step > 0 {
                // Sparse deterministic mutation every step so delta is small and eligible.
                data[(step as usize * 37) % 1024] += 0.5;
            }
            let raw = pack_f32_le(&data);
            let input = [WorkerCheckpointInput {
                worker_id: "w0",
                tensors: vec![TensorInput {
                    name: "p",
                    role: ShardRole::Weight,
                    dtype: TensorDType::F32,
                    shape: &[1024],
                    bytes: &raw,
                }],
            }];
            let ckpt = repo.commit_step(step, &input)?;
            assert!(
                ckpt.metrics.max_chain_depth_seen <= 3,
                "step {step} exceeded max_delta_chain: {}",
                ckpt.metrics.max_chain_depth_seen
            );
            snapshots.push(raw);
        }

        // Step 4 and Step 8 must have reset the modified block to a fresh keyframe (`Full`).
        let ckpt4 = &repo.manifests()[&4].workers["w0"].tensors[0].blocks;
        assert!(
            ckpt4
                .iter()
                .any(|b| matches!(b.encoding, BlockEncoding::Full { .. })),
            "expected keyframe reset after max_delta_chain=3"
        );

        for (step, expected) in snapshots.iter().enumerate() {
            let restored = repo.reconstruct_worker(step as u64, "w0")?;
            assert_eq!(restored[0].bytes, *expected);
        }
        Ok(())
    }

    #[test]
    fn corrupted_chunk_store_object_is_detected_on_restore() -> Result<()> {
        let cfg = AdaCheckConfig {
            chunk_bytes: 512,
            ..AdaCheckConfig::default()
        };
        let mut repo = AdaCheckRepository::new(cfg)?;
        let raw: Vec<u8> = (0..1024).map(|i| ((i * 7 + 1) % 253) as u8).collect();
        let input = [WorkerCheckpointInput {
            worker_id: "w0",
            tensors: vec![TensorInput {
                name: "w",
                role: ShardRole::Weight,
                dtype: TensorDType::U8,
                shape: &[1024],
                bytes: &raw,
            }],
        }];
        let ckpt = repo.commit_step(0, &input)?;
        let first_digest = ckpt.workers["w0"].tensors[0].blocks[0].expected_digest;
        assert!(repo.store_mut().corrupt_object_for_test(&first_digest, 7));

        let err = repo
            .reconstruct_worker(0, "w0")
            .expect_err("bit flip in ChunkStore must fail SHA-256 verification");
        assert!(
            err.to_string().contains("checksum mismatch"),
            "unexpected error message: {err}"
        );
        Ok(())
    }

    #[test]
    fn cluster_params_and_inner_round_checkpoint_roundtrips_bit_exact() -> Result<()> {
        use hanzo_ml::{Device, Tensor, Var};
        use hanzo_nn::VarMap;

        let vars = VarMap::new();
        {
            let mut m = vars.data().lock().unwrap();
            let w = Tensor::from_vec(
                (0..256).map(|i| i as f32 * 0.1).collect(),
                (16, 16),
                &Device::Cpu,
            )?;
            let b = Tensor::from_vec(vec![0.5f32; 16], 16, &Device::Cpu)?;
            m.insert("linear.weight".to_string(), Var::from_tensor(&w)?);
            m.insert("linear.bias".to_string(), Var::from_tensor(&b)?);
        }
        let params = Params::of(&vars, |_| true);
        let mut theta = params.read(&vars)?;
        let mut inner = BTreeMap::new();
        for name in ["worker-a", "worker-b"] {
            let state: Vec<f32> = (0..3 * params.len)
                .map(|i| if i < params.len { 0.01 * i as f32 } else { 0.0 })
                .collect();
            inner.insert(
                name.to_string(),
                Inner {
                    t: 10,
                    data: Arc::new(crate::cluster::net::pack(&state)),
                },
            );
        }

        let mut repo = AdaCheckRepository::new(AdaCheckConfig {
            chunk_bytes: 512,
            ..AdaCheckConfig::default()
        })?;
        let ckpt0 = commit_cluster_checkpoint(&mut repo, 0, &params, &theta, &inner)?;
        // Step 1: tiny theta nudge + worker-a step; worker-b unchanged.
        theta[0] += 1e-4;
        let ckpt1 = commit_cluster_checkpoint(&mut repo, 1, &params, &theta, &inner)?;
        assert!(ckpt1.metrics.byte_reduction_pct > 75.0);
        assert!(ckpt0.metrics.total_checkpoint_bytes > ckpt1.metrics.total_checkpoint_bytes);

        let restored = repo.reconstruct_worker(1, "worker-a")?;
        let restored_bias = restored[0].as_f32_le()?;
        let restored_weight = restored[1].as_f32_le()?;
        assert_eq!(restored_bias.len() + restored_weight.len(), theta.len());
        let mut reassembled = Vec::with_capacity(theta.len());
        reassembled.extend_from_slice(&restored_bias);
        reassembled.extend_from_slice(&restored_weight);
        assert_eq!(
            reassembled.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            theta.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        Ok(())
    }
}
