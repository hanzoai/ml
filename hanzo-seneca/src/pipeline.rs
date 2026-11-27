//! Real 3-stage training data-pipeline execution engine:
//! `FetchEncoded -> Decode -> Transform -> GPU Batch Consume`.
//!
//! Executes genuine LZ/RLE compression and decompression with IEEE CRC32
//! verification, real multi-modal tensor/token decoding and augmentation/feature
//! extraction over real repository corpus shards (`hanzoai/ml`), concurrent
//! multi-job scheduling over a shared bounded [`MultiStageCache`], and strict
//! runtime verification of:
//! - Bounded cache memory (`used_bytes <= total_budget_bytes`)
//! - Stochastic-augmentation freshness (`100%` fresh across epochs)
//! - Cross-tenant isolation (`0` leaks across private datasets)

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::allocator::{
    cpu_contention_factor, job_cpu_utilization, AllocationPolicy, AllocatorConfig, JobTelemetry,
    SenecaAllocator,
};
use crate::cache::{CacheEntry, CacheKey, CacheStage, DatasetVisibility, MultiStageCache, ScopeId};
use crate::codec::{compress_lz_rle, content_hash_128, crc32, decompress_lz_rle, hash_u64, Rng64};

/// Real repository corpus store loaded deterministically from local workspace files.
#[derive(Debug, Clone)]
pub struct CorpusStore {
    files_loaded: usize,
    total_source_bytes: usize,
    source_digest: u128,
    raw_pool: Vec<u8>,
}

impl CorpusStore {
    /// Loads real public source/doc files (`.rs`, `.md`, `.toml`) deterministically
    /// from `root_dir` (e.g. the `hanzoai/ml` repository root).
    pub fn load_from_workspace(root_dir: &Path) -> Result<Self, String> {
        let mut paths = Vec::new();
        let candidate_dirs = [
            "hanzo-ml",
            "hanzo-nn",
            "hanzo-datasets",
            "hanzo-train",
            "hanzo-research",
            "hanzo-transformers",
            "hanzo-learn",
            "hanzo-active-inference",
        ];
        for dir in &candidate_dirs {
            let p = root_dir.join(dir);
            if p.is_dir() {
                collect_corpus_files(&p, &mut paths);
            }
        }
        // Fallback: if called with a direct directory containing source files.
        if paths.is_empty() && root_dir.is_dir() {
            collect_corpus_files(root_dir, &mut paths);
        }
        paths.sort();

        if paths.is_empty() {
            return Err(format!(
                "no corpus source files (.rs/.md/.toml) found under {}",
                root_dir.display()
            ));
        }

        let mut raw_pool = Vec::with_capacity(2 * 1024 * 1024);
        let mut files_loaded = 0usize;
        for path in &paths {
            if let Ok(bytes) = fs::read(path) {
                if !bytes.is_empty() {
                    raw_pool.extend_from_slice(&bytes);
                    raw_pool.push(b'\n');
                    files_loaded += 1;
                }
            }
            // Cap pool at ~8 MiB so benchmark startup stays sub-second.
            if raw_pool.len() >= 8 * 1024 * 1024 {
                break;
            }
        }

        if raw_pool.is_empty() {
            return Err("corpus files were all empty".to_string());
        }

        let source_digest = content_hash_128(b"hanzo-seneca-corpus-v1", &raw_pool);
        let total_source_bytes = raw_pool.len();
        Ok(Self {
            files_loaded,
            total_source_bytes,
            source_digest,
            raw_pool,
        })
    }

    /// Creates a deterministic in-memory corpus store (for isolated unit tests).
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        let raw_pool = if bytes.is_empty() {
            b"hanzo-seneca-default-test-corpus\n".repeat(256)
        } else {
            bytes
        };
        let source_digest = content_hash_128(b"hanzo-seneca-unit-corpus", &raw_pool);
        let total_source_bytes = raw_pool.len();
        Self {
            files_loaded: 1,
            total_source_bytes,
            source_digest,
            raw_pool,
        }
    }

    #[inline]
    pub fn files_loaded(&self) -> usize {
        self.files_loaded
    }

    #[inline]
    pub fn total_source_bytes(&self) -> usize {
        self.total_source_bytes
    }

    #[inline]
    pub fn source_digest(&self) -> u128 {
        self.source_digest
    }

    /// Extracts a deterministic slice of `len` bytes starting at a sample-derived offset.
    pub fn slice_for_sample(&self, salt: u64, sample_id: u64, len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let pool_len = self.raw_pool.len();
        let start = (hash_u64(
            b"corpus-slice",
            &[salt.to_le_bytes(), sample_id.to_le_bytes()].concat(),
        ) as usize)
            % pool_len;
        let mut pos = start;
        while out.len() < len {
            let take = (len - out.len()).min(pool_len - pos);
            out.extend_from_slice(&self.raw_pool[pos..pos + take]);
            pos = 0;
        }
        out
    }
}

fn collect_corpus_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    let mut children: Vec<PathBuf> = entries.filter_map(|e| e.ok().map(|x| x.path())).collect();
    children.sort();
    for path in children {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name == "target" || name == ".git" || name == "node_modules" {
                continue;
            }
        }
        if path.is_dir() {
            collect_corpus_files(&path, out);
        } else if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if matches!(ext, "rs" | "md" | "toml") {
                out.push(path);
            }
        }
    }
}

/// Supported data modalities in the 3-stage pipeline.
#[derive(Debug, Clone, Copy)]
pub enum Modality {
    /// Text / code shards decoded and packed into token ID tensors.
    LlmTokenShard { raw_bytes: usize },
    /// Structured RGB image tiles; `u8` compressed -> `f32` CHW decoded tensor
    /// (naturally yields ~5x–8x decoded-to-encoded memory expansion!).
    VisionImageShard {
        width: usize,
        height: usize,
        channels: usize,
    },
    /// Compressed 16-bit PCM waveform frames decoded and transformed into
    /// compact log-energy mel-like spectrogram tensors.
    AudioFeatureShard { frames: usize, frame_len: usize },
}

impl Modality {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::LlmTokenShard { .. } => "llm_token_shard",
            Self::VisionImageShard { .. } => "vision_image_shard",
            Self::AudioFeatureShard { .. } => "audio_feature_shard",
        }
    }
}

/// Encoded shard stored in the simulated S3 / VFS object store.
#[derive(Debug, Clone)]
pub struct EncodedShard {
    pub sample_id: u64,
    pub bytes: Vec<u8>,
    pub crc32: u32,
}

/// Materialized shard store for a dataset (reusable across concurrent jobs
/// that share the same `SharedPublic` dataset).
#[derive(Debug, Clone)]
pub struct DatasetStore {
    pub name: String,
    pub modality: Modality,
    pub dataset_digest: u128,
    pub shards: Vec<EncodedShard>,
    pub avg_encoded_bytes: f64,
}

impl DatasetStore {
    /// Generates `num_samples` real compressed shards derived from `corpus`.
    pub fn build(
        name: impl Into<String>,
        corpus: &CorpusStore,
        modality: Modality,
        num_samples: usize,
        dataset_salt: u64,
    ) -> Self {
        let name = name.into();
        let mut shards = Vec::with_capacity(num_samples);
        let mut total_encoded = 0usize;
        let mut digest_material = Vec::with_capacity(num_samples * 4 + 32);
        digest_material.extend_from_slice(name.as_bytes());
        digest_material.extend_from_slice(&dataset_salt.to_le_bytes());

        for s in 0..num_samples {
            let sample_id = s as u64;
            let raw_payload = build_raw_modality_payload(corpus, modality, dataset_salt, sample_id);
            let encoded = compress_lz_rle(&raw_payload);
            let c = crc32(&encoded);
            total_encoded += encoded.len();
            digest_material.extend_from_slice(&c.to_le_bytes());
            shards.push(EncodedShard {
                sample_id,
                bytes: encoded,
                crc32: c,
            });
        }

        let dataset_digest = content_hash_128(b"hanzo-seneca-dataset", &digest_material);
        let avg_encoded_bytes = if num_samples == 0 {
            0.0
        } else {
            total_encoded as f64 / num_samples as f64
        };

        Self {
            name,
            modality,
            dataset_digest,
            shards,
            avg_encoded_bytes,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.shards.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.shards.is_empty()
    }
}

/// Builds a realistic uncompressed modality payload seeded from real corpus bytes.
fn build_raw_modality_payload(
    corpus: &CorpusStore,
    modality: Modality,
    salt: u64,
    sample_id: u64,
) -> Vec<u8> {
    match modality {
        Modality::LlmTokenShard { raw_bytes } => {
            corpus.slice_for_sample(salt, sample_id, raw_bytes)
        }
        Modality::VisionImageShard {
            width,
            height,
            channels,
        } => {
            // Spatially structured RGB image tiles (4x4 macroblocks seeded from real
            // repository bytes) so LZ/RLE achieves realistic JPEG/PNG-like compression
            // ratios before u8->f32 tensor expansion (~6x total decoded expansion).
            let total = width * height * channels;
            let seed_bytes = corpus.slice_for_sample(salt, sample_id, (total / 8).max(64));
            let mut img = vec![0u8; total];
            let block = 4usize;
            for y in 0..height {
                let by = y / block;
                for x in 0..width {
                    let bx = x / block;
                    let blk_idx = (by * (width / block).max(1) + bx) % seed_bytes.len();
                    let base = seed_bytes[blk_idx];
                    // Quantize smoothly within 4x4 block with gentle local gradient.
                    let grad = (((x & 3) + (y & 3)) as u8) & 0x06;
                    let idx = (y * width + x) * channels;
                    for c in 0..channels {
                        let ch_shift = (c as u8).wrapping_mul(17);
                        img[idx + c] = base.wrapping_add(ch_shift).wrapping_add(grad) & 0xFC;
                    }
                }
            }
            img
        }
        Modality::AudioFeatureShard { frames, frame_len } => {
            // 16-bit little-endian PCM frames derived from real corpus bytes.
            let n_samples = frames * frame_len;
            let seed_bytes = corpus.slice_for_sample(salt, sample_id, frames.max(32));
            let mut pcm = Vec::with_capacity(n_samples * 2);
            for f in 0..frames {
                let base = i16::from(seed_bytes[f % seed_bytes.len()]) - 128;
                let amp = (base * 64).clamp(-16000, 16000);
                for t in 0..frame_len {
                    // Piecewise-smooth quantized waveform so LZ/RLE compresses ~2.5x–3x.
                    let phase = ((t / 2) & 7) as i16 - 4;
                    let val = (amp / 4).saturating_mul(phase) & !0x0F;
                    pcm.extend_from_slice(&val.to_le_bytes());
                }
            }
            pcm
        }
    }
}

/// Real CPU Stage-2 decode: decompresses LZ/RLE frame, verifies CRC32, and expands
/// into decoded tensor / token representation.
pub fn execute_decode(
    encoded: &[u8],
    modality: Modality,
    compute_passes: u32,
) -> Result<Vec<u8>, String> {
    let raw = decompress_lz_rle(encoded).map_err(|e| format!("decode failed: {e}"))?;
    let passes = compute_passes.max(1);

    match modality {
        Modality::LlmTokenShard { .. } => {
            // Normalize ASCII control chars & fold runs for tokenizer ingestion.
            let mut out = raw;
            for p in 0..passes {
                let mut acc = 0u32;
                for b in &mut out {
                    if *b == b'\r' {
                        *b = b'\n';
                    }
                    acc = acc.wrapping_add(u32::from(*b));
                }
                if p + 1 < passes {
                    // Prevent dead-code elimination while preserving decoded bytes.
                    let last = out.len().saturating_sub(1);
                    if !out.is_empty() && acc == u32::MAX {
                        out[last] ^= 0;
                    }
                }
            }
            Ok(out)
        }
        Modality::VisionImageShard { .. } => {
            // Unpack u8 RGB -> f32 normalized tensor (4x expansion over raw RGB;
            // combined with LZ/RLE compression yields ~6x–10x over encoded bytes!).
            let n = raw.len();
            let mut floats = vec![0.0f32; n];
            let inv = 1.0f32 / 255.0f32;
            for (dst, &src) in floats.iter_mut().zip(raw.iter()) {
                *dst = f32::from(src) * inv;
            }
            // Real spatial decode / color-space filter passes.
            for _ in 0..passes {
                if floats.len() >= 3 {
                    let mut prev = floats[0];
                    for i in 1..floats.len() - 1 {
                        let cur = floats[i];
                        let next = floats[i + 1];
                        floats[i] = 0.25 * prev + 0.5 * cur + 0.25 * next;
                        prev = cur;
                    }
                }
            }
            let mut out = Vec::with_capacity(n * 4);
            for v in floats {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
        Modality::AudioFeatureShard { .. } => {
            // Unpack i16 PCM -> f32 with pre-emphasis filter y[t] = x[t] - 0.97 * x[t-1].
            let n = raw.len() / 2;
            let mut pcm = vec![0.0f32; n];
            let inv = 1.0f32 / 32768.0f32;
            for i in 0..n {
                let s = i16::from_le_bytes([raw[2 * i], raw[2 * i + 1]]);
                pcm[i] = f32::from(s) * inv;
            }
            for _ in 0..passes {
                let mut prev = 0.0f32;
                for x in &mut pcm {
                    let cur = *x;
                    *x = cur - 0.97 * prev;
                    prev = cur;
                }
            }
            let mut out = Vec::with_capacity(n * 4);
            for v in pcm {
                out.extend_from_slice(&v.to_le_bytes());
            }
            Ok(out)
        }
    }
}

/// Real CPU Stage-3 transform: tokenization, deterministic/stochastic image
/// augmentation, or heavy spectrogram feature extraction.
pub fn execute_transform(
    decoded: &[u8],
    modality: Modality,
    stochastic: bool,
    compute_passes: u32,
    sample_id: u64,
    epoch: u32,
    seed: u64,
) -> Vec<u8> {
    let passes = compute_passes.max(1);
    match modality {
        Modality::LlmTokenShard { .. } => {
            // Deterministic byte-pair / trigram hash tokenization -> packed u32 token IDs.
            let num_tokens = (decoded.len() / 3).max(1);
            let mut tokens = vec![0u32; num_tokens];
            for pass in 0..passes {
                for (i, tok) in tokens.iter_mut().enumerate() {
                    let off = (i * 3).min(decoded.len().saturating_sub(1));
                    let b0 = u32::from(decoded[off]);
                    let b1 = u32::from(*decoded.get(off + 1).unwrap_or(&0));
                    let b2 = u32::from(*decoded.get(off + 2).unwrap_or(&0));
                    let h = ((b0 << 16) | (b1 << 8) | b2)
                        .wrapping_mul(2654435761)
                        .wrapping_add(pass);
                    *tok = h & 0x0001_FFFF; // 131072 vocab size
                }
            }
            let mut out = Vec::with_capacity(num_tokens * 4);
            for t in tokens {
                out.extend_from_slice(&t.to_le_bytes());
            }
            out
        }
        Modality::VisionImageShard { .. } => {
            let n = decoded.len() / 4;
            let mut vals = Vec::with_capacity(n);
            for &chunk in decoded.as_chunks::<4>().0 {
                vals.push(f32::from_le_bytes(chunk));
            }

            if stochastic {
                // Epoch-and-sample seeded stochastic augmentation:
                // random crop shift + horizontal flip + brightness/contrast jitter.
                let aug_seed = seed
                    ^ sample_id.wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    ^ (u64::from(epoch).wrapping_add(1)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
                let mut rng = Rng64::new(aug_seed);
                let flip = (rng.next_u64() & 1) == 1;
                let shift = rng.gen_range(vals.len().max(1));
                let contrast = 0.85 + 0.30 * (rng.next_f64() as f32);
                let brightness = -0.10 + 0.20 * (rng.next_f64() as f32);

                if flip {
                    vals.reverse();
                }
                if !vals.is_empty() {
                    let rot = shift % vals.len();
                    vals.rotate_left(rot);
                }
                for _ in 0..passes {
                    for v in &mut vals {
                        *v = (*v * contrast + brightness).clamp(-3.0, 3.0);
                    }
                }
                // Embed epoch augmentation nonce cleanly in a deterministic checksum-friendly way
                // while keeping tensor values finite and normalized.
                if let Some(first) = vals.first_mut() {
                    let jitter = ((rng.next_u64() & 0xFFFF) as f32) * 1e-6;
                    *first += jitter;
                }
            } else {
                // Deterministic ImageNet-style standardization + gamma curve.
                let means = [0.485f32, 0.456f32, 0.406f32];
                let inv_stds = [1.0 / 0.229f32, 1.0 / 0.224f32, 1.0 / 0.225f32];
                for _ in 0..passes {
                    for (i, v) in vals.iter_mut().enumerate() {
                        let c = i % 3;
                        *v = (*v - means[c]) * inv_stds[c];
                    }
                }
            }

            let mut out = Vec::with_capacity(vals.len() * 4);
            for v in vals {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
        Modality::AudioFeatureShard { frames, frame_len } => {
            // Heavy deterministic windowed filterbank / log-mel-like spectrogram extraction.
            // Output is compact: `frames * MEL_BINS * 4` bytes (much smaller than decoded PCM!).
            const MEL_BINS: usize = 24;
            let n = decoded.len() / 4;
            let mut pcm = Vec::with_capacity(n);
            for &chunk in decoded.as_chunks::<4>().0 {
                pcm.push(f32::from_le_bytes(chunk));
            }
            let actual_frames = frames.min((n / frame_len.max(1)).max(1));
            let mut mel = vec![0.0f32; actual_frames * MEL_BINS];

            for pass in 0..passes {
                let phase_bias = (pass as f32) * 0.01;
                for f in 0..actual_frames {
                    let start = f * frame_len;
                    let end = (start + frame_len).min(pcm.len());
                    let slice = &pcm[start..end];
                    for b in 0..MEL_BINS {
                        let freq = ((b + 1) as f32) * 0.13 + phase_bias;
                        let mut re = 0.0f32;
                        let mut im = 0.0f32;
                        for (t, &x) in slice.iter().enumerate() {
                            let angle = freq * (t as f32);
                            let (s, c) = angle.sin_cos();
                            re += x * c;
                            im -= x * s;
                        }
                        let energy = (re * re + im * im + 1e-6).ln();
                        mel[f * MEL_BINS + b] = energy;
                    }
                }
            }

            let mut out = Vec::with_capacity(mel.len() * 4);
            for v in mel {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out
        }
    }
}

/// Configuration for one concurrent training job in a benchmark scenario.
#[derive(Debug, Clone)]
pub struct JobConfig {
    pub job_id: u64,
    pub tenant_id: u64,
    pub name: String,
    pub dataset_index: usize,
    pub visibility: DatasetVisibility,
    pub stochastic_transform: bool,
    pub transform_profile: String,
    pub epochs: u32,
    pub batch_size: usize,
    /// Calibrated S3/VFS round-trip latency per shard miss (microseconds).
    pub io_rtt_us: f64,
    /// Calibrated S3/VFS read bandwidth (MiB/s).
    pub io_bw_mb_s: f64,
    /// Number of concurrent async S3/VFS fetches per job.
    pub io_concurrency: f64,
    /// Calibrated GPU forward+backward step duration per batch (microseconds).
    pub gpu_step_us: f64,
    /// Number of real CPU compute passes during decode.
    pub decode_compute_passes: u32,
    /// Number of real CPU compute passes during transform.
    pub transform_compute_passes: u32,
    /// Bounded batch prefetch queue depth between host loader and GPU.
    pub prefetch_batches: usize,
}

/// Configuration for a full multi-job training scenario.
#[derive(Debug, Clone)]
pub struct ScenarioConfig {
    pub name: String,
    pub description: String,
    pub is_negative_regime: bool,
    pub total_cache_bytes: usize,
    pub host_cpu_cores: usize,
    pub seed: u64,
    pub datasets: Vec<DatasetStore>,
    pub jobs: Vec<JobConfig>,
}

/// Detailed per-job measured metrics.
#[derive(Debug, Clone)]
pub struct JobMetrics {
    pub job_id: u64,
    pub tenant_id: u64,
    pub name: String,
    pub modality: String,
    pub visibility: String,
    pub stochastic_transform: bool,
    pub examples_processed: u64,
    pub makespan_ms: f64,
    pub gpu_idle_ms: f64,
    pub gpu_idle_pct: f64,
    pub examples_per_sec: f64,
    pub encoded_hits: u64,
    pub decoded_hits: u64,
    pub transformed_hits: u64,
    pub storage_fetches: u64,
    pub s3_bytes_fetched: u64,
    pub s3_bytes_saved: u64,
    pub avg_encoded_bytes: f64,
    pub avg_decoded_bytes: f64,
    pub avg_transformed_bytes: f64,
    pub decode_expansion_ratio: f64,
    pub final_budget_encoded: usize,
    pub final_budget_decoded: usize,
    pub final_budget_transformed: usize,
}

/// Full measured results for one `(scenario, policy)` evaluation.
#[derive(Debug, Clone)]
pub struct PolicyRunMetrics {
    pub scenario: String,
    pub policy: String,
    pub seed: u64,
    pub is_negative_regime: bool,
    pub total_examples: u64,
    pub makespan_ms: f64,
    pub wall_time_ms: f64,
    pub aggregate_examples_per_sec: f64,
    pub gpu_idle_time_pct: f64,
    pub encoded_hit_rate_pct: f64,
    pub decoded_hit_rate_pct: f64,
    pub transformed_hit_rate_pct: f64,
    pub overall_cache_hit_rate_pct: f64,
    pub cache_budget_bytes: usize,
    pub cache_peak_used_bytes: usize,
    pub cache_avg_utilization_pct: f64,
    pub cache_peak_utilization_pct: f64,
    pub s3_bytes_fetched: u64,
    pub s3_bytes_saved: u64,
    pub s3_bytes_saved_pct: f64,
    pub stochastic_checks_total: u64,
    pub stochastic_checks_passed: u64,
    pub stochastic_freshness_pct: f64,
    pub cross_tenant_leaks: u64,
    pub unauthorized_probes_blocked: u64,
    pub rebalance_count: u64,
    pub streaming_bypass_activations: u64,
    pub per_job: Vec<JobMetrics>,
}

/// Event-queue entry ordering concurrent jobs by their next batch timestamp.
#[derive(Debug, Clone, Copy)]
struct ScheduledJob {
    next_time_us: f64,
    job_idx: usize,
}

impl PartialEq for ScheduledJob {
    fn eq(&self, other: &Self) -> bool {
        self.job_idx == other.job_idx && self.next_time_us.to_bits() == other.next_time_us.to_bits()
    }
}
impl Eq for ScheduledJob {}

impl Ord for ScheduledJob {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap by `next_time_us`, tie-breaking by `job_idx`.
        other
            .next_time_us
            .total_cmp(&self.next_time_us)
            .then_with(|| other.job_idx.cmp(&self.job_idx))
    }
}

impl PartialOrd for ScheduledJob {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Computes a deterministic 64-bit identity hash for the Stage-2 decoder configuration
/// (modality geometry + effective `decode_compute_passes`).
pub fn decode_profile_hash(modality: Modality, decode_compute_passes: u32) -> u64 {
    let passes = decode_compute_passes.max(1);
    let mut material = Vec::with_capacity(32);
    match modality {
        Modality::LlmTokenShard { raw_bytes } => {
            material.push(1u8);
            material.extend_from_slice(&(raw_bytes as u64).to_le_bytes());
        }
        Modality::VisionImageShard {
            width,
            height,
            channels,
        } => {
            material.push(2u8);
            material.extend_from_slice(&(width as u64).to_le_bytes());
            material.extend_from_slice(&(height as u64).to_le_bytes());
            material.extend_from_slice(&(channels as u64).to_le_bytes());
        }
        Modality::AudioFeatureShard { frames, frame_len } => {
            material.push(3u8);
            material.extend_from_slice(&(frames as u64).to_le_bytes());
            material.extend_from_slice(&(frame_len as u64).to_le_bytes());
        }
    }
    material.extend_from_slice(&passes.to_le_bytes());
    hash_u64(b"seneca-decode-v1", &material)
}

/// Computes a deterministic 64-bit identity hash for the Stage-3 deterministic
/// transform configuration, chained with the upstream `decode_hash` and
/// effective `transform_compute_passes`.
pub fn transform_profile_hash(
    decode_hash: u64,
    transform_profile: &str,
    transform_compute_passes: u32,
) -> u64 {
    let passes = transform_compute_passes.max(1);
    let mut material = Vec::with_capacity(12 + transform_profile.len());
    material.extend_from_slice(&decode_hash.to_le_bytes());
    material.extend_from_slice(&passes.to_le_bytes());
    material.extend_from_slice(transform_profile.as_bytes());
    hash_u64(b"seneca-transform-v1", &material)
}

/// Per-job runtime state during concurrent scenario execution.
struct JobRuntime {
    cfg: JobConfig,
    dataset_digest: u128,
    decode_hash: u64,
    transform_hash: u64,
    scope: ScopeId,
    rng: Rng64,
    current_epoch: u32,
    epoch_queue: Vec<u64>,
    epoch_visited: HashSet<u64>,
    finished: bool,

    // Producer-consumer GPU overlap timeline (microseconds).
    loader_free_us: f64,
    gpu_free_us: f64,
    gpu_idle_us: f64,
    gpu_active_us: f64,
    gpu_slot_free_us: Vec<f64>,
    step_count: u64,

    // Per-job counters.
    examples_processed: u64,
    encoded_hits: u64,
    decoded_hits: u64,
    transformed_hits: u64,
    storage_fetches: u64,
    s3_bytes_fetched: u64,
    s3_bytes_saved: u64,
    // Stochastic freshness audit: sample_id -> seen transformed CRC32 across prior epochs.
    stochastic_history: HashMap<u64, HashSet<u32>>,
    // Track recent private keys for cross-tenant adversarial isolation audits.
    sample_private_keys: Vec<CacheKey>,
    // Rolling uncontended CPU core utilization in [0, 1] for cross-job contention accounting.
    last_cpu_util: f64,
    // Last non-zero / active steady-state partition budgets observed while job was active.
    last_active_budget_encoded: usize,
    last_active_budget_decoded: usize,
    last_active_budget_transformed: usize,
}

impl JobRuntime {
    fn new(cfg: JobConfig, dataset: &DatasetStore, seed: u64) -> Self {
        let decode_hash = decode_profile_hash(dataset.modality, cfg.decode_compute_passes);
        let transform_hash = transform_profile_hash(
            decode_hash,
            &cfg.transform_profile,
            cfg.transform_compute_passes,
        );
        let scope = ScopeId::from_visibility(cfg.visibility);
        let job_seed = seed ^ cfg.job_id.wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let mut rng = Rng64::new(job_seed);
        let finished = dataset.is_empty() || cfg.epochs == 0;
        let mut epoch_queue: Vec<u64> = if finished {
            Vec::new()
        } else {
            (0..dataset.len() as u64).collect()
        };
        rng.shuffle(&mut epoch_queue);

        let prefetch = cfg.prefetch_batches.max(1);
        Self {
            cfg,
            dataset_digest: dataset.dataset_digest,
            decode_hash,
            transform_hash,
            scope,
            rng,
            current_epoch: 0,
            epoch_queue,
            epoch_visited: HashSet::new(),
            finished,
            loader_free_us: 0.0,
            gpu_free_us: 0.0,
            gpu_idle_us: 0.0,
            gpu_active_us: 0.0,
            gpu_slot_free_us: vec![0.0; prefetch],
            step_count: 0,
            examples_processed: 0,
            encoded_hits: 0,
            decoded_hits: 0,
            transformed_hits: 0,
            storage_fetches: 0,
            s3_bytes_fetched: 0,
            s3_bytes_saved: 0,
            stochastic_history: HashMap::new(),
            sample_private_keys: Vec::new(),
            last_cpu_util: 0.0,
            last_active_budget_encoded: 0,
            last_active_budget_decoded: 0,
            last_active_budget_transformed: 0,
        }
    }

    /// Draws the next batch of sample IDs for the current epoch.
    /// When `enable_ods` is true (Seneca Opportunistic Data Sampling), prefers
    /// unvisited samples within a bounded lookahead window that are already cached,
    /// while guaranteeing every sample is visited **exactly once per epoch**.
    ///
    /// Does NOT advance `current_epoch` or refill `epoch_queue`; epoch transitions
    /// are owned solely by [`Self::refill_epoch_if_needed`].
    fn draw_batch(&mut self, cache: &MultiStageCache, enable_ods: bool) -> Vec<(u64, u32)> {
        let target = self.cfg.batch_size.max(1);
        let mut batch = Vec::with_capacity(target);

        while batch.len() < target && !self.epoch_queue.is_empty() && !self.finished {
            let chosen_idx = if enable_ods && self.epoch_queue.len() > 1 {
                let window = self.epoch_queue.len().min(target * 4);
                let mut best_pos = self.epoch_queue.len() - 1;
                let mut best_tier = 0u8;
                let start = self.epoch_queue.len() - window;
                for pos in (start..self.epoch_queue.len()).rev() {
                    let sid = self.epoch_queue[pos];
                    let tier = self.cached_tier(cache, sid);
                    if tier > best_tier {
                        best_tier = tier;
                        best_pos = pos;
                        if best_tier == 3 {
                            break;
                        }
                    }
                }
                best_pos
            } else {
                self.epoch_queue.len() - 1
            };

            let sid = self.epoch_queue.swap_remove(chosen_idx);
            debug_assert!(
                self.epoch_visited.insert(sid),
                "duplicate sample {sid} within epoch {} for job {}",
                self.current_epoch,
                self.cfg.job_id
            );
            batch.push((sid, self.current_epoch));
        }

        batch
    }

    /// Sole owner of epoch advancement and queue refilling once an epoch is exhausted.
    fn refill_epoch_if_needed(&mut self, dataset_len: usize) {
        if self.epoch_queue.is_empty() && !self.finished {
            self.current_epoch += 1;
            if self.current_epoch >= self.cfg.epochs || dataset_len == 0 {
                self.finished = true;
                return;
            }
            self.epoch_visited.clear();
            self.epoch_queue = (0..dataset_len as u64).collect();
            self.rng.shuffle(&mut self.epoch_queue);
        }
    }

    fn cached_tier(&self, cache: &MultiStageCache, sample_id: u64) -> u8 {
        if !self.cfg.stochastic_transform {
            let kt = CacheKey::new(
                self.cfg.visibility,
                self.dataset_digest,
                sample_id,
                CacheStage::Transformed,
                self.transform_hash,
            );
            if cache.contains_accessible(&kt, self.cfg.tenant_id, self.cfg.job_id) {
                return 3;
            }
        }
        let kd = CacheKey::new(
            self.cfg.visibility,
            self.dataset_digest,
            sample_id,
            CacheStage::Decoded,
            self.decode_hash,
        );
        if cache.contains_accessible(&kd, self.cfg.tenant_id, self.cfg.job_id) {
            return 2;
        }
        let ke = CacheKey::new(
            self.cfg.visibility,
            self.dataset_digest,
            sample_id,
            CacheStage::Encoded,
            0,
        );
        if cache.contains_accessible(&ke, self.cfg.tenant_id, self.cfg.job_id) {
            return 1;
        }
        0
    }
}

/// Runs a full concurrent multi-job training scenario under the specified
/// [`AllocationPolicy`] and returns all measured metrics.
pub fn run_scenario(scenario: &ScenarioConfig, policy: AllocationPolicy) -> PolicyRunMetrics {
    let wall_start = Instant::now();
    let mut cache = MultiStageCache::new(scenario.total_cache_bytes);
    let mut allocator = SenecaAllocator::new(AllocatorConfig::new(
        policy,
        scenario.total_cache_bytes,
        scenario.host_cpu_cores,
    ));

    let mut runtimes: Vec<JobRuntime> = scenario
        .jobs
        .iter()
        .map(|cfg| {
            JobRuntime::new(
                cfg.clone(),
                &scenario.datasets[cfg.dataset_index],
                scenario.seed,
            )
        })
        .collect();

    let mut telemetry: Vec<JobTelemetry> = scenario
        .jobs
        .iter()
        .map(|cfg| {
            let ds = &scenario.datasets[cfg.dataset_index];
            JobTelemetry::new(
                cfg.job_id,
                ScopeId::from_visibility(cfg.visibility),
                cfg.stochastic_transform,
                ds.len(),
                cfg.epochs,
                cfg.batch_size,
                cfg.gpu_step_us,
                cfg.io_concurrency,
            )
        })
        .collect();

    // Initial allocation before step 0.
    let mut decision = allocator.rebalance(0, &telemetry, true);
    cache.apply_budgets(&decision.budgets);

    // Event-driven concurrent scheduler across all jobs.
    let mut heap = BinaryHeap::new();
    for (idx, _) in runtimes.iter().enumerate() {
        heap.push(ScheduledJob {
            next_time_us: 0.0,
            job_idx: idx,
        });
    }

    let mut global_step = 0u64;
    let mut stochastic_checks_total = 0u64;
    let mut stochastic_checks_passed = 0u64;
    let mut post_warmup_rebalanced = false;

    while let Some(ScheduledJob { job_idx, .. }) = heap.pop() {
        if runtimes[job_idx].finished {
            continue;
        }
        let ds_idx = runtimes[job_idx].cfg.dataset_index;
        let dataset = &scenario.datasets[ds_idx];

        let enable_ods = policy == AllocationPolicy::SenecaDynamic && decision.admit_enabled;
        let batch = runtimes[job_idx].draw_batch(&cache, enable_ods);

        if batch.is_empty() {
            runtimes[job_idx].finished = true;
            telemetry[job_idx].active = false;
            continue;
        }

        // Determine Seneca's disjoint stage placement target if running SenecaDynamic.
        let scope = runtimes[job_idx].scope;
        let b_e = cache.partition_budget(scope, CacheStage::Encoded);
        let b_d = cache.partition_budget(scope, CacheStage::Decoded);
        let b_t = if runtimes[job_idx].cfg.stochastic_transform {
            0
        } else {
            cache.partition_budget(scope, CacheStage::Transformed)
        };
        runtimes[job_idx].last_active_budget_encoded = b_e;
        runtimes[job_idx].last_active_budget_decoded = b_d;
        runtimes[job_idx].last_active_budget_transformed = b_t;

        // Zero-overhead fast path when cache is empty and admission is disabled
        // (both `no_cache` and `seneca_dynamic` streaming bypass).
        let cache_lookup_active = decision.admit_enabled || cache.total_used_bytes() > 0;

        let mut batch_io_us = 0.0f64;
        let mut batch_cpu_us = 0.0f64;

        for (sample_id, epoch) in batch {
            if !decision.streaming_bypass_active {
                telemetry[job_idx].record_access(sample_id);
            }
            let shard = &dataset.shards[sample_id as usize];
            let vis = runtimes[job_idx].cfg.visibility;
            let digest = dataset.dataset_digest;
            let tenant = runtimes[job_idx].cfg.tenant_id;
            let jid = runtimes[job_idx].cfg.job_id;
            let stochastic = runtimes[job_idx].cfg.stochastic_transform;
            let dhash = runtimes[job_idx].decode_hash;
            let thash = runtimes[job_idx].transform_hash;

            let key_t = CacheKey::new(vis, digest, sample_id, CacheStage::Transformed, thash);
            let key_d = CacheKey::new(vis, digest, sample_id, CacheStage::Decoded, dhash);
            let key_e = CacheKey::new(vis, digest, sample_id, CacheStage::Encoded, 0);

            if matches!(vis, DatasetVisibility::Private { .. })
                && runtimes[job_idx].sample_private_keys.len() < 16
            {
                runtimes[job_idx].sample_private_keys.push(key_e);
                runtimes[job_idx].sample_private_keys.push(key_d);
            }

            // Stage routing for SenecaDynamic (disjoint, zero cross-stage duplication +
            // deterministic pinning to avoid multi-epoch LRU scan thrashing).
            let seneca_target_stage =
                if policy == AllocationPolicy::SenecaDynamic && decision.admit_enabled {
                    choose_seneca_stage(
                        sample_id,
                        dataset.len(),
                        stochastic,
                        b_e,
                        b_d,
                        b_t,
                        &telemetry[job_idx],
                    )
                } else {
                    None
                };

            let mut transformed_bytes: Option<Vec<u8>> = None;

            // 1. Try Transformed cache (STRICTLY forbidden when `stochastic == true`!).
            if cache_lookup_active && !stochastic && b_t > 0 {
                let t0 = Instant::now();
                let hit_t = cache.get(&key_t, tenant, jid).map(|e| e.payload.clone());
                batch_cpu_us += t0.elapsed().as_secs_f64() * 1e6;
                if let Some(buf) = hit_t {
                    runtimes[job_idx].transformed_hits += 1;
                    runtimes[job_idx].s3_bytes_saved += shard.bytes.len() as u64;
                    transformed_bytes = Some(buf);
                }
            }

            // 2. On Transformed miss (or stochastic job), try Decoded cache.
            if transformed_bytes.is_none() {
                let hit_d = if cache_lookup_active && b_d > 0 {
                    let t0 = Instant::now();
                    let h = cache.get(&key_d, tenant, jid).map(|e| e.payload.clone());
                    batch_cpu_us += t0.elapsed().as_secs_f64() * 1e6;
                    h
                } else {
                    None
                };

                let decoded_buf = if let Some(buf) = hit_d {
                    runtimes[job_idx].decoded_hits += 1;
                    runtimes[job_idx].s3_bytes_saved += shard.bytes.len() as u64;
                    buf
                } else {
                    // 3. Try Encoded cache.
                    let hit_e = if cache_lookup_active && b_e > 0 {
                        let t1 = Instant::now();
                        let h = cache.get(&key_e, tenant, jid).map(|e| e.payload.clone());
                        batch_cpu_us += t1.elapsed().as_secs_f64() * 1e6;
                        h
                    } else {
                        None
                    };

                    let encoded_buf = if let Some(buf) = hit_e {
                        runtimes[job_idx].encoded_hits += 1;
                        runtimes[job_idx].s3_bytes_saved += shard.bytes.len() as u64;
                        buf
                    } else {
                        // 4. Storage (S3 / VFS) fetch with real copy + CRC32 verification.
                        let t_fetch_cpu = Instant::now();
                        let fetched = shard.bytes.clone();
                        let verified_crc = crc32(&fetched);
                        assert_eq!(
                            verified_crc, shard.crc32,
                            "S3/VFS shard CRC32 mismatch on sample {sample_id}"
                        );
                        batch_cpu_us += t_fetch_cpu.elapsed().as_secs_f64() * 1e6;

                        let transfer_us = (fetched.len() as f64)
                            / (runtimes[job_idx].cfg.io_bw_mb_s * 1024.0 * 1024.0)
                            * 1e6;
                        let fetch_latency_us = runtimes[job_idx].cfg.io_rtt_us + transfer_us;
                        batch_io_us += fetch_latency_us;
                        telemetry[job_idx].observe_fetch(fetch_latency_us, fetched.len());

                        runtimes[job_idx].storage_fetches += 1;
                        runtimes[job_idx].s3_bytes_fetched += fetched.len() as u64;

                        // Admit Encoded if allowed by policy & stage routing.
                        let admit_e = decision.admit_enabled
                            && match policy {
                                AllocationPolicy::SenecaDynamic => {
                                    seneca_target_stage == Some(CacheStage::Encoded)
                                }
                                _ => b_e > 0,
                            };
                        if admit_e {
                            let tc = Instant::now();
                            cache.insert(
                                key_e,
                                CacheEntry::new(fetched.clone(), verified_crc, epoch),
                                tenant,
                                jid,
                                true,
                            );
                            batch_cpu_us += tc.elapsed().as_secs_f64() * 1e6;
                        }
                        fetched
                    };

                    // Execute real CPU decode!
                    let td = Instant::now();
                    let dec = execute_decode(
                        &encoded_buf,
                        dataset.modality,
                        runtimes[job_idx].cfg.decode_compute_passes,
                    )
                    .expect("real shard decode must succeed");
                    let dec_us = td.elapsed().as_secs_f64() * 1e6;
                    batch_cpu_us += dec_us;
                    telemetry[job_idx].observe_decode(dec_us, dec.len());

                    // Admit Decoded if allowed by policy & stage routing.
                    let admit_d = decision.admit_enabled
                        && match policy {
                            AllocationPolicy::SenecaDynamic => {
                                seneca_target_stage == Some(CacheStage::Decoded)
                            }
                            _ => b_d > 0,
                        };
                    if admit_d {
                        let tc = Instant::now();
                        let dc = crc32(&dec);
                        cache.insert(
                            key_d,
                            CacheEntry::new(dec.clone(), dc, epoch),
                            tenant,
                            jid,
                            true,
                        );
                        batch_cpu_us += tc.elapsed().as_secs_f64() * 1e6;
                    }
                    dec
                };

                // Execute real CPU transform!
                let tt = Instant::now();
                let trans = execute_transform(
                    &decoded_buf,
                    dataset.modality,
                    stochastic,
                    runtimes[job_idx].cfg.transform_compute_passes,
                    sample_id,
                    epoch,
                    scenario.seed,
                );
                let trans_us = tt.elapsed().as_secs_f64() * 1e6;
                batch_cpu_us += trans_us;
                telemetry[job_idx].observe_transform(trans_us, trans.len());

                let trans_crc = crc32(&trans);

                // Stochastic-augmentation freshness audit across epochs.
                if stochastic {
                    let hist = runtimes[job_idx]
                        .stochastic_history
                        .entry(sample_id)
                        .or_default();
                    if epoch > 0 {
                        stochastic_checks_total += 1;
                        if !hist.contains(&trans_crc) {
                            stochastic_checks_passed += 1;
                        }
                    }
                    hist.insert(trans_crc);
                } else {
                    // Admit deterministic Transformed entry if allowed.
                    let admit_t = decision.admit_enabled
                        && match policy {
                            AllocationPolicy::SenecaDynamic => {
                                seneca_target_stage == Some(CacheStage::Transformed)
                            }
                            _ => b_t > 0,
                        };
                    if admit_t {
                        let tc = Instant::now();
                        cache.insert(
                            key_t,
                            CacheEntry::new(trans.clone(), trans_crc, epoch),
                            tenant,
                            jid,
                            true,
                        );
                        batch_cpu_us += tc.elapsed().as_secs_f64() * 1e6;
                    }
                }

                transformed_bytes = Some(trans);
            }

            // Consume transformed tensor on GPU (verify non-empty output).
            let out = transformed_bytes.expect("pipeline must produce transformed tensor");
            debug_assert!(!out.is_empty());
            runtimes[job_idx].examples_processed += 1;
        }

        // Model concurrent host CPU core contention from *other* active jobs only
        // (never charging a job for contention with its own CPU work).
        let eff_io_us = batch_io_us / runtimes[job_idx].cfg.io_concurrency.max(1.0);
        runtimes[job_idx].last_cpu_util =
            job_cpu_utilization(batch_cpu_us, eff_io_us, runtimes[job_idx].cfg.gpu_step_us);
        let other_cpu_util: f64 = runtimes
            .iter()
            .enumerate()
            .filter(|&(idx, r)| idx != job_idx && !r.finished)
            .map(|(_, r)| r.last_cpu_util)
            .sum();
        let contention = cpu_contention_factor(other_cpu_util, scenario.host_cpu_cores);
        let eff_cpu_us = batch_cpu_us * contention;
        let batch_prep_us = eff_io_us + eff_cpu_us;

        // Producer-consumer GPU timeline with bounded prefetch ring.
        let slot_idx =
            (runtimes[job_idx].step_count as usize) % runtimes[job_idx].gpu_slot_free_us.len();
        let slot_avail = runtimes[job_idx].gpu_slot_free_us[slot_idx];
        let load_start = runtimes[job_idx].loader_free_us.max(slot_avail);
        let data_ready = load_start + batch_prep_us;
        runtimes[job_idx].loader_free_us = data_ready;

        let gpu_start = runtimes[job_idx].gpu_free_us.max(data_ready);
        let idle = gpu_start - runtimes[job_idx].gpu_free_us;
        runtimes[job_idx].gpu_idle_us += idle;
        runtimes[job_idx].gpu_active_us += runtimes[job_idx].cfg.gpu_step_us;
        let gpu_end = gpu_start + runtimes[job_idx].cfg.gpu_step_us;
        runtimes[job_idx].gpu_free_us = gpu_end;
        runtimes[job_idx].gpu_slot_free_us[slot_idx] = gpu_start;
        runtimes[job_idx].step_count += 1;

        global_step += 1;

        // Rebalance Seneca once all jobs have completed their initial telemetry profile,
        // and periodically thereafter (with hysteresis preventing thrashing).
        let all_profiled = telemetry
            .iter()
            .filter(|t| t.active)
            .all(JobTelemetry::is_profiled);
        let force_rebalance = !post_warmup_rebalanced && all_profiled;
        if force_rebalance {
            post_warmup_rebalanced = true;
        }
        let poll_interval = if decision.streaming_bypass_active {
            32
        } else {
            4
        };
        if force_rebalance || global_step.is_multiple_of(poll_interval) {
            let t_alloc = Instant::now();
            let new_dec = allocator.rebalance(global_step, &telemetry, force_rebalance);
            if new_dec.rebalance_applied {
                cache.apply_budgets(&new_dec.budgets);
            }
            decision = new_dec;
            let _alloc_us = t_alloc.elapsed().as_secs_f64() * 1e6;
        }

        // Advance and refill epoch at most once after this batch completes.
        runtimes[job_idx].refill_epoch_if_needed(dataset.len());

        if runtimes[job_idx].finished {
            telemetry[job_idx].active = false;
            // Rebalance remaining active jobs if any.
            let new_dec = allocator.rebalance(global_step, &telemetry, true);
            cache.apply_budgets(&new_dec.budgets);
            decision = new_dec;
        } else {
            heap.push(ScheduledJob {
                next_time_us: runtimes[job_idx].gpu_free_us,
                job_idx,
            });
        }
    }

    // Active cross-tenant adversarial isolation audit: every tenant probes every
    // other tenant's private keys and MUST be blocked with 0 leaks.
    let private_probes: Vec<(u64, u64, Vec<CacheKey>)> = runtimes
        .iter()
        .map(|r| (r.cfg.tenant_id, r.cfg.job_id, r.sample_private_keys.clone()))
        .collect();
    for (victim_idx, (v_tenant, v_job, v_keys)) in private_probes.iter().enumerate() {
        for (attacker_idx, (a_tenant, a_job, _)) in private_probes.iter().enumerate() {
            if victim_idx == attacker_idx || (v_tenant == a_tenant && v_job == a_job) {
                continue;
            }
            for k in v_keys {
                let res = cache.probe_cross_tenant(k, *a_tenant, *a_job);
                assert!(
                    res.is_none(),
                    "SECURITY VIOLATION: cross-tenant private key leak from tenant {v_tenant}/job {v_job} to tenant {a_tenant}/job {a_job}"
                );
            }
        }
    }

    let wall_time_ms = wall_start.elapsed().as_secs_f64() * 1e3;

    // Aggregate metrics across all jobs.
    let makespan_us = runtimes
        .iter()
        .map(|r| r.gpu_free_us)
        .fold(0.0f64, f64::max)
        .max(1.0);
    let total_gpu_idle_us: f64 = runtimes.iter().map(|r| r.gpu_idle_us).sum();
    let total_gpu_wall_us: f64 = runtimes.iter().map(|r| r.gpu_free_us).sum::<f64>().max(1.0);
    let gpu_idle_time_pct = 100.0 * total_gpu_idle_us / total_gpu_wall_us;

    let total_examples: u64 = runtimes.iter().map(|r| r.examples_processed).sum();
    let aggregate_examples_per_sec = (total_examples as f64) / (makespan_us / 1e6);

    let total_s3_fetched: u64 = runtimes.iter().map(|r| r.s3_bytes_fetched).sum();
    let total_s3_saved: u64 = runtimes.iter().map(|r| r.s3_bytes_saved).sum();
    let total_s3_demand = total_s3_fetched + total_s3_saved;
    let s3_bytes_saved_pct = if total_s3_demand == 0 {
        0.0
    } else {
        100.0 * (total_s3_saved as f64) / (total_s3_demand as f64)
    };

    let enc_hits: u64 = runtimes.iter().map(|r| r.encoded_hits).sum();
    let dec_hits: u64 = runtimes.iter().map(|r| r.decoded_hits).sum();
    let trans_hits: u64 = runtimes.iter().map(|r| r.transformed_hits).sum();
    let denom = total_examples.max(1) as f64;
    let encoded_hit_rate_pct = 100.0 * (enc_hits as f64) / denom;
    let decoded_hit_rate_pct = 100.0 * (dec_hits as f64) / denom;
    let transformed_hit_rate_pct = 100.0 * (trans_hits as f64) / denom;
    let overall_cache_hit_rate_pct = 100.0 * ((enc_hits + dec_hits + trans_hits) as f64) / denom;

    let stochastic_freshness_pct = if stochastic_checks_total == 0 {
        100.0
    } else {
        100.0 * (stochastic_checks_passed as f64) / (stochastic_checks_total as f64)
    };

    let per_job: Vec<JobMetrics> = runtimes
        .iter()
        .enumerate()
        .map(|(idx, r)| {
            let ds = &scenario.datasets[r.cfg.dataset_index];
            let t = &telemetry[idx];
            let j_makespan_ms = r.gpu_free_us / 1e3;
            let j_idle_pct = if r.gpu_free_us <= 0.0 {
                0.0
            } else {
                100.0 * r.gpu_idle_us / r.gpu_free_us
            };
            let j_eps = if r.gpu_free_us <= 0.0 {
                0.0
            } else {
                (r.examples_processed as f64) / (r.gpu_free_us / 1e6)
            };
            let vis_str = match r.cfg.visibility {
                DatasetVisibility::Private { .. } => "private".to_string(),
                DatasetVisibility::SharedPublic { .. } => "shared_public".to_string(),
            };
            JobMetrics {
                job_id: r.cfg.job_id,
                tenant_id: r.cfg.tenant_id,
                name: r.cfg.name.clone(),
                modality: ds.modality.as_str().to_string(),
                visibility: vis_str,
                stochastic_transform: r.cfg.stochastic_transform,
                examples_processed: r.examples_processed,
                makespan_ms: j_makespan_ms,
                gpu_idle_ms: r.gpu_idle_us / 1e3,
                gpu_idle_pct: j_idle_pct,
                examples_per_sec: j_eps,
                encoded_hits: r.encoded_hits,
                decoded_hits: r.decoded_hits,
                transformed_hits: r.transformed_hits,
                storage_fetches: r.storage_fetches,
                s3_bytes_fetched: r.s3_bytes_fetched,
                s3_bytes_saved: r.s3_bytes_saved,
                avg_encoded_bytes: t.size_encoded,
                avg_decoded_bytes: t.size_decoded,
                avg_transformed_bytes: t.size_transformed,
                decode_expansion_ratio: t.decode_expansion_ratio(),
                final_budget_encoded: r.last_active_budget_encoded,
                final_budget_decoded: r.last_active_budget_decoded,
                final_budget_transformed: if r.cfg.stochastic_transform {
                    0
                } else {
                    r.last_active_budget_transformed
                },
            }
        })
        .collect();

    PolicyRunMetrics {
        scenario: scenario.name.clone(),
        policy: policy.as_str().to_string(),
        seed: scenario.seed,
        is_negative_regime: scenario.is_negative_regime,
        total_examples,
        makespan_ms: makespan_us / 1e3,
        wall_time_ms,
        aggregate_examples_per_sec,
        gpu_idle_time_pct,
        encoded_hit_rate_pct,
        decoded_hit_rate_pct,
        transformed_hit_rate_pct,
        overall_cache_hit_rate_pct,
        cache_budget_bytes: cache.total_budget_bytes(),
        cache_peak_used_bytes: cache.peak_used_bytes(),
        cache_avg_utilization_pct: cache.avg_utilization_pct(),
        cache_peak_utilization_pct: cache.peak_utilization_pct(),
        s3_bytes_fetched: total_s3_fetched,
        s3_bytes_saved: total_s3_saved,
        s3_bytes_saved_pct,
        stochastic_checks_total,
        stochastic_checks_passed,
        stochastic_freshness_pct,
        cross_tenant_leaks: cache.cross_tenant_leaks(),
        unauthorized_probes_blocked: cache.unauthorized_probes_blocked(),
        rebalance_count: allocator.rebalance_count(),
        streaming_bypass_activations: allocator.bypass_activations(),
        per_job,
    }
}

/// Deterministic non-overlapping stage selector for `SenecaDynamic`.
/// Maps each sample ID to at most ONE stage (`Transformed`, `Decoded`, or `Encoded`)
/// matching the exact item capacities `(n_T, n_D, n_E)` of the current partition budgets,
/// preventing both intra-job cross-stage byte duplication and cyclic LRU scan thrashing!
fn choose_seneca_stage(
    sample_id: u64,
    working_set: usize,
    stochastic: bool,
    b_e: usize,
    b_d: usize,
    b_t: usize,
    t: &JobTelemetry,
) -> Option<CacheStage> {
    let ws = working_set.max(1) as f64;
    let s_e = t.size_encoded.max(1.0);
    let s_d = t.size_decoded.max(1.0);
    let s_t = t.size_transformed.max(1.0);

    // Leave a tiny 4% safety margin for per-sample size variance so pinned items
    // don't churn each other out of the partition.
    let n_t = if stochastic || b_t == 0 {
        0.0
    } else {
        ((b_t as f64) * 0.96 / s_t).floor()
    };
    let n_d = if b_d == 0 {
        0.0
    } else {
        ((b_d as f64) * 0.96 / s_d).floor()
    };
    let n_e = if b_e == 0 {
        0.0
    } else {
        ((b_e as f64) * 0.96 / s_e).floor()
    };

    if n_t + n_d + n_e < 1.0 {
        return None;
    }

    let rank =
        (hash_u64(b"seneca-pin-v1", &sample_id.to_le_bytes()) % (working_set.max(1) as u64)) as f64;
    if rank < n_t.min(ws) {
        Some(CacheStage::Transformed)
    } else if rank < (n_t + n_d).min(ws) {
        Some(CacheStage::Decoded)
    } else if rank < (n_t + n_d + n_e).min(ws) {
        Some(CacheStage::Encoded)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stochastic_augmentation_is_100_percent_fresh_across_epochs() {
        let corpus =
            CorpusStore::from_bytes(b"stochastic vision test corpus 0123456789 ".repeat(512));
        let ds = DatasetStore::build(
            "vision_stochastic_test",
            &corpus,
            Modality::VisionImageShard {
                width: 24,
                height: 24,
                channels: 3,
            },
            32,
            101,
        );
        let vis = DatasetVisibility::Private {
            tenant_id: 1,
            job_id: 1,
        };
        let scenario = ScenarioConfig {
            name: "stochastic_unit".to_string(),
            description: "verify stochastic freshness".to_string(),
            is_negative_regime: false,
            total_cache_bytes: 512 * 1024,
            host_cpu_cores: 2,
            seed: 1349,
            datasets: vec![ds],
            jobs: vec![JobConfig {
                job_id: 1,
                tenant_id: 1,
                name: "vision_aug".to_string(),
                dataset_index: 0,
                visibility: vis,
                stochastic_transform: true,
                transform_profile: "rand_crop_flip_jitter".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 100.0,
                io_bw_mb_s: 100.0,
                io_concurrency: 2.0,
                gpu_step_us: 200.0,
                decode_compute_passes: 2,
                transform_compute_passes: 2,
                prefetch_batches: 2,
            }],
        };

        for policy in AllocationPolicy::ALL {
            let m = run_scenario(&scenario, policy);
            assert_eq!(m.stochastic_checks_total, 32 * 3);
            assert_eq!(m.stochastic_checks_passed, m.stochastic_checks_total);
            assert!((m.stochastic_freshness_pct - 100.0).abs() < 1e-9);
            assert_eq!(m.transformed_hit_rate_pct, 0.0);
            assert_eq!(m.cross_tenant_leaks, 0);
            assert!(m.cache_peak_used_bytes <= m.cache_budget_bytes);
        }
    }

    #[test]
    fn private_jobs_have_zero_cross_tenant_leaks_and_shared_public_shares() {
        let corpus =
            CorpusStore::from_bytes(b"multi tenant isolation corpus hanzo ml ".repeat(512));
        let ds_priv1 = DatasetStore::build(
            "priv1",
            &corpus,
            Modality::LlmTokenShard { raw_bytes: 2048 },
            24,
            1,
        );
        let ds_priv2 = DatasetStore::build(
            "priv2",
            &corpus,
            Modality::LlmTokenShard { raw_bytes: 2048 },
            24,
            2,
        );
        let ds_pub = DatasetStore::build(
            "pub_shared",
            &corpus,
            Modality::LlmTokenShard { raw_bytes: 2048 },
            24,
            3,
        );
        let pub_vis = DatasetVisibility::SharedPublic {
            dataset_digest: ds_pub.dataset_digest,
        };

        let scenario = ScenarioConfig {
            name: "isolation_unit".to_string(),
            description: "verify tenant isolation + shared public sharing".to_string(),
            is_negative_regime: false,
            total_cache_bytes: 512 * 1024,
            host_cpu_cores: 2,
            seed: 1349,
            datasets: vec![ds_priv1, ds_priv2, ds_pub],
            jobs: vec![
                JobConfig {
                    job_id: 1,
                    tenant_id: 10,
                    name: "tenant10_priv".to_string(),
                    dataset_index: 0,
                    visibility: DatasetVisibility::Private {
                        tenant_id: 10,
                        job_id: 1,
                    },
                    stochastic_transform: false,
                    transform_profile: "tok_v1".to_string(),
                    epochs: 3,
                    batch_size: 8,
                    io_rtt_us: 100.0,
                    io_bw_mb_s: 100.0,
                    io_concurrency: 2.0,
                    gpu_step_us: 200.0,
                    decode_compute_passes: 1,
                    transform_compute_passes: 1,
                    prefetch_batches: 2,
                },
                JobConfig {
                    job_id: 2,
                    tenant_id: 20,
                    name: "tenant20_priv".to_string(),
                    dataset_index: 1,
                    visibility: DatasetVisibility::Private {
                        tenant_id: 20,
                        job_id: 2,
                    },
                    stochastic_transform: false,
                    transform_profile: "tok_v1".to_string(),
                    epochs: 3,
                    batch_size: 8,
                    io_rtt_us: 100.0,
                    io_bw_mb_s: 100.0,
                    io_concurrency: 2.0,
                    gpu_step_us: 200.0,
                    decode_compute_passes: 1,
                    transform_compute_passes: 1,
                    prefetch_batches: 2,
                },
                JobConfig {
                    job_id: 3,
                    tenant_id: 30,
                    name: "tenant30_pub".to_string(),
                    dataset_index: 2,
                    visibility: pub_vis,
                    stochastic_transform: false,
                    transform_profile: "tok_v1".to_string(),
                    epochs: 3,
                    batch_size: 8,
                    io_rtt_us: 100.0,
                    io_bw_mb_s: 100.0,
                    io_concurrency: 2.0,
                    gpu_step_us: 200.0,
                    decode_compute_passes: 1,
                    transform_compute_passes: 1,
                    prefetch_batches: 2,
                },
                JobConfig {
                    job_id: 4,
                    tenant_id: 40,
                    name: "tenant40_pub".to_string(),
                    dataset_index: 2,
                    visibility: pub_vis,
                    stochastic_transform: false,
                    transform_profile: "tok_v1".to_string(),
                    epochs: 3,
                    batch_size: 8,
                    io_rtt_us: 100.0,
                    io_bw_mb_s: 100.0,
                    io_concurrency: 2.0,
                    gpu_step_us: 200.0,
                    decode_compute_passes: 1,
                    transform_compute_passes: 1,
                    prefetch_batches: 2,
                },
            ],
        };

        let m = run_scenario(&scenario, AllocationPolicy::SenecaDynamic);
        assert_eq!(m.cross_tenant_leaks, 0);
        assert!(m.unauthorized_probes_blocked > 0);
        assert!(
            m.overall_cache_hit_rate_pct > 55.0,
            "expected >55% overall hit rate across 3 epochs, got {:.2}%",
            m.overall_cache_hit_rate_pct
        );
        // Shared public jobs benefit from cross-job content-addressed sharing (fewer storage fetches
        // combined than 2 independent private jobs over the same cardinality).
        let priv_fetches = m.per_job[0].storage_fetches + m.per_job[1].storage_fetches;
        let pub_fetches = m.per_job[2].storage_fetches + m.per_job[3].storage_fetches;
        assert!(
            pub_fetches < priv_fetches,
            "expected shared public jobs ({pub_fetches} fetches) < private jobs ({priv_fetches} fetches)"
        );
        assert!(m.cache_peak_used_bytes <= m.cache_budget_bytes);
    }

    #[test]
    fn partial_batch_advances_epoch_only_once() {
        // Reproduces P1 finding: 5 samples, batch_size = 2, 4 epochs must process
        // all 20 examples (5 * 4), NOT 10 (skipping every other epoch).
        let corpus =
            CorpusStore::from_bytes(b"partial batch epoch advancement regression ".repeat(256));
        let ds = DatasetStore::build(
            "partial_batch_ds",
            &corpus,
            Modality::VisionImageShard {
                width: 16,
                height: 16,
                channels: 3,
            },
            5,
            77,
        );
        let vis = DatasetVisibility::Private {
            tenant_id: 1,
            job_id: 1,
        };
        let scenario = ScenarioConfig {
            name: "partial_batch_regression".to_string(),
            description: "5 samples, batch_size 2, 4 epochs".to_string(),
            is_negative_regime: false,
            total_cache_bytes: 256 * 1024,
            host_cpu_cores: 2,
            seed: 1349,
            datasets: vec![ds.clone()],
            jobs: vec![JobConfig {
                job_id: 1,
                tenant_id: 1,
                name: "partial_batch_job".to_string(),
                dataset_index: 0,
                visibility: vis,
                stochastic_transform: true,
                transform_profile: "rand_aug_v1".to_string(),
                epochs: 4,
                batch_size: 2,
                io_rtt_us: 50.0,
                io_bw_mb_s: 100.0,
                io_concurrency: 2.0,
                gpu_step_us: 100.0,
                decode_compute_passes: 1,
                transform_compute_passes: 1,
                prefetch_batches: 2,
            }],
        };

        for policy in AllocationPolicy::ALL {
            let m = run_scenario(&scenario, policy);
            assert_eq!(
                m.total_examples,
                20,
                "policy={} skipped epochs on partial batch: expected 20 examples, got {}",
                policy.as_str(),
                m.total_examples
            );
            assert_eq!(m.per_job[0].examples_processed, 20);
            assert_eq!(
                m.stochastic_checks_total, 15,
                "expected 5 samples * 3 post-epoch-0 epochs = 15 checks"
            );
            assert_eq!(m.stochastic_checks_passed, 15);
            assert!((m.stochastic_freshness_pct - 100.0).abs() < 1e-9);
        }

        // Also verify batch_size > dataset_len (5 samples, batch_size 8, 3 epochs -> 15 examples).
        let mut big_batch_scenario = scenario;
        big_batch_scenario.jobs[0].batch_size = 8;
        big_batch_scenario.jobs[0].epochs = 3;
        let m_big = run_scenario(&big_batch_scenario, AllocationPolicy::SenecaDynamic);
        assert_eq!(m_big.total_examples, 15);
        assert_eq!(m_big.stochastic_checks_total, 10);
        assert_eq!(m_big.stochastic_checks_passed, 10);
    }

    #[test]
    fn single_cpu_bound_job_has_no_quadratic_self_contention() {
        // Reproduces P1 finding: a single CPU-bound job on 1 core must NOT be charged
        // contention with its own CPU work. Previously, reducing gpu_step_us by 10x
        // caused active_cpu_rate to jump 10x and inflated simulated makespan by ~10x
        // despite identical CPU work.
        let corpus =
            CorpusStore::from_bytes(b"single job linear cpu contention regression ".repeat(256));
        let ds = DatasetStore::build(
            "cpu_linear_ds",
            &corpus,
            Modality::VisionImageShard {
                width: 24,
                height: 24,
                channels: 3,
            },
            16,
            88,
        );
        let vis = DatasetVisibility::Private {
            tenant_id: 1,
            job_id: 1,
        };
        let make_sc = |gpu_step_us: f64| ScenarioConfig {
            name: "single_cpu_job".to_string(),
            description: "verify linear single-job CPU timing".to_string(),
            is_negative_regime: false,
            total_cache_bytes: 0,
            host_cpu_cores: 1,
            seed: 1349,
            datasets: vec![ds.clone()],
            jobs: vec![JobConfig {
                job_id: 1,
                tenant_id: 1,
                name: "solo_cpu_heavy".to_string(),
                dataset_index: 0,
                visibility: vis,
                stochastic_transform: false,
                transform_profile: "norm_v1".to_string(),
                epochs: 2,
                batch_size: 4,
                io_rtt_us: 0.0,
                io_bw_mb_s: 100_000.0,
                io_concurrency: 1.0,
                gpu_step_us,
                decode_compute_passes: 60,
                transform_compute_passes: 10,
                prefetch_batches: 1,
            }],
        };

        let m_fast_gpu = run_scenario(&make_sc(10.0), AllocationPolicy::NoCache);
        let m_slow_gpu = run_scenario(&make_sc(100.0), AllocationPolicy::NoCache);

        // With no self-contention, a faster GPU (10 us vs 100 us) on the exact same
        // CPU-bound workload cannot make simulated makespan jump 5x-10x above wall time.
        assert!(
            m_fast_gpu.makespan_ms <= m_fast_gpu.wall_time_ms * 2.5 + 2.0,
            "simulated makespan ({:.3} ms) blew up quadratically relative to measured wall_time ({:.3} ms)",
            m_fast_gpu.makespan_ms,
            m_fast_gpu.wall_time_ms
        );
        assert!(
            m_fast_gpu.makespan_ms <= m_slow_gpu.makespan_ms * 2.5 + 2.0,
            "10x faster GPU step should not inflate makespan: fast_gpu={:.3} ms vs slow_gpu={:.3} ms",
            m_fast_gpu.makespan_ms,
            m_slow_gpu.makespan_ms
        );
    }

    #[test]
    fn shared_public_jobs_with_different_decode_passes_never_cross_serve_decoded_cache() {
        // Reproduces P2 finding: two SharedPublic jobs on the same dataset with
        // decode_compute_passes = 1 vs 2 produce incompatible decoded vision tensors
        // and must NEVER share Decoded or Transformed cache entries.
        let corpus =
            CorpusStore::from_bytes(b"shared public decoder config isolation test ".repeat(256));
        let modality = Modality::VisionImageShard {
            width: 24,
            height: 24,
            channels: 3,
        };
        let ds = DatasetStore::build("shared_vis_dec_cfg", &corpus, modality, 16, 99);

        // Confirm that 1 vs 2 decode passes genuinely produce different decoded bytes.
        let dec1 = execute_decode(&ds.shards[0].bytes, modality, 1).unwrap();
        let dec2 = execute_decode(&ds.shards[0].bytes, modality, 2).unwrap();
        assert_ne!(
            dec1, dec2,
            "1-pass and 2-pass vision decode must produce distinct decoded tensors"
        );
        let dh1 = decode_profile_hash(modality, 1);
        let dh2 = decode_profile_hash(modality, 2);
        assert_ne!(dh1, dh2);
        assert_ne!(
            transform_profile_hash(dh1, "same_profile", 1),
            transform_profile_hash(dh2, "same_profile", 1)
        );

        let vis = DatasetVisibility::SharedPublic {
            dataset_digest: ds.dataset_digest,
        };
        let make_pair = |passes_job2: u32| ScenarioConfig {
            name: "shared_decode_cfg_check".to_string(),
            description: "verify decoded cache keys include decoder configuration".to_string(),
            is_negative_regime: false,
            total_cache_bytes: 4 * 1024 * 1024,
            host_cpu_cores: 2,
            seed: 1349,
            datasets: vec![ds.clone()],
            jobs: vec![
                JobConfig {
                    job_id: 1,
                    tenant_id: 10,
                    name: "job_decode_1pass".to_string(),
                    dataset_index: 0,
                    visibility: vis,
                    stochastic_transform: false,
                    transform_profile: "same_profile".to_string(),
                    epochs: 1,
                    batch_size: 4,
                    io_rtt_us: 100.0,
                    io_bw_mb_s: 100.0,
                    io_concurrency: 2.0,
                    gpu_step_us: 200.0,
                    decode_compute_passes: 1,
                    transform_compute_passes: 1,
                    prefetch_batches: 2,
                },
                JobConfig {
                    job_id: 2,
                    tenant_id: 20,
                    name: "job_decode_other".to_string(),
                    dataset_index: 0,
                    visibility: vis,
                    stochastic_transform: false,
                    transform_profile: "same_profile".to_string(),
                    epochs: 1,
                    batch_size: 4,
                    io_rtt_us: 100.0,
                    io_bw_mb_s: 100.0,
                    io_concurrency: 2.0,
                    gpu_step_us: 200.0,
                    decode_compute_passes: passes_job2,
                    transform_compute_passes: 1,
                    prefetch_batches: 2,
                },
            ],
        };

        // Incompatible decode passes (1 vs 2): MUST have 0 Decoded and 0 Transformed hits across 1 epoch!
        let sc_diff = make_pair(2);
        let m_dec_diff = run_scenario(&sc_diff, AllocationPolicy::DecodedOnly);
        assert_eq!(
            m_dec_diff.per_job[0].decoded_hits + m_dec_diff.per_job[1].decoded_hits,
            0,
            "jobs with different decode_compute_passes must NEVER cross-hit Decoded cache"
        );
        let m_eq_diff = run_scenario(&sc_diff, AllocationPolicy::StaticEqualSplit);
        assert_eq!(
            m_eq_diff.per_job[0].decoded_hits + m_eq_diff.per_job[1].decoded_hits,
            0,
            "StaticEqualSplit must NEVER cross-hit Decoded cache across different decode configs"
        );
        assert_eq!(
            m_eq_diff.per_job[0].transformed_hits + m_eq_diff.per_job[1].transformed_hits,
            0,
            "StaticEqualSplit must NEVER cross-hit Transformed cache across different decode configs"
        );
        // Encoded compressed shards ARE identical and safely shared across both jobs.
        let m_enc_diff = run_scenario(&sc_diff, AllocationPolicy::EncodedOnly);
        assert!(
            m_enc_diff.per_job[0].encoded_hits + m_enc_diff.per_job[1].encoded_hits > 0,
            "Encoded compressed bytes should still be safely shared across SharedPublic jobs"
        );

        // Compatible decode passes (1 vs 1): safely share Decoded cache entries!
        let sc_same = make_pair(1);
        let m_dec_same = run_scenario(&sc_same, AllocationPolicy::DecodedOnly);
        assert!(
            m_dec_same.per_job[0].decoded_hits + m_dec_same.per_job[1].decoded_hits > 0,
            "SharedPublic jobs with identical decoder configuration should safely share Decoded entries"
        );
    }
}
