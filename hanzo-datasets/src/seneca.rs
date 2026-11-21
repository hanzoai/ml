//! Seneca: dynamic marginal-utility host RAM cache partitioning across
//! encoded, decoded, and transformed data-pipeline tiers for concurrent
//! training jobs (`HIP-1348`).
//!
//! Clean-room implementation inspired by Seneca (Desai et al., FAST '26):
//!
//! 1. **Three-tier data-pipeline hierarchy**:
//!    - [`CacheTier::Encoded`]: compressed source records/images from VFS/S3;
//!      hits eliminate storage/network I/O wait.
//!    - [`CacheTier::Decoded`]: decompressed pixels or parsed token structs;
//!      hits eliminate both I/O wait and CPU decode work.
//!    - [`CacheTier::Transformed`]: augmented/normalized tensor batches; hits
//!      eliminate the entire host preprocessing pipeline when augmentations are
//!      deterministic/reusable ([`JobWorkload::transform_reusable`]).
//! 2. **Marginal-utility & anti-starvation partitioning ([`SenecaPartitioner`])**:
//!    - Models host data-supply rate vs GPU compute step time per job so RAM is
//!      directed where it directly eliminates **GPU idle time**, avoiding RAM
//!      waste on already GPU-saturated jobs or non-reusable stochastic transforms.
//!    - Combines proportional-fair GPU-utilization recovery (`ln(util)`) with an
//!      anti-starvation floor so high-rate or large-tensor neighbors cannot starve
//!      I/O- or decode-bound concurrent jobs.
//! 3. **Byte-enforced [`TieredHostCache`] & multi-job simulation**:
//!    - Enforces per-`(job, tier)` byte quotas with deterministic LRU eviction on
//!      rebalance and measures hits, misses, GPU idle percentage, examples/sec,
//!      and starvation elimination across static, greedy-LRU, and Seneca policies.

use std::collections::{BTreeMap, HashMap, VecDeque};
use thiserror::Error;

/// Errors returned by Seneca partitioning and tiered cache operations.
#[derive(Debug, Error)]
pub enum SenecaError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("duplicate job id '{0}'")]
    DuplicateJob(String),
    #[error("unknown job id '{0}'")]
    UnknownJob(String),
}

pub type Result<T> = std::result::Result<T, SenecaError>;

/// Host RAM data-pipeline cache tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CacheTier {
    /// Compressed bytes read from storage/object store (saves I/O wait).
    Encoded,
    /// Decoded sample buffer in host RAM (saves I/O + CPU decode).
    Decoded,
    /// Fully transformed/augmented batch tensor (saves I/O + decode + transform).
    Transformed,
}

/// Pipeline cost, footprint, and GPU compute profile for one concurrent training job.
#[derive(Debug, Clone)]
pub struct JobWorkload {
    pub job_id: String,
    /// Distinct sample/batch items in the job's working set.
    pub working_set_items: usize,
    /// Examples consumed per GPU training step.
    pub batch_size: usize,
    /// Byte size of one item in the encoded tier.
    pub encoded_item_bytes: usize,
    /// Byte size of one item in the decoded tier.
    pub decoded_item_bytes: usize,
    /// Byte size of one item in the transformed tier.
    pub transformed_item_bytes: usize,
    /// Storage/VFS/S3 read latency per item on cache miss (ms).
    pub io_wait_ms: f64,
    /// CPU decode time per item (ms).
    pub decode_cpu_ms: f64,
    /// CPU transform/augmentation time per item (ms).
    pub transform_cpu_ms: f64,
    /// Whether transformed outputs are reusable across epochs (`false` for stochastic aug).
    pub transform_reusable: bool,
    /// GPU compute duration per batch step (ms).
    pub gpu_step_ms: f64,
    /// Concurrent host CPU data-loader worker threads assigned to this job.
    pub loader_workers: usize,
    /// Zipf access skew (`0.0` = uniform random epoch pass, `> 0.0` = skewed hot set).
    pub skew_alpha: f64,
    /// Relative scheduling weight for this job (default `1.0`).
    pub priority_weight: f64,
}

impl JobWorkload {
    /// Validate workload parameters and stage latencies.
    pub fn validate(&self) -> Result<()> {
        if self.job_id.is_empty() {
            return Err(SenecaError::InvalidConfig(
                "job_id must not be empty".to_string(),
            ));
        }
        if self.working_set_items == 0 || self.batch_size == 0 {
            return Err(SenecaError::InvalidConfig(format!(
                "job {} working_set_items and batch_size must be > 0",
                self.job_id
            )));
        }
        if self.encoded_item_bytes == 0
            || self.decoded_item_bytes == 0
            || self.transformed_item_bytes == 0
        {
            return Err(SenecaError::InvalidConfig(format!(
                "job {} tier item byte sizes must be > 0",
                self.job_id
            )));
        }
        for (label, val) in [
            ("io_wait_ms", self.io_wait_ms),
            ("decode_cpu_ms", self.decode_cpu_ms),
            ("transform_cpu_ms", self.transform_cpu_ms),
            ("gpu_step_ms", self.gpu_step_ms),
            ("skew_alpha", self.skew_alpha),
            ("priority_weight", self.priority_weight),
        ] {
            if !val.is_finite() || val < 0.0 {
                return Err(SenecaError::InvalidConfig(format!(
                    "job {} {label} must be finite and >= 0, got {val}",
                    self.job_id
                )));
            }
        }
        if self.gpu_step_ms <= 0.0 {
            return Err(SenecaError::InvalidConfig(format!(
                "job {} gpu_step_ms must be > 0",
                self.job_id
            )));
        }
        if self.priority_weight <= 0.0 {
            return Err(SenecaError::InvalidConfig(format!(
                "job {} priority_weight must be > 0",
                self.job_id
            )));
        }
        Ok(())
    }

    fn access_cdf(&self) -> Vec<f64> {
        let n = self.working_set_items;
        let mut weights = Vec::with_capacity(n);
        if self.skew_alpha <= 0.0 {
            let w = 1.0 / n as f64;
            weights.resize(n, w);
        } else {
            let mut sum = 0.0;
            for rank in 0..n {
                let w = 1.0 / ((rank + 1) as f64).powf(self.skew_alpha);
                weights.push(w);
                sum += w;
            }
            for w in &mut weights {
                *w /= sum;
            }
        }
        let mut cdf = Vec::with_capacity(n + 1);
        cdf.push(0.0);
        let mut acc = 0.0;
        for w in weights {
            acc += w;
            cdf.push(acc.min(1.0));
        }
        if let Some(last) = cdf.last_mut() {
            *last = 1.0;
        }
        cdf
    }
}

/// Configuration for [`SenecaPartitioner`].
#[derive(Debug, Clone, PartialEq)]
pub struct SenecaConfig {
    /// Total host RAM budget shared across all jobs and cache tiers (bytes).
    pub total_ram_bytes: usize,
    /// Allocation quantum in bytes used during marginal-utility water-filling.
    pub slab_bytes: usize,
    /// Fraction of `total_ram_bytes` reserved in Stage 1 to prevent job starvation (`0.0..=0.5`).
    pub starvation_guard_ratio: f64,
    /// Weight on proportional-fairness `ln(gpu_utilization)` vs raw utilization recovery.
    pub fairness_weight: f64,
}

impl Default for SenecaConfig {
    fn default() -> Self {
        Self {
            total_ram_bytes: 64 * 1024 * 1024,
            slab_bytes: 256 * 1024,
            starvation_guard_ratio: 0.15,
            fairness_weight: 0.35,
        }
    }
}

impl SenecaConfig {
    fn validate(&self) -> Result<()> {
        if self.slab_bytes == 0 {
            return Err(SenecaError::InvalidConfig(
                "slab_bytes must be > 0".to_string(),
            ));
        }
        if !self.starvation_guard_ratio.is_finite()
            || !(0.0..=0.5).contains(&self.starvation_guard_ratio)
        {
            return Err(SenecaError::InvalidConfig(format!(
                "starvation_guard_ratio must be in [0.0, 0.5], got {}",
                self.starvation_guard_ratio
            )));
        }
        if !self.fairness_weight.is_finite() || self.fairness_weight < 0.0 {
            return Err(SenecaError::InvalidConfig(format!(
                "fairness_weight must be >= 0, got {}",
                self.fairness_weight
            )));
        }
        Ok(())
    }
}

/// Byte quota assigned to one job across the three data-pipeline cache tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TierQuota {
    pub encoded_bytes: usize,
    pub decoded_bytes: usize,
    pub transformed_bytes: usize,
}

impl TierQuota {
    /// Total host RAM assigned to this job across all three tiers.
    pub fn total_bytes(&self) -> usize {
        self.encoded_bytes + self.decoded_bytes + self.transformed_bytes
    }

    /// Quota for a specific [`CacheTier`].
    pub fn for_tier(&self, tier: CacheTier) -> usize {
        match tier {
            CacheTier::Encoded => self.encoded_bytes,
            CacheTier::Decoded => self.decoded_bytes,
            CacheTier::Transformed => self.transformed_bytes,
        }
    }
}

/// Per-job allocation and predicted GPU/pipeline metrics inside [`AllocationPlan`].
#[derive(Debug, Clone, PartialEq)]
pub struct JobAllocationSummary {
    pub job_id: String,
    pub quota: TierQuota,
    pub cached_encoded_items: usize,
    pub cached_decoded_items: usize,
    pub cached_transformed_items: usize,
    pub predicted_hit_rate_encoded: f64,
    pub predicted_hit_rate_decoded: f64,
    pub predicted_hit_rate_transformed: f64,
    pub predicted_host_batch_ms: f64,
    pub predicted_gpu_idle_fraction: f64,
    pub predicted_examples_per_sec: f64,
}

/// Full cluster host-RAM partition computed by [`SenecaPartitioner`].
#[derive(Debug, Clone, PartialEq)]
pub struct AllocationPlan {
    pub total_ram_bytes: usize,
    pub allocated_bytes: usize,
    pub jobs: BTreeMap<String, JobAllocationSummary>,
    pub mean_gpu_idle_fraction: f64,
    pub max_gpu_idle_fraction: f64,
    pub total_examples_per_sec: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct ItemCounts {
    enc: usize,
    dec: usize,
    xform: usize,
}

impl ItemCounts {
    fn total_items(&self) -> usize {
        self.enc + self.dec + self.xform
    }
}

#[derive(Debug, Clone)]
struct JobState {
    workload: JobWorkload,
    cdf: Vec<f64>,
    counts: ItemCounts,
}

#[derive(Debug, Clone, Copy)]
struct StepStats {
    p_xform: f64,
    p_dec: f64,
    p_enc: f64,
    host_batch_ms: f64,
    gpu_idle_fraction: f64,
    gpu_utilization: f64,
    examples_per_sec: f64,
}

struct MoveCtx {
    cur_u: f64,
    remaining_ram: usize,
    slab_bytes: usize,
    fairness_weight: f64,
}

impl JobState {
    fn new(workload: JobWorkload) -> Self {
        let cdf = workload.access_cdf();
        Self {
            workload,
            cdf,
            counts: ItemCounts::default(),
        }
    }

    fn eval(&self, c: ItemCounts) -> StepStats {
        let w = &self.workload;
        let n = w.working_set_items;
        let x = c.xform.min(n);
        let d = c.dec.min(n.saturating_sub(x));
        let e = c.enc.min(n.saturating_sub(x + d));

        let p_xform = self.cdf[x];
        let p_dec = self.cdf[x + d] - self.cdf[x];
        let p_enc = self.cdf[x + d + e] - self.cdf[x + d];
        let p_miss = (1.0 - self.cdf[x + d + e]).max(0.0);

        let item_ms = p_dec * w.transform_cpu_ms
            + p_enc * (w.decode_cpu_ms + w.transform_cpu_ms)
            + p_miss * (w.io_wait_ms + w.decode_cpu_ms + w.transform_cpu_ms);

        let workers = w.loader_workers.max(1) as f64;
        let host_batch_ms = (w.batch_size as f64 * item_ms) / workers;
        let step_ms = host_batch_ms.max(w.gpu_step_ms);
        let gpu_idle_fraction = (host_batch_ms - w.gpu_step_ms).max(0.0) / step_ms;
        let gpu_utilization = w.gpu_step_ms / step_ms;
        let examples_per_sec = (w.batch_size as f64 * 1000.0) / step_ms;

        StepStats {
            p_xform,
            p_dec,
            p_enc,
            host_batch_ms,
            gpu_idle_fraction,
            gpu_utilization,
            examples_per_sec,
        }
    }

    fn utility(&self, stats: StepStats, fairness_weight: f64) -> f64 {
        let util_term = stats.gpu_utilization
            + fairness_weight * stats.gpu_utilization.max(1e-6).ln()
            - 1e-5 * stats.host_batch_ms;
        self.workload.priority_weight * util_term
    }

    fn push_add_move(
        &self,
        moves: &mut Vec<(ItemCounts, usize, f64)>,
        ctx: &MoveCtx,
        item_bytes: usize,
        next: impl FnOnce(usize) -> ItemCounts,
    ) {
        let free_items = self
            .workload
            .working_set_items
            .saturating_sub(self.counts.total_items());
        if free_items == 0 || item_bytes > ctx.remaining_ram {
            return;
        }
        let max_items = (ctx.remaining_ram / item_bytes).min(free_items);
        let k = (ctx.slab_bytes / item_bytes).max(1).min(max_items);
        let cost = k * item_bytes;
        let cand = next(k);
        let du = self.utility(self.eval(cand), ctx.fairness_weight) - ctx.cur_u;
        if du > 0.0 {
            moves.push((cand, cost, du / cost as f64));
        }
    }

    fn push_promote_move(
        &self,
        moves: &mut Vec<(ItemCounts, usize, f64)>,
        ctx: &MoveCtx,
        src_count: usize,
        src_bytes: usize,
        dst_bytes: usize,
        next: impl FnOnce(usize) -> ItemCounts,
    ) {
        if src_count == 0 {
            return;
        }
        let extra = dst_bytes.saturating_sub(src_bytes);
        if extra > ctx.remaining_ram {
            return;
        }
        let k = if extra == 0 {
            src_count
        } else {
            let max_items = (ctx.remaining_ram / extra).min(src_count);
            (ctx.slab_bytes / extra).max(1).min(max_items)
        };
        let cost = k * extra;
        let cand = next(k);
        let du = self.utility(self.eval(cand), ctx.fairness_weight) - ctx.cur_u;
        if du > 0.0 {
            let rate = if cost == 0 {
                f64::INFINITY
            } else {
                du / cost as f64
            };
            moves.push((cand, cost, rate));
        }
    }

    fn candidate_moves(
        &self,
        remaining_ram: usize,
        slab_bytes: usize,
        fairness_weight: f64,
    ) -> Vec<(ItemCounts, usize, f64)> {
        let w = &self.workload;
        let cur = self.counts;
        let ctx = MoveCtx {
            cur_u: self.utility(self.eval(cur), fairness_weight),
            remaining_ram,
            slab_bytes,
            fairness_weight,
        };
        let mut moves = Vec::with_capacity(5);

        self.push_add_move(&mut moves, &ctx, w.encoded_item_bytes, |k| ItemCounts {
            enc: cur.enc + k,
            ..cur
        });
        self.push_add_move(&mut moves, &ctx, w.decoded_item_bytes, |k| ItemCounts {
            dec: cur.dec + k,
            ..cur
        });
        self.push_promote_move(
            &mut moves,
            &ctx,
            cur.enc,
            w.encoded_item_bytes,
            w.decoded_item_bytes,
            |k| ItemCounts {
                enc: cur.enc - k,
                dec: cur.dec + k,
                xform: cur.xform,
            },
        );
        if w.transform_reusable {
            self.push_add_move(&mut moves, &ctx, w.transformed_item_bytes, |k| ItemCounts {
                xform: cur.xform + k,
                ..cur
            });
            self.push_promote_move(
                &mut moves,
                &ctx,
                cur.dec,
                w.decoded_item_bytes,
                w.transformed_item_bytes,
                |k| ItemCounts {
                    enc: cur.enc,
                    dec: cur.dec - k,
                    xform: cur.xform + k,
                },
            );
        }

        moves
    }
}

/// Dynamic marginal-utility host RAM partitioner across concurrent training jobs.
#[derive(Debug, Clone)]
pub struct SenecaPartitioner {
    config: SenecaConfig,
    jobs: BTreeMap<String, JobWorkload>,
}

impl SenecaPartitioner {
    /// Construct a partitioner with `config`.
    pub fn new(config: SenecaConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            jobs: BTreeMap::new(),
        })
    }

    /// Update the total host RAM budget (e.g., when co-resident system pressure changes).
    pub fn set_total_ram_bytes(&mut self, total_ram_bytes: usize) {
        self.config.total_ram_bytes = total_ram_bytes;
    }

    /// Register a concurrent training job.
    pub fn register_job(&mut self, job: JobWorkload) -> Result<()> {
        job.validate()?;
        if self.jobs.contains_key(&job.job_id) {
            return Err(SenecaError::DuplicateJob(job.job_id));
        }
        self.jobs.insert(job.job_id.clone(), job);
        Ok(())
    }

    /// Remove a finished or preempted training job so its cache shares are reclaimed immediately.
    pub fn unregister_job(&mut self, job_id: &str) -> Result<JobWorkload> {
        self.jobs
            .remove(job_id)
            .ok_or_else(|| SenecaError::UnknownJob(job_id.to_string()))
    }

    /// Update live observed latencies/skews for a running job before calling [`Self::rebalance`].
    pub fn update_job(&mut self, job: JobWorkload) -> Result<()> {
        job.validate()?;
        if !self.jobs.contains_key(&job.job_id) {
            return Err(SenecaError::UnknownJob(job.job_id));
        }
        self.jobs.insert(job.job_id.clone(), job);
        Ok(())
    }

    /// Compute the marginal-utility optimal byte partition across all registered jobs and tiers.
    pub fn rebalance(&self) -> AllocationPlan {
        let total_ram = self.config.total_ram_bytes;
        if self.jobs.is_empty() || total_ram == 0 {
            return AllocationPlan {
                total_ram_bytes: total_ram,
                allocated_bytes: 0,
                jobs: BTreeMap::new(),
                mean_gpu_idle_fraction: 0.0,
                max_gpu_idle_fraction: 0.0,
                total_examples_per_sec: 0.0,
            };
        }

        let mut states: Vec<JobState> = self.jobs.values().cloned().map(JobState::new).collect();
        let mut remaining = total_ram;

        // Stage 1: Anti-starvation reservation ensures every stalled job receives a small
        // bootstrap footprint in its lowest-cost bottleneck tier before global competition.
        let guard_budget = ((total_ram as f64) * self.config.starvation_guard_ratio) as usize;
        let stalled_count = states
            .iter()
            .filter(|s| s.eval(s.counts).gpu_idle_fraction > 0.0)
            .count();

        if guard_budget > 0 && stalled_count > 0 {
            let per_job_floor = (guard_budget / stalled_count).max(self.config.slab_bytes);
            for st in &mut states {
                let stats = st.eval(st.counts);
                if stats.gpu_idle_fraction <= 0.0 {
                    continue;
                }
                let budget = per_job_floor.min(remaining);
                // Pick the highest utility-per-byte starter tier for this job.
                let unit_enc = st.workload.encoded_item_bytes;
                let unit_dec = st.workload.decoded_item_bytes;
                let enc_eff = if st.workload.io_wait_ms > 0.0 {
                    st.workload.io_wait_ms / unit_enc as f64
                } else {
                    0.0
                };
                let dec_eff =
                    (st.workload.io_wait_ms + st.workload.decode_cpu_ms) / unit_dec as f64;
                if enc_eff >= dec_eff && unit_enc <= budget {
                    let k = (budget / unit_enc).min(st.workload.working_set_items);
                    st.counts.enc += k;
                    remaining -= k * unit_enc;
                } else if unit_dec <= budget {
                    let k = (budget / unit_dec).min(st.workload.working_set_items);
                    st.counts.dec += k;
                    remaining -= k * unit_dec;
                }
            }
        }

        // Stage 2: Marginal-utility greedy water-filling across all jobs and tier actions.
        loop {
            if remaining == 0 {
                break;
            }
            let mut best: Option<(usize, ItemCounts, usize, f64)> = None;
            for (idx, st) in states.iter().enumerate() {
                for (next_counts, cost, score) in st.candidate_moves(
                    remaining,
                    self.config.slab_bytes,
                    self.config.fairness_weight,
                ) {
                    let replace = match best {
                        None => true,
                        Some((_, _, _, best_score)) => score > best_score,
                    };
                    if replace {
                        best = Some((idx, next_counts, cost, score));
                    }
                }
            }
            match best {
                Some((idx, next_counts, cost, _)) => {
                    states[idx].counts = next_counts;
                    remaining = remaining.saturating_sub(cost);
                }
                None => break,
            }
        }

        let mut jobs_out = BTreeMap::new();
        let mut allocated_bytes = 0usize;
        let mut idle_sum = 0.0;
        let mut idle_max = 0.0f64;
        let mut total_eps = 0.0;

        for st in &states {
            let w = &st.workload;
            let c = st.counts;
            let quota = TierQuota {
                encoded_bytes: c.enc * w.encoded_item_bytes,
                decoded_bytes: c.dec * w.decoded_item_bytes,
                transformed_bytes: c.xform * w.transformed_item_bytes,
            };
            allocated_bytes += quota.total_bytes();
            let stats = st.eval(c);
            idle_sum += stats.gpu_idle_fraction;
            idle_max = idle_max.max(stats.gpu_idle_fraction);
            total_eps += stats.examples_per_sec;

            jobs_out.insert(
                w.job_id.clone(),
                JobAllocationSummary {
                    job_id: w.job_id.clone(),
                    quota,
                    cached_encoded_items: c.enc,
                    cached_decoded_items: c.dec,
                    cached_transformed_items: c.xform,
                    predicted_hit_rate_encoded: stats.p_enc,
                    predicted_hit_rate_decoded: stats.p_dec,
                    predicted_hit_rate_transformed: stats.p_xform,
                    predicted_host_batch_ms: stats.host_batch_ms,
                    predicted_gpu_idle_fraction: stats.gpu_idle_fraction,
                    predicted_examples_per_sec: stats.examples_per_sec,
                },
            );
        }

        let n_jobs = states.len().max(1) as f64;
        AllocationPlan {
            total_ram_bytes: total_ram,
            allocated_bytes,
            jobs: jobs_out,
            mean_gpu_idle_fraction: idle_sum / n_jobs,
            max_gpu_idle_fraction: idle_max,
            total_examples_per_sec: total_eps,
        }
    }
}

/// Static equal host-RAM split baseline (divides RAM equally across jobs and active tiers).
pub fn static_equal_partition(total_ram_bytes: usize, jobs: &[JobWorkload]) -> AllocationPlan {
    let mut part = SenecaPartitioner::new(SenecaConfig {
        total_ram_bytes,
        ..SenecaConfig::default()
    })
    .expect("default config valid");
    for j in jobs {
        let _ = part.register_job(j.clone());
    }
    if jobs.is_empty() {
        return part.rebalance();
    }
    let per_job = total_ram_bytes / jobs.len();
    let mut out = BTreeMap::new();
    let mut allocated = 0usize;
    let mut idle_sum = 0.0;
    let mut idle_max = 0.0f64;
    let mut total_eps = 0.0;

    for w in jobs {
        let st = JobState::new(w.clone());
        let active_tiers = if w.transform_reusable { 3 } else { 2 };
        let per_tier = per_job / active_tiers;
        let mut remaining_items = w.working_set_items;

        let xform = if w.transform_reusable {
            let k = (per_tier / w.transformed_item_bytes).min(remaining_items);
            remaining_items -= k;
            k
        } else {
            0
        };
        let dec = (per_tier / w.decoded_item_bytes).min(remaining_items);
        remaining_items -= dec;
        let enc = (per_tier / w.encoded_item_bytes).min(remaining_items);

        let c = ItemCounts { enc, dec, xform };
        let quota = TierQuota {
            encoded_bytes: enc * w.encoded_item_bytes,
            decoded_bytes: dec * w.decoded_item_bytes,
            transformed_bytes: xform * w.transformed_item_bytes,
        };
        allocated += quota.total_bytes();
        let stats = st.eval(c);
        idle_sum += stats.gpu_idle_fraction;
        idle_max = idle_max.max(stats.gpu_idle_fraction);
        total_eps += stats.examples_per_sec;

        out.insert(
            w.job_id.clone(),
            JobAllocationSummary {
                job_id: w.job_id.clone(),
                quota,
                cached_encoded_items: enc,
                cached_decoded_items: dec,
                cached_transformed_items: xform,
                predicted_hit_rate_encoded: stats.p_enc,
                predicted_hit_rate_decoded: stats.p_dec,
                predicted_hit_rate_transformed: stats.p_xform,
                predicted_host_batch_ms: stats.host_batch_ms,
                predicted_gpu_idle_fraction: stats.gpu_idle_fraction,
                predicted_examples_per_sec: stats.examples_per_sec,
            },
        );
    }

    AllocationPlan {
        total_ram_bytes,
        allocated_bytes: allocated,
        jobs: out,
        mean_gpu_idle_fraction: idle_sum / jobs.len() as f64,
        max_gpu_idle_fraction: idle_max,
        total_examples_per_sec: total_eps,
    }
}

#[derive(Debug)]
struct PartitionLru {
    quota_bytes: usize,
    used_bytes: usize,
    entries: HashMap<usize, Vec<u8>>,
    lru: VecDeque<usize>,
}

impl PartitionLru {
    fn new(quota_bytes: usize) -> Self {
        Self {
            quota_bytes,
            used_bytes: 0,
            entries: HashMap::new(),
            lru: VecDeque::new(),
        }
    }

    fn touch(&mut self, sample_id: usize) {
        if let Some(pos) = self.lru.iter().position(|&id| id == sample_id) {
            self.lru.remove(pos);
        }
        self.lru.push_back(sample_id);
    }

    fn evict_to_quota(&mut self) -> Vec<usize> {
        let mut evicted = Vec::new();
        while self.used_bytes > self.quota_bytes {
            let Some(victim) = self.lru.pop_front() else {
                break;
            };
            if let Some(buf) = self.entries.remove(&victim) {
                self.used_bytes = self.used_bytes.saturating_sub(buf.len());
                evicted.push(victim);
            }
        }
        evicted
    }

    fn set_quota(&mut self, quota_bytes: usize) -> usize {
        self.quota_bytes = quota_bytes;
        self.evict_to_quota().len()
    }

    fn remove(&mut self, sample_id: usize) {
        if let Some(buf) = self.entries.remove(&sample_id) {
            self.used_bytes = self.used_bytes.saturating_sub(buf.len());
            if let Some(pos) = self.lru.iter().position(|&id| id == sample_id) {
                self.lru.remove(pos);
            }
        }
    }

    fn insert(&mut self, sample_id: usize, payload: Vec<u8>) -> Vec<usize> {
        if payload.len() > self.quota_bytes {
            return Vec::new();
        }
        if let Some(prev) = self.entries.remove(&sample_id) {
            self.used_bytes = self.used_bytes.saturating_sub(prev.len());
            if let Some(pos) = self.lru.iter().position(|&id| id == sample_id) {
                self.lru.remove(pos);
            }
        }
        self.used_bytes += payload.len();
        self.entries.insert(sample_id, payload);
        self.lru.push_back(sample_id);
        self.evict_to_quota()
    }
}

/// Outcome of requesting one sample from [`TieredHostCache`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CacheAccessOutcome {
    pub hit_tier: Option<CacheTier>,
    pub host_prep_ms: f64,
    pub evictions: usize,
}

/// Byte-enforced multi-job, 3-tier host RAM cache executing a Seneca [`AllocationPlan`].
#[derive(Debug)]
pub struct TieredHostCache {
    total_ram_bytes: usize,
    partitions: HashMap<(String, CacheTier), PartitionLru>,
    hits: HashMap<(String, CacheTier), u64>,
    misses: HashMap<String, u64>,
    evictions: u64,
}

impl TieredHostCache {
    /// Construct an empty host cache bounded by `total_ram_bytes`.
    pub fn new(total_ram_bytes: usize) -> Self {
        Self {
            total_ram_bytes,
            partitions: HashMap::new(),
            hits: HashMap::new(),
            misses: HashMap::new(),
            evictions: 0,
        }
    }

    /// Apply a dynamically computed [`AllocationPlan`], resizing partitions and evicting excess LRU items.
    pub fn apply_plan(&mut self, plan: &AllocationPlan) {
        let mut evicted_total = 0usize;
        for (job_id, summary) in &plan.jobs {
            for tier in [
                CacheTier::Encoded,
                CacheTier::Decoded,
                CacheTier::Transformed,
            ] {
                let q = summary.quota.for_tier(tier);
                let part = self
                    .partitions
                    .entry((job_id.clone(), tier))
                    .or_insert_with(|| PartitionLru::new(q));
                evicted_total += part.set_quota(q);
            }
        }
        // Evict partitions belonging to jobs no longer in the plan.
        self.partitions
            .retain(|(job_id, _), _| plan.jobs.contains_key(job_id));
        self.evictions += evicted_total as u64;
    }

    /// Total host RAM bytes currently occupied across all jobs and tiers.
    pub fn total_used_bytes(&self) -> usize {
        self.partitions.values().map(|p| p.used_bytes).sum()
    }

    /// Total host RAM budget.
    pub fn total_ram_bytes(&self) -> usize {
        self.total_ram_bytes
    }

    /// Bytes currently occupied in one `(job_id, tier)` partition.
    pub fn partition_used_bytes(&self, job_id: &str, tier: CacheTier) -> usize {
        self.partitions
            .get(&(job_id.to_string(), tier))
            .map_or(0, |p| p.used_bytes)
    }

    /// Cache hits recorded for `(job_id, tier)`.
    pub fn tier_hits(&self, job_id: &str, tier: CacheTier) -> u64 {
        self.hits
            .get(&(job_id.to_string(), tier))
            .copied()
            .unwrap_or(0)
    }

    /// Total cache misses recorded for `job_id`.
    pub fn job_misses(&self, job_id: &str) -> u64 {
        self.misses.get(job_id).copied().unwrap_or(0)
    }

    /// Total LRU evictions performed across rebalances and inserts.
    pub fn total_evictions(&self) -> u64 {
        self.evictions
    }

    /// Serve `sample_id` for `job`, consulting `Transformed -> Decoded -> Encoded`:
    /// - Full misses enter the lowest active tier (`Encoded` -> `Decoded` -> `Transformed`).
    /// - Hits promote one tier up (`Encoded` -> `Decoded`, `Decoded` -> `Transformed`)
    ///   while demoting evicted hot victims down into lower active tiers if space permits.
    pub fn access_sample(&mut self, job: &JobWorkload, sample_id: usize) -> CacheAccessOutcome {
        // 1. Transformed tier hit
        if job.transform_reusable {
            if let Some(part) = self
                .partitions
                .get_mut(&(job.job_id.clone(), CacheTier::Transformed))
            {
                if part.entries.contains_key(&sample_id) {
                    part.touch(sample_id);
                    *self
                        .hits
                        .entry((job.job_id.clone(), CacheTier::Transformed))
                        .or_insert(0) += 1;
                    return CacheAccessOutcome {
                        hit_tier: Some(CacheTier::Transformed),
                        host_prep_ms: 0.0,
                        evictions: 0,
                    };
                }
            }
        }

        // 2. Decoded tier hit -> promote to Transformed if enabled & quota fits
        if let Some(part) = self
            .partitions
            .get_mut(&(job.job_id.clone(), CacheTier::Decoded))
        {
            if part.entries.contains_key(&sample_id) {
                part.touch(sample_id);
                *self
                    .hits
                    .entry((job.job_id.clone(), CacheTier::Decoded))
                    .or_insert(0) += 1;
                let ev = if job.transform_reusable {
                    self.promote_or_admit(
                        job,
                        sample_id,
                        Some(CacheTier::Decoded),
                        CacheTier::Transformed,
                    )
                } else {
                    0
                };
                return CacheAccessOutcome {
                    hit_tier: Some(CacheTier::Decoded),
                    host_prep_ms: job.transform_cpu_ms,
                    evictions: ev,
                };
            }
        }

        // 3. Encoded tier hit -> promote to Decoded (or Transformed if Decoded quota is 0)
        if let Some(part) = self
            .partitions
            .get_mut(&(job.job_id.clone(), CacheTier::Encoded))
        {
            if part.entries.contains_key(&sample_id) {
                part.touch(sample_id);
                *self
                    .hits
                    .entry((job.job_id.clone(), CacheTier::Encoded))
                    .or_insert(0) += 1;
                let ev = if job.transform_reusable && self.can_hold(job, CacheTier::Transformed) {
                    self.promote_or_admit(
                        job,
                        sample_id,
                        Some(CacheTier::Encoded),
                        CacheTier::Transformed,
                    )
                } else if self.can_hold(job, CacheTier::Decoded) {
                    self.promote_or_admit(
                        job,
                        sample_id,
                        Some(CacheTier::Encoded),
                        CacheTier::Decoded,
                    )
                } else {
                    0
                };
                return CacheAccessOutcome {
                    hit_tier: Some(CacheTier::Encoded),
                    host_prep_ms: job.decode_cpu_ms + job.transform_cpu_ms,
                    evictions: ev,
                };
            }
        }

        // 4. Full miss -> pipeline runs I/O + decode (+ transform), so admit into the
        // highest active tier with capacity; evicted items cascade down into lower tiers.
        *self.misses.entry(job.job_id.clone()).or_insert(0) += 1;
        let ev = if job.transform_reusable && self.can_hold(job, CacheTier::Transformed) {
            self.promote_or_admit(job, sample_id, None, CacheTier::Transformed)
        } else if self.can_hold(job, CacheTier::Decoded) {
            self.promote_or_admit(job, sample_id, None, CacheTier::Decoded)
        } else if self.can_hold(job, CacheTier::Encoded) {
            self.promote_or_admit(job, sample_id, None, CacheTier::Encoded)
        } else {
            0
        };
        CacheAccessOutcome {
            hit_tier: None,
            host_prep_ms: job.io_wait_ms + job.decode_cpu_ms + job.transform_cpu_ms,
            evictions: ev,
        }
    }

    fn item_bytes_for_tier(job: &JobWorkload, tier: CacheTier) -> usize {
        match tier {
            CacheTier::Encoded => job.encoded_item_bytes,
            CacheTier::Decoded => job.decoded_item_bytes,
            CacheTier::Transformed => job.transformed_item_bytes,
        }
    }

    fn can_hold(&self, job: &JobWorkload, tier: CacheTier) -> bool {
        let item_bytes = Self::item_bytes_for_tier(job, tier);
        self.partitions
            .get(&(job.job_id.clone(), tier))
            .is_some_and(|p| p.quota_bytes >= item_bytes)
    }

    fn promote_or_admit(
        &mut self,
        job: &JobWorkload,
        sample_id: usize,
        from_tier: Option<CacheTier>,
        to_tier: CacheTier,
    ) -> usize {
        if !self.can_hold(job, to_tier) {
            return 0;
        }
        if let Some(prev) = from_tier {
            if let Some(prev_part) = self.partitions.get_mut(&(job.job_id.clone(), prev)) {
                prev_part.remove(sample_id);
            }
        }
        let item_bytes = Self::item_bytes_for_tier(job, to_tier);
        let payload = vec![(sample_id & 0xff) as u8; item_bytes];
        let victims = self
            .partitions
            .get_mut(&(job.job_id.clone(), to_tier))
            .expect("checked above")
            .insert(sample_id, payload);
        let mut total_ev = victims.len();
        self.evictions += victims.len() as u64;
        total_ev += self.demote_cascade(job, to_tier, victims);
        total_ev
    }

    fn demote_cascade(
        &mut self,
        job: &JobWorkload,
        from_tier: CacheTier,
        mut victims: Vec<usize>,
    ) -> usize {
        let mut total_ev = 0usize;
        let mut cur_tier = from_tier;
        while !victims.is_empty() {
            let next_tier = match cur_tier {
                CacheTier::Transformed if self.can_hold(job, CacheTier::Decoded) => {
                    Some(CacheTier::Decoded)
                }
                CacheTier::Transformed | CacheTier::Decoded
                    if self.can_hold(job, CacheTier::Encoded) =>
                {
                    Some(CacheTier::Encoded)
                }
                _ => None,
            };
            let Some(dst_tier) = next_tier else {
                break;
            };
            let dst_bytes = Self::item_bytes_for_tier(job, dst_tier);
            let mut next_victims = Vec::new();
            let part = self
                .partitions
                .get_mut(&(job.job_id.clone(), dst_tier))
                .expect("checked above");
            for v_id in victims {
                let v_payload = vec![(v_id & 0xff) as u8; dst_bytes];
                let lower_victims = part.insert(v_id, v_payload);
                total_ev += lower_victims.len();
                self.evictions += lower_victims.len() as u64;
                next_victims.extend(lower_victims);
            }
            victims = next_victims;
            cur_tier = dst_tier;
        }
        total_ev
    }
}

/// Simulated execution metrics for one job in [`ConcurrentSimSummary`].
#[derive(Debug, Clone, PartialEq)]
pub struct JobSimResult {
    pub job_id: String,
    pub steps_run: usize,
    pub examples_processed: usize,
    pub total_wall_ms: f64,
    pub total_gpu_idle_ms: f64,
    pub gpu_idle_fraction: f64,
    pub examples_per_sec: f64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Aggregate multi-job simulation report comparing cache allocation strategies.
#[derive(Debug, Clone, PartialEq)]
pub struct ConcurrentSimSummary {
    pub policy_name: String,
    pub total_ram_bytes: usize,
    pub peak_used_bytes: usize,
    pub mean_gpu_idle_fraction: f64,
    pub max_gpu_idle_fraction: f64,
    pub min_job_gpu_utilization: f64,
    pub total_examples_per_sec: f64,
    pub jobs: BTreeMap<String, JobSimResult>,
}

/// Simulate concurrent multi-job training batches through a byte-bounded [`TieredHostCache`]
/// under `plan` using deterministic Zipf/uniform sample traces.
pub fn simulate_with_plan(
    policy_name: &str,
    jobs: &[JobWorkload],
    plan: &AllocationPlan,
    steps_per_job: usize,
    seed: u64,
) -> ConcurrentSimSummary {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let mut cache = TieredHostCache::new(plan.total_ram_bytes);
    cache.apply_plan(plan);
    let mut rng = StdRng::seed_from_u64(seed);
    let cdfs: Vec<Vec<f64>> = jobs.iter().map(JobWorkload::access_cdf).collect();

    let mut wall_ms = vec![0.0f64; jobs.len()];
    let mut idle_ms = vec![0.0f64; jobs.len()];
    let mut peak_used = 0usize;

    for _step in 0..steps_per_job {
        for (ji, job) in jobs.iter().enumerate() {
            let mut batch_item_sum_ms = 0.0;
            let cdf = &cdfs[ji];
            for _ in 0..job.batch_size {
                let u: f64 = rng.random();
                let idx = cdf
                    .partition_point(|&c| c < u)
                    .saturating_sub(1)
                    .min(job.working_set_items - 1);
                let out = cache.access_sample(job, idx);
                batch_item_sum_ms += out.host_prep_ms;
            }
            peak_used = peak_used.max(cache.total_used_bytes());
            let host_batch_ms = batch_item_sum_ms / (job.loader_workers.max(1) as f64);
            let step_total = host_batch_ms.max(job.gpu_step_ms);
            wall_ms[ji] += step_total;
            idle_ms[ji] += (host_batch_ms - job.gpu_step_ms).max(0.0);
        }
    }

    let mut job_results = BTreeMap::new();
    let mut idle_sum = 0.0;
    let mut idle_max = 0.0f64;
    let mut min_util = 1.0f64;
    let mut total_eps = 0.0;

    for (ji, job) in jobs.iter().enumerate() {
        let examples = steps_per_job * job.batch_size;
        let idle_frac = if wall_ms[ji] > 0.0 {
            idle_ms[ji] / wall_ms[ji]
        } else {
            0.0
        };
        let eps = if wall_ms[ji] > 0.0 {
            (examples as f64 * 1000.0) / wall_ms[ji]
        } else {
            0.0
        };
        let hits = cache.tier_hits(&job.job_id, CacheTier::Encoded)
            + cache.tier_hits(&job.job_id, CacheTier::Decoded)
            + cache.tier_hits(&job.job_id, CacheTier::Transformed);
        let misses = cache.job_misses(&job.job_id);

        idle_sum += idle_frac;
        idle_max = idle_max.max(idle_frac);
        min_util = min_util.min(1.0 - idle_frac);
        total_eps += eps;

        job_results.insert(
            job.job_id.clone(),
            JobSimResult {
                job_id: job.job_id.clone(),
                steps_run: steps_per_job,
                examples_processed: examples,
                total_wall_ms: wall_ms[ji],
                total_gpu_idle_ms: idle_ms[ji],
                gpu_idle_fraction: idle_frac,
                examples_per_sec: eps,
                cache_hits: hits,
                cache_misses: misses,
            },
        );
    }

    let n = jobs.len().max(1) as f64;
    ConcurrentSimSummary {
        policy_name: policy_name.to_string(),
        total_ram_bytes: plan.total_ram_bytes,
        peak_used_bytes: peak_used,
        mean_gpu_idle_fraction: idle_sum / n,
        max_gpu_idle_fraction: idle_max,
        min_job_gpu_utilization: min_util,
        total_examples_per_sec: total_eps,
        jobs: job_results,
    }
}

/// Shared global-LRU baseline: all jobs compete in one unpartitioned host-RAM pool,
/// each caching its highest reusable tier (`Transformed` or `Decoded`) and evicting globally.
pub fn simulate_global_lru(
    jobs: &[JobWorkload],
    total_ram_bytes: usize,
    steps_per_job: usize,
    seed: u64,
) -> ConcurrentSimSummary {
    use rand::{rngs::StdRng, Rng, SeedableRng};

    let mut rng = StdRng::seed_from_u64(seed);
    let cdfs: Vec<Vec<f64>> = jobs.iter().map(JobWorkload::access_cdf).collect();
    let mut entries: HashMap<(usize, usize), (CacheTier, usize)> = HashMap::new();
    let mut lru: VecDeque<(usize, usize)> = VecDeque::new();
    let mut used_bytes = 0usize;
    let mut peak_used = 0usize;
    let mut hits = vec![0u64; jobs.len()];
    let mut misses = vec![0u64; jobs.len()];
    let mut wall_ms = vec![0.0f64; jobs.len()];
    let mut idle_ms = vec![0.0f64; jobs.len()];

    for _step in 0..steps_per_job {
        for (ji, job) in jobs.iter().enumerate() {
            let mut batch_item_sum_ms = 0.0;
            let cdf = &cdfs[ji];
            for _ in 0..job.batch_size {
                let u: f64 = rng.random();
                let idx = cdf
                    .partition_point(|&c| c < u)
                    .saturating_sub(1)
                    .min(job.working_set_items - 1);
                let key = (ji, idx);
                if let Some(&(tier, _)) = entries.get(&key) {
                    hits[ji] += 1;
                    if let Some(pos) = lru.iter().position(|&k| k == key) {
                        lru.remove(pos);
                    }
                    lru.push_back(key);
                    batch_item_sum_ms += match tier {
                        CacheTier::Transformed => 0.0,
                        CacheTier::Decoded => job.transform_cpu_ms,
                        CacheTier::Encoded => job.decode_cpu_ms + job.transform_cpu_ms,
                    };
                } else {
                    misses[ji] += 1;
                    batch_item_sum_ms += job.io_wait_ms + job.decode_cpu_ms + job.transform_cpu_ms;
                    let (tier, bytes) = if job.transform_reusable {
                        (CacheTier::Transformed, job.transformed_item_bytes)
                    } else {
                        (CacheTier::Decoded, job.decoded_item_bytes)
                    };
                    if bytes <= total_ram_bytes {
                        used_bytes += bytes;
                        entries.insert(key, (tier, bytes));
                        lru.push_back(key);
                        while used_bytes > total_ram_bytes {
                            if let Some(vic) = lru.pop_front() {
                                if let Some((_, vbytes)) = entries.remove(&vic) {
                                    used_bytes = used_bytes.saturating_sub(vbytes);
                                }
                            } else {
                                break;
                            }
                        }
                    }
                }
            }
            peak_used = peak_used.max(used_bytes);
            let host_batch_ms = batch_item_sum_ms / (job.loader_workers.max(1) as f64);
            let step_total = host_batch_ms.max(job.gpu_step_ms);
            wall_ms[ji] += step_total;
            idle_ms[ji] += (host_batch_ms - job.gpu_step_ms).max(0.0);
        }
    }

    let mut job_results = BTreeMap::new();
    let mut idle_sum = 0.0;
    let mut idle_max = 0.0f64;
    let mut min_util = 1.0f64;
    let mut total_eps = 0.0;

    for (ji, job) in jobs.iter().enumerate() {
        let examples = steps_per_job * job.batch_size;
        let idle_frac = if wall_ms[ji] > 0.0 {
            idle_ms[ji] / wall_ms[ji]
        } else {
            0.0
        };
        let eps = if wall_ms[ji] > 0.0 {
            (examples as f64 * 1000.0) / wall_ms[ji]
        } else {
            0.0
        };
        idle_sum += idle_frac;
        idle_max = idle_max.max(idle_frac);
        min_util = min_util.min(1.0 - idle_frac);
        total_eps += eps;
        job_results.insert(
            job.job_id.clone(),
            JobSimResult {
                job_id: job.job_id.clone(),
                steps_run: steps_per_job,
                examples_processed: examples,
                total_wall_ms: wall_ms[ji],
                total_gpu_idle_ms: idle_ms[ji],
                gpu_idle_fraction: idle_frac,
                examples_per_sec: eps,
                cache_hits: hits[ji],
                cache_misses: misses[ji],
            },
        );
    }

    let n = jobs.len().max(1) as f64;
    ConcurrentSimSummary {
        policy_name: "global_lru".to_string(),
        total_ram_bytes,
        peak_used_bytes: peak_used,
        mean_gpu_idle_fraction: idle_sum / n,
        max_gpu_idle_fraction: idle_max,
        min_job_gpu_utilization: min_util,
        total_examples_per_sec: total_eps,
        jobs: job_results,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_concurrent_jobs() -> Vec<JobWorkload> {
        vec![
            // Job A: Vision training with slow S3/disk I/O and stochastic augmentations
            // (transform_reusable = false) -> Seneca should allocate Encoded/Decoded, 0 Transformed.
            JobWorkload {
                job_id: "vision-io-bound".to_string(),
                working_set_items: 256,
                batch_size: 8,
                encoded_item_bytes: 8 * 1024,   // 8 KiB JPEG
                decoded_item_bytes: 192 * 1024, // 192 KiB RGB
                transformed_item_bytes: 256 * 1024,
                io_wait_ms: 14.0,
                decode_cpu_ms: 3.5,
                transform_cpu_ms: 1.5,
                transform_reusable: false,
                gpu_step_ms: 14.0,
                loader_workers: 2,
                skew_alpha: 0.8,
                priority_weight: 1.0,
            },
            // Job B: Multimodal CPU-decode heavy job with reusable pre-tokenized/resized tensors.
            JobWorkload {
                job_id: "multimodal-decode-bound".to_string(),
                working_set_items: 192,
                batch_size: 8,
                encoded_item_bytes: 12 * 1024,
                decoded_item_bytes: 48 * 1024,
                transformed_item_bytes: 96 * 1024,
                io_wait_ms: 2.0,
                decode_cpu_ms: 12.0,
                transform_cpu_ms: 4.0,
                transform_reusable: true,
                gpu_step_ms: 12.0,
                loader_workers: 2,
                skew_alpha: 0.9,
                priority_weight: 1.0,
            },
            // Job C: GPU-compute bound large-tensor job that is ALREADY fast enough without cache.
            JobWorkload {
                job_id: "heavy-gpu-saturated".to_string(),
                working_set_items: 128,
                batch_size: 4,
                encoded_item_bytes: 64 * 1024,
                decoded_item_bytes: 256 * 1024,
                transformed_item_bytes: 512 * 1024,
                io_wait_ms: 1.0,
                decode_cpu_ms: 1.0,
                transform_cpu_ms: 0.5,
                transform_reusable: true,
                gpu_step_ms: 30.0,
                loader_workers: 2,
                skew_alpha: 0.0,
                priority_weight: 1.0,
            },
        ]
    }

    #[test]
    fn seneca_respects_ram_budget_and_skips_non_reusable_transformed_tier() -> Result<()> {
        let jobs = sample_concurrent_jobs();
        let ram_budget = 6 * 1024 * 1024; // 6 MiB tight shared host cache
        let mut part = SenecaPartitioner::new(SenecaConfig {
            total_ram_bytes: ram_budget,
            slab_bytes: 64 * 1024,
            starvation_guard_ratio: 0.15,
            fairness_weight: 0.4,
        })?;
        for j in &jobs {
            part.register_job(j.clone())?;
        }
        let plan = part.rebalance();
        assert!(plan.allocated_bytes <= ram_budget);

        let vision = &plan.jobs["vision-io-bound"];
        assert_eq!(
            vision.quota.transformed_bytes, 0,
            "stochastic augmentations must receive 0 Transformed cache bytes"
        );
        assert!(
            vision.quota.encoded_bytes + vision.quota.decoded_bytes > 0,
            "vision job must receive Encoded/Decoded cache"
        );

        let gpu_bound = &plan.jobs["heavy-gpu-saturated"];
        // GPU-saturated job needs almost no host RAM compared to the starved jobs.
        assert!(
            vision.quota.total_bytes() + plan.jobs["multimodal-decode-bound"].quota.total_bytes()
                > gpu_bound.quota.total_bytes() * 4
        );
        Ok(())
    }

    #[test]
    fn seneca_eliminates_starvation_and_cuts_gpu_idle_time_vs_uncached_and_static() -> Result<()> {
        let jobs = sample_concurrent_jobs();
        let ram_budget = 6 * 1024 * 1024; // 6 MiB

        // Baseline 1: Uncached (0 RAM).
        let uncached_plan = SenecaPartitioner::new(SenecaConfig {
            total_ram_bytes: 0,
            ..SenecaConfig::default()
        })?
        .rebalance();
        let mut empty_jobs_plan = AllocationPlan {
            total_ram_bytes: 0,
            allocated_bytes: 0,
            jobs: BTreeMap::new(),
            mean_gpu_idle_fraction: 0.0,
            max_gpu_idle_fraction: 0.0,
            total_examples_per_sec: 0.0,
        };
        let _ = uncached_plan;
        for j in &jobs {
            empty_jobs_plan.jobs.insert(
                j.job_id.clone(),
                JobAllocationSummary {
                    job_id: j.job_id.clone(),
                    quota: TierQuota::default(),
                    cached_encoded_items: 0,
                    cached_decoded_items: 0,
                    cached_transformed_items: 0,
                    predicted_hit_rate_encoded: 0.0,
                    predicted_hit_rate_decoded: 0.0,
                    predicted_hit_rate_transformed: 0.0,
                    predicted_host_batch_ms: 0.0,
                    predicted_gpu_idle_fraction: 0.0,
                    predicted_examples_per_sec: 0.0,
                },
            );
        }
        let sim_uncached = simulate_with_plan("uncached", &jobs, &empty_jobs_plan, 400, 42);

        // Baseline 2: Static equal split across jobs & tiers.
        let static_plan = static_equal_partition(ram_budget, &jobs);
        let sim_static = simulate_with_plan("static_equal", &jobs, &static_plan, 400, 42);

        // Seneca: Dynamic marginal-utility partitioner.
        let mut seneca = SenecaPartitioner::new(SenecaConfig {
            total_ram_bytes: ram_budget,
            slab_bytes: 64 * 1024,
            starvation_guard_ratio: 0.15,
            fairness_weight: 0.5,
        })?;
        for j in &jobs {
            seneca.register_job(j.clone())?;
        }
        let seneca_plan = seneca.rebalance();
        let sim_seneca = simulate_with_plan("seneca", &jobs, &seneca_plan, 400, 42);
        for (jid, summary) in &seneca_plan.jobs {
            println!(
                "[seneca plan] job={jid} enc={}B dec={}B xform={}B pred_idle={:.2}%",
                summary.quota.encoded_bytes,
                summary.quota.decoded_bytes,
                summary.quota.transformed_bytes,
                summary.predicted_gpu_idle_fraction * 100.0
            );
        }

        println!(
            "[seneca sim] uncached: mean_idle={:.2}% max_idle={:.2}% min_util={:.2}% total_ex_s={:.1}",
            sim_uncached.mean_gpu_idle_fraction * 100.0,
            sim_uncached.max_gpu_idle_fraction * 100.0,
            sim_uncached.min_job_gpu_utilization * 100.0,
            sim_uncached.total_examples_per_sec
        );
        println!(
            "[seneca sim] static:   mean_idle={:.2}% max_idle={:.2}% min_util={:.2}% total_ex_s={:.1} peak={}B",
            sim_static.mean_gpu_idle_fraction * 100.0,
            sim_static.max_gpu_idle_fraction * 100.0,
            sim_static.min_job_gpu_utilization * 100.0,
            sim_static.total_examples_per_sec,
            sim_static.peak_used_bytes
        );
        println!(
            "[seneca sim] seneca:   mean_idle={:.2}% max_idle={:.2}% min_util={:.2}% total_ex_s={:.1} peak={}B",
            sim_seneca.mean_gpu_idle_fraction * 100.0,
            sim_seneca.max_gpu_idle_fraction * 100.0,
            sim_seneca.min_job_gpu_utilization * 100.0,
            sim_seneca.total_examples_per_sec,
            sim_seneca.peak_used_bytes
        );

        assert!(sim_seneca.peak_used_bytes <= ram_budget);
        // Substantial GPU idle reduction vs both uncached and static equal split.
        assert!(
            sim_seneca.mean_gpu_idle_fraction < sim_static.mean_gpu_idle_fraction * 0.85,
            "seneca mean idle {:.4} should beat static {:.4}",
            sim_seneca.mean_gpu_idle_fraction,
            sim_static.mean_gpu_idle_fraction
        );
        assert!(
            sim_seneca.mean_gpu_idle_fraction < sim_uncached.mean_gpu_idle_fraction * 0.68,
            "seneca mean idle {:.4} should beat uncached {:.4}",
            sim_seneca.mean_gpu_idle_fraction,
            sim_uncached.mean_gpu_idle_fraction
        );
        // Starvation elimination: worst-off job's GPU utilization is substantially higher.
        assert!(
            sim_seneca.min_job_gpu_utilization > sim_static.min_job_gpu_utilization + 0.08,
            "seneca min_util {:.4} vs static {:.4}",
            sim_seneca.min_job_gpu_utilization,
            sim_static.min_job_gpu_utilization
        );
        assert!(sim_seneca.min_job_gpu_utilization > sim_uncached.min_job_gpu_utilization * 2.0);
        // Higher aggregate throughput (examples/sec).
        assert!(sim_seneca.total_examples_per_sec > sim_static.total_examples_per_sec * 1.25);
        assert!(sim_seneca.total_examples_per_sec > sim_uncached.total_examples_per_sec * 1.95);
        Ok(())
    }

    #[test]
    fn dynamic_rebalance_adapts_when_job_phase_shifts_or_job_leaves() -> Result<()> {
        let mut jobs = sample_concurrent_jobs();
        let ram_budget = 6 * 1024 * 1024;
        let mut seneca = SenecaPartitioner::new(SenecaConfig {
            total_ram_bytes: ram_budget,
            slab_bytes: 64 * 1024,
            starvation_guard_ratio: 0.15,
            fairness_weight: 0.4,
        })?;
        for j in &jobs {
            seneca.register_job(j.clone())?;
        }
        let plan1 = seneca.rebalance();

        let mut cache = TieredHostCache::new(ram_budget);
        cache.apply_plan(&plan1);
        for s in 0..80 {
            cache.access_sample(&jobs[0], s);
            cache.access_sample(&jobs[1], s);
        }
        assert!(cache.total_used_bytes() <= ram_budget);

        // Phase shift: heavy-gpu-saturated job suddenly becomes severely I/O-bound (S3 cold shard),
        // and multimodal-decode-bound finishes and leaves the cluster.
        seneca.unregister_job("multimodal-decode-bound")?;
        jobs[2].io_wait_ms = 24.0;
        jobs[2].gpu_step_ms = 10.0;
        jobs[2].encoded_item_bytes = 16 * 1024;
        jobs[2].skew_alpha = 0.85;
        seneca.update_job(jobs[2].clone())?;

        let plan2 = seneca.rebalance();
        cache.apply_plan(&plan2);
        assert!(plan2.allocated_bytes <= ram_budget);
        assert!(cache.total_used_bytes() <= ram_budget);
        assert!(
            plan2.jobs["heavy-gpu-saturated"].quota.total_bytes()
                > plan1.jobs["heavy-gpu-saturated"].quota.total_bytes(),
            "dynamic rebalance must shift RAM to the newly I/O-bound job"
        );
        // Unregistered job's partitions must be completely reclaimed.
        assert_eq!(
            cache.partition_used_bytes("multimodal-decode-bound", CacheTier::Decoded),
            0
        );
        Ok(())
    }

    #[test]
    fn starvation_elimination_under_greedy_flooding_neighbor() -> Result<()> {
        // One high-rate large-tensor scanning neighbor floods an unpartitioned global LRU
        // cache with 512 KiB items every step, starving a slow-IO vision job unless Seneca
        // protects and sizes its Encoded tier.
        let jobs = vec![
            JobWorkload {
                job_id: "starved-vision".to_string(),
                working_set_items: 256,
                batch_size: 4,
                encoded_item_bytes: 8 * 1024,
                decoded_item_bytes: 256 * 1024,
                transformed_item_bytes: 512 * 1024,
                io_wait_ms: 20.0,
                decode_cpu_ms: 3.0,
                transform_cpu_ms: 1.0,
                transform_reusable: false,
                gpu_step_ms: 8.0,
                loader_workers: 2,
                skew_alpha: 0.75,
                priority_weight: 1.0,
            },
            JobWorkload {
                job_id: "greedy-scanner".to_string(),
                working_set_items: 1000,
                batch_size: 12,
                encoded_item_bytes: 64 * 1024,
                decoded_item_bytes: 256 * 1024,
                transformed_item_bytes: 512 * 1024,
                io_wait_ms: 1.0,
                decode_cpu_ms: 0.5,
                transform_cpu_ms: 0.5,
                transform_reusable: true,
                gpu_step_ms: 25.0,
                loader_workers: 4,
                skew_alpha: 0.0,
                priority_weight: 1.0,
            },
        ];
        let ram_budget = 4 * 1024 * 1024; // 4 MiB
        let sim_lru = simulate_global_lru(&jobs, ram_budget, 500, 7);

        let mut seneca = SenecaPartitioner::new(SenecaConfig {
            total_ram_bytes: ram_budget,
            slab_bytes: 64 * 1024,
            starvation_guard_ratio: 0.15,
            fairness_weight: 0.5,
        })?;
        for j in &jobs {
            seneca.register_job(j.clone())?;
        }
        let plan = seneca.rebalance();
        let sim_seneca = simulate_with_plan("seneca", &jobs, &plan, 500, 7);

        let lru_vision = &sim_lru.jobs["starved-vision"];
        let sen_vision = &sim_seneca.jobs["starved-vision"];
        println!(
            "[seneca starvation] global_lru vision_idle={:.2}% vision_ex_s={:.1} hits={} | seneca vision_idle={:.2}% vision_ex_s={:.1} hits={}",
            lru_vision.gpu_idle_fraction * 100.0,
            lru_vision.examples_per_sec,
            lru_vision.cache_hits,
            sen_vision.gpu_idle_fraction * 100.0,
            sen_vision.examples_per_sec,
            sen_vision.cache_hits
        );

        assert!(sen_vision.cache_hits > lru_vision.cache_hits * 4);
        assert!(sen_vision.gpu_idle_fraction < lru_vision.gpu_idle_fraction * 0.55);
        assert!(sen_vision.examples_per_sec > lru_vision.examples_per_sec * 1.45);
        Ok(())
    }
}
