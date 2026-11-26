//! Seneca marginal-utility dynamic multi-stage cache partitioner (HIP-1349).
//!
//! Clean-room implementation inspired by Seneca (Desai et al., USENIX FAST '26):
//! dynamically partitions a bounded host-RAM/NVMe cache budget (`TotalCacheBytes`)
//! across `{Encoded, Decoded, Transformed} × concurrent training jobs` to minimize
//! aggregate GPU stall time, subject to:
//! - **Stochastic-augmentation correctness invariant**: jobs with
//!   `stochastic_transform == true` NEVER receive `Transformed` cache budget
//!   (`budget[Transformed] == 0`), shifting all capacity to `Decoded` / `Encoded`.
//! - **Smooth hysteresis**: EMA blending + deadband prevents allocation thrashing.
//! - **Streaming bypass**: single-epoch, zero-reuse workloads (`working_set >> cache`)
//!   automatically disable cache admission to avoid copy/eviction overhead.

use std::collections::{HashMap, HashSet};

use crate::cache::{CacheStage, ScopeId};

/// The 5 matched cache allocation policies evaluated in the benchmark harness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AllocationPolicy {
    /// Direct S3/VFS streaming baseline (0 bytes cached).
    NoCache,
    /// Traditional OS page-cache / S3 shard cache only (100% to `Encoded`).
    EncodedOnly,
    /// Caches decoded tensors only (100% to `Decoded`).
    DecodedOnly,
    /// Fixed 33% / 33% / 33% split across `{Encoded, Decoded, Transformed}` and jobs.
    StaticEqualSplit,
    /// Seneca marginal-utility dynamic multi-stage partitioner with hysteresis,
    /// stochastic-augmentation guard, multi-job coordination, and streaming bypass.
    SenecaDynamic,
}

impl AllocationPolicy {
    pub const ALL: [AllocationPolicy; 5] = [
        AllocationPolicy::NoCache,
        AllocationPolicy::EncodedOnly,
        AllocationPolicy::DecodedOnly,
        AllocationPolicy::StaticEqualSplit,
        AllocationPolicy::SenecaDynamic,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::NoCache => "no_cache",
            Self::EncodedOnly => "encoded_only",
            Self::DecodedOnly => "decoded_only",
            Self::StaticEqualSplit => "static_equal_split",
            Self::SenecaDynamic => "seneca_dynamic",
        }
    }
}

/// Online per-job telemetry continuously updated by the pipeline engine.
#[derive(Debug, Clone)]
pub struct JobTelemetry {
    pub job_id: u64,
    pub scope: ScopeId,
    pub stochastic_transform: bool,
    pub working_set_samples: usize,
    pub planned_epochs: u32,
    pub batch_size: usize,
    pub gpu_step_us: f64,
    pub io_concurrency: f64,

    // Online EMA timings (microseconds per sample).
    pub t_fetch_us: f64,
    pub t_decode_us: f64,
    pub t_transform_us: f64,

    // Online EMA byte sizes per sample.
    pub size_encoded: f64,
    pub size_decoded: f64,
    pub size_transformed: f64,

    // Access / reuse tracking.
    pub total_requests: u64,
    pub seen_samples: HashSet<u64>,
    pub active: bool,

    // Observation counts for EMA warmup.
    fetch_obs: u64,
    decode_obs: u64,
    transform_obs: u64,
}

impl JobTelemetry {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        job_id: u64,
        scope: ScopeId,
        stochastic_transform: bool,
        working_set_samples: usize,
        planned_epochs: u32,
        batch_size: usize,
        gpu_step_us: f64,
        io_concurrency: f64,
    ) -> Self {
        Self {
            job_id,
            scope,
            stochastic_transform,
            working_set_samples: working_set_samples.max(1),
            planned_epochs: planned_epochs.max(1),
            batch_size: batch_size.max(1),
            gpu_step_us: gpu_step_us.max(1.0),
            io_concurrency: io_concurrency.max(1.0),
            t_fetch_us: 0.0,
            t_decode_us: 0.0,
            t_transform_us: 0.0,
            size_encoded: 0.0,
            size_decoded: 0.0,
            size_transformed: 0.0,
            total_requests: 0,
            seen_samples: HashSet::new(),
            active: true,
            fetch_obs: 0,
            decode_obs: 0,
            transform_obs: 0,
        }
    }

    /// Records a sample access for reuse estimation.
    pub fn record_access(&mut self, sample_id: u64) {
        self.total_requests += 1;
        // Cap tracking set size to keep memory bounded on massive streams.
        if self.seen_samples.len() < 100_000 {
            self.seen_samples.insert(sample_id);
        }
    }

    /// Records an observed S3/VFS fetch latency and encoded byte size.
    pub fn observe_fetch(&mut self, latency_us: f64, encoded_bytes: usize) {
        self.fetch_obs += 1;
        let alpha = if self.fetch_obs <= 8 { 0.5 } else { 0.15 };
        self.t_fetch_us = ema(self.t_fetch_us, latency_us, alpha, self.fetch_obs == 1);
        self.size_encoded = ema(
            self.size_encoded,
            encoded_bytes.max(1) as f64,
            alpha,
            self.fetch_obs == 1,
        );
    }

    /// Records an observed CPU decode latency and decoded byte size.
    pub fn observe_decode(&mut self, latency_us: f64, decoded_bytes: usize) {
        self.decode_obs += 1;
        let alpha = if self.decode_obs <= 8 { 0.5 } else { 0.15 };
        self.t_decode_us = ema(self.t_decode_us, latency_us, alpha, self.decode_obs == 1);
        self.size_decoded = ema(
            self.size_decoded,
            decoded_bytes.max(1) as f64,
            alpha,
            self.decode_obs == 1,
        );
    }

    /// Records an observed CPU transform latency and transformed byte size.
    pub fn observe_transform(&mut self, latency_us: f64, transformed_bytes: usize) {
        self.transform_obs += 1;
        let alpha = if self.transform_obs <= 8 { 0.5 } else { 0.15 };
        self.t_transform_us = ema(
            self.t_transform_us,
            latency_us,
            alpha,
            self.transform_obs == 1,
        );
        self.size_transformed = ema(
            self.size_transformed,
            transformed_bytes.max(1) as f64,
            alpha,
            self.transform_obs == 1,
        );
    }

    /// Returns `true` once all three pipeline stages have been profiled at least once.
    #[inline]
    pub fn is_profiled(&self) -> bool {
        self.fetch_obs > 0 && self.decode_obs > 0 && self.transform_obs > 0
    }

    /// Empirical repeat access ratio observed online in `[0.0, 1.0]`.
    pub fn observed_repeat_ratio(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        let unique = self.seen_samples.len() as u64;
        if unique >= self.total_requests {
            0.0
        } else {
            (self.total_requests - unique) as f64 / self.total_requests as f64
        }
    }

    /// Estimated future reuse probability combining epoch multiplicity,
    /// multi-job shared-dataset overlap, and online observed repeat rate.
    pub fn effective_reuse_rate(&self, scope_sharers: usize, total_budget_bytes: usize) -> f64 {
        let obs = self.observed_repeat_ratio();
        let sharers = scope_sharers.max(1);

        // Single-epoch, unshared dataset: only rely on genuine online repeats!
        // If working set dwarfs the cache and no repeats occur, reuse is 0.0.
        if self.planned_epochs <= 1 && sharers == 1 {
            let min_stage_size = self
                .size_encoded
                .max(1.0)
                .min(self.size_decoded.max(1.0))
                .min(self.size_transformed.max(1.0));
            let working_set_bytes = (self.working_set_samples as f64) * min_stage_size;
            if working_set_bytes > (total_budget_bytes as f64) * 4.0 && obs < 0.02 {
                return 0.0;
            }
            return obs;
        }

        // Multi-epoch or multi-job shared dataset: structural reuse prior blended
        // with observed repeat rate.
        let epoch_reuse = 1.0 - (1.0 / self.planned_epochs as f64);
        let sharing_reuse = 1.0 - (1.0 / sharers as f64);
        let structural = 1.0 - (1.0 - epoch_reuse) * (1.0 - sharing_reuse);
        structural.max(obs).clamp(0.0, 1.0)
    }

    /// Returns the observed decoded-to-encoded memory expansion ratio.
    pub fn decode_expansion_ratio(&self) -> f64 {
        if self.size_encoded <= 0.0 {
            1.0
        } else {
            self.size_decoded / self.size_encoded
        }
    }
}

#[inline]
fn ema(current: f64, sample: f64, alpha: f64, first: bool) -> f64 {
    if first {
        sample
    } else {
        (1.0 - alpha) * current + alpha * sample
    }
}

/// Configuration for [`SenecaAllocator`].
#[derive(Debug, Clone)]
pub struct AllocatorConfig {
    pub policy: AllocationPolicy,
    pub total_budget_bytes: usize,
    pub host_cpu_cores: usize,
    /// Number of discrete allocation quanta for the marginal-utility solver.
    pub quanta: usize,
    /// Hysteresis EMA smoothing factor in `(0.0, 1.0]`.
    pub hysteresis_alpha: f64,
    /// Minimum L1 budget shift fraction required to trigger a rebalance.
    pub hysteresis_deadband_frac: f64,
    /// Minimum global steps between dynamic rebalances.
    pub rebalance_interval_steps: u64,
    /// Reuse rate below which Seneca activates streaming bypass.
    pub streaming_bypass_reuse_threshold: f64,
}

impl AllocatorConfig {
    pub fn new(policy: AllocationPolicy, total_budget_bytes: usize, host_cpu_cores: usize) -> Self {
        Self {
            policy,
            total_budget_bytes,
            host_cpu_cores: host_cpu_cores.max(1),
            quanta: 128,
            hysteresis_alpha: 0.45,
            hysteresis_deadband_frac: 0.04,
            rebalance_interval_steps: 4,
            streaming_bypass_reuse_threshold: 0.02,
        }
    }
}

/// Output of one allocation step.
#[derive(Debug, Clone)]
pub struct AllocationDecision {
    pub budgets: HashMap<(ScopeId, CacheStage), usize>,
    pub admit_enabled: bool,
    pub rebalance_applied: bool,
    pub streaming_bypass_active: bool,
}

/// Seneca dynamic multi-stage cache allocator.
#[derive(Debug, Clone)]
pub struct SenecaAllocator {
    config: AllocatorConfig,
    current_budgets: HashMap<(ScopeId, CacheStage), usize>,
    initialized: bool,
    profiled_initialized: bool,
    rebalance_count: u64,
    bypass_activations: u64,
    last_rebalance_step: u64,
}

impl SenecaAllocator {
    pub fn new(config: AllocatorConfig) -> Self {
        Self {
            config,
            current_budgets: HashMap::new(),
            initialized: false,
            profiled_initialized: false,
            rebalance_count: 0,
            bypass_activations: 0,
            last_rebalance_step: 0,
        }
    }

    #[inline]
    pub fn policy(&self) -> AllocationPolicy {
        self.config.policy
    }

    #[inline]
    pub fn rebalance_count(&self) -> u64 {
        self.rebalance_count
    }

    #[inline]
    pub fn bypass_activations(&self) -> u64 {
        self.bypass_activations
    }

    #[inline]
    pub fn current_budgets(&self) -> &HashMap<(ScopeId, CacheStage), usize> {
        &self.current_budgets
    }

    /// Computes initial or updated cache partition budgets for the active jobs.
    pub fn rebalance(
        &mut self,
        step: u64,
        jobs: &[JobTelemetry],
        force: bool,
    ) -> AllocationDecision {
        let active_jobs: Vec<&JobTelemetry> = jobs.iter().filter(|j| j.active).collect();
        if active_jobs.is_empty() || self.config.total_budget_bytes == 0 {
            self.current_budgets.clear();
            return AllocationDecision {
                budgets: HashMap::new(),
                admit_enabled: false,
                rebalance_applied: false,
                streaming_bypass_active: false,
            };
        }

        if self.initialized
            && !force
            && step.saturating_sub(self.last_rebalance_step) < self.config.rebalance_interval_steps
        {
            return AllocationDecision {
                budgets: self.current_budgets.clone(),
                admit_enabled: self.config.policy != AllocationPolicy::NoCache
                    && !self.current_budgets.is_empty()
                    && self.current_budgets.values().copied().sum::<usize>() > 0,
                rebalance_applied: false,
                streaming_bypass_active: self.config.policy == AllocationPolicy::SenecaDynamic
                    && self.current_budgets.values().copied().sum::<usize>() == 0,
            };
        }

        let decision = match self.config.policy {
            AllocationPolicy::NoCache => AllocationDecision {
                budgets: HashMap::new(),
                admit_enabled: false,
                rebalance_applied: !self.initialized,
                streaming_bypass_active: false,
            },
            AllocationPolicy::EncodedOnly => {
                let budgets = self.allocate_single_stage(&active_jobs, CacheStage::Encoded);
                AllocationDecision {
                    budgets,
                    admit_enabled: true,
                    rebalance_applied: true,
                    streaming_bypass_active: false,
                }
            }
            AllocationPolicy::DecodedOnly => {
                let budgets = self.allocate_single_stage(&active_jobs, CacheStage::Decoded);
                AllocationDecision {
                    budgets,
                    admit_enabled: true,
                    rebalance_applied: true,
                    streaming_bypass_active: false,
                }
            }
            AllocationPolicy::StaticEqualSplit => {
                let budgets = self.allocate_static_equal(&active_jobs);
                AllocationDecision {
                    budgets,
                    admit_enabled: true,
                    rebalance_applied: true,
                    streaming_bypass_active: false,
                }
            }
            AllocationPolicy::SenecaDynamic => self.allocate_seneca_dynamic(&active_jobs),
        };

        self.last_rebalance_step = step;
        if decision.rebalance_applied || !self.initialized {
            self.current_budgets = decision.budgets.clone();
            self.initialized = true;
            if decision.rebalance_applied {
                self.rebalance_count += 1;
            }
        }
        decision
    }

    fn unique_scopes(active_jobs: &[&JobTelemetry]) -> Vec<ScopeId> {
        let mut scopes: Vec<ScopeId> = active_jobs.iter().map(|j| j.scope).collect();
        scopes.sort();
        scopes.dedup();
        scopes
    }

    fn allocate_single_stage(
        &self,
        active_jobs: &[&JobTelemetry],
        stage: CacheStage,
    ) -> HashMap<(ScopeId, CacheStage), usize> {
        let scopes = Self::unique_scopes(active_jobs);
        let n = scopes.len().max(1);
        let share = self.config.total_budget_bytes / n;
        let rem = self.config.total_budget_bytes % n;
        let mut out = HashMap::new();
        for (i, scope) in scopes.into_iter().enumerate() {
            let b = share + usize::from(i < rem);
            out.insert((scope, stage), b);
        }
        out
    }

    fn allocate_static_equal(
        &self,
        active_jobs: &[&JobTelemetry],
    ) -> HashMap<(ScopeId, CacheStage), usize> {
        // Fixed 33% / 33% / 33% split across the 3 stages and equally across active scopes.
        // Note: For stochastic jobs, the pipeline still enforces the correctness invariant
        // (never reading stale Transformed entries), so StaticEqualSplit's 33% Transformed
        // share sits unused for stochastic jobs — accurately illustrating the waste of static splits.
        let scopes = Self::unique_scopes(active_jobs);
        let buckets = scopes.len() * 3;
        let share = self.config.total_budget_bytes / buckets.max(1);
        let rem = self.config.total_budget_bytes % buckets.max(1);
        let mut out = HashMap::with_capacity(buckets);
        let mut idx = 0usize;
        for scope in scopes {
            for stage in CacheStage::ALL {
                let b = share + usize::from(idx < rem);
                out.insert((scope, stage), b);
                idx += 1;
            }
        }
        out
    }

    fn allocate_seneca_dynamic(&mut self, active_jobs: &[&JobTelemetry]) -> AllocationDecision {
        // Count how many active jobs share each scope.
        let mut sharers_by_scope: HashMap<ScopeId, usize> = HashMap::new();
        for j in active_jobs {
            *sharers_by_scope.entry(j.scope).or_insert(0) += 1;
        }

        // Check streaming bypass condition: if every active job has ~0 effective reuse,
        // caching cannot yield hits and would only waste CPU on copies/LRU evictions.
        let any_reusable = active_jobs.iter().any(|j| {
            let sharers = sharers_by_scope.get(&j.scope).copied().unwrap_or(1);
            j.effective_reuse_rate(sharers, self.config.total_budget_bytes)
                >= self.config.streaming_bypass_reuse_threshold
        });
        if !any_reusable {
            self.bypass_activations += 1;
            let changed =
                !self.initialized || self.current_budgets.values().copied().sum::<usize>() > 0;
            return AllocationDecision {
                budgets: HashMap::new(),
                admit_enabled: false,
                rebalance_applied: changed,
                streaming_bypass_active: true,
            };
        }

        // If telemetry hasn't warmed up yet on any job, start with a safe Encoded/Decoded
        // bootstrap that strictly respects the stochastic-augmentation invariant.
        let all_profiled = active_jobs.iter().all(|j| j.is_profiled());
        let target = if !all_profiled {
            self.bootstrap_allocation(active_jobs)
        } else {
            self.solve_marginal_utility(active_jobs, &sharers_by_scope)
        };

        // Hard safety invariant: enforce Transformed == 0 for scopes where ALL jobs are stochastic.
        let target = Self::enforce_stochastic_guard(target, active_jobs);

        // Apply immediately on cold-start bootstrap OR on the very first transition
        // from bootstrap into a fully profiled marginal-utility solution.
        // Hysteresis applies strictly between steady-state profiled allocations.
        if !self.initialized || (all_profiled && !self.profiled_initialized) {
            if all_profiled {
                self.profiled_initialized = true;
            }
            return AllocationDecision {
                budgets: target,
                admit_enabled: true,
                rebalance_applied: true,
                streaming_bypass_active: false,
            };
        }

        let mut all_keys: Vec<(ScopeId, CacheStage)> = self
            .current_budgets
            .keys()
            .chain(target.keys())
            .copied()
            .collect();
        all_keys.sort();
        all_keys.dedup();

        let l1_diff: usize = all_keys
            .iter()
            .map(|k| {
                let cur = self.current_budgets.get(k).copied().unwrap_or(0);
                let tgt = target.get(k).copied().unwrap_or(0);
                cur.abs_diff(tgt)
            })
            .sum();

        let shift_frac = (l1_diff as f64) / (2.0 * self.config.total_budget_bytes.max(1) as f64);
        if shift_frac < self.config.hysteresis_deadband_frac {
            return AllocationDecision {
                budgets: self.current_budgets.clone(),
                admit_enabled: true,
                rebalance_applied: false,
                streaming_bypass_active: false,
            };
        }

        let alpha = self.config.hysteresis_alpha.clamp(0.05, 1.0);
        let mut blended = HashMap::with_capacity(all_keys.len());
        let mut total = 0usize;
        for k in &all_keys {
            let cur = self.current_budgets.get(k).copied().unwrap_or(0) as f64;
            let tgt = target.get(k).copied().unwrap_or(0) as f64;
            let val = ((1.0 - alpha) * cur + alpha * tgt).floor() as usize;
            if val > 0 {
                let capped = val.min(self.config.total_budget_bytes.saturating_sub(total));
                total += capped;
                blended.insert(*k, capped);
            }
        }
        // Re-verify stochastic guard after blending.
        let blended = Self::enforce_stochastic_guard(blended, active_jobs);
        AllocationDecision {
            budgets: blended,
            admit_enabled: true,
            rebalance_applied: true,
            streaming_bypass_active: false,
        }
    }

    fn bootstrap_allocation(
        &self,
        active_jobs: &[&JobTelemetry],
    ) -> HashMap<(ScopeId, CacheStage), usize> {
        // Before first full telemetry sample, allocate conservatively to Encoded (compact)
        // so we measure real stage timings without blowing up RAM.
        self.allocate_single_stage(active_jobs, CacheStage::Encoded)
    }

    fn enforce_stochastic_guard(
        mut budgets: HashMap<(ScopeId, CacheStage), usize>,
        active_jobs: &[&JobTelemetry],
    ) -> HashMap<(ScopeId, CacheStage), usize> {
        let scopes = Self::unique_scopes(active_jobs);
        for scope in scopes {
            let any_deterministic = active_jobs
                .iter()
                .any(|j| j.scope == scope && !j.stochastic_transform);
            if !any_deterministic {
                budgets.remove(&(scope, CacheStage::Transformed));
            }
        }
        budgets
    }

    /// Evaluates `(objective, total_stall_ratio)` across all concurrent jobs under
    /// candidate allocation `alloc`. Lower is better.
    fn evaluate_objective(
        &self,
        active_jobs: &[&JobTelemetry],
        sharers_by_scope: &HashMap<ScopeId, usize>,
        alloc: &HashMap<(ScopeId, CacheStage), usize>,
    ) -> (f64, f64) {
        // First compute per-job expected I/O and CPU times per batch.
        let mut io_batch_us = Vec::with_capacity(active_jobs.len());
        let mut cpu_batch_us = Vec::with_capacity(active_jobs.len());

        for job in active_jobs {
            let b_e = alloc
                .get(&(job.scope, CacheStage::Encoded))
                .copied()
                .unwrap_or(0) as f64;
            let b_d = alloc
                .get(&(job.scope, CacheStage::Decoded))
                .copied()
                .unwrap_or(0) as f64;
            let b_t = if job.stochastic_transform {
                0.0
            } else {
                alloc
                    .get(&(job.scope, CacheStage::Transformed))
                    .copied()
                    .unwrap_or(0) as f64
            };

            let sharers = sharers_by_scope.get(&job.scope).copied().unwrap_or(1);
            let reuse = job.effective_reuse_rate(sharers, self.config.total_budget_bytes);

            let ws = job.working_set_samples as f64;
            let s_e = job.size_encoded.max(1.0);
            let s_d = job.size_decoded.max(1.0);
            let s_t = job.size_transformed.max(1.0);

            // Disjoint multi-stage coverage (Seneca MDP model):
            // highest-utility stages serve their resident fraction first.
            let cov_t = if job.stochastic_transform {
                0.0
            } else {
                (b_t / (ws * s_t)).clamp(0.0, 1.0)
            };
            let cov_d = (b_d / (ws * s_d)).clamp(0.0, (1.0 - cov_t).max(0.0));
            let cov_e = (b_e / (ws * s_e)).clamp(0.0, (1.0 - cov_t - cov_d).max(0.0));

            let h_t = cov_t * reuse;
            let h_d = cov_d * reuse;
            let h_e = cov_e * reuse;

            let miss_fetch = (1.0 - h_t - h_d - h_e).max(0.0);
            let miss_decode = (1.0 - h_t - h_d).max(0.0);
            let miss_transform = (1.0 - h_t).max(0.0);

            let batch = job.batch_size as f64;
            let io_us = batch * miss_fetch * job.t_fetch_us / job.io_concurrency;
            let cpu_us =
                batch * (miss_decode * job.t_decode_us + miss_transform * job.t_transform_us);
            io_batch_us.push(io_us);
            cpu_batch_us.push(cpu_us);
        }

        // Shared host-CPU contention across concurrent jobs:
        // jobs running concurrently contend for `host_cpu_cores`.
        let total_cpu_rate: f64 = active_jobs
            .iter()
            .zip(cpu_batch_us.iter())
            .map(|(j, &c)| c / j.gpu_step_us.max(1.0))
            .sum();
        let cores = self.config.host_cpu_cores as f64;
        let contention = (total_cpu_rate / cores).max(1.0);

        let mut objective = 0.0f64;
        let mut total_stall = 0.0f64;
        for (idx, job) in active_jobs.iter().enumerate() {
            let eff_cpu =
                cpu_batch_us[idx] * contention / cores.min(active_jobs.len() as f64).max(1.0);
            let prep_us = io_batch_us[idx] + eff_cpu;
            let stall_ratio = (prep_us - job.gpu_step_us).max(0.0) / job.gpu_step_us;
            let prep_ratio = prep_us / job.gpu_step_us;
            total_stall += stall_ratio;
            // Strictly convex stall objective: prioritizes the most GPU-stalled job
            // (minimizing multi-job makespan) and responds smoothly to small telemetry jitter
            // instead of bang-bang flipping between similar jobs.
            objective += stall_ratio + 0.5 * stall_ratio * stall_ratio + 0.05 * prep_ratio;
        }
        (objective, total_stall)
    }

    /// Greedy marginal-utility solver over `quanta` discrete budget steps.
    /// Supports both direct stage additions (`+q_bytes`) and in-place hierarchical
    /// upgrades (`Encoded -> Decoded`, `Decoded -> Transformed`, `Encoded -> Transformed`)
    /// when a scope's working set is already covered at a lower stage.
    fn solve_marginal_utility(
        &self,
        active_jobs: &[&JobTelemetry],
        sharers_by_scope: &HashMap<ScopeId, usize>,
    ) -> HashMap<(ScopeId, CacheStage), usize> {
        let scopes = Self::unique_scopes(active_jobs);
        let quanta = self.config.quanta.max(16);
        let q_bytes = (self.config.total_budget_bytes / quanta).max(1);
        let mut alloc: HashMap<(ScopeId, CacheStage), usize> = HashMap::new();
        let mut remaining = self.config.total_budget_bytes;

        // Precompute scope metadata (whether any job allows Transformed, representative sizes).
        struct ScopeInfo {
            scope: ScopeId,
            allow_transformed: bool,
            s_e: f64,
            s_d: f64,
            s_t: f64,
            max_ws_e: usize,
            max_ws_d: usize,
            max_ws_t: usize,
        }

        let scope_infos: Vec<ScopeInfo> = scopes
            .iter()
            .map(|&scope| {
                let s_jobs: Vec<&JobTelemetry> = active_jobs
                    .iter()
                    .copied()
                    .filter(|j| j.scope == scope)
                    .collect();
                let allow_transformed = s_jobs.iter().any(|j| !j.stochastic_transform);
                let s_e = s_jobs.iter().map(|j| j.size_encoded.max(1.0)).sum::<f64>()
                    / s_jobs.len().max(1) as f64;
                let s_d = s_jobs.iter().map(|j| j.size_decoded.max(1.0)).sum::<f64>()
                    / s_jobs.len().max(1) as f64;
                let s_t = s_jobs
                    .iter()
                    .map(|j| j.size_transformed.max(1.0))
                    .sum::<f64>()
                    / s_jobs.len().max(1) as f64;
                let max_ws = s_jobs
                    .iter()
                    .map(|j| j.working_set_samples)
                    .max()
                    .unwrap_or(1) as f64;
                ScopeInfo {
                    scope,
                    allow_transformed,
                    s_e,
                    s_d,
                    s_t,
                    max_ws_e: (max_ws * s_e).ceil() as usize,
                    max_ws_d: (max_ws * s_d).ceil() as usize,
                    max_ws_t: (max_ws * s_t).ceil() as usize,
                }
            })
            .collect();

        let (mut base_obj, mut base_stall) =
            self.evaluate_objective(active_jobs, sharers_by_scope, &alloc);

        while remaining > 0 {
            let step_bytes = remaining.min(q_bytes);

            #[derive(Clone, Copy)]
            enum Move {
                Add(ScopeId, CacheStage, usize),
                Upgrade {
                    scope: ScopeId,
                    from: CacheStage,
                    to: CacheStage,
                    release_from: usize,
                    add_to: usize,
                },
            }

            let mut best_gain_per_byte = 0.0f64;
            let mut best_move: Option<Move> = None;

            for info in &scope_infos {
                let cur_e = alloc
                    .get(&(info.scope, CacheStage::Encoded))
                    .copied()
                    .unwrap_or(0);
                let cur_d = alloc
                    .get(&(info.scope, CacheStage::Decoded))
                    .copied()
                    .unwrap_or(0);
                let cur_t = alloc
                    .get(&(info.scope, CacheStage::Transformed))
                    .copied()
                    .unwrap_or(0);

                // Candidate Type A: Direct addition to each eligible stage.
                for stage in CacheStage::ALL {
                    if stage == CacheStage::Transformed && !info.allow_transformed {
                        continue;
                    }
                    let (cur, max_ws) = match stage {
                        CacheStage::Encoded => (cur_e, info.max_ws_e),
                        CacheStage::Decoded => (cur_d, info.max_ws_d),
                        CacheStage::Transformed => (cur_t, info.max_ws_t),
                    };
                    if cur >= max_ws {
                        continue;
                    }
                    let add = step_bytes.min(max_ws - cur);
                    if add == 0 {
                        continue;
                    }

                    let prev = alloc.insert((info.scope, stage), cur + add);
                    let (cand_obj, _) =
                        self.evaluate_objective(active_jobs, sharers_by_scope, &alloc);
                    if let Some(p) = prev {
                        alloc.insert((info.scope, stage), p);
                    } else {
                        alloc.remove(&(info.scope, stage));
                    }

                    let gain = (base_obj - cand_obj) / (add as f64);
                    if gain > best_gain_per_byte {
                        best_gain_per_byte = gain;
                        best_move = Some(Move::Add(info.scope, stage, add));
                    }
                }

                // Candidate Type B: Hierarchical upgrade from lower stage -> higher stage.
                // Only allowed when it reduces genuine GPU stall (`cand_stall < base_stall`),
                // preventing wasteful eviction of warm Encoded partitions for tiny CPU tie-breakers.
                let upgrade_pairs: &[(CacheStage, CacheStage, usize, f64, f64)] = &[
                    (
                        CacheStage::Encoded,
                        CacheStage::Decoded,
                        cur_e,
                        info.s_e,
                        info.s_d,
                    ),
                    (
                        CacheStage::Decoded,
                        CacheStage::Transformed,
                        cur_d,
                        info.s_d,
                        info.s_t,
                    ),
                    (
                        CacheStage::Encoded,
                        CacheStage::Transformed,
                        cur_e,
                        info.s_e,
                        info.s_t,
                    ),
                ];

                for &(from, to, cur_from, s_from, s_to) in upgrade_pairs {
                    if to == CacheStage::Transformed && !info.allow_transformed {
                        continue;
                    }
                    if cur_from == 0 {
                        continue;
                    }
                    let (release_from, add_to, net_cost) = if s_to > s_from {
                        let k_by_budget = (step_bytes as f64) / (s_to - s_from);
                        let k_by_avail = (cur_from as f64) / s_from;
                        let k = k_by_budget.min(k_by_avail);
                        let rel = ((k * s_from).floor() as usize).min(cur_from);
                        let add = ((k * s_to).floor() as usize).min(rel + step_bytes);
                        if rel == 0 || add <= rel {
                            continue;
                        }
                        (rel, add, add - rel)
                    } else {
                        let rel = cur_from.min(step_bytes);
                        let add = ((rel as f64) * (s_to / s_from)).ceil() as usize;
                        if rel == 0 || add == 0 {
                            continue;
                        }
                        (rel, add, 1)
                    };

                    let cur_to = alloc.get(&(info.scope, to)).copied().unwrap_or(0);
                    let prev_from = alloc.insert((info.scope, from), cur_from - release_from);
                    let prev_to = alloc.insert((info.scope, to), cur_to + add_to);
                    let (cand_obj, cand_stall) =
                        self.evaluate_objective(active_jobs, sharers_by_scope, &alloc);

                    // Restore.
                    if let Some(pf) = prev_from {
                        alloc.insert((info.scope, from), pf);
                    } else {
                        alloc.remove(&(info.scope, from));
                    }
                    if let Some(pt) = prev_to {
                        alloc.insert((info.scope, to), pt);
                    } else {
                        alloc.remove(&(info.scope, to));
                    }

                    if base_stall - cand_stall <= 1e-6 {
                        continue;
                    }

                    let gain = (base_obj - cand_obj) / (net_cost.max(1) as f64);
                    if gain > best_gain_per_byte {
                        best_gain_per_byte = gain;
                        best_move = Some(Move::Upgrade {
                            scope: info.scope,
                            from,
                            to,
                            release_from,
                            add_to,
                        });
                    }
                }
            }

            let Some(chosen) = best_move else {
                break;
            };
            if best_gain_per_byte <= 1e-12 {
                break;
            }

            match chosen {
                Move::Add(scope, stage, add) => {
                    *alloc.entry((scope, stage)).or_insert(0) += add;
                    remaining = remaining.saturating_sub(add);
                }
                Move::Upgrade {
                    scope,
                    from,
                    to,
                    release_from,
                    add_to,
                } => {
                    if let Some(v) = alloc.get_mut(&(scope, from)) {
                        *v = v.saturating_sub(release_from);
                    }
                    *alloc.entry((scope, to)).or_insert(0) += add_to;
                    let net = add_to.saturating_sub(release_from);
                    remaining = remaining.saturating_sub(net);
                }
            }
            (base_obj, base_stall) = self.evaluate_objective(active_jobs, sharers_by_scope, &alloc);
        }

        // Prune zero-byte entries for cleanliness.
        alloc.retain(|_, v| *v > 0);
        debug_assert!(alloc.values().copied().sum::<usize>() <= self.config.total_budget_bytes);
        alloc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::too_many_arguments)]
    fn make_profiled_job(
        job_id: u64,
        scope: ScopeId,
        stochastic: bool,
        ws: usize,
        epochs: u32,
        t_f: f64,
        t_d: f64,
        t_t: f64,
        s_e: usize,
        s_d: usize,
        s_t: usize,
    ) -> JobTelemetry {
        let mut j = JobTelemetry::new(job_id, scope, stochastic, ws, epochs, 16, 400.0, 4.0);
        j.observe_fetch(t_f, s_e);
        j.observe_decode(t_d, s_d);
        j.observe_transform(t_t, s_t);
        j
    }

    #[test]
    fn seneca_never_allocates_transformed_to_stochastic_job() {
        let scope = ScopeId::Private {
            tenant_id: 1,
            job_id: 1,
        };
        let job = make_profiled_job(
            1, scope, true, // stochastic!
            200, 4, 200.0, 600.0, 1200.0, 4096, 8192, 8192,
        );
        let mut alloc = SenecaAllocator::new(AllocatorConfig::new(
            AllocationPolicy::SenecaDynamic,
            1_024 * 1024,
            4,
        ));
        let dec = alloc.rebalance(1, &[job], true);
        assert!(dec.admit_enabled);
        assert_eq!(
            dec.budgets
                .get(&(scope, CacheStage::Transformed))
                .copied()
                .unwrap_or(0),
            0,
            "stochastic job MUST NEVER receive Transformed cache budget"
        );
        assert!(
            dec.budgets
                .get(&(scope, CacheStage::Decoded))
                .copied()
                .unwrap_or(0)
                > 0,
            "budget should shift to Decoded for CPU-heavy stochastic job"
        );
    }

    #[test]
    fn seneca_prefers_encoded_under_high_expansion_and_tight_ram() {
        let scope = ScopeId::Private {
            tenant_id: 1,
            job_id: 1,
        };
        // 6x expansion: 4 KiB encoded -> 24 KiB decoded; high S3 fetch latency, modest decode.
        let job = make_profiled_job(
            1, scope, false, 200, 4, 600.0, 80.0, 30.0, 4096, 24576, 24576,
        );
        // Tight RAM: fits ~80% of encoded working set (640 KiB), only ~13% of decoded.
        let mut alloc = SenecaAllocator::new(AllocatorConfig::new(
            AllocationPolicy::SenecaDynamic,
            640 * 1024,
            4,
        ));
        let dec = alloc.rebalance(1, &[job], true);
        let b_e = dec
            .budgets
            .get(&(scope, CacheStage::Encoded))
            .copied()
            .unwrap_or(0);
        let b_d = dec
            .budgets
            .get(&(scope, CacheStage::Decoded))
            .copied()
            .unwrap_or(0);
        assert!(
            b_e > b_d * 2,
            "expected Encoded ({b_e}) to dominate Decoded ({b_d}) under 6x expansion + tight RAM"
        );
    }

    #[test]
    fn seneca_prefers_transformed_when_cpu_transform_dominates() {
        let scope = ScopeId::Private {
            tenant_id: 1,
            job_id: 1,
        };
        // Fast storage, heavy deterministic transform, compact transformed output.
        let job = make_profiled_job(
            1, scope, false, 200, 4, 20.0, 300.0, 900.0, 4096, 8192, 6144,
        );
        let mut alloc = SenecaAllocator::new(AllocatorConfig::new(
            AllocationPolicy::SenecaDynamic,
            1_024 * 1024,
            2,
        ));
        let dec = alloc.rebalance(1, &[job], true);
        let b_t = dec
            .budgets
            .get(&(scope, CacheStage::Transformed))
            .copied()
            .unwrap_or(0);
        let b_e = dec
            .budgets
            .get(&(scope, CacheStage::Encoded))
            .copied()
            .unwrap_or(0);
        assert!(
            b_t > b_e,
            "expected Transformed ({b_t}) > Encoded ({b_e}) when CPU transform dominates"
        );
    }

    #[test]
    fn seneca_activates_streaming_bypass_on_single_epoch_no_reuse() {
        let scope = ScopeId::Private {
            tenant_id: 1,
            job_id: 1,
        };
        // 2400 samples * 4 KiB = 9.6 MiB >> 512 KiB budget, 1 epoch, zero repeats.
        let mut job = make_profiled_job(
            1, scope, false, 2400, 1, 200.0, 100.0, 100.0, 4096, 8192, 8192,
        );
        for s in 0..100u64 {
            job.record_access(s);
        }
        let mut alloc = SenecaAllocator::new(AllocatorConfig::new(
            AllocationPolicy::SenecaDynamic,
            512 * 1024,
            2,
        ));
        let dec = alloc.rebalance(1, &[job], true);
        assert!(dec.streaming_bypass_active);
        assert!(!dec.admit_enabled);
        assert_eq!(dec.budgets.values().copied().sum::<usize>(), 0);
    }

    #[test]
    fn hysteresis_prevents_small_jitter_rebalances() {
        let scope_a = ScopeId::Private {
            tenant_id: 1,
            job_id: 1,
        };
        let scope_b = ScopeId::Private {
            tenant_id: 2,
            job_id: 2,
        };
        let mut j1 = make_profiled_job(
            1, scope_a, false, 150, 4, 300.0, 200.0, 100.0, 4096, 8192, 8192,
        );
        let j2 = make_profiled_job(
            2, scope_b, false, 150, 4, 300.0, 200.0, 100.0, 4096, 8192, 8192,
        );

        let mut alloc = SenecaAllocator::new(AllocatorConfig::new(
            AllocationPolicy::SenecaDynamic,
            1_024 * 1024,
            4,
        ));
        let d1 = alloc.rebalance(1, &[j1.clone(), j2.clone()], true);
        assert!(d1.rebalance_applied);

        // Tiny 0.5% jitter should be absorbed by hysteresis deadband.
        j1.t_fetch_us *= 1.005;
        let d2 = alloc.rebalance(10, &[j1, j2], false);
        assert!(!d2.rebalance_applied);
    }
}
