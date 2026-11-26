//! # `hanzo-seneca` — Multi-Stage Training Data-Pipeline Cache Allocation (HIP-1349)
//!
//! Clean-room, pure-Rust (`std`-only, zero external crate dependencies,
//! `#![forbid(unsafe_code)]`) implementation of Seneca-style multi-stage cache
//! allocation for concurrent ML training jobs across **Encoded** (S3/VFS
//! compressed), **Decoded**, and **Transformed** (augmented/tokenized) pipeline
//! stages.
//!
//! ## Core Invariants
//! 1. **Bounded cache memory**: `total_used_bytes <= total_budget_bytes` at every
//!    instant, including mid-rebalance (shrink-first, expand-second).
//! 2. **Strict tenant/job isolation**: `Private` dataset entries are scoped to
//!    `(tenant_id, job_id)` with `0` cross-tenant leaks. Only explicitly marked
//!    `SharedPublic` datasets are shared by `(dataset_digest, stage, transform_profile_hash)`.
//! 3. **Stochastic-augmentation freshness**: Jobs with `stochastic_transform = true`
//!    never allocate `Transformed` cache budget and never serve stale transformed
//!    tensors across epochs (`100%` fresh).
//! 4. **Zero-reuse streaming safety**: Single-epoch streaming workloads with
//!    `working_set >> cache` trigger streaming bypass to avoid copy/eviction thrashing.

#![forbid(unsafe_code)]

pub mod allocator;
pub mod cache;
pub mod codec;
pub mod pipeline;
pub mod scenarios;

pub use allocator::{
    AllocationDecision, AllocationPolicy, AllocatorConfig, JobTelemetry, SenecaAllocator,
};
pub use cache::{
    CacheEntry, CacheKey, CacheStage, DatasetVisibility, LruPartition, MultiStageCache, ScopeId,
    StageStats,
};
pub use codec::{compress_lz_rle, content_hash_128, crc32, decompress_lz_rle, hash_u64, Rng64};
pub use pipeline::{
    execute_decode, execute_transform, run_scenario, CorpusStore, DatasetStore, EncodedShard,
    JobConfig, JobMetrics, Modality, PolicyRunMetrics, ScenarioConfig,
};
pub use scenarios::{
    build_all_scenarios, report_to_json, run_full_suite, run_full_suite_with_trials,
    BenchmarkSuiteReport, ScenarioReport,
};
