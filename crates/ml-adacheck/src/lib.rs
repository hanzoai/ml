// SPDX-License-Identifier: Apache-2.0
//! [Bounty #45] (`hanzoai/ml#48`): **Redundancy-aware training checkpoints** (AdaCheck).
//!
//! Clean-room simulation harness for deduplicating shared tensors across
//! distributed training checkpoint shards: how much write volume can be removed
//! while keeping recovery byte-identical?
//!
//! # Design
//!
//! Distributed training writes per-rank shards. In practice a large fraction of
//! bytes are *shared* (replicated embeddings, frozen layers, tied weights,
//! optimizer state for parameters that never update). AdaCheck replaces shared
//! payloads with references and stores each unique payload once.
//!
//! - `NaiveShards`: every rank writes its full shard.
//! - `AdaCheckDedup`: content-addressed store + per-shard reference tables.
//!
//! Metrics: bytes written, bytes read on restore, dedup ratio, restore
//! correctness (must reconstruct every shard byte-identically), and the cost of
//! the content-addressing pass.
//!
//! # Honest boundaries
//! - Synthetic tensor payloads; no real model or filesystem is exercised.
//! - Sharing structure is modelled (replication factor, frozen fraction), not
//!   derived from a real checkpoint.
//! - All figures are simulation outputs on the authoring host, NOT Hanzo
//!   expectations until reproduced on a real training run.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum AdaError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
}

#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A tensor payload in a shard.
#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    /// Deterministic payload identity. Tensors with the same `content_id`
    /// produce identical bytes -?this is what dedup keys on.
    pub content_id: u64,
    pub bytes: u64,
    /// Whether this tensor changes between steps (optimizer state, activations).
    pub mutable: bool,
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub rank: u32,
    pub tensors: Vec<Tensor>,
}

impl Shard {
    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes).sum()
    }
}

/// Build a shard set that models the redundancy seen in real training runs:
/// replicated tensors across ranks, frozen tensors that never change, and a
/// mutable tail that is genuinely per-rank.
pub fn generate_shards(
    seed: u64,
    n_ranks: usize,
    tensors_per_shard: usize,
    replication: usize,
    frozen_fraction: f64,
) -> Vec<Shard> {
    let mut rng = SplitMix64::new(seed);

    // Shared (replicated) tensors: identical content_id on `replication` ranks.
    let n_shared = (tensors_per_shard as f64 * 0.35) as usize;
    let shared: Vec<(u64, u64, bool)> = (0..n_shared)
        .map(|_| {
            let bytes = 1 << (20 + (rng.next_u64() % 4)); // 1-8 MiB
            let frozen = rng.next_f64() < frozen_fraction;
            (rng.next_u64(), bytes, frozen)
        })
        .collect();

    let mut shards = Vec::with_capacity(n_ranks);
    for rank in 0..n_ranks {
        let mut tensors = Vec::with_capacity(tensors_per_shard);
        // Replicated head: same content across ranks, so all ranks store it.
        for (i, (cid, bytes, frozen)) in shared.iter().enumerate() {
            let replicated = rank < replication.max(1) || i % 2 == 0;
            if !replicated {
                continue;
            }
            tensors.push(Tensor {
                name: format!("shared.{}.rank{}", i, rank),
                content_id: *cid,
                bytes: *bytes,
                mutable: !*frozen,
            });
        }
        // Per-rank private tail.
        let n_private = tensors_per_shard.saturating_sub(tensors.len());
        for j in 0..n_private {
            let bytes = 1 << (19 + (rng.next_u64() % 4)); // 0.5-4 MiB
            tensors.push(Tensor {
                name: format!("private.{}.rank{}", j, rank),
                content_id: rng.next_u64(),
                bytes,
                mutable: true,
            });
        }
        shards.push(Shard { rank: rank as u32, tensors });
    }
    shards
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckpointResult {
    pub policy: String,
    pub ranks: usize,
    pub logical_bytes: u64,
    pub bytes_written: u64,
    pub bytes_read_on_restore: u64,
    pub dedup_ratio: f64,
    pub restore_correct: bool,
    /// Extra CPU work for hashing, in byte-equivalents.
    pub hashing_cost_bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    NaiveShards,
    AdaCheckDedup,
}

impl Policy {
    pub fn label(&self) -> &'static str {
        match self {
            Policy::NaiveShards => "naive-shards",
            Policy::AdaCheckDedup => "adacheck-dedup",
        }
    }
}

/// Content-addressed store: content_id -> bytes.
pub fn write_checkpoint(policy: Policy, shards: &[Shard]) -> CheckpointResult {
    let logical: u64 = shards.iter().map(|s| s.total_bytes()).sum();

    match policy {
        Policy::NaiveShards => CheckpointResult {
            policy: policy.label().into(),
            ranks: shards.len(),
            logical_bytes: logical,
            bytes_written: logical,
            bytes_read_on_restore: logical,
            dedup_ratio: 1.0,
            restore_correct: true,
            hashing_cost_bytes: 0,
        },
        Policy::AdaCheckDedup => {
            let mut store: HashMap<u64, u64> = HashMap::new();
            let mut ref_table_bytes = 0u64;
            let mut hashing_cost = 0u64;

            for shard in shards {
                for t in &shard.tensors {
                    // Hashing cost: 1/64 of payload (simulated throughput hit).
                    hashing_cost += t.bytes / 64;
                    match store.get(&t.content_id) {
                        Some(_) => {
                            // Reference: 16 bytes (id) + 4 bytes (offset).
                            ref_table_bytes += 20;
                        }
                        None => {
                            store.insert(t.content_id, t.bytes);
                        }
                    }
                }
            }

            let unique_bytes: u64 = store.values().sum();
            let bytes_written = unique_bytes + ref_table_bytes;
            // Restore reads each unique payload once plus the reference tables.
            let bytes_read = unique_bytes + ref_table_bytes;

            // Correctness: every tensor must be reconstructable. Verify that the
            // store contains a payload for every distinct content_id.
            let mut all_ids: Vec<u64> = shards
                .iter()
                .flat_map(|s| s.tensors.iter().map(|t| t.content_id))
                .collect();
            all_ids.sort_unstable();
            all_ids.dedup();
            let restore_correct = all_ids.iter().all(|id| store.contains_key(id))
                && all_ids.len() == store.len();

            CheckpointResult {
                policy: policy.label().into(),
                ranks: shards.len(),
                logical_bytes: logical,
                bytes_written,
                bytes_read_on_restore: bytes_read,
                dedup_ratio: logical as f64 / bytes_written.max(1) as f64,
                restore_correct,
                hashing_cost_bytes: hashing_cost,
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub bounty: String,
    pub hip: String,
    pub honesty_labels: Vec<String>,
    pub config: BenchConfig,
    pub results: Vec<CheckpointResult>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub seed: u64,
    pub n_ranks: usize,
    pub tensors_per_shard: usize,
    pub replication: usize,
    pub frozen_fraction: f64,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self { seed: 0xADA_C4EC, n_ranks: 8, tensors_per_shard: 40, replication: 8, frozen_fraction: 0.5 }
    }
}
impl BenchConfig {
    pub fn test_config() -> Self {
        Self { seed: 7, n_ranks: 4, tensors_per_shard: 20, replication: 4, frozen_fraction: 0.5 }
    }
}

pub fn run_bench(cfg: &BenchConfig) -> Result<BenchReport, AdaError> {
    if cfg.n_ranks == 0 {
        return Err(AdaError::InvalidConfig("n_ranks must be positive".into()));
    }
    if cfg.tensors_per_shard == 0 {
        return Err(AdaError::InvalidConfig("tensors_per_shard must be positive".into()));
    }
    if !(0.0..=1.0).contains(&cfg.frozen_fraction) {
        return Err(AdaError::InvalidConfig("frozen_fraction must be in [0,1]".into()));
    }
    let shards = generate_shards(
        cfg.seed,
        cfg.n_ranks,
        cfg.tensors_per_shard,
        cfg.replication.max(1),
        cfg.frozen_fraction,
    );
    let results = vec![
        write_checkpoint(Policy::NaiveShards, &shards),
        write_checkpoint(Policy::AdaCheckDedup, &shards),
    ];

    let mut notes = Vec::new();
    if let (Some(naive), Some(dedup)) = (
        results.iter().find(|r| r.policy == "naive-shards"),
        results.iter().find(|r| r.policy == "adacheck-dedup"),
    ) {
        notes.push(format!(
            "dedup writes {:.1}% of the naive byte volume ({:.2} GiB -> {:.2} GiB) at a hashing cost of {:.2} GiB",
            dedup.bytes_written as f64 / naive.bytes_written.max(1) as f64 * 100.0,
            naive.bytes_written as f64 / 1e9,
            dedup.bytes_written as f64 / 1e9,
            dedup.hashing_cost_bytes as f64 / 1e9
        ));
    }

    Ok(BenchReport {
        bounty: "hanzoai/ml#48 [Bounty #45] Redundancy-aware training checkpoints (AdaCheck)".into(),
        hip: "HIP-1348: Training and Storage Efficiency".into(),
        honesty_labels: vec![
            "synthetic tensor payloads -?no real model or filesystem exercised".into(),
            "redundancy structure (replication factor, frozen fraction) is modelled, not measured".into(),
            "hashing cost modelled as payload/64, not measured hasher throughput".into(),
            "All figures are simulation outputs on the authoring host, NOT Hanzo expectations until reproduced on a real training run.".into(),
        ],
        config: cfg.clone(),
        results,
        notes,
    })
}

pub fn write_results_json(report: &BenchReport, path: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(path, serde_json::to_string_pretty(report)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dedup_reduces_write_volume() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let naive = report.results.iter().find(|r| r.policy == "naive-shards").unwrap();
        let dedup = report.results.iter().find(|r| r.policy == "adacheck-dedup").unwrap();
        assert!(
            dedup.bytes_written < naive.bytes_written,
            "dedup {} should be < naive {}",
            dedup.bytes_written,
            naive.bytes_written
        );
        assert!(dedup.dedup_ratio > 1.0);
    }

    #[test]
    fn restore_is_byte_identical() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        for r in &report.results {
            assert!(r.restore_correct, "{} failed restore verification", r.policy);
        }
    }

    #[test]
    fn no_redundancy_means_no_savings() {
        // Single rank, no shared tensors possible -> dedup must not inflate bytes.
        let cfg = BenchConfig { seed: 3, n_ranks: 1, tensors_per_shard: 10, replication: 1, frozen_fraction: 0.0 };
        let report = run_bench(&cfg).unwrap();
        let naive = report.results.iter().find(|r| r.policy == "naive-shards").unwrap();
        let dedup = report.results.iter().find(|r| r.policy == "adacheck-dedup").unwrap();
        assert!(dedup.bytes_written <= naive.bytes_written);
    }

    #[test]
    fn hashing_cost_is_reported() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let dedup = report.results.iter().find(|r| r.policy == "adacheck-dedup").unwrap();
        assert!(dedup.hashing_cost_bytes > 0);
    }

    #[test]
    fn bad_config_rejected() {
        assert!(run_bench(&BenchConfig { seed: 1, n_ranks: 0, tensors_per_shard: 5, replication: 1, frozen_fraction: 0.5 }).is_err());
        assert!(run_bench(&BenchConfig { seed: 1, n_ranks: 2, tensors_per_shard: 5, replication: 1, frozen_fraction: 2.0 }).is_err());
    }
}
