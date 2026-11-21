// SPDX-License-Identifier: Apache-2.0
//! [Bounty #46] (`hanzoai/ml#49`): **Training data-pipeline cache allocation** (Seneca).
//!
//! Clean-room simulation harness for splitting a fixed cache budget across the
//! stages of a training data pipeline so that stalls are minimised.
//!
//! # Design
//!
//! A pipeline has stages (`decode`, `transform`, `tokenize`, `collate`). Each
//! stage has its own working set with a Zipf-like popularity distribution and a
//! different per-miss stall cost (decode is expensive, collate is cheap).
//!
//! - `UniformSplit`: divide the budget equally across stages (baseline).
//! - `SenecaGreedy`: repeatedly give the next cache slot to whichever stage has
//!   the highest marginal stall reduction, i.e. `miss_cost * -hit_rate`.
//!
//! Metrics: total stalls, hit rate per stage, and effective throughput.
//!
//! # Honest boundaries
//! - Synthetic popularity distributions; no real dataset or pipeline exercised.
//! - Stall costs are per-stage constants, not measured latencies.
//! - All figures are simulation outputs on the authoring host, NOT Hanzo
//!   expectations until reproduced against a real data loader.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum SenecaError {
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

/// One pipeline stage.
#[derive(Debug, Clone)]
pub struct Stage {
    pub name: String,
    /// Number of distinct items in this stage's working set.
    pub working_set: usize,
    /// Zipf exponent: higher = more skewed popularity.
    pub skew: f64,
    /// Stall cost per cache miss (arbitrary time units).
    pub miss_cost: f64,
    /// Relative frequency with which this stage is exercised.
    pub weight: f64,
}

pub fn default_stages() -> Vec<Stage> {
    vec![
        Stage { name: "decode".into(), working_set: 4_000, skew: 1.1, miss_cost: 8.0, weight: 1.0 },
        Stage { name: "transform".into(), working_set: 12_000, skew: 0.9, miss_cost: 3.0, weight: 1.0 },
        Stage { name: "tokenize".into(), working_set: 20_000, skew: 0.7, miss_cost: 1.5, weight: 1.0 },
        Stage { name: "collate".into(), working_set: 8_000, skew: 1.3, miss_cost: 0.5, weight: 1.0 },
    ]
}

/// Cumulative generalised harmonic sums: `cum[k]` = sum_{i=1..k} 1/i^skew.
/// Built once per stage so the greedy allocation is O(1) per candidate slot.
pub fn cumulative_harmonic(n: usize, skew: f64) -> Vec<f64> {
    let mut cum = Vec::with_capacity(n + 1);
    cum.push(0.0);
    let mut acc = 0.0;
    for i in 1..=n {
        acc += 1.0 / (i as f64).powf(skew);
        cum.push(acc);
    }
    cum
}

/// Zipf hit rate for a cache holding the `k` most popular of `n` items.
///
/// h(k) = H(k, s) / H(n, s) where H is the generalised harmonic number.
/// This convenience form is O(n); hot paths use [`cumulative_harmonic`].
pub fn zipf_hit_rate(k: usize, n: usize, skew: f64) -> f64 {
    if n == 0 || k == 0 {
        return 0.0;
    }
    let k = k.min(n);
    let cum = cumulative_harmonic(n, skew);
    let total = *cum.last().unwrap_or(&0.0);
    if total <= 0.0 {
        0.0
    } else {
        (cum[k] / total).clamp(0.0, 1.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    UniformSplit,
    SenecaGreedy,
}

impl Policy {
    pub fn label(&self) -> &'static str {
        match self {
            Policy::UniformSplit => "uniform-split",
            Policy::SenecaGreedy => "seneca-greedy",
        }
    }

    /// Return the per-stage cache allocation summing to `budget`.
    pub fn allocate(&self, stages: &[Stage], budget: usize) -> Vec<usize> {
        let n = stages.len();
        if n == 0 {
            return Vec::new();
        }
        match self {
            Policy::UniformSplit => {
                let each = budget / n;
                let mut alloc = vec![each; n];
                // Hand out the remainder deterministically.
                for i in 0..(budget % n) {
                    alloc[i] += 1;
                }
                alloc
            }
            Policy::SenecaGreedy => {
                // Precompute cumulative harmonic sums so each candidate slot
                // costs O(1) instead of O(working_set).
                let cums: Vec<Vec<f64>> = stages
                    .iter()
                    .map(|s| cumulative_harmonic(s.working_set, s.skew))
                    .collect();
                let totals: Vec<f64> = cums
                    .iter()
                    .map(|c| *c.last().unwrap_or(&0.0))
                    .collect();
                let hit_at = |stage: usize, k: usize| -> f64 {
                    if totals[stage] <= 0.0 {
                        0.0
                    } else {
                        (cums[stage][k.min(stages[stage].working_set)] / totals[stage]).clamp(0.0, 1.0)
                    }
                };

                let mut alloc = vec![0usize; n];
                let mut hit = vec![0.0f64; n];
                for _ in 0..budget {
                    let mut best = 0usize;
                    let mut best_gain = f64::NEG_INFINITY;
                    for (i, s) in stages.iter().enumerate() {
                        if alloc[i] >= s.working_set {
                            continue;
                        }
                        let next = hit_at(i, alloc[i] + 1);
                        let gain = (next - hit[i]) * s.miss_cost * s.weight;
                        if gain > best_gain {
                            best_gain = gain;
                            best = i;
                        }
                    }
                    if !best_gain.is_finite() || best_gain <= 0.0 {
                        break;
                    }
                    alloc[best] += 1;
                    hit[best] = hit_at(best, alloc[best]);
                }
                alloc
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageStats {
    pub stage: String,
    pub cache_slots: usize,
    pub hit_rate: f64,
    pub stalls: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub policy: String,
    pub budget: usize,
    pub total_stalls: f64,
    pub effective_throughput: f64,
    pub per_stage: Vec<StageStats>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub bounty: String,
    pub hip: String,
    pub honesty_labels: Vec<String>,
    pub config: BenchConfig,
    pub results: Vec<PolicyResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub seed: u64,
    pub budget: usize,
    /// Total items processed in the simulated run.
    pub n_items: usize,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self { seed: 0x5E_ECA, budget: 8_000, n_items: 200_000 }
    }
}
impl BenchConfig {
    pub fn test_config() -> Self {
        Self { seed: 7, budget: 2_000, n_items: 20_000 }
    }
}

pub fn evaluate(policy: Policy, stages: &[Stage], cfg: &BenchConfig) -> PolicyResult {
    let alloc = policy.allocate(stages, cfg.budget);
    let mut per_stage = Vec::with_capacity(stages.len());
    let mut total_stalls = 0.0f64;

    for (i, s) in stages.iter().enumerate() {
        let slots = alloc.get(i).copied().unwrap_or(0);
        let hit = zipf_hit_rate(slots, s.working_set, s.skew);
        let accesses = cfg.n_items as f64 * s.weight;
        let stalls = accesses * (1.0 - hit) * s.miss_cost;
        total_stalls += stalls;
        per_stage.push(StageStats {
            stage: s.name.clone(),
            cache_slots: slots,
            hit_rate: hit,
            stalls,
        });
    }

    // Throughput model: baseline cost per item is 1.0; stalls add to it.
    let base = cfg.n_items as f64;
    let effective_throughput = base / (base + total_stalls).max(1e-9);

    PolicyResult {
        policy: policy.label().into(),
        budget: cfg.budget,
        total_stalls,
        effective_throughput,
        per_stage,
    }
}

pub fn run_bench(cfg: &BenchConfig) -> Result<BenchReport, SenecaError> {
    if cfg.budget == 0 {
        return Err(SenecaError::InvalidConfig("budget must be positive".into()));
    }
    if cfg.n_items == 0 {
        return Err(SenecaError::InvalidConfig("n_items must be positive".into()));
    }
    let stages = default_stages();
    let results = vec![
        evaluate(Policy::UniformSplit, &stages, cfg),
        evaluate(Policy::SenecaGreedy, &stages, cfg),
    ];

    Ok(BenchReport {
        bounty: "hanzoai/ml#49 [Bounty #46] Training data-pipeline cache allocation (Seneca)".into(),
        hip: "HIP-1348: Training and Storage Efficiency".into(),
        honesty_labels: vec![
            "synthetic Zipf popularity per stage -?no real dataset or pipeline exercised".into(),
            "stall costs are per-stage constants, not measured latencies".into(),
            "throughput model is additive cost, not a measured step time".into(),
            "All figures are simulation outputs on the authoring host, NOT Hanzo expectations until reproduced against a real data loader.".into(),
        ],
        config: cfg.clone(),
        results,
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
    fn allocations_respect_budget() {
        let stages = default_stages();
        for p in [Policy::UniformSplit, Policy::SenecaGreedy] {
            let alloc = p.allocate(&stages, 1_000);
            let sum: usize = alloc.iter().sum();
            assert_eq!(sum, 1_000, "{} allocated {} of 1000", p.label(), sum);
        }
    }

    #[test]
    fn greedy_beats_uniform() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let uni = report.results.iter().find(|r| r.policy == "uniform-split").unwrap();
        let greedy = report.results.iter().find(|r| r.policy == "seneca-greedy").unwrap();
        assert!(
            greedy.total_stalls < uni.total_stalls,
            "greedy stalls {:.0} should be < uniform {:.0}",
            greedy.total_stalls,
            uni.total_stalls
        );
        assert!(greedy.effective_throughput >= uni.effective_throughput);
    }

    #[test]
    fn greedy_prioritises_expensive_stage() {
        let stages = default_stages();
        let alloc = Policy::SenecaGreedy.allocate(&stages, 4_000);
        // decode has the highest miss cost, so it must get the most slots.
        let max_idx = alloc.iter().enumerate().max_by(|a, b| a.1.cmp(b.1)).unwrap().0;
        assert_eq!(stages[max_idx].name, "decode", "alloc={:?}", alloc);
    }

    #[test]
    fn hit_rate_is_monotone() {
        let mut prev = 0.0;
        for k in [0, 1, 10, 100, 1_000, 4_000] {
            let h = zipf_hit_rate(k, 4_000, 1.1);
            assert!(h >= prev, "hit rate decreased at k={}", k);
            prev = h;
        }
        assert!((zipf_hit_rate(4_000, 4_000, 1.1) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn bad_config_rejected() {
        assert!(run_bench(&BenchConfig { seed: 1, budget: 0, n_items: 100 }).is_err());
        assert!(run_bench(&BenchConfig { seed: 1, budget: 100, n_items: 0 }).is_err());
    }
}
