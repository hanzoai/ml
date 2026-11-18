// SPDX-License-Identifier: Apache-2.0
//! [Bounty #27] (`hanzoai/ml#47`): **Retrieval-aware fine-tuning** (RAFT).
//!
//! Clean-room simulation harness for fine-tuning a model on the *retrieval
//! distribution it will actually see*, including distractor passages, instead of
//! on clean question/answer pairs.
//!
//! # Design
//!
//! At inference the model always receives `k` retrieved passages of which only
//! some are relevant. A model trained on clean QA pairs never learns to ignore
//! the distractors and degrades as `k` grows.
//!
//! - `SftClean`: fine-tune on question -> answer with no context.
//! - `Raft`: fine-tune on question + `k` passages (with distractors) and train
//!   the model to cite the supporting passage before answering.
//!
//! Metrics: answer accuracy vs number of passages, citation accuracy, distractor
//! robustness (accuracy retained from k=1 to k=10), and training tokens.
//!
//! # Honest boundaries
//! - No model is trained; accuracy curves are modelled functions of context
//!   quality and distractor load.
//! - The citation and distractor-robustness gains are hand-set parameters, not
//!   measured from a real fine-tune.
//! - Token counts are approximations of a real SFT dataset.
//! - All figures are simulation outputs on the authoring host, NOT Hanzo
//!   expectations until reproduced with a real fine-tune and retriever.

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum RaftError {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Policy {
    /// Clean question/answer SFT, no retrieval context during training.
    SftClean,
    /// Retrieval-augmented fine-tuning with distractors and citation targets.
    Raft,
}

impl Policy {
    pub fn label(&self) -> &'static str {
        match self {
            Policy::SftClean => "sft-clean",
            Policy::Raft => "raft",
        }
    }

    /// Accuracy at k retrieved passages, given the fraction that are relevant.
    pub fn accuracy_at(&self, k: usize, relevant_fraction: f64) -> f64 {
        let k = k.max(1) as f64;
        // More passages = more chances a relevant one is present, but also more
        // distractors to be confused by.
        let recall = 1.0 - (1.0 - relevant_fraction).powf(k);
        let distractor_load = (k - 1.0) * (1.0 - relevant_fraction);
        match self {
            // Clean SFT: never learned to ignore distractors, so it degrades
            // roughly linearly with distractor load.
            Policy::SftClean => {
                let base = 0.34 + 0.42 * recall;
                (base - 0.021 * distractor_load).clamp(0.0, 1.0)
            }
            // RAFT: trained with distractors and citation supervision, so the
            // degradation is much flatter.
            Policy::Raft => {
                let base = 0.50 + 0.45 * recall;
                (base - 0.004 * distractor_load).clamp(0.0, 1.0)
            }
        }
    }

    /// Probability the model cites the actually-supporting passage.
    pub fn citation_accuracy(&self) -> f64 {
        match self {
            Policy::SftClean => 0.0, // never trained to cite
            Policy::Raft => 0.88,
        }
    }

    /// Training tokens per example.
    pub fn tokens_per_example(&self, k: usize, passage_tokens: usize) -> usize {
        match self {
            Policy::SftClean => 120,
            Policy::Raft => 120 + k.max(1) * passage_tokens,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KPoint {
    pub k: usize,
    pub accuracy: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyResult {
    pub policy: String,
    pub accuracy_by_k: Vec<KPoint>,
    /// Accuracy at k=10 with the configured distractor load, relative to k=10
    /// with no distractors. 1.0 means distractors cost nothing.
    pub distractor_robustness: f64,
    pub citation_accuracy: f64,
    pub training_tokens: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchReport {
    pub bounty: String,
    pub hip: String,
    pub honesty_labels: Vec<String>,
    pub config: BenchConfig,
    pub results: Vec<PolicyResult>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchConfig {
    pub seed: u64,
    pub n_train_examples: usize,
    pub relevant_fraction: f64,
    pub passage_tokens: usize,
    pub k_values: Vec<usize>,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            seed: 0x4AF_700,
            n_train_examples: 20_000,
            relevant_fraction: 0.3,
            passage_tokens: 180,
            k_values: vec![1, 2, 4, 8, 10],
        }
    }
}
impl BenchConfig {
    pub fn test_config() -> Self {
        Self {
            seed: 7,
            n_train_examples: 1_000,
            relevant_fraction: 0.3,
            passage_tokens: 180,
            k_values: vec![1, 2, 4, 8, 10],
        }
    }
}

pub fn evaluate(policy: Policy, cfg: &BenchConfig) -> PolicyResult {
    let accuracy_by_k: Vec<KPoint> = cfg
        .k_values
        .iter()
        .map(|&k| KPoint { k, accuracy: policy.accuracy_at(k, cfg.relevant_fraction) })
        .collect();

    let at10 = accuracy_by_k.iter().find(|p| p.k == 10).map(|p| p.accuracy).unwrap_or(0.0);
    // Robustness = accuracy at k=10 with the configured distractor load,
    // relative to the same k with no distractors at all.
    let clean_k10 = policy.accuracy_at(10, 1.0);

    PolicyResult {
        policy: policy.label().into(),
        accuracy_by_k,
        distractor_robustness: if clean_k10 > 0.0 { at10 / clean_k10 } else { 0.0 },
        citation_accuracy: policy.citation_accuracy(),
        training_tokens: cfg.n_train_examples * policy.tokens_per_example(10, cfg.passage_tokens),
    }
}

pub fn run_bench(cfg: &BenchConfig) -> Result<BenchReport, RaftError> {
    if cfg.n_train_examples == 0 {
        return Err(RaftError::InvalidConfig("n_train_examples must be positive".into()));
    }
    if !(0.0..=1.0).contains(&cfg.relevant_fraction) {
        return Err(RaftError::InvalidConfig("relevant_fraction must be in [0,1]".into()));
    }
    if cfg.passage_tokens == 0 {
        return Err(RaftError::InvalidConfig("passage_tokens must be positive".into()));
    }
    if cfg.k_values.is_empty() {
        return Err(RaftError::InvalidConfig("k_values must not be empty".into()));
    }
    let results = vec![evaluate(Policy::SftClean, cfg), evaluate(Policy::Raft, cfg)];

    let mut notes = Vec::new();
    if let (Some(clean), Some(raft)) = (
        results.iter().find(|r| r.policy == "sft-clean"),
        results.iter().find(|r| r.policy == "raft"),
    ) {
        let acc = |r: &PolicyResult, k: usize| {
            r.accuracy_by_k.iter().find(|p| p.k == k).map(|p| p.accuracy).unwrap_or(0.0)
        };
        notes.push(format!(
            "at k=10 RAFT holds {:.1}% accuracy vs {:.1}% for clean SFT; distractor robustness {:.2} vs {:.2}",
            acc(raft, 10) * 100.0,
            acc(clean, 10) * 100.0,
            raft.distractor_robustness,
            clean.distractor_robustness
        ));
        let tok = |n: usize| {
            let n = n as f64;
            if n >= 1e8 {
                format!("{:.2}B", n / 1e9)
            } else {
                format!("{:.1}M", n / 1e6)
            }
        };
        notes.push(format!(
            "RAFT trains on {:.1}x more tokens ({} vs {})",
            raft.training_tokens as f64 / clean.training_tokens.max(1) as f64,
            tok(raft.training_tokens),
            tok(clean.training_tokens)
        ));
    }

    Ok(BenchReport {
        bounty: "hanzoai/ml#47 [Bounty #27] Retrieval-aware Zen fine-tuning (RAFT)".into(),
        hip: "HIP-1348: Reliable, Economical and Verifiable AI Infrastructure".into(),
        honesty_labels: vec![
            "no model is trained - accuracy curves are modelled functions of context quality and distractor load".into(),
            "citation and robustness gains are hand-set parameters, not measured from a real fine-tune".into(),
            "token counts approximate a real SFT dataset".into(),
            "All figures are simulation outputs on the authoring host, NOT Hanzo expectations until reproduced with a real fine-tune and retriever.".into(),
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

    fn find<'a>(r: &'a [PolicyResult], p: &str) -> &'a PolicyResult {
        r.iter().find(|x| x.policy == p).expect(p)
    }

    #[test]
    fn raft_is_more_robust_to_distractors() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        let clean = find(&report.results, "sft-clean");
        let raft = find(&report.results, "raft");
        assert!(
            raft.distractor_robustness > clean.distractor_robustness,
            "raft {:.3} should beat clean {:.3}",
            raft.distractor_robustness,
            clean.distractor_robustness
        );
    }

    #[test]
    fn raft_has_nonzero_citation_accuracy() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        assert_eq!(find(&report.results, "sft-clean").citation_accuracy, 0.0);
        assert!(find(&report.results, "raft").citation_accuracy > 0.0);
    }

    #[test]
    fn raft_uses_more_training_tokens() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        assert!(find(&report.results, "raft").training_tokens > find(&report.results, "sft-clean").training_tokens);
    }

    #[test]
    fn accuracy_is_monotone_in_recall_when_all_relevant() {
        let cfg = BenchConfig { relevant_fraction: 1.0, ..BenchConfig::test_config() };
        // Sanity: the config is actually accepted.
        assert!(run_bench(&cfg).is_ok());
        for p in [Policy::SftClean, Policy::Raft] {
            let a1 = p.accuracy_at(1, cfg.relevant_fraction);
            let a4 = p.accuracy_at(4, cfg.relevant_fraction);
            assert!(a4 >= a1 - 1e-9, "{}: a4 {:.3} < a1 {:.3}", p.label(), a4, a1);
        }
    }

    #[test]
    fn accuracy_stays_in_unit_interval() {
        let report = run_bench(&BenchConfig::test_config()).unwrap();
        for r in &report.results {
            for p in &r.accuracy_by_k {
                assert!(p.accuracy >= 0.0 && p.accuracy <= 1.0);
            }
        }
    }

    #[test]
    fn bad_config_rejected() {
        let mut cfg = BenchConfig::test_config();
        cfg.n_train_examples = 0;
        assert!(run_bench(&cfg).is_err());
        let mut cfg = BenchConfig::test_config();
        cfg.relevant_fraction = 2.0;
        assert!(run_bench(&cfg).is_err());
        let mut cfg = BenchConfig::test_config();
        cfg.k_values.clear();
        assert!(run_bench(&cfg).is_err());
    }
}
