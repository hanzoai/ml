//! Automatic 4-axis transfer hypothesis generator and dead-end registry.

use serde::{Deserialize, Serialize};

use crate::protocol::{epistemic_verdict, ExperimentRecordV2, Opportunity};
use crate::wire::now;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TransferCatalog {
    pub model_scales: Vec<String>,
    pub sequence_lengths: Vec<usize>,
    pub backends: Vec<String>,
    pub workload_regimes: Vec<String>,
}

impl Default for TransferCatalog {
    fn default() -> Self {
        Self {
            model_scales: vec![
                "Zen-Nano-0.6B".into(),
                "Bonsai-2-2B".into(),
                "Bonsai-2-9B".into(),
                "Bonsai-2-27B".into(),
                "Qwen3.8-27B".into(),
            ],
            sequence_lengths: vec![512, 4096, 8192, 16384, 32768],
            backends: vec![
                "CUDA-GB10-sm121a".into(),
                "Vulkan-Wave32".into(),
                "Vulkan-Wave64".into(),
                "ROCm-gfx1151".into(),
                "Metal-M4Max".into(),
            ],
            workload_regimes: vec![
                "single-turn-decode".into(),
                "multiturn-agent-prefix".into(),
                "speculative-dflash2".into(),
                "batched-serving".into(),
            ],
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct DeadEndEntry {
    pub experiment_id: String,
    pub study: String,
    pub repo: String,
    pub commit: String,
    pub backend: String,
    pub shapes: Vec<String>,
    pub failure_diagnosis: String,
    pub next_hypothesis: String,
    pub recorded_at: String,
}

fn round4(x: f64) -> f64 {
    (x * 10000.0).round() / 10000.0
}

fn relative_gain(exp: &ExperimentRecordV2) -> f64 {
    let sc = &exp.statistical_comparison;
    if let Some(r) = sc.ratio.filter(|&r| r > 0.0) {
        let g = if sc.higher_is_better {
            r - 1.0
        } else {
            1.0 / r - 1.0
        };
        if g > 0.0 {
            return round4(g);
        }
    }
    let base = sc.baseline_mean.abs();
    if base > 1e-9 {
        let raw = if sc.higher_is_better {
            sc.candidate_mean - sc.baseline_mean
        } else {
            sc.baseline_mean - sc.candidate_mean
        };
        if raw > 0.0 {
            return round4(raw / base);
        }
    }
    0.10
}

fn match_next_scale(exp: &ExperimentRecordV2, scales: &[String]) -> Option<String> {
    let haystack = format!(
        "{} {} {} {}",
        exp.experiment_id,
        exp.study,
        exp.workload_manifest.dataset,
        exp.workload_manifest.manifest_id
    )
    .to_ascii_lowercase();
    for (idx, scale) in scales.iter().enumerate() {
        let tok = scale.to_ascii_lowercase();
        let short = tok.rsplit('-').next().unwrap_or("");
        if haystack.contains(&tok) || (short.ends_with('b') && haystack.contains(short)) {
            if idx + 1 < scales.len() {
                return Some(scales[idx + 1].clone());
            }
            return None;
        }
    }
    scales
        .iter()
        .find(|s| !haystack.contains(&s.to_ascii_lowercase()))
        .cloned()
}

fn match_next_seq_len(exp: &ExperimentRecordV2, lengths: &[usize]) -> Option<usize> {
    let cur = exp
        .workload_manifest
        .input_lengths
        .iter()
        .copied()
        .max()
        .unwrap_or(512);
    let mut sorted = lengths.to_vec();
    sorted.sort_unstable();
    sorted.into_iter().find(|&len| len > cur)
}

fn match_sibling_backends(exp: &ExperimentRecordV2, backends: &[String]) -> Vec<String> {
    let cur_b = exp.hardware.backend.to_ascii_lowercase();
    let cur_g = exp.hardware.gpu.to_ascii_lowercase();
    let mut out = Vec::new();
    for target in backends {
        let tl = target.to_ascii_lowercase();
        let prefix = tl.split('-').next().unwrap_or(&tl);
        if !cur_b.is_empty() && cur_b.contains(prefix) {
            continue;
        }
        if !cur_g.is_empty() && cur_g.contains(prefix) {
            continue;
        }
        out.push(target.clone());
        if out.len() == 2 {
            break;
        }
    }
    out
}

fn match_next_regime(exp: &ExperimentRecordV2, regimes: &[String]) -> Option<String> {
    let haystack = format!(
        "{} {} {}",
        exp.study, exp.workload_manifest.split, exp.experiment_id
    )
    .to_ascii_lowercase();
    for (idx, reg) in regimes.iter().enumerate() {
        let rl = reg.to_ascii_lowercase();
        let prefix = rl.split('-').next().unwrap_or(&rl);
        if haystack.contains(&rl) || haystack.contains(prefix) {
            if idx + 1 < regimes.len() {
                return Some(regimes[idx + 1].clone());
            }
            return None;
        }
    }
    regimes
        .iter()
        .find(|r| !haystack.contains(&r.to_ascii_lowercase()))
        .cloned()
}

pub struct TransferEngine {
    pub catalog: TransferCatalog,
}

impl TransferEngine {
    pub fn new(catalog: Option<TransferCatalog>) -> Self {
        Self {
            catalog: catalog.unwrap_or_default(),
        }
    }

    pub fn dead_end_from(&self, exp: &ExperimentRecordV2) -> Option<DeadEndEntry> {
        if exp.is_retracted() || epistemic_verdict(&exp.statistical_comparison.verdict) != "refuted"
        {
            return None;
        }
        Some(DeadEndEntry {
            experiment_id: exp.experiment_id.clone(),
            study: exp.study.clone(),
            repo: exp.repo.clone(),
            commit: exp.commit.clone(),
            backend: if exp.hardware.backend.is_empty() {
                "CPU".into()
            } else {
                exp.hardware.backend.clone()
            },
            shapes: exp.workload_manifest.shapes.clone(),
            failure_diagnosis: if exp.failure_diagnosis.is_empty() {
                exp.conclusion.clone()
            } else {
                exp.failure_diagnosis.clone()
            },
            next_hypothesis: exp.next_hypothesis.clone(),
            recorded_at: if exp.recorded_at.is_empty() {
                now()
            } else {
                exp.recorded_at.clone()
            },
        })
    }

    pub fn generate(&self, exp: &ExperimentRecordV2) -> Vec<Opportunity> {
        if exp.is_retracted() {
            return Vec::new();
        }
        let raw_verdict = exp
            .statistical_comparison
            .verdict
            .trim()
            .to_ascii_lowercase();
        let base_cost = if exp.gpu_time_sec > 0.0 {
            (exp.gpu_time_sec / 60.0).max(0.5)
        } else {
            1.0
        };

        if raw_verdict == "promising_inconclusive" {
            let cur_cont = exp
                .config
                .get("continuation_depth")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(0) as usize;
            let max_cont = exp
                .config
                .get("max_continuations")
                .and_then(serde_json::Value::as_u64)
                .unwrap_or(1) as usize;
            if cur_cont >= max_cont {
                return Vec::new();
            }
            let next_cont = cur_cont + 1;
            let gain = relative_gain(exp);
            return vec![Opportunity {
                opportunity_id: format!("{}->dev-continuation:{next_cont}", exp.experiment_id),
                domain: exp.study.clone(),
                objective: format!("{}:dev-continuation-{next_cont}", exp.study),
                current_best: exp.statistical_comparison.candidate_mean,
                proposed_approach: format!(
                    "Promising inconclusive dev signal (cand={:.4} vs base={:.4}); run bounded continuation {next_cont}/{max_cont} on an independent dev slice before canonical qualification",
                    exp.statistical_comparison.candidate_mean,
                    exp.statistical_comparison.baseline_mean
                ),
                expected_improvement: round4(gain),
                compute_cost: round4(base_cost),
                related_experiments: vec![exp.experiment_id.clone()],
                trigger_rule: format!(
                    "continuation_independent_dev:{next_cont}/{max_cont}:{}",
                    exp.experiment_id
                ),
                status: "open".into(),
            }];
        }

        let verdict = epistemic_verdict(&exp.statistical_comparison.verdict);

        if verdict == "refuted" {
            if exp.next_hypothesis.trim().is_empty() {
                return Vec::new();
            }
            let diag = if exp.failure_diagnosis.trim().is_empty() {
                exp.conclusion.trim()
            } else {
                exp.failure_diagnosis.trim()
            };
            return vec![Opportunity {
                opportunity_id: format!("{}->mechanism-fix", exp.experiment_id),
                domain: exp.study.clone(),
                objective: format!("{}:mechanism-fix", exp.study),
                current_best: exp.statistical_comparison.baseline_mean,
                proposed_approach: format!(
                    "Refuted mechanism diagnosis: {diag}. Falsifiable fix hypothesis: {}",
                    exp.next_hypothesis.trim()
                ),
                expected_improvement: 0.12,
                compute_cost: round4(base_cost),
                related_experiments: vec![exp.experiment_id.clone()],
                trigger_rule: format!("mechanism_fix_after_refutation:{}", exp.experiment_id),
                status: "open".into(),
            }];
        }

        if verdict != "proven" {
            return Vec::new();
        }

        let gain = relative_gain(exp);
        let mut ops = Vec::new();

        if let Some(next_scale) = match_next_scale(exp, &self.catalog.model_scales) {
            ops.push(Opportunity {
                opportunity_id: format!("{}->scale:{next_scale}", exp.experiment_id),
                domain: exp.study.clone(),
                objective: format!("{}@{next_scale}", exp.study),
                current_best: exp.statistical_comparison.candidate_mean,
                proposed_approach: format!(
                    "Test whether {} mechanism from {} preserves >={:.2}x relative gain at model scale {next_scale} with parity_passed=true",
                    exp.study,
                    exp.experiment_id,
                    gain * 0.75
                ),
                expected_improvement: round4(gain * 0.85),
                compute_cost: round4(base_cost * 1.8),
                related_experiments: vec![exp.experiment_id.clone()],
                trigger_rule: format!("transfer:model_scale:{}", exp.experiment_id),
                status: "open".into(),
            });
        }

        if let Some(next_seq) = match_next_seq_len(exp, &self.catalog.sequence_lengths) {
            ops.push(Opportunity {
                opportunity_id: format!("{}->seq:{next_seq}", exp.experiment_id),
                domain: exp.study.clone(),
                objective: format!("{}@seq{next_seq}", exp.study),
                current_best: exp.statistical_comparison.candidate_mean,
                proposed_approach: format!(
                    "Evaluate {} scaling to sequence length {next_seq} via progressive anytime-valid A/B screen without memory or precision regression",
                    exp.study
                ),
                expected_improvement: round4(gain * 0.80),
                compute_cost: round4(base_cost * 1.5),
                related_experiments: vec![exp.experiment_id.clone()],
                trigger_rule: format!("transfer:seq_len:{}", exp.experiment_id),
                status: "open".into(),
            });
        }

        for target_backend in match_sibling_backends(exp, &self.catalog.backends) {
            ops.push(Opportunity {
                opportunity_id: format!("{}->backend:{target_backend}", exp.experiment_id),
                domain: exp.study.clone(),
                objective: format!("{}@{target_backend}", exp.study),
                current_best: exp.statistical_comparison.candidate_mean,
                proposed_approach: format!(
                    "Verify {} portable kernel/runtime speedup on {target_backend} with bit-exact or bounded parity gate",
                    exp.study
                ),
                expected_improvement: round4(gain * 0.70),
                compute_cost: round4(base_cost * 1.2),
                related_experiments: vec![exp.experiment_id.clone()],
                trigger_rule: format!("transfer:backend:{}", exp.experiment_id),
                status: "open".into(),
            });
        }

        if let Some(next_regime) = match_next_regime(exp, &self.catalog.workload_regimes) {
            ops.push(Opportunity {
                opportunity_id: format!("{}->regime:{next_regime}", exp.experiment_id),
                domain: exp.study.clone(),
                objective: format!("{}@{next_regime}", exp.study),
                current_best: exp.statistical_comparison.candidate_mean,
                proposed_approach: format!(
                    "Validate {} win from {} under workload regime {next_regime} using paired anytime-valid look schedule",
                    exp.study, exp.experiment_id
                ),
                expected_improvement: round4(gain * 0.75),
                compute_cost: round4(base_cost * 1.4),
                related_experiments: vec![exp.experiment_id.clone()],
                trigger_rule: format!("transfer:workload_regime:{}", exp.experiment_id),
                status: "open".into(),
            });
        }

        if !exp.next_hypothesis.trim().is_empty() {
            ops.push(Opportunity {
                opportunity_id: format!("{}->followup", exp.experiment_id),
                domain: exp.study.clone(),
                objective: format!("{}:followup", exp.study),
                current_best: exp.statistical_comparison.candidate_mean,
                proposed_approach: exp.next_hypothesis.trim().to_string(),
                expected_improvement: round4(gain * 0.90),
                compute_cost: round4(base_cost),
                related_experiments: vec![exp.experiment_id.clone()],
                trigger_rule: format!("next_hypothesis:{}", exp.experiment_id),
                status: "open".into(),
            });
        }

        ops
    }
}

pub fn generate_transfer_hypotheses(
    exp: &ExperimentRecordV2,
    catalog: Option<TransferCatalog>,
) -> Vec<Opportunity> {
    TransferEngine::new(catalog).generate(exp)
}
