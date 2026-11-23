//! Unified first-class scientific experiment protocol, HIP-1403 records, WAL ledger, and opportunity queue.
//!
//! Rust peer of `hanzo_research.protocol`: produces compatible schemas on disk and over
//! `/v1/research` so `hanzoai/ml`, `hanzo-engine`, and `hanzo-inc/kai` share one ledger.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::canonical::rfc8785_digest;
use crate::transfer::{
    generate_transfer_hypotheses, DeadEndEntry, TransferCatalog, TransferEngine,
};
use crate::wal::ResearchWal;
use crate::wire::{now, Benchmark, Measure, Run, Study};
use crate::{Error, ExperimentRecord, Meta, Research, Result};

const T975: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];

/// Normalize statistical comparison verdict onto `/v1/research` epistemic verdict.
pub fn epistemic_verdict(verdict: &str) -> &'static str {
    match verdict.trim().to_ascii_lowercase().as_str() {
        "win" | "proven" => "proven",
        "loss" | "refuted" => "refuted",
        _ => "inconclusive",
    }
}

/// HIP-1403 resource budget cap for pre-registered experiments.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ResearchBudget {
    pub max_compute_seconds: f64,
    pub max_cost_usd: f64,
}

impl Default for ResearchBudget {
    fn default() -> Self {
        Self {
            max_compute_seconds: 3600.0,
            max_cost_usd: 25.0,
        }
    }
}

/// HIP-1403 pre-registered falsifiable hypothesis record.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HypothesisRecord {
    pub hypothesis_id: String,
    pub statement: String,
    pub null_hypothesis: String,
    pub gate_ref: String,
    pub dataset_refs: Vec<String>,
    pub budget: ResearchBudget,
    pub proposer: String,
    pub registered_at: String,
}

impl HypothesisRecord {
    pub fn with_computed_id(mut self) -> Result<Self> {
        if self.registered_at.is_empty() {
            self.registered_at = now();
        }
        if self.proposer.is_empty() {
            self.proposer = "did:hanzo:org:research-agent".into();
        }
        if self.hypothesis_id.is_empty() {
            let mut val = serde_json::to_value(&self)?;
            if let Some(obj) = val.as_object_mut() {
                obj.remove("hypothesis_id");
            }
            self.hypothesis_id = format!("sha256:{}", rfc8785_digest(&val)?);
        }
        Ok(self)
    }

    pub fn validate(&self) -> Result<()> {
        if self.statement.trim().is_empty()
            || self.null_hypothesis.trim().is_empty()
            || self.gate_ref.trim().is_empty()
        {
            return Err(Error::Message(
                "HypothesisRecord requires statement, null_hypothesis, and gate_ref".into(),
            ));
        }
        if self.budget.max_compute_seconds <= 0.0 || self.budget.max_cost_usd < 0.0 {
            return Err(Error::Message(
                "HypothesisRecord requires positive budget bounds".into(),
            ));
        }
        Ok(())
    }
}

/// HIP-1403 counter-evidence submission triggering claim retraction.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CounterEvidence {
    pub counter_evidence_id: String,
    pub claim_id: String,
    pub experiment_id: String,
    pub defect_kind: String,
    pub details: String,
    pub gate_ref: String,
    pub submitted_by: String,
    pub submitted_at: String,
}

impl CounterEvidence {
    pub fn digest(&self) -> Result<String> {
        Ok(rfc8785_digest(self)?)
    }
}

/// HIP-1403 Research Yield Y_R = sum(verified_gains) / sum(total_costs_including_failures).
pub fn research_yield(
    verified_gains: &[f64],
    total_costs_including_failures: &[f64],
) -> Result<f64> {
    if total_costs_including_failures
        .iter()
        .any(|&c| !c.is_finite() || c < 0.0)
    {
        return Err(Error::Message(
            "research costs must be non-negative finite numbers".into(),
        ));
    }
    let gains_sum: f64 = verified_gains
        .iter()
        .copied()
        .filter(|g| g.is_finite() && *g > 0.0)
        .sum();
    let total_cost: f64 = total_costs_including_failures.iter().sum();
    if total_cost <= 0.0 {
        if verified_gains.is_empty() && total_costs_including_failures.is_empty() {
            return Ok(0.0);
        }
        return Err(Error::Message(
            "total research cost must be strictly positive".into(),
        ));
    }
    Ok(gains_sum / total_cost)
}

/// HIP-1403 Successor Research Lift L_R = Y_R(successor) / Y_R(predecessor).
pub fn successor_research_lift(y_successor: f64, y_predecessor: f64) -> Result<f64> {
    if !y_successor.is_finite() || !y_predecessor.is_finite() || y_predecessor <= 0.0 {
        return Err(Error::Message(
            "y_predecessor must be finite and strictly positive".into(),
        ));
    }
    Ok(y_successor / y_predecessor)
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HardwareSpec {
    pub host: String,
    pub architecture: String,
    pub gpu: String,
    pub driver: String,
    pub compiler: String,
    pub backend: String,
    pub precision: String,
    pub memory_gb: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct WorkloadManifest {
    pub manifest_id: String,
    pub dataset: String,
    pub split: String,
    pub shapes: Vec<String>,
    pub input_lengths: Vec<usize>,
    pub batch_size: usize,
    pub seed: u64,
    pub manifest_sha256: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct StatisticalComparison {
    pub metric: String,
    pub unit: String,
    pub higher_is_better: bool,
    pub n: usize,
    pub baseline_mean: f64,
    pub candidate_mean: f64,
    pub baseline_ci95: Option<f64>,
    pub candidate_ci95: Option<f64>,
    pub delta: f64,
    pub ratio: Option<f64>,
    pub ratio_lo: Option<f64>,
    pub ratio_hi: Option<f64>,
    pub p_value: Option<f64>,
    pub anytime_valid: bool,
    pub verdict: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CorrectnessChecks {
    pub parity_passed: bool,
    pub max_abs_diff: Option<f64>,
    pub ulp_diff: Option<f64>,
    pub safety_passed: Option<bool>,
    pub memory_integrity_passed: bool,
    pub details: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExperimentRecordV2 {
    pub project: String,
    pub study: String,
    pub objective: String,
    pub hypothesis: String,
    pub hypothesis_id: String,
    pub null_hypothesis: String,
    pub gate_ref: String,
    pub experiment_id: String,
    pub parent_experiment: String,
    pub parent_ids: Vec<String>,
    pub prior_best: Option<f64>,
    pub repo: String,
    pub commit: String,
    pub branch: String,
    pub pr: String,
    pub config_digest: String,
    pub env_digest: String,
    pub hardware_class: String,
    pub artifact_hashes: BTreeMap<String, String>,
    pub artifact_refs: Vec<String>,
    pub hardware: HardwareSpec,
    pub config: BTreeMap<String, Value>,
    pub workload_manifest: WorkloadManifest,
    pub baseline: BTreeMap<String, Value>,
    pub candidate: BTreeMap<String, Value>,
    pub statistical_comparison: StatisticalComparison,
    pub correctness: CorrectnessChecks,
    pub raw_samples: BTreeMap<String, Vec<f64>>,
    pub verdict: String,
    pub gpu_time_sec: f64,
    pub compute_cost_usd: f64,
    pub conclusion: String,
    pub failure_diagnosis: String,
    pub next_hypothesis: String,
    pub historical: bool,
    pub recorded_at: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Opportunity {
    pub opportunity_id: String,
    pub domain: String,
    pub objective: String,
    pub current_best: f64,
    pub proposed_approach: String,
    pub expected_improvement: f64,
    pub compute_cost: f64,
    pub related_experiments: Vec<String>,
    pub trigger_rule: String,
    pub status: String,
}

impl Opportunity {
    pub fn priority_score(&self) -> f64 {
        self.expected_improvement / self.compute_cost.max(1e-6)
    }
}

pub struct OpportunityQueue {
    pub path: PathBuf,
    pub items: BTreeMap<String, Opportunity>,
}

impl OpportunityQueue {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let mut q = Self {
            path: path.as_ref().to_path_buf(),
            items: BTreeMap::new(),
        };
        q.reload()?;
        Ok(q)
    }

    pub fn reload(&mut self) -> Result<()> {
        self.items.clear();
        if self.path.is_file() {
            let rows: Vec<Opportunity> = serde_json::from_slice(&fs::read(&self.path)?)?;
            for op in rows {
                self.items.insert(op.opportunity_id.clone(), op);
            }
        }
        Ok(())
    }

    pub fn add(&mut self, op: Opportunity, preserve_closed: bool) -> Result<Opportunity> {
        if op.opportunity_id.is_empty() {
            return Err(Error::Message("opportunity_id required".into()));
        }
        if preserve_closed {
            if let Some(existing) = self.items.get(&op.opportunity_id) {
                if matches!(
                    existing.status.as_str(),
                    "completed" | "refuted" | "skipped"
                ) {
                    return Ok(existing.clone());
                }
            }
        }
        self.items.insert(op.opportunity_id.clone(), op.clone());
        self.save()?;
        Ok(op)
    }

    pub fn mark_status(
        &mut self,
        opportunity_id: &str,
        status: &str,
    ) -> Result<Option<Opportunity>> {
        let Some(op) = self.items.get_mut(opportunity_id) else {
            return Ok(None);
        };
        op.status = status.to_string();
        let out = op.clone();
        self.save()?;
        Ok(Some(out))
    }

    pub fn top_open(&self, domain: Option<&str>, top_k: usize) -> Vec<Opportunity> {
        let mut ops: Vec<Opportunity> = self
            .items
            .values()
            .filter(|op| {
                op.status == "open"
                    && domain.is_none_or(|d| op.domain == d || op.domain.contains(d))
            })
            .cloned()
            .collect();
        ops.sort_by(|a, b| {
            b.priority_score()
                .total_cmp(&a.priority_score())
                .then_with(|| a.opportunity_id.cmp(&b.opportunity_id))
        });
        ops.truncate(top_k);
        ops
    }

    pub fn save(&self) -> Result<()> {
        if let Some(parent) = self.path.parent() {
            fs::create_dir_all(parent)?;
        }
        let rows: Vec<&Opportunity> = self.items.values().collect();
        let tmp = self.path.with_file_name(format!(
            "{}.tmp",
            self.path.file_name().unwrap_or_default().to_string_lossy()
        ));
        let mut raw = serde_json::to_vec_pretty(&rows)?;
        raw.push(b'\n');
        fs::write(&tmp, raw)?;
        fs::rename(tmp, &self.path)?;
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SyncReceipt {
    pub experiment_id: String,
    pub record_sha256: String,
    pub artifact_sha256: String,
    pub uploaded_artifacts: BTreeMap<String, String>,
    pub status: String,
    pub readback_verified: bool,
    pub error: String,
    pub synced_at: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CiStats {
    pub n: usize,
    pub mean: f64,
    pub best: f64,
    pub std: f64,
    pub ci95: Option<f64>,
    pub cv_pct: Option<f64>,
}

pub fn student_t_ci95(samples: &[f64]) -> Result<CiStats> {
    if samples.is_empty() || samples.iter().any(|x| !x.is_finite()) {
        return Err(Error::Message(
            "samples must be non-empty finite floats".into(),
        ));
    }
    let n = samples.len();
    let mean = samples.iter().sum::<f64>() / n as f64;
    let best = samples.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if n == 1 {
        return Ok(CiStats {
            n: 1,
            mean,
            best,
            std: 0.0,
            ci95: None,
            cv_pct: None,
        });
    }
    let var = samples.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std = var.sqrt();
    let t_crit = T975.get(n - 2).copied().unwrap_or(1.96);
    let ci = t_crit * std / (n as f64).sqrt();
    let cv = if mean != 0.0 {
        100.0 * std / mean.abs()
    } else {
        0.0
    };
    Ok(CiStats {
        n,
        mean,
        best,
        std,
        ci95: Some(ci),
        cv_pct: Some(cv),
    })
}

pub fn paired_comparison(
    metric: &str,
    unit: &str,
    baseline_samples: &[f64],
    candidate_samples: &[f64],
    higher_is_better: bool,
) -> Result<StatisticalComparison> {
    let base = student_t_ci95(baseline_samples)?;
    let cand = student_t_ci95(candidate_samples)?;
    let delta = cand.mean - base.mean;
    let ratio = if base.mean != 0.0 {
        Some(cand.mean / base.mean)
    } else {
        None
    };
    let (ratio_lo, ratio_hi) = match (ratio, base.ci95, cand.ci95) {
        (Some(r), Some(bc), Some(cc)) if base.mean != 0.0 && cand.mean != 0.0 => {
            let rci = r.abs() * (bc / base.mean.abs()).hypot(cc / cand.mean.abs());
            (Some(r - rci), Some(r + rci))
        }
        _ => (None, None),
    };
    let verdict = match (ratio_lo, ratio_hi) {
        (Some(lo), Some(_hi)) if higher_is_better && lo > 1.0 => "win",
        (Some(_lo), Some(hi)) if higher_is_better && hi < 1.0 => "loss",
        (Some(_lo), Some(hi)) if !higher_is_better && hi < 1.0 => "win",
        (Some(lo), Some(_hi)) if !higher_is_better && lo > 1.0 => "loss",
        _ => "parity",
    };
    Ok(StatisticalComparison {
        metric: metric.to_string(),
        unit: unit.to_string(),
        higher_is_better,
        n: base.n.min(cand.n),
        baseline_mean: base.mean,
        candidate_mean: cand.mean,
        baseline_ci95: base.ci95,
        candidate_ci95: cand.ci95,
        delta,
        ratio,
        ratio_lo,
        ratio_hi,
        p_value: None,
        anytime_valid: false,
        verdict: verdict.to_string(),
    })
}

impl ExperimentRecordV2 {
    pub fn digest(&self) -> Result<String> {
        let val = serde_json::to_value(self)?;
        Ok(rfc8785_digest(&val)?)
    }

    pub fn dedup_key(&self) -> Result<String> {
        let cfg_d = if self.config_digest.is_empty() {
            rfc8785_digest(&self.config)?
        } else {
            self.config_digest.clone()
        };
        let ds_d = if !self.workload_manifest.manifest_sha256.is_empty() {
            self.workload_manifest.manifest_sha256.clone()
        } else if !self.workload_manifest.dataset.is_empty() {
            self.workload_manifest.dataset.clone()
        } else {
            self.workload_manifest.manifest_id.clone()
        };
        let hw_cls = if !self.hardware_class.is_empty() {
            self.hardware_class.clone()
        } else if !self.hardware.backend.is_empty() {
            self.hardware.backend.clone()
        } else if !self.hardware.gpu.is_empty() {
            self.hardware.gpu.clone()
        } else {
            "CPU".into()
        };
        let proj = if self.project.is_empty() {
            "default"
        } else {
            &self.project
        };
        let hyp = if self.hypothesis_id.is_empty() {
            &self.hypothesis
        } else {
            &self.hypothesis_id
        };
        Ok(rfc8785_digest(&json!({
            "project": proj,
            "hypothesis_id": hyp,
            "config_digest": cfg_d,
            "commit": self.commit,
            "dataset_digest": ds_d,
            "hardware_class": hw_cls,
        }))?)
    }

    pub fn validate(&self) -> Result<()> {
        let checks = [
            ("project", &self.project),
            ("study", &self.study),
            ("objective", &self.objective),
            ("hypothesis", &self.hypothesis),
            ("experiment_id", &self.experiment_id),
            ("repo", &self.repo),
            ("commit", &self.commit),
            ("conclusion", &self.conclusion),
        ];
        for (k, v) in checks {
            if v.trim().is_empty() {
                return Err(Error::Message(format!(
                    "ExperimentRecordV2 missing required field: {k}"
                )));
            }
        }
        if self.workload_manifest.manifest_id.is_empty()
            && self.workload_manifest.dataset.is_empty()
        {
            return Err(Error::Message(
                "ExperimentRecordV2 requires workload_manifest".into(),
            ));
        }
        if self.artifact_hashes.is_empty() {
            return Err(Error::Message(
                "ExperimentRecordV2 requires supporting artifact_hashes".into(),
            ));
        }
        if self.baseline.is_empty() || self.candidate.is_empty() {
            return Err(Error::Message(
                "ExperimentRecordV2 requires both baseline and candidate".into(),
            ));
        }
        if self.statistical_comparison.n == 0 {
            return Err(Error::Message(
                "ExperimentRecordV2 requires positive sample count n".into(),
            ));
        }
        if !self.statistical_comparison.baseline_mean.is_finite()
            || !self.statistical_comparison.candidate_mean.is_finite()
        {
            return Err(Error::Message(
                "ExperimentRecordV2 measurements must be finite".into(),
            ));
        }
        if !self.historical {
            let b = self.raw_samples.get("baseline").map_or(0, Vec::len);
            let c = self.raw_samples.get("candidate").map_or(0, Vec::len);
            if b == 0 || c == 0 {
                return Err(Error::Message(
                    "Reproduced experiment requires raw_samples['baseline'] and raw_samples['candidate']".into(),
                ));
            }
        }
        Ok(())
    }

    pub fn to_run(&self) -> Run {
        let n = self.statistical_comparison.n as i64;
        let ev = epistemic_verdict(&self.statistical_comparison.verdict);
        let mut r = Run {
            project: self.project.clone(),
            id: self.experiment_id.clone(),
            benchmark: self.study.clone(),
            split: if self.workload_manifest.split.is_empty() {
                "bench".into()
            } else {
                self.workload_manifest.split.clone()
            },
            system: self.repo.clone(),
            version: self.commit.chars().take(12).collect(),
            study: self.study.clone(),
            commit: self.commit.clone(),
            questions: Some(n),
            answered: Some(n),
            when: self.recorded_at.clone(),
            ended: self.recorded_at.clone(),
            notes: json!({
                "hypothesis": self.hypothesis,
                "conclusion": self.conclusion,
                "verdict": ev,
                "failure_diagnosis": self.failure_diagnosis,
                "next_hypothesis": self.next_hypothesis,
                "historical": self.historical,
                "backend": self.hardware.backend,
                "precision": self.hardware.precision,
                "artifact_hashes": self.artifact_hashes,
            })
            .to_string(),
            ..Run::default()
        };
        r.measures.push(Measure {
            category: "candidate".into(),
            metric: self.statistical_comparison.metric.clone(),
            value: self.statistical_comparison.candidate_mean,
            lo: self
                .statistical_comparison
                .candidate_ci95
                .map(|c| self.statistical_comparison.candidate_mean - c),
            hi: self
                .statistical_comparison
                .candidate_ci95
                .map(|c| self.statistical_comparison.candidate_mean + c),
            n: Some(n),
            of: Some(n),
        });
        r.measures.push(Measure {
            category: "baseline".into(),
            metric: self.statistical_comparison.metric.clone(),
            value: self.statistical_comparison.baseline_mean,
            lo: self
                .statistical_comparison
                .baseline_ci95
                .map(|c| self.statistical_comparison.baseline_mean - c),
            hi: self
                .statistical_comparison
                .baseline_ci95
                .map(|c| self.statistical_comparison.baseline_mean + c),
            n: Some(n),
            of: Some(n),
        });
        if let Some(ratio) = self.statistical_comparison.ratio {
            r.measures.push(Measure {
                category: "comparison".into(),
                metric: "ratio".into(),
                value: ratio,
                lo: self.statistical_comparison.ratio_lo,
                hi: self.statistical_comparison.ratio_hi,
                n: Some(n),
                of: Some(n),
            });
        }
        r
    }

    pub fn to_experiment_record(&self) -> ExperimentRecord {
        let ev = epistemic_verdict(&self.statistical_comparison.verdict);
        let because = if ev == "refuted" && !self.failure_diagnosis.trim().is_empty() {
            self.failure_diagnosis.trim().to_string()
        } else {
            self.conclusion.clone()
        };
        let backend = if self.hardware.backend.is_empty() {
            "CPU"
        } else {
            &self.hardware.backend
        };
        let note = json!({
            "study": self.study,
            "repo": self.repo,
            "backend": self.hardware.backend,
            "failure_diagnosis": self.failure_diagnosis,
            "next_hypothesis": self.next_hypothesis,
        })
        .to_string();
        ExperimentRecord {
            id: self.experiment_id.clone(),
            kind: "experiment-v2".into(),
            subject: format!("{}/{backend}", self.repo),
            task: self.study.clone(),
            metric: self.statistical_comparison.metric.clone(),
            value: self.statistical_comparison.candidate_mean,
            n: self.statistical_comparison.n as u64,
            n_total: self.statistical_comparison.n as u64,
            status: "complete".into(),
            git_sha: self.commit.clone(),
            git_branch: self.branch.clone(),
            git_dirty: false,
            lib_versions: BTreeMap::new(),
            meta: Meta {
                doc: String::new(),
                commits: Vec::new(),
                note,
                host: crate::provenance::Host {
                    hostname: self.hardware.host.clone(),
                    platform: self.hardware.architecture.clone(),
                },
                hypothesis: self.hypothesis.clone(),
                predict: self.objective.clone(),
                verdict: ev.into(),
                because,
                log: Vec::new(),
            },
            ..ExperimentRecord::default()
        }
    }
}

pub struct LocalResearchLedger {
    pub root: PathBuf,
    pub run_id: String,
}

impl LocalResearchLedger {
    pub fn open(root: impl AsRef<Path>, run_id: &str) -> Result<Self> {
        let root = root.as_ref().to_path_buf();
        for sub in ["records", "artifacts", "receipts", "wal"] {
            fs::create_dir_all(root.join(sub))?;
        }
        Ok(Self {
            root,
            run_id: run_id.to_string(),
        })
    }

    fn safe_name(id: &str) -> String {
        let mut h = Sha256::new();
        h.update(id.as_bytes());
        format!("{:x}", h.finalize())[..24].to_string()
    }

    pub fn opportunities(&self) -> Result<OpportunityQueue> {
        OpportunityQueue::open(self.root.join("opportunities.json"))
    }

    pub fn artifact_manifest(&self) -> Result<BTreeMap<String, String>> {
        let p = self.root.join("artifacts").join("manifest.json");
        if !p.is_file() {
            return Ok(BTreeMap::new());
        }
        Ok(serde_json::from_slice(&fs::read(p)?)?)
    }

    fn update_artifact_manifest(&self, entries: &BTreeMap<String, String>) -> Result<()> {
        let mut cur = self.artifact_manifest()?;
        for (k, v) in entries {
            cur.insert(k.clone(), v.clone());
        }
        let p = self.root.join("artifacts").join("manifest.json");
        let tmp = self.root.join("artifacts").join("manifest.json.tmp");
        let mut raw = serde_json::to_vec_pretty(&cur)?;
        raw.push(b'\n');
        fs::write(&tmp, raw)?;
        fs::rename(tmp, p)?;
        Ok(())
    }

    pub fn dead_ends(&self) -> Result<Vec<DeadEndEntry>> {
        let p = self.root.join("dead_ends.json");
        if !p.is_file() {
            return Ok(Vec::new());
        }
        Ok(serde_json::from_slice(&fs::read(p)?)?)
    }

    fn record_dead_end(&self, entry: DeadEndEntry) -> Result<()> {
        let mut map: BTreeMap<String, DeadEndEntry> = self
            .dead_ends()?
            .into_iter()
            .map(|d| (d.experiment_id.clone(), d))
            .collect();
        map.insert(entry.experiment_id.clone(), entry);
        let rows: Vec<&DeadEndEntry> = map.values().collect();
        let p = self.root.join("dead_ends.json");
        let tmp = self.root.join("dead_ends.json.tmp");
        let mut raw = serde_json::to_vec_pretty(&rows)?;
        raw.push(b'\n');
        fs::write(&tmp, raw)?;
        fs::rename(tmp, p)?;
        Ok(())
    }

    pub fn store_artifact(&self, bytes: &[u8]) -> Result<String> {
        self.store_named_artifact("", bytes)
    }

    pub fn store_named_artifact(&self, name: &str, bytes: &[u8]) -> Result<String> {
        let sha = format!("{:x}", Sha256::digest(bytes));
        let dst = self.root.join("artifacts").join(&sha);
        if !dst.exists() {
            let tmp = self.root.join("artifacts").join(format!("{sha}.tmp"));
            fs::write(&tmp, bytes)?;
            fs::rename(tmp, dst)?;
        }
        if !name.is_empty() {
            let mut m = BTreeMap::new();
            m.insert(name.to_string(), sha.clone());
            self.update_artifact_manifest(&m)?;
        }
        Ok(sha)
    }

    pub fn record_experiment(
        &self,
        exp: &mut ExperimentRecordV2,
        artifacts: &[(&str, &[u8])],
    ) -> Result<SyncReceipt> {
        self.record_experiment_with_options(exp, artifacts, true, None)
    }

    pub fn record_experiment_with_options(
        &self,
        exp: &mut ExperimentRecordV2,
        artifacts: &[(&str, &[u8])],
        auto_transfer: bool,
        catalog: Option<TransferCatalog>,
    ) -> Result<SyncReceipt> {
        if exp.recorded_at.is_empty() {
            exp.recorded_at = now();
        }
        if exp.config_digest.is_empty() {
            exp.config_digest = rfc8785_digest(&exp.config)?;
        }
        if exp.hardware_class.is_empty() {
            exp.hardware_class = if !exp.hardware.backend.is_empty() {
                exp.hardware.backend.clone()
            } else if !exp.hardware.gpu.is_empty() {
                exp.hardware.gpu.clone()
            } else {
                "CPU".into()
            };
        }
        if exp.verdict.is_empty() {
            exp.verdict = epistemic_verdict(&exp.statistical_comparison.verdict).to_string();
        }
        if !exp.parent_experiment.is_empty()
            && !exp.parent_ids.iter().any(|p| p == &exp.parent_experiment)
        {
            exp.parent_ids.push(exp.parent_experiment.clone());
        }

        let mut manifest_updates = BTreeMap::new();
        for (name, bytes) in artifacts {
            let key = format!("{}:{name}", exp.experiment_id);
            let sha = self.store_named_artifact(&key, bytes)?;
            exp.artifact_hashes.insert((*name).to_string(), sha.clone());
            manifest_updates.insert(key, sha);
        }
        for (name, sha) in &exp.artifact_hashes {
            manifest_updates
                .entry(format!("{}:{name}", exp.experiment_id))
                .or_insert_with(|| sha.clone());
            let ref_uri = format!("sha256:{sha}");
            if !exp.artifact_refs.contains(&ref_uri) {
                exp.artifact_refs.push(ref_uri);
            }
        }
        exp.validate()?;

        let rec_bytes = serde_json::to_vec_pretty(exp)?;
        let rec_key = format!("{}:record", exp.experiment_id);
        let rec_sha = self.store_named_artifact(&rec_key, &rec_bytes)?;
        manifest_updates.insert(rec_key, rec_sha.clone());
        self.update_artifact_manifest(&manifest_updates)?;

        let safe = Self::safe_name(&exp.experiment_id);
        let rec_path = self.root.join("records").join(format!("{safe}.json"));
        let tmp = self.root.join("records").join(format!("{safe}.json.tmp"));
        fs::write(&tmp, &rec_bytes)?;
        fs::rename(tmp, rec_path)?;

        let engine = TransferEngine::new(catalog.clone());
        if let Some(de) = engine.dead_end_from(exp) {
            self.record_dead_end(de)?;
        }
        if auto_transfer {
            let mut q = self.opportunities()?;
            for op in generate_transfer_hypotheses(exp, catalog) {
                q.add(op, true)?;
            }
        }

        let mut wal = ResearchWal::open(&self.root.join("wal"), &self.run_id)?;
        wal.append(
            "experiment_recorded",
            &json!({
                "experiment_id": exp.experiment_id,
                "study": exp.study,
                "project": exp.project,
                "repo": exp.repo,
                "commit": exp.commit,
                "metric": exp.statistical_comparison.metric,
                "candidate_mean": exp.statistical_comparison.candidate_mean,
                "baseline_mean": exp.statistical_comparison.baseline_mean,
                "verdict": exp.statistical_comparison.verdict,
                "historical": exp.historical,
                "record_sha256": rec_sha,
            }),
        )?;

        let art_sha = exp
            .artifact_hashes
            .values()
            .next()
            .cloned()
            .unwrap_or_else(|| rec_sha.clone());
        let receipt = SyncReceipt {
            experiment_id: exp.experiment_id.clone(),
            record_sha256: rec_sha,
            artifact_sha256: art_sha,
            uploaded_artifacts: BTreeMap::new(),
            status: "pending_sync".into(),
            readback_verified: false,
            error: String::new(),
            synced_at: now(),
        };
        self.write_receipt(&receipt)?;
        Ok(receipt)
    }

    pub fn write_receipt(&self, receipt: &SyncReceipt) -> Result<()> {
        let safe = Self::safe_name(&receipt.experiment_id);
        let path = self
            .root
            .join("receipts")
            .join(format!("{safe}.receipt.json"));
        let tmp = self
            .root
            .join("receipts")
            .join(format!("{safe}.receipt.json.tmp"));
        fs::write(&tmp, serde_json::to_vec_pretty(receipt)?)?;
        fs::rename(tmp, path)?;
        Ok(())
    }

    pub fn query_all(&self) -> Result<Vec<ExperimentRecordV2>> {
        let mut out = Vec::new();
        for entry in fs::read_dir(self.root.join("records"))? {
            let p = entry?.path();
            if p.extension().and_then(|s| s.to_str()) == Some("json") {
                let bytes = fs::read(p)?;
                let rec: ExperimentRecordV2 = serde_json::from_slice(&bytes)?;
                out.push(rec);
            }
        }
        out.sort_by(|a, b| {
            a.recorded_at
                .cmp(&b.recorded_at)
                .then(a.experiment_id.cmp(&b.experiment_id))
        });
        Ok(out)
    }

    pub fn query_experiments(
        &self,
        study: Option<&str>,
        verdict: Option<&str>,
        repo: Option<&str>,
        backend: Option<&str>,
        search: Option<&str>,
        include_failures: bool,
    ) -> Result<Vec<ExperimentRecordV2>> {
        let want_v = verdict.map(epistemic_verdict);
        let want_b = backend.map(str::to_ascii_lowercase);
        let tokens: Vec<String> = search
            .unwrap_or("")
            .split_whitespace()
            .map(str::to_ascii_lowercase)
            .collect();
        let mut out = Vec::new();
        for rec in self.query_all()? {
            let ev = epistemic_verdict(&rec.statistical_comparison.verdict);
            if !include_failures && ev == "refuted" {
                continue;
            }
            if let Some(st) = study {
                if rec.study != st
                    && !rec
                        .study
                        .to_ascii_lowercase()
                        .contains(&st.to_ascii_lowercase())
                {
                    continue;
                }
            }
            if let Some(wv) = want_v {
                let raw_v = verdict.unwrap_or("").trim().to_ascii_lowercase();
                if ev != wv && rec.statistical_comparison.verdict.to_ascii_lowercase() != raw_v {
                    continue;
                }
            }
            if let Some(rp) = repo {
                if rec.repo != rp {
                    continue;
                }
            }
            if let Some(wb) = &want_b {
                if !rec.hardware.backend.to_ascii_lowercase().contains(wb)
                    && !rec.hardware.gpu.to_ascii_lowercase().contains(wb)
                {
                    continue;
                }
            }
            if !tokens.is_empty() {
                let blob = format!(
                    "{} {} {} {} {} {} {} {} {} {} {}",
                    rec.experiment_id,
                    rec.study,
                    rec.objective,
                    rec.hypothesis,
                    rec.conclusion,
                    rec.failure_diagnosis,
                    rec.next_hypothesis,
                    rec.hardware.gpu,
                    rec.hardware.backend,
                    rec.workload_manifest.shapes.join(" "),
                    serde_json::to_string(&rec.config).unwrap_or_default()
                )
                .to_ascii_lowercase();
                if !tokens.iter().all(|tok| blob.contains(tok)) {
                    continue;
                }
            }
            out.push(rec);
        }
        Ok(out)
    }

    pub fn prior_failures(&self, domain_or_study: Option<&str>) -> Result<Vec<ExperimentRecordV2>> {
        let needle = domain_or_study.map(|s| s.trim().to_ascii_lowercase());
        let mut out = Vec::new();
        for rec in self.query_all()? {
            if epistemic_verdict(&rec.statistical_comparison.verdict) != "refuted" {
                continue;
            }
            if let Some(n) = &needle {
                let hay = format!(
                    "{} {} {} {}",
                    rec.study, rec.repo, rec.experiment_id, rec.hardware.backend
                )
                .to_ascii_lowercase();
                if !hay.contains(n) {
                    continue;
                }
            }
            out.push(rec);
        }
        Ok(out)
    }

    pub fn recommended_experiments(
        &self,
        domain: Option<&str>,
        top_k: usize,
    ) -> Result<Vec<Opportunity>> {
        Ok(self.opportunities()?.top_open(domain, top_k))
    }

    pub fn pull_from_cloud(&self, client: &Research) -> Value {
        match (
            client.query(None, None),
            client.runs(None, None, None, None),
        ) {
            (Ok(exps), Ok(runs)) => {
                let state = json!({
                    "synced_at": now(),
                    "experiments": exps,
                    "runs": runs.data,
                });
                let p = self.root.join("cloud_state.json");
                let tmp = self.root.join("cloud_state.json.tmp");
                if let Ok(raw) = serde_json::to_vec_pretty(&state) {
                    let _ = fs::write(&tmp, raw);
                    let _ = fs::rename(tmp, p);
                }
                json!({"status": "ok", "experiments": exps.len(), "runs": runs.data.len()})
            }
            (Err(e), _) | (_, Err(e)) => {
                json!({"status": "offline", "error": e.to_string(), "experiments": 0, "runs": 0})
            }
        }
    }

    pub fn cloud_state(&self) -> Value {
        let p = self.root.join("cloud_state.json");
        if !p.is_file() {
            return json!({"experiments": [], "runs": []});
        }
        fs::read(p)
            .ok()
            .and_then(|b| serde_json::from_slice(&b).ok())
            .unwrap_or_else(|| json!({"experiments": [], "runs": []}))
    }

    fn upload_and_verify_artifact(
        client: &Research,
        artifact_id: &str,
        run_id: &str,
        media_type: &str,
        canonicalization: &str,
        bytes: &[u8],
    ) -> Result<String> {
        let expected_sha = format!("{:x}", Sha256::digest(bytes));
        let stored = client.upload_artifact_with_meta(
            artifact_id,
            "report",
            run_id,
            media_type,
            canonicalization,
            "none",
            bytes,
        )?;
        if !stored.sha256.is_empty() && stored.sha256 != expected_sha {
            return Err(Error::Message(format!(
                "server artifact sha256 mismatch: expected {expected_sha}, got {}",
                stored.sha256
            )));
        }
        let rb = client.get_artifact(&expected_sha)?;
        let rb_sha = format!("{:x}", Sha256::digest(&rb));
        if rb_sha != expected_sha {
            return Err(Error::Message(format!(
                "artifact readback SHA-256 mismatch: expected {expected_sha}, got {rb_sha}"
            )));
        }
        Ok(expected_sha)
    }

    pub fn sync_to_cloud(&self, client: &Research) -> Result<Vec<SyncReceipt>> {
        let all_recs = self.query_all()?;
        let mut by_study: BTreeMap<String, Vec<&ExperimentRecordV2>> = BTreeMap::new();
        for r in &all_recs {
            by_study.entry(r.study.clone()).or_default().push(r);
        }

        let mut out = Vec::new();
        for rec in &all_recs {
            let safe = Self::safe_name(&rec.experiment_id);
            let rpath = self
                .root
                .join("receipts")
                .join(format!("{safe}.receipt.json"));
            if rpath.is_file() {
                if let Ok(existing) = serde_json::from_slice::<SyncReceipt>(&fs::read(&rpath)?) {
                    if existing.status == "synced_verified" && existing.readback_verified {
                        out.push(existing);
                        continue;
                    }
                }
            }
            let rec_bytes = fs::read(self.root.join("records").join(format!("{safe}.json")))?;
            let rec_sha = format!("{:x}", Sha256::digest(&rec_bytes));
            let mut uploaded_arts = BTreeMap::new();

            let receipt = match (|| -> Result<SyncReceipt> {
                let study_recs = by_study.get(&rec.study).cloned().unwrap_or_default();
                let study_runs: Vec<String> = study_recs
                    .iter()
                    .map(|sr| sr.experiment_id.clone())
                    .collect();
                let wins = study_recs
                    .iter()
                    .filter(|sr| epistemic_verdict(&sr.statistical_comparison.verdict) == "proven")
                    .count();
                let refuted = study_recs
                    .iter()
                    .filter(|sr| epistemic_verdict(&sr.statistical_comparison.verdict) == "refuted")
                    .count();
                let finding = format!(
                    "{wins}/{} proven, {refuted} refuted; latest: {}",
                    study_recs.len(),
                    rec.conclusion
                );
                let bench = Benchmark {
                    project: rec.project.clone(),
                    id: rec.study.clone(),
                    title: rec.objective.clone(),
                    dataset: if rec.workload_manifest.dataset.is_empty() {
                        rec.workload_manifest.manifest_id.clone()
                    } else {
                        rec.workload_manifest.dataset.clone()
                    },
                    metric: rec.statistical_comparison.metric.clone(),
                    ..Benchmark::default()
                };
                let study = Study {
                    project: rec.project.clone(),
                    id: rec.study.clone(),
                    title: rec.objective.clone(),
                    question: rec.hypothesis.clone(),
                    method: "paired_anytime_valid_ab".into(),
                    runs: study_runs,
                    finding,
                    limits: rec.failure_diagnosis.clone(),
                    when: rec.recorded_at.clone(),
                    ..Study::default()
                };
                client.post_benchmark(&bench)?;
                client.post_studies(std::slice::from_ref(&study))?;
                let runs = [rec.to_run()];
                client.post_batch(&runs, &[])?;
                let exp_wire = [rec.to_experiment_record()];
                client.ingest(&exp_wire, &[])?;

                for (art_name, art_sha) in &rec.artifact_hashes {
                    let art_file = self.root.join("artifacts").join(art_sha);
                    if art_file.is_file() {
                        let raw_data = fs::read(&art_file)?;
                        let mtype = if art_name.ends_with("json") {
                            "application/json"
                        } else {
                            "application/octet-stream"
                        };
                        let v_sha = Self::upload_and_verify_artifact(
                            client,
                            &format!("{}:{art_name}", rec.experiment_id),
                            &rec.experiment_id,
                            mtype,
                            "raw",
                            &raw_data,
                        )?;
                        uploaded_arts.insert(art_name.clone(), v_sha);
                    }
                }

                let verified_rec_sha = Self::upload_and_verify_artifact(
                    client,
                    &format!("{}:record", rec.experiment_id),
                    &rec.experiment_id,
                    "application/json",
                    "json",
                    &rec_bytes,
                )?;
                uploaded_arts.insert("record".into(), verified_rec_sha.clone());

                let rb_runs = client.runs(Some(&rec.experiment_id), None, None, None)?;
                let got_run = rb_runs.data.iter().any(|r| r.id == rec.experiment_id);
                let rb_exps = client.query(None, None)?;
                let got_exp = rb_exps.iter().any(|e| e.id == rec.experiment_id);

                let all_ok = got_run && got_exp && verified_rec_sha == rec_sha;
                Ok(SyncReceipt {
                    experiment_id: rec.experiment_id.clone(),
                    record_sha256: rec_sha.clone(),
                    artifact_sha256: verified_rec_sha,
                    uploaded_artifacts: uploaded_arts.clone(),
                    status: if all_ok {
                        "synced_verified".into()
                    } else {
                        "partial_sync".into()
                    },
                    readback_verified: all_ok,
                    error: if all_ok {
                        String::new()
                    } else {
                        "readback verification incomplete".into()
                    },
                    synced_at: now(),
                })
            })() {
                Ok(r) => r,
                Err(e) => SyncReceipt {
                    experiment_id: rec.experiment_id.clone(),
                    record_sha256: rec_sha.clone(),
                    artifact_sha256: rec_sha,
                    uploaded_artifacts: uploaded_arts,
                    status: "pending_sync".into(),
                    readback_verified: false,
                    error: e.to_string(),
                    synced_at: now(),
                },
            };
            self.write_receipt(&receipt)?;
            out.push(receipt);
        }
        Ok(out)
    }
}
