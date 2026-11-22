//! Unified first-class scientific experiment protocol, WAL-first ledger, and opportunity queue.
//!
//! Rust peer of `hanzo_research.protocol`: produces bit-identical schemas on disk and over
//! `/v1/research` so `hanzoai/ml`, `hanzo-engine`, and `hanzo-inc/kai` share one ledger.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::canonical::rfc8785_digest;
use crate::wal::ResearchWal;
use crate::wire::{now, Benchmark, Run};
use crate::{Error, Research, Result};

const T975: [f64; 30] = [
    12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
    2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
    2.052, 2.048, 2.045, 2.042,
];

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
    pub experiment_id: String,
    pub parent_experiment: String,
    pub prior_best: Option<f64>,
    pub repo: String,
    pub commit: String,
    pub branch: String,
    pub pr: String,
    pub artifact_hashes: BTreeMap<String, String>,
    pub hardware: HardwareSpec,
    pub config: BTreeMap<String, serde_json::Value>,
    pub workload_manifest: WorkloadManifest,
    pub baseline: BTreeMap<String, serde_json::Value>,
    pub candidate: BTreeMap<String, serde_json::Value>,
    pub statistical_comparison: StatisticalComparison,
    pub correctness: CorrectnessChecks,
    pub raw_samples: BTreeMap<String, Vec<f64>>,
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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SyncReceipt {
    pub experiment_id: String,
    pub record_sha256: String,
    pub artifact_sha256: String,
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
        if self.workload_manifest.manifest_id.is_empty() && self.workload_manifest.dataset.is_empty()
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
            questions: Some(self.statistical_comparison.n as i64),
            answered: Some(self.statistical_comparison.n as i64),
            when: self.recorded_at.clone(),
            ended: self.recorded_at.clone(),
            notes: serde_json::json!({
                "hypothesis": self.hypothesis,
                "conclusion": self.conclusion,
                "next_hypothesis": self.next_hypothesis,
                "historical": self.historical,
                "backend": self.hardware.backend,
                "precision": self.hardware.precision,
                "artifact_hashes": self.artifact_hashes,
            })
            .to_string(),
            ..Run::default()
        };
        r.measures.push(crate::wire::Measure {
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
            n: Some(self.statistical_comparison.n as i64),
            of: Some(self.statistical_comparison.n as i64),
        });
        r.measures.push(crate::wire::Measure {
            category: "baseline".into(),
            metric: self.statistical_comparison.metric.clone(),
            value: self.statistical_comparison.baseline_mean,
            lo: None,
            hi: None,
            n: Some(self.statistical_comparison.n as i64),
            of: Some(self.statistical_comparison.n as i64),
        });
        r
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

    pub fn store_artifact(&self, bytes: &[u8]) -> Result<String> {
        let mut h = Sha256::new();
        h.update(bytes);
        let sha = format!("{:x}", h.finalize());
        let dst = self.root.join("artifacts").join(&sha);
        if !dst.exists() {
            let tmp = self.root.join("artifacts").join(format!("{sha}.tmp"));
            fs::write(&tmp, bytes)?;
            fs::rename(tmp, dst)?;
        }
        Ok(sha)
    }

    pub fn record_experiment(
        &self,
        exp: &mut ExperimentRecordV2,
        artifacts: &[(&str, &[u8])],
    ) -> Result<SyncReceipt> {
        if exp.recorded_at.is_empty() {
            exp.recorded_at = now();
        }
        for (name, bytes) in artifacts {
            let sha = self.store_artifact(bytes)?;
            exp.artifact_hashes.insert((*name).to_string(), sha);
        }
        exp.validate()?;
        let rec_bytes = serde_json::to_vec_pretty(exp)?;
        let rec_sha = self.store_artifact(&rec_bytes)?;
        let safe = Self::safe_name(&exp.experiment_id);
        let rec_path = self.root.join("records").join(format!("{safe}.json"));
        let tmp = self.root.join("records").join(format!("{safe}.json.tmp"));
        fs::write(&tmp, &rec_bytes)?;
        fs::rename(tmp, rec_path)?;

        let mut wal = ResearchWal::open(&self.root.join("wal"), &self.run_id)?;
        wal.append(
            "experiment_recorded",
            &serde_json::json!({
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
        let path = self.root.join("receipts").join(format!("{safe}.receipt.json"));
        let tmp = self.root.join("receipts").join(format!("{safe}.receipt.json.tmp"));
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
        out.sort_by(|a, b| a.recorded_at.cmp(&b.recorded_at).then(a.experiment_id.cmp(&b.experiment_id)));
        Ok(out)
    }

    pub fn sync_to_cloud(&self, client: &Research) -> Result<Vec<SyncReceipt>> {
        let mut out = Vec::new();
        for rec in self.query_all()? {
            let safe = Self::safe_name(&rec.experiment_id);
            let rpath = self.root.join("receipts").join(format!("{safe}.receipt.json"));
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
            let receipt = match (|| -> Result<SyncReceipt> {
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
                client.post_benchmark(&bench)?;
                let runs = [rec.to_run()];
                client.post_batch(&runs, &[])?;
                let stored = client.upload_artifact("experiment_record_v2", &rec.experiment_id, &rec_bytes)?;
                let rb = client.get_artifact(&stored.sha256)?;
                let rb_sha = format!("{:x}", Sha256::digest(&rb));
                if rb_sha != rec_sha {
                    return Err(Error::Message(format!(
                        "artifact readback SHA-256 mismatch: expected {rec_sha}, got {rb_sha}"
                    )));
                }
                Ok(SyncReceipt {
                    experiment_id: rec.experiment_id.clone(),
                    record_sha256: rec_sha.clone(),
                    artifact_sha256: stored.sha256,
                    status: "synced_verified".into(),
                    readback_verified: true,
                    error: String::new(),
                    synced_at: now(),
                })
            })() {
                Ok(r) => r,
                Err(e) => SyncReceipt {
                    experiment_id: rec.experiment_id.clone(),
                    record_sha256: rec_sha.clone(),
                    artifact_sha256: rec_sha,
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
