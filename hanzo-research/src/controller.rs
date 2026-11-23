//! Domain-agnostic autonomous research controller promoted from Kai PR #19.
//!
//! Coordinates single-GPU and distributed workers across `hanzoai/ml`, `hanzo-engine`,
//! and `hanzo-inc/kai`:
//! - Objective and search-space registry with protected winners/baselines,
//! - UCB1 + forced-coverage arm selection skipping refuted `(commit, backend, shape)` dead ends,
//! - Cross-machine cloud lease with monotonic fencing tokens + local non-blocking GPU `flock`,
//! - Checkpointed progressive anytime-valid A/B execution (`micro -> medium -> qualification`),
//! - Write-ahead sealed final-qualification exposure ledger (`ExposureLedger`).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

use crate::lease::CloudLeaseManager;
use crate::protocol::{
    epistemic_verdict, CorrectnessChecks, ExperimentRecordV2, HardwareSpec, LocalResearchLedger,
    Opportunity, StatisticalComparison, WorkloadManifest,
};
use crate::provenance;
use crate::testing::{
    impacted_benchmarks, LineageRule, ProgressiveAbPlan, ProgressiveAbRunner, ProgressiveStage,
    ProgressiveStageOutput,
};
use crate::wal::ResearchWal;
use crate::wire::{now, token};
use crate::{Error, Research, Result};

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let tmp = path.with_file_name(format!(
        "{}.tmp",
        path.file_name().unwrap_or_default().to_string_lossy()
    ));
    fs::write(&tmp, bytes)?;
    fs::rename(tmp, path)?;
    Ok(())
}

pub struct GpuLock {
    file: File,
}

impl GpuLock {
    pub fn acquire(state_dir: &Path) -> Result<Self> {
        fs::create_dir_all(state_dir)?;
        let lock_path = state_dir.join("gpu.lock");
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(lock_path)?;
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
            if rc != 0 {
                return Err(Error::Message(
                    "GPU campaign already has an active tick; refusing concurrent execution".into(),
                ));
            }
        }
        Ok(Self { file })
    }
}

impl Drop for GpuLock {
    fn drop(&mut self) {
        #[cfg(unix)]
        {
            use std::os::fd::AsRawFd;
            let _ = unsafe { libc::flock(self.file.as_raw_fd(), libc::LOCK_UN) };
        }
    }
}

pub fn detect_host_backends() -> BTreeSet<String> {
    if let Ok(ov) = std::env::var("RESEARCH_HOST_BACKENDS") {
        let trimmed = ov.trim();
        if !trimmed.is_empty() {
            return trimmed
                .split(',')
                .map(|s| s.trim().to_ascii_uppercase())
                .filter(|s| !s.is_empty())
                .collect();
        }
    }
    let mut backends = BTreeSet::from(["CPU".to_string()]);
    if cfg!(target_os = "macos") {
        backends.insert("METAL".to_string());
    }
    if Path::new("/dev/nvidia0").exists() || Path::new("/usr/local/cuda").exists() {
        backends.insert("CUDA".to_string());
    }
    if Path::new("/dev/kfd").exists() || Path::new("/opt/rocm").exists() {
        backends.insert("ROCM".to_string());
    }
    if Path::new("/usr/share/vulkan").exists() || Path::new("/etc/vulkan").exists() {
        backends.insert("VULKAN".to_string());
    }
    backends
}

/// UCB1 + forced-coverage arm selection matching Python `controller.arm_candidates`.
pub fn arm_candidates(
    arms: &[(String, String)],
    trials: &[(String, String, Option<f64>, String)],
    cap: usize,
    higher_is_better: bool,
    blocked_arms: &BTreeSet<(String, String)>,
) -> Vec<(f64, isize, String, String)> {
    if arms.is_empty() {
        return Vec::new();
    }
    let mut counts: BTreeMap<(String, String), usize> =
        arms.iter().cloned().map(|k| (k, 0usize)).collect();
    let mut failures: BTreeMap<(String, String), usize> =
        arms.iter().cloned().map(|k| (k, 0usize)).collect();
    let mut best: BTreeMap<(String, String), f64> = arms
        .iter()
        .cloned()
        .map(|k| (k, f64::NEG_INFINITY))
        .collect();

    for (arm, var, score, state) in trials {
        let key = (arm.clone(), var.clone());
        if let Some(cnt) = counts.get_mut(&key) {
            *cnt += 1;
            if let Some(sc) = score.filter(|x| x.is_finite()) {
                let signed = if higher_is_better { sc } else { -sc };
                let entry = best.entry(key).or_insert(f64::NEG_INFINITY);
                *entry = entry.max(signed);
            } else if matches!(state.as_str(), "failed" | "blocked" | "refuted" | "loss") {
                *failures.entry(key).or_insert(0) += 1;
            }
        }
    }

    let total: usize = counts.values().sum();
    if total >= cap {
        return Vec::new();
    }

    let mut out = Vec::new();
    for (arm, var) in arms {
        let key = (arm.clone(), var.clone());
        if blocked_arms.contains(&key) {
            continue;
        }
        let n = counts[&key];
        let f = failures[&key];
        let b = best[&key];
        let ucb = if n == 0 {
            f64::INFINITY
        } else if b == f64::NEG_INFINITY {
            if f >= 2 {
                -1e6 - f as f64
            } else {
                -1.0 + 0.05 * (((total + 2) as f64).ln() / n as f64).sqrt()
            }
        } else {
            b + 0.12 * (((total + 2) as f64).ln() / n as f64).sqrt()
                - if f >= 2 { 0.25 } else { 0.0 }
        };
        out.push((ucb, -(n as isize), arm.clone(), var.clone()));
    }
    out.sort_by(|a, b| {
        b.0.total_cmp(&a.0)
            .then_with(|| b.1.cmp(&a.1))
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| b.3.cmp(&a.3))
    });
    out
}

pub fn execute_subprocess(
    executor: &Path,
    phase: &str,
    request: &Value,
    state_dir: &Path,
    timeout_minutes: u64,
    job_filename_prefix: Option<&str>,
) -> Value {
    if !executor.is_file() {
        return json!({"status": "failed", "reason": format!("executor missing: {}", executor.display())});
    }
    let trial_id = request["trial_id"].as_str().unwrap_or("unknown");
    let job_dir = state_dir.join("jobs").join(trial_id);
    if let Err(e) = fs::create_dir_all(&job_dir) {
        return json!({"status": "failed", "reason": e.to_string()});
    }
    let prefix = job_filename_prefix.unwrap_or(phase);
    let req_path = job_dir.join(format!("{prefix}.request.json"));
    let out_path = job_dir.join(format!("{prefix}.result.json"));
    let log_path = job_dir.join(format!("{prefix}.log"));

    let mut req_copy = request.clone();
    req_copy["phase"] = json!(phase);
    if let Ok(raw) = serde_json::to_vec_pretty(&req_copy) {
        let _ = write_atomic(&req_path, &raw);
    }

    if matches!(phase, "final" | "qualification") && out_path.exists() {
        return json!({"status": "failed", "reason": "sealed qualification output exists: manual reconciliation required"});
    }
    let _ = fs::remove_file(&out_path);

    let is_py = executor.extension().and_then(|s| s.to_str()) == Some("py");
    let log_file = match OpenOptions::new().create(true).append(true).open(&log_path) {
        Ok(f) => f,
        Err(e) => return json!({"status": "failed", "reason": e.to_string()}),
    };
    let log_err = match log_file.try_clone() {
        Ok(f) => f,
        Err(e) => return json!({"status": "failed", "reason": e.to_string()}),
    };

    let mut cmd = if is_py {
        let mut c = Command::new("python3");
        c.arg(executor);
        c
    } else {
        Command::new(executor)
    };
    cmd.arg(phase)
        .arg("--request")
        .arg(&req_path)
        .arg("--output")
        .arg(&out_path)
        .stdout(Stdio::from(log_file))
        .stderr(Stdio::from(log_err));

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return json!({"status": "failed", "reason": e.to_string()}),
    };

    let deadline = Instant::now() + Duration::from_secs(timeout_minutes.max(1) * 60);
    let status = loop {
        match child.try_wait() {
            Ok(Some(st)) => break st,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return json!({"status": "failed", "reason": format!("{phase} exceeded {timeout_minutes}m")});
                }
                thread::sleep(Duration::from_millis(50));
            }
            Err(e) => return json!({"status": "failed", "reason": e.to_string()}),
        }
    };

    if !status.success() || !out_path.is_file() {
        return json!({
            "status": "failed",
            "reason": format!("{phase} executor exit={:?}; log={}", status.code(), log_path.display())
        });
    }
    let Ok(bytes) = fs::read(&out_path) else {
        return json!({"status": "failed", "reason": "failed reading executor output"});
    };
    let Ok(res) = serde_json::from_slice::<Value>(&bytes) else {
        return json!({"status": "failed", "reason": "invalid JSON from executor"});
    };
    let expected_obj = request["task"]
        .as_str()
        .or_else(|| request["objective_id"].as_str());
    let got_obj = res["task"]
        .as_str()
        .or_else(|| res["objective_id"].as_str());
    if res["trial_id"].as_str() != Some(trial_id)
        || (expected_obj.is_some() && got_obj != expected_obj)
    {
        return json!({"status": "failed", "reason": "executor result identity mismatch"});
    }
    res
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ObjectiveSpec {
    pub id: String,
    pub study: String,
    pub objective: String,
    pub metric: String,
    pub unit: String,
    pub direction: String,
    pub family: String,
    pub protected: bool,
    pub baseline: Option<f64>,
    pub reference_score: Option<f64>,
    pub backends: Vec<String>,
    pub shapes: Vec<String>,
    pub dataset: String,
    pub eval_sha: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchArm {
    pub arm: String,
    pub variant: String,
    pub hypothesis: String,
    pub config: BTreeMap<String, Value>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct CampaignRegistry {
    pub campaign: String,
    pub repo: String,
    pub objectives: Vec<ObjectiveSpec>,
    pub search_space: BTreeMap<String, Vec<SearchArm>>,
    pub plan: ProgressiveAbPlan,
}

impl CampaignRegistry {
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let mut raw: Value = serde_json::from_slice(&fs::read(path.as_ref())?)?;
        if raw.get("objectives").is_none() && raw.get("tasks").is_some() {
            let tasks = raw["tasks"].as_array().cloned().unwrap_or_default();
            let objs: Vec<Value> = tasks
                .into_iter()
                .map(|t| {
                    json!({
                        "id": t["id"],
                        "study": t.get("study").cloned().unwrap_or_else(|| t["id"].clone()),
                        "objective": t.get("objective").cloned().unwrap_or_else(|| json!(format!("Improve {}", t["id"].as_str().unwrap_or("")))),
                        "metric": t["metric"],
                        "unit": t.get("unit").cloned().unwrap_or_else(|| json!("")),
                        "direction": t.get("direction").cloned().unwrap_or_else(|| json!("higher")),
                        "family": t.get("family").cloned().unwrap_or_else(|| json!("default")),
                        "protected": t.get("protected").and_then(Value::as_bool).unwrap_or(false),
                        "baseline": t.get("baseline").cloned().unwrap_or(Value::Null),
                        "reference_score": t.get("jev_reference").or_else(|| t.get("reference_score")).cloned().unwrap_or(Value::Null),
                        "backends": t.get("backends").cloned().unwrap_or_else(|| json!([])),
                        "shapes": t.get("shapes").cloned().unwrap_or_else(|| json!([])),
                        "dataset": t.get("dataset").cloned().unwrap_or_else(|| t["id"].clone()),
                        "eval_sha": t.get("eval_sha").cloned().unwrap_or_else(|| json!("")),
                    })
                })
                .collect();
            raw["objectives"] = Value::Array(objs);
        }
        if let Some(space) = raw.get_mut("search_space").and_then(Value::as_object_mut) {
            for (_fam, arr_val) in space.iter_mut() {
                if let Some(items) = arr_val.as_array_mut() {
                    for item in items.iter_mut() {
                        if let Some(pair) = item.as_array().cloned() {
                            if pair.len() >= 2 {
                                *item = json!({
                                    "arm": pair[0],
                                    "variant": pair[1],
                                    "hypothesis": "",
                                    "config": {},
                                });
                            }
                        }
                    }
                }
            }
        }
        let reg: Self = serde_json::from_value(raw)?;
        reg.validate()?;
        Ok(reg)
    }

    pub fn validate(&self) -> Result<()> {
        if self.campaign.trim().is_empty() {
            return Err(Error::Message(
                "CampaignRegistry requires non-empty campaign id".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        for obj in &self.objectives {
            if !seen.insert(obj.id.clone()) {
                return Err(Error::Message(format!(
                    "duplicate objective id: {}",
                    obj.id
                )));
            }
            if obj.direction != "higher" && obj.direction != "lower" {
                return Err(Error::Message(format!(
                    "unsupported direction {} for {}",
                    obj.direction, obj.id
                )));
            }
            if !obj.protected && !self.search_space.contains_key(&obj.family) {
                return Err(Error::Message(format!(
                    "objective {} references unknown search_space family {}",
                    obj.id, obj.family
                )));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct ExposureRecord {
    pub objective_id: String,
    pub eval_sha: String,
    pub candidate_sha: String,
    pub trial_id: String,
    pub exposed_at: String,
}

pub struct ExposureLedger {
    pub path: PathBuf,
}

impl ExposureLedger {
    pub fn new(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
        }
    }

    pub fn records(&self) -> Result<Vec<ExposureRecord>> {
        if !self.path.is_file() {
            return Ok(Vec::new());
        }
        Ok(serde_json::from_slice(&fs::read(&self.path)?)?)
    }

    pub fn is_exposed(&self, objective_id: &str, eval_sha: &str) -> Result<bool> {
        Ok(self
            .records()?
            .iter()
            .any(|r| r.objective_id == objective_id && r.eval_sha == eval_sha))
    }

    pub fn record_exposure(
        &self,
        objective_id: &str,
        eval_sha: &str,
        candidate_sha: &str,
        trial_id: &str,
    ) -> Result<ExposureRecord> {
        if eval_sha.len() < 8 {
            return Err(Error::Message(
                "sealed qualification requires immutable eval_sha".into(),
            ));
        }
        let mut rows = self.records()?;
        if rows
            .iter()
            .any(|r| r.objective_id == objective_id && r.eval_sha == eval_sha)
        {
            return Err(Error::Message(format!(
                "sealed qualification for {objective_id} already exposed"
            )));
        }
        let rec = ExposureRecord {
            objective_id: objective_id.into(),
            eval_sha: eval_sha.into(),
            candidate_sha: candidate_sha.into(),
            trial_id: trial_id.into(),
            exposed_at: now(),
        };
        rows.push(rec.clone());
        rows.sort_by(|a, b| {
            a.objective_id
                .cmp(&b.objective_id)
                .then(a.eval_sha.cmp(&b.eval_sha))
        });
        let mut raw = serde_json::to_vec_pretty(&rows)?;
        raw.push(b'\n');
        write_atomic(&self.path, &raw)?;
        Ok(rec)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrialSummary {
    pub trial_id: String,
    pub objective_id: String,
    pub arm: String,
    pub variant: String,
    pub commit: String,
    pub backend: String,
    pub shapes: Vec<String>,
    pub state: String,
    pub phase: String,
    pub score: Option<f64>,
    pub verdict: String,
    pub opportunity_id: String,
    pub started_at: String,
    pub finished_at: String,
    pub message: String,
}

pub struct AutonomousController {
    pub ledger: LocalResearchLedger,
    pub state_dir: PathBuf,
    pub checkpoints_dir: PathBuf,
    pub exposures: ExposureLedger,
    pub registry: Option<CampaignRegistry>,
    pub repo_dir: String,
    pub max_trials_per_objective: usize,
    pub worker_id: String,
}

impl AutonomousController {
    pub fn open(
        ledger_dir: impl AsRef<Path>,
        state_dir: Option<&Path>,
        registry: Option<CampaignRegistry>,
        repo_dir: &str,
        max_trials_per_objective: usize,
        worker_id: Option<&str>,
    ) -> Result<Self> {
        let ledger = LocalResearchLedger::open(ledger_dir.as_ref(), "hanzo-shared")?;
        let sdir = state_dir
            .map(Path::to_path_buf)
            .unwrap_or_else(|| ledger.root.join("controller"));
        let ckpt_dir = sdir.join("checkpoints");
        fs::create_dir_all(&ckpt_dir)?;
        let exposures = ExposureLedger::new(sdir.join("exposures.json"));
        let wid = worker_id
            .map(str::to_string)
            .unwrap_or_else(|| format!("{}-{}", provenance::host().hostname, std::process::id()));
        Ok(Self {
            ledger,
            state_dir: sdir,
            checkpoints_dir: ckpt_dir,
            exposures,
            registry,
            repo_dir: if repo_dir.is_empty() {
                ".".into()
            } else {
                repo_dir.into()
            },
            max_trials_per_objective,
            worker_id: wid,
        })
    }

    pub fn trials(&self) -> Result<Vec<TrialSummary>> {
        let p = self.state_dir.join("trials.json");
        if !p.is_file() {
            return Ok(Vec::new());
        }
        Ok(serde_json::from_slice(&fs::read(p)?)?)
    }

    fn save_trial(&self, trial: &TrialSummary) -> Result<()> {
        let mut map: BTreeMap<String, TrialSummary> = self
            .trials()?
            .into_iter()
            .map(|t| (t.trial_id.clone(), t))
            .collect();
        map.insert(trial.trial_id.clone(), trial.clone());
        let mut rows: Vec<&TrialSummary> = map.values().collect();
        rows.sort_by(|a, b| {
            a.started_at
                .cmp(&b.started_at)
                .then(a.trial_id.cmp(&b.trial_id))
        });
        let mut raw = serde_json::to_vec_pretty(&rows)?;
        raw.push(b'\n');
        write_atomic(&self.state_dir.join("trials.json"), &raw)
    }

    fn wal_emit(&self, event_type: &str, payload: &Value) -> Result<()> {
        let mut wal = ResearchWal::open(&self.ledger.root.join("wal"), &self.ledger.run_id)?;
        wal.append(event_type, payload)?;
        Ok(())
    }

    fn refuted_signatures(&self) -> Result<BTreeSet<(String, String, String, String)>> {
        let mut sigs = BTreeSet::new();
        for rec in self.ledger.prior_failures(None)? {
            let arm = rec
                .config
                .get("arm")
                .and_then(Value::as_str)
                .unwrap_or(&rec.experiment_id);
            let var = rec
                .config
                .get("variant")
                .and_then(Value::as_str)
                .unwrap_or("");
            let key_arm = if var.is_empty() {
                arm.to_string()
            } else {
                format!("{arm}:{var}")
            };
            let commit12: String = rec.commit.chars().take(12).collect();
            let b = if rec.hardware.backend.is_empty() {
                "CPU".to_string()
            } else {
                rec.hardware.backend.to_ascii_uppercase()
            };
            sigs.insert((rec.study, key_arm, commit12, b));
        }
        for t in self.trials()? {
            if matches!(t.state.as_str(), "refuted" | "failed" | "blocked")
                || matches!(t.verdict.as_str(), "refuted" | "loss")
            {
                let commit12: String = t.commit.chars().take(12).collect();
                let b = if t.backend.is_empty() {
                    "CPU".to_string()
                } else {
                    t.backend.to_ascii_uppercase()
                };
                sigs.insert((
                    t.objective_id,
                    format!("{}:{}", t.arm, t.variant),
                    commit12,
                    b,
                ));
            }
        }
        Ok(sigs)
    }

    fn already_completed_ids(&self) -> Result<BTreeSet<String>> {
        let mut done: BTreeSet<String> = self
            .ledger
            .query_all()?
            .into_iter()
            .map(|r| r.experiment_id)
            .collect();
        let cstate = self.ledger.cloud_state();
        if let Some(arr) = cstate["experiments"].as_array() {
            for exp in arr {
                if exp["status"].as_str() == Some("complete") {
                    if let Some(id) = exp["id"].as_str() {
                        done.insert(id.to_string());
                    }
                }
            }
        }
        Ok(done)
    }

    fn backend_compatible(required: &[String], host: &BTreeSet<String>) -> bool {
        if required.is_empty() {
            return true;
        }
        required.iter().any(|b| {
            let prefix = b.split('-').next().unwrap_or(b).trim().to_ascii_uppercase();
            host.contains(&prefix)
        })
    }

    pub fn status(&self) -> Result<Value> {
        let trials = self.trials()?;
        let recs = self.ledger.query_all()?;
        let open_ops = self.ledger.recommended_experiments(None, 50)?;
        let wins = recs
            .iter()
            .filter(|r| epistemic_verdict(&r.statistical_comparison.verdict) == "proven")
            .count();
        let failures = recs
            .iter()
            .filter(|r| epistemic_verdict(&r.statistical_comparison.verdict) == "refuted")
            .count();
        let top_ops: Vec<Value> = open_ops
            .iter()
            .take(5)
            .map(|op| {
                let mut v = serde_json::to_value(op).unwrap_or(Value::Null);
                v["priority_score"] = json!(op.priority_score());
                v
            })
            .collect();
        Ok(json!({
            "campaign": self.registry.as_ref().map(|r| r.campaign.as_str()).unwrap_or(&self.ledger.run_id),
            "host_backends": detect_host_backends().into_iter().collect::<Vec<_>>(),
            "total_experiments": recs.len(),
            "proven_experiments": wins,
            "refuted_experiments": failures,
            "open_opportunities": open_ops.len(),
            "top_opportunities": top_ops,
            "exposures_count": self.exposures.records()?.len(),
            "trials_count": trials.len(),
        }))
    }

    pub fn tick(
        &self,
        executor: &Path,
        changed_paths: Option<&[String]>,
        lineage: Option<&BTreeMap<String, LineageRule>>,
        changed_artifacts: Option<&[String]>,
        timeout_minutes: u64,
        auto_sync: bool,
        client_override: Option<&Research>,
        lease_ttl_sec: i64,
    ) -> Result<Value> {
        for mut t in self.trials()? {
            if t.state == "running" {
                if t.phase == "qualification" {
                    t.state = "blocked".into();
                    t.finished_at = now();
                    t.message =
                        "interrupted during sealed qualification; manual reconciliation required"
                            .into();
                    self.save_trial(&t)?;
                    self.wal_emit(
                        "trial_blocked",
                        &json!({"trial_id": t.trial_id, "objective_id": t.objective_id}),
                    )?;
                } else {
                    t.state = "failed".into();
                    t.finished_at = now();
                    t.message = "interrupted during screening; checkpoint retained".into();
                    self.save_trial(&t)?;
                }
            }
        }

        let env_client = if client_override.is_none() && auto_sync && token().is_ok() {
            Some(Research::from_env())
        } else {
            None
        };
        let active_client = client_override.or(env_client.as_ref());
        if let Some(c) = active_client {
            self.ledger.pull_from_cloud(c);
        }

        let impacted_ids: Option<BTreeSet<String>> =
            if changed_paths.is_some() || changed_artifacts.is_some() {
                let all_tasks: Vec<String> = if let Some(reg) = &self.registry {
                    reg.objectives.iter().map(|o| o.id.clone()).collect()
                } else if let Some(lin) = lineage {
                    lin.keys().cloned().collect()
                } else {
                    Vec::new()
                };
                let report = impacted_benchmarks(
                    &all_tasks,
                    changed_paths.unwrap_or(&[]),
                    lineage,
                    changed_artifacts.unwrap_or(&[]),
                )?;
                let set: BTreeSet<String> = report.tasks.into_iter().collect();
                if set.is_empty() {
                    return Ok(json!({"status": "skipped_unimpacted", "impacted_tasks": []}));
                }
                Some(set)
            } else {
                None
            };

        let host_backends = detect_host_backends();
        let git_info = provenance::git_state(&self.repo_dir);
        let commit_full = if git_info.git_sha.is_empty() {
            "0".repeat(40)
        } else {
            git_info.git_sha.clone()
        };
        let commit12: String = commit_full.chars().take(12).collect();
        let branch = if git_info.git_branch.is_empty() {
            "main".to_string()
        } else {
            git_info.git_branch.clone()
        };
        let refuted_sigs = self.refuted_signatures()?;
        let completed_ids = self.already_completed_ids()?;

        // Select work from OpportunityQueue first, else CampaignRegistry.
        let mut selected: Option<(
            ObjectiveSpec,
            SearchArm,
            Option<Opportunity>,
            String,
            String,
        )> = None;

        for op in self.ledger.recommended_experiments(None, 100)? {
            if let Some(imp) = &impacted_ids {
                if !imp.contains(&op.domain) && !imp.contains(&op.objective) {
                    continue;
                }
            }
            let target_b = op
                .opportunity_id
                .split_once("->backend:")
                .map(|(_, b)| b.to_string())
                .unwrap_or_default();
            if !target_b.is_empty()
                && !Self::backend_compatible(std::slice::from_ref(&target_b), &host_backends)
            {
                continue;
            }
            let chosen_backend = if !target_b.is_empty() {
                target_b
                    .split('-')
                    .next()
                    .unwrap_or("CPU")
                    .to_ascii_uppercase()
            } else {
                ["CUDA", "ROCM", "VULKAN", "METAL", "CPU"]
                    .into_iter()
                    .find(|&b| host_backends.contains(b))
                    .unwrap_or("CPU")
                    .to_string()
            };
            if refuted_sigs.contains(&(
                op.domain.clone(),
                op.opportunity_id.clone(),
                commit12.clone(),
                chosen_backend.clone(),
            )) {
                continue;
            }
            let matched_obj = if let Some(reg) = &self.registry {
                let Some(found) = reg
                    .objectives
                    .iter()
                    .find(|o| o.id == op.domain || o.study == op.domain)
                else {
                    continue;
                };
                found.clone()
            } else {
                ObjectiveSpec {
                    id: op.domain.clone(),
                    study: op.domain.clone(),
                    objective: op.objective.clone(),
                    metric: "throughput".into(),
                    unit: "tok/s".into(),
                    direction: "higher".into(),
                    dataset: op.domain.clone(),
                    ..ObjectiveSpec::default()
                }
            };
            let op_hash = &sha256_bytes(op.opportunity_id.as_bytes())[..10];
            let exp_id = format!("{}:opp:{op_hash}:{commit12}", op.domain);
            if completed_ids.contains(&exp_id) {
                let _ = self
                    .ledger
                    .opportunities()?
                    .mark_status(&op.opportunity_id, "completed");
                continue;
            }
            let arm = SearchArm {
                arm: "opportunity_transfer".into(),
                variant: op.opportunity_id.clone(),
                hypothesis: op.proposed_approach.clone(),
                config: BTreeMap::from([
                    ("opportunity_id".into(), json!(op.opportunity_id)),
                    ("trigger_rule".into(), json!(op.trigger_rule)),
                ]),
            };
            selected = Some((matched_obj, arm, Some(op), chosen_backend, exp_id));
            break;
        }

        if selected.is_none() {
            if let Some(reg) = &self.registry {
                let trials = self.trials()?;
                let mut candidates_pool: Vec<(
                    usize,
                    String,
                    ObjectiveSpec,
                    SearchArm,
                    String,
                    String,
                )> = Vec::new();
                for obj in &reg.objectives {
                    if obj.protected {
                        continue;
                    }
                    if let Some(imp) = &impacted_ids {
                        if !imp.contains(&obj.id) && !imp.contains(&obj.study) {
                            continue;
                        }
                    }
                    if !Self::backend_compatible(&obj.backends, &host_backends) {
                        continue;
                    }
                    let chosen_backend = obj
                        .backends
                        .iter()
                        .map(|b| b.split('-').next().unwrap_or(b).to_ascii_uppercase())
                        .find(|b| host_backends.contains(b))
                        .or_else(|| {
                            ["CUDA", "ROCM", "VULKAN", "METAL", "CPU"]
                                .into_iter()
                                .map(str::to_string)
                                .find(|b| host_backends.contains(b))
                        })
                        .unwrap_or_else(|| "CPU".into());

                    let already_won = trials.iter().any(|t| {
                        t.objective_id == obj.id
                            && t.state == "qualified"
                            && t.verdict == "proven"
                            && t.backend.eq_ignore_ascii_case(&chosen_backend)
                    });
                    if already_won {
                        continue;
                    }

                    let arms_list = reg
                        .search_space
                        .get(&obj.family)
                        .cloned()
                        .unwrap_or_default();
                    let arm_pairs: Vec<(String, String)> = arms_list
                        .iter()
                        .map(|a| (a.arm.clone(), a.variant.clone()))
                        .collect();
                    let seen_rows: Vec<(String, String, Option<f64>, String)> = trials
                        .iter()
                        .filter(|t| t.objective_id == obj.id)
                        .map(|t| (t.arm.clone(), t.variant.clone(), t.score, t.state.clone()))
                        .collect();
                    let blocked: BTreeSet<(String, String)> = arm_pairs
                        .iter()
                        .filter(|(a, v)| {
                            refuted_sigs.contains(&(
                                obj.id.clone(),
                                format!("{a}:{v}"),
                                commit12.clone(),
                                chosen_backend.clone(),
                            )) || completed_ids.contains(&format!("{}:{a}:{v}:{commit12}", obj.id))
                        })
                        .cloned()
                        .collect();

                    let ranked = arm_candidates(
                        &arm_pairs,
                        &seen_rows,
                        self.max_trials_per_objective,
                        obj.direction == "higher",
                        &blocked,
                    );
                    let Some((_, _, best_a, best_v)) = ranked.first() else {
                        continue;
                    };
                    let Some(arm_spec) = arms_list
                        .into_iter()
                        .find(|a| &a.arm == best_a && &a.variant == best_v)
                    else {
                        continue;
                    };
                    let exp_id = format!(
                        "{}:{}:{}:{commit12}",
                        obj.id, arm_spec.arm, arm_spec.variant
                    );
                    candidates_pool.push((
                        seen_rows.len(),
                        obj.id.clone(),
                        obj.clone(),
                        arm_spec,
                        chosen_backend,
                        exp_id,
                    ));
                }
                candidates_pool.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
                if let Some((_, _, obj, arm, backend, exp_id)) = candidates_pool.into_iter().next()
                {
                    selected = Some((obj, arm, None, backend, exp_id));
                }
            }
        }

        let Some((obj, arm, op, backend, exp_id)) = selected else {
            return Ok(
                json!({"status": "idle", "reason": "no eligible open opportunities or unexplored campaign arms"}),
            );
        };

        let campaign_name = self
            .registry
            .as_ref()
            .map(|r| r.campaign.clone())
            .unwrap_or_else(|| self.ledger.run_id.clone());
        let lease_target = op
            .as_ref()
            .map(|o| o.opportunity_id.clone())
            .unwrap_or_else(|| exp_id.clone());

        let lease_mgr = CloudLeaseManager::new(
            active_client.cloned(),
            &self.worker_id,
            &commit12,
            Some(&self.ledger.root),
        );
        let lease_out = lease_mgr.acquire_lease(&campaign_name, &lease_target, lease_ttl_sec)?;
        let Some(mut active_lease) = lease_out.lease else {
            return Ok(json!({
                "status": "lease_busy",
                "holder_worker_id": lease_out.holder_worker_id,
                "fencing_token": lease_out.fencing_token,
                "expires_at_unix": lease_out.expires_at_unix,
            }));
        };

        let mut plan = self
            .registry
            .as_ref()
            .map(|r| r.plan.clone())
            .unwrap_or_default();
        plan.higher_is_better = obj.direction == "higher";

        let trial_id = sha256_bytes(
            format!(
                "{exp_id}:{}:{}:{backend}:{}",
                arm.arm, arm.variant, active_lease.fencing_token
            )
            .as_bytes(),
        )[..20]
            .to_string();
        let ckpt_key =
            sha256_bytes(format!("{exp_id}:{}:{}:{backend}", arm.arm, arm.variant).as_bytes())
                [..24]
                .to_string();
        let ckpt_path = self.checkpoints_dir.join(format!("{ckpt_key}.looks.json"));

        let mut trial = TrialSummary {
            trial_id: trial_id.clone(),
            objective_id: obj.id.clone(),
            arm: arm.arm.clone(),
            variant: arm.variant.clone(),
            commit: commit_full.clone(),
            backend: backend.clone(),
            shapes: obj.shapes.clone(),
            state: "running".into(),
            phase: "micro".into(),
            score: None,
            verdict: "inconclusive".into(),
            opportunity_id: op
                .as_ref()
                .map(|o| o.opportunity_id.clone())
                .unwrap_or_default(),
            started_at: now(),
            finished_at: String::new(),
            message: String::new(),
        };

        let run_result = (|| -> Result<(Value, &'static str)> {
            let _gpu_guard = GpuLock::acquire(&self.state_dir)?;
            self.save_trial(&trial)?;
            self.wal_emit(
                "trial_started",
                &json!({
                    "trial_id": trial_id,
                    "experiment_id": exp_id,
                    "objective_id": obj.id,
                    "arm": arm.arm,
                    "variant": arm.variant,
                    "backend": backend,
                    "commit": commit_full,
                    "fencing_token": active_lease.fencing_token,
                    "reclaimed_from": active_lease.reclaimed_from,
                }),
            )?;

            let mut exec_failed_reason = String::new();
            let total_stages = plan.stages.len();
            let runner = ProgressiveAbRunner::new(plan.clone());

            let host_info = provenance::host();
            let mut config_map = arm.config.clone();
            config_map.insert("arm".into(), json!(arm.arm));
            config_map.insert("variant".into(), json!(arm.variant));
            config_map.insert("fencing_token".into(), json!(active_lease.fencing_token));

            let mut exp_record = ExperimentRecordV2 {
                project: "default".into(),
                study: if obj.study.is_empty() {
                    obj.id.clone()
                } else {
                    obj.study.clone()
                },
                objective: if obj.objective.is_empty() {
                    format!("Optimize {} ({})", obj.id, obj.metric)
                } else {
                    obj.objective.clone()
                },
                hypothesis: if arm.hypothesis.is_empty() {
                    format!(
                        "{}/{} improves {} on {}",
                        arm.arm, arm.variant, obj.metric, obj.id
                    )
                } else {
                    arm.hypothesis.clone()
                },
                experiment_id: exp_id.clone(),
                parent_experiment: format!("{}:baseline", obj.id),
                prior_best: obj.baseline,
                repo: self
                    .registry
                    .as_ref()
                    .filter(|r| !r.repo.is_empty())
                    .map(|r| r.repo.clone())
                    .unwrap_or_else(|| "hanzoai/research".into()),
                commit: commit_full.clone(),
                branch: branch.clone(),
                hardware: HardwareSpec {
                    host: host_info.hostname,
                    architecture: host_info.platform,
                    gpu: std::env::var("RESEARCH_GPU_MODEL").unwrap_or_else(|_| backend.clone()),
                    backend: backend.clone(),
                    precision: arm
                        .config
                        .get("precision")
                        .and_then(Value::as_str)
                        .unwrap_or("BF16")
                        .into(),
                    ..HardwareSpec::default()
                },
                config: config_map,
                workload_manifest: WorkloadManifest {
                    manifest_id: format!("{}:{}:{}", obj.id, arm.arm, arm.variant),
                    dataset: if obj.dataset.is_empty() {
                        obj.id.clone()
                    } else {
                        obj.dataset.clone()
                    },
                    split: "progressive-ab".into(),
                    shapes: if obj.shapes.is_empty() {
                        vec!["default".into()]
                    } else {
                        obj.shapes.clone()
                    },
                    input_lengths: plan.stages.iter().map(|s| s.prompt_len).collect(),
                    batch_size: 1,
                    seed: 17,
                    manifest_sha256: String::new(),
                },
                statistical_comparison: StatisticalComparison {
                    metric: obj.metric.clone(),
                    unit: obj.unit.clone(),
                    higher_is_better: obj.direction == "higher",
                    ..StatisticalComparison::default()
                },
                correctness: CorrectnessChecks::default(),
                next_hypothesis: arm
                    .config
                    .get("next_hypothesis")
                    .and_then(Value::as_str)
                    .unwrap_or("")
                    .into(),
                ..ExperimentRecordV2::default()
            };

            let study_name = exp_record.study.clone();
            let (ab_res, _receipt) = runner.run_and_record(
                &self.ledger,
                &mut exp_record,
                Some(&ckpt_path),
                &[],
                |stage_idx: usize, stage: &ProgressiveStage| -> Result<ProgressiveStageOutput> {
                    active_lease = lease_mgr.renew_lease(&active_lease, lease_ttl_sec)?;
                    let is_final = stage_idx == total_stages;
                    let eval_sha = if obj.eval_sha.is_empty() {
                        sha256_bytes(
                            format!("{}:{}:{}", obj.id, stage.name, obj.dataset).as_bytes(),
                        )
                    } else {
                        obj.eval_sha.clone()
                    };
                    let cand_sha = sha256_bytes(
                        format!("{commit_full}:{}:{}", arm.arm, arm.variant).as_bytes(),
                    );
                    let exp_key = format!("{eval_sha}:{}", &cand_sha[..16]);

                    if is_final {
                        lease_mgr.verify_fencing(&active_lease)?;
                        if self.exposures.is_exposed(&obj.id, &exp_key)? {
                            exec_failed_reason =
                                "sealed qualification already exposed for candidate".into();
                            return Ok(ProgressiveStageOutput {
                                stage_name: stage.name.clone(),
                                baseline_samples: vec![1.0, 1.0],
                                candidate_samples: vec![1.0, 1.0],
                                parity_passed: false,
                                memory_integrity_passed: true,
                                details: exec_failed_reason.clone(),
                                ..ProgressiveStageOutput::default()
                            });
                        }
                        trial.phase = "qualification".into();
                        self.save_trial(&trial)?;
                        self.exposures
                            .record_exposure(&obj.id, &exp_key, &cand_sha, &trial_id)?;
                    } else {
                        trial.phase = stage.name.clone();
                        self.save_trial(&trial)?;
                    }

                    let req = json!({
                        "trial_id": trial_id,
                        "experiment_id": exp_id,
                        "campaign": campaign_name,
                        "objective_id": obj.id,
                        "study": study_name,
                        "metric": obj.metric,
                        "unit": obj.unit,
                        "direction": obj.direction,
                        "arm": arm.arm,
                        "variant": arm.variant,
                        "config": arm.config,
                        "backend": backend,
                        "shapes": obj.shapes,
                        "stage_index": stage_idx,
                        "stage_name": stage.name,
                        "reps": stage.reps,
                        "prompt_len": stage.prompt_len,
                        "gen_len": stage.gen_len,
                        "commit": commit_full,
                        "eval_sha": eval_sha,
                        "candidate_sha": cand_sha,
                        "fencing_token": active_lease.fencing_token,
                    });
                    let prefix = format!("stage_{stage_idx}_{}", stage.name);
                    let res = execute_subprocess(
                        executor,
                        "stage",
                        &req,
                        &self.state_dir,
                        timeout_minutes,
                        Some(&prefix),
                    );
                    if res["status"].as_str() != Some("complete") {
                        exec_failed_reason = res["reason"]
                            .as_str()
                            .unwrap_or("stage execution failed")
                            .to_string();
                        return Ok(ProgressiveStageOutput {
                            stage_name: stage.name.clone(),
                            baseline_samples: vec![1.0, 1.0],
                            candidate_samples: vec![1.0, 1.0],
                            parity_passed: false,
                            memory_integrity_passed: false,
                            details: exec_failed_reason.clone(),
                            ..ProgressiveStageOutput::default()
                        });
                    }
                    let b_samples: Vec<f64> = res["baseline_samples"]
                        .as_array()
                        .map(|a| a.iter().filter_map(Value::as_f64).collect())
                        .unwrap_or_default();
                    let c_samples: Vec<f64> = res["candidate_samples"]
                        .as_array()
                        .map(|a| a.iter().filter_map(Value::as_f64).collect())
                        .unwrap_or_default();
                    Ok(ProgressiveStageOutput {
                        stage_name: stage.name.clone(),
                        baseline_samples: b_samples,
                        candidate_samples: c_samples,
                        parity_passed: res["parity_passed"].as_bool().unwrap_or(true),
                        memory_integrity_passed: res["memory_integrity_passed"]
                            .as_bool()
                            .unwrap_or(true),
                        max_abs_diff: res["max_abs_diff"].as_f64(),
                        ulp_diff: res["ulp_diff"].as_f64(),
                        gpu_time_sec: res["gpu_time_sec"].as_f64().unwrap_or(0.0),
                        details: res["details"].as_str().unwrap_or("").to_string(),
                    })
                },
            )?;

            if !exec_failed_reason.is_empty() && ab_res.looks.is_empty() {
                trial.state = "failed".into();
                trial.verdict = "inconclusive".into();
                trial.finished_at = now();
                trial.message = exec_failed_reason.clone();
                self.save_trial(&trial)?;
                return Ok((
                    json!({
                        "status": "failed",
                        "trial_id": trial_id,
                        "experiment_id": exp_id,
                        "reason": exec_failed_reason,
                    }),
                    "released",
                ));
            }

            let ev = epistemic_verdict(&ab_res.verdict);
            trial.state = match ev {
                "proven" => "qualified",
                "refuted" => "refuted",
                _ => "explored",
            }
            .into();
            trial.verdict = ev.into();
            trial.score = Some(exp_record.statistical_comparison.candidate_mean);
            trial.finished_at = now();
            trial.message = if ab_res.failure_diagnosis.is_empty() {
                exp_record.conclusion.clone()
            } else {
                ab_res.failure_diagnosis.clone()
            };
            self.save_trial(&trial)?;

            if let Some(op_item) = &op {
                let mut q = self.ledger.opportunities()?;
                let st = match ev {
                    "proven" => "completed",
                    "refuted" => "refuted",
                    _ => "explored",
                };
                q.mark_status(&op_item.opportunity_id, st)?;
            }

            let sync_receipts = if let Some(c) = active_client.filter(|_| auto_sync) {
                self.ledger.sync_to_cloud(c)?
            } else {
                Vec::new()
            };

            Ok((
                json!({
                    "status": trial.state,
                    "trial_id": trial_id,
                    "experiment_id": exp_id,
                    "verdict": ev,
                    "fencing_token": active_lease.fencing_token,
                    "reclaimed_from": active_lease.reclaimed_from,
                    "stopped_stage": ab_res.stopped_stage,
                    "early_stopped": ab_res.early_stopped,
                    "candidate_mean": exp_record.statistical_comparison.candidate_mean,
                    "baseline_mean": exp_record.statistical_comparison.baseline_mean,
                    "sync_receipts": sync_receipts,
                }),
                "completed",
            ))
        })();

        match run_result {
            Ok((val, lstate)) => {
                let _ = lease_mgr.release_lease(&active_lease, lstate);
                Ok(val)
            }
            Err(err) => {
                let _ = lease_mgr.release_lease(&active_lease, "released");
                if err.to_string().contains("fenc") {
                    trial.state = "fenced".into();
                    trial.finished_at = now();
                    trial.message = err.to_string();
                    let _ = self.save_trial(&trial);
                    return Ok(json!({
                        "status": "fenced",
                        "trial_id": trial_id,
                        "reason": err.to_string(),
                    }));
                }
                Err(err)
            }
        }
    }

    pub fn loop_run(
        &self,
        executor: &Path,
        max_ticks: usize,
        budget_minutes: f64,
        idle_exit: bool,
        sleep_seconds: f64,
        changed_paths: Option<&[String]>,
        lineage: Option<&BTreeMap<String, LineageRule>>,
        changed_artifacts: Option<&[String]>,
        auto_sync: bool,
        client_override: Option<&Research>,
    ) -> Result<Value> {
        let deadline = Instant::now() + Duration::from_secs_f64(budget_minutes.max(0.1) * 60.0);
        let mut ticks_run = 0usize;
        let mut outcomes = Vec::new();
        let stop_reason;
        loop {
            if max_ticks > 0 && ticks_run >= max_ticks {
                stop_reason = "max_ticks_reached";
                break;
            }
            if Instant::now() >= deadline {
                stop_reason = "budget_exhausted";
                break;
            }
            let rem_min = deadline
                .saturating_duration_since(Instant::now())
                .as_secs()
                .div_ceil(60)
                .max(1);
            let res = self.tick(
                executor,
                changed_paths,
                lineage,
                changed_artifacts,
                rem_min,
                auto_sync,
                client_override,
                600,
            )?;
            ticks_run += 1;
            let st = res["status"].as_str().unwrap_or("").to_string();
            outcomes.push(res);
            if idle_exit && (st == "idle" || st == "skipped_unimpacted") {
                stop_reason = if st == "idle" {
                    "idle"
                } else {
                    "skipped_unimpacted"
                };
                break;
            }
            if sleep_seconds > 0.0 && Instant::now() < deadline {
                let wait = Duration::from_secs_f64(sleep_seconds)
                    .min(deadline.saturating_duration_since(Instant::now()));
                thread::sleep(wait);
            }
        }
        Ok(json!({
            "ticks": ticks_run,
            "stop_reason": stop_reason,
            "outcomes": outcomes,
            "status_summary": self.status()?,
        }))
    }
}
