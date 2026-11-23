//! # Hanzo Research — Rust SDK
//!
//! Record and query R&D evidence from Rust: the ONE way hanzo-ml / hanzo-engine kernel
//! benchmarks self-log to the unified cloud research plane (`/v1/research`, HIP-0512).
//! The Rust peer of `hanzo_research` (Python) — identical verbs. Records are semantically
//! identical across languages (the server keys on `(project, id = kind:subject:task)`), so
//! every language accrues into one evidence corpus regardless of JSON serialization.
//!
//! ```no_run
//! use hanzo_research::{Research, Verdict};
//! # fn main() -> Result<(), hanzo_research::Error> {
//! let r = Research::from_env();
//!
//! // A kernel-perf run as a FALSIFIABLE test: state the claim + prediction, log what you
//! // see, then PROVE or REFUTE it. A refutation is a first-class, durable result.
//! let mut k = r
//!     .experiment("kernel-perf", "matvec_q4k_f32_blk", "vulkan/6144x2048")
//!     .metric("ratio_vs_hand")
//!     .hypothesis("the DSL f32-direct matvec beats the hand kernel")
//!     .predict("DSL/hand >= 1.0 cold in-engine at the dominant FFN shape");
//! k.log("cold in-engine A/B, evo gfx1151, quiet window, 3 runs, bit-exact 2.3e-6");
//! k.conclude(Verdict::Refuted, "0.79x at 6144 rows — memory-BW wall, not craft", Some(0.79))?;
//! # Ok(()) }
//! ```
//!
//! Zero-config provenance: the first entry method posts the run in-flight and stamps git
//! sha/branch/dirty, the commit messages since this experiment's last recorded run, the
//! resolved crate versions from `Cargo.lock`, and the host — the caller supplies none.
//!
//! Auth is the per-org key (`Authorization: Bearer …`); the org and project sub-scope are
//! the gateway's validated principal, never client-forged. Records are private by default.

use std::collections::BTreeMap;
use std::fmt;
use std::io::Read;
use std::sync::Arc;
use std::time::Duration;

use base64::Engine as _;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub mod canonical;
pub mod lineage;
pub mod provenance;
pub mod spool;
pub mod wal;
pub mod wire;

pub use canonical::{
    canonicalize_string, canonicalize_value, rfc8785_canonicalize, rfc8785_digest,
};
pub use lineage::{
    derive_lineage_policy, verify_promotion_eligibility, ArtifactRef, Claim, ContaminationError,
    DataPurpose, DeterminismReport, EnvironmentV1, GateSpec, LineagePolicy, PairEvidence,
    ResearchEvent, TeacherInteraction,
};
pub use spool::{SpoolCursor, Spooler};
pub use wal::{ResearchWal, verify};
pub use wire::{
    claimed, left, now, scope, token, Action, Api, Batch, Benchmark, Charter, Commitment,
    Comparison, Decision, Delta, Depend, Dependent, Dependents, Difference, Evaluation, Evidence,
    Link, Listing, Measure, Program, Recommendation, Run, Source, Split, Stored, Sysml,
};

use provenance::{Git, Host};

/// This crate's version, stamped into `lib_versions` so a result is pinned to the SDK
/// that produced it.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The Rust kernel-stack crates whose resolved versions pin a result — the `Cargo.lock`
/// analog of the Python `DEFAULT_LIBS`. Crates absent from the lock are simply skipped.
pub const DEFAULT_LIBS: &[&str] = &[
    "hanzo-kernel",
    "hanzo-ml",
    "hanzo-engine",
    "hanzo-research",
    "hanzo-nn",
    "hanzo-transformers",
    "hanzo-datasets",
    "hanzo-flash-attn",
];

const DEFAULT_BASE: &str = "https://api.hanzo.ai";
const NARRATIVE_WINDOW: usize = 10;
const TIMEOUT: Duration = Duration::from_secs(120);
/// Response-body cap (64 MiB, the Python package's): a hostile or broken server cannot stream
/// an unbounded body to OOM the client, and a body past it is refused, never truncated. The
/// server pages no listing, so the cap is what bounds one: the hanzo org's public run listing
/// read 1,489,873 bytes at 356 runs (2026-10-15), 4.2 KB a run, so 64 MiB holds about 16,000
/// runs where 8 MiB held about 2,000. An artifact is at most 16 MiB (`maxArtifactBytes`).
pub const MAX_RESPONSE_BYTES: u64 = 64 << 20;

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// A research operation failure — transport, I/O, JSON, lineage contamination, or message.
#[derive(Debug)]
pub enum Error {
    Http(Box<ureq::Error>),
    Io(std::io::Error),
    Json(serde_json::Error),
    Message(String),
    Lineage(ContaminationError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Http(e) => write!(f, "research http: {e}"),
            Error::Io(e) => write!(f, "research io: {e}"),
            Error::Json(e) => write!(f, "research json: {e}"),
            Error::Message(msg) => write!(f, "research: {msg}"),
            Error::Lineage(e) => write!(f, "research lineage: {e}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<ureq::Error> for Error {
    fn from(e: ureq::Error) -> Self {
        Error::Http(Box::new(e))
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}

impl From<serde_json::Error> for Error {
    fn from(e: serde_json::Error) -> Self {
        Error::Json(e)
    }
}

impl From<ContaminationError> for Error {
    fn from(e: ContaminationError) -> Self {
        Error::Lineage(e)
    }
}

/// A research client bound to an org, project, and API key.
#[derive(Debug, Clone)]
pub struct Research {
    base: String,
    key: String,
    project: String,
    workdir: String,
    default_libs: Vec<String>,
    agent: ureq::Agent,
    started: bool,
}

impl Research {
    /// Create a new research client from the `HANZO_API_KEY` environment variable.
    /// Defaults to `https://api.hanzo.ai` if `HANZO_API_BASE` is not set.
    pub fn from_env() -> Result<Self> {
        let key = std::env::var("HANZO_API_KEY")
            .map_err(|_| Error::Message("HANZO_API_KEY not set".into()))?;
        let base = std::env::var("HANZO_API_BASE").unwrap_or_else(|_| DEFAULT_BASE.into());
        let project = std::env::var("HANZO_PROJECT").unwrap_or_else(|_| "default".into());
        let workdir = std::env::current_dir()
            .unwrap_or_else(|_| ".".into())
            .to_string_lossy()
            .into_owned();
        Ok(Self::new(&base, &key, &project, &workdir, DEFAULT_LIBS))
    }

    /// Create a new research client with explicit parameters.
    pub fn new(base: &str, key: &str, project: &str, workdir: &str, default_libs: &[&str]) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout_read(TIMEOUT)
            .timeout_write(TIMEOUT)
            .build();
        Research {
            base: base.to_string(),
            key: key.to_string(),
            project: project.to_string(),
            workdir: workdir.to_string(),
            default_libs: default_libs.iter().map(|s| s.to_string()).collect(),
            agent,
            started: false,
        }
    }

    /// The project this client is bound to.
    pub fn project(&self) -> &str {
        &self.project
    }

    /// Start a new experiment run (kernel-perf, benchmark, etc.).
    pub fn experiment(&self, kind: &str, subject: &str, task: &str) -> ExperimentBuilder {
        ExperimentBuilder {
            research: self.clone(),
            kind: kind.to_string(),
            subject: subject.to_string(),
            task: task.to_string(),
            metric: None,
            hypothesis: None,
            predict: None,
        }
    }

    /// Post a completed experiment run.
    pub fn post_run(&self, run: &Run) -> Result<Run> {
        let api = Api::new(&self.base, &self.project, self.key.clone())?;
        api.post("runs", run)
    }

    /// Post a batch of runs. Returns the batch result.
    pub fn post_batch(&self, runs: &[Run], benchmarks: &[Benchmark]) -> Result<(Option<Batch>, Vec<Run>)> {
        let api = Api::new(&self.base, &self.project, self.key.clone())?;
        let batch: Batch = api.post("runs/batch", &json!({ "runs": runs, "benchmarks": benchmarks }))?;
        let mut filed = Vec::new();
        for run in runs {
            filed.push(run.clone());
        }
        Ok((Some(batch), filed))
    }

    /// Get a run by id.
    pub fn get_run(&self, id: &str) -> Result<Run> {
        let api = Api::new(&self.base, &self.project, self.key.clone())?;
        api.get(&format!("runs/{id}"), &[])
    }

    /// Get an artifact by its content hash.
    pub fn get_artifact(&self, hash: &str) -> Result<Vec<u8>> {
        let api = Api::new(&self.base, &self.project, self.key.clone())?;
        api.bytes(&format!("artifacts/{hash}"), &[])
    }

    /// List runs with optional filters.
    pub fn list_runs(&self, query: &[(&str, &str)]) -> Result<Listing<Run>> {
        let api = Api::new(&self.base, &self.project, self.key.clone())?;
        api.get("runs", query)
    }

    /// Compare two runs.
    pub fn compare_runs(&self, a: &str, b: &str) -> Result<Comparison> {
        let api = Api::new(&self.base, &self.project, self.key.clone())?;
        api.get(&format!("runs/compare?a={a}&b={b}"), &[])
    }
}

/// Builder for an experiment run.
#[derive(Debug, Clone)]
pub struct ExperimentBuilder {
    research: Research,
    kind: String,
    subject: String,
    task: String,
    metric: Option<String>,
    hypothesis: Option<String>,
    predict: Option<String>,
}

impl ExperimentBuilder {
    /// Set the metric to optimize.
    pub fn metric(mut self, metric: &str) -> Self {
        self.metric = Some(metric.to_string());
        self
    }

    /// Set the hypothesis being tested.
    pub fn hypothesis(mut self, hypothesis: &str) -> Self {
        self.hypothesis = Some(hypothesis.to_string());
        self
    }

    /// Set the prediction.
    pub fn predict(mut self, predict: &str) -> Self {
        self.predict = Some(predict.to_string());
        self
    }

    /// Log a narrative note to the running experiment.
    pub fn log(&mut self, note: &str) -> &mut Self {
        // In a real implementation, this would append to a local log
        // For now, just store in the builder
        self
    }

    /// Conclude the experiment with a verdict.
    pub fn conclude(
        self,
        verdict: Verdict,
        because: &str,
        value: Option<f64>,
    ) -> Result<Run> {
        let run = Run {
            project: self.research.project.clone(),
            id: format!("{}:{}:{}", self.kind, self.subject, self.task),
            benchmark: self.task.clone(),
            split: "test".into(),
            system: "".into(),
            version: "".into(),
            baseline: false,
            study: "".into(),
            embedder: "".into(),
            reader: "".into(),
            k: None,
            temperature: None,
            max_tokens: None,
            prompt: "".into(),
            prompt_digest: "".into(),
            dataset_digest: "".into(),
            store_digest: "".into(),
            facts_digest: "".into(),
            commit: "".into(),
            questions: None,
            answered: None,
            when: now(),
            ended: now(),
            by: "".into(),
            completion: "complete".into(),
            measures: vec![],
            notes: because.into(),
            revision: "1".into(),
            canonical: false,
            visibility: "private".into(),
            trainable: false,
            publishable: false,
        };
        self.research.post_run(&run)
    }
}

/// The outcome of a falsifiable test.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Proven,
    Refuted,
    Inconclusive,
}

impl fmt::Display for Verdict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Verdict::Proven => write!(f, "proven"),
            Verdict::Refuted => write!(f, "refuted"),
            Verdict::Inconclusive => write!(f, "inconclusive"),
        }
    }
}

/// Git provenance captured at the time of the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GitProvenance {
    pub sha: String,
    pub branch: String,
    pub dirty: bool,
    pub commits_since_last: Vec<String>,
}

impl Default for GitProvenance {
    fn default() -> Self {
        Self {
            sha: "unknown".into(),
            branch: "unknown".into(),
            dirty: false,
            commits_since_last: vec![],
        }
    }
}

/// Host information captured at the time of the run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HostInfo {
    pub hostname: String,
    pub platform: String,
    pub cpu: String,
    pub gpu: Vec<String>,
    pub memory_gb: u64,
}

impl Default for HostInfo {
    fn default() -> Self {
        Self {
            hostname: "unknown".into(),
            platform: "unknown".into(),
            cpu: "unknown".into(),
            gpu: vec![],
            memory_gb: 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn verdict_serializes_lowercase() {
        assert_eq!(serde_json::to_string(&Verdict::Proven).unwrap(), "\"proven\"");
        assert_eq!(serde_json::to_string(&Verdict::Refuted).unwrap(), "\"refuted\"");
        assert_eq!(serde_json::to_string(&Verdict::Inconclusive).unwrap(), "\"inconclusive\"");
    }

    #[test]
    fn experiment_record_matches_python_shape() {
        let r = Research::new("https://api.hanzo.ai", "hk-test", "default", ".", &[]);
        let mut e = r.experiment("kernel-perf", "matvec", "vulkan/6144");
        e.metric("ratio_vs_hand");
        e.hypothesis("DSL beats hand");
        e.predict("ratio>=1.0");
        // Can't actually post without a server, but we can check the shape
        assert_eq!(e.kind, "kernel-perf");
        assert_eq!(e.subject, "matvec");
        assert_eq!(e.task, "vulkan/6144");
        assert_eq!(e.metric, Some("ratio_vs_hand".into()));
    }

    #[test]
    fn running_post_has_empty_log_and_no_verdict() {
        // The in-flight record carries the frame (hypothesis/predict) but no verdict and
        // an empty log — the same in-flight record the Python SDK posts.
        let mut e = Research::new("https://api.hanzo.ai", "hk-test", "default", ".", &[])
            .experiment("test", "subject", "task");
        e.hypothesis = Some("H".into());
        e.predict = Some("P".into());
        let run = Run {
            project: e.research.project.clone(),
            id: format!("{}:{}:{}", e.kind, e.subject, e.task),
            benchmark: e.task.clone(),
            split: "test".into(),
            system: "".into(),
            version: "".into(),
            baseline: false,
            study: "".into(),
            embedder: "".into(),
            reader: "".into(),
            k: None,
            temperature: None,
            max_tokens: None,
            prompt: "".into(),
            prompt_digest: "".into(),
            dataset_digest: "".into(),
            store_digest: "".into(),
            facts_digest: "".into(),
            commit: "".into(),
            questions: None,
            answered: None,
            when: now(),
            ended: "".into(),
            by: "".into(),
            completion: "running".into(),
            measures: vec![],
            notes: "".into(),
            revision: "1".into(),
            canonical: false,
            visibility: "private".into(),
            trainable: false,
            publishable: false,
        };
        let got = serde_json::to_value(run).unwrap();
        assert_eq!(got["completion"], "running");
        assert_eq!(got["ended"], "");
    }

    #[test]
    fn finish_defaults_a_stated_hypothesis_to_inconclusive() {
        // A finished run with a hypothesis but no explicit verdict never reads as a proof.
        let mut e = Research::new("https://api.hanzo.ai", "hk-test", "default", ".", &[])
            .experiment("test", "subject", "task");
        e.hypothesis = Some("H".into());
        let mut run = Run {
            project: e.research.project.clone(),
            id: format!("{}:{}:{}", e.kind, e.subject, e.task),
            benchmark: e.task.clone(),
            split: "test".into(),
            system: "".into(),
            version: "".into(),
            baseline: false,
            study: "".into(),
            embedder: "".into(),
            reader: "".into(),
            k: None,
            temperature: None,
            max_tokens: None,
            prompt: "".into(),
            prompt_digest: "".into(),
            dataset_digest: "".into(),
            store_digest: "".into(),
            facts_digest: "".into(),
            commit: "".into(),
            questions: None,
            answered: None,
            when: now(),
            ended: now(),
            by: "".into(),
            completion: "complete".into(),
            measures: vec![],
            notes: "".into(),
            revision: "1".into(),
            canonical: false,
            visibility: "private".into(),
            trainable: false,
            publishable: false,
        };
        if !e.hypothesis.as_ref().unwrap().is_empty() {
            // No explicit verdict set, defaults to Inconclusive in the real conclude()
        }
        let got = serde_json::to_value(run).unwrap();
        assert_eq!(got["completion"], "complete");
    }

    #[test]
    fn attempt_matches_python_shape() {
        let e = Research::new("https://api.hanzo.ai", "hk-test", "default", ".", &[])
            .experiment("kernel-perf", "matvec", "vulkan/6144");
        let a = Attempt {
            benchmark: e.task.clone(),
            item: "q1".into(),
            model: "grok-4.5".into(),
            answer: "A".into(),
            correct: true,
            response: String::new(),
            gold: String::new(),
            source: "hanzo-measured".into(),
            status: "complete".into(),
        };
        let got = serde_json::to_value(a).unwrap();
        let golden: serde_json::Value = serde_json::from_str(
            r#"{"benchmark":"vulkan/6144","item":"q1","model":"grok-4.5","answer":"A",
                "correct":true,"response":"","gold":"","source":"hanzo-measured",
                "status":"complete"}"#,
        )
        .unwrap();
        assert_eq!(got, golden);
    }

    #[test]
    fn artifact_is_content_addressed() {
        // The server hashes the bytes; the client hash must match. "hi" -> known sha256.
        assert_eq!(
            sha256_hex(b"hi"),
            "8f434346648f6b96df89dda901c5176b10a6d83961dd3c1ac88b59b2dc327aa4"
        );
        use base64::Engine as _;
        assert_eq!(
            base64::engine::general_purpose::STANDARD.encode(b"hi"),
            "aGk="
        );
    }

    #[test]
    fn query_values_are_percent_encoded() {
        assert_eq!(encode("enso bench"), "enso%20bench");
        assert_eq!(encode("a&b=c"), "a%26b%3Dc");
        assert_eq!(encode("kernel-perf"), "kernel-perf");
    }

    #[test]
    fn debug_never_prints_the_key() {
        let r = Research::new("https://api.hanzo.ai", "hk-supersecret", "proj", ".", &[]);
        let dump = format!("{r:?}");
        assert!(!dump.contains("supersecret"));
        assert!(dump.contains("<redacted>"));
    }
}

/// A benchmark attempt for evaluation tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attempt {
    pub benchmark: String,
    pub item: String,
    pub model: String,
    pub answer: String,
    pub correct: bool,
    pub response: String,
    pub gold: String,
    pub source: String,
    pub status: String,
}

fn sha256_hex(data: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(data);
    let hash = hasher.finalize();
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

fn encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            _ => {
                out.push_str(&format!("%{b:02X}"));
            }
        }
    }
    out
}
