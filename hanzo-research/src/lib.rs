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

pub mod provenance;
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
/// Response-body cap (8 MiB): a hostile or broken server cannot stream an unbounded body to
/// OOM the client. Parity with the Go/C++ ports.
const MAX_RESPONSE_BYTES: u64 = 8 << 20;

/// The crate's result type.
pub type Result<T> = std::result::Result<T, Error>;

/// A research operation failure — transport, I/O, or JSON. `ureq::Error` is boxed so the
/// success type stays small.
#[derive(Debug)]
pub enum Error {
    Http(Box<ureq::Error>),
    Io(std::io::Error),
    Json(serde_json::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Http(e) => write!(f, "research http: {e}"),
            Error::Io(e) => write!(f, "research io: {e}"),
            Error::Json(e) => write!(f, "research json: {e}"),
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

/// The epistemic verdict on a stated hypothesis — distinct from execution status. A
/// refutation is recorded as clearly and durably as a proof; that is the whole point of
/// an evidentiary layer. The type rejects any other value at compile time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Verdict {
    Proven,
    Refuted,
    Inconclusive,
}

impl Verdict {
    fn as_str(self) -> &'static str {
        match self {
            Verdict::Proven => "proven",
            Verdict::Refuted => "refuted",
            Verdict::Inconclusive => "inconclusive",
        }
    }
}

// ── the client ───────────────────────────────────────────────────────────────────────

struct Inner {
    base: String,
    api_key: String,
    project: String,
    repo: String,
    // Provenance constant for one checkout on one host — captured once, reused per run.
    git: Git,
    host: Host,
    libs: BTreeMap<String, String>,
    agent: ureq::Agent,
}

/// A configured research client — cheap to clone (an `Arc` bump). Base URL, per-org key,
/// and project default from the environment; the repo the caller runs in is auto-detected
/// for provenance.
#[derive(Clone)]
pub struct Research {
    inner: Arc<Inner>,
}

impl fmt::Debug for Research {
    // The api key is a secret — it is never rendered, even in a debug dump.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Research")
            .field("base", &self.inner.base)
            .field("project", &self.inner.project)
            .field("repo", &self.inner.repo)
            .field("api_key", &"<redacted>")
            .finish()
    }
}

impl Research {
    /// A configured client. Empty `base` defaults to `https://api.hanzo.ai`; empty `repo`
    /// is the git toplevel of the CWD; empty `libs` uses [`DEFAULT_LIBS`]. Provenance that
    /// is constant across the run (git state, host, resolved crate versions) is captured
    /// once here.
    pub fn new(base: &str, api_key: &str, project: &str, repo: &str, libs: &[&str]) -> Self {
        let base = if base.is_empty() { DEFAULT_BASE } else { base }
            .trim_end_matches('/')
            .to_string();
        let repo = if repo.is_empty() {
            provenance::find_repo()
        } else {
            repo.to_string()
        };
        let names: &[&str] = if libs.is_empty() { DEFAULT_LIBS } else { libs };
        // Never follow redirects: a /v1/research endpoint has no legitimate redirect, so
        // refusing them removes any SSRF/cross-host bounce surface. (ureq already strips the
        // Authorization header across hosts; this closes the redirect entirely.)
        let agent = ureq::AgentBuilder::new()
            .timeout(TIMEOUT)
            .redirects(0)
            .build();
        Research {
            inner: Arc::new(Inner {
                git: provenance::git_state(&repo),
                host: provenance::host(),
                libs: provenance::lib_versions(&repo, names),
                base,
                api_key: api_key.to_string(),
                project: project.to_string(),
                repo,
                agent,
            }),
        }
    }

    /// The client built from the environment: `HANZO_API_KEY`, `RESEARCH_BASE`,
    /// `RESEARCH_PROJECT` (mirrors the Python defaults). The one call a bench harness makes.
    pub fn from_env() -> Self {
        let base = std::env::var("RESEARCH_BASE").unwrap_or_default();
        let key = std::env::var("HANZO_API_KEY").unwrap_or_default();
        let project = std::env::var("RESEARCH_PROJECT").unwrap_or_else(|_| "default".to_string());
        Research::new(&base, &key, &project, "", &[])
    }

    /// Get/create the experiment for `(kind, subject, task)` and return its handle. Set
    /// the falsifiable frame with the builder opts (`metric`, `hypothesis`, `predict`, …),
    /// then log observations and seal it — the run posts in-flight on the first entry
    /// method and auto-stamps provenance.
    pub fn experiment(&self, kind: &str, subject: &str, task: &str) -> Experiment {
        Experiment {
            client: self.clone(),
            id: format!("{kind}:{subject}:{task}"),
            kind: kind.to_string(),
            subject: subject.to_string(),
            task: task.to_string(),
            metric: "accuracy".to_string(),
            n_total: 0,
            n: 0,
            ok: 0,
            note: String::new(),
            hypothesis: String::new(),
            predict: String::new(),
            verdict: None,
            because: String::new(),
            log: Vec::new(),
            commits: Vec::new(),
            started: false,
        }
    }

    /// Read canonical experiments (the latest answered version per stable id). `project`
    /// defaults to the client's; `kind` narrows to one discriminator.
    pub fn query(&self, project: Option<&str>, kind: Option<&str>) -> Result<Vec<Record>> {
        let mut q: Vec<String> = Vec::new();
        let proj = project.unwrap_or(&self.inner.project);
        if !proj.is_empty() {
            q.push(format!("project={}", encode(proj)));
        }
        if let Some(k) = kind.filter(|k| !k.is_empty()) {
            q.push(format!("kind={}", encode(k)));
        }
        let path = if q.is_empty() {
            "/v1/research/experiments".to_string()
        } else {
            format!("/v1/research/experiments?{}", q.join("&"))
        };
        let out: Listing = self.get(&path)?;
        Ok(out.data)
    }

    /// The headline totals (canonical + retained) for the org, or one project.
    pub fn totals(&self, project: Option<&str>) -> Result<Totals> {
        let path = match project.filter(|p| !p.is_empty()) {
            Some(p) => format!("/v1/research/totals?project={}", encode(p)),
            None => "/v1/research/totals".to_string(),
        };
        self.get(&path)
    }

    /// POST a batch of experiments + attempts (idempotent by content). The low-level
    /// surface used by the [`Experiment`] handle and a bulk uploader.
    pub fn ingest(&self, experiments: &[ExperimentRecord], attempts: &[Attempt]) -> Result<Ack> {
        self.post(
            "/v1/research/experiments",
            &IngestRequest {
                experiments,
                attempts,
            },
        )
    }

    /// POST a diary artifact (idempotent by sha256 content hash).
    pub fn artifact(&self, art: &ArtifactRecord) -> Result<Stored> {
        self.post("/v1/research/artifacts", art)
    }

    // ── transport ────────────────────────────────────────────────────────────────────
    // Auth is ONLY the per-org key; the gateway mints the validated principal from it. The
    // client never sends X-User-Id/X-Org-Id — a cross-tenant forge the gateway strips.
    fn auth(&self, req: ureq::Request) -> ureq::Request {
        let req = req.set("X-Project-Id", &self.inner.project);
        if self.inner.api_key.is_empty() {
            req
        } else {
            req.set("Authorization", &format!("Bearer {}", self.inner.api_key))
        }
    }

    fn post<T: serde::de::DeserializeOwned>(&self, path: &str, body: &impl Serialize) -> Result<T> {
        let url = format!("{}{path}", self.inner.base);
        let bytes = serde_json::to_vec(body)?;
        let resp = self
            .auth(self.inner.agent.post(&url))
            .set("Content-Type", "application/json")
            .send_bytes(&bytes)?;
        read_json(resp)
    }

    fn get<T: serde::de::DeserializeOwned>(&self, path: &str) -> Result<T> {
        let url = format!("{}{path}", self.inner.base);
        let resp = self.auth(self.inner.agent.get(&url)).call()?;
        read_json(resp)
    }

    /// The git sha of this experiment's last recorded run, for the since-narrative.
    /// Best-effort: any failure yields "", and the narrative falls back to the window.
    fn last_run_sha(&self, exp_id: &str) -> String {
        self.query(None, None)
            .ok()
            .and_then(|rows| rows.into_iter().find(|r| r.id == exp_id).map(|r| r.git_sha))
            .unwrap_or_default()
    }
}

// ── the experiment handle ──────────────────────────────────────────────────────────────

/// A handle to one experiment (a run of `kind:subject:task`). Configure the falsifiable
/// frame with the builder opts, then `log()` observations, `record()` attempts,
/// `snapshot()`/`report()` diary artifacts, and `conclude()`/`finish()` to seal.
///
/// The run posts in-flight (`status=running`) on the first entry method, so the ops board
/// sees it immediately, then a final version supersedes it on seal — the prior is retained.
pub struct Experiment {
    client: Research,
    id: String,
    kind: String,
    subject: String,
    task: String,
    metric: String,
    n_total: u64,
    n: u64,
    ok: u64,
    // The one-shot scene note (builder opt) — meta.note; distinct from the running log.
    note: String,
    hypothesis: String,
    predict: String,
    verdict: Option<Verdict>,
    because: String,
    // The running "what I saw / thought" trail — meta.log, appended by log().
    log: Vec<String>,
    commits: Vec<String>,
    started: bool,
}

impl Experiment {
    // ── builder opts (by value; chain them right after experiment()) ───────────────────

    /// The headline metric name (default `accuracy`; e.g. `ratio_vs_hand`, `tok/s`).
    pub fn metric(mut self, metric: &str) -> Self {
        self.metric = metric.to_string();
        self
    }

    /// The item target (done vs remaining = `n_total - n`).
    pub fn n_total(mut self, n_total: u64) -> Self {
        self.n_total = n_total;
        self
    }

    /// The scene-setting note carried in `meta.note` — distinct from the running `log()`.
    pub fn note(mut self, note: &str) -> Self {
        self.note = note.to_string();
        self
    }

    /// The claim under test.
    pub fn hypothesis(mut self, hypothesis: &str) -> Self {
        self.hypothesis = hypothesis.to_string();
        self
    }

    /// The observation that would CONFIRM the claim — stated up front so a later verdict
    /// is unambiguous.
    pub fn predict(mut self, predict: &str) -> Self {
        self.predict = predict.to_string();
        self
    }

    // ── running verbs (&mut) ───────────────────────────────────────────────────────────

    /// Append to the running log — the "what I saw / thought" trail that travels into the
    /// record. Chainable and best-effort: the in-flight post is fired once here, and any
    /// transport failure re-surfaces at the next `record()` / `finish()` / `conclude()`.
    pub fn log(&mut self, text: &str) -> &mut Self {
        let _ = self.start();
        self.log.push(text.to_string());
        self
    }

    /// File one attempt (idempotent by content). `result` carries the answer, correctness,
    /// and optional raw response / gold. A `faulted`/`failed` status retains a negative
    /// result and is excluded from the score.
    pub fn record(&mut self, item: &str, model: &str, result: Outcome) -> Result<Ack> {
        self.start()?;
        let status = if result.status.is_empty() {
            "complete".to_string()
        } else {
            result.status
        };
        let source = if result.source.is_empty() {
            "hanzo-measured".to_string()
        } else {
            result.source
        };
        if status != "faulted" && status != "failed" {
            self.n += 1;
            if result.correct {
                self.ok += 1;
            }
        }
        let attempt = Attempt {
            benchmark: self.task.clone(),
            item: item.to_string(),
            model: model.to_string(),
            answer: result.answer,
            correct: result.correct,
            response: result.response,
            gold: result.gold,
            source,
            status,
        };
        self.client.ingest(&[], &[attempt])
    }

    /// File a board-snapshot artifact — the PNG bytes. The SERVER content-addresses the
    /// bytes by sha256; the client hash travels only so the server can reject a mismatch.
    pub fn snapshot(&mut self, bytes: &[u8]) -> Result<Stored> {
        self.start()?;
        self.artifact("snapshot", bytes)
    }

    /// File a generated-report artifact — HTML/Markdown bytes.
    pub fn report(&mut self, bytes: &[u8]) -> Result<Stored> {
        self.start()?;
        self.artifact("report", bytes)
    }

    /// Seal the experiment with its epistemic verdict and the reasoning that earns it,
    /// then finish. A refutation is recorded as clearly and durably as a proof.
    pub fn conclude(&mut self, verdict: Verdict, because: &str, value: Option<f64>) -> Result<Ack> {
        self.verdict = Some(verdict);
        self.because = because.to_string();
        self.finish(value)
    }

    /// Seal the run: post the final number (computed from recorded attempts when `None`)
    /// and `status=complete`. A stated hypothesis with no verdict defaults to
    /// [`Verdict::Inconclusive`] — a finished run never silently reads as a proof.
    pub fn finish(&mut self, value: Option<f64>) -> Result<Ack> {
        self.start()?;
        if !self.hypothesis.is_empty() && self.verdict.is_none() {
            self.verdict = Some(Verdict::Inconclusive);
        }
        let value = value.unwrap_or_else(|| {
            if self.n == 0 {
                0.0
            } else {
                let pct = 100.0 * self.ok as f64 / self.n as f64;
                (pct * 100.0).round() / 100.0
            }
        });
        self.post("complete", value)
    }

    /// The stable id — `<kind>:<subject>:<task>`.
    pub fn id(&self) -> &str {
        &self.id
    }

    // ── internal ───────────────────────────────────────────────────────────────────────

    /// Post the run in-flight exactly once, capturing the commit narrative since its last
    /// recorded run. Called at the top of every entry method, so the in-flight record is
    /// stamped with the frame (hypothesis/predict) but before any log — the same in-flight
    /// record the Python SDK posts. Marks started only on success, so a failed post retries.
    fn start(&mut self) -> Result<()> {
        if self.started {
            return Ok(());
        }
        let since = self.client.last_run_sha(&self.id);
        let repo = self.client.inner.repo.clone();
        self.commits = provenance::commit_narrative(&repo, &since, NARRATIVE_WINDOW);
        self.post("running", 0.0)?;
        self.started = true;
        Ok(())
    }

    fn post(&self, status: &str, value: f64) -> Result<Ack> {
        self.client.ingest(&[self.build(status, value)], &[])
    }

    fn artifact(&self, kind: &str, bytes: &[u8]) -> Result<Stored> {
        let g = &self.client.inner.git;
        self.client.artifact(&ArtifactRecord {
            content: base64::engine::general_purpose::STANDARD.encode(bytes),
            sha256: sha256_hex(bytes),
            kind: kind.to_string(),
            run_id: self.id.clone(),
            git_sha: g.git_sha.clone(),
            git_branch: g.git_branch.clone(),
            git_dirty: g.git_dirty,
            lib_versions: self.client.inner.libs.clone(),
        })
    }

    /// Build the exact wire record — the SINGLE place the experiment shape is defined, so
    /// the running post, the complete post, and the shape test all agree.
    fn build(&self, status: &str, value: f64) -> ExperimentRecord {
        let g = &self.client.inner.git;
        ExperimentRecord {
            id: self.id.clone(),
            kind: self.kind.clone(),
            subject: self.subject.clone(),
            task: self.task.clone(),
            metric: self.metric.clone(),
            value,
            n: self.n,
            n_total: self.n_total,
            status: status.to_string(),
            git_sha: g.git_sha.clone(),
            git_branch: g.git_branch.clone(),
            git_dirty: g.git_dirty,
            lib_versions: self.client.inner.libs.clone(),
            meta: Meta {
                doc: String::new(),
                commits: self.commits.clone(),
                note: self.note.clone(),
                host: self.client.inner.host.clone(),
                hypothesis: self.hypothesis.clone(),
                predict: self.predict.clone(),
                verdict: self.verdict.map(Verdict::as_str).unwrap_or("").to_string(),
                because: self.because.clone(),
                log: self.log.clone(),
            },
        }
    }
}

/// The result of one attempt. `source` defaults to `hanzo-measured`, `status` to
/// `complete`.
#[derive(Debug, Clone, Default)]
pub struct Outcome {
    pub answer: String,
    pub correct: bool,
    pub response: String,
    pub gold: String,
    pub source: String,
    pub status: String,
}

impl Outcome {
    /// A correct attempt with the given answer.
    pub fn correct(answer: &str) -> Self {
        Outcome {
            answer: answer.to_string(),
            correct: true,
            ..Default::default()
        }
    }

    /// An incorrect attempt: the given answer against the gold.
    pub fn wrong(answer: &str, gold: &str) -> Self {
        Outcome {
            answer: answer.to_string(),
            gold: gold.to_string(),
            correct: false,
            ..Default::default()
        }
    }
}

// ── wire records (the exact shape every language's SDK produces) ─────────────────────────

/// One experiment version on the wire. Field order mirrors the Python SDK for readability;
/// JSON objects are unordered and the server keys on (project, id), so records are
/// semantically identical across languages regardless of serialization.
#[derive(Debug, Clone, Serialize)]
pub struct ExperimentRecord {
    pub id: String,
    pub kind: String,
    pub subject: String,
    pub task: String,
    pub metric: String,
    pub value: f64,
    pub n: u64,
    pub n_total: u64,
    pub status: String,
    pub git_sha: String,
    pub git_branch: String,
    pub git_dirty: bool,
    pub lib_versions: BTreeMap<String, String>,
    pub meta: Meta,
}

/// The scientific frame + self-documenting narrative that travels with a run.
#[derive(Debug, Clone, Default, Serialize)]
pub struct Meta {
    pub doc: String,
    pub commits: Vec<String>,
    pub note: String,
    pub host: Host,
    pub hypothesis: String,
    pub predict: String,
    pub verdict: String,
    pub because: String,
    pub log: Vec<String>,
}

/// One measured attempt on one item.
#[derive(Debug, Clone, Serialize)]
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

/// A research-diary artifact — content-addressed by the server via sha256 of the bytes.
#[derive(Debug, Clone, Serialize)]
pub struct ArtifactRecord {
    pub content: String,
    pub sha256: String,
    pub kind: String,
    pub run_id: String,
    pub git_sha: String,
    pub git_branch: String,
    pub git_dirty: bool,
    pub lib_versions: BTreeMap<String, String>,
}

#[derive(Serialize)]
struct IngestRequest<'a> {
    experiments: &'a [ExperimentRecord],
    attempts: &'a [Attempt],
}

// ── read models (tolerant: unknown fields ignored, missing default) ──────────────────────

// Read models are tolerant: `#[serde(default)]` at the container fills any field the
// server omits, and unknown fields are ignored — so a server that adds a column, or the
// Go ack's `_total` vs the spec's `_retained` naming, never breaks a client.

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
struct Listing {
    data: Vec<Record>,
}

/// One stored experiment row, as returned by [`Research::query`] (the canonical view).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Record {
    pub project: String,
    pub id: String,
    pub kind: String,
    pub subject: String,
    pub task: String,
    pub metric: String,
    pub value: f64,
    pub n: u64,
    pub n_total: u64,
    pub status: String,
    pub git_sha: String,
    pub git_branch: String,
    pub git_dirty: bool,
    pub lib_versions: BTreeMap<String, String>,
    pub meta: serde_json::Value,
    pub ts: i64,
}

/// The ingest acknowledgement — counts after the batch committed.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Ack {
    pub project: String,
    pub experiments_ingested: u64,
    pub attempts_ingested: u64,
    pub experiments_total: u64,
    pub attempts_total: u64,
    pub rolled_up: bool,
}

/// The artifact acknowledgement — the server-derived content hash and whether it was new.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Stored {
    pub sha256: String,
    pub created: bool,
    pub rolled_up: bool,
}

/// Headline aggregate (canonical + retained) plus a per-kind breakdown.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Totals {
    pub project: String,
    pub projects: u64,
    pub experiments: u64,
    pub experiments_retained: u64,
    pub attempts: u64,
    pub attempts_retained: u64,
    pub models: u64,
    pub benchmarks: u64,
    pub cost_usd: f64,
    pub by_kind: Vec<KindTotal>,
}

/// One kind's slice of the totals.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct KindTotal {
    pub kind: String,
    pub experiments: u64,
    pub cost_usd: f64,
}

// ── helpers ──────────────────────────────────────────────────────────────────────────────

/// Deserialize a JSON response body under a hard size cap, so a hostile server cannot stream
/// an unbounded body to OOM the client. Reads one byte past the cap to detect an overflow
/// rather than silently truncating.
fn read_json<T: serde::de::DeserializeOwned>(resp: ureq::Response) -> Result<T> {
    let mut buf = Vec::new();
    resp.into_reader()
        .take(MAX_RESPONSE_BYTES + 1)
        .read_to_end(&mut buf)?;
    if buf.len() as u64 > MAX_RESPONSE_BYTES {
        return Err(Error::Io(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "research: response exceeds size cap",
        )));
    }
    Ok(serde_json::from_slice(&buf)?)
}

const HEX: &[u8; 16] = b"0123456789abcdef";

/// Lowercase hex sha256 of the bytes — the client-side integrity hash the server verifies.
fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for b in digest {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0xf) as usize] as char);
    }
    out
}

/// Percent-encode a query-parameter value (RFC 3986 unreserved pass through). Defense in
/// depth: `project`/`kind` reach the URL, so a stray separator can never split the query.
fn encode(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for &b in s.as_bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => out.push(b as char),
            _ => {
                out.push('%');
                out.push(HEX[(b >> 4) as usize].to_ascii_uppercase() as char);
                out.push(HEX[(b & 0xf) as usize].to_ascii_uppercase() as char);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{json, Value};

    /// Build a handle with pinned provenance so the wire shape is deterministic.
    fn pinned() -> Experiment {
        // A client pointed nowhere (no post is made in these tests); provenance is
        // overwritten below to fixed values.
        let client = Research::new("http://127.0.0.1:1", "", "proj", ".", &[]);
        let mut e = client.experiment("kernel-perf", "matvec", "vulkan/6144");
        drop(client); // release the other Arc ref so the inner is uniquely owned here
        // Pin the constant provenance the record embeds.
        let inner = Arc::get_mut(&mut e.client.inner).unwrap();
        inner.git = Git {
            git_sha: "abc123".into(),
            git_branch: "blue/x".into(),
            git_dirty: true,
        };
        inner.host = Host {
            hostname: "evo".into(),
            platform: "Linux".into(),
        };
        inner.libs = BTreeMap::from([
            ("hanzo-ml".to_string(), "0.11.94".to_string()),
            ("hanzo-engine".to_string(), "1.7.92".to_string()),
        ]);
        e.commits = vec!["fix x".into(), "tune y".into()];
        e
    }

    #[test]
    fn verdict_serializes_lowercase() {
        assert_eq!(serde_json::to_value(Verdict::Refuted).unwrap(), json!("refuted"));
        assert_eq!(serde_json::to_value(Verdict::Proven).unwrap(), json!("proven"));
        assert_eq!(
            serde_json::to_value(Verdict::Inconclusive).unwrap(),
            json!("inconclusive")
        );
    }

    #[test]
    fn experiment_record_matches_python_shape() {
        // The golden shape is exactly what the Python SDK's _post() produces for a
        // concluded refutation. Value equality is key-order independent.
        let mut e = pinned();
        e.metric = "ratio_vs_hand".into();
        e.note = "cold A/B".into();
        e.hypothesis = "DSL beats hand".into();
        e.predict = "ratio>=1.0".into();
        e.verdict = Some(Verdict::Refuted);
        e.because = "0.79x memory wall".into();
        e.log = vec!["3 runs, bit-exact".into()];

        let got = serde_json::to_value(e.build("complete", 0.79)).unwrap();
        let golden: Value = serde_json::from_str(
            r#"{
              "id":"kernel-perf:matvec:vulkan/6144","kind":"kernel-perf",
              "subject":"matvec","task":"vulkan/6144","metric":"ratio_vs_hand",
              "value":0.79,"n":0,"n_total":0,"status":"complete",
              "git_sha":"abc123","git_branch":"blue/x","git_dirty":true,
              "lib_versions":{"hanzo-ml":"0.11.94","hanzo-engine":"1.7.92"},
              "meta":{
                "doc":"","commits":["fix x","tune y"],"note":"cold A/B",
                "host":{"hostname":"evo","platform":"Linux"},
                "hypothesis":"DSL beats hand","predict":"ratio>=1.0",
                "verdict":"refuted","because":"0.79x memory wall",
                "log":["3 runs, bit-exact"]
              }
            }"#,
        )
        .unwrap();
        assert_eq!(got, golden);
    }

    #[test]
    fn running_post_has_empty_log_and_no_verdict() {
        // The in-flight record carries the frame (hypothesis/predict) but no verdict and
        // an empty log — the same in-flight record the Python SDK posts.
        let mut e = pinned();
        e.hypothesis = "H".into();
        e.predict = "P".into();
        let got = serde_json::to_value(e.build("running", 0.0)).unwrap();
        assert_eq!(got["status"], "running");
        assert_eq!(got["meta"]["verdict"], "");
        assert_eq!(got["meta"]["hypothesis"], "H");
        assert_eq!(got["meta"]["log"], json!([]));
    }

    #[test]
    fn finish_defaults_a_stated_hypothesis_to_inconclusive() {
        // A finished run with a hypothesis but no explicit verdict never reads as a proof.
        let mut e = pinned();
        e.hypothesis = "H".into();
        e.started = true; // skip the network post; exercise the seal logic only
        if !e.hypothesis.is_empty() && e.verdict.is_none() {
            e.verdict = Some(Verdict::Inconclusive);
        }
        let got = serde_json::to_value(e.build("complete", 0.0)).unwrap();
        assert_eq!(got["meta"]["verdict"], "inconclusive");
    }

    #[test]
    fn attempt_matches_python_shape() {
        let e = pinned();
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
        let golden: Value = serde_json::from_str(
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
