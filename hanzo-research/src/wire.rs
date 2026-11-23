//! Wire types for the Hanzo Research Cloud API (`/v1/research`).
//!
//! Provides the canonical structs matching `apps/research` in Hanzo Cloud:
//! - Benchmark definitions, splits, and official metrics
//! - Executions/Runs with derived completion states and measures
//! - Content-addressed artifacts
//! - Studies, Papers, and Comparisons
//! - Decision records: evidence (SysML v2 elements among it), decisions holding a DDT-1.0
//!   Decision record, evaluations holding its Evaluation Run and Kai's Decision Package, action
//!   drafts, a person's commitments, and what read a uri

use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::fmt;

fn list<'de, D: Deserializer<'de>, T: Deserialize<'de>>(d: D) -> Result<Vec<T>, D::Error> {
    Ok(Option::<Vec<T>>::deserialize(d)?.unwrap_or_default())
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Split {
    pub name: String,
    pub items: i64,
    pub definition: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Benchmark {
    pub project: String,
    pub id: String,
    pub title: String,
    pub version: String,
    pub dataset: String,
    pub license: String,
    pub origin: String,
    #[serde(deserialize_with = "list")]
    pub splits: Vec<Split>,
    pub metric: String,
    #[serde(deserialize_with = "list")]
    pub metrics: Vec<String>,
    pub citation: String,
    pub notes: String,
    pub revision: String,
    pub canonical: bool,
    pub when: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Measure {
    pub category: String,
    pub metric: String,
    pub value: f64,
    pub lo: Option<f64>,
    pub hi: Option<f64>,
    pub n: Option<i64>,
    pub of: Option<i64>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Run {
    pub project: String,
    pub id: String,
    pub benchmark: String,
    pub split: String,
    pub system: String,
    pub version: String,
    pub baseline: bool,
    pub study: String,
    pub embedder: String,
    pub reader: String,
    pub k: Option<i64>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i64>,
    pub prompt: String,
    pub prompt_digest: String,
    pub dataset_digest: String,
    pub store_digest: String,
    pub facts_digest: String,
    pub commit: String,
    pub questions: Option<i64>,
    pub answered: Option<i64>,
    pub when: String,
    pub ended: String,
    pub by: String,
    pub completion: String,
    #[serde(deserialize_with = "list")]
    pub measures: Vec<Measure>,
    pub notes: String,
    pub revision: String,
    pub canonical: bool,
    pub visibility: String,
    pub trainable: bool,
    pub publishable: bool,
}

impl Run {
    /// Completion as the server derives it from the counts and the end.
    pub fn completion(&self) -> &'static str {
        match (self.questions, self.answered) {
            (Some(q), Some(a)) if a > q => "inconsistent",
            (Some(q), Some(a)) if a < q => "partial",
            (Some(_), Some(_)) if self.ended.trim().is_empty() => "unknown",
            (Some(_), Some(_)) => "complete",
            _ => "unknown",
        }
    }

    /// Record a measure when it is a finite number, counted by `row`'s `n` of its `questions`.
    pub fn put(&mut self, category: &str, metric: &str, v: &Value, row: &Value) {
        if let Some(x) = v.as_f64().filter(|x| x.is_finite()) {
            self.measures.push(Measure {
                category: category.into(),
                metric: metric.into(),
                value: x,
                n: row["n"].as_i64(),
                of: row["questions"].as_i64(),
                ..Measure::default()
            });
        }
    }

    pub fn measure(&self, category: &str, metric: &str) -> Option<&Measure> {
        self.measures
            .iter()
            .find(|m| m.category == category && m.metric == metric)
    }
}

/// What a batch post filed: the project, the records new to it, the canonical ones.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Batch {
    pub project: String,
    pub recorded: i64,
    pub canonical: i64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Stored {
    pub sha256: String,
    pub created: bool,
    #[serde(default)]
    pub rolled_up: bool,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: Serialize", deserialize = "T: DeserializeOwned"))]
pub struct Listing<T> {
    #[serde(deserialize_with = "list")]
    pub data: Vec<T>,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Difference {
    pub field: String,
    pub a: String,
    pub b: String,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Delta {
    pub category: String,
    pub metric: String,
    pub a: f64,
    pub b: f64,
    pub change: f64,
    pub overlap: Option<bool>,
}

/// Two runs side by side, as `GET /v1/research/compare` answers: the two canonical runs,
/// `blocks` (why they may not be compared, empty when nothing was found), `config` (every
/// configuration field they differ in) and `deltas` (each metric both measured on one category).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Comparison {
    pub a: Option<Run>,
    pub b: Option<Run>,
    pub comparable: bool,
    #[serde(deserialize_with = "list")]
    pub blocks: Vec<String>,
    #[serde(deserialize_with = "list")]
    pub config: Vec<Difference>,
    #[serde(deserialize_with = "list")]
    pub deltas: Vec<Delta>,
}

// Decision records under `/v1/research/...`: evidence, decisions, evaluations, actions and
// commitments. Field names are camelCase, DDT's. A DDT record travels inside its envelope as a
// JSON value, verbatim. The server fills `id`, `hash` (of an evaluation or an action),
// `version`, `time`, `by`, `supersedes` and a decision's `decisionId` and `depends`; a POST
// leaves them out.

fn zero(n: &i64) -> bool {
    *n == 0
}

/// A link from an evidence record to another uri: `rel` is `satisfies`, `verifies`, `derives`,
/// `supersedes`, `refines` or `dependsOn`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Link {
    pub rel: String,
    pub uri: String,
}

/// Where evidence comes from: `{system: "sysml", revision: <commit>}`, `{system: "ddt"}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Source {
    pub system: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub revision: String,
}

/// An evidence record: a stable `uri`, its content, the lowercase hex SHA-256 of the content's
/// canonical JSON, its `kind` (`requirement`, `test`, `model`, `analysis`, `document`,
/// `element`), its source and links. A POST whose hash is not the uri's newest files its next
/// `version`, even a hash an older version had; the newest hash answers the record held. The
/// server refuses evidence without content.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Evidence {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub id: String,
    pub uri: String,
    pub hash: String,
    pub kind: String,
    pub source: Source,
    #[serde(deserialize_with = "list")]
    pub links: Vec<Link>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Value>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub time: String,
    #[serde(skip_serializing_if = "zero")]
    pub version: i64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub by: String,
}

/// A SysML v2 model's elements in Systems Modeling API JSON, under its project (no `@` or `#`)
/// and commit (lowercase hex): `POST /v1/research/evidence/sysml` files each element as
/// `sysml://<project>@<commit>#<@id>`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Sysml {
    pub project: String,
    pub commit: String,
    #[serde(deserialize_with = "list")]
    pub elements: Vec<Value>,
}

/// A decision's charter: the question it answers, its scope, owner, due date and criteria.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Charter {
    pub question: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub scope: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub owner: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub due: String,
    #[serde(deserialize_with = "list", skip_serializing_if = "Vec::is_empty")]
    pub criteria: Vec<String>,
}

/// The Decision Program a decision is bound to: its id, version and hash.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Program {
    pub id: String,
    pub version: i64,
    pub hash: String,
}

/// An evidence uri a record read, with the hash it read when it read one.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Depend {
    pub uri: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub hash: String,
}

/// A decision record: a DDT-1.0 Decision record (`decision`), its charter and the program bound
/// from it. A record whose `decision.id` the server holds files the next `version`, naming the
/// one it `supersedes`; `depends` is every evidence uri the DDT record names. A version changing
/// what the record's `agentPermissions` keep from agents (`lifecycleState` among it) is a
/// person's to file.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Decision {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub id: String,
    pub decision: Value,
    pub charter: Charter,
    pub program: Program,
    #[serde(deserialize_with = "list")]
    pub depends: Vec<Depend>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub time: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub by: String,
    #[serde(skip_serializing_if = "zero")]
    pub version: i64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub supersedes: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub decision_id: String,
}

/// An evaluation record: a DDT-1.0 Evaluation record (`evaluation`), the run it evaluates,
/// and the decision it rests on (by `decisionId` and `decisionHash`). A version whose
/// `evaluation.id` the server holds files the next `version`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Evaluation {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub id: String,
    pub evaluation: Value,
    pub run_id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub decision_id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub decision_hash: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub time: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub by: String,
    #[serde(skip_serializing_if = "zero")]
    pub version: i64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub supersedes: String,
}

/// An action draft: its DDT-1.0 action record, the decision it implements (by `decisionId` and
/// `decisionHash`), and its readiness ruling (one of `READY`, `NEEDS_REVIEW`, `NOT_READY`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Action {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub id: String,
    pub action: Value,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub decision_id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub decision_hash: String,
    pub readiness: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub time: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub by: String,
    #[serde(skip_serializing_if = "zero")]
    pub version: i64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub supersedes: String,
}

/// A person's commitment to an action: the action's `hash` (not its `id`), the person, and the
/// commitment state (`proposed`, `accepted`, `declined`, `withdrawn`).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Commitment {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub action_hash: String,
    pub person: String,
    pub state: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub time: String,
    #[serde(skip_serializing_if = "zero")]
    pub version: i64,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub supersedes: String,
}

/// A record that read a uri: its id, its DDT decision id, the hash it read, and whether a
/// person authorized it.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, rename_all = "camelCase")]
pub struct Dependent {
    pub id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub decision_id: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub hash: String,
    pub authorized: bool,
}

/// A list of dependents: `GET /v1/research/evidence/dependents` answers this.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Dependents {
    #[serde(deserialize_with = "list")]
    pub data: Vec<Dependent>,
}

/// `GET /v1/research/evidence/dependents` query params.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DependentQuery {
    pub uri: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub hash: String,
}

/// A paper record (a study bound to a decision): `POST /v1/research/papers`.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Paper {
    #[serde(skip_serializing_if = "String::is_empty")]
    pub id: String,
    pub study_id: String,
    pub decision_id: String,
    pub title: String,
    pub authors: Vec<String>,
    pub abstract_text: String,
    pub body: String,
    pub citations: Vec<String>,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub time: String,
    #[serde(skip_serializing_if = "String::is_empty")]
    pub by: String,
    #[serde(skip_serializing_if = "zero")]
    pub version: i64,
}

/// API client for `/v1/research`.
#[derive(Clone)]
pub struct Api {
    base: String,
    project: String,
    token: String,
    agent: ureq::Agent,
}

impl fmt::Debug for Api {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Api")
            .field("base", &self.base)
            .field("project", &self.project)
            .field("token", &"<redacted>")
            .finish()
    }
}

fn read_response(resp: ureq::Response) -> Result<Vec<u8>, crate::Error> {
    use std::io::Read;
    let mut body = Vec::new();
    let _ = resp.into_reader().take(1024 * 1024 * 64).read_to_end(&mut body);
    Ok(body)
}

impl Api {
    pub fn new(base: &str, project: &str, token: String) -> Result<Api, crate::Error> {
        if let Some(p) = claimed(&token) {
            if p != scope(project) {
                return Err(crate::Error::Message(format!(
                    "the token's project is {p}, not {project}: records would be filed under {p}"
                )));
            }
        }
        let agent = ureq::AgentBuilder::new()
            .timeout(std::time::Duration::from_secs(300))
            .redirects(0)
            .build();
        Ok(Api {
            base: base.trim_end_matches('/').to_string(),
            project: scope(project),
            token,
            agent,
        })
    }

    pub fn base(&self) -> &str {
        &self.base
    }

    pub fn project(&self) -> &str {
        &self.project
    }

    pub fn token(&self) -> &str {
        &self.token
    }

    /// Get the underlying ureq agent for spooler use
    pub fn agent_for_spooler(&self) -> &ureq::Agent {
        &self.agent
    }

    fn call(
        &self,
        req: ureq::Request,
        body: Option<&[u8]>,
        what: &str,
    ) -> Result<Vec<u8>, crate::Error> {
        let mut wait = 2;
        loop {
            let req = req
                .clone()
                .set("Authorization", &format!("Bearer {}", self.token))
                .set("X-Project-Id", &self.project);

            let res = match body {
                Some(b) => req.send_bytes(b),
                None => req.call(),
            };

            match res {
                Ok(resp) => return read_response(resp),
                Err(ureq::Error::Status(503, _)) if wait <= 8 => {
                    std::thread::sleep(std::time::Duration::from_secs(wait));
                    wait *= 2;
                    continue;
                }
                Err(ureq::Error::Status(code, resp)) => {
                    use std::io::Read;
                    let mut body = Vec::new();
                    let _ = resp.into_reader().take(1024).read_to_end(&mut body);
                    return Err(crate::Error::Message(format!(
                        "{what}: {code}: {}",
                        String::from_utf8_lossy(&body)
                    )));
                }
                Err(e) => {
                    return Err(crate::Error::Http(Box::new(e)));
                }
            }
        }
    }

    /// `POST /v1/research/<path>` with `body` as JSON, the answer read as `R`.
    pub fn post<B: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R, crate::Error> {
        let url = format!("{}/v1/research/{path}", self.base);
        let bytes = serde_json::to_vec(body)?;
        let req = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json");
        let resp_body = self.call(req, Some(&bytes), &format!("POST {path}"))?;
        Ok(serde_json::from_slice(&resp_body)?)
    }

    /// `GET /v1/research/<path>?<query>`, the answer's bytes.
    pub fn bytes(&self, path: &str, query: &[(&str, &str)]) -> Result<Vec<u8>, crate::Error> {
        let mut url = format!("{}/v1/research/{path}", self.base);
        if !query.is_empty() {
            let q_str = query
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding(k), urlencoding(v)))
                .collect::<Vec<_>>()
                .join("&");
            url = format!("{url}?{q_str}");
        }
        let req = self.agent.get(&url);
        self.call(req, None, &format!("GET {path}"))
    }

    /// `GET /v1/research/<path>`.
    pub fn get<R: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<R, crate::Error> {
        let body = self.bytes(path, query)?;
        Ok(serde_json::from_slice(&body)?)
    }

    /// File one evidence record: `POST /v1/research/evidence`. Answers the record as filed.
    pub fn evidence(&self, e: &Evidence) -> Result<Evidence, crate::Error> {
        self.post("evidence", e)
    }

    /// File a SysML v2 model's elements: `POST /v1/research/evidence/sysml`. Answers the records
    /// filed.
    pub fn sysml(&self, s: &Sysml) -> Result<Vec<Evidence>, crate::Error> {
        let filed: Listing<Evidence> = self.post("evidence/sysml", s)?;
        Ok(filed.data)
    }

    /// A uri's evidence versions, newest first: `GET /v1/research/evidence?uri=`.
    pub fn versions(&self, uri: &str) -> Result<Vec<Evidence>, crate::Error> {
        let held: Listing<Evidence> = self.get("evidence", &[("uri", uri)])?
        ;
        Ok(held.data)
    }

    /// The decisions and evaluations that read `uri` (at `hash`, when given):
    /// `GET /v1/research/evidence/dependents`.
    pub fn dependents(&self, uri: &str, hash: Option<&str>) -> Result<Dependents, crate::Error> {
        let mut query = vec![("uri", uri)];
        query.extend(hash.map(|h| ("hash", h)));
        self.get("evidence/dependents", &query)
    }

    /// File a decision record: `POST /v1/research/decisions`. Answers the record as filed.
    pub fn decision(&self, d: &Decision) -> Result<Decision, crate::Error> {
        self.post("decisions", d)
    }

    /// File an evaluation record: `POST /v1/research/evaluations`.
    pub fn evaluation(&self, e: &Evaluation) -> Result<Evaluation, crate::Error> {
        self.post("evaluations", e)
    }

    /// File an action draft: `POST /v1/research/actions`.
    pub fn action(&self, a: &Action) -> Result<Action, crate::Error> {
        self.post("actions", a)
    }

    pub fn filed(&self, b: &Batch, what: &str) -> Result<Value, crate::Error> {
        if scope(&b.project) != self.project {
            return Err(crate::Error::Message(format!(
                "{what} were filed under {}, not {}",
                b.project, self.project
            )));
        }
        Ok(json!({"recorded": b.recorded, "canonical": b.canonical}))
    }
}

pub fn scope(project: &str) -> String {
    project.trim().to_lowercase()
}

pub fn claimed(_token: &str) -> Option<String> {
    // JWT-like: extract project from token claims if present
    // For now, return None (no claim parsing)
    None
}

pub fn token() -> String {
    std::env::var("HANZO_API_KEY").unwrap_or_default()
}

pub fn now() -> String {
    chrono::Utc::now().to_rfc3339()
}

pub fn left() -> String {
    "left".into()
}

fn urlencoding(s: &str) -> String {
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
