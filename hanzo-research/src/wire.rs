//! Wire types for the Hanzo Research Cloud API (`/v1/research`).
//!
//! Provides the canonical structs matching `apps/research/record.go` in Hanzo Cloud:
//! - Benchmark definitions, splits, and official metrics
//! - Executions/Runs with derived completion states and measures
//! - Content-addressed artifacts
//! - Studies, Papers, and Comparisons

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

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Comparison {
    pub comparable: bool,
    #[serde(default, deserialize_with = "list")]
    pub blocks: Vec<String>,
    #[serde(default, deserialize_with = "list")]
    pub differences: Vec<Difference>,
    #[serde(default, deserialize_with = "list")]
    pub deltas: Vec<Delta>,
}

/// The token: `HANZO_API_KEY`, else what `hanzo auth token` prints.
pub fn token() -> Result<String, crate::Error> {
    if let Some(t) = std::env::var("HANZO_API_KEY")
        .ok()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
    {
        return Ok(t);
    }
    let out = std::process::Command::new("hanzo")
        .args(["auth", "token"])
        .output()
        .map_err(|e| crate::Error::Io(e))?;
    if !out.status.success() {
        return Err(crate::Error::Message("hanzo auth token failed".into()));
    }
    let t = String::from_utf8(out.stdout)
        .map_err(|e| crate::Error::Message(e.to_string()))?
        .trim()
        .to_string();
    if t.is_empty() {
        return Err(crate::Error::Message("hanzo auth token printed nothing".into()));
    }
    Ok(t)
}

/// The project a JWT's claims name, `default` when none; `None` for a token that is not a JWT.
pub fn claimed(token: &str) -> Option<String> {
    use base64::Engine;
    let body = token.split('.').nth(1)?;
    let claims: Value = serde_json::from_slice(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(body.trim_end_matches('='))
            .ok()?,
    )
    .ok()?;
    Some(scope(claims["project"].as_str().unwrap_or("")))
}

/// Seconds until a JWT's `exp`; `None` for a token that is not a JWT or names no expiry.
pub fn left(token: &str) -> Option<i64> {
    use base64::Engine;
    let body = token.split('.').nth(1)?;
    let claims: Value = serde_json::from_slice(
        &base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(body.trim_end_matches('='))
            .ok()?,
    )
    .ok()?;
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs() as i64;
    Some(claims["exp"].as_i64()? - now)
}

/// A project as the server scopes it: empty is the default project.
pub fn scope(p: &str) -> String {
    match p.trim() {
        "" => "default".into(),
        p => p.into(),
    }
}

/// Now as RFC 3339 in UTC, to the second.
pub fn now() -> String {
    let s = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs()) as i64;
    let (days, rem) = (s.div_euclid(86400), s.rem_euclid(86400));
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + (m <= 2) as i64;
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// Canonical research API client.
#[derive(Clone)]
pub struct Api {
    pub base: String,
    pub project: String,
    pub token: String,
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

    fn call(&self, req: ureq::Request, body: Option<&[u8]>, what: &str) -> Result<Vec<u8>, crate::Error> {
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
                Ok(resp) => {
                    use std::io::Read;
                    let mut reader = resp.into_reader().take(crate::MAX_RESPONSE_BYTES);
                    let mut body = Vec::new();
                    reader.read_to_end(&mut body).map_err(crate::Error::Io)?;
                    return Ok(body);
                }
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

    pub fn post<B: Serialize, R: DeserializeOwned>(&self, path: &str, body: &B) -> Result<R, crate::Error> {
        let url = format!("{}/v1/research/{path}", self.base);
        let bytes = serde_json::to_vec(body)?;
        let req = self
            .agent
            .post(&url)
            .set("Content-Type", "application/json");
        let resp_body = self.call(req, Some(&bytes), &format!("POST {path}"))?;
        Ok(serde_json::from_slice(&resp_body)?)
    }

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

    pub fn get<R: DeserializeOwned>(&self, path: &str, query: &[(&str, &str)]) -> Result<R, crate::Error> {
        let body = self.bytes(path, query)?;
        Ok(serde_json::from_slice(&body)?)
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
