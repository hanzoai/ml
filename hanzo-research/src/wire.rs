//! Wire types for the Hanzo Research Cloud API (`/v1/research`).
//!
//! Provides the canonical structs matching `apps/research/record.go` in Hanzo Cloud:
//! - Benchmark definitions, splits, and official metrics
//! - Executions/Runs with derived completion states and measures
//! - Content-addressed artifacts
//! - Studies, Papers, and Comparisons

use crate::canonical;
use crate::lineage::{Purpose, Report};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Deserializer, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
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

/// What an artifact write stored: the server's address for the bytes, and whether it was new.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Stored {
    pub sha256: String,
    #[serde(rename = "ref")]
    pub reference: String,
    pub created: bool,
    pub rolled_up: bool,
}

/// An artifact as `/v1/research/artifacts` records it (HIP-1334 §5). A write sends the fields
/// below `content`; the server sets `ref`, `project`, `visibility`, `retention_class`,
/// `compressed_size` and `received` itself.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Artifact {
    /// The SHA-256 of the canonical uncompressed representation, hex.
    pub sha256: String,
    /// The bytes, base64, as sent: compressed when `compression` says so. Write only.
    #[serde(skip_serializing_if = "String::is_empty")]
    pub content: String,
    pub kind: String,
    pub media_type: String,
    /// `rfc8785` for `application/json` and `application/x-ndjson`, else `none`.
    pub canonicalization: String,
    /// `none` or `gzip`.
    pub compression: String,
    /// Bytes of the canonical uncompressed representation.
    pub uncompressed_size: u64,
    pub compressed_size: Option<u64>,
    pub purpose: Option<Purpose>,
    pub parents: Vec<String>,
    pub run_id: String,
    pub git_sha: String,
    pub git_branch: String,
    pub git_dirty: bool,
    pub lib_versions: BTreeMap<String, String>,
    /// Unix seconds, the producer's clock.
    pub ts: i64,
    #[serde(rename = "ref")]
    pub reference: String,
    pub project: String,
    pub visibility: String,
    pub retention_class: String,
    /// RFC 3339 UTC with milliseconds, the server's receipt.
    pub received: String,
}

/// The canonical uncompressed representation of `bytes` under `media_type`: canonical JSON for
/// `application/json`, each line canonical for `application/x-ndjson`, else the bytes.
pub fn represent(media_type: &str, bytes: &[u8]) -> Result<(Vec<u8>, &'static str), crate::Error> {
    match media_type {
        "application/json" => Ok((canonical::bytes(bytes)?, "rfc8785")),
        "application/x-ndjson" => {
            let mut out = Vec::with_capacity(bytes.len());
            let body = bytes.strip_suffix(b"\n").ok_or_else(|| {
                crate::Error::Invalid("application/x-ndjson must end with a newline".into())
            })?;
            for line in body.split(|&b| b == b'\n') {
                out.extend_from_slice(&canonical::bytes(line)?);
                out.push(b'\n');
            }
            Ok((out, "rfc8785"))
        }
        _ => Ok((bytes.to_vec(), "none")),
    }
}

/// A claim (HIP-1334 §11) as the server holds it. Its status is the server's: `pending` until
/// the server evaluates the gate, then `passed` or `failed`; `retracted` by its author.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Claim {
    pub id: String,
    pub statement: String,
    pub subject_run: String,
    pub baseline_run: Option<String>,
    /// The gate artifact's sha256.
    pub gate: String,
    pub status: String,
    pub verdict: Option<Value>,
    pub reasons: Vec<String>,
    pub retraction: Option<String>,
    pub project: String,
    pub by: String,
    pub created: String,
    pub decided: Option<String>,
}

/// What a claim states and about what: the write half of [`Claim`].
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Claiming {
    pub statement: String,
    pub subject_run: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub baseline_run: Option<String>,
    pub gate: String,
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
        .map_err(crate::Error::Io)?;
    if !out.status.success() {
        return Err(crate::Error::Invalid("hanzo auth token failed".into()));
    }
    let t = String::from_utf8(out.stdout)
        .map_err(|e| crate::Error::Invalid(e.to_string()))?
        .trim()
        .to_string();
    if t.is_empty() {
        return Err(crate::Error::Invalid(
            "hanzo auth token printed nothing".into(),
        ));
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
    rfc3339(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map_or(0, |d| d.as_secs()) as i64,
    )
}

/// Unix second `s` as RFC 3339 in UTC.
pub fn rfc3339(s: i64) -> String {
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

/// The most bytes an artifact holds, uncompressed (the server's bound).
pub const MAX_ARTIFACT_BYTES: u64 = 16 << 20;

/// The research API's client: every call carries the token and the project it was built for.
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

impl Api {
    /// A client for `project`, refused when the token's own project claim names another.
    pub fn new(base: &str, project: &str, token: String) -> Result<Api, crate::Error> {
        if let Some(p) = claimed(&token) {
            if p != scope(project) {
                return Err(crate::Error::Invalid(format!(
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

    /// The response body, at most `cap` bytes; a longer one is an error, never truncated.
    fn call(
        &self,
        req: ureq::Request,
        body: Option<&[u8]>,
        what: &str,
        cap: u64,
    ) -> Result<Vec<u8>, crate::Error> {
        use std::io::Read;
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
                    let mut out = Vec::new();
                    resp.into_reader().take(cap + 1).read_to_end(&mut out)?;
                    if out.len() as u64 > cap {
                        return Err(crate::Error::Invalid(format!(
                            "{what}: response over {cap} bytes"
                        )));
                    }
                    return Ok(out);
                }
                Err(ureq::Error::Status(503, _)) if wait <= 8 => {
                    std::thread::sleep(std::time::Duration::from_secs(wait));
                    wait *= 2;
                }
                Err(ureq::Error::Status(code, resp)) => {
                    let mut out = Vec::new();
                    let _ = resp.into_reader().take(4096).read_to_end(&mut out);
                    return Err(crate::Error::Status(
                        code,
                        format!("{what}: {}", String::from_utf8_lossy(&out).trim()),
                    ));
                }
                Err(e) => return Err(crate::Error::Http(Box::new(e))),
            }
        }
    }

    fn url(&self, path: &str, query: &[(&str, &str)]) -> String {
        let mut url = format!("{}/v1/research/{path}", self.base);
        if !query.is_empty() {
            let q: Vec<String> = query
                .iter()
                .map(|(k, v)| format!("{}={}", urlencoding(k), urlencoding(v)))
                .collect();
            url = format!("{url}?{}", q.join("&"));
        }
        url
    }

    pub fn post<B: Serialize, R: DeserializeOwned>(
        &self,
        path: &str,
        body: &B,
    ) -> Result<R, crate::Error> {
        let bytes = serde_json::to_vec(body)?;
        let req = self
            .agent
            .post(&self.url(path, &[]))
            .set("Content-Type", "application/json");
        let out = self.call(
            req,
            Some(&bytes),
            &format!("POST {path}"),
            crate::MAX_RESPONSE_BYTES,
        )?;
        Ok(serde_json::from_slice(&out)?)
    }

    pub fn bytes(&self, path: &str, query: &[(&str, &str)]) -> Result<Vec<u8>, crate::Error> {
        let req = self.agent.get(&self.url(path, query));
        self.call(req, None, &format!("GET {path}"), crate::MAX_RESPONSE_BYTES)
    }

    pub fn get<R: DeserializeOwned>(
        &self,
        path: &str,
        query: &[(&str, &str)],
    ) -> Result<R, crate::Error> {
        Ok(serde_json::from_slice(&self.bytes(path, query)?)?)
    }

    /// `b`'s counts, once it names this client's project.
    pub fn filed(&self, b: &Batch, what: &str) -> Result<Value, crate::Error> {
        if scope(&b.project) != self.project {
            return Err(crate::Error::Invalid(format!(
                "{what} were filed under {}, not {}",
                b.project, self.project
            )));
        }
        Ok(json!({"recorded": b.recorded, "canonical": b.canonical}))
    }

    /// Register `bytes` as the artifact `a` describes: its `media_type`, `kind`, `purpose`,
    /// `parents`, `run_id` and provenance. The address, sizes and canonicalization are computed
    /// here from the canonical representation, which is what is sent; the server recomputes them
    /// and refuses a disagreement.
    pub fn put(&self, mut a: Artifact, bytes: &[u8]) -> Result<Stored, crate::Error> {
        use base64::Engine;
        let (canon, how) = represent(&a.media_type, bytes)?;
        a.sha256 = canonical::sha256(&canon);
        a.canonicalization = how.into();
        a.compression = "none".into();
        a.uncompressed_size = canon.len() as u64;
        a.content = base64::engine::general_purpose::STANDARD.encode(&canon);
        let stored: Stored = self.post("artifacts", &a)?;
        if stored.sha256 != a.sha256 {
            return Err(crate::Error::Invalid(format!(
                "the server stored {} for bytes that hash to {}",
                stored.sha256, a.sha256
            )));
        }
        Ok(stored)
    }

    /// The bytes the artifact `sha256` names, checked against it.
    pub fn artifact(&self, sha256: &str) -> Result<Vec<u8>, crate::Error> {
        if sha256.len() != 64 || !sha256.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(crate::Error::Invalid(format!("{sha256:?} is not a sha256")));
        }
        let req = self
            .agent
            .get(&self.url(&format!("artifacts/{sha256}"), &[]));
        let out = self.call(
            req,
            None,
            &format!("GET artifacts/{sha256}"),
            MAX_ARTIFACT_BYTES,
        )?;
        let got = canonical::sha256(&out);
        if !got.eq_ignore_ascii_case(sha256) {
            return Err(crate::Error::Invalid(format!(
                "artifact {sha256}: the server answered bytes that hash to {got}"
            )));
        }
        Ok(out)
    }

    /// Artifacts' metadata (`kind`, `run`, `purpose`, `since`).
    pub fn artifacts(&self, query: &[(&str, &str)]) -> Result<Vec<Artifact>, crate::Error> {
        Ok(self.get::<Listing<Artifact>>("artifacts", query)?.data)
    }

    /// The server's lineage walk of `sha256`.
    pub fn lineage(&self, sha256: &str) -> Result<Report, crate::Error> {
        self.get(&format!("artifacts/{sha256}/lineage"), &[])
    }

    /// Make a claim; the server checks it and answers it as stored.
    pub fn claim(&self, c: &Claiming) -> Result<Claim, crate::Error> {
        self.post("claims", c)
    }

    /// Claims (`status`, `run`, `gate`).
    pub fn claims(&self, query: &[(&str, &str)]) -> Result<Vec<Claim>, crate::Error> {
        Ok(self.get::<Listing<Claim>>("claims", query)?.data)
    }

    /// Retract claim `id`, saying why.
    pub fn retract(&self, id: &str, reason: &str) -> Result<Claim, crate::Error> {
        self.post(
            &format!("claims/{}/retract", urlencoding(id)),
            &json!({"reason": reason}),
        )
    }
}

fn urlencoding(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn representations() {
        let (c, how) = represent("application/json", b"{ \"b\": 1.50, \"a\": [1e21] }").unwrap();
        assert_eq!(
            (c.as_slice(), how),
            (&br#"{"a":[1e+21],"b":1.5}"#[..], "rfc8785")
        );
        let (c, _) = represent("application/x-ndjson", b"{\"b\":1, \"a\":2}\n[ 1 ]\n").unwrap();
        assert_eq!(c, b"{\"a\":2,\"b\":1}\n[1]\n");
        assert!(represent("application/x-ndjson", b"{}").is_err());
        assert!(represent("application/json", br#"{"a":1,"a":2}"#).is_err());
        let (c, how) = represent("image/png", b"\x89PNG").unwrap();
        assert_eq!((c.as_slice(), how), (&b"\x89PNG"[..], "none"));
    }

    #[test]
    fn times() {
        assert_eq!(rfc3339(0), "1970-01-01T00:00:00Z");
        assert_eq!(rfc3339(1_790_000_000), "2026-09-21T14:13:20Z");
    }
}
