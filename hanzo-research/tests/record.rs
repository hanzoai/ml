//! The record routes of `/v1/research` against a stand-in that answers as hanzo-inc/cloud
//! `apps/research` does (record.go, store.go, zip's typed ops): a batch is read from its kind's
//! key alone (Go's encoding/json v1: unknown keys ignored, names matched case-insensitively);
//! empty is 400 `nothing to record`, past 20,000 is 413, a run without an id and a benchmark
//! 422; versions append by content into (project, id), the project the caller's
//! `X-Project-Id`; a read answers the latest version that is not retracted, with completion
//! and key derived; `compare` states blocks, then config, then deltas; Go's nil slices are
//! null; a refusal is an RFC 9457 problem document.

use std::collections::{BTreeMap, HashSet};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use hanzo_research::{
    Api, Batch, Benchmark, Comparison, Listing, Measure, Research, Run, Split, MAX_RESPONSE_BYTES,
};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

// ── the stand-in ─────────────────────────────────────────────────────────────────────

/// One stored version: what the store keeps of a record, and the append clock it took.
struct Version {
    seq: u64,
    kind: &'static str,
    project: String,
    id: String,
    record: Map<String, Value>,
}

#[derive(Default)]
struct State {
    seq: u64,
    versions: Vec<Version>,
    // (kind, project, id, content hash): store.go's CreateIfAbsent key
    held: HashSet<(&'static str, String, String, String)>,
    // every request answered: method, path with its query, body as sent
    heard: Vec<(String, String, Value)>,
}

struct Request {
    method: String,
    path: String,
    query: BTreeMap<String, String>,
    project: String,
    bearer: bool,
    body: Vec<u8>,
}

type Answer = Result<Value, (u16, String)>;

struct Standin {
    base: String,
    state: Arc<Mutex<State>>,
}

impl Standin {
    fn start() -> Standin {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let base = format!("http://{}", listener.local_addr().unwrap());
        let state = Arc::new(Mutex::new(State::default()));
        let shared = state.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let state = shared.clone();
                thread::spawn(move || serve(stream.unwrap(), &state));
            }
        });
        Standin { base, state }
    }

    /// The bodies the client sent to `method path`, in order.
    fn sent(&self, method: &str, path: &str) -> Vec<Value> {
        let s = self.state.lock().unwrap();
        s.heard
            .iter()
            .filter(|(m, p, _)| m == method && p.split('?').next() == Some(path))
            .map(|(_, _, b)| b.clone())
            .collect()
    }
}

fn serve(mut s: TcpStream, state: &Mutex<State>) {
    let Some((req, target)) = parse(&mut s) else {
        return;
    };
    let (status, body) = match route(state, &req) {
        Ok(v) => (200, v),
        Err((code, detail)) => (code, problem(code, &detail)),
    };
    let sent = serde_json::from_slice(&req.body).unwrap_or(Value::Null);
    state
        .lock()
        .unwrap()
        .heard
        .push((req.method.clone(), target, sent));
    let bytes = serde_json::to_vec(&body).unwrap();
    let kind = if status == 200 {
        "application/json"
    } else {
        "application/problem+json"
    };
    let head = format!(
        "HTTP/1.1 {status} {}\r\nContent-Type: {kind}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        title(status),
        bytes.len()
    );
    let _ = s.write_all(head.as_bytes());
    let _ = s.write_all(&bytes);
}

fn parse(s: &mut TcpStream) -> Option<(Request, String)> {
    let mut reader = BufReader::new(s.try_clone().ok()?);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let target = parts.next()?.to_string();
    let (mut len, mut project, mut bearer) = (0usize, String::new(), false);
    loop {
        let mut h = String::new();
        reader.read_line(&mut h).ok()?;
        if h == "\r\n" || h.is_empty() {
            break;
        }
        let (name, value) = h.split_once(':')?;
        let value = value.trim();
        match name.to_ascii_lowercase().as_str() {
            "content-length" => len = value.parse().ok()?,
            "x-project-id" => project = value.to_string(),
            "authorization" => bearer = value.starts_with("Bearer ") && value.len() > 7,
            _ => {}
        }
    }
    let mut body = vec![0u8; len];
    reader.read_exact(&mut body).ok()?;
    let (path, q) = target.split_once('?').unwrap_or((&target, ""));
    let query = q
        .split('&')
        .filter(|p| !p.is_empty())
        .map(|p| {
            let (k, v) = p.split_once('=').unwrap_or((p, ""));
            (unescape(k), unescape(v))
        })
        .collect();
    // principal.Project: the X-Project-Id header, trimmed, else the default project
    let project = match project.trim() {
        "" => "default".to_string(),
        p => p.to_string(),
    };
    let req = Request {
        method,
        path: path.to_string(),
        query,
        project,
        bearer,
        body,
    };
    Some((req, target))
}

fn unescape(s: &str) -> String {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            out.push(u8::from_str_radix(&s[i + 1..i + 3], 16).unwrap());
            i += 3;
        } else {
            out.push(if b[i] == b'+' { b' ' } else { b[i] });
            i += 1;
        }
    }
    String::from_utf8(out).unwrap()
}

/// zip's refusal: an RFC 9457 problem document.
fn problem(status: u16, detail: &str) -> Value {
    json!({"type": "about:blank", "title": title(status), "status": status, "detail": detail})
}

fn title(status: u16) -> &'static str {
    match status {
        200 => "OK",
        400 => "Bad Request",
        401 => "Unauthorized",
        404 => "Not Found",
        413 => "Request Entity Too Large",
        422 => "Unprocessable Entity",
        _ => "Internal Server Error",
    }
}

fn route(state: &Mutex<State>, req: &Request) -> Answer {
    if !req.bearer {
        return Err((401, "unauthorized".into()));
    }
    let mut s = state.lock().unwrap();
    match (req.method.as_str(), req.path.as_str()) {
        ("POST", "/v1/research/runs") => put(&mut s, "runs", req),
        ("POST", "/v1/research/benchmarks") => put(&mut s, "benchmarks", req),
        ("GET", "/v1/research/runs") => runs(&s, req),
        ("GET", "/v1/research/benchmarks") => {
            let bs = benchmarks(&s, &req.query);
            Ok(json!({"total": bs.len(), "data": nil(bs)}))
        }
        ("GET", "/v1/research/compare") => compare(&s, req),
        _ => Err((404, "Not Found".into())),
    }
}

/// postRuns / postBenchmarks: the batch under its kind's key, bounded, each row checked,
/// appended by content.
fn put(s: &mut State, kind: &'static str, req: &Request) -> Answer {
    let body: Value =
        serde_json::from_slice(&req.body).map_err(|e| (400, format!("invalid body: {e}")))?;
    let obj = body
        .as_object()
        .ok_or_else(|| (400, "invalid body: not an object".to_string()))?;
    let rows = match field(obj, kind) {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::Array(rows)) => rows.clone(),
        Some(_) => return Err((400, format!("invalid body: {kind} is not an array"))),
    };
    if rows.is_empty() {
        return Err((400, "nothing to record".into()));
    }
    if rows.len() > 20000 {
        return Err((413, "batch too large; split the upload".into()));
    }
    let records = rows
        .iter()
        .map(|r| stored(kind, r))
        .collect::<Result<Vec<_>, _>>()?;
    for r in &records {
        let id = text(r, "id");
        let missing = id.is_empty() || (kind == "runs" && text(r, "benchmark").is_empty());
        if missing {
            return Err((
                422,
                if kind == "runs" {
                    "run needs an id and a benchmark".into()
                } else {
                    "benchmark needs an id".into()
                },
            ));
        }
    }
    let mut recorded = 0;
    for record in records {
        let id = text(&record, "id");
        let hash = hex(&Sha256::digest(serde_json::to_vec(&record).unwrap()));
        if s.held.insert((kind, req.project.clone(), id.clone(), hash)) {
            s.seq += 1;
            let seq = s.seq;
            s.versions.push(Version {
                seq,
                kind,
                project: req.project.clone(),
                id,
                record,
            });
            recorded += 1;
        }
    }
    let canonical = latest(s, kind).len();
    Ok(json!({"project": req.project, "recorded": recorded, "canonical": canonical}))
}

/// A row as the store keeps it: Go's field set (encoding/json v1 into the struct), less what a
/// read derives, the revision normalized (`revisionOf`), consent forced private.
fn stored(kind: &str, row: &Value) -> Result<Map<String, Value>, (u16, String)> {
    let obj = row
        .as_object()
        .ok_or_else(|| (400, "invalid body: a record is not an object".to_string()))?;
    let mut out = Map::new();
    let fields: &[(&str, &str)] = if kind == "runs" { RUN } else { BENCHMARK };
    for &(name, ty) in fields {
        let v = field(obj, name).cloned().unwrap_or(Value::Null);
        let v = match (ty, v) {
            ("s", Value::Null) => json!(""),
            ("b", Value::Null) => json!(false),
            ("s", v @ Value::String(_)) | ("b", v @ Value::Bool(_)) => v,
            ("i", v) if v.is_null() || v.is_i64() => v,
            ("f", v) if v.is_null() || v.is_number() => v,
            ("measures", v) => measures(v)?,
            ("splits", v) | ("strings", v) if v.is_null() || v.is_array() => v,
            (_, v) => return Err((400, format!("invalid body: {name} cannot read {v}"))),
        };
        out.insert(name.to_string(), v);
    }
    let rev = match text(&out, "revision").trim().to_lowercase().as_str() {
        "corrected" => "corrected",
        "retracted" => "retracted",
        _ => "original",
    };
    out.insert("revision".into(), json!(rev));
    Ok(out)
}

fn measures(v: Value) -> Result<Value, (u16, String)> {
    let Value::Array(rows) = v else {
        return match v {
            Value::Null => Ok(Value::Null),
            _ => Err((400, "invalid body: measures is not an array".into())),
        };
    };
    let mut out = Vec::new();
    for m in rows {
        let obj = m
            .as_object()
            .ok_or_else(|| (400, "invalid body: a measure is not an object".to_string()))?;
        let get = |k: &str| field(obj, k).cloned().unwrap_or(Value::Null);
        out.push(json!({
            "category": get("category").as_str().unwrap_or(""),
            "metric": get("metric").as_str().unwrap_or(""),
            "value": get("value").as_f64().unwrap_or(0.0),
            "lo": get("lo"), "hi": get("hi"), "n": get("n"), "of": get("of"),
        }));
    }
    Ok(Value::Array(out))
}

// The wire fields of record.go's Run and ResearchBenchmark a write keeps, with their Go types.
const RUN: &[(&str, &str)] = &[
    ("id", "s"),
    ("benchmark", "s"),
    ("split", "s"),
    ("system", "s"),
    ("version", "s"),
    ("baseline", "b"),
    ("study", "s"),
    ("embedder", "s"),
    ("reader", "s"),
    ("k", "i"),
    ("temperature", "f"),
    ("max_tokens", "i"),
    ("prompt", "s"),
    ("prompt_digest", "s"),
    ("dataset_digest", "s"),
    ("store_digest", "s"),
    ("facts_digest", "s"),
    ("commit", "s"),
    ("questions", "i"),
    ("answered", "i"),
    ("when", "s"),
    ("ended", "s"),
    ("by", "s"),
    ("measures", "measures"),
    ("notes", "s"),
    ("revision", "s"),
];
const BENCHMARK: &[(&str, &str)] = &[
    ("id", "s"),
    ("title", "s"),
    ("version", "s"),
    ("dataset", "s"),
    ("license", "s"),
    ("origin", "s"),
    ("splits", "splits"),
    ("metric", "s"),
    ("metrics", "strings"),
    ("citation", "s"),
    ("notes", "s"),
    ("revision", "s"),
    ("when", "s"),
];

/// encoding/json v1's field match: the exact name, else one that differs only in case.
fn field<'a>(obj: &'a Map<String, Value>, name: &str) -> Option<&'a Value> {
    obj.get(name).or_else(|| {
        obj.iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v)
    })
}

fn text(obj: &Map<String, Value>, name: &str) -> String {
    obj.get(name)
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// canonical(): the latest-appended version of each (project, id), withdrawn when retracted.
fn latest<'a>(s: &'a State, kind: &str) -> Vec<&'a Version> {
    let mut by: BTreeMap<(&str, &str), &Version> = BTreeMap::new();
    for v in s.versions.iter().filter(|v| v.kind == kind) {
        let key = (v.project.as_str(), v.id.as_str());
        if by.get(&key).is_none_or(|cur| v.seq > cur.seq) {
            by.insert(key, v);
        }
    }
    by.into_values()
        .filter(|v| text(&v.record, "revision") != "retracted")
        .collect()
}

/// A Go slice on the wire: nil (no element) is null.
fn nil(rows: Vec<Value>) -> Value {
    if rows.is_empty() {
        Value::Null
    } else {
        Value::Array(rows)
    }
}

/// store.toRun: a stored version as a read answers it, completion and key derived.
fn read_run(v: &Version) -> Value {
    let mut r = v.record.clone();
    let int = |k: &str| r.get(k).and_then(Value::as_i64);
    let completion = match (int("questions"), int("answered")) {
        (Some(q), Some(a)) if a > q => "inconsistent",
        (Some(q), Some(a)) if a < q => "partial",
        (Some(_), Some(_)) if text(&r, "ended").trim().is_empty() => "unknown",
        (Some(_), Some(_)) => "complete",
        _ => "unknown",
    };
    let opt = |k: &str| int(k).map(|x| x.to_string()).unwrap_or_default();
    let temperature = r
        .get("temperature")
        .and_then(Value::as_f64)
        .map(|t| t.to_string()) // strconv.FormatFloat(t, 'f', -1, 64)
        .unwrap_or_default();
    let mut key = Sha256::new();
    for part in [
        text(&r, "benchmark"),
        text(&r, "split"),
        text(&r, "system"),
        text(&r, "version"),
        opt("k"),
        temperature,
        opt("max_tokens"),
        text(&r, "embedder"),
        text(&r, "reader"),
        text(&r, "prompt_digest"),
        text(&r, "dataset_digest"),
        text(&r, "store_digest"),
        text(&r, "facts_digest"),
    ] {
        key.update(part.as_bytes());
        key.update([0]);
    }
    r.insert("project".into(), json!(v.project));
    r.insert("completion".into(), json!(completion));
    r.insert("key".into(), json!(hex(&key.finalize())));
    r.insert("canonical".into(), json!(true));
    r.insert("visibility".into(), json!("private"));
    r.insert("trainable".into(), json!(false));
    r.insert("publishable".into(), json!(false));
    Value::Object(r)
}

fn benchmarks(s: &State, q: &BTreeMap<String, String>) -> Vec<Value> {
    latest(s, "benchmarks")
        .into_iter()
        .filter(|v| keep(q, "project", &v.project) && keep(q, "id", &v.id))
        .map(|v| {
            let mut r = v.record.clone();
            r.insert("project".into(), json!(v.project));
            r.insert("canonical".into(), json!(true));
            Value::Object(r)
        })
        .collect()
}

fn keep(q: &BTreeMap<String, String>, name: &str, have: &str) -> bool {
    q.get(name)
        .is_none_or(|want| want.is_empty() || want == have)
}

/// store.runs with runFilter.keeps, ordered by (project, id).
fn filtered(s: &State, q: &BTreeMap<String, String>) -> Vec<Value> {
    latest(s, "runs")
        .into_iter()
        .map(read_run)
        .filter(|r| {
            [
                "project",
                "id",
                "key",
                "benchmark",
                "system",
                "study",
                "completion",
            ]
            .iter()
            .all(|f| keep(q, f, r[*f].as_str().unwrap_or("")))
        })
        .collect()
}

/// listRuns: the canonical runs, the census by completion, the gaps worst first.
fn runs(s: &State, req: &Request) -> Answer {
    if let Some(c) = req.query.get("completion") {
        if !c.is_empty()
            && !["complete", "partial", "inconsistent", "unknown"].contains(&c.as_str())
        {
            return Err((
                422,
                "completion must be complete, partial, inconsistent, or unknown".into(),
            ));
        }
    }
    let rs = filtered(s, &req.query);
    let count = |c: &str| rs.iter().filter(|r| r["completion"] == c).count();
    let census = json!({
        "complete": count("complete"), "partial": count("partial"),
        "inconsistent": count("inconsistent"), "unknown": count("unknown"),
    });
    let mut gaps: BTreeMap<&str, usize> = BTreeMap::new();
    for r in &rs {
        for f in [
            "system",
            "version",
            "split",
            "embedder",
            "reader",
            "prompt",
            "prompt_digest",
            "dataset_digest",
            "store_digest",
            "facts_digest",
            "commit",
            "when",
            "ended",
            "by",
        ] {
            if r[f] == "" {
                *gaps.entry(f).or_default() += 1;
            }
        }
        for f in ["k", "temperature", "max_tokens", "questions", "answered"] {
            if r[f].is_null() {
                *gaps.entry(f).or_default() += 1;
            }
        }
        if r["measures"].as_array().is_none_or(Vec::is_empty) {
            *gaps.entry("measures").or_default() += 1;
        }
    }
    let mut gaps: Vec<(&str, usize)> = gaps.into_iter().collect();
    gaps.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(b.0)));
    let gaps: Vec<Value> = gaps
        .into_iter()
        .map(|(f, n)| json!({"field": f, "runs": n}))
        .collect();
    Ok(json!({"data": nil(rs.clone()), "total": rs.len(), "completion": census, "gaps": gaps}))
}

/// compareRuns: both runs from one project, then record.go's compare.
fn compare(s: &State, req: &Request) -> Answer {
    let (a, b) = match (req.query.get("a"), req.query.get("b")) {
        (Some(a), Some(b)) if !a.is_empty() && !b.is_empty() => (a, b),
        _ => return Err((400, "compare needs two run ids, a and b".into())),
    };
    let project = match req.query.get("project") {
        Some(p) if !p.is_empty() => p.clone(),
        _ => req.project.clone(),
    };
    let one = |id: &String| {
        let mut q = BTreeMap::new();
        q.insert("project".to_string(), project.clone());
        q.insert("id".to_string(), id.clone());
        filtered(s, &q)
            .into_iter()
            .next()
            .ok_or_else(|| (404, format!("no canonical run {}", quote(id))))
    };
    let (a, b) = (one(a)?, one(b)?);
    let s = |r: &Value, k: &str| r[k].as_str().unwrap_or("").to_string();
    let mut blocks = Vec::new();
    if s(&a, "benchmark") != s(&b, "benchmark") {
        blocks.push(format!(
            "different benchmarks: {} and {}",
            quote(&s(&a, "benchmark")),
            quote(&s(&b, "benchmark"))
        ));
    }
    if s(&a, "split") != s(&b, "split") {
        blocks.push(format!(
            "different splits: {} and {}",
            quote(&s(&a, "split")),
            quote(&s(&b, "split"))
        ));
    }
    if s(&a, "dataset_digest") != s(&b, "dataset_digest") {
        blocks.push(format!(
            "different dataset digests: {} and {} — the two runs scored different items",
            quote(&s(&a, "dataset_digest")),
            quote(&s(&b, "dataset_digest"))
        ));
    }
    for r in [&a, &b] {
        if r["completion"] != "complete" {
            blocks.push(format!(
                "run {} is not established as complete (completion {})",
                quote(&s(r, "id")),
                s(r, "completion")
            ));
        }
        if s(r, "system").is_empty() {
            blocks.push(format!(
                "run {} does not name the system under test",
                quote(&s(r, "id"))
            ));
        }
    }
    // Run.setup: each configuration field as text, absent "" and present a number.
    let setup = |r: &Value| -> Vec<(&'static str, String)> {
        let int = |k: &str| r[k].as_i64().map(|x| x.to_string()).unwrap_or_default();
        let float = |k: &str| r[k].as_f64().map(go_g).unwrap_or_default();
        vec![
            ("benchmark", s(r, "benchmark")),
            ("split", s(r, "split")),
            ("system", s(r, "system")),
            ("version", s(r, "version")),
            ("embedder", s(r, "embedder")),
            ("reader", s(r, "reader")),
            ("k", int("k")),
            ("temperature", float("temperature")),
            ("max_tokens", int("max_tokens")),
            ("prompt", s(r, "prompt")),
            ("prompt_digest", s(r, "prompt_digest")),
            ("dataset_digest", s(r, "dataset_digest")),
            ("store_digest", s(r, "store_digest")),
            ("facts_digest", s(r, "facts_digest")),
            ("commit", s(r, "commit")),
            (
                "baseline",
                r["baseline"].as_bool().unwrap_or(false).to_string(),
            ),
        ]
    };
    let config: Vec<Value> = setup(&a)
        .into_iter()
        .zip(setup(&b))
        .filter(|(x, y)| x.1 != y.1)
        .map(|(x, y)| json!({"field": x.0, "a": x.1, "b": y.1}))
        .collect();
    let empty = Vec::new();
    let ms = |r: &Value| r["measures"].as_array().unwrap_or(&empty).clone();
    let index: BTreeMap<(String, String), Value> = ms(&b)
        .into_iter()
        .map(|m| ((s(&m, "category"), s(&m, "metric")), m))
        .collect();
    let mut deltas: Vec<Value> = Vec::new();
    for m in ms(&a) {
        let Some(n) = index.get(&(s(&m, "category"), s(&m, "metric"))) else {
            continue;
        };
        let (x, y) = (m["value"].as_f64().unwrap(), n["value"].as_f64().unwrap());
        let overlap = match (
            m["lo"].as_f64(),
            m["hi"].as_f64(),
            n["lo"].as_f64(),
            n["hi"].as_f64(),
        ) {
            (Some(ml), Some(mh), Some(nl), Some(nh)) => json!(ml.max(nl) <= mh.min(nh)),
            _ => Value::Null,
        };
        deltas.push(json!({"category": m["category"], "metric": m["metric"],
            "a": x, "b": y, "change": y - x, "overlap": overlap}));
    }
    deltas.sort_by(|p, q| {
        (s(p, "category"), s(p, "metric")).cmp(&(s(q, "category"), s(q, "metric")))
    });
    Ok(json!({
        "a": a, "b": b, "comparable": blocks.is_empty(),
        "blocks": nil(blocks.into_iter().map(Value::String).collect()),
        "config": nil(config), "deltas": nil(deltas),
    }))
}

/// record.go's quote: an empty value reads as (unrecorded), anything else as strconv.Quote.
fn quote(s: &str) -> String {
    if s.is_empty() {
        return "(unrecorded)".into();
    }
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 || c as u32 == 0x7f => {
                out.push_str(&format!("\\x{:02x}", c as u32))
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// strconv.FormatFloat(x, 'g', -1, 64): the shortest digits, in exponent form (two digits at
/// least) when the decimal exponent is under -4 or at least 6.
fn go_g(x: f64) -> String {
    let e = format!("{x:e}");
    let (mantissa, exp) = e.split_once('e').unwrap();
    let exp: i32 = exp.parse().unwrap();
    if !(-4..6).contains(&exp) {
        let sign = if exp < 0 { '-' } else { '+' };
        format!("{mantissa}e{sign}{:02}", exp.abs())
    } else {
        x.to_string()
    }
}

// ── the client against it ────────────────────────────────────────────────────────────

fn run(id: &str, system: &str, value: f64) -> Run {
    Run {
        id: id.into(),
        benchmark: "locomo".into(),
        split: "all".into(),
        system: system.into(),
        version: "1".into(),
        k: Some(20),
        temperature: Some(0.0),
        dataset_digest: "sha256:d".into(),
        commit: "c0ffee".into(),
        questions: Some(282),
        answered: Some(282),
        when: "2026-09-10T21:00:00Z".into(),
        ended: "2026-09-10T21:24:01Z".into(),
        measures: vec![Measure {
            category: "all".into(),
            metric: "f1".into(),
            value,
            lo: Some(value - 0.02),
            hi: Some(value + 0.02),
            n: Some(282),
            of: Some(282),
        }],
        ..Run::default()
    }
}

fn locomo() -> Benchmark {
    Benchmark {
        id: "locomo".into(),
        title: "LoCoMo".into(),
        dataset: "snap-research/locomo".into(),
        license: "CC BY-NC 4.0".into(),
        metric: "f1".into(),
        splits: vec![Split {
            name: "all".into(),
            items: 1986,
            definition: "conversations 0-9".into(),
        }],
        ..Benchmark::default()
    }
}

#[test]
fn post_run_and_post_benchmark_send_the_batch_body() {
    let srv = Standin::start();
    let r = Research::new(&srv.base, "hk-test", "enso-bench", ".", &[]);

    let b = r.post_benchmark(&locomo()).unwrap();
    assert_eq!(
        b,
        Batch {
            project: "enso-bench".into(),
            recorded: 1,
            canonical: 1
        }
    );
    let first = r
        .post_run(&run("locomo-all-k20-enso", "enso", 0.61))
        .unwrap();
    assert_eq!(
        first,
        Batch {
            project: "enso-bench".into(),
            recorded: 1,
            canonical: 1
        }
    );
    // A byte-identical re-send appends nothing: the store's content identity.
    let again = r
        .post_run(&run("locomo-all-k20-enso", "enso", 0.61))
        .unwrap();
    assert_eq!((again.recorded, again.canonical), (0, 1));

    // On the wire: one record under its kind's key, the shape record.go decodes.
    let sent = srv.sent("POST", "/v1/research/runs");
    assert_eq!(
        sent[0].as_object().unwrap().keys().collect::<Vec<_>>(),
        ["runs"]
    );
    assert_eq!(sent[0]["runs"][0]["id"], "locomo-all-k20-enso");
    let sent = srv.sent("POST", "/v1/research/benchmarks");
    assert_eq!(sent[0]["benchmarks"][0]["id"], "locomo");

    // Read back as the server answers: completion derived, consent forced private.
    let got = r.get_run("locomo-all-k20-enso").unwrap();
    assert_eq!(got.completion, "complete");
    assert_eq!(
        (got.project.as_str(), got.canonical, got.visibility.as_str()),
        ("enso-bench", true, "private")
    );
    assert_eq!(got.measures, run("x", "enso", 0.61).measures);
}

#[test]
fn a_bare_record_is_what_the_server_refuses() {
    // The body 0.2.1's post_run sent: the record itself, which runBatch reads as no runs.
    let srv = Standin::start();
    let api = Api::new(&srv.base, "default", "hk-test".into()).unwrap();
    let err = api
        .post::<_, Batch>("runs", &run("bare", "enso", 0.5))
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("400") && err.contains("nothing to record"),
        "{err}"
    );
    let err = api
        .post::<_, Batch>("benchmarks", &locomo())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("400") && err.contains("nothing to record"),
        "{err}"
    );
}

#[test]
fn post_batch_files_each_kind_at_its_own_route() {
    let srv = Standin::start();
    // The body 0.2.1's post_batch sent: the benchmarks beside the runs, where postRuns reads
    // only `runs` — the run is recorded and the benchmark is not.
    let api = Api::new(&srv.base, "default", "hk-test".into()).unwrap();
    let old: Batch = api
        .post(
            "runs",
            &json!({"runs": [run("old", "enso", 0.5)], "benchmarks": [locomo()]}),
        )
        .unwrap();
    assert_eq!(old.recorded, 1);
    let none: Listing<Benchmark> = api.get("benchmarks", &[]).unwrap();
    assert!(none.data.is_empty(), "the server ignored the benchmarks");

    let r = Research::new(&srv.base, "hk-test", "default", ".", &[]);
    let runs = [run("a", "enso", 0.61), run("b", "gpt", 0.55)];
    let (got_runs, got_benchmarks) = r.post_batch(&runs, &[locomo()]).unwrap();
    assert_eq!(
        got_runs,
        Some(Batch {
            project: "default".into(),
            recorded: 2,
            canonical: 3
        })
    );
    assert_eq!(
        got_benchmarks,
        Some(Batch {
            project: "default".into(),
            recorded: 1,
            canonical: 1
        })
    );
    let listed: Listing<Benchmark> = api.get("benchmarks", &[("id", "locomo")]).unwrap();
    assert_eq!(listed.data[0].title, "LoCoMo");
    assert_eq!(listed.data[0].splits[0].items, 1986);

    // Benchmarks before the runs that name them; a kind given nothing is not sent.
    let order: Vec<String> = srv
        .state
        .lock()
        .unwrap()
        .heard
        .iter()
        .filter(|(m, _, _)| m == "POST")
        .map(|(_, p, _)| p.clone())
        .collect();
    assert_eq!(order[1..], ["/v1/research/benchmarks", "/v1/research/runs"]);
    assert_eq!(r.post_batch(&[], &[]).unwrap(), (None, None));
    let (only, nothing) = r.post_batch(&[run("c", "enso", 0.7)], &[]).unwrap();
    assert_eq!((only.unwrap().recorded, nothing), (1, None));
    assert_eq!(srv.sent("POST", "/v1/research/benchmarks").len(), 1);
}

#[test]
fn comparison_reads_the_config_the_server_sends() {
    let srv = Standin::start();
    let r = Research::new(&srv.base, "hk-test", "default", ".", &[]);
    let mut b = run("locomo-all-k20-gpt", "gpt-oss-120b", 0.55);
    b.temperature = Some(0.7);
    b.baseline = true;
    r.post_batch(
        &[run("locomo-all-k20-enso", "enso-flash", 0.61), b],
        &[locomo()],
    )
    .unwrap();

    let c: Comparison = r
        .compare_runs("locomo-all-k20-enso", "locomo-all-k20-gpt")
        .unwrap();
    assert!(c.comparable && c.blocks.is_empty());
    let config: Vec<(&str, &str, &str)> = c
        .config
        .iter()
        .map(|d| (d.field.as_str(), d.a.as_str(), d.b.as_str()))
        .collect();
    assert_eq!(
        config,
        [
            ("system", "enso-flash", "gpt-oss-120b"),
            ("temperature", "0", "0.7"),
            ("baseline", "false", "true"),
        ]
    );
    assert_eq!(c.a.as_ref().unwrap().id, "locomo-all-k20-enso");
    assert_eq!(c.b.as_ref().unwrap().system, "gpt-oss-120b");
    let d = &c.deltas[0];
    assert_eq!(
        (d.category.as_str(), d.metric.as_str(), d.a, d.b),
        ("all", "f1", 0.61, 0.55)
    );
    assert_eq!(d.overlap, Some(false));

    // A pair that may not be compared: blocks first, and still its config.
    let mut partial = run("locomo-all-k20-half", "enso-flash", 0.4);
    partial.answered = Some(141);
    r.post_run(&partial).unwrap();
    let c = r
        .compare_runs("locomo-all-k20-enso", "locomo-all-k20-half")
        .unwrap();
    assert!(!c.comparable);
    assert_eq!(
        c.blocks,
        ["run \"locomo-all-k20-half\" is not established as complete (completion partial)"]
    );
    assert!(
        c.config.is_empty(),
        "the two runs differ in no configuration field"
    );
}

#[test]
fn a_listing_past_eight_mib_reads() {
    // 2,100 runs of 4.3 KB, the hanzo org's public listing's 4.2 KB a run: past the 8 MiB
    // 0.2.1 refused, within the 64 MiB cap.
    let srv = Standin::start();
    let r = Research::new(&srv.base, "hk-test", "default", ".", &[]);
    let runs: Vec<Run> = (0..2100)
        .map(|i| {
            let mut x = run(&format!("bench-{i:04}"), "kai", 0.5);
            x.measures = (0..34)
                .map(|j| Measure {
                    category: format!("suite.{j:02}"),
                    metric: "accuracy".into(),
                    value: 0.8123456789,
                    lo: Some(0.7891234567),
                    hi: Some(0.8345678912),
                    n: Some(400),
                    of: Some(400),
                })
                .collect();
            x
        })
        .collect();
    let (filed, _) = r.post_batch(&runs, &[]).unwrap();
    assert_eq!(filed.unwrap().recorded, 2100);

    let api = Api::new(&srv.base, "default", "hk-test".into()).unwrap();
    let bytes = api.bytes("runs", &[]).unwrap();
    assert!(
        bytes.len() > 8 << 20,
        "the listing is {} bytes",
        bytes.len()
    );
    let listing: Listing<Run> = api.get("runs", &[]).unwrap();
    assert_eq!(listing.data.len(), 2100);
    assert_eq!(r.get_run("bench-2099").unwrap().measures.len(), 34);
}

/// A server whose every answer is `len` bytes of `x`.
fn flood(len: usize) -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    thread::spawn(move || {
        let body = vec![b'x'; len];
        for stream in listener.incoming() {
            let mut s = stream.unwrap();
            let mut reader = BufReader::new(s.try_clone().unwrap());
            loop {
                let mut h = String::new();
                if reader.read_line(&mut h).unwrap_or(0) == 0 || h == "\r\n" {
                    break;
                }
            }
            let head = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/octet-stream\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n"
            );
            let _ = s.write_all(head.as_bytes());
            let _ = s.write_all(&body);
        }
    });
    base
}

#[test]
fn an_answer_past_the_cap_is_refused_not_cut() {
    let base = flood(MAX_RESPONSE_BYTES as usize + 1);
    let api = Api::new(&base, "default", "hk-test".into()).unwrap();
    let err = api.bytes("runs", &[]).unwrap_err().to_string();
    assert!(err.contains("is refused"), "{err}");
    let r = Research::new(&base, "hk-test", "default", ".", &[]);
    let err = r.get_artifact(&"0".repeat(64)).unwrap_err().to_string();
    assert!(err.contains("is refused"), "{err}");

    let base = flood(MAX_RESPONSE_BYTES as usize);
    let api = Api::new(&base, "default", "hk-test".into()).unwrap();
    assert_eq!(
        api.bytes("runs", &[]).unwrap().len() as u64,
        MAX_RESPONSE_BYTES
    );
}
