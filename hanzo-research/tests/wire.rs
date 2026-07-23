//! End-to-end proof: a kernel-perf run structured as a falsifiable test drives the exact
//! `/v1/research` wire calls against a throwaway mock server. Asserts the in-flight record
//! is shape-identical to the Python SDK (frame set, empty log) and the sealed record
//! carries the refutation verdict + auto-captured provenance.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use hanzo_research::{Outcome, Research, Verdict};
use serde_json::{json, Value};

/// A minimal HTTP/1.1 mock: one request per connection, captures (method, path, body),
/// answers GETs with an empty listing and POSTs with an ingest ack. std-only, no deps.
fn mock() -> (String, Receiver<(String, String, String)>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let (tx, rx) = mpsc::channel();
    thread::spawn(move || {
        for stream in listener.incoming() {
            let mut s = stream.unwrap();
            let (method, path, body) = read_request(&mut s);
            let resp_body = if method == "GET" {
                r#"{"data":[],"total":0}"#.to_string()
            } else {
                r#"{"project":"proj","experiments_ingested":1,"rolled_up":true}"#.to_string()
            };
            let resp = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                resp_body.len(),
                resp_body
            );
            let _ = s.write_all(resp.as_bytes());
            let _ = tx.send((method, path, body));
        }
    });
    (format!("http://{addr}"), rx)
}

fn read_request(s: &mut TcpStream) -> (String, String, String) {
    let mut reader = BufReader::new(s.try_clone().unwrap());
    let mut line = String::new();
    reader.read_line(&mut line).unwrap();
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut len = 0usize;
    loop {
        let mut h = String::new();
        reader.read_line(&mut h).unwrap();
        if h == "\r\n" || h.is_empty() {
            break;
        }
        if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
            len = v.trim().parse().unwrap_or(0);
        }
    }
    let mut body = vec![0u8; len];
    if len > 0 {
        reader.read_exact(&mut body).unwrap();
    }
    (method, path, String::from_utf8_lossy(&body).to_string())
}

#[test]
fn kernel_perf_refutation_drives_the_wire() {
    let (base, rx) = mock();
    let r = Research::new(&base, "test-key", "proj", "", &[]);

    let mut k = r
        .experiment("kernel-perf", "matvec_q4k_f32_blk", "vulkan/6144x2048")
        .metric("ratio_vs_hand")
        .hypothesis("the DSL f32-direct matvec beats the hand kernel")
        .predict("DSL/hand >= 1.0 cold in-engine at the dominant FFN shape");
    k.log("cold in-engine A/B, evo gfx1151, quiet window, 3 runs, bit-exact 2.3e-6");
    k.conclude(
        Verdict::Refuted,
        "0.79x at 6144 rows — memory-BW wall, not craft",
        Some(0.79),
    )
    .unwrap();

    // Drain: a since-narrative GET, the running POST, the complete POST.
    let mut got_get = false;
    let mut posts: Vec<Value> = Vec::new();
    for _ in 0..3 {
        let (m, p, b) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        assert!(p.starts_with("/v1/research/"), "path was {p}");
        if m == "GET" {
            got_get = true;
        } else {
            posts.push(serde_json::from_str(&b).unwrap());
        }
    }
    assert!(got_get, "the since-narrative GET was issued");
    assert_eq!(posts.len(), 2, "a running post then a complete post");

    // The in-flight record: frame stated, no verdict, empty log — as the Python SDK posts.
    let running = &posts[0]["experiments"][0];
    assert_eq!(running["status"], "running");
    assert_eq!(
        running["meta"]["hypothesis"],
        "the DSL f32-direct matvec beats the hand kernel"
    );
    assert_eq!(running["meta"]["verdict"], "");
    assert_eq!(running["meta"]["log"], json!([]), "running log empty, as in Python");

    // The sealed record: the refutation is first-class + provenance auto-stamped.
    let done = &posts[1]["experiments"][0];
    assert_eq!(done["status"], "complete");
    assert_eq!(done["value"], 0.79);
    assert_eq!(done["metric"], "ratio_vs_hand");
    assert_eq!(done["meta"]["verdict"], "refuted");
    assert_eq!(
        done["meta"]["because"],
        "0.79x at 6144 rows — memory-BW wall, not craft"
    );
    assert_eq!(
        done["meta"]["log"][0],
        "cold in-engine A/B, evo gfx1151, quiet window, 3 runs, bit-exact 2.3e-6"
    );
    assert_eq!(done["git_sha"], running["git_sha"], "provenance consistent across versions");
    assert!(done["lib_versions"].is_object());
    assert!(done["meta"]["host"]["hostname"].is_string());
}

#[test]
fn benchmark_attempts_then_finish_computes_the_score() {
    let (base, rx) = mock();
    let r = Research::new(&base, "test-key", "enso-bench", "", &[]);

    let mut b = r.experiment("benchmark", "grok-4.5", "gpqa_diamond").n_total(2);
    b.record("q1", "grok-4.5", Outcome::correct("A")).unwrap();
    b.record("q2", "grok-4.5", Outcome::wrong("B", "C")).unwrap();
    b.finish(None).unwrap(); // value computed from attempts: 1 of 2 -> 50.0

    // GET(since) + running POST + 2 attempt POSTs + complete POST.
    let mut posts: Vec<Value> = Vec::new();
    for _ in 0..5 {
        let (m, _p, body) = rx.recv_timeout(Duration::from_secs(5)).unwrap();
        if m == "POST" {
            posts.push(serde_json::from_str(&body).unwrap());
        }
    }
    let finish = posts.last().unwrap();
    let exp = &finish["experiments"][0];
    assert_eq!(exp["status"], "complete");
    assert_eq!(exp["value"], 50.0, "1 correct of 2 scored");
    assert_eq!(exp["n"], 2);

    // The first attempt post carried the item under the experiment's task/benchmark.
    let first_attempt = posts.iter().find(|p| !p["attempts"].as_array().unwrap().is_empty()).unwrap();
    assert_eq!(first_attempt["attempts"][0]["benchmark"], "gpqa_diamond");
    assert_eq!(first_attempt["attempts"][0]["source"], "hanzo-measured");
}
