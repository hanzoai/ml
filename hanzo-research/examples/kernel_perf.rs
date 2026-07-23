//! A kernel-perf run structured as a FALSIFIABLE test — the shape a `hanzo-bench` harness
//! writes. Runs the full flow against an in-process mock and prints the exact
//! `/v1/research` wire records, so the auto-captured provenance + the refutation are
//! visible.
//!
//!   cargo run -p hanzo-research --example kernel_perf

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver};
use std::thread;
use std::time::Duration;

use hanzo_research::{Research, Verdict};
use serde_json::Value;

fn main() {
    let (base, rx) = mock();
    // A bench harness makes ONE client (normally `Research::from_env()`); here it points at
    // the mock. repo "" auto-detects the git toplevel for zero-config provenance.
    let r = Research::new(&base, "demo-key", "enso-bench", "", &[]);

    // State the claim + prediction, log what you saw, then PROVE or REFUTE it.
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
    .expect("post to mock");

    println!("── /v1/research wire records the cloud received ─────────────");
    while let Ok((method, path, body)) = rx.recv_timeout(Duration::from_millis(300)) {
        if method != "POST" {
            continue;
        }
        let v: Value = serde_json::from_str(&body).unwrap();
        let exp = &v["experiments"][0];
        println!("\nPOST {path}   status={}  value={}", exp["status"], exp["value"]);
        println!("{}", serde_json::to_string_pretty(exp).unwrap());
    }
    println!("\nThe complete record's verdict is a first-class, durable REFUTATION,");
    println!("pinned to git_sha + lib_versions for longitudinal regression tracking.");
}

/// A minimal HTTP/1.1 mock: captures each request and prints nothing itself; the caller
/// drains the channel. std-only, no deps.
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
                r#"{"project":"enso-bench","experiments_ingested":1,"rolled_up":true}"#.to_string()
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
