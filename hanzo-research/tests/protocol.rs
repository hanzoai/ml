//! End-to-end Rust <-> Python ledger interoperability and docs/research-sync read-back verification.

use std::collections::BTreeMap;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::Command;
use std::sync::{Arc, Mutex};
use std::thread;

use hanzo_research::{
    paired_comparison, student_t_ci95, CorrectnessChecks, ExperimentRecordV2, HardwareSpec,
    LocalResearchLedger, Research, WorkloadManifest,
};
use serde_json::{json, Value};

fn sample_rust_experiment(exp_id: &str) -> ExperimentRecordV2 {
    let base_vals = [100.0, 101.0, 99.0, 100.5, 99.5];
    let cand_vals = [160.0, 161.0, 159.0, 162.0, 160.0];
    let stat = paired_comparison("tok_per_sec", "tok/s", &base_vals, &cand_vals, true).unwrap();
    let mut raw = BTreeMap::new();
    raw.insert("baseline".into(), base_vals.to_vec());
    raw.insert("candidate".into(), cand_vals.to_vec());
    ExperimentRecordV2 {
        project: "default".into(),
        study: "rust-protocol-test".into(),
        objective: "Verify Rust <-> Python WAL + read-back ledger interop".into(),
        hypothesis: "Candidate kernel achieves >=1.5x throughput with 0.0 ULP parity".into(),
        experiment_id: exp_id.into(),
        parent_experiment: "baseline:0".into(),
        prior_best: Some(100.0),
        repo: "hanzoai/ml".into(),
        commit: "51701f9a672c84d5220ae9a7d5576926d8907822".into(),
        branch: "main".into(),
        pr: "#76".into(),
        artifact_hashes: BTreeMap::new(),
        hardware: HardwareSpec {
            host: "dgx".into(),
            architecture: "aarch64".into(),
            gpu: "NVIDIA GB10".into(),
            driver: "570".into(),
            compiler: "rustc 1.88".into(),
            backend: "CUDA".into(),
            precision: "BF16".into(),
            memory_gb: Some(128.0),
        },
        config: BTreeMap::new(),
        workload_manifest: WorkloadManifest {
            manifest_id: "wl-test".into(),
            dataset: "synthetic-4096".into(),
            split: "bench".into(),
            shapes: vec!["4096x4096".into()],
            input_lengths: vec![128],
            batch_size: 1,
            seed: 17,
            manifest_sha256: "b".repeat(64),
        },
        baseline: BTreeMap::from([("mean".into(), json!(100.0))]),
        candidate: BTreeMap::from([("mean".into(), json!(160.4))]),
        statistical_comparison: stat,
        correctness: CorrectnessChecks {
            parity_passed: true,
            max_abs_diff: Some(0.0),
            ulp_diff: Some(0.0),
            safety_passed: Some(true),
            memory_integrity_passed: true,
            details: "bit-exact parity".into(),
        },
        raw_samples: raw,
        gpu_time_sec: 1.2,
        compute_cost_usd: 0.01,
        conclusion: "1.604x throughput gain verified".into(),
        failure_diagnosis: String::new(),
        next_hypothesis: "Propagate to 9B FFN shape".into(),
        historical: false,
        recorded_at: String::new(),
        ..ExperimentRecordV2::default()
    }
}

fn spawn_mock_research_server() -> String {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let runs: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let exps: Arc<Mutex<Vec<Value>>> = Arc::new(Mutex::new(Vec::new()));
    let arts: Arc<Mutex<BTreeMap<String, Vec<u8>>>> = Arc::new(Mutex::new(BTreeMap::new()));
    thread::spawn(move || {
        for stream in listener.incoming() {
            let mut s = stream.unwrap();
            let (method, path, body) = read_req(&mut s);
            let route = path.split('?').next().unwrap_or(&path);
            let (ctype, payload): (&str, Vec<u8>) = if method == "POST"
                && route == "/v1/research/runs"
            {
                let v: Value = serde_json::from_slice(&body).unwrap();
                let list = v["runs"].as_array().cloned().unwrap_or_default();
                let mut g = runs.lock().unwrap();
                let n = list.len();
                g.extend(list);
                (
                    "application/json",
                    format!(r#"{{"project":"default","recorded":{n},"canonical":{n}}}"#)
                        .into_bytes(),
                )
            } else if method == "GET" && route == "/v1/research/runs" {
                let g = runs.lock().unwrap();
                let data = json!({"data": *g, "total": g.len()});
                ("application/json", serde_json::to_vec(&data).unwrap())
            } else if method == "POST"
                && (route == "/v1/research/benchmarks" || route == "/v1/research/studies")
            {
                (
                    "application/json",
                    br#"{"project":"default","recorded":1,"canonical":1}"#.to_vec(),
                )
            } else if method == "POST" && route == "/v1/research/experiments" {
                let v: Value = serde_json::from_slice(&body).unwrap();
                let list = v["experiments"].as_array().cloned().unwrap_or_default();
                let mut g = exps.lock().unwrap();
                let n = list.len();
                g.extend(list);
                (
                    "application/json",
                    format!(
                        r#"{{"project":"default","experiments_ingested":{n},"rolled_up":true}}"#
                    )
                    .into_bytes(),
                )
            } else if method == "GET" && route == "/v1/research/experiments" {
                let g = exps.lock().unwrap();
                let data = json!({"data": *g, "total": g.len()});
                ("application/json", serde_json::to_vec(&data).unwrap())
            } else if method == "POST" && route == "/v1/research/artifacts" {
                let v: Value = serde_json::from_slice(&body).unwrap();
                let sha = v["sha256"].as_str().unwrap().to_string();
                let b64 = v["content"].as_str().unwrap();
                let raw = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)
                    .unwrap();
                arts.lock().unwrap().insert(sha.clone(), raw);
                (
                    "application/json",
                    format!(r#"{{"sha256":"{sha}","created":true}}"#).into_bytes(),
                )
            } else if method == "GET" && route.starts_with("/v1/research/artifacts/") {
                let sha = route.rsplit('/').next().unwrap();
                let bytes = arts.lock().unwrap().get(sha).cloned().unwrap_or_default();
                ("application/octet-stream", bytes)
            } else {
                ("application/json", br#"{"data":[],"total":0}"#.to_vec())
            };
            let hdr = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                payload.len()
            );
            let _ = s.write_all(hdr.as_bytes());
            let _ = s.write_all(&payload);
        }
    });
    format!("http://{addr}")
}

fn read_req(s: &mut TcpStream) -> (String, String, Vec<u8>) {
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
    (method, path, body)
}

#[test]
fn student_t_and_validation_work() {
    let st = student_t_ci95(&[10.0, 11.0, 12.0]).unwrap();
    assert!((st.mean - 11.0).abs() < 1e-12);
    assert!(st.ci95.unwrap() > 2.0);
    let mut bad = sample_rust_experiment("bad:1");
    bad.raw_samples.clear();
    assert!(bad.validate().is_err());
}

#[test]
fn rust_python_ledger_wal_and_cloud_readback_interop() {
    let dir = std::env::temp_dir().join(format!("hanzo-research-proto-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let ledger = LocalResearchLedger::open(&dir, "shared-e2e").unwrap();
    let mut exp = sample_rust_experiment("ml:bitcos:rust-e2e");
    let r = ledger
        .record_experiment(&mut exp, &[("metrics", b"{\"ok\":true}")])
        .unwrap();
    assert_eq!(r.status, "pending_sync");

    // Verify Python SDK reads the Rust-written ledger, appends a Python experiment, and Rust reads it back.
    let py = Command::new("python3")
        .arg("-c")
        .arg(format!(
            "from hanzo_research import LocalResearchLedger, ExperimentRecordV2, HardwareSpec, WorkloadManifest, StatisticalComparison, CorrectnessChecks, wal; \
             l = LocalResearchLedger({:?}, run_id='shared-e2e'); \
             recs = l.query(); \
             assert len(recs) == 1 and recs[0].experiment_id == 'ml:bitcos:rust-e2e'; \
             py_exp = ExperimentRecordV2( \
               project='default', study='py-proto', objective='Py->Rust check', hypothesis='Roundtrip exact', \
               experiment_id='engine:py-to-rust-e2e', repo='hanzoai/engine', commit='6ed8fc6f3', \
               artifact_hashes={{'a': 'c'*64}}, hardware=HardwareSpec(host='dgx', backend='CUDA', precision='BF16'), \
               workload_manifest=WorkloadManifest(manifest_id='m1', dataset='d1', split='bench'), \
               baseline={{'ms': 10.0}}, candidate={{'ms': 5.0}}, \
               statistical_comparison=StatisticalComparison(metric='ms', unit='ms', higher_is_better=False, n=3, baseline_mean=10.0, candidate_mean=5.0, verdict='win'), \
               correctness=CorrectnessChecks(parity_passed=True, memory_integrity_passed=True), \
               raw_samples={{'baseline': [10.0, 10.0, 10.0], 'candidate': [5.0, 5.0, 5.0]}}, \
               conclusion='2.0x speedup verified'); \
             l.record_experiment(py_exp); \
             wal.verify(str(l.wal_dir / 'wal.jsonl')); \
             print('PY_OK')",
            dir.to_string_lossy()
        ))
        .output();
    if let Ok(out) = py {
        let stderr = String::from_utf8_lossy(&out.stderr);
        if !stderr.contains("No module named 'hanzo_research'") {
            assert!(out.status.success(), "Python interop failed: {stderr}");
            assert!(String::from_utf8_lossy(&out.stdout).contains("PY_OK"));
            let all = ledger.query_all().unwrap();
            assert_eq!(
                all.len(),
                2,
                "Rust should read both Rust and Python experiments"
            );
            assert!(all
                .iter()
                .any(|r| r.experiment_id == "engine:py-to-rust-e2e"));
        }
    }

    // Verify docs/research-sync artifact upload & byte-for-byte SHA-256 readback via mock server.
    let base = spawn_mock_research_server();
    let client = Research::new(&base, "demo-key", "default", "", &[]);
    let receipts = ledger.sync_to_cloud(&client).unwrap();
    assert!(!receipts.is_empty());
    assert!(receipts
        .iter()
        .all(|rc| rc.status == "synced_verified" && rc.readback_verified));
    let _ = std::fs::remove_dir_all(&dir);
}
