//! Comprehensive autonomy, HIP-1403, WAL/spool crash-safety, cloud lease fencing, and CLI conformance tests.

use std::collections::BTreeMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;

use hanzo_research::{
    epistemic_verdict, generate_transfer_hypotheses, impacted_benchmarks, paired_comparison,
    research_yield, rfc8785_digest, successor_research_lift, wal, AutonomousController,
    CampaignRegistry, CloudLeaseManager, CorrectnessChecks, CounterEvidence, ExperimentRecordV2,
    HardwareSpec, HypothesisRecord, LineageRule, LocalResearchLedger, ObjectiveSpec,
    OrchestrateOptions, ProgressiveAbPlan, ProgressiveAbRunner, ProgressiveStage,
    ProgressiveStageOutput, Research, ResearchBudget, ResearchWal, ScheduleOptions, SearchArm,
    Spooler, TransferCatalog, WorkloadManifest,
};
use serde_json::{json, Value};

struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        static SEQ: AtomicUsize = AtomicUsize::new(1);
        let n = SEQ.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!(
            "hanzo-research-autonomy-{}-{}-{n}",
            tag,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        Self(p)
    }

    fn path(&self) -> &std::path::Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn test_client(base: &str) -> Research {
    Research::new(base, "hk-test-key", "default", "hanzoai/ml", &[])
}

fn read_http_req(stream: &mut TcpStream) -> (String, String, BTreeMap<String, String>, Vec<u8>) {
    let mut reader = BufReader::new(stream);
    let mut first = String::new();
    if reader.read_line(&mut first).unwrap_or(0) == 0 {
        return (String::new(), String::new(), BTreeMap::new(), Vec::new());
    }
    let mut parts = first.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut headers = BTreeMap::new();
    let mut content_len = 0usize;
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).unwrap();
        let trimmed = line.trim();
        if trimmed.is_empty() {
            break;
        }
        if let Some((k, v)) = trimmed.split_once(':') {
            let key = k.trim().to_ascii_lowercase();
            let val = v.trim().to_string();
            if key == "content-length" {
                content_len = val.parse().unwrap_or(0);
            }
            headers.insert(key, val);
        }
    }
    let mut body = vec![0u8; content_len];
    if content_len > 0 {
        reader.read_exact(&mut body).unwrap();
    }
    (method, path, headers, body)
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let Ok(hex) = std::str::from_utf8(&bytes[i + 1..i + 3]) {
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    out.push(b);
                    i += 3;
                    continue;
                }
            }
        }
        if bytes[i] == b'+' {
            out.push(b' ');
        } else {
            out.push(bytes[i]);
        }
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

struct MockCloudState {
    fail_evidence_posts: AtomicUsize,
    evidence_by_uri: Mutex<BTreeMap<String, Vec<Value>>>,
    evidence_idem: Mutex<BTreeMap<String, Value>>,
    runs: Mutex<Vec<Value>>,
    studies: Mutex<Vec<Value>>,
    benchmarks: Mutex<Vec<Value>>,
    exps: Mutex<Vec<Value>>,
    artifacts: Mutex<BTreeMap<String, Vec<u8>>>,
    artifact_meta: Mutex<Vec<Value>>,
    workflow_jobs: Mutex<BTreeMap<String, Value>>,
    workflow_schedules: Mutex<BTreeMap<String, Value>>,
    fleet_workers: Mutex<Vec<Value>>,
    agent_targets: Mutex<Vec<Value>>,
    fleet_presence: Mutex<BTreeMap<String, Value>>,
    busy_defer_queues: Mutex<Vec<String>>,
    sse_streams_opened: AtomicUsize,
    stage_mode: AtomicUsize,
}

fn spawn_full_mock_cloud(fail_first_evidence_posts: usize) -> (String, Arc<MockCloudState>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let state = Arc::new(MockCloudState {
        fail_evidence_posts: AtomicUsize::new(fail_first_evidence_posts),
        evidence_by_uri: Mutex::new(BTreeMap::new()),
        evidence_idem: Mutex::new(BTreeMap::new()),
        runs: Mutex::new(Vec::new()),
        studies: Mutex::new(Vec::new()),
        benchmarks: Mutex::new(Vec::new()),
        exps: Mutex::new(Vec::new()),
        artifacts: Mutex::new(BTreeMap::new()),
        artifact_meta: Mutex::new(Vec::new()),
        workflow_jobs: Mutex::new(BTreeMap::new()),
        workflow_schedules: Mutex::new(BTreeMap::new()),
        fleet_workers: Mutex::new(Vec::new()),
        agent_targets: Mutex::new(Vec::new()),
        fleet_presence: Mutex::new(BTreeMap::new()),
        busy_defer_queues: Mutex::new(Vec::new()),
        sse_streams_opened: AtomicUsize::new(0),
        stage_mode: AtomicUsize::new(0),
    });
    let st_listener = Arc::clone(&state);
    thread::spawn(move || {
        for stream in listener.incoming() {
            let Ok(mut s) = stream else { break };
            let st = Arc::clone(&st_listener);
            thread::spawn(move || {
                let (method, full_path, headers, body) = read_http_req(&mut s);
                if method.is_empty() {
                    return;
                }
                let (route, query) = full_path
                    .split_once('?')
                    .map_or((full_path.as_str(), ""), |(r, q)| (r, q));
                let mut qmap = BTreeMap::new();
                for pair in query.split('&').filter(|p| !p.is_empty()) {
                    if let Some((k, v)) = pair.split_once('=') {
                        qmap.insert(percent_decode(k), percent_decode(v));
                    }
                }

                if method == "GET" && route == "/v1/workflow/events" {
                    st.sse_streams_opened.fetch_add(1, Ordering::SeqCst);
                    let hdr = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n: hanzo-tasks event stream\n\n";
                    let _ = s.write_all(hdr.as_bytes());
                    let _ = s.flush();
                    let defer_qs = st.busy_defer_queues.lock().unwrap().clone();
                    let mut jobs = st.workflow_jobs.lock().unwrap();
                    for (jid, job) in jobs.iter_mut() {
                        let tq = job["taskQueue"].as_str().unwrap_or("").to_string();
                        let rid = job["runId"].as_str().unwrap_or(jid).to_string();
                        if defer_qs.contains(&tq) {
                            let dc = job["deferCount"].as_u64().unwrap_or(0).max(1);
                            job["deferCount"] = json!(dc);
                            if job["lastBusyReason"].as_str().unwrap_or("").is_empty() {
                                job["lastBusyReason"] = json!("active_dev_session");
                            }
                            let dev = json!({
                                "kind": "activity.deferred",
                                "namespace": "gpu-jobs",
                                "workflow_id": jid,
                                "run_id": rid,
                                "data": {"deferCount": dc, "lastBusyReason": job["lastBusyReason"]},
                            });
                            let frame = format!("event: activity.deferred\ndata: {dev}\n\n");
                            let _ = s.write_all(frame.as_bytes());
                        }
                        job["sse_completed"] = json!(true);
                        let cev = json!({
                            "kind": "activity.completed",
                            "namespace": "gpu-jobs",
                            "workflow_id": jid,
                            "run_id": rid,
                            "data": {"status": "ACTIVITY_TASK_STATE_COMPLETED"},
                        });
                        let frame = format!("event: activity.completed\ndata: {cev}\n\n");
                        let _ = s.write_all(frame.as_bytes());
                    }
                    let _ = s.flush();
                    return;
                }

                let (status, ctype, payload): (u16, &str, Vec<u8>) = if method == "POST"
                    && route == "/v1/research/evidence"
                {
                    let rem = st.fail_evidence_posts.load(Ordering::SeqCst);
                    if rem > 0 {
                        st.fail_evidence_posts.fetch_sub(1, Ordering::SeqCst);
                        (
                            503,
                            "application/json",
                            br#"{"error":"simulated offline"}"#.to_vec(),
                        )
                    } else {
                        let idem = headers.get("idempotency-key").cloned().unwrap_or_default();
                        if !idem.is_empty() {
                            if let Some(existing) =
                                st.evidence_idem.lock().unwrap().get(&idem).cloned()
                            {
                                let raw = serde_json::to_vec(&existing).unwrap();
                                let resp = format!(
                                    "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                                    raw.len()
                                );
                                let _ = s.write_all(resp.as_bytes());
                                let _ = s.write_all(&raw);
                                return;
                            }
                        }
                        let mut v: Value = serde_json::from_slice(&body).unwrap();
                        let uri = v["uri"].as_str().unwrap_or("").to_string();
                        let mut by_uri = st.evidence_by_uri.lock().unwrap();
                        let versions = by_uri.entry(uri).or_default();
                        let next_ver = (versions.len() + 1) as i64;
                        v["version"] = json!(next_ver);
                        v["project"] = json!("default");
                        versions.push(v.clone());
                        drop(by_uri);
                        if !idem.is_empty() {
                            st.evidence_idem.lock().unwrap().insert(idem, v.clone());
                        }
                        (200, "application/json", serde_json::to_vec(&v).unwrap())
                    }
                } else if method == "GET" && route == "/v1/research/evidence" {
                    let by_uri = st.evidence_by_uri.lock().unwrap();
                    let rows: Vec<Value> = if let Some(uri) = qmap.get("uri") {
                        by_uri.get(uri).cloned().unwrap_or_default()
                    } else {
                        by_uri.values().flatten().cloned().collect()
                    };
                    let data = json!({"data": rows, "total": rows.len()});
                    (200, "application/json", serde_json::to_vec(&data).unwrap())
                } else if method == "POST" && route == "/v1/research/runs" {
                    let v: Value = serde_json::from_slice(&body).unwrap();
                    let list = v["runs"].as_array().cloned().unwrap_or_default();
                    let mut g = st.runs.lock().unwrap();
                    let n = list.len();
                    g.extend(list);
                    (
                        200,
                        "application/json",
                        serde_json::to_vec(
                            &json!({"project":"default","recorded":n,"canonical":n}),
                        )
                        .unwrap(),
                    )
                } else if method == "GET" && route == "/v1/research/runs" {
                    let g = st.runs.lock().unwrap();
                    let rows: Vec<Value> = if let Some(id) = qmap.get("id") {
                        g.iter()
                            .filter(|r| r["id"].as_str() == Some(id.as_str()))
                            .cloned()
                            .collect()
                    } else {
                        g.clone()
                    };
                    let data = json!({"data": rows, "total": rows.len()});
                    (200, "application/json", serde_json::to_vec(&data).unwrap())
                } else if method == "POST" && route == "/v1/research/studies" {
                    let v: Value = serde_json::from_slice(&body).unwrap();
                    let list = v["studies"].as_array().cloned().unwrap_or_default();
                    let mut g = st.studies.lock().unwrap();
                    let n = list.len();
                    g.extend(list);
                    (
                        200,
                        "application/json",
                        serde_json::to_vec(
                            &json!({"project":"default","recorded":n,"canonical":n}),
                        )
                        .unwrap(),
                    )
                } else if method == "GET" && route == "/v1/research/studies" {
                    let g = st.studies.lock().unwrap();
                    let data = json!({"data": *g, "total": g.len()});
                    (200, "application/json", serde_json::to_vec(&data).unwrap())
                } else if method == "POST" && route == "/v1/research/benchmarks" {
                    let v: Value = serde_json::from_slice(&body).unwrap();
                    let list = v["benchmarks"].as_array().cloned().unwrap_or_default();
                    let mut g = st.benchmarks.lock().unwrap();
                    let n = list.len();
                    g.extend(list);
                    (
                        200,
                        "application/json",
                        serde_json::to_vec(
                            &json!({"project":"default","recorded":n,"canonical":n}),
                        )
                        .unwrap(),
                    )
                } else if method == "POST" && route == "/v1/research/experiments" {
                    let v: Value = serde_json::from_slice(&body).unwrap();
                    let list = v["experiments"].as_array().cloned().unwrap_or_default();
                    let mut g = st.exps.lock().unwrap();
                    let n = list.len();
                    for item in list {
                        g.retain(|x| x["id"] != item["id"]);
                        g.push(item);
                    }
                    (
                        200,
                        "application/json",
                        serde_json::to_vec(
                            &json!({"project":"default","experiments_ingested":n,"rolled_up":true}),
                        )
                        .unwrap(),
                    )
                } else if method == "GET" && route == "/v1/research/experiments" {
                    let g = st.exps.lock().unwrap();
                    let data = json!({"data": *g, "total": g.len()});
                    (200, "application/json", serde_json::to_vec(&data).unwrap())
                } else if method == "POST" && route == "/v1/research/artifacts" {
                    let v: Value = serde_json::from_slice(&body).unwrap();
                    let kind = v["kind"].as_str().unwrap_or("");
                    if kind != "snapshot" && kind != "report" {
                        (
                            400,
                            "application/json",
                            br#"{"error":"invalid artifact kind"}"#.to_vec(),
                        )
                    } else {
                        let sha = v["sha256"].as_str().unwrap().to_string();
                        let b64 = v["content"].as_str().unwrap();
                        let raw =
                            base64::Engine::decode(&base64::engine::general_purpose::STANDARD, b64)
                                .unwrap();
                        let mut stored_meta = v.clone();
                        stored_meta["received_at"] = json!(1_700_000_999_i64);
                        st.artifact_meta.lock().unwrap().push(stored_meta);
                        st.artifacts.lock().unwrap().insert(sha.clone(), raw);
                        (
                            200,
                            "application/json",
                            serde_json::to_vec(
                                &json!({"sha256":sha,"artifact_id":v["artifact_id"]}),
                            )
                            .unwrap(),
                        )
                    }
                } else if method == "GET" && route == "/v1/research/artifacts" {
                    let g = st.artifact_meta.lock().unwrap();
                    let data = json!({"data": *g, "total": g.len()});
                    (200, "application/json", serde_json::to_vec(&data).unwrap())
                } else if method == "GET" && route.starts_with("/v1/research/artifacts/") {
                    let sha = route.trim_start_matches("/v1/research/artifacts/");
                    let raw = st
                        .artifacts
                        .lock()
                        .unwrap()
                        .get(sha)
                        .cloned()
                        .unwrap_or_default();
                    (200, "application/octet-stream", raw)
                } else if method == "GET" && route == "/v1/compute/fleet/workers" {
                    let rows = st.fleet_workers.lock().unwrap().clone();
                    (
                        200,
                        "application/json",
                        serde_json::to_vec(&json!({"workers": rows})).unwrap(),
                    )
                } else if method == "GET" && route == "/v1/agent/targets" {
                    let rows = st.agent_targets.lock().unwrap().clone();
                    (
                        200,
                        "application/json",
                        serde_json::to_vec(&json!({"targets": rows})).unwrap(),
                    )
                } else if method == "POST" && route == "/v1/workflow/namespaces" {
                    let v: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
                    (200, "application/json", serde_json::to_vec(&v).unwrap())
                } else if method == "GET"
                    && route.starts_with("/v1/workflow/namespaces/fleet/activities/")
                {
                    let wid = route
                        .trim_start_matches("/v1/workflow/namespaces/fleet/activities/")
                        .split('/')
                        .next()
                        .unwrap_or("");
                    if let Some(pres) = st.fleet_presence.lock().unwrap().get(wid).cloned() {
                        (200, "application/json", serde_json::to_vec(&pres).unwrap())
                    } else {
                        (404, "application/json", b"{}".to_vec())
                    }
                } else if method == "POST" && route == "/v1/workflow/namespaces/gpu-jobs/activities"
                {
                    let v: Value = serde_json::from_slice(&body).unwrap();
                    let jid = v["activityId"]
                        .as_str()
                        .or_else(|| v["id"].as_str())
                        .unwrap_or("job-1")
                        .to_string();
                    let rid = v["runId"].as_str().unwrap_or(&jid).to_string();
                    let act_type = v["activityType"]["name"]
                        .as_str()
                        .or_else(|| v["type"].as_str())
                        .unwrap_or("research.bench")
                        .to_string();
                    let rec = json!({
                        "id": jid,
                        "activityId": jid,
                        "runId": rid,
                        "type": act_type,
                        "taskQueue": v["taskQueue"],
                        "input": v["input"],
                        "retryPolicy": v["retryPolicy"],
                        "heartbeatTimeout": v["heartbeatTimeout"],
                        "scheduleToCloseTimeout": v["scheduleToCloseTimeout"],
                        "polls": 0,
                        "deferCount": 0,
                        "lastBusyReason": "",
                        "sse_completed": false,
                    });
                    st.workflow_jobs.lock().unwrap().insert(jid.clone(), rec);
                    let out = json!({
                        "id": jid,
                        "activityId": jid,
                        "runId": rid,
                        "execution": {"workflowId": jid, "runId": rid},
                        "taskQueue": v["taskQueue"],
                        "type": act_type,
                        "input": v["input"],
                        "status": "ACTIVITY_TASK_STATE_SCHEDULED",
                        "state": "ACTIVITY_TASK_STATE_SCHEDULED",
                    });
                    (200, "application/json", serde_json::to_vec(&out).unwrap())
                } else if method == "POST"
                    && route.starts_with("/v1/workflow/namespaces/gpu-jobs/activities/")
                    && route.ends_with("/release")
                {
                    let rest =
                        route.trim_start_matches("/v1/workflow/namespaces/gpu-jobs/activities/");
                    let jid = rest.split('/').next().unwrap_or("").to_string();
                    let v: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
                    let mut jobs = st.workflow_jobs.lock().unwrap();
                    if let Some(job) = jobs.get_mut(&jid) {
                        let dc = job["deferCount"].as_u64().unwrap_or(0) + 1;
                        let reason = v["reason"].as_str().unwrap_or("busy");
                        job["deferCount"] = json!(dc);
                        job["lastBusyReason"] = json!(reason);
                        let out = json!({
                            "id": jid,
                            "status": "ACTIVITY_TASK_STATE_SCHEDULED",
                            "state": "ACTIVITY_TASK_STATE_SCHEDULED",
                            "deferCount": dc,
                            "lastBusyReason": reason,
                        });
                        (200, "application/json", serde_json::to_vec(&out).unwrap())
                    } else {
                        (404, "application/json", b"{}".to_vec())
                    }
                } else if method == "GET"
                    && route.starts_with("/v1/workflow/namespaces/gpu-jobs/activities/")
                {
                    let jid = route
                        .trim_start_matches("/v1/workflow/namespaces/gpu-jobs/activities/")
                        .split('/')
                        .next()
                        .unwrap_or("")
                        .to_string();
                    let mut jobs = st.workflow_jobs.lock().unwrap();
                    if let Some(job) = jobs.get_mut(&jid) {
                        let p = job["polls"].as_u64().unwrap_or(0) + 1;
                        job["polls"] = json!(p);
                        let state_str = if job["sse_completed"].as_bool().unwrap_or(false) || p >= 2
                        {
                            "ACTIVITY_TASK_STATE_COMPLETED"
                        } else {
                            "ACTIVITY_TASK_STATE_STARTED"
                        };
                        let out_val = if st.stage_mode.load(Ordering::SeqCst) == 1
                            && state_str == "ACTIVITY_TASK_STATE_COMPLETED"
                        {
                            json!({
                                "status": "complete",
                                "trial_id": job["input"]["trial_id"],
                                "objective_id": job["input"]["objective_id"],
                                "baseline_samples": [100.0, 100.2, 99.8, 100.1],
                                "candidate_samples": [172.0, 173.0, 171.5, 172.5],
                                "parity_passed": true,
                                "memory_integrity_passed": true,
                                "gpu_time_sec": 0.12,
                                "details": "remote dgx stage win",
                            })
                        } else {
                            json!({"tok_per_sec": 182.4})
                        };
                        let out = json!({
                            "id": jid,
                            "status": state_str,
                            "state": state_str,
                            "taskQueue": job["taskQueue"],
                            "deferCount": job["deferCount"].as_u64().unwrap_or(0),
                            "lastBusyReason": job["lastBusyReason"].as_str().unwrap_or(""),
                            "result": out_val,
                            "output": out_val,
                        });
                        (200, "application/json", serde_json::to_vec(&out).unwrap())
                    } else {
                        (404, "application/json", b"{}".to_vec())
                    }
                } else if method == "POST" && route == "/v1/workflow/namespaces/gpu-jobs/schedules"
                {
                    let mut v: Value = serde_json::from_slice(&body).unwrap();
                    let sid = v["scheduleId"].as_str().unwrap_or("").to_string();
                    if v["state"].is_null() {
                        v["state"] = json!({"paused": false, "note": ""});
                    }
                    v["info"] = json!({"actionCount": 0});
                    st.workflow_schedules.lock().unwrap().insert(sid, v.clone());
                    (200, "application/json", serde_json::to_vec(&v).unwrap())
                } else if method == "GET" && route == "/v1/workflow/namespaces/gpu-jobs/schedules" {
                    let rows: Vec<Value> = st
                        .workflow_schedules
                        .lock()
                        .unwrap()
                        .values()
                        .cloned()
                        .collect();
                    (
                        200,
                        "application/json",
                        serde_json::to_vec(&json!({"schedules": rows})).unwrap(),
                    )
                } else if route.starts_with("/v1/workflow/namespaces/gpu-jobs/schedules/") {
                    let rest =
                        route.trim_start_matches("/v1/workflow/namespaces/gpu-jobs/schedules/");
                    let parts: Vec<&str> = rest.split('/').filter(|p| !p.is_empty()).collect();
                    let mut scheds = st.workflow_schedules.lock().unwrap();
                    if parts.len() == 1 && method == "GET" {
                        if let Some(sc) = scheds.get(parts[0]).cloned() {
                            (200, "application/json", serde_json::to_vec(&sc).unwrap())
                        } else {
                            (404, "application/json", b"{}".to_vec())
                        }
                    } else if parts.len() == 1 && method == "DELETE" {
                        scheds.remove(parts[0]);
                        (
                            200,
                            "application/json",
                            serde_json::to_vec(&json!({"status": "deleted"})).unwrap(),
                        )
                    } else if parts.len() == 2 && parts[1] == "trigger" && method == "POST" {
                        (
                            200,
                            "application/json",
                            serde_json::to_vec(&json!({"status": "triggered"})).unwrap(),
                        )
                    } else if parts.len() == 2 && parts[1] == "pause" && method == "POST" {
                        let v: Value = serde_json::from_slice(&body).unwrap_or_else(|_| json!({}));
                        if let Some(sc) = scheds.get_mut(parts[0]) {
                            sc["state"] =
                                json!({"paused": true, "note": v["note"].as_str().unwrap_or("")});
                        }
                        (
                            200,
                            "application/json",
                            serde_json::to_vec(&json!({"status": "paused"})).unwrap(),
                        )
                    } else if parts.len() == 2 && parts[1] == "unpause" && method == "POST" {
                        if let Some(sc) = scheds.get_mut(parts[0]) {
                            sc["state"] = json!({"paused": false, "note": ""});
                        }
                        (
                            200,
                            "application/json",
                            serde_json::to_vec(&json!({"status": "running"})).unwrap(),
                        )
                    } else {
                        (404, "application/json", b"{}".to_vec())
                    }
                } else {
                    (404, "application/json", b"{}".to_vec())
                };

                let reason = match status {
                    200 => "OK",
                    400 => "Bad Request",
                    503 => "Service Unavailable",
                    _ => "Not Found",
                };
                let header = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: {ctype}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    payload.len()
                );
                let _ = s.write_all(header.as_bytes());
                let _ = s.write_all(&payload);
            });
        }
    });
    (format!("http://{addr}"), state)
}

#[test]
fn wal_multiprocess_helper_child() {
    let Ok(dir) = std::env::var("HANZO_TEST_WAL_CHILD_DIR") else {
        return;
    };
    let count: usize = std::env::var("HANZO_TEST_WAL_CHILD_COUNT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1000);
    let proc_tag = std::env::var("HANZO_TEST_WAL_CHILD_TAG").unwrap_or_else(|_| "p0".into());
    let mut w = ResearchWal::open(std::path::Path::new(&dir), "mp-run").unwrap();
    for i in 0..count {
        w.append("tick", &json!({"proc": proc_tag, "i": i}))
            .unwrap();
    }
}

#[test]
fn wal_2_procs_x_1000_crash_safety_and_cross_lang_floats() {
    let tmp = TempDir::new("wal");
    let wal_dir = tmp.path().join("mp_wal");
    fs::create_dir_all(&wal_dir).unwrap();

    let exe = std::env::current_exe().unwrap();
    let mut c1 = Command::new(&exe)
        .args(["--exact", "wal_multiprocess_helper_child", "--nocapture"])
        .env("HANZO_TEST_WAL_CHILD_DIR", &wal_dir)
        .env("HANZO_TEST_WAL_CHILD_COUNT", "1000")
        .env("HANZO_TEST_WAL_CHILD_TAG", "p1")
        .spawn()
        .unwrap();
    let mut c2 = Command::new(&exe)
        .args(["--exact", "wal_multiprocess_helper_child", "--nocapture"])
        .env("HANZO_TEST_WAL_CHILD_DIR", &wal_dir)
        .env("HANZO_TEST_WAL_CHILD_COUNT", "1000")
        .env("HANZO_TEST_WAL_CHILD_TAG", "p2")
        .spawn()
        .unwrap();
    assert!(c1.wait().unwrap().success());
    assert!(c2.wait().unwrap().success());

    let wal_file = wal_dir.join("wal.jsonl");
    let head = wal::verify(&wal_file).unwrap();
    assert_eq!(head.len(), 64);
    let events = wal::read(&wal_file).unwrap();
    assert_eq!(events.len(), 2000);
    for (idx, ev) in events.iter().enumerate() {
        assert_eq!(ev.sequence, (idx + 1) as u64);
    }

    // Torn final line mends cleanly; interior tamper fails closed on open.
    let crash_dir = tmp.path().join("crash_wal");
    {
        let mut w = ResearchWal::open(&crash_dir, "run-crash").unwrap();
        w.append("step", &json!({"k": 1})).unwrap();
        w.append("step", &json!({"k": 2})).unwrap();
    }
    {
        let mut f = OpenOptions::new()
            .append(true)
            .open(crash_dir.join("wal.jsonl"))
            .unwrap();
        f.write_all(br#"{"event_id":"torn","sequence":3"#).unwrap();
    }
    let mut reopened = ResearchWal::open(&crash_dir, "run-crash").unwrap();
    assert_eq!(reopened.sequence(), 2);
    let ev3 = reopened
        .append_idempotent(None, "resume_step", &json!({"step": 3, "v": 0.1}))
        .unwrap();
    let ev3_dup = reopened
        .append_idempotent(None, "resume_step", &json!({"v": 0.1, "step": 3}))
        .unwrap();
    assert_eq!(ev3.sequence, 3);
    assert_eq!(ev3.event_id, ev3_dup.event_id);
    assert_eq!(wal::read(crash_dir.join("wal.jsonl")).unwrap().len(), 3);

    // Collision with explicit event_id and different payload must error.
    let collision =
        reopened.append_idempotent(Some(&ev3.event_id), "resume_step", &json!({"step": 999}));
    assert!(collision.is_err());

    // Tamper with interior line 1 so ResearchWal::open fails closed.
    let raw = fs::read_to_string(crash_dir.join("wal.jsonl")).unwrap();
    let mut lines: Vec<String> = raw.lines().map(String::from).collect();
    lines[0] = lines[0].replace("\"k\":1", "\"k\":99");
    fs::write(
        crash_dir.join("wal.jsonl"),
        format!("{}\n", lines.join("\n")),
    )
    .unwrap();
    assert!(ResearchWal::open(&crash_dir, "run-crash").is_err());

    // Cross-language float edge cases: 1e21, 1e-7, -0.0, 0.1 + permuted keys.
    let xlang_dir = tmp.path().join("xlang");
    let mut xw = ResearchWal::open(&xlang_dir, "xlang-run").unwrap();
    xw.append(
        "floats_rust",
        &json!({
            "z_big": 1e21_f64,
            "a_small": 1e-7_f64,
            "m_neg_zero": -0.0_f64,
            "b_tenth": 0.1_f64,
        }),
    )
    .unwrap();

    let py_sdk = PathBuf::from("/home/z/work/hanzo/python-sdk/pkg/hanzo-research");
    if py_sdk.exists() {
        let py_script = r#"
import sys
from hanzo_research import wal
from hanzo_research.wal import ResearchWal

d = sys.argv[1]
head = wal.verify(d + "/wal.jsonl")
assert len(head) == 64
w = ResearchWal.open(d, "xlang-run")
w.append("floats_py", {"b_tenth": 0.1, "m_neg_zero": -0.0, "a_small": 1e-7, "z_big": 1e21})
"#;
        let out = Command::new("python3")
            .args(["-c", py_script, xlang_dir.to_str().unwrap()])
            .env("PYTHONPATH", &py_sdk)
            .output()
            .unwrap();
        assert!(
            out.status.success(),
            "python verify/append failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let final_events = wal::read(xlang_dir.join("wal.jsonl")).unwrap();
        assert_eq!(final_events.len(), 2);
        assert_eq!(final_events[0].payload_hash, final_events[1].payload_hash);
        assert!(wal::verify(xlang_dir.join("wal.jsonl")).is_ok());
    }
}

#[test]
fn spooler_offline_backoff_dedup_and_cursor_survival() {
    let tmp = TempDir::new("spool");
    let wal_dir = tmp.path().join("wal");
    let cursor_path = tmp.path().join("cursor.json");
    let mut w = ResearchWal::open(&wal_dir, "spool-run").unwrap();
    w.append("ev1", &json!({"idx": 1})).unwrap();
    w.append("ev2", &json!({"idx": 2})).unwrap();
    w.append("ev3", &json!({"idx": 3})).unwrap();

    let (base, state) = spawn_full_mock_cloud(2);
    let client = test_client(&base);

    // With max_retries=0, first transient 503 reports offline cleanly without losing state.
    let r_offline = Spooler::drain(&client, w.wal_path(), &cursor_path, 0, 5).unwrap();
    assert!(r_offline.offline);
    assert_eq!(r_offline.drained, 0);
    assert_eq!(r_offline.pending, 3);

    // Next drain retries past the remaining transient 503 and delivers all 3 events.
    let r_ok = Spooler::drain(&client, w.wal_path(), &cursor_path, 3, 5).unwrap();
    assert!(!r_ok.offline);
    assert_eq!(r_ok.drained, 3);
    assert_eq!(r_ok.committed_seq, 3);

    // Cursor survives restart: draining again sends 0 events.
    let r_noop = Spooler::drain(&client, w.wal_path(), &cursor_path, 2, 5).unwrap();
    assert_eq!(r_noop.drained, 0);
    assert_eq!(r_noop.committed_seq, 3);

    // Simulate crash before cursor fsync by deleting cursor.json; server Idempotency-Key deduplicates.
    fs::remove_file(&cursor_path).unwrap();
    let r_replay = Spooler::drain(&client, w.wal_path(), &cursor_path, 2, 5).unwrap();
    assert_eq!(r_replay.drained, 3);
    let by_uri = state.evidence_by_uri.lock().unwrap();
    assert_eq!(by_uri.len(), 3);
    for vers in by_uri.values() {
        assert_eq!(
            vers.len(),
            1,
            "Idempotency-Key must prevent duplicate versions"
        );
    }
}

#[test]
fn cloud_lease_fencing_reclaim_and_release() {
    let tmp = TempDir::new("lease");
    let (base, _state) = spawn_full_mock_cloud(0);
    let client_a = test_client(&base);
    let client_b = test_client(&base);

    let mut mgr_a = CloudLeaseManager::new(
        Some(client_a),
        "worker-a",
        "51701f9a672c",
        Some(&tmp.path().join("a")),
    );
    let mut mgr_b = CloudLeaseManager::new(
        Some(client_b),
        "worker-b",
        "51701f9a672c",
        Some(&tmp.path().join("b")),
    );
    mgr_a.set_clock(1_700_000_000);
    mgr_b.set_clock(1_700_000_000);

    let out_a = mgr_a.acquire_lease("ml-camp", "opp-bitcos", 60).unwrap();
    assert_eq!(out_a.status, "acquired");
    let lease_a = out_a.lease.unwrap();
    assert_eq!(lease_a.fencing_token, 1);

    // Concurrent worker B sees Busy while lease is active.
    let out_b_busy = mgr_b.acquire_lease("ml-camp", "opp-bitcos", 60).unwrap();
    assert_eq!(out_b_busy.status, "busy");
    assert_eq!(out_b_busy.holder_worker_id, "worker-a");
    assert!(out_b_busy.lease.is_none());

    // Worker A renews heartbeat so fencing token advances to 2.
    mgr_a.set_clock(1_700_000_030);
    let lease_a2 = mgr_a.renew_lease(&lease_a, 60).unwrap();
    assert_eq!(lease_a2.fencing_token, 2);
    mgr_a.verify_fencing(&lease_a2).unwrap();

    // Worker A stalls past TTL + grace; Worker B reclaims stale lease.
    mgr_b.set_clock(1_700_000_100);
    let reclaimed = mgr_b.reclaim_stale("ml-camp", "opp-bitcos", 60).unwrap();
    assert_eq!(reclaimed.status, "acquired");
    let lease_b = reclaimed.lease.unwrap();
    assert_eq!(lease_b.fencing_token, 3);
    assert_eq!(lease_b.reclaimed_from, "worker-a");

    // Stale Worker A is now fenced out.
    assert!(mgr_a.verify_fencing(&lease_a2).is_err());
    assert!(mgr_a.renew_lease(&lease_a2, 60).is_err());

    // Worker B releases immediately on completion; Worker A acquires cleanly at token 5.
    let rel = mgr_b.release_lease(&lease_b, "completed").unwrap().unwrap();
    assert_eq!(rel.version, 4);
    mgr_a.set_clock(1_700_000_101);
    let out_a_again = mgr_a.acquire_lease("ml-camp", "opp-bitcos", 60).unwrap();
    assert_eq!(out_a_again.status, "acquired");
    assert_eq!(out_a_again.lease.unwrap().fencing_token, 5);
}

#[test]
fn hip1403_types_dedup_key_and_4axis_transfer() {
    let hyp = HypothesisRecord {
        statement: "2-bit packed BITCOS beats scalar Q4_K by >= 1.25x".into(),
        null_hypothesis: "ratio_vs_q4k <= 1.0".into(),
        gate_ref: "gate:throughput_parity_v1".into(),
        dataset_refs: vec!["sha256:aaaa".into()],
        budget: ResearchBudget {
            max_compute_seconds: 600.0,
            max_cost_usd: 2.0,
        },
        proposer: "did:hanzo:agent:ml".into(),
        registered_at: "2026-11-22T00:00:00Z".into(),
        ..HypothesisRecord::default()
    }
    .with_computed_id()
    .unwrap();
    hyp.validate().unwrap();
    assert!(hyp.hypothesis_id.starts_with("sha256:"));

    let ce = CounterEvidence {
        counter_evidence_id: "ce-1".into(),
        claim_id: "claim-9".into(),
        experiment_id: "exp-9".into(),
        defect_kind: "holdout_leakage".into(),
        details: "calibration overlapped test split".into(),
        gate_ref: "gate:leakage_v1".into(),
        submitted_by: "did:hanzo:auditor".into(),
        submitted_at: "2026-11-22T01:00:00Z".into(),
    };
    assert_eq!(ce.digest().unwrap().len(), 64);

    let y_pred = research_yield(&[0.20, 0.0, 0.10], &[1.0, 2.0, 1.0]).unwrap();
    assert!((y_pred - 0.075).abs() < 1e-9);
    let y_succ = research_yield(&[0.30, 0.15], &[1.0, 1.0]).unwrap();
    let lift = successor_research_lift(y_succ, y_pred).unwrap();
    assert!((lift - 3.0).abs() < 1e-9);

    let stat = paired_comparison(
        "throughput",
        "tok/s",
        &[100.0, 101.0, 99.0],
        &[145.0, 146.0, 144.0],
        true,
    )
    .unwrap();
    let win_exp = ExperimentRecordV2 {
        project: "default".into(),
        study: "bitcos-study".into(),
        objective: "Accelerate ternary GEMV".into(),
        hypothesis: hyp.statement.clone(),
        hypothesis_id: hyp.hypothesis_id.clone(),
        null_hypothesis: hyp.null_hypothesis.clone(),
        gate_ref: hyp.gate_ref.clone(),
        experiment_id: "ml:bitcos:bonsai2-2b".into(),
        repo: "hanzoai/ml".into(),
        commit: "51701f9a672c84d5220ae9a7d5576926d8907822".into(),
        hardware: HardwareSpec {
            host: "dgx".into(),
            architecture: "aarch64".into(),
            gpu: "NVIDIA GB10".into(),
            backend: "CUDA-GB10-sm121a".into(),
            precision: "TERNARY".into(),
            ..HardwareSpec::default()
        },
        hardware_class: "CUDA-GB10-sm121a".into(),
        config: BTreeMap::from([
            ("model".into(), json!("Bonsai-2-2B")),
            ("workload_regime".into(), json!("single-turn-decode")),
        ]),
        workload_manifest: WorkloadManifest {
            manifest_id: "wl-bitcos".into(),
            dataset: "bonsai2-shapes".into(),
            split: "bench".into(),
            shapes: vec!["4096x4096".into()],
            input_lengths: vec![512],
            batch_size: 1,
            seed: 17,
            manifest_sha256: "c".repeat(64),
        },
        statistical_comparison: stat.clone(),
        verdict: epistemic_verdict(&stat.verdict).into(),
        correctness: CorrectnessChecks {
            parity_passed: true,
            memory_integrity_passed: true,
            ..CorrectnessChecks::default()
        },
        raw_samples: BTreeMap::from([
            ("baseline".into(), vec![100.0, 101.0, 99.0]),
            ("candidate".into(), vec![145.0, 146.0, 144.0]),
        ]),
        conclusion: "1.45x verified".into(),
        ..ExperimentRecordV2::default()
    };
    let dk = win_exp.dedup_key().unwrap();
    let expected_dk = rfc8785_digest(&json!({
        "project": "default",
        "hypothesis_id": win_exp.hypothesis_id,
        "config_digest": rfc8785_digest(&win_exp.config).unwrap(),
        "commit": win_exp.commit,
        "dataset_digest": win_exp.workload_manifest.manifest_sha256,
        "hardware_class": win_exp.hardware_class,
    }))
    .unwrap();
    assert_eq!(dk, expected_dk);
    assert_eq!(epistemic_verdict(&win_exp.verdict), "proven");

    let opps = generate_transfer_hypotheses(&win_exp, Some(TransferCatalog::default()));
    assert!(opps
        .iter()
        .any(|o| o.trigger_rule.starts_with("transfer:model_scale:")));
    assert!(opps
        .iter()
        .any(|o| o.trigger_rule.starts_with("transfer:seq_len:")));
    assert!(opps
        .iter()
        .any(|o| o.trigger_rule.starts_with("transfer:backend:")));
    assert!(opps
        .iter()
        .any(|o| o.trigger_rule.starts_with("transfer:workload_regime:")));

    // Refuted experiment emits targeted mechanism-fix instead of blind 4-axis transfer.
    let mut loss_exp = win_exp.clone();
    loss_exp.experiment_id = "ml:bitcos:bad-tile".into();
    loss_exp.statistical_comparison = paired_comparison(
        "throughput",
        "tok/s",
        &[100.0, 101.0, 99.0],
        &[80.0, 81.0, 79.0],
        true,
    )
    .unwrap();
    loss_exp.verdict = "refuted".into();
    loss_exp.failure_diagnosis = "shared memory bank conflict at stride 32".into();
    loss_exp.next_hypothesis = "Pad shared tile by +4 bytes to eliminate bank conflict".into();
    let fix_opps = generate_transfer_hypotheses(&loss_exp, None);
    assert_eq!(fix_opps.len(), 1);
    assert!(fix_opps[0]
        .trigger_rule
        .starts_with("mechanism_fix_after_refutation:"));
}

#[test]
fn ml_bench_lineage_router_and_progressive_ab_checkpoint_resume() {
    let manifest_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../bench_lineage.json");
    let lineage: BTreeMap<String, LineageRule> =
        serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
    let all_tasks: Vec<String> = lineage.keys().cloned().collect();

    // Editing bitcos.rs scopes ONLY bitcos and pq2_ptq1, NEVER flash_attn or gdn.
    let report = impacted_benchmarks(
        &all_tasks,
        &["hanzo-ml/src/quantized/bitcos.rs".to_string()],
        Some(&lineage),
        &[],
    )
    .unwrap();
    assert_eq!(
        report.tasks,
        vec!["bitcos".to_string(), "pq2_ptq1".to_string()]
    );
    assert!(!report.tasks.contains(&"flash_attn".to_string()));
    assert!(!report.tasks.contains(&"gdn".to_string()));

    // Early futility stop at Look 1 when candidate regresses.
    let runner = ProgressiveAbRunner::new(ProgressiveAbPlan::default());
    let mut calls = 0usize;
    let fut = runner
        .run("throughput", "tok/s", |idx, st| {
            calls += 1;
            Ok(ProgressiveStageOutput {
                stage_name: st.name.clone(),
                baseline_samples: vec![120.0, 121.0, 119.0],
                candidate_samples: vec![85.0, 84.0, 86.0],
                parity_passed: true,
                memory_integrity_passed: true,
                details: format!("stage {idx}"),
                ..ProgressiveStageOutput::default()
            })
        })
        .unwrap();
    assert_eq!(calls, 1);
    assert_eq!(fut.stopped_stage, 1);
    assert!(fut.early_stopped);
    assert_eq!(fut.verdict, "refuted");

    // Checkpoint resume after simulated kill -9 right after Look 1.
    let tmp = TempDir::new("ckpt");
    let ckpt = tmp.path().join("look_ckpt.json");
    let stage1_calls = AtomicUsize::new(0);
    let stage2_calls = AtomicUsize::new(0);
    let stage3_calls = AtomicUsize::new(0);

    let crash_res = runner.run_with_checkpoint("throughput", "tok/s", Some(&ckpt), |idx, st| {
        if idx == 1 {
            stage1_calls.fetch_add(1, Ordering::SeqCst);
            return Ok(ProgressiveStageOutput {
                stage_name: st.name.clone(),
                baseline_samples: vec![100.0, 100.5, 99.5, 100.2],
                candidate_samples: vec![150.0, 151.0, 149.5, 150.5],
                parity_passed: true,
                memory_integrity_passed: true,
                ..ProgressiveStageOutput::default()
            });
        }
        Err(hanzo_research::Error::Message("simulated kill -9".into()))
    });
    assert!(crash_res.is_err());
    assert!(ckpt.is_file());
    assert_eq!(stage1_calls.load(Ordering::SeqCst), 1);

    // Reclaiming worker resumes from Look 2 without re-running Look 1.
    let resumed = runner
        .run_with_checkpoint("throughput", "tok/s", Some(&ckpt), |idx, st| {
            match idx {
                1 => stage1_calls.fetch_add(1, Ordering::SeqCst),
                2 => stage2_calls.fetch_add(1, Ordering::SeqCst),
                _ => stage3_calls.fetch_add(1, Ordering::SeqCst),
            };
            Ok(ProgressiveStageOutput {
                stage_name: st.name.clone(),
                baseline_samples: vec![100.0, 100.2, 99.8, 100.1, 99.9],
                candidate_samples: vec![152.0, 151.8, 152.2, 152.1, 151.9],
                parity_passed: true,
                memory_integrity_passed: true,
                ..ProgressiveStageOutput::default()
            })
        })
        .unwrap();
    assert_eq!(
        stage1_calls.load(Ordering::SeqCst),
        1,
        "Look 1 must NOT re-execute"
    );
    assert_eq!(stage2_calls.load(Ordering::SeqCst), 1);
    assert_eq!(stage3_calls.load(Ordering::SeqCst), 1);
    assert_eq!(resumed.verdict, "proven");
    assert_eq!(resumed.stopped_stage, 3);
}

#[test]
fn autonomous_controller_tick_cloud_sync_and_cli_conformance() {
    let tmp = TempDir::new("ctrl");
    let ledger_dir = tmp.path().join("ledger");
    let (base, cloud_state) = spawn_full_mock_cloud(0);
    let client = test_client(&base);

    // Seed a GB10 32K decode record + raw artifact and verify cloud sync + query.
    let ledger = LocalResearchLedger::open(&ledger_dir, "gb10-test").unwrap();
    let mut gb10_exp = ExperimentRecordV2 {
        project: "default".into(),
        study: "gb10-32k-decode".into(),
        objective: "32K context single-stream decode on NVIDIA GB10".into(),
        hypothesis:
            "Split-KV + PhasePolicy + HybridPrefix + ReplayGDN + DFlash2 accelerate 32K decode"
                .into(),
        experiment_id: "engine:gb10-32k:hybrid-stack".into(),
        repo: "hanzoai/engine".into(),
        commit: "8c5742a280000000000000000000000000000000".into(),
        hardware: HardwareSpec {
            host: "spark".into(),
            architecture: "aarch64".into(),
            gpu: "NVIDIA GB10".into(),
            backend: "CUDA".into(),
            precision: "Q4_K".into(),
            ..HardwareSpec::default()
        },
        baseline: BTreeMap::from([("mean".into(), json!(18.2))]),
        candidate: BTreeMap::from([("mean".into(), json!(44.6))]),
        workload_manifest: WorkloadManifest {
            manifest_id: "gb10-32k".into(),
            dataset: "longctx-32k".into(),
            split: "bench".into(),
            shapes: vec!["32768".into()],
            input_lengths: vec![32768],
            batch_size: 1,
            seed: 7,
            manifest_sha256: "d".repeat(64),
        },
        statistical_comparison: paired_comparison(
            "decode_tps",
            "tok/s",
            &[18.2, 18.1, 18.3],
            &[44.6, 44.8, 44.4],
            true,
        )
        .unwrap(),
        correctness: CorrectnessChecks {
            parity_passed: true,
            memory_integrity_passed: true,
            details: "Split-KV #75 + PhasePolicy + HybridPrefix + ReplayGDN + DFlash2 verified"
                .into(),
            ..CorrectnessChecks::default()
        },
        raw_samples: BTreeMap::from([
            ("baseline".into(), vec![18.2, 18.1, 18.3]),
            ("candidate".into(), vec![44.6, 44.8, 44.4]),
        ]),
        conclusion: "2.45x speedup on GB10 32K decode".into(),
        ..ExperimentRecordV2::default()
    };
    let trace_bytes = b"raw-nsys-profile-trace-bytes-gb10-32k";
    ledger
        .record_experiment(&mut gb10_exp, &[("nsys_summary_txt", trace_bytes)])
        .unwrap();

    let receipts = ledger.sync_to_cloud(&client).unwrap();
    assert_eq!(receipts.len(), 1);
    assert!(receipts[0].readback_verified);
    assert!(cloud_state.artifacts.lock().unwrap().len() >= 2);

    let hits = ledger
        .query_experiments(None, Some("proven"), None, Some("CUDA"), Some("32k"), true)
        .unwrap();
    assert_eq!(hits.len(), 1);
    assert!(hits[0].hypothesis.contains("Split-KV"));

    // Write a real executable stage script for AutonomousController::tick.
    let exec_path = tmp.path().join("mock_executor.py");
    let script = r#"#!/usr/bin/env python3
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("phase")
p.add_argument("--request", required=True)
p.add_argument("--output", required=True)
args = p.parse_args()
req = json.load(open(args.request, encoding="utf-8"))
if req["variant"] == "z_bad":
    out = {
        "status": "complete",
        "trial_id": req["trial_id"],
        "objective_id": req["objective_id"],
        "baseline_samples": [100.0, 101.0, 99.0, 100.0],
        "candidate_samples": [70.0, 71.0, 69.0, 70.0],
        "parity_passed": True,
        "memory_integrity_passed": True,
        "gpu_time_sec": 0.05,
        "details": "regressed tile"
    }
else:
    out = {
        "status": "complete",
        "trial_id": req["trial_id"],
        "objective_id": req["objective_id"],
        "baseline_samples": [100.0, 100.5, 99.5, 100.2, 99.8],
        "candidate_samples": [165.0, 166.0, 164.5, 165.2, 164.8],
        "parity_passed": True,
        "memory_integrity_passed": True,
        "gpu_time_sec": 0.08,
        "details": "winning tile"
    }
json.dump(out, open(args.output, "w", encoding="utf-8"))
"#;
    fs::write(&exec_path, script).unwrap();
    let mut perms = fs::metadata(&exec_path).unwrap().permissions();
    perms.set_mode(0o755);
    fs::set_permissions(&exec_path, perms).unwrap();

    let reg = CampaignRegistry {
        campaign: "test-auto-camp".into(),
        repo: "hanzoai/ml".into(),
        plan: ProgressiveAbPlan {
            stages: vec![
                ProgressiveStage {
                    name: "micro".into(),
                    reps: 4,
                    prompt_len: 64,
                    gen_len: 16,
                },
                ProgressiveStage {
                    name: "qualification".into(),
                    reps: 5,
                    prompt_len: 256,
                    gen_len: 64,
                },
            ],
            ..ProgressiveAbPlan::default()
        },
        objectives: vec![ObjectiveSpec {
            id: "bitcos".into(),
            study: "bitcos".into(),
            family: "quant".into(),
            metric: "throughput".into(),
            unit: "tok/s".into(),
            direction: "higher".into(),
            backends: vec!["CPU".into(), "CUDA".into()],
            shapes: vec!["4096x4096".into()],
            ..ObjectiveSpec::default()
        }],
        search_space: BTreeMap::from([(
            "quant".into(),
            vec![
                SearchArm {
                    arm: "lut".into(),
                    variant: "z_bad".into(),
                    hypothesis: "Test bad tile".into(),
                    config: BTreeMap::new(),
                },
                SearchArm {
                    arm: "lut".into(),
                    variant: "b_good".into(),
                    hypothesis: "Test good tile".into(),
                    config: BTreeMap::new(),
                },
            ],
        )]),
    };
    let camp_json = tmp.path().join("campaign.json");
    fs::write(&camp_json, serde_json::to_vec_pretty(&reg).unwrap()).unwrap();

    let ctrl_ledger_dir = tmp.path().join("ctrl_ledger");
    let ctrl = AutonomousController::open(
        &ctrl_ledger_dir,
        None,
        Some(reg),
        env!("CARGO_MANIFEST_DIR"),
        10,
        Some("worker-ctrl"),
    )
    .unwrap();

    // Tick 1 runs first unexplored arm ("lut:bad"), early-stops at stage 1 as refuted.
    let t1 = ctrl
        .tick(&exec_path, None, None, None, 5, true, Some(&client), 120)
        .unwrap();
    assert_eq!(t1["verdict"], "refuted");
    assert_eq!(t1["early_stopped"], true);
    assert_eq!(t1["stopped_stage"], 1);

    // Tick 2 skips refuted "lut:bad" and runs "lut:good", qualifying at stage 2 as proven.
    let t2 = ctrl
        .tick(&exec_path, None, None, None, 5, true, Some(&client), 120)
        .unwrap();
    assert_eq!(t2["verdict"], "proven");
    assert_eq!(t2["stopped_stage"], 2);
    assert!(!ctrl.ledger.prior_failures(None).unwrap().is_empty());
    assert!(!ctrl
        .ledger
        .recommended_experiments(None, 10)
        .unwrap()
        .is_empty());

    // Exercise the compiled hanzo-research CLI binary end-to-end.
    let bin = env!("CARGO_BIN_EXE_hanzo-research");
    let lineage_file = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../bench_lineage.json");
    let cli_impact = Command::new(bin)
        .args([
            "impact",
            "--lineage",
            lineage_file.to_str().unwrap(),
            "--changed-paths",
            "hanzo-ml/src/quantized/bitcos.rs",
        ])
        .output()
        .unwrap();
    assert!(
        cli_impact.status.success(),
        "impact stderr: {}",
        String::from_utf8_lossy(&cli_impact.stderr)
    );
    let impact_val: Value = serde_json::from_slice(&cli_impact.stdout).unwrap();
    assert_eq!(impact_val["tasks"], json!(["bitcos", "pq2_ptq1"]));

    let cli_failures = Command::new(bin)
        .args([
            "failures",
            "--ledger-dir",
            ctrl_ledger_dir.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(cli_failures.status.success());
    let fail_rows: Value = serde_json::from_slice(&cli_failures.stdout).unwrap();
    assert_eq!(fail_rows.as_array().unwrap().len(), 1);

    let cli_validate = Command::new(bin)
        .args([
            "controller",
            "validate",
            "--ledger-dir",
            ctrl_ledger_dir.to_str().unwrap(),
            "--campaign",
            camp_json.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(cli_validate.status.success());
}

#[test]
fn cloud_938_timestamps_tags_artifacts_and_gpu_job() {
    let (base, _state) = spawn_full_mock_cloud(0);
    let client = test_client(&base);

    let mut exp = client
        .experiment("kernel-perf", "bitcos", "gb10")
        .tags(["gb10", "ternary"])
        .hypothesis("2-bit LUT speeds up GEMV");
    exp.conclude(hanzo_research::Verdict::Proven, "1.6x faster", Some(1.6))
        .unwrap();

    let rows = client.query(None, None).unwrap();
    assert_eq!(rows.len(), 1);
    assert!(
        rows[0].ts > 1_700_000_000,
        "ts == 0 must default to current unix seconds"
    );
    assert_eq!(
        rows[0].tags,
        vec!["gb10".to_string(), "ternary".to_string()]
    );

    let payload_bytes = b"{\"tok_s\":160.0}";
    client
        .upload_artifact_with_meta(
            "run-938:report",
            "report",
            "run-938",
            "application/json",
            "rfc8785",
            "none",
            payload_bytes,
        )
        .unwrap();

    let arts = client.artifacts(Some("run-938"), None, None).unwrap();
    assert_eq!(arts.len(), 1);
    assert_eq!(arts[0].artifact_id, "run-938:report");
    assert_eq!(arts[0].media_type, "application/json");
    assert_eq!(arts[0].canonicalization, "rfc8785");
    assert_eq!(arts[0].uncompressed_size, payload_bytes.len() as u64);
    assert!(!arts[0].content_sha256.is_empty());
    assert!(arts[0].ts > 1_700_000_000);
    assert_eq!(arts[0].received_at, Some(1_700_000_999));

    // Remote GPU job submission + polling to completion.
    let sub = client
        .submit_gpu_job(
            &json!({"cmd": "cargo run --release --example bitcos_profile"}),
            "research.bench",
            Some("gb10"),
            Some("job-gb10-1"),
            None,
        )
        .unwrap();
    assert_eq!(sub["taskQueue"], "gpu:gb10");
    let done = client
        .wait_gpu_job(
            "job-gb10-1",
            None,
            std::time::Duration::from_secs(5),
            std::time::Duration::from_millis(10),
        )
        .unwrap();
    assert_eq!(done["state"], "ACTIVITY_TASK_STATE_COMPLETED");
    assert_eq!(done["output"]["tok_per_sec"], json!(182.4));
}

#[test]
fn test_native_tasks_linked_machines_evo_dgx_sse_schedules_and_cli() {
    let dir = TempDir::new("native_tasks_evo_dgx");
    let (base, mock_state) = spawn_full_mock_cloud(0);

    *mock_state.fleet_workers.lock().unwrap() = vec![
        json!({
            "id": "dgx",
            "hostname": "spark-6826",
            "status": "online",
            "os": "linux",
            "arch": "aarch64",
            "cpus": 20,
            "memory": 137_438_953_472_u64,
            "gpus": [{"name": "NVIDIA GB10", "memoryTotal": "131072 MiB", "arch": "sm_121"}],
            "capabilities": ["research.bench", "research.tick", "fn.run"],
            "jobQueue": "gpu:dgx",
            "busy": false,
        }),
        json!({
            "id": "macbook-hanzod",
            "hostname": "zach-mbp-hanzod",
            "status": "online",
            "os": "darwin",
            "arch": "arm64",
            "cpus": 16,
            "memory": 68_719_476_736_u64,
            "gpus": [{"name": "Apple M4 Max", "memoryTotal": "65536 MiB"}],
            "capabilities": ["research.bench", "hanzod.run"],
            "jobQueue": "gpu:macbook-hanzod",
        }),
    ];
    *mock_state.agent_targets.lock().unwrap() = vec![
        json!({
            "id": "tgt_dgx_1",
            "label": "dgx",
            "host": "spark-6826",
            "capacity": "1x GB10 / 128G",
            "status": "online",
        }),
        json!({
            "id": "tgt_evo_1",
            "label": "evo",
            "host": "evo",
            "capacity": "32 vCPU / 102G",
            "status": "offline",
            "spec": {
                "os": "linux",
                "arch": "x86_64",
                "cpus": 32,
                "memory": 109_521_666_048_u64,
                "gpus": [{"name": "AMD Ryzen AI Max+ 395 (gfx1151)", "memoryTotal": "102 GiB"}],
            }
        }),
    ];
    mock_state.fleet_presence.lock().unwrap().insert(
        "macbook-hanzod".to_string(),
        json!({"input": {"busy": true, "busyReason": "active_dev_session"}}),
    );
    mock_state
        .busy_defer_queues
        .lock()
        .unwrap()
        .push("gpu:macbook-hanzod".to_string());

    let client = test_client(&base);

    // 1. Linked-machine discovery merges fleet workers + agent targets (dgx, macbook-hanzod, evo)
    let online = client.list_machines(true, None, None, false).unwrap();
    let online_ids: Vec<&str> = online.iter().map(|m| m.id.as_str()).collect();
    assert_eq!(online_ids, vec!["dgx", "macbook-hanzod"]);

    let all = client.list_machines(false, None, None, false).unwrap();
    let evo = all.iter().find(|m| m.id == "evo").unwrap();
    assert_eq!(evo.status, "offline");
    assert_eq!(evo.task_queue, "gpu:evo");
    assert!(evo.backends.contains(&"ROCM".to_string()));
    assert!(evo.backends.contains(&"VULKAN".to_string()));

    let cuda_only = client
        .list_machines(true, None, Some("CUDA"), false)
        .unwrap();
    assert_eq!(cuda_only.len(), 1);
    assert_eq!(cuda_only[0].id, "dgx");

    let metal_only = client
        .list_machines(true, None, Some("METAL"), false)
        .unwrap();
    assert_eq!(metal_only.len(), 1);
    assert!(metal_only[0].busy);
    assert_eq!(metal_only[0].busy_reason, "active_dev_session");

    let idle_only = client.list_machines(true, None, None, true).unwrap();
    assert_eq!(idle_only.len(), 1);
    assert_eq!(idle_only[0].id, "dgx");

    // 2. Durable --target evo,dgx queuing (never rejects offline evo) + fanout + SSE wait + release
    let evo_dgx = client
        .orchestrate_across_machines(
            &[json!({"command": "hanzo-research tick", "job_id": "pair"})],
            &OrchestrateOptions {
                target: Some("evo,dgx".to_string()),
                ..OrchestrateOptions::default()
            },
        )
        .unwrap();
    assert_eq!(evo_dgx.len(), 2);
    assert_eq!(evo_dgx[0]["machine"], "evo");
    assert_eq!(evo_dgx[0]["task_queue"], "gpu:evo");
    assert_eq!(evo_dgx[0]["machine_status"], "queued_until_online");
    assert_eq!(evo_dgx[0]["scheduleToCloseTimeout"], "72h");
    assert_eq!(evo_dgx[1]["machine"], "dgx");
    assert_eq!(evo_dgx[1]["task_queue"], "gpu:dgx");
    assert_eq!(evo_dgx[1]["machine_status"], "online");

    let fanout_jobs = client
        .orchestrate_across_machines(
            &[json!({"command": "hanzo-bench --progressive", "job_id": "xarch"})],
            &OrchestrateOptions {
                fanout: true,
                ..OrchestrateOptions::default()
            },
        )
        .unwrap();
    assert_eq!(fanout_jobs.len(), 2);
    assert_eq!(fanout_jobs[0]["taskQueue"], "gpu:dgx");
    assert_eq!(fanout_jobs[1]["taskQueue"], "gpu:macbook-hanzod");

    let rel = client
        .release_gpu_job(
            "xarch-macbook-hanzod",
            Some("xarch-macbook-hanzod"),
            "active_dev_session",
            "macbook-hanzod",
            5,
            None,
        )
        .unwrap();
    assert_eq!(rel["status"], "ACTIVITY_TASK_STATE_SCHEDULED");
    assert_eq!(rel["deferCount"], 1);

    let sse_before = mock_state.sse_streams_opened.load(Ordering::SeqCst);
    let done_mac = client
        .wait_gpu_job(
            "xarch-macbook-hanzod",
            None,
            std::time::Duration::from_secs(5),
            std::time::Duration::from_secs(2),
        )
        .unwrap();
    assert!(mock_state.sse_streams_opened.load(Ordering::SeqCst) > sse_before);
    assert_eq!(done_mac["status"], "ACTIVITY_TASK_STATE_COMPLETED");
    assert!(done_mac["deferCount"].as_u64().unwrap_or(0) >= 1);

    // 3. Native hanzoai/tasks schedules CRUD (including multi-target evo,dgx)
    let multi_sched = client
        .create_schedule(
            "ml-nightly",
            &ScheduleOptions {
                interval: Some("30m".to_string()),
                cron: Some("0 */2 * * *".to_string()),
                target: Some("evo,dgx".to_string()),
                campaign: Some("campaigns/ml_kernels_and_quant.json".to_string()),
                executor: Some("scripts/ml_research_executor.py".to_string()),
                max_compute_seconds: Some(900),
                ..ScheduleOptions::default()
            },
        )
        .unwrap();
    assert_eq!(
        multi_sched["scheduleIds"],
        json!(["ml-nightly-evo", "ml-nightly-dgx"])
    );
    let listed = client.list_schedules(None).unwrap();
    assert_eq!(listed.len(), 2);
    let desc_evo = client.describe_schedule("ml-nightly-evo", None).unwrap();
    assert_eq!(desc_evo["action"]["taskQueue"], "gpu:evo");
    assert_eq!(desc_evo["action"]["scheduleToCloseTimeout"], "72h");

    let trig = client
        .trigger_schedule("ml-nightly-dgx", Some("manual-1"), None)
        .unwrap();
    assert_eq!(trig["status"], "triggered");
    assert_eq!(
        client
            .pause_schedule("ml-nightly-dgx", "maint", None)
            .unwrap()["status"],
        "paused"
    );
    assert_eq!(
        client.describe_schedule("ml-nightly-dgx", None).unwrap()["state"]["paused"],
        true
    );
    assert_eq!(
        client.resume_schedule("ml-nightly-dgx", None).unwrap()["status"],
        "running"
    );
    assert_eq!(
        client.delete_schedule("ml-nightly-evo", None).unwrap()["status"],
        "deleted"
    );
    assert_eq!(
        client.delete_schedule("ml-nightly-dgx", None).unwrap()["status"],
        "deleted"
    );
    assert!(client.list_schedules(None).unwrap().is_empty());

    // 4. Rust CLI binary (`machines`, `dispatch --target evo,dgx`, `schedule`) + remote tick_with_tasks
    let bin = env!("CARGO_BIN_EXE_hanzo-research");
    let out_m = Command::new(bin)
        .env("HANZO_API_KEY", "hk-test-key")
        .args(["--base", &base, "machines", "--include-offline"])
        .output()
        .unwrap();
    assert!(out_m.status.success());
    let parsed_m: Value = serde_json::from_slice(&out_m.stdout).unwrap();
    assert_eq!(parsed_m.as_array().unwrap().len(), 3);

    let out_d = Command::new(bin)
        .env("HANZO_API_KEY", "hk-test-key")
        .args([
            "--base",
            &base,
            "dispatch",
            "--target",
            "evo,dgx",
            "--command",
            "hanzo-bench --progressive",
        ])
        .output()
        .unwrap();
    assert!(out_d.status.success());
    let parsed_d: Value = serde_json::from_slice(&out_d.stdout).unwrap();
    assert_eq!(parsed_d.as_array().unwrap().len(), 2);

    mock_state.stage_mode.store(1, Ordering::SeqCst);
    let mut space = BTreeMap::new();
    space.insert(
        "cuda_kern".to_string(),
        vec![SearchArm {
            arm: "marlin".into(),
            variant: "sm121_v2".into(),
            hypothesis: "SM121 warp tile".into(),
            config: BTreeMap::new(),
        }],
    );
    let reg = CampaignRegistry {
        campaign: "remote-dgx-rust".into(),
        repo: "hanzoai/ml".into(),
        objectives: vec![ObjectiveSpec {
            id: "qmm_cuda".into(),
            study: "qmm_cuda".into(),
            family: "cuda_kern".into(),
            metric: "tok_per_sec".into(),
            unit: "tok/s".into(),
            direction: "higher".into(),
            backends: vec!["CUDA".into()],
            shapes: vec!["4096x4096".into()],
            ..ObjectiveSpec::default()
        }],
        search_space: space,
        plan: ProgressiveAbPlan {
            stages: vec![
                ProgressiveStage {
                    name: "micro".into(),
                    reps: 4,
                    prompt_len: 64,
                    gen_len: 16,
                },
                ProgressiveStage {
                    name: "qualification".into(),
                    reps: 4,
                    prompt_len: 128,
                    gen_len: 32,
                },
            ],
            ..ProgressiveAbPlan::default()
        },
    };
    let ctrl = AutonomousController::open(
        dir.path().join("remote_ledger"),
        None,
        Some(reg),
        ".",
        16,
        Some("rust-cpu-orchestrator"),
    )
    .unwrap();
    let r_tick = ctrl
        .tick_with_tasks(
            std::path::Path::new("hanzo-bench"),
            None,
            None,
            None,
            10,
            true,
            Some(&client),
            600,
            true,
            Some("dgx"),
        )
        .unwrap();
    assert_eq!(r_tick["verdict"], "proven");
    assert_eq!(r_tick["stopped_stage"], 2);
}
