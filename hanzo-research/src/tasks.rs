//! Native `hanzoai/tasks` scheduling and linked-machine (`hanzod` / `hanzo link`) orchestration.

use std::collections::{BTreeMap, BTreeSet};
use std::io::{BufRead, BufReader};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::canonical::rfc8785_canonicalize;
use crate::{encode, sha256_hex, unix_now, Error, Research, Result};

/// Default retry policy for busy-aware standalone activities on `hanzoai/tasks`.
pub fn default_retry_policy() -> Value {
    json!({
        "initialInterval": "5s",
        "backoffCoefficient": 2.0,
        "maximumInterval": "60s",
        "maximumAttempts": 8,
    })
}

/// A linked compute or Desktop (`hanzod` / `hanzo link` / `/v1/agent/targets`) machine.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct LinkedMachine {
    pub id: String,
    pub hostname: String,
    pub status: String,
    pub os: String,
    pub arch: String,
    pub cpus: u32,
    pub memory_bytes: u64,
    pub memory_gb: f64,
    pub capacity: String,
    pub gpus: Vec<Value>,
    pub capabilities: Vec<String>,
    pub backends: Vec<String>,
    pub job_queue: String,
    pub task_queue: String,
    pub busy: bool,
    pub busy_reason: String,
}

/// Options for dispatching a durable standalone activity via `hanzoai/tasks`.
#[derive(Debug, Clone)]
pub struct TaskDispatchOptions {
    pub target: Option<String>,
    pub activity_type: String,
    pub when_idle_only: bool,
    pub required_backend: Option<String>,
    pub min_memory_gb: Option<f64>,
    pub job_id: Option<String>,
    pub run_id: Option<String>,
    pub namespace: String,
    pub retry_policy: Option<Value>,
    pub heartbeat_timeout: String,
    pub start_to_close_timeout: String,
    pub schedule_to_close_timeout: Option<String>,
    pub request_id: Option<String>,
    pub identity: String,
    pub machine_status_hint: Option<String>,
}

impl Default for TaskDispatchOptions {
    fn default() -> Self {
        Self {
            target: None,
            activity_type: "research.bench".to_string(),
            when_idle_only: true,
            required_backend: None,
            min_memory_gb: None,
            job_id: None,
            run_id: None,
            namespace: "gpu-jobs".to_string(),
            retry_policy: None,
            heartbeat_timeout: "60s".to_string(),
            start_to_close_timeout: "1800s".to_string(),
            schedule_to_close_timeout: None,
            request_id: None,
            identity: String::new(),
            machine_status_hint: None,
        }
    }
}

/// Options for orchestrating tasks across linked machines (`target`, `fanout`, or best-idle).
#[derive(Debug, Clone)]
pub struct OrchestrateOptions {
    pub target: Option<String>,
    pub capability: Option<String>,
    pub backend: Option<String>,
    pub activity_type: String,
    pub fanout: bool,
    pub wait: bool,
    pub when_idle_only: bool,
    pub min_memory_gb: Option<f64>,
    pub namespace: String,
    pub timeout: Duration,
    pub poll_interval: Duration,
}

impl Default for OrchestrateOptions {
    fn default() -> Self {
        Self {
            target: None,
            capability: Some("research.bench".to_string()),
            backend: None,
            activity_type: "research.bench".to_string(),
            fanout: false,
            wait: false,
            when_idle_only: true,
            min_memory_gb: None,
            namespace: "gpu-jobs".to_string(),
            timeout: Duration::from_secs(600),
            poll_interval: Duration::from_secs(2),
        }
    }
}

/// Options for creating recurring or cron schedules on `hanzoai/tasks`.
#[derive(Debug, Clone)]
pub struct ScheduleOptions {
    pub interval: Option<String>,
    pub cron: Option<String>,
    pub target: Option<String>,
    pub activity_type: String,
    pub workflow_type: Option<String>,
    pub input_payload: Option<Value>,
    pub command: Option<String>,
    pub campaign: Option<String>,
    pub executor: Option<String>,
    pub max_compute_seconds: Option<u64>,
    pub when_idle_only: bool,
    pub required_backend: Option<String>,
    pub paused: bool,
    pub note: String,
    pub namespace: String,
    pub retry_policy: Option<Value>,
    pub heartbeat_timeout: String,
    pub start_to_close_timeout: String,
}

impl Default for ScheduleOptions {
    fn default() -> Self {
        Self {
            interval: None,
            cron: None,
            target: None,
            activity_type: "research.tick".to_string(),
            workflow_type: None,
            input_payload: None,
            command: None,
            campaign: None,
            executor: None,
            max_compute_seconds: None,
            when_idle_only: true,
            required_backend: None,
            paused: false,
            note: String::new(),
            namespace: "gpu-jobs".to_string(),
            retry_policy: None,
            heartbeat_timeout: "60s".to_string(),
            start_to_close_timeout: "1800s".to_string(),
        }
    }
}

/// Parse single or comma-separated target names (`"evo,dgx"` -> `["evo", "dgx"]`).
pub fn parse_targets(raw: Option<&str>) -> Vec<String> {
    raw.unwrap_or("")
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect()
}

/// Map `None` or `"any"` to shared `"gpu-jobs"`, and `<machine_id>` to `"gpu:<machine_id>"`.
pub fn resolve_task_queue(target: Option<&str>) -> String {
    let cleaned = target.unwrap_or("").trim();
    if cleaned.is_empty() || cleaned.eq_ignore_ascii_case("any") || cleaned == "gpu-jobs" {
        "gpu-jobs".to_string()
    } else if cleaned.starts_with("gpu:") {
        cleaned.to_string()
    } else {
        format!("gpu:{cleaned}")
    }
}

fn is_online_status(status: &str) -> bool {
    matches!(
        status.trim().to_ascii_lowercase().as_str(),
        "online" | "ready" | "idle" | "busy" | "running" | "active"
    )
}

fn is_terminal_activity_state(state: &str) -> bool {
    matches!(
        state,
        "ACTIVITY_TASK_STATE_COMPLETED"
            | "ACTIVITY_TASK_STATE_FAILED"
            | "ACTIVITY_TASK_STATE_CANCELED"
            | "ACTIVITY_TASK_STATE_TIMED_OUT"
    )
}

fn parse_mem_bytes(val: &Value) -> u64 {
    if let Some(u) = val.as_u64() {
        return u;
    }
    if let Some(f) = val.as_f64() {
        return if f > 0.0 { f as u64 } else { 0 };
    }
    let Some(s) = val.as_str() else {
        return 0;
    };
    let trimmed = s.trim();
    let split_idx = trimmed
        .find(|c: char| !c.is_ascii_digit() && c != '.')
        .unwrap_or(trimmed.len());
    let Ok(num) = trimmed[..split_idx].trim().parse::<f64>() else {
        return 0;
    };
    let unit = trimmed[split_idx..].trim().to_ascii_lowercase();
    let mult: f64 = match unit.as_str() {
        "tib" | "tb" => (1024_u64.pow(4)) as f64,
        "gib" | "gb" | "g" => (1024_u64.pow(3)) as f64,
        "mib" | "mb" | "m" => (1024_u64.pow(2)) as f64,
        "kib" | "kb" | "k" => 1024.0,
        _ => 1.0,
    };
    (num * mult) as u64
}

fn parse_capacity_hint(cap: &str) -> (u32, u64) {
    let mut cpus = 0u32;
    let mut mem = 0u64;
    for part in cap.split(['/', ',', '·']) {
        let low = part.trim().to_ascii_lowercase();
        if low.contains("cpu") || low.contains("core") {
            let digits: String = low.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(n) = digits.parse::<u32>() {
                cpus = n;
            }
        } else if low.ends_with('g')
            || low.ends_with("gb")
            || low.ends_with("gib")
            || low.ends_with("mib")
        {
            mem = parse_mem_bytes(&Value::String(low));
        }
    }
    (cpus, mem)
}

/// Infer execution backends (`CUDA`, `ROCM`, `METAL`, `VULKAN`, `CPU`) from worker/target metadata.
pub fn infer_machine_backends(row: &Value) -> Vec<String> {
    let mut found: BTreeSet<String> = BTreeSet::new();
    found.insert("CPU".to_string());

    let os_name = row["os"]
        .as_str()
        .or_else(|| row["spec"]["os"].as_str())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if matches!(os_name.as_str(), "darwin" | "macos" | "ios") {
        found.insert("METAL".to_string());
    }
    if let Some(arr) = row["backends"].as_array() {
        for b in arr.iter().filter_map(Value::as_str) {
            let up = b.trim().to_ascii_uppercase();
            if !up.is_empty() {
                found.insert(up);
            }
        }
    }
    let mut hint_parts = vec![
        row["capacity"].as_str().unwrap_or("").to_string(),
        row["label"].as_str().unwrap_or("").to_string(),
        row["cpuModel"].as_str().unwrap_or("").to_string(),
    ];
    if let Some(caps) = row["capabilities"].as_array() {
        for c in caps.iter().filter_map(Value::as_str) {
            hint_parts.push(c.to_string());
        }
    }
    let hint = hint_parts.join(" ").to_ascii_lowercase();
    if [
        "cuda", "nvidia", "gb10", "gh200", "h100", "h200", "a100", "b200", "rtx",
    ]
    .iter()
    .any(|k| hint.contains(k))
    {
        found.insert("CUDA".to_string());
    }
    if hint.contains("metal")
        || [
            "apple m1", "apple m2", "apple m3", "apple m4", "m1 max", "m4 max",
        ]
        .iter()
        .any(|k| hint.contains(k))
    {
        found.insert("METAL".to_string());
    }
    if [
        "rocm",
        "hip",
        "strix",
        "gfx11",
        "radeon",
        "instinct",
        "ryzen ai max",
    ]
    .iter()
    .any(|k| hint.contains(k))
    {
        found.insert("ROCM".to_string());
    }
    if ["vulkan", "strix", "gfx11", "radeon", "ryzen ai max"]
        .iter()
        .any(|k| hint.contains(k))
    {
        found.insert("VULKAN".to_string());
    }

    let gpu_arrays = [row["gpus"].as_array(), row["spec"]["gpus"].as_array()];
    for arr in gpu_arrays.into_iter().flatten() {
        for g in arr {
            let blob = format!(
                "{} {} {}",
                g["name"].as_str().unwrap_or(""),
                g["arch"].as_str().unwrap_or(""),
                g["vendor"].as_str().unwrap_or("")
            )
            .to_ascii_lowercase();
            if [
                "nvidia", "cuda", "gb10", "gh200", "h100", "h200", "a100", "b200", "rtx",
                "geforce", "tesla", "sm_",
            ]
            .iter()
            .any(|k| blob.contains(k))
            {
                found.insert("CUDA".to_string());
            }
            if ["apple", "metal", "m1", "m2", "m3", "m4"]
                .iter()
                .any(|k| blob.contains(k))
            {
                found.insert("METAL".to_string());
            }
            if [
                "amd", "radeon", "instinct", "rocm", "gfx", "mi2", "mi3", "strix",
            ]
            .iter()
            .any(|k| blob.contains(k))
            {
                found.insert("ROCM".to_string());
            }
            if ["vulkan", "radeon", "gfx", "strix"]
                .iter()
                .any(|k| blob.contains(k))
            {
                found.insert("VULKAN".to_string());
            }
        }
    }

    let order = ["CUDA", "ROCM", "METAL", "VULKAN", "CPU"];
    let mut out: Vec<String> = order
        .iter()
        .filter(|b| found.contains(**b))
        .map(|b| (*b).to_string())
        .collect();
    for extra in found {
        if !order.contains(&extra.as_str()) {
            out.push(extra);
        }
    }
    out
}

fn extract_busy(v: &Value) -> (Option<bool>, String) {
    let reason = v["busyReason"]
        .as_str()
        .or_else(|| v["busy_reason"].as_str())
        .or_else(|| v["lastBusyReason"].as_str())
        .unwrap_or("")
        .trim()
        .to_string();
    if let Some(b) = v["busy"].as_bool() {
        return (Some(b), reason);
    }
    if let Some(run_cnt) = v["running"].as_u64().filter(|&n| n > 0) {
        let r = if reason.is_empty() {
            format!("active_sessions:{run_cnt}")
        } else {
            reason
        };
        return (Some(true), r);
    }
    let st = v["status"]
        .as_str()
        .or_else(|| v["state"].as_str())
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if st == "busy" {
        let r = if reason.is_empty() {
            "busy".to_string()
        } else {
            reason
        };
        return (Some(true), r);
    }
    if !reason.is_empty() {
        return (Some(true), reason);
    }
    (None, String::new())
}

fn canonical_target_handle(item: &Value) -> String {
    let label = item["label"].as_str().unwrap_or("").trim();
    let host = item["host"]
        .as_str()
        .or_else(|| item["hostname"].as_str())
        .unwrap_or("")
        .trim();
    let raw_id = item["id"].as_str().unwrap_or("").trim();
    if !label.is_empty()
        && label
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
    {
        return label.to_string();
    }
    if !host.is_empty() {
        return host.to_string();
    }
    if !raw_id.is_empty() {
        return raw_id.to_string();
    }
    label.to_string()
}

fn consume_sse_until_terminal(
    reader: &mut BufReader<Box<dyn std::io::Read + Send + Sync + 'static>>,
    ns: &str,
    job_id: &str,
    run_id: &str,
    mut on_defer: impl FnMut(u64),
) -> Option<String> {
    let mut event_kind = String::new();
    let mut data_lines: Vec<String> = Vec::new();
    let mut line = String::new();

    loop {
        line.clear();
        let Ok(n) = reader.read_line(&mut line) else {
            return None;
        };
        if n == 0 {
            return None;
        }
        let trimmed = line.trim_end_matches(['\r', '\n']);
        if trimmed.starts_with(':') {
            continue;
        }
        if trimmed.is_empty() {
            if !data_lines.is_empty() {
                let payload_txt = data_lines.join("\n");
                data_lines.clear();
                if let Ok(ev) = serde_json::from_str::<Value>(&payload_txt) {
                    let kind = ev["kind"]
                        .as_str()
                        .filter(|s| !s.is_empty())
                        .unwrap_or(&event_kind)
                        .to_string();
                    let ev_ns = ev["namespace"].as_str().unwrap_or("");
                    let ev_wf = ev["workflow_id"]
                        .as_str()
                        .or_else(|| ev["activityId"].as_str())
                        .or_else(|| ev["id"].as_str())
                        .unwrap_or("");
                    let ev_run = ev["run_id"]
                        .as_str()
                        .or_else(|| ev["runId"].as_str())
                        .unwrap_or("");
                    if (ev_ns.is_empty() || ev_ns == ns)
                        && ev_wf == job_id
                        && (ev_run.is_empty() || ev_run == run_id)
                    {
                        if kind == "activity.deferred" {
                            let dc = ev["data"]["deferCount"].as_u64().unwrap_or(1);
                            on_defer(dc);
                        } else if matches!(
                            kind.as_str(),
                            "activity.completed"
                                | "activity.failed"
                                | "activity.canceled"
                                | "activity.timedout"
                        ) {
                            return Some(kind);
                        }
                    }
                }
            }
            event_kind.clear();
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("event:") {
            event_kind = rest.trim().to_string();
        } else if let Some(rest) = trimmed.strip_prefix("data:") {
            data_lines.push(rest.trim_start().to_string());
        }
    }
}

impl Research {
    /// Discover linked machines by merging `/v1/compute/fleet/workers` and `/v1/agent/targets` (`evo`, `dgx`, etc.).
    pub fn list_machines(
        &self,
        only_online: bool,
        capability: Option<&str>,
        backend: Option<&str>,
        exclude_busy: bool,
    ) -> Result<Vec<LinkedMachine>> {
        let fleet_rows: Vec<Value> = match self.get::<Value>("/v1/compute/fleet/workers") {
            Ok(resp) => resp["workers"]
                .as_array()
                .or_else(|| resp["data"].as_array())
                .or_else(|| resp.as_array())
                .cloned()
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        };
        let target_rows: Vec<Value> = match self.get::<Value>("/v1/agent/targets") {
            Ok(resp) => resp["targets"]
                .as_array()
                .or_else(|| resp["data"].as_array())
                .or_else(|| resp.as_array())
                .cloned()
                .unwrap_or_default(),
            Err(_) => Vec::new(),
        };

        let mut order_keys: Vec<String> = Vec::new();
        let mut by_key: BTreeMap<String, LinkedMachine> = BTreeMap::new();
        let mut alias_to_key: BTreeMap<String, String> = BTreeMap::new();

        for item in fleet_rows {
            let mid = item["id"]
                .as_str()
                .or_else(|| item["hostname"].as_str())
                .unwrap_or("")
                .trim()
                .to_string();
            if mid.is_empty() {
                continue;
            }
            let hostname = item["hostname"].as_str().unwrap_or(&mid).trim().to_string();
            let status = item["status"]
                .as_str()
                .unwrap_or("unknown")
                .trim()
                .to_string();

            let (mut busy_opt, mut busy_reason) = extract_busy(&item);
            if busy_opt.is_none() {
                let enc_id = encode(&mid);
                let pres_path =
                    format!("/v1/workflow/namespaces/fleet/activities/{enc_id}/{enc_id}");
                if let Ok(pres) = self.get::<Value>(&pres_path) {
                    for sub in [&pres, &pres["input"], &pres["details"], &pres["result"]] {
                        let (b_sub, r_sub) = extract_busy(sub);
                        if b_sub.is_some() {
                            busy_opt = b_sub;
                            busy_reason = r_sub;
                            break;
                        }
                    }
                }
            }

            let gpus = item["gpus"].as_array().cloned().unwrap_or_default();
            let mut mem_bytes = parse_mem_bytes(&item["memory"]);
            if mem_bytes == 0 {
                mem_bytes = gpus
                    .iter()
                    .map(|g| parse_mem_bytes(&g["memoryTotal"]))
                    .max()
                    .unwrap_or(0);
            }
            let mem_gb = if mem_bytes > 0 {
                ((mem_bytes as f64 / (1024_f64.powi(3))) * 100.0).round() / 100.0
            } else {
                0.0
            };

            let caps: Vec<String> = item["capabilities"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_default();
            let backends = infer_machine_backends(&item);
            let queue = item["jobQueue"]
                .as_str()
                .or_else(|| item["taskQueue"].as_str())
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(str::to_string)
                .unwrap_or_else(|| format!("gpu:{mid}"));

            let key = mid.to_ascii_lowercase();
            if !by_key.contains_key(&key) {
                order_keys.push(key.clone());
            }
            alias_to_key.insert(key.clone(), key.clone());
            alias_to_key.insert(hostname.to_ascii_lowercase(), key.clone());
            if let Some(stem) = key.strip_suffix("-hanzod") {
                alias_to_key.insert(stem.to_string(), key.clone());
            }
            by_key.insert(
                key,
                LinkedMachine {
                    id: mid,
                    hostname,
                    status,
                    os: item["os"].as_str().unwrap_or("").to_string(),
                    arch: item["arch"].as_str().unwrap_or("").to_string(),
                    cpus: item["cpus"].as_u64().unwrap_or(0) as u32,
                    memory_bytes: mem_bytes,
                    memory_gb: mem_gb,
                    capacity: item["capacity"].as_str().unwrap_or("").to_string(),
                    gpus,
                    capabilities: caps,
                    backends,
                    job_queue: queue.clone(),
                    task_queue: queue,
                    busy: busy_opt.unwrap_or(false),
                    busy_reason,
                },
            );
        }

        for item in target_rows {
            let handle = canonical_target_handle(&item);
            if handle.is_empty() {
                continue;
            }
            let host = item["host"].as_str().unwrap_or(&handle).trim().to_string();
            let raw_id = item["id"].as_str().unwrap_or("").trim().to_string();
            let cap_str = item["capacity"].as_str().unwrap_or("").trim().to_string();
            let (cap_cpus, cap_mem) = parse_capacity_hint(&cap_str);

            let existing_key = alias_to_key
                .get(&handle.to_ascii_lowercase())
                .or_else(|| alias_to_key.get(&host.to_ascii_lowercase()))
                .or_else(|| {
                    if raw_id.is_empty() {
                        None
                    } else {
                        alias_to_key.get(&raw_id.to_ascii_lowercase())
                    }
                })
                .cloned();

            let (t_busy, t_reason) = extract_busy(&item);
            let t_backends = infer_machine_backends(&item);
            let t_gpus = item["spec"]["gpus"]
                .as_array()
                .or_else(|| item["gpus"].as_array())
                .cloned()
                .unwrap_or_default();
            let mut t_mem = parse_mem_bytes(&item["spec"]["memory"]);
            if t_mem == 0 {
                t_mem = cap_mem;
            }
            if t_mem == 0 {
                t_mem = t_gpus
                    .iter()
                    .map(|g| parse_mem_bytes(&g["memoryTotal"]))
                    .max()
                    .unwrap_or(0);
            }
            let t_cpus = item["spec"]["cpus"]
                .as_u64()
                .or_else(|| item["cpus"].as_u64())
                .map(|c| c as u32)
                .filter(|&c| c > 0)
                .unwrap_or(cap_cpus);

            if let Some(ek) = existing_key {
                if let Some(m) = by_key.get_mut(&ek) {
                    if m.capacity.is_empty() && !cap_str.is_empty() {
                        m.capacity = cap_str;
                    }
                    if m.cpus == 0 && t_cpus > 0 {
                        m.cpus = t_cpus;
                    }
                    if m.memory_bytes == 0 && t_mem > 0 {
                        m.memory_bytes = t_mem;
                        m.memory_gb = ((t_mem as f64 / (1024_f64.powi(3))) * 100.0).round() / 100.0;
                    }
                    if m.gpus.is_empty() && !t_gpus.is_empty() {
                        m.gpus = t_gpus;
                    }
                    for b in t_backends {
                        if !m.backends.contains(&b) {
                            m.backends.push(b);
                        }
                    }
                    if !m.busy && t_busy.unwrap_or(false) {
                        m.busy = true;
                        m.busy_reason = t_reason;
                    }
                    continue;
                }
            }

            let status = item["status"]
                .as_str()
                .unwrap_or("offline")
                .trim()
                .to_string();
            let caps: Vec<String> = item["capabilities"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .filter_map(Value::as_str)
                        .map(str::trim)
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect()
                })
                .unwrap_or_else(|| {
                    vec![
                        "research.bench".to_string(),
                        "research.tick".to_string(),
                        "hanzod.run".to_string(),
                    ]
                });
            let queue = format!("gpu:{handle}");
            let key = handle.to_ascii_lowercase();
            if !by_key.contains_key(&key) {
                order_keys.push(key.clone());
            }
            alias_to_key.insert(key.clone(), key.clone());
            alias_to_key.insert(host.to_ascii_lowercase(), key.clone());
            if !raw_id.is_empty() {
                alias_to_key.insert(raw_id.to_ascii_lowercase(), key.clone());
            }
            by_key.insert(
                key,
                LinkedMachine {
                    id: handle,
                    hostname: host,
                    status,
                    os: item["spec"]["os"]
                        .as_str()
                        .or_else(|| item["os"].as_str())
                        .unwrap_or("")
                        .to_string(),
                    arch: item["spec"]["arch"]
                        .as_str()
                        .or_else(|| item["arch"].as_str())
                        .unwrap_or("")
                        .to_string(),
                    cpus: t_cpus,
                    memory_bytes: t_mem,
                    memory_gb: if t_mem > 0 {
                        ((t_mem as f64 / (1024_f64.powi(3))) * 100.0).round() / 100.0
                    } else {
                        0.0
                    },
                    capacity: cap_str,
                    gpus: t_gpus,
                    capabilities: caps,
                    backends: t_backends,
                    job_queue: queue.clone(),
                    task_queue: queue,
                    busy: t_busy.unwrap_or(false),
                    busy_reason: t_reason,
                },
            );
        }

        let want_cap = capability
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_ascii_lowercase);
        let want_backend = backend
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_ascii_uppercase);

        let mut out = Vec::new();
        for k in order_keys {
            let Some(m) = by_key.remove(&k) else {
                continue;
            };
            if only_online && !is_online_status(&m.status) {
                continue;
            }
            if exclude_busy && m.busy {
                continue;
            }
            if let Some(wc) = &want_cap {
                if !m.capabilities.iter().any(|c| c.to_ascii_lowercase() == *wc) {
                    continue;
                }
            }
            if let Some(wb) = &want_backend {
                if !m.backends.iter().any(|b| b == wb) {
                    continue;
                }
            }
            out.push(m);
        }
        Ok(out)
    }

    fn lookup_machine_status_map(&self) -> BTreeMap<String, String> {
        let mut map = BTreeMap::new();
        let Ok(rows) = self.list_machines(false, None, None, false) else {
            return map;
        };
        for m in rows {
            let st = if is_online_status(&m.status) {
                "online".to_string()
            } else {
                "queued_until_online".to_string()
            };
            map.insert(m.id.to_ascii_lowercase(), st.clone());
            map.insert(m.hostname.to_ascii_lowercase(), st.clone());
            if let Some(stem) = m.id.to_ascii_lowercase().strip_suffix("-hanzod") {
                map.insert(stem.to_string(), st);
            }
        }
        map
    }

    /// Dispatch a durable, busy-aware standalone activity to `hanzoai/tasks` (never rejects offline targets).
    pub fn dispatch_task(&self, payload: &Value, opts: &TaskDispatchOptions) -> Result<Value> {
        let ns = if opts.namespace.trim().is_empty() {
            "gpu-jobs"
        } else {
            opts.namespace.trim()
        };
        let raw_target = opts.target.as_deref().unwrap_or("").trim();
        let is_targeted = !raw_target.is_empty()
            && !raw_target.eq_ignore_ascii_case("any")
            && raw_target != "gpu-jobs";
        let machine_name = if is_targeted {
            raw_target.strip_prefix("gpu:").unwrap_or(raw_target)
        } else {
            "any"
        };
        let queue = resolve_task_queue(opts.target.as_deref());
        let act_type = if opts.activity_type.trim().is_empty() {
            "research.bench"
        } else {
            opts.activity_type.trim()
        };
        let sched_to_close = opts
            .schedule_to_close_timeout
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(if is_targeted { "72h" } else { "24h" });

        let m_status = if let Some(hint) = &opts.machine_status_hint {
            hint.clone()
        } else if is_targeted {
            self.lookup_machine_status_map()
                .get(&machine_name.to_ascii_lowercase())
                .cloned()
                .unwrap_or_else(|| "queued_until_online".to_string())
        } else {
            "online".to_string()
        };

        let mut enriched = match payload.as_object() {
            Some(map) => Value::Object(map.clone()),
            None => json!({"payload": payload}),
        };
        if let Some(obj) = enriched.as_object_mut() {
            obj.entry("when_idle_only".to_string())
                .or_insert_with(|| json!(opts.when_idle_only));
            if is_targeted {
                obj.entry("target_machine".to_string())
                    .or_insert_with(|| json!(machine_name));
            }
            if let Some(req_b) = opts
                .required_backend
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                obj.insert(
                    "required_backend".to_string(),
                    json!(req_b.to_ascii_uppercase()),
                );
            }
            if let Some(min_gb) = opts.min_memory_gb {
                obj.insert("min_memory_gb".to_string(), json!(min_gb));
            }
        }

        let seed_val = json!({
            "namespace": ns,
            "taskQueue": queue,
            "activityType": act_type,
            "jobId": opts.job_id.as_deref().unwrap_or("").trim(),
            "input": enriched,
        });
        let canon = rfc8785_canonicalize(&seed_val)?;
        let seed_hex = sha256_hex(&canon);

        let jid = opts
            .job_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("job-{}-{}", unix_now(), &seed_hex[..12]));
        let rid = opts
            .run_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| jid.clone());
        let req_id = opts
            .request_id
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("req-{}", &seed_hex[..24]));

        let mut body = json!({
            "id": jid,
            "activityId": jid,
            "runId": rid,
            "type": act_type,
            "activityType": {"name": act_type},
            "taskQueue": queue,
            "input": enriched,
            "retryPolicy": opts.retry_policy.clone().unwrap_or_else(default_retry_policy),
            "heartbeatTimeout": opts.heartbeat_timeout,
            "startToCloseTimeout": opts.start_to_close_timeout,
            "scheduleToCloseTimeout": sched_to_close,
            "requestId": req_id,
        });
        if !opts.identity.trim().is_empty() {
            body["identity"] = json!(opts.identity.trim());
        }

        let path = format!("/v1/workflow/namespaces/{}/activities", encode(ns));
        let mut resp: Value = self.post(&path, &body)?;
        let resolved_id = resp["execution"]["workflowId"]
            .as_str()
            .or_else(|| resp["activityId"].as_str())
            .or_else(|| resp["id"].as_str())
            .unwrap_or(&jid)
            .to_string();
        let resolved_run = resp["execution"]["runId"]
            .as_str()
            .or_else(|| resp["runId"].as_str())
            .unwrap_or(&rid)
            .to_string();
        if let Some(obj) = resp.as_object_mut() {
            obj.entry("id".to_string())
                .or_insert_with(|| json!(resolved_id));
            obj.entry("activityId".to_string())
                .or_insert_with(|| json!(resolved_id));
            obj.entry("activity_id".to_string())
                .or_insert_with(|| json!(resolved_id));
            obj.entry("runId".to_string())
                .or_insert_with(|| json!(resolved_run));
            obj.entry("run_id".to_string())
                .or_insert_with(|| json!(resolved_run));
            obj.entry("taskQueue".to_string())
                .or_insert_with(|| json!(queue));
            obj.entry("task_queue".to_string())
                .or_insert_with(|| json!(queue));
            obj.entry("machine".to_string())
                .or_insert_with(|| json!(machine_name));
            obj.entry("machine_status".to_string())
                .or_insert_with(|| json!(m_status));
            obj.entry("scheduleToCloseTimeout".to_string())
                .or_insert_with(|| json!(sched_to_close));
            obj.entry("requestId".to_string())
                .or_insert_with(|| json!(req_id));
            obj.entry("input".to_string())
                .or_insert_with(|| enriched.clone());
        }
        Ok(resp)
    }

    fn describe_activity(&self, ns: &str, job_id: &str, run_id: &str) -> Result<Value> {
        let path = format!(
            "/v1/workflow/namespaces/{}/activities/{}/{}",
            encode(ns),
            encode(job_id),
            encode(run_id)
        );
        let mut val: Value = self.get(&path)?;
        let state = val["status"]
            .as_str()
            .or_else(|| val["state"].as_str())
            .unwrap_or("")
            .to_string();
        if let Some(obj) = val.as_object_mut() {
            obj.entry("id".to_string()).or_insert_with(|| json!(job_id));
            if !state.is_empty() {
                obj.entry("status".to_string())
                    .or_insert_with(|| json!(state));
                obj.entry("state".to_string())
                    .or_insert_with(|| json!(state));
            }
        }
        Ok(val)
    }

    /// Wait for activity completion via `GET /v1/workflow/events` SSE stream (reconnect backoff only on drop).
    pub fn wait_gpu_job_with_run(
        &self,
        job_id: &str,
        run_id: Option<&str>,
        namespace: Option<&str>,
        timeout: Duration,
        poll_interval: Duration,
    ) -> Result<Value> {
        let ns = namespace
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("gpu-jobs");
        let aid = job_id.trim();
        let rid = run_id
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .unwrap_or(aid);

        let start = Instant::now();
        let base_budget = timeout.max(Duration::from_millis(50));
        let mut deadline = start + base_budget;
        let hard_deadline = start + (base_budget * 3).max(base_budget + Duration::from_secs(120));
        let mut seen_defers: u64 = 0;
        let mut sse_supported = true;
        let mut reconnect_attempt = 0u32;
        let fallback_sleep = poll_interval.max(Duration::from_millis(10));

        let note_defer = |dc: u64, dl: &mut Instant, seen: &mut u64| {
            if dc > *seen {
                let step = (base_budget.as_secs_f64() * 0.25).clamp(0.05, 30.0);
                let extra = Duration::from_secs_f64(step * (dc - *seen) as f64);
                *dl = (*dl + extra).min(hard_deadline);
                *seen = dc;
            }
        };

        loop {
            let now_inst = Instant::now();
            if now_inst >= deadline {
                let snap = self.describe_activity(ns, aid, rid)?;
                let st = snap["status"].as_str().unwrap_or("");
                if is_terminal_activity_state(st) {
                    return Ok(snap);
                }
                let dc = snap["deferCount"].as_u64().unwrap_or(seen_defers);
                let busy_reason = snap["lastBusyReason"].as_str().unwrap_or("");
                let detail = if dc > 0 || !busy_reason.is_empty() {
                    format!(", deferCount={dc}, lastBusyReason={busy_reason}")
                } else {
                    String::new()
                };
                return Err(Error::Message(format!(
                    "timed out waiting for gpu job {aid} (last state: {st}{detail})"
                )));
            }
            let remaining = deadline.saturating_duration_since(now_inst);

            if sse_supported {
                let snap0 = self.describe_activity(ns, aid, rid)?;
                let st0 = snap0["status"].as_str().unwrap_or("");
                if is_terminal_activity_state(st0) {
                    return Ok(snap0);
                }
                note_defer(
                    snap0["deferCount"].as_u64().unwrap_or(0),
                    &mut deadline,
                    &mut seen_defers,
                );
                match self.open_event_stream("/v1/workflow/events", remaining) {
                    Ok(resp) => {
                        let mut reader = BufReader::new(resp.into_reader());
                        let term = consume_sse_until_terminal(&mut reader, ns, aid, rid, |dc| {
                            note_defer(dc, &mut deadline, &mut seen_defers)
                        });
                        let snap = self.describe_activity(ns, aid, rid)?;
                        let st = snap["status"].as_str().unwrap_or("");
                        if is_terminal_activity_state(st) || term.is_some() {
                            return Ok(snap);
                        }
                        reconnect_attempt = reconnect_attempt.saturating_add(1);
                    }
                    Err(Error::Http(http_err)) => {
                        if matches!(&*http_err, ureq::Error::Status(404 | 405 | 501, _)) {
                            sse_supported = false;
                        } else {
                            reconnect_attempt = reconnect_attempt.saturating_add(1);
                        }
                    }
                    Err(_) => {
                        if let Ok(snap) = self.describe_activity(ns, aid, rid) {
                            let st = snap["status"].as_str().unwrap_or("");
                            if is_terminal_activity_state(st) {
                                return Ok(snap);
                            }
                            note_defer(
                                snap["deferCount"].as_u64().unwrap_or(0),
                                &mut deadline,
                                &mut seen_defers,
                            );
                        }
                        reconnect_attempt = reconnect_attempt.saturating_add(1);
                    }
                }

                if sse_supported {
                    let rem = deadline.saturating_duration_since(Instant::now());
                    if !rem.is_zero() {
                        let backoff_secs =
                            (0.25 * (2_f64.powi(reconnect_attempt.min(5) as i32))).min(15.0);
                        std::thread::sleep(rem.min(Duration::from_secs_f64(backoff_secs)));
                    }
                    continue;
                }
            }

            let snap = self.describe_activity(ns, aid, rid)?;
            let st = snap["status"].as_str().unwrap_or("");
            if is_terminal_activity_state(st) {
                return Ok(snap);
            }
            note_defer(
                snap["deferCount"].as_u64().unwrap_or(0),
                &mut deadline,
                &mut seen_defers,
            );
            let rem = deadline.saturating_duration_since(Instant::now());
            if !rem.is_zero() {
                std::thread::sleep(rem.min(fallback_sleep));
            }
        }
    }

    /// Release/defer a claimed activity without failing it (`POST .../activities/<id>/<run>/release`).
    pub fn release_gpu_job(
        &self,
        activity_id: &str,
        run_id: Option<&str>,
        reason: &str,
        identity: &str,
        delay_seconds: u64,
        namespace: Option<&str>,
    ) -> Result<Value> {
        let ns = namespace
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("gpu-jobs");
        let enc_aid = encode(activity_id.trim());
        let enc_rid = encode(
            run_id
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .unwrap_or(activity_id.trim()),
        );
        let path = format!(
            "/v1/workflow/namespaces/{}/activities/{enc_aid}/{enc_rid}/release",
            encode(ns)
        );
        let body = json!({
            "reason": if reason.trim().is_empty() { "busy" } else { reason.trim() },
            "identity": identity,
            "delaySeconds": delay_seconds,
        });
        self.post(&path, &body)
    }

    /// Orchestrate research runs across linked machines (`--target evo,dgx`, `fanout`, or best-idle).
    pub fn orchestrate_across_machines(
        &self,
        tasks: &[Value],
        opts: &OrchestrateOptions,
    ) -> Result<Vec<Value>> {
        if tasks.is_empty() {
            return Ok(Vec::new());
        }

        let explicit_targets: Vec<String> = parse_targets(opts.target.as_deref())
            .into_iter()
            .filter(|t| !t.eq_ignore_ascii_case("any"))
            .collect();

        let mut dispatched = Vec::new();
        if !explicit_targets.is_empty() {
            let status_map = self.lookup_machine_status_map();
            for (idx, item) in tasks.iter().enumerate() {
                for raw_t in &explicit_targets {
                    let clean_t = raw_t.strip_prefix("gpu:").unwrap_or(raw_t);
                    let mut payload = if item.is_string() {
                        json!({"command": item.as_str().unwrap_or("")})
                    } else {
                        item.clone()
                    };
                    let mut explicit_id = String::new();
                    if let Some(obj) = payload.as_object_mut() {
                        if let Some(jid_val) = obj.remove("job_id") {
                            explicit_id = jid_val.as_str().unwrap_or("").trim().to_string();
                        }
                        obj.entry("target_machine".to_string())
                            .or_insert_with(|| json!(clean_t));
                    }
                    let jid = if !explicit_id.is_empty() && explicit_targets.len() > 1 {
                        Some(format!("{explicit_id}-{clean_t}"))
                    } else if !explicit_id.is_empty() {
                        Some(explicit_id)
                    } else {
                        None
                    };
                    let m_state = status_map
                        .get(&clean_t.to_ascii_lowercase())
                        .cloned()
                        .unwrap_or_else(|| "queued_until_online".to_string());
                    let canon = rfc8785_canonicalize(&payload)?;
                    let req_id = format!("target-{idx}-{clean_t}-{}", &sha256_hex(&canon)[..16]);
                    let d_opts = TaskDispatchOptions {
                        target: Some(clean_t.to_string()),
                        activity_type: opts.activity_type.clone(),
                        when_idle_only: opts.when_idle_only,
                        required_backend: opts.backend.clone(),
                        min_memory_gb: opts.min_memory_gb,
                        job_id: jid,
                        namespace: opts.namespace.clone(),
                        schedule_to_close_timeout: Some("72h".to_string()),
                        request_id: Some(req_id),
                        machine_status_hint: Some(m_state),
                        ..TaskDispatchOptions::default()
                    };
                    dispatched.push(self.dispatch_task(&payload, &d_opts)?);
                }
            }
        } else {
            let mut machines = self.list_machines(
                true,
                opts.capability.as_deref(),
                opts.backend.as_deref(),
                false,
            )?;
            if let Some(min_gb) = opts.min_memory_gb {
                machines.retain(|m| m.memory_gb <= 0.0 || m.memory_gb >= min_gb);
            }

            if opts.fanout {
                let targets: Vec<String> = if machines.is_empty() {
                    vec!["any".to_string()]
                } else {
                    machines.iter().map(|m| m.id.clone()).collect()
                };
                for (idx, item) in tasks.iter().enumerate() {
                    for t_id in &targets {
                        let mut payload = if item.is_string() {
                            json!({"command": item.as_str().unwrap_or("")})
                        } else {
                            item.clone()
                        };
                        let mut explicit_id = String::new();
                        if let Some(obj) = payload.as_object_mut() {
                            if let Some(jid_val) = obj.remove("job_id") {
                                explicit_id = jid_val.as_str().unwrap_or("").trim().to_string();
                            }
                            if t_id != "any" {
                                obj.entry("target_machine".to_string())
                                    .or_insert_with(|| json!(t_id));
                            }
                        }
                        let jid = if !explicit_id.is_empty() && t_id != "any" {
                            Some(format!("{explicit_id}-{t_id}"))
                        } else if !explicit_id.is_empty() {
                            Some(explicit_id)
                        } else {
                            None
                        };
                        let canon = rfc8785_canonicalize(&payload)?;
                        let req_id = format!("fanout-{idx}-{t_id}-{}", &sha256_hex(&canon)[..16]);
                        let d_opts = TaskDispatchOptions {
                            target: Some(t_id.clone()),
                            activity_type: opts.activity_type.clone(),
                            when_idle_only: opts.when_idle_only,
                            required_backend: opts.backend.clone(),
                            min_memory_gb: opts.min_memory_gb,
                            job_id: jid,
                            namespace: opts.namespace.clone(),
                            request_id: Some(req_id),
                            machine_status_hint: Some("online".to_string()),
                            ..TaskDispatchOptions::default()
                        };
                        dispatched.push(self.dispatch_task(&payload, &d_opts)?);
                    }
                }
            } else {
                let mut idle: Vec<&LinkedMachine> = machines.iter().filter(|m| !m.busy).collect();
                idle.sort_by(|a, b| {
                    (!b.gpus.is_empty())
                        .cmp(&!a.gpus.is_empty())
                        .then_with(|| b.memory_gb.total_cmp(&a.memory_gb))
                        .then_with(|| b.cpus.cmp(&a.cpus))
                        .then_with(|| b.id.cmp(&a.id))
                });
                for (idx, item) in tasks.iter().enumerate() {
                    let mut payload = if item.is_string() {
                        json!({"command": item.as_str().unwrap_or("")})
                    } else {
                        item.clone()
                    };
                    let mut explicit_id: Option<String> = None;
                    let chosen = if idle.is_empty() {
                        "any".to_string()
                    } else {
                        idle[idx % idle.len()].id.clone()
                    };
                    if let Some(obj) = payload.as_object_mut() {
                        if let Some(jid_val) = obj.remove("job_id") {
                            let s = jid_val.as_str().unwrap_or("").trim().to_string();
                            if !s.is_empty() {
                                explicit_id = Some(s);
                            }
                        }
                        if chosen != "any" {
                            obj.entry("target_machine".to_string())
                                .or_insert_with(|| json!(chosen));
                        }
                    }
                    let d_opts = TaskDispatchOptions {
                        target: Some(chosen.clone()),
                        activity_type: opts.activity_type.clone(),
                        when_idle_only: opts.when_idle_only,
                        required_backend: opts.backend.clone(),
                        min_memory_gb: opts.min_memory_gb,
                        job_id: explicit_id,
                        namespace: opts.namespace.clone(),
                        machine_status_hint: Some(if chosen != "any" {
                            "online".to_string()
                        } else {
                            "queued_until_online".to_string()
                        }),
                        ..TaskDispatchOptions::default()
                    };
                    dispatched.push(self.dispatch_task(&payload, &d_opts)?);
                }
            }
        }

        if !opts.wait {
            return Ok(dispatched);
        }

        let mut completed = Vec::with_capacity(dispatched.len());
        for d in &dispatched {
            let jid = d["id"].as_str().unwrap_or("");
            let rid = d["runId"].as_str();
            completed.push(self.wait_gpu_job_with_run(
                jid,
                rid,
                Some(&opts.namespace),
                opts.timeout,
                opts.poll_interval,
            )?);
        }
        Ok(completed)
    }

    /// Alias for [`Research::orchestrate_across_machines`].
    pub fn orchestrate_tasks(
        &self,
        tasks: &[Value],
        opts: &OrchestrateOptions,
    ) -> Result<Vec<Value>> {
        self.orchestrate_across_machines(tasks, opts)
    }

    /// Best-effort registration of `/v1/workflow/namespaces` before creating schedules.
    pub fn ensure_workflow_namespace(&self, namespace: &str) {
        let ns = if namespace.trim().is_empty() {
            "gpu-jobs"
        } else {
            namespace.trim()
        };
        let body = json!({
            "name": ns,
            "namespaceInfo": {
                "name": ns,
                "state": "NAMESPACE_STATE_REGISTERED",
                "description": "Hanzo autonomous research and GPU task queue",
            },
            "config": {"workflowExecutionRetentionTtl": "720h"},
            "isActive": true,
        });
        let _ = self.post::<Value>("/v1/workflow/namespaces", &body);
    }

    /// Create durable `hanzoai/tasks` schedule(s) (`--target evo`, `--target dgx`, or `--target evo,dgx`).
    pub fn create_schedule(&self, schedule_id: &str, opts: &ScheduleOptions) -> Result<Value> {
        let sid = schedule_id.trim();
        if sid.is_empty() {
            return Err(Error::Message("schedule_id is required".into()));
        }
        let ivl = opts.interval.as_deref().unwrap_or("").trim();
        let crn = opts.cron.as_deref().unwrap_or("").trim();
        if ivl.is_empty() && crn.is_empty() {
            return Err(Error::Message(
                "create_schedule requires interval (e.g. '30m') and/or cron".into(),
            ));
        }

        let targets = parse_targets(opts.target.as_deref());
        if targets.len() > 1 {
            let mut created = Vec::with_capacity(targets.len());
            let mut ids = Vec::with_capacity(targets.len());
            for raw_t in &targets {
                let clean_t = raw_t.strip_prefix("gpu:").unwrap_or(raw_t);
                let mut single_opts = opts.clone();
                single_opts.target = Some(clean_t.to_string());
                let child = self.create_schedule(&format!("{sid}-{clean_t}"), &single_opts)?;
                ids.push(
                    child["scheduleId"]
                        .as_str()
                        .unwrap_or(&format!("{sid}-{clean_t}"))
                        .to_string(),
                );
                created.push(child);
            }
            return Ok(json!({
                "schedules": created,
                "scheduleIds": ids,
            }));
        }

        let single_target = targets.first().map(String::as_str);
        let is_targeted =
            single_target.is_some_and(|t| !t.eq_ignore_ascii_case("any") && t != "gpu-jobs");
        let ns = if opts.namespace.trim().is_empty() {
            "gpu-jobs"
        } else {
            opts.namespace.trim()
        };
        self.ensure_workflow_namespace(ns);

        let mut spec = json!({});
        if !crn.is_empty() {
            spec["cronString"] = json!([crn]);
        }
        if !ivl.is_empty() {
            spec["interval"] = json!([{"interval": ivl}]);
        }

        let mut act_input = match &opts.input_payload {
            Some(Value::Object(m)) => Value::Object(m.clone()),
            _ => json!({}),
        };
        if let Some(obj) = act_input.as_object_mut() {
            obj.insert("when_idle_only".to_string(), json!(opts.when_idle_only));
            if let Some(st) = single_target.filter(|_| is_targeted) {
                obj.entry("target_machine".to_string())
                    .or_insert_with(|| json!(st.strip_prefix("gpu:").unwrap_or(st)));
            }
            if let Some(cmd) = opts
                .command
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                obj.insert("command".to_string(), json!(cmd));
            }
            if let Some(camp) = opts
                .campaign
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                obj.insert("campaign".to_string(), json!(camp));
            }
            if let Some(exec) = opts
                .executor
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                obj.insert("executor".to_string(), json!(exec));
            }
            if let Some(max_sec) = opts.max_compute_seconds {
                obj.insert("max_compute_seconds".to_string(), json!(max_sec));
            }
            if let Some(req_b) = opts
                .required_backend
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
            {
                obj.insert(
                    "required_backend".to_string(),
                    json!(req_b.to_ascii_uppercase()),
                );
            }
        }

        let mut action = json!({
            "taskQueue": resolve_task_queue(single_target),
            "input": act_input,
            "retryPolicy": opts.retry_policy.clone().unwrap_or_else(default_retry_policy),
            "heartbeatTimeout": opts.heartbeat_timeout,
            "startToCloseTimeout": opts.start_to_close_timeout,
            "scheduleToCloseTimeout": if is_targeted { "72h" } else { "24h" },
        });
        if let Some(wf) = opts
            .workflow_type
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        {
            action["workflowType"] = json!({"name": wf});
        }
        if !opts.activity_type.trim().is_empty() {
            action["activityType"] = json!({"name": opts.activity_type.trim()});
        }

        let body = json!({
            "scheduleId": sid,
            "namespace": ns,
            "spec": spec,
            "action": action,
            "state": {
                "paused": opts.paused,
                "note": opts.note,
            },
        });
        let path = format!("/v1/workflow/namespaces/{}/schedules", encode(ns));
        self.post(&path, &body)
    }

    /// List native `hanzoai/tasks` schedules in `/v1/workflow/namespaces/<ns>/schedules`.
    pub fn list_schedules(&self, namespace: Option<&str>) -> Result<Vec<Value>> {
        let ns = namespace
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("gpu-jobs");
        let path = format!("/v1/workflow/namespaces/{}/schedules", encode(ns));
        let resp: Value = self.get(&path)?;
        Ok(resp["schedules"]
            .as_array()
            .or_else(|| resp["data"].as_array())
            .cloned()
            .unwrap_or_default())
    }

    /// Describe one native `hanzoai/tasks` schedule.
    pub fn describe_schedule(&self, schedule_id: &str, namespace: Option<&str>) -> Result<Value> {
        let ns = namespace
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("gpu-jobs");
        let path = format!(
            "/v1/workflow/namespaces/{}/schedules/{}",
            encode(ns),
            encode(schedule_id.trim())
        );
        self.get(&path)
    }

    /// Trigger immediate execution of a native `hanzoai/tasks` schedule.
    pub fn trigger_schedule(
        &self,
        schedule_id: &str,
        request_id: Option<&str>,
        namespace: Option<&str>,
    ) -> Result<Value> {
        let ns = namespace
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("gpu-jobs");
        let sid = schedule_id.trim();
        let req_id = request_id
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .unwrap_or_else(|| format!("trig-{sid}-{}", unix_now()));
        let path = format!(
            "/v1/workflow/namespaces/{}/schedules/{}/trigger",
            encode(ns),
            encode(sid)
        );
        self.post(&path, &json!({"requestId": req_id}))
    }

    /// Pause a native `hanzoai/tasks` schedule.
    pub fn pause_schedule(
        &self,
        schedule_id: &str,
        note: &str,
        namespace: Option<&str>,
    ) -> Result<Value> {
        let ns = namespace
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("gpu-jobs");
        let path = format!(
            "/v1/workflow/namespaces/{}/schedules/{}/pause",
            encode(ns),
            encode(schedule_id.trim())
        );
        self.post(&path, &json!({"note": note}))
    }

    /// Resume/unpause a native `hanzoai/tasks` schedule.
    pub fn resume_schedule(&self, schedule_id: &str, namespace: Option<&str>) -> Result<Value> {
        let ns = namespace
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("gpu-jobs");
        let path = format!(
            "/v1/workflow/namespaces/{}/schedules/{}/unpause",
            encode(ns),
            encode(schedule_id.trim())
        );
        self.post(&path, &json!({}))
    }

    /// Delete a native `hanzoai/tasks` schedule.
    pub fn delete_schedule(&self, schedule_id: &str, namespace: Option<&str>) -> Result<Value> {
        let ns = namespace
            .filter(|s| !s.trim().is_empty())
            .unwrap_or("gpu-jobs");
        let path = format!(
            "/v1/workflow/namespaces/{}/schedules/{}",
            encode(ns),
            encode(schedule_id.trim())
        );
        self.delete(&path)
    }
}
