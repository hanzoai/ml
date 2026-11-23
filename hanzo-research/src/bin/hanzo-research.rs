//! Native Rust CLI for `hanzo-research`: query, failures, next, impact, sync, spool, tick, loop.

use std::collections::BTreeMap;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use hanzo_research::{
    impacted_benchmarks, AutonomousController, CampaignRegistry, LineageRule, LocalResearchLedger,
    Research, Spooler,
};
use serde_json::{json, Value};

fn default_ledger_dir() -> PathBuf {
    let home = env::var("HOME").unwrap_or_else(|_| ".".into());
    PathBuf::from(home).join(".hanzo").join("research")
}

fn read_lines_arg(raw: &str) -> Vec<String> {
    let p = Path::new(raw);
    if p.is_file() {
        fs::read_to_string(p)
            .unwrap_or_default()
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    } else {
        raw.split(',')
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect()
    }
}

fn print_usage() {
    eprintln!(
        "Usage: hanzo-research <command> [options]\n\
         Commands:\n\
         \x20 query       [--dir DIR] [--study S] [--verdict V] [--repo R] [--backend B] [--search Q]\n\
         \x20 failures    [--dir DIR] [--domain D]\n\
         \x20 next        [--dir DIR] [--domain D] [--top-k K]\n\
         \x20 impact      [--lineage FILE] [--tasks T1,T2] [--changed-paths FILE_OR_CSV]\n\
         \x20 sync        [--dir DIR]\n\
         \x20 spool       --wal FILE --cursor FILE\n\
         \x20 tick        --executor EXEC [--dir DIR] [--campaign FILE] [--lineage FILE] [--changed-paths P]\n\
         \x20 loop        --executor EXEC [--dir DIR] [--campaign FILE] [--max-ticks N] [--budget-minutes M]\n\
         \x20 controller  <status|validate|tick|loop> [options]"
    );
}

fn run_cli(args: &[String]) -> std::result::Result<Value, String> {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print_usage();
        return Ok(json!({"help": true}));
    }

    let mut base = String::new();
    let mut project = String::new();
    let mut idx = 0usize;
    while idx < args.len() && args[idx].starts_with("--") {
        match args[idx].as_str() {
            "--base" => {
                idx += 1;
                base = args.get(idx).cloned().unwrap_or_default();
            }
            "--project" => {
                idx += 1;
                project = args.get(idx).cloned().unwrap_or_default();
            }
            _ => break,
        }
        idx += 1;
    }

    let Some(cmd) = args.get(idx).map(String::as_str) else {
        print_usage();
        return Err("missing command".into());
    };
    idx += 1;

    let mut sub_action = String::new();
    if cmd == "controller" {
        sub_action = args
            .get(idx)
            .cloned()
            .ok_or_else(|| "controller requires <status|validate|tick|loop>".to_string())?;
        idx += 1;
    }

    let mut dir: Option<PathBuf> = None;
    let mut state_dir: Option<PathBuf> = None;
    let mut study: Option<String> = None;
    let mut verdict: Option<String> = None;
    let mut repo: Option<String> = None;
    let mut backend: Option<String> = None;
    let mut search: Option<String> = None;
    let mut domain: Option<String> = None;
    let mut include_failures = true;
    let mut top_k = 10usize;
    let mut lineage_path: Option<PathBuf> = None;
    let mut tasks_arg: Option<String> = None;
    let mut changed_paths_arg: Option<String> = None;
    let mut changed_artifacts_arg: Option<String> = None;
    let mut wal_path: Option<PathBuf> = None;
    let mut cursor_path: Option<PathBuf> = None;
    let mut campaign_path: Option<PathBuf> = None;
    let mut executor_path: Option<PathBuf> = None;
    let mut repo_dir = ".".to_string();
    let mut max_trials = 64usize;
    let mut timeout_minutes = 30u64;
    let mut max_ticks = 10usize;
    let mut budget_minutes = 60.0f64;
    let mut idle_exit = true;
    let mut sleep_seconds = 0.0f64;
    let mut auto_sync = true;

    while idx < args.len() {
        let flag = args[idx].as_str();
        let next_val = |i: &mut usize| -> std::result::Result<String, String> {
            *i += 1;
            args.get(*i)
                .cloned()
                .ok_or_else(|| format!("missing value for {flag}"))
        };
        match flag {
            "--dir" | "--ledger-dir" => dir = Some(PathBuf::from(next_val(&mut idx)?)),
            "--state" => state_dir = Some(PathBuf::from(next_val(&mut idx)?)),
            "--study" => study = Some(next_val(&mut idx)?),
            "--verdict" => verdict = Some(next_val(&mut idx)?),
            "--repo" => repo = Some(next_val(&mut idx)?),
            "--backend" => backend = Some(next_val(&mut idx)?),
            "--search" => search = Some(next_val(&mut idx)?),
            "--domain" => domain = Some(next_val(&mut idx)?),
            "--no-include-failures" | "--no-failures" => include_failures = false,
            "--include-failures" => include_failures = true,
            "--top-k" => {
                top_k = next_val(&mut idx)?
                    .parse()
                    .map_err(|_| "invalid --top-k".to_string())?
            }
            "--lineage" => lineage_path = Some(PathBuf::from(next_val(&mut idx)?)),
            "--tasks" => tasks_arg = Some(next_val(&mut idx)?),
            "--changed-paths" => changed_paths_arg = Some(next_val(&mut idx)?),
            "--changed-artifacts" => changed_artifacts_arg = Some(next_val(&mut idx)?),
            "--wal" => wal_path = Some(PathBuf::from(next_val(&mut idx)?)),
            "--cursor" => cursor_path = Some(PathBuf::from(next_val(&mut idx)?)),
            "--campaign" | "--registry" => campaign_path = Some(PathBuf::from(next_val(&mut idx)?)),
            "--executor" => executor_path = Some(PathBuf::from(next_val(&mut idx)?)),
            "--repo-dir" => repo_dir = next_val(&mut idx)?,
            "--max-trials" => {
                max_trials = next_val(&mut idx)?
                    .parse()
                    .map_err(|_| "invalid --max-trials".to_string())?
            }
            "--timeout-minutes" => {
                timeout_minutes = next_val(&mut idx)?
                    .parse()
                    .map_err(|_| "invalid --timeout-minutes".to_string())?
            }
            "--max-ticks" => {
                max_ticks = next_val(&mut idx)?
                    .parse()
                    .map_err(|_| "invalid --max-ticks".to_string())?
            }
            "--budget-minutes" => {
                budget_minutes = next_val(&mut idx)?
                    .parse()
                    .map_err(|_| "invalid --budget-minutes".to_string())?
            }
            "--idle-exit" => idle_exit = true,
            "--no-idle-exit" => idle_exit = false,
            "--sleep-seconds" => {
                sleep_seconds = next_val(&mut idx)?
                    .parse()
                    .map_err(|_| "invalid --sleep-seconds".to_string())?
            }
            "--auto-sync" => auto_sync = true,
            "--no-auto-sync" => auto_sync = false,
            "--base" => base = next_val(&mut idx)?,
            "--project" => project = next_val(&mut idx)?,
            other => return Err(format!("unknown option: {other}")),
        }
        idx += 1;
    }

    let make_client = || -> Research {
        if base.is_empty() && project.is_empty() {
            Research::from_env()
        } else {
            let env_c = Research::from_env();
            let key = std::env::var("HANZO_API_KEY")
                .ok()
                .filter(|k| !k.trim().is_empty())
                .unwrap_or_else(|| hanzo_research::token().unwrap_or_default());
            let proj = if project.is_empty() {
                hanzo_research::claimed(&key).unwrap_or_else(|| "default".into())
            } else {
                project.clone()
            };
            let _ = env_c;
            Research::new(&base, &key, &proj, &repo_dir, &[])
        }
    };

    match cmd {
        "query" => {
            if let Some(d) = dir {
                let ledger =
                    LocalResearchLedger::open(d, "hanzo-shared").map_err(|e| e.to_string())?;
                let rows = ledger
                    .query_experiments(
                        study.as_deref(),
                        verdict.as_deref(),
                        repo.as_deref(),
                        backend.as_deref(),
                        search.as_deref(),
                        include_failures,
                    )
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(rows).map_err(|e| e.to_string())
            } else {
                let client = make_client();
                let rows = client
                    .query_experiments(
                        study.as_deref(),
                        verdict.as_deref(),
                        repo.as_deref(),
                        backend.as_deref(),
                        search.as_deref(),
                        include_failures,
                        None,
                    )
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(rows).map_err(|e| e.to_string())
            }
        }
        "failures" => {
            if let Some(d) = dir {
                let ledger =
                    LocalResearchLedger::open(d, "hanzo-shared").map_err(|e| e.to_string())?;
                let rows = ledger
                    .prior_failures(domain.as_deref())
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(rows).map_err(|e| e.to_string())
            } else {
                let client = make_client();
                let rows = client
                    .prior_failures(domain.as_deref(), None)
                    .map_err(|e| e.to_string())?;
                serde_json::to_value(rows).map_err(|e| e.to_string())
            }
        }
        "next" => {
            if let Some(d) = dir {
                let ledger =
                    LocalResearchLedger::open(d, "hanzo-shared").map_err(|e| e.to_string())?;
                let ops = ledger
                    .recommended_experiments(domain.as_deref(), top_k)
                    .map_err(|e| e.to_string())?;
                let out: Vec<Value> = ops
                    .into_iter()
                    .map(|op| {
                        let score = op.priority_score();
                        let mut v = serde_json::to_value(op).unwrap_or(Value::Null);
                        v["priority_score"] = json!(score);
                        v
                    })
                    .collect();
                Ok(Value::Array(out))
            } else {
                let client = make_client();
                let ops = client
                    .recommended_experiments(domain.as_deref(), top_k, None)
                    .map_err(|e| e.to_string())?;
                let out: Vec<Value> = ops
                    .into_iter()
                    .map(|op| {
                        let score = op.priority_score();
                        let mut v = serde_json::to_value(op).unwrap_or(Value::Null);
                        v["priority_score"] = json!(score);
                        v
                    })
                    .collect();
                Ok(Value::Array(out))
            }
        }
        "impact" => {
            let lin: BTreeMap<String, LineageRule> = match &lineage_path {
                Some(p) => {
                    let bytes = fs::read(p).map_err(|e| e.to_string())?;
                    serde_json::from_slice(&bytes).map_err(|e| e.to_string())?
                }
                None => BTreeMap::new(),
            };
            let tasks: Vec<String> = match &tasks_arg {
                Some(t) => read_lines_arg(t),
                None => lin.keys().cloned().collect(),
            };
            if tasks.is_empty() {
                return Err("impact requires --tasks or --lineage".into());
            }
            let cpaths = changed_paths_arg
                .as_deref()
                .map(read_lines_arg)
                .unwrap_or_default();
            let carts = changed_artifacts_arg
                .as_deref()
                .map(read_lines_arg)
                .unwrap_or_default();
            let rep = impacted_benchmarks(&tasks, &cpaths, Some(&lin), &carts)
                .map_err(|e| e.to_string())?;
            serde_json::to_value(rep).map_err(|e| e.to_string())
        }
        "sync" | "sync-wal" => {
            let d = dir.unwrap_or_else(default_ledger_dir);
            let ledger = LocalResearchLedger::open(d, "hanzo-shared").map_err(|e| e.to_string())?;
            let client = make_client();
            let receipts = ledger.sync_to_cloud(&client).map_err(|e| e.to_string())?;
            serde_json::to_value(receipts).map_err(|e| e.to_string())
        }
        "spool" => {
            let w = wal_path.ok_or_else(|| "spool requires --wal".to_string())?;
            let c = cursor_path.ok_or_else(|| "spool requires --cursor".to_string())?;
            let client = make_client();
            let rep = Spooler::drain(&client, w, c, 3, 50).map_err(|e| e.to_string())?;
            serde_json::to_value(rep).map_err(|e| e.to_string())
        }
        "tick" | "loop" | "controller" => {
            let action = if cmd == "controller" {
                sub_action.as_str()
            } else {
                cmd
            };
            if action == "validate" {
                let cp = campaign_path
                    .ok_or_else(|| "controller validate requires --campaign".to_string())?;
                let reg = CampaignRegistry::from_file(cp).map_err(|e| e.to_string())?;
                return Ok(json!({
                    "campaign": reg.campaign,
                    "objectives": reg.objectives.len(),
                    "protected": reg.objectives.iter().filter(|o| o.protected).count(),
                    "families": reg.search_space.keys().collect::<Vec<_>>(),
                }));
            }
            let reg = match &campaign_path {
                Some(cp) => Some(CampaignRegistry::from_file(cp).map_err(|e| e.to_string())?),
                None => None,
            };
            let lin: Option<BTreeMap<String, LineageRule>> = match &lineage_path {
                Some(p) => {
                    let bytes = fs::read(p).map_err(|e| e.to_string())?;
                    Some(serde_json::from_slice(&bytes).map_err(|e| e.to_string())?)
                }
                None => None,
            };
            let d = dir.unwrap_or_else(default_ledger_dir);
            let ctrl = AutonomousController::open(
                &d,
                state_dir.as_deref(),
                reg,
                &repo_dir,
                max_trials,
                None,
            )
            .map_err(|e| e.to_string())?;
            if action == "status" {
                return ctrl.status().map_err(|e| e.to_string());
            }
            let exec = executor_path.ok_or_else(|| format!("{action} requires --executor"))?;
            let cpaths = changed_paths_arg.as_deref().map(read_lines_arg);
            let carts = changed_artifacts_arg.as_deref().map(read_lines_arg);
            let opt_client = if !base.is_empty() {
                Some(make_client())
            } else {
                None
            };
            if action == "tick" {
                ctrl.tick(
                    &exec,
                    cpaths.as_deref(),
                    lin.as_ref(),
                    carts.as_deref(),
                    timeout_minutes,
                    auto_sync,
                    opt_client.as_ref(),
                    600,
                )
                .map_err(|e| e.to_string())
            } else if action == "loop" {
                ctrl.loop_run(
                    &exec,
                    max_ticks,
                    budget_minutes,
                    idle_exit,
                    sleep_seconds,
                    cpaths.as_deref(),
                    lin.as_ref(),
                    carts.as_deref(),
                    auto_sync,
                    opt_client.as_ref(),
                )
                .map_err(|e| e.to_string())
            } else {
                Err(format!("unknown controller action: {action}"))
            }
        }
        other => Err(format!("unknown command: {other}")),
    }
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    match run_cli(&args) {
        Ok(val) => {
            println!(
                "{}",
                serde_json::to_string_pretty(&val).unwrap_or_else(|_| "null".into())
            );
        }
        Err(err) => {
            eprintln!("hanzo-research: {err}");
            process::exit(2);
        }
    }
}
