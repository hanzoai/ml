//! Reproducible CLI benchmark for `hanzo-seneca` (HIP-1349).
//!
//! Evaluates 5 matched cache allocation policies (`no_cache`, `encoded_only`,
//! `decoded_only`, `static_equal_split`, `seneca_dynamic`) across 4 concurrent
//! multi-job training scenarios over real `hanzoai/ml` repository corpus shards
//! and writes structured JSON results to `docs/research/hip-1349-seneca/results.json`.

#![forbid(unsafe_code)]

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;

use hanzo_seneca::{report_to_json, run_full_suite_with_trials, CorpusStore};

const DEFAULT_SEED: u64 = 1349;
const DEFAULT_TRIALS: usize = 3;
const DEFAULT_OUTPUT: &str = "docs/research/hip-1349-seneca/results.json";

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut seed = DEFAULT_SEED;
    let mut trials = DEFAULT_TRIALS;
    let mut output_path = PathBuf::from(DEFAULT_OUTPUT);
    let mut corpus_dir: Option<PathBuf> = None;

    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "--seed" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --seed requires a u64 argument");
                    process::exit(2);
                }
                seed = args[i].parse::<u64>().unwrap_or_else(|e| {
                    eprintln!("error: invalid --seed '{}': {e}", args[i]);
                    process::exit(2);
                });
            }
            "--trials" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --trials requires a usize argument");
                    process::exit(2);
                }
                trials = args[i].parse::<usize>().unwrap_or_else(|e| {
                    eprintln!("error: invalid --trials '{}': {e}", args[i]);
                    process::exit(2);
                });
                if trials == 0 {
                    eprintln!("error: --trials must be >= 1");
                    process::exit(2);
                }
            }
            "--output" | "-o" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --output requires a file path");
                    process::exit(2);
                }
                output_path = PathBuf::from(&args[i]);
            }
            "--corpus-dir" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("error: --corpus-dir requires a directory path");
                    process::exit(2);
                }
                corpus_dir = Some(PathBuf::from(&args[i]));
            }
            "--help" | "-h" => {
                println!(
                    "seneca-bench — HIP-1349 multi-stage training cache benchmark\n\
                     USAGE:\n  seneca-bench [--seed 1349] [--trials 3] [--output docs/research/hip-1349-seneca/results.json] [--corpus-dir <repo_root>]"
                );
                return;
            }
            other => {
                eprintln!("error: unknown argument '{other}' (try --help)");
                process::exit(2);
            }
        }
        i += 1;
    }

    let root = corpus_dir.unwrap_or_else(detect_workspace_root);
    let corpus = CorpusStore::load_from_workspace(&root).unwrap_or_else(|e| {
        eprintln!("error loading corpus from {}: {e}", root.display());
        process::exit(1);
    });

    println!(
        "hanzo-seneca benchmark (HIP-1349) | seed={} | trials={} (median) | corpus={} files ({:.2} MiB) | digest={:032x}",
        seed,
        trials,
        corpus.files_loaded(),
        corpus.total_source_bytes() as f64 / (1024.0 * 1024.0),
        corpus.source_digest()
    );

    let report = run_full_suite_with_trials(&corpus, seed, trials);

    for sc in &report.scenario_results {
        println!(
            "\n=== Scenario: {} {} (budget={} KiB, cores={}) ===",
            sc.name,
            if sc.is_negative_regime {
                "[NEGATIVE REGIME]"
            } else {
                ""
            },
            sc.total_cache_bytes / 1024,
            sc.host_cpu_cores
        );
        println!(
            "{:<20} {:>9} {:>11} {:>8} {:>8} {:>8} {:>9} {:>10} {:>8} {:>6}",
            "Policy",
            "GPU Idle%",
            "Ex/sec",
            "EncHit%",
            "DecHit%",
            "TrnHit%",
            "CacheUtil%",
            "S3 Saved%",
            "Fresh%",
            "Leaks"
        );
        for p in &sc.policies {
            println!(
                "{:<20} {:>8.2}% {:>11.1} {:>7.1}% {:>7.1}% {:>7.1}% {:>8.1}% {:>9.1}% {:>7.1}% {:>6}",
                p.policy,
                p.gpu_idle_time_pct,
                p.aggregate_examples_per_sec,
                p.encoded_hit_rate_pct,
                p.decoded_hit_rate_pct,
                p.transformed_hit_rate_pct,
                p.cache_avg_utilization_pct,
                p.s3_bytes_saved_pct,
                p.stochastic_freshness_pct,
                p.cross_tenant_leaks
            );
        }
    }

    let resolved_output = if output_path.is_absolute() || Path::new("Cargo.toml").exists() {
        output_path
    } else {
        root.join(output_path)
    };

    if let Some(parent) = resolved_output.parent() {
        if !parent.as_os_str().is_empty() {
            if let Err(e) = fs::create_dir_all(parent) {
                eprintln!("error creating output directory {}: {e}", parent.display());
                process::exit(1);
            }
        }
    }

    let json = report_to_json(&report);
    if let Err(e) = fs::write(&resolved_output, json) {
        eprintln!("error writing {}: {e}", resolved_output.display());
        process::exit(1);
    }
    println!("\nWrote benchmark report to {}", resolved_output.display());
}

fn detect_workspace_root() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    if let Some(parent) = manifest_dir.parent() {
        if parent.join("Cargo.toml").exists() {
            return parent.to_path_buf();
        }
    }
    env::current_dir().unwrap_or_else(|_| PathBuf::from("."))
}
