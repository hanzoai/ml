//! End-to-end correctness, concurrency, tenant-isolation, stochastic-freshness,
//! memory-bound, and negative-regime regression tests for `hanzo-seneca` (HIP-1349).

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;

use hanzo_seneca::scenarios::{
    build_cpu_bottlenecked_heavy_decode, build_mixed_concurrent_llm_and_vision,
    build_single_epoch_streaming_no_reuse, build_tight_ram_high_expansion,
};
use hanzo_seneca::{report_to_json, run_full_suite, run_scenario, AllocationPolicy, CorpusStore};

/// Serializes timing-sensitive integration tests within this test binary so
/// default parallel `cargo test` threads (and the 4-thread concurrency test)
/// do not starve each other's wall-clock `Instant::now()` CPU measurements
/// on 2-core CI runners.
static TIMING_TEST_LOCK: Mutex<()> = Mutex::new(());

fn lock_timing() -> std::sync::MutexGuard<'static, ()> {
    TIMING_TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

fn test_corpus() -> CorpusStore {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    CorpusStore::load_from_workspace(&root).unwrap_or_else(|_| {
        CorpusStore::from_bytes(
            b"hanzo-seneca fallback test corpus for isolated crate builds\n".repeat(4096),
        )
    })
}

#[test]
fn all_four_scenarios_satisfy_hard_safety_and_correctness_invariants() {
    let _guard = lock_timing();
    let corpus = test_corpus();
    let report = run_full_suite(&corpus, 1349);
    assert_eq!(report.scenario_results.len(), 4);

    for sc in &report.scenario_results {
        assert_eq!(sc.policies.len(), 5);
        for p in &sc.policies {
            // 1. Bounded memory invariant.
            assert!(
                p.cache_peak_used_bytes <= p.cache_budget_bytes,
                "scenario={} policy={}: peak {} > budget {}",
                sc.name,
                p.policy,
                p.cache_peak_used_bytes,
                p.cache_budget_bytes
            );
            // 2. Strict tenant isolation invariant.
            assert_eq!(
                p.cross_tenant_leaks, 0,
                "scenario={} policy={}: cross-tenant leak detected!",
                sc.name, p.policy
            );
            // 3. Stochastic-augmentation freshness invariant.
            assert!(
                (p.stochastic_freshness_pct - 100.0).abs() < 1e-9,
                "scenario={} policy={}: stochastic freshness {}% != 100%",
                sc.name,
                p.policy,
                p.stochastic_freshness_pct
            );
            for j in &p.per_job {
                if j.stochastic_transform {
                    assert_eq!(
                        j.transformed_hits, 0,
                        "stochastic job {} must never hit Transformed cache",
                        j.name
                    );
                    assert_eq!(
                        j.final_budget_transformed, 0,
                        "stochastic job {} must have 0 Transformed budget",
                        j.name
                    );
                }
            }
        }
    }

    // Verify zero-dependency JSON output is well-formed and complete.
    let json = report_to_json(&report);
    assert!(json.starts_with('{'));
    assert!(json.trim_end().ends_with('}'));
    assert!(json.contains("\"mixed_concurrent_llm_and_vision\""));
    assert!(json.contains("\"cpu_bottlenecked_heavy_decode\""));
    assert!(json.contains("\"tight_ram_high_expansion\""));
    assert!(json.contains("\"single_epoch_streaming_no_reuse\""));
    assert!(json.contains("\"seneca_dynamic\""));
}

#[test]
fn cpu_bottlenecked_seneca_beats_encoded_only_and_no_cache() {
    let _guard = lock_timing();
    let corpus = test_corpus();
    let sc = build_cpu_bottlenecked_heavy_decode(&corpus, 1349);
    let m_no = run_scenario(&sc, AllocationPolicy::NoCache);
    let m_enc = run_scenario(&sc, AllocationPolicy::EncodedOnly);
    let m_sen = run_scenario(&sc, AllocationPolicy::SenecaDynamic);

    assert!(
        m_sen.gpu_idle_time_pct < m_enc.gpu_idle_time_pct - 5.0,
        "expected Seneca GPU idle ({:.2}%) to be >5pp below encoded_only ({:.2}%)",
        m_sen.gpu_idle_time_pct,
        m_enc.gpu_idle_time_pct
    );
    assert!(
        m_sen.aggregate_examples_per_sec > m_no.aggregate_examples_per_sec * 1.5,
        "expected Seneca throughput ({:.1}) > 1.5x no_cache ({:.1})",
        m_sen.aggregate_examples_per_sec,
        m_no.aggregate_examples_per_sec
    );
}

#[test]
fn tight_ram_high_expansion_seneca_avoids_decoded_only_thrashing() {
    let _guard = lock_timing();
    let corpus = test_corpus();
    let sc = build_tight_ram_high_expansion(&corpus, 1349);
    let m_dec = run_scenario(&sc, AllocationPolicy::DecodedOnly);
    let m_eq = run_scenario(&sc, AllocationPolicy::StaticEqualSplit);
    let m_sen = run_scenario(&sc, AllocationPolicy::SenecaDynamic);

    // Confirm real >= 6x decoded-to-encoded expansion on vision shards.
    for j in &m_sen.per_job {
        assert!(
            j.decode_expansion_ratio >= 6.0,
            "expected >=6x decode expansion on {}, got {:.2}x",
            j.name,
            j.decode_expansion_ratio
        );
    }

    assert!(
        m_sen.gpu_idle_time_pct + 15.0 < m_dec.gpu_idle_time_pct,
        "expected Seneca GPU idle ({:.2}%) << decoded_only ({:.2}%)",
        m_sen.gpu_idle_time_pct,
        m_dec.gpu_idle_time_pct
    );
    assert!(
        m_sen.s3_bytes_saved_pct > 50.0 && m_dec.s3_bytes_saved_pct < 15.0,
        "expected Seneca S3 saved ({:.1}%) >> decoded_only ({:.1}%)",
        m_sen.s3_bytes_saved_pct,
        m_dec.s3_bytes_saved_pct
    );
    assert!(
        m_sen.aggregate_examples_per_sec > m_eq.aggregate_examples_per_sec * 1.4,
        "expected Seneca throughput ({:.1}) > 1.4x static_equal_split ({:.1})",
        m_sen.aggregate_examples_per_sec,
        m_eq.aggregate_examples_per_sec
    );
}

#[test]
fn mixed_concurrent_workload_seneca_uses_multi_stage_allocation() {
    let _guard = lock_timing();
    let corpus = test_corpus();
    let sc = build_mixed_concurrent_llm_and_vision(&corpus, 1349);
    let m_no = run_scenario(&sc, AllocationPolicy::NoCache);
    let m_dec = run_scenario(&sc, AllocationPolicy::DecodedOnly);
    let m_sen = run_scenario(&sc, AllocationPolicy::SenecaDynamic);

    assert!(
        m_sen.gpu_idle_time_pct < m_no.gpu_idle_time_pct - 15.0,
        "expected Seneca GPU idle ({:.2}%) << no_cache ({:.2}%)",
        m_sen.gpu_idle_time_pct,
        m_no.gpu_idle_time_pct
    );
    assert!(
        m_sen.gpu_idle_time_pct < m_dec.gpu_idle_time_pct - 5.0,
        "expected Seneca GPU idle ({:.2}%) < decoded_only ({:.2}%)",
        m_sen.gpu_idle_time_pct,
        m_dec.gpu_idle_time_pct
    );
    assert!(
        m_sen.overall_cache_hit_rate_pct > 35.0
            && (m_sen.decoded_hit_rate_pct + m_sen.transformed_hit_rate_pct) > 20.0,
        "expected multi-stage hits (overall {:.1}% > 35%, dec+trn {:.1}% > 20%)",
        m_sen.overall_cache_hit_rate_pct,
        m_sen.decoded_hit_rate_pct + m_sen.transformed_hit_rate_pct
    );
    assert_eq!(m_sen.cross_tenant_leaks, 0);
    assert!((m_sen.stochastic_freshness_pct - 100.0).abs() < 1e-9);
}

#[test]
fn negative_regime_streaming_triggers_bypass_with_zero_cache_churn() {
    let _guard = lock_timing();
    let corpus = test_corpus();
    let sc = build_single_epoch_streaming_no_reuse(&corpus, 1349);
    let m_sen = run_scenario(&sc, AllocationPolicy::SenecaDynamic);

    assert_eq!(m_sen.overall_cache_hit_rate_pct, 0.0);
    assert!(
        m_sen.streaming_bypass_activations > 0,
        "expected Seneca streaming bypass to activate on 1-epoch zero-reuse stream"
    );
    assert_eq!(
        m_sen.cache_peak_used_bytes, 0,
        "streaming bypass must avoid wasting host RAM on zero-reuse data"
    );
}

#[test]
fn concurrent_threads_execute_deterministically_with_zero_leaks() {
    let _guard = lock_timing();
    let corpus = Arc::new(test_corpus());
    let mut handles = Vec::new();
    for _ in 0..4 {
        let c = Arc::clone(&corpus);
        handles.push(thread::spawn(move || {
            let sc = build_tight_ram_high_expansion(&c, 1349);
            run_scenario(&sc, AllocationPolicy::SenecaDynamic)
        }));
    }
    let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let expected_examples = results[0].total_examples;
    for r in &results {
        assert_eq!(r.total_examples, expected_examples);
        assert_eq!(r.cross_tenant_leaks, 0);
        assert!((r.stochastic_freshness_pct - 100.0).abs() < 1e-9);
        assert!(r.cache_peak_used_bytes <= r.cache_budget_bytes);
        assert!(
            r.s3_bytes_saved_pct > 45.0,
            "expected >45% S3 bytes saved on every concurrent thread, got {:.2}%",
            r.s3_bytes_saved_pct
        );
    }
}
