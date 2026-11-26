//! Four reproducible concurrent multi-job training scenarios (HIP-1349) and
//! a zero-dependency JSON report emitter.

use std::fmt::Write as _;

use crate::allocator::AllocationPolicy;
use crate::cache::DatasetVisibility;
use crate::pipeline::{
    run_scenario, CorpusStore, DatasetStore, JobConfig, JobMetrics, Modality, PolicyRunMetrics,
    ScenarioConfig,
};

/// Builds all 4 benchmark scenarios over the real `hanzoai/ml` repository corpus.
pub fn build_all_scenarios(corpus: &CorpusStore, seed: u64) -> Vec<ScenarioConfig> {
    vec![
        build_mixed_concurrent_llm_and_vision(corpus, seed),
        build_cpu_bottlenecked_heavy_decode(corpus, seed),
        build_tight_ram_high_expansion(corpus, seed),
        build_single_epoch_streaming_no_reuse(corpus, seed),
    ]
}

/// Scenario 1: 4 heterogeneous concurrent jobs sharing one bounded cache:
/// - I/O-bound LLM token shards
/// - CPU-decode-heavy deterministic image shards
/// - Heavy-transform deterministic audio/spectrogram shards
/// - Stochastic-augmentation vision shards (`Transformed` strictly forbidden)
pub fn build_mixed_concurrent_llm_and_vision(corpus: &CorpusStore, seed: u64) -> ScenarioConfig {
    let ds_llm = DatasetStore::build(
        "hanzo_ml_llm_shards",
        corpus,
        Modality::LlmTokenShard { raw_bytes: 6144 },
        128,
        seed ^ 0x101,
    );
    let ds_vis_det = DatasetStore::build(
        "hanzo_ml_vision_det_shards",
        corpus,
        Modality::VisionImageShard {
            width: 32,
            height: 32,
            channels: 3,
        },
        96,
        seed ^ 0x102,
    );
    let ds_audio = DatasetStore::build(
        "hanzo_ml_audio_mel_shards",
        corpus,
        Modality::AudioFeatureShard {
            frames: 14,
            frame_len: 80,
        },
        96,
        seed ^ 0x103,
    );
    let ds_vis_stoch = DatasetStore::build(
        "hanzo_ml_vision_stoch_shards",
        corpus,
        Modality::VisionImageShard {
            width: 32,
            height: 32,
            channels: 3,
        },
        96,
        seed ^ 0x104,
    );

    // Budget sized to ~2.1 MiB: fits optimal stage per job (Encoded for I/O-bound LLM,
    // Decoded for vision jobs, compact Transformed for audio), while forcing trade-offs
    // on DecodedOnly and StaticEqualSplit (which wastes 33% on stochastic Transformed!).
    let total_cache_bytes = 2_150_000;

    ScenarioConfig {
        name: "mixed_concurrent_llm_and_vision".to_string(),
        description: "4 concurrent heterogeneous jobs (I/O-bound LLM, decode-heavy vision, transform-heavy audio, stochastic-aug vision) sharing one bounded cache".to_string(),
        is_negative_regime: false,
        total_cache_bytes,
        host_cpu_cores: 2,
        seed,
        datasets: vec![ds_llm, ds_vis_det, ds_audio, ds_vis_stoch],
        jobs: vec![
            JobConfig {
                job_id: 1,
                tenant_id: 101,
                name: "llm_io_bound".to_string(),
                dataset_index: 0,
                visibility: DatasetVisibility::Private {
                    tenant_id: 101,
                    job_id: 1,
                },
                stochastic_transform: false,
                transform_profile: "bpe_pack_v1".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 950.0,
                io_bw_mb_s: 30.0,
                io_concurrency: 2.0,
                gpu_step_us: 1200.0,
                decode_compute_passes: 1,
                transform_compute_passes: 1,
                prefetch_batches: 2,
            },
            JobConfig {
                job_id: 2,
                tenant_id: 102,
                name: "vision_decode_heavy".to_string(),
                dataset_index: 1,
                visibility: DatasetVisibility::Private {
                    tenant_id: 102,
                    job_id: 2,
                },
                stochastic_transform: false,
                transform_profile: "center_crop_norm_v1".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 180.0,
                io_bw_mb_s: 160.0,
                io_concurrency: 2.0,
                gpu_step_us: 1200.0,
                decode_compute_passes: 95,
                transform_compute_passes: 2,
                prefetch_batches: 2,
            },
            JobConfig {
                job_id: 3,
                tenant_id: 103,
                name: "audio_transform_heavy".to_string(),
                dataset_index: 2,
                visibility: DatasetVisibility::Private {
                    tenant_id: 103,
                    job_id: 3,
                },
                stochastic_transform: false,
                transform_profile: "mel24_log_energy_v1".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 140.0,
                io_bw_mb_s: 200.0,
                io_concurrency: 2.0,
                gpu_step_us: 1200.0,
                decode_compute_passes: 2,
                transform_compute_passes: 1,
                prefetch_batches: 2,
            },
            JobConfig {
                job_id: 4,
                tenant_id: 104,
                name: "vision_stochastic_aug".to_string(),
                dataset_index: 3,
                visibility: DatasetVisibility::Private {
                    tenant_id: 104,
                    job_id: 4,
                },
                stochastic_transform: true,
                transform_profile: "rand_crop_flip_jitter_v1".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 180.0,
                io_bw_mb_s: 160.0,
                io_concurrency: 2.0,
                gpu_step_us: 1200.0,
                decode_compute_passes: 95,
                transform_compute_passes: 2,
                prefetch_batches: 2,
            },
        ],
    }
}

/// Scenario 2: Fast NVMe / local object storage, but heavy CPU decode + transform
/// saturating host cores across 3 concurrent jobs (including 2 jobs sharing a
/// `SharedPublic` dataset with different transform policies).
pub fn build_cpu_bottlenecked_heavy_decode(corpus: &CorpusStore, seed: u64) -> ScenarioConfig {
    let ds_shared_vis = DatasetStore::build(
        "hanzo_public_vision_shared",
        corpus,
        Modality::VisionImageShard {
            width: 32,
            height: 32,
            channels: 3,
        },
        112,
        seed ^ 0x201,
    );
    let ds_audio_priv = DatasetStore::build(
        "hanzo_private_audio_asr",
        corpus,
        Modality::AudioFeatureShard {
            frames: 16,
            frame_len: 80,
        },
        112,
        seed ^ 0x202,
    );
    let shared_vis = DatasetVisibility::SharedPublic {
        dataset_digest: ds_shared_vis.dataset_digest,
    };

    ScenarioConfig {
        name: "cpu_bottlenecked_heavy_decode".to_string(),
        description: "Fast storage with CPU decode/transform saturating host cores across 3 concurrent jobs (2 sharing a public vision dataset + 1 private heavy-transform audio job)".to_string(),
        is_negative_regime: false,
        total_cache_bytes: 1_850_000,
        host_cpu_cores: 2,
        seed,
        datasets: vec![ds_shared_vis, ds_audio_priv],
        jobs: vec![
            JobConfig {
                job_id: 11,
                tenant_id: 201,
                name: "shared_vision_det".to_string(),
                dataset_index: 0,
                visibility: shared_vis,
                stochastic_transform: false,
                transform_profile: "resnet_norm_v1".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 20.0,
                io_bw_mb_s: 500.0,
                io_concurrency: 4.0,
                gpu_step_us: 600.0,
                decode_compute_passes: 110,
                transform_compute_passes: 40,
                prefetch_batches: 2,
            },
            JobConfig {
                job_id: 12,
                tenant_id: 202,
                name: "shared_vision_stochastic".to_string(),
                dataset_index: 0,
                visibility: shared_vis,
                stochastic_transform: true,
                transform_profile: "vit_randaug_v1".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 20.0,
                io_bw_mb_s: 500.0,
                io_concurrency: 4.0,
                gpu_step_us: 600.0,
                decode_compute_passes: 110,
                transform_compute_passes: 2,
                prefetch_batches: 2,
            },
            JobConfig {
                job_id: 13,
                tenant_id: 203,
                name: "private_audio_heavy_transform".to_string(),
                dataset_index: 1,
                visibility: DatasetVisibility::Private {
                    tenant_id: 203,
                    job_id: 13,
                },
                stochastic_transform: false,
                transform_profile: "whisper_mel24_v1".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 20.0,
                io_bw_mb_s: 500.0,
                io_concurrency: 4.0,
                gpu_step_us: 600.0,
                decode_compute_passes: 6,
                transform_compute_passes: 1,
                prefetch_batches: 2,
            },
        ],
    }
}

/// Scenario 3: Tight host RAM + high decoded memory expansion (`>= 6x` over encoded).
/// Proves `decoded_only` thrashes catastrophically while `seneca_dynamic` holds
/// hot encoded shards (+ selective decoded entries where marginal utility wins).
pub fn build_tight_ram_high_expansion(corpus: &CorpusStore, seed: u64) -> ScenarioConfig {
    // 36x36x3 RGB image -> 3888 raw u8 bytes, compresses via LZ/RLE to ~1.5-2.2 KiB,
    // expands on decode to 3888 * 4 = 15,552 bytes (~7x-10x over encoded!).
    let ds_a = DatasetStore::build(
        "hanzo_vision_highexp_a",
        corpus,
        Modality::VisionImageShard {
            width: 36,
            height: 36,
            channels: 3,
        },
        112,
        seed ^ 0x301,
    );
    let ds_b = DatasetStore::build(
        "hanzo_vision_highexp_b",
        corpus,
        Modality::VisionImageShard {
            width: 36,
            height: 36,
            channels: 3,
        },
        112,
        seed ^ 0x302,
    );

    // Compute tight RAM budget: ~85% of combined Encoded working set, which is only
    // ~10%-14% of the combined Decoded working set!
    let encoded_ws_bytes: usize = ds_a
        .shards
        .iter()
        .chain(ds_b.shards.iter())
        .map(|s| s.bytes.len())
        .sum();
    let total_cache_bytes = ((encoded_ws_bytes as f64) * 0.90).ceil() as usize;

    ScenarioConfig {
        name: "tight_ram_high_expansion".to_string(),
        description: "High decoded memory expansion (>=6x over encoded) under tight RAM (~90% of encoded working set, ~12% of decoded working set) with slow S3 object storage".to_string(),
        is_negative_regime: false,
        total_cache_bytes,
        host_cpu_cores: 4,
        seed,
        datasets: vec![ds_a, ds_b],
        jobs: vec![
            JobConfig {
                job_id: 21,
                tenant_id: 301,
                name: "vision_highexp_a".to_string(),
                dataset_index: 0,
                visibility: DatasetVisibility::Private {
                    tenant_id: 301,
                    job_id: 21,
                },
                stochastic_transform: false,
                transform_profile: "std_norm_v1".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 1100.0,
                io_bw_mb_s: 20.0,
                io_concurrency: 2.0,
                gpu_step_us: 1800.0,
                decode_compute_passes: 2,
                transform_compute_passes: 1,
                prefetch_batches: 2,
            },
            JobConfig {
                job_id: 22,
                tenant_id: 302,
                name: "vision_highexp_b".to_string(),
                dataset_index: 1,
                visibility: DatasetVisibility::Private {
                    tenant_id: 302,
                    job_id: 22,
                },
                stochastic_transform: true,
                transform_profile: "rand_flip_v1".to_string(),
                epochs: 4,
                batch_size: 8,
                io_rtt_us: 1100.0,
                io_bw_mb_s: 20.0,
                io_concurrency: 2.0,
                gpu_step_us: 1800.0,
                decode_compute_passes: 2,
                transform_compute_passes: 1,
                prefetch_batches: 2,
            },
        ],
    }
}

/// Scenario 4 (Explicit Negative Regime): massive 1-epoch streaming corpus
/// (`working_set > 14x cache`, zero sample reuse). Caching cannot improve hit rate;
/// proves `seneca_dynamic` activates streaming bypass with `< 2%` overhead vs `no_cache`.
pub fn build_single_epoch_streaming_no_reuse(corpus: &CorpusStore, seed: u64) -> ScenarioConfig {
    let ds_stream_a = DatasetStore::build(
        "hanzo_stream_crawl_a",
        corpus,
        Modality::LlmTokenShard { raw_bytes: 4096 },
        1024,
        seed ^ 0x401,
    );
    let ds_stream_b = DatasetStore::build(
        "hanzo_stream_crawl_b",
        corpus,
        Modality::LlmTokenShard { raw_bytes: 4096 },
        1024,
        seed ^ 0x402,
    );

    // Encoded working set is ~3.6–4.5 MiB (decoded = 8 MiB) vs 256 KiB cache -> >14x working set!
    let total_cache_bytes = 256 * 1024;

    ScenarioConfig {
        name: "single_epoch_streaming_no_reuse".to_string(),
        description: "Explicit Negative Regime: 1-epoch web-scale streaming corpus (working_set > 14x cache, zero sample reuse) where caching yields 0% hit rate".to_string(),
        is_negative_regime: true,
        total_cache_bytes,
        host_cpu_cores: 2,
        seed,
        datasets: vec![ds_stream_a, ds_stream_b],
        jobs: vec![
            JobConfig {
                job_id: 31,
                tenant_id: 401,
                name: "llm_stream_1epoch_a".to_string(),
                dataset_index: 0,
                visibility: DatasetVisibility::Private {
                    tenant_id: 401,
                    job_id: 31,
                },
                stochastic_transform: false,
                transform_profile: "stream_bpe_v1".to_string(),
                epochs: 1,
                batch_size: 16,
                io_rtt_us: 120.0,
                io_bw_mb_s: 150.0,
                io_concurrency: 4.0,
                gpu_step_us: 1000.0,
                decode_compute_passes: 1,
                transform_compute_passes: 1,
                prefetch_batches: 2,
            },
            JobConfig {
                job_id: 32,
                tenant_id: 402,
                name: "llm_stream_1epoch_b".to_string(),
                dataset_index: 1,
                visibility: DatasetVisibility::Private {
                    tenant_id: 402,
                    job_id: 32,
                },
                stochastic_transform: false,
                transform_profile: "stream_bpe_v1".to_string(),
                epochs: 1,
                batch_size: 16,
                io_rtt_us: 120.0,
                io_bw_mb_s: 150.0,
                io_concurrency: 4.0,
                gpu_step_us: 1000.0,
                decode_compute_passes: 1,
                transform_compute_passes: 1,
                prefetch_batches: 2,
            },
        ],
    }
}

/// Complete benchmark suite output across all scenarios and policies.
#[derive(Debug, Clone)]
pub struct BenchmarkSuiteReport {
    pub seed: u64,
    pub trials: usize,
    pub corpus_files: usize,
    pub corpus_bytes: usize,
    pub corpus_digest_hex: String,
    pub scenario_results: Vec<ScenarioReport>,
}

#[derive(Debug, Clone)]
pub struct ScenarioReport {
    pub name: String,
    pub description: String,
    pub is_negative_regime: bool,
    pub total_cache_bytes: usize,
    pub host_cpu_cores: usize,
    pub policies: Vec<PolicyRunMetrics>,
}

/// Runs all 4 scenarios across all 5 matched policies (single trial).
pub fn run_full_suite(corpus: &CorpusStore, seed: u64) -> BenchmarkSuiteReport {
    run_full_suite_with_trials(corpus, seed, 1)
}

/// Runs all 4 scenarios across all 5 matched policies for `trials` executions each
/// and selects the median trial by `makespan_ms` to filter transient OS scheduling jitter.
pub fn run_full_suite_with_trials(
    corpus: &CorpusStore,
    seed: u64,
    trials: usize,
) -> BenchmarkSuiteReport {
    let trials = trials.max(1);
    let scenarios = build_all_scenarios(corpus, seed);
    let mut scenario_results = Vec::with_capacity(scenarios.len());

    for sc in &scenarios {
        let mut policies = Vec::with_capacity(AllocationPolicy::ALL.len());
        for &policy in &AllocationPolicy::ALL {
            let mut samples = Vec::with_capacity(trials);
            for _ in 0..trials {
                samples.push(run_scenario(sc, policy));
            }
            samples.sort_by(|a, b| a.makespan_ms.total_cmp(&b.makespan_ms));
            let median = samples.swap_remove(samples.len() / 2);
            policies.push(median);
        }
        scenario_results.push(ScenarioReport {
            name: sc.name.clone(),
            description: sc.description.clone(),
            is_negative_regime: sc.is_negative_regime,
            total_cache_bytes: sc.total_cache_bytes,
            host_cpu_cores: sc.host_cpu_cores,
            policies,
        });
    }

    BenchmarkSuiteReport {
        seed,
        trials,
        corpus_files: corpus.files_loaded(),
        corpus_bytes: corpus.total_source_bytes(),
        corpus_digest_hex: format!("{:032x}", corpus.source_digest()),
        scenario_results,
    }
}

/// Serializes [`BenchmarkSuiteReport`] to deterministic, pretty-printed JSON
/// without any external serialization dependencies.
pub fn report_to_json(report: &BenchmarkSuiteReport) -> String {
    let mut out = String::with_capacity(32 * 1024);
    let _ = writeln!(out, "{{");
    let _ = writeln!(out, "  \"benchmark\": \"hanzo-seneca-hip-1349\",");
    let _ = writeln!(out, "  \"seed\": {},", report.seed);
    let _ = writeln!(out, "  \"trials_median\": {},", report.trials);
    let _ = writeln!(out, "  \"corpus\": {{");
    let _ = writeln!(
        out,
        "    \"source\": \"hanzoai/ml workspace (.rs/.md/.toml)\","
    );
    let _ = writeln!(out, "    \"files_loaded\": {},", report.corpus_files);
    let _ = writeln!(out, "    \"total_bytes\": {},", report.corpus_bytes);
    let _ = writeln!(out, "    \"digest_128\": \"{}\"", report.corpus_digest_hex);
    let _ = writeln!(out, "  }},");
    let _ = writeln!(out, "  \"scenarios\": [");

    for (si, sc) in report.scenario_results.iter().enumerate() {
        let _ = writeln!(out, "    {{");
        let _ = writeln!(out, "      \"name\": \"{}\",", json_escape(&sc.name));
        let _ = writeln!(
            out,
            "      \"description\": \"{}\",",
            json_escape(&sc.description)
        );
        let _ = writeln!(
            out,
            "      \"is_negative_regime\": {},",
            sc.is_negative_regime
        );
        let _ = writeln!(
            out,
            "      \"total_cache_bytes\": {},",
            sc.total_cache_bytes
        );
        let _ = writeln!(out, "      \"host_cpu_cores\": {},", sc.host_cpu_cores);
        let _ = writeln!(out, "      \"policies\": [");

        for (pi, p) in sc.policies.iter().enumerate() {
            write_policy_json(&mut out, p);
            if pi + 1 < sc.policies.len() {
                let _ = writeln!(out, "        }},");
            } else {
                let _ = writeln!(out, "        }}");
            }
        }

        let _ = writeln!(out, "      ]");
        if si + 1 < report.scenario_results.len() {
            let _ = writeln!(out, "    }},");
        } else {
            let _ = writeln!(out, "    }}");
        }
    }

    let _ = writeln!(out, "  ]");
    let _ = writeln!(out, "}}");
    out
}

fn write_policy_json(out: &mut String, p: &PolicyRunMetrics) {
    let _ = writeln!(out, "        {{");
    let _ = writeln!(out, "          \"policy\": \"{}\",", json_escape(&p.policy));
    let _ = writeln!(out, "          \"total_examples\": {},", p.total_examples);
    let _ = writeln!(out, "          \"makespan_ms\": {:.3},", p.makespan_ms);
    let _ = writeln!(out, "          \"wall_time_ms\": {:.3},", p.wall_time_ms);
    let _ = writeln!(
        out,
        "          \"aggregate_examples_per_sec\": {:.2},",
        p.aggregate_examples_per_sec
    );
    let _ = writeln!(
        out,
        "          \"gpu_idle_time_pct\": {:.2},",
        p.gpu_idle_time_pct
    );
    let _ = writeln!(
        out,
        "          \"encoded_hit_rate_pct\": {:.2},",
        p.encoded_hit_rate_pct
    );
    let _ = writeln!(
        out,
        "          \"decoded_hit_rate_pct\": {:.2},",
        p.decoded_hit_rate_pct
    );
    let _ = writeln!(
        out,
        "          \"transformed_hit_rate_pct\": {:.2},",
        p.transformed_hit_rate_pct
    );
    let _ = writeln!(
        out,
        "          \"overall_cache_hit_rate_pct\": {:.2},",
        p.overall_cache_hit_rate_pct
    );
    let _ = writeln!(
        out,
        "          \"cache_budget_bytes\": {},",
        p.cache_budget_bytes
    );
    let _ = writeln!(
        out,
        "          \"cache_peak_used_bytes\": {},",
        p.cache_peak_used_bytes
    );
    let _ = writeln!(
        out,
        "          \"cache_avg_utilization_pct\": {:.2},",
        p.cache_avg_utilization_pct
    );
    let _ = writeln!(
        out,
        "          \"cache_peak_utilization_pct\": {:.2},",
        p.cache_peak_utilization_pct
    );
    let _ = writeln!(
        out,
        "          \"s3_bytes_fetched\": {},",
        p.s3_bytes_fetched
    );
    let _ = writeln!(out, "          \"s3_bytes_saved\": {},", p.s3_bytes_saved);
    let _ = writeln!(
        out,
        "          \"s3_bytes_saved_pct\": {:.2},",
        p.s3_bytes_saved_pct
    );
    let _ = writeln!(
        out,
        "          \"stochastic_checks_total\": {},",
        p.stochastic_checks_total
    );
    let _ = writeln!(
        out,
        "          \"stochastic_checks_passed\": {},",
        p.stochastic_checks_passed
    );
    let _ = writeln!(
        out,
        "          \"stochastic_freshness_pct\": {:.2},",
        p.stochastic_freshness_pct
    );
    let _ = writeln!(
        out,
        "          \"cross_tenant_leaks\": {},",
        p.cross_tenant_leaks
    );
    let _ = writeln!(
        out,
        "          \"unauthorized_probes_blocked\": {},",
        p.unauthorized_probes_blocked
    );
    let _ = writeln!(out, "          \"rebalance_count\": {},", p.rebalance_count);
    let _ = writeln!(
        out,
        "          \"streaming_bypass_activations\": {},",
        p.streaming_bypass_activations
    );
    let _ = writeln!(out, "          \"per_job\": [");
    for (ji, j) in p.per_job.iter().enumerate() {
        write_job_json(out, j);
        if ji + 1 < p.per_job.len() {
            let _ = writeln!(out, "            }},");
        } else {
            let _ = writeln!(out, "            }}");
        }
    }
    let _ = writeln!(out, "          ]");
}

fn write_job_json(out: &mut String, j: &JobMetrics) {
    let _ = writeln!(out, "            {{");
    let _ = writeln!(out, "              \"job_id\": {},", j.job_id);
    let _ = writeln!(out, "              \"tenant_id\": {},", j.tenant_id);
    let _ = writeln!(out, "              \"name\": \"{}\",", json_escape(&j.name));
    let _ = writeln!(
        out,
        "              \"modality\": \"{}\",",
        json_escape(&j.modality)
    );
    let _ = writeln!(
        out,
        "              \"visibility\": \"{}\",",
        json_escape(&j.visibility)
    );
    let _ = writeln!(
        out,
        "              \"stochastic_transform\": {},",
        j.stochastic_transform
    );
    let _ = writeln!(
        out,
        "              \"examples_processed\": {},",
        j.examples_processed
    );
    let _ = writeln!(out, "              \"makespan_ms\": {:.3},", j.makespan_ms);
    let _ = writeln!(
        out,
        "              \"gpu_idle_pct\": {:.2},",
        j.gpu_idle_pct
    );
    let _ = writeln!(
        out,
        "              \"examples_per_sec\": {:.2},",
        j.examples_per_sec
    );
    let _ = writeln!(out, "              \"encoded_hits\": {},", j.encoded_hits);
    let _ = writeln!(out, "              \"decoded_hits\": {},", j.decoded_hits);
    let _ = writeln!(
        out,
        "              \"transformed_hits\": {},",
        j.transformed_hits
    );
    let _ = writeln!(
        out,
        "              \"storage_fetches\": {},",
        j.storage_fetches
    );
    let _ = writeln!(
        out,
        "              \"s3_bytes_fetched\": {},",
        j.s3_bytes_fetched
    );
    let _ = writeln!(
        out,
        "              \"s3_bytes_saved\": {},",
        j.s3_bytes_saved
    );
    let _ = writeln!(
        out,
        "              \"avg_encoded_bytes\": {:.1},",
        j.avg_encoded_bytes
    );
    let _ = writeln!(
        out,
        "              \"avg_decoded_bytes\": {:.1},",
        j.avg_decoded_bytes
    );
    let _ = writeln!(
        out,
        "              \"avg_transformed_bytes\": {:.1},",
        j.avg_transformed_bytes
    );
    let _ = writeln!(
        out,
        "              \"decode_expansion_ratio\": {:.2},",
        j.decode_expansion_ratio
    );
    let _ = writeln!(
        out,
        "              \"final_budget_encoded\": {},",
        j.final_budget_encoded
    );
    let _ = writeln!(
        out,
        "              \"final_budget_decoded\": {},",
        j.final_budget_decoded
    );
    let _ = writeln!(
        out,
        "              \"final_budget_transformed\": {}",
        j.final_budget_transformed
    );
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}
