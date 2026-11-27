# HIP-1349 Research Dossier: Multi-Stage Training Data-Pipeline Cache Allocation (Seneca)

- **Tracking Issue**: [`hanzoai/ml#46`](https://github.com/hanzoai/ml/issues/46) (`p2`, `bounty`, `perf`, `research`, `hip-1349`, `training`)
- **Canonical Owning Repository**: `hanzoai/ml` (workspace crate [`hanzo-seneca`](../../../hanzo-seneca))
- **Raw Measurement Artifact**: [`docs/research/hip-1349-seneca/results.json`](./results.json)
- **Implementation License**: `MIT OR Apache-2.0` (100% clean-room pure Rust `std`, zero external crate dependencies, `#![forbid(unsafe_code)]`)
- **Recommended Disposition**: Research bounty complete; production default-on rollout **`GATED`** on IP/patent review of the 2026 Seneca patent application (see Section 1) and internal GPU cluster qualification on Hanzo `vfs -> Hanzo S3 -> Visor` training clusters.

---

## 1. Primary Paper Link, Summary, and License / Patent Audit

### Primary Reference
- **Paper**: Omkar Desai, Ziyang Jiao, Shuyi Pei, Janki Bhimani, and Bryan S. Kim, *"Preparation Meets Opportunity: Enhancing Data Preprocessing for ML Training With Seneca"*, in **24th USENIX Conference on File and Storage Technologies (FAST '26)**, February 24–26, 2026, Santa Clara, CA, USA, pp. 221–238.
- **Links**:
  - USENIX Open Access: <https://www.usenix.org/conference/fast26/presentation/desai>
  - Preprint: <https://arxiv.org/abs/2511.13724>
  - Upstream prototype (referenced for audit only; **NOT imported**): <https://github.com/swiftomkar/seneca-fast26-pytorch>

### Paper Summary & Attribution Boundary
Seneca addresses data storage and ingestion (DSI) bottlenecks when multiple ML training jobs share host memory and preprocessing CPUs. The paper combines two mechanisms:
1. **Model-Driven Partitioning (MDP)**: partitions shared cache memory across three data representations — **Encoded** (compressed storage form), **Decoded** (in-memory tensor form), and **Augmented / Transformed** (post-preprocessing form) — using a pipeline throughput model.
2. **Opportunistic Data Sampling (ODS)**: opportunistically prioritizes already-cached unvisited samples during random epoch batch sampling so concurrent jobs sharing a dataset benefit from each other's cache fills while visiting every sample once per epoch.

> **Attribution Notice**: Desai et al. (FAST '26) report on their own PyTorch + Redis testbed a 45.23% makespan reduction over PyTorch and up to 3.45× higher DSI throughput than the next best dataloader. Those are **the paper authors' published claims on their hardware**, NOT Hanzo claims. All Hanzo claims in this dossier are restricted strictly to the clean-room Rust measurements reproduced in Section 6 and [`results.json`](./results.json).

### License, Dependency, Corpus, and Patent Audit

| Artifact / Component | Status | License / Legal Findings | Action Taken |
|---|---|---|---|
| Upstream `seneca-fast26-pytorch` (PyTorch/C++ + Redis) | **Not imported** | Research prototype; depends on PyTorch + external Redis daemon | Implemented 100% clean-room in pure Rust `std` (`hanzo-seneca`) |
| `hanzo-seneca` crate code | **Clean-room** | `MIT OR Apache-2.0` | Zero external crates (`[dependencies]` empty), `#![forbid(unsafe_code)]` |
| Evaluation corpus (`hanzoai/ml` workspace `.rs/.md/.toml`) | **Cleared** | `MIT OR Apache-2.0` (434 real public workspace files, 8.02 MiB, digest `c5ea90581ca4a4a86bd8aeecb6c4f258`) | Deterministically loaded offline directly from workspace; zero private/confidential data |
| **Patent Notice Check** | **Flagged for IP review** | Public author disclosures list a 2026 US Patent Application by O. Desai, S. Pei, J. Bhimani, and B. S. Kim (*"Cache systems and methods for training machine learning models"*) | Flagged explicitly per issue #46 & `PR_CONVERGENCE.md`; keep `hanzo-seneca` opt-in / research-gated until legal/IP clearance approves commercial production rollout |

---

## 2. Reproducible Baseline & Measurement Environment

### Hardware & Toolchain
- **Host OS**: Linux `6.18.57` `x86_64` (Ubuntu / NixOS devbox container)
- **CPU**: Intel Xeon CPU @ 2.30 GHz (2 physical/logical cores online, `RUSTFLAGS="-C target-cpu=native"`)
- **Host Memory**: 7.8 GiB RAM
- **Rust Toolchain**: `rustc 1.100.0 (0ef1bc386 2026-11-09)`, `cargo 1.100.0`
- **Offline Build**: `hanzo-seneca` has zero external dependencies; once the workspace `Cargo.lock` exists, every command runs completely offline (`--offline`).

### Exact Reproduction Commands

```bash
# 1. Formatting & lint checks (zero warnings)
cargo fmt -p hanzo-seneca -- --check
cargo clippy -p hanzo-seneca --all-targets --offline -- -D warnings

# 2. Unit + integration test suite (27 tests: memory bounds, tenant isolation,
#    stochastic freshness, partial-batch epochs, linear contention, decoder-config
#    keying, bounded HLL repeat tracking, concurrency, and all 4 benchmark regimes)
cargo test -p hanzo-seneca --release --offline

# 3. Reproducible 4-scenario × 5-policy benchmark (median of 3 interleaved trials, seed 1349)
cargo run -p hanzo-seneca --release --offline --bin seneca-bench -- \
  --seed 1349 --trials 3 --output docs/research/hip-1349-seneca/results.json
```

### Five Matched Policies in the Exact Same Harness
All 5 policies execute the **exact same** 3-stage pipeline (`FetchEncoded -> Decode -> Transform -> GPU Batch Consume`) over the same real LZ/RLE compressed shards derived from the `hanzoai/ml` workspace corpus, with real `Instant::now()` CPU timing on every compression/decompression, tensor decode, augmentation/spectrogram transform, CRC32 integrity check, and cache operation, paired with identical calibrated S3/VFS RTT/bandwidth and GPU step queues:

1. **`no_cache`**: Direct S3/VFS streaming baseline (0 bytes cached).
2. **`encoded_only`**: Traditional OS page-cache / S3 shard cache only (100% of `TotalCacheBytes` partitioned to `Encoded`).
3. **`decoded_only`**: Caches decoded tensors only (100% of `TotalCacheBytes` partitioned to `Decoded`).
4. **`static_equal_split`**: Fixed 33% / 33% / 33% split across `{Encoded, Decoded, Transformed}` and equally across concurrent scopes.
5. **`seneca_dynamic`**: Seneca marginal-utility dynamic multi-stage partitioner with strictly convex GPU-stall minimization, stall-gated stage upgrades, smooth EMA hysteresis + deadband, stochastic-augmentation guard (`Transformed = 0` for stochastic jobs), opportunistic cache-aware sampling (ODS), and streaming bypass for zero-reuse 1-epoch workloads.

---

## 3. Falsifiable Hypothesis, Primary Metrics, and Expected Failure Modes

### Primary Falsifiable Hypothesis
> **H1 (Multi-Stage Dynamic Allocation)**: Under a fixed host-RAM/NVMe cache budget shared by concurrent heterogeneous ML training jobs, marginal-utility dynamic partitioning across `{Encoded, Decoded, Transformed}` (`seneca_dynamic`) will achieve **strictly lower GPU idle time (%)** and **strictly higher aggregate training examples/sec** than every static single-stage or fixed-split baseline (`no_cache`, `encoded_only`, `decoded_only`, `static_equal_split`) across I/O-bound, CPU-decode/transform-bound, and high-expansion tight-RAM regimes — while maintaining `100.0%` stochastic-augmentation freshness, `0` cross-tenant leaks, and `< 2%` throughput overhead in the zero-reuse 1-epoch streaming negative regime.

### Falsification Criteria
H1 would be **falsified** if any of the following occurred:
- `seneca_dynamic` failed to beat `encoded_only` by >= 5 percentage points of GPU idle time in `cpu_bottlenecked_heavy_decode`;
- `seneca_dynamic` failed to beat `decoded_only` and `static_equal_split` by >= 15 percentage points of GPU idle time in `tight_ram_high_expansion`;
- `seneca_dynamic` degraded throughput by `>= 2%` relative to `no_cache` in `single_epoch_streaming_no_reuse`;
- Any stochastic-augmentation job served a stale cached `Transformed` tensor (`freshness < 100%`), any private dataset leaked across tenants (`cross_tenant_leaks > 0`), or cache memory exceeded `TotalCacheBytes`.

### Expected Failure Modes & Boundary Conditions
1. **Single-epoch zero-reuse streaming (`working_set >> cache`)**: No cache policy can achieve hits (`0.0%` hit rate). Without streaming bypass, caching degrades performance due to memory copy and LRU eviction churn. Evaluated explicitly in Scenario 4 (`single_epoch_streaming_no_reuse`).
2. **Mid-training stage upgrade thrashing**: If a dynamic partitioner upgrades a warm `Encoded` partition to `Transformed` for a tiny non-stall CPU tie-breaker when another job completes, evicting the warm `Encoded` entries forces cold S3 re-fetches. We observed this exact failure mode during development and eliminated it by gating hierarchical stage upgrades on genuine GPU-stall reduction (`base_stall - cand_stall > 0`).
3. **Cold-start hysteresis lag**: Blending the first profiled marginal-utility solve with the cold-start `Encoded` bootstrap delays Epoch-0 population of `Decoded`/`Transformed` partitions. Solved by separating cold-start bootstrap transition from steady-state hysteresis blending (`profiled_initialized`).
4. **Heavy stochastic augmentations with cheap decode**: When `t_transform >> t_decode` AND `stochastic_transform = true`, caching cannot skip the stochastic transform itself (doing so would corrupt training convergence), bounding the achievable speedup to eliminating `t_fetch + t_decode`.

---

## 4. Bounded Implementation & Canonical Architecture Integration Plan

### Crate Layout (`hanzo-seneca/`)
- [`src/codec.rs`](../../../hanzo-seneca/src/codec.rs): Pure-Rust LZ77+RLE byte codec with framed header + IEEE 802.3 CRC32 verification, domain-separated 128-bit content hashing, and deterministic `SplitMix64` PRNG.
- [`src/cache.rs`](../../../hanzo-seneca/src/cache.rs): Exact byte-bounded `LruPartition` (`O(log N)` monotonic-tick eviction, zero `unsafe`) and `MultiStageCache` with shrink-first / expand-second rebalancing, stage-profile keying for both `Decoded` and `Transformed` entries, and strict tenant/job access checks.
- [`src/allocator.rs`](../../../hanzo-seneca/src/allocator.rs): Online outlier-damped EMA `JobTelemetry` profiler with hybrid exact + bounded 16 KiB HyperLogLog (`p = 14`) distinct-sample tracking, zero-self-contention multi-job CPU contention modeling, and `SenecaAllocator` implementing all 5 policies, strictly convex GPU-stall marginal-utility solving, stall-gated stage upgrades, hysteresis, stochastic guard, and streaming bypass.
- [`src/pipeline.rs`](../../../hanzo-seneca/src/pipeline.rs): Real 3-stage pipeline engine over `hanzoai/ml` repository corpus shards (`LlmTokenShard`, `VisionImageShard` with real `8×–10×` `u8 -> f32` decode expansion, `AudioFeatureShard` with real windowed log-mel filterbank), event-driven concurrent multi-job scheduler with single-responsibility epoch advancement, opportunistic cache-aware sampler (ODS), stochastic-freshness auditor, and active adversarial cross-tenant isolation prober.
- [`src/scenarios.rs`](../../../hanzo-seneca/src/scenarios.rs) & [`src/bin/seneca-bench.rs`](../../../hanzo-seneca/src/bin/seneca-bench.rs): 4 realistic multi-job scenarios, interleaved multi-trial median aggregation, and zero-dependency JSON report emitter.

### Canonical Hanzo Architecture Alignment (`vfs -> Hanzo S3 -> Visor`)
`hanzo-seneca` introduces **no parallel storage service, no external Redis dependency, and no duplicate training orchestrator**. It is designed as a pure-Rust embeddable policy + bounded host-cache library consumed directly by canonical Hanzo ML components:

1. **Storage boundary (`vfs -> Hanzo S3 -> Visor`)**:
   - `Encoded` cache misses fetch compressed shards through the existing `vfs -> Hanzo S3` client path; `hanzo-seneca` sits strictly in host RAM / local NVMe scratch above `vfs` and reduces redundant S3 GET bandwidth (measured `48%–64%` S3 byte savings in multi-epoch scenarios).
   - Visor/Cloud telemetry consumes `MultiStageCache` & `PolicyRunMetrics` counters (`s3_bytes_saved`, `gpu_idle_time_pct`, `cache_peak_used_bytes`, `unauthorized_probes_blocked`) for observability and quota enforcement; Cloud continues to own lifecycle and billing.
2. **Integration with `hanzo-datasets` (`hanzo-datasets/src/batcher.rs`)**:
   - Wrap the sample iterator fed into `Batcher::new_r1` / `Batcher::new_r2` with an opt-in `SenecaCachedLoader` backed by `MultiStageCache` + `SenecaAllocator`.
   - Dataset descriptors declare `DatasetVisibility::Private { tenant_id, job_id }` by default; cross-job sharing requires explicit opt-in `DatasetVisibility::SharedPublic { dataset_digest }` plus deterministic decoder/transform profile hashes.
   - Any transform pipeline containing random ops sets `stochastic_transform = true`, structurally enforcing `Transformed` cache budget = `0`.
3. **Integration with `hanzo-train` (`hanzo-train/src/cache.rs`, `dspark.rs`)**:
   - `hanzo-train/src/cache.rs` currently reads single-shard DeepSpec v2 files directly from disk (`File::open` + `read_exact` + `bf16_to_f32`).
   - Integrating `hanzo-seneca` as an optional host-RAM tier above `hanzo_train::cache::Cache` allows concurrent DiLoCo / DSpark training jobs sharing a node to partition host RAM dynamically between compressed shard blocks (`Encoded`) and decoded `f32` hidden-state tensors (`Decoded`), while bypassing cache automatically on 1-epoch pretraining streams.

---

## 5. Correctness, Concurrency, Security, and Regression Evidence

### Automated Verification Summary
| Gate | Command | Result |
|---|---|---|
| Formatting | `cargo fmt -p hanzo-seneca -- --check` | **PASS** (0 diffs) |
| Static Analysis | `cargo clippy -p hanzo-seneca --all-targets --offline -- -D warnings` | **PASS** (0 warnings) |
| Unit Tests | `cargo test -p hanzo-seneca --lib --release --offline` | **PASS** (`21/21` passed) |
| Integration Tests | `cargo test -p hanzo-seneca --test invariants --release --offline` | **PASS** (`6/6` passed) |
| Unsafe Code Audit | `#![forbid(unsafe_code)]` in `src/lib.rs`, `src/bin/seneca-bench.rs`, `tests/invariants.rs` | **0 `unsafe` blocks** |
| External Dependencies | `hanzo-seneca/Cargo.toml` `[dependencies]` | **0 external crates** (`std` only) |
| JSON Schema Validation | `jq '{seed, trials_median, scenarios: (.scenarios \| length)}' docs/research/hip-1349-seneca/results.json` | **PASS** (valid JSON, 4 scenarios × 5 policies) |

### Security & Safety Invariants Verified
- **Bounded Memory Invariant**: Every cache insertion and rebalance asserts `total_used_bytes <= total_budget_bytes`. Dynamic rebalancing always shrinks over-budget partitions before expanding receiving partitions (shrink-first, expand-second). Verified across all 20 `(scenario, policy)` runs and dedicated rebalance unit tests.
- **Strict Tenant Isolation (`0` Cross-Tenant Leaks)**: Private keys embed `(tenant_id, job_id)` and verify caller authorization on every lookup. The benchmark engine actively executes cross-tenant adversarial read probes between every pair of distinct private tenants across all 4 scenarios (`192 + 32 + 32 + 32 = 288` unauthorized probes blocked per policy, **`0` leaks**).
- **Safe Public Sharing & Decoder/Transform Keying**: `SharedPublic` entries are keyed by `(dataset_digest, sample_id, stage, stage_profile_hash)` where `Encoded` uses `0`, `Decoded` preserves `decode_profile_hash(modality, decode_compute_passes)`, and `Transformed` chains `decode_profile_hash` + `transform_profile` + `transform_compute_passes`. Two `SharedPublic` jobs with different decoder or transform configurations never cross-serve incompatible tensors (`decoded_entries_with_different_decoder_profiles_never_collide`, `shared_public_jobs_with_different_decode_passes_never_cross_serve_decoded_cache`).
- **Bounded Cardinality Repeat Tracking**: `JobTelemetry` tracks distinct sample IDs exactly up to `100,000` unique samples and promotes to a bounded 16 KiB HyperLogLog (`p = 14`) estimator with conservative `2σ` upper-bound repeat estimation thereafter, keeping zero-reuse streams `> 100k` samples at `~0` repeat ratio while preserving accurate repeat detection on large multi-epoch streams (`repeat_estimates_stay_valid_beyond_100k_unique_cap`).
- **Linear Single-Job CPU Timing & Cross-Job Contention**: Host CPU contention models processor sharing strictly against *other* active jobs (`cpu_contention_factor(other_cpu_util, host_cpu_cores)`), never charging a job for contention with its own CPU work (`single_job_does_not_contend_with_its_own_cpu_work`, `single_cpu_bound_job_has_no_quadratic_self_contention`).
- **Exact Multi-Epoch Coverage Including Partial Batches**: `refill_epoch_if_needed` is the sole owner of epoch transitions, ensuring non-multiple dataset sizes process all `dataset_len × epochs` examples without skipping epochs (`partial_batch_advances_epoch_only_once`).
- **Stochastic-Augmentation Correctness (`100.0%` Fresh)**: Every epoch `e > 0` sample of every stochastic job is checked against all previously seen transformed CRC32 digests for that sample: `288/288` (Scenario 1), `336/336` (Scenario 2), `336/336` (Scenario 3) = **`960/960` (`100.0%`) fresh**, with `0` `Transformed` hits and `0` `Transformed` budget under `seneca_dynamic`.
- **Integrity & Concurrency**: Every LZ/RLE frame checks magic, exact uncompressed length, safe back-reference bounds, and IEEE 802.3 CRC32 (`lz_rle_detects_corruption`). Multi-threaded execution across 4 concurrent OS threads verified in `concurrent_threads_execute_deterministically_with_zero_leaks`.

---

## 6. Before/After Measurements, Raw Results, and HIP-1349 Update

All tables below are extracted directly from [`docs/research/hip-1349-seneca/results.json`](./results.json) (`seed=1349`, median of `3` interleaved trials, real `hanzoai/ml` corpus: 434 files / 8.02 MiB).

### Scenario 1: `mixed_concurrent_llm_and_vision`
*4 concurrent heterogeneous jobs (I/O-bound LLM shards, decode-heavy deterministic vision, transform-heavy audio mel-spectrogram, and stochastic-augmentation vision) sharing a 2,099 KiB (2,150,000 B) cache on 2 host CPU cores.*

| Policy | GPU Idle (%) | Aggregate Ex/sec | Enc Hit (%) | Dec Hit (%) | Trn Hit (%) | Avg Cache Util (%) | S3 Saved (%) | Stochastic Fresh (%) | Tenant Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cache` | 72.61% | 5,012.4 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0 |
| `encoded_only` | 61.03% | 6,959.5 | 75.0% | 0.0% | 0.0% | 15.3% | 75.0% | 100.0% | 0 |
| `decoded_only` | 70.09% | 6,182.0 | 0.0% | 38.4% | 0.0% | 77.5% | 37.4% | 100.0% | 0 |
| `static_equal_split` | 63.65% | 7,707.5 | 42.6% | 5.3% | 20.9% | 70.2% | 63.9% | 100.0% | 0 |
| **`seneca_dynamic`** | **54.16%** | **8,913.0** | **26.1%** | **13.2%** | **23.9%** | **44.8%** | **61.4%** | **100.0%** | **0** |

**Key takeaway**: `seneca_dynamic` cuts GPU idle time by **-18.45 pp** vs `no_cache` (**1.78×** throughput), **-9.49 pp** vs `static_equal_split` (**+15.6%** throughput), **-6.87 pp** vs `encoded_only` (**+28.1%** throughput), and **-15.93 pp** vs `decoded_only` (**1.44×** throughput), dynamically using all three cache stages simultaneously while enforcing `0 B` `Transformed` cache and `100.0%` (`288/288`) augmentation freshness on `vision_stochastic_aug`.

### Scenario 2: `cpu_bottlenecked_heavy_decode`
*Fast NVMe/local storage (`20 µs` RTT, `500 MiB/s`) with CPU decode/transform saturating 2 host cores across 3 concurrent jobs (2 jobs sharing a `SharedPublic` vision dataset — one deterministic, one stochastic — plus 1 private heavy-transform audio job) sharing 1,806 KiB (1,850,000 B).*

| Policy | GPU Idle (%) | Aggregate Ex/sec | Enc Hit (%) | Dec Hit (%) | Trn Hit (%) | Avg Cache Util (%) | S3 Saved (%) | Stochastic Fresh (%) | Tenant Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cache` | 77.90% | 6,291.0 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0 |
| `encoded_only` | 78.43% | 6,145.6 | 83.3% | 0.0% | 0.0% | 7.0% | 85.8% | 100.0% | 0 |
| `decoded_only` | 76.93% | 6,824.6 | 0.0% | 50.4% | 0.0% | 66.0% | 43.3% | 100.0% | 0 |
| `static_equal_split` | 72.70% | 7,374.2 | 51.3% | 5.2% | 26.9% | 61.3% | 85.8% | 100.0% | 0 |
| **`seneca_dynamic`** | **59.53%** | **13,356.4** | **0.1%** | **6.8%** | **46.1%** | **52.1%** | **48.3%** | **100.0%** | **0** |

**Key takeaway**: Because storage is already fast, `encoded_only` leaves GPUs **78.43% idle** despite an 83.3% hit rate. `seneca_dynamic` shifts budget into `Transformed` (for deterministic vision & audio) and `Decoded` (for the stochastic vision job sharing the public dataset), cutting GPU idle time by **-18.90 pp** vs `encoded_only` and boosting aggregate throughput **2.12×** vs `no_cache`, **2.17×** vs `encoded_only`, and **1.81×** vs `static_equal_split`.

### Scenario 3: `tight_ram_high_expansion`
*2 concurrent vision jobs with real `8.3×–9.5×` decoded-to-encoded memory expansion (`~1.7 KiB` LZ/RLE encoded -> `15,552 B` `f32` decoded), slow S3 storage (`1,100 µs` RTT, `20 MiB/s`), and a tight 344 KiB (353,137 B) host RAM budget (~90% of encoded working set, ~11% of decoded working set).*

| Policy | GPU Idle (%) | Aggregate Ex/sec | Enc Hit (%) | Dec Hit (%) | Trn Hit (%) | Avg Cache Util (%) | S3 Saved (%) | Stochastic Fresh (%) | Tenant Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `no_cache` | 65.47% | 3,054.1 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 100.0% | 0 |
| `encoded_only` | 42.13% | 5,083.7 | 48.2% | 0.0% | 0.0% | 77.3% | 48.3% | 100.0% | 0 |
| `decoded_only` | 67.98% | 2,838.9 | 0.0% | 0.1% | 0.0% | 95.2% | 0.1% | 100.0% | 0 |
| `static_equal_split` | 67.27% | 2,889.1 | 3.0% | 0.0% | 0.0% | 70.9% | 3.0% | 100.0% | 0 |
| **`seneca_dynamic`** | **38.23%** | **5,437.2** | **64.3%** | **0.0%** | **0.0%** | **48.9%** | **64.1%** | **100.0%** | **0** |

**Key takeaway**: Under ~9× expansion and tight RAM, `decoded_only` fills 95.2% of RAM yet achieves only **0.1%** hits (catastrophic LRU thrashing, performing *worse* than `no_cache` at 67.98% GPU idle). `seneca_dynamic` automatically shifts budget to compact `Encoded` shards and pins hot shards against multi-epoch scan thrashing, achieving **64.3%** hit rate, **64.1%** S3 bandwidth saved, **-29.75 pp** GPU idle reduction vs `decoded_only`, and **1.92×** throughput vs `decoded_only` (**1.88×** vs `static_equal_split`).

### Scenario 4 (Explicit Negative Regime): `single_epoch_streaming_no_reuse`
*Massive 1-epoch streaming corpus (2,048 unique shards across 2 concurrent private jobs, working set >14× larger than the 256 KiB cache budget, zero sample reuse).*

| Policy | GPU Idle (%) | Aggregate Ex/sec | Overall Hit (%) | Avg Cache Util (%) | S3 Saved (%) | Streaming Bypass | Tenant Leaks |
|---|---:|---:|---:|---:|---:|---:|---:|
| `no_cache` | 13.25% | 27,690.8 | 0.0% | 0.0% | 0.0% | N/A | 0 |
| `encoded_only` | 14.20% | 27,447.8 | 0.0% | 95.6% | 0.0% | N/A | 0 |
| `decoded_only` | 26.70% | 23,444.3 | 0.0% | 98.1% | 0.0% | N/A | 0 |
| `static_equal_split` | 28.25% | 22,916.2 | 0.0% | 96.2% | 0.0% | N/A | 0 |
| **`seneca_dynamic`** | **14.33%** | **27,406.9** | **0.0%** | **0.0%** | **0.0%** | **Active (6×)** | **0** |

**Negative Regime Verdict**: As hypothesized, caching cannot improve hit rate (`0.0%` across all 5 policies) when a dataset is streamed once with `working_set >> cache`. Naive policies burn 95%–98% of host RAM on useless LRU churn (`decoded_only` and `static_equal_split` degrade throughput by 15%–17%). `seneca_dynamic` detects zero effective reuse, activates streaming bypass (`cache_peak_used_bytes = 0`), frees 100% of host RAM for OS/training buffers, and incurs only **1.03% throughput overhead** (`27,406.9` vs `27,690.8` ex/sec — well under the `< 2%` falsification bound).

### Per-Job Steady-State Stage Allocation Under `seneca_dynamic`

| Scenario | Job | Modality | Visibility | Stochastic? | Decode Exp. | Hits (E / D / T) | Steady Budget (E / D / T) | Job Ex/sec |
|---|---|---|---|---|---:|---|---:|---:|
| `mixed_concurrent_llm_and_vision` | `llm_io_bound` | `llm_token_shard` | `private` | No | 2.21× | 75 / 162 / 72 | 154.3 / 757.1 / 994.8 KiB | 2,742.5 |
| `mixed_concurrent_llm_and_vision` | `vision_decode_heavy` | `vision_image_shard` | `private` | No | 9.86× | 189 / 0 / 55 | 119.9 / 0 / 353.3 KiB | 3,879.7 |
| `mixed_concurrent_llm_and_vision` | `audio_transform_heavy` | `audio_feature_shard` | `private` | No | 10.00× | 0 / 0 / 271 | 0 / 0 / 126.0 KiB | 3,014.0 |
| `mixed_concurrent_llm_and_vision` | `vision_stochastic_aug` | `vision_image_shard` | `private` | **Yes** | 8.24× | 170 / 57 / **0** | 134.3 / 1142.8 / **0 KiB** | 2,921.6 |
| `cpu_bottlenecked_heavy_decode` | `shared_vision_det` | `vision_image_shard` | `shared_public` | No | 8.96× | 0 / 33 / 310 | 0 / 89.1 / 1344.0 KiB | 5,456.9 |
| `cpu_bottlenecked_heavy_decode` | `shared_vision_stochastic` | `vision_image_shard` | `shared_public` | **Yes** | 8.48× | 1 / 58 / **0** | 0 / 294.6 / **0 KiB** | 6,752.3 |
| `cpu_bottlenecked_heavy_decode` | `private_audio_heavy_transform` | `audio_feature_shard` | `private` | No | 11.53× | 0 / 0 / 310 | 0 / 0 / 168.0 KiB | 4,452.1 |
| `tight_ram_high_expansion` | `vision_highexp_a` | `vision_image_shard` | `private` | No | 9.47× | 294 / 0 / 0 | 175.2 / 0 / 0 KiB | 2,772.8 |
| `tight_ram_high_expansion` | `vision_highexp_b` | `vision_image_shard` | `private` | **Yes** | 8.29× | 282 / 0 / **0** | 183.7 / 64.9 / **0 KiB** | 2,718.6 |
| `single_epoch_streaming_no_reuse` | `llm_stream_1epoch_a` | `llm_token_shard` | `private` | No | 2.11× | 0 / 0 / 0 | 0 / 0 / 0 KiB | 13,710.2 |
| `single_epoch_streaming_no_reuse` | `llm_stream_1epoch_b` | `llm_token_shard` | `private` | No | 1.86× | 0 / 0 / 0 | 0 / 0 / 0 KiB | 13,703.5 |

### HIP-1349 Status Update & Rollout / Rollback Plan
- **HIP-1349 Status**: Evidence-backed clean-room prototype and benchmark harness delivered (`hanzo-seneca`). Hypothesis **H1 is supported** across all 3 positive regimes and cleanly bounded in the 1-epoch streaming negative regime.
- **Rollout Plan**:
  1. Merge `hanzo-seneca` as an opt-in workspace crate (zero impact on existing `hanzo-ml`, `hanzo-datasets`, or `hanzo-train` default code paths).
  2. Complete legal/IP review of the 2026 Desai et al. patent application before enabling Seneca partitioning by default in commercial Hanzo Cloud training fleets.
  3. Run confidential internal qualification on real multi-GPU Hanzo `vfs -> Hanzo S3` training clusters (publishing aggregate-only receipts per `PR_CONVERGENCE.md`).
  4. Wire opt-in `--pipeline-cache=seneca` integration behind a feature flag in `hanzo-datasets` and `hanzo-train`.
- **Rollback Plan**: Single-commit revert of `hanzo-seneca/` and the 1-line `Cargo.toml` workspace entry; once wired into `hanzo-datasets` / `hanzo-train`, toggling `--pipeline-cache=off` (`AllocationPolicy::NoCache`) instantaneously restores direct S3/VFS streaming with zero state migration.
