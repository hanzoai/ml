# HIP-1345: Shape-Specialised Kernel Autotuning Laboratory (`hanzoai/ml#41`)

## 1. Provenance & Clean-Room License Audit

- **Target Issue**: `hanzoai/ml#41` (*Kernel laboratory: automatic shape-specialised tuning for CUDA, HIP and Metal*)
- **Candidate Research References**: TileLang (MIT), NVIDIA CUTLASS/CuTe (BSD-3-Clause), AMD AITER (MIT) — treated strictly as study/design references without adding wholesale external runtime dependencies.
- **Implementation**: 100% original clean-room Rust (`MIT OR Apache-2.0`) inside `hanzo-research/src/autotune_lab.rs` and `hanzo-research/tests/autotune_lab.rs` with `0` new crate dependencies.

## 2. Architecture & End-to-End Promotion Gate

1. **Multi-Backend Hardware Constraint Pruning (`check_hardware_feasibility`)**:
   - Enforces per-target shared-memory/LDS/threadgroup byte limits (`160 KB` `sm_90a` CUDA, `64 KB` `gfx1151` HIP, `32 KB` `apple_m4_max_gpu40` Metal), per-block register file ceilings, and quantization group-size alignment (`Int4Group128`, `Int4Group64`, `Int8Group128`, `Bf16`) prior to candidate evaluation.
2. **Microbenchmark-Trap Rejection (`evaluate_end_to_end_gate`)**:
   - Evaluates both isolated single-GEMM latency and the full **20-turn coding-agent loop** (`512`-token cold prompt + `19` warm cached turns x `96` decode tokens across `36`/`42`/`46` stacked transformer layers).
   - Explicitly classifies and rejects `PromotionVerdict::RejectedMicrobenchOnlyTrap` schedules (such as `split_k = 8` schedules that accelerate isolated single-op GEMMs via idle-CU parallelism while degrading full-network 20-turn latency through L2 cache eviction and extra reduction barrier launches).
3. **Versioned Reproducible Backend Export (`ExportedBackendPlan`)**:
   - Exports a SHA-256 content-addressed tuning plan pinning `(arch_name, driver_version, quant_format, git_commit)` and renders static zero-dependency Rust dispatch code (`render_rust_dispatch_table`).

## 3. Shared `4B` / `9B` / `27B` 20-Turn Coding-Agent Scorecard (`Int4Group128`)

| Target & Driver | Model | Winning Schedule | Cold / Warm TTFT (ms) | p50 / p95 ITL (ms) | 20-Turn Wall Time (ms, Baseline -> Tuned) | Peak Mem (MiB) | Energy / Token | Microbench Traps |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | :--- |
| `sm_90a` (`CUDA-12.6-r560`) | `zen-4b` | `t16x128x32_w4_s3_sk1_swiglu_rms_m8` | `24.09 / 2.63` | `1.11 / 1.23` | `4542.1` -> **`2207.5` (`2.06x`)** | `2650.6` | **`-56.4%`** (`0.6358 J`) | `110` rejected |
| `sm_90a` (`CUDA-12.6-r560`) | `zen-9b` | `t16x128x32_w4_s3_sk1_swiglu_rms_m8` | `30.44 / 5.83` | `1.98 / 2.20` | `7888.8` -> **`3939.2` (`2.00x`)** | `4748.6` | **`-55.2%`** (`1.1346 J`) | `102` rejected |
| `sm_90a` (`CUDA-12.6-r560`) | `zen-27b` | `t16x128x32_w4_s3_sk1_swiglu_rms_m8` | `44.13 / 12.70` | `3.73 / 4.14` | `14587.8` -> **`7449.5` (`1.96x`)** | `10710.1` | **`-54.2%`** (`2.1456 J`) | `74` rejected |
| `gfx1151` (`ROCm-6.4.0`) | `zen-4b` | `t16x128x32_w4_s3_sk1_swiglu_rms_m8` | `93.43 / 37.12` | `8.16 / 9.05` | `31664.7` -> **`16460.7` (`1.92x`)** | `2650.6` | **`-53.3%`** (`0.8127 J`) | `59` rejected |
| `gfx1151` (`ROCm-6.4.0`) | `zen-9b` | `t16x128x32_w4_s3_sk1_swiglu_rms_m8` | `194.90 / 87.57` | `18.44 / 20.47` | `71113.7` -> **`37262.7` (`1.91x`)** | `4748.6` | **`-53.0%`** (`1.8398 J`) | `55` rejected |
| `gfx1151` (`ROCm-6.4.0`) | `zen-27b` | `t16x128x32_w4_s3_sk2_swiglu_rms_m8` | `415.42 / 197.10` | `40.08 / 44.49` | `156307.9` -> **`81110.0` (`1.93x`)** | `10713.1` | **`-53.4%`** (`4.0048 J`) | `54` rejected |
| `apple_m4_max_gpu40` (`Metal-3.2`) | `zen-4b` | `t16x128x32_w4_s3_sk1_swiglu_rms_m8` | `130.14 / 54.06` | `4.17 / 4.62` | `17436.9` -> **`9155.4` (`1.90x`)** | `2650.6` | **`-52.9%`** (`0.2938 J`) | `37` rejected |
| `apple_m4_max_gpu40` (`Metal-3.2`) | `zen-9b` | `t16x128x32_w4_s3_sk1_swiglu_rms_m8` | `282.20 / 127.87` | `9.04 / 10.04` | `37690.7` -> **`20075.4` (`1.88x`)** | `4748.6` | **`-52.2%`** (`0.6443 J`) | `35` rejected |
| `apple_m4_max_gpu40` (`Metal-3.2`) | `zen-27b` | `t16x128x32_w4_s3_sk1_swiglu_rms_m8` | `613.69 / 288.77` | `19.51 / 21.65` | `81094.3` -> **`43550.2` (`1.86x`)** | `10710.1` | **`-51.8%`** (`1.3977 J`) | `32` rejected |
