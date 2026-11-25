# HIP-1345: Table-Lookup Ternary Inference on ARM and x86 CPUs (T-MAC Style)

- **Issue**: `hanzoai/ml#40` (tracked under benchmark umbrella `hip-1346`)
- **Crate**: `hanzo-nn` (`hanzo_nn::ternary_lut`)
- **License**: `MIT OR Apache-2.0` (100% clean-room Rust implementation)
- **Dispatch Status**: **Opt-in experimental operator** — callers explicitly construct `PackedTernaryMatrix` / `TmacLinear` or call `ternary_lut_matvec_f32` / `ternary_lut_matvec_f32_parallel`. Does **not** alter default `hanzo-ml` `PQ2_0`, `PTQ1_0`, `BitcosMatrix`, or `QMatMul` dispatch and introduces **zero** second model-loading paths.

## 1. Primary References & Clean-Room License Audit

1. **T-MAC (Study-Only Paper Reference)**: Jianyu Wei, Shijie Cao, Ting Cao, Lingxiao Ma, Lei Wang, Yanyong Zhang, and Mao Yang, *T-MAC: CPU Renaissance via Table Lookup for Low-Bit LLM Deployment on Edge*, EuroSys 2025 ([arXiv:2407.00088](https://arxiv.org/abs/2407.00088)).
2. **BitNet b1.58 Quantization Specification (Study-Only Paper Reference)**: Shuming Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits* ([arXiv:2402.17764](https://arxiv.org/abs/2402.17764)).
3. **Published Authors' Claims vs. Measured Hanzo Numbers**: Wei et al. report 2x-4x low-bit CPU inference speedups on edge ARM and x86 processors relative to weight-unpacking kernels. Per `hanzoai/ml#40`, published speedups are treated strictly as workload-specific hypotheses and verified directly against `hanzo-ml`'s canonical `PQ2_0`, `PTQ1_0`, and `BitcosMatrix` CPU implementations on identical ternary weights, per-row `f16` scales, model shapes, prompt/decode workloads, and thread settings (`1T` and `MT` via `hanzo_ml::utils::barrier_pool()`).
4. **Energy Accounting Policy**: No analytical power or energy formula (e.g., fixed-watt CPU estimates) is computed or reported. `HardwareMetadata.energy_telemetry` and `OperatorScorecardRow.measured_joules_per_token` remain `None` (`null`) unless populated from direct hardware counters (e.g., RAPL / `powermetrics`).
5. **Clean-Room & SPDX Provenance**: `hanzo-nn/src/ternary_lut.rs`, `hanzo-nn/tests/ternary_lut.rs`, and `hanzo-nn/examples/tmac_vs_canonical_bench.rs` are authored 100% from scratch in clean-room Rust under `MIT OR Apache-2.0`, using only existing workspace dependencies (`hanzo-ml`, `half`, `serde`, `serde_json`). Zero code, generated C kernels, or weights are copied from external repositories.

## 2. Quantization Format, Algebraic LUT Structure & Register Tiling

### 2.1 Bit-Packed Ternary Weight Format (`PackedTernaryMatrix`)
- Weights $w \in \{-1, 0, +1\}$ are quantized deterministically from continuous `f32` tensors via BitNet b1.58 per-row absmean scaling (`quantize_absmean_ternary`, with grouped support in `quantize_absmean_ternary_grouped`):
  $$\alpha_r = \frac{1}{K}\sum_{c=0}^{K-1} |W_{r,c}|, \qquad \hat{w}_{r,c} = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{W_{r,c}}{\alpha_r}\right), -1, +1\right)$$
- Four ternary weights ($g = 4$) pack into 1 byte (`0 -> 0b00`, `+1 -> 0b01`, `-1 -> 0b10`), achieving `4.00x` payload compression vs. `INT8` and `16.00x` payload compression vs. `FP32` (`~2.006–2.013` effective bpw including per-row `f32` scales) with zero runtime weight-expansion buffers.
- Output rows are interleaved into `N_TILE = 32` row tiles (`[ceil(N / 32), ceil(K / 4), 32]`) so a single 256-bit SIMD load (or two 128-bit NEON loads) fetches 32 output channels sharing the exact same 4-activation lookup table.
- Both the full `3^4 = 81` radix-3 bijection (`trits_to_radix81` / `radix81_to_trits`) and the `41`-entry odd-symmetric mirror table (`build_symmetric_lut41` / `lookup_symmetric_lut41`, exploiting $u \mapsto 80 - u$ negation symmetry around state `40 = [0, 0, 0, 0]`) are exposed and verified bit-exact.

### 2.2 Precomputed Activation Lookup Table (`ActivationLut` & `SimdLutGroup`)
- For each input token activation $x \in \mathbb{R}^K$, `ActivationLut::from_f32` performs dynamic symmetric `i8` quantization ($\gamma_x = \max_c |x_c| / 127$, $q_c \in [-127, +127]$) and precomputes one 64-byte cache-line-aligned `SimdLutGroup` per 4 channels (`O(K)` precomputation amortized across all `N` output channels).
- Each packed byte splits into a low 2-trit nibble (`b & 0x0F`) and high 2-trit nibble (`(b >> 4) & 0x0F`), indexing 16-entry two's-complement `i16` byte tables (`lo_lsb`, `lo_msb`, `hi_lsb`, `hi_msb`).
- Inner loops accumulate `i16` partial sums in chunks of `I16_ACCUM_GROUP_CHUNK = 32` groups (`128` input channels; worst-case magnitude $32 \times 4 \times 127 = 16,256 \le 32,767$, guaranteeing zero `i16` overflow) before widening to exact `i32` accumulators.

### 2.3 Hardware-Specific SIMD & Portable Backends (`TmacKernelBackend`)
- **`Avx512WideLut` (`x86_64` with runtime `avx512bw` + `avx2` detection)**: Unrolls 4 tiles (`128` output rows simultaneously) per LUT broadcast using `vpshufb` (`_mm256_shuffle_epi8`), `_mm256_unpacklo_epi8` / `_mm256_unpackhi_epi8`, and `_mm256_add_epi16`.
- **`Avx2Lut` (`x86_64` with runtime `avx2` detection)**: Accumulates 2 tiles (`64` output rows simultaneously across 13 YMM registers) via `_mm256_shuffle_epi8` and `_mm256_add_epi16`.
- **`NeonLut` (`aarch64` with runtime `neon` detection)**: Uses ARM NEON 16-byte register table lookup `vqtbl1q_s8` paired with `vzip1q_s8` / `vzip2q_s8` byte interleaving and `vaddq_s16` / `vaddw_s16` accumulation across 32 output rows per tile.
- **`ScalarTiledLut` (portable fallback)**: Safe Rust `4x` output-row register-tiled LUT kernel with 4 independent scalar accumulators (`acc0..acc3`), verified bit-identical (`0` integer accumulator error) to all SIMD backends across odd and unaligned `(N, K)` dimensions.
- **`ternary_lut_matvec_f32_parallel`**: Parallelizes over disjoint `N_TILE = 32` output tiles via `hanzo_ml::utils::barrier_pool()` (matching canonical `QMatMul` and `BitcosMatrix` threading), verified bit-identical to single-thread `ternary_lut_matvec_f32`.

## 3. Comparative Benchmark vs Canonical `PQ2_0` / `PTQ1_0` / `BitcosMatrix`

Reproducible benchmark: `cargo run --release -p hanzo-nn --example tmac_vs_canonical_bench`.
All four formats encode the **exact same ternary weights** ($w \in \{-1, 0, +1\}$) and **identical per-row `f16` scales**, evaluated at matched thread counts (`1T` and `MT` via `barrier_pool`) against exact `FP32` ground truth.

### 3.1 Host Environment (ARM64 NEON)
- **Host**: Apple M1 Max (`aarch64-apple-darwin`, `aarch64-neon-vqtbl1q-tile32`, `barrier_pool_threads=8`)
- **Energy telemetry**: `None` (`null` — no analytical watt estimates)
- **Note on MT variance**: Captured on a shared workstation with concurrent background processes; `1T` latency exhibits low variance (`CV 0.2%–4.9%`), whereas `MT` barrier synchronization reflects host scheduler jitter (reported explicitly as `Mean +/- SD` and `CV%`).

### 3.2 Part A: Full-Shape Model Projections (`4B / 9B / 27B`, `z = 0.33` dense & `z = 0.65` sparse)

| Shape (N x K) | Zero z | Format | Mode | Weight (MB) | BPW | p50 (us) | p95 (us) | Mean +/- SD (us) | CV% | GOPS | Eff GB/s | Cos vs F32 | Max Rel Err |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4B-attn` (2560x2560) | 0.33 | `TmacLut` | 1T | 1.57 | 2.013 | 119.9 | 123.8 | 122.4 +/- 6.0 | 4.9% | 109.34 | 13.75 | 0.999993 | 3.28e-3 |
| `4B-attn` (2560x2560) | 0.33 | `PQ2_0` | 1T | 1.66 | 2.125 | 281.1 | 298.6 | 284.6 +/- 8.7 | 3.1% | 46.63 | 6.19 | 0.999994 | 2.79e-3 |
| `4B-attn` (2560x2560) | 0.33 | `PTQ1_0` | 1T | 1.37 | 1.750 | 363.7 | 382.6 | 365.5 +/- 11.9 | 3.3% | 36.04 | 3.94 | 0.999994 | 2.79e-3 |
| `4B-attn` (2560x2560) | 0.33 | `Bitcos(G128)` | 1T | 1.45 | 1.857 | 663.2 | 732.8 | 677.0 +/- 29.6 | 4.4% | 19.76 | 2.29 | 0.999994 | 2.79e-3 |
| `4B-attn` (2560x2560) | 0.33 | `TmacLut` | MT | 1.57 | 2.013 | 57.7 | 101.9 | 70.3 +/- 40.8 | 58.0% | 227.29 | 28.59 | 0.999993 | 3.28e-3 |
| `4B-attn` (2560x2560) | 0.33 | `QMatMul(PQ2_0)` | MT | 1.66 | 2.125 | 53.1 | 470.7 | 127.8 +/- 162.2 | 126.9% | 246.91 | 32.79 | 0.999994 | 2.79e-3 |
| `4B-attn` (2560x2560) | 0.33 | `QMatMul(PTQ1_0)` | MT | 1.37 | 1.750 | 61.6 | 132.1 | 90.3 +/- 96.3 | 106.6% | 212.69 | 23.26 | 0.999994 | 2.79e-3 |
| `4B-attn` (2560x2560) | 0.33 | `Bitcos(G128)` | MT | 1.45 | 1.857 | 123.8 | 156.6 | 141.8 +/- 54.6 | 38.5% | 105.92 | 12.29 | 0.999994 | 2.79e-3 |
| `4B-ffn` (6912x2560) | 0.33 | `TmacLut` | 1T | 4.25 | 2.013 | 321.8 | 347.7 | 326.3 +/- 12.6 | 3.9% | 109.98 | 13.83 | 0.999992 | 3.89e-3 |
| `4B-ffn` (6912x2560) | 0.33 | `PQ2_0` | 1T | 4.48 | 2.125 | 762.3 | 811.2 | 765.6 +/- 29.9 | 3.9% | 46.43 | 6.17 | 0.999993 | 4.39e-3 |
| `4B-ffn` (6912x2560) | 0.33 | `PTQ1_0` | 1T | 3.69 | 1.750 | 959.2 | 980.7 | 958.7 +/- 14.8 | 1.5% | 36.89 | 4.04 | 0.999993 | 4.39e-3 |
| `4B-ffn` (6912x2560) | 0.33 | `Bitcos(G128)` | 1T | 3.92 | 1.858 | 1790.5 | 1851.9 | 1798.2 +/- 34.5 | 1.9% | 19.76 | 2.29 | 0.999993 | 4.39e-3 |
| `4B-ffn` (6912x2560) | 0.33 | `TmacLut` | MT | 4.25 | 2.013 | 79.2 | 99.9 | 83.0 +/- 8.6 | 10.3% | 446.79 | 56.20 | 0.999992 | 3.89e-3 |
| `4B-ffn` (6912x2560) | 0.33 | `QMatMul(PQ2_0)` | MT | 4.48 | 2.125 | 100.9 | 176.4 | 129.2 +/- 93.1 | 72.1% | 350.68 | 46.57 | 0.999993 | 4.39e-3 |
| `4B-ffn` (6912x2560) | 0.33 | `QMatMul(PTQ1_0)` | MT | 3.69 | 1.750 | 127.0 | 417.7 | 183.3 +/- 136.6 | 74.5% | 278.56 | 30.47 | 0.999993 | 4.39e-3 |
| `4B-ffn` (6912x2560) | 0.33 | `Bitcos(G128)` | MT | 3.92 | 1.858 | 261.3 | 583.8 | 352.9 +/- 215.9 | 61.2% | 135.44 | 15.72 | 0.999993 | 4.39e-3 |
| `9B-ffn` (11008x4096) | 0.33 | `TmacLut` | 1T | 10.79 | 2.008 | 828.0 | 923.9 | 844.7 +/- 35.3 | 4.2% | 108.92 | 13.67 | 0.999993 | 3.80e-3 |
| `9B-ffn` (11008x4096) | 0.33 | `PQ2_0` | 1T | 11.42 | 2.125 | 1920.3 | 2025.8 | 1939.3 +/- 44.8 | 2.3% | 46.96 | 6.24 | 0.999993 | 3.10e-3 |
| `9B-ffn` (11008x4096) | 0.33 | `PTQ1_0` | 1T | 9.41 | 1.750 | 2430.7 | 2470.8 | 2432.1 +/- 38.5 | 1.6% | 37.10 | 4.06 | 0.999993 | 3.10e-3 |
| `9B-ffn` (11008x4096) | 0.33 | `Bitcos(G128)` | 1T | 9.98 | 1.857 | 4575.5 | 4719.1 | 4593.9 +/- 71.4 | 1.6% | 19.71 | 2.29 | 0.999993 | 3.10e-3 |
| `9B-ffn` (11008x4096) | 0.33 | `TmacLut` | MT | 10.79 | 2.008 | 200.8 | 604.5 | 333.2 +/- 215.4 | 64.7% | 449.02 | 56.35 | 0.999993 | 3.80e-3 |
| `9B-ffn` (11008x4096) | 0.33 | `QMatMul(PQ2_0)` | MT | 11.42 | 2.125 | 397.7 | 861.6 | 504.6 +/- 236.0 | 46.8% | 226.77 | 30.12 | 0.999993 | 3.10e-3 |
| `9B-ffn` (11008x4096) | 0.33 | `QMatMul(PTQ1_0)` | MT | 9.41 | 1.750 | 593.2 | 973.0 | 676.8 +/- 230.1 | 34.0% | 152.02 | 16.63 | 0.999993 | 3.10e-3 |
| `9B-ffn` (11008x4096) | 0.33 | `Bitcos(G128)` | MT | 9.98 | 1.857 | 857.1 | 1275.5 | 884.8 +/- 217.7 | 24.6% | 105.21 | 12.21 | 0.999993 | 3.10e-3 |
| `27B-attn` (5120x5120) | 0.65 | `TmacLut` | 1T | 6.27 | 2.006 | 468.8 | 470.9 | 469.2 +/- 0.9 | 0.2% | 111.83 | 14.02 | 0.999992 | 4.80e-3 |
| `27B-attn` (5120x5120) | 0.65 | `PQ2_0` | 1T | 6.64 | 2.125 | 1123.2 | 1208.4 | 1139.5 +/- 37.9 | 3.3% | 46.68 | 6.20 | 0.999993 | 4.51e-3 |
| `27B-attn` (5120x5120) | 0.65 | `PTQ1_0` | 1T | 5.47 | 1.750 | 1402.1 | 1438.8 | 1404.5 +/- 24.4 | 1.7% | 37.39 | 4.09 | 0.999993 | 4.51e-3 |
| `27B-attn` (5120x5120) | 0.65 | `Bitcos(G128)` | 1T | 4.80 | 1.537 | 2641.4 | 2715.5 | 2645.4 +/- 42.5 | 1.6% | 19.85 | 1.91 | 0.999993 | 4.51e-3 |
| `27B-attn` (5120x5120) | 0.65 | `TmacLut` | MT | 6.27 | 2.006 | 128.0 | 140.0 | 137.0 +/- 32.7 | 23.9% | 409.60 | 51.36 | 0.999992 | 4.80e-3 |
| `27B-attn` (5120x5120) | 0.65 | `QMatMul(PQ2_0)` | MT | 6.64 | 2.125 | 150.0 | 195.6 | 160.7 +/- 22.4 | 14.0% | 349.43 | 46.41 | 0.999993 | 4.51e-3 |
| `27B-attn` (5120x5120) | 0.65 | `QMatMul(PTQ1_0)` | MT | 5.47 | 1.750 | 233.4 | 735.5 | 367.6 +/- 214.3 | 58.3% | 224.65 | 24.57 | 0.999993 | 4.51e-3 |
| `27B-attn` (5120x5120) | 0.65 | `Bitcos(G128)` | MT | 4.80 | 1.537 | 384.7 | 575.1 | 417.3 +/- 96.4 | 23.1% | 136.28 | 13.10 | 0.999993 | 4.51e-3 |
| `27B-ffn` (13824x5120) | 0.65 | `TmacLut` | 1T | 16.93 | 2.006 | 1288.4 | 1321.0 | 1290.1 +/- 22.8 | 1.8% | 109.87 | 13.78 | 0.999992 | 4.66e-3 |
| `27B-ffn` (13824x5120) | 0.65 | `PQ2_0` | 1T | 17.93 | 2.125 | 2992.8 | 3096.2 | 3025.7 +/- 76.4 | 2.5% | 47.30 | 6.28 | 0.999993 | 4.27e-3 |
| `27B-ffn` (13824x5120) | 0.65 | `PTQ1_0` | 1T | 14.77 | 1.750 | 3814.5 | 3874.1 | 3819.6 +/- 63.2 | 1.7% | 37.11 | 4.06 | 0.999993 | 4.27e-3 |
| `27B-ffn` (13824x5120) | 0.65 | `Bitcos(G128)` | 1T | 12.97 | 1.537 | 7163.2 | 7274.0 | 7187.6 +/- 94.8 | 1.3% | 19.76 | 1.90 | 0.999993 | 4.27e-3 |
| `27B-ffn` (13824x5120) | 0.65 | `TmacLut` | MT | 16.93 | 2.006 | 243.6 | 764.7 | 373.4 +/- 231.1 | 61.9% | 581.05 | 72.86 | 0.999992 | 4.66e-3 |
| `27B-ffn` (13824x5120) | 0.65 | `QMatMul(PQ2_0)` | MT | 17.93 | 2.125 | 431.0 | 993.1 | 597.5 +/- 247.7 | 41.5% | 328.47 | 43.63 | 0.999993 | 4.27e-3 |
| `27B-ffn` (13824x5120) | 0.65 | `QMatMul(PTQ1_0)` | MT | 14.77 | 1.750 | 554.4 | 695.3 | 610.4 +/- 171.0 | 28.0% | 255.35 | 27.93 | 0.999993 | 4.27e-3 |
| `27B-ffn` (13824x5120) | 0.65 | `Bitcos(G128)` | MT | 12.97 | 1.537 | 1155.9 | 1685.2 | 1294.2 +/- 305.6 | 23.6% | 122.46 | 11.77 | 0.999993 | 4.27e-3 |

### 3.3 Part B: End-to-End 20-Turn Coding-Agent Prefill + KV-Cached Decode (200 Tokens/Run, 5 Repeated Runs)

| Profile | Format | Mode | Weight (MB) | BPW | Wall (ms, mean +/- SD) | Effective Tok/s (mean +/- SD) | CV% | Max Logit Err | Cos vs F32 | Token Match |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `4B-equiv-4L` (h=512,ffn=1536) | `FP32-exact` | 1T | 47.00 | 32.000 | 2076.67 +/- 33.16 | 96.3 +/- 1.5 | 1.6% | 0.00e0 | 1.000000 | 1.000 |
| `4B-equiv-4L` (h=512,ffn=1536) | `TmacLut` | 1T | 3.01 | 2.052 | 75.53 +/- 2.78 | 2650.8 +/- 97.4 | 3.7% | 6.35e-4 | 0.999992 | 1.000 |
| `4B-equiv-4L` (h=512,ffn=1536) | `PQ2_0` | 1T | 3.12 | 2.125 | 137.29 +/- 6.42 | 1459.3 +/- 68.0 | 4.7% | 5.44e-4 | 0.999993 | 1.000 |
| `4B-equiv-4L` (h=512,ffn=1536) | `PTQ1_0` | 1T | 2.57 | 1.750 | 159.59 +/- 0.33 | 1253.2 +/- 2.6 | 0.2% | 5.44e-4 | 0.999993 | 1.000 |
| `4B-equiv-4L` (h=512,ffn=1536) | `Bitcos(G128)` | 1T | 2.73 | 1.858 | 264.07 +/- 2.46 | 757.4 +/- 7.1 | 0.9% | 5.44e-4 | 0.999993 | 1.000 |
| `4B-equiv-4L` (h=512,ffn=1536) | `TmacLut` | MT | 3.01 | 2.052 | 689.68 +/- 225.38 | 332.9 +/- 168.6 | 50.7% | 6.35e-4 | 0.999992 | 1.000 |
| `4B-equiv-4L` (h=512,ffn=1536) | `QMatMul(PQ2_0)` | MT | 3.12 | 2.125 | 479.45 +/- 230.60 | 647.0 +/- 644.1 | 99.5% | 5.44e-4 | 0.999993 | 1.000 |
| `4B-equiv-4L` (h=512,ffn=1536) | `QMatMul(PTQ1_0)` | MT | 2.57 | 1.750 | 123.46 +/- 21.18 | 1654.6 +/- 253.8 | 15.3% | 5.44e-4 | 0.999993 | 1.000 |
| `4B-equiv-4L` (h=512,ffn=1536) | `Bitcos(G128)` | MT | 2.73 | 1.858 | 315.03 +/- 75.80 | 661.5 +/- 139.5 | 21.1% | 5.44e-4 | 0.999993 | 1.000 |
| `9B-equiv-2L` (h=1024,ffn=2816) | `FP32-exact` | 1T | 86.00 | 32.000 | 4034.63 +/- 208.02 | 49.7 +/- 2.4 | 4.9% | 0.00e0 | 1.000000 | 1.000 |
| `9B-equiv-2L` (h=1024,ffn=2816) | `TmacLut` | 1T | 5.45 | 2.026 | 110.34 +/- 3.29 | 1813.9 +/- 53.3 | 2.9% | 6.30e-4 | 0.999991 | 1.000 |
| `9B-equiv-2L` (h=1024,ffn=2816) | `PQ2_0` | 1T | 5.71 | 2.125 | 205.88 +/- 2.37 | 971.6 +/- 11.1 | 1.1% | 5.08e-4 | 0.999993 | 0.975 |
| `9B-equiv-2L` (h=1024,ffn=2816) | `PTQ1_0` | 1T | 4.70 | 1.750 | 259.01 +/- 1.80 | 772.2 +/- 5.4 | 0.7% | 5.08e-4 | 0.999993 | 0.975 |
| `9B-equiv-2L` (h=1024,ffn=2816) | `Bitcos(G128)` | 1T | 5.00 | 1.859 | 455.34 +/- 4.57 | 439.3 +/- 4.4 | 1.0% | 5.08e-4 | 0.999993 | 0.975 |
| `9B-equiv-2L` (h=1024,ffn=2816) | `TmacLut` | MT | 5.45 | 2.026 | 199.22 +/- 61.74 | 1102.8 +/- 416.8 | 37.8% | 6.30e-4 | 0.999991 | 1.000 |
| `9B-equiv-2L` (h=1024,ffn=2816) | `QMatMul(PQ2_0)` | MT | 5.71 | 2.125 | 126.43 +/- 21.14 | 1618.3 +/- 276.0 | 17.1% | 5.08e-4 | 0.999993 | 0.975 |
| `9B-equiv-2L` (h=1024,ffn=2816) | `QMatMul(PTQ1_0)` | MT | 4.70 | 1.750 | 94.11 +/- 7.86 | 2137.0 +/- 178.8 | 8.4% | 5.08e-4 | 0.999993 | 0.975 |
| `9B-equiv-2L` (h=1024,ffn=2816) | `Bitcos(G128)` | MT | 5.00 | 1.859 | 142.09 +/- 30.83 | 1454.2 +/- 271.9 | 18.7% | 5.08e-4 | 0.999993 | 0.975 |

### 3.4 Key Comparative Takeaways
- **Single-thread ARM64 NEON (`1T`)**: `TmacLut` sustains `102–112 GOPS` across all full-size 4B/9B/27B projections (`2.2–2.4x` faster than `PQ2_0`, `2.8–3.0x` faster than `PTQ1_0`, `5.2–5.8x` faster than `Bitcos(G128)`), and delivers `1.82–1.87x` higher effective tok/s than `PQ2_0` and `2.12–2.35x` higher than `PTQ1_0` on the 20-turn E2E workload.
- **Multi-thread (`MT`, `barrier_pool`)**: On large full-width projections (`4B-ffn`, `9B-ffn`, `27B-attn`, `27B-ffn` at `z=0.65`), `TmacLut MT` reaches `410–581 GOPS` (`51–73 GB/s`), outperforming `QMatMul(PQ2_0)` (`1.17–1.98x`) and `QMatMul(PTQ1_0)` (`1.60–2.95x`). On smaller projections (`4B-attn` and `h=512`/`h=1024` E2E), canonical `QMatMul(PQ2_0)` / `QMatMul(PTQ1_0)` outperform `TmacLut MT` due to per-call `ActivationLut` construction + barrier overhead.
- **Resident Bytes**: `TmacLut` (`2.006–2.013 bpw`) is `5.5%` smaller than `PQ2_0` (`2.125 bpw`), while `PTQ1_0` (`1.750 bpw`) and sparse `Bitcos(G128)` (`1.537 bpw` at `z=0.65`) retain smaller memory footprints.
- **Accuracy**: `TmacLut` achieves `0.999991–0.999993` cosine similarity vs exact `FP32` and `1.000` token agreement (`200/200`) across both E2E profiles.
- **Dispatch Decision**: Because `PTQ1_0` / `Bitcos` remain smaller in bytes and canonical `QMatMul` wins on smaller MT matrices, `hanzo_nn::ternary_lut` remains strictly **opt-in experimental** with zero changes to default `hanzo-ml` quantized dispatch or model loading.

## 4. End-to-End Acceptance Gate & Documented Workload Boundaries

- **Anti-Microbenchmark Gate (`check_end_to_end_acceptance_gate`)**: Explicitly rejects any kernel variant that regresses 20-turn end-to-end wall time, sustained tokens/s, measured hardware joules/token (when direct energy telemetry is available for both runs), or numerical accuracy (`quality_regression` / `max_logit_abs_error_vs_fp32 >= 1e-4`).
- **Workload Boundaries & Known Limitations**:
  1. *Large-batch prefill crossover ($M \gg N$)*: Because `ActivationLut` construction costs $O(M \cdot K)$ per projection, T-MAC achieves its highest advantage in memory-bandwidth-bound autoregressive decoding ($M = 1$) and short-to-medium interactive agent turns; for massive offline batch prefill ($M \ge 512$), tiled dense GEMM or AMX/SME matrix engines can amortize weight expansion.
  2. *Multi-layer dynamic `i8` rounding sensitivity*: Chaining dynamic `round()` quantizers across many layers requires `sdsdot`-equivalent `f64` post-dot-product row scaling (`(acc_i32 as f64 * alpha as f64 * act_scale as f64) as f32`) so single-ULP `f32` scale products do not flip half-integer activation boundaries downstream.

## 5. Verification Commands

```bash
cargo fmt --all -- --check
cargo clippy -p hanzo-nn --all-targets -- -D warnings
cargo test -p hanzo-nn --lib
cargo test -p hanzo-nn --test ternary_lut --release -- --nocapture
cargo run --release -p hanzo-nn --example tmac_vs_canonical_bench
```
