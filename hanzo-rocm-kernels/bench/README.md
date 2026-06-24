# qmmq prefill GEMM: correctness oracle + microbench (gfx1151)

Standalone HIP harness for the unified native ROCm int8-WMMA **prefill** GEMM `qmmq_core<WTYPE>`
(`../src/kernels/quant.hip`). Self-contained: it builds one code object from the shipped kernels and
loads it via the HIP module API (the same path `hanzo-ml`'s `rocm_backend` uses), so it has zero Rust
workspace coupling (the ml workspace's wasm-only members break a plain `cargo test -p hanzo-ml`).

```
./run.sh            # builds, runs the numeric oracle (nbad==0 gate), then the microbench
```

Env: `ROCM_PATH` (default `/opt/rocm/core-7.13`), `ROCM_GFX_ARCH` (default `gfx1151`),
`HSA_OVERRIDE_GFX_VERSION` (default `11.5.1`).

## Files
- `qmmq_bench_kernels.hip` -- `#include`s the production `quant.hip` + adds `deq_q4k_f16`/`deq_q6k_f16`
  (canonical GGML `to_float`, one thread/super-block) for the dequant-f16 fallback A/B.
- `qmmq_oracle.hip` -- numeric gate. Builds Q4_K/Q6_K/Q8_0 weights in real GGML layout, runs
  `quantize_q8`/`quantize_q8_1` + `qmmq_<t>_f16` on the GPU, checks against a CPU dequant->f32 GEMM on
  the SAME per-32-block int8-requantized activation; asserts `nbad==0` at 1% tolerance.
- `qmmq_bench.hip` -- times native (`quantize + qmmq`) vs dequant-f16 + rocBLAS `hgemm` on FFN shapes,
  plus a GEMM-only weight-decode-cost probe (Q8_0 vs Q4_K vs Q6_K at one shape).

## Results (Radeon 8060S / gfx1151 / ROCm 7.13)

Correctness: **Q4_K and Q6_K both nbad=0** across all shapes incl. the production 512x4096x4096 prefill
(max_abs_diff <0.05% of output magnitude = f16-rounding noise). The `iu8` WMMA `is_signed` flags are
correct (`true,true`).

Occupancy (`-Rpass-analysis=kernel-resource-usage`): zero VGPR/SGPR spills on all four prefill kernels.
Q4_K 129 VGPR / 10 waves / 22528 B LDS; Q6_K 159 VGPR / 9 waves / 21504 B LDS; Q8_0 127 VGPR / 10 waves.

Performance (the honest result): the prefill is **weight-decode-bound, not WMMA-bound**. At a fixed
512x5120x8192 GEMM the throughput falls monotonically with decode cost --
Q8_0 (memcpy) ~23 TF/s, Q4_K (nibble) ~19.6 TF/s, Q6_K (6-bit reassemble) ~11 TF/s -- while rocBLAS
f16 `hgemm` does 28-41 TF/s. So `qmmq` is **0.5-0.7x** the hgemm GEMM rate (Q4_K) / **0.3-0.4x** (Q6_K).
The end-to-end FFN win (Q4_K ~1.0-1.16x vs the *shipped* dequant-f16 fallback) comes from avoiding the
dequant cost, not from a faster GEMM -- and the shipped ROCm fallback dequantizes on the **CPU** + H2D
every call, which is what makes keeping weights quantized worthwhile. The lever to actually beat hgemm
is to hoist/vectorize the per-block weight decode (esp. Q6_K's branchy quad loop) so the matrix cores
stop starving; that work is NOT done here.

`k%256` guard: the only divisibility constraint is on **K** (`k % block_elems() == 0`). **N is
unconstrained** (the kernel tiles N by 128 with `gn < N` guards). Standard FFN widths are multiples of
256, so the guard never bites in practice (e.g. 17408 = 68*256).
