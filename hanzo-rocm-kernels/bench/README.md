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
Q4_K 129 VGPR / 10 waves / 22528 B LDS; Q6_K (SWAR) 147 VGPR / 9 waves / 21504 B LDS (was 159 VGPR
pre-SWAR); Q8_0 127 VGPR / 10 waves.

Performance: the prefill was **weight-decode-bound, not WMMA-bound** -- at a fixed 512x5120x8192 GEMM
throughput fell monotonically with decode cost. The decode front-end is now **SWAR-vectorized** (one
128-bit load + lane-parallel 32-bit nibble/6-bit extract, replacing per-byte 16-bit bitops):

- **Q6_K SWAR win: 1.70-1.74x** (10.4 -> 18.2 TF/s at 512x5120x8192; 1.71x gate/up 512x4096x12288;
  1.70x down 512x12288x4096). The branchy per-byte 6-bit reassembly is gone -- ISA `v_and_b16`/
  `v_lshrrev_b16`/`v_lshlrev_b16` count **208 -> 0**, kernel body 2492 -> 1278 lines, VGPR 159 -> 147,
  still 0 spills. Q6_K now hits the SAME ~18 TF/s as Q4_K (decode is no longer its bottleneck).
- **Q4_K SWAR: ~1.00x (neutral)**. Q4_K's byte loads were ALREADY vectorized by the compiler (simple
  nibble mask), so SWAR is a wash -- no gain, no regression, nbad=0 preserved.

Both signed-center borrow-free via `((e|0x80808080) - 0x20202020) ^ 0x80808080` (gfx1151 has no
`__vsubss4`); the `iu8` WMMA `is_signed` flags stay `true,true`; the oracle re-runs nbad=0.

**Honest ceiling -- the native path does NOT beat rocBLAS f16 hgemm, and decode is no longer why.**
Even Q8_0 with a trivial memcpy decode tops out at ~18-21 TF/s while hgemm does ~28-41 TF/s. The inner
loop is 8 WMMA + 32 LDS ops (4:1 LDS:MMA) at 10 waves / 0 spills, so the shared MMA machine is **LDS-
bandwidth / scheduling bound at ~40-50% of the iu8 WMMA peak**, not decode-bound. Beating hgemm would
require restructuring that shared tile/MMA machine (wider LDS loads, larger register tiling, async
copies) -- deliberately out of scope (kept byte-identical). Net: the native path's value is (1) keeping
weights quantized (the shipped fallback dequantizes on the **CPU** + H2D every prefill) and (2) Q6_K
prefill is now 1.7x faster. It is fallback-competitive, not a raw-GEMM-rate win over hgemm.

`k%256` guard: the only divisibility constraint is on **K** (`k % block_elems() == 0`). **N is
unconstrained** (the kernel tiles N by 128 with `gn < N` guards). Standard FFN widths are multiples of
256, so the guard never bites in practice (e.g. 17408 = 68*256).
