# LLM.md - Hanzo Ml

## Overview
Hanzo ML is a minimalist ML framework for Rust with a focus on performance (including GPU support)

## Tech Stack
- **Language**: Rust

## Build & Run
```bash
cargo build
cargo test
```

## Structure
```
ml/
  CHANGELOG.md
  Cargo.lock
  Cargo.toml
  HANZO_INTEGRATION.md
  LICENSE
  LICENSE-APACHE
  LICENSE-BSD
  LICENSE-MIT
  LLM.md
  Makefile
  README.md
  hanzo-ml-book/
  hanzo-ml/
  hanzo-datasets/
  hanzo-ml-examples/
```

## Key Files
- `README.md` -- Project documentation
- `Cargo.toml` -- Rust crate config
- `Makefile` -- Build automation

## ROCm indexed-MoE decode (gfx1151 / WSL HIP graphs)
- The unified quant decode core is `qmatvec_core<WTYPE,XT>` (quant.hip); its indexed-MoE twin is
  `moe_qmatvec_core<WTYPE,XT>` -> per-type entry points `moe_qmatvecu_<type>_{f16,bf16}` (DEFINE_MOE_QMATVECU),
  selected by `RocmQuantType::moe_decode_kernel`. ONE batched launch over all routed slots: slot s on
  grid.y, expert = ids[s] read ON-DEVICE, bank offset by ids[s] in-kernel. `moe_matvec_quant` takes
  GPU `ids: &RocmStorage` (NOT `&[u32]`) so NO `copy_from_host`/`to_vec1` host round-trip fires.
  Q4_K keeps its dp4a path (`moe_matvec_q4k_dp4a`); every other wired type uses the new core. This
  eliminated the per-expert host launch loop for mixed-precision GGUF (Qwen3-30B-A3B-Q4_K_M has Q6_K
  ffn_down experts). Eager decode 10.22 -> 12.07 T/s. Correctness gate: `moe_matvec_unified_numeric`
  (hanzo-cli) now covers Q4_K + Q8_0 + Q6_K MoE, nbad=0 vs the CPU `to_float` oracle.
- INLINE DIMS/STRIDES (capture-clean strided ops) -- DONE. The per-op `clone_htod` of the tiny
  dims/strides metadata array was the capture-breaker: every strided op (Map1, Map2/broadcast,
  FastReduce/RMS-sum, copy_strided, cast/to_dtype, where_cond, const_set, index_select non-contig)
  uploaded metadata H2D every layer every token -> HIP 906 `hipErrorStreamCaptureImplicit`, killing
  capture on the FIRST op. FIX: the metadata now rides INLINE BY VALUE in the kernel param block as a
  `#[repr(C)] DimsStrides { v: [usize; ROCM_DS_MAX(8) * ROCM_DS_SETS(4)] }` (256 B, layout
  `[dims, strides_0, strides_1, strides_2]`, ROCM_DS_MAX-spaced). No device buffer, no host->device
  copy -> zero `clone_htod` for strided metadata (`DimsStrides::build`, rocm_backend/mod.rs; mirrored
  by `typedef struct {size_t v[32];} DimsStrides` + `num_dims==0` contiguous sentinel in all 8 strided
  .hip files). The kernel's own `is_contiguous` check still picks the contiguous fast loop, so there is
  ONE path (no isc/is split needed beyond index_select's existing one). Rank > 8 bails (never happens;
  decode/prefill are <=4D). EAGER decode also won big: removing the blocking H2D `hipMemcpy` from the
  hot path lifted Qwen3-30B-A3B-Q4_K_M from ~11 T/s to ~16-30 T/s (APU thermal variance). VERIFIED:
  byte-coherent eager ("1, 2, 3, 4, 5"), `moe_matvec_unified_numeric` still nbad=0, and
  AMD_LOG_LEVEL=4 capture shows ZERO `CaptureImplicit`/906.
- HIP-graph capture (`HANZO_ROCM_GRAPHS=1`) NOW CAPTURES + REPLAYS CLEAN: `hipStreamBeginCapture` ->
  `hipStreamEndCapture` -> `hipGraphLaunch` all return hipSuccess (no 906, no 901, no SIGSEGV). Part-2
  recovery hardened too (`end_capture` error path: device-wide `hipDeviceSynchronize` + `hipGetLastError`,
  NEVER the stream-level `device.synchronize()` which returns `hipErrorStreamCaptureUnsupported` on a
  still-capturing stream and leaves the 901->SIGSEGV poison). DENSE Qwen3-0.6B-Q8_0 graphs-ON = exit 0,
  fully COHERENT, 105 T/s -- proves the metadata fix + position-invariant decode replay are correct.
- MoE graphs (Qwen3-30B-A3B) ROOT-CAUSED + RESOLVED via a model-level gate (NOT a buffer fix). Two
  distinct issues were found and the prior "recycled routing/scale buffer" hypothesis was DISPROVEN:
  (1) The MoE F32 ROUTER GATE (`ffn_gate_inp.weight` is F32 dense, [hidden->128 experts]) was the ONE
  dense rocBLAS matmul in the captured forward (experts/attn/lm_head are all quantized -> pooled
  matvec, no rocBLAS). rocBLAS's `gemm_ex` dispatch records a vendor-specific PM4 indirect-buffer
  packet that WSL's HSA thunk rejects on graph replay -> the `VendorSpecificAqlToPm4` assert. FIX:
  `dense_matmul` (quantized/mod.rs) now computes a dense rocm matvec (rows==1) as pooled
  `broadcast_mul + sum` instead of rocBLAS -- capture-clean, numerically identical, GEMV-cheap at M=1.
  This removed the assert source and is a general dense-decode improvement. ALSO pinned a fixed
  process-lifetime rocBLAS workspace (`pin_rocblas_workspace`, device.rs) so rocBLAS never lazily
  `hipMalloc`s GEMM scratch outside our pool (eager prefill win + belt-and-suspenders).
  (2) The DEEPER blocker (the real reason MoE graphs can't replay): even with rocBLAS gone, replay
  still drifts. AMD_LOG_LEVEL=3 proves `GraphExec::UpdateStreams failed` (hip_graph_internal.cpp:1981)
  on EVERY `hipGraphLaunch` (30/30), with ZERO hipMalloc/hipFree during replay (so NOT a recycled
  pointer -- the capture reservation works). This is ROCm/hip#3887: HIP-graph replay reads stale
  device state for graphs with ~200+ kernel nodes on RDNA, no upstream fix, no env/flag workaround
  (GPU_MAX_HW_QUEUES=1 doesn't help). The MoE decode forward (per-layer router sort/top-k + on-device
  expert gather x48 layers) blows past that node count; dense Qwen3-0.6B (no MoE ops, fewer layers)
  stays under it and replays byte-coherent at ~104-106 T/s. RESOLUTION: `model_supports_rocm_decode_graph`
  (engine gguf.rs) now EXCLUDES the MoE variants (Qwen3MoE, Qwen35), so MoE always decodes eager
  (coherent, 30-55 T/s -- faster here than the broken graph path anyway) while dense keeps its working
  graph path. Verified: MoE eager + MoE graphs-ON(=eager fallback, no assert, 340-token coherent
  haiku+count) + dense graphs-ON (105.98 T/s) all clean.
