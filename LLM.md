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
  (2) The DEEPER blocker, RE-ROOT-CAUSED (the prior "#3887 / ~200-node UpdateStreams" label was WRONG):
  the failure is the closed WSL/WDDM thunk's AQL->PM4 queue ring, NOT a hanzo capture bug and NOT a
  kernel-node-count limit. EVIDENCE (rocgdb + AMD_LOG_LEVEL=4): the abort is in
  `wsl::thunk::ComputeQueue::VendorSpecificAqlToPm4` (librocdxg queue.cpp:841) on the thunk's async
  `AqlToPm4Thread` -- it reads a queue slot whose `ven_hdr` is stale/garbage (ring desync), asserting
  `ven_hdr == AMD_AQL_FORMAT_PM4_IB`. `GraphExec::UpdateStreams failed` is BENIGN: it logs `max_streams:
  1`, `hipGraphLaunch` still returns hipSuccess, and DENSE Qwen3-0.6B logs the SAME failure 42x yet
  replays 325 coherent tokens at ~105 T/s. So replay is numerically correct -- the captured MoE forward
  is single-stream (verified: no stream fork, no host-sync, no memset/copyRect graph nodes during
  replay -- 0 fillBuffer/0 copyRect after instantiate), and the per-token input refresh DOES reach the
  buffers (in_tok advances and the first replay tokens are the correct next tokens). The real mechanism:
  a `hipGraphLaunch` injects ONE `PM4_IB` packet whose indirect buffer scales with graph size; the MoE
  graph (router sort/top-k + on-device expert gather x48 layers, ~33ms to instantiate) is large enough
  that repeated PM4-IB submissions interleaved with the eager sampler packets overrun the WDDM ring --
  output drifts into stale repetition ("count to count to... e e e") then aborts after only ~9-132
  launches (NONDETERMINISTIC, timing-dependent -> a race, the corruption signature). Dense's tiny IB
  overruns the same ring only after ~325 launches (so dense ALSO eventually asserts -- this is not
  MoE-specific in mechanism, only in rate). NOT FIXABLE in-repo: `device.synchronize()` after every
  launch (drains the stream) does NOT help (MoE still drifts+aborts at ~85 tok), `AMD_DIRECT_DISPATCH=1`
  does NOT help, and `hipGraphInstantiateWithFlags` ignores all flags on ROCm (documented). RESOLUTION
  (unchanged): `model_supports_rocm_decode_graph` (engine gguf.rs) EXCLUDES the MoE variants so MoE
  decodes eager -- correct AND fast (377 coherent tokens, 31-55 T/s, exit 0) -- while dense keeps the
  working graph path. Verified: MoE eager (377-tok coherent haiku+count, no assert) + dense graphs-ON
  (105 T/s, 325-tok coherent) clean.
