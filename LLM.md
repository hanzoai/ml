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

### Upstream forks/patches we maintain (validated in our forks, ready to PR upstream, NOT filed)
Complete + stable-release OUR engine/ml first; file these PRs only on explicit go. Both forks are clean + PR-ready.
- **librocdxg** (upstream ROCm/librocdxg). Fork: /home/z/librocdxg branch `fix/aql-pm4-consumer-publication-race` (c301654, +19/-2 in src/wddm/queue.cpp); patch at /home/z/librocdxg_fix.patch; PR write-up at /home/z/librocdxg_pr.md.
  WHY: `AqlToPm4Thread` admits a ring slot before its body is published when a producer bumps `write_dispatch_id` by a whole burst at once (HIP graph replay submits a captured graph as one burst); `IsInvalidPacket` reads the header non-atomically and gates only `INVALID`, so a recycled slot mid-publication reads type `VENDOR_SPECIFIC` (0) with a stale `ven_hdr` and is processed as a malformed PM4-IB -> `assert(ven_hdr==AMD_AQL_FORMAT_PM4_IB)` at queue.cpp:841 (or SILENT GPU corruption on NDEBUG). FIX: acquire-`Load` the header (`wsl::atomic::Load`) + defer-and-retry an unpublished vendor packet (return HSA_STATUS_SUCCESS), mirroring the INVALID guard.
  RESULT: A/B assert builds -> abort before, 0 aborts after; dense + non-graph workloads unaffected; rocminfo still enumerates gfx1151. Benefits every AMD WSL HIP-graph user.
- **rocm-rs** (our fork hanzoai/rocm-rs @ master 09d39b0). rocrand/rocarray bindgen enum FFI types for the Windows ABI (rocblas_status etc. is bindgen-typed -> correct as u32 on Linux AND i32 on MSVC/Windows; upstream hardcodes u32, wrong on Windows) + edition-2024 unsafe blocks in rocblas wrappers + default-on `gpu-sort` feature gate for the amdgcn sort kernel. Engine pins it via `[patch.crates-io]`.

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
- MoE graphs now REPLAY CLEAN (Qwen3-30B-A3B-Q4_K_M, HANZO_ROCM_GRAPHS=1) -- the prior "WDDM PM4-IB
  ring / NOT FIXABLE / not a hanzo refresh bug" conclusion above was DISPROVEN. Two things changed:
  (1) the fixed librocdxg removed the AqlToPm4 ring abort (so the assert no longer masks anything);
  (2) the REAL bug was OURS: the MoE model (engine quantized_qwen3_moe.rs) baked the decode RoPE
  position from the HOST `start_offsets` at capture and never refreshed it -- so every replayed token
  rotated at the FROZEN warmup position -> attention drift -> "Let. Let. 1 1 1" garbage from ~tok 12.
  The dense path (quantized_qwen3.rs) was already fixed: decode reads RoPE off a STABLE device tensor
  `metadata.rope_positions` that the graph runner refreshes in place each token (rocm_graph.rs). The
  MoE model never got that fix. FIX (engine): thread `positions: &Tensor` into `LayerWeights::forward_attn`
  and call `forward_qk_norm_positions(&positions)` for seq_len==1 (host-offset `forward_qk_norm` kept
  for prefill); `ModelWeights::forward` resolves `positions` from `metadata.rope_positions` (graph path)
  else synthesizes -- byte-for-byte the dense pattern. Then re-add `Model::Qwen3MoE(_)` to
  `model_supports_rocm_decode_graph`. HOW IT WAS FOUND: env-gated replay-vs-eager max_abs_diff probe
  in the graph-hit path -- dense Q8_0/Q4_K(0.6B)/Q4_K(14B,40-layer) = 0.0; MoE = 4-16 from tok 0;
  eager-vs-eager = 0.0 (deterministic, no uninit read); bisecting the MoE forward (fix routing, skip
  gate/dp4a/combine, even FusedMoe=identity passthrough) all still diverged ~2-7 -> proved the divergence
  was in the SHARED attention path, not any MoE op; a `diff` of dense vs MoE `ModelWeights::forward`
  showed the dense `positions`-from-`rope_positions` block was simply absent in MoE. VERIFIED: graphs-ON
  "1,2,3,4,5" coherent + haiku+count-to-twenty 340-tok coherent x3 byte-identical (no drift, no assert,
  exit 0), ~33.6 T/s; eager (GRAPHS=0) 38 T/s coherent; dense graphs-ON still coherent. Qwen35
  (Qwen3-VL mRoPE + non-paged Sdpa decode) has the SAME class of bug (fresh per-forward cos/sin) and is
  deliberately LEFT OUT of the gate (decodes eager, correct) until its mRoPE is threaded through
  rope_positions too -- do that before enabling its graph path.
