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
  LICENSE-MIT (or LICENSE-APACHE)
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

## Current state (portable kernel DSL, auto-fusion, GDN, qwen3.5)
- **hanzo-kernel** (the portable kernel DSL, homed in this repo at `hanzo-kernel/`, published 0.2.15)
  is a CODEGEN not a runtime: one `#[kernel(targets(...))]` Rust source lowers to CUDA/ROCm/Vulkan/
  Metal/WebGPU/CPU via CubeCL. The built-in op library (`hanzo-kernel/src/`) is bit-exact on the CPU
  runtime: `norm` (rms/layer), `rope`, `attn` (`sdpa` + `sdpa_runtime`, online-softmax = the 8B
  flash-collapse structural cure), `gdn` (gating/conv1d/scan), `quant` (matvec_q8/q4k), and `fuse`.
- **`fuse` (auto-fusion): fusion is composition.** `hanzo-kernel/src/fuse.rs` implements the op algebra
  where legal fusion is a total function of op CLASS (Map = index-local, freely fusible by the functor
  law `map g . map f == map (g . f)`; Reduce = a fence). `Fuse::new(a).mul(w).add(b).silu().run()`
  compiles a chain to a comptime op program and dispatches ONE `fused_interp` launch (zero
  intermediates). No pattern-matcher, no separate fusion pass -- only composition of morphisms.
  Bit-exact-gated three ways (fused kernel == naive N-kernel == plain-Rust ref).
- **Fused GDN kernel** (Gated-DeltaNet linear attention) drives Qwen3.5/3.6 hybrid archs to near-parity;
  the portable `gdn_conv1d`/`gdn_gating`/`gdn_scan` DSL source replaces the ops-composed CUDA/ROCm path,
  with the engine model (`quantized_qwen3_5_moe`) fully wired (GGUF arch + loaders).
- **CPU test gates** (no GPU, safe): `cargo test -p hanzo-kernel --no-default-features --features cpu`
  = 15/15 (the norm/rope/attn/gdn/fuse bit-exact oracles); `cargo test -p hanzo-ml` = green except the
  pre-existing `serialization_tests::npz` (a missing git-tracked fixture `tests/test.npz`, NOT a code
  regression). The on-device numeric oracles (`qmatvec_unified_numeric`, `moe_route_numeric`,
  `qmmq_unified_numeric`, `rmsnorm_numeric`) live in `engine/hanzo-cli/tests/` and gate GPU builds.

### hanzo-kernel ROCm/HIP seam UNBLOCKED on ROCm 7.13 (gfx1151) -- the DSL now dispatches on AMD (0.2.18)
The `rocm` feature (`cubecl/hip`) was un-buildable against ROCm 7.13. TWO independent upstream cubecl 0.10
defects, both fixed WITHOUT bumping cubecl (0.10 is the newest published; no version escapes either bug):
1. **cubecl-hip-sys binding selection.** Its `build.rs` reads `hipconfig --version` and emits
   `feature="hip_<patch>"`; ROCm 7.13 reports patch **99004**, past the newest shipped binding
   (`hip_53211` = ROCm 7.2), so it emitted a `hip_99004` feature with NO bindings module -> every
   `cubecl_hip_sys` symbol undefined (the "67 R0600 errors"). FIX = a vendored `[patch.crates-io]` fork
   at `hanzo-kernel/vendor/cubecl-hip-sys` (upstream verbatim + a build.rs clamp: an unknown-newer patch
   falls back to the latest shipped binding, mirroring the existing no-hipconfig branch). ABI-SAFE and
   proven: ROCm 7.13's `hip_runtime_api.h` still `#define hipGetDeviceProperties hipGetDevicePropertiesR0600`
   (no R0700), so the 7.2 (R0600) bindings link against the 7.13 runtime unchanged.
2. **`multi_threading` cfg / std.** cubecl-hip's server uses `cubecl_runtime::stream::MultiStream`, gated
   behind cfg `multi_threading = all(feature="std", not(wasm))`. cubecl's `hip`/`rocm` features pull NO
   std, and hanzo-kernel builds `default-features=false`, so `MultiStream` was `#[cfg]`'d out. FIX =
   `rocm = ["cubecl/hip", "cubecl/rocm", "cubecl/std"]` (`cubecl/std` -> `cubecl-core/std` ->
   `cubecl-runtime/std`). NOTE: `cuda` almost certainly needs the same `cubecl/std` (untested, no NVIDIA
   on evo) -- add it when the CUDA DSL path is first built.
- PROOF (the falsifiable gate): `norm::tests::rms_norm_rocm_bit_exact` (feature `rocm`) lowers the `rms_norm`
  DSL source through cubecl-hip and dispatches on evo's gfx1151, bit-compared to the CPU oracle
  `rms_norm_ref`: **max_rel = 2.43e-7** (f32-reorder noise, gate 2e-3). End-to-end: R0600 device query +
  HIP-RTC module compile + launch + readback all correct on real hardware. `rms_norm` targets now list
  `rocm`. Zero regression: CPU gate 30/30. Build/run: `source ~/work/hanzo/rocmenv.sh` then
  `cargo test -p hanzo-kernel --features rocm` (ONE ROCm process at a time on evo -- concurrent kfd init
  wedges the driver).

### hanzo-kernel CUDA tensor-core path PROVEN on GB10 (sm_121 Blackwell) -- the DSL MMQ + cmma gate
Answers the ROCm-seam note above ("cuda ... untested, no NVIDIA on evo -- add it when the CUDA DSL path is
first built"): the CUDA DSL path now BUILDS + gates on GB10 (spark, CUDA 13, aarch64, sm_121). `cuda =
["cubecl/cuda"]` alone is sufficient -- the ROCm `cubecl/std` was a HIP-server (`MultiStream`) requirement,
not needed for the CUDA runtime.
- **int8 tensor-core MMQ GEMM in the DSL (`hanzo-kernel/src/mmq.rs`, MERGED).** The prefill lever written
  ONCE: `wmma_hello_i8` (cmma::Matrix i8 16x16x16 -> i32) + `mma_hello_i8` (manual cmma::MmaDefinition
  `mma.sync` m16n8k32, register element<->lane map via `position_of_nth`) + `mmq_q8_wmma` / `mmq_q8_wmma_blk`
  (Q8_0 x q8_1 block-scaled GEMM: int8 tensor-core dot per 32-block, f32 per-block scale in the epilogue).
  Probe bin `mmq-check` (required-features=cuda), staged cheapest-kill-first vs a CPU quantized oracle.
  GATE (GB10 sm_121, all green): WMMA hello EXACT (0/256), manual MMA EXACT (0/128), MMQ tiles BIT-CLOSE
  (rel_to_max 5.9e-8 / 1.25e-7 / 1.62e-7), full 512x4096x4096 GEMM verify rel_to_max 2.32e-7. So the DSL
  emits working int8 matrix-core ops on Blackwell, numerically faithful to the block-quantized reference.
- **PERF CAVEAT (foundation, not yet the optimized kernel).** The tiled 8-warp/32x64 kernel is 0.58x the
  naive 1-warp (4183 vs 7263 GFLOP/s) -- the high-level WMMA `cmma::store -> shared -> scale` round-trip
  (the per-block f32 scale applied through the OPAQUE i32 fragment) is the API tax the tiling can't hide.
  mmq.rs is the correct, bit-exact FOUNDATION of the MoE-prefill fusion lever; the next step is applying
  the scale IN-REGISTER on the manual-MMA path (`position_of_nth` exposes the element<->lane map), not the
  smem round-trip. Merged for the primitive + gate; the tuned prefill GEMM is the follow-up.
- **f16 cmma verified, REPORT-ONLY (branch `feat/dsl-flash-attn` NOT merged).** A go/no-go probe proved the
  SAME DSL cmma path lowers f16 16x16x16 -> f32 on GB10: max_rel 1.18e-7 (MATCH). This is the primitive a
  hand-rolled DSL flash-attention-2 needs (Q@K^T + P@V on tensor cores). The probe bin itself is NOT merged:
  it is a bin with no library surface, and mmq.rs's `wmma_hello_i8` already proves DSL->tensor-core lowering
  IN THE LIBRARY (i8); the f16 result is recorded here as the positive that makes a hand-rolled DSL
  flash-attn viable. Build/run: `cd hanzo-kernel && cargo run --release --features cuda --bin mmq-check`.

## Env vars (bare names, one-way -- de-brand DONE)
The env-flag convention is BARE names, no `HANZO_` brand prefix. The de-brand is COMPLETE: every runtime
knob is a bare name, and the one-off dev A/B "fallback" toggles are GONE (production always runs the fast
path; runtime HW auto-select is kept). One bare name per real knob, one way to do everything.
- **Bare runtime knobs**: `CUDA_GRAPHS`, `ROCM_GRAPHS`, `METAL_GRAPHS`, `FLASHINFER_DECODE` (default ON,
  `=0` forces eager); `VK_COOPMAT`, `VK_INT_DOT`, `VK_PROFILE`, `VK_PUSH_DESC`, `VK_DP4A_DECODE_OFF`,
  `VK_SUBGROUP_MATVEC`, `VK_Q4K_{LEGACY,DP4A,TILED2D,COOPMAT}` (Vulkan Q4_K path A/B gates, test-driven);
  `MOE_BACKEND`, `MTP_CONF_THRESHOLD`, `GEMMA4_DISABLE_FAST_PREFILL`, `V4_*`, `ZEN3_*`, `ZEN_OMNI_*`;
  `MOE_TILE_M`, `MOE_STATS`, `CUDA_FAST_MMQ`, `MIN_LEN`, `ML_CUDA_UMA_THRESHOLD` (ml); `ROCM_GFX_ARCH`,
  `ROCM_VERSION`, `GGUF_MMAP`, `CUDA_FLASH_BF16`, `FLASH_ATTN_BUILD_DIR`, `METAL_PRECOMPILE`
  (build/backend config).
- **DELETED (were one-off dev A/B toggles that forced the OLD/slow path; production always wants fast)**:
  `HANZO_Q{2,3,4,5,6}K_FALLBACK`, `HANZO_IQ*_FALLBACK`, `HANZO_IQ_DEQUANT_FALLBACK`,
  `HANZO_MOE_{QMMQ,ROUTE,GATEUP,COMBINE}_FALLBACK`, `HANZO_DBG_PATH`, `HANZO_ROCM_FLASH_DEBUG`,
  `SAMPLER_TRACE`, `DEQUANTIZE_ALL(_F16)`. The dp4a-vs-scalar decision is now purely runtime
  (`RocmQuantType::dp4a_active` == `dp4a_capable`, HW/type auto-select); the scalar core is still the
  path for non-dp4a types. Bit-exact oracle tests force the scalar core via the test-only
  `hanzo_ml::set_force_scalar_matvec(bool)` (a programmatic flag, never env, never set in production).
- **`VK_DP4A_DECODE` two-ways inconsistency RESOLVED**: ONE bare name `VK_DP4A_DECODE_OFF` (the doc that
  named a phantom `HANZO_VK_DP4A_DECODE` was corrected; only the bare `_OFF` is read).
- **Kept `HANZO_`-branded (product identity, NOT a perf knob)**: `HANZO_ENGINE_LICENSE{,_FILE,_TOKEN}`,
  `HANZO_LICENSE_SIGNING_KEY` -- the license identity vars are the ONLY remaining `HANZO_` env names.

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
  dp4a-capable types take the dp4a core (below); every other wired type uses this scalar core. This
  eliminated the per-expert host launch loop for mixed-precision GGUF (Qwen3-30B-A3B-Q4_K_M has Q6_K
  ffn_down experts). Eager decode 10.22 -> 12.07 T/s. Correctness gate: `moe_matvec_unified_numeric`
  (hanzo-cli) now covers Q4_K + Q8_0 + Q6_K MoE, nbad=0 vs the CPU `to_float` oracle.
- UNIFIED int8-dp4a DECODE core `qmatvec_dp4a_core<WTYPE,XT>` + MoE twin `moe_qmatvec_dp4a_core<WTYPE,XT>`
  (quant.hip) -> per-type entry points `qmatvec_dp4a_<t>_{f16,bf16}` / `moe_qmatvec_dp4a_<t>_{f16,bf16}`
  (DEFINE_QMATVEC_DP4A), one per-type packed-int decode `qdp4a<WTYPE>::block` (mirrors the scalar
  `qdec<WTYPE>::partial` bit layout). This REPLACED the hand-written Q4_K-only dp4a kernels: there is
  now ONE dp4a core + ONE scalar fallback, trait-selected by `RocmQuantType::dp4a_active()` (dp4a_capable
  AND `HANZO_<T>_FALLBACK` unset). Wired: Q4_K (ASYM) + Q6_K (SYM 6-bit) -- Q5_K is one `qdp4a<DW_Q5_K>`
  + one row away. The `if Q4K {dp4a} else {scalar}` braid is GONE from `matvec_quant` / `moe_matvec_quant`
  / the two `indexed_moe_forward` sites; the dp4a-vs-scalar A/B lives ONLY in `matvec_quant` (the
  forward-level `unified_qt` filter no longer forces a Q4_K dequantize -- the fallback now switches the
  decode CORE, the type stays native either way). Q6_K dp4a = the #1 decode lever (Q6_K = output.weight
  lm_head + attn_v + ffn_down experts = 37.7% of GPU work, was SCALAR ~80 GB/s vs Q4_K dp4a 460 GB/s).
  gfx1151 has NO `__vsubss4` (ROCm 7.13) so Q6_K reconstructs the centered (q-32) weight directly into
  signed int8 [-32,31] and `sudot4`s it -- no branchless subtract needed. Correctness: `qmatvec_dp4a_vs
  _scalar` (hanzo-cli) asserts dp4a == scalar core bit-faithfully (nbad=0, Q4_K+Q6_K, matvec+MoE) AND
  the existing oracle gates stay nbad=0. PERF: eager decode Qwen3-30B-A3B-Q4_K_M (GRAPHS=0, -n 0:48,
  3 sustained 560-tok runs) = 33.71 T/s mean (33.74/33.90/33.50) vs 33.8 baseline = FLAT; A/B
  HANZO_Q6K_FALLBACK=1 (scalar Q6_K) = 32.1 T/s -> dp4a is marginally faster but within thermal/run-
  length noise. The Q6_K kernel-level 4x (80->~460 GB/s) does NOT surface at the model level because
  EAGER MoE decode is LAUNCH-bound, not Q6_K-bandwidth-bound: ~1500-1900 tiny hipLaunchKernel/token
  over 48 layers (each matvec is its own launch; dp4a even ADDS the quantize_q8_1 launch), so the wall
  clock is dispatch-latency-dominated (~30 ms/tok / ~16-20 us per op) and shrinking Q6_K GPU-time
  barely moves it. The lever is real but is gated behind HIP-graph capture (where per-launch overhead
  vanishes) -- MoE graphs now replay clean, so the Q6_K dp4a win should surface with GRAPHS=1; revisit
  there. Correct + bit-faithful + zero-regression either way, so it ships as the default decode core.
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
- Q6_K dp4a decode kernel ~5x SPEEDUP (quant.hip `qdp4a<DW_Q6_K>::partial`, branch perf/moe-fused-decode).
  rocprofv3 on Qwen3-30B-A3B-Q4_K_M decode proved the EAGER MoE wall is COMPUTE (not pure dispatch):
  Q6_K kernels (output.weight lm_head + attn_v + ffn_down experts) were ~40% of GPU time at 260/249 us
  per call -- 5-12x slower than the Q4_K kernels (45/21 us) -- because the OLD Q6_K decode did, per
  16-elem scale-group, 16 byte-granular ql/qh gathers + a branchy per-element (if quad==0..3) 6-bit
  reconstruct. FIX: a 16-elem scale-group lies wholly inside ONE 32-quad so half/quad/l0 are CONSTANT
  across its 16 positions -> load the 16 ql + 16 qh bytes as four packed `int` reads each, reconstruct
  4 weights/int with packed nibble (`>>0|>>4 & 0x0F0F0F0F`) + 2-bit-high (`(hv>>(quad*2))&0x03030303<<4`)
  ops, ONE branch per group not per elem. The q6-32 centering is a BIAS TERM (sum_p(q-32)u = dp4a(q,u)
  - 32*dp4a(0x01010101,u)) -- raw q in [0,63] is a valid signed int8, and the bias-sum avoids the
  cross-byte-borrow a packed `int - 0x20202020` would suffer (gfx1151 has no per-byte `__vsubss4`).
  RESULT (rocprofv3, same workload): qmatvec_dp4a_q6k 260->53 us (4.9x), moe_qmatvec_dp4a_q6k 249->51 us
  (4.9x); total GPU kernel time 4622->2410 ms (-48%). Model-level eager decode 28.2 -> 33.0 T/s (+17%,
  3 runs 33.74/32.96/32.28) + prefill ~108->157 T/s; the model gain < the -48% GPU drop because eager
  carries fixed per-launch dispatch overhead that doesn't shrink with kernel time (the GRAPHS lever).
  Correctness: `qmatvec_unified_numeric` Q6_K dp4a-vs-scalar + MoE dp4a-vs-scalar all nbad=0 (bit-
  faithful to the CPU to_float oracle, f32 reorder only). ALSO in the same branch: the MoE dp4a core
  (`moe_qmatvec_dp4a_core`) now strides lanes over (super-block x sub-unit) work items, not whole
  super-blocks -- the expert shapes are narrow-k (gate/up k=2048->8 super-blocks, down k=768->3) so
  whole-block striding idled 24-29/32 lanes; NSUB sub-units/super-block (Q4_K=4 chunks, Q6_K=16 groups)
  via the new `qdp4a<WTYPE>::partial(blk,xq8,xd8,sub)` (block() = sum over sub, so non-MoE core is
  byte-unchanged) fills the warp. Occupancy fix alone was model-flat (Q4_K MoE was already 45 us);
  the Q6_K decode rewrite was the real lever.
- FUSED indexed-MoE PREFILL GEMM = the 2.58x prefill lever (branch perf/moe-fused-gemm). The MoE prefill
  was the ENTIRE deficit vs llama.cpp-HIP on gfx1151 (Qwen3-30B-A3B-Q4_K_M, pp1024/tg128): hanzo
  prefill 366 vs llama 1056 (0.34x) while dense 0.6B was already 1.12x. ROOT CAUSE (rocprofv3, matvec
  baseline): `indexed_moe_forward` routed prefill (rows>1) through the per-SLOT `moe_matvec_quant`
  (grid.y=nslots matvec) EXACTLY like decode, so every routed token re-read its expert's whole Q4_K/Q6_K
  weight and ran NO matrix cores -- moe_qmatvec_dp4a_q4k 4857 ms (53.1%) + _q6k 1607 ms (17.6%) = 70.7%
  of GPU time on the MoE matvecs. FIX: a fused expert-grouped WMMA GEMM `qmmq_core<WTYPE,true>` (the
  SAME proven 16-warp/128x128 iu8 WMMA prefill machine, +`if constexpr (MOE)` gather/scatter -- the
  dense path codegens byte-identical). Tokens are grouped BY EXPERT host-side (counting sort over the
  GPU ids; prefill is never graph-captured so the ids DtoH is free) into per-expert <=128-row tiles
  (slot_map/tile_expert/tile_pos0/tile_nrows work-items); each block stages ONE expert's weight and
  amortizes it over all its tokens via MROW gather (xq read) + scatter (y write) -- llama's mul_mat_id.
  `moe_qmmq_quant` (rocm_backend) + `moe_prefill_kernel` table + the two `indexed_moe_forward` sites
  (`use_qmmq = t>1`, HANZO_MOE_QMMQ_FALLBACK forces matvec for the A/B). DECODE (t==1) keeps the
  capture-clean dp4a matvec untouched. RESULT (model-level, same bench): prefill 366 -> 944.7 T/s
  (2.58x, now 0.89x of llama's 1056; A/B HANZO_MOE_QMMQ_FALLBACK=1 = 365.8 confirms the kernel is the
  sole lever), decode 41 -> ~51 (thermal, untargeted), coherent 30B output. rocprofv3: MoE matmul GPU
  time 6464 ms (matvec) -> 1452 ms (moe_qmmq_q4k 1242 + _q6k 210), -78%; total GPU 9147 -> 4357 ms (-52%).
  Bit-exact: new `moe_qmmq_numeric` gate (hanzo-cli, E=8/300-slot Q4_K+Q6_K+Q8_0, multi-tile + empty
  experts) nbad=0 vs the CPU to_float oracle; dense qmmq_unified_numeric still nbad=0 (MOE=false elides).
  REMAINING gap to llama (0.89x): avg ~64 tokens/expert vs TILE_M=128 = ~50% M-tile fill (the next
  lever = smaller M-tile / stream-k, needs the 4x4 warp layout reworked), plus the now-relatively-larger
  non-MoE overhead (fast_sum 11%, casts 10%, quantize_q8_1 4%) llama fuses.
- DYNAMIC M-TILE (LEVER 1, branch perf/moe-fused-gemm on 0.11.8). The fused MoE GEMM staged a 128-row
  WMMA M-tile regardless of how many tokens an expert routed; at ~64 tok/expert that left HALF the WMMA
  M-segments empty (measured fill 49.9% over the 432 full-prefill dispatches), and the few-token decode
  micro-batches (t=4 -> nslots=32, ~1 tok/expert) were ~1-6% full. FIX: `qmmq_core<WTYPE,MOE,NWAVE_M>`
  is now parametrized on the M-wave count -> TILE_M 128/64/32 (NWAVE_M 4/2/1) with THREADS=NWAVE_M*128.
  The hardcoded 512-thread STGI staging split was rewritten as THREAD-COUNT-AGNOSTIC strided loops
  (`for i=t; i<256+TM*2; i+=THREADS` for the int8 tiles, `128+TM` for scales) so ONE core covers every
  block size; the NWAVE_M=4 case codegens byte-identical (dense path untouched). The launcher
  (`moe_qmmq_quant`) picks TILE_M by routed fill (avg tok/active-expert: >=160->128, >=24->64, else 32;
  HANZO_MOE_TILE_M overrides) and sets block=TILE_M*4 + the matching `moe_qmmq_<t>[_tm64|_tm32]_f16`
  entry. RESULT (gfx1151, on the 0.11.8 combine+f32-quantize baseline): full-prefill M-fill 49.9% ->
  68.2% (all dispatches now TILE_M=64), prefill 1172.8 -> 1200.5 T/s (+2.4%, tight +-1 over 3 thermally-
  interleaved passes), decode ~51 flat. The +36% relative fill yields only +2.4% throughput because the
  full-prefill MoE GEMM is largely WEIGHT-BANDWIDTH-bound, not WMMA-compute-bound -- PROVEN by the A/B:
  TILE_M=32 (even less empty WMMA work) is SLOWER (~885 vs 952 on 0.11.7) because it re-reads each
  expert weight in 2 tiles; TILE_M=64 (one tile per ~64-tok expert, minimal weight re-reads + good fill)
  is the bandwidth/compute balance point. So M-fill is largely exhausted at TILE_M=64; the residual gap
  is irreducible expert-weight fetch. Bit-exact: `moe_qmmq_numeric` extended to gate all 3 TILE_M
  (128/64/32) x Q4_K/Q6_K/Q8_0, nbad=0; moe_combine_numeric + moe_matvec_unified_numeric + dense
  qmmq_unified_numeric all nbad=0; coherent 30B output ("The capital of France is Paris.").
- 0.11.9 SHIPPED -- LEVER-1 prefill M-tile (real) + f32-native decode core (bit-exact, NO decode
  speedup, honest). MoE prefill 1177 T/s = 1.09x llama-HIP (1062) / 1.23x Vulkan (957), bit-exact;
  M-fill exhausted at TILE_M=64 (weight-bandwidth-bound). Decode core (f32-native dp4a matvec via
  Act::F32 / *_f32 instantiations keeping the residual stream F32 = casts -63%, a dense_gemv block-
  per-row router gate replacing a materialized broadcast_mul+sum, dense-matvec NSUB lane-coalescing):
  ALL oracles nbad=0 (qmatvec_unified/bf16/q4k + moe_combine + qmmq_unified) but device-map-controlled
  A/B (-n 0:48 BOTH binaries, graphs ON) = real run-loop decode 42.06 -> 42.61 T/s = +1.3% FLAT. The
  agent's headline "+28% (46.8->59.8)" was a MEASUREMENT ARTIFACT: 46.8 = auto-mapper offloading ~1/3
  layers to CPU, 59.8 = forced-all-GPU; +28% was the -n device fix, not the kernel. The matvec is at
  88-99% of the 212 GB/s memory roofline (rocprofv3 FETCH_SIZE; Q6_K does more compute yet higher BW
  than Q4_K = bandwidth-bound proof), so cast/ALU changes cannot move it. The real remaining decode
  lever is OP-FUSION (fused softmax+topk routing = 6 kernels/layer -> 1; fused gate+up+silu; fused
  rope+cache) to cut the ~2725 kernels/token, NOT a matvec change. Decode merge kept for the kernel-
  count reduction + decomplect, not a perf claim. ALWAYS pass -n "0:48" for 30B-A3B decode A/B.

## 0.11.10 -- fused MoE routing kernel (the REAL decode lever: +13-15%, bit-exact)
- **`moe_route` (REAL decode win):** ONE ROCm kernel (one block/token, shared-mem softmax + top-k +
  normalize) replaces the softmax->sort->narrow->sum->div chain (~6 launches/layer x48 = ~240 fewer
  kernels/token). `hanzo_ml::quantized::moe_route(logits, topk, norm)` -> (indices, weights); rocm
  fast path + ml-op fallback elsewhere. THIS is the kernel-count lever the bandwidth-wall analysis
  pointed to (NOT the matvec). Device-map-controlled A/B (-n 0:48, graphs ON): 30B-A3B decode @d4
  52.9 -> **59.9 T/s (+13%)** = 0.90x llama-HIP (was 0.80x); run-loop long-context 40.4 -> 46.7
  (+15%); prefill 1177 -> 1192 = 1.12x llama-HIP. Bit-exact: new `moe_route_numeric` oracle nbad=0
  (max_w_err ~1e-8 vs the ml-op reference) across ntok 1..1024, experts 60..256, topk 4..8, norm
  on/off; all prior oracles still nbad=0. Next kernel-count levers: fused gate+up+silu, fused
  rope+cache.

## ROCm quant-zoo -- 16 new GGUF decode families merged to ONE wired set (22 types, all bit-exact)
- Four sibling branches each added native ROCm decode for a different quant family, all editing the
  SAME three files (`quant.hip`, `rocm_backend/mod.rs`, `quantized/mod.rs`); hand-merged into ONE
  wired set. Now `RocmQuantType` has 22 variants: the 6 base (Q8_0/Q4_0/Q4_K/Q6_K/IQ4_XS/TQ2_0) +
  Q2_K/Q3_K + Q5_K/Q4_1/Q5_0/Q5_1/Q8_1 + IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S + IQ4_NL/TQ1_0/IQ1_S/IQ1_M.
  Every type decodes through the SAME `qmatvec_core<WTYPE>` (+ `moe_qmatvec_core` MoE twin); the
  five K-quants Q4_K/Q6_K/Q2_K/Q3_K/Q5_K also ride the int8-dp4a `qdp4a<WTYPE>` core. IQ2/IQ3 carry
  embedded `__constant__` codebook grids (IQ2XXS/IQ2XS/IQ2S/IQ3XXS/IQ3S + KSIGNS); IQ1_S/IQ1_M carry
  IQ1S_GRID; IQ4_NL reuses the existing KVALUES_IQ4NL codebook.
- **Decomplected the prefill predicate to ONE `RocmQuantType::qmmq_capable(&self)`** (the branches had
  invented three names: `prefill_capable`, two `qmmq_capable`s). It is the int8-WMMA-prefill ALLOWLIST
  == exactly the types with a `DEFINE_QMMQ`/`wt_traits<WTYPE>` kernel: {Q8_0, Q4_0, Q4_K, Q6_K, IQ4_XS,
  TQ2_0, Q5_K, Q4_1, Q5_0, Q5_1, Q8_1} (11 true). The 11 decode-only types (Q2_K/Q3_K + all IQ*/TQ*
  codebook/fractional) are false -> dense prefill dequantizes-to-f16, MoE prefill rides the per-slot
  matvec core (correct at any token count). ONE gate, read at the dense `forward` filter + both MoE
  `use_qmmq` sites; an allowlist so a future decode-only type defaults to the safe dequant path.
- `DW_*` ids renumbered contiguous 0..21 (base 0-5 unchanged; the merge appends b1->b2->b3->b4). Ids
  are arbitrary tags -- every trait/macro selects by symbolic name and dispatch is by kernel-name
  string, so renumbering is free. `WT_*` stays 0..10 (only the qmmq-capable types). Verified block
  bytes against the structs: Q2_K=84, Q3_K=110, Q5_K=176, Q4_1=20, Q5_0=22, Q5_1=24, Q8_1=36.
- Oracle gate (engine hanzo-cli, gfx1151): all 22 types decode `nbad=0` (qmatvec_unified + iq2/iq3/
  iq1/iq4nl), all 11 prefill types `nbad=0` (qmmq_unified), MoE decode + moe_combine + moe_route
  `nbad=0`, and the 6 pre-existing types unchanged (zero regression). bit-exact vs the CPU `to_float`
  oracle. `BlockQ8_1::to_float` (k_quants.rs) implemented so the Q8_1 weight type has a CPU reference.

## IQ codebook dp4a decode -- the 2.27x i-quant decode lever (compute-bound, NOT launch-bound)
- The IQ codebook quants (IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M) decoded ONLY through the scalar
  `qdec<DW_IQ*>` core (a grid-table lookup + float MACs per element) = PATHOLOGICALLY slow because the
  decode is COMPUTE-bound (the per-element f32 dequant + MAC), not bandwidth-bound. Qwen3-30B-A3B
  IQ3_XXS decoded 30.4 T/s vs Q4_K's 63 (Q4_K rides dp4a) and llama-HIP 67.6 / Vulkan 91.8.
- FIX: `qdp4a<DW_IQ*>` specializations of the SAME unified `qmatvec_dp4a_core` / `moe_qmatvec_dp4a_core`
  the K-quants use (quant.hip). The grid coords land in int8 and `hip_dp4a` (sudot4) dots them against
  the once-q8_1-quantized activation -- llama.cpp `vec_dot_iq*_q8_1`. Purely ADDITIVE: 7 `qdp4a<>` +
  7 `qdp4a_traits<>` + 7 `DEFINE_QMATVEC_DP4A` rows + 4 `RocmQuantType` selector rows (dp4a_capable /
  fallback_env / dp4a_decode_kernel / dp4a_moe_kernel). NO new core, NO dispatch change -- `matvec_quant`
  / `moe_matvec_quant` / `indexed_moe_forward` already route by `dp4a_active()`. NSUB=8 (the 32-elem
  sub-block = the q8_1 32-scale unit). They live next to the scalar `qdec<DW_IQ*>` (after the grids)
  because they index the `__constant__` grid tables, so DEFINE_QMATVEC_DP4A is #undef'd there not at 866.
- SIGNS without CUDA `__vsub4`/`__vcmpne4` (ABSENT on ROCm 7.13 -- only `__byte_perm` exists): the IQ2/3
  grids store POSITIVE magnitudes (verified <=62, valid int8) + per-coord sign bits. By linearity
  `sum (-1)^s g u = dp4a(g,u) - 2*dp4a(g & bytemask(s), u)` -- `dp4a_signed` helper, `iq_bytemask4`
  spreads a 4-bit sign nibble to 0xFF-per-byte. NO per-byte subtract. Cheaper than the SWAR-vsub4
  alternative (2 dp4a + 6 ALU vs 1 dp4a + 10 ALU). IQ1 grids are PRE-SIGNED int8 -> direct dp4a + a
  `sum(u)` term (dp4a vs 0x01010101) for the fractional `+delta` bias.
- BIT-EXACT (gate leg 1): the f32 IQ scale (`db`) is applied in FLOAT exactly like the scalar `qdec`
  (NOT llama's integer `sumi*ls/8` folding, which would diverge from the f32 to_float oracle); only the
  integer grid*activation dot is dp4a. All 7 types nbad=0 vs the scalar core (`qmatvec_iq{2,3,1}_dp4a_
  vs_scalar`) AND vs to_float-on-q8_1_recon (`check`/`moe_check`); scalar weight decode still bit-exact
  (`decode_exact` forced-scalar via HANZO_IQ*_FALLBACK); all prior oracles + iq4nl unchanged.
- FASTER (gate leg 2): Qwen3-30B-A3B-IQ3_XXS decode (eager, -n 0:48, pp1024/tg128, in-binary A/B via
  the HANZO_IQ*_FALLBACK toggle): 30.4 -> 67.7-69.0 T/s = 2.27x, BEATING llama-HIP 67.6 (shipped v1.3.8
  = 31.5, matches). Prefill 75.3 -> 301.5 = 4.0x (IQ3 is NOT qmmq-capable, so MoE prefill rides the
  per-slot matvec core, which is now dp4a too). UNLIKE the launch-bound Q6_K case, the i-quant kernel
  win FULLY surfaces at the model level because i-quant decode was COMPUTE-bound: HANZO_ROCM_GRAPHS=1
  is FLAT/slightly lower (66.6) -- decode is no longer launch-bound. Weight BW at 68 T/s is only ~87
  GB/s (well under the 212 roofline) = the kernel is now grid/VALU-bound, not weight-fetch-bound; 68 is
  the practical wall (== llama-HIP). Tried `__device__` global grids (vs `__constant__`): SLOWER (65.4)
  -- the small grids fit the broadcast constant cache and global adds L1 pressure vs weight streaming,
  so `__constant__` is kept. IQ4_NL stays SCALAR (32-elem legacy block, not the 256-elem super-block
  the dp4a core strides; would need a separate core, and it is 4.5bpw = less bandwidth-critical).
## Decode op-fusion: gate+up shared-quantize + fused residual-add rmsnorm (+2.8%, bit-exact)
- rocprofv3 of the SHIPPED v1.3.8 decode (Qwen3-30B-A3B-Q4_K_M, GRAPHS=1) was the start: 1702
  kernels/token, GPU-busy 13.2ms = 98% of the 13.46ms wall (graphs already hide host-launch latency),
  matmul+attn 83% (roofline, untouched). Overhead histogram: quantize_q8_1 337/tok (7.02/layer),
  rmsnorm 193 (4.02/layer = Qwen3 QK-norm), cast 288 (DEAD -- the 0.11.9 f32-native cast cut was
  FLAT), badd 96, ucopy 96. The prior "40ms copies" hypothesis was wrong: those copies are MODEL-LOAD
  weight uploads, ~0.2% of decode. So the lever is removing overhead-kernel GPU-TIME, not dispatch
  latency -- and on gfx1151 each removed small kernel is worth ~2.7us, so kernel-count DOES surface.
- **moe_gate_up (gate+up shared quantize): +1.1-1.4 T/s.** gate_exps and up_exps consume the SAME
  routed token, so broadcast + q8_1-quantize it ONCE and dp4a both banks (`moe_matvec_pair`) instead of
  per bank. Decomplected `moe_matvec_dp4a` into `quantize_q8_1` + `moe_matvec_dp4a_act`; extracted the
  shared `QTensor::rocm_moe_bank` VRAM cache. CAVEAT (root-caused via a path probe): the experts load
  as `QMatMul::QTensor` (3D banks), NOT `RocmQuant` (2D) -- the first cut matched RocmQuant and silently
  fell back (FLAT). -48 quantize -48 ucopy/tok (337->289, 96->48). HANZO_MOE_GATEUP_FALLBACK A/B.
- **add_rmsnorm (fused residual-add + rmsnorm): +0.4-0.7 T/s.** `s=x+residual` (new residual stream) +
  `rmsnorm(s)*alpha` in ONE reduce.hip launch, vs a separate `badd` then `rmsnorm`. Wired QRmsNorm::
  forward_of_sum -> ops::rocm_rms_norm_of_sum (f32) -> RocmStorage::add_rms_norm. Only badd1
  (attn+residual -> ffn_norm) fuses; badd2 (moe+residual -> NEXT attn_norm) CANNOT -- it would span the
  per-layer device-map boundary (norm weight on the next layer's device). -48 badd/tok (96->48).
  HANZO_ADD_RMSNORM_FALLBACK A/B.
- STACKED: 1702 -> 1558 kernels/tok (-144, -8.5%); device-map-controlled decode A/B (-n 0:48,
  GRAPHS=1, clean rounds, vs shipped v1.3.8): 74.1 -> 76.0-76.2 T/s (+2.8%), coherent (17x4 / "Paris").
  Past llama-HIP 66; the residual gap to Vulkan 82 is STRUCTURAL (faster matvec/attn kernels, not fewer
  launches -- the matvec is already 88-99% roofline). Bit-exact: new `moe_matvec_pair` byte-equality
  (qmatvec_unified_numeric) + `add_rmsnorm_numeric` oracles nbad=0; all prior oracles unchanged.

## Fused q/k RMSNorm+RoPE (+2.8%) AND the quantize_q8_1-fold DEAD END (-22%) -- decode is busy-bound
- FRESH rocprofv3 (v1.3.11, GRAPHS=1, -n 0:48) RESOLVED the "where's the overhead" question that
  earlier agents disagreed on: decode = 12.17 ms GPU-busy/tok = **94.5%** of the 12.87 ms wall (the
  inter-kernel gap is only ~0.7 ms un-profiled; rocprof INSTRUMENTATION inflates it to ~1.87 ms, which
  is what the prior "5.3 ms overhead" estimate over-counted). matvec = 10.1 ms (bandwidth roofline,
  Q4_K+Q6_K dp4a). NON-matvec busy = 2.06 ms (quantize_q8_1 378us, rmsnorm 330, route 297, casts
  f32->f16 251 + f16->f32 126, add_rmsnorm 113, combine 108, reshape_cache 94, badd 93, ucopy 80,
  rope 76, silu 69). So decode is GPU-BUSY-bound, NOT launch-bound -- a fusion helps ONLY if it
  ELIMINATES work; one that ADDS redundant work loses even though it cuts launches.
- **DEAD END: folding quantize_q8_1 INTO the dp4a matvec (the "obvious" launch-cut) = -22% LOSS.** A
  fused `qmatvec_dp4a_q_core` that quantizes the activation to q8_1 in LDS then dp4a's (one launch
  instead of quantize + matvec) was bit-exact (nbad=0, fused vs 2-launch, Q4_K/Q6_K decode+MoE,
  f16/bf16/f32) but decode 77.4 -> 59 T/s. ROOT CAUSE: each matvec BLOCK re-quantizes the full
  activation into its own LDS (256 blocks for a 2048-row proj) + a `__syncthreads` before the dp4a --
  redundant serialized work injected into the bandwidth-bound matvec. Under HIP graphs the SEPARATE
  quantize_q8_1 launch is already ~free (graphs hide dispatch), so removing it buys nothing and the
  in-kernel quant costs. REVERTED. Lesson: do not fold a shared-once op into a many-block consumer.
- **WIN: fused q/k RMSNorm + positions-RoPE = +2.8% (77.4 -> 79.5 T/s), the true-elimination twin of
  moe_route.** ONE kernel (`rope.hip rope_norm_positions<TI,TO>`, one block per head-vector: f32
  rms-reduce -> scale -> *weight -> rope) replaces two standalone `rms_norm` launches + `rope_positions`
  AND the q/k intermediate write+read. SAME f32 math, NO redundant work -> rmsnorm 145 -> 49/tok,
  GPU-busy 12.175 -> 11.996 ms. `RocmDevice::rope_norm_positions` + engine
  `layers::rocm_qk_rms_norm_rope_positions` (fast path of `qk_rms_norm_rope_positions`, per-seq-offset
  shapes only; HANZO_QK_NORM_ROPE_FALLBACK forces the unfused chain). Bit-exact: `qk_rms_norm_rope_
  numeric` oracle fused-vs-fallback = 0.000000 (F32, bit-identical) / <f16-tol; coherent 30B. Interleaved
  A/B (5 rounds, GRAPHS=1): 77.36 -> 79.52 T/s, tight (79.2-79.7).
- RESIDUAL gap to Vulkan 83 (~3.5 T/s) is the casts (379us, f32<->f16 around the keep_f32 decode
  boundary) + quantize_q8_1 (377us) -- both resist clean fusion: the casts are dtype-tangled (removing
  one shifts cost across the F32-residual/F16-KV boundary; the f32-native cast cut was historically
  FLAT) and the quantize-fold loses (above). Closing it = the Vulkan strategy of fusing the WHOLE
  attention epilogue (norm+rope+cache, no casts), a multi-kernel rewrite, not a single lever.

## CUDA i-quant completeness: IQ1_S qmmq prefill + TQ1_0/TQ2_0 dp4a decode (bit-exact)
Two gaps closed on the CUDA path (GB10), both validated by `cuda_iquant_tests.rs` (now 13/13, was 11/11).
- **IQ1_S qmmq PREFILL (Part 1).** IQ1_S already had native dp4a DECODE but was MMQ-EXCLUDED at prefill
  because `mmq_common.cuh`'s `iq1s_grid_gpu` was a ZEROED `[512]` stub (mis-sized: `load_tiles_iq1_s`
  indexes `qs | ((qh>>3l)&7)<<8` = 11 bits = 2048). The whole int8-WMMA machine (`load_tiles_iq1_s`,
  `mmq_type_traits<IQ1_S>` via `vec_dot_q8_1_q8_1_*`, DS4 ds-layout) was already a faithful llama.cpp
  port -- only the table was missing. FIX: filled the real `iq1s_grid_gpu[2048]` verbatim from llama.cpp
  `ggml-common.h` (GPU-packed nibble codebook, DISTINCT from the iq_grids.rs decode codebook), added
  `mmq_instance_iq1_s.cu` (mirrors iq3_s), 2 ffi decls (`launch_mmq_gguf_iq1_s` + `_moe_iq1_s`), and wired
  `fast_mmq.rs` (supports/qk=256/`DsLayout::DS4`/both launcher tables). IQ1_S needs DS4 (not the IQ2/IQ3
  D4) because its delta bias rides the activation-sum term: ds = (d1q, d1q*delta). Bit-exact vs the GPU
  dequant-weight matmul: max_rel 5.6e-4 (vs ~1.5e-7 for the symmetric D4 types -- the gap is the f16
  `make_half2(d1q, d1q*delta)` ds rounding, not a bug; well under the 1e-3 gate). IQ1_M stays
  dequant-prefill (no MMQ kernel); both keep native dp4a decode.
- **TQ1_0/TQ2_0 native dp4a DECODE (Part 2).** Ternary types were dequant-fallback only. Added
  `qdp4a<DW_TQ2_0>`/`<DW_TQ1_0>` to `iquant_mmvq.cu` (NSUB=8, mirroring the IQ structs + the ROCm scalar
  `qdec<DW_TQ*>` bit layout) and wired `iquant_dp4a_suffix` (tq2_0/tq1_0) -> decode + MoE-decode route
  natively; prefill falls through to dequant-dense (not in fast_mmq `supports`). Symmetric ternary
  {-1,0,1} packs straight to SIGNED int8 -> dp4a directly, NO bias-sum term (unlike Q6_K/IQ1 centering).
  TQ2_0 (2-bit) is CLEAN: `((qs[half*32+m]>>2l)&3)-1`. TQ1_0 (base-3, 1.69bpw) FITS but is irregular:
  the 256 coords pack across digit-planes (5 base-3 digits/byte) so the contiguous-32 sub-block straddles
  planes (e>=5) and each coord needs a scalar `(uint8)(byte*pow3[n])*3>>8` unpack -- dp4a accelerates only
  the 32-wide dot, not the unpack. Both bit-exact vs CPU `to_float` (max_rel ~1.7e-4). NOTE: llama.cpp has
  NO CUDA ternary path at all (no vec_dot/mmq/mmvq) -- this is pioneering, so the ROCm `qdec` is the layout
  oracle.
- **Bench (native vs `HANZO_IQ_DEQUANT_FALLBACK=1`, 4096x4096, GB10).** Native: TQ2_0 decode 76.5us,
  TQ1_0 decode 390.6us, IQ1_S decode 311.7us, IQ1_S prefill(m=64) 180.6us. Fallback (dequant-dense):
  154-210 MILLISECONDS -- because these types have NO GPU dequant kernel, so the fallback DtoH's the whole
  weight, CPU `to_float`s 16M elems, and HtoD's it back EVERY call. So native is ~400-2700x (the ratio is
  dominated by the avoided CPU round-trip, not pure kernel-vs-kernel). Within native, TQ1_0 is ~5x TQ2_0
  -- the base-3 scalar unpack is the wall (compute-bound), exactly as the "fits-but-irregular" call
  predicted; TQ2_0's clean 2-bit dp4a is the real win. GOTCHA (unchanged): bindgen_cuda tracks .cu mtimes
  not .cuh -- a new mmq_instance_*.cu compiles fresh, but after a .cuh-only edit `touch` the dependents.

## CUDA i-quant decode: unpack_ksigns kills the KSIGNS gather -- IQ2_XXS 1.71x (the REAL GB10 wall)
The i-quant DECODE matvec (`iquant_mmvq.cu`) is ~75% of GB10 i-quant decode GPU-time (nsys). Diffing it
against llama.cpp's `vec_dot_iq2_xxs_q8_1` (the reference that was beating us) found TWO ROCm-port
artifacts, BOTH fixed bit-identically to match llama -- and the A/B isolates which one actually mattered.
- **THE LEVER -- kill the divergent KSIGNS `__constant__` gather (`unpack_ksigns`).** The sign codebook
  `KSIGNS_IQ2XS_D[idx]` was a per-coord `__constant__` load with a DIVERGENT index (each lane a different
  idx) -> serialized through the constant cache. llama computes it table-free: the 7-bit index's 8th sign
  is the popcount-parity of the other 7, so `__popc` reconstructs the 8-bit mask (`unpack_ksigns(v) =
  (v ^ ((popc(v&127)&1)<<7)) * 0x01010101`, broadcast for the byte selectors). PROVED bit-identical:
  `KSIGNS_IQ2XS_D[i] == i ^ ((popc(i)&1)<<7)` for all i (e.g. [1]=129=0x81, [2]=130, [4]=132). NO
  constant-cache gather on the decode hot path.
- **THE NEUTRAL HALF -- `__vsub4` 1-dp4a (matches llama, but flat).** Signs applied via CUDA-native
  `__vsub4((g^m)-m = -g)` -> ONE dp4a, vs the 2-dp4a `dp4a(g,u) - 2*dp4a(g&mask,u)` form RDNA3.5 needed
  (no `__vsub4`). Bit-identical. The A/B proves this alone is ~0% (the ROCm note's "2-dp4a vs vsub4 is a
  wash" holds on Blackwell too -- `__vsub4` is emulated post-Kepler).
- **A/B (kernel microbench, 4096x4096, GB10, `bench_iquant_decode`; controls IQ1_S/IQ4_XS byte-identical
  across runs -> harness thermal-stable):** ksigns types win, raw-byte types flat -- which ISOLATES the
  gather as the cause:
    IQ2_XXS 84.67->49.49 us = **1.71x** (51->87 GB/s) -- the dominant kernel
    IQ3_XXS 113.09->80.95 us = **1.40x** (57->79 GB/s)
    IQ2_XS  150.87->120.51 us = **1.25x** (32->40 GB/s)
    IQ2_S/IQ3_S (raw-byte signs, no gather to kill -> only the neutral __vsub4): FLAT (202/84 us)
- **CORRECTS the prior "47 GB/s VALU wall" conclusion.** The occupancy fork measured the BASELINE kernel
  (51 GB/s) and called it a VALU wall; it was actually the KSIGNS-GATHER wall. `unpack_ksigns` lifts
  IQ2_XXS to 87 GB/s == the ROCm/llama practical wall that GB10 had been LAGGING. Every prior "non-lever"
  (occupancy, graphs, op-fusion) missed it because none touched the codebook-decode sign math.
- DRY: one selector-based `dp4a_signed(b, sel, grid, u)` over all 5 sign types (IQ2_XXS/IQ2_XS/IQ3_XXS
  ksigns + IQ2_S/IQ3_S raw-byte broadcast); `iq_bytemask4` dropped; f32 scale kept (bit-exact vs the
  to_float oracle, only the integer grid*act dot is dp4a). FOLLOW-UP: `KSIGNS_IQ2XS_D` (iquant_grids.cuh)
  is now dead in the decode path -- drop from the table + `gen_iquant_grids.py`. Bit-exact: 13/13
  `cuda_iquant_tests` (max_rel ~1.7e-4 = the f32-weight-materialization floor). SHIPPED ml 0.11.26 /
  kernels 0.11.23.
- **MODEL-LEVEL: a GB10 NON-LEVER (the honest end-to-end result).** Same-engine A/B (hanzo-engine
  v1.4.1, ml 0.11.25 baseline vs 0.11.26 dp4a, Qwen3-4B-UD-IQ2_XXS = IQ2_XXS-dominated 204 tensors,
  `bench --prompt-len 128 --gen-len 256`): **FLAT** both graphs-ON (dp4a 66.1 vs base 66.2 T/s) AND
  eager (60.2 vs 60.2). The 1.71x KERNEL win does NOT surface end-to-end because GB10 hanzo i-quant
  decode is **HOST/launch-overhead-bound, not matvec-GPU-bound** -- the GPU finishes the (now faster)
  matvec and stalls on per-token host work, so shrinking matvec GPU-time doesn't move the wall. (The
  op-fusion profile's "75% matvec" was GPU-BUSY share, NOT wall share.) So dp4a joins occupancy /
  graphs-for-kernel / op-fusion / moe_route as a real kernel win that is a GB10 **model-level non-lever**;
  the actual GB10 decode lever is LAUNCH reduction -- which is exactly what the dense-Llama decode-graph
  gate (+6.8%, hanzo-engine v1.4.1) attacks. dp4a STILL ships and stays: bit-exact (zero regression),
  decomplects (one selector helper, drops a 128-entry table + a divergent constant gather, matches
  llama), and is a LATENT win for genuinely matvec-bound regimes (ROCm 30B-A3B eager surfaced the
  i-quant kernel 2.27x -- a different platform/bottleneck). NOTE: corrects this section's earlier
  "compute-bound, kernel wins fully surface" framing, which held on ROCm-eager but NOT GB10-under-graphs.

## CUDA fused moe_route router -- bit-exact, a GB10 model-level NON-LEVER but SHIPPED for backend symmetry
Ported the ROCm `moe_route` fusion (softmax->topk->renorm, one block/token) to CUDA + a bit-exact gate
(`cuda_moe_route_numeric`, nbad_id=0, max_w_err 5.96e-8) -- but the decode A/B on Qwen3-30B-A3B-UD-IQ2_M
is FLAT both graph regimes (graphs-ON 34.25 vs 34.37, eager 32.32 vs 32.38; +-3% noise, no direction).
WHY (the ROCm rule "a fusion helps only if it ELIMINATES work"): the fusion REPACKAGES the routing into
one block-serial launch, it doesn't eliminate it -- at decode ntok=1 the single 64-thread block
underutilizes the GPU exactly as the parallel ml-op sort/sum/div do, and CUDA decode is not router-
launch-bound (graphs buy ~6% total). ROCm's +13% came from its disproportionately expensive amdgcn
gpu-sort, which CUDA lacks. SHIPPED (ml 0.11.29) as the backend-TWIN of the ROCm moe_route: composing it
completes the abstraction -- ONE `quantized::moe_route` with ROCm + CUDA fast paths + the ml-op fallback,
symmetric backends -- which is MORE decomplected than CUDA-missing, so the earlier "2nd path" worry doesn't
hold (the ROCm fast path already exists; CUDA was the only backend still on the unfused ml-op chain). Flat
perf is fine to re-bench on a 256-expert model (DeepSeek-V4, 2x router math) where it may clear the floor.
ALSO in this release: dropped the now-dead `KSIGNS_IQ2XS_D` CUDA decode table (gen_iquant_grids.py
`DECODE_SKIP` -- unpack_ksigns reconstructs it in-kernel; the Rust ksigns stays for the CPU oracle).

## Vulkan prefill DEADLOCK fix (gfx1151 Strix Halo UMA) -- the dequant path self-deadlocks the allocator
hanzo Vulkan HUNG at the first long prefill of any GGUF on gfx1151 (Radeon 8060S / Strix Halo), while
llama.cpp-Vulkan AND hanzo's own ROCm ran the same model fine. NOT a shader/compute hang and NOT
push_desc/coopmat/subgroup (all `VK_*` kill-switches off still hang). ROOT CAUSE (gdb of the live hung
process): the engine thread blocks in the KERNEL at `__ioctl(DRM_IOCTL_AMDGPU_GEM_CREATE)` <- `amdgpu_bo
_alloc` <- RADV <- `VulkanDevice::raw_buffer` <- `upload_f32` <- `QTensor::dequantize` <- `QMatMul::
forward` (PREFILL). The Vulkan `VulkanQuant::forward` prefill `else` branch (rows > the gate) DEQUANTIZES
each Q4_K/Q6_K weight to a fresh ~100-235 MB f32 BO via `upload_f32`; under the deferred single command
batch (`BATCH_CAP`) NONE free until the end-of-forward flush, so a dense prefill re-expands the whole
model to f32 (~32 GB for an 8B) in fresh `amdgpu_bo_alloc`s. On the 64 GB-carveout + ~31 GB-GTT UMA the
accumulated BOs exhaust memory and GEM_CREATE BLOCKS waiting for a free that can only happen after the
still-recording batch flushes -- a self-deadlock. ROCm never hits it (prefill rides the int8-WMMA `qmmq`
GEMM, never dequantizes; that asymmetry is exactly why ROCm worked and Vulkan hung). Decode (rows==1)
also fine -- it uses the native `matvec_q4k_gpu` straight out of the block format, no dequant.
- FIX (`vulkan_prefill_gemm_max_rows`, quantized/mod.rs): for every dtype that HAS a native quantized
  GEMM (`matmul_q*_gpu` = {Q4_0,Q8_0,Q4K,Q5K,Q6K}) return `usize::MAX` -> ALWAYS use the GEMM, NEVER the
  f32-dequant fallback; types without a GEMM kernel return 0 (rows>1 dequantize, their only path, and a
  handful of such weights do not accumulate enough to deadlock). Liveness beats the old at-large-M
  throughput edge the dequant path won (the column-per-invocation GEMM re-reads the weight ceil(M/8)
  times so it is slower at big M, but it CANNOT deadlock). The gate is now effectively the "has a GEMM
  kernel?" predicate.
- BENCH STANDING (evo gfx1151, Qwen3-8B-Q4_K_M, pp512/tg128, idle box): hanzo ROCm decode 40.2 / prefill
  1002.6 T/s BEATS llama.cpp-HIP 38.95 / 953 (decode +3%, prefill +5%, with PagedAttention active) --
  ABOVE the prior "decode = parity" law. llama-Vulkan 41.07/1058; hanzo-Vulkan was the deadlock (now
  fixed). hanzo CPU 14.1/34 vs llama 18.93/556 (CPU is the known dispatch-bound gap, same F32-dequant
  pattern). So on its primary GPU path hanzo is now AHEAD of llama.cpp on gfx1151.
- DECOMPLECTION ROADMAP (the fix is liveness; the real cleanup is structural): the bug is a SYMPTOM of
  the missing uniform Vulkan GEMM. ROCm already decomplected this (`qmatvec_core<WTYPE>` / `qmmq_core
  <WTYPE>` -- ONE contraction, type injected as a trait); Vulkan still hand-writes per-type shaders AND
  only has GEMM for 5/22 types (the hole that forces the dequant fallback). L1: port the generic
  `qgemm<T>` to Vulkan (all types) -> DELETE `vulkan_prefill_gemm_max_rows` and the dequant `else`
  branch entirely (bug impossible by construction). L2: codegen the per-type `decode` into each backend
  from one block-format spec (extend gen_iquant_grids.py). L3: a kernel DSL (CubeCL/Triton/MLIR) so the
  contraction is written once and lowered to all backends -- collapses the backends x types x ops
  product to `types + ops + backends`. The 3 axes (decode value / parametric contraction / backend
  functor) want to be orthogonal.
- L1 ATTEMPT #1 -- naive per-column LDS tiling = a MEASURED DEAD-END (0.21x, bit-exact). `mul_mm_q4k_
  shared.comp` (env HANZO_VK_Q4K_TILED): one workgroup per output column cooperatively stages that
  column's Q4_K weight into LDS once, then 256 threads split the M rows -- weight VRAM traffic 64x->1x at
  M=512. BIT-EXACT to mul_mat_q4k (vulkan_q4k_tiled_prefill_matches_default, rel<1e-5, all shapes incl
  k=16384 LDS bound + M 2/8/9/64/512), but the perf A/B (m=512,nout=k=4096) is default 46.4ms vs tiled
  223.7ms = 0.21x (4.8x SLOWER). LESSON: weight re-read was NOT the prefill bottleneck -- the default's
  many-light-invocations (nout*M/8 register-tiled) beats few-heavy-workgroups (nout WGs, each thread
  loops full k, activations re-read from VRAM per column, ~16KB LDS caps occupancy). The default scalar
  f32 kernel runs at only ~187 GFLOP/s (gfx1151 ~10+ TFLOP) so there IS headroom, but the lever is
  real 2D tiling (BM x BN output tile staging BOTH a weight tile AND an activation tile in LDS, coalesced
  loads, sub-tile-per-thread -- llama mul_mm / ROCm qmmq) and/or wiring the int8 dp4a/coopmat prefill
  GEMM, NOT this 1D weight-only stage. The bit-exact + bench harness (vulkan_quant_tests.rs) is the
  reusable gate for the real 2D kernel; mul_mm_q4k_shared stays env-gated OFF (never default) as the
  documented negative. NOT committed as a win.
- L1 ATTEMPT #2 -- 2D-tiled GEMM = the REAL WIN (2.06x, correct). `mul_mm_q4k_tiled.comp` (env
  HANZO_VK_Q4K_TILED2D): a 64x64 output tile per workgroup stages a BMxBK(64x32) activation tile AND a
  BNxBK decoded-weight tile into LDS once per K-step (BK=32 == one Q4_K sub-block), then each of 256
  threads accumulates a 4x4 sub-tile from LDS -- BOTH operands read from VRAM once per K-step and reused
  across the tile (weight across BM rows, activation across BN cols), coalesced loads, full occupancy.
  Decode per element identical to mul_mat_q4k; accumulation is tiled (partial sums) so it matches within
  f32 reorder (vulkan_q4k_tiled2d_matches_default, rel<2e-3 across shapes incl partial tiles m=7/65,
  nout=320). PERF A/B (m=512,nout=k=4096): default 46.2ms vs tiled2d 22.4ms = 2.06x (187->390 GFLOP/s).
  This is the structure the column kernel + the 1D dead-end lacked. Still f32-scalar-bound (390 of ~10000
  GFLOP/s peak) so NOT yet llama-parity (1058 prefill T/s); the next layer is int8 dp4a or coopmat
  (tensor-core) staging on the SAME 2D tile -- that closes to parity. Committed env-gated
  (HANZO_VK_Q4K_TILED2D), ready to default-on for dense (woff==0) prefill after MoE-bank (woff!=0) + an
  end-to-end engine pass. The bit-exact/tolerance + bench harness gates both.
- L1 ATTEMPT #3 -- 2D-tiled int8 dp4a GEMM = the BIG WIN (9.35x, correct). `mul_mm_q4k_tiled_dp4a.comp`
  (env HANZO_VK_Q4K_DP4A): the same 64x64 2D tile, but the BK=32 contraction is an int8 dot-product
  (OpSDotAccSat) not f32 MACs. Activations are quantized to q8_1 once per prefill via the previously-
  UNWIRED quantize_act_q8.comp (now wired: VulkanDevice::quantize_act_q8 -> xq/xs/xsum), then per K-step
  the tile stages the BN cols' 32 q4 codes + d1/m1 and the BM rows' int8 xq + xs/xsum in LDS; each thread
  dp4a's its 4x4 sub-tile: acc += d1[col]*xs[row]*dot(q4,xq) - m1[col]*xsum[row] (the Q4_K x q8_1 affine
  identity, decode == mul_mat_q4k_dp4a). Gate: vulkan_q4k_dp4a2d_matches_default (rel<1.5e-2 -- q8_1
  activation quant adds ~0.5-1% on top of f32 reorder; a decode/pack bug is >>1.5%), all shapes incl
  partial tiles. PERF A/B (m=512,nout=k=4096): default 46.7ms / 2d_f32 22.5ms / 2d_dp4a 5.0ms =
  9.35x over default, 4.51x over the f32 tile (~1720 GFLOP/s effective). Lifts the engine's Vulkan
  prefill ~24 -> ~224 T/s (0.023x -> ~0.21x of llama-Vulkan 1058). The remaining gap to llama is
  tensor-cores: llama uses coopmat; dp4a is int8-ALU-bound. L4 = coopmat (f16 tile + cooperative-matrix
  ops) on the SAME 2D tile -> parity. Committed env-gated (HANZO_VK_Q4K_DP4A); this is the new best Q4_K
  prefill path, ready to default-on after the MoE-bank (woff!=0) + end-to-end pass. The bench harness now
  3-ways default/f32-tile/dp4a-tile. Both quantize_act_q8 + mul_mat_q4k_dp4a were written-but-unwired by
  an earlier pass; this wires the activation quant and supersedes the column dp4a with the 2D-tiled one.
- L1 DEFAULT-FLIP (shipped): the 2D tile is now the DEFAULT dense Q4_K prefill path (woff==0, m>1), no env
  needed -- mul_mm_q4k_tiled_dp4a where the device advertises integer-dot (new `int_dot8` gate: query
  PhysicalDeviceShaderIntegerDotProductFeatures at init), else the universal f32 mul_mm_q4k_tiled (2.06x).
  MoE banks (woff!=0), m==1, k>LDS-bound, and HANZO_VK_Q4K_LEGACY fall to the column kernel; the per-kernel
  envs {DP4A,TILED2D,TILED,LEGACY} force one path for the A/B gates. Validated: all 3 correctness gates
  pass vs the forced-legacy reference, and the default bench baseline now runs dp4a (5.0ms = the 9.00x
  default). VK_INT_DOT=0 forces the f32 tile. A gfx1151 engine gets ~224 T/s Vulkan prefill by default
  (was ~24). REMAINING: woff!=0 MoE-bank 2D path (untested -> still legacy) and L4 coopmat for parity.
- L4 ATTEMPT -- coopmat (tensor cores) = CORRECT but a measured LOSS to dp4a on gfx1151 (decode-bound).
  `mul_mm_q4k_coopmat.comp` (env HANZO_VK_Q4K_COOPMAT, gated on device coopmat): per 16-wide K-step the
  64 threads decode the Q4_K weight 16x64 slice to f16 LDS (transposed [K,N]) + cast the activation to
  f16 LDS, then subgroup 0 coopMatLoads from LDS and issues RM*RN(4x4) coopMatMulAdds into f32 accs --
  the llama-Vulkan tensor-core path. PROVED the matrix cores work on gfx1151 RADV (coopMatLoad-from-LDS
  validates + runs; correctness vulkan_q4k_coopmat_matches_default rel<1e-2 f16-rounding, all shapes).
  But PERF A/B (m=512,nout=k=4096): legacy 46.1 / dp4a 4.7 / coopmat 18.0 ms = coopmat is 0.26x of dp4a
  (3.8x SLOWER), only 2.56x over the column kernel. WHY: the kernel is DECODE-bound, not tensor-core-
  bound -- it re-decodes the weight to f16 every K=16 step (k/16=256 passes), and the Q4_K decode
  (get_scale_min_k4 + byte extraction per element) swamps the fast coopMatMulAdds; only subgroup 0
  computes while both subgroups decoded. So on gfx1151 dp4a-2D (4.7ms, the shipped default) is the
  PRACTICAL prefill ceiling, BEATING naive coopmat. llama's 1058 needs a DECODE-AMORTIZED coopmat (decode
  a large K-chunk to f16 LDS ONCE, reuse across many MulAdds, double-buffered) -- a bigger uncertain
  kernel; the f16-decode cost is the wall, not the cores. Committed env-gated OFF as the documented L4
  negative (mirrors the 1D dead-end); the coopmat infra + harness is proven for the amortized follow-up.
- L1 MAX-OUT sweep (the practical ceiling, measured): added a KERNEL-ISOLATED bench (`bench_matmul_q4k`:
  upload x+wq once, loop the GPU GEMM, one final synchronize -> the real per-matmul kernel cost, vs the
  host-wrapper bench whose per-iter 8MB upload+readback masked it). True kernel times (m=512,nout=k=4096,
  17.2 GFLOP): legacy 38.5 / f32-2D 19.0 / dp4a 3.4 / coopmat 17.4-19 ms -> dp4a = 5056 GFLOP/s = ~34% of
  the 8060S ~14.8 TFLOP f32 peak (~8.6% of int8 peak), 9x over the column kernel. TILE SWEEP pins the
  bottleneck as OCCUPANCY, not VRAM/LDS-reads: register-blocking the inner dp4a (LDS->regs) = FLAT (the
  compiler already did it / added VGPRs traded occupancy); BM=128 (half the weight re-read) = 3x SLOWER
  (10.7ms, occupancy collapse / register spill); BM=32 (more occupancy) = 3.72ms (slightly slower, re-
  read+barrier overhead). So 64x64 (4x4 sub-tile) is the OPTIMAL tile -- the shipped dp4a is already at
  its sweet spot. The residual gap to f32-peak is the INHERENT Q4_K cost (the per-sub-block f32 scale
  application acc+=d*xs*dot-m*xsum every K-step + the in-kernel decode + the q8_1 activation quant pass),
  not a tiling defect; closing it needs a structurally different kernel (decode-amortized coopmat, the
  one untried path -- my naive coopmat was decode-bound). CONCLUSION: dp4a-2D 64x64 (5056 GFLOP/s, 9x) is
  the validated practical max for blind kernel-tuning on gfx1151; the last-mile to peak is the amortized-
  coopmat follow-up (profiler-guided). The kernel-isolated bench (vulkan_q4k_kernel_bench) is the harness.

## Vulkan DECODE is the open frontier: op-count-serialization-bound (NOT recording-bound) -- fusion roadmap
- The L1 arc above fixed Vulkan PREFILL (dp4a-2D 9.35x default). Vulkan DECODE is the remaining gap vs
  llama: clean idle-box evo matrix (ml 0.11.32, Qwen3-8B-Q4_K, load<0.5) = hanzo-Vulkan decode 3.3 vs
  llama-Vulkan 41 T/s (~0.08x) AND vs hanzo's OWN ROCm decode 35 -- so it's a Vulkan-backend defect, not
  the kernel or the GPU. (ROCm clean matrix = 0.82-0.93x prefill / 0.79-0.91x decode across 0.6/4/8B =
  NEAR PARITY; ROCm is "there", Vulkan decode is not.)
- ROOT CAUSE (VK_PROFILE=1, one decode token, 8B): `flush: dispatch=1336 barriers=903 record=0.766ms
  submit=1.406ms fence_wait=142.827ms`. So decode is NOT recording-bound (CPU record 0.77ms = free ->
  command-buffer/graph REUSE would buy NOTHING, hypothesis killed by the profile) -- it is GPU-EXECUTION-
  bound on OP-COUNT SERIALIZATION: 1336 tiny M=1 dispatches chained through 903 GLOBAL memory barriers
  (vk::MemoryBarrier, full SHADER_WRITE->READ, emitted on TRUE read-after-write deps at vulkan_backend.rs
  ~3050) serialize the GPU at ~107us/op on compute that's only ~us. The barriers are CORRECT (real data
  deps); the ONLY way to remove them is to FUSE the dependent ops so the dep lives INSIDE a kernel.
- FIX = the SAME op-fusion campaign that took ROCm decode from launch/busy-bound to parity, ported to
  Vulkan .comp shaders (per the taxonomy: GPU-busy-bound decode -> lever is ELIMINATING ops, not faster
  arithmetic -- matvec is already roofline). Roadmap, each step ROCm-precedented + bit-exact-gated:
  (1) fuse attention epilogue: q/k RMSNorm+RoPE+KV-write into ONE shader (ROCm rope_norm_positions twin,
      +2.8% there) + residual-add+RMSNorm (add_rmsnorm); these ELIMINATE per-op barriers not just launches.
  (2) fuse FFN: gate+up shared-quantize + silu+mul.
  (3) fuse routing: moe_route shader (ROCm +13%; Vulkan pays the same sort cost so it should surface).
  (4) uniform Vulkan GEMM for all 22 formats -> deletes the dequant fallback + the deadlock gate by
      construction (the structural endpoint of vulkan_prefill_gemm_max_rows).
  STATUS: characterized down to dispatch/barrier counts, NOT yet shipped -- labeled frontier, not result.
- PAPER: the full cross-backend kernel-optimization catalog (every KEPT lever + every measured DEAD-END +
  the 4-regime bottleneck taxonomy + the surfacing rule "a kernel change helps end-to-end only if it
  eliminates work in the BINDING resource") is written up at ~/work/hanzo/papers/hanzo-kernel-optimization.tex
  (paper 06, the kernel-level companion to paper 05's model-level parity result; builds clean, 12pp). The
  same-kernel regime flips (dp4a i-quant: ROCm 2.27x / GB10 flat; moe_route: ROCm +13% / CUDA flat) are
  the sharpest confirmation of the taxonomy.

### Vulkan decode fusion campaign -- rungs 1+2 SHIPPED to ml (bit-exact), engine scaffold found ready
- Built the first two op-fusion kernels the barrier-serialization diagnosis calls for, both bit-exact-gated
  (tests/vulkan_quant_tests.rs, `matches_unfused`):
  (1) **`rope_norm.comp`** -- fused per-head RMSNorm + NeoX RoPE in ONE dispatch (VulkanStorage::rope_norm
      + host wrapper rope_norm_f32). Bit-identical to rms_norm.comp o rope.comp (same f32 ops, same
      order): maxabs 3.6e-7..4.8e-7 over decode (t=1) + prefill + GQA shapes. Removes the q_norm->rope and
      k_norm->rope global barriers (4 attn-epilogue ops -> 2 rope_norm calls).
  (2) **`add_rmsnorm.comp`** -- fused residual-add + RMSNorm, returns BOTH (s=x+res, y=rms_norm(s)*alpha)
      in one dispatch (mirrors ROCm add_rms_norm). s bit-IDENTICAL (maxabs 0.0), y 4.8e-7. Removes the
      badd->rmsnorm barrier.
  cos/sin convention = indexed by i_t=row%t (engine pre-slices the [s,d/2] cache to position, NOT a
  positions array) -- matches the existing rope.comp + the engine's Vulkan cos/sin path.
- ENGINE SCAFFOLD ALREADY EXISTS (swarm built the engine half): `vulkan_qk_rms_norm_rope` (layers.rs:2687,
  env HANZO_VK_FUSED_QKNORM at 2767) extracts all 6 Vulkan storages then returns Ok(None) with comment
  "Fused per-head RMSNorm+RoPE is not yet wired in hanzo-ml" -- i.e. it was WAITING for exactly this
  rope_norm op. Wiring = ~15 lines: `qv.rope_norm(q_l, qwv, qw_l, q_eps as f32, cv, cos_l, siv, sin_l)?`
  for q+k, wrap `Tensor::from((Storage::Vulkan(out), q_l.shape().clone()))` (mirror rocm_qk_rms_norm_rope_
  positions at 3003). silu_mul is ALREADY wired (ops.rs:3339); add_rms_norm engine path is ROCm-only
  (ops.rs:1686) -> add a Vulkan branch. DRY NOTE: wire the engine to the SHIPPED rope_norm (don't let the
  swarm add a parallel `qk_norm_rope_gpu`); reconcile to ONE op.
- PENDING (blocked on engine repo being free of the concurrent ml-bump push; do on a FEATURE branch since
  the swarm churns engine main): wire vulkan_qk_rms_norm_rope -> rope_norm, [patch.crates-io] -> local ml,
  build --features vulkan, measure decode T/s (the end-to-end proof; kernels are necessary-not-sufficient
  until wired). Then publish ml (rope_norm + add_rmsnorm) at the next patch + land the engine wiring.

### Vulkan decode rung-1 (rope_norm) is MODEL-FLAT -> the binding resource is the DECODE MATVEC, not op-fusion
- Wired rope_norm into engine vulkan_qk_rms_norm_rope (env HANZO_VK_FUSED_QKNORM), built --features vulkan
  with [patch]->local ml, in-binary A/B on Qwen3-8B-Q4K Vulkan decode: BASELINE 3.3 T/s (307.4ms/T) vs
  FUSED 3.3 T/s (305.8ms/T) = FLAT (0.5%, noise). Bit-exact + correct (model runs coherently), but the
  decode needle did not move. This is the PAPER'S OWN SURFACING RULE confirming itself: rope_norm
  eliminated work (norm/rope ops + ~72 of 1336 dispatches + their barriers) but NOT in the BINDING
  resource -> flat. The 903-barrier diagnosis was right that decode is GPU-bound, but the barriers I
  removed were on CHEAP ops; the expensive dispatches remain.
- ROOT CAUSE (read mul_mat_q4k.comp): DECODE (M=1) routes to the SCALAR COLUMN kernel -- one invocation
  per output row, FLOAT MACs (`acc += wlo*x[..] + whi*x[..]`), NO dp4a, NO subgroup reduction. It runs
  ~15x BELOW the bandwidth roofline. (The dp4a-2D win was PREFILL only: matmul_q4k_gpu_off routes dp4a
  iff m>1; m==1 falls to this scalar column kernel.) ROCm/CUDA/Metal decode at roofline via dp4a/tensor
  mat-vecs; Vulkan decode is the ONLY backend still scalar -> 0.08x vs llama, 12x behind llama's OWN
  Vulkan (which has an optimized mul_mat_vec). THE LEVER = a dp4a DECODE matvec (one WG per output row,
  threads dp4a-partial over k, subgroup-reduce), reusing the prefill dp4a infra (quantize_act_q8 + the
  Q4_K x q8_1 affine identity + the bit-exact harness). NOT more op-fusion.
- rope_norm + add_rmsnorm stay shipped (bit-exact, decomplecting, latent wins for a future busy-bound
  regime -- like the documented ROCm/CUDA model-level non-levers). The campaign redirects to the matvec.

### CROSS-BACKEND PARITY MATRIX (hanzo vs llama.cpp, clean idle-box, ml ~0.11.32, Qwen3 0.6/4/8B)
- CUDA (GB10/spark): decode geomean 0.99x (0.96/0.99/1.02 -- BEATS llama on 8B), prefill geomean 0.60x
  (0.52/0.60/0.70). Decode PARITY (stronger than ROCm); prefill = Blackwell tensor-core MMQ frontier.
- Metal (M4 Max/dbc): decode 0.96x (4B+8B), prefill 0.84x(4B)/0.91x(8B). Near parity. BUG: Qwen3-0.6B
  emits NaN/Inf logits on the ml Metal forward (quant-independent, specific to tied-embedding 0.6B) ->
  upstream ml Metal-forward bug to fix+republish.
- ROCm (gfx1151/evo): decode 0.79-0.91x, prefill 0.82-0.93x. Near parity both.
- Vulkan (gfx1151/evo): decode 0.08x (the scalar-matvec outlier above); prefill dp4a-2D shipped.
- LAW: decode = at/near parity on ROCm+CUDA+Metal (bandwidth-bound, both hit the wall); the ONE gap is
  Vulkan decode (scalar matvec). Prefill = universal frontier (0.60-0.93x), worst on Blackwell (llama
  CUDA prefill elite), best on ROCm (llama iGPU prefill weak). Universal prefill lever = tensor-core MMQ
  + kill f32<->bf16 casts.

### Vulkan decode dp4a matvec = the REAL lever, VALIDATED +1.8x (the redirect paid off)
- The rope_norm-flat result correctly redirected to the matvec. Confirmed decode (m==1) used a SCALAR
  subgroup matvec (mul_mat_vec_q4k_sg: subgroupAdd reduction but f32 decode+MAC per weight). Routed
  decode through int8 dp4a instead: quantize_act_q8(x,1,k) -> dp4a the Q4_K codes (the existing column
  dp4a mul_mat_q4k_dp4a.comp at mcount=1, now registered). matvec_q4k_dp4a host method + bit-exact gate.
- RESULT (Qwen3-8B-Q4K Vulkan decode A/B, in-binary via env, idle evo): scalar 6.0 T/s (166.9ms/T) ->
  dp4a 10.8 T/s (92.3ms/T) = **1.8x**. Proves Vulkan decode was MATVEC-COMPUTE-BOUND (the scalar f32
  decode was the wall), NOT op-count/barrier-bound (rope_norm was flat). CORRECT: dp4a vs scalar matvec
  rel=3.7e-3 (<2e-2 gate, q8_1 activation-quant floor) across 4 shapes; one-shot output BYTE-IDENTICAL
  to the scalar baseline (same seed). SHIPPED default-on where int_dot8 (HANZO_VK_DP4A_DECODE_OFF forces
  scalar); mirrors the prefill dp4a flip. Committed on ml branch wip/vulkan-decode-fusion.
- STILL 0.26x of llama-vulkan (41 T/s) -- the column dp4a is one-thread-per-row (lost the subgroup
  reduction's memory parallelism). NEXT LEVER = a SUBGROUP dp4a matvec (one subgroup per row,
  cooperative coalesced weight read + dp4a + subgroupAdd) = combine both wins -> toward parity. Then
  the same dp4a-decode treatment for Q5K/Q6K (still scalar). rope_norm + add_rmsnorm stay shipped
  (latent/decomplecting, model-flat here as the taxonomy predicted).
- SEPARATE PRE-EXISTING BUG (not the matvec): `hanzo run -i` one-shot on this raw Q4K GGUF emits
  garbage ("53A(Parameter...") for BOTH scalar AND dp4a (identical) -- a detokenize/template/sampling
  bug in the run path; the bench forward is clean (no NaN, tight T/s). Worth a separate fix.

### Subgroup-dp4a decode = MEASURED DEAD END; column dp4a (one-thread/row) is the decode optimum
- Tried mul_mat_vec_q4k_dp4a_sg (subgroup-reduced int8 dp4a: one subgroup/row, lane-strided super-
  blocks, subgroupAdd -- combine dp4a's compute win with the scalar-subgroup matvec's memory
  parallelism). Bit-exact (gate: both column + subgroup vs forced-scalar, rel 0.4%). PERF A/B (8B-Q4K
  decode, prompt-len 64): subgroup-dp4a 5.9 T/s = FLAT vs scalar 5.9, LOSES to the column dp4a 10.8
  (1.8x). WHY: decode wants MANY ROWS IN FLIGHT -- the per-thread COLUMN layout runs 64 rows/workgroup
  (memory-level parallelism across rows); the subgroup layout starves it at 1-2 rows/wg (gl_NumSubgroups),
  and dp4a's compute win cannot compensate a 32x parallelism cut. So the COLUMN dp4a stays the default
  (env HANZO_VK_DP4A_DECODE_SG forces the subgroup variant for the record). Mirrors the prefill finding
  (naive coopmat lost to dp4a): on this APU the simplest high-occupancy layout wins. Column dp4a 1.8x is
  the Vulkan Q4_K decode optimum for blind tuning; closing the rest to llama's 41 needs a different
  structure (llama's mul_mat_vec packs multiple super-blocks/thread + vectorized loads), not subgroup.
- TEST RIGOR FIX (a regression my earlier column-dp4a default-flip merged unnoticed -- I'd only run the
  dp4a gate, not the full GPU suite): the dp4a default put q8_1 activation-quant error (~0.4%) into the
  production matvec, breaking the EXACT-vs-CPU checks. Fixed by forcing the scalar path in the exact
  correctness tests (run_case Q4K -> matvec_q4k_scalar; vulkan_qmatmul_forward_matches_cpu sets
  HANZO_VK_DP4A_DECODE_OFF) and adding explicit matvec_q4k_{scalar,dp4a,dp4a_sg} host methods so the
  gate tests every path vs forced-scalar. vulkan_quant_tests decode set now GREEN.
- PRE-EXISTING (NOT this decode work, prefill path, left for the prefill owner): vulkan_qmatmul_prefill_
  matches_cpu Q4K fails deterministically (prefill dp4a-2D default vs exact f64 ref -- same q8_1 class,
  shipped by an earlier prefill default-flip without updating the test; fix = force HANZO_VK_Q4K_LEGACY
  or a per-dtype tolerance, but NOT via more env-set since the tiled tests already FLAKE on env-race).
  vulkan_q4k_tiled{2d,_prefill}_matches_default FLAKE (pass/fail at varying shapes) -- the tests'
  set_var/remove_var of HANZO_VK_Q4K_{LEGACY,TILED} races under cargo's parallel test threads; real fix
  = run the Vulkan GPU tests serially (--test-threads=1 or a serial guard), a test-infra change.

### R2 (2 output rows/thread) decode = 2nd MEASURED NON-LEVER -> column dp4a is bandwidth-access-bound
- mul_mat_q4k_dp4a_r2.comp: column dp4a but 2 output rows/invocation -- shared activation (xq) loaded
  once/super-block + reused across both rows, 2 independent dp4a chains for ILP (env HANZO_VK_DP4A_DECODE
  _R2, gate adds "r2" leg, bit-exact). A/B (8B-Q4K decode, same load): R2 5.8 vs column 5.9 = FLAT.
- CONCLUSION (two negatives bound it): subgroup-coalescing FLAT + ILP FLAT -> the column dp4a is neither
  occupancy- nor latency-bound; it is MEMORY-ACCESS-PATTERN-bound at ~22% of the 230 GB/s roofline (10.8
  T/s idle), while ROCm's HIP qmatvec hits ~72% and llama-Vulkan ~85% on the SAME APU. So the column dp4a
  1.8x is the practical ceiling for blind GLSL kernel tuning on RADV; the residual 3.8x to llama is NOT
  reachable by occupancy/ILP/subgroup tweaks. The real lever is the RADV-specific memory layout llama's
  mul_mat_vec uses (vectorized uvec4/uvec2 weight loads at the right 16-byte alignment + the staged
  activation tile) -- which needs RGP profiling to target, not blind iteration. Both subgroup + R2 kept
  env-gated as documented negatives (HANZO_VK_DP4A_DECODE_{SG,R2}); column stays default.
- NOTE on bench hygiene: evo runs a continuous swarm (load ~4-60); idle decode = 10.8, load-4 decode =
  5.9 -- absolute Vulkan-decode numbers need a truly idle box, but in-binary A/Bs at matched load are
  valid (the dp4a 1.8x, subgroup-flat, R2-flat all reproduce relatively).

### V2 (uvec2 vectorized weight loads) = 3rd MEASURED NON-LEVER -> blind RADV tuning DEFINITIVELY exhausted
- mul_mat_q4k_dp4a_v2.comp: column dp4a with the weight SSBO typed uvec2, the 8 qs u32/chunk read as 4
  uvec2 (8-byte, alignment-safe: blocks are 36 u32 even, qs at base+4 even) -- wider memory transactions
  for the bandwidth-access-bound path (env HANZO_VK_DP4A_DECODE_V2, gate "v2" leg, bit-exact rel 3.7e-3).
  A/B (8B-Q4K, load 1.87): V2 5.9 vs column 6.0 = FLAT (RADV likely already vectorizes the sequential
  scalar loads; transaction size is not the lever either).
- DEFINITIVE BOUND -- THREE negatives now fence the Vulkan Q4_K decode optimum: subgroup-coalescing FLAT,
  R2-ILP FLAT, V2-transaction-size FLAT. The column dp4a (one thread/output-row, 64 rows/wg, int8 dp4a,
  1.8x = 10.8 T/s idle) is the practical ceiling for BLIND GLSL kernel tuning on RADV/gfx1151. The
  residual gap (3.8x to llama-Vulkan 41, 3.3x to hanzo's OWN ROCm-HIP 35 on the SAME APU) is NOT an
  occupancy/ILP/coalescing/vectorization problem -- it is a fundamental RADV-Vulkan-compute vs ROCm-HIP
  codegen/driver-efficiency gap for int8 matvec on this APU (same kernel math, same dp4a, 22% vs 72%
  roofline). Closing it needs RGP/radv-shader-stats profiling to find what RADV mis-compiles vs HIP (or
  accepting the path difference), NOT more blind kernel variants. All three negatives kept env-gated
  (HANZO_VK_DP4A_DECODE_{SG,R2,V2}), bit-exact, as the documented boundary; column dp4a stays default.
- VULKAN DECODE CAMPAIGN CLOSED at 1.8x shipped. Honest cross-backend standing: decode parity on ROCm
  (0.8-0.9x), CUDA (0.99x), Metal (0.96x); Vulkan decode 0.26x of llama (1.8x lifted from 0.08x) = the
  one backend where hanzo's decode trails, bounded as a RADV-vs-HIP codegen gap not blind-tunable further.

### Vulkan GPU test suite: fixed the env-race flakiness (was the "pre-existing prefill failures")
- The intermittent qmatmul_prefill / tiled{2d,_prefill} / dp4a2d / coopmat failures were NOT kernel bugs:
  6 tests select a Q4_K kernel via process-global env (HANZO_VK_Q4K_{LEGACY,TILED,TILED2D,DP4A,COOPMAT},
  HANZO_VK_DP4A_DECODE_OFF) with set_var/remove_var, and cargo runs tests in parallel threads -> one
  test's remove_var clobbered another's set_var mid-run, so a forced-LEGACY prefill silently ran dp4a
  (q8_1 error 0.10) at random. FIX: a single poison-safe `static ENV_LOCK: Mutex<()>` that EVERY
  env-using test holds for its whole body (serializes them; non-env tests still parallel) + force the
  exact path in the correctness tests (qmatmul_forward: HANZO_VK_DP4A_DECODE_OFF; qmatmul_prefill:
  HANZO_VK_Q4K_LEGACY). Result: 28 passed / 0 failed, 3x consecutive, deterministic. (The #[ignore]'d
  bench tests also use env -- lock them too before running with --ignored.) So the earlier "pre-existing
  prefill kernel discrepancy" framing was WRONG: there is no kernel bug, it was test-harness env-racing.

### 30B-A3B MoE cross-backend (evo, the original-ask MoE data point)
- ROCm: hanzo prefill 1002 / decode 67.9 T/s vs llama 1150 / 66.0 = 0.87x prefill, **1.03x decode (BEATS
  llama)**. The shipped ROCm MoE stack (indexed dp4a decode + moe_route + qmmq MoE prefill) is at/above
  parity on the 30B-A3B-Q4K too -- confirms the dense cross-backend law extends to MoE.
- Vulkan: hanzo TIMED OUT (>600s, rc=143) -- the known-weak Vulkan decode (0.08-0.26x dense) is too slow
  on the 30B MoE to finish the 512pp/128tg bench in 600s (llama-Vulkan does 1108/88.5). Not a new bug;
  the Vulkan-decode codegen ceiling (RADV-vs-HIP, documented above) just makes the big MoE impractical.
- CUDA (spark) + Metal (dbc) 30B not benched (boxes busy with the prefill/NaN fix agents + the deepseek
  swarm); ROCm 30B numbers already canonical in the MoE sections above.

### Qwen3-8B-Q4_K_M produces GARBAGE on Vulkan (8B-specific forward bug) -- precisely isolated, OPEN
- REPRO: `hanzo run -i "The capital of France is /no_think" --format gguf -f Qwen_Qwen3-8B-Q4_K_M.gguf`
  on the Vulkan engine -> byte-identical garbage from token 1 (`53A(Parameter6,936$((Parameter...`).
- ISOLATION (what it is NOT): 8B-CPU is COHERENT ("...Paris"); 4B-Vulkan is COHERENT (both with/without
  --seed) -> NOT the run-path/chat-template/detokenizer/sampler, NOT --seed, NOT a universal Vulkan bug.
  It is 8B-SPECIFIC on the Vulkan forward, deterministic, from the first token.
- RULED OUT by A/B (all give the SAME byte-identical garbage): Q4_K matmul (default dp4a-2D, scalar
  decode HANZO_VK_DP4A_DECODE_OFF, legacy column HANZO_VK_Q4K_LEGACY -- all identical) AND the BufPool
  bucketing (HANZO_VK_POOL_EXACT=1 forcing exact-size keys -- still garbage, so not stale-tail reuse).
- KEY DEDUCTION: garbage is IDENTICAL regardless of which Q4_K kernel runs -> the source is COMMON to
  all variants, i.e. NOT Q4_K. The heavy 8B kernels untouched by any toggle = **Q6_K** (lm_head +
  ffn_down + attn_v are Q6_K in Q4_K_M) and attention/RMSNorm at n_embd=4096. Q6_K is the prime suspect:
  8B ffn_down k=12288 (48 Q6_K superblocks) vs 4B 9728 (38) -- a block-count/u32 threshold bug in
  mul_mat_q6k / mul_mat_vec_q6k at k>~40 superblocks would garble the FFN (and lm_head) -> all tokens,
  while 4B stays under it. NEXT: force Q6_K through dequant (vulkan_prefill_gemm_max_rows Q6K->0) to
  confirm, then bisect mul_mat_q6k for the large-k bug. (Vulkan is the weak backend; this is a real
  correctness bug that makes 8B-Q4K_M unusable there -- distinct from the decode-perf codegen ceiling.)

### DEFINITIVE re-bench (the honest "are we beating llama?" answer) -- NEAR-PARITY, NOT beating
- Fresh matched-load A/B on evo (gfx1151, hanzo-rocm vs llama-HIP, pp512/tg128, -n forced-all-GPU),
  box under the continuous swarm (load ~3.65 -- NOT idle; absolute T/s depressed, but the hanzo-vs-llama
  RATIO at matched load is valid, both suffer the same steal):
    Qwen3-8B-Q4_K_M (dense):  hanzo  968.5 pp / 32.3 tg  vs llama 1022 pp / 37.1 tg = prefill 0.95x, decode 0.87x
    Qwen3-30B-A3B-Q4K (MoE):  hanzo  953.5 pp / 55.8 tg  vs llama 1112 pp / 63.6 tg = prefill 0.86x, decode 0.88x
- HONEST CONCLUSION: hanzo is at **NEAR-PARITY, TRAILING (~0.86-0.95x)** on ROCm across dense + MoE,
  prefill + decode. We are **NOT** reliably "beating" llama.cpp. The earlier "30B decode 1.03x BEATS
  llama" was FAVORABLE VARIANCE, not a durable win: hanzo decode variance here was +-6.0 T/s vs llama's
  +-1.97 -- hanzo is MORE sensitive to background load, so a lucky low-contention window read as a "beat."
  On a truly idle box both numbers rise and the ratio recovers toward the canonical ~0.91x, but "beating"
  is not evidence-supported. The prior optimistic scoreboard mixed fresh + compacted-session numbers;
  this section is the corrected, freshly-measured record. Cross-backend law holds: decode near-parity,
  prefill the frontier -- but "near-parity" means ~0.9x, not >=1.0x.
- PUBLISH STATUS (blocked, ops): ml is version-staged at 0.11.36 on main (decode dp4a + Vulkan decode
  cleanup + DS4 Sinkhorn all merged, vulkan_quant_tests 28/0 green), but crates.io publishing is BROKEN:
  the publish.yml CI (workflow_dispatch + bare-semver-tag trigger) FAILS on every run (incl. a fresh
  2026-07-02 dispatch) with "CARGO_REGISTRY_TOKEN is empty" -- the repo secret is LISTED (set 2026-06-23)
  but reads empty at runtime on the self-hosted runner hanzo-build-linux-amd64. Zero successful publish
  runs since 0.11.19, so ml 0.11.20-0.11.36 were never published via CI (engine has been git-pinning ml).
  FIX (ops, NOT code -- I won't fabricate/plaintext a registry token): re-provision CARGO_REGISTRY_TOKEN
  from KMS into the repo/org secret AND ensure the self-hosted runner can read it, then re-fire
  `gh workflow run publish.yml -R hanzoai/ml --ref main` (idempotent: the script skips already-uploaded
  crates). Code + version are ready; publishing awaits the runner-secret fix.

## DSL collapse Phase 2 (ROCm norm) -- DSL is BIT-EXACT but the tuned incumbent WINS -> KEEP (survivor)
The ROCm mirror of the Vulkan `.spv` collapse was built + validated for `rms_norm` + `add_rmsnorm`, and
the result is a NEGATIVE (correctly gate-blocked): the DSL twin is bit-exact but ~84-92% of the
hand-written kernel, so the incumbent STAYS. This is the OPPOSITE of Vulkan (where the incumbent was a
naive one-invocation-per-row `.comp` and the DSL block kernel won ~10x) -- the ROCm `reduce.hip` norm
kernels are ALREADY warp-shuffle-optimal (adaptive 32/1024-thread block + `__shfl_xor` + one cross-warp
`s_sum` pass), so there is no headroom for the DSL to reclaim, only codegen overhead to lose.
- **Integration design (proven, the ROCm seam):** NOT runtime cubecl-hip dispatch -- cubecl-hip's
  `ComputeStorage` only `alloc()`s via `hipMallocAsync` into its own `StorageId` pool, with NO API to
  adopt ml's external `rocm-rs` device pointers, so a runtime handoff forces a per-call host round-trip
  (~15x slower on a 67 MB rms_norm; fails any perf gate). Instead: the `#[kernel]` Rust in
  `hanzo-kernel/src/norm.rs` is the ONE source; cubecl-hip LOWERS it to a HIP source STRING (captured
  via `CUBECL_DEBUG_LOG=<file>` during the `*_rocm_bit_exact` test); the `.hip` artifact is checked into
  `hanzo-rocm-kernels/src/kernels/dsl_*.hip`; ml compiles + launches it through its EXISTING hipcc +
  rocm-rs pipeline on its own device buffers -- zero-copy, single runtime, no 2nd HIP context on the
  fragile gfx1151/ROCm-7.13 stack. cubecl-cpp is a pure host transpiler; a pure-no-GPU codegen was
  blocked only because cubecl 0.10's `create_dummy_kernel` omits the client its `new` requires.
- **ABI facts (from the real generated source):** cubecl emits an unused `info_st*` metadata param even
  for metadata-free kernels (a cached zeroed dummy satisfies it -- the body never reads it); shared
  memory is DYNAMIC (`extern __shared__`, launch `sharedMemBytes = nt*4`); kernel params are in the DSL
  DECLARATION order (x,w,out,eps,ndim) + `info_ptr` last; entrypoints are dtype-suffixed
  (`rms_norm_blk_f_f32`, and `rms_norm_blk_f_` for f16). eps/ndim are DEVICE BUFFERS (cubecl has no
  inline scalar args on HIP) -- the incumbent passes them inline in registers, which is exactly the
  residual overhead (a per-block dependent load the DSL can't avoid).
- **The kernel work that took it 50%->~88%:** (1) `rms_norm_blk`/`add_rmsnorm_blk` reduce/scale in f32
  regardless of I/O dtype F (was F-typed -> a latent f16-accumulation bug; byte-identical for F=f32 so
  the shipped Vulkan spv is unchanged), `eps: &Array<f32>`. (2) An `island!`-gated block reduction:
  ROCm uses `plane_sum` (lowers to inline `__shfl_xor`, == the incumbent) + a SINGLE cross-warp barrier
  (every warp re-reads the per-warp partials and re-reduces, no broadcast), while `default` keeps the
  shared-mem tree so Vulkan/CPU are unchanged. nt=1024 to match the incumbent's wide-row thread count.
- **Numbers (interleaved-min A/B, gfx1151, matched load; `dsl_norm_bench` in rocm_backend/mod.rs):**
  bit-exact f32 max_rel ~1e-6, f16 ~5e-4. Perf dsl/incumbent: rms_norm 88% (rows=1) / 84% (rows=512);
  add_rmsnorm 87-88% (rows=1) / 91-99% (rows=512, higher because its heavier compute amortizes the fixed
  per-block overhead). ALL below the >=97% gate -> incumbents retained, exactly like the dp4a quant
  cores (the documented stop-line survivor). softmax + rope share the same tuned-incumbent situation, so
  they are the same class: the DSL provides COVERAGE (bit-exact portability), the hand-written kernels
  keep the ROCm PEAK. This is the migration policy in `hanzo-kernel/src/lib.rs` working as written.
- **Gotchas for the next agent:** `island!` needs a `#[comptime] <name>: Target` param (Target isn't a
  launchable arg); run helpers derive it via `Target::of(client)`. cubecl if-else expr arms must be the
  same CubeType (init a `let mut v = 0f32;` then conditional-assign, don't `if..{smem[t]} else {0f32}`).
  f16/f32 of one kernel share the `info_st` preamble ONLY if buffer-count matches (they do) -- merge by
  appending the 2nd `extern "C"` block, but keep the f16 preamble (it adds `#include <hip/hip_fp16.h>`).
  `clippy -D warnings` is PRE-EXISTING-red on ml+hanzo-kernel (the 106x `*const *mut c_void as *mut
  c_void` idiom + operator-method-confusion) -- not introduced here.
