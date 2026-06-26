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
