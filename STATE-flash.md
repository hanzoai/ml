# STATE-flash — portable DSL flash-attention (CPU-authored, GPU-validation-pending)

Branch `dsl-flash-attn` (off `origin/main`). Deliverable: `hanzo-kernel::flash::flash_attn` — the tiled
online-softmax attention primitive, one DSL source → every backend, with an f16 cooperative-matrix
island over a scalar CPU oracle. **Authored + CPU-gated this session; NOT GPU-benched (evo/spark/dbc are
sibling-owned). Do NOT publish `hanzo-kernel` until a GPU validates a real forward.**

## The mechanism (why this is the #1 cross-backend lever)

Flash attention never materializes the `[heads, seq_q, seq_k]` score matrix. It streams the keys in
tiles of `Bc=16`; for each key-tile it computes only a `Br×Bc` score tile, folds it into a running max
`m`, running denominator `l`, and output accumulator `Of[Br][d]`, rescaling `l`/`Of` by `exp(m_old −
m_new)` on the fly. One `Br×Bc` tile is the entire score footprint that ever exists.

That single property is the mechanism behind BOTH open gaps the brief names:
- **CUDA prefill** materializes the full `[40,512,512]` f32 scores → `softmax_f32` (~120 µs/inst, ~12×
  off roofline) + cutlass QK/PV. Flash removes the materialization and fuses QK→softmax→PV.
- **Vulkan decode** non-matvec cost is the attention op-chain; the tiled form shrinks it.

`flash_attn` is the online-softmax EVOLUTION of `attn::sdpa_blk`: `sdpa_blk` streams keys one-at-a-time
per thread (already online, but scalar and un-tiled); `flash_attn` streams them in 16-wide tiles so the
two contractions (Q@Kᵀ, P@V) map onto the f16 tensor cores.

### The island (accel = cmma, default = scalar oracle)
The two matmuls are each an `island!`:
- accel arm (`cuda | rocm | vulkan | metal`): stage the Q/K (then P/V) tiles to f16 shared memory,
  `cmma(16×16×16)` f16→f32, store the tile back, apply scale+mask (QK) / online-rescale-combine (PV).
  Lowers to WMMA (CUDA/ROCm), `OpCooperativeMatrixMulAddKHR` (SPIR-V), simdgroup matrix (Metal).
- `default` arm: the identical contraction as portable scalar MACs. This is the arm the CPU runtime
  runs (cubecl-cpu rejects every CoopMma op), so it is the bit-exact oracle for the whole structure —
  tiling, online rescale, shared-memory epilogue, f32 accumulation order. Exactly the `mmq_q8_wmma`
  pattern. The cmma arm is equivalent by construction (same algebra; f16 vs f32 intermediate precision)
  and is gated on-GPU in a scale-relative tolerance against the same materialized reference.

GQA-native (reads the shared KV head, no `repeat_kv`), batch-aware, causal-optional, and `seq_q`/`seq_k`
ride a runtime `meta` SSBO (like `sdpa_blk`) so ONE compiled `.spv` serves any (growing) sequence.
Tiles are `Br=Bc=16` to match one cooperative-matrix fragment; `d` is comptime (one `.spv` per head dim).

## CPU-gate results (scalar/default arm vs materialized two-pass `attn::sdpa_ref`)

`cargo test -p hanzo-kernel --no-default-features --features cpu --lib flash::` — **2 tests, both pass;
full crate suite 49/49.** Gate = scale-relative `max|Δ|/max|ref|` < 2e-3 (online softmax is not bit-exact
to a two-pass reference; per-element relative error blows up on near-zero softmax cancellation — use
scale-relative per PHILOSOPHY). Driven one cube at a time (`cube_base` offset, disjoint outputs summed)
to sidestep cubecl-cpu's cross-cube SharedMemory aliasing while staying faithful to the full-grid launch.

| shape | causal | scale_rel |
|---|---|---|
| decode kv1  (nh4/nkv2 d32)              | no  | 0.00e0 (bit-exact) |
| decode kv17 tail (nh4/nkv2 d32)         | no  | 1.50e-7 |
| decode kv128 (nh4/nkv2 d32)             | no  | 3.60e-7 |
| decode kv512 GQA4 (nh8/nkv2 d64)        | no  | 1.31e-6 |
| prefill qt1 GQA2 (nh4/nkv2 sq16 sk128 d32) | yes | 1.20e-7 |
| prefill qt1 GQA2 (nh4/nkv2 sq16 sk128 d32) | no  | 5.39e-7 |
| prefill qt1 MHA d64 (nh4/nkv4 sq16 sk64)   | yes | 1.20e-7 |
| prefill 3-tile aligned (nh4/nkv2 sq48 sk48 d32) | yes | 1.24e-7 |
| prefill tail (nh6/nkv3 sq40 sk40 d64)   | yes | 1.49e-7 |
| prefill 512 MHA d128 (nh2/nkv1 sq512 sk512) | yes | 1.95e-7 |
| production `flash_attn_run` on cpu (Target::of→Cpu) | yes | 1.23e-7 |

Covers prefill (single + multi query-tile, seq_q up to 512, aligned + tail), decode (seq_q=1, growing
kv, tails), GQA ratios 2/3/4 and MHA, causal on/off, d ∈ {32,64,128}. All ≈1e-7 (f32 online-softmax
floor), far under gate.

## f16 cmma accel arm — COMPILES + lowers to SPIR-V coopmat (host-side, GPU-free)

- The cmma island **compiles** in the CPU build (`F=f32` instantiation) — both islands, both arms.
- `.spv` dumped GPU-free: `matvec-check dump-flash` forced onto **llvmpipe** (software Vulkan;
  `VK_ICD_FILENAMES=…/lvp_icd.json`, so the sibling-owned RADV GPU is invisible — zero contention).
  cubecl emits the SPIR-V at codegen, before the driver validates coopmat, so the artifact lands even
  though a software dispatch can't run coopmat.
- Processed through `hanzo-ml/tools/dsl/spv_to_ml.sh` (entry `flash_attn_f_f32`→`main`, dead info-var
  stripped) → **`hanzo-ml/src/vulkan/spv/flash_attn_dsl_d128.spv`** (11472 B, d=128, one `.spv` any seq).
  spirv-dis confirms `OpCapability CooperativeMatrixKHR`, f16 A/B (use 0/1) + f32 accumulator (use 2),
  **two `OpCooperativeMatrixMulAddKHR`** (QK + PV). The spirv-val uniform-layout warning on the dead
  cubecl info-UBO is the known-spurious one every DSL `.spv` carries (driver tolerates).
- **STAGED + INERT**: the `.spv` is NOT `include_bytes!`d / registered (0 refs in `vulkan_backend.rs`).
  Name is distinct from the existing hand-written `flash_attn.spv` (the C++ `flash_attn.comp`).

## Per-backend wiring seam (the follow-up — replaces the sdpa+softmax op-chain)

The DSL `flash_attn` slots in wherever the model's attention currently does `bmm(Q,Kᵀ) → softmax →
bmm(P,V)`. Shared op seam = `hanzo_nn::ops::Sdpa` (the same op `sdpa_blk` wired through). `flash_attn`
is the prefill-capable, tensor-core sibling of `sdpa_blk` (which stays the decode-optimal path).

- **Vulkan**: add `VulkanDevice::flash_attn_dsl(q,k,v,meta,d,plane)` dispatching
  `flash_attn_dsl_d128.spv` (register in `vulkan_backend.rs` `kernel_spv` via `include_bytes!`, MAX_BINDINGS
  covers 6: q,k,v,out,scale,meta). Call from `hanzo_nn::ops::Sdpa::vulkan_fwd` ← engine
  `attention::vulkan_decode_attn` — for the **prefill** shape (seq_q>1, d=128 f32). Decode (seq_q=1) keeps
  `sdpa_blk` (already at DRAM roofline). A/B behind an env kill-switch, same as `HANZO_VK_FUSED_ATTN`.
- **CUDA**: `flash_attn` replaces the materialized `softmax_f32` + cutlass QK/PV in the prefill/regular
  branch. Route the `is_first_prompt_chunk=false / mask=None / seq_len>1` path (paged_attention
  `forward_impl` regular branch) — the SAME gate that neutered the flash-feature lever (#25). Gate numeric
  vs the naive path before trusting output; WMMA on sm_121 is the verified cmma target (cmma_probe).
- **ROCm**: same DSL source lowers to HIP WMMA via `hanzo-cubecl-hip` (the `--features rocm` path); slot
  into the rocBLAS/paged-attn attention call.
- **Metal**: Metal already ships a tiled simdgroup flash (ggml heritage, `kernel_flash_attn_*`). The DSL
  `flash_attn` is the portable twin — retire the hand kernel ONLY when the DSL twin is bit-exact AND ≥ as
  fast, measured. Until then it is the fallback for shapes the hand kernel doesn't cover.

## GPU-validation plan (the follow-up, on a freed GPU)

1. **Numeric gate on GPU** (evo Vulkan / spark CUDA — whichever frees first): dispatch `flash_attn` via
   the runtime, gate scale-relative vs `attn::sdpa_ref` at the live shapes (decode d128 seq_k∈{128,2048},
   prefill 512×512 causal). f16 tolerance target ≈ 1e-2–1e-3 (f16 QK/PV; cf. the P*V split-precision
   note on `feat/flash-f32pv-cure` if precision is short). This is the gate the cmma arm could not get on
   CPU (cubecl-cpu rejects CoopMma). Watch: the cmma path assumes tile-aligned loads read valid memory —
   partial tiles are 0-staged for Q/K/V but the direct-slice loads span the whole staged tile; confirm no
   OOB on ragged seq at the real strides.
2. **A/B vs the incumbent** (`sdpa_blk` decode; cutlass+softmax_f32 CUDA prefill) — per-op GPU time
   (`VK_PROFILE_GPU` / nsys). Flip live ONLY when bit-exact-ish AND ≥ incumbent AND ≥ llama, per the
   migration protocol. Correctness first; speed is this step, not the last.
3. **Re-dump per head dim** actually used (d=128 committed; add d=64/d=80 if a target model needs them —
   `d` is comptime, one `.spv` each; `matvec-check dump-flash` after editing the shape).

## Ship state
- `hanzo-kernel` 0.2.27 → **0.2.28** (bumped; NOT published — GPU-validation-pending). NB: the parked
  `origin/autotune` branch also staged 0.2.28 unpublished; reconcile the patch at merge (neither is on
  crates.io, so no collision).
- Files: `hanzo-kernel/src/flash.rs` (new), `hanzo-kernel/src/lib.rs` (`pub mod flash;`),
  `hanzo-kernel/src/main.rs` (`dump-flash` seam), `hanzo-ml/src/vulkan/spv/flash_attn_dsl_d128.spv` (new,
  inert). No live dispatch path touched — `sdpa`/`sdpa_blk` and every engine call are unchanged.
- Mark the branch **authored-not-benched**.
