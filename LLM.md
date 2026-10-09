# LLM.md — hanzoai/ml

Fast, multi-backend tensor & ML framework for **Rust** (CPU · CUDA · Metal · ROCm · Vulkan)
with quantization (GGUF/GGML/AFQ/GPTQ/AWQ). The compute core beneath Hanzo inference.

## Canonical role
- This repo is the **canonical implementation** of the Hanzo Rust ML/tensor core.
  One impl, one place — discovery/wrapper repos link here, never copy the code.
- Rust is the **2nd-most-complete** ecosystem (Python → Rust → C++ → Go → …).
- Crate `hanzo-ml` (crates.io) · docs at docs.rs/hanzo-ml.

## Install / run
- Core: `cargo add hanzo-ml` (`Tensor`/`Device`); add `hanzo-nn` to build models.
- GPU: `--features cuda` (+ `cudnn`), or `metal` / `rocm` / `vulkan`.
- Examples: `cargo run --example quantized --release` (see `hanzo-ml-examples/`).

## Key entry points
- `hanzo-ml/` — core ops, devices, `Tensor`.
- `hanzo-nn/` — layers & model building.
- `hanzo-transformers/` — model implementations.
- `hanzo-kernels/`, `hanzo-flash-attn/` — CUDA kernels & FlashAttention v2.
- `hanzo-onnx/`, `hanzo-datasets/`, `hanzo-ml-wasm-examples/`.
- `hanzo-train/` — training on hanzo-ml. `cluster`: one model across heterogeneous machines
  (Metal, CUDA, ROCm, CPU) by local SGD with an outer Nesterov step (DiLoCo). A coordinator owns
  the plan, θ_global in F32, the outer step and checkpoints; workers join any time, take batches
  one at a time, send θ_local − θ_global in bf16 with error feedback; deltas sum in join order, so
  a run equals its rounds replayed in one process (`cluster::replay`) bit for bit, and a
  checkpoint (outer momentum and carry, merged batches, round, every worker's AdamW state)
  resumes bit-identically. A worker that drops has its batches requeued; a round waits for no
  one past its deadline plus `grace` (a silent member is dropped, its link shut), a join under a
  member's name replaces it, links have keepalive and bounded reads/writes (`net::link`), and a
  worker whose link fails joins again on its own with backoff. A model plugs in behind
  `cluster::Model` (`vars` F32 masters, `rate` per parameter — `None` frozen, `step` over a batch
  of row indices the caller planned); the coordinator's owner behind `cluster::Keep` (round line,
  validation, save). `adam`: AdamW with per-parameter rates and open moments (hanzo-nn's
  `optim::adamw` update, fused on Metal). `gpu`: free-memory floor. `dspark`: the DSpark draft on
  the runtime (`hanzo-train fit` / `join`). Tests: `cargo test -p hanzo-train` (tiny classifier
  and synthetic DSpark cache, all CPU). Kai (hanzoai/decision `train`) is one user.

## Backends and models
- **Vulkan dtypes.** Storage is F32 or U32 (BF16 is held as F32, I64 as U32). U32 reductions
  run their own kernel (`reduce_u32.comp`; argmax/argmin take the first index on ties, as the
  CPU does) and U32 binary ops take the CPU fallback; any other storage type in a reduction is
  an error, never an F32 answer. `cmp`, `affine`, `powf` and unary kernels are F32 only.
  `hanzo-ml/tests/vulkan_dtype_tests.rs` holds each against the CPU.
- **Masked softmax on Vulkan.** A row's running max starts at `-inf` (`softmax_rows_blk.spv`),
  so a row masked throughout is uniform, not NaN (`softmax_masked_row_is_uniform`). The same
  `-3.4e38` start is still in the DSL (`hanzo-kernel` `dag.rs` `Red::Max`, `attn.rs`,
  `flash.rs`, `quant.rs`) and in `sdpa_blk.spv`, `flash_attn_dsl_d128.spv` and
  `moe_route.spv`: fix it in the DSL and regenerate, with a fully masked row in each test.
- **ModernBert past its trained positions.** `Config::yarn(s)` gives a longer sequence YaRN
  rotary tables (NTK-by-parts, beta 32 and 1, attention factor 0.1 ln s + 1) on the global
  layers and extended frequencies on the local ones; within `max_position_embeddings` every
  pass reads the trained tables bit for bit.
- **ModernBert's trained rotary tables take their angles from the host.** Each angle `p · f` is
  one F32 product on the host, its sine and cosine taken on the device in F32, then cast: the
  bits an exact F32 GEMM `t · inv_freq` gave on the CPU, ROCm, Metal and CUDA
  (`f32_tables_are_the_exact_gemm_bits`). The device GEMM it replaces ran on Vulkan's fp16
  matrix cores, rounding positions past 2,048 (rows of 4,096 and 8,192 tokens read 25% and 37%
  from the CPU's states), and in BF16 rounded positions past 256 to multiples of 8 and more.
  `trained_tables_hold_every_position_on_every_device` holds every device's tables, F32 and
  BF16, to the exact angles (`--features rocm|vulkan|metal|cuda`).
- **ModernBert's long padded pass, a tile at a time.** A padded pass whose row's scores pass
  `LOGITS` (2^26, 256 MiB in F32: a row past 2,364 tokens over 12 heads) attends a tile of
  queries at a time (`Tiles`): a global tile reads every key, at most 2^26 scores over its rows;
  a local tile holds 512 queries and the keys within their window. The keys' mask is
  `[B, 1, 1, L]` and each local tile's window `[t, t + 2w]` (tiles alike share one, built on the
  pass's device), so no `[L, L]` tensor exists; a shorter pass is the whole pass, bit for bit. Kᵀ and V are laid out
  once a layer (Vulkan's matmul copies a strided operand per call), and the raw scores drop once
  masked: two blocks live at once. (Dropping the masked scores too once the softmax is queued
  grew Metal's pool past 13 GiB at 16,384 tokens.) Each stage is a TRACE span (`layer` with
  `at`, `qkv`, `rope`, `tile` over `scores`, `mask`, `softmax`, `values`, then `join`, `proj`,
  `mlp`), which a subscriber can meter. `tests/modernbert_memory.rs` counts a 20,480-token
  pass's heap (no block of L x L bytes, less than one `[L, L]` F32 tensor held) and holds it to
  the packed pass within 5e-5 of the scale.
- **ModernBert's residual stream.** `ModernBert::stream(xs, mask, at)` stops the pass
  `forward` runs at the input of layer `at` (`Stream`: packed or padded, as that pass lays
  it out), and `ModernBert::finish` runs it on: `finish(stream(.., at))` is `forward` bit
  for bit at every `at`, packed and padded. `Stack::new(vb, config, at)` loads layers `at..`
  and the final norm apart (a trainable copy of a top, for hanzo-inc/kai's routed yes/no
  reader), reading another pass's stream; `Stream::rows` and `Stream::packed` take a packed
  stream apart by sequence and put it back, the same bits.

## Next: synthesis as a training loop
Capability training today is dataset → batches → optimizer → checkpoint, with curricula
generated and scored by scripts outside it (hanzo-inc/kai's jailbreak round, 2026-10-03).
Its pieces belong here, in this order:
1. `hanzo-train` keeps per-example signals (loss, margin, prediction, gradient norm) instead
   of folding them into the batch loss, and reports balanced accuracy and MCC per slice.
2. A dataset rows can join and leave mid-run, each row carrying its provenance and its
   factors (what a generated row varies), with weights a sampler reads.
3. Contrast pairs as a training item: two rows and a margin on their score gap.
4. Micro-fits with successive halving (`hanzo-train sweep`): many curricula at a tenth of a
   run, the best continued.
5. Gradient embeddings over a capability's parameters only, to select the generated rows
   whose updates align with real errors.
Generation itself (propose, mutate, verify against more than its own model) is a separate
crate on top of these; a generator that only checks itself learns nothing new.

## Releasing
- Registry is **crates.io**, owner `zeekay`. There is no Hanzo cargo registry: no
  `[registries]` in `.cargo/config.toml`, no `publish = [...]` allow-list, and the
  sibling Rust repo (`hanzoai/engine`) publishes the same way.
- **Each crate names everything itself: no workspace inheritance.** A crate's manifest holds
  its own version, package fields, lints and every dependency spec (a sibling as `path` plus
  the exact `version` it was published at); the root holds only `members`, `exclude`,
  `resolver` and profiles. What a crate ships is what its own manifest says.
- **Each crate carries its own version and moves by a patch bump from the version it
  last released.** Crates change at different rates, so their numbers differ — that
  is information, not drift. Never renumber a crate to match another.
- `scripts/publish-order` is the release set: every crate that does not say
  `publish = false`, topologically sorted. `publish = false` is the one way to keep a
  crate off the registry (examples, demos, the book, the PyPI extension module, and
  `tensor-tools`, whose crates.io name belongs to the upstream candle author).
- Publishing is CI's job: `.github/workflows/publish.yml`, on a `v*` tag or a manual
  dispatch (`gh workflow run publish.yml -R hanzoai/ml --ref main`), walks that order with
  `cargo publish --no-verify` (GPU build scripts can't run on crates.io builders). It
  publishes a crate only when its manifest version is ahead of crates.io, so re-running is
  safe. The token is `CARGO_REGISTRY_TOKEN` at `/@prod` in org hanzo's KMS, read as the
  `hanzo-ci` identity (universe `kms-grants.yaml`).
- **A crate whose source changes takes a patch bump in the same commit.** The ratchet only
  sees versions: a crate edited at an already-published version never ships, and its
  consumers pinning crates.io read older code than main (six crates had drifted, among them
  `hanzo-rocm-kernels`, before 027c26d9e). Consumers pin published versions, never a git rev
  (hanzo-inc/kai since 9b382b4).
- Run a crate's own tests before its version moves. ROCm needs
  `LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib` — `libhiprtc.so.7` lives there, not in
  `/opt/rocm/lib`, so `cargo test -p hanzo-kernel --features rocm` otherwise dies at
  load time with the test binary already built.

## Brand rules (enforce in all docs)
- Hanzo is the **Open AI Cloud / full AI SDK** — never an "LLM gateway", never
  positioned vs LiteLLM, never an "OpenAI-compatible proxy". Purge that framing.
- Paths are **`/v1/`**, never `/api/`.
- **Zen** models are our own family — don't present upstream model names as ours.

Spec: `~/work/hanzo/SDK-ARCHITECTURE.md` — the canonical one-way SDK model.
