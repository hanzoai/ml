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

## Releasing
- Registry is **crates.io**, owner `zeekay`. There is no Hanzo cargo registry: no
  `[registries]` in `.cargo/config.toml`, no `publish = [...]` allow-list, and the
  sibling Rust repo (`hanzoai/engine`) publishes the same way.
- **Each crate carries its own version and moves by a patch bump from the version it
  last released.** Crates change at different rates, so their numbers differ — that
  is information, not drift. Never renumber a crate to match another.
- `scripts/publish-order` is the release set: every crate that does not say
  `publish = false`, topologically sorted. `publish = false` is the one way to keep a
  crate off the registry (examples, demos, the book, the PyPI extension module, and
  `tensor-tools`, whose crates.io name belongs to the upstream candle author).
- Publishing is CI's job: push a `N.N.N` tag to the forge and `.hanzo/workflows/publish.yml`
  walks that order with `cargo publish --no-verify` (GPU build scripts can't run on
  crates.io builders). The tag names the release event; the manifests name the artifacts.
  Re-running is safe — a crate already at its manifest version is skipped.
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
