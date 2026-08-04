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
