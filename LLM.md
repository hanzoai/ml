# LLM.md — hanzoai/ml

Fast, multi-backend tensor & ML framework for **Rust** (CPU · CUDA · Metal · ROCm · Vulkan)
with quantization (GGUF/GGML/AFQ/GPTQ/AWQ). The compute core beneath Hanzo inference.

## Canonical role
- This repo is the **canonical implementation** of the Hanzo Rust ML/tensor core.
  One impl, one place — discovery/wrapper repos link here, never copy the code.
- Rust is the **2nd-most-complete** ecosystem (Python → Rust → C++ → Go → …).
- Crate `hanzo-ml` (crates.io) · docs at docs.rs/hanzo-ml.

## Install / run
- Core: `cargo add hanzo-ml-core` (`Tensor`/`Device`); add `hanzo-nn` to build models.
- GPU: `--features cuda` (+ `cudnn`), or `metal` / `rocm` / `vulkan`.
- Examples: `cargo run --example quantized --release` (see `hanzo-ml-examples/`).

## Key entry points
- `hanzo-ml/` — core ops, devices, `Tensor`.
- `hanzo-nn/` — layers & model building.
- `hanzo-transformers/` — model implementations.
- `hanzo-kernels/`, `hanzo-flash-attn/` — CUDA kernels & FlashAttention v2.
- `hanzo-onnx/`, `hanzo-datasets/`, `hanzo-ml-wasm-examples/`.

## Brand rules (enforce in all docs)
- Hanzo is the **Open AI Cloud / full AI SDK** — never an "LLM gateway", never
  positioned vs LiteLLM, never an "OpenAI-compatible proxy". Purge that framing.
- Paths are **`/v1/`**, never `/api/`.
- **Zen** models are our own family — don't present upstream model names as ours.

Spec: `~/work/hanzo/SDK-ARCHITECTURE.md` — the canonical one-way SDK model.
