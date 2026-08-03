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

## Licensing — read before touching a manifest

Fork of **candle** (huggingface/candle), **MIT OR Apache-2.0**. `NOTICE` is the
full record. The layout is flat (`hanzo-ml/`, `hanzo-nn/`, …) because candle's
is flat — there is no `crates/` directory and its absence is not a bug.

- **Never edit `LICENSE-MIT` or `LICENSE-APACHE`.** They are upstream's texts and
  they are what actually grant rights.
- **Every crate declares `MIT OR Apache-2.0`.** One exception, deliberate:
  `hanzo-bindgen-cuda` declares `MIT`, because it is a fork of `bindgen_cuda`
  by Nicolas Patry (github.com/Narsil/bindgen_cuda) which is MIT upstream, and
  its manifest retains his authorship. Do not "normalise" it — that would
  relicense a third party's work.

### The bug that shipped 191 times

From 2026-01-16 to 2026-08-03 the manifests declared a licence candle never
granted. `216fae85` flipped 24 manifests to `BSD-3-Clause OR Apache-2.0` and
added `LICENSE-BSD`; `16fabaed` later deleted `LICENSE-BSD` as "stray" without
touching a single manifest. For seven months the repo *offered* BSD-3-Clause
while shipping no BSD text, and `hanzo-kernel` and `hanzo-3d` offered
`BSD-3-Clause` with no Apache fallback at all.

All 27 manifests were corrected to `MIT OR Apache-2.0` on 2026-08-03. The **191
already-published versions across 12 crates** are **not** being yanked:
crates.io metadata is immutable per version, the LICENSE files inside each
package are what govern, and yanking would break every consumer to fix a label.
(25 of those — all of `hanzo-kernel`, `hanzo-kernel-macros`, `hanzo-3d` — went
out as `BSD-3-Clause` with no Apache fallback at all.)

Verify the count yourself rather than trusting a number in a doc; crates.io
rejects anonymous requests, so send a User-Agent:

    curl -s -A "you@hanzo.ai" https://crates.io/api/v1/crates/hanzo-ml \
      | jq '[.versions[].license] | group_by(.) | map({(.[0]): length}) | add'

The lesson worth keeping: `license =` in a manifest is a *claim*, and the
LICENSE files are the *fact*. When they disagree, the files win — and a
find-and-replace across manifests is a licensing change, not a chore.

### Open item — the root `LICENSE` is ours, not candle's

candle ships no `/LICENSE` (only `LICENSE-MIT` + `LICENSE-APACHE`). Ours was
added by `225e590f` and is a **modified** Apache-2.0: the APPENDIX is replaced
with a `Copyright 2024 Hanzo AI Inc` line and several sentences are reworded.
Because it is the only file named `LICENSE`, GitHub reads it and reports this
repo as Apache-2.0 rather than dual-licensed. Left untouched pending a
decision — altering a file named LICENSE is a licensing call, not a metadata
fix. Flagged, not resolved.
