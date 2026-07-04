# Hanzo ML

**One tensor API. Every accelerator. The full quant zoo. Rust-native.** The compute substrate behind [Hanzo](https://hanzo.ai) inference.

[![crates.io](https://img.shields.io/crates/v/hanzo-ml.svg?style=flat-square&color=black)](https://crates.io/crates/hanzo-ml)
[![docs.rs](https://img.shields.io/docsrs/hanzo-ml?style=flat-square&color=black)](https://docs.rs/hanzo-ml)
[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-black?style=flat-square)](https://github.com/hanzoai/ml)

```rust
use hanzo_ml::{Device, Tensor};

let device = Device::new_cuda(0)?; // or Cpu, new_metal(0)?, Vulkan, ROCm — same code
let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
let b = Tensor::randn(0f32, 1., (3, 4), &device)?;
println!("{}", a.matmul(&b)?); // Tensor[[2, 4], f32]
```

## Broad support, measured — not a wishlist

**Six backends, one API.** CPU (SIMD / Intel MKL / Apple Accelerate) · **CUDA** (incl. unified/managed memory for NVIDIA **GB10 / DGX Spark**, sm_121 Blackwell) · **Metal** (Apple Silicon) · **Vulkan** (any GPU) · **ROCm** (AMD RDNA3.5 APUs) · **WebAssembly** (models in the browser).

**At llama.cpp decode parity on every GPU backend** (measured, single-stream, same weights): AMD 0.98×, NVIDIA GB10 0.99–1.04×, Apple M4 Max 0.96× — decode streams the memory roofline everywhere. Prefill trails 0.76–0.91× (the tensor-core GEMM frontier) — honest engineering, not a rounding error.

**The whole GGUF/GGML quant zoo — 22 formats, GPU-resident and bit-exact:** Q4_0/1, Q5_0/1, Q8_0/1, K-quants (Q2_K–Q6_K), the i-quant codebooks (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS), ternary (TQ1_0/TQ2_0) — plus GPTQ · AWQ · AFQ · MXFP4. Every type decodes through one unified `dp4a` / tensor-core core, gated bit-exact against a CPU oracle.

**Multimodal:** text · vision · audio · 3D.

## The stack

| Crate | What |
|---|---|
| **hanzo-ml** | this crate — the multi-backend tensor + ML framework |
| [**hanzo-kernel**](https://crates.io/crates/hanzo-kernel) | the first-party GPU kernel DSL: write a kernel once, lower it to CUDA/ROCm/Vulkan/Metal/CPU |
| [**hanzo-flash-attn**](https://crates.io/crates/hanzo-flash-attn) | flash-attention-2 CUDA kernels |
| [**hanzo-kernels**](https://crates.io/crates/hanzo-kernels) | the hand-tuned CUDA quant kernels |
| [Hanzo Engine](https://github.com/hanzoai/engine) | the serving engine on top: OpenAI + Anthropic + MCP APIs |

## Install

```toml
[dependencies]
hanzo-ml = "0.11"
```

Enable a backend with a feature: `cuda` (+ `cudnn`, `nccl`, `flash-attn`), `metal`, `vulkan`, `rocm`, `mkl`, `accelerate`.

## Quick start

```rust
use hanzo_ml::{Device, Tensor};

let device = Device::Cpu;
let x = Tensor::randn(0f32, 1., (1, 4096), &device)?;
let w = Tensor::randn(0f32, 1., (4096, 4096), &device)?;
let y = x.matmul(&w)?.gelu()?;
```

See the [repository](https://github.com/hanzoai/ml) for examples, the guide book, and the model zoo.

---

Derived from `candle` and rebuilt for the Hanzo stack. Dual-licensed MIT / Apache-2.0.
Models and weights: [huggingface.co/hanzoai](https://huggingface.co/hanzoai) · [huggingface.co/zenlm](https://huggingface.co/zenlm).
