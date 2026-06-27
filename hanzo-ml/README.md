# Hanzo ML

A fast, multi-backend tensor and machine-learning framework for Rust — the compute substrate behind [Hanzo](https://hanzo.ai) inference.

[![crates.io](https://img.shields.io/crates/v/hanzo-ml.svg?style=flat-square&color=black)](https://crates.io/crates/hanzo-ml)
[![docs.rs](https://img.shields.io/docsrs/hanzo-ml?style=flat-square&color=black)](https://docs.rs/hanzo-ml)
[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-black?style=flat-square)](https://github.com/hanzoai/ml)

## Backends

One tensor API across CPU (SIMD / MKL / Accelerate), CUDA — including unified/managed memory for NVIDIA **GB10 / DGX Spark** — Metal (Apple Silicon), Vulkan, and ROCm (AMD RDNA3.5 APUs).

## Features

- **Quantization** — the full GGUF/GGML zoo (Q/K, legacy, IQ1–4, TQ) plus GPTQ / AWQ / AFQ / MXFP4, much of it bit-exact and GPU-resident.
- **Multimodal** — text, vision, audio, and 3D.
- **WebAssembly** — run models in the browser.
- **Rust-native** — memory-safe, zero-cost abstractions; pairs with [Hanzo Engine](https://github.com/hanzoai/engine) for serving.

## Quick start

```rust
use hanzo_ml::{Device, Tensor};

let device = Device::Cpu; // or Device::new_cuda(0)?, Device::new_metal(0)?
let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
let b = Tensor::randn(0f32, 1., (3, 4), &device)?;
println!("{}", a.matmul(&b)?); // Tensor[[2, 4], f32]
```

See the [repository](https://github.com/hanzoai/ml) for examples, the guide book, and the model zoo.

---

Derived from `candle` and rebuilt for the Hanzo stack. Dual-licensed MIT / Apache-2.0.
Models and weights: [huggingface.co/hanzoai](https://huggingface.co/hanzoai) · [huggingface.co/zenlm](https://huggingface.co/zenlm).
