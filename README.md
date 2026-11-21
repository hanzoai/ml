<p align="center"><img src=".github/hero.svg" alt="Hanzo ML" width="880"></p>

# Hanzo ML

Tensors, autograd, quantized inference and training in Rust. One `Tensor` API over CPU, CUDA,
ROCm, Metal, Vulkan and wgpu. It is the compute layer under [Hanzo Engine](https://github.com/hanzoai/engine).

```rust
use hanzo_ml::{Device, Tensor};

fn main() -> hanzo_ml::Result<()> {
    let device = Device::Cpu; // or Device::new_cuda(0)?, new_rocm(0)?, new_metal(0)?, new_vulkan(0)?
    let a = Tensor::randn(0f32, 1., (2, 3), &device)?;
    let b = Tensor::randn(0f32, 1., (3, 4), &device)?;
    println!("{}", a.matmul(&b)?); // a [2, 4] f32 tensor
    Ok(())
}
```

```bash
cargo add hanzo-ml                      # CPU
cargo add hanzo-ml --features cuda      # or rocm, metal, vulkan, wgpu, mkl, accelerate
```

## Backends

| Backend | Feature | Device | Notes |
|---|---|---|---|
| CPU | default (`mkl`, `accelerate` optional) | `Device::Cpu` | quantized dot products in AVX2/AVX-512, NEON and wasm simd128 |
| CUDA | `cuda` (`cudnn`, `nccl`, `cutile`) | `Device::new_cuda(i)` | on unified-memory parts such as GB10, allocations of `ML_CUDA_UMA_THRESHOLD` (8 GiB default) or more are managed memory |
| ROCm | `rocm` (`rocm-miopen` for convolution) | `Device::new_rocm(i)` | HIP kernels compiled by `hipcc` for the GPU's arch on first use and cached; developed and measured on gfx1151 (Strix Halo) |
| Metal | `metal` | `Device::new_metal(i)` | Apple Silicon |
| Vulkan | `vulkan` | `Device::new_vulkan(i)` | SPIR-V kernels through `ash`; uses cooperative matrix and integer dot product when the driver has them |
| wgpu | `wgpu` | `Device::new_wgpu(i)` | WGSL kernels for matmul, Q8_0/Q4_0/Q4_K matvecs, flash attention, norms and elementwise ops; convolution, pooling and upsampling return an error |
| WASM | build for `wasm32` | `Device::Cpu` | see [`hanzo-ml-wasm-examples`](hanzo-ml-wasm-examples) |

The book's [Backends](hanzo-ml-book/src/backends.md) chapter has the details per backend.

## What is ours

- **Kernel DSL** ([`hanzo-kernel`](hanzo-kernel)). A kernel is one Rust function marked
  `#[kernel(targets(...))]`. The same source runs on the CPU and lowers to CUDA, ROCm (HIP),
  Metal (MSL) and Vulkan (SPIR-V). Each kernel has a host reference beside it, and tests compare
  every backend against it. `hanzo_kernel::tune` times a kernel's schedules once per device and
  shape and caches the winner on disk. The Vulkan backend dispatches DSL kernels for RMSNorm,
  softmax, fused add-RMSNorm, decode attention, the MoE router, the MoE and large dense Q4_K/Q6_K
  matvecs, the Q4_K prefill GEMM and the single-row GEMV. On ROCm the DSL norms match the
  hand-written kernels but reach only 84–92% of their speed, so the hand-written ones stay.
- **ROCm backend.** Quantized weights stay on the device in GGML layout. One HIP core
  (`hanzo-rocm-kernels/src/kernels/quant.hip`) decodes 25 GGML types on the device and runs
  prefill as an int8 WMMA GEMM for Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q4_K, Q5_K, Q6_K, IQ4_XS,
  TQ2_0 and both ROCmFP4 variants. Autograd runs on it too.
- **Quantization.** GGUF and GGML files with 28 block types beside F32/F16/BF16/I32: the legacy
  and K quants, the IQ1–IQ4 codebooks, TQ1_0/TQ2_0, MXFP4, NVFP4, ROCmFP4 (dual scale and fast)
  and Q1_0. Every type decodes on the CPU; each GPU backend has native kernels for a subset.
  NVFP4 has no GPU kernel yet: CUDA dequantizes it to F32 for a dense matmul, Metal refuses it.
  F8E4M3 is a tensor dtype with CUDA kernels.
- **Training.** Reverse-mode autograd. The fused softmax, layer-norm, rms-norm, rope, silu-mul,
  GeGLU and select ops carry their own gradients, checked against the unfused composites on CPU,
  CUDA and Metal (`hanzo-nn/tests/grad.rs`). Optimizers: SGD, AdamW (one fused kernel on CUDA and
  Metal), Muon, MuonClip and Muon with AdamW. [`hanzo-train`](hanzo-train) trains one model across
  mixed machines by local SGD with an outer Nesterov step (DiLoCo); a resumed run matches the
  uninterrupted one bit for bit.
- **Classical learning.** [`hanzo-learn`](hanzo-learn): least squares, logistic regression,
  gradient-boosted trees, isolation forest and local outlier factor. Fitting and predicting are
  two traits, and each fitted model is named by a content hash.

## Measured

Each number is from the record cited; none is a projection.

| Result | Where | Source |
|---|---|---|
| DSL `add_rmsnorm` 23.4 ms vs 166.5 ms for the GLSL kernel it replaced (7.1x), max error 4.5e-6 | Vulkan, gfx1151, 4096x4096 | commit `38181d48b`, 2026-07-08 |
| DSL `rms_norm` 10.6x the GLSL kernel it replaced | Vulkan | commit `487ff722d`, 2026-07-02 |
| DSL Q4_K decode matvec vs the hand-written one: 1.022x at 6144 rows, 0.947x at 2048, 0.871x at 1024, so only shapes of 4096 rows or more take the DSL kernel | Vulkan, gfx1151 | commit `3e2f8b080`, 2026-07-22 |
| DSL `rms_norm` and `add_rmsnorm` at 84–92% of the hand-written HIP kernels; the hand-written ones stay | ROCm, gfx1151 | commit `b8e3f3a87`, 2026-07-08 |
| Q6_K prefill GEMM 1.70–1.74x after SWAR decode; int8 WMMA peak 57.1 TOP/s vs f16 56.8 TF/s | ROCm, gfx1151, ROCm 7.13 | [`hanzo-rocm-kernels/bench/README.md`](hanzo-rocm-kernels/bench/README.md), 2026-06-24 |

## Crates

| Crate | What | Published |
|---|---|---|
| [hanzo-ml](hanzo-ml) | `Tensor`, devices, autograd, quantization | [crates.io](https://crates.io/crates/hanzo-ml) |
| [hanzo-nn](hanzo-nn) | layers, fused ops, optimizers, `VarBuilder` | [crates.io](https://crates.io/crates/hanzo-nn) |
| [hanzo-transformers](hanzo-transformers) | model implementations and quantized variants | [crates.io](https://crates.io/crates/hanzo-transformers) |
| [hanzo-kernel](hanzo-kernel), [hanzo-kernel-macros](hanzo-kernel-macros) | the kernel DSL and its `#[kernel]` attribute | [crates.io](https://crates.io/crates/hanzo-kernel) |
| [hanzo-kernels](hanzo-kernels) | CUDA kernels | [crates.io](https://crates.io/crates/hanzo-kernels) |
| [hanzo-rocm-kernels](hanzo-rocm-kernels) | HIP kernels and their compile cache | [crates.io](https://crates.io/crates/hanzo-rocm-kernels) |
| [hanzo-metal-kernels](hanzo-metal-kernels) | Metal kernels | [crates.io](https://crates.io/crates/hanzo-metal-kernels) |
| [hanzo-flash-attn](hanzo-flash-attn), [hanzo-flash-attn-v3](hanzo-flash-attn-v3) | FlashAttention-2 (CUDA) and FlashAttention-3 (Hopper) | [crates.io](https://crates.io/crates/hanzo-flash-attn) |
| [hanzo-onnx](hanzo-onnx) | ONNX evaluation, including the `ai.onnx.ml` domain | [crates.io](https://crates.io/crates/hanzo-onnx) |
| [hanzo-datasets](hanzo-datasets) | MNIST, Fashion-MNIST, CIFAR-10 and TinyStories loaders and a batcher | [crates.io](https://crates.io/crates/hanzo-datasets) |
| [hanzo-train](hanzo-train) | distributed training (DiLoCo) and the DSpark draft trainer | git only |
| [hanzo-training](hanzo-training) | GRPO over hanzo-ml autograd | [crates.io](https://crates.io/crates/hanzo-training) |
| [hanzo-learn](hanzo-learn) | classical statistical learning | git only |
| [hanzo-research](hanzo-research) | client for `api.hanzo.ai/v1/research`: experiments, results and their provenance | [crates.io](https://crates.io/crates/hanzo-research) |
| [hanzo-3d](hanzo-3d) | meshes, Gaussian splats, voxels, OBJ/PLY I/O | [crates.io](https://crates.io/crates/hanzo-3d) |
| [hanzo-ml-pyo3](hanzo-ml-pyo3) | Python bindings, built with maturin | not published |
| [hanzo-ml-examples](hanzo-ml-examples) | runnable examples, one directory each | not published |

A git-only crate is used as `hanzo-train = { git = "https://github.com/hanzoai/ml" }`.

## Examples

```bash
make example NAME=quantized                  # CPU
make example NAME=quantized FEATURES=cuda    # or metal, mkl, accelerate
```

Each example directory in [`hanzo-ml-examples/examples`](hanzo-ml-examples/examples) has a README.
Browser builds are in [`hanzo-ml-wasm-examples`](hanzo-ml-wasm-examples).

## Develop

`make help` lists the targets. `make check` is the CI gate, `make test` runs the CPU tests,
`make test-kernel` tests the DSL, and `make book` builds the [book](hanzo-ml-book).

## Troubleshooting

**Undefined `sgemm_` or `dgemm_` with `mkl` or `accelerate`.** Link the BLAS crate from your
binary: `extern crate intel_mkl_src;` or `extern crate accelerate_src;`.

**`cute/algorithm/copy.hpp` not found building flash attention.** CUTLASS is a submodule:
`git submodule update --init`.

**`parameter packs not expanded` from gcc 11 under nvcc.** Point nvcc at another host compiler,
e.g. `NVCC_CCBIN=/usr/lib/gcc/x86_64-linux-gnu/10 cargo ...`.

**401 downloading a gated model.** Accept the model's terms on its hub page and log in with
`hf auth login`.

**Slow model loads under WSL.** Keep weights off `/mnt/c`.

**Where did the error come from?** Run with `RUST_BACKTRACE=1`. For CUDA, add
`CUDA_LAUNCH_BLOCKING=1` so the failing kernel is the one reported.

## License

MIT OR Apache-2.0, at your option ([LICENSE](LICENSE)). See [NOTICE](NOTICE) for third-party
attributions.
