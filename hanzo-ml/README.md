# hanzo-ml

Tensors, reverse-mode autograd and GGUF quantization for Rust. One API on the CPU, CUDA, ROCm,
Metal, Vulkan and wgpu.

```toml
[dependencies]
hanzo-ml = "0.11"
```

```rust
use hanzo_ml::{DType, Device, Tensor};

fn main() -> hanzo_ml::Result<()> {
    let device = Device::Cpu; // Device::new_cuda(0)?, new_rocm(0)?, new_metal(0)?, new_vulkan(0)?
    let x = Tensor::randn(0f32, 1., (1, 64), &device)?;
    let w = Tensor::randn(0f32, 1., (64, 64), &device)?;
    let y = x.matmul(&w)?.gelu()?.to_dtype(DType::BF16)?;
    println!("{y}");
    Ok(())
}
```

## Features

| Feature | Adds |
|---|---|
| `cuda` | CUDA backend; `cudnn`, `nccl` and `cutile` extend it |
| `rocm` | ROCm backend; `rocm-miopen` adds MIOpen convolutions |
| `metal` | Metal backend |
| `vulkan` | Vulkan backend (SPIR-V) |
| `wgpu` | wgpu backend (WGSL) |
| `mkl`, `accelerate` | BLAS for the CPU backend |

## Contents

- `Tensor`, `Var` and `backprop`: tensors and gradients.
- `quantized`: GGUF and GGML readers and writers, `QTensor`, `QMatMul`, and 28 block types from
  Q4_0 to MXFP4, NVFP4 and ROCmFP4, each with a CPU reference decode.
- `safetensors`, `npy`, `pickle`: weight formats.
- `CustomOp1`/`2`/`3`: plug your own kernel into the graph.

The [book](https://github.com/hanzoai/ml/tree/main/hanzo-ml-book) covers backends, the kernel DSL,
quantization and training.

## License

See [LICENSE](LICENSE), and [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
