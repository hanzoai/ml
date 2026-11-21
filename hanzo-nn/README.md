# hanzo-nn

Layers, fused ops and optimizers on [hanzo-ml](https://crates.io/crates/hanzo-ml).

```toml
[dependencies]
hanzo-ml = "0.11"
hanzo-nn = "0.11"
```

```rust
use hanzo_ml::{DType, Device, Tensor};
use hanzo_nn::{linear, loss, AdamW, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

fn main() -> hanzo_ml::Result<()> {
    let device = Device::Cpu;
    let vars = VarMap::new();
    let vb = VarBuilder::from_varmap(&vars, DType::F32, &device);
    let model = linear(4, 2, vb.pp("fc"))?;
    let mut opt = AdamW::new(vars.all_vars(), ParamsAdamW::default())?;

    let x = Tensor::randn(0f32, 1., (8, 4), &device)?;
    let y = Tensor::zeros(8, DType::U32, &device)?;
    for _ in 0..10 {
        let logits = model.forward(&x)?;
        let l = loss::cross_entropy(&logits, &y)?;
        opt.backward_step(&l)?;
    }
    Ok(())
}
```

## Contents

- **Layers**: linear, embedding, convolutions, batch/group/layer/RMS norm, GRU and LSTM,
  rotary embeddings, KV cache, mixture of experts, `AttentionResiduals`.
- **Attention** (`attention`): flash attention on the CPU (dense, causal and packed variable
  length), WMMA flash attention on ROCm gfx11, packed-sequence attention on CUDA, and an unfused
  variable-length fallback.
- **Fused ops** (`ops`, `rotary_emb`). Each carries its own backward pass, and
  `tests/grad.rs` checks its gradients against the unfused composite on CPU, CUDA and Metal.

  | Op | Fused forward kernel on |
  |---|---|
  | `softmax_last_dim`, `rms_norm` | CPU, CUDA, Metal, ROCm, Vulkan |
  | `layer_norm` | CPU, CUDA, Metal |
  | `rope`, `rope_i`, `rope_thd` | CPU, CUDA, Metal; `rope` also ROCm and Vulkan |
  | `select` | CPU, CUDA, Metal, ROCm |
  | `silu_mul` | CPU, ROCm, Vulkan (composite on CUDA and Metal) |
  | `geglu` | CUDA (composite elsewhere) |

- **Optimizers** (`optim`): SGD, AdamW, Muon, MuonClip and `HybridMuonAdamW`. The AdamW update
  is one function (`optim::adamw`), a single kernel on CUDA and Metal; `optim::grad_norm` is the
  global gradient norm.
- **Hashed memory** (`memory`): rows addressed by hashed n-grams or keys, served from mapped
  safetensors shards (F32, BF16 or F8E4M3), trained with AdamW state for touched rows only.
- **SIGReg** (`sigreg`): the LeJEPA isotropic-Gaussian regularizer.
- **Weights**: `VarBuilder` over safetensors, npz, PyTorch files or a `VarMap`.

Features `cuda`, `cudnn`, `rocm`, `metal`, `vulkan`, `mkl` and `accelerate` forward to hanzo-ml;
`cutile` adds a BF16 routed mixture-of-experts GEMM (`moe::cutile`).

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
