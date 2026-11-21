# hanzo-kernel

Write a GPU kernel once, as a Rust function. It runs on the CPU and lowers to CUDA, ROCm (HIP),
Metal (MSL) and Vulkan (SPIR-V). Each kernel has a host reference beside it, and the tests
compare every backend's output against that reference.

```toml
[dependencies]
hanzo-kernel = "0.2"                                                                  # CPU runtime
# hanzo-kernel = { version = "0.2", default-features = false, features = ["vulkan"] } # GPU only
```

`norm::rms_norm`, one thread per row, from [`src/norm.rs`](src/norm.rs):

```rust
use hanzo_kernel::prelude::*;

/// RMSNorm over the last dim: `out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]`, per row of `n`.
#[kernel(targets(cuda, rocm, metal, vulkan, webgpu, cpu), unchecked)]
pub fn rms_norm<F: Float>(
    x: &Array<F>,
    w: &Array<F>,
    out: &mut Array<F>,
    eps: &Array<F>,
    #[comptime] n: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() / n {
        let base = row * n;
        let mut ss = F::new(0.0f32);
        for i in 0..n {
            let v = x[base + i];
            ss += v * v;
        }
        let denom = (ss / F::cast_from(n as u32) + eps[0]).sqrt();
        for i in 0..n {
            out[base + i] = x[base + i] / denom * w[i];
        }
    }
}
```

Its reference is `norm::rms_norm_ref`. The block-per-row twin `rms_norm_blk` in the same file is
the one hanzo-ml's Vulkan backend runs. `cargo run --release --example hello_kernel --features
vulkan` runs `rms_norm` on the CPU and on the enabled GPU and prints each output's checksum and
its error against the reference.

## Features

| Feature | Runtime |
|---|---|
| `cpu` (default) | the CPU runtime, built on MLIR; its build downloads a prebuilt LLVM |
| `cuda` | CUDA |
| `rocm` | HIP, through `hanzo-cubecl-hip` (builds on ROCm 7.13 and later) |
| `vulkan` | wgpu with the SPIR-V compiler |
| `metal` | wgpu with the MSL compiler |
| `spirv-dump` | dump compiled SPIR-V; hanzo-ml's committed Vulkan DSL kernels are exported this way |

`targets(...)` also accepts `webgpu`; no feature here selects the plain WGSL compiler yet.

On ROCm 7.13, `libhiprtc.so.7` is in `/opt/rocm/core-7.13/lib`, so run the ROCm tests with
`LD_LIBRARY_PATH=/opt/rocm/core-7.13/lib cargo test --features rocm`.

## Surface

- `#[kernel(targets(...))]`: a launchable kernel. An unknown target is a compile error.
- `#[device]`: a helper inlined into kernels.
- `island! { cuda | rocm | metal | vulkan => { ... } default => { ... } }`: one region that picks
  a target-specific instruction. The kernel takes a `#[comptime] target: Target`; only the
  selected arm is compiled, and the CPU always takes `default`, which defines the answer.
- `Grid`, `Block`: the launch shape. `Array`, `Float`, `Line`: the data.
- `Line<i8>::dot` lowers to the 4-way int8 dot product (dp4a, `OpSDotAccSat`); `cmma` to the
  matrix cores; `plane_sum` and friends to subgroup operations.

The lowering engine is CubeCL 0.10. Kernel source names only `hanzo_kernel::*`.

## Modules

| Module | Holds |
|---|---|
| `norm` | `rms_norm`, `layer_norm`, `add_rmsnorm`, block-per-row variants |
| `rope` | rotary embedding, half-split and interleaved |
| `quant` | Q8_0, Q4_K, Q6_K matvecs, dp4a variants, mixture-of-experts matvecs and routing, f32 GEMV |
| `mmq` | int8 matrix-core GEMM for quantized prefill |
| `attn`, `flash` | scaled-dot-product attention with online softmax; tiled flash attention |
| `packed` | attention over packed sequences from a packed QKV projection, for encoder training |
| `grad`, `glu`, `optim` | softmax and LayerNorm backward, GeGLU and its gradient, AdamW and gradient sum of squares |
| `gdn` | Gated DeltaNet conv, gating and scan |
| `fuse`, `dag` | pointwise chains and DAGs fused into one launch |
| `place`, `route` | placement and routing of frozen experts across devices |
| `tune` | the autotuner |
| `island` | the `Target` tag behind `island!` |

## Autotuner

`tune::Tuned` collects named schedules of one op, times each once per `(device, op, shape)`, and
caches the winner in `$XDG_CACHE_HOME/hanzo-kernel/autotune/<device>.tsv`. Later calls run only
the winner. `norm::rms_norm_autotuned` and `quant::matvec_q8_dp4a_autotuned` use it; their tests
require every schedule to match the reference bit for bit. `tune::Evolution` searches a larger
`tune::Space` of knobs with a two-tier `Evaluator`: a free static check rejects a schedule before
any GPU time is spent on it.

## Checking

`matvec-check` runs the quant, norm and attention kernels on every compiled backend and prints
the maximum relative error against the host reference, `MATCH` or `MISMATCH`, and GB/s:

```bash
cargo run --release --bin matvec-check --no-default-features --features "cpu,vulkan"
```

`cargo test` runs the modules' tests on the CPU runtime.

## License

MIT OR Apache-2.0 ([LICENSE](LICENSE)). See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE)
for third-party attributions.
