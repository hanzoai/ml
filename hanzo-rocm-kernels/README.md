# hanzo-ml-rocm-kernels

ROCm/HIP kernel support for the Hanzo deep learning framework.

## Overview

This crate provides ROCm (AMD GPU) kernel support for Hanzo. Unlike CUDA which can embed PTX directly, ROCm/HIP requires ahead-of-time (AOT) compilation for specific GPU architectures.

## Architecture

### AOT Cache System

We use an **Ahead-of-Time (AOT) compilation cache** approach:

1. **Source Code**: HIP kernels are shipped as source code (`.hip` files)
2. **On-Demand Compilation**: First time a kernel is needed, it's compiled using `hipcc`
3. **Caching**: Compiled binaries are cached for reuse
4. **Future Runs**: Load cached binaries directly (no recompilation)

### Cache Location

Compiled binaries are stored at:

```
~/.cache/hanzo-ml-rocm/{arch}-{rocm_version}/
```

For example:
- `~/.cache/hanzo-ml-rocm/gfx908-6.1/binary_a1b2c3d4.cso`
- `~/.cache/hanzo-ml-rocm/gfx942-6.2/binary_a1b2c3d4.cso`

Where:
- `{arch}` = GPU architecture (gfx908, gfx90a, gfx942, etc.)
- `{rocm_version}` = ROCm version (6.0, 6.1, 6.2, etc.)
- `{hash}` = SHA256 hash of source code (first 16 chars)

### Key Components

#### CacheManager (`src/cache.rs`)

Manages the AOT compilation cache:

- **GPU Detection**: Automatically detects GPU architecture using `rocminfo` or falls back to environment variable `ML_ROCM_ARCH`
- **Version Detection**: Detects ROCm version using `hipcc --version` or environment variable `ML_ROCM_VERSION`
- **Compilation**: Invokes `hipcc` with appropriate flags:
  ```bash
  hipcc --offload-arch={arch} -O3 -fPIC -c -o output.o input.hip
  ```
- **Caching**: Stores compiled `.cso` (code object) files with source hash versioning

Usage:
```rust
use hanzo_rocm_kernels::CacheManager;
use rocm_rs::hip::Device;

let device = Device::new(0)?;
let cache = CacheManager::new(&device)?;
let binary = cache.get_or_compile("binary", source_code)?;
```

#### KernelManager (`src/manager.rs`)

Higher-level manager that:

- Wraps CacheManager
- Loads compiled binaries as `rocm_rs::hip::Module`
- Returns `Arc<Module>` for thread-safe sharing
- Maintains in-memory module cache

Usage:
```rust
use hanzo_rocm_kernels::KernelManager;
use hanzo_rocm_kernels::source::Source;

let device = Device::new(0)?;
let manager = KernelManager::new(&device)?;
let module = manager.get_or_compile_module(Source::Binary)?;
```

### Environment Variables

- `ML_ROCM_ARCH` - Override GPU architecture detection (e.g., "gfx908")
- `ML_ROCM_VERSION` - Override ROCm version detection (e.g., "6.1")

### Experimental grouped GGUF MoE prefill on gfx1201

Set `ML_ROCM_GFX1201_GROUPED_Q4K=hip` before creating a ROCm device.
The original opt-in value `1` also selects HIP. Set the value to
`dsl` to use the generated projection with the same routing plan.
Both modes enable the shared Q4_K routing plan for F16 multi-token prefill
when a model calls `try_moe_shared_q4k`. The gate and up projections
share Q8_1 activations. A compatible Q4_K down projection reuses the
routing plan as well. Each wave reuses its packed weights across four
routes and uses the indexed matvec reduction for remainder routes.

The plan reads back one tile-count scalar to validate expert IDs before
scattering the routes. Single-token decode and unsupported expert
projections on `gfx1201` use the indexed matvec path. If kernel target
detection selects another architecture, set `ML_ROCM_ARCH=gfx1201` for
the ROCm kernel cache.

The explicit HIP fallback remains the faster opt-in on a Radeon AI PRO R9700
(gfx1201, ROCm 7.2.2). With Qwen3-30B-A3B layer 6 Q4_K weights and synthetic
activations and routes, the vectorized DSL projection matched HIP bit for bit
at eight stages in each of eight workloads (65 or 257 tokens, top-2 or top-8,
concentrated or spread routes). Across these workloads, median paired DSL
latency relative to HIP was 0.300–17.854% higher for gate/up/SwiGLU and Q8_1
requantization, 6.145–19.213% higher through combine, and 4.986–16.674% higher
including route planning and the initial Q8_1 quantization (GPU event timing).
The corresponding wall-time ranges were 0.424–17.817%, 6.108–19.153%, and
4.994–16.663%. Each workload and scope used 12 alternating paired samples.
These are isolated layer measurements on one GPU.

Run the hardware regression from the workspace root on a gfx1201 device
with ROCm configured:

```sh
for mode in hip dsl; do
  ML_ROCM_GFX1201_GROUPED_Q4K="$mode" ML_ROCM_ARCH=gfx1201 \
    cargo test -p hanzo-ml --features rocm --test rocm_grouped_q4k -- --ignored
done
```

Regenerate and check the projection on a gfx1201 device from `hanzo-kernel/`.

```sh
cargo run --release --no-default-features --features rocm --bin kernel-gen -- \
  --target rocm --only moe_grouped_q4k_gfx1201 --out ../hanzo-ml/src/rocm/dsl
cargo run --release --no-default-features --features rocm --bin kernel-gen -- \
  --target rocm --only moe_grouped_q4k_gfx1201 --check ../hanzo-ml/src/rocm/dsl
```

## Requirements

- ROCm/HIP installed (provides `hipcc`)
- AMD GPU with supported architecture

## Kernel Types

Currently supports:
- **Binary operations**: Add, Sub, Mul, Div, Minimum, Maximum

## Building

```bash
cd hanzo-ml-rocm-kernels
cargo build
```

Note: First build will compile dependencies. No GPU required for building, but `hipcc` must be in PATH if you want to compile kernels.

## Testing

```bash
cargo test
```

## Implementation Notes

### Why AOT instead of JIT?

The `rocm-rs` crate (v0.5) doesn't support runtime compilation. It only supports:
- `Module::load(path)` - Load from file
- `Module::load_data(bytes)` - Load from bytes

This makes JIT compilation (via hiprtc) unavailable, so we compile ahead-of-time on first run.

### Supported GPU Architectures

Common AMD GPU architectures:
- CDNA2: gfx90a (MI200 series)
- CDNA3: gfx942 (MI300 series)
- RDNA3: gfx1100, gfx1101, gfx1102 (RX 7000 series)

The system will try to auto-detect, but you can override with `ML_ROCM_ARCH`.

## License

MIT OR Apache-2.0
