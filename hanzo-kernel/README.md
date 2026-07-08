# hanzo-kernel

**Write a GPU kernel once in Rust. Run it on CUDA, ROCm, Vulkan, Metal, WebGPU, and CPU — at or above hand-tuned speed.**

[![crates.io](https://img.shields.io/crates/v/hanzo-kernel.svg)](https://crates.io/crates/hanzo-kernel)
[![docs.rs](https://img.shields.io/docsrs/hanzo-kernel)](https://docs.rs/hanzo-kernel)
[![license](https://img.shields.io/crates/l/hanzo-kernel.svg)](./LICENSE)

```rust
use hanzo_kernel::prelude::*;

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu))]
fn rms_norm<F: Float>(x: &Array<F>, w: &Array<F>, out: &mut Array<F>, #[comptime] n: usize) {
    let row = ABSOLUTE_POS;
    if row < out.len() / n {
        let base = row * n;
        let mut ss = F::new(0.0);
        for i in 0..n { let v = x[base + i]; ss += v * v; }
        let denom = (ss / F::cast_from(n) + F::new(1e-6)).sqrt();
        for i in 0..n { out[base + i] = x[base + i] / denom * w[i]; }
    }
}
```

That is the whole kernel. It lowers to native SPIR-V, PTX, HIP, and MSL — no per-backend rewrite, no shading-language dialect, no `unsafe` FFI.

---

## Why

A production inference stack hand-writes the same ~20 operations four times — once per backend, in four languages (CUDA C++, Metal Shading Language, HIP C++, GLSL). That Cartesian product isn't just a maintenance tax; it is where the bugs breed, because **one operation with four implementations has four numeric behaviors.** A model that generates coherently on Metal can collapse into repetition on CUDA — same weights, same math, different rounding in a hand-tuned kernel.

`hanzo-kernel` collapses the **product** into a **sum**: `backends × types × ops` → `backends + types + ops`. One Rust source per operation, lowered everywhere. Uniform numeric behavior stops being something you chase with four test suites and becomes a structural property of a single source.

## The claim, measured

Portability usually means a performance downgrade. It doesn't here. Every number below is measured, bit-exact-gated against a CPU oracle, and reproducible with the bundled `matvec-check` binary.

| Kernel | Backend | Result |
|---|---|---|
| int8 dp4a matvec | Vulkan (AMD gfx1151) | **196 GB/s — 118% of the hand-tuned kernel it replaces** |
| int8 dp4a matvec | Metal (Apple M4 Max) | 206 GB/s |
| `rms_norm` | Vulkan, in the live engine | **10.6× the hand-written kernel** it replaced (now deleted) |
| `softmax` | Vulkan, in the live engine | 351 GB/s (replaced the hand-written kernel) |
| all ops | CPU / Vulkan / Metal | **bit-exact from one source** (max_rel ≤ 1e-3) |

The `dot` on a `Line<i8>` lowers to a single `OpSDot` on Vulkan (verified in the emitted SPIR-V binary) — the *same* hardware integer-dot instruction the hand-written kernels use, 2.35× a scalar loop. The DSL doesn't emulate the fast path; it reaches it.

## Install

```toml
[dependencies]
hanzo-kernel = "0.2"
```

Pick your backend with a feature flag (drop `cpu` for GPU-only builds — it pulls an LLVM/MLIR toolchain):

```toml
# GPU-only, no CPU toolchain
hanzo-kernel = { version = "0.2", default-features = false, features = ["vulkan"] }
```

| Feature | Backend |
|---|---|
| `cpu` *(default)* | CPU reference runtime — the bit-exact oracle |
| `vulkan` | Vulkan (SPIR-V) — AMD, Intel, NVIDIA |
| `metal` | Metal (MSL) — Apple Silicon |
| `cuda` | CUDA (PTX/NVRTC) — NVIDIA |
| `rocm` | ROCm (HIP) — AMD |

## The surface

Three attributes and two nouns — named for what they are, not for a framework.

- `#[kernel(targets(...))]` — a launch entry point. `targets(...)` is validated at compile time (an unknown backend is a compile error) and self-documents where the kernel runs.
- `#[device]` — an on-GPU helper function, inlined into a kernel. A kernel is *launched*; a device function is a *piece* of one.
- `Grid` / `Block` — the launch shape: the grid of thread blocks, and the block of threads.
- `Array`, `Float`, `Line` — the data. Already the simple, accurate words.

Every performance primitive is a first-class part of the surface and lowers to the native instruction:

| Primitive | Lowers to |
|---|---|
| `Line<i8>::dot` | `dp4a` / `OpSDotAccSat` (int8 4-way dot) |
| `cmma` | tensor cores (WMMA / cooperative-matrix / simdgroup) |
| `plane_sum`, `plane_broadcast` | subgroup / warp reduce & shuffle |
| `SharedMemory`, `Atomic`, barriers | native shared memory, atomics, sync |

## Built-in op library

Reusable, bit-exact kernels — not toys. Each ships with a CPU oracle and a bit-exact gate.

- **`quant`** — `matvec_q8`, `matvec_q4k` (in-kernel K-quant decode), dp4a matvec (`_blk` block-per-row + `_sg` subgroup variants), packed-Q8_0 decode with in-kernel fp16 unpack.
- **`norm`** — `rms_norm`, `layer_norm`, `add_rmsnorm` (fused, multi-output).
- **`rope`** — `rope_half` (GPT-NeoX) and `rope_interleaved` (GPT-J) conventions.
- **`attn`** — `sdpa` and `sdpa_runtime`: GQA + online (flash-style) softmax with a runtime-length KV cache. One stable attention implementation across backends — the structural cure for repetition-collapse.
- **`gdn`** — Gated-DeltaNet linear-attention for hybrid `qwen3_5_moe` archs (Qwen3.5 / 3.6 / AgentWorld): `gdn_conv1d` (causal depthwise conv1d + SiLU), `gdn_gating` (fused `beta`/`g` gates), `gdn_scan` (the recurrent gated delta-rule, fused into one launch). One source replaces the ops-composed CUDA/ROCm GDN path.
- **`fuse`** — auto-fusion, because **fusion is composition**. `Fuse::new(a).mul(w).add(b).silu().run()` folds a chain of pointwise Maps into ONE kernel launch with zero materialized intermediates — the functor law `map g . map f == map (g . f)` read right-to-left. Legal fusion is a total function of op *class* (Map = index-local, freely fusible; Reduce = a fence), not a pattern-matcher. Bit-exact-gated three ways (fused kernel == naive N-kernel == plain-Rust reference).

Call them straight from your own crate — the common case when you fork a model and want the transformer ops without writing kernels. `use hanzo_kernel::prelude::*;` (it brings the `Runtime` trait into scope; you always want it):

```rust
use hanzo_kernel::cubecl::cpu::{CpuDevice, CpuRuntime};
use hanzo_kernel::prelude::*;

let client = CpuRuntime::client(&CpuDevice::default()); // or a CUDA/Metal/Vulkan runtime — same call

let normed = hanzo_kernel::norm::rms_norm_run::<CpuRuntime>(&client, &x, &weight, rows, hidden, 1e-6);
let rotated = hanzo_kernel::rope::rope_run::<CpuRuntime>(&client, &x, &cos, &sin, rows, head_dim, /*interleaved=*/false);
let y = hanzo_kernel::quant::matvec_q8_run::<CpuRuntime>(&client, &scales, &qweight, &x, out_rows, k);
```

`cargo run --example model_ops` runs all of them; `cargo run --example hello_kernel` shows authoring a `#[kernel]`. Run the correctness + throughput gate yourself:

```bash
cargo run --release --bin matvec-check --no-default-features --features "cpu,vulkan"
```

## Design

Four ideas, borrowed from Rich Hickey and applied to kernels:

- **Decomplect.** Ops, quantization types, and backends are three orthogonal axes braided together in hand-written code. Pulled apart, they compose — you write each axis once. `backends × types × ops` becomes `backends + types + ops`.
- **Values, not places.** [CubeCL](https://github.com/tracel-ai/cubecl) is the lowering engine — the implementation *value*. `hanzo_kernel` is the stable *namespace* you build against; the engine is named only in this crate's `Cargo.toml`, never in a kernel source. It can be upgraded, vendored, or forked (the `rocm` feature pulls `hanzo-cubecl-hip`, our published fork that builds on ROCm ≥ 7.13) without touching a kernel.
- **One numeric behavior.** A single source means a single rounding path. The "coherent on Metal, collapsed on CUDA" class of bug cannot occur — not because four copies were tested into agreement, but because there is one copy.
- **Perf-gated migration, never a downgrade.** A hand-tuned kernel is retired for its DSL twin **only** when the twin is bit-exact *and* at least as fast. Where a hand-tuned kernel still wins, it stays as the peak path and the DSL is the portable fallback for backends that lack it. Coverage always grows; speed never regresses.

## Status

Published and in production. `rms_norm` and `softmax` are live DSL kernels in the [Hanzo ML](https://github.com/hanzoai/ml) inference engine — their hand-written predecessors are deleted. The full op library is bit-exact on CPU, Vulkan, and Metal, and shelf-ready for the backends still on hand-tuned kernels. A generated DSL kernel dispatches through the engine's real Vulkan and Metal pipelines with `maxerr = 0` — the DSL plugs in as a **code generator**, not a second runtime.

## The stack

| Crate | What |
|---|---|
| **hanzo-kernel** | this crate — write a GPU kernel once, lower it to every backend |
| [**hanzo-ml**](https://crates.io/crates/hanzo-ml) | the multi-backend tensor + ML framework (6 backends, the full quant zoo) |
| [**hanzo-flash-attn**](https://crates.io/crates/hanzo-flash-attn) | flash-attention-2 CUDA kernels |
| [**hanzo-kernels**](https://crates.io/crates/hanzo-kernels) | the hand-tuned CUDA quant kernels the DSL is migrating |
| [Hanzo Engine](https://github.com/hanzoai/engine) | the serving engine: OpenAI + Anthropic + MCP APIs |

## License

BSD-3-Clause. See [LICENSE](./LICENSE).
