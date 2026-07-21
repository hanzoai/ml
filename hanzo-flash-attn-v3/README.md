# hanzo-flash-attn-v3

FlashAttention-3 (Hopper `sm_90a`) attention kernels for the Hanzo ML stack.

FlashAttention-3 (Shah et al., [arXiv:2407.08608](https://arxiv.org/abs/2407.08608))
is built on Hopper-specific hardware: a warp-specialized producer/consumer
pipeline, TMA async copy, `wgmma`, and the GEMM↔softmax "pingpong" overlap.
Its kernels therefore run only on `sm_90a` datacenter GPUs (H100 / H200).

## What is here

- **Kernels** (`hkernel/`): the full FA3 Hopper forward set — head dims
  **64 / 128 / 256 / 512**, `fp16` and `bf16`, dense plus GQA-packed
  (`gqa2/4/8/16/32`), causal, local (windowed), and variable-length, with paged
  KV TMA copies. Ported faithfully from the reference `hopper/` kernels.
- **Rust API** (`src/`): `flash_attn`, `flash_attn_windowed`, `flash_attn_alibi`,
  their varlen counterparts, and the alibi/softcap variants, all as
  `CustomOp3`s over `hanzo_ml::Tensor`.

## Arch gate

The kernels compile **only** under the `cuda` feature, and `build.rs`
cross-compiles them for `sm_90a` alone (`nvcc -gencode arch=compute_90a,code=sm_90a`
— which targets Hopper from any CUDA host, so a datacenter binary builds
anywhere). Without the feature this crate is a **pure-Rust no-op stub**: ROCm,
Metal, Vulkan, CPU and non-Hopper CUDA (e.g. sm_121 / GB10) builds pull in zero
CUDA and link nothing, exactly as when the crate was excluded from the
workspace.

Selection between FA3 and a portable kernel is the caller's job. `hanzo-engine`'s
CUDA attention backend reads the device compute capability at runtime and routes
`sm_90a` here, everything else to the portable FA2-class path.

## Build

```bash
# Datacenter build (kernels compiled for sm_90a):
cargo build -p hanzo-flash-attn-v3 --features cuda

# Everything else (stub, no CUDA):
cargo build -p hanzo-flash-attn-v3
```

Env knobs (feature build only): `FLASH_ATTN_BUILD_DIR` (persistent object
cache), `FLASH_ATTN_V3_NVCC_JOBS` (parallel nvcc jobs, default 6), `NVCC`,
`CUTLASS_DIR` (pre-staged CUTLASS, else fetched at the pinned commit).

## Correctness

`tests/flash_attn_tests.rs` asserts FA3 matches an f32 CPU oracle within a
scale-relative bound across head dims, dtypes, causal, GQA, and varlen. The
tests require Hopper and are `#[ignore]`d; on an H100/H200:

```bash
cargo test -p hanzo-flash-attn-v3 --features cuda -- --ignored
```

## FP8

The `hopper/` reference also carries an `e4m3` (FP8) path with Hadamard
incoherent processing. It is **not** compiled here: this fork's `PREC_SWITCH`
does not dispatch fp8 and the extern-C entry point carries no de-scale factors,
so an fp8 build would be inert. Re-enabling it means adding the `descale_{q,k,v}`
pointers to `run_mha_v3` and the fp8 arm to `PREC_SWITCH` — a self-contained
follow-up, left as a documented hook rather than shipped unvalidated.
