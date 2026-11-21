# hanzo-rocm-kernels

HIP kernels for the [hanzo-ml](https://crates.io/crates/hanzo-ml) ROCm backend, and the cache that
compiles them.

## How a kernel gets to the GPU

Kernels ship as HIP source (`src/kernels/*.hip`). The first time one is needed, `KernelCache`
compiles it with `hipcc --offload-arch=<arch> -O3 --genco`, extracts the code object for that
arch with `clang-offload-bundler`, and stores it as
`~/.cache/hanzo-ml-rocm/<arch>-<rocm version>/<name>_<source hash>.cso`. Later runs load the
cached object. A changed source gets a new hash and recompiles.

| Variable | Effect |
|---|---|
| `ROCM_GFX_ARCH` | the target arch, e.g. `gfx1151`; else read from `rocminfo`, else `gfx1151` |
| `ROCM_VERSION` | the version in the cache path; else parsed from `hipcc --version` |
| `ROCM_PATH` | where to find `clang-offload-bundler` and the other ROCm tools |

## Kernels

| File | Holds |
|---|---|
| `quant.hip` | the unified quantized core: int8 dot-product matvec, int8 WMMA prefill GEMM, on-device dequantize, mixture-of-experts matvecs |
| `flash.hip` | flash-attention decode (single query, strided KV, GQA) |
| `gdn.hip` | Gated DeltaNet conv, gating and a fused scan |
| `dsl_rms_norm.hip`, `dsl_add_rmsnorm.hip` | generated from the [hanzo-kernel](https://crates.io/crates/hanzo-kernel) DSL, do not edit; used by hanzo-ml's A/B tests, while production keeps the faster hand-written norms in `reduce.hip` |
| `affine`, `binary`, `unary`, `ternary`, `cast`, `fill`, `indexing`, `reduce`, `sort`, `conv`, `rope` | the rest of the op set |

The prefill GEMM covers Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q4_K, Q5_K, Q6_K, IQ4_XS, TQ2_0,
ROCmFP4 and ROCmFP4 fast. WMMA paths are compiled for gfx11 parts (gfx1100–gfx1103,
gfx1150–gfx1152).

## Bench

[`bench/`](bench) is a standalone HIP harness for the prefill GEMM: a numeric oracle and a
microbenchmark against dequantize-plus-rocBLAS. Its README records the gfx1151 results.

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
