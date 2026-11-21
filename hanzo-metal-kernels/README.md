# hanzo-metal-kernels

Metal kernels and the Metal command layer for the [hanzo-ml](https://crates.io/crates/hanzo-ml)
Metal backend. MSL sources are in `src/metal_src`: elementwise, reductions, sort, indexing,
convolution, GEMM and GEMV, quantized matvec and matmul (including the IQ codebooks), scaled
dot-product attention, random numbers, and the training kernels (`optim.metal` for AdamW and
the gradient sum of squares, `backward.metal` for layer-norm and softmax backward).

| Setting | Effect |
|---|---|
| feature `debug-labels` | a debug group per dispatch, for Xcode GPU captures |
| `METAL_PROFILE_GPU` | report GPU-busy time at each host sync |
| `METAL_PROFILE_OPS` | report GPU time per kernel (one dispatch per command buffer) |
| `METAL_COMPUTE_PER_BUFFER` | dispatches per command buffer (default 50) |

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
