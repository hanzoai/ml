# hanzo-kernels

CUDA kernels for the [hanzo-ml](https://crates.io/crates/hanzo-ml) CUDA backend. Built by
`build.rs` with `nvcc` through [hanzo-bindgen-cuda](https://crates.io/crates/hanzo-bindgen-cuda).

- **PTX modules**, loaded at runtime: affine, binary, cast, conv, fill, indexing, reduce, sort,
  ternary, unary, the quantized matvec and matmul kernels, the i-quant matvecs, packed-sequence
  attention, and the training kernels (layer-norm and softmax backward, GeGLU forward and
  backward, AdamW and sum of squares).
- **Linked launchers** (`ffi`), compiled into a static library: GGUF matvec and matmul per quant
  type, mixture-of-experts GEMMs (plain, WMMA and GGUF), and a fused head-dim-512 flash-decode.

Set `CUDA_COMPUTE_CAP` (e.g. `121` for GB10) to compile for one architecture; otherwise the
build asks `nvidia-smi`. `NVCC` and `NVCC_CCBIN` choose the compiler and its host compiler.

Some of these kernels come from the [dfdx](https://github.com/coreylowman/dfdx) crate.

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
