# hanzo-flash-attn

FlashAttention-2 forward kernels (the `sm80` kernel family) for the
[hanzo-ml](https://crates.io/crates/hanzo-ml) CUDA backend.

- `flash_attn`, `flash_attn_windowed`, `flash_attn_alibi`, `flash_attn_alibi_windowed`,
  `flash_attn_alibi_windowed_softcap`, and `flash_attn_varlen*` for packed variable-length
  batches, including a paged-KV variant.
- F16 and BF16; head dims in multiples of 8, up to 256 for `flash_attn*` and up to 512 for
  `flash_attn_varlen*`.
- BF16 inputs run through the F16 kernel by default, since the softmax probabilities lose
  precision as BF16 before the second matmul. `CUDA_FLASH_BF16=1` runs the BF16 kernel instead.

The build fetches CUTLASS headers at a pinned commit through
[hanzo-flash-attn-build](https://crates.io/crates/hanzo-flash-attn-build). For Hopper `sm_90a`,
see [hanzo-flash-attn-v3](https://crates.io/crates/hanzo-flash-attn-v3).

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
