# hanzo-ml-examples

Runnable programs on [hanzo-ml](https://github.com/hanzoai/ml): one directory per model or
technique in [`examples/`](examples), most with a README of their own.

```bash
cargo run --release -p hanzo-ml-examples --example quantized
cargo run --release -p hanzo-ml-examples --example whisper --features cuda
# or, from the repository root
make example NAME=quantized FEATURES=metal
```

| Feature | Adds |
|---|---|
| `cuda`, `cudnn`, `nccl`, `flash-attn` | CUDA, cuDNN, multi-GPU and FlashAttention-2 |
| `metal` | Metal |
| `mkl`, `accelerate` | BLAS for the CPU |
| `onnx` | the ONNX examples |
| `microphone`, `encodec`, `mimi`, `snac` | audio capture and codecs for the speech examples |

Most examples download their weights on first run and cache them; each README names the
checkpoint it uses. This crate is not published.

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
