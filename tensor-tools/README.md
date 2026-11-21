# tensor-tools

Command-line tools for tensor files, on [hanzo-ml](https://github.com/hanzoai/ml). Not
published.

```bash
cargo run --release -p tensor-tools --bin tensor-tools -- ls model.safetensors            # names, dtypes, shapes
cargo run --release -p tensor-tools --bin tensor-tools -- print model.gguf --full         # tensor contents
cargo run --release -p tensor-tools --bin tensor-tools -- quantize model.safetensors --out-file model.gguf --quantization q4k
cargo run --release -p tensor-tools --bin tensor-tools -- dequantize model.gguf --out-file model.safetensors
```

`ls` and `print` read safetensors, npz, GGML, GGUF and PyTorch (`.pt`/`.pth`, pickle) files,
inferring the format from the extension unless `--format` names it.

The `soup` binary merges safetensors checkpoints and compresses a fine-tune to a 1-bit delta:

```bash
cargo run --release -p tensor-tools --bin soup -- average -o merged.safetensors a.safetensors b.safetensors
cargo run --release -p tensor-tools --bin soup -- delta -b base.safetensors -o merged.safetensors ft1.safetensors ft2.safetensors
cargo run --release -p tensor-tools --bin soup -- bitdelta-encode -b base.safetensors -f ft.safetensors -o ft.bitdelta
cargo run --release -p tensor-tools --bin soup -- bitdelta-apply -b base.safetensors -d ft.bitdelta -o restored.safetensors
```

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
