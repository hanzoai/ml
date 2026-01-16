# Hanzo ML Framework Integration Guide

## Overview

Hanzo ML is the official ML framework for the Hanzo ecosystem, based on Candle from Hugging Face with optimizations for edge AI, multimodal workloads, and integration with Hanzo Engine.

## Integration with Hanzo Engine

### Repository Structure
```
~/work/hanzo/
├── ml/                 # hanzoai/ml - ML framework (based on HF candle)
├── engine/             # hanzoai/engine - Inference engine (based on mistral-rs)
├── jin/                # Jin multimodal models
└── llm/                # LLM Gateway proxy
```

### Dependencies in Hanzo Engine

Add to `~/work/hanzo/engine/Cargo.toml`:

```toml
[dependencies]
hanzo-ml = { git = "https://github.com/hanzoai/ml", branch = "main" }
hanzo-ml-nn = { git = "https://github.com/hanzoai/ml", branch = "main" }
hanzo-ml-transformers = { git = "https://github.com/hanzoai/ml", branch = "main" }
```

### Feature Alignment

Both projects support consistent feature flags:

```toml
[features]
default = ["metal"]
metal = ["hanzo-ml/metal", "hanzo-ml-nn/metal"]
cuda = ["hanzo-ml/cuda"] 
mkl = ["hanzo-ml/mkl"]
accelerate = ["hanzo-ml/accelerate"]
```

## Model Loading Integration

### In Hanzo Engine (mistral-rs fork)

```rust
use hanzo_ml_core::{Device, Tensor};
use hanzo_ml_transformers::models::llama::LlamaConfig;

// Load model using Hanzo ML
let device = Device::new_metal(0)?;
let model = LlamaConfig::load(&device, &config_path)?;

// Use with mistral-rs pipeline
let pipeline = Pipeline::new(model, tokenizer)?;
```

### Quantization Support

Both frameworks support:
- **AFQ (Affine Quantization)** - Optimized for Metal/Apple Silicon
- **GGUF/GGML** - Universal quantization format
- **GPTQ/AWQ** - GPU-optimized quantization
- **In-Situ Quantization (ISQ)** - Runtime quantization

## Development Workflow

### 1. Update ML Framework

```bash
cd ~/work/hanzo/ml
git fetch upstream
git merge upstream/main  # Merge HF candle updates
cargo test --workspace
git push origin main
```

### 2. Update Engine Dependencies

```bash
cd ~/work/hanzo/engine
cargo update hanzo-ml hanzo-ml-nn hanzo-ml-transformers
cargo test
```

### 3. Test Integration

```bash
cd ~/work/hanzo/engine
cargo run --features metal --release -- \
    -i --isq 4 plain -m meta-llama/Llama-3.2-3B-Instruct
```

## Publishing to Crates.io

### Hanzo ML Crates

The framework publishes these crates:
- `hanzo-ml` - Core tensor operations
- `hanzo-ml-nn` - Neural network layers  
- `hanzo-ml-transformers` - Transformer models
- `hanzo-ml-datasets` - Dataset utilities
- `hanzo-ml-pyo3` - Python bindings

### Release Process

```bash
cd ~/work/hanzo/ml
cargo release --workspace minor
git push --tags
```

## Sync Status

### Latest Upstream Sync
- **HF Candle**: `a2029da3` (Jan 2025)
- **Features Added**: SmolLM3, Qwen3 WASM, Mamba2, PaddleOCR-VL

### Engine Integration Status
- ✅ Metal backend support
- ✅ AFQ quantization compatibility  
- ✅ SIMD optimizations
- ✅ Memory introspection
- 🔄 Jin model integration (in progress)

## Performance Optimizations

### Apple Silicon (Metal)
- Use `AFQ4` quantization for best performance
- Enable `--features "metal accelerate"`
- Set group size to 64 for balanced speed/accuracy

### CUDA
- Use `GPTQ` or `AWQ` quantization
- Enable Flash Attention for long sequences
- Use PagedAttention for memory efficiency

### CPU
- Use `GGUF` models with appropriate quantization
- Enable `mkl` feature for Intel optimizations
- Consider `accelerate` on Apple platforms

## Troubleshooting

### Build Issues
```bash
# Clean and rebuild
cd ~/work/hanzo/ml
cargo clean
cargo build --workspace

# Check feature alignment
cd ~/work/hanzo/engine  
cargo tree | grep hanzo-ml
```

### Runtime Issues
```bash
# Metal validation
cd ~/work/hanzo/engine
cargo run --features metal -- --help

# Check device detection
RUST_LOG=debug cargo run --features metal
```

## Future Roadmap

1. **Model Format Standardization** - Universal model interchange
2. **Joint Training Pipeline** - Train models for both frameworks  
3. **Distributed Inference** - Multi-device model serving
4. **WebAssembly Optimization** - Browser-based inference
5. **MCP Integration** - Model Context Protocol support

## Contact

For issues with Hanzo ML integration:
- GitHub Issues: [hanzoai/ml](https://github.com/hanzoai/ml/issues)
- Engine Issues: [hanzoai/engine](https://github.com/hanzoai/engine/issues)