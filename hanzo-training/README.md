# Hanzo Training

**Native Rust training framework for Hanzo ML models** using the zen-agentic-dataset and zen-identity-dataset.

[![Crates.io](https://img.shields.io/crates/v/hanzo-training.svg)](https://crates.io/crates/hanzo-training)
[![License: BSD-3-Clause OR Apache-2.0](https://img.shields.io/badge/license-BSD--3--Clause%20OR%20Apache--2.0-blue)](LICENSE)

## Features

- 🚀 **Native Rust Performance** - No Python overhead, maximum speed
- 🎯 **Zero-Copy Data Loading** - Efficient memory usage with large datasets  
- 🔥 **Multi-GPU Support** - CUDA and Metal acceleration via hanzo-ug
- 📊 **Built-in Evaluation** - Integrated benchmarking and metrics
- 🎛️ **Flexible Configuration** - YAML/TOML configuration files
- 📈 **W&B Integration** - Optional Weights & Biases logging

## Quick Start

### Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
hanzo-training = "0.9.2-alpha.2"
```

Or install the CLI:

```bash
cargo install hanzo-training
```

### Training a Model

```rust
use hanzo_training::{TrainingConfig, Trainer};

let config = TrainingConfig::from_file("config.yaml")?;
let trainer = Trainer::new(config)?;
trainer.train()?;
```

### Command Line

```bash
# Train Qwen-4B on zen-agentic-dataset
hanzo-train --config configs/qwen4b.yaml --dataset /path/to/zen-agentic-dataset

# Evaluate model
hanzo-eval --model ./output/qwen4b --benchmarks all

# Train with GPU acceleration
hanzo-train --config configs/llama7b.yaml --device cuda:0

# Train with multiple GPUs
hanzo-train --config configs/llama70b.yaml --device cuda --multi-gpu
```

## Integration with Zen Agentic Dataset

The Hanzo Training framework is specifically designed to work with the **10.5B token zen-agentic-dataset**:

### Dataset Structure

```
zen-agentic-dataset-private/
├── claude-interactions.jsonl    # Real Claude Code sessions
├── train_chunks/               # Training data splits
│   ├── chunk_001.jsonl
│   ├── chunk_002.jsonl
│   └── ...
├── valid.jsonl                 # Validation data  
└── metadata.json              # Dataset metadata
```

### Sample Data Format

```json
{
  "conversation": [
    {
      "role": "human",
      "content": "Help me implement a neural network in Rust",
      "timestamp": "2024-01-15T10:30:00Z",
      "tools_used": ["editor", "terminal"]
    },
    {
      "role": "assistant", 
      "content": "I'll help you create a neural network in Rust...",
      "timestamp": "2024-01-15T10:30:15Z"
    }
  ],
  "code_blocks": [
    {
      "language": "rust",
      "content": "use candle_core::*;\n\nstruct NeuralNet {\n    layers: Vec<Linear>,\n}",
      "file_path": "src/neural_net.rs",
      "diff": "+use candle_core::*;\n+\n+struct NeuralNet {"
    }
  ],
  "git_context": {
    "commit_hash": "a1b2c3d4",
    "branch": "feature/neural-net",
    "files_changed": ["src/neural_net.rs", "Cargo.toml"]
  },
  "metadata": {
    "session_id": "claude_session_12345",
    "project_type": "machine_learning",
    "complexity": "intermediate"
  }
}
```

### Identity Training Dataset

For personality and identity fine-tuning:

```
zen-identity-dataset/
├── train.jsonl                # Identity training samples
├── zen-coder_train.jsonl     # Zen Coder personality
├── zen-eco_train.jsonl       # Zen Eco personality  
├── zen-nano_train.jsonl      # Zen Nano personality
├── zen-omni_train.jsonl      # Zen Omni personality
└── zen-next_train.jsonl      # Zen Next personality
```

## Training Pipeline

### 1. Foundation Training (Zen Agentic Dataset)

```bash
# Train base model on agentic programming data
hanzo-train --config configs/zen-agentic-qwen4b.yaml
```

**Configuration:**
```yaml
model:
  name: "qwen3-4b"
  checkpoint: "Qwen/Qwen3-4B-Instruct"
  max_seq_length: 4096

dataset:
  name: "zen-agentic"
  path: "/Users/z/work/zen/zen-agentic-dataset-private"
  format: "jsonl"

training:
  batch_size: 4
  learning_rate: 2e-4
  epochs: 2
  lora:
    enabled: true
    r: 64
    alpha: 128
```

### 2. Identity Fine-tuning (Zen Identity Dataset)

```bash  
# Fine-tune personality and identity
hanzo-train --config configs/zen-identity.yaml \
  --checkpoint ./output/zen-coder-4b
```

**Configuration:**
```yaml
model:
  name: "zen-coder-4b"
  checkpoint: "./output/zen-coder-4b"  # Pre-trained model

dataset:
  name: "zen-identity"
  path: "/Users/z/work/zen/zen-identity-dataset"
  
training:
  batch_size: 2
  learning_rate: 1e-4  # Lower for fine-tuning
  epochs: 3
```

### 3. Evaluation

```bash
# Run comprehensive evaluation
hanzo-eval all --model ./output/zen-identity-model \
  --output ./results
```

## Supported Models

| Model Family | Sizes | Architecture | Memory (LoRA) | Status |
|--------------|--------|--------------|---------------|---------|
| **Qwen3** | 4B, 7B, 14B | Transformer | 8-24 GB | ✅ Supported |
| **Llama 3.1** | 8B, 70B | Transformer | 16-80 GB | ✅ Supported |
| **Mistral** | 7B, 22B | Transformer | 14-32 GB | ✅ Supported |
| **Gemma** | 2B, 7B | Transformer | 4-14 GB | ✅ Supported |
| **Phi-3** | 3.8B, 14B | Transformer | 8-24 GB | 🔄 In Progress |

## Performance Benchmarks

Training speeds on different hardware configurations:

| Model | Hardware | Batch Size | Tokens/sec | Memory | Cost/Hour |
|-------|----------|------------|------------|---------|-----------|
| Qwen3-4B | RTX 4090 | 4 | 2,048 | 18 GB | Local |
| Qwen3-4B | M3 Max | 4 | 1,024 | 32 GB | Local |
| Llama3-8B | RTX 4090 | 2 | 1,536 | 22 GB | Local |
| Llama3-8B | A100-40GB | 8 | 4,096 | 35 GB | $3.50 |
| Mistral-7B | RTX 3090 | 2 | 1,280 | 20 GB | Local |

**Training Cost Estimates** (8.5B tokens, zen-agentic-dataset):

| Model | Cloud (8xH200) | Mac Studio 512GB |
|-------|----------------|-------------------|
| Qwen3-4B | $326 (9h) | 2 days (FREE) |
| Llama3-8B | $814 (23h) | 4 days (FREE) |
| Mistral-7B | $650 (18h) | 3 days (FREE) |

## API Reference

### TrainingConfig

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub dataset: DatasetConfig,
    pub training: TrainingParameters,
    pub evaluation: Option<EvaluationConfig>,
    pub logging: Option<LoggingConfig>,
}

impl TrainingConfig {
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self>;
    pub fn validate(&self) -> Result<()>;
    pub fn device(&self) -> String;
    pub fn is_lora_enabled(&self) -> bool;
}
```

### Trainer

```rust
pub struct Trainer {
    config: TrainingConfig,
    model: Box<dyn TrainableModel>,
    dataset: Box<dyn Dataset>,
}

impl Trainer {
    pub fn new(config: TrainingConfig) -> Result<Self>;
    pub fn train(&mut self) -> Result<TrainingResult>;
    pub fn evaluate(&self) -> Result<f64>;
    pub fn save_checkpoint<P: AsRef<Path>>(&self, path: P) -> Result<()>;
}
```

### Dataset Implementations

```rust
// For zen-agentic-dataset
let dataset = ZenAgenticDataset::new(
    "/path/to/zen-agentic-dataset-private",
    4096, // max_seq_length
    device
)?;

// For zen-identity-dataset  
let dataset = ZenIdentityDataset::new(
    "/path/to/zen-identity-dataset",
    2048, // max_seq_length
    device
)?;
```

## Configuration Templates

### Create Configuration Templates

```bash
# Create Qwen template
hanzo-train template --output qwen4b.yaml --template qwen

# Create Llama template  
hanzo-train template --output llama8b.yaml --template llama

# Create identity fine-tuning template
hanzo-train template --output identity.yaml --template identity
```

### Example Configuration

```yaml
model:
  name: "qwen3-4b"
  architecture: "qwen"
  checkpoint: "Qwen/Qwen3-4B-Instruct"
  max_seq_length: 4096

dataset:
  name: "zen-agentic"
  path: "/Users/z/work/zen/zen-agentic-dataset-private"
  format: "jsonl"
  preprocessing:
    tokenizer: "Qwen/Qwen3-4B-Instruct"
    add_eos: true
    truncation: true

training:
  batch_size: 4
  gradient_accumulation_steps: 2
  learning_rate: 2e-4
  epochs: 2
  
  lora:
    enabled: true
    r: 64
    alpha: 128
    target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
    
  mixed_precision: "bf16"
  device: "cuda:0"

logging:
  wandb:
    enabled: true
    project: "hanzo-training"
    name: "qwen3-4b-zen-agentic"
```

## Integration with Python zen-trainer

Hanzo Training can work alongside the Python zen-trainer:

```python
# Use zen-trainer for initial setup
from zen_trainer import ZenTrainer

# Train with zen-trainer (Python)
trainer = ZenTrainer(
    model_key="qwen3-4b",
    dataset_path="zen-agentic-dataset",
    output_dir="./output/python-trained"
)
trainer.train()
```

```bash
# Continue with hanzo-training (Rust) for fine-tuning
hanzo-train --config identity.yaml \
  --checkpoint ./output/python-trained \
  --output ./output/identity-tuned
```

## Examples

### Basic Training

```rust
use hanzo_training::{TrainingConfig, Trainer};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = TrainingConfig::from_file("config.yaml")?;
    let mut trainer = Trainer::new(config)?;
    
    let result = trainer.train()?;
    println!("Training completed: final_loss={:.6}", result.final_loss);
    
    trainer.save_checkpoint("./output/final_model")?;
    Ok(())
}
```

### Custom Dataset

```rust
use hanzo_training::{Dataset, TrainingConfig, Trainer, TrainingSample};

struct CustomDataset {
    // Your dataset implementation
}

impl Dataset for CustomDataset {
    fn len(&self) -> usize { /* ... */ }
    fn get_item(&self, index: usize) -> Result<TrainingSample> { /* ... */ }
}
```

### Multi-Stage Training Pipeline

```bash
#!/bin/bash
# Complete training pipeline

echo "🚀 Starting Zen model training pipeline"

# Stage 1: Foundation training on agentic data
echo "📚 Stage 1: Foundation training"
hanzo-train --config configs/zen-agentic-qwen4b.yaml \
  --output ./models/foundation

# Stage 2: Identity fine-tuning  
echo "🎭 Stage 2: Identity fine-tuning"
hanzo-train --config configs/zen-identity.yaml \
  --checkpoint ./models/foundation \
  --output ./models/identity

# Stage 3: Evaluation
echo "📊 Stage 3: Evaluation"
hanzo-eval all --model ./models/identity \
  --output ./results

echo "✅ Training pipeline completed!"
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Related Projects

- [Hanzo ML](https://crates.io/crates/hanzo-ml) - Core tensor operations
- [Hanzo NN](https://crates.io/crates/hanzo-nn) - Neural network layers
- [Hanzo Transformers](https://crates.io/crates/hanzo-transformers) - Transformer models
- [Zen Trainer](https://github.com/zenlm/zen-trainer) - Python training framework
- [Zen Agentic Dataset](https://huggingface.co/datasets/hanzoai/zen-agentic-dataset) - Training data

## License

This project is licensed under either of:

- BSD 3-Clause License ([LICENSE-BSD](LICENSE-BSD))
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))

at your option.

---

**Maintainer:** [z@hanzo.ai](mailto:z@hanzo.ai)