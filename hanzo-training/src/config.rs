//! Configuration management for training pipeline

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Main training configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub dataset: DatasetConfig,
    pub training: TrainingParameters,
    pub evaluation: Option<EvaluationConfig>,
    pub logging: Option<LoggingConfig>,
}

/// Model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub architecture: String,
    pub checkpoint: Option<String>,
    pub max_seq_length: usize,
    pub vocab_size: Option<usize>,
    pub hidden_size: Option<usize>,
    pub num_layers: Option<usize>,
    pub num_heads: Option<usize>,
    pub custom_config: Option<HashMap<String, serde_json::Value>>,
}

/// Dataset configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetConfig {
    pub name: String,
    pub path: String,
    pub format: DatasetFormat,
    pub train_split: Option<String>,
    pub validation_split: Option<String>,
    pub max_seq_length: usize,
    pub preprocessing: Option<PreprocessingConfig>,
    pub cache_dir: Option<String>,
}

/// Dataset format options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DatasetFormat {
    Jsonl,
    Json,
    Parquet,
    Csv,
    HuggingFace,
}

/// Preprocessing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreprocessingConfig {
    pub tokenizer: String,
    pub add_eos: bool,
    pub add_bos: bool,
    pub truncation: bool,
    pub padding: Option<PaddingStrategy>,
    pub special_tokens: Option<HashMap<String, String>>,
}

/// Padding strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PaddingStrategy {
    Left,
    Right,
    None,
}

/// Training hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingParameters {
    // Core parameters
    pub batch_size: usize,
    pub gradient_accumulation_steps: Option<usize>,
    pub learning_rate: f64,
    pub weight_decay: Option<f64>,
    pub warmup_steps: Option<usize>,
    pub max_steps: Option<usize>,
    pub epochs: Option<usize>,

    // Optimization
    pub optimizer: OptimizerType,
    pub scheduler: Option<SchedulerType>,
    pub gradient_clipping: Option<f64>,

    // LoRA configuration
    pub lora: Option<LoRAConfig>,

    // Checkpointing
    pub save_steps: Option<usize>,
    pub eval_steps: Option<usize>,
    pub logging_steps: Option<usize>,

    // Hardware
    pub mixed_precision: Option<MixedPrecision>,
    pub device: Option<String>,
    pub multi_gpu: Option<bool>,
    pub compile: Option<bool>,
}

/// Optimizer types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum OptimizerType {
    Adam,
    AdamW,
    SGD,
    RMSprop,
}

/// Learning rate scheduler types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SchedulerType {
    Linear,
    Cosine,
    CosineWithRestarts,
    Polynomial,
    Constant,
    ConstantWithWarmup,
}

/// LoRA configuration for parameter-efficient fine-tuning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub enabled: bool,
    pub r: usize,
    pub alpha: f64,
    pub dropout: f64,
    pub target_modules: Vec<String>,
    pub bias: Option<LoRABias>,
}

/// LoRA bias configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LoRABias {
    None,
    All,
    LoraOnly,
}

/// Mixed precision training options
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MixedPrecision {
    Fp16,
    Bf16,
    Fp32,
}

/// Evaluation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationConfig {
    pub benchmarks: Vec<String>,
    pub metrics: Vec<String>,
    pub eval_dataset: Option<String>,
    pub output_dir: Option<String>,
}

/// Logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub wandb: Option<WandbConfig>,
    pub tensorboard: Option<bool>,
    pub console_level: Option<String>,
    pub file_logging: Option<FileLoggingConfig>,
}

/// Weights & Biases configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WandbConfig {
    pub enabled: bool,
    pub project: String,
    pub name: Option<String>,
    pub tags: Option<Vec<String>>,
    pub notes: Option<String>,
}

/// File logging configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileLoggingConfig {
    pub enabled: bool,
    pub path: String,
    pub level: Option<String>,
}

impl TrainingConfig {
    /// Load configuration from a file (YAML or JSON)
    pub fn from_file<P: AsRef<Path>>(path: P) -> crate::Result<Self> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config file '{}': {}", path.display(), e))?;

        match path.extension().and_then(|ext| ext.to_str()) {
            Some("yaml") | Some("yml") => Self::from_yaml(&content),
            Some("json") => Self::from_json(&content),
            Some("toml") => Self::from_toml(&content),
            _ => Err("Unsupported config file format. Use .yaml, .json, or .toml".into()),
        }
    }

    /// Parse configuration from YAML string
    pub fn from_yaml(content: &str) -> crate::Result<Self> {
        serde_yaml::from_str(content)
            .map_err(|e| format!("Failed to parse YAML config: {}", e).into())
    }

    /// Parse configuration from JSON string
    pub fn from_json(content: &str) -> crate::Result<Self> {
        serde_json::from_str(content)
            .map_err(|e| format!("Failed to parse JSON config: {}", e).into())
    }

    /// Parse configuration from TOML string
    pub fn from_toml(content: &str) -> crate::Result<Self> {
        toml::from_str(content)
            .map_err(|e| format!("Failed to parse TOML config: {}", e).into())
    }

    /// Validate the configuration
    pub fn validate(&self) -> crate::Result<()> {
        // Model validation
        if self.model.name.is_empty() {
            return Err("Model name cannot be empty".into());
        }

        if self.model.max_seq_length == 0 {
            return Err("Model max_seq_length must be greater than 0".into());
        }

        // Dataset validation
        if self.dataset.path.is_empty() {
            return Err("Dataset path cannot be empty".into());
        }

        if self.dataset.max_seq_length == 0 {
            return Err("Dataset max_seq_length must be greater than 0".into());
        }

        // Training validation
        if self.training.batch_size == 0 {
            return Err("Batch size must be greater than 0".into());
        }

        if self.training.learning_rate <= 0.0 {
            return Err("Learning rate must be positive".into());
        }

        // Must have either epochs or max_steps
        if self.training.epochs.is_none() && self.training.max_steps.is_none() {
            return Err("Must specify either epochs or max_steps".into());
        }

        Ok(())
    }

    /// Get the device string for computation
    pub fn device(&self) -> String {
        self.training.device.clone().unwrap_or_else(|| {
            if hanzo_ml::Device::cuda_if_available(0).is_ok() {
                "cuda:0".to_string()
            } else {
                "cpu".to_string()
            }
        })
    }

    /// Check if LoRA is enabled
    pub fn is_lora_enabled(&self) -> bool {
        self.training
            .lora
            .as_ref()
            .map(|lora| lora.enabled)
            .unwrap_or(false)
    }

    /// Get the effective batch size (including gradient accumulation)
    pub fn effective_batch_size(&self) -> usize {
        self.training.batch_size
            * self
                .training
                .gradient_accumulation_steps
                .unwrap_or(1)
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            model: ModelConfig {
                name: "default".to_string(),
                architecture: "transformer".to_string(),
                checkpoint: None,
                max_seq_length: 2048,
                vocab_size: None,
                hidden_size: None,
                num_layers: None,
                num_heads: None,
                custom_config: None,
            },
            dataset: DatasetConfig {
                name: "default".to_string(),
                path: "./dataset".to_string(),
                format: DatasetFormat::Jsonl,
                train_split: Some("train".to_string()),
                validation_split: Some("validation".to_string()),
                max_seq_length: 2048,
                preprocessing: None,
                cache_dir: None,
            },
            training: TrainingParameters {
                batch_size: 4,
                gradient_accumulation_steps: Some(1),
                learning_rate: 2e-4,
                weight_decay: Some(0.01),
                warmup_steps: Some(100),
                max_steps: None,
                epochs: Some(3),
                optimizer: OptimizerType::AdamW,
                scheduler: Some(SchedulerType::Linear),
                gradient_clipping: Some(1.0),
                lora: None,
                save_steps: Some(1000),
                eval_steps: Some(500),
                logging_steps: Some(10),
                mixed_precision: Some(MixedPrecision::Bf16),
                device: None,
                multi_gpu: Some(false),
                compile: Some(false),
            },
            evaluation: None,
            logging: None,
        }
    }
}