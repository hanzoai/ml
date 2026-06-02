//! Command-line training interface

use anyhow::Result;
use clap::{Parser, Subcommand};
use hanzo_training::{init_logging, Trainer, TrainingConfig};
use std::path::PathBuf;
use tracing::info;

#[derive(Parser)]
#[command(name = "hanzo-train")]
#[command(about = "Train Hanzo ML models on zen-agentic-dataset")]
#[command(version)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,

    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    /// Dataset path (overrides config)
    #[arg(short, long, value_name = "PATH")]
    dataset: Option<PathBuf>,

    /// Output directory (overrides config)
    #[arg(short, long, value_name = "DIR")]
    output: Option<PathBuf>,

    /// Device to use (overrides config)
    #[arg(long, value_name = "DEVICE")]
    device: Option<String>,

    /// Enable multi-GPU training
    #[arg(long)]
    multi_gpu: bool,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// Train a model
    Train {
        /// Configuration file
        #[arg(short, long, value_name = "FILE")]
        config: PathBuf,

        /// Override dataset path
        #[arg(short, long, value_name = "PATH")]
        dataset: Option<PathBuf>,

        /// Override output directory
        #[arg(short, long, value_name = "DIR")]
        output: Option<PathBuf>,

        /// Resume from checkpoint
        #[arg(long, value_name = "PATH")]
        resume: Option<PathBuf>,
    },

    /// Validate configuration
    Validate {
        /// Configuration file to validate
        #[arg(short, long, value_name = "FILE")]
        config: PathBuf,
    },

    /// Create a template configuration
    Template {
        /// Output path for template
        #[arg(short, long, value_name = "FILE")]
        output: PathBuf,

        /// Template type
        #[arg(short, long, default_value = "qwen")]
        template: String,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Initialize logging
    if cli.verbose {
        std::env::set_var("RUST_LOG", "hanzo_training=debug,info");
    }
    init_logging()?;

    match &cli.command {
        Some(Commands::Train {
            config,
            dataset,
            output,
            resume,
        }) => train_command(config, dataset, output, resume).await,
        Some(Commands::Validate { config }) => validate_command(config).await,
        Some(Commands::Template { output, template }) => template_command(output, template).await,
        None => {
            // Legacy mode - use global args
            if let Some(config) = cli.config {
                train_command(&config, &cli.dataset, &cli.output, &None).await
            } else {
                eprintln!("Error: Configuration file required");
                std::process::exit(1);
            }
        }
    }
}

async fn train_command(
    config_path: &PathBuf,
    dataset_override: &Option<PathBuf>,
    output_override: &Option<PathBuf>,
    resume_from: &Option<PathBuf>,
) -> Result<()> {
    info!("Loading configuration from {}", config_path.display());
    let mut config = TrainingConfig::from_file(config_path)?;

    // Apply overrides
    if let Some(dataset_path) = dataset_override {
        config.dataset.path = dataset_path.to_string_lossy().to_string();
        info!("Overriding dataset path: {}", config.dataset.path);
    }

    if let Some(_output_path) = output_override {
        // Store output path in logging config for now
        if config.logging.is_none() {
            config.logging = Some(hanzo_training::LoggingConfig {
                wandb: None,
                tensorboard: Some(true),
                console_level: Some("info".to_string()),
                file_logging: None,
            });
        }
    }

    // Validate configuration
    config.validate()?;
    info!("Configuration validated successfully");

    // Create trainer
    info!("Creating trainer...");
    let mut trainer = Trainer::new(config)?;

    // Resume from checkpoint if specified
    if let Some(checkpoint_path) = resume_from {
        info!("Resuming from checkpoint: {}", checkpoint_path.display());
        trainer.load_checkpoint(checkpoint_path)?;
    }

    // Start training
    info!("Starting training...");
    let result = trainer.train()?;

    info!("Training completed successfully!");
    info!("Final loss: {:.6}", result.final_loss);
    info!("Total steps: {}", result.total_steps);
    info!("Training time: {:.2}s", result.training_time);

    Ok(())
}

async fn validate_command(config_path: &PathBuf) -> Result<()> {
    info!("Validating configuration: {}", config_path.display());

    let config = TrainingConfig::from_file(config_path)?;
    config.validate()?;

    println!("✅ Configuration is valid");
    println!("📋 Summary:");
    println!(
        "  Model: {} ({})",
        config.model.name, config.model.architecture
    );
    println!(
        "  Dataset: {} ({})",
        config.dataset.name, config.dataset.path
    );
    println!("  Batch size: {}", config.training.batch_size);
    println!("  Learning rate: {}", config.training.learning_rate);
    println!("  Device: {}", config.device());

    if config.is_lora_enabled() {
        if let Some(lora) = &config.training.lora {
            println!("  LoRA: enabled (r={}, α={})", lora.r, lora.alpha);
        }
    } else {
        println!("  LoRA: disabled");
    }

    Ok(())
}

async fn template_command(output_path: &PathBuf, template: &str) -> Result<()> {
    info!("Creating {} template configuration", template);

    let config = match template {
        "qwen" => create_qwen_template(),
        "llama" => create_llama_template(),
        "mistral" => create_mistral_template(),
        "identity" => create_identity_template(),
        _ => {
            return Err(anyhow::anyhow!("Unknown template type: {}", template));
        }
    };

    let yaml_content = serde_yaml::to_string(&config)?;
    std::fs::write(output_path, yaml_content)?;

    println!("✅ Template created: {}", output_path.display());
    Ok(())
}

fn create_qwen_template() -> TrainingConfig {
    use hanzo_training::*;

    TrainingConfig {
        model: ModelConfig {
            name: "qwen3-4b".to_string(),
            architecture: "qwen".to_string(),
            checkpoint: Some("Qwen/Qwen3-4B-Instruct".to_string()),
            max_seq_length: 4096,
            vocab_size: Some(151936),
            hidden_size: Some(3584),
            num_layers: Some(32),
            num_heads: Some(28),
            custom_config: None,
        },
        dataset: DatasetConfig {
            name: "zen-agentic".to_string(),
            path: "/Users/z/work/zen/zen-agentic-dataset-private".to_string(),
            format: config::DatasetFormat::Jsonl,
            train_split: Some("train".to_string()),
            validation_split: Some("valid".to_string()),
            max_seq_length: 4096,
            preprocessing: Some(config::PreprocessingConfig {
                tokenizer: "Qwen/Qwen3-4B-Instruct".to_string(),
                add_eos: true,
                add_bos: false,
                truncation: true,
                padding: Some(config::PaddingStrategy::Right),
                special_tokens: None,
            }),
            cache_dir: None,
        },
        training: TrainingParameters {
            batch_size: 4,
            gradient_accumulation_steps: Some(2),
            learning_rate: 2e-4,
            weight_decay: Some(0.01),
            warmup_steps: Some(100),
            max_steps: Some(10000),
            epochs: Some(2),
            optimizer: config::OptimizerType::AdamW,
            scheduler: Some(config::SchedulerType::Cosine),
            gradient_clipping: Some(1.0),
            lora: Some(config::LoRAConfig {
                enabled: true,
                r: 64,
                alpha: 128.0,
                dropout: 0.1,
                target_modules: vec![
                    "q_proj".to_string(),
                    "v_proj".to_string(),
                    "k_proj".to_string(),
                    "o_proj".to_string(),
                ],
                bias: Some(config::LoRABias::None),
            }),
            save_steps: Some(1000),
            eval_steps: Some(500),
            logging_steps: Some(10),
            mixed_precision: Some(config::MixedPrecision::Bf16),
            device: Some("cuda:0".to_string()),
            multi_gpu: Some(false),
            compile: Some(true),
        },
        evaluation: Some(EvaluationConfig {
            benchmarks: vec!["perplexity".to_string(), "accuracy".to_string()],
            metrics: vec!["loss".to_string(), "accuracy".to_string()],
            eval_dataset: None,
            output_dir: Some("./eval_results".to_string()),
        }),
        logging: Some(LoggingConfig {
            wandb: Some(config::WandbConfig {
                enabled: true,
                project: "hanzo-training".to_string(),
                name: Some("qwen3-4b-zen-agentic".to_string()),
                tags: Some(vec!["qwen".to_string(), "agentic".to_string()]),
                notes: Some("Training on zen-agentic-dataset".to_string()),
            }),
            tensorboard: Some(true),
            console_level: Some("info".to_string()),
            file_logging: Some(config::FileLoggingConfig {
                enabled: true,
                path: "./logs/training.log".to_string(),
                level: Some("debug".to_string()),
            }),
        }),
    }
}

fn create_llama_template() -> TrainingConfig {
    let mut config = create_qwen_template();
    config.model.name = "llama3-8b".to_string();
    config.model.architecture = "llama".to_string();
    config.model.checkpoint = Some("meta-llama/Meta-Llama-3-8B-Instruct".to_string());
    config.model.vocab_size = Some(128256);
    config.model.hidden_size = Some(4096);
    config.model.num_layers = Some(32);
    config.model.num_heads = Some(32);

    if let Some(ref mut preprocessing) = config.dataset.preprocessing {
        preprocessing.tokenizer = "meta-llama/Meta-Llama-3-8B-Instruct".to_string();
    }

    config
}

fn create_mistral_template() -> TrainingConfig {
    let mut config = create_qwen_template();
    config.model.name = "mistral-7b".to_string();
    config.model.architecture = "mistral".to_string();
    config.model.checkpoint = Some("mistralai/Mistral-7B-Instruct-v0.2".to_string());
    config.model.vocab_size = Some(32000);
    config.model.hidden_size = Some(4096);
    config.model.num_layers = Some(32);
    config.model.num_heads = Some(32);

    if let Some(ref mut preprocessing) = config.dataset.preprocessing {
        preprocessing.tokenizer = "mistralai/Mistral-7B-Instruct-v0.2".to_string();
    }

    config
}

fn create_identity_template() -> TrainingConfig {
    let mut config = create_qwen_template();
    config.dataset.name = "zen-identity".to_string();
    config.dataset.path = "/Users/z/work/zen/zen-identity-dataset".to_string();
    config.training.learning_rate = 1e-4;
    config.training.epochs = Some(3);
    config.training.max_steps = Some(5000);

    if let Some(ref mut wandb) = config.logging.as_mut().and_then(|l| l.wandb.as_mut()) {
        wandb.name = Some("qwen3-4b-zen-identity".to_string());
        wandb.tags = Some(vec!["qwen".to_string(), "identity".to_string()]);
        wandb.notes = Some("Identity fine-tuning on zen-identity-dataset".to_string());
    }

    config
}
