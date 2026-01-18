//! Example: Train Llama 3.1 8B on zen-agentic-dataset

use anyhow::Result;
use hanzo_training::{TrainingConfig, Trainer, init_logging, utils};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    init_logging()?;
    
    info!("🚀 Starting Llama 3.1 8B training on zen-agentic-dataset");

    // Load configuration
    let config = TrainingConfig::from_file("configs/llama3-8b.yaml")?;
    
    // Validate configuration
    config.validate()?;
    info!("✅ Configuration validated");

    // Print memory and time estimates
    let estimated_memory = utils::estimate_memory_requirements(
        config.model.hidden_size.unwrap_or(4096) * config.model.num_layers.unwrap_or(32),
        config.training.batch_size,
        config.model.max_seq_length,
        2, // bf16
    );
    
    let estimated_time = utils::estimate_training_time(
        config.model.hidden_size.unwrap_or(4096) * config.model.num_layers.unwrap_or(32),
        10000, // estimated dataset size
        config.effective_batch_size(),
        config.training.epochs.unwrap_or(1),
        1000, // tokens per second estimate
    );

    info!("📊 Estimated memory usage: {}", utils::format_bytes(estimated_memory));
    info!("⏱️  Estimated training time: {}", utils::format_duration(estimated_time));

    // Create output directory structure
    utils::create_output_dirs("./output/llama3-8b-zen-agentic")?;

    // Create trainer
    let mut trainer = Trainer::new(config)?;
    info!("✅ Trainer created");

    // Start training with progress tracking
    info!("🎯 Starting training...");
    let progress = utils::ProgressBar::new(10000); // Estimated steps
    progress.set_message("Training model...");
    
    let result = trainer.train()?;
    progress.finish_with_message("Training completed!");

    // Print results
    info!("🎉 Training completed successfully!");
    info!("Final loss: {:.6}", result.final_loss);
    info!("Total steps: {}", result.total_steps);
    info!("Training time: {}", utils::format_duration(result.training_time));

    // Save final model
    trainer.save_checkpoint(std::path::Path::new("./output/llama3-8b-zen-agentic"))?;
    info!("💾 Model saved to ./output/llama3-8b-zen-agentic");

    Ok(())
}