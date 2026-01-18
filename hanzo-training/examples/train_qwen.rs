//! Example: Train Qwen3-4B on zen-agentic-dataset

use anyhow::Result;
use hanzo_training::{TrainingConfig, Trainer, init_logging};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    init_logging()?;
    
    info!("🚀 Starting Qwen3-4B training on zen-agentic-dataset");

    // Load configuration
    let config = TrainingConfig::from_file("configs/zen-agentic-qwen4b.yaml")?;
    
    // Validate configuration
    config.validate()?;
    info!("✅ Configuration validated");

    // Create trainer
    let mut trainer = Trainer::new(config)?;
    info!("✅ Trainer created");

    // Start training
    info!("🎯 Starting training...");
    let result = trainer.train()?;

    // Print results
    info!("🎉 Training completed successfully!");
    info!("Final loss: {:.6}", result.final_loss);
    info!("Total steps: {}", result.total_steps);
    info!("Training time: {:.2}s", result.training_time);

    // Save final model
    trainer.save_checkpoint(std::path::Path::new("./output/qwen3-4b-zen-agentic"))?;
    info!("💾 Model saved to ./output/qwen3-4b-zen-agentic");

    Ok(())
}