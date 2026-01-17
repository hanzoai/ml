//! # Hanzo Training
//!
//! Native Rust training framework for Hanzo ML models using the zen-agentic-dataset.

pub mod config;
pub mod dataset;
pub mod evaluation;
pub mod logging;
pub mod model;
pub mod optimizer;
pub mod trainer;
pub mod utils;

// Re-export main types
pub use config::{TrainingConfig, ModelConfig, DatasetConfig, TrainingParameters};
pub use dataset::{Dataset, TrainingSample, ZenAgenticDataset, ZenIdentityDataset};
pub use evaluation::{EvaluationConfig, EvaluationResult, Benchmark};
pub use logging::{LoggingConfig, Logger};
pub use model::{TrainableModel, ModelWrapper};
pub use optimizer::{OptimizerConfig, OptimizerWrapper};
pub use trainer::{Trainer, TrainingResult};

/// Result type used throughout the crate
pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

/// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Initialize logging for the training framework
pub fn init_logging() -> Result<()> {
    tracing_subscriber::fmt()
        .init();
    Ok(())
}