//! Hanzo Training - ML Training Framework
//!
//! A high-performance training framework for machine learning models.

pub mod config;
pub mod dataset;
pub mod evaluation;
pub mod logging;
pub mod metrics;
pub mod model;
pub mod optimizer;
pub mod trainer;
pub mod utils;

// Re-exports
pub use config::{
    DatasetConfig, EvaluationConfig, LoggingConfig, ModelConfig, TrainingConfig, TrainingParameters,
};
pub use dataset::{Dataset, JsonlDataset, TrainingSample, ZenAgenticDataset, ZenIdentityDataset};
pub use evaluation::{AccuracyBenchmark, Benchmark, BenchmarkRunner, PerplexityBenchmark};
pub use logging::{ConsoleLogger, Logger, MultiLogger};
pub use metrics::{EvaluationMetrics, MetricsCollector, TrainingMetrics};
pub use model::TrainableModel;
pub use optimizer::OptimizerConfig;
pub use trainer::Trainer;

// Use anyhow for error handling
pub type Result<T> = anyhow::Result<T>;

/// Initialize logging (tracing subscriber)
pub fn init_logging() -> Result<()> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    tracing_subscriber::registry()
        .with(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "hanzo_training=info".into()),
        )
        .with(tracing_subscriber::fmt::layer())
        .try_init()
        .map_err(|e| anyhow::anyhow!("Failed to initialize logging: {}", e))?;

    Ok(())
}

/// Initialize logging from config
pub fn init_logging_from_config(config: Option<&LoggingConfig>) -> Result<MultiLogger> {
    if let Some(log_config) = config {
        MultiLogger::from_config(log_config)
    } else {
        let mut logger = MultiLogger::new();
        logger.add_logger(Box::new(ConsoleLogger::new(Some("info".to_string()))));
        Ok(logger)
    }
}

// Training error types
#[derive(Debug, thiserror::Error)]
pub enum TrainingError {
    #[error("Model error: {0}")]
    Model(String),

    #[error("Dataset error: {0}")]
    Dataset(String),

    #[error("Training error: {0}")]
    Training(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}
