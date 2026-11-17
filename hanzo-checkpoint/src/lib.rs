pub mod adacheck;
pub mod cas;
pub mod manifest;
pub mod strategy;

pub use adacheck::AdaCheckController;
pub use cas::ContentAddressableStore;
pub use strategy::CheckpointStrategy;

#[derive(thiserror::Error, Debug)]
pub enum CheckpointError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Tensor error: {0}")]
    Tensor(#[from] candle_core::Error),
    #[error("Hash mismatch during restoration")]
    HashMismatch,
}

pub type Result<T> = std::result::Result<T, CheckpointError>;
