//! Hanzo ML Inference Layer
//! 
//! Provides high-level inference capabilities for use in hanzo-engine,
//! hanzo-node, and other components that need AI inference.

use anyhow::Result;
use hanzo_ml::{Device, Tensor};
use std::path::Path;

pub mod models;
pub mod pipeline;
pub mod quantization;
pub mod distributed;

pub use models::*;
pub use pipeline::*;
pub use quantization::*;
pub use distributed::*;

/// High-level inference engine
pub struct InferenceEngine {
    device: Device,
    models: Vec<Box<dyn InferenceModel>>,
}

impl InferenceEngine {
    /// Create new inference engine
    pub fn new(device: Device) -> Self {
        Self {
            device,
            models: Vec::new(),
        }
    }

    /// Load model from path
    pub async fn load_model<P: AsRef<Path>>(&mut self, path: P) -> Result<()> {
        // Implementation will be added based on hanzo-engine integration
        Ok(())
    }

    /// Run inference
    pub async fn infer(&self, input: &Tensor) -> Result<Tensor> {
        // Implementation will be added
        Ok(input.clone())
    }
}

/// Trait for inference models
pub trait InferenceModel: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn device(&self) -> &Device;
    fn model_type(&self) -> ModelType;
}

/// Supported model types
#[derive(Debug, Clone)]
pub enum ModelType {
    LLM,
    Vision,
    Audio,
    Multimodal,
    Embedding,
}