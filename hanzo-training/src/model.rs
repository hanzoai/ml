//! Model wrappers for training

use crate::{config::ModelConfig, Result};
use hanzo_ml::{Device, Tensor};

/// Trait for trainable models
pub trait TrainableModel: Send + Sync {
    fn forward(&self, input: &Tensor) -> Result<Tensor>;
    fn backward(&mut self, loss: &Tensor) -> Result<()>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn save(&self, path: &std::path::Path) -> Result<()>;
}

/// Wrapper for Hanzo ML models
pub struct ModelWrapper {
    model_type: String,
    device: Device,
    // This would contain the actual model implementation
    // For now, keeping it simple
}

impl ModelWrapper {
    pub fn new(config: &ModelConfig, device: Device) -> Result<Self> {
        Ok(Self {
            model_type: config.architecture.clone(),
            device,
        })
    }
}

impl TrainableModel for ModelWrapper {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        // Simplified forward pass - in practice, this would call the actual model
        // For demonstration, just return a scalar loss tensor
        let loss = Tensor::new(&[1.0f32], &self.device)?;
        Ok(loss)
    }

    fn backward(&mut self, _loss: &Tensor) -> Result<()> {
        // Simplified backward pass - in practice, this would compute gradients
        Ok(())
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // Return model parameters for optimizer
        vec![]
    }

    fn save(&self, path: &std::path::Path) -> Result<()> {
        std::fs::create_dir_all(path)?;

        // Save model metadata
        let metadata = serde_json::json!({
            "model_type": self.model_type,
            "device": format!("{:?}", self.device),
        });

        std::fs::write(
            path.join("model_config.json"),
            serde_json::to_string_pretty(&metadata)?,
        )?;

        Ok(())
    }
}
