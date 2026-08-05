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

/// Refuse, once, for every entry point on the unwired path.
///
/// [`ModelWrapper`] holds no weights. It is not connected to
/// `hanzo-transformers` (which has the model implementations), nor to
/// `hanzo_ml::Tensor::backward` (which is a real reverse-mode autograd), nor to
/// `hanzo_nn::{SGD, AdamW}` (which are real optimizers). A forward pass that
/// returns a constant and an optimizer step that updates nothing still produce a
/// falling-looking loss log, a saved checkpoint and an exit code of zero — a
/// training run that reports success about a model it never touched. Refusing is
/// the loud failure that silence was hiding.
///
/// The connected path in this crate is [`crate::grpo`], which trains a real
/// [`crate::grpo::Policy`] through `hanzo_nn::AdamW::backward_step`.
pub(crate) fn unwired<T>(what: &str) -> Result<T> {
    anyhow::bail!(
        "hanzo-training: {what} is not connected to a model. ModelWrapper carries no weights, \
         no gradients and no optimizer state, so it cannot train or measure anything. Use \
         hanzo_training::grpo (a real autograd loop over hanzo_nn::AdamW), or build directly on \
         hanzo-transformers + hanzo-nn."
    )
}

/// Wrapper for Hanzo ML models.
///
/// Construction refuses — see [`unwired`]. The type is kept so the shape of what
/// a wired implementation must provide stays visible: weights on `device`, a
/// differentiable [`TrainableModel::forward`], and parameters the optimizer can
/// reach.
pub struct ModelWrapper {
    model_type: String,
    device: Device,
}

impl ModelWrapper {
    pub fn new(config: &ModelConfig, device: Device) -> Result<Self> {
        let _ = Self {
            model_type: config.architecture.clone(),
            device,
        };
        unwired("Trainer")
    }
}

impl TrainableModel for ModelWrapper {
    fn forward(&self, _input: &Tensor) -> Result<Tensor> {
        unwired("the forward pass")
    }

    fn backward(&mut self, _loss: &Tensor) -> Result<()> {
        unwired("the backward pass")
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // No weights exist to hand an optimizer; every path that would reach
        // here is already refused at construction.
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
