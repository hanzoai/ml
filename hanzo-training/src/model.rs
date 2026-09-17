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
/// reach. The fields are documentary — every entry point refuses before any of
/// them is read — hence `allow(dead_code)`.
#[allow(dead_code)]
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

    /// Refuses: there are no weights to serialize. It previously created the
    /// directory and wrote a `model_config.json` (model type + device) — a
    /// checkpoint on disk with exit status zero for a model that never trained.
    /// A wired model writes safetensors of real `Var`s here.
    fn save(&self, _path: &std::path::Path) -> Result<()> {
        unwired("saving a checkpoint")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_refuses() {
        let config = ModelConfig {
            name: "x".to_string(),
            architecture: "transformer".to_string(),
            checkpoint: None,
            max_seq_length: 8,
            vocab_size: None,
            hidden_size: None,
            num_layers: None,
            num_heads: None,
            custom_config: None,
        };
        let err = ModelWrapper::new(&config, Device::Cpu)
            .err()
            .expect("ModelWrapper::new must refuse: it holds no weights");
        assert!(err.to_string().contains("not connected to a model"));
    }

    #[test]
    fn save_refuses_and_writes_nothing() {
        // new() refuses, so build the wrapper directly to reach save(). A
        // weightless model must not leave a checkpoint on disk.
        let wrapper = ModelWrapper {
            model_type: "test".to_string(),
            device: Device::Cpu,
        };
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("checkpoint");
        let err = wrapper
            .save(&dir)
            .expect_err("save must refuse for a weightless model");
        assert!(err.to_string().contains("not connected to a model"));
        assert!(
            !dir.exists(),
            "save must not create the checkpoint directory"
        );
    }
}
