//! Inference pipeline management

use crate::InferenceModel;
use anyhow::Result;
use hanzo_ml_core::Tensor;

pub struct Pipeline {
    models: Vec<Box<dyn InferenceModel>>,
}

impl Pipeline {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
        }
    }

    pub fn add_model(&mut self, model: Box<dyn InferenceModel>) {
        self.models.push(model);
    }

    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut output = input.clone();
        for model in &self.models {
            output = model.forward(&output)?;
        }
        Ok(output)
    }
}