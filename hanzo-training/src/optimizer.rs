//! Optimizer implementations for training

use crate::{
    config::{OptimizerType, SchedulerType, TrainingParameters},
    Result,
};
use hanzo_ml::Tensor;

/// Optimizer configuration
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f64,
    pub weight_decay: f64,
    pub scheduler: Option<SchedulerType>,
    pub warmup_steps: Option<usize>,
}

impl OptimizerConfig {
    pub fn from_training_params(params: &TrainingParameters) -> Self {
        Self {
            optimizer_type: params.optimizer.clone(),
            learning_rate: params.learning_rate,
            weight_decay: params.weight_decay.unwrap_or(0.0),
            scheduler: params.scheduler.clone(),
            warmup_steps: params.warmup_steps,
        }
    }
}

/// Optimizer trait
pub trait Optimizer: Send + Sync {
    fn step(&mut self, parameters: Vec<&Tensor>) -> Result<()>;
    fn zero_grad(&mut self, parameters: Vec<&Tensor>) -> Result<()>;
    fn get_learning_rate(&self) -> f64;
    fn set_learning_rate(&mut self, lr: f64);
}

/// Optimizer wrapper
///
/// Carries the learning rate the trait can report and set, and nothing else. It held a
/// step count and a warmup/cosine schedule that advanced on every [`Optimizer::step`] —
/// but a step here moves no weights, so the schedule only ever produced a falling
/// learning-rate log next to parameters standing still. [`SchedulerType`] is still
/// parsed and validated config; a wired optimizer implements it against real steps.
pub struct OptimizerWrapper {
    current_lr: f64,
}

impl OptimizerWrapper {
    pub fn new(config: OptimizerConfig) -> Result<Self> {
        Ok(Self {
            current_lr: config.learning_rate,
        })
    }
}

impl Optimizer for OptimizerWrapper {
    /// Refuses: this optimizer has no gradients to apply.
    ///
    /// `hanzo_nn::{SGD, AdamW}` are the real optimizers — they take
    /// `hanzo_ml::backprop::GradStore` from `Tensor::backward()` and update
    /// `Var`s. This wrapper only ever received `Vec<&Tensor>`, which carries no
    /// gradient, so a "step" here could advance the schedule and the step count
    /// while leaving every weight exactly where it was.
    fn step(&mut self, _parameters: Vec<&Tensor>) -> Result<()> {
        crate::model::unwired("the optimizer step")
    }

    fn zero_grad(&mut self, _parameters: Vec<&Tensor>) -> Result<()> {
        crate::model::unwired("zeroing gradients")
    }

    fn get_learning_rate(&self) -> f64 {
        self.current_lr
    }

    fn set_learning_rate(&mut self, lr: f64) {
        self.current_lr = lr;
    }
}
