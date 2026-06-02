//! AdamW wrapper around `candle_nn::AdamW`.
//!
//! ### Why not re-implement?
//!
//! candle's AdamW already follows the Loshchilov–Hutter update:
//!
//! ```text
//! m_t = β1·m_{t-1} + (1-β1)·g
//! v_t = β2·v_{t-1} + (1-β2)·g²
//! m̂  = m_t / (1 - β1^t)
//! v̂  = v_t / (1 - β2^t)
//! θ_t = (1 - lr·wd) · θ_{t-1} - lr · m̂ / (√v̂ + ε)
//! ```
//!
//! which is bit-for-bit `torch.optim.AdamW` (with `amsgrad=False`,
//! `maximize=False`, `foreach=False`). The compatibility test in
//! `tests/adamw_matches_torch.rs` pins this against the closed-form
//! reference implementation it replicates from PyTorch's source.

use candle::Var;
use candle_nn::{AdamW as InnerAdamW, Optimizer, ParamsAdamW};

use super::schedule::LrSchedule;

/// Hyperparameters. Field names are PyTorch-equivalent; defaults match
/// `torch.optim.AdamW`'s defaults.
#[derive(Clone, Debug)]
pub struct TrainableAdamWConfig {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for TrainableAdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-4,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            // Standard QLoRA / LoRA recipes use 0.0 here; we follow.
            weight_decay: 0.0,
        }
    }
}

impl TrainableAdamWConfig {
    fn as_inner(&self) -> ParamsAdamW {
        ParamsAdamW {
            lr: self.lr,
            beta1: self.beta1,
            beta2: self.beta2,
            eps: self.eps,
            weight_decay: self.weight_decay,
        }
    }
}

/// AdamW + LR schedule + step counter. Delegates the update math to
/// `candle_nn::AdamW`.
pub struct TrainableAdamW {
    inner: InnerAdamW,
    schedule: LrSchedule,
    step: usize,
    base_lr: f64,
}

impl TrainableAdamW {
    pub fn new(vars: Vec<Var>, cfg: TrainableAdamWConfig, schedule: LrSchedule) -> crate::Result<Self> {
        let base_lr = cfg.lr;
        let inner = InnerAdamW::new(vars, cfg.as_inner())
            .map_err(|e| anyhow::anyhow!("AdamW init failed: {e}"))?;
        Ok(Self {
            inner,
            schedule,
            step: 0,
            base_lr,
        })
    }

    /// Run one optimizer step on a precomputed scalar loss tensor.
    /// Returns the LR that was used for this step.
    pub fn backward_step(&mut self, loss: &candle::Tensor) -> crate::Result<f64> {
        let lr = self.schedule.lr_at(self.step, self.base_lr);
        self.inner.set_learning_rate(lr);
        self.inner
            .backward_step(loss)
            .map_err(|e| anyhow::anyhow!("AdamW step failed: {e}"))?;
        self.step += 1;
        Ok(lr)
    }

    pub fn step_count(&self) -> usize {
        self.step
    }

    pub fn current_lr(&self) -> f64 {
        self.inner.learning_rate()
    }
}
