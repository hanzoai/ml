//! Learning-rate schedules.
//!
//! All schedules expose `lr_at(step, base_lr) -> f64`. The base LR is
//! taken from the optimizer config so the schedule does not have to
//! capture it; this lets you change LR mid-run without rebuilding.

use std::f64::consts::PI;

/// LR schedule. Constructed via the free functions [`cosine_with_warmup`]
/// and [`linear_with_warmup`].
#[derive(Clone, Debug)]
pub enum LrSchedule {
    /// Constant LR throughout.
    Constant,
    /// Linear ramp from 0 to base_lr over `warmup`, then constant.
    LinearWarmup { warmup: usize },
    /// Linear ramp 0→base_lr over `warmup`, then linear decay to 0
    /// over the remaining `total - warmup` steps.
    LinearWarmupDecay { warmup: usize, total: usize },
    /// Linear ramp, then half-cosine decay to `min_lr_ratio * base_lr`.
    /// Matches HuggingFace `get_cosine_schedule_with_warmup`.
    CosineWarmup {
        warmup: usize,
        total: usize,
        min_lr_ratio: f64,
    },
}

impl LrSchedule {
    pub fn lr_at(&self, step: usize, base_lr: f64) -> f64 {
        match *self {
            LrSchedule::Constant => base_lr,
            LrSchedule::LinearWarmup { warmup } => {
                if step < warmup {
                    base_lr * (step as f64 / warmup.max(1) as f64)
                } else {
                    base_lr
                }
            }
            LrSchedule::LinearWarmupDecay { warmup, total } => {
                if step < warmup {
                    base_lr * (step as f64 / warmup.max(1) as f64)
                } else if step >= total {
                    0.0
                } else {
                    let progress = (step - warmup) as f64 / (total - warmup).max(1) as f64;
                    base_lr * (1.0 - progress)
                }
            }
            LrSchedule::CosineWarmup {
                warmup,
                total,
                min_lr_ratio,
            } => {
                if step < warmup {
                    base_lr * (step as f64 / warmup.max(1) as f64)
                } else if step >= total {
                    base_lr * min_lr_ratio
                } else {
                    let progress = (step - warmup) as f64 / (total - warmup).max(1) as f64;
                    let cos = 0.5 * (1.0 + (PI * progress).cos());
                    base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cos)
                }
            }
        }
    }
}

/// HuggingFace-compatible cosine schedule with linear warmup.
pub fn cosine_with_warmup(warmup: usize, total: usize, min_lr_ratio: f64) -> LrSchedule {
    LrSchedule::CosineWarmup {
        warmup,
        total,
        min_lr_ratio,
    }
}

/// Linear warmup then linear decay to zero (default Trainer schedule).
pub fn linear_with_warmup(warmup: usize, total: usize) -> LrSchedule {
    LrSchedule::LinearWarmupDecay { warmup, total }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_endpoints() {
        let s = cosine_with_warmup(10, 100, 0.0);
        // HF convention: step 0 -> 0.
        assert!(s.lr_at(0, 1.0).abs() < 1e-12);
        // At step == warmup, full LR.
        assert!((s.lr_at(10, 1.0) - 1.0).abs() < 1e-12);
        // At total step, hits min_lr_ratio.
        assert!(s.lr_at(100, 1.0).abs() < 1e-12);
    }

    #[test]
    fn linear_decay_zero_at_end() {
        let s = linear_with_warmup(0, 100);
        // No warmup -> step 0 already at full LR.
        assert!((s.lr_at(0, 1.0) - 1.0).abs() < 1e-12);
        assert!(s.lr_at(100, 1.0).abs() < 1e-12);
    }
}
