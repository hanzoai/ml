//! Optimizers and learning-rate schedules.
//!
//! We re-export `candle_nn::AdamW` rather than reimplementing it —
//! candle's AdamW already matches `torch.optim.AdamW`'s update rule
//! (Loshchilov–Hutter decoupled weight decay with bias correction).
//! [`TrainableAdamW`] is a thin wrapper that owns the schedule and
//! collects the LoRA `Var`s.

pub mod adamw;
pub mod schedule;
