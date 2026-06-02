//! Metrics returned by a single GRPO step. Mirrors the subset of
//! `self._metrics` the reference trainer logs (`reward`, `reward_std`,
//! completion lengths, loss).

/// Metrics for one [`GrpoTrainer::step`](crate::grpo::GrpoTrainer::step).
#[derive(Debug, Clone)]
pub struct GrpoMetrics {
    /// Total policy-gradient + KL loss for the step (the value optimized).
    pub loss: f32,
    /// Policy-gradient component of the loss.
    pub pg_loss: f32,
    /// KL component of the loss (0.0 when KL is disabled).
    pub kl_loss: f32,
    /// Mean reward across all completions in the step (mean of group means).
    pub mean_reward: f32,
    /// Mean within-group reward std across groups.
    pub reward_std: f32,
    /// Mean completion length (tokens) across all completions.
    pub mean_completion_len: f32,
    /// Number of prompts (groups) processed this step.
    pub num_groups: usize,
    /// Number of completions processed this step (`num_groups * group_size`).
    pub num_completions: usize,
}

impl GrpoMetrics {
    /// Whether all loss values are finite (no NaN/Inf). Used by the smoke test.
    pub fn is_finite(&self) -> bool {
        self.loss.is_finite() && self.pg_loss.is_finite() && self.kl_loss.is_finite()
    }
}
