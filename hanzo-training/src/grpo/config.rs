//! Configuration for trainer-free GRPO (Group Relative Policy Optimization).
//!
//! Ported from the algorithm in `zooai-gym` GRPO trainer (`trainer.py`,
//! `args.py`). Only the knobs that matter for the *algorithm* are kept; the
//! distributed / sequence-parallel machinery from the reference is intentionally
//! dropped (it is orthogonal to the math and out of scope for a single-process,
//! pure-`hanzo-ml`-autograd trainer).

use serde::{Deserialize, Serialize};

/// How completions are sampled from the policy during a GRPO step.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub enum SamplingMode {
    /// Always take the arg-max token (deterministic). Useful for debugging but
    /// produces zero intra-group variance, which makes advantages degenerate.
    Greedy,
    /// Temperature sampling over the soft-maxed logits.
    #[default]
    Temperature,
}

/// Configuration for a [`GrpoTrainer`](crate::grpo::GrpoTrainer).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrpoConfig {
    /// Number of completions sampled per prompt (the "group"). Must be >= 2 so
    /// that a group-relative mean/std is meaningful. Mirrors TRL/axolotl's
    /// `num_generations`.
    pub group_size: usize,

    /// Maximum number of tokens to generate per completion.
    pub max_completion_len: usize,

    /// Sampling strategy used to roll out completions.
    pub sampling: SamplingMode,

    /// Sampling temperature (only used when `sampling == Temperature`).
    pub temperature: f64,

    /// Whether to divide the group-centered reward by `std + eps` to form the
    /// advantage. Mirrors `args.scale_rewards` in the reference trainer. When
    /// `false`, the advantage is just the mean-centered reward.
    pub scale_rewards: bool,

    /// Numerical epsilon added to the per-group std when scaling rewards.
    /// Reference uses `1e-4`.
    pub advantage_eps: f64,

    /// Optional KL penalty coefficient against a frozen reference policy
    /// (`beta` in the reference). `None` (or `0.0`) disables the KL term and the
    /// reference policy is never invoked.
    pub kl_beta: Option<f64>,

    /// AdamW learning rate.
    pub learning_rate: f64,

    /// AdamW weight decay.
    pub weight_decay: f64,

    /// RNG seed for the sampler.
    pub seed: u64,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: 8,
            max_completion_len: 64,
            sampling: SamplingMode::Temperature,
            temperature: 1.0,
            scale_rewards: true,
            advantage_eps: 1e-4,
            kl_beta: None,
            learning_rate: 1e-3,
            weight_decay: 0.0,
            seed: 0,
        }
    }
}

impl GrpoConfig {
    /// Validate the configuration, returning a descriptive error on misuse.
    pub fn validate(&self) -> crate::Result<()> {
        if self.group_size < 2 {
            anyhow::bail!(
                "GrpoConfig.group_size must be >= 2 (got {}); group-relative \
                 advantages require at least two completions per prompt",
                self.group_size
            );
        }
        if self.max_completion_len == 0 {
            anyhow::bail!("GrpoConfig.max_completion_len must be >= 1");
        }
        if matches!(self.sampling, SamplingMode::Temperature) && self.temperature <= 0.0 {
            anyhow::bail!(
                "GrpoConfig.temperature must be > 0 for Temperature sampling (got {})",
                self.temperature
            );
        }
        if let Some(beta) = self.kl_beta {
            if beta < 0.0 {
                anyhow::bail!("GrpoConfig.kl_beta must be >= 0 (got {})", beta);
            }
        }
        Ok(())
    }

    /// Whether the KL penalty is active (beta present and > 0).
    pub fn kl_active(&self) -> bool {
        matches!(self.kl_beta, Some(b) if b > 0.0)
    }
}
