//! Group-Relative Policy Optimization (GRPO) scaffold.
//!
//! GRPO is the algorithm DeepSeek used to train R1-zero. For each prompt
//! the policy generates a *group* of completions; rewards are normalised
//! within the group so the policy moves toward the relatively-better
//! samples without needing a learned value baseline.
//!
//! ## Sketch of the loop
//!
//! ```text
//! for batch in dataset:
//!     # 1. group sampling
//!     groups = [policy.sample(prompt, n=G) for prompt in batch]
//!
//!     # 2. score each completion
//!     rewards = [[reward.score(p, c) for c in group] for (p, group) in groups]
//!
//!     # 3. group-relative advantage
//!     advantages = [(r - mean(group_r)) / (std(group_r) + eps)
//!                   for group_r in rewards for r in group_r]
//!
//!     # 4. surrogate loss
//!     #     L = -E[ min( ratio * A, clip(ratio, 1-eps, 1+eps) * A ) ]
//!     # 5. KL penalty against the frozen reference policy
//!     #     L += beta * KL(policy || ref)
//!     loss = surrogate_loss(samples, advantages) + beta * kl(policy, ref)
//!     loss.backward()
//!     optimizer.step()
//! ```
//!
//! ## What's implemented today
//!
//! * Group-relative advantage normalisation ([`GroupRelativeEstimator`]).
//! * Group-mean / group-std utility ([`group_stats`]).
//! * Trait wiring through to the policy-update step ([`CandleGrpoUpdater`]).
//!
//! ## What's not
//!
//! The backward + optimizer step. candle ships forward kernels for the
//! attention shapes we want, but doesn't yet have the training-side
//! flash attention or the fused AdamW we need to run GRPO at scale.
//! The Python trainer (TRL `GRPOTrainer`) bridged via
//! [`crate::python_bridge`] is the production path until those land.

use std::marker::PhantomData;

use candle_nn::Module;

use crate::{AdvantageEstimator, PolicyUpdater, Sample};

/// Number of completions sampled per prompt. DeepSeek's R1-zero recipe
/// uses 64; we default to 8 for fitness on a single Spark / H100.
pub const DEFAULT_GROUP_SIZE: usize = 8;

/// PPO-style clip parameter for the importance ratio. 0.2 matches TRL.
pub const DEFAULT_CLIP_EPS: f32 = 0.2;

/// KL coefficient β. Higher → policy stays closer to reference.
/// 0.04 is the DeepSeek R1-zero default; turn it up if you see the policy
/// drift into degenerate outputs.
pub const DEFAULT_KL_BETA: f32 = 0.04;

/// GRPO hyperparameters. Mirrored on the Python side in TRL's
/// `GRPOConfig`; field names match for ease of comparison.
#[derive(Debug, Clone)]
pub struct GrpoConfig {
    /// Completions per prompt. See [`DEFAULT_GROUP_SIZE`].
    pub group_size: usize,
    /// Importance-ratio clip. See [`DEFAULT_CLIP_EPS`].
    pub clip_eps: f32,
    /// KL penalty coefficient. See [`DEFAULT_KL_BETA`].
    pub kl_beta: f32,
    /// Learning rate for the inner AdamW.
    pub learning_rate: f32,
    /// Numerical floor for the per-group std.
    pub std_eps: f32,
}

impl Default for GrpoConfig {
    fn default() -> Self {
        Self {
            group_size: DEFAULT_GROUP_SIZE,
            clip_eps: DEFAULT_CLIP_EPS,
            kl_beta: DEFAULT_KL_BETA,
            learning_rate: 1e-6,
            std_eps: 1e-8,
        }
    }
}

/// Group-relative advantage estimator. `rewards` must be laid out as
/// `[g0_r0, g0_r1, …, g0_r{G-1}, g1_r0, …]` for a fixed group size
/// [`GrpoConfig::group_size`].
#[derive(Debug, Clone)]
pub struct GroupRelativeEstimator {
    /// Group size G. See [`GrpoConfig::group_size`].
    pub group_size: usize,
    /// Numerical floor on the per-group std.
    pub std_eps: f32,
}

impl Default for GroupRelativeEstimator {
    fn default() -> Self {
        Self {
            group_size: DEFAULT_GROUP_SIZE,
            std_eps: 1e-8,
        }
    }
}

impl AdvantageEstimator for GroupRelativeEstimator {
    fn advantage(&self, rewards: &[f32]) -> Vec<f32> {
        if self.group_size == 0 {
            return Vec::new();
        }
        let mut out = Vec::with_capacity(rewards.len());
        for chunk in rewards.chunks(self.group_size) {
            let (mean, std) = group_stats(chunk);
            let denom = std + self.std_eps;
            for &r in chunk {
                out.push((r - mean) / denom);
            }
        }
        out
    }
}

/// Per-group mean and (population) standard deviation. Pure compute; no
/// allocations besides the two `f32` outputs.
pub fn group_stats(group: &[f32]) -> (f32, f32) {
    let n = group.len() as f32;
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let mean = group.iter().sum::<f32>() / n;
    let var = group.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
    (mean, var.sqrt())
}

/// The native GRPO updater. Holds the [`GrpoConfig`] and a frozen
/// reference policy used for the KL term. Update is `unimplemented!()`
/// today — see the module docs for why.
pub struct CandleGrpoUpdater<P: Module + Send + Sync> {
    /// Configuration knobs.
    pub config: GrpoConfig,
    /// Frozen reference policy. Used for the KL penalty against the
    /// current (trainable) policy.
    pub reference: P,
    /// Phantom so we can name `P` in the struct without owning more
    /// state. The trainable policy is passed into [`PolicyUpdater::update`].
    _policy: PhantomData<P>,
}

impl<P: Module + Send + Sync + std::fmt::Debug> std::fmt::Debug for CandleGrpoUpdater<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleGrpoUpdater")
            .field("config", &self.config)
            .field("reference", &self.reference)
            .finish()
    }
}

impl<P: Module + Send + Sync> CandleGrpoUpdater<P> {
    /// Build a new updater. `reference` must be a frozen snapshot of the
    /// policy at the start of the run.
    pub fn new(config: GrpoConfig, reference: P) -> Self {
        Self {
            config,
            reference,
            _policy: PhantomData,
        }
    }
}

impl<P: Module + Send + Sync> PolicyUpdater for CandleGrpoUpdater<P> {
    fn update(
        &mut self,
        _policy: &mut dyn Module,
        _samples: &[Sample],
        _advantages: &[f32],
    ) -> anyhow::Result<f32> {
        // TODO: this needs candle backward + flash attention training kernels.
        // The intended body:
        //   1. forward(policy, samples) -> logprobs_new
        //   2. forward(reference, samples) -> logprobs_ref
        //   3. ratio = exp(logprobs_new - sample.logprobs)
        //   4. clip = clamp(ratio, 1 - clip_eps, 1 + clip_eps)
        //   5. surrogate = -mean( min(ratio * A, clip * A) )
        //   6. kl = mean(logprobs_new - logprobs_ref)
        //   7. loss = surrogate + kl_beta * kl
        //   8. loss.backward(); optimizer.step()
        // Until candle ships those primitives, GRPO runs through
        // crate::python_bridge::BridgeUpdater which dispatches to a TRL
        // GRPOTrainer behind hanzo-federation.
        unimplemented!(
            "native GRPO not implemented yet — use crate::python_bridge for a real run"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn group_relative_zero_mean_per_group() {
        let est = GroupRelativeEstimator {
            group_size: 4,
            std_eps: 0.0,
        };
        let rewards = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let adv = est.advantage(&rewards);
        // Per group: mean ≈ 0
        let g0: f32 = adv[..4].iter().sum();
        let g1: f32 = adv[4..].iter().sum();
        assert!(g0.abs() < 1e-4, "group 0 mean {g0}");
        assert!(g1.abs() < 1e-4, "group 1 mean {g1}");
    }

    #[test]
    fn group_relative_handles_zero_variance() {
        // If everyone in the group gets the same reward, normalising
        // by std=0 must not produce NaN/inf — that's what std_eps is for.
        let est = GroupRelativeEstimator::default();
        let rewards = vec![3.0; est.group_size];
        let adv = est.advantage(&rewards);
        for a in adv {
            assert!(a.is_finite(), "advantage was non-finite: {a}");
        }
    }

    #[test]
    fn group_stats_matches_textbook() {
        let g = [1.0, 2.0, 3.0, 4.0, 5.0];
        let (m, s) = group_stats(&g);
        assert!((m - 3.0).abs() < 1e-6, "mean was {m}");
        // population std of 1..=5 is sqrt(2) ≈ 1.41421
        assert!((s - 2f32.sqrt()).abs() < 1e-4, "std was {s}");
    }

    #[test]
    fn algorithm_strings_are_canonical() {
        use crate::RlhfAlgorithm::*;
        assert_eq!(Grpo.as_str(), "grpo");
        assert_eq!(Dpo.as_str(), "dpo");
        assert_eq!(Ppo.as_str(), "ppo");
        assert_eq!(Kto.as_str(), "kto");
        assert_eq!(Simpo.as_str(), "simpo");
        assert_eq!(Orpo.as_str(), "orpo");
    }
}
