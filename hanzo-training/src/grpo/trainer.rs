//! The trainer-free GRPO loop.
//!
//! "Trainer-free" = no external trainer framework (no HF `Trainer`, no
//! `accelerate`); just pure `hanzo-ml` autograd + `hanzo_nn::AdamW`. One
//! [`GrpoTrainer::step`] performs the full GRPO update for a batch of prompts:
//!
//! 1. **Rollout**: sample `group_size` completions per prompt from the current
//!    policy ([`Policy::sample_group`]).
//! 2. **Reward**: score every completion with each [`Reward`] and sum them.
//! 3. **Advantage**: group-relative normalization (`math::batch_advantages`).
//! 4. **Loss**: differentiable policy-gradient loss (+ optional KL to a frozen
//!    reference policy).
//! 5. **Step**: `AdamW.backward_step(&loss)`.

use crate::grpo::config::GrpoConfig;
use crate::grpo::math;
use crate::grpo::metrics::GrpoMetrics;
use crate::grpo::policy::Policy;
use crate::grpo::reward::{Completion, Reward};
use crate::grpo::sampler::LogitsSampler;
use crate::Result;
use hanzo_nn::{AdamW, Optimizer, ParamsAdamW};

/// Trainer-free GRPO over a [`Policy`] `P`, with one or more [`Reward`]s and an
/// optional frozen reference policy `R` for the KL penalty.
pub struct GrpoTrainer<P: Policy, R: Policy = P> {
    config: GrpoConfig,
    policy: P,
    rewards: Vec<Box<dyn Reward>>,
    reward_weights: Vec<f32>,
    reference: Option<R>,
    optimizer: AdamW,
    step_count: usize,
    /// Per-step seed offset so successive steps don't replay identical rollouts.
    seed_cursor: u64,
}

impl<P: Policy> GrpoTrainer<P, P> {
    /// Construct a GRPO trainer without a reference policy (KL disabled).
    pub fn new(config: GrpoConfig, policy: P, rewards: Vec<Box<dyn Reward>>) -> Result<Self> {
        Self::with_reference(config, policy, rewards, None)
    }
}

impl<P: Policy, R: Policy> GrpoTrainer<P, R> {
    /// Construct a GRPO trainer, optionally with a frozen reference policy used
    /// for the KL penalty. If `config.kl_active()` is true a reference *must* be
    /// supplied.
    pub fn with_reference(
        config: GrpoConfig,
        policy: P,
        rewards: Vec<Box<dyn Reward>>,
        reference: Option<R>,
    ) -> Result<Self> {
        config.validate()?;
        if rewards.is_empty() {
            anyhow::bail!("GrpoTrainer requires at least one reward function");
        }
        if config.kl_active() && reference.is_none() {
            anyhow::bail!(
                "kl_beta is set (> 0) but no reference policy was provided; \
                 pass one via with_reference(..)"
            );
        }

        let reward_weights = vec![1.0f32; rewards.len()];
        let params = ParamsAdamW {
            lr: config.learning_rate,
            weight_decay: config.weight_decay,
            ..ParamsAdamW::default()
        };
        let seed_cursor = config.seed;
        let optimizer = AdamW::new(policy.trainable_vars(), params)?;

        Ok(Self {
            config,
            policy,
            rewards,
            reward_weights,
            reference,
            optimizer,
            step_count: 0,
            seed_cursor,
        })
    }

    /// Override the per-reward weights (defaults to all `1.0`). Length must equal
    /// the number of rewards.
    pub fn set_reward_weights(&mut self, weights: Vec<f32>) -> Result<()> {
        if weights.len() != self.rewards.len() {
            anyhow::bail!(
                "expected {} reward weights, got {}",
                self.rewards.len(),
                weights.len()
            );
        }
        self.reward_weights = weights;
        Ok(())
    }

    /// Immutable access to the wrapped policy (e.g. to evaluate after training).
    pub fn policy(&self) -> &P {
        &self.policy
    }

    /// The configuration this trainer was built with.
    pub fn config(&self) -> &GrpoConfig {
        &self.config
    }

    /// Number of completed steps.
    pub fn step_count(&self) -> usize {
        self.step_count
    }

    /// Compute the weighted scalar reward for a completion across all reward
    /// functions: `sum_f w_f * r_f(prompt, completion)`. Mirrors the reference's
    /// `(rewards_per_func * reward_weights).nansum(dim=1)`.
    fn score(&self, completion: &Completion) -> f32 {
        self.rewards
            .iter()
            .zip(self.reward_weights.iter())
            .map(|(r, &w)| w * r.reward(completion))
            .sum()
    }

    /// Run one GRPO step over `prompts` (each prompt is a slice of token ids).
    /// Returns metrics; the policy parameters are updated in place.
    pub fn step(&mut self, prompts: &[Vec<u32>]) -> Result<GrpoMetrics> {
        if prompts.is_empty() {
            anyhow::bail!("GrpoTrainer::step called with no prompts");
        }
        let g = self.config.group_size;

        // ---- Phase 1: rollout + reward (no autograd) ----------------------
        // Flat, group-major layout: [p0g0, p0g1, ..., p1g0, ...].
        let mut completions: Vec<Completion> = Vec::with_capacity(prompts.len() * g);
        let mut rewards: Vec<f32> = Vec::with_capacity(prompts.len() * g);
        let mut per_group_reward_mean: Vec<f32> = Vec::with_capacity(prompts.len());
        let mut per_group_reward_std: Vec<f32> = Vec::with_capacity(prompts.len());

        for (pi, prompt) in prompts.iter().enumerate() {
            // Decorrelate sampling across prompts and across steps.
            let seed = self
                .seed_cursor
                .wrapping_add((pi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
            let mut sampler = LogitsSampler::with_seed(&self.config, seed);

            let mut group =
                self.policy
                    .sample_group(prompt, g, self.config.max_completion_len, &mut sampler)?;
            debug_assert_eq!(group.completions.len(), g);

            // Populate decoded text for text-based rewards if the policy can.
            let mut group_rewards = Vec::with_capacity(g);
            for c in group.completions.iter_mut() {
                if c.completion_text.is_none() {
                    c.completion_text = self.policy.decode(&c.completion_tokens);
                }
                let r = self.score(c);
                group_rewards.push(r);
            }

            let mean = group_rewards.iter().copied().sum::<f32>() / g as f32;
            let var = if g > 1 {
                group_rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / (g as f32 - 1.0)
            } else {
                0.0
            };
            per_group_reward_mean.push(mean);
            per_group_reward_std.push(var.sqrt());

            completions.extend(group.completions);
            rewards.extend(group_rewards);
        }

        // ---- Phase 2: group-relative advantages (detached constants) ------
        let advantages = math::batch_advantages(
            &rewards,
            g,
            self.config.scale_rewards,
            self.config.advantage_eps,
        );

        // ---- Phase 3: differentiable per-completion sequence log-probs -----
        let mut seq_logprobs = Vec::with_capacity(completions.len());
        let mut lengths = Vec::with_capacity(completions.len());
        for c in completions.iter() {
            let lp = self
                .policy
                .sequence_logprob(&c.prompt_tokens, &c.completion_tokens)?;
            seq_logprobs.push(lp);
            lengths.push(c.completion_tokens.len());
        }

        // ---- Phase 4: loss = PG (+ optional KL) ---------------------------
        let pg_loss = math::policy_gradient_loss(&advantages, &seq_logprobs, &lengths)?;
        let pg_val = pg_loss.to_scalar::<f32>()?;

        let (loss, kl_val) = if self.config.kl_active() {
            let beta = self.config.kl_beta.unwrap();
            let reference = self
                .reference
                .as_ref()
                .expect("kl_active implies reference present (checked at construction)");
            // Reference log-probs are constants (detached scalars).
            let mut ref_logprobs = Vec::with_capacity(completions.len());
            for c in completions.iter() {
                let lp = reference
                    .sequence_logprob(&c.prompt_tokens, &c.completion_tokens)?
                    .to_scalar::<f32>()?;
                ref_logprobs.push(lp);
            }
            let kl = math::kl_penalty(beta, &seq_logprobs, &ref_logprobs, &lengths)?;
            let kl_val = kl.to_scalar::<f32>()?;
            let total = (&pg_loss + &kl)?;
            (total, kl_val)
        } else {
            (pg_loss, 0.0f32)
        };

        let loss_val = loss.to_scalar::<f32>()?;

        // ---- Phase 5: optimizer step --------------------------------------
        // Skip the optimizer when the loss has no gradient path or is
        // non-finite, but still surface the metrics so callers can detect it.
        if loss_val.is_finite() {
            self.optimizer.backward_step(&loss)?;
        }

        self.step_count += 1;
        // Advance the seed cursor so the next step samples differently.
        self.seed_cursor = self
            .seed_cursor
            .wrapping_add((prompts.len() as u64).wrapping_add(1));

        // ---- Metrics ------------------------------------------------------
        let num_groups = prompts.len();
        let num_completions = completions.len();
        let mean_reward =
            per_group_reward_mean.iter().copied().sum::<f32>() / num_groups.max(1) as f32;
        let reward_std =
            per_group_reward_std.iter().copied().sum::<f32>() / num_groups.max(1) as f32;
        let mean_completion_len =
            lengths.iter().map(|&l| l as f32).sum::<f32>() / num_completions.max(1) as f32;

        Ok(GrpoMetrics {
            loss: loss_val,
            pg_loss: pg_val,
            kl_loss: kl_val,
            mean_reward,
            reward_std,
            mean_completion_len,
            num_groups,
            num_completions,
        })
    }
}
