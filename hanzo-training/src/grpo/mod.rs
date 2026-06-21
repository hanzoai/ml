//! Trainer-free GRPO (Group Relative Policy Optimization).
//!
//! A pure-`hanzo-ml`-autograd implementation of GRPO — no external trainer
//! framework. The algorithm is ported from the `zooai-gym` GRPO trainer
//! (`trainers/grpo/{args,sampler,trainer}.py`); see [`math`] for the exact
//! formula correspondence.
//!
//! # Pipeline (one [`GrpoTrainer::step`])
//! 1. For each prompt, sample a *group* of `G` completions from the current
//!    policy ([`Policy::sample_group`]).
//! 2. Score each completion with one or more [`Reward`]s.
//! 3. Compute the **group-relative advantage**
//!    `Aᵢ = (rᵢ − mean(r)) / (std(r) + eps)` ([`math::group_advantages`]).
//! 4. Form the policy-gradient loss `−mean(Aᵢ · logprob(completionᵢ))`, with an
//!    optional KL penalty to a frozen reference policy
//!    ([`math::policy_gradient_loss`], [`math::kl_penalty`]).
//! 5. Take an AdamW step ([`hanzo_nn::AdamW`]).
//!
//! # Plugging in a real model
//! Implement [`Policy`] for your model:
//! - `trainable_vars()` returns the model's `hanzo-ml` [`Var`](hanzo_ml::Var)s.
//! - `sample_group()` is your autoregressive `generate` loop (uses the provided
//!   [`Sampler`], which wraps the real `LogitsProcessor`).
//! - `sequence_logprob()` is a differentiable teacher-forced forward pass that
//!   returns `Σ_t log p(c_t | prompt, c_<t)`.
//!
//! The GRPO math in [`math`] and the driver in [`GrpoTrainer`] are identical for
//! the toy policy used in tests and for a full transformer — only those three
//! methods change.

pub mod config;
pub mod math;
pub mod metrics;
pub mod policy;
pub mod reward;
pub mod sampler;
pub mod trainer;

pub use config::{GrpoConfig, SamplingMode};
pub use metrics::GrpoMetrics;
pub use policy::{Policy, SampledGroup, Sampler};
pub use reward::{Completion, ExactMatchReward, LengthTargetReward, Reward, TokenMatchReward};
pub use sampler::LogitsSampler;
pub use trainer::GrpoTrainer;

#[cfg(test)]
mod toy_tests {
    //! End-to-end smoke test on a minimal toy policy.
    //!
    //! The toy policy is a *contextual bandit over a tiny vocab*: it holds a
    //! single learnable logits table of shape `[max_len, vocab]` (one categorical
    //! per output position, prompt-independent). A completion is `max_len`
    //! independently-sampled tokens. This is the smallest setup that still
    //! exercises the **real** GRPO code path end to end:
    //!   - rollout via the real `LogitsProcessor`-backed [`Sampler`],
    //!   - a differentiable `log_softmax`+`gather` sequence log-prob,
    //!   - the real [`math::group_advantages`] / [`math::policy_gradient_loss`],
    //!   - a real `hanzo_nn::AdamW` step.
    //!
    //! Reward = number of target tokens emitted ([`TokenMatchReward`]), which is
    //! trivially learnable: the policy should learn to put all its mass on the
    //! target token, driving mean reward up to `max_len`.

    use super::*;
    use crate::Result;
    use hanzo_ml::{DType, Device, Tensor, Var};
    use hanzo_nn::ops::log_softmax;
    use hanzo_nn::{VarBuilder, VarMap};

    /// Minimal learnable policy: `logits[pos, token]`, prompt-independent.
    struct ToyPolicy {
        logits: Tensor, // shape [max_len, vocab], tied to a Var in `varmap`
        varmap: VarMap,
        vocab: usize,
        max_len: usize,
        device: Device,
    }

    impl ToyPolicy {
        fn new(vocab: usize, max_len: usize) -> Result<Self> {
            let device = Device::Cpu;
            let varmap = VarMap::new();
            let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
            // Initialize logits to zero -> uniform policy at start.
            let logits = vb.get_with_hints(
                (max_len, vocab),
                "logits",
                hanzo_nn::Init::Const(0.0),
            )?;
            Ok(Self {
                logits,
                varmap,
                vocab,
                max_len,
                device,
            })
        }
    }

    impl Policy for ToyPolicy {
        fn trainable_vars(&self) -> Vec<Var> {
            self.varmap.all_vars()
        }

        fn sample_group(
            &self,
            _prompt_tokens: &[u32],
            group_size: usize,
            max_len: usize,
            sampler: &mut dyn Sampler,
        ) -> Result<SampledGroup> {
            let steps = max_len.min(self.max_len);
            let mut completions = Vec::with_capacity(group_size);
            for _ in 0..group_size {
                let mut toks = Vec::with_capacity(steps);
                for pos in 0..steps {
                    // logits for this position: shape [vocab]
                    let pos_logits = self.logits.narrow(0, pos, 1)?.squeeze(0)?;
                    let tok = sampler.sample(&pos_logits)?;
                    toks.push(tok);
                }
                completions.push(Completion::from_tokens(vec![], toks));
            }
            Ok(SampledGroup { completions })
        }

        fn sequence_logprob(
            &self,
            _prompt_tokens: &[u32],
            completion_tokens: &[u32],
        ) -> Result<Tensor> {
            // Σ_t log softmax(logits[t])[c_t], differentiable w.r.t. `logits`.
            let logp = log_softmax(&self.logits, 1)?; // [max_len, vocab]
            let mut total: Option<Tensor> = None;
            for (pos, &tok) in completion_tokens.iter().enumerate() {
                debug_assert!((tok as usize) < self.vocab);
                let idx = Tensor::new(&[tok], &self.device)?; // [1]
                // row `pos`, then gather the token's logprob -> scalar tensor
                let row = logp.narrow(0, pos, 1)?.squeeze(0)?; // [vocab]
                let val = row.index_select(&idx, 0)?.squeeze(0)?; // scalar
                total = Some(match total {
                    Some(acc) => (acc + val)?,
                    None => val,
                });
            }
            match total {
                Some(t) => Ok(t),
                None => Ok(Tensor::new(0.0f32, &self.device)?),
            }
        }
    }

    #[test]
    fn grpo_smoke_reward_increases() -> Result<()> {
        let vocab = 5usize;
        let max_len = 4usize;
        let target_token = 3u32;

        let policy = ToyPolicy::new(vocab, max_len)?;
        let cfg = GrpoConfig {
            group_size: 16,
            max_completion_len: max_len,
            sampling: SamplingMode::Temperature,
            temperature: 1.0,
            scale_rewards: true,
            advantage_eps: 1e-4,
            kl_beta: None,
            learning_rate: 0.5,
            weight_decay: 0.0,
            seed: 42,
        };
        let rewards: Vec<Box<dyn Reward>> = vec![Box::new(TokenMatchReward::new(target_token))];
        let mut trainer = GrpoTrainer::new(cfg, policy, rewards)?;

        // A few prompts (content irrelevant for the bandit policy, but exercises
        // the batching/group layout).
        let prompts = vec![vec![1u32, 2], vec![0u32], vec![4u32, 4, 4]];

        let first = trainer.step(&prompts)?;
        assert!(first.is_finite(), "first-step loss not finite: {first:?}");

        let mut last = first.clone();
        let mut rewards_trace = vec![first.mean_reward];
        for _ in 0..40 {
            last = trainer.step(&prompts)?;
            assert!(last.is_finite(), "loss became non-finite: {last:?}");
            rewards_trace.push(last.mean_reward);
        }

        println!(
            "[grpo smoke] first mean_reward={:.4} last mean_reward={:.4} \
             (max possible={}) first_loss={:.4} last_loss={:.4} last_kl={:.4}",
            first.mean_reward, last.mean_reward, max_len, first.loss, last.loss, last.kl_loss
        );
        println!("[grpo smoke] reward trace: {rewards_trace:?}");

        // Core assertion: GRPO learned the trivially-learnable reward. Mean
        // reward must increase substantially from the (uniform-policy) start.
        assert!(
            last.mean_reward > first.mean_reward + 0.5,
            "mean reward did not increase enough: {} -> {}",
            first.mean_reward,
            last.mean_reward
        );
        // It should approach the ceiling (all `max_len` tokens == target).
        assert!(
            last.mean_reward > max_len as f32 * 0.6,
            "mean reward {} did not approach ceiling {}",
            last.mean_reward,
            max_len
        );
        Ok(())
    }

    #[test]
    fn grpo_smoke_with_kl_runs_and_is_finite() -> Result<()> {
        // Same toy setup but with a frozen reference policy and KL penalty
        // active. Verifies the KL branch wires up and stays finite.
        let vocab = 5usize;
        let max_len = 3usize;
        let policy = ToyPolicy::new(vocab, max_len)?;
        let reference = ToyPolicy::new(vocab, max_len)?; // frozen (never stepped)

        let cfg = GrpoConfig {
            group_size: 8,
            max_completion_len: max_len,
            sampling: SamplingMode::Temperature,
            temperature: 1.0,
            scale_rewards: true,
            advantage_eps: 1e-4,
            kl_beta: Some(0.05),
            learning_rate: 0.2,
            weight_decay: 0.0,
            seed: 7,
        };
        let rewards: Vec<Box<dyn Reward>> = vec![Box::new(TokenMatchReward::new(2))];
        let mut trainer =
            GrpoTrainer::with_reference(cfg, policy, rewards, Some(reference))?;

        let prompts = vec![vec![1u32], vec![2u32]];
        let mut last = trainer.step(&prompts)?;
        for _ in 0..10 {
            last = trainer.step(&prompts)?;
            assert!(last.is_finite(), "loss became non-finite with KL: {last:?}");
        }
        // KL term should be active and non-negative.
        assert!(last.kl_loss >= 0.0, "kl_loss negative: {}", last.kl_loss);
        println!(
            "[grpo smoke kl] last loss={:.4} pg={:.4} kl={:.4} reward={:.4}",
            last.loss, last.pg_loss, last.kl_loss, last.mean_reward
        );
        Ok(())
    }
}
