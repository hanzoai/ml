//! Runnable, dependency-free GRPO demo on a tiny toy policy.
//!
//! Run with:
//! ```bash
//! cargo run -p hanzo-training --example grpo_toy
//! ```
//!
//! This mirrors the smoke test but as a standalone binary so you can watch the
//! mean reward climb to the ceiling under the *real* trainer-free GRPO loop
//! (`GrpoTrainer::step`). The toy policy is a contextual bandit over a tiny
//! vocab; swapping in a real transformer is purely a matter of implementing the
//! `Policy` trait's `sample_group` / `sequence_logprob` for it (see the module
//! docs on `hanzo_training::grpo`).

use anyhow::Result;
use hanzo_ml::{DType, Device, Tensor, Var};
use hanzo_nn::ops::log_softmax;
use hanzo_nn::{Init, VarBuilder, VarMap};
use hanzo_training::grpo::{
    Completion, GrpoConfig, GrpoTrainer, Policy, Reward, SampledGroup, Sampler, SamplingMode,
    TokenMatchReward,
};

/// Minimal learnable policy: a `[max_len, vocab]` logits table (one categorical
/// per output position, prompt-independent). The smallest setup that exercises
/// the full GRPO autograd path.
struct ToyPolicy {
    logits: Tensor,
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
        let logits = vb.get_with_hints((max_len, vocab), "logits", Init::Const(0.0))?;
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
                let pos_logits = self.logits.narrow(0, pos, 1)?.squeeze(0)?;
                toks.push(sampler.sample(&pos_logits)?);
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
        let logp = log_softmax(&self.logits, 1)?;
        let mut total: Option<Tensor> = None;
        for (pos, &tok) in completion_tokens.iter().enumerate() {
            debug_assert!((tok as usize) < self.vocab);
            let idx = Tensor::new(&[tok], &self.device)?;
            let row = logp.narrow(0, pos, 1)?.squeeze(0)?;
            let val = row.index_select(&idx, 0)?.squeeze(0)?;
            total = Some(match total {
                Some(acc) => (acc + val)?,
                None => val,
            });
        }
        Ok(match total {
            Some(t) => t,
            None => Tensor::new(0.0f32, &self.device)?,
        })
    }
}

fn main() -> Result<()> {
    let vocab = 6usize;
    let max_len = 5usize;
    let target_token = 4u32;

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
        seed: 1234,
    };
    let rewards: Vec<Box<dyn Reward>> = vec![Box::new(TokenMatchReward::new(target_token))];
    let mut trainer = GrpoTrainer::new(cfg, policy, rewards)?;

    let prompts = vec![vec![1u32, 2, 3], vec![0u32], vec![5u32, 5]];

    println!(
        "GRPO toy demo: learn to emit token {target_token} (reward ceiling = {max_len} per completion)\n"
    );
    println!(
        "{:>4}  {:>10}  {:>10}  {:>10}  {:>12}",
        "step", "loss", "pg_loss", "reward", "mean_len"
    );
    for step in 0..30 {
        let m = trainer.step(&prompts)?;
        if step % 3 == 0 || step == 29 {
            println!(
                "{:>4}  {:>10.4}  {:>10.4}  {:>10.4}  {:>12.2}",
                step, m.loss, m.pg_loss, m.mean_reward, m.mean_completion_len
            );
        }
    }
    println!("\nDone. Mean reward should approach {max_len} as the policy concentrates on the target token.");
    Ok(())
}
