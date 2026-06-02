//! # hanzo-rlhf
//!
//! RLHF facade for the Hanzo stack. There is one and only one way to run
//! preference / reward training from inside hanzod: this crate.
//!
//! ## Why a facade?
//!
//! The reference implementations of GRPO, DPO, PPO, KTO, SimPO, and ORPO
//! live in Python (TRL, Unsloth, etc.) and depend on the
//! Transformers/PyTorch ecosystem for backward, mixed precision, flash
//! attention kernels, deepspeed/FSDP sharding, and dataset adapters. We
//! aren't going to reimplement that in candle this quarter; the candle
//! training side does not have the kernels yet.
//!
//! So the contract is:
//!
//! * The **Rust API** ([`RewardModel`], [`AdvantageEstimator`],
//!   [`PolicyUpdater`], [`RlhfAlgorithm`]) is the only thing hanzod
//!   callers ever see. It's typed, allocation-light, and stable.
//! * For each algorithm in [`RlhfAlgorithm`], [`python_bridge::run_via_federation`]
//!   posts the config to a Python trainer process discoverable through
//!   the federation coordinator. The trainer streams back checkpoints
//!   as canonical BF16 delta blobs that hanzo-federation knows how to
//!   apply via [`hanzo_zen5::Zen5Engine::apply_delta`].
//! * The native loops ([`grpo`]) exist as scaffolding — types, function
//!   signatures, glue — so the day candle gains training kernels we can
//!   light them up without changing call sites.
//!
//! ## Boundary: what's Rust, what's Python
//!
//! | Concern                | Rust (this crate)     | Python (trainer)        |
//! | ---------------------- | --------------------- | ----------------------- |
//! | API surface            | full                  | none (hanzod hides it)  |
//! | Reward computation     | trait + ref impl      | optional override       |
//! | Advantage estimation   | trait + ref impl      | TRL implementations     |
//! | Backward + optimizer   | unimplemented         | TRL / Unsloth           |
//! | Delta encoding         | hanzo-federation BF16 | hanzo-federation BF16   |
//! | Delta application      | hanzo_zen5 / candle   | not applicable          |
//! | Coordinator transport  | hanzo-federation HTTP | hanzo-federation HTTP   |
//!
//! Both sides talk the *same wire format* (canonical BF16 blob defined
//! in `hanzo_federation::codec`), so a Python trainer can post deltas
//! that any Rust worker can ingest, and vice versa.

#![deny(rust_2018_idioms)]
#![warn(missing_docs)]
#![warn(missing_debug_implementations)]

pub mod grpo;
pub mod python_bridge;

use candle_nn::Module;
use serde::{Deserialize, Serialize};

/// One token in a prompt or completion. Opaque to this crate; tokenization
/// is owned by the caller (typically `hanzo_zen5` or the Python trainer).
pub type Token = i32;

/// One (prompt, completion) pair scored by a [`RewardModel`].
#[derive(Debug, Clone)]
pub struct Sample {
    /// The prompt token ids fed to the policy.
    pub prompt: Vec<Token>,
    /// The completion the policy produced for that prompt.
    pub completion: Vec<Token>,
    /// Per-token log-probability the policy assigned at sampling time.
    /// Required for PPO/GRPO importance ratios; can be empty for
    /// algorithms that recompute log-probs each step (DPO).
    pub logprobs: Vec<f32>,
}

/// A reward model scores `(prompt, completion)` pairs.
///
/// Implementations can be:
/// * a pre-trained scalar reward model (the classical RLHF case),
/// * a programmatic rubric (`+1` if the completion compiles, `0` else),
/// * an LLM-as-a-judge that calls another model via `hanzo_engine::infer`,
/// * or, for KTO / DPO style "preference" training, a binary preference
///   over pairs — wrap the pair in a `Sample` with the higher-preferred
///   completion and score it `+1`, the dispreferred `-1`.
pub trait RewardModel: Send + Sync {
    /// Score `(prompt, completion)` and return a scalar reward in arbitrary
    /// units. The estimator is responsible for normalising across a batch.
    fn score(&self, prompt: &[Token], completion: &[Token]) -> f32;
}

/// Convert a batch of per-sample rewards into per-sample advantages.
///
/// For GRPO this is the group-relative normalisation
/// `A_i = (r_i - mean(r_group)) / std(r_group)`.
/// For PPO this is GAE on a learned value baseline (implemented in the
/// Python trainer today — see [`crate::python_bridge`]).
pub trait AdvantageEstimator: Send + Sync {
    /// Given `rewards.len()` samples produce `rewards.len()` advantages.
    fn advantage(&self, rewards: &[f32]) -> Vec<f32>;
}

/// Apply one policy update over `samples` weighted by `advantages`.
///
/// Implementations either:
/// * own a backward pass (the future native path; today
///   [`grpo::CandleGrpoUpdater::update`] panics with `unimplemented!()`
///   because candle does not yet ship the training kernels we need), or
/// * delegate to the Python trainer over the federation transport
///   ([`python_bridge::BridgeUpdater`]).
pub trait PolicyUpdater: Send + Sync {
    /// Run one optimisation step and return the average loss.
    fn update(
        &mut self,
        policy: &mut dyn Module,
        samples: &[Sample],
        advantages: &[f32],
    ) -> anyhow::Result<f32>;
}

/// Marker for the algorithm a caller wants to run. The set is closed: if
/// you need a new one, add a variant here and a Python trainer endpoint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RlhfAlgorithm {
    /// Group-Relative Policy Optimization. See [`grpo`].
    Grpo,
    /// Direct Preference Optimization.
    Dpo,
    /// Proximal Policy Optimization.
    Ppo,
    /// Kahneman-Tversky Optimization.
    Kto,
    /// Simple Preference Optimization.
    Simpo,
    /// Odds-Ratio Preference Optimization.
    Orpo,
}

impl RlhfAlgorithm {
    /// The wire-format string we send to the Python trainer endpoint.
    /// Lowercase matches TRL's `trainer_name` conventions.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Grpo => "grpo",
            Self::Dpo => "dpo",
            Self::Ppo => "ppo",
            Self::Kto => "kto",
            Self::Simpo => "simpo",
            Self::Orpo => "orpo",
        }
    }
}

/// Result handle for a training run dispatched through the federation
/// bridge. Drop it to detach; the trainer keeps running until completion
/// or the coordinator signals shutdown.
#[derive(Debug)]
pub struct RlhfHandle {
    /// Trainer job id assigned by the coordinator.
    pub job_id: String,
    /// Background task polling status updates. Polled by [`RlhfHandle::wait`].
    pub join: tokio::task::JoinHandle<anyhow::Result<RlhfOutcome>>,
}

impl RlhfHandle {
    /// Block until the run terminates and return the final outcome.
    pub async fn wait(self) -> anyhow::Result<RlhfOutcome> {
        self.join.await?
    }
}

/// Terminal outcome of an RLHF run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RlhfOutcome {
    /// Algorithm that produced this outcome.
    pub algorithm: RlhfAlgorithm,
    /// Total optimization steps consumed.
    pub steps: u32,
    /// Final mean loss (algorithm-specific units).
    pub final_loss: f32,
    /// Path or URL to the final canonical BF16 delta. Apply via
    /// `hanzo_zen5::Zen5Engine::apply_delta` or its Mistral equivalent.
    pub delta_uri: String,
}
