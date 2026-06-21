//! Token sampler used during GRPO rollouts.
//!
//! Thin wrapper over the real [`hanzo_transformers::generation::LogitsProcessor`]
//! so rollouts use the same battle-tested arg-max / temperature / top-p sampling
//! path as inference. The reference trainer samples via HF `generate`; this is
//! the `hanzo-ml` equivalent.

use crate::grpo::config::{GrpoConfig, SamplingMode};
use crate::grpo::policy::Sampler;
use crate::Result;
use hanzo_ml::Tensor;
use hanzo_transformers::generation::{LogitsProcessor, Sampling};

/// [`Sampler`] backed by [`LogitsProcessor`].
pub struct LogitsSampler {
    proc: LogitsProcessor,
}

impl LogitsSampler {
    /// Build a sampler from the GRPO config (seed + sampling mode + temperature).
    pub fn from_config(cfg: &GrpoConfig) -> Self {
        let sampling = match cfg.sampling {
            SamplingMode::Greedy => Sampling::ArgMax,
            SamplingMode::Temperature => Sampling::All {
                temperature: cfg.temperature,
            },
        };
        Self {
            proc: LogitsProcessor::from_sampling(cfg.seed, sampling),
        }
    }

    /// Build a sampler with an explicit seed (used to decorrelate per-prompt
    /// rollouts within a step).
    pub fn with_seed(cfg: &GrpoConfig, seed: u64) -> Self {
        let sampling = match cfg.sampling {
            SamplingMode::Greedy => Sampling::ArgMax,
            SamplingMode::Temperature => Sampling::All {
                temperature: cfg.temperature,
            },
        };
        Self {
            proc: LogitsProcessor::from_sampling(seed, sampling),
        }
    }
}

impl Sampler for LogitsSampler {
    fn sample(&mut self, logits: &Tensor) -> Result<u32> {
        let tok = self.proc.sample(logits)?;
        Ok(tok)
    }
}
