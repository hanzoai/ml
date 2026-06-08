//! The [`Policy`] abstraction: what a model must provide to be trained with
//! GRPO, and how a real autoregressive transformer plugs in.
//!
//! A GRPO step needs exactly two capabilities from the model, mirroring the two
//! phases of the reference `trainer.py`:
//!
//! 1. **Rollout** (`sample_group`): given a prompt, sample `G` completions. This
//!    reads logits but the sampled tokens are *discrete*, so it requires no
//!    autograd graph — equivalent to `model.generate(...)` under `no_grad`.
//! 2. **Scoring** (`token_logprobs`): given prompt + a completion, compute the
//!    per-completion-token log-probabilities under the *current* policy as a
//!    differentiable [`Tensor`]. Equivalent to `_get_per_token_logps`.
//!
//! Keeping these two methods on the trait means the GRPO advantage/loss code
//! (in `math.rs`) is identical for a 1-layer toy policy and a full transformer;
//! only the implementation of these two methods differs.

use crate::grpo::reward::Completion;
use crate::Result;
use hanzo_ml::{Tensor, Var};

/// A rolled-out group: one prompt, `G` sampled completions, and (lazily) the
/// per-token log-probs the trainer will need.
pub struct SampledGroup {
    /// The completions sampled for this prompt (length == `group_size`).
    pub completions: Vec<Completion>,
}

/// Trait implemented by anything trainable with GRPO.
///
/// Implementors own their parameters as `hanzo_ml` [`Var`]s and expose them via
/// [`Policy::trainable_vars`] so the trainer can build an optimizer over them.
pub trait Policy {
    /// The trainable parameters of this policy, handed to the optimizer.
    fn trainable_vars(&self) -> Vec<Var>;

    /// Roll out `group_size` completions for `prompt_tokens`.
    ///
    /// Implementations should sample autoregressively up to `max_len` tokens
    /// using the provided RNG-backed [`Sampler`], stopping early at EOS if the
    /// model defines one. No autograd graph needs to be retained here.
    fn sample_group(
        &self,
        prompt_tokens: &[u32],
        group_size: usize,
        max_len: usize,
        sampler: &mut dyn Sampler,
    ) -> Result<SampledGroup>;

    /// Compute the **summed** log-probability of `completion_tokens` given
    /// `prompt_tokens`, as a scalar [`Tensor`] that is differentiable w.r.t.
    /// [`Policy::trainable_vars`].
    ///
    /// Concretely this is `sum_t log p(c_t | prompt, c_<t)`. Returning the sum
    /// (a single scalar per completion) is the sequence-level objective; a
    /// per-token vector is also valid for token-level KL but the GRPO PG term
    /// only needs the sum (see `math::policy_gradient_loss`).
    fn sequence_logprob(
        &self,
        prompt_tokens: &[u32],
        completion_tokens: &[u32],
    ) -> Result<Tensor>;

    /// Optional decoder from token ids to text, used to populate
    /// [`Completion::completion_text`] for text-based rewards. Default: `None`.
    fn decode(&self, _tokens: &[u32]) -> Option<String> {
        None
    }
}

/// Abstraction over token sampling so policies don't each re-implement
/// temperature / arg-max selection. Backed by [`crate::grpo::sampler`].
pub trait Sampler {
    /// Pick the next token id given a 1-D `logits` tensor of shape `[vocab]`.
    fn sample(&mut self, logits: &Tensor) -> Result<u32>;
}
