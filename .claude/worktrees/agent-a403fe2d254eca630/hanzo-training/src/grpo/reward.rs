//! Reward functions for GRPO.
//!
//! In the reference trainer a reward function maps `(prompts, completions) ->
//! list[float]`. We model the same contract with a [`Reward`] trait that scores
//! a single `(prompt, completion)` pair. Token ids are passed alongside the
//! decoded text so rewards can be defined over either representation (e.g. a
//! length-target reward works directly on token ids, while a regex/exact-match
//! reward works on text).

/// A completion produced by the policy together with everything a reward needs
/// to score it.
#[derive(Debug, Clone)]
pub struct Completion {
    /// The prompt token ids this completion was generated from.
    pub prompt_tokens: Vec<u32>,
    /// The generated completion token ids (EOS-truncated).
    pub completion_tokens: Vec<u32>,
    /// Optional decoded prompt text (if a tokenizer/decoder is available).
    pub prompt_text: Option<String>,
    /// Optional decoded completion text.
    pub completion_text: Option<String>,
}

impl Completion {
    /// Construct a token-only completion (no decoded text).
    pub fn from_tokens(prompt_tokens: Vec<u32>, completion_tokens: Vec<u32>) -> Self {
        Self {
            prompt_tokens,
            completion_tokens,
            prompt_text: None,
            completion_text: None,
        }
    }
}

/// Scores completions. The scalar return value is the (unweighted) reward; the
/// trainer handles group-relative normalization.
pub trait Reward: Send + Sync {
    /// Human-readable name, used in metrics (mirrors `reward_func_names`).
    fn name(&self) -> &str;

    /// Score a single completion. Higher is better.
    fn reward(&self, completion: &Completion) -> f32;
}

/// Reward that peaks when the completion length matches a target number of
/// tokens. `r = -|len - target|`, so the optimum (reward 0) is achieved exactly
/// at `target` tokens and it decreases linearly away from it. This is trivially
/// learnable by a policy that controls its own completion length, which makes it
/// ideal for smoke-testing the GRPO loop end-to-end.
#[derive(Debug, Clone)]
pub struct LengthTargetReward {
    target: usize,
    name: String,
}

impl LengthTargetReward {
    pub fn new(target: usize) -> Self {
        Self {
            target,
            name: format!("len_target_{target}"),
        }
    }
}

impl Reward for LengthTargetReward {
    fn name(&self) -> &str {
        &self.name
    }

    fn reward(&self, completion: &Completion) -> f32 {
        let len = completion.completion_tokens.len() as i64;
        let target = self.target as i64;
        -(len - target).abs() as f32
    }
}

/// Reward that returns `+1.0` for every occurrence of a specific target token in
/// the completion (and `0.0` otherwise). A policy can maximize this by learning
/// to emit the target token, which exercises the same gradient path as a real
/// preference reward while being deterministic and dependency-free (no regex
/// crate needed). Think of it as a token-level "exact match" reward.
#[derive(Debug, Clone)]
pub struct TokenMatchReward {
    target_token: u32,
    name: String,
}

impl TokenMatchReward {
    pub fn new(target_token: u32) -> Self {
        Self {
            target_token,
            name: format!("token_match_{target_token}"),
        }
    }
}

impl Reward for TokenMatchReward {
    fn name(&self) -> &str {
        &self.name
    }

    fn reward(&self, completion: &Completion) -> f32 {
        completion
            .completion_tokens
            .iter()
            .filter(|&&t| t == self.target_token)
            .count() as f32
    }
}

/// Exact-string-match reward over decoded completion text: `+1.0` if the
/// completion text equals `target`, else `0.0`. Falls back to `0.0` when no
/// decoded text is available. Demonstrates a text-based reward analogous to the
/// regex/exact-match rewards used with real tokenizers.
#[derive(Debug, Clone)]
pub struct ExactMatchReward {
    target: String,
    name: String,
}

impl ExactMatchReward {
    pub fn new(target: impl Into<String>) -> Self {
        Self {
            target: target.into(),
            name: "exact_match".to_string(),
        }
    }
}

impl Reward for ExactMatchReward {
    fn name(&self) -> &str {
        &self.name
    }

    fn reward(&self, completion: &Completion) -> f32 {
        match &completion.completion_text {
            Some(text) if *text == self.target => 1.0,
            _ => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_target_reward_peaks_at_target() {
        let r = LengthTargetReward::new(3);
        let at = Completion::from_tokens(vec![], vec![1, 2, 3]);
        let short = Completion::from_tokens(vec![], vec![1]);
        let long = Completion::from_tokens(vec![], vec![1, 2, 3, 4, 5]);
        assert_eq!(r.reward(&at), 0.0);
        assert_eq!(r.reward(&short), -2.0);
        assert_eq!(r.reward(&long), -2.0);
    }

    #[test]
    fn token_match_counts_occurrences() {
        let r = TokenMatchReward::new(7);
        let c = Completion::from_tokens(vec![], vec![7, 1, 7, 7, 2]);
        assert_eq!(r.reward(&c), 3.0);
    }
}
