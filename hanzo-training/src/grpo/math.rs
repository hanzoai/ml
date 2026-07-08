//! The core GRPO algorithm: group-relative advantages and the policy-gradient
//! loss. This module is deliberately model-agnostic — it operates on plain
//! reward scalars and on log-prob [`Tensor`]s produced by any [`Policy`]. The
//! exact same code path is used by the toy smoke-test policy and by a real
//! transformer.
//!
//! Ported from `zooai-gym` `trainer.py::_generate_and_score_completions`:
//!
//! ```text
//! mean_grouped = rewards.view(-1, G).mean(dim=1)
//! std_grouped  = rewards.view(-1, G).std(dim=1)
//! advantages   = rewards - mean_grouped          # broadcast over the group
//! if scale_rewards:
//!     advantages = advantages / (std_grouped + 1e-4)
//! ```
//!
//! and the loss (TRL GRPO, `num_iterations == 1`, no clipping):
//!
//! ```text
//! per_token_loss = -(advantage * logprob)        # + beta * KL
//! loss           = mean over tokens / completions
//! ```

use crate::Result;
use hanzo_ml::Tensor;

/// Group-relative advantages for a *single* group of completions sharing one
/// prompt. `rewards` has length `G` (the group size).
///
/// Returns `A_i = (r_i - mean(r))` optionally divided by `std(r) + eps`, exactly
/// as in the reference. The (population) std matches PyTorch's default unbiased
/// std only up to the `ddof`; we use the **unbiased** (`n-1`) std to match
/// `torch.std`'s default, falling back to `0` for degenerate groups.
pub fn group_advantages(rewards: &[f32], scale_rewards: bool, eps: f64) -> Vec<f32> {
    let n = rewards.len();
    debug_assert!(n >= 2, "group_advantages requires a group of >= 2");
    let mean = rewards.iter().copied().sum::<f32>() / n as f32;

    if !scale_rewards {
        return rewards.iter().map(|&r| r - mean).collect();
    }

    // Unbiased sample std (ddof=1), matching torch.std default. For n == 1 this
    // would divide by zero; callers guarantee n >= 2 but we guard anyway.
    let var = if n > 1 {
        rewards.iter().map(|&r| (r - mean).powi(2)).sum::<f32>() / (n as f32 - 1.0)
    } else {
        0.0
    };
    let std = var.sqrt();
    let denom = std + eps as f32;
    rewards.iter().map(|&r| (r - mean) / denom).collect()
}

/// Compute advantages for a whole batch of groups laid out as `[group0..,
/// group1.., ...]`, each of length `group_size`. Returns a flat vector aligned
/// with the input. This is the batched equivalent of [`group_advantages`].
pub fn batch_advantages(
    rewards: &[f32],
    group_size: usize,
    scale_rewards: bool,
    eps: f64,
) -> Vec<f32> {
    debug_assert!(group_size >= 2);
    debug_assert_eq!(rewards.len() % group_size, 0);
    let mut out = Vec::with_capacity(rewards.len());
    for chunk in rewards.chunks(group_size) {
        out.extend(group_advantages(chunk, scale_rewards, eps));
    }
    out
}

/// The GRPO policy-gradient loss for one batch.
///
/// `advantages[i]` is the (already group-normalized, detached) scalar advantage
/// of completion `i`, and `seq_logprobs[i]` is the *differentiable* summed
/// sequence log-prob `sum_t log p(c_t | ...)` for completion `i`, normalized by
/// the completion length so that long and short completions contribute
/// comparably (matching the per-token mean in the reference loss).
///
/// loss = - mean_i ( A_i * mean_logprob_i )
///
/// Note the sign: maximizing expected advantage-weighted log-prob ==> we
/// *minimize* its negation with the optimizer. Advantages are constants w.r.t.
/// the parameters (detached), so the gradient is the REINFORCE/PG estimator
/// `- A_i * d/dθ logprob_i`.
pub fn policy_gradient_loss(
    advantages: &[f32],
    seq_logprobs: &[Tensor],
    completion_lengths: &[usize],
) -> Result<Tensor> {
    assert_eq!(advantages.len(), seq_logprobs.len());
    assert_eq!(advantages.len(), completion_lengths.len());
    assert!(
        !advantages.is_empty(),
        "empty batch in policy_gradient_loss"
    );

    // Per-completion contribution: A_i * (logprob_i / len_i). We build it as a
    // sum of differentiable scalars then divide by N for the mean.
    let mut total: Option<Tensor> = None;
    for ((&adv, logprob), &len) in advantages
        .iter()
        .zip(seq_logprobs.iter())
        .zip(completion_lengths.iter())
    {
        let len = len.max(1) as f64;
        // mean per-token logprob (differentiable), then weight by the constant
        // advantage. `affine(adv/len, 0)` keeps the graph and folds both
        // constants in one op.
        let weighted = logprob.affine(adv as f64 / len, 0.0)?;
        total = Some(match total {
            Some(acc) => (acc + weighted)?,
            None => weighted,
        });
    }

    let n = advantages.len() as f64;
    // mean, then negate: loss = -(1/N) sum_i A_i * mean_logprob_i.
    // Fold the mean (1/N) and the negation into one affine: scale by -1/N.
    let loss = total.unwrap().affine(-1.0 / n, 0.0)?;
    Ok(loss)
}

/// Optional KL-penalty term, `beta * mean_i KL(policy_i || ref_i)`, added to the
/// PG loss when a reference policy is configured.
///
/// We use the unbiased low-variance KL estimator from the GRPO/TRL paper applied
/// at the sequence level: for sequence log-prob ratios `Δ = logπ_ref - logπ`,
///
/// ```text
/// k3 = exp(Δ) - Δ - 1   >= 0
/// ```
///
/// which is a non-negative estimator of `KL(π || π_ref)`. `policy_logprobs` are
/// differentiable; `ref_logprobs` are detached constants.
pub fn kl_penalty(
    beta: f64,
    policy_logprobs: &[Tensor],
    ref_logprobs: &[f32],
    completion_lengths: &[usize],
) -> Result<Tensor> {
    assert_eq!(policy_logprobs.len(), ref_logprobs.len());
    assert_eq!(policy_logprobs.len(), completion_lengths.len());
    assert!(!policy_logprobs.is_empty());

    let mut total: Option<Tensor> = None;
    for ((policy_lp, &ref_lp), &len) in policy_logprobs
        .iter()
        .zip(ref_logprobs.iter())
        .zip(completion_lengths.iter())
    {
        let len = len.max(1) as f64;
        // Work in per-token mean space to match the PG term's normalization.
        let policy_mean = policy_lp.affine(1.0 / len, 0.0)?;
        let ref_mean = ref_lp as f64 / len;
        // Δ = ref - policy  (differentiable through `policy_mean`).
        let delta = policy_mean.affine(-1.0, ref_mean)?;
        // k3 = exp(Δ) - Δ - 1
        let k3 = ((delta.exp()? - &delta)? - 1.0)?;
        total = Some(match total {
            Some(acc) => (acc + k3)?,
            None => k3,
        });
    }
    let n = policy_logprobs.len() as f64;
    let mean_kl = total.unwrap().affine(beta / n, 0.0)?;
    Ok(mean_kl)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::{Device, Tensor};

    #[test]
    fn advantages_are_zero_mean_within_group() {
        let r = vec![1.0f32, 2.0, 3.0, 4.0];
        let a = group_advantages(&r, false, 1e-4);
        let sum: f32 = a.iter().sum();
        assert!(sum.abs() < 1e-5, "advantages should sum to ~0, got {sum}");
        // monotonic: higher reward -> higher advantage
        assert!(a[0] < a[1] && a[1] < a[2] && a[2] < a[3]);
    }

    #[test]
    fn scaled_advantages_have_unit_scale() {
        let r = vec![0.0f32, 10.0];
        let a = group_advantages(&r, true, 1e-4);
        // With ddof=1 std of {0,10} is ~7.071; (10-5)/7.071 ~= 0.707
        assert!(
            (a[1] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-2,
            "got {}",
            a[1]
        );
        assert!(
            (a[0] + std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-2,
            "got {}",
            a[0]
        );
    }

    #[test]
    fn degenerate_group_gives_zero_advantage() {
        let r = vec![5.0f32, 5.0, 5.0];
        let a = group_advantages(&r, true, 1e-4);
        for v in a {
            assert_eq!(v, 0.0);
        }
    }

    #[test]
    fn pg_loss_sign_and_finiteness() -> Result<()> {
        let dev = Device::Cpu;
        // logprob scalars (as if from policy). Positive advantage on the larger
        // logprob should yield a negative loss contribution.
        let lp0 = Tensor::new(-2.0f32, &dev)?;
        let lp1 = Tensor::new(-1.0f32, &dev)?;
        let loss = policy_gradient_loss(&[-1.0, 1.0], &[lp0, lp1], &[1, 1])?;
        let v = loss.to_scalar::<f32>()?;
        assert!(v.is_finite());
        // loss = -mean( (-1)*(-2) + (1)*(-1) )/... = -mean(2 + -1)= -(1)/2 = -0.5
        assert!((v + 0.5).abs() < 1e-5, "got {v}");
        Ok(())
    }

    #[test]
    fn kl_is_nonnegative_and_zero_at_match() -> Result<()> {
        let dev = Device::Cpu;
        // Policy logprob equals ref logprob -> Δ=0 -> k3=0.
        let p = Tensor::new(-3.0f32, &dev)?;
        let kl = kl_penalty(0.1, &[p], &[-3.0], &[1])?;
        let v = kl.to_scalar::<f32>()?;
        assert!(v.abs() < 1e-5, "KL at match should be 0, got {v}");

        // Mismatch -> strictly positive.
        let p2 = Tensor::new(-1.0f32, &dev)?;
        let kl2 = kl_penalty(0.1, &[p2], &[-3.0], &[1])?;
        assert!(kl2.to_scalar::<f32>()? > 0.0);
        Ok(())
    }
}
