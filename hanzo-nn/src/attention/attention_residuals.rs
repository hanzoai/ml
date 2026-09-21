//! Attention Residuals (Moonshot AI Kimi K3 depth-wise attention).
//!
//! Replaces standard fixed residual addition $x_{l+1} = x_l + F_l(x_l)$ with depth-wise attention
//! over all preceding layer representations $\{x_0, x_1, \dots, x_l\}$.
//!
//! Solves the "pre-norm dilution" problem in deep transformers (48-80+ layers), enabling
//! direct highway gradient routing from layer L back to layer 1.

use hanzo_ml::{Module, Result, Tensor};
use crate::Linear;

/// Depth-wise attention residual connection module.
#[derive(Debug, Clone)]
pub struct AttentionResiduals {
    q_proj: Linear,
    k_proj: Linear,
    scale: f64,
}

impl AttentionResiduals {
    pub fn new(vb: crate::VarBuilder, hidden_dim: usize, depth_dim: usize) -> Result<Self> {
        let q_proj = crate::linear_no_bias(hidden_dim, depth_dim, vb.pp("q_proj"))?;
        let k_proj = crate::linear_no_bias(hidden_dim, depth_dim, vb.pp("k_proj"))?;
        let scale = 1.0 / (depth_dim as f64).sqrt();
        Ok(Self { q_proj, k_proj, scale })
    }

    /// Forward pass aggregating preceding layer hidden states.
    ///
    /// `states`: slice of hidden states from previous layers [x_0, x_1, ..., x_l], each of shape `[B, S, D]`.
    /// `current_layer_output`: output of current transformer block F_l(x_l) of shape `[B, S, D]`.
    pub fn forward(&self, states: &[Tensor], current_layer_output: &Tensor) -> Result<Tensor> {
        if states.is_empty() {
            return Ok(current_layer_output.clone());
        }

        // Project current output to query: [B, S, DepthDim]
        let query = self.q_proj.forward(current_layer_output)?;

        // Project each candidate state to key: [B, S, DepthDim]
        let mut weights = Vec::with_capacity(states.len());
        for state in states {
            let key = self.k_proj.forward(state)?;
            // Dot product along depth dimension: [B, S, 1]
            let score = (&query * &key)?.sum_keepdim(hanzo_ml::D::Minus1)?.affine(self.scale, 0.0)?;
            weights.push(score);
        }

        // Stack weights along depth dimension and compute softmax: [B, S, L]
        let weight_stack = Tensor::cat(&weights, hanzo_ml::D::Minus1)?;
        let attn_weights = crate::ops::softmax(&weight_stack, hanzo_ml::D::Minus1)?;

        // Weighted combination of past states
        let mut residual_sum = Tensor::zeros_like(current_layer_output)?;
        for (i, state) in states.iter().enumerate() {
            let w_i = attn_weights.narrow(hanzo_ml::D::Minus1, i, 1)?;
            let weighted_state = state.broadcast_mul(&w_i)?;
            residual_sum = residual_sum.add(&weighted_state)?;
        }

        residual_sum.add(current_layer_output)
    }
}
