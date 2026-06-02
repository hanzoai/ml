//! LoRA training primitives.
//!
//! The wire format we emit and consume is the **PEFT v0.x** safetensors
//! layout: weights live at `base_model.model.<dotted_path>.lora_A.weight`
//! and `.lora_B.weight`, alongside an `adapter_config.json`. That is
//! exactly what `mistralrs-core/src/lora/mod.rs::make_adapter` reads.

pub mod adapter;
pub mod attach;
pub mod linear;

pub use adapter::{load_peft_adapter, save_peft_adapter, PeftAdapterConfig};
pub use linear::TrainableLoraLinear;

use serde::{Deserialize, Serialize};

/// LoRA configuration. Fields mirror PEFT's `LoraConfig`.
///
/// Wire compatibility: `rank` -> `r`, `alpha` -> `lora_alpha`, `dropout`
/// -> `lora_dropout` when serialised. We expose Rust-natural names here
/// and translate at save-time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoraConfig {
    /// LoRA rank `r`. Trainable params per layer = `r * (in + out)`.
    pub rank: usize,
    /// Scaling factor. Effective contribution: `(alpha / rank) * B(A(x))`.
    pub alpha: f64,
    /// Dropout applied to the LoRA input. 0.0 disables.
    pub dropout: f32,
    /// Leaf module names to attach to, e.g. `["q_proj", "v_proj", ...]`.
    pub target_modules: Vec<String>,
}

impl LoraConfig {
    /// Effective scale `alpha / rank` (or `1.0` when rank is zero).
    pub fn scale(&self) -> f64 {
        if self.rank == 0 {
            1.0
        } else {
            self.alpha / self.rank as f64
        }
    }

    /// Standard attention-only target set used by the PEFT defaults
    /// for Qwen3 / DeepSeek-style models.
    pub fn attention_only(rank: usize, alpha: f64) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".into(),
                "k_proj".into(),
                "v_proj".into(),
                "o_proj".into(),
            ],
        }
    }

    /// All seven projections in a Qwen3 / DeepSeek decoder layer.
    /// MoE layers further restrict via [`attach::MoeMode`].
    pub fn all_linear(rank: usize, alpha: f64) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            target_modules: vec![
                "q_proj".into(),
                "k_proj".into(),
                "v_proj".into(),
                "o_proj".into(),
                "gate_proj".into(),
                "up_proj".into(),
                "down_proj".into(),
            ],
        }
    }

    /// GLM-4 attention layout (fused QKV + dense output).
    pub fn glm4_attention(rank: usize, alpha: f64) -> Self {
        Self {
            rank,
            alpha,
            dropout: 0.0,
            target_modules: vec!["query_key_value".into(), "dense".into()],
        }
    }
}
