//! `hanzo-train` — LoRA + QLoRA trainer for hanzo-engine architectures.
//!
//! Scope: this crate trains **LoRA / QLoRA adapters** for the model families
//! the inference engine (`mistralrs-core`) already serves:
//!
//! * DeepSeek V3 (the Kimi K2.5 base for zen4-ultra / zen5-max)
//! * Qwen3, Qwen3-MoE, Qwen3-Next
//! * GLM-4, GLM-4-MoE
//!
//! It produces PEFT-style `adapter_model.safetensors` files that
//! `mistralrs-core::lora::LoraLinear` and `QLoraLinear` load directly.
//!
//! ## What this crate is NOT
//!
//! * **Not** a full SFT trainer for entire base weights — that lives in
//!   `~/work/hanzo/ml/hanzo-training` (YAML harness around larger Python-side
//!   workflows). hanzo-train is for parameter-efficient training only.
//! * **Not** GRPO / DPO / RLHF — those stay in Python until Rust has a
//!   batched, sampling-aware training engine. See note below.
//!
//! ## Layout
//!
//! * [`lora`] — trainable LoRA module + attach helpers + PEFT save/load
//! * [`qlora`] — 4-bit base weights via `candle::quantized` + fp16 LoRA on top
//! * [`optim`] — AdamW (re-exported from `candle_nn`) + LR schedules
//! * [`data`] — JSONL reader, packer, tokenizer bridge
//! * [`trainer`] — glue: model + optimizer + schedule + data
//! * [`federation_hook`] — every N steps, export adapter as BitDelta and
//!   POST to a federation coordinator (stub: production wire-up requires
//!   the `hanzo-federation` crate which is not yet on this workspace path).
//!
//! ## TODO: GRPO
//!
//! Rust does not yet have a vectorised sampler + reward-model pass that
//! could feed a Group-Relative Policy Optimisation loop. Until that exists,
//! point the federation HTTP path (`federation_hook::push_bitdelta`) at a
//! Python coordinator that runs the GRPO/DPO step and returns merged
//! adapter weights. See `examples/finetune_qwen3.rs` for the shape of
//! the exchange.

pub mod data;
pub mod federation_hook;
pub mod lora;
pub mod optim;
pub mod qlora;
pub mod trainer;

pub use lora::{
    attach::{attach_lora, AttachReport, AttachTarget, MoeMode},
    LoraConfig, PeftAdapterConfig, TrainableLoraLinear,
};
pub use optim::{
    adamw::{TrainableAdamW, TrainableAdamWConfig},
    schedule::{cosine_with_warmup, linear_with_warmup, LrSchedule},
};
pub use qlora::{QloraConfig, QuantBase};
pub use trainer::{StepStats, Trainer, TrainerConfig};

/// Result alias used across the crate.
pub type Result<T> = anyhow::Result<T>;
