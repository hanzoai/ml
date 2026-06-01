//! QLoRA: keep the base model in 4-bit and train fp16 LoRA on top.
//!
//! ## Implementation choices
//!
//! Original QLoRA (Dettmers et al., 2023) uses NF4 + double-quant from
//! BitsAndBytes. Pure Rust does not have NF4 yet, but candle ships
//! `Q4_0` / `Q4_K` k-quants from llama.cpp that are within ~0.1
//! perplexity of NF4 for fine-tuning purposes. We use `Q4_K` by default
//! (smaller error, k-block size 256).
//!
//! ## Gradient flow
//!
//! The base weight is **frozen and quantised** so it does not appear
//! in the optimizer's `Var` list. On forward we dequantise it on the
//! fly and run a normal matmul, which gives the LoRA tensors a clean
//! gradient path. Memory savings come from holding the base in 4-bit
//! between forward calls.

pub mod attach;

pub use attach::{attach_qlora, QloraConfig, QuantBase};
