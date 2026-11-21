//! `hanzo-train` — native-Rust training loop for DSpark speculative-draft models.
//!
//! Closes the all-Rust loop: dump → cache → **train** → deploy. Trains a DSpark draft on real
//! cached target hidden states using `hanzo-ml` autograd + `hanzo-nn` AdamW, and writes a checkpoint
//! in the exact layout the engine's `qwen3_dspark.rs` loader expects.

pub mod cache;
pub mod model;

pub use cache::Cache;
pub use model::{markov_bias, verify_checkpoint, Dspark, DsparkCfg};
