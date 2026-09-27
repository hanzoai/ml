//! `hanzo-train` — training on hanzo-ml.
//!
//! - [`cluster`]: one model trained across heterogeneous machines (Metal, CUDA, ROCm, CPU) by
//!   local SGD with an outer Nesterov step (DiLoCo): a coordinator, workers that join at any
//!   time, bf16 deltas with error feedback, requeue on drop, checkpoints that resume bit for bit.
//!   A model plugs in behind [`cluster::Model`].
//! - [`adam`]: AdamW with a learning rate per parameter and its state in the open.
//! - [`gpu`]: a floor on free memory, and GPU utilization on macOS.
//! - [`model`], [`cache`], [`dspark`]: the DSpark speculative-draft model, the target cache it
//!   trains on, and the two on the cluster runtime; the `hanzo-train` binary (`fit`, `join`)
//!   trains it and writes a checkpoint in the exact layout the engine's `qwen3_dspark.rs` loader
//!   expects.

pub mod adam;
pub mod cache;
pub mod cluster;
pub mod dspark;
pub mod gpu;
pub mod model;

pub use cache::Cache;
pub use model::{markov_bias, verify_checkpoint, Dspark, DsparkCfg};
