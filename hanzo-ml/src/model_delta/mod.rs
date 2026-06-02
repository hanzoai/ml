//! Model merging and delta-compression utilities, implemented natively on top of
//! the `hanzo-ml` tensor + safetensors APIs (no Python, no external tooling).
//!
//! Two features live here:
//!
//! * [`soup`] / [`delta_soup`] — *model soups*: average the weights (or the
//!   fine-tune *deltas*) of several checkpoints into a single merged checkpoint.
//!   See <https://arxiv.org/abs/2203.05482>.
//! * [`encode`] / [`decode_apply`] — *BitDelta*: compress a fine-tune into a
//!   1-bit-per-weight sign mask plus one f32 scale per tensor, reconstructing
//!   `finetuned ≈ base + scale · sign(Δ)`.
//!   See <https://arxiv.org/abs/2402.10193>.
//!
//! ```no_run
//! use std::path::{Path, PathBuf};
//! use hanzo_ml::model_delta;
//!
//! # fn main() -> hanzo_ml::Result<()> {
//! // Average three checkpoints with uniform weights.
//! let models = vec![PathBuf::from("a.safetensors"), PathBuf::from("b.safetensors")];
//! model_delta::soup(&models, None, Path::new("souped.safetensors"))?;
//!
//! // Compress a fine-tune relative to its base into a 1-bit delta, then restore it.
//! model_delta::encode(Path::new("base.safetensors"), Path::new("ft.safetensors"), Path::new("ft.bitdelta"))?;
//! model_delta::decode_apply(Path::new("base.safetensors"), Path::new("ft.bitdelta"), Path::new("restored.safetensors"))?;
//! # Ok(()) }
//! ```

mod bitdelta;
mod soup;

pub use bitdelta::{
    decode_apply, encode, BitDelta, BitDeltaHeader, TensorDeltaHeader, BITDELTA_MAGIC,
    BITDELTA_VERSION,
};
pub use soup::{delta_soup, soup};

use crate::{Device, Result, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// All of this module's work happens on the CPU; merging/compression is memory-bound
/// elementwise arithmetic and keeping it on the host avoids needless device transfers.
pub(crate) const DEVICE: Device = Device::Cpu;

/// Load a safetensors checkpoint into a name -> tensor map on the CPU.
pub(crate) fn load_checkpoint(path: &Path) -> Result<HashMap<String, Tensor>> {
    crate::safetensors::load(path, &DEVICE)
}

/// Write a name -> tensor map out as a safetensors checkpoint.
pub(crate) fn save_checkpoint(tensors: &HashMap<String, Tensor>, path: &Path) -> Result<()> {
    crate::safetensors::save(tensors, path)
}

/// Format a shape as `[a, b, c]` for error messages.
pub(crate) fn fmt_shape(dims: &[usize]) -> String {
    let inner = dims
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    format!("[{inner}]")
}
