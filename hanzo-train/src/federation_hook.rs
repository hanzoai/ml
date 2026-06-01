//! Federation hook.
//!
//! ## Status: stub
//!
//! The production path here is:
//!
//! 1. Every `N` optimizer steps, snapshot the current LoRA tensors.
//! 2. Compute a BitDelta against the last snapshot we exported (so we
//!    transmit only the 1-bit / multi-bit signs of weight changes).
//! 3. POST to a federation coordinator via `hanzo-federation`'s
//!    `TransportClient`.
//!
//! `hanzo-federation` is not yet present in this workspace path
//! (`grep -r "hanzo-federation" ~/work/hanzo/ml` returns nothing as of
//! the task date). Same for `hanzo-quantize::bitdelta`. Until those
//! crates land we expose [`push_bitdelta`] as a no-op that returns
//! `Err` with a clear message, plus the type signatures the eventual
//! implementation will use.
//!
//! When the federation crate lands, swap [`push_bitdelta`]'s body to:
//!
//! ```ignore
//! let delta = hanzo_quantize::bitdelta::compute(prev, current, bits)?;
//! hanzo_federation::TransportClient::new(coordinator_url)?
//!     .push_delta("hanzo-train", run_id, delta)
//!     .await
//! ```

use std::collections::HashMap;

use candle::Tensor;

use crate::lora::TrainableLoraLinear;

/// What we eventually send. Holds enough info to apply or aggregate at
/// the coordinator without round-tripping a full safetensors file.
#[derive(Clone, Debug)]
pub struct LoraDeltaSnapshot {
    pub run_id: String,
    pub step: usize,
    pub layer_count: usize,
    /// per-key f32 delta tensors (placeholder; production uses BitDelta).
    pub deltas: HashMap<String, Tensor>,
}

/// Compute the f32 delta of each layer's `(A, B)` against the previous
/// snapshot. Returns a fresh `HashMap`.
pub fn snapshot_diff(
    prev: &HashMap<String, Tensor>,
    layers: &[(String, TrainableLoraLinear)],
) -> crate::Result<HashMap<String, Tensor>> {
    let mut out = HashMap::with_capacity(layers.len() * 2);
    for (path, lin) in layers {
        let key_a = format!("{path}.lora_A");
        let key_b = format!("{path}.lora_B");
        let now_a = lin.lora_a().as_tensor();
        let now_b = lin.lora_b().as_tensor();
        let delta_a = match prev.get(&key_a) {
            Some(p) => now_a.sub(p)?,
            None => now_a.clone(),
        };
        let delta_b = match prev.get(&key_b) {
            Some(p) => now_b.sub(p)?,
            None => now_b.clone(),
        };
        out.insert(key_a, delta_a);
        out.insert(key_b, delta_b);
    }
    Ok(out)
}

/// Stub: push a delta snapshot to a federation coordinator.
///
/// Returns an error explaining what to wire up. Replace the body when
/// `hanzo-federation` is on the workspace path.
pub fn push_bitdelta(
    _coordinator_url: &str,
    _snapshot: LoraDeltaSnapshot,
) -> crate::Result<()> {
    anyhow::bail!(
        "federation_hook::push_bitdelta is stubbed — hanzo-federation \
         and hanzo-quantize::bitdelta are not yet on this workspace \
         path. See module docs for the production wire-up."
    )
}
