//! Attach LoRA adapters to a set of named [`candle_nn::Linear`] layers
//! that the caller extracts from a transformer model.
//!
//! ## Why no reflection?
//!
//! Candle transformer modules hold linears as struct fields, not as a
//! dynamic registry. Rather than fight the type system with a custom
//! derive, we ask the caller to hand us a `Vec<AttachTarget>` listing
//! the named linears it wants wrapped. The example
//! `finetune_qwen3.rs` shows the one-liner per architecture.
//!
//! This keeps the attach logic completely architecture-agnostic — it
//! works for Qwen3, Qwen3-MoE, Qwen3-Next, DeepSeek V3 and GLM-4 with
//! the same single function, because it only cares about the **leaf
//! name** of each layer (`q_proj`, `gate_proj`, `query_key_value`, …).

use std::collections::HashSet;

use candle_nn::Linear;

use super::{LoraConfig, TrainableLoraLinear};

/// One linear layer the caller wants to wrap.
///
/// `dotted_path` is the fully-qualified module path that will be used
/// as the PEFT key, e.g. `model.layers.0.self_attn.q_proj`. The trailing
/// segment is matched against `LoraConfig::target_modules`.
#[derive(Debug)]
pub struct AttachTarget {
    pub dotted_path: String,
    pub linear: Linear,
}

impl AttachTarget {
    pub fn new<S: Into<String>>(dotted_path: S, linear: Linear) -> Self {
        Self {
            dotted_path: dotted_path.into(),
            linear,
        }
    }

    pub fn leaf(&self) -> &str {
        self.dotted_path
            .rsplit_once('.')
            .map(|(_, leaf)| leaf)
            .unwrap_or(self.dotted_path.as_str())
    }
}

/// How to handle MoE FFN modules when the user asks for `gate_proj` /
/// `up_proj` / `down_proj`.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum MoeMode {
    /// Only attach LoRA to FFN linears that live under `shared_experts.*`
    /// (or anything that is **not** under a `.experts.<n>.` path).
    /// Default — keeps the routing topology identical and the adapter
    /// small. This is what most LoRA-on-MoE recipes do.
    #[default]
    SharedOnly,
    /// Attach LoRA to **every** matching FFN linear, including each
    /// routed expert. Multiplies adapter size by `(n_experts + 1)`.
    AllExperts,
}

/// Outcome of attach — useful for tests / logging.
#[derive(Debug, Default)]
pub struct AttachReport {
    /// Names of layers that were wrapped, in the order encountered.
    pub attached: Vec<String>,
    /// Names of layers that the user pre-listed but were filtered out
    /// (e.g. routed experts when [`MoeMode::SharedOnly`]).
    pub skipped: Vec<String>,
}

impl AttachReport {
    pub fn count(&self) -> usize {
        self.attached.len()
    }
}

/// Walk `targets` and emit a `TrainableLoraLinear` for each layer whose
/// leaf name matches `cfg.target_modules`, applying [`MoeMode`] routing.
///
/// Returns the wrapped layers in the same order they were supplied, and
/// a sibling report listing what we did.
pub fn attach_lora(
    targets: Vec<AttachTarget>,
    cfg: &LoraConfig,
    moe_mode: MoeMode,
) -> crate::Result<(Vec<(String, TrainableLoraLinear)>, AttachReport)> {
    let want: HashSet<&str> = cfg.target_modules.iter().map(String::as_str).collect();
    let mut report = AttachReport::default();
    let mut out = Vec::new();
    for t in targets {
        let leaf = t.leaf();
        if !want.contains(leaf) {
            continue;
        }
        if moe_mode == MoeMode::SharedOnly && is_routed_expert(&t.dotted_path) {
            report.skipped.push(t.dotted_path);
            continue;
        }
        let wrapped = TrainableLoraLinear::new(t.linear, cfg)?;
        report.attached.push(t.dotted_path.clone());
        out.push((t.dotted_path, wrapped));
    }
    Ok((out, report))
}

/// Heuristic for "this linear lives under a routed expert":
/// the dotted path contains `.experts.` followed by a numeric segment,
/// e.g. `model.layers.7.mlp.experts.0.gate_proj`. This matches the
/// naming used by Qwen3-MoE, DeepSeek V3, GLM-4-MoE and Qwen3-Next.
///
/// Shared experts are typically at `mlp.shared_experts.*` or
/// `mlp.shared_expert.*` which do **not** contain a numeric segment
/// after `experts`, so they pass through.
fn is_routed_expert(path: &str) -> bool {
    let mut iter = path.split('.');
    while let Some(seg) = iter.next() {
        if seg == "experts" {
            if let Some(next) = iter.next() {
                if next.chars().all(|c| c.is_ascii_digit()) {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn routed_expert_detection() {
        assert!(is_routed_expert("model.layers.7.mlp.experts.0.gate_proj"));
        assert!(is_routed_expert("model.layers.7.mlp.experts.255.up_proj"));
        assert!(!is_routed_expert("model.layers.7.mlp.shared_experts.gate_proj"));
        assert!(!is_routed_expert("model.layers.7.mlp.shared_expert.gate_proj"));
        assert!(!is_routed_expert("model.layers.7.self_attn.q_proj"));
        assert!(!is_routed_expert("model.layers.7.mlp.gate_proj"));
    }

    #[test]
    fn leaf_extraction() {
        let t = AttachTarget::new(
            "model.layers.0.self_attn.q_proj",
            // We don't need a real Linear for this test — we won't call attach.
            Linear::new(
                candle::Tensor::zeros((1, 1), candle::DType::F32, &candle::Device::Cpu).unwrap(),
                None,
            ),
        );
        assert_eq!(t.leaf(), "q_proj");
    }
}
