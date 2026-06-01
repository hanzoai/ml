//! PEFT-compatible adapter persistence.
//!
//! ## Wire format (matches HuggingFace PEFT v0.x)
//!
//! Two files written into a single directory:
//!
//! * `adapter_model.safetensors` — flat safetensors archive whose keys are
//!   `base_model.model.<dotted_path>.lora_A.weight` (shape `(rank, in)`)
//!   and `.lora_B.weight` (shape `(out, rank)`).
//! * `adapter_config.json` — see [`PeftAdapterConfig`].
//!
//! `mistralrs-core::lora::make_adapter` reads exactly these shapes and
//! names — see `engine/mistralrs-core/src/lora/mod.rs` lines 75–92.

use std::{collections::HashMap, fs, path::Path};

use candle::Tensor;
use serde::{Deserialize, Serialize};

use super::{LoraConfig, TrainableLoraLinear};

/// `adapter_config.json` contents (PEFT-compatible subset).
///
/// Notable choices:
///
/// * `peft_type = "LORA"` — what PEFT writes for plain LoRA.
/// * `task_type = "CAUSAL_LM"` — what `transformers`/`mistralrs` look for.
/// * `bias = "none"` — we never train biases.
/// * `fan_in_fan_out = false` — matches HuggingFace Linear convention.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PeftAdapterConfig {
    pub peft_type: String,
    pub task_type: String,
    pub r: usize,
    pub lora_alpha: f64,
    pub lora_dropout: f32,
    pub target_modules: Vec<String>,
    pub bias: String,
    pub fan_in_fan_out: bool,
    pub base_model_name_or_path: Option<String>,
}

impl PeftAdapterConfig {
    pub fn from_lora_config(lora: &LoraConfig, base_model: Option<String>) -> Self {
        Self {
            peft_type: "LORA".into(),
            task_type: "CAUSAL_LM".into(),
            r: lora.rank,
            lora_alpha: lora.alpha,
            lora_dropout: lora.dropout,
            target_modules: lora.target_modules.clone(),
            bias: "none".into(),
            fan_in_fan_out: false,
            base_model_name_or_path: base_model,
        }
    }

    pub fn to_lora_config(&self) -> LoraConfig {
        LoraConfig {
            rank: self.r,
            alpha: self.lora_alpha,
            dropout: self.lora_dropout,
            target_modules: self.target_modules.clone(),
        }
    }
}

/// Save the wrapped layers as a PEFT directory at `out_dir`.
///
/// Layers are keyed by their dotted path (the same one passed to
/// [`crate::lora::attach::attach_lora`]). The output paths are
/// `base_model.model.<path>.lora_A.weight` and `.lora_B.weight`.
pub fn save_peft_adapter<P: AsRef<Path>>(
    out_dir: P,
    layers: &[(String, TrainableLoraLinear)],
    cfg: &LoraConfig,
    base_model: Option<String>,
) -> crate::Result<()> {
    let out_dir = out_dir.as_ref();
    fs::create_dir_all(out_dir)?;

    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(layers.len() * 2);
    for (path, lin) in layers {
        let key_a = format!("base_model.model.{path}.lora_A.weight");
        let key_b = format!("base_model.model.{path}.lora_B.weight");
        tensors.insert(key_a, lin.lora_a().as_tensor().clone());
        tensors.insert(key_b, lin.lora_b().as_tensor().clone());
    }

    candle::safetensors::save(&tensors, out_dir.join("adapter_model.safetensors"))?;

    let adapter_cfg = PeftAdapterConfig::from_lora_config(cfg, base_model);
    let cfg_json = serde_json::to_string_pretty(&adapter_cfg)?;
    fs::write(out_dir.join("adapter_config.json"), cfg_json)?;

    Ok(())
}

/// Load a PEFT adapter directory back into a flat tensor map. Useful
/// for tests and federation merge logic. Inference loading is handled
/// by `mistralrs-core::lora::make_adapter`, not by this crate.
pub fn load_peft_adapter<P: AsRef<Path>>(
    dir: P,
    device: &candle::Device,
) -> crate::Result<(PeftAdapterConfig, HashMap<String, Tensor>)> {
    let dir = dir.as_ref();
    let cfg: PeftAdapterConfig =
        serde_json::from_str(&fs::read_to_string(dir.join("adapter_config.json"))?)?;
    let tensors =
        candle::safetensors::load(dir.join("adapter_model.safetensors"), device)?;
    Ok((cfg, tensors))
}
