//! UltraBERT numerical parity test against PyTorch HuggingFace reference.

use hanzo_ml::{DType, Device, Result};
use hanzo_nn::VarBuilder;
use hanzo_transformers::models::ultrabert::{Config, UltrabertModel};
use std::path::PathBuf;

#[test]
fn test_ultrabert_parity() -> Result<()> {
    let base_path = match std::env::var("HOME") {
        Ok(home) => PathBuf::from(home).join(".cache/decision-long/models/ultrabert-380m"),
        Err(_) => return Ok(()),
    };

    let config_path = base_path.join("config.json");
    let weights_path = base_path.join("model.safetensors");
    let golden_path = base_path.join("golden_out.safetensors");

    if !weights_path.exists() || !golden_path.exists() {
        println!("Skipping parity test: weights or golden output not found at {:?}", base_path);
        return Ok(());
    }

    let dev = Device::Cpu;

    // Load config
    let config_str = std::fs::read_to_string(&config_path)
        .map_err(|e| hanzo_ml::Error::Msg(format!("failed to read config: {e}")))?;
    let cfg: Config = serde_json::from_str(&config_str)
        .map_err(|e| hanzo_ml::Error::Msg(format!("failed to parse config: {e}")))?;

    // Load weights
    let vb = unsafe {
        VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &dev)?
    };

    println!("Loading UltrabertModel in Rust...");
    let model = UltrabertModel::load(vb.pp("ultrabert"), &cfg)?;

    // Load golden inputs and outputs
    let golden_tensors = hanzo_ml::safetensors::load(&golden_path, &dev)?;
    let input_ids = golden_tensors.get("input_ids")
        .ok_or_else(|| hanzo_ml::Error::Msg("missing input_ids in golden".into()))?;
    let expected_out = golden_tensors.get("output")
        .ok_or_else(|| hanzo_ml::Error::Msg("missing output in golden".into()))?;

    // In PyTorch, input_ids is int64, in hanzo-ml it should be u32
    let input_ids = if input_ids.dtype() == DType::I64 {
        input_ids.to_dtype(DType::U32)?
    } else {
        input_ids.clone()
    };

    println!("Running Rust forward pass on input_ids {:?}...", input_ids.dims());
    let actual_out = model.forward(&input_ids, None)?;

    assert_eq!(actual_out.dims(), expected_out.dims());

    let diff = (&actual_out - expected_out)?.abs()?;
    let max_diff: f32 = diff.max_all()?.to_scalar()?;
    let mean_diff: f32 = diff.mean_all()?.to_scalar()?;

    println!("Max absolute difference: {:e}", max_diff);
    println!("Mean absolute difference: {:e}", mean_diff);

    // Assert parity tolerance
    assert!(
        max_diff < 1e-4,
        "Parity check failed! max |Δz| = {:e} (expected < 1e-4)",
        max_diff
    );

    println!("UltraBERT PyTorch numerical bit parity verified successfully! (max |Δ| = {:e})", max_diff);
    Ok(())
}
