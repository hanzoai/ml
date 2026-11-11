//! UltraBERT model smoke and architecture test.

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{VarBuilder, VarMap};
use hanzo_transformers::models::ultrabert::{Config, UltrabertModel};

fn tiny_config() -> Config {
    Config {
        vocab_size: 100,
        hidden_size: 64,
        intermediate_size: 128,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        num_key_value_heads: Some(2),
        head_dim: 32,
        rms_norm_eps: 1e-6,
        pad_token_id: 0,
        attention_windows: vec![Some(4), Some(4), None],
        rope_theta: vec![10000.0, 10000.0, 2560000.0],
        prediction_head_size: Some(64),
        max_position_embeddings: Some(128),
    }
}

#[test]
fn test_ultrabert_forward() -> Result<()> {
    let dev = Device::Cpu;
    let cfg = tiny_config();
    let vm = VarMap::new();
    let vb = VarBuilder::from_varmap(&vm, DType::F32, &dev);

    let model = UltrabertModel::load(vb, &cfg)?;

    let input_ids = Tensor::new(&[[1u32, 5, 12, 18, 24, 30]], &dev)?;
    let out = model.forward(&input_ids, None)?;

    assert_eq!(out.dims(), &[1, 6, 64]);
    Ok(())
}
