//! ModernBERT's CPU inference pass against its training pass, on a tiny random backbone.

use hanzo_ml::quantized::GgmlDType;
use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::{VarBuilder, VarMap};
use hanzo_transformers::models::modernbert::{Config, ModernBert};

fn config() -> Config {
    Config {
        vocab_size: 50,
        hidden_size: 64,
        num_hidden_layers: 3,
        num_attention_heads: 2,
        intermediate_size: 64,
        max_position_embeddings: 64,
        layer_norm_eps: 1e-5,
        pad_token_id: 0,
        global_attn_every_n_layers: 3,
        global_rope_theta: 160000.,
        local_attention: 8,
        local_rope_theta: 10000.,
        classifier_config: None,
    }
}

/// Rows of 12, 7 and 3 tokens, right-padded, and their mask.
fn rows(dev: &Device) -> Result<(Tensor, Tensor)> {
    let lens = [12usize, 7, 3];
    let ids: Vec<u32> = lens
        .iter()
        .enumerate()
        .flat_map(|(r, &n)| {
            (0..12).map(move |t| {
                if t < n {
                    1 + ((r * 13 + t * 7) % 49) as u32
                } else {
                    0
                }
            })
        })
        .collect();
    let mask: Vec<u32> = lens
        .iter()
        .flat_map(|&n| (0..12).map(move |t| u32::from(t < n)))
        .collect();
    Ok((
        Tensor::from_vec(ids, (3, 12), dev)?,
        Tensor::from_vec(mask, (3, 12), dev)?,
    ))
}

/// The packed pass over each row's real tokens gives, at every real position, what the padded
/// pass gives: to rounding with F32 projections, to quantization with Q8_0 blocks; its pad
/// positions are zero.
#[test]
fn the_packed_pass_is_the_padded_pass() -> Result<()> {
    let dev = Device::Cpu;
    let vm = VarMap::new();
    let bert = ModernBert::new(VarBuilder::from_varmap(&vm, DType::F32, &dev), &config())?;
    let (ids, mask) = rows(&dev)?;
    let keep = mask.to_dtype(DType::F32)?.unsqueeze(2)?;
    let want = bert.forward(&ids, &mask)?.broadcast_mul(&keep)?;
    let scale = want.abs()?.max_all()?.to_scalar::<f32>()?;
    for (dtype, tol) in [(GgmlDType::F32, 1e-5f32), (GgmlDType::Q8_0, 0.05)] {
        let mut q = bert.clone();
        q.quantize(dtype)?;
        let got = q.forward(&ids, &mask)?;
        let pads = got
            .broadcast_mul(&keep.affine(-1.0, 1.0)?)?
            .abs()?
            .max_all()?
            .to_scalar::<f32>()?;
        assert_eq!(pads, 0.0);
        let gap = (got - &want)?.abs()?.max_all()?.to_scalar::<f32>()?;
        assert!(gap <= tol * scale, "{dtype:?}: {gap} against {scale}");
    }
    Ok(())
}
