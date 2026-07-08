//! Falsifiable parity gate for the I-JEPA encoder against `transformers.IJepaModel`.
//!
//! `ijepa_tiny_matches_hf` (committed, no network) loads a tiny 2-layer I-JEPA whose
//! weights + reference output came from the real Hugging Face `IJepaModel`, and asserts
//! per-patch **absolute** parity to 1e-4 fp32 on CPU. With O(1) activations and only two
//! layers this isolates arithmetic/layout/naming correctness from accumulation depth —
//! it matches to ~7e-7, i.e. exactly.
//!
//! `ijepa_real_vith14_matches_hf` (`#[ignore]`, needs the 2.5GB checkpoint) runs the real
//! `facebook/ijepa_vith14_1k` ViT-H/14. Its `last_hidden_state` reaches magnitude ~21, so
//! the meaningful gate for a 32-layer fp32 comparison across two different GEMM backends
//! (torch/oneDNN vs the `gemm` crate) is **relative** max error < 1e-4 and cosine ≈ 1;
//! the absolute max diff (~1.7e-3, i.e. 8e-5 relative) is backend accumulation, not a bug.
//! Run with:
//!   IJEPA_REAL_WEIGHTS=.../model.safetensors IJEPA_REAL_IO=.../real_io.safetensors \
//!     cargo test -p hanzo-transformers --test ijepa -- --ignored --nocapture

use std::collections::HashMap;
use std::fmt;

use hanzo_ml::{DType, Device, Result, Tensor};
use hanzo_nn::VarBuilder;
use hanzo_transformers::models::ijepa::{Config, IJepaModel};

const TINY_FIXTURE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/ijepa_tiny.safetensors"
);

fn tiny_config() -> Config {
    Config {
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        hidden_act: hanzo_nn::Activation::Gelu,
        layer_norm_eps: 1e-6,
        image_size: 28,
        patch_size: 14,
        num_channels: 3,
        qkv_bias: true,
    }
}

struct Metrics {
    max_abs: f32,
    mean_abs: f32,
    rel: f32,
    cosine: f32,
}

impl fmt::Display for Metrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "max_abs={:.3e} mean_abs={:.3e} rel={:.3e} cosine={:.9}",
            self.max_abs, self.mean_abs, self.rel, self.cosine
        )
    }
}

fn compare(out: &Tensor, reference: &Tensor) -> Result<Metrics> {
    let a = out.flatten_all()?.to_vec1::<f32>()?;
    let b = reference.flatten_all()?.to_vec1::<f32>()?;
    assert_eq!(a.len(), b.len(), "shape mismatch in comparison");
    let mut max_abs = 0f32;
    let mut sum_abs = 0f64;
    let mut max_ref = 0f32;
    let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
        max_ref = max_ref.max(y.abs());
        dot += (x * y) as f64;
        na += (x * x) as f64;
        nb += (y * y) as f64;
    }
    Ok(Metrics {
        max_abs,
        mean_abs: (sum_abs / a.len() as f64) as f32,
        rel: max_abs / max_ref.max(f32::MIN_POSITIVE),
        cosine: (dot / (na.sqrt() * nb.sqrt())) as f32,
    })
}

/// Runs the encoder against the stored HF reference and returns
/// (per-patch metrics, mean-pooled metrics).
fn run(
    weights: &str,
    io: &HashMap<String, Tensor>,
    cfg: &Config,
    device: &Device,
) -> Result<(Metrics, Metrics)> {
    let pixel_values = io
        .get("__io.pixel_values")
        .expect("pixel_values in fixture");
    let ref_lhs = io
        .get("__io.ref_last_hidden_state")
        .expect("ref_last_hidden_state in fixture");
    let ref_pooled = io.get("__io.ref_pooled").expect("ref_pooled in fixture");

    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[weights], DType::F32, device)? };
    let model = IJepaModel::new(cfg, vb)?;

    let out = model.forward(pixel_values)?;
    assert_eq!(out.dims(), ref_lhs.dims(), "last_hidden_state shape != HF");
    let lhs = compare(&out, ref_lhs)?;

    let pooled = model.forward_pooled(pixel_values)?;
    let pooled_m = compare(&pooled, ref_pooled)?;
    Ok((lhs, pooled_m))
}

#[test]
fn ijepa_tiny_matches_hf() -> Result<()> {
    let device = Device::Cpu;
    let io = hanzo_ml::safetensors::load(TINY_FIXTURE, &device)?;
    let (lhs, pooled) = run(TINY_FIXTURE, &io, &tiny_config(), &device)?;
    println!("tiny per-patch: {lhs}");
    println!("tiny pooled:    {pooled}");
    // O(1) activations, 2 layers -> exact match well inside the absolute 1e-4 gate.
    assert!(
        lhs.max_abs < 1e-4,
        "per-patch max_abs {:.3e} >= 1e-4",
        lhs.max_abs
    );
    assert!(
        pooled.max_abs < 1e-4,
        "pooled max_abs {:.3e} >= 1e-4",
        pooled.max_abs
    );
    Ok(())
}

#[test]
#[ignore = "requires the 2.5GB facebook/ijepa_vith14_1k checkpoint via IJEPA_REAL_* env vars"]
fn ijepa_real_vith14_matches_hf() -> Result<()> {
    let weights = std::env::var("IJEPA_REAL_WEIGHTS").expect("set IJEPA_REAL_WEIGHTS");
    let io_path = std::env::var("IJEPA_REAL_IO").expect("set IJEPA_REAL_IO");
    let device = Device::Cpu;
    let io = hanzo_ml::safetensors::load(&io_path, &device)?;
    let (lhs, pooled) = run(&weights, &io, &Config::vit_huge_patch14_224(), &device)?;
    println!("vith14 per-patch: {lhs}");
    println!("vith14 pooled:    {pooled}");
    // 32 layers, activations up to ~21: gate on relative error + cosine (cross-backend fp32).
    assert!(
        lhs.rel < 1e-4,
        "per-patch relative error {:.3e} >= 1e-4",
        lhs.rel
    );
    assert!(
        lhs.cosine > 0.99999,
        "per-patch cosine {:.9} <= 0.99999",
        lhs.cosine
    );
    assert!(
        pooled.rel < 1e-4,
        "pooled relative error {:.3e} >= 1e-4",
        pooled.rel
    );
    Ok(())
}
