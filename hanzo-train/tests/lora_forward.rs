//! Pin `TrainableLoraLinear::forward` against the closed-form PEFT
//! LoRA equation:
//!
//! ```text
//! y = base(x) + (alpha / r) · B @ A @ x
//! ```
//!
//! We instantiate a `TrainableLoraLinear` with hand-picked tensors,
//! compute both sides, and assert they agree to better than 1e-6 in f32
//! (well within bf16/f16 tolerance).

use candle::{DType, Device, Module, Tensor};
use candle_nn::Linear;

use hanzo_train::{LoraConfig, TrainableLoraLinear};

const IN_F: usize = 8;
const OUT_F: usize = 12;
const RANK: usize = 4;
const ALPHA: f64 = 8.0;
const SEQ: usize = 5;
const BATCH: usize = 3;

#[test]
fn lora_forward_matches_peft_equation() -> candle::Result<()> {
    let device = Device::Cpu;

    // ----- Deterministic base weight + bias. -----
    let base_w = Tensor::randn(0f32, 0.1f32, (OUT_F, IN_F), &device)?;
    let base_b = Tensor::randn(0f32, 0.1f32, OUT_F, &device)?;
    let base = Linear::new(base_w.clone(), Some(base_b.clone()));

    // ----- Construct LoRA with zero rank-init then overwrite to known.
    let cfg = LoraConfig {
        rank: RANK,
        alpha: ALPHA,
        dropout: 0.0,
        target_modules: vec!["any".into()],
    };
    let lora = TrainableLoraLinear::new(base, &cfg)?;

    // Replace A and B with known values so the comparison is exact.
    let a_known = Tensor::randn(0f32, 0.05f32, (RANK, IN_F), &device)?;
    let b_known = Tensor::randn(0f32, 0.05f32, (OUT_F, RANK), &device)?;
    lora.lora_a().set(&a_known)?;
    lora.lora_b().set(&b_known)?;

    // ----- Input batch.
    let x = Tensor::randn(0f32, 1f32, (BATCH, SEQ, IN_F), &device)?;

    // ----- Forward through the wrapper (eval mode -> no dropout).
    let got = lora.forward(&x)?;

    // ----- Reference: base(x) + scale * B @ A @ x   (right-multiply form).
    let scale = ALPHA / RANK as f64;
    // base_out: x @ base_w^T + bias
    let x2 = x.reshape((BATCH * SEQ, IN_F))?;
    let base_out = x2.matmul(&base_w.t()?)?.broadcast_add(&base_b)?;
    let base_out = base_out.reshape((BATCH, SEQ, OUT_F))?;
    // lora_out: x @ A^T @ B^T * scale
    let xa = x2.matmul(&a_known.t()?)?;
    let xab = xa.matmul(&b_known.t()?)?;
    let lora_out = (xab * scale)?.reshape((BATCH, SEQ, OUT_F))?;
    let want = base_out.add(&lora_out)?;

    let got_v: Vec<f32> = got.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let want_v: Vec<f32> = want.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    assert_eq!(got_v.len(), want_v.len());
    let mut max_abs = 0f32;
    for (g, w) in got_v.iter().zip(want_v.iter()) {
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    eprintln!(
        "LoRA forward max abs diff vs PEFT equation: {:.3e}",
        max_abs
    );
    assert!(
        max_abs < 1e-5,
        "max abs diff {max_abs} exceeded 1e-5 (bf16 tol = ~1e-2 so this is generous)"
    );
    Ok(())
}

#[test]
fn lora_forward_zero_at_init() -> candle::Result<()> {
    // PEFT default: B starts at zero, so the wrapper's first forward
    // must equal the base's forward exactly.
    let device = Device::Cpu;
    let base_w = Tensor::randn(0f32, 0.1f32, (OUT_F, IN_F), &device)?;
    let base_b = Tensor::randn(0f32, 0.1f32, OUT_F, &device)?;
    let base = Linear::new(base_w.clone(), Some(base_b.clone()));
    let base_for_ref = Linear::new(base_w, Some(base_b));

    let cfg = LoraConfig {
        rank: RANK,
        alpha: ALPHA,
        dropout: 0.0,
        target_modules: vec!["any".into()],
    };
    let lora = TrainableLoraLinear::new(base, &cfg)?;

    let x = Tensor::randn(0f32, 1f32, (BATCH, SEQ, IN_F), &device)?;
    let got = lora.forward(&x)?;
    let want = base_for_ref.forward(&x)?;

    let got_v: Vec<f32> = got.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let want_v: Vec<f32> = want.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
    let mut max_abs = 0f32;
    for (g, w) in got_v.iter().zip(want_v.iter()) {
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs < 1e-6,
        "B=0 invariant broken: got {max_abs} vs base"
    );
    Ok(())
}
