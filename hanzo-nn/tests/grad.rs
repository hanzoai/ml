//! The fused ops take part in autograd: each one's gradients equal those of its unfused
//! composite, on every device the fused kernel runs on.

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

use hanzo_ml::{test_device, DType, Device, Result, Tensor, Var};
use hanzo_nn::{ops, rotary_emb, Module};

/// Gradients of `Σ f(x…) ⊙ w` with respect to every input, for a fixed random `w`.
fn grads(inputs: &[&Var], y: &Tensor, w: &Tensor) -> Result<Vec<Tensor>> {
    let g = (y * w)?.sum_all()?.backward()?;
    inputs
        .iter()
        .map(|v| {
            g.get(v.as_tensor())
                .cloned()
                .ok_or_else(|| hanzo_ml::Error::Msg("no gradient reached an input".into()))
        })
        .collect()
}

fn max_diff(a: &Tensor, b: &Tensor) -> Result<f32> {
    (a.to_dtype(DType::F32)? - b.to_dtype(DType::F32)?)?
        .abs()?
        .flatten_all()?
        .max(0)?
        .to_scalar::<f32>()
}

fn assert_close(fused: &[Tensor], slow: &[Tensor], tol: f32, what: &str) -> Result<()> {
    assert_eq!(fused.len(), slow.len());
    for (i, (a, b)) in fused.iter().zip(slow).enumerate() {
        let d = max_diff(a, b)?;
        assert!(d < tol, "{what}: input {i} gradient differs by {d}");
    }
    Ok(())
}

fn var(shape: &[usize], device: &Device) -> Result<Var> {
    Var::from_tensor(&Tensor::randn(0f32, 1f32, shape, device)?)
}

fn softmax_last_dim(device: &Device) -> Result<()> {
    let x = var(&[3, 4, 37], device)?;
    let w = Tensor::randn(0f32, 1f32, (3, 4, 37), device)?;
    let fused = grads(&[&x], &ops::softmax_last_dim(x.as_tensor())?, &w)?;
    let slow = grads(&[&x], &ops::softmax(x.as_tensor(), 2)?, &w)?;
    assert_close(&fused, &slow, 1e-5, "softmax_last_dim")
}

fn layer_norm(device: &Device) -> Result<()> {
    let (x, a, b) = (
        var(&[2, 5, 64], device)?,
        var(&[64], device)?,
        var(&[64], device)?,
    );
    let w = Tensor::randn(0f32, 1f32, (2, 5, 64), device)?;
    let ins = [&x, &a, &b];
    let fused = grads(
        &ins,
        &ops::layer_norm(x.as_tensor(), a.as_tensor(), b.as_tensor(), 1e-5)?,
        &w,
    )?;
    let slow = grads(
        &ins,
        &ops::layer_norm_slow(x.as_tensor(), a.as_tensor(), b.as_tensor(), 1e-5)?,
        &w,
    )?;
    assert_close(&fused, &slow, 1e-4, "layer_norm")
}

fn layer_norm_no_bias(device: &Device) -> Result<()> {
    let (x, a) = (var(&[4, 7, 32], device)?, var(&[32], device)?);
    let w = Tensor::randn(0f32, 1f32, (4, 7, 32), device)?;
    let ln = hanzo_nn::LayerNorm::new_no_bias(a.as_tensor().clone(), 1e-5);
    let fused = grads(&[&x, &a], &ln.forward(x.as_tensor())?, &w)?;
    let zero = a.as_tensor().zeros_like()?;
    let slow = grads(
        &[&x, &a],
        &ops::layer_norm_slow(x.as_tensor(), a.as_tensor(), &zero, 1e-5)?,
        &w,
    )?;
    assert_close(&fused, &slow, 1e-4, "layer_norm (no bias)")
}

fn rms_norm(device: &Device) -> Result<()> {
    let (x, a) = (var(&[3, 6, 48], device)?, var(&[48], device)?);
    let w = Tensor::randn(0f32, 1f32, (3, 6, 48), device)?;
    let fused = grads(
        &[&x, &a],
        &ops::rms_norm(x.as_tensor(), a.as_tensor(), 1e-5)?,
        &w,
    )?;
    let slow = grads(
        &[&x, &a],
        &ops::rms_norm_slow(x.as_tensor(), a.as_tensor(), 1e-5)?,
        &w,
    )?;
    assert_close(&fused, &slow, 1e-4, "rms_norm")
}

fn silu_mul(device: &Device) -> Result<()> {
    let (g, u) = (var(&[5, 33], device)?, var(&[5, 33], device)?);
    let w = Tensor::randn(0f32, 1f32, (5, 33), device)?;
    let fused = grads(&[&g, &u], &ops::silu_mul(g.as_tensor(), u.as_tensor())?, &w)?;
    let slow = grads(&[&g, &u], &(g.as_tensor().silu()? * u.as_tensor())?, &w)?;
    assert_close(&fused, &slow, 1e-5, "silu_mul")
}

fn rope(device: &Device) -> Result<()> {
    let (b, h, t, d) = (2, 3, 11, 16);
    let x = var(&[b, h, t, d], device)?;
    // tables longer than the sequence, as a model builds them
    let cos = Tensor::randn(0f32, 1f32, (t + 5, d / 2), device)?;
    let sin = Tensor::randn(0f32, 1f32, (t + 5, d / 2), device)?;
    let w = Tensor::randn(0f32, 1f32, (b, h, t, d), device)?;
    let fused = grads(&[&x], &rotary_emb::rope(x.as_tensor(), &cos, &sin)?, &w)?;
    let slow = grads(
        &[&x],
        &rotary_emb::rope_slow(x.as_tensor(), &cos, &sin)?,
        &w,
    )?;
    assert_close(&fused, &slow, 1e-5, "rope")?;

    let fused = grads(&[&x], &rotary_emb::rope_i(x.as_tensor(), &cos, &sin)?, &w)?;
    let slow = grads(
        &[&x],
        &rotary_emb::rope_i_slow(x.as_tensor(), &cos, &sin)?,
        &w,
    )?;
    assert_close(&fused, &slow, 1e-5, "rope_i")?;

    // (b, t, h, d): the same rotation with the head and time axes swapped
    let xt = var(&[b, t, h, d], device)?;
    let wt = w.transpose(1, 2)?.contiguous()?;
    let fused = grads(
        &[&xt],
        &rotary_emb::rope_thd(xt.as_tensor(), &cos, &sin)?,
        &wt,
    )?;
    let slow = rotary_emb::rope_slow(&xt.as_tensor().transpose(1, 2)?.contiguous()?, &cos, &sin)?
        .transpose(1, 2)?;
    let slow = grads(&[&xt], &slow, &wt)?;
    assert_close(&fused, &slow, 1e-5, "rope_thd")
}

test_device!(softmax_last_dim, softmax_cpu, softmax_gpu, softmax_metal);
test_device!(layer_norm, ln_cpu, ln_gpu, ln_metal);
test_device!(layer_norm_no_bias, lnnb_cpu, lnnb_gpu, lnnb_metal);
test_device!(rms_norm, rms_cpu, rms_gpu, rms_metal);
test_device!(silu_mul, silu_mul_cpu, silu_mul_gpu, silu_mul_metal);
test_device!(rope, rope_cpu, rope_gpu, rope_metal);
