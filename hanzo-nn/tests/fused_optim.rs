//! The fused Metal AdamW step and gradient norm equal the tensor-op path on the CPU.
#![cfg(feature = "metal")]

use hanzo_ml::{DType, Device, Result, Tensor, Var};
use hanzo_nn::optim::{grad_norm, AdamW, ParamsAdamW};
use hanzo_nn::Optimizer;

fn params() -> ParamsAdamW {
    ParamsAdamW {
        lr: 3e-3,
        beta1: 0.9,
        beta2: 0.98,
        eps: 1e-6,
        weight_decay: 0.05,
    }
}

/// Runs `steps` AdamW steps from `w0` with the gradient of `Σ w ⊙ c` scaled by `scale`.
fn run(dev: &Device, w0: &Tensor, c: &Tensor, steps: usize, scale: f64) -> Result<(Vec<f32>, f64)> {
    let w = Var::from_tensor(&w0.to_device(dev)?)?;
    let c = c.to_device(dev)?;
    let mut opt = AdamW::new(vec![w.clone()], params())?;
    let mut norm = 0.0;
    for _ in 0..steps {
        let loss = (w.as_tensor().sqr()? * &c)?.sum_all()?;
        let grads = loss.backward()?;
        norm = grad_norm(&grads, &[w.clone()])?;
        opt.step_scaled(&grads, scale)?;
    }
    Ok((w.as_tensor().flatten_all()?.to_vec1::<f32>()?, norm))
}

#[test]
fn adamw_fused_matches_tensor_path() -> Result<()> {
    let metal = Device::new_metal(0)?;
    // an odd length: the kernel's last threadgroup is partial
    let w0 = Tensor::randn(0f32, 1f32, (37, 129), &Device::Cpu)?;
    let c = Tensor::randn(0f32, 1f32, (37, 129), &Device::Cpu)?;
    for scale in [1.0, 0.25] {
        let (cpu, n_cpu) = run(&Device::Cpu, &w0, &c, 5, scale)?;
        let (gpu, n_gpu) = run(&metal, &w0, &c, 5, scale)?;
        let d = cpu
            .iter()
            .zip(&gpu)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        assert!(d < 1e-5, "scale {scale}: parameters differ by {d}");
        assert!(
            (n_cpu - n_gpu).abs() / n_cpu < 1e-5,
            "norm {n_cpu} vs {n_gpu}"
        );
    }
    Ok(())
}

#[test]
fn grad_norm_mixes_fused_and_other_dtypes() -> Result<()> {
    let metal = Device::new_metal(0)?;
    let a = Var::from_tensor(&Tensor::randn(0f32, 1f32, 1000, &metal)?)?;
    let b = Var::from_tensor(&Tensor::randn(0f32, 1f32, 300, &metal)?.to_dtype(DType::BF16)?)?;
    let loss = (a.as_tensor().sum_all()? * 3.0)?
        .add(&(b.as_tensor().to_dtype(DType::F32)?.sum_all()? * 2.0)?)?;
    let grads = loss.backward()?;
    let n = grad_norm(&grads, &[a, b])?;
    let want = (1000.0 * 9.0 + 300.0 * 4.0f64).sqrt();
    assert!((n - want).abs() / want < 1e-5, "{n} vs {want}");
    Ok(())
}
