//! The fused AdamW step and gradient norm equal the tensor-op path on the CPU.
#![cfg(any(feature = "metal", feature = "cuda"))]

use hanzo_ml::{DType, Device, Result, Tensor, Var};
use hanzo_nn::optim::{grad_norm, AdamW, ParamsAdamW};
use hanzo_nn::Optimizer;

fn gpu() -> Result<Device> {
    #[cfg(feature = "metal")]
    return Device::new_metal(0);
    #[cfg(not(feature = "metal"))]
    Device::new_cuda(0)
}

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
    let metal = gpu()?;
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
    let metal = gpu()?;
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

/// One step of the fused kernel against the same step in tensor ops on the same device.
#[test]
fn adamw_one_step_is_the_composite() -> Result<()> {
    use hanzo_nn::optim::{adamw, Update};
    let dev = gpu()?;
    let shape = (1025, 769);
    let theta = Tensor::randn(0f32, 1f32, shape, &dev)?;
    let g = (Tensor::randn(0f32, 1f32, shape, &dev)? * 3.0)?;
    let m = (Tensor::randn(0f32, 1f32, shape, &dev)? * 0.1)?;
    let v = Tensor::randn(0f32, 1f32, shape, &dev)?.sqr()?;
    let u = Update {
        lr: 2e-5,
        beta1: 0.9,
        beta2: 0.98,
        eps: 1e-6,
        weight_decay: 0.01,
        scale_m: 1.0 / (1.0 - 0.9f64.powi(7)),
        scale_v: 1.0 / (1.0 - 0.98f64.powi(7)),
        grad_scale: 0.37,
    };
    let (vt, vm, vv) = (
        Var::from_tensor(&theta)?,
        Var::from_tensor(&m)?,
        Var::from_tensor(&v)?,
    );
    adamw(&vt, &g, &vm, &vv, &u)?;
    let g = (g * u.grad_scale)?;
    let m = ((m * u.beta1)? + (&g * (1.0 - u.beta1))?)?;
    let v = ((v * u.beta2)? + (g.sqr()? * (1.0 - u.beta2))?)?;
    let step = ((&m * u.scale_m)? / ((&v * u.scale_v)?.sqrt()? + u.eps)?)?;
    let theta = ((theta * (1.0 - u.lr * u.weight_decay))? - (step * u.lr)?)?;
    for (name, fused, composite) in [
        ("theta", vt.as_tensor(), &theta),
        ("m", vm.as_tensor(), &m),
        ("v", vv.as_tensor(), &v),
    ] {
        let d = (fused - composite)?
            .abs()?
            .flatten_all()?
            .max(0)?
            .to_scalar::<f32>()?;
        let s = composite.abs()?.flatten_all()?.max(0)?.to_scalar::<f32>()?;
        println!("adamw one step {name}: max |Δ| {d:.2e} at scale {s:.2}");
        assert!(d <= 1e-6 * s, "{name}: {d} at scale {s}");
    }
    Ok(())
}
