//! hanzo-nn's custom ops on a ROCm device against their formulas.
#![cfg(feature = "rocm")]

use hanzo_ml::{DType, Device, Result, Tensor};

/// `ops::sigmoid` is one kernel on ROCm, computed in f32 and rounded once to the tensor's dtype.
#[test]
fn sigmoid_matches_the_formula_in_every_float_dtype() -> Result<()> {
    let gpu = Device::new_rocm(0)?;
    let x: Vec<f32> = (0..4099).map(|i| ((i * 7919) % 2001) as f32 / 125.0 - 8.0).collect();
    for (dtype, bound) in [(DType::F32, 1e-6), (DType::F16, 1e-3), (DType::BF16, 1e-2)] {
        let xg = Tensor::from_vec(x.clone(), x.len(), &gpu)?.to_dtype(dtype)?;
        // The kernel sees the input rounded to its dtype.
        let seen = xg.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        let got = hanzo_nn::ops::sigmoid(&xg)?
            .to_dtype(DType::F32)?
            .to_vec1::<f32>()?;
        let worst = got
            .iter()
            .zip(&seen)
            .map(|(g, v)| (*g as f64 - 1.0 / (1.0 + (-*v as f64).exp())).abs())
            .fold(0f64, f64::max);
        println!("sigmoid {dtype:?}: {worst:.2e}");
        assert!(worst < bound, "sigmoid {dtype:?} off by {worst}");
    }
    // A strided view goes through the same kernel.
    let m = Tensor::from_vec(x[..4096].to_vec(), (64, 64), &gpu)?;
    let got = hanzo_nn::ops::sigmoid(&m.t()?)?;
    let want = hanzo_nn::ops::sigmoid(&m.t()?.to_device(&Device::Cpu)?)?;
    let diff = (got.to_device(&Device::Cpu)? - want)?
        .abs()?
        .max_all()?
        .to_scalar::<f32>()?;
    assert!(diff < 1e-6, "transposed sigmoid off by {diff}");
    Ok(())
}
