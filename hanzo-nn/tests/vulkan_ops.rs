//! hanzo-nn ops on the Vulkan device against the CPU. Skips when no Vulkan GPU is present.
#![cfg(feature = "vulkan")]

use hanzo_ml::{Device, Tensor};

#[test]
fn sigmoid_matches_cpu() -> hanzo_ml::Result<()> {
    let dev = match Device::new_vulkan(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("no Vulkan GPU ({e}); skipping");
            return Ok(());
        }
    };
    let x: Vec<f32> = (0..3000).map(|i| (i as f32 - 1500.0) / 97.0).collect();
    // A transposed view, so the kernel reads a strided input made contiguous first.
    let on = |d: &Device| -> hanzo_ml::Result<Vec<f32>> {
        let t = Tensor::from_vec(x.clone(), (1000, 3), d)?.t()?;
        hanzo_nn::ops::sigmoid(&t)?.flatten_all()?.to_vec1::<f32>()
    };
    let (want, got) = (on(&Device::Cpu)?, on(&dev)?);
    let err = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(err < 1e-6, "sigmoid off by {err}");
    Ok(())
}
