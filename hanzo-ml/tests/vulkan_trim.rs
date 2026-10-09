//! A trim gives Vulkan's idle pool back to the driver, keeping 1 GiB, and the device computes
//! as before it. Skips with no GPU.
#![cfg(feature = "vulkan")]

use hanzo_ml::{Device, Result, Tensor};

#[test]
fn a_trim_gives_the_idle_pool_back() -> Result<()> {
    let dev = match Device::new_vulkan(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[trim] no Vulkan GPU ({e}); skipping");
            return Ok(());
        }
    };
    let vk = dev.as_vulkan_device()?;
    // three blocks of 512 MiB held, then dropped: idle in the pool once the device has run past
    let big = (0..3)
        .map(|_| Tensor::ones(1 << 27, hanzo_ml::DType::F32, &dev))
        .collect::<Result<Vec<_>>>()?;
    for b in &big {
        assert_eq!(b.narrow(0, (1 << 27) - 4, 4)?.to_vec1::<f32>()?, [1.0; 4]);
    }
    drop(big);
    dev.synchronize()?;
    assert!(vk.pool_free_bytes() >= 3 << 29, "{} idle", vk.pool_free_bytes());
    dev.trim();
    assert!(vk.pool_free_bytes() <= 1 << 30, "{} idle after a trim", vk.pool_free_bytes());
    let x = Tensor::arange(0f32, 8., &dev)?;
    assert_eq!((&x + &x)?.to_vec1::<f32>()?, (0..8).map(|i| 2. * i as f32).collect::<Vec<_>>());
    Ok(())
}
