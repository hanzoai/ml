//! Every ROCm kernel source the crate ships compiles for this GPU. The kernels are compiled from
//! source at first use, so a source that hipcc rejects fails at the first forward that needs it,
//! not at build time; this is the build-time check.
#![cfg(feature = "rocm")]

use hanzo_rocm_kernels::kernel::KernelSource;

fn compiles<K: KernelSource>() {
    let dir = std::env::temp_dir();
    let src = dir.join(format!("hanzo_{}_compile_check.hip", K::NAME));
    let out = dir.join(format!("hanzo_{}_compile_check.hsaco", K::NAME));
    std::fs::write(&src, K::CODE).unwrap();
    let arch = std::env::var("HANZO_ROCM_ARCH").unwrap_or_else(|_| "gfx1151".into());
    let status = std::process::Command::new("hipcc")
        .args(["--genco", "-O1", &format!("--offload-arch={arch}"), "-o"])
        .arg(&out)
        .arg(&src)
        .status()
        .expect("hipcc on PATH");
    assert!(status.success(), "{} does not compile", K::NAME);
}

#[test]
fn every_kernel_source_compiles() {
    use hanzo_rocm_kernels::kernel::*;
    compiles::<BinaryKernel>();
    compiles::<UnaryKernel>();
    compiles::<AffineKernel>();
    compiles::<FillKernel>();
    compiles::<ReduceKernel>();
    compiles::<ConvKernel>();
    compiles::<IndexingKernel>();
    compiles::<CastKernel>();
    compiles::<TernaryKernel>();
    compiles::<SortKernel>();
    compiles::<QuantKernel>();
    compiles::<RopeKernel>();
    compiles::<FlashKernel>();
    compiles::<DslRmsNormKernel>();
    compiles::<DslAddRmsNormKernel>();
    compiles::<GdnKernel>();
}
