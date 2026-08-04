//! The whole thesis in one file: write a kernel ONCE, run it on every accelerator, get the right answer.
//!
//!   cargo run --release --example hello_kernel                  # CPU oracle only
//!   cargo run --release --example hello_kernel --features rocm   # + AMD ROCm/HIP
//!   cargo run --release --example hello_kernel --features cuda   # + NVIDIA CUDA
//!   cargo run --release --example hello_kernel --features metal  # + Apple Metal
//!   cargo run --release --example hello_kernel --features vulkan # + Vulkan (SPIR-V)
//!
//! `rms_norm` is a PRODUCTION DSL kernel that lives in `hanzo_kernel::norm`, where it replaced a
//! hand-written GLSL shader. This example does not restate its body -- it IMPORTS it. So there is
//! exactly ONE `rms_norm` source in the repo, and every backend below lowers those same bytes.
//!
//! The inputs are generated deterministically, so the printed checksum is comparable ACROSS boxes:
//! the same number on an AMD APU, an NVIDIA GPU and an Apple GPU is the falsifiable claim that one
//! source computed one answer on three architectures.

use hanzo_kernel::norm::{rms_norm_ref, rms_norm_run};
use hanzo_kernel::prelude::*;

/// Shape and epsilon are fixed so every box runs the identical problem.
const ROWS: usize = 37;
const N: usize = 128;
const EPS: f32 = 1e-5;

/// Deterministic inputs (xorshift64) -- identical bytes on every box, no RNG seeding to get wrong.
fn data() -> (Vec<f32>, Vec<f32>) {
    let mut s = 0x2545F491_4F6CDD1Du64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s % 2000) as f32 / 1000.0 - 1.0
    };
    let x = (0..ROWS * N).map(|_| next()).collect();
    let w = (0..N).map(|_| next() * 0.5 + 1.0).collect();
    (x, w)
}

fn max_rel(want: &[f32], got: &[f32]) -> f32 {
    want.iter()
        .zip(got)
        .map(|(a, b)| (a - b).abs() / a.abs().max(1e-6))
        .fold(0.0, f32::max)
}

/// The whole proof, over ANY runtime. `Target::of` reports which backend the DSL lowered to, so the
/// output names the architecture it actually ran on rather than the one we hoped for.
fn run<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let (x, w) = data();
    let got = rms_norm_run::<R>(client, &x, &w, ROWS, N, EPS);
    let want = rms_norm_ref(&x, &w, ROWS, N, EPS);

    let rel = max_rel(&want, &got);
    let ok = rel < 2e-3;
    // f64 accumulation so the checksum reflects the kernel's output, not the summation order.
    let checksum: f64 = got.iter().map(|&v| v as f64).sum();

    println!(
        "target={:?}  runtime={}  {}x{}",
        Target::of(client),
        R::name(client),
        ROWS,
        N
    );
    println!("  out[0..4]    = {:?}", &got[..4]);
    println!("  oracle[0..4] = {:?}", &want[..4]);
    println!(
        "  checksum     = {checksum:.6}   max_rel = {rel:.3e}   {}",
        if ok { "MATCH" } else { "MISMATCH" }
    );
    ok
}

fn main() {
    let mut ran = 0usize;
    let mut ok = true;

    // The CPU runtime is the oracle target: every `island!` resolves to its normative `default` arm,
    // so it is the reference every accelerator below is compared against.
    #[cfg(feature = "cpu")]
    {
        use hanzo_kernel::cubecl::cpu::{CpuDevice, CpuRuntime};
        ok &= run::<CpuRuntime>(&CpuRuntime::client(&CpuDevice::default()));
        ran += 1;
    }
    #[cfg(feature = "rocm")]
    {
        use hanzo_cubecl_hip::{AmdDevice, HipRuntime};
        ok &= run::<HipRuntime>(&HipRuntime::client(&AmdDevice::default()));
        ran += 1;
    }
    #[cfg(feature = "cuda")]
    {
        use hanzo_kernel::cubecl::cuda::{CudaDevice, CudaRuntime};
        ok &= run::<CudaRuntime>(&CudaRuntime::client(&CudaDevice::default()));
        ran += 1;
    }
    // Metal and Vulkan are both the wgpu runtime; the feature picks the backend it compiles to, and
    // `Target::of` reports which one ("wgpu<msl>" vs "wgpu<spirv>") rather than us asserting it.
    #[cfg(any(feature = "metal", feature = "vulkan"))]
    {
        use hanzo_kernel::cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        ok &= run::<WgpuRuntime>(&WgpuRuntime::client(&WgpuDevice::default()));
        ran += 1;
    }

    println!(
        "\n{ran} backend(s) ran from ONE kernel source: {}",
        if ok { "all MATCH" } else { "MISMATCH" }
    );
    if !ok {
        std::process::exit(1);
    }
}
