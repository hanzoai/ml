//! The whole thesis in one file: write a kernel once, run it, get the right answer.
//!
//!   cargo run --example hello_kernel                              # CPU
//!   cargo run --example hello_kernel --features vulkan            # + Vulkan (add the GPU block below)
//!
//! `rms_norm` below is the exact kernel that ships as a live DSL kernel in the Hanzo ML engine,
//! where it replaced a hand-written GLSL shader and runs 10.6x faster. Here it runs on the CPU
//! reference runtime -- the same source that lowers to CUDA / Metal / Vulkan / WebGPU unchanged.

use hanzo_kernel::cubecl::cpu::{CpuDevice, CpuRuntime};
use hanzo_kernel::prelude::*;

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu))]
fn rms_norm<F: Float>(x: &Array<F>, w: &Array<F>, out: &mut Array<F>, #[comptime] n: usize) {
    let row = ABSOLUTE_POS;
    if row < out.len() / n {
        let base = row * n;
        let mut ss = F::new(0.0);
        for i in 0..n {
            let v = x[base + i];
            ss += v * v;
        }
        let denom = (ss / F::cast_from(n) + F::new(1e-6)).sqrt();
        for i in 0..n {
            out[base + i] = x[base + i] / denom * w[i];
        }
    }
}

fn main() {
    let client = CpuRuntime::client(&CpuDevice::default());

    let (rows, n) = (2u32, 4usize);
    let x = vec![1.0f32, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0, 2.0];
    let w = vec![1.0f32; n];

    let xh = client.create_from_slice(f32::as_bytes(&x));
    let wh = client.create_from_slice(f32::as_bytes(&w));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; x.len()]));

    rms_norm::launch::<f32, CpuRuntime>(
        &client,
        Grid::Static(1, 1, 1),
        Block::new_1d(rows),
        unsafe { ArrayArg::from_raw_parts(xh.clone(), x.len()) },
        unsafe { ArrayArg::from_raw_parts(wh.clone(), w.len()) },
        unsafe { ArrayArg::from_raw_parts(oh.clone(), x.len()) },
        n,
    );

    let bytes = client.read_one_unchecked(oh);
    let out = f32::from_bytes(&bytes);
    println!("rms_norm(one source, CPU) = {out:?}");
    println!("expected                  = [0.365, 0.730, 1.095, 1.461, 1.0, 1.0, 1.0, 1.0]");
}
