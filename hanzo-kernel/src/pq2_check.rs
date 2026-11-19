//! pq2-check: the per-backend correctness gate for the native-layout PQ2_0 DSL matvec.
//!
//!   pq2-check --backend metal|cuda|rocm|vulkan
//!
//! Same fixtures (`PQ2_GGUF_SHAPES`, synthetic GGUF bytes, the f64 oracle that decodes exactly like
//! hanzo-ml's `BlockPQ2_0::to_float`) on every backend; each backend passes or fails on its own.
//! Exit 0 only if every case is within REL_GATE scale-relative error.

use hanzo_kernel::prelude::*;
use hanzo_kernel::ternary::pq2_0_gguf_board;

const REL_GATE: f32 = 1e-5;

fn report<R: Runtime>(client: &ComputeClient<R>) -> bool {
    let props = &client.properties().hardware;
    eprintln!(
        "runtime={} plane_size={}..{}",
        R::name(client),
        props.plane_size_min,
        props.plane_size_max
    );
    if props.plane_size_min != 32 || props.plane_size_max != 32 {
        eprintln!("SKIP: kernel requires 32-lane planes (this backend keeps its fallback)");
        return false;
    }
    // Compiler fixture first: the f16 decode the PQ2 scale path uses, through this backend's real
    // compiler (the vendored cubecl-cpp bitcast-of-literal patch).
    let bad = hanzo_kernel::ternary::f16_decode_probe_run(client);
    println!(
        "{} f16_decode_probe: {} of {} patterns differ from `half`",
        if bad == 0 { "ok  " } else { "FAIL" },
        bad,
        hanzo_kernel::ternary::f16_probe_bits().len()
    );
    let mut worst = 0f32;
    let mut fails = 0;
    for (rows, k, nsg, nr0, rel) in pq2_0_gguf_board(client) {
        let ok = rel <= REL_GATE;
        worst = worst.max(rel);
        if !ok {
            fails += 1;
        }
        println!(
            "{} rows={rows:<6} k={k:<6} nsg={nsg} nr0={nr0} rel={rel:.3e}",
            if ok { "ok  " } else { "FAIL" }
        );
    }
    println!("worst_rel={worst:.3e} gate={REL_GATE:.0e} fails={fails}");
    fails == 0 && bad == 0
}

fn main() {
    let mut args = std::env::args().skip(1);
    let backend = match (args.next().as_deref(), args.next()) {
        (Some("--backend"), Some(b)) => b,
        _ => {
            eprintln!("usage: pq2-check --backend metal|cuda|rocm|vulkan");
            std::process::exit(2);
        }
    };
    let ok = match backend.as_str() {
        #[cfg(any(feature = "metal", feature = "vulkan"))]
        "metal" | "vulkan" => {
            use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
            report(&WgpuRuntime::client(&WgpuDevice::default()))
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use cubecl::cuda::{CudaDevice, CudaRuntime};
            report(&CudaRuntime::client(&CudaDevice::default()))
        }
        #[cfg(feature = "rocm")]
        "rocm" => {
            use hanzo_cubecl_hip::{AmdDevice, HipRuntime};
            report(&HipRuntime::client(&AmdDevice::default()))
        }
        b => {
            eprintln!("backend `{b}` not compiled in; enable its cargo feature");
            std::process::exit(2);
        }
    };
    std::process::exit(if ok { 0 } else { 1 });
}
