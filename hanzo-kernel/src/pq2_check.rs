//! pq2-check: the per-backend correctness gate for the native-layout PQ2_0 DSL matvec.
//!
//!   pq2-check --backend metal|cuda|rocm|vulkan
//!
//! Same fixtures (`PQ2_GGUF_SHAPES`, synthetic GGUF bytes, the f64 oracle that decodes exactly like
//! hanzo-ml's `BlockPQ2_0::to_float`) on every backend; each backend passes or fails on its own.
//! Exit 0 only if every case is within REL_GATE scale-relative error.

use hanzo_kernel::prelude::*;
use hanzo_kernel::ternary::{pq2_0_gguf_board, pq2_0_gguf_gemm_board};

const REL_GATE: f32 = 1e-5;
const GEMM_F32_GATE: f32 = 5e-4;
const GEMM_BF16_GATE: f32 = 8e-3;

fn report<R: Runtime>(client: &ComputeClient<R>, gemm_only: bool, with_gemm: bool) -> bool {
    let props = &client.properties().hardware;
    eprintln!(
        "runtime={} plane_size={}..{}",
        R::name(client),
        props.plane_size_min,
        props.plane_size_max
    );
    if !gemm_only && (props.plane_size_min != 32 || props.plane_size_max != 32) {
        eprintln!("SKIP: kernel requires 32-lane planes (this backend keeps its fallback)");
        return false;
    }
    let mut bad = 0usize;
    let mut fails = 0usize;
    if !gemm_only {
        // Compiler fixture first: the f16 decode the PQ2 scale path uses, through this backend's real
        // compiler (the vendored cubecl-cpp bitcast-of-literal patch).
        bad = hanzo_kernel::ternary::f16_decode_probe_run(client);
        println!(
            "{} f16_decode_probe: {} of {} patterns differ from `half`",
            if bad == 0 { "ok  " } else { "FAIL" },
            bad,
            hanzo_kernel::ternary::f16_probe_bits().len()
        );
        let mut worst = 0f32;
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
    }
    if with_gemm || gemm_only {
        let mut worst_f32 = 0f32;
        let mut worst_bf16 = 0f32;
        let mut gfails = 0usize;
        for (m, n, k, rel32, rel_b) in pq2_0_gguf_gemm_board(client) {
            let ok = rel32 <= GEMM_F32_GATE && rel_b <= GEMM_BF16_GATE;
            worst_f32 = worst_f32.max(rel32);
            worst_bf16 = worst_bf16.max(rel_b);
            if !ok {
                gfails += 1;
            }
            println!(
                "{} gemm m={m:<4} n={n:<5} k={k:<5} f32={rel32:.3e} bf16={rel_b:.3e}",
                if ok { "ok  " } else { "FAIL" }
            );
        }
        println!(
            "gemm_worst f32={worst_f32:.3e} (gate={GEMM_F32_GATE:.0e}) bf16={worst_bf16:.3e} (gate={GEMM_BF16_GATE:.0e}) fails={gfails}"
        );
        fails += gfails;
    }
    fails == 0 && bad == 0
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut backend: Option<&str> = None;
    let mut gemm_only = false;
    let mut with_gemm = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--backend" if i + 1 < args.len() => {
                backend = Some(&args[i + 1]);
                i += 2;
            }
            "--gemm" => {
                gemm_only = true;
                i += 1;
            }
            "--all" => {
                with_gemm = true;
                i += 1;
            }
            _ => {
                eprintln!("usage: pq2-check --backend metal|cuda|rocm|vulkan [--gemm|--all]");
                std::process::exit(2);
            }
        }
    }
    let Some(backend) = backend else {
        eprintln!("usage: pq2-check --backend metal|cuda|rocm|vulkan [--gemm|--all]");
        std::process::exit(2);
    };
    let ok = match backend {
        #[cfg(any(feature = "metal", feature = "vulkan"))]
        "metal" | "vulkan" => {
            use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
            report(&WgpuRuntime::client(&WgpuDevice::default()), gemm_only, with_gemm)
        }
        #[cfg(feature = "cuda")]
        "cuda" => {
            use cubecl::cuda::{CudaDevice, CudaRuntime};
            report(&CudaRuntime::client(&CudaDevice::default()), gemm_only, with_gemm)
        }
        #[cfg(feature = "rocm")]
        "rocm" => {
            use hanzo_cubecl_hip::{AmdDevice, HipRuntime};
            report(&HipRuntime::client(&AmdDevice::default()), gemm_only, with_gemm)
        }
        b => {
            eprintln!("backend `{b}` not compiled in; enable its cargo feature");
            std::process::exit(2);
        }
    };
    std::process::exit(if ok { 0 } else { 1 });
}
