//! mmq-check: does the DSL emit a working int8 tensor-core MMQ GEMM on THIS backend?
//!
//! Staged, cheapest-kill-first (`--features cuda` or `--features vulkan` on the GPU box):
//!   1a  high-level WMMA hello  (i8 16x16x16 -> i32)        exact-int gate
//!   1b  manual MMA hello       (i8 m16n8k32 -> i32)        exact-int gate, CUDA/ROCm only
//!   2   Q8_0 x q8_1 MMQ tile   (one/few tiles, K-loop)     scale-relative vs the quantized oracle
//!   3   full prefill GEMM      (M=512 N=4096 K=4096)       verify + kernel-only GFLOP/s & GB/s
//!
//! Stage 1b is skipped off CUDA/ROCm by construction, not by failure: SPIR-V has no per-lane fragment
//! layout to query, so the manual-MMA family cannot lower there (see `hanzo_kernel::mmq::mma_hello_i8`).
//!
//! Each stage is isolated with catch_unwind so a JIT failure in one path still lets the rest report
//! (a decisive negative -- "the DSL can't express X here" -- must survive to the log).

use hanzo_kernel::mmq::*;
use hanzo_kernel::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        m = m.max((x - y).abs() / x.abs().max(1e-6));
    }
    m
}

/// Robust error vs the quantized oracle: max abs error, and rel to the tile's max |ref| (not to a
/// per-element denom, which explodes on near-zero cancellations of signed int8 sums). Returns
/// (max_abs, rel_to_max). A real decode/accumulation bug moves BOTH; a near-zero cancellation moves
/// only per-element max_rel.
fn err_robust(a: &[f32], b: &[f32]) -> (f32, f32) {
    let mut maxabs = 0f32;
    let mut refmax = 1e-9f32;
    for (x, y) in a.iter().zip(b.iter()) {
        maxabs = maxabs.max((x - y).abs());
        refmax = refmax.max(x.abs());
    }
    (maxabs, maxabs / refmax)
}

fn rand_i8(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s % 255) as i64 - 127) as i8
        })
        .collect()
}

/// The whole probe, over any runtime -- ONE staged gate, every backend. `name` labels the log.
fn probe<R: Runtime>(name: &str, client: &ComputeClient<R>) {
    let client = &client.clone();
    let plane = client.properties().hardware.plane_size_max;
    let target = Target::of(client);
    println!("== hanzo-kernel MMQ probe ({name}, plane={plane}, target={target:?}) ==\n");

    // ---- Stage 1a: high-level WMMA hello ----
    let r = catch_unwind(AssertUnwindSafe(|| {
        let a = rand_i8(256, 0x1234);
        let b = rand_i8(256, 0x9E37);
        let got = wmma_hello_i8_run::<R>(client, &a, &b);
        let want = hello_i8_ref(&a, &b);
        let bad = got.iter().zip(&want).filter(|(x, y)| x != y).count();
        println!(
            "[1a] WMMA hello   i8 16x16x16->i32   {} ({} / 256 mismatched)",
            if bad == 0 {
                "EXACT ✓"
            } else {
                "MISMATCH ✗"
            },
            bad
        );
        bad == 0
    }));
    if r.is_err() {
        println!("[1a] WMMA hello   i8 16x16x16->i32   PANIC/JIT-FAIL ✗ (see stderr above)");
    }

    // ---- Stage 1b: manual MMA hello (CUDA/ROCm only -- SPIR-V has no queryable fragment layout) ----
    if matches!(target, Target::Cuda | Target::Rocm) {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let a = rand_i8(16 * 32, 0xABCD);
            let b = rand_i8(32 * 8, 0x5555);
            let got = mma_hello_i8_run::<R>(client, &a, &b, plane);
            let want = mma_hello_ref(&a, &b);
            let bad = got.iter().zip(&want).filter(|(x, y)| x != y).count();
            println!(
                "[1b] manual MMA   i8 m16n8k32->i32   {} ({} / 128 mismatched)",
                if bad == 0 {
                    "EXACT ✓"
                } else {
                    "MISMATCH ✗"
                },
                bad
            );
            bad == 0
        }));
        if r.is_err() {
            println!("[1b] manual MMA   i8 m16n8k32->i32   PANIC/JIT-FAIL ✗ (see stderr above)");
        }
    } else {
        println!("[1b] manual MMA   i8 m16n8k32->i32   N/A (no fragment layout on {target:?})");
    }

    // ---- Stage 2: MMQ tile(s), bit-close vs the quantized oracle ----
    for (m, n, k) in [(16usize, 16usize, 64usize), (32, 64, 256), (64, 128, 512)] {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let (xq, xs, wq, wd) = gen_mmq(m, n, k);
            let (got, _) = mmq_q8_wmma_run::<R>(client, &xq, &xs, &wq, &wd, m, n, k, 1);
            let want = mmq_q8_ref(&xq, &xs, &wq, &wd, m, n, k);
            let rel = max_rel(&want, &got);
            let (maxabs, relmax) = err_robust(&want, &got);
            println!(
                "[2 ] MMQ {}x{}x{:<4}  per-elt max_rel={:.2e}  max_abs={:.2e}  rel_to_max={:.2e}  {}",
                m, n, k, rel, maxabs, relmax,
                if relmax < 1e-3 { "BIT-CLOSE ✓" } else { "MISMATCH ✗" }
            );
        }));
        if r.is_err() {
            println!("[2 ] MMQ {m}x{n}x{k}  PANIC/JIT-FAIL ✗");
        }
    }

    // ---- Stage 3: full prefill GEMM, verify + kernel-only bench (naive 1-warp vs tiled 8-warp) ----
    let r = catch_unwind(AssertUnwindSafe(|| {
        let (m, n, k) = (512usize, 4096usize, 4096usize);
        let (xq, xs, wq, wd) = gen_mmq(m, n, k);
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let wbytes = (n * k) as f64; // int8 weight stream (the MMQ footprint)
        let mrows = 8usize;
        let want = mmq_q8_ref(&xq, &xs, &wq, &wd, mrows, n, k); // exact 8-row stripe oracle

        let (g0, ms0) = mmq_q8_wmma_run::<R>(client, &xq, &xs, &wq, &wd, m, n, k, 50);
        let (a0, r0) = err_robust(&want, &g0[..mrows * n]);
        println!(
            "\n[3 ] GEMM {m}x{n}x{k}  (naive 1-warp/16x16 tile)   verify rel_to_max={:.2e} max_abs={:.2e} {}",
            r0, a0, if r0 < 3e-3 { "✓" } else { "✗" }
        );
        println!(
            "     {:.3} ms/dispatch   {:.0} GFLOP/s   {:.0} GB/s (W-stream)",
            ms0,
            flop / (ms0 * 1e6),
            wbytes / (ms0 * 1e6)
        );

        let (g1, ms1) = mmq_q8_wmma_blk_run::<R>(client, &xq, &xs, &wq, &wd, m, n, k, 50);
        let (a1, r1) = err_robust(&want, &g1[..mrows * n]);
        println!(
            "[3 ] GEMM {m}x{n}x{k}  (tiled 8-warp/32x64 tile)   verify rel_to_max={:.2e} max_abs={:.2e} {}",
            r1, a1, if r1 < 3e-3 { "✓" } else { "✗" }
        );
        println!(
            "     {:.3} ms/dispatch   {:.0} GFLOP/s   {:.0} GB/s (W-stream)   [{:.2}x the naive kernel]",
            ms1, flop / (ms1 * 1e6), wbytes / (ms1 * 1e6), ms0 / ms1
        );
    }));
    if r.is_err() {
        println!("[3 ] GEMM 512x4096x4096  PANIC/JIT-FAIL ✗");
    }
}

fn main() {
    #[cfg(feature = "cuda")]
    {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        probe::<CudaRuntime>("CUDA", &CudaRuntime::client(&CudaDevice::default()));
    }
    #[cfg(feature = "vulkan")]
    {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        probe::<WgpuRuntime>("VULKAN", &WgpuRuntime::client(&WgpuDevice::default()));
    }
}
