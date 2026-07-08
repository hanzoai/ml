//! mmq-check: does the DSL emit a working int8 tensor-core MMQ GEMM on CUDA?
//!
//! Staged, cheapest-kill-first (run with --features cuda on the GPU box):
//!   1a  high-level WMMA hello  (i8 16x16x16 -> i32)        exact-int gate
//!   1b  manual MMA hello       (i8 m16n8k32 -> i32)        exact-int gate
//!   2   Q8_0 x q8_1 MMQ tile   (one/few tiles, K-loop)     bit-close vs the quantized CPU oracle
//!   3   full prefill GEMM      (M=512 N=4096 K=4096)       verify + kernel-only GFLOP/s & GB/s
//!
//! Each stage is isolated with catch_unwind so a JIT failure in one path still lets the rest report
//! (a decisive negative -- "the DSL can't express X on sm_121" -- must survive to the log).

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

fn main() {
    use cubecl::cuda::{CudaDevice, CudaRuntime};
    let client = CudaRuntime::client(&CudaDevice::default());
    let plane = client.properties().hardware.plane_size_max;
    println!("== hanzo-kernel MMQ probe (CUDA, plane={plane}) ==\n");

    // ---- Stage 1a: high-level WMMA hello ----
    let r = catch_unwind(AssertUnwindSafe(|| {
        let a = rand_i8(256, 0x1234);
        let b = rand_i8(256, 0x9E37);
        let got = wmma_hello_i8_run::<CudaRuntime>(&client, &a, &b);
        let want = hello_i8_ref(&a, &b);
        let bad = got.iter().zip(&want).filter(|(x, y)| x != y).count();
        println!(
            "[1a] WMMA hello   i8 16x16x16->i32   {} ({} / 256 mismatched)",
            if bad == 0 { "EXACT ✓" } else { "MISMATCH ✗" }, bad
        );
        bad == 0
    }));
    if r.is_err() {
        println!("[1a] WMMA hello   i8 16x16x16->i32   PANIC/JIT-FAIL ✗ (see stderr above)");
    }

    // ---- Stage 1b: manual MMA hello ----
    let r = catch_unwind(AssertUnwindSafe(|| {
        let a = rand_i8(16 * 32, 0xABCD);
        let b = rand_i8(32 * 8, 0x5555);
        let got = mma_hello_i8_run::<CudaRuntime>(&client, &a, &b, plane);
        let want = mma_hello_ref(&a, &b);
        let bad = got.iter().zip(&want).filter(|(x, y)| x != y).count();
        println!(
            "[1b] manual MMA   i8 m16n8k32->i32   {} ({} / 128 mismatched)",
            if bad == 0 { "EXACT ✓" } else { "MISMATCH ✗" }, bad
        );
        bad == 0
    }));
    if r.is_err() {
        println!("[1b] manual MMA   i8 m16n8k32->i32   PANIC/JIT-FAIL ✗ (see stderr above)");
    }

    // ---- Stage 2: MMQ tile(s), bit-close vs the quantized oracle ----
    for (m, n, k) in [(16usize, 16usize, 64usize), (32, 64, 256), (64, 128, 512)] {
        let r = catch_unwind(AssertUnwindSafe(|| {
            let (xq, xs, wq, wd) = gen_mmq(m, n, k);
            let (got, _) = mmq_q8_wmma_run::<CudaRuntime>(&client, &xq, &xs, &wq, &wd, m, n, k, 1);
            let want = mmq_q8_ref(&xq, &xs, &wq, &wd, m, n, k);
            let rel = max_rel(&want, &got);
            println!(
                "[2 ] MMQ {}x{}x{:<4}  max_rel={:.2e}  {}",
                m, n, k, rel,
                if rel < 3e-3 { "BIT-CLOSE ✓" } else { "MISMATCH ✗" }
            );
        }));
        if r.is_err() {
            println!("[2 ] MMQ {m}x{n}x{k}  PANIC/JIT-FAIL ✗");
        }
    }

    // ---- Stage 3: full prefill GEMM, verify + kernel-only bench ----
    let r = catch_unwind(AssertUnwindSafe(|| {
        let (m, n, k) = (512usize, 4096usize, 4096usize);
        let (xq, xs, wq, wd) = gen_mmq(m, n, k);
        let (got, ms) = mmq_q8_wmma_run::<CudaRuntime>(&client, &xq, &xs, &wq, &wd, m, n, k, 50);
        // verify on a stripe (full oracle is O(M*N*K) on host -- check first 8 rows exactly).
        let mrows = 8usize;
        let want = mmq_q8_ref(&xq, &xs, &wq, &wd, mrows, n, k);
        let rel = max_rel(&want, &got[..mrows * n]);
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        let wbytes = (n * k) as f64;          // int8 weight stream (the MMQ footprint)
        let abytes = (m * k) as f64;          // int8 activation
        println!(
            "\n[3 ] GEMM {m}x{n}x{k}  max_rel(8 rows)={:.2e} {}\n     {:.3} ms/dispatch   {:.0} GFLOP/s   {:.0} GB/s (W)   {:.0} GB/s (W+X)",
            rel, if rel < 3e-3 { "✓" } else { "✗" },
            ms, flop / (ms * 1e6), wbytes / (ms * 1e6), (wbytes + abytes) / (ms * 1e6),
        );
    }));
    if r.is_err() {
        println!("[3 ] GEMM 512x4096x4096  PANIC/JIT-FAIL ✗");
    }
}
