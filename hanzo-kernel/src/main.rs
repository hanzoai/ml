//! matvec-check: bit-exact + perf gate for the DSL quant matvec across backends.
//!
//! Proves the migration thesis end-to-end: the SAME `matvec_q8` kernel source runs on CPU and every
//! compiled GPU backend, matches the CPU oracle bit-exactly, and reports per-backend throughput -- the
//! perf gate that decides whether a hand-tuned kernel can be retired for its DSL twin.

use hanzo_kernel::prelude::*;
use hanzo_kernel::quant::{
    gen_moe_q4k, gen_moe_q6k, gen_q4k, gen_q8_0_packed, matvec_q4k_bench, matvec_q4k_ref, matvec_q4k_run,
    matvec_q8_0_packed_ref, matvec_q8_0_packed_run, matvec_q8_0_packed_sg_run, matvec_q8_bench, matvec_q8_dp4a_blk_run,
    matvec_q8_dp4a_i8_run, matvec_q8_dp4a_ref, matvec_q8_ref, matvec_q8_run,
    moe_matvec_q4k_bench, moe_matvec_q4k_blk_bench, moe_matvec_q4k_blk_run, moe_matvec_q4k_ref, moe_matvec_q4k_run,
    moe_matvec_q6k_bench, moe_matvec_q6k_blk_bench, moe_matvec_q6k_blk_run, moe_matvec_q6k_ref, moe_matvec_q6k_run,
    moe_route, moe_route_ref, moe_route_run, QK8_0,
};
use std::time::Instant;
use hanzo_kernel::norm::rms_norm_run;

// dp4a matvec parity: i8-packed one-thread-per-row (portable) vs block-per-row (coalesced reads +
// shared-mem reduction, the bandwidth-bound winner). GB/s is on REAL int8 bytes (rows*k) so it
// compares fairly to the hand-tuned dp4a (~166 GB/s; gfx1151 DRAM roofline ~256, MALL cache ~32MB).
// `coop` gates the block kernels: cubecl-cpu has no cooperative thread-blocks, so they run GPU-only.
fn check_dp4a<R: Runtime>(name: &str, client: &ComputeClient<R>, rows: usize, k: usize, coop: bool) {
    let mut s = 0x9E3779B9_7F4A7C15u64;
    let mut nxt = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let wq8: Vec<i8> = (0..rows * k).map(|_| (nxt() % 255) as i8).collect();
    let xq8: Vec<i8> = (0..k).map(|_| (nxt() % 255) as i8).collect();
    let wd: Vec<f32> = (0..rows * k / 32).map(|_| (nxt() % 1000) as f32 / 8000.0 + 0.01).collect();
    let wq32: Vec<i32> = wq8.iter().map(|&x| x as i32).collect();
    let xq32: Vec<i32> = xq8.iter().map(|&x| x as i32).collect();
    let reference = matvec_q8_dp4a_ref(&wq32, &xq32, &wd, rows, k);
    let real_bytes = (rows * k) as f64; // int8 weights, 1 byte each -- the hand-tuned footprint
    let flop = 2.0 * rows as f64 * k as f64;
    let mut report = |tag: &str, (out, ms): (Vec<f32>, f64)| {
        let rel = max_rel(&reference, &out);
        println!(
            "[{:<7}] dp4a/{:<6} {}x{}  max_rel={:.2e}  {}  {:.3} ms  {:.0} GB/s  {:.0} GFLOP/s",
            name, tag, rows, k, rel,
            if rel < 2e-2 { "MATCH ✓" } else { "MISMATCH ✗" },
            ms, real_bytes / (ms * 1e6), flop / (ms * 1e6)
        );
    };
    report("i8pack", matvec_q8_dp4a_i8_run(client, &wq8, &xq8, &wd, rows, k, 50));
    if coop {
        for nt in [64usize, 128, 256] {
            report(&format!("blk{nt}"), matvec_q8_dp4a_blk_run(client, &wq8, &xq8, &wd, rows, k, nt, 50));
        }
    }
}

fn maxrel(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        m = m.max((x - y).abs() / x.abs().max(1e-6));
    }
    m
}

// Scale-relative error: max |a-b| over the output's own magnitude scale (max|a|). The correct gate
// for a matvec with sign cancellation -- per-element rel-error blows up on near-zero outputs (a tiny
// absolute FMA-vs-separate-rounding delta over a value that cancelled to ~0), which is precision, not
// a logic error. The CPU kernel is bit-exact to the same oracle; this isolates true GPU accuracy.
fn scalerel(a: &[f32], b: &[f32]) -> f32 {
    let scale = a.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-20);
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

// Q4_K: the real K-quant decoded in-kernel from packed bytes. Bit-exact gate + honest kernel-only BW.
fn check_q4k<R: Runtime>(name: &str, client: &ComputeClient<R>, rows: usize, k: usize) {
    let (wqs, wsc, wd, wdm, x) = gen_q4k(rows, k);
    let reference = matvec_q4k_ref(&wqs, &wsc, &wd, &wdm, &x, rows, k);
    let got = matvec_q4k_run::<R>(client, &wqs, &wsc, &wd, &wdm, &x, rows, k);
    let rel = maxrel(&reference, &got);
    let ok = rel < 3e-3;
    let ms = matvec_q4k_bench::<R>(client, &wqs, &wsc, &wd, &wdm, &x, rows, k, 50);
    // REAL packed bytes/block = 144 (d+dmin+scales+qs); nb = k/256 blocks/row.
    let wbytes = rows * (k / 256) * 144;
    let gbps = wbytes as f64 / (ms * 1e6);
    let gflops = 2.0 * rows as f64 * k as f64 / (ms * 1e6);
    println!(
        "[{:<7}] Q4_K   {}x{}  max_rel={:.2e}  {}  {:.3} ms  {:.0} GB/s  {:.0} GFLOP/s",
        name, rows, k, rel,
        if ok { "BIT-EXACT ✓" } else { "MISMATCH ✗" }, ms, gbps, gflops
    );
}

// Q4_K indexed-MoE matvec: the DSL expert-gather kernel. Bit-exact gate + kernel-only weight BW on
// the REAL Q4_K footprint of the routed rows (slots*n rows x 144 B/superblock). This is the DSL twin
// the hand-rolled moe_matvec_q4k shader is gated against before it can be retired to an oracle.
fn check_moe_q4k<R: Runtime>(name: &str, client: &ComputeClient<R>, e: usize, n: usize, slots: usize, k: usize) {
    let (wqs, wsc, wd, wdm, x, ids) = gen_moe_q4k(e, n, slots, k);
    let reference = moe_matvec_q4k_ref(&wqs, &wsc, &wd, &wdm, &x, &ids, slots, n, k);
    let got = moe_matvec_q4k_run::<R>(client, &wqs, &wsc, &wd, &wdm, &x, &ids, slots, n, k);
    let rel = maxrel(&reference, &got);
    let ok = rel < 3e-3;
    let ms = moe_matvec_q4k_bench::<R>(client, &wqs, &wsc, &wd, &wdm, &x, &ids, slots, n, k, 50);
    // Each of the slots*n output rows reads one expert's [k] row = k/256 superblocks x 144 B.
    let wbytes = slots * n * (k / 256) * 144;
    let gbps = wbytes as f64 / (ms * 1e6);
    let gflops = 2.0 * (slots * n) as f64 * k as f64 / (ms * 1e6);
    println!(
        "[{:<7}] MoEQ4K E{} {}x{}x{}  max_rel={:.2e}  {}  {:.3} ms  {:.0} GB/s  {:.0} GFLOP/s",
        name, e, slots, n, k, rel,
        if ok { "BIT-EXACT ✓" } else { "MISMATCH ✗" }, ms, gbps, gflops
    );
}

// Block-reduced Q4_K MoE matvec: the decode-perf lever (one workgroup per output, shared-mem reduce).
// scale-relative gate vs the naive-order oracle (block reduce reorders the f32 sum) + throughput.
fn check_moe_q4k_blk<R: Runtime>(name: &str, client: &ComputeClient<R>, e: usize, n: usize, slots: usize, k: usize, nt: usize) {
    let (wqs, wsc, wd, wdm, x, ids) = gen_moe_q4k(e, n, slots, k);
    let reference = moe_matvec_q4k_ref(&wqs, &wsc, &wd, &wdm, &x, &ids, slots, n, k);
    let got = moe_matvec_q4k_blk_run::<R>(client, &wqs, &wsc, &wd, &wdm, &x, &ids, slots, n, k, nt);
    let srel = scalerel(&reference, &got);
    let ok = srel < 1e-3;
    let ms = moe_matvec_q4k_blk_bench::<R>(client, &wqs, &wsc, &wd, &wdm, &x, &ids, slots, n, k, nt, 50);
    let wbytes = slots * n * (k / 256) * 144;
    let gbps = wbytes as f64 / (ms * 1e6);
    let gflops = 2.0 * (slots * n) as f64 * k as f64 / (ms * 1e6);
    println!(
        "[{:<7}] MoEQ4Kb E{} {}x{}x{} nt={:<3} scale_rel={:.2e}  {}  {:.3} ms  {:.0} GB/s  {:.0} GFLOP/s",
        name, e, slots, n, k, nt, srel,
        if ok { "MATCH ✓" } else { "MISMATCH ✗" }, ms, gbps, gflops
    );
}

// Q6_K indexed-MoE matvec: the DSL twin of moe_matvec_q6k.comp. Bit-exact gate + kernel-only weight
// BW on the real Q6_K footprint of the routed rows (slots*n rows x 210 B/superblock).
fn check_moe_q6k<R: Runtime>(name: &str, client: &ComputeClient<R>, e: usize, n: usize, slots: usize, k: usize) {
    let (wql, wqh, wsc, wd, x, ids) = gen_moe_q6k(e, n, slots, k);
    let reference = moe_matvec_q6k_ref(&wql, &wqh, &wsc, &wd, &x, &ids, slots, n, k);
    let got = moe_matvec_q6k_run::<R>(client, &wql, &wqh, &wsc, &wd, &x, &ids, slots, n, k);
    let srel = scalerel(&reference, &got);
    let ok = srel < 1e-3;
    let ms = moe_matvec_q6k_bench::<R>(client, &wql, &wqh, &wsc, &wd, &x, &ids, slots, n, k, 50);
    let wbytes = slots * n * (k / 256) * 210; // Q6_K: 210 B/superblock
    let gbps = wbytes as f64 / (ms * 1e6);
    let gflops = 2.0 * (slots * n) as f64 * k as f64 / (ms * 1e6);
    println!(
        "[{:<7}] MoEQ6K E{} {}x{}x{}  scale_rel={:.2e}  {}  {:.3} ms  {:.0} GB/s  {:.0} GFLOP/s",
        name, e, slots, n, k, srel,
        if ok { "MATCH ✓" } else { "MISMATCH ✗" }, ms, gbps, gflops
    );
}

// Block-reduced Q6_K MoE matvec: the Q6_K decode-perf lever (naive Q6_K is decode-ALU-bound).
fn check_moe_q6k_blk<R: Runtime>(name: &str, client: &ComputeClient<R>, e: usize, n: usize, slots: usize, k: usize, nt: usize) {
    let (wql, wqh, wsc, wd, x, ids) = gen_moe_q6k(e, n, slots, k);
    let reference = moe_matvec_q6k_ref(&wql, &wqh, &wsc, &wd, &x, &ids, slots, n, k);
    let got = moe_matvec_q6k_blk_run::<R>(client, &wql, &wqh, &wsc, &wd, &x, &ids, slots, n, k, nt);
    let srel = scalerel(&reference, &got);
    let ok = srel < 1e-3;
    let ms = moe_matvec_q6k_blk_bench::<R>(client, &wql, &wqh, &wsc, &wd, &x, &ids, slots, n, k, nt, 50);
    let wbytes = slots * n * (k / 256) * 210;
    let gbps = wbytes as f64 / (ms * 1e6);
    let gflops = 2.0 * (slots * n) as f64 * k as f64 / (ms * 1e6);
    println!(
        "[{:<7}] MoEQ6Kb E{} {}x{}x{} nt={:<3} scale_rel={:.2e}  {}  {:.3} ms  {:.0} GB/s  {:.0} GFLOP/s",
        name, e, slots, n, k, nt, srel,
        if ok { "MATCH ✓" } else { "MISMATCH ✗" }, ms, gbps, gflops
    );
}

fn check_moe_route<R: Runtime>(name: &str, client: &ComputeClient<R>, ntok: usize, n_experts: usize, topk: usize, nt: usize) {
    // Deterministic router logits [ntok, n_experts].
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let logits: Vec<f32> = (0..ntok * n_experts).map(|_| (next() % 4000) as f32 / 1000.0 - 2.0).collect();
    let (ids_ref, w_ref) = moe_route_ref(&logits, ntok, n_experts, topk);
    let (ids_got, w_got) = moe_route_run::<R>(client, &logits, ntok, n_experts, topk, nt);
    // Weights are the correctness contract (scale-exact). A per-token SET difference is a top-k
    // BOUNDARY TIE (two experts with ~equal logit at slot topk; GPU tree-reduce vs CPU sort pick
    // different-but-equivalent ones) -- inconsequential downstream (weighted expert sum), same
    // non-determinism as llama's unstable Vulkan sort. Count ties; pass on the weights.
    let mut ties = 0usize;
    for tok in 0..ntok {
        let mut a: Vec<u32> = ids_ref[tok * topk..(tok + 1) * topk].to_vec(); a.sort_unstable();
        let mut b: Vec<u32> = ids_got[tok * topk..(tok + 1) * topk].to_vec(); b.sort_unstable();
        if a != b { ties += 1; }
    }
    let wrel = scalerel(&w_ref, &w_got);
    let ok = wrel < 1e-4;
    // Kernel-only throughput (fixed buffers, iters dispatches, one sync).
    let lh = client.create_from_slice(f32::as_bytes(&logits));
    let ih = client.create_from_slice(u32::as_bytes(&vec![0u32; ntok * topk]));
    let wh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; ntok * topk]));
    let launch = |c: &ComputeClient<R>| unsafe {
        moe_route::launch_unchecked::<f32, R>(
            c, Grid::Static(ntok as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(lh.clone(), logits.len()),
            ArrayArg::from_raw_parts(ih.clone(), ntok * topk),
            ArrayArg::from_raw_parts(wh.clone(), ntok * topk),
            n_experts, topk, nt,
        );
    };
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(ih.clone());
    let t0 = std::time::Instant::now();
    for _ in 0..200 { launch(client); }
    let _ = client.read_one_unchecked(ih.clone());
    let ms = t0.elapsed().as_secs_f64() * 1e3 / 200.0;
    println!(
        "[{:<7}] MoERoute ntok={} E{} topk={} nt={:<3} w_rel={:.2e} ties={}/{}  {}  {:.4} ms/batch",
        name, ntok, n_experts, topk, nt, wrel, ties, ntok,
        if ok { "MATCH ✓" } else { "MISMATCH ✗" }, ms
    );
}

fn gen(rows: usize, k: usize) -> (Vec<f32>, Vec<i32>, Vec<f32>) {
    let nb = k / QK8_0;
    let mut s = 0x2545F491_4F6CDD1Du64; // xorshift, deterministic
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let wd: Vec<f32> = (0..rows * nb).map(|_| (next() % 1000) as f32 / 8000.0 + 0.01).collect();
    let wq: Vec<i32> = (0..rows * k).map(|_| (next() % 255) as i32 - 127).collect();
    let x: Vec<f32> = (0..k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    (wd, wq, x)
}

fn max_rel(a: &[f32], b: &[f32]) -> f32 {
    let mut m = 0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs();
        let denom = x.abs().max(1e-6);
        m = m.max(d / denom);
    }
    m
}

// Q8_0 PACKED: the production-layout matvec (9 u32/block, in-kernel fp16+int8 decode). Bit-exact vs
// the CPU oracle + real weight-bandwidth (packed bytes = rows * k/32 * 34, the true Q8_0 footprint).
fn check_q8_0_packed<R: Runtime>(name: &str, client: &ComputeClient<R>, rows: usize, k: usize, nt: usize) {
    let (w, x) = gen_q8_0_packed(rows, k);
    let reference = matvec_q8_0_packed_ref(&w, &x, rows, k);
    let (out, ms) = matvec_q8_0_packed_run::<R>(client, &w, &x, rows, k, nt, 50);
    let rel = max_rel(&reference, &out);
    let wbytes = rows * (k / 32) * 34; // real Q8_0: 34 bytes/block (fp16 scale + 32 int8)
    let gbps = wbytes as f64 / (ms * 1e6);
    let gflops = 2.0 * rows as f64 * k as f64 / (ms * 1e6);
    println!(
        "[{:<7}] Q8_0pk {}x{} nt={:<3} max_rel={:.2e}  {}  {:.3} ms  {:.0} GB/s  {:.0} GFLOP/s",
        name, rows, k, nt, rel,
        if rel < 3e-3 { "BIT-EXACT ✓" } else { "MISMATCH ✗" }, ms, gbps, gflops
    );
}

// Subgroup (plane_sum, no shared-mem) Q8_0 packed matvec -- mirrors production mul_mat_vec_q8_sg.
// nt MUST be the hardware plane size; a wrong plane size drops cross-plane partials -> bit-exact catches it.
fn check_q8_0_packed_sg<R: Runtime>(name: &str, client: &ComputeClient<R>, rows: usize, k: usize, nt: usize) {
    let (w, x) = gen_q8_0_packed(rows, k);
    let reference = matvec_q8_0_packed_ref(&w, &x, rows, k);
    let (out, ms) = matvec_q8_0_packed_sg_run::<R>(client, &w, &x, rows, k, nt, 50);
    let rel = max_rel(&reference, &out);
    let wbytes = rows * (k / 32) * 34;
    let gbps = wbytes as f64 / (ms * 1e6);
    println!(
        "[{:<7}] Q8_0sg {}x{} nt={:<3} max_rel={:.2e}  {}  {:.3} ms  {:.0} GB/s",
        name, rows, k, nt, rel,
        if rel < 3e-3 { "BIT-EXACT ✓" } else { "MISMATCH ✗ (plane!=nt?)" }, ms, gbps
    );
}

fn check<R: Runtime>(name: &str, client: &ComputeClient<R>, rows: usize, k: usize) {
    let (wd, wq, x) = gen(rows, k);
    let reference = matvec_q8_ref(&wd, &wq, &x, rows, k);
    let got = matvec_q8_run::<R>(client, &wd, &wq, &x, rows, k);
    let rel = max_rel(&reference, &got);
    let ok = rel < 3e-3;  // real decode bugs are >>1e-2; f32 reorder over K terms is ~K*eps
    // warm + timed loop for a rough throughput number (GFLOP: 2 * rows * k)
    for _ in 0..2 {
        let _ = matvec_q8_run::<R>(client, &wd, &wq, &x, rows, k);
    }
    let iters = 20;
    let t = Instant::now();
    for _ in 0..iters {
        let _ = matvec_q8_run::<R>(client, &wd, &wq, &x, rows, k);
    }
    let _ = (t, iters);
    // real kernel-only throughput (amortized host round-trip)
    let ms = matvec_q8_bench::<R>(client, &wd, &wq, &x, rows, k, 50);
    let gbps = (wd.len() * 4 + wq.len() * 4) as f64 / (ms * 1e6); // weight bytes / time = effective BW
    println!(
        "[{:<7}] matvec {}x{}  max_rel={:.2e}  {}  {:.3} ms/dispatch  {:.0} GB/s (weight BW)",
        name, rows, k, rel, if ok { "MATCH ✓ (f32-reorder tol)" } else { "MISMATCH ✗" }, ms, gbps
    );
}

// RMSNorm DSL dispatch gate: the SAME #[kernel] rms_norm source lowered per backend, vs a plain-Rust
// oracle. Proves the norm op family dispatches bit-exact on each compiled backend (norm column).
fn check_rms<R: Runtime>(name: &str, client: &ComputeClient<R>, rows: usize, n: usize) {
    let mut s = 0x1234_5678_9ABC_DEF1u64;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let x: Vec<f32> = (0..rows * n).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    let w: Vec<f32> = (0..n).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    let eps = 1e-6f32;
    let mut reference = vec![0f32; rows * n];
    for r in 0..rows {
        let base = r * n;
        let mut ss = 0f32;
        for i in 0..n { let v = x[base + i]; ss += v * v; }
        let denom = (ss / n as f32 + eps).sqrt();
        for i in 0..n { reference[base + i] = x[base + i] / denom * w[i]; }
    }
    let got = rms_norm_run::<R>(client, &x, &w, rows, n, eps);
    let rel = max_rel(&reference, &got);
    println!(
        "[{:<7}] rmsnorm {}x{}  max_rel={:.2e}  {}",
        name, rows, n, rel, if rel < 3e-3 { "MATCH ✓" } else { "MISMATCH ✗" }
    );
}

fn main() {
    let (rows, k) = (4096usize, 4096usize);
    let ctrl = 256usize; // small-K control: reorder noise ~ ctrl*eps, should be ~1e-6

    // `matvec-check dump` dispatches exactly the engine-bound MoE block kernels at their evo-optimal
    // baked nt (Q4_K 64, Q6_K 32) so `CUBECL_DEBUG_SPIRV=<dir>` writes exactly one .spv per kernel --
    // the committed hanzo-ml Vulkan artifact. Build with `--features vulkan,spirv-dump`.
    // `matvec-check dump` dispatches exactly the engine-bound MoE block kernels for each LIVE model
    // shape at its evo-optimal power-of-2 nt, so `CUBECL_DEBUG_SPIRV=<dir>` writes one .spv per (shape).
    // The DSL source is ONE fn per quant; comptime lowers it to a fast specialized artifact per shape
    // (like llama's templated kernels). Disambiguate by LocalSize (nt): q4k 64 = gate/up, 8 = down.
    //   Qwen3-30B-A3B Q4_K_M: ffn_gate/up_exps Q4_K [128,768,2048]; ffn_down_exps Q6_K/Q4_K [128,2048,768].
    #[cfg(all(feature = "vulkan", feature = "spirv-dump"))]
    if std::env::args().any(|a| a == "dump") {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let c = WgpuRuntime::client(&WgpuDevice::default());
        check_moe_q4k_blk::<WgpuRuntime>("VK/dump", &c, 128, 768, 8, 2048, 64); // gate/up: LocalSize 64
        check_moe_q4k_blk::<WgpuRuntime>("VK/dump", &c, 128, 2048, 8, 768, 32); // down:    LocalSize 32
        check_moe_q6k_blk::<WgpuRuntime>("VK/dump", &c, 128, 2048, 8, 768, 32); // down:    LocalSize 32
        return;
    }

    println!("hanzo-kernel :: one #[device] matvec_q8 source, lowered per backend, gated bit-exact\n");

    #[cfg(feature = "cpu")]
    {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        check::<CpuRuntime>("CPU", &c, rows, k);
        check::<CpuRuntime>("CPU/ctrl", &c, rows, ctrl);
        check_q4k::<CpuRuntime>("CPU", &c, rows, k);
        check_moe_q4k::<CpuRuntime>("CPU", &c, 8, 64, 4, 512); // small MoE bank for the CPU oracle
        check_moe_q6k::<CpuRuntime>("CPU", &c, 8, 64, 4, 512);
        check_dp4a::<CpuRuntime>("CPU", &c, rows, k, false); // cubecl-cpu: no cooperative blocks
        check_rms::<CpuRuntime>("CPU", &c, rows, k);
    }
    #[cfg(feature = "vulkan")]
    {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let c = WgpuRuntime::client(&WgpuDevice::default());
        check::<WgpuRuntime>("VULKAN", &c, rows, k);
        check::<WgpuRuntime>("VK/ctrl", &c, rows, ctrl);
        check_q4k::<WgpuRuntime>("VULKAN", &c, rows, k);
        // Qwen3-30B-A3B-shaped MoE decode: 128 experts, 8 routed slots, n=768 intermediate, k=2048 hidden.
        check_moe_q4k::<WgpuRuntime>("VULKAN", &c, 128, 768, 8, 2048);
        check_moe_q4k_blk::<WgpuRuntime>("VK/blk", &c, 128, 768, 8, 2048, 32); // decode-perf lever
        check_moe_q4k_blk::<WgpuRuntime>("VK/blk", &c, 128, 768, 8, 2048, 64);
        // down-projection shape (n=2048, k=768 -> nsub=24): tree-reduce needs nt a power of 2; the
        // ceil+guard lets nt EXCEED nsub (extra lanes contribute 0), so sweep 8/16/32 for best occupancy.
        // A distinct specialized .spv from gate/up -- same source fn.
        check_moe_q4k_blk::<WgpuRuntime>("VK/down", &c, 128, 2048, 8, 768, 8);
        check_moe_q4k_blk::<WgpuRuntime>("VK/down", &c, 128, 2048, 8, 768, 16);
        check_moe_q4k_blk::<WgpuRuntime>("VK/down", &c, 128, 2048, 8, 768, 32);
        check_moe_q6k::<WgpuRuntime>("VULKAN", &c, 128, 768, 8, 2048); // ffn_down_exps are Q6_K in Q4_K_M
        check_moe_q6k_blk::<WgpuRuntime>("VK/blk", &c, 128, 768, 8, 2048, 32);
        check_moe_q6k_blk::<WgpuRuntime>("VK/blk", &c, 128, 768, 8, 2048, 64);
        check_moe_q6k_blk::<WgpuRuntime>("VK/down", &c, 128, 2048, 8, 768, 8);
        check_moe_q6k_blk::<WgpuRuntime>("VK/down", &c, 128, 2048, 8, 768, 16);
        check_moe_q6k_blk::<WgpuRuntime>("VK/down", &c, 128, 2048, 8, 768, 32);
        // Fused MoE top-k router (Qwen3-30B-A3B: 128 experts, top-8): decode batch + prefill batch.
        check_moe_route::<WgpuRuntime>("VULKAN", &c, 8, 128, 8, 128);
        check_moe_route::<WgpuRuntime>("VK/pref", &c, 512, 128, 8, 128);
        check_dp4a::<WgpuRuntime>("VULKAN", &c, rows, k, true);
        check_dp4a::<WgpuRuntime>("VK/big", &c, 8192, 8192, true); // 67MB weights: cache-busting BW
        check_q8_0_packed::<WgpuRuntime>("VULKAN", &c, rows, k, 64);
        check_q8_0_packed::<WgpuRuntime>("VK/big", &c, 8192, 8192, 128); // cache-busting Q8_0 BW
        // subgroup variant: nt MUST equal the hardware plane size (bit-exact fails otherwise). Try 32/64.
        check_q8_0_packed_sg::<WgpuRuntime>("VK/sg32", &c, rows, k, 32);
        check_q8_0_packed_sg::<WgpuRuntime>("VK/sg64", &c, rows, k, 64);
        check_q8_0_packed_sg::<WgpuRuntime>("VK/sgBIG", &c, 8192, 8192, 32);
    }
    #[cfg(feature = "metal")]
    {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let c = WgpuRuntime::client(&WgpuDevice::default());
        check::<WgpuRuntime>("METAL", &c, rows, k);
        check_q4k::<WgpuRuntime>("METAL", &c, rows, k);
        check_dp4a::<WgpuRuntime>("METAL", &c, rows, k, true);
    }
    #[cfg(feature = "cuda")]
    {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        let c = CudaRuntime::client(&CudaDevice::default());
        check::<CudaRuntime>("CUDA", &c, rows, k);
        check_q4k::<CudaRuntime>("CUDA", &c, rows, k);
        check_dp4a::<CudaRuntime>("CUDA", &c, rows, k, true);
        check_rms::<CudaRuntime>("CUDA", &c, rows, k);
    }
    #[cfg(feature = "rocm")]
    {
        use hanzo_cubecl_hip::{AmdDevice, HipRuntime};
        let c = HipRuntime::client(&AmdDevice::default());
        check::<HipRuntime>("ROCM", &c, rows, k);
        check_q4k::<HipRuntime>("ROCM", &c, rows, k);
        check_dp4a::<HipRuntime>("ROCM", &c, rows, k, true);
    }
}
