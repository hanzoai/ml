//! flash-metal-check: GPU-validate the portable DSL flash-attention on Apple Metal.
//!
//! Runs the REAL kernel on dbc's Metal GPU, FULL grid (`cube_base = 0`, the production launch), across
//! the exact shape space the CPU oracle (`flash_matches_materialized_ref_on_cpu`) proved for the scalar
//! arm, gating each against the SAME materialized two-pass reference (`sdpa_ref`).
//!
//! `argv[1]` selects the island arm via the explicit `target` tag passed to `flash_attn_launch` (the
//! island match is on the comptime tag, independent of the runtime -- so we can force EITHER arm onto
//! the Metal GPU):
//!   scalar  (default)  -> Target::Cpu tag  -> the portable scalar-MAC arm, lowered to MSL, on Metal.
//!                         Expected bit-exact (~1e-6): proves the algorithm + runtime + full-grid launch.
//!   cmma               -> Target::Metal tag -> the f16 simdgroup_matrix arm. This is the arm under test.
//!
//! Each launch is isolated with catch_unwind. NOTE: a cubecl JIT codegen panic fires on cubecl's own
//! worker thread, NOT the closure's thread, so catch_unwind here does NOT trap it -- the launch returns
//! the zero-initialized output buffer, which shows as rel_ref ~= 1.0. Read stderr for the codegen panic.

use half::f16;
use hanzo_kernel::attn::sdpa_ref;
use hanzo_kernel::flash::{flash_attn, flash_attn_cubes, flash_attn_launch, BC, BR};
use hanzo_kernel::island::Target;
use hanzo_kernel::prelude::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

/// Warm GPU-device time for ONE shape's flash dispatch, measured through cubecl's `client.profile`
/// (`TimingMethod::Device` = hardware timestamps) on the SAME wgpu<msl> path the kernel lowers to.
/// Buffers are created ONCE and reused; only the dispatch loop is inside the timed region -- no host
/// alloc/readback pollutes it. `target` selects the island arm (Metal = 8x8 cmma, Cpu = scalar), so
/// cmma-vs-scalar is a same-kernel same-path A/B with no cross-API confound. Returns mean ns/dispatch.
#[allow(clippy::too_many_arguments)]
fn bench_shape<R: Runtime>(
    mc: &ComputeClient<R>,
    target: Target,
    nh: usize,
    nkv: usize,
    sq: usize,
    sk: usize,
    d: usize,
    causal: bool,
    plane: usize,
) -> (f64, String) {
    let q = rnd(nh * sq * d, 0x1234_5678);
    let k = rnd(nkv * sk * d, 0x9ABC_DEF0);
    let v = rnd(nkv * sk * d, 0x0FED_CBA9);
    let cubes = flash_attn_cubes(1, nh, sq) as u32;
    let scale = 1.0f32 / (d as f32).sqrt();
    let meta = [
        sq as u32, sk as u32, nh as u32, nkv as u32, causal as u32,
        (nkv * sk * d) as u32, (sk * d) as u32, d as u32, 0u32,
    ];
    let qh = mc.create_from_slice(f32::as_bytes(&q));
    let kh = mc.create_from_slice(f32::as_bytes(&k));
    let vh = mc.create_from_slice(f32::as_bytes(&v));
    let sh = mc.create_from_slice(f32::as_bytes(&[scale]));
    let mh = mc.create_from_slice(u32::as_bytes(&meta));
    let oh = mc.create_from_slice(f32::as_bytes(&vec![0.0f32; nh * sq * d]));
    let ql = q.len();
    let kl = k.len();
    let vl = v.len();
    let ol = nh * sq * d;
    let one = |mc: &ComputeClient<R>| unsafe {
        flash_attn::launch_unchecked::<f32, R>(
            mc,
            Grid::Static(cubes, 1, 1),
            Block::new_1d(plane as u32),
            ArrayArg::from_raw_parts(qh.clone(), ql),
            ArrayArg::from_raw_parts(kh.clone(), kl),
            ArrayArg::from_raw_parts(vh.clone(), vl),
            ArrayArg::from_raw_parts(oh.clone(), ol),
            ArrayArg::from_raw_parts(sh.clone(), 1),
            ArrayArg::from_raw_parts(mh.clone(), 9),
            d, BR, BC, plane, target,
        );
    };
    for _ in 0..20 {
        one(mc);
    }
    let _ = mc.read_one_unchecked(oh.clone()); // barrier: pipeline compiled + caches warm
    let iters = 100u32;
    let (_, prof) = mc.profile(|| for _ in 0..iters { one(mc); }, "flash").unwrap();
    let method = format!("{}", prof.timing_method());
    let ticks = cubecl::future::block_on(prof.resolve());
    (ticks.duration().as_nanos() as f64 / iters as f64, method)
}

/// Minimal 8x8x8 f16 cmma probe: C[8,8] = A[8,8] @ Bᵀ, f16 in -> f32 accumulate, ONE simdgroup.
/// B is stored row-major but loaded ColMajor (stride 8) -> gives Bᵀ, EXACTLY the flash Q@Kᵀ mechanic
/// (the ColMajor-K fragment-layout path the author flagged). This is the ONLY fragment size cubecl
/// 0.10's Metal dialect emits (`simdgroup_{ty}8x8`), so it probes whether a Metal-native 8x8 flash arm
/// is a viable fix for the 16x16x16 wall -- and whether the transpose load is numerically correct.
#[kernel(targets(metal), unchecked)]
pub fn cmma8<F: Float>(a: &Array<F>, b: &Array<F>, out: &mut Array<F>) {
    let lane = UNIT_POS as usize;
    let mut ash = SharedMemory::<f16>::new(64usize);
    let mut bsh = SharedMemory::<f16>::new(64usize);
    for e in 0..2usize {
        let idx = lane * 2usize + e;
        ash[idx] = f16::cast_from(a[idx]);
        bsh[idx] = f16::cast_from(b[idx]);
    }
    sync_cube();
    let cacc = cmma::Matrix::<F>::from_value(
        cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, F::new(0.0),
    );
    let am = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor, &ash.to_slice().slice(0usize, 64usize), 8u32,
    );
    let bm = cmma::Matrix::<f16>::from_slice(
        cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::ColMajor, &bsh.to_slice().slice(0usize, 64usize), 8u32,
    );
    cmma::execute::<f16, f16, F, F>(&am, &bm, &cacc, &cacc);
    cmma::store(&mut out.to_slice_mut(), &cacc, 8u32, cmma::MatrixLayout::RowMajor);
}

/// Run the 8x8 cmma probe on Metal and gate C = A @ Bᵀ vs a scalar f32 oracle (scale-relative, f16).
fn probe8<R: Runtime>(mc: &ComputeClient<R>) {
    let a = rnd(64, 0xA11);
    let b = rnd(64, 0xB22);
    let mut want = vec![0.0f32; 64];
    for i in 0..8 {
        for jj in 0..8 {
            let mut acc = 0.0f32;
            for kk in 0..8 {
                acc += a[i * 8 + kk] * b[jj * 8 + kk]; // Bᵀ: row jj of B is column jj after transpose
            }
            want[i * 8 + jj] = acc;
        }
    }
    let ah = mc.create_from_slice(f32::as_bytes(&a));
    let bh = mc.create_from_slice(f32::as_bytes(&b));
    let oh = mc.create_from_slice(f32::as_bytes(&vec![0.0f32; 64]));
    let got = catch_unwind(AssertUnwindSafe(|| {
        unsafe {
            cmma8::launch_unchecked::<f32, R>(
                mc,
                Grid::Static(1, 1, 1),
                Block::new_1d(32),
                ArrayArg::from_raw_parts(ah.clone(), 64),
                ArrayArg::from_raw_parts(bh.clone(), 64),
                ArrayArg::from_raw_parts(oh.clone(), 64),
            );
        }
        f32::from_bytes(&mc.read_one_unchecked(oh.clone())).to_vec()
    }));
    match got {
        Ok(g) => {
            let refmax = want.iter().fold(0.0f32, |a, x| a.max(x.abs())).max(1e-6);
            let maxd = g.iter().zip(&want).fold(0.0f32, |a, (x, y)| a.max((x - y).abs()));
            eprintln!("[cmma8 probe] C=A@Bᵀ (B loaded ColMajor) 8x8x8 f16 -> f32   scale_rel={:.2e}", maxd / refmax);
            eprintln!("[cmma8 probe] got[0..4]  = {:?}", &g[0..4]);
            eprintln!("[cmma8 probe] want[0..4] = {:?}", &want[0..4]);
        }
        Err(_) => eprintln!("[cmma8 probe] LOWERING-FAIL even at 8x8x8 (see stderr panic above)"),
    }
}

/// Same xorshift PRNG and seeds as the CPU oracle test: input data byte-identical to the proven case.
fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        })
        .collect()
}

/// Scale-relative error `max|Δ| / max|ref|` -- correct gate for online softmax vs materialized ref.
fn scale_rel(got: &[f32], want: &[f32]) -> f32 {
    let refmax = want.iter().fold(0.0f32, |a, x| a.max(x.abs())).max(1e-6);
    let maxd = got.iter().zip(want).fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
    maxd / refmax
}

struct Row {
    tag: String,
    shape: String,
    rel_ref: Result<f32, String>,
}

#[allow(clippy::too_many_arguments)]
fn gate<R: Runtime>(
    mc: &ComputeClient<R>,
    target: Target,
    nh: usize,
    nkv: usize,
    sq: usize,
    sk: usize,
    d: usize,
    causal: bool,
    plane: usize,
    tag: &str,
) -> Row {
    let q = rnd(nh * sq * d, 0x1234_5678);
    let k = rnd(nkv * sk * d, 0x9ABC_DEF0);
    let v = rnd(nkv * sk * d, 0x0FED_CBA9);
    let want = sdpa_ref(&q, &k, &v, nh, nkv, sq, sk, d, causal);
    let cubes = flash_attn_cubes(1, nh, sq);

    // FULL grid on the Metal GPU, cube_base = 0, all cubes in one dispatch (the production launch).
    let got = catch_unwind(AssertUnwindSafe(|| {
        flash_attn_launch::<R>(mc, &q, &k, &v, 1, nh, nkv, sq, sk, sk, d, causal, plane, 0, cubes, target)
    }));

    let rel_ref = match got {
        Ok(g) => Ok(scale_rel(&g, &want)),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "<non-string panic>".into());
            Err(msg)
        }
    };
    Row {
        tag: tag.to_string(),
        shape: format!("nh{nh}/nkv{nkv} sq{sq} sk{sk} d{d} causal{}", causal as u8),
        rel_ref,
    }
}

fn main() {
    use cubecl::wgpu::{WgpuDevice, WgpuRuntime};

    let mode = std::env::args().nth(1).unwrap_or_else(|| "scalar".into());

    let mc = WgpuRuntime::client(&WgpuDevice::default());
    if mode == "probe8" {
        eprintln!("=== flash-metal-check :: 8x8x8 cmma probe ===");
        eprintln!("runtime = {:?}\n", WgpuRuntime::name(&mc));
        probe8(&mc);
        return;
    }
    if mode == "bench" {
        eprintln!("=== flash-metal-check :: warm GPU A/B -- flash cmma(8x8) vs flash scalar, SAME wgpu<msl> path ===");
        eprintln!("runtime = {:?}   (device timestamps; buffers reused, only dispatch loop timed)\n", WgpuRuntime::name(&mc));
        // Same 10 shapes as the correctness gate.
        let shapes: [(&str, usize, usize, usize, usize, usize, bool); 10] = [
            ("decode kv1", 4, 2, 1, 1, 32, false),
            ("decode kv17 (tail)", 4, 2, 1, 17, 32, false),
            ("decode kv128", 4, 2, 1, 128, 32, false),
            ("decode kv512 GQA4", 8, 2, 1, 512, 64, false),
            ("prefill qt1 causal GQA2", 4, 2, 16, 128, 32, true),
            ("prefill qt1 noncausal GQA2", 4, 2, 16, 128, 32, false),
            ("prefill qt1 MHA causal d64", 4, 4, 16, 64, 64, true),
            ("prefill 3-tile causal (aligned)", 4, 2, 48, 48, 32, true),
            ("prefill causal tail sq40 d64", 6, 3, 40, 40, 64, true),
            ("prefill 512 causal MHA d128", 2, 1, 512, 512, 128, true),
        ];
        eprintln!("{:<34} {:>13} {:>13} {:>11}   {}", "case", "cmma us", "scalar us", "cmma speedup", "timing");
        for (tag, nh, nkv, sq, sk, d, causal) in shapes {
            let (c, mth) = bench_shape(&mc, Target::Metal, nh, nkv, sq, sk, d, causal, 32);
            let (s, _) = bench_shape(&mc, Target::Cpu, nh, nkv, sq, sk, d, causal, 32);
            eprintln!("{:<34} {:>13.3} {:>13.3} {:>10.2}x   {}", tag, c / 1e3, s / 1e3, s / c, mth);
        }
        return;
    }
    let (target, arm) = match mode.as_str() {
        "cmma" => (Target::Metal, "cmma (f16 simdgroup_matrix)"),
        _ => (Target::Cpu, "scalar (portable MAC oracle)"),
    };
    let tname = WgpuRuntime::name(&mc);
    eprintln!("=== flash-metal-check ===");
    eprintln!("runtime = {tname:?}   Target::of = {:?}", Target::of(&mc));
    eprintln!("island arm under test = {arm}  (tag = {target:?})\n");

    // The EXACT 10 shapes from flash_matches_materialized_ref_on_cpu.
    let rows = vec![
        gate(&mc, target, 4, 2, 1, 1, 32, false, 32, "decode kv1"),
        gate(&mc, target, 4, 2, 1, 17, 32, false, 32, "decode kv17 (tail)"),
        gate(&mc, target, 4, 2, 1, 128, 32, false, 32, "decode kv128"),
        gate(&mc, target, 8, 2, 1, 512, 64, false, 32, "decode kv512 GQA4"),
        gate(&mc, target, 4, 2, 16, 128, 32, true, 32, "prefill qt1 causal GQA2"),
        gate(&mc, target, 4, 2, 16, 128, 32, false, 32, "prefill qt1 noncausal GQA2"),
        gate(&mc, target, 4, 4, 16, 64, 64, true, 32, "prefill qt1 MHA causal d64"),
        gate(&mc, target, 4, 2, 48, 48, 32, true, 32, "prefill 3-tile causal GQA2 (aligned)"),
        gate(&mc, target, 6, 3, 40, 40, 64, true, 32, "prefill causal GQA2 tail sq40 d64"),
        gate(&mc, target, 2, 1, 512, 512, 128, true, 32, "prefill 512 causal MHA d128"),
    ];

    eprintln!("{:<40} {:<40} {:>10}", "case", "shape", "rel_ref");
    let (mut pass_1e5, mut pass_2e3, mut pass_2e2, mut ran) = (0, 0, 0, 0);
    let total = rows.len();
    for r in &rows {
        match &r.rel_ref {
            Ok(rr) => {
                ran += 1;
                if *rr < 1e-5 {
                    pass_1e5 += 1;
                }
                if *rr < 2e-3 {
                    pass_2e3 += 1;
                }
                if *rr < 2e-2 {
                    pass_2e2 += 1;
                }
                eprintln!("{:<40} {:<40} {:>10.2e}", r.tag, r.shape, rr);
            }
            Err(e) => {
                let short: String = e.lines().next().unwrap_or("").chars().take(90).collect();
                eprintln!("{:<40} {:<40} {:>10}   PANIC(main): {}", r.tag, r.shape, "ERR", short);
            }
        }
    }
    eprintln!(
        "\narm={arm}\nran {ran}/{total}   pass<1e-5 (bit-exact) {pass_1e5}/{total}   pass<2e-3 (scalar-grade) {pass_2e3}/{total}   pass<2e-2 (f16-grade) {pass_2e2}/{total}"
    );
    eprintln!("(for the cmma arm, watch stderr for '16x16x16 fragments not supported' worker-thread panics: rel~=1.0 means the zero buffer came back)");
}
