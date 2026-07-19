//! Norm ops in the DSL: RMSNorm and LayerNorm, one source -> every backend.
//!
//! One thread per row: each invocation reduces its own row (mean / mean-square) then normalizes it.
//! Correctness-first; the block-per-row shared-mem reduction (see `quant::matvec_q8_dp4a_blk`) is the
//! perf follow-up with the identical shape. `n` (the normalized dimension) is comptime, so the bounded
//! loops lower cleanly and no runtime `.len()` metadata buffer is needed for it.

use crate::prelude::*;
use crate::tune::Tuned;

/// RMSNorm over the last dim: `out[i] = x[i] / sqrt(mean(x^2) + eps) * w[i]`, per row of `n`.
#[kernel(targets(cuda, rocm, metal, vulkan, webgpu, cpu), unchecked)]
pub fn rms_norm<F: Float>(
    x: &Array<F>,
    w: &Array<F>,
    out: &mut Array<F>,
    eps: &Array<F>,
    #[comptime] n: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() / n {
        let base = row * n;
        let mut ss = F::new(0.0);
        for i in 0..n {
            let v = x[base + i];
            ss += v * v;
        }
        let denom = (ss / F::cast_from(n as u32) + eps[0]).sqrt();
        for i in 0..n {
            out[base + i] = x[base + i] / denom * w[i];
        }
    }
}

/// Block-per-row RMSNorm: `nt` threads cooperate on one row -- COALESCED reads (adjacent threads hit
/// adjacent addresses) + shared-mem tree reduction. The memory-bound win over one-thread-per-row (each
/// thread strides a full row, uncoalesced) is large, and it stays fast at low row counts (decode, rows=1).
/// Same shape as `quant::matvec_q8_dp4a_blk`. Block kernels are GPU-only (cubecl-cpu has no cooperative blocks).
///
/// `n` (the normalized dim) is RUNTIME (`ndim[0]`), so one compiled kernel is a drop-in for every model's
/// hidden size and any row width -- no per-dim specialization, no `n % nt == 0` requirement (strided guard).
/// Only `nt` (the block/shared-mem size) is comptime, because shared memory is statically sized.
///
/// ROCm: this is the production `rms_norm` on the ROCm backend. It lowers to a HIP source artifact
/// (metadata-free -> `has_info=false` -> a clean pointer-only `extern "C" __global__` signature) that
/// hanzo-ml compiles + launches through its existing rocm-rs pipeline, on its own device buffers -- the
/// ROCm mirror of the Vulkan `.spv` collapse. `eps`/`ndim` ride cached 1-elem device buffers (capture-clean).
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm), unchecked)]
pub fn rms_norm_blk<F: Float>(
    x: &Array<F>,
    w: &Array<F>,
    out: &mut Array<F>,
    eps: &Array<f32>,      // f32 regardless of F -- the reduction + scale are f32 (see below)
    ndim: &Array<u32>,     // ndim[0] = n; runtime so the kernel is dim-agnostic
    #[comptime] nt: usize, // threads per block (one block per row); shared-mem size
    #[comptime] tgt: Target, // island scrutinee: picks the block-reduction idiom per backend
) {
    // Reduce + scale in f32 regardless of the I/O dtype F, matching every hand-written norm kernel
    // (ggml/ROCm `rmsnorm<T>`, CUDA, Metal): f16 sum-of-squares overflows/underflows, so accumulate in
    // f32 and cast F only at load/store. For F=f32 every cast is the identity.
    let n = ndim[0] as usize;
    let base = CUBE_POS as usize * n;
    let step = CUBE_DIM as usize;
    let t = UNIT_POS as usize;
    let mut partial = 0f32;
    let mut idx = t; // seed from a runtime builtin (comptime consts can't be mutated)
    while idx < n {
        let v = f32::cast_from(x[base + idx]);
        partial += v * v;
        idx += step;
    }
    let mut smem = SharedMemory::<f32>::new(nt);
    // Block reduction, target-gated (island): the whole point of the ROCm collapse's perf parity. The
    // hand-written ROCm `rmsnorm` reduces via warp-shuffle (`__shfl_xor`) + a single cross-warp shared
    // pass -- decode (rows=1) is a SINGLE block, so the tree reduction's log(nt) `__syncthreads` is ~2x
    // slower. ROCm therefore uses `plane_sum` (lowers to `__shfl`) + one cross-warp combine, matching the
    // incumbent. Every other backend takes the shared-mem tree (`default`), so the shipped Vulkan .spv --
    // regenerated at target=Vulkan -- is byte-identical to the one on main. `s` = the block sum-of-squares.
    let s = island! {
        rocm => {
            let ws = plane_sum(partial); // shuffle-reduce within each plane (warp)
            let nwarps = CUBE_DIM / PLANE_DIM;
            if UNIT_POS % PLANE_DIM == 0 {
                smem[(UNIT_POS / PLANE_DIM) as usize] = ws;
            }
            sync_cube();
            // EVERY warp reads the per-warp partials and reduces -> all threads get the total with a
            // SINGLE barrier (no broadcast), exactly like the incumbent's `s_sum` + second warp_reduce.
            let lane = UNIT_POS % PLANE_DIM;
            let mut v = 0f32;
            if lane < nwarps {
                v = smem[lane as usize];
            }
            plane_sum(v)
        }
        default => {
            smem[t] = partial;
            sync_cube();
            let mut stride = CUBE_DIM / 2;
            while stride > 0 {
                if UNIT_POS < stride {
                    let v = smem[(UNIT_POS + stride) as usize];
                    smem[t] += v;
                }
                sync_cube();
                stride /= 2;
            }
            smem[0]
        }
    };
    let denom = (s / f32::cast_from(n as u32) + eps[0]).sqrt();
    let mut o = t;
    while o < n {
        out[base + o] = F::cast_from(f32::cast_from(x[base + o]) / denom * f32::cast_from(w[o]));
        o += step;
    }
}

/// Fused residual-add + RMSNorm, block-per-row (coalesced + shared-mem reduction). Emits BOTH
/// `s = x + res` and `y = s / sqrt(mean(s^2) + eps) * alpha` in one dispatch -- bit-identical to
/// `add.comp` then `rms_norm.comp` (same f32 ops, same order), the coalesced twin of the naive
/// per-row `add_rmsnorm.comp`. GPU-only (cubecl-cpu has no cooperative blocks).
///
/// ROCm: the production fused residual-add + rmsnorm on the ROCm backend, same collapse as `rms_norm_blk`.
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm), unchecked)]
pub fn add_rmsnorm_blk<F: Float>(
    x: &Array<F>,
    res: &Array<F>,
    alpha: &Array<F>,
    s_out: &mut Array<F>,
    y: &mut Array<F>,
    eps: &Array<f32>,      // f32 regardless of F (see `rms_norm_blk`)
    ndim: &Array<u32>,     // ndim[0] = n; runtime so the kernel is dim-agnostic
    #[comptime] nt: usize, // threads per block (one block per row); shared-mem size
    #[comptime] tgt: Target, // island scrutinee (see `rms_norm_blk`)
) {
    // Same f32-internal reduce/scale as `rms_norm_blk`: sum in f32, cast F only at load/store. The
    // residual add `s = x + res` is done in f32 then cast to F for `s_out` (the new residual stream),
    // exactly as the hand-written ROCm `add_rmsnorm`. Byte-identical for F=f32; bit-faithful for f16.
    let n = ndim[0] as usize;
    let base = CUBE_POS as usize * n;
    let step = CUBE_DIM as usize;
    let t = UNIT_POS as usize;
    // Pass 1: write the summed residual stream and accumulate sum-of-squares.
    let mut partial = 0f32;
    let mut idx = t;
    while idx < n {
        let v = f32::cast_from(x[base + idx]) + f32::cast_from(res[base + idx]);
        s_out[base + idx] = F::cast_from(v);
        partial += v * v;
        idx += step;
    }
    let mut smem = SharedMemory::<f32>::new(nt);
    // Target-gated block reduction (see `rms_norm_blk`): warp-shuffle on ROCm, shared-mem tree elsewhere.
    let ss = island! {
        rocm => {
            let ws = plane_sum(partial);
            let nwarps = CUBE_DIM / PLANE_DIM;
            if UNIT_POS % PLANE_DIM == 0 {
                smem[(UNIT_POS / PLANE_DIM) as usize] = ws;
            }
            sync_cube();
            let mut v = 0f32;
            if UNIT_POS < nwarps {
                v = smem[t];
            }
            let comb = plane_sum(v);
            if UNIT_POS == 0 {
                smem[0] = comb;
            }
            sync_cube();
            smem[0]
        }
        default => {
            smem[t] = partial;
            sync_cube();
            let mut stride = CUBE_DIM / 2;
            while stride > 0 {
                if UNIT_POS < stride {
                    let v = smem[(UNIT_POS + stride) as usize];
                    smem[t] += v;
                }
                sync_cube();
                stride /= 2;
            }
            smem[0]
        }
    };
    let denom = (ss / f32::cast_from(n as u32) + eps[0]).sqrt();
    // Pass 2: normalize. Recompute (x+res) in f32 (bit-identical to s_out, avoids reading a writeonly buffer).
    let mut o = t;
    while o < n {
        let s = f32::cast_from(x[base + o]) + f32::cast_from(res[base + o]);
        y[base + o] = F::cast_from(s / denom * f32::cast_from(alpha[o]));
        o += step;
    }
}

/// LayerNorm over the last dim: `out[i] = (x[i] - mean) / sqrt(var + eps) * w[i] + b[i]`, per row of `n`.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn layer_norm<F: Float>(
    x: &Array<F>,
    w: &Array<F>,
    b: &Array<F>,
    out: &mut Array<F>,
    eps: &Array<F>,
    #[comptime] n: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() / n {
        let base = row * n;
        let ninv = F::new(1.0) / F::cast_from(n as u32);
        let mut sum = F::new(0.0);
        for i in 0..n {
            sum += x[base + i];
        }
        let mean = sum * ninv;
        let mut var = F::new(0.0);
        for i in 0..n {
            let d = x[base + i] - mean;
            var += d * d;
        }
        let denom = (var * ninv + eps[0]).sqrt();
        for i in 0..n {
            out[base + i] = (x[base + i] - mean) / denom * w[i] + b[i];
        }
    }
}

/// Host launch for RMSNorm, generic over the runtime (CPU / Vulkan / Metal / CUDA / ROCm).
pub fn rms_norm_run<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Vec<f32> {
    let xh = client.create_from_slice(f32::as_bytes(x));
    let wh = client.create_from_slice(f32::as_bytes(w));
    let eph = client.create_from_slice(f32::as_bytes(&[eps]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows * n]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    unsafe {
        rms_norm::launch_unchecked::<f32, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows * n),
            ArrayArg::from_raw_parts(eph.clone(), 1),
            n,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Host launch for the block-per-row RMSNorm: one block per row, `nt` cooperating threads. Generic
/// over the I/O element `E` (f32 or f16 -- `eps` is always f32, matching the kernel). f32 callers
/// infer `E=f32` from the slice, so this stays source-compatible.
pub fn rms_norm_blk_run<E: Float + CubeElement + Default, R: Runtime>(
    client: &ComputeClient<R>,
    x: &[E],
    w: &[E],
    rows: usize,
    n: usize,
    eps: f32,
    nt: usize,
) -> Vec<E> {
    let xh = client.create_from_slice(E::as_bytes(x));
    let wh = client.create_from_slice(E::as_bytes(w));
    let eph = client.create_from_slice(f32::as_bytes(&[eps]));
    let ndh = client.create_from_slice(u32::as_bytes(&[n as u32]));
    let oh = client.create_from_slice(E::as_bytes(&vec![E::default(); rows * n]));
    unsafe {
        rms_norm_blk::launch_unchecked::<E, R>(
            client,
            Grid::Static(rows as u32, 1, 1),
            Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows * n),
            ArrayArg::from_raw_parts(eph.clone(), 1),
            ArrayArg::from_raw_parts(ndh.clone(), 1),
            nt,
            Target::of(client),
        );
    }
    E::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Kernel-only timing (ms/dispatch) for the block RMSNorm -- reads the output handle to force
/// completion, matching `quant::matvec_q8_dp4a_blk_run`. Returns (output, ms/dispatch).
pub fn rms_norm_blk_bench<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
    nt: usize,
    iters: usize,
) -> (Vec<f32>, f64) {
    let xh = client.create_from_slice(f32::as_bytes(x));
    let wh = client.create_from_slice(f32::as_bytes(w));
    let eph = client.create_from_slice(f32::as_bytes(&[eps]));
    let ndh = client.create_from_slice(u32::as_bytes(&[n as u32]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows * n]));
    let launch = |c: &ComputeClient<R>| unsafe {
        rms_norm_blk::launch_unchecked::<f32, R>(
            c,
            Grid::Static(rows as u32, 1, 1),
            Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows * n),
            ArrayArg::from_raw_parts(eph.clone(), 1),
            ArrayArg::from_raw_parts(ndh.clone(), 1),
            nt,
            Target::of(client),
        );
    };
    for _ in 0..3 {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters {
        launch(client);
    }
    let out = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    (f32::from_bytes(&out).to_vec(), ms)
}

/// Host launch for LayerNorm.
pub fn layer_norm_run<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    w: &[f32],
    b: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Vec<f32> {
    let xh = client.create_from_slice(f32::as_bytes(x));
    let wh = client.create_from_slice(f32::as_bytes(w));
    let bh = client.create_from_slice(f32::as_bytes(b));
    let eph = client.create_from_slice(f32::as_bytes(&[eps]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows * n]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    unsafe {
        layer_norm::launch_unchecked::<f32, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(bh.clone(), b.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows * n),
            ArrayArg::from_raw_parts(eph.clone(), 1),
            n,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// CPU oracle for RMSNorm -- the trusted reference the DSL kernel is gated against.
pub fn rms_norm_ref(x: &[f32], w: &[f32], rows: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for row in 0..rows {
        let base = row * n;
        let ss: f32 = (0..n).map(|i| x[base + i] * x[base + i]).sum();
        let denom = (ss / n as f32 + eps).sqrt();
        for i in 0..n {
            out[base + i] = x[base + i] / denom * w[i];
        }
    }
    out
}

/// CPU oracle for fused add+RMSNorm: returns `(s = x+res, y = rms_norm(s)*alpha)`.
pub fn add_rmsnorm_ref(
    x: &[f32],
    res: &[f32],
    alpha: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> (Vec<f32>, Vec<f32>) {
    let mut s = vec![0.0f32; rows * n];
    let mut y = vec![0.0f32; rows * n];
    for row in 0..rows {
        let base = row * n;
        let mut ss = 0.0f32;
        for i in 0..n {
            let v = x[base + i] + res[base + i];
            s[base + i] = v;
            ss += v * v;
        }
        let denom = (ss / n as f32 + eps).sqrt();
        for i in 0..n {
            y[base + i] = (x[base + i] + res[base + i]) / denom * alpha[i];
        }
    }
    (s, y)
}

/// Host launch for the block add+RMSNorm (GPU-only). Returns `(s, y)`.
pub fn add_rmsnorm_blk_run<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    res: &[f32],
    alpha: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
    nt: usize,
) -> (Vec<f32>, Vec<f32>) {
    let xh = client.create_from_slice(f32::as_bytes(x));
    let rh = client.create_from_slice(f32::as_bytes(res));
    let ah = client.create_from_slice(f32::as_bytes(alpha));
    let eph = client.create_from_slice(f32::as_bytes(&[eps]));
    let ndh = client.create_from_slice(u32::as_bytes(&[n as u32]));
    let sh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows * n]));
    let yh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows * n]));
    unsafe {
        add_rmsnorm_blk::launch_unchecked::<f32, R>(
            client,
            Grid::Static(rows as u32, 1, 1),
            Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(rh.clone(), res.len()),
            ArrayArg::from_raw_parts(ah.clone(), alpha.len()),
            ArrayArg::from_raw_parts(sh.clone(), rows * n),
            ArrayArg::from_raw_parts(yh.clone(), rows * n),
            ArrayArg::from_raw_parts(eph.clone(), 1),
            ArrayArg::from_raw_parts(ndh.clone(), 1),
            nt,
            Target::of(client),
        );
    }
    let s = f32::from_bytes(&client.read_one_unchecked(sh)).to_vec();
    let y = f32::from_bytes(&client.read_one_unchecked(yh)).to_vec();
    (s, y)
}

/// CPU oracle for LayerNorm.
pub fn layer_norm_ref(x: &[f32], w: &[f32], b: &[f32], rows: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * n];
    for row in 0..rows {
        let base = row * n;
        let mean: f32 = (0..n).map(|i| x[base + i]).sum::<f32>() / n as f32;
        let var: f32 = (0..n).map(|i| (x[base + i] - mean).powi(2)).sum::<f32>() / n as f32;
        let denom = (var + eps).sqrt();
        for i in 0..n {
            out[base + i] = (x[base + i] - mean) / denom * w[i] + b[i];
        }
    }
    out
}

// ============================================================================================
// Autotuned RMSNorm -- ONE source, the schedule exposed as comptime knobs the tuner picks over.
// The "collapse": no per-device fork; the same `rms_norm_tuned` is monomorphized per comptime tuple,
// and the tuner caches the winning tuple per (device, rows x n). Mirror of `quant::matvec_q8_dp4a_tuned`.
// ============================================================================================

/// Autotuned RMSNorm source. `rpt` = rows-per-thread (the unroll knob); the launch block size (cube
/// dim) is the other knob, set by the host. Every `(rpt, block)` is a distinct compiled kernel yet
/// bit-IDENTICAL to `rms_norm_ref`: the per-row sum-of-squares is the same left fold in the same order,
/// so only the thread->row mapping and occupancy differ. This is the object the autotuner schedules.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn rms_norm_tuned<F: Float>(
    x: &Array<F>,
    w: &Array<F>,
    out: &mut Array<F>,
    eps: &Array<F>,
    #[comptime] n: usize,
    #[comptime] rpt: usize,
) {
    let nrows = out.len() / n;
    let base = ABSOLUTE_POS * rpt;
    #[unroll]
    for r in 0..rpt {
        let row = base + r;
        if row < nrows {
            let b = row * n;
            let mut ss = F::new(0.0);
            for i in 0..n {
                let v = x[b + i];
                ss += v * v;
            }
            let denom = (ss / F::cast_from(n as u32) + eps[0]).sqrt();
            for i in 0..n {
                out[b + i] = x[b + i] / denom * w[i];
            }
        }
    }
}

/// Kernel-only bench for one `(rpt, block)` schedule of `rms_norm_tuned`; returns `(output, ms/dispatch)`
/// -- the [`crate::tune::Variant`] contract. `iters == 1` on a cache hit (single launch, produce output);
/// many iters when tuning. Grid rounds up so `ceil(rows/rpt)` row-groups are covered.
pub fn rms_norm_tuned_bench<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
    rpt: usize,
    block: u32,
    iters: usize,
) -> (Vec<f32>, f64) {
    let xh = client.create_from_slice(f32::as_bytes(x));
    let wh = client.create_from_slice(f32::as_bytes(w));
    let eph = client.create_from_slice(f32::as_bytes(&[eps]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows * n]));
    let grid = (rows as u32).div_ceil(block * rpt as u32);
    let launch = |c: &ComputeClient<R>| unsafe {
        rms_norm_tuned::launch_unchecked::<f32, R>(
            c,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows * n),
            ArrayArg::from_raw_parts(eph.clone(), 1),
            n,
            rpt,
        );
    };
    for _ in 0..3 {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters {
        launch(client);
    }
    let out = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    (f32::from_bytes(&out).to_vec(), ms)
}

/// The 4-variant RMSNorm set (cube dim {64,128,256} x unroll {1,2}) as a reusable [`Tuned`], so tests
/// can drive it against an explicit tuner. All four are bit-identical; they differ only in schedule.
pub fn rms_norm_tuned_set<'a, R: Runtime>(
    client: &'a ComputeClient<R>,
    x: &'a [f32],
    w: &'a [f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Tuned<'a, Vec<f32>> {
    Tuned::new("rms_norm", format!("rows={rows},n={n}"))
        .variant("b64_r1", move |it| rms_norm_tuned_bench(client, x, w, rows, n, eps, 1, 64, it))
        .variant("b128_r1", move |it| rms_norm_tuned_bench(client, x, w, rows, n, eps, 1, 128, it))
        .variant("b256_r1", move |it| rms_norm_tuned_bench(client, x, w, rows, n, eps, 1, 256, it))
        .variant("b64_r2", move |it| rms_norm_tuned_bench(client, x, w, rows, n, eps, 2, 64, it))
}

/// Autotuned RMSNorm: tune (or read the cache) over the schedule knobs and run the winner. ONE source,
/// picked per `(device, rows x n)`, no forks. The production surface.
pub fn rms_norm_autotuned<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32],
    w: &[f32],
    rows: usize,
    n: usize,
    eps: f32,
) -> Vec<f32> {
    rms_norm_tuned_set(client, x, w, rows, n, eps).run(client)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(1e-6))
            .fold(0.0, f32::max)
    }

    fn data(rows: usize, n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut s = 0x2545F491_4F6CDD1Du64;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2000) as f32 / 1000.0 - 1.0
        };
        let x: Vec<f32> = (0..rows * n).map(|_| next()).collect();
        let w: Vec<f32> = (0..n).map(|_| next() * 0.5 + 1.0).collect();
        let b: Vec<f32> = (0..n).map(|_| next() * 0.1).collect();
        (x, w, b)
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn rms_norm_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, n) = (37, 128);
        let (x, w, _) = data(rows, n);
        let c = CpuRuntime::client(&CpuDevice::default());
        let got = rms_norm_run::<CpuRuntime>(&c, &x, &w, rows, n, EPS);
        let want = rms_norm_ref(&x, &w, rows, n, EPS);
        let rel = max_rel(&want, &got);
        eprintln!("[rms_norm  CPU] {rows}x{n} max_rel={rel:.2e}");
        assert!(rel < 2e-3, "rms_norm max_rel {rel}");
    }

    /// The ROCm/HIP dispatch gate: the SAME `rms_norm` DSL source lowered and dispatched through
    /// cubecl-hip on real AMD hardware (gfx1151), bit-compared to the CPU oracle `rms_norm_ref`.
    /// This is the falsifiable proof that the cubecl-hip -> ROCm 7.13 seam works end to end (device
    /// query via the R0600 bindings + module load + kernel launch + readback). Requires a HIP device.
    #[cfg(feature = "rocm")]
    #[test]
    fn rms_norm_rocm_bit_exact() {
        use hanzo_cubecl_hip::{AmdDevice, HipRuntime};
        let (rows, n) = (37, 128);
        let (x, w, _) = data(rows, n);
        let c = HipRuntime::client(&AmdDevice::default());
        let got = rms_norm_run::<HipRuntime>(&c, &x, &w, rows, n, EPS);
        let want = rms_norm_ref(&x, &w, rows, n, EPS);
        let rel = max_rel(&want, &got);
        eprintln!("[rms_norm ROCM] {rows}x{n} max_rel={rel:.2e}");
        assert!(rel < 2e-3, "rms_norm ROCm max_rel {rel}");
    }

    /// Autotuned RMSNorm on the CPU runtime: (1) every schedule variant is bit-IDENTICAL to the oracle
    /// (the knobs don't touch the numerics), and (2) the tuner benchmarks all 4 on a miss, caches the
    /// winner, and the second call skips timing entirely.
    #[cfg(feature = "cpu")]
    #[test]
    fn rms_norm_autotune_cpu_bit_exact_and_cached() {
        use crate::tune::Tuner;
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, n) = (64usize, 128usize);
        let (x, w, _) = data(rows, n);
        let c = CpuRuntime::client(&CpuDevice::default());
        let want = rms_norm_ref(&x, &w, rows, n, EPS);
        let wbits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();

        // (1) each of the 4 schedules is byte-identical to the oracle (same left-fold, same order).
        for &(rpt, block) in &[(1usize, 64u32), (1, 128), (1, 256), (2, 64)] {
            let (got, _ms) = rms_norm_tuned_bench::<CpuRuntime>(&c, &x, &w, rows, n, EPS, rpt, block, 1);
            let gbits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            assert_eq!(gbits, wbits, "rms_norm_tuned rpt={rpt} block={block} not bit-exact vs oracle");
        }

        // (2) select + cache, against an isolated temp tuner (no env, no globals).
        let dir = std::env::temp_dir().join(format!(
            "hk-tune-rms-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let tuner = Tuner::new(&dir);
        let p = rms_norm_tuned_set::<CpuRuntime>(&c, &x, &w, rows, n, EPS).pick_with(&tuner, "cpu");
        assert!(!p.from_cache && p.benched == 4, "first call must benchmark all 4 variants");
        assert_eq!(p.output.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), wbits, "winner not bit-exact");
        eprintln!("[rms_norm autotune CPU] winner={} timings={:?}", p.winner, p.timings);

        let p2 = rms_norm_tuned_set::<CpuRuntime>(&c, &x, &w, rows, n, EPS).pick_with(&tuner, "cpu");
        assert!(p2.from_cache && p2.benched == 0, "second call must hit the cache and skip timing");
        assert_eq!(p2.winner, p.winner, "cache returned a different winner");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "cpu")]
    #[test]
    fn layer_norm_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, n) = (37, 128);
        let (x, w, b) = data(rows, n);
        let c = CpuRuntime::client(&CpuDevice::default());
        let got = layer_norm_run::<CpuRuntime>(&c, &x, &w, &b, rows, n, EPS);
        let want = layer_norm_ref(&x, &w, &b, rows, n, EPS);
        let rel = max_rel(&want, &got);
        eprintln!("[layer_norm CPU] {rows}x{n} max_rel={rel:.2e}");
        assert!(rel < 2e-3, "layer_norm max_rel {rel}");
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn rms_norm_blk_vulkan_bit_exact_and_bench() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let (rows, n, nt) = (4096usize, 4096usize, 256usize); // 67MB x -> cache-busting on gfx1151 MALL
        let (x, w, _) = data(rows, n);
        let c = WgpuRuntime::client(&WgpuDevice::default());
        // bit-exact vs the CPU oracle (block tree-reduction reorders the sum -> f32 tol)
        let want = rms_norm_ref(&x, &w, rows, n, EPS);
        let (got, ms) = rms_norm_blk_bench::<WgpuRuntime>(&c, &x, &w, rows, n, EPS, nt, 50);
        let rel = max_rel(&want, &got);
        let bytes = (2 * rows * n + n) as f64 * 4.0; // read x + write out (+w)
        let gbps = bytes / (ms * 1e6);
        eprintln!("[rms_norm_blk VULKAN] {rows}x{n} nt={nt}  max_rel={rel:.2e}  {ms:.3} ms  {gbps:.0} GB/s");
        assert!(rel < 2e-3, "rms_norm_blk max_rel {rel}");
        // dim-agnostic: n not a multiple of nt, and n < nt -- the strided guard must still be exact
        for &m in &[130usize, 1536, 3072] {
            let (xx, ww, _) = data(11, m);
            let g = rms_norm_blk_run::<f32, WgpuRuntime>(&c, &xx, &ww, 11, m, EPS, 256);
            let r = max_rel(&rms_norm_ref(&xx, &ww, 11, m, EPS), &g);
            eprintln!("[rms_norm_blk VULKAN] 11x{m} (n%nt!=0)  max_rel={r:.2e}");
            assert!(r < 2e-3, "rms_norm_blk n={m} max_rel {r}");
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn add_rmsnorm_blk_vulkan_bit_exact() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let c = WgpuRuntime::client(&WgpuDevice::default());
        // distinct residual stream (different seed) so s = x+res is a real add, not a doubling
        let gen_res = |rows: usize, n: usize| -> Vec<f32> {
            let mut s = 0x9E3779B9_7F4A7C15u64;
            (0..rows * n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s % 2000) as f32 / 1000.0 - 1.0
                })
                .collect()
        };
        // 4096^2 cache-busting + dim-agnostic shapes (n%nt!=0, n<nt)
        for &(rows, n, nt) in &[(4096usize, 4096usize, 256usize), (11, 130, 256), (11, 1536, 256), (7, 3072, 256)] {
            let (x, alpha, _) = data(rows, n);
            let res = gen_res(rows, n);
            let (ws, wy) = add_rmsnorm_ref(&x, &res, &alpha, rows, n, EPS);
            let (gs, gy) = add_rmsnorm_blk_run::<WgpuRuntime>(&c, &x, &res, &alpha, rows, n, EPS, nt);
            let (rs, ry) = (max_rel(&ws, &gs), max_rel(&wy, &gy));
            eprintln!("[add_rmsnorm_blk VULKAN] {rows}x{n} nt={nt}  s_rel={rs:.2e} y_rel={ry:.2e}");
            assert!(rs < 2e-3 && ry < 2e-3, "add_rmsnorm {rows}x{n} s={rs} y={ry}");
        }
    }

    /// The ROCm/HIP gate for the block-per-row RMSNorm -- the DSL kernel hanzo-ml's ROCm backend
    /// migrates its hand-written `rmsnorm` to. Lowered through cubecl-hip on gfx1151, bit-compared to
    /// the CPU oracle at the migration's gate shapes (hidden 4096/5120, rows 1 decode + 512 prefill)
    /// plus dim-agnostic (n%nt!=0, n<nt). Run with `CUBECL_DEBUG_LOG=<file>` to also DUMP the generated
    /// HIP source (the checked-in `.hip` artifact ml compiles). Requires a HIP device.
    #[cfg(feature = "rocm")]
    #[test]
    fn rms_norm_blk_rocm_bit_exact() {
        use cubecl::hip::{AmdDevice, HipRuntime};
        let c = HipRuntime::client(&AmdDevice::default());
        for &(rows, n) in &[(1usize, 4096usize), (512, 4096), (1, 5120), (512, 5120), (11, 130), (7, 3072)] {
            let (x, w, _) = data(rows, n);
            let want = rms_norm_ref(&x, &w, rows, n, EPS);
            let got = rms_norm_blk_run::<f32, HipRuntime>(&c, &x, &w, rows, n, EPS, 1024);
            let rel = max_rel(&want, &got);
            eprintln!("[rms_norm_blk ROCM f32] {rows}x{n} nt=1024  max_rel={rel:.2e}");
            assert!(rel < 2e-3, "rms_norm_blk ROCm f32 {rows}x{n} max_rel {rel}");
        }
        // f16 I/O: the kernel reduces + scales in f32 (eps is f32), casting F only at load/store, so it
        // is bit-faithful to the incumbent ROCm `rmsnorm<half>` (f32 accumulation). Compare to the f32
        // oracle evaluated on the f16-ROUNDED inputs -> only f16 output rounding remains (tol 6e-3).
        use half::f16;
        for &(rows, n) in &[(1usize, 4096usize), (512, 4096), (7, 3072)] {
            let (xf, wf, _) = data(rows, n);
            let x: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
            let w: Vec<f16> = wf.iter().map(|&v| f16::from_f32(v)).collect();
            let xr: Vec<f32> = x.iter().map(|v| v.to_f32()).collect();
            let wr: Vec<f32> = w.iter().map(|v| v.to_f32()).collect();
            let want = rms_norm_ref(&xr, &wr, rows, n, EPS);
            let got_h = rms_norm_blk_run::<f16, HipRuntime>(&c, &x, &w, rows, n, EPS, 1024);
            let got: Vec<f32> = got_h.iter().map(|v| v.to_f32()).collect();
            let rel = max_rel(&want, &got);
            eprintln!("[rms_norm_blk ROCM f16] {rows}x{n} nt=1024  max_rel={rel:.2e}");
            assert!(rel < 6e-3, "rms_norm_blk ROCm f16 {rows}x{n} max_rel {rel}");
        }
    }

    /// The ROCm/HIP gate for the fused residual-add + RMSNorm block kernel (ml's `add_rms_norm` twin).
    #[cfg(feature = "rocm")]
    #[test]
    fn add_rmsnorm_blk_rocm_bit_exact() {
        use cubecl::hip::{AmdDevice, HipRuntime};
        let c = HipRuntime::client(&AmdDevice::default());
        let gen_res = |rows: usize, n: usize| -> Vec<f32> {
            let mut s = 0x9E3779B9_7F4A7C15u64;
            (0..rows * n)
                .map(|_| {
                    s ^= s << 13;
                    s ^= s >> 7;
                    s ^= s << 17;
                    (s % 2000) as f32 / 1000.0 - 1.0
                })
                .collect()
        };
        for &(rows, n) in &[(1usize, 4096usize), (512, 4096), (1, 5120), (512, 5120), (11, 130), (7, 3072)] {
            let (x, alpha, _) = data(rows, n);
            let res = gen_res(rows, n);
            let (ws, wy) = add_rmsnorm_ref(&x, &res, &alpha, rows, n, EPS);
            let (gs, gy) = add_rmsnorm_blk_run::<HipRuntime>(&c, &x, &res, &alpha, rows, n, EPS, 1024);
            let (rs, ry) = (max_rel(&ws, &gs), max_rel(&wy, &gy));
            eprintln!("[add_rmsnorm_blk ROCM] {rows}x{n} nt=1024  s_rel={rs:.2e} y_rel={ry:.2e}");
            assert!(rs < 2e-3 && ry < 2e-3, "add_rmsnorm_blk ROCm {rows}x{n} s={rs} y={ry}");
        }
    }

    #[cfg(feature = "metal")]
    #[test]
    fn norm_metal_bit_exact() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let (rows, n) = (37, 128);
        let (x, w, b) = data(rows, n);
        let c = WgpuRuntime::client(&WgpuDevice::default());
        let r = rms_norm_run::<WgpuRuntime>(&c, &x, &w, rows, n, EPS);
        let l = layer_norm_run::<WgpuRuntime>(&c, &x, &w, &b, rows, n, EPS);
        let blk = rms_norm_blk_run::<f32, WgpuRuntime>(&c, &x, &w, rows, n, EPS, n);
        let rr = max_rel(&rms_norm_ref(&x, &w, rows, n, EPS), &r);
        let lr = max_rel(&layer_norm_ref(&x, &w, &b, rows, n, EPS), &l);
        let br = max_rel(&rms_norm_ref(&x, &w, rows, n, EPS), &blk);
        eprintln!("[rms_norm METAL] {rr:.2e}  [layer_norm METAL] {lr:.2e}  [rms_norm_blk METAL] {br:.2e}");
        assert!(rr < 2e-3 && lr < 2e-3 && br < 2e-3);
    }
}
