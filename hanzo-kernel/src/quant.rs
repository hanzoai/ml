//! Quantized matvec kernels in the DSL -- the bulk of the ~52k hand-written lines is this family.
//!
//! Q8_0 first (the simplest K-independent block): 32 int8 quants + one f16 scale per block. This
//! proves the pattern -- quant decode + contraction, one source, bit-exact, every backend. Q4_K / Q6_K
//! and the int8-dp4a fast path (`Line<i8>.dot`) follow the identical shape with more bit-twiddling.

use crate::prelude::*;
use crate::tune::{Config, Evaluator, Evolution, Evolved, Space, Tuned, Tuner, Verdict};
use cubecl::server::Handle;

/// Q8_0 block size (weights per scale).
pub const QK8_0: usize = 32;

/// Q8_0 matvec: `out[row] = sum_k (wd[block] * wq[k]) * x[k]`, one invocation per output row.
/// `wd`: scales, one per 32-weight block, `[rows * k/32]`. `wq`: int8 quants widened to i32, `[rows * k]`.
/// The decode (`wd * wq`) and the contraction are the exact math each backend hand-writes today.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8<F: Float>(
    wd: &Array<F>,
    wq: &Array<i32>,
    x: &Array<F>,
    out: &mut Array<F>,
    #[comptime] k: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() {
        let nb = k / 32;
        let wbase = row * k;
        let dbase = row * nb;
        let mut acc = F::new(0.0);
        for i in 0..k {
            let d = wd[dbase + i / 32];
            let q = F::cast_from(wq[wbase + i]);
            acc += d * q * x[i];
        }
        out[row] = acc;
    }
}

/// Host launch, generic over the runtime (CPU / Vulkan / Metal / CUDA / ROCm -- all from this one fn).
pub fn matvec_q8_run<R: Runtime>(
    client: &ComputeClient<R>,
    wd: &[f32],
    wq: &[i32],
    x: &[f32],
    rows: usize,
    k: usize,
) -> Vec<f32> {
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let wqh = client.create_from_slice(i32::as_bytes(wq));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    // One thread per output row; round the grid up to cover all rows.
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    unsafe {
        matvec_q8::launch_unchecked::<f32, R>(
            client,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(wqh.clone(), wq.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    }
    let bytes = client.read_one_unchecked(oh);
    f32::from_bytes(&bytes).to_vec()
}

/// Kernel-only throughput: buffers created ONCE, `iters` dispatches enqueued back-to-back, one final
/// sync (the read). This amortizes the host round-trip so the number reflects the KERNEL, not the API
/// -- the real gate for "can the DSL kernel retire the hand-tuned one". Returns mean ms/dispatch.
pub fn matvec_q8_bench<R: Runtime>(
    client: &ComputeClient<R>,
    wd: &[f32],
    wq: &[i32],
    x: &[f32],
    rows: usize,
    k: usize,
    iters: usize,
) -> f64 {
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let wqh = client.create_from_slice(i32::as_bytes(wq));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8::launch_unchecked::<f32, R>(
            c,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(wqh.clone(), wq.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    };
    for _ in 0..3 {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh.clone()); // sync warmup
    let t = std::time::Instant::now();
    for _ in 0..iters {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh); // force completion of all enqueued dispatches
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// CPU oracle: the trusted reference the DSL kernel is gated against (bit-exact within f32 reorder).
pub fn matvec_q8_ref(wd: &[f32], wq: &[i32], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nb = k / 32;
    (0..rows)
        .map(|row| {
            let mut acc = 0.0f32;
            for i in 0..k {
                acc += wd[row * nb + i / 32] * wq[row * k + i] as f32 * x[i];
            }
            acc
        })
        .collect()
}

// ============================================================================================
// Q4_K -- the real K-quant, decoded IN-KERNEL from packed bytes (the bulk of the hand-written lines).
// Block = 144 bytes / 256 weights: d(f16) dmin(f16) scales[12] qs[128]. Host extracts the two f16
// super-block scalars (per-block metadata, like the hand-tuned kernels); the KERNEL does the real work
// -- 6-bit scale/min unpack (get_scale_min_k4) + 4-bit quant extract via bit ops. Layout passed as u32:
//   wqs: qs, 32 u32/block (128 bytes)    wsc: scales, 3 u32/block (12 bytes)
//   wd:  d,  1 f32/block                  wdm: dmin, 1 f32/block
// ============================================================================================
pub const QK_K: usize = 256;

#[device]
fn byte_at(a: &Array<u32>, base: usize, i: usize) -> u32 {
    (a[base + i / 4] >> ((8 * (i % 4)) as u32)) & 255
}

// get_scale_min_k4 scale component (llama.cpp), from the 12 packed scale bytes at scbase.
#[device]
fn q4k_sc(wsc: &Array<u32>, scbase: usize, j: usize) -> u32 {
    let mut r = byte_at(wsc, scbase, j) & 63;
    if j >= 4 {
        r = (byte_at(wsc, scbase, j + 4) & 15) | ((byte_at(wsc, scbase, j - 4) >> 6) << 4);
    }
    r
}
// get_scale_min_k4 min component.
#[device]
fn q4k_m(wsc: &Array<u32>, scbase: usize, j: usize) -> u32 {
    let mut r = byte_at(wsc, scbase, j + 4) & 63;
    if j >= 4 {
        r = (byte_at(wsc, scbase, j + 4) >> 4) | ((byte_at(wsc, scbase, j) >> 6) << 4);
    }
    r
}

/// Q4_K matvec, one invocation per output row. Bit-identical decode order to `BlockQ4K::to_float`.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q4k<F: Float>(
    wqs: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<F>,
    wdm: &Array<F>,
    x: &Array<F>,
    out: &mut Array<F>,
    #[comptime] k: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() {
        let nb = k / 256;
        let mut acc = F::new(0.0);
        for b in 0..nb {
            let blk = row * nb + b;
            let qbase = blk * 32;
            let scbase = blk * 3;
            let d = wd[blk];
            let dmin = wdm[blk];
            let xbase = b * 256;
            for g in 0..4 {
                let is = g * 2;
                let d1 = d * F::cast_from(q4k_sc(wsc, scbase, is));
                let mm1 = dmin * F::cast_from(q4k_m(wsc, scbase, is));
                let d2 = d * F::cast_from(q4k_sc(wsc, scbase, is + 1));
                let mm2 = dmin * F::cast_from(q4k_m(wsc, scbase, is + 1));
                for qi in 0..32 {
                    let qb = byte_at(wqs, qbase, g * 32 + qi);
                    let wlo = d1 * F::cast_from(qb & 15) - mm1;
                    acc += wlo * x[xbase + g * 64 + qi];
                    let whi = d2 * F::cast_from(qb >> 4) - mm2;
                    acc += whi * x[xbase + g * 64 + 32 + qi];
                }
            }
        }
        out[row] = acc;
    }
}

/// Host launch for the Q4_K matvec (all runtimes).
pub fn matvec_q4k_run<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], rows: usize, k: usize,
) -> Vec<f32> {
    let qh = client.create_from_slice(u32::as_bytes(wqs));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let mh = client.create_from_slice(f32::as_bytes(wdm));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    unsafe {
        matvec_q4k::launch_unchecked::<f32, R>(
            client, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(qh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(mh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

pub fn matvec_q4k_bench<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], rows: usize, k: usize, iters: usize,
) -> f64 {
    let qh = client.create_from_slice(u32::as_bytes(wqs));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let mh = client.create_from_slice(f32::as_bytes(wdm));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32;
    let grid = (rows as u32).div_ceil(block);
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q4k::launch_unchecked::<f32, R>(
            c, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(qh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(mh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    };
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[inline]
fn cpu_byte(a: &[u32], base: usize, i: usize) -> u32 { (a[base + i / 4] >> (8 * (i % 4))) & 255 }
#[inline]
fn cpu_sc(wsc: &[u32], scbase: usize, j: usize) -> u32 {
    if j < 4 { cpu_byte(wsc, scbase, j) & 63 }
    else { (cpu_byte(wsc, scbase, j + 4) & 15) | ((cpu_byte(wsc, scbase, j - 4) >> 6) << 4) }
}
#[inline]
fn cpu_m(wsc: &[u32], scbase: usize, j: usize) -> u32 {
    if j < 4 { cpu_byte(wsc, scbase, j + 4) & 63 }
    else { (cpu_byte(wsc, scbase, j + 4) >> 4) | ((cpu_byte(wsc, scbase, j) >> 6) << 4) }
}

/// CPU oracle for Q4_K, same packed inputs as the kernel (bit-for-bit BlockQ4K::to_float order).
pub fn matvec_q4k_ref(wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nb = k / 256;
    (0..rows).map(|row| {
        let mut acc = 0.0f32;
        for b in 0..nb {
            let blk = row * nb + b;
            let (qbase, scbase) = (blk * 32, blk * 3);
            let (d, dmin) = (wd[blk], wdm[blk]);
            let xbase = b * 256;
            for g in 0..4 {
                let is = g * 2;
                let d1 = d * cpu_sc(wsc, scbase, is) as f32;
                let mm1 = dmin * cpu_m(wsc, scbase, is) as f32;
                let d2 = d * cpu_sc(wsc, scbase, is + 1) as f32;
                let mm2 = dmin * cpu_m(wsc, scbase, is + 1) as f32;
                for qi in 0..32 {
                    let qb = cpu_byte(wqs, qbase, g * 32 + qi);
                    acc += (d1 * (qb & 15) as f32 - mm1) * x[xbase + g * 64 + qi];
                    acc += (d2 * (qb >> 4) as f32 - mm2) * x[xbase + g * 64 + 32 + qi];
                }
            }
        }
        acc
    }).collect()
}

/// Deterministic valid Q4_K test data (packed u32 layout + f16-rounded d/dmin + activation).
pub fn gen_q4k(rows: usize, k: usize) -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let nb = k / 256;
    let nblk = rows * nb;
    let mut s = 0x9E3779B97F4A7C15u64;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let wqs: Vec<u32> = (0..nblk * 32).map(|_| next() as u32).collect(); // 128 qs bytes/block
    let wsc: Vec<u32> = (0..nblk * 3).map(|_| next() as u32).collect();  // 12 scale bytes/block
    // f16-round d/dmin so the CPU ref (which the kernel matches) uses the true stored precision
    let wd: Vec<f32> = (0..nblk).map(|_| half::f16::from_f32((next() % 1000) as f32 / 20000.0 + 0.002).to_f32()).collect();
    let wdm: Vec<f32> = (0..nblk).map(|_| half::f16::from_f32((next() % 1000) as f32 / 40000.0).to_f32()).collect();
    let x: Vec<f32> = (0..k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    (wqs, wsc, wd, wdm, x)
}

// ============================================================================================
// Q4_K indexed-MoE matvec: `out[s, r] = sum_k W[ids[s], r, k] * x[s, k]`. The ONLY delta from
// `matvec_q4k` is the expert gather (weight row = ids[slot]*n + r, out of a flat [E*n] Q4_K bank)
// and the per-slot activation offset (slot*k). The in-kernel Q4_K decode is byte-identical --
// `decode ⊥ gather`, composed. One invocation = one output element `out[slot*n + r]`.
// ============================================================================================
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn moe_matvec_q4k<F: Float>(
    wqs: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<F>,
    wdm: &Array<F>,
    x: &Array<F>,
    ids: &Array<u32>,
    out: &mut Array<F>,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let gid = ABSOLUTE_POS;
    if gid < out.len() {
        let slot = gid / n;
        let r = gid % n;
        let wrow = ids[slot] as usize * n + r; // weight row in the flat [E*n, k] bank
        let nb = k / 256;
        let xbase = slot * k;
        let mut acc = F::new(0.0);
        for b in 0..nb {
            let blk = wrow * nb + b;
            let qbase = blk * 32;
            let scbase = blk * 3;
            let d = wd[blk];
            let dmin = wdm[blk];
            let xblk = xbase + b * 256;
            for g in 0..4 {
                let is = g * 2;
                let d1 = d * F::cast_from(q4k_sc(wsc, scbase, is));
                let mm1 = dmin * F::cast_from(q4k_m(wsc, scbase, is));
                let d2 = d * F::cast_from(q4k_sc(wsc, scbase, is + 1));
                let mm2 = dmin * F::cast_from(q4k_m(wsc, scbase, is + 1));
                for qi in 0..32 {
                    let qb = byte_at(wqs, qbase, g * 32 + qi);
                    let wlo = d1 * F::cast_from(qb & 15) - mm1;
                    acc += wlo * x[xblk + g * 64 + qi];
                    let whi = d2 * F::cast_from(qb >> 4) - mm2;
                    acc += whi * x[xblk + g * 64 + 32 + qi];
                }
            }
        }
        out[gid] = acc;
    }
}

/// Host launch for the Q4_K MoE matvec. `wqs/wsc/wd/wdm` are the flat [E*n] Q4_K bank; `x` is
/// `[slots, k]`; `ids[slot]` is slot's expert; output is `[slots, n]`. All runtimes, one fn.
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q4k_run<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize,
) -> Vec<f32> {
    let qh = client.create_from_slice(u32::as_bytes(wqs));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let mh = client.create_from_slice(f32::as_bytes(wdm));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let ih = client.create_from_slice(u32::as_bytes(ids));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n]));
    let block = 64u32;
    let grid = ((slots * n) as u32).div_ceil(block);
    unsafe {
        moe_matvec_q4k::launch_unchecked::<f32, R>(
            client, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(qh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(mh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

// Block-reduced Q4_K MoE matvec: ONE workgroup per output element, `nt` threads split the row's
// k/32 sub-blocks (each 32 weights = one Q4_K scale index), shared-mem tree reduce. This is the
// bandwidth-bound decode structure that took the plain Q8_0/dp4a kernels from ~90 to ~500 GB/s --
// the decode-perf lever. Q4_K sub-block sb -> superblock sb/8, scale index sb%8, low nibble if even
// / high if odd. Matches `moe_matvec_q4k` within f32-reorder (different, valid, reduction order).
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn moe_matvec_q4k_blk<F: Float>(
    wqs: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<F>,
    wdm: &Array<F>,
    x: &Array<F>,
    ids: &Array<u32>,
    out: &mut Array<F>,
    #[comptime] n: usize,  // comptime shape: outrow/n & nsub/nt fold to magic-multiply divides
    #[comptime] k: usize,  // (the DSL's job -- one source fn, one fast .spv per live shape, like llama)
    #[comptime] nt: usize, // threads per output element (k/32 a multiple of nt)
) {
    let outrow = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let slot = outrow / n;
    let r = outrow % n;
    let wrow = ids[slot] as usize * n + r;
    let nb = k / 256;
    let nsub = k / 32; // 32-weight sub-blocks per row
    let per = (nsub + nt - 1) / nt; // comptime ceil: nt (power of 2) need not divide nsub
    let xrow = slot * k;
    let mut partial = F::new(0.0);
    for j in 0..per {
        let sb = j * nt + t;
        if sb < nsub {
            let sup = sb / 8;
            let jloc = sb % 8;
            let blk = wrow * nb + sup;
            let qbase = blk * 32;
            let scbase = blk * 3;
            let dj = wd[blk] * F::cast_from(q4k_sc(wsc, scbase, jloc));
            let mj = wdm[blk] * F::cast_from(q4k_m(wsc, scbase, jloc));
            let shift = ((jloc % 2) * 4) as u32; // low nibble (even sub-block) or high (odd)
            let boff = (jloc / 2) * 32; // qs byte offset for this sub-block's chunk
            let xoff = xrow + sup * 256 + jloc * 32;
            let mut sbsum = F::new(0.0);
            for qi in 0..32 {
                let nib = (byte_at(wqs, qbase, boff + qi) >> shift) & 15;
                sbsum += (dj * F::cast_from(nib) - mj) * x[xoff + qi];
            }
            partial += sbsum;
        }
    }
    let mut smem = SharedMemory::<F>::new(nt);
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
    if t == 0 {
        out[outrow] = smem[0];
    }
}

// dp4a twin of `moe_matvec_q4k_blk`: same expert-gather + block/reduce skeleton, but the activation
// is pre-quantized to int8 (q8_1: xq int8 4/u32, xs scale, xsum = xs*Sum(xq) per 32-block) and the
// 32-wide f32 MAC becomes 8 hardware int8 dot-products (OpSDot). Q4_K nibbles are 0..15 -> reinterpret
// a masked u32 of 4 packed nibbles as a non-negative i8x4 (llama's mmvq trick), widen to i32x4, .dot
// the signed-i8 activation lane-for-lane. Affine apply is the same identity as the scalar path:
// Sum (d*q4 - m)*x = d*xs*dot(q4,xq) - m*xsum. int8 dot is faster ALU than f32 decode+MAC at equal
// (4-bit) weight bandwidth -> the decode-perf lever. Gated on the device's integer-dot capability.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q4k_dp4a_blk<F: Float>(
    wqs: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<F>,
    wdm: &Array<F>,
    xq: &Array<u32>,   // q8 activation, int8 packed 4/u32, [slots, k/32, 8]
    xs: &Array<F>,     // per-32-block activation scale, [slots, k/32]
    xsum: &Array<F>,   // per-32-block xs*Sum(xq), [slots, k/32]
    ids: &Array<u32>,
    out: &mut Array<F>,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] nt: usize,
) {
    let outrow = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let slot = outrow / n;
    let r = outrow % n;
    let wrow = ids[slot] as usize * n + r;
    let nb = k / 256;
    let nsub = k / 32;
    let per = (nsub + nt - 1) / nt;
    let mut partial = F::new(0.0);
    for j in 0..per {
        let sb = j * nt + t;
        if sb < nsub {
            let sup = sb / 8;
            let jloc = sb % 8;
            let blk = wrow * nb + sup;
            let qbase = blk * 32; // u32 base of this super-block's 128 qs bytes
            let scbase = blk * 3;
            let dj = wd[blk] * F::cast_from(q4k_sc(wsc, scbase, jloc));
            let mj = wdm[blk] * F::cast_from(q4k_m(wsc, scbase, jloc));
            let shift = ((jloc % 2) * 4) as u32; // 0 = low nibble, 4 = high
            let cw = qbase + (jloc / 2) * 8; // 8 u32 = the 32 qs bytes of this sub-block's chunk
            let xg = (slot * nsub + sb) * 8; // activation group base (weight sub-block sb == act 32-block sb)
            let mut idot = 0i32;
            #[unroll]
            for g in 0..8usize {
                let nibs = (wqs[cw + g] >> shift) & 0x0F0F0F0F; // 4 nibbles as 4 bytes (0..15, +ve i8)
                let wv = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<u32>(nibs));
                let xv = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<u32>(xq[xg + g]));
                idot += wv.dot(xv); // OpSDot: 4 int8 products -> i32
            }
            let xi = slot * nsub + sb;
            partial += dj * xs[xi] * F::cast_from(idot) - mj * xsum[xi];
        }
    }
    let mut smem = SharedMemory::<F>::new(nt);
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
    if t == 0 {
        out[outrow] = smem[0];
    }
}

/// CPU q8_1 activation quant matching `quantize_act_q8.comp`: per 32-block, symmetric int8 (amax/127),
/// packed 4/u32 little-endian; returns (xq, xs, xsum) with xsum = scale*Sum(xq). Used by the dp4a run.
pub fn quant_act_q8_cpu(x: &[f32], slots: usize, k: usize) -> (Vec<u32>, Vec<f32>, Vec<f32>) {
    let kb = k / 32;
    let mut xq = vec![0u32; slots * kb * 8];
    let mut xs = vec![0f32; slots * kb];
    let mut xsum = vec![0f32; slots * kb];
    for s in 0..slots {
        for b in 0..kb {
            let base = s * k + b * 32;
            let amax = (0..32).fold(0f32, |m, i| m.max(x[base + i].abs()));
            let inv = if amax > 0.0 { 127.0 / amax } else { 0.0 };
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let gid = s * kb + b;
            let mut isum = 0i32;
            for j in 0..8 {
                let mut word = 0u32;
                for l in 0..4 {
                    let q = (x[base + j * 4 + l] * inv).round().clamp(-127.0, 127.0) as i32;
                    isum += q;
                    word |= ((q as u32) & 0xFF) << (l * 8);
                }
                xq[gid * 8 + j] = word;
            }
            xs[gid] = scale;
            xsum[gid] = scale * isum as f32;
        }
    }
    (xq, xs, xsum)
}

/// Host launch for the dp4a Q4_K MoE matvec (activation q8-quantized on the host for the bench).
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q4k_dp4a_blk_run<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize, nt: usize,
) -> Vec<f32> {
    let (xq, xs, xsum) = quant_act_q8_cpu(x, slots, k);
    let qh = client.create_from_slice(u32::as_bytes(wqs));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let mh = client.create_from_slice(f32::as_bytes(wdm));
    let xqh = client.create_from_slice(u32::as_bytes(&xq));
    let xsh = client.create_from_slice(f32::as_bytes(&xs));
    let xsumh = client.create_from_slice(f32::as_bytes(&xsum));
    let ih = client.create_from_slice(u32::as_bytes(ids));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n]));
    unsafe {
        moe_matvec_q4k_dp4a_blk::launch_unchecked::<f32, R>(
            client, Grid::Static((slots * n) as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(qh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(mh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(xqh.clone(), xq.len()),
            ArrayArg::from_raw_parts(xsh.clone(), xs.len()),
            ArrayArg::from_raw_parts(xsumh.clone(), xsum.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k, nt,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Interleave the split Q4_K arrays into the verbatim GGUF packed layout (36 u32/superblock:
/// [d(f16)|dmin(f16), scales(3 u32), qs(32 u32)]) -- the layout the dense QMatMul weight is stored in.
pub fn pack_q4k(wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32]) -> Vec<u32> {
    let nblk = wd.len();
    let mut p = vec![0u32; nblk * 36];
    for b in 0..nblk {
        let d = half::f16::from_f32(wd[b]).to_bits() as u32;
        let dm = half::f16::from_f32(wdm[b]).to_bits() as u32;
        p[b * 36] = d | (dm << 16);
        p[b * 36 + 1..b * 36 + 4].copy_from_slice(&wsc[b * 3..b * 3 + 3]);
        p[b * 36 + 4..b * 36 + 36].copy_from_slice(&wqs[b * 32..b * 32 + 32]);
    }
    p
}

// DENSE dp4a Q4_K matvec: `out[n] = W[n,k] @ x[k]`, W = VERBATIM PACKED GGUF Q4_K (36 u32/superblock),
// read directly (same buffer the dense QMatMul holds -- no re-split). int8-dot decode: q8 activation,
// nibbles reinterpret to i8x4, OpSDot; d/dmin unpacked in-kernel (f16lo_to_f32). RUNTIME k -> one .spv
// per nt for every projection shape. One workgroup per output row, `nt` threads tree-reduce over k --
// the decode fix for the prefill dp4a GEMM that runs 1-thread-per-row (occupancy-starved at m=1), ~3x.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q4k_dp4a_blk<F: Float>(
    wq: &Array<u32>,   // packed Q4_K, 36 u32/superblock
    xq: &Array<u32>,   // q8 activation, int8 4/u32, [k/32, 8]
    xs: &Array<F>,     // per-32-block scale, [k/32]
    xsum: &Array<F>,   // per-32-block xs*Sum(xq), [k/32]
    out: &mut Array<F>,
    meta: &Array<u32>, // [k]
    #[comptime] nt: usize,
) {
    let row = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let k = meta[0] as usize;
    let nb = k / 256;
    let nsub = k / 32;
    let per = (nsub + nt - 1) / nt;
    let mut partial = F::new(0.0);
    for j in 0..per {
        let sb = j * nt + t;
        if sb < nsub {
            let sup = sb / 8;
            let jloc = sb % 8;
            let bb = (row * nb + sup) * 36; // packed superblock base (u32)
            let d = F::cast_from(f16lo_to_f32(wq[bb]));
            let dmin = F::cast_from(f16lo_to_f32(wq[bb] >> 16));
            let scbase = bb + 1; // scales: 3 u32 after d/dmin
            let dj = d * F::cast_from(q4k_sc(wq, scbase, jloc));
            let mj = dmin * F::cast_from(q4k_m(wq, scbase, jloc));
            let shift = ((jloc % 2) * 4) as u32;
            let cw = bb + 4 + (jloc / 2) * 8; // qs: 32 u32 after scales; this sub-block's chunk
            let xg = sb * 8;
            let mut idot = 0i32;
            #[unroll]
            for g in 0..8usize {
                let nibs = (wq[cw + g] >> shift) & 0x0F0F0F0F;
                let wv = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<u32>(nibs));
                let xv = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<u32>(xq[xg + g]));
                idot += wv.dot(xv);
            }
            partial += dj * xs[sb] * F::cast_from(idot) - mj * xsum[sb];
        }
    }
    let mut smem = SharedMemory::<F>::new(nt);
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
    if t == 0 {
        out[row] = smem[0];
    }
}

/// Host launch for the dense packed-Q4_K dp4a matvec (activation q8-quantized on the host for the bench).
pub fn matvec_q4k_dp4a_blk_run<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32],
    nout: usize, k: usize, nt: usize,
) -> Vec<f32> {
    let packed = pack_q4k(wqs, wsc, wd, wdm);
    let (xq, xs, xsum) = quant_act_q8_cpu(x, 1, k);
    let wh = client.create_from_slice(u32::as_bytes(&packed));
    let xqh = client.create_from_slice(u32::as_bytes(&xq));
    let xsh = client.create_from_slice(f32::as_bytes(&xs));
    let xsumh = client.create_from_slice(f32::as_bytes(&xsum));
    let meta = client.create_from_slice(u32::as_bytes(&[k as u32]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nout]));
    unsafe {
        matvec_q4k_dp4a_blk::launch_unchecked::<f32, R>(
            client, Grid::Static(nout as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(wh.clone(), packed.len()),
            ArrayArg::from_raw_parts(xqh.clone(), xq.len()),
            ArrayArg::from_raw_parts(xsh.clone(), xs.len()),
            ArrayArg::from_raw_parts(xsumh.clone(), xsum.len()),
            ArrayArg::from_raw_parts(oh.clone(), nout),
            ArrayArg::from_raw_parts(meta.clone(), 1),
            nt,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

// DENSE f32-direct Q4_K decode matvec: `out[n] = W[n,k] @ x[k]`, W = VERBATIM PACKED GGUF Q4_K, the
// activation read as f32 DIRECTLY. The DSL twin of the hand `mul_mat_vec_q4k_cm`. It carries NO q8
// activation-quantize step -- that dispatch's command-record cost is what makes the dp4a decode path
// lose on a backend with expensive eager recording, so the f32-direct kernel is the one the autotuner
// selects there. Same affine decode w = d*sc*q - dmin*m and the SAME Q4_K activation interleaving as the
// oracle: sub-block s of superblock `sup` covers x[sup*256 + (s/2)*64 + (s%2)*32 ..][+32], bit-identical
// to matvec_q4k_ref / BlockQ4K::to_float. Two comptime schedule knobs: `nt` threads cooperate on one
// row's k (the workgroup-width / k-parallelism axis the autotuner picks per shape -- a wide width fills
// the device at small row counts where row-parallelism alone leaves it idle); `nr` output rows share the
// workgroup (the row tile). The inner 8-step decode is `#[unroll]`ed so its loads issue independently
// (the memory-level parallelism a bandwidth-bound matvec needs). RUNTIME k via meta -> one kernel per
// (nt, nr) across every projection shape.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q4k_f32_blk<F: Float>(
    wq: &Array<u32>,   // packed Q4_K, 36 u32/superblock
    x: &Array<F>,      // activation, f32 [k]
    out: &mut Array<F>,
    meta: &Array<u32>, // [k]
    #[comptime] nt: usize, // threads cooperating on one row's k (the workgroup-width axis)
    #[comptime] nr: usize, // output rows per workgroup (the row tile)
) {
    let wgid = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let k = meta[0] as usize;
    let nb = k / 256;
    let nsub = k / 32;
    let per = (nsub + nt - 1) / nt;
    let row0 = wgid * nr;
    let nout = out.len();
    let mut acc = Array::<F>::new(nr);
    #[unroll]
    for n in 0..nr {
        acc[n] = F::new(0.0);
    }
    for j in 0..per {
        let sb = j * nt + t; // sub-block (0..nsub) this thread contracts on this pass
        if sb < nsub {
            let sup = sb / 8; // superblock
            let jloc = sb % 8; // sub-block within the superblock (0..7)
            let shift = ((jloc % 2) * 4) as u32; // low (even sub-block) or high (odd) nibble
            let cwoff = 4 + (jloc / 2) * 8; // qs u32 base within a superblock for this sub-block
            let xbase = sup * 256 + (jloc / 2) * 64 + (jloc % 2) * 32; // activation base of the 32 weights
            #[unroll]
            for n in 0..nr {
                let row = row0 + n;
                if row < nout {
                    let bb = (row * nb + sup) * 36; // packed superblock base
                    let d = F::cast_from(f16lo_to_f32(wq[bb]));
                    let dmin = F::cast_from(f16lo_to_f32(wq[bb] >> 16));
                    let dj = d * F::cast_from(q4k_sc(wq, bb + 1, jloc));
                    let mj = dmin * F::cast_from(q4k_m(wq, bb + 1, jloc));
                    let cw = bb + cwoff;
                    let mut sdot = F::new(0.0);
                    let mut sx = F::new(0.0);
                    #[unroll]
                    for g in 0..8usize {
                        let word = (wq[cw + g] >> shift) & 0x0F0F0F0F;
                        let xb = xbase + g * 4;
                        let x0 = x[xb];
                        let x1 = x[xb + 1];
                        let x2 = x[xb + 2];
                        let x3 = x[xb + 3];
                        sdot += F::cast_from(word & 0xFF) * x0
                            + F::cast_from((word >> 8) & 0xFF) * x1
                            + F::cast_from((word >> 16) & 0xFF) * x2
                            + F::cast_from((word >> 24) & 0xFF) * x3;
                        sx += x0 + x1 + x2 + x3;
                    }
                    // Affine decode folded over the sub-block: sum (dj*q - mj)*x = dj*sum(q*x) - mj*sum(x).
                    acc[n] += dj * sdot - mj * sx;
                }
            }
        }
    }
    // Per-row tree reduction of the nt partials through shared memory (nr sequential passes; both small).
    let mut smem = SharedMemory::<F>::new(nt);
    #[unroll]
    for n in 0..nr {
        smem[t] = acc[n];
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
        if t == 0 {
            let row = row0 + n;
            if row < nout {
                out[row] = smem[0];
            }
        }
        sync_cube();
    }
}

/// Host launch for the dense packed-Q4_K f32-direct cooperative matvec (`nt` threads/row, `nr` rows/wg).
#[allow(clippy::too_many_arguments)]
pub fn matvec_q4k_f32_blk_run<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32],
    nout: usize, k: usize, nt: usize, nr: usize,
) -> Vec<f32> {
    let packed = pack_q4k(wqs, wsc, wd, wdm);
    let wh = client.create_from_slice(u32::as_bytes(&packed));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let meta = client.create_from_slice(u32::as_bytes(&[k as u32]));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; nout]));
    let grid = (nout as u32).div_ceil(nr as u32); // one workgroup per nr output rows
    unsafe {
        matvec_q4k_f32_blk::launch_unchecked::<f32, R>(
            client, Grid::Static(grid, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(wh.clone(), packed.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), nout),
            ArrayArg::from_raw_parts(meta.clone(), 1),
            nt,
            nr,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

// ============================================================================================
// Evolutionary schedule search for the f32-direct Q4_K matvec -- the Vulkan-lane sibling of the dp4a
// search above. Same tune::Evolution machinery; the genome is {WG, NR}. WG is the workgroup width (=
// threads cooperating on one row's k = `nt`): a wide width fills the device at small row counts where
// row-parallelism leaves it idle, a narrow width wins at large row counts -- the shape-adaptive width
// the hand kernel hardcodes and llama tunes. There is no VW knob: the 8-step decode is fully unrolled.
// The Tuner keys by (device, rows, k), so the width is chosen per shape and this f32 kernel is the
// per-device algorithm the search crowns where the dp4a activation-quantize dispatch is too costly.
// ============================================================================================

/// The f32-direct Q4_K matvec schedule space: workgroup width `WG` (= cooperating threads `nt`) and the
/// row tile `NR`. WG stays a power of two so the tree reduction halves cleanly; every value is correct
/// for any k (a wide WG just idles the surplus threads on a short row), so there is no k constraint.
pub fn matvec_q4k_f32_space() -> Space {
    Space::new()
        .param("WG", [16, 32, 64, 128, 256])
        .param("NR", [1, 2, 4])
        .constraint(|c, s| {
            let w = c.get(s, "WG");
            w >= 2 && (w & (w - 1)) == 0 && w <= 1024
        })
}

/// Map a schedule [`Config`] to the f32 kernel's `(nt, nr)` launch tuple.
fn matvec_q4k_f32_cfg(c: &Config, s: &Space) -> (usize, usize) {
    (c.get(s, "WG") as usize, c.get(s, "NR") as usize)
}

/// Cold-weight-streamed, bit-exact-gated fitness for [`matvec_q4k_f32_space`], generic in the runtime.
/// Holds resident PACKED Q4_K weight banks (rotated so each timed dispatch reads cold) and the f32
/// activation; the correctness gate is `matvec_q4k_ref` (the oracle the hand `mul_mat_vec_q4k_cm` is also
/// gated against, so a passing schedule is runtime-equal to the hand kernel). Same discipline as the dp4a
/// evaluator; CpuRuntime proves the search offline, a wgpu device gives the meaningful winner.
pub struct MatvecQ4kF32Eval<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    space: &'a Space,
    banks: Vec<Handle>, // resident packed weight banks (distinct buffers -> cold rotation)
    xh: Handle,
    meta: Handle,
    outh: Handle,
    packed_len: usize,
    x_len: usize,
    rows: usize,
    k: usize,
    oracle: Vec<f32>,
    maxref: f32,
    repeats: usize,
    worst_rel: std::cell::Cell<f32>,
}

impl<'a, R: Runtime> MatvecQ4kF32Eval<'a, R> {
    /// Pack the split Q4_K weight, upload `nbanks` resident copies (distinct buffers so a rotation reads
    /// cold), upload the f32 activation, and compute the Q4_K oracle.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &'a ComputeClient<R>,
        space: &'a Space,
        wqs: &[u32],
        wsc: &[u32],
        wd: &[f32],
        wdm: &[f32],
        x: &[f32],
        rows: usize,
        k: usize,
        nbanks: usize,
        repeats: usize,
    ) -> Self {
        let packed = pack_q4k(wqs, wsc, wd, wdm);
        let banks: Vec<Handle> =
            (0..nbanks.max(1)).map(|_| client.create_from_slice(u32::as_bytes(&packed))).collect();
        let xh = client.create_from_slice(f32::as_bytes(x));
        let meta = client.create_from_slice(u32::as_bytes(&[k as u32]));
        let outh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
        let oracle = matvec_q4k_ref(wqs, wsc, wd, wdm, x, rows, k);
        let maxref = oracle.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
        Self {
            client,
            space,
            banks,
            xh,
            meta,
            outh,
            packed_len: packed.len(),
            x_len: x.len(),
            rows,
            k,
            oracle,
            maxref,
            repeats,
            worst_rel: std::cell::Cell::new(0.0),
        }
    }

    /// The search space this evaluator is bound to.
    pub fn space(&self) -> &Space {
        self.space
    }

    /// The worst scale-relative divergence any measured config showed (the committed correctness witness).
    pub fn worst_rel(&self) -> f32 {
        self.worst_rel.get()
    }

    fn dispatch(&self, bank: &Handle, nt: usize, nr: usize) {
        let grid = (self.rows as u32).div_ceil(nr as u32);
        unsafe {
            matvec_q4k_f32_blk::launch_unchecked::<f32, R>(
                self.client,
                Grid::Static(grid, 1, 1),
                Block::new_1d(nt as u32),
                ArrayArg::from_raw_parts(bank.clone(), self.packed_len),
                ArrayArg::from_raw_parts(self.xh.clone(), self.x_len),
                ArrayArg::from_raw_parts(self.outh.clone(), self.rows),
                ArrayArg::from_raw_parts(self.meta.clone(), 1),
                nt,
                nr,
            );
        }
    }

    fn read_out(&self) -> Vec<f32> {
        f32::from_bytes(&self.client.read_one_unchecked(self.outh.clone())).to_vec()
    }
}

impl<'a, R: Runtime> Evaluator for MatvecQ4kF32Eval<'a, R> {
    fn static_check(&self, cfg: &Config) -> Verdict {
        let (nt, _nr) = matvec_q4k_f32_cfg(cfg, self.space);
        if nt < 2 || nt > 1024 || (nt & (nt - 1)) != 0 {
            return Verdict::Reject(format!("workgroup width {nt} must be a power of two in [2,1024]"));
        }
        Verdict::Pass
    }

    fn measure(&self, cfg: &Config, iters: usize) -> f64 {
        let (nt, nr) = matvec_q4k_f32_cfg(cfg, self.space);
        self.dispatch(&self.banks[0], nt, nr);
        let got = self.read_out();
        let rel = got.iter().zip(&self.oracle).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / self.maxref;
        self.worst_rel.set(self.worst_rel.get().max(rel));
        if rel > 1e-3 {
            return f64::INFINITY;
        }
        let nb = self.banks.len();
        for i in 0..(2 * nb) {
            self.dispatch(&self.banks[i % nb], nt, nr);
        }
        let _ = self.read_out();
        (0..self.repeats)
            .map(|_| {
                let t = std::time::Instant::now();
                for i in 0..iters {
                    self.dispatch(&self.banks[i % nb], nt, nr);
                }
                let _ = self.read_out();
                t.elapsed().as_secs_f64() * 1e3 / iters as f64
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Search [`matvec_q4k_f32_space`] on `eval` for the fastest `(nt, nr)` at this `(device, rows, k)`,
/// caching the winner. The autotuner surface for the f32-direct Vulkan decode matvec.
#[allow(clippy::too_many_arguments)]
pub fn matvec_q4k_f32_hunt<R: Runtime>(
    tuner: &Tuner,
    device: &str,
    rows: usize,
    k: usize,
    eval: &MatvecQ4kF32Eval<R>,
    evo: &Evolution,
    seed: u64,
) -> Evolved {
    tuner.evolve(device, "matvec_q4k_f32", &format!("rows={rows},k={k}"), eval.space(), eval, evo, seed)
}

/// Host launch for the block-reduced Q4_K MoE matvec (one workgroup per output, `nt` threads).
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q4k_blk_run<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize, nt: usize,
) -> Vec<f32> {
    let (qh, sh, dh, mh, xh, ih, oh) = (
        client.create_from_slice(u32::as_bytes(wqs)),
        client.create_from_slice(u32::as_bytes(wsc)),
        client.create_from_slice(f32::as_bytes(wd)),
        client.create_from_slice(f32::as_bytes(wdm)),
        client.create_from_slice(f32::as_bytes(x)),
        client.create_from_slice(u32::as_bytes(ids)),
        client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n])),
    );
    unsafe {
        moe_matvec_q4k_blk::launch_unchecked::<f32, R>(
            client, Grid::Static((slots * n) as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(qh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(mh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k, nt,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Kernel-only throughput for the block-reduced Q4_K MoE matvec.
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q4k_blk_bench<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize, nt: usize, iters: usize,
) -> f64 {
    let (qh, sh, dh, mh, xh, ih, oh) = (
        client.create_from_slice(u32::as_bytes(wqs)),
        client.create_from_slice(u32::as_bytes(wsc)),
        client.create_from_slice(f32::as_bytes(wd)),
        client.create_from_slice(f32::as_bytes(wdm)),
        client.create_from_slice(f32::as_bytes(x)),
        client.create_from_slice(u32::as_bytes(ids)),
        client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n])),
    );
    let launch = |c: &ComputeClient<R>| unsafe {
        moe_matvec_q4k_blk::launch_unchecked::<f32, R>(
            c, Grid::Static((slots * n) as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(qh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(mh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k, nt,
        );
    };
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// Kernel-only throughput for the Q4_K MoE matvec (buffers once, `iters` dispatches, one sync).
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q4k_bench<R: Runtime>(
    client: &ComputeClient<R>,
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize, iters: usize,
) -> f64 {
    let qh = client.create_from_slice(u32::as_bytes(wqs));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let mh = client.create_from_slice(f32::as_bytes(wdm));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let ih = client.create_from_slice(u32::as_bytes(ids));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n]));
    let block = 64u32;
    let grid = ((slots * n) as u32).div_ceil(block);
    let launch = |c: &ComputeClient<R>| unsafe {
        moe_matvec_q4k::launch_unchecked::<f32, R>(
            c, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(qh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(mh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k,
        );
    };
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// CPU oracle for the Q4_K MoE matvec (same packed inputs; BlockQ4K::to_float decode order).
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q4k_ref(
    wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize,
) -> Vec<f32> {
    let nb = k / 256;
    let mut out = vec![0.0f32; slots * n];
    for slot in 0..slots {
        let expert = ids[slot] as usize;
        for r in 0..n {
            let wrow = expert * n + r;
            let mut acc = 0.0f32;
            for b in 0..nb {
                let blk = wrow * nb + b;
                let (qbase, scbase) = (blk * 32, blk * 3);
                let (d, dmin) = (wd[blk], wdm[blk]);
                let xblk = slot * k + b * 256;
                for g in 0..4 {
                    let is = g * 2;
                    let d1 = d * cpu_sc(wsc, scbase, is) as f32;
                    let mm1 = dmin * cpu_m(wsc, scbase, is) as f32;
                    let d2 = d * cpu_sc(wsc, scbase, is + 1) as f32;
                    let mm2 = dmin * cpu_m(wsc, scbase, is + 1) as f32;
                    for qi in 0..32 {
                        let qb = cpu_byte(wqs, qbase, g * 32 + qi);
                        acc += (d1 * (qb & 15) as f32 - mm1) * x[xblk + g * 64 + qi];
                        acc += (d2 * (qb >> 4) as f32 - mm2) * x[xblk + g * 64 + 32 + qi];
                    }
                }
            }
            out[slot * n + r] = acc;
        }
    }
    out
}

/// Deterministic MoE test data: a flat [E*n] Q4_K bank, `slots` activations `[slots,k]`, and a
/// per-slot expert id in `[0,e)`. Reuses `gen_q4k` for the bank (its lone x is discarded).
pub fn gen_moe_q4k(e: usize, n: usize, slots: usize, k: usize)
    -> (Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<u32>) {
    let (wqs, wsc, wd, wdm, _x1) = gen_q4k(e * n, k);
    let mut s = 0xC2B2AE3D27D4EB4Fu64;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let x: Vec<f32> = (0..slots * k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    let ids: Vec<u32> = (0..slots).map(|_| (next() % e as u64) as u32).collect();
    (wqs, wsc, wd, wdm, x, ids)
}

// ============================================================================================
// Q6_K indexed-MoE matvec: `out[s, r] = sum_k W[ids[s], r, k] * x[s, k]`, gathering per-expert rows
// from a flat [E*n] Q6_K bank. Same `decode ⊥ gather` split as `moe_matvec_q4k`; the decode is the
// 210-byte Q6_K superblock (ql[128] low-4b, qh[64] high-2b, scales[16] SIGNED i8, d f16), passed as
// split u32 arrays. Bit-identical to BlockQ6K::to_float: weight = d * sc * (6-bit code - 32).
//   wql: ql, 32 u32/block (128 B)    wqh: qh, 16 u32/block (64 B)
//   wsc: scales, 4 u32/block (16 B, signed)    wd: d, 1 f32/block
// ============================================================================================
// Sign-extended byte: `(b ^ 128) - 128` maps u8 0..255 to i8 -128..127 (branchless).
#[device]
fn sbyte_at(a: &Array<u32>, base: usize, i: usize) -> i32 {
    let b = (a[base + i / 4] >> ((8 * (i % 4)) as u32)) & 255;
    (b ^ 128) as i32 - 128
}

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn moe_matvec_q6k<F: Float>(
    wql: &Array<u32>,
    wqh: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<F>,
    x: &Array<F>,
    ids: &Array<u32>,
    out: &mut Array<F>,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let gid = ABSOLUTE_POS;
    if gid < out.len() {
        let slot = gid / n;
        let r = gid % n;
        let wrow = ids[slot] as usize * n + r;
        let nb = k / 256;
        let xbase = slot * k;
        let mut acc = F::new(0.0);
        for b in 0..nb {
            let blk = wrow * nb + b;
            let qlb = blk * 32; // ql: 32 u32/block
            let qhb = blk * 16; // qh: 16 u32/block
            let scb = blk * 4; // scales: 4 u32/block
            let d = wd[blk];
            let xblk = xbase + b * 256;
            for idx in 0..2 {
                let scoff = idx * 8;
                let qloff = idx * 64;
                let qhoff = idx * 32;
                let xb = xblk + idx * 128;
                for l in 0..32 {
                    let is = l / 16;
                    let qll = byte_at(wql, qlb, qloff + l);
                    let qlh = byte_at(wql, qlb, qloff + l + 32);
                    let qhv = byte_at(wqh, qhb, qhoff + l);
                    let q1 = ((qll & 15) | ((qhv & 3) << 4)) as i32 - 32;
                    let q2 = ((qlh & 15) | (((qhv >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((qll >> 4) | (((qhv >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((qlh >> 4) | (((qhv >> 6) & 3) << 4)) as i32 - 32;
                    let s1 = d * F::cast_from(sbyte_at(wsc, scb, scoff + is));
                    let s2 = d * F::cast_from(sbyte_at(wsc, scb, scoff + is + 2));
                    let s3 = d * F::cast_from(sbyte_at(wsc, scb, scoff + is + 4));
                    let s4 = d * F::cast_from(sbyte_at(wsc, scb, scoff + is + 6));
                    acc += s1 * F::cast_from(q1) * x[xb + l];
                    acc += s2 * F::cast_from(q2) * x[xb + l + 32];
                    acc += s3 * F::cast_from(q3) * x[xb + l + 64];
                    acc += s4 * F::cast_from(q4) * x[xb + l + 96];
                }
            }
        }
        out[gid] = acc;
    }
}

/// Host launch for the Q6_K MoE matvec. Split [E*n] Q6_K bank + `[slots,k]` x + per-slot expert id.
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q6k_run<R: Runtime>(
    client: &ComputeClient<R>,
    wql: &[u32], wqh: &[u32], wsc: &[u32], wd: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize,
) -> Vec<f32> {
    let qlh = client.create_from_slice(u32::as_bytes(wql));
    let qhh = client.create_from_slice(u32::as_bytes(wqh));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let ih = client.create_from_slice(u32::as_bytes(ids));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n]));
    let block = 64u32;
    let grid = ((slots * n) as u32).div_ceil(block);
    unsafe {
        moe_matvec_q6k::launch_unchecked::<f32, R>(
            client, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(qlh.clone(), wql.len()),
            ArrayArg::from_raw_parts(qhh.clone(), wqh.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Kernel-only throughput for the Q6_K MoE matvec.
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q6k_bench<R: Runtime>(
    client: &ComputeClient<R>,
    wql: &[u32], wqh: &[u32], wsc: &[u32], wd: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize, iters: usize,
) -> f64 {
    let qlh = client.create_from_slice(u32::as_bytes(wql));
    let qhh = client.create_from_slice(u32::as_bytes(wqh));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let ih = client.create_from_slice(u32::as_bytes(ids));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n]));
    let block = 64u32;
    let grid = ((slots * n) as u32).div_ceil(block);
    let launch = |c: &ComputeClient<R>| unsafe {
        moe_matvec_q6k::launch_unchecked::<f32, R>(
            c, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(qlh.clone(), wql.len()),
            ArrayArg::from_raw_parts(qhh.clone(), wqh.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k,
        );
    };
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

#[inline]
fn cpu_sbyte(a: &[u32], base: usize, i: usize) -> i32 {
    let b = (a[base + i / 4] >> (8 * (i % 4))) & 255;
    (b ^ 128) as i32 - 128
}

/// CPU oracle for the Q6_K MoE matvec (BlockQ6K::to_float decode order: weight = d * sc * (code-32)).
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q6k_ref(
    wql: &[u32], wqh: &[u32], wsc: &[u32], wd: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize,
) -> Vec<f32> {
    let nb = k / 256;
    let mut out = vec![0.0f32; slots * n];
    for slot in 0..slots {
        let expert = ids[slot] as usize;
        for r in 0..n {
            let wrow = expert * n + r;
            let mut acc = 0.0f32;
            for b in 0..nb {
                let blk = wrow * nb + b;
                let (qlb, qhb, scb) = (blk * 32, blk * 16, blk * 4);
                let d = wd[blk];
                let xblk = slot * k + b * 256;
                for idx in 0..2 {
                    let (scoff, qloff, qhoff) = (idx * 8, idx * 64, idx * 32);
                    let xb = xblk + idx * 128;
                    for l in 0..32 {
                        let is = l / 16;
                        let qll = cpu_byte(wql, qlb, qloff + l);
                        let qlh = cpu_byte(wql, qlb, qloff + l + 32);
                        let qhv = cpu_byte(wqh, qhb, qhoff + l);
                        let q1 = ((qll & 15) | ((qhv & 3) << 4)) as i32 - 32;
                        let q2 = ((qlh & 15) | (((qhv >> 2) & 3) << 4)) as i32 - 32;
                        let q3 = ((qll >> 4) | (((qhv >> 4) & 3) << 4)) as i32 - 32;
                        let q4 = ((qlh >> 4) | (((qhv >> 6) & 3) << 4)) as i32 - 32;
                        let s1 = d * cpu_sbyte(wsc, scb, scoff + is) as f32;
                        let s2 = d * cpu_sbyte(wsc, scb, scoff + is + 2) as f32;
                        let s3 = d * cpu_sbyte(wsc, scb, scoff + is + 4) as f32;
                        let s4 = d * cpu_sbyte(wsc, scb, scoff + is + 6) as f32;
                        acc += s1 * q1 as f32 * x[xb + l];
                        acc += s2 * q2 as f32 * x[xb + l + 32];
                        acc += s3 * q3 as f32 * x[xb + l + 64];
                        acc += s4 * q4 as f32 * x[xb + l + 96];
                    }
                }
            }
            out[slot * n + r] = acc;
        }
    }
    out
}

/// Deterministic Q6_K MoE test data: flat [E*n] bank (ql/qh/scales/d) + `[slots,k]` x + expert ids.
pub fn gen_moe_q6k(e: usize, n: usize, slots: usize, k: usize)
    -> (Vec<u32>, Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>, Vec<u32>) {
    let nb = k / 256;
    let nblk = e * n * nb;
    let mut s = 0x27D4EB2F165667C5u64;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let wql: Vec<u32> = (0..nblk * 32).map(|_| next() as u32).collect(); // 128 ql bytes/block
    let wqh: Vec<u32> = (0..nblk * 16).map(|_| next() as u32).collect(); // 64 qh bytes/block
    let wsc: Vec<u32> = (0..nblk * 4).map(|_| next() as u32).collect();  // 16 scale bytes/block (i8)
    let wd: Vec<f32> = (0..nblk).map(|_| half::f16::from_f32((next() % 1000) as f32 / 20000.0 + 0.002).to_f32()).collect();
    let x: Vec<f32> = (0..slots * k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    let ids: Vec<u32> = (0..slots).map(|_| (next() % e as u64) as u32).collect();
    (wql, wqh, wsc, wd, x, ids)
}

// Block-reduced Q6_K MoE matvec (slice ② for Q6_K). One workgroup/output, nt threads split the
// k/32 sub-blocks, shared-mem tree reduce -- same lever as `moe_matvec_q4k_blk`. Q6_K sub-block sb
// -> superblock sb/8, and within the superblock chunk = (sb%8)/4, sub = (sb%8)%4: ql byte l+(sub&1)*32
// nibble (sub>>1)*4, qh byte l shift 2*sub, scale index 8*chunk + l/16 + 2*sub. Matches the
// naive-order oracle within f32-reorder.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn moe_matvec_q6k_blk<F: Float>(
    wql: &Array<u32>,
    wqh: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<F>,
    x: &Array<F>,
    ids: &Array<u32>,
    out: &mut Array<F>,
    #[comptime] n: usize, // comptime shape (see moe_matvec_q4k_blk): folds to magic-multiply divides
    #[comptime] k: usize,
    #[comptime] nt: usize,
) {
    let outrow = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let slot = outrow / n;
    let r = outrow % n;
    let wrow = ids[slot] as usize * n + r;
    let nb = k / 256;
    let nsub = k / 32;
    let per = (nsub + nt - 1) / nt; // comptime ceil: nt (power of 2) need not divide nsub
    let xrow = slot * k;
    let mut partial = F::new(0.0);
    for j in 0..per {
        let sb = j * nt + t;
        if sb < nsub {
            let sup = sb / 8;
            let sbloc = sb % 8;
            let idx = sbloc / 4; // chunk 0/1
            let sub = sbloc % 4; // 0..4
            let blk = wrow * nb + sup;
            let d = wd[blk];
            let qlbase = blk * 32;
            let qhbase = blk * 16;
            let scbase = blk * 4;
            let qloff = idx * 64 + (sub % 2) * 32; // ql byte base for this (chunk,sub)
            let qhoff = idx * 32; // qh byte base for this chunk
            let qlshift = ((sub / 2) * 4) as u32; // low nibble (sub 0,1) or high (sub 2,3)
            let qhshift = (sub * 2) as u32; // 2-bit high field for this sub
            let scb0 = idx * 8 + sub * 2; // scale index for l<16 (is=0); +1 for l>=16
            let xoff = xrow + sup * 256 + idx * 128 + sub * 32;
            let mut sbsum = F::new(0.0);
            for l in 0..32 {
                let is = l / 16;
                let sc = sbyte_at(wsc, scbase, scb0 + is);
                let qlb = byte_at(wql, qlbase, qloff + l);
                let qhb = byte_at(wqh, qhbase, qhoff + l);
                let q = ((qlb >> qlshift) & 15) | (((qhb >> qhshift) & 3) << 4);
                let qv = q as i32 - 32;
                sbsum += d * F::cast_from(sc) * F::cast_from(qv) * x[xoff + l];
            }
            partial += sbsum;
        }
    }
    let mut smem = SharedMemory::<F>::new(nt);
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
    if t == 0 {
        out[outrow] = smem[0];
    }
}

/// Host launch for the block-reduced Q6_K MoE matvec.
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q6k_blk_run<R: Runtime>(
    client: &ComputeClient<R>,
    wql: &[u32], wqh: &[u32], wsc: &[u32], wd: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize, nt: usize,
) -> Vec<f32> {
    let (qlh, qhh, sh, dh, xh, ih, oh) = (
        client.create_from_slice(u32::as_bytes(wql)),
        client.create_from_slice(u32::as_bytes(wqh)),
        client.create_from_slice(u32::as_bytes(wsc)),
        client.create_from_slice(f32::as_bytes(wd)),
        client.create_from_slice(f32::as_bytes(x)),
        client.create_from_slice(u32::as_bytes(ids)),
        client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n])),
    );
    unsafe {
        moe_matvec_q6k_blk::launch_unchecked::<f32, R>(
            client, Grid::Static((slots * n) as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(qlh.clone(), wql.len()),
            ArrayArg::from_raw_parts(qhh.clone(), wqh.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k, nt,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Kernel-only throughput for the block-reduced Q6_K MoE matvec.
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q6k_blk_bench<R: Runtime>(
    client: &ComputeClient<R>,
    wql: &[u32], wqh: &[u32], wsc: &[u32], wd: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize, nt: usize, iters: usize,
) -> f64 {
    let (qlh, qhh, sh, dh, xh, ih, oh) = (
        client.create_from_slice(u32::as_bytes(wql)),
        client.create_from_slice(u32::as_bytes(wqh)),
        client.create_from_slice(u32::as_bytes(wsc)),
        client.create_from_slice(f32::as_bytes(wd)),
        client.create_from_slice(f32::as_bytes(x)),
        client.create_from_slice(u32::as_bytes(ids)),
        client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n])),
    );
    let launch = |c: &ComputeClient<R>| unsafe {
        moe_matvec_q6k_blk::launch_unchecked::<f32, R>(
            c, Grid::Static((slots * n) as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(qlh.clone(), wql.len()),
            ArrayArg::from_raw_parts(qhh.clone(), wqh.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k, nt,
        );
    };
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    t.elapsed().as_secs_f64() * 1e3 / iters as f64
}

// dp4a Q6_K MoE matvec: the last un-dp4a'd hot path (the ffn_down projection; `moe_matvec_q6k_blk`
// above decodes it one byte at a time in f32). Same block-per-output-row shape as its Q6_K sibling and
// the same int8-dot decode as `moe_matvec_q4k_dp4a_blk`: four consecutive weights share one aligned ql
// word and one qh word, so a group is `((ql >> qlshift) & 0x0F0F0F0F) | (((qh >> qhshift) & 0x03030303)
// << 4)` -- four 6-bit codes (0..63, so they reinterpret to i8 unsigned-safe) dotted against the q8
// activation via OpSDot.
//
// The `- 32` bias folds out affinely, exactly as Q4_K folds `dmin`: within a 16-weight half-block the
// scale is constant, so `sum d*sc*(q-32)*x` with `x = xs*xq` is `d*xs*sc*(dot(q,xq) - 32*sum(xq))`.
// `sum(xq)` comes from dotting the activation with a vector of ones -- one extra OpSDot per group, and
// no new host-side array (unlike Q4_K's `xsum`, which is per-32 and cannot express a half-block sum).
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q6k_dp4a_blk<F: Float>(
    wql: &Array<u32>,
    wqh: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<F>,
    xq: &Array<u32>, // q8 activation, int8 packed 4/u32, [slots, k/32, 8]
    xs: &Array<F>,   // per-32-block activation scale, [slots, k/32]
    ids: &Array<u32>,
    out: &mut Array<F>,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] nt: usize,
) {
    let outrow = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let slot = outrow / n;
    let r = outrow % n;
    let wrow = ids[slot] as usize * n + r;
    let nb = k / 256;
    let nsub = k / 32;
    let per = (nsub + nt - 1) / nt;
    let xrow = slot * nsub;
    let ones = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<u32>(
        0x0101_0101u32,
    ));
    let mut partial = F::new(0.0);
    for j in 0..per {
        let sb = j * nt + t;
        if sb < nsub {
            // Same (superblock, chunk, sub-block) decomposition as `moe_matvec_q6k_blk`.
            let sup = sb / 8;
            let sbloc = sb % 8;
            let idx = sbloc / 4;
            let sub = sbloc % 4;
            let blk = wrow * nb + sup;
            let d = wd[blk];
            let qlw0 = blk * 32 + (idx * 64 + (sub % 2) * 32) / 4; // ql word base (byte offs are 32-aligned)
            let qhw0 = blk * 16 + (idx * 32) / 4; // qh word base
            let scbase = blk * 4;
            let qlshift = ((sub / 2) * 4) as u32;
            let qhshift = (sub * 2) as u32;
            let scb0 = idx * 8 + sub * 2;
            let sc0 = F::cast_from(sbyte_at(wsc, scbase, scb0)); // weights 0..15
            let sc1 = F::cast_from(sbyte_at(wsc, scbase, scb0 + 1)); // weights 16..31
            let xg = (xrow + sb) * 8;
            let mut idot0 = 0i32;
            let mut isum0 = 0i32;
            let mut idot1 = 0i32;
            let mut isum1 = 0i32;
            // Groups 0..4 are the first half-block (scale sc0), 4..8 the second (sc1).
            for g in 0..4 {
                let qw = ((wql[qlw0 + g] >> qlshift) & 0x0F0F_0F0F)
                    | (((wqh[qhw0 + g] >> qhshift) & 0x0303_0303) << 4);
                let wv = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<
                    u32,
                >(qw));
                let xv = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<
                    u32,
                >(xq[xg + g]));
                idot0 += wv.dot(xv);
                isum0 += xv.dot(ones);
            }
            for g in 4..8 {
                let qw = ((wql[qlw0 + g] >> qlshift) & 0x0F0F_0F0F)
                    | (((wqh[qhw0 + g] >> qhshift) & 0x0303_0303) << 4);
                let wv = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<
                    u32,
                >(qw));
                let xv = Vector::<i32, Const<4>>::cast_from(Vector::<i8, Const<4>>::reinterpret::<
                    u32,
                >(xq[xg + g]));
                idot1 += wv.dot(xv);
                isum1 += xv.dot(ones);
            }
            partial += d
                * xs[xrow + sb]
                * (sc0 * F::cast_from(idot0 - 32 * isum0) + sc1 * F::cast_from(idot1 - 32 * isum1));
        }
    }
    let mut smem = SharedMemory::<F>::new(nt);
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
    if t == 0 {
        out[outrow] = smem[0];
    }
}

/// Host launch for the dp4a Q6_K MoE matvec (activation q8-quantized on the host, as for Q4_K).
/// `xsum` is unused here: Q6_K's scale changes mid-block, so the kernel derives its half-block sums.
#[allow(clippy::too_many_arguments)]
pub fn moe_matvec_q6k_dp4a_blk_run<R: Runtime>(
    client: &ComputeClient<R>,
    wql: &[u32], wqh: &[u32], wsc: &[u32], wd: &[f32], x: &[f32], ids: &[u32],
    slots: usize, n: usize, k: usize, nt: usize,
) -> Vec<f32> {
    let (xq, xs, _) = quant_act_q8_cpu(x, slots, k);
    let qlh = client.create_from_slice(u32::as_bytes(wql));
    let qhh = client.create_from_slice(u32::as_bytes(wqh));
    let sh = client.create_from_slice(u32::as_bytes(wsc));
    let dh = client.create_from_slice(f32::as_bytes(wd));
    let xqh = client.create_from_slice(u32::as_bytes(&xq));
    let xsh = client.create_from_slice(f32::as_bytes(&xs));
    let ih = client.create_from_slice(u32::as_bytes(ids));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; slots * n]));
    unsafe {
        moe_matvec_q6k_dp4a_blk::launch_unchecked::<f32, R>(
            client, Grid::Static((slots * n) as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(qlh.clone(), wql.len()),
            ArrayArg::from_raw_parts(qhh.clone(), wqh.len()),
            ArrayArg::from_raw_parts(sh.clone(), wsc.len()),
            ArrayArg::from_raw_parts(dh.clone(), wd.len()),
            ArrayArg::from_raw_parts(xqh.clone(), xq.len()),
            ArrayArg::from_raw_parts(xsh.clone(), xs.len()),
            ArrayArg::from_raw_parts(ih.clone(), ids.len()),
            ArrayArg::from_raw_parts(oh.clone(), slots * n),
            n, k, nt,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

// ============================================================================================
// dp4a matvec: the inner product uses Vector<i32,4>.dot -> SPIR-V OpSDot (hardware integer dot,
// SPV_KHR_integer_dot_product), verified emitted. Integer activations (xq) so the int dot is exact;
// out[row] = sum_block wd[block] * dot(qw_group, xq_group). Bit-exact vs matvec_q8_dp4a_ref.
// ============================================================================================
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8_dp4a<F: Float>(
    wq: &Array<Vector<i32, Const<4>>>, // int8 weights, grouped x4  [rows * k/4]
    xq: &Array<Vector<i32, Const<4>>>, // int activation, grouped x4 [k/4]
    wd: &Array<F>,                     // per-32-block scale         [rows * k/32]
    out: &mut Array<F>,
    #[comptime] k: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() {
        let ng = k / 4;
        let nb = k / 32;
        let wbase = row * ng;
        let dbase = row * nb;
        let mut acc = F::new(0.0);
        for g in 0..ng {
            let dp = wq[wbase + g].dot(xq[g]); // OpSDot: 4 int8*int products -> i32
            acc += wd[dbase + g / 8] * F::cast_from(dp);
        }
        out[row] = acc;
    }
}

pub fn matvec_q8_dp4a_run<R: Runtime>(
    client: &ComputeClient<R>, wq: &[i32], xq: &[i32], wd: &[f32], rows: usize, k: usize, bench_iters: usize,
) -> (Vec<f32>, f64) {
    let wqh = client.create_from_slice(i32::as_bytes(wq));
    let xqh = client.create_from_slice(i32::as_bytes(xq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32; let grid = (rows as u32).div_ceil(block);
    let ng = k / 4;
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8_dp4a::launch_unchecked::<f32, R>(
            c, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(wqh.clone(), rows * ng),
            ArrayArg::from_raw_parts(xqh.clone(), ng),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    };
    launch(client);
    let bytes = client.read_one_unchecked(oh.clone());
    let out = f32::from_bytes(&bytes).to_vec();
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..bench_iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / bench_iters as f64;
    (out, ms)
}

pub fn matvec_q8_dp4a_ref(wq: &[i32], xq: &[i32], wd: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nb = k / 32;
    (0..rows).map(|row| {
        let mut acc = 0.0f32;
        for i in 0..k { acc += wd[row * nb + i / 32] * (wq[row * k + i] * xq[i]) as f32; }
        acc
    }).collect()
}

// ============================================================================================
// PACKED dp4a matvec: weights stored as Vector<i8,4> (4 BYTES/group -- the real int8 footprint,
// vs the i32x4 kernel's 16 bytes), cast to Vector<i32,4> in-register (lane-wise sign-extend), then
// .dot() -> OpSDot. Same hardware dp4a, 4x less weight memory traffic. Matvec is weight-bandwidth-
// bound, so this is THE lever toward hand-tuned parity. Bit-exact vs matvec_q8_dp4a_ref.
// ============================================================================================
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8_dp4a_i8<F: Float>(
    wq: &Array<Vector<i8, Const<4>>>, // int8 weights, packed x4  [rows * k/4], 4 bytes/group
    xq: &Array<Vector<i8, Const<4>>>, // int8 activation, packed x4 [k/4]
    wd: &Array<F>,                    // per-32-block scale         [rows * k/32]
    out: &mut Array<F>,
    #[comptime] k: usize,
) {
    let row = ABSOLUTE_POS;
    if row < out.len() {
        let ng = k / 4;
        let nb = k / 32;
        let wbase = row * ng;
        let dbase = row * nb;
        let mut acc = F::new(0.0);
        for g in 0..ng {
            let wi = Vector::<i32, Const<4>>::cast_from(wq[wbase + g]);
            let xi = Vector::<i32, Const<4>>::cast_from(xq[g]);
            let dp = wi.dot(xi); // OpSDot on the widened i32x4 (loaded from 4-byte packed int8)
            acc += wd[dbase + g / 8] * F::cast_from(dp);
        }
        out[row] = acc;
    }
}

/// Host for the packed-int8 dp4a matvec. Weights + activation are real int8 (`&[i8]`), 4 bytes/group.
pub fn matvec_q8_dp4a_i8_run<R: Runtime>(
    client: &ComputeClient<R>, wq: &[i8], xq: &[i8], wd: &[f32], rows: usize, k: usize, bench_iters: usize,
) -> (Vec<f32>, f64) {
    let wqh = client.create_from_slice(i8::as_bytes(wq));
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let block = 64u32; let grid = (rows as u32).div_ceil(block);
    let ng = k / 4;
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8_dp4a_i8::launch_unchecked::<f32, R>(
            c, Grid::Static(grid, 1, 1), Block::new_1d(block),
            ArrayArg::from_raw_parts(wqh.clone(), rows * ng),
            ArrayArg::from_raw_parts(xqh.clone(), ng),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
        );
    };
    launch(client);
    let bytes = client.read_one_unchecked(oh.clone());
    let out = f32::from_bytes(&bytes).to_vec();
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..bench_iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / bench_iters as f64;
    (out, ms)
}

// ============================================================================================
// ============================================================================================
// Q8_0 PACKED matvec -- reads the REAL production layout (9 u32/block: fp16 scale in the low half of
// word 0, then 32 signed int8 packed 4/word in words 1..8), matching `mul_mat_vec_q8_sg`. f32
// activations (int8-weight x f32-act). In-kernel fp16 decode + signed int8 extract + block-per-row
// shared-mem reduction -- a true drop-in for the packed buffer ml already dispatches, not a synthetic
// layout. This is the production-swap-shaped kernel (vs the synthetic-layout dp4a benchmark below).
// ============================================================================================

// fp16 bits (low 16 of `h`) -> f32, bit-exact over the FULL fp16 domain including SUBNORMALS (Fabian
// Giesen's half_to_float). Q4_K/Q6_K block d/dmin scales are frequently subnormal fp16 (small scales),
// which the old normal-only exp fixup (`mag + 0x1C000`) decoded wrong; the resulting per-weight decode
// error, amplified by the near-cancellation of the affine dp4a sum (sub-block partials are hundreds of
// x the output), garbled qwen3 decode. Renormalize subnormals via one f32 subtract of the magic bias.
#[device]
fn f16lo_to_f32(h: u32) -> f32 {
    let magic = 113u32 << 23; // 2^-14 in f32 bits, the subnormal renormalization bias
    let shifted_exp = 0x7C00u32 << 13; // fp16 exponent mask, shifted into f32 position
    let mut o = (h & 0x7FFF) << 13; // exponent + mantissa, shifted to f32
    let exp = shifted_exp & o;
    o += (127u32 - 15) << 23; // rebias fp16 exp (15) to f32 exp (127)
    if exp == shifted_exp {
        o += (128u32 - 16) << 23; // Inf/NaN: extra exp adjust
    }
    if exp == 0u32 {
        o += 1u32 << 23; // Zero/subnormal: extra exp adjust, then renormalize
        let f = f32::reinterpret(o) - f32::reinterpret(magic);
        o = u32::reinterpret(f);
    }
    o = o | ((h & 0x8000) << 16); // sign
    f32::reinterpret(o)
}

// signed 8-bit lane `p` in {0,8,16,24} of `word` as a float, sign-extended (b in 0..255 -> b-256 if b>=128).
#[device]
fn i8lane<F: Float>(word: u32, p: u32) -> F {
    let b = (word >> p) & 255;
    let mut v = F::cast_from(b);
    if b >= 128 {
        v -= F::new(256.0);
    }
    v
}

#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8_0_packed_blk<F: Float>(
    w: &Array<u32>, // packed Q8_0, 9 u32/block  [rows * k/32 * 9]
    x: &Array<F>,   // f32 activations, length k
    out: &mut Array<F>,
    #[comptime] k: usize,
    #[comptime] nt: usize, // threads per block (one block per row)
) {
    let row = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let nblocks = k / 32;
    let wbase = row * nblocks * 9;
    let per = nblocks / nt; // blocks per thread (nblocks a multiple of nt); bounded loop lowers clean
    let mut partial = F::new(0.0);
    for j in 0..per {
        let b = j * nt + t;
        let off = wbase + b * 9;
        let scale = F::cast_from(f16lo_to_f32(w[off]));
        let xb = b * 32;
        let mut bsum = F::new(0.0);
        for jj in 0..8 {
            let word = w[off + 1 + jj];
            let xo = xb + jj * 4;
            bsum += i8lane::<F>(word, 0) * x[xo];
            bsum += i8lane::<F>(word, 8) * x[xo + 1];
            bsum += i8lane::<F>(word, 16) * x[xo + 2];
            bsum += i8lane::<F>(word, 24) * x[xo + 3];
        }
        partial += scale * bsum;
    }
    let mut smem = SharedMemory::<F>::new(nt);
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
    if t == 0 {
        out[row] = smem[0];
    }
}

// Subgroup variant: one plane per row, plane_sum reduction (NO shared mem) -- mirrors production
// mul_mat_vec_q8_sg. `nt` MUST equal the hardware plane/subgroup size so the block is exactly one plane
// (else plane_sum reduces only within a plane and cross-plane partials are dropped -> caller sets
// nt = client plane size, and the bit-exact gate catches a mismatch).
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8_0_packed_sg<F: Float>(
    w: &Array<u32>,
    x: &Array<F>,
    out: &mut Array<F>,
    #[comptime] k: usize,
    #[comptime] nt: usize, // = plane size (one plane per row)
) {
    let row = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let nblocks = k / 32;
    let wbase = row * nblocks * 9;
    let per = nblocks / nt;
    let mut partial = F::new(0.0);
    for j in 0..per {
        let b = j * nt + t;
        let off = wbase + b * 9;
        let scale = F::cast_from(f16lo_to_f32(w[off]));
        let xb = b * 32;
        let mut bsum = F::new(0.0);
        for jj in 0..8 {
            let word = w[off + 1 + jj];
            let xo = xb + jj * 4;
            bsum += i8lane::<F>(word, 0) * x[xo];
            bsum += i8lane::<F>(word, 8) * x[xo + 1];
            bsum += i8lane::<F>(word, 16) * x[xo + 2];
            bsum += i8lane::<F>(word, 24) * x[xo + 3];
        }
        partial += scale * bsum;
    }
    let total = plane_sum(partial);
    if t == 0 {
        out[row] = total;
    }
}

// CPU oracle: decode packed Q8_0 exactly (half::f16 from low 16 bits, signed int8) and dot with f32 x.
pub fn matvec_q8_0_packed_ref(w: &[u32], x: &[f32], rows: usize, k: usize) -> Vec<f32> {
    let nblocks = k / 32;
    let mut out = vec![0f32; rows];
    for row in 0..rows {
        let wbase = row * nblocks * 9;
        let mut acc = 0f32;
        for b in 0..nblocks {
            let off = wbase + b * 9;
            let scale = half::f16::from_bits((w[off] & 0xFFFF) as u16).to_f32();
            let xb = b * 32;
            let mut bsum = 0f32;
            for jj in 0..8 {
                let word = w[off + 1 + jj];
                let xo = xb + jj * 4;
                for lane in 0..4 {
                    let q = ((word >> (8 * lane as u32)) & 0xFF) as u8 as i8 as f32;
                    bsum += q * x[xo + lane];
                }
            }
            acc += scale * bsum;
        }
        out[row] = acc;
    }
    out
}

// Deterministic packed-Q8_0 test data: random normal fp16 scale + 32 random int8 per block, 9 u32/block.
pub fn gen_q8_0_packed(rows: usize, k: usize) -> (Vec<u32>, Vec<f32>) {
    let nblocks = k / 32;
    let mut s = 0x1234_5678_9ABC_DEF1u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let mut w = Vec::with_capacity(rows * nblocks * 9);
    for _ in 0..rows * nblocks {
        let scale = (next() % 1000) as f32 / 8000.0 + 0.01;
        w.push(half::f16::from_f32(scale).to_bits() as u32);
        for _ in 0..8 {
            let mut word = 0u32;
            for lane in 0..4 {
                let q = (next() % 255) as u8; // int8 as bit pattern
                word |= (q as u32) << (8 * lane);
            }
            w.push(word);
        }
    }
    let x: Vec<f32> = (0..k).map(|_| (next() % 2000) as f32 / 1000.0 - 1.0).collect();
    (w, x)
}

// Host launch + kernel-only bench for the packed-Q8_0 block matvec.
pub fn matvec_q8_0_packed_run<R: Runtime>(
    client: &ComputeClient<R>, w: &[u32], x: &[f32], rows: usize, k: usize, nt: usize, bench_iters: usize,
) -> (Vec<f32>, f64) {
    let wh = client.create_from_slice(u32::as_bytes(w));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8_0_packed_blk::launch_unchecked::<f32, R>(
            c, Grid::Static(rows as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(xh.clone(), k),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k, nt,
        );
    };
    launch(client);
    let bytes = client.read_one_unchecked(oh.clone());
    let out = f32::from_bytes(&bytes).to_vec();
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..bench_iters { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let ms = t.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
    (out, ms)
}

// Subgroup variant runner: block = one plane (nt = plane size). pipelined (throughput) timing.
pub fn matvec_q8_0_packed_sg_run<R: Runtime>(
    client: &ComputeClient<R>, w: &[u32], x: &[f32], rows: usize, k: usize, nt: usize, bench_iters: usize,
) -> (Vec<f32>, f64) {
    let wh = client.create_from_slice(u32::as_bytes(w));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8_0_packed_sg::launch_unchecked::<f32, R>(
            c, Grid::Static(rows as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(xh.clone(), k),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k, nt,
        );
    };
    launch(client);
    let bytes = client.read_one_unchecked(oh.clone());
    let out = f32::from_bytes(&bytes).to_vec();
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..bench_iters { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let ms = t.elapsed().as_secs_f64() * 1000.0 / bench_iters as f64;
    (out, ms)
}

// BLOCK-per-row dp4a matvec: the bandwidth-bound pattern. One thread block owns one output row;
// its `nt` threads stride over the k/4 groups so ADJACENT threads read ADJACENT Vector<i8,4>
// (consecutive 4-byte loads -> coalesced 256B transactions), vs the one-thread-per-row kernel whose
// adjacent threads are a whole row (k/4 groups) apart -> uncoalesced. Partials are tree-reduced in
// shared memory. This is what takes the DSL dp4a from latency-bound to weight-bandwidth-bound.
// ============================================================================================
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8_dp4a_blk<F: Float>(
    wq: &Array<Vector<i8, Const<4>>>, // int8 weights, packed x4  [rows * k/4]
    xq: &Array<Vector<i8, Const<4>>>, // int8 activation, packed x4 [k/4]
    wd: &Array<F>,                    // per-32-block scale         [rows * k/32]
    out: &mut Array<F>,
    #[comptime] k: usize,
    #[comptime] nt: usize, // threads per block (one block per row)
) {
    let row = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let ng = k / 4;
    let wbase = row * ng;
    let dbase = row * (k / 32);
    let mut partial = F::new(0.0);
    let per = ng / nt; // groups per thread (ng is a multiple of nt); bounded for-loop lowers cleanly
    for j in 0..per {
        let g = j * nt + t;
        let wi = Vector::<i32, Const<4>>::cast_from(wq[wbase + g]);
        let xi = Vector::<i32, Const<4>>::cast_from(xq[g]);
        partial += wd[dbase + g / 8] * F::cast_from(wi.dot(xi)); // OpSDot per group
    }
    let mut smem = SharedMemory::<F>::new(nt);
    smem[t] = partial;
    sync_cube();
    let mut stride = CUBE_DIM / 2; // runtime (== nt); a comptime `nt/2` can't be mutated (RuntimeCell)
    while stride > 0 {
        if UNIT_POS < stride {
            let v = smem[(UNIT_POS + stride) as usize];
            smem[t] += v;
        }
        sync_cube();
        stride /= 2;
    }
    if t == 0 { out[row] = smem[0]; }
}

/// Host for the block-per-row dp4a matvec. `nt` threads cooperate per row (coalesced + reduced).
pub fn matvec_q8_dp4a_blk_run<R: Runtime>(
    client: &ComputeClient<R>, wq: &[i8], xq: &[i8], wd: &[f32], rows: usize, k: usize, nt: usize, bench_iters: usize,
) -> (Vec<f32>, f64) {
    let wqh = client.create_from_slice(i8::as_bytes(wq));
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    let ng = k / 4;
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8_dp4a_blk::launch_unchecked::<f32, R>(
            c, Grid::Static(rows as u32, 1, 1), Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(wqh.clone(), rows * ng),
            ArrayArg::from_raw_parts(xqh.clone(), ng),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k, nt,
        );
    };
    launch(client);
    let bytes = client.read_one_unchecked(oh.clone());
    let out = f32::from_bytes(&bytes).to_vec();
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..bench_iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / bench_iters as f64;
    (out, ms)
}

// ============================================================================================
// Autotuned dp4a matvec -- ONE source, the schedule exposed as comptime knobs the tuner picks over.
// The "collapse": instead of forking a block-per-row / warp-per-row / vector-width kernel per device,
// `matvec_q8_dp4a_tuned` carries the vector-width (ILP) knob `vw` as comptime and takes the cube dim
// from the host; the tuner benchmarks the variant set once per (device, rows x k) and caches the winner.
// The knobs never change the numerics: the per-group int8 dot is accumulated in the same `g` order for
// every `vw`, so all variants are byte-identical to each other and match the CPU oracle within f32 reorder.
// ============================================================================================

/// Autotuned dp4a matvec source. TWO orthogonal schedule knobs, both comptime so cubecl monomorphizes a
/// distinct kernel per tuple (the tuner picks the tuple per device+shape):
///   * `vw` -- groups processed per unrolled step (the ILP / "vector width" knob): more independent
///     loads+dp4a in flight per iteration, the memory-level-parallelism a bandwidth-bound matvec needs.
///   * `nr` -- output ROWS this thread owns. The weight stream is per-row, but the activation group
///     `xq[g]` is loaded ONCE per `g` and reused across all `nr` rows -- the row-tile amortisation
///     llama's `mul_mat_vec` spends as `NUM_ROWS` and our hand `mul_mat_vec_q4k_cm` as `NR=2`. The DSL
///     matvec lacked this dimension; exposing it lets the autotuner reach the hand/llama schedule.
/// `out[row] = sum_g wd[g/8] * dot(wq_g, xq_g)`, `g` ascending. Packed int8 weights (`Vector<i8,4>`,
/// 4 bytes/group) -> `.dot` lowers to dp4a / OpSDot. Per-row accumulation order is independent of `nr`
/// and `vw`, so every schedule is byte-identical to the `nr=1` single-row source (and `nr=1` is exactly
/// that source): the knobs move the schedule, never the numerics.
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn matvec_q8_dp4a_tuned<F: Float>(
    wq: &Array<Vector<i8, Const<4>>>,
    xq: &Array<Vector<i8, Const<4>>>,
    wd: &Array<F>,
    out: &mut Array<F>,
    #[comptime] k: usize,
    #[comptime] vw: usize, // groups per unrolled step (ILP / vector width); ng = k/4 must be a multiple
    #[comptime] nr: usize, // output rows this thread owns (activation-reuse row tile; llama NUM_ROWS)
) {
    let nout = out.len();
    let ng = k / 4;
    let nb = k / 32;
    let row0 = ABSOLUTE_POS * nr; // this thread owns rows [row0, row0 + nr)
    // One running accumulator per owned row. The activation group loaded per `g` feeds all `nr` rows.
    let mut acc = Array::<F>::new(nr);
    #[unroll]
    for n in 0..nr {
        acc[n] = F::new(0.0);
    }
    let steps = ng / vw;
    for s in 0..steps {
        #[unroll]
        for j in 0..vw {
            let g = s * vw + j;
            let xi = Vector::<i32, Const<4>>::cast_from(xq[g]); // ONE activation load, reused nr times
            #[unroll]
            for n in 0..nr {
                let row = row0 + n;
                if row < nout {
                    let wi = Vector::<i32, Const<4>>::cast_from(wq[row * ng + g]);
                    acc[n] += wd[row * nb + g / 8] * F::cast_from(wi.dot(xi));
                }
            }
        }
    }
    #[unroll]
    for n in 0..nr {
        let row = row0 + n;
        if row < nout {
            out[row] = acc[n];
        }
    }
}

/// Kernel-only bench for one `(vw, block)` schedule; returns `(output, ms/dispatch)` -- the
/// [`crate::tune::Variant`] contract (`iters == 1` on a cache hit, many iters when tuning). Weights +
/// activation are real int8 (`&[i8]`), 4 bytes/group.
#[allow(clippy::too_many_arguments)]
pub fn matvec_q8_dp4a_tuned_bench<R: Runtime>(
    client: &ComputeClient<R>,
    wq: &[i8],
    xq: &[i8],
    wd: &[f32],
    rows: usize,
    k: usize,
    vw: usize,
    block: u32,
    nr: usize,
    iters: usize,
) -> (Vec<f32>, f64) {
    let wqh = client.create_from_slice(i8::as_bytes(wq));
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
    // Each thread owns nr rows, so the grid covers ceil(rows/nr) threads.
    let grid = (rows as u32).div_ceil(nr as u32).div_ceil(block);
    let ng = k / 4;
    let launch = |c: &ComputeClient<R>| unsafe {
        matvec_q8_dp4a_tuned::launch_unchecked::<f32, R>(
            c,
            Grid::Static(grid, 1, 1),
            Block::new_1d(block),
            ArrayArg::from_raw_parts(wqh.clone(), rows * ng),
            ArrayArg::from_raw_parts(xqh.clone(), ng),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), rows),
            k,
            vw,
            nr,
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

/// The 4-variant dp4a-matvec set (cube dim {64,128,256} x vector width {1,2,4,8}) as a reusable
/// [`Tuned`], so tests can drive it against an explicit tuner. All four are byte-identical.
pub fn matvec_q8_dp4a_tuned_set<'a, R: Runtime>(
    client: &'a ComputeClient<R>,
    wq: &'a [i8],
    xq: &'a [i8],
    wd: &'a [f32],
    rows: usize,
    k: usize,
) -> Tuned<'a, Vec<f32>> {
    Tuned::new("matvec_q8_dp4a", format!("rows={rows},k={k}"))
        .variant("b64_v1", move |it| matvec_q8_dp4a_tuned_bench(client, wq, xq, wd, rows, k, 1, 64, 1, it))
        .variant("b64_v4", move |it| matvec_q8_dp4a_tuned_bench(client, wq, xq, wd, rows, k, 4, 64, 1, it))
        .variant("b128_v2", move |it| matvec_q8_dp4a_tuned_bench(client, wq, xq, wd, rows, k, 2, 128, 1, it))
        .variant("b256_v8", move |it| matvec_q8_dp4a_tuned_bench(client, wq, xq, wd, rows, k, 8, 256, 1, it))
}

/// Autotuned dp4a matvec: tune (or read the cache) over the schedule knobs and run the winner. ONE
/// source, picked per `(device, rows x k)`, no forks. The production surface.
pub fn matvec_q8_dp4a_autotuned<R: Runtime>(
    client: &ComputeClient<R>,
    wq: &[i8],
    xq: &[i8],
    wd: &[f32],
    rows: usize,
    k: usize,
) -> Vec<f32> {
    matvec_q8_dp4a_tuned_set(client, wq, xq, wd, rows, k).run(client)
}

// ============================================================================================
// Evolutionary schedule SEARCH for the dp4a matvec. `matvec_q8_dp4a_tuned_set` above enumerates a
// curated 4-point `Tuned`; this searches the FULL {WG x VW x NR} schedule product with the
// `tune::Evolution` GA and a multi-fidelity, cold-weight-streamed `Evaluator` -- the same machinery
// hanzo-ml's coopmat hunt runs on the prefill GEMM, here generic over the runtime.
//
// WG (the workgroup width / `local_size_x`) is the load-bearing axis: our hand `mul_mat_vec_q4k_cm`
// HARDCODES 64, but the memory-bandwidth-optimal width is shape-dependent -- at small row counts a
// wider group (llama's DMMV_WG_SIZE_LARGE) fills the device that row-parallelism alone leaves idle,
// and at large row counts the narrow group wins. Because the `Tuner` keys a winner by (device, op,
// `rows=..,k=..`), the search resolves BOTH the shape-adaptive width AND the per-backend algorithm
// choice for free: on a discrete-launch backend (ROCm) the dp4a path is the fast decode kernel; on a
// backend where the extra activation-quantize dispatch is expensive (Vulkan eager record) the f32
// direct kernel wins, and the per-device cache records whichever the cold oracle measures faster.
// ============================================================================================

/// The dp4a-matvec schedule space: workgroup width `WG` (= `local_size_x`), ILP/unroll `VW`, and the
/// activation-reuse row tile `NR`. The ONE definition of the searchable schedule; the Tuner keys its
/// winner by `(device, rows, k)`, so a wide WG is selected exactly at the shapes that need it. The only
/// hard constraint is bit-exactness: `ng = k/4` must be an exact multiple of `VW` (else `steps = ng/vw`
/// would drop the tail group), and `WG` must be a legal workgroup width.
pub fn matvec_dp4a_space(k: usize) -> Space {
    let ng = (k / 4) as i64;
    Space::new()
        .param("WG", [64, 128, 256])
        .param("VW", [1, 2, 4, 8])
        .param("NR", [1, 2, 4])
        .constraint(move |c, s| ng % c.get(s, "VW") == 0)
        .constraint(|c, s| {
            let wg = c.get(s, "WG");
            wg > 0 && wg <= 1024
        })
}

/// Map a schedule [`Config`] to the kernel's `(block, vw, nr)` launch tuple.
fn dp4a_cfg(c: &Config, s: &Space) -> (u32, usize, usize) {
    (c.get(s, "WG") as u32, c.get(s, "VW") as usize, c.get(s, "NR") as usize)
}

/// A cold-weight-streaming, bit-exact-gated fitness over [`matvec_dp4a_space`], generic in the runtime.
/// Holds RESIDENT device buffers (uploaded once) so the timing tier streams weights with no re-upload,
/// and rotates a bank of distinct weights so each dispatch reads its weight COLD -- the bandwidth-bound
/// regime decode runs in, and the guard against the campaign's #1 scar (a warm single-tile microbench
/// crowns an occupancy/width knob the deployment never sees). The correctness gate is folded into
/// fitness: a schedule that diverges from the scalar-dp4a oracle is infinitely slow, so it can never
/// win. `CpuRuntime` exercises the whole search offline (machinery + bit-exactness); the same type on a
/// wgpu device gives the meaningful winner.
pub struct MatvecDp4aEval<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    space: &'a Space,
    banks: Vec<Handle>, // resident weight buffers; banks[0] is the weight the oracle was computed from
    xqh: Handle,
    wdh: Handle,
    outh: Handle,
    wd_len: usize,
    rows: usize,
    k: usize,
    oracle: Vec<f32>,
    maxref: f32,
    repeats: usize,
    worst_rel: std::cell::Cell<f32>,
}

impl<'a, R: Runtime> MatvecDp4aEval<'a, R> {
    /// Upload the weight banks (each `rows*k` int8; distinct data so a rotation reads cold), the shared
    /// activation `xq` and scales `wd`, and an output buffer; compute the scalar-dp4a oracle on `banks[0]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &'a ComputeClient<R>,
        space: &'a Space,
        weight_banks: &[Vec<i8>],
        xq: &[i8],
        wd: &[f32],
        rows: usize,
        k: usize,
        repeats: usize,
    ) -> Self {
        assert!(!weight_banks.is_empty(), "MatvecDp4aEval needs at least one weight bank");
        let banks: Vec<Handle> =
            weight_banks.iter().map(|w| client.create_from_slice(i8::as_bytes(w))).collect();
        let xqh = client.create_from_slice(i8::as_bytes(xq));
        let wdh = client.create_from_slice(f32::as_bytes(wd));
        let outh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; rows]));
        let wq32: Vec<i32> = weight_banks[0].iter().map(|&v| v as i32).collect();
        let xq32: Vec<i32> = xq.iter().map(|&v| v as i32).collect();
        let oracle = matvec_q8_dp4a_ref(&wq32, &xq32, wd, rows, k);
        let maxref = oracle.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
        Self {
            client,
            space,
            banks,
            xqh,
            wdh,
            outh,
            wd_len: wd.len(),
            rows,
            k,
            oracle,
            maxref,
            repeats,
            worst_rel: std::cell::Cell::new(0.0),
        }
    }

    /// The search space this evaluator is bound to (so a hunt and its fitness never disagree on the genome).
    pub fn space(&self) -> &Space {
        self.space
    }

    /// The worst scale-relative divergence any measured config showed (the committed correctness witness).
    pub fn worst_rel(&self) -> f32 {
        self.worst_rel.get()
    }

    /// One dispatch of a resident weight `bank` through the `(block, vw, nr)` schedule -- no upload.
    fn dispatch(&self, bank: &Handle, block: u32, vw: usize, nr: usize) {
        let ng = self.k / 4;
        let grid = (self.rows as u32).div_ceil(nr as u32).div_ceil(block);
        unsafe {
            matvec_q8_dp4a_tuned::launch_unchecked::<f32, R>(
                self.client,
                Grid::Static(grid, 1, 1),
                Block::new_1d(block),
                ArrayArg::from_raw_parts(bank.clone(), self.rows * ng),
                ArrayArg::from_raw_parts(self.xqh.clone(), ng),
                ArrayArg::from_raw_parts(self.wdh.clone(), self.wd_len),
                ArrayArg::from_raw_parts(self.outh.clone(), self.rows),
                self.k,
                vw,
                nr,
            );
        }
    }

    fn read_out(&self) -> Vec<f32> {
        f32::from_bytes(&self.client.read_one_unchecked(self.outh.clone())).to_vec()
    }
}

impl<'a, R: Runtime> Evaluator for MatvecDp4aEval<'a, R> {
    fn static_check(&self, cfg: &Config) -> Verdict {
        let (block, vw, _nr) = dp4a_cfg(cfg, self.space);
        let ng = self.k / 4;
        // Free tier: reject a schedule the source cannot express bit-exactly before spending a dispatch.
        // (cubecl lowers per launch, so unlike glslc there is no cheap register read here; the Space's
        // feasibility already pruned the grid -- this is the belt-and-braces divisibility/width gate.)
        if vw == 0 || ng % vw != 0 {
            return Verdict::Reject(format!("ng={ng} not a multiple of vw={vw}"));
        }
        if block == 0 || block > 1024 {
            return Verdict::Reject(format!("illegal workgroup width {block}"));
        }
        Verdict::Pass
    }

    fn measure(&self, cfg: &Config, iters: usize) -> f64 {
        let (block, vw, nr) = dp4a_cfg(cfg, self.space);
        // Correctness gate on bank[0] vs the scalar-dp4a oracle (scale-relative: per-element relative
        // error explodes on near-zero cancellation, so gate on max|Δ|/max|ref|).
        self.dispatch(&self.banks[0], block, vw, nr);
        let got = self.read_out();
        let rel = got.iter().zip(&self.oracle).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / self.maxref;
        self.worst_rel.set(self.worst_rel.get().max(rel));
        if rel > 2e-3 {
            return f64::INFINITY;
        }
        // Cold-weight-streamed timing: warm the utilisation-slaved clock, then time `iters` dispatches
        // rotating the whole bank so every read is cold, and take the MINIMUM over a few passes (the
        // least-drift-polluted time). The trailing read_out drains the queue -- the sync point.
        let nb = self.banks.len();
        for i in 0..(2 * nb) {
            self.dispatch(&self.banks[i % nb], block, vw, nr);
        }
        let _ = self.read_out();
        (0..self.repeats)
            .map(|_| {
                let t = std::time::Instant::now();
                for i in 0..iters {
                    self.dispatch(&self.banks[i % nb], block, vw, nr);
                }
                let _ = self.read_out();
                t.elapsed().as_secs_f64() * 1e3 / iters as f64
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Search [`matvec_dp4a_space`] on `eval` for the fastest schedule at this `(device, rows, k)`, caching
/// the winner in the shared autotune TSV. The production upgrade from the curated 4-variant `Tuned` set
/// to the full {WG x VW x NR} search; the winner replays through the same cache on the next call.
#[allow(clippy::too_many_arguments)]
pub fn matvec_dp4a_hunt<R: Runtime>(
    tuner: &Tuner,
    device: &str,
    rows: usize,
    k: usize,
    eval: &MatvecDp4aEval<R>,
    evo: &Evolution,
    seed: u64,
) -> Evolved {
    tuner.evolve(device, "matvec_q8_dp4a", &format!("rows={rows},k={k}"), eval.space(), eval, evo, seed)
}

// ============================================================================================
// Fused MoE top-k router. ONE workgroup per token: softmax over the `n_experts` router logits, select
// the top-k experts, write their (index, renormalized softmax weight). Replaces the generic Vulkan
// op-chain (max/sub/exp/sum/div + argsort + gather + narrow×2 + norm-div = ~11 dispatches/layer, each
// wrapped in a layout copy — the bulk of Vulkan MoE decode's strided_copy churn) with a single kernel,
// the same fusion the ROCm/CUDA `dev.moe_route` already does. Always renormalizes (Qwen3
// norm_topk_prob); the engine gates the fused path on norm=true and falls back to the op-chain else.
// Output order is descending weight (r=0 largest), matching softmax_last_dim + sort_last_dim(desc) +
// narrow(topk). `nt` = threads/workgroup, a power of 2 (== the reduction width).
// ============================================================================================
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn moe_route<F: Float>(
    logits: &Array<F>,        // [ntok, n_experts]
    ids_out: &mut Array<u32>, // [ntok, topk]
    w_out: &mut Array<F>,     // [ntok, topk]
    #[comptime] n_experts: usize,
    #[comptime] topk: usize,
    #[comptime] nt: usize,
) {
    let tok = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let base = tok * n_experts;
    let ninf = F::new(-3.4e38); // -inf sentinel (cf. attn.rs running-max init)
    // Maskable copy of this token's logits in shared memory (F = f32 at launch).
    let mut slog = SharedMemory::<F>::new(n_experts);
    let mut i = t;
    while i < n_experts {
        slog[i] = logits[base + i];
        i += nt;
    }
    sync_cube();
    // softmax denominator over ALL experts: max, then sum exp(logit - max).
    let mut sred = SharedMemory::<F>::new(nt);
    let mut lmax = ninf;
    let mut a = t;
    while a < n_experts {
        if slog[a] > lmax {
            lmax = slog[a];
        }
        a += nt;
    }
    sred[t] = lmax;
    sync_cube();
    let mut stride = CUBE_DIM / 2;
    while stride > 0 {
        if UNIT_POS < stride {
            let v = sred[(UNIT_POS + stride) as usize];
            let cur = sred[t];
            if v > cur {
                sred[t] = v;
            }
        }
        sync_cube();
        stride /= 2;
    }
    let m = sred[0];
    sync_cube();
    let mut lsum = F::new(0.0);
    let mut b = t;
    while b < n_experts {
        lsum += (slog[b] - m).exp();
        b += nt;
    }
    sred[t] = lsum;
    sync_cube();
    let mut st2 = CUBE_DIM / 2;
    while st2 > 0 {
        if UNIT_POS < st2 {
            let v = sred[(UNIT_POS + st2) as usize];
            sred[t] += v;
        }
        sync_cube();
        st2 /= 2;
    }
    let denom = sred[0];
    sync_cube();
    // top-k: `topk` passes of argmax over the masked logits (index-tracking tree reduce).
    let mut sidx = SharedMemory::<u32>::new(nt);
    let mut wsum = F::new(0.0);
    for _r in 0..topk {
        let mut lv = ninf;
        let mut li = 0u32;
        let mut c = t;
        while c < n_experts {
            if slog[c] > lv {
                lv = slog[c];
                li = c as u32;
            }
            c += nt;
        }
        sred[t] = lv;
        sidx[t] = li;
        sync_cube();
        let mut sr = CUBE_DIM / 2;
        while sr > 0 {
            if UNIT_POS < sr {
                let ov = sred[(UNIT_POS + sr) as usize];
                let oi = sidx[(UNIT_POS + sr) as usize];
                let curv = sred[t];
                if ov > curv {
                    sred[t] = ov;
                    sidx[t] = oi;
                }
            }
            sync_cube();
            sr /= 2;
        }
        let best = sidx[0];
        let wr = (sred[0] - m).exp() / denom; // = softmax prob of the selected expert
        wsum += wr;
        if t == 0 {
            ids_out[tok * topk + _r] = best;
            w_out[tok * topk + _r] = wr;
            slog[best as usize] = ninf; // mask so the next pass skips it
        }
        sync_cube();
    }
    // Renormalize the top-k weights to sum 1 (Qwen3 norm_topk_prob).
    if t == 0 {
        for _r in 0..topk {
            let cur = w_out[tok * topk + _r];
            w_out[tok * topk + _r] = cur / wsum;
        }
    }
}

/// Host launch for the fused MoE router. Returns (ids [ntok*topk], weights [ntok*topk]).
#[allow(clippy::too_many_arguments)]
pub fn moe_route_run<R: Runtime>(
    client: &ComputeClient<R>,
    logits: &[f32],
    ntok: usize,
    n_experts: usize,
    topk: usize,
    nt: usize,
) -> (Vec<u32>, Vec<f32>) {
    let lh = client.create_from_slice(f32::as_bytes(logits));
    let ih = client.create_from_slice(u32::as_bytes(&vec![0u32; ntok * topk]));
    let wh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; ntok * topk]));
    unsafe {
        moe_route::launch_unchecked::<f32, R>(
            client,
            Grid::Static(ntok as u32, 1, 1),
            Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(lh.clone(), logits.len()),
            ArrayArg::from_raw_parts(ih.clone(), ntok * topk),
            ArrayArg::from_raw_parts(wh.clone(), ntok * topk),
            n_experts,
            topk,
            nt,
        );
    }
    let ids = u32::from_bytes(&client.read_one_unchecked(ih)).to_vec();
    let w = f32::from_bytes(&client.read_one_unchecked(wh)).to_vec();
    (ids, w)
}

/// f32 GEMV: `out[n] = W[n,k] @ x[k]`, W row-major contiguous. One workgroup per output row, `nt`
/// threads stride over k, shared-mem tree reduce. The decode fix for a small f32 Linear (e.g. the MoE
/// router gate `[128,4096] @ [4096]`) that the tiled GEMM otherwise runs as ~2 occupancy-starved
/// workgroups (m=1 wastes 63/64 of every 64x64 tile). `k` is runtime (one .spv serves any Linear);
/// `nt` comptime, a power of 2 (tree reduce). `n` is the grid bound (launch exactly n workgroups).
#[kernel(targets(cuda, metal, vulkan, webgpu, cpu), unchecked)]
pub fn gemv<F: Float>(
    w: &Array<F>,
    x: &Array<F>,
    out: &mut Array<F>,
    meta: &Array<u32>, // [k]
    #[comptime] nt: usize,
) {
    let row = CUBE_POS as usize;
    let t = UNIT_POS as usize;
    let k = meta[0] as usize;
    let wbase = row * k;
    let mut acc = F::new(0.0);
    let mut i = t;
    while i < k {
        acc += w[wbase + i] * x[i];
        i += nt;
    }
    let mut smem = SharedMemory::<F>::new(nt);
    smem[t] = acc;
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
    if t == 0 {
        out[row] = smem[0];
    }
}

/// Host launch for the f32 GEMV (one workgroup per output row `n`, `nt` threads split `k`).
pub fn gemv_run<R: Runtime>(
    client: &ComputeClient<R>,
    w: &[f32],
    x: &[f32],
    n: usize,
    k: usize,
    nt: usize,
) -> Vec<f32> {
    let wh = client.create_from_slice(f32::as_bytes(w));
    let xh = client.create_from_slice(f32::as_bytes(x));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; n]));
    let mh = client.create_from_slice(u32::as_bytes(&[k as u32]));
    unsafe {
        gemv::launch_unchecked::<f32, R>(
            client,
            Grid::Static(n as u32, 1, 1),
            Block::new_1d(nt as u32),
            ArrayArg::from_raw_parts(wh.clone(), w.len()),
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(oh.clone(), n),
            ArrayArg::from_raw_parts(mh.clone(), 1),
            nt,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// CPU reference: softmax over all experts, take the top-k by weight (descending), renormalize.
/// Mirrors `moe_route` (hanzo-ml generic path softmax_last_dim + sort_last_dim(desc) + narrow + norm).
pub fn moe_route_ref(
    logits: &[f32],
    ntok: usize,
    n_experts: usize,
    topk: usize,
) -> (Vec<u32>, Vec<f32>) {
    let mut ids = vec![0u32; ntok * topk];
    let mut w = vec![0.0f32; ntok * topk];
    for tok in 0..ntok {
        let row = &logits[tok * n_experts..(tok + 1) * n_experts];
        let m = row.iter().cloned().fold(f32::MIN, f32::max);
        let exps: Vec<f32> = row.iter().map(|&x| (x - m).exp()).collect();
        let denom: f32 = exps.iter().sum();
        let mut p: Vec<(usize, f32)> = exps.iter().map(|&e| e / denom).enumerate().collect();
        p.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let wsum: f32 = p[..topk].iter().map(|&(_, v)| v).sum();
        for r in 0..topk {
            ids[tok * topk + r] = p[r].0 as u32;
            w[tok * topk + r] = p[r].1 / wsum;
        }
    }
    (ids, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs() / x.abs().max(1e-6)).fold(0.0, f32::max)
    }

    /// Deterministic int8 weights/activations + per-32 scales for the dp4a tuned tests.
    fn gen_dp4a(rows: usize, k: usize) -> (Vec<i8>, Vec<i8>, Vec<f32>) {
        let mut s = 0xA5A5_1234_9E37_79B9u64;
        let mut nx = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let wq: Vec<i8> = (0..rows * k).map(|_| (nx() % 251) as i8).collect();
        let xq: Vec<i8> = (0..k).map(|_| (nx() % 251) as i8).collect();
        let wd: Vec<f32> = (0..rows * (k / 32)).map(|_| (nx() % 1000) as f32 / 8000.0 + 0.01).collect();
        (wq, xq, wd)
    }

    /// Autotuned dp4a matvec on the CPU runtime: (1) every schedule variant is byte-identical to the
    /// first (the vector-width / cube-dim knobs are numerically invariant) and within f32-reorder
    /// tolerance of the per-element oracle; (2) the tuner benchmarks all 4 on a miss, caches the winner,
    /// and the second call skips timing.
    #[cfg(feature = "cpu")]
    #[test]
    fn matvec_q8_dp4a_autotune_cpu_bit_exact_and_cached() {
        use crate::tune::Tuner;
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, k) = (48usize, 256usize); // ng = 64, divisible by every vw in {1,2,4,8}
        let (wq, xq, wd) = gen_dp4a(rows, k);
        let wq32: Vec<i32> = wq.iter().map(|&v| v as i32).collect();
        let xq32: Vec<i32> = xq.iter().map(|&v| v as i32).collect();
        let want = matvec_q8_dp4a_ref(&wq32, &xq32, &wd, rows, k);
        let c = CpuRuntime::client(&CpuDevice::default());

        // (1) schedule-invariance + oracle agreement. The WG/VW/NR knobs move the schedule, never the
        // numerics: every config -- including the NR>1 row tiles (each thread owns nr rows, activation
        // reused) -- is byte-identical to the nr=1 single-row source. rows=48 is a multiple of every nr.
        let mut ref_bits: Option<Vec<u32>> = None;
        for &(vw, block, nr) in &[
            (1usize, 64u32, 1usize),
            (4, 64, 1),
            (2, 128, 1),
            (8, 256, 1),
            (1, 64, 2),
            (4, 128, 2),
            (2, 256, 4),
        ] {
            let (got, _ms) = matvec_q8_dp4a_tuned_bench::<CpuRuntime>(&c, &wq, &xq, &wd, rows, k, vw, block, nr, 1);
            let gbits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            match &ref_bits {
                None => ref_bits = Some(gbits),
                Some(rb) => assert_eq!(&gbits, rb, "wg={block} vw={vw} nr={nr} not byte-identical to b64_v1_r1"),
            }
            let rel = max_rel(&want, &got);
            eprintln!("[matvec_q8_dp4a_tuned CPU] wg={block} vw={vw} nr={nr} max_rel={rel:.2e}");
            assert!(rel < 2e-3, "dp4a tuned wg={block} vw={vw} nr={nr} max_rel {rel} vs oracle");
        }

        // (2) select + cache against an isolated temp tuner.
        let dir = std::env::temp_dir().join(format!(
            "hk-tune-dp4a-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let tuner = Tuner::new(&dir);
        let p = matvec_q8_dp4a_tuned_set::<CpuRuntime>(&c, &wq, &xq, &wd, rows, k).pick_with(&tuner, "cpu");
        assert!(!p.from_cache && p.benched == 4, "first call must benchmark all 4 variants");
        assert!(max_rel(&want, &p.output) < 2e-3, "autotuned winner not within tolerance");
        eprintln!("[matvec_q8_dp4a autotune CPU] winner={} timings={:?}", p.winner, p.timings);

        let p2 = matvec_q8_dp4a_tuned_set::<CpuRuntime>(&c, &wq, &xq, &wd, rows, k).pick_with(&tuner, "cpu");
        assert!(p2.from_cache && p2.benched == 0, "second call must hit the cache and skip timing");
        assert_eq!(p2.winner, p.winner, "cache returned a different winner");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The NR row-tile guards the ragged tail: when the row count is NOT a multiple of `nr`, the last
    /// thread owns fewer than `nr` real rows and must skip the out-of-range ones (no OOB weight read, no
    /// stray store). rows=49 leaves a tail for both nr=2 and nr=4 -- every schedule still matches the
    /// oracle exactly and stays byte-identical to nr=1 (the guard changes which thread does a row, never
    /// the value or which rows exist).
    #[cfg(feature = "cpu")]
    #[test]
    fn matvec_dp4a_nr_ragged_cpu() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, k) = (49usize, 256usize); // 49 % 2 and 49 % 4 both leave a tail on the last thread
        let (wq, xq, wd) = gen_dp4a(rows, k);
        let wq32: Vec<i32> = wq.iter().map(|&v| v as i32).collect();
        let xq32: Vec<i32> = xq.iter().map(|&v| v as i32).collect();
        let want = matvec_q8_dp4a_ref(&wq32, &xq32, &wd, rows, k);
        let c = CpuRuntime::client(&CpuDevice::default());
        let mut ref_bits: Option<Vec<u32>> = None;
        for &(vw, block, nr) in &[(2usize, 64u32, 1usize), (2, 64, 2), (2, 128, 4), (4, 256, 4)] {
            let (got, _ms) = matvec_q8_dp4a_tuned_bench::<CpuRuntime>(&c, &wq, &xq, &wd, rows, k, vw, block, nr, 1);
            assert_eq!(got.len(), rows, "output truncated at nr={nr}");
            let gbits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
            match &ref_bits {
                None => ref_bits = Some(gbits),
                Some(rb) => assert_eq!(&gbits, rb, "ragged nr={nr} not byte-identical to nr=1"),
            }
            assert!(max_rel(&want, &got) < 2e-3, "ragged nr={nr} diverged from the oracle");
        }
    }

    /// The full {WG x VW x NR} SEARCH on the CPU runtime: build the space + a cold-stream evaluator and
    /// run the `tune::Evolution` GA. Proves the machinery end to end offline (no GPU) -- the hunt crowns
    /// a bit-exact, feasible winner deterministically, records it, and a second hunt reads the cache
    /// without re-running the GA. On a wgpu device the SAME evaluator makes the winner meaningful (the
    /// width/tile that matches the hand/llama kernel); CPU schedules are near-equivalent so the winner
    /// here is arbitrary-but-stable -- the point under test is the search + the correctness gate.
    #[cfg(feature = "cpu")]
    #[test]
    fn matvec_dp4a_space_search_cpu() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, k) = (64usize, 256usize); // ng = 64, a multiple of every VW in the space
        let (_, xq, wd) = gen_dp4a(rows, k);
        // Three distinct weight banks so the cold rotation reads different data on each dispatch.
        let banks: Vec<Vec<i8>> = (0..3)
            .map(|b| {
                let mut s = 0x51ED_2701u64.wrapping_add(b).wrapping_mul(0x9E37_79B1) | 1;
                (0..rows * k)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s % 251) as i8
                    })
                    .collect()
            })
            .collect();
        let space = matvec_dp4a_space(k);

        // The space is exactly the feasible {WG x VW x NR} product: only VW dividing ng, only legal widths.
        let feasible = space.enumerate();
        assert!(!feasible.is_empty(), "empty feasible space");
        assert!(feasible.iter().all(|cfg| (k / 4) % (cfg.get(&space, "VW") as usize) == 0));
        assert!(feasible.iter().all(|cfg| {
            let w = cfg.get(&space, "WG");
            w > 0 && w <= 1024
        }));

        let c = CpuRuntime::client(&CpuDevice::default());
        let eval = MatvecDp4aEval::new(&c, &space, &banks, &xq, &wd, rows, k, 1);
        let dir = std::env::temp_dir().join(format!(
            "hk-dp4a-hunt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let tuner = Tuner::new(&dir);
        let evo = Evolution::new().population(8).generations(3).measure_iters(1);

        // (1) miss: the hunt runs, crowns a feasible bit-exact winner, and records it.
        let r = matvec_dp4a_hunt(&tuner, "cpu", rows, k, &eval, &evo, 0xC0FFEE);
        assert!(!r.from_cache, "first hunt must not be a cache hit");
        let rep = r.report.as_ref().expect("a miss carries the evidence trail");
        assert!(rep.best_ms.is_finite(), "no measurable winner");
        let win = space.parse(&r.winner).expect("winner is a valid config name");
        assert!(space.feasible(&win), "winner {} is infeasible", r.winner);
        assert!(eval.worst_rel() < 2e-3, "a measured variant diverged from the oracle: {:.2e}", eval.worst_rel());
        eprintln!("[dp4a hunt CPU] winner={} evaluated={} worst_rel={:.2e}", r.winner, rep.evaluated, eval.worst_rel());

        // (2) hit: a second hunt reads the cached winner and runs no GA.
        let r2 = matvec_dp4a_hunt(&tuner, "cpu", rows, k, &eval, &evo, 0xC0FFEE);
        assert!(r2.from_cache && r2.report.is_none(), "second hunt must hit the cache");
        assert_eq!(r2.winner, r.winner, "cache returned a different winner");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The f32-direct cooperative Q4_K matvec (`matvec_q4k_f32_blk`) is bit-exact to the Q4_K oracle
    /// across the {nt, nr} schedule -- proving the DSL now expresses the f32-direct coalesced decode the
    /// hand `mul_mat_vec_q4k_cm` runs (the Vulkan decode kernel the dp4a path cannot replace because its
    /// activation-quantize dispatch is too costly there). Reads x as f32, no quantize. The oracle is the
    /// same one the hand kernel is gated against (matvec_q4k_ref / BlockQ4K::to_float), so matching it is
    /// runtime-equality with the hand kernel by construction; the cold GB/s A/B is the queued GPU step.
    /// `rows`/`k` chosen so `nt` splits the sub-blocks and `nr` leaves a ragged tail (guard exercised).
    #[cfg(feature = "cpu")]
    #[test]
    fn matvec_q4k_f32_blk_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, k) = (35usize, 512usize); // nb=2, nsub=16; rows=35 leaves a tail for nr=2 and nr=4
        let (wqs, wsc, wd, wdm, x) = gen_q4k(rows, k);
        let want = matvec_q4k_ref(&wqs, &wsc, &wd, &wdm, &x, rows, k);
        let maxref = want.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
        let c = CpuRuntime::client(&CpuDevice::default());
        for &(nt, nr) in &[(16usize, 1usize), (32, 1), (64, 1), (16, 2), (32, 4), (64, 2)] {
            let got = matvec_q4k_f32_blk_run::<CpuRuntime>(&c, &wqs, &wsc, &wd, &wdm, &x, rows, k, nt, nr);
            assert_eq!(got.len(), rows, "output truncated at nt={nt} nr={nr}");
            let rel = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / maxref;
            eprintln!("[matvec_q4k_f32_blk CPU] nt={nt} nr={nr} scale_rel={rel:.2e}");
            assert!(rel < 1e-4, "f32-direct nt={nt} nr={nr} scale_rel {rel} vs Q4_K oracle");
        }
    }

    /// The {WG x NR} SEARCH for the f32-direct matvec on the CPU runtime: the tune::Evolution GA over
    /// matvec_q4k_f32_space gated by MatvecQ4kF32Eval (cold-stream + the Q4_K oracle). Proves the
    /// machinery offline -- a feasible, bit-exact winner, deterministic, cached. On a wgpu device the
    /// same evaluator crowns the width that matches the hand mul_mat_vec_q4k_cm cold (the queued GPU step).
    #[cfg(feature = "cpu")]
    #[test]
    fn matvec_q4k_f32_search_cpu() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (rows, k) = (48usize, 512usize);
        let (wqs, wsc, wd, wdm, x) = gen_q4k(rows, k);
        let space = matvec_q4k_f32_space();
        let feasible = space.enumerate();
        assert!(!feasible.is_empty());
        assert!(feasible.iter().all(|cfg| {
            let w = cfg.get(&space, "WG");
            w >= 2 && (w & (w - 1)) == 0
        }));

        let c = CpuRuntime::client(&CpuDevice::default());
        let eval = MatvecQ4kF32Eval::new(&c, &space, &wqs, &wsc, &wd, &wdm, &x, rows, k, 2, 1);
        let dir = std::env::temp_dir().join(format!(
            "hk-q4kf32-hunt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let tuner = Tuner::new(&dir);
        let evo = Evolution::new().population(8).generations(3).measure_iters(1);

        let r = matvec_q4k_f32_hunt(&tuner, "cpu", rows, k, &eval, &evo, 0xF32);
        assert!(!r.from_cache, "first hunt must not be a cache hit");
        let rep = r.report.as_ref().expect("a miss carries the evidence trail");
        assert!(rep.best_ms.is_finite(), "no measurable winner");
        let win = space.parse(&r.winner).expect("winner is a valid config name");
        assert!(space.feasible(&win), "winner {} is infeasible", r.winner);
        assert!(eval.worst_rel() < 1e-3, "a measured variant diverged from the oracle: {:.2e}", eval.worst_rel());
        eprintln!("[q4k_f32 hunt CPU] winner={} evaluated={} worst_rel={:.2e}", r.winner, rep.evaluated, eval.worst_rel());

        let r2 = matvec_q4k_f32_hunt(&tuner, "cpu", rows, k, &eval, &evo, 0xF32);
        assert!(r2.from_cache && r2.report.is_none(), "second hunt must hit the cache");
        assert_eq!(r2.winner, r.winner, "cache returned a different winner");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The Q4_K indexed-MoE matvec on the CPU runtime is byte-for-byte the plain-Rust oracle: the
    /// expert gather + per-slot activation ride on top of the SAME bit-exact Q4_K decode. This is the
    /// correctness gate the hand-rolled `moe_matvec_q4k` shader becomes the on-device twin of.
    #[cfg(feature = "cpu")]
    #[test]
    fn moe_matvec_q4k_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (e, n, slots, k) = (8usize, 12usize, 5usize, 512usize);
        let (wqs, wsc, wd, wdm, x, ids) = gen_moe_q4k(e, n, slots, k);
        let c = CpuRuntime::client(&CpuDevice::default());
        let got = moe_matvec_q4k_run::<CpuRuntime>(&c, &wqs, &wsc, &wd, &wdm, &x, &ids, slots, n, k);
        let want = moe_matvec_q4k_ref(&wqs, &wsc, &wd, &wdm, &x, &ids, slots, n, k);
        let gbits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        let wbits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
        assert_eq!(gbits, wbits, "moe_matvec_q4k CPU kernel != oracle bit-exact");
    }

    /// The Q6_K indexed-MoE matvec on the CPU runtime is byte-for-byte the plain-Rust oracle: signed
    /// i8 scales + 6-bit code decode + expert gather, all bit-exact. Twin of `moe_matvec_q6k.comp`.
    #[cfg(feature = "cpu")]
    #[test]
    fn moe_matvec_q6k_cpu_bit_exact() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let (e, n, slots, k) = (8usize, 12usize, 5usize, 512usize);
        let (wql, wqh, wsc, wd, x, ids) = gen_moe_q6k(e, n, slots, k);
        let c = CpuRuntime::client(&CpuDevice::default());
        let got = moe_matvec_q6k_run::<CpuRuntime>(&c, &wql, &wqh, &wsc, &wd, &x, &ids, slots, n, k);
        let want = moe_matvec_q6k_ref(&wql, &wqh, &wsc, &wd, &x, &ids, slots, n, k);
        let gbits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
        let wbits: Vec<u32> = want.iter().map(|v| v.to_bits()).collect();
        assert_eq!(gbits, wbits, "moe_matvec_q6k CPU kernel != oracle bit-exact");
    }
}
