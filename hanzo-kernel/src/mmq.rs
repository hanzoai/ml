//! int8 tensor-core MMQ GEMM in the DSL -- the prefill lever, written ONCE.
//!
//! The universal prefill gap (dense 0.6-0.9x vs llama.cpp on every backend) is a tensor-core-MMQ gap:
//! llama quantizes the activation to q8_1 (int8) and contracts Q8_0/Q4_K weights against it on the
//! tensor cores. This module proves whether that same kernel can be expressed ONCE in `hanzo_kernel`
//! and lowered to the hardware int8 matrix cores, instead of hand-written four times (CUDA fast_mmq,
//! Vulkan mul_mm_q8_mmq, ROCm qmmq, Metal simdgroup).
//!
//! Two DSL routes to the int8 matrix cores, both registered by CubeCL on CUDA sm>=70/80:
//!   - high-level WMMA (`cmma::Matrix`, i8->i32 at 16x16x16): auto fragment load/store, one tile at a
//!     time. Ergonomic; the fragment layout is opaque so the per-block dequant-scale is applied via a
//!     shared-memory round-trip in the epilogue.
//!   - manual MMA (`cmma::MmaDefinition`, i8->i32 at m16n8k32): `mma.sync` with explicit registers.
//!     The fragment element<->lane map IS exposed (`position_of_nth`), so scales could be applied in
//!     register -- the higher-ceiling path. Included here as the proven-by-CubeCL int8 probe.
//!
//! MMQ math (Q8_0 weight x q8_1 activation), the exact contraction llama does:
//!   out[m,n] = sum over 32-blocks kb of  (xs[m,kb] * wd[n,kb]) * sum_{k in block} xq[m,k]*wq[n,k]
//! the inner int8 dot is the tensor-core op; the per-block f32 scale is the epilogue. int8 accumulation
//! is only valid WITHIN one constant-scale 32-block, so the scale is applied once per block -- inherent
//! to block-quantized GEMM, not a kernel choice.

use crate::prelude::*;

// ================================================================================================
// STAGE 1a -- high-level WMMA hello: 16x16x16 i8 -> i32, out = A @ B^T, no scales.
// The smallest kernel that proves the DSL can emit an int8 tensor-core op that JITs + runs on the GPU.
// ================================================================================================

/// out[16,16] = A[16,16] @ B[16,16]^T, int8 inputs, i32 accumulate. Single warp, one WMMA tile.
#[kernel(targets(cuda), unchecked)]
pub fn wmma_hello_i8(a: &Array<i8>, b: &Array<i8>, out: &mut Array<i32>) {
    let ma = cmma::Matrix::<i8>::from_slice(
        cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor, &a.to_slice(), 16,
    );
    let mb = cmma::Matrix::<i8>::from_slice(
        cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor, &b.to_slice(), 16,
    );
    let mc = cmma::Matrix::<i32>::from_value(
        cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0i32,
    );
    cmma::execute::<i8, i8, i32, i32>(&ma, &mb, &mc, &mc);
    cmma::store(&mut out.to_slice_mut(), &mc, 16, cmma::MatrixLayout::RowMajor);
}

/// Run the WMMA hello on a runtime; returns the 256 i32 outputs.
pub fn wmma_hello_i8_run<R: Runtime>(client: &ComputeClient<R>, a: &[i8], b: &[i8]) -> Vec<i32> {
    let ah = client.create_from_slice(i8::as_bytes(a));
    let bh = client.create_from_slice(i8::as_bytes(b));
    let oh = client.create_from_slice(i32::as_bytes(&vec![0i32; 256]));
    unsafe {
        wmma_hello_i8::launch_unchecked::<R>(
            client, Grid::Static(1, 1, 1), Block::new_1d(32),
            ArrayArg::from_raw_parts(ah.clone(), 256),
            ArrayArg::from_raw_parts(bh.clone(), 256),
            ArrayArg::from_raw_parts(oh.clone(), 256),
        );
    }
    i32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// CPU oracle: out = A @ B^T over int8, exact i32 (the WMMA result must match bit-for-bit).
pub fn hello_i8_ref(a: &[i8], b: &[i8]) -> Vec<i32> {
    let mut out = vec![0i32; 256];
    for m in 0..16 {
        for n in 0..16 {
            let mut acc = 0i32;
            for k in 0..16 {
                acc += a[m * 16 + k] as i32 * b[n * 16 + k] as i32; // B^T: B row n = B[n,k]
            }
            out[m * 16 + n] = acc;
        }
    }
    out
}

// ================================================================================================
// STAGE 1b -- manual MMA hello: m16n8k32 i8 -> i32 (the CubeCL-tested int8 tensor-core path).
// mma.sync with explicit register staging; the element<->lane map is exposed via position_of_nth.
// Faithful reduction of CubeCL's own `kernel_manual` to i8/i8/i32, Array (contiguous) inputs.
// ================================================================================================

/// out[16,8] = A[16,32] @ B[32,8] over int8, i32 accumulate, via one `mma.sync.m16n8k32.s8`.
#[kernel(targets(cuda), unchecked)]
pub fn mma_hello_i8(
    a: &Array<i8>,   // [16,32] row-major
    b: &Array<i8>,   // [32,8]  row-major
    out: &mut Array<i32>, // [16,8] row-major
    #[comptime] size_m: usize,
    #[comptime] size_n: usize,
    #[comptime] size_k: usize,
) {
    let def = cmma::MmaDefinition::<i8, i8, i32>::new(size_m, size_n, size_k);
    let lane_id = UNIT_POS_PLANE;

    let vector_size_a = def.vector_size(cmma::MatrixIdent::A);
    let size!(NA) = vector_size_a;
    let vector_count_a = def.vectors_per_lane(cmma::MatrixIdent::A);
    let mut registers_a = Array::<Vector<i8, NA>>::new(vector_count_a);

    let vector_size_b = def.vector_size(cmma::MatrixIdent::B);
    let size!(NB) = vector_size_b;
    let vector_count_b = def.vectors_per_lane(cmma::MatrixIdent::B);
    let mut registers_b = Array::<Vector<i8, NB>>::new(vector_count_b);

    let vector_size_c = def.vector_size(cmma::MatrixIdent::Accumulator);
    let size!(NC) = vector_size_c;
    let vector_count_c = def.vectors_per_lane(cmma::MatrixIdent::Accumulator);
    let mut registers_c = Array::<Vector<i32, NC>>::new(vector_count_c);

    // Load A registers.
    #[unroll]
    for i in 0..vector_count_a {
        let mut reg = Vector::<i8, NA>::empty();
        #[unroll]
        for kk in 0..vector_size_a {
            let n_elem = i * vector_size_a + kk;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, cmma::MatrixIdent::A);
            reg[kk] = a[(row * size_k as u32 + col) as usize];
        }
        registers_a[i] = reg;
    }
    // Load B registers.
    #[unroll]
    for i in 0..vector_count_b {
        let mut reg = Vector::<i8, NB>::empty();
        #[unroll]
        for kk in 0..vector_size_b {
            let n_elem = i * vector_size_b + kk;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, cmma::MatrixIdent::B);
            reg[kk] = b[(row * size_n as u32 + col) as usize];
        }
        registers_b[i] = reg;
    }
    // Zero C.
    #[unroll]
    for i in 0..vector_count_c {
        let mut reg = Vector::<i32, NC>::empty();
        #[unroll]
        for kk in 0..vector_size_c {
            reg[kk] = 0i32;
        }
        registers_c[i] = reg;
    }

    let registers_d = def.execute(&registers_a, &registers_b, &registers_c);

    // Store D.
    #[unroll]
    for i in 0..vector_count_c {
        let reg = registers_d[i];
        #[unroll]
        for kk in 0..vector_size_c {
            let n_elem = i * vector_size_c + kk;
            let (row, col) = def.position_of_nth(lane_id, n_elem as u32, cmma::MatrixIdent::Accumulator);
            out[(row * size_n as u32 + col) as usize] = reg[kk];
        }
    }
}

/// Run the manual MMA hello (m16n8k32). Returns 128 i32 (16x8).
pub fn mma_hello_i8_run<R: Runtime>(client: &ComputeClient<R>, a: &[i8], b: &[i8], plane: u32) -> Vec<i32> {
    let ah = client.create_from_slice(i8::as_bytes(a));
    let bh = client.create_from_slice(i8::as_bytes(b));
    let oh = client.create_from_slice(i32::as_bytes(&vec![0i32; 16 * 8]));
    unsafe {
        mma_hello_i8::launch_unchecked::<R>(
            client, Grid::Static(1, 1, 1), Block::new_1d(plane),
            ArrayArg::from_raw_parts(ah.clone(), 16 * 32),
            ArrayArg::from_raw_parts(bh.clone(), 32 * 8),
            ArrayArg::from_raw_parts(oh.clone(), 16 * 8),
            16, 8, 32,
        );
    }
    i32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// CPU oracle for m16n8k32 A@B, exact i32.
pub fn mma_hello_ref(a: &[i8], b: &[i8]) -> Vec<i32> {
    let (m, n, k) = (16usize, 8usize, 32usize);
    let mut out = vec![0i32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0i32;
            for l in 0..k {
                acc += a[i * k + l] as i32 * b[l * n + j] as i32;
            }
            out[i * n + j] = acc;
        }
    }
    out
}

// ================================================================================================
// STAGE 2 + 3 -- Q8_0 x q8_1 MMQ GEMM via high-level WMMA. out[M,N] = X @ W^T, block-scaled.
// X: int8 [M,K] + f32 per-32-block scale xs [M,K/32] (q8_1 activation).
// W: int8 [N,K] + f32 per-32-block scale wd [N,K/32] (Q8_0 weight, host-decoded f16->f32 scale).
// One warp owns a 16x16 output tile; over each 32-wide K-block it does 2 WMMA (K=16) into an i32
// fragment, stores the i32 tile to shared memory, then applies xs*wd per element into an f32 tile.
// ================================================================================================

/// int8 tensor-core MMQ GEMM. Grid = (N/16, M/16); block = one warp (32 lanes).
#[kernel(targets(cuda), unchecked)]
pub fn mmq_q8_wmma(
    xq: &Array<i8>,
    xs: &Array<f32>,
    wq: &Array<i8>,
    wd: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
) {
    let nt = CUBE_POS_X as usize; // output column-tile (16 cols of N)
    let mt = CUBE_POS_Y as usize; // output row-tile (16 rows of M)
    let lane = UNIT_POS as usize; // 0..31
    let kb_count = k / 32;
    let mrow0 = mt * 16;
    let ncol0 = nt * 16;

    let mut accf = SharedMemory::<f32>::new(256usize); // f32 output tile [16x16]
    let mut ci = SharedMemory::<i32>::new(256usize);   // i32 fragment store scratch [16x16]

    // Zero the f32 accumulator (each lane owns 8 of 256).
    #[unroll]
    for e in 0usize..8 {
        accf[lane * 8 + e] = 0.0f32;
    }
    sync_cube();

    for kb in 0..kb_count {
        let k0 = kb * 32;
        // Accumulate int8 over the 32-wide block into one i32 fragment (2 WMMA K=16 steps).
        let c = cmma::Matrix::<i32>::from_value(
            cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0i32,
        );
        let a0 = cmma::Matrix::<i8>::from_slice(
            cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
            &xq.slice(mrow0 * k + k0, m * k), k as u32,
        );
        let b0 = cmma::Matrix::<i8>::from_slice(
            cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
            &wq.slice(ncol0 * k + k0, n * k), k as u32,
        );
        cmma::execute::<i8, i8, i32, i32>(&a0, &b0, &c, &c);

        let a1 = cmma::Matrix::<i8>::from_slice(
            cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
            &xq.slice(mrow0 * k + k0 + 16, m * k), k as u32,
        );
        let b1 = cmma::Matrix::<i8>::from_slice(
            cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
            &wq.slice(ncol0 * k + k0 + 16, n * k), k as u32,
        );
        cmma::execute::<i8, i8, i32, i32>(&a1, &b1, &c, &c);

        // Spill the i32 tile to shared memory, then apply the per-block f32 scale in the epilogue.
        cmma::store(&mut ci.to_slice_mut(), &c, 16, cmma::MatrixLayout::RowMajor);
        sync_cube();
        #[unroll]
        for e in 0usize..8 {
            let idx = lane * 8 + e;
            let mm = idx / 16;
            let nn = idx % 16;
            let xsc = xs[(mrow0 + mm) * kb_count + kb];
            let wsc = wd[(ncol0 + nn) * kb_count + kb];
            accf[idx] += f32::cast_from(ci[idx]) * xsc * wsc;
        }
        sync_cube();
    }

    #[unroll]
    for e in 0usize..8 {
        let idx = lane * 8 + e;
        let mm = idx / 16;
        let nn = idx % 16;
        out[(mrow0 + mm) * n + (ncol0 + nn)] = accf[idx];
    }
}

/// Host launch for the MMQ GEMM. `iters` amortized kernel-only bench dispatches; returns (out, ms).
pub fn mmq_q8_wmma_run<R: Runtime>(
    client: &ComputeClient<R>,
    xq: &[i8], xs: &[f32], wq: &[i8], wd: &[f32],
    m: usize, n: usize, k: usize, iters: usize,
) -> (Vec<f32>, f64) {
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let xsh = client.create_from_slice(f32::as_bytes(xs));
    let wqh = client.create_from_slice(i8::as_bytes(wq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
    let grid = Grid::Static((n / 16) as u32, (m / 16) as u32, 1);
    let launch = |c: &ComputeClient<R>| unsafe {
        mmq_q8_wmma::launch_unchecked::<R>(
            c, grid.clone(), Block::new_1d(32),
            ArrayArg::from_raw_parts(xqh.clone(), xq.len()),
            ArrayArg::from_raw_parts(xsh.clone(), xs.len()),
            ArrayArg::from_raw_parts(wqh.clone(), wq.len()),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), m * n),
            m, n, k,
        );
    };
    launch(client);
    let out = f32::from_bytes(&client.read_one_unchecked(oh.clone())).to_vec();
    for _ in 0..3 { launch(client); }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters { launch(client); }
    let _ = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    (out, ms)
}

/// CPU oracle: the exact MMQ math -- int8 dot per 32-block, f32 per-block scale, summed in f32.
/// The DSL kernel must match this to f32-reorder precision (this IS the quantized reference).
pub fn mmq_q8_ref(xq: &[i8], xs: &[f32], wq: &[i8], wd: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let kb = k / 32;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for b in 0..kb {
                let mut isum = 0i32;
                for l in 0..32 {
                    isum += xq[i * k + b * 32 + l] as i32 * wq[j * k + b * 32 + l] as i32;
                }
                acc += isum as f32 * xs[i * kb + b] * wd[j * kb + b];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Deterministic MMQ test data: random int8 X/W in [-127,127], positive f32 block scales.
pub fn gen_mmq(m: usize, n: usize, k: usize) -> (Vec<i8>, Vec<f32>, Vec<i8>, Vec<f32>) {
    let kb = k / 32;
    let mut s = 0xD1B54A32D192ED03u64;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        s
    };
    let xq: Vec<i8> = (0..m * k).map(|_| ((next() % 255) as i64 - 127) as i8).collect();
    let wq: Vec<i8> = (0..n * k).map(|_| ((next() % 255) as i64 - 127) as i8).collect();
    // q8_1 / Q8_0 scales are amax/127-ish small positives.
    let xs: Vec<f32> = (0..m * kb).map(|_| (next() % 1000) as f32 / 50000.0 + 0.002).collect();
    let wd: Vec<f32> = (0..n * kb).map(|_| (next() % 1000) as f32 / 50000.0 + 0.002).collect();
    (xq, xs, wq, wd)
}
