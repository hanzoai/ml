//! int8 tensor-core MMQ GEMM in the DSL -- the prefill lever, written ONCE.
//!
//! The universal prefill gap (dense 0.6-0.9x vs llama.cpp on every backend) is a tensor-core-MMQ gap:
//! llama quantizes the activation to q8_1 (int8) and contracts Q8_0/Q4_K weights against it on the
//! tensor cores. This module proves whether that same kernel can be expressed ONCE in `hanzo_kernel`
//! and lowered to the hardware int8 matrix cores, instead of hand-written four times (CUDA fast_mmq,
//! Vulkan mul_mm_q8_mmq, ROCm qmmq, Metal simdgroup).
//!
//! Two DSL routes to the int8 matrix cores, and they do NOT have the same reach:
//!   - high-level WMMA (`cmma::Matrix`, i8->i32 at 16x16x16): auto fragment load/store, one tile at a
//!     time. Ergonomic; the fragment layout is opaque so the per-block dequant-scale is applied via a
//!     shared-memory round-trip in the epilogue. PORTABLE -- lowers to WMMA on CUDA/ROCm, to
//!     `OpCooperativeMatrixMulAddKHR` (signed-component operands) on SPIR-V, and to simdgroup matrix on
//!     Metal. This is the prefill path.
//!   - manual MMA (`cmma::MmaDefinition`, i8->i32 at m16n8k32): `mma.sync` with explicit registers.
//!     The fragment element<->lane map IS exposed (`position_of_nth`), so scales could be applied in
//!     register -- the higher ceiling, but CUDA/ROCm ONLY: SPIR-V's cooperative matrix is an opaque
//!     type with no queryable per-lane layout, so cubecl-spirv rejects the whole manual family
//!     (see `mma_hello_i8`). Kept as the tuned peak path, not the portable one.
//!
//! Portability is carried by ONE island per GEMM, not by a second kernel: the accelerated arm issues
//! cmma, the `default` arm expresses the identical int8 contraction as scalar MACs. `default` is what
//! the CPU runtime runs (cubecl-cpu rejects every CoopMma op), which is what makes a bit-exact CPU
//! oracle possible at all -- it gates the tiling, the shared-memory epilogue and the f32 accumulation
//! order. The cmma arm itself is equivalent by construction (int8 accumulation is exact) and is gated
//! on-GPU against the same `mmq_q8_ref`.
//!
//! MMQ math (Q8_0 weight x q8_1 activation), the exact contraction llama does:
//!   out[m,n] = sum over 32-blocks kb of  (xs[m,kb] * wd[n,kb]) * sum_{k in block} xq[m,k]*wq[n,k]
//! the inner int8 dot is the tensor-core op; the per-block f32 scale is the epilogue. int8 accumulation
//! is only valid WITHIN one constant-scale 32-block, so the scale is applied once per block -- inherent
//! to block-quantized GEMM, not a kernel choice.

use crate::prelude::*;
use crate::tune::{Config, Evaluator, Evolution, Evolved, Space, Tuner, Verdict};
use cubecl::server::Handle;

// ================================================================================================
// STAGE 1a -- high-level WMMA hello: 16x16x16 i8 -> i32, out = A @ B^T, no scales.
// The smallest kernel that proves the DSL can emit an int8 tensor-core op that JITs + runs on the GPU.
// ================================================================================================

/// out[16,16] = A[16,16] @ B[16,16]^T, int8 inputs, i32 accumulate. Single warp, one WMMA tile.
///
/// No island: the kernel IS the cmma probe, so there is no meaningful non-cmma arm to fall back to.
/// It therefore has no CPU oracle (cubecl-cpu rejects every CoopMma op) and is proven on-GPU against
/// [`hello_i8_ref`]. The MMQ GEMMs below carry the oracle; this only answers "does the op JIT and run".
#[kernel(targets(cuda, rocm, vulkan, metal), unchecked)]
pub fn wmma_hello_i8(a: &Array<i8>, b: &Array<i8>, out: &mut Array<i32>) {
    let ma = cmma::Matrix::<i8>::from_slice(
        cmma::MatrixIdent::A,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::RowMajor,
        &a.to_slice(),
        16,
    );
    let mb = cmma::Matrix::<i8>::from_slice(
        cmma::MatrixIdent::B,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::ColMajor,
        &b.to_slice(),
        16,
    );
    let mc = cmma::Matrix::<i32>::from_value(
        cmma::MatrixIdent::Accumulator,
        16usize,
        16usize,
        16usize,
        cmma::MatrixLayout::Undefined,
        0i32,
    );
    cmma::execute::<i8, i8, i32, i32>(&ma, &mb, &mc, &mc);
    cmma::store(
        &mut out.to_slice_mut(),
        &mc,
        16,
        cmma::MatrixLayout::RowMajor,
    );
}

/// Run the WMMA hello on a runtime; returns the 256 i32 outputs.
///
/// The block is ONE PLANE wide, queried from the runtime -- not a literal 32. A cooperative-matrix op
/// is plane-scoped: the fragment's 256 elements are distributed across every lane of the plane, so a
/// block narrower than the plane leaves the absent lanes' elements unwritten. The plane is 32 lanes on
/// CUDA but 64 on RADV, which is why a hardcoded 32 silently produces a half-written tile there.
pub fn wmma_hello_i8_run<R: Runtime>(client: &ComputeClient<R>, a: &[i8], b: &[i8]) -> Vec<i32> {
    let plane = client.properties().hardware.plane_size_max;
    let ah = client.create_from_slice(i8::as_bytes(a));
    let bh = client.create_from_slice(i8::as_bytes(b));
    let oh = client.create_from_slice(i32::as_bytes(&vec![0i32; 256]));
    unsafe {
        wmma_hello_i8::launch_unchecked::<R>(
            client,
            Grid::Static(1, 1, 1),
            Block::new_1d(plane),
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
///
/// NOT portable to Vulkan, and not by omission: manual MMA needs the fragment element<->lane map, and
/// cubecl-spirv rejects every op that exposes it -- `RowIndex`/`ColIndex` (what `position_of_nth`
/// emits), `LoadMatrix`/`StoreMatrix`, `ExecuteManual`, `ExecuteScaled` all hit one
/// `panic!("Manual register management not currently supported in SPIR-V")`
/// (cubecl-spirv-0.10.0/src/cmma.rs:37-44). SPIR-V's CooperativeMatrixKHR type is opaque by design:
/// it has no defined per-lane layout to query, so this is a spec-level gap, not a missing match arm.
/// The cubecl-cpp backends do implement it (cuda/dialect.rs, hip/dialect.rs:171, metal/dialect.rs:1219),
/// hence cuda + rocm. The portable prefill path is the high-level WMMA route below.
#[kernel(targets(cuda, rocm), unchecked)]
pub fn mma_hello_i8(
    a: &Array<i8>,        // [16,32] row-major
    b: &Array<i8>,        // [32,8]  row-major
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
            let (row, col) =
                def.position_of_nth(lane_id, n_elem as u32, cmma::MatrixIdent::Accumulator);
            out[(row * size_n as u32 + col) as usize] = reg[kk];
        }
    }
}

/// Run the manual MMA hello (m16n8k32). Returns 128 i32 (16x8).
pub fn mma_hello_i8_run<R: Runtime>(
    client: &ComputeClient<R>,
    a: &[i8],
    b: &[i8],
    plane: u32,
) -> Vec<i32> {
    let ah = client.create_from_slice(i8::as_bytes(a));
    let bh = client.create_from_slice(i8::as_bytes(b));
    let oh = client.create_from_slice(i32::as_bytes(&vec![0i32; 16 * 8]));
    unsafe {
        mma_hello_i8::launch_unchecked::<R>(
            client,
            Grid::Static(1, 1, 1),
            Block::new_1d(plane),
            ArrayArg::from_raw_parts(ah.clone(), 16 * 32),
            ArrayArg::from_raw_parts(bh.clone(), 32 * 8),
            ArrayArg::from_raw_parts(oh.clone(), 16 * 8),
            16,
            8,
            32,
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

/// int8 tensor-core MMQ GEMM. Grid = (N/16, M/16); block = ONE PLANE.
///
/// `plane` is comptime because a cooperative-matrix op is plane-scoped: the 16x16 fragment's 256
/// elements are spread over every lane of the plane, so the block must be exactly one plane wide and
/// the scalar prologue/epilogue must partition 256 by that same width (8 each on a 32-lane CUDA warp,
/// 4 each on a 64-lane RADV wave). Hardcoding 32 leaves the absent lanes' elements unwritten.
#[kernel(targets(cuda, rocm, vulkan, metal, cpu), unchecked)]
pub fn mmq_q8_wmma(
    xq: &Array<i8>,
    xs: &Array<f32>,
    wq: &Array<i8>,
    wd: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] plane: usize,
    #[comptime] target: Target,
) {
    let nt = CUBE_POS_X as usize; // output column-tile (16 cols of N)
    let mt = CUBE_POS_Y as usize; // output row-tile (16 rows of M)
    let lane = UNIT_POS as usize; // 0..plane-1
    let kb_count = k / 32;
    let mrow0 = mt * 16;
    let ncol0 = nt * 16;
    let per = 256 / plane; // tile elements this lane owns in the scalar prologue/epilogue

    let mut accf = SharedMemory::<f32>::new(256usize); // f32 output tile [16x16]
    let mut ci = SharedMemory::<i32>::new(256usize); // i32 tile: the 32-block's int8 contraction

    // Zero the f32 accumulator (each lane owns `per` of 256).
    #[unroll]
    for e in 0usize..per {
        accf[lane * per + e] = 0.0f32;
    }
    sync_cube();

    for kb in 0..kb_count {
        let k0 = kb * 32;
        // The 32-wide K-block's int8 contraction into the i32 tile `ci`. int8 accumulation is exact,
        // so both arms are the SAME integer value -- only the idiom differs.
        island! {
            // Accelerated: the hardware int8 matrix core, 2 x cmma(16x16x16) i8->i32. Lowers to WMMA
            // (CUDA/ROCm), OpCooperativeMatrixMulAddKHR with signed-component operands (SPIR-V), and
            // simdgroup matrix (Metal). The fragment layout is opaque, so the tile is spilled to
            // shared memory for the scale epilogue below.
            cuda | rocm | vulkan | metal => {
                let c = cmma::Matrix::<i32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::Undefined,
                    0i32,
                );
                let a0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::RowMajor,
                    &xq.slice(mrow0 * k + k0, m * k),
                    k as u32,
                );
                let b0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::ColMajor,
                    &wq.slice(ncol0 * k + k0, n * k),
                    k as u32,
                );
                cmma::execute::<i8, i8, i32, i32>(&a0, &b0, &c, &c);

                let a1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::RowMajor,
                    &xq.slice(mrow0 * k + k0 + 16, m * k),
                    k as u32,
                );
                let b1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::ColMajor,
                    &wq.slice(ncol0 * k + k0 + 16, n * k),
                    k as u32,
                );
                cmma::execute::<i8, i8, i32, i32>(&a1, &b1, &c, &c);

                cmma::store(&mut ci.to_slice_mut(), &c, 16, cmma::MatrixLayout::RowMajor);
            }
            // NORMATIVE oracle: the identical i32 contraction as portable scalar MACs, each lane
            // owning 8 of the tile's 256 elements. This arm carries no cmma op, so it is the arm the
            // CPU runtime executes -- and thus the bit-exact gate for the whole kernel's structure
            // (tiling, shared-memory epilogue, f32 accumulation order).
            default => {
                #[unroll]
                for e in 0usize..per {
                    let idx = lane * per + e;
                    let mm = idx / 16;
                    let nn = idx % 16;
                    let mut isum = 0i32;
                    for l in 0usize..32 {
                        isum += i32::cast_from(xq[(mrow0 + mm) * k + k0 + l])
                            * i32::cast_from(wq[(ncol0 + nn) * k + k0 + l]);
                    }
                    ci[idx] = isum;
                }
            }
        };
        sync_cube();

        // Epilogue: apply the per-block f32 scale. Shared by both arms -- the scale is a property of
        // block-quantized GEMM, not of the idiom that produced the int8 sum.
        #[unroll]
        for e in 0usize..per {
            let idx = lane * per + e;
            let mm = idx / 16;
            let nn = idx % 16;
            let xsc = xs[(mrow0 + mm) * kb_count + kb];
            let wsc = wd[(ncol0 + nn) * kb_count + kb];
            accf[idx] += f32::cast_from(ci[idx]) * xsc * wsc;
        }
        sync_cube();
    }

    #[unroll]
    for e in 0usize..per {
        let idx = lane * per + e;
        let mm = idx / 16;
        let nn = idx % 16;
        out[(mrow0 + mm) * n + (ncol0 + nn)] = accf[idx];
    }
}

/// Host launch for the MMQ GEMM -- the PRODUCTION surface. Derives the island tag from the runtime and
/// hides it: callers pass only data. `iters` amortized kernel-only bench dispatches; returns (out, ms).
#[allow(clippy::too_many_arguments)]
pub fn mmq_q8_wmma_run<R: Runtime>(
    client: &ComputeClient<R>,
    xq: &[i8],
    xs: &[f32],
    wq: &[i8],
    wd: &[f32],
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> (Vec<f32>, f64) {
    mmq_q8_wmma_run_with(client, xq, xs, wq, wd, m, n, k, iters, Target::of(client))
}

/// Host launch with an explicit island tag. The oracle test pins [`Target::Cpu`] to run the normative
/// arm; production goes through [`mmq_q8_wmma_run`], which derives the tag from the runtime.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q8_wmma_run_with<R: Runtime>(
    client: &ComputeClient<R>,
    xq: &[i8],
    xs: &[f32],
    wq: &[i8],
    wd: &[f32],
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
    target: Target,
) -> (Vec<f32>, f64) {
    let plane = client.properties().hardware.plane_size_max;
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let xsh = client.create_from_slice(f32::as_bytes(xs));
    let wqh = client.create_from_slice(i8::as_bytes(wq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
    let grid = Grid::Static((n / 16) as u32, (m / 16) as u32, 1);
    let launch = |c: &ComputeClient<R>| unsafe {
        mmq_q8_wmma::launch_unchecked::<R>(
            c,
            grid.clone(),
            Block::new_1d(plane),
            ArrayArg::from_raw_parts(xqh.clone(), xq.len()),
            ArrayArg::from_raw_parts(xsh.clone(), xs.len()),
            ArrayArg::from_raw_parts(wqh.clone(), wq.len()),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), m * n),
            m,
            n,
            k,
            plane as usize,
            target,
        );
    };
    launch(client);
    let out = f32::from_bytes(&client.read_one_unchecked(oh.clone())).to_vec();
    for _ in 0..3 {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    (out, ms)
}

/// CPU oracle: the exact MMQ math -- int8 dot per 32-block, f32 per-block scale, summed in f32.
/// The DSL kernel must match this to f32-reorder precision (this IS the quantized reference).
pub fn mmq_q8_ref(
    xq: &[i8],
    xs: &[f32],
    wq: &[i8],
    wd: &[f32],
    m: usize,
    n: usize,
    k: usize,
) -> Vec<f32> {
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

// ================================================================================================
// STAGE 3 (tiled) -- the FAIR high-level-WMMA ceiling. 32x64 output tile per block, 8 warps, A/B
// staged to shared memory ONCE per 32-block and reused across the tile (A across 4 N-subtiles, B
// across 2 M-subtiles). This isolates "DSL/API limit" from "naive 1-warp structure": occupancy +
// operand reuse are fixed; the ONLY remaining WMMA-API tax is the per-block scale round-trip through
// the opaque i32 fragment (cmma::store -> shared -> scale). Requires M%32==0, N%64==0, K%32==0.
// ================================================================================================

/// Tiled int8 MMQ GEMM. Grid = (N/64, M/32); block = 8 PLANES (2x4 of 16x16 subtiles, one per plane).
///
/// `plane` is comptime for the reason given on [`mmq_q8_wmma`]: one plane cooperatively owns one 16x16
/// fragment, so the block is 8 planes wide -- 256 threads on a 32-lane CUDA warp, 512 on a 64-lane RADV
/// wave -- and every scalar partition below divides its buffer by that width rather than by a literal.
#[kernel(targets(cuda, rocm, vulkan, metal, cpu), unchecked)]
pub fn mmq_q8_wmma_blk(
    xq: &Array<i8>,
    xs: &Array<f32>,
    wq: &Array<i8>,
    wd: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] plane: usize,
    #[comptime] target: Target,
) {
    let tid = UNIT_POS as usize;
    let warp = tid / plane; // 0..7  (one plane per 16x16 subtile)
    let lane = tid % plane; // 0..plane-1
    let wm = warp / 4; // 0..1  (M-subtile)
    let wn = warp % 4; // 0..3  (N-subtile)
    let mrow0 = CUBE_POS_Y as usize * 32;
    let ncol0 = CUBE_POS_X as usize * 64;
    let kb_count = k / 32;
    let nthread = 8 * plane; // block width: 8 subtiles, one plane each
    let per = 256 / plane; // fragment elements this lane owns in the scalar epilogue

    let mut sa = SharedMemory::<i8>::new(1024usize); // A tile [32 x 32]
    let mut sb = SharedMemory::<i8>::new(2048usize); // B tile [64 x 32]
    let mut ci = SharedMemory::<i32>::new(2048usize); // per-warp i32 fragment scratch [8 x 256]
    let mut accf = SharedMemory::<f32>::new(2048usize); // f32 output tile [32 x 64]

    for e in 0usize..(2048 / nthread) {
        accf[tid * (2048 / nthread) + e] = 0.0f32;
    }
    sync_cube();

    for kb in 0..kb_count {
        let k0 = kb * 32;
        // Stage A[32x32] and B[64x32] int8 tiles into shared memory (coalesced, reused across warps).
        for i in 0usize..(1024 / nthread) {
            let idx = tid + i * nthread;
            sa[idx] = xq[(mrow0 + idx / 32) * k + k0 + idx % 32];
        }
        for i in 0usize..(2048 / nthread) {
            let idx = tid + i * nthread;
            sb[idx] = wq[(ncol0 + idx / 32) * k + k0 + idx % 32];
        }
        sync_cube();

        // This warp's 16x16 subtile: the staged 32-block's int8 contraction into `ci`. Both arms read
        // the SAME shared-memory A/B tiles, so operand staging is proven by the oracle too.
        island! {
            // Accelerated: 2 x cmma(16x16x16) i8->i32 straight out of shared memory.
            cuda | rocm | vulkan | metal => {
                let c = cmma::Matrix::<i32>::from_value(
                    cmma::MatrixIdent::Accumulator,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::Undefined,
                    0i32,
                );
                let a0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::RowMajor,
                    &sa.slice(wm * 512, 1024),
                    32,
                );
                let b0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::ColMajor,
                    &sb.slice(wn * 512, 2048),
                    32,
                );
                cmma::execute::<i8, i8, i32, i32>(&a0, &b0, &c, &c);
                let a1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::RowMajor,
                    &sa.slice(wm * 512 + 16, 1024),
                    32,
                );
                let b1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B,
                    16usize,
                    16usize,
                    16usize,
                    cmma::MatrixLayout::ColMajor,
                    &sb.slice(wn * 512 + 16, 2048),
                    32,
                );
                cmma::execute::<i8, i8, i32, i32>(&a1, &b1, &c, &c);

                cmma::store(
                    &mut ci.slice_mut(warp * 256, warp * 256 + 256),
                    &c,
                    16,
                    cmma::MatrixLayout::RowMajor,
                );
            }
            // NORMATIVE oracle: the identical i32 contraction as portable scalar MACs.
            default => {
                for e in 0usize..per {
                    let p = lane * per + e;
                    let smm = p / 16;
                    let snn = p % 16;
                    let mut isum = 0i32;
                    for l in 0usize..32 {
                        isum += i32::cast_from(sa[(wm * 16 + smm) * 32 + l])
                            * i32::cast_from(sb[(wn * 16 + snn) * 32 + l]);
                    }
                    ci[warp * 256 + p] = isum;
                }
            }
        };
        sync_cube();

        // Apply the per-block f32 scale, accumulate into the f32 tile.
        for e in 0usize..per {
            let p = lane * per + e;
            let smm = p / 16;
            let snn = p % 16;
            let gmm = wm * 16 + smm;
            let gnn = wn * 16 + snn;
            let xsc = xs[(mrow0 + gmm) * kb_count + kb];
            let wsc = wd[(ncol0 + gnn) * kb_count + kb];
            accf[gmm * 64 + gnn] += f32::cast_from(ci[warp * 256 + p]) * xsc * wsc;
        }
        sync_cube();
    }

    for e in 0usize..per {
        let p = lane * per + e;
        let gmm = wm * 16 + p / 16;
        let gnn = wn * 16 + p % 16;
        out[(mrow0 + gmm) * n + (ncol0 + gnn)] = accf[gmm * 64 + gnn];
    }
}

/// Host launch for the tiled MMQ GEMM -- the PRODUCTION surface; derives the island tag from the
/// runtime. Returns (out, ms/dispatch over `iters`).
#[allow(clippy::too_many_arguments)]
pub fn mmq_q8_wmma_blk_run<R: Runtime>(
    client: &ComputeClient<R>,
    xq: &[i8],
    xs: &[f32],
    wq: &[i8],
    wd: &[f32],
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
) -> (Vec<f32>, f64) {
    mmq_q8_wmma_blk_run_with(client, xq, xs, wq, wd, m, n, k, iters, Target::of(client))
}

/// Host launch for the tiled MMQ GEMM with an explicit island tag. See [`mmq_q8_wmma_run_with`].
#[allow(clippy::too_many_arguments)]
pub fn mmq_q8_wmma_blk_run_with<R: Runtime>(
    client: &ComputeClient<R>,
    xq: &[i8],
    xs: &[f32],
    wq: &[i8],
    wd: &[f32],
    m: usize,
    n: usize,
    k: usize,
    iters: usize,
    target: Target,
) -> (Vec<f32>, f64) {
    let plane = client.properties().hardware.plane_size_max;
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let xsh = client.create_from_slice(f32::as_bytes(xs));
    let wqh = client.create_from_slice(i8::as_bytes(wq));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
    let grid = Grid::Static((n / 64) as u32, (m / 32) as u32, 1);
    let launch = |c: &ComputeClient<R>| unsafe {
        mmq_q8_wmma_blk::launch_unchecked::<R>(
            c,
            grid.clone(),
            Block::new_1d(8 * plane),
            ArrayArg::from_raw_parts(xqh.clone(), xq.len()),
            ArrayArg::from_raw_parts(xsh.clone(), xs.len()),
            ArrayArg::from_raw_parts(wqh.clone(), wq.len()),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(oh.clone(), m * n),
            n,
            k,
            plane as usize,
            target,
        );
    };
    launch(client);
    let out = f32::from_bytes(&client.read_one_unchecked(oh.clone())).to_vec();
    for _ in 0..3 {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh.clone());
    let t = std::time::Instant::now();
    for _ in 0..iters {
        launch(client);
    }
    let _ = client.read_one_unchecked(oh);
    let ms = t.elapsed().as_secs_f64() * 1e3 / iters as f64;
    (out, ms)
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
    let xq: Vec<i8> = (0..m * k)
        .map(|_| ((next() % 255) as i64 - 127) as i8)
        .collect();
    let wq: Vec<i8> = (0..n * k)
        .map(|_| ((next() % 255) as i64 - 127) as i8)
        .collect();
    // q8_1 / Q8_0 scales are amax/127-ish small positives.
    let xs: Vec<f32> = (0..m * kb)
        .map(|_| (next() % 1000) as f32 / 50000.0 + 0.002)
        .collect();
    let wd: Vec<f32> = (0..n * kb)
        .map(|_| (next() % 1000) as f32 / 50000.0 + 0.002)
        .collect();
    (xq, xs, wq, wd)
}

// ================================================================================================
// Q4_K MMQ GEMM: the same int8 tensor-core contraction, made AFFINE for K-quant weights.
//
// Q8_0/q8_1 is symmetric (w = scale*code), so the block scale is a single post-multiply. Q4_K is
// AFFINE -- w = d*sc*nibble - dmin*m -- and the subtracted `dmin*m` offset does not survive a
// tensor-core dot. It factors out of the contraction instead, because it is constant across a
// 32-block (one Q4_K sub-block == one 32-wide MMQ tile: a 256-weight super-block is 8 sub-blocks):
//
//   sum_k (D*q_k - M) * (xs*xq_k) = xs * ( D * sum_k q_k*xq_k  -  M * sum_k xq_k )
//                                          ^^^^^^ the cmma dot ^^^      ^^^ xsum ^^^
//
// So the cmma stage is IDENTICAL (int8 nibble x int8 activation -> i32); only the epilogue gains the
// `- M*xsum` correction, and the kernel takes two extra inputs: `xsum` (per activation 32-block, sum
// of xq -- `quantize_act_q8` already emits it) and `wmin` (M = dmin*m per weight 32-block). `wq` here
// is the Q4_K nibble (0..15) as i8 and `wd` is D = d*sc, both produced host-side, mirroring how the
// q8 kernel takes pre-quantized int8 + a scale. Decoding the super-block in-kernel to keep 4.5 bits
// is the bandwidth follow-up; this proves the affine contraction first.
// ================================================================================================
// Q4_K decode helpers, byte-for-byte the ones in `quant.rs` that `matvec_q4k` is gated against
// (BlockQ4K::to_float). Duplicated here rather than cross-module-exported: a `#[device]` fn is
// module-private, and the copy is three lines each and proven.
#[device]
fn q4k_byte(a: &Array<u32>, base: usize, i: usize) -> u32 {
    (a[base + i / 4] >> ((8 * (i % 4)) as u32)) & 255
}
#[device]
fn q4k_sc(wsc: &Array<u32>, scbase: usize, j: usize) -> u32 {
    let mut r = q4k_byte(wsc, scbase, j) & 63;
    if j >= 4 {
        r = (q4k_byte(wsc, scbase, j + 4) & 15) | ((q4k_byte(wsc, scbase, j - 4) >> 6) << 4);
    }
    r
}
#[device]
fn q4k_m(wsc: &Array<u32>, scbase: usize, j: usize) -> u32 {
    let mut r = q4k_byte(wsc, scbase, j + 4) & 63;
    if j >= 4 {
        r = (q4k_byte(wsc, scbase, j + 4) >> 4) | ((q4k_byte(wsc, scbase, j) >> 6) << 4);
    }
    r
}

#[kernel(targets(cuda, rocm, vulkan, metal, cpu), unchecked)]
pub fn mmq_q4k_wmma_blk(
    xq: &Array<i8>,
    xs: &Array<f32>,
    xsum: &Array<f32>, // per-32-block xs*Sum(xq) -- the DEQUANTIZED block sum, matching quantize_act_q8
    wqs: &Array<u32>, // packed Q4_K qs: 32 u32 (128 B) / super-block
    wsc: &Array<u32>, // packed Q4_K scales: 3 u32 (12 B) / super-block
    wd: &Array<f32>,  // d per super-block   [n * k/256]
    wdm: &Array<f32>, // dmin per super-block [n * k/256]
    out: &mut Array<f32>,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] plane: usize,
    #[comptime] target: Target,
) {
    let tid = UNIT_POS as usize;
    let warp = tid / plane;
    let lane = tid % plane;
    let wm = warp / 4; // M-subtile (0..1) -- a LOCAL, distinct from any weight input
    let wn = warp % 4; // N-subtile (0..3)
    let mrow0 = CUBE_POS_Y as usize * 32;
    let ncol0 = CUBE_POS_X as usize * 64;
    let kb_count = k / 32;
    let nsb = k / 256; // super-blocks per weight row
    let nthread = 8 * plane;
    let per = 256 / plane;

    let mut sa = SharedMemory::<i8>::new(1024usize);
    let mut sb = SharedMemory::<i8>::new(2048usize);
    let mut ci = SharedMemory::<i32>::new(2048usize);
    let mut accf = SharedMemory::<f32>::new(2048usize);

    for e in 0usize..(2048 / nthread) {
        accf[tid * (2048 / nthread) + e] = 0.0f32;
    }
    sync_cube();

    for kb in 0..kb_count {
        let k0 = kb * 32;
        for i in 0usize..(1024 / nthread) {
            let idx = tid + i * nthread;
            sa[idx] = xq[(mrow0 + idx / 32) * k + k0 + idx % 32];
        }
        // B tile: decode the Q4_K nibble in-place, weights stay 4.5-bit. One 32-block == one Q4_K
        // sub-block `is = kb % 8`; within the super-block's 4 groups of 64, sub-block 2g is the low
        // nibble and 2g+1 the high nibble of qs byte `g*32 + qi` -- exactly matvec_q4k's mapping, so
        // qi = column-in-block and the byte index is (is/2)*32 + qi. The staged i8 (0..15) then feeds
        // the SAME cmma as the q8 kernel.
        let is = kb % 8;
        let g = is / 2;
        let sbk = kb / 8; // super-block index along k
        for i in 0usize..(2048 / nthread) {
            let idx = tid + i * nthread;
            let nrow = ncol0 + idx / 32;
            let qi = idx % 32;
            let qb = q4k_byte(wqs, (nrow * nsb + sbk) * 32, g * 32 + qi);
            let nib = (qb >> (4u32 * (is % 2) as u32)) & 15;
            sb[idx] = i8::cast_from(nib);
        }
        sync_cube();

        island! {
            cuda | rocm | vulkan | metal => {
                let c = cmma::Matrix::<i32>::from_value(
                    cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::Undefined, 0i32,
                );
                let a0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::RowMajor, &sa.slice(wm * 512, 1024), 32,
                );
                let b0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::ColMajor, &sb.slice(wn * 512, 2048), 32,
                );
                cmma::execute::<i8, i8, i32, i32>(&a0, &b0, &c, &c);
                let a1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::RowMajor, &sa.slice(wm * 512 + 16, 1024), 32,
                );
                let b1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::ColMajor, &sb.slice(wn * 512 + 16, 2048), 32,
                );
                cmma::execute::<i8, i8, i32, i32>(&a1, &b1, &c, &c);
                cmma::store(&mut ci.slice_mut(warp * 256, warp * 256 + 256), &c, 16, cmma::MatrixLayout::RowMajor);
            }
            default => {
                for e in 0usize..per {
                    let p = lane * per + e;
                    let smm = p / 16;
                    let snn = p % 16;
                    let mut isum = 0i32;
                    for l in 0usize..32 {
                        isum += i32::cast_from(sa[(wm * 16 + smm) * 32 + l])
                            * i32::cast_from(sb[(wn * 16 + snn) * 32 + l]);
                    }
                    ci[warp * 256 + p] = isum;
                }
            }
        };
        sync_cube();

        // Affine epilogue: xs * (D*dot - M*xsum) per 32-block, with D = d*sc and M = dmin*m decoded
        // from the super-block's 6-bit packed scales for sub-block `is` (get_scale_min_k4).
        for e in 0usize..per {
            let p = lane * per + e;
            let smm = p / 16;
            let snn = p % 16;
            let gmm = wm * 16 + smm;
            let gnn = wn * 16 + snn;
            let nrow = ncol0 + gnn;
            let blk = nrow * nsb + sbk;
            let scbase = blk * 3;
            let wdd = wd[blk] * f32::cast_from(q4k_sc(wsc, scbase, is));
            let wmm = wdm[blk] * f32::cast_from(q4k_m(wsc, scbase, is));
            let xsc = xs[(mrow0 + gmm) * kb_count + kb];
            let xsm = xsum[(mrow0 + gmm) * kb_count + kb];
            // xs folds into the dot; xsum already carries xs (= xs*Sum(xq)), so the offset needs no
            // outer xs -- the convention the dp4a MoE kernel and quantize_act_q8 share.
            accf[gmm * 64 + gnn] += xsc * wdd * f32::cast_from(ci[warp * 256 + p]) - wmm * xsm;
        }
        sync_cube();
    }

    for e in 0usize..per {
        let p = lane * per + e;
        let gmm = wm * 16 + p / 16;
        let gnn = wn * 16 + p % 16;
        out[(mrow0 + gmm) * n + (ncol0 + gnn)] = accf[gmm * 64 + gnn];
    }
}

/// Host launch for the Q4_K affine MMQ GEMM (production surface; island tag from the runtime).
/// Weights are the packed Q4_K super-block arrays (wqs/wsc/wd/wdm), decoded in-kernel.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_wmma_blk_run<R: Runtime>(
    client: &ComputeClient<R>,
    xq: &[i8], xs: &[f32], xsum: &[f32], wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32],
    m: usize, n: usize, k: usize, iters: usize,
) -> (Vec<f32>, f64) {
    let target = Target::of(client);
    let plane = client.properties().hardware.plane_size_max;
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let xsh = client.create_from_slice(f32::as_bytes(xs));
    let xsumh = client.create_from_slice(f32::as_bytes(xsum));
    let wqsh = client.create_from_slice(u32::as_bytes(wqs));
    let wsch = client.create_from_slice(u32::as_bytes(wsc));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let wdmh = client.create_from_slice(f32::as_bytes(wdm));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
    let grid = Grid::Static((n / 64) as u32, (m / 32) as u32, 1);
    let launch = |c: &ComputeClient<R>| unsafe {
        mmq_q4k_wmma_blk::launch_unchecked::<R>(
            c, grid.clone(), Block::new_1d(8 * plane),
            ArrayArg::from_raw_parts(xqh.clone(), xq.len()),
            ArrayArg::from_raw_parts(xsh.clone(), xs.len()),
            ArrayArg::from_raw_parts(xsumh.clone(), xsum.len()),
            ArrayArg::from_raw_parts(wqsh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(wsch.clone(), wsc.len()),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(wdmh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(oh.clone(), m * n),
            n, k, plane as usize, target,
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

/// Runtime-dims twin of [`mmq_q4k_wmma_blk`]. `m/n/k` arrive in a `meta` SSBO (the pattern
/// `attn::sdpa_blk` uses for a growing `seq_k`) instead of `#[comptime]`, so ONE compiled `.spv`
/// serves EVERY prefill shape -- the production seam that replaces `mul_mm_q4k_tiled_dp4a`. The
/// 32x64 output tile and the 32-wide K-block are fixed (shared-memory sizes are shape-independent),
/// so only the loop bounds and strides read from `meta`; `plane`/`target` stay comptime (a plane is
/// a hardware constant, the island tag is the codegen target). Tail guards on `m` and `n` let the
/// grid cover a partial tile -- k is a Q4_K super-block multiple (256) so it never tails.
#[kernel(targets(cuda, rocm, vulkan, metal, cpu), unchecked)]
pub fn mmq_q4k_wmma_rt(
    xq: &Array<i8>,
    xs: &Array<f32>,
    xsum: &Array<f32>,
    wqs: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<f32>,
    wdm: &Array<f32>,
    out: &mut Array<f32>,
    meta: &Array<u32>, // [m, n, k]
    #[comptime] plane: usize,
    #[comptime] target: Target,
) {
    let m = meta[0] as usize;
    let n = meta[1] as usize;
    let k = meta[2] as usize;
    let tid = UNIT_POS as usize;
    let warp = tid / plane;
    let lane = tid % plane;
    let wm = warp / 4;
    let wn = warp % 4;
    let mrow0 = CUBE_POS_Y as usize * 32;
    let ncol0 = CUBE_POS_X as usize * 64;
    let kb_count = k / 32;
    let nsb = k / 256;
    let nthread = 8 * plane;
    let per = 256 / plane;

    let mut sa = SharedMemory::<i8>::new(1024usize);
    let mut sb = SharedMemory::<i8>::new(2048usize);
    let mut ci = SharedMemory::<i32>::new(2048usize);
    let mut accf = SharedMemory::<f32>::new(2048usize);

    for e in 0usize..(2048 / nthread) {
        accf[tid * (2048 / nthread) + e] = 0.0f32;
    }
    sync_cube();

    for kb in 0..kb_count {
        let k0 = kb * 32;
        // A tile: rows past m stage as 0 so their cmma fragment is inert -- only in-range outputs
        // are stored, so an out-of-range row can never pollute a real result.
        for i in 0usize..(1024 / nthread) {
            let idx = tid + i * nthread;
            let arow = mrow0 + idx / 32;
            sa[idx] = 0i8;
            if arow < m {
                sa[idx] = xq[arow * k + k0 + idx % 32];
            }
        }
        let is = kb % 8;
        let g = is / 2;
        let sbk = kb / 8;
        for i in 0usize..(2048 / nthread) {
            let idx = tid + i * nthread;
            let nrow = ncol0 + idx / 32;
            let qi = idx % 32;
            sb[idx] = 0i8;
            if nrow < n {
                let qb = q4k_byte(wqs, (nrow * nsb + sbk) * 32, g * 32 + qi);
                let nib = (qb >> (4u32 * (is % 2) as u32)) & 15;
                sb[idx] = i8::cast_from(nib);
            }
        }
        sync_cube();

        island! {
            cuda | rocm | vulkan | metal => {
                let c = cmma::Matrix::<i32>::from_value(
                    cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::Undefined, 0i32,
                );
                let a0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::RowMajor, &sa.slice(wm * 512, 1024), 32,
                );
                let b0 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::ColMajor, &sb.slice(wn * 512, 2048), 32,
                );
                cmma::execute::<i8, i8, i32, i32>(&a0, &b0, &c, &c);
                let a1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::A, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::RowMajor, &sa.slice(wm * 512 + 16, 1024), 32,
                );
                let b1 = cmma::Matrix::<i8>::from_slice(
                    cmma::MatrixIdent::B, 16usize, 16usize, 16usize,
                    cmma::MatrixLayout::ColMajor, &sb.slice(wn * 512 + 16, 2048), 32,
                );
                cmma::execute::<i8, i8, i32, i32>(&a1, &b1, &c, &c);
                cmma::store(&mut ci.slice_mut(warp * 256, warp * 256 + 256), &c, 16, cmma::MatrixLayout::RowMajor);
            }
            default => {
                for e in 0usize..per {
                    let p = lane * per + e;
                    let smm = p / 16;
                    let snn = p % 16;
                    let mut isum = 0i32;
                    for l in 0usize..32 {
                        isum += i32::cast_from(sa[(wm * 16 + smm) * 32 + l])
                            * i32::cast_from(sb[(wn * 16 + snn) * 32 + l]);
                    }
                    ci[warp * 256 + p] = isum;
                }
            }
        };
        sync_cube();

        for e in 0usize..per {
            let p = lane * per + e;
            let gmm = wm * 16 + p / 16;
            let gnn = wn * 16 + p % 16;
            let arow = mrow0 + gmm;
            let nrow = ncol0 + gnn;
            if arow < m && nrow < n {
                let blk = nrow * nsb + sbk;
                let scbase = blk * 3;
                let wdd = wd[blk] * f32::cast_from(q4k_sc(wsc, scbase, is));
                let wmm = wdm[blk] * f32::cast_from(q4k_m(wsc, scbase, is));
                let xsc = xs[arow * kb_count + kb];
                let xsm = xsum[arow * kb_count + kb];
                accf[gmm * 64 + gnn] += xsc * wdd * f32::cast_from(ci[warp * 256 + p]) - wmm * xsm;
            }
        }
        sync_cube();
    }

    for e in 0usize..per {
        let p = lane * per + e;
        let gmm = wm * 16 + p / 16;
        let gnn = wn * 16 + p % 16;
        let arow = mrow0 + gmm;
        let nrow = ncol0 + gnn;
        if arow < m && nrow < n {
            out[arow * n + nrow] = accf[gmm * 64 + gnn];
        }
    }
}

/// Host launch for the runtime-dims Q4_K MMQ. `m/n/k` go in `meta`; the grid rounds up so a partial
/// output tile is covered and clipped by the in-kernel tail guards.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_wmma_rt_run<R: Runtime>(
    client: &ComputeClient<R>,
    xq: &[i8], xs: &[f32], xsum: &[f32], wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32],
    m: usize, n: usize, k: usize, iters: usize,
) -> (Vec<f32>, f64) {
    let target = Target::of(client);
    let plane = client.properties().hardware.plane_size_max;
    let meta = [m as u32, n as u32, k as u32];
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let xsh = client.create_from_slice(f32::as_bytes(xs));
    let xsumh = client.create_from_slice(f32::as_bytes(xsum));
    let wqsh = client.create_from_slice(u32::as_bytes(wqs));
    let wsch = client.create_from_slice(u32::as_bytes(wsc));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let wdmh = client.create_from_slice(f32::as_bytes(wdm));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
    let mh = client.create_from_slice(u32::as_bytes(&meta));
    let grid = Grid::Static(n.div_ceil(64) as u32, m.div_ceil(32) as u32, 1);
    let launch = |c: &ComputeClient<R>| unsafe {
        mmq_q4k_wmma_rt::launch_unchecked::<R>(
            c, grid.clone(), Block::new_1d(8 * plane),
            ArrayArg::from_raw_parts(xqh.clone(), xq.len()),
            ArrayArg::from_raw_parts(xsh.clone(), xs.len()),
            ArrayArg::from_raw_parts(xsumh.clone(), xsum.len()),
            ArrayArg::from_raw_parts(wqsh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(wsch.clone(), wsc.len()),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(wdmh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(oh.clone(), m * n),
            ArrayArg::from_raw_parts(mh.clone(), meta.len()),
            plane as usize, target,
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

fn cpu_q4k_byte(a: &[u32], base: usize, i: usize) -> u32 { (a[base + i / 4] >> (8 * (i % 4))) & 255 }
fn cpu_q4k_sc(wsc: &[u32], sb: usize, j: usize) -> u32 {
    if j < 4 { cpu_q4k_byte(wsc, sb, j) & 63 }
    else { (cpu_q4k_byte(wsc, sb, j + 4) & 15) | ((cpu_q4k_byte(wsc, sb, j - 4) >> 6) << 4) }
}
fn cpu_q4k_m(wsc: &[u32], sb: usize, j: usize) -> u32 {
    if j < 4 { cpu_q4k_byte(wsc, sb, j + 4) & 63 }
    else { (cpu_q4k_byte(wsc, sb, j + 4) >> 4) | ((cpu_q4k_byte(wsc, sb, j) >> 6) << 4) }
}

/// CPU oracle: the exact affine MMQ math over packed Q4_K weights, decoded the same way the kernel
/// (and BlockQ4K::to_float) does. out[m,n] = sum_kb xs*(D*<xq,q> - M*xsum), summed in f32.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_ref(
    xq: &[i8], xs: &[f32], xsum: &[f32], wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32],
    m: usize, n: usize, k: usize,
) -> Vec<f32> {
    let kb = k / 32;
    let nsb = k / 256;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for b in 0..kb {
                let is = b % 8;
                let g = is / 2;
                let blk = j * nsb + b / 8;
                let mut isum = 0i32;
                for qi in 0..32 {
                    let qbyte = cpu_q4k_byte(wqs, blk * 32, g * 32 + qi);
                    let nib = ((qbyte >> (4 * (is % 2))) & 15) as i32;
                    isum += xq[i * k + b * 32 + qi] as i32 * nib;
                }
                let dd = wd[blk] * cpu_q4k_sc(wsc, blk * 3, is) as f32;
                let mm = wdm[blk] * cpu_q4k_m(wsc, blk * 3, is) as f32;
                acc += xs[i * kb + b] * dd * isum as f32 - mm * xsum[i * kb + b];
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Deterministic packed-Q4_K MMQ test data: (xq, xs, xsum, wqs, wsc, wd, wdm). `xsum` is DERIVED from
/// `xq` (sum per 32-block) so the affine correction is exact; wqs are random nibbles, wsc random 6-bit
/// scales, d/dmin small positives -- valid inputs for the get_scale_min_k4 decode.
pub fn gen_mmq_q4k(m: usize, n: usize, k: usize)
    -> (Vec<i8>, Vec<f32>, Vec<f32>, Vec<u32>, Vec<u32>, Vec<f32>, Vec<f32>) {
    let kb = k / 32;
    let nsb = k / 256;
    let mut s = 0x243F6A8885A308D3u64;
    let mut next = || { s ^= s << 13; s ^= s >> 7; s ^= s << 17; s };
    let xq: Vec<i8> = (0..m * k).map(|_| ((next() % 255) as i64 - 127) as i8).collect();
    let wqs: Vec<u32> = (0..n * nsb * 32).map(|_| next() as u32).collect(); // 128 qs bytes/super-block
    let wsc: Vec<u32> = (0..n * nsb * 3).map(|_| next() as u32).collect(); //  12 scale bytes/super-block
    let wd: Vec<f32> = (0..n * nsb).map(|_| (next() % 1000) as f32 / 20000.0 + 0.002).collect();
    let wdm: Vec<f32> = (0..n * nsb).map(|_| (next() % 1000) as f32 / 40000.0).collect();
    let xs: Vec<f32> = (0..m * kb).map(|_| (next() % 1000) as f32 / 50000.0 + 0.002).collect();
    // xsum = xs*Sum(xq) per block -- the dequantized block sum, the convention quantize_act_q8 emits.
    let mut xsum = vec![0.0f32; m * kb];
    for i in 0..m {
        for b in 0..kb {
            let mut acc = 0i32;
            for l in 0..32 { acc += xq[i * k + b * 32 + l] as i32; }
            xsum[i * kb + b] = xs[i * kb + b] * acc as f32;
        }
    }
    (xq, xs, xsum, wqs, wsc, wd, wdm)
}

// ================================================================================================
// COMPTIME-TILED Q4_K affine MMQ GEMM -- the autotunable prefill GEMM.
//
// `mmq_q4k_wmma_blk` fixes the schedule: a 32x64 output tile, 8 warps, one 16x16 subtile per warp, the
// f32 accumulator in shared memory. That FIXED tile is the codegen gap that keeps the DSL from reaching
// a hand-tuned coopmat prefill schedule: the hand `mul_mm_q4k_coopmat` runs a 128x128 tile with a
// per-warp RMxRN register-blocked accumulator held across the whole K loop. This kernel exposes the same
// axes as `#[comptime]` knobs so `tune::Evolution` can search them per (device, m, n, k):
//   * wm x wn -- the warp GRID (nwarp = wm*wn subgroups, wm rows x wn cols of the block);
//   * rm x rn -- the per-warp register tile (each warp owns rm x rn of the 16x16 subtiles);
//   => BM = wm*rm*16 rows, BN = wn*rn*16 cols.
// It holds the f32 output in REGISTERS across the K loop (like the hand kernel's acc[][]), spilling only
// the per-32-block i32 fragment through a shared scratch for the affine scale -- the one round-trip int8
// MMQ cannot avoid, because int8 accumulation is exact only within a constant-scale 32-block, so the
// `D*dot - M*xsum` scale is applied per K-block (the f16 coopmat path instead folds the scale into the
// f16 operand before the contraction, so its accumulator can stay in coopmat registers and skip this
// scratch -- the algorithmic reason an int8 MMQ tile and an f16 coopmat tile are different kernels).
// One island per GEMM: the cmma arm issues the matrix-core op, the `default` arm is the scalar oracle.
// ================================================================================================

/// Comptime-tiled Q4_K affine MMQ GEMM. Grid = (n/BN, m/BM); block = nwarp planes (nwarp = wm*wn).
/// `wm x wn` is the warp grid, `rm x rn` the per-warp 16x16 register tile: BM = wm*rm*16, BN = wn*rn*16.
/// Tail guards on m/n let the grid cover a partial edge tile; k is a Q4_K super-block multiple (256).
#[kernel(targets(cuda, rocm, vulkan, metal, cpu), unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_wmma_tile(
    xq: &Array<i8>,
    xs: &Array<f32>,
    xsum: &Array<f32>,
    wqs: &Array<u32>,
    wsc: &Array<u32>,
    wd: &Array<f32>,
    wdm: &Array<f32>,
    out: &mut Array<f32>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] wm: usize,
    #[comptime] wn: usize,
    #[comptime] rm: usize,
    #[comptime] rn: usize,
    #[comptime] plane: usize,
    #[comptime] target: Target,
) {
    let nwarp = wm * wn;
    let bm = wm * rm * 16;
    let bn = wn * rn * 16;
    let tid = UNIT_POS as usize;
    let warp = tid / plane;
    let lane = tid % plane;
    let warp_m = warp / wn; // warp-grid row (0..wm-1)
    let warp_n = warp % wn; // warp-grid col (0..wn-1)
    let mrow0 = CUBE_POS_Y as usize * bm;
    let ncol0 = CUBE_POS_X as usize * bn;
    let kb_count = k / 32;
    let nsb = k / 256; // super-blocks along k
    let nthread = nwarp * plane;
    let per = 256 / plane; // 16x16-fragment elements this lane owns per subtile

    let mut sa = SharedMemory::<i8>::new(bm * 32); // A tile [BM x 32]
    let mut sb = SharedMemory::<i8>::new(bn * 32); // B tile [BN x 32]
    let mut ci = SharedMemory::<i32>::new(bm * bn); // per-(warp,subtile) i32 fragment scratch, nwarp*rm*rn*256

    // f32 output held in registers across the K loop: rm*rn subtiles, `per` elements each per lane.
    let mut acc = Array::<f32>::new(rm * rn * per);
    for a in 0..(rm * rn * per) {
        acc[a] = 0.0f32;
    }

    let a_elems = bm * 32;
    let a_iters = (bm * 32 + nthread - 1) / nthread;
    let b_elems = bn * 32;
    let b_iters = (bn * 32 + nthread - 1) / nthread;

    for kb in 0..kb_count {
        let k0 = kb * 32;
        // Stage activation A[BM,32] int8, coalesced + tail-guarded on m (a past-edge row stages 0).
        for i in 0..a_iters {
            let idx = tid + i * nthread;
            if idx < a_elems {
                let arow = mrow0 + idx / 32;
                let mut v = 0i8;
                if arow < m {
                    v = xq[arow * k + k0 + idx % 32];
                }
                sa[idx] = v;
            }
        }
        // Stage weight B[BN,32] as decoded Q4_K nibbles (0..15), tail-guarded on n. One 32-block == one
        // Q4_K sub-block `is`; the byte index is (is/2)*32 + qi, the nibble half is is%2 -- matvec_q4k's map.
        let is = kb % 8;
        let g = is / 2;
        let sbk = kb / 8;
        for i in 0..b_iters {
            let idx = tid + i * nthread;
            if idx < b_elems {
                let nrow = ncol0 + idx / 32;
                let qi = idx % 32;
                let mut v = 0i8;
                if nrow < n {
                    let qb = q4k_byte(wqs, (nrow * nsb + sbk) * 32, g * 32 + qi);
                    let nib = (qb >> (4u32 * (is % 2) as u32)) & 15;
                    v = i8::cast_from(nib);
                }
                sb[idx] = v;
            }
        }
        sync_cube();

        // Contract each of this warp's rm*rn subtiles into its private ci region (2 x cmma per 32-block).
        #[unroll]
        for im in 0..rm {
            #[unroll]
            for jn in 0..rn {
                let sm = warp_m * rm + im; // M-subtile of the block
                let sn = warp_n * rn + jn; // N-subtile of the block
                let cbase = (warp * rm * rn + im * rn + jn) * 256;
                island! {
                    cuda | rocm | vulkan | metal => {
                        let c = cmma::Matrix::<i32>::from_value(
                            cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize,
                            cmma::MatrixLayout::Undefined, 0i32,
                        );
                        let a0 = cmma::Matrix::<i8>::from_slice(
                            cmma::MatrixIdent::A, 16usize, 16usize, 16usize,
                            cmma::MatrixLayout::RowMajor, &sa.slice(sm * 512, bm * 32), 32,
                        );
                        let b0 = cmma::Matrix::<i8>::from_slice(
                            cmma::MatrixIdent::B, 16usize, 16usize, 16usize,
                            cmma::MatrixLayout::ColMajor, &sb.slice(sn * 512, bn * 32), 32,
                        );
                        cmma::execute::<i8, i8, i32, i32>(&a0, &b0, &c, &c);
                        let a1 = cmma::Matrix::<i8>::from_slice(
                            cmma::MatrixIdent::A, 16usize, 16usize, 16usize,
                            cmma::MatrixLayout::RowMajor, &sa.slice(sm * 512 + 16, bm * 32), 32,
                        );
                        let b1 = cmma::Matrix::<i8>::from_slice(
                            cmma::MatrixIdent::B, 16usize, 16usize, 16usize,
                            cmma::MatrixLayout::ColMajor, &sb.slice(sn * 512 + 16, bn * 32), 32,
                        );
                        cmma::execute::<i8, i8, i32, i32>(&a1, &b1, &c, &c);
                        cmma::store(&mut ci.slice_mut(cbase, cbase + 256), &c, 16, cmma::MatrixLayout::RowMajor);
                    }
                    default => {
                        for e in 0usize..per {
                            let p = lane * per + e;
                            let smm = p / 16;
                            let snn = p % 16;
                            let mut isum = 0i32;
                            for l in 0usize..32 {
                                isum += i32::cast_from(sa[(sm * 16 + smm) * 32 + l])
                                    * i32::cast_from(sb[(sn * 16 + snn) * 32 + l]);
                            }
                            ci[cbase + p] = isum;
                        }
                    }
                };
            }
        }
        sync_cube();

        // Affine epilogue: acc += xs*(D*dot) - M*xsum per subtile element, into the register accumulator.
        #[unroll]
        for im in 0..rm {
            #[unroll]
            for jn in 0..rn {
                let sm = warp_m * rm + im;
                let sn = warp_n * rn + jn;
                let cbase = (warp * rm * rn + im * rn + jn) * 256;
                for e in 0usize..per {
                    let p = lane * per + e;
                    let gmm = sm * 16 + p / 16;
                    let gnn = sn * 16 + p % 16;
                    let arow = mrow0 + gmm;
                    let nrow = ncol0 + gnn;
                    if arow < m && nrow < n {
                        let blk = nrow * nsb + sbk;
                        let scbase = blk * 3;
                        let wdd = wd[blk] * f32::cast_from(q4k_sc(wsc, scbase, is));
                        let wmm = wdm[blk] * f32::cast_from(q4k_m(wsc, scbase, is));
                        let xsc = xs[arow * kb_count + kb];
                        let xsm = xsum[arow * kb_count + kb];
                        acc[(im * rn + jn) * per + e] +=
                            xsc * wdd * f32::cast_from(ci[cbase + p]) - wmm * xsm;
                    }
                }
            }
        }
        sync_cube();
    }

    // Write the register accumulators to global memory.
    #[unroll]
    for im in 0..rm {
        #[unroll]
        for jn in 0..rn {
            let sm = warp_m * rm + im;
            let sn = warp_n * rn + jn;
            for e in 0usize..per {
                let p = lane * per + e;
                let gmm = sm * 16 + p / 16;
                let gnn = sn * 16 + p % 16;
                let arow = mrow0 + gmm;
                let nrow = ncol0 + gnn;
                if arow < m && nrow < n {
                    out[arow * n + nrow] = acc[(im * rn + jn) * per + e];
                }
            }
        }
    }
}

/// Host launch for the comptime-tiled Q4_K MMQ GEMM. `iters` amortizes a kernel-only bench; returns
/// (out, ms/dispatch). The tile knobs `(wm, wn, rm, rn)` are the autotune genome.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_wmma_tile_run<R: Runtime>(
    client: &ComputeClient<R>,
    xq: &[i8], xs: &[f32], xsum: &[f32], wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32],
    m: usize, n: usize, k: usize, wm: usize, wn: usize, rm: usize, rn: usize, iters: usize,
) -> (Vec<f32>, f64) {
    let target = Target::of(client);
    let plane = client.properties().hardware.plane_size_max as usize;
    let bm = wm * rm * 16;
    let bn = wn * rn * 16;
    let xqh = client.create_from_slice(i8::as_bytes(xq));
    let xsh = client.create_from_slice(f32::as_bytes(xs));
    let xsumh = client.create_from_slice(f32::as_bytes(xsum));
    let wqsh = client.create_from_slice(u32::as_bytes(wqs));
    let wsch = client.create_from_slice(u32::as_bytes(wsc));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let wdmh = client.create_from_slice(f32::as_bytes(wdm));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
    let grid = Grid::Static(n.div_ceil(bn) as u32, m.div_ceil(bm) as u32, 1);
    let block = Block::new_1d((wm * wn * plane) as u32);
    let launch = |c: &ComputeClient<R>| unsafe {
        mmq_q4k_wmma_tile::launch_unchecked::<R>(
            c, grid.clone(), block,
            ArrayArg::from_raw_parts(xqh.clone(), xq.len()),
            ArrayArg::from_raw_parts(xsh.clone(), xs.len()),
            ArrayArg::from_raw_parts(xsumh.clone(), xsum.len()),
            ArrayArg::from_raw_parts(wqsh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(wsch.clone(), wsc.len()),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(wdmh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(oh.clone(), m * n),
            m, n, k, wm, wn, rm, rn, plane, target,
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

// ================================================================================================
// f16-dequant COOPMAT Q4_K GEMM -- the DSL twin of the hand `mul_mm_q4k_coopmat`.
//
// Same algorithm as the hand shader (NOT the int8 MMQ above): decode Q4_K to f16 with the affine scale
// FOLDED into the f16 weight (d*sc*q - dmin*m), round the activation to f16, and contract on the f16
// matrix cores. Because the scale is folded before the contraction, the accumulator sums across ALL K
// and can stay in registers -- a `Sequence<cmma::Matrix>` of rm*rn f32 accumulators held across the K
// loop, exactly the hand kernel's `coopmat acc[WMT][WNT]`. That register residency is what lets an f16
// coopmat tile reach 128x128 with few warps, unlike the int8 kernel whose per-32-block scale forces the
// i32 fragment through a shared scratch every step. Comptime tile: wm x wn warp grid, rm x rn per-warp
// register tile (BM = wm*rm*16, BN = wn*rn*16). One island: the cmma arm is the matrix-core contraction;
// the `default` arm is the scalar f16 oracle the CPU runtime runs (bit-exact to the cmma arm by
// construction, both rounding the same operands to f16 and accumulating in f32). m,n multiples of 16.
// ================================================================================================

/// f16-dequant register-blocked coopmat Q4_K GEMM. Grid = (n/BN, m/BM); block = nwarp planes.
#[kernel(targets(cuda, rocm, vulkan, metal, cpu), unchecked)]
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_coopmat_tile<F: Float>(
    x: &Array<F>,     // activation [m, k], row-major
    wqs: &Array<u32>, // packed Q4_K qs: 32 u32 / super-block
    wsc: &Array<u32>, // packed Q4_K scales: 3 u32 / super-block
    wd: &Array<F>,    // d per super-block   [n * k/256]
    wdm: &Array<F>,   // dmin per super-block [n * k/256]
    out: &mut Array<F>,
    #[comptime] m: usize,
    #[comptime] n: usize,
    #[comptime] k: usize,
    #[comptime] wm: usize,
    #[comptime] wn: usize,
    #[comptime] rm: usize,
    #[comptime] rn: usize,
    #[comptime] plane: usize,
    #[comptime] target: Target,
) {
    let nwarp = wm * wn;
    let bm = wm * rm * 16;
    let bn = wn * rn * 16;
    let tid = UNIT_POS as usize;
    let warp = tid / plane;
    let lane = tid % plane;
    let warp_m = warp / wn;
    let warp_n = warp % wn;
    let mrow0 = CUBE_POS_Y as usize * bm;
    let ncol0 = CUBE_POS_X as usize * bn;
    let kb_count = k / 32;
    let nsb = k / 256;
    let nthread = nwarp * plane;
    let per = 256 / plane;
    let a_iters = (bm * 32 + nthread - 1) / nthread;
    let b_iters = (bn * 32 + nthread - 1) / nthread;

    island! {
        // Accelerated: f16 x f16 -> f32 on the matrix cores, rm*rn accumulators held across K.
        cuda | rocm | vulkan | metal => {
            let mut sa = SharedMemory::<half::f16>::new(bm * 32);
            let mut sb = SharedMemory::<half::f16>::new(bn * 32);
            let mut acc = Sequence::<cmma::Matrix<F>>::new();
            #[unroll]
            for _i in 0..(rm * rn) {
                acc.push(cmma::Matrix::<F>::from_value(
                    cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, F::new(0.0),
                ));
            }
            for kb in 0..kb_count {
                let k0 = kb * 32;
                for i in 0..a_iters {
                    let idx = tid + i * nthread;
                    if idx < bm * 32 {
                        let arow = mrow0 + idx / 32;
                        let mut v = F::new(0.0);
                        if arow < m {
                            v = x[arow * k + k0 + idx % 32];
                        }
                        sa[idx] = half::f16::cast_from(v);
                    }
                }
                let is = kb % 8;
                let g = is / 2;
                let sbk = kb / 8;
                let shift = 4u32 * (is % 2) as u32;
                for i in 0..b_iters {
                    let idx = tid + i * nthread;
                    if idx < bn * 32 {
                        let nrow = ncol0 + idx / 32;
                        let qi = idx % 32;
                        let mut wv = F::new(0.0);
                        if nrow < n {
                            let blk = nrow * nsb + sbk;
                            let ds = wd[blk] * F::cast_from(q4k_sc(wsc, blk * 3, is));
                            let ms = wdm[blk] * F::cast_from(q4k_m(wsc, blk * 3, is));
                            let qb = q4k_byte(wqs, blk * 32, g * 32 + qi);
                            let nib = (qb >> shift) & 15;
                            wv = ds * F::cast_from(nib) - ms;
                        }
                        sb[idx] = half::f16::cast_from(wv);
                    }
                }
                sync_cube();
                // Two cmma per 32-block (the K-subtile halves at column 0 and 16).
                #[unroll]
                for kc in 0..2usize {
                    #[unroll]
                    for im in 0..rm {
                        let sm = warp_m * rm + im;
                        let ma = cmma::Matrix::<half::f16>::from_slice(
                            cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &sa.slice(sm * 512 + kc * 16, bm * 32), 32,
                        );
                        #[unroll]
                        for jn in 0..rn {
                            let sn = warp_n * rn + jn;
                            let mb = cmma::Matrix::<half::f16>::from_slice(
                                cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                                &sb.slice(sn * 512 + kc * 16, bn * 32), 32,
                            );
                            let c = acc.index(im * rn + jn);
                            cmma::execute::<half::f16, half::f16, F, F>(&ma, &mb, c, c);
                        }
                    }
                }
                sync_cube();
            }
            #[unroll]
            for im in 0..rm {
                #[unroll]
                for jn in 0..rn {
                    let sm = warp_m * rm + im;
                    let sn = warp_n * rn + jn;
                    let orow = mrow0 + sm * 16;
                    let ocol = ncol0 + sn * 16;
                    if orow < m && ocol < n {
                        cmma::store(
                            &mut out.slice_mut(orow * n + ocol, out.len()),
                            acc.index(im * rn + jn),
                            n as u32,
                            cmma::MatrixLayout::RowMajor,
                        );
                    }
                }
            }
        }
        // NORMATIVE oracle: the identical f16-rounded contraction as scalar MACs, f32 accumulate. The CPU
        // runtime runs this arm, so it is the bit-exact gate for the tiling + f16 dequant + accumulation.
        default => {
            let mut sa = SharedMemory::<half::f16>::new(bm * 32);
            let mut sb = SharedMemory::<half::f16>::new(bn * 32);
            let mut acc = Array::<F>::new(rm * rn * per);
            for a in 0..(rm * rn * per) {
                acc[a] = F::new(0.0);
            }
            for kb in 0..kb_count {
                let k0 = kb * 32;
                for i in 0..a_iters {
                    let idx = tid + i * nthread;
                    if idx < bm * 32 {
                        let arow = mrow0 + idx / 32;
                        let mut v = F::new(0.0);
                        if arow < m {
                            v = x[arow * k + k0 + idx % 32];
                        }
                        sa[idx] = half::f16::cast_from(v);
                    }
                }
                let is = kb % 8;
                let g = is / 2;
                let sbk = kb / 8;
                let shift = 4u32 * (is % 2) as u32;
                for i in 0..b_iters {
                    let idx = tid + i * nthread;
                    if idx < bn * 32 {
                        let nrow = ncol0 + idx / 32;
                        let qi = idx % 32;
                        let mut wv = F::new(0.0);
                        if nrow < n {
                            let blk = nrow * nsb + sbk;
                            let ds = wd[blk] * F::cast_from(q4k_sc(wsc, blk * 3, is));
                            let ms = wdm[blk] * F::cast_from(q4k_m(wsc, blk * 3, is));
                            let qb = q4k_byte(wqs, blk * 32, g * 32 + qi);
                            let nib = (qb >> shift) & 15;
                            wv = ds * F::cast_from(nib) - ms;
                        }
                        sb[idx] = half::f16::cast_from(wv);
                    }
                }
                sync_cube();
                #[unroll]
                for im in 0..rm {
                    #[unroll]
                    for jn in 0..rn {
                        let sm = warp_m * rm + im;
                        let sn = warp_n * rn + jn;
                        for e in 0..per {
                            let p = lane * per + e;
                            let smm = p / 16;
                            let snn = p % 16;
                            let mut s = F::new(0.0);
                            for l in 0..32 {
                                s += F::cast_from(sa[(sm * 16 + smm) * 32 + l])
                                    * F::cast_from(sb[(sn * 16 + snn) * 32 + l]);
                            }
                            acc[(im * rn + jn) * per + e] += s;
                        }
                    }
                }
                sync_cube();
            }
            #[unroll]
            for im in 0..rm {
                #[unroll]
                for jn in 0..rn {
                    let sm = warp_m * rm + im;
                    let sn = warp_n * rn + jn;
                    for e in 0..per {
                        let p = lane * per + e;
                        let orow = mrow0 + sm * 16 + p / 16;
                        let ocol = ncol0 + sn * 16 + p % 16;
                        if orow < m && ocol < n {
                            out[orow * n + ocol] = acc[(im * rn + jn) * per + e];
                        }
                    }
                }
            }
        }
    };
}

/// f32 full-precision reference for the f16 coopmat twin: out[m,n] = sum_k x[m,k]*(d*sc*q - dmin*m),
/// the true (non-int8-quantized) Q4_K matmul the f16 kernel approximates. The kernel matches this to an
/// f16-rounding tolerance (both operands round to f16), the same gate the hand `mul_mm_q4k_coopmat` uses.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_f16_ref(
    x: &[f32], wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32], m: usize, n: usize, k: usize,
) -> Vec<f32> {
    let kb = k / 32;
    let nsb = k / 256;
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for b in 0..kb {
                let is = b % 8;
                let g = is / 2;
                let blk = j * nsb + b / 8;
                let dd = wd[blk] * cpu_q4k_sc(wsc, blk * 3, is) as f32;
                let mm = wdm[blk] * cpu_q4k_m(wsc, blk * 3, is) as f32;
                for qi in 0..32 {
                    let qbyte = cpu_q4k_byte(wqs, blk * 32, g * 32 + qi);
                    let nib = ((qbyte >> (4 * (is % 2))) & 15) as f32;
                    acc += x[i * k + b * 32 + qi] * (dd * nib - mm);
                }
            }
            out[i * n + j] = acc;
        }
    }
    out
}

/// Host launch for the f16 coopmat Q4_K twin. `iters` amortizes a kernel-only bench; returns (out, ms).
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_coopmat_tile_run<R: Runtime>(
    client: &ComputeClient<R>,
    x: &[f32], wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32],
    m: usize, n: usize, k: usize, wm: usize, wn: usize, rm: usize, rn: usize, iters: usize,
) -> (Vec<f32>, f64) {
    let target = Target::of(client);
    let plane = client.properties().hardware.plane_size_max as usize;
    let bm = wm * rm * 16;
    let bn = wn * rn * 16;
    let xh = client.create_from_slice(f32::as_bytes(x));
    let wqsh = client.create_from_slice(u32::as_bytes(wqs));
    let wsch = client.create_from_slice(u32::as_bytes(wsc));
    let wdh = client.create_from_slice(f32::as_bytes(wd));
    let wdmh = client.create_from_slice(f32::as_bytes(wdm));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
    let grid = Grid::Static(n.div_ceil(bn) as u32, m.div_ceil(bm) as u32, 1);
    let block = Block::new_1d((wm * wn * plane) as u32);
    let launch = |c: &ComputeClient<R>| unsafe {
        mmq_q4k_coopmat_tile::launch_unchecked::<f32, R>(
            c, grid.clone(), block,
            ArrayArg::from_raw_parts(xh.clone(), x.len()),
            ArrayArg::from_raw_parts(wqsh.clone(), wqs.len()),
            ArrayArg::from_raw_parts(wsch.clone(), wsc.len()),
            ArrayArg::from_raw_parts(wdh.clone(), wd.len()),
            ArrayArg::from_raw_parts(wdmh.clone(), wdm.len()),
            ArrayArg::from_raw_parts(oh.clone(), m * n),
            m, n, k, wm, wn, rm, rn, plane, target,
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

// --- autotuner: search the {WM x WN x RM x RN} tile per (device, m, n, k) ------------------------
//
// The DSL sibling of hanzo-ml's `coopmat_hunt` (which tunes the hand `mul_mm_q4k_coopmat` via glslc -D),
// generic in the runtime like the matvec hunts. Same `tune::Evolution` GA + a cold-weight-streamed,
// bit-exact-gated `Evaluator`; `Tuner::evolve` persists the winner keyed by (device, op, m=..,n=..,k=..),
// so first-run tuning replays through the shared TSV cache thereafter -- the "autokernel" cache.

/// The Q4_K MMQ tile schedule space: the warp grid `WM x WN` and the per-warp register tile `RM x RN`
/// (BM = WM*RM*16, BN = WN*RN*16, nwarp = WM*WN). The Space carries only device-independent feasibility
/// (tile-shape sanity, nwarp bound); the plane-dependent workgroup-width limit and the LDS-occupancy
/// budget are the evaluator's free static tier, since both need the device's plane size.
pub fn mmq_q4k_space() -> Space {
    Space::new()
        .param("WM", [1, 2, 4])
        .param("WN", [1, 2, 4])
        .param("RM", [1, 2, 4])
        .param("RN", [1, 2, 4, 8])
        // nwarp = WM*WN in [1, 16] (a wave64 workgroup of 16 subgroups is the 1024-thread ceiling).
        .constraint(|c, s| {
            let w = c.get(s, "WM") * c.get(s, "WN");
            (1..=16).contains(&w)
        })
}

/// Map a schedule [`Config`] to the kernel's `(wm, wn, rm, rn)` launch tuple.
fn mmq_tile_cfg(c: &Config, s: &Space) -> (usize, usize, usize, usize) {
    (
        c.get(s, "WM") as usize,
        c.get(s, "WN") as usize,
        c.get(s, "RM") as usize,
        c.get(s, "RN") as usize,
    )
}

/// Derived tile geometry `(BM, BN, nwarp)` for a config.
fn mmq_geom(c: &Config, s: &Space) -> (usize, usize, usize) {
    let (wm, wn, rm, rn) = mmq_tile_cfg(c, s);
    (wm * rm * 16, wn * rn * 16, wm * wn)
}

/// LDS bytes for a tile: sa[BM*32] i8 + sb[BN*32] i8 + ci[BM*BN] i32 (the per-subtile fragment scratch).
/// The register accumulator (rm*rn*`per` f32/lane) is not LDS; the evaluator bounds it separately.
fn mmq_lds_bytes(c: &Config, s: &Space) -> usize {
    let (bm, bn, _) = mmq_geom(c, s);
    bm * 32 + bn * 32 + bm * bn * 4
}

/// The always-feasible default MMQ tile used before any hunt has run: a 32x64 block, 8 warps -- the shape
/// the fixed `mmq_q4k_wmma_blk` ships, so the autokernel's cold-start dispatch is a known-good schedule.
pub fn mmq_q4k_incumbent() -> (usize, usize, usize, usize) {
    (2, 4, 1, 1) // BM=32, BN=64, nwarp=8
}

/// A cold-weight-streamed, bit-exact-gated fitness over [`mmq_q4k_space`], generic in the runtime. Holds
/// RESIDENT device buffers (the Q4_K weight uploaded as `banks` distinct copies rotated so each timed
/// dispatch reads cold, plus the shared activation + scale inputs) and the `mmq_q4k_ref` oracle. The
/// correctness gate is folded into fitness -- a tile that diverges from the oracle is infinitely slow, so
/// it can never win. `CpuRuntime` exercises the search offline (machinery + bit-exactness); the same type
/// on a wgpu/vulkan device gives the meaningful winner. Mirrors [`crate::quant`]'s dp4a evaluator and
/// hanzo-ml's `CoopmatEval`.
pub struct MmqQ4kEval<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    space: &'a Space,
    banks: Vec<Handle>, // resident Q4_K `wqs` copies; banks[0] is the weight the oracle was computed from
    xqh: Handle,
    xsh: Handle,
    xsumh: Handle,
    wsch: Handle,
    wdh: Handle,
    wdmh: Handle,
    outh: Handle,
    wqs_len: usize,
    xq_len: usize,
    xs_len: usize,
    xsum_len: usize,
    wsc_len: usize,
    wd_len: usize,
    wdm_len: usize,
    m: usize,
    n: usize,
    k: usize,
    plane: usize,
    lds_budget: usize,
    reg_budget: usize,
    oracle: Vec<f32>,
    maxref: f32,
    repeats: usize,
    worst_rel: std::cell::Cell<f32>,
}

impl<'a, R: Runtime> MmqQ4kEval<'a, R> {
    /// Upload the weight banks (distinct `wqs` copies so a rotation reads cold), the shared scale inputs
    /// and activation, and an output buffer; compute the `mmq_q4k_ref` oracle on `banks[0]`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &'a ComputeClient<R>,
        space: &'a Space,
        wqs_banks: &[Vec<u32>],
        xq: &[i8],
        xs: &[f32],
        xsum: &[f32],
        wsc: &[u32],
        wd: &[f32],
        wdm: &[f32],
        m: usize,
        n: usize,
        k: usize,
        repeats: usize,
    ) -> Self {
        assert!(!wqs_banks.is_empty(), "MmqQ4kEval needs at least one weight bank");
        let banks: Vec<Handle> =
            wqs_banks.iter().map(|w| client.create_from_slice(u32::as_bytes(w))).collect();
        let xqh = client.create_from_slice(i8::as_bytes(xq));
        let xsh = client.create_from_slice(f32::as_bytes(xs));
        let xsumh = client.create_from_slice(f32::as_bytes(xsum));
        let wsch = client.create_from_slice(u32::as_bytes(wsc));
        let wdh = client.create_from_slice(f32::as_bytes(wd));
        let wdmh = client.create_from_slice(f32::as_bytes(wdm));
        let outh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
        let oracle = mmq_q4k_ref(xq, xs, xsum, &wqs_banks[0], wsc, wd, wdm, m, n, k);
        let maxref = oracle.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
        let plane = client.properties().hardware.plane_size_max as usize;
        Self {
            client,
            space,
            banks,
            xqh,
            xsh,
            xsumh,
            wsch,
            wdh,
            wdmh,
            outh,
            wqs_len: wqs_banks[0].len(),
            xq_len: xq.len(),
            xs_len: xs.len(),
            xsum_len: xsum.len(),
            wsc_len: wsc.len(),
            wd_len: wd.len(),
            wdm_len: wdm.len(),
            m,
            n,
            k,
            plane,
            lds_budget: 48 * 1024, // gfx1151 LDS is 64 KB; 48 KB keeps a spare wave resident
            reg_budget: 16,        // accumulator fragments per warp (rm*rn), the hand kernel's budget
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

    fn dispatch(&self, bank: &Handle, wm: usize, wn: usize, rm: usize, rn: usize) {
        let bm = wm * rm * 16;
        let bn = wn * rn * 16;
        let grid = Grid::Static(self.n.div_ceil(bn) as u32, self.m.div_ceil(bm) as u32, 1);
        let block = Block::new_1d((wm * wn * self.plane) as u32);
        let target = Target::of(self.client);
        unsafe {
            mmq_q4k_wmma_tile::launch_unchecked::<R>(
                self.client,
                grid,
                block,
                ArrayArg::from_raw_parts(self.xqh.clone(), self.xq_len),
                ArrayArg::from_raw_parts(self.xsh.clone(), self.xs_len),
                ArrayArg::from_raw_parts(self.xsumh.clone(), self.xsum_len),
                ArrayArg::from_raw_parts(bank.clone(), self.wqs_len),
                ArrayArg::from_raw_parts(self.wsch.clone(), self.wsc_len),
                ArrayArg::from_raw_parts(self.wdh.clone(), self.wd_len),
                ArrayArg::from_raw_parts(self.wdmh.clone(), self.wdm_len),
                ArrayArg::from_raw_parts(self.outh.clone(), self.m * self.n),
                self.m,
                self.n,
                self.k,
                wm,
                wn,
                rm,
                rn,
                self.plane,
                target,
            );
        }
    }

    fn read_out(&self) -> Vec<f32> {
        f32::from_bytes(&self.client.read_one_unchecked(self.outh.clone())).to_vec()
    }
}

impl<'a, R: Runtime> Evaluator for MmqQ4kEval<'a, R> {
    fn static_check(&self, cfg: &Config) -> Verdict {
        let (bm, bn, nwarp) = mmq_geom(cfg, self.space);
        let wg = nwarp * self.plane;
        if wg > 1024 {
            return Verdict::Reject(format!("workgroup width {wg} = nwarp {nwarp} x plane {} > 1024", self.plane));
        }
        let lds = mmq_lds_bytes(cfg, self.space);
        if lds > self.lds_budget {
            return Verdict::Reject(format!(
                "LDS {}KB (BM {bm} x BN {bn}) > {}KB budget (occupancy)",
                lds / 1024,
                self.lds_budget / 1024
            ));
        }
        // Accumulator fragments per warp = rm*rn 16x16 f32 tiles held in registers across K (the hand
        // coopmat kernel budgets WMT*WNT == 16). A tile needing more spills to scratch and cannot beat a
        // resident one -- reject before timing. Fragment count is plane-independent (unlike the per-lane
        // f32 share 256/plane), so the same budget gates the CPU search and the GPU deployment.
        let (wm, wn, rm, rn) = mmq_tile_cfg(cfg, self.space);
        let _ = (wm, wn, bm, bn);
        let frags = rm * rn;
        if frags > self.reg_budget {
            return Verdict::Reject(format!("register tile {frags} fragments/warp (rm {rm} x rn {rn}) > {} budget", self.reg_budget));
        }
        Verdict::Pass
    }

    fn measure(&self, cfg: &Config, iters: usize) -> f64 {
        let (wm, wn, rm, rn) = mmq_tile_cfg(cfg, self.space);
        // Correctness gate on bank[0] vs mmq_q4k_ref (scale-relative: a signed int8 sum cancels to near
        // zero, so a per-element relative error is a false failure; gate on max|Δ|/max|ref|).
        self.dispatch(&self.banks[0], wm, wn, rm, rn);
        let got = self.read_out();
        let rel = got.iter().zip(&self.oracle).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / self.maxref;
        self.worst_rel.set(self.worst_rel.get().max(rel));
        if rel > 1e-3 {
            return f64::INFINITY;
        }
        // Cold-weight-streamed timing: warm the utilization-slaved clock, then rotate the whole bank so
        // every dispatch reads its weight cold from GTT (the memory-bound regime prefill runs in), taking
        // the MINIMUM over a few passes (the least-drift-polluted time). The trailing read drains the queue.
        let nb = self.banks.len();
        for i in 0..(2 * nb) {
            self.dispatch(&self.banks[i % nb], wm, wn, rm, rn);
        }
        let _ = self.read_out();
        (0..self.repeats)
            .map(|_| {
                let t = std::time::Instant::now();
                for i in 0..iters {
                    self.dispatch(&self.banks[i % nb], wm, wn, rm, rn);
                }
                let _ = self.read_out();
                t.elapsed().as_secs_f64() * 1e3 / iters as f64
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Search [`mmq_q4k_space`] on `eval` for the fastest `(wm, wn, rm, rn)` tile at this `(device, m, n, k)`,
/// caching the winner in the shared autotune TSV. The autotuner surface for the DSL coopmat prefill GEMM.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_hunt<R: Runtime>(
    tuner: &Tuner,
    device: &str,
    m: usize,
    n: usize,
    k: usize,
    eval: &MmqQ4kEval<R>,
    evo: &Evolution,
    seed: u64,
) -> Evolved {
    tuner.evolve(device, "mmq_q4k_tile", &format!("m={m},n={n},k={k}"), eval.space(), eval, evo, seed)
}

/// Autotuned-default dispatch -- the "autokernel". Consults the persisted per-(device, m, n, k) tuned
/// genome and launches THAT tile; with no cached winner it launches the incumbent tile. This is the
/// mechanism that makes the DSL the PREFERRED path: the dispatch runs the runtime-tuned winner over a
/// fixed committed schedule, auto-optimizing on first run (populate the cache with [`mmq_q4k_hunt`]) and
/// replaying the winner from the shared TSV cache thereafter -- no per-shape hand-forked kernel. Returns
/// (output, the genome name that served) so a caller can log which tile the dispatch picked.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_autokernel<R: Runtime>(
    client: &ComputeClient<R>,
    tuner: &Tuner,
    xq: &[i8], xs: &[f32], xsum: &[f32], wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32],
    m: usize, n: usize, k: usize,
) -> (Vec<f32>, String) {
    let device = crate::tune::device_id(client);
    let space = mmq_q4k_space();
    let key = format!("m={m},n={n},k={k}");
    let (wm, wn, rm, rn, name) = match tuner.cached_winner(&device, "mmq_q4k_tile", &key) {
        Some(w) => match space.parse(&w) {
            Some(cfg) => {
                let (a, b, c, d) = mmq_tile_cfg(&cfg, &space);
                (a, b, c, d, w)
            }
            None => {
                let (a, b, c, d) = mmq_q4k_incumbent();
                (a, b, c, d, "incumbent".to_string())
            }
        },
        None => {
            let (a, b, c, d) = mmq_q4k_incumbent();
            (a, b, c, d, "incumbent".to_string())
        }
    };
    let (out, _ms) =
        mmq_q4k_wmma_tile_run(client, xq, xs, xsum, wqs, wsc, wd, wdm, m, n, k, wm, wn, rm, rn, 1);
    (out, name)
}

// --- autotuner for the f16 coopmat twin -- the SAME {WM x WN x RM x RN} Space, a different kernel ------
//
// Reuses `mmq_q4k_space` and the tile helpers; only the fitness differs (it dispatches the f16 kernel,
// gates against the full-precision f32 reference at an f16 tolerance, and drops the int8 kernel's ci
// scratch from the LDS budget). Two evaluators over one Space is the matvec crate's pattern (Dp4a +
// Q4kF32). This is the DSL side of the CTO's board: the autotuned f16 coopmat GEMM vs hand mul_mm_q4k_coopmat.

/// LDS bytes for the f16 coopmat tile: sa[BM*32] + sb[BN*32], f16 (2 B) each. No i32 fragment scratch --
/// the scale is folded into the f16 weight so the accumulator stays in registers, which is exactly why
/// this tile reaches 128x128 where the int8 kernel's ci+accumulator scratch cannot.
fn mmq_coopmat_lds_bytes(c: &Config, s: &Space) -> usize {
    let (bm, bn, _) = mmq_geom(c, s);
    (bm * 32 + bn * 32) * 2
}

/// Cold-weight-streamed, f16-tolerance-gated fitness for the f16 coopmat twin over [`mmq_q4k_space`].
/// Same discipline as [`MmqQ4kEval`]; the activation is f32 (the f16 twin rounds it), the oracle is the
/// full-precision [`mmq_q4k_f16_ref`], and the static tier uses the smaller f16-tile LDS budget.
pub struct CoopmatF16Eval<'a, R: Runtime> {
    client: &'a ComputeClient<R>,
    space: &'a Space,
    banks: Vec<Handle>, // resident Q4_K `wqs` copies; banks[0] is the oracle weight
    xh: Handle,
    wsch: Handle,
    wdh: Handle,
    wdmh: Handle,
    outh: Handle,
    wqs_len: usize,
    x_len: usize,
    wsc_len: usize,
    wd_len: usize,
    wdm_len: usize,
    m: usize,
    n: usize,
    k: usize,
    plane: usize,
    lds_budget: usize,
    reg_budget: usize,
    oracle: Vec<f32>,
    maxref: f32,
    repeats: usize,
    worst_rel: std::cell::Cell<f32>,
}

impl<'a, R: Runtime> CoopmatF16Eval<'a, R> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        client: &'a ComputeClient<R>,
        space: &'a Space,
        wqs_banks: &[Vec<u32>],
        x: &[f32],
        wsc: &[u32],
        wd: &[f32],
        wdm: &[f32],
        m: usize,
        n: usize,
        k: usize,
        repeats: usize,
    ) -> Self {
        assert!(!wqs_banks.is_empty(), "CoopmatF16Eval needs at least one weight bank");
        let banks: Vec<Handle> =
            wqs_banks.iter().map(|w| client.create_from_slice(u32::as_bytes(w))).collect();
        let xh = client.create_from_slice(f32::as_bytes(x));
        let wsch = client.create_from_slice(u32::as_bytes(wsc));
        let wdh = client.create_from_slice(f32::as_bytes(wd));
        let wdmh = client.create_from_slice(f32::as_bytes(wdm));
        let outh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; m * n]));
        let oracle = mmq_q4k_f16_ref(x, &wqs_banks[0], wsc, wd, wdm, m, n, k);
        let maxref = oracle.iter().fold(0f32, |a, &v| a.max(v.abs())).max(1e-30);
        let plane = client.properties().hardware.plane_size_max as usize;
        Self {
            client, space, banks, xh, wsch, wdh, wdmh, outh,
            wqs_len: wqs_banks[0].len(), x_len: x.len(), wsc_len: wsc.len(), wd_len: wd.len(), wdm_len: wdm.len(),
            m, n, k, plane, lds_budget: 48 * 1024, reg_budget: 16, oracle, maxref, repeats,
            worst_rel: std::cell::Cell::new(0.0),
        }
    }

    pub fn space(&self) -> &Space {
        self.space
    }
    pub fn worst_rel(&self) -> f32 {
        self.worst_rel.get()
    }

    fn dispatch(&self, bank: &Handle, wm: usize, wn: usize, rm: usize, rn: usize) {
        let bm = wm * rm * 16;
        let bn = wn * rn * 16;
        let grid = Grid::Static(self.n.div_ceil(bn) as u32, self.m.div_ceil(bm) as u32, 1);
        let block = Block::new_1d((wm * wn * self.plane) as u32);
        let target = Target::of(self.client);
        unsafe {
            mmq_q4k_coopmat_tile::launch_unchecked::<f32, R>(
                self.client, grid, block,
                ArrayArg::from_raw_parts(self.xh.clone(), self.x_len),
                ArrayArg::from_raw_parts(bank.clone(), self.wqs_len),
                ArrayArg::from_raw_parts(self.wsch.clone(), self.wsc_len),
                ArrayArg::from_raw_parts(self.wdh.clone(), self.wd_len),
                ArrayArg::from_raw_parts(self.wdmh.clone(), self.wdm_len),
                ArrayArg::from_raw_parts(self.outh.clone(), self.m * self.n),
                self.m, self.n, self.k, wm, wn, rm, rn, self.plane, target,
            );
        }
    }

    fn read_out(&self) -> Vec<f32> {
        f32::from_bytes(&self.client.read_one_unchecked(self.outh.clone())).to_vec()
    }
}

impl<'a, R: Runtime> Evaluator for CoopmatF16Eval<'a, R> {
    fn static_check(&self, cfg: &Config) -> Verdict {
        let (bm, bn, nwarp) = mmq_geom(cfg, self.space);
        let wg = nwarp * self.plane;
        if wg > 1024 {
            return Verdict::Reject(format!("workgroup width {wg} = nwarp {nwarp} x plane {} > 1024", self.plane));
        }
        let lds = mmq_coopmat_lds_bytes(cfg, self.space);
        if lds > self.lds_budget {
            return Verdict::Reject(format!("LDS {}KB (BM {bm} x BN {bn}) > {}KB budget", lds / 1024, self.lds_budget / 1024));
        }
        let (_wm, _wn, rm, rn) = mmq_tile_cfg(cfg, self.space);
        let frags = rm * rn;
        if frags > self.reg_budget {
            return Verdict::Reject(format!("register tile {frags} fragments/warp (rm {rm} x rn {rn}) > {} budget", self.reg_budget));
        }
        Verdict::Pass
    }

    fn measure(&self, cfg: &Config, iters: usize) -> f64 {
        let (wm, wn, rm, rn) = mmq_tile_cfg(cfg, self.space);
        self.dispatch(&self.banks[0], wm, wn, rm, rn);
        let got = self.read_out();
        let rel = got.iter().zip(&self.oracle).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max) / self.maxref;
        self.worst_rel.set(self.worst_rel.get().max(rel));
        // f16-rounding tolerance (both operands round to f16), the hand coopmat gate class.
        if rel > 5e-2 {
            return f64::INFINITY;
        }
        let nb = self.banks.len();
        for i in 0..(2 * nb) {
            self.dispatch(&self.banks[i % nb], wm, wn, rm, rn);
        }
        let _ = self.read_out();
        (0..self.repeats)
            .map(|_| {
                let t = std::time::Instant::now();
                for i in 0..iters {
                    self.dispatch(&self.banks[i % nb], wm, wn, rm, rn);
                }
                let _ = self.read_out();
                t.elapsed().as_secs_f64() * 1e3 / iters as f64
            })
            .fold(f64::INFINITY, f64::min)
    }
}

/// Search [`mmq_q4k_space`] on `eval` for the fastest f16-coopmat tile at this `(device, m, n, k)`,
/// caching the winner. The DSL side of the coopmat prefill board; the winner replays via the same cache.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_coopmat_hunt<R: Runtime>(
    tuner: &Tuner,
    device: &str,
    m: usize,
    n: usize,
    k: usize,
    eval: &CoopmatF16Eval<R>,
    evo: &Evolution,
    seed: u64,
) -> Evolved {
    tuner.evolve(device, "mmq_q4k_coopmat", &format!("m={m},n={n},k={k}"), eval.space(), eval, evo, seed)
}

/// Autotuned-default dispatch for the f16 coopmat twin: consult the persisted per-(device,m,n,k) tuned
/// genome and launch that tile, else the incumbent. The one-source, auto-optimized coopmat prefill path.
#[allow(clippy::too_many_arguments)]
pub fn mmq_q4k_coopmat_autokernel<R: Runtime>(
    client: &ComputeClient<R>,
    tuner: &Tuner,
    x: &[f32], wqs: &[u32], wsc: &[u32], wd: &[f32], wdm: &[f32],
    m: usize, n: usize, k: usize,
) -> (Vec<f32>, String) {
    let device = crate::tune::device_id(client);
    let space = mmq_q4k_space();
    let key = format!("m={m},n={n},k={k}");
    let (wm, wn, rm, rn, name) = match tuner.cached_winner(&device, "mmq_q4k_coopmat", &key) {
        Some(w) => match space.parse(&w) {
            Some(cfg) => {
                let (a, b, c, d) = mmq_tile_cfg(&cfg, &space);
                (a, b, c, d, w)
            }
            None => {
                let (a, b, c, d) = mmq_q4k_incumbent();
                (a, b, c, d, "incumbent".to_string())
            }
        },
        None => {
            let (a, b, c, d) = mmq_q4k_incumbent();
            (a, b, c, d, "incumbent".to_string())
        }
    };
    let (out, _ms) = mmq_q4k_coopmat_tile_run(client, x, wqs, wsc, wd, wdm, m, n, k, wm, wn, rm, rn, 1);
    (out, name)
}

#[cfg(all(test, feature = "cpu"))]
mod tests {
    use super::*;
    use cubecl::cpu::{CpuDevice, CpuRuntime};

    fn cpu_client() -> ComputeClient<CpuRuntime> {
        CpuRuntime::client(&CpuDevice)
    }

    /// Scale-relative gate: max|delta| / max|ref|. A per-element relative error is meaningless here --
    /// a signed int8 sum cancels to near-zero often, which explodes the per-element denominator and
    /// reports a false failure. A real decode/accumulation bug moves this number; cancellation does not.
    fn rel_to_max(got: &[f32], want: &[f32]) -> f32 {
        let mut maxabs = 0f32;
        let mut refmax = 1e-9f32;
        for (g, w) in got.iter().zip(want) {
            maxabs = maxabs.max((g - w).abs());
            refmax = refmax.max(w.abs());
        }
        maxabs / refmax
    }

    /// The oracle. The naive MMQ GEMM's normative arm, on the CPU runtime, IS `mmq_q8_ref` -- bit-exact,
    /// not merely close. It can be bit-exact because the kernel accumulates the f32 scale in the same
    /// order the reference does (one 32-block at a time, ascending), and the int8 inner sum is integer.
    #[test]
    fn mmq_default_arm_is_bit_exact_on_cpu() {
        let client = cpu_client();
        for (m, n, k) in [(16usize, 16usize, 64usize), (32, 16, 128), (16, 32, 256)] {
            let (xq, xs, wq, wd) = gen_mmq(m, n, k);
            let (got, _) = mmq_q8_wmma_run_with::<CpuRuntime>(
                &client, &xq, &xs, &wq, &wd, m, n, k, 1, Target::Cpu,
            );
            let want = mmq_q8_ref(&xq, &xs, &wq, &wd, m, n, k);
            let gb: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
            let wb: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
            assert_eq!(gb, wb, "MMQ {m}x{n}x{k}: default arm != mmq_q8_ref (bit-exact gate)");
        }
    }

    /// The tiled GEMM's normative arm agrees with the same reference through the shared-memory A/B
    /// staging path. Gated scale-relative rather than bit-exact: this kernel accumulates each warp's
    /// subtile independently, so the f32 add order differs from the reference's single ascending sweep.
    #[test]
    fn mmq_blk_default_arm_matches_ref_on_cpu() {
        let client = cpu_client();
        let (m, n, k) = (32usize, 64usize, 128usize);
        let (xq, xs, wq, wd) = gen_mmq(m, n, k);
        let (got, _) = mmq_q8_wmma_blk_run_with::<CpuRuntime>(
            &client, &xq, &xs, &wq, &wd, m, n, k, 1, Target::Cpu,
        );
        let want = mmq_q8_ref(&xq, &xs, &wq, &wd, m, n, k);
        let rel = rel_to_max(&got, &want);
        assert!(rel < 1e-6, "tiled MMQ {m}x{n}x{k}: rel_to_max={rel:.3e} vs mmq_q8_ref");
    }

    /// The AFFINE Q4_K MMQ agrees with `mmq_q4k_ref` through the same cmma/staging path. This is the
    /// contract the symmetric kernel could not express: the `- M*xsum` correction is what a plain int8
    /// tensor-core dot drops for a K-quant weight. `xsum` is the exact per-block sum of `xq`, so the
    /// correction is not an approximation -- a wrong epilogue (e.g. omitting the term) fails loudly.
    #[test]
    fn mmq_q4k_affine_matches_ref_on_cpu() {
        let client = cpu_client();
        for (m, n, k) in [(32usize, 64usize, 256usize), (32, 64, 512)] {
            let (xq, xs, xsum, wqs, wsc, wd, wdm) = gen_mmq_q4k(m, n, k);
            let (got, _) = mmq_q4k_wmma_blk_run::<CpuRuntime>(
                &client, &xq, &xs, &xsum, &wqs, &wsc, &wd, &wdm, m, n, k, 1,
            );
            let want = mmq_q4k_ref(&xq, &xs, &xsum, &wqs, &wsc, &wd, &wdm, m, n, k);
            let rel = rel_to_max(&got, &want);
            assert!(rel < 1e-6, "affine Q4_K MMQ {m}x{n}x{k}: rel_to_max={rel:.3e} vs mmq_q4k_ref");
        }
    }

    /// The runtime-dims twin agrees with the SAME `mmq_q4k_ref`, exercising both tail guards: m<32
    /// (partial M) and n<64 (partial single N-block), across k=256 and k=512. The guards are the only
    /// new logic over the comptime kernel; a missing one reads past a buffer or writes a garbage row,
    /// which the oracle catches. Shapes stay single-N-block (n<=64): the cubecl CPU runtime does not
    /// isolate SharedMemory across cubes (a cross-cube race -- the comptime kernel fails multi-N-block
    /// on CPU identically), so multi-block N is gated on the real GPU where cmma is proven (the comptime
    /// Vulkan test passes at n=2048 = 32 N-blocks, rel 4.7e-7). k stays a 256-multiple, never tails.
    #[test]
    fn mmq_q4k_rt_matches_ref_with_tails_on_cpu() {
        let client = cpu_client();
        for (m, n, k) in [(32usize, 64usize, 256usize), (32, 64, 512), (17, 64, 256), (32, 50, 256), (17, 50, 512), (1, 64, 256)] {
            let (xq, xs, xsum, wqs, wsc, wd, wdm) = gen_mmq_q4k(m, n, k);
            let (got, _) = mmq_q4k_wmma_rt_run::<CpuRuntime>(
                &client, &xq, &xs, &xsum, &wqs, &wsc, &wd, &wdm, m, n, k, 1,
            );
            let want = mmq_q4k_ref(&xq, &xs, &xsum, &wqs, &wsc, &wd, &wdm, m, n, k);
            let rel = rel_to_max(&got, &want);
            assert!(rel < 1e-6, "runtime Q4_K MMQ {m}x{n}x{k}: rel_to_max={rel:.3e} vs mmq_q4k_ref");
        }
    }

    /// The production surface derives the island tag from the runtime, so the CPU runtime resolves to
    /// the normative arm with no caller naming a target.
    #[test]
    fn production_run_selects_oracle_on_cpu() {
        let client = cpu_client();
        let (m, n, k) = (16usize, 16usize, 64usize);
        let (xq, xs, wq, wd) = gen_mmq(m, n, k);
        let (got, _) = mmq_q8_wmma_run::<CpuRuntime>(&client, &xq, &xs, &wq, &wd, m, n, k, 1);
        let want = mmq_q8_ref(&xq, &xs, &wq, &wd, m, n, k);
        let gb: Vec<u32> = got.iter().map(|x| x.to_bits()).collect();
        let wb: Vec<u32> = want.iter().map(|x| x.to_bits()).collect();
        assert_eq!(gb, wb);
    }

    /// The comptime-tiled Q4_K MMQ agrees with `mmq_q4k_ref` across the schedule axes: warp-grid only
    /// (wm x wn, one subtile per warp), register-block only (rm x rn, one warp), and both. Each shape is
    /// m=BM, n=BN so every subtile carries real output while the grid stays 1x1 (one cube -- the CPU
    /// runtime does not isolate SharedMemory across cubes, so multi-block is gated on the GPU). Proves the
    /// generalized tiling, the register accumulator, and the affine epilogue are correct for every genome.
    #[test]
    fn mmq_q4k_tile_matches_ref_on_cpu() {
        let client = cpu_client();
        for (wm, wn, rm, rn) in
            [(1, 1, 1, 1), (2, 2, 1, 1), (1, 1, 2, 2), (2, 1, 1, 2), (1, 2, 2, 1), (2, 2, 2, 2)]
        {
            let bm = wm * rm * 16;
            let bn = wn * rn * 16;
            for k in [256usize, 512] {
                let (xq, xs, xsum, wqs, wsc, wd, wdm) = gen_mmq_q4k(bm, bn, k);
                let (got, _) = mmq_q4k_wmma_tile_run::<CpuRuntime>(
                    &client, &xq, &xs, &xsum, &wqs, &wsc, &wd, &wdm, bm, bn, k, wm, wn, rm, rn, 1,
                );
                let want = mmq_q4k_ref(&xq, &xs, &xsum, &wqs, &wsc, &wd, &wdm, bm, bn, k);
                let rel = rel_to_max(&got, &want);
                assert!(rel < 1e-6, "tile wm{wm} wn{wn} rm{rm} rn{rn} {bm}x{bn}x{k}: rel_to_max={rel:.3e}");
            }
        }
    }

    /// The f16-dequant coopmat twin (`mmq_q4k_coopmat_tile`, the DSL structural match of the hand
    /// `mul_mm_q4k_coopmat`) agrees with the full-precision f32 reference to an f16-rounding tolerance,
    /// across the warp-grid x register-tile axes INCLUDING the hand kernel's own 128x128 tile
    /// (wm=2,wn=2,rm=4,rn=4). Proves the DSL reaches the hand tile geometry with the register-resident
    /// `Sequence<cmma::Matrix>` accumulator held across K -- the structure whose feasibility the int8
    /// kernel could not express. Each shape is m=BM, n=BN (grid 1x1, one cube: CPU shared-memory safe).
    #[test]
    fn mmq_q4k_coopmat_tile_matches_f16_ref_on_cpu() {
        let client = cpu_client();
        for (wm, wn, rm, rn) in [(1, 1, 1, 1), (2, 2, 1, 1), (1, 1, 2, 2), (2, 2, 2, 2), (2, 2, 4, 4)] {
            let bm = wm * rm * 16;
            let bn = wn * rn * 16;
            for k in [256usize, 512] {
                let (_xq, _xs, _xsum, wqs, wsc, wd, wdm) = gen_mmq_q4k(bm, bn, k);
                // f32 activation in [-1,1]; the f16 twin rounds it to f16 (unlike the int8 MMQ path).
                let mut s = 0x1234_5678_9ABC_DEF0u64;
                let x: Vec<f32> = (0..bm * k)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s % 2000) as f32 / 1000.0 - 1.0
                    })
                    .collect();
                let (got, _) = mmq_q4k_coopmat_tile_run::<CpuRuntime>(
                    &client, &x, &wqs, &wsc, &wd, &wdm, bm, bn, k, wm, wn, rm, rn, 1,
                );
                let want = mmq_q4k_f16_ref(&x, &wqs, &wsc, &wd, &wdm, bm, bn, k);
                let rel = rel_to_max(&got, &want);
                eprintln!("[f16 coopmat] wm{wm} wn{wn} rm{rm} rn{rn} {bm}x{bn}x{k}: rel_to_max={rel:.3e}");
                assert!(rel < 5e-2, "f16 coopmat wm{wm} wn{wn} rm{rm} rn{rn} {bm}x{bn}x{k}: rel_to_max={rel:.3e}");
            }
        }
    }

    /// The f16-coopmat tile SEARCH on the CPU runtime: the same GA over the same Space as the int8 hunt,
    /// with the f16 evaluator (f32 activation, full-precision oracle, f16 tolerance). Proves the autotuner
    /// drives the f16 twin -- the DSL side of the coopmat prefill board -- end to end offline.
    #[cfg(feature = "cpu")]
    #[test]
    fn mmq_q4k_coopmat_space_search_cpu() {
        let (m, n, k) = (16usize, 16usize, 256usize);
        let (_xq, _xs, _xsum, wqs, wsc, wd, wdm) = gen_mmq_q4k(m, n, k);
        let mut s = 0xABCD_1234u64;
        let x: Vec<f32> = (0..m * k)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect();
        let mut wqs2 = wqs.clone();
        for (i, v) in wqs2.iter_mut().enumerate() {
            *v ^= 0x9E37_79B1u32.wrapping_mul(i as u32 + 1);
        }
        let banks = vec![wqs.clone(), wqs2];
        let space = mmq_q4k_space();
        let client = cpu_client();
        let eval = CoopmatF16Eval::new(&client, &space, &banks, &x, &wsc, &wd, &wdm, m, n, k, 1);
        let dir = std::env::temp_dir().join(format!(
            "hk-coopmat-hunt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let tuner = Tuner::new(&dir);
        let evo = Evolution::new().population(8).generations(3).measure_iters(1);

        let r = mmq_q4k_coopmat_hunt(&tuner, "cpu", m, n, k, &eval, &evo, 0xC0FFEE);
        assert!(!r.from_cache, "first hunt must not be a cache hit");
        let rep = r.report.as_ref().expect("a miss carries the evidence trail");
        assert!(rep.best_ms.is_finite(), "no measurable winner");
        let win = space.parse(&r.winner).expect("winner is a valid config name");
        assert!(space.feasible(&win), "winner {} is infeasible", r.winner);
        assert!(eval.worst_rel() < 5e-2, "a measured tile diverged from the oracle: {:.2e}", eval.worst_rel());
        eprintln!(
            "[coopmat f16 hunt CPU] winner={} evaluated={} measured={} rejected={} worst_rel={:.2e}",
            r.winner, rep.evaluated, rep.measured.len(), rep.rejected.len(), eval.worst_rel()
        );

        let r2 = mmq_q4k_coopmat_hunt(&tuner, "cpu", m, n, k, &eval, &evo, 0xC0FFEE);
        assert!(r2.from_cache && r2.report.is_none(), "second hunt must hit the cache");
        assert_eq!(r2.winner, r.winner, "cache returned a different winner");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The full {WM x WN x RM x RN} tile SEARCH on the CPU runtime: build the space + a cold-stream
    /// evaluator and run the `tune::Evolution` GA. Proves the machinery end to end offline (no GPU) -- the
    /// hunt crowns a bit-exact, feasible winner deterministically, records it, and a second hunt reads the
    /// cache without re-running the GA. Shape m=n=16 keeps the grid 1x1 for every tile in the space (each
    /// tail-guards its oversized block), so the whole search runs in one cube. On a wgpu/vulkan device the
    /// SAME evaluator makes the winner meaningful; the point under test here is the search + correctness gate.
    #[cfg(feature = "cpu")]
    #[test]
    fn mmq_q4k_space_search_cpu() {
        let (m, n, k) = (16usize, 16usize, 256usize);
        let (xq, xs, xsum, wqs, wsc, wd, wdm) = gen_mmq_q4k(m, n, k);
        // Two distinct weight banks so the cold rotation reads different data on each dispatch.
        let mut wqs2 = wqs.clone();
        for (i, v) in wqs2.iter_mut().enumerate() {
            *v ^= 0x9E37_79B1u32.wrapping_mul(i as u32 + 1);
        }
        let banks = vec![wqs.clone(), wqs2];
        let space = mmq_q4k_space();
        let feasible = space.enumerate();
        assert!(!feasible.is_empty(), "empty feasible space");

        let client = cpu_client();
        let eval = MmqQ4kEval::new(&client, &space, &banks, &xq, &xs, &xsum, &wsc, &wd, &wdm, m, n, k, 1);
        let dir = std::env::temp_dir().join(format!(
            "hk-mmq-hunt-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));
        let tuner = Tuner::new(&dir);
        let evo = Evolution::new().population(8).generations(3).measure_iters(1);

        // (1) miss: the hunt runs, crowns a feasible bit-exact winner, and records it.
        let r = mmq_q4k_hunt(&tuner, "cpu", m, n, k, &eval, &evo, 0xC0FFEE);
        assert!(!r.from_cache, "first hunt must not be a cache hit");
        let rep = r.report.as_ref().expect("a miss carries the evidence trail");
        assert!(rep.best_ms.is_finite(), "no measurable winner");
        let win = space.parse(&r.winner).expect("winner is a valid config name");
        assert!(space.feasible(&win), "winner {} is infeasible", r.winner);
        assert!(eval.worst_rel() < 1e-3, "a measured tile diverged from the oracle: {:.2e}", eval.worst_rel());
        eprintln!(
            "[mmq hunt CPU] winner={} evaluated={} measured={} rejected={} worst_rel={:.2e}",
            r.winner,
            rep.evaluated,
            rep.measured.len(),
            rep.rejected.len(),
            eval.worst_rel()
        );

        // (2) hit: a second hunt reads the cached winner and runs no GA.
        let r2 = mmq_q4k_hunt(&tuner, "cpu", m, n, k, &eval, &evo, 0xC0FFEE);
        assert!(r2.from_cache && r2.report.is_none(), "second hunt must hit the cache");
        assert_eq!(r2.winner, r.winner, "cache returned a different winner");
        std::fs::remove_dir_all(&dir).ok();
    }

    /// The autokernel dispatch PREFERS the tuned winner over the fixed default: a cold cache serves the
    /// incumbent tile (bit-exact), and once a hunt populates the shared cache the autokernel replays the
    /// hunted winner (bit-exact). This is the "autotuner-as-default-dispatch" contract on the CPU runtime;
    /// the same call on a wgpu/vulkan device auto-optimizes the real prefill GEMM. The hunt is driven under
    /// the autokernel's OWN device id so the cache keys match (the production dispatch keys the same way).
    #[cfg(feature = "cpu")]
    #[test]
    fn mmq_q4k_autokernel_prefers_cached_genome_cpu() {
        let (m, n, k) = (16usize, 16usize, 256usize);
        let (xq, xs, xsum, wqs, wsc, wd, wdm) = gen_mmq_q4k(m, n, k);
        let want = mmq_q4k_ref(&xq, &xs, &xsum, &wqs, &wsc, &wd, &wdm, m, n, k);
        let client = cpu_client();
        let device = crate::tune::device_id(&client);
        let dir = std::env::temp_dir().join(format!(
            "hk-mmq-auto-{}-{}",
            std::process::id(),
            std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()
        ));

        // (1) cold cache: the autokernel falls back to the incumbent tile, bit-exact.
        let tuner = Tuner::new(&dir);
        let (out0, name0) =
            mmq_q4k_autokernel(&client, &tuner, &xq, &xs, &xsum, &wqs, &wsc, &wd, &wdm, m, n, k);
        assert_eq!(name0, "incumbent", "a cold cache must serve the incumbent tile");
        assert!(rel_to_max(&out0, &want) < 1e-6, "incumbent tile diverged from the oracle");

        // (2) after a hunt populates the cache, the autokernel prefers the tuned winner over the default.
        let space = mmq_q4k_space();
        let banks = vec![wqs.clone()];
        let eval = MmqQ4kEval::new(&client, &space, &banks, &xq, &xs, &xsum, &wsc, &wd, &wdm, m, n, k, 1);
        let evo = Evolution::new().population(8).generations(3).measure_iters(1);
        let r = mmq_q4k_hunt(&tuner, &device, m, n, k, &eval, &evo, 0xC0FFEE);
        assert!(!r.from_cache, "the seeding hunt must actually run");
        assert!(r.report.as_ref().expect("a miss carries the trail").best_ms.is_finite(), "the seeding hunt found no measurable winner");

        let (out1, name1) =
            mmq_q4k_autokernel(&client, &tuner, &xq, &xs, &xsum, &wqs, &wsc, &wd, &wdm, m, n, k);
        assert_eq!(name1, r.winner, "the autokernel must replay the hunted winner, not the incumbent");
        assert_ne!(name1, "incumbent", "the tuned dispatch must differ from the cold-start label");
        assert!(rel_to_max(&out1, &want) < 1e-6, "the tuned tile diverged from the oracle");
        std::fs::remove_dir_all(&dir).ok();
    }
}
