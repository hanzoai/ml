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
}
