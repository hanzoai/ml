//! Flash-attention in the DSL: tiled online softmax, one source -> every backend.
//!
//! This is the online-softmax EVOLUTION of `attn::sdpa_blk`. `sdpa_blk` streams keys one at a time per
//! thread; `flash_attn` streams them in TILES (`Bc` keys at once), maintaining the running max `m`,
//! running denom `l`, and output accumulator `Of[Br][d]` and rescaling on the fly. The [Br, Bc] score
//! tile is the ONLY score state that ever exists -- the `[heads, seq_q, seq_k]` matrix is never
//! materialized. That single property is the #1 transferable lesson from llama.cpp's `flash_attn_ext`:
//! it kills the CUDA prefill's materialized `softmax_f32` (~12x off roofline) + cutlass QK/PV, and
//! shrinks the Vulkan decode's attention cost.
//!
//! The two matmuls (Q@Kᵀ -> scores, P@V -> output) are each an [`island!`]: the accelerated arm issues
//! the f16 cooperative-matrix op (`cmma`, WMMA / OpCooperativeMatrixMulAddKHR / simdgroup matrix), the
//! `default` arm expresses the identical contraction as portable scalar MACs. `default` is what the CPU
//! runtime runs (cubecl-cpu rejects every CoopMma op), which is what makes a bit-exact CPU oracle
//! possible at all -- it gates the tiling, the online-softmax rescale, the shared-memory epilogue and
//! the f32 accumulation order. The cmma arm is equivalent by construction (the algebra is identical;
//! only the intermediate precision differs -- f16 accumulate vs f32) and is gated on-GPU, in a
//! scale-relative tolerance, against the same materialized reference.
//!
//! Tiles are 16x16x16 (`Br = Bc = 16`) to match one cooperative-matrix fragment on CUDA/ROCm/Vulkan;
//! Metal (simdgroup_matrix caps at 8x8x8) composes the same 16x16 tile from a 2x2 grid of 8x8. GQA-native (reads the
//! shared KV head, no `repeat_kv`), batch-aware, causal-optional, and runtime `seq_q`/`seq_k` via a
//! `meta` SSBO (one compiled kernel serves a KV cache that grows every decode step, like `sdpa_blk`).

use crate::prelude::*;
// The cmma operand type. f16 tensor-core inputs accumulate in f32 (the scores/output stay f32); named
// distinctly so it never collides with a prelude `f16`. Only the accelerated island arms reference it.
use half::f16;

/// Query rows per workgroup tile (one cooperative-matrix row dimension).
pub const BR: usize = 16;
/// Key columns streamed per online-softmax step. Parameterized: `BC` is contracted in `BC/16`
/// cooperative-matrix col fragments (the `cg`/`kc` loops in the QK/PV arms), so it is a tunable knob,
/// not baked structure. `BC = 16` (one fragment) is the default: on a shared-memory-tight target,
/// widening to 32 doubles the score/K/V/P shared tiles and trades workgroup occupancy for K/V-stage
/// amortization -- an occupancy loss that is not worth it on an LDS-tight APU. Tune per target.
pub const BC: usize = 16;

/// Tiled flash attention. Grid = one cube per `(batch, head, query-tile)`; block = one plane of `plane`
/// threads. `d` (head dim), `br`/`bc` (tile), `plane` are comptime; `seq_q`/`seq_k`/`n_heads`/`n_kv`/
/// `causal` ride the runtime `meta` SSBO so one compiled kernel serves any (growing) sequence length.
///
/// `meta = [seq_q, seq_k, n_heads, n_kv, causal, kv_batch_stride, kv_head_stride, key_stride]` -- the
/// same shape as `sdpa_blk`, so the KV cache is read in place at its real (padded) strides with no
/// defensive `.contiguous()`. `d` is always the innermost contiguous dim (element stride 1).
#[allow(clippy::too_many_arguments)]
#[kernel(targets(cuda, rocm, vulkan, metal, cpu), unchecked)]
pub fn flash_attn<F: Float>(
    q: &Array<F>,
    k: &Array<F>,
    v: &Array<F>,
    out: &mut Array<F>,
    scale: &Array<F>,
    meta: &Array<u32>,
    #[comptime] d: usize,
    #[comptime] br: usize,
    #[comptime] bc: usize,
    #[comptime] plane: usize,
    #[comptime] target: Target,
) {
    // `cube_base` (meta[8]) offsets the flat cube index. Production launches the whole grid with
    // cube_base = 0; the CPU oracle launches one cube at a time (cubes are fully independent) with
    // cube_base = n, which sidesteps cubecl-cpu's cross-cube SharedMemory aliasing without changing a
    // single result -- one compiled kernel, faithful to the full-grid launch.
    let cube = CUBE_POS as usize + meta[8] as usize; // (batch, head, query-tile), row-major
    let lane = UNIT_POS as usize; // 0..plane-1
    let sc = scale[0];

    let seq_q = meta[0] as usize;
    let seq_k = meta[1] as usize;
    let n_heads = meta[2] as usize;
    let n_kv = meta[3] as usize;
    let causal = meta[4];
    let kv_batch_stride = meta[5] as usize;
    let kv_head_stride = meta[6] as usize;
    let key_stride = meta[7] as usize;

    // Decompose the flat cube index into (batch, head, query-tile). GQA maps head -> shared kv head.
    let n_qtiles = (seq_q + br - 1) / br;
    let tiles_per_head = n_qtiles;
    let hq = n_heads * tiles_per_head;
    let b_i = cube / hq;
    let rem = cube - b_i * hq;
    let h = rem / tiles_per_head;
    let qt = rem % tiles_per_head;
    let kv = h / (n_heads / n_kv);
    let q_head_base = ((b_i * n_heads + h) * seq_q) * d; // q/out are [b, n_heads, seq_q, d] contiguous
    let kvbase = b_i * kv_batch_stride + kv * kv_head_stride; // k/v read in place at their real strides

    // Per-lane element partitions of the shared tiles.
    let per_s = br * bc / plane; // score-tile elements this lane owns
    let per_o = br * d / plane; // output-tile elements this lane owns
    let per_q = br * d / plane; // Q-tile elements this lane owns (staged once, below)

    // Shared state: query tile [br, d] (staged ONCE), score/prob tile [br, bc], output accumulator
    // [br, d], and per-row (m, l, rescale). `qsh` is the f16 Q the coopmat arms read every key tile;
    // the default (CPU-oracle) arm reads Q from global directly and never touches it.
    let mut qsh = SharedMemory::<f16>::new(br * d);
    let mut sf = SharedMemory::<F>::new(br * bc);
    let mut of = SharedMemory::<F>::new(br * d);
    let mut mf = SharedMemory::<F>::new(br);
    let mut lf = SharedMemory::<F>::new(br);
    let mut emf = SharedMemory::<F>::new(br); // per-row rescale exp(m_old - m_new) for this tile

    // Init the output accumulator and per-row running state.
    for e in 0..per_o {
        of[lane * per_o + e] = F::new(0.0f32);
    }
    if lane < br {
        mf[lane] = F::new(-3.4e38f32);
        lf[lane] = F::new(0.0f32);
    }

    // Stage Q[br, d] from global f32 to f16 shared ONCE, before the key loop (llama Qf,
    // flash_attn_cm1.comp:87). Ragged query rows (qpos >= seq_q) stage 0. The coopmat arms below read
    // this every key tile; hoisting it out of the loop removes the per-tile Q re-stage that capped the
    // kernel at 0.38 TFLOP/s. Inert for the CPU oracle (its default arm reads Q from global).
    for e in 0..per_q {
        let idx = lane * per_q + e;
        let r = idx / d;
        let dd = idx % d;
        let qpos = qt * br + r;
        let val = if qpos < seq_q {
            q[q_head_base + qpos * d + dd]
        } else {
            F::new(0.0f32)
        };
        qsh[idx] = f16::cast_from(val);
    }
    sync_cube();

    // Causal early-exit: a key tile whose first key (j*bc) already exceeds this query tile's last row
    // (qt*br + br - 1) is entirely above the causal diagonal -> every score is -inf -> P = 0 and the
    // running (m, l, O) are unchanged (a mathematical no-op). Bounding the loop to the diagonal tile is
    // therefore bit-exact and drops ~half the key tiles on causal prefill -- llama's masked-block skip
    // (flash_attn_cm1.comp:146), expressed directly since our causality is computed inline, not masked.
    let n_jt = (seq_k + bc - 1) / bc;
    let n_jt_eff = if causal == 1 {
        let bound = (qt * br + br - 1) / bc + 1;
        if bound < n_jt {
            bound
        } else {
            n_jt
        }
    } else {
        n_jt
    };
    for j in 0..n_jt_eff {
        // ---- Q@Kᵀ: fill the [br, bc] score tile. Out-of-range / causally-masked entries -> -inf. ----
        island! {
            // Accelerated: f16 cooperative-matrix. Stage Q[br,d]/K[bc,d] tiles to f16 shared (bounds
            // -> 0), then C[br,bc] = Q @ Kᵀ over d/16 cmma(16x16x16) chunks (K loaded ColMajor gives
            // Kᵀ, exactly `mmq_q8_wmma`'s W layout), f16 in -> f32 accumulate. Store to `sf`, then the
            // scale+mask pass matches the default arm's semantics. Lowers to WMMA (CUDA/ROCm) +
            // OpCooperativeMatrixMulAddKHR (SPIR-V), one 16x16x16 fragment. Metal composes it from 8x8
            // (its simdgroup_matrix caps at 8x8x8) in the sibling `metal =>` arm below.
            cuda | rocm | vulkan => {
                // Q is already in `qsh` (staged once, above). Stage only this key tile's K[bc,d].
                let mut ksh = SharedMemory::<f16>::new(bc * d);
                let per_k = bc * d / plane;
                for e in 0..per_k {
                    let idx = lane * per_k + e;
                    let c = idx / d;
                    let dd = idx % d;
                    let kpos = j * bc + c;
                    let val = if kpos < seq_k { k[kvbase + kpos * key_stride + dd] } else { F::new(0.0f32) };
                    ksh[idx] = f16::cast_from(val);
                }
                sync_cube();
                // The bc score columns split into bc/16 cooperative-matrix col-groups. Each accumulates
                // C[br,16] = Q @ Kᵀ over d/16 contraction chunks (K ColMajor -> Kᵀ, `mmq_q8_wmma`'s W
                // layout), then stores to sf columns [cg*16, cg*16+16). Bc=32 amortizes the K stage +
                // barrier over twice the keys per online-softmax step (llama base Bc=32).
                for cg in 0..bc / 16usize {
                    let cacc = cmma::Matrix::<F>::from_value(
                        cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, F::new(0.0f32),
                    );
                    for dk in 0..d / 16usize {
                        let a = cmma::Matrix::<f16>::from_slice(
                            cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &qsh.to_slice().slice(dk * 16usize, br * d), d as u32,
                        );
                        let b = cmma::Matrix::<f16>::from_slice(
                            cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                            &ksh.to_slice().slice(cg * 16usize * d + dk * 16usize, bc * d), d as u32,
                        );
                        cmma::execute::<f16, f16, F, F>(&a, &b, &cacc, &cacc);
                    }
                    cmma::store(&mut sf.to_slice_mut().slice_mut(cg * 16usize, br * bc), &cacc, bc as u32, cmma::MatrixLayout::RowMajor);
                }
                sync_cube();
                for e in 0..per_s {
                    let idx = lane * per_s + e;
                    let r = idx / bc;
                    let c = idx % bc;
                    let qpos = qt * br + r;
                    let kpos = j * bc + c;
                    let masked = causal == 1 && kpos > qpos;
                    if qpos < seq_q && kpos < seq_k && !masked {
                        sf[idx] = sf[idx] * sc;
                    } else {
                        sf[idx] = F::new(-3.4e38f32);
                    }
                }
            }
            // Metal simdgroup_matrix caps fragments at 8x8x8 (cubecl-cpp metal dialect). Stage the SAME
            // f16 Q[br,d]/K[bc,d] tiles, then compose the 16x16 score tile as a (br/8)x(bc/8) grid of
            // 8x8 accumulators, each contracting d in 8-wide steps. K is loaded ColMajor -> Kᵀ (the
            // 8x8-probe-proven transpose mechanic). Every load reads inside the zero-padded staged tile,
            // so ragged seq is handled at staging exactly as the 16x16 arm -- no OOB direct-slice read.
            metal => {
                // Q is already in `qsh` (staged once, above). Stage only this key tile's K[bc,d].
                let mut ksh = SharedMemory::<f16>::new(bc * d);
                let per_k = bc * d / plane;
                for e in 0..per_k {
                    let idx = lane * per_k + e;
                    let c = idx / d;
                    let dd = idx % d;
                    let kpos = j * bc + c;
                    let val = if kpos < seq_k { k[kvbase + kpos * key_stride + dd] } else { F::new(0.0f32) };
                    ksh[idx] = f16::cast_from(val);
                }
                sync_cube();
                for ti in 0..br / 8usize {
                    for tj in 0..bc / 8usize {
                        let cacc = cmma::Matrix::<F>::from_value(
                            cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, F::new(0.0f32),
                        );
                        for dk in 0..d / 8usize {
                            let a = cmma::Matrix::<f16>::from_slice(
                                cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &qsh.to_slice().slice(ti * 8usize * d + dk * 8usize, br * d), d as u32,
                            );
                            let b = cmma::Matrix::<f16>::from_slice(
                                cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::ColMajor,
                                &ksh.to_slice().slice(tj * 8usize * d + dk * 8usize, bc * d), d as u32,
                            );
                            cmma::execute::<f16, f16, F, F>(&a, &b, &cacc, &cacc);
                        }
                        cmma::store(
                            &mut sf.to_slice_mut().slice_mut(ti * 8usize * bc + tj * 8usize, br * bc),
                            &cacc, bc as u32, cmma::MatrixLayout::RowMajor,
                        );
                    }
                }
                sync_cube();
                for e in 0..per_s {
                    let idx = lane * per_s + e;
                    let r = idx / bc;
                    let c = idx % bc;
                    let qpos = qt * br + r;
                    let kpos = j * bc + c;
                    let masked = causal == 1 && kpos > qpos;
                    if qpos < seq_q && kpos < seq_k && !masked {
                        sf[idx] = sf[idx] * sc;
                    } else {
                        sf[idx] = F::new(-3.4e38f32);
                    }
                }
            }
            // NORMATIVE oracle: the identical Q@Kᵀ as portable scalar MACs, bounds/mask applied inline.
            // This is the arm the CPU runtime executes -- the bit-exact gate for the whole structure.
            default => {
                for e in 0..per_s {
                    let idx = lane * per_s + e;
                    let r = idx / bc;
                    let c = idx % bc;
                    let qpos = qt * br + r;
                    let kpos = j * bc + c;
                    let masked = causal == 1 && kpos > qpos;
                    if qpos < seq_q && kpos < seq_k && !masked {
                        let qb = q_head_base + qpos * d;
                        let kb = kvbase + kpos * key_stride;
                        let mut acc = F::new(0.0f32);
                        for dd in 0..d {
                            acc += q[qb + dd] * k[kb + dd];
                        }
                        sf[idx] = acc * sc;
                    } else {
                        sf[idx] = F::new(-3.4e38f32);
                    }
                }
            }
        };
        sync_cube();

        // ---- Online softmax over this tile: update per-row (m, l) and write P = exp(S - m) in place. ----
        if lane < br {
            let r = lane;
            let qpos = qt * br + r;
            if qpos < seq_q {
                let mut rowmax = F::new(-3.4e38f32);
                for c in 0..bc {
                    let s = sf[r * bc + c];
                    if s > rowmax {
                        rowmax = s;
                    }
                }
                let m_old = mf[r];
                let mut m_new = m_old;
                if rowmax > m_new {
                    m_new = rowmax;
                }
                let em = (m_old - m_new).exp();
                let mut psum = F::new(0.0f32);
                for c in 0..bc {
                    let p = (sf[r * bc + c] - m_new).exp();
                    sf[r * bc + c] = p;
                    psum += p;
                }
                lf[r] = lf[r] * em + psum;
                mf[r] = m_new;
                emf[r] = em;
            } else {
                emf[r] = F::new(1.0f32);
            }
        }
        sync_cube();

        // ---- P@V: rescale the running output by exp(m_old - m_new), then add this tile's P@V. ----
        island! {
            // Accelerated: f16 cooperative-matrix. Stage P[br,bc] (from `sf`) and V[bc,d] to f16 shared
            // (invalid rows / out-of-range keys -> 0), then O_tile[br,d] = P @ V over d/16 cmma chunks
            // (K-dim = bc = 16, one chunk) into `ovt`, then rescale-combine of = emf*of + O_tile. Causal
            // masking rides in P (masked entries are 0 after the online softmax). Metal composes the
            // same tile from 8x8 fragments in the sibling `metal =>` arm below.
            cuda | rocm | vulkan => {
                let mut psh = SharedMemory::<f16>::new(br * bc);
                let mut vsh = SharedMemory::<f16>::new(bc * d);
                let mut ovt = SharedMemory::<F>::new(br * d);
                let per_p = br * bc / plane;
                let per_v = bc * d / plane;
                for e in 0..per_p {
                    let idx = lane * per_p + e;
                    let r = idx / bc;
                    let qpos = qt * br + r;
                    let val = if qpos < seq_q { sf[idx] } else { F::new(0.0f32) };
                    psh[idx] = f16::cast_from(val);
                }
                for e in 0..per_v {
                    let idx = lane * per_v + e;
                    let c = idx / d;
                    let dd = idx % d;
                    let kpos = j * bc + c;
                    let val = if kpos < seq_k { v[kvbase + kpos * key_stride + dd] } else { F::new(0.0f32) };
                    vsh[idx] = f16::cast_from(val);
                }
                sync_cube();
                // O_tile[br,d] = P @ V. The d output columns split into d/16 groups; each accumulates
                // over the bc/16 contraction chunks of the key dimension (P[br,kc*16..] @ V[kc*16..,dn*16..]),
                // then stores O_tile columns [dn*16, dn*16+16). Bc=32 -> two contraction chunks per group.
                for dn in 0..d / 16usize {
                    let cacc = cmma::Matrix::<F>::from_value(
                        cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, F::new(0.0f32),
                    );
                    for kc in 0..bc / 16usize {
                        let a = cmma::Matrix::<f16>::from_slice(
                            cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &psh.to_slice().slice(kc * 16usize, br * bc), bc as u32,
                        );
                        let b = cmma::Matrix::<f16>::from_slice(
                            cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &vsh.to_slice().slice(kc * 16usize * d + dn * 16usize, bc * d), d as u32,
                        );
                        cmma::execute::<f16, f16, F, F>(&a, &b, &cacc, &cacc);
                    }
                    cmma::store(&mut ovt.to_slice_mut().slice_mut(dn * 16usize, br * d), &cacc, d as u32, cmma::MatrixLayout::RowMajor);
                }
                sync_cube();
                for e in 0..per_o {
                    let idx = lane * per_o + e;
                    let r = idx / d;
                    of[idx] = of[idx] * emf[r] + ovt[idx];
                }
            }
            // Metal 8x8x8 simdgroup_matrix. Stage the SAME P[br,bc]/V[bc,d] f16 tiles, then compose the
            // 16xd context tile as a (br/8)x(d/8) grid of 8x8 accumulators, each contracting bc in
            // 8-wide steps (V loaded RowMajor; P is the online-softmax probabilities from `sf`). The
            // rescale-combine of = emf*of + O_tile is identical to the 16x16 arm.
            metal => {
                let mut psh = SharedMemory::<f16>::new(br * bc);
                let mut vsh = SharedMemory::<f16>::new(bc * d);
                let mut ovt = SharedMemory::<F>::new(br * d);
                let per_p = br * bc / plane;
                let per_v = bc * d / plane;
                for e in 0..per_p {
                    let idx = lane * per_p + e;
                    let r = idx / bc;
                    let qpos = qt * br + r;
                    let val = if qpos < seq_q { sf[idx] } else { F::new(0.0f32) };
                    psh[idx] = f16::cast_from(val);
                }
                for e in 0..per_v {
                    let idx = lane * per_v + e;
                    let c = idx / d;
                    let dd = idx % d;
                    let kpos = j * bc + c;
                    let val = if kpos < seq_k { v[kvbase + kpos * key_stride + dd] } else { F::new(0.0f32) };
                    vsh[idx] = f16::cast_from(val);
                }
                sync_cube();
                for ti in 0..br / 8usize {
                    for tn in 0..d / 8usize {
                        let cacc = cmma::Matrix::<F>::from_value(
                            cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, F::new(0.0f32),
                        );
                        for k8 in 0..bc / 8usize {
                            let a = cmma::Matrix::<f16>::from_slice(
                                cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &psh.to_slice().slice(ti * 8usize * bc + k8 * 8usize, br * bc), bc as u32,
                            );
                            let b = cmma::Matrix::<f16>::from_slice(
                                cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &vsh.to_slice().slice(k8 * 8usize * d + tn * 8usize, bc * d), d as u32,
                            );
                            cmma::execute::<f16, f16, F, F>(&a, &b, &cacc, &cacc);
                        }
                        cmma::store(
                            &mut ovt.to_slice_mut().slice_mut(ti * 8usize * d + tn * 8usize, br * d),
                            &cacc, d as u32, cmma::MatrixLayout::RowMajor,
                        );
                    }
                }
                sync_cube();
                for e in 0..per_o {
                    let idx = lane * per_o + e;
                    let r = idx / d;
                    of[idx] = of[idx] * emf[r] + ovt[idx];
                }
            }
            // NORMATIVE oracle: the identical rescale + P@V as portable scalar MACs. The arm the CPU
            // runtime executes -- gating the online rescale and the f32 accumulation order.
            default => {
                for e in 0..per_o {
                    let idx = lane * per_o + e;
                    let r = idx / d;
                    let dd = idx % d;
                    let mut o = of[idx] * emf[r];
                    for c in 0..bc {
                        let kpos = j * bc + c;
                        if kpos < seq_k {
                            let kb = kvbase + kpos * key_stride;
                            o += sf[r * bc + c] * v[kb + dd];
                        }
                    }
                    of[idx] = o;
                }
            }
        };
        sync_cube();
    }

    // ---- Normalize: out = Of / l. Only rows inside seq_q are written. ----
    for e in 0..per_o {
        let idx = lane * per_o + e;
        let r = idx / d;
        let dd = idx % d;
        let qpos = qt * br + r;
        if qpos < seq_q {
            out[q_head_base + qpos * d + dd] = of[idx] / lf[r];
        }
    }
}

/// Number of cubes a full launch dispatches for `(b, n_heads, seq_q)`: one per `(batch, head, qtile)`.
pub fn flash_attn_cubes(b: usize, n_heads: usize, seq_q: usize) -> usize {
    b * n_heads * seq_q.div_ceil(BR)
}

/// Number of cubes for [`flash_attn_paged`] with tile height `br`.
pub fn flash_attn_paged_cubes(b: usize, n_heads: usize, seq_q: usize, br: usize) -> usize {
    b * n_heads * seq_q.div_ceil(br)
}

/// Read one K element from either:
/// - `kv_layout == 0`: flat 4D paged `[num_blocks, n_kv, block_size, d]`
/// - `kv_layout == 1`: Engine Standard 5D paged `[num_blocks, n_kv, d/x, block_size, x]`
/// - `kv_layout == 2`: contiguous `[batch, n_kv, kv_seq_stride, d]` (`kv_seq_stride = block_size`)
#[device]
#[inline]
#[allow(clippy::too_many_arguments)]
fn load_paged_k<F: Float>(
    k_cache: &Array<F>,
    block_tables: &Array<u32>,
    b_i: usize,
    kv_head: usize,
    kpos: usize,
    dd: usize,
    n_kv: usize,
    block_size: usize,
    max_blocks: usize,
    kv_layout: usize,
    x_vec: usize,
    #[comptime] d: usize,
) -> F {
    if kv_layout == 2usize {
        k_cache[((b_i * n_kv + kv_head) * block_size + kpos) * d + dd]
    } else {
        let blk = block_tables[b_i * max_blocks + kpos / block_size] as usize;
        let off = kpos % block_size;
        if kv_layout == 0usize {
            k_cache[((blk * n_kv + kv_head) * block_size + off) * d + dd]
        } else {
            let dx = d / x_vec;
            let k_idx = ((((blk * n_kv + kv_head) * dx + (dd / x_vec)) * block_size + off) * x_vec)
                + (dd % x_vec);
            k_cache[k_idx]
        }
    }
}

/// Read one V element from either:
/// - `kv_layout == 0`: flat 4D paged `[num_blocks, n_kv, block_size, d]`
/// - `kv_layout == 1`: Engine Standard 4D paged `[num_blocks, n_kv, d, block_size]`
/// - `kv_layout == 2`: contiguous `[batch, n_kv, kv_seq_stride, d]` (`kv_seq_stride = block_size`)
#[device]
#[inline]
#[allow(clippy::too_many_arguments)]
fn load_paged_v<F: Float>(
    v_cache: &Array<F>,
    block_tables: &Array<u32>,
    b_i: usize,
    kv_head: usize,
    kpos: usize,
    dd: usize,
    n_kv: usize,
    block_size: usize,
    max_blocks: usize,
    kv_layout: usize,
    #[comptime] d: usize,
) -> F {
    if kv_layout == 2usize {
        v_cache[((b_i * n_kv + kv_head) * block_size + kpos) * d + dd]
    } else {
        let blk = block_tables[b_i * max_blocks + kpos / block_size] as usize;
        let off = kpos % block_size;
        if kv_layout == 0usize {
            v_cache[((blk * n_kv + kv_head) * block_size + off) * d + dd]
        } else {
            let v_idx = (((blk * n_kv + kv_head) * d + dd) * block_size) + off;
            v_cache[v_idx]
        }
    }
}

/// Production paged & contiguous FlashAttention in the canonical DSL (`F = f32 | f16 | bf16`).
///
/// Supports:
/// - Noncontiguous paged KV caches via per-request `block_tables: [B, max_blocks]` and variable
///   `context_lens: [B]` (both flat 4D paged `kv_layout=0` and Engine Standard 5D K / 4D V
///   `kv_layout=1`), plus contiguous KV (`kv_layout=2`),
/// - Native GQA head sharing (`n_heads / n_kv`),
/// - Causal masking with automatic chunked-prefill / decode offset (`q_offset = ctx_len - seq_q`)
///   or explicit `q_offset` (`causal == 2`),
/// - Optional sliding window (`sliding_window > 0`) and logit softcapping (`softcap > 0.0`),
/// - `d = 64, 128, 256` using compact tile-wise PV accumulation (`ovt[br * 16]`) to stay under
///   Metal's 32 KB threadgroup memory ceiling.
///
/// `scale_cfg = [scale, softcap]`.
/// `meta = [seq_q, n_heads, n_kv, causal, block_size, max_blocks, kv_layout, x_vec,
///          sliding_window, q_offset_override, cube_base, 0]`.
#[allow(clippy::too_many_arguments)]
#[kernel(targets(cuda, rocm, vulkan, metal, cpu), unchecked)]
pub fn flash_attn_paged<F: Float>(
    q: &Array<F>,
    k_cache: &Array<F>,
    v_cache: &Array<F>,
    block_tables: &Array<u32>,
    context_lens: &Array<u32>,
    out: &mut Array<F>,
    scale_cfg: &Array<f32>,
    meta: &Array<u32>,
    #[comptime] d: usize,
    #[comptime] br: usize,
    #[comptime] bc: usize,
    #[comptime] plane: usize,
    #[comptime] target: Target,
) {
    let cube = CUBE_POS as usize + meta[10] as usize;
    let lane = UNIT_POS as usize;
    let sc = scale_cfg[0];
    let softcap = scale_cfg[1];

    let seq_q = meta[0] as usize;
    let n_heads = meta[1] as usize;
    let n_kv = meta[2] as usize;
    let causal = meta[3];
    let block_size = meta[4] as usize;
    let max_blocks = meta[5] as usize;
    let kv_layout = meta[6] as usize;
    let x_vec = meta[7] as usize;
    let sliding_window = meta[8] as usize;
    let q_offset_override = meta[9] as usize;

    let n_qtiles = (seq_q + br - 1usize) / br;
    let hq = n_heads * n_qtiles;
    let b_i = cube / hq;
    let rem = cube - b_i * hq;
    let h = rem / n_qtiles;
    let qt = rem % n_qtiles;
    let kv = h / (n_heads / n_kv);
    let ctx_len = context_lens[b_i] as usize;
    let mut q_offset = 0usize;
    if causal == 2u32 {
        q_offset = q_offset_override;
    } else if ctx_len >= seq_q {
        q_offset = ctx_len - seq_q;
    }
    let q_head_base = ((b_i * n_heads + h) * seq_q) * d;

    let per_s = br * bc / plane;
    let per_o = br * d / plane;
    let per_q = br * d / plane;

    let mut qsh = SharedMemory::<f16>::new(br * d);
    let mut sf = SharedMemory::<f32>::new(br * bc);
    let mut of = SharedMemory::<f32>::new(br * d);
    let mut mf = SharedMemory::<f32>::new(br);
    let mut lf = SharedMemory::<f32>::new(br);
    let mut emf = SharedMemory::<f32>::new(br);

    for e in 0..per_o {
        of[lane * per_o + e] = 0.0f32;
    }
    if lane < br {
        mf[lane] = -3.4e38f32;
        lf[lane] = 0.0f32;
    }

    for e in 0..per_q {
        let idx = lane * per_q + e;
        let r = idx / d;
        let dd = idx % d;
        let qpos = qt * br + r;
        let mut val = 0.0f32;
        if qpos < seq_q {
            val = f32::cast_from(q[q_head_base + qpos * d + dd]);
        }
        qsh[idx] = f16::cast_from(val);
    }
    sync_cube();

    let n_jt = (ctx_len + bc - 1usize) / bc;
    let mut j_end = n_jt;
    if causal != 0u32 {
        let max_q_abs = q_offset + qt * br + br - 1usize;
        let bound = max_q_abs / bc + 1usize;
        if bound < j_end {
            j_end = bound;
        }
    }
    let mut j_start = 0usize;
    if sliding_window > 0usize {
        let min_q_abs = q_offset + qt * br;
        if min_q_abs >= sliding_window {
            j_start = (min_q_abs - sliding_window + 1usize) / bc;
        }
    }

    for j in j_start..j_end {
        island! {
            cuda | rocm | vulkan => {
                let mut ksh = SharedMemory::<f16>::new(bc * d);
                let per_k = bc * d / plane;
                for e in 0..per_k {
                    let idx = lane * per_k + e;
                    let c = idx / d;
                    let dd = idx % d;
                    let kpos = j * bc + c;
                    let mut val = 0.0f32;
                    if kpos < ctx_len {
                        val = f32::cast_from(load_paged_k(
                            k_cache, block_tables, b_i, kv, kpos, dd, n_kv,
                            block_size, max_blocks, kv_layout, x_vec, d,
                        ));
                    }
                    ksh[idx] = f16::cast_from(val);
                }
                sync_cube();
                for cg in 0..bc / 16usize {
                    let cacc = cmma::Matrix::<f32>::from_value(
                        cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32,
                    );
                    for dk in 0..d / 16usize {
                        let a = cmma::Matrix::<f16>::from_slice(
                            cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &qsh.to_slice().slice(dk * 16usize, br * d), d as u32,
                        );
                        let b = cmma::Matrix::<f16>::from_slice(
                            cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                            &ksh.to_slice().slice(cg * 16usize * d + dk * 16usize, bc * d), d as u32,
                        );
                        cmma::execute::<f16, f16, f32, f32>(&a, &b, &cacc, &cacc);
                    }
                    cmma::store(&mut sf.to_slice_mut().slice_mut(cg * 16usize, br * bc), &cacc, bc as u32, cmma::MatrixLayout::RowMajor);
                }
                sync_cube();
                for e in 0..per_s {
                    let idx = lane * per_s + e;
                    let r = idx / bc;
                    let c = idx % bc;
                    let qpos = qt * br + r;
                    let kpos = j * bc + c;
                    let q_abs = q_offset + qpos;
                    let causal_ok = causal == 0u32 || kpos <= q_abs;
                    let win_ok = sliding_window == 0usize || kpos + sliding_window > q_abs;
                    if qpos < seq_q && kpos < ctx_len && causal_ok && win_ok {
                        let mut s = sf[idx] * sc;
                        if softcap > 0.0f32 {
                            s = softcap * (s / softcap).tanh();
                        }
                        sf[idx] = s;
                    } else {
                        sf[idx] = -3.4e38f32;
                    }
                }
            }
            metal => {
                let mut ksh = SharedMemory::<f16>::new(bc * d);
                let per_k = bc * d / plane;
                for e in 0..per_k {
                    let idx = lane * per_k + e;
                    let c = idx / d;
                    let dd = idx % d;
                    let kpos = j * bc + c;
                    let mut val = 0.0f32;
                    if kpos < ctx_len {
                        val = f32::cast_from(load_paged_k(
                            k_cache, block_tables, b_i, kv, kpos, dd, n_kv,
                            block_size, max_blocks, kv_layout, x_vec, d,
                        ));
                    }
                    ksh[idx] = f16::cast_from(val);
                }
                sync_cube();
                for ti in 0..br / 8usize {
                    for tj in 0..bc / 8usize {
                        let cacc = cmma::Matrix::<f32>::from_value(
                            cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, 0.0f32,
                        );
                        for dk in 0..d / 8usize {
                            let a = cmma::Matrix::<f16>::from_slice(
                                cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &qsh.to_slice().slice(ti * 8usize * d + dk * 8usize, br * d), d as u32,
                            );
                            let b = cmma::Matrix::<f16>::from_slice(
                                cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::ColMajor,
                                &ksh.to_slice().slice(tj * 8usize * d + dk * 8usize, bc * d), d as u32,
                            );
                            cmma::execute::<f16, f16, f32, f32>(&a, &b, &cacc, &cacc);
                        }
                        cmma::store(
                            &mut sf.to_slice_mut().slice_mut(ti * 8usize * bc + tj * 8usize, br * bc),
                            &cacc, bc as u32, cmma::MatrixLayout::RowMajor,
                        );
                    }
                }
                sync_cube();
                for e in 0..per_s {
                    let idx = lane * per_s + e;
                    let r = idx / bc;
                    let c = idx % bc;
                    let qpos = qt * br + r;
                    let kpos = j * bc + c;
                    let q_abs = q_offset + qpos;
                    let causal_ok = causal == 0u32 || kpos <= q_abs;
                    let win_ok = sliding_window == 0usize || kpos + sliding_window > q_abs;
                    if qpos < seq_q && kpos < ctx_len && causal_ok && win_ok {
                        let mut s = sf[idx] * sc;
                        if softcap > 0.0f32 {
                            s = softcap * (s / softcap).tanh();
                        }
                        sf[idx] = s;
                    } else {
                        sf[idx] = -3.4e38f32;
                    }
                }
            }
            default => {
                for e in 0..per_s {
                    let idx = lane * per_s + e;
                    let r = idx / bc;
                    let c = idx % bc;
                    let qpos = qt * br + r;
                    let kpos = j * bc + c;
                    let q_abs = q_offset + qpos;
                    let causal_ok = causal == 0u32 || kpos <= q_abs;
                    let win_ok = sliding_window == 0usize || kpos + sliding_window > q_abs;
                    if qpos < seq_q && kpos < ctx_len && causal_ok && win_ok {
                        let qb = q_head_base + qpos * d;
                        let mut acc = 0.0f32;
                        for dd in 0..d {
                            let kv_val = f32::cast_from(load_paged_k(
                                k_cache, block_tables, b_i, kv, kpos, dd, n_kv,
                                block_size, max_blocks, kv_layout, x_vec, d,
                            ));
                            acc += f32::cast_from(q[qb + dd]) * kv_val;
                        }
                        let mut s = acc * sc;
                        if softcap > 0.0f32 {
                            s = softcap * (s / softcap).tanh();
                        }
                        sf[idx] = s;
                    } else {
                        sf[idx] = -3.4e38f32;
                    }
                }
            }
        };
        sync_cube();

        // Online softmax over this tile.
        if lane < br {
            let r = lane;
            let qpos = qt * br + r;
            if qpos < seq_q {
                let mut rowmax = f32::new(-3.4e38f32);
                for c in 0..bc {
                    let s = sf[r * bc + c];
                    if s > rowmax {
                        rowmax = s;
                    }
                }
                let m_old = mf[r];
                let mut m_new = m_old;
                if rowmax > m_new {
                    m_new = rowmax;
                }
                let em = (m_old - m_new).exp();
                let mut psum = 0.0f32;
                for c in 0..bc {
                    let p = (sf[r * bc + c] - m_new).exp();
                    sf[r * bc + c] = p;
                    psum += p;
                }
                lf[r] = lf[r] * em + psum;
                mf[r] = m_new;
                emf[r] = em;
            } else {
                emf[r] = 1.0f32;
            }
        }
        sync_cube();

        // P @ V with compact `ovt[br * 16]` tile scratch so `d = 64, 128, 256` fit under 32 KB.
        island! {
            cuda | rocm | vulkan => {
                let mut psh = SharedMemory::<f16>::new(br * bc);
                let mut vsh = SharedMemory::<f16>::new(bc * d);
                let mut ovt = SharedMemory::<f32>::new(br * 16usize);
                let per_p = br * bc / plane;
                let per_v = bc * d / plane;
                let per_tile = br * 16usize / plane;
                for e in 0..per_p {
                    let idx = lane * per_p + e;
                    let r = idx / bc;
                    let qpos = qt * br + r;
                    let mut val = 0.0f32;
                    if qpos < seq_q {
                        val = sf[idx];
                    }
                    psh[idx] = f16::cast_from(val);
                }
                for e in 0..per_v {
                    let idx = lane * per_v + e;
                    let c = idx / d;
                    let dd = idx % d;
                    let kpos = j * bc + c;
                    let mut val = 0.0f32;
                    if kpos < ctx_len {
                        val = f32::cast_from(load_paged_v(
                            v_cache, block_tables, b_i, kv, kpos, dd, n_kv,
                            block_size, max_blocks, kv_layout, d,
                        ));
                    }
                    vsh[idx] = f16::cast_from(val);
                }
                sync_cube();
                for dn in 0..d / 16usize {
                    let cacc = cmma::Matrix::<f32>::from_value(
                        cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32,
                    );
                    for kc in 0..bc / 16usize {
                        let a = cmma::Matrix::<f16>::from_slice(
                            cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &psh.to_slice().slice(kc * 16usize, br * bc), bc as u32,
                        );
                        let b = cmma::Matrix::<f16>::from_slice(
                            cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &vsh.to_slice().slice(kc * 16usize * d + dn * 16usize, bc * d), d as u32,
                        );
                        cmma::execute::<f16, f16, f32, f32>(&a, &b, &cacc, &cacc);
                    }
                    cmma::store(&mut ovt.to_slice_mut(), &cacc, 16u32, cmma::MatrixLayout::RowMajor);
                    sync_cube();
                    for e in 0..per_tile {
                        let tidx = lane * per_tile + e;
                        let r = tidx / 16usize;
                        let col = tidx % 16usize;
                        let oidx = r * d + dn * 16usize + col;
                        of[oidx] = of[oidx] * emf[r] + ovt[tidx];
                    }
                    sync_cube();
                }
            }
            metal => {
                let mut psh = SharedMemory::<f16>::new(br * bc);
                let mut vsh = SharedMemory::<f16>::new(bc * d);
                let mut ovt = SharedMemory::<f32>::new(br * 16usize);
                let per_p = br * bc / plane;
                let per_v = bc * d / plane;
                let per_tile = br * 16usize / plane;
                for e in 0..per_p {
                    let idx = lane * per_p + e;
                    let r = idx / bc;
                    let qpos = qt * br + r;
                    let mut val = 0.0f32;
                    if qpos < seq_q {
                        val = sf[idx];
                    }
                    psh[idx] = f16::cast_from(val);
                }
                for e in 0..per_v {
                    let idx = lane * per_v + e;
                    let c = idx / d;
                    let dd = idx % d;
                    let kpos = j * bc + c;
                    let mut val = 0.0f32;
                    if kpos < ctx_len {
                        val = f32::cast_from(load_paged_v(
                            v_cache, block_tables, b_i, kv, kpos, dd, n_kv,
                            block_size, max_blocks, kv_layout, d,
                        ));
                    }
                    vsh[idx] = f16::cast_from(val);
                }
                sync_cube();
                for dn in 0..d / 16usize {
                    for ti in 0..br / 8usize {
                        for tj in 0..2usize {
                            let cacc = cmma::Matrix::<f32>::from_value(
                                cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, 0.0f32,
                            );
                            for k8 in 0..bc / 8usize {
                                let a = cmma::Matrix::<f16>::from_slice(
                                    cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                    &psh.to_slice().slice(ti * 8usize * bc + k8 * 8usize, br * bc), bc as u32,
                                );
                                let b = cmma::Matrix::<f16>::from_slice(
                                    cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                    &vsh.to_slice().slice(k8 * 8usize * d + dn * 16usize + tj * 8usize, bc * d), d as u32,
                                );
                                cmma::execute::<f16, f16, f32, f32>(&a, &b, &cacc, &cacc);
                            }
                            cmma::store(
                                &mut ovt.to_slice_mut().slice_mut(ti * 8usize * 16usize + tj * 8usize, br * 16usize),
                                &cacc, 16u32, cmma::MatrixLayout::RowMajor,
                            );
                        }
                    }
                    sync_cube();
                    for e in 0..per_tile {
                        let tidx = lane * per_tile + e;
                        let r = tidx / 16usize;
                        let col = tidx % 16usize;
                        let oidx = r * d + dn * 16usize + col;
                        of[oidx] = of[oidx] * emf[r] + ovt[tidx];
                    }
                    sync_cube();
                }
            }
            default => {
                for e in 0..per_o {
                    let idx = lane * per_o + e;
                    let r = idx / d;
                    let dd = idx % d;
                    let mut o = of[idx] * emf[r];
                    for c in 0..bc {
                        let kpos = j * bc + c;
                        if kpos < ctx_len {
                            let vv = f32::cast_from(load_paged_v(
                                v_cache, block_tables, b_i, kv, kpos, dd, n_kv,
                                block_size, max_blocks, kv_layout, d,
                            ));
                            o += sf[r * bc + c] * vv;
                        }
                    }
                    of[idx] = o;
                }
            }
        };
        sync_cube();
    }

    for e in 0..per_o {
        let idx = lane * per_o + e;
        let r = idx / d;
        let dd = idx % d;
        let qpos = qt * br + r;
        if qpos < seq_q {
            let mut denom = 1.0f32;
            if lf[r] > 0.0f32 {
                denom = lf[r];
            }
            out[q_head_base + qpos * d + dd] = F::cast_from(of[idx] / denom);
        }
    }
}

/// Host launch for [`flash_attn_paged`] over any `CubeElement + Float` (`f32`, `f16`, `bf16`).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_paged_run_as<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    q: &[E],
    k_cache: &[E],
    v_cache: &[E],
    block_tables: &[u32],
    context_lens: &[u32],
    b: usize,
    n_heads: usize,
    n_kv: usize,
    seq_q: usize,
    d: usize,
    br: usize,
    bc: usize,
    causal: u32,
    block_size: usize,
    max_blocks: usize,
    kv_layout: usize,
    x_vec: usize,
    sliding_window: usize,
    q_offset_override: usize,
    scale: f32,
    softcap: f32,
) -> Vec<E> {
    let plane = (client.properties().hardware.plane_size_max as usize).max(br);
    let target = Target::of(client);
    let cubes = flash_attn_paged_cubes(b, n_heads, seq_q, br);
    let out_len = b * n_heads * seq_q * d;

    let qh = client.create_from_slice(E::as_bytes(q));
    let kh = client.create_from_slice(E::as_bytes(k_cache));
    let vh = client.create_from_slice(E::as_bytes(v_cache));
    let bth = client.create_from_slice(u32::as_bytes(block_tables));
    let clh = client.create_from_slice(u32::as_bytes(context_lens));
    let oh = client.create_from_slice(E::as_bytes(&vec![E::from_int(0); out_len]));
    let sch = client.create_from_slice(f32::as_bytes(&[scale, softcap]));

    let meta = |cube_base: usize| -> [u32; 12] {
        [
            seq_q as u32,
            n_heads as u32,
            n_kv as u32,
            causal,
            block_size as u32,
            max_blocks as u32,
            kv_layout as u32,
            x_vec as u32,
            sliding_window as u32,
            q_offset_override as u32,
            cube_base as u32,
            0u32,
        ]
    };

    if target == Target::Cpu {
        for c_idx in 0..cubes {
            let mh = client.create_from_slice(u32::as_bytes(&meta(c_idx)));
            unsafe {
                flash_attn_paged::launch_unchecked::<E, R>(
                    client,
                    Grid::Static(1, 1, 1),
                    Block::new_1d(plane as u32),
                    ArrayArg::from_raw_parts(qh.clone(), q.len()),
                    ArrayArg::from_raw_parts(kh.clone(), k_cache.len()),
                    ArrayArg::from_raw_parts(vh.clone(), v_cache.len()),
                    ArrayArg::from_raw_parts(bth.clone(), block_tables.len()),
                    ArrayArg::from_raw_parts(clh.clone(), context_lens.len()),
                    ArrayArg::from_raw_parts(oh.clone(), out_len),
                    ArrayArg::from_raw_parts(sch.clone(), 2),
                    ArrayArg::from_raw_parts(mh.clone(), 12),
                    d,
                    br,
                    bc,
                    plane,
                    target,
                );
            }
        }
    } else {
        let mh = client.create_from_slice(u32::as_bytes(&meta(0)));
        unsafe {
            flash_attn_paged::launch_unchecked::<E, R>(
                client,
                Grid::Static(cubes as u32, 1, 1),
                Block::new_1d(plane as u32),
                ArrayArg::from_raw_parts(qh.clone(), q.len()),
                ArrayArg::from_raw_parts(kh.clone(), k_cache.len()),
                ArrayArg::from_raw_parts(vh.clone(), v_cache.len()),
                ArrayArg::from_raw_parts(bth.clone(), block_tables.len()),
                ArrayArg::from_raw_parts(clh.clone(), context_lens.len()),
                ArrayArg::from_raw_parts(oh.clone(), out_len),
                ArrayArg::from_raw_parts(sch.clone(), 2),
                ArrayArg::from_raw_parts(mh.clone(), 12),
                d,
                br,
                bc,
                plane,
                target,
            );
        }
    }
    E::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

/// Exact f64 CPU oracle for [`flash_attn_paged`] across all 3 KV layouts, GQA, variable
/// `context_lens`, causal `q_offset`, sliding window, and softcapping.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_paged_ref(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    block_tables: &[u32],
    context_lens: &[u32],
    b: usize,
    n_heads: usize,
    n_kv: usize,
    seq_q: usize,
    d: usize,
    causal: u32,
    block_size: usize,
    max_blocks: usize,
    kv_layout: usize,
    x_vec: usize,
    sliding_window: usize,
    q_offset_override: usize,
    scale: f32,
    softcap: f32,
) -> Vec<f32> {
    let mut out = vec![0.0f32; b * n_heads * seq_q * d];
    let gqa = n_heads / n_kv;
    let read_k = |bi: usize, kvh: usize, kpos: usize, dd: usize| -> f64 {
        let idx = if kv_layout == 2 {
            ((bi * n_kv + kvh) * block_size + kpos) * d + dd
        } else {
            let blk = block_tables[bi * max_blocks + kpos / block_size] as usize;
            let off = kpos % block_size;
            if kv_layout == 0 {
                ((blk * n_kv + kvh) * block_size + off) * d + dd
            } else {
                let dx = d / x_vec;
                ((((blk * n_kv + kvh) * dx + (dd / x_vec)) * block_size + off) * x_vec)
                    + (dd % x_vec)
            }
        };
        k_cache[idx] as f64
    };
    let read_v = |bi: usize, kvh: usize, kpos: usize, dd: usize| -> f64 {
        let idx = if kv_layout == 2 {
            ((bi * n_kv + kvh) * block_size + kpos) * d + dd
        } else {
            let blk = block_tables[bi * max_blocks + kpos / block_size] as usize;
            let off = kpos % block_size;
            if kv_layout == 0 {
                ((blk * n_kv + kvh) * block_size + off) * d + dd
            } else {
                (((blk * n_kv + kvh) * d + dd) * block_size) + off
            }
        };
        v_cache[idx] as f64
    };

    for bi in 0..b {
        let ctx_len = context_lens[bi] as usize;
        let q_offset = if causal == 2 {
            q_offset_override
        } else {
            ctx_len.saturating_sub(seq_q)
        };
        for h in 0..n_heads {
            let kvh = h / gqa;
            let q_base = (bi * n_heads + h) * seq_q * d;
            for qp in 0..seq_q {
                let q_abs = q_offset + qp;
                let mut scores = vec![f64::NEG_INFINITY; ctx_len];
                let mut m = f64::NEG_INFINITY;
                for kp in 0..ctx_len {
                    let causal_ok = causal == 0 || kp <= q_abs;
                    let win_ok = sliding_window == 0 || kp + sliding_window > q_abs;
                    if causal_ok && win_ok {
                        let mut dot = 0.0f64;
                        for dd in 0..d {
                            dot += (q[q_base + qp * d + dd] as f64) * read_k(bi, kvh, kp, dd);
                        }
                        let mut s = dot * (scale as f64);
                        if softcap > 0.0 {
                            let sc = softcap as f64;
                            s = sc * (s / sc).tanh();
                        }
                        scores[kp] = s;
                        if s > m {
                            m = s;
                        }
                    }
                }
                let mut l = 0.0f64;
                let mut probs = vec![0.0f64; ctx_len];
                for kp in 0..ctx_len {
                    if scores[kp].is_finite() {
                        let p = (scores[kp] - m).exp();
                        probs[kp] = p;
                        l += p;
                    }
                }
                let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                for dd in 0..d {
                    let mut acc = 0.0f64;
                    for kp in 0..ctx_len {
                        if probs[kp] != 0.0 {
                            acc += probs[kp] * read_v(bi, kvh, kp, dd);
                        }
                    }
                    out[q_base + qp * d + dd] = (acc * inv_l) as f32;
                }
            }
        }
    }
    out
}

/// Host launch: `q`/`out` are `[b, n_heads, seq_q, d]`, `k`/`v` are `[b, n_kv, kv_seq_pad, d]` read as
/// `[b, n_kv, seq_k, d]` (the padded KV cache). GQA `n_kv_groups = n_heads / n_kv`. The plane (threads
/// per cube) is derived from the device (`plane_size_max`). One cube computes one `(batch, head,
/// query-tile)`. Production dispatches the whole grid at once (`cube_base = 0`).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_run<R: Runtime>(
    client: &ComputeClient<R>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    n_heads: usize,
    n_kv: usize,
    seq_q: usize,
    seq_k: usize,
    kv_seq_pad: usize,
    d: usize,
    causal: bool,
) -> Vec<f32> {
    // The plane (threads per cube) is one hardware plane: 32 on a CUDA/NVIDIA warp, 64 on a RADV
    // subgroup. A cooperative-matrix op is plane-scoped -- the 16x16 fragment is spread over every lane
    // of the plane -- so the block must be exactly one plane wide. Deriving it from the device is the one
    // portable value (same rule as `mmq_q8_wmma`); a baked constant is right on at most one backend and
    // mispartitions the fragment on the others. `.max(BR)` floors the scalar CPU simulator, which reports
    // plane_size_max = 1 (no SIMD) yet still needs `br` logical lanes: the online-softmax pass assigns
    // query-row r to lane r, so a plane below `br` leaves rows >= plane with divide-by-zero (l = 0) ->
    // inf. Real GPU planes are 32/64 >= br = 16, so the floor is a no-op there.
    let plane = (client.properties().hardware.plane_size_max as usize).max(BR);
    let cubes = flash_attn_cubes(b, n_heads, seq_q);
    let target = Target::of(client);
    flash_attn_launch(
        client, q, k, v, b, n_heads, n_kv, seq_q, seq_k, kv_seq_pad, d, causal, plane, 0, cubes,
        target,
    )
}

/// The launch primitive, with an explicit `cube_base` offset, `cube_count`, and island `target`.
/// Production calls it via [`flash_attn_run`] (`cube_base = 0`, whole grid, `Target::of`); the CPU
/// oracle drives it one cube at a time with `Target::Cpu` to pin the normative scalar arm.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_launch<R: Runtime>(
    client: &ComputeClient<R>,
    q: &[f32],
    k: &[f32],
    v: &[f32],
    b: usize,
    n_heads: usize,
    n_kv: usize,
    seq_q: usize,
    seq_k: usize,
    kv_seq_pad: usize,
    d: usize,
    causal: bool,
    plane: usize,
    cube_base: usize,
    cube_count: usize,
    target: Target,
) -> Vec<f32> {
    let scale = 1.0f32 / (d as f32).sqrt();
    let meta = [
        seq_q as u32,
        seq_k as u32,
        n_heads as u32,
        n_kv as u32,
        causal as u32,
        (n_kv * kv_seq_pad * d) as u32,
        (kv_seq_pad * d) as u32,
        d as u32,
        cube_base as u32,
    ];
    let qh = client.create_from_slice(f32::as_bytes(q));
    let kh = client.create_from_slice(f32::as_bytes(k));
    let vh = client.create_from_slice(f32::as_bytes(v));
    let sh = client.create_from_slice(f32::as_bytes(&[scale]));
    let mh = client.create_from_slice(u32::as_bytes(&meta));
    let oh = client.create_from_slice(f32::as_bytes(&vec![0.0f32; b * n_heads * seq_q * d]));
    unsafe {
        flash_attn::launch_unchecked::<f32, R>(
            client,
            Grid::Static(cube_count as u32, 1, 1),
            Block::new_1d(plane as u32),
            ArrayArg::from_raw_parts(qh.clone(), q.len()),
            ArrayArg::from_raw_parts(kh.clone(), k.len()),
            ArrayArg::from_raw_parts(vh.clone(), v.len()),
            ArrayArg::from_raw_parts(oh.clone(), b * n_heads * seq_q * d),
            ArrayArg::from_raw_parts(sh.clone(), 1),
            ArrayArg::from_raw_parts(mh.clone(), 9),
            d,
            BR,
            BC,
            plane,
            target,
        );
    }
    f32::from_bytes(&client.read_one_unchecked(oh)).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attn::sdpa_ref;

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

    /// Scale-relative error: `max|Δ| / max|ref|`. The correct gate for an online softmax vs a
    /// materialized two-pass reference -- per-element relative error explodes on near-zero output
    /// cancellation and produces false failures (PHILOSOPHY: signed/affine -> scale-relative).
    fn scale_rel(got: &[f32], want: &[f32]) -> f32 {
        let refmax = want.iter().fold(0.0f32, |a, x| a.max(x.abs())).max(1e-6);
        let maxd = got
            .iter()
            .zip(want)
            .fold(0.0f32, |a, (g, w)| a.max((g - w).abs()));
        maxd / refmax
    }

    /// Gate one shape: flash (online, tiled) vs `sdpa_ref` (materialized two-pass softmax). batch=1 so
    /// the reference layout `[n_heads, seq_q, d]` / `[n_kv, seq_k, d]` matches (kv_seq_pad = seq_k).
    ///
    /// Driven one cube at a time (`cube_base = n`, 1-cube grid) and summed: cubes write disjoint output
    /// slices over a zero-initialized buffer, so the element-wise sum reconstructs the full-grid result
    /// exactly (adding 0 is exact) -- while sidestepping cubecl-cpu's cross-cube SharedMemory aliasing.
    #[allow(clippy::too_many_arguments)]
    fn gate<R: Runtime>(
        c: &ComputeClient<R>,
        nh: usize,
        nkv: usize,
        sq: usize,
        sk: usize,
        d: usize,
        causal: bool,
        plane: usize,
        tag: &str,
    ) {
        let q = rnd(nh * sq * d, 0x1234_5678);
        let k = rnd(nkv * sk * d, 0x9ABC_DEF0);
        let v = rnd(nkv * sk * d, 0x0FED_CBA9);
        let ncubes = flash_attn_cubes(1, nh, sq);
        let mut got = vec![0.0f32; nh * sq * d];
        for n in 0..ncubes {
            let part = flash_attn_launch::<R>(
                c,
                &q,
                &k,
                &v,
                1,
                nh,
                nkv,
                sq,
                sk,
                sk,
                d,
                causal,
                plane,
                n,
                1,
                Target::Cpu,
            );
            for (g, p) in got.iter_mut().zip(&part) {
                *g += *p;
            }
        }
        let want = sdpa_ref(&q, &k, &v, nh, nkv, sq, sk, d, causal);
        let rel = scale_rel(&got, &want);
        eprintln!("[flash {tag}] nh{nh}/nkv{nkv} sq{sq} sk{sk} d{d} causal{causal} plane{plane} cubes{ncubes} scale_rel={rel:.2e}");
        assert!(
            rel < 2e-3,
            "flash {tag}: scale_rel {rel} exceeds 2e-3 vs materialized sdpa_ref"
        );
    }

    /// The scalar (default) arm, on the CPU runtime, is the online-softmax flash and matches the
    /// materialized reference to online-softmax tolerance across the production shape space: prefill
    /// (seq_q == seq_k, causal), decode (seq_q = 1, growing kv, non-causal), GQA ratios, causal on/off.
    #[cfg(feature = "cpu")]
    #[test]
    fn flash_matches_materialized_ref_on_cpu() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        // decode: one query sees the whole (growing) cache, non-causal, GQA ratios 2 and 4.
        gate::<CpuRuntime>(&c, 4, 2, 1, 1, 32, false, 32, "decode kv1");
        gate::<CpuRuntime>(&c, 4, 2, 1, 17, 32, false, 32, "decode kv17 (tail)");
        gate::<CpuRuntime>(&c, 4, 2, 1, 128, 32, false, 32, "decode kv128");
        gate::<CpuRuntime>(&c, 8, 2, 1, 512, 64, false, 32, "decode kv512 GQA4");
        // prefill: causal + non-causal, single and multiple query-tiles, GQA and MHA, d in {32,64,128}.
        gate::<CpuRuntime>(&c, 4, 2, 16, 128, 32, true, 32, "prefill qt1 causal GQA2");
        gate::<CpuRuntime>(
            &c,
            4,
            2,
            16,
            128,
            32,
            false,
            32,
            "prefill qt1 noncausal GQA2",
        );
        gate::<CpuRuntime>(&c, 4, 4, 16, 64, 64, true, 32, "prefill qt1 MHA causal d64");
        gate::<CpuRuntime>(
            &c,
            4,
            2,
            48,
            48,
            32,
            true,
            32,
            "prefill 3-tile causal GQA2 (aligned)",
        );
        gate::<CpuRuntime>(
            &c,
            6,
            3,
            40,
            40,
            64,
            true,
            32,
            "prefill causal GQA2 tail sq40 d64",
        );
        gate::<CpuRuntime>(
            &c,
            2,
            1,
            512,
            512,
            128,
            true,
            32,
            "prefill 512 causal MHA d128",
        );
    }

    /// The f16 cooperative-matrix arm, EXECUTED on RADV (Vulkan/wgpu), matched against the materialized
    /// two-pass reference across the same production shape space as the CPU oracle. `flash_attn_run`
    /// derives `Target::Vulkan` from `WgpuRuntime` and launches the WHOLE grid at once (one workgroup
    /// per (batch,head,query-tile)) -- the production launch, not the CPU oracle's cube-by-cube walk
    /// (real GPU SharedMemory is per-workgroup, so there is no cross-cube aliasing to sidestep). Gate is
    /// the f16-tensor-core tolerance (~1e-3..1e-2 scale-relative for the chained f16 QK + f16 PV), not
    /// the scalar arm's ~1e-7. This is the on-GPU gate that the CPU oracle cannot give (cubecl-cpu
    /// rejects every CoopMma op); it validates the ColMajor-K -> Kᵀ fragment layout, the PV store
    /// stride, the partial-tile zero-pad on ragged seq, and the f16 QK/PV precision, all on real silicon.
    #[cfg(feature = "vulkan")]
    #[test]
    fn flash_coopmat_matches_ref_on_vulkan() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let c = WgpuRuntime::client(&WgpuDevice::default());
        let mut fails = 0usize;
        // `flash_attn_run` derives the plane from the device. RADV gfx1151's coopmat runs at
        // subgroupSize=64 (VkPhysicalDeviceCooperativeMatrixPropertiesKHR), so the whole-grid launch is
        // one 64-lane workgroup per cube -- a 32-lane launch would feed the 16x16 f16 fragment only half
        // its lanes and leave the output tile past query-row 1 undefined. mmq_q4k / sdpa_blk derive it too.
        let mut gate_gpu = |nh: usize,
                            nkv: usize,
                            sq: usize,
                            sk: usize,
                            d: usize,
                            causal: bool,
                            tag: &str| {
            let q = rnd(nh * sq * d, 0x1234_5678);
            let k = rnd(nkv * sk * d, 0x9ABC_DEF0);
            let v = rnd(nkv * sk * d, 0x0FED_CBA9);
            // Whole-grid Vulkan launch (Target::of(WgpuRuntime) = Vulkan -> the coopmat arm).
            let got =
                flash_attn_run::<WgpuRuntime>(&c, &q, &k, &v, 1, nh, nkv, sq, sk, sk, d, causal);
            let want = sdpa_ref(&q, &k, &v, nh, nkv, sq, sk, d, causal);
            let rel = scale_rel(&got, &want);
            let nonfin = got.iter().filter(|x| !x.is_finite()).count();
            // On failure, localize the first disagreeing element (row-major [nh, sq, d]) -- a fragment
            // layout / mask / OOB bug shows up as a clean row or column boundary.
            let where_ = if rel < 2e-3 {
                String::new()
            } else {
                got.iter()
                    .zip(&want)
                    .enumerate()
                    .find(|(_, (g, w))| (**g - **w).abs() > 1e-2 || !g.is_finite())
                    .map(|(i, (g, w))| {
                        let (h, rem) = (i / (sq * d), i % (sq * d));
                        format!(
                            " first-bad@[h{h},q{},d{}] got={g:.3e} want={w:.3e}",
                            rem / d,
                            rem % d
                        )
                    })
                    .unwrap_or_default()
            };
            eprintln!("[flash-vk {tag}] nh{nh}/nkv{nkv} sq{sq} sk{sk} d{d} causal{causal} scale_rel={rel:.2e} nonfinite={nonfin}{where_}");
            if !(rel < 2e-2) {
                fails += 1;
            }
        };
        // The same 10 shapes as the CPU oracle `flash_matches_materialized_ref_on_cpu`: decode (ragged
        // kv 1/17), decode aligned/long GQA, prefill causal + non-causal, GQA and MHA, single + multi
        // query-tile, the ragged-tail sq40, and the 512x512 d128. d in {32,64,128}; every one on RADV.
        gate_gpu(4, 2, 1, 1, 32, false, "decode kv1");
        gate_gpu(4, 2, 1, 17, 32, false, "decode kv17 (ragged tail)");
        gate_gpu(4, 2, 1, 128, 32, false, "decode kv128");
        gate_gpu(8, 2, 1, 512, 64, false, "decode kv512 GQA4");
        gate_gpu(4, 2, 16, 128, 32, true, "prefill qt1 causal GQA2");
        gate_gpu(4, 2, 16, 128, 32, false, "prefill qt1 noncausal GQA2");
        gate_gpu(4, 4, 16, 64, 64, true, "prefill qt1 MHA causal d64");
        gate_gpu(
            4,
            2,
            48,
            48,
            32,
            true,
            "prefill 3-tile causal GQA2 (aligned)",
        );
        gate_gpu(6, 3, 40, 40, 64, true, "prefill causal GQA2 tail sq40 d64");
        gate_gpu(2, 1, 512, 512, 128, true, "prefill 512 causal MHA d128");
        assert_eq!(
            fails, 0,
            "flash coopmat: {fails} shape(s) exceeded 2e-2 vs materialized sdpa_ref"
        );
    }

    /// The production surface (`flash_attn_run`, which derives the island tag from the runtime) resolves
    /// to the scalar oracle on CPU with no caller naming a target -- and matches the materialized ref.
    /// The cmma accelerated arm is never forced onto CPU: cubecl-cpu rejects every CoopMma op, so the
    /// arm is gated on the GPU (see STATE-flash.md), exactly like `mmq_q8_wmma`.
    #[cfg(feature = "cpu")]
    #[test]
    fn production_run_selects_scalar_oracle_on_cpu() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        // Single (batch, head, qtile) so the full-grid production launch is one cube -- no cross-cube
        // SharedMemory aliasing -- yet still exercises `flash_attn_run` / `Target::of` end to end.
        let (nh, nkv, sq, sk, d) = (1usize, 1usize, 16usize, 96usize, 64usize);
        let q = rnd(nh * sq * d, 0x2222_3333);
        let k = rnd(nkv * sk * d, 0x4444_5555);
        let v = rnd(nkv * sk * d, 0x6666_7777);
        let got = flash_attn_run::<CpuRuntime>(&c, &q, &k, &v, 1, nh, nkv, sq, sk, sk, d, true);
        let want = sdpa_ref(&q, &k, &v, nh, nkv, sq, sk, d, true);
        let rel = scale_rel(&got, &want);
        eprintln!("[flash production] scalar-on-cpu scale_rel={rel:.2e}");
        assert!(rel < 2e-3, "production run on cpu: scale_rel {rel}");
    }

    /// The f16 cooperative-matrix arm, EXECUTED on CUDA (GB10, sm_121) where `cmma` lowers to
    /// `nvcuda::wmma`, matched against the materialized two-pass reference across the same 10 production
    /// shapes as the CPU oracle and the Vulkan gate. This validates the 4th (and last untested) backend
    /// of `flash_attn`: the ColMajor-K -> Kᵀ WMMA fragment layout, the PV store stride, the partial-tile
    /// zero-pad on ragged seq, and the f16 QK/PV precision -- all on real NVIDIA silicon (~2-3e-4
    /// scale-relative, the chained-f16 tensor-core tolerance, not the scalar arm's ~1e-7).
    ///
    /// `flash_attn_run` derives the plane from the device, so this loop validates at the true CUDA
    /// production plane (`plane_size_max` = 32, one warp per 16x16x16 WMMA fragment -- no redundant
    /// lanes). A WMMA fragment is WARP-scoped (32, fixed); a RADV coopmat fragment is SUBGROUP-scoped
    /// (64). There is therefore no single portable plane CONSTANT -- the device width IS the value,
    /// which is why `flash_attn_run` derives it rather than baking one. Baking 64 (the RADV width) is
    /// only accidentally correct on CUDA -- it runs 2 redundant warps that recompute the identical
    /// warp-scoped fragment -- and would MISPARTITION a 32-lane NVIDIA-Vulkan coopmat, whose fragment is
    /// subgroup-sized. The plane-agree block below pins that CUDA stays warp-scoped (plane 32 == plane 64
    /// byte-for-byte), guarding any future change that makes the coopmat op block- rather than
    /// warp/subgroup-scoped (which would leave half the 64-lane output tile undefined).
    #[cfg(feature = "cuda")]
    #[test]
    fn flash_coopmat_matches_ref_on_cuda() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        let c = CudaRuntime::client(&CudaDevice::default());
        // The device plane the production surface derives (measured: 32 on this GB10 warp).
        eprintln!(
            "[flash-cuda] device plane_size_max={}",
            c.properties().hardware.plane_size_max
        );
        let shapes: [(usize, usize, usize, usize, usize, bool, &str); 10] = [
            (4, 2, 1, 1, 32, false, "decode kv1"),
            (4, 2, 1, 17, 32, false, "decode kv17 (ragged tail)"),
            (4, 2, 1, 128, 32, false, "decode kv128"),
            (8, 2, 1, 512, 64, false, "decode kv512 GQA4"),
            (4, 2, 16, 128, 32, true, "prefill qt1 causal GQA2"),
            (4, 2, 16, 128, 32, false, "prefill qt1 noncausal GQA2"),
            (4, 4, 16, 64, 64, true, "prefill qt1 MHA causal d64"),
            (
                4,
                2,
                48,
                48,
                32,
                true,
                "prefill 3-tile causal GQA2 (aligned)",
            ),
            (6, 3, 40, 40, 64, true, "prefill causal GQA2 tail sq40 d64"),
            (2, 1, 512, 512, 128, true, "prefill 512 causal MHA d128"),
        ];
        let mut fails = 0usize;
        for (nh, nkv, sq, sk, d, causal, tag) in shapes {
            let q = rnd(nh * sq * d, 0x1234_5678);
            let k = rnd(nkv * sk * d, 0x9ABC_DEF0);
            let v = rnd(nkv * sk * d, 0x0FED_CBA9);
            let got =
                flash_attn_run::<CudaRuntime>(&c, &q, &k, &v, 1, nh, nkv, sq, sk, sk, d, causal);
            let want = sdpa_ref(&q, &k, &v, nh, nkv, sq, sk, d, causal);
            let rel = scale_rel(&got, &want);
            let nonfin = got.iter().filter(|x| !x.is_finite()).count();
            let where_ = if rel < 2e-3 {
                String::new()
            } else {
                got.iter()
                    .zip(&want)
                    .enumerate()
                    .find(|(_, (g, w))| (**g - **w).abs() > 1e-2 || !g.is_finite())
                    .map(|(i, (g, w))| {
                        let (h, rem) = (i / (sq * d), i % (sq * d));
                        format!(
                            " first-bad@[h{h},q{},d{}] got={g:.3e} want={w:.3e}",
                            rem / d,
                            rem % d
                        )
                    })
                    .unwrap_or_default()
            };
            eprintln!("[flash-cuda {tag}] nh{nh}/nkv{nkv} sq{sq} sk{sk} d{d} causal{causal} scale_rel={rel:.2e} nonfinite={nonfin}{where_}");
            if !(rel < 2e-2) {
                fails += 1;
            }
        }
        assert_eq!(
            fails, 0,
            "flash coopmat CUDA: {fails} shape(s) exceeded 2e-2 vs materialized sdpa_ref"
        );

        // Portability regression guard: CUDA must agree at plane=32 and plane=64. The extra warp at
        // plane=64 is redundant, not wrong (warp-scoped WMMA); a future change making the coopmat op
        // subgroup/block-scoped would diverge these massively (half the output tile undefined).
        for (nh, nkv, sq, sk, d, causal, tag) in [
            (
                4usize,
                2usize,
                16usize,
                128usize,
                32usize,
                true,
                "prefill causal",
            ),
            (8, 2, 1, 512, 64, false, "decode kv512"),
            (2, 1, 512, 512, 128, true, "prefill 512 d128"),
        ] {
            let q = rnd(nh * sq * d, 0x1234_5678);
            let k = rnd(nkv * sk * d, 0x9ABC_DEF0);
            let v = rnd(nkv * sk * d, 0x0FED_CBA9);
            let cubes = flash_attn_cubes(1, nh, sq);
            let p32 = flash_attn_launch::<CudaRuntime>(
                &c,
                &q,
                &k,
                &v,
                1,
                nh,
                nkv,
                sq,
                sk,
                sk,
                d,
                causal,
                32,
                0,
                cubes,
                Target::of(&c),
            );
            let p64 = flash_attn_launch::<CudaRuntime>(
                &c,
                &q,
                &k,
                &v,
                1,
                nh,
                nkv,
                sq,
                sk,
                sk,
                d,
                causal,
                64,
                0,
                cubes,
                Target::of(&c),
            );
            let agree = scale_rel(&p32, &p64);
            eprintln!("[flash-cuda plane-agree {tag}] scale_rel(p32,p64)={agree:.2e}");
            assert!(
                agree < 1e-4,
                "CUDA plane 32 vs 64 diverged ({agree}) -- coopmat no longer warp-scoped"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn flash_attn_paged_cuda_all_layouts_and_dtypes() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        let c = CudaRuntime::client(&CudaDevice::new(0));

        // 3 requests in batch with variable context_lens, shuffled noncontiguous physical pages,
        // GQA (8 Q heads / 2 KV heads), chunked prefill (seq_q=5) and decode (seq_q=1),
        // layouts 0 (4D paged), 1 (Engine Standard 5D K / 4D V), and 2 (contiguous),
        // d = 64, 128, 256, sliding window, softcapping, and F32/F16/BF16!
        let block_size = 16usize;
        let max_blocks = 4usize;
        let num_blocks = 12usize;
        let b = 3usize;
        let n_heads = 8usize;
        let n_kv = 2usize;
        let block_tables: Vec<u32> = vec![
            5, 2, 9, 0, // req 0: noncontiguous physical blocks
            11, 4, 1, 7, // req 1
            8, 3, 10, 6, // req 2
        ];
        let ctx_lens: Vec<u32> = vec![23, 51, 9];

        for &(d, br, seq_q, layout, window, softcap) in &[
            (64usize, 16usize, 1usize, 0usize, 0usize, 0.0f32),
            (128, 16, 1, 1, 0, 0.0),
            (128, 16, 5, 1, 0, 0.0),
            (128, 16, 5, 0, 16, 30.0),
            (256, 16, 1, 1, 0, 0.0),
            (128, 16, 7, 2, 0, 0.0),
        ] {
            let x_vec = 4usize; // 16 / sizeof(f32)
            let scale = 1.0f32 / (d as f32).sqrt();
            let q = rnd(b * n_heads * seq_q * d, 0xD100 + d as u64 + layout as u64);
            let kv_elems = if layout == 2 {
                b * n_kv * 64 * d
            } else {
                num_blocks * n_kv * block_size * d
            };
            let k_cache = rnd(kv_elems, 0xD200 + d as u64 + layout as u64);
            let v_cache = rnd(kv_elems, 0xD300 + d as u64 + layout as u64);
            let bs_arg = if layout == 2 { 64 } else { block_size };

            let want = flash_attn_paged_ref(
                &q, &k_cache, &v_cache, &block_tables, &ctx_lens,
                b, n_heads, n_kv, seq_q, d, 1, bs_arg, max_blocks, layout, x_vec,
                window, 0, scale, softcap,
            );
            let got = flash_attn_paged_run_as::<f32, CudaRuntime>(
                &c, &q, &k_cache, &v_cache, &block_tables, &ctx_lens,
                b, n_heads, n_kv, seq_q, d, br, 16, 1, bs_arg, max_blocks, layout, x_vec,
                window, 0, scale, softcap,
            );
            let rel = scale_rel(&want, &got);
            eprintln!("[flash_paged CUDA f32] d={d:3} sq={seq_q} layout={layout} win={window:2} cap={softcap:.0} rel={rel:.2e}");
            assert!(rel < 2e-2, "flash_attn_paged f32 failed: rel={rel:.3e}");

            // BF16 & F16 verification on Standard Engine 5D layout (x_vec=8 for 2-byte dtypes)
            if layout == 1 && d == 128 {
                let x_vec_16 = 8usize;
                let q_h: Vec<half::f16> = q.iter().map(|&v| half::f16::from_f32(v)).collect();
                let k_h: Vec<half::f16> = k_cache.iter().map(|&v| half::f16::from_f32(v)).collect();
                let v_h: Vec<half::f16> = v_cache.iter().map(|&v| half::f16::from_f32(v)).collect();
                let want_h = flash_attn_paged_ref(
                    &q_h.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                    &k_h.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                    &v_h.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                    &block_tables, &ctx_lens,
                    b, n_heads, n_kv, seq_q, d, 1, block_size, max_blocks, 1, x_vec_16,
                    window, 0, scale, softcap,
                );
                let got_h = flash_attn_paged_run_as::<half::f16, CudaRuntime>(
                    &c, &q_h, &k_h, &v_h, &block_tables, &ctx_lens,
                    b, n_heads, n_kv, seq_q, d, br, 16, 1, block_size, max_blocks, 1, x_vec_16,
                    window, 0, scale, softcap,
                );
                let rel_h = scale_rel(&want_h, &got_h.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
                eprintln!("[flash_paged CUDA f16] d={d:3} sq={seq_q} layout=1 rel={rel_h:.2e}");
                assert!(rel_h < 2e-2, "flash_attn_paged f16 failed: rel={rel_h:.3e}");

                let q_b: Vec<half::bf16> = q.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let k_b: Vec<half::bf16> = k_cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let v_b: Vec<half::bf16> = v_cache.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let want_b = flash_attn_paged_ref(
                    &q_b.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                    &k_b.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                    &v_b.iter().map(|v| v.to_f32()).collect::<Vec<_>>(),
                    &block_tables, &ctx_lens,
                    b, n_heads, n_kv, seq_q, d, 1, block_size, max_blocks, 1, x_vec_16,
                    window, 0, scale, softcap,
                );
                let got_b = flash_attn_paged_run_as::<half::bf16, CudaRuntime>(
                    &c, &q_b, &k_b, &v_b, &block_tables, &ctx_lens,
                    b, n_heads, n_kv, seq_q, d, br, 16, 1, block_size, max_blocks, 1, x_vec_16,
                    window, 0, scale, softcap,
                );
                let rel_b = scale_rel(&want_b, &got_b.iter().map(|v| v.to_f32()).collect::<Vec<_>>());
                eprintln!("[flash_paged CUDA bf16] d={d:3} sq={seq_q} layout=1 rel={rel_b:.2e}");
                assert!(rel_b < 2e-2, "flash_attn_paged bf16 failed: rel={rel_b:.3e}");
            }
        }
    }
}
