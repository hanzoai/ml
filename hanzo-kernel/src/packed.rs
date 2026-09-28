//! Attention over packed sequences straight from a packed QKV projection, rotated in the kernel,
//! each query reading its own sequence's keys within an optional window: the encoder's training
//! pass. `attend` is the forward (outputs and each row's log-sum-exp).
//!
//! - `qkv [T, 3·H·D]`: each token's queries, keys and values, heads contiguous within each.
//! - `cu [S + 1]`: each sequence's first token; `tiles [NT, 2]`: each `br`-row tile's sequence
//!   and first row.
//! - `cos`, `sin [P, D/2]`: rotary tables by position within a sequence.
//! - `meta = [heads, window + 1 (0: every key), T, tile base, head base]`; the bases offset the cube
//!   position so the CPU oracle can run one cube at a time.

use crate::prelude::*;

/// The head dimension.
pub const D: usize = 64;

/// One tile of `br` queries of one head against every key tile its window reaches: scores and the
/// output product on the matrix units (`cuda | rocm`: 16×16×16 fragments, a plane per 16 rows;
/// `metal`: 8×8×8), the softmax online per row. `units` is `br / 16` planes of the device's width.
#[allow(clippy::too_many_arguments)]
#[kernel(targets(cuda, rocm, metal, cpu), unchecked)]
pub fn attend<F: Float, M: Float>(
    qkv: &Array<F>,
    cos: &Array<F>,
    sin: &Array<F>,
    cu: &Array<u32>,
    tiles: &Array<u32>,
    out: &mut Array<F>,
    lse: &mut Array<f32>,
    meta: &Array<u32>,
    scale: &Array<f32>,
    #[comptime] d: usize,
    #[comptime] br: usize,
    #[comptime] bc: usize,
    #[comptime] units: usize,
    #[comptime] target: Target,
) {
    let t = CUBE_POS_X as usize + meta[3] as usize;
    let h = CUBE_POS_Y as usize + meta[4] as usize;
    let u = UNIT_POS as usize;
    let heads = meta[0] as usize;
    let window = meta[1] as usize;
    let total = meta[2] as usize;
    let hd = heads * d;
    let ld = 3 * hd;
    let half = d / 2;
    let seq = tiles[2 * t] as usize;
    let i0 = tiles[2 * t + 1] as usize;
    let start = cu[seq] as usize;
    let len = cu[seq + 1] as usize - start;
    let sc = scale[0];

    let mut qs = SharedMemory::<M>::new(br * d);
    let mut ks = SharedMemory::<M>::new(bc * d);
    let mut vs = SharedMemory::<M>::new(bc * d);
    let mut sf = SharedMemory::<f32>::new(br * bc);
    let mut ps = SharedMemory::<M>::new(br * bc);
    let mut of = SharedMemory::<f32>::new(br * d);
    let mut row = SharedMemory::<f32>::new(3 * br);

    // the query tile, rotated and scaled; rows past the sequence zero
    let per_q = br * half / units;
    for e in 0..per_q {
        let idx = u * per_q + e;
        let r = idx / half;
        let j = idx % half;
        let i = i0 + r;
        let mut y1 = f32::new(0.0f32);
        let mut y2 = f32::new(0.0f32);
        if i < len {
            let at = (start + i) * ld + h * d;
            let x1 = f32::cast_from(qkv[at + j]);
            let x2 = f32::cast_from(qkv[at + j + half]);
            let c = f32::cast_from(cos[i * half + j]);
            let s = f32::cast_from(sin[i * half + j]);
            y1 = (x1 * c - x2 * s) * sc;
            y2 = (x1 * s + x2 * c) * sc;
        }
        qs[r * d + j] = M::cast_from(y1);
        qs[r * d + j + half] = M::cast_from(y2);
    }
    let per_o = br * d / units;
    for e in 0..per_o {
        of[u * per_o + e] = f32::new(0.0f32);
    }
    if u < br {
        row[u] = f32::new(-3.0e38f32);
        row[br + u] = f32::new(0.0f32);
    }

    let mut lo = len - len;
    let mut hi = len;
    if window > 0 {
        if i0 + 1 > window {
            lo = (i0 + 1 - window) / bc * bc;
        }
        if i0 + br + window - 1 < len {
            hi = i0 + br + window - 1;
        }
    }
    let spans = (hi - lo + bc - 1) / bc;
    for jt in 0..spans {
        let j0 = lo + jt * bc;
        // the key tile rotated, the value tile as is; keys past the sequence zero
        let per_k = bc * half / units;
        for e in 0..per_k {
            let idx = u * per_k + e;
            let c = idx / half;
            let jj = idx % half;
            let j = j0 + c;
            let mut y1 = f32::new(0.0f32);
            let mut y2 = f32::new(0.0f32);
            let mut v1 = f32::new(0.0f32);
            let mut v2 = f32::new(0.0f32);
            if j < len {
                let at = (start + j) * ld + hd + h * d;
                let x1 = f32::cast_from(qkv[at + jj]);
                let x2 = f32::cast_from(qkv[at + jj + half]);
                let cs = f32::cast_from(cos[j * half + jj]);
                let sn = f32::cast_from(sin[j * half + jj]);
                y1 = x1 * cs - x2 * sn;
                y2 = x1 * sn + x2 * cs;
                v1 = f32::cast_from(qkv[at + hd + jj]);
                v2 = f32::cast_from(qkv[at + hd + jj + half]);
            }
            ks[c * d + jj] = M::cast_from(y1);
            ks[c * d + jj + half] = M::cast_from(y2);
            vs[c * d + jj] = M::cast_from(v1);
            vs[c * d + jj + half] = M::cast_from(v2);
        }
        sync_cube();

        // scores S = Q Kᵀ into `sf`
        island! {
            cuda | rocm => {
                let p = PLANE_POS as usize;
                for cg in 0..bc / 16usize {
                    let acc = cmma::Matrix::<f32>::from_value(
                        cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize,
                        cmma::MatrixLayout::Undefined, 0.0f32,
                    );
                    for dk in 0..d / 16usize {
                        let a = cmma::Matrix::<M>::from_slice(
                            cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &qs.to_slice().slice(p * 16usize * d + dk * 16usize, br * d), d as u32,
                        );
                        let b = cmma::Matrix::<M>::from_slice(
                            cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                            &ks.to_slice().slice(cg * 16usize * d + dk * 16usize, bc * d), d as u32,
                        );
                        cmma::execute::<M, M, f32, f32>(&a, &b, &acc, &acc);
                    }
                    cmma::store(
                        &mut sf.to_slice_mut().slice_mut(p * 16usize * bc + cg * 16usize, br * bc),
                        &acc, bc as u32, cmma::MatrixLayout::RowMajor,
                    );
                }
            }
            metal => {
                let p = PLANE_POS as usize;
                for ti in 0..2usize {
                    for tj in 0..bc / 8usize {
                        let acc = cmma::Matrix::<f32>::from_value(
                            cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
                            cmma::MatrixLayout::Undefined, 0.0f32,
                        );
                        for dk in 0..d / 8usize {
                            let a = cmma::Matrix::<M>::from_slice(
                                cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &qs.to_slice().slice((p * 16usize + ti * 8usize) * d + dk * 8usize, br * d), d as u32,
                            );
                            let b = cmma::Matrix::<M>::from_slice(
                                cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::ColMajor,
                                &ks.to_slice().slice(tj * 8usize * d + dk * 8usize, bc * d), d as u32,
                            );
                            cmma::execute::<M, M, f32, f32>(&a, &b, &acc, &acc);
                        }
                        cmma::store(
                            &mut sf.to_slice_mut().slice_mut((p * 16usize + ti * 8usize) * bc + tj * 8usize, br * bc),
                            &acc, bc as u32, cmma::MatrixLayout::RowMajor,
                        );
                    }
                }
            }
            default => {
                let per_s = br * bc / units;
                for e in 0..per_s {
                    let idx = u * per_s + e;
                    let r = idx / bc;
                    let c = idx % bc;
                    let mut acc = f32::new(0.0f32);
                    for k in 0..d {
                        acc += f32::cast_from(qs[r * d + k]) * f32::cast_from(ks[c * d + k]);
                    }
                    sf[idx] = acc;
                }
            }
        };
        sync_cube();

        // the online softmax, a row per unit: masked keys weigh exactly zero
        if u < br {
            let i = i0 + u;
            let mut top = f32::new(-3.0e38f32);
            for c in 0..bc {
                let j = j0 + c;
                let mut near = true;
                if window > 0 {
                    let gap = if i > j { i - j } else { j - i };
                    near = gap < window;
                }
                if i < len && j < len && near {
                    let s = sf[u * bc + c];
                    if s > top {
                        top = s;
                    }
                }
            }
            let old = row[u];
            let mut m = old;
            if top > m {
                m = top;
            }
            if m == f32::new(-3.0e38f32) {
                row[2 * br + u] = f32::new(1.0f32);
                for c in 0..bc {
                    ps[u * bc + c] = M::cast_from(f32::new(0.0f32));
                }
            } else {
                let em = (old - m).exp();
                let mut sum = f32::new(0.0f32);
                for c in 0..bc {
                    let j = j0 + c;
                    let mut near = true;
                    if window > 0 {
                        let gap = if i > j { i - j } else { j - i };
                        near = gap < window;
                    }
                    let mut w = f32::new(0.0f32);
                    if i < len && j < len && near {
                        w = (sf[u * bc + c] - m).exp();
                    }
                    ps[u * bc + c] = M::cast_from(w);
                    sum += w;
                }
                row[br + u] = row[br + u] * em + sum;
                row[u] = m;
                row[2 * br + u] = em;
            }
        }
        sync_cube();
        for e in 0..per_o {
            let idx = u * per_o + e;
            of[idx] = of[idx] * row[2 * br + idx / d];
        }
        sync_cube();

        // the output O += P V
        island! {
            cuda | rocm => {
                let p = PLANE_POS as usize;
                for dn in 0..d / 16usize {
                    let acc = cmma::Matrix::<f32>::from_value(
                        cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize,
                        cmma::MatrixLayout::Undefined, 0.0f32,
                    );
                    cmma::load_with_layout(
                        &acc, &of.to_slice().slice(p * 16usize * d + dn * 16usize, br * d), d as u32,
                        cmma::MatrixLayout::RowMajor,
                    );
                    for kc in 0..bc / 16usize {
                        let a = cmma::Matrix::<M>::from_slice(
                            cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &ps.to_slice().slice(p * 16usize * bc + kc * 16usize, br * bc), bc as u32,
                        );
                        let b = cmma::Matrix::<M>::from_slice(
                            cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &vs.to_slice().slice(kc * 16usize * d + dn * 16usize, bc * d), d as u32,
                        );
                        cmma::execute::<M, M, f32, f32>(&a, &b, &acc, &acc);
                    }
                    cmma::store(
                        &mut of.to_slice_mut().slice_mut(p * 16usize * d + dn * 16usize, br * d),
                        &acc, d as u32, cmma::MatrixLayout::RowMajor,
                    );
                }
            }
            metal => {
                let p = PLANE_POS as usize;
                for ti in 0..2usize {
                    for tn in 0..d / 8usize {
                        let acc = cmma::Matrix::<f32>::from_value(
                            cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize,
                            cmma::MatrixLayout::Undefined, 0.0f32,
                        );
                        cmma::load_with_layout(
                            &acc, &of.to_slice().slice((p * 16usize + ti * 8usize) * d + tn * 8usize, br * d),
                            d as u32, cmma::MatrixLayout::RowMajor,
                        );
                        for k8 in 0..bc / 8usize {
                            let a = cmma::Matrix::<M>::from_slice(
                                cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &ps.to_slice().slice((p * 16usize + ti * 8usize) * bc + k8 * 8usize, br * bc), bc as u32,
                            );
                            let b = cmma::Matrix::<M>::from_slice(
                                cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &vs.to_slice().slice(k8 * 8usize * d + tn * 8usize, bc * d), d as u32,
                            );
                            cmma::execute::<M, M, f32, f32>(&a, &b, &acc, &acc);
                        }
                        cmma::store(
                            &mut of.to_slice_mut().slice_mut((p * 16usize + ti * 8usize) * d + tn * 8usize, br * d),
                            &acc, d as u32, cmma::MatrixLayout::RowMajor,
                        );
                    }
                }
            }
            default => {
                for e in 0..per_o {
                    let idx = u * per_o + e;
                    let r = idx / d;
                    let dd = idx % d;
                    let mut acc = of[idx];
                    for c in 0..bc {
                        acc += f32::cast_from(ps[r * bc + c]) * f32::cast_from(vs[c * d + dd]);
                    }
                    of[idx] = acc;
                }
            }
        };
        sync_cube();
    }

    for e in 0..per_o {
        let idx = u * per_o + e;
        let r = idx / d;
        let i = i0 + r;
        if i < len {
            out[(start + i) * hd + h * d + idx % d] = F::cast_from(of[idx] / row[br + r]);
        }
    }
    if u < br {
        let i = i0 + u;
        if i < len {
            lse[h * total + start + i] = row[u] + row[br + u].ln();
        }
    }
}

/// `delta[h, t] = Σ_d dO[t, h, d] O[t, h, d]`, a unit per (token, head).
#[kernel(targets(cuda, rocm, metal, cpu), unchecked)]
pub fn delta<F: Float>(
    out: &Array<F>,
    dout: &Array<F>,
    delta: &mut Array<f32>,
    meta: &Array<u32>,
    #[comptime] d: usize,
) {
    let heads = meta[0] as usize;
    let total = meta[2] as usize;
    let idx = ABSOLUTE_POS;
    if idx < total * heads {
        let t = idx / heads;
        let h = idx % heads;
        let at = (t * heads + h) * d;
        let mut acc = f32::new(0.0f32);
        for k in 0..d {
            acc += f32::cast_from(out[at + k]) * f32::cast_from(dout[at + k]);
        }
        delta[h * total + t] = acc;
    }
}

/// One tile of `bc` keys of one head against every query tile that reaches it (FlashAttention-2),
/// query tiles of `bc` rows: probabilities recomputed from `lse`, dK and dV accumulated in F32 and
/// written rotated back into `dqkv`, dQ added in F32 into `dq`. The shared tiles live in two
/// arrays, `mf` for the matrix operands and `sf` for F32; `lead` is `max(2 bc², bc d)`.
#[allow(clippy::too_many_arguments)]
#[kernel(targets(cuda, rocm, metal, cpu), unchecked)]
pub fn attend_back<F: Float, M: Float>(
    qkv: &Array<F>,
    cos: &Array<F>,
    sin: &Array<F>,
    cu: &Array<u32>,
    tiles: &Array<u32>,
    dout: &Array<F>,
    lse: &Array<f32>,
    delta: &Array<f32>,
    dq: &mut Array<Atomic<f32>>,
    dqkv: &mut Array<F>,
    meta: &Array<u32>,
    scale: &Array<f32>,
    #[comptime] d: usize,
    #[comptime] bc: usize,
    #[comptime] lead: usize,
    #[comptime] units: usize,
    #[comptime] target: Target,
) {
    let t = CUBE_POS_X as usize + meta[3] as usize;
    let h = CUBE_POS_Y as usize + meta[4] as usize;
    let u = UNIT_POS as usize;
    let heads = meta[0] as usize;
    let window = meta[1] as usize;
    let total = meta[2] as usize;
    let hd = heads * d;
    let ld = 3 * hd;
    let half = d / 2;
    let seq = tiles[2 * t] as usize;
    let j0 = tiles[2 * t + 1] as usize;
    let start = cu[seq] as usize;
    let len = cu[seq + 1] as usize - start;
    let sc = scale[0];

    // mf: ks, vs [bc, d]; qs, os [bc, d]; pm, dm [bc, bc] (keys by queries)
    let vs = bc * d;
    let qs = 2 * bc * d;
    let os = 3 * bc * d;
    let pm = 4 * bc * d;
    let dm = 4 * bc * d + bc * bc;
    let mut mf = SharedMemory::<M>::new(4 * bc * d + 2 * bc * bc);
    // sf: st, dpt [bc, bc], later dQ [bc, d]; dK, dV [bc, d]; lse and delta rows [2 bc]
    let dpt = bc * bc;
    let dka = lead;
    let dva = lead + bc * d;
    let rows = lead + 2 * bc * d;
    let mut sf = SharedMemory::<f32>::new(lead + 2 * bc * d + 2 * bc);

    let per_k = bc * half / units;
    for e in 0..per_k {
        let idx = u * per_k + e;
        let c = idx / half;
        let jj = idx % half;
        let j = j0 + c;
        let mut y1 = f32::new(0.0f32);
        let mut y2 = f32::new(0.0f32);
        let mut v1 = f32::new(0.0f32);
        let mut v2 = f32::new(0.0f32);
        if j < len {
            let at = (start + j) * ld + hd + h * d;
            let x1 = f32::cast_from(qkv[at + jj]);
            let x2 = f32::cast_from(qkv[at + jj + half]);
            let cs = f32::cast_from(cos[j * half + jj]);
            let sn = f32::cast_from(sin[j * half + jj]);
            y1 = x1 * cs - x2 * sn;
            y2 = x1 * sn + x2 * cs;
            v1 = f32::cast_from(qkv[at + hd + jj]);
            v2 = f32::cast_from(qkv[at + hd + jj + half]);
        }
        mf[c * d + jj] = M::cast_from(y1);
        mf[c * d + jj + half] = M::cast_from(y2);
        mf[vs + c * d + jj] = M::cast_from(v1);
        mf[vs + c * d + jj + half] = M::cast_from(v2);
    }
    let per_o = bc * d / units;
    for e in 0..per_o {
        sf[dka + u * per_o + e] = f32::new(0.0f32);
        sf[dva + u * per_o + e] = f32::new(0.0f32);
    }

    let mut lo = len - len;
    let mut hi = len;
    if window > 0 {
        if j0 + 1 > window {
            lo = (j0 + 1 - window) / bc * bc;
        }
        if j0 + bc + window - 1 < len {
            hi = j0 + bc + window - 1;
        }
    }
    let spans = (hi - lo + bc - 1) / bc;
    let per_q = bc * half / units;
    let per_s = bc * bc / units;
    for it in 0..spans {
        let i0 = lo + it * bc;
        sync_cube();
        for e in 0..per_q {
            let idx = u * per_q + e;
            let r = idx / half;
            let j = idx % half;
            let i = i0 + r;
            let mut y1 = f32::new(0.0f32);
            let mut y2 = f32::new(0.0f32);
            let mut o1 = f32::new(0.0f32);
            let mut o2 = f32::new(0.0f32);
            if i < len {
                let at = (start + i) * ld + h * d;
                let x1 = f32::cast_from(qkv[at + j]);
                let x2 = f32::cast_from(qkv[at + j + half]);
                let c = f32::cast_from(cos[i * half + j]);
                let s = f32::cast_from(sin[i * half + j]);
                y1 = (x1 * c - x2 * s) * sc;
                y2 = (x1 * s + x2 * c) * sc;
                let ot = (start + i) * hd + h * d;
                o1 = f32::cast_from(dout[ot + j]);
                o2 = f32::cast_from(dout[ot + j + half]);
            }
            mf[qs + r * d + j] = M::cast_from(y1);
            mf[qs + r * d + j + half] = M::cast_from(y2);
            mf[os + r * d + j] = M::cast_from(o1);
            mf[os + r * d + j + half] = M::cast_from(o2);
        }
        if u < bc {
            let i = i0 + u;
            let mut l = f32::new(0.0f32);
            let mut dl = f32::new(0.0f32);
            if i < len {
                l = lse[h * total + start + i];
                dl = delta[h * total + start + i];
            }
            sf[rows + u] = l;
            sf[rows + bc + u] = dl;
        }
        sync_cube();

        // Sᵀ = K Qᵀ and dPᵀ = V dOᵀ, keys by queries
        island! {
            cuda | rocm => {
                let p = PLANE_POS as usize;
                for g in 0..bc / 16usize {
                    let s = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
                    let dp = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
                    for dk in 0..d / 16usize {
                        let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &mf.to_slice().slice(p * 16usize * d + dk * 16usize, bc * d), d as u32);
                        let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                            &mf.to_slice().slice(qs + g * 16usize * d + dk * 16usize, qs + bc * d), d as u32);
                        cmma::execute::<M, M, f32, f32>(&a, &b, &s, &s);
                        let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &mf.to_slice().slice(vs + p * 16usize * d + dk * 16usize, vs + bc * d), d as u32);
                        let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                            &mf.to_slice().slice(os + g * 16usize * d + dk * 16usize, os + bc * d), d as u32);
                        cmma::execute::<M, M, f32, f32>(&a, &b, &dp, &dp);
                    }
                    cmma::store(&mut sf.to_slice_mut().slice_mut(p * 16usize * bc + g * 16usize, bc * bc), &s, bc as u32, cmma::MatrixLayout::RowMajor);
                    cmma::store(&mut sf.to_slice_mut().slice_mut(dpt + p * 16usize * bc + g * 16usize, dpt + bc * bc), &dp, bc as u32, cmma::MatrixLayout::RowMajor);
                }
            }
            metal => {
                let p = PLANE_POS as usize;
                for ti in 0..2usize {
                    let r0 = p * 16usize + ti * 8usize;
                    for g in 0..bc / 8usize {
                        let s = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, 0.0f32);
                        let dp = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, 0.0f32);
                        for dk in 0..d / 8usize {
                            let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &mf.to_slice().slice(r0 * d + dk * 8usize, bc * d), d as u32);
                            let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::ColMajor,
                                &mf.to_slice().slice(qs + g * 8usize * d + dk * 8usize, qs + bc * d), d as u32);
                            cmma::execute::<M, M, f32, f32>(&a, &b, &s, &s);
                            let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &mf.to_slice().slice(vs + r0 * d + dk * 8usize, vs + bc * d), d as u32);
                            let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::ColMajor,
                                &mf.to_slice().slice(os + g * 8usize * d + dk * 8usize, os + bc * d), d as u32);
                            cmma::execute::<M, M, f32, f32>(&a, &b, &dp, &dp);
                        }
                        cmma::store(&mut sf.to_slice_mut().slice_mut(r0 * bc + g * 8usize, bc * bc), &s, bc as u32, cmma::MatrixLayout::RowMajor);
                        cmma::store(&mut sf.to_slice_mut().slice_mut(dpt + r0 * bc + g * 8usize, dpt + bc * bc), &dp, bc as u32, cmma::MatrixLayout::RowMajor);
                    }
                }
            }
            default => {
                for e in 0..per_s {
                    let idx = u * per_s + e;
                    let c = idx / bc;
                    let r = idx % bc;
                    let mut s = f32::new(0.0f32);
                    let mut dp = f32::new(0.0f32);
                    for k in 0..d {
                        s += f32::cast_from(mf[c * d + k]) * f32::cast_from(mf[qs + r * d + k]);
                        dp += f32::cast_from(mf[vs + c * d + k]) * f32::cast_from(mf[os + r * d + k]);
                    }
                    sf[idx] = s;
                    sf[dpt + idx] = dp;
                }
            }
        };
        sync_cube();

        // Pᵀ = exp(Sᵀ − lse), dSᵀ = Pᵀ ⊙ (dPᵀ − delta); a pair out of reach is zero
        for e in 0..per_s {
            let idx = u * per_s + e;
            let c = idx / bc;
            let r = idx % bc;
            let i = i0 + r;
            let j = j0 + c;
            let mut near = true;
            if window > 0 {
                let gap = if i > j { i - j } else { j - i };
                near = gap < window;
            }
            let mut pv = f32::new(0.0f32);
            let mut ds = f32::new(0.0f32);
            if i < len && j < len && near {
                pv = (sf[idx] - sf[rows + r]).exp();
                ds = pv * (sf[dpt + idx] - sf[rows + bc + r]);
            }
            mf[pm + idx] = M::cast_from(pv);
            mf[dm + idx] = M::cast_from(ds);
        }
        sync_cube();

        // dV += Pᵀ dO and dK += dSᵀ Q, a plane per 16 keys; dQ = dS K, a plane per 16 queries
        island! {
            cuda | rocm => {
                let p = PLANE_POS as usize;
                for dn in 0..d / 16usize {
                    let v = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
                    let k = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
                    let q = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 16usize, 16usize, 16usize, cmma::MatrixLayout::Undefined, 0.0f32);
                    cmma::load_with_layout(&v, &sf.to_slice().slice(dva + p * 16usize * d + dn * 16usize, dva + bc * d), d as u32, cmma::MatrixLayout::RowMajor);
                    cmma::load_with_layout(&k, &sf.to_slice().slice(dka + p * 16usize * d + dn * 16usize, dka + bc * d), d as u32, cmma::MatrixLayout::RowMajor);
                    for kc in 0..bc / 16usize {
                        let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &mf.to_slice().slice(pm + p * 16usize * bc + kc * 16usize, pm + bc * bc), bc as u32);
                        let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &mf.to_slice().slice(os + kc * 16usize * d + dn * 16usize, os + bc * d), d as u32);
                        cmma::execute::<M, M, f32, f32>(&a, &b, &v, &v);
                        let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &mf.to_slice().slice(dm + p * 16usize * bc + kc * 16usize, dm + bc * bc), bc as u32);
                        let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &mf.to_slice().slice(qs + kc * 16usize * d + dn * 16usize, qs + bc * d), d as u32);
                        cmma::execute::<M, M, f32, f32>(&a, &b, &k, &k);
                        let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 16usize, 16usize, 16usize, cmma::MatrixLayout::ColMajor,
                            &mf.to_slice().slice(dm + kc * 16usize * bc + p * 16usize, dm + bc * bc), bc as u32);
                        let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 16usize, 16usize, 16usize, cmma::MatrixLayout::RowMajor,
                            &mf.to_slice().slice(kc * 16usize * d + dn * 16usize, bc * d), d as u32);
                        cmma::execute::<M, M, f32, f32>(&a, &b, &q, &q);
                    }
                    cmma::store(&mut sf.to_slice_mut().slice_mut(dva + p * 16usize * d + dn * 16usize, dva + bc * d), &v, d as u32, cmma::MatrixLayout::RowMajor);
                    cmma::store(&mut sf.to_slice_mut().slice_mut(dka + p * 16usize * d + dn * 16usize, dka + bc * d), &k, d as u32, cmma::MatrixLayout::RowMajor);
                    cmma::store(&mut sf.to_slice_mut().slice_mut(p * 16usize * d + dn * 16usize, bc * d), &q, d as u32, cmma::MatrixLayout::RowMajor);
                }
            }
            metal => {
                let p = PLANE_POS as usize;
                for ti in 0..2usize {
                    let r0 = p * 16usize + ti * 8usize;
                    for tn in 0..d / 8usize {
                        let v = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, 0.0f32);
                        let k = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, 0.0f32);
                        let q = cmma::Matrix::<f32>::from_value(cmma::MatrixIdent::Accumulator, 8usize, 8usize, 8usize, cmma::MatrixLayout::Undefined, 0.0f32);
                        cmma::load_with_layout(&v, &sf.to_slice().slice(dva + r0 * d + tn * 8usize, dva + bc * d), d as u32, cmma::MatrixLayout::RowMajor);
                        cmma::load_with_layout(&k, &sf.to_slice().slice(dka + r0 * d + tn * 8usize, dka + bc * d), d as u32, cmma::MatrixLayout::RowMajor);
                        for kc in 0..bc / 8usize {
                            let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &mf.to_slice().slice(pm + r0 * bc + kc * 8usize, pm + bc * bc), bc as u32);
                            let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &mf.to_slice().slice(os + kc * 8usize * d + tn * 8usize, os + bc * d), d as u32);
                            cmma::execute::<M, M, f32, f32>(&a, &b, &v, &v);
                            let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &mf.to_slice().slice(dm + r0 * bc + kc * 8usize, dm + bc * bc), bc as u32);
                            let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &mf.to_slice().slice(qs + kc * 8usize * d + tn * 8usize, qs + bc * d), d as u32);
                            cmma::execute::<M, M, f32, f32>(&a, &b, &k, &k);
                            let a = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::A, 8usize, 8usize, 8usize, cmma::MatrixLayout::ColMajor,
                                &mf.to_slice().slice(dm + kc * 8usize * bc + r0, dm + bc * bc), bc as u32);
                            let b = cmma::Matrix::<M>::from_slice(cmma::MatrixIdent::B, 8usize, 8usize, 8usize, cmma::MatrixLayout::RowMajor,
                                &mf.to_slice().slice(kc * 8usize * d + tn * 8usize, bc * d), d as u32);
                            cmma::execute::<M, M, f32, f32>(&a, &b, &q, &q);
                        }
                        cmma::store(&mut sf.to_slice_mut().slice_mut(dva + r0 * d + tn * 8usize, dva + bc * d), &v, d as u32, cmma::MatrixLayout::RowMajor);
                        cmma::store(&mut sf.to_slice_mut().slice_mut(dka + r0 * d + tn * 8usize, dka + bc * d), &k, d as u32, cmma::MatrixLayout::RowMajor);
                        cmma::store(&mut sf.to_slice_mut().slice_mut(r0 * d + tn * 8usize, bc * d), &q, d as u32, cmma::MatrixLayout::RowMajor);
                    }
                }
            }
            default => {
                for e in 0..per_o {
                    let idx = u * per_o + e;
                    let c = idx / d;
                    let dd = idx % d;
                    let mut gv = sf[dva + idx];
                    let mut gk = sf[dka + idx];
                    let mut gq = f32::new(0.0f32);
                    for r in 0..bc {
                        gv += f32::cast_from(mf[pm + c * bc + r]) * f32::cast_from(mf[os + r * d + dd]);
                        gk += f32::cast_from(mf[dm + c * bc + r]) * f32::cast_from(mf[qs + r * d + dd]);
                        gq += f32::cast_from(mf[dm + r * bc + c]) * f32::cast_from(mf[r * d + dd]);
                    }
                    sf[dva + idx] = gv;
                    sf[dka + idx] = gk;
                    sf[idx] = gq;
                }
            }
        };
        sync_cube();
        for e in 0..per_o {
            let idx = u * per_o + e;
            let i = i0 + idx / d;
            if i < len {
                dq[(start + i) * hd + h * d + idx % d].fetch_add(sf[idx]);
            }
        }
    }
    sync_cube();

    // dK rotated back, and dV, for this tile's keys
    for e in 0..per_k {
        let idx = u * per_k + e;
        let c = idx / half;
        let jj = idx % half;
        let j = j0 + c;
        if j < len {
            let cs = f32::cast_from(cos[j * half + jj]);
            let sn = f32::cast_from(sin[j * half + jj]);
            let x1 = sf[dka + c * d + jj];
            let x2 = sf[dka + c * d + jj + half];
            let at = (start + j) * ld + h * d;
            dqkv[at + hd + jj] = F::cast_from(x1 * cs + x2 * sn);
            dqkv[at + hd + jj + half] = F::cast_from(x2 * cs - x1 * sn);
            dqkv[at + 2 * hd + jj] = F::cast_from(sf[dva + c * d + jj]);
            dqkv[at + 2 * hd + jj + half] = F::cast_from(sf[dva + c * d + jj + half]);
        }
    }
}

/// dQ rotated back and scaled into `dqkv`: a unit per (token, head, dimension pair).
#[kernel(targets(cuda, rocm, metal, cpu), unchecked)]
pub fn rotate_back<F: Float>(
    dq: &Array<f32>,
    cos: &Array<F>,
    sin: &Array<F>,
    pos: &Array<u32>,
    dqkv: &mut Array<F>,
    meta: &Array<u32>,
    scale: &Array<f32>,
    #[comptime] d: usize,
) {
    let heads = meta[0] as usize;
    let total = meta[2] as usize;
    let half = d / 2;
    let idx = ABSOLUTE_POS;
    if idx < total * heads * half {
        let j = idx % half;
        let th = idx / half;
        let h = th % heads;
        let t = th / heads;
        let at = (t * heads + h) * d;
        let x1 = dq[at + j];
        let x2 = dq[at + j + half];
        let p = pos[t] as usize;
        let cs = f32::cast_from(cos[p * half + j]);
        let sn = f32::cast_from(sin[p * half + j]);
        let sc = scale[0];
        let out = t * 3 * heads * d + h * d;
        dqkv[out + j] = F::cast_from(sc * (x1 * cs + x2 * sn));
        dqkv[out + j + half] = F::cast_from(sc * (x2 * cs - x1 * sn));
    }
}

/// A schedule of [`attend`]: query rows per cube (16 a plane) and keys per tile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub br: usize,
    pub bc: usize,
}

/// Each `br`-row tile of every sequence of `lens`, as `[sequence, first row]` pairs, and the
/// sequences' first tokens.
pub fn layout(lens: &[usize], br: usize) -> (Vec<u32>, Vec<u32>) {
    let mut cu = Vec::with_capacity(lens.len() + 1);
    let mut tiles = Vec::new();
    let mut at = 0u32;
    for (s, &l) in lens.iter().enumerate() {
        cu.push(at);
        for i0 in (0..l).step_by(br) {
            tiles.extend([s as u32, i0 as u32]);
        }
        at += l as u32;
    }
    cu.push(at);
    (cu, tiles)
}

/// [`attend`] on `client` from host data: `qkv [T, 3·heads·64]`, tables `[P, 32]`; returns `out
/// [T, heads·64]` and `lse [heads, T]`. `window` is `None` for every key of a sequence. On the CPU
/// runtime the cubes run one at a time.
#[allow(clippy::too_many_arguments)]
pub fn attend_run<R: Runtime, F: Float + CubeElement, M: Float>(
    client: &ComputeClient<R>,
    qkv: &[F],
    cos: &[F],
    sin: &[F],
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    scale: f32,
    tile: Tile,
) -> (Vec<F>, Vec<f32>) {
    let target = Target::of(client);
    let (cu, tiles) = layout(lens, tile.br);
    let nt = (tiles.len() / 2) as u32;
    if target != Target::Cpu {
        let units = tile.br / 16 * client.properties().hardware.plane_size_max as usize;
        return attend_launch::<R, F, M>(
            client,
            qkv,
            cos,
            sin,
            &cu,
            &tiles,
            heads,
            window,
            scale,
            tile,
            units,
            (0, 0),
            (nt, heads as u32),
        );
    }
    // the CPU runtime one cube at a time, each writing only its own rows
    let t: usize = lens.iter().sum();
    let mut out = vec![0u8; t * heads * D * std::mem::size_of::<F>()];
    let mut lse = vec![0f32; heads * t];
    for x in 0..nt {
        for y in 0..heads as u32 {
            let (o, l) = attend_launch::<R, F, M>(
                client,
                qkv,
                cos,
                sin,
                &cu,
                &tiles,
                heads,
                window,
                scale,
                tile,
                tile.br,
                (x, y),
                (1, 1),
            );
            let size = std::mem::size_of::<F>();
            for (a, b) in out.chunks_mut(size).zip(F::as_bytes(&o).chunks(size)) {
                if b.iter().any(|&x| x != 0) {
                    a.copy_from_slice(b);
                }
            }
            for (a, b) in lse.iter_mut().zip(&l) {
                *a += b;
            }
        }
    }
    (F::from_bytes(&out).to_vec(), lse)
}

/// One launch of [`attend`] over the cubes `grid` from `base`, on fresh buffers.
#[allow(clippy::too_many_arguments)]
fn attend_launch<R: Runtime, F: Float + CubeElement, M: Float>(
    client: &ComputeClient<R>,
    qkv: &[F],
    cos: &[F],
    sin: &[F],
    cu: &[u32],
    tiles: &[u32],
    heads: usize,
    window: Option<usize>,
    scale: f32,
    tile: Tile,
    units: usize,
    base: (u32, u32),
    grid: (u32, u32),
) -> (Vec<F>, Vec<f32>) {
    let t = *cu.last().unwrap_or(&0) as usize;
    let win = window.map_or(0, |w| w as u32 + 1);
    let meta = [heads as u32, win, t as u32, base.0, base.1];
    let bufs = [
        client.create_from_slice(F::as_bytes(qkv)),
        client.create_from_slice(F::as_bytes(cos)),
        client.create_from_slice(F::as_bytes(sin)),
        client.create_from_slice(u32::as_bytes(cu)),
        client.create_from_slice(u32::as_bytes(tiles)),
        client.create_from_slice(&vec![0u8; t * heads * D * std::mem::size_of::<F>()]),
        client.create_from_slice(f32::as_bytes(&vec![0f32; heads * t])),
        client.create_from_slice(u32::as_bytes(&meta)),
        client.create_from_slice(f32::as_bytes(&[scale])),
    ];
    let lens = [
        qkv.len(),
        cos.len(),
        sin.len(),
        cu.len(),
        tiles.len(),
        t * heads * D,
        heads * t,
        5,
        1,
    ];
    unsafe {
        let arg = |i: usize| ArrayArg::from_raw_parts(bufs[i].clone(), lens[i]);
        attend::launch_unchecked::<F, M, R>(
            client,
            Grid::Static(grid.0, grid.1, 1),
            Block::new_1d(units as u32),
            arg(0),
            arg(1),
            arg(2),
            arg(3),
            arg(4),
            arg(5),
            arg(6),
            arg(7),
            arg(8),
            D,
            tile.br,
            tile.bc,
            units,
            Target::of(client),
        );
    }
    let out = F::from_bytes(&client.read_one_unchecked(bufs[5].clone())).to_vec();
    let lse = f32::from_bytes(&client.read_one_unchecked(bufs[6].clone())).to_vec();
    (out, lse)
}

/// [`attend_back`] with [`delta`] and [`rotate_back`] on `client` from host data: `out`, `dout
/// [T, heads·64]` and the forward's `lse`; returns `dqkv [T, 3·heads·64]`. Key tiles of `bc` rows.
#[allow(clippy::too_many_arguments)]
pub fn attend_back_run<R: Runtime, F: Float + CubeElement, M: Float>(
    client: &ComputeClient<R>,
    qkv: &[F],
    cos: &[F],
    sin: &[F],
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    scale: f32,
    out: &[F],
    dout: &[F],
    lse: &[f32],
    bc: usize,
) -> Vec<F> {
    let target = Target::of(client);
    let t: usize = lens.iter().sum();
    let (cu, tiles) = layout(lens, bc);
    let nt = (tiles.len() / 2) as u32;
    let pos: Vec<u32> = lens.iter().flat_map(|&l| 0..l as u32).collect();
    let win = window.map_or(0, |w| w as u32 + 1);
    let units = if target == Target::Cpu {
        bc
    } else {
        bc / 16 * client.properties().hardware.plane_size_max as usize
    };
    let lead = (2 * bc * bc).max(bc * D);
    let make = |bytes: &[u8]| client.create_from_slice(bytes);
    let (qh, csh, snh, ch, th) = (
        make(F::as_bytes(qkv)),
        make(F::as_bytes(cos)),
        make(F::as_bytes(sin)),
        make(u32::as_bytes(&cu)),
        make(u32::as_bytes(&tiles)),
    );
    let (oh, doh, lh, sh, ph) = (
        make(F::as_bytes(out)),
        make(F::as_bytes(dout)),
        make(f32::as_bytes(lse)),
        make(f32::as_bytes(&[scale])),
        make(u32::as_bytes(&pos)),
    );
    let dh = make(f32::as_bytes(&vec![0f32; heads * t]));
    let dqh = make(f32::as_bytes(&vec![0f32; t * heads * D]));
    let gh = make(&vec![0u8; t * 3 * heads * D * std::mem::size_of::<F>()]);
    let whole = make(u32::as_bytes(&[heads as u32, win, t as u32, 0, 0]));
    let count = (t * heads) as u32;
    unsafe {
        delta::launch_unchecked::<F, R>(
            client,
            Grid::Static(count.div_ceil(64), 1, 1),
            Block::new_1d(64),
            ArrayArg::from_raw_parts(oh.clone(), out.len()),
            ArrayArg::from_raw_parts(doh.clone(), dout.len()),
            ArrayArg::from_raw_parts(dh.clone(), heads * t),
            ArrayArg::from_raw_parts(whole.clone(), 5),
            D,
        );
    }
    let cubes: Vec<(u32, u32, u32, u32)> = if target == Target::Cpu {
        (0..nt)
            .flat_map(|x| (0..heads as u32).map(move |y| (x, y, 1, 1)))
            .collect()
    } else {
        vec![(0, 0, nt, heads as u32)]
    };
    let metas: Vec<_> = cubes
        .iter()
        .map(|&(x, y, _, _)| make(u32::as_bytes(&[heads as u32, win, t as u32, x, y])))
        .collect();
    for (&(_, _, gx, gy), mh) in cubes.iter().zip(&metas) {
        unsafe {
            attend_back::launch_unchecked::<F, M, R>(
                client,
                Grid::Static(gx, gy, 1),
                Block::new_1d(units as u32),
                ArrayArg::from_raw_parts(qh.clone(), qkv.len()),
                ArrayArg::from_raw_parts(csh.clone(), cos.len()),
                ArrayArg::from_raw_parts(snh.clone(), sin.len()),
                ArrayArg::from_raw_parts(ch.clone(), cu.len()),
                ArrayArg::from_raw_parts(th.clone(), tiles.len()),
                ArrayArg::from_raw_parts(doh.clone(), dout.len()),
                ArrayArg::from_raw_parts(lh.clone(), lse.len()),
                ArrayArg::from_raw_parts(dh.clone(), heads * t),
                ArrayArg::from_raw_parts(dqh.clone(), t * heads * D),
                ArrayArg::from_raw_parts(gh.clone(), t * 3 * heads * D),
                ArrayArg::from_raw_parts(mh.clone(), 5),
                ArrayArg::from_raw_parts(sh.clone(), 1),
                D,
                bc,
                lead,
                units,
                target,
            );
        }
        if target == Target::Cpu {
            let _ = client.read_one_unchecked(dqh.clone());
        }
    }
    let pairs = (t * heads * D / 2) as u32;
    unsafe {
        rotate_back::launch_unchecked::<F, R>(
            client,
            Grid::Static(pairs.div_ceil(64), 1, 1),
            Block::new_1d(64),
            ArrayArg::from_raw_parts(dqh.clone(), t * heads * D),
            ArrayArg::from_raw_parts(csh.clone(), cos.len()),
            ArrayArg::from_raw_parts(snh.clone(), sin.len()),
            ArrayArg::from_raw_parts(ph.clone(), t),
            ArrayArg::from_raw_parts(gh.clone(), t * 3 * heads * D),
            ArrayArg::from_raw_parts(whole.clone(), 5),
            ArrayArg::from_raw_parts(sh.clone(), 1),
            D,
        );
    }
    F::from_bytes(&client.read_one_unchecked(gh)).to_vec()
}

/// The composite's gradient in f64 with respect to `qkv` of `Σ out ⊙ dout`.
#[allow(clippy::too_many_arguments)]
pub fn attend_back_ref(
    qkv: &[f32],
    cos: &[f32],
    sin: &[f32],
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    scale: f32,
    dout: &[f32],
) -> Vec<f32> {
    let (d, half) = (D, D / 2);
    let t: usize = lens.iter().sum();
    let (hd, ld) = (heads * d, 3 * heads * d);
    let mut g = vec![0f32; t * ld];
    let rot = |x: &[f32], pos: usize| -> Vec<f64> {
        let mut out = vec![0f64; d];
        for j in 0..half {
            let (c, s) = (cos[pos * half + j] as f64, sin[pos * half + j] as f64);
            let (x1, x2) = (x[j] as f64, x[j + half] as f64);
            out[j] = x1 * c - x2 * s;
            out[j + half] = x1 * s + x2 * c;
        }
        out
    };
    let back = |y: &[f64], pos: usize| -> Vec<f64> {
        let mut out = vec![0f64; d];
        for j in 0..half {
            let (c, s) = (cos[pos * half + j] as f64, sin[pos * half + j] as f64);
            out[j] = y[j] * c + y[j + half] * s;
            out[j + half] = y[j + half] * c - y[j] * s;
        }
        out
    };
    let mut start = 0;
    for &l in lens {
        for h in 0..heads {
            let q: Vec<Vec<f64>> = (0..l)
                .map(|i| rot(&qkv[(start + i) * ld + h * d..][..d], i))
                .collect();
            let k: Vec<Vec<f64>> = (0..l)
                .map(|j| rot(&qkv[(start + j) * ld + hd + h * d..][..d], j))
                .collect();
            let v = |j: usize, dd: usize| qkv[(start + j) * ld + 2 * hd + h * d + dd] as f64;
            let o = |i: usize, dd: usize| dout[(start + i) * hd + h * d + dd] as f64;
            let (mut dq, mut dk, mut dv) = (
                vec![vec![0f64; d]; l],
                vec![vec![0f64; d]; l],
                vec![vec![0f64; d]; l],
            );
            for i in 0..l {
                let reach = |j: usize| window.is_none_or(|w| i.abs_diff(j) <= w);
                let s: Vec<f64> = (0..l)
                    .map(|j| {
                        if reach(j) {
                            scale as f64 * (0..d).map(|x| q[i][x] * k[j][x]).sum::<f64>()
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = s.iter().map(|x| (x - m).exp()).sum();
                let p: Vec<f64> = s.iter().map(|x| (x - m).exp() / sum).collect();
                let dp: Vec<f64> = (0..l)
                    .map(|j| (0..d).map(|x| o(i, x) * v(j, x)).sum::<f64>())
                    .collect();
                let dl: f64 = (0..l).map(|j| p[j] * dp[j]).sum();
                for j in 0..l {
                    let ds = p[j] * (dp[j] - dl) * scale as f64;
                    for x in 0..d {
                        dv[j][x] += p[j] * o(i, x);
                        dq[i][x] += ds * k[j][x];
                        dk[j][x] += ds * q[i][x];
                    }
                }
            }
            for i in 0..l {
                let at = (start + i) * ld + h * d;
                for (x, y) in back(&dq[i], i).iter().enumerate() {
                    g[at + x] = *y as f32;
                }
                for (x, y) in back(&dk[i], i).iter().enumerate() {
                    g[at + hd + x] = *y as f32;
                }
                for x in 0..d {
                    g[at + 2 * hd + x] = dv[i][x] as f32;
                }
            }
        }
        start += l;
    }
    g
}

/// The composite in f64: rotation, scaled scores, the window, a softmax over each sequence's
/// keys, the product with the values; `out [T, heads·64]`, `lse [heads, T]`.
#[allow(clippy::too_many_arguments)]
pub fn attend_ref(
    qkv: &[f32],
    cos: &[f32],
    sin: &[f32],
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    scale: f32,
) -> (Vec<f32>, Vec<f32>) {
    let (d, half) = (D, D / 2);
    let t: usize = lens.iter().sum();
    let (hd, ld) = (heads * d, 3 * heads * d);
    let mut out = vec![0f32; t * hd];
    let mut lse = vec![0f32; heads * t];
    let rot = |x: &[f32], pos: usize, out: &mut [f64]| {
        for j in 0..half {
            let (c, s) = (cos[pos * half + j] as f64, sin[pos * half + j] as f64);
            let (x1, x2) = (x[j] as f64, x[j + half] as f64);
            out[j] = x1 * c - x2 * s;
            out[j + half] = x1 * s + x2 * c;
        }
    };
    let mut start = 0;
    for &l in lens {
        for h in 0..heads {
            let mut ks = vec![0f64; l * d];
            for j in 0..l {
                let at = (start + j) * ld + hd + h * d;
                rot(&qkv[at..at + d], j, &mut ks[j * d..(j + 1) * d]);
            }
            for i in 0..l {
                let at = (start + i) * ld + h * d;
                let mut q = vec![0f64; d];
                rot(&qkv[at..at + d], i, &mut q);
                let reach = |j: usize| window.is_none_or(|w| i.abs_diff(j) <= w);
                let s: Vec<f64> = (0..l)
                    .map(|j| {
                        if reach(j) {
                            scale as f64 * (0..d).map(|k| q[k] * ks[j * d + k]).sum::<f64>()
                        } else {
                            f64::NEG_INFINITY
                        }
                    })
                    .collect();
                let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let sum: f64 = s.iter().map(|x| (x - m).exp()).sum();
                lse[h * t + start + i] = (m + sum.ln()) as f32;
                for dd in 0..d {
                    let mut acc = 0f64;
                    for j in 0..l {
                        let v = qkv[(start + j) * ld + 2 * hd + h * d + dd] as f64;
                        acc += (s[j] - m).exp() / sum * v;
                    }
                    out[(start + i) * hd + h * d + dd] = acc as f32;
                }
            }
        }
        start += l;
    }
    (out, lse)
}

/// The lengths the speed tests share with hanzo-nn's: 32 sequences of 64 to 191 tokens.
pub fn bench_lens() -> Vec<usize> {
    (0..32).map(|i| 64 + (i * 37) % 128).collect()
}

/// [`attend`] launched `iters` times on `client` after a warmup; mean ms a launch.
#[allow(clippy::too_many_arguments)]
pub fn attend_bench<R: Runtime, F: Float + CubeElement, M: Float>(
    client: &ComputeClient<R>,
    qkv: &[F],
    cos: &[F],
    sin: &[F],
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    scale: f32,
    tile: Tile,
    iters: usize,
) -> f64 {
    let t: usize = lens.iter().sum();
    let (cu, tiles) = layout(lens, tile.br);
    let nt = (tiles.len() / 2) as u32;
    let units = tile.br / 16 * client.properties().hardware.plane_size_max as usize;
    let win = window.map_or(0, |w| w as u32 + 1);
    let make = |bytes: &[u8]| client.create_from_slice(bytes);
    let bufs = [
        make(F::as_bytes(qkv)),
        make(F::as_bytes(cos)),
        make(F::as_bytes(sin)),
        make(u32::as_bytes(&cu)),
        make(u32::as_bytes(&tiles)),
        make(&vec![0u8; t * heads * D * std::mem::size_of::<F>()]),
        make(f32::as_bytes(&vec![0f32; heads * t])),
        make(u32::as_bytes(&[heads as u32, win, t as u32, 0, 0])),
        make(f32::as_bytes(&[scale])),
    ];
    let lens = [
        qkv.len(),
        cos.len(),
        sin.len(),
        cu.len(),
        tiles.len(),
        t * heads * D,
        heads * t,
        5,
        1,
    ];
    let launch = || unsafe {
        let arg = |i: usize| ArrayArg::from_raw_parts(bufs[i].clone(), lens[i]);
        attend::launch_unchecked::<F, M, R>(
            client,
            Grid::Static(nt, heads as u32, 1),
            Block::new_1d(units as u32),
            arg(0),
            arg(1),
            arg(2),
            arg(3),
            arg(4),
            arg(5),
            arg(6),
            arg(7),
            arg(8),
            D,
            tile.br,
            tile.bc,
            units,
            Target::of(client),
        );
    };
    for _ in 0..3 {
        launch();
    }
    let _ = client.read_one_unchecked(bufs[6].clone());
    let clock = std::time::Instant::now();
    for _ in 0..iters {
        launch();
    }
    let _ = client.read_one_unchecked(bufs[6].clone());
    clock.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// [`attend_back_run`]'s three kernels launched `iters` times after a warmup; mean ms a pass.
#[allow(clippy::too_many_arguments)]
pub fn attend_back_bench<R: Runtime, F: Float + CubeElement, M: Float>(
    client: &ComputeClient<R>,
    qkv: &[F],
    cos: &[F],
    sin: &[F],
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    scale: f32,
    out: &[F],
    dout: &[F],
    lse: &[f32],
    bc: usize,
    iters: usize,
) -> f64 {
    let t: usize = lens.iter().sum();
    let (cu, tiles) = layout(lens, bc);
    let nt = (tiles.len() / 2) as u32;
    let pos: Vec<u32> = lens.iter().flat_map(|&l| 0..l as u32).collect();
    let win = window.map_or(0, |w| w as u32 + 1);
    let units = bc / 16 * client.properties().hardware.plane_size_max as usize;
    let lead = (2 * bc * bc).max(bc * D);
    let make = |bytes: &[u8]| client.create_from_slice(bytes);
    let (qh, csh, snh, ch, th) = (
        make(F::as_bytes(qkv)),
        make(F::as_bytes(cos)),
        make(F::as_bytes(sin)),
        make(u32::as_bytes(&cu)),
        make(u32::as_bytes(&tiles)),
    );
    let (oh, doh, lh, sh, ph) = (
        make(F::as_bytes(out)),
        make(F::as_bytes(dout)),
        make(f32::as_bytes(lse)),
        make(f32::as_bytes(&[scale])),
        make(u32::as_bytes(&pos)),
    );
    let dh = make(f32::as_bytes(&vec![0f32; heads * t]));
    let dqh = make(f32::as_bytes(&vec![0f32; t * heads * D]));
    let gh = make(&vec![0u8; t * 3 * heads * D * std::mem::size_of::<F>()]);
    let mh = make(u32::as_bytes(&[heads as u32, win, t as u32, 0, 0]));
    let count = (t * heads) as u32;
    let pairs = (t * heads * D / 2) as u32;
    let launch = || unsafe {
        delta::launch_unchecked::<F, R>(
            client,
            Grid::Static(count.div_ceil(64), 1, 1),
            Block::new_1d(64),
            ArrayArg::from_raw_parts(oh.clone(), out.len()),
            ArrayArg::from_raw_parts(doh.clone(), dout.len()),
            ArrayArg::from_raw_parts(dh.clone(), heads * t),
            ArrayArg::from_raw_parts(mh.clone(), 5),
            D,
        );
        attend_back::launch_unchecked::<F, M, R>(
            client,
            Grid::Static(nt, heads as u32, 1),
            Block::new_1d(units as u32),
            ArrayArg::from_raw_parts(qh.clone(), qkv.len()),
            ArrayArg::from_raw_parts(csh.clone(), cos.len()),
            ArrayArg::from_raw_parts(snh.clone(), sin.len()),
            ArrayArg::from_raw_parts(ch.clone(), cu.len()),
            ArrayArg::from_raw_parts(th.clone(), tiles.len()),
            ArrayArg::from_raw_parts(doh.clone(), dout.len()),
            ArrayArg::from_raw_parts(lh.clone(), lse.len()),
            ArrayArg::from_raw_parts(dh.clone(), heads * t),
            ArrayArg::from_raw_parts(dqh.clone(), t * heads * D),
            ArrayArg::from_raw_parts(gh.clone(), t * 3 * heads * D),
            ArrayArg::from_raw_parts(mh.clone(), 5),
            ArrayArg::from_raw_parts(sh.clone(), 1),
            D,
            bc,
            lead,
            units,
            Target::of(client),
        );
        rotate_back::launch_unchecked::<F, R>(
            client,
            Grid::Static(pairs.div_ceil(64), 1, 1),
            Block::new_1d(64),
            ArrayArg::from_raw_parts(dqh.clone(), t * heads * D),
            ArrayArg::from_raw_parts(csh.clone(), cos.len()),
            ArrayArg::from_raw_parts(snh.clone(), sin.len()),
            ArrayArg::from_raw_parts(ph.clone(), t),
            ArrayArg::from_raw_parts(gh.clone(), t * 3 * heads * D),
            ArrayArg::from_raw_parts(mh.clone(), 5),
            ArrayArg::from_raw_parts(sh.clone(), 1),
            D,
        );
    };
    for _ in 0..3 {
        launch();
    }
    let _ = client.read_one_unchecked(dh.clone());
    let clock = std::time::Instant::now();
    for _ in 0..iters {
        launch();
    }
    let _ = client.read_one_unchecked(dh.clone());
    clock.elapsed().as_secs_f64() * 1e3 / iters as f64
}

/// [`attend`] at the fastest tile for `(device, shape)`, tuned once and cached ([`crate::tune`]).
#[allow(clippy::too_many_arguments)]
pub fn attend_tuned<R: Runtime, F: Float + CubeElement, M: Float>(
    client: &ComputeClient<R>,
    qkv: &[F],
    cos: &[F],
    sin: &[F],
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    scale: f32,
) -> crate::tune::Pick<Tile> {
    let t: usize = lens.iter().sum();
    let key = format!(
        "heads={heads},window={},tokens={},longest={}",
        window.map_or(0, |w| w + 1),
        t.next_power_of_two(),
        lens.iter().max().copied().unwrap_or(0).next_power_of_two()
    );
    let mut tuned = crate::tune::Tuned::new("packed_attend", key);
    let m = std::mem::size_of::<M>();
    for (name, tile) in TILES
        .into_iter()
        .filter(|(_, t)| fits(client, attend_shared(*t, m)))
    {
        tuned = tuned.variant(name, move |iters| {
            let ms = attend_bench::<R, F, M>(
                client, qkv, cos, sin, lens, heads, window, scale, tile, iters,
            );
            (tile, ms)
        });
    }
    tuned.pick(client)
}

/// [`attend_back_run`]'s pass at the fastest key block for `(device, shape)`, tuned once and
/// cached ([`crate::tune`]).
#[allow(clippy::too_many_arguments)]
pub fn attend_back_tuned<R: Runtime, F: Float + CubeElement, M: Float>(
    client: &ComputeClient<R>,
    qkv: &[F],
    cos: &[F],
    sin: &[F],
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    scale: f32,
    out: &[F],
    dout: &[F],
    lse: &[f32],
) -> crate::tune::Pick<usize> {
    let t: usize = lens.iter().sum();
    let key = format!(
        "heads={heads},window={},tokens={},longest={}",
        window.map_or(0, |w| w + 1),
        t.next_power_of_two(),
        lens.iter().max().copied().unwrap_or(0).next_power_of_two()
    );
    let mut tuned = crate::tune::Tuned::new("packed_attend_back", key);
    let m = std::mem::size_of::<M>();
    for (name, bc) in BACKS
        .into_iter()
        .filter(|(_, bc)| fits(client, attend_back_shared(*bc, m)))
    {
        tuned = tuned.variant(name, move |iters| {
            let ms = attend_back_bench::<R, F, M>(
                client, qkv, cos, sin, lens, heads, window, scale, out, dout, lse, bc, iters,
            );
            (bc, ms)
        });
    }
    tuned.pick(client)
}

/// Shared bytes [`attend`] takes at `tile`, its matrix operands `m` bytes each.
pub fn attend_shared(tile: Tile, m: usize) -> usize {
    let (br, bc) = (tile.br, tile.bc);
    m * (br * D + 2 * bc * D + br * bc) + 4 * (br * bc + br * D + 3 * br)
}

/// Shared bytes [`attend_back`] takes at `bc`, its matrix operands `m` bytes each.
pub fn attend_back_shared(bc: usize, m: usize) -> usize {
    m * (4 * bc * D + 2 * bc * bc) + 4 * ((2 * bc * bc).max(bc * D) + 2 * bc * D + 2 * bc)
}

/// Whether `bytes` of shared memory fit a cube on `client`'s device.
pub fn fits<R: Runtime>(client: &ComputeClient<R>, bytes: usize) -> bool {
    bytes <= client.properties().hardware.max_shared_memory_size
}

/// The schedules [`attend_tuned`] times.
pub const TILES: [(&str, Tile); 5] = [
    ("r16_c16", Tile { br: 16, bc: 16 }),
    ("r32_c32", Tile { br: 32, bc: 32 }),
    ("r64_c32", Tile { br: 64, bc: 32 }),
    ("r32_c64", Tile { br: 32, bc: 64 }),
    ("r64_c64", Tile { br: 64, bc: 64 }),
];

/// The key blocks [`attend_back_tuned`] times.
pub const BACKS: [(&str, usize); 3] = [("c16", 16), ("c32", 32), ("c64", 64)];

#[cfg(test)]
mod tests {
    use super::*;

    fn rnd(n: usize, seed: u64, amp: f32) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s % 20001) as f32 / 10000.0 - 1.0) * amp
            })
            .collect()
    }

    fn tables(p: usize) -> (Vec<f32>, Vec<f32>) {
        let half = D / 2;
        let mut cos = Vec::with_capacity(p * half);
        let mut sin = Vec::with_capacity(p * half);
        for pos in 0..p {
            for i in 0..half {
                let f = pos as f64 / 10000f64.powf(2.0 * i as f64 / D as f64);
                cos.push(f.cos() as f32);
                sin.push(f.sin() as f32);
            }
        }
        (cos, sin)
    }

    /// `max|a − b| / max|b|`.
    fn rel(a: &[f32], b: &[f32]) -> f32 {
        let top = b.iter().fold(0f32, |m, x| m.max(x.abs())).max(1e-6);
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / top
    }

    const LENS: [usize; 5] = [5, 40, 1, 33, 70];
    const HEADS: usize = 2;

    #[cfg(feature = "cpu")]
    #[test]
    fn attend_is_the_composite_on_cpu() {
        use cubecl::cpu::{CpuDevice, CpuRuntime};
        let c = CpuRuntime::client(&CpuDevice::default());
        let t: usize = LENS.iter().sum();
        let qkv = rnd(t * 3 * HEADS * D, 7, 1.0);
        let (cos, sin) = tables(128);
        let scale = (D as f32).powf(-0.5);
        for window in [None, Some(8)] {
            let (want, wlse) = attend_ref(&qkv, &cos, &sin, &LENS, HEADS, window, scale);
            let (got, glse) = attend_run::<CpuRuntime, f32, f32>(
                &c,
                &qkv,
                &cos,
                &sin,
                &LENS,
                HEADS,
                window,
                scale,
                Tile { br: 16, bc: 16 },
            );
            let (r, rl) = (rel(&got, &want), rel(&glse, &wlse));
            eprintln!("[packed cpu] window {window:?}: out {r:.2e}, lse {rl:.2e}");
            assert!(r < 1e-5 && rl < 1e-5, "window {window:?}: {r}, {rl}");
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn attend_is_the_composite_on_cuda() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        use half::bf16;
        let c = CudaRuntime::client(&CudaDevice::default());
        let t: usize = LENS.iter().sum();
        let round = |v: Vec<f32>| -> Vec<f32> {
            v.into_iter().map(|x| bf16::from_f32(x).to_f32()).collect()
        };
        let qkv = round(rnd(t * 3 * HEADS * D, 7, 1.0));
        let (cos, sin) = tables(128);
        let (cos, sin) = (round(cos), round(sin));
        let scale = (D as f32).powf(-0.5);
        let narrow = |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
        for (_, tile) in TILES {
            for window in [None, Some(8), Some(64)] {
                let (want, wlse) = attend_ref(&qkv, &cos, &sin, &LENS, HEADS, window, scale);
                let (got, glse) = attend_run::<CudaRuntime, bf16, bf16>(
                    &c,
                    &narrow(&qkv),
                    &narrow(&cos),
                    &narrow(&sin),
                    &LENS,
                    HEADS,
                    window,
                    scale,
                    tile,
                );
                let got: Vec<f32> = got.iter().map(|x| x.to_f32()).collect();
                let (r, rl) = (rel(&got, &want), rel(&glse, &wlse));
                eprintln!("[packed cuda] {tile:?} window {window:?}: out {r:.2e}, lse {rl:.2e}");
                assert!(
                    r < 2e-2 && rl < 1e-2,
                    "{tile:?} window {window:?}: {r}, {rl}"
                );
            }
        }
    }
    #[cfg(feature = "cuda")]
    #[test]
    fn attend_back_is_the_gradient_on_cuda() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        use half::bf16;
        let c = CudaRuntime::client(&CudaDevice::default());
        let t: usize = LENS.iter().sum();
        let round = |v: Vec<f32>| -> Vec<f32> {
            v.into_iter().map(|x| bf16::from_f32(x).to_f32()).collect()
        };
        let narrow = |v: &[f32]| -> Vec<bf16> { v.iter().map(|&x| bf16::from_f32(x)).collect() };
        let qkv = round(rnd(t * 3 * HEADS * D, 7, 1.0));
        let dout = round(rnd(t * HEADS * D, 9, 1.0));
        let (cos, sin) = tables(128);
        let (cos, sin) = (round(cos), round(sin));
        let scale = (D as f32).powf(-0.5);
        for (_, bc) in BACKS
            .into_iter()
            .filter(|(_, bc)| fits(&c, attend_back_shared(*bc, 2)))
        {
            for window in [None, Some(8), Some(64)] {
                let (out, lse) = attend_ref(&qkv, &cos, &sin, &LENS, HEADS, window, scale);
                let want = attend_back_ref(&qkv, &cos, &sin, &LENS, HEADS, window, scale, &dout);
                let got = attend_back_run::<CudaRuntime, bf16, bf16>(
                    &c,
                    &narrow(&qkv),
                    &narrow(&cos),
                    &narrow(&sin),
                    &LENS,
                    HEADS,
                    window,
                    scale,
                    &narrow(&out),
                    &narrow(&dout),
                    &lse,
                    bc,
                );
                let got: Vec<f32> = got.iter().map(|x| x.to_f32()).collect();
                let r = rel(&got, &want);
                eprintln!("[packed back cuda] bc {bc} window {window:?}: dqkv {r:.2e}");
                assert!(r < 3e-2, "bc {bc} window {window:?}: {r}");
            }
        }
    }

    /// Mean ms a pass at the shared bench shape per tile and key block, then the tuners' picks.
    fn speed<R: Runtime, F: Float + CubeElement, M: Float>(
        c: &ComputeClient<R>,
        name: &str,
        conv: fn(f32) -> F,
    ) {
        let lens = bench_lens();
        let heads = 12;
        let t: usize = lens.iter().sum();
        let narrow = |v: Vec<f32>| -> Vec<F> { v.into_iter().map(conv).collect() };
        let qkv = narrow(rnd(t * 3 * heads * D, 7, 1.0));
        let dout = narrow(rnd(t * heads * D, 9, 1.0));
        let (cos, sin) = tables(256);
        let (cos, sin) = (narrow(cos), narrow(sin));
        let scale = (D as f32).powf(-0.5);
        let m = std::mem::size_of::<M>();
        for window in [None, Some(64)] {
            for (tag, tile) in TILES
                .into_iter()
                .filter(|(_, t)| fits(c, attend_shared(*t, m)))
            {
                let ms = attend_bench::<R, F, M>(
                    c, &qkv, &cos, &sin, &lens, heads, window, scale, tile, 50,
                );
                eprintln!(
                    "[packed speed {name}] tokens {t} window {window:?} forward {tag}: {ms:.3} ms"
                );
            }
            let first = TILES
                .into_iter()
                .find(|(_, t)| fits(c, attend_shared(*t, m)))
                .unwrap()
                .1;
            let (out, lse) =
                attend_run::<R, F, M>(c, &qkv, &cos, &sin, &lens, heads, window, scale, first);
            for (_, bc) in BACKS
                .into_iter()
                .filter(|(_, bc)| fits(c, attend_back_shared(*bc, m)))
            {
                let ms = attend_back_bench::<R, F, M>(
                    c, &qkv, &cos, &sin, &lens, heads, window, scale, &out, &dout, &lse, bc, 20,
                );
                eprintln!("[packed speed {name}] tokens {t} window {window:?} backward bc {bc}: {ms:.3} ms");
            }
            let f = attend_tuned::<R, F, M>(c, &qkv, &cos, &sin, &lens, heads, window, scale);
            let b = attend_back_tuned::<R, F, M>(
                c, &qkv, &cos, &sin, &lens, heads, window, scale, &out, &dout, &lse,
            );
            eprintln!(
                "[packed speed {name}] window {window:?} tuned: forward {} (cached {}), backward {} (cached {})",
                f.winner, f.from_cache, b.winner, b.from_cache
            );
        }
    }

    /// `cargo test --features cuda packed_speed -- --ignored --nocapture`
    #[cfg(feature = "cuda")]
    #[test]
    #[ignore]
    fn packed_speed_on_cuda() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        use half::bf16;
        let c = CudaRuntime::client(&CudaDevice::default());
        speed::<CudaRuntime, bf16, bf16>(&c, "cuda", bf16::from_f32);
    }

    /// `cargo test --no-default-features --features metal packed_speed -- --ignored --nocapture`
    #[cfg(feature = "metal")]
    #[test]
    #[ignore]
    fn packed_speed_on_metal() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        use half::f16;
        let c = WgpuRuntime::client(&WgpuDevice::default());
        speed::<WgpuRuntime, f32, f16>(&c, "metal", |x| x);
    }

    #[cfg(feature = "metal")]
    #[test]
    fn attend_is_the_composite_on_metal() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        use half::f16;
        let c = WgpuRuntime::client(&WgpuDevice::default());
        let t: usize = LENS.iter().sum();
        let qkv = rnd(t * 3 * HEADS * D, 7, 1.0);
        let dout = rnd(t * HEADS * D, 9, 1.0);
        let (cos, sin) = tables(128);
        let scale = (D as f32).powf(-0.5);
        for window in [None, Some(8)] {
            let (want, wlse) = attend_ref(&qkv, &cos, &sin, &LENS, HEADS, window, scale);
            for (_, tile) in TILES
                .into_iter()
                .filter(|(_, t)| fits(&c, attend_shared(*t, 2)))
            {
                let (got, glse) = attend_run::<WgpuRuntime, f32, f16>(
                    &c, &qkv, &cos, &sin, &LENS, HEADS, window, scale, tile,
                );
                let (r, rl) = (rel(&got, &want), rel(&glse, &wlse));
                eprintln!("[packed metal] {tile:?} window {window:?}: out {r:.2e}, lse {rl:.2e}");
                assert!(
                    r < 1e-2 && rl < 1e-2,
                    "{tile:?} window {window:?}: {r}, {rl}"
                );
            }
            let grad = attend_back_ref(&qkv, &cos, &sin, &LENS, HEADS, window, scale, &dout);
            for (_, bc) in BACKS
                .into_iter()
                .filter(|(_, b)| fits(&c, attend_back_shared(*b, 2)))
            {
                let got = attend_back_run::<WgpuRuntime, f32, f16>(
                    &c, &qkv, &cos, &sin, &LENS, HEADS, window, scale, &want, &dout, &wlse, bc,
                );
                let r = rel(&got, &grad);
                eprintln!("[packed back metal] bc {bc} window {window:?}: dqkv {r:.2e}");
                assert!(r < 2e-2, "bc {bc} window {window:?}: {r}");
            }
        }
    }
}
