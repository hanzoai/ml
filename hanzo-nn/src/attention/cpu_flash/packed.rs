//! Bidirectional and block-causal attention over packed sequences, straight from a packed QKV
//! projection: the encoder's inference path on the CPU.
//!
//! One parallel region per call: each (sequence, head) task rotates its queries and keys, scores
//! a block of at most [`BLOCK`] queries against the keys their window and chunk reach with one
//! GEMM, takes the softmax of each row, and writes the block's output with a second GEMM into the
//! head's columns. Padding never enters: a sequence's keys are its own tokens.
//!
//! [`packed`] is bidirectional. [`chunked`] is block-causal over each sequence's chunks: a query
//! in chunk `c` reads the keys of chunks `0..=c`. [`extend`] runs one chunk over a cache of the
//! keys and values the earlier chunks left, so a sequence grows one chunk at a time and gives, at
//! every position, what [`chunked`] gives over the whole.

use hanzo_ml::{CpuStorage, Result, Storage, Tensor};
use rayon::prelude::*;
use std::cell::RefCell;

/// Queries scored together.
pub const BLOCK: usize = 64;

fn slice(t: &Tensor) -> Result<(impl std::ops::Deref<Target = Storage> + '_, usize)> {
    let (s, l) = t.storage_and_layout();
    if !l.is_contiguous() {
        hanzo_ml::bail!("packed attention: contiguous inputs only");
    }
    if !matches!(&*s, Storage::Cpu(CpuStorage::F32(_))) {
        hanzo_ml::bail!("packed attention: f32 on the CPU only");
    }
    Ok((s, l.start_offset()))
}

fn f32s(s: &Storage) -> &[f32] {
    match s {
        Storage::Cpu(CpuStorage::F32(v)) => v,
        _ => unreachable!("checked by slice"),
    }
}

/// `dst [m, n] = lhs [m, k] · rhs [k, n]`, each with its own row and column strides, on the
/// calling thread.
#[allow(clippy::too_many_arguments)]
unsafe fn mm(
    (m, n, k): (usize, usize, usize),
    dst: *mut f32,
    (dst_rs, dst_cs): (isize, isize),
    lhs: *const f32,
    (lhs_rs, lhs_cs): (isize, isize),
    rhs: *const f32,
    (rhs_rs, rhs_cs): (isize, isize),
) {
    gemm::gemm(
        m,
        n,
        k,
        dst,
        dst_cs,
        dst_rs,
        false,
        lhs,
        lhs_cs,
        lhs_rs,
        rhs,
        rhs_cs,
        rhs_rs,
        0f32,
        1f32,
        false,
        false,
        false,
        gemm::Parallelism::None,
    )
}

thread_local! {
    static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
}

/// Rotate one head's `x` (`d` wide) by the tables at position `pos` into `out`, scaled by `mul`.
#[inline]
fn rotate(x: &[f32], out: &mut [f32], cos: &[f32], sin: &[f32], half: usize, mul: f32) {
    for j in 0..half {
        let (x1, x2) = (x[j], x[j + half]);
        out[j] = (x1 * cos[j] - x2 * sin[j]) * mul;
        out[j + half] = (x1 * sin[j] + x2 * cos[j]) * mul;
    }
}

/// Softmax of `row` in place, entries at or past `bound` (an exclusive key index relative to
/// `ka`) and beyond `window` of query `i` masked out first.
#[inline]
fn softmax(row: &mut [f32], i: usize, ka: usize, bound: usize, window: Option<usize>) {
    let len = row.len();
    let (left, right) = if let Some(w) = window {
        let l = i.saturating_sub(w).saturating_sub(ka).min(len);
        let r = (i + w + 1).min(bound).saturating_sub(ka).min(len);
        (l, r)
    } else if ka + len > bound {
        (0, bound.saturating_sub(ka).min(len))
    } else {
        (0, len)
    };
    if left >= right {
        row.fill(f32::NAN);
        return;
    }
    row[..left].fill(0.0);
    row[right..].fill(0.0);
    let active = &mut row[left..right];
    let max = active.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for x in active.iter_mut() {
        *x = (*x - max).exp();
        sum += *x;
    }
    for x in active.iter_mut() {
        *x /= sum;
    }
}

/// Vectorized `e^x` for `x ≤ 0` (Cephes polynomial, within 2 ULP of `expf`), used by [`packed_dlsa`].
#[inline(always)]
fn exp_neg(x: f32) -> f32 {
    let x = x.max(-87.0);
    let n = (x * std::f32::consts::LOG2_E + 0.5).floor();
    let r = x - n * 0.693_359_4 + n * 2.121_944_4e-4;
    let p = ((((1.987_569_1e-4 * r + 1.398_199_9e-3) * r + 8.333_452e-3) * r + 4.166_579_6e-2) * r
        + 1.666_666_5e-1)
        * r
        + 0.5;
    let y = p * r * r + r + 1.0;
    y * f32::from_bits(((n as i32 + 127) << 23) as u32)
}

#[inline]
fn softmax_unmasked_fast(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0f32;
    for x in row.iter_mut() {
        let e = exp_neg(*x - max);
        *x = e;
        sum += e;
    }
    let inv = 1.0 / sum.max(1e-30);
    for x in row.iter_mut() {
        *x *= inv;
    }
}

/// The exclusive key bound of query `i` of a sequence whose chunk lengths are `chunks`: the end
/// of its chunk.
fn chunk_end(chunks: &[usize], i: usize) -> usize {
    let mut end = 0;
    for &c in chunks {
        end += c;
        if i < end {
            return end;
        }
    }
    end
}

fn check(qkv: &Tensor, lens: &[usize], heads: usize, cos: &Tensor, sin: &Tensor) -> Result<usize> {
    let (t, w3) = qkv.dims2()?;
    let (p, half) = cos.dims2()?;
    let d = 2 * half;
    if w3 != 3 * heads * d || sin.dims2()? != (p, half) {
        hanzo_ml::bail!(
            "packed attention: qkv {:?}, cos {:?}, {heads} heads",
            qkv.shape(),
            cos.shape()
        );
    }
    if lens.iter().sum::<usize>() != t || lens.iter().any(|&l| l > p) {
        hanzo_ml::bail!("packed attention: lengths {lens:?} over {t} tokens, {p} positions");
    }
    Ok(d)
}

/// Bidirectional multi-head attention over packed sequences, F32 on the CPU.
///
/// - `qkv [T, 3·H·D]`: each token's queries, keys and values, heads contiguous within each.
/// - `lens`: tokens per sequence, summing to `T`; token `t` of a sequence sits at position `t`.
/// - `window`: a query reads the keys within this distance only; `None` reads every key.
/// - `cos`, `sin [P, D/2]`: rotary tables over positions (`P` at least the longest sequence),
///   applied to queries and keys as `rotary_emb::rope` does, then queries scaled by `scale`.
///
/// Returns `[T, H·D]`: each token's heads' outputs, contiguous. Equal, to rounding, to rotating,
/// scaling, a masked softmax over each sequence's keys and the product with its values.
#[allow(clippy::too_many_arguments)]
pub fn packed(
    qkv: &Tensor,
    lens: &[usize],
    heads: usize,
    window: Option<usize>,
    cos: &Tensor,
    sin: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    core(qkv, lens, None, heads, window, cos, sin, scale)
}

/// [`packed`], block-causal: `chunks[s]` are sequence `s`'s chunk lengths (summing to
/// `lens[s]`), and a query in chunk `c` reads the keys of chunks `0..=c` only, within `window`.
/// One chunk per sequence is [`packed`].
#[allow(clippy::too_many_arguments)]
pub fn chunked(
    qkv: &Tensor,
    lens: &[usize],
    chunks: &[&[usize]],
    heads: usize,
    window: Option<usize>,
    cos: &Tensor,
    sin: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    if chunks.len() != lens.len()
        || chunks
            .iter()
            .zip(lens)
            .any(|(c, &l)| c.iter().sum::<usize>() != l || c.contains(&0))
    {
        hanzo_ml::bail!("packed attention: chunks {chunks:?} do not sum to {lens:?}");
    }
    core(qkv, lens, Some(chunks), heads, window, cos, sin, scale)
}

#[allow(clippy::too_many_arguments)]
fn core(
    qkv: &Tensor,
    lens: &[usize],
    chunks: Option<&[&[usize]]>,
    heads: usize,
    window: Option<usize>,
    cos: &Tensor,
    sin: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let d = check(qkv, lens, heads, cos, sin)?;
    let half = d / 2;
    let t = qkv.dim(0)?;
    let w3 = 3 * heads * d;
    let hd = heads * d;
    let (qg, qo) = slice(qkv)?;
    let (cg, co) = slice(cos)?;
    let (sg, so) = slice(sin)?;
    let (src, cos, sin) = (&f32s(&qg)[qo..], &f32s(&cg)[co..], &f32s(&sg)[so..]);
    let mut out = vec![0f32; t * hd];
    let starts: Vec<usize> = lens
        .iter()
        .scan(0, |s, &l| {
            let at = *s;
            *s += l;
            Some(at)
        })
        .collect();
    let dst = out.as_mut_ptr() as usize;

    (0..lens.len() * heads).into_par_iter().for_each(|task| {
        let (seq, h) = (task / heads, task % heads);
        let (start, l) = (starts[seq], lens[seq]);
        if l == 0 {
            return;
        }
        let bound = |i: usize| chunks.map_or(l, |c| chunk_end(c[seq], i));
        let use_vh = chunks.is_none() && l >= BLOCK;
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = (if use_vh { 3 } else { 2 }) * l * d + BLOCK * l;
            if scratch.len() < need {
                scratch.resize(need, 0.);
            }
            let (q, rest) = scratch.split_at_mut(l * d);
            let (k, rest) = rest.split_at_mut(l * d);
            let (vh, s) = if use_vh {
                rest.split_at_mut(l * d)
            } else {
                (&mut [][..], rest)
            };
            // rotate: the first half of a head against the second, by position
            for i in 0..l {
                let row = &src[(start + i) * w3..(start + i + 1) * w3];
                let (cs, sn) = (
                    &cos[i * half..(i + 1) * half],
                    &sin[i * half..(i + 1) * half],
                );
                rotate(&row[h * d..], &mut q[i * d..], cs, sn, half, scale);
                rotate(&row[hd + h * d..], &mut k[i * d..], cs, sn, half, 1.);
                if use_vh {
                    vh[i * d..(i + 1) * d]
                        .copy_from_slice(&row[2 * hd + h * d..2 * hd + (h + 1) * d]);
                }
            }
            let mut i0 = 0;
            while i0 < l {
                let i1 = (i0 + BLOCK).min(l);
                // the keys any query of the block reaches: its window, and its chunk's end
                let ka = window.map_or(0, |w| i0.saturating_sub(w));
                let kb = window.map_or(l, |w| (i1 + w).min(l)).min(bound(i1 - 1));
                let (bm, bn) = (i1 - i0, kb - ka);
                // scores [bm, bn] = q[i0..i1] · k[ka..kb]ᵀ
                unsafe {
                    mm(
                        (bm, bn, d),
                        s.as_mut_ptr(),
                        (bn as isize, 1),
                        q[i0 * d..].as_ptr(),
                        (d as isize, 1),
                        k[ka * d..].as_ptr(),
                        (1, d as isize),
                    )
                };
                for i in i0..i1 {
                    let row = &mut s[(i - i0) * bn..(i - i0 + 1) * bn];
                    softmax(row, i, ka, bound(i), window);
                }
                // out[i0..i1, head] = probs [bm, bn] · v[ka..kb, head]
                unsafe {
                    if use_vh {
                        mm(
                            (bm, d, bn),
                            (dst as *mut f32).add((start + i0) * hd + h * d),
                            (hd as isize, 1),
                            s.as_ptr(),
                            (bn as isize, 1),
                            vh[ka * d..].as_ptr(),
                            (d as isize, 1),
                        );
                    } else {
                        mm(
                            (bm, d, bn),
                            (dst as *mut f32).add((start + i0) * hd + h * d),
                            (hd as isize, 1),
                            s.as_ptr(),
                            (bn as isize, 1),
                            src[(start + ka) * w3 + 2 * hd + h * d..].as_ptr(),
                            (w3 as isize, 1),
                        );
                    }
                };
                i0 = i1;
            }
        });
    });
    drop((qg, cg, sg));
    Tensor::from_vec(out, (t, hd), qkv.device())
}

/// Coalesced contiguous key spans `(ka, kb)` and total selected tokens `bn` for one 64-query block.
#[derive(Clone, Debug)]
struct BlockSpans {
    spans: Vec<(usize, usize)>,
    bn: usize,
}

/// Build per-query-block coalesced key spans for sequence of length `l` from `q_idx` and `k_idx`.
fn build_sequence_dlsa_spans(
    q_idx: &[f32],
    k_idx: &[f32],
    l: usize,
    d_idx: usize,
    k_budget: usize,
    sink_tokens: usize,
    local_window: usize,
) -> Vec<BlockSpans> {
    let nb = l.div_ceil(BLOCK);
    let target_nb = k_budget.div_ceil(BLOCK).clamp(1, nb);
    if target_nb >= nb {
        return (0..nb)
            .map(|_| BlockSpans {
                spans: vec![(0, l)],
                bn: l,
            })
            .collect();
    }

    // Pool q_idx and k_idx into [nb, d_idx] bucket summaries (0.5 * mean + 0.5 * absmax, L2-normalized)
    let mut bq = vec![0f32; nb * d_idx];
    let mut bk = vec![0f32; nb * d_idx];
    bq.par_chunks_mut(d_idx)
        .zip(bk.par_chunks_mut(d_idx))
        .enumerate()
        .for_each(|(b, (q_out, k_out))| {
            let (t0, t1) = (b * BLOCK, ((b + 1) * BLOCK).min(l));
            let inv = 1.0 / ((t1 - t0) as f32);
            let mut q_max = vec![0f32; d_idx];
            let mut k_max = vec![0f32; d_idx];
            for t in t0..t1 {
                let qr = &q_idx[t * d_idx..(t + 1) * d_idx];
                let kr = &k_idx[t * d_idx..(t + 1) * d_idx];
                for c in 0..d_idx {
                    let (qv, kv) = (qr[c], kr[c]);
                    q_out[c] += qv;
                    k_out[c] += kv;
                    if qv.abs() > q_max[c].abs() {
                        q_max[c] = qv;
                    }
                    if kv.abs() > k_max[c].abs() {
                        k_max[c] = kv;
                    }
                }
            }
            let (mut qn, mut kn) = (0f32, 0f32);
            for c in 0..d_idx {
                q_out[c] = 0.5 * (q_out[c] * inv) + 0.5 * q_max[c];
                k_out[c] = 0.5 * (k_out[c] * inv) + 0.5 * k_max[c];
                qn += q_out[c] * q_out[c];
                kn += k_out[c] * k_out[c];
            }
            let q_scale = 1.0 / (qn.sqrt() + 1e-8);
            let k_scale = 1.0 / (kn.sqrt() + 1e-8);
            for c in 0..d_idx {
                q_out[c] *= q_scale;
                k_out[c] *= k_scale;
            }
        });

    let sink_nb = sink_tokens.div_ceil(BLOCK).min(nb);
    let local_nb = local_window.div_ceil(BLOCK);

    (0..nb)
        .into_par_iter()
        .map(|bi| {
            let mut forced = vec![false; nb];
            let mut cands: Vec<(f32, usize)> = Vec::with_capacity(nb);
            let mut sel: Vec<usize> = Vec::with_capacity(target_nb + 4);
            for b in 0..sink_nb {
                forced[b] = true;
                sel.push(b);
            }
            let lo = bi.saturating_sub(local_nb);
            let hi = (bi + local_nb).min(nb - 1);
            for b in lo..=hi {
                if !forced[b] {
                    forced[b] = true;
                    sel.push(b);
                }
            }
            if sel.len() < target_nb {
                let rem = target_nb - sel.len();
                let qv = &bq[bi * d_idx..(bi + 1) * d_idx];
                for bj in 0..nb {
                    if !forced[bj] {
                        let kv = &bk[bj * d_idx..(bj + 1) * d_idx];
                        let mut dot = 0f32;
                        for c in 0..d_idx {
                            dot += qv[c] * kv[c];
                        }
                        cands.push((dot, bj));
                    }
                }
                if cands.len() > rem {
                    cands.select_nth_unstable_by(rem - 1, |a, b| {
                        b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    cands.truncate(rem);
                }
                for &(_, bj) in &cands {
                    sel.push(bj);
                }
            }
            sel.sort_unstable();

            // Coalesce adjacent buckets into contiguous token spans (ka, kb)
            let mut spans: Vec<(usize, usize)> = Vec::with_capacity(sel.len());
            let mut bn = 0;
            for &b in &sel {
                let (ka, kb) = (b * BLOCK, ((b + 1) * BLOCK).min(l));
                bn += kb - ka;
                if let Some(last) = spans.last_mut() {
                    if last.1 == ka {
                        last.1 = kb;
                        continue;
                    }
                }
                spans.push((ka, kb));
            }
            BlockSpans { spans, bn }
        })
        .collect()
}

/// Bidirectional Block-Packed Distilled Latent Sparse Attention (D-LSA) over packed sequences,
/// F32 on the CPU.
///
/// Uses `gemm::gemm` over coalesced 64-token buckets per query block, preserving prefix
/// `sink_tokens`, bilateral `local_window`, and top-$K$ retrieved buckets up to `k_budget`.
#[allow(clippy::too_many_arguments)]
pub fn packed_dlsa(
    qkv: &Tensor,
    q_idx: &Tensor,
    k_idx: &Tensor,
    lens: &[usize],
    heads: usize,
    cos: &Tensor,
    sin: &Tensor,
    scale: f32,
    k_budget: usize,
    sink_tokens: usize,
    local_window: usize,
) -> Result<Tensor> {
    if lens.iter().all(|&l| l <= k_budget) {
        return packed(qkv, lens, heads, None, cos, sin, scale);
    }
    let d = check(qkv, lens, heads, cos, sin)?;
    let half = d / 2;
    let t = qkv.dim(0)?;
    let (tq, d_idx) = q_idx.dims2()?;
    let (tk, d_idx_k) = k_idx.dims2()?;
    if tq != t || tk != t || d_idx != d_idx_k {
        hanzo_ml::bail!(
            "packed_dlsa: q_idx {:?}, k_idx {:?} for {t} tokens",
            q_idx.shape(),
            k_idx.shape()
        );
    }
    let w3 = 3 * heads * d;
    let hd = heads * d;
    let (qg, qo) = slice(qkv)?;
    let (qig, qio) = slice(q_idx)?;
    let (kig, kio) = slice(k_idx)?;
    let (cg, co) = slice(cos)?;
    let (sg, so) = slice(sin)?;
    let (src, qi_s, ki_s, cos_s, sin_s) = (
        &f32s(&qg)[qo..],
        &f32s(&qig)[qio..],
        &f32s(&kig)[kio..],
        &f32s(&cg)[co..],
        &f32s(&sg)[so..],
    );

    let starts: Vec<usize> = lens
        .iter()
        .scan(0, |s, &l| {
            let at = *s;
            *s += l;
            Some(at)
        })
        .collect();

    // Precompute bucket spans once per sequence (shared across all heads)
    let seq_spans: Vec<Vec<BlockSpans>> = (0..lens.len())
        .map(|seq| {
            let (start, l) = (starts[seq], lens[seq]);
            if l == 0 {
                return Vec::new();
            }
            build_sequence_dlsa_spans(
                &qi_s[start * d_idx..(start + l) * d_idx],
                &ki_s[start * d_idx..(start + l) * d_idx],
                l,
                d_idx,
                k_budget,
                sink_tokens,
                local_window,
            )
        })
        .collect();

    let mut out = vec![0f32; t * hd];
    let dst = out.as_mut_ptr() as usize;

    (0..lens.len() * heads).into_par_iter().for_each(|task| {
        let (seq, h) = (task / heads, task % heads);
        let (start, l) = (starts[seq], lens[seq]);
        if l == 0 {
            return;
        }
        let blocks = &seq_spans[seq];
        let max_bn = blocks.iter().map(|b| b.bn).max().unwrap_or(0);
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = 3 * l * d + 2 * max_bn * d + BLOCK * max_bn;
            if scratch.len() < need {
                scratch.resize(need, 0.);
            }
            let (q, rest) = scratch.split_at_mut(l * d);
            let (k, rest) = rest.split_at_mut(l * d);
            let (vh, rest) = rest.split_at_mut(l * d);
            let (ks, rest) = rest.split_at_mut(max_bn * d);
            let (vs, s) = rest.split_at_mut(max_bn * d);

            for i in 0..l {
                let row = &src[(start + i) * w3..(start + i + 1) * w3];
                let (cs, sn) = (
                    &cos_s[i * half..(i + 1) * half],
                    &sin_s[i * half..(i + 1) * half],
                );
                rotate(&row[h * d..], &mut q[i * d..], cs, sn, half, scale);
                rotate(&row[hd + h * d..], &mut k[i * d..], cs, sn, half, 1.);
                vh[i * d..(i + 1) * d].copy_from_slice(&row[2 * hd + h * d..2 * hd + (h + 1) * d]);
            }

            for (bi, blk) in blocks.iter().enumerate() {
                let i0 = bi * BLOCK;
                let i1 = (i0 + BLOCK).min(l);
                let bm = i1 - i0;
                let bn = blk.bn;

                let (k_ptr, v_ptr) = if blk.spans.len() == 1 {
                    let (ka, _) = blk.spans[0];
                    (k[ka * d..].as_ptr(), vh[ka * d..].as_ptr())
                } else {
                    let mut off = 0;
                    for &(ka, kb) in &blk.spans {
                        let span_elems = (kb - ka) * d;
                        ks[off..off + span_elems].copy_from_slice(&k[ka * d..kb * d]);
                        vs[off..off + span_elems].copy_from_slice(&vh[ka * d..kb * d]);
                        off += span_elems;
                    }
                    (ks.as_ptr(), vs.as_ptr())
                };

                unsafe {
                    mm(
                        (bm, bn, d),
                        s.as_mut_ptr(),
                        (bn as isize, 1),
                        q[i0 * d..].as_ptr(),
                        (d as isize, 1),
                        k_ptr,
                        (1, d as isize),
                    );
                }
                for r in 0..bm {
                    let row = &mut s[r * bn..(r + 1) * bn];
                    softmax_unmasked_fast(row);
                }
                unsafe {
                    mm(
                        (bm, d, bn),
                        (dst as *mut f32).add((start + i0) * hd + h * d),
                        (hd as isize, 1),
                        s.as_ptr(),
                        (bn as isize, 1),
                        v_ptr,
                        (d as isize, 1),
                    );
                }
            }
        });
    });
    drop((qg, qig, kig, cg, sg));
    Tensor::from_vec(out, (t, hd), qkv.device())
}

/// One sequence's next chunk over the keys and values its earlier chunks left, F32 on the CPU.
///
/// - `qkv [n, 3·H·D]`: the chunk's tokens, at positions `at..at + n` of the sequence.
/// - `k`, `v [at + n, H·D]`: rows `0..at` hold every earlier token's rotated keys and values,
///   as this function wrote them; rows `at..` take the chunk's.
/// - `window`, `cos`, `sin`, `scale`: as [`packed`].
///
/// Returns `[n, H·D]`: the chunk's heads' outputs, equal at every position to [`chunked`] over
/// the whole sequence with this chunk last: a query reads every earlier token and its own chunk.
#[allow(clippy::too_many_arguments)]
pub fn extend(
    qkv: &Tensor,
    at: usize,
    k: &mut [f32],
    v: &mut [f32],
    heads: usize,
    window: Option<usize>,
    cos: &Tensor,
    sin: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    let (n, w3) = qkv.dims2()?;
    let (p, half) = cos.dims2()?;
    let d = 2 * half;
    let hd = heads * d;
    let l = at + n;
    if w3 != 3 * hd || sin.dims2()? != (p, half) {
        hanzo_ml::bail!(
            "packed attention: qkv {:?}, cos {:?}, {heads} heads",
            qkv.shape(),
            cos.shape()
        );
    }
    if l > p || k.len() < l * hd || v.len() < l * hd {
        hanzo_ml::bail!(
            "packed attention: a chunk of {n} at {at} over {p} positions, a cache of {} rows",
            k.len() / hd
        );
    }
    let (qg, qo) = slice(qkv)?;
    let (cg, co) = slice(cos)?;
    let (sg, so) = slice(sin)?;
    let (src, cos, sin) = (&f32s(&qg)[qo..], &f32s(&cg)[co..], &f32s(&sg)[so..]);
    let mut out = vec![0f32; n * hd];
    let dst = out.as_mut_ptr() as usize;
    let (kp, vp) = (k.as_mut_ptr() as usize, v.as_mut_ptr() as usize);

    (0..heads).into_par_iter().for_each(|h| {
        // each head reads and writes its own `d` columns of the cache rows and of `out`, through
        // raw pointers: no two tasks touch the same element
        let (kp, vp) = (kp as *mut f32, vp as *mut f32);
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = n * d + d + BLOCK * l;
            if scratch.len() < need {
                scratch.resize(need, 0.);
            }
            let (q, rest) = scratch.split_at_mut(n * d);
            let (kr, s) = rest.split_at_mut(d);
            for i in 0..n {
                let row = &src[i * w3..(i + 1) * w3];
                let pos = at + i;
                let (cs, sn) = (
                    &cos[pos * half..(pos + 1) * half],
                    &sin[pos * half..(pos + 1) * half],
                );
                rotate(&row[h * d..], &mut q[i * d..], cs, sn, half, scale);
                rotate(&row[hd + h * d..], kr, cs, sn, half, 1.);
                // SAFETY: row `pos` of the cache exists (checked above) and columns
                // `h·d..(h+1)·d` are this task's alone.
                unsafe {
                    std::ptr::copy_nonoverlapping(kr.as_ptr(), kp.add(pos * hd + h * d), d);
                    std::ptr::copy_nonoverlapping(
                        row[2 * hd + h * d..].as_ptr(),
                        vp.add(pos * hd + h * d),
                        d,
                    );
                }
            }
            let mut i0 = 0;
            while i0 < n {
                let i1 = (i0 + BLOCK).min(n);
                let ka = window.map_or(0, |w| (at + i0).saturating_sub(w));
                let kb = window.map_or(l, |w| (at + i1 + w).min(l));
                let (bm, bn) = (i1 - i0, kb - ka);
                unsafe {
                    mm(
                        (bm, bn, d),
                        s.as_mut_ptr(),
                        (bn as isize, 1),
                        q[i0 * d..].as_ptr(),
                        (d as isize, 1),
                        kp.add(ka * hd + h * d),
                        (1, hd as isize),
                    )
                };
                for i in i0..i1 {
                    let row = &mut s[(i - i0) * bn..(i - i0 + 1) * bn];
                    softmax(row, at + i, ka, l, window);
                }
                unsafe {
                    mm(
                        (bm, d, bn),
                        (dst as *mut f32).add(i0 * hd + h * d),
                        (hd as isize, 1),
                        s.as_ptr(),
                        (bn as isize, 1),
                        vp.add(ka * hd + h * d),
                        (hd as isize, 1),
                    )
                };
                i0 = i1;
            }
        });
    });
    drop((qg, cg, sg));
    Tensor::from_vec(out, (n, hd), qkv.device())
}

#[cfg(test)]
mod tests {
    use super::*;
    use hanzo_ml::Device;

    fn tables(p: usize, half: usize) -> (Tensor, Tensor) {
        let dev = Device::Cpu;
        let inv: Vec<f32> = (0..half)
            .map(|i| 1f32 / 10000f32.powf(i as f32 / half as f32))
            .collect();
        let mut c = Vec::with_capacity(p * half);
        let mut s = Vec::with_capacity(p * half);
        for pos in 0..p {
            for &f in &inv {
                c.push((pos as f32 * f).cos());
                s.push((pos as f32 * f).sin());
            }
        }
        (
            Tensor::from_vec(c, (p, half), &dev).unwrap(),
            Tensor::from_vec(s, (p, half), &dev).unwrap(),
        )
    }

    /// The largest elementwise difference between two equally shaped outputs.
    fn gap(a: &[Vec<f32>], b: &[Vec<f32>]) -> f32 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .flatten()
            .zip(b.iter().flatten())
            .map(|(x, y)| (x - y).abs())
            .fold(0f32, f32::max)
    }

    /// A chunk over the cache equals the block-causal pass at its positions (to the last bits of
    /// an f32 sum where two kernels compute it); one chunk is the bidirectional pass, bit for
    /// bit; no key past a query's chunk is read.
    #[test]
    fn a_chunk_over_the_cache_is_the_chunked_pass() -> Result<()> {
        let dev = Device::Cpu;
        let (heads, d, l) = (2, 8, 21);
        let hd = heads * d;
        let (cos, sin) = tables(32, d / 2);
        let qkv = Tensor::randn(0f32, 1., (l, 3 * hd), &dev)?;
        for window in [None, Some(3)] {
            let whole = packed(&qkv, &[l], heads, window, &cos, &sin, 0.5)?;
            let one = chunked(&qkv, &[l], &[&[l]], heads, window, &cos, &sin, 0.5)?;
            assert_eq!(whole.to_vec2::<f32>()?, one.to_vec2::<f32>()?);
            let chunks = [4usize, 7, 10];
            let full = chunked(&qkv, &[l], &[&chunks], heads, window, &cos, &sin, 0.5)?
                .to_vec2::<f32>()?;
            // the first chunk sees nothing past itself: equal to the pass over it alone. The two
            // are different kernels (packed over 4 rows, chunked over 21), so they agree to the
            // last bits of an f32 sum, not bit for bit; a key read past the chunk would be off by
            // the size of a score.
            let head = packed(&qkv.narrow(0, 0, 4)?, &[4], heads, window, &cos, &sin, 0.5)?
                .to_vec2::<f32>()?;
            let worst = gap(&full[..4], &head);
            assert!(
                worst <= 1e-6,
                "first chunk differs from its own pass by {worst}"
            );
            let (mut k, mut v) = (vec![0f32; l * hd], vec![0f32; l * hd]);
            let mut at = 0;
            let mut grown = Vec::new();
            for &c in &chunks {
                let part = qkv.narrow(0, at, c)?;
                let out = extend(&part, at, &mut k, &mut v, heads, window, &cos, &sin, 0.5)?;
                grown.extend(out.to_vec2::<f32>()?);
                at += c;
            }
            // the cache path (`extend`) and the block pass are different kernels too
            let worst = gap(&grown, &full);
            assert!(
                worst <= 1e-6,
                "the cache path differs from the block pass by {worst}"
            );
        }
        // two sequences packed, chunked differently
        let qkv2 = Tensor::randn(0f32, 1., (l + 5, 3 * hd), &dev)?;
        let both = chunked(
            &qkv2,
            &[l, 5],
            &[&[4, 7, 10], &[5]],
            heads,
            Some(3),
            &cos,
            &sin,
            0.5,
        )?
        .to_vec2::<f32>()?;
        let first = chunked(
            &qkv2.narrow(0, 0, l)?,
            &[l],
            &[&[4, 7, 10]],
            heads,
            Some(3),
            &cos,
            &sin,
            0.5,
        )?
        .to_vec2::<f32>()?;
        assert_eq!(&both[..l], &first[..]);
        assert!(chunked(&qkv, &[l], &[&[4, 7]], heads, None, &cos, &sin, 0.5).is_err());
        Ok(())
    }
}
