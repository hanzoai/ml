//! Bidirectional attention over packed sequences, straight from a packed QKV projection: the
//! encoder's inference path on the CPU.
//!
//! One parallel region per call: each (sequence, head) task rotates its queries and keys, scores
//! a block of at most [`BLOCK`] queries against the keys their window reaches with one GEMM,
//! takes the softmax of each row, and writes the block's output with a second GEMM into the
//! head's columns. Padding never enters: a sequence's keys are its own tokens.

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

    thread_local! {
        static SCRATCH: RefCell<Vec<f32>> = const { RefCell::new(Vec::new()) };
    }
    (0..lens.len() * heads).into_par_iter().for_each(|task| {
        let (seq, h) = (task / heads, task % heads);
        let (start, l) = (starts[seq], lens[seq]);
        if l == 0 {
            return;
        }
        SCRATCH.with(|cell| {
            let mut scratch = cell.borrow_mut();
            let need = 2 * l * d + BLOCK * l;
            if scratch.len() < need {
                scratch.resize(need, 0.);
            }
            let (q, rest) = scratch.split_at_mut(l * d);
            let (k, s) = rest.split_at_mut(l * d);
            // rotate: the first half of a head against the second, by position
            for i in 0..l {
                let row = &src[(start + i) * w3..(start + i + 1) * w3];
                let (cs, sn) = (
                    &cos[i * half..(i + 1) * half],
                    &sin[i * half..(i + 1) * half],
                );
                for (x, out, mul) in [
                    (&row[h * d..], &mut q[i * d..], scale),
                    (&row[hd + h * d..], &mut k[i * d..], 1.),
                ] {
                    for j in 0..half {
                        let (x1, x2) = (x[j], x[j + half]);
                        out[j] = (x1 * cs[j] - x2 * sn[j]) * mul;
                        out[j + half] = (x1 * sn[j] + x2 * cs[j]) * mul;
                    }
                }
            }
            let v = src[start * w3 + 2 * hd + h * d..].as_ptr();
            let mut i0 = 0;
            while i0 < l {
                let i1 = (i0 + BLOCK).min(l);
                let (ka, kb) = match window {
                    Some(w) => (i0.saturating_sub(w), (i1 - 1 + w + 1).min(l)),
                    None => (0, l),
                };
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
                    if let Some(w) = window {
                        for (j, x) in row.iter_mut().enumerate() {
                            if (ka + j).abs_diff(i) > w {
                                *x = f32::NEG_INFINITY;
                            }
                        }
                    }
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0f32;
                    for x in row.iter_mut() {
                        *x = (*x - max).exp();
                        sum += *x;
                    }
                    for x in row.iter_mut() {
                        *x /= sum;
                    }
                }
                // out[i0..i1, head] = probs [bm, bn] · v[ka..kb, head]
                unsafe {
                    mm(
                        (bm, d, bn),
                        (dst as *mut f32).add((start + i0) * hd + h * d),
                        (hd as isize, 1),
                        s.as_ptr(),
                        (bn as isize, 1),
                        v.add(ka * w3),
                        (w3 as isize, 1),
                    )
                };
                i0 = i1;
            }
        });
    });
    drop((qg, cg, sg));
    Tensor::from_vec(out, (t, hd), qkv.device())
}
