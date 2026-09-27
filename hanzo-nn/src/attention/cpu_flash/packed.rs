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
    for (j, x) in row.iter_mut().enumerate() {
        let key = ka + j;
        if key >= bound || window.is_some_and(|w| key.abs_diff(i) > w) {
            *x = f32::NEG_INFINITY;
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
                rotate(&row[h * d..], &mut q[i * d..], cs, sn, half, scale);
                rotate(&row[hd + h * d..], &mut k[i * d..], cs, sn, half, 1.);
            }
            let v = src[start * w3 + 2 * hd + h * d..].as_ptr();
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

    /// A chunk over the cache equals the block-causal pass at its positions, bit for bit; one
    /// chunk is the bidirectional pass; no key past a query's chunk is read.
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
            // the first chunk sees nothing past itself: equal to the pass over it alone
            let head = packed(&qkv.narrow(0, 0, 4)?, &[4], heads, window, &cos, &sin, 0.5)?
                .to_vec2::<f32>()?;
            assert_eq!(&full[..4], &head[..]);
            let (mut k, mut v) = (vec![0f32; l * hd], vec![0f32; l * hd]);
            let mut at = 0;
            let mut grown = Vec::new();
            for &c in &chunks {
                let part = qkv.narrow(0, at, c)?;
                let out = extend(&part, at, &mut k, &mut v, heads, window, &cos, &sin, 0.5)?;
                grown.extend(out.to_vec2::<f32>()?);
                at += c;
            }
            assert_eq!(grown, full);
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
