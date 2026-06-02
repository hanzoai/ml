//! DeltaSoup: aggregate multiple compressed adapters into one.
//!
//! The cheapest path for federation is:
//!
//! 1. Each worker compresses its `(full - base)` into a [`BitDeltaAdapter`].
//! 2. Coordinator collects N adapters, dequantizes each into a delta tensor,
//!    aggregates coordinate-wise, then re-compresses or applies on top of the
//!    base.
//!
//! This module just handles step 2 — the math layer. It's intentionally
//! shape-agnostic so you can also pass in raw `Tensor` deltas from any source.

use candle_core::{DType, Tensor};
use serde::{Deserialize, Serialize};

use crate::{bitdelta::BitDeltaAdapter, Error, Result};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AggregateMethod {
    Mean,
    Median,
    /// Trim `floor(trim * N)` from each tail; falls back to Mean for N < 4.
    TrimmedMean { trim: f32 },
    /// Blanchard et al. 2017. `f` = max assumed Byzantine workers.
    /// Requires N >= 2f + 3.
    Krum { f: usize },
    /// Mean of the top-m by Krum score.
    MultiKrum { f: usize, m: usize },
}

/// Aggregate raw f32 deltas coordinate-wise.
pub fn aggregate(method: AggregateMethod, deltas: &[Tensor]) -> Result<Tensor> {
    if deltas.is_empty() {
        return Err(Error::Empty("soup::aggregate: no deltas"));
    }
    let dev = deltas[0].device().clone();
    let shape = deltas[0].dims().to_vec();
    for d in deltas {
        if d.dims() != shape.as_slice() {
            return Err(Error::ShapeMismatch {
                full: shape.clone(),
                base: d.dims().to_vec(),
            });
        }
    }
    let numel: usize = shape.iter().product();
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(deltas.len());
    for d in deltas {
        let v: Vec<f32> = d.flatten_all()?.to_dtype(DType::F32)?.to_vec1()?;
        rows.push(v);
    }
    let out: Vec<f32> = match method {
        AggregateMethod::Mean => coord_mean(&rows, numel),
        AggregateMethod::Median => coord_median(&rows, numel),
        AggregateMethod::TrimmedMean { trim } => coord_trimmed_mean(&rows, numel, trim),
        AggregateMethod::Krum { f } => krum(&rows, numel, f, 1)?,
        AggregateMethod::MultiKrum { f, m } => krum(&rows, numel, f, m)?,
    };
    Ok(Tensor::from_vec(out, shape.as_slice(), &dev)?)
}

/// Convenience: aggregate a slice of `BitDeltaAdapter`s, returning a single
/// `Tensor` delta on `device`. Re-compress with [`BitDeltaAdapter::
/// compress_against_full`] (using a zero base) if you want the result back
/// as a compressed adapter.
pub fn aggregate_adapters(
    method: AggregateMethod,
    adapters: &[BitDeltaAdapter],
    device: &candle_core::Device,
) -> Result<Tensor> {
    if adapters.is_empty() {
        return Err(Error::Empty("soup::aggregate_adapters: no adapters"));
    }
    let mut deltas = Vec::with_capacity(adapters.len());
    for a in adapters {
        deltas.push(a.decode(device)?);
    }
    aggregate(method, &deltas)
}

fn coord_mean(rows: &[Vec<f32>], numel: usize) -> Vec<f32> {
    let n = rows.len() as f32;
    let mut out = vec![0.0_f32; numel];
    for r in rows {
        for (i, &v) in r.iter().enumerate() {
            out[i] += v;
        }
    }
    for x in &mut out {
        *x /= n;
    }
    out
}

fn coord_median(rows: &[Vec<f32>], numel: usize) -> Vec<f32> {
    let n = rows.len();
    let mut out = Vec::with_capacity(numel);
    let mut col = Vec::with_capacity(n);
    for i in 0..numel {
        col.clear();
        for r in rows {
            col.push(r[i]);
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = if n % 2 == 1 {
            col[n / 2]
        } else {
            0.5 * (col[n / 2 - 1] + col[n / 2])
        };
        out.push(m);
    }
    out
}

fn coord_trimmed_mean(rows: &[Vec<f32>], numel: usize, trim: f32) -> Vec<f32> {
    let n = rows.len();
    if n < 4 {
        return coord_mean(rows, numel);
    }
    let raw = (trim * n as f32).floor() as usize;
    let trim_n = raw.max(1).min((n - 1) / 2);
    let keep = n - 2 * trim_n;
    let mut out = Vec::with_capacity(numel);
    let mut col = Vec::with_capacity(n);
    for i in 0..numel {
        col.clear();
        for r in rows {
            col.push(r[i]);
        }
        col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let slice = &col[trim_n..n - trim_n];
        debug_assert_eq!(slice.len(), keep);
        let s: f32 = slice.iter().sum();
        out.push(s / keep as f32);
    }
    out
}

fn krum(rows: &[Vec<f32>], numel: usize, f: usize, m: usize) -> Result<Vec<f32>> {
    let n = rows.len();
    if n < 2 * f + 3 {
        return Err(Error::NotEnough { needed: 2 * f + 3, got: n });
    }
    if m == 0 || m > n {
        return Err(Error::NotEnough { needed: m.max(1), got: n });
    }
    let mut dist = vec![vec![0.0_f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let mut s = 0.0_f32;
            for k in 0..numel {
                let d = rows[i][k] - rows[j][k];
                s += d * d;
            }
            dist[i][j] = s;
            dist[j][i] = s;
        }
    }
    let take = n.saturating_sub(f + 2);
    let mut scores: Vec<(usize, f32)> = (0..n)
        .map(|i| {
            let mut row: Vec<f32> = (0..n).filter(|&j| j != i).map(|j| dist[i][j]).collect();
            row.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let s: f32 = row.iter().take(take).sum();
            (i, s)
        })
        .collect();
    scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    let chosen: Vec<usize> = scores.into_iter().take(m).map(|(i, _)| i).collect();
    let mut out = vec![0.0_f32; numel];
    for &idx in &chosen {
        for k in 0..numel {
            out[k] += rows[idx][k];
        }
    }
    let denom = chosen.len() as f32;
    for x in &mut out {
        *x /= denom;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn t(v: &[f32]) -> Tensor {
        Tensor::from_vec(v.to_vec(), v.len(), &Device::Cpu).unwrap()
    }

    #[test]
    fn mean_matches_arith() {
        let deltas = vec![t(&[1.0, 2.0]), t(&[3.0, 4.0]), t(&[5.0, 6.0])];
        let out: Vec<f32> = aggregate(AggregateMethod::Mean, &deltas).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![3.0, 4.0]);
    }

    #[test]
    fn median_picks_middle() {
        let deltas = vec![t(&[1.0]), t(&[100.0]), t(&[2.0])];
        let out: Vec<f32> = aggregate(AggregateMethod::Median, &deltas).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![2.0]);
    }

    #[test]
    fn trimmed_mean_drops_outliers() {
        let deltas = vec![t(&[1.0]), t(&[2.0]), t(&[3.0]), t(&[100.0]), t(&[-100.0])];
        let out: Vec<f32> =
            aggregate(AggregateMethod::TrimmedMean { trim: 0.2 }, &deltas).unwrap().to_vec1().unwrap();
        assert_eq!(out, vec![2.0]);
    }

    #[test]
    fn krum_rejects_outlier() {
        let deltas = vec![
            t(&[0.0, 0.0]),
            t(&[0.1, 0.0]),
            t(&[0.0, 0.1]),
            t(&[0.1, 0.1]),
            t(&[1000.0, 1000.0]),
        ];
        let out: Vec<f32> = aggregate(AggregateMethod::Krum { f: 1 }, &deltas).unwrap().to_vec1().unwrap();
        assert!(out[0].abs() < 1.0 && out[1].abs() < 1.0);
    }

    #[test]
    fn aggregate_adapters_flow() {
        use crate::bitdelta::BitDeltaAdapter;
        let dev = Device::Cpu;
        let base = Tensor::zeros((4, 4), DType::F32, &dev).unwrap();
        let a = BitDeltaAdapter::compress_against_full(
            &Tensor::from_vec(vec![0.1_f32; 16], (4, 4), &dev).unwrap(),
            &base,
        )
        .unwrap();
        let b = BitDeltaAdapter::compress_against_full(
            &Tensor::from_vec(vec![-0.1_f32; 16], (4, 4), &dev).unwrap(),
            &base,
        )
        .unwrap();
        let avg = aggregate_adapters(AggregateMethod::Mean, &[a, b], &dev).unwrap();
        // Each entry should be ~0 (0.1 and -0.1 averaged).
        let v: Vec<f32> = avg.flatten_all().unwrap().to_vec1().unwrap();
        for x in v {
            assert!(x.abs() < 1e-6);
        }
    }
}
