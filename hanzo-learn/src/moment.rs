//! Sufficient statistics, as values that combine.
//!
//! A scaler does not need the data. It needs a few numbers per feature that the data
//! implies. Naming those numbers, and making them combine associatively, is what turns
//! one implementation into a single-threaded fit, a parallel fit, an out-of-core fit and
//! a multi-device fit with no second code path anywhere:
//!
//! ```text
//!   one matrix     Moments::of(x)
//!   many chunks    chunks.fold(Moments::zero(p), |a, c| a.merge(&Moments::of(c)))
//!   many threads   Moments::par(x)                  // rayon reduce, the SAME merge
//!   many devices   ship a Moments (2·p + p f64), merge on arrival
//! ```
//!
//! Both types here are monoids: [`Moments::zero`] and [`Extent::zero`] are identities and
//! `merge` is associative, which the tests check on data chosen to be numerically
//! awkward. That is the entire scale story for everything in [`crate::scale`] — rows are
//! unbounded and state is `O(p)`, so data larger than memory never has to be resident.
//!
//! # Why two passes over a chunk
//!
//! [`Moments::of`] walks its chunk twice: once for the mean, once for the deviations from
//! it. That is the textbook variance, and it is what `numpy` computes, so a single-chunk
//! fit agrees with scikit-learn to the last few bits. Merging chunks uses Chan's pairwise
//! correction, which is what scikit-learn's own incremental path uses, so a chunked fit
//! agrees with a chain of `partial_fit` calls. The chunk is already in memory; the second
//! walk costs bandwidth, not I/O — and the reference implementation being compared
//! against pays for two walks as well.

use rayon::prelude::*;

use crate::Matrix;

/// Below this many values a parallel fit spends more on coordination than it saves on
/// arithmetic, so the `par` constructors run serially. A threshold, not a mode: the
/// answer is the same either side of it, to within the merge order.
const SERIAL: usize = 1 << 16;

/// Count, mean, and sum of squared deviations, per feature.
#[derive(Debug, Clone, PartialEq)]
pub struct Moments {
    n: u64,
    mean: Vec<f64>,
    square: Vec<f64>,
}

impl Moments {
    /// The identity: no observations, so merging it changes nothing.
    pub fn zero(features: usize) -> Self {
        Self {
            n: 0,
            mean: vec![0.0; features],
            square: vec![0.0; features],
        }
    }

    /// The moments of one design matrix.
    pub fn of(x: &Matrix) -> Self {
        let p = x.p();
        let mut mean = vec![0.0f64; p];
        for i in 0..x.n() {
            for (acc, v) in mean.iter_mut().zip(x.row(i)) {
                *acc += v;
            }
        }
        let n = x.n() as f64;
        for m in &mut mean {
            *m /= n;
        }
        let mut square = vec![0.0f64; p];
        for i in 0..x.n() {
            for (j, v) in x.row(i).iter().enumerate() {
                let d = v - mean[j];
                square[j] += d * d;
            }
        }
        Self {
            n: x.n() as u64,
            mean,
            square,
        }
    }

    /// The moments of one design matrix, computed over every core.
    ///
    /// This is [`Moments::of`] over row blocks reduced with [`Moments::merge`] — the same
    /// mechanism an out-of-core fit uses, pointed at threads instead of at storage. Not
    /// bit-identical to the serial answer, because the merge order differs; equal to
    /// within the tolerance the oracle checks.
    pub fn par(x: &Matrix) -> Self {
        if x.n() * x.p() < SERIAL {
            return Self::of(x);
        }
        let p = x.p();
        let blocks = rayon::current_num_threads().max(1);
        let each = x.n().div_ceil(blocks);
        (0..blocks)
            .into_par_iter()
            .map(|b| {
                let start = b * each;
                let stop = ((b + 1) * each).min(x.n());
                if start >= stop {
                    return Self::zero(p);
                }
                Self::rows(x, start, stop)
            })
            .reduce(|| Self::zero(p), |a, b| a.merge(&b))
    }

    /// The moments of rows `start..stop`, the unit a parallel or chunked fit works in.
    pub fn rows(x: &Matrix, start: usize, stop: usize) -> Self {
        let p = x.p();
        let mut mean = vec![0.0f64; p];
        for i in start..stop {
            for (acc, v) in mean.iter_mut().zip(x.row(i)) {
                *acc += v;
            }
        }
        let n = (stop - start) as f64;
        for m in &mut mean {
            *m /= n;
        }
        let mut square = vec![0.0f64; p];
        for i in start..stop {
            for (j, v) in x.row(i).iter().enumerate() {
                let d = v - mean[j];
                square[j] += d * d;
            }
        }
        Self {
            n: (stop - start) as u64,
            mean,
            square,
        }
    }

    /// The monoid operation: the moments of the concatenation of two chunks.
    ///
    /// Panics only on a programming error — moments of different widths cannot describe
    /// one design, and no data can cause it.
    pub fn merge(mut self, other: &Self) -> Self {
        assert_eq!(
            self.mean.len(),
            other.mean.len(),
            "moments of different widths cannot describe one design"
        );
        if other.n == 0 {
            return self;
        }
        if self.n == 0 {
            return other.clone();
        }
        let (a, b) = (self.n as f64, other.n as f64);
        let total = a + b;
        for j in 0..self.mean.len() {
            let delta = other.mean[j] - self.mean[j];
            self.mean[j] += delta * b / total;
            self.square[j] += other.square[j] + delta * delta * a * b / total;
        }
        self.n += other.n;
        self
    }

    /// How many features these moments describe.
    pub fn features(&self) -> usize {
        self.mean.len()
    }

    /// How many observations went in.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// The per-feature mean.
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// The per-feature variance with no degrees-of-freedom correction, which is the
    /// convention a scaler divides by.
    pub fn variance(&self) -> Vec<f64> {
        let n = self.n as f64;
        self.square
            .iter()
            .map(|s| if self.n == 0 { 0.0 } else { s / n })
            .collect()
    }

    /// The per-feature variance corrected by `lost` degrees of freedom — `1` for the
    /// sample variance an explained-variance figure is quoted in.
    pub fn variance_of(&self, lost: u64) -> Vec<f64> {
        self.square
            .iter()
            .map(|s| {
                if self.n <= lost {
                    0.0
                } else {
                    s / (self.n - lost) as f64
                }
            })
            .collect()
    }
}

/// The smallest and largest value per feature.
#[derive(Debug, Clone, PartialEq)]
pub struct Extent {
    n: u64,
    low: Vec<f64>,
    high: Vec<f64>,
}

impl Extent {
    /// The identity: no observations.
    pub fn zero(features: usize) -> Self {
        Self {
            n: 0,
            low: vec![f64::INFINITY; features],
            high: vec![f64::NEG_INFINITY; features],
        }
    }

    /// The extent of one design matrix.
    pub fn of(x: &Matrix) -> Self {
        Self::rows(x, 0, x.n())
    }

    /// The extent of rows `start..stop`.
    pub fn rows(x: &Matrix, start: usize, stop: usize) -> Self {
        let mut e = Self::zero(x.p());
        for i in start..stop {
            for (j, &v) in x.row(i).iter().enumerate() {
                if v < e.low[j] {
                    e.low[j] = v;
                }
                if v > e.high[j] {
                    e.high[j] = v;
                }
            }
        }
        e.n = (stop - start) as u64;
        e
    }

    /// The extent of one design matrix, computed over every core.
    pub fn par(x: &Matrix) -> Self {
        if x.n() * x.p() < SERIAL {
            return Self::of(x);
        }
        let p = x.p();
        let blocks = rayon::current_num_threads().max(1);
        let each = x.n().div_ceil(blocks);
        (0..blocks)
            .into_par_iter()
            .map(|b| {
                let start = b * each;
                let stop = ((b + 1) * each).min(x.n());
                if start >= stop {
                    return Self::zero(p);
                }
                Self::rows(x, start, stop)
            })
            .reduce(|| Self::zero(p), |a, b| a.merge(&b))
    }

    /// The monoid operation.
    pub fn merge(mut self, other: &Self) -> Self {
        assert_eq!(
            self.low.len(),
            other.low.len(),
            "extents of different widths cannot describe one design"
        );
        for j in 0..self.low.len() {
            if other.low[j] < self.low[j] {
                self.low[j] = other.low[j];
            }
            if other.high[j] > self.high[j] {
                self.high[j] = other.high[j];
            }
        }
        self.n += other.n;
        self
    }

    /// How many features this extent describes.
    pub fn features(&self) -> usize {
        self.low.len()
    }

    /// How many observations went in.
    pub fn count(&self) -> u64 {
        self.n
    }

    /// The smallest value per feature.
    pub fn low(&self) -> &[f64] {
        &self.low
    }

    /// The largest value per feature.
    pub fn high(&self) -> &[f64] {
        &self.high
    }
}

/// A scale of zero cannot divide, and a feature with no spread carries no information, so
/// a zero scale becomes one and the scaled feature becomes zero rather than infinite.
///
/// Every scaler goes through this one function, so there is one rule and one place it
/// lives. The `10·eps` floor is scikit-learn's: a spread that small is rounding noise.
pub(crate) fn usable(scale: f64) -> f64 {
    if scale.abs() < 10.0 * f64::EPSILON {
        1.0
    } else {
        scale
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Large offsets with small variation: the case where a naive one-pass variance
    /// loses every significant digit, so associativity here is a real claim.
    fn awkward() -> Matrix {
        let (n, p) = (97usize, 5usize);
        let mut data = Vec::with_capacity(n * p);
        for i in 0..n {
            for j in 0..p {
                data.push(1.0e8 + ((i * 7 + j * 13) % 11) as f64);
            }
        }
        Matrix::new(n, p, data).unwrap()
    }

    #[test]
    fn moments_are_associative_under_any_cut() {
        let x = awkward();
        let whole = Moments::of(&x);
        for parts in [2usize, 3, 7, 16, 97] {
            let each = x.n().div_ceil(parts);
            let cuts: Vec<(usize, usize)> = (0..parts)
                .map(|b| (b * each, ((b + 1) * each).min(x.n())))
                .filter(|(a, b)| a < b)
                .collect();
            let left = cuts.iter().fold(Moments::zero(x.p()), |a, &(s, e)| {
                a.merge(&Moments::rows(&x, s, e))
            });
            let right = cuts.iter().rev().fold(Moments::zero(x.p()), |a, &(s, e)| {
                Moments::rows(&x, s, e).merge(&a)
            });
            assert_eq!(whole.count(), left.count());
            for j in 0..x.p() {
                assert!(
                    (whole.mean()[j] - left.mean()[j]).abs() < 1e-6,
                    "mean {j} parts {parts}"
                );
                assert!(
                    (whole.variance()[j] - left.variance()[j]).abs() < 1e-6,
                    "variance {j} parts {parts}: {} vs {}",
                    whole.variance()[j],
                    left.variance()[j]
                );
                assert!((right.mean()[j] - left.mean()[j]).abs() < 1e-6);
                assert!((right.variance()[j] - left.variance()[j]).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn the_identity_merges_to_nothing() {
        let x = awkward();
        let m = Moments::of(&x);
        assert_eq!(m.clone().merge(&Moments::zero(x.p())), m);
        assert_eq!(Moments::zero(x.p()).merge(&m), m);
        let e = Extent::of(&x);
        assert_eq!(e.clone().merge(&Extent::zero(x.p())), e);
    }

    #[test]
    fn parallel_agrees_with_serial() {
        let (n, p) = (40_000usize, 4usize);
        let data: Vec<f64> = (0..n * p)
            .map(|i| (i as f64 * 0.017).sin() * 30.0)
            .collect();
        let x = Matrix::new(n, p, data).unwrap();
        let a = Moments::of(&x);
        let b = Moments::par(&x);
        assert_eq!(a.count(), b.count());
        for j in 0..p {
            assert!((a.mean()[j] - b.mean()[j]).abs() < 1e-9);
            assert!((a.variance()[j] - b.variance()[j]).abs() < 1e-9);
        }
        let e = Extent::of(&x);
        let f = Extent::par(&x);
        assert_eq!(e.low(), f.low());
        assert_eq!(e.high(), f.high());
        assert_eq!(e.count(), f.count());
    }
}
