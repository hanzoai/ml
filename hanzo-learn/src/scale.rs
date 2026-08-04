//! Putting features on comparable footing.
//!
//! Both scalers fit from a [`crate::moment`] statistic, so both stream: rows unbounded,
//! state `O(p)`, and a fit spans threads, chunks or devices by combining statistics rather
//! than by moving data. Neither ever needs the design resident.
//!
//! # Scale, concretely
//!
//! `fit` takes `&Matrix`, so the data it is handed is resident and owned by the caller —
//! `8·n·p` bytes, which on this 128 GB box is about `1.6·10⁸` rows of 100 `f64` features
//! with room to work in. Past that, do not hand it one matrix: fold [`crate::moment`]
//! statistics over chunks and call [`Standard::of`] on the result. The fitted value is
//! `O(p)` either way — three `Vec<f64>` of length `p`, whatever `n` was.
//!
//! Single device. There is no GPU path, and the measured reason is in `oracle/README.md`:
//! a scaler is one pass of one multiply-add per value, so it runs at memory bandwidth and
//! a transfer to a device would cost more than the arithmetic it saves.

use crate::moment::{usable, Extent, Moments};
use crate::{Error, Matrix, Result, Transform};

/// Centre each feature on its mean and divide by its standard deviation.
///
/// The fitted value is three vectors of length `p` and nothing else, whatever `n` was.
/// [`Standard::fit`] and [`Standard::of`] are the only ways to obtain one, so an unfitted
/// scaler is not a value that can be spoken.
///
/// # What is deliberately absent
///
/// scikit-learn's `with_mean=False` exists so that centring cannot densify a sparse
/// matrix. [`Matrix`] is dense, so the flag would be a mode with nothing to protect. A
/// `Standard` always centres and always scales.
#[derive(Debug, Clone, PartialEq)]
pub struct Standard {
    centre: Vec<f64>,
    scale: Vec<f64>,
    variance: Vec<f64>,
}

impl Standard {
    /// Fit on one design matrix.
    pub fn fit(x: &Matrix) -> Self {
        Self::of(&Moments::par(x))
    }

    /// Fit from an already-accumulated statistic: the streaming, parallel and
    /// multi-device entry point.
    ///
    /// Total — every `Moments` implies a scaler. A feature with no spread gets a divisor
    /// of one rather than an error, because a constant column is information a caller
    /// usually wants passed through as zero rather than refused.
    pub fn of(m: &Moments) -> Self {
        let variance = m.variance();
        Self {
            centre: m.mean().to_vec(),
            scale: variance.iter().map(|v| usable(v.sqrt())).collect(),
            variance,
        }
    }

    /// The per-feature mean that gets subtracted.
    pub fn centre(&self) -> &[f64] {
        &self.centre
    }

    /// The per-feature divisor: the standard deviation, or one for a feature with no
    /// spread.
    pub fn scale(&self) -> &[f64] {
        &self.scale
    }

    /// The per-feature variance as fitted, so a feature with no spread reads zero here
    /// even though its divisor reads one.
    pub fn variance(&self) -> &[f64] {
        &self.variance
    }

    /// Undo the transform.
    pub fn invert(&self, x: &Matrix) -> Result<Matrix> {
        width(self.centre.len(), x)?;
        let mut out = Vec::with_capacity(x.n() * x.p());
        for i in 0..x.n() {
            for (j, v) in x.row(i).iter().enumerate() {
                out.push(v * self.scale[j] + self.centre[j]);
            }
        }
        Matrix::new(x.n(), x.p(), out)
    }
}

impl Transform for Standard {
    fn features(&self) -> usize {
        self.centre.len()
    }

    fn width(&self) -> usize {
        self.centre.len()
    }

    fn apply(&self, x: &Matrix) -> Result<Matrix> {
        width(self.centre.len(), x)?;
        let mut out = Vec::with_capacity(x.n() * x.p());
        for i in 0..x.n() {
            for (j, v) in x.row(i).iter().enumerate() {
                out.push((v - self.centre[j]) / self.scale[j]);
            }
        }
        Matrix::new(x.n(), x.p(), out)
    }
}

/// The interval a [`Range`] maps into.
///
/// Constructed only from a low bound strictly below a finite high one, so an inverted or
/// degenerate target is not a value that exists and no scaler has to check for one.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    low: f64,
    high: f64,
}

impl Span {
    /// Zero to one, which is what almost every caller wants.
    pub const UNIT: Span = Span {
        low: 0.0,
        high: 1.0,
    };

    /// An interval from `low` up to `high`.
    pub fn new(low: f64, high: f64) -> Result<Self> {
        if !low.is_finite() || !high.is_finite() || low >= high {
            return Err(Error::Config(format!(
                "a span runs from a finite low bound up to a strictly greater finite high \
                 bound, not {low} to {high}"
            )));
        }
        Ok(Self { low, high })
    }

    /// The bottom of the interval.
    pub fn low(&self) -> f64 {
        self.low
    }

    /// The top of the interval.
    pub fn high(&self) -> f64 {
        self.high
    }
}

/// Map each feature's observed range onto a fixed interval.
///
/// Values outside the range seen while fitting land outside the target interval: the
/// transform is affine and is not clipped, which is scikit-learn's behaviour. Clipping
/// would hide the distribution shift a caller usually wants to see.
#[derive(Debug, Clone, PartialEq)]
pub struct Range {
    low: Vec<f64>,
    high: Vec<f64>,
    scale: Vec<f64>,
    shift: Vec<f64>,
    span: Span,
}

impl Range {
    /// Fit on one design matrix, mapping into `span`.
    pub fn fit(x: &Matrix, span: Span) -> Self {
        Self::of(&Extent::par(x), span)
    }

    /// Fit from an already-accumulated statistic: the streaming and multi-device path.
    pub fn of(e: &Extent, span: Span) -> Self {
        let reach = span.high - span.low;
        let scale: Vec<f64> = (0..e.features())
            .map(|j| reach / usable(e.high()[j] - e.low()[j]))
            .collect();
        let shift = (0..e.features())
            .map(|j| span.low - e.low()[j] * scale[j])
            .collect();
        Self {
            low: e.low().to_vec(),
            high: e.high().to_vec(),
            scale,
            shift,
            span,
        }
    }

    /// The smallest value seen per feature.
    pub fn low(&self) -> &[f64] {
        &self.low
    }

    /// The largest value seen per feature.
    pub fn high(&self) -> &[f64] {
        &self.high
    }

    /// The interval fitted into.
    pub fn span(&self) -> Span {
        self.span
    }

    /// Undo the transform.
    pub fn invert(&self, x: &Matrix) -> Result<Matrix> {
        width(self.low.len(), x)?;
        let mut out = Vec::with_capacity(x.n() * x.p());
        for i in 0..x.n() {
            for (j, v) in x.row(i).iter().enumerate() {
                out.push((v - self.shift[j]) / self.scale[j]);
            }
        }
        Matrix::new(x.n(), x.p(), out)
    }
}

impl Transform for Range {
    fn features(&self) -> usize {
        self.low.len()
    }

    fn width(&self) -> usize {
        self.low.len()
    }

    fn apply(&self, x: &Matrix) -> Result<Matrix> {
        width(self.low.len(), x)?;
        let mut out = Vec::with_capacity(x.n() * x.p());
        for i in 0..x.n() {
            for (j, v) in x.row(i).iter().enumerate() {
                out.push(v * self.scale[j] + self.shift[j]);
            }
        }
        Matrix::new(x.n(), x.p(), out)
    }
}

/// The one width check every fitted transform runs, stated once.
pub(crate) fn width(fitted: usize, x: &Matrix) -> Result<()> {
    if x.p() != fitted {
        return Err(Error::Shape(format!(
            "fitted on {fitted} features, given {}",
            x.p()
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_constant_feature_scales_to_zero_and_not_to_infinity() {
        let x = Matrix::new(3, 2, vec![5.0, 1.0, 5.0, 2.0, 5.0, 3.0]).unwrap();
        let s = Standard::fit(&x);
        assert_eq!(s.scale()[0], 1.0);
        assert_eq!(s.variance()[0], 0.0);
        // Matrix::new refuses non-finite values, so a finite result is what makes this
        // call succeed at all.
        let out = s.apply(&x).unwrap();
        assert_eq!(out.at(0, 0), 0.0);
        assert!(Range::fit(&x, Span::UNIT).apply(&x).is_ok());
    }

    #[test]
    fn a_width_mismatch_is_named_and_not_guessed() {
        let x = Matrix::new(2, 2, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let narrow = Matrix::new(4, 1, vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let e = Standard::fit(&x).apply(&narrow).unwrap_err();
        assert!(
            format!("{e}").contains("fitted on 2 features, given 1"),
            "{e}"
        );
    }

    #[test]
    fn inverting_returns_what_went_in() {
        let data: Vec<f64> = (0..60).map(|i| (i as f64 * 0.37).sin() * 40.0).collect();
        let x = Matrix::new(20, 3, data).unwrap();
        let s = Standard::fit(&x);
        let back = s.invert(&s.apply(&x).unwrap()).unwrap();
        for i in 0..x.n() {
            for j in 0..x.p() {
                assert!((back.at(i, j) - x.at(i, j)).abs() < 1e-9);
            }
        }
        let r = Range::fit(&x, Span::new(-3.0, 7.0).unwrap());
        let back = r.invert(&r.apply(&x).unwrap()).unwrap();
        for i in 0..x.n() {
            for j in 0..x.p() {
                assert!((back.at(i, j) - x.at(i, j)).abs() < 1e-9);
            }
        }
    }

    #[test]
    fn an_impossible_span_is_not_a_value() {
        assert!(Span::new(1.0, 1.0).is_err());
        assert!(Span::new(2.0, 1.0).is_err());
        assert!(Span::new(f64::NAN, 1.0).is_err());
        assert!(Span::new(0.0, f64::INFINITY).is_err());
        assert!(Span::new(-3.0, 7.0).is_ok());
    }

    #[test]
    fn a_chunked_fit_equals_a_whole_fit() {
        let data: Vec<f64> = (0..1200)
            .map(|i| (i as f64 * 0.11).cos() * 9.0 + 3.0)
            .collect();
        let x = Matrix::new(300, 4, data).unwrap();
        let whole = Standard::fit(&x);
        let cuts = [(0usize, 43usize), (43, 100), (100, 271), (271, 300)];
        let streamed = Standard::of(&cuts.iter().fold(Moments::zero(4), |a, &(s, e)| {
            a.merge(&Moments::rows(&x, s, e))
        }));
        for j in 0..4 {
            assert!((whole.centre()[j] - streamed.centre()[j]).abs() < 1e-9);
            assert!((whole.scale()[j] - streamed.scale()[j]).abs() < 1e-9);
        }
    }
}
