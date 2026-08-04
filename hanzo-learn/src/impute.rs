//! Completing data that has holes in it.
//!
//! [`Matrix`] refuses a value that is not finite, so a design with holes is not a
//! `Matrix` and cannot reach a fit. That is not an obstacle to work around — it is the
//! statement this module completes. [`Partial`] is the type of data that has holes;
//! [`Fill`] is the only arrow from `Partial` to `Matrix`. So "fitted a model on data with
//! missing values and got parameters that are all `NaN`" is not a bug that can be
//! written here, and the imputation a caller performed is a value they still hold rather
//! than a step that happened somewhere upstream.
//!
//! ```text
//!   Partial  --Fill::apply-->  Matrix  --Fit::fit-->  Model
//!   holes                      complete               fitted
//! ```
//!
//! [`Fill`] therefore does NOT implement [`crate::Transform`]: that trait is
//! `Matrix -> Matrix`, and this arrow starts somewhere else.
//!
//! # Scale
//!
//! Filling with a mean streams — [`Tally`] is a monoid, so it folds over chunks and
//! reduces over threads exactly as [`crate::moment::Moments`] does, with `O(p)` state and
//! unbounded rows. A median or a mode does NOT: neither is computable in bounded space
//! from a stream, so [`Fill::fit`] needs the columns resident and says so here rather
//! than in a comment. The types keep the two honest: [`Fill::of`] takes a `Tally` and can
//! only ever produce a mean-filled imputer, so a streamed median is not a thing a caller
//! can ask for and be quietly given something else.

use crate::{Error, Matrix, Result};

/// A design matrix whose cells are each a finite number or nothing.
///
/// Stored as `f64` with a `NaN` standing for absence, which is how such data arrives and
/// is half the memory of an `Option<f64>`; the representation is private and the value is
/// [`Partial::at`] returning an `Option`.
#[derive(Debug, Clone, PartialEq)]
pub struct Partial {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Partial {
    /// From cells that each may be absent.
    pub fn new(rows: usize, cols: usize, data: Vec<Option<f64>>) -> Result<Self> {
        Self::nan(
            rows,
            cols,
            data.into_iter().map(|v| v.unwrap_or(f64::NAN)).collect(),
        )
    }

    /// From values in which `NaN` marks absence — the convention the data arrives in.
    ///
    /// Fails on a shape that does not multiply out, on an empty extent, and on an
    /// infinity: an infinity is neither an observation a mean can use nor a hole an
    /// imputer can fill, so it is refused at the boundary with its position.
    pub fn nan(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::Shape(format!(
                "a partial design is {rows}x{cols}; both dimensions must be non-zero"
            )));
        }
        if data.len() != rows * cols {
            return Err(Error::Shape(format!(
                "{rows}x{cols} needs {} values, got {}",
                rows * cols,
                data.len()
            )));
        }
        if let Some(i) = data.iter().position(|v| v.is_infinite()) {
            return Err(Error::Shape(format!(
                "value at row {} column {} is infinite, which is neither an observation \
                 nor a hole",
                i / cols,
                i % cols
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// A complete matrix seen as a partial one with no holes.
    pub fn complete(x: &Matrix) -> Self {
        let mut data = Vec::with_capacity(x.n() * x.p());
        for i in 0..x.n() {
            data.extend_from_slice(x.row(i));
        }
        Self {
            rows: x.n(),
            cols: x.p(),
            data,
        }
    }

    /// Number of observations.
    pub fn n(&self) -> usize {
        self.rows
    }

    /// Number of features.
    pub fn p(&self) -> usize {
        self.cols
    }

    /// One cell: a finite number, or nothing.
    pub fn at(&self, i: usize, j: usize) -> Option<f64> {
        let v = self.data[i * self.cols + j];
        if v.is_nan() {
            None
        } else {
            Some(v)
        }
    }

    /// How many cells of `column` were observed.
    pub fn observed(&self, column: usize) -> usize {
        (0..self.rows)
            .filter(|&i| !self.data[i * self.cols + column].is_nan())
            .count()
    }

    /// Every observed value of one feature, in row order.
    pub fn column(&self, j: usize) -> Vec<f64> {
        (0..self.rows).filter_map(|i| self.at(i, j)).collect()
    }
}

/// Count and sum of the observed values, per feature: the sufficient statistic for
/// filling with a mean, and a monoid.
#[derive(Debug, Clone, PartialEq)]
pub struct Tally {
    n: Vec<u64>,
    sum: Vec<f64>,
}

impl Tally {
    /// The identity: nothing observed.
    pub fn zero(features: usize) -> Self {
        Self {
            n: vec![0; features],
            sum: vec![0.0; features],
        }
    }

    /// The tally of one partial design.
    pub fn of(x: &Partial) -> Self {
        let mut t = Self::zero(x.p());
        for i in 0..x.n() {
            for j in 0..x.p() {
                if let Some(v) = x.at(i, j) {
                    t.n[j] += 1;
                    t.sum[j] += v;
                }
            }
        }
        t
    }

    /// The monoid operation.
    pub fn merge(mut self, other: &Self) -> Self {
        assert_eq!(
            self.n.len(),
            other.n.len(),
            "tallies of different widths cannot describe one design"
        );
        for j in 0..self.n.len() {
            self.n[j] += other.n[j];
            self.sum[j] += other.sum[j];
        }
        self
    }

    /// How many features this tally describes.
    pub fn features(&self) -> usize {
        self.n.len()
    }

    /// How many values of `column` were observed.
    pub fn count(&self, column: usize) -> u64 {
        self.n[column]
    }
}

/// Which value a hole is filled with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Statistic {
    /// The mean of the observed values. Streams.
    Mean,
    /// The median of the observed values, averaging the two middles at even count, which
    /// is what `numpy` does. Needs the column resident.
    Median,
    /// The most frequent observed value, smallest wins a tie. Needs the column resident.
    Mode,
    /// A value the caller chose, which needs no data at all.
    Constant(f64),
}

impl Statistic {
    fn name(&self) -> &'static str {
        match self {
            Self::Mean => "mean",
            Self::Median => "median",
            Self::Mode => "mode",
            Self::Constant(_) => "constant",
        }
    }
}

/// A per-feature fill value: what a hole becomes.
#[derive(Debug, Clone, PartialEq)]
pub struct Fill {
    fill: Vec<f64>,
    statistic: Statistic,
}

impl Fill {
    /// Fit on one partial design.
    ///
    /// Fails when a feature has no observed value at all and the statistic is one that
    /// needs data: there is no mean of nothing. scikit-learn DROPS such a column from its
    /// output, which silently changes the width of everything downstream; this names the
    /// column instead. Use [`Statistic::Constant`] to fill a feature that was never
    /// observed.
    pub fn fit(x: &Partial, statistic: Statistic) -> Result<Self> {
        if let Statistic::Constant(v) = statistic {
            if !v.is_finite() {
                return Err(Error::Config(format!(
                    "a constant fill must be finite, not {v}"
                )));
            }
            return Ok(Self {
                fill: vec![v; x.p()],
                statistic,
            });
        }
        let mut fill = Vec::with_capacity(x.p());
        for j in 0..x.p() {
            let mut observed = x.column(j);
            if observed.is_empty() {
                return Err(Error::Shape(format!(
                    "feature {j} has no observed value, so it has no {}",
                    statistic.name()
                )));
            }
            fill.push(match statistic {
                Statistic::Mean => observed.iter().sum::<f64>() / observed.len() as f64,
                Statistic::Median => median(&mut observed),
                Statistic::Mode => mode(&mut observed),
                Statistic::Constant(_) => unreachable!("handled above"),
            });
        }
        Ok(Self { fill, statistic })
    }

    /// Fit from an already-accumulated statistic: the streaming and multi-device path.
    ///
    /// Mean only, by construction — a `Tally` carries a count and a sum, so this function
    /// could not return a median-filled imputer even if a caller wanted one.
    pub fn of(t: &Tally) -> Result<Self> {
        let mut fill = Vec::with_capacity(t.features());
        for j in 0..t.features() {
            if t.n[j] == 0 {
                return Err(Error::Shape(format!(
                    "feature {j} has no observed value, so it has no mean"
                )));
            }
            fill.push(t.sum[j] / t.n[j] as f64);
        }
        Ok(Self {
            fill,
            statistic: Statistic::Mean,
        })
    }

    /// What each feature's holes are filled with.
    pub fn value(&self) -> &[f64] {
        &self.fill
    }

    /// Which statistic produced those values.
    pub fn statistic(&self) -> Statistic {
        self.statistic
    }

    /// How many features this was fitted on.
    pub fn features(&self) -> usize {
        self.fill.len()
    }

    /// Complete a partial design: the one arrow from [`Partial`] to [`Matrix`].
    pub fn apply(&self, x: &Partial) -> Result<Matrix> {
        if x.p() != self.fill.len() {
            return Err(Error::Shape(format!(
                "fitted on {} features, given {}",
                self.fill.len(),
                x.p()
            )));
        }
        let mut out = Vec::with_capacity(x.n() * x.p());
        for i in 0..x.n() {
            for j in 0..x.p() {
                out.push(x.at(i, j).unwrap_or(self.fill[j]));
            }
        }
        Matrix::new(x.n(), x.p(), out)
    }
}

/// The middle of a sample, averaging the two middles at even count.
fn median(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        // Halve each side rather than summing then halving: the sum of two large values
        // can leave the representable range where their average cannot.
        v[n / 2 - 1] / 2.0 + v[n / 2] / 2.0
    }
}

/// The most frequent value, smallest winning a tie.
fn mode(v: &mut [f64]) -> f64 {
    v.sort_by(f64::total_cmp);
    let (mut best, mut best_run) = (v[0], 0usize);
    let mut i = 0;
    while i < v.len() {
        let mut j = i;
        while j < v.len() && v[j] == v[i] {
            j += 1;
        }
        if j - i > best_run {
            best = v[i];
            best_run = j - i;
        }
        i = j;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    fn holed() -> Partial {
        Partial::new(
            4,
            3,
            vec![
                Some(1.0),
                None,
                Some(7.0),
                Some(3.0),
                Some(2.0),
                None,
                None,
                Some(2.0),
                Some(9.0),
                Some(4.0),
                Some(6.0),
                Some(11.0),
            ],
        )
        .unwrap()
    }

    #[test]
    fn a_partial_design_reports_its_holes_as_absence_and_not_as_a_number() {
        let x = holed();
        assert_eq!(x.at(0, 0), Some(1.0));
        assert_eq!(x.at(0, 1), None);
        assert_eq!(x.observed(0), 3);
        assert_eq!(x.observed(2), 3);
        assert_eq!(x.column(1), vec![2.0, 2.0, 6.0]);
    }

    #[test]
    fn an_infinity_is_neither_an_observation_nor_a_hole() {
        let e = Partial::nan(1, 2, vec![1.0, f64::INFINITY]).unwrap_err();
        assert!(format!("{e}").contains("row 0 column 1"), "{e}");
    }

    #[test]
    fn each_statistic_fills_with_what_it_says() {
        let x = holed();
        // column 0 observed: 1, 3, 4 -> mean 8/3, median 3, mode 1 (all tie at 1, smallest)
        assert!((Fill::fit(&x, Statistic::Mean).unwrap().value()[0] - 8.0 / 3.0).abs() < 1e-12);
        assert_eq!(Fill::fit(&x, Statistic::Median).unwrap().value()[0], 3.0);
        assert_eq!(Fill::fit(&x, Statistic::Mode).unwrap().value()[0], 1.0);
        // column 1 observed: 2, 2, 6 -> mode is the repeated 2, median 2
        assert_eq!(Fill::fit(&x, Statistic::Mode).unwrap().value()[1], 2.0);
        assert_eq!(Fill::fit(&x, Statistic::Median).unwrap().value()[1], 2.0);
        assert_eq!(
            Fill::fit(&x, Statistic::Constant(-1.0)).unwrap().value(),
            &[-1.0, -1.0, -1.0]
        );
    }

    #[test]
    fn filling_produces_a_matrix_a_fit_will_accept() {
        let x = holed();
        let f = Fill::fit(&x, Statistic::Median).unwrap();
        let m = f.apply(&x).unwrap();
        assert_eq!((m.n(), m.p()), (4, 3));
        assert_eq!(m.at(0, 1), 2.0);
        assert_eq!(m.at(1, 2), 9.0);
        assert_eq!(m.at(2, 0), 3.0);
        // Untouched cells are exactly what they were.
        assert_eq!(m.at(3, 2), 11.0);
    }

    #[test]
    fn a_feature_of_nothing_is_named_and_not_dropped() {
        let x = Partial::new(2, 2, vec![Some(1.0), None, Some(2.0), None]).unwrap();
        let e = Fill::fit(&x, Statistic::Mean).unwrap_err();
        assert!(format!("{e}").contains("feature 1"), "{e}");
        // A constant needs no data, so it is the stated way through.
        assert!(Fill::fit(&x, Statistic::Constant(0.0)).is_ok());
    }

    #[test]
    fn a_streamed_mean_equals_a_resident_mean_and_a_tally_is_a_monoid() {
        let x = holed();
        let whole = Tally::of(&x);
        let a = Partial::new(2, 3, (0..6).map(|k| x.at(k / 3, k % 3)).collect()).unwrap();
        let b = Partial::new(2, 3, (6..12).map(|k| x.at(k / 3, k % 3)).collect()).unwrap();
        let streamed = Tally::zero(3).merge(&Tally::of(&a)).merge(&Tally::of(&b));
        assert_eq!(whole, streamed);
        let resident = Fill::fit(&x, Statistic::Mean).unwrap();
        let from_statistic = Fill::of(&streamed).unwrap();
        for j in 0..3 {
            assert!((resident.value()[j] - from_statistic.value()[j]).abs() < 1e-12);
        }
        assert_eq!(from_statistic.statistic(), Statistic::Mean);
    }
}
