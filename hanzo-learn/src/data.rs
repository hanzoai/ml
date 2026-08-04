//! The data plane: a design matrix, and observations paired with their targets.

use crate::error::{Error, Result};

/// A design matrix: `n` observations of `p` features, dense, row-major.
///
/// RANK 2 IS IN THE TYPE. A general tensor reaching a least-squares solver carries its
/// rank as a runtime fact, so every solver opens by asking whether it was handed a
/// matrix at all. Here that question has one answer, established once, at
/// construction.
///
/// Row-major because every consumer walks observations: the tree splitter gathers one
/// feature at a time but does so through an index permutation, and the linear solvers
/// walk rows. A column-major variant would help the splitter and hurt everything else;
/// one layout, chosen for the common walk, is simpler than two that must agree.
#[derive(Clone, PartialEq, Debug)]
pub struct Matrix {
    rows: usize,
    cols: usize,
    data: Vec<f64>,
}

impl Matrix {
    /// A matrix from row-major values.
    ///
    /// Fails unless `data.len() == rows * cols`, and unless both are non-zero — a
    /// matrix with no columns has no features to fit against and one with no rows has
    /// nothing to fit from, so both are refused here rather than producing a model
    /// that answers everything with a constant.
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::Shape(format!(
                "a design matrix is {rows}x{cols}; both dimensions must be non-zero"
            )));
        }
        if data.len() != rows * cols {
            return Err(Error::Shape(format!(
                "{rows}x{cols} needs {} values, got {}",
                rows * cols,
                data.len()
            )));
        }
        if let Some(i) = data.iter().position(|v| !v.is_finite()) {
            // Refused at the boundary, with the position, because the alternative is a
            // model whose every parameter is NaN and whose predictions are all NaN,
            // discovered downstream with nothing left pointing at the cause.
            return Err(Error::Shape(format!(
                "value at row {} column {} is not finite",
                i / cols,
                i % cols
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// From rows, which is how callers usually hold their data.
    pub fn rows(rows: &[Vec<f64>]) -> Result<Self> {
        let n = rows.len();
        let p = rows.first().map_or(0, Vec::len);
        if let Some(i) = rows.iter().position(|r| r.len() != p) {
            return Err(Error::Shape(format!(
                "row 0 has {p} features but row {i} has {}",
                rows[i].len()
            )));
        }
        Self::new(n, p, rows.concat())
    }

    /// Number of observations.
    pub fn n(&self) -> usize {
        self.rows
    }

    /// Number of features.
    pub fn p(&self) -> usize {
        self.cols
    }

    /// One observation.
    pub fn row(&self, i: usize) -> &[f64] {
        &self.data[i * self.cols..(i + 1) * self.cols]
    }

    /// One cell. Hot in the tree splitter, which reaches for a single feature across a
    /// permutation of rows rather than for whole rows.
    pub fn at(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    /// Column means — the term both linear fits centre on.
    pub(crate) fn column_means(&self) -> Vec<f64> {
        let mut m = vec![0.0; self.cols];
        for i in 0..self.rows {
            for (acc, v) in m.iter_mut().zip(self.row(i)) {
                *acc += v;
            }
        }
        let n = self.rows as f64;
        for v in &mut m {
            *v /= n;
        }
        m
    }
}

/// Observations paired with their targets.
///
/// THE ONE PLACE their agreement in `n` is established. Without this type every fit
/// would open by comparing two lengths, and every fit would be a partial function of
/// two arguments that must agree. With it, `fit` is total in `n`: a `Samples` that
/// exists is a dataset whose rows have targets, so no solver below re-checks it and
/// none can forget to.
///
/// It is the closest this crate gets to a dependent type. The agreement is a property
/// of a PAIR, so it is checked where the pair is made.
#[derive(Clone, PartialEq, Debug)]
pub struct Samples<T> {
    x: Matrix,
    y: Vec<T>,
}

impl<T> Samples<T> {
    /// Pair a design matrix with its targets.
    pub fn new(x: Matrix, y: Vec<T>) -> Result<Self> {
        if x.n() != y.len() {
            return Err(Error::Shape(format!(
                "{} observations but {} targets",
                x.n(),
                y.len()
            )));
        }
        Ok(Self { x, y })
    }

    /// The observations.
    pub fn x(&self) -> &Matrix {
        &self.x
    }

    /// The targets.
    pub fn y(&self) -> &[T] {
        &self.y
    }

    /// Number of observations. Equal to `y().len()` by construction.
    pub fn n(&self) -> usize {
        self.x.n()
    }

    /// Number of features.
    pub fn p(&self) -> usize {
        self.x.p()
    }
}

/// The mean of a slice, accumulated in index order.
///
/// Summation order is part of the answer at this precision, so it is stated here once
/// and every caller inherits the same one: two fits of the same data must agree
/// bit-for-bit with each other, which a parallel reduction over an unspecified tree
/// would not guarantee.
pub(crate) fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_matrix_refuses_a_length_that_is_not_its_shape() {
        assert!(Matrix::new(2, 3, vec![1.0; 5]).is_err());
        assert!(Matrix::new(0, 3, vec![]).is_err());
        assert!(Matrix::new(2, 3, vec![1.0; 6]).is_ok());
    }

    #[test]
    fn a_matrix_refuses_a_value_that_is_not_finite_and_says_where() {
        let e = Matrix::new(2, 2, vec![1.0, 2.0, f64::NAN, 4.0]).unwrap_err();
        assert!(format!("{e}").contains("row 1 column 0"), "{e}");
    }

    #[test]
    fn ragged_rows_are_refused_with_the_offending_row() {
        let e = Matrix::rows(&[vec![1.0, 2.0], vec![3.0]]).unwrap_err();
        assert!(format!("{e}").contains("row 1"), "{e}");
    }

    #[test]
    fn samples_are_the_one_place_n_agreement_is_checked() {
        let x = Matrix::new(3, 2, vec![1.0; 6]).unwrap();
        assert!(Samples::new(x.clone(), vec![1.0, 2.0]).is_err());
        assert!(Samples::new(x, vec![1.0, 2.0, 3.0]).is_ok());
    }
}
