//! Least squares: the regressor.
//!
//! Fits `y ~ Xb + c` by minimising `||Xb - y||^2`.
//!
//! # Why Householder QR and not the normal equations
//!
//! `(X'X)b = X'y` is the shortest route and the wrong one: forming `X'X` SQUARES the
//! condition number, so a design matrix conditioned at 1e8 — ordinary for unscaled
//! features like "age" beside "income" — loses every significant digit through a
//! solver that was itself exact. Householder QR works on `X` directly and its error
//! grows with `cond(X)`, not `cond(X)^2`.
//!
//! scikit-learn reaches the same answer by a different route (LAPACK's SVD-based
//! `gelsd`). Both are backward stable and the minimiser is UNIQUE at full column rank,
//! so the two cannot disagree about the answer — only about the last bits of it. That
//! uniqueness is what makes an exact fixture assertion legitimate; see
//! `tests/sklearn.rs`.
//!
//! # Why the intercept is not a column of ones
//!
//! Appending a ones column is the textbook move and it is measurably worse: it makes
//! the intercept share the solver's conditioning with the slopes, and for features far
//! from the origin the ones column is nearly collinear with everything. Centring
//! instead — fit the slopes on centred data, then recover `c = mean(y) - mean(X)·b` —
//! keeps the intercept out of the linear algebra entirely. It is also exactly what
//! scikit-learn does, which is why the coefficients agree to 1e-13 rather than to
//! whatever the ones column cost.

use crate::address::{Address, Digest};
use crate::data::{mean, Matrix, Samples};
use crate::error::{Error, Result};
// The `Model` trait is referenced as `crate::Model` rather than imported: this module
// already binds that name to its own fitted value, which is the point of naming values
// by their namespace instead of by a compound.
use crate::{Fit, Predict};

/// How to fit a least-squares regressor.
///
/// Hyperparameters, and nothing learned. There is no `predict` on this type and no way
/// to put parameters into it.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Config {
    /// Fit a constant term. When false the model is `y ~ Xb` through the origin, which
    /// is a genuinely different model rather than one with `c` pinned to zero.
    pub intercept: bool,

    /// How small a QR pivot may be, relative to the largest, before the fit is refused
    /// as rank deficient.
    ///
    /// The default is `n * eps`, the standard rank tolerance: it is the scale at which a
    /// pivot is indistinguishable from the rounding that a backward-stable
    /// factorisation of that size introduces, so below it "this column is dependent" and
    /// "this column is independent and we cannot tell" are the same statement.
    pub rank_tolerance: f64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            intercept: true,
            rank_tolerance: 0.0,
        }
    }
}

impl Config {
    /// The default: with a constant term.
    pub fn new() -> Self {
        Self::default()
    }

    /// Through the origin.
    pub fn through_origin() -> Self {
        Self {
            intercept: false,
            ..Self::default()
        }
    }

    fn tolerance(&self, n: usize) -> f64 {
        if self.rank_tolerance > 0.0 {
            self.rank_tolerance
        } else {
            n as f64 * f64::EPSILON
        }
    }
}

/// A fitted least-squares regressor: a VALUE.
///
/// Private fields, no public constructor, no mutator, no `Default`. The only way to
/// hold one is to have fitted it, which is why nothing here checks whether it is
/// fitted.
#[derive(Clone, PartialEq, Debug)]
pub struct Model {
    config: Config,
    coefficients: Vec<f64>,
    intercept: f64,
}

impl Model {
    /// One coefficient per feature, in column order.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The constant term. Zero when fitted through the origin.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// The hyperparameters this value was produced under. Carried because they are part
    /// of the value's identity — see [`Model::address`].
    pub fn config(&self) -> &Config {
        &self.config
    }
}

impl crate::Model for Model {
    fn address(&self) -> Address {
        Digest::new("hanzo.learn.linear")
            .flag(self.config.intercept)
            .size(self.coefficients.len())
            .reals(&self.coefficients)
            .real(self.intercept)
            .finish()
    }

    fn features(&self) -> usize {
        self.coefficients.len()
    }
}

impl Predict for Model {
    /// A real number. THIS is what makes it a regressor — not a flag, not a base class.
    type Answer = f64;

    fn predict(&self, x: &Matrix) -> Result<Vec<f64>> {
        if x.p() != self.coefficients.len() {
            return Err(Error::Shape(format!(
                "fitted on {} features, asked about {}",
                self.coefficients.len(),
                x.p()
            )));
        }
        Ok((0..x.n())
            .map(|i| {
                // Accumulated in column order, matching the order the fit used, so
                // predicting from a model twice cannot differ in the last bit.
                self.intercept
                    + x.row(i)
                        .iter()
                        .zip(&self.coefficients)
                        .map(|(v, b)| v * b)
                        .sum::<f64>()
            })
            .collect())
    }
}

impl Fit<f64> for Config {
    type Model = Model;

    fn fit(&self, data: &Samples<f64>) -> Result<Model> {
        let (x, y) = (data.x(), data.y());
        if data.n() < data.p() {
            return Err(Error::Shape(format!(
                "{} observations cannot determine {} coefficients",
                data.n(),
                data.p()
            )));
        }

        // Centre, exactly as scikit-learn does, so the intercept never enters the
        // factorisation. With `intercept: false` nothing is centred and the fit runs on
        // the raw data — the model through the origin, not a centred model with the
        // constant discarded.
        let (x_mean, y_mean) = if self.intercept {
            (x.column_means(), mean(y))
        } else {
            (vec![0.0; data.p()], 0.0)
        };

        let mut a = vec![0.0; data.n() * data.p()];
        for i in 0..data.n() {
            for (j, (cell, m)) in a[i * data.p()..(i + 1) * data.p()]
                .iter_mut()
                .zip(&x_mean)
                .enumerate()
            {
                *cell = x.at(i, j) - m;
            }
        }
        let mut b: Vec<f64> = y.iter().map(|v| v - y_mean).collect();

        let coefficients = solve(&mut a, &mut b, data.n(), data.p(), self.tolerance(data.n()))?;
        let intercept = if self.intercept {
            y_mean
                - x_mean
                    .iter()
                    .zip(&coefficients)
                    .map(|(m, c)| m * c)
                    .sum::<f64>()
        } else {
            0.0
        };

        Ok(Model {
            config: *self,
            coefficients,
            intercept,
        })
    }
}

/// Least squares by Householder QR, in place.
///
/// `a` is `n x p` row-major and is consumed: on return its upper triangle holds `R`.
/// `b` is consumed likewise and its first `p` entries hold `Q'b`. The answer is
/// `R^-1 Q'b`.
///
/// Reflectors rather than Gram-Schmidt because a reflector is orthogonal to working
/// precision by construction, where Gram-Schmidt loses orthogonality progressively and
/// needs a second pass to get it back.
fn solve(a: &mut [f64], b: &mut [f64], n: usize, p: usize, tolerance: f64) -> Result<Vec<f64>> {
    let mut diagonal = vec![0.0; p];
    let mut v = vec![0.0; n];

    for k in 0..p {
        // The reflector that maps a[k.., k] onto a multiple of e1.
        let mut norm = 0.0;
        for i in k..n {
            let x = a[i * p + k];
            norm += x * x;
        }
        norm = norm.sqrt();
        if norm == 0.0 {
            return Err(Error::Rank {
                column: k,
                pivot: 0.0,
            });
        }
        // Sign chosen AWAY from a[k][k], so `v[0] = a[k][k] - alpha` is a sum of
        // like-signed terms. The other choice is a subtraction of nearly equal numbers
        // — the classic cancellation that costs the whole factorisation its accuracy
        // when the column is already nearly axis-aligned.
        let alpha = if a[k * p + k] >= 0.0 { -norm } else { norm };

        for i in k..n {
            v[i] = a[i * p + k];
        }
        v[k] -= alpha;
        let vv: f64 = (k..n).map(|i| v[i] * v[i]).sum();
        diagonal[k] = alpha;

        if vv > 0.0 {
            // Apply to the remaining columns and to the right-hand side. Column k is
            // known: alpha above the diagonal, zero below, so it is written rather than
            // computed.
            for j in k + 1..p {
                let dot: f64 = (k..n).map(|i| v[i] * a[i * p + j]).sum();
                let scale = 2.0 * dot / vv;
                for i in k..n {
                    a[i * p + j] -= scale * v[i];
                }
            }
            let dot: f64 = (k..n).map(|i| v[i] * b[i]).sum();
            let scale = 2.0 * dot / vv;
            for i in k..n {
                b[i] -= scale * v[i];
            }
        }
    }

    // Rank, judged on the pivots RELATIVE to the largest. An absolute threshold would
    // call a well-conditioned problem in small units rank deficient and a badly
    // conditioned one in large units full rank; only the ratio carries the information.
    let largest = diagonal.iter().fold(0.0f64, |m, d| m.max(d.abs()));
    if let Some(k) = diagonal.iter().position(|d| d.abs() <= tolerance * largest) {
        return Err(Error::Rank {
            column: k,
            pivot: diagonal[k].abs(),
        });
    }

    // Back substitution through R.
    let mut out = vec![0.0; p];
    for k in (0..p).rev() {
        let known: f64 = (k + 1..p).map(|j| a[k * p + j] * out[j]).sum();
        out[k] = (b[k] - known) / diagonal[k];
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model as _;

    fn samples(rows: &[[f64; 2]], y: &[f64]) -> Samples<f64> {
        let m = Matrix::rows(&rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>()).unwrap();
        Samples::new(m, y.to_vec()).unwrap()
    }

    #[test]
    fn an_exact_line_is_recovered_exactly() {
        // y = 2*x1 - 3*x2 + 5, no noise: the residual is zero and the fit is the plane.
        let rows = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 3.0]];
        let y: Vec<f64> = rows.iter().map(|r| 2.0 * r[0] - 3.0 * r[1] + 5.0).collect();
        let m = Config::new().fit(&samples(&rows, &y)).unwrap();
        assert!((m.coefficients()[0] - 2.0).abs() < 1e-12, "{:?}", m);
        assert!((m.coefficients()[1] + 3.0).abs() < 1e-12, "{:?}", m);
        assert!((m.intercept() - 5.0).abs() < 1e-12, "{:?}", m);
    }

    #[test]
    fn a_collinear_column_is_refused_and_not_silently_answered() {
        // x2 = 2*x1 exactly, so the minimiser is a line and no member of it is "the"
        // answer. This is the claim in Error::Rank, made falsifiable.
        let rows = [[1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0]];
        let y = [1.0, 2.0, 3.0, 4.0];
        match Config::new().fit(&samples(&rows, &y)) {
            Err(Error::Rank { .. }) => {}
            other => panic!("expected a rank refusal, got {other:?}"),
        }
    }

    #[test]
    fn fewer_observations_than_coefficients_is_refused() {
        let rows = [[1.0, 2.0]];
        assert!(Config::new().fit(&samples(&rows, &[1.0])).is_err());
    }

    #[test]
    fn predicting_with_the_wrong_feature_count_is_refused() {
        let rows = [[0.0, 0.0], [1.0, 1.0], [2.0, 4.0]];
        let m = Config::new()
            .fit(&samples(&rows, &[1.0, 2.0, 3.0]))
            .unwrap();
        assert!(m
            .predict(&Matrix::new(1, 3, vec![1.0, 2.0, 3.0]).unwrap())
            .is_err());
        assert!(m
            .predict(&Matrix::new(1, 2, vec![1.0, 2.0]).unwrap())
            .is_ok());
    }

    #[test]
    fn through_the_origin_is_a_different_model_and_a_different_name() {
        let rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
        let y = [2.0, 1.0, 4.0, 5.0];
        let a = Config::new().fit(&samples(&rows, &y)).unwrap();
        let b = Config::through_origin().fit(&samples(&rows, &y)).unwrap();
        assert_eq!(b.intercept(), 0.0);
        assert_ne!(a.address(), b.address());
    }

    #[test]
    fn the_same_data_names_the_same_model_twice() {
        let rows = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 1.0]];
        let y = [2.0, 1.0, 4.0, 5.0];
        let a = Config::new().fit(&samples(&rows, &y)).unwrap();
        let b = Config::new().fit(&samples(&rows, &y)).unwrap();
        assert_eq!(a.address(), b.address());
        assert_eq!(a.address().hex().len(), 64);
    }
}
