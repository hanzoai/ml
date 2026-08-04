//! Logistic regression: the classifier.
//!
//! Fits `P(class 1 | x) = sigma(x·w + b)` by minimising
//!
//! ```text
//!     L(w, b) = SUM log(1 + exp(-y~ (x·w + b)))  +  ||w||^2 / 2C
//! ```
//!
//! with labels `y~` in `{-1, +1}` and THE INTERCEPT UNPENALISED. That is
//! scikit-learn's objective term for term, including the convention that `C` scales the
//! data fit rather than the penalty, and including the fact that shrinking `b` toward
//! zero would be a statement about the base rate that nobody asked for.
//!
//! # Why Newton, when scikit-learn uses L-BFGS
//!
//! Because it does not matter, and that is a result rather than a shrug. The penalty
//! makes `L` STRICTLY convex — its Hessian is `X'WX + I/C`, positive definite for any
//! finite `C` — so `L` has exactly ONE minimiser. Two optimisers that both reach it
//! must agree; the only question is how far each stopped from it. So the tolerance in
//! `tests/sklearn.rs` is not a fudge factor absorbing an algorithmic difference, it is
//! the sum of two stopping distances from a point both are aiming at.
//!
//! Strict convexity is also what makes this fit total where an unpenalised one is not:
//! on separable data the unpenalised likelihood has no minimiser at all — the
//! coefficients run to infinity — and `C` is what bounds them. There is deliberately no
//! way to ask for no penalty.
//!
//! Newton is chosen for the shape of the problem: `p` is small and `n` is large, so the
//! `(p+1)x(p+1)` Hessian is cheap to form and to factor, and quadratic convergence
//! reaches 1e-12 in around ten iterations where a quasi-Newton method needs hundreds.
//! At large `p` that trade inverts and the seam for it is named in `THE SCALE SEAM`
//! below.
//!
//! # THE SCALE SEAM
//!
//! Two operations here are the whole cost at scale and both are a GEMM: forming
//! `X'WX` (`p^2 n` work) and the `Xw` product each iteration (`pn`). Both are already
//! isolated in [`hessian`] and [`scores`] and neither touches the optimiser's logic, so
//! routing them through `hanzo-ml`'s multi-backend `matmul` is a change to two function
//! bodies. It is NOT done here — see the crate's own report on why the accelerated path
//! is named rather than claimed.

use crate::address::{Address, Digest};
use crate::data::{Matrix, Samples};
use crate::error::{Error, Result};
use crate::{Class, Fit, Predict};

/// How to fit a binary logistic classifier.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Config {
    /// Inverse penalty strength, scikit-learn's `C`. Smaller means a stronger pull of
    /// `w` toward zero. Must be finite and positive: `C = inf` is the unpenalised fit,
    /// which has no minimiser on separable data, so it is not expressible.
    pub c: f64,

    /// Fit a constant term.
    pub intercept: bool,

    /// Stop when every gradient component is under this.
    ///
    /// The default is far tighter than scikit-learn's `1e-4` because the default here is
    /// not trying to save iterations — Newton reaches 1e-11 about two steps after it
    /// reaches 1e-4, so the loose default buys nothing and costs reproducibility.
    pub tolerance: f64,

    /// Give up after this many Newton steps, and say so.
    pub iterations: usize,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            c: 1.0,
            intercept: true,
            tolerance: 1e-11,
            iterations: 100,
        }
    }
}

impl Config {
    /// The default: `C = 1`, with a constant term.
    pub fn new() -> Self {
        Self::default()
    }

    /// With a different penalty.
    pub fn penalty(c: f64) -> Self {
        Self {
            c,
            ..Self::default()
        }
    }

    fn validate(&self) -> Result<()> {
        if !(self.c.is_finite() && self.c > 0.0) {
            return Err(Error::Config(format!(
                "C is {} — it must be finite and positive; an unpenalised logistic fit \
                 has no minimiser on separable data",
                self.c
            )));
        }
        if !(self.tolerance.is_finite() && self.tolerance > 0.0) {
            return Err(Error::Config(format!("tolerance is {}", self.tolerance)));
        }
        Ok(())
    }
}

/// A fitted binary logistic classifier: a VALUE.
#[derive(Clone, PartialEq, Debug)]
pub struct Model {
    config: Config,
    coefficients: Vec<f64>,
    intercept: f64,
    /// The caller's own labels, ascending. Index 1 is the class the coefficients point
    /// at, which is scikit-learn's convention and the only one that makes a single
    /// coefficient vector meaningful.
    classes: [i64; 2],
    /// How the fit ended. Reported rather than discarded, because "converged in 8" and
    /// "stopped at the cap" are different facts about the same coefficients.
    iterations: usize,
}

impl Model {
    /// One coefficient per feature, pointing at class index 1.
    pub fn coefficients(&self) -> &[f64] {
        &self.coefficients
    }

    /// The constant term.
    pub fn intercept(&self) -> f64 {
        self.intercept
    }

    /// The caller's labels, ascending. `label(Class)` resolves a prediction through it.
    pub fn classes(&self) -> [i64; 2] {
        self.classes
    }

    /// The caller's own label for a predicted class.
    ///
    /// Total, and that is the point of [`Class`] being unconstructable: there is no
    /// class this model could be handed that it does not know.
    pub fn label(&self, class: Class) -> i64 {
        self.classes[class.index()]
    }

    /// Newton steps spent reaching these coefficients.
    pub fn iterations(&self) -> usize {
        self.iterations
    }

    /// The hyperparameters this value was produced under.
    pub fn config(&self) -> &Config {
        &self.config
    }

    /// `P(class index 1 | x)` for every row.
    ///
    /// An inherent method and deliberately NOT on [`Predict`]. A calibrated probability
    /// is a property some classifiers have; hoisting it onto the shared interface is how
    /// Python ends up with `predict_proba` on estimators that only ever return a hard
    /// label, and with callers who cannot tell which is which from the type.
    pub fn probability(&self, x: &Matrix) -> Result<Vec<f64>> {
        Ok(self.z(x)?.into_iter().map(sigmoid).collect())
    }

    /// The log-odds of class index 1 — the linear score before the link.
    pub fn decision(&self, x: &Matrix) -> Result<Vec<f64>> {
        self.z(x)
    }

    fn z(&self, x: &Matrix) -> Result<Vec<f64>> {
        if x.p() != self.coefficients.len() {
            return Err(Error::Shape(format!(
                "fitted on {} features, asked about {}",
                self.coefficients.len(),
                x.p()
            )));
        }
        Ok((0..x.n())
            .map(|i| {
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

impl crate::Model for Model {
    fn address(&self) -> Address {
        Digest::new("hanzo.learn.logistic")
            .real(self.config.c)
            .flag(self.config.intercept)
            .ints(&self.classes)
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
    /// A class this model knows. THIS is what makes it a classifier — the answer type,
    /// not a flag and not a base class. `Answer = Class` and `Answer = f64` do not
    /// unify, so a regressor and a classifier cannot be confused by any caller.
    type Answer = Class;

    fn predict(&self, x: &Matrix) -> Result<Vec<Class>> {
        // Strictly greater than zero, matching scikit-learn: an exact tie on the
        // decision boundary goes to class index 0.
        Ok(self
            .z(x)?
            .into_iter()
            .map(|z| Class::at(usize::from(z > 0.0)))
            .collect())
    }
}

impl Fit<i64> for Config {
    type Model = Model;

    fn fit(&self, data: &Samples<i64>) -> Result<Model> {
        self.validate()?;
        let (x, labels) = (data.x(), data.y());
        let classes = two_classes(labels)?;
        let (n, p) = (data.n(), data.p());

        // Targets as 0/1 against the ASCENDING class order, so which class the
        // coefficients point at is a property of the data and not of the row order.
        let y: Vec<f64> = labels.iter().map(|l| f64::from(*l == classes[1])).collect();

        // The intercept is the LAST coordinate of the augmented system, and the penalty
        // is added only to the first p diagonal entries — that placement is the whole
        // implementation of "the intercept is unpenalised".
        let dim = if self.intercept { p + 1 } else { p };
        let alpha = 1.0 / self.c;
        let mut w = vec![0.0; dim];
        let mut spent = 0;

        for step in 1..=self.iterations {
            spent = step;
            let z = scores(x, &w, self.intercept);
            let gradient = gradient(x, &y, &z, &w, p, alpha, self.intercept);
            let worst = gradient.iter().fold(0.0f64, |m, g| m.max(g.abs()));
            if worst < self.tolerance {
                spent = step - 1;
                break;
            }

            let mut h = hessian(x, &z, p, alpha, self.intercept);
            // The step solves H d = -g. H is positive definite for any finite C, so a
            // Cholesky factorisation exists; if rounding loses it anyway, the ridge is
            // grown rather than the fit abandoned.
            let direction = loop {
                match cholesky_solve(&mut h.clone(), &gradient, dim) {
                    Some(d) => break d,
                    None => {
                        for i in 0..dim {
                            h[i * dim + i] += 1e-10 * (1.0 + h[i * dim + i].abs());
                        }
                    }
                }
            };

            // Backtracking, because an undamped Newton step can overshoot on data far
            // from the optimum and a fit that oscillates is worse than one that crawls.
            // Halving is bounded: the direction is a descent direction, so SOME step
            // reduces the objective unless the gradient is already zero.
            let before = objective(x, &y, &w, p, alpha, self.intercept);
            let mut scale = 1.0;
            let mut next = w.clone();
            for _ in 0..40 {
                for (t, (v, d)) in next.iter_mut().zip(w.iter().zip(&direction)) {
                    *t = v - scale * d;
                }
                if objective(x, &y, &next, p, alpha, self.intercept) <= before {
                    break;
                }
                scale *= 0.5;
            }
            w = next;
        }

        let z = scores(x, &w, self.intercept);
        let gradient = gradient(x, &y, &z, &w, p, alpha, self.intercept);
        let worst = gradient.iter().fold(0.0f64, |m, g| m.max(g.abs()));
        if worst >= self.tolerance {
            return Err(Error::Converge {
                iterations: spent,
                gradient: worst,
                tolerance: self.tolerance,
            });
        }
        let _ = n;

        Ok(Model {
            config: *self,
            coefficients: w[..p].to_vec(),
            intercept: if self.intercept { w[p] } else { 0.0 },
            classes,
            iterations: spent,
        })
    }
}

/// The two classes in the data, ascending.
///
/// Exactly two, or refused by name. One-versus-rest and multinomial disagree on
/// multi-class problems, so a library that quietly picked one would be a library whose
/// answers depend on a choice the caller never saw.
fn two_classes(labels: &[i64]) -> Result<[i64; 2]> {
    let mut seen: Vec<i64> = labels.to_vec();
    seen.sort_unstable();
    seen.dedup();
    match seen.len() {
        2 => Ok([seen[0], seen[1]]),
        1 => Err(Error::Classes(format!(
            "every label is {} — a classifier fitted on one class has nothing to \
             discriminate",
            seen[0]
        ))),
        k => Err(Error::Classes(format!(
            "{k} distinct labels; this is a binary classifier. Multi-class is not \
             implemented, and is refused here rather than run as one-versus-rest, which \
             would disagree with a multinomial fit without saying so"
        ))),
    }
}

/// `sigma(z)`, without overflowing on either tail.
///
/// `1/(1+exp(-z))` alone overflows `exp` for z around -750 and returns a NaN where the
/// answer is 0. Branching on the sign keeps the exponent negative in both arms.
fn sigmoid(z: f64) -> f64 {
    if z >= 0.0 {
        1.0 / (1.0 + (-z).exp())
    } else {
        let e = z.exp();
        e / (1.0 + e)
    }
}

/// `log(1 + exp(-t))`, without overflowing.
fn softplus_neg(t: f64) -> f64 {
    if t >= 0.0 {
        (-t).exp().ln_1p()
    } else {
        -t + t.exp().ln_1p()
    }
}

/// The linear scores. ONE of the two GEMMs — see `THE SCALE SEAM`.
fn scores(x: &Matrix, w: &[f64], intercept: bool) -> Vec<f64> {
    let p = x.p();
    let b = if intercept { w[p] } else { 0.0 };
    (0..x.n())
        .map(|i| {
            b + x
                .row(i)
                .iter()
                .zip(&w[..p])
                .map(|(v, c)| v * c)
                .sum::<f64>()
        })
        .collect()
}

fn objective(x: &Matrix, y: &[f64], w: &[f64], p: usize, alpha: f64, intercept: bool) -> f64 {
    let z = scores(x, w, intercept);
    // Labels back to {-1,+1}: the objective is stated in that convention and evaluating
    // it in another is where a sign error hides.
    let fit: f64 = y
        .iter()
        .zip(&z)
        .map(|(y01, z)| softplus_neg((2.0 * y01 - 1.0) * z))
        .sum();
    let penalty: f64 = 0.5 * alpha * w[..p].iter().map(|v| v * v).sum::<f64>();
    fit + penalty
}

fn gradient(
    x: &Matrix,
    y: &[f64],
    z: &[f64],
    w: &[f64],
    p: usize,
    alpha: f64,
    intercept: bool,
) -> Vec<f64> {
    let dim = if intercept { p + 1 } else { p };
    let mut g = vec![0.0; dim];
    for i in 0..x.n() {
        let residual = sigmoid(z[i]) - y[i];
        for (gj, v) in g[..p].iter_mut().zip(x.row(i)) {
            *gj += residual * v;
        }
        if intercept {
            g[p] += residual;
        }
    }
    for (gj, wj) in g[..p].iter_mut().zip(&w[..p]) {
        *gj += alpha * wj;
    }
    g
}

/// `X'WX + alpha*I`, augmented with the intercept row and column.
///
/// Symmetric, so only the lower triangle is accumulated and then mirrored — half the
/// work and, more importantly, exact symmetry by construction rather than by luck,
/// which is what the Cholesky below assumes.
///
/// THE OTHER GEMM — see `THE SCALE SEAM`.
fn hessian(x: &Matrix, z: &[f64], p: usize, alpha: f64, intercept: bool) -> Vec<f64> {
    let dim = if intercept { p + 1 } else { p };
    let mut h = vec![0.0; dim * dim];
    for i in 0..x.n() {
        let s = sigmoid(z[i]);
        let weight = s * (1.0 - s);
        let row = x.row(i);
        for a in 0..p {
            let wa = weight * row[a];
            for b in 0..=a {
                h[a * dim + b] += wa * row[b];
            }
            if intercept {
                h[p * dim + a] += wa;
            }
        }
        if intercept {
            h[p * dim + p] += weight;
        }
    }
    for a in 0..p {
        h[a * dim + a] += alpha;
    }
    for a in 0..dim {
        for b in 0..a {
            h[b * dim + a] = h[a * dim + b];
        }
    }
    h
}

/// Solves `H d = g` for a symmetric positive definite `H`, by Cholesky.
///
/// `None` when `H` is not positive definite to working precision — reported rather than
/// papered over, so the caller above can grow a ridge and retry instead of proceeding
/// on a direction that is not a descent direction.
fn cholesky_solve(h: &mut [f64], g: &[f64], dim: usize) -> Option<Vec<f64>> {
    // In-place lower-triangular factor.
    for i in 0..dim {
        for j in 0..=i {
            let mut sum = h[i * dim + j];
            for k in 0..j {
                sum -= h[i * dim + k] * h[j * dim + k];
            }
            if i == j {
                // `is_nan()` is NOT redundant beside `<= 0.0`: a NaN pivot fails that
                // comparison too, and would then be square-rooted into a NaN factor whose
                // back substitution yields a NaN step — a "descent direction" the caller
                // would follow into a model of nothing but NaN. Both cases are exactly
                // "not positive definite to working precision", the one thing this
                // function reports, so both return here.
                if sum.is_nan() || sum <= 0.0 {
                    return None;
                }
                h[i * dim + j] = sum.sqrt();
            } else {
                h[i * dim + j] = sum / h[j * dim + j];
            }
        }
    }
    // Forward then back.
    let mut d = vec![0.0; dim];
    for i in 0..dim {
        let mut sum = g[i];
        for k in 0..i {
            sum -= h[i * dim + k] * d[k];
        }
        d[i] = sum / h[i * dim + i];
    }
    for i in (0..dim).rev() {
        let mut sum = d[i];
        for k in i + 1..dim {
            sum -= h[k * dim + i] * d[k];
        }
        d[i] = sum / h[i * dim + i];
    }
    Some(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Model as _;

    fn samples(rows: &[[f64; 2]], y: &[i64]) -> Samples<i64> {
        let m = Matrix::rows(&rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>()).unwrap();
        Samples::new(m, y.to_vec()).unwrap()
    }

    const ROWS: [[f64; 2]; 8] = [
        [-2.0, 1.0],
        [-1.5, 0.5],
        [-1.0, -0.5],
        [-0.5, 0.0],
        [0.5, 0.5],
        [1.0, -1.0],
        [1.5, 1.0],
        [2.0, 0.0],
    ];
    const LABELS: [i64; 8] = [0, 0, 0, 0, 1, 1, 1, 1];

    #[test]
    fn a_separable_problem_still_converges_because_the_penalty_bounds_it() {
        // Unpenalised, these coefficients would run to infinity and no optimiser would
        // stop. This is the claim in the module header, made falsifiable.
        let m = Config::new().fit(&samples(&ROWS, &LABELS)).unwrap();
        assert!(
            m.coefficients()[0].is_finite() && m.coefficients()[0] > 0.0,
            "{m:?}"
        );
        assert!(m.iterations() < 100, "took {}", m.iterations());
    }

    #[test]
    fn a_weaker_penalty_grows_the_coefficients() {
        // The direction of C is a property worth pinning: a sign error in the penalty
        // would leave every other test passing.
        let tight = Config::penalty(0.01).fit(&samples(&ROWS, &LABELS)).unwrap();
        let loose = Config::penalty(100.0)
            .fit(&samples(&ROWS, &LABELS))
            .unwrap();
        assert!(
            tight.coefficients()[0].abs() < loose.coefficients()[0].abs(),
            "tight {:?} loose {:?}",
            tight.coefficients(),
            loose.coefficients()
        );
    }

    #[test]
    fn one_class_and_three_classes_are_both_refused_by_name() {
        assert!(matches!(
            Config::new().fit(&samples(&ROWS, &[0; 8])),
            Err(Error::Classes(_))
        ));
        assert!(matches!(
            Config::new().fit(&samples(&ROWS, &[0, 1, 2, 0, 1, 2, 0, 1])),
            Err(Error::Classes(_))
        ));
    }

    #[test]
    fn a_predicted_class_resolves_to_the_callers_own_label() {
        // Labels are 7 and 9, not 0 and 1. A model that returned raw indices would look
        // right in every test that happened to use 0/1.
        let labels: Vec<i64> = LABELS.iter().map(|l| if *l == 0 { 7 } else { 9 }).collect();
        let m = Config::new().fit(&samples(&ROWS, &labels)).unwrap();
        assert_eq!(m.classes(), [7, 9]);
        let x = Matrix::rows(&[vec![2.0, 0.0], vec![-2.0, 1.0]]).unwrap();
        let got: Vec<i64> = m
            .predict(&x)
            .unwrap()
            .into_iter()
            .map(|c| m.label(c))
            .collect();
        assert_eq!(got, vec![9, 7]);
    }

    #[test]
    fn probability_and_predict_agree_on_the_boundary_they_share() {
        let m = Config::new().fit(&samples(&ROWS, &LABELS)).unwrap();
        let x = Matrix::rows(&ROWS.iter().map(|r| r.to_vec()).collect::<Vec<_>>()).unwrap();
        for (c, p) in m
            .predict(&x)
            .unwrap()
            .into_iter()
            .zip(m.probability(&x).unwrap())
        {
            assert_eq!(
                c.index() == 1,
                p > 0.5,
                "class and probability disagree at p={p}"
            );
        }
    }

    #[test]
    fn an_unpenalised_fit_is_not_expressible() {
        assert!(matches!(
            Config::penalty(f64::INFINITY).fit(&samples(&ROWS, &LABELS)),
            Err(Error::Config(_))
        ));
        assert!(matches!(
            Config::penalty(0.0).fit(&samples(&ROWS, &LABELS)),
            Err(Error::Config(_))
        ));
    }

    #[test]
    fn the_penalty_is_part_of_the_name() {
        let a = Config::penalty(1.0).fit(&samples(&ROWS, &LABELS)).unwrap();
        let b = Config::penalty(2.0).fit(&samples(&ROWS, &LABELS)).unwrap();
        assert_ne!(a.address(), b.address());
    }
}
