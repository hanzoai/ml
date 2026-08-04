//! The unsupervised half of the seam.

use crate::{Matrix, Result};

/// A fitted value that rewrites a design matrix: a scaler, an encoding, a projection.
///
/// The counterpart to [`crate::Fit`] for estimators that have no target to fit against.
/// [`crate::Fit`] is parameterised by the target type because pairing a config with the
/// kind of question it can be asked is that trait's whole job; a transform has no target,
/// so there is nothing to pair and the trait carries no parameter.
///
/// Both widths are known before any data arrives, which is what lets a chain of transforms
/// be checked once when it is built rather than on every batch that flows through it.
///
/// # Why fitting is not part of this trait
///
/// `fit` is an associated function on each fitted type — [`crate::scale::Standard::fit`],
/// [`crate::scale::Range::fit`] — and not a trait method. Each takes its own arguments and
/// returns its own type, so a `Fit`-shaped trait over them would have exactly one
/// implementor per signature and no call site could ever dispatch through it. Fitting is a
/// constructor, and constructors do not share a type.
pub trait Transform {
    /// How many features an input row must have: what it was fitted on.
    fn features(&self) -> usize;

    /// How many columns come out. Equal to [`Transform::features`] for a scaler, larger for
    /// an expansion, smaller for a projection.
    fn width(&self) -> usize;

    /// Rewrite `x`, producing a fresh matrix of `x.n()` rows by [`Transform::width`]
    /// columns.
    ///
    /// Fails when `x` is not as wide as this value was fitted on. Named `apply` rather than
    /// `transform` so that a call reads `scaler.apply(&x)`; scikit-learn's verb for it is
    /// `transform`.
    fn apply(&self, x: &Matrix) -> Result<Matrix>;
}

/// A fitted value that rates how unlike the fitted data each row is.
///
/// The third and last shape of estimator here, and the one an anomaly detector is:
/// [`crate::Fit`] needs a target, [`Transform`] rewrites a design, and this answers a
/// question about rows without either. [`crate::isolation::Forest`] and
/// [`crate::neighbour::Local`] are the implementors, and a caller can hold them behind this
/// trait precisely because that is the only thing they agree about — how they arrive at the
/// number could not be less alike.
///
/// # LARGER IS MORE ANOMALOUS, and why that is the opposite of scikit-learn
///
/// scikit-learn's `score_samples` returns larger for more NORMAL, because its whole API
/// obeys a "greater is better" convention inherited from scoring classifiers. That
/// convention costs a sign here: the quantity everyone actually wants is how suspicious a
/// row is, and a risk desk that feeds `score_samples` straight into a ranking gets its
/// alert queue upside down. So this trait reports suspicion directly, which is
/// `-score_samples` exactly, and the number composes with
/// [`crate::metric::Curve::of`] — whose positive class is the anomaly — with no negation at
/// the call site to forget.
pub trait Outlier {
    /// How many features an input row must have: what it was fitted on.
    fn features(&self) -> usize;

    /// How anomalous each row of `x` is. Larger is more anomalous.
    ///
    /// Fails when `x` is not as wide as this value was fitted on.
    fn outlier(&self, x: &Matrix) -> Result<Vec<f64>>;
}
