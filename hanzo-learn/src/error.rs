//! What can go wrong, named.
//!
//! The list is short because most of what scikit-learn raises at runtime is a type
//! error here: `NotFittedError` has no variant because an unfitted model is not a
//! value, and "wrong estimator kind" has none because a regressor and a classifier do
//! not unify. What remains is genuinely runtime — the shape of data nobody has read
//! yet, a matrix whose rank the caller did not know, and an optimiser reporting where
//! it stopped.

/// The result of every fallible operation in this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// What can go wrong.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Data whose dimensions do not agree, or values that are not finite.
    Shape(String),

    /// The design matrix does not have full column rank, so `||Xb - y||` has a whole
    /// affine SET of minimisers rather than one.
    ///
    /// REFUSED rather than answered. Every library picks a member of that set —
    /// scikit-learn's SVD path returns the minimum-norm one, a QR path returns a
    /// different one — and both are correct answers to a question the caller did not
    /// ask. Returning a coefficient vector here would mean silently disagreeing with
    /// another correct implementation on the same input, which is worse than failing:
    /// the caller cannot tell it happened. The named remedy is to drop the collinear
    /// column or to penalise, both of which restore uniqueness.
    Rank {
        /// Which pivot vanished.
        column: usize,
        /// How small it was, against the tolerance in force.
        pivot: f64,
    },

    /// A classifier was handed something other than two classes.
    ///
    /// Binary is the whole of what [`crate::logistic`] implements, so this is where
    /// multi-class says so out loud rather than quietly running one-versus-rest and
    /// disagreeing with a multinomial fit.
    Classes(String),

    /// An iterative fit stopped without reaching its tolerance, reported with what it
    /// actually achieved so the caller can judge rather than guess.
    Converge {
        /// Iterations spent.
        iterations: usize,
        /// The largest absolute gradient component still standing.
        gradient: f64,
        /// What was being aimed for.
        tolerance: f64,
    },

    /// A hyperparameter outside its own domain.
    Config(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Shape(m) => write!(f, "shape: {m}"),
            Self::Rank { column, pivot } => write!(
                f,
                "the design matrix is rank deficient at column {column} (pivot {pivot:e}); \
                 least squares has no unique solution, so none is returned — drop the \
                 collinear column or penalise the fit"
            ),
            Self::Classes(m) => write!(f, "classes: {m}"),
            Self::Converge {
                iterations,
                gradient,
                tolerance,
            } => write!(
                f,
                "stopped after {iterations} iterations with gradient {gradient:e}, \
                 short of {tolerance:e}"
            ),
            Self::Config(m) => write!(f, "config: {m}"),
        }
    }
}

impl std::error::Error for Error {}
