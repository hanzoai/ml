//! Statistical learning as values: fitting is a transformation, a model is a value,
//! predicting is a query.
//!
//! # The two verbs
//!
//! scikit-learn's surface is enormous and its load-bearing part is two verbs. This
//! crate is those two verbs — [`Fit`] and [`Predict`] — and nothing else. There is no
//! `get_params`, no `set_params`, no `**kwargs`, no `BaseEstimator`, no `Pipeline`, no
//! clone protocol, no tags dictionary. Those exist in Python because a config and a
//! fitted model are THE SAME OBJECT there, so Python needs a reflective protocol to
//! ask an object which of its attributes are hyperparameters and which are learned
//! state. Split the two apart and the entire protocol stops having a job.
//!
//! # What is separated from what
//!
//! ```text
//!   Config          hyperparameters. A plain value the caller writes down. Has no
//!                   predict, has no learned state, and cannot acquire either.
//!      |
//!      | fit(&Samples)          Config -> Data -> Model
//!      v
//!   Model           the fitted value: parameters, and the shape they are only
//!                   meaningful against. Immutable. Nameable ([`Address`]).
//!      |
//!      | predict(&Matrix)       a QUERY against a value
//!      v
//!   Answer          f64 for a regressor, [`Class`] for a classifier — a DIFFERENT
//!                   TYPE, which is what makes the two kinds of model different
//!                   kinds of thing rather than one thing with a mode flag.
//! ```
//!
//! # `check_is_fitted` is not implemented here, and cannot be needed
//!
//! sklearn ships `check_is_fitted` because `LinearRegression()` is a predictable-from
//! object before it has seen data, so every `predict` must open by asking whether the
//! object is in a state its own type permits but its mathematics does not. That is a
//! runtime tax on a type error.
//!
//! Here, a `Model` is produced only by [`Fit::fit`]. Every `Model` in this crate holds
//! its parameters in private fields with no public constructor, no `Default`, and no
//! mutator, so an unfitted one is not a state to be checked for — it is a value that
//! cannot be spoken. The error does not need handling because it cannot be written.
//!
//! # A model is a value, and it has a name
//!
//! [`Model::address`] is a content address over the model's shape, its
//! hyperparameters and its learned parameters — the same construction, and the same
//! reasoning, as the risk plane's published model values. Two fits that agree in
//! every term that changes an answer get one name; a fit under different
//! hyperparameters gets a different name even when the parameters land identically.
//! So a fitted model can be stored, cited by a decision, compared against a
//! challenger and rolled back to, without a registry handing out identifiers.
//!
//! # What is deliberately absent
//!
//! Multi-class logistic regression, sparse matrices, sample weights, cross
//! validation, pipelines and the `ai.onnx.ml` operator surface are all NOT here. Each
//! is named where it would go, and refused with a specific error where a caller could
//! otherwise believe it happened silently. An absent feature that says so is worth
//! more than one that guesses.
//!
//! # Licence
//!
//! Clean-room. The algorithms are the published literature (Householder 1958,
//! Friedman 2001, Breiman et al. 1984) and the numerical conventions are matched to
//! scikit-learn's BEHAVIOUR, observed through fixtures its own code produced. No
//! scikit-learn source or documentation text is reproduced here, so the BSD-3-Clause
//! attribution obligation is not triggered; the fixtures record which version of it
//! was consulted as an oracle.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod address;
mod data;
mod error;
mod transform;

pub mod boost;
pub mod encode;
pub mod impute;
pub mod isolation;
pub mod linear;
pub mod logistic;
pub mod metric;
pub mod moment;
pub mod neighbour;
pub mod scale;
pub mod split;
pub mod tree;
pub mod twister;

pub use address::{Address, Digest};
pub use data::{Matrix, Samples};
pub use error::{Error, Result};
pub use transform::{Outlier, Transform};

/// Fitting: `Config -> Data -> Model`.
///
/// Implemented on CONFIG types, never on model types — a fitted model has nothing
/// left to fit, and refitting is producing a second value rather than mutating the
/// first. `&self` and not `self` because a config is a plain value worth reusing
/// across folds, and `fit` consumes nothing.
///
/// `T` is the target's own type, so the trait is what pairs a config with the kind of
/// question it can be asked: [`linear::Config`] fits against `f64` targets and
/// [`logistic::Config`] against `i64` labels, and neither can be handed the other's
/// data.
pub trait Fit<T> {
    /// The value this config produces. It is the only way to obtain one.
    type Model: Predict;

    /// Fit one model value from these observations.
    fn fit(&self, data: &Samples<T>) -> Result<Self::Model>;
}

/// Predicting: a QUERY against a fitted value.
///
/// [`Predict::Answer`] is the whole reason a regressor and a classifier are different
/// types. `Answer = f64` and `Answer = Class` do not unify, so no caller can write
/// code that treats one as the other, and no model needs a flag saying which it is.
pub trait Predict {
    /// What one row's answer is.
    type Answer;

    /// Answer for every row of `x`.
    ///
    /// Fails if `x` has a different number of features than the model was fitted on.
    /// This is the one shape agreement this crate checks at runtime rather than in the
    /// type: the feature count is not known until data is read, so putting it in the
    /// type would mean a const generic on every matrix, every config and every model,
    /// to catch a mismatch that occurs at exactly one call site.
    fn predict(&self, x: &Matrix) -> Result<Vec<Self::Answer>>;
}

/// What every fitted value in this crate is: named by its content, and honest about
/// the shape its parameters are meaningful against.
pub trait Model {
    /// This model's name, computed from its content.
    ///
    /// A PURE function of the value: no clock, no identity, no counter. Two processes
    /// holding the same model agree on the name without talking.
    fn address(&self) -> Address;

    /// How many features an input row must have.
    fn features(&self) -> usize;
}

/// One class, as an index into the model that predicted it.
///
/// A caller cannot construct one. That is the point: a `Class` in hand is always a
/// class some model was actually fitted on, so the perennial confusion between "the
/// label 2" and "the third class" is not a bug that can be written. Resolve it to the
/// caller's own label with [`logistic::Model::label`].
///
/// Labels go IN raw (`&[i64]`, the caller's data) and come OUT model-owned. The
/// asymmetry is deliberate.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug, Hash)]
pub struct Class(u32);

impl Class {
    /// Position of this class in its model's own class list.
    pub fn index(self) -> usize {
        self.0 as usize
    }

    /// Only a fitted model mints these.
    pub(crate) fn at(index: usize) -> Self {
        Self(index as u32)
    }
}
