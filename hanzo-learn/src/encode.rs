//! Turning categories into numbers, and back.
//!
//! Two values, composed rather than merged into one:
//!
//! ```text
//!   &[T]   --Label::codes-->   Vec<Class>   --OneHot::apply-->   Matrix
//!   raw labels                 one column                        indicators
//! ```
//!
//! scikit-learn's `OneHotEncoder` does both steps at once and therefore holds the
//! vocabulary of every column inside one object. Splitting them means the vocabulary
//! lives in exactly one type ([`Label`]), which is also the type a caller needs on its own
//! to encode a target column — so the vocabulary is not implemented twice, and a
//! `LabelEncoder` and a `OneHotEncoder` do not have to agree about anything.
//!
//! # Scale
//!
//! [`Label`] holds one entry per distinct category, so its size is the vocabulary and not
//! the data. Fitting sorts the distinct values, which needs them resident: `O(k)` where
//! `k` is the number of distinct categories, not `O(n)`. Encoding and one-hot expansion
//! are single passes with no state.

use std::collections::BTreeSet;
use std::fmt::Debug;

use crate::{Class, Error, Matrix, Result};

/// A vocabulary: the distinct categories of one feature, in ascending order.
///
/// Ascending because two independently fitted vocabularies over the same categories must
/// agree on which class is which, and the order values arrive in is not a property of the
/// data. It is also what scikit-learn does, so codes are directly comparable with a Python
/// baseline.
#[derive(Debug, Clone, PartialEq)]
pub struct Label<T> {
    classes: Vec<T>,
}

impl<T: Ord + Clone + Debug> Label<T> {
    /// Fit on the observed values of one feature.
    ///
    /// Fails on no observations: a vocabulary of nothing can encode nothing, and every
    /// later call would have to answer for it.
    pub fn fit(values: &[T]) -> Result<Self> {
        if values.is_empty() {
            return Err(Error::Shape(
                "a vocabulary needs at least one observation".to_string(),
            ));
        }
        let classes: Vec<T> = values
            .iter()
            .cloned()
            .collect::<BTreeSet<T>>()
            .into_iter()
            .collect();
        Ok(Self { classes })
    }

    /// The categories, in code order.
    pub fn classes(&self) -> &[T] {
        &self.classes
    }

    /// How many categories there are.
    pub fn len(&self) -> usize {
        self.classes.len()
    }

    /// Whether the vocabulary is empty. Never true: [`Label::fit`] refuses to build one.
    pub fn is_empty(&self) -> bool {
        self.classes.is_empty()
    }

    /// The class of one raw value.
    ///
    /// Fails on a category that was not present when fitting, which is scikit-learn's
    /// default and the only defensible one: silently mapping an unknown category onto an
    /// existing class would make a model answer confidently about something it never saw.
    pub fn code(&self, value: &T) -> Result<Class> {
        match self.classes.binary_search(value) {
            Ok(i) => Ok(Class::at(i)),
            Err(_) => Err(Error::Classes(format!(
                "category {value:?} was not present when the vocabulary was fitted"
            ))),
        }
    }

    /// The classes of many raw values, reporting the first that was never seen.
    pub fn codes(&self, values: &[T]) -> Result<Vec<Class>> {
        values.iter().map(|v| self.code(v)).collect()
    }

    /// The raw value a class stands for, or nothing if the class is not from this
    /// vocabulary.
    pub fn label(&self, class: Class) -> Option<&T> {
        self.classes.get(class.index())
    }
}

/// A categorical design: `n` observations of `k` categorical features, as classes.
///
/// Only classes go in, so a continuous feature cannot be one-hot expanded by mistake —
/// the type of the input to [`OneHot::apply`] rules it out rather than a check inside it.
#[derive(Debug, Clone, PartialEq)]
pub struct Codes {
    rows: usize,
    cols: usize,
    data: Vec<Class>,
}

impl Codes {
    /// From row-major classes.
    pub fn new(rows: usize, cols: usize, data: Vec<Class>) -> Result<Self> {
        if rows == 0 || cols == 0 {
            return Err(Error::Shape(format!(
                "a categorical design is {rows}x{cols}; both dimensions must be non-zero"
            )));
        }
        if data.len() != rows * cols {
            return Err(Error::Shape(format!(
                "{rows}x{cols} needs {} values, got {}",
                rows * cols,
                data.len()
            )));
        }
        Ok(Self { rows, cols, data })
    }

    /// Stitch one already-encoded column per feature into a design.
    ///
    /// Fails unless every column has the same length, which is the one agreement a design
    /// made out of columns has to establish.
    pub fn columns(columns: &[Vec<Class>]) -> Result<Self> {
        let cols = columns.len();
        let rows = columns.first().map_or(0, Vec::len);
        if let Some(j) = columns.iter().position(|c| c.len() != rows) {
            return Err(Error::Shape(format!(
                "feature 0 has {rows} observations but feature {j} has {}",
                columns[j].len()
            )));
        }
        let mut data = Vec::with_capacity(rows * cols);
        for i in 0..rows {
            for c in columns {
                data.push(c[i]);
            }
        }
        Self::new(rows, cols, data)
    }

    /// Number of observations.
    pub fn n(&self) -> usize {
        self.rows
    }

    /// Number of categorical features.
    pub fn p(&self) -> usize {
        self.cols
    }

    /// One cell.
    pub fn at(&self, i: usize, j: usize) -> Class {
        self.data[i * self.cols + j]
    }
}

/// Expand categorical features into indicator columns, one per category.
///
/// Built from the sizes of the vocabularies rather than from data. That is the whole
/// design: the set of categories is a property of the [`Label`] that defined it, and a
/// batch of data is not evidence about it. An encoder fitted from a sample would silently
/// narrow when a rare category is absent from that sample, and every downstream width
/// would move with it.
#[derive(Debug, Clone, PartialEq)]
pub struct OneHot {
    levels: Vec<usize>,
    offset: Vec<usize>,
    width: usize,
}

impl OneHot {
    /// From the number of categories of each feature, in feature order — that is
    /// [`Label::len`] of each vocabulary.
    pub fn of(levels: &[usize]) -> Result<Self> {
        if levels.is_empty() {
            return Err(Error::Config(
                "a one-hot encoding needs at least one feature".to_string(),
            ));
        }
        if let Some(j) = levels.iter().position(|&k| k == 0) {
            return Err(Error::Config(format!(
                "feature {j} has no categories, so it has no indicator columns"
            )));
        }
        let mut offset = Vec::with_capacity(levels.len());
        let mut width = 0;
        for &k in levels {
            offset.push(width);
            width += k;
        }
        Ok(Self {
            levels: levels.to_vec(),
            offset,
            width,
        })
    }

    /// How many categorical features are expected.
    pub fn features(&self) -> usize {
        self.levels.len()
    }

    /// How many indicator columns are produced: the total number of categories.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Which output columns one input feature occupies, so a caller can name a
    /// coefficient afterwards.
    pub fn span(&self, feature: usize) -> std::ops::Range<usize> {
        self.offset[feature]..self.offset[feature] + self.levels[feature]
    }

    /// Expand a categorical design into indicators.
    ///
    /// Fails on a class outside its feature's vocabulary, naming where — which is the
    /// unseen-category case arriving through a hand-built [`Codes`] rather than through
    /// [`Label::code`].
    pub fn apply(&self, codes: &Codes) -> Result<Matrix> {
        if codes.p() != self.levels.len() {
            return Err(Error::Shape(format!(
                "fitted on {} features, given {}",
                self.levels.len(),
                codes.p()
            )));
        }
        let mut out = vec![0.0f64; codes.n() * self.width];
        for i in 0..codes.n() {
            for j in 0..codes.p() {
                let c = codes.at(i, j).index();
                if c >= self.levels[j] {
                    return Err(Error::Classes(format!(
                        "row {i} feature {j} is class {c}, outside the {} categories the \
                         encoding was built for",
                        self.levels[j]
                    )));
                }
                out[i * self.width + self.offset[j] + c] = 1.0;
            }
        }
        Matrix::new(codes.n(), self.width, out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_vocabulary_is_ascending_so_two_fits_agree() {
        let a = Label::fit(&["red", "blue", "green", "blue"]).unwrap();
        let b = Label::fit(&["green", "red", "blue"]).unwrap();
        assert_eq!(a.classes(), b.classes());
        assert_eq!(a.classes(), &["blue", "green", "red"]);
        assert_eq!(a.code(&"blue").unwrap().index(), 0);
        assert_eq!(a.code(&"red").unwrap().index(), 2);
        assert_eq!(a.label(a.code(&"green").unwrap()), Some(&"green"));
    }

    #[test]
    fn an_unseen_category_is_refused_and_named() {
        let v = Label::fit(&[1i64, 5, 9]).unwrap();
        let e = v.code(&7).unwrap_err();
        assert!(format!("{e}").contains('7'), "{e}");
        assert!(v.codes(&[1, 9, 5]).is_ok());
        assert!(v.codes(&[1, 7]).is_err());
    }

    #[test]
    fn one_hot_lays_features_out_side_by_side_in_feature_order() {
        let colour = Label::fit(&["red", "blue", "green"]).unwrap();
        let size = Label::fit(&["big", "small"]).unwrap();
        let codes = Codes::columns(&[
            colour.codes(&["red", "blue", "green"]).unwrap(),
            size.codes(&["small", "big", "small"]).unwrap(),
        ])
        .unwrap();
        let hot = OneHot::of(&[colour.len(), size.len()]).unwrap();
        assert_eq!(hot.width(), 5);
        assert_eq!(hot.span(0), 0..3);
        assert_eq!(hot.span(1), 3..5);
        let m = hot.apply(&codes).unwrap();
        // red = class 2 of {blue, green, red}; small = class 1 of {big, small}
        assert_eq!(m.row(0), &[0.0, 0.0, 1.0, 0.0, 1.0]);
        assert_eq!(m.row(1), &[1.0, 0.0, 0.0, 1.0, 0.0]);
        assert_eq!(m.row(2), &[0.0, 1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn an_encoding_is_built_from_the_vocabulary_and_not_from_a_sample() {
        let colour = Label::fit(&["red", "blue", "green"]).unwrap();
        let hot = OneHot::of(&[colour.len()]).unwrap();
        // A batch containing only one category still produces the full width, which is
        // the entire reason the encoding is not fitted from data.
        let codes = Codes::columns(&[colour.codes(&["blue", "blue"]).unwrap()]).unwrap();
        assert_eq!(hot.apply(&codes).unwrap().p(), 3);
    }

    #[test]
    fn a_class_from_outside_the_vocabulary_is_refused_with_its_position() {
        let hot = OneHot::of(&[2, 2]).unwrap();
        let codes = Codes::new(1, 2, vec![Class::at(0), Class::at(5)]).unwrap();
        let e = hot.apply(&codes).unwrap_err();
        assert!(format!("{e}").contains("row 0 feature 1"), "{e}");
        assert!(OneHot::of(&[]).is_err());
        assert!(OneHot::of(&[2, 0]).is_err());
    }

    #[test]
    fn ragged_columns_cannot_become_a_design() {
        let e = Codes::columns(&[vec![Class::at(0); 3], vec![Class::at(0); 2]]).unwrap_err();
        assert!(format!("{e}").contains("feature 1"), "{e}");
    }
}
