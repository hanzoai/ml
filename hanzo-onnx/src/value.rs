//! What sits on a graph edge.
//!
//! ONNX has three value kinds — `tensor`, `seq` and `map` — and `ai.onnx` produces
//! only the first. That is why this module used to be the single line
//! `pub type Value = Tensor;`, and why the classical domain could not be read: the
//! `ai.onnx.ml` operators that scikit-learn, XGBoost and LightGBM actually export
//! produce the other two.
//!
//! * `ZipMap` — on the probability output of EVERY default classifier export —
//!   produces `seq(map(K, tensor(float)))`.
//! * A classifier fitted on string labels reports a `tensor(string)`, and
//!   `LabelEncoder`'s whole job is to move between text and numbers.
//!
//! So the value type is the sum it always was, and each kind is modelled as the thing
//! it is rather than as a tensor with a convention layered on top.

use hanzo_ml::{bail, DType, Device, IndexOp, Result, Tensor};

/// A value on a graph edge.
#[derive(Debug, Clone)]
pub enum Value {
    /// Numbers with a shape — everything `ai.onnx` reads and writes.
    Tensor(Tensor),
    /// `tensor(string)`.
    Text(Text),
    /// `seq(map(K, tensor(float)))` — scores under the labels they belong to.
    Table(Table),
}

impl Value {
    /// Which kind this is, for an error that names what it found.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Tensor(_) => "a tensor",
            Self::Text(_) => "a tensor(string)",
            Self::Table(_) => "a sequence of maps",
        }
    }

    /// The tensor this value is.
    ///
    /// The one place the tensor-only operators of `ai.onnx` meet the wider value type:
    /// an operator that cannot work on text or on a table says so by calling this and
    /// gets an error naming what arrived instead.
    pub fn tensor(&self) -> Result<&Tensor> {
        match self {
            Self::Tensor(t) => Ok(t),
            other => bail!("expected a tensor, got {}", other.kind()),
        }
    }

    /// The text this value is.
    pub fn text(&self) -> Result<&Text> {
        match self {
            Self::Text(t) => Ok(t),
            other => bail!("expected a tensor(string), got {}", other.kind()),
        }
    }

    /// The table this value is.
    pub fn table(&self) -> Result<&Table> {
        match self {
            Self::Table(t) => Ok(t),
            other => bail!("expected a sequence of maps, got {}", other.kind()),
        }
    }

    /// The tensor this value is, taking ownership.
    pub fn into_tensor(self) -> Result<Tensor> {
        match self {
            Self::Tensor(t) => Ok(t),
            other => bail!("expected a tensor, got {}", other.kind()),
        }
    }
}

impl From<Tensor> for Value {
    fn from(t: Tensor) -> Self {
        Self::Tensor(t)
    }
}

impl From<Text> for Value {
    fn from(t: Text) -> Self {
        Self::Text(t)
    }
}

impl From<Table> for Value {
    fn from(t: Table) -> Self {
        Self::Table(t)
    }
}

/// `tensor(string)`: elements, and the shape they lie in.
///
/// Not a [`Tensor`]: `hanzo_ml::DType` has no string element type, and inventing one
/// would put a variable-length heap value inside a type whose whole contract is a flat
/// numeric buffer a GPU can address. Text is a different kind of value, so it is a
/// different type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Text {
    dims: Vec<usize>,
    data: Vec<String>,
}

impl Text {
    /// Text of this shape. Fails when the shape does not account for every element.
    pub fn new(data: Vec<String>, dims: impl Into<Vec<usize>>) -> Result<Self> {
        let dims = dims.into();
        let count: usize = dims.iter().product();
        if count != data.len() {
            bail!(
                "a tensor(string) of shape {dims:?} holds {count} elements, but {} were given",
                data.len()
            );
        }
        Ok(Self { dims, data })
    }

    /// Rank-1 text — the shape every classical operator that emits text produces.
    pub fn vector(data: Vec<String>) -> Self {
        let dims = vec![data.len()];
        Self { dims, data }
    }

    /// This value's shape.
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }

    /// The elements, in row-major order.
    pub fn elements(&self) -> &[String] {
        &self.data
    }
}

/// A model's own labels: integers or text, never both and never neither.
///
/// ONNX writes a classifier's classes into two mutually exclusive attributes —
/// `classlabels_int64s` and `classlabels_strings` — and `ZipMap` keys its maps the same
/// two ways. A reader holding two vectors would admit a node that declared both, or
/// neither; a sum admits exactly one, which is the number a classifier has.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Labels {
    /// Integer labels.
    Ints(Vec<i64>),
    /// Text labels.
    Text(Vec<String>),
}

impl Labels {
    /// How many labels there are — the width of a score row.
    pub fn len(&self) -> usize {
        match self {
            Self::Ints(v) => v.len(),
            Self::Text(v) => v.len(),
        }
    }

    /// Whether there are none, which no fitted classifier has.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The labels at these positions, as the value a graph reports.
    pub fn at(&self, positions: &[usize], device: &Device) -> Result<Value> {
        match self {
            Self::Ints(v) => {
                let picked: Vec<i64> = positions.iter().map(|&k| v[k]).collect();
                let n = picked.len();
                Ok(Tensor::from_vec(picked, n, device)?.into())
            }
            Self::Text(v) => {
                Ok(Text::vector(positions.iter().map(|&k| v[k].clone()).collect()).into())
            }
        }
    }
}

/// `seq(map(K, tensor(float)))` as the value it is: one score per row per label.
///
/// `ZipMap` zips ONE label list against every row of a score matrix, so every map in
/// the sequence carries the SAME keys — that is the operator's entire contract.
/// `Vec<HashMap<K, f32>>` would admit a sequence whose maps disagree about their keys,
/// a state `ZipMap` cannot produce and no reader should have to consider. Keeping the
/// scores as one `(rows, labels)` tensor also means the numbers are not copied out of
/// the layout every other operator reads them in.
#[derive(Debug, Clone)]
pub struct Table {
    labels: Labels,
    scores: Tensor,
}

impl Table {
    /// A table of scores under these labels.
    ///
    /// Fails unless `scores` is rank-2 with one column per label — the shape a
    /// classifier's score output has, and the only shape a label list can name.
    pub fn new(labels: Labels, scores: Tensor) -> Result<Self> {
        if scores.rank() != 2 {
            bail!(
                "a table's scores are one row per sample and one column per label, so rank 2; \
                 got rank {}",
                scores.rank()
            );
        }
        let columns = scores.dim(1)?;
        if columns != labels.len() {
            bail!(
                "a table has {} labels but its scores have {columns} columns",
                labels.len()
            );
        }
        Ok(Self { labels, scores })
    }

    /// The labels its columns are under.
    pub fn labels(&self) -> &Labels {
        &self.labels
    }

    /// The scores, one row per sample.
    pub fn scores(&self) -> &Tensor {
        &self.scores
    }

    /// How many samples were scored.
    pub fn rows(&self) -> usize {
        self.scores.dims()[0]
    }

    /// One row as `(label, score)` pairs, in the model's own label order.
    ///
    /// The dictionary form a caller reading ONNX's `seq(map(...))` expects. Built on
    /// demand rather than stored, because the scores are already here in a layout that
    /// answers every other question too.
    pub fn row(&self, at: usize) -> Result<Vec<(Key, f32)>> {
        if at >= self.rows() {
            bail!("row {at} of a {}-row table", self.rows());
        }
        let scores = self.scores.i(at)?.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        Ok(match &self.labels {
            Labels::Ints(v) => v.iter().copied().map(Key::Int).zip(scores).collect(),
            Labels::Text(v) => v.iter().cloned().map(Key::Text).zip(scores).collect(),
        })
    }
}

/// One map key: whichever of the two kinds its table is labelled with.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Key {
    /// An integer key.
    Int(i64),
    /// A text key.
    Text(String),
}
