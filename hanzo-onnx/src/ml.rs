//! The `ai.onnx.ml` operator domain: the classical, non-neural half of ONNX.
//!
//! `eval.rs` implements `ai.onnx` — convolutions, attention, activations. That is the
//! domain a neural network exports to. Everything that is not a neural network —
//! gradient-boosted trees, random forests, logistic regression, support vector
//! machines, the feature transforms in front of them — exports to a *second* domain,
//! `ai.onnx.ml`, whose operators carry their whole model in their ATTRIBUTES rather
//! than in graph edges. A `TreeEnsembleClassifier` node holds every threshold of every
//! tree; there is no `Conv` anywhere in such a graph.
//!
//! This domain IS the scikit-learn surface in serialized form, and not only
//! scikit-learn's: `TreeEnsembleClassifier` is also what XGBoost and LightGBM export
//! to, so three ecosystems arrive through one operator.
//!
//! # Where the numbers come from
//!
//! Several operators have behaviour the specification states loosely or not at all —
//! what a binary classifier's second score is, whether `AVERAGE` divides before or
//! after `base_values`, what `MAX` normalization does to an all-negative row, what
//! shape `ArrayFeatureExtractor` gives a rank-1 input. Each such case here was
//! MEASURED against onnxruntime 1.28.0 rather than guessed, and the measurement is
//! recorded at the code it decided. `tests/ml.rs` re-checks the same cases end to end
//! against real exports.
//!
//! # What this domain does not cover here
//!
//! `CastMap` and `DictVectorizer` take a `map` as INPUT; nothing in this evaluator
//! produces one on an edge (`ZipMap` is terminal in every real export), so they are
//! refused by name. The opset-5 `TreeEnsemble` — a different attribute schema
//! superseding `TreeEnsembleClassifier`/`Regressor`, which no exporter emits by default
//! yet — is refused by name too.

use crate::eval::{get_attr, get_attr_opt, get_attr_opt_owned};
use crate::onnx;
use crate::value::{Labels, Table, Text, Value};
use hanzo_ml::{bail, DType, Device, Error, Result, Tensor};
use std::collections::hash_map::Entry;
use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------------
// Reading the numeric plane
// ---------------------------------------------------------------------------------

/// A feature matrix flattened to `rows x features` f32, row-major.
///
/// Classical operators read scalar features by index and compare them to thresholds, so
/// they want rows of numbers, not a strided tensor. One conversion at the edge keeps
/// every operator below free of layout concerns.
struct Matrix {
    rows: usize,
    features: usize,
    data: Vec<f32>,
}

impl Matrix {
    fn of(x: &Tensor) -> Result<Self> {
        let (rows, features) = match x.rank() {
            1 => (1, x.dim(0)?),
            2 => (x.dim(0)?, x.dim(1)?),
            rank => bail!("expected a rank-1 or rank-2 feature matrix, got rank {rank}"),
        };
        let x = if x.dtype() == DType::F32 {
            x.clone()
        } else {
            x.to_dtype(DType::F32)?
        };
        let data = x.contiguous()?.flatten_all()?.to_vec1::<f32>()?;
        Ok(Self {
            rows,
            features,
            data,
        })
    }

    fn row(&self, r: usize) -> &[f32] {
        &self.data[r * self.features..(r + 1) * self.features]
    }
}

/// Rewrite every element as a function of its column index and its value, keeping the
/// input's shape.
///
/// The shape of `Scaler`, `Binarizer` and `Imputer`: a per-feature map. One helper, so
/// three operators cannot disagree about which axis "per feature" means or what happens
/// to a rank-1 input.
fn per_feature(x: &Tensor, f: impl Fn(usize, f32) -> f32) -> Result<Tensor> {
    let m = Matrix::of(x)?;
    let features = m.features;
    let mut data = m.data;
    for (i, v) in data.iter_mut().enumerate() {
        *v = f(i % features.max(1), *v);
    }
    Tensor::from_vec(data, x.dims().to_vec(), x.device())
}

/// Rewrite every row in place, keeping the input's shape.
///
/// The shape of `Normalizer`: a whole row decides its own answer.
fn per_row(x: &Tensor, f: impl Fn(&mut [f32]) -> Result<()>) -> Result<Tensor> {
    let m = Matrix::of(x)?;
    let features = m.features;
    let mut data = m.data;
    for row in data.chunks_mut(features.max(1)) {
        f(row)?;
    }
    Tensor::from_vec(data, x.dims().to_vec(), x.device())
}

// ---------------------------------------------------------------------------------
// Scores
// ---------------------------------------------------------------------------------

/// How raw scores become reported ones — ONNX's `post_transform` attribute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PostTransform {
    None,
    Softmax,
    /// Softmax that leaves exact zeros at zero and out of the sum, so classes a model
    /// assigned no mass keep none.
    SoftmaxZero,
    Logistic,
}

impl PostTransform {
    /// The four transforms, under the names a file spells them with. [`Self::of`] and
    /// [`Self::spelling`] read the SAME table, so a name cannot drift from its meaning.
    const NAMES: [(&'static str, Self); 4] = [
        ("NONE", Self::None),
        ("SOFTMAX", Self::Softmax),
        ("SOFTMAX_ZERO", Self::SoftmaxZero),
        ("LOGISTIC", Self::Logistic),
    ];

    fn of(node: &onnx::NodeProto) -> Result<Self> {
        let named = get_attr_opt::<str>(node, "post_transform")?.unwrap_or("NONE");
        Self::NAMES
            .iter()
            .find(|(name, _)| *name == named)
            .map(|(_, transform)| *transform)
            // PROBIT is the inverse normal CDF, which needs an erf inverse this crate
            // does not carry. Named rather than approximated: a wrong probability is
            // worse than a refused one.
            .ok_or_else(|| {
                Error::Msg(format!(
                    "unsupported post_transform {named:?} in {} ({})",
                    node.op_type, node.name
                ))
            })
    }

    /// How a file spells this transform, for a message that has to name it.
    fn spelling(self) -> &'static str {
        Self::NAMES
            .iter()
            .find(|(_, transform)| *transform == self)
            .map(|(name, _)| *name)
            .expect("every transform is named in NAMES")
    }

    fn apply(self, row: &mut [f32]) {
        match self {
            Self::None => {}
            Self::Logistic => {
                for s in row.iter_mut() {
                    *s = 1.0 / (1.0 + (-*s).exp());
                }
            }
            Self::Softmax => softmax(row, false),
            Self::SoftmaxZero => softmax(row, true),
        }
    }
}

/// Numerically-stable softmax over one row, shifting by the row maximum so a large
/// score cannot overflow `exp`.
fn softmax(row: &mut [f32], skip_zeros: bool) {
    let taken = |s: f32| !skip_zeros || s != 0.0;
    let max = row
        .iter()
        .copied()
        .filter(|&s| taken(s))
        .fold(f32::NEG_INFINITY, f32::max);
    if !max.is_finite() {
        return;
    }
    let mut sum = 0.0f32;
    for s in row.iter_mut() {
        if !taken(*s) {
            continue;
        }
        *s = (*s - max).exp();
        sum += *s;
    }
    if sum > 0.0 {
        for s in row.iter_mut() {
            *s /= sum;
        }
    }
}

// ---------------------------------------------------------------------------------
// Tree ensembles
// ---------------------------------------------------------------------------------

/// The comparison a branch makes.
///
/// `LEAF` is deliberately not a variant: a leaf is a different kind of node, not a
/// branch whose test is impossible, so nothing here has to invent an answer for one.
/// See [`Node`].
#[derive(Debug, Clone, Copy)]
enum Test {
    Leq,
    Lt,
    Gte,
    Gt,
    Eq,
    Neq,
}

impl Test {
    fn of(mode: &str) -> Result<Self> {
        Ok(match mode {
            "BRANCH_LEQ" => Self::Leq,
            "BRANCH_LT" => Self::Lt,
            "BRANCH_GTE" => Self::Gte,
            "BRANCH_GT" => Self::Gt,
            "BRANCH_EQ" => Self::Eq,
            "BRANCH_NEQ" => Self::Neq,
            other => bail!("unsupported tree node mode {other:?}"),
        })
    }

    fn takes_yes(self, x: f32, threshold: f32) -> bool {
        match self {
            Self::Leq => x <= threshold,
            Self::Lt => x < threshold,
            Self::Gte => x >= threshold,
            Self::Gt => x > threshold,
            Self::Eq => x == threshold,
            Self::Neq => x != threshold,
        }
    }
}

/// One resolved tree node. Successors are flat indices, so a walk is a pointer chase
/// with no lookups in it.
enum Node {
    Branch {
        test: Test,
        feature: usize,
        threshold: f32,
        yes: usize,
        no: usize,
        /// Where a NaN feature goes. ONNX spells this
        /// `nodes_missing_value_tracks_true`, and LightGBM sets it.
        missing_takes_yes: bool,
    },
    /// The `(output index, weight)` pairs this leaf contributes.
    Leaf(Vec<(usize, f32)>),
}

/// How one tree's contributions combine with the others'.
///
/// Measured against onnxruntime with two stumps whose leaves are `1/3` and `10/30`:
/// `SUM = 11`, `AVERAGE = 5.5`, `MIN = 1`, `MAX = 10`; and with `base_values = [100]`,
/// `MIN = 101` and `AVERAGE = 105.5`. So the base is added AFTER the aggregation,
/// including after `AVERAGE`'s division — the one ordering a reader is likely to get
/// backwards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Aggregate {
    Sum,
    Average,
    Min,
    Max,
}

impl Aggregate {
    fn of(node: &onnx::NodeProto) -> Result<Self> {
        Ok(
            match get_attr_opt::<str>(node, "aggregate_function")?.unwrap_or("SUM") {
                "SUM" => Self::Sum,
                "AVERAGE" => Self::Average,
                "MIN" => Self::Min,
                "MAX" => Self::Max,
                other => bail!(
                    "unsupported {} aggregate_function {other:?} in {}",
                    node.op_type,
                    node.name
                ),
            },
        )
    }

    fn fold(self, running: f32, next: f32) -> f32 {
        match self {
            Self::Sum | Self::Average => running + next,
            Self::Min => running.min(next),
            Self::Max => running.max(next),
        }
    }
}

/// A decision-tree ensemble resolved from ONNX's flat attribute arrays.
///
/// One ensemble arrives as ~8 equal-length parallel arrays describing nodes, plus a
/// second group keyed by `(tree id, node id)` describing leaf weights. Resolving that
/// once per node — successors to flat indices, weights onto their leaves — is what
/// leaves [`Self::score`] with nothing to look up.
struct Ensemble {
    nodes: Vec<Node>,
    /// Flat index of each tree's root, in the order the trees first appear.
    roots: Vec<usize>,
    /// Output columns: classes for a classifier, targets for a regressor.
    width: usize,
    base: Vec<f32>,
    /// Whether every leaf weight is non-negative. Only the binary classifier reads it,
    /// and only to reproduce onnxruntime's choice of second score.
    weights_all_positive: bool,
}

impl Ensemble {
    /// `group` is the attribute prefix naming the leaf weights: `"class"` for
    /// `TreeEnsembleClassifier`, `"target"` for `TreeEnsembleRegressor`. The two
    /// operators are the same machine over differently-named arrays, so they share one
    /// builder.
    ///
    /// `fold` maps a weight's declared output index onto a column of this ensemble. It
    /// is the identity except in the binary classifier case, where every weight
    /// addresses one column whatever index it names.
    fn build(
        node: &onnx::NodeProto,
        width: usize,
        group: &str,
        fold: impl Fn(i64) -> Option<usize>,
    ) -> Result<Self> {
        let treeids = get_attr::<[i64]>(node, "nodes_treeids")?;
        let nodeids = get_attr::<[i64]>(node, "nodes_nodeids")?;
        let featureids = get_attr::<[i64]>(node, "nodes_featureids")?;
        let thresholds = get_attr::<[f32]>(node, "nodes_values")?;
        let yes_ids = get_attr::<[i64]>(node, "nodes_truenodeids")?;
        let no_ids = get_attr::<[i64]>(node, "nodes_falsenodeids")?;
        let missing =
            get_attr_opt::<[i64]>(node, "nodes_missing_value_tracks_true")?.unwrap_or(&[]);
        let modes = get_attr_opt_owned::<Vec<String>>(node, "nodes_modes")?.ok_or_else(|| {
            Error::Msg(format!(
                "{} in {} has no nodes_modes attribute",
                node.op_type, node.name
            ))
        })?;

        let n = treeids.len();
        for (name, len) in [
            ("nodes_nodeids", nodeids.len()),
            ("nodes_featureids", featureids.len()),
            ("nodes_values", thresholds.len()),
            ("nodes_truenodeids", yes_ids.len()),
            ("nodes_falsenodeids", no_ids.len()),
            ("nodes_modes", modes.len()),
        ] {
            if len != n {
                bail!(
                    "{} in {}: {name} has {len} entries but nodes_treeids has {n}",
                    node.op_type,
                    node.name
                );
            }
        }

        // (tree id, node id) -> flat index. Needed to resolve successors, and dropped
        // once they are resolved.
        let mut index = HashMap::with_capacity(n);
        for i in 0..n {
            if index.insert((treeids[i], nodeids[i]), i).is_some() {
                bail!(
                    "{} in {}: tree {} declares node {} twice",
                    node.op_type,
                    node.name,
                    treeids[i],
                    nodeids[i]
                );
            }
        }

        // Leaf weights, keyed the same way, gathered onto the leaves they belong to. A
        // leaf carries one entry per output it gives mass to.
        let weights = get_attr::<[f32]>(node, &format!("{group}_weights"))?;
        let weight_ids = get_attr::<[i64]>(node, &format!("{group}_ids"))?;
        let weight_treeids = get_attr::<[i64]>(node, &format!("{group}_treeids"))?;
        let weight_nodeids = get_attr::<[i64]>(node, &format!("{group}_nodeids"))?;
        for (name, len) in [
            (format!("{group}_ids"), weight_ids.len()),
            (format!("{group}_treeids"), weight_treeids.len()),
            (format!("{group}_nodeids"), weight_nodeids.len()),
        ] {
            if len != weights.len() {
                bail!(
                    "{} in {}: {name} has {len} entries but {group}_weights has {}",
                    node.op_type,
                    node.name,
                    weights.len()
                );
            }
        }
        let mut leaves: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        for w in 0..weights.len() {
            let at = *index
                .get(&(weight_treeids[w], weight_nodeids[w]))
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "{} in {}: {group}_weights[{w}] names tree {} node {}, which no node declares",
                        node.op_type, node.name, weight_treeids[w], weight_nodeids[w]
                    ))
                })?;
            let k = fold(weight_ids[w]).filter(|k| *k < width).ok_or_else(|| {
                Error::Msg(format!(
                    "{} in {}: {group}_ids[{w}] is {}, outside the {width} outputs",
                    node.op_type, node.name, weight_ids[w]
                ))
            })?;
            leaves[at].push((k, weights[w]));
        }

        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            nodes.push(match modes[i].as_str() {
                "LEAF" => Node::Leaf(std::mem::take(&mut leaves[i])),
                mode => {
                    let successor = |id: i64| -> Result<usize> {
                        index.get(&(treeids[i], id)).copied().ok_or_else(|| {
                            Error::Msg(format!(
                                "{} in {}: tree {} node {} branches to node {id}, which it does not declare",
                                node.op_type, node.name, treeids[i], nodeids[i]
                            ))
                        })
                    };
                    let feature = featureids[i];
                    if feature < 0 {
                        bail!(
                            "{} in {}: tree {} node {} reads feature {feature}",
                            node.op_type,
                            node.name,
                            treeids[i],
                            nodeids[i]
                        );
                    }
                    Node::Branch {
                        test: Test::of(mode)?,
                        feature: feature as usize,
                        threshold: thresholds[i],
                        yes: successor(yes_ids[i])?,
                        no: successor(no_ids[i])?,
                        missing_takes_yes: missing.get(i).is_some_and(|&v| v != 0),
                    }
                }
            });
        }

        // A tree's root is its lowest node id. Trees keep the order in which they first
        // appear, so the walk is deterministic across runs.
        let mut roots: Vec<usize> = Vec::new();
        let mut slot_of: HashMap<i64, usize> = HashMap::new();
        for i in 0..n {
            match slot_of.entry(treeids[i]) {
                Entry::Vacant(e) => {
                    e.insert(roots.len());
                    roots.push(i);
                }
                Entry::Occupied(e) => {
                    let slot = *e.get();
                    if nodeids[i] < nodeids[roots[slot]] {
                        roots[slot] = i;
                    }
                }
            }
        }

        let base = match get_attr_opt::<[f32]>(node, "base_values")? {
            Some(b) if !b.is_empty() => {
                if b.len() != width {
                    bail!(
                        "{} in {}: base_values has {} entries for {width} outputs",
                        node.op_type,
                        node.name,
                        b.len()
                    );
                }
                b.to_vec()
            }
            _ => vec![0.0; width],
        };

        Ok(Self {
            nodes,
            roots,
            width,
            base,
            weights_all_positive: weights.iter().all(|w| *w >= 0.0),
        })
    }

    /// The leaf one tree sends this row to.
    fn leaf(&self, root: usize, row: &[f32]) -> Result<&[(usize, f32)]> {
        let mut at = root;
        let mut steps = 0usize;
        loop {
            match &self.nodes[at] {
                Node::Leaf(weights) => return Ok(weights),
                Node::Branch {
                    test,
                    feature,
                    threshold,
                    yes,
                    no,
                    missing_takes_yes,
                } => {
                    let x = match row.get(*feature) {
                        Some(&x) => x,
                        None => bail!(
                            "a tree node reads feature {feature} of a {}-feature row",
                            row.len()
                        ),
                    };
                    at = if x.is_nan() {
                        if *missing_takes_yes {
                            *yes
                        } else {
                            *no
                        }
                    } else if test.takes_yes(x, *threshold) {
                        *yes
                    } else {
                        *no
                    };
                }
            }
            // The resolved tree is acyclic, but the file it came from is not trusted to
            // be: bound the descent so a malformed graph fails instead of spinning.
            steps += 1;
            if steps > self.nodes.len() {
                bail!(
                    "a tree walk passed {} nodes without reaching a leaf — the ensemble has a cycle",
                    self.nodes.len()
                );
            }
        }
    }

    /// Score every row: aggregate the trees, add `base_values`, then `post_transform`.
    ///
    /// The accumulator is `Option<f32>` per output and not `0.0`, because `MIN` and
    /// `MAX` have no identity element: a target that no tree contributes to must report
    /// zero, not `f32::MAX`.
    fn score(&self, m: &Matrix, post: PostTransform, agg: Aggregate) -> Result<Vec<f32>> {
        let trees = self.roots.len().max(1) as f32;
        let mut scores = vec![0f32; m.rows * self.width];
        let mut acc: Vec<Option<f32>> = vec![None; self.width];
        for r in 0..m.rows {
            acc.iter_mut().for_each(|a| *a = None);
            let row = m.row(r);
            for &root in self.roots.iter() {
                for &(k, w) in self.leaf(root, row)?.iter() {
                    acc[k] = Some(match acc[k] {
                        None => w,
                        Some(running) => agg.fold(running, w),
                    });
                }
            }
            let out = &mut scores[r * self.width..(r + 1) * self.width];
            for k in 0..self.width {
                let total = acc[k].unwrap_or(0.0);
                let total = if agg == Aggregate::Average {
                    total / trees
                } else {
                    total
                };
                out[k] = self.base[k] + total;
            }
            post.apply(out);
        }
        Ok(scores)
    }
}

// ---------------------------------------------------------------------------------
// Classifier outputs
// ---------------------------------------------------------------------------------

/// A classifier's own class labels, from whichever attribute it declares them in.
///
/// `ints` is a parameter because the domain is not self-consistent about the name:
/// `TreeEnsembleClassifier` and `SVMClassifier` write `classlabels_int64s` and
/// `classlabels_ints` respectively — measured on real exports, not read off the spec.
fn labels(node: &onnx::NodeProto, ints: &str) -> Result<Labels> {
    if let Some(v) = get_attr_opt::<[i64]>(node, ints)? {
        if !v.is_empty() {
            return Ok(Labels::Ints(v.to_vec()));
        }
    }
    match get_attr_opt_owned::<Vec<String>>(node, "classlabels_strings")? {
        Some(v) if !v.is_empty() => Ok(Labels::Text(v)),
        _ => bail!(
            "{} in {} declares no class labels — neither {ints} nor classlabels_strings",
            node.op_type,
            node.name
        ),
    }
}

/// The two outputs every classical classifier has: the winning label per row, and the
/// per-class scores.
///
/// ONE function, so `TreeEnsembleClassifier`, `LinearClassifier` and `SVMClassifier`
/// cannot disagree about how a tie is broken or how a text label is carried. Ties go to
/// the lowest class index, which is what every implementation of `argmax` here does.
fn classified(
    labels: &Labels,
    scores: Vec<f32>,
    rows: usize,
    device: &Device,
) -> Result<Vec<Value>> {
    let classes = labels.len();
    let best: Vec<usize> = scores
        .chunks(classes)
        .map(|row| {
            let mut best = 0usize;
            for k in 1..classes {
                if row[k] > row[best] {
                    best = k;
                }
            }
            best
        })
        .collect();
    Ok(vec![
        labels.at(&best, device)?,
        Tensor::from_vec(scores, (rows, classes), device)?.into(),
    ])
}

/// `scores[r][k] = sum_j coefficients[k*features + j] * row[r][j] + intercepts[k]`.
///
/// The one affine map that both `LinearRegressor` and `LinearClassifier` are.
fn affine(node: &onnx::NodeProto, m: &Matrix, width: usize) -> Result<Vec<f32>> {
    let coefficients = get_attr::<[f32]>(node, "coefficients")?;
    let intercepts = get_attr_opt::<[f32]>(node, "intercepts")?.unwrap_or(&[]);
    if coefficients.len() != width * m.features {
        bail!(
            "{} in {}: {} coefficients for {width} outputs over {} features",
            node.op_type,
            node.name,
            coefficients.len(),
            m.features
        );
    }
    if !intercepts.is_empty() && intercepts.len() != width {
        bail!(
            "{} in {}: {} intercepts for {width} outputs",
            node.op_type,
            node.name,
            intercepts.len()
        );
    }
    let mut out = vec![0f32; m.rows * width];
    for r in 0..m.rows {
        let row = m.row(r);
        for k in 0..width {
            let w = &coefficients[k * m.features..(k + 1) * m.features];
            let mut acc = intercepts.get(k).copied().unwrap_or(0.0);
            for j in 0..m.features {
                acc += w[j] * row[j];
            }
            out[r * width + k] = acc;
        }
    }
    Ok(out)
}

/// A two-class model that scores ONE column, widened to the two columns it reports.
///
/// XGBoost and LightGBM both export binary classifiers this way — one score, two labels
/// — and the second score is not stated by the specification. MEASURED against
/// onnxruntime 1.28.0 on hand-built nodes:
///
/// ```text
///   post_transform  leaf weights    score s   reported
///   NONE            all >= 0        0.25      [0.75, 0.25]    = [1 - s, s]
///   NONE            mixed sign     -0.25      [0.25, -0.25]   = [-s, s]
///   LOGISTIC        all >= 0        0.25      [0.4378, 0.5622] = [1 - p, p], p = sigmoid(s)
///   LOGISTIC        mixed sign     -0.25      [0.5622, 0.4378] = [1 - p, p]
/// ```
///
/// So the transform is applied to the single score FIRST and the complement is `1 - p`
/// except in the one case where no transform ran and a weight was negative — where a
/// probability was never claimed, so a complement would be a lie.
fn widen(scores: Vec<f32>, post: PostTransform, weights_all_positive: bool) -> Vec<f32> {
    let complement = post != PostTransform::None || weights_all_positive;
    let mut out = Vec::with_capacity(scores.len() * 2);
    for p in scores {
        out.push(if complement { 1.0 - p } else { -p });
        out.push(p);
    }
    out
}

// ---------------------------------------------------------------------------------
// The element plane: text and numbers, mapped
// ---------------------------------------------------------------------------------

/// One column of elements, in whichever of ONNX's three classical element types it
/// carries.
///
/// `LabelEncoder`, `CategoryMapper` and `OneHotEncoder` are all the same lookup over
/// this type — find each element's position in a key column, then answer from that
/// position — so there is ONE implementation instead of one per key/value pairing.
#[derive(Debug, Clone, PartialEq)]
enum Column {
    Ints(Vec<i64>),
    Reals(Vec<f32>),
    Text(Vec<String>),
}

impl Column {
    /// The column an input edge carries.
    fn of(value: &Value) -> Result<Self> {
        match value {
            Value::Text(t) => Ok(Self::Text(t.elements().to_vec())),
            Value::Tensor(t) => {
                let flat = t.contiguous()?.flatten_all()?;
                Ok(match t.dtype() {
                    DType::I64 => Self::Ints(flat.to_vec1::<i64>()?),
                    DType::U32 => {
                        Self::Ints(flat.to_vec1::<u32>()?.into_iter().map(i64::from).collect())
                    }
                    DType::F32 | DType::F64 | DType::F16 | DType::BF16 => {
                        Self::Reals(flat.to_dtype(DType::F32)?.to_vec1::<f32>()?)
                    }
                    other => bail!("an ai.onnx.ml lookup cannot read {other:?} elements"),
                })
            }
            other => bail!("expected elements to map, got {}", other.kind()),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::Ints(v) => v.len(),
            Self::Reals(v) => v.len(),
            Self::Text(v) => v.len(),
        }
    }

    fn kind(&self) -> &'static str {
        match self {
            Self::Ints(_) => "int64",
            Self::Reals(_) => "float",
            Self::Text(_) => "string",
        }
    }

    /// Where each of `self`'s elements sits in `keys`, or `None` where absent.
    ///
    /// Reals are matched on their bit pattern, which is what makes the lookup a lookup:
    /// a key that is not bit-identical is a different key, and NaN — which is equal to
    /// nothing, including itself — is findable exactly when the key list holds the same
    /// NaN.
    fn positions(&self, keys: &Self) -> Result<Vec<Option<usize>>> {
        Ok(match (self, keys) {
            (Self::Ints(xs), Self::Ints(ks)) => {
                let index: HashMap<i64, usize> = enumerate(ks.iter().copied());
                xs.iter().map(|x| index.get(x).copied()).collect()
            }
            (Self::Reals(xs), Self::Reals(ks)) => {
                let index: HashMap<u32, usize> = enumerate(ks.iter().map(|k| k.to_bits()));
                xs.iter()
                    .map(|x| index.get(&x.to_bits()).copied())
                    .collect()
            }
            (Self::Text(xs), Self::Text(ks)) => {
                let index: HashMap<&str, usize> = enumerate(ks.iter().map(String::as_str));
                xs.iter().map(|x| index.get(x.as_str()).copied()).collect()
            }
            (x, k) => bail!(
                "an ai.onnx.ml lookup was given {} elements and {} keys",
                x.kind(),
                k.kind()
            ),
        })
    }

    /// The elements at these positions, with `absent` filling the gaps.
    fn gather(&self, at: &[Option<usize>], absent: &Cell) -> Result<Self> {
        Ok(match (self, absent) {
            (Self::Ints(v), Cell::Int(d)) => {
                Self::Ints(at.iter().map(|p| p.map_or(*d, |p| v[p])).collect())
            }
            (Self::Reals(v), Cell::Real(d)) => {
                Self::Reals(at.iter().map(|p| p.map_or(*d, |p| v[p])).collect())
            }
            (Self::Text(v), Cell::Text(d)) => Self::Text(
                at.iter()
                    .map(|p| p.map_or_else(|| d.clone(), |p| v[p].clone()))
                    .collect(),
            ),
            (v, _) => bail!(
                "an ai.onnx.ml lookup has {} values but a default of another type",
                v.kind()
            ),
        })
    }

    /// This column as a value of the given shape.
    fn value(self, dims: Vec<usize>, device: &Device) -> Result<Value> {
        Ok(match self {
            Self::Ints(v) => Tensor::from_vec(v, dims, device)?.into(),
            Self::Reals(v) => Tensor::from_vec(v, dims, device)?.into(),
            Self::Text(v) => Text::new(v, dims)?.into(),
        })
    }
}

/// The first position of each key, so an earlier duplicate wins — the order a lookup
/// table is read in.
fn enumerate<K: std::hash::Hash + Eq>(keys: impl Iterator<Item = K>) -> HashMap<K, usize> {
    let mut index = HashMap::new();
    for (i, k) in keys.enumerate() {
        index.entry(k).or_insert(i);
    }
    index
}

/// One element: what a lookup answers where a key is absent.
#[derive(Debug, Clone, PartialEq)]
enum Cell {
    Int(i64),
    Real(f32),
    Text(String),
}

/// A key column, a value column, and what to answer for a key that is in neither.
///
/// `LabelEncoder` and `CategoryMapper` are this value read out of differently-named
/// attributes. Nothing below them knows which operator it came from.
struct Mapping {
    keys: Column,
    values: Column,
    absent: Cell,
}

impl Mapping {
    /// Read whichever of the three key attributes and three value attributes a node
    /// declares. Exactly one of each, so a node that declares two — or none — is
    /// refused rather than served from whichever the reader happened to check first.
    fn of(node: &onnx::NodeProto, keys: [&str; 3], values: [&str; 3]) -> Result<Self> {
        let ints = |name: &str| -> Result<Option<Column>> {
            Ok(get_attr_opt::<[i64]>(node, name)?
                .filter(|v| !v.is_empty())
                .map(|v| Column::Ints(v.to_vec())))
        };
        let reals = |name: &str| -> Result<Option<Column>> {
            Ok(get_attr_opt::<[f32]>(node, name)?
                .filter(|v| !v.is_empty())
                .map(|v| Column::Reals(v.to_vec())))
        };
        let text = |name: &str| -> Result<Option<Column>> {
            Ok(get_attr_opt_owned::<Vec<String>>(node, name)?
                .filter(|v| !v.is_empty())
                .map(Column::Text))
        };
        let one = |group: &str, names: [&str; 3]| -> Result<Column> {
            let found: Vec<Column> = [ints(names[0])?, reals(names[1])?, text(names[2])?]
                .into_iter()
                .flatten()
                .collect();
            match found.len() {
                1 => Ok(found.into_iter().next().expect("just counted one")),
                n => bail!(
                    "{} in {} declares {n} of the {group} attributes {names:?}; exactly one is a mapping",
                    node.op_type,
                    node.name
                ),
            }
        };
        let keys = one("key", keys)?;
        let values = one("value", values)?;
        if keys.len() != values.len() {
            bail!(
                "{} in {}: {} keys and {} values",
                node.op_type,
                node.name,
                keys.len(),
                values.len()
            );
        }
        let absent = match &values {
            Column::Ints(_) => {
                Cell::Int(*get_attr_opt::<i64>(node, "default_int64")?.unwrap_or(&-1))
            }
            Column::Reals(_) => {
                Cell::Real(*get_attr_opt::<f32>(node, "default_float")?.unwrap_or(&f32::NAN))
            }
            Column::Text(_) => Cell::Text(
                get_attr_opt::<str>(node, "default_string")?
                    .unwrap_or("_Unused")
                    .to_string(),
            ),
        };
        Ok(Self {
            keys,
            values,
            absent,
        })
    }

    fn apply(&self, input: &Value) -> Result<Value> {
        let column = Column::of(input)?;
        let at = column.positions(&self.keys)?;
        let dims = match input {
            Value::Tensor(t) => t.dims().to_vec(),
            Value::Text(t) => t.dims().to_vec(),
            other => bail!("expected elements to map, got {}", other.kind()),
        };
        self.values
            .gather(&at, &self.absent)?
            .value(dims, device_of(input))
    }
}

/// Where an input's value lives, so an output lands on the same device.
fn device_of(value: &Value) -> &Device {
    match value {
        Value::Tensor(t) => t.device(),
        Value::Table(t) => t.scores().device(),
        // Text is not on a device; the graph's numeric plane decides, and CPU is where
        // a `tensor(string)` was read.
        Value::Text(_) => &Device::Cpu,
    }
}

// ---------------------------------------------------------------------------------
// Support vector machines
// ---------------------------------------------------------------------------------

/// The kernel an SVM scores through.
///
/// `kernel_params` is `[gamma, coef0, degree]` — positional, which is why it is read
/// once here and never indexed again.
#[derive(Debug, Clone, Copy)]
enum Kernel {
    Linear,
    Polynomial { gamma: f32, coef0: f32, degree: f32 },
    Radial { gamma: f32 },
    Sigmoid { gamma: f32, coef0: f32 },
}

impl Kernel {
    fn of(node: &onnx::NodeProto) -> Result<Self> {
        let p = get_attr_opt::<[f32]>(node, "kernel_params")?.unwrap_or(&[]);
        let at = |i: usize| p.get(i).copied().unwrap_or(0.0);
        let (gamma, coef0, degree) = (at(0), at(1), at(2));
        Ok(
            match get_attr_opt::<str>(node, "kernel_type")?.unwrap_or("LINEAR") {
                "LINEAR" => Self::Linear,
                "POLY" => Self::Polynomial {
                    gamma,
                    coef0,
                    degree,
                },
                "RBF" => Self::Radial { gamma },
                "SIGMOID" => Self::Sigmoid { gamma, coef0 },
                other => bail!(
                    "unsupported {} kernel_type {other:?} in {}",
                    node.op_type,
                    node.name
                ),
            },
        )
    }

    fn of_pair(self, a: &[f32], b: &[f32]) -> f32 {
        let dot = || a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        match self {
            Self::Linear => dot(),
            Self::Polynomial {
                gamma,
                coef0,
                degree,
            } => (gamma * dot() + coef0).powf(degree),
            Self::Radial { gamma } => {
                let square: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
                (-gamma * square).exp()
            }
            Self::Sigmoid { gamma, coef0 } => (gamma * dot() + coef0).tanh(),
        }
    }
}

/// A fitted support vector machine, resolved from its attributes.
struct Machine {
    kernel: Kernel,
    /// `support` rows of `features` columns.
    support: Vec<f32>,
    features: usize,
    vectors: usize,
    coefficients: Vec<f32>,
    rho: Vec<f32>,
}

impl Machine {
    fn of(node: &onnx::NodeProto, features: usize) -> Result<Self> {
        let support = get_attr_opt::<[f32]>(node, "support_vectors")?
            .unwrap_or(&[])
            .to_vec();
        if features == 0 || support.len() % features != 0 {
            bail!(
                "{} in {}: {} support vector values do not divide into rows of {features} features",
                node.op_type,
                node.name,
                support.len()
            );
        }
        let vectors = support.len() / features;
        let coefficients = get_attr::<[f32]>(node, "coefficients")?;
        // Every support vector carries at least one dual coefficient — the regressor one
        // each, the classifier one per decision plane — so a machine with fewer
        // coefficients than vectors is not a small model, it is a model whose scoring
        // loop would read past the array it was given. onnxruntime refuses the same file:
        // "coefficients size (1) must be >= n_supports (10)".
        if coefficients.len() < vectors {
            bail!(
                "{} in {}: {} coefficients for {vectors} support vectors; a machine has at \
                 least one per vector",
                node.op_type,
                node.name,
                coefficients.len()
            );
        }
        Ok(Self {
            kernel: Kernel::of(node)?,
            vectors,
            support,
            features,
            coefficients: coefficients.to_vec(),
            rho: get_attr_opt::<[f32]>(node, "rho")?.unwrap_or(&[]).to_vec(),
        })
    }

    fn vector(&self, i: usize) -> &[f32] {
        &self.support[i * self.features..(i + 1) * self.features]
    }

    /// `kernel(x, sv)` for every support vector.
    fn kernels(&self, row: &[f32]) -> Vec<f32> {
        (0..self.vectors)
            .map(|i| self.kernel.of_pair(row, self.vector(i)))
            .collect()
    }
}

/// libsvm's one-against-one decision values, one per class pair, in the pair order
/// `(0,1), (0,2), ..., (0,n-1), (1,2), ...` that `rho` is written in.
///
/// `coefficients` is `(classes - 1) x vectors`: row `j - 1` holds the dual coefficients
/// of the vectors that belong to class `i` when scoring the pair `(i, j)`, and row `i`
/// holds class `j`'s. That indexing is libsvm's and is why this is one function rather
/// than something a reader is expected to rederive.
///
/// `rho` is ADDED. It is not libsvm's `rho`, which is subtracted — it is scikit-learn's
/// `intercept_`, which is the negation, and skl2onnx writes that. Measured: for a
/// three-class iris SVC, `sigma + rho` reproduces onnxruntime's node output to seven
/// digits on all three pairs and `sigma - rho` reproduces none of them.
fn one_against_one(m: &Machine, kernels: &[f32], per_class: &[usize]) -> Result<Vec<f32>> {
    let classes = per_class.len();
    let starts: Vec<usize> = per_class
        .iter()
        .scan(0usize, |at, n| {
            let start = *at;
            *at += n;
            Some(start)
        })
        .collect();
    let pairs = classes * (classes - 1) / 2;
    if m.rho.len() != pairs {
        bail!(
            "an SVM over {classes} classes has {pairs} class pairs but {} rho values",
            m.rho.len()
        );
    }
    if m.coefficients.len() != (classes - 1) * m.vectors {
        bail!(
            "an SVM over {classes} classes and {} vectors wants {} coefficients, not {}",
            m.vectors,
            (classes - 1) * m.vectors,
            m.coefficients.len()
        );
    }
    let coefficient = |plane: usize, at: usize| m.coefficients[plane * m.vectors + at];
    let mut out = Vec::with_capacity(pairs);
    let mut p = 0usize;
    for i in 0..classes {
        for j in (i + 1)..classes {
            let mut sum = m.rho[p];
            for k in 0..per_class[i] {
                let at = starts[i] + k;
                sum += coefficient(j - 1, at) * kernels[at];
            }
            for k in 0..per_class[j] {
                let at = starts[j] + k;
                sum += coefficient(i, at) * kernels[at];
            }
            out.push(sum);
            p += 1;
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------------
// The operator table
// ---------------------------------------------------------------------------------

/// The widest row `FeatureVectorizer` will build — 16 MiB of f32 per row.
///
/// Its output width is the sum of an attribute, not of anything the inputs carry, so it is
/// the one width in this domain an untrusted file can name freely.
const MAX_WIDTH: usize = 1 << 22;

/// Evaluate one `ai.onnx.ml` node. Inputs arrive resolved and positional; outputs come
/// back positional, for the caller to bind to `node.output`.
pub(crate) fn eval(node: &onnx::NodeProto, inputs: &[Value]) -> Result<Vec<Value>> {
    let input = |i: usize| -> Result<&Value> {
        inputs.get(i).ok_or_else(|| {
            Error::Msg(format!(
                "{} in {} wants input {i} but got {} inputs",
                node.op_type,
                node.name,
                inputs.len()
            ))
        })
    };
    let numeric = |i: usize| -> Result<&Tensor> { input(i)?.tensor() };

    match node.op_type.as_str() {
        // Y = (X - offset) * scale, per feature. A length-1 offset or scale applies to
        // every feature.
        "Scaler" => {
            let x = numeric(0)?;
            let offset = get_attr::<[f32]>(node, "offset")?;
            let scale = get_attr::<[f32]>(node, "scale")?;
            let features = features_of(x)?;
            for (name, v) in [("offset", offset), ("scale", scale)] {
                if v.len() != 1 && v.len() != features {
                    bail!(
                        "Scaler in {}: {name} has {} entries for a {features}-feature row",
                        node.name,
                        v.len()
                    );
                }
            }
            let pick = |v: &[f32], j: usize| if v.len() == 1 { v[0] } else { v[j] };
            Ok(vec![per_feature(x, |j, v| {
                (v - pick(offset, j)) * pick(scale, j)
            })?
            .into()])
        }

        // Each row divided by its MAX, L1 or L2 norm.
        //
        // MAX is the SIGNED maximum, not the largest magnitude: measured against
        // onnxruntime, `[-4, 2, -1]` normalizes to `[-2, 1, -0.5]` and the all-negative
        // `[-4, -2, -1]` to `[4, 2, 1]`, dividing by -1. A zero norm leaves the row
        // alone rather than producing NaNs — also measured.
        "Normalizer" => {
            let x = numeric(0)?;
            let norm = get_attr_opt::<str>(node, "norm")?.unwrap_or("MAX");
            Ok(vec![per_row(x, |row| {
                let divisor = match norm {
                    "MAX" => row.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    "L1" => row.iter().map(|v| v.abs()).sum(),
                    "L2" => row.iter().map(|v| v * v).sum::<f32>().sqrt(),
                    other => {
                        bail!("unsupported Normalizer norm {other:?} in {}", node.name)
                    }
                };
                if divisor != 0.0 && divisor.is_finite() {
                    for v in row.iter_mut() {
                        *v /= divisor;
                    }
                }
                Ok(())
            })?
            .into()])
        }

        // Y = X > threshold. Strictly greater: measured, a value EQUAL to the threshold
        // reports 0.
        "Binarizer" => {
            let x = numeric(0)?;
            let threshold = *get_attr_opt::<f32>(node, "threshold")?.unwrap_or(&0.0);
            Ok(vec![per_feature(
                x,
                |_, v| {
                    if v > threshold {
                        1.0
                    } else {
                        0.0
                    }
                },
            )?
            .into()])
        }

        // Replace one value with another, per feature. `replaced_value_float` may be
        // NaN, which is how a pipeline's missing-value marker arrives, and NaN is equal
        // to nothing — so the comparison is a NaN test in that case rather than `==`.
        "Imputer" => {
            let x = numeric(0)?;
            let features = features_of(x)?;
            let imputed = match get_attr_opt::<[f32]>(node, "imputed_value_floats")? {
                Some(v) if !v.is_empty() => v.to_vec(),
                _ => match get_attr_opt::<[i64]>(node, "imputed_value_int64s")? {
                    Some(v) if !v.is_empty() => v.iter().map(|i| *i as f32).collect(),
                    _ => bail!(
                        "Imputer in {} declares neither imputed_value_floats nor \
                         imputed_value_int64s",
                        node.name
                    ),
                },
            };
            if imputed.len() != 1 && imputed.len() != features {
                bail!(
                    "Imputer in {}: {} imputed values for a {features}-feature row",
                    node.name,
                    imputed.len()
                );
            }
            let replaced = match get_attr_opt::<f32>(node, "replaced_value_float")? {
                Some(v) => *v,
                None => get_attr_opt::<i64>(node, "replaced_value_int64")?
                    .map(|i| *i as f32)
                    .unwrap_or(0.0),
            };
            let hit = |v: f32| {
                if replaced.is_nan() {
                    v.is_nan()
                } else {
                    v == replaced
                }
            };
            let pick = |j: usize| {
                if imputed.len() == 1 {
                    imputed[0]
                } else {
                    imputed[j]
                }
            };
            Ok(vec![per_feature(
                x,
                |j, v| if hit(v) { pick(j) } else { v },
            )?
            .into()])
        }

        "LinearRegressor" => {
            let x = numeric(0)?;
            let m = Matrix::of(x)?;
            let targets = (*get_attr_opt::<i64>(node, "targets")?.unwrap_or(&1)).max(1) as usize;
            let mut out = affine(node, &m, targets)?;
            let post = PostTransform::of(node)?;
            for row in out.chunks_mut(targets) {
                post.apply(row);
            }
            Ok(vec![
                Tensor::from_vec(out, (m.rows, targets), x.device())?.into()
            ])
        }

        "LinearClassifier" => {
            let x = numeric(0)?;
            let m = Matrix::of(x)?;
            let labels = labels(node, "classlabels_ints")?;
            let classes = labels.len();
            if classes < 2 {
                bail!(
                    "LinearClassifier in {} declares {classes} class labels",
                    node.name
                );
            }
            let coefficients = get_attr::<[f32]>(node, "coefficients")?;
            // A two-class model MAY carry one weight row and report the second class as
            // the first's complement. skl2onnx writes the full matrix even for binary
            // problems — measured: 8 coefficients for 2 classes over 4 features, the
            // second row the negation of the first — so both shapes are read, and the
            // single-row one is widened exactly as a binary tree ensemble's is.
            let single = classes == 2 && coefficients.len() == m.features;
            let post = PostTransform::of(node)?;
            let mut scores = affine(node, &m, if single { 1 } else { classes })?;
            if single {
                for s in scores.iter_mut() {
                    let mut one = [*s];
                    post.apply(&mut one);
                    *s = one[0];
                }
                // No leaf weights to inspect, so the complement is the only reading
                // available: a linear score has no sign convention that says otherwise.
                scores = widen(scores, post, true);
            } else {
                for row in scores.chunks_mut(classes) {
                    post.apply(row);
                }
            }
            classified(&labels, scores, m.rows, x.device())
        }

        "TreeEnsembleClassifier" => {
            let x = numeric(0)?;
            let m = Matrix::of(x)?;
            let labels = labels(node, "classlabels_int64s")?;
            // XGBoost and LightGBM export a binary classifier as ONE score column
            // addressed by a single class id, with two class labels. onnxruntime calls
            // this the binary case and widens it; `class_ids` having exactly one
            // distinct value is the condition, measured on both exporters.
            let distinct: HashSet<i64> = get_attr::<[i64]>(node, "class_ids")?
                .iter()
                .copied()
                .collect();
            let binary = labels.len() == 2 && distinct.len() == 1;
            let width = if binary { 1 } else { labels.len() };
            let fold = |id: i64| {
                if binary {
                    Some(0)
                } else if id < 0 {
                    None
                } else {
                    Some(id as usize)
                }
            };
            let ensemble = Ensemble::build(node, width, "class", fold)?;
            let post = PostTransform::of(node)?;
            // A softmax over ONE column is 1.0 whatever the score is, so the widening
            // below would report [0, 1] for every row and invert the label of any model
            // whose single score is negative. onnxruntime does not answer this file
            // either: MEASURED, it ignores the transform and reports [1 - s, s], the same
            // as post_transform NONE. Refused by name for the reason PROBIT is — a wrong
            // probability is worse than a refused one. Every mainstream exporter writes
            // LOGISTIC here, or two columns.
            if binary && matches!(post, PostTransform::Softmax | PostTransform::SoftmaxZero) {
                bail!(
                    "TreeEnsembleClassifier in {} has two class labels over ONE score column \
                     with post_transform {}, which is 1.0 for every row. Re-export with \
                     post_transform LOGISTIC, which is what a binary tree ensemble means, or \
                     with one score column per class.",
                    node.name,
                    post.spelling()
                );
            }
            // The classifier has no aggregate_function: the specification fixes it to a
            // sum over trees, which is what every exporter relies on.
            let scores = ensemble.score(&m, post, Aggregate::Sum)?;
            let scores = if binary {
                widen(scores, post, ensemble.weights_all_positive)
            } else {
                scores
            };
            classified(&labels, scores, m.rows, x.device())
        }

        "TreeEnsembleRegressor" => {
            let x = numeric(0)?;
            let m = Matrix::of(x)?;
            let targets = (*get_attr_opt::<i64>(node, "n_targets")?.unwrap_or(&1)).max(1) as usize;
            let ensemble = Ensemble::build(node, targets, "target", |id| {
                (id >= 0).then_some(id as usize)
            })?;
            let out = ensemble.score(&m, PostTransform::of(node)?, Aggregate::of(node)?)?;
            Ok(vec![
                Tensor::from_vec(out, (m.rows, targets), x.device())?.into()
            ])
        }

        "SVMRegressor" => {
            let x = numeric(0)?;
            let m = Matrix::of(x)?;
            let machine = Machine::of(node, m.features)?;
            if machine.rho.len() != 1 {
                bail!(
                    "SVMRegressor in {} has {} rho values; a regressor has one",
                    node.name,
                    machine.rho.len()
                );
            }
            let post = PostTransform::of(node)?;
            let mut out = Vec::with_capacity(m.rows);
            for r in 0..m.rows {
                let kernels = machine.kernels(m.row(r));
                // `rho` is scikit-learn's `intercept_` and is ADDED — see
                // [`one_against_one`] for the measurement that settles the sign.
                let mut score = machine.rho[0];
                for (i, k) in kernels.iter().enumerate() {
                    score += machine.coefficients[i] * k;
                }
                let mut one = [score];
                post.apply(&mut one);
                out.push(one[0]);
            }
            Ok(vec![Tensor::from_vec(out, (m.rows, 1), x.device())?.into()])
        }

        // libsvm's one-against-one machine. The second output is the decision value per
        // class pair, and the label is the pair vote — which is what scikit-learn's own
        // `decision_function` is built from, so skl2onnx follows this node with the
        // `ai.onnx` arithmetic that turns votes into one-against-rest scores.
        "SVMClassifier" => {
            let x = numeric(0)?;
            let m = Matrix::of(x)?;
            let labels = labels(node, "classlabels_ints")?;
            let classes = labels.len();
            if classes < 2 {
                bail!(
                    "SVMClassifier in {} declares {classes} class labels",
                    node.name
                );
            }
            if get_attr_opt::<[f32]>(node, "prob_a")?.is_some_and(|v| !v.is_empty()) {
                bail!(
                    "SVMClassifier in {} carries prob_a/prob_b, so its scores are Platt \
                     probabilities over the class pairs rather than decision values. That \
                     calibration is a second model this evaluator does not read; re-export \
                     with probability=False, or ask for it to be added.",
                    node.name
                );
            }
            let machine = Machine::of(node, m.features)?;
            // Read BEFORE the cast to `usize`, which is why this is a loop and not a map:
            // `-1 as usize` is 2^64 - 1, and a count of 2^64 - 1 beside a count of 5 sums
            // — wrapping — to exactly the 4 support vectors the file also declares, so the
            // equality check below would PASS and `one_against_one` would then index the
            // machine off its own arrays. onnxruntime refuses at the same point:
            // "vectors_per_class[0] must be non-negative. Got -1".
            let counts = get_attr_opt::<[i64]>(node, "vectors_per_class")?.unwrap_or(&[]);
            let mut per_class: Vec<usize> = Vec::with_capacity(counts.len());
            for &n in counts {
                if n < 0 {
                    bail!(
                        "SVMClassifier in {}: vectors_per_class declares {n} vectors for a class",
                        node.name
                    );
                }
                per_class.push(n as usize);
            }
            let post = PostTransform::of(node)?;

            // With no support vectors the model is linear and `coefficients` holds one
            // weight row per class — the shape `LinearClassifier` has.
            if machine.vectors == 0 {
                let mut scores = affine(node, &m, classes)?;
                for row in scores.chunks_mut(classes) {
                    post.apply(row);
                }
                return classified(&labels, scores, m.rows, x.device());
            }
            // `checked_add`, because the sum of counts a FILE chose is not a number this
            // process picked: two counts near `usize::MAX` wrap to a small total, and a
            // small total is what the check below is looking for.
            let counted = per_class
                .iter()
                .try_fold(0usize, |total, n| total.checked_add(*n))
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "SVMClassifier in {}: vectors_per_class sums past what a machine can \
                         address",
                        node.name
                    ))
                })?;
            if counted != machine.vectors {
                bail!(
                    "SVMClassifier in {}: vectors_per_class sums to {counted} but there are {} \
                     support vectors",
                    node.name,
                    machine.vectors
                );
            }
            if per_class.len() != classes {
                bail!(
                    "SVMClassifier in {}: {} class labels but {} vector counts",
                    node.name,
                    classes,
                    per_class.len()
                );
            }
            // Two classes have ONE pair, and the node reports two columns rather than
            // that one: measured against onnxruntime, a binary iris SVC whose single
            // decision value is `D` reports `[-D, D]` — negated, and in that order. The
            // vote still decides the label, so a row with `D > 0` is class 0 even though
            // its larger reported score sits in column 1.
            let pairs = classes * (classes - 1) / 2;
            let width = if classes == 2 { 2 } else { pairs };
            let mut scores = Vec::with_capacity(m.rows * width);
            let mut best = Vec::with_capacity(m.rows);
            for r in 0..m.rows {
                let kernels = machine.kernels(m.row(r));
                let decisions = one_against_one(&machine, &kernels, &per_class)?;
                // One vote to the winner of each pair. A pair that scores exactly zero
                // goes to the higher class, which is what `> 0` gives.
                let mut votes = vec![0u32; classes];
                let mut p = 0usize;
                for i in 0..classes {
                    for j in (i + 1)..classes {
                        if decisions[p] > 0.0 {
                            votes[i] += 1;
                        } else {
                            votes[j] += 1;
                        }
                        p += 1;
                    }
                }
                let mut at = 0usize;
                for k in 1..classes {
                    if votes[k] > votes[at] {
                        at = k;
                    }
                }
                best.push(at);
                let mut reported = if classes == 2 {
                    vec![-decisions[0], decisions[0]]
                } else {
                    decisions
                };
                post.apply(&mut reported);
                scores.extend(reported);
            }
            Ok(vec![
                labels.at(&best, x.device())?,
                Tensor::from_vec(scores, (m.rows, width), x.device())?.into(),
            ])
        }

        // Select positions along the LAST axis.
        //
        // A rank-1 input reports rank 2: measured, a 3-element vector indexed by `[2]`
        // gives shape `(1, 1)`, not `(1)`. skl2onnx relies on it — a binary LinearSVC
        // export is `LinearClassifier` followed by this operator picking the positive
        // class's column.
        "ArrayFeatureExtractor" => {
            let source = input(0)?;
            let wanted = Column::of(input(1)?)?;
            let Column::Ints(wanted) = wanted else {
                bail!(
                    "ArrayFeatureExtractor in {} was given {} indices; they are int64",
                    node.name,
                    wanted.kind()
                );
            };
            let dims = match source {
                Value::Tensor(t) => t.dims().to_vec(),
                Value::Text(t) => t.dims().to_vec(),
                other => bail!(
                    "ArrayFeatureExtractor in {} cannot select from {}",
                    node.name,
                    other.kind()
                ),
            };
            let last = *dims.last().unwrap_or(&0);
            let mut out = dims.clone();
            if out.len() < 2 {
                out = vec![1, wanted.len()];
            } else {
                *out.last_mut().expect("rank at least 2") = wanted.len();
            }
            // A position is an index, not an offset: there is no counting from the end
            // here. onnxruntime refuses a negative one — "index is out of range: Y[0] (-1)
            // must be in [0, 3)" — and wrapping it around would answer a DIFFERENT
            // question than the file asked, silently.
            let at: Vec<usize> = wanted
                .iter()
                .map(|&i| {
                    if i < 0 || i as usize >= last {
                        bail!(
                            "ArrayFeatureExtractor in {} selects position {i}, which is not in \
                             [0, {last})",
                            node.name
                        )
                    }
                    Ok(i as usize)
                })
                .collect::<Result<_>>()?;
            let picked: Vec<Option<usize>> = (0..dims.iter().product::<usize>() / last.max(1))
                .flat_map(|group| at.iter().map(move |&k| Some(group * last + k)))
                .collect();
            let column = Column::of(source)?;
            let absent = match &column {
                Column::Ints(_) => Cell::Int(0),
                Column::Reals(_) => Cell::Real(0.0),
                Column::Text(_) => Cell::Text(String::new()),
            };
            Ok(vec![column
                .gather(&picked, &absent)?
                .value(out, device_of(source))?])
        }

        // Concatenate inputs along the last axis, each contributing exactly the width it
        // declares: measured, an input narrower than its declared size is padded with
        // zeros, and a wider one is truncated.
        "FeatureVectorizer" => {
            let declared = get_attr::<[i64]>(node, "inputdimensions")?;
            let mut widths: Vec<usize> = Vec::with_capacity(declared.len());
            for &n in declared {
                if n < 0 {
                    bail!(
                        "FeatureVectorizer in {}: inputdimensions declares a width of {n}",
                        node.name
                    );
                }
                widths.push(n as usize);
            }
            if widths.len() != inputs.len() {
                bail!(
                    "FeatureVectorizer in {} declares {} input widths for {} inputs",
                    node.name,
                    widths.len(),
                    inputs.len()
                );
            }
            let parts: Vec<Matrix> = inputs
                .iter()
                .map(|v| Matrix::of(v.tensor()?))
                .collect::<Result<_>>()?;
            let rows = parts.first().map(|p| p.rows).unwrap_or(0);
            for p in &parts {
                if p.rows != rows {
                    bail!(
                        "FeatureVectorizer in {} was given inputs of {rows} and {} rows",
                        node.name,
                        p.rows
                    );
                }
            }
            // The output width comes from an ATTRIBUTE, so it is a number the file chose
            // rather than one the data implies: a 200-byte node can name a gibibyte, and
            // the zero-padding above means it does not need any input to back it up. The
            // bound is orders of magnitude above the widest fitted pipeline — a
            // one-hot-encoded text feature space reaches hundreds of thousands of columns —
            // and the multiplication by the row count is checked for the same reason.
            let total: usize = widths
                .iter()
                .try_fold(0usize, |total, w| total.checked_add(*w))
                .filter(|total| *total <= MAX_WIDTH)
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "FeatureVectorizer in {}: inputdimensions asks for more than \
                         {MAX_WIDTH} output columns",
                        node.name
                    ))
                })?;
            let cells = rows.checked_mul(total).ok_or_else(|| {
                Error::Msg(format!(
                    "FeatureVectorizer in {}: {rows} rows of {total} columns is more than can be \
                     addressed",
                    node.name
                ))
            })?;
            let mut out = vec![0f32; cells];
            for r in 0..rows {
                let mut at = 0usize;
                for (p, &want) in parts.iter().zip(widths.iter()) {
                    let row = p.row(r);
                    for j in 0..want.min(row.len()) {
                        out[r * total + at + j] = row[j];
                    }
                    at += want;
                }
            }
            let device = inputs.first().map(device_of).unwrap_or(&Device::Cpu);
            Ok(vec![Tensor::from_vec(out, (rows, total), device)?.into()])
        }

        // Map elements between int64, float and string through a key/value pair of
        // attributes. Opset 1 spells the same thing as `classes_strings` with an
        // implicit integer position as the value.
        "LabelEncoder" => {
            let mapping = match Mapping::of(
                node,
                ["keys_int64s", "keys_floats", "keys_strings"],
                ["values_int64s", "values_floats", "values_strings"],
            ) {
                Ok(mapping) => mapping,
                Err(general) => match get_attr_opt_owned::<Vec<String>>(node, "classes_strings")? {
                    Some(classes) if !classes.is_empty() => {
                        let positions = (0..classes.len() as i64).collect();
                        Mapping {
                            keys: Column::Text(classes),
                            values: Column::Ints(positions),
                            absent: Cell::Int(
                                *get_attr_opt::<i64>(node, "default_int64")?.unwrap_or(&-1),
                            ),
                        }
                    }
                    _ => return Err(general),
                },
            };
            Ok(vec![mapping.apply(input(0)?)?])
        }

        // The same lookup as `LabelEncoder`, over one pair of parallel category
        // attributes rather than named key and value attributes. Which direction it runs
        // in is decided by what arrives: text in, integers out, or the reverse.
        "CategoryMapper" => {
            let strings = get_attr_opt_owned::<Vec<String>>(node, "cats_strings")?
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "CategoryMapper in {} declares no cats_strings",
                        node.name
                    ))
                })?;
            let ints = get_attr_opt::<[i64]>(node, "cats_int64s")?
                .filter(|v| !v.is_empty())
                .ok_or_else(|| {
                    Error::Msg(format!(
                        "CategoryMapper in {} declares no cats_int64s",
                        node.name
                    ))
                })?
                .to_vec();
            if strings.len() != ints.len() {
                bail!(
                    "CategoryMapper in {}: {} cats_strings and {} cats_int64s",
                    node.name,
                    strings.len(),
                    ints.len()
                );
            }
            let source = input(0)?;
            let mapping = match Column::of(source)? {
                Column::Text(_) => Mapping {
                    keys: Column::Text(strings),
                    values: Column::Ints(ints),
                    absent: Cell::Int(*get_attr_opt::<i64>(node, "default_int64")?.unwrap_or(&-1)),
                },
                _ => Mapping {
                    keys: Column::Ints(ints),
                    values: Column::Text(strings),
                    absent: Cell::Text(
                        get_attr_opt::<str>(node, "default_string")?
                            .unwrap_or("_Unused")
                            .to_string(),
                    ),
                },
            };
            Ok(vec![mapping.apply(source)?])
        }

        // One extra trailing axis, one column per category. An element in no category
        // reports an all-zero row when `zeros` is set, and is an ERROR otherwise —
        // measured: onnxruntime fails the node with "Unknown Category and zeros = 0".
        "OneHotEncoder" => {
            let categories = match get_attr_opt_owned::<Vec<String>>(node, "cats_strings")? {
                Some(v) if !v.is_empty() => Column::Text(v),
                _ => match get_attr_opt::<[i64]>(node, "cats_int64s")? {
                    Some(v) if !v.is_empty() => Column::Ints(v.to_vec()),
                    _ => bail!(
                        "OneHotEncoder in {} declares neither cats_strings nor cats_int64s",
                        node.name
                    ),
                },
            };
            let zeros = *get_attr_opt::<i64>(node, "zeros")?.unwrap_or(&1) != 0;
            let source = input(0)?;
            let column = Column::of(source)?;
            // An integer input against string categories is how skl2onnx spells a
            // numeric one-hot: the numbers name the categories by their decimal form.
            let at = match (&column, &categories) {
                (Column::Reals(v), Column::Ints(_)) => {
                    Column::Ints(v.iter().map(|x| *x as i64).collect()).positions(&categories)?
                }
                _ => column.positions(&categories)?,
            };
            let width = categories.len();
            let mut out = vec![0f32; at.len() * width];
            for (i, p) in at.iter().enumerate() {
                match p {
                    Some(k) => out[i * width + k] = 1.0,
                    None if zeros => {}
                    None => bail!(
                        "OneHotEncoder in {} was given element {i}, which is in none of its \
                         {width} categories, and zeros is not set",
                        node.name
                    ),
                }
            }
            let mut dims = match source {
                Value::Tensor(t) => t.dims().to_vec(),
                Value::Text(t) => t.dims().to_vec(),
                other => bail!(
                    "OneHotEncoder in {} cannot read {}",
                    node.name,
                    other.kind()
                ),
            };
            dims.push(width);
            Ok(vec![Tensor::from_vec(out, dims, device_of(source))?.into()])
        }

        // The scores, under the labels they belong to.
        //
        // ONNX types this `seq(map(K, tensor(float)))`, and it sits on the probability
        // output of EVERY default scikit-learn, XGBoost and LightGBM classifier export —
        // so a reader without it cannot load a model anyone actually has. Every map in
        // the sequence carries the same keys by construction, which is what [`Table`]
        // says and a vector of dictionaries would not.
        "ZipMap" => {
            let scores = numeric(0)?;
            let labels = labels(node, "classlabels_int64s")?;
            let scores = if scores.rank() == 1 {
                scores.reshape((1, scores.dim(0)?))?
            } else {
                scores.clone()
            };
            Ok(vec![Table::new(labels, scores)?.into()])
        }

        // The two operators of this domain that take a MAP as input. Nothing here puts
        // one on an edge — `ZipMap` is terminal in every export measured — so they are
        // named rather than half-served.
        "CastMap" | "DictVectorizer" => bail!(
            "ai.onnx.ml {} in {} reads a map from a graph edge. Nothing in this evaluator \
             produces one: ZipMap is terminal in every export measured. Re-export without \
             the map, or ask for map-valued edges to be added.",
            node.op_type,
            node.name
        ),

        // The opset-5 unification of TreeEnsembleClassifier and TreeEnsembleRegressor.
        // A different attribute schema (`nodes_splits`, `leaf_weights`, `tree_roots`,
        // `membership_values`) rather than a rename, and no exporter emits it by
        // default yet, so it is named rather than mistaken for its predecessors.
        "TreeEnsemble" => bail!(
            "ai.onnx.ml TreeEnsemble in {} is the opset-5 operator, whose attributes are a \
             different schema from TreeEnsembleClassifier/TreeEnsembleRegressor. Export at \
             ai.onnx.ml opset 3, which every exporter still targets by default.",
            node.name
        ),

        op => bail!("unsupported ai.onnx.ml op_type {op} for op {node:?}"),
    }
}

/// How many features a row of this tensor has — its last axis, or its length when it is
/// a single row.
fn features_of(x: &Tensor) -> Result<usize> {
    match x.rank() {
        1 => x.dim(0),
        2 => x.dim(1),
        rank => bail!("expected a rank-1 or rank-2 feature matrix, got rank {rank}"),
    }
}
