//! `ai.onnx.ml` — the classical operator domain, checked against real exports.
//!
//! Every `.onnx` in `tests/ml/` is either genuine exporter output for a fitted estimator
//! or a hand-built single-operator graph, and every expected number is what a TRUSTED
//! IMPLEMENTATION answered for the same input rows:
//!
//!   * for a real export, the fitted library's OWN prediction — scikit-learn's
//!     `predict_proba`, XGBoost's, LightGBM's. So these are not assertions about what
//!     this evaluator does; they are assertions that it agrees with the library on the
//!     library's own model.
//!   * for a hand-built operator, onnxruntime's answer. These pin the behaviour the
//!     specification leaves open and a fitted estimator never reaches.
//!
//! The exports are the DEFAULT ones. No `zipmap=False`, no options at all — a reader
//! that needs the model re-exported to be readable has not solved the problem, because
//! the models people have were exported before anyone asked.
//!
//! `tests/ml/generate.py` writes the fixtures and `tests/ml/oracle.rs` in ONE pass,
//! because a fixture and its expectation must come from one fit: generated separately,
//! an unseeded forest fixture gets compared against a *different* unseeded forest's
//! probabilities, which is how this file was first wrong. The generator also refuses to
//! record a fixture whose export does not already reproduce its own library.

use hanzo_ml::{Device, Result, Tensor};
use hanzo_onnx::{Domain, Key, Labels, Table, Text, Value};
use std::collections::HashMap;

include!("ml/oracle.rs");

/// The `ai.onnx.ml` operators this evaluator claims. Every one must be reached by a
/// fixture, or the claim is untested — see [`every_claimed_operator_is_exercised`].
const CLAIMED: &[&str] = &[
    "TreeEnsembleClassifier",
    "TreeEnsembleRegressor",
    "LinearClassifier",
    "LinearRegressor",
    "SVMClassifier",
    "SVMRegressor",
    "ZipMap",
    "Normalizer",
    "Scaler",
    "Binarizer",
    "Imputer",
    "LabelEncoder",
    "CategoryMapper",
    "OneHotEncoder",
    "ArrayFeatureExtractor",
    "FeatureVectorizer",
];

fn path(name: &str) -> String {
    format!("{}/tests/ml/{name}.onnx", env!("CARGO_MANIFEST_DIR"))
}

impl Data {
    /// This value, as something to feed a graph.
    fn feed(&self) -> Result<Value> {
        Ok(match self {
            Self::Reals { dims, values } => {
                Tensor::from_slice(values, dims.to_vec(), &Device::Cpu)?.into()
            }
            Self::Ints { dims, values } => {
                Tensor::from_slice(values, dims.to_vec(), &Device::Cpu)?.into()
            }
            Self::Text { dims, values } => Text::new(
                values.iter().map(|s| s.to_string()).collect(),
                dims.to_vec(),
            )?
            .into(),
            Self::Table { .. } => {
                panic!("a table is an output, not an input — no operator here reads one")
            }
        })
    }
}

impl Names {
    fn matches(&self, labels: &Labels) -> Option<String> {
        let same = match (self, labels) {
            (Self::Ints(want), Labels::Ints(got)) => want == &got.as_slice(),
            (Self::Text(want), Labels::Text(got)) => {
                want.len() == got.len() && want.iter().zip(got).all(|(w, g)| w == g)
            }
            _ => false,
        };
        (!same).then(|| format!("labels {labels:?}, wanted {self:?}"))
    }

    /// The keys these names would put on a table row, so the dictionary view is checked
    /// against the same source as the tensor view.
    fn keys(&self) -> Vec<Key> {
        match self {
            Self::Ints(v) => v.iter().copied().map(Key::Int).collect(),
            Self::Text(v) => v.iter().map(|s| Key::Text(s.to_string())).collect(),
        }
    }
}

impl std::fmt::Debug for Names {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Ints(v) => write!(f, "{v:?}"),
            Self::Text(v) => write!(f, "{v:?}"),
        }
    }
}

/// Agreement to f32 precision, relative on the scale of the expected value.
///
/// Not bit equality: the libraries compute in f64 and the graph runs in f32 throughout,
/// so a probability of 1e-6 can differ in its fourth significant digit while being right
/// to 5e-8 in absolute terms. The absolute floor is what makes that pass and a genuinely
/// wrong number fail.
fn close(got: f32, want: f32, tolerance: f32) -> bool {
    (got - want).abs() <= tolerance + 1e-4 * want.abs()
}

fn reals(got: &[f32], want: &[f32], tolerance: f32, what: &str) -> Option<String> {
    if got.len() != want.len() {
        return Some(format!(
            "{what}: {} values, wanted {}",
            got.len(),
            want.len()
        ));
    }
    for (i, (&g, &w)) in got.iter().zip(want).enumerate() {
        if !close(g, w, tolerance) {
            return Some(format!("{what}[{i}]: got {g:e}, wanted {w:e}"));
        }
    }
    None
}

/// What this value should be, or the first way it is not.
fn differs(got: &Value, want: &Data, tolerance: f32) -> Option<String> {
    match (got, want) {
        (Value::Tensor(t), Data::Reals { dims, values }) => {
            if t.dims() != *dims {
                return Some(format!("shape {:?}, wanted {dims:?}", t.dims()));
            }
            let flat = t
                .to_dtype(hanzo_ml::DType::F32)
                .and_then(|t| t.flatten_all())
                .and_then(|t| t.to_vec1::<f32>())
                .map_err(|e| e.to_string());
            match flat {
                Err(e) => Some(e),
                Ok(flat) => reals(&flat, values, tolerance, "value"),
            }
        }
        (Value::Tensor(t), Data::Ints { dims, values }) => {
            if t.dims() != *dims {
                return Some(format!("shape {:?}, wanted {dims:?}", t.dims()));
            }
            match t.flatten_all().and_then(|t| t.to_vec1::<i64>()) {
                Err(e) => Some(e.to_string()),
                Ok(flat) if flat != *values => Some(format!("got {flat:?}, wanted {values:?}")),
                Ok(_) => None,
            }
        }
        (Value::Text(t), Data::Text { dims, values }) => {
            if t.dims() != *dims {
                return Some(format!("shape {:?}, wanted {dims:?}", t.dims()));
            }
            let same = t.elements().len() == values.len()
                && t.elements().iter().zip(*values).all(|(g, w)| g == w);
            (!same).then(|| format!("got {:?}, wanted {values:?}", t.elements()))
        }
        (Value::Table(t), Data::Table { keys, rows, scores }) => {
            table(t, keys, *rows, scores, tolerance)
        }
        (got, want) => Some(format!(
            "got {}, wanted {}",
            got.kind(),
            match want {
                Data::Reals { .. } => "a float tensor",
                Data::Ints { .. } => "an int64 tensor",
                Data::Text { .. } => "a tensor(string)",
                Data::Table { .. } => "a sequence of maps",
            }
        )),
    }
}

/// A table is checked twice: as the score matrix it holds, and as the per-row dictionary
/// a caller reading `seq(map(...))` expects. Both views must agree with the oracle, since
/// a table that stored the right numbers under the wrong keys would pass only the first.
fn table(got: &Table, keys: &Names, rows: usize, scores: &[f32], tolerance: f32) -> Option<String> {
    if let Some(why) = keys.matches(got.labels()) {
        return Some(why);
    }
    if got.rows() != rows {
        return Some(format!("{} rows, wanted {rows}", got.rows()));
    }
    let flat = got
        .scores()
        .flatten_all()
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| e.to_string());
    match flat {
        Err(e) => return Some(e),
        Ok(flat) => {
            if let Some(why) = reals(&flat, scores, tolerance, "score") {
                return Some(why);
            }
        }
    }
    let want = keys.keys();
    for r in 0..rows {
        match got.row(r) {
            Err(e) => return Some(e.to_string()),
            Ok(pairs) => {
                if pairs.len() != want.len() {
                    return Some(format!("row {r} has {} entries", pairs.len()));
                }
                for (i, (key, score)) in pairs.iter().enumerate() {
                    if key != &want[i] {
                        return Some(format!("row {r} key {i} is {key:?}"));
                    }
                    let expected = scores[r * want.len() + i];
                    if !close(*score, expected, tolerance) {
                        return Some(format!(
                            "row {r} entry {key:?}: got {score:e}, wanted {expected:e}"
                        ));
                    }
                }
            }
        }
    }
    None
}

/// Run one fixture and report every way it disagrees with its oracle.
fn check(fixture: &Fixture) -> Vec<String> {
    let model = match hanzo_onnx::read_file(path(fixture.name)) {
        Ok(model) => model,
        Err(e) => return vec![format!("{}: cannot read: {e}", fixture.name)],
    };
    let mut inputs: HashMap<String, Value> = HashMap::new();
    for (name, data) in fixture.inputs {
        match data.feed() {
            Ok(value) => {
                inputs.insert(name.to_string(), value);
            }
            Err(e) => return vec![format!("{}: cannot build input {name}: {e}", fixture.name)],
        }
    }
    let out = match hanzo_onnx::simple_eval(&model, inputs) {
        Ok(out) => out,
        Err(e) => return vec![format!("{} ({}): {e}", fixture.name, fixture.about)],
    };
    let mut failures = Vec::new();
    for (name, want) in fixture.expect {
        match out.get(*name) {
            None => failures.push(format!(
                "{}: no output {name}; the graph produced {:?}",
                fixture.name,
                out.keys().collect::<Vec<_>>()
            )),
            Some(got) => {
                if let Some(why) = differs(got, want, fixture.tolerance) {
                    failures.push(format!(
                        "{}/{name}: {why}\n      about: {}\n      oracle: {}",
                        fixture.name, fixture.about, fixture.oracle
                    ));
                }
            }
        }
    }
    failures
}

#[test]
fn every_fixture_agrees_with_its_oracle() {
    let mut failures = Vec::new();
    for fixture in FIXTURES {
        failures.extend(check(fixture));
    }
    assert!(
        failures.is_empty(),
        "{} of {} fixtures disagree:\n\n  - {}",
        failures.len(),
        FIXTURES.len(),
        failures.join("\n  - ")
    );
}

/// Being able to run an operator and CLAIMING to run it are different things, and the
/// second is what a reader of this crate acts on. Every operator in [`CLAIMED`] must be
/// reached by at least one fixture above, so the claim is not a comment.
#[test]
fn every_claimed_operator_is_exercised() -> Result<()> {
    let mut seen: HashMap<&str, usize> = HashMap::new();
    for fixture in FIXTURES {
        let model = hanzo_onnx::read_file(path(fixture.name))?;
        let graph = model.graph.as_ref().expect("a fixture has a graph");
        for node in &graph.node {
            if Domain::of(node)? == Domain::Ml {
                for claimed in CLAIMED {
                    if node.op_type == *claimed {
                        *seen.entry(claimed).or_default() += 1;
                    }
                }
            }
        }
    }
    let missing: Vec<&&str> = CLAIMED
        .iter()
        .filter(|op| !seen.contains_key(**op))
        .collect();
    assert!(
        missing.is_empty(),
        "claimed but never exercised by a fixture: {missing:?}"
    );
    Ok(())
}

/// The measurement that made the classical domain readable at all: a node's operator is
/// named by `(domain, op_type)`, not by `op_type`. Every classifier fixture has an
/// `ai.onnx.ml` node AND an `ai.onnx` one (`Cast` or `Identity` on the label), so a
/// table keyed on `op_type` alone could not serve both.
#[test]
fn a_real_export_mixes_both_domains_on_one_graph() -> Result<()> {
    let mut mixed = 0usize;
    for fixture in FIXTURES {
        let model = hanzo_onnx::read_file(path(fixture.name))?;
        let graph = model.graph.as_ref().expect("a fixture has a graph");
        let mut domains = std::collections::HashSet::new();
        for node in &graph.node {
            domains.insert(Domain::of(node)?);
        }
        if domains.len() == 2 {
            mixed += 1;
        }
    }
    assert!(
        mixed >= 8,
        "only {mixed} fixtures mix ai.onnx with ai.onnx.ml; the domain split is not being \
         exercised"
    );
    Ok(())
}

#[test]
fn the_default_domain_is_the_standard_one() -> Result<()> {
    use hanzo_onnx::onnx;
    // Exporters spell ai.onnx as the empty string; both must mean the same set.
    for domain in ["", "ai.onnx"] {
        let node = onnx::NodeProto {
            domain: domain.to_string(),
            op_type: "Add".to_string(),
            ..Default::default()
        };
        assert_eq!(Domain::of(&node)?, Domain::Standard);
    }
    let node = onnx::NodeProto {
        domain: "ai.onnx.ml".to_string(),
        op_type: "Scaler".to_string(),
        ..Default::default()
    };
    assert_eq!(Domain::of(&node)?, Domain::Ml);
    Ok(())
}

#[test]
fn an_unknown_domain_is_refused_by_name() {
    use hanzo_onnx::onnx;
    let node = onnx::NodeProto {
        domain: "com.microsoft".to_string(),
        op_type: "FusedMatMul".to_string(),
        name: "n0".to_string(),
        ..Default::default()
    };
    let err =
        Domain::of(&node).expect_err("an unknown domain must not be served by the ai.onnx table");
    let msg = err.to_string();
    assert!(msg.contains("com.microsoft"), "{msg}");
    assert!(msg.contains("FusedMatMul"), "{msg}");
}

/// A map-valued edge and the opset-5 `TreeEnsemble` are the two things this domain can
/// ask for that are genuinely absent. Both must fail by NAME, with the fix in the
/// message — a generic "unsupported op_type" is what sent the first reader of this crate
/// looking for a bug in the protobuf.
#[test]
fn what_is_absent_says_what_it_is_and_what_to_do() -> Result<()> {
    use hanzo_onnx::onnx;
    let model = |op: &str| -> onnx::ModelProto {
        onnx::ModelProto {
            graph: Some(onnx::GraphProto {
                node: vec![onnx::NodeProto {
                    domain: "ai.onnx.ml".to_string(),
                    op_type: op.to_string(),
                    name: "n0".to_string(),
                    input: vec!["x".to_string()],
                    output: vec!["y".to_string()],
                    ..Default::default()
                }],
                output: vec![onnx::ValueInfoProto {
                    name: "y".to_string(),
                    ..Default::default()
                }],
                ..Default::default()
            }),
            ..Default::default()
        }
    };
    let x = Tensor::from_slice(&[1f32, 2.0], (1, 2), &Device::Cpu)?;
    for (op, expected) in [
        ("CastMap", "ZipMap is terminal"),
        ("DictVectorizer", "ZipMap is terminal"),
        ("TreeEnsemble", "opset-5"),
    ] {
        let inputs = HashMap::from([("x".to_string(), x.clone())]);
        let err = hanzo_onnx::simple_eval(&model(op), inputs)
            .expect_err("an absent operator must not answer");
        let msg = err.to_string();
        assert!(msg.contains(op), "{op}: {msg}");
        assert!(msg.contains(expected), "{op}: {msg}");
    }
    Ok(())
}

/// `hanzo_ml::DType` has no string member, so a `tensor(string)` cannot be a `Tensor` —
/// which is why [`Value`] is a sum and not an alias. A label list that is text stays
/// text all the way to the caller, rather than being renumbered into integers nobody
/// asked for.
#[test]
fn a_text_label_survives_as_text() -> Result<()> {
    let model = hanzo_onnx::read_file(path("text_clf"))?;
    let graph = model.graph.as_ref().expect("a fixture has a graph");
    let rows = Tensor::from_slice(
        &[5.1f32, 3.5, 1.4, 0.2, 6.9, 3.2, 5.7, 2.3],
        (2, 4),
        &Device::Cpu,
    )?;
    let out =
        hanzo_onnx::simple_eval(&model, HashMap::from([(graph.input[0].name.clone(), rows)]))?;
    let label = out["output_label"].text()?;
    assert_eq!(
        label.elements(),
        &["setosa".to_string(), "virginica".to_string()]
    );
    let scores = out["output_probability"].table()?;
    assert!(
        matches!(scores.labels(), Labels::Text(v) if v[0] == "setosa"),
        "a text-labelled classifier's table must be keyed by text"
    );
    // And the tensor plane refuses it rather than reinterpreting the bytes.
    assert!(out["output_label"].tensor().is_err());
    Ok(())
}
