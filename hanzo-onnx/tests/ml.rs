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
use hanzo_onnx::onnx::{attribute_proto::AttributeType, AttributeProto};
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

/// One attribute, in each of the three kinds a classical operator reads, so a hand-built
/// node is written as the attribute list it is.
fn ints(name: &str, values: &[i64]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Ints.into(),
        ints: values.to_vec(),
        ..Default::default()
    }
}

fn floats(name: &str, values: &[f32]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Floats.into(),
        floats: values.to_vec(),
        ..Default::default()
    }
}

fn strings(name: &str, values: &[&str]) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Strings.into(),
        strings: values.iter().map(|s| s.as_bytes().to_vec()).collect(),
        ..Default::default()
    }
}

fn word(name: &str, value: &str) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::String.into(),
        s: value.as_bytes().to_vec(),
        ..Default::default()
    }
}

fn number(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Float.into(),
        f: value,
        ..Default::default()
    }
}

fn count(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        r#type: AttributeType::Int.into(),
        i: value,
        ..Default::default()
    }
}

/// A one-node `ai.onnx.ml` graph over the edges named.
fn graph(
    op: &str,
    inputs: &[&str],
    outputs: &[&str],
    attribute: Vec<AttributeProto>,
) -> hanzo_onnx::onnx::ModelProto {
    use hanzo_onnx::onnx;
    onnx::ModelProto {
        graph: Some(onnx::GraphProto {
            node: vec![onnx::NodeProto {
                domain: "ai.onnx.ml".to_string(),
                op_type: op.to_string(),
                name: "n0".to_string(),
                input: inputs.iter().map(|n| n.to_string()).collect(),
                output: outputs.iter().map(|n| n.to_string()).collect(),
                attribute,
                ..Default::default()
            }],
            output: outputs
                .iter()
                .map(|name| onnx::ValueInfoProto {
                    name: name.to_string(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }),
        ..Default::default()
    }
}

/// A one-node `ai.onnx.ml` graph reading `x` and reporting a classifier's two outputs.
fn classifier(op: &str, attribute: Vec<AttributeProto>) -> hanzo_onnx::onnx::ModelProto {
    graph(op, &["x"], &["label", "scores"], attribute)
}

/// Run a one-input graph over one feature matrix.
fn feed(model: &hanzo_onnx::onnx::ModelProto, x: Tensor) -> Result<HashMap<String, Value>> {
    hanzo_onnx::simple_eval(model, HashMap::from([("x".to_string(), x)]))
}

/// The error a graph refuses this input with, or a panic naming what it answered instead.
fn refused(model: &hanzo_onnx::onnx::ModelProto, x: Tensor) -> String {
    match feed(model, x) {
        Ok(out) => panic!(
            "a malformed model must be refused, not answered with {:?}",
            out.keys().collect::<Vec<_>>()
        ),
        Err(e) => e.to_string(),
    }
}

/// Every attribute a stump-shaped `TreeEnsembleClassifier` needs, splitting feature 0 at
/// 0.5 with `weights` on its two leaves under ONE class id — the shape XGBoost and
/// LightGBM export a two-class model in.
fn binary_stump(weights: [f32; 2]) -> Vec<AttributeProto> {
    vec![
        ints("nodes_treeids", &[0, 0, 0]),
        ints("nodes_nodeids", &[0, 1, 2]),
        ints("nodes_featureids", &[0, 0, 0]),
        floats("nodes_values", &[0.5, 0.0, 0.0]),
        ints("nodes_truenodeids", &[1, 0, 0]),
        ints("nodes_falsenodeids", &[2, 0, 0]),
        strings("nodes_modes", &["BRANCH_LEQ", "LEAF", "LEAF"]),
        ints("class_treeids", &[0, 0]),
        ints("class_nodeids", &[1, 2]),
        ints("class_ids", &[0, 0]),
        floats("class_weights", &weights),
        ints("classlabels_int64s", &[0, 1]),
    ]
}

/// A classifier declares its classes in `classlabels_int64s`/`classlabels_ints` or in
/// `classlabels_strings`, and EVERY width downstream is that count: the score matrix has
/// one column per class, and `SVMClassifier`'s decision values one per class PAIR,
/// `n(n - 1)/2`. A count of zero is not a small model, it is an arithmetic impossibility
/// — `chunks(0)` panics, and `0 * (0 - 1) / 2` underflows — so a file that declares none
/// must be refused where the count is READ, which is the one place all four readers
/// share. onnxruntime refuses the same models.
///
/// Every node below is otherwise COMPLETE: each array a fitted classifier carries is
/// present and mutually consistent, so the label count is the only thing that can stop
/// it and the refusal under test is the one being reached. Take the check out of
/// `labels()` and this test does not fail, it PANICS — which is the whole point of it.
#[test]
fn a_classifier_with_no_labels_is_an_error_not_a_panic() -> Result<()> {
    // A single leaf carrying no weight: the whole ensemble, valid, with a width taken
    // from the class count alone.
    let tree = || {
        vec![
            ints("nodes_treeids", &[0]),
            ints("nodes_nodeids", &[0]),
            ints("nodes_featureids", &[0]),
            floats("nodes_values", &[0.0]),
            ints("nodes_truenodeids", &[0]),
            ints("nodes_falsenodeids", &[0]),
            strings("nodes_modes", &["LEAF"]),
            ints("class_ids", &[]),
            ints("class_treeids", &[]),
            ints("class_nodeids", &[]),
            floats("class_weights", &[]),
        ]
    };
    // One weight row per class, of which there are none.
    let linear = || vec![floats("coefficients", &[])];
    let x = Tensor::from_slice(&[1f32, 2.0], (1, 2), &Device::Cpu)?;

    for (op, spelling, complete) in [
        (
            "TreeEnsembleClassifier",
            "classlabels_int64s",
            tree() as Vec<AttributeProto>,
        ),
        ("LinearClassifier", "classlabels_ints", linear()),
        ("SVMClassifier", "classlabels_ints", linear()),
        ("ZipMap", "classlabels_int64s", vec![]),
    ] {
        // Absent, and present but empty. An exporter writes the second when a fit went
        // wrong; a hand-edited file carries either.
        for labels in [
            vec![],
            vec![ints(spelling, &[]), strings("classlabels_strings", &[])],
        ] {
            let mut attribute = complete.clone();
            attribute.extend(labels);
            let inputs = HashMap::from([("x".to_string(), x.clone())]);
            let err = hanzo_onnx::simple_eval(&classifier(op, attribute), inputs)
                .expect_err("a classifier with no classes must not be run");
            let msg = err.to_string();
            for named in [op, spelling, "classlabels_strings", "no class labels"] {
                assert!(msg.contains(named), "{op}: {msg}");
            }
        }
    }
    Ok(())
}

/// `LinearClassifier` and `SVMClassifier` refuse ONE class as well, because the machine
/// under each of them is a comparison BETWEEN classes — a widened complement, a
/// one-against-one pair vote — and neither is defined for a single one.
///
/// These two checks need a case of their own: [`labels()`] refuses zero before they are
/// reached, so the count they see is never below one from any other test, and both could
/// be deleted with nothing going red. Each node here carries the two coefficients a
/// one-class model would need, so the class count is again the only fault.
#[test]
fn a_classifier_with_one_class_is_refused_by_the_two_that_compare() -> Result<()> {
    let x = Tensor::from_slice(&[1f32, 2.0], (1, 2), &Device::Cpu)?;
    for op in ["LinearClassifier", "SVMClassifier"] {
        let attribute = vec![
            floats("coefficients", &[1.0, 1.0]),
            ints("classlabels_ints", &[7]),
        ];
        let inputs = HashMap::from([("x".to_string(), x.clone())]);
        let err = hanzo_onnx::simple_eval(&classifier(op, attribute), inputs)
            .expect_err("a classifier with one class must not be run");
        let msg = err.to_string();
        for named in [op, "1 class labels"] {
            assert!(msg.contains(named), "{op}: {msg}");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------------
// What a hostile file can ask for
// ---------------------------------------------------------------------------------
//
// Every model below is one a fitted estimator cannot produce and an editor can: an array
// one entry short of the length another array implies, a count written negative, a width
// that names more memory than the file. They are here because "load the .onnx you already
// have" means the bytes are not trusted, and because each of these USED to reach an
// indexing panic or a silent misread rather than an error. onnxruntime refuses the same
// files, and its message is quoted where it decided the shape of ours.

/// A machine reads one dual coefficient per support vector — the regressor one each, the
/// classifier one per decision plane — so ten vectors and one coefficient is not a small
/// model, it is a read past the end of the array. Measured: onnxruntime refuses with
/// "coefficients size (1) must be >= n_supports (10)".
#[test]
fn an_svm_with_fewer_coefficients_than_vectors_is_refused() -> Result<()> {
    let support: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let model = graph(
        "SVMRegressor",
        &["x"],
        &["y"],
        vec![
            word("kernel_type", "LINEAR"),
            floats("kernel_params", &[0.0, 0.0, 0.0]),
            floats("support_vectors", &support),
            floats("coefficients", &[1.0]),
            floats("rho", &[0.0]),
        ],
    );
    let msg = refused(
        &model,
        Tensor::from_slice(&[1f32, 2.0], (1, 2), &Device::Cpu)?,
    );
    for named in ["SVMRegressor", "1 coefficients", "10 support vectors"] {
        assert!(msg.contains(named), "{msg}");
    }
    Ok(())
}

/// A negative count is refused where it is READ, before the cast that would hide it.
///
/// `vectors_per_class = [-1, 5]` casts to `[2^64 - 1, 5]`, whose WRAPPING sum is 4 — which
/// is exactly the number of support vectors the same file declares, so the count check
/// would agree with itself and the pair scoring would then index the machine off its own
/// arrays. `checked_add` is there for the same reason from the other side: three counts of
/// `i64::MAX` also sum to a small number. onnxruntime refuses at the same point:
/// "vectors_per_class[0] must be non-negative. Got -1".
#[test]
fn a_negative_vector_count_is_refused_before_the_cast_hides_it() -> Result<()> {
    let support: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let svm = |labels: &[i64], counts: &[i64]| {
        graph(
            "SVMClassifier",
            &["x"],
            &["label", "scores"],
            vec![
                word("kernel_type", "LINEAR"),
                floats("kernel_params", &[0.0, 0.0, 0.0]),
                floats("support_vectors", &support),
                floats("coefficients", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
                floats("rho", &[0.0, 0.0, 0.0]),
                ints("classlabels_ints", labels),
                ints("vectors_per_class", counts),
            ],
        )
    };
    let x = || Tensor::from_slice(&[1f32, 2.0], (1, 2), &Device::Cpu);
    let msg = refused(&svm(&[0, 1], &[-1, 5]), x()?);
    for named in ["SVMClassifier", "vectors_per_class", "-1"] {
        assert!(msg.contains(named), "{msg}");
    }
    let msg = refused(&svm(&[0, 1, 2], &[i64::MAX, i64::MAX, i64::MAX]), x()?);
    assert!(msg.contains("sums past"), "{msg}");
    Ok(())
}

/// An absent float lookup answers -0.0, which is the specification's default and
/// onnxruntime's — MEASURED, bits 0x80000000.
///
/// NaN would not be a default but a poison: it is equal to nothing, so it survives every
/// comparison downstream, and one missing optional attribute would turn a Scaler and a
/// classifier behind it into an argmax over NaNs, which answers class 0 for every row.
#[test]
fn an_absent_float_lookup_answers_negative_zero() -> Result<()> {
    let model = graph(
        "LabelEncoder",
        &["x"],
        &["y"],
        vec![
            floats("keys_floats", &[1.5]),
            floats("values_floats", &[10.0]),
        ],
    );
    let out = feed(&model, Tensor::from_slice(&[9f32], 1, &Device::Cpu)?)?;
    let got = out["y"].tensor()?.to_vec1::<f32>()?[0];
    assert_eq!(
        got.to_bits(),
        (-0.0f32).to_bits(),
        "an absent key answered {got:e}, not -0.0"
    );
    Ok(())
}

/// `Imputer` and `Binarizer` are typed `T -> T`, so an int64 column comes back int64.
///
/// The float plane cannot stand in for the integer one: `9007199254740993` — 2^53 + 1 —
/// is `9007199000000000` as f32, so a key routed through it becomes a different key. The
/// `impute_int` fixture pins the values against onnxruntime; this pins the TYPE, and pins
/// `Binarizer`, for which onnxruntime carries no int64 kernel at all ("Could not find an
/// implementation for Binarizer(1)") and the specification is therefore the only oracle.
#[test]
fn an_int64_column_keeps_its_type() -> Result<()> {
    let column = || Tensor::from_slice(&[0i64, 2, 9007199254740993], (1, 3), &Device::Cpu);
    let binarizer = graph("Binarizer", &["x"], &["y"], vec![number("threshold", 1.5)]);
    let out = feed(&binarizer, column()?)?;
    let y = out["y"].tensor()?;
    assert_eq!(
        y.dtype(),
        hanzo_ml::DType::I64,
        "Binarizer widened the type"
    );
    assert_eq!(y.flatten_all()?.to_vec1::<i64>()?, vec![0i64, 1, 1]);

    let imputer = graph(
        "Imputer",
        &["x"],
        &["y"],
        vec![
            ints("imputed_value_int64s", &[7]),
            count("replaced_value_int64", 0),
        ],
    );
    let out = feed(&imputer, column()?)?;
    let y = out["y"].tensor()?;
    assert_eq!(y.dtype(), hanzo_ml::DType::I64, "Imputer widened the type");
    assert_eq!(
        y.flatten_all()?.to_vec1::<i64>()?,
        vec![7i64, 2, 9007199254740993],
        "the imputed column did not survive the round trip"
    );
    Ok(())
}

/// ONNX gives `Imputer` one attribute set per plane, and the INPUT chooses which runs. A
/// node whose values are of the other plane is refused rather than converted — measured,
/// onnxruntime refuses both directions with "Empty value of imputed values".
#[test]
fn an_imputer_will_not_cross_the_two_planes_onnx_gives_it() -> Result<()> {
    let imputer = |attribute: Vec<AttributeProto>| graph("Imputer", &["x"], &["y"], attribute);
    let msg = refused(
        &imputer(vec![floats("imputed_value_floats", &[7.0])]),
        Tensor::from_slice(&[0i64, 5], (1, 2), &Device::Cpu)?,
    );
    for named in [
        "tensor(int64)",
        "imputed_value_int64s",
        "imputed_value_floats",
    ] {
        assert!(msg.contains(named), "{msg}");
    }
    let msg = refused(
        &imputer(vec![ints("imputed_value_int64s", &[7])]),
        Tensor::from_slice(&[0f32, 5.0], (1, 2), &Device::Cpu)?,
    );
    assert!(msg.contains("imputed_value_floats"), "{msg}");
    Ok(())
}

/// A softmax over ONE column is 1.0 whatever the score, so a two-class ensemble that
/// declares one is refused BY NAME rather than answered with [0, 1] for every row.
///
/// onnxruntime does not answer this file either — measured, it ignores the transform and
/// reports [1 - s, s], the same as post_transform NONE. Neither reading is the model's, so
/// this follows PROBIT: a wrong probability is worse than a refused one.
#[test]
fn a_softmax_over_one_score_column_is_refused_by_name() -> Result<()> {
    for spelling in ["SOFTMAX", "SOFTMAX_ZERO"] {
        let mut attribute = binary_stump([0.25, 0.75]);
        attribute.push(word("post_transform", spelling));
        let msg = refused(
            &classifier("TreeEnsembleClassifier", attribute),
            Tensor::from_slice(&[0f32], (1, 1), &Device::Cpu)?,
        );
        for named in ["TreeEnsembleClassifier", spelling, "LOGISTIC"] {
            assert!(msg.contains(named), "{spelling}: {msg}");
        }
    }
    // The same node with the transform every binary exporter actually writes is served.
    let mut attribute = binary_stump([0.25, 0.75]);
    attribute.push(word("post_transform", "LOGISTIC"));
    let out = feed(
        &classifier("TreeEnsembleClassifier", attribute),
        Tensor::from_slice(&[0f32], (1, 1), &Device::Cpu)?,
    )?;
    assert_eq!(out["label"].tensor()?.to_vec1::<i64>()?, vec![1i64]);
    Ok(())
}

/// A position is an index, not an offset from the end: `ArrayFeatureExtractor` refuses a
/// negative one instead of wrapping it around and answering a different question.
/// Measured: onnxruntime says "index is out of range: Y[0] (-1) must be in [0, 3)".
#[test]
fn a_selected_position_is_not_an_offset_from_the_end() -> Result<()> {
    let model = graph("ArrayFeatureExtractor", &["x", "at"], &["y"], vec![]);
    for (position, named) in [(-1i64, "-1"), (3, "3")] {
        let inputs = HashMap::from([
            (
                "x".to_string(),
                Value::from(Tensor::from_slice(
                    &[0f32, 1.0, 2.0, 3.0, 4.0, 5.0],
                    (2, 3),
                    &Device::Cpu,
                )?),
            ),
            (
                "at".to_string(),
                Value::from(Tensor::from_slice(&[position], 1, &Device::Cpu)?),
            ),
        ]);
        let err = hanzo_onnx::simple_eval(&model, inputs)
            .expect_err("a position outside the axis must not be served");
        let msg = err.to_string();
        for wanted in ["ArrayFeatureExtractor", named, "[0, 3)"] {
            assert!(msg.contains(wanted), "{position}: {msg}");
        }
    }
    Ok(())
}

/// `FeatureVectorizer` pads a narrow input out to the width it declares, so its output
/// size is a number the FILE chose and no input has to back it up: a 200-byte node can
/// name a gibibyte. The bound is far above the widest fitted pipeline and far below an
/// allocation that matters.
#[test]
fn a_declared_width_cannot_name_more_memory_than_the_file() -> Result<()> {
    let vectorizer = |widths: &[i64]| {
        graph(
            "FeatureVectorizer",
            &["x"],
            &["y"],
            vec![ints("inputdimensions", widths)],
        )
    };
    let x = || Tensor::from_slice(&[1f32], (1, 1), &Device::Cpu);
    let msg = refused(&vectorizer(&[1 << 30]), x()?);
    for named in ["FeatureVectorizer", "output columns"] {
        assert!(msg.contains(named), "{msg}");
    }
    let msg = refused(&vectorizer(&[-5]), x()?);
    assert!(msg.contains("width of -5"), "{msg}");
    // The width a real pipeline asks for is still served, padded with zeros.
    let out = feed(&vectorizer(&[3]), x()?)?;
    assert_eq!(out["y"].tensor()?.dims(), &[1, 3]);
    Ok(())
}
