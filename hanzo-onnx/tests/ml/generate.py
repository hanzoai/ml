#!/usr/bin/env python3
"""Write the `ai.onnx.ml` fixtures, and the expectations `tests/ml.rs` asserts.

Two kinds of fixture, one table:

  * REAL EXPORTS of fitted estimators, whose expectation is the estimator's OWN
    prediction. So the Rust test checks agreement with scikit-learn, XGBoost and
    LightGBM on their own models — not agreement with itself. The exports are the
    DEFAULT ones: no `zipmap=False`, no options at all, because "loads the file you
    already have" is the whole point.

  * HAND-BUILT single-operator graphs, whose expectation is onnxruntime's answer. These
    pin the behaviour the specification states loosely and a fitted estimator never
    reaches: the binary widening rule, `AVERAGE` against `base_values`, `MAX`
    normalization of a negative row, `ArrayFeatureExtractor`'s rank-1 shape.

Fixture and expectation MUST come from one pass: every estimator is seeded, and this
script writes both halves together. Generated separately, an unseeded forest fixture
gets compared against a *different* unseeded forest's probabilities — which is how the
first version of this file was wrong.

    uv venv .venv && uv pip install --python .venv/bin/python \
        scikit-learn skl2onnx onnx xgboost lightgbm onnxmltools onnxruntime packaging
    .venv/bin/python generate.py

Writes `<name>.onnx` beside itself, and `oracle.rs`, which `tests/ml.rs` includes.
"""

import pathlib

import numpy as np
import onnx
import onnxruntime as ort
import sklearn
from onnx import TensorProto, helper
from sklearn.datasets import load_iris
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Binarizer, MinMaxScaler, StandardScaler
from sklearn.svm import SVC, SVR, LinearSVC
from sklearn.tree import DecisionTreeClassifier

import lightgbm
import onnxmltools
import skl2onnx
import xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
from skl2onnx import to_onnx

HERE = pathlib.Path(__file__).parent
SEED = 0

X, y = load_iris(return_X_y=True)
X = X.astype(np.float32)
# Iris samples 0 and 120, whose classes are 0 and 2.
PROBE = np.stack([X[0], X[120]]).astype(np.float32)
BINARY = (y == 2).astype(np.int64)
TEXT = np.array(["setosa", "versicolor", "virginica"])[y]

FIXTURES = []


# --------------------------------------------------------------------------------------
# Emitting Rust
# --------------------------------------------------------------------------------------


def real(x):
    """A float as Rust source. NaN and the infinities have names, not digits."""
    if np.isnan(x):
        return "f32::NAN"
    if np.isinf(x):
        return "f32::INFINITY" if x > 0 else "f32::NEG_INFINITY"
    # str() of a float32 is the shortest decimal that reads back as the same
    # float32. Nine significant digits wrote more than the type carries -- 5.1
    # came out as 5.0999999 -- and clippy's excessive_precision then refused the
    # file this script had just written.
    return f"{np.float32(x)}f32"


def reals(v):
    return "&[" + ", ".join(real(x) for x in np.ravel(v).tolist()) + "]"


def ints(v):
    return "&[" + ", ".join(f"{int(x)}i64" for x in np.ravel(v).tolist()) + "]"


def text(v):
    return "&[" + ", ".join('"' + str(s) + '"' for s in np.ravel(v).tolist()) + "]"


def dims(v):
    return "&[" + ", ".join(str(d) for d in np.asarray(v).shape) + "]"


def data(v, keys=None):
    """One `Data` value: whatever kind the numpy array is, plus the table case."""
    a = np.asarray(v)
    if keys is not None:
        rows, cols = a.shape
        return f"Data::Table {{ keys: {names(keys)}, rows: {rows}, scores: {reals(a)} }}"
    if a.dtype.kind in "OU":
        return f"Data::Text {{ dims: {dims(a)}, values: {text(a)} }}"
    if a.dtype.kind in "iub":
        return f"Data::Ints {{ dims: {dims(a)}, values: {ints(a)} }}"
    return f"Data::Reals {{ dims: {dims(a)}, values: {reals(a)} }}"


def names(v):
    a = np.asarray(v)
    if a.dtype.kind in "OU":
        return f"Names::Text({text(a)})"
    return f"Names::Ints({ints(a)})"


def fixture(name, about, oracle, inputs, expect, tolerance=1e-5):
    FIXTURES.append(
        f"""    Fixture {{
        name: "{name}",
        about: "{about}",
        oracle: "{oracle}",
        tolerance: {real(tolerance)},
        inputs: &[{", ".join(f'("{n}", {data(v)})' for n, v in inputs)}],
        expect: &[{", ".join(f'("{n}", {v})' for n, v in expect)}],
    }},"""
    )


def save(name, model):
    (HERE / f"{name}.onnx").write_bytes(model.SerializeToString())
    return model


def outputs(model):
    return [o.name for o in model.graph.output]


def table(model, index):
    """Whether this graph output is a `seq(map(...))` — i.e. whether ZipMap produced it."""
    return model.graph.output[index].type.HasField("sequence_type")


def flatten(value):
    """onnxruntime's answer as a plain numpy array, whatever kind it is."""
    if isinstance(value, list) and value and isinstance(value[0], dict):
        keys = list(value[0].keys())
        return np.array([[row[k] for k in keys] for row in value], np.float32), keys
    return np.asarray(value), None


def run(model, feeds):
    session = ort.InferenceSession(
        model.SerializeToString(), providers=["CPUExecutionProvider"]
    )
    return [flatten(v) for v in session.run(None, feeds)]


def agrees(name, output, mine, theirs, tolerance):
    """A fixture is only a fair test if the exporter and the library already agree."""
    a, b = np.ravel(np.asarray(mine, np.float64)), np.ravel(np.asarray(theirs, np.float64))
    if a.shape != b.shape:
        raise SystemExit(f"{name}/{output}: library gave {a.shape}, onnxruntime {b.shape}")
    off = np.abs(a - b) - (tolerance + 1e-4 * np.abs(b))
    if off.max() > 0:
        raise SystemExit(
            f"{name}/{output}: the export does not reproduce the library it came from "
            f"(worst gap {np.abs(a - b).max():.3e}) — the fixture would test the wrong thing"
        )


# --------------------------------------------------------------------------------------
# Real exports, checked against the library's own prediction
# --------------------------------------------------------------------------------------


def record(name, about, model, oracle, answers, tolerance=1e-5):
    """Record the LIBRARY's numbers, in the shape the graph reports them in.

    The shape comes from onnxruntime, which is authoritative about the graph; the numbers
    come from the fitted library, which is authoritative about the model. Before either
    is written down the two are checked against each other — a fixture whose export does
    not reproduce its own library would test the wrong thing, so this fails loudly rather
    than pinning a discrepancy.
    """
    feeds = {model.graph.input[0].name: PROBE}
    theirs = run(model, feeds)
    expect = []
    for index, mine in enumerate(answers):
        got, keys = theirs[index]
        out = outputs(model)[index]
        # `None` means the library does not expose this output's exact vector, so
        # onnxruntime is the oracle for it. The fixture's `oracle` field says which.
        if mine is None:
            expect.append((out, data(got, keys=keys) if keys is not None else data(got)))
            continue
        if np.asarray(mine).dtype.kind in "OU":
            if list(np.ravel(mine)) != list(np.ravel(got)):
                raise SystemExit(f"{name}/{out}: labels {mine} vs onnxruntime {got}")
            expect.append((out, data(np.ravel(mine))))
            continue
        agrees(name, out, np.asarray(mine).reshape(got.shape), got, tolerance)
        shaped = np.asarray(mine, np.float32).reshape(got.shape)
        if table(model, index):
            expect.append((out, data(shaped, keys=keys)))
        elif np.asarray(mine).dtype.kind in "iub":
            expect.append((out, data(np.asarray(mine).reshape(got.shape))))
        else:
            expect.append((out, data(shaped)))
    fixture(name, about, oracle, list(feeds.items()), expect, tolerance)


def estimator(name, about, est, target, convert, oracle, tolerance=1e-5):
    """Fit, export at the exporter's defaults, and record the estimator's own answer."""
    est.fit(X, target)
    model = save(name, convert(est))
    if hasattr(est, "predict_proba"):
        answers = [est.predict(PROBE), est.predict_proba(PROBE)]
    else:
        answers = [est.predict(PROBE)]
    record(name, about, model, oracle, answers, tolerance)
    return model


def disputed(name, about, model, oracle, label, probability, classes, tolerance=1e-5):
    """Record the fitted library's answer for a graph onnxruntime reads DIFFERENTLY.

    [`record`] takes the output shapes from onnxruntime and refuses a fixture whose export
    does not reproduce its library. That is the right default: a disagreement is almost
    always the fixture's fault. It is not always — onnxruntime picks a binary tree
    ensemble's label by comparing the RAW score against 0.5, before `post_transform` runs,
    so on an all-positive-weight model whose margins lie in (0, 0.5] it reports class 0 for
    rows whose probability is above a half, contradicting its own score matrix and the
    library that fitted the model. There the library is the only oracle, and this records
    it without asking onnxruntime anything.
    """
    feeds = {model.graph.input[0].name: PROBE}
    out = outputs(model)
    scores = np.asarray(probability, np.float32)
    expect = [
        (out[0], data(np.ravel(label))),
        (out[1], data(scores, keys=classes) if table(model, 1) else data(scores)),
    ]
    fixture(name, about, oracle, list(feeds.items()), expect, tolerance)


def binary_positive(name, model, est):
    """Refuse to write a `disputed` fixture unless the export IS the disputed case.

    Four conditions make it one, and every one of them is what the fixture pins: one score
    column under two labels, `LOGISTIC`, every leaf weight non-negative, and margins inside
    (0, 0.5] — the window where onnxruntime's raw-score threshold and the sigmoid disagree
    about which side of a half the row is on. Without the last two this fixture would pass
    against a reading that never looks at the weights, which is how the branch it exists
    for came to be pinned by nothing.
    """
    node = next(n for n in model.graph.node if n.op_type == "TreeEnsembleClassifier")
    a = {at.name: helper.get_attribute_value(at) for at in node.attribute}
    weights = np.asarray(a["class_weights"], np.float64)
    ids = {int(i) for i in a["class_ids"]}
    post = a.get("post_transform", b"NONE")
    post = post.decode() if isinstance(post, bytes) else post
    margin = np.ravel(est.predict(PROBE, output_margin=True)).astype(np.float64)
    for what, ok in [
        (f"two class labels, got {len(a['classlabels_int64s'])}",
         len(a["classlabels_int64s"]) == 2),
        (f"one distinct class id, got {sorted(ids)}", len(ids) == 1),
        (f"post_transform LOGISTIC, got {post}", post == "LOGISTIC"),
        (f"every leaf weight >= 0, got a minimum of {weights.min():+.6f}", (weights >= 0).all()),
        (f"margins in (0, 0.5], got {np.round(margin, 6).tolist()}",
         bool(((margin > 0) & (margin <= 0.5)).all())),
    ]:
        if not ok:
            raise SystemExit(f"{name}: this fixture needs {what}")
    # The reading this evaluator implements, in numpy: sigmoid of the one score, then the
    # complement. If THAT does not reproduce the library, the fixture is wrong about the
    # model rather than about onnxruntime.
    p = 1.0 / (1.0 + np.exp(-margin))
    agrees(name, "probability", np.stack([1.0 - p, p], 1), est.predict_proba(PROBE), 1e-5)


def sk(est):
    return to_onnx(est, X[:1])


def ml(est):
    convert = (
        onnxmltools.convert_xgboost
        if isinstance(est, (xgboost.XGBClassifier, xgboost.XGBRegressor))
        else onnxmltools.convert_lightgbm
    )
    return convert(est, initial_types=[("X", FloatTensorType([None, 4]))])


SK = f"scikit-learn {sklearn.__version__}"
XGB = f"xgboost {xgboost.__version__}"
LGBM = f"lightgbm {lightgbm.__version__}"

estimator(
    "tree_clf",
    "one decision tree, the default export: TreeEnsembleClassifier + Cast + ZipMap",
    DecisionTreeClassifier(max_depth=3, random_state=SEED),
    y,
    sk,
    f"{SK} DecisionTreeClassifier.predict_proba",
)
estimator(
    "forest_clf",
    "three trees, so a leaf weight is a third of a vote and the walk must sum the ensemble",
    RandomForestClassifier(n_estimators=3, max_depth=3, random_state=SEED),
    y,
    sk,
    f"{SK} RandomForestClassifier.predict_proba",
)
estimator(
    "boost_clf",
    "gradient boosting: base_values, and a softmax over accumulated per-class scores",
    GradientBoostingClassifier(n_estimators=5, max_depth=2, random_state=SEED),
    y,
    sk,
    f"{SK} GradientBoostingClassifier.predict_proba",
)
estimator(
    "text_clf",
    "classlabels_strings: the label output is a tensor(string) and ZipMap is keyed by text",
    RandomForestClassifier(n_estimators=3, max_depth=3, random_state=SEED),
    TEXT,
    sk,
    f"{SK} RandomForestClassifier.predict_proba, string labels",
)
estimator(
    "linear_clf",
    "logistic regression: LinearClassifier(SOFTMAX) + Normalizer(L1) + Cast + ZipMap",
    LogisticRegression(max_iter=500),
    y,
    sk,
    f"{SK} LogisticRegression.predict_proba",
)
estimator(
    "binary_clf",
    "two classes: LinearClassifier with post_transform LOGISTIC and no Normalizer",
    LogisticRegression(max_iter=500),
    BINARY,
    sk,
    f"{SK} LogisticRegression.predict_proba, binary",
)
estimator(
    "scaler_pipe",
    "Scaler then LinearClassifier then Normalizer: three ai.onnx.ml ops in a row",
    make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)),
    y,
    sk,
    f"{SK} StandardScaler + LogisticRegression.predict_proba",
)
estimator(
    "impute_pipe",
    "Imputer in front of a classifier",
    make_pipeline(SimpleImputer(), LogisticRegression(max_iter=500)),
    y,
    sk,
    f"{SK} SimpleImputer + LogisticRegression.predict_proba",
)
estimator(
    "binarize_pipe",
    "Binarizer in front of a classifier",
    make_pipeline(Binarizer(threshold=3.0), LogisticRegression(max_iter=500)),
    y,
    sk,
    f"{SK} Binarizer + LogisticRegression.predict_proba",
)
estimator(
    "linear_reg",
    "LinearRegressor on its own",
    LinearRegression(),
    y.astype(np.float32),
    sk,
    f"{SK} LinearRegression.predict",
)
estimator(
    "minmax_reg",
    "MinMaxScaler then a ridge regression: Scaler + LinearRegressor",
    make_pipeline(MinMaxScaler(), Ridge()),
    y.astype(np.float32),
    sk,
    f"{SK} MinMaxScaler + Ridge.predict",
)
estimator(
    "forest_reg",
    "TreeEnsembleRegressor with aggregate_function AVERAGE",
    RandomForestRegressor(n_estimators=3, max_depth=3, random_state=SEED),
    y.astype(np.float32),
    sk,
    f"{SK} RandomForestRegressor.predict",
)
estimator(
    "boost_reg",
    "TreeEnsembleRegressor with base_values and aggregate_function SUM",
    GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=SEED),
    y.astype(np.float32),
    sk,
    f"{SK} GradientBoostingRegressor.predict",
)
estimator(
    "svm_reg",
    "SVMRegressor with an RBF kernel",
    SVR(),
    y.astype(np.float32),
    sk,
    f"{SK} SVR.predict",
)
estimator(
    "xgb_clf",
    "XGBoost multi-class: one TreeEnsembleClassifier node with post_transform SOFTMAX",
    xgboost.XGBClassifier(n_estimators=3, max_depth=2, random_state=SEED),
    y,
    ml,
    f"{XGB} XGBClassifier.predict_proba",
)
estimator(
    "xgb_bin",
    "XGBoost binary: ONE score column, two class labels, base_values of length one",
    xgboost.XGBClassifier(n_estimators=3, max_depth=2, random_state=SEED),
    BINARY,
    ml,
    f"{XGB} XGBClassifier.predict_proba, binary",
)

# The binary case again, on the branch the two fixtures above cannot reach. `xgb_bin` and
# `lgbm_bin` both happen to fit at least one NEGATIVE leaf weight, so the widening rule
# they exercise is the mixed-sign one; the all-positive branch — where onnxruntime picks the
# label from the raw score and gets it wrong — was pinned by nothing.
#
# Making every leaf weight positive takes a target the four features cannot explain: with
# the negatives scattered at random, no split isolates them, so every leaf keeps a positive
# residual and therefore a positive weight. `base_score=0.5` puts the margin intercept at
# zero and a small learning rate over three stumps keeps the total margin under a half,
# which is the window where the two readings disagree about the label.
_noise = np.random.default_rng(SEED)
_scattered = np.ones(len(y), np.int64)
_scattered[_noise.permutation(len(y))[: len(y) // 10]] = 0
_positive = xgboost.XGBClassifier(
    n_estimators=3, max_depth=1, learning_rate=0.1, base_score=0.5, random_state=SEED
).fit(X, _scattered)
_positive_model = save("xgb_bin_positive", ml(_positive))
binary_positive("xgb_bin_positive", _positive_model, _positive)
disputed(
    "xgb_bin_positive",
    "XGBoost binary with every leaf weight positive: the label is the argmax of [1 - p, p]",
    _positive_model,
    f"{XGB} XGBClassifier.predict/.predict_proba — NOT onnxruntime, which reports the other "
    f"class for these rows",
    _positive.predict(PROBE),
    _positive.predict_proba(PROBE),
    _positive.classes_,
)

estimator(
    "xgb_reg",
    "XGBoost regression through TreeEnsembleRegressor",
    xgboost.XGBRegressor(n_estimators=3, max_depth=2, random_state=SEED),
    y.astype(np.float32),
    ml,
    f"{XGB} XGBRegressor.predict",
)
estimator(
    "lgbm_clf",
    "LightGBM multi-class through TreeEnsembleClassifier",
    lightgbm.LGBMClassifier(n_estimators=3, max_depth=2, random_state=SEED, verbose=-1),
    y,
    ml,
    f"{LGBM} LGBMClassifier.predict_proba",
)
estimator(
    "lgbm_bin",
    "LightGBM binary: the same one-column widening, plus Identity and Cast on the label",
    lightgbm.LGBMClassifier(n_estimators=3, max_depth=2, random_state=SEED, verbose=-1),
    BINARY,
    ml,
    f"{LGBM} LGBMClassifier.predict_proba, binary",
)
estimator(
    "lgbm_reg",
    "LightGBM regression through TreeEnsembleRegressor",
    lightgbm.LGBMRegressor(n_estimators=3, max_depth=2, random_state=SEED, verbose=-1),
    y.astype(np.float32),
    ml,
    f"{LGBM} LGBMRegressor.predict",
)


def decisions(name, about, est, target, oracle, tolerance=1e-5, scores=True):
    """An estimator whose reported scores are decision values, not probabilities.

    `scores=False` where the library does not expose the exact vector the graph reports —
    a two-class SVC reports one decision value but its export keys a map by both classes —
    in which case onnxruntime is the oracle for that output and `oracle` says so.
    """
    est.fit(X, target)
    model = save(name, sk(est))
    record(
        name,
        about,
        model,
        oracle,
        [est.predict(PROBE), est.decision_function(PROBE) if scores else None],
        tolerance,
    )


decisions(
    "svm_clf",
    "SVMClassifier: libsvm one-against-one, then the ai.onnx arithmetic that votes",
    SVC(),
    y,
    f"{SK} SVC.decision_function",
)
decisions(
    "svm_bin",
    "SVMClassifier over two classes: one decision value, reported under both class keys",
    SVC(),
    BINARY,
    f"{SK} SVC.predict; onnxruntime {ort.__version__} for the two-column score map",
    scores=False,
)
decisions(
    "linear_svc",
    "LinearClassifier followed by ArrayFeatureExtractor picking the positive class column",
    LinearSVC(max_iter=20000),
    BINARY,
    f"{SK} LinearSVC.decision_function",
    tolerance=1e-4,
)


# IsolationForest is what an anomaly detector wants, and its export is a large
# ai.onnx graph around TreeEnsembleRegressor rather than one classical node.
forest = IsolationForest(n_estimators=4, random_state=SEED).fit(X)
# skl2onnx builds this one at ai.onnx.ml 4 by default and then refuses its own output;
# 3 is the version every other exporter targets and the one this evaluator reads.
record(
    "isolation",
    "IsolationForest: TreeEnsembleRegressor inside a graph of ai.onnx arithmetic",
    save("isolation", to_onnx(forest, X[:1], target_opset={"ai.onnx.ml": 3, "": 18})),
    f"{SK} IsolationForest.decision_function",
    [forest.predict(PROBE), forest.decision_function(PROBE)],
    tolerance=1e-4,
)


# --------------------------------------------------------------------------------------
# Hand-built operators, checked against onnxruntime
# --------------------------------------------------------------------------------------

RT = f"onnxruntime {ort.__version__}"
KIND = {
    np.dtype(np.float32): TensorProto.FLOAT,
    np.dtype(np.int64): TensorProto.INT64,
}


def info(name, a):
    a = np.asarray(a)
    kind = TensorProto.STRING if a.dtype.kind in "OU" else KIND[a.dtype]
    return helper.make_tensor_value_info(name, kind, list(a.shape))


def out_info(name, kind):
    """`None` means `seq(map(int64, tensor(float)))` — the only non-tensor output here."""
    if kind is not None:
        return helper.make_tensor_value_info(name, kind, None)
    maps = helper.make_map_type_proto(
        TensorProto.INT64, helper.make_tensor_type_proto(TensorProto.FLOAT, None)
    )
    return helper.make_value_info(name, helper.make_sequence_type_proto(maps))


def probe(name, about, node, feeds, out_kinds, mlv=3):
    """Run one node through onnxruntime and record what it answered."""
    graph = helper.make_graph(
        [node],
        name,
        [info(n, v) for n, v in feeds.items()],
        [out_info(n, k) for n, k in out_kinds],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("ai.onnx.ml", mlv), helper.make_opsetid("", 13)],
    )
    model.ir_version = 8
    save(name, model)
    expect = []
    for index, (values, keys) in enumerate(run(model, feeds)):
        out = outputs(model)[index]
        expect.append((out, data(values, keys=keys) if keys is not None else data(values)))
    fixture(name, about, RT, list(feeds.items()), expect)


F, I, S = TensorProto.FLOAT, TensorProto.INT64, TensorProto.STRING


def stump(weights, post, labels, base=None, class_ids=None):
    """One stump splitting feature 0 at 0.5, with `weights` on its two leaves."""
    n = len(weights)
    kw = dict(
        nodes_treeids=[0, 0, 0],
        nodes_nodeids=[0, 1, 2],
        nodes_featureids=[0, 0, 0],
        nodes_values=[0.5, 0.0, 0.0],
        nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
        nodes_truenodeids=[1, 0, 0],
        nodes_falsenodeids=[2, 0, 0],
        class_treeids=[0] * n,
        class_nodeids=[1 if i < n // 2 or n == 2 else 2 for i in range(n)],
        class_ids=list(class_ids) if class_ids else [0] * n,
        class_weights=list(weights),
        classlabels_int64s=list(labels),
        post_transform=post,
    )
    if n == 2:
        kw["class_nodeids"] = [1, 2]
    if base is not None:
        kw["base_values"] = list(base)
    return helper.make_node(
        "TreeEnsembleClassifier", ["X"], ["L", "S"], domain="ai.onnx.ml", **kw
    )


ONE = np.array([[0.0], [1.0]], np.float32)
CLF = [("L", I), ("S", F)]

probe(
    "widen_none_positive",
    "one score column, two labels, no post_transform, every weight non-negative: [1-s, s]",
    stump([0.25, 0.75], "NONE", [0, 1]),
    {"X": ONE},
    CLF,
)
probe(
    "widen_none_mixed",
    "the same with a negative weight: [-s, s], because no probability was claimed",
    stump([-0.25, 0.75], "NONE", [0, 1]),
    {"X": ONE},
    CLF,
)
probe(
    "widen_logistic",
    "with LOGISTIC the transform runs on the one score first, then [1-p, p]",
    stump([-0.25, 0.75], "LOGISTIC", [0, 1]),
    {"X": ONE},
    CLF,
)
probe(
    "widen_base",
    "base_values has ONE entry in the binary case, not one per label",
    stump([0.25, 0.75], "NONE", [0, 1], base=[-0.5]),
    {"X": ONE},
    CLF,
)
probe(
    "no_widen_three",
    "three labels and one class id is NOT the binary case: three columns, two of them zero",
    stump([0.25, 0.75], "NONE", [0, 1, 2]),
    {"X": ONE},
    CLF,
)
probe(
    "argmax_tie",
    "a tie between two classes goes to the LOWER class index, not the later one",
    helper.make_node(
        "TreeEnsembleClassifier",
        ["X"],
        ["L", "S"],
        domain="ai.onnx.ml",
        nodes_treeids=[0, 0, 0],
        nodes_nodeids=[0, 1, 2],
        nodes_featureids=[0, 0, 0],
        nodes_values=[0.5, 0.0, 0.0],
        nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"],
        nodes_truenodeids=[1, 0, 0],
        nodes_falsenodeids=[2, 0, 0],
        class_treeids=[0, 0, 0, 0],
        class_nodeids=[1, 1, 2, 2],
        class_ids=[0, 2, 1, 2],
        class_weights=[0.5, 0.5, 0.25, 0.75],
        classlabels_int64s=[0, 1, 2],
        post_transform="NONE",
    ),
    {"X": ONE},
    CLF,
)


def regressor(aggregate, base=None):
    kw = dict(
        nodes_treeids=[0, 0, 0, 1, 1, 1],
        nodes_nodeids=[0, 1, 2, 0, 1, 2],
        nodes_featureids=[0] * 6,
        nodes_values=[0.5, 0, 0, 0.5, 0, 0],
        nodes_modes=["BRANCH_LEQ", "LEAF", "LEAF"] * 2,
        nodes_truenodeids=[1, 0, 0, 1, 0, 0],
        nodes_falsenodeids=[2, 0, 0, 2, 0, 0],
        target_treeids=[0, 0, 1, 1],
        target_nodeids=[1, 2, 1, 2],
        target_ids=[0, 0, 0, 0],
        target_weights=[1.0, 3.0, 10.0, 30.0],
        n_targets=1,
        aggregate_function=aggregate,
    )
    if base is not None:
        kw["base_values"] = list(base)
    return helper.make_node(
        "TreeEnsembleRegressor", ["X"], ["Y"], domain="ai.onnx.ml", **kw
    )


for aggregate in ["SUM", "AVERAGE", "MIN", "MAX"]:
    probe(
        f"aggregate_{aggregate.lower()}",
        f"two stumps with leaves 1/3 and 10/30 combined by {aggregate}",
        regressor(aggregate),
        {"X": ONE},
        [("Y", F)],
    )

def nan_tree(mode, missing):
    """One stump under `mode`, whose yes leaf is 1 and no leaf is 2, fed a NaN."""
    return helper.make_node(
        "TreeEnsembleRegressor",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        nodes_treeids=[0, 0, 0],
        nodes_nodeids=[0, 1, 2],
        nodes_featureids=[0, 0, 0],
        nodes_values=[0.5, 0.0, 0.0],
        nodes_modes=[mode, "LEAF", "LEAF"],
        nodes_truenodeids=[1, 0, 0],
        nodes_falsenodeids=[2, 0, 0],
        nodes_missing_value_tracks_true=[missing, 0, 0],
        target_treeids=[0, 0],
        target_nodeids=[1, 2],
        target_ids=[0, 0],
        target_weights=[1.0, 2.0],
        n_targets=1,
    )


NAN = np.array([[np.nan]], np.float32)

probe(
    "branch_neq_nan",
    "NaN != threshold is TRUE, so BRANCH_NEQ takes the yes branch on a missing feature",
    nan_tree("BRANCH_NEQ", 0),
    {"X": NAN},
    [("Y", F)],
)
probe(
    "branch_leq_nan",
    "every other comparison is false on a NaN, so the missing-value flag decides alone",
    nan_tree("BRANCH_LEQ", 1),
    {"X": NAN},
    [("Y", F)],
)
probe(
    "aggregate_average_base",
    "base_values is added AFTER the division, so 100 + 11/2 and not (100 + 11)/2",
    regressor("AVERAGE", base=[100.0]),
    {"X": ONE},
    [("Y", F)],
)
probe(
    "aggregate_min_base",
    "and after MIN too",
    regressor("MIN", base=[100.0]),
    {"X": ONE},
    [("Y", F)],
)

probe(
    "normalize_max_signed",
    "MAX normalization divides by the SIGNED maximum, so a negative row flips sign",
    helper.make_node("Normalizer", ["X"], ["Y"], domain="ai.onnx.ml", norm="MAX"),
    {"X": np.array([[-4.0, 2.0, -1.0], [-4.0, -2.0, -1.0]], np.float32)},
    [("Y", F)],
)
probe(
    "normalize_zero",
    "a zero norm leaves the row alone rather than producing NaNs",
    helper.make_node("Normalizer", ["X"], ["Y"], domain="ai.onnx.ml", norm="L1"),
    {"X": np.zeros((1, 3), np.float32)},
    [("Y", F)],
)
probe(
    "impute_nan",
    "replaced_value_float may be NaN, which is equal to nothing, so the test is isnan",
    helper.make_node(
        "Imputer",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        imputed_value_floats=[10.0, 20.0, 30.0],
        replaced_value_float=float("nan"),
    ),
    {"X": np.array([[np.nan, 1, np.nan], [0, np.nan, 3]], np.float32)},
    [("Y", F)],
)
probe(
    "impute_scalar",
    "one imputed value applies to every feature",
    helper.make_node(
        "Imputer",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        imputed_value_floats=[99.0],
        replaced_value_float=0.0,
    ),
    {"X": np.array([[0, 1, 0], [0, 2, 3]], np.float32)},
    [("Y", F)],
)
probe(
    "impute_int",
    "an int64 column is imputed in int64: 2^53 + 1 does not survive the float plane",
    helper.make_node(
        "Imputer",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        imputed_value_int64s=[10, 20, 30],
        replaced_value_int64=-1,
    ),
    {"X": np.array([[-1, 1, 9007199254740993], [0, -1, 3]], np.int64)},
    [("Y", I)],
)
probe(
    "binarize",
    "STRICTLY greater: a value equal to the threshold reports zero",
    helper.make_node("Binarizer", ["X"], ["Y"], domain="ai.onnx.ml", threshold=1.5),
    {"X": np.array([[1.0, 1.5, 2.0, -3.0]], np.float32)},
    [("Y", F)],
)
probe(
    "extract_matrix",
    "ArrayFeatureExtractor selects along the last axis",
    helper.make_node("ArrayFeatureExtractor", ["X", "I"], ["Y"], domain="ai.onnx.ml"),
    {"X": np.arange(6, dtype=np.float32).reshape(2, 3), "I": np.array([0, 2], np.int64)},
    [("Y", F)],
)
probe(
    "extract_vector",
    "a rank-1 input gives a rank-2 answer: shape (1, k), not (k)",
    helper.make_node("ArrayFeatureExtractor", ["X", "I"], ["Y"], domain="ai.onnx.ml"),
    {"X": np.array([5, 6, 7], np.float32), "I": np.array([2], np.int64)},
    [("Y", F)],
)
probe(
    "vectorize_pad",
    "an input narrower than its declared width is padded with zeros",
    helper.make_node(
        "FeatureVectorizer", ["A", "B"], ["Y"], domain="ai.onnx.ml", inputdimensions=[2, 3]
    ),
    {"A": np.array([[1, 2]], np.float32), "B": np.array([[7, 8]], np.float32)},
    [("Y", F)],
)
probe(
    "encode_text_to_int",
    "LabelEncoder from text keys to integer values, with a default for the absent one",
    helper.make_node(
        "LabelEncoder",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        keys_strings=["a", "b"],
        values_int64s=[10, 20],
        default_int64=-7,
    ),
    {"X": np.array(["b", "a", "zz"], object)},
    [("Y", I)],
)
probe(
    "encode_int_to_text",
    "and the other direction",
    helper.make_node(
        "LabelEncoder",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        keys_int64s=[1, 2],
        values_strings=["one", "two"],
        default_string="NA",
    ),
    {"X": np.array([2, 1, 9], np.int64)},
    [("Y", S)],
)
probe(
    "encode_real",
    "float keys are matched exactly, which is what a lookup means",
    helper.make_node(
        "LabelEncoder",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        keys_floats=[1.5, 2.5],
        values_floats=[10.0, 20.0],
        default_float=-1.0,
    ),
    {"X": np.array([2.5, 1.5, 9.0], np.float32)},
    [("Y", F)],
)
probe(
    "encode_duplicate",
    "a key listed twice is answered from its FIRST position, the order a table is read in",
    helper.make_node(
        "LabelEncoder",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        keys_strings=["a", "a", "b"],
        values_int64s=[1, 2, 3],
        default_int64=-1,
    ),
    {"X": np.array(["a", "b", "zz"], object)},
    [("Y", I)],
)
probe(
    "map_text_to_int",
    "CategoryMapper reads its direction from the input it is given",
    helper.make_node(
        "CategoryMapper",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        cats_strings=["a", "b"],
        cats_int64s=[5, 6],
        default_int64=-1,
    ),
    {"X": np.array(["a", "b", "q"], object)},
    [("Y", I)],
)
probe(
    "map_int_to_text",
    "the same node's other direction",
    helper.make_node(
        "CategoryMapper",
        ["X"],
        ["Y"],
        domain="ai.onnx.ml",
        cats_strings=["a", "b"],
        cats_int64s=[5, 6],
        default_string="ZZ",
    ),
    {"X": np.array([6, 5, 0], np.int64)},
    [("Y", S)],
)
probe(
    "onehot",
    "one extra trailing axis; an element in no category is an all-zero row when zeros is set",
    helper.make_node(
        "OneHotEncoder", ["X"], ["Y"], domain="ai.onnx.ml", cats_int64s=[1, 2, 3], zeros=1
    ),
    {"X": np.array([[2, 9]], np.int64)},
    [("Y", F)],
)
probe(
    "scale",
    "Scaler is (x - offset) * scale, per feature",
    helper.make_node(
        "Scaler", ["X"], ["Y"], domain="ai.onnx.ml", offset=[1.0, 2.0], scale=[0.5, 4.0]
    ),
    {"X": np.array([[3.0, 4.0], [-1.0, 0.0]], np.float32)},
    [("Y", F)],
)
def solo(name, about, est, target, op):
    """One node lifted out of a real export, so the operator is tested on its own.

    skl2onnx wraps `SVMClassifier` in a dozen `ai.onnx` nodes that turn pair votes into
    one-against-rest scores. Running the whole graph checks the composition; running the
    node alone checks the operator — and only the second says which of the two is wrong.
    """
    est.fit(X, target)
    node = next(n for n in to_onnx(est, X[:1]).graph.node if n.op_type == op)
    lifted = helper.make_node(
        op,
        ["X"],
        ["L", "S"],
        domain="ai.onnx.ml",
        **{a.name: helper.get_attribute_value(a) for a in node.attribute},
    )
    probe(name, about, lifted, {"X": PROBE}, [("L", I), ("S", F)])


solo(
    "svm_node_multi",
    "SVMClassifier alone: one decision value per class pair, plus the pair vote",
    SVC(),
    y,
    "SVMClassifier",
)
solo(
    "svm_node_binary",
    "two classes report [-D, D] from their single pair, and the vote still picks the label",
    SVC(),
    BINARY,
    "SVMClassifier",
)

probe(
    "zipmap",
    "ZipMap keys a score matrix by the labels its columns are under",
    helper.make_node(
        "ZipMap", ["X"], ["Y"], domain="ai.onnx.ml", classlabels_int64s=[7, 8, 9]
    ),
    {"X": np.array([[0.1, 0.2, 0.7], [0.5, 0.25, 0.25]], np.float32)},
    [("Y", None)],
)


# --------------------------------------------------------------------------------------

versions = ", ".join(
    f"{name} {mod.__version__}"
    for name, mod in [
        ("scikit-learn", sklearn),
        ("skl2onnx", skl2onnx),
        ("onnx", onnx),
        ("onnxruntime", ort),
        ("xgboost", xgboost),
        ("lightgbm", lightgbm),
        ("onnxmltools", onnxmltools),
    ]
)

(HERE / "oracle.rs").write_text(
    f"""// GENERATED by tests/ml/generate.py — do not edit by hand.
//
// Produced with {versions}.
//
// Every `Fixture` names the oracle its expectation came from: the fitted library's own
// prediction for a real export, or onnxruntime for a hand-built operator whose behaviour
// the specification leaves open.

/// One value, in whichever of ONNX's kinds it is.
#[allow(dead_code)]
enum Data {{
    Reals {{ dims: &'static [usize], values: &'static [f32] }},
    Ints {{ dims: &'static [usize], values: &'static [i64] }},
    Text {{ dims: &'static [usize], values: &'static [&'static str] }},
    Table {{ keys: Names, rows: usize, scores: &'static [f32] }},
}}

/// Labels, or a table's keys: integers or text, the same sum `hanzo_onnx::Labels` is.
#[allow(dead_code)]
enum Names {{
    Ints(&'static [i64]),
    Text(&'static [&'static str]),
}}

/// One graph, what to feed it, and what a trusted implementation answers.
struct Fixture {{
    /// The `.onnx` beside this file, without its extension.
    name: &'static str,
    /// What this case pins down.
    about: &'static str,
    /// Who produced `expect`.
    oracle: &'static str,
    tolerance: f32,
    inputs: &'static [(&'static str, Data)],
    expect: &'static [(&'static str, Data)],
}}

const FIXTURES: &[Fixture] = &[
{chr(10).join(FIXTURES)}
];
"""
)

print(f"wrote {len(FIXTURES)} fixtures and oracle.rs")
