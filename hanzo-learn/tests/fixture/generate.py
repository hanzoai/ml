#!/usr/bin/env python3
"""Mint the fixtures that hold hanzo-learn to scikit-learn's own answer.

scikit-learn IS the oracle here. This script is the only place it is consulted:
it runs once, writes JSON, and the JSON is committed. The Rust tests read the
JSON and never call Python — so the bar is checked on every `cargo test` on a
box with no Python at all, and the oracle's version is a recorded fact rather
than whatever happened to be installed.

WHY THE FIXTURES ARE DETERMINISTIC, which is what makes an exact assertion
legitimate rather than lucky:

  Least squares      The minimiser of ||Xb - y||^2 is UNIQUE when X has full
                     column rank. Two correct solvers must therefore agree to
                     within conditioning; they cannot disagree on the answer,
                     only on the rounding. X here is drawn from a continuous
                     distribution with p << n, so full rank holds with
                     probability 1 and the condition number is small.

  Logistic           The L2-penalised logistic objective is STRICTLY convex
                     (the penalty makes it so even under separability), so it
                     has exactly one minimiser. sklearn reaches it with lbfgs
                     and hanzo-learn reaches it with Newton steps; agreement is
                     forced by uniqueness, and the tolerance is just how far
                     each stopped from the shared optimum.

  Boosted trees      NOT determinstic at scikit-learn's defaults, and this is
                     the sharpest thing measured here. See DETERMINISM below.

DETERMINISM — a tree fit is not a function of its data at scikit-learn's defaults.

  `min_samples_leaf=1` lets the recursion reach nodes holding two samples. EVERY
  feature splits two samples into the SAME two singletons, so every feature earns
  the IDENTICAL split score, and scikit-learn breaks that tie by whichever feature
  a `random_state`-seeded partial Fisher-Yates shuffle visited first. Measured on
  this data:

      DecisionTreeRegressor(max_depth=3)                 20 structures / 40 seeds
                                                         prediction spread 3.9e+00
      DecisionTreeRegressor(max_depth=3, leaf>=7)         1 structure
                                                         prediction spread 4.4e-16
      GradientBoostingRegressor(depth=3)                 12 structures / 12 seeds
                                                         prediction spread 3.9e-01
      GradientBoostingRegressor(depth=5)                 12 structures
                                                         prediction spread 1.1e+00
      GradientBoostingRegressor(depth=5, leaf>=10)        1 structure
                                                         prediction spread 8.9e-16

  The spread at the defaults is O(1) on a target whose own spread is O(1) — it is a
  DIFFERENT MODEL, not a different rounding. So "the tree scikit-learn fits to this
  data" is not a well-defined value at the defaults, and no port can be held to it.

  Every case below therefore pins a leaf bound (or a depth) that removes the tie,
  and `deterministic()` PROVES it over 12 seeds — structure and predictions both —
  before a single number is recorded. A fixture that recorded an arbitrary
  tie-break would be a test that fails for a correct implementation.

Regenerate with:
    <venv>/bin/python hanzo-learn/tests/fixture/generate.py
"""

import json
import pathlib

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

HERE = pathlib.Path(__file__).parent

# One generator, one seed, stated. Every fixture below draws from it in order,
# so the data is reproducible from this file alone.
SEED = 20260804


def design(rng, n, p):
    """A design matrix from a continuous distribution.

    Continuity is load-bearing twice over: it makes X full-rank almost surely
    (so least squares has one answer) and it makes every candidate tree split
    score distinct (so the best split is unique).
    """
    return rng.standard_normal((n, p))


def finite(v):
    """JSON has no infinity, and `json.dumps` writes a bare `Infinity` that is not JSON.

    scikit-learn's `roc_curve` legitimately returns `inf` as its first threshold — the
    operating point that flags nothing — so the value has to survive the trip. It is
    written as a STRING and read back as one, rather than as a large number that would
    silently become a real threshold.
    """
    if isinstance(v, float):
        if v == float("inf"):
            return "inf"
        if v == float("-inf"):
            return "-inf"
        if v != v:
            return "nan"
        return v
    if isinstance(v, dict):
        return {k: finite(x) for k, x in v.items()}
    if isinstance(v, list):
        return [finite(x) for x in v]
    return v


def dump(name, payload):
    payload["oracle"] = {"sklearn": sklearn.__version__, "numpy": np.__version__}
    path = HERE / name
    path.write_text(json.dumps(finite(payload), indent=1, sort_keys=True) + "\n")
    # Prove it is JSON, here, rather than discovering it from a Rust parse error.
    json.loads(path.read_text())
    print(f"{path.name}: {path.stat().st_size} bytes")


def linear():
    rng = np.random.default_rng(SEED)
    n, p, n_test = 200, 5, 40
    x = design(rng, n, p)
    beta = np.array([1.5, -2.0, 0.5, 0.0, 3.25])
    y = x @ beta + 7.0 + 0.1 * rng.standard_normal(n)
    x_test = design(rng, n_test, p)

    out = {"x": x.tolist(), "y": y.tolist(), "x_test": x_test.tolist(), "cases": {}}
    for label, intercept in (("intercept", True), ("no_intercept", False)):
        m = LinearRegression(fit_intercept=intercept).fit(x, y)
        out["cases"][label] = {
            "fit_intercept": intercept,
            "coef": m.coef_.tolist(),
            "intercept": float(m.intercept_),
            "predict_test": m.predict(x_test).tolist(),
        }
    # The condition number is recorded, not assumed: it is the term that bounds
    # how far two correct solvers may differ, so the tolerance cites it.
    out["cond"] = float(np.linalg.cond(np.c_[np.ones(n), x]))
    dump("linear.json", out)


def logistic():
    rng = np.random.default_rng(SEED + 1)
    n, p, n_test = 300, 4, 60
    x = design(rng, n, p)
    beta = np.array([2.0, -1.5, 0.75, 0.0])
    # Labels drawn from the true Bernoulli model, so the classes OVERLAP. A
    # separable fixture would let any large-margin solution pass; overlap forces
    # the optimum to be interior and makes the coefficients themselves testable.
    logit = x @ beta - 0.4
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    x_test = design(rng, n_test, p)

    out = {
        "x": x.tolist(),
        "y": y.tolist(),
        "x_test": x_test.tolist(),
        "class_balance": [int((y == 0).sum()), int((y == 1).sum())],
        "cases": {},
    }
    for label, c in (("c1", 1.0), ("c100", 100.0)):
        # tol is tightened well past the default 1e-4 so sklearn's own distance
        # from the optimum is not what the tolerance is measuring.
        m = LogisticRegression(C=c, tol=1e-12, max_iter=10000).fit(x, y)
        out["cases"][label] = {
            "c": c,
            "coef": m.coef_[0].tolist(),
            "intercept": float(m.intercept_[0]),
            "predict_test": m.predict(x_test).astype(int).tolist(),
            "proba_test": m.predict_proba(x_test)[:, 1].tolist(),
        }
    dump("logistic.json", out)


def deterministic(label, x, y, x_test, kw, seeds=12):
    """Refuse to record an oracle that is not a function of its data.

    Fits the same configuration under `seeds` different `random_state` values and
    demands ONE structure and predictions agreeing to 1e-12. If the tie described in
    DETERMINISM is reachable under `kw`, this raises instead of writing a fixture
    that a correct implementation would fail.
    """
    shapes, preds = set(), []
    for s in range(seeds):
        m = GradientBoostingRegressor(random_state=s, **kw).fit(x, y)
        shapes.add(tuple(tuple(e.tree_.feature.tolist()) for e in m.estimators_[:, 0]))
        preds.append(m.predict(x_test))
    spread = float(np.abs(np.array(preds) - preds[0]).max())
    if len(shapes) != 1 or spread > 1e-12:
        raise SystemExit(
            f"{label}: scikit-learn is not deterministic under {kw} — "
            f"{len(shapes)} structures over {seeds} seeds, prediction spread {spread:.3e}. "
            "Raise min_samples_leaf or lower max_depth until the tie is unreachable."
        )
    print(f"  {label}: deterministic over {seeds} seeds (spread {spread:.1e})")


def boosted():
    rng = np.random.default_rng(SEED + 2)
    n, p, n_test = 200, 4, 50
    x = design(rng, n, p)
    # A nonlinear target, so the trees have real structure to find rather than a
    # linear signal a depth-3 stump ladder would only approximate.
    y = np.sin(2.0 * x[:, 0]) + x[:, 1] ** 2 - 0.5 * x[:, 2] * x[:, 3] + 0.05 * rng.standard_normal(n)
    x_test = design(rng, n_test, p)

    out = {"x": x.tolist(), "y": y.tolist(), "x_test": x_test.tolist(), "cases": {}}
    for label, kw in (
        # Every case pins the tie away. `stumps` needs no leaf bound because a
        # depth-1 tree never reaches a small node.
        ("default", dict(n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_leaf=8)),
        ("deep", dict(n_estimators=25, max_depth=5, learning_rate=0.25, min_samples_leaf=10)),
        ("stumps", dict(n_estimators=40, max_depth=1, learning_rate=0.5)),
        # A larger leaf bound also exercises the bound itself in the split search,
        # which is where an off-by-one in a candidate scan hides.
        ("leafbound", dict(n_estimators=30, max_depth=4, learning_rate=0.2, min_samples_leaf=7)),
    ):
        deterministic(label, x, y, x_test, kw)
        m = GradientBoostingRegressor(random_state=0, **kw).fit(x, y)
        # Every tree is dumped in full. Asserting STRUCTURE — the split feature,
        # the threshold and the leaf value at all ~1500 nodes — is a far stronger
        # claim than asserting predictions, and it localises a divergence to the
        # first node that differs instead of reporting one wrong number at the end.
        trees = []
        for est in m.estimators_[:, 0]:
            t = est.tree_
            trees.append(
                {
                    "left": t.children_left.tolist(),
                    "right": t.children_right.tolist(),
                    "feature": t.feature.tolist(),
                    "threshold": t.threshold.tolist(),
                    # value is the leaf's own mean of the residuals it was fitted
                    # on; shrinkage is applied by the ensemble, not stored here.
                    "value": t.value.reshape(-1).tolist(),
                }
            )
        out["cases"][label] = {
            "config": kw,
            "init": float(m.init_.constant_.ravel()[0]),
            "predict_test": m.predict(x_test).tolist(),
            "trees": trees,
        }
    dump("boosted.json", out)


def prepare():
    """The unglamorous half: scaling, encoding, imputing, splitting.

    Every one of these is a CLOSED FORM — a mean, a range, a category order, a
    fold assignment. There is no optimiser and no tie to break, so agreement
    here is exact by construction and the fixture asserts 0.0 difference, not a
    tolerance. That is why these are worth pinning: a preprocessing step that
    disagrees in the 8th digit has silently changed every model downstream of
    it, and nothing else in a pipeline would report it.
    """
    rng = np.random.default_rng(SEED + 3)
    n, p = 240, 6
    x = design(rng, n, p) * np.array([1.0, 50.0, 0.01, 3.0, 1e4, 7.0]) + 2.0
    # A column with no spread at all: the case where a scaler must choose
    # between a divisor of zero and a stated convention.
    x[:, 3] = 4.5
    x_test = design(rng, 40, p) * 2.0 + 1.0

    out = {"x": x.tolist(), "x_test": x_test.tolist()}

    s = StandardScaler().fit(x)
    out["standard"] = {
        "mean": s.mean_.tolist(),
        "var": s.var_.tolist(),
        "scale": s.scale_.tolist(),
        "apply": s.transform(x_test).tolist(),
        "invert": s.inverse_transform(s.transform(x_test)).tolist(),
    }

    for label, span in (("unit", (0.0, 1.0)), ("shifted", (-3.0, 7.0))):
        m = MinMaxScaler(feature_range=span).fit(x)
        out.setdefault("range", {})[label] = {
            "span": list(span),
            "min": m.min_.tolist(),
            "scale": m.scale_.tolist(),
            "data_min": m.data_min_.tolist(),
            "data_max": m.data_max_.tolist(),
            "apply": m.transform(x_test).tolist(),
        }

    # Labels as arbitrary integers, deliberately not 0..k and not sorted, so the
    # fixture pins that classes come out in ASCENDING order rather than in the
    # order they were met.
    labels = np.array([7, 3, 3, 90, 7, -2, 90, 7, -2, -2, 3])
    le = LabelEncoder().fit(labels)
    out["label"] = {
        "values": labels.tolist(),
        "classes": le.classes_.tolist(),
        "codes": le.transform(labels).tolist(),
    }

    codes = np.array([[0, 2], [1, 0], [2, 1], [0, 0], [1, 2], [2, 2]])
    oh = OneHotEncoder(sparse_output=False, categories=[[0, 1, 2], [0, 1, 2]]).fit(codes)
    out["onehot"] = {
        "codes": codes.tolist(),
        "levels": [3, 3],
        "apply": oh.transform(codes).tolist(),
    }

    # Missing values in a pattern, including a column missing in over half its
    # rows, so the statistic is computed on what is there and not on a default.
    gapped = x[:60, :4].copy()
    gapped[0, 0] = np.nan
    gapped[5:40, 1] = np.nan
    gapped[7, 2] = np.nan
    gapped[8, 2] = np.nan
    gapped[59, 3] = np.nan
    out["impute"] = {"gapped": [[None if np.isnan(v) else v for v in r] for r in gapped]}
    for how in ("mean", "median", "most_frequent"):
        im = SimpleImputer(strategy=how).fit(gapped)
        out["impute"][how] = {
            "statistic": im.statistics_.tolist(),
            "apply": im.transform(gapped).tolist(),
        }

    # train_test_split and the fold generators, pinned as INDICES. A
    # distributional check would pass for any shuffle; only the indices prove
    # the same rows were held back.
    out["split"] = {"n": n}
    for share in (0.25, 0.1, 0.333):
        tr, te = train_test_split(np.arange(n), test_size=share, shuffle=False)
        out["split"][f"sequential_{share}"] = {"train": tr.tolist(), "test": te.tolist()}
    out["folds"] = {}
    for k in (3, 5, 7):
        out["folds"][f"sequential_{k}"] = [
            {"train": tr.tolist(), "test": te.tolist()}
            for tr, te in KFold(n_splits=k, shuffle=False).split(np.arange(n))
        ]
    # Stratified on a deliberately imbalanced label — 8% positive, which is the
    # shape risk work actually has.
    y = (rng.random(n) < 0.08).astype(int)
    out["stratified"] = {
        "labels": y.tolist(),
        "folds": {
            f"sequential_{k}": [
                {"train": tr.tolist(), "test": te.tolist()}
                for tr, te in StratifiedKFold(n_splits=k, shuffle=False).split(
                    np.zeros((n, 1)), y
                )
            ]
            for k in (3, 5)
        },
    }
    dump("prepare.json", out)


def measure():
    """The metrics that matter when one class is rare.

    Ranking metrics are exact functions of the score ORDER, and the threshold
    metrics of one comparison, so these are closed forms too. The scores here
    are deliberately given ties and a duplicated value, because the tie is where
    a trapezoid and a step function part company and where a careless
    average-precision drifts from scikit-learn's.
    """
    rng = np.random.default_rng(SEED + 4)
    n = 500
    truth = (rng.random(n) < 0.07).astype(int)
    # A score that is informative but not perfect, then coarsened onto a grid so
    # that exact ties are common.
    score = 0.35 * truth + rng.normal(size=n) * 0.4
    score = np.round(score, 2)

    out = {
        "truth": truth.tolist(),
        "score": score.tolist(),
        "roc_auc": float(roc_auc_score(truth, score)),
        "average_precision": float(average_precision_score(truth, score)),
        "log_loss_clipped": None,
    }
    # drop_intermediate=False: sklearn's default drops collinear interior points to make a
    # lighter PLOT, which is a presentation choice and not the curve. Asking for the whole
    # curve is asking the oracle the same question hanzo-learn answers.
    fpr, tpr, thr = roc_curve(truth, score, drop_intermediate=False)
    out["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "thresholds": thr.tolist()}
    pr, rc, pthr = precision_recall_curve(truth, score)
    out["pr"] = {
        "precision": pr.tolist(),
        "recall": rc.tolist(),
        "thresholds": pthr.tolist(),
    }
    out["confusion"] = {}
    for t in (-0.5, 0.0, 0.2, 0.5):
        flagged = (score >= t).astype(int)
        # sklearn's layout is [[tn, fp], [fn, tp]]; recorded in that order.
        out["confusion"][str(t)] = confusion_matrix(
            truth, flagged, labels=[0, 1]
        ).ravel().tolist()

    # log loss needs probabilities, so a separate calibrated draw. Includes a 0
    # and a 1, which is where every implementation must clip and where they
    # differ if they clip differently.
    prob = np.clip(1.0 / (1.0 + np.exp(-(3.0 * truth - 1.5 + rng.normal(size=n)))), 0, 1)
    prob[0] = 0.0
    prob[1] = 1.0
    out["probability"] = prob.tolist()
    out["log_loss"] = float(log_loss(truth, prob, labels=[0, 1]))
    out["log_loss_clip"] = float(np.finfo(np.float64).eps)
    dump("measure.json", out)


def outlier():
    """Anomaly detection, and the two halves are pinned differently ON PURPOSE.

    LOCAL OUTLIER FACTOR is deterministic — no subsample, no cut, no seed — so
    it is pinned EXACTLY: negative_outlier_factor_ and score_samples to the last
    bit.

    ISOLATION FOREST is not a function of its data. Its trees depend on numpy's
    stream through scikit-learn's own sampler, which no independent
    implementation reproduces, so pinning our forest against their forest would
    be pinning a coincidence. What IS a function of the data is the SCORING of a
    GIVEN forest, so this exports scikit-learn's own trees — node arrays and
    all — and the Rust test walks them. That isolates the two claims: the
    scoring arithmetic is proved exactly right, and the tree BUILDER is held to
    a distributional bar instead of a false exact one.
    """
    rng = np.random.default_rng(SEED + 5)
    n, p = 300, 4
    x = design(rng, n, p)
    # Plant a handful of far-out rows: what an anomaly detector is for.
    x[:6] += rng.normal(size=(6, p)) * 0.5 + 9.0
    x_test = np.vstack([design(rng, 30, p), design(rng, 5, p) + 8.0])

    out = {"x": x.tolist(), "x_test": x_test.tolist(), "lof": {}}
    for k in (5, 20, 50):
        m = LocalOutlierFactor(n_neighbors=k, novelty=True).fit(x)
        out["lof"][str(k)] = {
            "negative_outlier_factor": m.negative_outlier_factor_.tolist(),
            "score_samples_test": m.score_samples(x_test).tolist(),
            "score_samples_train": m.score_samples(x).tolist(),
            "k_distance": m._distances_fit_X_[:, -1].tolist(),
            "lrd": m._lrd.tolist(),
        }

    f = IsolationForest(n_estimators=20, max_samples=128, random_state=0).fit(x)
    trees = []
    for est in f.estimators_:
        t = est.tree_
        trees.append(
            {
                "left": t.children_left.tolist(),
                "right": t.children_right.tolist(),
                "feature": t.feature.tolist(),
                "threshold": t.threshold.tolist(),
                "n_node_samples": t.n_node_samples.tolist(),
            }
        )
    out["forest"] = {
        "max_samples": int(f.max_samples_),
        "trees": trees,
        # sklearn's score_samples is -s(x); our Outlier::outlier is +s(x).
        "score_samples_test": f.score_samples(x_test).tolist(),
        "score_samples_train": f.score_samples(x).tolist(),
    }
    # The correction term, pinned at the sizes the trees actually reach.
    from sklearn.ensemble._iforest import _average_path_length

    sizes = [0, 1, 2, 3, 4, 7, 10, 64, 128, 256, 1000, 100000]
    out["average_path_length"] = {
        str(m): float(_average_path_length(np.array([m]))[0]) for m in sizes
    }
    dump("outlier.json", out)


if __name__ == "__main__":
    linear()
    logistic()
    boosted()
    prepare()
    measure()
    outlier()
