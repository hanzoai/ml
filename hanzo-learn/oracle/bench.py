#!/usr/bin/env python3
"""The scikit-learn half of the benchmark in `benches/learn.rs`.

WHY THIS IS COMPARABLE, which is the only interesting thing about a benchmark:

  SAME DATA, provably. Both sides draw from numpy's legacy Mersenne stream at the
  same seed. `hanzo_learn::twister::Twister::next_real` IS
  `RandomState.random_sample` bit for bit — `twister`'s own unit test pins it
  against numpy's output — so the two benchmarks see identical VALUES and not
  merely the same distribution. That matters for the estimators whose cost
  depends on the values rather than on the shape: a tree's depth and a neighbour
  frontier's insert rate both do.

  SAME WORK. Each case below is the scikit-learn call that answers the same
  question as the Rust one, at the same shape, with the parameters matched. Where
  scikit-learn does MORE work by default the difference is named in the notes
  column of `README.md` rather than quietly left in the number.

  SAME MEASUREMENT. Median of `REPEATS` timed runs after one warm-up, wall clock,
  `perf_counter`. criterion reports a median too, so the two columns are the same
  statistic. Allocation of inputs is outside the timed region on both sides.

  ONE CORE, DECLARED. scikit-learn's IsolationForest and LocalOutlierFactor take
  `n_jobs`; this runs them at the default (1) and hanzo-learn's forest fit spans
  cores through rayon. That is a REAL difference and it is reported as one — the
  Rust fit column says how many threads it had, and the `n_jobs=-1` row is
  measured too so the single-core comparison is available.

Run:
    <venv>/bin/python hanzo-learn/oracle/bench.py           # the default shapes
    <venv>/bin/python hanzo-learn/oracle/bench.py --quick   # skip 1m rows
"""

import argparse
import json
import platform
import sys
import time

import numpy as np
import sklearn
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
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

REPEATS = 5

SHAPES = [("10k_x_20", 10_000, 20), ("100k_x_20", 100_000, 20), ("1m_x_20", 1_000_000, 20)]
NEAR_SHAPES = [("2k_x_20", 2_000, 20), ("20k_x_20", 20_000, 20)]


def design(n, p, seed):
    """Bit-identical to `benches/learn.rs::design`."""
    return np.random.RandomState(seed).random_sample(n * p).reshape(n, p) * 2.0 - 1.0


def planted(n, p, seed):
    """Bit-identical to `benches/learn.rs::planted`."""
    x = design(n, p, seed)
    x[: max(n // 100, 1), :] += 9.0
    return x


def time_it(fn):
    """Median of REPEATS runs, in seconds, after one untimed warm-up."""
    fn()
    took = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()
        took.append(time.perf_counter() - start)
    return float(np.median(took))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip the 1m-row shapes")
    args = ap.parse_args()
    shapes = SHAPES[:-1] if args.quick else SHAPES

    out = {
        "oracle": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "repeats": REPEATS,
        "cases": {},
    }

    def record(group, case, seconds):
        out["cases"].setdefault(group, {})[case] = seconds
        print(f"  {group:11s} {case:32s} {seconds * 1e3:12.3f} ms")

    print(f"scikit-learn {sklearn.__version__} / numpy {np.__version__}, median of {REPEATS}")

    for name, n, p in shapes:
        x = design(n, p, 1)
        record("scale", f"standard_fit/{name}", time_it(lambda: StandardScaler().fit(x)))
        fitted = StandardScaler().fit(x)
        record("scale", f"standard_apply/{name}", time_it(lambda: fitted.transform(x)))
        record("scale", f"range_fit/{name}", time_it(lambda: MinMaxScaler().fit(x)))
        ranged = MinMaxScaler().fit(x)
        record("scale", f"range_apply/{name}", time_it(lambda: ranged.transform(x)))

    for name, n, p in shapes:
        x = design(n, p, 2)
        # Stride 19 against 20 columns, coprime, so the gaps walk across every column. A
        # stride of p would empty ONE column entirely; scikit-learn then drops it with a
        # warning and the two sides are no longer doing the same work.
        gapped = x.copy().ravel()
        gapped[np.arange(gapped.size) % 19 == 7] = np.nan
        gapped = gapped.reshape(n, p)
        record(
            "impute",
            f"mean_fit/{name}",
            time_it(lambda: SimpleImputer(strategy="mean").fit(gapped)),
        )
        record(
            "impute",
            f"median_fit/{name}",
            time_it(lambda: SimpleImputer(strategy="median").fit(gapped)),
        )
        fill = SimpleImputer(strategy="mean").fit(gapped)
        record("impute", f"apply/{name}", time_it(lambda: fill.transform(gapped)))

    for name, n, _ in shapes:
        raw = np.random.RandomState(3).randint(0, 10, size=(n, 8))
        flat = raw.ravel()
        record("encode", f"label_fit/{name}", time_it(lambda: LabelEncoder().fit(flat)))
        le = LabelEncoder().fit(flat)
        record("encode", f"label_codes/{name}", time_it(lambda: le.transform(flat)))
        oh = OneHotEncoder(sparse_output=False, categories=[list(range(10))] * 8).fit(raw)
        record("encode", f"onehot_apply/{name}", time_it(lambda: oh.transform(raw)))

    for name, n, _ in shapes:
        index = np.arange(n)
        y = (np.random.RandomState(4).random_sample(n) < 0.08).astype(int)
        record(
            "split",
            f"train_test/{name}",
            time_it(lambda: train_test_split(index, test_size=0.25, random_state=0)),
        )
        record(
            "split",
            f"kfold_5/{name}",
            time_it(lambda: list(KFold(n_splits=5, shuffle=True, random_state=0).split(index))),
        )
        zeros = np.zeros((n, 1))
        record(
            "split",
            f"stratified_5/{name}",
            time_it(
                lambda: list(
                    StratifiedKFold(n_splits=5, shuffle=True, random_state=0).split(zeros, y)
                )
            ),
        )

    for name, n, _ in shapes:
        r = np.random.RandomState(5)
        truth = (r.random_sample(n) < 0.07).astype(int)
        score = np.where(truth == 1, 0.35, 0.0) + r.random_sample(n)
        # `Curve::of` builds the whole confusion ladder once; roc_curve is the
        # nearest single scikit-learn call that does the same accumulation.
        record(
            "metric",
            f"curve_of/{name}",
            time_it(lambda: roc_curve(truth, score, drop_intermediate=False)),
        )
        record("metric", f"roc_auc/{name}", time_it(lambda: roc_auc_score(truth, score)))
        record(
            "metric",
            f"average_precision/{name}",
            time_it(lambda: average_precision_score(truth, score)),
        )
        record(
            "metric",
            f"precision_recall/{name}",
            time_it(lambda: precision_recall_curve(truth, score)),
        )
        # log_loss needs a probability; the score is squashed so the call is legal.
        prob = 1.0 / (1.0 + np.exp(-score))
        record("metric", f"log_loss/{name}", time_it(lambda: log_loss(truth, prob, labels=[0, 1])))

    for name, n, p in shapes:
        x = planted(n, p, 6)
        record(
            "isolation",
            f"fit_100_trees/{name}",
            time_it(lambda: IsolationForest(n_estimators=100, max_samples=256).fit(x)),
        )
        record(
            "isolation",
            f"fit_100_trees_njobs-1/{name}",
            time_it(lambda: IsolationForest(n_estimators=100, max_samples=256, n_jobs=-1).fit(x)),
        )
        forest = IsolationForest(n_estimators=100, max_samples=256).fit(x)
        record("isolation", f"outlier/{name}", time_it(lambda: forest.score_samples(x)))

    for name, n, p in NEAR_SHAPES:
        x = planted(n, p, 7)
        record(
            "neighbour",
            f"fit_k20/{name}",
            time_it(lambda: LocalOutlierFactor(n_neighbors=20, novelty=True).fit(x)),
        )
        record(
            "neighbour",
            f"fit_k20_njobs-1/{name}",
            time_it(lambda: LocalOutlierFactor(n_neighbors=20, novelty=True, n_jobs=-1).fit(x)),
        )
        fitted = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(x)
        query = design(1_000, p, 8)
        record(
            "neighbour",
            f"outlier_1k_queries/{name}",
            time_it(lambda: fitted.score_samples(query)),
        )

    path = __file__.rsplit("/", 1)[0] + "/sklearn.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1, sort_keys=True)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
