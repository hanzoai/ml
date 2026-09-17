# Naming: the literature, not the vendors

Vendor names are brands. `sklearn` is a Python library, `xgb` and `lgb` are two
implementations of one idea, and none of the three is a concept. The literature
already named every one of these, and ONNX already serialized those names — so
there is nothing to invent.

Three rules, in order of authority:

1. **The ONNX `ai.onnx.ml` operator name**, where one exists. That is the
   interchange standard, so it is what every other tool already agrees on.
2. **The originating paper's term**, where ONNX has no operator.
3. **What it does, in one word.** No compound words. Never a vendor.

## The decomplect: three brands are one algorithm

`XGBoost`, `LightGBM` and `CatBoost` are not three algorithms. They are three
implementations of **gradient boosting** (Friedman 2001), differing in histogram
binning, leaf-wise vs level-wise growth, and categorical handling — all of which
are *parameters*, not identities. ONNX agrees: all three export to the single
operator `TreeEnsembleClassifier`.

So the vendor axis collapses entirely:

| vendor name | the actual concept | ONNX operator |
|---|---|---|
| XGBoost, LightGBM, CatBoost | gradient boosting | `TreeEnsembleClassifier` / `Regressor` |
| RandomForest, ExtraTrees | bagging (Breiman 1996) + feature subsampling | `TreeEnsembleClassifier` / `Regressor` |
| LogisticRegression | logistic regression | `LinearClassifier` |
| LinearRegression | ordinary least squares | `LinearRegressor` |
| Ridge | Tikhonov regularization | `LinearRegressor` |
| SVC / SVR | support vector machine | `SVMClassifier` / `SVMRegressor` |
| StandardScaler | standardization (z-score) | `Scaler` |
| MinMaxScaler | min-max normalization | `Scaler` |
| Normalizer | vector normalization | `Normalizer` |
| SimpleImputer | imputation | `Imputer` |
| OneHotEncoder | indicator coding (statistics: dummy coding) | `OneHotEncoder` |
| LabelEncoder | label encoding | `LabelEncoder` |
| KMeans | Lloyd's algorithm / k-means | — |
| DBSCAN | density-based clustering | — |
| PCA | principal component analysis | — |
| IsolationForest | isolation forest (Liu, Ting, Zhou 2008) | — |
| LocalOutlierFactor | local outlier factor (Breunig 2000) | — |

Note what risk already runs: **half-space mass** — streaming HS-Trees
(Tan, Ting, Liu 2011). Its own source already uses the literature term. That is
the standard to hold; `apps/risk` got this right before we did.

## Curry: the task is the type

sklearn organizes by MODULE — `linear_model`, `ensemble`, `tree`, `svm`. That is
a *place*. The literature organizes by what is learned, which is a *value*:

| what is learned | returns | examples |
|---|---|---|
| a decision function | a label | logistic, boosting, kernel machine |
| a conditional expectation | a real | least squares, ridge, boosting |
| a score | a deviation | isolation, half-space mass, local outlier factor |
| a partition | an assignment | k-means, density clustering |
| a basis | a projection | principal components |

So `predict` is NOT one signature. A classifier returns a label, a regressor
returns a real, a density model returns a score. Making those one method with
one return type is exactly the complecting to avoid — and ONNX already agrees,
which is why `TreeEnsembleClassifier` and `TreeEnsembleRegressor` are two
operators and not one with a flag.

**Illegal states, unrepresentable:** an unfitted model has no `predict` at all —
not a `check_is_fitted` that raises. Fitting is `Config -> Data -> Model`, and
`Model` is a value with a content address (the pattern `cloud/apps/risk/address.go`
already uses), so a fitted value cannot be confused with one from different
hyperparameters.

## Pike: two verbs, discovered from use

`fit` and `predict`. That is the whole interface. Do not import Python's object
model along with its algorithms — no `get_params`, no `**kwargs`, no `Pipeline`
class before there is a pipeline to run.

## Module names: one word, no compounds

```
boosting     bagging      forest       logistic     squares      ridge
kernel       neighbors    components   clusters     isolation    mass
standardize  normalize    impute       encode       split        metrics
```

Each is the concept, spelled the way the literature spells it. `gradient_boosting`
is a compound; `boosting` is what it is. `RandomForestClassifier` is three
concepts welded together; `forest` plus the task type says the same thing without
the weld.

## The payoff

Because the names ARE the ONNX operators, the port and the interchange layer are
one effort. `fit` gives training; the operator gives interop with every model
anyone already exported. And `TreeEnsembleClassifier` covers gradient boosting,
random forests and extra trees — three vendor ecosystems, one implementation.
