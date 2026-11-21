# hanzo-learn

Classical statistical learning in pure Rust. A config is a value, fitting turns it into a model
value, and predicting is a query against that value.

```rust
use hanzo_learn::{Fit, Model, Predict};

let model = config.fit(&samples)?; // Config -> Samples -> Model
let answers = model.predict(&x)?;  // f64 for a regressor, a class for a classifier
let name = model.address();        // a content hash of shape, hyperparameters and parameters
```

| Module | Holds |
|---|---|
| `linear` | least squares, by Householder QR |
| `logistic` | binary logistic regression |
| `tree`, `boost` | CART regression trees and gradient-boosted trees |
| `isolation`, `neighbour` | isolation forest and local outlier factor |
| `scale`, `impute`, `encode` | z-score and min-max scaling, imputation, label and one-hot encoding |
| `split`, `metric`, `moment` | resampling plans, scores for rare classes, combinable sufficient statistics |
| `twister` | the Mersenne twister, drawing exactly as numpy does |

An unfitted model has no `predict`: a model value exists only as the output of `Fit::fit`.
Numerics are matched to scikit-learn's behaviour through fixtures its own code produced
(`tests/sklearn.rs`); no scikit-learn source is reproduced. Multi-class logistic regression,
sparse matrices, sample weights, cross validation and pipelines are not here, and each is
refused with a specific error rather than approximated.

Not yet published; use it from git: `hanzo-learn = { git = "https://github.com/hanzoai/ml" }`.

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
