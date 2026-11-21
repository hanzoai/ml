# hanzo-onnx

Evaluate ONNX models on [hanzo-ml](https://crates.io/crates/hanzo-ml).

```rust
let model = hanzo_onnx::read_file("model.onnx")?;
let outputs = hanzo_onnx::simple_eval(&model, inputs)?; // HashMap<String, Tensor> in, HashMap<String, Value> out
```

Two operator domains:

- `ai.onnx`: the neural-network operators (convolution, attention, activations, shape ops).
- `ai.onnx.ml`: the classical ones, which carry their whole model in their attributes:
  `TreeEnsembleClassifier`/`Regressor`, `LinearClassifier`/`Regressor`, `SVMClassifier`/
  `Regressor`, `Scaler`, `Normalizer`, `Imputer`, `Binarizer`, `OneHotEncoder`,
  `LabelEncoder`, `CategoryMapper`, `ArrayFeatureExtractor`, `FeatureVectorizer` and `ZipMap`.
  Where the specification is loose, behaviour was measured against onnxruntime 1.28.0 and is
  checked in `tests/ml.rs`. `CastMap` and `DictVectorizer` are refused by name.

## Building

The build runs `prost-build`, which needs `protoc` on `PATH` (e.g.
`apt-get install protobuf-compiler`).

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
