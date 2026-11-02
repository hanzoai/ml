# Hanzo ML for Python

The Python bindings for the public `hanzo-ml` tensor library. Distribution: `hanzo-ml`; import: `hanzo_ml`. The extension is `hanzo_ml._native`. This library contains generic tensor operations; application model weights are separate.

Build from a checkout of hanzoai/ml:

```sh
python -m pip install maturin
RUSTFLAGS="-C target-cpu=native" maturin build --release --manifest-path hanzo-ml-pyo3/Cargo.toml
python -m pip install target/wheels/hanzo_ml-*.whl
```

```python
import hanzo_ml as ml
features = (ml.Tensor([10.0, 20.0, 0.0, 0.0]) / 100).values()
print(features)
```

Python and Rust share the same tensor implementation. The binding exposes `Tensor`, dtype constants, factory functions, and `functional` operations. Packaging changes do not publish a PyPI release automatically; use a built wheel until a release is confirmed.
