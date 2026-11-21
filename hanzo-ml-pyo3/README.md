# hanzo-ml-pyo3

Python bindings for [hanzo-ml](https://github.com/hanzoai/ml): tensors, `nn` modules, ONNX
evaluation and utilities, as the Python module `hanzo`. Built locally with
[maturin](https://www.maturin.rs); not published.

```bash
cd hanzo-ml-pyo3
maturin develop -r      # inside a virtual env
python test.py
```

## Type stubs

The package ships `*.pyi` stubs, generated from the built module:

```bash
python stub.py          # regenerate
python stub.py --check  # fail if the stubs differ from the module
```

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
