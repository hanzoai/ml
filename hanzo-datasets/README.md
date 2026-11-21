# hanzo-datasets

Dataset loaders and a batcher for [hanzo-ml](https://crates.io/crates/hanzo-ml).

| Module | Loads |
|---|---|
| `vision::mnist` | MNIST: `load()` fetches the parquet files from the dataset hub, `load_dir(dir)` reads the raw files |
| `vision::fashion_mnist` | Fashion-MNIST: `load()` |
| `vision::cifar` | CIFAR-10: `load()` from the hub, `load_dir(dir)` from the binary files |
| `nlp::tinystories` | TinyStories, pre-tokenized, with a sequence-window iterator |
| `batcher` | `Batcher` stacks an iterator of tensors, or tensor pairs, into batches |
| `hub` | `from_hub(api, dataset_id)`: a dataset's parquet files as readers |

```rust
let m = hanzo_datasets::vision::mnist::load()?;
println!("{:?} {:?}", m.train_images.shape(), m.train_labels.shape());
```

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
