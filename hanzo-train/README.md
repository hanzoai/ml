# hanzo-train

Training on [hanzo-ml](https://crates.io/crates/hanzo-ml) across machines that do not match:
Metal, CUDA, ROCm and CPU workers training one model.

- **`cluster`**: local SGD with an outer Nesterov step (DiLoCo). A coordinator owns the plan,
  the F32 global weights, the outer step and checkpoints. Workers join at any time, take batches
  one at a time, and send their weight deltas in BF16 with error feedback. Deltas sum in join
  order, so a run equals its rounds replayed in one process (`cluster::replay`) bit for bit, and
  a checkpoint resumes bit-identically. A worker that drops has its batches requeued; a round
  waits for no one past its deadline plus a grace period. A model plugs in behind
  `cluster::Model`.
- **`adam`**: AdamW with a learning rate per parameter and its moments in the open, on
  hanzo-nn's `optim::adamw` update.
- **`gpu`**: a floor on free memory, below which training exits, and GPU utilization on macOS.
- **`dspark`**: the DSpark speculative-draft model trained on the cluster runtime. It writes a
  checkpoint in the layout Hanzo Engine's DSpark loader reads.

```bash
hanzo-train fit  --listen 0.0.0.0:7000 ...   # coordinate a run and train in it
hanzo-train join coordinator:7000            # train as a worker; rejoins if the link drops
cargo test -p hanzo-train                    # tiny classifier and synthetic DSpark cache, CPU only
```

Not yet published; use it from git: `hanzo-train = { git = "https://github.com/hanzoai/ml" }`.

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
