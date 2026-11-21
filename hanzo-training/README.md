# hanzo-training

GRPO (group relative policy optimization) on [hanzo-ml](https://crates.io/crates/hanzo-ml)
autograd, plus a training-config format.

```toml
[dependencies]
hanzo-training = "0.11"
```

## GRPO

`grpo` is the connected training path. One `GrpoTrainer::step`:

1. samples a group of `G` completions per prompt from the current policy;
2. scores each with one or more `Reward`s (`ExactMatchReward`, `TokenMatchReward`,
   `LengthTargetReward`, or your own);
3. computes the group-relative advantage `Aᵢ = (rᵢ − mean(r)) / (std(r) + eps)`;
4. forms the loss `−mean(Aᵢ · logprob(completionᵢ))`, with an optional KL penalty to a frozen
   reference policy;
5. takes an AdamW step (`hanzo_nn::AdamW`).

A model plugs in by implementing `Policy`: `trainable_vars()`, `sample_group()` (its generation
loop, through the provided `Sampler`) and `sequence_logprob()` (a differentiable teacher-forced
pass). `examples/grpo_toy.rs` trains a toy policy end to end on the CPU:

```bash
cargo run -p hanzo-training --example grpo_toy
```

## Configs and the CLI

`TrainingConfig::from_file` reads YAML, JSON or TOML; `configs/` has examples. The
`hanzo-training` binary runs `validate` and `template` on those files. `train`, the
config-driven dataset loaders (`BasicDataset`, `ZenAgenticDataset`, `ZenIdentityDataset`) and the
`hanzo-eval` benchmarks are not yet connected to a model: they return an error that says so,
rather than report numbers nothing computed. `JsonlDataset` loads `{"input": ..., "output": ...}`
lines.

Features `cuda` and `metal` forward to hanzo-ml and hanzo-nn.

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
