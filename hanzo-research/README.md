# hanzo-research

Rust client for `api.hanzo.ai/v1/research`: record an experiment as a claim, its observations
and a verdict, with the provenance filled in for you.

```rust
use hanzo_research::{Research, Verdict};

let r = Research::from_env(); // HANZO_API_KEY, RESEARCH_BASE, RESEARCH_PROJECT
let mut run = r
    .experiment("kernel-perf", "rms_norm_blk", "vulkan/4096x4096")
    .metric("ratio_vs_hand")
    .hypothesis("the DSL kernel is at least as fast as the hand-written one")
    .predict("DSL/hand >= 1.0");
run.log("cold, 3 runs, max error 4.5e-6 against the reference");
run.conclude(Verdict::Proven, "measured ratio", Some(ratio))?;
```

- **Verbs**: `experiment` (get or create by `kind:subject:task`), `log`, `record` (one scored
  item), `snapshot` and `report` (stored artifacts), `conclude` (`Proven`, `Refuted` or
  `Inconclusive`) and `finish`; `query`, `totals` and `ingest` read and bulk-write.
- **Provenance**: the first call stamps the git sha, branch and dirty flag, the commit messages
  since the experiment's last run, the resolved versions of the hanzo crates from `Cargo.lock`,
  and the host.
- **Auth**: `Authorization: Bearer $HANZO_API_KEY`. The org and project come from the key.
  `RESEARCH_BASE` overrides `https://api.hanzo.ai`; records are private by default.

`cargo run -p hanzo-research --example kernel_perf` runs the flow against an in-process mock and
prints the records it sends.

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
