# `hanzo-seneca`

Clean-room, zero-external-dependency Rust (`std`-only, `#![forbid(unsafe_code)]`) implementation of **Seneca** multi-stage training data-pipeline cache allocation across **Encoded**, **Decoded**, and **Transformed** stages for concurrent training jobs (tracked by **HIP-1349** / `hanzoai/ml#46`).

## Features

- **3-Stage Data Pipeline Execution Engine** (`src/pipeline.rs`): Real LZ/RLE byte compression & decompression with IEEE CRC32 checksums (`src/codec.rs`), multi-modal tensor/token decoding, and deterministic/stochastic augmentation over real `hanzoai/ml` repository corpus shards.
- **Bounded Multi-Stage, Multi-Tenant LRU Cache** (`src/cache.rs`): Enforces `total_used_bytes <= total_budget_bytes` at every instant (shrink-first, expand-second during rebalances), strict per-tenant/per-job isolation (`0` cross-tenant leaks), and content-addressed sharing strictly for `SharedPublic` datasets.
- **Seneca Marginal-Utility Dynamic Partitioner** (`src/allocator.rs`): Maximizes marginal GPU-stall reduction per byte across `{Encoded, Decoded, Transformed} × concurrent jobs`, with smooth hysteresis, stochastic-augmentation freshness enforcement (`Transformed = 0` for stochastic jobs), opportunistic cache-aware sampling, and streaming bypass for zero-reuse 1-epoch workloads.
- **5 Matched Policies**: `no_cache`, `encoded_only`, `decoded_only`, `static_equal_split`, and `seneca_dynamic`.

## Quick Start

```bash
# Run unit & integration tests (offline, < 3s)
cargo test -p hanzo-seneca --release --offline

# Run the reproducible 4-scenario benchmark (--seed 1349)
cargo run -p hanzo-seneca --release --offline --bin seneca-bench -- --seed 1349
```

Full research dossier, license & patent audit, methodology, raw measurements, and integration plan live in [`docs/research/hip-1349-seneca/README.md`](../docs/research/hip-1349-seneca/README.md).
