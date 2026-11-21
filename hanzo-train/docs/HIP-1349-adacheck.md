# HIP-1349 Research and Bounded Implementation: Redundancy-Aware Training Checkpoints (AdaCheck)

## 1. Primary Reference and License Audit

- Primary paper: Weijie Liu et al., "AdaCheck: Redundancy-Aware Training Checkpoints", USENIX FAST 2026 (`https://www.usenix.org/conference/fast26/presentation/liu-weijie`).
- Tracking issue: `hanzoai/ml#45` (HIP-1349).
- License audit: 100% clean-room Rust (`MIT OR Apache-2.0`) authored directly against `hanzo-train/src/cluster` using only workspace crates already declared in `hanzo-train/Cargo.toml` (`sha2`, `serde`, `serde_json`, `anyhow`, `rayon`, `std`). Zero external paper code, datasets, or patent-encumbered artifacts were imported.

## 2. Upstream Baseline

In `hanzo-train/src/cluster/coordinator.rs`, every checkpoint boundary writes `model.safetensors` (`theta` in F32), `state.safetensors` (`momentum`, `carry`, `merged`, `round`), and `inner/{worker}.safetensors` (`state` of `3 * N` F32 values plus step `t`) for every worker that has joined the DiLoCo run. Across successive checkpoints and heterogeneous workers, full serialization rewrites unchanged bytes from:

1. Straggler, paused, or dropped workers whose `3 * N` inner AdamW buffers did not advance in the latest round,
2. Zero-initialized or inactive parameter chunks (`carry`, early `momentum`, frozen or layer-selective blocks), and
3. Identical initial or synchronized chunks across workers.

## 3. Falsifiable Hypothesis and Failure Modes

- Hypothesis: Content-addressed fixed-chunk SHA-256 deduplication across `theta`, outer `momentum`/`carry`, and per-worker `inner` buffers cuts cumulative checkpoint storage and I/O by `> 2.2x` under heterogeneous/staggered DiLoCo training while preserving exact IEEE-754 bit-for-bit resume (`f32::to_bits()`), whereas under dense 100%-mutating training across all workers temporal deduplication collapses toward `1.00x`.
- Expected failure mode (confirmed by negative control): When 100% of parameters and moments mutate every round across every worker with non-zero `carry`, chunk digests miss completely (`1.00x` deduplication ratio) and pay SHA-256 hashing overhead without space savings.

## 4. Bounded Implementation (`hanzo-train`)

- `hanzo-train/src/cluster/adacheck.rs` (exported via `hanzo_train::cluster::AdaCheckStore`):
  - Splits `f32` slices into aligned little-endian IEEE-754 chunks, hashes chunks in parallel via `rayon::par_chunks` + `Sha256`, and writes unique blobs once to `blobs/<prefix>/<sha256>.bin` via temp-file + `sync_all` + atomic `fs::rename`.
  - Restores `Start` bit-for-bit (`f32::from_bits(u32::from_le_bytes(...))`, preserving `-0.0` and subnormals) while verifying every referenced blob's SHA-256 digest before assembly.
  - Rejects path-traversal worker names (`..`, `/`, `\`, NUL) and malformed digests.

## 5. Correctness, Concurrency, Security, and Regression Evidence

- `cargo test -p hanzo-train` passes `10/10` library unit tests, `3/3` `adacheck` tests, `dspark` integration tests, and `11/11` `cluster` DiLoCo TCP integration tests (`0` failures).
- Verified signed-zero (`-0.0` vs `+0.0`) and subnormal bit preservation, 8-thread concurrent checkpoint saves into a shared blob store, path-traversal rejection, and fail-closed SHA-256 bit-flip corruption detection on restore.

## 6. Before and After Measurements

Reproduce locally with `cargo test -p hanzo-train --test adacheck -- --nocapture` (`5` successive checkpoints, `4` workers, `N = 262,144` parameters):

| Regime | Logical Baseline (`safetensors` raw state) | Physical `AdaCheckStore` Blobs | Deduplication Ratio | Avg Save Time | Avg Verified Restore Time |
|---|---|---|---|---|---|
| Heterogeneous DiLoCo (`2/4` active workers, `50%` active params) | `75.00 MiB` | `29.52 MiB` | `2.54x` | `2998.0 ms` | `1809.2 ms` |
| Dense Full-Mutation Negative Control (`4/4` active workers, `100%` params) | `75.00 MiB` | `75.00 MiB` | `1.00x` | `5611.9 ms` | `1823.9 ms` |

### Integration Recommendation

Keep `AdaCheckStore` opt-in alongside `Start::resume` for heterogeneous multi-node DiLoCo and layer-selective / adapter runs where worker stragglers or frozen blocks create high chunk reuse (`> 2x`), and bypass chunk hashing on synchronous dense full-parameter runs where reuse approaches `1.00x`. Canonical `vfs -> Hanzo S3 -> Visor` storage boundaries remain unchanged.
