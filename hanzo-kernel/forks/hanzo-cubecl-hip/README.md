# hanzo-cubecl-hip

The AMD ROCm HIP runtime for CubeCL.

This is a **fork of [`cubecl-hip`](https://crates.io/crates/cubecl-hip) `0.10.0`** (Tracel
Technologies Inc. / the CubeCL Framework Contributors, <https://github.com/tracel-ai/cubecl>),
redistributed under the original `MIT OR Apache-2.0` license. See `NOTICE`, `LICENSE-MIT`, and
`LICENSE-APACHE`.

## Why this fork exists

Upstream `cubecl-hip` depends on `cubecl-hip-sys`, whose `build.rs` cannot resolve bindings on ROCm
>= 7.13 (the detected HIP patch is newer than any it ships — see `hanzo-cubecl-hip-sys`). That makes
the HIP backend unbuildable from crates.io on current ROCm without a local `[patch.crates-io]`.

## The only change

The `cubecl-hip-sys` dependency is repointed to **`hanzo-cubecl-hip-sys`** — a fork of
`cubecl-hip-sys` that adds a ROCm 7.13+ `build.rs` fallback — via a package rename:

```toml
cubecl-hip-sys = { package = "hanzo-cubecl-hip-sys", version = "7.2.5321101" }
```

The dependency **key** stays `cubecl-hip-sys`, so every `use cubecl_hip_sys::...` in `src/` resolves
unchanged; only the crate behind it changes. The library target is renamed to `hanzo_cubecl_hip` so
downstreams import it as `hanzo_cubecl_hip` (e.g. `hanzo-kernel`'s `rocm` feature). **All Rust source
under `src/` is verbatim from upstream `cubecl-hip` 0.10.0.** No upstream authorship is claimed.

## Version

`0.10.0` mirrors the upstream `cubecl-hip` release this forks; the fork carries no source change
beyond the manifest, so the version stays aligned. Future fork-only fixes bump the patch (`0.10.1`,
`0.10.2`, ...).

## Upstream

The underlying `build.rs` fix has been offered upstream against `cubecl-hip-sys`; see `NOTICE` for
the PR link when filed. If upstream ships the fallback, `hanzo-kernel` can return to depending on the
`cubecl` facade's `hip` feature directly and both forks can be retired.
