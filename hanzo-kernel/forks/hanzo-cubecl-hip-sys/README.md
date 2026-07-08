# hanzo-cubecl-hip-sys

Rust system bindings for the AMD ROCm HIP runtime (hiprtc, amdhip64) used by CubeCL.

This is a **fork of [`cubecl-hip-sys`](https://crates.io/crates/cubecl-hip-sys) `7.2.5321100`**
(Tracel Technologies Inc., <https://github.com/tracel-ai/cubecl-hip>), redistributed under the
original `MIT OR Apache-2.0` license. See `NOTICE`, `LICENSE-MIT`, and `LICENSE-APACHE`.

## Why this fork exists

Upstream `cubecl-hip-sys` ships bindgen bindings keyed by HIP **patch** version and its `build.rs`
selects the module named `hip_<patch>` for whatever `hipconfig --version` reports. ROCm **7.13**
(gfx1151 tech preview) reports HIP patch **`99004`**, which is *newer* than any bindings the crate
ships (newest is `53211`, ROCm 7.2.1). Upstream then emits `cargo:rustc-cfg=feature="hip_99004"`,
which has no bindings module, so every `cubecl_hip_sys` symbol is undefined and `cubecl-hip` fails to
build on ROCm >= 7.3.

## The only change

`build.rs` `set_hip_feature`: when the detected HIP patch has no matching `hip_<patch>` bindings,
fall back to the **latest** shipped bindings (currently `hip_53211`) instead of an undefined feature.
This mirrors the crate's existing no-`hipconfig` branch (which already clamps to the latest
bindings). Two small helpers (`extract_latest_hip_feature_from_path`, `hip_feature_available`, with
unit tests) live in `src/hipconfig.rs` to support it.

**Why it is ABI-safe:** AMD versions every changed HIP symbol (the `R0600` revisions —
`hipGetDevicePropertiesR0600`, `hipDeviceProp_tR0600`, ...). ROCm 7.13's headers still map the
wrapped entry points to their `R0600` revision, so the latest (`53211`) bindings link against the
newer runtime unchanged. Proven end-to-end: a CubeCL DSL kernel device-query + module-load + launch +
readback runs bit-exact on real gfx1151 hardware under ROCm 7.13.

All other files are **verbatim from upstream**; no upstream authorship is claimed.

## Versioning

Upstream's scheme is HIP-patch-based: `7.2.5321100` = the first release of the bindings for HIP
patch `53211` (ROCm 7.2.1), with the two trailing digits a monotonic fix-release counter for the
*same* patch bindings. This fork still ships exactly those `53211` bindings — it only adds a
`build.rs` fix — so it is published as **`7.2.5321101`** (the second release of the `53211`
bindings). This keeps `^`-resolution sane for consumers: `hanzo-cubecl-hip` depends on
`>= 7.2.5321101, < 8.0.0`, and future fork fixes bump `...02`, `...03`, ....

## Upstream

The `build.rs` fallback has been offered upstream:
<https://github.com/tracel-ai/cubecl-hip-sys/pull/35>. If it merges, this fork can be retired.
