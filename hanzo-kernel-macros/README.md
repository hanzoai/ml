# hanzo-kernel-macros

The attributes behind [hanzo-kernel](https://crates.io/crates/hanzo-kernel). Use them through
`hanzo_kernel::prelude::*`, not directly.

- `#[kernel(targets(...))]` marks a launchable kernel. Targets are `cpu`, `cuda`, `rocm`, `metal`,
  `vulkan` and `webgpu`; any other name, or an empty list, is a compile error. Add `unchecked` to
  generate `launch_unchecked` (no bounds checks) instead of `launch`.
- `#[device]` marks a helper that is inlined into kernels.
- Inside a kernel, `island! { <targets> => { ... } default => { ... } }` becomes a compile-time
  `match` on the kernel's `#[comptime] Target` parameter, so each backend compiles only its own
  arm. `default` is required.

## License

MIT OR Apache-2.0. See [NOTICE](https://github.com/hanzoai/ml/blob/main/NOTICE) for
third-party attributions.
