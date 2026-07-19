//! hanzo-kernel: Hanzo's first-party GPU kernel DSL.
//!
//! Write a kernel ONCE in Rust; it lowers to CUDA / ROCm / Vulkan / Metal. This crate is the
//! first-party facade over the CubeCL lowering engine: kernel source names only `hanzo_kernel::*`,
//! never `cubecl`. "Values, not places" -- CubeCL is the implementation value; `hanzo_kernel` is the
//! stable namespace we build against, so the engine can be upgraded (or forked) without touching a
//! single kernel.
//!
//! The perf primitives our hand-tuned kernels rely on are all here and lower to the native
//! instruction on each backend:
//!   - `dot` on a vectorized `Line<i8>` -> dp4a / OpSDotAccSat (int8 4-way dot)
//!   - `cmma` -> tensor cores (WMMA / cooperative-matrix / simdgroup)
//!   - `SharedMemory`, `plane_*` (subgroup reduce/shuffle), `Atomic`, barriers
//!
//! MIGRATION POLICY (perf-gated, never a downgrade): the DSL provides COVERAGE -- every quant type on
//! every backend, from one source, killing the "same op, N impls, N numeric behaviors" bug class. A
//! hand-tuned kernel is replaced by its DSL twin ONLY when the DSL version is bit-exact AND within perf
//! noise (bench-gated). Where a hand-tuned kernel still wins, it stays as the specialized peak path and
//! the DSL is the portable fallback for the backends that lack a tuned version.

/// The lowering engine, behind a stable path so `#[kernel]` never has to name `cubecl` and neither
/// does kernel source. Values, not places: the engine can be swapped without touching a kernel.
pub mod flash;
pub mod engine {
    pub use cubecl::cube;
}

/// The kernel-authoring surface. Kernels write `use hanzo_kernel::prelude::*;` and `#[kernel(...)]`.
///
/// First-principles names: `#[kernel]` says what it is (a GPU kernel), not `cube` (a brand). `Grid`
/// and `Block` name the launch shape directly -- the grid of thread blocks, and the block of threads.
/// `Array`, `Float`, `Line` are kept: they are already the simple, accurate word for the thing.
pub mod prelude {
    pub use cubecl::prelude::*;
    pub use hanzo_kernel_macros::{device, kernel};
    pub use cubecl::CubeCount as Grid;
    pub use cubecl::CubeDim as Block;
    // The intrinsic-island tag. `#[kernel]` rewrites `island! { ... }` into a comptime match over a
    // `Target` param, so `Target` must be in scope wherever islands are authored — the prelude carries
    // it exactly like `Array`/`Float`, the one import a kernel source needs.
    pub use crate::island::Target;
    // Internal: `#[kernel]`/`#[device]` expand to `#[cube(...)]`, resolved via this glob. Kernel
    // *source* only writes `#[kernel]` / `#[device]`; `cube` is never named by a human.
    #[doc(hidden)]
    pub use cubecl::cube;
    // Internal: the `#[cube]` expansion emits `cubecl::` paths, which must resolve in a DOWNSTREAM
    // crate that depends only on `hanzo-kernel` (not `cubecl`). Re-export the crate through the prelude
    // so `use hanzo_kernel::prelude::*;` brings the `cubecl` name into scope. Never named by a human.
    #[doc(hidden)]
    pub use cubecl;
}

pub use cubecl;

pub mod tune;

pub mod quant;
pub mod mmq;

pub mod island;
pub mod norm;
pub mod rope;
pub mod attn;
pub mod gdn;
pub mod fuse;
pub mod dag;
pub mod place;
pub mod route;
