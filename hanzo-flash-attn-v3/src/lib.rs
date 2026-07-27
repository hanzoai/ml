// SPDX-License-Identifier: Apache-2.0 OR MIT
// Copyright (c) 2024 Michael Feil
//               2025 adjusted by Eric Buehler for hanzo_ml repo.
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! FlashAttention-3 (Hopper `sm_90a`) attention for the Hanzo ML stack.
//!
//! FlashAttention-3 (Shah et al., arXiv:2407.08608) is built around
//! Hopper-specific hardware — warp-specialized producer/consumer pipelining,
//! TMA async copy, `wgmma`, and the GEMM<->softmax "pingpong" overlap — so its
//! kernels run only on `sm_90a` datacenter GPUs (H100 / H200).
//!
//! # Arch gate
//! The CUDA kernels compile only under the `cuda` feature, and `build.rs`
//! cross-compiles them for `sm_90a` alone. Without that feature this crate is a
//! pure-Rust no-op stub: every non-datacenter target — sm_121 (GB10), ROCm
//! (`gfx*`), Metal, Vulkan, CPU — builds and links with zero CUDA involvement,
//! exactly as before it was a workspace member.
//!
//! Runtime selection between FA3 and a portable attention kernel is the
//! caller's responsibility. `hanzo-engine`'s CUDA attention backend reads the
//! device compute capability and routes `sm_90a` here, everything else to the
//! portable path.

#[cfg(feature = "cuda")]
mod ffi;

#[cfg(feature = "cuda")]
mod cuda_impl;
#[cfg(feature = "cuda")]
pub use cuda_impl::*;

#[cfg(not(feature = "cuda"))]
mod stub;
#[cfg(not(feature = "cuda"))]
pub use stub::*;
