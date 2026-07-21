// SPDX-License-Identifier: Apache-2.0 OR MIT
//
// No-op stub used when the crate is built without the `cuda` feature.
//
// FlashAttention-3 targets `sm_90a` datacenter GPUs (H100 / H200) and its
// kernels only exist in a `--features cuda` build. On every other target the
// crate still exposes the same public surface so downstream code compiles
// unchanged; any call resolves to a clear runtime error instead of a link
// against absent kernels.

use hanzo_ml::{Result, Tensor};

#[cold]
#[inline(never)]
fn unavailable<T>() -> Result<T> {
    hanzo_ml::bail!(
        "hanzo-flash-attn-v3 was built without the `cuda` feature; \
         FlashAttention-3 requires an sm_90a datacenter GPU build"
    )
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _softmax_scale: f32,
    _causal: bool,
    _use_gqa_packing: bool,
) -> Result<Tensor> {
    unavailable()
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_windowed(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _softmax_scale: f32,
    _window_size_left: Option<usize>,
    _window_size_right: Option<usize>,
    _use_gqa_packing: bool,
) -> Result<Tensor> {
    unavailable()
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_alibi(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _alibi_slopes: &Tensor,
    _softmax_scale: f32,
    _causal: bool,
    _use_gqa_packing: bool,
) -> Result<Tensor> {
    unavailable()
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_alibi_windowed(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _alibi_slopes: &Tensor,
    _softmax_scale: f32,
    _window_size_left: Option<usize>,
    _window_size_right: Option<usize>,
    _use_gqa_packing: bool,
) -> Result<Tensor> {
    unavailable()
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _seqlens_q: &Tensor,
    _seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    _softmax_scale: f32,
    _causal: bool,
    _use_gqa_packing: bool,
) -> Result<Tensor> {
    unavailable()
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_windowed(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _seqlens_q: &Tensor,
    _seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    _softmax_scale: f32,
    _window_size_left: Option<usize>,
    _window_size_right: Option<usize>,
    _use_gqa_packing: bool,
) -> Result<Tensor> {
    unavailable()
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_alibi(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _alibi_slopes: &Tensor,
    _seqlens_q: &Tensor,
    _seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    _softmax_scale: f32,
    _causal: bool,
    _use_gqa_packing: bool,
) -> Result<Tensor> {
    unavailable()
}

#[allow(clippy::too_many_arguments)]
pub fn flash_attn_varlen_alibi_windowed(
    _q: &Tensor,
    _k: &Tensor,
    _v: &Tensor,
    _alibi_slopes: &Tensor,
    _seqlens_q: &Tensor,
    _seqlens_k: &Tensor,
    _max_seqlen_q: usize,
    _max_seqlen_k: usize,
    _softmax_scale: f32,
    _window_size_left: Option<usize>,
    _window_size_right: Option<usize>,
    _use_gqa_packing: bool,
) -> Result<Tensor> {
    unavailable()
}
