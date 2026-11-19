//! Backend-independent production Paged & Contiguous FlashAttention powered by canonical
//! `hanzo-kernel` DSL artifacts (Metal, CUDA, Vulkan).
//!
//! Gated behind [`crate::quantized::dsl::flash_impl`] (`HANZO_FLASH_IMPL`, defaulting to `legacy`),
//! returning `Ok(None)` when legacy execution is active or when a backend/shape/dtype is outside
//! the committed DSL matrix (`d = 64, 128, 256`; `F32`, `F16`, `BF16` on Metal/CUDA; `F32` Wave32
//! coopmat on Vulkan).

#![allow(unused_variables, unused_mut, unreachable_code, dead_code)]

#[allow(unused_imports)]
use crate::op::BackpropOp;
use crate::quantized::dsl::{self, QkImpl};
#[allow(unused_imports)]
use crate::{DType, Result, Storage, Tensor};

/// KV storage layout accepted by [`try_flash_attn_paged`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvCacheLayout {
    /// Flat 4D paged KV: `[num_blocks, n_kv, block_size, head_dim]`.
    FlatPaged4D,
    /// Engine Standard paged KV: 5D K `[num_blocks, n_kv, head_dim/x, block_size, x]` +
    /// 4D V `[num_blocks, n_kv, head_dim, block_size]`.
    EngineStandard5D4D { x_vec: usize },
    /// Contiguous `[batch, n_kv, kv_seq_stride, head_dim]`.
    Contiguous { kv_seq_stride: usize },
}

/// Optional causal/window/softcap parameters for [`try_flash_attn_paged`] and
/// [`try_flash_attn_contiguous`].
#[derive(Clone, Copy, Debug)]
pub struct FlashAttnConfig {
    pub scale: f32,
    /// `0.0` disables logit softcapping.
    pub softcap: f32,
    /// `0` = non-causal, `1` = causal with `q_offset = ctx_len - seq_q`, `2` = causal with
    /// explicit `q_offset_override`.
    pub causal_mode: u32,
    /// `0` disables sliding-window masking.
    pub sliding_window: usize,
    /// Used only when `causal_mode == 2`.
    pub q_offset_override: usize,
}

impl FlashAttnConfig {
    pub fn causal(scale: f32) -> Self {
        Self {
            scale,
            softcap: 0.0,
            causal_mode: 1,
            sliding_window: 0,
            q_offset_override: 0,
        }
    }

    pub fn non_causal(scale: f32) -> Self {
        Self {
            scale,
            softcap: 0.0,
            causal_mode: 0,
            sliding_window: 0,
            q_offset_override: 0,
        }
    }
}

fn ensure_contiguous_view(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

fn ensure_contiguous_offset_zero(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() && t.layout().start_offset() == 0 {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// Dispatch production DSL FlashAttention over paged or contiguous KV storage.
///
/// - `q`: `[batch, n_heads, seq_q, d]` (`F32`/`F16`/`BF16` on Metal/CUDA; `F32` on Vulkan)
/// - `k_cache`, `v_cache`: matching `layout` and `q.dtype()`
/// - `block_tables`: `[batch, max_blocks]` (`U32`) — ignored (can be 1-element dummy `U32`) when
///   `layout` is [`KvCacheLayout::Contiguous`]
/// - `context_lens`: `[batch]` (`U32`) — per-request valid KV length
///
/// Returns `Ok(Some(out))` with shape `[batch, n_heads, seq_q, d]` in `q.dtype()`, or `Ok(None)`
/// to fall back to the legacy attention kernel.
#[allow(clippy::too_many_arguments)]
pub fn try_flash_attn_paged(
    q: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    n_kv: usize,
    layout: KvCacheLayout,
    cfg: FlashAttnConfig,
) -> Result<Option<Tensor>> {
    if dsl::flash_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    let (batch, n_heads, seq_q, d) = q.dims4()?;
    if !matches!(d, 64 | 128 | 256) || n_kv == 0 || n_heads % n_kv != 0 {
        return Ok(None);
    }
    let dtype = q.dtype();
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Ok(None);
    }
    if k_cache.dtype() != dtype || v_cache.dtype() != dtype {
        return Ok(None);
    }
    if block_tables.dtype() != DType::U32 || context_lens.dtype() != DType::U32 {
        return Ok(None);
    }

    let (kv_layout, block_size, max_blocks, x_vec) = match layout {
        KvCacheLayout::FlatPaged4D => {
            let (_nb, k_nkv, bs, kd) = k_cache.dims4()?;
            if k_nkv != n_kv || kd != d {
                return Ok(None);
            }
            let mb = block_tables.dim(1)?;
            (0usize, bs, mb, 1usize)
        }
        KvCacheLayout::EngineStandard5D4D { x_vec } => {
            let (_nb, k_nkv, dx, bs, xv) = k_cache.dims5()?;
            if k_nkv != n_kv || xv != x_vec || dx * xv != d {
                return Ok(None);
            }
            let mb = block_tables.dim(1)?;
            (1usize, bs, mb, x_vec)
        }
        KvCacheLayout::Contiguous { kv_seq_stride } => (2usize, kv_seq_stride, 1usize, 1usize),
    };

    match q.device() {
        #[cfg(feature = "cuda")]
        crate::Device::Cuda(dev) => {
            let q_c = ensure_contiguous_view(q)?;
            let k_c = ensure_contiguous_view(k_cache)?;
            let v_c = ensure_contiguous_view(v_cache)?;
            let bt_c = ensure_contiguous_view(block_tables)?;
            let cl_c = ensure_contiguous_view(context_lens)?;
            let (qs, ql) = q_c.storage_and_layout();
            let (ks, kl) = k_c.storage_and_layout();
            let (vs, vl) = v_c.storage_and_layout();
            let (bts, btl) = bt_c.storage_and_layout();
            let (cls, cll) = cl_c.storage_and_layout();
            let (
                Storage::Cuda(qs),
                Storage::Cuda(ks),
                Storage::Cuda(vs),
                Storage::Cuda(bts),
                Storage::Cuda(cls),
            ) = (&*qs, &*ks, &*vs, &*bts, &*cls)
            else {
                return Ok(None);
            };
            let out_s = dev.flash_attn_paged_dsl(
                dtype,
                (qs, ql.start_offset()),
                (ks, kl.start_offset()),
                (vs, vl.start_offset()),
                (bts, btl.start_offset()),
                (cls, cll.start_offset()),
                batch,
                n_heads,
                n_kv,
                seq_q,
                d,
                cfg.scale,
                cfg.softcap,
                cfg.causal_mode,
                block_size,
                max_blocks,
                kv_layout,
                x_vec,
                cfg.sliding_window,
                cfg.q_offset_override,
            )?;
            Ok(Some(Tensor::from_storage(
                Storage::Cuda(out_s),
                (batch, n_heads, seq_q, d),
                BackpropOp::none(),
                false,
            )))
        }
        #[cfg(feature = "metal")]
        crate::Device::Metal(dev) => {
            let q_c = ensure_contiguous_view(q)?;
            let k_c = ensure_contiguous_view(k_cache)?;
            let v_c = ensure_contiguous_view(v_cache)?;
            let bt_c = ensure_contiguous_view(block_tables)?;
            let cl_c = ensure_contiguous_view(context_lens)?;
            let elem_bytes = dtype.size_in_bytes();
            let out_elems = batch * n_heads * seq_q * d;
            let out_buf = dev.new_buffer(out_elems, dtype, "flash_attn_paged_dsl_out")?;
            {
                let (qs, ql) = q_c.storage_and_layout();
                let (ks, kl) = k_c.storage_and_layout();
                let (vs, vl) = v_c.storage_and_layout();
                let (bts, btl) = bt_c.storage_and_layout();
                let (cls, cll) = cl_c.storage_and_layout();
                let (
                    Storage::Metal(qs),
                    Storage::Metal(ks),
                    Storage::Metal(vs),
                    Storage::Metal(bts),
                    Storage::Metal(cls),
                ) = (&*qs, &*ks, &*vs, &*bts, &*cls)
                else {
                    return Ok(None);
                };
                dev.flash_attn_paged_dsl(
                    dtype,
                    (qs.buffer(), ql.start_offset() * elem_bytes),
                    (ks.buffer(), kl.start_offset() * elem_bytes),
                    (vs.buffer(), vl.start_offset() * elem_bytes),
                    (bts.buffer(), btl.start_offset() * 4),
                    (cls.buffer(), cll.start_offset() * 4),
                    (&out_buf, 0),
                    batch,
                    n_heads,
                    n_kv,
                    seq_q,
                    d,
                    cfg.scale,
                    cfg.softcap,
                    cfg.causal_mode,
                    block_size,
                    max_blocks,
                    kv_layout,
                    x_vec,
                    cfg.sliding_window,
                    cfg.q_offset_override,
                )?;
            }
            let out_s = crate::MetalStorage::new(out_buf, dev.clone(), out_elems, dtype);
            Ok(Some(Tensor::from_storage(
                Storage::Metal(out_s),
                (batch, n_heads, seq_q, d),
                BackpropOp::none(),
                false,
            )))
        }
        #[cfg(feature = "vulkan")]
        crate::Device::Vulkan(dev) => {
            if dtype != DType::F32 {
                return Ok(None);
            }
            let q_c = ensure_contiguous_offset_zero(q)?;
            let k_c = ensure_contiguous_offset_zero(k_cache)?;
            let v_c = ensure_contiguous_offset_zero(v_cache)?;
            let bt_c = ensure_contiguous_offset_zero(block_tables)?;
            let cl_c = ensure_contiguous_offset_zero(context_lens)?;
            let (qs, _) = q_c.storage_and_layout();
            let (ks, _) = k_c.storage_and_layout();
            let (vs, _) = v_c.storage_and_layout();
            let (bts, _) = bt_c.storage_and_layout();
            let (cls, _) = cl_c.storage_and_layout();
            let (
                Storage::Vulkan(qs),
                Storage::Vulkan(ks),
                Storage::Vulkan(vs),
                Storage::Vulkan(bts),
                Storage::Vulkan(cls),
            ) = (&*qs, &*ks, &*vs, &*bts, &*cls)
            else {
                return Ok(None);
            };
            let Some(out_s) = dev.flash_attn_paged_dsl(
                qs,
                ks,
                vs,
                bts,
                cls,
                batch,
                n_heads,
                n_kv,
                seq_q,
                d,
                cfg.scale,
                cfg.softcap,
                cfg.causal_mode,
                block_size,
                max_blocks,
                kv_layout,
                x_vec,
                cfg.sliding_window,
                cfg.q_offset_override,
            )?
            else {
                return Ok(None);
            };
            Ok(Some(Tensor::from_storage(
                Storage::Vulkan(out_s),
                (batch, n_heads, seq_q, d),
                BackpropOp::none(),
                false,
            )))
        }
        _ => Ok(None),
    }
}

/// Convenience wrapper for contiguous SDPA (`q: [B, H, seq_q, d]`, `k, v: [B, H_kv, seq_k, d]`).
pub fn try_flash_attn_contiguous(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: FlashAttnConfig,
) -> Result<Option<Tensor>> {
    if dsl::flash_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    let (batch, _n_heads, _seq_q, d) = q.dims4()?;
    let (kb, n_kv, seq_k, kd) = k.dims4()?;
    if kb != batch || kd != d || v.dims() != [batch, n_kv, seq_k, d] {
        return Ok(None);
    }
    let dummy_bt = Tensor::zeros((batch, 1), DType::U32, q.device())?;
    let ctx_lens = Tensor::from_vec(vec![seq_k as u32; batch], batch, q.device())?;
    try_flash_attn_paged(
        q,
        k,
        v,
        &dummy_bt,
        &ctx_lens,
        n_kv,
        KvCacheLayout::Contiguous {
            kv_seq_stride: seq_k,
        },
        cfg,
    )
}
