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
        t.force_contiguous()
    }
}

fn is_kv_strided_contiguous(
    t: &Tensor,
    batch: usize,
    n_kv: usize,
    seq_k: usize,
    d: usize,
    kv_seq_stride: usize,
) -> bool {
    if kv_seq_stride < seq_k {
        return false;
    }
    let l = t.layout();
    if l.dims() != [batch, n_kv, seq_k, d] {
        return false;
    }
    let s = l.stride();
    let d_ok = d <= 1 || s[3] == 1;
    let seq_ok = seq_k <= 1 || s[2] == d;
    let head_ok = (batch <= 1 && n_kv <= 1) || s[1] == kv_seq_stride * d;
    let batch_ok = batch <= 1 || s[0] == n_kv * kv_seq_stride * d;
    d_ok && seq_ok && head_ok && batch_ok
}

thread_local! {
    static CONTIG_META_CACHE: std::cell::RefCell<Option<(usize, usize, Tensor, Tensor)>> =
        const { std::cell::RefCell::new(None) };
}

const SPLIT_KV_UNINIT: usize = usize::MAX;
const SPLIT_KV_AUTO: usize = 0;

static FLASH_SPLIT_KV_OVERRIDE: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(SPLIT_KV_UNINIT);

fn read_flash_split_kv_override() -> Option<usize> {
    use std::sync::atomic::Ordering;
    let v = FLASH_SPLIT_KV_OVERRIDE.load(Ordering::Relaxed);
    if v == SPLIT_KV_AUTO {
        return None;
    }
    if v != SPLIT_KV_UNINIT {
        return Some(v);
    }
    let parsed = std::env::var("HANZO_FLASH_SPLIT_KV")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(SPLIT_KV_AUTO);
    FLASH_SPLIT_KV_OVERRIDE.store(parsed, Ordering::Relaxed);
    (parsed != SPLIT_KV_AUTO).then_some(parsed)
}

/// Set or clear an in-process override for the Split-KV partition count (`num_splits`).
///
/// Pass `Some(n)` (`n >= 1`) to force `n` KV partitions across all subsequent Split-KV
/// dispatches, or `None` to restore automatic partition selection.
pub fn set_flash_split_kv_override(num_splits: Option<usize>) {
    use std::sync::atomic::Ordering;
    let encoded = num_splits.filter(|&n| n >= 1).unwrap_or(SPLIT_KV_AUTO);
    FLASH_SPLIT_KV_OVERRIDE.store(encoded, Ordering::Relaxed);
}

/// Select the KV partition factor (`num_splits`) for canonical DSL Split-KV FlashAttention decode
/// (`seq_q <= 4`).
///
/// Balances GPU occupancy (`total_qh * num_splits` active warps across the device) against
/// per-split KV granularity (`min_tokens_per_split` keys per warp) and pass-2 reduction overhead:
/// - Short contexts (`active_ctx <= 128`) return `1` (single-pass warp-register decode, no pass-2 launch).
/// - Medium/long contexts (`512 .. 32K+`) scale `num_splits` through powers of two based on
///   context length, head dimension, batch/head count, and backend occupancy.
/// - Override with `HANZO_FLASH_SPLIT_KV=<N>` (`N >= 1`) or [`set_flash_split_kv_override`] for
///   diagnostics and A/B sweeps.
pub fn choose_flash_split_kv(
    batch: usize,
    n_heads: usize,
    seq_q: usize,
    d: usize,
    max_ctx_len: usize,
    sliding_window: usize,
) -> usize {
    choose_flash_split_kv_for(
        dsl::QkBackend::Metal,
        batch,
        n_heads,
        seq_q,
        d,
        max_ctx_len,
        sliding_window,
    )
}

/// Backend-aware Split-KV partition selector used by CUDA, ROCm, Metal, and Vulkan dispatches.
///
/// Partition schedule calibrated against `num_splits = [1, 8, 16, 32, 64, 128, 256]` sweeps across
/// `d in {64, 128, 256}` and `seq_k in {512, 4K, 8K, 16K, 32K}` (`B=1, H=24, H_kv=2, seq_q=1`):
/// - `active_ctx <= 128`: `1` (single-pass warp-register decode, zero pass-2 reduction overhead).
/// - `active_ctx <= 1024`: up to `16` splits (`32` keys/split floor; beats `8` across `d=64..256`).
/// - `active_ctx <= 4096`: up to `32` splits (`64` on Vulkan `d >= 256`).
/// - `active_ctx <= 8192`: up to `64` splits (`128` on Vulkan where higher split counts improve occupancy).
/// - `active_ctx >= 16384`: up to `128` splits (`-23%` to `-39%` latency vs `64` at 32K on CUDA,
///   `-14%` to `-21%` on Vulkan; avoids `256`-split pass-2 reduction overhead).
/// - Automatically scales down by `target_warps / total_qh` when `batch * n_heads * seq_q` is large.
pub fn choose_flash_split_kv_for(
    backend: dsl::QkBackend,
    batch: usize,
    n_heads: usize,
    seq_q: usize,
    d: usize,
    max_ctx_len: usize,
    sliding_window: usize,
) -> usize {
    if let Some(forced) = read_flash_split_kv_override() {
        return forced;
    }
    let active_ctx = if sliding_window > 0 {
        max_ctx_len.min(sliding_window)
    } else {
        max_ctx_len
    };
    if active_ctx <= 128 {
        return 1;
    }
    let total_qh = (batch * n_heads * seq_q).max(1);
    let (min_tokens_per_split, target_warps, max_splits) = if active_ctx <= 1024 {
        (32usize, 384usize, 16usize)
    } else if active_ctx <= 4096 {
        if backend == dsl::QkBackend::Vulkan && d >= 256 {
            (64usize, 1536usize, 64usize)
        } else {
            (128usize, 768usize, 32usize)
        }
    } else if active_ctx <= 8192 {
        if backend == dsl::QkBackend::Vulkan {
            (64usize, 3072usize, 128usize)
        } else {
            (128usize, 1536usize, 64usize)
        }
    } else {
        (128usize, 3072usize, 128usize)
    };
    let max_by_work = (active_ctx / min_tokens_per_split).max(1);
    let max_by_occ = (target_warps / total_qh).max(4);
    let raw = max_by_work.min(max_by_occ).clamp(1, max_splits);
    if raw <= 1 {
        1
    } else {
        raw.next_power_of_two().min(max_splits)
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
    let (batch, n_heads, seq_q, d) = q.dims4()?;
    let seq_k_est = match layout {
        KvCacheLayout::FlatPaged4D => {
            let (_nb, _k_nkv, bs, _kd) = k_cache.dims4()?;
            let mb = block_tables.dim(1)?;
            mb.saturating_mul(bs)
        }
        KvCacheLayout::EngineStandard5D4D { .. } => {
            let (_nb, _k_nkv, _dx, bs, _xv) = k_cache.dims5()?;
            let mb = block_tables.dim(1)?;
            mb.saturating_mul(bs)
        }
        KvCacheLayout::Contiguous { .. } => {
            let (_kb, _k_nkv, seq_k, _kd) = k_cache.dims4()?;
            seq_k
        }
    };
    if dsl::select_impl(
        dsl::QkBackend::from_device(q.device()),
        dsl::QkOp::FlashAttn {
            batch,
            n_heads,
            seq_q,
            seq_k: seq_k_est,
            head_dim: d,
        },
    ) != QkImpl::Dsl
    {
        return Ok(None);
    }
    if !matches!(d, 64 | 128 | 256) || n_kv == 0 || !n_heads.is_multiple_of(n_kv) {
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

    let need_zero_offset = q.device().is_vulkan();
    let (k_c, v_c, kv_layout, block_size, max_blocks, x_vec) = match layout {
        KvCacheLayout::FlatPaged4D => {
            let (_nb, k_nkv, bs, kd) = k_cache.dims4()?;
            if k_nkv != n_kv || kd != d {
                return Ok(None);
            }
            let mb = block_tables.dim(1)?;
            let kc = if need_zero_offset {
                ensure_contiguous_offset_zero(k_cache)?
            } else {
                ensure_contiguous_view(k_cache)?
            };
            let vc = if need_zero_offset {
                ensure_contiguous_offset_zero(v_cache)?
            } else {
                ensure_contiguous_view(v_cache)?
            };
            (kc, vc, 0usize, bs, mb, 1usize)
        }
        KvCacheLayout::EngineStandard5D4D { x_vec } => {
            let (_nb, k_nkv, dx, bs, xv) = k_cache.dims5()?;
            if k_nkv != n_kv || xv != x_vec || dx * xv != d {
                return Ok(None);
            }
            let mb = block_tables.dim(1)?;
            let kc = if need_zero_offset {
                ensure_contiguous_offset_zero(k_cache)?
            } else {
                ensure_contiguous_view(k_cache)?
            };
            let vc = if need_zero_offset {
                ensure_contiguous_offset_zero(v_cache)?
            } else {
                ensure_contiguous_view(v_cache)?
            };
            (kc, vc, 1usize, bs, mb, x_vec)
        }
        KvCacheLayout::Contiguous { kv_seq_stride } => {
            let (kb, k_nkv, seq_k, kd) = k_cache.dims4()?;
            if kb != batch
                || k_nkv != n_kv
                || kd != d
                || seq_k > kv_seq_stride
                || v_cache.dims() != [batch, n_kv, seq_k, d]
            {
                return Ok(None);
            }
            let k_in_place = is_kv_strided_contiguous(k_cache, batch, n_kv, seq_k, d, kv_seq_stride)
                && (!need_zero_offset || k_cache.layout().start_offset() == 0);
            let v_in_place = is_kv_strided_contiguous(v_cache, batch, n_kv, seq_k, d, kv_seq_stride)
                && (!need_zero_offset || v_cache.layout().start_offset() == 0);
            if k_in_place && v_in_place {
                (
                    k_cache.clone(),
                    v_cache.clone(),
                    2usize,
                    kv_seq_stride,
                    seq_k,
                    1usize,
                )
            } else {
                let kc = if need_zero_offset {
                    ensure_contiguous_offset_zero(k_cache)?
                } else {
                    ensure_contiguous_view(k_cache)?
                };
                let vc = if need_zero_offset {
                    ensure_contiguous_offset_zero(v_cache)?
                } else {
                    ensure_contiguous_view(v_cache)?
                };
                (kc, vc, 2usize, seq_k, seq_k, 1usize)
            }
        }
    };

    match q.device() {
        #[cfg(feature = "cuda")]
        crate::Device::Cuda(dev) => {
            let q_c = ensure_contiguous_view(q)?;
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
        #[cfg(feature = "rocm")]
        crate::Device::Rocm(dev) => {
            let q_c = ensure_contiguous_view(q)?;
            let bt_c = ensure_contiguous_view(block_tables)?;
            let cl_c = ensure_contiguous_view(context_lens)?;
            let (qs, ql) = q_c.storage_and_layout();
            let (ks, kl) = k_c.storage_and_layout();
            let (vs, vl) = v_c.storage_and_layout();
            let (bts, btl) = bt_c.storage_and_layout();
            let (cls, cll) = cl_c.storage_and_layout();
            let (
                Storage::Rocm(qs),
                Storage::Rocm(ks),
                Storage::Rocm(vs),
                Storage::Rocm(bts),
                Storage::Rocm(cls),
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
                Storage::Rocm(out_s),
                (batch, n_heads, seq_q, d),
                BackpropOp::none(),
                false,
            )))
        }
        #[cfg(feature = "metal")]
        crate::Device::Metal(dev) => {
            let q_c = ensure_contiguous_view(q)?;
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

/// Convenience wrapper for contiguous or zero-copy strided SDPA (`q: [B, H, seq_q, d]`,
/// `k, v: [B, H_kv, seq_k, d]`, including narrowed zero-copy KV cache views with
/// `stride = [n_kv * kv_seq_stride * d, kv_seq_stride * d, d, 1]`).
pub fn try_flash_attn_contiguous(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    cfg: FlashAttnConfig,
) -> Result<Option<Tensor>> {
    let (batch, n_heads, seq_q, d) = q.dims4()?;
    let (kb, n_kv, seq_k, kd) = k.dims4()?;
    if dsl::select_impl(
        dsl::QkBackend::from_device(q.device()),
        dsl::QkOp::FlashAttn {
            batch,
            n_heads,
            seq_q,
            seq_k,
            head_dim: d,
        },
    ) != QkImpl::Dsl
    {
        return Ok(None);
    }
    if kb != batch || kd != d || v.dims() != [batch, n_kv, seq_k, d] {
        return Ok(None);
    }
    let ks = k.layout().stride();
    let vs = v.layout().stride();
    let kv_seq_stride = if batch <= 1 && n_kv <= 1 {
        seq_k
    } else if ks.len() == 4
        && vs.len() == 4
        && ks[1] == vs[1]
        && d > 0
        && ks[1].is_multiple_of(d)
        && ks[1] / d >= seq_k
    {
        ks[1] / d
    } else {
        seq_k
    };
    let dev = q.device();
    let (dummy_bt, ctx_lens) = CONTIG_META_CACHE.with(|cell| -> Result<(Tensor, Tensor)> {
        let mut guard = cell.borrow_mut();
        if let Some((cb, csk, bt, cl)) = guard.as_ref() {
            if *cb == batch && bt.device().same_device(dev) {
                if *csk == seq_k {
                    return Ok((bt.clone(), cl.clone()));
                }
                let bt_reuse = bt.clone();
                let cl_new = Tensor::from_vec(vec![seq_k as u32; batch], batch, dev)?;
                *guard = Some((batch, seq_k, bt_reuse.clone(), cl_new.clone()));
                return Ok((bt_reuse, cl_new));
            }
        }
        let bt_new = Tensor::zeros((batch, 1), DType::U32, dev)?;
        let cl_new = Tensor::from_vec(vec![seq_k as u32; batch], batch, dev)?;
        *guard = Some((batch, seq_k, bt_new.clone(), cl_new.clone()));
        Ok((bt_new, cl_new))
    })?;
    try_flash_attn_paged(
        q,
        k,
        v,
        &dummy_bt,
        &ctx_lens,
        n_kv,
        KvCacheLayout::Contiguous { kv_seq_stride },
        cfg,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::quantized::dsl::QkBackend;

    #[test]
    fn split_kv_partition_schedule_and_override() {
        set_flash_split_kv_override(None);

        // Short context stays single-pass (num_splits == 1).
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 64, 0), 1);
        assert_eq!(choose_flash_split_kv_for(QkBackend::Metal, 1, 24, 1, 128, 128, 0), 1);

        // Medium/long single-request decode scales up to 128 splits at 16K..32K.
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 512, 0), 16);
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 4096, 0), 32);
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 8192, 0), 64);
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 16384, 0), 128);
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 32768, 0), 128);

        // Vulkan favors higher split counts at 4K (d=256) and 8K.
        assert_eq!(choose_flash_split_kv_for(QkBackend::Vulkan, 1, 24, 1, 256, 4096, 0), 64);
        assert_eq!(choose_flash_split_kv_for(QkBackend::Vulkan, 1, 24, 1, 128, 8192, 0), 128);

        // Sliding window clamps active context before partitioning.
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 32768, 128), 1);
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 32768, 512), 16);

        // Large batch scales splits down proportionally to avoid oversubscription.
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 4, 24, 1, 128, 32768, 0), 32);
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 16, 24, 1, 128, 32768, 0), 8);

        // In-process override takes precedence and clears cleanly.
        set_flash_split_kv_override(Some(256));
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 64, 0), 256);
        set_flash_split_kv_override(None);
        assert_eq!(choose_flash_split_kv_for(QkBackend::Cuda, 1, 24, 1, 128, 64, 0), 1);
    }
}
