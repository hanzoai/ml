//! Backend-independent production Gated DeltaNet (GDN) operators powered by canonical
//! `hanzo-kernel` DSL artifacts (Metal, CUDA, Vulkan).
//!
//! All operators are gated behind [`crate::quantized::dsl::gdn_impl`] (`HANZO_GDN_IMPL`,
//! defaulting to `legacy`), returning `Ok(None)` when legacy execution is active or when a
//! backend/shape/dtype does not have a committed DSL specialization, so callers fall back cleanly
//! without duplicating backend-specific dispatch logic.

#![allow(unused_variables, unused_mut, unreachable_code, dead_code)]

#[allow(unused_imports)]
use crate::op::BackpropOp;
use crate::quantized::dsl::{self, QkImpl};
#[allow(unused_imports)]
use crate::{DType, Result, Storage, Tensor};

fn ensure_contiguous_offset_zero(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() && t.layout().start_offset() == 0 {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

fn ensure_contiguous_view(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// Single-token GDN causal conv1d decode step (`k = 4`, in-place `conv_state` shift + SiLU).
///
/// - `x`: `[batch, 1, conv_dim]` or `[batch, conv_dim]` (`F32`, `F16`, or `BF16` on Metal/CUDA;
///   `F32` on Vulkan)
/// - `weight`: `[conv_dim, 4]` or `[conv_dim, 1, 4]` (same dtype as `x`)
/// - `conv_state`: `[batch, conv_dim, 4]` (same dtype as `x`), updated in place (or replaced with
///   a contiguous copy if it was non-contiguous)
///
/// Returns `Ok(Some(out))` with shape `[batch, 1, conv_dim]` in `x.dtype()` when `HANZO_GDN_IMPL=dsl`
/// handled the op, or `Ok(None)` to fall back to the legacy path.
pub fn try_conv1d_step(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
) -> Result<Option<Tensor>> {
    if dsl::gdn_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    let (batch, conv_dim) = match x.dims() {
        [b, 1, c] => (*b, *c),
        [b, c] => (*b, *c),
        _ => return Ok(None),
    };
    let (s_b, s_c, s_k) = conv_state.dims3()?;
    if s_b != batch || s_c != conv_dim || s_k != 4 {
        return Ok(None);
    }
    let dtype = x.dtype();
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Ok(None);
    }
    let w = if weight.rank() == 3 && weight.dim(1)? == 1 {
        weight.squeeze(1)?
    } else {
        weight.clone()
    };
    if w.dims() != [conv_dim, 4] {
        return Ok(None);
    }
    let w = if w.dtype() == dtype {
        w
    } else {
        w.to_dtype(dtype)?
    };
    if conv_state.dtype() != dtype {
        *conv_state = conv_state.to_dtype(dtype)?;
    }

    match x.device() {
        #[cfg(feature = "cuda")]
        crate::Device::Cuda(dev) => {
            let x_c = ensure_contiguous_view(x)?;
            let w_c = ensure_contiguous_view(&w)?;
            if !conv_state.is_contiguous() {
                *conv_state = conv_state.contiguous()?;
            }
            let (xs, xl) = x_c.storage_and_layout();
            let (ws, wl) = w_c.storage_and_layout();
            let (mut ss, sl) = conv_state.storage_mut_and_layout();
            let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(ss)) =
                (&*xs, &*ws, &mut *ss)
            else {
                return Ok(None);
            };
            let out_s = dev.gdn_conv1d_step_dsl(
                dtype,
                (xs, xl.start_offset()),
                (ws, wl.start_offset()),
                (ss, sl.start_offset()),
                batch,
                conv_dim,
            )?;
            Ok(Some(Tensor::from_storage(
                Storage::Cuda(out_s),
                (batch, 1, conv_dim),
                BackpropOp::none(),
                false,
            )))
        }
        #[cfg(feature = "metal")]
        crate::Device::Metal(dev) => {
            let x_c = ensure_contiguous_view(x)?;
            let w_c = ensure_contiguous_view(&w)?;
            if !conv_state.is_contiguous() {
                *conv_state = conv_state.contiguous()?;
            }
            let elem_bytes = dtype.size_in_bytes();
            let out_buf = dev.new_buffer(batch * conv_dim, dtype, "gdn_conv1d_step_dsl_out")?;
            {
                let (xs, xl) = x_c.storage_and_layout();
                let (ws, wl) = w_c.storage_and_layout();
                let (mut ss, sl) = conv_state.storage_mut_and_layout();
                let (Storage::Metal(xs), Storage::Metal(ws), Storage::Metal(ss)) =
                    (&*xs, &*ws, &mut *ss)
                else {
                    return Ok(None);
                };
                dev.gdn_conv1d_step_dsl(
                    dtype,
                    (xs.buffer(), xl.start_offset() * elem_bytes),
                    (ws.buffer(), wl.start_offset() * elem_bytes),
                    (ss.buffer(), sl.start_offset() * elem_bytes),
                    (&out_buf, 0),
                    batch,
                    conv_dim,
                )?;
            }
            let out_s = crate::MetalStorage::new(out_buf, dev.clone(), batch * conv_dim, dtype);
            Ok(Some(Tensor::from_storage(
                Storage::Metal(out_s),
                (batch, 1, conv_dim),
                BackpropOp::none(),
                false,
            )))
        }
        #[cfg(feature = "vulkan")]
        crate::Device::Vulkan(dev) => {
            if dtype != DType::F32 {
                return Ok(None);
            }
            let x_c = ensure_contiguous_offset_zero(x)?;
            let w_c = ensure_contiguous_offset_zero(&w)?;
            if !conv_state.is_contiguous() || conv_state.layout().start_offset() != 0 {
                *conv_state = conv_state.contiguous()?;
            }
            let (xs, _) = x_c.storage_and_layout();
            let (ws, _) = w_c.storage_and_layout();
            let (mut ss, _) = conv_state.storage_mut_and_layout();
            let (Storage::Vulkan(xs), Storage::Vulkan(ws), Storage::Vulkan(ss)) =
                (&*xs, &*ws, &mut *ss)
            else {
                return Ok(None);
            };
            let out_s = dev.gdn_conv1d_step_dsl(xs, ws, ss, batch, conv_dim)?;
            Ok(Some(Tensor::from_storage(
                Storage::Vulkan(out_s),
                (batch, 1, conv_dim),
                BackpropOp::none(),
                false,
            )))
        }
        _ => Ok(None),
    }
}

/// Stateful GDN causal conv1d for one-shot or chunked prefill (`k = 4`, SiLU fused).
///
/// - `x_t`: `[batch, conv_dim, seq_len]` (`F32`, `F16`, or `BF16` on Metal/CUDA; `F32` on Vulkan)
/// - `weight`: `[conv_dim, 4]` or `[conv_dim, 1, 4]`
/// - `state_in`: `[batch, conv_dim, 4]` (used when `has_state == true`; ignored when `false`)
///
/// Returns `Ok(Some((out_t, state_out)))` where `out_t` is `[batch, conv_dim, seq_len]` and
/// `state_out` is `[batch, conv_dim, 4]` holding the exact trailing 4 tokens for subsequent chunks
/// or decode steps.
pub fn try_conv1d_stateful(
    x_t: &Tensor,
    weight: &Tensor,
    state_in: &Tensor,
    has_state: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    if dsl::gdn_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    let (batch, conv_dim, seq_len) = x_t.dims3()?;
    if state_in.dims() != [batch, conv_dim, 4] {
        return Ok(None);
    }
    let dtype = x_t.dtype();
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Ok(None);
    }
    let w = if weight.rank() == 3 && weight.dim(1)? == 1 {
        weight.squeeze(1)?
    } else {
        weight.clone()
    };
    if w.dims() != [conv_dim, 4] {
        return Ok(None);
    }
    let w = if w.dtype() == dtype {
        w
    } else {
        w.to_dtype(dtype)?
    };
    let s_in = if state_in.dtype() == dtype {
        state_in.clone()
    } else {
        state_in.to_dtype(dtype)?
    };

    match x_t.device() {
        #[cfg(feature = "cuda")]
        crate::Device::Cuda(dev) => {
            let x_c = ensure_contiguous_view(x_t)?;
            let w_c = ensure_contiguous_view(&w)?;
            let s_c = ensure_contiguous_view(&s_in)?;
            let (xs, xl) = x_c.storage_and_layout();
            let (ws, wl) = w_c.storage_and_layout();
            let (ss, sl) = s_c.storage_and_layout();
            let (Storage::Cuda(xs), Storage::Cuda(ws), Storage::Cuda(ss)) = (&*xs, &*ws, &*ss)
            else {
                return Ok(None);
            };
            let (out_s, state_out_s) = dev.gdn_conv1d_stateful_dsl(
                dtype,
                (xs, xl.start_offset()),
                (ws, wl.start_offset()),
                (ss, sl.start_offset()),
                batch,
                conv_dim,
                seq_len,
                has_state,
            )?;
            Ok(Some((
                Tensor::from_storage(
                    Storage::Cuda(out_s),
                    (batch, conv_dim, seq_len),
                    BackpropOp::none(),
                    false,
                ),
                Tensor::from_storage(
                    Storage::Cuda(state_out_s),
                    (batch, conv_dim, 4),
                    BackpropOp::none(),
                    false,
                ),
            )))
        }
        #[cfg(feature = "metal")]
        crate::Device::Metal(dev) => {
            let x_c = ensure_contiguous_view(x_t)?;
            let w_c = ensure_contiguous_view(&w)?;
            let s_c = ensure_contiguous_view(&s_in)?;
            let elem_bytes = dtype.size_in_bytes();
            let out_buf = dev.new_buffer(
                batch * conv_dim * seq_len,
                dtype,
                "gdn_conv1d_stateful_dsl_out",
            )?;
            let state_buf = dev.new_buffer(
                batch * conv_dim * 4,
                dtype,
                "gdn_conv1d_stateful_dsl_state",
            )?;
            {
                let (xs, xl) = x_c.storage_and_layout();
                let (ws, wl) = w_c.storage_and_layout();
                let (ss, sl) = s_c.storage_and_layout();
                let (Storage::Metal(xs), Storage::Metal(ws), Storage::Metal(ss)) =
                    (&*xs, &*ws, &*ss)
                else {
                    return Ok(None);
                };
                dev.gdn_conv1d_stateful_dsl(
                    dtype,
                    (xs.buffer(), xl.start_offset() * elem_bytes),
                    (ws.buffer(), wl.start_offset() * elem_bytes),
                    (ss.buffer(), sl.start_offset() * elem_bytes),
                    (&state_buf, 0),
                    (&out_buf, 0),
                    batch,
                    conv_dim,
                    seq_len,
                    has_state,
                )?;
            }
            let out_s =
                crate::MetalStorage::new(out_buf, dev.clone(), batch * conv_dim * seq_len, dtype);
            let st_s =
                crate::MetalStorage::new(state_buf, dev.clone(), batch * conv_dim * 4, dtype);
            Ok(Some((
                Tensor::from_storage(
                    Storage::Metal(out_s),
                    (batch, conv_dim, seq_len),
                    BackpropOp::none(),
                    false,
                ),
                Tensor::from_storage(
                    Storage::Metal(st_s),
                    (batch, conv_dim, 4),
                    BackpropOp::none(),
                    false,
                ),
            )))
        }
        #[cfg(feature = "vulkan")]
        crate::Device::Vulkan(dev) => {
            if dtype != DType::F32 {
                return Ok(None);
            }
            let x_c = ensure_contiguous_offset_zero(x_t)?;
            let w_c = ensure_contiguous_offset_zero(&w)?;
            let s_c = ensure_contiguous_offset_zero(&s_in)?;
            let (xs, _) = x_c.storage_and_layout();
            let (ws, _) = w_c.storage_and_layout();
            let (ss, _) = s_c.storage_and_layout();
            let (Storage::Vulkan(xs), Storage::Vulkan(ws), Storage::Vulkan(ss)) =
                (&*xs, &*ws, &*ss)
            else {
                return Ok(None);
            };
            let (out_s, state_out_s) =
                dev.gdn_conv1d_stateful_dsl(xs, ws, ss, batch, conv_dim, seq_len, has_state)?;
            Ok(Some((
                Tensor::from_storage(
                    Storage::Vulkan(out_s),
                    (batch, conv_dim, seq_len),
                    BackpropOp::none(),
                    false,
                ),
                Tensor::from_storage(
                    Storage::Vulkan(state_out_s),
                    (batch, conv_dim, 4),
                    BackpropOp::none(),
                    false,
                ),
            )))
        }
        _ => Ok(None),
    }
}

/// Fused GDN gating: `beta = sigmoid(b)`, `g = -exp(a_log) * softplus(a + dt_bias)`.
///
/// - `b`, `a`: `[..., num_heads]` (`F32`, `F16`, or `BF16` on Metal/CUDA; `F32` on Vulkan)
/// - `a_log`, `dt_bias`: `[num_heads]` (converted to `F32` if needed)
///
/// Returns `Ok(Some((beta, g)))` with the same shape and dtype as `b`.
pub fn try_gating(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
) -> Result<Option<(Tensor, Tensor)>> {
    if dsl::gdn_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    if b.shape() != a.shape() || b.rank() == 0 {
        return Ok(None);
    }
    let dtype = b.dtype();
    if a.dtype() != dtype || !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Ok(None);
    }
    let num_heads = b.dims()[b.rank() - 1];
    let total = b.elem_count();
    if num_heads == 0 || total == 0 || total % num_heads != 0 {
        return Ok(None);
    }
    let batch_tokens = total / num_heads;
    let al = if a_log.dtype() == DType::F32 {
        a_log.clone()
    } else {
        a_log.to_dtype(DType::F32)?
    };
    let dt = if dt_bias.dtype() == DType::F32 {
        dt_bias.clone()
    } else {
        dt_bias.to_dtype(DType::F32)?
    };
    let out_shape = b.shape().clone();

    match b.device() {
        #[cfg(feature = "cuda")]
        crate::Device::Cuda(dev) => {
            let b_c = ensure_contiguous_view(b)?;
            let a_c = ensure_contiguous_view(a)?;
            let al_c = ensure_contiguous_view(&al)?;
            let dt_c = ensure_contiguous_view(&dt)?;
            let (bs, bl) = b_c.storage_and_layout();
            let (as_, al_l) = a_c.storage_and_layout();
            let (als, all) = al_c.storage_and_layout();
            let (dts, dtl) = dt_c.storage_and_layout();
            let (
                Storage::Cuda(bs),
                Storage::Cuda(as_),
                Storage::Cuda(als),
                Storage::Cuda(dts),
            ) = (&*bs, &*as_, &*als, &*dts)
            else {
                return Ok(None);
            };
            let (beta_s, g_s) = dev.gdn_gating_io_dsl(
                dtype,
                (bs, bl.start_offset()),
                (as_, al_l.start_offset()),
                (als, all.start_offset()),
                (dts, dtl.start_offset()),
                batch_tokens,
                num_heads,
            )?;
            Ok(Some((
                Tensor::from_storage(
                    Storage::Cuda(beta_s),
                    out_shape.clone(),
                    BackpropOp::none(),
                    false,
                ),
                Tensor::from_storage(Storage::Cuda(g_s), out_shape, BackpropOp::none(), false),
            )))
        }
        #[cfg(feature = "metal")]
        crate::Device::Metal(dev) => {
            let b_c = ensure_contiguous_view(b)?;
            let a_c = ensure_contiguous_view(a)?;
            let al_c = ensure_contiguous_view(&al)?;
            let dt_c = ensure_contiguous_view(&dt)?;
            let elem_bytes = dtype.size_in_bytes();
            let beta_buf = dev.new_buffer(total, dtype, "gdn_gating_dsl_beta")?;
            let g_buf = dev.new_buffer(total, dtype, "gdn_gating_dsl_g")?;
            {
                let (bs, bl) = b_c.storage_and_layout();
                let (as_, al_l) = a_c.storage_and_layout();
                let (als, all) = al_c.storage_and_layout();
                let (dts, dtl) = dt_c.storage_and_layout();
                let (
                    Storage::Metal(bs),
                    Storage::Metal(as_),
                    Storage::Metal(als),
                    Storage::Metal(dts),
                ) = (&*bs, &*as_, &*als, &*dts)
                else {
                    return Ok(None);
                };
                dev.gdn_gating_io_dsl(
                    dtype,
                    (bs.buffer(), bl.start_offset() * elem_bytes),
                    (as_.buffer(), al_l.start_offset() * elem_bytes),
                    (als.buffer(), all.start_offset() * 4),
                    (dts.buffer(), dtl.start_offset() * 4),
                    (&beta_buf, 0),
                    (&g_buf, 0),
                    batch_tokens,
                    num_heads,
                )?;
            }
            let beta_s = crate::MetalStorage::new(beta_buf, dev.clone(), total, dtype);
            let g_s = crate::MetalStorage::new(g_buf, dev.clone(), total, dtype);
            Ok(Some((
                Tensor::from_storage(
                    Storage::Metal(beta_s),
                    out_shape.clone(),
                    BackpropOp::none(),
                    false,
                ),
                Tensor::from_storage(Storage::Metal(g_s), out_shape, BackpropOp::none(), false),
            )))
        }
        #[cfg(feature = "vulkan")]
        crate::Device::Vulkan(dev) => {
            if dtype != DType::F32 {
                return Ok(None);
            }
            let b_c = ensure_contiguous_offset_zero(b)?;
            let a_c = ensure_contiguous_offset_zero(a)?;
            let al_c = ensure_contiguous_offset_zero(&al)?;
            let dt_c = ensure_contiguous_offset_zero(&dt)?;
            let (bs, _) = b_c.storage_and_layout();
            let (as_, _) = a_c.storage_and_layout();
            let (als, _) = al_c.storage_and_layout();
            let (dts, _) = dt_c.storage_and_layout();
            let (
                Storage::Vulkan(bs),
                Storage::Vulkan(as_),
                Storage::Vulkan(als),
                Storage::Vulkan(dts),
            ) = (&*bs, &*as_, &*als, &*dts)
            else {
                return Ok(None);
            };
            let (beta_s, g_s) = dev.gdn_gating_io_dsl(bs, as_, als, dts, batch_tokens, num_heads)?;
            Ok(Some((
                Tensor::from_storage(
                    Storage::Vulkan(beta_s),
                    out_shape.clone(),
                    BackpropOp::none(),
                    false,
                ),
                Tensor::from_storage(Storage::Vulkan(g_s), out_shape, BackpropOp::none(), false),
            )))
        }
        _ => Ok(None),
    }
}

/// Fused GDN gated delta rule recurrence scan (single-token decode or multi-token prefill/chunk).
///
/// - `q`, `k`: `[batch, seq, num_v_heads, k_dim]` (`F32`/`F16`/`BF16` on Metal/CUDA; `F32` on Vulkan)
/// - `v`: `[batch, seq, num_v_heads, v_dim]` (same dtype as `q`)
/// - `g`, `beta`: `[batch, seq, num_v_heads]` (same dtype as `q`)
/// - `state`: `[batch, num_v_heads, k_dim, v_dim]` (`F32`), updated in place
/// - `apply_q_scale`: when `true`, scales `q` by `1 / sqrt(k_dim)` (matching Engine's
///   `gated_delta_rule_recurrence` contract).
///
/// Returns `Ok(Some(out))` with shape `[batch, seq, num_v_heads, v_dim]` in `q.dtype()`.
#[allow(clippy::too_many_arguments)]
pub fn try_recurrence(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
    apply_q_scale: bool,
) -> Result<Option<Tensor>> {
    if dsl::gdn_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    let (b, seq, nh, kd) = q.dims4()?;
    let vd = v.dim(3)?;
    if !matches!((kd, vd), (128, 128) | (64, 64)) {
        return Ok(None);
    }
    let dtype = q.dtype();
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Ok(None);
    }
    if state.dims() != [b, nh, kd, vd] {
        return Ok(None);
    }
    #[cfg(feature = "vulkan")]
    if q.device().is_vulkan() && dtype != DType::F32 {
        return Ok(None);
    }

    let bh = b * nh;
    let q_scaled = if apply_q_scale {
        let scale = 1.0 / (kd as f64).sqrt();
        (q.to_dtype(DType::F32)? * scale)?.to_dtype(dtype)?
    } else {
        q.clone()
    };

    // Transpose [B, S, H, D] -> [BH, S, D] (no-op reshape when S == 1).
    let to_bh_sd = |t: &Tensor, d: usize| -> Result<Tensor> {
        let t = if t.dtype() == dtype {
            t.clone()
        } else {
            t.to_dtype(dtype)?
        };
        if seq == 1 {
            t.reshape((bh, 1, d))?.contiguous()
        } else {
            t.transpose(1, 2)?.contiguous()?.reshape((bh, seq, d))
        }
    };
    let to_bh_s = |t: &Tensor| -> Result<Tensor> {
        let t = if t.dtype() == dtype {
            t.clone()
        } else {
            t.to_dtype(dtype)?
        };
        if seq == 1 {
            t.reshape((bh, 1))?.contiguous()
        } else {
            t.transpose(1, 2)?.contiguous()?.reshape((bh, seq))
        }
    };

    let q_bh = to_bh_sd(&q_scaled, kd)?;
    let k_bh = to_bh_sd(k, kd)?;
    let v_bh = to_bh_sd(v, vd)?;
    let g_bh = to_bh_s(g)?;
    let beta_bh = to_bh_s(beta)?;

    let orig_state_dtype = state.dtype();
    let mut state_f32 = if orig_state_dtype == DType::F32
        && state.is_contiguous()
        && (!q.device().is_vulkan() || state.layout().start_offset() == 0)
    {
        state.clone()
    } else {
        state.to_dtype(DType::F32)?.contiguous()?
    };

    let out_bh: Tensor = match q.device() {
        #[cfg(feature = "cuda")]
        crate::Device::Cuda(dev) => {
            let (qs, ql) = q_bh.storage_and_layout();
            let (ks, kl) = k_bh.storage_and_layout();
            let (vs, vl) = v_bh.storage_and_layout();
            let (gs, gl) = g_bh.storage_and_layout();
            let (bs, bl) = beta_bh.storage_and_layout();
            let (mut ss, sl) = state_f32.storage_mut_and_layout();
            let (
                Storage::Cuda(qs),
                Storage::Cuda(ks),
                Storage::Cuda(vs),
                Storage::Cuda(gs),
                Storage::Cuda(bs),
                Storage::Cuda(ss),
            ) = (&*qs, &*ks, &*vs, &*gs, &*bs, &mut *ss)
            else {
                return Ok(None);
            };
            let out_s = dev.gdn_scan_lanes_io_dsl(
                dtype,
                (qs, ql.start_offset()),
                (ks, kl.start_offset()),
                (vs, vl.start_offset()),
                (gs, gl.start_offset()),
                (bs, bl.start_offset()),
                (ss, sl.start_offset()),
                bh,
                seq,
                kd,
                vd,
            )?;
            Tensor::from_storage(
                Storage::Cuda(out_s),
                (bh, seq, vd),
                BackpropOp::none(),
                false,
            )
        }
        #[cfg(feature = "metal")]
        crate::Device::Metal(dev) => {
            let elem_bytes = dtype.size_in_bytes();
            let out_buf = dev.new_buffer(bh * seq * vd, dtype, "gdn_scan_lanes_io_dsl_out")?;
            {
                let (qs, ql) = q_bh.storage_and_layout();
                let (ks, kl) = k_bh.storage_and_layout();
                let (vs, vl) = v_bh.storage_and_layout();
                let (gs, gl) = g_bh.storage_and_layout();
                let (bs, bl) = beta_bh.storage_and_layout();
                let (mut ss, sl) = state_f32.storage_mut_and_layout();
                let (
                    Storage::Metal(qs),
                    Storage::Metal(ks),
                    Storage::Metal(vs),
                    Storage::Metal(gs),
                    Storage::Metal(bs),
                    Storage::Metal(ss),
                ) = (&*qs, &*ks, &*vs, &*gs, &*bs, &mut *ss)
                else {
                    return Ok(None);
                };
                dev.gdn_scan_lanes_io_dsl(
                    dtype,
                    (qs.buffer(), ql.start_offset() * elem_bytes),
                    (ks.buffer(), kl.start_offset() * elem_bytes),
                    (vs.buffer(), vl.start_offset() * elem_bytes),
                    (gs.buffer(), gl.start_offset() * elem_bytes),
                    (bs.buffer(), bl.start_offset() * elem_bytes),
                    (ss.buffer(), sl.start_offset() * 4),
                    (&out_buf, 0),
                    bh,
                    seq,
                    kd,
                    vd,
                )?;
            }
            let out_s = crate::MetalStorage::new(out_buf, dev.clone(), bh * seq * vd, dtype);
            Tensor::from_storage(
                Storage::Metal(out_s),
                (bh, seq, vd),
                BackpropOp::none(),
                false,
            )
        }
        #[cfg(feature = "vulkan")]
        crate::Device::Vulkan(dev) => {
            let q_c = ensure_contiguous_offset_zero(&q_bh)?;
            let k_c = ensure_contiguous_offset_zero(&k_bh)?;
            let v_c = ensure_contiguous_offset_zero(&v_bh)?;
            let g_c = ensure_contiguous_offset_zero(&g_bh)?;
            let b_c = ensure_contiguous_offset_zero(&beta_bh)?;
            let (qs, _) = q_c.storage_and_layout();
            let (ks, _) = k_c.storage_and_layout();
            let (vs, _) = v_c.storage_and_layout();
            let (gs, _) = g_c.storage_and_layout();
            let (bs, _) = b_c.storage_and_layout();
            let (mut ss, _) = state_f32.storage_mut_and_layout();
            let (
                Storage::Vulkan(qs),
                Storage::Vulkan(ks),
                Storage::Vulkan(vs),
                Storage::Vulkan(gs),
                Storage::Vulkan(bs),
                Storage::Vulkan(ss),
            ) = (&*qs, &*ks, &*vs, &*gs, &*bs, &mut *ss)
            else {
                return Ok(None);
            };
            let Some(out_s) = dev.gdn_scan_lanes_io_dsl(qs, ks, vs, gs, bs, ss, bh, seq, kd, vd)?
            else {
                return Ok(None);
            };
            Tensor::from_storage(
                Storage::Vulkan(out_s),
                (bh, seq, vd),
                BackpropOp::none(),
                false,
            )
        }
        _ => return Ok(None),
    };

    // Write back if we had to convert or copy `state`.
    if orig_state_dtype != DType::F32 {
        *state = state_f32.to_dtype(orig_state_dtype)?;
    } else if !state.is_contiguous() || (q.device().is_vulkan() && state.layout().start_offset() != 0)
    {
        *state = state_f32;
    }

    let out = if seq == 1 {
        out_bh.reshape((b, 1, nh, vd))?
    } else {
        out_bh
            .reshape((b, nh, seq, vd))?
            .transpose(1, 2)?
            .contiguous()?
    };
    Ok(Some(out))
}
