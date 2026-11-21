//! Backend-independent DSL dispatch for Gated Delta Net (GDN) primitives.
//!
//! Controlled by `HANZO_GDN_IMPL=dsl|legacy` (or [`set_gdn_impl`]), defaulting to `legacy`.
//! When enabled and the device/dtype/specialization is supported by a generated DSL kernel,
//! returns `Ok(Some(...))` and records the dispatch in [`crate::quantized::dsl`]; otherwise returns
//! `Ok(None)` so the caller falls back cleanly to its existing legacy path.

pub use crate::quantized::dsl::{gdn_impl, set_gdn_impl, QkImpl};
#[cfg(feature = "metal")]
use crate::{DType, Device};
use crate::{Result, Tensor};

#[cfg(feature = "metal")]
fn metal_artifact_name(dtype: DType, f32_name: &'static str, f16_name: &'static str, bf16_name: &'static str) -> Option<&'static str> {
    match dtype {
        DType::F32 => Some(f32_name),
        DType::F16 => Some(f16_name),
        DType::BF16 => Some(bf16_name),
        _ => None,
    }
}

#[cfg(feature = "metal")]
fn ensure_contiguous(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// Depthwise causal conv1d + SiLU over a sequence:
/// `x`: `[batch, conv_dim, seq_len]`, `weight`: `[conv_dim, kernel_size]`.
/// Returns `[batch, conv_dim, seq_len]`.
pub fn gdn_conv1d_full(
    x: &Tensor,
    weight: &Tensor,
    kernel_size: usize,
) -> Result<Option<Tensor>> {
    if gdn_impl() != QkImpl::Dsl || kernel_size != 4 {
        return Ok(None);
    }
    let (batch, conv_dim, seq_len) = x.dims3()?;
    if batch == 0 || conv_dim == 0 || seq_len == 0 {
        return Ok(None);
    }
    if weight.dims2()? != (conv_dim, kernel_size) || x.dtype() != weight.dtype() {
        return Ok(None);
    }
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = x.device() {
        let Some(artifact) = metal_artifact_name(
            x.dtype(),
            "gdn_conv1d_k4",
            "gdn_conv1d_k4_f16",
            "gdn_conv1d_k4_bf16",
        ) else {
            return Ok(None);
        };
        let x = ensure_contiguous(x)?;
        let weight = ensure_contiguous(weight)?;
        let out = crate::metal_backend::metal_uninit_tensor(dev, x.shape().clone(), x.dtype())?;
        let xb = crate::metal_backend::metal_tensor_buf(&x)?;
        let wb = crate::metal_backend::metal_tensor_buf(&weight)?;
        let ob = crate::metal_backend::metal_tensor_buf(&out)?;
        dev.gdn_conv1d_dsl_as(
            (&xb.0, xb.1),
            (&wb.0, wb.1),
            (&ob.0, ob.1),
            batch,
            conv_dim,
            seq_len,
            x.dtype(),
        )?;
        crate::quantized::dsl::note_dispatch(QkImpl::Dsl, artifact, conv_dim, seq_len);
        return Ok(Some(out));
    }
    let _ = weight;
    Ok(None)
}

/// Single-step causal conv1d + SiLU (`seq_len == 1`):
/// `x`: `[batch, conv_dim]` or `[batch, conv_dim, 1]`, `weight`: `[conv_dim, 4]`,
/// `conv_state`: `[batch, conv_dim, 4]` (updated via copy-on-write replacement so snapshots are preserved).
/// Returns `[batch, conv_dim]` matching `x`'s rank (`[batch, conv_dim]` or `[batch, conv_dim, 1]`).
pub fn gdn_conv1d_step(
    x: &Tensor,
    weight: &Tensor,
    conv_state: &mut Tensor,
    kernel_size: usize,
) -> Result<Option<Tensor>> {
    if gdn_impl() != QkImpl::Dsl || kernel_size != 4 {
        return Ok(None);
    }
    let (batch, conv_dim, rank3) = match x.dims() {
        [b, c] => (*b, *c, false),
        [b, c, 1] => (*b, *c, true),
        _ => return Ok(None),
    };
    if batch == 0 || conv_dim == 0 {
        return Ok(None);
    }
    if weight.dims2()? != (conv_dim, kernel_size)
        || conv_state.dims3()? != (batch, conv_dim, kernel_size)
        || x.dtype() != weight.dtype()
        || x.dtype() != conv_state.dtype()
    {
        return Ok(None);
    }
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = x.device() {
        let Some(artifact) = metal_artifact_name(
            x.dtype(),
            "gdn_conv1d_step_k4_f32",
            "gdn_conv1d_step_k4_f16",
            "gdn_conv1d_step_k4_bf16",
        ) else {
            return Ok(None);
        };
        let x_c = ensure_contiguous(x)?;
        let weight_c = ensure_contiguous(weight)?;
        let next_state = crate::metal_backend::metal_fresh_copy(conv_state, x.dtype())?;
        let out_shape = if rank3 {
            crate::Shape::from((batch, conv_dim, 1))
        } else {
            crate::Shape::from((batch, conv_dim))
        };
        let out = crate::metal_backend::metal_uninit_tensor(dev, out_shape, x.dtype())?;
        let xb = crate::metal_backend::metal_tensor_buf(&x_c)?;
        let wb = crate::metal_backend::metal_tensor_buf(&weight_c)?;
        let csb = crate::metal_backend::metal_tensor_buf(&next_state)?;
        let ob = crate::metal_backend::metal_tensor_buf(&out)?;
        dev.gdn_conv1d_step_dsl(
            (&xb.0, xb.1),
            (&wb.0, wb.1),
            (&csb.0, csb.1),
            (&ob.0, ob.1),
            batch,
            conv_dim,
            x.dtype(),
        )?;
        *conv_state = next_state;
        crate::quantized::dsl::note_dispatch(QkImpl::Dsl, artifact, batch, conv_dim);
        return Ok(Some(out));
    }
    let _ = (weight, conv_state, rank3);
    Ok(None)
}

/// Fused GDN gating:
/// `beta = sigmoid(b)`, `g = -exp(a_log) * softplus(a + dt_bias)` (or `a_log * softplus(...)`
/// when `precomputed_a` is true, e.g. GGUF `ssm_a`).
/// `b` and `a` have shape `[.., num_heads]` in `F32 | F16 | BF16`;
/// `a_log` and `dt_bias` have shape `[num_heads]` (cast to `F32` if needed).
/// Returns `(beta, g)` matching `b`'s shape and dtype.
pub fn gdn_gating(
    b: &Tensor,
    a: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    precomputed_a: bool,
) -> Result<Option<(Tensor, Tensor)>> {
    if gdn_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    if b.shape() != a.shape() || b.dtype() != a.dtype() || b.elem_count() == 0 {
        return Ok(None);
    }
    let num_heads = a_log.dims1()?;
    if num_heads == 0 || dt_bias.dims1()? != num_heads || b.elem_count() % num_heads != 0 {
        return Ok(None);
    }
    let batch_tokens = b.elem_count() / num_heads;
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = b.device() {
        let Some(artifact) =
            metal_artifact_name(b.dtype(), "gdn_gating", "gdn_gating_f16", "gdn_gating_bf16")
        else {
            return Ok(None);
        };
        let b_c = ensure_contiguous(b)?;
        let a_c = ensure_contiguous(a)?;
        let a_log_f32 = ensure_contiguous(&if a_log.dtype() == DType::F32 {
            a_log.clone()
        } else {
            a_log.to_dtype(DType::F32)?
        })?;
        let dt_bias_f32 = ensure_contiguous(&if dt_bias.dtype() == DType::F32 {
            dt_bias.clone()
        } else {
            dt_bias.to_dtype(DType::F32)?
        })?;
        let beta = crate::metal_backend::metal_uninit_tensor(dev, b.shape().clone(), b.dtype())?;
        let g = crate::metal_backend::metal_uninit_tensor(dev, b.shape().clone(), b.dtype())?;
        let bb = crate::metal_backend::metal_tensor_buf(&b_c)?;
        let ab = crate::metal_backend::metal_tensor_buf(&a_c)?;
        let alb = crate::metal_backend::metal_tensor_buf(&a_log_f32)?;
        let dtb = crate::metal_backend::metal_tensor_buf(&dt_bias_f32)?;
        let betab = crate::metal_backend::metal_tensor_buf(&beta)?;
        let gb = crate::metal_backend::metal_tensor_buf(&g)?;
        dev.gdn_gating_dsl_as(
            (&bb.0, bb.1),
            (&ab.0, ab.1),
            (&alb.0, alb.1),
            (&dtb.0, dtb.1),
            (&betab.0, betab.1),
            (&gb.0, gb.1),
            batch_tokens,
            num_heads,
            precomputed_a,
            b.dtype(),
        )?;
        crate::quantized::dsl::note_dispatch(QkImpl::Dsl, artifact, batch_tokens, num_heads);
        return Ok(Some((beta, g)));
    }
    let _ = (a_log, dt_bias, batch_tokens, precomputed_a);
    Ok(None)
}

/// Recurrent gated delta-rule scan in `[bh, seq, dim]` layout (`q` pre-scaled by `1/sqrt(k_dim)`):
/// `q`, `k`: `[bh, seq, k_dim]`, `v`: `[bh, seq, v_dim]`, `g`, `beta`: `[bh, seq]`,
/// `state`: `[bh, k_dim, v_dim]` (updated via copy-on-write replacement so snapshots are preserved).
/// Returns `out: [bh, seq, v_dim]`.
pub fn gdn_scan(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    state: &mut Tensor,
) -> Result<Option<Tensor>> {
    if gdn_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    let (bh, seq, k_dim) = q.dims3()?;
    let (bh_v, seq_v, v_dim) = v.dims3()?;
    if bh == 0 || seq == 0 || (bh_v, seq_v) != (bh, seq) || k.dims3()? != (bh, seq, k_dim) {
        return Ok(None);
    }
    if g.dims2()? != (bh, seq) || beta.dims2()? != (bh, seq) {
        return Ok(None);
    }
    if state.dims3()? != (bh, k_dim, v_dim) {
        return Ok(None);
    }
    let dtype = q.dtype();
    if k.dtype() != dtype || v.dtype() != dtype || g.dtype() != dtype || beta.dtype() != dtype {
        return Ok(None);
    }
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = q.device() {
        let lanes = match (k_dim, v_dim) {
            (64, 64) => 4usize,
            (128, 128) => if seq >= 16 { 16usize } else { 8usize },
            _ => return Ok(None),
        };
        let artifact = match (k_dim, v_dim, lanes, dtype) {
            (64, 64, 4, DType::F32) => "gdn_scan_lanes_k64_v64_l4",
            (64, 64, 4, DType::F16) => "gdn_scan_lanes_k64_v64_l4_f16",
            (64, 64, 4, DType::BF16) => "gdn_scan_lanes_k64_v64_l4_bf16",
            (128, 128, 8, DType::F32) => "gdn_scan_lanes_k128_v128_l8",
            (128, 128, 8, DType::F16) => "gdn_scan_lanes_k128_v128_l8_f16",
            (128, 128, 8, DType::BF16) => "gdn_scan_lanes_k128_v128_l8_bf16",
            (128, 128, 16, DType::F32) => "gdn_scan_lanes_k128_v128_l16_f32",
            (128, 128, 16, DType::F16) => "gdn_scan_lanes_k128_v128_l16_f16",
            (128, 128, 16, DType::BF16) => "gdn_scan_lanes_k128_v128_l16_bf16",
            _ => return Ok(None),
        };
        let q_c = ensure_contiguous(q)?;
        let k_c = ensure_contiguous(k)?;
        let v_c = ensure_contiguous(v)?;
        let g_c = ensure_contiguous(g)?;
        let beta_c = ensure_contiguous(beta)?;
        let next_state = crate::metal_backend::metal_fresh_copy(state, DType::F32)?;
        let out = crate::metal_backend::metal_uninit_tensor(dev, (bh, seq, v_dim), dtype)?;
        let qb = crate::metal_backend::metal_tensor_buf(&q_c)?;
        let kb = crate::metal_backend::metal_tensor_buf(&k_c)?;
        let vb = crate::metal_backend::metal_tensor_buf(&v_c)?;
        let gb = crate::metal_backend::metal_tensor_buf(&g_c)?;
        let betab = crate::metal_backend::metal_tensor_buf(&beta_c)?;
        let sb = crate::metal_backend::metal_tensor_buf(&next_state)?;
        let ob = crate::metal_backend::metal_tensor_buf(&out)?;
        dev.gdn_scan_lanes_dsl_as(
            (&qb.0, qb.1),
            (&kb.0, kb.1),
            (&vb.0, vb.1),
            (&gb.0, gb.1),
            (&betab.0, betab.1),
            (&sb.0, sb.1),
            (&ob.0, ob.1),
            bh,
            seq,
            k_dim,
            v_dim,
            lanes,
            dtype,
        )?;
        *state = next_state;
        crate::quantized::dsl::note_dispatch(QkImpl::Dsl, artifact, bh, seq);
        return Ok(Some(out));
    }
    let _ = state;
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DType, Device};

    #[test]
    fn gdn_legacy_returns_none_by_default() -> Result<()> {
        let prev = gdn_impl();
        set_gdn_impl(QkImpl::Legacy);
        let x = Tensor::zeros((1, 4, 4), DType::F32, &Device::Cpu)?;
        let w = Tensor::zeros((4, 4), DType::F32, &Device::Cpu)?;
        assert!(gdn_conv1d_full(&x, &w, 4)?.is_none());
        set_gdn_impl(prev);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn gdn_dsl_metal_f16_bf16_offsets_and_snapshot_isolation() -> Result<()> {
        let prev = gdn_impl();
        set_gdn_impl(QkImpl::Dsl);
        let dev = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let (batch, cdim, seq) = (2usize, 8usize, 7usize);
            // Offset views via narrow(0, 1, batch) exercise nonzero start_offset().
            let x_host: Vec<f32> = (0..(batch + 1) * cdim * seq)
                .map(|i| ((i * 17 % 29) as f32 - 14.0) * 0.05)
                .collect();
            let w_host: Vec<f32> = (0..(cdim + 1) * 4)
                .map(|i| ((i * 13 % 19) as f32 - 9.0) * 0.04)
                .collect();
            let x_full = Tensor::from_vec(x_host, (batch + 1, cdim, seq), &dev)?.to_dtype(dtype)?;
            let w_full = Tensor::from_vec(w_host, (cdim + 1, 4), &dev)?.to_dtype(dtype)?;
            let x = x_full.narrow(0, 1, batch)?;
            let w = w_full.narrow(0, 1, cdim)?;
            assert!(x.layout().start_offset() > 0 && w.layout().start_offset() > 0);

            let full_out = gdn_conv1d_full(&x, &w, 4)?.expect("metal dsl conv1d_full");
            let mut conv_state = Tensor::zeros((batch, cdim, 4), dtype, &dev)?;
            let snap_before = conv_state.clone();
            let mut stepped = Vec::with_capacity(seq);
            for t in 0..seq {
                let xt = x.narrow(2, t, 1)?.squeeze(2)?;
                let yt = gdn_conv1d_step(&xt, &w, &mut conv_state, 4)?.expect("conv1d_step");
                stepped.push(yt.unsqueeze(2)?);
            }
            // Snapshot taken before stepping must remain all zeros (no aliasing mutation).
            let snap_sum: f32 = snap_before
                .to_dtype(DType::F32)?
                .abs()?
                .sum_all()?
                .to_scalar()?;
            assert_eq!(snap_sum, 0.0, "conv_state snapshot was mutated in place");

            let cat = Tensor::cat(&stepped, 2)?;
            let diff: f32 = (full_out.to_dtype(DType::F32)? - cat.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let tol = if dtype == DType::F32 { 1e-5 } else { 2e-2 };
            assert!(diff <= tol, "conv1d {dtype:?} diff={diff}");

            // Gating + scan equivalence + state snapshot isolation.
            let (bh, k_dim, v_dim) = (4usize, 64usize, 64usize);
            let q = Tensor::randn(0f32, 0.2, (bh, seq, k_dim), &dev)?.to_dtype(dtype)?;
            let k = Tensor::randn(0f32, 0.2, (bh, seq, k_dim), &dev)?.to_dtype(dtype)?;
            let v = Tensor::randn(0f32, 0.2, (bh, seq, v_dim), &dev)?.to_dtype(dtype)?;
            let b_raw = Tensor::randn(0f32, 0.5, (bh, seq), &dev)?.to_dtype(dtype)?;
            let a_raw = Tensor::randn(0f32, 0.5, (bh, seq), &dev)?.to_dtype(dtype)?;
            let a_log = Tensor::randn(-2f32, 0.2, (bh,), &dev)?;
            let dt_bias = Tensor::randn(0f32, 0.2, (bh,), &dev)?;
            let (beta, g) = gdn_gating(
                &b_raw.transpose(0, 1)?,
                &a_raw.transpose(0, 1)?,
                &a_log,
                &dt_bias,
                false,
            )?
            .expect("metal dsl gating");
            let neg_exp_a = a_log.exp()?.neg()?;
            let (beta_p, g_p) = gdn_gating(
                &b_raw.transpose(0, 1)?,
                &a_raw.transpose(0, 1)?,
                &neg_exp_a,
                &dt_bias,
                true,
            )?
            .expect("metal dsl gating precomputed");
            let gp_diff: f32 = (g.to_dtype(DType::F32)? - g_p.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar()?;
            let bp_diff: f32 = (beta.to_dtype(DType::F32)? - beta_p.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar()?;
            assert!(gp_diff <= tol && bp_diff <= tol, "precomputed gating diff {gp_diff} {bp_diff}");
            let beta = beta.transpose(0, 1)?.contiguous()?;
            let g = g.transpose(0, 1)?.contiguous()?;

            let mut state_full = Tensor::zeros((bh, k_dim, v_dim), DType::F32, &dev)?;
            let state_snap = state_full.clone();
            let y_full = gdn_scan(&q, &k, &v, &g, &beta, &mut state_full)?.expect("gdn_scan full");
            let snap_state_sum: f32 = state_snap.abs()?.sum_all()?.to_scalar()?;
            assert_eq!(snap_state_sum, 0.0, "state snapshot was mutated in place");

            let mut state_step = Tensor::zeros((bh, k_dim, v_dim), DType::F32, &dev)?;
            let mut step_ys = Vec::with_capacity(seq);
            for t in 0..seq {
                let y_t = gdn_scan(
                    &q.narrow(1, t, 1)?,
                    &k.narrow(1, t, 1)?,
                    &v.narrow(1, t, 1)?,
                    &g.narrow(1, t, 1)?,
                    &beta.narrow(1, t, 1)?,
                    &mut state_step,
                )?
                .expect("gdn_scan step");
                step_ys.push(y_t);
            }
            let y_cat = Tensor::cat(&step_ys, 1)?;
            let scan_diff: f32 = (y_full.to_dtype(DType::F32)? - y_cat.to_dtype(DType::F32)?)?
                .abs()?
                .max_all()?
                .to_scalar::<f32>()?;
            let s_diff: f32 = (&state_full - &state_step)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(scan_diff <= tol && s_diff <= tol, "scan {dtype:?} out={scan_diff} st={s_diff}");
        }
        set_gdn_impl(prev);
        Ok(())
    }
}
