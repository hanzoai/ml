//! Backend-independent DSL dispatch for FlashAttention, PagedAttention decode, and paged KV
//! cache updates.
//!
//! Controlled by `HANZO_FLASH_IMPL=dsl|legacy` (or [`set_flash_impl`]), defaulting to `legacy`.
//! When enabled and the device/dtype/head-dim specialization is supported, dispatches the
//! generated DSL kernel (`hanzo-kernel/src/flash.rs`) and records stats via
//! [`crate::quantized::dsl`]; otherwise returns `Ok(None)` so callers retain their legacy fallback.

pub use crate::quantized::dsl::{flash_impl, set_flash_impl, QkImpl};
use crate::{DType, Result, Tensor};

#[cfg(feature = "metal")]
use crate::Device;

fn ensure_contiguous(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

fn ensure_u32(t: &Tensor) -> Result<Option<Tensor>> {
    match t.dtype() {
        DType::U32 => Ok(Some(ensure_contiguous(t)?)),
        DType::I32 => Ok(Some(t.to_dtype(DType::U32)?.contiguous()?)),
        _ => Ok(None),
    }
}

/// Scaled dot-product FlashAttention over `[b, n_heads, seq_q, d]` and KV `[b, n_kv, seq_k, d]`:
/// - Supports GQA (`n_heads % n_kv == 0`), `d in {64, 128, 256}`, `F32 | F16 | BF16`.
/// - `causal_mode`: `0` non-causal, `1` top-left (`k <= q`), `2` bottom-right aligned (`k <= seq_k - seq_q + q`).
/// - Reads `k` and `v` in place if they share strides and have unit innermost stride.
pub fn flash_attn(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f32,
    causal_mode: u32,
) -> Result<Option<Tensor>> {
    if flash_impl() != QkImpl::Dsl || causal_mode > 2 {
        return Ok(None);
    }
    let (batch, n_heads, seq_q, d) = q.dims4()?;
    let (kb, n_kv, seq_k, kd) = k.dims4()?;
    let (vb, v_nkv, v_seqk, vd) = v.dims4()?;
    if batch == 0
        || n_heads == 0
        || n_kv == 0
        || seq_q == 0
        || seq_k == 0
        || kb != batch
        || vb != batch
        || v_nkv != n_kv
        || v_seqk != seq_k
        || kd != d
        || vd != d
        || n_heads % n_kv != 0
        || !matches!(d, 64 | 128 | 256)
    {
        return Ok(None);
    }
    let dtype = q.dtype();
    if k.dtype() != dtype || v.dtype() != dtype || !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Ok(None);
    }
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = q.device() {
        let artifact = match (d, dtype) {
            (64, DType::F32) => "flash_attn_d64_p32",
            (64, DType::F16) => "flash_attn_d64_p32_f16",
            (64, DType::BF16) => "flash_attn_d64_p32_bf16",
            (128, DType::F32) => "flash_attn_d128_p32",
            (128, DType::F16) => "flash_attn_d128_p32_f16",
            (128, DType::BF16) => "flash_attn_d128_p32_bf16",
            (256, DType::F32) => "flash_attn_d256_p32_f32",
            (256, DType::F16) => "flash_attn_d256_p32_f16",
            (256, DType::BF16) => "flash_attn_d256_p32_bf16",
            _ => return Ok(None),
        };
        let q_c = ensure_contiguous(q)?;
        let (k_c, v_c) = if k.stride()[3] == 1 && v.stride()[3] == 1 && k.stride() == v.stride() {
            (k.clone(), v.clone())
        } else {
            (k.contiguous()?, v.contiguous()?)
        };
        let ks = k_c.stride();
        let kv_strides = [ks[0], ks[1], ks[2]];
        let out = crate::metal_backend::metal_uninit_tensor(dev, (batch, n_heads, seq_q, d), dtype)?;
        let qb = crate::metal_backend::metal_tensor_buf(&q_c)?;
        let kb = crate::metal_backend::metal_tensor_buf(&k_c)?;
        let vb = crate::metal_backend::metal_tensor_buf(&v_c)?;
        let ob = crate::metal_backend::metal_tensor_buf(&out)?;
        dev.flash_attn_dsl_as(
            (&qb.0, qb.1),
            (&kb.0, kb.1),
            (&vb.0, vb.1),
            (&ob.0, ob.1),
            batch,
            n_heads,
            n_kv,
            seq_q,
            seq_k,
            d,
            causal_mode,
            scale,
            kv_strides,
            0,
            dtype,
        )?;
        crate::quantized::dsl::note_dispatch(QkImpl::Dsl, artifact, seq_q, seq_k);
        return Ok(Some(out));
    }
    let _ = (scale, ensure_u32);
    Ok(None)
}

/// Paged decode attention over physical KV blocks (`hanzo-paged-attn` layout):
/// - `q`: `[num_seqs, num_heads, d]`
/// - `key_cache`: `[num_blocks, num_kv_heads, d/x, block_size, x]`
/// - `value_cache`: `[num_blocks, num_kv_heads, d, block_size]`
/// - `block_tables`: `[num_seqs, max_blocks_per_seq]` (`u32` or `i32`)
/// - `context_lens`: `[num_seqs]` (`u32` or `i32`)
/// - `softcap`: `<= 0.0` disabled.
/// Returns `out: [num_seqs, num_heads, d]`.
#[allow(clippy::too_many_arguments)]
pub fn paged_attn(
    q: &Tensor,
    key_cache: &Tensor,
    value_cache: &Tensor,
    block_tables: &Tensor,
    context_lens: &Tensor,
    scale: f32,
    softcap: f32,
    sliding_window: usize,
) -> Result<Option<Tensor>> {
    if flash_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    let (num_seqs, num_heads, d) = q.dims3()?;
    let (num_blocks, num_kv_heads, d_div_x, block_size, x) = key_cache.dims5()?;
    let (vb, vnkv, vd, vbs) = value_cache.dims4()?;
    let (bt_seqs, max_blocks) = block_tables.dims2()?;
    let cl_seqs = context_lens.dims1()?;
    if num_seqs == 0
        || num_heads == 0
        || num_kv_heads == 0
        || block_size == 0
        || x == 0
        || d_div_x * x != d
        || vb != num_blocks
        || vnkv != num_kv_heads
        || vd != d
        || vbs != block_size
        || bt_seqs != num_seqs
        || cl_seqs != num_seqs
        || num_heads % num_kv_heads != 0
        || !matches!(d, 64 | 128 | 256)
    {
        return Ok(None);
    }
    let dtype = q.dtype();
    if key_cache.dtype() != dtype || value_cache.dtype() != dtype {
        return Ok(None);
    }
    let expected_x = match dtype {
        DType::F32 => 4usize,
        DType::F16 | DType::BF16 => 8usize,
        _ => return Ok(None),
    };
    if x != expected_x || !key_cache.is_contiguous() || !value_cache.is_contiguous() {
        return Ok(None);
    }
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = q.device() {
        let Some(bt_u32) = ensure_u32(block_tables)? else {
            return Ok(None);
        };
        let Some(cl_u32) = ensure_u32(context_lens)? else {
            return Ok(None);
        };
        let artifact = match (d, x, dtype) {
            (64, 4, DType::F32) => "paged_attn_d64_x4_f32",
            (64, 8, DType::F16) => "paged_attn_d64_x8_f16",
            (64, 8, DType::BF16) => "paged_attn_d64_x8_bf16",
            (128, 4, DType::F32) => "paged_attn_d128_x4_f32",
            (128, 8, DType::F16) => "paged_attn_d128_x8_f16",
            (128, 8, DType::BF16) => "paged_attn_d128_x8_bf16",
            (256, 4, DType::F32) => "paged_attn_d256_x4_f32",
            (256, 8, DType::F16) => "paged_attn_d256_x8_f16",
            (256, 8, DType::BF16) => "paged_attn_d256_x8_bf16",
            _ => return Ok(None),
        };
        let q_c = ensure_contiguous(q)?;
        let out = crate::metal_backend::metal_uninit_tensor(dev, (num_seqs, num_heads, d), dtype)?;
        let qb = crate::metal_backend::metal_tensor_buf(&q_c)?;
        let kcb = crate::metal_backend::metal_tensor_buf(key_cache)?;
        let vcb = crate::metal_backend::metal_tensor_buf(value_cache)?;
        let btb = crate::metal_backend::metal_tensor_buf(&bt_u32)?;
        let clb = crate::metal_backend::metal_tensor_buf(&cl_u32)?;
        let ob = crate::metal_backend::metal_tensor_buf(&out)?;
        dev.paged_attn_dsl(
            (&qb.0, qb.1),
            (&kcb.0, kcb.1),
            (&vcb.0, vcb.1),
            (&btb.0, btb.1),
            (&clb.0, clb.1),
            (&ob.0, ob.1),
            num_seqs,
            num_heads,
            num_kv_heads,
            d,
            block_size,
            x,
            max_blocks,
            scale,
            softcap,
            sliding_window,
            dtype,
        )?;
        crate::quantized::dsl::note_dispatch(QkImpl::Dsl, artifact, num_heads, d);
        return Ok(Some(out));
    }
    let _ = (max_blocks, scale, softcap, sliding_window);
    Ok(None)
}

/// Write `[num_tokens, num_kv_heads, d]` keys/values into paged physical caches in place.
/// `slot_mapping` may be 1D `[num_tokens]` or 2D `[batch, seq]` in `I64` or `I32`.
pub fn reshape_and_cache(
    key: &Tensor,
    value: &Tensor,
    key_cache: &mut Tensor,
    value_cache: &mut Tensor,
    slot_mapping: &Tensor,
) -> Result<Option<()>> {
    if flash_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    let (num_tokens, num_kv_heads, d) = key.dims3()?;
    if value.dims3()? != (num_tokens, num_kv_heads, d) || num_tokens == 0 || d == 0 {
        return Ok(None);
    }
    let (num_blocks, kvn, d_div_x, block_size, x) = key_cache.dims5()?;
    if kvn != num_kv_heads
        || d_div_x * x != d
        || value_cache.dims4()? != (num_blocks, num_kv_heads, d, block_size)
        || slot_mapping.elem_count() != num_tokens
    {
        return Ok(None);
    }
    let dtype = key.dtype();
    if value.dtype() != dtype || key_cache.dtype() != dtype || value_cache.dtype() != dtype {
        return Ok(None);
    }
    let expected_x = match dtype {
        DType::F32 => 4usize,
        DType::F16 | DType::BF16 => 8usize,
        _ => return Ok(None),
    };
    if x != expected_x || !key_cache.is_contiguous() || !value_cache.is_contiguous() {
        return Ok(None);
    }
    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = key.device() {
        let slots_i64 = match slot_mapping.dtype() {
            DType::I64 => ensure_contiguous(slot_mapping)?,
            DType::I32 => slot_mapping.to_dtype(DType::I64)?.contiguous()?,
            _ => return Ok(None),
        };
        let artifact = match (x, dtype) {
            (4, DType::F32) => "reshape_and_cache_x4_f32",
            (8, DType::F16) => "reshape_and_cache_x8_f16",
            (8, DType::BF16) => "reshape_and_cache_x8_bf16",
            _ => return Ok(None),
        };
        let k_c = ensure_contiguous(key)?;
        let v_c = ensure_contiguous(value)?;
        let kb = crate::metal_backend::metal_tensor_buf(&k_c)?;
        let vb = crate::metal_backend::metal_tensor_buf(&v_c)?;
        let kcb = crate::metal_backend::metal_tensor_buf(key_cache)?;
        let vcb = crate::metal_backend::metal_tensor_buf(value_cache)?;
        let sb = crate::metal_backend::metal_tensor_buf(&slots_i64)?;
        dev.reshape_and_cache_dsl(
            (&kb.0, kb.1),
            (&vb.0, vb.1),
            (&kcb.0, kcb.1),
            (&vcb.0, vcb.1),
            (&sb.0, sb.1),
            num_tokens,
            num_kv_heads,
            d,
            block_size,
            x,
            dtype,
        )?;
        crate::quantized::dsl::note_dispatch(QkImpl::Dsl, artifact, num_tokens, d);
        return Ok(Some(()));
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn flash_legacy_returns_none_by_default() -> Result<()> {
        let prev = flash_impl();
        set_flash_impl(QkImpl::Legacy);
        let q = Tensor::zeros((1, 2, 4, 64), DType::F32, &Device::Cpu)?;
        assert!(flash_attn(&q, &q, &q, 0.125, 1)?.is_none());
        set_flash_impl(prev);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn attn_dsl_metal_f16_bf16_d256_causal2_and_paged() -> Result<()> {
        let prev = flash_impl();
        set_flash_impl(QkImpl::Dsl);
        let dev = Device::new_metal(0)?;
        for dtype in [DType::F32, DType::F16, DType::BF16] {
            let (nh, nkv, sq, sk, d) = (6usize, 2usize, 8usize, 24usize, 256usize);
            let scale = 1.0f32 / (d as f32).sqrt();
            // Include nonzero start_offset() via narrow on batch.
            let q = Tensor::randn(0f32, 0.2, (2, nh, sq, d), &dev)?
                .to_dtype(dtype)?
                .narrow(0, 1, 1)?;
            let k = Tensor::randn(0f32, 0.2, (2, nkv, sk, d), &dev)?
                .to_dtype(dtype)?
                .narrow(0, 1, 1)?;
            let v = Tensor::randn(0f32, 0.2, (2, nkv, sk, d), &dev)?
                .to_dtype(dtype)?
                .narrow(0, 1, 1)?;
            assert!(q.layout().start_offset() > 0);
            let got = flash_attn(&q, &k, &v, scale, 2)?
                .expect("metal dsl flash_attn d=256 causal=2")
                .to_dtype(DType::F32)?;

            // Eager bottom-right causal reference on Metal/CPU in f32.
            let groups = nh / nkv;
            let q32 = q.to_dtype(DType::F32)?;
            let k32 = k
                .to_dtype(DType::F32)?
                .unsqueeze(2)?
                .repeat((1, 1, groups, 1, 1))?
                .reshape((1, nh, sk, d))?;
            let v32 = v
                .to_dtype(DType::F32)?
                .unsqueeze(2)?
                .repeat((1, 1, groups, 1, 1))?
                .reshape((1, nh, sk, d))?;
            let mut mask = vec![0.0f32; sq * sk];
            let q_past = sk - sq;
            for i in 0..sq {
                for j in (q_past + i + 1)..sk {
                    mask[i * sk + j] = f32::NEG_INFINITY;
                }
            }
            let mask_t = Tensor::from_vec(mask, (1, 1, sq, sk), &dev)?;
            let scores = (q32.matmul(&k32.transpose(2, 3)?)? * (scale as f64))?;
            let scores = scores.broadcast_add(&mask_t)?;
            let max_s = scores.max_keepdim(3)?;
            let exps = scores.broadcast_sub(&max_s)?.exp()?;
            let probs = exps.broadcast_div(&exps.sum_keepdim(3)?)?;
            let want = probs.matmul(&v32)?;
            let rel: f32 = got.sub(&want)?.abs()?.max_all()?.to_scalar::<f32>()?
                / (1.0 + want.abs()?.max_all()?.to_scalar::<f32>()?);
            let tol = if dtype == DType::F32 { 3e-3 } else { 2e-2 };
            assert!(rel <= tol, "flash d256 causal=2 {dtype:?} rel={rel}");

            // PagedAttention + reshape_and_cache round-trip on d=256, block_size=32.
            let block_size = 32usize;
            let x = if dtype == DType::F32 { 4 } else { 8 };
            let num_blocks = 2usize;
            let mut kc = Tensor::zeros((num_blocks, nkv, d / x, block_size, x), dtype, &dev)?;
            let mut vc = Tensor::zeros((num_blocks, nkv, d, block_size), dtype, &dev)?;
            let kf = k.squeeze(0)?.transpose(0, 1)?.contiguous()?; // [sk, nkv, d]
            let vf = v.squeeze(0)?.transpose(0, 1)?.contiguous()?;
            let slots: Vec<i64> = (0..sk).map(|p| (block_size + p) as i64).collect(); // physical blk 1
            let slot_t = Tensor::from_vec(slots, (sk,), &dev)?;
            reshape_and_cache(&kf, &vf, &mut kc, &mut vc, &slot_t)?.expect("reshape_and_cache");

            let q1 = q.narrow(2, sq - 1, 1)?.squeeze(2)?; // [1, nh, d]
            let bt = Tensor::from_vec(vec![1u32], (1, 1), &dev)?;
            let cl = Tensor::from_vec(vec![sk as u32], (1,), &dev)?;
            let got_p = paged_attn(&q1, &kc, &vc, &bt, &cl, scale, 0.0, 0)?
                .expect("paged_attn")
                .to_dtype(DType::F32)?;
            let want_p = want.narrow(2, sq - 1, 1)?.squeeze(2)?;
            let p_rel: f32 = got_p.sub(&want_p)?.abs()?.max_all()?.to_scalar::<f32>()?
                / (1.0 + want_p.abs()?.max_all()?.to_scalar::<f32>()?);
            assert!(p_rel <= tol, "paged d256 {dtype:?} rel={p_rel}");
        }
        set_flash_impl(prev);
        Ok(())
    }
}
