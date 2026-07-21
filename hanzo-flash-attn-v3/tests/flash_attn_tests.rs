// Numerical-equivalence harness for FlashAttention-3 (Hopper `sm_90a`).
//
// FA3 attention is not bit-exact against a different kernel (fp16/bf16 matmul +
// online softmax reorder rounding), so equivalence is asserted with a
// scale-relative bound: max|FA3 - oracle| / max|oracle| < tol. Per-element
// relative error is avoided — it explodes on near-zero attention outputs.
//
// These tests require an `sm_90a` GPU (H100 / H200). The build host here is not
// Hopper, so every test is `#[ignore]`d. On real Hopper, run:
//
//     cargo test -p hanzo-flash-attn-v3 --features cuda -- --ignored
//
// The whole file is gated on `cuda`: without it the crate is a stub and a
// default `cargo test` compiles this target to nothing.
#![cfg(feature = "cuda")]

use anyhow::Result;
use hanzo_ml::{DType, Device, Tensor, D};
use rstest::rstest;

/// Softmax over the last dim (numerically stabilised), f32.
fn softmax_last(x: &Tensor) -> Result<Tensor> {
    let max = x.max_keepdim(D::Minus1)?;
    let ex = x.broadcast_sub(&max)?.exp()?;
    let sum = ex.sum_keepdim(D::Minus1)?;
    Ok(ex.broadcast_div(&sum)?)
}

/// Reference scaled-dot-product attention in f32 on the same device.
///
/// Inputs are the FA3 layout `(batch, seqlen, heads, head_dim)`; `k`/`v` may
/// carry fewer heads (GQA), block-repeated to `heads` the standard way (query
/// heads `[i*g, (i+1)*g)` share kv head `i`).
fn oracle(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, causal: bool) -> Result<Tensor> {
    let (b, s, h, d) = q.dims4()?;
    let (_b, sk, hkv, _d) = k.dims4()?;

    let q = q.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous()?; // (b,h,s,d)
    let mut k = k.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous()?; // (b,hkv,sk,d)
    let mut v = v.transpose(1, 2)?.to_dtype(DType::F32)?.contiguous()?;
    if hkv != h {
        let g = h / hkv;
        k = k
            .unsqueeze(2)?
            .broadcast_as((b, hkv, g, sk, d))?
            .reshape((b, h, sk, d))?;
        v = v
            .unsqueeze(2)?
            .broadcast_as((b, hkv, g, sk, d))?
            .reshape((b, h, sk, d))?;
    }

    let att = (q.matmul(&k.transpose(2, 3)?.contiguous()?)? * scale as f64)?; // (b,h,s,sk)
    let att = if causal {
        // Additive causal mask; query i attends keys <= i + (sk - s) offset.
        let off = sk - s;
        let mut mask = vec![0f32; s * sk];
        for i in 0..s {
            for j in 0..sk {
                if j > i + off {
                    mask[i * sk + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask, (s, sk), q.device())?;
        att.broadcast_add(&mask)?
    } else {
        att
    };
    let att = softmax_last(&att)?;
    let out = att.matmul(&v)?; // (b,h,s,d)
    Ok(out.transpose(1, 2)?.contiguous()?) // (b,s,h,d)
}

/// max|a - b| / max(|b|, eps) over all elements, both moved to CPU f32.
fn rel_err(a: &Tensor, b: &Tensor) -> Result<f32> {
    let a = a.to_dtype(DType::F32)?.flatten_all()?;
    let b = b.to_dtype(DType::F32)?.flatten_all()?;
    let num = a.sub(&b)?.abs()?.max(0)?.to_vec0::<f32>()?;
    let den = b.abs()?.max(0)?.to_vec0::<f32>()?.max(1e-6);
    Ok(num / den)
}

fn randish(b: usize, s: usize, h: usize, d: usize, dt: DType, dev: &Device) -> Result<Tensor> {
    // Deterministic, well-conditioned inputs in [-0.5, 0.5).
    let n = b * s * h * d;
    let data: Vec<f32> = (0..n).map(|i| ((i * 2654435761) % 1000) as f32 / 1000.0 - 0.5).collect();
    Ok(Tensor::from_vec(data, (b, s, h, d), dev)?.to_dtype(dt)?)
}

// Dense multi-head, non-causal and causal, across head dims and dtypes.
#[rstest]
#[ignore = "requires sm_90a (H100/H200); run on Hopper with --ignored"]
fn fa3_matches_oracle(
    #[values(64, 128, 256)] head_dim: usize,
    #[values(DType::F16, DType::BF16)] dtype: DType,
    #[values(false, true)] causal: bool,
) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let (b, s, h) = (2usize, 64usize, 8usize);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = randish(b, s, h, head_dim, dtype, &dev)?;
    let k = randish(b, s, h, head_dim, dtype, &dev)?;
    let v = randish(b, s, h, head_dim, dtype, &dev)?;

    let got = hanzo_flash_attn_v3::flash_attn(&q, &k, &v, scale, causal, false)?;
    let want = oracle(&q, &k, &v, scale, causal)?;
    let err = rel_err(&got, &want)?;
    let tol = if dtype == DType::BF16 { 3e-2 } else { 8e-3 };
    assert!(err < tol, "hdim={head_dim} dtype={dtype:?} causal={causal}: rel_err {err} >= {tol}");
    Ok(())
}

// Grouped-query attention with the packed FA3 path (h=8, kv=2).
#[rstest]
#[ignore = "requires sm_90a (H100/H200); run on Hopper with --ignored"]
fn fa3_gqa_matches_oracle(#[values(64, 128)] head_dim: usize) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let (b, s, h, hkv) = (1usize, 64usize, 8usize, 2usize);
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = randish(b, s, h, head_dim, DType::F16, &dev)?;
    let k = randish(b, s, hkv, head_dim, DType::F16, &dev)?;
    let v = randish(b, s, hkv, head_dim, DType::F16, &dev)?;

    let got = hanzo_flash_attn_v3::flash_attn(&q, &k, &v, scale, false, true)?;
    let want = oracle(&q, &k, &v, scale, false)?;
    let err = rel_err(&got, &want)?;
    assert!(err < 8e-3, "gqa hdim={head_dim}: rel_err {err} >= 8e-3");
    Ok(())
}

// Variable-length (single sequence packed) path.
#[rstest]
#[ignore = "requires sm_90a (H100/H200); run on Hopper with --ignored"]
fn fa3_varlen_matches_oracle(#[values(64, 128)] head_dim: usize) -> Result<()> {
    let dev = Device::new_cuda(0)?;
    let (s, h) = (48usize, 8usize);
    let scale = 1.0 / (head_dim as f32).sqrt();
    // varlen takes rank-3 (total_tokens, heads, head_dim).
    let q3 = randish(1, s, h, head_dim, DType::F16, &dev)?.reshape((s, h, head_dim))?;
    let k3 = randish(1, s, h, head_dim, DType::F16, &dev)?.reshape((s, h, head_dim))?;
    let v3 = randish(1, s, h, head_dim, DType::F16, &dev)?.reshape((s, h, head_dim))?;
    let seqlens = Tensor::new(&[0u32, s as u32], &dev)?;

    let got = hanzo_flash_attn_v3::flash_attn_varlen(
        &q3, &k3, &v3, &seqlens, &seqlens, s, s, scale, false, false,
    )?;
    // Oracle over the batched view (1, s, h, d).
    let q = q3.reshape((1, s, h, head_dim))?;
    let k = k3.reshape((1, s, h, head_dim))?;
    let v = v3.reshape((1, s, h, head_dim))?;
    let want = oracle(&q, &k, &v, scale, false)?.reshape((s, h, head_dim))?;
    let err = rel_err(&got, &want)?;
    assert!(err < 8e-3, "varlen hdim={head_dim}: rel_err {err} >= 8e-3");
    Ok(())
}
