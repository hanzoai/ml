//! Backend-independent DSL dispatch for Fast Walsh-Hadamard Transform (FWHT).
//!
//! Computes the normalized Sylvester-Walsh-Hadamard transform over contiguous `block_size`-wide
//! blocks along the last dimension with optional explicit `±1` signs:
//! - Forward (`!is_inverse`): `y = H_B (x * signs) / sqrt(B)`
//! - Inverse (`is_inverse`):  `y = (H_B x / sqrt(B)) * signs`
//!
//! Enabled by default (`HANZO_FWHT_IMPL=dsl`); set `HANZO_FWHT_IMPL=legacy` (or call
//! [`set_fwht_impl`]) to force `Ok(None)` so callers fall back to dense `H_B` matmul for A/B
//! verification.

use std::sync::atomic::{AtomicU8, Ordering};

pub use crate::quantized::dsl::QkImpl;
use crate::{DType, Result, Tensor};

#[cfg(any(feature = "metal", feature = "cuda"))]
use crate::Device;

static FWHT_IMPL_STATE: AtomicU8 = AtomicU8::new(0); // 0 = unset, 1 = legacy, 2 = dsl

/// Active FWHT implementation (`HANZO_FWHT_IMPL=dsl|legacy`, defaulting to `dsl`).
pub fn fwht_impl() -> QkImpl {
    match FWHT_IMPL_STATE.load(Ordering::Relaxed) {
        1 => QkImpl::Legacy,
        2 => QkImpl::Dsl,
        _ => {
            let imp = match std::env::var("HANZO_FWHT_IMPL").ok().as_deref() {
                Some("legacy") | Some("0") | Some("false") => QkImpl::Legacy,
                _ => QkImpl::Dsl,
            };
            FWHT_IMPL_STATE.store(if imp == QkImpl::Dsl { 2 } else { 1 }, Ordering::Relaxed);
            imp
        }
    }
}

/// Override the FWHT implementation in-process (useful for A/B parity tests).
pub fn set_fwht_impl(imp: QkImpl) {
    FWHT_IMPL_STATE.store(if imp == QkImpl::Dsl { 2 } else { 1 }, Ordering::Relaxed);
}

#[cfg(feature = "metal")]
fn ensure_contiguous(t: &Tensor) -> Result<Tensor> {
    if t.is_contiguous() {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

/// Run fused normalized FWHT along the last dimension of `x`, preserving `x`'s shape and dtype.
/// Returns `Ok(None)` if disabled (`HANZO_FWHT_IMPL=legacy`) or if the device/dtype/block_size
/// combination is unsupported.
pub fn hadamard_fwht(
    x: &Tensor,
    signs: Option<&Tensor>,
    block_size: usize,
    is_inverse: bool,
) -> Result<Option<Tensor>> {
    if fwht_impl() != QkImpl::Dsl {
        return Ok(None);
    }
    if !block_size.is_power_of_two() || block_size == 0 {
        return Ok(None);
    }
    let orig_shape = x.shape().clone();
    let Some(&width) = orig_shape.dims().last() else {
        return Ok(None);
    };
    let elem_count = orig_shape.elem_count();
    if width == 0 || width % block_size != 0 || elem_count == 0 {
        return Ok(None);
    }
    if let Some(s) = signs {
        if s.elem_count() != width {
            return Ok(None);
        }
    }
    let dtype = x.dtype();
    if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Ok(None);
    }

    #[cfg(feature = "metal")]
    if let Device::Metal(dev) = x.device() {
        if !matches!(block_size, 64 | 128 | 256 | 512 | 1024 | 2048 | 4096) {
            return Ok(None);
        }
        let x_c = ensure_contiguous(x)?;
        let s_c = match signs {
            Some(s) => {
                let sf = if s.dtype() == DType::F32 {
                    ensure_contiguous(s)?
                } else {
                    s.to_dtype(DType::F32)?.contiguous()?
                };
                Some(sf)
            }
            None => None,
        };
        let out = crate::metal_backend::metal_uninit_tensor(dev, orig_shape, dtype)?;
        let xb = crate::metal_backend::metal_tensor_buf(&x_c)?;
        let sb = match s_c.as_ref() {
            Some(s) => Some(crate::metal_backend::metal_tensor_buf(s)?),
            None => None,
        };
        let ob = crate::metal_backend::metal_tensor_buf(&out)?;
        let artifact = dev.hadamard_fwht_dsl(
            (&xb.0, xb.1),
            sb.as_ref().map(|b| (&b.0, b.1)),
            (&ob.0, ob.1),
            elem_count / block_size,
            width,
            block_size,
            is_inverse,
            dtype,
        )?;
        crate::quantized::dsl::note_dispatch(
            QkImpl::Dsl,
            artifact,
            elem_count / block_size,
            block_size,
        );
        return Ok(Some(out));
    }

    #[cfg(feature = "cuda")]
    if let Device::Cuda(_) = x.device() {
        if block_size <= 8192 {
            let out = crate::quantized::cuda::hadamard_fwht(x, signs, block_size, is_inverse)?;
            crate::quantized::dsl::note_dispatch(
                QkImpl::Dsl,
                "cuda_hadamard_fwht",
                elem_count / block_size,
                block_size,
            );
            return Ok(Some(out));
        }
    }

    let _ = is_inverse;
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Device;

    #[test]
    fn fwht_legacy_override_returns_none() -> Result<()> {
        let prev = fwht_impl();
        set_fwht_impl(QkImpl::Legacy);
        let x = Tensor::zeros((1, 64), DType::F32, &Device::Cpu)?;
        assert!(hadamard_fwht(&x, None, 64, false)?.is_none());
        set_fwht_impl(prev);
        Ok(())
    }

    #[cfg(feature = "metal")]
    #[test]
    fn hadamard_fwht_metal_matches_dense_all_sizes_and_dtypes() -> Result<()> {
        let prev = fwht_impl();
        set_fwht_impl(QkImpl::Dsl);
        let dev = Device::new_metal(0)?;
        for block_size in [64usize, 128, 256, 512, 1024, 2048, 4096] {
            let width = if block_size == 1024 {
                5120
            } else {
                block_size * 2
            };
            let rows = 3usize;
            let x_data: Vec<f32> = (0..rows * width)
                .map(|i| (((i * 7919) % 1009) as f32 / 504.5) - 1.0)
                .collect();
            let s_data: Vec<f32> = (0..width)
                .map(|i| if ((i * 2654435761) >> 31) == 0 { 1.0 } else { -1.0 })
                .collect();
            let signs = Tensor::from_vec(s_data.clone(), width, &dev)?;
            for dtype in [DType::F32, DType::F16, DType::BF16] {
                let x = Tensor::from_vec(x_data.clone(), (rows, width), &dev)?.to_dtype(dtype)?;
                let x_host = x.to_dtype(DType::F32)?.flatten_all()?.to_vec1::<f32>()?;
                let tol = match dtype {
                    DType::F32 => 2e-5f32,
                    DType::F16 => 5e-3f32,
                    DType::BF16 => 3e-2f32,
                    _ => unreachable!(),
                };
                for is_inv in [false, true] {
                    let got = hadamard_fwht(&x, Some(&signs), block_size, is_inv)?
                        .expect("metal dsl fwht")
                        .to_dtype(DType::F32)?
                        .flatten_all()?
                        .to_vec1::<f32>()?;
                    let want =
                        hanzo_kernel::fwht::fwht_ref(&x_host, Some(&s_data), width, block_size, is_inv);
                    let max_err = want
                        .iter()
                        .zip(&got)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    assert!(
                        max_err < tol,
                        "metal fwht B={block_size} dtype={dtype:?} inv={is_inv}: max_err={max_err} >= {tol}"
                    );
                }
            }
        }
        set_fwht_impl(prev);
        Ok(())
    }
}
