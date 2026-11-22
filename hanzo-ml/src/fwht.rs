//! Backend-independent Fast Walsh-Hadamard Transform (FWHT) powered by canonical
//! `hanzo-kernel` DSL artifacts (Metal, CUDA, Vulkan).
//!
//! Replaces the dense `O(B^2)` `matmul(&H_B)` activation rotation in `DeviceHadamard` with an
//! `O(B log B)` fused shared-memory butterfly (`fwht_hadamard_b{64..2048}_{f32,f16,bf16}`):
//! - Forward (`inverse == false`): `y = ((x * signs).reshape(-1, B) @ H_B).reshape_as(x)`
//! - Inverse (`inverse == true`): `y = (x.reshape(-1, B) @ H_B).reshape_as(x) * signs`
//!
//! Controlled by [`crate::quantized::dsl::fwht_impl`] (`ML_FWHT_IMPL=dsl|legacy`, default
//! `dsl` so Metal and Vulkan get the ~6x FWHT speedup out of the box, with `ML_FWHT_IMPL=legacy`
//! as an instant fallback switch for A/B).

#![allow(unused_variables, unused_mut, unreachable_code, dead_code)]

#[allow(unused_imports)]
use crate::op::BackpropOp;
use crate::quantized::dsl::{self, QkImpl};
#[allow(unused_imports)]
use crate::{DType, Result, Storage, Tensor};

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

/// Apply normalized Sylvester-Walsh-Hadamard transform with optional fused `±1` signs along the
/// last dimension (`block_size` in `{64, 128, 256, 512, 1024, 2048}`).
///
/// - `x`: arbitrary rank `>= 1` tensor whose last dim is a multiple of `block_size` (`F32`, `F16`,
///   or `BF16` on Metal/CUDA; `F32`/`F16`/`BF16` via F32 compute on Vulkan)
/// - `signs`: optional 1D `F32` tensor of length `width` (the last dim of `x`, or any multiple of
///   `block_size` dividing `x.elem_count()`)
/// - `block_size`: Hadamard block size `B`
/// - `inverse`: `false` for pre-sign forward fold `H(x * s)`, `true` for post-sign inverse fold `H(x) * s`
///
/// Returns `Ok(Some(out))` with the same shape and dtype as `x`, or `Ok(None)` when disabled via
/// `ML_FWHT_IMPL=legacy` or on an unsupported backend/block_size.
pub fn try_hadamard_fwht(
    x: &Tensor,
    signs: Option<&Tensor>,
    block_size: usize,
    inverse: bool,
) -> Result<Option<Tensor>> {
    if dsl::select_impl(
        dsl::QkBackend::from_device(x.device()),
        dsl::QkOp::Fwht { block_size },
    ) != QkImpl::Dsl
    {
        return Ok(None);
    }
    if !matches!(block_size, 64 | 128 | 256 | 512 | 1024 | 2048) {
        return Ok(None);
    }
    let orig_dtype = x.dtype();
    if !matches!(orig_dtype, DType::F32 | DType::F16 | DType::BF16) {
        return Ok(None);
    }
    let orig_shape = x.shape().clone();
    let Some(&last_dim) = orig_shape.dims().last() else {
        return Ok(None);
    };
    if last_dim == 0 || !last_dim.is_multiple_of(block_size) {
        return Ok(None);
    }
    let total = orig_shape.elem_count();
    if total == 0 {
        return Ok(Some(x.clone()));
    }
    let n_blocks = total / block_size;

    let signs_f32 = match signs {
        Some(s) => {
            let s_len = s.elem_count();
            if s_len == 0 || !s_len.is_multiple_of(block_size) {
                return Ok(None);
            }
            let sf = if s.dtype() == DType::F32 {
                s.clone()
            } else {
                s.to_dtype(DType::F32)?
            };
            Some(sf)
        }
        None => None,
    };

    match x.device() {
        #[cfg(feature = "metal")]
        crate::Device::Metal(dev) => {
            let x_c = ensure_contiguous_view(x)?;
            let s_c = match &signs_f32 {
                Some(s) => Some(ensure_contiguous_view(s)?),
                None => None,
            };
            let elem_bytes = orig_dtype.size_in_bytes();
            let out_buf = dev.new_buffer(total, orig_dtype, "fwht_hadamard_dsl_out")?;
            {
                let (xs, xl) = x_c.storage_and_layout();
                let Storage::Metal(xs) = &*xs else {
                    return Ok(None);
                };
                if let Some(sc) = &s_c {
                    let (ss, sl) = sc.storage_and_layout();
                    let Storage::Metal(ss) = &*ss else {
                        return Ok(None);
                    };
                    dev.fwht_hadamard_dsl(
                        orig_dtype,
                        (xs.buffer(), xl.start_offset() * elem_bytes),
                        Some((ss.buffer(), sl.start_offset() * 4, sc.elem_count())),
                        (&out_buf, 0),
                        n_blocks,
                        block_size,
                        inverse,
                    )?;
                } else {
                    dev.fwht_hadamard_dsl(
                        orig_dtype,
                        (xs.buffer(), xl.start_offset() * elem_bytes),
                        None,
                        (&out_buf, 0),
                        n_blocks,
                        block_size,
                        inverse,
                    )?;
                }
            }
            let out_s =
                crate::metal_backend::MetalStorage::new(out_buf, dev.clone(), total, orig_dtype);
            Ok(Some(Tensor::from_storage(
                Storage::Metal(out_s),
                orig_shape,
                BackpropOp::none(),
                false,
            )))
        }
        #[cfg(feature = "cuda")]
        crate::Device::Cuda(dev) => {
            let x_c = ensure_contiguous_view(x)?;
            let s_c = match &signs_f32 {
                Some(s) => Some(ensure_contiguous_view(s)?),
                None => None,
            };
            let out_s = {
                let (xs, xl) = x_c.storage_and_layout();
                let Storage::Cuda(xs) = &*xs else {
                    return Ok(None);
                };
                if let Some(sc) = &s_c {
                    let (ss, sl) = sc.storage_and_layout();
                    let Storage::Cuda(ss) = &*ss else {
                        return Ok(None);
                    };
                    dev.fwht_hadamard_dsl(
                        orig_dtype,
                        (xs, xl.start_offset()),
                        Some((ss, sl.start_offset(), sc.elem_count())),
                        n_blocks,
                        block_size,
                        inverse,
                    )?
                } else {
                    dev.fwht_hadamard_dsl(
                        orig_dtype,
                        (xs, xl.start_offset()),
                        None,
                        n_blocks,
                        block_size,
                        inverse,
                    )?
                }
            };
            Ok(Some(Tensor::from_storage(
                Storage::Cuda(out_s),
                orig_shape,
                BackpropOp::none(),
                false,
            )))
        }
        #[cfg(feature = "rocm")]
        crate::Device::Rocm(dev) => {
            let x_c = ensure_contiguous_view(x)?;
            let s_c = match &signs_f32 {
                Some(s) => Some(ensure_contiguous_view(s)?),
                None => None,
            };
            let out_s = {
                let (xs, xl) = x_c.storage_and_layout();
                let Storage::Rocm(xs) = &*xs else {
                    return Ok(None);
                };
                if let Some(sc) = &s_c {
                    let (ss, sl) = sc.storage_and_layout();
                    let Storage::Rocm(ss) = &*ss else {
                        return Ok(None);
                    };
                    dev.fwht_hadamard_dsl(
                        orig_dtype,
                        (xs, xl.start_offset()),
                        Some((ss, sl.start_offset(), sc.elem_count())),
                        n_blocks,
                        block_size,
                        inverse,
                    )?
                } else {
                    dev.fwht_hadamard_dsl(
                        orig_dtype,
                        (xs, xl.start_offset()),
                        None,
                        n_blocks,
                        block_size,
                        inverse,
                    )?
                }
            };
            Ok(Some(Tensor::from_storage(
                Storage::Rocm(out_s),
                orig_shape,
                BackpropOp::none(),
                false,
            )))
        }
        #[cfg(feature = "vulkan")]
        crate::Device::Vulkan(dev) => {
            let xf = if orig_dtype == DType::F32 {
                ensure_contiguous_offset_zero(x)?
            } else {
                ensure_contiguous_offset_zero(&x.to_dtype(DType::F32)?)?
            };
            let s_c = match &signs_f32 {
                Some(s) => Some(ensure_contiguous_offset_zero(s)?),
                None => None,
            };
            let out_opt = {
                let (xs, _) = xf.storage_and_layout();
                let Storage::Vulkan(xs) = &*xs else {
                    return Ok(None);
                };
                if let Some(sc) = &s_c {
                    let (ss, _) = sc.storage_and_layout();
                    let Storage::Vulkan(ss) = &*ss else {
                        return Ok(None);
                    };
                    dev.fwht_hadamard_dsl(
                        xs,
                        Some((ss, sc.elem_count())),
                        n_blocks,
                        block_size,
                        inverse,
                    )?
                } else {
                    dev.fwht_hadamard_dsl(xs, None, n_blocks, block_size, inverse)?
                }
            };
            let Some(out_s) = out_opt else {
                return Ok(None);
            };
            let out = Tensor::from_storage(
                Storage::Vulkan(out_s),
                orig_shape,
                BackpropOp::none(),
                false,
            );
            if orig_dtype == DType::F32 {
                Ok(Some(out))
            } else {
                Ok(Some(out.to_dtype(orig_dtype)?))
            }
        }
        _ => Ok(None),
    }
}
