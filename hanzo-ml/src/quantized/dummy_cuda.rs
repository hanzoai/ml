#![allow(unused)]
use super::GgmlDType;
use crate::{CudaDevice, CudaStorage, Error, Result};

pub struct QCudaStorage {
    dtype: GgmlDType,
    device: CudaDevice,
}

impl QCudaStorage {
    pub fn zeros(_: &CudaDevice, _: usize, _: GgmlDType) -> Result<Self> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn dequantize_f16(&self, _elem_count: usize) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize(&mut self, _src: &CudaStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize_imatrix(
        &mut self,
        _src: &CudaStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        _src: &crate::CpuStorage,
        _imatrix_weights: &[f32],
        _n_per_row: usize,
    ) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn quantize_onto(&mut self, _src: &crate::CpuStorage) -> Result<()> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        0
    }

    pub fn embedding(
        &self,
        _rows: usize,
        _hidden: usize,
        _ids: &CudaStorage,
        _ids_l: &crate::Layout,
    ) -> Result<CudaStorage> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn fwd(
        &self,
        _self_shape: &crate::Shape,
        _storage: &CudaStorage,
        _layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        Err(Error::NotCompiledWithCudaSupport)
    }

    pub fn supports_indexed_moe(_dtype: GgmlDType) -> bool {
        false
    }

    pub fn indexed_moe_forward(
        &self,
        _: &crate::Shape,
        _: &CudaStorage,
        _: &crate::Layout,
        _: &CudaStorage,
        _: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        Err(Error::NotCompiledWithCudaSupport)
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    _device: &CudaDevice,
    _data: &[T],
) -> Result<super::QStorage> {
    Err(Error::NotCompiledWithCudaSupport)
}

#[cfg(not(feature = "cuda"))]
mod cpu_fwht_fallback {
    use crate::{DType, Device, Module, Result, Shape, Tensor};

    #[derive(Clone, Debug)]
    pub struct IqQ8Activation {
        xs: Tensor,
        dtype: DType,
        b_size: usize,
        ncols: usize,
    }

    impl IqQ8Activation {
        pub fn quantize(xs: &Tensor) -> Result<Self> {
            let shape = xs.shape();
            if shape.rank() < 2 {
                crate::bail!("IqQ8Activation expects rank >= 2, got {:?}", shape);
            }
            let ncols = *shape.dims().last().unwrap();
            if !ncols.is_multiple_of(32) {
                crate::bail!("IqQ8Activation ncols {ncols} must be a multiple of 32");
            }
            let b_size = shape.elem_count() / ncols;
            if b_size == 0 {
                crate::bail!("IqQ8Activation requires non-empty batch");
            }
            let dtype = xs.dtype();
            if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
                crate::bail!("IqQ8Activation unsupported dtype {dtype:?}");
            }
            Ok(Self {
                xs: xs.clone(),
                dtype,
                b_size,
                ncols,
            })
        }

        pub fn shape(&self) -> &Shape {
            self.xs.shape()
        }

        pub fn dtype(&self) -> DType {
            self.dtype
        }

        pub fn b_size(&self) -> usize {
            self.b_size
        }

        pub fn ncols(&self) -> usize {
            self.ncols
        }

        pub fn with_dtype(&self, dtype: DType) -> Result<Self> {
            if !matches!(dtype, DType::F32 | DType::BF16 | DType::F16) {
                crate::bail!("IqQ8Activation unsupported output dtype {dtype:?}");
            }
            Ok(Self {
                xs: self.xs.to_dtype(dtype)?,
                dtype,
                b_size: self.b_size,
                ncols: self.ncols,
            })
        }
    }

    fn fwht_cpu_inplace(
        buf: &mut [f32],
        signs: Option<&[f32]>,
        width: usize,
        block_size: usize,
        is_inverse: bool,
    ) {
        let scale = 1.0f32 / (block_size as f32).sqrt();
        if !is_inverse {
            if let Some(sv) = signs {
                for row in buf.chunks_exact_mut(width) {
                    for (v, &s) in row.iter_mut().zip(sv.iter()) {
                        *v *= s;
                    }
                }
            }
        }
        for blk in buf.chunks_exact_mut(block_size) {
            let mut h = 1usize;
            while h < block_size {
                let mut base = 0usize;
                while base < block_size {
                    for j in 0..h {
                        let a = blk[base + j];
                        let b = blk[base + j + h];
                        blk[base + j] = a + b;
                        blk[base + j + h] = a - b;
                    }
                    base += 2 * h;
                }
                h <<= 1;
            }
            for v in blk.iter_mut() {
                *v *= scale;
            }
        }
        if is_inverse {
            if let Some(sv) = signs {
                for row in buf.chunks_exact_mut(width) {
                    for (v, &s) in row.iter_mut().zip(sv.iter()) {
                        *v *= s;
                    }
                }
            }
        }
    }

    pub fn hadamard_fwht(
        x: &Tensor,
        signs: Option<&Tensor>,
        block_size: usize,
        is_inverse: bool,
    ) -> Result<Tensor> {
        if !block_size.is_power_of_two() || block_size == 0 || block_size > 8192 {
            crate::bail!(
                "hadamard_fwht block_size must be a power of 2 in 1..=8192, got {block_size}"
            );
        }
        let Some(&width) = x.dims().last() else {
            crate::bail!("hadamard_fwht expects rank >= 1");
        };
        if !width.is_multiple_of(block_size) {
            crate::bail!("hadamard_fwht width {width} is not divisible by block_size {block_size}");
        }
        if let Some(s) = signs {
            if s.dtype() != DType::F32 || s.elem_count() != width {
                crate::bail!("hadamard_fwht signs must be F32 of len {width}");
            }
        }
        let orig_dtype = x.dtype();
        let orig_dev = x.device().clone();
        let mut vals = x
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let signs_vec = match signs {
            Some(s) => Some(s.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?),
            None => None,
        };
        fwht_cpu_inplace(
            &mut vals,
            signs_vec.as_deref(),
            width,
            block_size,
            is_inverse,
        );
        Tensor::from_vec(vals, x.shape(), &Device::Cpu)?
            .to_dtype(orig_dtype)?
            .to_device(&orig_dev)
    }

    pub fn rms_norm_hadamard_fwht(
        x: &Tensor,
        residual: Option<&Tensor>,
        weight: &Tensor,
        eps: f32,
        signs: Option<&Tensor>,
        block_size: usize,
    ) -> Result<(Option<Tensor>, Tensor)> {
        if !block_size.is_power_of_two() || block_size == 0 || block_size > 8192 {
            crate::bail!(
                "rms_norm_hadamard_fwht block_size must be a power of 2 in 1..=8192, got {block_size}"
            );
        }
        let Some(&width) = x.dims().last() else {
            crate::bail!("rms_norm_hadamard_fwht expects rank >= 1");
        };
        if !width.is_multiple_of(block_size) || weight.dims1()? != width {
            crate::bail!("rms_norm_hadamard_fwht width/weight mismatch");
        }
        if let Some(s) = signs {
            if s.dtype() != DType::F32 || s.elem_count() != width {
                crate::bail!("rms_norm_hadamard_fwht signs must be F32 of len {width}");
            }
        }
        let orig_dtype = x.dtype();
        let orig_dev = x.device().clone();
        let mut vals = x
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let res_out = if let Some(res) = residual {
            let r_vals = res
                .to_device(&Device::Cpu)?
                .to_dtype(DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?;
            match orig_dtype {
                DType::BF16 => {
                    let mut h_bf16 = Vec::with_capacity(vals.len());
                    for (v, &rv) in vals.iter_mut().zip(r_vals.iter()) {
                        let q = half::bf16::from_f32(*v + rv);
                        *v = q.to_f32();
                        h_bf16.push(q);
                    }
                    Some(Tensor::from_vec(h_bf16, x.shape(), &Device::Cpu)?.to_device(&orig_dev)?)
                }
                DType::F16 => {
                    let mut h_f16 = Vec::with_capacity(vals.len());
                    for (v, &rv) in vals.iter_mut().zip(r_vals.iter()) {
                        let q = half::f16::from_f32(*v + rv);
                        *v = q.to_f32();
                        h_f16.push(q);
                    }
                    Some(Tensor::from_vec(h_f16, x.shape(), &Device::Cpu)?.to_device(&orig_dev)?)
                }
                _ => {
                    for (v, &rv) in vals.iter_mut().zip(r_vals.iter()) {
                        *v += rv;
                    }
                    Some(
                        Tensor::from_vec(vals.clone(), x.shape(), &Device::Cpu)?
                            .to_dtype(orig_dtype)?
                            .to_device(&orig_dev)?,
                    )
                }
            }
        } else {
            None
        };
        let w_vec = weight
            .to_device(&Device::Cpu)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let signs_vec = match signs {
            Some(s) => Some(s.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?),
            None => None,
        };
        for row in vals.chunks_exact_mut(width) {
            let mean_sq =
                (row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (width as f64)) as f32;
            let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
            for (col, v) in row.iter_mut().enumerate() {
                *v = *v * inv_rms * w_vec[col];
            }
        }
        fwht_cpu_inplace(&mut vals, signs_vec.as_deref(), width, block_size, false);
        let y = Tensor::from_vec(vals, x.shape(), &Device::Cpu)?.to_device(&orig_dev)?;
        Ok((res_out, y))
    }

    impl Tensor {
        pub fn rms_norm_hadamard_fwht(
            &self,
            weight: &Tensor,
            eps: f32,
            signs: Option<&Tensor>,
            block_size: usize,
        ) -> Result<Tensor> {
            let (_, y) = rms_norm_hadamard_fwht(self, None, weight, eps, signs, block_size)?;
            Ok(y)
        }

        pub fn add_rms_norm_hadamard_fwht(
            &self,
            residual: &Tensor,
            weight: &Tensor,
            eps: f32,
            signs: Option<&Tensor>,
            block_size: usize,
        ) -> Result<(Tensor, Tensor)> {
            let (res_out, y) =
                rms_norm_hadamard_fwht(self, Some(residual), weight, eps, signs, block_size)?;
            Ok((res_out.unwrap(), y))
        }
    }

    #[allow(private_interfaces)]
    impl super::super::QMatMul {
        pub fn quantize_iq_q8(xs: &Tensor) -> Result<IqQ8Activation> {
            IqQ8Activation::quantize(xs)
        }

        pub fn hadamard_fwht_iq_q8(
            xs: &Tensor,
            signs: Option<&Tensor>,
            block_size: usize,
            is_inverse: bool,
        ) -> Result<IqQ8Activation> {
            let y = super::super::hadamard_fwht(xs, signs, block_size, is_inverse)?;
            IqQ8Activation::quantize(&y)
        }

        pub fn rms_norm_hadamard_fwht_iq_q8(
            xs: &Tensor,
            weight: &Tensor,
            eps: f32,
            signs: Option<&Tensor>,
            block_size: usize,
        ) -> Result<IqQ8Activation> {
            let (_, y) = rms_norm_hadamard_fwht(xs, None, weight, eps, signs, block_size)?;
            IqQ8Activation::quantize(&y)
        }

        pub fn add_rms_norm_hadamard_fwht_iq_q8(
            xs: &Tensor,
            residual: &Tensor,
            weight: &Tensor,
            eps: f32,
            signs: Option<&Tensor>,
            block_size: usize,
        ) -> Result<(Tensor, IqQ8Activation)> {
            let (res_out, y) =
                rms_norm_hadamard_fwht(xs, Some(residual), weight, eps, signs, block_size)?;
            Ok((res_out.unwrap(), IqQ8Activation::quantize(&y)?))
        }

        pub fn forward_iq_q8(&self, q8: &IqQ8Activation) -> Result<Tensor> {
            self.forward_iq_q8_dtype(q8, q8.dtype)
        }

        pub fn forward_iq_q8_dtype(&self, q8: &IqQ8Activation, out_dtype: DType) -> Result<Tensor> {
            let y = self.forward(&q8.xs)?;
            if y.dtype() == out_dtype {
                Ok(y)
            } else {
                y.to_dtype(out_dtype)
            }
        }
    }
}

#[cfg(not(feature = "cuda"))]
pub use cpu_fwht_fallback::{hadamard_fwht, rms_norm_hadamard_fwht, IqQ8Activation};
