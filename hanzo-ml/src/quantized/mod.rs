use crate::{
    backend::BackendStorage, CpuStorage, DType, Device, Result, Shape, Storage, Tensor, D,
};
use k_quants::*;
use std::borrow::Cow;

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
pub mod imatrix_file;
pub mod k_quants;
#[cfg(feature = "metal")]
pub mod metal;
#[cfg(not(target_arch = "wasm32"))]
pub mod tokenizer;
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(feature = "cuda")]
pub mod fast_mmq;
#[cfg(feature = "cuda")]
pub mod fast_mmvq;
#[cfg(not(feature = "cuda"))]
mod cuda {
    pub use super::dummy_cuda::*;
}

#[cfg(target_feature = "neon")]
pub mod neon;
#[cfg(target_feature = "simd128")]
pub mod simd128;
pub mod utils;
use half::{bf16, f16};

pub use k_quants::GgmlType;

// Borrows `data` (does not consume it) so the returned slice stays valid for the caller's lifetime.
// Taking `Cow` by value here was a use-after-free: the Cow dropped at return, dangling the slice for
// Cow::Owned inputs (segfault on large quantized tensors; mmap'd Cow::Borrowed happened to survive).
fn as_t_slice<T>(data: &[u8]) -> &[T] {
    let size = std::mem::size_of::<T>();
    assert_eq!(
        data.len() % size,
        0,
        "Data length must be a multiple of T's size"
    );
    let ptr = data.as_ptr();
    assert_eq!(
        (ptr as usize) % std::mem::align_of::<T>(),
        0,
        "Data pointer must be aligned to T's alignment"
    );
    unsafe { std::slice::from_raw_parts(ptr as *const T, data.len() / size) }
}

pub struct QTensor {
    storage: QStorage,
    shape: Shape,
}

impl Device {
    fn qzeros(&self, elem_count: usize, dtype: GgmlDType) -> Result<QStorage> {
        match self {
            Device::Cpu => {
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Cpu(storage))
            }
            Device::Metal(metal) => {
                let storage = metal::QMetalStorage::zeros(metal, elem_count, dtype)?;
                Ok(QStorage::Metal(storage))
            }
            Device::Cuda(cuda) => {
                let storage = cuda::QCudaStorage::zeros(cuda, elem_count, dtype)?;
                Ok(QStorage::Cuda(storage))
            }
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => crate::bail!("quantized tensors on rocm are not supported yet"),
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => {
                // Keep quantized blocks in a CPU box + the device (same as from_data); QMatMul's
                // VulkanQuant path uploads/dequantizes them to the GPU on use.
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Vulkan(storage, d.clone()))
            }
            #[cfg(feature = "wgpu")]
            Device::Wgpu(d) => {
                // Same as Vulkan: keep the quantized blocks in a CPU box + the device; QMatMul's
                // WgpuQuant path uploads them to the GPU on use.
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Wgpu(storage, d.clone()))
            }
        }
    }
}

pub enum QStorage {
    Cpu(Box<dyn QuantizedType>),
    Metal(metal::QMetalStorage),
    Cuda(cuda::QCudaStorage),
    // Vulkan keeps the quantized blocks in a CPU box and dequantizes to an f32 Vulkan tensor on
    // demand (QMatMul forces dequantize for Vulkan). Lets GGUF-quantized models run on the GPU;
    // a direct quantized matmul (the Q8 kernel) is the bandwidth-win follow-up.
    #[cfg(feature = "vulkan")]
    Vulkan(Box<dyn QuantizedType>, crate::VulkanDevice),
    // wgpu mirror of the Vulkan path: quantized blocks held in a CPU box + the device. QMatMul's
    // WgpuQuant path reads the GGML bytes straight in the native quant matvec kernel (decode), or
    // dequantizes to an f32 wgpu tensor on demand.
    #[cfg(feature = "wgpu")]
    Wgpu(Box<dyn QuantizedType>, crate::WgpuDevice),
}

impl QStorage {
    pub fn from_data(data: Cow<'_, [u8]>, device: &Device, dtype: GgmlDType) -> Result<Self> {
        match device {
            Device::Cpu => Ok(Self::Cpu(dtype.from_data(data))),
            Device::Metal(d) => match dtype {
                GgmlDType::F32 => metal::load_quantized(d, as_t_slice::<f32>(&data)),
                GgmlDType::F16 => metal::load_quantized(d, as_t_slice::<f16>(&data)),
                GgmlDType::Q4_0 => metal::load_quantized(d, as_t_slice::<BlockQ4_0>(&data)),
                GgmlDType::Q4_1 => metal::load_quantized(d, as_t_slice::<BlockQ4_1>(&data)),
                GgmlDType::Q5_0 => metal::load_quantized(d, as_t_slice::<BlockQ5_0>(&data)),
                GgmlDType::Q5_1 => metal::load_quantized(d, as_t_slice::<BlockQ5_1>(&data)),
                GgmlDType::Q8_0 => metal::load_quantized(d, as_t_slice::<BlockQ8_0>(&data)),
                GgmlDType::Q8_1 => metal::load_quantized(d, as_t_slice::<BlockQ8_1>(&data)),
                GgmlDType::Q2K => metal::load_quantized(d, as_t_slice::<BlockQ2K>(&data)),
                GgmlDType::Q3K => metal::load_quantized(d, as_t_slice::<BlockQ3K>(&data)),
                GgmlDType::Q4K => metal::load_quantized(d, as_t_slice::<BlockQ4K>(&data)),
                GgmlDType::Q5K => metal::load_quantized(d, as_t_slice::<BlockQ5K>(&data)),
                GgmlDType::Q6K => metal::load_quantized(d, as_t_slice::<BlockQ6K>(&data)),
                GgmlDType::Q8K => metal::load_quantized(d, as_t_slice::<BlockQ8K>(&data)),
                GgmlDType::IQ4_NL => metal::load_quantized(d, as_t_slice::<BlockIQ4nl>(&data)),
                GgmlDType::IQ4_XS => metal::load_quantized(d, as_t_slice::<BlockIQ4xs>(&data)),
                GgmlDType::BF16 => metal::load_quantized(d, as_t_slice::<bf16>(&data)),
            },
            Device::Cuda(d) => match dtype {
                GgmlDType::F32 => cuda::load_quantized(d, as_t_slice::<f32>(&data)),
                GgmlDType::F16 => cuda::load_quantized(d, as_t_slice::<f16>(&data)),
                GgmlDType::Q4_0 => cuda::load_quantized(d, as_t_slice::<BlockQ4_0>(&data)),
                GgmlDType::Q4_1 => cuda::load_quantized(d, as_t_slice::<BlockQ4_1>(&data)),
                GgmlDType::Q5_0 => cuda::load_quantized(d, as_t_slice::<BlockQ5_0>(&data)),
                GgmlDType::Q5_1 => cuda::load_quantized(d, as_t_slice::<BlockQ5_1>(&data)),
                GgmlDType::Q8_0 => cuda::load_quantized(d, as_t_slice::<BlockQ8_0>(&data)),
                GgmlDType::Q8_1 => cuda::load_quantized(d, as_t_slice::<BlockQ8_1>(&data)),
                GgmlDType::Q2K => cuda::load_quantized(d, as_t_slice::<BlockQ2K>(&data)),
                GgmlDType::Q3K => cuda::load_quantized(d, as_t_slice::<BlockQ3K>(&data)),
                GgmlDType::Q4K => cuda::load_quantized(d, as_t_slice::<BlockQ4K>(&data)),
                GgmlDType::Q5K => cuda::load_quantized(d, as_t_slice::<BlockQ5K>(&data)),
                GgmlDType::Q6K => cuda::load_quantized(d, as_t_slice::<BlockQ6K>(&data)),
                GgmlDType::Q8K => cuda::load_quantized(d, as_t_slice::<BlockQ8K>(&data)),
                GgmlDType::IQ4_NL => cuda::load_quantized(d, as_t_slice::<BlockIQ4nl>(&data)),
                GgmlDType::IQ4_XS => cuda::load_quantized(d, as_t_slice::<BlockIQ4xs>(&data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(&data)),
            },
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => crate::bail!("quantized tensors on rocm are not supported yet"),
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => Ok(Self::Vulkan(dtype.from_data(data), d.clone())),
            #[cfg(feature = "wgpu")]
            Device::Wgpu(d) => Ok(Self::Wgpu(dtype.from_data(data), d.clone())),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(storage, _) => storage.block_size(),
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(storage, _) => storage.block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(storage, _) => storage.dtype(),
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(storage, _) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(_storage, device) => Device::Vulkan(device.clone()),
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(_storage, device) => Device::Wgpu(device.clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(storage, _) => storage.storage_size_in_bytes(),
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(storage, _) => storage.storage_size_in_bytes(),
        }
    }

    fn quantize(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => storage.quantize(src)?,
            (QStorage::Cuda(storage), Storage::Cuda(src)) => storage.quantize(src)?,
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_imatrix(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Metal(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cuda(src)) => {
                storage.quantize_imatrix(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn quantize_onto(&mut self, src: &Storage) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float(src.as_slice::<f32>()?);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            (QStorage::Cuda(storage), Storage::Cpu(src)) => storage.quantize_onto(src)?,
            _ => crate::bail!("Invalid quantize source storage locations: not on cpu"),
        }
        Ok(())
    }

    fn quantize_imatrix_onto(
        &mut self,
        src: &Storage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        match (self, src) {
            (QStorage::Cpu(storage), Storage::Cpu(src)) => {
                storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
            }
            (QStorage::Metal(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            (QStorage::Cuda(storage), Storage::Cpu(src)) => {
                storage.quantize_imatrix_onto(src, imatrix_weights, n_per_row)?
            }
            _ => crate::bail!("Invalid quantize storage locations do not match"),
        }
        Ok(())
    }

    fn dequantize(&self, elem_count: usize) -> Result<Storage> {
        match self {
            QStorage::Cpu(storage) => Ok(Storage::Cpu(storage.dequantize(elem_count)?)),
            QStorage::Metal(storage) => Ok(Storage::Metal(storage.dequantize(elem_count)?)),
            QStorage::Cuda(storage) => Ok(Storage::Cuda(storage.dequantize(elem_count)?)),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(storage, device) => {
                // Dequantize on the CPU, then upload the f32 weights to the GPU.
                let cpu = storage.dequantize(elem_count)?;
                Ok(Storage::Vulkan(device.upload_f32(cpu.as_slice::<f32>()?)?))
            }
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(storage, device) => {
                // Dequantize on the CPU, then upload the f32 weights to the GPU.
                let cpu = storage.dequantize(elem_count)?;
                Ok(Storage::Wgpu(device.upload_f32(cpu.as_slice::<f32>()?)?))
            }
        }
    }

    fn data(&self) -> Result<Cow<'_, [u8]>> {
        match self {
            QStorage::Cpu(storage) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            QStorage::Cuda(storage) => Ok(Cow::from(storage.data()?)),
            QStorage::Metal(storage) => Ok(Cow::from(storage.data()?)),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(storage, _) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(storage, _) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(..) => crate::bail!("not implemented"),
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(..) => crate::bail!("not implemented"),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr_with_guard(stream),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgmlDType {
    F32,
    F16,
    BF16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    #[allow(non_camel_case_types)]
    IQ4_NL,
    #[allow(non_camel_case_types)]
    IQ4_XS,
}

impl GgmlDType {
    pub(crate) fn from_u32(u: u32) -> Result<Self> {
        let dtype = match u {
            0 => Self::F32,
            1 => Self::F16,
            2 => Self::Q4_0,
            3 => Self::Q4_1,
            6 => Self::Q5_0,
            7 => Self::Q5_1,
            8 => Self::Q8_0,
            9 => Self::Q8_1,
            10 => Self::Q2K,
            11 => Self::Q3K,
            12 => Self::Q4K,
            13 => Self::Q5K,
            14 => Self::Q6K,
            15 => Self::Q8K,
            // IQ4_NL / IQ4_XS: ggml type ids 20 / 23 (llama.cpp ggml.h GGML_TYPE_IQ4_NL=20, _XS=23).
            20 => Self::IQ4_NL,
            23 => Self::IQ4_XS,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            _ => crate::bail!("unknown dtype for tensor {u}"),
        };
        Ok(dtype)
    }

    pub(crate) fn to_u32(self) -> u32 {
        match self {
            Self::F32 => 0,
            Self::F16 => 1,
            Self::Q4_0 => 2,
            Self::Q4_1 => 3,
            Self::Q5_0 => 6,
            Self::Q5_1 => 7,
            Self::Q8_0 => 8,
            Self::Q8_1 => 9,
            Self::Q2K => 10,
            Self::Q3K => 11,
            Self::Q4K => 12,
            Self::Q5K => 13,
            Self::Q6K => 14,
            Self::Q8K => 15,
            Self::IQ4_NL => 20,
            Self::IQ4_XS => 23,
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
        }
    }

    /// The block dtype
    pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
            Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
            Self::Q4_0 => Box::new(vec![BlockQ4_0::zeros(); elem_count / BlockQ4_0::BLCK_SIZE]),
            Self::Q4_1 => Box::new(vec![BlockQ4_1::zeros(); elem_count / BlockQ4_1::BLCK_SIZE]),
            Self::Q5_0 => Box::new(vec![BlockQ5_0::zeros(); elem_count / BlockQ5_0::BLCK_SIZE]),
            Self::Q5_1 => Box::new(vec![BlockQ5_1::zeros(); elem_count / BlockQ5_1::BLCK_SIZE]),
            Self::Q8_0 => Box::new(vec![BlockQ8_0::zeros(); elem_count / BlockQ8_0::BLCK_SIZE]),
            Self::Q8_1 => Box::new(vec![BlockQ8_1::zeros(); elem_count / BlockQ8_1::BLCK_SIZE]),
            Self::Q2K => Box::new(vec![BlockQ2K::zeros(); elem_count / BlockQ2K::BLCK_SIZE]),
            Self::Q3K => Box::new(vec![BlockQ3K::zeros(); elem_count / BlockQ3K::BLCK_SIZE]),
            Self::Q4K => Box::new(vec![BlockQ4K::zeros(); elem_count / BlockQ4K::BLCK_SIZE]),
            Self::Q5K => Box::new(vec![BlockQ5K::zeros(); elem_count / BlockQ5K::BLCK_SIZE]),
            Self::Q6K => Box::new(vec![BlockQ6K::zeros(); elem_count / BlockQ6K::BLCK_SIZE]),
            Self::Q8K => Box::new(vec![BlockQ8K::zeros(); elem_count / BlockQ8K::BLCK_SIZE]),
            Self::IQ4_NL => {
                Box::new(vec![BlockIQ4nl::zeros(); elem_count / BlockIQ4nl::BLCK_SIZE])
            }
            Self::IQ4_XS => {
                Box::new(vec![BlockIQ4xs::zeros(); elem_count / BlockIQ4xs::BLCK_SIZE])
            }
            Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
        }
    }

    pub fn from_data(&self, data: Cow<'_, [u8]>) -> Box<dyn QuantizedType> {
        match self {
            Self::F32 => Box::new(as_t_slice::<f32>(&data).to_vec()),
            Self::F16 => Box::new(as_t_slice::<f16>(&data).to_vec()),
            Self::Q4_0 => Box::new(as_t_slice::<BlockQ4_0>(&data).to_vec()),
            Self::Q4_1 => Box::new(as_t_slice::<BlockQ4_1>(&data).to_vec()),
            Self::Q5_0 => Box::new(as_t_slice::<BlockQ5_0>(&data).to_vec()),
            Self::Q5_1 => Box::new(as_t_slice::<BlockQ5_1>(&data).to_vec()),
            Self::Q8_0 => Box::new(as_t_slice::<BlockQ8_0>(&data).to_vec()),
            Self::Q8_1 => Box::new(as_t_slice::<BlockQ8_1>(&data).to_vec()),
            Self::Q2K => Box::new(as_t_slice::<BlockQ2K>(&data).to_vec()),
            Self::Q3K => Box::new(as_t_slice::<BlockQ3K>(&data).to_vec()),
            Self::Q4K => Box::new(as_t_slice::<BlockQ4K>(&data).to_vec()),
            Self::Q5K => Box::new(as_t_slice::<BlockQ5K>(&data).to_vec()),
            Self::Q6K => Box::new(as_t_slice::<BlockQ6K>(&data).to_vec()),
            Self::Q8K => Box::new(as_t_slice::<BlockQ8K>(&data).to_vec()),
            Self::IQ4_NL => Box::new(as_t_slice::<BlockIQ4nl>(&data).to_vec()),
            Self::IQ4_XS => Box::new(as_t_slice::<BlockIQ4xs>(&data).to_vec()),
            Self::BF16 => Box::new(as_t_slice::<bf16>(&data).to_vec()),
        }
    }

    /// The type size for blocks in bytes.
    pub fn type_size(&self) -> usize {
        use k_quants::*;
        match self {
            Self::F32 => 4,
            Self::F16 | Self::BF16 => 2,
            Self::Q4_0 => std::mem::size_of::<BlockQ4_0>(),
            Self::Q4_1 => std::mem::size_of::<BlockQ4_1>(),
            Self::Q5_0 => std::mem::size_of::<BlockQ5_0>(),
            Self::Q5_1 => std::mem::size_of::<BlockQ5_1>(),
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            Self::Q8_0 => std::mem::size_of::<BlockQ8_0>(),
            Self::Q8_1 => std::mem::size_of::<BlockQ8_1>(),
            Self::Q2K => std::mem::size_of::<BlockQ2K>(),
            Self::Q3K => std::mem::size_of::<BlockQ3K>(),
            Self::Q4K => std::mem::size_of::<BlockQ4K>(),
            Self::Q5K => std::mem::size_of::<BlockQ5K>(),
            Self::Q6K => std::mem::size_of::<BlockQ6K>(),
            Self::Q8K => std::mem::size_of::<BlockQ8K>(),
            Self::IQ4_NL => std::mem::size_of::<BlockIQ4nl>(),
            Self::IQ4_XS => std::mem::size_of::<BlockIQ4xs>(),
        }
    }

    /// The block size, i.e. the number of elements stored in each block.
    pub fn block_size(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 | Self::BF16 => 1,
            Self::Q4_0 => k_quants::QK4_0,
            Self::Q4_1 => k_quants::QK4_1,
            Self::Q5_0 => k_quants::QK5_0,
            Self::Q5_1 => k_quants::QK5_1,
            Self::Q8_0 => k_quants::QK8_0,
            Self::Q8_1 => k_quants::QK8_1,
            Self::IQ4_NL => k_quants::QK4_NL,
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K | Self::IQ4_XS => {
                k_quants::QK_K
            }
        }
    }
}

// A version of GgmlType without `vec_dot` so that it can be dyn boxed.
pub trait QuantizedType: Send + Sync {
    fn dtype(&self) -> GgmlDType;
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()>;
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()>;
    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage>;
    fn storage_size_in_bytes(&self) -> usize;
    fn as_ptr(&self) -> *const u8;
    fn block_size(&self) -> usize;
    #[allow(clippy::wrong_self_convention)]
    fn from_float(&mut self, xs: &[f32]);
    #[allow(clippy::wrong_self_convention)]
    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize);
    fn size(&self) -> usize;
}

impl<T: k_quants::GgmlType + Send + Sync> QuantizedType for Vec<T> {
    fn matmul_t(&self, mkn: (usize, usize, usize), lhs: &[f32], dst: &mut [f32]) -> Result<()> {
        k_quants::matmul(mkn, lhs, self.as_slice(), dst)
    }
    fn matmul_t_f16(&self, mkn: (usize, usize, usize), lhs: &[f16], dst: &mut [f16]) -> Result<()> {
        k_quants::matmul_f16(mkn, lhs, self.as_slice(), dst)
    }

    fn size(&self) -> usize {
        self.len() * core::mem::size_of::<T>()
    }

    fn from_float(&mut self, xs: &[f32]) {
        T::from_float(xs, self)
    }

    fn from_float_imatrix(&mut self, xs: &[f32], imatrix_weights: &[f32], n_per_row: usize) {
        T::from_float_imatrix(xs, self, imatrix_weights, n_per_row)
    }

    fn dtype(&self) -> GgmlDType {
        T::DTYPE
    }

    fn block_size(&self) -> usize {
        T::BLCK_SIZE
    }

    fn dequantize(&self, elem_count: usize) -> Result<CpuStorage> {
        let mut ys = vec![0.0f32; elem_count];
        T::to_float(self.as_slice(), &mut ys);
        Ok(CpuStorage::F32(ys))
    }

    fn storage_size_in_bytes(&self) -> usize {
        self.len() * std::mem::size_of::<T>()
    }

    fn as_ptr(&self) -> *const u8 {
        self.as_ptr() as *const u8
    }
}

impl std::fmt::Debug for QTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "QTensor[{:?}; {:?}]", self.shape, self.dtype())
    }
}

fn check_shape(shape: &Shape, block_size: usize) -> Result<()> {
    let dims = shape.dims();
    if dims.is_empty() {
        crate::bail!("scalar tensor cannot be quantized {shape:?}")
    }
    if !dims[dims.len() - 1].is_multiple_of(block_size) {
        crate::bail!(
            "quantized tensor must have their last dim divisible by block size {shape:?} {}",
            block_size
        )
    }
    Ok(())
}

impl QTensor {
    pub fn new<S: Into<Shape>>(storage: QStorage, shape: S) -> Result<Self> {
        let shape = shape.into();
        check_shape(&shape, storage.block_size())?;
        Ok(Self { storage, shape })
    }

    pub fn quantize(src: &Tensor, dtype: GgmlDType) -> Result<Self> {
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn quantize_imatrix(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
    ) -> Result<Self> {
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }

        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            );
        }
        let mut storage = src.device().qzeros(elem_count, dtype)?;
        storage.quantize_imatrix(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_imatrix_onto(
        src: &Tensor,
        imatrix_weights: &[f32],
        dtype: GgmlDType,
        dev: &Device,
    ) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        // (n_per_row/QK_K-1)*QK_K+(QK_K/32-1)*32+32=n_per_row
        // Size of imatrix == last dim of tensor
        let n_per_row = src.dim(D::Minus1)?;
        if imatrix_weights.len() != n_per_row {
            crate::bail!(
                "imatrix weights must have the same length {} as the last dim of src {}",
                imatrix_weights.len(),
                src.dim(D::Minus1)?
            );
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_imatrix_onto(&src.storage(), imatrix_weights, n_per_row)?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    /// Quantize `src` (currently on the CPU) to a QTensor on `dev`
    pub fn quantize_onto(src: &Tensor, dtype: GgmlDType, dev: &Device) -> Result<Self> {
        if !src.device().is_cpu() {
            crate::bail!(
                "`quantize_onto` expects a `src` to be on the cpu, got {:?}.",
                src.device()
            )
        }
        let shape = src.shape();
        let block_size = dtype.block_size();
        check_shape(shape, block_size)?;
        let src = src.to_dtype(crate::DType::F32)?.flatten_all()?;
        let elem_count = shape.elem_count();
        if !elem_count.is_multiple_of(block_size) {
            crate::bail!(
                "tensor size ({shape:?}) is not divisible by block size {}",
                block_size
            )
        }
        // storage is on the `dev`, src is on `cpu`
        let mut storage = dev.qzeros(elem_count, dtype)?;
        storage.quantize_onto(&src.storage())?;
        Ok(Self {
            storage,
            shape: shape.clone(),
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.storage.dtype()
    }

    pub fn device(&self) -> Device {
        self.storage.device()
    }

    pub fn rank(&self) -> usize {
        self.shape.rank()
    }

    pub fn shape(&self) -> &Shape {
        &self.shape
    }

    pub fn dequantize(&self, device: &Device) -> Result<Tensor> {
        let storage = self.storage.dequantize(self.shape.elem_count())?;
        let none = crate::op::BackpropOp::none();
        crate::tensor::from_storage(storage, self.shape.clone(), none, false).to_device(device)
    }

    pub fn dequantize_f16(&self, device: &Device) -> Result<Tensor> {
        // In the CUDA case, we have a specialized kernel as this can be useful for volta
        // architectures. https://github.com/hanzoai/ml/issues/2136
        match &self.storage {
            QStorage::Cuda(s) => {
                let s = s.dequantize_f16(self.shape.elem_count())?;
                let none = crate::op::BackpropOp::none();
                crate::tensor::from_storage(Storage::Cuda(s), self.shape.clone(), none, false)
                    .to_device(device)
            }
            _ => {
                let s = self.dequantize(device)?.to_dtype(crate::DType::F16)?;
                Ok(s)
            }
        }
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.storage.size_in_bytes()
    }

    pub fn data(&self) -> Result<Cow<'_, [u8]>> {
        self.storage.data()
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match &self.storage {
            // Only dtypes with a fused CUDA indexed-MoE kernel take the fast path; others (e.g. MXFP4)
            // fall through to the generic per-expert path below, which dequantizes via QMatMul.
            QStorage::Cuda(s)
                if matches!(
                    s.dtype(),
                    GgmlDType::Q8_0
                        | GgmlDType::Q2K
                        | GgmlDType::Q3K
                        | GgmlDType::Q4K
                        | GgmlDType::Q5K
                        | GgmlDType::Q6K
                ) =>
            {
                match (&*x.storage(), &*ids.storage()) {
                (Storage::Cuda(x_storage), Storage::Cuda(ids_storage)) => {
                    let (storage, out_shape) = s.indexed_moe_forward(
                        self.shape(),
                        x_storage,
                        x.layout(),
                        ids_storage,
                        ids.layout(),
                    )?;
                    Ok(crate::tensor::from_storage(
                        Storage::Cuda(storage),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                }
                _ => {
                    panic!("Non-cuda indexed_moe_forward is not implemented!");
                }
                }
            },
            // Native Vulkan MoE: one fused grouped quant matvec dispatch reads the per-expert slice
            // out of the GGML weight bank [E, n, k] resident in VRAM and gathers by the router ids --
            // the whole expert compute runs on the GPU (no CPU expert loop; the CPU fallback below
            // would also hit the unimplemented Vulkan index_add). Supported for Q4_0/Q8_0/Q4_K; other
            // quant dtypes fall through to the (CPU-bound) generic path.
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(_, vk_dev)
                if matches!(
                    self.storage.dtype(),
                    GgmlDType::Q4_0 | GgmlDType::Q8_0 | GgmlDType::Q4K
                ) =>
            {
                let out_dtype = x.dtype();
                let (e_cnt, n, k) = self.shape().dims3()?;
                let (t, topk) = ids.dims2()?;
                let s = x.dim(1)?; // 1 (gate/up: shared input) or topk (down: per-slot)
                let x_exp = if s == topk {
                    x.clone()
                } else {
                    x.broadcast_as((t, topk, k))?
                };
                // [S, k] contiguous f32 on the Vulkan device; S = t*topk routed slots.
                let nrows = t * topk;
                let x_flat = x_exp
                    .reshape((nrows, k))?
                    .to_dtype(crate::DType::F32)?
                    .contiguous()?;
                let ids_vec = ids
                    .reshape((nrows,))?
                    .to_dtype(crate::DType::U32)?
                    .to_vec1::<u32>()?;
                // Defend against a stray id (model bug / corrupt router) reading OOB in the bank.
                if let Some(&bad) = ids_vec.iter().find(|&&e| e as usize >= e_cnt) {
                    crate::bail!("indexed_moe_forward: expert id {bad} >= num_experts {e_cnt}");
                }

                let kernel = match self.storage.dtype() {
                    GgmlDType::Q4_0 => "moe_matvec_q4_0",
                    GgmlDType::Q8_0 => "moe_matvec_q8_0",
                    GgmlDType::Q4K => "moe_matvec_q4k",
                    other => crate::bail!("vulkan MoE: no kernel for {other:?}"),
                };
                let bank = self.data()?; // raw GGML bytes for all E experts, [E, n, k]
                let wbank = vk_dev.upload_qweight(&bank)?;
                let ids_buf = vk_dev.upload_ids(&ids_vec)?;
                let y = {
                    let (store, _) = x_flat.storage_and_layout();
                    let xv = match &*store {
                        Storage::Vulkan(v) => v,
                        _ => crate::bail!("vulkan MoE: x not on vulkan after contiguous()"),
                    };
                    vk_dev.moe_matvec_gpu(kernel, &wbank, xv, &ids_buf, nrows, n, k)?
                };
                let out = crate::tensor::from_storage(
                    Storage::Vulkan(y),
                    (nrows, n),
                    crate::op::BackpropOp::none(),
                    false,
                );
                out.reshape((t, topk, n))?.to_dtype(out_dtype)
            }
            // Native wgpu MoE: mirror of the Vulkan fused grouped quant matvec dispatch. The GGML
            // weight bank [E, n, k] is uploaded once and the router gather + per-expert GEMM run in
            // one WGSL dispatch. Supported for Q4_0/Q8_0/Q4_K.
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(_, wgpu_dev)
                if matches!(
                    self.storage.dtype(),
                    GgmlDType::Q4_0 | GgmlDType::Q8_0 | GgmlDType::Q4K
                ) =>
            {
                let out_dtype = x.dtype();
                let (e_cnt, n, k) = self.shape().dims3()?;
                let (t, topk) = ids.dims2()?;
                let s = x.dim(1)?; // 1 (gate/up: shared input) or topk (down: per-slot)
                let x_exp = if s == topk {
                    x.clone()
                } else {
                    x.broadcast_as((t, topk, k))?
                };
                let nrows = t * topk;
                let x_flat = x_exp
                    .reshape((nrows, k))?
                    .to_dtype(crate::DType::F32)?
                    .contiguous()?;
                let ids_vec = ids
                    .reshape((nrows,))?
                    .to_dtype(crate::DType::U32)?
                    .to_vec1::<u32>()?;
                if let Some(&bad) = ids_vec.iter().find(|&&e| e as usize >= e_cnt) {
                    crate::bail!("indexed_moe_forward: expert id {bad} >= num_experts {e_cnt}");
                }
                let kernel = match self.storage.dtype() {
                    GgmlDType::Q4_0 => "moe_matvec_q4_0",
                    GgmlDType::Q8_0 => "moe_matvec_q8_0",
                    GgmlDType::Q4K => "moe_matvec_q4k",
                    other => crate::bail!("wgpu MoE: no kernel for {other:?}"),
                };
                let bank = self.data()?; // raw GGML bytes for all E experts, [E, n, k]
                let wbank = wgpu_dev.upload_qweight(&bank)?;
                let ids_buf = wgpu_dev.upload_ids(&ids_vec)?;
                let y = {
                    let (store, _) = x_flat.storage_and_layout();
                    let xv = match &*store {
                        Storage::Wgpu(v) => v,
                        _ => crate::bail!("wgpu MoE: x not on wgpu after contiguous()"),
                    };
                    wgpu_dev.moe_matvec_gpu(kernel, &wbank, xv, &ids_buf, nrows, n, k)?
                };
                let out = crate::tensor::from_storage(
                    Storage::Wgpu(y),
                    (nrows, n),
                    crate::op::BackpropOp::none(),
                    false,
                );
                out.reshape((t, topk, n))?.to_dtype(out_dtype)
            }
            _ => {
                // CPU / non-CUDA fallback: per-expert quantized matmul. The packed expert bank
                // [E, n, k] is sliced into equal, contiguous per-expert quantized blocks; for
                // each expert that is actually selected we run hanzo-ml's native quantized matmul
                // on just the tokens routed to it. Nothing is dequantized, so quantized MoE runs
                // on any backend (CPU, Metal, ...) at a cost proportional to the active experts.
                use crate::Module; // brings QMatMul::forward into scope
                use std::collections::HashMap;
                use std::sync::Arc;
                let device = x.device();
                let out_dtype = x.dtype();
                let (e_cnt, n, k) = self.shape().dims3()?;
                let (t, topk) = ids.dims2()?;
                let s = x.dim(1)?; // 1 (gate/up: shared input) or topk (down: per-slot)
                let x_exp = if s == topk {
                    x.clone()
                } else {
                    x.broadcast_as((t, topk, k))?
                };
                let x_flat = x_exp
                    .reshape((t * topk, k))?
                    .to_dtype(crate::DType::F32)?
                    .contiguous()?;
                let ids_flat = ids.reshape((t * topk,))?.to_dtype(crate::DType::U32)?;
                let ids_vec = ids_flat.to_vec1::<u32>()?;
                let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
                for (slot, eid) in ids_vec.iter().enumerate() {
                    groups.entry(*eid).or_default().push(slot as u32);
                }
                let dtype = self.storage.dtype();
                let all_bytes = self.data()?;
                let expert_bytes = all_bytes.len() / e_cnt;
                let mut out_flat = Tensor::zeros((t * topk, n), crate::DType::F32, device)?;
                for (eid, slots) in groups.into_iter() {
                    let off = eid as usize * expert_bytes;
                    let qs = QStorage::from_data(
                        std::borrow::Cow::Borrowed(&all_bytes[off..off + expert_bytes]),
                        device,
                        dtype,
                    )?;
                    let shape: crate::Shape = (n, k).into();
                    let w_e = QTensor { storage: qs, shape };
                    let qm = QMatMul::from_arc(Arc::new(w_e))?;
                    let m = slots.len();
                    let idx = Tensor::from_vec(slots, (m,), device)?;
                    let x_e = x_flat.index_select(&idx, 0)?; // [m, k]
                    let y_e = qm.forward(&x_e)?.to_dtype(crate::DType::F32)?; // [m, n]
                    out_flat = out_flat.index_add(&idx, &y_e, 0)?;
                }
                out_flat.reshape((t, topk, n))?.to_dtype(out_dtype)
            }
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match &self.storage {
            QStorage::Cuda(storage) => storage.device_ptr(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(..) => crate::bail!("not implemented"),
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(..) => crate::bail!("not implemented"),
            QStorage::Metal(_) | QStorage::Cpu(_) => {
                crate::bail!("not implemented");
            }
        }
    }

    #[cfg(feature = "cuda")]
    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a crate::cuda_backend::cudarc::driver::CudaStream,
    ) -> Result<(
        *const u8,
        crate::cuda_backend::cudarc::driver::SyncOnDrop<'a>,
    )> {
        self.storage.device_ptr_with_guard(stream)
    }
}

#[derive(Clone, Debug)]
pub enum QMatMul {
    QTensor(std::sync::Arc<QTensor>),
    Tensor(Tensor),
    TensorF16(Tensor),
    // Native Vulkan quantized weight: the GGML quantized blocks live in VRAM (Q4_0/Q4_K ~0.5 B/elem,
    // Q8_0 ~1.06 B/elem). Decode (1 row) runs the matching on-GPU quant matvec kernel directly out of
    // the block format (no CPU dequant, no re-pack) -- the bandwidth lever for memory-bound decode.
    // Prefill (>1 row) dequantizes the original `qtensor` to a temporary f32 weight. `dtype` selects
    // the kernel; `n`/`k` are the weight dims.
    #[cfg(feature = "vulkan")]
    VulkanQuant {
        qtensor: std::sync::Arc<QTensor>,
        wq: std::sync::Arc<crate::VulkanStorage>,
        dtype: GgmlDType,
        n: usize,
        k: usize,
    },
    // wgpu mirror of VulkanQuant: GGML quantized blocks live in VRAM and decode (1 row) runs the
    // matching native-GGML quant matvec WGSL kernel straight out of the block format. Prefill (>1
    // row) dequantizes to a temporary f32 weight. `dtype` selects the kernel; `n`/`k` are the dims.
    #[cfg(feature = "wgpu")]
    WgpuQuant {
        qtensor: std::sync::Arc<QTensor>,
        wq: std::sync::Arc<crate::WgpuStorage>,
        dtype: GgmlDType,
        n: usize,
        k: usize,
    },
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("CANDLE_DEQUANTIZE_ALL_F16") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        // Native Vulkan quantized path: keep the GGML quantized blocks in VRAM and run the matching
        // on-GPU quant matvec for decode, instead of dequantizing the whole model to f32 (4x the
        // decode bandwidth). The kernel reads the GGML block format straight from the uploaded bytes
        // -- no CPU dequant, no re-pack -- so this is exact w.r.t. the CPU reference. Q4_0/Q4_K need
        // k a multiple of their block (32 / 256); Q8_0 needs a multiple of 32.
        #[cfg(feature = "vulkan")]
        {
            let dt = qtensor.dtype();
            let native_vk = matches!(dt, GgmlDType::Q4_0 | GgmlDType::Q8_0 | GgmlDType::Q4K);
            if native_vk {
                if let Device::Vulkan(d) = qtensor.device() {
                    if let Ok((n, k)) = qtensor.shape().dims2() {
                        let blk = dt.block_size();
                        if k % blk == 0 {
                            let bytes = qtensor.data()?;
                            let wq = d.upload_qweight(&bytes)?;
                            return Ok(Self::VulkanQuant {
                                qtensor,
                                wq: std::sync::Arc::new(wq),
                                dtype: dt,
                                n,
                                k,
                            });
                        }
                    }
                }
            }
        }
        // Native wgpu quantized path: same idea as Vulkan. Ships the Q4_0/Q8_0/Q4_K WGSL matvec
        // kernels; other dtypes fall through to the dequantize path below.
        #[cfg(feature = "wgpu")]
        {
            let dt = qtensor.dtype();
            let native_wgpu = matches!(dt, GgmlDType::Q4_0 | GgmlDType::Q8_0 | GgmlDType::Q4K);
            if native_wgpu {
                if let Device::Wgpu(d) = qtensor.device() {
                    if let Ok((n, k)) = qtensor.shape().dims2() {
                        let blk = dt.block_size();
                        if k % blk == 0 {
                            let bytes = qtensor.data()?;
                            let wq = d.upload_qweight(&bytes)?;
                            return Ok(Self::WgpuQuant {
                                qtensor,
                                wq: std::sync::Arc::new(wq),
                                dtype: dt,
                                n,
                                k,
                            });
                        }
                    }
                }
            }
        }
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            // The Vulkan/wgpu backends have no generic native quantized matmul, so dequantize to f32
            // here (once, at construction) and run the regular f32 GPU matmul.
            _ => DEQUANTIZE_ALL.with(|b| *b) || qtensor.device().is_vulkan() || qtensor.device().is_wgpu(),
        };
        let t = if dequantize {
            let tensor = qtensor.dequantize(&qtensor.device())?;
            Self::Tensor(tensor)
        } else if DEQUANTIZE_ALL_F16.with(|b| *b) {
            let tensor = qtensor.dequantize_f16(&qtensor.device())?;
            Self::TensorF16(tensor)
        } else {
            Self::QTensor(qtensor)
        };
        Ok(t)
    }

    pub fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        Self::from_arc(std::sync::Arc::new(qtensor))
    }

    pub fn dequantize_f16(&self) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.dequantize_f16(&t.device()),
            Self::Tensor(t) => t.to_dtype(DType::F16),
            Self::TensorF16(t) => Ok(t.clone()),
            #[cfg(feature = "vulkan")]
            Self::VulkanQuant { qtensor, .. } => qtensor.dequantize_f16(&qtensor.device()),
            #[cfg(feature = "wgpu")]
            Self::WgpuQuant { qtensor, .. } => qtensor.dequantize_f16(&qtensor.device()),
        }
    }

    pub fn forward_via_f16(&self, xs: &Tensor) -> Result<Tensor> {
        let w = self.dequantize_f16()?;
        let in_dtype = xs.dtype();
        let w = match *xs.dims() {
            [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
            [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
            _ => w.t()?,
        };
        xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match self {
            Self::QTensor(t) => t.indexed_moe_forward(x, ids),
            #[cfg(feature = "vulkan")]
            Self::VulkanQuant { qtensor, .. } => qtensor.indexed_moe_forward(x, ids),
            #[cfg(feature = "wgpu")]
            Self::WgpuQuant { qtensor, .. } => qtensor.indexed_moe_forward(x, ids),
            _ => {
                panic!("Not implemented!")
            }
        }
    }
}

impl crate::CustomOp1 for QTensor {
    fn name(&self) -> &'static str {
        "qmatmul"
    }

    fn cpu_fwd(
        &self,
        storage: &crate::CpuStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CpuStorage, Shape)> {
        if !layout.is_contiguous() {
            crate::bail!("input tensor is not contiguous {layout:?}")
        }
        let src_shape = layout.shape();
        // self is transposed so n is first then k.
        let (n, k) = self.shape.dims2()?;
        if src_shape.rank() < 2 {
            crate::bail!("input tensor has only one dimension {layout:?}")
        }
        let mut dst_shape = src_shape.dims().to_vec();
        let last_k = dst_shape.pop().unwrap();
        if last_k != k {
            crate::bail!("input tensor {layout:?} incompatible with {:?}", self.shape)
        }
        dst_shape.push(n);
        let dst_shape = Shape::from(dst_shape);
        #[allow(clippy::infallible_destructuring_match)]
        let self_storage = match &self.storage {
            QStorage::Cpu(storage) => storage,
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(..) => crate::bail!("Invalid storage"),
            #[cfg(feature = "wgpu")]
            QStorage::Wgpu(..) => crate::bail!("Invalid storage"),
            QStorage::Metal(_) | QStorage::Cuda(_) => crate::bail!("Invalid storage"),
        };
        match storage.dtype() {
            DType::F32 => {
                let slice = storage.as_slice::<f32>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![0f32; dst_shape.elem_count()];
                self_storage.matmul_t(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F32(dst_storage), dst_shape))
            }
            DType::F16 => {
                let slice = storage.as_slice::<f16>()?;
                let slice =
                    &slice[layout.start_offset()..layout.start_offset() + src_shape.elem_count()];
                let mut dst_storage = vec![f16::ZERO; dst_shape.elem_count()];
                self_storage.matmul_t_f16(
                    (dst_shape.elem_count() / n, k, n),
                    slice,
                    &mut dst_storage,
                )?;
                Ok((crate::CpuStorage::F16(dst_storage), dst_shape))
            }
            _ => crate::bail!("Expected f32/f16"),
        }
    }

    fn metal_fwd(
        &self,
        storage: &crate::MetalStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::MetalStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Metal(metal) => metal,
            _ => unreachable!("Cannot call metal matmul on non metal QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }

    fn cuda_fwd(
        &self,
        storage: &crate::CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(crate::CudaStorage, Shape)> {
        let self_storage = match &self.storage {
            QStorage::Cuda(cuda) => cuda,
            _ => unreachable!("Cannot call cuda matmul on non cuda QTensor"),
        };
        self_storage.fwd(&self.shape, storage, layout)
    }
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            #[cfg(feature = "vulkan")]
            Self::VulkanQuant {
                qtensor,
                wq,
                dtype,
                n,
                k,
            } => {
                let rows: usize = xs.elem_count() / *k;
                if rows == 1 {
                    // Decode: weights stay quantized in VRAM; the matching native-GGML quant matvec
                    // runs straight out of the block format (no dequant, no copy).
                    let xs = xs.contiguous()?;
                    let d = match xs.device() {
                        Device::Vulkan(d) => d,
                        _ => crate::bail!("VulkanQuant input not on vulkan"),
                    };
                    let y = {
                        let (store, _) = xs.storage_and_layout();
                        let xv = match &*store {
                            crate::Storage::Vulkan(v) => v,
                            _ => crate::bail!("VulkanQuant expected vulkan storage"),
                        };
                        match dtype {
                            GgmlDType::Q4_0 => d.matvec_q4_0_gpu(wq, xv, *n, *k)?,
                            GgmlDType::Q8_0 => d.matvec_q8_0_gpu(wq, xv, *n, *k)?,
                            GgmlDType::Q4K => d.matvec_q4k_gpu(wq, xv, *n, *k)?,
                            other => crate::bail!("VulkanQuant: no native matvec for {other:?}"),
                        }
                    };
                    let mut dims = xs.dims().to_vec();
                    let last = dims.len() - 1;
                    dims[last] = *n;
                    Ok(crate::tensor::from_storage(
                        crate::Storage::Vulkan(y),
                        dims,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                } else {
                    // Prefill: dequantize to a temporary f32 weight (reuses the NT matmul path).
                    let w = qtensor.dequantize(&xs.device())?;
                    let w = match *xs.dims() {
                        [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                        [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                        _ => w.t()?,
                    };
                    xs.matmul(&w)
                }
            }
            #[cfg(feature = "wgpu")]
            Self::WgpuQuant {
                qtensor,
                wq,
                dtype,
                n,
                k,
            } => {
                let rows: usize = xs.elem_count() / *k;
                if rows == 1 {
                    // Decode: weights stay quantized in VRAM; the matching native-GGML quant matvec
                    // WGSL kernel runs straight out of the block format (no dequant, no copy).
                    let xs = xs.contiguous()?;
                    let d = match xs.device() {
                        Device::Wgpu(d) => d,
                        _ => crate::bail!("WgpuQuant input not on wgpu"),
                    };
                    let y = {
                        let (store, _) = xs.storage_and_layout();
                        let xv = match &*store {
                            crate::Storage::Wgpu(v) => v,
                            _ => crate::bail!("WgpuQuant expected wgpu storage"),
                        };
                        match dtype {
                            GgmlDType::Q4_0 => d.matvec_q4_0_gpu(wq, xv, *n, *k)?,
                            GgmlDType::Q8_0 => d.matvec_q8_0_gpu(wq, xv, *n, *k)?,
                            GgmlDType::Q4K => d.matvec_q4k_gpu(wq, xv, *n, *k)?,
                            other => crate::bail!("WgpuQuant: no native matvec for {other:?}"),
                        }
                    };
                    let mut dims = xs.dims().to_vec();
                    let last = dims.len() - 1;
                    dims[last] = *n;
                    Ok(crate::tensor::from_storage(
                        crate::Storage::Wgpu(y),
                        dims,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                } else {
                    // Prefill: dequantize to a temporary f32 weight (reuses the NT matmul path).
                    let w = qtensor.dequantize(&xs.device())?;
                    let w = match *xs.dims() {
                        [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                        [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                        _ => w.t()?,
                    };
                    xs.matmul(&w)
                }
            }
            Self::QTensor(t) => xs.apply_op1_no_bwd(t.as_ref()),
            Self::Tensor(w) => {
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.matmul(&w)
            }
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                let w = match *xs.dims() {
                    [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                    [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                    _ => w.t()?,
                };
                xs.to_dtype(DType::F16)?.matmul(&w)?.to_dtype(in_dtype)
            }
        }
    }
}
