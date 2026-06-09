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
#[cfg(not(feature = "metal"))]
mod metal {
    pub use super::dummy_metal::*;
}
#[cfg(feature = "cuda")]
pub mod cuda;
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
                GgmlDType::MXFP4 => metal::load_quantized(d, as_t_slice::<BlockMXFP4>(&data)),
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
                GgmlDType::MXFP4 => cuda::load_quantized(d, as_t_slice::<BlockMXFP4>(&data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(&data)),
            },
            #[cfg(feature = "rocm")]
            Device::Rocm(_) => crate::bail!("quantized tensors on rocm are not supported yet"),
            #[cfg(feature = "vulkan")]
            Device::Vulkan(d) => Ok(Self::Vulkan(dtype.from_data(data), d.clone())),
        }
    }

    fn block_size(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.block_size(),
            QStorage::Metal(storage) => storage.dtype().block_size(),
            QStorage::Cuda(storage) => storage.dtype().block_size(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(storage, _) => storage.block_size(),
        }
    }

    fn dtype(&self) -> GgmlDType {
        match self {
            QStorage::Cpu(storage) => storage.dtype(),
            QStorage::Metal(storage) => storage.dtype(),
            QStorage::Cuda(storage) => storage.dtype(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(storage, _) => storage.dtype(),
        }
    }

    fn device(&self) -> Device {
        match self {
            QStorage::Cpu(_storage) => Device::Cpu,
            QStorage::Metal(storage) => Device::Metal(storage.device().clone()),
            QStorage::Cuda(storage) => Device::Cuda(storage.device().clone()),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(_storage, device) => Device::Vulkan(device.clone()),
        }
    }

    fn size_in_bytes(&self) -> usize {
        match self {
            QStorage::Cpu(storage) => storage.storage_size_in_bytes(),
            QStorage::Metal(storage) => storage.storage_size_in_bytes(),
            QStorage::Cuda(storage) => storage.storage_size_in_bytes(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(storage, _) => storage.storage_size_in_bytes(),
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
        }
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        match self {
            QStorage::Cuda(storage) => storage.device_ptr(),
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(..) => crate::bail!("not implemented"),
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
            #[cfg(feature = "vulkan")]
            QStorage::Vulkan(..) => crate::bail!("not implemented"),
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
    // 4-bit microscaling float (MXFP4), ggml type 39: 32 elems / block, E8M0 scale + 16 nibble-pairs.
    MXFP4,
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
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            30 => Self::BF16,
            39 => Self::MXFP4,
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
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            Self::BF16 => 30,
            Self::MXFP4 => 39,
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
            Self::MXFP4 => Box::new(vec![BlockMXFP4::zeros(); elem_count / BlockMXFP4::BLCK_SIZE]),
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
            Self::MXFP4 => Box::new(as_t_slice::<BlockMXFP4>(&data).to_vec()),
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
            Self::MXFP4 => std::mem::size_of::<BlockMXFP4>(),
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
            Self::Q2K | Self::Q3K | Self::Q4K | Self::Q5K | Self::Q6K | Self::Q8K => k_quants::QK_K,
            Self::MXFP4 => k_quants::QK_MXFP4,
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
            QStorage::Cuda(s) => match (&*x.storage(), &*ids.storage()) {
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
            },
            _ => {
                // CPU / Vulkan / non-CUDA fallback: per-expert quantized matmul. The packed expert
                // bank [E, n, k] stays quantized; it is sliced into equal, contiguous per-expert
                // blocks and only the SELECTED experts are touched per token. Each selected expert
                // goes through QMatMul::from_arc + forward, so on Vulkan a Q4_K expert uses the
                // in-shader-decode matvec and other dtypes dequantize just that one expert's weight
                // (bounded to `topk` experts, never the whole [E,n,k] bank -- that bank is the 35B
                // MoE's ~130GB f32 OOM). Cost is proportional to the active experts.
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
                    // VulkanQuant's single-row matvec returns rank-1 [n]; force [m, n] for index_add.
                    let y_e = qm
                        .forward(&x_e)?
                        .to_dtype(crate::DType::F32)?
                        .reshape((m, n))?;
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
    // Native Vulkan quantized weight: quantized blocks resident in VRAM (Q8_0 ~1.125 B/elem, Q4_K
    // ~0.5625 B/elem) instead of a dequantized f32 copy (4 B/elem). Decode (1 row) runs the matching
    // GPU matvec directly (bandwidth-optimal); prefill dequantizes to a temporary f32. `qtensor` is
    // the original (CPU-side blocks) for the prefill dequant; `dtype` selects the matvec kernel;
    // `n`/`k` are the weight dims.
    #[cfg(feature = "vulkan")]
    VulkanQuant {
        qtensor: std::sync::Arc<QTensor>,
        wq: std::sync::Arc<crate::VulkanStorage>,
        dtype: GgmlDType,
        n: usize,
        k: usize,
        // Lazily-filled f16 dequant of `wq` (row-major [n,k]) for the coopmat prefill GEMM. Built
        // once on the first M>1 forward and reused; None until then (decode never needs it).
        w16: std::sync::Arc<std::sync::Mutex<Option<std::sync::Arc<crate::VulkanStorage>>>>,
    },
    // Resident MoE expert bank: the WHOLE [e, n, k] quantized weight uploaded to VRAM ONCE at load,
    // kept quantized (verbatim blocks, never dequantized -- the f32 bank is a ~140GB OOM on the 35B).
    // indexed_moe_forward indexes per routed expert by its block offset into `bank` instead of
    // re-uploading that expert every token (the 97x-slower-than-CUDA decode bug). `per_expert_words`
    // is the u32 stride between experts in `bank` (n * k/blocksize * gpu_block_u32 for `dtype`).
    #[cfg(feature = "vulkan")]
    VulkanQuantBank {
        qtensor: std::sync::Arc<QTensor>,
        bank: std::sync::Arc<crate::VulkanStorage>,
        dtype: GgmlDType,
        e: usize,
        n: usize,
        k: usize,
        per_expert_words: usize,
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

// u32 words a single quantized block occupies in the GPU bank/weight buffer for `dtype`, i.e. the
// stride the matvec/matmul shaders step by. This is the GPU (uploaded) block size, which differs
// from the source block byte size for the padded/repacked dtypes: Q6_K source is 210 B but uploads
// as 212 B (53 u32) and Q8_0 source is 34 B but uploads as 36 B (9 u32); Q4_K (144 B) and Q5_K
// (176 B) upload verbatim. Returns None for dtypes with no native GPU quantized kernel.
#[cfg(feature = "vulkan")]
fn gpu_block_u32(dtype: GgmlDType) -> Option<usize> {
    match dtype {
        GgmlDType::Q8_0 => Some(9),  // 36 B / 4 (repacked from 34 B source)
        GgmlDType::Q4K => Some(36),  // 144 B / 4 (verbatim)
        GgmlDType::Q5K => Some(44),  // 176 B / 4 (verbatim)
        GgmlDType::Q6K => Some(53),  // 212 B / 4 (210 B source padded to 212)
        _ => None,
    }
}

impl QMatMul {
    pub fn from_arc(qtensor: std::sync::Arc<QTensor>) -> Result<Self> {
        // Native Vulkan quantized path: keep weights quantized in VRAM and run the GPU matvec for
        // decode, instead of dequantizing the whole model to f32 (4x+ the decode bandwidth, and for
        // big MoE expert banks an outright VRAM blow-up). 2D weights (one matmul) become VulkanQuant
        // with a GPU matvec; 3D banks ([E,n,k] MoE experts) stay as QTensor so indexed_moe_forward
        // dequantizes ONLY the routed experts per token, never all E at once.
        #[cfg(feature = "vulkan")]
        if let Device::Vulkan(d) = qtensor.device() {
            match qtensor.dtype() {
                GgmlDType::Q8_0 => {
                    if let Ok((n, k)) = qtensor.shape().dims2() {
                        if k % 32 == 0 {
                            let wf = qtensor
                                .dequantize(&Device::Cpu)?
                                .flatten_all()?
                                .to_vec1::<f32>()?;
                            let wq = d.quantize_q8(&wf, n, k)?;
                            return Ok(Self::VulkanQuant {
                                qtensor,
                                wq: std::sync::Arc::new(wq),
                                dtype: GgmlDType::Q8_0,
                                n,
                                k,
                                w16: std::sync::Arc::new(std::sync::Mutex::new(None)),
                            });
                        }
                    }
                }
                GgmlDType::Q4K => {
                    if let Ok((n, k)) = qtensor.shape().dims2() {
                        if k % 256 == 0 {
                            // Upload the native Q4_K blocks verbatim (no requantize): the in-shader
                            // decode matches BlockQ4K::to_float, so this is lossless vs the CPU path.
                            let bytes = qtensor.data()?;
                            let wq = d.quantize_q4k(bytes.as_ref(), n, k)?;
                            return Ok(Self::VulkanQuant {
                                qtensor,
                                wq: std::sync::Arc::new(wq),
                                dtype: GgmlDType::Q4K,
                                n,
                                k,
                                w16: std::sync::Arc::new(std::sync::Mutex::new(None)),
                            });
                        }
                    }
                }
                _ => {}
            }
            // 3D quantized expert banks ([E,n,k]): upload the WHOLE bank to VRAM ONCE here, kept
            // quantized (verbatim blocks, NEVER dequantized -- the f32 bank is a ~140GB OOM on the
            // 35B MoE). indexed_moe_forward then runs each routed expert's matvec/matmul by indexing
            // its block offset into the resident `bank`, with NO per-token re-upload (the prior path
            // re-uploaded every active expert every token: 97x slower than CUDA). Covers the
            // Q4_K_XL/M mix dtypes that have a native GPU quantized kernel (Q4K gate/up, Q6K down,
            // Q5K, Q8_0). A bank dtype with no kernel (e.g. Q4_0/Q2K) keeps the old QTensor path.
            let is_quantized = !matches!(
                qtensor.dtype(),
                GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16
            );
            if is_quantized && qtensor.shape().rank() == 3 {
                let dtype = qtensor.dtype();
                if let Ok((e, n, k)) = qtensor.shape().dims3() {
                    let bs8 = GgmlDType::Q8_0.block_size();
                    // Q4K/Q8_0 banks upload verbatim (their woff kernels are proven: Q4K via the 35B
                    // attention path, Q8 via the dense path). Other quantized dtypes (Q5K/Q6K, e.g. the
                    // Q4_K_M down-experts) requantize to Q8 at load -- identical numerics to the old
                    // per-token Q6K->Q8 path, but resident -- since their verbatim matvec kernels are
                    // unproven. One bank's f32 transient is sub-GB, not the whole-model OOM.
                    let built = if matches!(dtype, GgmlDType::Q4K) && k % dtype.block_size() == 0 {
                        Some((d.quantize_q4k(qtensor.data()?.as_ref(), e * n, k)?, GgmlDType::Q4K))
                    } else if matches!(dtype, GgmlDType::Q8_0) && k % bs8 == 0 {
                        Some((
                            d.quantize_q8_blocks(qtensor.data()?.as_ref(), e * n, k)?,
                            GgmlDType::Q8_0,
                        ))
                    } else if k % bs8 == 0 {
                        let wf = qtensor
                            .dequantize(&Device::Cpu)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        Some((d.quantize_q8(&wf, e * n, k)?, GgmlDType::Q8_0))
                    } else {
                        None
                    };
                    if let Some((bank, store)) = built {
                        let blk_u32 = gpu_block_u32(store).expect("Q4K/Q8 have gpu blocks");
                        let per_expert_words = n * (k / store.block_size()) * blk_u32;
                        // A stride mismatch means we would silently read the wrong expert at forward.
                        assert_eq!(
                            bank.len_words(),
                            e * per_expert_words,
                            "vulkan MoE bank stride mismatch: {} words != {e}*{per_expert_words}",
                            bank.len_words()
                        );
                        return Ok(Self::VulkanQuantBank {
                            qtensor,
                            bank: std::sync::Arc::new(bank),
                            dtype: store,
                            e,
                            n,
                            k,
                            per_expert_words,
                        });
                    }
                }
                // Unsupported bank dtype/shape: keep the per-expert dequant QTensor path.
                return Ok(Self::QTensor(qtensor));
            }
            // Any other 2D quantized linear (Q6K down/output, Q5K, Q4_0, Q5_0, ...) has no native
            // keep-quantized matvec kernel, so the old path dequantized it to a 4 B/elem f32 weight
            // in VRAM -- 3.5x the decode bandwidth and the f32 VRAM blowup. Instead requantize the
            // (already-lossy) dequant to Q8_0 on the GPU and run the native Q8 matvec: every decode
            // linear now reads ~1.125 B/elem. The extra error vs the source quant is tiny (Q8 of an
            // already-Q5K/Q6K row), and prefill still dequantizes `qtensor` to f32 for the NT matmul.
            if is_quantized {
                if let Ok((n, k)) = qtensor.shape().dims2() {
                    if k % 32 == 0 {
                        let wf = qtensor
                            .dequantize(&Device::Cpu)?
                            .flatten_all()?
                            .to_vec1::<f32>()?;
                        let wq = d.quantize_q8(&wf, n, k)?;
                        return Ok(Self::VulkanQuant {
                            qtensor,
                            wq: std::sync::Arc::new(wq),
                            dtype: GgmlDType::Q8_0,
                            n,
                            k,
                            w16: std::sync::Arc::new(std::sync::Mutex::new(None)),
                        });
                    }
                }
            }
        }
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            // The Vulkan backend has no native quantized matmul yet, so dequantize to f32 here
            // (once, at construction) and run the regular f32 GPU matmul.
            _ => DEQUANTIZE_ALL.with(|b| *b) || qtensor.device().is_vulkan(),
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
            #[cfg(feature = "vulkan")]
            Self::VulkanQuantBank { qtensor, .. } => qtensor.dequantize_f16(&qtensor.device()),
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
            #[cfg(feature = "vulkan")]
            Self::VulkanQuantBank {
                bank,
                dtype,
                e,
                n,
                k,
                per_expert_words,
                ..
            } => self.bank_indexed_moe_forward(x, ids, bank, *dtype, *e, *n, *k, *per_expert_words),
            _ => {
                panic!("Not implemented!")
            }
        }
    }

    /// Resident-bank MoE forward (Vulkan). The whole [E,n,k] quantized expert bank is already in VRAM
    /// (`bank`); this routes tokens to experts, groups slots by expert, and for each selected expert
    /// dispatches a banked matvec (decode: that expert has m==1 routed slot) or banked matmul
    /// (prefill: m>1) AT the expert's u32 offset (`eid * per_expert_words`) into `bank`, accumulating
    /// into the output. NO per-expert re-upload, NO whole-bank f32 dequant -- the fix for the 97x
    /// decode regression. Mirrors the group-by-expert/index_select/index_add structure of the old
    /// QTensor path so the routing math is unchanged; only the per-expert weight access differs.
    #[cfg(feature = "vulkan")]
    #[allow(clippy::too_many_arguments)]
    fn bank_indexed_moe_forward(
        &self,
        x: &Tensor,
        ids: &Tensor,
        bank: &crate::VulkanStorage,
        dtype: GgmlDType,
        e_cnt: usize,
        n: usize,
        k: usize,
        per_expert_words: usize,
    ) -> Result<Tensor> {
        use std::collections::HashMap;
        let device = x.device();
        let d = match device {
            Device::Vulkan(d) => d.clone(),
            _ => crate::bail!("VulkanQuantBank indexed_moe_forward: input not on vulkan"),
        };
        let out_dtype = x.dtype();
        let (t, topk) = ids.dims2()?;
        let s = x.dim(1)?; // 1 (gate/up: shared input) or topk (down: per-slot)
        let x_exp = if s == topk {
            x.clone()
        } else {
            x.broadcast_as((t, topk, k))?
        };
        // Flatten activations to [t*topk, k] f32 contiguous (the matvec/matmul kernels read f32 x).
        let x_flat = x_exp
            .reshape((t * topk, k))?
            .to_dtype(crate::DType::F32)?
            .contiguous()?;
        let ids_flat = ids
            .reshape((t * topk,))?
            .to_dtype(crate::DType::U32)?
            .contiguous()?;
        let m_all = t * topk;
        // Fused grouped matvec: ONE dispatch per projection reads the routed expert id per slot from a
        // GPU buffer and computes every routed-slot output at once -- NO routing-ids GPU->CPU sync, NO
        // HashMap, NO Tensor::zeros, NO index_select/index_add. Decode is bit-identical to the
        // matvec_*_gpu_off kernels. Handles BOTH decode (t==1) and prefill (t>1): slots index
        // x_flat/ids_flat directly. Banks are only ever stored Q4K (gate/up) or Q8_0 (down, and any
        // Q5K/Q6K requantized at load); any other dtype falls through to the loop below.
        if matches!(dtype, GgmlDType::Q4K | GgmlDType::Q8_0) {
            let (xstore, _) = x_flat.storage_and_layout();
            let xv = match &*xstore {
                crate::Storage::Vulkan(v) => v,
                _ => crate::bail!("VulkanQuantBank expected vulkan storage"),
            };
            let (istore, _) = ids_flat.storage_and_layout();
            let iv = match &*istore {
                crate::Storage::Vulkan(v) => v,
                _ => crate::bail!("VulkanQuantBank expected vulkan storage"),
            };
            let y_store = match dtype {
                GgmlDType::Q4K => {
                    d.mul_mat_vec_id_q4k_gpu(bank, xv, iv, m_all, n, k, per_expert_words)?
                }
                _ => d.mul_mat_vec_id_q8_gpu(bank, xv, iv, m_all, n, k, per_expert_words)?,
            };
            let y = crate::tensor::from_storage(
                crate::Storage::Vulkan(y_store),
                (m_all, n),
                crate::op::BackpropOp::none(),
                false,
            );
            return y.reshape((t, topk, n))?.to_dtype(out_dtype);
        }
        // Fallback (no id-kernel for this bank dtype): group slots by expert on the CPU and run a
        // banked matvec/matmul per expert. The one-edit revert to this path: replace the
        // `matches!(dtype, ..)` guard above with `false`.
        let ids_vec = ids_flat.to_vec1::<u32>()?;
        let mut groups: HashMap<u32, Vec<u32>> = HashMap::new();
        for (slot, eid) in ids_vec.iter().enumerate() {
            groups.entry(*eid).or_default().push(slot as u32);
        }
        let mut out_flat = Tensor::zeros((t * topk, n), crate::DType::F32, device)?;
        for (eid, slots) in groups.into_iter() {
            if eid as usize >= e_cnt {
                crate::bail!("VulkanQuantBank: expert id {eid} >= E {e_cnt}");
            }
            let woff = eid as usize * per_expert_words;
            let m = slots.len();
            let idx = Tensor::from_vec(slots, (m,), device)?;
            let x_e = x_flat.index_select(&idx, 0)?.contiguous()?; // [m, k]
            // Run expert `eid` at its block offset in the resident bank. Decode is m==1 (matvec);
            // prefill m>1 uses the banked matmul where one exists (Q4K/Q8), else loops the matvec per
            // row (Q5K/Q6K have no matmul kernel) -- still no re-upload, weights stay quantized.
            let y_store = {
                let (store, _) = x_e.storage_and_layout();
                let xv = match &*store {
                    crate::Storage::Vulkan(v) => v,
                    _ => crate::bail!("VulkanQuantBank expected vulkan storage"),
                };
                if m == 1 {
                    match dtype {
                        GgmlDType::Q4K => d.matvec_q4k_gpu_off(bank, xv, n, k, woff)?,
                        GgmlDType::Q5K => d.matvec_q5k_gpu_off(bank, xv, n, k, woff)?,
                        GgmlDType::Q6K => d.matvec_q6k_gpu_off(bank, xv, n, k, woff)?,
                        _ => d.matvec_q8_gpu_off(bank, xv, n, k, woff)?,
                    }
                } else {
                    match dtype {
                        // Q4_K banked prefill: int8 dp4a when the device has hw int8 dot, else
                        // f32-decode. Both run at the bank's woff with no re-upload.
                        GgmlDType::Q4K if d.int_dot8() => {
                            d.matmul_q4k_dp4a_gpu_off(bank, xv, m, n, k, woff)?
                        }
                        GgmlDType::Q4K => d.matmul_q4k_gpu_off(bank, xv, m, n, k, woff)?,
                        // Q8_0 banked prefill: int8 dp4a when the device has hw int8 dot, else
                        // f32-decode. Both run at the bank's woff with no re-upload.
                        GgmlDType::Q8_0 if d.int_dot8() => {
                            d.matmul_q8_dp4a_gpu_off(bank, xv, m, n, k, woff)?
                        }
                        GgmlDType::Q8_0 => d.matmul_q8_gpu_off(bank, xv, m, n, k, woff)?,
                        // No matmul kernel for this dtype: one banked matvec per routed row.
                        _ => {
                            let mut rows = Vec::with_capacity(m);
                            for r in 0..m {
                                let xr = x_e.narrow(0, r, 1)?.contiguous()?;
                                let (rstore, _) = xr.storage_and_layout();
                                let rv = match &*rstore {
                                    crate::Storage::Vulkan(v) => v,
                                    _ => crate::bail!("VulkanQuantBank expected vulkan storage"),
                                };
                                let yr = match dtype {
                                    GgmlDType::Q5K => d.matvec_q5k_gpu_off(bank, rv, n, k, woff)?,
                                    GgmlDType::Q6K => d.matvec_q6k_gpu_off(bank, rv, n, k, woff)?,
                                    _ => d.matvec_q8_gpu_off(bank, rv, n, k, woff)?,
                                };
                                let yr = crate::tensor::from_storage(
                                    crate::Storage::Vulkan(yr),
                                    (1usize, n),
                                    crate::op::BackpropOp::none(),
                                    false,
                                );
                                rows.push(yr);
                            }
                            let y = Tensor::cat(&rows, 0)?;
                            out_flat = out_flat.index_add(&idx, &y, 0)?;
                            continue;
                        }
                    }
                }
            };
            // matvec returns [n]; matmul returns [m*n]. Reshape to [m, n] for index_add.
            let y_e = crate::tensor::from_storage(
                crate::Storage::Vulkan(y_store),
                (m, n),
                crate::op::BackpropOp::none(),
                false,
            );
            out_flat = out_flat.index_add(&idx, &y_e, 0)?;
        }
        out_flat.reshape((t, topk, n))?.to_dtype(out_dtype)
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
                w16,
            } => {
                let rows: usize = xs.elem_count() / *k;
                if rows == 1 {
                    // Decode: weights stay quantized in VRAM, GPU matvec (no dequant, no copy).
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
                            GgmlDType::Q4K => d.matvec_q4k_gpu(wq, xv, *n, *k)?,
                            // Subgroup matvec is the best decode kernel here: decode is HBM-bandwidth-
                            // bound, and the dp4a matmul (great for prefill's M-row reuse) is slower at
                            // M=1 (no reuse + an extra activation-quantize dispatch). Measured 9.3 vs 6.1.
                            _ => d.matvec_q8_gpu(wq, xv, *n, *k)?,
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
                } else if matches!(dtype, GgmlDType::Q4K | GgmlDType::Q8_0)
                    && xs.device().is_vulkan()
                {
                    // Prefill: keep weights quantized in VRAM and decode in-shader (M>1 matmul),
                    // amortizing each decoded weight block across all M rows. This replaces the old
                    // path that dequantized the ENTIRE weight to f32 every forward (the dominant
                    // time-to-first-token cost). Flatten xs to [M, k], run the GPU matmul, reshape
                    // the [M, n] result back to the input's batch dims with last dim = n.
                    let xs = xs.contiguous()?;
                    let m = xs.elem_count() / *k;
                    let xs2 = xs.reshape((m, *k))?;
                    let d = match xs2.device() {
                        Device::Vulkan(d) => d,
                        _ => crate::bail!("VulkanQuant input not on vulkan"),
                    };
                    let y = {
                        let (store, _) = xs2.storage_and_layout();
                        let xv = match &*store {
                            crate::Storage::Vulkan(v) => v,
                            _ => crate::bail!("VulkanQuant expected vulkan storage"),
                        };
                        match dtype {
                            // Q4_K prefill: int8 dp4a (compute-bound lever) when the device has hw
                            // int8 dot, else the f32-decode matmul. Both produce [m, n] identically.
                            GgmlDType::Q4K if d.int_dot8() => d.matmul_q4k_dp4a_gpu(wq, xv, m, *n, *k)?,
                            GgmlDType::Q4K => d.matmul_q4k_gpu(wq, xv, m, *n, *k)?,
                            // Q8_0 prefill GEMM selection. The RDNA3 warp-tiled int8-dp4a kernel
                            // (mul_mm_q8_mmq, llama MMQ tiling) is the default when the device has hw
                            // int8 dot; it A/B'd ahead of the coopmat (WMMA) GEMM on the 8060S. Override
                            // for benchmarking via HANZO_VK_Q8_PREFILL = mmq | dp4a | coopmat.
                            _ if d.int_dot8() && k.is_multiple_of(32) => {
                                let _ = &w16;
                                match std::env::var("HANZO_VK_Q8_PREFILL").as_deref() {
                                    Ok("coopmat") if d.coopmat_info().is_some() => {
                                        d.matmul_q8_cm_gpu(wq, xv, m, *n, *k)?
                                    }
                                    Ok("dp4a") => d.matmul_q8_mm_dp4a_gpu(wq, xv, m, *n, *k)?,
                                    _ => d.matmul_q8_mmq_gpu(wq, xv, m, *n, *k)?,
                                }
                            }
                            // Q8_0 prefill on coopmat devices without int8 dot: WMMA GEMM.
                            _ if d.coopmat_info().is_some() && k.is_multiple_of(32) => {
                                let _ = &w16;
                                d.matmul_q8_cm_gpu(wq, xv, m, *n, *k)?
                            }
                            _ => d.matmul_q8_gpu(wq, xv, m, *n, *k)?,
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
                    // Fallback (non-Q4K/Q8 dtype or non-Vulkan device): dequantize to a temporary f32
                    // weight and reuse the NT matmul path.
                    let w = qtensor.dequantize(&xs.device())?;
                    let w = match *xs.dims() {
                        [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                        [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                        _ => w.t()?,
                    };
                    xs.matmul(&w)
                }
            }
            #[cfg(feature = "vulkan")]
            Self::VulkanQuantBank { .. } => {
                // A 3D MoE expert bank has no single-matrix forward; it is only ever driven through
                // indexed_moe_forward (the engine never calls .forward on an expert bank).
                crate::bail!("VulkanQuantBank: use indexed_moe_forward, not forward")
            }
            Self::QTensor(t) => {
                // The `qmatmul` CustomOp1 has no correct Vulkan path for non-VulkanQuant weights
                // (e.g. Q6_K MoE down experts); dequantize this one weight and matmul like Self::Tensor.
                #[cfg(feature = "vulkan")]
                if xs.device().is_vulkan() {
                    let w = t.dequantize(xs.device())?;
                    let w = match *xs.dims() {
                        [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                        [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                        _ => w.t()?,
                    };
                    return xs.matmul(&w);
                }
                xs.apply_op1_no_bwd(t.as_ref())
            }
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
