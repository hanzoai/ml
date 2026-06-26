use crate::{
    backend::BackendStorage, CpuStorage, DType, Device, Result, Shape, Storage, Tensor, D,
};
use iq_quants::*;
use k_quants::*;
use std::borrow::Cow;

#[cfg(target_feature = "avx2")]
pub mod avx;
mod dummy_cuda;
mod dummy_metal;
pub mod ggml_file;
pub mod gguf_file;
mod iq_grids;
pub mod iq_quants;
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
// Declarative GGUF quant-type formatter (Cut 2). Additive: defines `quant_format!` and a
// `#[cfg(test)]` equivalence proof; does not alter any existing type or wiring.
pub mod quant_format;
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
            Device::Rocm(d) => {
                // Mirror Vulkan: keep quantized blocks in a CPU box + the device; QMatMul
                // dequantizes them to a dense ROCm tensor on use.
                let storage = dtype.cpu_zeros(elem_count);
                Ok(QStorage::Rocm(storage, d.clone()))
            }
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
    // ROCm mirrors the Vulkan path: quantized blocks held in a CPU box + the device; QMatMul
    // dequantizes them to a dense f32 ROCm tensor on demand (rocBLAS matmul). A native HIP quant
    // matmul is the bandwidth-win follow-up.
    #[cfg(feature = "rocm")]
    Rocm(Box<dyn QuantizedType>, crate::RocmDevice),
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
                GgmlDType::MXFP4 => metal::load_quantized(d, as_t_slice::<BlockMXFP4>(&data)),
                GgmlDType::BF16 => metal::load_quantized(d, as_t_slice::<bf16>(&data)),
                // IQ / ternary / 1-bit / NVFP4 codec types have no native Metal loader (CPU-decode only).
                other => crate::bail!("{other:?} is not supported on the Metal backend"),
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
                GgmlDType::MXFP4 => cuda::load_quantized(d, as_t_slice::<BlockMXFP4>(&data)),
                GgmlDType::BF16 => cuda::load_quantized(d, as_t_slice::<bf16>(&data)),
                // IQ / ternary / 1-bit / NVFP4 codec types: no native on-GPU quant-matmul kernel, but the
                // GGML blocks upload to VRAM byte-for-byte like any other quant. QCudaStorage::fwd then
                // dequantizes them to f32 (CPU codebook decode + upload) for a dense matmul -- see
                // `has_native_q8_1_matmul`. So an i-quant GGUF that used to bail here now LOADS and DECODES
                // on CUDA, exact w.r.t. the CPU reference; a native mmvq kernel is the bandwidth follow-up.
                // Exhaustive on purpose: a future GgmlDType must be wired here (fail-closed at compile) rather
                // than silently bailing at load.
                GgmlDType::IQ2_XXS => cuda::load_quantized(d, as_t_slice::<BlockIQ2xxs>(&data)),
                GgmlDType::IQ2_XS => cuda::load_quantized(d, as_t_slice::<BlockIQ2xs>(&data)),
                GgmlDType::IQ2_S => cuda::load_quantized(d, as_t_slice::<BlockIQ2s>(&data)),
                GgmlDType::IQ3_XXS => cuda::load_quantized(d, as_t_slice::<BlockIQ3xxs>(&data)),
                GgmlDType::IQ3_S => cuda::load_quantized(d, as_t_slice::<BlockIQ3s>(&data)),
                GgmlDType::IQ1_S => cuda::load_quantized(d, as_t_slice::<BlockIQ1s>(&data)),
                GgmlDType::IQ1_M => cuda::load_quantized(d, as_t_slice::<BlockIQ1m>(&data)),
                GgmlDType::TQ1_0 => cuda::load_quantized(d, as_t_slice::<BlockTQ1_0>(&data)),
                GgmlDType::TQ2_0 => cuda::load_quantized(d, as_t_slice::<BlockTQ2_0>(&data)),
                GgmlDType::NVFP4 => cuda::load_quantized(d, as_t_slice::<BlockNVFP4>(&data)),
                GgmlDType::Q1_0 => cuda::load_quantized(d, as_t_slice::<BlockQ1_0>(&data)),
            },
            #[cfg(feature = "rocm")]
            Device::Rocm(d) => Ok(Self::Rocm(dtype.from_data(data), d.clone())),
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
            #[cfg(feature = "rocm")]
            QStorage::Rocm(storage, _) => storage.block_size(),
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
            #[cfg(feature = "rocm")]
            QStorage::Rocm(storage, _) => storage.dtype(),
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
            #[cfg(feature = "rocm")]
            QStorage::Rocm(_storage, device) => Device::Rocm(device.clone()),
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
            #[cfg(feature = "rocm")]
            QStorage::Rocm(storage, _) => storage.storage_size_in_bytes(),
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
            #[cfg(feature = "rocm")]
            QStorage::Rocm(storage, device) => {
                // Dequantize on the CPU, then upload the dense f32 weights to the ROCm device.
                use crate::backend::BackendDevice;
                let cpu = storage.dequantize(elem_count)?;
                Ok(Storage::Rocm(device.storage_from_cpu_storage(&cpu)?))
            }
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
            #[cfg(feature = "rocm")]
            QStorage::Rocm(storage, _) => {
                let data_ptr = storage.as_ptr();
                let size_in_bytes = storage.storage_size_in_bytes();
                let data = unsafe { std::slice::from_raw_parts(data_ptr, size_in_bytes) };
                Ok(Cow::from(data))
            }
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
            #[cfg(feature = "rocm")]
            QStorage::Rocm(..) => crate::bail!("not implemented"),
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
    // 4-bit microscaling float (MXFP4), ggml type 39: 32 elems / block, E8M0 scale + 16 nibble-pairs.
    MXFP4,
    // IQ / ternary / 1-bit / NVFP4 codec types (decode-only; dequant->matmul on every backend).
    #[allow(non_camel_case_types)]
    IQ2_XXS,
    #[allow(non_camel_case_types)]
    IQ2_XS,
    #[allow(non_camel_case_types)]
    IQ3_XXS,
    #[allow(non_camel_case_types)]
    IQ1_S,
    #[allow(non_camel_case_types)]
    IQ3_S,
    #[allow(non_camel_case_types)]
    IQ2_S,
    #[allow(non_camel_case_types)]
    IQ1_M,
    TQ1_0,
    TQ2_0,
    NVFP4,
    Q1_0,
}

// --- Single-source-of-truth wiring (Cut 2) -----------------------------------------
//
// The five purely 1:1-per-block `GgmlDType` methods (`from_u32`, `to_u32`, `cpu_zeros`,
// `from_data`, `type_size`) are generated from the ONE `for_each_quant!` table in
// `quant_format.rs` instead of six hand-maintained match blocks. Each generator below is
// handed the whole `Variant => Block @ ggml_id` list and emits the block arms; the three
// non-block pseudo-types (F32/F16/BF16) that the table intentionally omits keep their
// bespoke arms inline. Behavior is byte-identical to the previous hand-written matches —
// the table's ids/blocks were verified equal to them. (`block_size` is left hand-written:
// the table carries no block-size data and its grouped form — F32=1, Q1_0/NVFP4 special,
// the big QK_K group — is already minimal.)
use crate::for_each_quant;

macro_rules! gen_from_u32 {
    ($($v:ident => $b:ident @ $id:literal),+ $(,)?) => {
        pub(crate) fn from_u32(u: u32) -> Result<Self> {
            let dtype = match u {
                0 => Self::F32,
                1 => Self::F16,
                30 => Self::BF16,
                $( $id => Self::$v, )+
                _ => crate::bail!("unknown dtype for tensor {u}"),
            };
            Ok(dtype)
        }
    };
}

macro_rules! gen_to_u32 {
    ($($v:ident => $b:ident @ $id:literal),+ $(,)?) => {
        /// GGML type id for this dtype. Single source of truth (generated from the
        /// `for_each_quant!` table); cross-crate callers (e.g. hanzo-quant UQFF
        /// serialization) use this instead of hand-rolled id maps that drift.
        pub fn to_u32(self) -> u32 {
            match self {
                Self::F32 => 0,
                Self::F16 => 1,
                Self::BF16 => 30,
                $( Self::$v => $id, )+
            }
        }
    };
}

macro_rules! gen_cpu_zeros {
    ($($v:ident => $b:ident @ $id:literal),+ $(,)?) => {
        /// The block dtype
        pub fn cpu_zeros(&self, elem_count: usize) -> Box<dyn QuantizedType> {
            match self {
                Self::F32 => Box::new(vec![f32::zeros(); elem_count]),
                Self::F16 => Box::new(vec![f16::zeros(); elem_count]),
                Self::BF16 => Box::new(vec![bf16::zeros(); elem_count]),
                $( Self::$v => Box::new(vec![<$b>::zeros(); elem_count / <$b>::BLCK_SIZE]), )+
            }
        }
    };
}

macro_rules! gen_from_data {
    ($($v:ident => $b:ident @ $id:literal),+ $(,)?) => {
        pub fn from_data(&self, data: Cow<'_, [u8]>) -> Box<dyn QuantizedType> {
            match self {
                Self::F32 => Box::new(as_t_slice::<f32>(&data).to_vec()),
                Self::F16 => Box::new(as_t_slice::<f16>(&data).to_vec()),
                Self::BF16 => Box::new(as_t_slice::<bf16>(&data).to_vec()),
                $( Self::$v => Box::new(as_t_slice::<$b>(&data).to_vec()), )+
            }
        }
    };
}

macro_rules! gen_type_size {
    ($($v:ident => $b:ident @ $id:literal),+ $(,)?) => {
        /// The type size for blocks in bytes.
        pub fn type_size(&self) -> usize {
            use k_quants::*;
            match self {
                Self::F32 => 4,
                Self::F16 | Self::BF16 => 2,
                $( Self::$v => std::mem::size_of::<$b>(), )+
            }
        }
    };
}

impl GgmlDType {
    for_each_quant!(gen_from_u32);
    for_each_quant!(gen_to_u32);
    for_each_quant!(gen_cpu_zeros);
    for_each_quant!(gen_from_data);
    for_each_quant!(gen_type_size);

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
            Self::MXFP4 => k_quants::QK_MXFP4,
            Self::Q1_0 => iq_quants::QK1_0,
            Self::NVFP4 => iq_quants::QK_NVFP4,
            Self::Q2K
            | Self::Q3K
            | Self::Q4K
            | Self::Q5K
            | Self::Q6K
            | Self::Q8K
            | Self::IQ4_XS
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ3_XXS
            | Self::IQ1_S
            | Self::IQ3_S
            | Self::IQ2_S
            | Self::IQ1_M
            | Self::TQ1_0
            | Self::TQ2_0 => k_quants::QK_K,
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

    // Upload this expert bank's GGML bytes to VRAM ONCE and keep it resident, keyed by the stable
    // (ptr,len) of the QTensor's CPU bytes. Re-uploading the multi-GB bank per token/layer would
    // dominate decode, so the cache makes MoE bandwidth-bound on the quant matvec, not the H2D copy.
    #[cfg(feature = "rocm")]
    fn rocm_moe_bank(
        &self,
        dev: &crate::RocmDevice,
    ) -> Result<std::sync::Arc<crate::RocmStorage>> {
        use crate::backend::BackendDevice;
        let bank = self.data()?;
        let key = (bank.as_ref().as_ptr() as usize, bank.as_ref().len());
        let cache = rocm_moe_bank_cache();
        let mut guard = cache.lock().expect("moe bank cache lock");
        if let Some(w) = guard.get(&key) {
            Ok(w.clone())
        } else {
            let w = std::sync::Arc::new(dev.storage_from_slice(bank.as_ref())?);
            guard.insert(key, w.clone());
            Ok(w)
        }
    }

    pub fn indexed_moe_forward(&self, x: &Tensor, ids: &Tensor) -> Result<Tensor> {
        match &self.storage {
            // Only dtypes with a fused CUDA indexed-MoE kernel take the fast path; others (e.g. MXFP4,
            // i-quant/ternary) fall through to the generic per-expert path below, which dequantizes via
            // QMatMul. The supported set is derived from the ONE kernel-name table (cuda.rs), so this gate
            // can't drift from the kernels that actually exist -- which is exactly what had stranded
            // Q4_0/Q4_1/Q5_0/Q5_1 on the CPU even though their fused kernels are now compiled.
            QStorage::Cuda(s) if cuda::QCudaStorage::supports_indexed_moe(s.dtype()) =>
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
            // Native ROCm MoE: the GGML expert bank [E,n,k] is uploaded once and each routed slot
            // is dispatched through the SAME unified qmatvec_core<WTYPE> as ordinary decode (no
            // MoE-per-quant kernel; works for every wired quant). Avoids ROCm's missing index_add:
            // each routed slot writes exactly one output row, placed directly by slot index.
            #[cfg(feature = "rocm")]
            QStorage::Rocm(_, rocm_dev)
                if crate::RocmQuantType::from_ggml(self.storage.dtype()).is_some() =>
            {
                let qt = crate::RocmQuantType::from_ggml(self.storage.dtype()).unwrap();
                let out_dtype = x.dtype();
                // e_cnt is not read: ids stay on-device and the router guarantees the bound, so the
                // old host bounds check (and its DtoH `to_vec1`) is gone.
                let (_e_cnt, n, k) = self.shape().dims3()?;
                let (t, topk) = ids.dims2()?;
                let s = x.dim(1)?; // 1 (gate/up: shared input) or topk (down: per-slot)
                let x_exp = if s == topk {
                    x.clone()
                } else {
                    x.broadcast_as((t, topk, k))?
                };
                let nrows = t * topk;
                // PREFILL (t>1) routes to the fused expert-grouped WMMA GEMM (f16 activations); DECODE
                // (t==1) keeps the model's native bf16/f16 on the capture-clean matvec. See the twin in
                // QStorage::indexed_moe_forward. HANZO_MOE_QMMQ_FALLBACK forces matvec for the A/B.
                // Decode-only types (no qmmq kernel) ride the per-slot matvec core for prefill too
                // (correct at any token count). ONE predicate gates every prefill site.
                let use_qmmq =
                    t > 1 && qt.qmmq_capable() && std::env::var("HANZO_MOE_QMMQ_FALLBACK").is_err();
                let x_flat = match x_exp.dtype() {
                    // qmmq quantizes f16/f32 activations natively, so keep the model's dtype and skip
                    // the f32->f16 cast (a 16.7M-elem read+write per gate/up). Other dtypes (bf16 with
                    // a symmetric expert type) still cast to f16.
                    DType::F16 | DType::F32 if use_qmmq => x_exp.reshape((nrows, k))?.contiguous()?,
                    _ if use_qmmq => x_exp.reshape((nrows, k))?.to_dtype(DType::F16)?.contiguous()?,
                    DType::BF16 | DType::F16 => x_exp.reshape((nrows, k))?.contiguous()?,
                    // DECODE f32-native: dp4a experts quantize q8_1 from f32 and store f32, so an F32
                    // routed activation stays F32 (the matvec returns F32 -> the .to_dtype(out_dtype)
                    // below is a no-op), removing the cast pair that wrapped each gate/up/down matvec.
                    DType::F32 if qt.dp4a_active() => x_exp.reshape((nrows, k))?.contiguous()?,
                    _ => x_exp.reshape((nrows, k))?.to_dtype(DType::F16)?.contiguous()?,
                };
                let wbank = self.rocm_moe_bank(rocm_dev)?;
                // Keep router ids ON the GPU for EVERY wired quant type and run ONE batched launch
                // (experts on grid.y, ids read on-device). No `to_vec1` DtoH sync -- that host round-
                // trip (3 per layer x 48 layers per token) was both the dominant WSL decode stall AND
                // what made HIP stream capture illegal (hipErrorStreamCaptureImplicit -> the graph-path
                // SIGSEGV). The router emits a top-k over the e_cnt expert logits, so 0 <= id < e_cnt
                // by construction; the prior host bounds check is dropped to stay capture-clean.
                let ids_u32 = ids
                    .reshape((nrows,))?
                    .to_dtype(crate::DType::U32)?
                    .contiguous()?;
                let (store, _) = x_flat.storage_and_layout();
                let xr = match &*store {
                    crate::Storage::Rocm(r) => r,
                    _ => crate::bail!("rocm MoE: x not on rocm after contiguous()"),
                };
                let (idstore, _) = ids_u32.storage_and_layout();
                let idr = match &*idstore {
                    crate::Storage::Rocm(r) => r,
                    _ => crate::bail!("rocm MoE: ids not on rocm"),
                };
                let y = if use_qmmq {
                    rocm_dev.moe_qmmq_quant(qt, wbank.as_ref(), xr, idr, nrows, n, k)?
                } else {
                    rocm_dev.moe_matvec_quant(qt, wbank.as_ref(), xr, idr, nrows, n, k)?
                };
                let out = crate::tensor::from_storage(
                    crate::Storage::Rocm(y),
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
            #[cfg(feature = "rocm")]
            QStorage::Rocm(..) => crate::bail!("not implemented"),
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
    // Native ROCm quantized weight: the GGML blocks live in VRAM. Decode (1 row) runs the ONE
    // unified on-GPU quant matvec (qmatvec_core<WTYPE>) straight out of the block format; prefill
    // (>1 row) runs the ONE unified int8 WMMA GEMM (qmmq_core<WTYPE>) -- both for the full wired
    // spread (Q8_0/Q4_0/Q4_K/Q6_K/IQ4_XS/TQ2_0). Unwired types dequantize to a temporary f16
    // weight (RDNA matrix-core matmul). `n`/`k` are the weight dims.
    #[cfg(feature = "rocm")]
    RocmQuant {
        qtensor: std::sync::Arc<QTensor>,
        wq: std::sync::Arc<crate::RocmStorage>,
        dtype: GgmlDType,
        n: usize,
        k: usize,
    },
}

// Resident ROCm MoE expert-bank cache: maps a (CPU bank ptr, len) to its uploaded VRAM copy so
// the multi-GB GGML expert bank is host->device copied ONCE, not per token/layer. The CPU bytes
// are owned by the model's QTensor for its lifetime, so the pointer is a stable key. Keyed by
// usize (raw ptr) to stay Send+Sync; the RocmStorage is reference-counted and reused.
#[cfg(feature = "rocm")]
fn rocm_moe_bank_cache(
) -> &'static std::sync::Mutex<std::collections::HashMap<(usize, usize), std::sync::Arc<crate::RocmStorage>>>
{
    static CACHE: std::sync::OnceLock<
        std::sync::Mutex<std::collections::HashMap<(usize, usize), std::sync::Arc<crate::RocmStorage>>>,
    > = std::sync::OnceLock::new();
    CACHE.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// FUSED MoE expert-combine: `out[i,j] = sum_e scores[i,e] * ys[i,e,j]`, reducing the per-expert
/// outputs `ys` [t, topk, n] by the router weights `scores` [t, topk] into [t, n]. On ROCm this is
/// ONE fused kernel (`RocmDevice::moe_combine`) instead of `ys.broadcast_mul(scores).sum(Minus2)`,
/// which cast ys -> f32, wrote a [t,topk,n] f32 product temp, and ran an 8-wide strided reduce.
/// Other backends keep the generic broadcast-mul + sum.
pub fn moe_combine(ys: &Tensor, scores: &Tensor) -> Result<Tensor> {
    let (t, topk, n) = ys.dims3()?;
    #[cfg(feature = "rocm")]
    if let Device::Rocm(dev) = ys.device() {
        if std::env::var("HANZO_MOE_COMBINE_FALLBACK").is_ok() {
            return ys.broadcast_mul(&scores.unsqueeze(D::Minus1)?)?.sum(D::Minus2);
        }
        let ys_c = ys.contiguous()?;
        let scores_c = scores.to_dtype(DType::F32)?.contiguous()?;
        let (ys_store, _) = ys_c.storage_and_layout();
        let yr = match &*ys_store {
            Storage::Rocm(r) => r,
            _ => crate::bail!("moe_combine: ys not on rocm after contiguous()"),
        };
        let (sc_store, _) = scores_c.storage_and_layout();
        let sr = match &*sc_store {
            Storage::Rocm(r) => r,
            _ => crate::bail!("moe_combine: scores not on rocm after contiguous()"),
        };
        let out = dev.moe_combine(yr, sr, t, topk, n)?;
        return Ok(crate::tensor::from_storage(
            Storage::Rocm(out),
            (t, n),
            crate::op::BackpropOp::none(),
            false,
        ));
    }
    ys.broadcast_mul(&scores.unsqueeze(D::Minus1)?)?.sum(D::Minus2)
}

/// Fused MoE router. Reduces the F32 router logits [ntok, n_experts] to the topk selected expert
/// ids [ntok, topk] (descending logit) and their softmax weights [ntok, topk]; `norm` renormalizes
/// the topk weights to sum 1 (norm_topk_prob). On ROCm this is ONE `moe_route` kernel replacing the
/// softmax->sort->narrow->sum->div chain (~6 launches/layer); elsewhere it is that chain via ml ops.
pub fn moe_route(logits: &Tensor, topk: usize, norm: bool) -> Result<(Tensor, Tensor)> {
    let (ntok, n_experts) = logits.dims2()?;
    #[cfg(feature = "rocm")]
    if let Device::Rocm(dev) = logits.device() {
        if std::env::var("HANZO_MOE_ROUTE_FALLBACK").is_err() {
            let logits_c = logits.to_dtype(DType::F32)?.contiguous()?;
            let (lg_store, _) = logits_c.storage_and_layout();
            let lr = match &*lg_store {
                Storage::Rocm(r) => r,
                _ => crate::bail!("moe_route: logits not on rocm after contiguous()"),
            };
            let (ids, w) = dev.moe_route(lr, ntok, n_experts, topk, norm)?;
            let ids_t = crate::tensor::from_storage(
                Storage::Rocm(ids),
                (ntok, topk),
                crate::op::BackpropOp::none(),
                false,
            );
            let w_t = crate::tensor::from_storage(
                Storage::Rocm(w),
                (ntok, topk),
                crate::op::BackpropOp::none(),
                false,
            );
            return Ok((ids_t, w_t));
        }
    }
    let lf = logits.to_dtype(DType::F32)?;
    let mx = lf.max_keepdim(D::Minus1)?;
    let e = lf.broadcast_sub(&mx)?.exp()?;
    let z = e.sum_keepdim(D::Minus1)?;
    let p = e.broadcast_div(&z)?;
    let (sv, si) = p.sort_last_dim(false)?;
    let ids = si.narrow(D::Minus1, 0, topk)?.contiguous()?;
    let mut w = sv.narrow(D::Minus1, 0, topk)?.contiguous()?;
    if norm {
        w = w.broadcast_div(&w.sum_keepdim(D::Minus1)?)?;
    }
    Ok((ids, w))
}

/// Fused MoE gate+up projections. Both expert banks consume the SAME routed token `x` [t,1,k], so the
/// shared input is broadcast + quantized ONCE and matvec'd against both banks (vs once per bank,
/// re-materializing + re-quantizing the identical activation). Returns the raw (gate_out, up_out)
/// [t,topk,n]; the caller applies silu(gate)*up. Each output is bit-identical to the unfused
/// `indexed_moe_forward` because the q8_1 activation is deterministic in `x`. ROCm decode/matvec path
/// only (the prefill qmmq path keeps its own per-bank quantize); every other case runs the two
/// unfused forwards. HANZO_MOE_GATEUP_FALLBACK forces the unfused path (the A/B + equivalence lever).
pub fn moe_gate_up(
    x: &Tensor,
    ids: &Tensor,
    gate: &QMatMul,
    up: &QMatMul,
) -> Result<(Tensor, Tensor)> {
    #[cfg(feature = "rocm")]
    if std::env::var("HANZO_MOE_GATEUP_FALLBACK").is_err() {
        if let (QMatMul::QTensor(gq), QMatMul::QTensor(uq)) = (gate, up) {
            if let (QStorage::Rocm(_, dev), QStorage::Rocm(..)) = (&gq.storage, &uq.storage) {
                let dt = gq.storage.dtype();
                if dt == uq.storage.dtype() {
                    if let Some(qt) = crate::RocmQuantType::from_ggml(dt) {
                        let (_e, n, k) = gq.shape().dims3()?;
                        let (t, topk) = ids.dims2()?;
                        let use_qmmq = t > 1
                            && qt.qmmq_capable()
                            && std::env::var("HANZO_MOE_QMMQ_FALLBACK").is_err();
                        if x.dim(1)? == 1 && !use_qmmq {
                            let nrows = t * topk;
                            let x_exp = x.broadcast_as((t, topk, k))?;
                            let x_flat = match x_exp.dtype() {
                                DType::BF16 | DType::F16 => x_exp.reshape((nrows, k))?.contiguous()?,
                                DType::F32 if qt.dp4a_active() => {
                                    x_exp.reshape((nrows, k))?.contiguous()?
                                }
                                _ => {
                                    x_exp.reshape((nrows, k))?.to_dtype(DType::F16)?.contiguous()?
                                }
                            };
                            let out_dtype = x.dtype();
                            let ids_u32 =
                                ids.reshape((nrows,))?.to_dtype(DType::U32)?.contiguous()?;
                            let gwb = gq.rocm_moe_bank(dev)?;
                            let uwb = uq.rocm_moe_bank(dev)?;
                            let (xstore, _) = x_flat.storage_and_layout();
                            let xr = match &*xstore {
                                Storage::Rocm(r) => r,
                                _ => crate::bail!("moe_gate_up: x not on rocm after contiguous()"),
                            };
                            let (idstore, _) = ids_u32.storage_and_layout();
                            let idr = match &*idstore {
                                Storage::Rocm(r) => r,
                                _ => crate::bail!("moe_gate_up: ids not on rocm"),
                            };
                            let (gy, uy) = dev.moe_matvec_pair(
                                qt,
                                gwb.as_ref(),
                                uwb.as_ref(),
                                xr,
                                idr,
                                nrows,
                                n,
                                k,
                            )?;
                            let g = crate::tensor::from_storage(
                                Storage::Rocm(gy),
                                (nrows, n),
                                crate::op::BackpropOp::none(),
                                false,
                            )
                            .reshape((t, topk, n))?
                            .to_dtype(out_dtype)?;
                            let u = crate::tensor::from_storage(
                                Storage::Rocm(uy),
                                (nrows, n),
                                crate::op::BackpropOp::none(),
                                false,
                            )
                            .reshape((t, topk, n))?
                            .to_dtype(out_dtype)?;
                            return Ok((g, u));
                        }
                    }
                }
            }
        }
    }
    Ok((gate.indexed_moe_forward(x, ids)?, up.indexed_moe_forward(x, ids)?))
}

thread_local! {
    static DEQUANTIZE_ALL: bool = {
        match std::env::var("DEQUANTIZE_ALL") {
            Ok(s) => {
                !s.is_empty() && s != "0"
            },
            Err(_) => false,
        }
    }
}

thread_local! {
    static DEQUANTIZE_ALL_F16: bool = {
        match std::env::var("DEQUANTIZE_ALL_F16") {
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
        // Native ROCm quantized path: keep the GGML blocks in VRAM and run the ONE unified on-GPU
        // quant decode core (qmatvec_core<WTYPE>; Q8_0+Q4_0 also have the int8 WMMA prefill gemm),
        // instead of dequantizing the whole model to dense f16 (2x+ the decode bandwidth). Reads the
        // block format straight from the uploaded bytes -- exact w.r.t. the CPU reference. The wired
        // set is exactly RocmQuantType::from_ggml; k must be a multiple of that type's block size.
        #[cfg(feature = "rocm")]
        {
            let dt = qtensor.dtype();
            // Decode AND prefill native iff the unified core has this type wired (one enum row in
            // RocmQuantType): decode rides qmatvec_core<WTYPE>, prefill (rows>1) rides the int8 WMMA
            // qmmq_core<WTYPE>. Both cover the SAME wired spread; adding a type is one enum row + the
            // in-kernel decode, no per-quant kernel. Unwired types dequantize-to-f16 in forward().
            if let Some(qt) = crate::RocmQuantType::from_ggml(dt) {
                if let Device::Rocm(d) = qtensor.device() {
                    if let Ok((n, k)) = qtensor.shape().dims2() {
                        let blk_ok = k % qt.block_elems() == 0;
                        if blk_ok {
                            use crate::backend::BackendDevice;
                            let bytes = qtensor.data()?;
                            let wq = d.storage_from_slice(bytes.as_ref())?;
                            return Ok(Self::RocmQuant {
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
        // ROCm MoE bank: a 3D [E,n,k] expert bank of a wired quant type stays QUANTIZED (kept as
        // QTensor), so `indexed_moe_forward` runs each routed expert through the ONE unified
        // qmatvec_core (no per-expert kernel) instead of dequantizing the whole bank to dense f16
        // (which for a 30B-A3B model is many GB of resident f16 AND has no indexed_moe path). The 2D
        // RocmQuant decode/prefill path above already handles ordinary weights; this is the MoE case.
        #[cfg(feature = "rocm")]
        {
            if qtensor.device().is_rocm()
                && qtensor.shape().dims().len() == 3
                && crate::RocmQuantType::from_ggml(qtensor.dtype()).is_some()
                && !DEQUANTIZE_ALL.with(|b| *b)
            {
                return Ok(Self::QTensor(qtensor));
            }
        }
        let dequantize = match qtensor.dtype() {
            GgmlDType::F32 | GgmlDType::F16 | GgmlDType::BF16 => true,
            // The Vulkan/wgpu/ROCm backends have no generic native quantized matmul, so dequantize
            // to f32 here (once, at construction) and run the regular f32 GPU matmul.
            _ => {
                DEQUANTIZE_ALL.with(|b| *b)
                    || qtensor.device().is_vulkan()
                    || qtensor.device().is_wgpu()
                    || qtensor.device().is_rocm()
            }
        };
        let t = if dequantize {
            // ROCm: dequantize to f16 so the matmul hits RDNA3.5 matrix cores (WMMA). Dense f32
            // (sgemm) has no matrix-core path on RDNA and runs ~an order of magnitude slower, and
            // f16 also halves the resident weight memory.
            if qtensor.device().is_rocm() {
                Self::TensorF16(qtensor.dequantize_f16(&qtensor.device())?)
            } else {
                Self::Tensor(qtensor.dequantize(&qtensor.device())?)
            }
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
            #[cfg(feature = "rocm")]
            Self::RocmQuant { qtensor, .. } => qtensor.dequantize_f16(&qtensor.device()),
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
            // Resident-bank MoE: `wq` already holds the [E,n,k] GGML blocks in VRAM (uploaded at
            // load), so route straight to the batched on-GPU quant matvec. Delegating to `qtensor`
            // (CPU-side) instead drops to the generic fallback that re-uploads every routed expert
            // every token -- the 20-50x decode cliff. `qtensor` is read only for its shape.
            #[cfg(feature = "rocm")]
            Self::RocmQuant {
                qtensor, wq, dtype, ..
            } if crate::RocmQuantType::from_ggml(*dtype).is_some() => {
                let qt = crate::RocmQuantType::from_ggml(*dtype).unwrap();
                let wbank = wq.as_ref();
                // e_cnt unused: ids stay on-device, router guarantees the bound (no host check).
                let (_e_cnt, n, k) = qtensor.shape().dims3()?;
                let (t, topk) = ids.dims2()?;
                let s = x.dim(1)?; // 1 (gate/up: shared input) or topk (down: per-slot)
                let x_exp = if s == topk {
                    x.clone()
                } else {
                    x.broadcast_as((t, topk, k))?
                };
                let nrows = t * topk;
                // PREFILL (t>1, never graph-captured) routes to the FUSED expert-grouped WMMA GEMM
                // (`moe_qmmq_quant`), which needs f16 activations; DECODE (t==1) stays on the
                // capture-clean dp4a/scalar matvec, which takes bf16/f16 natively. HANZO_MOE_QMMQ_FALLBACK
                // forces the matvec on prefill too (the before/after A/B + oracle-equivalence lever).
                // Decode-only types (no qmmq kernel) ride the per-slot matvec core for prefill too
                // (correct at any token count). ONE predicate gates every prefill site.
                let use_qmmq =
                    t > 1 && qt.qmmq_capable() && std::env::var("HANZO_MOE_QMMQ_FALLBACK").is_err();
                let x_flat = match x_exp.dtype() {
                    // qmmq quantizes f16/f32 activations natively, so keep the model's dtype and skip
                    // the f32->f16 cast (a 16.7M-elem read+write per gate/up). Other dtypes (bf16 with
                    // a symmetric expert type) still cast to f16.
                    DType::F16 | DType::F32 if use_qmmq => x_exp.reshape((nrows, k))?.contiguous()?,
                    _ if use_qmmq => x_exp.reshape((nrows, k))?.to_dtype(DType::F16)?.contiguous()?,
                    DType::BF16 | DType::F16 => x_exp.reshape((nrows, k))?.contiguous()?,
                    // DECODE f32-native dp4a: keep F32 routed activation F32 end-to-end (matvec stores
                    // F32), eliding the cast pair around each expert matvec. See QStorage twin above.
                    DType::F32 if qt.dp4a_active() => x_exp.reshape((nrows, k))?.contiguous()?,
                    _ => x_exp.reshape((nrows, k))?.to_dtype(DType::F16)?.contiguous()?,
                };
                let out_dtype = x.dtype();
                // Keep router ids ON the GPU for EVERY wired quant type: the batched kernels index
                // experts on-device, so there is no per-call `to_vec1` host round-trip. That DtoH
                // sync (3 per layer x 48 layers per token) was both the dominant decode stall on WSL
                // AND the HIP-graph capture breaker. Router top-k guarantees 0 <= id < e_cnt.
                let ids_u32 = ids
                    .reshape((nrows,))?
                    .to_dtype(crate::DType::U32)?
                    .contiguous()?;
                let (xstore, _) = x_flat.storage_and_layout();
                let xr = match &*xstore {
                    crate::Storage::Rocm(r) => r,
                    _ => crate::bail!("rocm MoE: x not on rocm after contiguous()"),
                };
                let (idstore, _) = ids_u32.storage_and_layout();
                let idr = match &*idstore {
                    crate::Storage::Rocm(r) => r,
                    _ => crate::bail!("rocm MoE: ids not on rocm"),
                };
                let y = if use_qmmq {
                    wbank.device.moe_qmmq_quant(qt, wbank, xr, idr, nrows, n, k)?
                } else {
                    wbank.device.moe_matvec_quant(qt, wbank, xr, idr, nrows, n, k)?
                };
                let out = crate::tensor::from_storage(
                    crate::Storage::Rocm(y),
                    (nrows, n),
                    crate::op::BackpropOp::none(),
                    false,
                );
                out.reshape((t, topk, n))?.to_dtype(out_dtype)
            }
            // Unwired ROCm quant dtypes (no on-GPU quant matvec): CPU per-expert fallback.
            #[cfg(feature = "rocm")]
            Self::RocmQuant { qtensor, .. } => qtensor.indexed_moe_forward(x, ids),
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
            #[cfg(feature = "rocm")]
            QStorage::Rocm(..) => crate::bail!("Invalid storage"),
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

/// Dense (non-quantized) matmul `xs @ w^T` for the `Tensor`/`TensorF16` `QMatMul` variants, where
/// the stored weight `w` is `[n, k]` and `xs` is `[.., k]`. On ROCm at decode (a single-row matvec)
/// this computes the result as `sum_k(xs[k] * w[n, k])` via pooled broadcast-mul + reduce instead of
/// rocBLAS `gemm_ex`. rocBLAS's GEMM dispatch records a vendor-specific PM4 indirect-buffer packet
/// that WSL's HSA thunk rejects on hipGraph replay (`VendorSpecificAqlToPm4` assert), so a captured
/// decode forward containing one (e.g. the MoE F32 router gate) corrupts/aborts on replay. The
/// reduce path uses only ops already exercised under capture (RMSNorm etc.), so it replays cleanly,
/// and at M=1 a GEMV-as-reduce is as cheap as the GEMM (it materializes only the `[n, k]` weight).
/// Prefill (rows > 1, never graph-captured) keeps the rocBLAS GEMM. Non-ROCm devices are unchanged.
fn dense_matmul(xs: &Tensor, w: &Tensor) -> Result<Tensor> {
    let k = *w.dims().last().unwrap();
    let rows = xs.elem_count() / k;
    if rows == 1 && xs.device().is_rocm() {
        let n = w.dim(0)?;
        #[cfg(feature = "rocm")]
        {
            // Dense decode GEMV: read the [n,k] weight ONCE (warp/row dot) instead of materializing
            // and re-reading the broadcast_mul product. The activation is matched to the weight dtype
            // (a [k] cast, negligible); the GEMV stays capture-clean (no rocBLAS).
            let d = match xs.device() {
                Device::Rocm(d) => d.clone(),
                _ => unreachable!(),
            };
            let xs1 = xs.reshape((k,))?.to_dtype(w.dtype())?.contiguous()?;
            let w = w.contiguous()?;
            let (wstore, _) = w.storage_and_layout();
            let wr = match &*wstore {
                crate::Storage::Rocm(r) => r,
                _ => crate::bail!("dense_matmul: weight not on rocm"),
            };
            let (xstore, _) = xs1.storage_and_layout();
            let xr = match &*xstore {
                crate::Storage::Rocm(r) => r,
                _ => crate::bail!("dense_matmul: x not on rocm"),
            };
            let y = d.dense_gemv(wr, xr, n, k)?;
            let mut dims = xs.dims().to_vec();
            *dims.last_mut().unwrap() = n;
            return crate::tensor::from_storage(
                crate::Storage::Rocm(y),
                dims,
                crate::op::BackpropOp::none(),
                false,
            )
            .to_dtype(xs.dtype());
        }
        #[cfg(not(feature = "rocm"))]
        {
            let out = xs.reshape((1, k))?.broadcast_mul(w)?.sum(D::Minus1)?;
            let mut dims = xs.dims().to_vec();
            *dims.last_mut().unwrap() = n;
            return out.reshape(dims);
        }
    }
    let w = match *xs.dims() {
        [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
        [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
        _ => w.t()?,
    };
    xs.matmul(&w)
}

impl crate::Module for QMatMul {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            #[cfg(feature = "rocm")]
            Self::RocmQuant {
                qtensor,
                wq,
                dtype,
                n,
                k,
            } => {
                // Device-residency guard: multi-token attention prefill can leave the activation
                // off-device (an upstream op leaked to host); recover instead of bailing so prefill
                // stays correct. The leak is the bug to fix for speed; this keeps us correct meanwhile.
                let xs_recovered = if xs.device().is_rocm() {
                    None
                } else {
                    if std::env::var("HANZO_DBG_PATH").is_ok() {
                        eprintln!(
                            "[ROCm-RECOVER] activation off-device {:?} dims {:?}",
                            xs.device().location(),
                            xs.dims()
                        );
                    }
                    Some(xs.to_device(&qtensor.device())?)
                };
                let xs = xs_recovered.as_ref().unwrap_or(xs);
                let rows: usize = xs.elem_count() / *k;
                #[cfg(feature = "rocm")]
                {
                    use std::sync::atomic::{AtomicUsize, Ordering};
                    static DBG_DECODE: AtomicUsize = AtomicUsize::new(0);
                    static DBG_PREFILL: AtomicUsize = AtomicUsize::new(0);
                    static DBG_FALLBACK: AtomicUsize = AtomicUsize::new(0);
                    if std::env::var("HANZO_DBG_PATH").is_ok() {
                        // Buckets MATCH the real dispatch below: decode = rows==1 wired (the dp4a-vs-
                        // scalar A/B is internal to matvec_quant, still native); prefill = rows>1 wired
                        // native qmmq_core<WTYPE>; fallback = unwired OR HANZO_QMMQ_FALLBACK dequantize.
                        let qt = crate::RocmQuantType::from_ggml(*dtype);
                        let decode_wired = qt.is_some();
                        let prefill_wired = qt.is_some_and(|qt| qt.qmmq_capable());
                        let prefill_fb = std::env::var("HANZO_QMMQ_FALLBACK").is_ok();
                        if rows == 1 && decode_wired {
                            DBG_DECODE.fetch_add(1, Ordering::Relaxed);
                        } else if rows > 1 && prefill_wired && !prefill_fb {
                            DBG_PREFILL.fetch_add(1, Ordering::Relaxed);
                        } else {
                            DBG_FALLBACK.fetch_add(1, Ordering::Relaxed);
                        }
                        // Print on EVERY rows>1 (prefill is rare vs decode) + every 200 total, so the
                        // native-prefill path is always visible the moment it is taken.
                        let tot = DBG_DECODE.load(Ordering::Relaxed)
                            + DBG_PREFILL.load(Ordering::Relaxed)
                            + DBG_FALLBACK.load(Ordering::Relaxed);
                        if rows > 1 || tot % 200 == 0 {
                            eprintln!(
                                "[DBG_PATH] decode={} prefill(native)={} fallback={} (dt={:?} rows={} n={} k={})",
                                DBG_DECODE.load(Ordering::Relaxed),
                                DBG_PREFILL.load(Ordering::Relaxed),
                                DBG_FALLBACK.load(Ordering::Relaxed),
                                dtype, rows, *n, *k
                            );
                        }
                    }
                }
                // Table-driven unified decode: a type is decode-native iff the single
                // `qmatvec_core<WTYPE>` has a `decode_block` wired for it (RocmQuantType::from_ggml).
                // Q8_0/Q4_0/Q4_K/Q6_K/IQ4_XS/TQ2_0 today; adding a type is one enum row, no kernel.
                // The dp4a-vs-scalar A/B for dp4a-capable types (HANZO_Q4K_FALLBACK / HANZO_Q6K_FALLBACK)
                // lives entirely inside `matvec_quant` (dp4a_active) -- ONE fallback, one place; it
                // switches the decode core, the type stays on the native path either way.
                #[cfg(feature = "rocm")]
                let unified_qt = crate::RocmQuantType::from_ggml(*dtype);
                #[cfg(not(feature = "rocm"))]
                let unified_qt: Option<()> = None;
                // Native int8-WMMA prefill exists only for `qmmq_capable` types; the decode-only types
                // (Q2_K/Q3_K + every IQ*/TQ* codebook/fractional type) dequantize-to-f16 for rows>1 via
                // the `else` branch below -- correct, just not WMMA-accelerated. ONE predicate, read here
                // and at the two MoE `use_qmmq` sites.
                #[cfg(feature = "rocm")]
                let qmmq_ok = unified_qt.map(|qt| qt.qmmq_capable()).unwrap_or(false);
                #[cfg(not(feature = "rocm"))]
                let qmmq_ok = false;
                if rows == 1 && unified_qt.is_some() {
                    // Decode: weights stay quantized in VRAM; the ONE native on-GPU quant matvec core
                    // dequantizes per-block on-the-fly (no dense f16 copy). The matvec consumes
                    // bf16/f16 activations directly and returns the same dtype, so the model's working
                    // dtype (bf16) is kept end-to-end -- no bf16->f32->f16->bf16 cast detour. Only fall
                    // back to an f16 cast for exotic input dtypes. Every wired type (symmetric 8-bit
                    // through asymmetric super-block through sub-4-bit ternary) rides the same core.
                    // dp4a-capable types accept the F32 residual/norm stream DIRECTLY (q8_1 quantize
                    // from f32 + f32-store matvec), so an F32 activation stays F32 end-to-end with no
                    // f16 bounce -- this removes the cast_f32_f16-before / cast_f16_f32-after pair that
                    // wrapped every decode matvec. Non-dp4a (scalar) types keep the f16 cast.
                    #[cfg(feature = "rocm")]
                    let keep_f32 = unified_qt.map(|qt| qt.dp4a_active()).unwrap_or(false);
                    #[cfg(not(feature = "rocm"))]
                    let keep_f32 = false;
                    let xs = match xs.dtype() {
                        DType::BF16 | DType::F16 => xs.contiguous()?,
                        DType::F32 if keep_f32 => xs.contiguous()?,
                        _ => xs.to_dtype(DType::F16)?.contiguous()?,
                    };
                    let d = match xs.device() {
                        Device::Rocm(d) => d,
                        _ => crate::bail!("RocmQuant input not on rocm"),
                    };
                    let y = {
                        let (store, _) = xs.storage_and_layout();
                        let xr = match &*store {
                            crate::Storage::Rocm(r) => r,
                            _ => crate::bail!("RocmQuant expected rocm storage"),
                        };
                        #[cfg(feature = "rocm")]
                        {
                            d.matvec_quant(unified_qt.unwrap(), wq, xr, *n, *k)?
                        }
                        #[cfg(not(feature = "rocm"))]
                        {
                            crate::bail!("rocm feature disabled")
                        }
                    };
                    let mut dims = xs.dims().to_vec();
                    let last = dims.len() - 1;
                    dims[last] = *n;
                    Ok(crate::tensor::from_storage(
                        crate::Storage::Rocm(y),
                        dims,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                } else if let Some(qt) =
                    unified_qt.filter(|_| qmmq_ok && std::env::var("HANZO_QMMQ_FALLBACK").is_err())
                {
                    // Prefill (rows>1): native int8 WMMA gemm through the ONE unified core
                    // (`qmmq_core<WTYPE>` in quant.hip). Weights stay quantized in VRAM (no resident
                    // dense f16, which would slow the memory-bound decode) and the MAC runs on the
                    // RDNA3 int8 matrix cores instead of rocBLAS. The SAME core covers the whole wired
                    // spread: Q8_0/Q4_0 (symmetric, proven), Q4_K (asymmetric -- min bias via the
                    // q8_1 block-sum), and the symmetric super-block / IQ / ternary types (Q6_K,
                    // IQ4_XS, TQ2_0). Selecting the type is one `RocmQuantType` row + the in-kernel
                    // decode; there is NO per-quant prefill kernel. HANZO_QMMQ_FALLBACK=1 forces the
                    // dequant-f16 matmul below (the prefill before/after A/B benchmark lever).
                    let xs = xs.to_dtype(DType::F16)?.contiguous()?;
                    let d = match xs.device() {
                        Device::Rocm(d) => d,
                        _ => crate::bail!("RocmQuant input not on rocm"),
                    };
                    let m = xs.elem_count() / *k;
                    let y = {
                        let (store, _) = xs.storage_and_layout();
                        let xr = match &*store {
                            crate::Storage::Rocm(r) => r,
                            _ => crate::bail!("RocmQuant expected rocm storage"),
                        };
                        #[cfg(feature = "rocm")]
                        {
                            d.qmmq_quant(qt, xr, wq, m, *n, *k)?
                        }
                        #[cfg(not(feature = "rocm"))]
                        {
                            let _ = qt;
                            crate::bail!("rocm feature disabled")
                        }
                    };
                    let mut dims = xs.dims().to_vec();
                    let last = dims.len() - 1;
                    dims[last] = *n;
                    Ok(crate::tensor::from_storage(
                        crate::Storage::Rocm(y),
                        dims,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                } else {
                    // Unwired-type / forced-fallback prefill: dequantize to a temporary f16 weight
                    // (freed after; a persistent f16 copy would slow the memory-bound decode). Only
                    // reached for quants with no `qmmq_core<WTYPE>` wired (e.g. Q5_K/MXFP4) or when
                    // HANZO_QMMQ_FALLBACK forces it for the prefill A/B measurement.
                    let w = qtensor.dequantize_f16(&xs.device())?;
                    let w = match *xs.dims() {
                        [b1, b2, _, _] => w.broadcast_left((b1, b2))?.t()?,
                        [bsize, _, _] => w.broadcast_left(bsize)?.t()?,
                        _ => w.t()?,
                    };
                    xs.to_dtype(DType::F16)?.matmul(&w)
                }
            }
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
            Self::Tensor(w) => dense_matmul(xs, w),
            Self::TensorF16(w) => {
                let in_dtype = xs.dtype();
                dense_matmul(&xs.to_dtype(DType::F16)?, w)?.to_dtype(in_dtype)
            }
        }
    }
}
