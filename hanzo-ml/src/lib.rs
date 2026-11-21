//! Tensors, autograd and quantization for Rust, on the CPU, CUDA, ROCm, Metal, Vulkan and wgpu.
//!
//! ```rust
//! use hanzo_ml::{Tensor, DType, Device};
//! # use hanzo_ml::Error;
//! # fn main() -> Result<(), Error>{
//!
//! let a = Tensor::arange(0f32, 6f32, &Device::Cpu)?.reshape((2, 3))?;
//! let b = Tensor::arange(0f32, 12f32, &Device::Cpu)?.reshape((3, 4))?;
//! let c = a.matmul(&b)?;
//!
//! # Ok(())}
//! ```
//!
//! - `Tensor` and `Device`: n-dimensional arrays on one device; the backend is a cargo feature
//!   (`cuda`, `rocm`, `metal`, `vulkan`, `wgpu`; `mkl` or `accelerate` for CPU BLAS).
//! - `Var` and `backprop`: reverse-mode autograd.
//! - `quantized`: GGUF and GGML files, `QTensor`, `QMatMul`, and 28 block types with a CPU
//!   reference decode for each.
//! - `safetensors`, `npy`, `pickle`: weight formats.
//! - `CustomOp1`, `CustomOp2`, `CustomOp3`: a kernel of your own as a graph op, with its backward.
//!
//! Layers and optimizers are in [hanzo-nn](https://docs.rs/hanzo-nn), model implementations in
//! [hanzo-transformers](https://docs.rs/hanzo-transformers), and the kernel DSL in
//! [hanzo-kernel](https://docs.rs/hanzo-kernel).

#[cfg(feature = "accelerate")]
mod accelerate;
pub mod backend;
pub mod backprop;
pub mod compute;
pub mod conv;
mod convert;
pub mod cpu;
pub mod cpu_backend;
#[cfg(feature = "cuda")]
pub mod cuda_backend;
mod custom_op;
mod device;
pub mod display;
mod dtype;
pub mod dummy_cuda_backend;
pub mod dummy_dtype;
mod dummy_metal_backend;
pub mod dummy_vulkan_backend;
pub mod dummy_wgpu_backend;
pub mod error;
pub mod fattn_decode;
mod hc_sinkhorn;
mod indexer;
pub mod layout;
#[cfg(feature = "metal")]
pub mod metal_backend;
#[cfg(feature = "mkl")]
mod mkl;
pub mod model_delta;
pub mod nditer;
pub mod npy;
pub mod op;
pub mod pickle;
pub mod quantized;
#[cfg(feature = "rocm")]
pub mod rocm_backend;
pub mod safetensors;
pub mod scalar;
pub mod shape;
mod sort;
mod storage;
pub mod streaming;
mod strided_index;
mod tensor;
mod tensor_cat;
pub mod test_utils;
pub mod utils;
mod variable;
#[cfg(feature = "vulkan")]
pub mod vulkan_backend;
#[cfg(feature = "wgpu")]
pub mod wgpu_backend;

#[cfg(feature = "cudnn")]
pub use cuda_backend::cudnn;
#[cfg(feature = "cutile")]
pub use cuda_backend::cutile;

pub use cpu_backend::{CpuStorage, CpuStorageRef};
pub use custom_op::{
    CustomOp1, CustomOp2, CustomOp3, InplaceOp1, InplaceOp2, InplaceOp3, InplaceOpN,
};
pub use device::{Device, DeviceLocation, NdArray, Timer};
pub use dtype::{DType, DTypeParseError, FloatDType, IntDType, WithDType};
pub use dummy_dtype::{F4, F6E2M3, F6E3M2, F8E8M0};
pub use error::{Context, Error, Result};
pub use indexer::{IndexOp, TensorIndexer};
pub use layout::Layout;
pub use nditer::NdIter;
pub use shape::{Shape, D};
pub use storage::{Storage, StorageMutRef, StorageRef};
pub use streaming::{StreamTensor, StreamingBinOp, StreamingModule};
pub use strided_index::{StridedBlocks, StridedIndex};
pub use tensor::{Tensor, TensorId};
pub use variable::Var;

#[cfg(feature = "cuda")]
pub use cuda_backend as cuda;

#[cfg(not(feature = "cuda"))]
pub use dummy_cuda_backend as cuda;

pub use cuda::{CudaDevice, CudaStorage};

#[cfg(feature = "metal")]
pub use metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(not(feature = "metal"))]
pub use dummy_metal_backend::{MetalDevice, MetalError, MetalStorage};

#[cfg(feature = "rocm")]
pub use rocm_backend::{
    set_force_scalar_matvec, RocmDevice, RocmError, RocmQuantType, RocmStorage,
};

#[cfg(feature = "vulkan")]
pub use vulkan_backend as vulkan;

#[cfg(not(feature = "vulkan"))]
pub use dummy_vulkan_backend as vulkan;

pub use vulkan::{VulkanDevice, VulkanStorage};

#[cfg(feature = "vulkan")]
pub use vulkan::{PagedAttnArgs, ReshapeCacheArgs, VkGraph, VkGraphAttn};

// `wgpu_be` alias to avoid clashing with the `wgpu` crate name.
#[cfg(feature = "wgpu")]
pub use wgpu_backend as wgpu_be;

#[cfg(not(feature = "wgpu"))]
pub use dummy_wgpu_backend as wgpu_be;

pub use wgpu_be::{WgpuDevice, WgpuStorage};

#[cfg(feature = "mkl")]
extern crate intel_mkl_src;

#[cfg(feature = "accelerate")]
extern crate accelerate_src;

pub trait ToUsize2 {
    fn to_usize2(self) -> (usize, usize);
}

impl ToUsize2 for usize {
    fn to_usize2(self) -> (usize, usize) {
        (self, self)
    }
}

impl ToUsize2 for (usize, usize) {
    fn to_usize2(self) -> (usize, usize) {
        self
    }
}

/// Defining a module with forward method using a single argument.
pub trait Module {
    fn forward(&self, xs: &Tensor) -> Result<Tensor>;
}

impl<T: Fn(&Tensor) -> Result<Tensor>> Module for T {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self(xs)
    }
}

impl<M: Module> Module for Option<&M> {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        match self {
            None => Ok(xs.clone()),
            Some(m) => m.forward(xs),
        }
    }
}

/// A single forward method using a single tensor argument and a flag to
/// separate the training and evaluation behaviors.
pub trait ModuleT {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Result<Tensor>;
}

impl<M: Module> ModuleT for M {
    fn forward_t(&self, xs: &Tensor, _train: bool) -> Result<Tensor> {
        self.forward(xs)
    }
}
