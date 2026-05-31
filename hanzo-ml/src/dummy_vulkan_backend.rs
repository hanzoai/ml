#![allow(dead_code)]
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Error, Layout, Result, Shape};

#[derive(Debug, Clone)]
pub struct VulkanDevice;

#[derive(Debug)]
pub struct VulkanStorage;

#[derive(thiserror::Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    Message(String),
}

impl From<String> for VulkanError {
    fn from(e: String) -> Self {
        VulkanError::Message(e)
    }
}

fn err<T>() -> Result<T> {
    Err(Error::Msg(
        "candle was not compiled with vulkan support".into(),
    ))
}

macro_rules! fail {
    () => {
        unimplemented!("vulkan support has not been enabled, add `vulkan` feature to enable.")
    };
}

impl crate::backend::BackendStorage for VulkanStorage {
    type Device = VulkanDevice;

    fn try_clone(&self, _: &Layout) -> Result<Self> {
        err()
    }

    fn dtype(&self) -> DType {
        fail!()
    }

    fn device(&self) -> &Self::Device {
        fail!()
    }

    fn const_set(&mut self, _: crate::scalar::Scalar, _: &Layout) -> Result<()> {
        err()
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        err()
    }

    fn affine(&self, _: &Layout, _: f64, _: f64) -> Result<Self> {
        err()
    }

    fn powf(&self, _: &Layout, _: f64) -> Result<Self> {
        err()
    }

    fn elu(&self, _: &Layout, _: f64) -> Result<Self> {
        err()
    }

    fn reduce_op(&self, _: ReduceOp, _: &Layout, _: &[usize]) -> Result<Self> {
        err()
    }

    fn cmp(&self, _: CmpOp, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        err()
    }

    fn to_dtype(&self, _: &Layout, _: DType) -> Result<Self> {
        err()
    }

    fn unary_impl<B: UnaryOpT>(&self, _: &Layout) -> Result<Self> {
        err()
    }

    fn binary_impl<B: BinaryOpT>(&self, _: &Self, _: &Layout, _: &Layout) -> Result<Self> {
        err()
    }

    fn where_cond(&self, _: &Layout, _: &Self, _: &Layout, _: &Self, _: &Layout) -> Result<Self> {
        err()
    }

    fn conv1d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        err()
    }

    fn conv_transpose1d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        err()
    }

    fn conv2d(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        err()
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kernel_l: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        err()
    }

    fn index_select(&self, _: &Self, _: &Layout, _: &Layout, _: usize) -> Result<Self> {
        err()
    }
    fn gather(&self, _: &Layout, _: &Self, _: &Layout, _: usize) -> Result<Self> {
        err()
    }

    fn scatter_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        err()
    }

    fn scatter_add_set(
        &mut self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<()> {
        err()
    }

    fn index_add(
        &self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: &Self,
        _: &Layout,
        _: usize,
    ) -> Result<Self> {
        err()
    }

    fn matmul(
        &self,
        _: &Self,
        _: (usize, usize, usize, usize),
        _: &Layout,
        _: &Layout,
    ) -> Result<Self> {
        err()
    }

    fn copy_strided_src(&self, _: &mut Self, _: usize, _: &Layout) -> Result<()> {
        err()
    }

    fn copy2d(
        &self,
        _: &mut Self,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
        _: usize,
    ) -> Result<()> {
        err()
    }

    fn avg_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        err()
    }

    fn max_pool2d(&self, _: &Layout, _: (usize, usize), _: (usize, usize)) -> Result<Self> {
        err()
    }

    fn upsample_nearest1d(&self, _: &Layout, _: usize) -> Result<Self> {
        err()
    }

    fn upsample_nearest2d(&self, _: &Layout, _: usize, _: usize) -> Result<Self> {
        err()
    }

    fn upsample_bilinear2d(
        &self,
        _: &Layout,
        _: usize,
        _: usize,
        _: bool,
        _: Option<f64>,
        _: Option<f64>,
    ) -> Result<Self> {
        err()
    }
}

impl crate::backend::BackendDevice for VulkanDevice {
    type Storage = VulkanStorage;
    fn new(_: usize) -> Result<Self> {
        err()
    }

    fn set_seed(&self, _: u64) -> Result<()> {
        err()
    }

    fn get_current_seed(&self) -> Result<u64> {
        err()
    }

    fn location(&self) -> crate::DeviceLocation {
        fail!()
    }

    fn same_device(&self, _: &Self) -> bool {
        fail!()
    }

    fn zeros_impl(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        err()
    }

    unsafe fn alloc_uninit(&self, _shape: &Shape, _dtype: DType) -> Result<Self::Storage> {
        err()
    }

    fn storage_from_slice<T: crate::WithDType>(&self, _: &[T]) -> Result<Self::Storage> {
        err()
    }

    fn storage_from_cpu_storage(&self, _: &CpuStorage) -> Result<Self::Storage> {
        err()
    }

    fn storage_from_cpu_storage_owned(&self, _: CpuStorage) -> Result<Self::Storage> {
        err()
    }

    fn rand_uniform(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        err()
    }

    fn rand_normal(&self, _: &Shape, _: DType, _: f64, _: f64) -> Result<Self::Storage> {
        err()
    }

    fn synchronize(&self) -> Result<()> {
        Ok(())
    }
}
