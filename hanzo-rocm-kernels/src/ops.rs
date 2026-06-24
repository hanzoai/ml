//! Kernel operations for binary and unary ops.

use crate::compile::KernelCache;
use crate::error::KernelError;
use crate::kernel::{dtype_suffix, BinaryOp, UnaryOp};
use crate::utils::grid_block_config;
use rocm_rs::hip::{DeviceMemory, Dim3, Stream};
use std::sync::Arc;

/// Q/K/V/O shape for a flash-attention launch. Q,O are `[B, Hq, Lq, head_dim]`; K,V are
/// `[B, Hkv, Lk, head_dim]` (GQA: `Hq % Hkv == 0`). `head_dim` must be 128.
#[derive(Clone, Copy, Debug)]
pub struct FlashAttnShape {
    pub batch: i32,
    pub q_heads: i32,
    pub kv_heads: i32,
    pub q_len: i32,
    pub kv_len: i32,
    pub head_dim: i32,
    pub scale: f32,
    pub causal: bool,
}

/// Launcher for kernel operations.
///
/// This struct provides a unified interface for launching both
/// binary and unary operations, reducing code duplication.
pub struct OpLauncher {
    cache: Arc<KernelCache>,
}

impl OpLauncher {
    /// Create a new OpLauncher for the given device
    pub fn new(device: &rocm_rs::hip::Device) -> Result<Self, KernelError> {
        let cache = Arc::new(KernelCache::new(device)?);
        Ok(Self { cache })
    }

    /// Launch a binary operation kernel.
    ///
    /// # Arguments
    /// * `stream` - The HIP stream to launch on
    /// * `op` - The binary operation to perform
    /// * `numel` - Number of elements
    /// * `num_dims` - Number of dimensions (0 for contiguous)
    /// * `dims_and_strides` - Optional buffer with dimension and stride info
    /// * `lhs` - Left-hand side input
    /// * `rhs` - Right-hand side input
    /// * `output` - Output buffer
    pub fn launch_binary<T: Copy + Send + Sync + 'static>(
        &self,
        stream: &Stream,
        op: BinaryOp,
        numel: usize,
        num_dims: usize,
        dims_and_strides: Option<&DeviceMemory<usize>>,
        lhs: &DeviceMemory<T>,
        rhs: &DeviceMemory<T>,
        output: &mut DeviceMemory<T>,
    ) -> Result<(), KernelError> {
        use crate::kernel::BinaryKernel;
        use crate::kernel::KernelSource;

        let module = self
            .cache
            .get_or_load(BinaryKernel::NAME, BinaryKernel::CODE)?;
        let kernel_name = format!("{}_{}", op.kernel_name(), dtype_suffix::<T>());

        let function = module
            .get_function(&kernel_name)
            .map_err(|e| KernelError::Launch(format!("Kernel {} not found: {}", kernel_name, e)))?;

        let (grid, block) = grid_block_config(numel);

        // hipModuleLaunchKernel reads each kernel_params[i] as a pointer TO arg i's value.
        // Device pointers are themselves values, so we pass the ADDRESS of a local slot holding
        // each pointer/scalar, and keep those slots alive until after `launch` returns.
        let info_ptr: *mut std::ffi::c_void = dims_and_strides
            .map(|m| m.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        let lhs_ptr = lhs.as_ptr();
        let rhs_ptr = rhs.as_ptr();
        let out_ptr = output.as_ptr();

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&numel) as *const usize as *mut std::ffi::c_void,
            (&num_dims) as *const usize as *mut std::ffi::c_void,
            (&info_ptr) as *const _ as *mut std::ffi::c_void,
            (&lhs_ptr) as *const _ as *mut std::ffi::c_void,
            (&rhs_ptr) as *const _ as *mut std::ffi::c_void,
            (&out_ptr) as *const _ as *mut std::ffi::c_void,
        ];

        function
            .launch(grid, block, 0, Some(stream), &mut args)
            .map_err(|e| KernelError::Launch(e.to_string()))?;

        Ok(())
    }

    /// Launch a unary operation kernel.
    ///
    /// # Arguments
    /// * `stream` - The HIP stream to launch on
    /// * `op` - The unary operation to perform
    /// * `numel` - Number of elements
    /// * `num_dims` - Number of dimensions (0 for contiguous)
    /// * `dims_and_strides` - Optional buffer with dimension and stride info
    /// * `input` - Input buffer
    /// * `output` - Output buffer
    pub fn launch_unary<T: Copy + Send + Sync + 'static>(
        &self,
        stream: &Stream,
        op: UnaryOp,
        numel: usize,
        num_dims: usize,
        dims_and_strides: Option<&DeviceMemory<usize>>,
        input: &DeviceMemory<T>,
        output: &mut DeviceMemory<T>,
    ) -> Result<(), KernelError> {
        use crate::kernel::KernelSource;
        use crate::kernel::UnaryKernel;

        let module = self
            .cache
            .get_or_load(UnaryKernel::NAME, UnaryKernel::CODE)?;
        let kernel_name = format!("{}_{}", op.kernel_name(), dtype_suffix::<T>());

        let function = module
            .get_function(&kernel_name)
            .map_err(|e| KernelError::Launch(format!("Kernel {} not found: {}", kernel_name, e)))?;

        let (grid, block) = grid_block_config(numel);

        // hipModuleLaunchKernel reads each kernel_params[i] as a pointer TO arg i's value.
        // Pass the ADDRESS of a local slot holding each pointer/scalar (slots outlive `launch`).
        let info_ptr: *mut std::ffi::c_void = dims_and_strides
            .map(|m| m.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_ptr();

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&numel) as *const usize as *mut std::ffi::c_void,
            (&num_dims) as *const usize as *mut std::ffi::c_void,
            (&info_ptr) as *const _ as *mut std::ffi::c_void,
            (&in_ptr) as *const _ as *mut std::ffi::c_void,
            (&out_ptr) as *const _ as *mut std::ffi::c_void,
        ];

        function
            .launch(grid, block, 0, Some(stream), &mut args)
            .map_err(|e| KernelError::Launch(e.to_string()))?;

        Ok(())
    }

    /// Launch a pow unary operation with an exponent value.
    ///
    /// # Arguments
    /// * `stream` - The HIP stream to launch on
    /// * `numel` - Number of elements
    /// * `num_dims` - Number of dimensions (0 for contiguous)
    /// * `dims_and_strides` - Optional buffer with dimension and stride info
    /// * `input` - Input buffer
    /// * `exp_val` - Exponent value
    /// * `output` - Output buffer
    pub fn launch_pow<T: Copy + Send + Sync + 'static>(
        &self,
        stream: &Stream,
        numel: usize,
        num_dims: usize,
        dims_and_strides: Option<&DeviceMemory<usize>>,
        input: &DeviceMemory<T>,
        exp_val: T,
        output: &mut DeviceMemory<T>,
    ) -> Result<(), KernelError> {
        use crate::kernel::KernelSource;
        use crate::kernel::UnaryKernel;

        let module = self
            .cache
            .get_or_load(UnaryKernel::NAME, UnaryKernel::CODE)?;
        let kernel_name = format!("upow_{}", dtype_suffix::<T>());

        let function = module
            .get_function(&kernel_name)
            .map_err(|e| KernelError::Launch(format!("Kernel {} not found: {}", kernel_name, e)))?;

        let (grid, block) = grid_block_config(numel);

        // hipModuleLaunchKernel reads each kernel_params[i] as a pointer TO arg i's value.
        // Pass the ADDRESS of a local slot holding each pointer/scalar (slots outlive `launch`).
        // `exp_val` is a by-value scalar arg: its address is the slot, the value rides inline.
        let info_ptr: *mut std::ffi::c_void = dims_and_strides
            .map(|m| m.as_ptr())
            .unwrap_or(std::ptr::null_mut());
        let in_ptr = input.as_ptr();
        let out_ptr = output.as_ptr();

        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&numel) as *const usize as *mut std::ffi::c_void,
            (&num_dims) as *const usize as *mut std::ffi::c_void,
            (&info_ptr) as *const _ as *mut std::ffi::c_void,
            (&in_ptr) as *const _ as *mut std::ffi::c_void,
            (&exp_val) as *const T as *mut std::ffi::c_void,
            (&out_ptr) as *const _ as *mut std::ffi::c_void,
        ];

        function
            .launch(grid, block, 0, Some(stream), &mut args)
            .map_err(|e| KernelError::Launch(e.to_string()))?;

        Ok(())
    }

    /// Launch the matrix-core (WMMA) flash-attention forward kernel.
    ///
    /// `dtype` selects the entry point: `"f16"` or `"bf16"`. Q/K/V/O are raw device pointers of the
    /// matching element type (head_dim must be 128). Output `o` is `[B, Hq, Lq, head_dim]`.
    pub fn launch_flash_attn<T: Copy + Send + Sync + 'static>(
        &self,
        stream: &Stream,
        dtype: &str,
        shape: FlashAttnShape,
        q: &DeviceMemory<T>,
        k: &DeviceMemory<T>,
        v: &DeviceMemory<T>,
        o: &mut DeviceMemory<T>,
    ) -> Result<(), KernelError> {
        use crate::kernel::FlashKernel;
        use crate::kernel::KernelSource;

        const BR: i32 = 64;
        const BLOCK: u32 = 128;

        let module = self
            .cache
            .get_or_load(FlashKernel::NAME, FlashKernel::CODE)?;
        let kernel_name = format!("flash_attn_{}", dtype);
        let function = module
            .get_function(&kernel_name)
            .map_err(|e| KernelError::Launch(format!("Kernel {} not found: {}", kernel_name, e)))?;

        let grid = Dim3 {
            x: ((shape.q_len + BR - 1) / BR) as u32,
            y: shape.q_heads as u32,
            z: shape.batch as u32,
        };
        let block = Dim3 {
            x: BLOCK,
            y: 1,
            z: 1,
        };

        let causal: i32 = i32::from(shape.causal);
        // hipModuleLaunchKernel reads each kernel_params[i] as a pointer TO arg i's value. Pointer
        // args are themselves values, so we pass the ADDRESS of each device pointer, not the pointer.
        let qp = q.as_ptr();
        let kp = k.as_ptr();
        let vp = v.as_ptr();
        let op = o.as_ptr();
        let mut args: Vec<*mut std::ffi::c_void> = vec![
            (&shape.batch) as *const i32 as *mut std::ffi::c_void,
            (&shape.q_heads) as *const i32 as *mut std::ffi::c_void,
            (&shape.kv_heads) as *const i32 as *mut std::ffi::c_void,
            (&shape.q_len) as *const i32 as *mut std::ffi::c_void,
            (&shape.kv_len) as *const i32 as *mut std::ffi::c_void,
            (&shape.scale) as *const f32 as *mut std::ffi::c_void,
            (&causal) as *const i32 as *mut std::ffi::c_void,
            (&qp) as *const _ as *mut std::ffi::c_void,
            (&kp) as *const _ as *mut std::ffi::c_void,
            (&vp) as *const _ as *mut std::ffi::c_void,
            (&op) as *const _ as *mut std::ffi::c_void,
        ];

        function
            .launch(grid, block, 0, Some(stream), &mut args)
            .map_err(|e| KernelError::Launch(e.to_string()))?;

        Ok(())
    }

    /// Get the underlying kernel cache
    pub fn cache(&self) -> &Arc<KernelCache> {
        &self.cache
    }
}
