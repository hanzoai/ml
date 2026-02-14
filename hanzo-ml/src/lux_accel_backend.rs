// Package accel provides Rust bindings for lux-accel native ML acceleration
//
// This bridges hanzo-ml (Rust) with lux-accel (Go/C) for maximum performance

use std::ffi::{c_char, c_float, c_int, c_void};
use crate::{CpuStorage, Device, DType, Result, Storage, Tensor};

/// C FFI bindings to lux-accel
extern "C" {
    // Library management
    fn lux_accel_init() -> c_int;
    fn lux_accel_shutdown();
    fn lux_accel_available() -> c_int;
    
    // Device management  
    fn lux_accel_device_count() -> c_int;
    fn lux_accel_device_info(device_id: c_int, info: *mut DeviceInfo) -> c_int;
    
    // ML Operations
    fn lux_accel_matmul(
        a_data: *const c_void, a_shape: *const c_int, a_ndim: c_int,
        b_data: *const c_void, b_shape: *const c_int, b_ndim: c_int,
        c_data: *mut c_void, c_shape: *const c_int, c_ndim: c_int,
        dtype: c_int
    ) -> c_int;
    
    fn lux_accel_relu(
        input: *const c_void, output: *mut c_void,
        shape: *const c_int, ndim: c_int, dtype: c_int
    ) -> c_int;
    
    fn lux_accel_attention(
        q_data: *const c_void, k_data: *const c_void, v_data: *const c_void,
        output: *mut c_void, shape: *const c_int, ndim: c_int,
        scale: c_float, dtype: c_int
    ) -> c_int;
    
    fn lux_accel_layer_norm(
        input: *const c_void, gamma: *const c_void, beta: *const c_void,
        output: *mut c_void, shape: *const c_int, ndim: c_int,
        eps: c_float, dtype: c_int
    ) -> c_int;
}

#[repr(C)]
pub struct DeviceInfo {
    pub id: c_int,
    pub name: [c_char; 256],
    pub memory_total: u64,
    pub memory_free: u64,
    pub compute_units: c_int,
}

/// Convert DType to lux-accel dtype code
/// f32 = 0, f16 = 1, bf16 = 2
fn dtype_to_code(dtype: DType) -> Result<c_int> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        _ => Err(format!("Unsupported dtype for lux-accel: {:?}", dtype).into()),
    }
}

/// Convert shape dimensions to c_int array
fn shape_to_c_int(dims: &[usize]) -> Vec<c_int> {
    dims.iter().map(|&d| d as c_int).collect()
}

/// Get raw data pointer from contiguous CPU storage
/// Returns pointer offset by start_offset elements
unsafe fn storage_data_ptr(storage: &CpuStorage, dtype: DType, offset: usize) -> *const c_void {
    match (storage, dtype) {
        (CpuStorage::F32(data), DType::F32) => data.as_ptr().add(offset) as *const c_void,
        (CpuStorage::F16(data), DType::F16) => data.as_ptr().add(offset) as *const c_void,
        (CpuStorage::BF16(data), DType::BF16) => data.as_ptr().add(offset) as *const c_void,
        _ => std::ptr::null(),
    }
}

/// Get mutable raw data pointer from CPU storage
unsafe fn storage_data_ptr_mut(storage: &mut CpuStorage, dtype: DType) -> *mut c_void {
    match (storage, dtype) {
        (CpuStorage::F32(data), DType::F32) => data.as_mut_ptr() as *mut c_void,
        (CpuStorage::F16(data), DType::F16) => data.as_mut_ptr() as *mut c_void,
        (CpuStorage::BF16(data), DType::BF16) => data.as_mut_ptr() as *mut c_void,
        _ => std::ptr::null_mut(),
    }
}

/// Allocate output storage for given shape and dtype
fn allocate_output_storage(elem_count: usize, dtype: DType) -> Result<CpuStorage> {
    match dtype {
        DType::F32 => Ok(CpuStorage::F32(vec![0.0f32; elem_count])),
        DType::F16 => Ok(CpuStorage::F16(vec![half::f16::ZERO; elem_count])),
        DType::BF16 => Ok(CpuStorage::BF16(vec![half::bf16::ZERO; elem_count])),
        _ => Err(format!("Unsupported dtype for output allocation: {:?}", dtype).into()),
    }
}

/// Lux-Accel accelerated device
pub struct LuxAccelDevice {
    device_id: i32,
    initialized: bool,
}

impl LuxAccelDevice {
    pub fn new() -> Result<Self> {
        unsafe {
            if lux_accel_init() != 0 {
                return Err("Failed to initialize lux-accel".into());
            }
            if lux_accel_available() == 0 {
                return Err("No lux-accel devices available".into());
            }
        }
        
        Ok(Self {
            device_id: 0,
            initialized: true,
        })
    }
    
    pub fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Validate dtypes match and are supported
        let dtype = a.dtype();
        if dtype != b.dtype() {
            return Err(format!(
                "matmul dtype mismatch: {:?} vs {:?}",
                dtype,
                b.dtype()
            )
            .into());
        }
        let dtype_code = dtype_to_code(dtype)?;

        // Ensure tensors are contiguous for FFI
        let a = a.contiguous()?;
        let b = b.contiguous()?;

        // Get dimensions: A[..., m, k] @ B[..., k, n] = C[..., m, n]
        let a_dims = a.dims();
        let b_dims = b.dims();

        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err("matmul requires at least 2D tensors".into());
        }

        let a_rank = a_dims.len();
        let b_rank = b_dims.len();
        let m = a_dims[a_rank - 2];
        let k = a_dims[a_rank - 1];
        let k2 = b_dims[b_rank - 2];
        let n = b_dims[b_rank - 1];

        if k != k2 {
            return Err(format!(
                "matmul inner dimensions mismatch: {} vs {}",
                k, k2
            )
            .into());
        }

        // Compute output shape (handle batching)
        let mut c_shape: Vec<usize> = a_dims[..a_rank - 2].to_vec();
        c_shape.push(m);
        c_shape.push(n);
        let c_elem_count: usize = c_shape.iter().product();

        // Convert shapes to c_int
        let a_shape_c = shape_to_c_int(a_dims);
        let b_shape_c = shape_to_c_int(b_dims);
        let c_shape_c = shape_to_c_int(&c_shape);

        // Get storage and data pointers
        let (a_storage, a_layout) = a.storage_and_layout();
        let (b_storage, b_layout) = b.storage_and_layout();

        let a_cpu = match &*a_storage {
            Storage::Cpu(s) => s,
            _ => return Err("matmul requires CPU tensors for lux-accel bridge".into()),
        };
        let b_cpu = match &*b_storage {
            Storage::Cpu(s) => s,
            _ => return Err("matmul requires CPU tensors for lux-accel bridge".into()),
        };

        // Allocate output
        let mut c_storage = allocate_output_storage(c_elem_count, dtype)?;

        unsafe {
            let a_ptr = storage_data_ptr(a_cpu, dtype, a_layout.start_offset());
            let b_ptr = storage_data_ptr(b_cpu, dtype, b_layout.start_offset());
            let c_ptr = storage_data_ptr_mut(&mut c_storage, dtype);

            if a_ptr.is_null() || b_ptr.is_null() || c_ptr.is_null() {
                return Err("Failed to get data pointers for matmul".into());
            }

            let result = lux_accel_matmul(
                a_ptr,
                a_shape_c.as_ptr(),
                a_rank as c_int,
                b_ptr,
                b_shape_c.as_ptr(),
                b_rank as c_int,
                c_ptr,
                c_shape_c.as_ptr(),
                c_shape.len() as c_int,
                dtype_code,
            );

            if result != 0 {
                return Err(format!("lux_accel_matmul failed with code: {}", result).into());
            }
        }

        // Create output tensor from storage
        match c_storage {
            CpuStorage::F32(v) => Tensor::from_vec(v, c_shape.as_slice(), &Device::Cpu),
            CpuStorage::F16(v) => Tensor::from_vec(v, c_shape.as_slice(), &Device::Cpu),
            CpuStorage::BF16(v) => Tensor::from_vec(v, c_shape.as_slice(), &Device::Cpu),
            _ => Err("Unexpected storage type".into()),
        }
    }
    
    pub fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor, scale: f32) -> Result<Tensor> {
        // Validate dtypes match and are supported
        let dtype = q.dtype();
        if dtype != k.dtype() || dtype != v.dtype() {
            return Err(format!(
                "attention dtype mismatch: Q={:?}, K={:?}, V={:?}",
                dtype,
                k.dtype(),
                v.dtype()
            )
            .into());
        }
        let dtype_code = dtype_to_code(dtype)?;

        // Ensure tensors are contiguous for FFI
        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        // For attention: Q, K, V typically have shape [batch, heads, seq_len, head_dim]
        // or [batch, seq_len, hidden_dim] - we pass the shape to the C function
        let q_dims = q.dims();
        let k_dims = k.dims();
        let v_dims = v.dims();

        // Basic shape validation - K and V should have compatible shapes
        if q_dims.len() != k_dims.len() || q_dims.len() != v_dims.len() {
            return Err(format!(
                "attention rank mismatch: Q={}, K={}, V={}",
                q_dims.len(),
                k_dims.len(),
                v_dims.len()
            )
            .into());
        }

        // Output shape matches Q shape (attention output has same shape as query)
        let output_shape = q_dims.to_vec();
        let output_elem_count: usize = output_shape.iter().product();

        // Convert shape to c_int (using Q's shape for the FFI call)
        let shape_c = shape_to_c_int(q_dims);

        // Get storage and data pointers
        let (q_storage, q_layout) = q.storage_and_layout();
        let (k_storage, k_layout) = k.storage_and_layout();
        let (v_storage, v_layout) = v.storage_and_layout();

        let q_cpu = match &*q_storage {
            Storage::Cpu(s) => s,
            _ => return Err("attention requires CPU tensors for lux-accel bridge".into()),
        };
        let k_cpu = match &*k_storage {
            Storage::Cpu(s) => s,
            _ => return Err("attention requires CPU tensors for lux-accel bridge".into()),
        };
        let v_cpu = match &*v_storage {
            Storage::Cpu(s) => s,
            _ => return Err("attention requires CPU tensors for lux-accel bridge".into()),
        };

        // Allocate output
        let mut output_storage = allocate_output_storage(output_elem_count, dtype)?;

        unsafe {
            let q_ptr = storage_data_ptr(q_cpu, dtype, q_layout.start_offset());
            let k_ptr = storage_data_ptr(k_cpu, dtype, k_layout.start_offset());
            let v_ptr = storage_data_ptr(v_cpu, dtype, v_layout.start_offset());
            let output_ptr = storage_data_ptr_mut(&mut output_storage, dtype);

            if q_ptr.is_null() || k_ptr.is_null() || v_ptr.is_null() || output_ptr.is_null() {
                return Err("Failed to get data pointers for attention".into());
            }

            let result = lux_accel_attention(
                q_ptr,
                k_ptr,
                v_ptr,
                output_ptr,
                shape_c.as_ptr(),
                q_dims.len() as c_int,
                scale as c_float,
                dtype_code,
            );

            if result != 0 {
                return Err(format!("lux_accel_attention failed with code: {}", result).into());
            }
        }

        // Create output tensor from storage
        match output_storage {
            CpuStorage::F32(v) => Tensor::from_vec(v, output_shape.as_slice(), &Device::Cpu),
            CpuStorage::F16(v) => Tensor::from_vec(v, output_shape.as_slice(), &Device::Cpu),
            CpuStorage::BF16(v) => Tensor::from_vec(v, output_shape.as_slice(), &Device::Cpu),
            _ => Err("Unexpected storage type".into()),
        }
    }
}

impl Drop for LuxAccelDevice {
    fn drop(&mut self) {
        if self.initialized {
            unsafe { lux_accel_shutdown(); }
        }
    }
}