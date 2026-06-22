//! Implementation of Backend traits for ROCm device
//!
use crate::backend::BackendStorage;
use crate::op::{BinaryOpT, CmpOp, ReduceOp, UnaryOpT};
use crate::{CpuStorage, DType, Layout, Result, WithDType};
use half::{bf16, f16};
pub use hanzo_rocm_kernels as kernels;
use hanzo_rocm_kernels::kernel::KernelSource;
pub use rocm_rs;
use rocm_rs::hip::bindings;
use rocm_rs::rocblas::{self, level3::GemmStridedBatchedType, types::Operation};

mod device;
mod error;
#[cfg(feature = "rocm-miopen")]
mod miopen;
mod wrappers;
pub use device::{DeviceId, RocmDevice};
pub use error::{RocmError, WrapErr};
pub use wrappers::SendSyncDeviceMemory;
pub mod utils;

pub enum RocmStorageSlice {
    U8(SendSyncDeviceMemory<u8>),
    U32(SendSyncDeviceMemory<u32>),
    I16(SendSyncDeviceMemory<i16>),
    I32(SendSyncDeviceMemory<i32>),
    I64(SendSyncDeviceMemory<i64>),
    BF16(SendSyncDeviceMemory<bf16>),
    F16(SendSyncDeviceMemory<f16>),
    F32(SendSyncDeviceMemory<f32>),
    F64(SendSyncDeviceMemory<f64>),
    F8E4M3(SendSyncDeviceMemory<u8>),
}

impl std::fmt::Debug for RocmStorageSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RocmStorageSlice::U8(m) => write!(f, "U8({} bytes)", m.size()),
            RocmStorageSlice::U32(m) => write!(f, "U32({} bytes)", m.size()),
            RocmStorageSlice::I16(m) => write!(f, "I16({} bytes)", m.size()),
            RocmStorageSlice::I32(m) => write!(f, "I32({} bytes)", m.size()),
            RocmStorageSlice::I64(m) => write!(f, "I64({} bytes)", m.size()),
            RocmStorageSlice::BF16(m) => write!(f, "BF16({} bytes)", m.size()),
            RocmStorageSlice::F16(m) => write!(f, "F16({} bytes)", m.size()),
            RocmStorageSlice::F32(m) => write!(f, "F32({} bytes)", m.size()),
            RocmStorageSlice::F64(m) => write!(f, "F64({} bytes)", m.size()),
            RocmStorageSlice::F8E4M3(m) => write!(f, "F8E4M3({} bytes)", m.size()),
        }
    }
}

impl RocmStorageSlice {
    pub fn dtype(&self) -> DType {
        match self {
            RocmStorageSlice::U8(_) => DType::U8,
            RocmStorageSlice::U32(_) => DType::U32,
            RocmStorageSlice::I16(_) => DType::I16,
            RocmStorageSlice::I32(_) => DType::I32,
            RocmStorageSlice::I64(_) => DType::I64,
            RocmStorageSlice::BF16(_) => DType::BF16,
            RocmStorageSlice::F16(_) => DType::F16,
            RocmStorageSlice::F32(_) => DType::F32,
            RocmStorageSlice::F64(_) => DType::F64,
            RocmStorageSlice::F8E4M3(_) => DType::F8E4M3,
        }
    }

    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        match self {
            RocmStorageSlice::U8(m) => m.as_ptr(),
            RocmStorageSlice::U32(m) => m.as_ptr(),
            RocmStorageSlice::I16(m) => m.as_ptr(),
            RocmStorageSlice::I32(m) => m.as_ptr(),
            RocmStorageSlice::I64(m) => m.as_ptr(),
            RocmStorageSlice::BF16(m) => m.as_ptr(),
            RocmStorageSlice::F16(m) => m.as_ptr(),
            RocmStorageSlice::F32(m) => m.as_ptr(),
            RocmStorageSlice::F64(m) => m.as_ptr(),
            RocmStorageSlice::F8E4M3(m) => m.as_ptr(),
        }
    }

    fn elem_size(&self) -> usize {
        match self {
            RocmStorageSlice::U8(_) | RocmStorageSlice::F8E4M3(_) => 1,
            RocmStorageSlice::I16(_) | RocmStorageSlice::BF16(_) | RocmStorageSlice::F16(_) => 2,
            RocmStorageSlice::U32(_) | RocmStorageSlice::I32(_) | RocmStorageSlice::F32(_) => 4,
            RocmStorageSlice::I64(_) | RocmStorageSlice::F64(_) => 8,
        }
    }

    unsafe fn offset_ptr(&self, offset: usize) -> *mut std::ffi::c_void {
        self.as_ptr().add(offset * self.elem_size())
    }
}

pub struct RocmStorage {
    pub slice: RocmStorageSlice,
    pub device: RocmDevice,
}

struct GemmConfig<T> {
    alpha: T,
    beta: T,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    transa: Operation,
    transb: Operation,
}

struct StridedBatchedConfig<T> {
    batch_size: i32,
    gemm: GemmConfig<T>,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
}

fn gemm_config<T: Copy>(
    alpha: T,
    beta: T,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> std::result::Result<StridedBatchedConfig<T>, RocmError> {
    let lhs_stride = lhs_l.stride();
    let rhs_stride = rhs_l.stride();
    let rhs_m1 = rhs_stride[rhs_stride.len() - 1];
    let rhs_m2 = rhs_stride[rhs_stride.len() - 2];
    let lhs_m1 = lhs_stride[lhs_stride.len() - 1];
    let lhs_m2 = lhs_stride[lhs_stride.len() - 2];

    let (lda, transa) = if (rhs_m1 == 1 || n == 1) && (rhs_m2 == n || k == 1) {
        (n as i32, Operation::None)
    } else if (rhs_m1 == k || n == 1) && (rhs_m2 == 1 || k == 1) {
        (k as i32, Operation::Transpose)
    } else {
        return Err(RocmError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        });
    };

    let (ldb, transb) = if (lhs_m1 == 1 || k == 1) && (lhs_m2 == k || m == 1) {
        (k as i32, Operation::None)
    } else if (lhs_m1 == m || k == 1) && (lhs_m2 == 1 || m == 1) {
        (m as i32, Operation::Transpose)
    } else {
        return Err(RocmError::MatMulNonContiguous {
            lhs_stride: lhs_l.clone(),
            rhs_stride: rhs_l.clone(),
            mnk: (m, n, k),
        });
    };

    let gemm = GemmConfig {
        alpha,
        beta,
        m: n as i32,
        n: m as i32,
        k: k as i32,
        lda,
        ldb,
        ldc: n as i32,
        transa,
        transb,
    };

    let stride_b: usize = match lhs_stride[..lhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * lhs_l.dims()[1] => stride,
        [_, stride] if lhs_l.dims()[0] == 1 => stride,
        [stride, _] if lhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => m * k,
        _ => {
            return Err(RocmError::MatMulNonContiguous {
                lhs_stride: lhs_l.clone(),
                rhs_stride: rhs_l.clone(),
                mnk: (m, n, k),
            })
        }
    };
    let stride_a: usize = match rhs_stride[..rhs_stride.len() - 2] {
        [s1, stride] if s1 == stride * rhs_l.dims()[1] => stride,
        [_, stride] if rhs_l.dims()[0] == 1 => stride,
        [stride, _] if rhs_l.dims()[1] == 1 => stride,
        [stride] => stride,
        [] => n * k,
        _ => {
            return Err(RocmError::MatMulNonContiguous {
                lhs_stride: lhs_l.clone(),
                rhs_stride: rhs_l.clone(),
                mnk: (m, n, k),
            })
        }
    };
    Ok(StridedBatchedConfig {
        batch_size: b as i32,
        gemm,
        stride_a: stride_a as i64,
        stride_b: stride_b as i64,
        stride_c: (m * n) as i64,
    })
}

unsafe fn gemm_strided_batched<T: GemmStridedBatchedType>(
    blas: &rocblas::Handle,
    cfg: StridedBatchedConfig<T>,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    c: *mut std::ffi::c_void,
) -> std::result::Result<(), RocmError> {
    rocblas::gemm_strided_batched(
        blas,
        cfg.gemm.transa,
        cfg.gemm.transb,
        cfg.gemm.m,
        cfg.gemm.n,
        cfg.gemm.k,
        &cfg.gemm.alpha,
        a as *const T,
        cfg.gemm.lda,
        cfg.stride_a,
        b as *const T,
        cfg.gemm.ldb,
        cfg.stride_b,
        &cfg.gemm.beta,
        c as *mut T,
        cfg.gemm.ldc,
        cfg.stride_c,
        cfg.batch_size,
    )
    .map_err(|e| RocmError::Rocblas(e.to_string()))
}

struct GemmExConfig {
    alpha: f32,
    beta: f32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    transa: Operation,
    transb: Operation,
}

struct StridedBatchedExConfig {
    batch_size: i32,
    gemm: GemmExConfig,
    stride_a: i64,
    stride_b: i64,
    stride_c: i64,
}

fn gemm_ex_config(
    alpha: f32,
    beta: f32,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs_l: &Layout,
    rhs_l: &Layout,
) -> std::result::Result<StridedBatchedExConfig, RocmError> {
    let inner = gemm_config(alpha, beta, (b, m, n, k), lhs_l, rhs_l)?;
    Ok(StridedBatchedExConfig {
        batch_size: inner.batch_size,
        gemm: GemmExConfig {
            alpha: inner.gemm.alpha,
            beta: inner.gemm.beta,
            m: inner.gemm.m,
            n: inner.gemm.n,
            k: inner.gemm.k,
            lda: inner.gemm.lda,
            ldb: inner.gemm.ldb,
            ldc: inner.gemm.ldc,
            transa: inner.gemm.transa,
            transb: inner.gemm.transb,
        },
        stride_a: inner.stride_a,
        stride_b: inner.stride_b,
        stride_c: inner.stride_c,
    })
}

unsafe fn gemm_strided_batched_ex(
    blas: &rocblas::Handle,
    cfg: StridedBatchedExConfig,
    a: *const std::ffi::c_void,
    b: *const std::ffi::c_void,
    c: *mut std::ffi::c_void,
    datatype: rocm_rs::rocblas::ffi::rocblas_datatype,
) -> std::result::Result<(), RocmError> {
    use rocm_rs::rocblas::ffi;
    use rocm_rs::rocblas::utils::GemmAlgo;

    let status = unsafe {
        rocblas_gemm_strided_batched_ex(
            blas.as_raw(),
            cfg.gemm.transa.into(),
            cfg.gemm.transb.into(),
            cfg.gemm.m,
            cfg.gemm.n,
            cfg.gemm.k,
            &cfg.gemm.alpha as *const f32 as *const std::ffi::c_void,
            a,
            datatype,
            cfg.gemm.lda,
            cfg.stride_a,
            b,
            datatype,
            cfg.gemm.ldb,
            cfg.stride_b,
            &cfg.gemm.beta as *const f32 as *const std::ffi::c_void,
            c,
            datatype,
            cfg.gemm.ldc,
            cfg.stride_c,
            c,
            datatype,
            cfg.gemm.ldc,
            cfg.stride_c,
            cfg.batch_size,
            ffi::rocblas_datatype__rocblas_datatype_f32_r,
            GemmAlgo::Standard.into(),
            0,
            0,
        )
    };
    if status != ffi::rocblas_status__rocblas_status_success {
        return Err(RocmError::Rocblas(format!(
            "rocblas_gemm_strided_batched_ex failed with status {}",
            status
        )));
    }
    Ok(())
}

extern "C" {
    fn rocblas_gemm_strided_batched_ex(
        handle: rocm_rs::rocblas::ffi::rocblas_handle,
        transA: rocm_rs::rocblas::ffi::rocblas_operation,
        transB: rocm_rs::rocblas::ffi::rocblas_operation,
        m: rocm_rs::rocblas::ffi::rocblas_int,
        n: rocm_rs::rocblas::ffi::rocblas_int,
        k: rocm_rs::rocblas::ffi::rocblas_int,
        alpha: *const std::ffi::c_void,
        a: *const std::ffi::c_void,
        a_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        lda: rocm_rs::rocblas::ffi::rocblas_int,
        stride_a: rocm_rs::rocblas::ffi::rocblas_stride,
        b: *const std::ffi::c_void,
        b_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        ldb: rocm_rs::rocblas::ffi::rocblas_int,
        stride_b: rocm_rs::rocblas::ffi::rocblas_stride,
        beta: *const std::ffi::c_void,
        c: *const std::ffi::c_void,
        c_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        ldc: rocm_rs::rocblas::ffi::rocblas_int,
        stride_c: rocm_rs::rocblas::ffi::rocblas_stride,
        d: *mut std::ffi::c_void,
        d_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        ldd: rocm_rs::rocblas::ffi::rocblas_int,
        stride_d: rocm_rs::rocblas::ffi::rocblas_stride,
        batch_count: rocm_rs::rocblas::ffi::rocblas_int,
        compute_type: rocm_rs::rocblas::ffi::rocblas_datatype,
        algo: rocm_rs::rocblas::ffi::rocblas_gemm_algo,
        solution_index: i32,
        flags: u32,
    ) -> rocm_rs::rocblas::ffi::rocblas_status;
}

macro_rules! dispatch_matmul {
    ($self:expr, $rhs:expr, $b:expr, $m:expr, $n:expr, $k:expr, $lhs_l:expr, $rhs_l:expr, $dev:expr,
     $(($variant:ident, $rust_ty:ty, $alpha:expr, $zero:expr, $cfg_fn:expr, $gemm_fn:expr $(, $ex_datatype:expr)?)),+ $(,)?) => {{
        let elem_count = $b * $m * $n;
        let lhs_ptr = unsafe { $self.slice.offset_ptr($lhs_l.start_offset()) };
        let rhs_ptr = unsafe { $rhs.slice.offset_ptr($rhs_l.start_offset()) };
        let device = $dev.clone();
        let slice = match (&$self.slice, &$rhs.slice) {
            $(
                (RocmStorageSlice::$variant(_), RocmStorageSlice::$variant(_)) => {
                    let cfg = $cfg_fn($alpha, $zero, ($b, $m, $n, $k), $lhs_l, $rhs_l)?;
                    let out = $dev.alloc::<$rust_ty>(elem_count)?;
                    unsafe { $gemm_fn(&$dev.blas, cfg, rhs_ptr, lhs_ptr, out.as_ptr() $(, $ex_datatype)?)?; }
                    RocmStorageSlice::$variant(out)
                }
            )+
            _ => return Err(RocmError::Internal("dtype mismatch in matmul".into()).into()),
        };
        Ok(Self { slice, device })
    }};
}

#[cfg(feature = "rocm-miopen")]
macro_rules! dispatch_miopen_conv {
    ($self:expr, $kernel:expr, $l:expr, $kernel_l:expr, $dst_el:expr, $device:expr, $handle:expr, $func:ident, $($arg:expr),* $(,)?) => {{
        let device = $device.clone();
        let slice = match (&$self.slice, &$kernel.slice) {
            (RocmStorageSlice::F32(s), RocmStorageSlice::F32(w)) => {
                let x_ptr = unsafe { s.offset_ptr($l.start_offset()) } as *mut _;
                let w_ptr = unsafe { w.offset_ptr($kernel_l.start_offset()) } as *mut _;
                let o = device.alloc_zeros::<f32>($dst_el)?;
                $func::<f32>($handle, x_ptr, w_ptr, o.as_ptr() as *mut _, $($arg),*)?;
                RocmStorageSlice::F32(o)
            }
            (RocmStorageSlice::F16(s), RocmStorageSlice::F16(w)) => {
                let x_ptr = unsafe { s.offset_ptr($l.start_offset()) } as *mut _;
                let w_ptr = unsafe { w.offset_ptr($kernel_l.start_offset()) } as *mut _;
                let o = device.alloc_zeros::<f16>($dst_el)?;
                $func::<f16>($handle, x_ptr, w_ptr, o.as_ptr() as *mut _, $($arg),*)?;
                RocmStorageSlice::F16(o)
            }
            (RocmStorageSlice::BF16(s), RocmStorageSlice::BF16(w)) => {
                let x_ptr = unsafe { s.offset_ptr($l.start_offset()) } as *mut _;
                let w_ptr = unsafe { w.offset_ptr($kernel_l.start_offset()) } as *mut _;
                let o = device.alloc_zeros::<bf16>($dst_el)?;
                $func::<bf16>($handle, x_ptr, w_ptr, o.as_ptr() as *mut _, $($arg),*)?;
                RocmStorageSlice::BF16(o)
            }
            (RocmStorageSlice::F64(s), RocmStorageSlice::F64(w)) => {
                let x_ptr = unsafe { s.offset_ptr($l.start_offset()) } as *mut _;
                let w_ptr = unsafe { w.offset_ptr($kernel_l.start_offset()) } as *mut _;
                let o = device.alloc_zeros::<f64>($dst_el)?;
                $func::<f64>($handle, x_ptr, w_ptr, o.as_ptr() as *mut _, $($arg),*)?;
                RocmStorageSlice::F64(o)
            }
            _ => return Err(crate::Error::Msg(
                "conv only supports f32, f16, bf16, f64 for ROCm".to_string(),
            )),
        };
        Ok(Self { slice, device })
    }};
}

macro_rules! cast_launch {
    ($dev:expr, $grid:expr, $block:expr, $el:expr, $dims_len:expr, $ds_ptr:expr, $src_ptr:expr, $src_dtype:expr, $rust_type:ty, $variant:ident) => {{
        let out = $dev.alloc::<$rust_type>($el)?;
        let out_ptr = out.as_ptr() as *mut std::ffi::c_void;
        let func_name = format!("cast_{}_{}", $src_dtype.as_str(), stringify!($rust_type));
        unsafe {
            launch_kernel(
                &$dev,
                hanzo_rocm_kernels::kernel::CastKernel::NAME,
                hanzo_rocm_kernels::kernel::CastKernel::CODE,
                &func_name,
                $grid,
                $block,
                &mut [
                    &$el as *const usize as *mut std::ffi::c_void,
                    &$dims_len as *const usize as *mut std::ffi::c_void,
                    (&$ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&$src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }
        RocmStorageSlice::$variant(out)
    }};
}

pub fn kernel_name<T: Copy + Send + Sync + 'static>(kernel: &str) -> String {
    let type_name = std::any::type_name::<T>();
    let suffix = if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f64") {
        "f64"
    } else if type_name.contains("u8") {
        "u8"
    } else if type_name.contains("u32") {
        "u32"
    } else if type_name.contains("i64") {
        "i64"
    } else if type_name.contains("bf16") {
        "bf16"
    } else if type_name.contains("f16") {
        "f16"
    } else if type_name.contains("i16") {
        "i16"
    } else if type_name.contains("i32") {
        "i32"
    } else {
        panic!("Unsupported dtype for kernel: {}", type_name)
    };
    format!("{}_{}", kernel, suffix)
}

pub fn launch_config(num_elems: usize) -> (rocm_rs::hip::Dim3, rocm_rs::hip::Dim3) {
    const BLOCK_SIZE: u32 = 256;
    let num_blocks = ((num_elems as u32) + BLOCK_SIZE - 1) / BLOCK_SIZE;
    let grid_dim = num_blocks.min(65535);
    (
        rocm_rs::hip::Dim3::from(grid_dim),
        rocm_rs::hip::Dim3::from(BLOCK_SIZE),
    )
}

/// The set of GGML quant weight types the UNIFIED decode core (`qmatvec_core<WTYPE>` in
/// quant.hip) supports. ONE enum row + ONE `decode_kernel` arm + ONE `decode_block<WTYPE>` +
/// `qdw_traits<WTYPE>` in the .hip = a fully wired type, for BOTH decode and MoE. No per-quant
/// kernel, no per-quant launcher: `matvec_quant` reads the (type, activation) pair off this enum
/// and dispatches the single core's f16/bf16 entry point. Mirrors the CPU `for_each_quant!` table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // intentionally mirror GgmlDType variant spelling (Q8_0/IQ4_XS/TQ2_0)
pub enum RocmQuantType {
    Q8_0,
    Q4_0,
    Q4K,
    Q6K,
    IQ4_XS,
    TQ2_0,
}

impl RocmQuantType {
    /// Map a `GgmlDType` to a unified-core type, or `None` if no `decode_block<WTYPE>` is wired
    /// yet (caller falls back to dequant). This is the ONE place new types get recognized.
    pub fn from_ggml(dt: crate::quantized::GgmlDType) -> Option<Self> {
        use crate::quantized::GgmlDType as G;
        Some(match dt {
            G::Q8_0 => Self::Q8_0,
            G::Q4_0 => Self::Q4_0,
            G::Q4K => Self::Q4K,
            G::Q6K => Self::Q6K,
            G::IQ4_XS => Self::IQ4_XS,
            G::TQ2_0 => Self::TQ2_0,
            _ => return None,
        })
    }

    /// Elements per block (must divide `k`). Matches `qdw_traits<WTYPE>::ELEMS` in quant.hip.
    pub fn block_elems(self) -> usize {
        match self {
            Self::Q8_0 | Self::Q4_0 => 32,
            Self::Q4K | Self::Q6K | Self::IQ4_XS | Self::TQ2_0 => 256,
        }
    }

    /// On-disk bytes per block. Matches `qdw_traits<WTYPE>::BYTES` in quant.hip and the GGML
    /// block byte size. Used to stride the MoE expert bank.
    pub fn block_bytes(self) -> usize {
        match self {
            Self::Q8_0 => 34,
            Self::Q4_0 => 18,
            Self::Q4K => 144,
            Self::Q6K => 210,
            Self::IQ4_XS => 136,
            Self::TQ2_0 => 66,
        }
    }

    /// Whether the weight dequant is symmetric (`val = scale*q`, no per-block min). Mirrors
    /// `wt_traits<WTYPE>::SYMMETRIC` in quant.hip: the ASYMMETRIC types (Q4_K, and later Q5_K)
    /// carry a per-sub-block min, so their prefill GEMM also reads the q8_1 activation block-sum
    /// (`quantize_q8_1`) for the `-dmin*m*d_x*sum` bias term; symmetric types never need it.
    pub fn symmetric(self) -> bool {
        !matches!(self, Self::Q4K)
    }

    /// The UNIFIED prefill GEMM entry point for this type (int8 WMMA, `qmmq_core<WTYPE>` in
    /// quant.hip). EVERY one of these is the SAME core with a different WTYPE -- one core, per-type
    /// decode + a symmetric/asymmetric flag, NO per-quant kernel. Output is f16 (the prefill dtype).
    fn prefill_kernel(self) -> &'static str {
        match self {
            Self::Q8_0 => "qmmq_q8_0_f16",
            Self::Q4_0 => "qmmq_q4_0_f16",
            Self::Q4K => "qmmq_q4k_f16",
            Self::Q6K => "qmmq_q6k_f16",
            Self::IQ4_XS => "qmmq_iq4xs_f16",
            Self::TQ2_0 => "qmmq_tq2_0_f16",
        }
    }

    /// The unified-core entry point name for this (type, activation-dtype) pair. EVERY one of these
    /// is `qmatvec_core<WTYPE,XT>` with a different WTYPE -- there is exactly one core.
    fn decode_kernel(self, f16: bool) -> &'static str {
        match (self, f16) {
            (Self::Q8_0, true) => "qmatvecu_q8_0_f16",
            (Self::Q8_0, false) => "qmatvecu_q8_0_bf16",
            (Self::Q4_0, true) => "qmatvecu_q4_0_f16",
            (Self::Q4_0, false) => "qmatvecu_q4_0_bf16",
            (Self::Q4K, true) => "qmatvecu_q4k_f16",
            (Self::Q4K, false) => "qmatvecu_q4k_bf16",
            (Self::Q6K, true) => "qmatvecu_q6k_f16",
            (Self::Q6K, false) => "qmatvecu_q6k_bf16",
            (Self::IQ4_XS, true) => "qmatvecu_iq4xs_f16",
            (Self::IQ4_XS, false) => "qmatvecu_iq4xs_bf16",
            (Self::TQ2_0, true) => "qmatvecu_tq2_0_f16",
            (Self::TQ2_0, false) => "qmatvecu_tq2_0_bf16",
        }
    }

    /// The unified indexed-MoE decode entry point for this (type, activation-dtype) pair. Twin of
    /// `decode_kernel`: every symbol is `moe_qmatvec_core<WTYPE,XT>` with a different WTYPE -- one
    /// batched on-device-ids launch over all routed slots (experts on grid.y), capture-clean. This
    /// REPLACES the per-expert host launch loop for the non-Q4_K types.
    fn moe_decode_kernel(self, f16: bool) -> &'static str {
        match (self, f16) {
            (Self::Q8_0, true) => "moe_qmatvecu_q8_0_f16",
            (Self::Q8_0, false) => "moe_qmatvecu_q8_0_bf16",
            (Self::Q4_0, true) => "moe_qmatvecu_q4_0_f16",
            (Self::Q4_0, false) => "moe_qmatvecu_q4_0_bf16",
            (Self::Q4K, true) => "moe_qmatvecu_q4k_f16",
            (Self::Q4K, false) => "moe_qmatvecu_q4k_bf16",
            (Self::Q6K, true) => "moe_qmatvecu_q6k_f16",
            (Self::Q6K, false) => "moe_qmatvecu_q6k_bf16",
            (Self::IQ4_XS, true) => "moe_qmatvecu_iq4xs_f16",
            (Self::IQ4_XS, false) => "moe_qmatvecu_iq4xs_bf16",
            (Self::TQ2_0, true) => "moe_qmatvecu_tq2_0_f16",
            (Self::TQ2_0, false) => "moe_qmatvecu_tq2_0_bf16",
        }
    }
}

impl RocmDevice {
    /// UNIFIED native quant matvec (decode): `y[n] = W_q[n,k] · x[k]`, reading the GGML block
    /// format straight from VRAM for ANY wired quant type via the SINGLE `qmatvec_core<WTYPE>`.
    /// `qt` selects the per-block decode (the only thing that varies); `wq` = the raw GGML weight
    /// bytes (u8); `x` = f16 OR bf16 `[k]` contiguous; returns the SAME float dtype as `x` (`[n]`).
    /// One warp per output row, whole warp cooperates per block. `k` must be a multiple of the
    /// type's block size. This REPLACES the old per-quant `matvec_q8_0`/`matvec_q4k` launchers --
    /// adding a quant type is one `RocmQuantType` row, not a new launcher.
    ///
    /// Accepting bf16 directly (the model's working dtype) lets the decode path skip the
    /// bf16->f32->f16 cast detour entirely: 0 cast launches per matvec instead of 3.
    pub fn matvec_quant(
        &self,
        qt: RocmQuantType,
        wq: &RocmStorage,
        x: &RocmStorage,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let elems = qt.block_elems();
        if k % elems != 0 {
            crate::bail!("matvec_quant({qt:?}): k={k} not a multiple of block size {elems}");
        }
        let wq_ptr = match &wq.slice {
            RocmStorageSlice::U8(m) => m.as_ptr(),
            _ => crate::bail!("matvec_quant: weights must be u8 (raw GGML block bytes)"),
        };
        // Q4_K FAST PATH: int8 dp4a decode (faithful port of llama.cpp vec_dot_q4_K_q8_1). Pre-
        // quantize the activation row to q8_1 (int8 + per-32-block f16 scale), then run v_dot4 against
        // the weight nibbles -- ~2x the scalar-float dequant path. Set HANZO_Q4K_FALLBACK=1 to force
        // the proven scalar `qmatvecu_q4k_*` core instead (A/B + correctness reference).
        if qt == RocmQuantType::Q4K
            && std::env::var_os("HANZO_Q4K_FALLBACK").is_none()
        {
            return self.matvec_q4k_dp4a(wq_ptr, x, n, k);
        }
        let nrows = n as i32;
        let ncols = k as i32;
        // 256 threads = 8 warps = 8 output rows per block (one warp per row).
        let grid = rocm_rs::hip::Dim3::from((n.div_ceil(8)) as u32);
        let block = rocm_rs::hip::Dim3::from(256u32);

        macro_rules! launch_matvec {
            ($variant:ident, $ty:ty, $f16:expr) => {{
                let func = qt.decode_kernel($f16);
                let x_ptr = match &x.slice {
                    RocmStorageSlice::$variant(m) => m.as_ptr(),
                    _ => unreachable!(),
                };
                let out = self.alloc::<$ty>(n)?;
                let out_ptr = out.as_ptr();
                unsafe {
                    launch_kernel(
                        self,
                        QuantKernel::NAME,
                        QuantKernel::CODE,
                        func,
                        grid,
                        block,
                        &mut [
                            &nrows as *const i32 as *mut std::ffi::c_void,
                            &ncols as *const i32 as *mut std::ffi::c_void,
                            (&wq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok(RocmStorage {
                    slice: RocmStorageSlice::$variant(out),
                    device: self.clone(),
                })
            }};
        }

        match &x.slice {
            RocmStorageSlice::F16(_) => launch_matvec!(F16, f16, true),
            RocmStorageSlice::BF16(_) => launch_matvec!(BF16, bf16, false),
            other => crate::bail!(
                "matvec_quant: activations must be f16 or bf16, got {:?}",
                other.dtype()
            ),
        }
    }

    /// Q4_K decode via int8 dp4a -- faithful port of llama.cpp `vec_dot_q4_K_q8_1` (+ `_impl_vmmq`).
    /// Two launches: (1) `quantize_q8_1[_bf16]` quantizes the activation row `x[k]` to q8_1 (int8 `xq`
    /// + per-32-block f16 scale `xd`; the q8_1 `s`/sum is recomputed in-kernel by the dp4a so `xs` is
    /// discarded here), (2) `qmatvec_q4k_dp4a_{f16,bf16}` dots each weight nibble against the q8_1
    /// int8 via `v_dot4` and scales. `xq`/`xd` own their VRAM until the end of this fn (past the
    /// stream-ordered launches). Output dtype = `x`'s dtype.
    fn matvec_q4k_dp4a(
        &self,
        wq_ptr: *mut std::ffi::c_void,
        x: &RocmStorage,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let nblk = k / 32;
        // FUSED single-launch path (HANZO_Q4K_FUSED=1): quantize the activation to q8_1 in shared
        // memory AND dp4a in ONE kernel, skipping the separate quantize_q8_1 launch + xq/xd/xs
        // scratch allocs + inter-launch sync. Wins on the small overhead-bound matvecs (k/v proj).
        // Shared = k bytes int8 (16B-padded) + k/32 f16 scales.
        if std::env::var_os("HANZO_Q4K_FUSED").is_some() {
            let nrows = n as i32;
            let ncols = k as i32;
            let grid = rocm_rs::hip::Dim3::from((n.div_ceil(8)) as u32);
            let block = rocm_rs::hip::Dim3::from(256u32);
            let shmem = ((((k + 15) & !15) + nblk * std::mem::size_of::<f16>()) as u32 + 15) & !15;
            macro_rules! launch_fused {
                ($variant:ident, $ty:ty, $func:expr) => {{
                    let x_ptr = match &x.slice {
                        RocmStorageSlice::$variant(s) => s.as_ptr(),
                        _ => unreachable!(),
                    };
                    let out = self.alloc::<$ty>(n)?;
                    let out_ptr = out.as_ptr();
                    unsafe {
                        launch_kernel_shmem(
                            self,
                            QuantKernel::NAME,
                            QuantKernel::CODE,
                            $func,
                            grid,
                            block,
                            shmem,
                            &mut [
                                &nrows as *const i32 as *mut std::ffi::c_void,
                                &ncols as *const i32 as *mut std::ffi::c_void,
                                (&wq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                                (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                                (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            ],
                        )?;
                    }
                    Ok(RocmStorage {
                        slice: RocmStorageSlice::$variant(out),
                        device: self.clone(),
                    })
                }};
            }
            return match &x.slice {
                RocmStorageSlice::F16(_) => launch_fused!(F16, f16, "qmatvec_q4k_dp4a_fused_f16"),
                RocmStorageSlice::BF16(_) => launch_fused!(BF16, bf16, "qmatvec_q4k_dp4a_fused_bf16"),
                other => crate::bail!(
                    "matvec_q4k_dp4a fused: activations must be f16 or bf16, got {:?}",
                    other.dtype()
                ),
            };
        }
        // Quantize the single activation row (m=1) to q8_1. Dispatch the quant kernel on x's dtype so
        // bf16 activations stay bf16 end-to-end (no bf16->f16 cast detour); both emit the SAME int8
        // xq + f16 xd. xs (the int block-sum) is allocated but unused -- the dp4a recomputes the sum.
        let xq = self.alloc::<u8>(k)?;
        let xd = self.alloc::<f16>(nblk)?;
        let xs = self.alloc::<i32>(nblk)?;
        let xq_ptr = xq.as_ptr();
        let xd_ptr = xd.as_ptr();
        let xs_ptr = xs.as_ptr();
        let mi = 1i32;
        let ki = k as i32;
        let qgrid = rocm_rs::hip::Dim3::from((nblk.div_ceil(8)) as u32);
        let qblock = rocm_rs::hip::Dim3::from(256u32);
        let (quant_func, x_ptr) = match &x.slice {
            RocmStorageSlice::F16(s) => ("quantize_q8_1", s.as_ptr()),
            RocmStorageSlice::BF16(s) => ("quantize_q8_1_bf16", s.as_ptr()),
            other => crate::bail!(
                "matvec_q4k_dp4a: activations must be f16 or bf16, got {:?}",
                other.dtype()
            ),
        };
        unsafe {
            launch_kernel(
                self,
                QuantKernel::NAME,
                QuantKernel::CODE,
                quant_func,
                qgrid,
                qblock,
                &mut [
                    &mi as *const i32 as *mut std::ffi::c_void,
                    &ki as *const i32 as *mut std::ffi::c_void,
                    (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xd_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        // dp4a matvec: FAITHFUL port of llama.cpp mul_mat_vec_q launch tiling (mmvq.cu) for the
        // decode (ncols_dst=1) case. The tiled kernel makes 16 lanes cooperate per Q4_K super-block
        // (qi/vdr = QI4_K/VDR = 32/2 = 16), so all 32 lanes stay busy even for the small-nblocks
        // decode shapes here (k=2048 -> nblocks=8; the old whole-super-block-per-lane kernel left
        // 24/32 lanes idle). Each warp computes ROWS output rows (Q4K_TILED_ROWS in quant.hip,
        // default 2), reusing the loaded q8_1 activation across rows; `nwarps` warps per block stack
        // the warp's row-groups. block = (32, nwarps, 1); grid.x = ceil(nrows / (nwarps*ROWS)).
        // ROWS is compiled into the kernel; nwarps is tuned here (HANZO_Q4K_NWARPS for A/B sweeps).
        const Q4K_TILED_ROWS: usize = 2; // must match Q4K_TILED_ROWS in quant.hip
        // Kernel choice. The original whole-super-block-per-lane kernel (each lane reads one
        // contiguous 144-byte super-block, fully unrolled) is the DEFAULT: on gfx1151 it sustains
        // up to ~460 GB/s via L1/L2 reuse of the small q8_1 activation and is 15-55% FASTER than the
        // llama.cpp-style 16-lanes-per-super-block tiling (which adds strided activation reads +
        // multi-row register pressure that this APU's cache already obviates). HANZO_Q4K_TILED=1
        // opts into the faithful llama tiling (kept for reference / larger-GPU portability).
        let tiled = std::env::var_os("HANZO_Q4K_TILED").is_some();
        let nwarps: usize = std::env::var("HANZO_Q4K_NWARPS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|&w| w >= 1 && w <= 16)
            .unwrap_or(1);
        let rows_per_block_total = nwarps * Q4K_TILED_ROWS;
        let nrows = n as i32;
        let ncols = k as i32;
        let (grid, block) = if tiled {
            (
                rocm_rs::hip::Dim3::from((n.div_ceil(rows_per_block_total)) as u32),
                rocm_rs::hip::Dim3::from((32u32, nwarps as u32, 1u32)),
            )
        } else {
            // Original: 256 threads = 8 warps = 8 rows/block, one warp per row, lane-strided blocks.
            (
                rocm_rs::hip::Dim3::from((n.div_ceil(8)) as u32),
                rocm_rs::hip::Dim3::from(256u32),
            )
        };
        macro_rules! launch_dp4a {
            ($variant:ident, $ty:ty, $func:expr) => {{
                let out = self.alloc::<$ty>(n)?;
                let out_ptr = out.as_ptr();
                unsafe {
                    launch_kernel(
                        self,
                        QuantKernel::NAME,
                        QuantKernel::CODE,
                        $func,
                        grid,
                        block,
                        &mut [
                            &nrows as *const i32 as *mut std::ffi::c_void,
                            &ncols as *const i32 as *mut std::ffi::c_void,
                            (&wq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&xq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&xd_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok(RocmStorage {
                    slice: RocmStorageSlice::$variant(out),
                    device: self.clone(),
                })
            }};
        }
        // Keep xq/xd/xs alive across the second launch (stream-ordered): bind after the macro so the
        // borrow checker sees them used past the kernel that reads them.
        let result = match (&x.slice, tiled) {
            (RocmStorageSlice::F16(_), true) => launch_dp4a!(F16, f16, "qmatvec_q4k_dp4a_tiled_f16"),
            (RocmStorageSlice::BF16(_), true) => launch_dp4a!(BF16, bf16, "qmatvec_q4k_dp4a_tiled_bf16"),
            (RocmStorageSlice::F16(_), false) => launch_dp4a!(F16, f16, "qmatvec_q4k_dp4a_f16"),
            (RocmStorageSlice::BF16(_), false) => launch_dp4a!(BF16, bf16, "qmatvec_q4k_dp4a_bf16"),
            _ => unreachable!(),
        };
        drop((xq, xd, xs));
        result
    }

    /// Native ROCm fused indexed MoE: grouped quant matvec where each routed slot is dispatched
    /// through the SAME unified `qmatvec_core<WTYPE>` as ordinary decode -- so MoE works for EVERY
    /// wired quant expert automatically, with NO MoE-per-quant kernel. The GGML expert bank
    /// [E,n,k] is resident in VRAM (`wbank`, raw block bytes, uploaded once); `x` is the [nrows,k]
    /// routed-activation matrix (f16 or bf16); `ids[s]` is slot s's expert. For each slot the core
    /// runs on that expert's byte-slice of the bank into output row s -- each output row is written
    /// exactly once, so there is no scatter/index_add (which ROCm lacks). Returns [nrows,n] in x's
    /// dtype. This is the "dispatch each expert through the one core" path the decode core enables.
    pub fn moe_matvec_quant(
        &self,
        qt: RocmQuantType,
        wbank: &RocmStorage,
        x: &RocmStorage,
        ids: &RocmStorage,
        nrows: usize,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let elems = qt.block_elems();
        if k % elems != 0 {
            crate::bail!("moe_matvec_quant({qt:?}): k={k} not a multiple of block size {elems}");
        }
        let wbank_mem = match &wbank.slice {
            RocmStorageSlice::U8(m) => m,
            _ => crate::bail!("moe_matvec_quant: weight bank must be u8 (raw GGML block bytes)"),
        };
        let ids_ptr = match &ids.slice {
            RocmStorageSlice::U32(m) => m.as_ptr(),
            _ => crate::bail!("moe_matvec_quant: ids must be u32 on device"),
        };
        // Per-expert byte stride in the bank: one expert is an [n,k] weight = n*(k/elems) blocks,
        // each block `block_bytes` on disk. Validate the bank is an exact multiple of that.
        let nblk = k / elems;
        let expert_bytes = n * nblk * qt.block_bytes();
        if expert_bytes == 0 || wbank_mem.size() % expert_bytes != 0 {
            crate::bail!(
                "moe_matvec_quant: bank size {} not a multiple of expert bytes {expert_bytes}",
                wbank_mem.size()
            );
        }
        // Q4_K: ONE batched int8 dp4a launch over all routed slots (experts on grid.y), activation
        // quantized to q8_1 once -- collapses the per-expert scalar launch loop into the same dp4a
        // roofline as non-MoE decode (mirrors CUDA indexed_moe_forward_q4k_q8_1).
        if matches!(qt, RocmQuantType::Q4K) {
            return self.moe_matvec_q4k_dp4a(wbank, x, ids, nrows, n, k);
        }

        // Other wired quant types: ONE batched `moe_qmatvec_core<WTYPE>` launch over all routed
        // slots. Experts on grid.y (slot s = blockIdx.y), expert id read ON-DEVICE (ids_ptr) and the
        // bank offset by ids[s] IN-KERNEL -- no per-slot host loop, no host ids round-trip, so the
        // forward stays HIP-graph-capture-clean. grid.x covers the n output rows (one warp per row,
        // one full [n] output vector per slot). This is the non-Q4_K twin of moe_matvec_q4k_dp4a.
        let n_i = n as i32;
        let ncols = k as i32;
        let nslots = nrows as i32;
        let grid = rocm_rs::hip::Dim3::from(((n.div_ceil(8)) as u32, nrows as u32, 1u32));
        let block = rocm_rs::hip::Dim3::from(256u32);
        let wbank_ptr = wbank_mem.as_ptr();

        macro_rules! launch_moe {
            ($variant:ident, $ty:ty, $f16:expr) => {{
                let func = qt.moe_decode_kernel($f16);
                let x_ptr = match &x.slice {
                    RocmStorageSlice::$variant(m) => m.as_ptr(),
                    _ => unreachable!(),
                };
                let out = self.alloc::<$ty>(nrows * n)?;
                let out_ptr = out.as_ptr();
                unsafe {
                    launch_kernel(
                        self,
                        QuantKernel::NAME,
                        QuantKernel::CODE,
                        func,
                        grid,
                        block,
                        &mut [
                            &n_i as *const i32 as *mut std::ffi::c_void,
                            &ncols as *const i32 as *mut std::ffi::c_void,
                            &nslots as *const i32 as *mut std::ffi::c_void,
                            (&wbank_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&ids_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok(RocmStorage {
                    slice: RocmStorageSlice::$variant(out),
                    device: self.clone(),
                })
            }};
        }

        match &x.slice {
            RocmStorageSlice::F16(_) => launch_moe!(F16, f16, true),
            RocmStorageSlice::BF16(_) => launch_moe!(BF16, bf16, false),
            other => crate::bail!(
                "moe_matvec_quant: activations must be f16 or bf16, got {:?}",
                other.dtype()
            ),
        }
    }

    /// Q4_K batched indexed-MoE decode: quantize the [nrows,k] routed activations to q8_1 ONCE, then
    /// one `moe_qmatvec_q4k_dp4a_*` launch with the routed experts on grid.y (expert = ids[s] per
    /// slot). One well-occupied grid + int8 dp4a, vs the per-expert scalar launch loop. Returns
    /// [nrows,n] in x's dtype; routing + bank-residency are the caller's job.
    pub(crate) fn moe_matvec_q4k_dp4a(
        &self,
        wbank: &RocmStorage,
        x: &RocmStorage,
        ids_dev: &RocmStorage,
        nrows: usize,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let wbank_mem = match &wbank.slice {
            RocmStorageSlice::U8(m) => m,
            _ => crate::bail!("moe_matvec_q4k_dp4a: weight bank must be u8 (raw GGML blocks)"),
        };
        let ids_ptr = match &ids_dev.slice {
            RocmStorageSlice::U32(m) => m.as_ptr(),
            _ => crate::bail!("moe_matvec_q4k_dp4a: ids must be u32 on device"),
        };
        let nblk32 = k / 32;
        // (1) Quantize the [nrows,k] activations to q8_1 (int8 xq + per-32-block f16 xd). `xs` is the
        // block-sum the quant kernel writes; the dp4a recomputes it in-kernel, so it stays unused.
        let xq = self.alloc::<u8>(nrows * k)?;
        let xd = self.alloc::<f16>(nrows * nblk32)?;
        let xs = self.alloc::<i32>(nrows * nblk32)?;
        let xq_ptr = xq.as_ptr();
        let xd_ptr = xd.as_ptr();
        let xs_ptr = xs.as_ptr();
        let mrows = nrows as i32;
        let ki = k as i32;
        let qgrid = rocm_rs::hip::Dim3::from(((nrows * nblk32).div_ceil(8)) as u32);
        let qblock = rocm_rs::hip::Dim3::from(256u32);
        let (quant_func, x_ptr) = match &x.slice {
            RocmStorageSlice::F16(s) => ("quantize_q8_1", s.as_ptr()),
            RocmStorageSlice::BF16(s) => ("quantize_q8_1_bf16", s.as_ptr()),
            other => crate::bail!(
                "moe_matvec_q4k_dp4a: activations must be f16 or bf16, got {:?}",
                other.dtype()
            ),
        };
        unsafe {
            launch_kernel(
                self,
                QuantKernel::NAME,
                QuantKernel::CODE,
                quant_func,
                qgrid,
                qblock,
                &mut [
                    &mrows as *const i32 as *mut std::ffi::c_void,
                    &ki as *const i32 as *mut std::ffi::c_void,
                    (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xd_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }
        // (2) ONE batched dp4a launch: grid.x = expert output rows, grid.y = routed slot. Expert ids
        // are read on-device (ids_ptr) -- no per-call host round-trip, which is the decode hot path.
        let n_i = n as i32;
        let ncols = k as i32;
        let nslots = nrows as i32;
        let grid = rocm_rs::hip::Dim3::from(((n.div_ceil(8)) as u32, nrows as u32, 1u32));
        let block = rocm_rs::hip::Dim3::from(256u32);
        let wbank_ptr = wbank_mem.as_ptr();
        macro_rules! launch_moe_dp4a {
            ($variant:ident, $ty:ty, $func:expr) => {{
                let out = self.alloc::<$ty>(nrows * n)?;
                let out_ptr = out.as_ptr();
                unsafe {
                    launch_kernel(
                        self,
                        QuantKernel::NAME,
                        QuantKernel::CODE,
                        $func,
                        grid,
                        block,
                        &mut [
                            &n_i as *const i32 as *mut std::ffi::c_void,
                            &ncols as *const i32 as *mut std::ffi::c_void,
                            &nslots as *const i32 as *mut std::ffi::c_void,
                            (&wbank_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&ids_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&xq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&xd_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok(RocmStorage {
                    slice: RocmStorageSlice::$variant(out),
                    device: self.clone(),
                })
            }};
        }
        let result = match &x.slice {
            RocmStorageSlice::F16(_) => launch_moe_dp4a!(F16, f16, "moe_qmatvec_q4k_dp4a_f16"),
            RocmStorageSlice::BF16(_) => launch_moe_dp4a!(BF16, bf16, "moe_qmatvec_q4k_dp4a_bf16"),
            _ => unreachable!(),
        };
        // Keep q8_1 scratch alive past the (stream-ordered) dp4a launch that reads it.
        drop((xq, xd, xs));
        result
    }

    /// Back-compat thin wrapper: Q8_0 decode via the unified core. Kept so existing callers/tests
    /// (`dev.matvec_q8_0(...)`) need no change; new code should call `matvec_quant`.
    pub fn matvec_q8_0(
        &self,
        wq: &RocmStorage,
        x: &RocmStorage,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        self.matvec_quant(RocmQuantType::Q8_0, wq, x, n, k)
    }

    /// Back-compat thin wrapper: Q4_K decode via the unified core. Kept so existing callers/tests
    /// (`dev.matvec_q4k(...)`) need no change; new code should call `matvec_quant`.
    pub fn matvec_q4k(
        &self,
        wq: &RocmStorage,
        x: &RocmStorage,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        self.matvec_quant(RocmQuantType::Q4K, wq, x, n, k)
    }

    /// Native Q8_0 quant GEMM (prefill): `Y[M,N] = X[M,K] (f16) * W[N,K]^T`, W kept Q8_0 in VRAM.
    /// One wave per 16x16 output tile, RDNA3 WMMA matrix cores, dequant-in-kernel. Returns f16 [M,N].
    pub fn qgemm_q8_0(
        &self,
        x: &RocmStorage,
        wq: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let x_ptr = match &x.slice {
            RocmStorageSlice::F16(s) => s.as_ptr(),
            _ => crate::bail!("qgemm_q8_0: activations must be f16"),
        };
        let wq_ptr = match &wq.slice {
            RocmStorageSlice::U8(s) => s.as_ptr(),
            _ => crate::bail!("qgemm_q8_0: weights must be u8 (raw Q8_0 bytes)"),
        };
        let out = self.alloc::<f16>(m * n)?;
        let out_ptr = out.as_ptr();
        let mi = m as i32;
        let ni = n as i32;
        let ki = k as i32;
        let ncol_tiles = n.div_ceil(64);
        let ncol_tiles_i = ncol_tiles as i32;
        let nrow_tiles = m.div_ceil(64);
        let grid = rocm_rs::hip::Dim3::from((ncol_tiles * nrow_tiles) as u32);
        let block = rocm_rs::hip::Dim3::from(128u32);
        unsafe {
            launch_kernel(
                self,
                QuantKernel::NAME,
                QuantKernel::CODE,
                "qgemm_q8_0_f16",
                grid,
                block,
                &mut [
                    &mi as *const i32 as *mut std::ffi::c_void,
                    &ni as *const i32 as *mut std::ffi::c_void,
                    &ki as *const i32 as *mut std::ffi::c_void,
                    &ncol_tiles_i as *const i32 as *mut std::ffi::c_void,
                    (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&wq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(RocmStorage {
            slice: RocmStorageSlice::F16(out),
            device: self.clone(),
        })
    }

    /// Quantize activations to symmetric int8 + per-32 f16 scale (llama's quantize_q8_1, symmetric).
    /// x[m,k] f16 -> (xq[m,k] int8 stored as u8, xd[m, k/32] f16).
    pub fn quantize_q8(
        &self,
        x: &RocmStorage,
        m: usize,
        k: usize,
    ) -> Result<(RocmStorage, RocmStorage)> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let x_ptr = match &x.slice {
            RocmStorageSlice::F16(s) => s.as_ptr(),
            _ => crate::bail!("quantize_q8: x must be f16"),
        };
        let nblk = k / 32;
        let xq = self.alloc::<u8>(m * k)?;
        let xd = self.alloc::<f16>(m * nblk)?;
        let xq_ptr = xq.as_ptr();
        let xd_ptr = xd.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let nwarps = m * nblk;
        let grid = rocm_rs::hip::Dim3::from((nwarps.div_ceil(8)) as u32);
        let block = rocm_rs::hip::Dim3::from(256u32);
        unsafe {
            launch_kernel(
                self,
                QuantKernel::NAME,
                QuantKernel::CODE,
                "quantize_q8",
                grid,
                block,
                &mut [
                    &mi as *const i32 as *mut std::ffi::c_void,
                    &ki as *const i32 as *mut std::ffi::c_void,
                    (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xd_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok((
            RocmStorage { slice: RocmStorageSlice::U8(xq), device: self.clone() },
            RocmStorage { slice: RocmStorageSlice::F16(xd), device: self.clone() },
        ))
    }

    /// Native Q8_0 × int8 GEMM (prefill) on the RDNA3 int8 matrix cores. Takes f16 activations `x`,
    /// `wq` = the Q8_0 weight bytes; returns f16 [m,n].
    ///
    /// DE-FUSED (matches llama's mul_mat_q): the activation is quantized to int8 ONCE up front via
    /// `quantize_q8` (-> int8 `xq` + per-32-block f16 `xd` in VRAM, llama's quantize_mmq_q8_1
    /// equivalent), then the gemm kernel consumes the pre-quantized `xq`/`xd`. Round 1 re-quantized
    /// `x` inside the kernel once per N-col-tile (~96x for a wide FFN); pre-quantizing removes that
    /// redundant activation traffic + absmax reductions and lowers the kernel's VGPR pressure.
    pub fn qmmq_q8_0(
        &self,
        x: &RocmStorage,
        wq: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        self.qmmq_quant(RocmQuantType::Q8_0, x, wq, m, n, k)
    }

    /// Native Q4_0 × int8 GEMM (prefill) on the RDNA3 int8 matrix cores. Q4_0 = 18-byte block
    /// (f16 d + 16 nibble-pairs; weight = (nibble - 8) * d). Uses the SAME symmetric int8 core /
    /// activation int8 quant as Q8_0 (scale = dA·dB); only the per-block weight decode differs
    /// (nibble-8 centering, done in-kernel). `wq` = the raw Q4_0 weight bytes.
    pub fn qmmq_q4_0(
        &self,
        x: &RocmStorage,
        wq: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        self.qmmq_quant(RocmQuantType::Q4_0, x, wq, m, n, k)
    }

    /// Q8_1-style activation quant (prefill ASYMMETRIC path): int8 `xq` + per-32-block f16 scale
    /// `xd` (identical to `quantize_q8`) PLUS the per-32-block int8 SUM `xs` (i32) -- llama's
    /// block_q8_1 `s`. The sum feeds the `-dmin*m*d_x*sum` min bias for asymmetric weights (Q4_K).
    pub fn quantize_q8_1(
        &self,
        x: &RocmStorage,
        m: usize,
        k: usize,
    ) -> Result<(RocmStorage, RocmStorage, RocmStorage)> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let x_ptr = match &x.slice {
            RocmStorageSlice::F16(s) => s.as_ptr(),
            _ => crate::bail!("quantize_q8_1: x must be f16"),
        };
        let nblk = k / 32;
        let xq = self.alloc::<u8>(m * k)?;
        let xd = self.alloc::<f16>(m * nblk)?;
        let xs = self.alloc::<i32>(m * nblk)?;
        let xq_ptr = xq.as_ptr();
        let xd_ptr = xd.as_ptr();
        let xs_ptr = xs.as_ptr();
        let mi = m as i32;
        let ki = k as i32;
        let nwarps = m * nblk;
        let grid = rocm_rs::hip::Dim3::from((nwarps.div_ceil(8)) as u32);
        let block = rocm_rs::hip::Dim3::from(256u32);
        unsafe {
            launch_kernel(
                self,
                QuantKernel::NAME,
                QuantKernel::CODE,
                "quantize_q8_1",
                grid,
                block,
                &mut [
                    &mi as *const i32 as *mut std::ffi::c_void,
                    &ki as *const i32 as *mut std::ffi::c_void,
                    (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xd_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok((
            RocmStorage { slice: RocmStorageSlice::U8(xq), device: self.clone() },
            RocmStorage { slice: RocmStorageSlice::F16(xd), device: self.clone() },
            RocmStorage { slice: RocmStorageSlice::I32(xs), device: self.clone() },
        ))
    }

    /// UNIFIED int8 quant GEMM (prefill) for ANY wired quant type via the SINGLE `qmmq_core<WTYPE>`
    /// in quant.hip -- ONE core, per-type in-kernel weight decode + a symmetric/asymmetric flag, NO
    /// per-quant kernel. Pre-quantizes the activation to int8 once (`quantize_q8`; ASYMMETRIC types
    /// use `quantize_q8_1` to also emit the q8_1 block-sum for the min bias), then launches the
    /// type's `qmmq_core<WTYPE>` entry point. Q8_0/Q4_0 take the proven symmetric path byte-for-byte
    /// (the asym branches `if constexpr`-elide; `xs` is passed but never read). Returns f16 [m,n].
    pub fn qmmq_quant(
        &self,
        qt: RocmQuantType,
        x: &RocmStorage,
        wq: &RocmStorage,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        if k % qt.block_elems() != 0 {
            crate::bail!("qmmq_quant({qt:?}): k={k} not a multiple of block size {}", qt.block_elems());
        }
        // Pre-quantize activations once. SYMMETRIC weights need only xq/xd; ASYMMETRIC (Q4_K) also
        // needs the per-block int8 sum xs (the q8_1 bias term). xq/xd/xs own their VRAM until the end
        // of this fn (past the stream-ordered launch), so the kernel reads valid memory. For the
        // symmetric path we still allocate a tiny 1-block dummy xs so a valid (unread) ptr is passed.
        let (xq, xd, xs_opt) = if qt.symmetric() {
            let (xq, xd) = self.quantize_q8(x, m, k)?;
            (xq, xd, None)
        } else {
            let (xq, xd, xs) = self.quantize_q8_1(x, m, k)?;
            (xq, xd, Some(xs))
        };
        // Dummy 1-elem xs for the symmetric path (the kernel's asym branch elides, so it is never
        // dereferenced; we just need a non-null device pointer to bind to the kernel parameter).
        let xs_dummy = if xs_opt.is_none() { Some(self.alloc::<i32>(1)?) } else { None };
        let xq_ptr = match &xq.slice {
            RocmStorageSlice::U8(s) => s.as_ptr(),
            _ => crate::bail!("qmmq_quant: xq must be u8"),
        };
        let xd_ptr = match &xd.slice {
            RocmStorageSlice::F16(s) => s.as_ptr(),
            _ => crate::bail!("qmmq_quant: xd must be f16"),
        };
        let xs_ptr = match (&xs_opt, &xs_dummy) {
            (Some(xs), _) => match &xs.slice {
                RocmStorageSlice::I32(s) => s.as_ptr(),
                _ => crate::bail!("qmmq_quant: xs must be i32"),
            },
            (None, Some(d)) => d.as_ptr(),
            _ => unreachable!(),
        };
        let wq_ptr = match &wq.slice {
            RocmStorageSlice::U8(s) => s.as_ptr(),
            _ => crate::bail!("qmmq_quant: wq must be u8 (raw GGML block bytes)"),
        };
        let out = self.alloc::<f16>(m * n)?;
        let out_ptr = out.as_ptr();
        let mi = m as i32;
        let ni = n as i32;
        let ki = k as i32;
        // 128x128 tile, 16 warps (512 threads, 4x4 grid of 32x32 sub-tiles). Kernel/launcher config
        // must stay in sync.
        let ncol_tiles = n.div_ceil(128);
        let ncol_tiles_i = ncol_tiles as i32;
        let nrow_tiles = m.div_ceil(128);
        let grid = rocm_rs::hip::Dim3::from((ncol_tiles * nrow_tiles) as u32);
        let block = rocm_rs::hip::Dim3::from(512u32);
        unsafe {
            launch_kernel(
                self,
                QuantKernel::NAME,
                QuantKernel::CODE,
                qt.prefill_kernel(),
                grid,
                block,
                &mut [
                    &mi as *const i32 as *mut std::ffi::c_void,
                    &ni as *const i32 as *mut std::ffi::c_void,
                    &ki as *const i32 as *mut std::ffi::c_void,
                    &ncol_tiles_i as *const i32 as *mut std::ffi::c_void,
                    (&xq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xd_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&wq_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&xs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(RocmStorage { slice: RocmStorageSlice::F16(out), device: self.clone() })
    }
}

unsafe fn launch_kernel(
    dev: &RocmDevice,
    module_name: &'static str,
    module_source: &'static str,
    func_name: &str,
    grid: rocm_rs::hip::Dim3,
    block: rocm_rs::hip::Dim3,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    let raw = {
        let kernel_manager = dev
            .kernel_manager()
            .lock()
            .map_err(|_| crate::Error::Msg("Failed to lock kernel manager".to_string()))?;
        kernel_manager
            .get_func_raw(module_name, module_source, func_name)
            .map_err(|e| crate::Error::Msg(e.to_string()))?
    };
    let kernel = rocm_rs::hip::Function::from_raw(raw as _);
    kernel
        .launch(grid, block, 0, Some(&dev.stream), args)
        .map_err(|e| crate::Error::Msg(format!("Kernel launch failed: {}", e)))
}

#[allow(clippy::too_many_arguments)]
unsafe fn launch_kernel_shmem(
    dev: &RocmDevice,
    module_name: &'static str,
    module_source: &'static str,
    func_name: &str,
    grid: rocm_rs::hip::Dim3,
    block: rocm_rs::hip::Dim3,
    shared_mem: u32,
    args: &mut [*mut std::ffi::c_void],
) -> Result<()> {
    let raw = {
        let kernel_manager = dev
            .kernel_manager()
            .lock()
            .map_err(|_| crate::Error::Msg("Failed to lock kernel manager".to_string()))?;
        kernel_manager
            .get_func_raw(module_name, module_source, func_name)
            .map_err(|e| crate::Error::Msg(e.to_string()))?
    };
    let kernel = rocm_rs::hip::Function::from_raw(raw as _);
    kernel
        .launch(grid, block, shared_mem, Some(&dev.stream), args)
        .map_err(|e| crate::Error::Msg(format!("Kernel launch failed: {}", e)))
}

impl RocmStorage {
    /// Argsort along the last dim: returns u32 indices that sort each row.
    /// Single-block bitonic sort (one block per row); used by Tensor::arg_sort_last_dim.
    pub fn asort(&self, layout: &Layout, asc: bool, last_dim: usize) -> Result<Self> {
        use hanzo_rocm_kernels::kernel::{KernelSource, SortKernel};
        let device = self.device.clone();
        let elem_count = layout.shape().elem_count();
        let ncols = last_dim as i32;
        let nrows = (elem_count / last_dim) as u32;
        let mut ncols_pad = 1usize;
        while ncols_pad < last_dim {
            ncols_pad *= 2;
        }
        let block_dim = ncols_pad.min(1024) as u32;
        let shared_mem = (ncols_pad * std::mem::size_of::<u32>()) as u32;
        let ncols_pad_i = ncols_pad as i32;

        let suffix = match &self.slice {
            RocmStorageSlice::F32(_) => "f32",
            RocmStorageSlice::F64(_) => "f64",
            RocmStorageSlice::U8(_) => "u8",
            RocmStorageSlice::U32(_) => "u32",
            RocmStorageSlice::I64(_) => "i64",
            RocmStorageSlice::BF16(_) => "bf16",
            RocmStorageSlice::F16(_) => "f16",
            _ => crate::bail!("unsupported dtype for argsort on rocm"),
        };
        let func_name = format!("asort_{}_{}", if asc { "asc" } else { "desc" }, suffix);

        let dst = device.alloc::<u32>(elem_count)?;
        let x_ptr = unsafe { self.slice.offset_ptr(layout.start_offset()) };
        let dst_ptr = dst.as_ptr();
        let grid = rocm_rs::hip::Dim3::from(nrows);
        let block = rocm_rs::hip::Dim3::from(block_dim);
        unsafe {
            launch_kernel_shmem(
                &device,
                SortKernel::NAME,
                SortKernel::CODE,
                &func_name,
                grid,
                block,
                shared_mem,
                &mut [
                    (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&dst_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    &ncols as *const i32 as *mut std::ffi::c_void,
                    &ncols_pad_i as *const i32 as *mut std::ffi::c_void,
                ],
            )?;
        }
        Ok(Self {
            slice: RocmStorageSlice::U32(dst),
            device,
        })
    }

    /// Fused RMSNorm along the last dim: `out[i] = x[i] * rsqrt(mean(x^2)+eps) * alpha[i]`.
    /// One block per row (row = elem_count / last_dim); f32 accumulation regardless of dtype.
    /// `alpha` must be a contiguous vector of length `last_dim` on this device. Supports
    /// F16 and F32 (CustomOp2 falls back to rms_norm_slow for other dtypes). Mirrors the
    /// CUDA `rmsnorm` launcher's ABI/block-size policy and reuses the `ReduceKernel` source.
    pub fn rms_norm(
        &self,
        x_layout: &Layout,
        alpha: &RocmStorage,
        alpha_layout: &Layout,
        eps: f32,
    ) -> Result<Self> {
        use hanzo_rocm_kernels::kernel::{KernelSource, ReduceKernel};
        let x = self;
        if !x_layout.is_contiguous() || !alpha_layout.is_contiguous() {
            crate::bail!("rms_norm on rocm requires contiguous inputs");
        }
        let device = self.device.clone();
        let dims = x_layout.shape().dims();
        let elem_count = x_layout.shape().elem_count();
        let last_dim = dims[dims.len() - 1];
        if last_dim == 0 {
            crate::bail!("rms_norm: last dim must be non-zero");
        }
        let n_rows = (elem_count / last_dim) as u32;
        let n_cols = last_dim as i32;
        // Match the CUDA launcher: a single warp for narrow rows, a full block of 1024
        // otherwise. Both are exact multiples of WARP_SIZE so the cross-warp s_sum is fully
        // populated; the kernel is also hardened against partial warps.
        let block_size: i32 = if last_dim < 1024 { 32 } else { 1024 };

        let x_off = x_layout.start_offset();
        let alpha_off = alpha_layout.start_offset();

        macro_rules! launch_rmsnorm {
            ($variant:ident, $ty:ty, $func:literal) => {{
                let x_mem = match &x.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => unreachable!(),
                };
                let alpha_mem = match &alpha.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!(
                        "rms_norm: alpha dtype {:?} must match x dtype {:?}",
                        alpha.slice.dtype(),
                        x.slice.dtype()
                    ),
                };
                let out = device.alloc::<$ty>(elem_count)?;
                let x_ptr = unsafe { x_mem.offset_ptr(x_off) };
                let alpha_ptr = unsafe { alpha_mem.offset_ptr(alpha_off) };
                let out_ptr = out.as_ptr();
                let grid = rocm_rs::hip::Dim3::from(n_rows);
                let block = rocm_rs::hip::Dim3::from(block_size as u32);
                unsafe {
                    launch_kernel(
                        &device,
                        ReduceKernel::NAME,
                        ReduceKernel::CODE,
                        $func,
                        grid,
                        block,
                        &mut [
                            (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&alpha_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            &n_cols as *const i32 as *mut std::ffi::c_void,
                            &block_size as *const i32 as *mut std::ffi::c_void,
                            &eps as *const f32 as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok(Self {
                    slice: RocmStorageSlice::$variant(out),
                    device: device.clone(),
                })
            }};
        }

        match &x.slice {
            RocmStorageSlice::F32(_) => launch_rmsnorm!(F32, f32, "rmsnorm_f32"),
            RocmStorageSlice::F16(_) => launch_rmsnorm!(F16, f16, "rmsnorm_f16"),
            other => crate::bail!("rms_norm on rocm unsupported dtype {:?}", other.dtype()),
        }
    }

    /// Fused rotary embedding (GPT-NeoX half-rotation), replacing the ~7-op `rope_slow` composite.
    /// `self`=src `[b,h,t,d]`, `cos`/`sin` `[t,d/2]` (or `[b,t,d/2]` -> stride_b>0). One thread per
    /// (i1,i2) pair; all math in f32. Mirrors the CUDA `rope` kernel ABI and reuses ReduceKernel.
    pub fn rope(
        &self,
        src_layout: &Layout,
        cos: &RocmStorage,
        cos_layout: &Layout,
        sin: &RocmStorage,
        sin_layout: &Layout,
    ) -> Result<Self> {
        use hanzo_rocm_kernels::kernel::{KernelSource, ReduceKernel};
        let src = self;
        if !src_layout.is_contiguous() || !cos_layout.is_contiguous() || !sin_layout.is_contiguous()
        {
            crate::bail!("rope on rocm requires contiguous inputs");
        }
        let (b, h, t, d) = src_layout.shape().dims4()?;
        if d % 2 != 0 {
            crate::bail!("rope on rocm requires even head dim, got {d}");
        }
        let device = self.device.clone();
        let el = b * h * t * d;
        let bh = (b * h) as u32;
        let td = (t * d) as u32;
        let d_u = d as u32;
        // cos/sin are [t, d/2] (shared across batch) -> stride_b=0; [b, t, d/2] -> per-batch stride
        // over the src element span (matches the CUDA `rope` launcher: stride_b = h*t*d).
        let stride_b: u32 = if cos_layout.dims().len() == 3 && sin_layout.dims().len() == 3 {
            (h * t * d) as u32
        } else {
            0
        };
        let src_off = src_layout.start_offset();
        let cos_off = cos_layout.start_offset();
        let sin_off = sin_layout.start_offset();
        // One thread per (i1,i2) pair = el/2 threads; kernel guards 2*idx >= bh*td.
        let n_pairs = el / 2;
        let grid = rocm_rs::hip::Dim3::from((n_pairs.div_ceil(256)).max(1) as u32);
        let block = rocm_rs::hip::Dim3::from(256u32);

        macro_rules! launch_rope {
            ($variant:ident, $ty:ty, $func:literal) => {{
                let src_mem = match &src.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => unreachable!(),
                };
                let cos_mem = match &cos.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!(
                        "rope: cos dtype {:?} must match src dtype {:?}",
                        cos.slice.dtype(),
                        src.slice.dtype()
                    ),
                };
                let sin_mem = match &sin.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!(
                        "rope: sin dtype {:?} must match src dtype {:?}",
                        sin.slice.dtype(),
                        src.slice.dtype()
                    ),
                };
                let out = device.alloc::<$ty>(el)?;
                let src_ptr = unsafe { src_mem.offset_ptr(src_off) };
                let cos_ptr = unsafe { cos_mem.offset_ptr(cos_off) };
                let sin_ptr = unsafe { sin_mem.offset_ptr(sin_off) };
                let out_ptr = out.as_ptr();
                unsafe {
                    launch_kernel(
                        &device,
                        ReduceKernel::NAME,
                        ReduceKernel::CODE,
                        $func,
                        grid,
                        block,
                        &mut [
                            (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&cos_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&sin_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            &bh as *const u32 as *mut std::ffi::c_void,
                            &td as *const u32 as *mut std::ffi::c_void,
                            &d_u as *const u32 as *mut std::ffi::c_void,
                            &stride_b as *const u32 as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok(Self {
                    slice: RocmStorageSlice::$variant(out),
                    device: device.clone(),
                })
            }};
        }

        match &src.slice {
            RocmStorageSlice::F32(_) => launch_rope!(F32, f32, "rope_f32"),
            RocmStorageSlice::F16(_) => launch_rope!(F16, f16, "rope_f16"),
            RocmStorageSlice::BF16(_) => launch_rope!(BF16, bf16, "rope_bf16"),
            other => crate::bail!("rope on rocm unsupported dtype {:?}", other.dtype()),
        }
    }

    /// Positions-aware fused rotary embedding for q and k, computing per-token cache rows
    /// IN-KERNEL from a device `positions` tensor (u32, length == batch). Matches the engine
    /// CPU reference `apply_rotary_cpu_inner`: cache_row = positions[batch_idx] + seq_idx,
    /// neox pairs (i, i+d/2), gpt-j pairs (2i, 2i+1). `cos`/`sin` are the full [max_pos, d/2]
    /// cache tables. `self`=q and `k` are `[b, h, t, d]` / `[b, kh, t, d]` contiguous; no host
    /// round-trip. Returns (q_out, k_out). Supports F32/F16/BF16 (q/k/cos/sin same dtype).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_positions(
        &self,
        q_layout: &Layout,
        k: &RocmStorage,
        k_layout: &Layout,
        cos: &RocmStorage,
        cos_layout: &Layout,
        sin: &RocmStorage,
        sin_layout: &Layout,
        positions: &RocmStorage,
        positions_layout: &Layout,
        is_neox: bool,
    ) -> Result<(Self, Self)> {
        use hanzo_rocm_kernels::kernel::{KernelSource, RopeKernel};
        let q = self;
        if !q_layout.is_contiguous()
            || !k_layout.is_contiguous()
            || !cos_layout.is_contiguous()
            || !sin_layout.is_contiguous()
            || !positions_layout.is_contiguous()
        {
            crate::bail!("rope_positions on rocm requires contiguous inputs");
        }
        let (b, h, t, d) = q_layout.shape().dims4()?;
        let (kb, kh, kt, kd) = k_layout.shape().dims4()?;
        if (kb, kt, kd) != (b, t, d) {
            crate::bail!("rope_positions q/k shape mismatch {q_layout:?} {k_layout:?}");
        }
        if d % 2 != 0 {
            crate::bail!("rope_positions on rocm requires even head dim, got {d}");
        }
        let pos_mem = match &positions.slice {
            RocmStorageSlice::U32(m) => m,
            other => crate::bail!("rope_positions positions must be u32, got {:?}", other.dtype()),
        };
        let device = self.device.clone();
        let q_el = b * h * t * d;
        let k_el = b * kh * t * d;
        let b_u = b as u32;
        let h_u = h as u32;
        let kh_u = kh as u32;
        let t_u = t as u32;
        let d_u = d as u32;
        let neox: u32 = if is_neox { 1 } else { 0 };
        let max_heads = h.max(kh);
        let total = b * max_heads * t * (d / 2);
        let grid = rocm_rs::hip::Dim3::from((total.div_ceil(256)).max(1) as u32);
        let block = rocm_rs::hip::Dim3::from(256u32);

        let q_off = q_layout.start_offset();
        let k_off = k_layout.start_offset();
        let cos_off = cos_layout.start_offset();
        let sin_off = sin_layout.start_offset();
        let pos_off = positions_layout.start_offset();

        macro_rules! launch_rope_pos {
            ($variant:ident, $ty:ty, $func:literal) => {{
                let q_mem = match &q.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => unreachable!(),
                };
                let k_mem = match &k.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!(
                        "rope_positions: k dtype {:?} must match q dtype {:?}",
                        k.slice.dtype(),
                        q.slice.dtype()
                    ),
                };
                let cos_mem = match &cos.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!(
                        "rope_positions: cos dtype {:?} must match q dtype {:?}",
                        cos.slice.dtype(),
                        q.slice.dtype()
                    ),
                };
                let sin_mem = match &sin.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!(
                        "rope_positions: sin dtype {:?} must match q dtype {:?}",
                        sin.slice.dtype(),
                        q.slice.dtype()
                    ),
                };
                let q_out = device.alloc::<$ty>(q_el)?;
                let k_out = device.alloc::<$ty>(k_el)?;
                let q_ptr = unsafe { q_mem.offset_ptr(q_off) };
                let k_ptr = unsafe { k_mem.offset_ptr(k_off) };
                let cos_ptr = unsafe { cos_mem.offset_ptr(cos_off) };
                let sin_ptr = unsafe { sin_mem.offset_ptr(sin_off) };
                let pos_ptr = unsafe { pos_mem.offset_ptr(pos_off) };
                let q_out_ptr = q_out.as_ptr();
                let k_out_ptr = k_out.as_ptr();
                unsafe {
                    launch_kernel(
                        &device,
                        RopeKernel::NAME,
                        RopeKernel::CODE,
                        $func,
                        grid,
                        block,
                        &mut [
                            (&q_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&k_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&cos_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&sin_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&pos_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&q_out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&k_out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            &b_u as *const u32 as *mut std::ffi::c_void,
                            &h_u as *const u32 as *mut std::ffi::c_void,
                            &kh_u as *const u32 as *mut std::ffi::c_void,
                            &t_u as *const u32 as *mut std::ffi::c_void,
                            &d_u as *const u32 as *mut std::ffi::c_void,
                            &neox as *const u32 as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok((
                    Self {
                        slice: RocmStorageSlice::$variant(q_out),
                        device: device.clone(),
                    },
                    Self {
                        slice: RocmStorageSlice::$variant(k_out),
                        device: device.clone(),
                    },
                ))
            }};
        }

        match &q.slice {
            RocmStorageSlice::F32(_) => launch_rope_pos!(F32, f32, "rope_positions_f32"),
            RocmStorageSlice::F16(_) => launch_rope_pos!(F16, f16, "rope_positions_f16"),
            RocmStorageSlice::BF16(_) => launch_rope_pos!(BF16, bf16, "rope_positions_bf16"),
            other => crate::bail!("rope_positions on rocm unsupported dtype {:?}", other.dtype()),
        }
    }

    /// Fused softmax over the last dim (max-subtract-exp-sum-div, f32 accumulation), replacing the
    /// composite that ran 5 ops + 2 casts. One warp per row (32 lanes along threadIdx.y); the
    /// reduction is a warp shuffle (no shared memory). Reuses the ReduceKernel `softmax_{...}`.
    pub fn softmax_last_dim(&self, layout: &Layout) -> Result<Self> {
        use hanzo_rocm_kernels::kernel::{KernelSource, ReduceKernel};
        if !layout.is_contiguous() {
            crate::bail!("softmax_last_dim on rocm requires contiguous input");
        }
        let device = self.device.clone();
        let dims = layout.shape().dims();
        let elem_count = layout.shape().elem_count();
        let last_dim = dims[dims.len() - 1];
        if last_dim == 0 {
            crate::bail!("softmax_last_dim: last dim must be non-zero");
        }
        let n_rows = (elem_count / last_dim) as u32;
        let n_cols = last_dim as i32;
        let off = layout.start_offset();
        // Match the CUDA softmax launcher: one row per block, a single warp (32 lanes) along y.
        let grid = rocm_rs::hip::Dim3::from((n_rows, 1u32, 1u32));
        let block = rocm_rs::hip::Dim3::from((1u32, 32u32, 1u32));

        macro_rules! launch_softmax {
            ($variant:ident, $ty:ty, $func:literal) => {{
                let x_mem = match &self.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => unreachable!(),
                };
                let out = device.alloc::<$ty>(elem_count)?;
                let x_ptr = unsafe { x_mem.offset_ptr(off) };
                let out_ptr = out.as_ptr();
                unsafe {
                    launch_kernel(
                        &device,
                        ReduceKernel::NAME,
                        ReduceKernel::CODE,
                        $func,
                        grid,
                        block,
                        &mut [
                            (&x_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            &n_cols as *const i32 as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok(Self {
                    slice: RocmStorageSlice::$variant(out),
                    device: device.clone(),
                })
            }};
        }

        match &self.slice {
            RocmStorageSlice::F32(_) => launch_softmax!(F32, f32, "softmax_f32"),
            RocmStorageSlice::F16(_) => launch_softmax!(F16, f16, "softmax_f16"),
            RocmStorageSlice::BF16(_) => launch_softmax!(BF16, bf16, "softmax_bf16"),
            other => crate::bail!("softmax_last_dim on rocm unsupported dtype {:?}", other.dtype()),
        }
    }

    /// Fused SwiGLU: `out[i] = silu(self[i]) * up[i]`, elementwise (both contiguous, same shape).
    /// One kernel launch instead of a separate silu + multiply. silu accumulates in f32. Reuses
    /// the BinaryKernel `silu_mul_{...}` source.
    pub fn silu_mul(&self, layout: &Layout, up: &RocmStorage, up_layout: &Layout) -> Result<Self> {
        use hanzo_rocm_kernels::kernel::{BinaryKernel, KernelSource};
        if !layout.is_contiguous() || !up_layout.is_contiguous() {
            crate::bail!("silu_mul on rocm requires contiguous inputs");
        }
        if layout.shape().dims() != up_layout.shape().dims() {
            crate::bail!(
                "silu_mul: shape mismatch {:?} vs {:?}",
                layout.shape().dims(),
                up_layout.shape().dims()
            );
        }
        let device = self.device.clone();
        let numel = layout.shape().elem_count();
        let lhs_off = layout.start_offset();
        let rhs_off = up_layout.start_offset();
        let (grid, block) = launch_config(numel);
        // Contiguous fast path: num_dims=0 + null dims_and_strides (the kernel treats null as
        // contiguous), so the start offsets are folded into the base pointers below.
        let num_dims: usize = 0;
        let mut ds_null: *mut std::ffi::c_void = std::ptr::null_mut();

        macro_rules! launch_silu_mul {
            ($variant:ident, $ty:ty, $func:literal) => {{
                let lhs_mem = match &self.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => unreachable!(),
                };
                let rhs_mem = match &up.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!(
                        "silu_mul: up dtype {:?} must match gate dtype {:?}",
                        up.slice.dtype(),
                        self.slice.dtype()
                    ),
                };
                let out = device.alloc::<$ty>(numel)?;
                let lhs_ptr = unsafe { lhs_mem.offset_ptr(lhs_off) };
                let rhs_ptr = unsafe { rhs_mem.offset_ptr(rhs_off) };
                let out_ptr = out.as_ptr();
                unsafe {
                    launch_kernel(
                        &device,
                        BinaryKernel::NAME,
                        BinaryKernel::CODE,
                        $func,
                        grid,
                        block,
                        &mut [
                            &numel as *const usize as *mut std::ffi::c_void,
                            &num_dims as *const usize as *mut std::ffi::c_void,
                            (&mut ds_null) as *mut *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&lhs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&rhs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok(Self {
                    slice: RocmStorageSlice::$variant(out),
                    device: device.clone(),
                })
            }};
        }

        match &self.slice {
            RocmStorageSlice::F32(_) => launch_silu_mul!(F32, f32, "silu_mul_f32"),
            RocmStorageSlice::F16(_) => launch_silu_mul!(F16, f16, "silu_mul_f16"),
            RocmStorageSlice::BF16(_) => launch_silu_mul!(BF16, bf16, "silu_mul_bf16"),
            other => crate::bail!("silu_mul on rocm unsupported dtype {:?}", other.dtype()),
        }
    }
}

fn dims_and_strides(
    dev: &RocmDevice,
    layout: &Layout,
    n_strides: usize,
) -> Result<Option<SendSyncDeviceMemory<usize>>> {
    if layout.is_contiguous() {
        return Ok(None);
    }
    let dims = layout.shape().dims();
    let strides = layout.stride();
    let mut data = Vec::with_capacity(dims.len() + n_strides * dims.len());
    for &d in dims {
        data.push(d as usize);
    }
    for _ in 0..n_strides {
        for &s in strides {
            data.push(s as usize);
        }
    }
    Ok(Some(dev.clone_htod(&data)?))
}

fn dims_and_strides_pair(
    dev: &RocmDevice,
    l1: &Layout,
    l2: &Layout,
) -> Result<Option<SendSyncDeviceMemory<usize>>> {
    if l1.is_contiguous() && l2.is_contiguous() {
        return Ok(None);
    }
    let dims = l1.shape().dims();
    let mut data = Vec::with_capacity(dims.len() * 3);
    for &d in dims {
        data.push(d as usize);
    }
    for &s in l1.stride() {
        data.push(s as usize);
    }
    for &s in l2.stride() {
        data.push(s as usize);
    }
    Ok(Some(dev.clone_htod(&data)?))
}

/// Trait for applying unary operations to ROCm storage.
pub trait Map1 {
    /// Apply the operation to a single type.
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>>;

    /// Map the operation over all supported types.
    fn map(&self, s: &RocmStorageSlice, d: &RocmDevice, l: &Layout) -> Result<RocmStorageSlice> {
        let out = match s {
            RocmStorageSlice::U8(s) => RocmStorageSlice::U8(self.f(s, d, l)?),
            RocmStorageSlice::U32(s) => RocmStorageSlice::U32(self.f(s, d, l)?),
            RocmStorageSlice::I16(s) => RocmStorageSlice::I16(self.f(s, d, l)?),
            RocmStorageSlice::I32(s) => RocmStorageSlice::I32(self.f(s, d, l)?),
            RocmStorageSlice::I64(s) => RocmStorageSlice::I64(self.f(s, d, l)?),
            RocmStorageSlice::BF16(s) => RocmStorageSlice::BF16(self.f(s, d, l)?),
            RocmStorageSlice::F16(s) => RocmStorageSlice::F16(self.f(s, d, l)?),
            RocmStorageSlice::F32(s) => RocmStorageSlice::F32(self.f(s, d, l)?),
            RocmStorageSlice::F64(s) => RocmStorageSlice::F64(self.f(s, d, l)?),
            RocmStorageSlice::F8E4M3(_) => {
                crate::bail!("Map1 does not support F8E4M3 for ROCm")
            }
        };
        Ok(out)
    }
}

/// Trait for applying binary operations to ROCm storage.
pub trait Map2 {
    /// Apply the operation to a single type.
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        lhs: &SendSyncDeviceMemory<T>,
        lhs_l: &Layout,
        rhs: &SendSyncDeviceMemory<T>,
        rhs_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>>;

    /// Map the operation over all supported types.
    fn map(
        &self,
        s1: &RocmStorageSlice,
        l1: &Layout,
        s2: &RocmStorageSlice,
        l2: &Layout,
        d: &RocmDevice,
    ) -> Result<RocmStorageSlice> {
        let out = match (s1, s2) {
            (RocmStorageSlice::U8(a), RocmStorageSlice::U8(b)) => {
                RocmStorageSlice::U8(self.f(a, l1, b, l2, d)?)
            }
            (RocmStorageSlice::U32(a), RocmStorageSlice::U32(b)) => {
                RocmStorageSlice::U32(self.f(a, l1, b, l2, d)?)
            }
            (RocmStorageSlice::I16(a), RocmStorageSlice::I16(b)) => {
                RocmStorageSlice::I16(self.f(a, l1, b, l2, d)?)
            }
            (RocmStorageSlice::I32(a), RocmStorageSlice::I32(b)) => {
                RocmStorageSlice::I32(self.f(a, l1, b, l2, d)?)
            }
            (RocmStorageSlice::I64(a), RocmStorageSlice::I64(b)) => {
                RocmStorageSlice::I64(self.f(a, l1, b, l2, d)?)
            }
            (RocmStorageSlice::BF16(a), RocmStorageSlice::BF16(b)) => {
                RocmStorageSlice::BF16(self.f(a, l1, b, l2, d)?)
            }
            (RocmStorageSlice::F16(a), RocmStorageSlice::F16(b)) => {
                RocmStorageSlice::F16(self.f(a, l1, b, l2, d)?)
            }
            (RocmStorageSlice::F32(a), RocmStorageSlice::F32(b)) => {
                RocmStorageSlice::F32(self.f(a, l1, b, l2, d)?)
            }
            (RocmStorageSlice::F64(a), RocmStorageSlice::F64(b)) => {
                RocmStorageSlice::F64(self.f(a, l1, b, l2, d)?)
            }
            _ => crate::bail!("dtype mismatch in binary op"),
        };
        Ok(out)
    }
}

impl<U: crate::op::UnaryOpT> Map1 for U {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        use hanzo_rocm_kernels::kernel::UnaryKernel;
        let shape = layout.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>(U::KERNEL);
        let ds = dims_and_strides(dev, layout, 1)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr: *const usize = ds
                .as_ref()
                .map(|d| d.as_ptr() as *const usize)
                .unwrap_or(std::ptr::null());

            launch_kernel(
                dev,
                UnaryKernel::NAME,
                UnaryKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

impl<U: crate::op::BinaryOpT> Map2 for U {
    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        lhs: &SendSyncDeviceMemory<T>,
        lhs_l: &Layout,
        rhs: &SendSyncDeviceMemory<T>,
        rhs_l: &Layout,
        dev: &RocmDevice,
    ) -> Result<SendSyncDeviceMemory<T>> {
        use hanzo_rocm_kernels::kernel::BinaryKernel;
        let shape = lhs_l.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>(U::KERNEL);
        let ds = dims_and_strides_pair(dev, lhs_l, rhs_l)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        unsafe {
            let lhs_ptr = lhs.offset_ptr(lhs_l.start_offset());
            let rhs_ptr = rhs.offset_ptr(rhs_l.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr: *const usize = ds
                .as_ref()
                .map(|d| d.as_ptr() as *const usize)
                .unwrap_or(std::ptr::null());

            launch_kernel(
                dev,
                BinaryKernel::NAME,
                BinaryKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&lhs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&rhs_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

pub(crate) struct Affine(pub f64, pub f64);

impl Affine {
    fn map(&self, s: &RocmStorageSlice, d: &RocmDevice, l: &Layout) -> Result<RocmStorageSlice> {
        let out = match s {
            RocmStorageSlice::U8(s) => RocmStorageSlice::U8(self.f(s, d, l)?),
            RocmStorageSlice::U32(s) => RocmStorageSlice::U32(self.f(s, d, l)?),
            RocmStorageSlice::I16(s) => RocmStorageSlice::I16(self.f(s, d, l)?),
            RocmStorageSlice::I32(s) => RocmStorageSlice::I32(self.f(s, d, l)?),
            RocmStorageSlice::I64(s) => RocmStorageSlice::I64(self.f(s, d, l)?),
            RocmStorageSlice::BF16(s) => RocmStorageSlice::BF16(self.f(s, d, l)?),
            RocmStorageSlice::F16(s) => RocmStorageSlice::F16(self.f(s, d, l)?),
            RocmStorageSlice::F32(s) => RocmStorageSlice::F32(self.f(s, d, l)?),
            RocmStorageSlice::F64(s) => RocmStorageSlice::F64(self.f(s, d, l)?),
            RocmStorageSlice::F8E4M3(_) => {
                crate::bail!("Affine does not support F8E4M3 for ROCm")
            }
        };
        Ok(out)
    }

    fn f<T: Copy + Send + Sync + WithDType + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        use hanzo_rocm_kernels::kernel::AffineKernel;
        let shape = layout.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>("affine");
        let ds = dims_and_strides(dev, layout, 1)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        let mul_val = T::from_f64(self.0);
        let add_val = T::from_f64(self.1);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr: *const usize = ds
                .as_ref()
                .map(|d| d.as_ptr() as *const usize)
                .unwrap_or(std::ptr::null());

            launch_kernel(
                dev,
                AffineKernel::NAME,
                AffineKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    &mul_val as *const T as *mut std::ffi::c_void,
                    &add_val as *const T as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

struct Powf(f64);

impl Powf {
    fn map(&self, s: &RocmStorageSlice, d: &RocmDevice, l: &Layout) -> Result<RocmStorageSlice> {
        let out = match s {
            RocmStorageSlice::U8(s) => RocmStorageSlice::U8(self.f(s, d, l)?),
            RocmStorageSlice::U32(s) => RocmStorageSlice::U32(self.f(s, d, l)?),
            RocmStorageSlice::I16(s) => RocmStorageSlice::I16(self.f(s, d, l)?),
            RocmStorageSlice::I32(s) => RocmStorageSlice::I32(self.f(s, d, l)?),
            RocmStorageSlice::I64(s) => RocmStorageSlice::I64(self.f(s, d, l)?),
            RocmStorageSlice::BF16(s) => RocmStorageSlice::BF16(self.f(s, d, l)?),
            RocmStorageSlice::F16(s) => RocmStorageSlice::F16(self.f(s, d, l)?),
            RocmStorageSlice::F32(s) => RocmStorageSlice::F32(self.f(s, d, l)?),
            RocmStorageSlice::F64(s) => RocmStorageSlice::F64(self.f(s, d, l)?),
            RocmStorageSlice::F8E4M3(_) => {
                crate::bail!("Powf does not support F8E4M3 for ROCm")
            }
        };
        Ok(out)
    }

    fn f<T: Copy + Send + Sync + WithDType + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        use hanzo_rocm_kernels::kernel::UnaryKernel;
        let shape = layout.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>("upowf");
        let ds = dims_and_strides(dev, layout, 1)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        let scalar_val = T::from_f64(self.0);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr: *const usize = ds
                .as_ref()
                .map(|d| d.as_ptr() as *const usize)
                .unwrap_or(std::ptr::null());

            launch_kernel(
                dev,
                UnaryKernel::NAME,
                UnaryKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    &scalar_val as *const T as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

struct Elu(f64);

impl Elu {
    fn map(&self, s: &RocmStorageSlice, d: &RocmDevice, l: &Layout) -> Result<RocmStorageSlice> {
        let out = match s {
            RocmStorageSlice::U8(s) => RocmStorageSlice::U8(self.f(s, d, l)?),
            RocmStorageSlice::U32(s) => RocmStorageSlice::U32(self.f(s, d, l)?),
            RocmStorageSlice::I16(s) => RocmStorageSlice::I16(self.f(s, d, l)?),
            RocmStorageSlice::I32(s) => RocmStorageSlice::I32(self.f(s, d, l)?),
            RocmStorageSlice::I64(s) => RocmStorageSlice::I64(self.f(s, d, l)?),
            RocmStorageSlice::BF16(s) => RocmStorageSlice::BF16(self.f(s, d, l)?),
            RocmStorageSlice::F16(s) => RocmStorageSlice::F16(self.f(s, d, l)?),
            RocmStorageSlice::F32(s) => RocmStorageSlice::F32(self.f(s, d, l)?),
            RocmStorageSlice::F64(s) => RocmStorageSlice::F64(self.f(s, d, l)?),
            RocmStorageSlice::F8E4M3(_) => {
                crate::bail!("Elu does not support F8E4M3 for ROCm")
            }
        };
        Ok(out)
    }

    fn f<T: Copy + Send + Sync + WithDType + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        use hanzo_rocm_kernels::kernel::UnaryKernel;
        let shape = layout.shape();
        let dims = shape.dims();
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>("uelu");
        let ds = dims_and_strides(dev, layout, 1)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        let alpha_val = T::from_f64(self.0);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr: *const usize = ds
                .as_ref()
                .map(|d| d.as_ptr() as *const usize)
                .unwrap_or(std::ptr::null());

            launch_kernel(
                dev,
                UnaryKernel::NAME,
                UnaryKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    &alpha_val as *const T as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

struct FastReduce<'a>(&'a [usize], ReduceOp);

impl FastReduce<'_> {
    fn map(
        &self,
        s: &RocmStorageSlice,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<RocmStorageSlice> {
        match s {
            RocmStorageSlice::U8(s) => {
                let out = self.f(s, dev, layout)?;
                Ok(RocmStorageSlice::U8(out))
            }
            RocmStorageSlice::U32(s) => {
                let out = self.f(s, dev, layout)?;
                Ok(RocmStorageSlice::U32(out))
            }
            RocmStorageSlice::I16(s) => {
                let out = self.f(s, dev, layout)?;
                Ok(RocmStorageSlice::I16(out))
            }
            RocmStorageSlice::I32(s) => {
                let out = self.f(s, dev, layout)?;
                Ok(RocmStorageSlice::I32(out))
            }
            RocmStorageSlice::I64(s) => {
                let out = self.f(s, dev, layout)?;
                Ok(RocmStorageSlice::I64(out))
            }
            RocmStorageSlice::BF16(s) => {
                let out = self.f(s, dev, layout)?;
                Ok(RocmStorageSlice::BF16(out))
            }
            RocmStorageSlice::F16(s) => {
                let out = self.f(s, dev, layout)?;
                Ok(RocmStorageSlice::F16(out))
            }
            RocmStorageSlice::F32(s) => {
                let out = self.f(s, dev, layout)?;
                Ok(RocmStorageSlice::F32(out))
            }
            RocmStorageSlice::F64(s) => {
                let out = self.f(s, dev, layout)?;
                Ok(RocmStorageSlice::F64(out))
            }
            RocmStorageSlice::F8E4M3(_) => {
                crate::bail!("reduce_op does not support F8E4M3 for ROCm")
            }
        }
    }

    fn f<T: Copy + Send + Sync + 'static>(
        &self,
        src: &SendSyncDeviceMemory<T>,
        dev: &RocmDevice,
        layout: &Layout,
    ) -> Result<SendSyncDeviceMemory<T>> {
        use hanzo_rocm_kernels::kernel::ReduceKernel;
        let src_dims = layout.shape().dims();
        let src_el: usize = src_dims.iter().product();

        let mut dims = vec![];
        let mut stride = vec![];
        let mut dst_el: usize = 1;
        for (dim_idx, &d) in src_dims.iter().enumerate() {
            if !self.0.contains(&dim_idx) {
                dst_el *= d;
                dims.push(d);
                stride.push(layout.stride()[dim_idx]);
            }
        }
        for &dim_idx in self.0.iter() {
            dims.push(src_dims[dim_idx]);
            stride.push(layout.stride()[dim_idx]);
        }
        let el_to_sum_per_block = src_el / dst_el;
        let block_dim = usize::min(1024, el_to_sum_per_block).next_power_of_two();

        let (name, _return_index) = match self.1 {
            ReduceOp::Sum => ("fast_sum", false),
            ReduceOp::Min => ("fast_min", false),
            ReduceOp::Max => ("fast_max", false),
            ReduceOp::ArgMin => ("fast_argmin", true),
            ReduceOp::ArgMax => ("fast_argmax", true),
        };

        let func_name = kernel_name::<T>(name);

        let ds_data: Vec<usize> = [dims.as_slice(), stride.as_slice()].concat();
        let ds = dev.clone_htod(&ds_data)?;

        let output = dev.alloc::<T>(dst_el)?;
        let grid = rocm_rs::hip::Dim3::from(dst_el as u32);
        let block = rocm_rs::hip::Dim3::from(block_dim as u32);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();
            let ds_ptr = ds.as_ptr() as *const usize;

            launch_kernel(
                dev,
                ReduceKernel::NAME,
                ReduceKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &src_el as *const usize as *mut std::ffi::c_void,
                    &el_to_sum_per_block as *const usize as *mut std::ffi::c_void,
                    &src_dims.len() as *const usize as *mut std::ffi::c_void,
                    (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                    (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                    (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                ],
            )?;
        }

        Ok(output)
    }
}

impl std::fmt::Debug for RocmStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "RocmStorage {{ slice: {:?}, device: {:?} }}",
            self.slice, self.device
        )
    }
}

fn where_cond_typed<T: Copy + Send + Sync + WithDType + 'static>(
    cond_prefix: &str,
    cond_ptr: *mut std::ffi::c_void,
    t_ptr: *mut std::ffi::c_void,
    f_ptr: *mut std::ffi::c_void,
    ds: &SendSyncDeviceMemory<usize>,
    numel: usize,
    num_dims: usize,
    device: &RocmDevice,
) -> Result<SendSyncDeviceMemory<T>> {
    use hanzo_rocm_kernels::kernel::{KernelSource, TernaryKernel};
    let func_name = kernel_name::<T>(cond_prefix);
    let output = device.alloc::<T>(numel)?;
    let (grid, block) = launch_config(numel);
    unsafe {
        let out_ptr = output.as_ptr();
        let ds_ptr = ds.as_ptr() as *const usize;
        launch_kernel(
            device,
            TernaryKernel::NAME,
            TernaryKernel::CODE,
            &func_name,
            grid,
            block,
            &mut [
                &numel as *const usize as *mut std::ffi::c_void,
                &num_dims as *const usize as *mut std::ffi::c_void,
                (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                (&cond_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&t_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&f_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
            ],
        )?;
    }
    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn index_select_typed<T: Copy + Send + Sync + WithDType + 'static>(
    ids_prefix: &str,
    ids_ptr: *mut std::ffi::c_void,
    // `Some(ds)`: general path, the shape/stride metadata buffer (an H2D clone_htod
    // upstream). `None`: capture-safe contiguous fast path — the kernel never reads
    // `info`, so no clone_htod fires (the op that trips hipGraph capture HIP 906).
    ds: Option<&SendSyncDeviceMemory<usize>>,
    src_ptr: *mut std::ffi::c_void,
    left_size: usize,
    src_dim_size: usize,
    ids_dim_size: usize,
    right_size: usize,
    dst_el: usize,
    device: &RocmDevice,
) -> Result<SendSyncDeviceMemory<T>> {
    use hanzo_rocm_kernels::kernel::IndexingKernel;

    // Contiguous fast path uses the `isc_*` kernels (ignore `info`); general path the
    // `is_*` kernels. `ids_prefix` already carries the right base ("isc_u32"/"is_u32").
    let func_name = kernel_name::<T>(ids_prefix);
    let output = device.alloc::<T>(dst_el)?;
    let num_dims = ds.map(|d| d.count() / 2).unwrap_or(0);
    let (grid, block) = launch_config(dst_el);

    unsafe {
        let out_ptr = output.as_ptr();
        let ds_ptr = ds
            .map(|d| d.as_ptr() as *const usize)
            .unwrap_or(std::ptr::null());

        launch_kernel(
            device,
            IndexingKernel::NAME,
            IndexingKernel::CODE,
            &func_name,
            grid,
            block,
            &mut [
                &dst_el as *const usize as *mut std::ffi::c_void,
                &num_dims as *const usize as *mut std::ffi::c_void,
                (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                (&ids_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                &left_size as *const usize as *mut std::ffi::c_void,
                &src_dim_size as *const usize as *mut std::ffi::c_void,
                &ids_dim_size as *const usize as *mut std::ffi::c_void,
                &right_size as *const usize as *mut std::ffi::c_void,
            ],
        )?;
    }

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
fn gather_typed<T: Copy + Send + Sync + WithDType + 'static>(
    ids_prefix: &str,
    ids_ptr: *mut std::ffi::c_void,
    src_ptr: *mut std::ffi::c_void,
    left_size: usize,
    src_dim_size: usize,
    ids_dim_size: usize,
    right_size: usize,
    dst_el: usize,
    device: &RocmDevice,
) -> Result<SendSyncDeviceMemory<T>> {
    use hanzo_rocm_kernels::kernel::IndexingKernel;
    let func_name = kernel_name::<T>(ids_prefix);
    let output = device.alloc::<T>(dst_el)?;
    let (grid, block) = launch_config(dst_el);
    unsafe {
        let out_ptr = output.as_ptr();
        launch_kernel(
            device,
            IndexingKernel::NAME,
            IndexingKernel::CODE,
            &func_name,
            grid,
            block,
            &mut [
                &dst_el as *const usize as *mut std::ffi::c_void,
                (&ids_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                &left_size as *const usize as *mut std::ffi::c_void,
                &src_dim_size as *const usize as *mut std::ffi::c_void,
                &ids_dim_size as *const usize as *mut std::ffi::c_void,
                &right_size as *const usize as *mut std::ffi::c_void,
            ],
        )?;
    }
    Ok(output)
}

impl BackendStorage for RocmStorage {
    type Device = RocmDevice;

    fn try_clone(&self, layout: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let elem_count = layout.shape().elem_count();
        let slice = match &self.slice {
            RocmStorageSlice::U8(s) => {
                let mut dst = device.alloc::<u8>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::U8(dst)
            }
            RocmStorageSlice::U32(s) => {
                let mut dst = device.alloc::<u32>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::U32(dst)
            }
            RocmStorageSlice::I16(s) => {
                let mut dst = device.alloc::<i16>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::I16(dst)
            }
            RocmStorageSlice::I32(s) => {
                let mut dst = device.alloc::<i32>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::I32(dst)
            }
            RocmStorageSlice::I64(s) => {
                let mut dst = device.alloc::<i64>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::I64(dst)
            }
            RocmStorageSlice::BF16(s) => {
                let mut dst = device.alloc::<bf16>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::BF16(dst)
            }
            RocmStorageSlice::F16(s) => {
                let mut dst = device.alloc::<f16>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::F16(dst)
            }
            RocmStorageSlice::F32(s) => {
                let mut dst = device.alloc::<f32>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::F32(dst)
            }
            RocmStorageSlice::F64(s) => {
                let mut dst = device.alloc::<f64>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::F64(dst)
            }
            RocmStorageSlice::F8E4M3(s) => {
                let mut dst = device.alloc::<u8>(elem_count)?;
                dst.copy_from_device(s)?;
                RocmStorageSlice::F8E4M3(dst)
            }
        };
        Ok(Self { slice, device })
    }

    fn dtype(&self) -> DType {
        self.slice.dtype()
    }

    fn device(&self) -> &Self::Device {
        &self.device
    }

    fn to_cpu_storage(&self) -> Result<CpuStorage> {
        match &self.slice {
            RocmStorageSlice::U8(s) => Ok(CpuStorage::U8(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::U32(s) => Ok(CpuStorage::U32(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::I16(s) => Ok(CpuStorage::I16(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::I32(s) => Ok(CpuStorage::I32(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::I64(s) => Ok(CpuStorage::I64(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::BF16(s) => Ok(CpuStorage::BF16(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F16(s) => Ok(CpuStorage::F16(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F32(s) => Ok(CpuStorage::F32(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F64(s) => Ok(CpuStorage::F64(self.device.clone_dtoh(s)?.into())),
            RocmStorageSlice::F8E4M3(s) => {
                let bytes = self.device.clone_dtoh(s)?;
                let v: Vec<float8::F8E4M3> =
                    bytes.into_iter().map(float8::F8E4M3::from_bits).collect();
                Ok(CpuStorage::F8E4M3(v.into()))
            }
        }
    }

    fn affine(&self, layout: &Layout, mul: f64, add: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Affine(mul, add).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn powf(&self, layout: &Layout, e: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Powf(e).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn elu(&self, layout: &Layout, alpha: f64) -> Result<Self> {
        let device = self.device.clone();
        let slice = Elu(alpha).map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn reduce_op(&self, op: ReduceOp, l: &Layout, sum_dims: &[usize]) -> Result<Self> {
        let device = self.device.clone();
        let slice = FastReduce(sum_dims, op).map(&self.slice, &device, l)?;
        Ok(Self { slice, device })
    }

    fn cmp(&self, _op: CmpOp, _rhs: &Self, _l1: &Layout, _l2: &Layout) -> Result<Self> {
        Err(crate::Error::Msg(
            "cmp not yet implemented for ROCm".to_string(),
        ))
    }

    fn to_dtype(&self, layout: &Layout, dtype: DType) -> Result<Self> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el = shape.elem_count();
        let dev = self.device.clone();

        let ds = dims_and_strides(&dev, layout, 1)?;
        let start_o = layout.start_offset();
        let src_ptr = unsafe { self.slice.offset_ptr(start_o) };

        let (grid, block) = launch_config(el);
        let ds_ptr: *const usize = ds
            .as_ref()
            .map(|d| d.as_ptr() as *const usize)
            .unwrap_or(std::ptr::null());

        let src_dtype = self.slice.dtype();
        let slice = match dtype {
            DType::U8 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                u8,
                U8
            ),
            DType::U32 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                u32,
                U32
            ),
            DType::I64 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                i64,
                I64
            ),
            DType::BF16 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                bf16,
                BF16
            ),
            DType::F16 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                f16,
                F16
            ),
            DType::F32 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                f32,
                F32
            ),
            DType::F64 => cast_launch!(
                dev,
                grid,
                block,
                el,
                dims.len(),
                ds_ptr,
                src_ptr,
                src_dtype,
                f64,
                F64
            ),
            DType::I16 | DType::I32 => {
                return Err(crate::Error::Msg(
                    "i16/i32 dtypes are not supported for to_dtype on ROCm".to_string(),
                ))
            }
            DType::F8E4M3 | DType::F4 | DType::F6E2M3 | DType::F6E3M2 | DType::F8E8M0 => {
                return Err(crate::Error::Msg(format!(
                    "{:?} dtype is not supported for to_dtype on ROCm",
                    dtype
                )))
            }
        };

        Ok(Self { slice, device: dev })
    }

    fn unary_impl<B: UnaryOpT>(&self, layout: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = B::V.map(&self.slice, &device, layout)?;
        Ok(Self { slice, device })
    }

    fn binary_impl<B: BinaryOpT>(&self, rhs: &Self, l1: &Layout, l2: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let slice = B::V.map(&self.slice, l1, &rhs.slice, l2, &device)?;
        Ok(Self { slice, device })
    }

    fn where_cond(&self, l: &Layout, a: &Self, la: &Layout, b: &Self, lb: &Layout) -> Result<Self> {
        let device = self.device.clone();
        let numel = l.shape().elem_count();
        let num_dims = l.dims().len();
        let ds = device.clone_htod(&[l.dims(), l.stride(), la.stride(), lb.stride()].concat())?;
        let (cond_prefix, cond_ptr) = match &self.slice {
            RocmStorageSlice::U8(s) => ("where_u8", unsafe { s.offset_ptr(l.start_offset()) }
                as *mut std::ffi::c_void),
            RocmStorageSlice::U32(s) => ("where_u32", unsafe { s.offset_ptr(l.start_offset()) }
                as *mut std::ffi::c_void),
            RocmStorageSlice::I64(s) => ("where_i64", unsafe { s.offset_ptr(l.start_offset()) }
                as *mut std::ffi::c_void),
            _ => crate::bail!("where_cond condition must be u8, u32, or i64"),
        };
        let t_ptr = unsafe { a.slice.offset_ptr(la.start_offset()) };
        let f_ptr = unsafe { b.slice.offset_ptr(lb.start_offset()) };
        let slice = match &a.slice {
            RocmStorageSlice::F32(_) => RocmStorageSlice::F32(where_cond_typed::<f32>(
                cond_prefix,
                cond_ptr,
                t_ptr,
                f_ptr,
                &ds,
                numel,
                num_dims,
                &device,
            )?),
            RocmStorageSlice::F64(_) => RocmStorageSlice::F64(where_cond_typed::<f64>(
                cond_prefix,
                cond_ptr,
                t_ptr,
                f_ptr,
                &ds,
                numel,
                num_dims,
                &device,
            )?),
            RocmStorageSlice::U8(_) => RocmStorageSlice::U8(where_cond_typed::<u8>(
                cond_prefix,
                cond_ptr,
                t_ptr,
                f_ptr,
                &ds,
                numel,
                num_dims,
                &device,
            )?),
            RocmStorageSlice::U32(_) => RocmStorageSlice::U32(where_cond_typed::<u32>(
                cond_prefix,
                cond_ptr,
                t_ptr,
                f_ptr,
                &ds,
                numel,
                num_dims,
                &device,
            )?),
            RocmStorageSlice::I64(_) => RocmStorageSlice::I64(where_cond_typed::<i64>(
                cond_prefix,
                cond_ptr,
                t_ptr,
                f_ptr,
                &ds,
                numel,
                num_dims,
                &device,
            )?),
            RocmStorageSlice::BF16(_) => RocmStorageSlice::BF16(where_cond_typed::<half::bf16>(
                cond_prefix,
                cond_ptr,
                t_ptr,
                f_ptr,
                &ds,
                numel,
                num_dims,
                &device,
            )?),
            RocmStorageSlice::F16(_) => RocmStorageSlice::F16(where_cond_typed::<half::f16>(
                cond_prefix,
                cond_ptr,
                t_ptr,
                f_ptr,
                &ds,
                numel,
                num_dims,
                &device,
            )?),
            _ => crate::bail!("where_cond does not support this dtype for ROCm"),
        };
        Ok(Self { slice, device })
    }

    fn conv1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv1D,
    ) -> Result<Self> {
        #[cfg(feature = "rocm-miopen")]
        {
            use crate::rocm_backend::miopen::conv2d_forward;

            let device = self.device();
            let miopen_handle = device.miopen();
            let dst_el = params.b_size * params.c_out * params.l_out();

            return dispatch_miopen_conv!(
                self,
                kernel,
                l,
                kernel_l,
                dst_el,
                device,
                &miopen_handle.0,
                conv2d_forward,
                params.b_size,
                params.c_in,
                params.c_out,
                1,
                params.l_in,
                1,
                params.k_size,
                1,
                params.l_out(),
                params.padding,
                0,
                params.stride,
                1,
                params.dilation,
                1,
            );
        }
        #[cfg(not(feature = "rocm-miopen"))]
        {
            let _ = (l, kernel, kernel_l, params);
            crate::bail!("conv1d on ROCm requires MIOpen (not in the Windows ROCm SDK); rebuild hanzo-ml with --features rocm-miopen on a ROCm install that ships MIOpen")
        }
    }

    fn conv_transpose1d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConvTranspose1D,
    ) -> Result<Self> {
        #[cfg(feature = "rocm-miopen")]
        {
            use crate::rocm_backend::miopen::conv_transpose1d_forward;

            let device = self.device();
            let miopen_handle = device.miopen();
            let dst_el = params.b_size * params.c_out * params.l_out();

            return dispatch_miopen_conv!(
                self,
                kernel,
                l,
                kernel_l,
                dst_el,
                device,
                &miopen_handle.0,
                conv_transpose1d_forward,
                params.b_size,
                params.c_in,
                params.c_out,
                params.l_in,
                params.k_size,
                params.l_out(),
                params.padding,
                params.output_padding,
                params.stride,
                params.dilation,
            );
        }
        #[cfg(not(feature = "rocm-miopen"))]
        {
            let _ = (l, kernel, kernel_l, params);
            crate::bail!("conv_transpose1d on ROCm requires MIOpen (not in the Windows ROCm SDK); rebuild hanzo-ml with --features rocm-miopen on a ROCm install that ships MIOpen")
        }
    }

    fn conv2d(
        &self,
        l: &Layout,
        kernel: &Self,
        kernel_l: &Layout,
        params: &crate::conv::ParamsConv2D,
    ) -> Result<Self> {
        #[cfg(feature = "rocm-miopen")]
        {
            use crate::rocm_backend::miopen::conv2d_forward;

            let device = self.device();
            let miopen_handle = device.miopen();
            let out_h = params.out_h();
            let out_w = params.out_w();
            let dst_el = params.b_size * params.c_out * out_h * out_w;

            return dispatch_miopen_conv!(
                self,
                kernel,
                l,
                kernel_l,
                dst_el,
                device,
                &miopen_handle.0,
                conv2d_forward,
                params.b_size,
                params.c_in,
                params.c_out,
                params.i_h,
                params.i_w,
                params.k_h,
                params.k_w,
                out_h,
                out_w,
                params.padding,
                params.padding,
                params.stride,
                params.stride,
                params.dilation,
                params.dilation,
            );
        }
        #[cfg(not(feature = "rocm-miopen"))]
        {
            let _ = (l, kernel, kernel_l, params);
            crate::bail!("conv2d on ROCm requires MIOpen (not in the Windows ROCm SDK); rebuild hanzo-ml with --features rocm-miopen on a ROCm install that ships MIOpen")
        }
    }

    fn conv_transpose2d(
        &self,
        _l: &Layout,
        _kernel: &Self,
        _kl: &Layout,
        _params: &crate::conv::ParamsConvTranspose2D,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "conv_transpose2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn avg_pool2d(&self, _l: &Layout, _k: (usize, usize), _s: (usize, usize)) -> Result<Self> {
        Err(crate::Error::Msg(
            "avg_pool2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn max_pool2d(&self, _l: &Layout, _k: (usize, usize), _s: (usize, usize)) -> Result<Self> {
        Err(crate::Error::Msg(
            "max_pool2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn upsample_nearest1d(&self, _l: &Layout, _sz: usize) -> Result<Self> {
        Err(crate::Error::Msg(
            "upsample_nearest1d not yet implemented for ROCm".to_string(),
        ))
    }

    fn upsample_nearest2d(&self, _l: &Layout, _w: usize, _h: usize) -> Result<Self> {
        Err(crate::Error::Msg(
            "upsample_nearest2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn upsample_bilinear2d(
        &self,
        _l: &Layout,
        _w: usize,
        _h: usize,
        _align: bool,
        _fh: Option<f64>,
        _fv: Option<f64>,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "upsample_bilinear2d not yet implemented for ROCm".to_string(),
        ))
    }

    fn gather(&self, l: &Layout, idx: &Self, il: &Layout, dim: usize) -> Result<Self> {
        let device = self.device.clone();
        let src_dims = l.dims();
        let ids_dims = il.dims();
        let left_size: usize = src_dims[..dim].iter().product();
        let right_size: usize = src_dims[dim + 1..].iter().product();
        let src_dim_size = src_dims[dim];
        let ids_dim_size = ids_dims[dim];
        let dst_el: usize = ids_dims.iter().product();

        let src_ptr = match l.contiguous_offsets() {
            Some((o1, _)) => unsafe { self.slice.offset_ptr(o1) },
            None => Err(crate::Error::RequiresContiguous { op: "gather" }.bt())?,
        };
        let (ids_prefix, ids_ptr) = match &idx.slice {
            RocmStorageSlice::U32(s) => ("gather_u32", unsafe { s.offset_ptr(il.start_offset()) }),
            RocmStorageSlice::U8(s) => ("gather_u8", unsafe { s.offset_ptr(il.start_offset()) }),
            RocmStorageSlice::I64(s) => ("gather_i64", unsafe { s.offset_ptr(il.start_offset()) }),
            _ => crate::bail!("gather ids should be u8, u32, or i64"),
        };
        macro_rules! g {
            ($variant:ident, $ty:ty) => {
                RocmStorageSlice::$variant(gather_typed::<$ty>(
                    ids_prefix,
                    ids_ptr,
                    src_ptr,
                    left_size,
                    src_dim_size,
                    ids_dim_size,
                    right_size,
                    dst_el,
                    &device,
                )?)
            };
        }
        let slice = match &self.slice {
            RocmStorageSlice::F32(_) => g!(F32, f32),
            RocmStorageSlice::F64(_) => g!(F64, f64),
            RocmStorageSlice::U8(_) => g!(U8, u8),
            RocmStorageSlice::U32(_) => g!(U32, u32),
            RocmStorageSlice::I64(_) => g!(I64, i64),
            RocmStorageSlice::BF16(_) => g!(BF16, half::bf16),
            RocmStorageSlice::F16(_) => g!(F16, half::f16),
            _ => crate::bail!("gather does not support this dtype for ROCm"),
        };
        Ok(Self { slice, device })
    }

    fn scatter_set(
        &mut self,
        _l: &Layout,
        _val: &Self,
        _vl: &Layout,
        _idx: &Self,
        _il: &Layout,
        _dim: usize,
    ) -> Result<()> {
        Err(crate::Error::Msg(
            "scatter_set not yet implemented for ROCm".to_string(),
        ))
    }

    fn scatter_add_set(
        &mut self,
        _l: &Layout,
        _val: &Self,
        _vl: &Layout,
        _idx: &Self,
        _il: &Layout,
        _dim: usize,
    ) -> Result<()> {
        Err(crate::Error::Msg(
            "scatter_add_set not yet implemented for ROCm".to_string(),
        ))
    }

    fn index_select(&self, idx: &Self, src_l: &Layout, ids_l: &Layout, dim: usize) -> Result<Self> {
        let device = self.device.clone();
        let left_size: usize = src_l.dims()[..dim].iter().product();
        let right_size: usize = src_l.dims()[dim + 1..].iter().product();
        let src_dim_size = src_l.dims()[dim];
        let ids_dim_size = ids_l.shape().elem_count();
        let dst_el = ids_dim_size * left_size * right_size;

        // Capture-safe fast path: when the ids layout is contiguous, the index_select
        // kernel's `b = is_contiguous(...)` is always true, so `info` (dims/strides) is
        // never dereferenced. We can then skip the `clone_htod` of the shape metadata —
        // that H2D copy is the FIRST op to trip hipGraph capture (HIP 906) on the
        // embedding gather and the RoPE cos/sin gather (both use contiguous 1-D ids).
        // The `isc_*` kernels are byte-for-byte identical math to `is_*` with b=true.
        // Non-contiguous ids (general / prefill, not captured) keep the original path.
        let ids_contig = ids_l.is_contiguous();
        let ds = if ids_contig {
            None
        } else {
            let ids_dims = ids_l.shape().dims();
            Some(device.clone_htod(&[ids_dims, ids_l.stride()].concat())?)
        };
        let ds_ref = ds.as_ref();

        let src_ptr = match src_l.contiguous_offsets() {
            Some((o1, _)) => unsafe { self.slice.offset_ptr(o1) },
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };

        let prefix_base = if ids_contig { "isc" } else { "is" };
        let (ids_prefix, ids_ptr): (String, *mut std::ffi::c_void) = match &idx.slice {
            RocmStorageSlice::U32(s) => (format!("{prefix_base}_u32"), unsafe {
                s.offset_ptr(ids_l.start_offset())
            } as *mut std::ffi::c_void),
            RocmStorageSlice::U8(s) => (format!("{prefix_base}_u8"), unsafe {
                s.offset_ptr(ids_l.start_offset())
            } as *mut std::ffi::c_void),
            RocmStorageSlice::I64(s) => (format!("{prefix_base}_i64"), unsafe {
                s.offset_ptr(ids_l.start_offset())
            } as *mut std::ffi::c_void),
            _ => crate::bail!("index_select ids should be u8, u32, or i64"),
        };
        let ids_prefix = ids_prefix.as_str();

        let slice = match &self.slice {
            RocmStorageSlice::F32(_) => RocmStorageSlice::F32(index_select_typed::<f32>(
                ids_prefix,
                ids_ptr,
                ds_ref,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::F64(_) => RocmStorageSlice::F64(index_select_typed::<f64>(
                ids_prefix,
                ids_ptr,
                ds_ref,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::U8(_) => RocmStorageSlice::U8(index_select_typed::<u8>(
                ids_prefix,
                ids_ptr,
                ds_ref,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::U32(_) => RocmStorageSlice::U32(index_select_typed::<u32>(
                ids_prefix,
                ids_ptr,
                ds_ref,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::I64(_) => RocmStorageSlice::I64(index_select_typed::<i64>(
                ids_prefix,
                ids_ptr,
                ds_ref,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::BF16(_) => RocmStorageSlice::BF16(index_select_typed::<half::bf16>(
                ids_prefix,
                ids_ptr,
                ds_ref,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::F16(_) => RocmStorageSlice::F16(index_select_typed::<half::f16>(
                ids_prefix,
                ids_ptr,
                ds_ref,
                src_ptr,
                left_size,
                src_dim_size,
                ids_dim_size,
                right_size,
                dst_el,
                &device,
            )?),
            RocmStorageSlice::I16(_) | RocmStorageSlice::I32(_) | RocmStorageSlice::F8E4M3(_) => {
                crate::bail!("index_select does not support this dtype for ROCm")
            }
        };
        Ok(Self { slice, device })
    }

    fn index_add(
        &self,
        _l: &Layout,
        _idx: &Self,
        _il: &Layout,
        _val: &Self,
        _vl: &Layout,
        _dim: usize,
    ) -> Result<Self> {
        Err(crate::Error::Msg(
            "index_add not yet implemented for ROCm".to_string(),
        ))
    }

    fn matmul(
        &self,
        rhs: &Self,
        (b, m, n, k): (usize, usize, usize, usize),
        lhs_l: &Layout,
        rhs_l: &Layout,
    ) -> Result<Self> {
        use rocm_rs::rocblas::ffi;
        dispatch_matmul!(
            self,
            rhs,
            b,
            m,
            n,
            k,
            lhs_l,
            rhs_l,
            &self.device,
            (F32, f32, 1.0f32, 0.0f32, gemm_config, gemm_strided_batched),
            (F64, f64, 1.0f64, 0.0f64, gemm_config, gemm_strided_batched),
            (
                F16,
                f16,
                1.0f32,
                0.0f32,
                gemm_ex_config,
                gemm_strided_batched_ex,
                ffi::rocblas_datatype__rocblas_datatype_f16_r
            ),
            (
                BF16,
                bf16,
                1.0f32,
                0.0f32,
                gemm_ex_config,
                gemm_strided_batched_ex,
                ffi::rocblas_datatype__rocblas_datatype_bf16_r
            ),
        )
    }

    fn copy_strided_src(&self, dst: &mut Self, dst_offset: usize, src_l: &Layout) -> Result<()> {
        let src_shape = src_l.shape();
        let dims = src_shape.dims();
        let el_count = src_shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }

        if src_l.is_contiguous() {
            let (src_ptr, el_size) = match &self.slice {
                RocmStorageSlice::U8(s) => (s.as_ptr(), 1usize),
                RocmStorageSlice::U32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::I16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::I32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::I64(s) => (s.as_ptr(), 8),
                RocmStorageSlice::BF16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::F16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::F32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::F64(s) => (s.as_ptr(), 8),
                RocmStorageSlice::F8E4M3(s) => (s.as_ptr(), 1),
            };
            let (dst_ptr, _) = match &mut dst.slice {
                RocmStorageSlice::U8(s) => (s.as_ptr(), 1usize),
                RocmStorageSlice::U32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::I16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::I32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::I64(s) => (s.as_ptr(), 8),
                RocmStorageSlice::BF16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::F16(s) => (s.as_ptr(), 2),
                RocmStorageSlice::F32(s) => (s.as_ptr(), 4),
                RocmStorageSlice::F64(s) => (s.as_ptr(), 8),
                RocmStorageSlice::F8E4M3(s) => (s.as_ptr(), 1),
            };
            let src_ptr = unsafe { src_ptr.add(src_l.start_offset() * el_size) };
            let dst_ptr = unsafe { dst_ptr.add(dst_offset * el_size) };
            let byte_count = el_count * el_size;
            // Capture-safe D2D copy: both src and dst are device pointers, so an
            // async memcpy on the backend's single stream is fully recordable inside
            // a hipGraph capture (the blocking `hipMemcpy` trips HIP 906). Ordering on
            // the single stream preserves correctness.
            let result = unsafe {
                bindings::hipMemcpyAsync(
                    dst_ptr,
                    src_ptr,
                    byte_count,
                    bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
                    self.device.stream().as_raw(),
                )
            };
            if result != bindings::hipError_t_hipSuccess {
                crate::bail!("hipMemcpyAsync failed with error {}", result);
            }
            return Ok(());
        }

        let (grid, block) = launch_config(el_count);
        let ds = dims_and_strides(&self.device, src_l, 1)?;

        macro_rules! copy_strided {
            ($variant:ident, $suffix:expr, $ty:ty) => {{
                let (src_mem, dst_mem) = match (&self.slice, &mut dst.slice) {
                    (RocmStorageSlice::$variant(s), RocmStorageSlice::$variant(d)) => (s, d),
                    _ => crate::bail!("dtype mismatch in copy_strided_src"),
                };
                let func_name = format!("ucopy_{}", $suffix);
                // as_ptr() is a byte (c_void) pointer; offsets are in elements, so
                // cast to the element type before `.add` to scale by element size.
                let (src_ptr, dst_ptr) = unsafe {
                    (
                        (src_mem.as_ptr() as *const $ty).add(src_l.start_offset())
                            as *mut std::ffi::c_void,
                        (dst_mem.as_ptr() as *mut $ty).add(dst_offset) as *mut std::ffi::c_void,
                    )
                };
                let ds_ptr: *const usize = ds
                    .as_ref()
                    .map(|d| d.as_ptr() as *const usize)
                    .unwrap_or(std::ptr::null());
                unsafe {
                    launch_kernel(
                        &self.device,
                        hanzo_rocm_kernels::kernel::UnaryKernel::NAME,
                        hanzo_rocm_kernels::kernel::UnaryKernel::CODE,
                        &func_name,
                        grid,
                        block,
                        &mut [
                            &el_count as *const usize as *mut std::ffi::c_void,
                            &dims.len() as *const usize as *mut std::ffi::c_void,
                            (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                            (&src_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&dst_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }};
        }

        match &self.slice {
            RocmStorageSlice::U8(_) => copy_strided!(U8, "u8", u8),
            RocmStorageSlice::U32(_) => copy_strided!(U32, "u32", u32),
            RocmStorageSlice::I16(_) => copy_strided!(I16, "i16", i16),
            RocmStorageSlice::I32(_) => copy_strided!(I32, "i32", i32),
            RocmStorageSlice::I64(_) => copy_strided!(I64, "i64", i64),
            RocmStorageSlice::BF16(_) => copy_strided!(BF16, "bf16", bf16),
            RocmStorageSlice::F16(_) => copy_strided!(F16, "f16", f16),
            RocmStorageSlice::F32(_) => copy_strided!(F32, "f32", f32),
            RocmStorageSlice::F64(_) => copy_strided!(F64, "f64", f64),
            RocmStorageSlice::F8E4M3(_) => {
                crate::bail!("copy_strided_src not supported for F8E4M3 on ROCm")
            }
        }

        Ok(())
    }

    fn copy2d(
        &self,
        dst: &mut Self,
        d1: usize,
        d2: usize,
        src_s1: usize,
        dst_s1: usize,
        src_o: usize,
        dst_o: usize,
    ) -> Result<()> {
        if d1 == 0 || d2 == 0 {
            return Ok(());
        }
        let (src_ptr, dst_ptr, el_size) = match (&self.slice, &mut dst.slice) {
            (RocmStorageSlice::U8(s), RocmStorageSlice::U8(d)) => (s.as_ptr(), d.as_ptr(), 1usize),
            (RocmStorageSlice::U32(s), RocmStorageSlice::U32(d)) => (s.as_ptr(), d.as_ptr(), 4),
            (RocmStorageSlice::I16(s), RocmStorageSlice::I16(d)) => (s.as_ptr(), d.as_ptr(), 2),
            (RocmStorageSlice::I32(s), RocmStorageSlice::I32(d)) => (s.as_ptr(), d.as_ptr(), 4),
            (RocmStorageSlice::I64(s), RocmStorageSlice::I64(d)) => (s.as_ptr(), d.as_ptr(), 8),
            (RocmStorageSlice::BF16(s), RocmStorageSlice::BF16(d)) => (s.as_ptr(), d.as_ptr(), 2),
            (RocmStorageSlice::F16(s), RocmStorageSlice::F16(d)) => (s.as_ptr(), d.as_ptr(), 2),
            (RocmStorageSlice::F32(s), RocmStorageSlice::F32(d)) => (s.as_ptr(), d.as_ptr(), 4),
            (RocmStorageSlice::F64(s), RocmStorageSlice::F64(d)) => (s.as_ptr(), d.as_ptr(), 8),
            (RocmStorageSlice::F8E4M3(s), RocmStorageSlice::F8E4M3(d)) => {
                (s.as_ptr(), d.as_ptr(), 1)
            }
            _ => crate::bail!("dtype mismatch in copy2d"),
        };
        let src_ptr = unsafe { src_ptr.add(src_o * el_size) };
        let dst_ptr = unsafe { dst_ptr.add(dst_o * el_size) };
        let width = d2 * el_size;
        let spitch = src_s1 * el_size;
        let dpitch = dst_s1 * el_size;
        // Capture-safe 2D D2D copy (see copy_strided_src): async on the single stream.
        let result = unsafe {
            bindings::hipMemcpy2DAsync(
                dst_ptr,
                dpitch,
                src_ptr,
                spitch,
                width,
                d1,
                bindings::hipMemcpyKind_hipMemcpyDeviceToDevice,
                self.device.stream().as_raw(),
            )
        };
        if result != bindings::hipError_t_hipSuccess {
            crate::bail!("hipMemcpy2DAsync failed with error {}", result);
        }
        Ok(())
    }

    fn const_set(&mut self, val: crate::scalar::Scalar, layout: &Layout) -> Result<()> {
        let shape = layout.shape();
        let dims = shape.dims();
        let el_count = shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }

        let (grid, block) = launch_config(el_count);
        let ds = dims_and_strides(&self.device, layout, 1)?;

        macro_rules! const_set {
            ($variant:ident, $suffix:expr, $ty:ty, $val:expr) => {{
                let mem = match &mut self.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!("dtype mismatch in const_set"),
                };
                let func_name = format!("const_set_{}", $suffix);
                let out_ptr = unsafe { mem.offset_ptr(layout.start_offset()) };
                let scalar_val: $ty = $val;
                let ds_ptr: *const usize = ds
                    .as_ref()
                    .map(|d| d.as_ptr() as *const usize)
                    .unwrap_or(std::ptr::null());
                unsafe {
                    launch_kernel(
                        &self.device,
                        hanzo_rocm_kernels::kernel::FillKernel::NAME,
                        hanzo_rocm_kernels::kernel::FillKernel::CODE,
                        &func_name,
                        grid,
                        block,
                        &mut [
                            &el_count as *const usize as *mut std::ffi::c_void,
                            &dims.len() as *const usize as *mut std::ffi::c_void,
                            (&ds_ptr) as *const *const usize as *mut std::ffi::c_void,
                            &scalar_val as *const $ty as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
            }};
        }

        match (&mut self.slice, val) {
            (RocmStorageSlice::U8(_), crate::scalar::Scalar::U8(v)) => const_set!(U8, "u8", u8, v),
            (RocmStorageSlice::U32(_), crate::scalar::Scalar::U32(v)) => {
                const_set!(U32, "u32", u32, v)
            }
            (RocmStorageSlice::I64(_), crate::scalar::Scalar::I64(v)) => {
                const_set!(I64, "i64", i64, v)
            }
            (RocmStorageSlice::F32(_), crate::scalar::Scalar::F32(v)) => {
                const_set!(F32, "f32", f32, v)
            }
            (RocmStorageSlice::F64(_), crate::scalar::Scalar::F64(v)) => {
                const_set!(F64, "f64", f64, v)
            }
            (RocmStorageSlice::BF16(_), crate::scalar::Scalar::BF16(v)) => {
                const_set!(BF16, "bf16", bf16, v)
            }
            (RocmStorageSlice::F16(_), crate::scalar::Scalar::F16(v)) => {
                const_set!(F16, "f16", f16, v)
            }
            (RocmStorageSlice::I16(_), crate::scalar::Scalar::I16(v)) => {
                const_set!(I16, "i16", i16, v)
            }
            (RocmStorageSlice::I32(_), crate::scalar::Scalar::I32(v)) => {
                const_set!(I32, "i32", i32, v)
            }
            (RocmStorageSlice::F8E4M3(_), _) => {
                crate::bail!("const_set not supported for F8E4M3 on ROCm")
            }
            _ => crate::bail!("dtype mismatch in const_set"),
        }

        Ok(())
    }
}
