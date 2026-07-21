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
    ($dev:expr, $grid:expr, $block:expr, $el:expr, $num_dims:expr, $ds:expr, $src_ptr:expr, $src_dtype:expr, $rust_type:ty, $variant:ident) => {{
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
                    &$num_dims as *const usize as *mut std::ffi::c_void,
                    $ds.as_arg(),
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
static FORCE_SCALAR_MATVEC: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

/// Test-only: force `dp4a_active` to false so the scalar `qmatvec_core` runs, letting the bit-exact
/// numeric oracles gate the scalar decode against the exact reference. Never set in production.
pub fn set_force_scalar_matvec(force: bool) {
    FORCE_SCALAR_MATVEC.store(force, std::sync::atomic::Ordering::Relaxed);
}

fn force_scalar_matvec() -> bool {
    FORCE_SCALAR_MATVEC.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)] // intentionally mirror GgmlDType variant spelling (Q8_0/IQ4_XS/TQ2_0)
pub enum RocmQuantType {
    Q8_0,
    Q4_0,
    Q4K,
    Q6K,
    IQ4_XS,
    TQ2_0,
    // Q2_K/Q3_K super-block k-quants (decode + dp4a; no qmmq prefill).
    Q2K,
    Q3K,
    // Q5_K super-block + Q4_1/Q5_0/Q5_1/Q8_1 legacy blocks (decode + dp4a[Q5_K] + qmmq prefill).
    Q5K,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_1,
    // IQ2_*/IQ3_* codebook i-quants (decode + MoE-decode only; no qmmq prefill).
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    // IQ4_NL codebook + TQ1_0 ternary-base-3 + IQ1_S/IQ1_M 1-bit (decode + MoE-decode only).
    IQ4_NL,
    TQ1_0,
    IQ1_S,
    IQ1_M,
    // MXFP4 (gpt-oss): E8M0 1-byte scale + FP4 codebook, 32-elem block (decode + MoE-decode only).
    MXFP4,
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
            G::Q2K => Self::Q2K,
            G::Q3K => Self::Q3K,
            G::Q5K => Self::Q5K,
            G::Q4_1 => Self::Q4_1,
            G::Q5_0 => Self::Q5_0,
            G::Q5_1 => Self::Q5_1,
            G::Q8_1 => Self::Q8_1,
            G::IQ2_XXS => Self::IQ2_XXS,
            G::IQ2_XS => Self::IQ2_XS,
            G::IQ2_S => Self::IQ2_S,
            G::IQ3_XXS => Self::IQ3_XXS,
            G::IQ3_S => Self::IQ3_S,
            G::IQ4_NL => Self::IQ4_NL,
            G::TQ1_0 => Self::TQ1_0,
            G::IQ1_S => Self::IQ1_S,
            G::IQ1_M => Self::IQ1_M,
            G::MXFP4 => Self::MXFP4,
            _ => return None,
        })
    }

    /// Elements per block (must divide `k`). Matches `qdw_traits<WTYPE>::ELEMS` in quant.hip.
    pub fn block_elems(self) -> usize {
        match self {
            Self::Q8_0
            | Self::Q4_0
            | Self::Q4_1
            | Self::Q5_0
            | Self::Q5_1
            | Self::Q8_1
            | Self::IQ4_NL
            | Self::MXFP4 => 32,
            Self::Q4K
            | Self::Q6K
            | Self::IQ4_XS
            | Self::TQ2_0
            | Self::Q2K
            | Self::Q3K
            | Self::Q5K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::TQ1_0
            | Self::IQ1_S
            | Self::IQ1_M => 256,
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
            Self::Q2K => 84,
            Self::Q3K => 110,
            Self::Q5K => 176,
            Self::Q4_1 => 20,
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q8_1 => 36,
            Self::IQ2_XXS => 66,
            Self::IQ2_XS => 74,
            Self::IQ2_S => 82,
            Self::IQ3_XXS => 98,
            Self::IQ3_S => 110,
            Self::IQ4_NL => 18,
            Self::TQ1_0 => 54,
            Self::IQ1_S => 50,
            Self::IQ1_M => 56,
            Self::MXFP4 => 17,
        }
    }

    /// Whether the weight dequant is symmetric (`val = scale*q`, no per-block min). Mirrors
    /// `wt_traits<WTYPE>::SYMMETRIC` in quant.hip: the ASYMMETRIC types (Q4_K, and later Q5_K)
    /// carry a per-sub-block min, so their prefill GEMM also reads the q8_1 activation block-sum
    /// (`quantize_q8_1`) for the `-dmin*m*d_x*sum` bias term; symmetric types never need it.
    pub fn symmetric(self) -> bool {
        // ASYMMETRIC (per-block min -> qmmq prefill reads the q8_1 activation block-sum for the
        // min-bias term): Q4_K/Q5_K super-blocks + Q4_1/Q5_1 legacy + Q2_K super-block. Every other
        // type is symmetric. Only read on the qmmq prefill path; decode handles asymmetry in `qdec`.
        !matches!(
            self,
            Self::Q4K | Self::Q2K | Self::Q5K | Self::Q4_1 | Self::Q5_1
        )
    }

    /// THE ONE predicate: does this type have a native int8-WMMA prefill GEMM (`qmmq_core<WTYPE>`,
    /// i.e. a `DEFINE_QMMQ`/`wt_traits<WTYPE>` in quant.hip)? It is the SINGLE gate every prefill
    /// dispatch site reads -- dense `forward` (rows>1) and both MoE `use_qmmq` sites -- replacing the
    /// three predicates the feature branches each invented (`prefill_capable`, two `qmmq_capable`s).
    /// Decode-capability (`from_ggml`, the bandwidth-bound lever) and prefill-capability are ORTHOGONAL:
    /// the codebook/fractional types (IQ2_*/IQ3_*/IQ1_*/IQ4_NL/TQ1_0) and the per-16 asym/signed-scale
    /// k-quants (Q2_K/Q3_K) are decode-native only; their rows>1 path dequantizes-to-f16 (dense) or
    /// rides the per-slot matvec core (MoE) -- correct at any token count, just not WMMA-accelerated.
    /// An ALLOWLIST (not a denylist) so a future decode-only type defaults to the safe dequant prefill
    /// instead of dispatching a `qmmq_*` kernel that was never compiled. Mirrors `DEFINE_QMMQ` exactly:
    /// the 6 base types (Q8_0/Q4_0/Q4_K/Q6_K/IQ4_XS/TQ2_0) + Q5_K/Q4_1/Q5_0/Q5_1/Q8_1.
    pub fn qmmq_capable(self) -> bool {
        matches!(
            self,
            Self::Q8_0
                | Self::Q4_0
                | Self::Q4K
                | Self::Q6K
                | Self::IQ4_XS
                | Self::TQ2_0
                | Self::Q5K
                | Self::Q4_1
                | Self::Q5_0
                | Self::Q5_1
                | Self::Q8_1
        )
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
            Self::Q5K => "qmmq_q5k_f16",
            Self::Q4_1 => "qmmq_q4_1_f16",
            Self::Q5_0 => "qmmq_q5_0_f16",
            Self::Q5_1 => "qmmq_q5_1_f16",
            Self::Q8_1 => "qmmq_q8_1_f16",
            Self::Q2K
            | Self::Q3K
            | Self::IQ2_XXS
            | Self::IQ2_XS
            | Self::IQ2_S
            | Self::IQ3_XXS
            | Self::IQ3_S
            | Self::IQ4_NL
            | Self::TQ1_0
            | Self::IQ1_S
            | Self::IQ1_M
            | Self::MXFP4 => {
                unreachable!("prefill_kernel: {self:?} is decode-only (gated by qmmq_capable)")
            }
        }
    }

    /// The FUSED indexed-MoE prefill GEMM entry point (`qmmq_core<WTYPE,true,NWAVE_M>`) for a given
    /// M-tile size: 128 (full prefill), 64, or 32 (few-token-per-expert micro-batches). Same WMMA core,
    /// the M-tile shrinks so the routed rows fill the WMMA M-segments instead of leaving them empty.
    fn moe_prefill_kernel(self, tile_m: usize) -> &'static str {
        match (self, tile_m) {
            (Self::Q8_0, 128) => "moe_qmmq_q8_0_f16",
            (Self::Q8_0, 64) => "moe_qmmq_q8_0_tm64_f16",
            (Self::Q8_0, _) => "moe_qmmq_q8_0_tm32_f16",
            (Self::Q4_0, 128) => "moe_qmmq_q4_0_f16",
            (Self::Q4_0, 64) => "moe_qmmq_q4_0_tm64_f16",
            (Self::Q4_0, _) => "moe_qmmq_q4_0_tm32_f16",
            (Self::Q4K, 128) => "moe_qmmq_q4k_f16",
            (Self::Q4K, 64) => "moe_qmmq_q4k_tm64_f16",
            (Self::Q4K, _) => "moe_qmmq_q4k_tm32_f16",
            (Self::Q6K, 128) => "moe_qmmq_q6k_f16",
            (Self::Q6K, 64) => "moe_qmmq_q6k_tm64_f16",
            (Self::Q6K, _) => "moe_qmmq_q6k_tm32_f16",
            (Self::IQ4_XS, 128) => "moe_qmmq_iq4xs_f16",
            (Self::IQ4_XS, 64) => "moe_qmmq_iq4xs_tm64_f16",
            (Self::IQ4_XS, _) => "moe_qmmq_iq4xs_tm32_f16",
            (Self::TQ2_0, 128) => "moe_qmmq_tq2_0_f16",
            (Self::TQ2_0, 64) => "moe_qmmq_tq2_0_tm64_f16",
            (Self::TQ2_0, _) => "moe_qmmq_tq2_0_tm32_f16",
            (Self::Q5K, 128) => "moe_qmmq_q5k_f16",
            (Self::Q5K, 64) => "moe_qmmq_q5k_tm64_f16",
            (Self::Q5K, _) => "moe_qmmq_q5k_tm32_f16",
            (Self::Q4_1, 128) => "moe_qmmq_q4_1_f16",
            (Self::Q4_1, 64) => "moe_qmmq_q4_1_tm64_f16",
            (Self::Q4_1, _) => "moe_qmmq_q4_1_tm32_f16",
            (Self::Q5_0, 128) => "moe_qmmq_q5_0_f16",
            (Self::Q5_0, 64) => "moe_qmmq_q5_0_tm64_f16",
            (Self::Q5_0, _) => "moe_qmmq_q5_0_tm32_f16",
            (Self::Q5_1, 128) => "moe_qmmq_q5_1_f16",
            (Self::Q5_1, 64) => "moe_qmmq_q5_1_tm64_f16",
            (Self::Q5_1, _) => "moe_qmmq_q5_1_tm32_f16",
            (Self::Q8_1, 128) => "moe_qmmq_q8_1_f16",
            (Self::Q8_1, 64) => "moe_qmmq_q8_1_tm64_f16",
            (Self::Q8_1, _) => "moe_qmmq_q8_1_tm32_f16",
            (
                Self::Q2K
                | Self::Q3K
                | Self::IQ2_XXS
                | Self::IQ2_XS
                | Self::IQ2_S
                | Self::IQ3_XXS
                | Self::IQ3_S
                | Self::IQ4_NL
                | Self::TQ1_0
                | Self::IQ1_S
                | Self::IQ1_M
                | Self::MXFP4,
                _,
            ) => {
                unreachable!("moe_prefill_kernel: {self:?} is decode-only (gated by qmmq_capable)")
            }
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
            (Self::Q2K, true) => "qmatvecu_q2k_f16",
            (Self::Q2K, false) => "qmatvecu_q2k_bf16",
            (Self::Q3K, true) => "qmatvecu_q3k_f16",
            (Self::Q3K, false) => "qmatvecu_q3k_bf16",
            (Self::Q5K, true) => "qmatvecu_q5k_f16",
            (Self::Q5K, false) => "qmatvecu_q5k_bf16",
            (Self::Q4_1, true) => "qmatvecu_q4_1_f16",
            (Self::Q4_1, false) => "qmatvecu_q4_1_bf16",
            (Self::Q5_0, true) => "qmatvecu_q5_0_f16",
            (Self::Q5_0, false) => "qmatvecu_q5_0_bf16",
            (Self::Q5_1, true) => "qmatvecu_q5_1_f16",
            (Self::Q5_1, false) => "qmatvecu_q5_1_bf16",
            (Self::Q8_1, true) => "qmatvecu_q8_1_f16",
            (Self::Q8_1, false) => "qmatvecu_q8_1_bf16",
            (Self::IQ2_XXS, true) => "qmatvecu_iq2xxs_f16",
            (Self::IQ2_XXS, false) => "qmatvecu_iq2xxs_bf16",
            (Self::IQ2_XS, true) => "qmatvecu_iq2xs_f16",
            (Self::IQ2_XS, false) => "qmatvecu_iq2xs_bf16",
            (Self::IQ2_S, true) => "qmatvecu_iq2s_f16",
            (Self::IQ2_S, false) => "qmatvecu_iq2s_bf16",
            (Self::IQ3_XXS, true) => "qmatvecu_iq3xxs_f16",
            (Self::IQ3_XXS, false) => "qmatvecu_iq3xxs_bf16",
            (Self::IQ3_S, true) => "qmatvecu_iq3s_f16",
            (Self::IQ3_S, false) => "qmatvecu_iq3s_bf16",
            (Self::IQ4_NL, true) => "qmatvecu_iq4nl_f16",
            (Self::IQ4_NL, false) => "qmatvecu_iq4nl_bf16",
            (Self::TQ1_0, true) => "qmatvecu_tq1_0_f16",
            (Self::TQ1_0, false) => "qmatvecu_tq1_0_bf16",
            (Self::IQ1_S, true) => "qmatvecu_iq1_s_f16",
            (Self::IQ1_S, false) => "qmatvecu_iq1_s_bf16",
            (Self::IQ1_M, true) => "qmatvecu_iq1_m_f16",
            (Self::IQ1_M, false) => "qmatvecu_iq1_m_bf16",
            (Self::MXFP4, true) => "qmatvecu_mxfp4_f16",
            (Self::MXFP4, false) => "qmatvecu_mxfp4_bf16",
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
            (Self::Q2K, true) => "moe_qmatvecu_q2k_f16",
            (Self::Q2K, false) => "moe_qmatvecu_q2k_bf16",
            (Self::Q3K, true) => "moe_qmatvecu_q3k_f16",
            (Self::Q3K, false) => "moe_qmatvecu_q3k_bf16",
            (Self::Q5K, true) => "moe_qmatvecu_q5k_f16",
            (Self::Q5K, false) => "moe_qmatvecu_q5k_bf16",
            (Self::Q4_1, true) => "moe_qmatvecu_q4_1_f16",
            (Self::Q4_1, false) => "moe_qmatvecu_q4_1_bf16",
            (Self::Q5_0, true) => "moe_qmatvecu_q5_0_f16",
            (Self::Q5_0, false) => "moe_qmatvecu_q5_0_bf16",
            (Self::Q5_1, true) => "moe_qmatvecu_q5_1_f16",
            (Self::Q5_1, false) => "moe_qmatvecu_q5_1_bf16",
            (Self::Q8_1, true) => "moe_qmatvecu_q8_1_f16",
            (Self::Q8_1, false) => "moe_qmatvecu_q8_1_bf16",
            (Self::IQ2_XXS, true) => "moe_qmatvecu_iq2xxs_f16",
            (Self::IQ2_XXS, false) => "moe_qmatvecu_iq2xxs_bf16",
            (Self::IQ2_XS, true) => "moe_qmatvecu_iq2xs_f16",
            (Self::IQ2_XS, false) => "moe_qmatvecu_iq2xs_bf16",
            (Self::IQ2_S, true) => "moe_qmatvecu_iq2s_f16",
            (Self::IQ2_S, false) => "moe_qmatvecu_iq2s_bf16",
            (Self::IQ3_XXS, true) => "moe_qmatvecu_iq3xxs_f16",
            (Self::IQ3_XXS, false) => "moe_qmatvecu_iq3xxs_bf16",
            (Self::IQ3_S, true) => "moe_qmatvecu_iq3s_f16",
            (Self::IQ3_S, false) => "moe_qmatvecu_iq3s_bf16",
            (Self::IQ4_NL, true) => "moe_qmatvecu_iq4nl_f16",
            (Self::IQ4_NL, false) => "moe_qmatvecu_iq4nl_bf16",
            (Self::TQ1_0, true) => "moe_qmatvecu_tq1_0_f16",
            (Self::TQ1_0, false) => "moe_qmatvecu_tq1_0_bf16",
            (Self::IQ1_S, true) => "moe_qmatvecu_iq1_s_f16",
            (Self::IQ1_S, false) => "moe_qmatvecu_iq1_s_bf16",
            (Self::IQ1_M, true) => "moe_qmatvecu_iq1_m_f16",
            (Self::IQ1_M, false) => "moe_qmatvecu_iq1_m_bf16",
            (Self::MXFP4, true) => "moe_qmatvecu_mxfp4_f16",
            (Self::MXFP4, false) => "moe_qmatvecu_mxfp4_bf16",
        }
    }

    /// Whether this type has a faithful int8-dp4a decode (`qdp4a<WTYPE>` in quant.hip). The K-quants
    /// whose 256-element super-block decodes to (int quant)*(per-block scale)[+min] dot a once-
    /// quantized q8_1 int8 activation via `v_dot4` -- ~4x the scalar-float dequant on this APU. Plus
    /// the 256-element IQ codebook quants (IQ2_XXS/XS/S, IQ3_XXS/S, IQ1_S/M): the grid coords land in
    /// int8 (magnitudes + sign, or pre-signed for IQ1) and dp4a against the q8_1 activation, killing
    /// the COMPUTE-bound scalar grid-table float MACs. IQ4_NL stays scalar (32-elem legacy block, not
    /// the 256-elem super-block the dp4a core strides). This is the ONE predicate that routes
    /// decode/MoE to the dp4a core; every other type falls to the scalar `qmatvec_core`.
    fn dp4a_capable(self) -> bool {
        matches!(
            self,
            Self::Q4K
                | Self::Q6K
                | Self::Q2K
                | Self::Q3K
                | Self::Q5K
                | Self::IQ2_XXS
                | Self::IQ2_XS
                | Self::IQ2_S
                | Self::IQ3_XXS
                | Self::IQ3_S
                | Self::IQ1_S
                | Self::IQ1_M
        )
    }

    /// Whether the dp4a path is selected: exactly the dp4a-capable types. The single gate
    /// `matvec_quant` and `moe_matvec_quant` read to choose the dp4a core vs the scalar core.
    /// `force_scalar_matvec` (test-only) forces the scalar core so the bit-exact oracles can gate
    /// the scalar decode against the exact reference; it is never set in production.
    pub fn dp4a_active(self) -> bool {
        self.dp4a_capable() && !force_scalar_matvec()
    }

    /// The unified dp4a decode entry point (`qmatvec_dp4a_core<WTYPE,XT>`). One core, one table.
    /// `act` is the activation dtype: it picks the q8_1 quantize source AND the output store type, so
    /// an F32 residual stream stays F32 end-to-end (no f16 bounce) -- the decode store dtype mirrors
    /// the input dtype, just as the q8_1 dot is dtype-independent.
    fn dp4a_decode_kernel(self, act: Act) -> &'static str {
        match (self, act) {
            (Self::Q4K, Act::F16) => "qmatvec_dp4a_q4k_f16",
            (Self::Q4K, Act::Bf16) => "qmatvec_dp4a_q4k_bf16",
            (Self::Q4K, Act::F32) => "qmatvec_dp4a_q4k_f32",
            (Self::Q6K, Act::F16) => "qmatvec_dp4a_q6k_f16",
            (Self::Q6K, Act::Bf16) => "qmatvec_dp4a_q6k_bf16",
            (Self::Q6K, Act::F32) => "qmatvec_dp4a_q6k_f32",
            (Self::Q2K, Act::F16) => "qmatvec_dp4a_q2k_f16",
            (Self::Q2K, Act::Bf16) => "qmatvec_dp4a_q2k_bf16",
            (Self::Q2K, Act::F32) => "qmatvec_dp4a_q2k_f32",
            (Self::Q3K, Act::F16) => "qmatvec_dp4a_q3k_f16",
            (Self::Q3K, Act::Bf16) => "qmatvec_dp4a_q3k_bf16",
            (Self::Q3K, Act::F32) => "qmatvec_dp4a_q3k_f32",
            (Self::Q5K, Act::F16) => "qmatvec_dp4a_q5k_f16",
            (Self::Q5K, Act::Bf16) => "qmatvec_dp4a_q5k_bf16",
            (Self::Q5K, Act::F32) => "qmatvec_dp4a_q5k_f32",
            (Self::IQ2_XXS, Act::F16) => "qmatvec_dp4a_iq2xxs_f16",
            (Self::IQ2_XXS, Act::Bf16) => "qmatvec_dp4a_iq2xxs_bf16",
            (Self::IQ2_XXS, Act::F32) => "qmatvec_dp4a_iq2xxs_f32",
            (Self::IQ2_XS, Act::F16) => "qmatvec_dp4a_iq2xs_f16",
            (Self::IQ2_XS, Act::Bf16) => "qmatvec_dp4a_iq2xs_bf16",
            (Self::IQ2_XS, Act::F32) => "qmatvec_dp4a_iq2xs_f32",
            (Self::IQ2_S, Act::F16) => "qmatvec_dp4a_iq2s_f16",
            (Self::IQ2_S, Act::Bf16) => "qmatvec_dp4a_iq2s_bf16",
            (Self::IQ2_S, Act::F32) => "qmatvec_dp4a_iq2s_f32",
            (Self::IQ3_XXS, Act::F16) => "qmatvec_dp4a_iq3xxs_f16",
            (Self::IQ3_XXS, Act::Bf16) => "qmatvec_dp4a_iq3xxs_bf16",
            (Self::IQ3_XXS, Act::F32) => "qmatvec_dp4a_iq3xxs_f32",
            (Self::IQ3_S, Act::F16) => "qmatvec_dp4a_iq3s_f16",
            (Self::IQ3_S, Act::Bf16) => "qmatvec_dp4a_iq3s_bf16",
            (Self::IQ3_S, Act::F32) => "qmatvec_dp4a_iq3s_f32",
            (Self::IQ1_S, Act::F16) => "qmatvec_dp4a_iq1_s_f16",
            (Self::IQ1_S, Act::Bf16) => "qmatvec_dp4a_iq1_s_bf16",
            (Self::IQ1_S, Act::F32) => "qmatvec_dp4a_iq1_s_f32",
            (Self::IQ1_M, Act::F16) => "qmatvec_dp4a_iq1_m_f16",
            (Self::IQ1_M, Act::Bf16) => "qmatvec_dp4a_iq1_m_bf16",
            (Self::IQ1_M, Act::F32) => "qmatvec_dp4a_iq1_m_f32",
            _ => unreachable!("dp4a_decode_kernel: {self:?} is not dp4a-capable"),
        }
    }

    /// The unified indexed-MoE dp4a decode entry point (`moe_qmatvec_dp4a_core<WTYPE,XT>`).
    fn dp4a_moe_kernel(self, act: Act) -> &'static str {
        match (self, act) {
            (Self::Q4K, Act::F16) => "moe_qmatvec_dp4a_q4k_f16",
            (Self::Q4K, Act::Bf16) => "moe_qmatvec_dp4a_q4k_bf16",
            (Self::Q4K, Act::F32) => "moe_qmatvec_dp4a_q4k_f32",
            (Self::Q6K, Act::F16) => "moe_qmatvec_dp4a_q6k_f16",
            (Self::Q6K, Act::Bf16) => "moe_qmatvec_dp4a_q6k_bf16",
            (Self::Q6K, Act::F32) => "moe_qmatvec_dp4a_q6k_f32",
            (Self::Q2K, Act::F16) => "moe_qmatvec_dp4a_q2k_f16",
            (Self::Q2K, Act::Bf16) => "moe_qmatvec_dp4a_q2k_bf16",
            (Self::Q2K, Act::F32) => "moe_qmatvec_dp4a_q2k_f32",
            (Self::Q3K, Act::F16) => "moe_qmatvec_dp4a_q3k_f16",
            (Self::Q3K, Act::Bf16) => "moe_qmatvec_dp4a_q3k_bf16",
            (Self::Q3K, Act::F32) => "moe_qmatvec_dp4a_q3k_f32",
            (Self::Q5K, Act::F16) => "moe_qmatvec_dp4a_q5k_f16",
            (Self::Q5K, Act::Bf16) => "moe_qmatvec_dp4a_q5k_bf16",
            (Self::Q5K, Act::F32) => "moe_qmatvec_dp4a_q5k_f32",
            (Self::IQ2_XXS, Act::F16) => "moe_qmatvec_dp4a_iq2xxs_f16",
            (Self::IQ2_XXS, Act::Bf16) => "moe_qmatvec_dp4a_iq2xxs_bf16",
            (Self::IQ2_XXS, Act::F32) => "moe_qmatvec_dp4a_iq2xxs_f32",
            (Self::IQ2_XS, Act::F16) => "moe_qmatvec_dp4a_iq2xs_f16",
            (Self::IQ2_XS, Act::Bf16) => "moe_qmatvec_dp4a_iq2xs_bf16",
            (Self::IQ2_XS, Act::F32) => "moe_qmatvec_dp4a_iq2xs_f32",
            (Self::IQ2_S, Act::F16) => "moe_qmatvec_dp4a_iq2s_f16",
            (Self::IQ2_S, Act::Bf16) => "moe_qmatvec_dp4a_iq2s_bf16",
            (Self::IQ2_S, Act::F32) => "moe_qmatvec_dp4a_iq2s_f32",
            (Self::IQ3_XXS, Act::F16) => "moe_qmatvec_dp4a_iq3xxs_f16",
            (Self::IQ3_XXS, Act::Bf16) => "moe_qmatvec_dp4a_iq3xxs_bf16",
            (Self::IQ3_XXS, Act::F32) => "moe_qmatvec_dp4a_iq3xxs_f32",
            (Self::IQ3_S, Act::F16) => "moe_qmatvec_dp4a_iq3s_f16",
            (Self::IQ3_S, Act::Bf16) => "moe_qmatvec_dp4a_iq3s_bf16",
            (Self::IQ3_S, Act::F32) => "moe_qmatvec_dp4a_iq3s_f32",
            (Self::IQ1_S, Act::F16) => "moe_qmatvec_dp4a_iq1_s_f16",
            (Self::IQ1_S, Act::Bf16) => "moe_qmatvec_dp4a_iq1_s_bf16",
            (Self::IQ1_S, Act::F32) => "moe_qmatvec_dp4a_iq1_s_f32",
            (Self::IQ1_M, Act::F16) => "moe_qmatvec_dp4a_iq1_m_f16",
            (Self::IQ1_M, Act::Bf16) => "moe_qmatvec_dp4a_iq1_m_bf16",
            (Self::IQ1_M, Act::F32) => "moe_qmatvec_dp4a_iq1_m_f32",
            _ => unreachable!("dp4a_moe_kernel: {self:?} is not dp4a-capable"),
        }
    }
}

/// Activation dtype for the dp4a decode path: selects the q8_1 quantize source kernel and the matvec
/// output store dtype. F32 keeps an F32 residual stream cast-free through the matvec (the q8_1 int8
/// dot is the same; only the load source and store type differ).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Act {
    F16,
    Bf16,
    F32,
}

impl Act {
    fn of(x: &RocmStorage) -> Result<Self> {
        match &x.slice {
            RocmStorageSlice::F16(_) => Ok(Act::F16),
            RocmStorageSlice::BF16(_) => Ok(Act::Bf16),
            RocmStorageSlice::F32(_) => Ok(Act::F32),
            other => crate::bail!(
                "dp4a activation must be f16/bf16/f32, got {:?}",
                other.dtype()
            ),
        }
    }

    /// The `quantize_q8_1*` kernel that produces the q8_1 (int8 xq + f16 xd) activation from this dtype.
    fn quantize_kernel(self) -> &'static str {
        match self {
            Act::F16 => "quantize_q8_1",
            Act::Bf16 => "quantize_q8_1_bf16",
            Act::F32 => "quantize_q8_1_f32",
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
        // dp4a FAST PATH (trait-selected, NOT a per-quant special-case): for any dp4a-capable type
        // (Q4_K/Q6_K now, Q5_K-ready) pre-quantize the activation row to q8_1 once and run v_dot4
        // against the weight blocks -- ~4x the scalar-float dequant on this APU. Every other type
        // falls through to the scalar `qmatvec_core` below. ONE dp4a path, one scalar path, chosen
        // by `dp4a_active`.
        if qt.dp4a_active() {
            return self.matvec_dp4a(qt, wq_ptr, x, n, k);
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

    /// Decode matvec via int8 dp4a for any dp4a-capable type (`qt.dp4a_capable`), the int8-SIMD twin
    /// of `matvec_quant`'s scalar path -- a faithful port of llama.cpp's `vec_dot_<T>_q8_1`. Two
    /// launches: (1) `quantize_q8_1[_bf16]` quantizes the activation row `x[k]` to q8_1 (int8 `xq` +
    /// per-32-block f16 scale `xd`; the unused `xs` block-sum is recomputed in-kernel), (2)
    /// `qmatvec_dp4a_<t>_{f16,bf16}` dots each weight block against the q8_1 int8 via `v_dot4` and
    /// scales. `qt` selects the per-type packed-int decode (`qdp4a<WTYPE>`) -- the ONLY thing that
    /// varies, so Q4_K/Q6_K/Q5_K ride the SAME launcher. `xq`/`xd` own their VRAM past the
    /// stream-ordered launches. Output dtype = `x`'s dtype.
    fn matvec_dp4a(
        &self,
        qt: RocmQuantType,
        wq_ptr: *mut std::ffi::c_void,
        x: &RocmStorage,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let nblk = k / 32;
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
        let (act, x_ptr) = match &x.slice {
            RocmStorageSlice::F16(s) => (Act::F16, s.as_ptr()),
            RocmStorageSlice::BF16(s) => (Act::Bf16, s.as_ptr()),
            RocmStorageSlice::F32(s) => (Act::F32, s.as_ptr()),
            other => crate::bail!(
                "matvec_dp4a: activations must be f16, bf16, or f32, got {:?}",
                other.dtype()
            ),
        };
        let quant_func = act.quantize_kernel();
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

        // dp4a matvec: warp per output row, lane-strided over the row's 256-elem super-blocks (each
        // lane fully int8-decodes its super-blocks via qdp4a<WTYPE>, then a warp-shuffle reduces). On
        // gfx1151 this sustains up to ~460 GB/s via L1/L2 reuse of the small q8_1 activation. 256
        // threads = 8 warps = 8 rows/block.
        let nrows = n as i32;
        let ncols = k as i32;
        let grid = rocm_rs::hip::Dim3::from((n.div_ceil(8)) as u32);
        let block = rocm_rs::hip::Dim3::from(256u32);
        macro_rules! launch_dp4a {
            ($variant:ident, $ty:ty) => {{
                let func = qt.dp4a_decode_kernel(act);
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
        // Bind result after the macro so the borrow checker sees xq/xd/xs used past the (stream-
        // ordered) dp4a launch that reads them.
        let result = match &x.slice {
            RocmStorageSlice::F16(_) => launch_dp4a!(F16, f16),
            RocmStorageSlice::BF16(_) => launch_dp4a!(BF16, bf16),
            RocmStorageSlice::F32(_) => launch_dp4a!(F32, f32),
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
        // dp4a-capable types (Q4_K/Q6_K now): ONE batched int8 dp4a launch over all routed slots
        // (experts on grid.y), activation quantized to q8_1 once -- collapses the per-expert scalar
        // launch loop into the same dp4a roofline as non-MoE decode (mirrors CUDA indexed_moe_forward
        // _q4k_q8_1). Trait-selected like the non-MoE decode, so there is ONE dp4a MoE path + ONE
        // scalar MoE fallback (the `qmatvec_core` branch below).
        if qt.dp4a_active() {
            return self.moe_matvec_dp4a(qt, wbank, x, ids, nrows, n, k);
        }

        // Other wired quant types: ONE batched `moe_qmatvec_core<WTYPE>` launch over all routed
        // slots. Experts on grid.y (slot s = blockIdx.y), expert id read ON-DEVICE (ids_ptr) and the
        // bank offset by ids[s] IN-KERNEL -- no per-slot host loop, no host ids round-trip, so the
        // forward stays HIP-graph-capture-clean. grid.x covers the n output rows (one warp per row,
        // one full [n] output vector per slot). This is the scalar twin of moe_matvec_dp4a.
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

    /// Batched indexed-MoE decode via int8 dp4a for any dp4a-capable type (`qt`): quantize the
    /// [nrows,k] routed activations to q8_1 ONCE, then one `moe_qmatvec_dp4a_<t>_*` launch with the
    /// routed experts on grid.y (expert = ids[s] per slot). One well-occupied grid + int8 dp4a, vs the
    /// per-expert scalar launch loop. `qt` selects the per-type packed-int decode -- Q4_K/Q6_K/Q5_K
    /// ride the SAME launcher. Returns [nrows,n] in x's dtype; routing + bank-residency are the
    /// caller's job. The quantize and dp4a-launch halves are split (`quantize_q8_1` +
    /// `moe_matvec_dp4a_act`) so a shared routed activation (gate+up) can quantize once, matvec twice.
    pub(crate) fn moe_matvec_dp4a(
        &self,
        qt: RocmQuantType,
        wbank: &RocmStorage,
        x: &RocmStorage,
        ids_dev: &RocmStorage,
        nrows: usize,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        let act = Act::of(x)?;
        // xs (the q8_1 block-sum) is recomputed in the dp4a kernel, so it stays unused here.
        let (xq, xd, _xs) = self.quantize_q8_1(x, nrows, k)?;
        self.moe_matvec_dp4a_act(qt, wbank, &xq, &xd, ids_dev, nrows, n, k, act)
    }

    /// The dp4a-launch half of `moe_matvec_dp4a`: one batched launch (grid.x = expert output rows,
    /// grid.y = routed slot, expert ids read on-device) dotting the pre-quantized q8_1 activation
    /// (`xq` int8 + `xd` f16 scale) against the routed expert blocks. `act` is the output store dtype.
    fn moe_matvec_dp4a_act(
        &self,
        qt: RocmQuantType,
        wbank: &RocmStorage,
        xq: &RocmStorage,
        xd: &RocmStorage,
        ids_dev: &RocmStorage,
        nrows: usize,
        n: usize,
        k: usize,
        act: Act,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let wbank_mem = match &wbank.slice {
            RocmStorageSlice::U8(m) => m,
            _ => crate::bail!("moe_matvec_dp4a: weight bank must be u8 (raw GGML blocks)"),
        };
        let ids_ptr = match &ids_dev.slice {
            RocmStorageSlice::U32(m) => m.as_ptr(),
            _ => crate::bail!("moe_matvec_dp4a: ids must be u32 on device"),
        };
        let xq_ptr = match &xq.slice {
            RocmStorageSlice::U8(m) => m.as_ptr(),
            _ => crate::bail!("moe_matvec_dp4a: xq must be u8 (q8_1 int8)"),
        };
        let xd_ptr = match &xd.slice {
            RocmStorageSlice::F16(m) => m.as_ptr(),
            _ => crate::bail!("moe_matvec_dp4a: xd must be f16 (q8_1 scale)"),
        };
        let n_i = n as i32;
        let ncols = k as i32;
        let nslots = nrows as i32;
        let grid = rocm_rs::hip::Dim3::from(((n.div_ceil(8)) as u32, nrows as u32, 1u32));
        let block = rocm_rs::hip::Dim3::from(256u32);
        let wbank_ptr = wbank_mem.as_ptr();
        macro_rules! launch_moe_dp4a {
            ($variant:ident, $ty:ty) => {{
                let func = qt.dp4a_moe_kernel(act);
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
        match act {
            Act::F16 => launch_moe_dp4a!(F16, f16),
            Act::Bf16 => launch_moe_dp4a!(BF16, bf16),
            Act::F32 => launch_moe_dp4a!(F32, f32),
        }
    }

    /// Fused gate+up indexed-MoE matvec: both expert banks consume the SAME routed activation `x`
    /// [nrows,k], so quantize it to q8_1 ONCE and dp4a-matvec both banks (vs a quantize per bank). The
    /// q8_1 activation is deterministic in `x`, so each output is bit-identical to the unfused
    /// `moe_matvec_dp4a`. Scalar (non-dp4a) types have no shared pre-quantize and run the two unfused
    /// matvecs. Returns (gate_out, up_out), each [nrows,n] in x's dtype.
    pub fn moe_matvec_pair(
        &self,
        qt: RocmQuantType,
        gate_bank: &RocmStorage,
        up_bank: &RocmStorage,
        x: &RocmStorage,
        ids_dev: &RocmStorage,
        nrows: usize,
        n: usize,
        k: usize,
    ) -> Result<(RocmStorage, RocmStorage)> {
        if qt.dp4a_active() {
            let act = Act::of(x)?;
            let (xq, xd, _xs) = self.quantize_q8_1(x, nrows, k)?;
            let g = self.moe_matvec_dp4a_act(qt, gate_bank, &xq, &xd, ids_dev, nrows, n, k, act)?;
            let u = self.moe_matvec_dp4a_act(qt, up_bank, &xq, &xd, ids_dev, nrows, n, k, act)?;
            Ok((g, u))
        } else {
            let g = self.moe_matvec_quant(qt, gate_bank, x, ids_dev, nrows, n, k)?;
            let u = self.moe_matvec_quant(qt, up_bank, x, ids_dev, nrows, n, k)?;
            Ok((g, u))
        }
    }

    /// Dense (non-quantized) decode GEMV: `y[n] = W[n,k] . x[k]` for an f16/f32 dense weight, the
    /// direct twin of the quant `matvec_quant`. Replaces the `broadcast_mul + sum` reduce (which
    /// materialized and re-read the whole [n,k] product) for the one dense decode matvec per MoE layer
    /// (the F32 router gate). One warp per output row, lane-strided dot, f32 accumulate. Capture-clean.
    pub fn dense_gemv(
        &self,
        w: &RocmStorage,
        x: &RocmStorage,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        let n_i = n as i32;
        let k_i = k as i32;
        // One block per output row (DGBLK=256 threads split the k-dot); see dense_gemv_core.
        let grid = rocm_rs::hip::Dim3::from(n as u32);
        let block = rocm_rs::hip::Dim3::from(256u32);
        macro_rules! launch_gemv {
            ($variant:ident, $ty:ty, $func:expr) => {{
                let w_ptr = match &w.slice {
                    RocmStorageSlice::$variant(m) => m.as_ptr(),
                    _ => unreachable!(),
                };
                let x_ptr = match &x.slice {
                    RocmStorageSlice::$variant(m) => m.as_ptr(),
                    other => crate::bail!("dense_gemv: x dtype {:?} != w dtype", other.dtype()),
                };
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
                            &n_i as *const i32 as *mut std::ffi::c_void,
                            &k_i as *const i32 as *mut std::ffi::c_void,
                            (&w_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
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
        match &w.slice {
            RocmStorageSlice::F16(_) => launch_gemv!(F16, f16, "dense_gemv_f16"),
            RocmStorageSlice::F32(_) => launch_gemv!(F32, f32, "dense_gemv_f32"),
            other => crate::bail!(
                "dense_gemv: weight dtype {:?} unsupported (f16/f32 only)",
                other.dtype()
            ),
        }
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
        // Activations arrive f16 (dense prefill) or f32 (the MoE compute dtype); quantize each in its
        // native dtype so the MoE glue need not pay an f32->f16 cast before this kernel.
        let (kname, x_ptr) = match &x.slice {
            RocmStorageSlice::F16(s) => ("quantize_q8", s.as_ptr()),
            RocmStorageSlice::F32(s) => ("quantize_q8_f32", s.as_ptr()),
            other => crate::bail!("quantize_q8: x must be f16 or f32, got {:?}", other.dtype()),
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
                kname,
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
            RocmStorage {
                slice: RocmStorageSlice::U8(xq),
                device: self.clone(),
            },
            RocmStorage {
                slice: RocmStorageSlice::F16(xd),
                device: self.clone(),
            },
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
        // f16 (dense prefill), f32 (MoE compute dtype), or bf16 (decode) -- quantize in the native
        // dtype, no f32->f16 cast in the MoE glue.
        let (kname, x_ptr) = match &x.slice {
            RocmStorageSlice::F16(s) => ("quantize_q8_1", s.as_ptr()),
            RocmStorageSlice::F32(s) => ("quantize_q8_1_f32", s.as_ptr()),
            RocmStorageSlice::BF16(s) => ("quantize_q8_1_bf16", s.as_ptr()),
            other => crate::bail!(
                "quantize_q8_1: x must be f16/f32/bf16, got {:?}",
                other.dtype()
            ),
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
                kname,
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
            RocmStorage {
                slice: RocmStorageSlice::U8(xq),
                device: self.clone(),
            },
            RocmStorage {
                slice: RocmStorageSlice::F16(xd),
                device: self.clone(),
            },
            RocmStorage {
                slice: RocmStorageSlice::I32(xs),
                device: self.clone(),
            },
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
            crate::bail!(
                "qmmq_quant({qt:?}): k={k} not a multiple of block size {}",
                qt.block_elems()
            );
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
        let xs_dummy = if xs_opt.is_none() {
            Some(self.alloc::<i32>(1)?)
        } else {
            None
        };
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
        Ok(RocmStorage {
            slice: RocmStorageSlice::F16(out),
            device: self.clone(),
        })
    }

    /// FUSED indexed-MoE PREFILL GEMM. Groups the routed slots by expert (host-side counting sort over
    /// the GPU `ids`), then runs the ONE tiled iu8 WMMA core (`qmmq_core<WTYPE,true>`) per expert-row-
    /// tile so each expert's weight is staged ONCE and amortized over all its tokens via the matrix
    /// cores -- llama's mul_mat_id. Replaces the per-slot `moe_matvec_quant` matvec on the prefill
    /// (rows>1) path, which re-read an expert's weight once per routed token (no matrix cores). `ids` is
    /// GPU u32 [nslots]; prefill is never graph-captured, so the small ids DtoH + metadata HtoD here are
    /// free (decode keeps the capture-clean matvec). Returns [nslots, n] f16.
    pub fn moe_qmmq_quant(
        &self,
        qt: RocmQuantType,
        wbank: &RocmStorage,
        x: &RocmStorage,
        ids: &RocmStorage,
        nslots: usize,
        n: usize,
        k: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        use std::ffi::c_void;
        if k % qt.block_elems() != 0 {
            crate::bail!(
                "moe_qmmq_quant({qt:?}): k={k} not a multiple of block size {}",
                qt.block_elems()
            );
        }
        let wbank_mem = match &wbank.slice {
            RocmStorageSlice::U8(m) => m,
            _ => crate::bail!("moe_qmmq_quant: weight bank must be u8 (raw GGML block bytes)"),
        };
        let nblk = k / qt.block_elems();
        let expert_bytes = n * nblk * qt.block_bytes();
        if expert_bytes == 0 || wbank_mem.size() % expert_bytes != 0 {
            crate::bail!(
                "moe_qmmq_quant: bank size {} not a multiple of expert bytes {expert_bytes}",
                wbank_mem.size()
            );
        }
        let e_cnt = wbank_mem.size() / expert_bytes;

        let ids_host: Vec<u32> = match &ids.slice {
            RocmStorageSlice::U32(m) => self.clone_dtoh(m)?,
            _ => crate::bail!("moe_qmmq_quant: ids must be u32 on device"),
        };
        let mut counts = vec![0i32; e_cnt];
        for &e in &ids_host {
            counts[e as usize] += 1;
        }
        let mut offsets = vec![0i32; e_cnt + 1];
        for e in 0..e_cnt {
            offsets[e + 1] = offsets[e] + counts[e];
        }
        let mut slot_map = vec![0i32; nslots];
        let mut cursor = offsets.clone();
        for (s, &e) in ids_host.iter().enumerate() {
            let e = e as usize;
            slot_map[cursor[e] as usize] = s as i32;
            cursor[e] += 1;
        }
        // M-tile fill selection (LEVER 1): a 128-row WMMA tile wastes its empty M-rows when an expert
        // routes few tokens. Pick TILE_M so the AVERAGE active expert ~fills one tile -- 128 for the
        // full prefill (~64 tok/expert -> ~50% fill at 128, the headline), 64/32 for the micro-batches
        // (~1-8 tok/expert -> a 128-tile would be ~1-6% full). Capacity-rounding only: tile count and
        // weight reads scale with ceil(c/TILE_M), so a smaller tile trades a little weight re-read for
        // far less empty WMMA work. MOE_TILE_M overrides for the A/B. The dense path is untouched.
        let active_experts = counts.iter().filter(|&&c| c > 0).count().max(1);
        let avg_tok = nslots / active_experts;
        // Thresholds from the gfx1151 A/B (Qwen3-30B-A3B): TILE_M=64 wins the full prefill (~64 tok/
        // expert, ~50% fill at 128 -> ~75% at 64, +5%) AND is robust to the per-expert spread, so it
        // beats 128 across the whole prefill range; TILE_M=32 wins the few-token decode micro-batches
        // (~1-8 tok/expert, where a 128-tile is ~1-6% full); 128 only pays off for very dense routing
        // (avg >> 64), which needs a very long prompt. MOE_TILE_M overrides for the A/B.
        let tile_m: i32 = match std::env::var("MOE_TILE_M")
            .ok()
            .and_then(|s| s.parse().ok())
        {
            Some(v @ (32 | 64 | 128)) => v,
            _ if avg_tok >= 160 => 128,
            _ if avg_tok >= 24 => 64,
            _ => 32,
        };
        let mut tile_expert: Vec<i32> = Vec::new();
        let mut tile_pos0: Vec<i32> = Vec::new();
        let mut tile_nrows: Vec<i32> = Vec::new();
        for e in 0..e_cnt {
            let c = counts[e];
            let mut done = 0i32;
            while done < c {
                tile_expert.push(e as i32);
                tile_pos0.push(offsets[e] + done);
                tile_nrows.push((c - done).min(tile_m));
                done += tile_m;
            }
        }
        let num_row_tiles = tile_expert.len();
        if num_row_tiles == 0 {
            let out = self.alloc::<f16>(nslots * n)?;
            return Ok(RocmStorage {
                slice: RocmStorageSlice::F16(out),
                device: self.clone(),
            });
        }
        // M-tile fill telemetry (LEVER-1 measurement): valid rows vs WMMA tile capacity. The wasted
        // fraction = empty WMMA M-rows = the directly-recoverable matrix-core work. Env-gated.
        if std::env::var("MOE_STATS").is_ok() {
            let cap = (num_row_tiles as i32) * tile_m;
            let valid: i32 = tile_nrows.iter().sum();
            eprintln!(
                "[MOE_FILL] nslots={nslots} n={n} k={k} experts_active={active_experts}/{e_cnt} tiles={num_row_tiles} valid={valid} cap={cap} fill={:.1}% TILE_M={tile_m}",
                100.0 * valid as f32 / cap as f32
            );
        }

        let (xq, xd, xs_opt) = if qt.symmetric() {
            let (xq, xd) = self.quantize_q8(x, nslots, k)?;
            (xq, xd, None)
        } else {
            let (xq, xd, xs) = self.quantize_q8_1(x, nslots, k)?;
            (xq, xd, Some(xs))
        };
        let xs_dummy = if xs_opt.is_none() {
            Some(self.alloc::<i32>(1)?)
        } else {
            None
        };
        let xq_ptr = match &xq.slice {
            RocmStorageSlice::U8(s) => s.as_ptr(),
            _ => crate::bail!("moe_qmmq_quant: xq must be u8"),
        };
        let xd_ptr = match &xd.slice {
            RocmStorageSlice::F16(s) => s.as_ptr(),
            _ => crate::bail!("moe_qmmq_quant: xd must be f16"),
        };
        let xs_ptr = match (&xs_opt, &xs_dummy) {
            (Some(xs), _) => match &xs.slice {
                RocmStorageSlice::I32(s) => s.as_ptr(),
                _ => crate::bail!("moe_qmmq_quant: xs must be i32"),
            },
            (None, Some(d)) => d.as_ptr(),
            _ => unreachable!(),
        };

        let slot_dev = self.clone_htod(&slot_map)?;
        let te_dev = self.clone_htod(&tile_expert)?;
        let tp_dev = self.clone_htod(&tile_pos0)?;
        let tn_dev = self.clone_htod(&tile_nrows)?;
        let slot_ptr = slot_dev.as_ptr();
        let te_ptr = te_dev.as_ptr();
        let tp_ptr = tp_dev.as_ptr();
        let tn_ptr = tn_dev.as_ptr();

        let out = self.alloc::<f16>(nslots * n)?;
        let out_ptr = out.as_ptr();
        let wbank_ptr = wbank_mem.as_ptr();
        let mi = nslots as i32;
        let ni = n as i32;
        let ki = k as i32;
        let ncol_tiles = n.div_ceil(128);
        let ncol_tiles_i = ncol_tiles as i32;
        let grid = rocm_rs::hip::Dim3::from((ncol_tiles * num_row_tiles) as u32);
        // Block dim MUST match the kernel's NWAVE_M (4 N-waves * 32 lanes * NWAVE_M M-waves = TILE_M*4).
        let block = rocm_rs::hip::Dim3::from((tile_m as u32) * 4);
        unsafe {
            launch_kernel(
                self,
                QuantKernel::NAME,
                QuantKernel::CODE,
                qt.moe_prefill_kernel(tile_m as usize),
                grid,
                block,
                &mut [
                    &mi as *const i32 as *mut c_void,
                    &ni as *const i32 as *mut c_void,
                    &ki as *const i32 as *mut c_void,
                    &ncol_tiles_i as *const i32 as *mut c_void,
                    (&xq_ptr) as *const *mut c_void as *mut c_void,
                    (&xd_ptr) as *const *mut c_void as *mut c_void,
                    (&wbank_ptr) as *const *mut c_void as *mut c_void,
                    (&out_ptr) as *const *mut c_void as *mut c_void,
                    (&xs_ptr) as *const *mut c_void as *mut c_void,
                    (&slot_ptr) as *const *mut c_void as *mut c_void,
                    (&te_ptr) as *const *mut c_void as *mut c_void,
                    (&tp_ptr) as *const *mut c_void as *mut c_void,
                    (&tn_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok(RocmStorage {
            slice: RocmStorageSlice::F16(out),
            device: self.clone(),
        })
    }

    /// FUSED MoE expert-combine: `out[i,j] = sum_e scores[i,e] * ys[i,e,j]` in ONE launch. Replaces
    /// the engine's `ys.broadcast_mul(scores).sum(Minus2)` (a 16.7M-elem ys->f32 cast + a 16.7M-elem
    /// product temp + an 8-wide reduce at 2M blocks x 8 threads). `ys` is [ntok, topk, n] (f16/bf16/
    /// f32, contiguous), `scores` is [ntok, topk] f32 contiguous; `out` is [ntok, n] in ys's dtype.
    pub fn moe_combine(
        &self,
        ys: &RocmStorage,
        scores: &RocmStorage,
        ntok: usize,
        topk: usize,
        n: usize,
    ) -> Result<RocmStorage> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        use std::ffi::c_void;
        let sc_ptr = match &scores.slice {
            RocmStorageSlice::F32(s) => s.as_ptr(),
            other => crate::bail!("moe_combine: scores must be f32, got {:?}", other.dtype()),
        };
        let total = ntok * n;
        let nt = ntok as i32;
        let tk = topk as i32;
        let ni = n as i32;
        let (grid, block) = launch_config(total);
        macro_rules! launch_combine {
            ($variant:ident, $rty:ty, $kernel:literal) => {{
                let ys_ptr = match &ys.slice {
                    RocmStorageSlice::$variant(s) => s.as_ptr(),
                    _ => unreachable!(),
                };
                let out = self.alloc::<$rty>(total)?;
                let out_ptr = out.as_ptr();
                unsafe {
                    launch_kernel(
                        self,
                        QuantKernel::NAME,
                        QuantKernel::CODE,
                        $kernel,
                        grid,
                        block,
                        &mut [
                            &nt as *const i32 as *mut c_void,
                            &tk as *const i32 as *mut c_void,
                            &ni as *const i32 as *mut c_void,
                            (&ys_ptr) as *const *mut c_void as *mut c_void,
                            (&sc_ptr) as *const *mut c_void as *mut c_void,
                            (&out_ptr) as *const *mut c_void as *mut c_void,
                        ],
                    )?;
                }
                RocmStorage {
                    slice: RocmStorageSlice::$variant(out),
                    device: self.clone(),
                }
            }};
        }
        let out = match &ys.slice {
            RocmStorageSlice::F16(_) => launch_combine!(F16, f16, "moe_combine_f16"),
            RocmStorageSlice::BF16(_) => launch_combine!(BF16, bf16, "moe_combine_bf16"),
            RocmStorageSlice::F32(_) => launch_combine!(F32, f32, "moe_combine_f32"),
            other => crate::bail!("moe_combine: ys dtype {:?} unsupported", other.dtype()),
        };
        Ok(out)
    }

    /// FUSED MoE router: one `moe_route` launch (one block per token) over the F32 router logits
    /// [ntok, n_experts] -> selected expert ids [ntok, topk] (u32, descending logit) + softmax
    /// weights [ntok, topk] (f32). Replaces the softmax->sort->narrow->sum->div chain (~6 launches
    /// /layer). `norm` renormalizes the topk weights to sum 1 (norm_topk_prob). Logits must be f32.
    pub fn moe_route(
        &self,
        logits: &RocmStorage,
        ntok: usize,
        n_experts: usize,
        topk: usize,
        norm: bool,
    ) -> Result<(RocmStorage, RocmStorage)> {
        use hanzo_rocm_kernels::kernel::{KernelSource, QuantKernel};
        use std::ffi::c_void;
        const MOE_ROUTE_MAX_E: usize = 256;
        if n_experts > MOE_ROUTE_MAX_E {
            crate::bail!("moe_route: n_experts {n_experts} exceeds kernel max {MOE_ROUTE_MAX_E}");
        }
        let lg_ptr = match &logits.slice {
            RocmStorageSlice::F32(s) => s.as_ptr(),
            other => crate::bail!("moe_route: logits must be f32, got {:?}", other.dtype()),
        };
        let ids = self.alloc::<u32>(ntok * topk)?;
        let w = self.alloc::<f32>(ntok * topk)?;
        let ids_ptr = ids.as_ptr();
        let w_ptr = w.as_ptr();
        let nt = ntok as i32;
        let ne = n_experts as i32;
        let tk = topk as i32;
        let nm = i32::from(norm);
        let grid = rocm_rs::hip::Dim3::from(ntok as u32);
        let block = rocm_rs::hip::Dim3::from(64u32);
        unsafe {
            launch_kernel(
                self,
                QuantKernel::NAME,
                QuantKernel::CODE,
                "moe_route",
                grid,
                block,
                &mut [
                    &nt as *const i32 as *mut c_void,
                    &ne as *const i32 as *mut c_void,
                    &tk as *const i32 as *mut c_void,
                    &nm as *const i32 as *mut c_void,
                    (&lg_ptr) as *const *mut c_void as *mut c_void,
                    (&ids_ptr) as *const *mut c_void as *mut c_void,
                    (&w_ptr) as *const *mut c_void as *mut c_void,
                ],
            )?;
        }
        Ok((
            RocmStorage {
                slice: RocmStorageSlice::U32(ids),
                device: self.clone(),
            },
            RocmStorage {
                slice: RocmStorageSlice::F32(w),
                device: self.clone(),
            },
        ))
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
        //
        // DSL collapse Phase 2 (ROCm): the DSL `rms_norm_blk` twin is BIT-EXACT (see
        // `dsl_norm_bench::rms_norm_dsl_vs_incumbent`) but benches only ~84-88% of this hand-written
        // warp-shuffle kernel -- cubecl reads eps/ndim from device buffers (HIP has no inline scalar
        // args) + carries codegen overhead the tuned `__shfl_xor` reduction avoids. BELOW the >=97%
        // gate, so this incumbent STAYS: a benchmark-gated survivor, like the dp4a quant cores. (Unlike
        // Vulkan, whose incumbent was a naive per-row shader the DSL block kernel beat ~10x.)
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

    /// Fused residual-add + rmsnorm: returns (sum = x + residual, normed = rmsnorm(sum) * alpha) in
    /// ONE launch (vs a separate add then rmsnorm). F32 (prefill/router stream) and F16 (the decode
    /// residual stream); other dtypes fall back to add + `rms_norm` at the op layer. Bit-identical to
    /// that fallback: the sum is stored rounded to the tensor dtype and the sum-of-squares reduces over
    /// that rounded value, matching a separate `add` (rounds to dtype) then `rmsnorm` (squares stored).
    pub fn add_rms_norm(
        &self,
        x_layout: &Layout,
        residual: &RocmStorage,
        residual_layout: &Layout,
        alpha: &RocmStorage,
        alpha_layout: &Layout,
        eps: f32,
    ) -> Result<(Self, Self)> {
        use hanzo_rocm_kernels::kernel::{KernelSource, ReduceKernel};
        let x = self;
        if !x_layout.is_contiguous()
            || !residual_layout.is_contiguous()
            || !alpha_layout.is_contiguous()
        {
            crate::bail!("add_rms_norm on rocm requires contiguous inputs");
        }
        let device = self.device.clone();
        let dims = x_layout.shape().dims();
        let elem_count = x_layout.shape().elem_count();
        let last_dim = dims[dims.len() - 1];
        if last_dim == 0 {
            crate::bail!("add_rms_norm: last dim must be non-zero");
        }
        let n_rows = (elem_count / last_dim) as u32;
        let x_off = x_layout.start_offset();
        let r_off = residual_layout.start_offset();
        let alpha_off = alpha_layout.start_offset();
        // DSL collapse Phase 2 (ROCm): the DSL `add_rmsnorm_blk` twin is bit-exact but benches ~84-92%
        // of this incumbent (same reason as `rms_norm`); below the >=97% gate, so the hand-written
        // kernel STAYS. See `dsl_norm_bench::add_rms_norm_dsl_vs_incumbent`.
        let n_cols = last_dim as i32;
        let block_size: i32 = if last_dim < 1024 { 32 } else { 1024 };
        let grid = rocm_rs::hip::Dim3::from(n_rows);
        let block = rocm_rs::hip::Dim3::from(block_size as u32);

        macro_rules! launch_add_rmsnorm {
            ($variant:ident, $ty:ty, $func:literal) => {{
                let (x_mem, r_mem, alpha_mem) = match (&x.slice, &residual.slice, &alpha.slice) {
                    (
                        RocmStorageSlice::$variant(xm),
                        RocmStorageSlice::$variant(rm),
                        RocmStorageSlice::$variant(am),
                    ) => (xm, rm, am),
                    _ => unreachable!(),
                };
                let sum = device.alloc::<$ty>(elem_count)?;
                let out = device.alloc::<$ty>(elem_count)?;
                let x_ptr = unsafe { x_mem.offset_ptr(x_off) };
                let r_ptr = unsafe { r_mem.offset_ptr(r_off) };
                let alpha_ptr = unsafe { alpha_mem.offset_ptr(alpha_off) };
                let sum_ptr = sum.as_ptr();
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
                            (&r_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&sum_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&alpha_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            &n_cols as *const i32 as *mut std::ffi::c_void,
                            &block_size as *const i32 as *mut std::ffi::c_void,
                            &eps as *const f32 as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok((
                    Self {
                        slice: RocmStorageSlice::$variant(sum),
                        device: device.clone(),
                    },
                    Self {
                        slice: RocmStorageSlice::$variant(out),
                        device: device.clone(),
                    },
                ))
            }};
        }

        match (&x.slice, &residual.slice, &alpha.slice) {
            (
                RocmStorageSlice::F32(_),
                RocmStorageSlice::F32(_),
                RocmStorageSlice::F32(_),
            ) => launch_add_rmsnorm!(F32, f32, "add_rmsnorm_f32"),
            (
                RocmStorageSlice::F16(_),
                RocmStorageSlice::F16(_),
                RocmStorageSlice::F16(_),
            ) => launch_add_rmsnorm!(F16, f16, "add_rmsnorm_f16"),
            _ => crate::bail!(
                "add_rms_norm: inputs must all be F32 or all F16 (x={:?} residual={:?} alpha={:?})",
                x.slice.dtype(),
                residual.slice.dtype(),
                alpha.slice.dtype()
            ),
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
            other => crate::bail!(
                "rope_positions positions must be u32, got {:?}",
                other.dtype()
            ),
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
            other => crate::bail!(
                "rope_positions on rocm unsupported dtype {:?}",
                other.dtype()
            ),
        }
    }

    /// Fused per-head RMSNorm + positions-RoPE for q and k in ONE launch (one block per head-vector):
    /// `rms_norm(q)*q_weight` + RoPE and the same for k, replacing two standalone `rms_norm` launches +
    /// `rope_positions`. Output dtype == q's dtype (the q/k cast still wraps it; the f32->f16 fold is a
    /// separate entry). Bit-faithful to the unfused chain modulo f32 sum order (decode numeric gate).
    #[allow(clippy::too_many_arguments)]
    pub fn rope_norm_positions(
        &self,
        q_layout: &Layout,
        k: &RocmStorage,
        k_layout: &Layout,
        q_weight: &RocmStorage,
        k_weight: &RocmStorage,
        q_eps: f32,
        k_eps: f32,
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
            crate::bail!("rope_norm_positions on rocm requires contiguous inputs");
        }
        let (b, h, t, d) = q_layout.shape().dims4()?;
        let (kb, kh, kt, kd) = k_layout.shape().dims4()?;
        if (kb, kt, kd) != (b, t, d) {
            crate::bail!("rope_norm_positions q/k shape mismatch {q_layout:?} {k_layout:?}");
        }
        if d % 2 != 0 || d > 1024 {
            crate::bail!("rope_norm_positions requires even head dim <= 1024, got {d}");
        }
        let pos_mem = match &positions.slice {
            RocmStorageSlice::U32(m) => m,
            other => crate::bail!(
                "rope_norm_positions positions must be u32, got {:?}",
                other.dtype()
            ),
        };
        let device = self.device.clone();
        let q_el = b * h * t * d;
        let k_el = b * kh * t * d;
        let (b_u, h_u, kh_u, t_u, d_u) = (b as u32, h as u32, kh as u32, t as u32, d as u32);
        let neox: u32 = if is_neox { 1 } else { 0 };
        let nblocks = b * (h + kh) * t;
        let grid = rocm_rs::hip::Dim3::from(nblocks.max(1) as u32);
        let block = rocm_rs::hip::Dim3::from(d as u32);
        let nwarps = d.div_ceil(32);
        let shmem = ((d + nwarps) * std::mem::size_of::<f32>()) as u32;

        let q_off = q_layout.start_offset();
        let k_off = k_layout.start_offset();
        let cos_off = cos_layout.start_offset();
        let sin_off = sin_layout.start_offset();
        let pos_off = positions_layout.start_offset();

        macro_rules! launch_rope_norm {
            ($variant:ident, $ty:ty, $func:literal) => {{
                let (q_mem, k_mem, qw_mem, kw_mem, cos_mem, sin_mem) = match (
                    &q.slice,
                    &k.slice,
                    &q_weight.slice,
                    &k_weight.slice,
                    &cos.slice,
                    &sin.slice,
                ) {
                    (
                        RocmStorageSlice::$variant(a),
                        RocmStorageSlice::$variant(b),
                        RocmStorageSlice::$variant(c),
                        RocmStorageSlice::$variant(d),
                        RocmStorageSlice::$variant(e),
                        RocmStorageSlice::$variant(f),
                    ) => (a, b, c, d, e, f),
                    _ => crate::bail!(
                        "rope_norm_positions: q/k/weights/cos/sin must share q's dtype"
                    ),
                };
                let q_out = device.alloc::<$ty>(q_el)?;
                let k_out = device.alloc::<$ty>(k_el)?;
                let q_ptr = unsafe { q_mem.offset_ptr(q_off) };
                let k_ptr = unsafe { k_mem.offset_ptr(k_off) };
                let qw_ptr = qw_mem.as_ptr();
                let kw_ptr = kw_mem.as_ptr();
                let cos_ptr = unsafe { cos_mem.offset_ptr(cos_off) };
                let sin_ptr = unsafe { sin_mem.offset_ptr(sin_off) };
                let pos_ptr = unsafe { pos_mem.offset_ptr(pos_off) };
                let q_out_ptr = q_out.as_ptr();
                let k_out_ptr = k_out.as_ptr();
                unsafe {
                    launch_kernel_shmem(
                        &device,
                        RopeKernel::NAME,
                        RopeKernel::CODE,
                        $func,
                        grid,
                        block,
                        shmem,
                        &mut [
                            (&q_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&k_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&qw_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&kw_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&cos_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&sin_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&pos_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&q_out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&k_out_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            &q_eps as *const f32 as *mut std::ffi::c_void,
                            &k_eps as *const f32 as *mut std::ffi::c_void,
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
            RocmStorageSlice::F32(_) => launch_rope_norm!(F32, f32, "rope_norm_positions_f32"),
            RocmStorageSlice::F16(_) => launch_rope_norm!(F16, f16, "rope_norm_positions_f16"),
            RocmStorageSlice::BF16(_) => launch_rope_norm!(BF16, bf16, "rope_norm_positions_bf16"),
            other => crate::bail!(
                "rope_norm_positions on rocm unsupported dtype {:?}",
                other.dtype()
            ),
        }
    }

    /// Matrix-core (WMMA) flash-attention forward. `self`=Q `[B,Hq,Lq,D]`, `k`/`v` `[B,Hkv,Lk,D]`
    /// (GQA: `Hq % Hkv == 0`), head dim D must be 128. Returns O `[B,Hq,Lq,D]`. f16/bf16 only.
    /// `causal` enables the per-query upper-triangle skip. Inputs must be contiguous.
    #[allow(clippy::too_many_arguments)]
    pub fn flash_attn(
        &self,
        q_layout: &Layout,
        k: &RocmStorage,
        k_layout: &Layout,
        v: &RocmStorage,
        v_layout: &Layout,
        scale: f32,
        causal: bool,
    ) -> Result<Self> {
        use hanzo_rocm_kernels::kernel::{FlashKernel, KernelSource};
        const FA_DH: usize = 128;
        let q = self;
        if !q_layout.is_contiguous() || !k_layout.is_contiguous() || !v_layout.is_contiguous() {
            crate::bail!("flash_attn on rocm requires contiguous q/k/v");
        }
        let (b, hq, lq, d) = q_layout.shape().dims4()?;
        let (kb, hkv, lk, kd) = k_layout.shape().dims4()?;
        let (vb, vhkv, vlk, vd) = v_layout.shape().dims4()?;
        if d != FA_DH || kd != FA_DH || vd != FA_DH {
            crate::bail!(
                "flash_attn on rocm requires head_dim == {FA_DH}, got q={d} k={kd} v={vd}"
            );
        }
        if (kb, vb, vhkv, vlk) != (b, b, hkv, lk) {
            crate::bail!("flash_attn q/k/v batch/kv-head/kv-len mismatch {q_layout:?} {k_layout:?} {v_layout:?}");
        }
        if hkv == 0 || hq % hkv != 0 {
            crate::bail!("flash_attn requires Hq % Hkv == 0, got Hq={hq} Hkv={hkv}");
        }
        let device = self.device.clone();
        let o_el = b * hq * lq * d;
        let b_i = b as i32;
        let hq_i = hq as i32;
        let hkv_i = hkv as i32;
        let lq_i = lq as i32;
        let lk_i = lk as i32;
        let causal_i: i32 = i32::from(causal);
        const BR: usize = 64;
        const BLOCK: u32 = 128;
        let grid = rocm_rs::hip::Dim3 {
            x: lq.div_ceil(BR) as u32,
            y: hq as u32,
            z: b as u32,
        };
        let block = rocm_rs::hip::Dim3::from(BLOCK);
        let q_off = q_layout.start_offset();
        let k_off = k_layout.start_offset();
        let v_off = v_layout.start_offset();

        macro_rules! launch_flash {
            ($variant:ident, $ty:ty, $func:literal) => {{
                let q_mem = match &q.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => unreachable!(),
                };
                let k_mem = match &k.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!(
                        "flash_attn: k dtype {:?} must match q dtype {:?}",
                        k.slice.dtype(),
                        q.slice.dtype()
                    ),
                };
                let v_mem = match &v.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!(
                        "flash_attn: v dtype {:?} must match q dtype {:?}",
                        v.slice.dtype(),
                        q.slice.dtype()
                    ),
                };
                let o_out = device.alloc::<$ty>(o_el)?;
                let q_ptr = unsafe { q_mem.offset_ptr(q_off) };
                let k_ptr = unsafe { k_mem.offset_ptr(k_off) };
                let v_ptr = unsafe { v_mem.offset_ptr(v_off) };
                let o_ptr = o_out.as_ptr();
                unsafe {
                    launch_kernel(
                        &device,
                        FlashKernel::NAME,
                        FlashKernel::CODE,
                        $func,
                        grid,
                        block,
                        &mut [
                            &b_i as *const i32 as *mut std::ffi::c_void,
                            &hq_i as *const i32 as *mut std::ffi::c_void,
                            &hkv_i as *const i32 as *mut std::ffi::c_void,
                            &lq_i as *const i32 as *mut std::ffi::c_void,
                            &lk_i as *const i32 as *mut std::ffi::c_void,
                            &scale as *const f32 as *mut std::ffi::c_void,
                            &causal_i as *const i32 as *mut std::ffi::c_void,
                            (&q_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&k_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&v_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                            (&o_ptr) as *const *mut std::ffi::c_void as *mut std::ffi::c_void,
                        ],
                    )?;
                }
                Ok(Self {
                    slice: RocmStorageSlice::$variant(o_out),
                    device: device.clone(),
                })
            }};
        }

        match &q.slice {
            RocmStorageSlice::F16(_) => launch_flash!(F16, f16, "flash_attn_f16"),
            RocmStorageSlice::BF16(_) => launch_flash!(BF16, bf16, "flash_attn_bf16"),
            other => crate::bail!(
                "flash_attn on rocm unsupported dtype {:?} (f16/bf16 only)",
                other.dtype()
            ),
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
            other => crate::bail!(
                "softmax_last_dim on rocm unsupported dtype {:?}",
                other.dtype()
            ),
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

/// Max tensor rank the inline dims/strides payload supports. Mirrors `ROCM_DS_MAX`
/// in every strided HIP kernel (see e.g. unary.hip). Decode/prefill tensors are all
/// <= 4D; 8 is a comfortable ceiling. A higher rank bails rather than truncating.
pub(crate) const ROCM_DS_MAX: usize = 8;

/// Number of stride-sets the payload reserves room for: dims + up to 3 stride
/// vectors (the ternary/where_cond op has 3). `[dims, s0, s1, s2]`.
const ROCM_DS_SETS: usize = 4;

/// Inline dims/strides metadata passed BY VALUE in the kernel param block instead
/// of via a device buffer. Eliminates the per-op `clone_htod` H2D copy that trips
/// hipGraph capture (HIP 906 hipErrorStreamCaptureImplicit) on every strided op
/// every layer every token. Layout is `[dims (ROCM_DS_MAX), strides_0, strides_1,
/// strides_2]`, each block of `ROCM_DS_MAX` `usize`s; the kernel reads only the
/// first `num_dims` of each block. `#[repr(C)]` so it matches the HIP `DimsStrides`
/// struct byte-for-byte. The kernel still runs its own `is_contiguous` check on the
/// inline values, so the contiguous fast path needs no special-casing here.
#[repr(C)]
#[derive(Clone, Copy)]
struct DimsStrides {
    v: [usize; ROCM_DS_MAX * ROCM_DS_SETS],
}

impl DimsStrides {
    /// Build the inline payload from `dims` and `n_strides` stride vectors. Returns
    /// the payload plus `num_dims` (the rank the kernel iterates). Bails for rank >
    /// `ROCM_DS_MAX` rather than silently truncating.
    fn build(dims: &[usize], strides: &[&[usize]]) -> Result<(Self, usize)> {
        let num_dims = dims.len();
        if num_dims > ROCM_DS_MAX {
            crate::bail!(
                "ROCm strided op rank {num_dims} exceeds ROCM_DS_MAX {ROCM_DS_MAX}; raise the cap"
            );
        }
        debug_assert!(strides.len() <= ROCM_DS_SETS - 1);
        let mut v = [0usize; ROCM_DS_MAX * ROCM_DS_SETS];
        v[..num_dims].copy_from_slice(dims);
        for (set, stride) in strides.iter().enumerate() {
            let base = (set + 1) * ROCM_DS_MAX;
            v[base..base + num_dims].copy_from_slice(stride);
        }
        Ok((Self { v }, num_dims))
    }

    /// Pointer to the payload for the kernel param block (HIP copies it by value).
    fn as_arg(&self) -> *mut std::ffi::c_void {
        &self.v as *const _ as *mut std::ffi::c_void
    }
}

/// Inline dims + single stride-set for a one-input strided op (Map1, copy, cast,
/// affine, fill, to_dtype).
fn dims_and_strides(layout: &Layout) -> Result<(DimsStrides, usize)> {
    DimsStrides::build(layout.shape().dims(), &[layout.stride()])
}

/// Inline dims + two stride-sets for a broadcast binary op (Map2). Both layouts
/// share `l1`'s dims (the broadcast output shape).
fn dims_and_strides_pair(l1: &Layout, l2: &Layout) -> Result<(DimsStrides, usize)> {
    DimsStrides::build(l1.shape().dims(), &[l1.stride(), l2.stride()])
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
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>(U::KERNEL);
        let (ds, num_dims) = dims_and_strides(layout)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();

            launch_kernel(
                dev,
                UnaryKernel::NAME,
                UnaryKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &num_dims as *const usize as *mut std::ffi::c_void,
                    ds.as_arg(),
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
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>(U::KERNEL);
        let (ds, num_dims) = dims_and_strides_pair(lhs_l, rhs_l)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        unsafe {
            let lhs_ptr = lhs.offset_ptr(lhs_l.start_offset());
            let rhs_ptr = rhs.offset_ptr(rhs_l.start_offset());
            let out_ptr = output.as_ptr();

            launch_kernel(
                dev,
                BinaryKernel::NAME,
                BinaryKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &num_dims as *const usize as *mut std::ffi::c_void,
                    ds.as_arg(),
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
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>("affine");
        let (ds, num_dims) = dims_and_strides(layout)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        let mul_val = T::from_f64(self.0);
        let add_val = T::from_f64(self.1);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();

            launch_kernel(
                dev,
                AffineKernel::NAME,
                AffineKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &num_dims as *const usize as *mut std::ffi::c_void,
                    ds.as_arg(),
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
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>("upowf");
        let (ds, num_dims) = dims_and_strides(layout)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        let scalar_val = T::from_f64(self.0);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();

            launch_kernel(
                dev,
                UnaryKernel::NAME,
                UnaryKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &num_dims as *const usize as *mut std::ffi::c_void,
                    ds.as_arg(),
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
        let elem_count = shape.elem_count();

        let func_name = kernel_name::<T>("uelu");
        let (ds, num_dims) = dims_and_strides(layout)?;
        let output = dev.alloc::<T>(elem_count)?;
        let (grid, block) = launch_config(elem_count);

        let alpha_val = T::from_f64(self.0);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();

            launch_kernel(
                dev,
                UnaryKernel::NAME,
                UnaryKernel::CODE,
                &func_name,
                grid,
                block,
                &mut [
                    &elem_count as *const usize as *mut std::ffi::c_void,
                    &num_dims as *const usize as *mut std::ffi::c_void,
                    ds.as_arg(),
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

        // Reduce reorders to [kept dims..., reduced dims...]; the kernel iterates the
        // full src rank over this reordered (dims, stride) pair, so the payload carries
        // those exact arrays (NOT the raw layout) inline by value.
        let (ds, num_dims) = DimsStrides::build(&dims, &[stride.as_slice()])?;

        let output = dev.alloc::<T>(dst_el)?;
        let grid = rocm_rs::hip::Dim3::from(dst_el as u32);
        let block = rocm_rs::hip::Dim3::from(block_dim as u32);

        unsafe {
            let src_ptr = src.offset_ptr(layout.start_offset());
            let out_ptr = output.as_ptr();

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
                    &num_dims as *const usize as *mut std::ffi::c_void,
                    ds.as_arg(),
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
    ds: &DimsStrides,
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
                ds.as_arg(),
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
    // Inline dims/strides payload (by value in the param block, no device buffer ->
    // capture-clean). `num_dims == 0` is the contiguous `isc_*` fast path (the kernel
    // ignores `info`); otherwise the `is_*` kernel reads the inline metadata.
    ds: &DimsStrides,
    num_dims: usize,
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
                &num_dims as *const usize as *mut std::ffi::c_void,
                ds.as_arg(),
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
        let el = shape.elem_count();
        let dev = self.device.clone();

        let (ds, num_dims) = dims_and_strides(layout)?;
        let start_o = layout.start_offset();
        let src_ptr = unsafe { self.slice.offset_ptr(start_o) };

        let (grid, block) = launch_config(el);

        let src_dtype = self.slice.dtype();
        let slice = match dtype {
            DType::U8 => {
                cast_launch!(dev, grid, block, el, num_dims, ds, src_ptr, src_dtype, u8, U8)
            }
            DType::U32 => {
                cast_launch!(dev, grid, block, el, num_dims, ds, src_ptr, src_dtype, u32, U32)
            }
            DType::I64 => {
                cast_launch!(dev, grid, block, el, num_dims, ds, src_ptr, src_dtype, i64, I64)
            }
            DType::BF16 => {
                cast_launch!(dev, grid, block, el, num_dims, ds, src_ptr, src_dtype, bf16, BF16)
            }
            DType::F16 => {
                cast_launch!(dev, grid, block, el, num_dims, ds, src_ptr, src_dtype, f16, F16)
            }
            DType::F32 => {
                cast_launch!(dev, grid, block, el, num_dims, ds, src_ptr, src_dtype, f32, F32)
            }
            DType::F64 => {
                cast_launch!(dev, grid, block, el, num_dims, ds, src_ptr, src_dtype, f64, F64)
            }
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
        let (ds, num_dims) = DimsStrides::build(l.dims(), &[l.stride(), la.stride(), lb.stride()])?;
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

        // Capture-clean metadata: dims/strides ride INLINE in the kernel param block
        // (no device buffer, no clone_htod -- the H2D copy that trips hipGraph capture
        // with HIP 906). Contiguous ids use the `isc_*` kernels (which ignore `info`
        // entirely), so they pass an empty payload; non-contiguous ids use `is_*` with
        // the filled inline payload. Both are byte-identical math.
        let ids_contig = ids_l.is_contiguous();
        let (ds, num_dims) = if ids_contig {
            (
                DimsStrides {
                    v: [0; ROCM_DS_MAX * ROCM_DS_SETS],
                },
                0,
            )
        } else {
            DimsStrides::build(ids_l.shape().dims(), &[ids_l.stride()])?
        };

        let src_ptr = match src_l.contiguous_offsets() {
            Some((o1, _)) => unsafe { self.slice.offset_ptr(o1) },
            None => Err(crate::Error::RequiresContiguous { op: "index-select" }.bt())?,
        };

        let prefix_base = if ids_contig { "isc" } else { "is" };
        let (ids_prefix, ids_ptr): (String, *mut std::ffi::c_void) = match &idx.slice {
            RocmStorageSlice::U32(s) => {
                (
                    format!("{prefix_base}_u32"),
                    unsafe { s.offset_ptr(ids_l.start_offset()) } as *mut std::ffi::c_void,
                )
            }
            RocmStorageSlice::U8(s) => {
                (
                    format!("{prefix_base}_u8"),
                    unsafe { s.offset_ptr(ids_l.start_offset()) } as *mut std::ffi::c_void,
                )
            }
            RocmStorageSlice::I64(s) => {
                (
                    format!("{prefix_base}_i64"),
                    unsafe { s.offset_ptr(ids_l.start_offset()) } as *mut std::ffi::c_void,
                )
            }
            _ => crate::bail!("index_select ids should be u8, u32, or i64"),
        };
        let ids_prefix = ids_prefix.as_str();

        let slice = match &self.slice {
            RocmStorageSlice::F32(_) => RocmStorageSlice::F32(index_select_typed::<f32>(
                ids_prefix,
                ids_ptr,
                &ds,
                num_dims,
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
                &ds,
                num_dims,
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
                &ds,
                num_dims,
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
                &ds,
                num_dims,
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
                &ds,
                num_dims,
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
                &ds,
                num_dims,
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
                &ds,
                num_dims,
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
        let (ds, num_dims) = dims_and_strides(src_l)?;

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
                            &num_dims as *const usize as *mut std::ffi::c_void,
                            ds.as_arg(),
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
        let el_count = shape.elem_count();
        if el_count == 0 {
            return Ok(());
        }

        let (grid, block) = launch_config(el_count);
        let (ds, num_dims) = dims_and_strides(layout)?;

        macro_rules! const_set {
            ($variant:ident, $suffix:expr, $ty:ty, $val:expr) => {{
                let mem = match &mut self.slice {
                    RocmStorageSlice::$variant(m) => m,
                    _ => crate::bail!("dtype mismatch in const_set"),
                };
                let func_name = format!("const_set_{}", $suffix);
                let out_ptr = unsafe { mem.offset_ptr(layout.start_offset()) };
                let scalar_val: $ty = $val;
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
                            &num_dims as *const usize as *mut std::ffi::c_void,
                            ds.as_arg(),
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

/// A/B gates for the DSL-lowered norm kernels vs the hand-written `reduce.hip` incumbents they replace.
/// Bit-exactness (vs a CPU oracle) + the >=97% throughput gate the migration is conditioned on. GPU-only;
/// run serially (one HIP runtime): `cargo test -p hanzo-ml --features rocm dsl_norm -- --test-threads=1 --nocapture`.
#[cfg(all(test, feature = "rocm"))]
mod dsl_norm_bench {
    use super::*;
    use hanzo_rocm_kernels::kernel::{
        DslAddRmsNormKernel, DslRmsNormKernel, KernelSource, ReduceKernel,
    };

    const EPS: f32 = 1e-5;
    const ITERS: usize = 200;

    fn rnd(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed | 1;
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s % 2000) as f32 / 1000.0 - 1.0
            })
            .collect()
    }
    fn max_rel(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs() / x.abs().max(1e-6))
            .fold(0.0, f32::max)
    }
    fn rms_ref(x: &[f32], w: &[f32], rows: usize, n: usize) -> Vec<f32> {
        let mut o = vec![0f32; rows * n];
        for r in 0..rows {
            let b = r * n;
            let ss: f32 = (0..n).map(|i| x[b + i] * x[b + i]).sum();
            let d = (ss / n as f32 + EPS).sqrt();
            for i in 0..n {
                o[b + i] = x[b + i] / d * w[i];
            }
        }
        o
    }
    fn ptr(p: &*mut std::ffi::c_void) -> *mut std::ffi::c_void {
        p as *const *mut std::ffi::c_void as *mut std::ffi::c_void
    }
    /// Interleaved A/B timing, robust to the swarm's load noise: alternate the two kernels per round
    /// and take the MIN us/iter of each over `rounds` -- the least-contended round is the closest
    /// estimate of true kernel cost, and interleaving keeps both under the same instantaneous load.
    fn ab_min(
        device: &RocmDevice,
        iters: usize,
        rounds: usize,
        mut a: impl FnMut(),
        mut b: impl FnMut(),
    ) -> (f64, f64) {
        for _ in 0..10 {
            a();
            b();
        }
        device.synchronize().unwrap();
        let (mut ba, mut bb) = (f64::MAX, f64::MAX);
        for _ in 0..rounds {
            let t = std::time::Instant::now();
            for _ in 0..iters {
                a();
            }
            device.synchronize().unwrap();
            ba = ba.min(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
            let t = std::time::Instant::now();
            for _ in 0..iters {
                b();
            }
            device.synchronize().unwrap();
            bb = bb.min(t.elapsed().as_secs_f64() * 1e6 / iters as f64);
        }
        (ba, bb)
    }

    #[test]
    fn rms_norm_dsl_vs_incumbent() {
        let device = RocmDevice::new(0).unwrap();
        // DSL kernels read eps/ndim from device buffers + take an unused info dummy (built once, held).
        let eps_buf = device.clone_htod(&[EPS]).unwrap();
        let eps_ptr = eps_buf.as_ptr();
        let info_buf = device.alloc_zeros::<u32>(64).unwrap();
        let info_ptr = info_buf.as_ptr();
        println!("\n[rms_norm A/B]  shape      incumbent_us  dsl_us   dsl/inc   dsl_rel");
        for &(rows, n) in &[(1usize, 4096usize), (512, 4096), (1, 5120), (512, 5120)] {
            let x = rnd(rows * n, 0x1234 + rows as u64 * 7 + n as u64);
            let w = rnd(n, 0x9876 + n as u64);
            let want = rms_ref(&x, &w, rows, n);
            let x_mem = device.clone_htod(&x).unwrap();
            let w_mem = device.clone_htod(&w).unwrap();
            let out_inc = device.alloc::<f32>(rows * n).unwrap();
            let out_dsl = device.alloc::<f32>(rows * n).unwrap();
            let ndim_buf = device.clone_htod(&[n as u32]).unwrap();
            let ndim_ptr = ndim_buf.as_ptr();
            let (xp, wp, ip, dp) = (
                x_mem.as_ptr(),
                w_mem.as_ptr(),
                out_inc.as_ptr(),
                out_dsl.as_ptr(),
            );
            let n_cols = n as i32;
            let block_size: i32 = if n < 1024 { 32 } else { 1024 };
            let grid = rocm_rs::hip::Dim3 {
                x: rows as u32,
                y: 1,
                z: 1,
            };
            let inc_block = rocm_rs::hip::Dim3::from(block_size as u32);
            let dsl_block = rocm_rs::hip::Dim3 {
                x: 1024,
                y: 1,
                z: 1,
            };
            let inc = || unsafe {
                launch_kernel(
                    &device,
                    ReduceKernel::NAME,
                    ReduceKernel::CODE,
                    "rmsnorm_f32",
                    grid,
                    inc_block,
                    &mut [
                        ptr(&xp),
                        ptr(&ip),
                        ptr(&wp),
                        &n_cols as *const i32 as *mut std::ffi::c_void,
                        &block_size as *const i32 as *mut std::ffi::c_void,
                        &EPS as *const f32 as *mut std::ffi::c_void,
                    ],
                )
                .unwrap();
            };
            let dsl = || unsafe {
                launch_kernel_shmem(
                    &device,
                    DslRmsNormKernel::NAME,
                    DslRmsNormKernel::CODE,
                    "rms_norm_blk_f_f32",
                    grid,
                    dsl_block,
                    1024 * 4,
                    &mut [
                        ptr(&xp),
                        ptr(&wp),
                        ptr(&dp),
                        ptr(&eps_ptr),
                        ptr(&ndim_ptr),
                        ptr(&info_ptr),
                    ],
                )
                .unwrap();
            };
            let (inc_us, dsl_us) = ab_min(&device, ITERS, 40, inc, dsl);
            let dsl_rel = max_rel(&want, &device.clone_dtoh(&out_dsl).unwrap());
            let inc_rel = max_rel(&want, &device.clone_dtoh(&out_inc).unwrap());
            println!("            {rows}x{n}\t{inc_us:8.1}\t{dsl_us:7.1}\t{:.3}\tdsl={dsl_rel:.2e} inc={inc_rel:.2e}", dsl_us / inc_us);
            assert!(
                dsl_rel < 2e-3,
                "rms_norm DSL not bit-exact vs oracle at {rows}x{n}: {dsl_rel}"
            );
            if dsl_us > inc_us / 0.97 {
                println!(
                    "            NOTE rms_norm {rows}x{n} below 97%: {:.1}%",
                    100.0 * inc_us / dsl_us
                );
            }
        }
    }

    #[test]
    fn add_rms_norm_dsl_vs_incumbent() {
        let device = RocmDevice::new(0).unwrap();
        // DSL kernels read eps/ndim from device buffers + take an unused info dummy (built once, held).
        let eps_buf = device.clone_htod(&[EPS]).unwrap();
        let eps_ptr = eps_buf.as_ptr();
        let info_buf = device.alloc_zeros::<u32>(64).unwrap();
        let info_ptr = info_buf.as_ptr();
        println!("\n[add_rmsnorm A/B]  shape    incumbent_us  dsl_us   dsl/inc   y_rel");
        for &(rows, n) in &[(1usize, 4096usize), (512, 4096), (1, 5120), (512, 5120)] {
            let x = rnd(rows * n, 0x2345 + rows as u64 * 3 + n as u64);
            let res = rnd(rows * n, 0xBEEF + rows as u64 + n as u64 * 5);
            let alpha = rnd(n, 0x77 + n as u64);
            // oracle: s = x+res, y = rms_norm(s)*alpha
            let mut s_ref = vec![0f32; rows * n];
            for i in 0..rows * n {
                s_ref[i] = x[i] + res[i];
            }
            let want_y = rms_ref(&s_ref, &alpha, rows, n);
            let x_mem = device.clone_htod(&x).unwrap();
            let r_mem = device.clone_htod(&res).unwrap();
            let a_mem = device.clone_htod(&alpha).unwrap();
            let s_inc = device.alloc::<f32>(rows * n).unwrap();
            let y_inc = device.alloc::<f32>(rows * n).unwrap();
            let s_dsl = device.alloc::<f32>(rows * n).unwrap();
            let y_dsl = device.alloc::<f32>(rows * n).unwrap();
            let ndim_buf = device.clone_htod(&[n as u32]).unwrap();
            let ndim_ptr = ndim_buf.as_ptr();
            let (xp, rp, ap) = (x_mem.as_ptr(), r_mem.as_ptr(), a_mem.as_ptr());
            let (sip, yip, sdp, ydp) = (
                s_inc.as_ptr(),
                y_inc.as_ptr(),
                s_dsl.as_ptr(),
                y_dsl.as_ptr(),
            );
            let n_cols = n as i32;
            let block_size: i32 = if n < 1024 { 32 } else { 1024 };
            let grid = rocm_rs::hip::Dim3 {
                x: rows as u32,
                y: 1,
                z: 1,
            };
            let inc_block = rocm_rs::hip::Dim3::from(block_size as u32);
            let dsl_block = rocm_rs::hip::Dim3 {
                x: 1024,
                y: 1,
                z: 1,
            };
            // incumbent add_rmsnorm_f32(x, residual, sum_out, dst, alpha, n_cols, block_size, eps)
            let inc = || unsafe {
                launch_kernel(
                    &device,
                    ReduceKernel::NAME,
                    ReduceKernel::CODE,
                    "add_rmsnorm_f32",
                    grid,
                    inc_block,
                    &mut [
                        ptr(&xp),
                        ptr(&rp),
                        ptr(&sip),
                        ptr(&yip),
                        ptr(&ap),
                        &n_cols as *const i32 as *mut std::ffi::c_void,
                        &block_size as *const i32 as *mut std::ffi::c_void,
                        &EPS as *const f32 as *mut std::ffi::c_void,
                    ],
                )
                .unwrap();
            };
            // DSL add_rmsnorm_blk_f_f32(x, res, alpha, s_out, y, eps, ndim, info)
            let dsl = || unsafe {
                launch_kernel_shmem(
                    &device,
                    DslAddRmsNormKernel::NAME,
                    DslAddRmsNormKernel::CODE,
                    "add_rmsnorm_blk_f_f32",
                    grid,
                    dsl_block,
                    1024 * 4,
                    &mut [
                        ptr(&xp),
                        ptr(&rp),
                        ptr(&ap),
                        ptr(&sdp),
                        ptr(&ydp),
                        ptr(&eps_ptr),
                        ptr(&ndim_ptr),
                        ptr(&info_ptr),
                    ],
                )
                .unwrap();
            };
            let (inc_us, dsl_us) = ab_min(&device, ITERS, 40, inc, dsl);
            let y_rel = max_rel(&want_y, &device.clone_dtoh(&y_dsl).unwrap());
            let s_rel = max_rel(&s_ref, &device.clone_dtoh(&s_dsl).unwrap());
            println!("               {rows}x{n}\t{inc_us:8.1}\t{dsl_us:7.1}\t{:.3}\ty={y_rel:.2e} s={s_rel:.2e}", dsl_us / inc_us);
            assert!(
                y_rel < 2e-3 && s_rel < 2e-3,
                "add_rmsnorm DSL not bit-exact at {rows}x{n}: y={y_rel} s={s_rel}"
            );
            if dsl_us > inc_us / 0.97 {
                println!(
                    "               NOTE add_rmsnorm {rows}x{n} below 97%: {:.1}%",
                    100.0 * inc_us / dsl_us
                );
            }
        }
    }

    /// COHERENCE GATE (decode residual stream): the fused `add_rmsnorm_f16` must be BIT-IDENTICAL to a
    /// separate elementwise add (`badd_f16`, which rounds `x+residual` to f16) followed by `rmsnorm_f16`
    /// (which squares the stored f16). If these agree bit-for-bit, wiring `forward_of_sum` to the fused
    /// F16 kernel changes only launch count -- never model output -- so qwen2/qwen3 decode stays coherent.
    #[test]
    fn add_rmsnorm_f16_bit_identical_to_separate() {
        let device = RocmDevice::new(0).unwrap();
        for &(rows, n) in &[(1usize, 4096usize), (7, 4096), (1, 5120), (512, 2560), (33, 2560)] {
            let xf = rnd(rows * n, 0x51 + rows as u64 * 7 + n as u64);
            let rf = rnd(rows * n, 0xC0DE + rows as u64 + n as u64 * 3);
            let af = rnd(n, 0xA1 + n as u64);
            let x16: Vec<f16> = xf.iter().map(|&v| f16::from_f32(v)).collect();
            let r16: Vec<f16> = rf.iter().map(|&v| f16::from_f32(v)).collect();
            let a16: Vec<f16> = af.iter().map(|&v| f16::from_f32(v)).collect();
            // Reference sum = (half)((float)x + (float)residual), byte-for-byte the badd_f16 kernel.
            let sum_ref: Vec<f16> = (0..rows * n)
                .map(|i| f16::from_f32(x16[i].to_f32() + r16[i].to_f32()))
                .collect();

            let x_dev = device.clone_htod(&x16).unwrap();
            let r_dev = device.clone_htod(&r16).unwrap();
            let a_dev = device.clone_htod(&a16).unwrap();
            let sum_ref_dev = device.clone_htod(&sum_ref).unwrap();
            let sum_fused = device.alloc::<f16>(rows * n).unwrap();
            let dst_fused = device.alloc::<f16>(rows * n).unwrap();
            let dst_ref = device.alloc::<f16>(rows * n).unwrap();

            let n_cols = n as i32;
            let block_size: i32 = if n < 1024 { 32 } else { 1024 };
            let grid = rocm_rs::hip::Dim3 { x: rows as u32, y: 1, z: 1 };
            let block = rocm_rs::hip::Dim3::from(block_size as u32);
            let (xp, rp, ap, srp) = (
                x_dev.as_ptr(),
                r_dev.as_ptr(),
                a_dev.as_ptr(),
                sum_ref_dev.as_ptr(),
            );
            let (sfp, dfp, drp) = (sum_fused.as_ptr(), dst_fused.as_ptr(), dst_ref.as_ptr());

            // fused add_rmsnorm_f16(x, residual, sum_out, dst, alpha, n_cols, block_size, eps)
            unsafe {
                launch_kernel(
                    &device,
                    ReduceKernel::NAME,
                    ReduceKernel::CODE,
                    "add_rmsnorm_f16",
                    grid,
                    block,
                    &mut [
                        ptr(&xp),
                        ptr(&rp),
                        ptr(&sfp),
                        ptr(&dfp),
                        ptr(&ap),
                        &n_cols as *const i32 as *mut std::ffi::c_void,
                        &block_size as *const i32 as *mut std::ffi::c_void,
                        &EPS as *const f32 as *mut std::ffi::c_void,
                    ],
                )
                .unwrap();
            }
            // separate rmsnorm_f16(sum_ref, dst, alpha, n_cols, block_size, eps)
            unsafe {
                launch_kernel(
                    &device,
                    ReduceKernel::NAME,
                    ReduceKernel::CODE,
                    "rmsnorm_f16",
                    grid,
                    block,
                    &mut [
                        ptr(&srp),
                        ptr(&drp),
                        ptr(&ap),
                        &n_cols as *const i32 as *mut std::ffi::c_void,
                        &block_size as *const i32 as *mut std::ffi::c_void,
                        &EPS as *const f32 as *mut std::ffi::c_void,
                    ],
                )
                .unwrap();
            }

            let sum_got = device.clone_dtoh(&sum_fused).unwrap();
            let dst_got = device.clone_dtoh(&dst_fused).unwrap();
            let dst_want = device.clone_dtoh(&dst_ref).unwrap();
            for i in 0..rows * n {
                assert_eq!(
                    sum_got[i].to_bits(),
                    sum_ref[i].to_bits(),
                    "add_rmsnorm_f16 sum bit mismatch at {rows}x{n}[{i}]"
                );
                assert_eq!(
                    dst_got[i].to_bits(),
                    dst_want[i].to_bits(),
                    "add_rmsnorm_f16 normed bit mismatch at {rows}x{n}[{i}]"
                );
            }
            println!("[add_rmsnorm f16 bit-exact] {rows}x{n} OK");
        }
    }
}
