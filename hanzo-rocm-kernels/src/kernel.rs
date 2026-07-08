//! Kernel sources and types for ROCm operations.

/// Trait for kernel source definitions.
///
/// This trait simplifies the old enum-based approach by using
/// compile-time constants instead of runtime matching.
pub trait KernelSource {
    /// Unique kernel name (used for caching)
    const NAME: &'static str;
    /// The HIP source code
    const CODE: &'static str;
}

/// Binary operations kernel source
pub struct BinaryKernel;
impl KernelSource for BinaryKernel {
    const NAME: &'static str = "binary";
    const CODE: &'static str = include_str!("kernels/binary.hip");
}

/// Unary operations kernel source
pub struct UnaryKernel;
impl KernelSource for UnaryKernel {
    const NAME: &'static str = "unary";
    const CODE: &'static str = include_str!("kernels/unary.hip");
}

/// Affine operations kernel source
pub struct AffineKernel;
impl KernelSource for AffineKernel {
    const NAME: &'static str = "affine";
    const CODE: &'static str = include_str!("kernels/affine.hip");
}

/// Fill operations kernel source
pub struct FillKernel;
impl KernelSource for FillKernel {
    const NAME: &'static str = "fill";
    const CODE: &'static str = include_str!("kernels/fill.hip");
}

/// Reduce operations kernel source
pub struct ReduceKernel;
impl KernelSource for ReduceKernel {
    const NAME: &'static str = "reduce";
    const CODE: &'static str = include_str!("kernels/reduce.hip");
}

/// Convolution operations kernel source
pub struct ConvKernel;
impl KernelSource for ConvKernel {
    const NAME: &'static str = "conv";
    const CODE: &'static str = include_str!("kernels/conv.hip");
}

/// Indexing operations kernel source
pub struct IndexingKernel;
impl KernelSource for IndexingKernel {
    const NAME: &'static str = "indexing";
    const CODE: &'static str = include_str!("kernels/indexing.hip");
}

/// Cast operations kernel source
pub struct CastKernel;
impl KernelSource for CastKernel {
    const NAME: &'static str = "cast";
    const CODE: &'static str = include_str!("kernels/cast.hip");
}

pub struct TernaryKernel;
impl KernelSource for TernaryKernel {
    const NAME: &'static str = "ternary";
    const CODE: &'static str = include_str!("kernels/ternary.hip");
}

/// Argsort (bitonic, one block per row) kernel source
pub struct SortKernel;
impl KernelSource for SortKernel {
    const NAME: &'static str = "sort";
    const CODE: &'static str = include_str!("kernels/sort.hip");
}

/// Native quantized matvec kernels (Q8_0, decode path)
pub struct QuantKernel;
impl KernelSource for QuantKernel {
    const NAME: &'static str = "quant";
    const CODE: &'static str = include_str!("kernels/quant.hip");
}

/// Native positions-aware rotary embedding (neox and gpt-j) kernel source
pub struct RopeKernel;
impl KernelSource for RopeKernel {
    const NAME: &'static str = "rope";
    const CODE: &'static str = include_str!("kernels/rope.hip");
}

/// Matrix-core (WMMA) flash-attention forward kernel source
pub struct FlashKernel;
impl KernelSource for FlashKernel {
    const NAME: &'static str = "flash";
    const CODE: &'static str = include_str!("kernels/flash.hip");
}

// DSL-lowered kernels (the ROCm collapse): one authored `#[kernel]` in hanzo-kernel/src/*.rs, lowered
// to HIP via cubecl-hip and checked in here. Compiled + launched through the SAME hipcc pipeline as the
// hand-written kernels above -- the DSL is a codegen frontend, not a second runtime. See the DSL launch
// helper in `rocm_backend`. Entry points are cubecl's dtype-suffixed names (e.g. `rms_norm_blk_f_f32`).

/// DSL block-per-row RMSNorm (f32 + f16 I/O). Replaces the hand-written `rmsnorm` in `reduce.hip`.
pub struct DslRmsNormKernel;
impl KernelSource for DslRmsNormKernel {
    const NAME: &'static str = "dsl_rms_norm";
    const CODE: &'static str = include_str!("kernels/dsl_rms_norm.hip");
}

/// DSL fused residual-add + RMSNorm (f32). Replaces the hand-written `add_rmsnorm` in `reduce.hip`.
pub struct DslAddRmsNormKernel;
impl KernelSource for DslAddRmsNormKernel {
    const NAME: &'static str = "dsl_add_rmsnorm";
    const CODE: &'static str = include_str!("kernels/dsl_add_rmsnorm.hip");
}

/// Binary operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Minimum,
    Maximum,
}

impl BinaryOp {
    /// Get the kernel function name for this operation
    pub fn kernel_name(&self) -> &'static str {
        match self {
            BinaryOp::Add => "badd",
            BinaryOp::Sub => "bsub",
            BinaryOp::Mul => "bmul",
            BinaryOp::Div => "bdiv",
            BinaryOp::Minimum => "bminimum",
            BinaryOp::Maximum => "bmaximum",
        }
    }
}

/// Unary operation types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Copy,
    Relu,
    Sigmoid,
    Tan,
    Exp,
    Log,
    Sin,
    Cos,
    Sqrt,
    Abs,
    Neg,
    Recip,
    Floor,
    Ceil,
    Round,
    Gelu,
    Silu,
    Erf,
}

impl UnaryOp {
    /// Get the kernel function name for this operation
    pub fn kernel_name(&self) -> &'static str {
        match self {
            UnaryOp::Copy => "ucopy",
            UnaryOp::Relu => "urelu",
            UnaryOp::Sigmoid => "usigmoid",
            UnaryOp::Tan => "utan",
            UnaryOp::Exp => "uexp",
            UnaryOp::Log => "ulog",
            UnaryOp::Sin => "usin",
            UnaryOp::Cos => "ucos",
            UnaryOp::Sqrt => "usqrt",
            UnaryOp::Abs => "uabs",
            UnaryOp::Neg => "uneg",
            UnaryOp::Recip => "urecip",
            UnaryOp::Floor => "ufloor",
            UnaryOp::Ceil => "uceil",
            UnaryOp::Round => "uround",
            UnaryOp::Gelu => "ugelu",
            UnaryOp::Silu => "usilu",
            UnaryOp::Erf => "uerf",
        }
    }
}

/// Data types supported by kernels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    F16,
    F32,
    F64,
    I64,
    U32,
    U8,
}

impl DType {
    /// Get the size of this dtype in bytes
    pub fn size_in_bytes(&self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 => 4,
            Self::I64 => 8,
            Self::BF16 => 2,
            Self::F16 => 2,
            Self::F32 => 4,
            Self::F64 => 8,
        }
    }
}

/// Get the dtype suffix for kernel function naming
pub fn dtype_suffix<T: Copy + Send + Sync + 'static>() -> &'static str {
    let type_name = std::any::type_name::<T>();
    if type_name.contains("f32") {
        "f32"
    } else if type_name.contains("f64") {
        "f64"
    } else if type_name.contains("u8") {
        "u8"
    } else if type_name.contains("u32") {
        "u32"
    } else if type_name.contains("i64") {
        "i64"
    } else {
        panic!("Unsupported dtype for kernel: {}", type_name)
    }
}
