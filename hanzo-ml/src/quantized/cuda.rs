use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{
    backend::{BackendDevice, BackendStorage},
    cuda_backend::WrapErr,
};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result};
use half::{bf16, f16};

use cudarc::driver::{CudaSlice, CudaStream, CudaView, DevicePtr, PushKernelArg, SyncOnDrop};

#[derive(Clone, Debug)]
struct PaddedCudaSlice {
    inner: CudaSlice<u8>,
    len: usize,
}

#[derive(Clone, Debug)]
pub struct QCudaStorage {
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    device: CudaDevice,
}

pub(crate) static FORCE_DMMV: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);

pub fn set_force_dmmv(f: bool) {
    FORCE_DMMV.store(f, std::sync::atomic::Ordering::Relaxed)
}

// Opt-in switch for the modern llama-style mmq/mmvq path (int8 MMA matrix-core + stream-K) ported
// in fast_mmq.rs / fast_mmvq.rs. Off by default until validated on NVIDIA hardware; enable with
// CUDA_FAST_MMQ=1 (env) or `set_fast_mmq(true)` (programmatic — a model that needs the int8
// dp4a path for *correctness*, e.g. DeepSeek-V4's W8A8 QAT, turns it on at load). When on,
// QCudaStorage::fwd tries it first and falls back to the legacy on-GPU q8_1 kernels for any
// dtype/shape it doesn't support. Mirrors `FORCE_DMMV`: an AtomicBool seeded once from the env.
pub(crate) static FAST_MMQ: std::sync::atomic::AtomicBool =
    std::sync::atomic::AtomicBool::new(false);
static FAST_MMQ_INIT: std::sync::Once = std::sync::Once::new();

/// Programmatically enable/disable the fast int8 mmq/mmvq path (overrides the env default,
/// and pins it so a later first-read won't reseed from env). Validated on NVIDIA sm_121
/// (GB10) for DeepSeek-V4 — exact W8A8 (int32 accumulate) matching antirez/ds4.
pub fn set_fast_mmq(f: bool) {
    FAST_MMQ_INIT.call_once(|| {});
    FAST_MMQ.store(f, std::sync::atomic::Ordering::Relaxed);
}

pub(crate) fn fast_mmq_enabled() -> bool {
    FAST_MMQ_INIT.call_once(|| {
        let env = matches!(
            std::env::var("CUDA_FAST_MMQ").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        );
        FAST_MMQ.store(env, std::sync::atomic::Ordering::Relaxed);
    });
    FAST_MMQ.load(std::sync::atomic::Ordering::Relaxed)
}

#[derive(Clone, Copy, Debug, Default)]
pub struct CudaProfileSnapshot {
    pub dp4a_decode_ms: f64,
    pub dp4a_decode_calls: usize,
    pub dp4a_verify_ms: f64,
    pub dp4a_verify_calls: usize,
    pub dp4a_ge64m_ms: f64,
    pub dp4a_ge64m_calls: usize,
    pub bf16_dequant_ms: f64,
    pub bf16_dequant_calls: usize,
    pub bf16_gemm_ms: f64,
    pub mmq_quant_ms: f64,
    pub mmq_kernel_ms: f64,
    pub mmq_cast_ms: f64,
    pub mmq_calls: usize,
    pub fwht_ms: f64,
    pub fwht_calls: usize,
}

static CUDA_PROFILE_ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
static PROF_DP4A_DECODE_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_DP4A_DECODE_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static PROF_DP4A_VERIFY_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_DP4A_VERIFY_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static PROF_DP4A_GE64M_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_DP4A_GE64M_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static PROF_BF16_DEQUANT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_BF16_DEQUANT_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static PROF_BF16_GEMM_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub(crate) static PROF_MMQ_QUANT_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static PROF_MMQ_KERNEL_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static PROF_MMQ_CAST_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static PROF_MMQ_CALLS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);
static PROF_FWHT_NS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static PROF_FWHT_CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Selects the `b * m` threshold below which CUDA falls back to the DP4A
/// vector kernel instead of MMQ tensor-core GEMM.
///
/// GB10 / Blackwell (`SM >= 100`, `cc >= 1000`) crossovers were measured at
/// `m=2` for `PQ2_0` and `m=4` for `PTQ1_0`; earlier NVIDIA architectures keep
/// the conservative `8` threshold unless overridden via
/// `HANZO_PQ2_MMQ_MAX_BM` / `HANZO_PTQ1_MMQ_MAX_BM`.
fn mmq_max_vec_bm(dtype: GgmlDType, compute_cap: i32) -> usize {
    let env_key = match dtype {
        GgmlDType::PQ2_0 => Some("HANZO_PQ2_MMQ_MAX_BM"),
        GgmlDType::PTQ1_0 => Some("HANZO_PTQ1_MMQ_MAX_BM"),
        GgmlDType::Q1_0 => Some("HANZO_Q1_MMQ_MAX_BM"),
        GgmlDType::Q2_0 => Some("HANZO_Q2_MMQ_MAX_BM"),
        _ => None,
    };
    if let Some(key) = env_key {
        if let Ok(v) = std::env::var(key) {
            if let Ok(parsed) = v.parse::<usize>() {
                return parsed.max(1);
            }
        }
    }
    match dtype {
        GgmlDType::PQ2_0 | GgmlDType::Q2_0 if compute_cap >= 1000 => 1,
        GgmlDType::PTQ1_0 | GgmlDType::Q1_0 if compute_cap >= 1000 => 3,
        _ => 8,
    }
}

pub fn cuda_profile_enabled() -> bool {
    *CUDA_PROFILE_ENABLED.get_or_init(|| {
        matches!(
            std::env::var("HANZO_CUDA_PROFILE").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    })
}

pub fn cuda_profile_reset() {
    use std::sync::atomic::Ordering::Relaxed;
    PROF_DP4A_DECODE_NS.store(0, Relaxed);
    PROF_DP4A_DECODE_CALLS.store(0, Relaxed);
    PROF_DP4A_VERIFY_NS.store(0, Relaxed);
    PROF_DP4A_VERIFY_CALLS.store(0, Relaxed);
    PROF_DP4A_GE64M_NS.store(0, Relaxed);
    PROF_DP4A_GE64M_CALLS.store(0, Relaxed);
    PROF_BF16_DEQUANT_NS.store(0, Relaxed);
    PROF_BF16_DEQUANT_CALLS.store(0, Relaxed);
    PROF_BF16_GEMM_NS.store(0, Relaxed);
    PROF_MMQ_QUANT_NS.store(0, Relaxed);
    PROF_MMQ_KERNEL_NS.store(0, Relaxed);
    PROF_MMQ_CAST_NS.store(0, Relaxed);
    PROF_MMQ_CALLS.store(0, Relaxed);
    PROF_FWHT_NS.store(0, Relaxed);
    PROF_FWHT_CALLS.store(0, Relaxed);
}

pub fn cuda_profile_snapshot() -> CudaProfileSnapshot {
    use std::sync::atomic::Ordering::Relaxed;
    CudaProfileSnapshot {
        dp4a_decode_ms: PROF_DP4A_DECODE_NS.load(Relaxed) as f64 / 1e6,
        dp4a_decode_calls: PROF_DP4A_DECODE_CALLS.load(Relaxed),
        dp4a_verify_ms: PROF_DP4A_VERIFY_NS.load(Relaxed) as f64 / 1e6,
        dp4a_verify_calls: PROF_DP4A_VERIFY_CALLS.load(Relaxed),
        dp4a_ge64m_ms: PROF_DP4A_GE64M_NS.load(Relaxed) as f64 / 1e6,
        dp4a_ge64m_calls: PROF_DP4A_GE64M_CALLS.load(Relaxed),
        bf16_dequant_ms: PROF_BF16_DEQUANT_NS.load(Relaxed) as f64 / 1e6,
        bf16_dequant_calls: PROF_BF16_DEQUANT_CALLS.load(Relaxed),
        bf16_gemm_ms: PROF_BF16_GEMM_NS.load(Relaxed) as f64 / 1e6,
        mmq_quant_ms: PROF_MMQ_QUANT_NS.load(Relaxed) as f64 / 1e6,
        mmq_kernel_ms: PROF_MMQ_KERNEL_NS.load(Relaxed) as f64 / 1e6,
        mmq_cast_ms: PROF_MMQ_CAST_NS.load(Relaxed) as f64 / 1e6,
        mmq_calls: PROF_MMQ_CALLS.load(Relaxed),
        fwht_ms: PROF_FWHT_NS.load(Relaxed) as f64 / 1e6,
        fwht_calls: PROF_FWHT_CALLS.load(Relaxed),
    }
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_GET_ROWS_BLOCK_SIZE: usize = 256;
pub const MATRIX_ROW_PADDING: usize = 512;

fn ceil_div(p: usize, q: usize) -> usize {
    p.div_ceil(q)
}

fn pad(p: usize, q: usize) -> usize {
    ceil_div(p, q) * q
}

fn quantize_q8_1(
    src: &CudaView<f32>,
    dst: &mut CudaSlice<u8>,
    k: usize,
    ky: usize,
    dev: &CudaDevice,
) -> Result<()> {
    let kx_padded = pad(k, MATRIX_ROW_PADDING);
    let num_blocks = ceil_div(kx_padded, CUDA_QUANTIZE_BLOCK_SIZE);

    let total_rows = ky;
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = kx_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;

    const CHUNK_SIZE: usize = 65535; // gridDim.y limit
    let func = dev.get_or_load_func("quantize_q8_1", &hanzo_kernels::QUANTIZED)?;

    let mut rows_processed = 0;
    while rows_processed < total_rows {
        // --- calculate the number of rows for this chunk ---
        let remaining_rows = total_rows - rows_processed;
        // This is our gridDim.y, now <= 65535
        let rows_in_chunk = std::cmp::min(CHUNK_SIZE, remaining_rows);

        // --- slice the source (f32) tensor by elements ---
        let src_start_elem = rows_processed * k;
        let src_num_elems = rows_in_chunk * k;
        let src_chunk = src.slice(src_start_elem..(src_start_elem + src_num_elems));

        // --- slice the destination (u8) tensor by bytes ---
        let dst_start_byte = rows_processed * dst_row_size_bytes;
        let dst_num_bytes = rows_in_chunk * dst_row_size_bytes;
        let dst_chunk = dst.slice(dst_start_byte..(dst_start_byte + dst_num_bytes));

        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (num_blocks as u32, rows_in_chunk as u32, 1),
            block_dim: (CUDA_QUANTIZE_BLOCK_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        let mut builder = func.builder();
        builder.arg(&src_chunk);
        builder.arg(&dst_chunk);
        barg!(builder, k as i32, kx_padded as i32);
        unsafe { builder.launch(cfg) }.w()?;

        rows_processed += rows_in_chunk;
    }

    Ok(())
}

fn dequantize_f32(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if matches!(
        dtype,
        GgmlDType::PQ2_0 | GgmlDType::PTQ1_0 | GgmlDType::Q1_0 | GgmlDType::Q2_0
    ) {
        let kernel_name = match dtype {
            GgmlDType::PQ2_0 => "dequantize_block_pq2_0_f32",
            GgmlDType::PTQ1_0 => "dequantize_block_ptq1_0_f32",
            GgmlDType::Q1_0 => "dequantize_block_q1_0_f32",
            GgmlDType::Q2_0 => "dequantize_block_q2_0_f32",
            _ => unreachable!(),
        };
        let bs = dtype.block_size();
        let nblocks = elem_count / bs;
        let func = dev.get_or_load_func(kernel_name, &hanzo_kernels::IQUANT_MMVQ)?;
        let dst = unsafe { dev.alloc::<f32>(elem_count)? };
        if nblocks > 0 {
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nblocks.min(65535) as u32, 1, 1),
                block_dim: (bs as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            builder.arg(&data.inner);
            builder.arg(&dst);
            barg!(builder, nblocks as i64);
            unsafe { builder.launch(cfg) }.w()?;
        }
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f32", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f32", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f32",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f32", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f32", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f32", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f32", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f32", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f32", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f32", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &hanzo_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_f16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    if matches!(
        dtype,
        GgmlDType::PQ2_0 | GgmlDType::PTQ1_0 | GgmlDType::Q1_0 | GgmlDType::Q2_0
    ) {
        let kernel_name = match dtype {
            GgmlDType::PQ2_0 => "dequantize_block_pq2_0_f16",
            GgmlDType::PTQ1_0 => "dequantize_block_ptq1_0_f16",
            GgmlDType::Q1_0 => "dequantize_block_q1_0_f16",
            GgmlDType::Q2_0 => "dequantize_block_q2_0_f16",
            _ => unreachable!(),
        };
        let bs = dtype.block_size();
        let nblocks = elem_count / bs;
        let func = dev.get_or_load_func(kernel_name, &hanzo_kernels::IQUANT_MMVQ)?;
        let dst = unsafe { dev.alloc::<f16>(elem_count)? };
        if nblocks > 0 {
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (nblocks.min(65535) as u32, 1, 1),
                block_dim: (bs as u32, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            builder.arg(&data.inner);
            builder.arg(&dst);
            barg!(builder, nblocks as i64);
            unsafe { builder.launch(cfg) }.w()?;
        }
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    let nb = elem_count.div_ceil(256);
    let (kernel_name, is_k, block_dim, num_blocks) = match dtype {
        GgmlDType::Q4_0 => ("dequantize_block_q4_0_f16", false, 32, nb),
        GgmlDType::Q4_1 => ("dequantize_block_q4_1_f16", false, 32, nb),
        GgmlDType::Q5_0 => (
            "dequantize_block_q5_0_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q5_1 => (
            "dequantize_block_q5_1_f16",
            false,
            CUDA_DEQUANTIZE_BLOCK_SIZE,
            ceil_div(elem_count, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE),
        ),
        GgmlDType::Q8_0 => ("dequantize_block_q8_0_f16", false, 32, nb),
        GgmlDType::Q2K => ("dequantize_block_q2_K_f16", true, 64, nb),
        GgmlDType::Q3K => ("dequantize_block_q3_K_f16", true, 64, nb),
        GgmlDType::Q4K => ("dequantize_block_q4_K_f16", true, 32, nb),
        GgmlDType::Q5K => ("dequantize_block_q5_K_f16", true, 64, nb),
        GgmlDType::Q6K => ("dequantize_block_q6_K_f16", true, 64, nb),
        GgmlDType::Q8K => ("dequantize_block_q8_K_f16", true, 32, nb),
        _ => crate::bail!("unsupported dtype for dequantize {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &hanzo_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f16>(elem_count)? };
    // See e.g.
    // https://github.com/ggerganov/llama.cpp/blob/cbbd1efa06f8c09f9dff58ff9d9af509cc4c152b/ggml-cuda.cu#L7270
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks as u32, 1, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };

    if is_k {
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        unsafe { builder.launch(cfg) }.w()?;
    } else {
        let nb32 = match dtype {
            GgmlDType::Q5_0 | GgmlDType::Q5_1 => elem_count,
            _ => elem_count / 32,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nb32 as i32);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_bf16(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    elem_count: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let kernel_name = match dtype {
        GgmlDType::PQ2_0 => "dequantize_block_pq2_0_bf16",
        GgmlDType::PTQ1_0 => "dequantize_block_ptq1_0_bf16",
        GgmlDType::Q1_0 => "dequantize_block_q1_0_bf16",
        GgmlDType::Q2_0 => "dequantize_block_q2_0_bf16",
        _ => crate::bail!("unsupported dtype for dequantize_bf16 {dtype:?}"),
    };
    let bs = dtype.block_size();
    let nblocks = elem_count / bs;
    let func = dev.get_or_load_func(kernel_name, &hanzo_kernels::IQUANT_MMVQ)?;
    let dst = unsafe { dev.alloc::<bf16>(elem_count)? };
    if nblocks > 0 {
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (nblocks.min(65535) as u32, 1, 1),
            block_dim: (bs as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        builder.arg(&data.inner);
        builder.arg(&dst);
        barg!(builder, nblocks as i64);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn get_rows(
    data: &PaddedCudaSlice,
    dtype: GgmlDType,
    hidden: usize,
    ids: &CudaView<u32>,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let (kernel_name, block_dim, block_num_y, can_stride_y) = match dtype {
        GgmlDType::F32 => (
            "get_rows_f32",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::F16 => (
            "get_rows_f16",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::BF16 => (
            "get_rows_bf16",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_0 => (
            "get_rows_q4_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q4_1 => (
            "get_rows_q4_1",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_0 => (
            "get_rows_q5_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q5_1 => (
            "get_rows_q5_1",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q8_0 => (
            "get_rows_q8_0",
            CUDA_GET_ROWS_BLOCK_SIZE,
            ceil_div(hidden, 2 * CUDA_GET_ROWS_BLOCK_SIZE),
            true,
        ),
        GgmlDType::Q2K => ("get_rows_q2_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q3K => ("get_rows_q3_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q4K => ("get_rows_q4_K", 32, hidden / dtype.block_size(), false),
        GgmlDType::Q5K => ("get_rows_q5_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::Q6K => ("get_rows_q6_K", 64, hidden / dtype.block_size(), false),
        GgmlDType::PQ2_0 => ("get_rows_pq2_0", 128, hidden / dtype.block_size(), true),
        GgmlDType::PTQ1_0 => ("get_rows_ptq1_0", 128, hidden / dtype.block_size(), true),
        GgmlDType::Q1_0 => ("get_rows_q1_0", 128, hidden / dtype.block_size(), true),
        GgmlDType::Q2_0 => ("get_rows_q2_0", 64, hidden / dtype.block_size(), true),
        _ => crate::bail!("unsupported dtype for CUDA quantized embedding {dtype:?}"),
    };
    let ptx = if matches!(
        dtype,
        GgmlDType::PQ2_0 | GgmlDType::PTQ1_0 | GgmlDType::Q1_0 | GgmlDType::Q2_0
    ) {
        &hanzo_kernels::IQUANT_MMVQ
    } else {
        &hanzo_kernels::QUANTIZED
    };
    let func = dev.get_or_load_func(kernel_name, ptx)?;
    let ids_len = ids.len();
    let dst = unsafe { dev.alloc::<f32>(ids_len * hidden)? };
    if ids_len == 0 {
        return Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()));
    }
    if !can_stride_y && block_num_y > u16::MAX as usize {
        crate::bail!("quantized embedding hidden size {hidden} exceeds CUDA grid y limit")
    }
    let grid_y = if can_stride_y {
        block_num_y.min(u16::MAX as usize)
    } else {
        block_num_y
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (ids_len as u32, grid_y as u32, 1),
        block_dim: (block_dim as u32, 1, 1),
        shared_mem_bytes: 0,
    };
    let row_stride = hidden * dtype.type_size() / dtype.block_size();

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(ids);
    builder.arg(&dst);
    barg!(builder, hidden as i64, row_stride);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

fn dequantize_mul_mat_vec(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "dequantize_mul_mat_vec_q4_0_cuda",
        GgmlDType::Q4_1 => "dequantize_mul_mat_vec_q4_1_cuda",
        GgmlDType::Q5_0 => "dequantize_mul_mat_vec_q5_0_cuda",
        GgmlDType::Q5_1 => "dequantize_mul_mat_vec_q5_1_cuda",
        GgmlDType::Q8_0 => "dequantize_mul_mat_vec_q8_0_cuda",
        GgmlDType::Q2K => "dequantize_mul_mat_vec_q2_k",
        GgmlDType::Q3K => "dequantize_mul_mat_vec_q3_k",
        GgmlDType::Q4K => "dequantize_mul_mat_vec_q4_k",
        GgmlDType::Q5K => "dequantize_mul_mat_vec_q5_k",
        GgmlDType::Q6K => "dequantize_mul_mat_vec_q6_k",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &hanzo_kernels::QUANTIZED)?;
    let dst = unsafe { dev.alloc::<f32>(nrows)? };
    let block_num_y = ceil_div(nrows, GGML_CUDA_MMV_Y);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (block_num_y as u32, 1, 1),
        block_dim: (WARP_SIZE as u32, GGML_CUDA_MMV_Y as u32, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(y);
    builder.arg(&dst);
    barg!(builder, ncols as i32, nrows as i32);
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// Fused MoE router: softmax(logits over ALL experts) -> top-k by descending logit -> optional
/// renorm of the top-k weights to sum 1 (`norm_topk_prob`). ONE `moe_route` kernel (one block/token)
/// replaces the candle softmax->asort->narrow->sum->div op chain (~6 launches/layer). `logits` is a
/// contiguous [ntok, n_experts] f32 view; returns (ids [ntok,topk] u32, weights [ntok,topk] f32).
/// The full-softmax denominator makes norm=0 byte-faithful to a plain softmax-then-topk and to the
/// ROCm kernel of the same name (one source of truth for the routing math).
pub fn moe_route(
    logits: &CudaView<f32>,
    ntok: usize,
    n_experts: usize,
    topk: usize,
    norm: bool,
    dev: &CudaDevice,
) -> Result<(CudaStorage, CudaStorage)> {
    const MOE_ROUTE_MAX_E: usize = 256;
    const MOE_ROUTE_MAX_K: usize = 32;
    if n_experts > MOE_ROUTE_MAX_E {
        crate::bail!("moe_route: n_experts {n_experts} exceeds kernel max {MOE_ROUTE_MAX_E}");
    }
    if topk > MOE_ROUTE_MAX_K {
        crate::bail!("moe_route: topk {topk} exceeds kernel max {MOE_ROUTE_MAX_K}");
    }
    if logits.len() != ntok * n_experts {
        crate::bail!(
            "moe_route: logits len {} != ntok*n_experts {}",
            logits.len(),
            ntok * n_experts
        );
    }
    let func = dev.get_or_load_func("moe_route", &hanzo_kernels::QUANTIZED)?;
    let ids = unsafe { dev.alloc::<u32>(ntok * topk)? };
    let w = unsafe { dev.alloc::<f32>(ntok * topk)? };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (ntok as u32, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut builder = func.builder();
    barg!(
        builder,
        ntok as i32,
        n_experts as i32,
        topk as i32,
        i32::from(norm)
    );
    builder.arg(logits);
    builder.arg(&ids);
    builder.arg(&w);
    unsafe { builder.launch(cfg) }.w()?;
    Ok((
        CudaStorage::wrap_cuda_slice(ids, dev.clone()),
        CudaStorage::wrap_cuda_slice(w, dev.clone()),
    ))
}

fn mul_mat_vec_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < ncols * nrows {
        crate::bail!("unexpected data size {}, ncols {ncols} {nrows}", data_elems)
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} {nrows}", y.len())
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}")
    }
    // Start by quantizing y
    let ncols_padded = pad(ncols, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        b_size * ncols_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, ncols, b_size, dev)?;

    let kernel_name = match dtype {
        GgmlDType::Q4_0 => "mul_mat_vec_q4_0_q8_1_cuda",
        GgmlDType::Q4_1 => "mul_mat_vec_q4_1_q8_1_cuda",
        GgmlDType::Q5_0 => "mul_mat_vec_q5_0_q8_1_cuda",
        GgmlDType::Q5_1 => "mul_mat_vec_q5_1_q8_1_cuda",
        GgmlDType::Q8_0 => "mul_mat_vec_q8_0_q8_1_cuda",
        GgmlDType::Q2K => "mul_mat_vec_q2_K_q8_1_cuda",
        GgmlDType::Q3K => "mul_mat_vec_q3_K_q8_1_cuda",
        GgmlDType::Q4K => "mul_mat_vec_q4_K_q8_1_cuda",
        GgmlDType::Q5K => "mul_mat_vec_q5_K_q8_1_cuda",
        GgmlDType::Q6K => "mul_mat_vec_q6_K_q8_1_cuda",
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let kernel_name = format!("{kernel_name}{b_size}");
    let func = dev.get_or_load_func(&kernel_name, &hanzo_kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(nrows * b_size)?;
    // https://github.com/ggerganov/llama.cpp/blob/facb8b56f8fd3bb10a693bf0943ae9d69d0828ef/ggml-cuda/mmvq.cu#L98
    let (nblocks, nwarps) = match b_size {
        1 => (nrows as u32, 4),
        2..=4 => ((nrows as u32).div_ceil(2), 4),
        5..=8 => ((nrows as u32).div_ceil(2), 2),
        _ => crate::bail!("unexpected bsize {b_size}"),
    };
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, 1, 1),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(&data.inner);
    builder.arg(&y_q8_1);
    builder.arg(&dst);
    barg!(
        builder,
        /* ncols_x */ ncols as i32,
        /* nrows_x */ nrows as i32,
        /* nrows_y */ ncols_padded as i32,
        /* nrows_dst */ nrows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// i-quant / ternary codebook kernel suffix for the native dp4a decode (qmatvec_dp4a_<suffix>_f32 in
/// iquant_mmvq.cu). `Some` iff this dtype has a native codebook decode path; `None` routes to the
/// dequantize-to-f32 dense fallback. This is the ONE predicate that selects the SotA decode path --
/// the CUDA twin of `RocmQuantType::dp4a_capable` (the i-quant + ternary subset).
fn iquant_dp4a_suffix(dtype: GgmlDType) -> Option<&'static str> {
    Some(match dtype {
        GgmlDType::IQ2_XXS => "iq2xxs",
        GgmlDType::IQ2_XS => "iq2xs",
        GgmlDType::IQ2_S => "iq2s",
        GgmlDType::IQ3_XXS => "iq3xxs",
        GgmlDType::IQ3_S => "iq3s",
        GgmlDType::IQ1_S => "iq1_s",
        GgmlDType::IQ1_M => "iq1_m",
        GgmlDType::IQ4_XS => "iq4xs",
        // Symmetric ternary {-1,0,1}: 2-bit (TQ2_0, clean) + base-3 (TQ1_0, scalar unpack + dp4a dot).
        GgmlDType::TQ2_0 => "tq2_0",
        GgmlDType::TQ1_0 => "tq1_0",
        // Prism / Bonsai-2 128-elem ternary: 2-bit asymmetric {-1,0,+1,+2} (PQ2_0) + base-3 {-1,0,+1} (PTQ1_0).
        GgmlDType::PQ2_0 => "pq2_0",
        GgmlDType::PTQ1_0 => "ptq1_0",
        GgmlDType::Q1_0 => "q1_0",
        GgmlDType::Q2_0 => "q2_0",
        _ => return None,
    })
}

const IQ_Q8_ROW_PAD: usize = 256;
const IQ_MATVEC_WARPS_PER_BLOCK: u32 = 4;
const IQ_MATVEC_ROWS_PER_WARP: u32 = 4;
const IQ_MATVEC_ROWS_PER_BLOCK: u32 = IQ_MATVEC_WARPS_PER_BLOCK * IQ_MATVEC_ROWS_PER_WARP;
const IQ_BATCH_WARPS_PER_BLOCK: u32 = 8;
const IQ_BATCH_BM: u32 = 16;

fn iq_quantize_storage_to_q8(
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
    b_size: usize,
    ncols: usize,
    dev: &CudaDevice,
) -> Result<(CudaSlice<u8>, CudaSlice<f16>)> {
    if !ncols.is_multiple_of(32) {
        crate::bail!("iq_quantize_q8 ncols {ncols} must be a multiple of 32");
    }
    let (o1, o2) = rhs_l
        .contiguous_offsets()
        .ok_or_else(|| crate::Error::RequiresContiguous { op: "iq-quantize-q8" }.bt())?;
    let ncols_padded = pad(ncols, IQ_Q8_ROW_PAD);
    let total_blocks_padded = b_size * (ncols_padded / 32);
    let xq = unsafe { dev.alloc::<u8>(total_blocks_padded * 32)? };
    let xd = unsafe { dev.alloc::<f16>(total_blocks_padded)? };
    let threads = 256u32;
    let blocks = (total_blocks_padded as u32)
        .div_ceil(threads / WARP_SIZE as u32)
        .clamp(1, 65_535);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };
    match rhs.dtype() {
        crate::DType::F32 => {
            let y = rhs.as_cuda_slice::<f32>()?.slice(o1..o2);
            if y.len() != ncols * b_size {
                crate::bail!("unexpected y size {}, ncols {ncols} bsize {b_size}", y.len());
            }
            let func = dev.get_or_load_func("iq_quantize_q8_f32", &hanzo_kernels::IQUANT_MMVQ)?;
            let mut builder = func.builder();
            barg!(builder, b_size as i32, ncols as i32, total_blocks_padded as i32);
            builder.arg(&y);
            builder.arg(&xq);
            builder.arg(&xd);
            unsafe { builder.launch(cfg) }.w()?;
        }
        crate::DType::BF16 => {
            let y = rhs.as_cuda_slice::<bf16>()?.slice(o1..o2);
            if y.len() != ncols * b_size {
                crate::bail!("unexpected y size {}, ncols {ncols} bsize {b_size}", y.len());
            }
            let func = dev.get_or_load_func("iq_quantize_q8_bf16", &hanzo_kernels::IQUANT_MMVQ)?;
            let mut builder = func.builder();
            barg!(builder, b_size as i32, ncols as i32, total_blocks_padded as i32);
            builder.arg(&y);
            builder.arg(&xq);
            builder.arg(&xd);
            unsafe { builder.launch(cfg) }.w()?;
        }
        crate::DType::F16 => {
            let y = rhs.as_cuda_slice::<f16>()?.slice(o1..o2);
            if y.len() != ncols * b_size {
                crate::bail!("unexpected y size {}, ncols {ncols} bsize {b_size}", y.len());
            }
            let func = dev.get_or_load_func("iq_quantize_q8_f16", &hanzo_kernels::IQUANT_MMVQ)?;
            let mut builder = func.builder();
            barg!(builder, b_size as i32, ncols as i32, total_blocks_padded as i32);
            builder.arg(&y);
            builder.arg(&xq);
            builder.arg(&xd);
            unsafe { builder.launch(cfg) }.w()?;
        }
        dt => crate::bail!("unsupported activation dtype {dt:?} for iq_quantize_q8"),
    }
    Ok((xq, xd))
}

#[allow(clippy::too_many_arguments)]
fn launch_iquant_dp4a_from_q8(
    data: &PaddedCudaSlice,
    xq: &CudaSlice<u8>,
    xd: &CudaSlice<f16>,
    w_dtype: GgmlDType,
    out_dtype: crate::DType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let suffix = iquant_dp4a_suffix(w_dtype)
        .ok_or_else(|| crate::Error::Msg(format!("no i-quant dp4a kernel for {w_dtype:?}")).bt())?;
    let blk_sz = w_dtype.block_size();
    if !ncols.is_multiple_of(blk_sz) {
        crate::bail!("i-quant ncols {ncols} must be a multiple of {blk_sz}");
    }
    if b_size == 0 {
        crate::bail!("bsize must be >= 1, got {b_size}");
    }
    let dt_sfx = match out_dtype {
        crate::DType::F32 => "f32",
        crate::DType::BF16 => "bf16",
        crate::DType::F16 => "f16",
        dt => crate::bail!("unsupported output dtype {dt:?} for i-quant dp4a"),
    };
    let kernel_name = if b_size == 1 {
        format!("qmatvec_dp4a_{suffix}_{dt_sfx}")
    } else {
        format!("qmatmul_dp4a_{suffix}_{dt_sfx}")
    };
    let func = dev.get_or_load_func(&kernel_name, &hanzo_kernels::IQUANT_MMVQ)?;
    let cfg = if b_size == 1 {
        cudarc::driver::LaunchConfig {
            grid_dim: ((nrows as u32).div_ceil(IQ_MATVEC_ROWS_PER_BLOCK), 1, 1),
            block_dim: (IQ_MATVEC_WARPS_PER_BLOCK * WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    } else {
        cudarc::driver::LaunchConfig {
            grid_dim: (
                (nrows as u32).div_ceil(IQ_BATCH_WARPS_PER_BLOCK),
                (b_size as u32).div_ceil(IQ_BATCH_BM),
                1,
            ),
            block_dim: (IQ_BATCH_WARPS_PER_BLOCK * WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        }
    };

    let nrows_i32 = nrows as i32;
    let ncols_i32 = ncols as i32;
    let bsize_i32 = b_size as i32;

    match out_dtype {
        crate::DType::F32 => {
            let dst = unsafe { dev.alloc::<f32>(nrows * b_size)? };
            let mut builder = func.builder();
            builder.arg(&nrows_i32);
            builder.arg(&ncols_i32);
            if b_size != 1 {
                builder.arg(&bsize_i32);
            }
            builder.arg(&data.inner);
            builder.arg(xq);
            builder.arg(xd);
            builder.arg(&dst);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
        }
        crate::DType::BF16 => {
            let dst = unsafe { dev.alloc::<bf16>(nrows * b_size)? };
            let mut builder = func.builder();
            builder.arg(&nrows_i32);
            builder.arg(&ncols_i32);
            if b_size != 1 {
                builder.arg(&bsize_i32);
            }
            builder.arg(&data.inner);
            builder.arg(xq);
            builder.arg(xd);
            builder.arg(&dst);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
        }
        crate::DType::F16 => {
            let dst = unsafe { dev.alloc::<f16>(nrows * b_size)? };
            let mut builder = func.builder();
            builder.arg(&nrows_i32);
            builder.arg(&ncols_i32);
            if b_size != 1 {
                builder.arg(&bsize_i32);
            }
            builder.arg(&data.inner);
            builder.arg(xq);
            builder.arg(xd);
            builder.arg(&dst);
            unsafe { builder.launch(cfg) }.w()?;
            Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
        }
        _ => unreachable!(),
    }
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_vec_iquant_dp4a(
    data: &PaddedCudaSlice,
    rhs: &CudaStorage,
    rhs_l: &crate::Layout,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let blk_sz = dtype.block_size();
    if !ncols.is_multiple_of(blk_sz) {
        crate::bail!("i-quant ncols {ncols} must be a multiple of {blk_sz}");
    }
    if b_size == 0 {
        crate::bail!("bsize must be >= 1, got {b_size}");
    }
    let (xq, xd) = iq_quantize_storage_to_q8(rhs, rhs_l, b_size, ncols, dev)?;
    launch_iquant_dp4a_from_q8(data, &xq, &xd, dtype, rhs.dtype(), ncols, nrows, b_size, dev)
}

#[allow(clippy::too_many_arguments)]
fn mul_mat_via_q8_1(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    x_rows: usize,
    x_cols: usize,
    y_rows: usize,
    y_cols: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let data_elems = data.len / dtype.type_size() * dtype.block_size();
    if data_elems < x_rows * x_cols {
        crate::bail!("unexpected lhs size {}, {x_rows} {x_cols}", data_elems)
    }
    if y.len() != y_rows * y_cols {
        crate::bail!("unexpected y size {}, {y_rows} {y_cols}", y.len())
    }
    if x_cols != y_rows {
        crate::bail!("unexpected x/y size {x_rows} {x_cols} {y_rows} {y_cols}")
    }
    let k = x_cols;
    // Start by quantizing y
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    let y_size_in_bytes =
        k_padded * y_cols * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
    let mut y_q8_1 = dev.alloc_zeros::<u8>(y_size_in_bytes)?;
    quantize_q8_1(y, &mut y_q8_1, k, y_cols, dev)?;

    let (kernel_name, mmq_x, mmq_y) = match dtype {
        GgmlDType::Q4_0 => ("mul_mat_q4_0", 64, 128),
        GgmlDType::Q4_1 => ("mul_mat_q4_1", 64, 128),
        GgmlDType::Q5_0 => ("mul_mat_q5_0", 128, 64),
        GgmlDType::Q5_1 => ("mul_mat_q5_1", 128, 64),
        GgmlDType::Q8_0 => ("mul_mat_q8_0", 128, 64),
        GgmlDType::Q2K => ("mul_mat_q2_K", 64, 128),
        GgmlDType::Q3K => ("mul_mat_q3_K", 128, 128),
        GgmlDType::Q4K => ("mul_mat_q4_K", 64, 128),
        GgmlDType::Q5K => ("mul_mat_q5_K", 64, 128),
        GgmlDType::Q6K => ("mul_mat_q6_K", 64, 64),
        _ => crate::bail!("unsupported dtype for quantized matmul {dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &hanzo_kernels::QUANTIZED)?;
    let dst = dev.alloc_zeros::<f32>(x_rows * y_cols)?;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (
            ceil_div(x_rows, mmq_y) as u32,
            ceil_div(y_cols, mmq_x) as u32,
            1,
        ),
        block_dim: (WARP_SIZE as u32, 4, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(/* vx */ &data.inner);
    builder.arg(/* vy */ &y_q8_1);
    builder.arg(/* dst */ &dst);
    barg!(
        builder,
        /* ncols_x */ x_cols as i32,
        /* nrows_x */ x_rows as i32,
        /* ncols_y */ y_cols as i32,
        /* nrows_y */ k_padded as i32,
        /* nrows_dst */ x_rows as i32
    );
    unsafe { builder.launch(cfg) }.w()?;
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
}

/// The compiled fused indexed-MoE q8_1 kernel name for `dtype`, or None when none exists (those types
/// take the CPU per-expert fallback). This Option is the SINGLE source of truth for "does CUDA have a
/// fused MoE kernel for this type": `QCudaStorage::supports_indexed_moe` (used by both the inner gate
/// here and the `QTensor::indexed_moe_forward` fast-path gate in mod.rs) derives from it, so a gate can
/// never claim support the compiled kernels don't have -- the exact drift that stranded Q4_0/Q4_1/Q5_0/
/// Q5_1 on the CPU while the kernel-name table already referenced them.
fn indexed_moe_kernel_name(dtype: GgmlDType) -> Option<&'static str> {
    Some(match dtype {
        GgmlDType::Q2K => "indexed_moe_forward_q2k_q8_1",
        GgmlDType::Q3K => "indexed_moe_forward_q3k_q8_1",
        GgmlDType::Q4K => "indexed_moe_forward_q4k_q8_1",
        GgmlDType::Q5K => "indexed_moe_forward_q5k_q8_1",
        GgmlDType::Q6K => "indexed_moe_forward_q6k_q8_1",
        GgmlDType::Q8_0 => "indexed_moe_forward_q8_0_q8_1",
        GgmlDType::Q4_0 => "indexed_moe_forward_q4_0_q8_1",
        GgmlDType::Q4_1 => "indexed_moe_forward_q4_1_q8_1",
        GgmlDType::Q5_0 => "indexed_moe_forward_q5_0_q8_1",
        GgmlDType::Q5_1 => "indexed_moe_forward_q5_1_q8_1",
        _ => return None,
    })
}

fn indexed_moe_forward_fused_q8_1_input(
    weight: &CudaView<u8>,
    w_shape: &crate::Shape, //[num_experts, n, k]
    w_dtype: GgmlDType,
    input: &CudaSlice<f32>,
    in_shape: &crate::Shape, //[batch, topk or 1, k]
    ids: &CudaView<u32>,
    idx_shape: &crate::Shape, //[batch, topk] or flat [batch*topk]
    dev: &CudaDevice,
) -> Result<(CudaStorage, crate::Shape)> {
    let (_, n, k) = w_shape.dims3()?;
    let batch = in_shape.dims()[0];
    let input_dim1 = in_shape.dims()[1];

    // The kernel reads ids as a flat [batch*topk] buffer (task_id = token*topk + slot), so derive topk
    // from the element count and accept any ids rank: a router top-k may emit [batch, topk] (2-D) or a
    // flattened [batch*topk] (1-D). batch comes from the input, which is always [batch, topk|1, k].
    let n_slots = idx_shape.elem_count();
    assert!(
        n_slots.is_multiple_of(batch),
        "ids count {n_slots} not a multiple of batch {batch}"
    );
    let topk = n_slots / batch;

    // Prefill (rows>1) -> expert-grouped int8 MMQ GEMM (llama mul_mat_id): group the routed tokens by
    // expert and stage each expert's weight ONCE, amortizing it over all its tokens via the tensor
    // cores. The per-slot matvec below instead re-streams the whole expert weight for EVERY routed token
    // (no matrix cores) -- fine at decode (one token), the 10x deficit at prefill. Decode (batch==1)
    // keeps the per-slot matvec (bandwidth-bound, capture-clean). Unsupported weight dtypes return None
    // and fall through to per-slot.
    if batch > 1 {
        if let Some(res) = super::fast_mmq::indexed_moe_grouped(
            weight, w_shape, w_dtype, input, in_shape, ids, idx_shape, dev,
        )? {
            return Ok(res);
        }
    }

    // Quantize input into q8_1.
    let total_rows = batch * input_dim1;
    let k_padded = pad(k, MATRIX_ROW_PADDING);
    // Get Q8_1 metadata.
    let q8_1_block_size = GgmlDType::Q8_1.block_size();
    let q8_1_type_size = GgmlDType::Q8_1.type_size();

    // Calculate the size of the output buffer in bytes.
    let num_blocks_per_row = k_padded / q8_1_block_size;
    let dst_row_size_bytes = num_blocks_per_row * q8_1_type_size;
    let y_size_in_bytes = total_rows * dst_row_size_bytes;
    let mut input_quant = dev.alloc_zeros::<u8>(y_size_in_bytes)?;

    let input_view = input.slice(0..);
    quantize_q8_1(&input_view, &mut input_quant, k, total_rows, dev)?;

    // output buffer
    let outsize = batch * topk * n;
    let out = dev.alloc_zeros::<f32>(outsize)?;

    let kernel_name = match indexed_moe_kernel_name(w_dtype) {
        Some(name) => name,
        None => crate::bail!("unsupported dtype for indexed_moe_forward {w_dtype:?}"),
    };
    let func = dev.get_or_load_func(kernel_name, &hanzo_kernels::QUANTIZED)?;
    // ONE block per output row: the kernel computes exactly current_output_ptr[blockIdx.x] (its
    // rows_per_cuda_block is 1). A prior "tile 4 rows/block" change shrank this to ceil(n/4) blocks
    // without making the kernel tile, so 3/4 of every expert's output rows were left at zero -- the
    // grid must cover all n rows. (gridDim.y = batch, gridDim.z = topk select the routed slot.)
    let nblocks = n as u32;
    let nwarps = 4u32;
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (nblocks, batch as u32, topk as u32),
        block_dim: (WARP_SIZE as u32, nwarps, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(weight);
    builder.arg(&input_quant);
    builder.arg(ids);
    builder.arg(&out);

    barg!(
        builder,
        n as i32,
        k as i32,
        batch as i32,
        topk as i32,
        k_padded as i32,
        input_dim1 as i32
    );
    unsafe { builder.launch(cfg) }.w()?;

    let mut out_shape = in_shape.dims().to_vec();
    out_shape.pop();
    out_shape.push(n);
    out_shape[1] = topk;
    Ok((
        CudaStorage::wrap_cuda_slice(out, dev.clone()),
        out_shape.into(),
    ))
}

impl QCudaStorage {
    /// True iff a fused on-GPU indexed-MoE q8_1 kernel exists for `dtype` (derives from the ONE
    /// kernel-name table). The `QTensor::indexed_moe_forward` fast-path gate consults this so it routes
    /// to the fused path exactly for the types that have a kernel and to the CPU per-expert fallback for
    /// the rest -- no second hand-maintained type list to drift.
    pub fn supports_indexed_moe(dtype: GgmlDType) -> bool {
        indexed_moe_kernel_name(dtype).is_some()
    }

    pub fn indexed_moe_forward(
        &self,
        self_shape: &crate::Shape, //[num_experts, n, k]
        input: &CudaStorage,       //[batch, topk or 1, k]
        input_l: &crate::Layout,
        ids: &CudaStorage, //[batch, topk]
        ids_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        if Self::supports_indexed_moe(self.dtype()) {
            let input_storage = input.as_cuda_slice::<f32>()?;
            let ids_storage = ids.as_cuda_slice::<u32>()?;
            indexed_moe_forward_fused_q8_1_input(
                &self.data.inner.slice(0..),
                self_shape, //[num_experts, n, k]
                self.dtype(),
                &input_storage,
                input_l.shape(), //[batch, topk or 1, k]
                &ids_storage.slice(0..),
                ids_l.shape(), //[batch, topk]
                &self.device,
            )
        } else {
            crate::bail!(
                "The given quantized dtype {:?} is not supported for indexed_moe_forward!",
                self.dtype()
            );
        }
    }

    /// True for the i-quant codebook types, which have a native dp4a indexed-MoE DECODE kernel
    /// (moe_qmatvec_dp4a_<iq*>) but NO Blue-C fused q8_1 MoE kernel. Distinct from `supports_indexed_moe`
    /// (the Blue-C set) -- the dispatch in mod.rs tries that first, then this, then the generic fallback.
    pub fn supports_iquant_moe(dtype: GgmlDType) -> bool {
        iquant_dp4a_suffix(dtype).is_some()
    }

    /// Fused i-quant indexed-MoE DECODE. The [E,n,k] expert bank stays RESIDENT in VRAM and the router
    /// gather runs on-device (slot s on grid.y, expert = ids[s] read in-kernel) -- the dp4a twin of the
    /// ROCm moe_matvec_quant. This REPLACES the generic per-expert fallback, which DtoH'd the whole
    /// expert bank to host (`self.data()`) and re-uploaded each selected expert EVERY token. `x_flat`
    /// is the already-per-slot-broadcast routed activation [nrows, k] f32 (nrows = batch*topk).
    pub fn moe_iquant_dp4a(
        &self,
        x_flat: &CudaView<f32>,
        ids: &CudaView<u32>,
        nrows: usize,
        n: usize,
        k: usize,
    ) -> Result<CudaStorage> {
        let suffix = iquant_dp4a_suffix(self.dtype).ok_or_else(|| {
            crate::Error::Msg(format!("no i-quant MoE kernel for {:?}", self.dtype)).bt()
        })?;
        let blk_sz = self.dtype.block_size();
        if !k.is_multiple_of(blk_sz) {
            crate::bail!("i-quant MoE ncols {k} must be a multiple of {blk_sz}");
        }
        if x_flat.len() != nrows * k {
            crate::bail!(
                "unexpected x_flat size {}, nrows {nrows} k {k}",
                x_flat.len()
            );
        }
        let dev = self.device();
        let k_padded = pad(k, IQ_Q8_ROW_PAD);
        let total_blocks_padded = nrows * (k_padded / 32);

        let xq = unsafe { dev.alloc::<u8>(total_blocks_padded * 32)? };
        let xd = unsafe { dev.alloc::<f16>(total_blocks_padded)? };
        {
            let func = dev.get_or_load_func("iq_quantize_q8_f32", &hanzo_kernels::IQUANT_MMVQ)?;
            let nwarp = total_blocks_padded as u32;
            let threads = 256u32;
            let blocks = nwarp.div_ceil(threads / WARP_SIZE as u32).clamp(1, 65_535);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            barg!(builder, nrows as i32, k as i32, total_blocks_padded as i32);
            builder.arg(x_flat);
            builder.arg(&xq);
            builder.arg(&xd);
            unsafe { builder.launch(cfg) }.w()?;
        }

        let kernel_name = format!("moe_qmatvec_dp4a_{suffix}_f32");
        let func = dev.get_or_load_func(&kernel_name, &hanzo_kernels::IQUANT_MMVQ)?;
        let out = unsafe { dev.alloc::<f32>(nrows * n)? };
        const ROWS_PER_BLOCK: u32 = 8;
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((n as u32).div_ceil(ROWS_PER_BLOCK), nrows as u32, 1),
            block_dim: (ROWS_PER_BLOCK * WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        barg!(builder, n as i32, k as i32, nrows as i32);
        builder.arg(&self.data.inner);
        builder.arg(ids);
        builder.arg(&xq);
        builder.arg(&xd);
        builder.arg(&out);
        unsafe { builder.launch(cfg) }.w()?;
        Ok(CudaStorage::wrap_cuda_slice(out, dev.clone()))
    }

    /// Expert-grouped int8-WMMA MMQ MoE PREFILL (qmmq) for i-quants -- the prefill twin of
    /// `moe_iquant_dp4a`. Stages each expert's weight ONCE and amortizes it over all its routed tokens
    /// via the tensor cores (llama mul_mat_id), vs the per-slot dp4a re-streaming the weight per token.
    /// Wraps `fast_mmq::indexed_moe_grouped` with the resident [E,n,k] bank; returns None for IQ1_M
    /// (no MMQ kernel) or unsupported shapes -> caller falls back to the per-slot dp4a path.
    pub fn moe_iquant_qmmq(
        &self,
        w_shape: &crate::Shape,
        input: &CudaSlice<f32>,
        in_shape: &crate::Shape,
        ids: &CudaView<u32>,
        idx_shape: &crate::Shape,
    ) -> Result<Option<(CudaStorage, crate::Shape)>> {
        super::fast_mmq::indexed_moe_grouped(
            &self.data.inner.slice(0..),
            w_shape,
            self.dtype,
            input,
            in_shape,
            ids,
            idx_shape,
            self.device(),
        )
    }

    pub fn zeros(device: &CudaDevice, el_count: usize, dtype: GgmlDType) -> Result<Self> {
        let size_in_bytes = ceil_div(el_count, dtype.block_size()) * dtype.type_size();
        let padded_size_in_bytes =
            ceil_div(el_count + MATRIX_ROW_PADDING, dtype.block_size()) * dtype.type_size();
        let inner = device.alloc_zeros::<u8>(padded_size_in_bytes)?;
        Ok(QCudaStorage {
            data: PaddedCudaSlice {
                inner,
                len: size_in_bytes,
            },
            device: device.clone(),
            dtype,
        })
    }

    pub fn dtype(&self) -> GgmlDType {
        self.dtype
    }

    pub fn device(&self) -> &CudaDevice {
        &self.device
    }

    pub fn dequantize(&self, elem_count: usize) -> Result<CudaStorage> {
        fn deq<T: GgmlType>(buffer: &[u8], n: usize, dst: &mut [f32]) {
            let slice = unsafe { std::slice::from_raw_parts(buffer.as_ptr() as *const T, n) };
            let vec = slice.to_vec();
            T::to_float(&vec, dst)
        }

        let fast_kernel = matches!(
            self.dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
                | GgmlDType::Q8K
                | GgmlDType::PQ2_0
                | GgmlDType::PTQ1_0
                | GgmlDType::Q1_0
                | GgmlDType::Q2_0
        );
        if fast_kernel {
            return dequantize_f32(&self.data, self.dtype, elem_count, self.device());
        }
        // Run the dequantization on cpu.

        let buffer = self
            .device
            .clone_dtoh(&self.data.inner.slice(..self.data.len))?;
        let mut out = vec![0.0; elem_count];
        let block_len = elem_count / self.dtype.block_size();
        match self.dtype {
            GgmlDType::F32 => deq::<f32>(&buffer, block_len, &mut out),
            GgmlDType::F16 => deq::<half::f16>(&buffer, block_len, &mut out),
            GgmlDType::BF16 => deq::<half::bf16>(&buffer, block_len, &mut out),
            GgmlDType::I32 => deq::<i32>(&buffer, block_len, &mut out),
            GgmlDType::Q4_0 => deq::<crate::quantized::BlockQ4_0>(&buffer, block_len, &mut out),
            GgmlDType::Q4_1 => deq::<crate::quantized::BlockQ4_1>(&buffer, block_len, &mut out),
            GgmlDType::Q5_0 => deq::<crate::quantized::BlockQ5_0>(&buffer, block_len, &mut out),
            GgmlDType::Q5_1 => deq::<crate::quantized::BlockQ5_1>(&buffer, block_len, &mut out),
            GgmlDType::Q8_0 => deq::<crate::quantized::BlockQ8_0>(&buffer, block_len, &mut out),
            GgmlDType::Q8_1 => deq::<crate::quantized::BlockQ8_1>(&buffer, block_len, &mut out),
            GgmlDType::Q2K => deq::<crate::quantized::BlockQ2K>(&buffer, block_len, &mut out),
            GgmlDType::Q3K => deq::<crate::quantized::BlockQ3K>(&buffer, block_len, &mut out),
            GgmlDType::Q4K => deq::<crate::quantized::BlockQ4K>(&buffer, block_len, &mut out),
            GgmlDType::Q5K => deq::<crate::quantized::BlockQ5K>(&buffer, block_len, &mut out),
            GgmlDType::Q6K => deq::<crate::quantized::BlockQ6K>(&buffer, block_len, &mut out),
            GgmlDType::Q8K => deq::<crate::quantized::BlockQ8K>(&buffer, block_len, &mut out),
            GgmlDType::IQ4_NL => deq::<crate::quantized::BlockIQ4nl>(&buffer, block_len, &mut out),
            GgmlDType::IQ4_XS => deq::<crate::quantized::BlockIQ4xs>(&buffer, block_len, &mut out),
            GgmlDType::MXFP4 => deq::<crate::quantized::BlockMXFP4>(&buffer, block_len, &mut out),
            GgmlDType::IQ2_XXS => {
                deq::<crate::quantized::iq_quants::BlockIQ2xxs>(&buffer, block_len, &mut out)
            }
            GgmlDType::IQ2_XS => {
                deq::<crate::quantized::iq_quants::BlockIQ2xs>(&buffer, block_len, &mut out)
            }
            GgmlDType::IQ3_XXS => {
                deq::<crate::quantized::iq_quants::BlockIQ3xxs>(&buffer, block_len, &mut out)
            }
            GgmlDType::IQ1_S => {
                deq::<crate::quantized::iq_quants::BlockIQ1s>(&buffer, block_len, &mut out)
            }
            GgmlDType::IQ3_S => {
                deq::<crate::quantized::iq_quants::BlockIQ3s>(&buffer, block_len, &mut out)
            }
            GgmlDType::IQ2_S => {
                deq::<crate::quantized::iq_quants::BlockIQ2s>(&buffer, block_len, &mut out)
            }
            GgmlDType::IQ1_M => {
                deq::<crate::quantized::iq_quants::BlockIQ1m>(&buffer, block_len, &mut out)
            }
            GgmlDType::TQ1_0 => {
                deq::<crate::quantized::iq_quants::BlockTQ1_0>(&buffer, block_len, &mut out)
            }
            GgmlDType::TQ2_0 => {
                deq::<crate::quantized::iq_quants::BlockTQ2_0>(&buffer, block_len, &mut out)
            }
            GgmlDType::NVFP4 => {
                deq::<crate::quantized::iq_quants::BlockNVFP4>(&buffer, block_len, &mut out)
            }
            GgmlDType::Q1_0 => {
                deq::<crate::quantized::iq_quants::BlockQ1_0>(&buffer, block_len, &mut out)
            }
            GgmlDType::ROCMFP4 => {
                deq::<crate::quantized::iq_quants::BlockROCMFP4>(&buffer, block_len, &mut out)
            }
            GgmlDType::ROCMFP4_FAST => {
                deq::<crate::quantized::iq_quants::BlockROCMFP4Fast>(&buffer, block_len, &mut out)
            }
            GgmlDType::PQ2_0 => {
                deq::<crate::quantized::iq_quants::BlockPQ2_0>(&buffer, block_len, &mut out)
            }
            GgmlDType::PTQ1_0 => {
                deq::<crate::quantized::iq_quants::BlockPTQ1_0>(&buffer, block_len, &mut out)
            }
            GgmlDType::Q2_0 => {
                deq::<crate::quantized::iq_quants::BlockQ2_0>(&buffer, block_len, &mut out)
            }
        }

        self.device
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    pub fn dequantize_f16(&self, elem_count: usize) -> Result<CudaStorage> {
        dequantize_f16(&self.data, self.dtype, elem_count, self.device())
    }

    pub fn quantize(&mut self, src: &CudaStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize(&src)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&data[..], &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix(
        &mut self,
        src: &CudaStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src = match &src.slice {
            crate::cuda_backend::CudaStorageSlice::F32(data) => self.device.clone_dtoh(data)?,
            _ => crate::bail!("only f32 can be quantized"),
        };
        let src_len = src.len();
        let src = crate::Storage::Cpu(crate::CpuStorage::F32(src));
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;
        qcpu_storage.quantize_imatrix(&src, imatrix_weights, n_per_row)?;
        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&data[..], &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_imatrix_onto(
        &mut self,
        src: &crate::CpuStorage,
        imatrix_weights: &[f32],
        n_per_row: usize,
    ) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float_imatrix(src.as_slice::<f32>()?, imatrix_weights, n_per_row);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&data[..], &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn quantize_onto(&mut self, src: &crate::CpuStorage) -> Result<()> {
        // Run the quantization on cpu.
        let src_len = src.as_slice::<f32>()?.len();
        let mut qcpu_storage = crate::Device::Cpu.qzeros(src_len, self.dtype)?;

        if let QStorage::Cpu(storage) = &mut qcpu_storage {
            storage.from_float(src.as_slice::<f32>()?);
        } else {
            unreachable!()
        }

        let data = qcpu_storage.data()?;
        let padded_len =
            data.len() + MATRIX_ROW_PADDING * self.dtype.type_size() / self.dtype.block_size();
        let mut inner = unsafe { self.device.alloc::<u8>(padded_len)? };
        self.device
            .memcpy_htod(&data[..], &mut inner.slice_mut(..data.len()))?;
        self.data = PaddedCudaSlice {
            inner,
            len: data.len(),
        };
        Ok(())
    }

    pub fn storage_size_in_bytes(&self) -> usize {
        self.data.len
    }

    pub fn embedding(
        &self,
        rows: usize,
        hidden: usize,
        ids: &CudaStorage,
        ids_l: &crate::Layout,
    ) -> Result<CudaStorage> {
        if !ids_l.is_contiguous() {
            crate::bail!("quantized embedding requires contiguous ids")
        }
        if !hidden.is_multiple_of(self.dtype.block_size()) {
            crate::bail!(
                "quantized embedding hidden size {hidden} is not divisible by block size {}",
                self.dtype.block_size()
            )
        }
        let expected_size = rows * hidden * self.dtype.type_size() / self.dtype.block_size();
        if self.storage_size_in_bytes() != expected_size {
            crate::bail!(
                "quantized tensor has {} bytes, expected {expected_size}",
                self.storage_size_in_bytes()
            )
        }
        let ids = ids.as_cuda_slice::<u32>()?;
        let ids = match ids_l.contiguous_offsets() {
            Some((o1, o2)) => ids.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous {
                op: "quantized-embedding",
            }
            .bt())?,
        };
        match self.dtype {
            GgmlDType::F32
            | GgmlDType::F16
            | GgmlDType::BF16
            | GgmlDType::Q4_0
            | GgmlDType::Q4_1
            | GgmlDType::Q5_0
            | GgmlDType::Q5_1
            | GgmlDType::Q8_0
            | GgmlDType::Q2K
            | GgmlDType::Q3K
            | GgmlDType::Q4K
            | GgmlDType::Q5K
            | GgmlDType::Q6K
            | GgmlDType::PQ2_0
            | GgmlDType::PTQ1_0
            | GgmlDType::Q1_0
            | GgmlDType::Q2_0 => get_rows(&self.data, self.dtype, hidden, &ids, self.device()),
            // dtypes without a get_rows kernel (i-quant, ternary, NVFP4): gather each
            // requested row through the CPU to_float reference and upload the f32 result.
            _ => self.embedding_dequant_rows(hidden, &ids),
        }
    }

    fn embedding_dequant_rows(&self, hidden: usize, ids: &CudaView<u32>) -> Result<CudaStorage> {
        let ids: Vec<u32> = self.device().clone_dtoh(ids)?;
        let row_bytes = hidden * self.dtype.type_size() / self.dtype.block_size();
        let table = self.data.inner.slice(..self.data.len);
        let mut out = Vec::with_capacity(ids.len() * hidden);
        for id in ids {
            let lo = id as usize * row_bytes;
            let row = table.slice(lo..lo + row_bytes);
            let bytes: Vec<u8> = self.device().clone_dtoh(&row)?;
            let q = self.dtype.from_data(std::borrow::Cow::Borrowed(&bytes));
            let crate::CpuStorage::F32(v) = q.dequantize(hidden)? else {
                crate::bail!("embedding dequant did not yield f32")
            };
            out.extend_from_slice(&v);
        }
        self.device()
            .storage_from_cpu_storage(&crate::CpuStorage::F32(out))
    }

    /// Native-GGUF DSL dispatch for `PQ2_0` and `PTQ1_0` on CUDA (`HANZO_PQ2_IMPL=dsl` /
    /// `HANZO_PTQ1_IMPL=dsl`; `legacy` remains the default). Handles both `m == 1` decode matvec
    /// and `m > 1` prefill GEMM for contiguous `F32` and `BF16` activations (including nonzero
    /// `layout.start_offset()`).
    fn try_dispatch_ternary_dsl(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<Option<(CudaStorage, crate::Shape)>> {
        use super::dsl::{self, QkImpl};
        use crate::cuda_backend::CudaStorageSlice;
        use cudarc::driver::{DevicePtr, DevicePtrMut};

        if !matches!(self.dtype, GgmlDType::PQ2_0 | GgmlDType::PTQ1_0) {
            return Ok(None);
        }
        if !layout.is_contiguous() || self_shape.rank() != 2 {
            return Ok(None);
        }
        let src1_bf16 = match &storage.slice {
            CudaStorageSlice::F32(_) => false,
            CudaStorageSlice::BF16(_) => true,
            _ => return Ok(None),
        };
        let (n, k) = self_shape.dims2()?;
        let src_dims = layout.shape().dims();
        if src_dims.len() < 2 || src_dims[src_dims.len() - 1] != k {
            return Ok(None);
        }
        let m_total: usize = src_dims[..src_dims.len() - 1].iter().product();
        if m_total == 0 {
            return Ok(None);
        }
        let is_gemm = m_total > 1;
        let selected = match self.dtype {
            GgmlDType::PTQ1_0 => {
                dsl::select_impl(dsl::QkBackend::Cuda, dsl::QkOp::Ptq1 { m: m_total, n, k })
            }
            _ => dsl::select_impl(dsl::QkBackend::Cuda, dsl::QkOp::Pq2 { m: m_total, n, k }),
        };
        if selected != QkImpl::Dsl {
            let legacy_name = match (self.dtype, is_gemm) {
                (GgmlDType::PQ2_0, false) => "cuda_legacy_pq2_0_dp4a",
                (GgmlDType::PQ2_0, true) => "cuda_legacy_pq2_0_prefill",
                (GgmlDType::PTQ1_0, false) => "cuda_legacy_ptq1_0_dp4a",
                _ => "cuda_legacy_ptq1_0_prefill",
            };
            dsl::note_dispatch(QkImpl::Legacy, legacy_name, n, k);
            return Ok(None);
        }
        let eligible = match self.dtype {
            GgmlDType::PTQ1_0 => dsl::ptq1_dsl_eligible(n, k, self.data.len),
            _ => dsl::pq2_dsl_eligible(n, k, self.data.len),
        };
        let nwords = match eligible {
            Ok(nw) => nw,
            Err(_) => return Ok(None),
        };

        let (artifact, meta_words, grid, block) = if !is_gemm {
            let (a, nsg, nr0) = match self.dtype {
                GgmlDType::PTQ1_0 => dsl::ptq1_matvec_artifact(n, src1_bf16),
                _ => dsl::pq2_matvec_artifact(n, src1_bf16),
            };
            (
                a,
                [k as u32, n as u32, nwords as u32],
                (n.div_ceil(nsg * nr0) as u32, 1, 1),
                (32, nsg as u32, 1),
            )
        } else {
            let (a, bm) = match self.dtype {
                GgmlDType::PTQ1_0 => dsl::ptq1_matmul_artifact(m_total, n, src1_bf16),
                _ => dsl::pq2_matmul_artifact(m_total, n, src1_bf16),
            };
            (
                a,
                [m_total as u32, n as u32, k as u32],
                (m_total.div_ceil(bm) as u32, n.div_ceil(64) as u32, 1),
                (128, 1, 1),
            )
        };

        let mut out_dims = src_dims.to_vec();
        *out_dims.last_mut().unwrap() = n;
        let out_shape = crate::Shape::from(out_dims);
        let out_elems = out_shape.elem_count();

        let stream = self.device.cuda_stream();
        let meta = self.device.dsl_const_u32(&meta_words)?;
        let (w_ptr, _gw) = self.data.inner.device_ptr(&stream);
        let (m_ptr, _gm) = meta.device_ptr(&stream);

        let out_slice = if src1_bf16 {
            let x_slice = storage.as_cuda_slice::<half::bf16>()?;
            let (x_base, _gx) = x_slice.device_ptr(&stream);
            let x_ptr = x_base + (layout.start_offset() * std::mem::size_of::<half::bf16>()) as u64;
            let mut out = unsafe { self.device.alloc::<half::bf16>(out_elems)? };
            {
                let (o_ptr, _go) = out.device_ptr_mut(&stream);
                unsafe {
                    self.device.dispatch_dsl_4(
                        artifact,
                        [w_ptr, x_ptr, o_ptr, m_ptr],
                        grid,
                        block,
                    )?;
                }
            }
            CudaStorageSlice::BF16(out)
        } else {
            let x_slice = storage.as_cuda_slice::<f32>()?;
            let (x_base, _gx) = x_slice.device_ptr(&stream);
            let x_ptr = x_base + (layout.start_offset() * std::mem::size_of::<f32>()) as u64;
            let mut out = unsafe { self.device.alloc::<f32>(out_elems)? };
            {
                let (o_ptr, _go) = out.device_ptr_mut(&stream);
                unsafe {
                    self.device.dispatch_dsl_4(
                        artifact,
                        [w_ptr, x_ptr, o_ptr, m_ptr],
                        grid,
                        block,
                    )?;
                }
            }
            CudaStorageSlice::F32(out)
        };

        dsl::note_dispatch(QkImpl::Dsl, artifact, n, k);
        Ok(Some((
            CudaStorage {
                slice: out_slice,
                device: self.device.clone(),
            },
            out_shape,
        )))
    }

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        if let Some(out) = self.try_dispatch_ternary_dsl(self_shape, storage, layout)? {
            return Ok(out);
        }
        // Fallback
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            let cc = crate::quantized::fast_mmq::compute_cap(&self.device);
            mmq_max_vec_bm(self.dtype, cc)
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        // Modern llama mmq/mmvq path (int8 MMA matrix-core + stream-K). Opt-in via
        // CUDA_FAST_MMQ=1; try_fwd returns Ok(None) for unsupported dtype/shape, falling
        // through to the legacy on-GPU q8_1 kernels below. Skipped under FORCE_DMMV.
        if fast_mmq_enabled() && !FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            if use_vec_kernel {
                if let Some(out) = super::fast_mmvq::try_fwd(self, self_shape, storage, layout)? {
                    return Ok(out);
                }
            } else if let Some(out) = super::fast_mmq::try_fwd(self, self_shape, storage, layout)? {
                return Ok(out);
            }
        }
        // i-quant / ternary / NVFP4 weights have no native on-GPU q8_1 matvec/mmq kernel (the kernel-name
        // tables in mul_mat_vec_via_q8_1 / mul_mat_via_q8_1 / dequantize_mul_mat_vec cover only the 10
        // q8_1 types). Dequantize the weight to f32 once and run a dense matmul -- correct for any shape
        // (decode m=1 and prefill m>1), exact w.r.t. the CPU reference, and the only way these types
        // matmul on CUDA until a native mmvq kernel lands. ONE predicate, gated here, one fallback path.
        if !Self::has_native_q8_1_matmul(self.dtype) {
            // SotA native i-quant DECODE: codebook-decode the weight to int8 grid magnitudes and dp4a
            // against the q8_1-quantized activation directly (no dequant->f32 round-trip) -- the CUDA
            // twin of the ROCm `qdp4a<IQ*>` 2.27x lever. Only the decode/vec shape (small b*m) takes
            // it; prefill (m>1) keeps the dequant->dense matmul (i-quant qmmq prefill is a separate
            // lever). ONE predicate (`iquant_dp4a_suffix`), gated here next to the fallback.
            if iquant_dp4a_suffix(self.dtype).is_some() {
                if use_vec_kernel {
                    // DECODE: native dp4a mmvq.
                    if cuda_profile_enabled() {
                        crate::backend::BackendDevice::synchronize(&self.device)?;
                        let t0 = std::time::Instant::now();
                        let res = self.mul_mat_vec_iquant(self_shape, storage, layout)?;
                        crate::backend::BackendDevice::synchronize(&self.device)?;
                        PROF_DP4A_DECODE_NS.fetch_add(
                            t0.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        PROF_DP4A_DECODE_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        return Ok(res);
                    }
                    return self.mul_mat_vec_iquant(self_shape, storage, layout);
                }
                // PREFILL: native int8-WMMA MMQ GEMM (qmmq) for the 7 i-quants with an MMQ kernel
                // (IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS, IQ1_S).
                if let Some(out) = super::fast_mmq::try_fwd(self, self_shape, storage, layout)? {
                    return Ok(out);
                }
                // Types without a WMMA MMQ kernel (PQ2_0, PTQ1_0, TQ1_0, TQ2_0, IQ1_M) use the
                // single-launch batched qmatmul_dp4a kernel for speculative verification / small-batch
                // prefill (b*m <= 64) and for huge vocabulary matrices (>= 64M weights) to avoid
                // multi-GB FP32 weight dequantization.
                let bm = match layout.shape().dims() {
                    [b, m, _k] => b * m,
                    [b, _k] => *b,
                    _ => usize::MAX,
                };
                if !FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed)
                    && (bm <= 64 || self_shape.elem_count() >= 64 * 1024 * 1024)
                {
                    if cuda_profile_enabled() {
                        crate::backend::BackendDevice::synchronize(&self.device)?;
                        let t0 = std::time::Instant::now();
                        let res = self.mul_mat_vec_iquant(self_shape, storage, layout)?;
                        crate::backend::BackendDevice::synchronize(&self.device)?;
                        let ns = t0.elapsed().as_nanos() as u64;
                        if bm <= 64 && self_shape.elem_count() < 64 * 1024 * 1024 {
                            PROF_DP4A_VERIFY_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
                            PROF_DP4A_VERIFY_CALLS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        } else {
                            PROF_DP4A_GE64M_NS.fetch_add(ns, std::sync::atomic::Ordering::Relaxed);
                            PROF_DP4A_GE64M_CALLS
                                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        }
                        return Ok(res);
                    }
                    return self.mul_mat_vec_iquant(self_shape, storage, layout);
                }
            }
            return self.dequantize_matmul_dense(self_shape, storage, layout);
        }
        if use_vec_kernel {
            self.dequantize_matmul_vec(self_shape, storage, layout)
        } else {
            self.dequantize_matmul(self_shape, storage, layout)
        }
    }

    pub fn data(&self) -> Result<Vec<u8>> {
        let mut out = vec![0u8; self.data.len];
        self.device
            .memcpy_dtoh(&self.data.inner.slice(..self.data.len), &mut out)?;
        Ok(out)
    }

    pub fn device_ptr(&self) -> Result<*const u8> {
        Ok(self.data.inner.device_ptr(self.data.inner.stream()).0 as *const u8)
    }

    pub fn device_ptr_with_guard<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> Result<(*const u8, SyncOnDrop<'a>)> {
        let (ptr, guard) = self.data.inner.device_ptr(stream);
        Ok((ptr as *const u8, guard))
    }
}

impl QCudaStorage {
    fn dequantize_matmul_vec(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let rhs = rhs.as_cuda_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "dmmv" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in dmmv {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }

        let out = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            dequantize_mul_mat_vec(&self.data, &rhs, self.dtype, ncols, nrows, self.device())?
        } else {
            mul_mat_vec_via_q8_1(
                &self.data,
                &rhs,
                self.dtype,
                ncols,
                nrows,
                b_size,
                self.device(),
            )?
        };
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    // Native i-quant mmvq DECODE (the SotA path for the IQ codebook quants). Mirrors
    // `dequantize_matmul_vec`'s shape handling but routes to `mul_mat_vec_iquant_dp4a` (codebook-decode
    // + dp4a vs the q8_1 activation) instead of a dequant round-trip. Reached from `fwd` only for the
    // decode/vec shape of an i-quant weight.
    fn mul_mat_vec_iquant(
        &self,
        self_shape: &crate::Shape,
        rhs: &CudaStorage,
        rhs_l: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in iq-mmvq {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }
        let out = mul_mat_vec_iquant_dp4a(
            &self.data,
            rhs,
            rhs_l,
            self.dtype,
            ncols,
            nrows,
            b_size,
            self.device(),
        )?;
        let mut out_shape = rhs_l.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }

    // dtypes with a native on-GPU q8_1 matmul kernel (the mmvq decode, mmq prefill, and dmmv kernels in
    // mul_mat_vec_via_q8_1 / mul_mat_via_q8_1 / dequantize_mul_mat_vec -- the same 10-type spread in all
    // three). Everything else (i-quant codebooks, ternary TQ*, NVFP4, Q1_0) has no such kernel and is
    // matmul'd by dequantizing the weight to f32 then running a dense matmul (`dequantize_matmul_dense`).
    fn has_native_q8_1_matmul(dtype: GgmlDType) -> bool {
        matches!(
            dtype,
            GgmlDType::Q4_0
                | GgmlDType::Q4_1
                | GgmlDType::Q5_0
                | GgmlDType::Q5_1
                | GgmlDType::Q8_0
                | GgmlDType::Q2K
                | GgmlDType::Q3K
                | GgmlDType::Q4K
                | GgmlDType::Q5K
                | GgmlDType::Q6K
        )
    }

    // Dequantize the quantized weight to a dense f32 tensor, then `input @ weight^T` as a regular matmul.
    // Backs both the FORCE_DMMV path and `fwd`'s fallback for dtypes without a native q8_1 kernel. Correct
    // for any input rank/shape that reaches `fwd` (2D [m,k] or 3D [b,m,k]); `dequantize` decodes every
    // GgmlDType via the CPU `to_float` reference, so the result is exact w.r.t. that reference.
    fn dequantize_matmul_dense(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        use crate::backend::BackendStorage;
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        let out_shape_s: crate::Shape = out_shape.into();

        if matches!(
            self.dtype,
            GgmlDType::PQ2_0 | GgmlDType::PTQ1_0 | GgmlDType::Q1_0 | GgmlDType::Q2_0
        )
            && std::env::var_os("HANZO_PQ2_FP32_PREFILL").is_none()
        {
            let prof = cuda_profile_enabled();
            if prof {
                crate::backend::BackendDevice::synchronize(&self.device)?;
            }
            let t_dq = prof.then(std::time::Instant::now);
            let data_bf16 = dequantize_bf16(&self.data, self.dtype, n * k, &self.device)?;
            if let Some(t0) = t_dq {
                crate::backend::BackendDevice::synchronize(&self.device)?;
                PROF_BF16_DEQUANT_NS.fetch_add(
                    t0.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                PROF_BF16_DEQUANT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            let t_gemm = prof.then(std::time::Instant::now);
            let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
            let out_dtype = storage.dtype();
            let (act_bf16, act_l) = if out_dtype == crate::DType::BF16 {
                (None, layout.clone())
            } else {
                let converted = storage.to_dtype(layout, crate::DType::BF16)?;
                (Some(converted), crate::Layout::contiguous(layout.shape()))
            };
            let act_ref = act_bf16.as_ref().unwrap_or(storage);
            let out_bf16 = act_ref.matmul(&data_bf16, (b, m, n, k), &act_l, &rhs_l)?;
            let out = if out_dtype == crate::DType::BF16 {
                out_bf16
            } else {
                let out_l = crate::Layout::contiguous(&out_shape_s);
                out_bf16.to_dtype(&out_l, out_dtype)?
            };
            if let Some(t0) = t_gemm {
                crate::backend::BackendDevice::synchronize(&self.device)?;
                PROF_BF16_GEMM_NS.fetch_add(
                    t0.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            return Ok((out, out_shape_s));
        }

        let data_f32 = self.dequantize(n * k)?;
        let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
        let out = storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?;
        Ok((out, out_shape_s))
    }

    fn dequantize_matmul(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            return self.dequantize_matmul_dense(self_shape, storage, layout);
        }
        let (n, k) = self_shape.dims2()?;
        let (b, m, k2) = match layout.shape().dims() {
            &[b, m, k2] => (b, m, k2),
            &[m, k2] => (1, m, k2),
            s => crate::bail!("unexpected shape for input {s:?}"),
        };
        if k2 != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", layout.shape())
        }
        let storage = storage.as_cuda_slice::<f32>()?;
        let storage = match layout.contiguous_offsets() {
            Some((o1, o2)) => storage.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous {
                op: "quantized-matmul",
            }
            .bt())?,
        };
        let out = mul_mat_via_q8_1(
            &self.data,
            &storage,
            self.dtype,
            /* x_rows */ n,
            /* x_cols */ k,
            /* y_rows */ k,
            /* y_cols */ b * m,
            self.device(),
        )?;
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
    }
}

pub fn load_quantized<T: super::GgmlType + Send + Sync + 'static>(
    device: &CudaDevice,
    data: &[T],
) -> Result<super::QStorage> {
    let data = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, core::mem::size_of_val(data))
    };
    let dtype = T::DTYPE;
    let padded_len = data.len() + MATRIX_ROW_PADDING * dtype.type_size() / dtype.block_size();
    let mut inner = device.alloc_zeros::<u8>(padded_len)?;
    device.memcpy_htod(data, &mut inner.slice_mut(..data.len()))?;
    Ok(QStorage::Cuda(QCudaStorage {
        data: PaddedCudaSlice {
            inner,
            len: data.len(),
        },
        device: device.clone(),
        dtype,
    }))
}

#[derive(Clone, Debug)]
enum IqQ8Inner {
    Cuda {
        xq: CudaSlice<u8>,
        xd: CudaSlice<f16>,
        device: CudaDevice,
    },
    HostFallback(crate::Tensor),
}

#[derive(Clone, Debug)]
pub struct IqQ8Activation {
    inner: IqQ8Inner,
    shape: crate::Shape,
    dtype: crate::DType,
    b_size: usize,
    ncols: usize,
}

impl IqQ8Activation {
    pub fn quantize(xs: &crate::Tensor) -> Result<Self> {
        let shape = xs.shape().clone();
        if shape.rank() < 2 {
            crate::bail!("IqQ8Activation expects rank >= 2, got {:?}", shape);
        }
        let ncols = *shape.dims().last().unwrap();
        if !ncols.is_multiple_of(32) {
            crate::bail!("IqQ8Activation ncols {ncols} must be a multiple of 32");
        }
        let b_size = shape.elem_count() / ncols;
        if b_size == 0 {
            crate::bail!("IqQ8Activation requires non-empty batch");
        }
        let dtype = xs.dtype();
        if !matches!(dtype, crate::DType::F32 | crate::DType::BF16 | crate::DType::F16) {
            crate::bail!("IqQ8Activation unsupported dtype {dtype:?}");
        }
        if !xs.device().is_cuda() {
            return Ok(Self {
                inner: IqQ8Inner::HostFallback(xs.clone()),
                shape,
                dtype,
                b_size,
                ncols,
            });
        }
        let xs_contig = xs.contiguous()?;
        let (storage, layout) = xs_contig.storage_and_layout();
        let crate::Storage::Cuda(cuda_s) = &*storage else {
            unreachable!()
        };
        let (xq, xd) = iq_quantize_storage_to_q8(cuda_s, layout, b_size, ncols, &cuda_s.device)?;
        let device = cuda_s.device.clone();
        drop(storage);
        Ok(Self {
            inner: IqQ8Inner::Cuda { xq, xd, device },
            shape,
            dtype,
            b_size,
            ncols,
        })
    }

    pub fn shape(&self) -> &crate::Shape {
        &self.shape
    }

    pub fn dtype(&self) -> crate::DType {
        self.dtype
    }

    pub fn b_size(&self) -> usize {
        self.b_size
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    pub fn with_dtype(&self, dtype: crate::DType) -> Result<Self> {
        if !matches!(dtype, crate::DType::F32 | crate::DType::BF16 | crate::DType::F16) {
            crate::bail!("IqQ8Activation unsupported output dtype {dtype:?}");
        }
        let inner = match &self.inner {
            IqQ8Inner::Cuda { xq, xd, device } => IqQ8Inner::Cuda {
                xq: xq.clone(),
                xd: xd.clone(),
                device: device.clone(),
            },
            IqQ8Inner::HostFallback(t) => IqQ8Inner::HostFallback(t.to_dtype(dtype)?),
        };
        Ok(Self {
            inner,
            shape: self.shape.clone(),
            dtype,
            b_size: self.b_size,
            ncols: self.ncols,
        })
    }
}

impl QCudaStorage {
    pub fn fwd_shared_q8(
        &self,
        self_shape: &crate::Shape,
        q8: &IqQ8Activation,
    ) -> Result<(CudaStorage, crate::Shape)> {
        self.fwd_shared_q8_dtype(self_shape, q8, q8.dtype)
    }

    pub fn fwd_shared_q8_dtype(
        &self,
        self_shape: &crate::Shape,
        q8: &IqQ8Activation,
        out_dtype: crate::DType,
    ) -> Result<(CudaStorage, crate::Shape)> {
        let (nrows, ncols) = self_shape.dims2()?;
        if ncols != q8.ncols {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", q8.shape);
        }
        let IqQ8Inner::Cuda { xq, xd, .. } = &q8.inner else {
            crate::bail!("QCudaStorage::fwd_shared_q8 requires a CUDA IqQ8Activation");
        };
        let out = launch_iquant_dp4a_from_q8(
            &self.data,
            xq,
            xd,
            self.dtype,
            out_dtype,
            ncols,
            nrows,
            q8.b_size,
            self.device(),
        )?;
        let mut out_shape = q8.shape.dims().to_vec();
        out_shape.pop();
        out_shape.push(nrows);
        Ok((out, out_shape.into()))
    }
}

fn fwht_launch_params(block_size: usize) -> (u32, u32) {
    if block_size < 64 {
        return (block_size as u32, 0);
    }
    let nt = (block_size / 2).min(1024);
    let e = block_size / nt;
    let nwarps = nt / 32;
    let smem_bytes = if nwarps > 1 {
        (nwarps * (32 * e + 1) * std::mem::size_of::<f32>()) as u32
    } else {
        0
    };
    (nt as u32, smem_bytes)
}

fn fwht_cpu_inplace(buf: &mut [f32], signs: Option<&[f32]>, width: usize, block_size: usize, is_inverse: bool) {
    let scale = 1.0f32 / (block_size as f32).sqrt();
    if !is_inverse {
        if let Some(sv) = signs {
            for row in buf.chunks_exact_mut(width) {
                for (v, &s) in row.iter_mut().zip(sv.iter()) {
                    *v *= s;
                }
            }
        }
    }
    for blk in buf.chunks_exact_mut(block_size) {
        let mut h = 1usize;
        while h < block_size {
            let mut base = 0usize;
            while base < block_size {
                for j in 0..h {
                    let a = blk[base + j];
                    let b = blk[base + j + h];
                    blk[base + j] = a + b;
                    blk[base + j + h] = a - b;
                }
                base += 2 * h;
            }
            h <<= 1;
        }
        for v in blk.iter_mut() {
            *v *= scale;
        }
    }
    if is_inverse {
        if let Some(sv) = signs {
            for row in buf.chunks_exact_mut(width) {
                for (v, &s) in row.iter_mut().zip(sv.iter()) {
                    *v *= s;
                }
            }
        }
    }
}

fn hadamard_fwht_cpu_fallback(
    x: &crate::Tensor,
    signs: Option<&crate::Tensor>,
    width: usize,
    block_size: usize,
    is_inverse: bool,
) -> Result<crate::Tensor> {
    let orig_dtype = x.dtype();
    let orig_dev = x.device().clone();
    let mut vals = x
        .to_device(&crate::Device::Cpu)?
        .to_dtype(crate::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let signs_vec = match signs {
        Some(s) => Some(
            s.to_device(&crate::Device::Cpu)?
                .to_dtype(crate::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        ),
        None => None,
    };
    fwht_cpu_inplace(&mut vals, signs_vec.as_deref(), width, block_size, is_inverse);
    crate::Tensor::from_vec(vals, x.shape(), &crate::Device::Cpu)?
        .to_dtype(orig_dtype)?
        .to_device(&orig_dev)
}

fn rms_norm_hadamard_fwht_cpu_fallback(
    x: &crate::Tensor,
    residual: Option<&crate::Tensor>,
    weight: &crate::Tensor,
    eps: f32,
    signs: Option<&crate::Tensor>,
    width: usize,
    block_size: usize,
) -> Result<(Option<crate::Tensor>, crate::Tensor)> {
    let orig_dtype = x.dtype();
    let orig_dev = x.device().clone();
    let mut vals = x
        .to_device(&crate::Device::Cpu)?
        .to_dtype(crate::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let res_out = if let Some(res) = residual {
        let r_vals = res
            .to_device(&crate::Device::Cpu)?
            .to_dtype(crate::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        match orig_dtype {
            crate::DType::BF16 => {
                let mut h_bf16 = Vec::with_capacity(vals.len());
                for (v, &rv) in vals.iter_mut().zip(r_vals.iter()) {
                    let q = bf16::from_f32(*v + rv);
                    *v = q.to_f32();
                    h_bf16.push(q);
                }
                Some(crate::Tensor::from_vec(h_bf16, x.shape(), &crate::Device::Cpu)?.to_device(&orig_dev)?)
            }
            crate::DType::F16 => {
                let mut h_f16 = Vec::with_capacity(vals.len());
                for (v, &rv) in vals.iter_mut().zip(r_vals.iter()) {
                    let q = f16::from_f32(*v + rv);
                    *v = q.to_f32();
                    h_f16.push(q);
                }
                Some(crate::Tensor::from_vec(h_f16, x.shape(), &crate::Device::Cpu)?.to_device(&orig_dev)?)
            }
            _ => {
                for (v, &rv) in vals.iter_mut().zip(r_vals.iter()) {
                    *v += rv;
                }
                Some(
                    crate::Tensor::from_vec(vals.clone(), x.shape(), &crate::Device::Cpu)?
                        .to_dtype(orig_dtype)?
                        .to_device(&orig_dev)?,
                )
            }
        }
    } else {
        None
    };
    let w_vec = weight
        .to_device(&crate::Device::Cpu)?
        .to_dtype(crate::DType::F32)?
        .flatten_all()?
        .to_vec1::<f32>()?;
    let signs_vec = match signs {
        Some(s) => Some(
            s.to_device(&crate::Device::Cpu)?
                .to_dtype(crate::DType::F32)?
                .flatten_all()?
                .to_vec1::<f32>()?,
        ),
        None => None,
    };
    for row in vals.chunks_exact_mut(width) {
        let mean_sq =
            (row.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / (width as f64)) as f32;
        let inv_rms = 1.0f32 / (mean_sq + eps).sqrt();
        for (col, v) in row.iter_mut().enumerate() {
            *v = *v * inv_rms * w_vec[col];
        }
    }
    fwht_cpu_inplace(&mut vals, signs_vec.as_deref(), width, block_size, false);
    let y = crate::Tensor::from_vec(vals, x.shape(), &crate::Device::Cpu)?.to_device(&orig_dev)?;
    Ok((res_out, y))
}

pub fn hadamard_fwht(
    x: &crate::Tensor,
    signs: Option<&crate::Tensor>,
    block_size: usize,
    is_inverse: bool,
) -> Result<crate::Tensor> {
    if !block_size.is_power_of_two() || block_size == 0 || block_size > 8192 {
        crate::bail!("hadamard_fwht block_size must be a power of 2 in 1..=8192, got {block_size}");
    }
    let orig_shape = x.shape().clone();
    let Some(&width) = orig_shape.dims().last() else {
        crate::bail!("hadamard_fwht expects rank >= 1");
    };
    if !width.is_multiple_of(block_size) {
        crate::bail!("hadamard_fwht width {width} is not divisible by block_size {block_size}");
    }
    let elem_count = orig_shape.elem_count();
    let total_blocks = (elem_count / block_size) as i64;
    if total_blocks == 0 {
        return x.contiguous();
    }

    if let Some(s) = signs {
        if s.dtype() != crate::DType::F32 {
            crate::bail!("hadamard_fwht signs must be F32, got {:?}", s.dtype());
        }
        if s.elem_count() != width {
            crate::bail!("hadamard_fwht signs len {} != width {width}", s.elem_count());
        }
    }

    if !x.device().is_cuda() {
        return hadamard_fwht_cpu_fallback(x, signs, width, block_size, is_inverse);
    }

    let x = x.contiguous()?;
    let signs_contig = match signs {
        Some(s) => Some(s.to_device(x.device())?.contiguous()?),
        None => None,
    };

    let (x_storage, x_layout) = x.storage_and_layout();
    let crate::Storage::Cuda(x_cuda) = &*x_storage else {
        unreachable!()
    };
    let dev = &x_cuda.device;
    let (x_o1, x_o2) = x_layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "hadamard_fwht",
        }
        .bt()
    })?;

    let signs_guard = signs_contig.as_ref().map(|s| s.storage_and_layout());
    let signs_view = if let Some((s_storage, s_layout)) = signs_guard.as_ref() {
        let crate::Storage::Cuda(s_cuda) = &**s_storage else {
            crate::bail!("hadamard_fwht signs must be on CUDA");
        };
        let (s_o1, s_o2) = s_layout.contiguous_offsets().ok_or_else(|| {
            crate::Error::RequiresContiguous {
                op: "hadamard_fwht_signs",
            }
            .bt()
        })?;
        Some(s_cuda.as_cuda_slice::<f32>()?.slice(s_o1..s_o2))
    } else {
        None
    };

    let has_signs = i32::from(signs_view.is_some());
    let is_inv = i32::from(is_inverse);
    let b_i32 = block_size as i32;
    let w_i32 = width as i32;
    let grid_x = (total_blocks as u32).clamp(1, 65_535);
    let (threads, shared_mem_bytes) = fwht_launch_params(block_size);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (grid_x, 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let null_ptr = 0u64;

    let prof = cuda_profile_enabled();
    if prof {
        crate::backend::BackendDevice::synchronize(dev)?;
    }
    let t_fwht = prof.then(std::time::Instant::now);

    let out_storage = match x.dtype() {
        crate::DType::F32 => {
            let x_slice = x_cuda.as_cuda_slice::<f32>()?.slice(x_o1..x_o2);
            let dst = unsafe { dev.alloc::<f32>(elem_count)? };
            let func = dev.get_or_load_func("hadamard_fwht_f32", &hanzo_kernels::IQUANT_MMVQ)?;
            let mut builder = func.builder();
            builder.arg(&x_slice);
            if let Some(ref sv) = signs_view {
                builder.arg(sv);
            } else {
                builder.arg(&null_ptr);
            }
            builder.arg(&dst);
            barg!(builder, b_i32, w_i32, has_signs, is_inv, total_blocks);
            unsafe { builder.launch(cfg) }.w()?;
            CudaStorage::wrap_cuda_slice(dst, dev.clone())
        }
        crate::DType::F16 => {
            let x_slice = x_cuda.as_cuda_slice::<f16>()?.slice(x_o1..x_o2);
            let dst = unsafe { dev.alloc::<f16>(elem_count)? };
            let func = dev.get_or_load_func("hadamard_fwht_f16", &hanzo_kernels::IQUANT_MMVQ)?;
            let mut builder = func.builder();
            builder.arg(&x_slice);
            if let Some(ref sv) = signs_view {
                builder.arg(sv);
            } else {
                builder.arg(&null_ptr);
            }
            builder.arg(&dst);
            barg!(builder, b_i32, w_i32, has_signs, is_inv, total_blocks);
            unsafe { builder.launch(cfg) }.w()?;
            CudaStorage::wrap_cuda_slice(dst, dev.clone())
        }
        crate::DType::BF16 => {
            let x_slice = x_cuda.as_cuda_slice::<half::bf16>()?.slice(x_o1..x_o2);
            let dst = unsafe { dev.alloc::<half::bf16>(elem_count)? };
            let func = dev.get_or_load_func("hadamard_fwht_bf16", &hanzo_kernels::IQUANT_MMVQ)?;
            let mut builder = func.builder();
            builder.arg(&x_slice);
            if let Some(ref sv) = signs_view {
                builder.arg(sv);
            } else {
                builder.arg(&null_ptr);
            }
            builder.arg(&dst);
            barg!(builder, b_i32, w_i32, has_signs, is_inv, total_blocks);
            unsafe { builder.launch(cfg) }.w()?;
            CudaStorage::wrap_cuda_slice(dst, dev.clone())
        }
        dt => crate::bail!("hadamard_fwht unsupported dtype {dt:?}"),
    };
    if let Some(t0) = t_fwht {
        crate::backend::BackendDevice::synchronize(dev)?;
        PROF_FWHT_NS.fetch_add(
            t0.elapsed().as_nanos() as u64,
            std::sync::atomic::Ordering::Relaxed,
        );
        PROF_FWHT_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    drop(signs_guard);
    drop(x_storage);
    Ok(crate::tensor::from_storage(
        crate::Storage::Cuda(out_storage),
        orig_shape,
        crate::op::BackpropOp::none(),
        false,
    ))
}

#[allow(clippy::too_many_arguments)]
fn rms_norm_hadamard_fwht_impl(
    x: &crate::Tensor,
    residual: Option<&crate::Tensor>,
    weight: &crate::Tensor,
    eps: f32,
    signs: Option<&crate::Tensor>,
    block_size: usize,
    want_dst: bool,
    want_q8: bool,
) -> Result<(Option<crate::Tensor>, Option<crate::Tensor>, Option<IqQ8Activation>)> {
    if !block_size.is_power_of_two() || block_size == 0 || block_size > 8192 {
        crate::bail!(
            "rms_norm_hadamard_fwht block_size must be a power of 2 in 1..=8192, got {block_size}"
        );
    }
    let orig_shape = x.shape().clone();
    let Some(&width) = orig_shape.dims().last() else {
        crate::bail!("rms_norm_hadamard_fwht expects rank >= 1");
    };
    if !width.is_multiple_of(block_size) {
        crate::bail!(
            "rms_norm_hadamard_fwht width {width} is not divisible by block_size {block_size}"
        );
    }
    if weight.dims1()? != width {
        crate::bail!(
            "rms_norm_hadamard_fwht weight shape {:?} does not match width {width}",
            weight.shape()
        );
    }
    if let Some(res) = residual {
        if res.shape() != x.shape() || res.dtype() != x.dtype() {
            crate::bail!(
                "rms_norm_hadamard_fwht residual {:?}/{:?} must match x {:?}/{:?}",
                res.shape(),
                res.dtype(),
                x.shape(),
                x.dtype()
            );
        }
    }
    if let Some(s) = signs {
        if s.dtype() != crate::DType::F32 {
            crate::bail!("rms_norm_hadamard_fwht signs must be F32, got {:?}", s.dtype());
        }
        if s.elem_count() != width {
            crate::bail!(
                "rms_norm_hadamard_fwht signs len {} != width {width}",
                s.elem_count()
            );
        }
    }

    if !x.device().is_cuda() || block_size < 32 {
        let (res_out, dst_f32) = rms_norm_hadamard_fwht_cpu_fallback(
            x, residual, weight, eps, signs, width, block_size,
        )?;
        let q8 = if want_q8 {
            Some(IqQ8Activation::quantize(&dst_f32)?)
        } else {
            None
        };
        let dst = if want_dst { Some(dst_f32) } else { None };
        return Ok((res_out, dst, q8));
    }

    let elem_count = orig_shape.elem_count();
    let nrows = elem_count / width;
    if nrows == 0 {
        let res_out = match residual {
            Some(r) => Some((x + r)?),
            None => None,
        };
        let dst = if want_dst {
            Some(x.to_dtype(crate::DType::F32)?)
        } else {
            None
        };
        return Ok((res_out, dst, None));
    }

    let x_contig = x.contiguous()?;
    let res_contig = match residual {
        Some(r) => Some(r.to_device(x.device())?.contiguous()?),
        None => None,
    };
    let w_contig = weight.to_device(x.device())?.contiguous()?;
    let signs_contig = match signs {
        Some(s) => Some(s.to_device(x.device())?.contiguous()?),
        None => None,
    };

    let (x_storage, x_layout) = x_contig.storage_and_layout();
    let crate::Storage::Cuda(x_cuda) = &*x_storage else {
        unreachable!()
    };
    let dev = &x_cuda.device;
    let (x_o1, x_o2) = x_layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "rms_norm_hadamard_fwht",
        }
        .bt()
    })?;

    let res_guard = res_contig.as_ref().map(|r| r.storage_and_layout());
    let (w_storage, w_layout) = w_contig.storage_and_layout();
    let crate::Storage::Cuda(w_cuda) = &*w_storage else {
        crate::bail!("rms_norm_hadamard_fwht weight must be on CUDA");
    };
    let (w_o1, w_o2) = w_layout.contiguous_offsets().ok_or_else(|| {
        crate::Error::RequiresContiguous {
            op: "rms_norm_hadamard_fwht_weight",
        }
        .bt()
    })?;
    let w_dtype_code: i32 = match w_contig.dtype() {
        crate::DType::F32 => 0,
        crate::DType::F16 => 1,
        crate::DType::BF16 => 2,
        dt => crate::bail!("rms_norm_hadamard_fwht unsupported weight dtype {dt:?}"),
    };

    let signs_guard = signs_contig.as_ref().map(|s| s.storage_and_layout());
    let signs_view = if let Some((s_storage, s_layout)) = signs_guard.as_ref() {
        let crate::Storage::Cuda(s_cuda) = &**s_storage else {
            crate::bail!("rms_norm_hadamard_fwht signs must be on CUDA");
        };
        let (s_o1, s_o2) = s_layout.contiguous_offsets().ok_or_else(|| {
            crate::Error::RequiresContiguous {
                op: "rms_norm_hadamard_fwht_signs",
            }
            .bt()
        })?;
        Some(s_cuda.as_cuda_slice::<f32>()?.slice(s_o1..s_o2))
    } else {
        None
    };

    let dst_slice = if want_dst {
        Some(unsafe { dev.alloc::<f32>(elem_count)? })
    } else {
        None
    };

    let (q8_slices, total_q8_blocks_padded) = if want_q8 {
        if orig_shape.rank() < 2 || !width.is_multiple_of(32) {
            crate::bail!(
                "rms_norm_hadamard_fwht_iq_q8 requires rank >= 2 and width % 32 == 0, got {:?}",
                orig_shape
            );
        }
        let width_padded = pad(width, IQ_Q8_ROW_PAD);
        let total_padded = nrows * (width_padded / 32);
        let xq = unsafe { dev.alloc::<u8>(total_padded * 32)? };
        let xd = unsafe { dev.alloc::<f16>(total_padded)? };
        (Some((xq, xd)), total_padded as i64)
    } else {
        (None, 0i64)
    };

    let has_residual = i32::from(residual.is_some());
    let has_residual_out = has_residual;
    let has_signs = i32::from(signs_view.is_some());
    let has_dst = i32::from(want_dst);
    let has_q8 = i32::from(want_q8);
    let b_i32 = block_size as i32;
    let w_i32 = width as i32;
    let nrows_i64 = nrows as i64;
    let (threads, shared_mem_bytes) = fwht_launch_params(block_size);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((nrows as u32).clamp(1, 65_535), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let null_ptr = 0u64;

    let w_f32_view = (w_dtype_code == 0)
        .then(|| w_cuda.as_cuda_slice::<f32>().map(|s| s.slice(w_o1..w_o2)))
        .transpose()?;
    let w_f16_view = (w_dtype_code == 1)
        .then(|| w_cuda.as_cuda_slice::<f16>().map(|s| s.slice(w_o1..w_o2)))
        .transpose()?;
    let w_bf16_view = (w_dtype_code == 2)
        .then(|| w_cuda.as_cuda_slice::<bf16>().map(|s| s.slice(w_o1..w_o2)))
        .transpose()?;

    macro_rules! push_common_tail {
        ($builder:ident) => {{
            if let Some(ref ws) = w_f32_view {
                $builder.arg(ws);
            } else if let Some(ref ws) = w_f16_view {
                $builder.arg(ws);
            } else if let Some(ref ws) = w_bf16_view {
                $builder.arg(ws);
            }
            if let Some(ref sv) = signs_view {
                $builder.arg(sv);
            } else {
                $builder.arg(&null_ptr);
            }
            if let Some(ref ds) = dst_slice {
                $builder.arg(ds);
            } else {
                $builder.arg(&null_ptr);
            }
            if let Some((ref xq, ref xd)) = q8_slices {
                $builder.arg(xq);
                $builder.arg(xd);
            } else {
                $builder.arg(&null_ptr);
                $builder.arg(&null_ptr);
            }
            barg!(
                $builder,
                eps,
                b_i32,
                w_i32,
                w_dtype_code,
                has_residual,
                has_residual_out,
                has_signs,
                has_dst,
                has_q8,
                nrows_i64,
                total_q8_blocks_padded
            );
            unsafe { $builder.launch(cfg) }.w()?;
        }};
    }

    let res_out_storage = match x_contig.dtype() {
        crate::DType::F32 => {
            let x_slice = x_cuda.as_cuda_slice::<f32>()?.slice(x_o1..x_o2);
            let func = dev.get_or_load_func(
                "rms_norm_hadamard_fwht_f32",
                &hanzo_kernels::IQUANT_MMVQ,
            )?;
            let mut builder = func.builder();
            builder.arg(&x_slice);
            if let Some((r_storage, r_layout)) = res_guard.as_ref() {
                let crate::Storage::Cuda(r_cuda) = &**r_storage else {
                    crate::bail!("residual must be on CUDA");
                };
                let (r_o1, r_o2) = r_layout.contiguous_offsets().unwrap();
                let r_slice = r_cuda.as_cuda_slice::<f32>()?.slice(r_o1..r_o2);
                let r_out = unsafe { dev.alloc::<f32>(elem_count)? };
                builder.arg(&r_slice);
                builder.arg(&r_out);
                push_common_tail!(builder);
                Some(CudaStorage::wrap_cuda_slice(r_out, dev.clone()))
            } else {
                builder.arg(&null_ptr);
                builder.arg(&null_ptr);
                push_common_tail!(builder);
                None
            }
        }
        crate::DType::BF16 => {
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?.slice(x_o1..x_o2);
            let func = dev.get_or_load_func(
                "rms_norm_hadamard_fwht_bf16",
                &hanzo_kernels::IQUANT_MMVQ,
            )?;
            let mut builder = func.builder();
            builder.arg(&x_slice);
            if let Some((r_storage, r_layout)) = res_guard.as_ref() {
                let crate::Storage::Cuda(r_cuda) = &**r_storage else {
                    crate::bail!("residual must be on CUDA");
                };
                let (r_o1, r_o2) = r_layout.contiguous_offsets().unwrap();
                let r_slice = r_cuda.as_cuda_slice::<bf16>()?.slice(r_o1..r_o2);
                let r_out = unsafe { dev.alloc::<bf16>(elem_count)? };
                builder.arg(&r_slice);
                builder.arg(&r_out);
                push_common_tail!(builder);
                Some(CudaStorage::wrap_cuda_slice(r_out, dev.clone()))
            } else {
                builder.arg(&null_ptr);
                builder.arg(&null_ptr);
                push_common_tail!(builder);
                None
            }
        }
        crate::DType::F16 => {
            let x_slice = x_cuda.as_cuda_slice::<f16>()?.slice(x_o1..x_o2);
            let func = dev.get_or_load_func(
                "rms_norm_hadamard_fwht_f16",
                &hanzo_kernels::IQUANT_MMVQ,
            )?;
            let mut builder = func.builder();
            builder.arg(&x_slice);
            if let Some((r_storage, r_layout)) = res_guard.as_ref() {
                let crate::Storage::Cuda(r_cuda) = &**r_storage else {
                    crate::bail!("residual must be on CUDA");
                };
                let (r_o1, r_o2) = r_layout.contiguous_offsets().unwrap();
                let r_slice = r_cuda.as_cuda_slice::<f16>()?.slice(r_o1..r_o2);
                let r_out = unsafe { dev.alloc::<f16>(elem_count)? };
                builder.arg(&r_slice);
                builder.arg(&r_out);
                push_common_tail!(builder);
                Some(CudaStorage::wrap_cuda_slice(r_out, dev.clone()))
            } else {
                builder.arg(&null_ptr);
                builder.arg(&null_ptr);
                push_common_tail!(builder);
                None
            }
        }
        dt => crate::bail!("rms_norm_hadamard_fwht unsupported dtype {dt:?}"),
    };

    let dev_clone = dev.clone();
    drop(signs_guard);
    drop(w_storage);
    drop(res_guard);
    drop(x_storage);

    let res_out_tensor = res_out_storage.map(|s| {
        crate::tensor::from_storage(
            crate::Storage::Cuda(s),
            orig_shape.clone(),
            crate::op::BackpropOp::none(),
            false,
        )
    });
    let dst_tensor = dst_slice.map(|ds| {
        crate::tensor::from_storage(
            crate::Storage::Cuda(CudaStorage::wrap_cuda_slice(ds, dev_clone.clone())),
            orig_shape.clone(),
            crate::op::BackpropOp::none(),
            false,
        )
    });
    let q8_act = q8_slices.map(|(xq, xd)| IqQ8Activation {
        inner: IqQ8Inner::Cuda {
            xq,
            xd,
            device: dev_clone,
        },
        shape: orig_shape,
        dtype: crate::DType::F32,
        b_size: nrows,
        ncols: width,
    });
    Ok((res_out_tensor, dst_tensor, q8_act))
}

pub fn rms_norm_hadamard_fwht(
    x: &crate::Tensor,
    residual: Option<&crate::Tensor>,
    weight: &crate::Tensor,
    eps: f32,
    signs: Option<&crate::Tensor>,
    block_size: usize,
) -> Result<(Option<crate::Tensor>, crate::Tensor)> {
    let (res_out, dst, _) = rms_norm_hadamard_fwht_impl(
        x, residual, weight, eps, signs, block_size, true, false,
    )?;
    Ok((res_out, dst.unwrap()))
}

fn hadamard_fwht_q8_impl(
    x: &crate::Tensor,
    signs: Option<&crate::Tensor>,
    block_size: usize,
    is_inverse: bool,
) -> Result<IqQ8Activation> {
    let orig_shape = x.shape().clone();
    if orig_shape.rank() < 2 {
        crate::bail!("hadamard_fwht_iq_q8 expects rank >= 2, got {:?}", orig_shape);
    }
    let width = *orig_shape.dims().last().unwrap();
    if !x.device().is_cuda() || block_size < 32 || !width.is_multiple_of(32) {
        let y = hadamard_fwht(x, signs, block_size, is_inverse)?;
        return IqQ8Activation::quantize(&y);
    }
    if !block_size.is_power_of_two() || block_size > 8192 || !width.is_multiple_of(block_size) {
        crate::bail!(
            "hadamard_fwht_iq_q8 invalid width {width} or block_size {block_size}"
        );
    }
    if let Some(s) = signs {
        if s.dtype() != crate::DType::F32 || s.elem_count() != width {
            crate::bail!("hadamard_fwht_iq_q8 signs must be F32 of len {width}");
        }
    }
    let elem_count = orig_shape.elem_count();
    let nrows = elem_count / width;
    let total_blocks = (elem_count / block_size) as i64;
    let x_contig = x.contiguous()?;
    let signs_contig = match signs {
        Some(s) => Some(s.to_device(x.device())?.contiguous()?),
        None => None,
    };
    let (x_storage, x_layout) = x_contig.storage_and_layout();
    let crate::Storage::Cuda(x_cuda) = &*x_storage else {
        unreachable!()
    };
    let dev = &x_cuda.device;
    let (x_o1, x_o2) = x_layout.contiguous_offsets().unwrap();
    let signs_guard = signs_contig.as_ref().map(|s| s.storage_and_layout());
    let signs_view = if let Some((s_storage, s_layout)) = signs_guard.as_ref() {
        let crate::Storage::Cuda(s_cuda) = &**s_storage else {
            crate::bail!("signs must be on CUDA");
        };
        let (s_o1, s_o2) = s_layout.contiguous_offsets().unwrap();
        Some(s_cuda.as_cuda_slice::<f32>()?.slice(s_o1..s_o2))
    } else {
        None
    };

    let width_padded = pad(width, IQ_Q8_ROW_PAD);
    let total_q8_blocks_padded = (nrows * (width_padded / 32)) as i64;
    let xq = unsafe { dev.alloc::<u8>((total_q8_blocks_padded as usize) * 32)? };
    let xd = unsafe { dev.alloc::<f16>(total_q8_blocks_padded as usize)? };

    let has_signs = i32::from(signs_view.is_some());
    let is_inv = i32::from(is_inverse);
    let has_y = 0i32;
    let b_i32 = block_size as i32;
    let w_i32 = width as i32;
    let (threads, shared_mem_bytes) = fwht_launch_params(block_size);
    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: ((total_blocks as u32).clamp(1, 65_535), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes,
    };
    let null_ptr = 0u64;

    match x_contig.dtype() {
        crate::DType::F32 => {
            let x_slice = x_cuda.as_cuda_slice::<f32>()?.slice(x_o1..x_o2);
            let func = dev.get_or_load_func("hadamard_fwht_q8_f32", &hanzo_kernels::IQUANT_MMVQ)?;
            let mut builder = func.builder();
            builder.arg(&x_slice);
            if let Some(ref sv) = signs_view {
                builder.arg(sv);
            } else {
                builder.arg(&null_ptr);
            }
            builder.arg(&null_ptr);
            builder.arg(&xq);
            builder.arg(&xd);
            barg!(
                builder,
                b_i32,
                w_i32,
                has_signs,
                is_inv,
                has_y,
                total_blocks,
                total_q8_blocks_padded
            );
            unsafe { builder.launch(cfg) }.w()?;
        }
        crate::DType::BF16 => {
            let x_slice = x_cuda.as_cuda_slice::<bf16>()?.slice(x_o1..x_o2);
            let func = dev.get_or_load_func("hadamard_fwht_q8_bf16", &hanzo_kernels::IQUANT_MMVQ)?;
            let mut builder = func.builder();
            builder.arg(&x_slice);
            if let Some(ref sv) = signs_view {
                builder.arg(sv);
            } else {
                builder.arg(&null_ptr);
            }
            builder.arg(&null_ptr);
            builder.arg(&xq);
            builder.arg(&xd);
            barg!(
                builder,
                b_i32,
                w_i32,
                has_signs,
                is_inv,
                has_y,
                total_blocks,
                total_q8_blocks_padded
            );
            unsafe { builder.launch(cfg) }.w()?;
        }
        crate::DType::F16 => {
            let x_slice = x_cuda.as_cuda_slice::<f16>()?.slice(x_o1..x_o2);
            let func = dev.get_or_load_func("hadamard_fwht_q8_f16", &hanzo_kernels::IQUANT_MMVQ)?;
            let mut builder = func.builder();
            builder.arg(&x_slice);
            if let Some(ref sv) = signs_view {
                builder.arg(sv);
            } else {
                builder.arg(&null_ptr);
            }
            builder.arg(&null_ptr);
            builder.arg(&xq);
            builder.arg(&xd);
            barg!(
                builder,
                b_i32,
                w_i32,
                has_signs,
                is_inv,
                has_y,
                total_blocks,
                total_q8_blocks_padded
            );
            unsafe { builder.launch(cfg) }.w()?;
        }
        dt => crate::bail!("hadamard_fwht_iq_q8 unsupported dtype {dt:?}"),
    }

    let dev_clone = dev.clone();
    let out_dt = x_contig.dtype();
    drop(signs_guard);
    drop(x_storage);
    Ok(IqQ8Activation {
        inner: IqQ8Inner::Cuda {
            xq,
            xd,
            device: dev_clone,
        },
        shape: orig_shape,
        dtype: out_dt,
        b_size: nrows,
        ncols: width,
    })
}

impl crate::Tensor {
    pub fn rms_norm_hadamard_fwht(
        &self,
        weight: &crate::Tensor,
        eps: f32,
        signs: Option<&crate::Tensor>,
        block_size: usize,
    ) -> Result<crate::Tensor> {
        let (_, y) = rms_norm_hadamard_fwht(self, None, weight, eps, signs, block_size)?;
        Ok(y)
    }

    pub fn add_rms_norm_hadamard_fwht(
        &self,
        residual: &crate::Tensor,
        weight: &crate::Tensor,
        eps: f32,
        signs: Option<&crate::Tensor>,
        block_size: usize,
    ) -> Result<(crate::Tensor, crate::Tensor)> {
        let (res_out, y) =
            rms_norm_hadamard_fwht(self, Some(residual), weight, eps, signs, block_size)?;
        Ok((res_out.unwrap(), y))
    }
}

impl super::QMatMul {
    pub fn quantize_iq_q8(xs: &crate::Tensor) -> Result<IqQ8Activation> {
        IqQ8Activation::quantize(xs)
    }

    pub fn hadamard_fwht_iq_q8(
        xs: &crate::Tensor,
        signs: Option<&crate::Tensor>,
        block_size: usize,
        is_inverse: bool,
    ) -> Result<IqQ8Activation> {
        hadamard_fwht_q8_impl(xs, signs, block_size, is_inverse)
    }

    pub fn rms_norm_hadamard_fwht_iq_q8(
        xs: &crate::Tensor,
        weight: &crate::Tensor,
        eps: f32,
        signs: Option<&crate::Tensor>,
        block_size: usize,
    ) -> Result<IqQ8Activation> {
        let (_, _, q8) = rms_norm_hadamard_fwht_impl(
            xs, None, weight, eps, signs, block_size, false, true,
        )?;
        Ok(q8.unwrap())
    }

    pub fn add_rms_norm_hadamard_fwht_iq_q8(
        xs: &crate::Tensor,
        residual: &crate::Tensor,
        weight: &crate::Tensor,
        eps: f32,
        signs: Option<&crate::Tensor>,
        block_size: usize,
    ) -> Result<(crate::Tensor, IqQ8Activation)> {
        let (res_out, _, q8) = rms_norm_hadamard_fwht_impl(
            xs,
            Some(residual),
            weight,
            eps,
            signs,
            block_size,
            false,
            true,
        )?;
        Ok((res_out.unwrap(), q8.unwrap()))
    }

    pub fn forward_iq_q8(&self, q8: &IqQ8Activation) -> Result<crate::Tensor> {
        self.forward_iq_q8_dtype(q8, q8.dtype)
    }

    pub fn forward_iq_q8_dtype(
        &self,
        q8: &IqQ8Activation,
        out_dtype: crate::DType,
    ) -> Result<crate::Tensor> {
        if let IqQ8Inner::HostFallback(ref xs) = q8.inner {
            use crate::Module;
            let y = self.forward(xs)?;
            return if y.dtype() == out_dtype {
                Ok(y)
            } else {
                y.to_dtype(out_dtype)
            };
        }
        match self {
            Self::QTensor(qt) => match &qt.storage {
                QStorage::Cuda(qcuda) if iquant_dp4a_suffix(qcuda.dtype).is_some() => {
                    let (out, out_shape) =
                        qcuda.fwd_shared_q8_dtype(&qt.shape, q8, out_dtype)?;
                    Ok(crate::tensor::from_storage(
                        crate::Storage::Cuda(out),
                        out_shape,
                        crate::op::BackpropOp::none(),
                        false,
                    ))
                }
                _ => crate::bail!(
                    "QMatMul::forward_iq_q8 requires an i-quant/ternary CUDA QTensor, got {:?}",
                    qt.dtype()
                ),
            },
            _ => crate::bail!("QMatMul::forward_iq_q8 requires QMatMul::QTensor variant"),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cuda_quantize_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let el = 256;
        let el_padded = pad(el, MATRIX_ROW_PADDING);
        let y_size_in_bytes =
            el_padded * GgmlDType::Q8_1.type_size() / GgmlDType::Q8_1.block_size();
        let mut y_q8_1 = unsafe { dev.alloc::<u8>(y_size_in_bytes)? };
        let vs: Vec<f32> = (0..el).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        quantize_q8_1(&y.as_view(), &mut y_q8_1, el, 1, &dev)?;
        Ok(())
    }

    #[test]
    fn cuda_mmv_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols).map(|v| v as f32).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_vec_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            /* b_size */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        // for n = 255, n.(n+1).(2n+1) / 6 = 5559680
        // Q8 means 1/256 precision.
        assert_eq!(vs[0], 5561664.5);

        let cuda_storage = dequantize_mul_mat_vec(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* ncols */ ncols,
            /* nrows */ 1,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;
        assert_eq!(vs.len(), 1);
        assert_eq!(vs[0], 5561851.0);
        Ok(())
    }

    #[test]
    fn cuda_mm_q8_1() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let ncols = 256;
        let vs: Vec<f32> = (0..ncols * 4).map(|v| v as f32 / 4.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * 4, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ 4,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ 4,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let vs = dev.clone_dtoh(&vs.as_view())?;

        /*
           x = torch.tensor([float(v) for v in range(1024)]).reshape(4, 256)
           x @ x.t() / 16
        tensor([[  347480.0000,   869720.0000,  1391960.0000,  1914200.0000],
                [  869720.0000,  2440536.0000,  4011352.0000,  5582166.5000],
                [ 1391960.0000,  4011352.0000,  6630742.0000,  9250132.0000],
                [ 1914200.0000,  5582166.5000,  9250132.0000, 12918099.0000]])
                */
        assert_eq!(vs.len(), 16);
        assert_eq!(vs[0], 347604.0);
        assert_eq!(vs[1], 888153.06);
        assert_eq!(vs[4], 869780.7);
        assert_eq!(vs[5], 2483145.0);
        assert_eq!(vs[11], 9407368.0);
        assert_eq!(vs[14], 9470856.0);
        assert_eq!(vs[15], 13138824.0);
        Ok(())
    }

    // The following test used to fail under compute-sanitizer until #2526.
    #[test]
    fn cuda_mm_q8_1_pad() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (x_rows, ncols, y_cols) = (4, 16, 2048);
        let vs: Vec<f32> = (0..ncols * y_cols).map(|v| v as f32 / 256.).collect();
        let y = dev.clone_htod(&vs)?;
        let mut xs = QCudaStorage::zeros(&dev, ncols * x_rows, GgmlDType::Q4_0)?;
        xs.quantize(&CudaStorage::wrap_cuda_slice(y.clone(), dev.clone()))?;
        let cuda_storage = mul_mat_via_q8_1(
            &xs.data,
            &y.as_view(),
            /* dtype */ GgmlDType::Q4_0,
            /* x_rows */ x_rows,
            /* x_cols */ ncols,
            /* y_rows */ ncols,
            /* y_cols */ y_cols,
            &dev,
        )?;
        let vs = cuda_storage.as_cuda_slice::<f32>()?;
        let _vs = dev.clone_dtoh(&vs.as_view())?;
        Ok(())
    }

    // Fused indexed-MoE for the 32-element block types, MULTI-EXPERT. ids select experts {0,1,2,0}, so
    // slots route to expert_id > 0 -- a wrong per-expert weight stride (the QK_K-vs-`qk` bug) reads the
    // wrong bank and yields garbage, so this is the stride-fix gate. Q4_0/Q4_1/Q5_0/Q5_1 are the newly
    // added kernels; Q8_0 re-checks the pre-existing one whose stride this fix corrects (no regression).
    // Oracle = the SAME stored quant weights dequantized (zero weight error) matmul'd against the raw f32
    // input; the only delta is the kernel's internal q8_1 input quant (~1/256 per element).
    #[test]
    fn cuda_indexed_moe_legacy_32block() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        // k=256 is a multiple of BOTH the 32-element legacy block and the 256-element K-quant block, so
        // ONE test covers every fused type. n=8 > nblocks-if-broken (ceil(8/4)=2), so the row-coverage
        // bug leaves rows 2..8 at zero -> caught here.
        let (e, n, k) = (3usize, 8usize, 256usize);
        let (batch, topk) = (2usize, 2usize);
        let ids: Vec<u32> = vec![0, 1, 2, 0]; // [batch, topk] row-major; covers expert_id > 0

        // Deterministic varied small weights/input (LCG) so distinct experts have distinct weights.
        let mut st: u32 = 0x1234_5678;
        let mut nextf = || -> f32 {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            ((st >> 8) as f32 / (1u32 << 24) as f32) - 0.5 // [-0.5, 0.5)
        };
        let w: Vec<f32> = (0..e * n * k).map(|_| nextf() * 0.5).collect();
        let inp: Vec<f32> = (0..batch * k).map(|_| nextf()).collect();

        // The 5 newly-added legacy 32-block kernels + the 5 pre-existing K-quant kernels: the launch-row
        // and stride fixes are shared, so all 10 must agree with the oracle.
        for dtype in [
            GgmlDType::Q4_0,
            GgmlDType::Q4_1,
            GgmlDType::Q5_0,
            GgmlDType::Q5_1,
            GgmlDType::Q8_0,
            GgmlDType::Q2K,
            GgmlDType::Q3K,
            GgmlDType::Q4K,
            GgmlDType::Q5K,
            GgmlDType::Q6K,
        ] {
            // Quantized [e, n, k] expert bank.
            let mut bank = QCudaStorage::zeros(&dev, e * n * k, dtype)?;
            let w_cuda = CudaStorage::wrap_cuda_slice(dev.clone_htod(&w)?, dev.clone());
            bank.quantize(&w_cuda)?;
            // The dequantized bank IS the weights the kernel reads -> exact-weight oracle.
            let deq = bank.dequantize(e * n * k)?;
            let w_deq = dev.clone_dtoh(&deq.as_cuda_slice::<f32>()?.as_view())?;

            // input [batch, 1, k] (input_dim1 == 1: shared input vector per batch), ids [batch, topk].
            let input = CudaStorage::wrap_cuda_slice(dev.clone_htod(&inp)?, dev.clone());
            let input_l = crate::Layout::contiguous((batch, 1usize, k));
            let ids_storage = CudaStorage::wrap_cuda_slice(dev.clone_htod(&ids)?, dev.clone());
            let ids_l = crate::Layout::contiguous((batch, topk));
            let self_shape: crate::Shape = (e, n, k).into();

            let (out_storage, out_shape) =
                bank.indexed_moe_forward(&self_shape, &input, &input_l, &ids_storage, &ids_l)?;
            assert_eq!(out_shape.dims().to_vec(), vec![batch, topk, n]);
            let got = dev.clone_dtoh(&out_storage.as_cuda_slice::<f32>()?.as_view())?;

            // Oracle: out[b,t,j] = sum_i w_deq[expert][j][i] * inp[b][i].
            let mut max_abs = 0f32;
            let mut max_err = 0f32;
            for b in 0..batch {
                for t in 0..topk {
                    let expert = ids[b * topk + t] as usize;
                    for j in 0..n {
                        let mut acc = 0f32;
                        for i in 0..k {
                            acc += w_deq[expert * n * k + j * k + i] * inp[b * k + i];
                        }
                        let g = got[(b * topk + t) * n + j];
                        max_abs = max_abs.max(acc.abs());
                        max_err = max_err.max((g - acc).abs());
                    }
                }
            }
            // Correct result tracks the oracle to within the q8_1 input-quant error (~1/256); the stride
            // bug instead reads the wrong expert bank (garbage, ~100% off). 5% of the output scale cleanly
            // separates the two.
            let tol = 0.05 * max_abs + 1e-3;
            assert!(
                max_err <= tol,
                "{dtype:?}: indexed-MoE max_err {max_err} > tol {tol} (max_abs {max_abs})"
            );
        }
        Ok(())
    }

    // i-quant / ternary / NVFP4 weights have no native q8_1 matmul kernel, so QStorage::from_data used to
    // BAIL when loading them onto CUDA -- an i-quant GGUF could not run on CUDA at all. They now upload
    // like any quant and QCudaStorage::fwd matmuls them: the i-quant / ternary types via the native dp4a
    // mmvq DECODE kernel (codebook-decode the weight + dp4a against the q8_1-quantized activation), the
    // rest (NVFP4, Q1_0) via the dequantize-to-f32 dense fallback. This loads a deterministic block
    // pattern for EACH formerly-bailing type onto CUDA, runs fwd, and checks it against the matmul of the
    // SAME dequantized weights on the host -> proves each type LOADS and DECODES (matmuls) coherently on
    // CUDA, within the numeric bound of the path fwd routes it to (dp4a q8_1-activation vs f32-exact dense).
    #[test]
    fn cuda_iquant_load_and_dense_matmul() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let cuda_dev = crate::Device::Cuda(dev.clone());
        let (m, n, k) = (2usize, 4usize, 256usize); // m tokens; [n,k] weight; k spans 256- and 32-elem blocks
        let inp: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.05 - 0.4).collect();

        for dtype in [
            GgmlDType::IQ2_XXS,
            GgmlDType::IQ2_XS,
            GgmlDType::IQ2_S,
            GgmlDType::IQ3_XXS,
            GgmlDType::IQ3_S,
            GgmlDType::IQ4_XS,
            GgmlDType::IQ1_S,
            GgmlDType::IQ1_M,
            GgmlDType::TQ1_0,
            GgmlDType::TQ2_0,
            GgmlDType::NVFP4,
            GgmlDType::Q1_0,
            GgmlDType::PQ2_0,
            GgmlDType::PTQ1_0,
        ] {
            // Bounded byte pattern (<= 0x3F) so every embedded f16 scale decodes finite (no NaN/Inf), yet
            // varied. Sizing comes from the dtype so it is correct for both 256- and 32-element blocks.
            let nbytes = (n * k / dtype.block_size()) * dtype.type_size();
            let bytes: Vec<u8> = (0..nbytes)
                .map(|i| (i.wrapping_mul(37).wrapping_add(11) & 0x3F) as u8)
                .collect();

            // LOAD onto CUDA (this is the call that used to bail for these dtypes).
            let qcuda = match QStorage::from_data(std::borrow::Cow::Owned(bytes), &cuda_dev, dtype)?
            {
                QStorage::Cuda(s) => s,
                _ => unreachable!("from_data on a CUDA device must yield CUDA storage"),
            };

            // Dequantized weights [n,k] (the exact values the dense matmul consumes) -> host oracle.
            let w_dev = qcuda.dequantize(n * k)?;
            let w_host = dev.clone_dtoh(&w_dev.as_cuda_slice::<f32>()?.as_view())?;

            // CUDA forward: input [m,k] @ weight^T. For the i-quant / ternary types this is the native
            // dp4a mmvq DECODE path (b*m=2 <= 8 -> vec shape); NVFP4 / Q1_0 take the dequant-dense fallback.
            let input = CudaStorage::wrap_cuda_slice(dev.clone_htod(&inp)?, dev.clone());
            let input_l = crate::Layout::contiguous((m, k));
            let self_shape: crate::Shape = (n, k).into();
            let (out, out_shape) = qcuda.fwd(&self_shape, &input, &input_l)?;
            assert_eq!(out_shape.dims().to_vec(), vec![m, n]);
            let got = dev.clone_dtoh(&out.as_cuda_slice::<f32>()?.as_view())?;

            // Host oracle: dense matmul of the SAME dequantized weights against the f32 input,
            // out[r,j] = sum_i w[j,i] * inp[r,i].
            let mut max_abs = 0f32;
            let mut max_err = 0f32;
            for r in 0..m {
                for j in 0..n {
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += w_host[j * k + i] * inp[r * k + i];
                    }
                    let g = got[r * n + j];
                    assert!(g.is_finite(), "{dtype:?}: non-finite CUDA output {g}");
                    max_abs = max_abs.max(acc.abs());
                    max_err = max_err.max((g - acc).abs());
                }
            }
            // Tolerance follows the SAME predicate fwd() routes on. The i-quant / ternary types decode via
            // the dp4a mmvq kernel, which quantizes the f32 activation to q8_1 (int8 + per-32 scale) before
            // the dot -> the result carries the q8_1 activation-quant error, ~1/256 of the activation and
            // inflated by cancellation in the short k=256 dot (measured worst case IQ1_M ~2.7% of scale).
            // 0.05*max_abs + 1e-3 is that q8_1 bound (same gate as the sibling cuda_indexed_moe_legacy_32block
            // test) with ~2x headroom. NVFP4 / Q1_0 have no dp4a kernel -> dequant-dense, f32-exact vs this
            // oracle, so they keep the tight 1e-3*max_abs + 1e-4 accumulation bound. The gate still catches a
            // real defect: a dp4a kernel decoding the wrong weight is ~100% off (>> 5%), and any dense-path
            // regression breaks the tight f32 bound.
            let tol = if iquant_dp4a_suffix(dtype).is_some() {
                0.05 * max_abs + 1e-3
            } else {
                1e-3 * max_abs + 1e-4
            };
            assert!(
                max_err <= tol,
                "{dtype:?}: i-quant CUDA matmul max_err {max_err} > tol {tol} (max_abs {max_abs})"
            );
        }
        Ok(())
    }

    // Dense 2D qmmq PREFILL numeric gate (Lever-1). For a native-q8_1 weight (Q4_K etc.) and a
    // prefill-shaped input (m>1), fast_mmq::try_fwd runs the int8-WMMA stream-K MMQ GEMM -- the SAME
    // kernel family the engine uses for dense prefill. Checks it against (a) the exact-weight f32
    // oracle within q8_1 tolerance, and (b) the dequant-dense reference (dequantize_matmul_dense).
    #[test]
    fn cuda_dense_qmmq_prefill_vs_dequant_and_oracle() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let cuda_dev = crate::Device::Cuda(dev.clone());
        let (m, n, k) = (16usize, 128usize, 512usize);

        let mut st: u32 = 0x9E37_79B9;
        let mut nextf = || -> f32 {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            ((st >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        };
        let inp: Vec<f32> = (0..m * k).map(|_| nextf()).collect();
        let wf: Vec<f32> = (0..n * k).map(|_| nextf() * 0.5).collect();

        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let qcpu = crate::Device::Cpu.qzeros(n * k, dtype)?;
            let qcpu = match qcpu {
                QStorage::Cpu(mut storage) => {
                    storage.from_float(&wf);
                    QStorage::Cpu(storage)
                }
                _ => unreachable!(),
            };
            let bytes = qcpu.data()?.into_owned();
            let qcuda = match QStorage::from_data(std::borrow::Cow::Owned(bytes), &cuda_dev, dtype)?
            {
                QStorage::Cuda(s) => s,
                _ => unreachable!(),
            };

            let w_dev = qcuda.dequantize(n * k)?;
            let w_host = dev.clone_dtoh(&w_dev.as_cuda_slice::<f32>()?.as_view())?;

            let input = CudaStorage::wrap_cuda_slice(dev.clone_htod(&inp)?, dev.clone());
            let input_l = crate::Layout::contiguous((m, k));
            let self_shape: crate::Shape = (n, k).into();

            let (q_out, q_shape) =
                crate::quantized::fast_mmq::try_fwd(&qcuda, &self_shape, &input, &input_l)?
                    .expect("fast_mmq::try_fwd must support q8_1 dense prefill");
            assert_eq!(q_shape.dims().to_vec(), vec![m, n]);
            let q_got = dev.clone_dtoh(&q_out.as_cuda_slice::<f32>()?.as_view())?;

            let (d_out, d_shape) = qcuda.dequantize_matmul_dense(&self_shape, &input, &input_l)?;
            assert_eq!(d_shape.dims().to_vec(), vec![m, n]);
            let d_got = dev.clone_dtoh(&d_out.as_cuda_slice::<f32>()?.as_view())?;

            let mut max_abs = 0f32;
            let mut err_q = 0f32;
            let mut err_d = 0f32;
            let mut err_qd = 0f32;
            for r in 0..m {
                for j in 0..n {
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += w_host[j * k + i] * inp[r * k + i];
                    }
                    let gq = q_got[r * n + j];
                    let gd = d_got[r * n + j];
                    assert!(
                        gq.is_finite() && gd.is_finite(),
                        "{dtype:?}: non-finite output"
                    );
                    max_abs = max_abs.max(acc.abs());
                    err_q = err_q.max((gq - acc).abs());
                    err_d = err_d.max((gd - acc).abs());
                    err_qd = err_qd.max((gq - gd).abs());
                }
            }
            let tol_q = 0.05 * max_abs + 1e-3;
            let tol_d = 1e-3 * max_abs + 1e-4;
            assert!(
                err_q <= tol_q,
                "{dtype:?}: DENSE QMMQ vs oracle err {err_q} > tol {tol_q} (max_abs {max_abs})"
            );
            assert!(
                err_d <= tol_d,
                "{dtype:?}: DEQUANT-DENSE vs oracle err {err_d} > tol {tol_d} (max_abs {max_abs})"
            );
            assert!(
                err_qd <= tol_q,
                "{dtype:?}: QMMQ vs DEQUANT-DENSE err {err_qd} > tol {tol_q} (max_abs {max_abs})"
            );
            eprintln!("[DENSE_QMMQ] {dtype:?}: max_abs={max_abs:.4} err_qmmq={err_q:.5} err_dequant={err_d:.6} err_qmmq_vs_dequant={err_qd:.5}");
        }
        Ok(())
    }

    // Expert-grouped MoE PREFILL GEMM (llama mul_mat_id) numeric gate. batch>1 routes
    // indexed_moe_forward through fast_mmq::indexed_moe_grouped (the new int8 MMQ path); batch==1 keeps
    // the per-slot matvec (decode). This exercises a multi-tile, multi-expert prefill (160 routed slots
    // over 8 experts, with expert 5 deliberately EMPTY to test the expert_bounds zero-range path) and
    // checks BOTH paths against the exact-weight f32 oracle (dequantized bank @ raw input; only the
    // kernel's internal q8_1 activation quant ~1/256 differs). A wrong grouping (bad expert offset,
    // dropped tail columns, scatter collision) reads/writes the wrong place -> ~100% off, far past tol.
    #[test]
    fn cuda_indexed_moe_grouped_prefill_vs_oracle() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let (e, n, k) = (8usize, 64usize, 512usize);
        let (batch, topk) = (40usize, 4usize); // nslots=160, ~20 tok/expert -> real multi-column tiling

        // Deterministic routing leaving expert 5 with ZERO tokens (empty-expert bounds coverage).
        let mut sr: u32 = 0xC0FF_EE11;
        let ids: Vec<u32> = (0..batch * topk)
            .map(|_| {
                sr = sr.wrapping_mul(1664525).wrapping_add(1013904223);
                let ex = (sr >> 9) % (e as u32);
                if ex == 5 {
                    4
                } else {
                    ex
                }
            })
            .collect();

        // Deterministic varied weights/input (LCG), distinct per expert.
        let mut st: u32 = 0x1234_5678;
        let mut nextf = || -> f32 {
            st = st.wrapping_mul(1664525).wrapping_add(1013904223);
            ((st >> 8) as f32 / (1u32 << 24) as f32) - 0.5
        };
        let w: Vec<f32> = (0..e * n * k).map(|_| nextf() * 0.5).collect();
        let inp: Vec<f32> = (0..batch * k).map(|_| nextf()).collect();
        // One-token input for the decode (per-slot) leg: reuse the first token's k values.
        let inp1: Vec<f32> = inp[0..k].to_vec();
        let ids1: Vec<u32> = ids[0..topk].to_vec();

        for dtype in [GgmlDType::Q4K, GgmlDType::Q6K, GgmlDType::Q8_0] {
            let mut bank = QCudaStorage::zeros(&dev, e * n * k, dtype)?;
            bank.quantize(&CudaStorage::wrap_cuda_slice(
                dev.clone_htod(&w)?,
                dev.clone(),
            ))?;
            let deq = bank.dequantize(e * n * k)?;
            let w_deq = dev.clone_dtoh(&deq.as_cuda_slice::<f32>()?.as_view())?;
            let self_shape: crate::Shape = (e, n, k).into();

            // GROUPED prefill (batch>1 -> int8 MMQ).
            let input = CudaStorage::wrap_cuda_slice(dev.clone_htod(&inp)?, dev.clone());
            let input_l = crate::Layout::contiguous((batch, 1usize, k));
            let ids_storage = CudaStorage::wrap_cuda_slice(dev.clone_htod(&ids)?, dev.clone());
            let ids_l = crate::Layout::contiguous((batch, topk));
            let (g_st, g_sh) =
                bank.indexed_moe_forward(&self_shape, &input, &input_l, &ids_storage, &ids_l)?;
            assert_eq!(g_sh.dims().to_vec(), vec![batch, topk, n]);
            let got_g = dev.clone_dtoh(&g_st.as_cuda_slice::<f32>()?.as_view())?;

            let mut max_abs = 0f32;
            let mut err_g = 0f32;
            for b in 0..batch {
                for t in 0..topk {
                    let ex = ids[b * topk + t] as usize;
                    for j in 0..n {
                        let mut acc = 0f32;
                        for i in 0..k {
                            acc += w_deq[ex * n * k + j * k + i] * inp[b * k + i];
                        }
                        max_abs = max_abs.max(acc.abs());
                        err_g = err_g.max((got_g[(b * topk + t) * n + j] - acc).abs());
                    }
                }
            }
            let tol = 0.05 * max_abs + 1e-3;
            assert!(
                err_g <= tol,
                "{dtype:?}: GROUPED prefill vs oracle err {err_g} > tol {tol} (max_abs {max_abs})"
            );

            // PER-SLOT decode (batch==1) on the same weights -> still tracks the oracle.
            let input1 = CudaStorage::wrap_cuda_slice(dev.clone_htod(&inp1)?, dev.clone());
            let input1_l = crate::Layout::contiguous((1usize, 1usize, k));
            let ids1_storage = CudaStorage::wrap_cuda_slice(dev.clone_htod(&ids1)?, dev.clone());
            let ids1_l = crate::Layout::contiguous((1usize, topk));
            let (p_st, _) =
                bank.indexed_moe_forward(&self_shape, &input1, &input1_l, &ids1_storage, &ids1_l)?;
            let got_p = dev.clone_dtoh(&p_st.as_cuda_slice::<f32>()?.as_view())?;
            let mut err_p = 0f32;
            for t in 0..topk {
                let ex = ids1[t] as usize;
                for j in 0..n {
                    let mut acc = 0f32;
                    for i in 0..k {
                        acc += w_deq[ex * n * k + j * k + i] * inp1[i];
                    }
                    err_p = err_p.max((got_p[t * n + j] - acc).abs());
                }
            }
            assert!(
                err_p <= tol,
                "{dtype:?}: PER-SLOT decode vs oracle err {err_p} > tol {tol}"
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_pq2_ptq1_128blk_dp4a_dequant_and_get_rows() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let cuda_dev = crate::Device::Cuda(dev.clone());
        // k = 384 = 3 * 128 (deliberately NOT a multiple of 256!) verifies 128-element super-block sizing.
        let (n, k) = (6usize, 384usize);

        for dtype in [
            GgmlDType::PQ2_0,
            GgmlDType::PTQ1_0,
            GgmlDType::Q1_0,
            GgmlDType::Q2_0,
        ] {
            let nbytes = (n * k / dtype.block_size()) * dtype.type_size();
            let bytes: Vec<u8> = (0..nbytes)
                .map(|i| (i.wrapping_mul(41).wrapping_add(7) & 0x3F) as u8)
                .collect();

            // CPU reference dequantization.
            let qcpu = dtype.from_data(std::borrow::Cow::Borrowed(&bytes));
            let crate::CpuStorage::F32(w_cpu) = qcpu.dequantize(n * k)? else {
                unreachable!()
            };

            let qcuda = match QStorage::from_data(std::borrow::Cow::Owned(bytes), &cuda_dev, dtype)?
            {
                QStorage::Cuda(s) => s,
                _ => unreachable!(),
            };

            // 1. Native GPU dequantize_f32 must match CPU to_float bit-for-bit.
            let w_gpu_s = qcuda.dequantize(n * k)?;
            let w_gpu = dev.clone_dtoh(&w_gpu_s.as_cuda_slice::<f32>()?.as_view())?;
            for idx in 0..n * k {
                assert_eq!(
                    w_gpu[idx].to_bits(),
                    w_cpu[idx].to_bits(),
                    "{dtype:?} GPU dequantize_f32 mismatch at {idx}: gpu={} cpu={}",
                    w_gpu[idx],
                    w_cpu[idx]
                );
            }

            // 2. Native GPU dequantize_f16 must match f16::from_f32(w_cpu).
            let w_f16_s = qcuda.dequantize_f16(n * k)?;
            let w_f16 = dev.clone_dtoh(&w_f16_s.as_cuda_slice::<f16>()?.as_view())?;
            for idx in 0..n * k {
                let diff = (w_f16[idx].to_f32() - w_cpu[idx]).abs();
                assert!(
                    diff <= 1e-3 * w_cpu[idx].abs() + 1e-5,
                    "{dtype:?} f16 diff {diff}"
                );
            }

            // 3. Native GPU get_rows (embedding) must match CPU rows bit-for-bit.
            let ids_vec: Vec<u32> = vec![5, 0, 3];
            let ids_cuda = CudaStorage::wrap_cuda_slice(dev.clone_htod(&ids_vec)?, dev.clone());
            let ids_l = crate::Layout::contiguous(ids_vec.len());
            let emb_s = qcuda.embedding(n, k, &ids_cuda, &ids_l)?;
            let emb = dev.clone_dtoh(&emb_s.as_cuda_slice::<f32>()?.as_view())?;
            for (ri, &row_id) in ids_vec.iter().enumerate() {
                for c in 0..k {
                    assert_eq!(
                        emb[ri * k + c].to_bits(),
                        w_cpu[row_id as usize * k + c].to_bits(),
                        "{dtype:?} get_rows mismatch at row {row_id} col {c}"
                    );
                }
            }

            // 4. Decode (m=1, qmatvec_dp4a), batched verification (m=2..32, qmatmul_dp4a), and
            //    prefill (m=128, BF16 Tensor Core GEMM via dequantize_matmul_dense) vs oracle,
            //    plus bit-for-bit agreement between m=1 single-token GEMV and m=16 batched GEMM on every row!
            for m in [1usize, 2, 8, 12, 16, 32, 128] {
                let inp: Vec<f32> = (0..m * k)
                    .map(|i| ((i * 13 + 5) % 29) as f32 * 0.03 - 0.4)
                    .collect();
                let input = CudaStorage::wrap_cuda_slice(dev.clone_htod(&inp)?, dev.clone());
                let input_l = crate::Layout::contiguous((m, k));
                let self_shape: crate::Shape = (n, k).into();
                let (out, out_shape) = qcuda.fwd(&self_shape, &input, &input_l)?;
                assert_eq!(out_shape.dims().to_vec(), vec![m, n]);
                let got = dev.clone_dtoh(&out.as_cuda_slice::<f32>()?.as_view())?;
                let mut max_abs = 0f32;
                let mut max_err = 0f32;
                for r in 0..m {
                    for j in 0..n {
                        let mut acc = 0f32;
                        for c in 0..k {
                            acc += w_cpu[j * k + c] * inp[r * k + c];
                        }
                        max_abs = max_abs.max(acc.abs());
                        max_err = max_err.max((got[r * n + j] - acc).abs());
                    }
                }
                let tol = 0.03 * max_abs + 1e-3;
                assert!(
                    max_err <= tol,
                    "{dtype:?} m={m}: max_err {max_err} > tol {tol} (max_abs {max_abs})"
                );

                if m == 16 {
                    // Verify bit-for-bit parity between batched qmatmul_dp4a (m=16) and single-token qmatvec_dp4a (m=1).
                    for r in 0..m {
                        let row_inp = &inp[r * k..(r + 1) * k];
                        let row_cuda =
                            CudaStorage::wrap_cuda_slice(dev.clone_htod(row_inp)?, dev.clone());
                        let row_l = crate::Layout::contiguous((1usize, k));
                        let (row_out, _) = qcuda.fwd(&self_shape, &row_cuda, &row_l)?;
                        let row_got = dev.clone_dtoh(&row_out.as_cuda_slice::<f32>()?.as_view())?;
                        for j in 0..n {
                            assert_eq!(
                                got[r * n + j].to_bits(),
                                row_got[j].to_bits(),
                                "{dtype:?} batched qmatmul_dp4a vs qmatvec_dp4a mismatch at row {r} col {j}: {} vs {}",
                                got[r * n + j],
                                row_got[j]
                            );
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn cuda_hadamard_fwht_matches_dense_hb() -> Result<()> {
        let cuda_dev = crate::Device::new_cuda(0)?;
        let b = 512usize;
        let width = 2560usize; // 5 * 512 (Bonsai-2 4B hidden_size)
        let rows = 6usize;

        let scale = 1.0f32 / (b as f32).sqrt();
        let mut h_data = vec![0.0f32; b * b];
        for i in 0..b {
            for j in 0..b {
                h_data[i * b + j] = if (i & j).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                };
            }
        }
        let h_b = crate::Tensor::from_vec(h_data, (b, b), &cuda_dev)?;
        let signs_vec: Vec<f32> = (0..width)
            .map(|i| if (i * 7 + 3) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let signs = crate::Tensor::from_vec(signs_vec, width, &cuda_dev)?;

        let x_vec: Vec<f32> = (0..rows * width)
            .map(|i| ((i * 19 + 11) % 97) as f32 * 0.02 - 0.9)
            .collect();
        let x_f32 = crate::Tensor::from_vec(x_vec, (rows, width), &cuda_dev)?;

        // Forward F32 with signs
        let ref_fwd = x_f32
            .broadcast_mul(&signs)?
            .reshape((rows * (width / b), b))?
            .matmul(&h_b)?
            .reshape((rows, width))?;
        let got_fwd = hadamard_fwht(&x_f32, Some(&signs), b, false)?;
        let ref_v = ref_fwd.flatten_all()?.to_vec1::<f32>()?;
        let got_v = got_fwd.flatten_all()?.to_vec1::<f32>()?;
        for i in 0..ref_v.len() {
            let diff = (ref_v[i] - got_v[i]).abs();
            assert!(
                diff < 1e-4,
                "fwht fwd f32 mismatch at {i}: {} vs {}",
                ref_v[i],
                got_v[i]
            );
        }

        // Inverse BF16 with signs: T^{-1}(T(x)) == x
        let x_bf16 = x_f32.to_dtype(crate::DType::BF16)?;
        let y_bf16 = hadamard_fwht(&x_bf16, Some(&signs), b, false)?;
        let rec_bf16 = hadamard_fwht(&y_bf16, Some(&signs), b, true)?;
        let x_bf16_v = x_bf16
            .to_dtype(crate::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rec_v = rec_bf16
            .to_dtype(crate::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for i in 0..x_bf16_v.len() {
            let diff = (x_bf16_v[i] - rec_v[i]).abs();
            assert!(
                diff < 2e-2,
                "fwht roundtrip bf16 mismatch at {i}: {} vs {}",
                x_bf16_v[i],
                rec_v[i]
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_hadamard_fwht_4096_matches_dense_hb() -> Result<()> {
        let cuda_dev = crate::Device::new_cuda(0)?;
        let b = 4096usize;
        let width = 4096usize; // Bonsai-2 9B prism.hadamard.block_size
        let rows = 3usize;

        let scale = 1.0f32 / (b as f32).sqrt();
        let mut h_data = vec![0.0f32; b * b];
        for i in 0..b {
            for j in 0..b {
                h_data[i * b + j] = if (i & j).count_ones() % 2 == 0 {
                    scale
                } else {
                    -scale
                };
            }
        }
        let h_b = crate::Tensor::from_vec(h_data, (b, b), &cuda_dev)?;
        let signs_vec: Vec<f32> = (0..width)
            .map(|i| if (i * 7 + 3) % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let signs = crate::Tensor::from_vec(signs_vec, width, &cuda_dev)?;

        let x_vec: Vec<f32> = (0..rows * width)
            .map(|i| ((i * 19 + 11) % 97) as f32 * 0.02 - 0.9)
            .collect();
        let x_f32 = crate::Tensor::from_vec(x_vec, (rows, width), &cuda_dev)?;

        let ref_fwd = x_f32.broadcast_mul(&signs)?.matmul(&h_b)?;
        let got_fwd = hadamard_fwht(&x_f32, Some(&signs), b, false)?;
        let ref_v = ref_fwd.flatten_all()?.to_vec1::<f32>()?;
        let got_v = got_fwd.flatten_all()?.to_vec1::<f32>()?;
        for i in 0..ref_v.len() {
            let diff = (ref_v[i] - got_v[i]).abs();
            assert!(
                diff < 1e-4,
                "fwht 4096 fwd f32 mismatch at {i}: {} vs {}",
                ref_v[i],
                got_v[i]
            );
        }

        let x_bf16 = x_f32.to_dtype(crate::DType::BF16)?;
        let y_bf16 = hadamard_fwht(&x_bf16, Some(&signs), b, false)?;
        let rec_bf16 = hadamard_fwht(&y_bf16, Some(&signs), b, true)?;
        let x_bf16_v = x_bf16
            .to_dtype(crate::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let rec_v = rec_bf16
            .to_dtype(crate::DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        for i in 0..x_bf16_v.len() {
            let diff = (x_bf16_v[i] - rec_v[i]).abs();
            assert!(
                diff < 3e-2,
                "fwht 4096 roundtrip bf16 mismatch at {i}: {} vs {}",
                x_bf16_v[i],
                rec_v[i]
            );
        }
        Ok(())
    }

    #[test]
    fn cuda_pq2_ptq1_mmq_gemm_matches_reference() -> Result<()> {
        use crate::backend::BackendStorage;
        let dev = CudaDevice::new(0)?;
        let cuda_dev = crate::Device::Cuda(dev.clone());

        let shapes_and_ms: &[(usize, usize, &[usize])] = &[
            (256, 256, &[9, 16, 64]),
            // nrows = 192 tests need_check = true (not divisible by mmq_y = 128)
            (192, 512, &[9, 16, 64, 512]),
            (256, 5120, &[9, 21, 64]),
            // 27B FFN projection dimensions (> 64M weights)
            (17408, 5120, &[16, 64]),
            (5120, 17408, &[16, 64]),
        ];

        for dtype in [
            GgmlDType::PQ2_0,
            GgmlDType::PTQ1_0,
            GgmlDType::Q1_0,
            GgmlDType::Q2_0,
        ] {
            for &(n, k, ms) in shapes_and_ms {
                let nbytes = (n * k / dtype.block_size()) * dtype.type_size();
                // Deterministic pseudo-random block bytes; keep valid f16 scales (msb exponents near 1.0).
                let mut bytes: Vec<u8> = (0..nbytes)
                    .map(|i| (i.wrapping_mul(131).wrapping_add(17) & 0xFF) as u8)
                    .collect();
                let blk_bytes = dtype.type_size();
                let d_off = if dtype == GgmlDType::PTQ1_0 { 26 } else { 0 };
                let scale_bits = f16::from_f32(0.0625).to_bits().to_le_bytes();
                for b in 0..(n * k / dtype.block_size()) {
                    let off = b * blk_bytes + d_off;
                    bytes[off] = scale_bits[0];
                    bytes[off + 1] = scale_bits[1];
                }

                let qcuda =
                    match QStorage::from_data(std::borrow::Cow::Owned(bytes), &cuda_dev, dtype)? {
                        QStorage::Cuda(s) => s,
                        _ => unreachable!(),
                    };
                let self_shape: crate::Shape = (n, k).into();
                // GPU dequantized f32 reference weight matrix [k, n] stored as [n, k] transposed
                let w_f32 = qcuda.dequantize(n * k)?;
                let rhs_ref_l =
                    crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((1, k, n))?;

                for &m in ms {
                    let inp: Vec<f32> = (0..m * k)
                        .map(|i| ((i.wrapping_mul(29).wrapping_add(3)) % 47) as f32 * 0.04 - 0.9)
                        .collect();
                    let inp_f32 = CudaStorage::wrap_cuda_slice(dev.clone_htod(&inp)?, dev.clone());
                    let inp_l = crate::Layout::contiguous((m, k));

                    // Reference in F32 using GPU dense matmul over CPU-exact dequantized weights
                    let ref_s = inp_f32.matmul(
                        &w_f32,
                        (1, m, n, k),
                        &crate::Layout::contiguous((1, m, k)),
                        &rhs_ref_l,
                    )?;
                    let ref_v = dev.clone_dtoh(&ref_s.as_cuda_slice::<f32>()?.as_view())?;

                    // Test F32, BF16 (only smaller shapes to keep test runtime fast), and F16
                    let test_dtypes: &[crate::DType] = if n <= 512 && k <= 1024 && m == 16 {
                        &[crate::DType::F32, crate::DType::BF16, crate::DType::F16]
                    } else {
                        &[crate::DType::F32]
                    };

                    for &in_dt in test_dtypes {
                        let (in_st, dt_tol) = if in_dt == crate::DType::F32 {
                            (inp_f32.try_clone(&inp_l)?, 0.03f32)
                        } else {
                            (inp_f32.to_dtype(&inp_l, in_dt)?, 0.05f32)
                        };

                        // Verify fast_mmq::try_fwd explicitly triggers (returns Some) AND matches reference.
                        let (mmq_out, mmq_shape) = crate::quantized::fast_mmq::try_fwd(
                            &qcuda,
                            &self_shape,
                            &in_st,
                            &inp_l,
                        )?
                        .unwrap_or_else(|| {
                            panic!(
                                "{dtype:?} fast_mmq::try_fwd returned None for m={m}, n={n}, k={k}"
                            )
                        });
                        assert_eq!(mmq_shape.dims().to_vec(), vec![m, n]);
                        let mmq_f32 = if in_dt == crate::DType::F32 {
                            mmq_out
                        } else {
                            mmq_out.to_dtype(
                                &crate::Layout::contiguous(&mmq_shape),
                                crate::DType::F32,
                            )?
                        };
                        let got = dev.clone_dtoh(&mmq_f32.as_cuda_slice::<f32>()?.as_view())?;

                        // Also verify qcuda.fwd takes the MMQ path when m > 8.
                        let (fwd_out, _) = qcuda.fwd(&self_shape, &in_st, &inp_l)?;
                        let fwd_f32 = if in_dt == crate::DType::F32 {
                            fwd_out
                        } else {
                            fwd_out.to_dtype(
                                &crate::Layout::contiguous(&mmq_shape),
                                crate::DType::F32,
                            )?
                        };
                        let got_fwd = dev.clone_dtoh(&fwd_f32.as_cuda_slice::<f32>()?.as_view())?;

                        let mut max_abs = 0f32;
                        let mut max_err = 0f32;
                        let mut max_fwd_diff = 0f32;
                        for idx in 0..m * n {
                            max_abs = max_abs.max(ref_v[idx].abs());
                            max_err = max_err.max((got[idx] - ref_v[idx]).abs());
                            max_fwd_diff = max_fwd_diff.max((got[idx] - got_fwd[idx]).abs());
                        }
                        let tol = dt_tol * max_abs + 2e-2;
                        assert!(
                            max_err <= tol,
                            "{dtype:?} MMQ {in_dt:?} m={m} n={n} k={k}: max_err={max_err} > tol={tol} (max_abs={max_abs})"
                        );
                        assert!(
                            max_fwd_diff <= 1e-5,
                            "{dtype:?} qcuda.fwd diverged from fast_mmq::try_fwd: {max_fwd_diff}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[test]
    fn cuda_pq2_ptq1_mmq_moe_matches_reference() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let cuda_dev = crate::Device::Cuda(dev.clone());
        let (experts, n, k, batch, topk) = (4usize, 192usize, 512usize, 16usize, 2usize);

        for dtype in [
            GgmlDType::PQ2_0,
            GgmlDType::PTQ1_0,
            GgmlDType::Q1_0,
            GgmlDType::Q2_0,
        ] {
            let blk_bytes = dtype.type_size();
            let total_blks = experts * n * k / dtype.block_size();
            let mut bytes: Vec<u8> = (0..total_blks * blk_bytes)
                .map(|i| (i.wrapping_mul(89).wrapping_add(11) & 0xFF) as u8)
                .collect();
            let d_off = if dtype == GgmlDType::PTQ1_0 { 26 } else { 0 };
            let scale_bits = f16::from_f32(0.05).to_bits().to_le_bytes();
            for b in 0..total_blks {
                let off = b * blk_bytes + d_off;
                bytes[off] = scale_bits[0];
                bytes[off + 1] = scale_bits[1];
            }

            let qcuda = match QStorage::from_data(std::borrow::Cow::Owned(bytes), &cuda_dev, dtype)?
            {
                QStorage::Cuda(s) => s,
                _ => unreachable!(),
            };
            let w_shape: crate::Shape = (experts, n, k).into();
            let inp: Vec<f32> = (0..batch * k)
                .map(|i| ((i * 17 + 9) % 41) as f32 * 0.04 - 0.8)
                .collect();
            let inp_slice = dev.clone_htod(&inp)?;
            let in_shape: crate::Shape = (batch, 1usize, k).into();
            let ids: Vec<u32> = (0..batch * topk)
                .map(|i| ((i * 3 + 1) % experts) as u32)
                .collect();
            let ids_slice = dev.clone_htod(&ids)?;
            let idx_shape: crate::Shape = (batch, topk).into();

            let (out, out_shape) = qcuda
                .moe_iquant_qmmq(
                    &w_shape,
                    &inp_slice,
                    &in_shape,
                    &ids_slice.as_view(),
                    &idx_shape,
                )?
                .expect("PQ2_0/PTQ1_0 moe_iquant_qmmq returned None");
            assert_eq!(out_shape.dims().to_vec(), vec![batch, topk, n]);
            let got = dev.clone_dtoh(&out.as_cuda_slice::<f32>()?.as_view())?;

            // Oracle via dequantize
            let w_f32_s = qcuda.dequantize(experts * n * k)?;
            let w_f32 = dev.clone_dtoh(&w_f32_s.as_cuda_slice::<f32>()?.as_view())?;
            let mut max_abs = 0f32;
            let mut max_err = 0f32;
            for b_idx in 0..batch {
                for slot in 0..topk {
                    let e = ids[b_idx * topk + slot] as usize;
                    for j in 0..n {
                        let mut acc = 0f32;
                        for i in 0..k {
                            acc += w_f32[(e * n + j) * k + i] * inp[b_idx * k + i];
                        }
                        let g = got[(b_idx * topk + slot) * n + j];
                        max_abs = max_abs.max(acc.abs());
                        max_err = max_err.max((g - acc).abs());
                    }
                }
            }
            let tol = 0.03 * max_abs + 2e-2;
            assert!(
                max_err <= tol,
                "{dtype:?} MoE MMQ max_err={max_err} > tol={tol}"
            );
        }
        Ok(())
    }

    #[test]
    #[ignore]
    fn bench_pq2_ptq1_mmq_crossover() -> Result<()> {
        let dev = CudaDevice::new(0)?;
        let cuda_dev = crate::Device::Cuda(dev.clone());
        for (n, k) in [
            (5120usize, 5120usize),
            (17408usize, 5120usize),
            (5120usize, 17408usize),
        ] {
            for dtype in [
            GgmlDType::PQ2_0,
            GgmlDType::PTQ1_0,
            GgmlDType::Q1_0,
            GgmlDType::Q2_0,
        ] {
                let nbytes = (n * k / dtype.block_size()) * dtype.type_size();
                let bytes = vec![0x55u8; nbytes];
                let qcuda =
                    match QStorage::from_data(std::borrow::Cow::Owned(bytes), &cuda_dev, dtype)? {
                        QStorage::Cuda(s) => s,
                        _ => unreachable!(),
                    };
                let self_shape: crate::Shape = (n, k).into();
                for m in [2usize, 4, 8, 9, 16, 21, 64, 512] {
                    let inp = vec![0.1f32; m * k];
                    let in_st = CudaStorage::wrap_cuda_slice(dev.clone_htod(&inp)?, dev.clone());
                    let in_l = crate::Layout::contiguous((m, k));
                    // warmup
                    let _ = qcuda.mul_mat_vec_iquant(&self_shape, &in_st, &in_l)?;
                    let _ =
                        crate::quantized::fast_mmq::try_fwd(&qcuda, &self_shape, &in_st, &in_l)?;
                    crate::backend::BackendDevice::synchronize(&dev)?;

                    let iters = if m <= 21 { 20 } else { 10 };
                    let t0 = std::time::Instant::now();
                    for _ in 0..iters {
                        let _ = qcuda.mul_mat_vec_iquant(&self_shape, &in_st, &in_l)?;
                    }
                    crate::backend::BackendDevice::synchronize(&dev)?;
                    let dp4a_us = t0.elapsed().as_micros() as f64 / iters as f64;

                    let t1 = std::time::Instant::now();
                    for _ in 0..iters {
                        let _ = crate::quantized::fast_mmq::try_fwd(
                            &qcuda,
                            &self_shape,
                            &in_st,
                            &in_l,
                        )?;
                    }
                    crate::backend::BackendDevice::synchronize(&dev)?;
                    let mmq_us = t1.elapsed().as_micros() as f64 / iters as f64;

                    eprintln!(
                        "CROSSOVER {dtype:?} n={n} k={k} m={m}: dp4a={dp4a_us:.1}us mmq={mmq_us:.1}us ratio(dp4a/mmq)={:.2}x",
                        dp4a_us / mmq_us
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn mmq_max_vec_bm_profiles_by_compute_cap() {
        // Only check defaults when env overrides are unset.
        if std::env::var_os("HANZO_PQ2_MMQ_MAX_BM").is_none() {
            assert_eq!(mmq_max_vec_bm(GgmlDType::PQ2_0, 1210), 1);
            assert_eq!(mmq_max_vec_bm(GgmlDType::PQ2_0, 1000), 1);
            assert_eq!(mmq_max_vec_bm(GgmlDType::PQ2_0, 900), 8);
            assert_eq!(mmq_max_vec_bm(GgmlDType::PQ2_0, 890), 8);
        }
        if std::env::var_os("HANZO_PTQ1_MMQ_MAX_BM").is_none() {
            assert_eq!(mmq_max_vec_bm(GgmlDType::PTQ1_0, 1210), 3);
            assert_eq!(mmq_max_vec_bm(GgmlDType::PTQ1_0, 900), 8);
        }
        assert_eq!(mmq_max_vec_bm(GgmlDType::Q4_0, 1210), 8);
    }
}
