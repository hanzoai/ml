use super::{GgmlDType, QStorage};
use crate::quantized::k_quants::GgmlType;
use crate::{backend::BackendDevice, cuda_backend::WrapErr};
use crate::{builder_arg as barg, CudaDevice, CudaStorage, Result};
use half::f16;

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
// HANZO_CUDA_FAST_MMQ=1. When on, QCudaStorage::fwd tries it first and falls back to the legacy
// on-GPU q8_1 kernels for any dtype/shape it doesn't support.
pub(crate) fn fast_mmq_enabled() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        matches!(
            std::env::var("HANZO_CUDA_FAST_MMQ").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    })
}

// Force i-quant codebook weights back to the dequantize-to-f32 dense matmul instead of the native
// dp4a decode -- the A/B knob (mirrors ROCm's HANZO_IQ*_FALLBACK) and a production safety fallback.
// Read once. Set HANZO_IQ_DEQUANT_FALLBACK=1 to disable the native i-quant mmvq decode path.
pub(crate) fn iq_dequant_fallback() -> bool {
    use std::sync::OnceLock;
    static EN: OnceLock<bool> = OnceLock::new();
    *EN.get_or_init(|| {
        matches!(
            std::env::var("HANZO_IQ_DEQUANT_FALLBACK").as_deref(),
            Ok("1") | Ok("true") | Ok("TRUE")
        )
    })
}

pub const WARP_SIZE: usize = 32;
pub const MMQ_X_Q4_0_AMPERE: usize = 4;
pub const MMQ_Y_Q4_0_AMPERE: usize = 32;
pub const NWARPS_Q4_0_AMPERE: usize = 4;
pub const GGML_CUDA_MMV_X: usize = 32;
pub const GGML_CUDA_MMV_Y: usize = 1;
pub const CUDA_QUANTIZE_BLOCK_SIZE: usize = 256;
pub const CUDA_DEQUANTIZE_BLOCK_SIZE: usize = 256;
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
        _ => return None,
    })
}

/// Native i-quant mmvq DECODE: codebook-decode the weight to int8 grid magnitudes (+sign, or
/// pre-signed for IQ1) and dp4a against the once-q8_1-quantized activation -- NO dequant->f32
/// round-trip. The SotA decode twin of the ROCm `qdp4a<IQ*>` path (bit-exact, 2.27x). The activation
/// `y` arrives f32; it is quantized once to the separated (int8 `xq`, f16 scale `xd`) layout the
/// `qmatvec_dp4a_core` consumes, then each of the (small) `b_size` activation rows is decoded.
#[allow(clippy::too_many_arguments)]
fn mul_mat_vec_iquant_dp4a(
    data: &PaddedCudaSlice,
    y: &CudaView<f32>,
    dtype: GgmlDType,
    ncols: usize,
    nrows: usize,
    b_size: usize,
    dev: &CudaDevice,
) -> Result<CudaStorage> {
    let suffix = iquant_dp4a_suffix(dtype)
        .ok_or_else(|| crate::Error::Msg(format!("no i-quant dp4a kernel for {dtype:?}")).bt())?;
    // i-quant super-block is 256 weights (8 q8_1 sub-blocks of 32) -- ncols is always a multiple, so
    // the activation needs no MATRIX_ROW_PADDING (the dp4a core reads exactly ncols int8 + ncols/32 f16).
    if ncols % 256 != 0 {
        crate::bail!("i-quant ncols {ncols} must be a multiple of 256");
    }
    if b_size == 0 || b_size > 8 {
        crate::bail!("only bsize between 1 and 8 are supported, got {b_size}");
    }
    if y.len() != ncols * b_size {
        crate::bail!("unexpected y size {}, ncols {ncols} bsize {b_size}", y.len());
    }
    let nblk32 = ncols / 32;

    // Quantize the f32 activation -> separated (int8 `xq` [b_size, ncols], f16 scale `xd`
    // [b_size, ncols/32]). One warp per 32-block; guard `wid >= b_size*nblk32` in-kernel.
    let mut xq = dev.alloc_zeros::<u8>(b_size * ncols)?;
    let mut xd = dev.alloc_zeros::<f16>(b_size * nblk32)?;
    {
        let func = dev.get_or_load_func("iq_quantize_q8_f32", &hanzo_kernels::IQUANT_MMVQ)?;
        let nwarp = (b_size * nblk32) as u32;
        let threads = 256u32; // 8 warps/block
        let blocks = nwarp.div_ceil(threads / WARP_SIZE as u32);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        barg!(builder, b_size as i32, ncols as i32);
        builder.arg(y);
        builder.arg(&xq);
        builder.arg(&xd);
        unsafe { builder.launch(cfg) }.w()?;
    }

    // Decode matvec: warp-per-row, lane-strided over (super-block x sub-unit). One launch per
    // activation row (b_size is tiny on the decode/vec path).
    let kernel_name = format!("qmatvec_dp4a_{suffix}_f32");
    let func = dev.get_or_load_func(&kernel_name, &hanzo_kernels::IQUANT_MMVQ)?;
    let dst = dev.alloc_zeros::<f32>(nrows * b_size)?;
    const ROWS_PER_BLOCK: u32 = 8; // 8 warps/block
    for bi in 0..b_size {
        let xq_b = xq.slice(bi * ncols..(bi + 1) * ncols);
        let xd_b = xd.slice(bi * nblk32..(bi + 1) * nblk32);
        let dst_b = dst.slice(bi * nrows..(bi + 1) * nrows);
        let cfg = cudarc::driver::LaunchConfig {
            grid_dim: ((nrows as u32).div_ceil(ROWS_PER_BLOCK), 1, 1),
            block_dim: (ROWS_PER_BLOCK * WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut builder = func.builder();
        barg!(builder, nrows as i32, ncols as i32);
        builder.arg(&data.inner);
        builder.arg(&xq_b);
        builder.arg(&xd_b);
        builder.arg(&dst_b);
        unsafe { builder.launch(cfg) }.w()?;
    }
    Ok(CudaStorage::wrap_cuda_slice(dst, dev.clone()))
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
        n_slots % batch == 0,
        "ids count {n_slots} not a multiple of batch {batch}"
    );
    let topk = n_slots / batch;

    // Prefill (rows>1) -> expert-grouped int8 MMQ GEMM (llama mul_mat_id): group the routed tokens by
    // expert and stage each expert's weight ONCE, amortizing it over all its tokens via the tensor
    // cores. The per-slot matvec below instead re-streams the whole expert weight for EVERY routed token
    // (no matrix cores) -- fine at decode (one token), the 10x deficit at prefill. Decode (batch==1)
    // keeps the per-slot matvec (bandwidth-bound, capture-clean). HANZO_MOE_QMMQ_FALLBACK forces the
    // per-slot path for the A/B; unsupported weight dtypes return None and fall through to per-slot.
    if batch > 1 && std::env::var("HANZO_MOE_QMMQ_FALLBACK").is_err() {
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
        let suffix = iquant_dp4a_suffix(self.dtype)
            .ok_or_else(|| crate::Error::Msg(format!("no i-quant MoE kernel for {:?}", self.dtype)).bt())?;
        if k % 256 != 0 {
            crate::bail!("i-quant MoE ncols {k} must be a multiple of 256");
        }
        if x_flat.len() != nrows * k {
            crate::bail!("unexpected x_flat size {}, nrows {nrows} k {k}", x_flat.len());
        }
        let dev = self.device();
        let nblk32 = k / 32;

        // Quantize the nrows routed activations -> separated (int8 xq [nrows,k], f16 scale xd [nrows,k/32]).
        let mut xq = dev.alloc_zeros::<u8>(nrows * k)?;
        let mut xd = dev.alloc_zeros::<f16>(nrows * nblk32)?;
        {
            let func = dev.get_or_load_func("iq_quantize_q8_f32", &hanzo_kernels::IQUANT_MMVQ)?;
            let nwarp = (nrows * nblk32) as u32;
            let threads = 256u32;
            let blocks = nwarp.div_ceil(threads / WARP_SIZE as u32);
            let cfg = cudarc::driver::LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut builder = func.builder();
            barg!(builder, nrows as i32, k as i32);
            builder.arg(x_flat);
            builder.arg(&xq);
            builder.arg(&xd);
            unsafe { builder.launch(cfg) }.w()?;
        }

        // Fused decode: one batched launch, slots on grid.y, expert gathered on-device from `ids`.
        let kernel_name = format!("moe_qmatvec_dp4a_{suffix}_f32");
        let func = dev.get_or_load_func(&kernel_name, &hanzo_kernels::IQUANT_MMVQ)?;
        let out = dev.alloc_zeros::<f32>(nrows * n)?;
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

    pub fn fwd(
        &self,
        self_shape: &crate::Shape,
        storage: &CudaStorage,
        layout: &crate::Layout,
    ) -> Result<(CudaStorage, crate::Shape)> {
        // Fallback
        let max_bm = if FORCE_DMMV.load(std::sync::atomic::Ordering::Relaxed) {
            1
        } else {
            8
        };
        let use_vec_kernel = match layout.shape().dims() {
            [b, m, _k] => b * m <= max_bm,
            [b, _k] => *b <= max_bm,
            _ => false,
        };
        // Modern llama mmq/mmvq path (int8 MMA matrix-core + stream-K). Opt-in via
        // HANZO_CUDA_FAST_MMQ=1; try_fwd returns Ok(None) for unsupported dtype/shape, falling
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
            if iquant_dp4a_suffix(self.dtype).is_some() && !iq_dequant_fallback() {
                if use_vec_kernel {
                    // DECODE: native dp4a mmvq.
                    return self.mul_mat_vec_iquant(self_shape, storage, layout);
                }
                // PREFILL: native int8-WMMA MMQ GEMM (qmmq) for the 7 i-quants with an MMQ kernel
                // (IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS, IQ1_S). Decode-only types (IQ1_M, TQ1_0/TQ2_0)
                // have no MMQ kernel -> try_fwd returns None and they fall to the dequant dense matmul.
                if let Some(out) = super::fast_mmq::try_fwd(self, self_shape, storage, layout)? {
                    return Ok(out);
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
        let rhs = rhs.as_cuda_slice::<f32>()?;
        let rhs = match rhs_l.contiguous_offsets() {
            Some((o1, o2)) => rhs.slice(o1..o2),
            None => Err(crate::Error::RequiresContiguous { op: "iq-mmvq" }.bt())?,
        };
        let (b_size, k) = match rhs_l.shape().dims() {
            [b, m, k] => (b * m, *k),
            [b, k] => (*b, *k),
            _ => crate::bail!("unexpected rhs shape in iq-mmvq {:?}", rhs_l.shape()),
        };
        if ncols != k {
            crate::bail!("mismatch on matmul dim {self_shape:?} {:?}", rhs_l.shape())
        }
        let out =
            mul_mat_vec_iquant_dp4a(&self.data, &rhs, self.dtype, ncols, nrows, b_size, self.device())?;
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
        let data_f32 = self.dequantize(n * k)?;
        let rhs_l = crate::Layout::new((k, n).into(), vec![1, k], 0).broadcast_as((b, k, n))?;
        let out = storage.matmul(&data_f32, (b, m, n, k), layout, &rhs_l)?;
        let mut out_shape = layout.shape().dims().to_vec();
        out_shape.pop();
        out_shape.push(n);
        Ok((out, out_shape.into()))
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

    // i-quant / ternary / NVFP4 weights have no native CUDA matmul kernel, so QStorage::from_data used to
    // BAIL when loading them onto CUDA -- an i-quant GGUF could not run on CUDA at all. They now upload
    // like any quant and QCudaStorage::fwd dequantizes-to-f32 for a dense matmul. This loads a
    // deterministic block pattern for EACH formerly-bailing type onto CUDA, runs fwd, and checks it
    // against the matmul of the SAME dequantized weights computed on the host -> proves each type LOADS
    // and DECODES (matmuls) coherently on CUDA, exact w.r.t. the dequantize reference.
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
            GgmlDType::IQ1_S,
            GgmlDType::IQ1_M,
            GgmlDType::TQ1_0,
            GgmlDType::TQ2_0,
            GgmlDType::NVFP4,
            GgmlDType::Q1_0,
        ] {
            // Bounded byte pattern (<= 0x3F) so every embedded f16 scale decodes finite (no NaN/Inf), yet
            // varied. Sizing comes from the dtype so it is correct for both 256- and 32-element blocks.
            let nbytes = (n * k / dtype.block_size()) * dtype.type_size();
            let bytes: Vec<u8> = (0..nbytes)
                .map(|i| (i.wrapping_mul(37).wrapping_add(11) & 0x3F) as u8)
                .collect();

            // LOAD onto CUDA (this is the call that used to bail for these dtypes).
            let qcuda = match QStorage::from_data(std::borrow::Cow::Owned(bytes), &cuda_dev, dtype)? {
                QStorage::Cuda(s) => s,
                _ => unreachable!("from_data on a CUDA device must yield CUDA storage"),
            };

            // Dequantized weights [n,k] (the exact values the dense matmul consumes) -> host oracle.
            let w_dev = qcuda.dequantize(n * k)?;
            let w_host = dev.clone_dtoh(&w_dev.as_cuda_slice::<f32>()?.as_view())?;

            // CUDA forward: input [m,k] @ weight^T via the dequant-dense fallback (no native kernel).
            let input = CudaStorage::wrap_cuda_slice(dev.clone_htod(&inp)?, dev.clone());
            let input_l = crate::Layout::contiguous((m, k));
            let self_shape: crate::Shape = (n, k).into();
            let (out, out_shape) = qcuda.fwd(&self_shape, &input, &input_l)?;
            assert_eq!(out_shape.dims().to_vec(), vec![m, n]);
            let got = dev.clone_dtoh(&out.as_cuda_slice::<f32>()?.as_view())?;

            // Host oracle with the SAME dequantized weights: out[r,j] = sum_i w[j,i] * inp[r,i].
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
            // Same weights + same input on GPU vs host -> only f32 accumulation-order noise remains.
            let tol = 1e-3 * max_abs + 1e-4;
            assert!(
                max_err <= tol,
                "{dtype:?}: i-quant CUDA dense matmul max_err {max_err} > tol {tol} (max_abs {max_abs})"
            );
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
            bank.quantize(&CudaStorage::wrap_cuda_slice(dev.clone_htod(&w)?, dev.clone()))?;
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
}
