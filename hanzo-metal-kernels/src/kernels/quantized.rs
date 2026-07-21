use crate::utils::EncoderProvider;
use crate::{
    set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

// Variant names mirror `hanzo_ml::quantized::GgmlDType` one-to-one (IQ2_XXS, IQ4_XS, ...) so the
// TryFrom mapping stays mechanical; the i-quant acronyms are not upper-camel-case.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy)]
pub enum GgmlDType {
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
    F16,
    F32,
    BF16,
    // i-quant codebook family. The MSL kernels (get_rows / mul_mv / mul_mm / mul_mv_id /
    // mul_mm_id) are all present in quantized.metal; these variants just route the dispatch.
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ3_XXS,
    IQ3_S,
    IQ1_S,
    IQ1_M,
    IQ4_NL,
    IQ4_XS,
}

/// Threadgroup scratch (bytes) the matvec kernel stages its codebook grid into. The IQ2/IQ3 kernels
/// copy their `__constant__` grid + sign table into threadgroup memory; IQ4_NL/IQ4_XS stage the
/// 16-value LUT. IQ1_S/IQ1_M pass `nullptr` (no staging) and every non-i-quant reduces in-simdgroup,
/// so they need none. 8192 is the ggml-metal upper bound (max real need is IQ2_XS at 512*8+128=4224)
/// and matches the flat scratch `call_mul_mv_id` binds; well within the 32 KiB threadgroup limit.
fn matvec_threadgroup_mem(dtype: GgmlDType) -> usize {
    match dtype {
        GgmlDType::IQ2_XXS
        | GgmlDType::IQ2_XS
        | GgmlDType::IQ2_S
        | GgmlDType::IQ3_XXS
        | GgmlDType::IQ3_S
        | GgmlDType::IQ4_NL
        | GgmlDType::IQ4_XS => 8192,
        _ => 0,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    dims: (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    dst_offset: usize,
    dst: &Buffer,
    src1_bf16: bool,
) -> Result<(), MetalKernelError> {
    call_quantized_matmul_mv_t_offset(
        device, ep, kernels, dtype, dims, lhs, lhs_offset, rhs, 0, dst_offset, dst, src1_bf16,
    )
}

/// As [`call_quantized_matmul_mv_t`], but the quantized weight (`rhs`) is read
/// starting `rhs_offset` bytes into its buffer -- the keep-quantized MoE path,
/// where each routed expert's `[n, k]` block sits at a byte offset inside one
/// packed `[E, n, k]` GGUF bank.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mv_t_offset(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    (b, m, n, k): (usize, usize, usize, usize),
    lhs: &Buffer,
    lhs_offset: usize,
    rhs: &Buffer,
    rhs_offset: usize,
    dst_offset: usize,
    dst: &Buffer,
    src1_bf16: bool,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = b as i64;
    let ne03 = 1i64;

    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = 0i64;

    let ne10 = k as i64;
    let ne11 = m as i64;
    let ne12 = b as i64;
    let ne13 = 1i64;

    let nb10 = 0i64;
    let nb11 = 0i64;
    let nb12 = 0i64;

    let ne0 = n as i64;
    let ne1 = m as i64;
    let r2: u32 = (ne12 / ne02) as u32;
    let r3: u32 = (ne13 / ne03) as u32;

    let (nth0, nth1, align) = match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => {
            let nth0 = 8;
            let nth1 = 8;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::Q2K => {
            // Fixing a bug in Metal for GGML
            // https://github.com/ggerganov/llama.cpp/blob/b8109bc0139f15a5b321909f47510b89dca47ffc/ggml-metal.m#L1576
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q4K => {
            // 2 simdgroups (64 threads) so kernel_mul_mv_q4_K runs ggml N_SG=2 x N_R0=2.
            let nth0 = 4;
            let nth1 = 16;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q3K | GgmlDType::Q5K => {
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::Q6K => {
            // 2 simdgroups x N_R0=2 rows/threadgroup (ggml N_SG_Q6_K=2, N_R0_Q6_K=2).
            let nth0 = 2;
            let nth1 = 32;
            let align = 4;
            (nth0, nth1, align)
        }
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => {
            // Original implem uses rows
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        GgmlDType::F32 => {
            let nth0 = 32;
            let nth1 = 1;
            let align = 8;
            (nth0, nth1, align)
        }
        // i-quant matvec kernels run 2 simdgroups (nth0*nth1 == 2*32 == 64 threads). IQ1/IQ2/IQ3
        // emit N_DST=4 rows/simdgroup -> 8 rows/threadgroup (align 8); IQ4_NL/IQ4_XS emit 2 rows
        // -> 4 rows/threadgroup (align 4). Matches ggml-metal's per-type dispatch.
        GgmlDType::IQ2_XXS
        | GgmlDType::IQ2_XS
        | GgmlDType::IQ2_S
        | GgmlDType::IQ3_XXS
        | GgmlDType::IQ3_S
        | GgmlDType::IQ1_S
        | GgmlDType::IQ1_M => (4, 16, 8),
        GgmlDType::IQ4_NL | GgmlDType::IQ4_XS => (4, 16, 4),
    };
    let thread_groups_count = MTLSize {
        width: divide(ne01 as usize, align),
        height: ne11 as usize,
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    // A bf16 activation (src1) + bf16 dst skips the bf16<->f32 round-trip GgufMatMul otherwise wraps
    // around the projection. Only Q4_K/Q6_K carry a bf16-native matvec (the two K-quants Q4_K_M uses);
    // any other dtype must reach here with an f32 activation.
    let name = if src1_bf16 {
        match dtype {
            GgmlDType::Q4K => "kernel_mul_mv_q4_K_bf16",
            GgmlDType::Q6K => "kernel_mul_mv_q6_K_bf16",
            other => {
                return Err(MetalKernelError::LoadFunctionError(format!(
                    "bf16 activation matvec unsupported for {other:?}"
                )))
            }
        }
    } else {
        match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_q8_0_f32",
        GgmlDType::Q8_1 => "kernel_mul_mv_q8_1_f32",
        GgmlDType::Q2K => "kernel_mul_mv_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_q6_K_f32",
        GgmlDType::Q8K => "kernel_mul_mv_q8_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mv_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mv_f32_f32",
        GgmlDType::IQ2_XXS => "kernel_mul_mv_iq2_xxs_f32",
        GgmlDType::IQ2_XS => "kernel_mul_mv_iq2_xs_f32",
        GgmlDType::IQ2_S => "kernel_mul_mv_iq2_s_f32",
        GgmlDType::IQ3_XXS => "kernel_mul_mv_iq3_xxs_f32",
        GgmlDType::IQ3_S => "kernel_mul_mv_iq3_s_f32",
        GgmlDType::IQ1_S => "kernel_mul_mv_iq1_s_f32",
        GgmlDType::IQ1_M => "kernel_mul_mv_iq1_m_f32",
        GgmlDType::IQ4_NL => "kernel_mul_mv_iq4_nl_f32",
        GgmlDType::IQ4_XS => "kernel_mul_mv_iq4_xs_f32",
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (rhs, rhs_offset),
            (lhs, lhs_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            r2,
            r3
        )
    );

    // The i-quant matvec kernels stage their codebook grid into threadgroup scratch (IQ1_S/IQ1_M
    // pass nullptr and need none). Non-i-quants reduce in-simdgroup and bind nothing, as before.
    let tg_mem = matvec_threadgroup_mem(dtype);
    if tg_mem > 0 {
        encoder.set_threadgroup_memory_length(0, tg_mem);
    }

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

/// - src0 is usually weight
/// - src1 is usually xs
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
    src1_bf16: bool,
    src1_half: bool,
) -> Result<(), MetalKernelError> {
    call_quantized_matmul_mm_t_offset(
        device,
        ep,
        kernels,
        dtype,
        src0_shape,
        src0_stride,
        src0,
        0,
        src1_shape,
        src1_stride,
        src1,
        src1_offset,
        dst_shape,
        dst_offset,
        dst,
        src1_bf16,
        src1_half,
    )
}

/// As [`call_quantized_matmul_mm_t`], but the quantized weight (`src0`) is read
/// starting `src0_offset` bytes into its buffer -- the keep-quantized MoE path.
#[allow(clippy::too_many_arguments)]
pub fn call_quantized_matmul_mm_t_offset(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    src0_shape: &[usize],
    src0_stride: &[usize],
    src0: &Buffer,
    src0_offset: usize,
    src1_shape: &[usize],
    src1_stride: &[usize],
    src1: &Buffer,
    src1_offset: usize,
    dst_shape: &[usize],
    dst_offset: usize,
    dst: &Buffer,
    src1_bf16: bool,
    src1_half: bool,
) -> Result<(), MetalKernelError> {
    // Everything is in reverse
    let ne00 = src0_shape[src0_shape.len() - 1] as i64;
    let ne01 = src0_shape[src0_shape.len() - 2] as i64;
    let ne02 = src0_shape[src0_shape.len() - 3] as i64;
    let ne03 = src0_shape[src0_shape.len() - 4] as i64;

    let nb01 = src0_stride[src0_stride.len() - 2] as i64;
    let nb02 = src0_stride[src0_stride.len() - 3] as i64;
    let nb03 = src0_stride[src0_stride.len() - 4] as i64;

    let ne11 = src1_shape[src1_shape.len() - 2] as i64;
    let ne12 = src1_shape[src1_shape.len() - 3] as i64;
    let ne13 = src1_shape[src1_shape.len() - 4] as i64;

    let nb10 = src1_stride[src1_stride.len() - 1] as i64;
    let nb11 = src1_stride[src1_stride.len() - 2] as i64;
    let nb12 = src1_stride[src1_stride.len() - 3] as i64;
    let nb13 = src1_stride[src1_stride.len() - 4] as i64;

    let ne0 = dst_shape[dst_shape.len() - 1] as i64;
    let ne1 = dst_shape[dst_shape.len() - 2] as i64;
    let r2 = (ne12 / ne02) as u32;
    let r3 = (ne13 / ne03) as u32;

    let thread_groups_count = MTLSize {
        width: divide(ne11 as usize, 32),
        height: divide(ne01 as usize, 64),
        depth: (ne12 * ne13) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    // bf16-native prefill: a bf16 activation (src1) + bf16 dst reads/writes bf16 directly, dropping
    // the bf16<->f32 round-trip GgufMatMul otherwise wraps around the projection. Only Q4_K/Q6_K
    // (the two K-quants a Q4_K_M model carries) have a bf16 mm; the simdgroup math is identical to
    // the _f32 kernel (src1 is widened to f32 on stage), so this is the mm twin of the bf16 matvec.
    // `src1_half` selects the fp16-matmul variant: the activation tile is staged as half so the outer
    // product runs at the GPU's half-precision simdgroup rate (~2x); the weight tile is half in both,
    // so this trades a small precision cost (half products) for the faster matmul. The caller (ml
    // `fwd`) gates it on an opt-in flag; the default bf16 path stays the f32-tile bit-exact kernel.
    let name = if src1_bf16 {
        match (dtype, src1_half) {
            (GgmlDType::Q4K, false) => "kernel_mul_mm_q4_K_bf16",
            (GgmlDType::Q6K, false) => "kernel_mul_mm_q6_K_bf16",
            (GgmlDType::Q4K, true) => "kernel_mul_mm_q4_K_bf16_half",
            (GgmlDType::Q6K, true) => "kernel_mul_mm_q6_K_bf16_half",
            _ => Err(MetalKernelError::UnsupportedDTypeForOp(
                "non-Q4K/Q6K",
                "bf16 qmatmul mm",
            ))?,
        }
    } else {
        match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_f16_f32",
        GgmlDType::BF16 => "kernel_mul_mm_bf16_f32",
        GgmlDType::F32 => "kernel_mul_mm_f32_f32",
        GgmlDType::IQ2_XXS => "kernel_mul_mm_iq2_xxs_f32",
        GgmlDType::IQ2_XS => "kernel_mul_mm_iq2_xs_f32",
        GgmlDType::IQ2_S => "kernel_mul_mm_iq2_s_f32",
        GgmlDType::IQ3_XXS => "kernel_mul_mm_iq3_xxs_f32",
        GgmlDType::IQ3_S => "kernel_mul_mm_iq3_s_f32",
        GgmlDType::IQ1_S => "kernel_mul_mm_iq1_s_f32",
        GgmlDType::IQ1_M => "kernel_mul_mm_iq1_m_f32",
        GgmlDType::IQ4_NL => "kernel_mul_mm_iq4_nl_f32",
        GgmlDType::IQ4_XS => "kernel_mul_mm_iq4_xs_f32",
        GgmlDType::Q8_1 => Err(MetalKernelError::UnsupportedDTypeForOp("Q8_1", "qmatmul"))?,
        GgmlDType::Q8K => Err(MetalKernelError::UnsupportedDTypeForOp("Q8K", "qmatmul"))?,
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (src0, src0_offset),
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
            ne00,
            ne02,
            nb01,
            nb02,
            nb03,
            ne12,
            nb10,
            nb11,
            nb12,
            nb13,
            ne0,
            ne1,
            r2,
            r3
        )
    );

    encoder.set_threadgroup_memory_length(0, 8192);

    encoder.dispatch_thread_groups(thread_groups_count, threads_per_threadgroup);
    Ok(())
}

fn divide(m: usize, b: usize) -> usize {
    m.div_ceil(b)
}

/// Fused MoE matvec — ggml `kernel_mul_mv_id`. One dispatch computes the routed expert matvec for
/// every (token, expert-slot) pair, reading the expert id per row from the on-device `ids` buffer:
/// no host round-trip, no per-expert loop. `src0s` is the resident `[n_experts, n, k]` quantized
/// bank (expert blocks `expert_bytes` apart); `src1` is `[t, s, k]` f32 (`s == 1` shared input for
/// gate/up, or `s == topk` per-slot for down); `ids` is `[t, topk]` u32/i32; `dst` is `[t, topk, n]`
/// f32. Mirrors the classic ggml-metal dispatch: same per-quant `(nth0, nth1, align)` table as the
/// plain `kernel_mul_mv_*`, with the expert/token pair carried on the grid's z axis.
#[allow(clippy::too_many_arguments)]
pub fn call_mul_mv_id(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    n_experts: usize,
    n: usize,
    k: usize,
    t: usize,
    topk: usize,
    s: usize,
    expert_bytes: usize,
    src0s: &Buffer,
    src0_offset: usize,
    src1: &Buffer,
    src1_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    dst: &Buffer,
    dst_offset: usize,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne01 = n as i64;
    let ne02 = n_experts as i64;
    let nb00 = 0i64;
    let nb01 = 0i64;
    let nb02 = expert_bytes as i64;

    let ne10 = k as i64;
    let ne11 = s as i64;
    let ne12 = t as i64;
    let ne13 = 1i64;
    let nb10 = std::mem::size_of::<f32>() as i64;
    let nb11 = (k * std::mem::size_of::<f32>()) as i64;
    let nb12 = (s * k * std::mem::size_of::<f32>()) as i64;

    let nei0 = topk as i64;
    let nei1 = t as i64;
    let nbi1 = (topk * std::mem::size_of::<u32>()) as i64;

    let ne0 = n as i64;
    let ne1 = topk as i64;
    let nb1 = (n * std::mem::size_of::<f32>()) as i64;

    let (nth0, nth1, align): (usize, usize, usize) = match dtype {
        GgmlDType::Q4_0
        | GgmlDType::Q4_1
        | GgmlDType::Q5_0
        | GgmlDType::Q5_1
        | GgmlDType::Q8_0
        | GgmlDType::Q8_1 => (8, 8, 8),
        GgmlDType::Q2K => (2, 32, 4),
        GgmlDType::Q4K => (4, 8, 4),
        GgmlDType::Q3K | GgmlDType::Q5K => (2, 32, 4),
        GgmlDType::Q6K => (2, 32, 2),
        GgmlDType::F16 | GgmlDType::BF16 | GgmlDType::Q8K => (32, 1, 8),
        GgmlDType::F32 => (32, 1, 8),
        // Same per-type layout as the plain matvec above (2 simdgroups; 8 rows/threadgroup for
        // IQ1/IQ2/IQ3, 4 for IQ4_NL/IQ4_XS). The `_id` kernels stage their grid into the flat
        // 8192-byte threadgroup scratch already bound below.
        GgmlDType::IQ2_XXS
        | GgmlDType::IQ2_XS
        | GgmlDType::IQ2_S
        | GgmlDType::IQ3_XXS
        | GgmlDType::IQ3_S
        | GgmlDType::IQ1_S
        | GgmlDType::IQ1_M => (4, 16, 8),
        GgmlDType::IQ4_NL | GgmlDType::IQ4_XS => (4, 16, 4),
    };

    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mv_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mv_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mv_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mv_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mv_id_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mv_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mv_id_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mv_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mv_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mv_id_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mv_id_f16_f32",
        GgmlDType::F32 => "kernel_mul_mv_id_f32_f32",
        GgmlDType::IQ2_XXS => "kernel_mul_mv_id_iq2_xxs_f32",
        GgmlDType::IQ2_XS => "kernel_mul_mv_id_iq2_xs_f32",
        GgmlDType::IQ2_S => "kernel_mul_mv_id_iq2_s_f32",
        GgmlDType::IQ3_XXS => "kernel_mul_mv_id_iq3_xxs_f32",
        GgmlDType::IQ3_S => "kernel_mul_mv_id_iq3_s_f32",
        GgmlDType::IQ1_S => "kernel_mul_mv_id_iq1_s_f32",
        GgmlDType::IQ1_M => "kernel_mul_mv_id_iq1_m_f32",
        GgmlDType::IQ4_NL => "kernel_mul_mv_id_iq4_nl_f32",
        GgmlDType::IQ4_XS => "kernel_mul_mv_id_iq4_xs_f32",
        other => {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "no kernel_mul_mv_id for {other:?}"
            )))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (src0s, src0_offset),
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
            (ids, ids_offset),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne01,
            ne02,
            nb00,
            nb01,
            nb02,
            ne10,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );

    // The k-quant / q_n matvec impls reduce in-simdgroup and ignore the threadgroup scratch (their
    // plain wrappers pass nullptr), but the `_id` kernel declares `shared_values [[threadgroup(0)]]`;
    // bind a nominal buffer so the argument is valid. Well within the 32 KiB threadgroup limit.
    encoder.set_threadgroup_memory_length(0, 8192);

    let thread_groups = MTLSize {
        width: divide(ne01 as usize, align),
        height: 1,
        depth: (nei0 * nei1) as usize,
    };
    let threads_per_threadgroup = MTLSize {
        width: nth0,
        height: nth1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_threadgroup);
    Ok(())
}

/// Fused MoE prefill GEMM -- ggml `kernel_mul_mm_id`, the multi-token twin of [`call_mul_mv_id`].
/// One dispatch tiles a 64x32 simdgroup matmul over the tokens routed to each expert (expert on the
/// grid's z axis; the ids are scanned on-device into a threadgroup rowid map), so each expert's
/// quantized weight is read once and amortized over all its tokens -- llama's `mul_mat_id`. Args
/// mirror `call_mul_mv_id`: `src0s` is the resident `[n_experts, n, k]` bank (blocks `expert_bytes`
/// apart), `src1` is `[t, s, k]` f32 (`s == 1` shared gate/up input or `s == topk` per-slot down),
/// `ids` is `[t, topk]` u32, `dst` is `[t, topk, n]` f32 -- same layout the matvec path writes.
#[allow(clippy::too_many_arguments)]
pub fn call_mul_mm_id(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: GgmlDType,
    n_experts: usize,
    n: usize,
    k: usize,
    t: usize,
    topk: usize,
    s: usize,
    expert_bytes: usize,
    src0s: &Buffer,
    src0_offset: usize,
    src1: &Buffer,
    src1_offset: usize,
    ids: &Buffer,
    ids_offset: usize,
    dst: &Buffer,
    dst_offset: usize,
) -> Result<(), MetalKernelError> {
    let ne00 = k as i64;
    let ne02 = n_experts as i64;
    // The mm impl indexes weight rows explicitly (unlike the matvec wrappers that pass nb01=0), so the
    // real quantized row stride is required: expert_bytes = n * (k/block_size) * type_size.
    let nb01 = (expert_bytes / n) as i64;
    let nb02 = expert_bytes as i64;

    let ne11 = s as i64;
    let ne12 = t as i64;
    let ne13 = 1i64;
    let nb10 = std::mem::size_of::<f32>() as i64;
    let nb11 = (k * std::mem::size_of::<f32>()) as i64;
    let nb12 = (s * k * std::mem::size_of::<f32>()) as i64;

    let nei0 = topk as i64;
    let nei1 = t as i64;
    let nbi1 = (topk * std::mem::size_of::<u32>()) as i64;

    let ne0 = n as i64;
    let ne1 = topk as i64;
    let nb1 = (n * std::mem::size_of::<f32>()) as i64;

    let name = match dtype {
        GgmlDType::Q4_0 => "kernel_mul_mm_id_q4_0_f32",
        GgmlDType::Q4_1 => "kernel_mul_mm_id_q4_1_f32",
        GgmlDType::Q5_0 => "kernel_mul_mm_id_q5_0_f32",
        GgmlDType::Q5_1 => "kernel_mul_mm_id_q5_1_f32",
        GgmlDType::Q8_0 => "kernel_mul_mm_id_q8_0_f32",
        GgmlDType::Q2K => "kernel_mul_mm_id_q2_K_f32",
        GgmlDType::Q3K => "kernel_mul_mm_id_q3_K_f32",
        GgmlDType::Q4K => "kernel_mul_mm_id_q4_K_f32",
        GgmlDType::Q5K => "kernel_mul_mm_id_q5_K_f32",
        GgmlDType::Q6K => "kernel_mul_mm_id_q6_K_f32",
        GgmlDType::F16 => "kernel_mul_mm_id_f16_f32",
        GgmlDType::F32 => "kernel_mul_mm_id_f32_f32",
        GgmlDType::IQ2_XXS => "kernel_mul_mm_id_iq2_xxs_f32",
        GgmlDType::IQ2_XS => "kernel_mul_mm_id_iq2_xs_f32",
        GgmlDType::IQ2_S => "kernel_mul_mm_id_iq2_s_f32",
        GgmlDType::IQ3_XXS => "kernel_mul_mm_id_iq3_xxs_f32",
        GgmlDType::IQ3_S => "kernel_mul_mm_id_iq3_s_f32",
        GgmlDType::IQ1_S => "kernel_mul_mm_id_iq1_s_f32",
        GgmlDType::IQ1_M => "kernel_mul_mm_id_iq1_m_f32",
        GgmlDType::IQ4_NL => "kernel_mul_mm_id_iq4_nl_f32",
        GgmlDType::IQ4_XS => "kernel_mul_mm_id_iq4_xs_f32",
        other => {
            return Err(MetalKernelError::LoadFunctionError(format!(
                "no kernel_mul_mm_id for {other:?}"
            )))
        }
    };

    let pipeline = kernels.load_pipeline(device, Source::Quantized, name)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(
        encoder,
        (
            (src0s, src0_offset),
            (src1, src1_offset),
            Output::with_offset(dst, dst_offset),
            (ids, ids_offset),
            nei0,
            nei1,
            nbi1,
            ne00,
            ne02,
            nb01,
            nb02,
            ne11,
            ne12,
            ne13,
            nb10,
            nb11,
            nb12,
            ne0,
            ne1,
            nb1
        )
    );

    // 8192 B of sa/sb GEMM tiles + the on-device rowid map (one ushort2 per routed row). MoE top-k ids
    // are distinct per token, so a single expert receives at most `t` rows -> `t` ushort2 entries.
    let rowids_bytes = (t * std::mem::size_of::<u32>()).next_multiple_of(16);
    encoder.set_threadgroup_memory_length(0, 8192 + rowids_bytes);

    // Classic ggml mul_mat_id grid: token tiles (x) x output-row tiles (y) x experts (z); 128 threads.
    let thread_groups = MTLSize {
        width: divide(t, 32),
        height: divide(n, 64),
        depth: n_experts,
    };
    let threads_per_threadgroup = MTLSize {
        width: 128,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(thread_groups, threads_per_threadgroup);
    Ok(())
}
