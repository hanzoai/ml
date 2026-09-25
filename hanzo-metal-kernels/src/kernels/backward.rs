//! Fused backward passes of the row-wise normalizations (LayerNorm, last-dim softmax).

use crate::utils::{BufferOffset, EncoderProvider, Input};
use crate::{
    set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError, Output, Source,
};
use objc2_metal::MTLSize;

/// Rows of the parameter-gradient pass each thread sums before its atomic add.
const ROWS_PER: usize = 64;

fn size(width: usize) -> MTLSize {
    MTLSize {
        width,
        height: 1,
        depth: 1,
    }
}

/// Threads per row: the row width rounded up to a power of two, at least one simdgroup.
fn row_threads(d: usize, max: usize) -> usize {
    d.next_power_of_two().clamp(32, max.min(1024))
}

/// LayerNorm backward over `n` rows of width `d`. Writes `dx` (the input's dtype), the rows'
/// `(mean, rstd)` to `stats` (`n × 2` F32) and adds `dalpha`, `dbeta` into `dparams`
/// (`2 × d` F32, zeroed by the caller). `dtype` is `f32`, `f16` or `bf16`.
#[allow(clippy::too_many_arguments)]
pub fn call_layer_norm_bwd(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: &str,
    n: usize,
    d: usize,
    eps: f32,
    x: BufferOffset,
    dy: BufferOffset,
    alpha: BufferOffset,
    dx: BufferOffset,
    stats: &Buffer,
    dparams: &Buffer,
) -> Result<(), MetalKernelError> {
    let (rows, params) = match dtype {
        "f32" => ("layernorm_bwd_f32", "layernorm_bwd_params_f32"),
        "f16" => ("layernorm_bwd_f16", "layernorm_bwd_params_f16"),
        "bf16" => ("layernorm_bwd_bf16", "layernorm_bwd_params_bf16"),
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "layernorm_bwd {dtype}"
            )))
        }
    };
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    let pipeline = kernels.load_pipeline(device, Source::Backward, rows)?;
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            d as u32,
            eps,
            Input::from_buffer_offset(&x),
            Input::from_buffer_offset(&dy),
            Input::from_buffer_offset(&alpha),
            Output::from_buffer_offset(&dx),
            Output::new(stats)
        )
    );
    let tg = row_threads(d, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(size(n), size(tg));

    let pipeline = kernels.load_pipeline(device, Source::Backward, params)?;
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            n as u32,
            d as u32,
            ROWS_PER as u32,
            Input::from_buffer_offset(&x),
            Input::from_buffer_offset(&dy),
            Input::new(stats),
            Output::new(dparams)
        )
    );
    let width = 256usize.min(d.next_power_of_two());
    encoder.dispatch_thread_groups(
        MTLSize {
            width: d.div_ceil(width),
            height: n.div_ceil(ROWS_PER),
            depth: 1,
        },
        size(width),
    );
    Ok(())
}

/// Last-dim softmax backward over `n` rows of width `d`: `dx = y (dy − ⟨dy, y⟩)`.
#[allow(clippy::too_many_arguments)]
pub fn call_softmax_bwd(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    dtype: &str,
    n: usize,
    d: usize,
    y: BufferOffset,
    dy: BufferOffset,
    dx: BufferOffset,
) -> Result<(), MetalKernelError> {
    let name = match dtype {
        "f32" => "softmax_bwd_f32",
        "f16" => "softmax_bwd_f16",
        "bf16" => "softmax_bwd_bf16",
        _ => {
            return Err(MetalKernelError::LoadLibraryError(format!(
                "softmax_bwd {dtype}"
            )))
        }
    };
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let pipeline = kernels.load_pipeline(device, Source::Backward, name)?;
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            d as u32,
            Input::from_buffer_offset(&y),
            Input::from_buffer_offset(&dy),
            Output::from_buffer_offset(&dx)
        )
    );
    let tg = row_threads(d, pipeline.max_total_threads_per_threadgroup());
    encoder.dispatch_thread_groups(size(n), size(tg));
    Ok(())
}
