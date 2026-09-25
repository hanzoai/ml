use crate::utils::{BufferOffset, EncoderParam, EncoderProvider, Input};
use crate::{
    linear_split, set_params, Buffer, ComputeCommandEncoder, Device, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

/// Hyper-parameters of one AdamW step, laid out as the kernel reads them.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWStep {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    /// `1 / (1 − β₁ᵗ)`.
    pub scale_m: f32,
    /// `1 / (1 − β₂ᵗ)`.
    pub scale_v: f32,
    /// Multiplies the gradient before it is used, for clipping.
    pub grad_scale: f32,
}

impl EncoderParam for AdamWStep {
    fn set_param(encoder: &ComputeCommandEncoder, position: usize, data: Self) {
        encoder.set_bytes(position, &data);
    }
}

/// One AdamW step over `n` contiguous F32 elements: `w`, `m` and `v` are updated in place
/// from `g` in a single pass over memory.
#[allow(clippy::too_many_arguments)]
pub fn call_adamw(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    n: usize,
    step: AdamWStep,
    w: BufferOffset,
    g: BufferOffset,
    m: BufferOffset,
    v: BufferOffset,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Optim, "adamw_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            n as u32,
            step,
            Output::from_buffer_offset(&w),
            Input::from_buffer_offset(&g),
            Output::from_buffer_offset(&m),
            Output::from_buffer_offset(&v)
        )
    );
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, n);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

/// Adds the sum of squares of `n` contiguous F32 elements of `x` into `out[0]`.
pub fn call_sumsq(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    n: usize,
    x: BufferOffset,
    out: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Optim, "sumsq_f32")?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (n as u32, Input::from_buffer_offset(&x), Output::new(out))
    );
    let width = 1024usize;
    let groups = n.div_ceil(width * 8).clamp(1, 1024);
    encoder.dispatch_thread_groups(
        MTLSize {
            width: groups,
            height: 1,
            depth: 1,
        },
        MTLSize {
            width,
            height: 1,
            depth: 1,
        },
    );
    Ok(())
}
