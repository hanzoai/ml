use crate::kernels::macros::ops;
use crate::utils::{BufferOffset, EncoderProvider};
use crate::{get_block_dims, get_tile_size, linear_split};
use crate::{
    set_params, Buffer, ComputeCommandEncoder, Device, EncoderParam, Kernels, MetalKernelError,
    Output, Source,
};
use objc2_metal::MTLSize;

ops!(
    cos, sin, exp, sqr, sqrt, neg, log, gelu, abs, ceil, floor, relu, round, erf, gelu_erf, tanh,
    recip, silu, sign, sigmoid, const_set
);

#[allow(clippy::too_many_arguments)]
pub fn call_unary_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: BufferOffset,
    output: &Buffer,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel_name.0)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);

    set_params!(encoder, (length, &input, Output::new(output)));

    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_unary_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: BufferOffset,
    strides: &[usize],
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;

    let length: usize = shape.iter().product();
    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            strides,
            &input,
            Output::from_buffer_offset(&output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_const_set_contiguous(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    kernel_name: contiguous::Kernel,
    dtype_size: usize,
    length: usize,
    input: impl EncoderParam,
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, kernel_name.0)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();

    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (length, input, Output::from_buffer_offset(&output))
    );

    let tile_size = get_tile_size(dtype_size);
    let tiles = length.div_ceil(tile_size);
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, tiles);
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn call_const_set_strided(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: strided::Kernel,
    shape: &[usize],
    input: impl EncoderParam,
    strides: &[usize],
    output: BufferOffset,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;

    let length: usize = shape.iter().product();
    let num_dims: usize = shape.len();
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    let (thread_group_count, thread_group_size) = linear_split(&pipeline, length);

    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            length,
            num_dims,
            shape,
            strides,
            input,
            Output::from_buffer_offset(&output)
        )
    );
    encoder.dispatch_thread_groups(thread_group_count, thread_group_size);
    Ok(())
}

pub mod copy2d {
    pub struct Kernel(pub &'static str);
    pub const FLOAT: Kernel = Kernel("copy2d_f32");
    pub const HALF: Kernel = Kernel("copy2d_f16");
    pub const BFLOAT: Kernel = Kernel("copy2d_bf16");
    pub const I64: Kernel = Kernel("copy2d_i64");
    pub const I32: Kernel = Kernel("copy2d_i32");
    pub const I16: Kernel = Kernel("copy2d_i16");
    pub const U32: Kernel = Kernel("copy2d_u32");
    pub const U8: Kernel = Kernel("copy2d_u8");
}

#[allow(clippy::too_many_arguments)]
pub fn call_copy2d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: copy2d::Kernel,
    input: &Buffer,
    output: &Buffer,
    d1: usize,
    d2: usize,
    src_s: usize,
    dst_s: usize,
    src_o_in_bytes: usize,
    dst_o_in_bytes: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            d1 as i64,
            d2 as i64,
            src_s as i64,
            dst_s as i64,
            (input, src_o_in_bytes),
            Output::with_offset(output, dst_o_in_bytes)
        )
    );

    let grid_dims = MTLSize {
        width: d1,
        height: d2,
        depth: 1,
    };
    let group_dims = get_block_dims(d1, d2, 1);
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}

pub mod copy3d {
    pub struct Kernel(pub &'static str);
    pub const FLOAT: Kernel = Kernel("copy3d_f32");
    pub const HALF: Kernel = Kernel("copy3d_f16");
    pub const BFLOAT: Kernel = Kernel("copy3d_bf16");
    pub const I64: Kernel = Kernel("copy3d_i64");
    pub const I32: Kernel = Kernel("copy3d_i32");
    pub const I16: Kernel = Kernel("copy3d_i16");
    pub const U32: Kernel = Kernel("copy3d_u32");
    pub const U8: Kernel = Kernel("copy3d_u8");
}

/// Coalesced batched-transpose copy: a strided source with a contiguous innermost dim,
/// reduced to three effective dimensions `(d0, d1, d2)` with source strides `(s0, s1, s2)`
/// where `s2 == 1`, copied into a contiguous destination. The innermost dim `d2` maps to
/// grid x so adjacent lanes touch contiguous memory; the source offset is a fused product
/// of the grid position rather than a per-element multi-dimensional index. See `copy3d`
/// in `unary.metal`.
#[allow(clippy::too_many_arguments)]
pub fn call_copy3d(
    device: &Device,
    ep: impl EncoderProvider,
    kernels: &Kernels,
    name: copy3d::Kernel,
    input: &Buffer,
    output: &Buffer,
    d0: usize,
    d1: usize,
    d2: usize,
    s0: usize,
    s1: usize,
    s2: usize,
    src_o_in_bytes: usize,
    dst_o_in_bytes: usize,
) -> Result<(), MetalKernelError> {
    let pipeline = kernels.load_pipeline(device, Source::Unary, name.0)?;
    let encoder = ep.encoder();
    let encoder: &ComputeCommandEncoder = encoder.as_ref();
    encoder.set_compute_pipeline_state(&pipeline);
    set_params!(
        encoder,
        (
            d0 as i64,
            d1 as i64,
            d2 as i64,
            s0 as i64,
            s1 as i64,
            s2 as i64,
            (input, src_o_in_bytes),
            Output::with_offset(output, dst_o_in_bytes)
        )
    );

    let grid_dims = MTLSize {
        width: d2,
        height: d1,
        depth: d0,
    };
    let group_dims = get_block_dims(d2, d1, d0);
    encoder.dispatch_threads(grid_dims, group_dims);
    Ok(())
}
