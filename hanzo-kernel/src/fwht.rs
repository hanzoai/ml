//! Fast Walsh-Hadamard Transform (FWHT) with optional per-column explicit `±1` signs.
//!
//! Computes the normalized Sylvester-Walsh-Hadamard transform over contiguous `block_size`-wide
//! blocks along the last dimension (`width % block_size == 0`):
//! - Forward (`is_inverse == false`): `y = H_B (x * signs)` (scaled by `1 / sqrt(B)`)
//! - Inverse (`is_inverse == true`):  `y = (H_B x) * signs` (scaled by `1 / sqrt(B)`)
//!
//! Uses warp-level `plane_shuffle_xor` for the lower 5 butterfly stages (`stride = 1..=16`,
//! zero shared-memory traffic and zero barriers on 32-lane targets), shared-memory stages for
//! `stride = 32..threads`, and register butterflies across the `E = B / threads` elements per
//! thread. Single DSL source lowered via `kernel-gen` to Metal, CUDA, ROCm, and Vulkan.

use crate::prelude::*;

#[device]
#[inline]
fn butterfly_stage_xor(val: f32, tid: usize, #[comptime] stride: usize) -> f32 {
    let partner = plane_shuffle_xor(val, comptime!(stride as u32));
    let mut out = val + partner;
    if (tid & stride) != 0usize {
        out = partner - val;
    }
    out
}

/// Fused normalized FWHT with optional explicit `±1` signs.
///
/// Launch: `Block::new_1d(threads as u32)` where `threads = min(block_size, 256)`,
/// `Grid::Static(total_blocks.min(65535) as u32, 1, 1)`.
/// `meta = [width, blocks_per_row, has_signs, is_inverse, total_blocks]`.
#[kernel(targets(cuda, metal, vulkan, webgpu, rocm), unchecked)]
pub fn hadamard_fwht<F: Float>(
    x: &Array<F>,
    signs: &Array<f32>,
    y: &mut Array<F>,
    meta: &Array<u32>,
    #[comptime] block_size: usize,
    #[comptime] use_plane_xor: bool,
) {
    let threads = comptime!(if block_size <= 256usize {
        block_size
    } else {
        256usize
    });
    let e = comptime!(block_size / threads);
    let chunk_e = comptime!(if e > 16usize { 16usize } else { e });
    let smem_len = comptime!(chunk_e * threads);
    let inv_sqrt_b = comptime!(1.0f32 / (block_size as f32).sqrt());

    let tid = UNIT_POS_X as usize;
    let blocks_per_row = meta[1] as usize;
    let has_signs = meta[2] as usize;
    let is_inverse = meta[3] as usize;
    let total_blocks = meta[4] as usize;

    let mut smem = SharedMemory::<f32>::new(smem_len);
    let mut v = Array::<f32>::new(e);
    let mut nv = Array::<f32>::new(chunk_e);

    let mut blk = CUBE_POS_X as usize;
    let grid_stride = CUBE_COUNT_X as usize;
    while blk < total_blocks {
        let blk_base = blk * block_size;
        let row_col_base = (blk % blocks_per_row) * block_size;

        #[unroll]
        for r in 0usize..e {
            let pos = r * threads + tid;
            let mut val = f32::cast_from(x[blk_base + pos]);
            if is_inverse == 0usize && has_signs != 0usize {
                val *= signs[row_col_base + pos];
            }
            if comptime!(use_plane_xor && threads >= 32usize) {
                val = butterfly_stage_xor(val, tid, 1usize);
                val = butterfly_stage_xor(val, tid, 2usize);
                val = butterfly_stage_xor(val, tid, 4usize);
                val = butterfly_stage_xor(val, tid, 8usize);
                val = butterfly_stage_xor(val, tid, 16usize);
            }
            v[r] = val;
        }

        let start_stride = comptime!(if use_plane_xor && threads >= 32usize {
            32usize
        } else {
            1usize
        });

        if comptime!(start_stride < threads) {
            let num_chunks = comptime!(e / chunk_e);
            #[unroll]
            for c in 0usize..num_chunks {
                let r0 = comptime!(c * chunk_e);
                #[unroll]
                for r in 0usize..chunk_e {
                    smem[r * threads + tid] = v[r0 + r];
                }
                sync_cube();

                let mut stride = comptime!(start_stride);
                // Comptime unrolled log2 stages from start_stride up to threads / 2.
                let stages = comptime!((threads / start_stride).trailing_zeros() as usize);
                #[unroll]
                for s in 0usize..stages {
                    let st = comptime!(start_stride << s);
                    let partner_tid = tid ^ st;
                    let is_upper = (tid & st) != 0usize;
                    #[unroll]
                    for r in 0usize..chunk_e {
                        let cur = v[r0 + r];
                        let partner = smem[r * threads + partner_tid];
                        let mut next = cur + partner;
                        if is_upper {
                            next = partner - cur;
                        }
                        nv[r] = next;
                    }
                    sync_cube();
                    #[unroll]
                    for r in 0usize..chunk_e {
                        let next = nv[r];
                        v[r0 + r] = next;
                        if comptime!(s + 1usize < stages) {
                            smem[r * threads + tid] = next;
                        }
                    }
                    if comptime!(s + 1usize < stages) {
                        sync_cube();
                    }
                    stride = comptime!(st * 2usize);
                }
                let _ = stride;
            }
        }

        // In-register butterflies across the E elements held by this thread.
        if comptime!(e > 1usize) {
            let reg_stages = comptime!(e.trailing_zeros() as usize);
            #[unroll]
            for s in 0usize..reg_stages {
                let sr = comptime!(1usize << s);
                let step = comptime!(2usize * sr);
                let groups = comptime!(e / step);
                #[unroll]
                for g in 0usize..groups {
                    let base = comptime!(g * step);
                    #[unroll]
                    for off in 0usize..sr {
                        let a = v[base + off];
                        let b = v[base + off + sr];
                        v[base + off] = a + b;
                        v[base + off + sr] = a - b;
                    }
                }
            }
        }

        #[unroll]
        for r in 0usize..e {
            let pos = r * threads + tid;
            let mut val = v[r] * inv_sqrt_b;
            if is_inverse != 0usize && has_signs != 0usize {
                val *= signs[row_col_base + pos];
            }
            y[blk_base + pos] = F::cast_from(val);
        }

        blk += grid_stride;
    }
}

/// Host reference implementation of normalized Sylvester-Walsh-Hadamard transform with optional signs.
pub fn fwht_ref(
    x: &[f32],
    signs: Option<&[f32]>,
    width: usize,
    block_size: usize,
    is_inverse: bool,
) -> Vec<f32> {
    assert!(block_size.is_power_of_two() && width > 0 && width % block_size == 0);
    assert_eq!(x.len() % width, 0);
    let mut out = x.to_vec();
    let scale = 1.0f32 / (block_size as f32).sqrt();
    for row in out.chunks_exact_mut(width) {
        if !is_inverse {
            if let Some(s) = signs {
                for (v, &sv) in row.iter_mut().zip(s.iter()) {
                    *v *= sv;
                }
            }
        }
        for blk in row.chunks_exact_mut(block_size) {
            let mut h = 1usize;
            while h < block_size {
                for base in (0..block_size).step_by(h * 2) {
                    for off in 0..h {
                        let a = blk[base + off];
                        let b = blk[base + off + h];
                        blk[base + off] = a + b;
                        blk[base + off + h] = a - b;
                    }
                }
                h <<= 1;
            }
            for v in blk.iter_mut() {
                *v *= scale;
            }
        }
        if is_inverse {
            if let Some(s) = signs {
                for (v, &sv) in row.iter_mut().zip(s.iter()) {
                    *v *= sv;
                }
            }
        }
    }
    out
}

/// Launch [`hadamard_fwht`] on any CubeCL runtime.
#[allow(clippy::too_many_arguments)]
pub fn hadamard_fwht_run_as<F: Float + CubeElement + Default, R: Runtime>(
    client: &ComputeClient<R>,
    x: &[F],
    signs: Option<&[f32]>,
    width: usize,
    block_size: usize,
    is_inverse: bool,
    use_plane_xor: bool,
) -> Vec<F> {
    assert!(block_size.is_power_of_two() && block_size >= 32 && block_size <= 4096);
    assert!(width > 0 && width % block_size == 0 && x.len() % width == 0);
    let total_blocks = x.len() / block_size;
    let blocks_per_row = width / block_size;
    let threads = block_size.min(256);
    let has_signs = if signs.is_some() { 1u32 } else { 0u32 };
    let dummy = [1.0f32];
    let s_slice = signs.unwrap_or(&dummy);
    let xh = client.create_from_slice(F::as_bytes(x));
    let sh = client.create_from_slice(f32::as_bytes(s_slice));
    let yh = client.create_from_slice(F::as_bytes(&vec![F::default(); x.len()]));
    let meta = [
        width as u32,
        blocks_per_row as u32,
        has_signs,
        u32::from(is_inverse),
        total_blocks as u32,
    ];
    let mh = client.create_from_slice(u32::as_bytes(&meta));
    let grid_x = (total_blocks.min(65535)).max(1) as u32;
    unsafe {
        hadamard_fwht::launch_unchecked::<F, R>(
            client,
            Grid::Static(grid_x, 1, 1),
            Block::new_1d(threads as u32),
            ArrayArg::from_raw_parts(xh, x.len()),
            ArrayArg::from_raw_parts(sh, s_slice.len()),
            ArrayArg::from_raw_parts(yh.clone(), x.len()),
            ArrayArg::from_raw_parts(mh, meta.len()),
            block_size,
            use_plane_xor,
        );
    }
    F::from_bytes(&client.read_one_unchecked(yh)).to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fwht_ref_is_involutive_with_signs() {
        let width = 2048usize;
        let block_size = 1024usize;
        let x: Vec<f32> = (0..3 * width)
            .map(|i| (((i * 1103515245 + 12345) >> 16) & 1023) as f32 / 512.0 - 1.0)
            .collect();
        let signs: Vec<f32> = (0..width)
            .map(|i| if (i.wrapping_mul(2654435761) >> 31) == 0 { 1.0 } else { -1.0 })
            .collect();
        let fwd = fwht_ref(&x, Some(&signs), width, block_size, false);
        let rec = fwht_ref(&fwd, Some(&signs), width, block_size, true);
        let max_err = x
            .iter()
            .zip(&rec)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-5, "fwht_ref not involutive: max_err={max_err}");
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fwht_cuda_matches_ref_all_block_sizes() {
        use cubecl::cuda::{CudaDevice, CudaRuntime};
        let client = CudaRuntime::client(&CudaDevice::new(0));
        for block_size in [64usize, 128, 256, 512, 1024, 2048, 4096] {
            let width = if block_size == 1024 { 5120 } else { block_size * 2 };
            let rows = 3usize;
            let x: Vec<f32> = (0..rows * width)
                .map(|i| (((i * 7919) % 1009) as f32 / 504.5) - 1.0)
                .collect();
            let signs: Vec<f32> = (0..width)
                .map(|i| if ((i * 31337) & 1) == 0 { 1.0 } else { -1.0 })
                .collect();
            for is_inv in [false, true] {
                let want = fwht_ref(&x, Some(&signs), width, block_size, is_inv);
                let got = hadamard_fwht_run_as::<f32, CudaRuntime>(
                    &client,
                    &x,
                    Some(&signs),
                    width,
                    block_size,
                    is_inv,
                    true,
                );
                let max_err = want
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 2e-5,
                    "cuda fwht f32 B={block_size} inv={is_inv}: max_err={max_err}"
                );

                let x_bf16: Vec<half::bf16> = x.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let x_bf16_f: Vec<f32> = x_bf16.iter().map(|v| v.to_f32()).collect();
                let want_b = fwht_ref(&x_bf16_f, Some(&signs), width, block_size, is_inv);
                let got_b = hadamard_fwht_run_as::<half::bf16, CudaRuntime>(
                    &client,
                    &x_bf16,
                    Some(&signs),
                    width,
                    block_size,
                    is_inv,
                    true,
                );
                let max_err_b = want_b
                    .iter()
                    .zip(&got_b)
                    .map(|(a, b)| (a - b.to_f32()).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err_b < 2e-2,
                    "cuda fwht bf16 B={block_size} inv={is_inv}: max_err={max_err_b}"
                );
            }
        }
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn fwht_vulkan_matches_ref() {
        use cubecl::wgpu::{WgpuDevice, WgpuRuntime};
        let client = WgpuRuntime::client(&WgpuDevice::DefaultDevice);
        let plane = client.properties().hardware.plane_size_max as usize;
        let use_plane = plane == 32;
        for block_size in [64usize, 256, 1024, 4096] {
            let width = if block_size == 1024 { 5120 } else { block_size * 2 };
            let x: Vec<f32> = (0..2 * width)
                .map(|i| (((i * 7919) % 1009) as f32 / 504.5) - 1.0)
                .collect();
            let signs: Vec<f32> = (0..width)
                .map(|i| if ((i * 31337) & 1) == 0 { 1.0 } else { -1.0 })
                .collect();
            for is_inv in [false, true] {
                let want = fwht_ref(&x, Some(&signs), width, block_size, is_inv);
                let got = hadamard_fwht_run_as::<f32, WgpuRuntime>(
                    &client,
                    &x,
                    Some(&signs),
                    width,
                    block_size,
                    is_inv,
                    use_plane,
                );
                let max_err = want
                    .iter()
                    .zip(&got)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, f32::max);
                assert!(
                    max_err < 2e-5,
                    "vulkan fwht f32 B={block_size} inv={is_inv}: max_err={max_err}"
                );
            }
        }
    }
}
