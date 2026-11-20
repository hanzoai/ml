//! Canonical Fast Walsh-Hadamard Transform (FWHT) kernel in the Hanzo Kernels DSL.
//!
//! Replaces the dense `O(B^2)` `matmul(&H_B)` activation rotation in `DeviceHadamard` (`11.7 ms/tok`
//! on M4 Max 27B) with an `O(B log B)` fused butterfly in shared memory + register radix-4 prelude:
//! - Forward (`inverse == 0`): `y = ((x * signs).reshape(-1, B) @ H_B).reshape_as(x)`
//! - Inverse (`inverse != 0`): `y = (x.reshape(-1, B) @ H_B).reshape_as(x) * signs`
//! where `H_B[r, c] = (1 / sqrt(B)) * (-1)^{popcnt(r & c)}` (normalized Sylvester-Walsh-Hadamard).
//!
//! One single Rust source lowered via `kernel-gen` to Metal (`.metal`), CUDA (`.cu`), ROCm (`.hip`),
//! and Vulkan (`.spv`) across `F32`, `F16`, and `BF16` I/O (internal `F32` butterfly arithmetic).

use cubecl::prelude::*;
use hanzo_kernel_macros::kernel;

/// Normalized Sylvester-Walsh-Hadamard transform with fused optional `±1` sign fold.
///
/// - `x`: `[n_blocks * B]` input elements (`F32`, `F16`, or `BF16`)
/// - `signs`: `[sign_width]` `F32` sign vector (`sign_width` is a multiple of `B`; unused when
///   `has_signs == 0`)
/// - `out`: `[n_blocks * B]` output elements (same dtype as `x`)
/// - `dims`: `[n_blocks, sign_width, has_signs, inverse]`
/// - `b`: power-of-two block size (`64, 128, 256, 512, 1024, 2048`), `b % 4 == 0`
/// - `nt`: threads per cube (`b / 4`, up to `256`; each thread loads/stores `b / nt` elements)
#[kernel(targets(cuda, rocm, vulkan, metal, cpu), unchecked)]
pub fn fwht_hadamard<F: Float>(
    x: &Array<F>,
    signs: &Array<f32>,
    out: &mut Array<F>,
    dims: &Array<u32>,
    #[comptime] b: usize,
    #[comptime] nt: usize,
    #[comptime] log2_b: usize,
) {
    let blk = CUBE_POS as usize;
    let n_blocks = dims[0] as usize;
    if blk < n_blocks {
        let tid = UNIT_POS as usize;
        let sign_w = dims[1] as usize;
        let has_signs = dims[2];
        let inverse = dims[3];
        let base = blk * b;
        let mut s_base = 0usize;
        if has_signs != 0u32 {
            s_base = base % sign_w;
        }
        let scale = comptime!(1.0f32 / (b as f32).sqrt());
        let elems_per_t = comptime!(b / nt);
        let quads_per_t = comptime!(elems_per_t / 4);
        let pairs_per_t = comptime!(b / (2 * nt));

        let mut sm = SharedMemory::<f32>::new(b);

        // Load 4-element contiguous quads per thread, apply optional forward signs, and fuse the
        // first two butterfly stages (half = 1 and half = 2) directly in registers before writing
        // to shared memory (saves 2 sync_cube barriers and 2 shared-memory round trips).
        #[unroll]
        for q in 0..quads_per_t {
            let off = (tid + q * nt) * 4usize;
            let mut v0 = f32::cast_from(x[base + off]);
            let mut v1 = f32::cast_from(x[base + off + 1usize]);
            let mut v2 = f32::cast_from(x[base + off + 2usize]);
            let mut v3 = f32::cast_from(x[base + off + 3usize]);
            if has_signs != 0u32 && inverse == 0u32 {
                v0 *= signs[s_base + off];
                v1 *= signs[s_base + off + 1usize];
                v2 *= signs[s_base + off + 2usize];
                v3 *= signs[s_base + off + 3usize];
            }
            let a0 = v0 + v1;
            let a1 = v0 - v1;
            let a2 = v2 + v3;
            let a3 = v2 - v3;
            sm[off] = a0 + a2;
            sm[off + 1usize] = a1 + a3;
            sm[off + 2usize] = a0 - a2;
            sm[off + 3usize] = a1 - a3;
        }
        sync_cube();

        // Remaining butterfly stages (stage = 2..log2_b, i.e., half = 4, 8, ..., b/2).
        #[unroll]
        for stage in 2..log2_b {
            let half = comptime!(1usize << stage);
            let span = comptime!(half << 1);
            let mask = comptime!(half - 1usize);
            #[unroll]
            for p in 0..pairs_per_t {
                let pair = tid + p * nt;
                let grp = pair >> stage;
                let off = pair & mask;
                let i0 = grp * span + off;
                let i1 = i0 + half;
                let a = sm[i0];
                let bv = sm[i1];
                sm[i0] = a + bv;
                sm[i1] = a - bv;
            }
            sync_cube();
        }

        // Normalize by 1/sqrt(B), apply optional inverse signs, and store.
        #[unroll]
        for q in 0..quads_per_t {
            let off = (tid + q * nt) * 4usize;
            let mut v0 = sm[off] * scale;
            let mut v1 = sm[off + 1usize] * scale;
            let mut v2 = sm[off + 2usize] * scale;
            let mut v3 = sm[off + 3usize] * scale;
            if has_signs != 0u32 && inverse != 0u32 {
                v0 *= signs[s_base + off];
                v1 *= signs[s_base + off + 1usize];
                v2 *= signs[s_base + off + 2usize];
                v3 *= signs[s_base + off + 3usize];
            }
            out[base + off] = F::cast_from(v0);
            out[base + off + 1usize] = F::cast_from(v1);
            out[base + off + 2usize] = F::cast_from(v2);
            out[base + off + 3usize] = F::cast_from(v3);
        }
    }
}

/// Optimal threads-per-cube for `fwht_hadamard` at power-of-two `b` (capped at 256).
pub const fn fwht_threads(b: usize) -> usize {
    let t = b / 4;
    if t > 256 {
        256
    } else if t < 16 {
        16
    } else {
        t
    }
}

/// Host launcher for [`fwht_hadamard`] (used by tests and `kernel-gen`).
#[allow(clippy::too_many_arguments)]
pub fn fwht_hadamard_run_as<E: Float + CubeElement, R: Runtime>(
    client: &ComputeClient<R>,
    x: &[E],
    signs: &[f32],
    b: usize,
    has_signs: bool,
    inverse: bool,
) -> Vec<E> {
    assert!(
        b >= 64 && b.is_power_of_two(),
        "unsupported FWHT block size {b}"
    );
    assert!(
        x.len().is_multiple_of(b),
        "input len must be multiple of {b}"
    );
    let n_blocks = x.len() / b;
    let sign_w = if has_signs { signs.len() } else { b };
    let signs_buf = if has_signs { signs } else { &[1.0f32] };
    let nt = fwht_threads(b);
    let log2_b = b.trailing_zeros() as usize;

    let xh = client.create_from_slice(E::as_bytes(x));
    let sh = client.create_from_slice(f32::as_bytes(signs_buf));
    let oh = client.create_from_slice(E::as_bytes(&vec![E::from_int(0); x.len()]));
    let dh = client.create_from_slice(u32::as_bytes(&[
        n_blocks as u32,
        sign_w as u32,
        u32::from(has_signs),
        u32::from(inverse),
    ]));
    unsafe {
        fwht_hadamard::launch_unchecked::<E, R>(
            client,
            CubeCount::Static(n_blocks as u32, 1, 1),
            CubeDim::new_1d(nt as u32),
            ArrayArg::from_raw_parts(xh, x.len()),
            ArrayArg::from_raw_parts(sh, signs_buf.len()),
            ArrayArg::from_raw_parts(oh.clone(), x.len()),
            ArrayArg::from_raw_parts(dh, 4),
            b,
            nt,
            log2_b,
        );
    }
    E::from_bytes(&client.read_one(oh).unwrap()).to_vec()
}

/// Exact CPU f64 reference for normalized Sylvester-Walsh-Hadamard transform with optional signs.
pub fn fwht_hadamard_ref(x: &[f32], signs: Option<&[f32]>, b: usize, inverse: bool) -> Vec<f32> {
    let n_blocks = x.len() / b;
    let scale = 1.0f64 / (b as f64).sqrt();
    let mut out = vec![0.0f32; x.len()];
    let mut buf = vec![0.0f64; b];
    for blk in 0..n_blocks {
        let base = blk * b;
        for i in 0..b {
            let mut v = x[base + i] as f64;
            if let Some(s) = signs {
                if !inverse {
                    v *= s[(base + i) % s.len()] as f64;
                }
            }
            buf[i] = v;
        }
        let mut h = 1usize;
        while h < b {
            let mut i = 0usize;
            while i < b {
                for j in i..i + h {
                    let a = buf[j];
                    let c = buf[j + h];
                    buf[j] = a + c;
                    buf[j + h] = a - c;
                }
                i += h * 2;
            }
            h <<= 1;
        }
        for i in 0..b {
            let mut v = buf[i] * scale;
            if let Some(s) = signs {
                if inverse {
                    v *= s[(base + i) % s.len()] as f64;
                }
            }
            out[base + i] = v as f32;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed.wrapping_add(0x9e37_79b9_7f4a_7c15);
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                ((s >> 40) as f32) / ((1u32 << 24) as f32) * 2.0 - 1.0
            })
            .collect()
    }

    fn rel_err(a: &[f32], b: &[f32]) -> f32 {
        let md = a
            .iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let mm = b.iter().map(|y| y.abs()).fold(0.0f32, f32::max).max(1e-6);
        md / mm
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn fwht_hadamard_cuda_all_sizes_and_dtypes() {
        let client = cubecl::cuda::CudaRuntime::client(&Default::default());
        for &b in &[64usize, 128, 256, 512, 1024, 2048] {
            let width = b * 5; // matches Zen 6 27B 5120 = 5 * 1024
            let rows = 3usize;
            let x = seeded(rows * width, 100 + b as u64);
            let signs: Vec<f32> = seeded(width, 200 + b as u64)
                .into_iter()
                .map(|v| if v >= 0.0 { 1.0 } else { -1.0 })
                .collect();
            for &inverse in &[false, true] {
                let want = fwht_hadamard_ref(&x, Some(&signs), b, inverse);
                let got_f32 = fwht_hadamard_run_as::<f32, cubecl::cuda::CudaRuntime>(
                    &client, &x, &signs, b, true, inverse,
                );
                let e32 = rel_err(&got_f32, &want);
                assert!(e32 < 2e-6, "fwht f32 b={b} inv={inverse} rel={e32:.2e}");

                let x16: Vec<half::f16> = x.iter().map(|&v| half::f16::from_f32(v)).collect();
                let x16_f32: Vec<f32> = x16.iter().map(|v| v.to_f32()).collect();
                let want16 = fwht_hadamard_ref(&x16_f32, Some(&signs), b, inverse);
                let got16: Vec<f32> = fwht_hadamard_run_as::<half::f16, cubecl::cuda::CudaRuntime>(
                    &client, &x16, &signs, b, true, inverse,
                )
                .into_iter()
                .map(|v| v.to_f32())
                .collect();
                let e16 = rel_err(&got16, &want16);
                assert!(e16 < 2e-3, "fwht f16 b={b} inv={inverse} rel={e16:.2e}");

                let xb16: Vec<half::bf16> = x.iter().map(|&v| half::bf16::from_f32(v)).collect();
                let xb16_f32: Vec<f32> = xb16.iter().map(|v| v.to_f32()).collect();
                let wantb16 = fwht_hadamard_ref(&xb16_f32, Some(&signs), b, inverse);
                let gotb16: Vec<f32> =
                    fwht_hadamard_run_as::<half::bf16, cubecl::cuda::CudaRuntime>(
                        &client, &xb16, &signs, b, true, inverse,
                    )
                    .into_iter()
                    .map(|v| v.to_f32())
                    .collect();
                let eb16 = rel_err(&gotb16, &wantb16);
                assert!(
                    eb16 < 1.5e-2,
                    "fwht bf16 b={b} inv={inverse} rel={eb16:.2e}"
                );
                eprintln!("[fwht CUDA] b={b:4} inv={inverse:<5} f32={e32:.2e} f16={e16:.2e} bf16={eb16:.2e}");
            }
            // Round-trip: inverse(forward(x)) == x
            let fwd = fwht_hadamard_run_as::<f32, cubecl::cuda::CudaRuntime>(
                &client, &x, &signs, b, true, false,
            );
            let rt = fwht_hadamard_run_as::<f32, cubecl::cuda::CudaRuntime>(
                &client, &fwd, &signs, b, true, true,
            );
            assert!(rel_err(&rt, &x) < 3e-6, "fwht round-trip failed for b={b}");
        }
    }
}
