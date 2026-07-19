//! Bit-exact validation of the Metal `copy3d` coalesced strided-copy path (unary.metal /
//! metal_backend `copy_strided_src`), which materializes a batched transpose such as
//! `contiguous(x.transpose(1, 2))` without the generic per-element strided kernel.
//!
//! A contiguity copy must be an identity on values, so the Metal result is compared
//! BIT-FOR-BIT (max abs diff == 0) against the same operation on the CPU oracle. Covers the
//! live prefill attention shapes (q, attn-output, and the K/V cache append, head_dim 128), a
//! small odd shape, a 3D transpose, and a shape with four non-unit dims that must fall back to
//! the generic strided path (still exact). Skips cleanly with no Metal GPU. Needs `metal`.
#![cfg(feature = "metal")]

use hanzo_ml::{DType, Device, Tensor};

// Deterministic pseudo-random f32 in [-1, 1) (splitmix64-ish; reproducible, no rng dep).
fn pseudo(i: usize) -> f32 {
    let mut z = (i as u64).wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    ((z >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
}

fn flat_f32(t: &Tensor) -> Vec<f32> {
    t.to_dtype(DType::F32)
        .unwrap()
        .flatten_all()
        .unwrap()
        .to_vec1::<f32>()
        .unwrap()
}

// `contiguous(x.transpose(t1, t2))` on Metal must equal the CPU oracle bit-for-bit.
fn check(dev: &Device, dims: &[usize], t1: usize, t2: usize, dtype: DType) {
    let n: usize = dims.iter().product();
    let data: Vec<f32> = (0..n).map(pseudo).collect();
    let cpu = Tensor::from_vec(data, dims.to_vec(), &Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let met = cpu.to_device(dev).unwrap();

    let cpu_r = flat_f32(&cpu.transpose(t1, t2).unwrap().contiguous().unwrap());
    let met_r = flat_f32(&met.transpose(t1, t2).unwrap().contiguous().unwrap());

    assert_eq!(cpu_r.len(), met_r.len());
    let max = cpu_r
        .iter()
        .zip(&met_r)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert_eq!(
        max, 0.0,
        "copy3d mismatch dims={dims:?} transpose=({t1},{t2}) dtype={dtype:?} max_abs={max}"
    );
}

#[test]
fn metal_copy3d_transpose_bit_exact() {
    let dev = match Device::new_metal(0) {
        Ok(d) => d,
        Err(_) => return, // no Metal GPU
    };
    for dtype in [DType::BF16, DType::F32] {
        // Live prefill attention shapes (Qwen3-4B, head_dim 128), transpose(1, 2):
        check(&dev, &[1, 512, 8, 128], 1, 2, dtype); // -> [1,8,512,128]  K/V cache append
        check(&dev, &[1, 512, 32, 128], 1, 2, dtype); // -> [1,32,512,128] q RoPE contiguous
        check(&dev, &[1, 32, 512, 128], 1, 2, dtype); // -> [1,512,32,128] attn-output reshape
        // Odd small shape (index-math corner cases):
        check(&dev, &[1, 5, 3, 7], 1, 2, dtype);
        // Pure 3D transpose (no unit dim to drop):
        check(&dev, &[4, 6, 8], 0, 1, dtype);
        // Four non-unit dims: not reducible to 3D -> generic strided fallback, still exact:
        check(&dev, &[2, 8, 16, 4], 1, 2, dtype);
    }
}
