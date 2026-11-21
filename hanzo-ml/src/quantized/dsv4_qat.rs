//! DeepSeek-V4 quantization-aware-training (QAT) activation round-trips.
//!
//! These are *fake-quant* (quantize -> dequantize, value stays `f32`) round-trips that
//! reproduce, bit-for-bit, the simulated-quantization the official DeepSeek-V4 graph bakes
//! into two activation paths:
//!
//!   1. **FP8 KV-latent round-trip** — an E4M3 (FN, OCP) round-trip with a per-64-element
//!      power-of-two block scale, applied to the non-RoPE ("nope") part of the compressed
//!      KV latent only. The trailing RoPE dims are left untouched.
//!   2. **Indexer QAT round-trip** — a 128-wide normalized Hadamard transform followed by an
//!      E2M1 FP4 round-trip with a per-32-element power-of-two block scale, applied to both the
//!      indexer Q and the indexer-compressor KV.
//!
//! Skipping either drifts the indexer's top-k compressed-row selection (and therefore the
//! logits) away from the official model's computation graph.
//!
//! ## Why this is its own primitive (decomplected from the storage dtype)
//! [`float8::F8E4M3`] is the *storage* representation of an 8-bit float. This module is a
//! distinct concern: the *activation simulation snap* the model was trained against — a
//! dynamic per-block scale, saturating clamp, round-to-nearest-even grid snap, kept in `f32`.
//! The two share the E4M3 *grid* (proven equal in tests via `F8E4M3` as the oracle) but are
//! orthogonal operations, so this lives in its own slot.
//!
//! ## Authoritative reference
//! Every function here is a line-faithful port of the CPU reference in `ds4-study/ds4.c`:
//!   - [`e4m3_nearest`]            <- `dsv4_e4m3fn_value_cpu` / `dsv4_e4m3fn_dequant_cpu`   (ds4.c:2444-2484)
//!   - [`fp8_kv_quantize_rows_inplace`] <- `dsv4_fp8_kv_quantize_row_inplace_cpu`           (ds4.c:2489-2507)
//!   - [`e2m1_nearest`]            <- `dsv4_e2m1fn_value_cpu` / `dsv4_e2m1fn_dequant_cpu`   (ds4.c:2509-2529)
//!   - [`hadamard128_inplace`]     <- `dsv4_hadamard128_inplace_cpu`                        (ds4.c:2531-2544)
//!   - fp4 act round-trip          <- `dsv4_fp4_act_quantize_row_inplace_cpu`              (ds4.c:2546-2564)
//!   - [`indexer_qat_rows_inplace`]<- `dsv4_indexer_qat_row(s)_inplace_cpu`                (ds4.c:2570-2580)

use crate::{DType, Device, Result, Tensor};

// ---------------------------------------------------------------------------------------------
// Power-of-two helper (== C `ldexpf(1.0f, e)`, bit-exact across the full f32 exponent range).
// ---------------------------------------------------------------------------------------------

/// `2^e` as an `f32`, bit-identical to C `ldexpf(1.0f, e)` for every `e`.
///
/// The QAT scales are always `2^ceil(log2(amax/divisor))`; the `amax` floors in the callers
/// keep `e` inside the normal range, but this handles subnormals and overflow exactly anyway.
#[inline]
fn pow2(e: i32) -> f32 {
    if e > 127 {
        f32::INFINITY
    } else if e >= -126 {
        // Normal: biased exponent (e + 127) in [1, 254], zero mantissa.
        f32::from_bits(((e + 127) as u32) << 23)
    } else if e >= -149 {
        // Subnormal: single mantissa bit, exponent field zero.
        f32::from_bits(1u32 << (e + 149))
    } else {
        0.0
    }
}

/// `2^ceil(log2(amax / divisor))` — the shared per-block scale (C `ldexpf(1, ceilf(log2f(...)))`).
/// `f32::log2`/`f32::ceil` lower to the same libm `log2f`/`ceilf` C uses, so this is bit-exact.
#[inline]
fn block_scale(amax: f32, divisor: f32) -> f32 {
    pow2((amax / divisor).log2().ceil() as i32)
}

// ---------------------------------------------------------------------------------------------
// E4M3 (FN, OCP) grid — port of ds4.c:2444-2484.
// ---------------------------------------------------------------------------------------------

/// Magnitude of E4M3FN code `i` (0..=126; 127 is the NaN slot and is never selected).
/// Port of `dsv4_e4m3fn_value_cpu` (ds4.c:2444-2457).
#[inline]
fn e4m3_grid_value(code: u32) -> f32 {
    // exp_scale[e] = 2^(e-7); index 0 marks the subnormal range.
    const EXP_SCALE: [f32; 16] = [
        0.0, 0.015625, 0.03125, 0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0,
        128.0, 256.0,
    ];
    let exp = ((code >> 3) & 0x0f) as usize;
    let mant = (code & 0x07) as f32;
    if exp == 0 {
        mant * 0.001953125 // 2^-9 subnormal step
    } else {
        (1.0 + mant * 0.125) * EXP_SCALE[exp]
    }
}

/// Snap `x` to the nearest E4M3FN value (round-to-nearest, ties-to-even), saturating at ±448.
/// Port of `dsv4_e4m3fn_dequant_cpu` (ds4.c:2459-2484).
#[inline]
pub fn e4m3_nearest(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(448.0);

    // Largest code in [0, 126] whose grid value is <= ax.
    let (mut lo, mut hi) = (0i32, 126i32);
    while lo < hi {
        let mid = (lo + hi + 1) >> 1;
        if e4m3_grid_value(mid as u32) <= ax {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }

    let mut best = lo;
    if best < 126 {
        let best_diff = (ax - e4m3_grid_value(best as u32)).abs();
        let next_diff = (ax - e4m3_grid_value((best + 1) as u32)).abs();
        // Ties resolve to the even code (even mantissa LSB == IEEE round-half-to-even).
        if next_diff < best_diff
            || (next_diff == best_diff && ((best + 1) & 1) == 0 && (best & 1) != 0)
        {
            best += 1;
        }
    }
    sign * e4m3_grid_value(best as u32)
}

// ---------------------------------------------------------------------------------------------
// E2M1 (FP4) grid — port of ds4.c:2509-2529.
// ---------------------------------------------------------------------------------------------

/// E2M1 grid magnitudes. Same value set as the GGUF `KVALUES_MXFP4` LUT at half scale
/// (`[0,1,2,3,4,6,8,12] / 2`), but expressed directly as the DeepSeek graph uses them.
const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// Snap `x` to the nearest E2M1 value (round-to-nearest, ties-to-even), saturating at ±6.
/// Port of `dsv4_e2m1fn_dequant_cpu` (ds4.c:2516-2529).
#[inline]
pub fn e2m1_nearest(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(6.0);

    let mut best = 0usize;
    let mut best_diff = (ax - E2M1_VALUES[0]).abs();
    for i in 1..8 {
        let diff = (ax - E2M1_VALUES[i]).abs();
        // Ties resolve to the even index (round-half-to-even on the grid index).
        if diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0) {
            best = i;
            best_diff = diff;
        }
    }
    sign * E2M1_VALUES[best]
}

// ---------------------------------------------------------------------------------------------
// 128-wide normalized Hadamard transform — port of ds4.c:2531-2544.
// ---------------------------------------------------------------------------------------------

/// `1 / sqrt(128)` — the orthonormal normalization (matches ds4.c's literal).
const HADAMARD128_SCALE: f32 = 0.088_388_347_648_318_45;

/// In-place size-128 fast Walsh-Hadamard transform, normalized by `1/sqrt(128)`.
/// `x.len()` must be 128. Port of `dsv4_hadamard128_inplace_cpu` (ds4.c:2531-2544).
///
/// The natural-ordered Hadamard matrix is symmetric, so the normalized transform is its own
/// inverse: applying it twice returns the input (proven in tests).
pub fn hadamard128_inplace(x: &mut [f32]) -> Result<()> {
    if x.len() != 128 {
        crate::bail!("DSV4 Hadamard-128 requires a 128-wide row, got {}", x.len());
    }
    let mut stride = 1usize;
    while stride < 128 {
        let mut base = 0usize;
        while base < 128 {
            for i in 0..stride {
                let a = x[base + i];
                let b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
            base += 2 * stride;
        }
        stride <<= 1;
    }
    for v in x.iter_mut() {
        *v *= HADAMARD128_SCALE;
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Per-block activation round-trips (private kernels; the public API validates and dispatches).
// ---------------------------------------------------------------------------------------------

/// E2M1 FP4 round-trip over 32-element blocks. `block.len()` is assumed 32-aligned.
/// Port of `dsv4_fp4_act_quantize_row_inplace_cpu` (ds4.c:2546-2564).
fn fp4_act_quantize_blocks(x: &mut [f32]) {
    for block in x.chunks_mut(32) {
        let mut amax = 0.0f32;
        for &v in block.iter() {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        // 6 * FLT_MIN: keeps amax/6 >= 2^-126 so the scale stays a normal float.
        if amax < 7.052_966_104_933_725e-38 {
            amax = 7.052_966_104_933_725e-38;
        }
        let scale = block_scale(amax, 6.0);
        for v in block.iter_mut() {
            let mut t = *v / scale;
            if t > 6.0 {
                t = 6.0;
            } else if t < -6.0 {
                t = -6.0;
            }
            *v = e2m1_nearest(t) * scale;
        }
    }
}

/// E4M3 FP8 round-trip over the leading `n_nope` dims of `row`, in 64-element blocks.
/// `n_nope` is assumed 64-aligned and `<= row.len()`.
/// Port of `dsv4_fp8_kv_quantize_row_inplace_cpu` (ds4.c:2489-2507).
fn fp8_kv_quantize_row(row: &mut [f32], n_nope: usize) {
    for block in row[..n_nope].chunks_mut(64) {
        let mut amax = 0.0f32;
        for &v in block.iter() {
            let av = v.abs();
            if av > amax {
                amax = av;
            }
        }
        if amax < 1.0e-4 {
            amax = 1.0e-4;
        }
        let scale = block_scale(amax, 448.0);
        for v in block.iter_mut() {
            let mut t = *v / scale;
            if t > 448.0 {
                t = 448.0;
            } else if t < -448.0 {
                t = -448.0;
            }
            *v = e4m3_nearest(t) * scale;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Public slice API (the bit-exact parity baseline the V4 model and tests call).
// ---------------------------------------------------------------------------------------------

/// FP8 E4M3 KV-latent QAT round-trip over a flat `[rows * head_dim]` buffer.
///
/// For each `head_dim`-wide row, the leading `head_dim - n_rot` "nope" dims are E4M3
/// round-tripped per 64-element block; the trailing `n_rot` RoPE dims are left untouched.
/// `head_dim - n_rot` must be a multiple of 64. Port of `dsv4_fp8_kv_quantize_row_inplace_cpu`.
pub fn fp8_kv_quantize_rows_inplace(x: &mut [f32], head_dim: usize, n_rot: usize) -> Result<()> {
    if head_dim == 0 {
        crate::bail!("DSV4 FP8 KV round-trip: head_dim must be non-zero");
    }
    if n_rot > head_dim {
        crate::bail!("DSV4 FP8 KV round-trip: n_rot {n_rot} exceeds head_dim {head_dim}");
    }
    let n_nope = head_dim - n_rot;
    if !n_nope.is_multiple_of(64) {
        crate::bail!("DSV4 FP8 KV round-trip: nope width {n_nope} must be a multiple of 64");
    }
    if !x.len().is_multiple_of(head_dim) {
        crate::bail!(
            "DSV4 FP8 KV round-trip: buffer len {} is not a multiple of head_dim {head_dim}",
            x.len()
        );
    }
    for row in x.chunks_mut(head_dim) {
        fp8_kv_quantize_row(row, n_nope);
    }
    Ok(())
}

/// Indexer QAT round-trip over a flat `[rows * head_dim]` buffer: for each row, a 128-wide
/// normalized Hadamard transform followed by an E2M1 FP4 round-trip (per 32-element block).
/// `head_dim` must be 128. Port of `dsv4_indexer_qat_rows_inplace_cpu`.
pub fn indexer_qat_rows_inplace(x: &mut [f32], head_dim: usize) -> Result<()> {
    if head_dim != 128 {
        crate::bail!("DSV4 indexer QAT expects 128-wide rows, got {head_dim}");
    }
    if !x.len().is_multiple_of(128) {
        crate::bail!(
            "DSV4 indexer QAT: buffer len {} is not a multiple of 128",
            x.len()
        );
    }
    for row in x.chunks_mut(128) {
        hadamard128_inplace(row)?;
        fp4_act_quantize_blocks(row);
    }
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Tensor API (ergonomic entry points for the V4 model; CPU compute, device-preserving).
// ---------------------------------------------------------------------------------------------

/// Apply an in-place row kernel to the last dimension of `t`, returning an `f32` tensor on the
/// same device and shape. The compute runs on the host (the bit-exact parity path); a CUDA
/// tensor is copied in/out. The CUDA kernel is the documented speed follow-on.
fn apply_rows<F>(t: &Tensor, kernel: F) -> Result<Tensor>
where
    F: FnOnce(&mut [f32], usize) -> Result<()>,
{
    let dims: Vec<usize> = t.dims().to_vec();
    let head_dim = match dims.last() {
        Some(&d) => d,
        None => crate::bail!("DSV4 QAT: scalar tensor has no row dimension"),
    };
    let device = t.device().clone();
    let mut data = t
        .to_dtype(DType::F32)?
        .contiguous()?
        .flatten_all()?
        .to_vec1::<f32>()?;
    kernel(&mut data, head_dim)?;
    let out = Tensor::from_vec(data, dims, &Device::Cpu)?;
    if device.is_cpu() {
        Ok(out)
    } else {
        out.to_device(&device)
    }
}

/// FP8 E4M3 KV-latent QAT round-trip on the last dim of `t` (see [`fp8_kv_quantize_rows_inplace`]).
pub fn fp8_kv_quantize(t: &Tensor, n_rot: usize) -> Result<Tensor> {
    apply_rows(t, |data, head_dim| {
        fp8_kv_quantize_rows_inplace(data, head_dim, n_rot)
    })
}

/// Indexer QAT round-trip on the last dim of `t` (see [`indexer_qat_rows_inplace`]).
pub fn indexer_qat(t: &Tensor) -> Result<Tensor> {
    apply_rows(t, indexer_qat_rows_inplace)
}

// ---------------------------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use float8::F8E4M3;

    #[test]
    fn pow2_matches_ldexp() {
        for e in -149i32..=127 {
            let expected = (2f64.powi(e)) as f32;
            assert_eq!(pow2(e), expected, "pow2({e})");
        }
        assert_eq!(pow2(128), f32::INFINITY);
        assert_eq!(pow2(-150), 0.0);
    }

    /// The E4M3 grid we port must equal the canonical `float8::F8E4M3` grid for every code.
    #[test]
    fn e4m3_grid_matches_float8_oracle() {
        for code in 0u32..=126 {
            let ours = e4m3_grid_value(code);
            let oracle = F8E4M3::from_bits(code as u8).to_f32();
            assert_eq!(ours, oracle, "E4M3 grid code {code}");
        }
        assert_eq!(e4m3_grid_value(0), 0.0);
        assert_eq!(e4m3_grid_value(1), 0.001953125); // 2^-9 subnormal step
        assert_eq!(e4m3_grid_value(126), 448.0); // max finite
    }

    /// Snapping an exact grid point returns it; `float8` agrees (both round-to-nearest-even).
    #[test]
    fn e4m3_nearest_grid_points_are_fixed() {
        for code in 0u32..=126 {
            let v = e4m3_grid_value(code);
            assert_eq!(e4m3_nearest(v), v, "+grid {code}");
            assert_eq!(e4m3_nearest(-v), -v, "-grid {code}");
            // Oracle: float8 snapping a grid point is identity too.
            assert_eq!(F8E4M3::from_f32(v).to_f32(), v, "oracle grid {code}");
        }
    }

    #[test]
    fn e4m3_nearest_saturates_and_rounds_half_even() {
        assert_eq!(e4m3_nearest(500.0), 448.0);
        assert_eq!(e4m3_nearest(-1.0e9), -448.0);
        assert_eq!(e4m3_nearest(0.0), 0.0);
        // 272 is the midpoint of 256 (code 120, even) and 288 (code 121, odd) -> even -> 256.
        assert_eq!(e4m3_nearest(272.0), 256.0);
        // 304 is the midpoint of 288 (code 121, odd) and 320 (code 122, even) -> even -> 320.
        assert_eq!(e4m3_nearest(304.0), 320.0);
    }

    #[test]
    fn e2m1_nearest_known_and_half_even() {
        for (i, &v) in E2M1_VALUES.iter().enumerate() {
            assert_eq!(e2m1_nearest(v), v, "grid {i}");
            assert_eq!(e2m1_nearest(-v), -v, "neg grid {i}");
        }
        assert_eq!(e2m1_nearest(10.0), 6.0); // saturate
        assert_eq!(e2m1_nearest(-10.0), -6.0);
        assert_eq!(e2m1_nearest(0.25), 0.0); // mid(0,0.5): even idx 0
        assert_eq!(e2m1_nearest(1.25), 1.0); // mid(1.0,1.5): even idx 2
        assert_eq!(e2m1_nearest(1.75), 2.0); // mid(1.5,2.0): even idx 4
        assert_eq!(e2m1_nearest(2.5), 2.0); // mid(2,3): even idx 4
        assert_eq!(e2m1_nearest(5.0), 4.0); // mid(4,6): even idx 6
    }

    #[test]
    fn hadamard128_is_its_own_inverse() {
        // Deterministic pseudo-random input.
        let mut x = [0f32; 128];
        let mut s: u32 = 0x1234_5678;
        for v in x.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
        }
        let orig = x;
        hadamard128_inplace(&mut x).unwrap();
        hadamard128_inplace(&mut x).unwrap();
        for i in 0..128 {
            assert!((x[i] - orig[i]).abs() < 1e-5, "involution at {i}");
        }
    }

    #[test]
    fn hadamard128_dc_concentrates_in_first_bin() {
        // H_norm * [1; 128] = [sqrt(128), 0, 0, ...].
        let mut x = [1f32; 128];
        hadamard128_inplace(&mut x).unwrap();
        assert!((x[0] - 128f32.sqrt()).abs() < 1e-4, "DC bin = {}", x[0]);
        for &v in x.iter().skip(1) {
            assert!(v.abs() < 1e-4);
        }
    }

    #[test]
    fn hadamard128_wrong_len_errors() {
        let mut x = [0f32; 64];
        assert!(hadamard128_inplace(&mut x).is_err());
    }

    #[test]
    fn fp8_kv_roundtrip_exact_on_grid_values() {
        // value 2.0: amax=2 -> scale=2^-7; 2/2^-7 = 256 (grid) -> 2.0 exactly.
        let mut row = vec![2.0f32; 64];
        fp8_kv_quantize_rows_inplace(&mut row, 64, 0).unwrap();
        for &v in &row {
            assert_eq!(v, 2.0);
        }
    }

    #[test]
    fn fp8_kv_roundtrip_is_idempotent() {
        let mut row = vec![0f32; 64];
        let mut s: u32 = 0xC0FF_EE11;
        for v in row.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((s >> 7) as f32 / (1u32 << 25) as f32 - 0.5) * 3.0;
        }
        let mut once = row.clone();
        fp8_kv_quantize_rows_inplace(&mut once, 64, 0).unwrap();
        let mut twice = once.clone();
        fp8_kv_quantize_rows_inplace(&mut twice, 64, 0).unwrap();
        assert_eq!(once, twice, "FP8 KV round-trip must be idempotent");
        // And every output is within E4M3 precision of the input.
        for i in 0..64 {
            let rel = (once[i] - row[i]).abs() / row[i].abs().max(1e-6);
            assert!(rel < 0.15, "rel err {rel} at {i}");
        }
    }

    #[test]
    fn fp8_kv_leaves_rope_part_untouched() {
        // head_dim 128, n_rot 64 -> only first 64 (nope) quantized.
        let mut row = vec![0.123_456f32; 128];
        for (i, v) in row.iter_mut().enumerate() {
            *v = i as f32 * 0.01 + 0.001; // distinct values
        }
        let rope_before: Vec<f32> = row[64..].to_vec();
        let nope_before: Vec<f32> = row[..64].to_vec();
        fp8_kv_quantize_rows_inplace(&mut row, 128, 64).unwrap();
        assert_eq!(&row[64..], rope_before.as_slice(), "RoPE part changed");
        assert_ne!(
            &row[..64],
            nope_before.as_slice(),
            "nope part not quantized"
        );
    }

    #[test]
    fn fp4_act_roundtrip_is_idempotent() {
        let mut row = vec![0f32; 128];
        let mut s: u32 = 0xBEEF_1234;
        for v in row.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = ((s >> 6) as f32 / (1u32 << 26) as f32 - 0.5) * 4.0;
        }
        let mut once = row.clone();
        fp4_act_quantize_blocks(&mut once);
        let mut twice = once.clone();
        fp4_act_quantize_blocks(&mut twice);
        assert_eq!(once, twice, "FP4 act round-trip must be idempotent");
    }

    #[test]
    fn indexer_qat_equals_hadamard_then_fp4() {
        let mut row = vec![0f32; 128];
        let mut s: u32 = 0x0BAD_F00D;
        for v in row.iter_mut() {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            *v = (s >> 8) as f32 / (1u32 << 24) as f32 - 0.5;
        }
        let mut expected = row.clone();
        hadamard128_inplace(&mut expected).unwrap();
        fp4_act_quantize_blocks(&mut expected);

        let mut got = row.clone();
        indexer_qat_rows_inplace(&mut got, 128).unwrap();
        assert_eq!(got, expected, "indexer QAT must be hadamard-then-fp4");
    }

    #[test]
    fn indexer_qat_rejects_wrong_width() {
        let mut x = vec![0f32; 256];
        assert!(indexer_qat_rows_inplace(&mut x, 64).is_err());
        // 256 elements at head_dim 128 = 2 rows, OK.
        assert!(indexer_qat_rows_inplace(&mut x, 128).is_ok());
    }

    #[test]
    fn fp8_kv_rejects_unaligned_nope() {
        let mut x = vec![0f32; 100];
        // nope = 100 - 0 = 100, not a multiple of 64.
        assert!(fp8_kv_quantize_rows_inplace(&mut x, 100, 0).is_err());
    }

    #[test]
    fn tensor_fp8_matches_slice() {
        let rows = 3usize;
        let head_dim = 128usize;
        let n = rows * head_dim;
        let raw: Vec<f32> = (0..n).map(|i| (i as f32 * 0.013).sin() * 5.0).collect();

        let mut slice_out = raw.clone();
        fp8_kv_quantize_rows_inplace(&mut slice_out, head_dim, 64).unwrap();

        let t = Tensor::from_vec(raw, (rows, head_dim), &Device::Cpu).unwrap();
        let out = fp8_kv_quantize(&t, 64).unwrap();
        assert_eq!(out.dims(), &[rows, head_dim]);
        let tensor_out = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(tensor_out, slice_out);
    }

    #[test]
    fn tensor_indexer_matches_slice() {
        let rows = 2usize;
        let head_dim = 128usize;
        let n = rows * head_dim;
        let raw: Vec<f32> = (0..n).map(|i| (i as f32 * 0.007).cos()).collect();

        let mut slice_out = raw.clone();
        indexer_qat_rows_inplace(&mut slice_out, head_dim).unwrap();

        let t = Tensor::from_vec(raw, (rows, head_dim), &Device::Cpu).unwrap();
        let out = indexer_qat(&t).unwrap();
        assert_eq!(out.dims(), &[rows, head_dim]);
        let tensor_out = out.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        assert_eq!(tensor_out, slice_out);
    }
}
