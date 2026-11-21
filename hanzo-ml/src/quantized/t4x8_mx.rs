//! Blackwell Pairwise 4:8 Structured Sparse Ternary with MX Scale (`T4x8Mx64`).
//!
//! # Architecture & Theoretical Foundations
//! Blackwell Tensor Cores (`tcgen05.mma.sp`) introduce hardware-accelerated block-scaled
//! sparse MMA with **pairwise 4:8 structured sparsity**. Out of every 8 consecutive logical
//! weights, exactly two 2-element pairs are active and two pairs are zeroed.
//!
//! Because there are C(4, 2) = 6 valid 2-element pair configurations per 8 weights, the
//! sparsity topology requires only 3 bits per octet (24 bits across 64 weights).
//! Combined with 1 sign bit per active weight (32 bits across 64 weights) and an 8-bit
//! UE8M0 block exponent scale (8 bits per 64 weights), the persistent storage is exactly:
//!
//!     32 (signs) + 24 (pair topologies) + 8 (UE8M0 scale) = 64 bits = 8 bytes per 64 weights
//!
//! This achieves **1.0000 bpw persistent storage** in unified LPDDR5X memory (Spark GB10),
//! streaming a 27B model in ~3.4 GB (80 full model sweeps/sec across Spark's 273 GB/s bus)
//! while matching the Blackwell native sparse Tensor Core register layout without expansion.

use super::k_quants::BlockQ8_0;
use super::quant_format::quant_format;

pub const QK_T4X8_MX: usize = 64;

/// Pair index choices out of 4 adjacent pairs: C(4, 2) = 6 combinations.
pub const PAIR_COMBOS: [(usize, usize); 6] = [
    (0, 1), // combo 0: elements [0, 1, 2, 3]
    (0, 2), // combo 1: elements [0, 1, 4, 5]
    (0, 3), // combo 2: elements [0, 1, 6, 7]
    (1, 2), // combo 3: elements [2, 3, 4, 5]
    (1, 3), // combo 4: elements [2, 3, 6, 7]
    (2, 3), // combo 5: elements [4, 5, 6, 7]
];

/// Active element indices within an 8-element octet for each of the 6 combinations.
pub const PAIR_ACTIVE_INDICES: [[usize; 4]; 6] = [
    [0, 1, 2, 3],
    [0, 1, 4, 5],
    [0, 1, 6, 7],
    [2, 3, 4, 5],
    [2, 3, 6, 7],
    [4, 5, 6, 7],
];

/// 4-bit pair active mask LUT packed into 24 bits:
/// combo 0: 0b0011 (0x3), combo 1: 0b0101 (0x5), combo 2: 0b1001 (0x9),
/// combo 3: 0b0110 (0x6), combo 4: 0b1010 (0xA), combo 5: 0b1100 (0xC).
pub const LUT_PAIR_MASK_24: u32 = 0xCA6953;

/// 8-bit element active mask LUT packed into 48 bits (scalar u64):
/// combo 0: 0x0F, combo 1: 0x33, combo 2: 0xC3,
/// combo 3: 0x3C, combo 4: 0xCC, combo 5: 0xF0.
pub const LUT_ELEM_MASK_48: u64 = 0xF0CC3CC3330F;

quant_format! {
    /// T4x8Mx64 — 1.000 bpw Blackwell Pairwise 4:8 Structured Sparse Ternary.
    ///
    /// Memory layout (8 bytes total):
    /// - `signs`: u32
    ///     8 octets x 4 active weights = 32 sign bits.
    ///     Bit (4*g + k) = 0 => +scale, Bit (4*g + k) = 1 => -scale.
    /// - `pairs_scale`: u32
    ///     bits [0..23]:  8 octets x 3-bit pair combination selector (0..5).
    ///     bits [24..31]: UE8M0 unsigned exponent scale (value = 2^(e - 127)).
    name: BlockT4x8Mx64,
    dtype: T4X8_MX,
    block_elems: QK_T4X8_MX,
    byte_size: 8,
    vec_dot: BlockQ8_0,
    fields: {
        signs: u32,
        pairs_scale: u32,
    },
    decode: |xs, ys| {
        dequantize_row_t4x8_mx(xs, ys);
    },
}

/// Dequantize a contiguous row of `BlockT4x8Mx64` blocks to f32.
#[inline]
pub fn dequantize_row_t4x8_mx(xs: &[BlockT4x8Mx64], ys: &mut [f32]) {
    let k = ys.len();
    debug_assert!(
        k % QK_T4X8_MX == 0,
        "dequantize_row_t4x8_mx: length {k} must be multiple of {QK_T4X8_MX}"
    );
    let nb = k / QK_T4X8_MX;
    debug_assert!(
        xs.len() >= nb,
        "dequantize_row_t4x8_mx: xs.len() {} < nb {}",
        xs.len(),
        nb
    );

    for (bx, block) in xs.iter().take(nb).enumerate() {
        let out_base = bx * QK_T4X8_MX;
        let out_slice = &mut ys[out_base..out_base + QK_T4X8_MX];

        let scale_exp = (block.pairs_scale >> 24) as u8;
        if scale_exp == 0 {
            out_slice.fill(0.0f32);
            continue;
        }

        let scale = f32::from_bits((scale_exp as u32) << 23);
        let neg_scale = -scale;

        out_slice.fill(0.0f32);

        let pairs_scale = block.pairs_scale;
        let signs = block.signs;

        for g in 0..8 {
            let combo = ((pairs_scale >> (3 * g)) & 0x7) as usize;
            let combo = if combo > 5 { 0 } else { combo };
            let active_indices = &PAIR_ACTIVE_INDICES[combo];
            let sign_bits = (signs >> (4 * g)) & 0xF;

            let octet_offset = g * 8;
            for (k, &idx) in active_indices.iter().enumerate() {
                let s = (sign_bits >> k) & 1;
                out_slice[octet_offset + idx] = if s == 1 { neg_scale } else { scale };
            }
        }
    }
}

/// Project / quantize dense float weights into `BlockT4x8Mx64`.
///
/// For each octet of 8 elements:
/// 1. Computes pair energy E_p = |w_{2p}| + |w_{2p+1}| for p in 0..3.
/// 2. Selects top-2 energetic pairs (satisfying 4:8 pairwise sparsity).
/// 3. Computes optimal UE8M0 power-of-2 scale matching active weight magnitudes.
/// 4. Emits signs and 3-bit pair selectors.
pub fn quantize_row_t4x8_mx(src: &[f32], dst: &mut [BlockT4x8Mx64]) {
    let k = src.len();
    assert!(
        k % QK_T4X8_MX == 0,
        "quantize_row_t4x8_mx: length {k} must be multiple of {QK_T4X8_MX}"
    );
    let nb = k / QK_T4X8_MX;
    assert!(
        dst.len() >= nb,
        "quantize_row_t4x8_mx: dst buffer too small ({nb} blocks required)"
    );

    for b in 0..nb {
        let block_src = &src[b * QK_T4X8_MX..(b + 1) * QK_T4X8_MX];
        let mut combos = [0usize; 8];
        let mut active_indices = [0usize; 32];
        let mut active_count = 0;

        for g in 0..8 {
            let octet = &block_src[g * 8..(g + 1) * 8];
            let pair_energies = [
                octet[0].abs() + octet[1].abs(),
                octet[2].abs() + octet[3].abs(),
                octet[4].abs() + octet[5].abs(),
                octet[6].abs() + octet[7].abs(),
            ];

            // Rank pairs by energy
            let mut pair_order = [0, 1, 2, 3];
            pair_order.sort_unstable_by(|&a, &b| {
                pair_energies[b]
                    .partial_cmp(&pair_energies[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            // Keep top 2 pairs, sort them so p0 < p1
            let mut p0 = pair_order[0];
            let mut p1 = pair_order[1];
            if p0 > p1 {
                std::mem::swap(&mut p0, &mut p1);
            }

            // Map (p0, p1) to combo index
            let combo = match (p0, p1) {
                (0, 1) => 0,
                (0, 2) => 1,
                (0, 3) => 2,
                (1, 2) => 3,
                (1, 3) => 4,
                (2, 3) => 5,
                _ => unreachable!(),
            };
            combos[g] = combo;

            let active = &PAIR_ACTIVE_INDICES[combo];
            for &a in active {
                active_indices[active_count] = g * 8 + a;
                active_count += 1;
            }
        }

        // Compute average active weight magnitude
        let mut sum_mag = 0.0f32;
        for &idx in &active_indices {
            sum_mag += block_src[idx].abs();
        }
        let mean_mag = sum_mag / 32.0f32;

        if mean_mag <= 1e-12 {
            dst[b] = BlockT4x8Mx64 {
                signs: 0,
                pairs_scale: 0,
            };
            continue;
        }

        // Optimal UE8M0 scale exponent
        let exp = (mean_mag.log2() + 127.0).round() as i32;
        let exp = exp.clamp(1, 254) as u32;

        let mut pairs_scale = exp << 24;
        for g in 0..8 {
            pairs_scale |= ((combos[g] as u32) & 0x7) << (3 * g);
        }

        let mut signs = 0u32;
        for g in 0..8 {
            let combo = combos[g];
            let active = &PAIR_ACTIVE_INDICES[combo];
            let mut s_g = 0u32;
            for (k, &a) in active.iter().enumerate() {
                let val = block_src[g * 8 + a];
                let s = if val < 0.0 { 1u32 } else { 0u32 };
                s_g |= s << k;
            }
            signs |= s_g << (4 * g);
        }

        dst[b] = BlockT4x8Mx64 { signs, pairs_scale };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_t4x8_mx64_layout_and_size() {
        assert_eq!(std::mem::size_of::<BlockT4x8Mx64>(), 8);
        assert_eq!(std::mem::align_of::<BlockT4x8Mx64>(), 4);
    }

    #[test]
    fn test_t4x8_mx64_lut_consistency() {
        for combo in 0..6 {
            let (p0, p1) = PAIR_COMBOS[combo];
            let expected_pair_mask = (1u32 << p0) | (1u32 << p1);
            let lut_pair_mask = (LUT_PAIR_MASK_24 >> (4 * combo)) & 0xF;
            assert_eq!(lut_pair_mask, expected_pair_mask);

            let expected_elem_mask = (3u64 << (2 * p0)) | (3u64 << (2 * p1));
            let lut_elem_mask = (LUT_ELEM_MASK_48 >> (8 * combo)) & 0xFF;
            assert_eq!(lut_elem_mask, expected_elem_mask);

            let active = PAIR_ACTIVE_INDICES[combo];
            assert_eq!(active, [2 * p0, 2 * p0 + 1, 2 * p1, 2 * p1 + 1]);
        }
    }

    #[test]
    fn test_roundtrip_quantize_dequantize() {
        let mut original = [0.0f32; 64];
        // Populate octet 0: pairs 0 and 2 are active, pairs 1 and 3 are zero
        original[0] = 0.5;
        original[1] = -0.5;
        original[2] = 0.0;
        original[3] = 0.0;
        original[4] = -0.5;
        original[5] = 0.5;
        original[6] = 0.0;
        original[7] = 0.0;

        // Octets 1..7: similar synthetic pattern
        for g in 1..8 {
            for i in 0..8 {
                original[g * 8 + i] = if i % 2 == 0 { 0.5 } else { -0.5 };
            }
        }

        let mut blocks = [BlockT4x8Mx64 {
            signs: 0,
            pairs_scale: 0,
        }; 1];
        quantize_row_t4x8_mx(&original, &mut blocks);

        let mut decoded = [0.0f32; 64];
        dequantize_row_t4x8_mx(&blocks, &mut decoded);

        // Verify pairwise 4:8 sparsity in octet 0
        assert_eq!(decoded[0], 0.5);
        assert_eq!(decoded[1], -0.5);
        assert_eq!(decoded[2], 0.0);
        assert_eq!(decoded[3], 0.0);
        assert_eq!(decoded[4], -0.5);
        assert_eq!(decoded[5], 0.5);
        assert_eq!(decoded[6], 0.0);
        assert_eq!(decoded[7], 0.0);

        // Exactly 32 nonzeros per block
        let nonzeros = decoded.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(nonzeros, 32);
    }
}
