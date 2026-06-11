// LOAD + DEQUANTIZE for 11 GGUF quant types ported verbatim from llama.cpp
// (ggml/src/ggml-quants.c dequantize_row_*). Each block struct is a bit-for-bit
// replica of the matching block_<t> in ggml-common.h; the `quant_format!` macro
// re-asserts the byte size against the GGUF on-disk layout at compile time.
//
// These types are decode-only: from_float panics, and matmul is performed by
// dequantizing each super-block to f32 and dotting against a Q8_0-quantized lhs
// (the generic dequant->matmul path, native matmul is out of scope).
//
// MIGRATED (Cut 2): the per-type `struct + const _ assert + impl GgmlType` triple is
// now expressed via the `quant_format!` macro (quant_format.rs). The ONLY freeform
// part — the validated `dequantize_row_*` decode body — is pasted verbatim into the
// macro's `decode:` slot; `from_float`/`vec_dot`/`vec_dot_unopt`/the size assert are
// generated identically. See quant-macro-migration.md.
#![allow(clippy::all)]

use super::iq_grids::*;
use super::k_quants::{BlockQ8_0, QK_K};
use super::quant_format::quant_format;
use half::f16;

// ggml block sizes that are not QK_K.
pub const QK1_0: usize = 128;
pub const QK_NVFP4: usize = 64;
pub const QK_NVFP4_SUB: usize = 16;

// IQ1_S / IQ1_M codebook delta (llama.cpp ggml-common.h:1121, IQ1S_DELTA 0.125f).
const IQ1S_DELTA: f32 = 0.125;

// MXFP4 nibble -> value LUT, shared with NVFP4 (llama.cpp ggml-common.h kvalues_mxfp4).
// k_quants.rs keeps its copy private, so replicate here for the NVFP4 decoder.
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

// kmask_iq2xs = {1,2,4,8,16,32,64,128}: bit j of the per-8 sign byte (computed inline).
#[inline]
fn kmask(j: usize) -> u8 {
    1u8 << j
}

// UE4M3: unsigned, 4 exp bits (bias 7), 3 mantissa bits. Returns value*0.5 to match the
// kvalues_mxfp4 convention. Bit-for-bit replica of llama.cpp ggml-impl.h:502 ggml_ue4m3_to_fp32.
fn ue4m3_to_fp32(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = ((x >> 3) & 0xF) as i32;
    let man = (x & 0x7) as i32;
    // ldexpf(m, e) == m * 2^e; Rust f32::exp2 computes 2^x.
    let raw = if exp == 0 {
        (man as f32) * (-9f32).exp2()
    } else {
        (1.0 + man as f32 / 8.0) * ((exp - 7) as f32).exp2()
    };
    raw * 0.5
}

// ============================================================================
// 1. Q1_0 — 1 bit/weight, 128 elems/block. bit set -> +d, else -d.
//    llama.cpp dequantize_row_q1_0 (ggml-quants.c:381); block_q1_0 (ggml-common.h:178).
// ============================================================================
quant_format! {
    name: BlockQ1_0,
    dtype: Q1_0,
    block_elems: QK1_0,
    byte_size: 18,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [u8; QK1_0 / 8],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK1_0), "dequantize_row_q1_0: {k} % {QK1_0} != 0");
        let nb = k / QK1_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let neg_d = -d;
            for j in 0..QK1_0 {
                let byte = xs[i].qs[j / 8];
                let bit = (byte >> (j % 8)) & 1;
                ys[i * QK1_0 + j] = if bit != 0 { d } else { neg_d };
            }
        }
    },
}

// ============================================================================
// 2. TQ2_0 — ternary 2.0625 bpw, 256 elems. value = (q&3 - 1) * d.
//    llama.cpp dequantize_row_tq2_0 (ggml-quants.c:2395); block_tq2_0 (ggml-common.h:273).
//    NOTE: field order is qs[64] THEN d.
// ============================================================================
quant_format! {
    name: BlockTQ2_0,
    dtype: TQ2_0,
    block_elems: QK_K,
    byte_size: 66,
    vec_dot: BlockQ8_0,
    fields: {
        qs: [u8; QK_K / 4],
        d: f16,
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_tq2_0: {k} % {QK_K} != 0");
        let nb = k / QK_K;
        let mut yi = 0usize;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            // qs has QK_K/4 = 64 bytes, processed in chunks of 32.
            let qslen = QK_K / 4;
            let mut j = 0usize;
            while j < qslen {
                for l in 0..4 {
                    for m in 0..32 {
                        let q = ((xs[i].qs[j + m] >> (l * 2)) & 3) as i32;
                        ys[yi] = (q - 1) as f32 * d;
                        yi += 1;
                    }
                }
                j += 32;
            }
        }
    },
}

// ============================================================================
// 3. TQ1_0 — ternary 1.6875 bpw, 256 elems. base-3 packing (5 elems / byte).
//    llama.cpp dequantize_row_tq1_0 (ggml-quants.c:2356); block_tq1_0 (ggml-common.h:265).
//    xi = ((q as u16 * 3) >> 8); value = (xi - 1) * d, with q = byte.wrapping_mul(pow3[n]).
// ============================================================================
quant_format! {
    name: BlockTQ1_0,
    dtype: TQ1_0,
    block_elems: QK_K,
    byte_size: 54,
    vec_dot: BlockQ8_0,
    fields: {
        qs: [u8; (QK_K - 4 * QK_K / 64) / 5],
        qh: [u8; QK_K / 64],
        d: f16,
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_tq1_0: {k} % {QK_K} != 0");
        let nb = k / QK_K;
        let pow3: [u8; 6] = [1, 3, 9, 27, 81, 243];
        let qslen = (QK_K - 4 * QK_K / 64) / 5; // 48
        let qhlen = QK_K / 64; // 4
        let mut yi = 0usize;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            // First loop: j in [0, qslen - qslen%32) step 32, 32 elems x 5 powers.
            let main_end = qslen - qslen % 32; // 32
            let mut j = 0usize;
            while j < main_end {
                for n in 0..5 {
                    for m in 0..32 {
                        let q = xs[i].qs[j + m].wrapping_mul(pow3[n]);
                        let xi = ((q as u16) * 3) >> 8;
                        ys[yi] = (xi as i32 - 1) as f32 * d;
                        yi += 1;
                    }
                }
                j += 32;
            }
            // Tail loop: j in [main_end, qslen) step 16, 16 elems x 5 powers.
            let mut j = main_end;
            while j < qslen {
                for n in 0..5 {
                    for m in 0..16 {
                        let q = xs[i].qs[j + m].wrapping_mul(pow3[n]);
                        let xi = ((q as u16) * 3) >> 8;
                        ys[yi] = (xi as i32 - 1) as f32 * d;
                        yi += 1;
                    }
                }
                j += 16;
            }
            // qh: 4 powers x qhlen bytes.
            for n in 0..4 {
                for j in 0..qhlen {
                    let q = xs[i].qh[j].wrapping_mul(pow3[n]);
                    let xi = ((q as u16) * 3) >> 8;
                    ys[yi] = (xi as i32 - 1) as f32 * d;
                    yi += 1;
                }
            }
        }
    },
}

// ============================================================================
// 4. NVFP4 — 4-bit microscaling float, 64 elems, 4 UE4M3 sub-scales of 16.
//    llama.cpp dequantize_row_nvfp4 (ggml-quants.c:531); block_nvfp4 (ggml-common.h:213).
// ============================================================================
quant_format! {
    name: BlockNVFP4,
    dtype: NVFP4,
    block_elems: QK_NVFP4,
    byte_size: 36,
    vec_dot: BlockQ8_0,
    fields: {
        d: [u8; QK_NVFP4 / QK_NVFP4_SUB],
        qs: [u8; QK_NVFP4 / 2],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_NVFP4), "dequantize_row_nvfp4: {k} % {QK_NVFP4} != 0");
        let nb = k / QK_NVFP4;
        let n_sub = QK_NVFP4 / QK_NVFP4_SUB; // 4
        for i in 0..nb {
            for s in 0..n_sub {
                let d = ue4m3_to_fp32(xs[i].d[s]);
                let yb = i * QK_NVFP4 + s * QK_NVFP4_SUB;
                for j in 0..QK_NVFP4_SUB / 2 {
                    let q = xs[i].qs[s * (QK_NVFP4_SUB / 2) + j];
                    let v0 = KVALUES_MXFP4[(q & 0x0F) as usize];
                    let v1 = KVALUES_MXFP4[(q >> 4) as usize];
                    ys[yb + j] = v0 as f32 * d;
                    ys[yb + j + QK_NVFP4_SUB / 2] = v1 as f32 * d;
                }
            }
        }
    },
}

// ============================================================================
// 5. IQ2_XXS — "true" 2-bit, 256 elems. Grid (uint8 bytes) + ksigns.
//    llama.cpp dequantize_row_iq2_xxs (ggml-quants.c:2416); block_iq2_xxs (ggml-common.h:371).
//    db = d * (0.5 + scale) * 0.25, scale = aux32[1] >> 28.
// ============================================================================
quant_format! {
    name: BlockIQ2xxs,
    dtype: IQ2_XXS,
    block_elems: QK_K,
    byte_size: 66,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [u16; QK_K / 8],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_iq2_xxs: {k} % {QK_K} != 0");
        let nb = k / QK_K;
        let mut yi = 0usize;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for ib32 in 0..QK_K / 32 {
                // ggml memcpy's 8 bytes (2 u32) from qs+4*ib32 (uint16_t*) into aux32[2].
                // aux32[0] = u16[0]|u16[1]<<16, aux32[1] = u16[2]|u16[3]<<16.
                let q = &xs[i].qs[4 * ib32..4 * ib32 + 4];
                let aux32_0 = q[0] as u32 | ((q[1] as u32) << 16);
                let aux32_1 = q[2] as u32 | ((q[3] as u32) << 16);
                // aux8[l] (l in 0..4) = low 4 bytes of aux32_0 (little-endian).
                let aux8 = aux32_0.to_le_bytes();
                let db = d * (0.5 + (aux32_1 >> 28) as f32) * 0.25;
                for l in 0..4 {
                    let entry = IQ2XXS_GRID[aux8[l] as usize];
                    let signs = KSIGNS_IQ2XS[((aux32_1 >> (7 * l)) & 127) as usize];
                    for j in 0..8 {
                        let g = (entry >> (8 * j)) as u8 as f32;
                        let sign = if signs & kmask(j) != 0 { -1.0 } else { 1.0 };
                        ys[yi + j] = db * g * sign;
                    }
                    yi += 8;
                }
            }
        }
    },
}

// ============================================================================
// 6. IQ2_XS — 2.3125 bpw, 256 elems. Grid (uint8) + ksigns + per-block scales.
//    llama.cpp dequantize_row_iq2_xs (ggml-quants.c:2444); block_iq2_xs (ggml-common.h:378).
//    idx = qs&511, signs = qs>>9; db = d*(0.5 + nibble)*0.25 (two nibbles per scales byte).
// ============================================================================
quant_format! {
    name: BlockIQ2xs,
    dtype: IQ2_XS,
    block_elems: QK_K,
    byte_size: 74,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [u16; QK_K / 8],
        scales: [u8; QK_K / 32],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_iq2_xs: {k} % {QK_K} != 0");
        let nb = k / QK_K;
        let mut yi = 0usize;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for ib32 in 0..QK_K / 32 {
                let sc = xs[i].scales[ib32];
                let db = [
                    d * (0.5 + (sc & 0xf) as f32) * 0.25,
                    d * (0.5 + (sc >> 4) as f32) * 0.25,
                ];
                for l in 0..4 {
                    let qv = xs[i].qs[4 * ib32 + l];
                    let entry = IQ2XS_GRID[(qv & 511) as usize];
                    let signs = KSIGNS_IQ2XS[(qv >> 9) as usize];
                    let dl = db[l / 2];
                    for j in 0..8 {
                        let g = (entry >> (8 * j)) as u8 as f32;
                        let sign = if signs & kmask(j) != 0 { -1.0 } else { 1.0 };
                        ys[yi + j] = dl * g * sign;
                    }
                    yi += 8;
                }
            }
        }
    },
}

// ============================================================================
// 7. IQ2_S — 2.5625 bpw, 256 elems. Grid (uint8). Signs are the UPPER 32 bytes of qs.
//    llama.cpp dequantize_row_iq2_s (ggml-quants.c:2471); block_iq2_s (ggml-common.h:386).
//    idx = qs[l] | ((qh[ib32] << (8-2l)) & 0x300); signs = qs[QK_K/8 + ...].
// ============================================================================
quant_format! {
    name: BlockIQ2s,
    dtype: IQ2_S,
    block_elems: QK_K,
    byte_size: 82,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [u8; QK_K / 4],
        qh: [u8; QK_K / 32],
        scales: [u8; QK_K / 32],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_iq2_s: {k} % {QK_K} != 0");
        let nb = k / QK_K;
        let mut yi = 0usize;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            // qs pointer walks the low 32 bytes (4 per ib32); signs walks the upper 32 bytes.
            let mut qs_off = 0usize; // offset into the low half of qs
            let signs_base = QK_K / 8; // 32: start of signs within qs
            let mut signs_off = 0usize;
            for ib32 in 0..QK_K / 32 {
                let sc = xs[i].scales[ib32];
                let db = [
                    d * (0.5 + (sc & 0xf) as f32) * 0.25,
                    d * (0.5 + (sc >> 4) as f32) * 0.25,
                ];
                let qh = xs[i].qh[ib32];
                for l in 0..4 {
                    let dl = db[l / 2];
                    let idx = xs[i].qs[qs_off + l] as usize
                        | (((qh as usize) << (8 - 2 * l)) & 0x300);
                    let entry = IQ2S_GRID[idx];
                    let signs = xs[i].qs[signs_base + signs_off + l];
                    for j in 0..8 {
                        let g = (entry >> (8 * j)) as u8 as f32;
                        let sign = if signs & kmask(j) != 0 { -1.0 } else { 1.0 };
                        ys[yi + j] = dl * g * sign;
                    }
                    yi += 8;
                }
                qs_off += 4;
                signs_off += 4;
            }
        }
    },
}

// ============================================================================
// 8. IQ3_XXS — 3.0625 bpw, 256 elems. Grid is u32 (4 uint8 bytes each).
//    llama.cpp dequantize_row_iq3_xxs (ggml-quants.c:2503); block_iq3_xxs (ggml-common.h:397).
//    scales_and_signs = qs + QK_K/4; db = d*(0.5 + (aux32>>28))*0.5 (NOTE *0.5).
// ============================================================================
quant_format! {
    name: BlockIQ3xxs,
    dtype: IQ3_XXS,
    block_elems: QK_K,
    byte_size: 98,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [u8; 3 * QK_K / 8],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_iq3_xxs: {k} % {QK_K} != 0");
        let nb = k / QK_K;
        let scales_base = QK_K / 4; // 64: start of scales_and_signs within qs
        let mut yi = 0usize;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let mut qs_off = 0usize; // walks the grid-index half of qs (8 per ib32)
            for ib32 in 0..QK_K / 32 {
                // aux32 = u32 read from scales_and_signs + 4*ib32 (little-endian).
                let sb = scales_base + 4 * ib32;
                let aux32 = u32::from_le_bytes([
                    xs[i].qs[sb],
                    xs[i].qs[sb + 1],
                    xs[i].qs[sb + 2],
                    xs[i].qs[sb + 3],
                ]);
                let db = d * (0.5 + (aux32 >> 28) as f32) * 0.5;
                for l in 0..4 {
                    let signs = KSIGNS_IQ2XS[((aux32 >> (7 * l)) & 127) as usize];
                    let g1 = IQ3XXS_GRID[xs[i].qs[qs_off + 2 * l] as usize];
                    let g2 = IQ3XXS_GRID[xs[i].qs[qs_off + 2 * l + 1] as usize];
                    for j in 0..4 {
                        let v1 = (g1 >> (8 * j)) as u8 as f32;
                        let v2 = (g2 >> (8 * j)) as u8 as f32;
                        let s1 = if signs & kmask(j) != 0 { -1.0 } else { 1.0 };
                        let s2 = if signs & kmask(j + 4) != 0 { -1.0 } else { 1.0 };
                        ys[yi + j] = db * v1 * s1;
                        ys[yi + j + 4] = db * v2 * s2;
                    }
                    yi += 8;
                }
                qs_off += 8;
            }
        }
    },
}

// ============================================================================
// 9. IQ3_S — 3.3125 bpw, 256 elems. Grid is u32. Paired sub-blocks; qh extends idx bit8.
//    llama.cpp dequantize_row_iq3_s (ggml-quants.c:2535); block_iq3_s (ggml-common.h:405).
//    db = d * (1 + 2*scale)  (DIFFERENT formula).
// ============================================================================
quant_format! {
    name: BlockIQ3s,
    dtype: IQ3_S,
    block_elems: QK_K,
    byte_size: 110,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [u8; QK_K / 4],
        qh: [u8; QK_K / 32],
        signs: [u8; QK_K / 8],
        scales: [u8; QK_K / 64],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_iq3_s: {k} % {QK_K} != 0");
        let nb = k / QK_K;
        let mut yi = 0usize;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let mut qs_off = 0usize; // walks qs in steps of 8
            let mut signs_off = 0usize; // walks signs in steps of 4
            let mut qh_off = 0usize; // walks qh in steps of 2
            // ib32 runs 0,2,4,6; each iteration emits two 32-elem sub-blocks.
            let mut ib32 = 0usize;
            while ib32 < QK_K / 32 {
                let sc = xs[i].scales[ib32 / 2];
                let db1 = d * (1.0 + 2.0 * (sc & 0xf) as f32);
                let db2 = d * (1.0 + 2.0 * (sc >> 4) as f32);
                // First sub-block: qh[qh_off].
                let qh0 = xs[i].qh[qh_off] as usize;
                for l in 0..4 {
                    let idx1 = xs[i].qs[qs_off + 2 * l] as usize | ((qh0 << (8 - 2 * l)) & 256);
                    let idx2 = xs[i].qs[qs_off + 2 * l + 1] as usize | ((qh0 << (7 - 2 * l)) & 256);
                    let g1 = IQ3S_GRID[idx1];
                    let g2 = IQ3S_GRID[idx2];
                    let signs = xs[i].signs[signs_off + l];
                    for j in 0..4 {
                        let v1 = (g1 >> (8 * j)) as u8 as f32;
                        let v2 = (g2 >> (8 * j)) as u8 as f32;
                        let s1 = if signs & kmask(j) != 0 { -1.0 } else { 1.0 };
                        let s2 = if signs & kmask(j + 4) != 0 { -1.0 } else { 1.0 };
                        ys[yi + j] = db1 * v1 * s1;
                        ys[yi + j + 4] = db1 * v2 * s2;
                    }
                    yi += 8;
                }
                qs_off += 8;
                signs_off += 4;
                // Second sub-block: qh[qh_off + 1].
                let qh1 = xs[i].qh[qh_off + 1] as usize;
                for l in 0..4 {
                    let idx1 = xs[i].qs[qs_off + 2 * l] as usize | ((qh1 << (8 - 2 * l)) & 256);
                    let idx2 = xs[i].qs[qs_off + 2 * l + 1] as usize | ((qh1 << (7 - 2 * l)) & 256);
                    let g1 = IQ3S_GRID[idx1];
                    let g2 = IQ3S_GRID[idx2];
                    let signs = xs[i].signs[signs_off + l];
                    for j in 0..4 {
                        let v1 = (g1 >> (8 * j)) as u8 as f32;
                        let v2 = (g2 >> (8 * j)) as u8 as f32;
                        let s1 = if signs & kmask(j) != 0 { -1.0 } else { 1.0 };
                        let s2 = if signs & kmask(j + 4) != 0 { -1.0 } else { 1.0 };
                        ys[yi + j] = db2 * v1 * s1;
                        ys[yi + j + 4] = db2 * v2 * s2;
                    }
                    yi += 8;
                }
                qh_off += 2;
                qs_off += 8;
                signs_off += 4;
                ib32 += 2;
            }
        }
    },
}

// ============================================================================
// 10. IQ1_S — 1.5625 bpw, 256 elems. Grid bytes are SIGNED i8. qh is [u16;8].
//     llama.cpp dequantize_row_iq1_s (ggml-quants.c:2578); block_iq1_s (ggml-common.h:415).
//     dl = d*(2*((qh>>12)&7)+1); delta = +/-0.125 by qh&0x8000; y = dl*(grid_i8 + delta).
// ============================================================================
quant_format! {
    name: BlockIQ1s,
    dtype: IQ1_S,
    block_elems: QK_K,
    byte_size: 50,
    vec_dot: BlockQ8_0,
    fields: {
        d: f16,
        qs: [u8; QK_K / 8],
        qh: [u16; QK_K / 32],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_iq1_s: {k} % {QK_K} != 0");
        let nb = k / QK_K;
        let mut yi = 0usize;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            let mut qs_off = 0usize; // walks qs in steps of 4
            for ib in 0..QK_K / 32 {
                let qh = xs[i].qh[ib];
                let dl = d * (2.0 * ((qh >> 12) & 7) as f32 + 1.0);
                let delta = if qh & 0x8000 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA };
                for l in 0..4 {
                    let idx = xs[i].qs[qs_off + l] as usize | ((((qh >> (3 * l)) & 7) as usize) << 8);
                    let entry = IQ1S_GRID[idx];
                    for j in 0..8 {
                        // Grid bytes are SIGNED for IQ1_S/IQ1_M.
                        let g = (entry >> (8 * j)) as u8 as i8 as f32;
                        ys[yi + j] = dl * (g + delta);
                    }
                    yi += 8;
                }
                qs_off += 4;
            }
        }
    },
}

// ============================================================================
// 11. IQ1_M — 1.75 bpw, 256 elems. Grid bytes SIGNED (shared IQ1S_GRID). NO top-level d:
//     scale is reconstructed from 4 nibbles spread across scales[8] (as [u16;4]).
//     llama.cpp dequantize_row_iq1_m (ggml-quants.c:2603); block_iq1_m (ggml-common.h:423).
// ============================================================================
quant_format! {
    name: BlockIQ1m,
    dtype: IQ1_M,
    block_elems: QK_K,
    byte_size: 56,
    vec_dot: BlockQ8_0,
    fields: {
        qs: [u8; QK_K / 8],
        qh: [u8; QK_K / 16],
        scales: [u8; QK_K / 32],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_K), "dequantize_row_iq1_m: {k} % {QK_K} != 0");
        let nb = k / QK_K;
        let mut yi = 0usize;
        for i in 0..nb {
            // scales[8] reinterpreted as [u16;4] (little-endian).
            let sc: [u16; 4] = [
                u16::from_le_bytes([xs[i].scales[0], xs[i].scales[1]]),
                u16::from_le_bytes([xs[i].scales[2], xs[i].scales[3]]),
                u16::from_le_bytes([xs[i].scales[4], xs[i].scales[5]]),
                u16::from_le_bytes([xs[i].scales[6], xs[i].scales[7]]),
            ];
            let scale_u16 = (sc[0] >> 12)
                | ((sc[1] >> 8) & 0x00f0)
                | ((sc[2] >> 4) & 0x0f00)
                | (sc[3] & 0xf000);
            let d = f16::from_bits(scale_u16).to_f32();
            let mut qs_off = 0usize; // walks qs in steps of 4
            let mut qh_off = 0usize; // walks qh in steps of 2
            for ib in 0..QK_K / 32 {
                let dl1 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2))) & 0x7) as f32 + 1.0);
                let dl2 = d * (2.0 * ((sc[ib / 2] >> (6 * (ib % 2) + 3)) & 0x7) as f32 + 1.0);
                let qh0 = xs[i].qh[qh_off] as usize;
                let qh1 = xs[i].qh[qh_off + 1] as usize;
                let idx = [
                    xs[i].qs[qs_off] as usize | ((qh0 << 8) & 0x700),
                    xs[i].qs[qs_off + 1] as usize | ((qh0 << 4) & 0x700),
                    xs[i].qs[qs_off + 2] as usize | ((qh1 << 8) & 0x700),
                    xs[i].qs[qs_off + 3] as usize | ((qh1 << 4) & 0x700),
                ];
                let delta = [
                    if qh0 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh0 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh1 & 0x08 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                    if qh1 & 0x80 != 0 { -IQ1S_DELTA } else { IQ1S_DELTA },
                ];
                for l in 0..2 {
                    let entry = IQ1S_GRID[idx[l]];
                    for j in 0..8 {
                        let g = (entry >> (8 * j)) as u8 as i8 as f32;
                        ys[yi + j] = dl1 * (g + delta[l]);
                    }
                    yi += 8;
                }
                for l in 2..4 {
                    let entry = IQ1S_GRID[idx[l]];
                    for j in 0..8 {
                        let g = (entry >> (8 * j)) as u8 as i8 as f32;
                        ys[yi + j] = dl2 * (g + delta[l]);
                    }
                    yi += 8;
                }
                qs_off += 4;
                qh_off += 2;
            }
        }
    },
}
