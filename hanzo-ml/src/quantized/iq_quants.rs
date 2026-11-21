// LOAD + DEQUANTIZE for the GGUF quant types ported verbatim from llama.cpp
// (ggml/src/ggml-quants.c dequantize_row_*), the Prism (Bonsai 2) ternary family
// PQ2_0/PTQ1_0, and the ROCmFPX fork ROCMFP4 types. Each block struct is a bit-for-bit
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
// Prism (Bonsai 2) ternary family: both are group-128 (ggml-common.h QK_PQ2_0 / QK_PTQ1_0).
pub const QK_PQ2_0: usize = 128;
pub const QK_PTQ1_0: usize = 128;

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

// ============================================================================
// 6. ROCMFP4 — 4-bit, 32 elems/block, UE4M3 half-scales.
//    ROCmFPX fork (charlie12345) types: GGML_TYPE_Q4_0_ROCMFP4 = 100 (dual-scale,
//    18 B), GGML_TYPE_Q4_0_ROCMFP4_FAST = 101 (single-scale, 17 B).
//    Codebook10: E2M1-derived, max level retuned 12 -> 10.
//    qs[j] = code(w_j) | code(w_j+16) << 4: low nibble uses e[0] (fast: e),
//    high nibble uses e[1]. No bias, no secondary scale.
// ============================================================================
pub const QK_ROCMFP4: usize = 32;

// kvalues_rocmfp4 (fork ggml-common.h Codebook10).
const KVALUES_ROCMFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10];

// HALF(e): the UE4M3 byte divided by two. 0x7f (NaN) and sign-set bytes are
// invalid per the fork's row validator and decode to zero.
fn rocmfp4_half_to_f32(e: u8) -> f32 {
    if e > 0x7e {
        return 0.0;
    }
    ue4m3_to_fp32(e)
}

quant_format! {
    name: BlockROCMFP4,
    dtype: ROCMFP4,
    block_elems: QK_ROCMFP4,
    byte_size: 18,
    vec_dot: BlockQ8_0,
    fields: {
        qs: [u8; 16],
        e: [u8; 2],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_ROCMFP4), "dequantize_row_rocmfp4: {k} % {QK_ROCMFP4} != 0");
        let nb = k / QK_ROCMFP4;
        for i in 0..nb {
            let d0 = rocmfp4_half_to_f32(xs[i].e[0]);
            let d1 = rocmfp4_half_to_f32(xs[i].e[1]);
            for j in 0..16 {
                let q = xs[i].qs[j];
                ys[i * QK_ROCMFP4 + j] = KVALUES_ROCMFP4[(q & 0x0F) as usize] as f32 * d0;
                ys[i * QK_ROCMFP4 + j + 16] = KVALUES_ROCMFP4[(q >> 4) as usize] as f32 * d1;
            }
        }
    },
}

quant_format! {
    name: BlockROCMFP4Fast,
    dtype: ROCMFP4_FAST,
    block_elems: QK_ROCMFP4,
    byte_size: 17,
    vec_dot: BlockQ8_0,
    fields: {
        qs: [u8; 16],
        e: u8,
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_ROCMFP4), "dequantize_row_rocmfp4_fast: {k} % {QK_ROCMFP4} != 0");
        let nb = k / QK_ROCMFP4;
        for i in 0..nb {
            let d = rocmfp4_half_to_f32(xs[i].e);
            for j in 0..16 {
                let q = xs[i].qs[j];
                ys[i * QK_ROCMFP4 + j] = KVALUES_ROCMFP4[(q & 0x0F) as usize] as f32 * d;
                ys[i * QK_ROCMFP4 + j + 16] = KVALUES_ROCMFP4[(q >> 4) as usize] as f32 * d;
            }
        }
    },
}

// ============================================================================
// 12. PQ2_0 - Prism (Bonsai 2) 2-bit ternary, 128 elems. 00=-1, 01=0, 10=+1, 11=+2:
//     value = (code - 1) * d (asymmetric; the +2 slot is what buys the Hadamard basis).
//     block_pq2_0 (ggml-common.h): d FIRST, then 32 qs bytes -> 34 B.
//     ProCreations dequantize_row_pq2_0 (ggml-quants.c:494).
// ============================================================================
quant_format! {
    name: BlockPQ2_0,
    dtype: PQ2_0,
    block_elems: QK_PQ2_0,
    byte_size: 34,
    vec_dot: BlockQ8_0,
    vec_dot_fn: vec_dot_pq2_0_q8_0,
    fields: {
        d: f16,
        qs: [u8; QK_PQ2_0 / 4],
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_PQ2_0), "dequantize_row_pq2_0: {k} % {QK_PQ2_0} != 0");
        let nb = k / QK_PQ2_0;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            for j in 0..QK_PQ2_0 {
                let byte = xs[i].qs[j / 4];
                let q = ((byte >> ((j % 4) * 2)) & 3) as i32;
                ys[i * QK_PQ2_0 + j] = (q - 1) as f32 * d;
            }
        }
    },
}

// ============================================================================
// 13. PTQ1_0 - Prism (Bonsai 2) 1-bit ternary, 128 elems. Base-3 packing: 5 trits per
//     qs byte + 4 per qh byte, value = (trit - 1) * d. Staged element order (ggml-quants.c
//     ptq1_0_stages {32,16,8}): e = n*16+m over qs[0..15], e = 80+n*8+m over qs[16..23],
//     e = 120+n*2+m over qh. xi = ((byte.wrapping_mul(3^n) as u16) * 3) >> 8.
//     block_ptq1_0 (ggml-common.h): scale d at the END -> 24 + 2 + 2 = 28 B.
// ============================================================================
quant_format! {
    name: BlockPTQ1_0,
    dtype: PTQ1_0,
    block_elems: QK_PTQ1_0,
    byte_size: 28,
    vec_dot: BlockQ8_0,
    vec_dot_fn: vec_dot_ptq1_0_q8_0,
    fields: {
        qs: [u8; (QK_PTQ1_0 - 4 * QK_PTQ1_0 / 64) / 5],
        qh: [u8; QK_PTQ1_0 / 64],
        d: f16,
    },
    decode: |xs, ys| {
        let k = ys.len();
        debug_assert!(k.is_multiple_of(QK_PTQ1_0), "dequantize_row_ptq1_0: {k} % {QK_PTQ1_0} != 0");
        let nb = k / QK_PTQ1_0;
        let pow3: [u8; 6] = [1, 3, 9, 27, 81, 243];
        let mut yi = 0usize;
        for i in 0..nb {
            let d = xs[i].d.to_f32();
            // 16-byte chunks: 5 digits x 16 elements.
            let mut j = 0usize;
            while j + 16 <= 24 {
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
            // 8-byte tail chunk: 5 digits x 8 elements.
            while j + 8 <= 24 {
                for n in 0..5 {
                    for m in 0..8 {
                        let q = xs[i].qs[j + m].wrapping_mul(pow3[n]);
                        let xi = ((q as u16) * 3) >> 8;
                        ys[yi] = (xi as i32 - 1) as f32 * d;
                        yi += 1;
                    }
                }
                j += 8;
            }
            // qh: 4 digits x 2 elements.
            for n in 0..4 {
                for m in 0..2 {
                    let q = xs[i].qh[m].wrapping_mul(pow3[n]);
                    let xi = ((q as u16) * 3) >> 8;
                    ys[yi] = (xi as i32 - 1) as f32 * d;
                    yi += 1;
                }
            }
        }
    },
}

// ============================================================================
// Fast SIMD + integer-accumulating CPU vec_dot for PQ2_0 and PTQ1_0 vs Q8_0
// ============================================================================

#[inline]
pub(crate) fn vec_dot_pq2_0_q8_0(n: usize, xs: &[BlockPQ2_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_PQ2_0),
        "vec_dot BlockPQ2_0: {n} not divisible by {QK_PQ2_0}"
    );
    #[cfg(target_arch = "aarch64")]
    {
        // Baseline NEON is mandatory on aarch64.
        return unsafe { vec_dot_pq2_0_q8_0_neon(n, xs, ys) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_pq2_0_q8_0_avx2(n, xs, ys) };
        }
    }
    #[allow(unreachable_code)]
    vec_dot_pq2_0_q8_0_scalar(n, xs, ys)
}

#[inline]
pub(crate) fn vec_dot_ptq1_0_q8_0(n: usize, xs: &[BlockPTQ1_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_PTQ1_0),
        "vec_dot BlockPTQ1_0: {n} not divisible by {QK_PTQ1_0}"
    );
    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { vec_dot_ptq1_0_q8_0_neon(n, xs, ys) };
    }
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            return unsafe { vec_dot_ptq1_0_q8_0_avx2(n, xs, ys) };
        }
    }
    #[allow(unreachable_code)]
    vec_dot_ptq1_0_q8_0_scalar(n, xs, ys)
}

pub(crate) fn vec_dot_pq2_0_q8_0_scalar(n: usize, xs: &[BlockPQ2_0], ys: &[BlockQ8_0]) -> f32 {
    let nb = n / QK_PQ2_0;
    let mut sumf = 0f32;
    for i in 0..nb {
        let x = &xs[i];
        let d0 = x.d.to_f32();
        let mut sumi = 0f32;
        for sub in 0..4 {
            let yb = &ys[i * 4 + sub];
            let qs_sub = &x.qs[sub * 8..sub * 8 + 8];
            let mut isum = 0i32;
            for b_idx in 0..8 {
                let byte = qs_sub[b_idx];
                let y4 = &yb.qs[b_idx * 4..b_idx * 4 + 4];
                isum += (((byte & 3) as i32) - 1) * (y4[0] as i32)
                    + ((((byte >> 2) & 3) as i32) - 1) * (y4[1] as i32)
                    + ((((byte >> 4) & 3) as i32) - 1) * (y4[2] as i32)
                    + ((((byte >> 6) & 3) as i32) - 1) * (y4[3] as i32);
            }
            sumi += yb.d.to_f32() * (isum as f32);
        }
        sumf += d0 * sumi;
    }
    sumf
}

pub(crate) fn vec_dot_ptq1_0_q8_0_scalar(n: usize, xs: &[BlockPTQ1_0], ys: &[BlockQ8_0]) -> f32 {
    const POW3: [u8; 5] = [1, 3, 9, 27, 81];
    let nb = n / QK_PTQ1_0;
    let mut sumf = 0f32;
    let mut q = [0i8; QK_PTQ1_0];
    for i in 0..nb {
        let x = &xs[i];
        let d0 = x.d.to_f32();
        let mut idx = 0usize;
        for &p in &POW3 {
            for m in 0..16 {
                let xi = (((x.qs[m].wrapping_mul(p) as u16) * 3) >> 8) as i8 - 1;
                q[idx + m] = xi;
            }
            idx += 16;
        }
        for &p in &POW3 {
            for m in 0..8 {
                let xi = (((x.qs[16 + m].wrapping_mul(p) as u16) * 3) >> 8) as i8 - 1;
                q[idx + m] = xi;
            }
            idx += 8;
        }
        for &p in &POW3[..4] {
            q[idx] = (((x.qh[0].wrapping_mul(p) as u16) * 3) >> 8) as i8 - 1;
            q[idx + 1] = (((x.qh[1].wrapping_mul(p) as u16) * 3) >> 8) as i8 - 1;
            idx += 2;
        }
        let mut sumi = 0f32;
        for sub in 0..4 {
            let yb = &ys[i * 4 + sub];
            let qs_sub = &q[sub * 32..sub * 32 + 32];
            let mut isum = 0i32;
            for j in 0..32 {
                isum += (qs_sub[j] as i32) * (yb.qs[j] as i32);
            }
            sumi += yb.d.to_f32() * (isum as f32);
        }
        sumf += d0 * sumi;
    }
    sumf
}

#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn neon_dot32_s8(
    q0: std::arch::aarch64::int8x16_t,
    q1: std::arch::aarch64::int8x16_t,
    y_ptr: *const i8,
) -> i32 {
    use std::arch::aarch64::*;
    let y0 = vld1q_s8(y_ptr);
    let y1 = vld1q_s8(y_ptr.add(16));
    #[cfg(target_feature = "dotprod")]
    {
        let acc = vdotq_s32(vdupq_n_s32(0), q0, y0);
        let acc = vdotq_s32(acc, q1, y1);
        return vaddvq_s32(acc);
    }
    #[cfg(not(target_feature = "dotprod"))]
    {
        // |q| <= 2 and |y| <= 128 => |q*y| <= 256; sum of 4 <= 1024, never overflows i16.
        let s0 = vaddq_s16(
            vmull_s8(vget_low_s8(q0), vget_low_s8(y0)),
            vmull_high_s8(q0, y0),
        );
        let s1 = vaddq_s16(
            vmull_s8(vget_low_s8(q1), vget_low_s8(y1)),
            vmull_high_s8(q1, y1),
        );
        vaddlvq_s16(vaddq_s16(s0, s1))
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn vec_dot_pq2_0_q8_0_neon(n: usize, xs: &[BlockPQ2_0], ys: &[BlockQ8_0]) -> f32 {
    use std::arch::aarch64::*;
    const IDX_LO: [u8; 16] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
    const IDX_HI: [u8; 16] = [4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7];
    const SHIFTS: [i8; 16] = [0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6, 0, -2, -4, -6];

    let nb = n / QK_PQ2_0;
    debug_assert!(xs.len() >= nb && ys.len() >= nb * 4);
    let idx_lo = vld1q_u8(IDX_LO.as_ptr());
    let idx_hi = vld1q_u8(IDX_HI.as_ptr());
    let shifts = vld1q_s8(SHIFTS.as_ptr());
    let mask3 = vdupq_n_u8(3);
    let ones = vdupq_n_s8(1);
    let zero8 = vdup_n_u8(0);

    let mut sumf = 0f32;
    for i in 0..nb {
        let x = xs.get_unchecked(i);
        let d0 = x.d.to_f32();
        let mut sumi = 0f32;
        for sub in 0..4 {
            let yb = ys.get_unchecked(i * 4 + sub);
            let raw = vcombine_u8(vld1_u8(x.qs.as_ptr().add(sub * 8)), zero8);
            let q0 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(raw, idx_lo), shifts), mask3)),
                ones,
            );
            let q1 = vsubq_s8(
                vreinterpretq_s8_u8(vandq_u8(vshlq_u8(vqtbl1q_u8(raw, idx_hi), shifts), mask3)),
                ones,
            );
            let isum = neon_dot32_s8(q0, q1, yb.qs.as_ptr());
            sumi += yb.d.to_f32() * (isum as f32);
        }
        sumf += d0 * sumi;
    }
    sumf
}

#[cfg(target_arch = "aarch64")]
unsafe fn vec_dot_ptq1_0_q8_0_neon(n: usize, xs: &[BlockPTQ1_0], ys: &[BlockQ8_0]) -> f32 {
    use std::arch::aarch64::*;
    const POW3_QH: [u8; 8] = [1, 1, 3, 3, 9, 9, 27, 27];

    let nb = n / QK_PTQ1_0;
    debug_assert!(xs.len() >= nb && ys.len() >= nb * 4);
    let three8 = vdup_n_u8(3);
    let three16 = vdupq_n_u8(3);
    let ones16 = vdupq_n_s8(1);
    let ones8 = vdup_n_s8(1);
    let pow3_qh = vld1_u8(POW3_QH.as_ptr());

    #[inline(always)]
    unsafe fn trits16(
        v: uint8x16_t,
        p: u8,
        three8: uint8x8_t,
        three16: uint8x16_t,
        ones16: int8x16_t,
    ) -> int8x16_t {
        let q = vmulq_u8(v, vdupq_n_u8(p));
        let lo = vshrn_n_u16::<8>(vmull_u8(vget_low_u8(q), three8));
        let hi = vshrn_n_u16::<8>(vmull_high_u8(q, three16));
        vsubq_s8(vreinterpretq_s8_u8(vcombine_u8(lo, hi)), ones16)
    }

    #[inline(always)]
    unsafe fn trits8(v: uint8x8_t, p: uint8x8_t, three8: uint8x8_t, ones8: int8x8_t) -> int8x8_t {
        let q = vmul_u8(v, p);
        vsub_s8(
            vreinterpret_s8_u8(vshrn_n_u16::<8>(vmull_u8(q, three8))),
            ones8,
        )
    }

    let mut sumf = 0f32;
    for i in 0..nb {
        let x = xs.get_unchecked(i);
        let d0 = x.d.to_f32();
        let qs0 = vld1q_u8(x.qs.as_ptr());
        let qs1 = vld1_u8(x.qs.as_ptr().add(16));
        let qh_rep = vreinterpret_u8_u16(vdup_n_u16(u16::from_le_bytes(x.qh)));

        let t0 = trits16(qs0, 1, three8, three16, ones16);
        let t1 = trits16(qs0, 3, three8, three16, ones16);
        let t2 = trits16(qs0, 9, three8, three16, ones16);
        let t3 = trits16(qs0, 27, three8, three16, ones16);
        let t4 = trits16(qs0, 81, three8, three16, ones16);

        let u0 = trits8(qs1, vdup_n_u8(1), three8, ones8);
        let u1 = trits8(qs1, vdup_n_u8(3), three8, ones8);
        let u2 = trits8(qs1, vdup_n_u8(9), three8, ones8);
        let u3 = trits8(qs1, vdup_n_u8(27), three8, ones8);
        let u4 = trits8(qs1, vdup_n_u8(81), three8, ones8);
        let uh = trits8(qh_rep, pow3_qh, three8, ones8);

        let y0 = ys.get_unchecked(i * 4);
        let y1 = ys.get_unchecked(i * 4 + 1);
        let y2 = ys.get_unchecked(i * 4 + 2);
        let y3 = ys.get_unchecked(i * 4 + 3);

        let s0 = y0.d.to_f32() * (neon_dot32_s8(t0, t1, y0.qs.as_ptr()) as f32);
        let s1 = y1.d.to_f32() * (neon_dot32_s8(t2, t3, y1.qs.as_ptr()) as f32);
        let s2 = y2.d.to_f32() * (neon_dot32_s8(t4, vcombine_s8(u0, u1), y2.qs.as_ptr()) as f32);
        let s3 = y3.d.to_f32()
            * (neon_dot32_s8(vcombine_s8(u2, u3), vcombine_s8(u4, uh), y3.qs.as_ptr()) as f32);
        sumf += d0 * ((s0 + s1) + (s2 + s3));
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[inline]
unsafe fn avx2_dot32_u8_minus1(codes_u8: std::arch::x86_64::__m256i, y_ptr: *const i8) -> i32 {
    use std::arch::x86_64::*;
    let qy = _mm256_loadu_si256(y_ptr as *const __m256i);
    let ones_u8 = _mm256_set1_epi8(1);
    let ones_16 = _mm256_set1_epi16(1);
    let prod16 = _mm256_maddubs_epi16(codes_u8, qy);
    let sumy16 = _mm256_maddubs_epi16(ones_u8, qy);
    let diff16 = _mm256_sub_epi16(prod16, sumy16);
    let sum32 = _mm256_madd_epi16(diff16, ones_16);
    let sum128 = _mm_add_epi32(
        _mm256_castsi256_si128(sum32),
        _mm256_extracti128_si256(sum32, 1),
    );
    let hi64 = _mm_unpackhi_epi64(sum128, sum128);
    let sum64 = _mm_add_epi32(sum128, hi64);
    let hi32 = _mm_shuffle_epi32(sum64, 0x01);
    _mm_cvtsi128_si32(_mm_add_epi32(sum64, hi32))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_pq2_0_q8_0_avx2(n: usize, xs: &[BlockPQ2_0], ys: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;
    let nb = n / QK_PQ2_0;
    debug_assert!(xs.len() >= nb && ys.len() >= nb * 4);
    let shuf = _mm256_setr_epi8(
        0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 6, 6, 7, 7,
        7, 7,
    );
    let odd_byte_mask = _mm256_set1_epi16(0xFF00u16 as i16);
    let mask3 = _mm256_set1_epi8(3);

    let mut sumf = 0f32;
    for i in 0..nb {
        let x = xs.get_unchecked(i);
        let d0 = x.d.to_f32();
        let mut sumi = 0f32;
        for sub in 0..4 {
            let yb = ys.get_unchecked(i * 4 + sub);
            let raw64 = std::ptr::read_unaligned(x.qs.as_ptr().add(sub * 8) as *const u64) as i64;
            let rep = _mm256_shuffle_epi8(_mm256_set1_epi64x(raw64), shuf);
            let s02 = _mm256_blendv_epi8(rep, _mm256_srli_epi16(rep, 2), odd_byte_mask);
            let s46 = _mm256_blendv_epi8(
                _mm256_srli_epi16(rep, 4),
                _mm256_srli_epi16(rep, 6),
                odd_byte_mask,
            );
            let codes = _mm256_and_si256(_mm256_blend_epi16(s02, s46, 0xAA), mask3);
            let isum = avx2_dot32_u8_minus1(codes, yb.qs.as_ptr());
            sumi += yb.d.to_f32() * (isum as f32);
        }
        sumf += d0 * sumi;
    }
    sumf
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vec_dot_ptq1_0_q8_0_avx2(n: usize, xs: &[BlockPTQ1_0], ys: &[BlockQ8_0]) -> f32 {
    use std::arch::x86_64::*;
    let nb = n / QK_PTQ1_0;
    debug_assert!(xs.len() >= nb && ys.len() >= nb * 4);
    let mask_ff = _mm256_set1_epi16(0x00FF);
    let three16 = _mm256_set1_epi16(3);
    let p_1_3 = _mm256_setr_epi16(1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3);
    let p_9_27 = _mm256_setr_epi16(9, 9, 9, 9, 9, 9, 9, 9, 27, 27, 27, 27, 27, 27, 27, 27);
    let p_81_qh = _mm256_setr_epi16(81, 81, 81, 81, 81, 81, 81, 81, 1, 1, 3, 3, 9, 9, 27, 27);

    let mut sumf = 0f32;
    for i in 0..nb {
        let x = xs.get_unchecked(i);
        let d0 = x.d.to_f32();

        let qs0 = _mm256_cvtepu8_epi16(_mm_loadu_si128(x.qs.as_ptr() as *const __m128i));
        let qs16_u64 = std::ptr::read_unaligned(x.qs.as_ptr().add(16) as *const u64) as i64;
        let qs1_dup = _mm256_cvtepu8_epi16(_mm_set1_epi64x(qs16_u64));
        let qh_rep_u64 = ((u16::from_le_bytes(x.qh) as u64) * 0x0001_0001_0001_0001u64) as i64;
        let tail16 = _mm256_cvtepu8_epi16(_mm_set_epi64x(qh_rep_u64, qs16_u64));

        let dec = |b16: __m256i, p16: __m256i| -> __m256i {
            let q8 = _mm256_and_si256(_mm256_mullo_epi16(b16, p16), mask_ff);
            _mm256_srli_epi16(_mm256_mullo_epi16(q8, three16), 8)
        };
        let pack32 = |a: __m256i, b: __m256i| -> __m256i {
            _mm256_permute4x64_epi64(_mm256_packus_epi16(a, b), 0xD8)
        };

        let xi0 = dec(qs0, _mm256_set1_epi16(1));
        let xi1 = dec(qs0, _mm256_set1_epi16(3));
        let xi2 = dec(qs0, _mm256_set1_epi16(9));
        let xi3 = dec(qs0, _mm256_set1_epi16(27));
        let xi4 = dec(qs0, _mm256_set1_epi16(81));
        let xi_u01 = dec(qs1_dup, p_1_3);
        let xi_u23 = dec(qs1_dup, p_9_27);
        let xi_u4h = dec(tail16, p_81_qh);

        let c0 = pack32(xi0, xi1);
        let c1 = pack32(xi2, xi3);
        let c2 = pack32(xi4, xi_u01);
        let c3 = pack32(xi_u23, xi_u4h);

        let y0 = ys.get_unchecked(i * 4);
        let y1 = ys.get_unchecked(i * 4 + 1);
        let y2 = ys.get_unchecked(i * 4 + 2);
        let y3 = ys.get_unchecked(i * 4 + 3);

        let s0 = y0.d.to_f32() * (avx2_dot32_u8_minus1(c0, y0.qs.as_ptr()) as f32);
        let s1 = y1.d.to_f32() * (avx2_dot32_u8_minus1(c1, y1.qs.as_ptr()) as f32);
        let s2 = y2.d.to_f32() * (avx2_dot32_u8_minus1(c2, y2.qs.as_ptr()) as f32);
        let s3 = y3.d.to_f32() * (avx2_dot32_u8_minus1(c3, y3.qs.as_ptr()) as f32);
        sumf += d0 * ((s0 + s1) + (s2 + s3));
    }
    sumf
}

#[cfg(test)]
mod rocmfp4_tests {
    use super::*;
    use crate::quantized::k_quants::{GgmlType, QK8_0};

    // Known-answer: one dual block, one fast block, every codebook entry hit.
    #[test]
    fn rocmfp4_dual_matches_manual_decode() {
        let mut b = BlockROCMFP4 {
            qs: [0; 16],
            e: [0; 2],
        };
        // e0 = 0x50 (exp bits 6..3 = 10, man 0) -> ue4m3 = 2^3 = 8, half = 4.0
        // e1 = 0x38 (exp 7, man 0) -> ue4m3 = 1, half = 0.5
        b.e = [0x50, 0x38];
        // qs[j]: low nibble -> code j (0..15), high -> 15..0 reversed
        for j in 0..16 {
            b.qs[j] = j as u8 | (((15 - j) as u8) << 4);
        }
        let mut ys = [0f32; 32];
        BlockROCMFP4::to_float(std::slice::from_ref(&b), &mut ys);
        let d0 = rocmfp4_half_to_f32(0x50);
        let d1 = rocmfp4_half_to_f32(0x38);
        assert!((d0 - 4.0).abs() < 1e-9, "d0 {d0}");
        assert!((d1 - 0.5).abs() < 1e-9, "d1 {d1}");
        for j in 0..16 {
            assert_eq!(ys[j], KVALUES_ROCMFP4[j] as f32 * d0, "low half {j}");
            assert_eq!(
                ys[j + 16],
                KVALUES_ROCMFP4[15 - j] as f32 * d1,
                "high half {j}"
            );
        }
    }

    #[test]
    fn rocmfp4_fast_matches_manual_decode() {
        let mut b = BlockROCMFP4Fast {
            qs: [0; 16],
            e: 0x20,
        }; // exp 4, man 0 -> ue4m3 = 1/8, half = 1/16
        for j in 0..16 {
            b.qs[j] = j as u8 | ((j as u8) << 4);
        }
        let mut ys = [0f32; 32];
        BlockROCMFP4Fast::to_float(std::slice::from_ref(&b), &mut ys);
        let d = rocmfp4_half_to_f32(0x20);
        assert!((d - 0.0625).abs() < 1e-9, "d {d}");
        for j in 0..32 {
            assert_eq!(ys[j], KVALUES_ROCMFP4[j % 16] as f32 * d, "elem {j}");
        }
    }

    #[test]
    fn rocmfp4_invalid_scales_decode_zero() {
        let b = BlockROCMFP4Fast {
            qs: [0x0F; 16],
            e: 0x7F,
        };
        let mut ys = [0f32; 32];
        BlockROCMFP4Fast::to_float(std::slice::from_ref(&b), &mut ys);
        assert!(ys.iter().all(|v| *v == 0.0));
    }

    // vec_dot: dequant-dot against a Q8_0 lhs must equal the f32 dot within
    // Q8_0 lhs quantization error.
    #[test]
    fn rocmfp4_vec_dot_close_to_f32() {
        let nb = 8;
        let n = nb * QK_ROCMFP4;
        let xs: Vec<BlockROCMFP4Fast> = (0..nb)
            .map(|i| {
                let mut b = BlockROCMFP4Fast { qs: [0; 16], e: 0 };
                b.e = 0x30 + (i % 4) as u8;
                for j in 0..16 {
                    b.qs[j] = ((i * 5 + j) & 0xFF) as u8;
                }
                b
            })
            .collect();
        let wf32: Vec<f32> = {
            let mut v = vec![0f32; n];
            for (i, b) in xs.iter().enumerate() {
                BlockROCMFP4Fast::to_float(
                    std::slice::from_ref(b),
                    &mut v[i * QK_ROCMFP4..][..QK_ROCMFP4],
                );
            }
            v
        };
        let yf32: Vec<f32> = (0..n).map(|i| (i % 7) as f32 * 0.25 - 1.0).collect();
        let exact: f32 = wf32.iter().zip(&yf32).map(|(a, b)| a * b).sum();
        let ys_q8: Vec<BlockQ8_0> = {
            let nb8 = n / QK8_0;
            let mut v = vec![
                BlockQ8_0 {
                    d: f16::from_bits(0),
                    qs: [0; QK8_0]
                };
                nb8
            ];
            for (i, b) in v.iter_mut().enumerate() {
                BlockQ8_0::from_float(&yf32[i * QK8_0..][..QK8_0], std::slice::from_mut(b));
            }
            v
        };
        let got = BlockROCMFP4Fast::vec_dot(n, &xs, &ys_q8);
        let err = ((got - exact).abs() / exact.abs()).min(1.0);
        assert!(
            err < 2e-2,
            "vec_dot rel err {err} (got {got}, exact {exact})"
        );
    }
}

// ============================================================================
// PQ2_0 / PTQ1_0 known-answer tests.
// The .hex fixtures hold the f32 bit patterns (u32 bits, %08x) printed by
// ProCreations' own dequantize_row_pq2_0 / dequantize_row_ptq1_0
// (runtime/source/llama/ggml/src/ggml-quants.c) on 16 packed blocks:
//   pq2_real_attnqkv / pq2_real_embd: first 16 blocks of blk.0.attn_qkv.weight
//   and token_embd.weight in Ternary-Bonsai-2-27B-PQ2_0.gguf (2026-09-20);
//   pq2_synth / ptq1_synth: deterministic xorshift bytes, finite f16 scales.
// Bit-exact agreement pins the layout to the ggml-common.h blocks and the decode
// to the Prism reference.
// ============================================================================
#[cfg(test)]
mod pq_ptq_kat_tests {
    use super::*;
    use crate::quantized::k_quants::GgmlType;
    use crate::quantized::GgmlDType;

    const NBLK: usize = 16;

    // include_bytes! gives 1-aligned storage; re-back it on 16 so it can be read
    // as blocks (same trick as quant_format's proof module).
    fn aligned_copy(raw: &[u8]) -> Vec<u8> {
        let words = raw.len().div_ceil(16);
        let mut backing: Vec<u128> = vec![0u128; words];
        let out =
            unsafe { std::slice::from_raw_parts_mut(backing.as_mut_ptr() as *mut u8, words * 16) };
        out[..raw.len()].copy_from_slice(raw);
        let ptr = backing.as_ptr() as *const u8;
        std::mem::forget(backing);
        unsafe { Vec::from_raw_parts(ptr as *mut u8, raw.len(), words * 16) }
    }

    fn expected_bits(fixture: &str, block: usize, elem: usize) -> u32 {
        let tok = fixture
            .lines()
            .nth(block)
            .unwrap_or("")
            .split_whitespace()
            .nth(elem)
            .unwrap_or("0");
        u32::from_str_radix(tok, 16).unwrap()
    }

    #[test]
    fn pq2_0_block_layout_matches_ggml_common() {
        assert_eq!(std::mem::size_of::<BlockPQ2_0>(), 34);
        assert_eq!(std::mem::offset_of!(BlockPQ2_0, d), 0);
        assert_eq!(std::mem::offset_of!(BlockPQ2_0, qs), 2);
    }

    #[test]
    fn ptq1_0_block_layout_matches_ggml_common() {
        assert_eq!(std::mem::size_of::<BlockPTQ1_0>(), 28);
        assert_eq!(std::mem::offset_of!(BlockPTQ1_0, qs), 0);
        assert_eq!(std::mem::offset_of!(BlockPTQ1_0, qh), 24);
        assert_eq!(std::mem::offset_of!(BlockPTQ1_0, d), 26);
    }

    fn assert_pq2_bits(raw: &[u8], expected: &str) {
        let buf = aligned_copy(raw);
        let blocks = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const BlockPQ2_0, NBLK) };
        let mut ys = vec![0f32; NBLK * QK_PQ2_0];
        BlockPQ2_0::to_float(blocks, &mut ys);
        for b in 0..NBLK {
            for j in 0..QK_PQ2_0 {
                assert_eq!(
                    ys[b * QK_PQ2_0 + j].to_bits(),
                    expected_bits(expected, b, j),
                    "block {b} elem {j}"
                );
            }
        }
    }

    fn assert_ptq1_bits(raw: &[u8], expected: &str) {
        let buf = aligned_copy(raw);
        let blocks =
            unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const BlockPTQ1_0, NBLK) };
        let mut ys = vec![0f32; NBLK * QK_PTQ1_0];
        BlockPTQ1_0::to_float(blocks, &mut ys);
        for b in 0..NBLK {
            for j in 0..QK_PTQ1_0 {
                assert_eq!(
                    ys[b * QK_PTQ1_0 + j].to_bits(),
                    expected_bits(expected, b, j),
                    "block {b} elem {j}"
                );
            }
        }
    }

    macro_rules! kat {
        ($name:ident, $check:ident, $fixture:literal) => {
            #[test]
            fn $name() {
                $check(
                    include_bytes!(concat!("testdata/", $fixture, ".bin")),
                    include_str!(concat!("testdata/", $fixture, ".hex")),
                );
            }
        };
    }

    kat!(
        pq2_0_real_attn_qkv_bitexact,
        assert_pq2_bits,
        "pq2_real_attnqkv"
    );
    kat!(
        pq2_0_real_token_embd_bitexact,
        assert_pq2_bits,
        "pq2_real_embd"
    );
    kat!(pq2_0_synth_bitexact, assert_pq2_bits, "pq2_synth");
    kat!(ptq1_0_synth_bitexact, assert_ptq1_bits, "ptq1_synth");

    #[test]
    fn pq2_ptq1_dtype_wiring() {
        assert_eq!(GgmlDType::from_u32(142).unwrap(), GgmlDType::PQ2_0);
        assert_eq!(GgmlDType::from_u32(143).unwrap(), GgmlDType::PTQ1_0);
        assert_eq!(GgmlDType::PQ2_0.to_u32(), 142);
        assert_eq!(GgmlDType::PTQ1_0.to_u32(), 143);
        assert_eq!(GgmlDType::PQ2_0.block_size(), 128);
        assert_eq!(GgmlDType::PTQ1_0.block_size(), 128);
        assert_eq!(GgmlDType::PQ2_0.type_size(), 34);
        assert_eq!(GgmlDType::PTQ1_0.type_size(), 28);
    }

    #[test]
    fn pq2_0_vec_dot_matches_reference() {
        use crate::quantized::k_quants::QK8_0;

        let raw = include_bytes!("testdata/pq2_real_attnqkv.bin");
        let buf = aligned_copy(raw);
        let xs = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const BlockPQ2_0, NBLK) };
        let n = NBLK * QK_PQ2_0;

        let mut wf32 = vec![0f32; n];
        BlockPQ2_0::to_float(xs, &mut wf32);

        let yf32: Vec<f32> = (0..n)
            .map(|i| ((i.wrapping_mul(73).wrapping_add(19) % 251) as f32 - 125.0) * 0.013)
            .collect();
        let nb8 = n / QK8_0;
        let mut ys_q8 = vec![
            BlockQ8_0 {
                d: f16::from_bits(0),
                qs: [0; QK8_0],
            };
            nb8
        ];
        for (i, b) in ys_q8.iter_mut().enumerate() {
            BlockQ8_0::from_float(&yf32[i * QK8_0..][..QK8_0], std::slice::from_mut(b));
        }

        // Exact reference against the quantized Q8_0 rhs (`to_float(xs) . to_float(ys_q8)`).
        let mut y_deq = vec![0f32; n];
        BlockQ8_0::to_float(&ys_q8, &mut y_deq);
        let ref_q8: f32 = wf32.iter().zip(&y_deq).map(|(a, b)| a * b).sum();
        let unopt = BlockPQ2_0::vec_dot_unopt(n, xs, &ys_q8);
        let simd = BlockPQ2_0::vec_dot(n, xs, &ys_q8);
        let scalar = vec_dot_pq2_0_q8_0_scalar(n, xs, &ys_q8);

        let rel_simd = (simd - ref_q8).abs() / ref_q8.abs().max(1e-6);
        let rel_unopt = (simd - unopt).abs() / unopt.abs().max(1e-6);
        let rel_scalar = (scalar - ref_q8).abs() / ref_q8.abs().max(1e-6);
        assert!(
            rel_simd < 1e-4,
            "PQ2_0 SIMD vec_dot rel_err {rel_simd} (simd={simd}, ref_q8={ref_q8})"
        );
        assert!(
            rel_unopt < 1e-4,
            "PQ2_0 SIMD vs unopt rel_err {rel_unopt} (simd={simd}, unopt={unopt})"
        );
        assert!(
            rel_scalar < 1e-4,
            "PQ2_0 scalar vec_dot rel_err {rel_scalar} (scalar={scalar}, ref_q8={ref_q8})"
        );
    }

    #[test]
    fn ptq1_0_vec_dot_matches_reference() {
        use crate::quantized::k_quants::QK8_0;

        let raw = include_bytes!("testdata/ptq1_synth.bin");
        let buf = aligned_copy(raw);
        let xs = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const BlockPTQ1_0, NBLK) };
        let n = NBLK * QK_PTQ1_0;

        let mut wf32 = vec![0f32; n];
        BlockPTQ1_0::to_float(xs, &mut wf32);

        let yf32: Vec<f32> = (0..n)
            .map(|i| ((i.wrapping_mul(97).wrapping_add(31) % 241) as f32 - 120.0) * 0.017)
            .collect();
        let nb8 = n / QK8_0;
        let mut ys_q8 = vec![
            BlockQ8_0 {
                d: f16::from_bits(0),
                qs: [0; QK8_0],
            };
            nb8
        ];
        for (i, b) in ys_q8.iter_mut().enumerate() {
            BlockQ8_0::from_float(&yf32[i * QK8_0..][..QK8_0], std::slice::from_mut(b));
        }

        let mut y_deq = vec![0f32; n];
        BlockQ8_0::to_float(&ys_q8, &mut y_deq);
        let ref_q8: f32 = wf32.iter().zip(&y_deq).map(|(a, b)| a * b).sum();
        let unopt = BlockPTQ1_0::vec_dot_unopt(n, xs, &ys_q8);
        let simd = BlockPTQ1_0::vec_dot(n, xs, &ys_q8);
        let scalar = vec_dot_ptq1_0_q8_0_scalar(n, xs, &ys_q8);

        let rel_simd = (simd - ref_q8).abs() / ref_q8.abs().max(1e-6);
        let rel_unopt = (simd - unopt).abs() / unopt.abs().max(1e-6);
        let rel_scalar = (scalar - ref_q8).abs() / ref_q8.abs().max(1e-6);
        assert!(
            rel_simd < 1e-4,
            "PTQ1_0 SIMD vec_dot rel_err {rel_simd} (simd={simd}, ref_q8={ref_q8})"
        );
        assert!(
            rel_unopt < 1e-4,
            "PTQ1_0 SIMD vs unopt rel_err {rel_unopt} (simd={simd}, unopt={unopt})"
        );
        assert!(
            rel_scalar < 1e-4,
            "PTQ1_0 scalar vec_dot rel_err {rel_scalar} (scalar={scalar}, ref_q8={ref_q8})"
        );
    }
}
