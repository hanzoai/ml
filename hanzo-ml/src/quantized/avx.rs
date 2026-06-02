use super::k_quants::{
    BlockQ2K, BlockQ3K, BlockQ4K, BlockQ4_0, BlockQ5K, BlockQ6K, BlockQ8K, BlockQ8_0, QK8_0, QK_K,
};
use byteorder::{ByteOrder, LittleEndian};
use half::f16;

#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[inline(always)]
pub(crate) unsafe fn sum_i16_pairs_float(x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);
    _mm256_cvtepi32_ps(summed_pairs)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_us8_pairs_float(ax: __m256i, sy: __m256i) -> __m256 {
    let dot = _mm256_maddubs_epi16(ax, sy);
    sum_i16_pairs_float(dot)
}

#[inline(always)]
pub(crate) unsafe fn hsum_float_8(x: __m256) -> f32 {
    let res = _mm256_extractf128_ps(x, 1);
    let res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    let res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    let res = _mm_add_ss(res, _mm_movehdup_ps(res));
    _mm_cvtss_f32(res)
}

#[inline(always)]
pub(crate) unsafe fn bytes_from_nibbles_32(rsi: *const u8) -> __m256i {
    let tmp = _mm_loadu_si128(rsi as *const __m128i);
    let bytes = _mm256_insertf128_si256::<1>(_mm256_castsi128_si256(tmp), _mm_srli_epi16(tmp, 4));
    let low_mask = _mm256_set1_epi8(0xF);
    _mm256_and_si256(low_mask, bytes)
}

#[inline(always)]
pub(crate) unsafe fn mul_sum_i8_pairs_float(x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);
    let sy = _mm256_sign_epi8(y, x);
    mul_sum_us8_pairs_float(ax, sy)
}

#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q4_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            let bx = bytes_from_nibbles_32(x.qs.as_ptr());
            let off = _mm256_set1_epi8(8);
            let bx = _mm256_sub_epi8(bx, off);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q8_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = _mm256_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            let bx = _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let q = mul_sum_i8_pairs_float(bx, by);
            acc = _mm256_fmadd_ps(d, q, acc);
        }
        hsum_float_8(acc)
    }
}

#[inline(always)]
unsafe fn get_scale_shuffle(i: usize) -> __m128i {
    const K_SHUFFLE: [u8; 128] = [
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3,
        3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7,
        7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10,
        11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 13, 13, 13, 13, 13, 13, 13,
        13, 14, 14, 14, 14, 14, 14, 14, 14, 15, 15, 15, 15, 15, 15, 15, 15,
    ];
    _mm_loadu_si128((K_SHUFFLE.as_ptr() as *const __m128i).add(i))
}

#[inline(always)]
unsafe fn get_scale_shuffle_k4(i: usize) -> __m256i {
    const K_SHUFFLE: [u8; 256] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,
        4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,
        8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10,
        11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13,
        12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 12, 13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
        14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    _mm256_loadu_si256((K_SHUFFLE.as_ptr() as *const __m256i).add(i))
}

#[inline(always)]
unsafe fn get_scale_shuffle_q3k(i: usize) -> __m256i {
    const K_SHUFFLE: [u8; 128] = [
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
        2, 3, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
        6, 7, 6, 7, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 11, 10, 11, 10, 11, 10, 11,
        10, 11, 10, 11, 10, 11, 10, 11, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12, 13, 12,
        13, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15, 14, 15,
    ];
    _mm256_loadu_si256((K_SHUFFLE.as_ptr() as *const __m256i).add(i))
}

#[inline(always)]
pub(crate) fn vec_dot_q6k_q8k(n: usize, xs: &[BlockQ6K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q6k_8k: {n} is not divisible by {QK_K}"
    );

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let m2 = _mm256_set1_epi8(3);
        let m32s = _mm256_set1_epi8(32);
        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let mut q4 = x.ql.as_ptr();
            let mut qh = x.qh.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 128 {
                let is = j * 4;
                let scale_0 = _mm_shuffle_epi8(scales, get_scale_shuffle(is));
                let scale_1 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 1));
                let scale_2 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 2));
                let scale_3 = _mm_shuffle_epi8(scales, get_scale_shuffle(is + 3));

                let q4bits1 = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4bits2 = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4bits_h = _mm256_loadu_si256(qh as *const __m256i);
                qh = qh.add(32);

                let q4h_0 = _mm256_slli_epi16(_mm256_and_si256(q4bits_h, m2), 4);
                let q4h_1 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 2), m2), 4);
                let q4h_2 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 4), m2), 4);
                let q4h_3 =
                    _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(q4bits_h, 6), m2), 4);

                let q4_0 = _mm256_or_si256(_mm256_and_si256(q4bits1, m4), q4h_0);
                let q4_1 = _mm256_or_si256(_mm256_and_si256(q4bits2, m4), q4h_1);
                let q4_2 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits1, 4), m4), q4h_2);
                let q4_3 =
                    _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(q4bits2, 4), m4), q4h_3);

                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                let q8s_0 = _mm256_maddubs_epi16(m32s, q8_0);
                let q8s_1 = _mm256_maddubs_epi16(m32s, q8_1);
                let q8s_2 = _mm256_maddubs_epi16(m32s, q8_2);
                let q8s_3 = _mm256_maddubs_epi16(m32s, q8_3);

                let p16_0 = _mm256_maddubs_epi16(q4_0, q8_0);
                let p16_1 = _mm256_maddubs_epi16(q4_1, q8_1);
                let p16_2 = _mm256_maddubs_epi16(q4_2, q8_2);
                let p16_3 = _mm256_maddubs_epi16(q4_3, q8_3);

                let p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                let p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                let p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                let p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                let p16_0 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_0), p16_0);
                let p16_1 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_1), p16_1);
                let p16_2 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_2), p16_2);
                let p16_3 = _mm256_madd_epi16(_mm256_cvtepi8_epi16(scale_3), p16_3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_2, p16_3));
            }
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc)
    }
}

#[inline(always)]
unsafe fn mm256_set_m128i(a: __m128i, b: __m128i) -> __m256i {
    _mm256_insertf128_si256(_mm256_castsi128_si256(b), a, 1)
}

#[inline(always)]
pub(crate) fn vec_dot_q2k_q8k(n: usize, xs: &[BlockQ2K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q2k_q8k: {n} is not divisible by {QK_K}"
    );

    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let m4 = _mm_set1_epi8(0xF);

        let mut acc = _mm256_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            let mut q2 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mins_and_scales = _mm_loadu_si128(x.scales.as_ptr() as *const __m128i);
            let scales8 = _mm_and_si128(mins_and_scales, m4);
            let mins8 = _mm_and_si128(_mm_srli_epi16(mins_and_scales, 4), m4);
            let mins = _mm256_cvtepi8_epi16(mins8);
            let prod =
                _mm256_madd_epi16(mins, _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i));

            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&dmin), _mm256_cvtepi32_ps(prod), acc);

            let all_scales = _mm256_cvtepi8_epi16(scales8);
            let l_scales = _mm256_extracti128_si256(all_scales, 0);
            let h_scales = _mm256_extracti128_si256(all_scales, 1);
            let scales = [
                mm256_set_m128i(l_scales, l_scales),
                mm256_set_m128i(h_scales, h_scales),
            ];

            let mut sumi = _mm256_setzero_si256();

            for scale in scales {
                let q2bits = _mm256_loadu_si256(q2 as *const __m256i);
                q2 = q2.add(32);

                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                let q2_0 = _mm256_and_si256(q2bits, m3);
                let q2_1 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), m3);
                let q2_2 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), m3);
                let q2_3 = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), m3);

                let p0 = _mm256_maddubs_epi16(q2_0, q8_0);
                let p1 = _mm256_maddubs_epi16(q2_1, q8_1);
                let p2 = _mm256_maddubs_epi16(q2_2, q8_2);
                let p3 = _mm256_maddubs_epi16(q2_3, q8_3);

                let p0 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(0)), p0);
                let p1 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(1)), p1);
                let p2 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(2)), p2);
                let p3 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(scale, get_scale_shuffle_q3k(3)), p3);

                let p0 = _mm256_add_epi32(p0, p1);
                let p2 = _mm256_add_epi32(p2, p3);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p0, p2));
            }
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }

        hsum_float_8(acc)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q3k_q8k(n: usize, xs: &[BlockQ3K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q3k_q8k: {n} is not divisible by {QK_K}"
    );

    const KMASK1: u32 = 0x03030303;
    const KMASK2: u32 = 0x0f0f0f0f;

    let mut aux = [0u32; 3];

    unsafe {
        let m3 = _mm256_set1_epi8(3);
        let mone = _mm256_set1_epi8(1);
        let m32 = _mm_set1_epi8(32);

        let mut acc = _mm256_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();

            let mut q3 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            LittleEndian::read_u32_into(&x.scales, &mut aux);
            let scales128 = _mm_set_epi32(
                (((aux[1] >> 4) & KMASK2) | (((aux[2] >> 6) & KMASK1) << 4)) as i32,
                (((aux[0] >> 4) & KMASK2) | (((aux[2] >> 4) & KMASK1) << 4)) as i32,
                ((aux[1] & KMASK2) | (((aux[2] >> 2) & KMASK1) << 4)) as i32,
                ((aux[0] & KMASK2) | (((aux[2]) & KMASK1) << 4)) as i32,
            );
            let scales128 = _mm_sub_epi8(scales128, m32);
            let all_scales = _mm256_cvtepi8_epi16(scales128);
            let l_scales = _mm256_extracti128_si256(all_scales, 0);
            let h_scales = _mm256_extracti128_si256(all_scales, 1);
            let scales = [
                mm256_set_m128i(l_scales, l_scales),
                mm256_set_m128i(h_scales, h_scales),
            ];

            // high bit
            let hbits = _mm256_loadu_si256(x.hmask.as_ptr() as *const __m256i);

            let mut sumi = _mm256_setzero_si256();

            for (j, scale) in scales.iter().enumerate() {
                // load low 2 bits
                let q3bits = _mm256_loadu_si256(q3 as *const __m256i);
                q3 = q3.add(32);

                // Prepare low and high bits
                // We hardcode the shifts here to avoid loading them into a separate register
                let q3l_0 = _mm256_and_si256(q3bits, m3);
                let q3h_0 = if j == 0 {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 0)), 0)
                } else {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 4)), 4)
                };
                let q3h_0 = _mm256_slli_epi16(q3h_0, 2);

                let q3l_1 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 2), m3);
                let q3h_1 = if j == 0 {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 1)), 1)
                } else {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 5)), 5)
                };
                let q3h_1 = _mm256_slli_epi16(q3h_1, 2);

                let q3l_2 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 4), m3);
                let q3h_2 = if j == 0 {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 2)), 2)
                } else {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 6)), 6)
                };
                let q3h_2 = _mm256_slli_epi16(q3h_2, 2);

                let q3l_3 = _mm256_and_si256(_mm256_srli_epi16(q3bits, 6), m3);
                let q3h_3 = if j == 0 {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 3)), 3)
                } else {
                    _mm256_srli_epi16(_mm256_andnot_si256(hbits, _mm256_slli_epi16(mone, 7)), 7)
                };
                let q3h_3 = _mm256_slli_epi16(q3h_3, 2);

                // load Q8 quants
                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_2 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_3 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                // Dot product: we multiply the 2 low bits and 1 high bit part separately, so we
                // can use _mm256_maddubs_epi16, and then subtract. The high bit part has the 2
                // already subtracted (and so, it is zero if the high bit was not set, and 2 if the
                // high bit was set)
                let q8s_0 = _mm256_maddubs_epi16(q3h_0, q8_0);
                let q8s_1 = _mm256_maddubs_epi16(q3h_1, q8_1);
                let q8s_2 = _mm256_maddubs_epi16(q3h_2, q8_2);
                let q8s_3 = _mm256_maddubs_epi16(q3h_3, q8_3);

                let p16_0 = _mm256_maddubs_epi16(q3l_0, q8_0);
                let p16_1 = _mm256_maddubs_epi16(q3l_1, q8_1);
                let p16_2 = _mm256_maddubs_epi16(q3l_2, q8_2);
                let p16_3 = _mm256_maddubs_epi16(q3l_3, q8_3);

                let p16_0 = _mm256_sub_epi16(p16_0, q8s_0);
                let p16_1 = _mm256_sub_epi16(p16_1, q8s_1);
                let p16_2 = _mm256_sub_epi16(p16_2, q8s_2);
                let p16_3 = _mm256_sub_epi16(p16_3, q8s_3);

                // multiply with scales
                let p16_0 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(0)), p16_0);
                let p16_1 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(1)), p16_1);
                let p16_2 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(2)), p16_2);
                let p16_3 =
                    _mm256_madd_epi16(_mm256_shuffle_epi8(*scale, get_scale_shuffle_q3k(3)), p16_3);

                // accumulate
                let p16_0 = _mm256_add_epi32(p16_0, p16_1);
                let p16_2 = _mm256_add_epi32(p16_2, p16_3);
                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_2));
            }

            // multiply with block scale and accumulate
            acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&d), _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q4k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);

        let mut acc = _mm256_setzero_ps();
        let mut acc_m = _mm_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));

            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = mm256_set_m128i(sc128, sc128);

            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 64 {
                let scale_l = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                let scale_h = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                let q4bits = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4l = _mm256_and_si256(q4bits, m4);
                let q4h = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), m4);

                let q8l = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let p16l = _mm256_maddubs_epi16(q4l, q8l);
                let p16l = _mm256_madd_epi16(scale_l, p16l);
                sumi = _mm256_add_epi32(sumi, p16l);

                let q8h = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let p16h = _mm256_maddubs_epi16(q4h, q8h);
                let p16h = _mm256_madd_epi16(scale_h, p16h);
                sumi = _mm256_add_epi32(sumi, p16h);
            }

            let vd = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        }

        let acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        let acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

        hsum_float_8(acc) + _mm_cvtss_f32(acc_m)
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q5k_q8k(n: usize, xs: &[BlockQ5K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q5k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);
        let mzero = _mm_setzero_si128();
        let mone = _mm256_set1_epi8(1);

        let mut acc = _mm256_setzero_ps();
        let mut summs = 0.0;

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mut q5 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));

            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            let hsum = _mm_hadd_epi32(_mm_hadd_epi32(prod, mzero), mzero);
            summs += dmin * _mm_extract_epi32(hsum, 0) as f32;

            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            let scales = mm256_set_m128i(sc128, sc128);

            let hbits = _mm256_loadu_si256(x.qh.as_ptr() as *const __m256i);
            let mut hmask = mone;

            let mut sumi = _mm256_setzero_si256();

            for j in 0..QK_K / 64 {
                let scale_0 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j));
                let scale_1 = _mm256_shuffle_epi8(scales, get_scale_shuffle_k4(2 * j + 1));

                let q5bits = _mm256_loadu_si256(q5 as *const __m256i);
                q5 = q5.add(32);

                //Similar to q3k we hardcode the shifts here to avoid loading them into a separate register
                let q5l_0 = _mm256_and_si256(q5bits, m4);
                let q5l_0_shift_input = _mm256_and_si256(hbits, hmask);
                let q5l_0_right_shift = match j {
                    0 => _mm256_srli_epi16(q5l_0_shift_input, 0),
                    1 => _mm256_srli_epi16(q5l_0_shift_input, 2),
                    2 => _mm256_srli_epi16(q5l_0_shift_input, 4),
                    3 => _mm256_srli_epi16(q5l_0_shift_input, 6),
                    _ => unreachable!(),
                };
                let q5h_0 = _mm256_slli_epi16(q5l_0_right_shift, 4);
                let q5_0 = _mm256_add_epi8(q5l_0, q5h_0);
                hmask = _mm256_slli_epi16(hmask, 1);

                let q5l_1 = _mm256_and_si256(_mm256_srli_epi16(q5bits, 4), m4);
                let q5l_1_shift_input = _mm256_and_si256(hbits, hmask);
                let q5l_1_right_shift = match j {
                    0 => _mm256_srli_epi16(q5l_1_shift_input, 1),
                    1 => _mm256_srli_epi16(q5l_1_shift_input, 3),
                    2 => _mm256_srli_epi16(q5l_1_shift_input, 5),
                    3 => _mm256_srli_epi16(q5l_1_shift_input, 7),
                    _ => unreachable!(),
                };

                let q5h_1 = _mm256_slli_epi16(q5l_1_right_shift, 4);
                let q5_1 = _mm256_add_epi8(q5l_1, q5h_1);
                hmask = _mm256_slli_epi16(hmask, 1);

                let q8_0 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8_1 = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                let p16_0 = _mm256_maddubs_epi16(q5_0, q8_0);
                let p16_1 = _mm256_maddubs_epi16(q5_1, q8_1);

                let p16_0 = _mm256_madd_epi16(scale_0, p16_0);
                let p16_1 = _mm256_madd_epi16(scale_1, p16_1);

                sumi = _mm256_add_epi32(sumi, _mm256_add_epi32(p16_0, p16_1));
            }
            let vd = _mm256_set1_ps(d);
            acc = _mm256_fmadd_ps(vd, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc) + summs
    }
}

#[inline(always)]
pub(crate) fn vec_dot_q8k_q8k(n: usize, xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q8k_8k: {n} is not divisible by {QK_K}"
    );
    unsafe {
        let mut acc = _mm256_setzero_ps();
        for (xs, ys) in xs.iter().zip(ys.iter()) {
            let mut sumi = _mm256_setzero_si256();
            let x_qs = xs.qs.as_ptr();
            let y_qs = ys.qs.as_ptr();
            for j in (0..QK_K).step_by(32) {
                let xs = _mm256_loadu_si256(x_qs.add(j) as *const __m256i);
                let ys = _mm256_loadu_si256(y_qs.add(j) as *const __m256i);

                let xs0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xs, 0));
                let ys0 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(ys, 0));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(xs0, ys0));

                let xs1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(xs, 1));
                let ys1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(ys, 1));
                sumi = _mm256_add_epi32(sumi, _mm256_madd_epi16(xs1, ys1));
            }
            let d = _mm256_set1_ps(xs.d * ys.d);
            acc = _mm256_fmadd_ps(d, _mm256_cvtepi32_ps(sumi), acc);
        }
        hsum_float_8(acc)
    }
}

// ---------------------------------------------------------------------------
// AVX-512 / AVX-512-VNNI kernels.
//
// These mirror the AVX2 kernels above but operate on 512-bit registers and, for
// the q4_0/q4k/q8_0 paths, use the VNNI `vpdpbusd` instruction
// (`_mm512_dpbusd_epi32`) which fuses an unsigned*signed byte multiply with a
// horizontal-by-4 i32 accumulation in a single uop.
//
// Everything is gated behind `x86_64` + the concrete target features that the
// intrinsics require (`avx512f` for the float/FMA core, `avx512bw` for the
// byte/word integer ops, `avx512vnni` for `vpdpbusd`). On a target that does
// not enable all three (older x86, ARM, wasm, ...) none of this is compiled and
// the AVX2 / NEON / scalar fallbacks remain the only kernels, so this cannot
// break those builds. The compile-time dispatch in `k_quants.rs` prefers these
// kernels over the AVX2 ones when the features are enabled.
// ---------------------------------------------------------------------------

#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
#[inline(always)]
unsafe fn hsum_float_16(x: __m512) -> f32 {
    _mm512_reduce_add_ps(x)
}

/// q4_0 x q8_0 using VNNI.
///
/// q4_0 stores 32 weights as 16 packed nibbles in `[0, 15]`; the real signed
/// weight is `nibble - 8`. We feed the *unsigned* nibbles to `vpdpbusd`
/// (unsigned * signed) and afterwards subtract `8 * sum(q8)`, computed with a
/// second `vpdpbusd` against an all-ones byte vector. Two blocks are processed
/// per iteration so each 512-bit register is fully utilised; an odd trailing
/// block is handled with a widen-to-i16 `madd` tail.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
#[inline(always)]
pub(crate) fn vec_dot_q4_0_q8_0_avx512(n: usize, xs: &[BlockQ4_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q4_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    unsafe {
        let low_mask = _mm512_set1_epi8(0xF);
        let ones = _mm512_set1_epi8(1);
        let nb = xs.len().min(ys.len());
        let mut acc = _mm512_setzero_ps();
        let mut i = 0;
        // Two 32-element blocks at a time -> 64 nibbles per 512-bit lane.
        while i + 2 <= nb {
            let x0 = &xs[i];
            let x1 = &xs[i + 1];
            let y0 = &ys[i];
            let y1 = &ys[i + 1];

            // Expand the two blocks' nibbles into 64 unsigned bytes in [0, 15].
            // Low nibbles of block0 (16) | low nibbles of block1 (16) | high
            // nibbles of block0 (16) | high nibbles of block1 (16) — the exact
            // ordering only has to match the q8 byte ordering we build below.
            let bx0 = bytes_from_nibbles_32(x0.qs.as_ptr()); // 32 bytes, block0
            let bx1 = bytes_from_nibbles_32(x1.qs.as_ptr()); // 32 bytes, block1
            let bx = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(bx0), bx1);
            let bx = _mm512_and_si512(bx, low_mask);

            let by0 = _mm256_loadu_si256(y0.qs.as_ptr() as *const __m256i);
            let by1 = _mm256_loadu_si256(y1.qs.as_ptr() as *const __m256i);
            let by = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(by0), by1);

            // raw = sum(nibble * q8) per i32 lane (4 bytes folded together)
            let raw = _mm512_dpbusd_epi32(_mm512_setzero_si512(), bx, by);
            // corr = sum(q8) per i32 lane
            let corr = _mm512_dpbusd_epi32(_mm512_setzero_si512(), ones, by);
            // dot = sum((nibble - 8) * q8) = raw - 8 * corr
            let prod = _mm512_sub_epi32(raw, _mm512_slli_epi32::<3>(corr));

            // Lanes 0..7 belong to block0, lanes 8..15 to block1. Scale each
            // half by its own (dx * dy) and accumulate.
            let prodf = _mm512_cvtepi32_ps(prod);
            let d0 = f16::to_f32(x0.d) * f16::to_f32(y0.d);
            let d1 = f16::to_f32(x1.d) * f16::to_f32(y1.d);
            let mut ds = [d1; 16];
            ds[0..8].fill(d0);
            let dv = _mm512_loadu_ps(ds.as_ptr());
            acc = _mm512_fmadd_ps(dv, prodf, acc);

            i += 2;
        }
        let mut sumf = hsum_float_16(acc);
        // Odd trailing block via widen-to-i16 madd (exact signed*signed).
        if i < nb {
            let x = &xs[i];
            let y = &ys[i];
            let bx = bytes_from_nibbles_32(x.qs.as_ptr());
            let off = _mm256_set1_epi8(8);
            let bx = _mm256_sub_epi8(bx, off);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let xs16 = _mm512_cvtepi8_epi16(bx);
            let ys16 = _mm512_cvtepi8_epi16(by);
            let sumi = _mm512_madd_epi16(xs16, ys16);
            let d = f16::to_f32(x.d) * f16::to_f32(y.d);
            sumf += d * _mm512_reduce_add_epi32(sumi) as f32;
        }
        sumf
    }
}

/// q8_0 x q8_0 — both operands are signed bytes, so VNNI's unsigned*signed
/// `vpdpbusd` cannot be used directly. We widen each 32-byte block to 32 i16
/// and use a single 512-bit `madd_epi16` (exact signed*signed), accumulating
/// the per-block i32 sums and scaling by `dx*dy`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
#[inline(always)]
pub(crate) fn vec_dot_q8_0_q8_0_avx512(n: usize, xs: &[BlockQ8_0], ys: &[BlockQ8_0]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK8_0),
        "vec_dot_q8_0_q8_0: {n} is not divisible by {QK8_0}"
    );
    unsafe {
        let mut acc = _mm512_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let bx = _mm256_loadu_si256(x.qs.as_ptr() as *const __m256i);
            let by = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);
            let xs16 = _mm512_cvtepi8_epi16(bx);
            let ys16 = _mm512_cvtepi8_epi16(by);
            let sumi = _mm512_madd_epi16(xs16, ys16);
            let d = _mm512_set1_ps(f16::to_f32(x.d) * f16::to_f32(y.d));
            acc = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(sumi), acc);
        }
        hsum_float_16(acc)
    }
}

/// q4k x q8k.
///
/// Q4K packs `QK_K`(=256) 4-bit weights with 8 sub-blocks of 32, each carrying
/// a 6-bit scale and a 6-bit min (decoded into `utmp`). The q4 nibbles are
/// already unsigned in `[0, 15]` (no -8 offset), which is exactly what VNNI
/// wants. We process one 64-byte chunk of q4 (two sub-blocks: low + high
/// nibbles) per 512-bit step, applying the per-sub-block scale with an i16
/// `madd` after the `maddubs` widening — mirroring the AVX2 kernel but at
/// double width. The min term is handled exactly as in AVX2 with the 128-bit
/// `q8sums` reduction.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
#[inline(always)]
pub(crate) fn vec_dot_q4k_q8k_avx512(n: usize, xs: &[BlockQ4K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q4k_q8k: {n} is not divisible by {QK_K}"
    );
    let mut utmp = [0u32; 4];
    const KMASK1: u32 = 0x3f3f3f3f;
    const KMASK2: u32 = 0x0f0f0f0f;
    const KMASK3: u32 = 0x03030303;

    unsafe {
        let m4 = _mm256_set1_epi8(0xF);

        let mut acc = _mm512_setzero_ps();
        let mut acc_m = _mm_setzero_ps();

        for (x, y) in xs.iter().zip(ys.iter()) {
            let d = y.d * x.d.to_f32();
            let dmin = -y.d * x.dmin.to_f32();

            LittleEndian::read_u32_into(&x.scales, &mut utmp[0..3]);

            utmp[3] = ((utmp[2] >> 4) & KMASK2) | (((utmp[1] >> 6) & KMASK3) << 4);
            let uaux = utmp[1] & KMASK1;
            utmp[1] = (utmp[2] & KMASK2) | (((utmp[0] >> 6) & KMASK3) << 4);
            utmp[2] = uaux;
            utmp[0] &= KMASK1;

            let mut q4 = x.qs.as_ptr();
            let mut q8 = y.qs.as_ptr();

            // 8 scales then 8 mins, sign-extended to i16 (they are < 64).
            let mins_and_scales = _mm256_cvtepu8_epi16(_mm_set_epi32(
                utmp[3] as i32,
                utmp[2] as i32,
                utmp[1] as i32,
                utmp[0] as i32,
            ));

            // ---- min contribution (identical to AVX2) ----
            let q8sums = _mm256_loadu_si256(y.bsums.as_ptr() as *const __m256i);
            let q8s = _mm_hadd_epi16(
                _mm256_extracti128_si256(q8sums, 0),
                _mm256_extracti128_si256(q8sums, 1),
            );
            let prod = _mm_madd_epi16(_mm256_extracti128_si256(mins_and_scales, 1), q8s);
            acc_m = _mm_fmadd_ps(_mm_set1_ps(dmin), _mm_cvtepi32_ps(prod), acc_m);

            // The 8 scales live in the low 128 bits of `mins_and_scales`.
            let sc128 = _mm256_extracti128_si256(mins_and_scales, 0);
            // Duplicate the 8 i16 scales into a 256-bit lane so the existing
            // 256-bit shuffle table addresses them, then we build 512-bit
            // scale vectors from two consecutive groups.
            let scales256 = mm256_set_m128i(sc128, sc128);

            let mut sumi = _mm512_setzero_si512();

            // QK_K/64 == 4 iterations of the AVX2 kernel; each handles 64 q4
            // values (low+high nibble of a 32-byte load) and two scale groups.
            // We pack two AVX2 iterations into one 512-bit step (j and j+1).
            for j in (0..QK_K / 64).step_by(2) {
                // ---- group j ----
                let sl_a = _mm256_shuffle_epi8(scales256, get_scale_shuffle_k4(2 * j));
                let sh_a = _mm256_shuffle_epi8(scales256, get_scale_shuffle_k4(2 * j + 1));
                let q4b_a = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4l_a = _mm256_and_si256(q4b_a, m4);
                let q4h_a = _mm256_and_si256(_mm256_srli_epi16(q4b_a, 4), m4);
                let q8l_a = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8h_a = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                // ---- group j+1 ----
                let sl_b = _mm256_shuffle_epi8(scales256, get_scale_shuffle_k4(2 * (j + 1)));
                let sh_b = _mm256_shuffle_epi8(scales256, get_scale_shuffle_k4(2 * (j + 1) + 1));
                let q4b_b = _mm256_loadu_si256(q4 as *const __m256i);
                q4 = q4.add(32);
                let q4l_b = _mm256_and_si256(q4b_b, m4);
                let q4h_b = _mm256_and_si256(_mm256_srli_epi16(q4b_b, 4), m4);
                let q8l_b = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);
                let q8h_b = _mm256_loadu_si256(q8 as *const __m256i);
                q8 = q8.add(32);

                // Combine the two groups' low-nibble products into one 512-bit op.
                let q4l = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q4l_a), q4l_b);
                let q8l = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q8l_a), q8l_b);
                let scl = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(sl_a), sl_b);
                let p16l = _mm512_maddubs_epi16(q4l, q8l);
                let p16l = _mm512_madd_epi16(scl, p16l);
                sumi = _mm512_add_epi32(sumi, p16l);

                // ... and the two groups' high-nibble products.
                let q4h = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q4h_a), q4h_b);
                let q8h = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(q8h_a), q8h_b);
                let sch = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(sh_a), sh_b);
                let p16h = _mm512_maddubs_epi16(q4h, q8h);
                let p16h = _mm512_madd_epi16(sch, p16h);
                sumi = _mm512_add_epi32(sumi, p16h);
            }

            let vd = _mm512_set1_ps(d);
            acc = _mm512_fmadd_ps(vd, _mm512_cvtepi32_ps(sumi), acc);
        }

        let acc_m = _mm_add_ps(acc_m, _mm_movehl_ps(acc_m, acc_m));
        let acc_m = _mm_add_ss(acc_m, _mm_movehdup_ps(acc_m));

        hsum_float_16(acc) + _mm_cvtss_f32(acc_m)
    }
}

/// q8k x q8k — signed*signed sum over the 256 i8 weights, widened to i16 and
/// reduced with 512-bit `madd_epi16`.
#[cfg(all(
    target_arch = "x86_64",
    target_feature = "avx512f",
    target_feature = "avx512bw",
    target_feature = "avx512vnni"
))]
#[inline(always)]
pub(crate) fn vec_dot_q8k_q8k_avx512(n: usize, xs: &[BlockQ8K], ys: &[BlockQ8K]) -> f32 {
    debug_assert!(
        n.is_multiple_of(QK_K),
        "vec_dot_q8k_8k: {n} is not divisible by {QK_K}"
    );
    unsafe {
        let mut acc = _mm512_setzero_ps();
        for (x, y) in xs.iter().zip(ys.iter()) {
            let mut sumi = _mm512_setzero_si512();
            let x_qs = x.qs.as_ptr();
            let y_qs = y.qs.as_ptr();
            // 256 i8 -> eight 32-byte chunks, each widened to 32 i16.
            for j in (0..QK_K).step_by(32) {
                let xv = _mm256_loadu_si256(x_qs.add(j) as *const __m256i);
                let yv = _mm256_loadu_si256(y_qs.add(j) as *const __m256i);
                let xv16 = _mm512_cvtepi8_epi16(xv);
                let yv16 = _mm512_cvtepi8_epi16(yv);
                sumi = _mm512_add_epi32(sumi, _mm512_madd_epi16(xv16, yv16));
            }
            let d = _mm512_set1_ps(x.d * y.d);
            acc = _mm512_fmadd_ps(d, _mm512_cvtepi32_ps(sumi), acc);
        }
        hsum_float_16(acc)
    }
}
