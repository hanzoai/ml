// ============================================================================
// CUDA native i-quant mmvq DECODE -- the SotA decode-path twin of the ROCm
// qdp4a<IQ*> kernels (bit-exact, 2.27x). i-quant weights are codebook-decoded into
// int8 grid magnitudes (+ sign, or pre-signed for IQ1) and dp4a'd against the q8_1
// activation row directly -- NO dequant->f32 round-trip. Faithful slice of the proven
// quant.hip decode (hip_dp4a -> __dp4a, hip_bfloat16 -> __nv_bfloat16, __shfl_* ->
// *_sync); grids generated from iq_grids.rs (the one source of truth) into the .cuh.
//
// PTX module (NOT in build.rs static list): every extern "C" __global__ here is a
// runtime entry point launched by name via cudarc from quantized/cuda.rs, exactly
// mirroring rocm_backend/mod.rs's launch-by-name dispatch.
// ============================================================================
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include "iquant_grids.cuh"

// Decode weight-type ids (match quant.hip's DW_* id space for the ported structs/traits).
#define DW_IQ2_XXS 13
#define DW_IQ2_XS  14
#define DW_IQ2_S   15
#define DW_IQ3_XXS 16
#define DW_IQ3_S   17
#define DW_IQ1_S   20
#define DW_IQ1_M   21
#define DW_IQ4_XS  4
// Ternary (base-3 / 2-bit) weight ids (match quant.hip's DW_* id space). Symmetric ternary {-1,0,1}.
#define DW_TQ2_0   5
#define DW_TQ1_0   19
// Prism / Bonsai-2 128-element ternary formats: PQ2_0 ({-1,0,+1,+2}, 34B) and PTQ1_0 ({-1,0,+1} base-3, 28B).
#define DW_PQ2_0   25
#define DW_PTQ1_0  26
#define DW_Q1_0    27
#define DW_Q2_0    28

// IQ1_S/IQ1_M signed-grid delta bias (iq_quants.rs:28); both 1-bit types share it.
constexpr float IQ1S_DELTA = 0.125f;

// IQ4_XS/IQ4_NL int8 codebook LUT (KVALUES_IQ4NL, k_quants.rs:43): a 4-bit nibble indexes one of 16
// signed int8 values. IQ4_XS is symmetric + LUT-based (no grid, no signs) -> decode = dl*KVALUES[idx].
__device__ __constant__ int8_t KVALUES_IQ4NL[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// dp4a: 4x signed-int8 dot-accumulate. Native on CUDA (sm_61+, incl. GB10 Blackwell).
// Kept under the name `hip_dp4a` so the extracted dp4a_signed + qdp4a<> structs are
// byte-identical to the proven source (zero edits to the decode bodies).
__device__ __forceinline__ int hip_dp4a(int a, int b, int c) { return __dp4a(a, b, c); }

// Store-dispatch by output dtype (the XT template arg of the matvec core).
__device__ __forceinline__ void qstore(__half* p, float v)        { *p = __float2half(v); }
__device__ __forceinline__ void qstore(__nv_bfloat16* p, float v) { *p = __float2bfloat16(v); }
__device__ __forceinline__ void qstore(float* p, float v)         { *p = v; }

// Forward decls (specialized by the extracted i-quant structs/traits below).
template <int WTYPE> struct qdp4a;
template <int WTYPE> struct qdp4a_traits;

// ---- unpack_ksigns + dp4a_signed + the 7 qdp4a<IQ*> decoders + traits ----
// Table-free ksigns (llama.cpp parity trick): the 7-bit index carries 7 sign bits; the 8th is the
// popcount-parity of those 7, so __popc reconstructs it -- NO divergent __constant__ gather. Returns
// the 8 sign bits broadcast to all 4 bytes (== the ksigns_iq2xs[v & 127] codebook value, broadcast),
// ready for the dp4a_signed byte selectors below.
__device__ __forceinline__ uint32_t unpack_ksigns(uint32_t v) {
    v &= 127u;
    return (v ^ ((uint32_t)(__popc(v) & 1) << 7)) * 0x01010101u;
}

// Signed-grid 4-wide int8 dot: sum_i (-1)^sgn_i * g_i * u_i for the 4 magnitude bytes packed in `grid`
// (each in [0,127]) against the q8_1 int `u`. `b` is the 8 sign bits broadcast to every byte (from
// unpack_ksigns, or raw_sign_byte * 0x01010101); `sel` selects which 4 map to these 4 grid bytes --
// 0x08040201 for coords 0-3, 0x80402010 for coords 4-7. Negates each byte in place via CUDA-native
// __vsub4 ((g ^ m) - m = -g where m = 0xFF) -> ONE dp4a. Replaces the 2-dp4a `dp4a(g,u) - 2*dp4a(g &
// mask, u)` form the ROCm port needed for RDNA3.5 (no __vsub4); bit-identical. == llama vec_dot_iq2*.
__device__ __forceinline__ int dp4a_signed(uint32_t b, uint32_t sel, int grid, int u) {
    const int s = __vcmpne4(b & sel, 0);
    return hip_dp4a(__vsub4(grid ^ s, s), u, 0);
}

constexpr uint32_t Q2_LUT_SIGNED = 0x020100FFu;
constexpr uint32_t Q1_LUT_NIBBLE = 0x11100100u;
constexpr uint32_t Q1_LUT_SIGNED = 0x01FF01FFu;
constexpr uint32_t SUB4_ONES     = 0x01010101u;
constexpr uint32_t MASK_16_LO    = 0x00FF00FFu;

__device__ __forceinline__ uint16_t load_u16_b2(const uint8_t* p) {
    return *reinterpret_cast<const uint16_t*>(p);
}

// Unpacks 8 bytes (32 2-bit codes in {0..3}) into 8 signed int8x4 words in {-1, 0, +1, +2}.
__device__ __forceinline__ void q2_unpack32(const uint8_t* __restrict__ b, int (&wp)[8]) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint32_t q = (uint32_t)load_u16_b2(b + 2 * i);
        const uint32_t qe = __byte_perm(Q2_LUT_SIGNED, Q2_LUT_SIGNED, q);
        const uint32_t qo = __byte_perm(Q2_LUT_SIGNED, Q2_LUT_SIGNED, q >> 2);
        wp[2 * i + 0] = (int)__byte_perm(qe, qo, 0x5140u);
        wp[2 * i + 1] = (int)__byte_perm(qe, qo, 0x7362u);
    }
}

// Unpacks 4 bytes (32 LSB-first 1-bit codes) into 8 signed int8x4 words in {-1, +1}.
__device__ __forceinline__ void q1_0_unpack32(const uint8_t* __restrict__ b, int (&wp)[8]) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const uint32_t q = (uint32_t)load_u16_b2(b + 2 * i);
        const uint32_t n0 = __byte_perm(Q1_LUT_NIBBLE, Q1_LUT_NIBBLE, q);
        const uint32_t n1 = __byte_perm(Q1_LUT_NIBBLE, Q1_LUT_NIBBLE, q >> 2);
        const uint32_t s0 = __byte_perm(Q1_LUT_SIGNED, Q1_LUT_SIGNED, n0);
        const uint32_t s1 = __byte_perm(Q1_LUT_SIGNED, Q1_LUT_SIGNED, n1);
        const uint32_t s2 = __byte_perm(Q1_LUT_SIGNED, Q1_LUT_SIGNED, n0 >> 16);
        const uint32_t s3 = __byte_perm(Q1_LUT_SIGNED, Q1_LUT_SIGNED, n1 >> 16);
        wp[4 * i + 0] = (int)__byte_perm(s0, s1, 0x5410u);
        wp[4 * i + 1] = (int)__byte_perm(s0, s1, 0x7632u);
        wp[4 * i + 2] = (int)__byte_perm(s2, s3, 0x5410u);
        wp[4 * i + 3] = (int)__byte_perm(s2, s3, 0x7632u);
    }
}

__device__ __forceinline__ int ptq1_swar4(uint32_t src4, uint32_t p_lo, uint32_t p_hi) {
    const uint32_t vlo = ((__byte_perm(src4, 0, 0x4140u) * p_lo) & MASK_16_LO) * 3u;
    const uint32_t vhi = ((__byte_perm(src4, 0, 0x4342u) * p_hi) & MASK_16_LO) * 3u;
    return (int)__vsub4(__byte_perm(vlo, vhi, 0x7531u), SUB4_ONES);
}

// Predicated SWAR base-3 unpack of 32 trits for 4-byte-aligned 28B PTQ1_0 block without warp divergence.
__device__ __forceinline__ float ptq1_0_unpack32_and_scale(const uint8_t* __restrict__ blk, const int e, int (&wp)[8]) {
    const uint32_t* wb = reinterpret_cast<const uint32_t*>(blk);
    const uint32_t w6 = __ldg(wb + 6);
    const float d = __half2float(__ushort_as_half((uint16_t)(w6 >> 16)));
    const uint32_t s0 = __ldg(wb + ((e == 3) ? 4 : 0));
    const uint32_t s1 = __ldg(wb + ((e == 3) ? 5 : 1));
    const uint32_t s2 = (e == 3) ? s0 : __ldg(wb + 2);
    const uint32_t s3 = (e == 3) ? s1 : __ldg(wb + 3);
    const uint32_t s4 = (e == 2) ? __ldg(wb + 4) : s0;
    const uint32_t s5 = (e == 2) ? __ldg(wb + 5) : s1;
    const uint32_t qhw = (w6 & 0xFFFFu) | (w6 << 16);
    const uint32_t s6 = (e < 2) ? s2 : ((e == 2) ? s4 : qhw);
    const uint32_t s7 = (e < 2) ? s3 : ((e == 2) ? s5 : qhw);

    const int shift = e * 8;
    const uint32_t p01 = (0x09510901u >> shift) & 0xFFu;
    const uint32_t p23 = (0x1B510901u >> shift) & 0xFFu;
    const uint32_t p45 = (0x51011B03u >> shift) & 0xFFu;
    const uint32_t p6_lo = (0x01031B03u >> shift) & 0xFFu;
    const uint32_t p6_hi = (0x03031B03u >> shift) & 0xFFu;
    const uint32_t p7_lo = (0x09031B03u >> shift) & 0xFFu;
    const uint32_t p7_hi = (0x1B031B03u >> shift) & 0xFFu;

    wp[0] = ptq1_swar4(s0, p01, p01);
    wp[1] = ptq1_swar4(s1, p01, p01);
    wp[2] = ptq1_swar4(s2, p23, p23);
    wp[3] = ptq1_swar4(s3, p23, p23);
    wp[4] = ptq1_swar4(s4, p45, p45);
    wp[5] = ptq1_swar4(s5, p45, p45);
    wp[6] = ptq1_swar4(s6, p6_lo, p6_hi);
    wp[7] = ptq1_swar4(s7, p7_lo, p7_hi);
    return d;
}

__device__ __forceinline__ int ptq1_0_elem_swar(const uint8_t* __restrict__ blk, const int tid) {
    const bool seg1 = tid >= 80;
    const bool seg2 = tid >= 120;
    const int off = seg2 ? (24 + ((tid - 120) & 1))
                  : (seg1 ? (16 + ((tid - 80) & 7)) : (tid & 15));
    const int n = seg2 ? ((tid - 120) >> 1)
                : (seg1 ? ((tid - 80) >> 3) : (tid >> 4));
    const uint32_t p = (uint32_t)((0x511B090301ull >> (n * 8)) & 0xFFull);
    const uint32_t q = ((uint32_t)blk[off] * p) & 0xFFu;
    return (int)((q * 3u) >> 8) - 1;
}

template <> struct qdp4a<DW_IQ2_XXS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
        const uint32_t aux32_0 = (uint32_t)qs[4 * e] | ((uint32_t)qs[4 * e + 1] << 16);
        const uint32_t aux32_1 = (uint32_t)qs[4 * e + 2] | ((uint32_t)qs[4 * e + 3] << 16);
        const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;
        int idot = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int gi = (int)((aux32_0 >> (8 * l)) & 0xFF);
            const uint64_t entry = IQ2XXS_GRID_D[gi];
            const uint32_t signs = unpack_ksigns(aux32_1 >> (7 * l));
            idot += dp4a_signed(signs, 0x08040201u, (int)(uint32_t)entry, u[2 * l]);
            idot += dp4a_signed(signs, 0x80402010u, (int)(uint32_t)(entry >> 32), u[2 * l + 1]);
        }
        return db * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

template <> struct qdp4a<DW_IQ2_XS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
        const uint8_t sc = blk[66 + e];
        const float db_lo = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db_hi = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        int idot_lo = 0, idot_hi = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint16_t qv = qs[4 * e + l];
            const uint64_t entry = IQ2XS_GRID_D[qv & 511];
            const uint32_t signs = unpack_ksigns(qv >> 9);
            const int part = dp4a_signed(signs, 0x08040201u, (int)(uint32_t)entry, u[2 * l])
                           + dp4a_signed(signs, 0x80402010u, (int)(uint32_t)(entry >> 32), u[2 * l + 1]);
            if (l < 2) idot_lo += part; else idot_hi += part;
        }
        return d8 * (db_lo * (float)idot_lo + db_hi * (float)idot_hi);
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

template <> struct qdp4a<DW_IQ2_S> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t sc = blk[74 + e];
        const float db_lo = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db_hi = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        int idot_lo = 0, idot_hi = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = (int)qs[4 * e + l] | (((int)qh[e] << (8 - 2 * l)) & 0x300);
            const uint64_t entry = IQ2S_GRID_D[idx];
            const uint32_t signs = (uint32_t)qs[32 + 4 * e + l] * SUB4_ONES;
            const int part = dp4a_signed(signs, 0x08040201u, (int)(uint32_t)entry, u[2 * l])
                           + dp4a_signed(signs, 0x80402010u, (int)(uint32_t)(entry >> 32), u[2 * l + 1]);
            if (l < 2) idot_lo += part; else idot_hi += part;
        }
        return d8 * (db_lo * (float)idot_lo + db_hi * (float)idot_hi);
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

template <> struct qdp4a<DW_IQ3_XXS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* ss = blk + 66;
        const uint32_t aux32 = (uint32_t)ss[4 * e] | ((uint32_t)ss[4 * e + 1] << 8)
                             | ((uint32_t)ss[4 * e + 2] << 16) | ((uint32_t)ss[4 * e + 3] << 24);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        int idot = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l));
            idot += dp4a_signed(signs, 0x08040201u, (int)IQ3XXS_GRID_D[qs[8 * e + 2 * l]], u[2 * l]);
            idot += dp4a_signed(signs, 0x80402010u, (int)IQ3XXS_GRID_D[qs[8 * e + 2 * l + 1]], u[2 * l + 1]);
        }
        return db * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

template <> struct qdp4a<DW_IQ3_S> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t* signs_b = blk + 74;
        const uint8_t* scales = blk + 106;
        const uint8_t sc = scales[e >> 1];
        const int nib = (e & 1) ? (sc >> 4) : (sc & 0xF);
        const float db = d * (1.0f + 2.0f * (float)nib);
        int idot = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx0 = (int)qs[8 * e + 2 * l] | (((int)qh[e] << (8 - 2 * l)) & 256);
            const int idx1 = (int)qs[8 * e + 2 * l + 1] | (((int)qh[e] << (7 - 2 * l)) & 256);
            const uint32_t signs = (uint32_t)signs_b[4 * e + l] * SUB4_ONES;
            idot += dp4a_signed(signs, 0x08040201u, (int)IQ3S_GRID_D[idx0], u[2 * l]);
            idot += dp4a_signed(signs, 0x80402010u, (int)IQ3S_GRID_D[idx1], u[2 * l + 1]);
        }
        return db * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

template <> struct qdp4a<DW_IQ1_S> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint16_t* qh = reinterpret_cast<const uint16_t*>(blk + 34);
        const uint16_t qhv = qh[e];
        const float dl = d * (2.0f * (float)((qhv >> 12) & 7) + 1.0f);
        const float delta = (qhv & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
        int idot = 0, isum = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = (int)qs[e * 4 + l] | ((((int)(qhv >> (3 * l))) & 7) << 8);
            const uint64_t entry = IQ1S_GRID_D[idx];
            idot = hip_dp4a((int)(uint32_t)entry, u[2 * l], idot);
            idot = hip_dp4a((int)(uint32_t)(entry >> 32), u[2 * l + 1], idot);
            isum = hip_dp4a(SUB4_ONES, u[2 * l], isum);
            isum = hip_dp4a(SUB4_ONES, u[2 * l + 1], isum);
        }
        return dl * d8 * ((float)idot + delta * (float)isum);
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

template <> struct qdp4a<DW_IQ1_M> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const uint8_t* qs = blk;
        const uint8_t* qh = blk + 32;
        const uint8_t* scales = blk + 48;
        const uint16_t sc0 = (uint16_t)scales[0] | ((uint16_t)scales[1] << 8);
        const uint16_t sc1 = (uint16_t)scales[2] | ((uint16_t)scales[3] << 8);
        const uint16_t sc2 = (uint16_t)scales[4] | ((uint16_t)scales[5] << 8);
        const uint16_t sc3 = (uint16_t)scales[6] | ((uint16_t)scales[7] << 8);
        const uint16_t scale_u16 = (uint16_t)((sc0 >> 12) | ((sc1 >> 8) & 0x00f0) | ((sc2 >> 4) & 0x0f00) | (sc3 & 0xf000));
        const float d = __half2float(__ushort_as_half(scale_u16));
        const uint16_t sc[4] = {sc0, sc1, sc2, sc3};
        const uint16_t scv = sc[e >> 1];
        const float dl1 = d * (2.0f * (float)((scv >> (6 * (e & 1)))     & 0x7) + 1.0f);
        const float dl2 = d * (2.0f * (float)((scv >> (6 * (e & 1) + 3)) & 0x7) + 1.0f);
        const uint8_t qh0 = qh[e * 2];
        const uint8_t qh1 = qh[e * 2 + 1];
        float fs = 0.0f;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            int idx; float delta; float dl;
            if (l == 0)      { idx = (int)qs[e * 4]     | (((int)qh0 << 8) & 0x700); delta = (qh0 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl1; }
            else if (l == 1) { idx = (int)qs[e * 4 + 1] | (((int)qh0 << 4) & 0x700); delta = (qh0 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl1; }
            else if (l == 2) { idx = (int)qs[e * 4 + 2] | (((int)qh1 << 8) & 0x700); delta = (qh1 & 0x08) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl2; }
            else             { idx = (int)qs[e * 4 + 3] | (((int)qh1 << 4) & 0x700); delta = (qh1 & 0x80) ? -IQ1S_DELTA : IQ1S_DELTA; dl = dl2; }
            const uint64_t entry = IQ1S_GRID_D[idx];
            int idot = hip_dp4a((int)(uint32_t)entry, u[2 * l], 0);
            idot = hip_dp4a((int)(uint32_t)(entry >> 32), u[2 * l + 1], idot);
            int isum = hip_dp4a(SUB4_ONES, u[2 * l], 0);
            isum = hip_dp4a(SUB4_ONES, u[2 * l + 1], isum);
            fs += dl * ((float)idot + delta * (float)isum);
        }
        return d8 * fs;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

template <> struct qdp4a<DW_IQ4_XS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t scales_h = *reinterpret_cast<const uint16_t*>(blk + 2);
        const uint8_t* scales_l = blk + 4;
        const uint8_t* qs = blk + 8 + (size_t)e * 16;
        const int ls = ((scales_l[e >> 1] >> (4 * (e & 1))) & 0xF) | (((scales_h >> (2 * e)) & 3) << 4);
        const float dl = d * (float)(ls - 32);
        int idot = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const uint8_t* b = qs + 4 * (i & 3);
            const bool hi = i >= 4;
            const int n0 = hi ? (b[0] >> 4) : (b[0] & 0xF);
            const int n1 = hi ? (b[1] >> 4) : (b[1] & 0xF);
            const int n2 = hi ? (b[2] >> 4) : (b[2] & 0xF);
            const int n3 = hi ? (b[3] >> 4) : (b[3] & 0xF);
            const int kv = (KVALUES_IQ4NL[n0] & 0xFF) | ((KVALUES_IQ4NL[n1] & 0xFF) << 8)
                         | ((KVALUES_IQ4NL[n2] & 0xFF) << 16) | ((KVALUES_IQ4NL[n3] & 0xFF) << 24);
            idot += hip_dp4a(kv, u[i], 0);
        }
        return dl * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

template <> struct qdp4a<DW_TQ2_0> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 64));
        const int shift = (e & 3) * 2;
        const uint8_t* b = blk + (size_t)(e >> 2) * 32;
        int idot = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int w0 = (int)((b[4 * i + 0] >> shift) & 3) - 1;
            const int w1 = (int)((b[4 * i + 1] >> shift) & 3) - 1;
            const int w2 = (int)((b[4 * i + 2] >> shift) & 3) - 1;
            const int w3 = (int)((b[4 * i + 3] >> shift) & 3) - 1;
            const int wp = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
            idot = hip_dp4a(wp, u[i], idot);
        }
        return d * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

__device__ __forceinline__ int tq1_0_w(const uint8_t* __restrict__ qs, const uint8_t* __restrict__ qh, int p) {
    const bool seg1 = p >= 160;
    const bool seg2 = p >= 240;
    const int off = seg2 ? (48 + ((p - 240) & 3))
                  : (seg1 ? (32 + ((p - 160) & 15)) : (p & 31));
    const int n = seg2 ? ((p - 240) >> 2)
                : (seg1 ? ((p - 160) >> 4) : (p >> 5));
    const uint32_t pow3 = (uint32_t)((0x511B090301ull >> (n * 8)) & 0xFFull);
    const uint32_t b = seg2 ? qh[(p - 240) & 3] : qs[off];
    const uint32_t q = (b * pow3) & 0xFFu;
    return (int)((q * 3u) >> 8) - 1;
}

template <> struct qdp4a<DW_TQ1_0> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const uint8_t* qs = blk;
        const uint8_t* qh = blk + 48;
        const float d = __half2float(*reinterpret_cast<const __half*>(blk + 52));
        const int p0 = e * 32;
        int idot = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int w0 = tq1_0_w(qs, qh, p0 + 4 * i + 0);
            const int w1 = tq1_0_w(qs, qh, p0 + 4 * i + 1);
            const int w2 = tq1_0_w(qs, qh, p0 + 4 * i + 2);
            const int w3 = tq1_0_w(qs, qh, p0 + 4 * i + 3);
            const int wp = (w0 & 0xFF) | ((w1 & 0xFF) << 8) | ((w2 & 0xFF) << 16) | ((w3 & 0xFF) << 24);
            idot = hip_dp4a(wp, u[i], idot);
        }
        return d * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// PQ2_0: QK=128, 34B (f16 d at +0, u8 qs[32] at +2), w_j = (c_j - 1) * d.
template <> struct qdp4a<DW_PQ2_0> {
    static constexpr int NSUB = 4;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        int wp[8];
        q2_unpack32(blk + 2 + e * 8, wp);
        int idot = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            idot = hip_dp4a(wp[i], u[i], idot);
        }
        return d * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// PTQ1_0: QK=128, 28B (qs[24] at +0, qh[2] at +24, f16 d at +26), base-3 ternary {-d, 0, +d}.
template <> struct qdp4a<DW_PTQ1_0> {
    static constexpr int NSUB = 4;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        int wp[8];
        const float d = ptq1_0_unpack32_and_scale(blk, e, wp);
        int idot = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            idot = hip_dp4a(wp[i], u[i], idot);
        }
        return d * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// Q1_0: QK=128, 18B (f16 d at +0, u8 qs[16] at +2), LSB-first binary w_j = (2*b_j - 1) * d.
template <> struct qdp4a<DW_Q1_0> {
    static constexpr int NSUB = 4;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        int wp[8];
        q1_0_unpack32(blk + 2 + e * 4, wp);
        int idot = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            idot = hip_dp4a(wp[i], u[i], idot);
        }
        return d * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// Q2_0: QK=64, 18B (f16 d at +0, u8 qs[16] at +2), w_j = (c_j - 1) * d.
template <> struct qdp4a<DW_Q2_0> {
    static constexpr int NSUB = 2;
    static __device__ __forceinline__ float partial_reg(
            const uint8_t* __restrict__ blk, const int* __restrict__ u, const float d8, int e) {
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        int wp[8];
        q2_unpack32(blk + 2 + e * 8, wp);
        int idot = 0;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            idot = hip_dp4a(wp[i], u[i], idot);
        }
        return d * d8 * (float)idot;
    }
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        return partial_reg(blk, reinterpret_cast<const int*>(xq8) + e * 8, __half2float(xd8[e]), e);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

template <> struct qdp4a_traits<DW_IQ2_XXS> { static constexpr int BYTES = 66;  static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_IQ2_XS>  { static constexpr int BYTES = 74;  static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_IQ2_S>   { static constexpr int BYTES = 82;  static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_IQ3_XXS> { static constexpr int BYTES = 98;  static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_IQ3_S>   { static constexpr int BYTES = 110; static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_IQ1_S>   { static constexpr int BYTES = 50;  static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_IQ1_M>   { static constexpr int BYTES = 56;  static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_IQ4_XS>  { static constexpr int BYTES = 136; static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_TQ2_0>   { static constexpr int BYTES = 66;  static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_TQ1_0>   { static constexpr int BYTES = 54;  static constexpr int ELEMS = 256; };
template <> struct qdp4a_traits<DW_PQ2_0>   { static constexpr int BYTES = 34;  static constexpr int ELEMS = 128; };
template <> struct qdp4a_traits<DW_PTQ1_0>  { static constexpr int BYTES = 28;  static constexpr int ELEMS = 128; };
template <> struct qdp4a_traits<DW_Q1_0>    { static constexpr int BYTES = 18;  static constexpr int ELEMS = 128; };
template <> struct qdp4a_traits<DW_Q2_0>    { static constexpr int BYTES = 18;  static constexpr int ELEMS = 64;  };

constexpr float Q8_MAX_MAG = 127.0f;
constexpr int   Q8_MAX_INT = 127;

__device__ __forceinline__ float fwht_load_f32(const float* p, size_t idx) { return p[idx]; }
__device__ __forceinline__ float fwht_load_f32(const __half* p, size_t idx) { return __half2float(p[idx]); }
__device__ __forceinline__ float fwht_load_f32(const __nv_bfloat16* p, size_t idx) { return __bfloat162float(p[idx]); }

template <typename T>
__device__ __forceinline__ float round_via_dtype(float v);
template <>
__device__ __forceinline__ float round_via_dtype<float>(float v) { return v; }
template <>
__device__ __forceinline__ float round_via_dtype<__half>(float v) { return __half2float(__float2half(v)); }
template <>
__device__ __forceinline__ float round_via_dtype<__nv_bfloat16>(float v) { return __bfloat162float(__float2bfloat16(v)); }

__device__ __forceinline__ float load_weight_f32(const void* w, size_t idx, const int w_dtype) {
    if (w_dtype == 1) {
        return __half2float(static_cast<const __half*>(w)[idx]);
    }
    if (w_dtype == 2) {
        return __bfloat162float(static_cast<const __nv_bfloat16*>(w)[idx]);
    }
    return static_cast<const float*>(w)[idx];
}

// Quantizes M rows of length K into separated (xq, xd) and zeroes padded tail blocks up to total_blocks_padded.
template <typename T>
__device__ __forceinline__ void iq_quantize_q8_core(
    const int M,
    const int K,
    const int total_blocks_padded,
    const T* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    const int nblk = (K + 31) >> 5;
    const int active_blocks = M * nblk;
    const int lane = threadIdx.x & 31;
    const int warps_per_cta = blockDim.x >> 5;
    const int grid_warps = gridDim.x * warps_per_cta;
    const int wid0 = blockIdx.x * warps_per_cta + (threadIdx.x >> 5);

    for (int wid = wid0; wid < total_blocks_padded; wid += grid_warps) {
        if (wid < active_blocks) {
            const int m = wid / nblk;
            const int blk = wid - m * nblk;
            const int col = blk * 32 + lane;
            const float v = (col < K) ? fwht_load_f32(x, (size_t)m * K + col) : 0.0f;
            float a = fabsf(v);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
            }
            const float inv = (a > 0.0f) ? __fdiv_rn(Q8_MAX_MAG, a) : 0.0f;
            int q = (int)roundf(v * inv);
            q = max(-Q8_MAX_INT, min(Q8_MAX_INT, q));
            xq[(size_t)wid * 32 + lane] = (int8_t)q;
            if (lane == 0) {
                xd[wid] = __float2half(a / Q8_MAX_MAG);
            }
        } else {
            xq[(size_t)wid * 32 + lane] = 0;
            if (lane == 0) {
                xd[wid] = __ushort_as_half(0);
            }
        }
    }
}

extern "C" __global__ void iq_quantize_q8_f16(
    const int M,
    const int K,
    const int total_blocks_padded,
    const __half* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    iq_quantize_q8_core<__half>(M, K, total_blocks_padded, x, xq, xd);
}

extern "C" __global__ void iq_quantize_q8_bf16(
    const int M,
    const int K,
    const int total_blocks_padded,
    const __nv_bfloat16* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    iq_quantize_q8_core<__nv_bfloat16>(M, K, total_blocks_padded, x, xq, xd);
}

extern "C" __global__ void iq_quantize_q8_f32(
    const int M,
    const int K,
    const int total_blocks_padded,
    const float* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    iq_quantize_q8_core<float>(M, K, total_blocks_padded, x, xq, xd);
}

// Multi-row-per-warp decode (NR = 4): loads (xq, xd) once per K sub-unit and accumulates across NR weight rows.
template <int WTYPE, typename XT, int NR = 4>
__device__ __forceinline__ void qmatvec_dp4a_core(
    const int nrows,
    const int ncols,
    const uint8_t* __restrict__ wq,
    const int8_t*  __restrict__ xq,
    const __half*  __restrict__ xd,
    XT* __restrict__ y
) {
    constexpr int WBYTES = qdp4a_traits<WTYPE>::BYTES;
    constexpr int WELEMS = qdp4a_traits<WTYPE>::ELEMS;
    constexpr int WSUB32 = WELEMS / 32;
    constexpr int NSUB = qdp4a<WTYPE>::NSUB;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int row0 = (blockIdx.x * warps_per_block + (threadIdx.x >> 5)) * NR;
    if (row0 >= nrows) {
        return;
    }
    const int rem = nrows - row0;
    const int nblocks = ncols / WELEMS;
    const size_t row_stride = (size_t)nblocks * WBYTES;
    const uint8_t* row_ptr0 = wq + (size_t)row0 * row_stride;
    const int nunits = nblocks * NSUB;

    float acc[NR];
    #pragma unroll
    for (int r = 0; r < NR; ++r) {
        acc[r] = 0.0f;
    }

    if (rem >= NR) {
        for (int u = lane; u < nunits; u += 32) {
            const int b = u / NSUB;
            const int sub = u - b * NSUB;
            const float d8 = __half2float(__ldg(xd + (size_t)b * WSUB32 + sub));
            const int4* xq4 = reinterpret_cast<const int4*>(xq + (size_t)b * WELEMS + sub * 32);
            const int4 u0 = __ldg(xq4 + 0);
            const int4 u1 = __ldg(xq4 + 1);
            const int u_reg[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
            const size_t blk_off = (size_t)b * WBYTES;
            #pragma unroll
            for (int r = 0; r < NR; ++r) {
                acc[r] += qdp4a<WTYPE>::partial_reg(row_ptr0 + (size_t)r * row_stride + blk_off, u_reg, d8, sub);
            }
        }
        #pragma unroll
        for (int r = 0; r < NR; ++r) {
            float v = acc[r];
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_down_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                qstore(&y[row0 + r], v);
            }
        }
    } else {
        for (int u = lane; u < nunits; u += 32) {
            const int b = u / NSUB;
            const int sub = u - b * NSUB;
            const float d8 = __half2float(__ldg(xd + (size_t)b * WSUB32 + sub));
            const int4* xq4 = reinterpret_cast<const int4*>(xq + (size_t)b * WELEMS + sub * 32);
            const int4 u0 = __ldg(xq4 + 0);
            const int4 u1 = __ldg(xq4 + 1);
            const int u_reg[8] = {u0.x, u0.y, u0.z, u0.w, u1.x, u1.y, u1.z, u1.w};
            const size_t blk_off = (size_t)b * WBYTES;
            #pragma unroll
            for (int r = 0; r < NR; ++r) {
                if (r < rem) {
                    acc[r] += qdp4a<WTYPE>::partial_reg(row_ptr0 + (size_t)r * row_stride + blk_off, u_reg, d8, sub);
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < NR; ++r) {
            if (r < rem) {
                float v = acc[r];
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    v += __shfl_down_sync(0xffffffffu, v, off);
                }
                if (lane == 0) {
                    qstore(&y[row0 + r], v);
                }
            }
        }
    }
}

template <int WTYPE, typename XT>
__device__ __forceinline__ void moe_qmatvec_dp4a_core(
    const int n,
    const int ncols,
    const int nslots,
    const uint8_t* __restrict__ wbank,
    const int* __restrict__ ids,
    const int8_t* __restrict__ xq,
    const __half* __restrict__ xd,
    XT* __restrict__ y
) {
    constexpr int WBYTES = qdp4a_traits<WTYPE>::BYTES;
    constexpr int WELEMS = qdp4a_traits<WTYPE>::ELEMS;
    constexpr int WSUB32 = WELEMS / 32;
    const int s = blockIdx.y;
    if (s >= nslots) {
        return;
    }
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= n) {
        return;
    }
    const int nblocks = ncols / WELEMS;
    const int expert = ids[s];
    const uint8_t* row_ptr = wbank + ((size_t)expert * n + row) * (size_t)nblocks * WBYTES;
    const int8_t* xq_row = xq + (size_t)s * ncols;
    const __half* xd_row = xd + (size_t)s * (ncols >> 5);

    constexpr int NSUB = qdp4a<WTYPE>::NSUB;
    const int nunits = nblocks * NSUB;
    float acc = 0.0f;
    for (int u = lane; u < nunits; u += 32) {
        const int b = u / NSUB;
        const int sub = u - b * NSUB;
        acc += qdp4a<WTYPE>::partial(
            row_ptr + (size_t)b * WBYTES, xq_row + (size_t)b * WELEMS, xd_row + (size_t)b * WSUB32, sub);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        qstore(&y[(size_t)s * n + row], acc);
    }
}

template <int WTYPE, int BM>
struct qdp4a_batch {
    static __device__ __forceinline__ void accum(
        const uint8_t* __restrict__ blk,
        const int8_t*  __restrict__ xq_base,
        const __half*  __restrict__ xd_base,
        const int ncols,
        const int nblk32,
        const int b_elem_off,
        const int b_sub_off,
        const int sub,
        const int m_cnt,
        float (&acc)[BM]
    ) {
        for (int m = 0; m < m_cnt; ++m) {
            acc[m] += qdp4a<WTYPE>::partial(
                blk,
                xq_base + (size_t)m * ncols + b_elem_off,
                xd_base + (size_t)m * nblk32 + b_sub_off,
                sub);
        }
    }
};

template <int BM>
struct qdp4a_batch<DW_PQ2_0, BM> {
    static __device__ __forceinline__ void accum(
        const uint8_t* __restrict__ blk,
        const int8_t*  __restrict__ xq_base,
        const __half*  __restrict__ xd_base,
        const int ncols,
        const int nblk32,
        const int b_elem_off,
        const int b_sub_off,
        const int e,
        const int m_cnt,
        float (&acc)[BM]
    ) {
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        int wp[8];
        q2_unpack32(blk + 2 + e * 8, wp);
        for (int m = 0; m < m_cnt; ++m) {
            const float d8 = __half2float(xd_base[(size_t)m * nblk32 + b_sub_off + e]);
            const int* u = reinterpret_cast<const int*>(xq_base + (size_t)m * ncols + b_elem_off) + e * 8;
            int idot = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                idot = hip_dp4a(wp[i], u[i], idot);
            }
            acc[m] += d * d8 * (float)idot;
        }
    }
};

template <int BM>
struct qdp4a_batch<DW_PTQ1_0, BM> {
    static __device__ __forceinline__ void accum(
        const uint8_t* __restrict__ blk,
        const int8_t*  __restrict__ xq_base,
        const __half*  __restrict__ xd_base,
        const int ncols,
        const int nblk32,
        const int b_elem_off,
        const int b_sub_off,
        const int e,
        const int m_cnt,
        float (&acc)[BM]
    ) {
        int wp[8];
        const float d = ptq1_0_unpack32_and_scale(blk, e, wp);
        for (int m = 0; m < m_cnt; ++m) {
            const float d8 = __half2float(xd_base[(size_t)m * nblk32 + b_sub_off + e]);
            const int* u = reinterpret_cast<const int*>(xq_base + (size_t)m * ncols + b_elem_off) + e * 8;
            int idot = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                idot = hip_dp4a(wp[i], u[i], idot);
            }
            acc[m] += d * d8 * (float)idot;
        }
    }
};

template <int BM>
struct qdp4a_batch<DW_Q1_0, BM> {
    static __device__ __forceinline__ void accum(
        const uint8_t* __restrict__ blk,
        const int8_t*  __restrict__ xq_base,
        const __half*  __restrict__ xd_base,
        const int ncols,
        const int nblk32,
        const int b_elem_off,
        const int b_sub_off,
        const int e,
        const int m_cnt,
        float (&acc)[BM]
    ) {
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        int wp[8];
        q1_0_unpack32(blk + 2 + e * 4, wp);
        for (int m = 0; m < m_cnt; ++m) {
            const float d8 = __half2float(xd_base[(size_t)m * nblk32 + b_sub_off + e]);
            const int* u = reinterpret_cast<const int*>(xq_base + (size_t)m * ncols + b_elem_off) + e * 8;
            int idot = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                idot = hip_dp4a(wp[i], u[i], idot);
            }
            acc[m] += d * d8 * (float)idot;
        }
    }
};

template <int BM>
struct qdp4a_batch<DW_Q2_0, BM> {
    static __device__ __forceinline__ void accum(
        const uint8_t* __restrict__ blk,
        const int8_t*  __restrict__ xq_base,
        const __half*  __restrict__ xd_base,
        const int ncols,
        const int nblk32,
        const int b_elem_off,
        const int b_sub_off,
        const int e,
        const int m_cnt,
        float (&acc)[BM]
    ) {
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        int wp[8];
        q2_unpack32(blk + 2 + e * 8, wp);
        for (int m = 0; m < m_cnt; ++m) {
            const float d8 = __half2float(xd_base[(size_t)m * nblk32 + b_sub_off + e]);
            const int* u = reinterpret_cast<const int*>(xq_base + (size_t)m * ncols + b_elem_off) + e * 8;
            int idot = 0;
            #pragma unroll
            for (int i = 0; i < 8; ++i) {
                idot = hip_dp4a(wp[i], u[i], idot);
            }
            acc[m] += d * d8 * (float)idot;
        }
    }
};

template <int WTYPE, typename XT, int BM = 16>
__device__ __forceinline__ void qmatmul_dp4a_core(
    const int nrows,
    const int ncols,
    const int b_size,
    const uint8_t* __restrict__ wq,
    const int8_t*  __restrict__ xq,
    const __half*  __restrict__ xd,
    XT* __restrict__ y
) {
    constexpr int WBYTES = qdp4a_traits<WTYPE>::BYTES;
    constexpr int WELEMS = qdp4a_traits<WTYPE>::ELEMS;
    constexpr int WSUB32 = WELEMS / 32;
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    const int m0 = blockIdx.y * BM;
    if (row >= nrows || m0 >= b_size) {
        return;
    }
    const int m_cnt = min(BM, b_size - m0);
    const int nblocks = ncols / WELEMS;
    const int nblk32 = ncols >> 5;
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * WBYTES;
    const int8_t* xq_base = xq + (size_t)m0 * ncols;
    const __half* xd_base = xd + (size_t)m0 * nblk32;

    float acc[BM];
    #pragma unroll
    for (int m = 0; m < BM; ++m) {
        acc[m] = 0.0f;
    }

    constexpr int NSUB = qdp4a<WTYPE>::NSUB;
    const int nunits = nblocks * NSUB;
    for (int u = lane; u < nunits; u += 32) {
        const int b = u / NSUB;
        const int sub = u - b * NSUB;
        qdp4a_batch<WTYPE, BM>::accum(
            row_ptr + (size_t)b * WBYTES,
            xq_base,
            xd_base,
            ncols,
            nblk32,
            b * WELEMS,
            b * WSUB32,
            sub,
            m_cnt,
            acc);
    }

    for (int m = 0; m < m_cnt; ++m) {
        float v = acc[m];
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (lane == 0) {
            qstore(&y[(size_t)(m0 + m) * nrows + row], v);
        }
    }
}

#define DEFINE_QMATVEC_DP4A(NAME, WTYPE)                                                             \
    extern "C" __global__ void qmatvec_dp4a_##NAME##_f16(                                            \
        const int nrows, const int ncols, const uint8_t* __restrict__ wq,                            \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd, __half* __restrict__ y) {      \
        qmatvec_dp4a_core<WTYPE, __half>(nrows, ncols, wq, xq, xd, y);                                \
    }                                                                                                \
    extern "C" __global__ void qmatvec_dp4a_##NAME##_bf16(                                           \
        const int nrows, const int ncols, const uint8_t* __restrict__ wq,                            \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd, __nv_bfloat16* __restrict__ y) {\
        qmatvec_dp4a_core<WTYPE, __nv_bfloat16>(nrows, ncols, wq, xq, xd, y);                          \
    }                                                                                                \
    extern "C" __global__ void qmatmul_dp4a_##NAME##_f16(                                            \
        const int nrows, const int ncols, const int b_size, const uint8_t* __restrict__ wq,          \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd, __half* __restrict__ y) {      \
        qmatmul_dp4a_core<WTYPE, __half>(nrows, ncols, b_size, wq, xq, xd, y);                        \
    }                                                                                                \
    extern "C" __global__ void qmatmul_dp4a_##NAME##_bf16(                                           \
        const int nrows, const int ncols, const int b_size, const uint8_t* __restrict__ wq,          \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd, __nv_bfloat16* __restrict__ y) {\
        qmatmul_dp4a_core<WTYPE, __nv_bfloat16>(nrows, ncols, b_size, wq, xq, xd, y);                  \
    }                                                                                                \
    extern "C" __global__ void qmatmul_dp4a_##NAME##_f32(                                            \
        const int nrows, const int ncols, const int b_size, const uint8_t* __restrict__ wq,          \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd, float* __restrict__ y) {       \
        qmatmul_dp4a_core<WTYPE, float>(nrows, ncols, b_size, wq, xq, xd, y);                         \
    }                                                                                                \
    extern "C" __global__ void moe_qmatvec_dp4a_##NAME##_f16(                                        \
        const int n, const int ncols, const int nslots, const uint8_t* __restrict__ wbank,           \
        const int* __restrict__ ids, const int8_t* __restrict__ xq, const __half* __restrict__ xd,   \
        __half* __restrict__ y) {                                                                    \
        moe_qmatvec_dp4a_core<WTYPE, __half>(n, ncols, nslots, wbank, ids, xq, xd, y);                \
    }                                                                                                \
    extern "C" __global__ void moe_qmatvec_dp4a_##NAME##_bf16(                                       \
        const int n, const int ncols, const int nslots, const uint8_t* __restrict__ wbank,           \
        const int* __restrict__ ids, const int8_t* __restrict__ xq, const __half* __restrict__ xd,   \
        __nv_bfloat16* __restrict__ y) {                                                              \
        moe_qmatvec_dp4a_core<WTYPE, __nv_bfloat16>(n, ncols, nslots, wbank, ids, xq, xd, y);          \
    }                                                                                                \
    extern "C" __global__ void qmatvec_dp4a_##NAME##_f32(                                            \
        const int nrows, const int ncols, const uint8_t* __restrict__ wq,                            \
        const int8_t* __restrict__ xq, const __half* __restrict__ xd, float* __restrict__ y) {       \
        qmatvec_dp4a_core<WTYPE, float>(nrows, ncols, wq, xq, xd, y);                                 \
    }                                                                                                \
    extern "C" __global__ void moe_qmatvec_dp4a_##NAME##_f32(                                        \
        const int n, const int ncols, const int nslots, const uint8_t* __restrict__ wbank,           \
        const int* __restrict__ ids, const int8_t* __restrict__ xq, const __half* __restrict__ xd,   \
        float* __restrict__ y) {                                                                     \
        moe_qmatvec_dp4a_core<WTYPE, float>(n, ncols, nslots, wbank, ids, xq, xd, y);                 \
    }

DEFINE_QMATVEC_DP4A(iq2xxs, DW_IQ2_XXS)
DEFINE_QMATVEC_DP4A(iq2xs,  DW_IQ2_XS)
DEFINE_QMATVEC_DP4A(iq2s,   DW_IQ2_S)
DEFINE_QMATVEC_DP4A(iq3xxs, DW_IQ3_XXS)
DEFINE_QMATVEC_DP4A(iq3s,   DW_IQ3_S)
DEFINE_QMATVEC_DP4A(iq1_s,  DW_IQ1_S)
DEFINE_QMATVEC_DP4A(iq1_m,  DW_IQ1_M)
DEFINE_QMATVEC_DP4A(iq4xs,  DW_IQ4_XS)
DEFINE_QMATVEC_DP4A(tq2_0,  DW_TQ2_0)
DEFINE_QMATVEC_DP4A(tq1_0,  DW_TQ1_0)
DEFINE_QMATVEC_DP4A(pq2_0,  DW_PQ2_0)
DEFINE_QMATVEC_DP4A(ptq1_0, DW_PTQ1_0)
DEFINE_QMATVEC_DP4A(q1_0,   DW_Q1_0)
DEFINE_QMATVEC_DP4A(q2_0,   DW_Q2_0)

extern "C" __global__ void qmatvec_q2_0_dp4a_f16(
    const int nrows, const int ncols, const uint8_t* __restrict__ wq,
    const int8_t* __restrict__ xq, const __half* __restrict__ xd, __half* __restrict__ y) {
    qmatvec_dp4a_core<DW_Q2_0, __half>(nrows, ncols, wq, xq, xd, y);
}

extern "C" __global__ void qmatvec_q2_0_dp4a_bf16(
    const int nrows, const int ncols, const uint8_t* __restrict__ wq,
    const int8_t* __restrict__ xq, const __half* __restrict__ xd, __nv_bfloat16* __restrict__ y) {
    qmatvec_dp4a_core<DW_Q2_0, __nv_bfloat16>(nrows, ncols, wq, xq, xd, y);
}

extern "C" __global__ void qmatvec_q2_0_dp4a_f32(
    const int nrows, const int ncols, const uint8_t* __restrict__ wq,
    const int8_t* __restrict__ xq, const __half* __restrict__ xd, float* __restrict__ y) {
    qmatvec_dp4a_core<DW_Q2_0, float>(nrows, ncols, wq, xq, xd, y);
}

extern "C" __global__ void qmatmul_q2_0_dp4a_f16(
    const int nrows, const int ncols, const int b_size, const uint8_t* __restrict__ wq,
    const int8_t* __restrict__ xq, const __half* __restrict__ xd, __half* __restrict__ y) {
    qmatmul_dp4a_core<DW_Q2_0, __half>(nrows, ncols, b_size, wq, xq, xd, y);
}

extern "C" __global__ void qmatmul_q2_0_dp4a_bf16(
    const int nrows, const int ncols, const int b_size, const uint8_t* __restrict__ wq,
    const int8_t* __restrict__ xq, const __half* __restrict__ xd, __nv_bfloat16* __restrict__ y) {
    qmatmul_dp4a_core<DW_Q2_0, __nv_bfloat16>(nrows, ncols, b_size, wq, xq, xd, y);
}

extern "C" __global__ void qmatmul_q2_0_dp4a_f32(
    const int nrows, const int ncols, const int b_size, const uint8_t* __restrict__ wq,
    const int8_t* __restrict__ xq, const __half* __restrict__ xd, float* __restrict__ y) {
    qmatmul_dp4a_core<DW_Q2_0, float>(nrows, ncols, b_size, wq, xq, xd, y);
}

template <typename DST_T>
__device__ __forceinline__ void dequantize_pq2_0_impl(
    const uint8_t* __restrict__ vx,
    DST_T* __restrict__ y,
    const int64_t nblocks
) {
    const int tid = threadIdx.x;
    if (tid >= 128) return;
    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const uint8_t* blk = vx + (size_t)ib * 34;
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        const uint8_t byte = blk[2 + (tid >> 2)];
        const int q = (int)((byte >> ((tid & 3) * 2)) & 3) - 1;
        qstore(&y[(size_t)ib * 128 + tid], (float)q * d);
    }
}

template <typename DST_T>
__device__ __forceinline__ void dequantize_ptq1_0_impl(
    const uint8_t* __restrict__ vx,
    DST_T* __restrict__ y,
    const int64_t nblocks
) {
    const int tid = threadIdx.x;
    if (tid >= 128) return;
    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const uint8_t* blk = vx + (size_t)ib * 28;
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk + 26)));
        const int q = ptq1_0_elem_swar(blk, tid);
        qstore(&y[(size_t)ib * 128 + tid], (float)q * d);
    }
}

template <typename DST_T>
__device__ __forceinline__ void dequantize_q1_0_impl(
    const uint8_t* __restrict__ vx,
    DST_T* __restrict__ y,
    const int64_t nblocks
) {
    const int tid = threadIdx.x;
    if (tid >= 128) return;
    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const uint8_t* blk = vx + (size_t)ib * 18;
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        const int bit = (blk[2 + (tid >> 3)] >> (tid & 7)) & 1;
        qstore(&y[(size_t)ib * 128 + tid], bit ? d : -d);
    }
}

template <typename DST_T>
__device__ __forceinline__ void dequantize_q2_0_impl(
    const uint8_t* __restrict__ vx,
    DST_T* __restrict__ y,
    const int64_t nblocks
) {
    const int tid = threadIdx.x;
    if (tid >= 64) return;
    for (int64_t ib = blockIdx.x; ib < nblocks; ib += gridDim.x) {
        const uint8_t* blk = vx + (size_t)ib * 18;
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        const uint8_t byte = blk[2 + (tid >> 2)];
        const int q = (int)((byte >> ((tid & 3) * 2)) & 3) - 1;
        qstore(&y[(size_t)ib * 64 + tid], (float)q * d);
    }
}

extern "C" __global__ void dequantize_block_pq2_0_f32(
    const uint8_t* __restrict__ vx, float* __restrict__ y, const int64_t nblocks) {
    dequantize_pq2_0_impl<float>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_pq2_0_f16(
    const uint8_t* __restrict__ vx, __half* __restrict__ y, const int64_t nblocks) {
    dequantize_pq2_0_impl<__half>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_pq2_0_bf16(
    const uint8_t* __restrict__ vx, __nv_bfloat16* __restrict__ y, const int64_t nblocks) {
    dequantize_pq2_0_impl<__nv_bfloat16>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_ptq1_0_f32(
    const uint8_t* __restrict__ vx, float* __restrict__ y, const int64_t nblocks) {
    dequantize_ptq1_0_impl<float>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_ptq1_0_f32(
    const uint8_t* __restrict__ vx, float* __restrict__ y, const int64_t nblocks) {
    dequantize_ptq1_0_impl<float>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_ptq1_0_f16(
    const uint8_t* __restrict__ vx, __half* __restrict__ y, const int64_t nblocks) {
    dequantize_ptq1_0_impl<__half>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_ptq1_0_bf16(
    const uint8_t* __restrict__ vx, __nv_bfloat16* __restrict__ y, const int64_t nblocks) {
    dequantize_ptq1_0_impl<__nv_bfloat16>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_q1_0_f32(
    const uint8_t* __restrict__ vx, float* __restrict__ y, const int64_t nblocks) {
    dequantize_q1_0_impl<float>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_q1_0_f32(
    const uint8_t* __restrict__ vx, float* __restrict__ y, const int64_t nblocks) {
    dequantize_q1_0_impl<float>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_q1_0_f16(
    const uint8_t* __restrict__ vx, __half* __restrict__ y, const int64_t nblocks) {
    dequantize_q1_0_impl<__half>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_q1_0_bf16(
    const uint8_t* __restrict__ vx, __nv_bfloat16* __restrict__ y, const int64_t nblocks) {
    dequantize_q1_0_impl<__nv_bfloat16>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_q2_0_f32(
    const uint8_t* __restrict__ vx, float* __restrict__ y, const int64_t nblocks) {
    dequantize_q2_0_impl<float>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_q2_0_f32(
    const uint8_t* __restrict__ vx, float* __restrict__ y, const int64_t nblocks) {
    dequantize_q2_0_impl<float>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_q2_0_f16(
    const uint8_t* __restrict__ vx, __half* __restrict__ y, const int64_t nblocks) {
    dequantize_q2_0_impl<__half>(vx, y, nblocks);
}

extern "C" __global__ void dequantize_block_q2_0_bf16(
    const uint8_t* __restrict__ vx, __nv_bfloat16* __restrict__ y, const int64_t nblocks) {
    dequantize_q2_0_impl<__nv_bfloat16>(vx, y, nblocks);
}

extern "C" __global__ void get_rows_pq2_0(
    const uint8_t* __restrict__ src0,
    const uint32_t* __restrict__ src1,
    float* __restrict__ dst,
    const int64_t ne00,
    const size_t nb01
) {
    const int tid = threadIdx.x;
    if (tid >= 128) return;
    const size_t i10 = (size_t)blockIdx.x;
    const size_t i01 = (size_t)src1[i10];
    const uint8_t* src_row = src0 + i01 * nb01;
    float* dst_row = dst + i10 * (size_t)ne00;
    const int64_t blocks_per_row = ne00 / 128;
    for (int64_t ib = blockIdx.y; ib < blocks_per_row; ib += gridDim.y) {
        const uint8_t* blk = src_row + (size_t)ib * 34;
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        const uint8_t byte = blk[2 + (tid >> 2)];
        const int q = (int)((byte >> ((tid & 3) * 2)) & 3) - 1;
        dst_row[(size_t)ib * 128 + tid] = (float)q * d;
    }
}

extern "C" __global__ void get_rows_ptq1_0(
    const uint8_t* __restrict__ src0,
    const uint32_t* __restrict__ src1,
    float* __restrict__ dst,
    const int64_t ne00,
    const size_t nb01
) {
    const int tid = threadIdx.x;
    if (tid >= 128) return;
    const size_t i10 = (size_t)blockIdx.x;
    const size_t i01 = (size_t)src1[i10];
    const uint8_t* src_row = src0 + i01 * nb01;
    float* dst_row = dst + i10 * (size_t)ne00;
    const int64_t blocks_per_row = ne00 / 128;
    for (int64_t ib = blockIdx.y; ib < blocks_per_row; ib += gridDim.y) {
        const uint8_t* blk = src_row + (size_t)ib * 28;
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk + 26)));
        const int q = ptq1_0_elem_swar(blk, tid);
        dst_row[(size_t)ib * 128 + tid] = (float)q * d;
    }
}

template <typename DST_T>
__device__ __forceinline__ void get_rows_q1_0_impl(
    const uint8_t* __restrict__ src0,
    const uint32_t* __restrict__ src1,
    DST_T* __restrict__ dst,
    const int64_t ne00,
    const size_t nb01
) {
    const int tid = threadIdx.x;
    if (tid >= 128) return;
    const size_t i10 = (size_t)blockIdx.x;
    const size_t i01 = (size_t)src1[i10];
    const uint8_t* src_row = src0 + i01 * nb01;
    DST_T* dst_row = dst + i10 * (size_t)ne00;
    const int64_t blocks_per_row = ne00 / 128;
    for (int64_t ib = blockIdx.y; ib < blocks_per_row; ib += gridDim.y) {
        const uint8_t* blk = src_row + (size_t)ib * 18;
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        const int bit = (blk[2 + (tid >> 3)] >> (tid & 7)) & 1;
        qstore(&dst_row[(size_t)ib * 128 + tid], bit ? d : -d);
    }
}

template <typename DST_T>
__device__ __forceinline__ void get_rows_q2_0_impl(
    const uint8_t* __restrict__ src0,
    const uint32_t* __restrict__ src1,
    DST_T* __restrict__ dst,
    const int64_t ne00,
    const size_t nb01
) {
    const int tid = threadIdx.x;
    if (tid >= 64) return;
    const size_t i10 = (size_t)blockIdx.x;
    const size_t i01 = (size_t)src1[i10];
    const uint8_t* src_row = src0 + i01 * nb01;
    DST_T* dst_row = dst + i10 * (size_t)ne00;
    const int64_t blocks_per_row = ne00 / 64;
    for (int64_t ib = blockIdx.y; ib < blocks_per_row; ib += gridDim.y) {
        const uint8_t* blk = src_row + (size_t)ib * 18;
        const float d = __half2float(__ushort_as_half(load_u16_b2(blk)));
        const uint8_t byte = blk[2 + (tid >> 2)];
        const int q = (int)((byte >> ((tid & 3) * 2)) & 3) - 1;
        qstore(&dst_row[(size_t)ib * 64 + tid], (float)q * d);
    }
}

extern "C" __global__ void get_rows_q1_0(
    const uint8_t* __restrict__ src0, const uint32_t* __restrict__ src1,
    float* __restrict__ dst, const int64_t ne00, const size_t nb01) {
    get_rows_q1_0_impl<float>(src0, src1, dst, ne00, nb01);
}

extern "C" __global__ void get_rows_q1_0_f32(
    const uint8_t* __restrict__ src0, const uint32_t* __restrict__ src1,
    float* __restrict__ dst, const int64_t ne00, const size_t nb01) {
    get_rows_q1_0_impl<float>(src0, src1, dst, ne00, nb01);
}

extern "C" __global__ void get_rows_q1_0_bf16(
    const uint8_t* __restrict__ src0, const uint32_t* __restrict__ src1,
    __nv_bfloat16* __restrict__ dst, const int64_t ne00, const size_t nb01) {
    get_rows_q1_0_impl<__nv_bfloat16>(src0, src1, dst, ne00, nb01);
}

extern "C" __global__ void get_rows_q1_0_f16(
    const uint8_t* __restrict__ src0, const uint32_t* __restrict__ src1,
    __half* __restrict__ dst, const int64_t ne00, const size_t nb01) {
    get_rows_q1_0_impl<__half>(src0, src1, dst, ne00, nb01);
}

extern "C" __global__ void get_rows_q2_0(
    const uint8_t* __restrict__ src0, const uint32_t* __restrict__ src1,
    float* __restrict__ dst, const int64_t ne00, const size_t nb01) {
    get_rows_q2_0_impl<float>(src0, src1, dst, ne00, nb01);
}

extern "C" __global__ void get_rows_q2_0_f32(
    const uint8_t* __restrict__ src0, const uint32_t* __restrict__ src1,
    float* __restrict__ dst, const int64_t ne00, const size_t nb01) {
    get_rows_q2_0_impl<float>(src0, src1, dst, ne00, nb01);
}

extern "C" __global__ void get_rows_q2_0_bf16(
    const uint8_t* __restrict__ src0, const uint32_t* __restrict__ src1,
    __nv_bfloat16* __restrict__ dst, const int64_t ne00, const size_t nb01) {
    get_rows_q2_0_impl<__nv_bfloat16>(src0, src1, dst, ne00, nb01);
}

extern "C" __global__ void get_rows_q2_0_f16(
    const uint8_t* __restrict__ src0, const uint32_t* __restrict__ src1,
    __half* __restrict__ dst, const int64_t ne00, const size_t nb01) {
    get_rows_q2_0_impl<__half>(src0, src1, dst, ne00, nb01);
}

// Wide-CTA block-FWHT butterfly: ascending-stride in-reg + warp shuffles + single SMEM transpose.
template <int NT, int E>
__device__ __forceinline__ void fwht_block_butterfly(float (&v)[E], float* smem) {
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    #pragma unroll
    for (int stride = 1; stride < E; stride <<= 1) {
        #pragma unroll
        for (int base_r = 0; base_r < E; base_r += 2 * stride) {
            #pragma unroll
            for (int off = 0; off < stride; ++off) {
                const float a = v[base_r + off];
                const float b = v[base_r + off + stride];
                v[base_r + off] = a + b;
                v[base_r + off + stride] = a - b;
            }
        }
    }

    #pragma unroll
    for (int ls = 1; ls < 32; ls <<= 1) {
        const bool hi = (lane & ls) != 0;
        #pragma unroll
        for (int r = 0; r < E; ++r) {
            const float partner = __shfl_xor_sync(0xffffffffu, v[r], ls);
            v[r] = hi ? (partner - v[r]) : (v[r] + partner);
        }
    }

    if constexpr (NT > 32) {
        constexpr int NWARPS = NT / 32;
        constexpr int W = 32 * E;
        constexpr int PAD_W = W + 1;
        constexpr int GRP_PER_WARP = 32 / NWARPS;

        #pragma unroll
        for (int r = 0; r < E; ++r) {
            smem[warp_id * PAD_W + lane * E + r] = v[r];
        }
        __syncthreads();

        const int row_idx = lane & (NWARPS - 1);
        const int col_base = (warp_id * GRP_PER_WARP + (lane / NWARPS)) * E;
        float tv[E];
        #pragma unroll
        for (int r = 0; r < E; ++r) {
            tv[r] = smem[row_idx * PAD_W + col_base + r];
        }

        #pragma unroll
        for (int ls = 1; ls < NWARPS; ls <<= 1) {
            const bool hi = (row_idx & ls) != 0;
            #pragma unroll
            for (int r = 0; r < E; ++r) {
                const float partner = __shfl_xor_sync(0xffffffffu, tv[r], ls);
                tv[r] = hi ? (partner - tv[r]) : (tv[r] + partner);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < E; ++r) {
            smem[row_idx * PAD_W + col_base + r] = tv[r];
        }
        __syncthreads();

        #pragma unroll
        for (int r = 0; r < E; ++r) {
            v[r] = smem[warp_id * PAD_W + lane * E + r];
        }
    }
}

template <int E>
__device__ __forceinline__ void fwht_quantize_q8_in_reg(
    const float (&out_v)[E],
    const size_t elem_off,
    const size_t qblk_base,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    constexpr int G = 32 / E;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    float local_max = 0.0f;
    #pragma unroll
    for (int r = 0; r < E; ++r) {
        local_max = fmaxf(local_max, fabsf(out_v[r]));
    }
    #pragma unroll
    for (int off = G >> 1; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, off));
    }
    const float inv = (local_max > 0.0f) ? __fdiv_rn(Q8_MAX_MAG, local_max) : 0.0f;
    #pragma unroll
    for (int r = 0; r < E; ++r) {
        int q = (int)roundf(out_v[r] * inv);
        q = max(-Q8_MAX_INT, min(Q8_MAX_INT, q));
        xq[elem_off + r] = (int8_t)q;
    }
    if ((lane & (G - 1)) == 0) {
        xd[qblk_base + warp_id * E + (lane / G)] = __float2half(local_max / Q8_MAX_MAG);
    }
}

__device__ __forceinline__ void zero_q8_tail_blocks(
    const int64_t active_blocks,
    const int64_t total_blocks_padded,
    const int nt,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    if (total_blocks_padded <= active_blocks) return;
    const int tid = threadIdx.x;
    if (tid >= nt) return;
    const int lane = tid & 31;
    const int warps_per_cta = nt >> 5;
    if (warps_per_cta <= 0) return;
    const int64_t total_warps = (int64_t)gridDim.x * warps_per_cta;
    const int64_t wid0 = (int64_t)blockIdx.x * warps_per_cta + (tid >> 5);
    for (int64_t qb = active_blocks + wid0; qb < total_blocks_padded; qb += total_warps) {
        xq[qb * 32 + lane] = 0;
        if (lane == 0) {
            xd[qb] = __ushort_as_half(0);
        }
    }
}

template <typename T>
__device__ __forceinline__ void hadamard_fwht_small(
    const T* __restrict__ x,
    const float* __restrict__ signs,
    T* __restrict__ y,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const int block_size,
    const int width,
    const int has_signs,
    const int is_inverse,
    const int has_y,
    const int has_q8,
    const int64_t total_blocks
) {
    const int tid = threadIdx.x;
    if (tid >= block_size) return;
    const uint32_t mask = (block_size == 32) ? 0xffffffffu : ((1u << block_size) - 1u);
    const float inv_sqrt_b = __fdiv_rn(1.0f, __fsqrt_rn((float)block_size));
    const int blocks_per_row = width / block_size;

    for (int64_t blk = blockIdx.x; blk < total_blocks; blk += gridDim.x) {
        const size_t blk_base = (size_t)blk * (size_t)block_size;
        const int row_col = (int)(blk % blocks_per_row) * block_size + tid;
        float val = fwht_load_f32(x, blk_base + tid);
        if (!is_inverse && has_signs) {
            val *= signs[row_col];
        }
        for (int ls = 1; ls < block_size; ls <<= 1) {
            const float partner = __shfl_xor_sync(mask, val, ls);
            val = (tid & ls) ? (partner - val) : (val + partner);
        }
        val *= inv_sqrt_b;
        if (is_inverse && has_signs) {
            val *= signs[row_col];
        }
        if (has_y) {
            qstore(&y[blk_base + tid], val);
        }
        if (has_q8 && block_size == 32) {
            const float out_v[1] = {val};
            fwht_quantize_q8_in_reg<1>(out_v, blk_base + tid, (size_t)blk, xq, xd);
        }
    }
}

template <typename T, int NT, int E>
__device__ __forceinline__ void hadamard_fwht_wide(
    const T* __restrict__ x,
    const float* __restrict__ signs,
    T* __restrict__ y,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const int width,
    const int has_signs,
    const int is_inverse,
    const int has_y,
    const int has_q8,
    const int64_t total_blocks
) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    if (tid >= NT) return;
    constexpr int B = NT * E;
    const float inv_sqrt_b = __fdiv_rn(1.0f, __fsqrt_rn((float)B));
    const int blocks_per_row = width / B;
    const int lane_base = tid * E;

    for (int64_t blk = blockIdx.x; blk < total_blocks; blk += gridDim.x) {
        const size_t blk_base = (size_t)blk * B;
        const int row_col_base = (int)(blk % blocks_per_row) * B + lane_base;

        float v[E];
        #pragma unroll
        for (int r = 0; r < E; ++r) {
            float val = fwht_load_f32(x, blk_base + lane_base + r);
            if (!is_inverse && has_signs) {
                val *= signs[row_col_base + r];
            }
            v[r] = val;
        }

        fwht_block_butterfly<NT, E>(v, smem);
        if constexpr (NT > 32) {
            if (blk + gridDim.x < total_blocks) {
                __syncthreads();
            }
        }

        float out_v[E];
        #pragma unroll
        for (int r = 0; r < E; ++r) {
            float val = v[r] * inv_sqrt_b;
            if (is_inverse && has_signs) {
                val *= signs[row_col_base + r];
            }
            out_v[r] = val;
            if (has_y) {
                qstore(&y[blk_base + lane_base + r], val);
            }
        }
        if (has_q8) {
            fwht_quantize_q8_in_reg<E>(out_v, blk_base + lane_base, (size_t)blk * (B / 32), xq, xd);
        }
    }
}

template <typename T>
__device__ __forceinline__ void hadamard_fwht_dispatch(
    const T* __restrict__ x,
    const float* __restrict__ signs,
    T* __restrict__ y,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const int block_size,
    const int width,
    const int has_signs,
    const int is_inverse,
    const int has_y,
    const int has_q8,
    const int64_t total_blocks,
    const int64_t total_q8_blocks_padded
) {
    int nt = block_size / 2;
    if (nt > 1024) nt = 1024;
    switch (block_size) {
        case 64:
            hadamard_fwht_wide<T, 32, 2>(x, signs, y, xq, xd, width, has_signs, is_inverse, has_y, has_q8, total_blocks);
            break;
        case 128:
            hadamard_fwht_wide<T, 64, 2>(x, signs, y, xq, xd, width, has_signs, is_inverse, has_y, has_q8, total_blocks);
            break;
        case 256:
            hadamard_fwht_wide<T, 128, 2>(x, signs, y, xq, xd, width, has_signs, is_inverse, has_y, has_q8, total_blocks);
            break;
        case 512:
            hadamard_fwht_wide<T, 256, 2>(x, signs, y, xq, xd, width, has_signs, is_inverse, has_y, has_q8, total_blocks);
            break;
        case 1024:
            hadamard_fwht_wide<T, 512, 2>(x, signs, y, xq, xd, width, has_signs, is_inverse, has_y, has_q8, total_blocks);
            break;
        case 2048:
            hadamard_fwht_wide<T, 1024, 2>(x, signs, y, xq, xd, width, has_signs, is_inverse, has_y, has_q8, total_blocks);
            break;
        case 4096:
            hadamard_fwht_wide<T, 1024, 4>(x, signs, y, xq, xd, width, has_signs, is_inverse, has_y, has_q8, total_blocks);
            break;
        case 8192:
            hadamard_fwht_wide<T, 1024, 8>(x, signs, y, xq, xd, width, has_signs, is_inverse, has_y, has_q8, total_blocks);
            break;
        default:
            nt = block_size;
            hadamard_fwht_small<T>(x, signs, y, xq, xd, block_size, width, has_signs, is_inverse, has_y, has_q8, total_blocks);
            break;
    }
    if (has_q8 && block_size >= 32) {
        const int64_t active_q8 = total_blocks * (block_size / 32);
        zero_q8_tail_blocks(active_q8, total_q8_blocks_padded, nt, xq, xd);
    }
}

extern "C" __global__ void hadamard_fwht_f32(
    const float* __restrict__ x,
    const float* __restrict__ signs,
    float* __restrict__ y,
    const int block_size,
    const int width,
    const int has_signs,
    const int is_inverse,
    const int64_t total_blocks
) {
    hadamard_fwht_dispatch<float>(x, signs, y, nullptr, nullptr, block_size, width, has_signs, is_inverse, 1, 0, total_blocks, 0);
}

extern "C" __global__ void hadamard_fwht_f16(
    const __half* __restrict__ x,
    const float* __restrict__ signs,
    __half* __restrict__ y,
    const int block_size,
    const int width,
    const int has_signs,
    const int is_inverse,
    const int64_t total_blocks
) {
    hadamard_fwht_dispatch<__half>(x, signs, y, nullptr, nullptr, block_size, width, has_signs, is_inverse, 1, 0, total_blocks, 0);
}

extern "C" __global__ void hadamard_fwht_bf16(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ signs,
    __nv_bfloat16* __restrict__ y,
    const int block_size,
    const int width,
    const int has_signs,
    const int is_inverse,
    const int64_t total_blocks
) {
    hadamard_fwht_dispatch<__nv_bfloat16>(x, signs, y, nullptr, nullptr, block_size, width, has_signs, is_inverse, 1, 0, total_blocks, 0);
}

extern "C" __global__ void hadamard_fwht_q8_f32(
    const float* __restrict__ x,
    const float* __restrict__ signs,
    float* __restrict__ y,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const int block_size,
    const int width,
    const int has_signs,
    const int is_inverse,
    const int has_y,
    const int64_t total_blocks,
    const int64_t total_q8_blocks_padded
) {
    hadamard_fwht_dispatch<float>(
        x, signs, y, xq, xd, block_size, width, has_signs, is_inverse, has_y, 1, total_blocks, total_q8_blocks_padded);
}

extern "C" __global__ void hadamard_fwht_q8_f16(
    const __half* __restrict__ x,
    const float* __restrict__ signs,
    __half* __restrict__ y,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const int block_size,
    const int width,
    const int has_signs,
    const int is_inverse,
    const int has_y,
    const int64_t total_blocks,
    const int64_t total_q8_blocks_padded
) {
    hadamard_fwht_dispatch<__half>(
        x, signs, y, xq, xd, block_size, width, has_signs, is_inverse, has_y, 1, total_blocks, total_q8_blocks_padded);
}

extern "C" __global__ void hadamard_fwht_q8_bf16(
    const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ signs,
    __nv_bfloat16* __restrict__ y,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const int block_size,
    const int width,
    const int has_signs,
    const int is_inverse,
    const int has_y,
    const int64_t total_blocks,
    const int64_t total_q8_blocks_padded
) {
    hadamard_fwht_dispatch<__nv_bfloat16>(
        x, signs, y, xq, xd, block_size, width, has_signs, is_inverse, has_y, 1, total_blocks, total_q8_blocks_padded);
}

// Fused [optional residual add ->] RMSNorm(x, w, eps) -> optional signs -> Wide-CTA block-FWHT -> f32 dst / Q8.
template <typename T, int NT, int E>
__device__ __forceinline__ void rms_norm_hadamard_fwht_wide(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    T* __restrict__ residual_out,
    const void* __restrict__ weight,
    const float* __restrict__ signs,
    float* __restrict__ dst,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const float eps,
    const int width,
    const int w_dtype,
    const int has_residual,
    const int has_residual_out,
    const int has_signs,
    const int has_dst,
    const int has_q8,
    const int64_t nrows
) {
    extern __shared__ float smem[];
    const int tid = threadIdx.x;
    if (tid >= NT) return;
    constexpr int B = NT * E;
    constexpr int NWARPS = NT / 32;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int lane_base = tid * E;
    const int blocks_per_row = width / B;
    const float inv_sqrt_b = __fdiv_rn(1.0f, __fsqrt_rn((float)B));

    for (int64_t row = blockIdx.x; row < nrows; row += gridDim.x) {
        const size_t row_off = (size_t)row * (size_t)width;
        float sum_sq = 0.0f;
        float h_single[E];

        if (blocks_per_row == 1) {
            #pragma unroll
            for (int r = 0; r < E; ++r) {
                const int col = lane_base + r;
                float h = fwht_load_f32(x, row_off + col);
                if (has_residual) {
                    h = round_via_dtype<T>(h + fwht_load_f32(residual, row_off + col));
                    if (has_residual_out) {
                        qstore(&residual_out[row_off + col], h);
                    }
                }
                h_single[r] = h;
                sum_sq += h * h;
            }
        } else {
            for (int col = tid; col < width; col += NT) {
                float h = fwht_load_f32(x, row_off + col);
                if (has_residual) {
                    h = round_via_dtype<T>(h + fwht_load_f32(residual, row_off + col));
                    if (has_residual_out) {
                        qstore(&residual_out[row_off + col], h);
                    }
                }
                sum_sq += h * h;
            }
        }

        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            sum_sq += __shfl_xor_sync(0xffffffffu, sum_sq, off);
        }
        float inv_rms;
        if constexpr (NWARPS == 1) {
            inv_rms = __fdiv_rn(1.0f, __fsqrt_rn(__fdiv_rn(sum_sq, (float)width) + eps));
        } else {
            if (lane == 0) {
                smem[warp_id] = sum_sq;
            }
            __syncthreads();
            if (warp_id == 0) {
                float wsum = (lane < NWARPS) ? smem[lane] : 0.0f;
                #pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    wsum += __shfl_xor_sync(0xffffffffu, wsum, off);
                }
                if (lane == 0) {
                    smem[0] = __fdiv_rn(1.0f, __fsqrt_rn(__fdiv_rn(wsum, (float)width) + eps));
                }
            }
            __syncthreads();
            inv_rms = smem[0];
            __syncthreads();
        }

        for (int seg = 0; seg < blocks_per_row; ++seg) {
            const int seg_col = seg * B + lane_base;
            float v[E];
            #pragma unroll
            for (int r = 0; r < E; ++r) {
                const int col = seg_col + r;
                float h;
                if (blocks_per_row == 1) {
                    h = h_single[r];
                } else if (has_residual && has_residual_out) {
                    h = fwht_load_f32(residual_out, row_off + col);
                } else {
                    h = fwht_load_f32(x, row_off + col);
                    if (has_residual) {
                        h = round_via_dtype<T>(h + fwht_load_f32(residual, row_off + col));
                    }
                }
                float val = h * inv_rms * load_weight_f32(weight, col, w_dtype);
                if (has_signs) {
                    val *= signs[col];
                }
                v[r] = val;
            }

            fwht_block_butterfly<NT, E>(v, smem);
            if constexpr (NT > 32) {
                if (seg + 1 < blocks_per_row || row + gridDim.x < nrows) {
                    __syncthreads();
                }
            }

            float out_v[E];
            #pragma unroll
            for (int r = 0; r < E; ++r) {
                const float val = v[r] * inv_sqrt_b;
                out_v[r] = val;
                if (has_dst) {
                    dst[row_off + seg_col + r] = val;
                }
            }
            if (has_q8) {
                fwht_quantize_q8_in_reg<E>(
                    out_v,
                    row_off + seg_col,
                    (row_off + (size_t)seg * B) >> 5,
                    xq,
                    xd);
            }
        }
    }
}

template <typename T>
__device__ __forceinline__ void rms_norm_hadamard_fwht_dispatch(
    const T* __restrict__ x,
    const T* __restrict__ residual,
    T* __restrict__ residual_out,
    const void* __restrict__ weight,
    const float* __restrict__ signs,
    float* __restrict__ dst,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const float eps,
    const int block_size,
    const int width,
    const int w_dtype,
    const int has_residual,
    const int has_residual_out,
    const int has_signs,
    const int has_dst,
    const int has_q8,
    const int64_t nrows,
    const int64_t total_q8_blocks_padded
) {
    int nt = block_size / 2;
    if (nt < 32) nt = 32;
    if (nt > 1024) nt = 1024;
    switch (block_size) {
        case 32:
            rms_norm_hadamard_fwht_wide<T, 32, 1>(
                x, residual, residual_out, weight, signs, dst, xq, xd, eps, width, w_dtype,
                has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows);
            break;
        case 64:
            rms_norm_hadamard_fwht_wide<T, 32, 2>(
                x, residual, residual_out, weight, signs, dst, xq, xd, eps, width, w_dtype,
                has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows);
            break;
        case 128:
            rms_norm_hadamard_fwht_wide<T, 64, 2>(
                x, residual, residual_out, weight, signs, dst, xq, xd, eps, width, w_dtype,
                has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows);
            break;
        case 256:
            rms_norm_hadamard_fwht_wide<T, 128, 2>(
                x, residual, residual_out, weight, signs, dst, xq, xd, eps, width, w_dtype,
                has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows);
            break;
        case 512:
            rms_norm_hadamard_fwht_wide<T, 256, 2>(
                x, residual, residual_out, weight, signs, dst, xq, xd, eps, width, w_dtype,
                has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows);
            break;
        case 1024:
            rms_norm_hadamard_fwht_wide<T, 512, 2>(
                x, residual, residual_out, weight, signs, dst, xq, xd, eps, width, w_dtype,
                has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows);
            break;
        case 2048:
            rms_norm_hadamard_fwht_wide<T, 1024, 2>(
                x, residual, residual_out, weight, signs, dst, xq, xd, eps, width, w_dtype,
                has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows);
            break;
        case 4096:
            rms_norm_hadamard_fwht_wide<T, 1024, 4>(
                x, residual, residual_out, weight, signs, dst, xq, xd, eps, width, w_dtype,
                has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows);
            break;
        case 8192:
            rms_norm_hadamard_fwht_wide<T, 1024, 8>(
                x, residual, residual_out, weight, signs, dst, xq, xd, eps, width, w_dtype,
                has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows);
            break;
        default:
            break;
    }
    if (has_q8 && block_size >= 32) {
        const int64_t active_q8 = nrows * (width / 32);
        zero_q8_tail_blocks(active_q8, total_q8_blocks_padded, nt, xq, xd);
    }
}

extern "C" __global__ void rms_norm_hadamard_fwht_f32(
    const float* __restrict__ x,
    const float* __restrict__ residual,
    float* __restrict__ residual_out,
    const void* __restrict__ weight,
    const float* __restrict__ signs,
    float* __restrict__ dst,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const float eps,
    const int block_size,
    const int width,
    const int w_dtype,
    const int has_residual,
    const int has_residual_out,
    const int has_signs,
    const int has_dst,
    const int has_q8,
    const int64_t nrows,
    const int64_t total_q8_blocks_padded
) {
    rms_norm_hadamard_fwht_dispatch<float>(
        x, residual, residual_out, weight, signs, dst, xq, xd, eps, block_size, width, w_dtype,
        has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows, total_q8_blocks_padded);
}

extern "C" __global__ void rms_norm_hadamard_fwht_bf16(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ residual,
    __nv_bfloat16* __restrict__ residual_out,
    const void* __restrict__ weight,
    const float* __restrict__ signs,
    float* __restrict__ dst,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const float eps,
    const int block_size,
    const int width,
    const int w_dtype,
    const int has_residual,
    const int has_residual_out,
    const int has_signs,
    const int has_dst,
    const int has_q8,
    const int64_t nrows,
    const int64_t total_q8_blocks_padded
) {
    rms_norm_hadamard_fwht_dispatch<__nv_bfloat16>(
        x, residual, residual_out, weight, signs, dst, xq, xd, eps, block_size, width, w_dtype,
        has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows, total_q8_blocks_padded);
}

extern "C" __global__ void rms_norm_hadamard_fwht_f16(
    const __half* __restrict__ x,
    const __half* __restrict__ residual,
    __half* __restrict__ residual_out,
    const void* __restrict__ weight,
    const float* __restrict__ signs,
    float* __restrict__ dst,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd,
    const float eps,
    const int block_size,
    const int width,
    const int w_dtype,
    const int has_residual,
    const int has_residual_out,
    const int has_signs,
    const int has_dst,
    const int has_q8,
    const int64_t nrows,
    const int64_t total_q8_blocks_padded
) {
    rms_norm_hadamard_fwht_dispatch<__half>(
        x, residual, residual_out, weight, signs, dst, xq, xd, eps, block_size, width, w_dtype,
        has_residual, has_residual_out, has_signs, has_dst, has_q8, nrows, total_q8_blocks_padded);
}




