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

// IQ1_S/IQ1_M signed-grid delta bias (iq_quants.rs:28); both 1-bit types share it.
constexpr float IQ1S_DELTA = 0.125f;

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

// ---- iq_bytemask4 + dp4a_signed + the 7 qdp4a<IQ*> decoders + traits ----
__device__ __forceinline__ unsigned iq_bytemask4(unsigned sgn) {
    const unsigned s = (sgn & 1u) | ((sgn & 2u) << 7) | ((sgn & 4u) << 14) | ((sgn & 8u) << 21);
    return s * 0xFFu;
}

// Signed-grid 4-wide int8 dot: sum_i (-1)^sgn_i * g_i * u_i for the 4 magnitude bytes packed in `grid`
// (each in [0,127]) against the q8_1 int `u`, signs from the low 4 bits of `sgn`. dp4a-only (above).
__device__ __forceinline__ int dp4a_signed(int grid, int u, unsigned sgn) {
    const int gm = (int)((unsigned)grid & iq_bytemask4(sgn));
    return hip_dp4a(grid, u, 0) - 2 * hip_dp4a(gm, u, 0);
}

// IQ2_XXS (SYM 2.06bpw). Mirrors qdec<DW_IQ2_XXS>: per sub-block e, db = d*(0.5+(aux32_1>>28))*0.25
// is constant across all 32 coords; grid byte j of IQ2XXS_GRID_D[gi] is the magnitude, ksigns the sign.
template <> struct qdp4a<DW_IQ2_XXS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
        const uint32_t aux32_0 = (uint32_t)qs[4 * e] | ((uint32_t)qs[4 * e + 1] << 16);
        const uint32_t aux32_1 = (uint32_t)qs[4 * e + 2] | ((uint32_t)qs[4 * e + 3] << 16);
        const float db = d * (0.5f + (float)(aux32_1 >> 28)) * 0.25f;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int gi = (int)((aux32_0 >> (8 * l)) & 0xFF);
            const uint64_t entry = IQ2XXS_GRID_D[gi];
            const uint32_t signs = KSIGNS_IQ2XS_D[(aux32_1 >> (7 * l)) & 127];
            idot += dp4a_signed((int)(uint32_t)entry, u[2 * l], signs & 0xF);
            idot += dp4a_signed((int)(uint32_t)(entry >> 32), u[2 * l + 1], (signs >> 4) & 0xF);
        }
        return db * d8 * (float)idot;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ2_XS (SYM 2.31bpw). Mirrors qdec<DW_IQ2_XS>: scale nibble per sub-block e is sc&0xF for groups
// 0,1 (db_lo) and sc>>4 for groups 2,3 (db_hi) -- two partial dots scaled separately (llama ls0/ls1).
template <> struct qdp4a<DW_IQ2_XS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint16_t* qs = reinterpret_cast<const uint16_t*>(blk + 2);
        const uint8_t sc = blk[66 + e];
        const float db_lo = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db_hi = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot_lo = 0, idot_hi = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint16_t qv = qs[4 * e + l];
            const uint64_t entry = IQ2XS_GRID_D[qv & 511];
            const uint32_t signs = KSIGNS_IQ2XS_D[qv >> 9];
            const int part = dp4a_signed((int)(uint32_t)entry, u[2 * l], signs & 0xF)
                           + dp4a_signed((int)(uint32_t)(entry >> 32), u[2 * l + 1], (signs >> 4) & 0xF);
            if (l < 2) idot_lo += part; else idot_hi += part;
        }
        return d8 * (db_lo * (float)idot_lo + db_hi * (float)idot_hi);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ2_S (SYM 2.56bpw). Mirrors qdec<DW_IQ2_S>: idx = qs[4e+l] | ((qh[e]<<(8-2l))&0x300), signs are a
// raw byte qs[32+4e+l] (NOT ksigns), scale nibble per sub-block like IQ2_XS (db_lo / db_hi).
template <> struct qdp4a<DW_IQ2_S> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t sc = blk[74 + e];
        const float db_lo = d * (0.5f + (float)(sc & 0xF)) * 0.25f;
        const float db_hi = d * (0.5f + (float)(sc >> 4)) * 0.25f;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot_lo = 0, idot_hi = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = (int)qs[4 * e + l] | (((int)qh[e] << (8 - 2 * l)) & 0x300);
            const uint64_t entry = IQ2S_GRID_D[idx];
            const uint32_t signs = qs[32 + 4 * e + l];
            const int part = dp4a_signed((int)(uint32_t)entry, u[2 * l], signs & 0xF)
                           + dp4a_signed((int)(uint32_t)(entry >> 32), u[2 * l + 1], (signs >> 4) & 0xF);
            if (l < 2) idot_lo += part; else idot_hi += part;
        }
        return d8 * (db_lo * (float)idot_lo + db_hi * (float)idot_hi);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ3_XXS (SYM 3.06bpw). Mirrors qdec<DW_IQ3_XXS>: db = d*(0.5+(aux32>>28))*0.5 const per sub-block;
// each group l has TWO u32 grid entries (coords 0..3 from qs[8e+2l], 4..7 from qs[8e+2l+1]), ksigns.
template <> struct qdp4a<DW_IQ3_XXS> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* ss = blk + 2 + 64;
        const uint32_t aux32 = (uint32_t)ss[4 * e] | ((uint32_t)ss[4 * e + 1] << 8)
                             | ((uint32_t)ss[4 * e + 2] << 16) | ((uint32_t)ss[4 * e + 3] << 24);
        const float db = d * (0.5f + (float)(aux32 >> 28)) * 0.5f;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t signs = KSIGNS_IQ2XS_D[(aux32 >> (7 * l)) & 127];
            idot += dp4a_signed((int)IQ3XXS_GRID_D[qs[8 * e + 2 * l]], u[2 * l], signs & 0xF);
            idot += dp4a_signed((int)IQ3XXS_GRID_D[qs[8 * e + 2 * l + 1]], u[2 * l + 1], (signs >> 4) & 0xF);
        }
        return db * d8 * (float)idot;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ3_S (SYM 3.31bpw). Mirrors qdec<DW_IQ3_S>: db = d*(1+2*nib) const per sub-block (nib from
// scales[e>>1]); idx high bit from qh with shift 8-2l (coords 0..3) / 7-2l (coords 4..7); raw sign byte.
template <> struct qdp4a<DW_IQ3_S> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint8_t* qh = blk + 66;
        const uint8_t* signs_b = blk + 74;
        const uint8_t* scales = blk + 106;
        const uint8_t sc = scales[e >> 1];
        const int nib = (e & 1) ? (sc >> 4) : (sc & 0xF);
        const float db = d * (1.0f + 2.0f * (float)nib);
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx0 = (int)qs[8 * e + 2 * l] | (((int)qh[e] << (8 - 2 * l)) & 256);
            const int idx1 = (int)qs[8 * e + 2 * l + 1] | (((int)qh[e] << (7 - 2 * l)) & 256);
            const uint32_t signs = signs_b[4 * e + l];
            idot += dp4a_signed((int)IQ3S_GRID_D[idx0], u[2 * l], signs & 0xF);
            idot += dp4a_signed((int)IQ3S_GRID_D[idx1], u[2 * l + 1], (signs >> 4) & 0xF);
        }
        return db * d8 * (float)idot;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ1_S (1.56bpw). Mirrors qdec<DW_IQ1_S>: grid coords are SIGNED int8 already (no sign field) -> direct
// dp4a; dl = d*(2*((qh>>12)&7)+1), delta = +/-IQ1S_DELTA per sub-block. val = dl*(g+delta), so the
// fractional bias rides a sum_u term: result = dl*d8*(sum g*u + delta*sum u).
template <> struct qdp4a<DW_IQ1_S> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
        const float d = __half2float(*reinterpret_cast<const __half*>(blk));
        const uint8_t* qs = blk + 2;
        const uint16_t* qh = reinterpret_cast<const uint16_t*>(blk + 34);
        const uint16_t qhv = qh[e];
        const float dl = d * (2.0f * (float)((qhv >> 12) & 7) + 1.0f);
        const float delta = (qhv & 0x8000) ? -IQ1S_DELTA : IQ1S_DELTA;
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
        int idot = 0, isum = 0;
        #pragma unroll
        for (int l = 0; l < 4; ++l) {
            const int idx = (int)qs[e * 4 + l] | ((((int)(qhv >> (3 * l))) & 7) << 8);
            const uint64_t entry = IQ1S_GRID_D[idx];
            idot = hip_dp4a((int)(uint32_t)entry, u[2 * l], idot);
            idot = hip_dp4a((int)(uint32_t)(entry >> 32), u[2 * l + 1], idot);
            isum = hip_dp4a(0x01010101, u[2 * l], isum);
            isum = hip_dp4a(0x01010101, u[2 * l + 1], isum);
        }
        return dl * d8 * ((float)idot + delta * (float)isum);
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ1_M (1.75bpw). Mirrors qdec<DW_IQ1_M>: signed grid (direct dp4a); super-scale d from 4 scale-nibble
// tops; dl1 (groups 0,1) / dl2 (groups 2,3) from 3-bit fields; idx high bits + delta sign per group from
// qh. delta varies PER group -> accumulate dl*(sum g*u + delta*sum u) in f32 per group.
template <> struct qdp4a<DW_IQ1_M> {
    static constexpr int NSUB = 8;
    static __device__ __forceinline__ float partial(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8, int e) {
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
        const float d8 = __half2float(xd8[e]);
        const int* u = reinterpret_cast<const int*>(xq8) + e * 8;
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
            int isum = hip_dp4a(0x01010101, u[2 * l], 0);
            isum = hip_dp4a(0x01010101, u[2 * l + 1], isum);
            fs += dl * ((float)idot + delta * (float)isum);
        }
        return d8 * fs;
    }
    static __device__ __forceinline__ float block(
            const uint8_t* __restrict__ blk, const int8_t* __restrict__ xq8, const __half* __restrict__ xd8) {
        float s = 0.0f;
        #pragma unroll
        for (int e = 0; e < NSUB; ++e) s += partial(blk, xq8, xd8, e);
        return s;
    }
};

// IQ dp4a byte strides (= on-disk super-block bytes; mirror qdw_traits::BYTES). All 256-elem.
template <> struct qdp4a_traits<DW_IQ2_XXS> { static constexpr int BYTES = 66;  };
template <> struct qdp4a_traits<DW_IQ2_XS>  { static constexpr int BYTES = 74;  };
template <> struct qdp4a_traits<DW_IQ2_S>   { static constexpr int BYTES = 82;  };
template <> struct qdp4a_traits<DW_IQ3_XXS> { static constexpr int BYTES = 98;  };
template <> struct qdp4a_traits<DW_IQ3_S>   { static constexpr int BYTES = 110; };
template <> struct qdp4a_traits<DW_IQ1_S>   { static constexpr int BYTES = 50;  };
template <> struct qdp4a_traits<DW_IQ1_M>   { static constexpr int BYTES = 56;  };

// ---- activation -> q8 (separated int8 qs `xq` + per-32 f16 scale `xd`) ----
extern "C" __global__ void iq_quantize_q8_f16(
    const int M,
    const int K,
    const __half* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    const int nblk = K >> 5;
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wid >= M * nblk) {
        return;
    }
    const int m = wid / nblk;
    const int blk = wid % nblk;
    const int lane = threadIdx.x & 31;
    const size_t idx = (size_t)m * K + (size_t)blk * 32 + lane;
    const float v = __half2float(x[idx]);
    float a = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off)); // warp max-reduce -> every lane has the block absmax
    }
    // IEEE round-to-nearest division (NOT the --use_fast_math approximate reciprocal): deterministic,
    // spec-compliant q8_1 activation quant that matches llama.cpp / the CPU reference bit-for-bit, so
    // decode is reproducible. Negligible cost -- one divide per 32-block, amortized over the matvec.
    const float inv = (a > 0.0f) ? __fdiv_rn(127.0f, a) : 0.0f;
    // roundf = round-half-AWAY-from-zero, matching llama's quantize_mmq_q8_1 (quantize.cu) and the
    // CPU reference in qmmq_numeric.rs (Rust f32::round). rintf (round-half-to-even) differed from
    // both on exact .5 ties; roundf removes that 1-ULP divergence vs llama at no perf cost.
    int q = (int)roundf(v * inv);
    q = max(-127, min(127, q));
    xq[idx] = (int8_t)q;
    if (lane == 0) {
        xd[(size_t)m * nblk + blk] = __float2half(a / 127.0f);
    }
}

extern "C" __global__ void iq_quantize_q8_bf16(
    const int M,
    const int K,
    const __nv_bfloat16* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    const int nblk = K >> 5;
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wid >= M * nblk) {
        return;
    }
    const int m = wid / nblk;
    const int blk = wid % nblk;
    const int lane = threadIdx.x & 31;
    const size_t idx = (size_t)m * K + (size_t)blk * 32 + lane;
    const float v = __bfloat162float(x[idx]);
    float a = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off)); // warp max-reduce -> every lane has the block absmax
    }
    // IEEE round-to-nearest division (NOT the --use_fast_math approximate reciprocal): deterministic,
    // spec-compliant q8_1 activation quant that matches llama.cpp / the CPU reference bit-for-bit, so
    // decode is reproducible. Negligible cost -- one divide per 32-block, amortized over the matvec.
    const float inv = (a > 0.0f) ? __fdiv_rn(127.0f, a) : 0.0f;
    // roundf = round-half-AWAY-from-zero, matching llama's quantize_mmq_q8_1 (quantize.cu) and the
    // CPU reference in qmmq_numeric.rs (Rust f32::round). rintf (round-half-to-even) differed from
    // both on exact .5 ties; roundf removes that 1-ULP divergence vs llama at no perf cost.
    int q = (int)roundf(v * inv);
    q = max(-127, min(127, q));
    xq[idx] = (int8_t)q;
    if (lane == 0) {
        xd[(size_t)m * nblk + blk] = __float2half(a / 127.0f);
    }
}

extern "C" __global__ void iq_quantize_q8_f32(
    const int M,
    const int K,
    const float* __restrict__ x,
    int8_t* __restrict__ xq,
    __half* __restrict__ xd
) {
    const int nblk = K >> 5;
    const int wid = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (wid >= M * nblk) {
        return;
    }
    const int m = wid / nblk;
    const int blk = wid % nblk;
    const int lane = threadIdx.x & 31;
    const size_t idx = (size_t)m * K + (size_t)blk * 32 + lane;
    const float v = x[idx];
    float a = fabsf(v);
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, off));
    }
    // IEEE round-to-nearest division (NOT the --use_fast_math approximate reciprocal): deterministic,
    // spec-compliant q8_1 activation quant that matches llama.cpp / the CPU reference bit-for-bit, so
    // decode is reproducible. Negligible cost -- one divide per 32-block, amortized over the matvec.
    const float inv = (a > 0.0f) ? __fdiv_rn(127.0f, a) : 0.0f;
    int q = (int)roundf(v * inv);
    q = max(-127, min(127, q));
    xq[idx] = (int8_t)q;
    if (lane == 0) {
        xd[(size_t)m * nblk + blk] = __float2half(a / 127.0f);
    }
}

// ---- unified dp4a decode matvec cores (dense + indexed-MoE) ----
template <int WTYPE, typename XT>
__device__ __forceinline__ void qmatvec_dp4a_core(
    const int nrows,
    const int ncols,                  // multiple of 256
    const uint8_t* __restrict__ wq,
    const int8_t*  __restrict__ xq,   // [ncols] q8_1 int8 quants
    const __half*  __restrict__ xd,   // [ncols/32] q8_1 f16 scales
    XT* __restrict__ y
) {
    constexpr int WBYTES = qdp4a_traits<WTYPE>::BYTES;
    const int lane = threadIdx.x & 31;
    const int rows_per_block = blockDim.x >> 5;
    const int row = blockIdx.x * rows_per_block + (threadIdx.x >> 5);
    if (row >= nrows) {
        return;
    }
    const int nblocks = ncols >> 8; // ncols / 256
    const uint8_t* row_ptr = wq + (size_t)row * (size_t)nblocks * WBYTES;

    // Stride lanes over (super-block x sub-unit) work items, NOT whole super-blocks -- the SAME fix the
    // MoE twin already carries. Narrow-k decode shapes (attn q/k/v/o k=2048 -> 8 super-blocks) left
    // 24/32 lanes idle under whole-super-block striding; NSUB sub-units (Q4_K=4 chunks, Q6_K=16 groups)
    // lift the work-item count to nblocks*NSUB so every lane carries a coalesced dp4a. block() = sum
    // over sub, so this is byte-identical math (f32 reorder only) -- the one lane-coalesced core.
    constexpr int NSUB = qdp4a<WTYPE>::NSUB;
    const int nunits = nblocks * NSUB;
    float acc = 0.0f;
    for (int u = lane; u < nunits; u += 32) {
        const int b = u / NSUB;
        const int sub = u - b * NSUB;
        acc += qdp4a<WTYPE>::partial(
            row_ptr + (size_t)b * WBYTES, xq + (size_t)b * 256, xd + (size_t)b * 8, sub);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        qstore(&y[row], acc);
    }
}

template <int WTYPE, typename XT>
__device__ __forceinline__ void moe_qmatvec_dp4a_core(
    const int n,                        // output rows per expert (weight rows)
    const int ncols,                    // k, multiple of 256
    const int nslots,                   // routed slots (= nrows)
    const uint8_t* __restrict__ wbank,  // [E, n, k] blocks
    const int* __restrict__ ids,        // [nslots] expert id per slot
    const int8_t* __restrict__ xq,      // [nslots, ncols] q8_1 int8 quants
    const __half* __restrict__ xd,      // [nslots, ncols/32] q8_1 f16 scales
    XT* __restrict__ y                  // [nslots, n]
) {
    constexpr int WBYTES = qdp4a_traits<WTYPE>::BYTES;
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
    const int nblocks = ncols >> 8; // ncols / 256
    const int expert = ids[s];
    const uint8_t* row_ptr = wbank + ((size_t)expert * n + row) * (size_t)nblocks * WBYTES;
    const int8_t* xq_row = xq + (size_t)s * ncols;
    const __half* xd_row = xd + (size_t)s * (ncols >> 5);

    // Stride lanes over (super-block x sub-unit) work items, NOT whole super-blocks. The expert
    // shapes are narrow-k (gate/up k=2048 -> 8 super-blocks, down k=768 -> 3), so whole-super-block
    // striding leaves 24/32 (gate/up) or 29/32 (down) lanes idle. NSUB sub-units per super-block
    // (Q4_K=4 chunks, Q6_K=16 groups) lifts the parallel unit count to nblocks*NSUB (>=32 here), so
    // every lane carries dp4a work; sub-unit partials sum identically to qdp4a<WTYPE>::block.
    constexpr int NSUB = qdp4a<WTYPE>::NSUB;
    const int nunits = nblocks * NSUB;
    float acc = 0.0f;
    for (int u = lane; u < nunits; u += 32) {
        const int b = u / NSUB;
        const int sub = u - b * NSUB;
        acc += qdp4a<WTYPE>::partial(
            row_ptr + (size_t)b * WBYTES, xq_row + (size_t)b * 256, xd_row + (size_t)b * 8, sub);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        qstore(&y[(size_t)s * n + row], acc);
    }
}

// ---- per-type extern "C" launchers (f16/bf16/f32 dst, dense + MoE) ----
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

