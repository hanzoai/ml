// Matrix-core flash-attention forward for RDNA3/RDNA3.5 (gfx11). Replaces a materialized QK^T +
// F32-softmax-roundtrip SDPA. One block per (batch, q_head, BR-query-tile); 4 waves (128 threads),
// each wave owns WROWS=16 query rows. QK^T and P@V run on the WMMA matrix cores; the online softmax
// runs ENTIRELY IN REGISTERS via cross-lane shuffles (no score materialization, no F32 LDS round-
// trip), causal tiles early-exit, and GQA is handled by reading the KV head as head/gqa (no
// repeat_kv copy).
//
// RDNA3 wave32 WMMA 16x16x16 fragment layout (verified on gfx1151): for C[m,n] = sum_k A[m,k]*B[k,n],
//   A input  a[e], lane l -> A[row = l%16][k = e]
//   B input  b[e], lane l -> B[k = e][n = l%16]
//   C output acc[e], lane l -> C[row = e*2 + l/16][col = l%16]
// Scores S = QK^T land in the C layout: lane l, elem e -> S[query = e*2 + l/16][key = ct*16 + l%16].
// A query row is therefore keyed by (e, l/16); its keys are spread over l%16 (16 lanes) and the
// col-tiles ct. So a per-ROW max/sum reduction = a butterfly shfl_xor over the low-4 lane bits
// (xor 1,2,4,8 stays within the lane's half) + a local reduce over ct. After the butterfly every
// lane in a half-warp holds that row's reduced value, and the O accumulator has the SAME C layout
// (row = e*2 + l/16), so the online rescale (acc[dt][e] *= corr[e]) is a purely local per-element
// multiply -- no cross-lane traffic, no LDS. (Verified by a standalone reduction probe, exact.)
//
// Tile shape: BR=64 query rows, BC=64 key cols/iter, DH=128 head dim. K/V are single-buffered in LDS
// (the wider BC=64 tile + register softmax frees the LDS that a double buffer would need; the halved
// outer-tile count beats the lost prefetch -- measured). The only LDS used for the softmax is sP, the
// f16 probabilities staged once per tile to transpose them into the P@V A-fragment layout.
//
// Resources on gfx1151: 163 VGPR, 0 spill, ~57 KB LDS -> LDS-bound to 1 workgroup (4 waves)/CU.
// Raising occupancy (BR=32 -> 2 blocks/CU) measured SLOWER: the kernel is throughput-bound, not
// latency-bound. The register softmax + BC=64 widening together lifted causal prefill speedup vs
// rocBLAS naive_sdpa from ~1.0/1.33/1.32x (L=512/1024/2048) to ~1.06/1.44/1.36x.
//
// REQUIRES <hip/hip_bf16.h> (WMMA-capable __hip_bfloat16), NOT the legacy <hip/hip_bfloat16.h>.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

#if defined(__HIPCC__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__))

typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef float    v8f  __attribute__((ext_vector_type(8)));

#define FA_BR 64
#define FA_BC 64
#define FA_DH 128
#define FA_NW 4
#define FA_WROWS 16
#define FA_NCT (FA_BC / 16)   // key col-tiles per iteration

__device__ __forceinline__ float fa_ld(const __half* p, size_t i) { return __half2float(p[i]); }
__device__ __forceinline__ float fa_ld(const __hip_bfloat16* p, size_t i) { return (float)p[i]; }
__device__ __forceinline__ void  fa_st(__half* p, size_t i, float v) { p[i] = __float2half(v); }
__device__ __forceinline__ void  fa_st(__hip_bfloat16* p, size_t i, float v) { p[i] = (__hip_bfloat16)v; }

template <typename T>
__device__ void flash_fwd_core(
    int B, int Hq, int Hkv, int Lq, int Lk, float scale, int causal,
    const T* __restrict__ q, const T* __restrict__ k, const T* __restrict__ v, T* __restrict__ o)
{
    const int gqa = Hq / Hkv;
    const int q_tile = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    const int kv_head = head / gqa;
    const int t = threadIdx.x;
    const int wave = t >> 5;
    const int lane = t & 31;
    const int row0 = q_tile * FA_BR;
    const int wrow0 = row0 + wave * FA_WROWS;
    const int half = lane >> 4;   // 0/1: selects which row of each (e) pair this lane owns

    __shared__ _Float16 sQ[FA_BR * FA_DH];
    __shared__ _Float16 sK[FA_BC * FA_DH];
    __shared__ _Float16 sV[FA_BC * FA_DH];
    __shared__ _Float16 sP[FA_NW][FA_WROWS][FA_BC];   // probs, staged once/tile for the P@V transpose

    for (int i = t; i < FA_BR * FA_DH; i += 128) {
        const int r = i / FA_DH, c = i % FA_DH;
        const int gr = row0 + r;
        sQ[i] = (gr < Lq) ? (_Float16)fa_ld(q, (((size_t)batch * Hq + head) * Lq + gr) * FA_DH + c) : (_Float16)0;
    }
    v8f acc[8];
    #pragma unroll
    for (int f = 0; f < 8; ++f) acc[f] = (v8f){0,0,0,0,0,0,0,0};
    // running per-element (per-row = e*2+half) softmax max/sum, held redundantly across each half-warp.
    float m_run[8], l_run[8];
    #pragma unroll
    for (int e = 0; e < 8; ++e) { m_run[e] = -1e30f; l_run[e] = 0.0f; }
    __syncthreads();

    auto stage_kv = [&](int k0) {
        for (int i = t; i < FA_BC * FA_DH; i += 128) {
            const int r = i / FA_DH, c = i % FA_DH;
            const int gk = k0 + r;
            const size_t base = (((size_t)batch * Hkv + kv_head) * Lk + gk) * FA_DH + c;
            sK[i] = (gk < Lk) ? (_Float16)fa_ld(k, base) : (_Float16)0;
            sV[i] = (gk < Lk) ? (_Float16)fa_ld(v, base) : (_Float16)0;
        }
    };

    const int kc_end = causal ? min(Lk, row0 + FA_BR) : Lk;
    const int n_kt = (kc_end + FA_BC - 1) / FA_BC;

    for (int kt = 0; kt < n_kt; ++kt) {
        const int k0 = kt * FA_BC;
        stage_kv(k0);
        __syncthreads();

        // S = Q @ K^T into C-fragments sc[ct]; scale + mask in-register.
        v8f sc[FA_NCT];
        #pragma unroll
        for (int ct = 0; ct < FA_NCT; ++ct) sc[ct] = (v8f){0,0,0,0,0,0,0,0};
        #pragma unroll
        for (int kk = 0; kk < FA_DH; kk += 16) {
            v16h a;
            #pragma unroll
            for (int e = 0; e < 16; ++e) a[e] = sQ[(wave * FA_WROWS + (lane & 15)) * FA_DH + kk + e];
            #pragma unroll
            for (int ct = 0; ct < FA_NCT; ++ct) {
                v16h b;
                #pragma unroll
                for (int e = 0; e < 16; ++e) b[e] = sK[(ct * 16 + (lane & 15)) * FA_DH + kk + e];
                sc[ct] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, sc[ct]);
            }
        }
        #pragma unroll
        for (int ct = 0; ct < FA_NCT; ++ct)
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = e * 2 + half;
                const int col = ct * 16 + (lane & 15);
                const int gq = wrow0 + row, gk = k0 + col;
                float val = sc[ct][e] * scale;
                if (gk >= Lk || (causal && gk > gq)) val = -1e30f;
                sc[ct][e] = val;
            }

        // per-row max over this tile (reduce over ct + butterfly over low-4 lane bits)
        float m_tile[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            float m = -1e30f;
            #pragma unroll
            for (int ct = 0; ct < FA_NCT; ++ct) if (sc[ct][e] > m) m = sc[ct][e];
            #pragma unroll
            for (int off = 1; off < 16; off <<= 1) { float ot = __shfl_xor(m, off); if (ot > m) m = ot; }
            m_tile[e] = m;
        }
        // online combine + correction factor
        float corr[8];
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            float mn = m_run[e] > m_tile[e] ? m_run[e] : m_tile[e];
            corr[e] = __expf(m_run[e] - mn);
            m_run[e] = mn;
        }
        // probabilities p = exp(s - m_run); stage to sP in A-layout for P@V; accumulate row sum.
        #pragma unroll
        for (int ct = 0; ct < FA_NCT; ++ct)
            #pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = e * 2 + half;
                const int col = ct * 16 + (lane & 15);
                float p = __expf(sc[ct][e] - m_run[e]);
                sc[ct][e] = p;
                sP[wave][row][col] = (_Float16)p;
            }
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            float s = 0;
            #pragma unroll
            for (int ct = 0; ct < FA_NCT; ++ct) s += sc[ct][e];
            #pragma unroll
            for (int off = 1; off < 16; off <<= 1) s += __shfl_xor(s, off);
            l_run[e] = l_run[e] * corr[e] + s;
        }
        __syncwarp();

        // O = O*corr + P@V (acc rescale is local: acc[dt][e] owns row e*2+half).
        #pragma unroll
        for (int dt = 0; dt < 8; ++dt) {
            v8f o0;
            #pragma unroll
            for (int e = 0; e < 8; ++e) o0[e] = acc[dt][e] * corr[e];
            #pragma unroll
            for (int kk = 0; kk < FA_NCT; ++kk) {
                v16h pa, vb;
                #pragma unroll
                for (int e = 0; e < 16; ++e) {
                    pa[e] = sP[wave][lane & 15][kk * 16 + e];
                    vb[e] = sV[(kk * 16 + e) * FA_DH + dt * 16 + (lane & 15)];
                }
                o0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(pa, vb, o0);
            }
            acc[dt] = o0;
        }
        __syncthreads();
    }

    #pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
        #pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = e * 2 + half;
            const int col = dt * 16 + (lane & 15);
            const int gq = wrow0 + row;
            if (gq < Lq) {
                float l = l_run[e];
                float val = (l > 0.0f) ? acc[dt][e] / l : 0.0f;
                fa_st(o, (((size_t)batch * Hq + head) * Lq + gq) * FA_DH + col, val);
            }
        }
    }
}

extern "C" __global__ void flash_attn_f16(
    int B, int Hq, int Hkv, int Lq, int Lk, float scale, int causal,
    const __half* q, const __half* k, const __half* v, __half* o) {
    flash_fwd_core<__half>(B, Hq, Hkv, Lq, Lk, scale, causal, q, k, v, o);
}

extern "C" __global__ void flash_attn_bf16(
    int B, int Hq, int Hkv, int Lq, int Lk, float scale, int causal,
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v, __hip_bfloat16* o) {
    flash_fwd_core<__hip_bfloat16>(B, Hq, Hkv, Lq, Lk, scale, causal, q, k, v, o);
}

#endif
