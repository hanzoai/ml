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

// ---------------------------------------------------------------------------------------------
// Flash-attention DECODE: single query per head (Lq==1) attending the whole KV cache. Fuses the
// eager path's repeat_kv (cat -> copy2d) + QKᵀ bmm + f32-softmax roundtrip + P@V bmm + the per-layer
// `.contiguous()` copy of the whole active cache into ONE dispatch. One block per (batch, q_head),
// 128 threads (= head dim). GQA is implicit (kv_head = q_head / (Hq/Hkv)); the KV cache is read IN
// PLACE via (batch,head,seq) strides with the innermost head-dim contiguous, so no cache copy. All
// accumulation is f32: scores, the online (running max/sum) softmax, and the P@V accumulator, so the
// output matches the eager naive_sdpa within f16 rounding. A decode query sees only past keys, so the
// full cache is unmasked. Keys are streamed in FD_TILE tiles staged coalesced to LDS.
#define FD_DH 128
#define FD_TILE 128
template <typename T>
__device__ void flash_decode_core(
    int B, int Hq, int Hkv, int Lk, float scale,
    const T* __restrict__ q,
    const T* __restrict__ k, long long kb_s, long long kh_s, long long kl_s,
    const T* __restrict__ v, long long vb_s, long long vh_s, long long vl_s,
    T* __restrict__ o)
{
    const int gqa = Hq / Hkv;
    const int qh = blockIdx.x;          // query head
    const int b  = blockIdx.y;          // batch
    const int kvh = qh / gqa;           // GQA-mapped kv head (no repeat_kv)
    const int t = threadIdx.x;          // 0..127, doubles as key index (scores) then dim index (output)

    __shared__ _Float16 sQ[FD_DH];
    __shared__ _Float16 sK[FD_TILE * FD_DH];
    __shared__ float sP[FD_TILE];       // tile probabilities
    __shared__ float sred[FD_TILE];     // block-reduction scratch

    sQ[t] = (_Float16) fa_ld(q, ((size_t)(b * Hq + qh)) * FD_DH + t);

    const size_t kbase = (size_t)b * (size_t)kb_s + (size_t)kvh * (size_t)kh_s;
    const size_t vbase = (size_t)b * (size_t)vb_s + (size_t)kvh * (size_t)vh_s;

    float m_run = -1e30f, l_run = 0.0f, o_acc = 0.0f;   // running max, running sum, this thread's O[dim=t]
    __syncthreads();

    for (int k0 = 0; k0 < Lk; k0 += FD_TILE) {
        const int tile = min(FD_TILE, Lk - k0);
        // Coalesced tile load: sK[r][c] = K[batch, kv_head, k0+r, c]  (c = head dim, contiguous).
        for (int i = t; i < FD_TILE * FD_DH; i += FD_TILE) {
            const int r = i / FD_DH, c = i % FD_DH;
            sK[i] = (r < tile) ? (_Float16) fa_ld(k, kbase + (size_t)(k0 + r) * (size_t)kl_s + c) : (_Float16)0;
        }
        __syncthreads();

        // score[key=t] = scale * (Q . K[t])
        float score = -1e30f;
        if (t < tile) {
            float s = 0.0f;
            #pragma unroll
            for (int d = 0; d < FD_DH; ++d) s += (float)sQ[d] * (float)sK[t * FD_DH + d];
            score = s * scale;
        }
        // block max
        sred[t] = score; __syncthreads();
        for (int s = FD_TILE / 2; s > 0; s >>= 1) { if (t < s) sred[t] = fmaxf(sred[t], sred[t + s]); __syncthreads(); }
        const float tmax = sred[0]; __syncthreads();

        const float m_new = fmaxf(m_run, tmax);
        const float corr = __expf(m_run - m_new);
        const float p = (t < tile) ? __expf(score - m_new) : 0.0f;
        sP[t] = p;
        // block sum of p
        sred[t] = p; __syncthreads();
        for (int s = FD_TILE / 2; s > 0; s >>= 1) { if (t < s) sred[t] += sred[t + s]; __syncthreads(); }
        l_run = l_run * corr + sred[0];
        m_run = m_new;
        __syncthreads();   // sP fully written + sred consumed before the O accumulation / next tile

        // O[dim=t] = O*corr + sum_j p[j] * V[k0+j][t]   (V read coalesced across threads at fixed j)
        float acc = o_acc * corr;
        for (int j = 0; j < tile; ++j)
            acc += sP[j] * (float) fa_ld(v, vbase + (size_t)(k0 + j) * (size_t)vl_s + t);
        o_acc = acc;
        __syncthreads();   // before the next tile overwrites sK/sP
    }

    const float out = (l_run > 0.0f) ? o_acc / l_run : 0.0f;
    fa_st(o, ((size_t)(b * Hq + qh)) * FD_DH + t, out);
}

extern "C" __global__ void flash_decode_f16(
    int B, int Hq, int Hkv, int Lk, float scale,
    const __half* q,
    const __half* k, long long kb_s, long long kh_s, long long kl_s,
    const __half* v, long long vb_s, long long vh_s, long long vl_s,
    __half* o) {
    flash_decode_core<__half>(B, Hq, Hkv, Lk, scale, q, k, kb_s, kh_s, kl_s, v, vb_s, vh_s, vl_s, o);
}

extern "C" __global__ void flash_decode_bf16(
    int B, int Hq, int Hkv, int Lk, float scale,
    const __hip_bfloat16* q,
    const __hip_bfloat16* k, long long kb_s, long long kh_s, long long kl_s,
    const __hip_bfloat16* v, long long vb_s, long long vh_s, long long vl_s,
    __hip_bfloat16* o) {
    flash_decode_core<__hip_bfloat16>(B, Hq, Hkv, Lk, scale, q, k, kb_s, kh_s, kl_s, v, vb_s, vh_s, vl_s, o);
}

extern "C" __global__ void flash_attn_bf16(
    int B, int Hq, int Hkv, int Lq, int Lk, float scale, int causal,
    const __hip_bfloat16* q, const __hip_bfloat16* k, const __hip_bfloat16* v, __hip_bfloat16* o) {
    flash_fwd_core<__hip_bfloat16>(B, Hq, Hkv, Lq, Lk, scale, causal, q, k, v, o);
}

#endif
