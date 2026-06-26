#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
#endif

// Native positions-aware rotary embedding. Matches engine `apply_rotary_cpu_inner`
// (hanzo-quant/src/rotary/mod.rs) and hanzo_nn::rotary_emb::rope / rope_i.
//
// src is [b, h, t, d] contiguous. cos/sin are the full [max_pos, d/2] cache tables.
// positions is u32[batch]: the per-sequence start offset, so the cache row for
// token (batch_idx, seq_idx) is positions[batch_idx] + seq_idx (no host index build).
//
// One thread per (x,y) rotation pair; there are b*h*t*(d/2) pairs total. neox pairs
// are (i_d, i_d + d/2); gpt-j (interleaved) pairs are (2*i_d, 2*i_d + 1). All math in
// float so f16/bf16 stay accurate. Writes both q and k when k_ptr != nullptr (k shares
// the same shape contract, with kh heads instead of h).
template <typename T>
__device__ __forceinline__ void rope_positions(
    const T * q, const T * k, const T * cos, const T * sin,
    const uint32_t * positions, T * q_out, T * k_out,
    const unsigned int b, const unsigned int h, const unsigned int kh,
    const unsigned int t, const unsigned int d, const unsigned int is_neox) {
    const unsigned int half = d / 2;
    const unsigned int pairs_per_head = t * half;
    // Total pairs across the larger of the two head counts; each thread maps to one
    // (head, token, pair) slot and applies it to q and (if present) k.
    const unsigned int max_heads = h > kh ? h : kh;
    const unsigned int total = b * max_heads * pairs_per_head;
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const unsigned int pair = idx % half;
    unsigned int tmp = idx / half;
    const unsigned int i_t = tmp % t;
    tmp /= t;
    const unsigned int i_head = tmp % max_heads;
    const unsigned int i_b = tmp / max_heads;

    const unsigned int cache_row = positions[i_b] + i_t;
    const unsigned int i_cs = cache_row * half + pair;
    const float c = (float)cos[i_cs];
    const float s = (float)sin[i_cs];

    unsigned int x_off, y_off;
    if (is_neox) {
        x_off = pair;
        y_off = pair + half;
    } else {
        x_off = pair * 2;
        y_off = pair * 2 + 1;
    }

    if (i_head < h) {
        const unsigned int base = ((i_b * h + i_head) * t + i_t) * d;
        const float x = (float)q[base + x_off];
        const float y = (float)q[base + y_off];
        q_out[base + x_off] = (T)(x * c - y * s);
        q_out[base + y_off] = (T)(y * c + x * s);
    }
    if (k != nullptr && i_head < kh) {
        const unsigned int base = ((i_b * kh + i_head) * t + i_t) * d;
        const float x = (float)k[base + x_off];
        const float y = (float)k[base + y_off];
        k_out[base + x_off] = (T)(x * c - y * s);
        k_out[base + y_off] = (T)(y * c + x * s);
    }
}

#define ROPE_POS_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TYPENAME *q, const TYPENAME *k, const TYPENAME *cos, const TYPENAME *sin, \
    const uint32_t *positions, TYPENAME *q_out, TYPENAME *k_out, \
    const unsigned int b, const unsigned int h, const unsigned int kh, \
    const unsigned int t, const unsigned int d, const unsigned int is_neox) { \
    rope_positions<TYPENAME>(q, k, cos, sin, positions, q_out, k_out, b, h, kh, t, d, is_neox); \
}

ROPE_POS_OP(float, rope_positions_f32)

#if defined(__HIPCC__)
ROPE_POS_OP(__half, rope_positions_f16)
ROPE_POS_OP(hip_bfloat16, rope_positions_bf16)
#endif

// Fused per-head RMSNorm + positions-RoPE for q and k: ONE block per (which, batch, head, token)
// normalizes that head's `d`-vector (rms over d, scale = rsqrt(mean(x^2)+eps), *weight) IN f32 then
// rotates it, in a single launch -- replaces two standalone rms_norm kernels + the rope kernel (and
// the q/k intermediate write+read). The rms math is faithful to reduce.hip `rmsnorm` (f32 accum,
// mean+eps under rsqrt, *alpha); only the f32 sum ORDER differs (within the decode numeric gate).
// Output dtype TO lets the decode store f16 directly (the cast that wrapped the old rope folds in).
#if defined(__HIPCC__)
template <typename TI, typename TO>
__device__ __forceinline__ void rope_norm_positions(
    const TI* q, const TI* k, const TI* q_weight, const TI* k_weight,
    const TI* cos, const TI* sin, const uint32_t* positions,
    TO* q_out, TO* k_out, const float q_eps, const float k_eps,
    const unsigned int b, const unsigned int h, const unsigned int kh,
    const unsigned int t, const unsigned int d, const unsigned int is_neox) {
    extern __shared__ float rn_smem[];          // [d] normed vector + [ceil(blockDim/32)] warp partials
    float* normed = rn_smem;
    float* s_sum = rn_smem + d;

    const unsigned int qheads = b * h * t;
    unsigned int gid = blockIdx.x;
    const bool isq = gid < qheads;
    const TI* xin; const TI* w; TO* xout; float eps; unsigned int heads;
    if (isq) { xin = q; w = q_weight; xout = q_out; eps = q_eps; heads = h; }
    else { gid -= qheads; xin = k; w = k_weight; xout = k_out; eps = k_eps; heads = kh; }
    const unsigned int i_t = gid % t; gid /= t;
    const unsigned int head = gid % heads; gid /= heads;
    const unsigned int i_b = gid;
    const size_t base = ((size_t)(i_b * heads + head) * t + i_t) * d;

    const int tid = threadIdx.x;
    const int nwarps = (blockDim.x + 31) >> 5;
    const float xi = (float)xin[base + tid];
    float tmp = xi * xi;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) tmp += __shfl_xor(tmp, off);
    if ((tid & 31) == 0) s_sum[tid >> 5] = tmp;
    __syncthreads();
    float ss = 0.0f;
    for (int w_i = 0; w_i < nwarps; ++w_i) ss += s_sum[w_i];   // sum the per-warp partials (f32)
    const float scale = rsqrtf(ss / (float)d + eps);
    normed[tid] = scale * xi * (float)w[tid];
    __syncthreads();

    const unsigned int half = d >> 1;
    const unsigned int cache_row = positions[i_b] + i_t;
    unsigned int p, partner; bool is_x;
    if (is_neox) {
        if ((unsigned)tid < half) { p = tid; partner = tid + half; is_x = true; }
        else { p = tid - half; partner = tid - half; is_x = false; }
    } else {
        p = tid >> 1; is_x = (tid & 1) == 0; partner = is_x ? (tid + 1) : (tid - 1);
    }
    const float c = (float)cos[(size_t)cache_row * half + p];
    const float s = (float)sin[(size_t)cache_row * half + p];
    const float self_v = normed[tid];
    const float part_v = normed[partner];
    const float out = is_x ? (self_v * c - part_v * s) : (self_v * c + part_v * s);
    xout[base + tid] = (TO)out;
}

#define ROPE_NORM_OP(TI, TO, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const TI *q, const TI *k, const TI *q_weight, const TI *k_weight, \
    const TI *cos, const TI *sin, const uint32_t *positions, TO *q_out, TO *k_out, \
    const float q_eps, const float k_eps, \
    const unsigned int b, const unsigned int h, const unsigned int kh, \
    const unsigned int t, const unsigned int d, const unsigned int is_neox) { \
    rope_norm_positions<TI, TO>(q, k, q_weight, k_weight, cos, sin, positions, q_out, k_out, \
                                q_eps, k_eps, b, h, kh, t, d, is_neox); \
}

ROPE_NORM_OP(float, float, rope_norm_positions_f32)
ROPE_NORM_OP(float, __half, rope_norm_positions_f32_f16)
ROPE_NORM_OP(__half, __half, rope_norm_positions_f16)
ROPE_NORM_OP(hip_bfloat16, hip_bfloat16, rope_norm_positions_bf16)
#endif
