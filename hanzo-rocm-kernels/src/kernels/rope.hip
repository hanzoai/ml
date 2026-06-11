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
