// Fused F32 head_dim-512 online-softmax flash-decode attention.
//
// Port of ds4's `attention_decode_mixed_heads8_online_kernel` (ds4_cuda.cu:5423-5586),
// the DeepSeek-V4 decode speed keystone. V4 decode uses head_dim 512, above the 256 cap
// of the tensor-core flash path, so the engine otherwise falls back to an eager 3-pass
// attention. This is the single-token decode kernel: all-F32 CUDA cores (decode is
// memory-bound, so F32 keeps bit-closeness to the ds4 reference rather than using tensor
// cores), one warp per query head, 8 heads per block, Q resident in registers, K/V
// streamed 4 rows at a time through shared memory, and a running-max / running-sum online
// softmax so the whole thing is a single pass over the KV cache. The attention sink is
// folded in as an extra logit into the denominator only (no value contribution) after the
// KV loop, exactly as in ds4.
//
// Two adaptations from the ds4 reference (both faithful, documented for the port notes):
//   1. ds4 fuses K and V into one MLA latent tensor (the same shared tile serves both the
//      score dot and the output accumulation). Here K and V are separate inputs: the score
//      reads the K tile, the output accumulates the V tile. With k == v this is byte-for-
//      byte the ds4 arithmetic.
//   2. ds4's decode dispatch drives row selection through a KV ring buffer (pos0/raw_start/
//      raw_cap) and passes window == 0. Here the KV cache is a plain contiguous
//      [n_kv_head, kv_len, 512] tensor and the sliding window is applied directly with the
//      identical clamp from the kernel's own window branch (ds4_cuda.cu:5470-5472).
//
// Speculative-verify query width: q is [n_head, q_len, 512] with q_len in 1..=8 (the trailing
// block of tokens a draft proposed; the KV cache already holds all q_len new K/V rows appended
// at the end). One grid.z plane per query row, so each block is the 1-row kernel body applied to
// row s = blockIdx.z. Row s is the token at absolute position qpos = kv_len - q_len + s, and
// attends kv rows [start_row, end_row): end_row = qpos + 1 (causal, up to and including its own
// position) and start_row = max(0, qpos + 1 - window) when window != 0 (its own sliding window).
// q_len == 1 collapses to qpos = kv_len - 1, byte-identical to plain single-token decode.
//
// GQA/MQA: ds4 indexes the KV cache with no per-head offset (raw_kv + row*head_dim), i.e.
// a single shared KV head (MQA / n_kv_head == 1), which V4 uses. This port generalizes to
// n_kv_head >= 1: every query head h maps to kv head h / (n_head / n_kv_head). The grid is
// laid out so that all 8 warps (heads) in a block belong to the SAME kv head, preserving
// the ds4 8x global-load reuse of each K/V row across the heads that share it.

#include <cuda_runtime.h>
#include <stdint.h>

#define FATTN_WARP 32u
#define FATTN_HEADS_PER_BLOCK 8u
#define FATTN_BLOCK (FATTN_WARP * FATTN_HEADS_PER_BLOCK) // 256 threads
#define FATTN_HEAD_DIM 512u
#define FATTN_F4_PER_ROW (FATTN_HEAD_DIM / 4u) // 128 float4 per 512-dim row
#define FATTN_TILE_ROWS 4u

__device__ static float fattn_dot4_f32(float4 a, float4 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

__device__ static float fattn_warp_sum_f32(float v) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

extern "C" __global__ void hanzo_fattn_decode_f32_hd512_kernel(
        const float *q,     // [n_head, q_len, 512]
        const float *k,     // [n_kv_head, kv_len, 512]
        const float *v,     // [n_kv_head, kv_len, 512]
        const float *sinks, // [n_head] or nullptr
        float *out,         // [n_head, q_len, 512]
        uint32_t n_head,
        uint32_t n_kv_head,
        uint32_t kv_len,
        uint32_t q_len,
        uint32_t window,
        float scale) {
    const uint32_t lane = threadIdx.x & (FATTN_WARP - 1u);
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t s = blockIdx.z; // query row within the trailing block, 0..q_len-1

    // Head <-> block mapping. group = q-heads per kv-head (== n_head for MQA). Each block
    // owns 8 heads of ONE kv head, so the shared KV tile below is valid for all 8 warps.
    const uint32_t group = n_head / n_kv_head;
    const uint32_t blocks_per_kv = (group + FATTN_HEADS_PER_BLOCK - 1u) / FATTN_HEADS_PER_BLOCK;
    const uint32_t kv_head = blockIdx.y / blocks_per_kv;
    const uint32_t subgroup = blockIdx.y % blocks_per_kv;
    const uint32_t head_in_kv = subgroup * FATTN_HEADS_PER_BLOCK + warp;
    const uint32_t head = kv_head * group + head_in_kv;
    const bool valid_head = (head_in_kv < group) && (head < n_head);

    // Per-row causal + sliding window. Query row s is the token at absolute position
    // qpos = kv_len - q_len + s; it attends kv rows [start_row, end_row): end_row = qpos + 1
    // (causal, up to & incl. its own position), start_row = max(0, qpos+1-window) (its own
    // window). Identical clamp to ds4 (lo = qpos+1-window). q_len==1 => qpos = kv_len-1.
    const uint32_t qpos = kv_len - q_len + s;
    const uint32_t end_row = qpos + 1u;
    uint32_t start_row = 0u;
    if (window != 0u && end_row > window) start_row = end_row - window;
    const uint32_t n_score = end_row - start_row;

    __shared__ float4 k_shared[FATTN_TILE_ROWS * FATTN_F4_PER_ROW]; // 8 KB
    __shared__ float4 v_shared[FATTN_TILE_ROWS * FATTN_F4_PER_ROW]; // 8 KB

    // Q resident in registers: 4 float4 per lane spanning all 512 dims (lane, +32, +64, +96).
    float4 q0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 q1 = q0, q2 = q0, q3 = q0;
    if (valid_head) {
        const float4 *q4 = (const float4 *)(q + ((uint64_t)head * q_len + s) * FATTN_HEAD_DIM);
        q0 = q4[lane + 0u];
        q1 = q4[lane + 32u];
        q2 = q4[lane + 64u];
        q3 = q4[lane + 96u];
    }

    float max_s = -INFINITY;
    float sum_s = 0.f;
    float4 o0 = make_float4(0.f, 0.f, 0.f, 0.f);
    float4 o1 = o0, o2 = o0, o3 = o0;

    const uint64_t kv_base = (uint64_t)kv_head * kv_len; // row 0 of this kv-head

    for (uint32_t row0 = 0u; row0 < n_score; row0 += FATTN_TILE_ROWS) {
        const uint32_t nr = (n_score - row0 < FATTN_TILE_ROWS) ? (n_score - row0) : FATTN_TILE_ROWS;
        // Whole block cooperatively stages up to 4 K rows and 4 V rows into shared memory.
        for (uint32_t off = threadIdx.x; off < nr * FATTN_F4_PER_ROW; off += FATTN_BLOCK) {
            const uint32_t rr = off >> 7u;                       // which of the (<=4) tile rows
            const uint32_t c4 = off & (FATTN_F4_PER_ROW - 1u);   // which float4 within the row
            const uint32_t j = start_row + row0 + rr;            // absolute kv row
            const float4 *ksrc = (const float4 *)(k + (kv_base + j) * FATTN_HEAD_DIM);
            const float4 *vsrc = (const float4 *)(v + (kv_base + j) * FATTN_HEAD_DIM);
            k_shared[off] = ksrc[c4];
            v_shared[off] = vsrc[c4];
        }
        __syncthreads();
        if (valid_head) {
            for (uint32_t rr = 0u; rr < nr; rr++) {
                const float4 *ks = k_shared + rr * FATTN_F4_PER_ROW;
                const float4 *vs = v_shared + rr * FATTN_F4_PER_ROW;
                float4 k0 = ks[lane + 0u], k1 = ks[lane + 32u], k2 = ks[lane + 64u], k3 = ks[lane + 96u];
                float4 v0 = vs[lane + 0u], v1 = vs[lane + 32u], v2 = vs[lane + 64u], v3 = vs[lane + 96u];

                // 512-dim dot: per-lane partial over its 16 dims, warp-reduced, then scaled.
                float score = fattn_dot4_f32(q0, k0) + fattn_dot4_f32(q1, k1) +
                              fattn_dot4_f32(q2, k2) + fattn_dot4_f32(q3, k3);
                score = fattn_warp_sum_f32(score) * scale;
                score = __shfl_sync(0xffffffffu, score, 0);

                // Online softmax: rescale running sum + the four float4 V accumulators.
                const float new_m = fmaxf(max_s, score);
                const float old_scale = expf(max_s - new_m);
                const float row_scale = expf(score - new_m);
                sum_s = sum_s * old_scale + row_scale;
                o0.x = o0.x * old_scale + v0.x * row_scale;
                o0.y = o0.y * old_scale + v0.y * row_scale;
                o0.z = o0.z * old_scale + v0.z * row_scale;
                o0.w = o0.w * old_scale + v0.w * row_scale;
                o1.x = o1.x * old_scale + v1.x * row_scale;
                o1.y = o1.y * old_scale + v1.y * row_scale;
                o1.z = o1.z * old_scale + v1.z * row_scale;
                o1.w = o1.w * old_scale + v1.w * row_scale;
                o2.x = o2.x * old_scale + v2.x * row_scale;
                o2.y = o2.y * old_scale + v2.y * row_scale;
                o2.z = o2.z * old_scale + v2.z * row_scale;
                o2.w = o2.w * old_scale + v2.w * row_scale;
                o3.x = o3.x * old_scale + v3.x * row_scale;
                o3.y = o3.y * old_scale + v3.y * row_scale;
                o3.z = o3.z * old_scale + v3.z * row_scale;
                o3.w = o3.w * old_scale + v3.w * row_scale;
                max_s = new_m;
            }
        }
        __syncthreads();
    }

    if (!valid_head) return;

    // Attention sink: an extra logit folded into the denominator only (no value).
    if (sinks != nullptr) {
        const float sink = sinks[head];
        const float new_m = fmaxf(max_s, sink);
        const float old_scale = expf(max_s - new_m);
        const float sink_scale = expf(sink - new_m);
        sum_s = sum_s * old_scale + sink_scale;
        o0.x *= old_scale; o0.y *= old_scale; o0.z *= old_scale; o0.w *= old_scale;
        o1.x *= old_scale; o1.y *= old_scale; o1.z *= old_scale; o1.w *= old_scale;
        o2.x *= old_scale; o2.y *= old_scale; o2.z *= old_scale; o2.w *= old_scale;
        o3.x *= old_scale; o3.y *= old_scale; o3.z *= old_scale; o3.w *= old_scale;
    }

    const float inv_s = (sum_s == 0.f) ? 0.f : 1.f / sum_s;
    o0.x *= inv_s; o0.y *= inv_s; o0.z *= inv_s; o0.w *= inv_s;
    o1.x *= inv_s; o1.y *= inv_s; o1.z *= inv_s; o1.w *= inv_s;
    o2.x *= inv_s; o2.y *= inv_s; o2.z *= inv_s; o2.w *= inv_s;
    o3.x *= inv_s; o3.y *= inv_s; o3.z *= inv_s; o3.w *= inv_s;

    float4 *out4 = (float4 *)(out + ((uint64_t)head * q_len + s) * FATTN_HEAD_DIM);
    out4[lane + 0u] = o0;
    out4[lane + 32u] = o1;
    out4[lane + 64u] = o2;
    out4[lane + 96u] = o3;
}

// Host launcher. `stream` is a cudaStream_t (passed as void*). q/out are [n_head, q_len, 512]
// with q_len in 1..=8; k/v are [n_kv_head, kv_len, 512] (n_kv_head == 1 => plain [kv_len, 512])
// and must already hold all q_len new positions (kv_len >= q_len). `sinks` may be null. `scale`
// is the softmax scale applied to the QK dot (e.g. 1/sqrt(512)).
extern "C" void hanzo_fattn_decode_f32_hd512(
        void *stream,
        const float *q,
        const float *k,
        const float *v,
        const float *sinks,
        float *out,
        int n_head,
        int n_kv_head,
        int kv_len,
        int q_len,
        int window,
        float scale) {
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const uint32_t group = (uint32_t)n_head / (uint32_t)n_kv_head;
    const uint32_t blocks_per_kv = (group + FATTN_HEADS_PER_BLOCK - 1u) / FATTN_HEADS_PER_BLOCK;
    dim3 grid(1u, (uint32_t)n_kv_head * blocks_per_kv, (uint32_t)q_len);
    dim3 block(FATTN_BLOCK, 1u, 1u);
    hanzo_fattn_decode_f32_hd512_kernel<<<grid, block, 0, s>>>(
            q, k, v, sinks, out,
            (uint32_t)n_head, (uint32_t)n_kv_head, (uint32_t)kv_len,
            (uint32_t)q_len, (uint32_t)window, scale);
}
