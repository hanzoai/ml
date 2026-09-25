#include <metal_stdlib>
using namespace metal;

// Backward passes of the row-wise normalizations, one threadgroup per row, accumulated in F32.

// Sum of `v` over the threadgroup, returned to every thread. `shared` holds one float per
// simdgroup and is free again when this returns.
METAL_FUNC float tg_sum(float v, threadgroup float *shared, uint lane, uint sg, uint nsg) {
    v = simd_sum(v);
    if (lane == 0) {
        shared[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float t = lane < nsg ? shared[lane] : 0.0f;
    t = simd_sum(t);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return t;
}

// LayerNorm, per row of width d, with x^ = (x - mean) r, r = (var + eps)^-1/2 and u = dy alpha:
//   dx = r (u - mean(u) - x^ mean(u x^))
// The row's (mean, r) go to `stats` for the parameter gradients.
template <typename T>
METAL_FUNC void layernorm_bwd_rows(
    constant uint &d,
    constant float &eps,
    device const T *x,
    device const T *dy,
    device const T *alpha,
    device T *dx,
    device float *stats,
    uint row,
    uint tid,
    uint tg,
    uint lane,
    uint sg,
    threadgroup float *shared
) {
    const uint off = row * d;
    const uint nsg = (tg + 31) / 32;
    const float n = float(d);
    float s = 0.0f;
    for (uint i = tid; i < d; i += tg) {
        s += float(x[off + i]);
    }
    const float mean = tg_sum(s, shared, lane, sg, nsg) / n;
    float q = 0.0f;
    for (uint i = tid; i < d; i += tg) {
        const float c = float(x[off + i]) - mean;
        q += c * c;
    }
    const float r = rsqrt(tg_sum(q, shared, lane, sg, nsg) / n + eps);
    float su = 0.0f;
    float sux = 0.0f;
    for (uint i = tid; i < d; i += tg) {
        const float xh = (float(x[off + i]) - mean) * r;
        const float u = float(dy[off + i]) * float(alpha[i]);
        su += u;
        sux += u * xh;
    }
    su = tg_sum(su, shared, lane, sg, nsg) / n;
    sux = tg_sum(sux, shared, lane, sg, nsg) / n;
    for (uint i = tid; i < d; i += tg) {
        const float xh = (float(x[off + i]) - mean) * r;
        const float u = float(dy[off + i]) * float(alpha[i]);
        dx[off + i] = T(r * (u - su - xh * sux));
    }
    if (tid == 0) {
        stats[2 * row] = mean;
        stats[2 * row + 1] = r;
    }
}

// dalpha = sum_rows dy x^, dbeta = sum_rows dy, into dparams[0..d] and dparams[d..2d] (F32,
// zeroed by the caller). Thread (j, c) sums column j over the rows of chunk c.
template <typename T>
METAL_FUNC void layernorm_bwd_params(
    constant uint &n,
    constant uint &d,
    constant uint &rows_per,
    device const T *x,
    device const T *dy,
    device const float *stats,
    device atomic_float *dparams,
    uint2 pos
) {
    const uint j = pos.x;
    if (j >= d) {
        return;
    }
    const uint r0 = pos.y * rows_per;
    const uint r1 = min(n, r0 + rows_per);
    float da = 0.0f;
    float db = 0.0f;
    for (uint r = r0; r < r1; r++) {
        const float g = float(dy[r * d + j]);
        da += g * (float(x[r * d + j]) - stats[2 * r]) * stats[2 * r + 1];
        db += g;
    }
    atomic_fetch_add_explicit(&dparams[j], da, memory_order_relaxed);
    atomic_fetch_add_explicit(&dparams[d + j], db, memory_order_relaxed);
}

// Softmax over the last dim, per row: dx = y (dy - <dy, y>).
template <typename T>
METAL_FUNC void softmax_bwd_rows(
    constant uint &d,
    device const T *y,
    device const T *dy,
    device T *dx,
    uint row,
    uint tid,
    uint tg,
    uint lane,
    uint sg,
    threadgroup float *shared
) {
    const uint off = row * d;
    const uint nsg = (tg + 31) / 32;
    float dot = 0.0f;
    for (uint i = tid; i < d; i += tg) {
        dot += float(dy[off + i]) * float(y[off + i]);
    }
    dot = tg_sum(dot, shared, lane, sg, nsg);
    for (uint i = tid; i < d; i += tg) {
        const float yi = float(y[off + i]);
        dx[off + i] = T(yi * (float(dy[off + i]) - dot));
    }
}

#define NORM_BWD(NAME, T)                                                                      \
kernel void layernorm_bwd_##NAME(                                                             \
    constant uint &d, constant float &eps,                                                     \
    device const T *x, device const T *dy, device const T *alpha,                              \
    device T *dx, device float *stats,                                                         \
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],    \
    uint tg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],             \
    uint sg [[simdgroup_index_in_threadgroup]]) {                                              \
    threadgroup float shared[32];                                                              \
    layernorm_bwd_rows<T>(d, eps, x, dy, alpha, dx, stats, row, tid, tg, lane, sg, shared);    \
}                                                                                              \
kernel void layernorm_bwd_params_##NAME(                                                      \
    constant uint &n, constant uint &d, constant uint &rows_per,                               \
    device const T *x, device const T *dy, device const float *stats,                          \
    device atomic_float *dparams, uint2 pos [[thread_position_in_grid]]) {                     \
    layernorm_bwd_params<T>(n, d, rows_per, x, dy, stats, dparams, pos);                       \
}                                                                                              \
kernel void softmax_bwd_##NAME(                                                               \
    constant uint &d, device const T *y, device const T *dy, device T *dx,                     \
    uint row [[threadgroup_position_in_grid]], uint tid [[thread_position_in_threadgroup]],    \
    uint tg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],             \
    uint sg [[simdgroup_index_in_threadgroup]]) {                                              \
    threadgroup float shared[32];                                                              \
    softmax_bwd_rows<T>(d, y, dy, dx, row, tid, tg, lane, sg, shared);                         \
}

NORM_BWD(f32, float)
NORM_BWD(f16, half)
NORM_BWD(bf16, bfloat)
