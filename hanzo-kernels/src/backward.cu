// Backward passes of the last-dim softmax and LayerNorm, a block per row, in F32.
#include "cuda_utils.cuh"
#include <stdint.h>

static __device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

static __device__ float bsum(float v, float *shared) {
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32, warps = blockDim.x / 32;
    v = wsum(v);
    __syncthreads();
    if (lane == 0) {
        shared[warp] = v;
    }
    __syncthreads();
    float t = lane < warps ? shared[lane] : 0.0f;
    return wsum(t);
}

template <typename T>
__device__ void layernorm_bwd_rows(
    const uint32_t d,
    const float eps,
    const T *x,
    const T *dy,
    const T *alpha,
    T *dx,
    float *stats
) {
    __shared__ float shared[32];
    const uint32_t row = blockIdx.x;
    const size_t off = (size_t)row * d;
    const float n = float(d);
    float s = 0.0f;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        s += static_cast<float>(x[off + i]);
    }
    const float mean = bsum(s, shared) / n;
    float q = 0.0f;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float c = static_cast<float>(x[off + i]) - mean;
        q += c * c;
    }
    const float r = rsqrtf(bsum(q, shared) / n + eps);
    float su = 0.0f;
    float sux = 0.0f;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float xh = (static_cast<float>(x[off + i]) - mean) * r;
        const float u = static_cast<float>(dy[off + i]) * static_cast<float>(alpha[i]);
        su += u;
        sux += u * xh;
    }
    su = bsum(su, shared) / n;
    sux = bsum(sux, shared) / n;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float xh = (static_cast<float>(x[off + i]) - mean) * r;
        const float u = static_cast<float>(dy[off + i]) * static_cast<float>(alpha[i]);
        dx[off + i] = static_cast<T>(r * (u - su - xh * sux));
    }
    if (threadIdx.x == 0) {
        stats[2 * row] = mean;
        stats[2 * row + 1] = r;
    }
}

// `dparams` is [dalpha, dbeta], zeroed by the caller.
template <typename T>
__device__ void layernorm_bwd_params(
    const uint32_t n,
    const uint32_t d,
    const uint32_t rows_per,
    const T *x,
    const T *dy,
    const float *stats,
    float *dparams
) {
    const uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= d) {
        return;
    }
    const uint32_t r0 = blockIdx.y * rows_per;
    const uint32_t r1 = min(n, r0 + rows_per);
    float da = 0.0f;
    float db = 0.0f;
    for (uint32_t r = r0; r < r1; r++) {
        const float g = static_cast<float>(dy[(size_t)r * d + j]);
        da += g * (static_cast<float>(x[(size_t)r * d + j]) - stats[2 * r]) * stats[2 * r + 1];
        db += g;
    }
    atomicAdd(&dparams[j], da);
    atomicAdd(&dparams[d + j], db);
}

template <typename T>
__device__ void softmax_bwd_rows(const uint32_t d, const T *y, const T *dy, T *dx) {
    __shared__ float shared[32];
    const size_t off = (size_t)blockIdx.x * d;
    float dot = 0.0f;
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        dot += static_cast<float>(dy[off + i]) * static_cast<float>(y[off + i]);
    }
    dot = bsum(dot, shared);
    for (uint32_t i = threadIdx.x; i < d; i += blockDim.x) {
        const float yi = static_cast<float>(y[off + i]);
        dx[off + i] = static_cast<T>(yi * (static_cast<float>(dy[off + i]) - dot));
    }
}

#define NORM_BWD(NAME, T)                                                                      \
extern "C" __global__ void layernorm_bwd_##NAME(                                              \
    const uint32_t d, const float eps, const T *x, const T *dy, const T *alpha, T *dx,         \
    float *stats) {                                                                            \
    layernorm_bwd_rows<T>(d, eps, x, dy, alpha, dx, stats);                                    \
}                                                                                              \
extern "C" __global__ void layernorm_bwd_params_##NAME(                                       \
    const uint32_t n, const uint32_t d, const uint32_t rows_per, const T *x, const T *dy,      \
    const float *stats, float *dparams) {                                                      \
    layernorm_bwd_params<T>(n, d, rows_per, x, dy, stats, dparams);                            \
}                                                                                              \
extern "C" __global__ void softmax_bwd_##NAME(                                                \
    const uint32_t d, const T *y, const T *dy, T *dx) {                                        \
    softmax_bwd_rows<T>(d, y, dy, dx);                                                         \
}

NORM_BWD(f32, float)
NORM_BWD(f16, __half)
NORM_BWD(bf16, __nv_bfloat16)
