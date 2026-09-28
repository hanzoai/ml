// GeGLU, y = gelu(g) u, and its gradient.
#include "cuda_utils.cuh"
#include <stdint.h>

__device__ __forceinline__ float cdf(float x) {
    return 0.5f * (1.0f + erff(x * 0.70710678118654752f));
}

__device__ __forceinline__ float pdf(float x) {
    return 0.39894228040143268f * expf(-0.5f * x * x);
}

template <typename T>
__device__ void geglu_fwd(const size_t n, const T *g, const T *u, T *y) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const float x = static_cast<float>(g[i]);
        y[i] = static_cast<T>(x * cdf(x) * static_cast<float>(u[i]));
    }
}

template <typename T>
__device__ void geglu_bwd(const size_t n, const T *g, const T *u, const T *dy, T *dg, T *du) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        const float x = static_cast<float>(g[i]);
        const float d = static_cast<float>(dy[i]);
        const float c = cdf(x);
        dg[i] = static_cast<T>(d * static_cast<float>(u[i]) * (c + x * pdf(x)));
        du[i] = static_cast<T>(d * x * c);
    }
}

#define GEGLU(NAME, T)                                                                         \
extern "C" __global__ void geglu_fwd_##NAME(const size_t n, const T *g, const T *u, T *y) {   \
    geglu_fwd<T>(n, g, u, y);                                                                  \
}                                                                                              \
extern "C" __global__ void geglu_bwd_##NAME(                                                  \
    const size_t n, const T *g, const T *u, const T *dy, T *dg, T *du) {                       \
    geglu_bwd<T>(n, g, u, dy, dg, du);                                                         \
}

GEGLU(f32, float)
GEGLU(f16, __half)
GEGLU(bf16, __nv_bfloat16)
