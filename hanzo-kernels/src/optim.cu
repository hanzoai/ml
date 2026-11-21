// AdamW over a contiguous F32 parameter, and a buffer's sum of squares.
#include "cuda_utils.cuh"
#include <stdint.h>

extern "C" __global__ void adamw_f32(
    const uint32_t n,
    const float lr,
    const float beta1,
    const float beta2,
    const float eps,
    const float weight_decay,
    const float scale_m,
    const float scale_v,
    const float grad_scale,
    float *w,
    const float *g,
    float *m,
    float *v
) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float gi = g[i] * grad_scale;
        const float mi = beta1 * m[i] + (1.0f - beta1) * gi;
        const float vi = beta2 * v[i] + (1.0f - beta2) * gi * gi;
        m[i] = mi;
        v[i] = vi;
        const float update = (mi * scale_m) / (sqrtf(vi * scale_v) + eps);
        w[i] = w[i] * (1.0f - lr * weight_decay) - lr * update;
    }
}

static __device__ __forceinline__ float wsum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

extern "C" __global__ void sumsq_f32(const uint32_t n, const float *x, float *out) {
    __shared__ float partial[32];
    float acc = 0.0f;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float xi = x[i];
        acc += xi * xi;
    }
    acc = wsum(acc);
    const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) {
        partial[warp] = acc;
    }
    __syncthreads();
    if (warp == 0) {
        const int warps = (blockDim.x + 31) / 32;
        float total = lane < warps ? partial[lane] : 0.0f;
        total = wsum(total);
        if (lane == 0) {
            atomicAdd(out, total);
        }
    }
}
