// ROCm/HIP single-block bitonic argsort (ported from hanzo-ml/ggml argsort.cu).
// One block per row; shared memory holds the row's indices. Compiled at runtime.
#ifndef __HIPCC__
#define __device__
#define __global__
#else
#include <hip/hip_runtime.h>
#endif

#include <stdint.h>

#define SORT_ORDER_ASC 1
#define SORT_ORDER_DESC 0

template <typename T>
static inline __device__ void hip_swap(T &a, T &b) {
    T tmp = a;
    a = b;
    b = tmp;
}

template <int order, typename T>
static __device__ void k_argsort(const T *x, uint32_t *dst, const int ncols, int ncols_pad) {
    int row = blockIdx.x;
    const T *x_row = x + row * ncols;
    extern __shared__ int dst_row[];

    for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
        dst_row[col] = col;
    }
    __syncthreads();

    for (int k = 2; k <= ncols_pad; k *= 2) {
        for (int j = k / 2; j > 0; j /= 2) {
            for (int col = threadIdx.x; col < ncols_pad; col += blockDim.x) {
                int ixj = col ^ j;
                if (ixj > col) {
                    if ((col & k) == 0) {
                        if (dst_row[col] >= ncols ||
                            (dst_row[ixj] < ncols && (order == SORT_ORDER_ASC
                                ? x_row[dst_row[col]] > x_row[dst_row[ixj]]
                                : x_row[dst_row[col]] < x_row[dst_row[ixj]]))) {
                            hip_swap(dst_row[col], dst_row[ixj]);
                        }
                    } else {
                        if (dst_row[ixj] >= ncols ||
                            (dst_row[col] < ncols && (order == SORT_ORDER_ASC
                                ? x_row[dst_row[col]] < x_row[dst_row[ixj]]
                                : x_row[dst_row[col]] > x_row[dst_row[ixj]]))) {
                            hip_swap(dst_row[col], dst_row[ixj]);
                        }
                    }
                }
            }
            __syncthreads();
        }
    }

    for (int col = threadIdx.x; col < ncols; col += blockDim.x) {
        dst[row * ncols + col] = dst_row[col];
    }
}

#define ASORT_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void asort_asc_##RUST_NAME( \
    const TYPENAME *x, uint32_t *dst, const int ncols, int ncols_pad) { \
    k_argsort<SORT_ORDER_ASC>(x, dst, ncols, ncols_pad); \
} \
extern "C" __global__ void asort_desc_##RUST_NAME( \
    const TYPENAME *x, uint32_t *dst, const int ncols, int ncols_pad) { \
    k_argsort<SORT_ORDER_DESC>(x, dst, ncols, ncols_pad); \
}

ASORT_OP(float, f32)
ASORT_OP(double, f64)
ASORT_OP(uint8_t, u8)
ASORT_OP(uint32_t, u32)
ASORT_OP(int64_t, i64)

// ---- Top-k selection over the last dim (k-pass block argmax) ----
// One block per row selects the k largest with a CANONICAL tie-break -- value DESC, index ASC
// (first-index-wins on equal scores) -- identical to a stable descending sort narrowed to k, and to
// the fused moe_route selection. Each round is a block-wide argmax over the not-yet-selected columns;
// the chosen column is clobbered to -inf. This drops the full O(n log^2 n) bitonic sort + its memory
// traffic when only k of n are wanted (MoE routing: k<=8 of n_experts<=256). Values are reduced in
// f32 (bit-exact for the f16/bf16 round-trip since each value is loaded once and written back).
// Outputs: values [nrows,k] in the input dtype, indices [nrows,k] as u32.
#define TOPK_THREADS 256
#define TOPK_MAX_N 1024
#define TOPK_NEG_INF (-3.402823466e38f)

static inline __device__ float topk_to_f32(float v) { return v; }
template <typename T> static inline __device__ T topk_from_f32(float v);
template <> inline __device__ float topk_from_f32<float>(float v) { return v; }

template <typename T>
static __device__ void k_topk(const T *__restrict__ x, T *__restrict__ out_v,
                              uint32_t *__restrict__ out_i, const int ncols, const int k) {
    const int row = blockIdx.x;
    const int t = threadIdx.x;
    const T *__restrict__ xr = x + (size_t)row * ncols;
    __shared__ float sv[TOPK_MAX_N];
    __shared__ float rv[TOPK_THREADS];
    __shared__ int ri[TOPK_THREADS];
    for (int c = t; c < ncols; c += TOPK_THREADS) sv[c] = topk_to_f32(xr[c]);
    __syncthreads();
    for (int r = 0; r < k; ++r) {
        float bv = TOPK_NEG_INF;
        int bi = ncols;
        for (int c = t; c < ncols; c += TOPK_THREADS) {
            const float v = sv[c];
            if (v > bv || (v == bv && c < bi)) { bv = v; bi = c; }
        }
        rv[t] = bv;
        ri[t] = bi;
        __syncthreads();
        for (int s = TOPK_THREADS / 2; s > 0; s >>= 1) {
            if (t < s) {
                const float ov = rv[t + s];
                const int oi = ri[t + s];
                if (ov > rv[t] || (ov == rv[t] && oi < ri[t])) { rv[t] = ov; ri[t] = oi; }
            }
            __syncthreads();
        }
        if (t == 0) {
            const int best = ri[0];
            out_i[(size_t)row * k + r] = (unsigned int)best;
            out_v[(size_t)row * k + r] = topk_from_f32<T>(rv[0]);
            sv[best] = TOPK_NEG_INF;
        }
        __syncthreads();
    }
}

#define TOPK_OP(TYPENAME, RUST_NAME) \
extern "C" __global__ void topk_last_dim_##RUST_NAME( \
    const TYPENAME *x, TYPENAME *out_v, uint32_t *out_i, const int ncols, const int k) { \
    k_topk<TYPENAME>(x, out_v, out_i, ncols, k); \
}

TOPK_OP(float, f32)

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
ASORT_OP(__half, f16)
ASORT_OP(hip_bfloat16, bf16)

static inline __device__ float topk_to_f32(__half v) { return __half2float(v); }
static inline __device__ float topk_to_f32(hip_bfloat16 v) { return (float)v; }
template <> inline __device__ __half topk_from_f32<__half>(float v) { return __float2half(v); }
template <> inline __device__ hip_bfloat16 topk_from_f32<hip_bfloat16>(float v) { return hip_bfloat16(v); }
TOPK_OP(__half, f16)
TOPK_OP(hip_bfloat16, bf16)
#endif
