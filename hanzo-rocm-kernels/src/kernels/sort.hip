// ROCm/HIP single-block bitonic argsort (ported from candle/ggml argsort.cu).
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

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>
ASORT_OP(__half, f16)
ASORT_OP(hip_bfloat16, bf16)
#endif
