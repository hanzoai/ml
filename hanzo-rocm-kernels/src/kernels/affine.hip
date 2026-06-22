#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>
#include <math.h>

// Inline dims/strides metadata passed BY VALUE (no device buffer -> hipGraph-capture
// clean). Must match Rust `DimsStrides`: ROCM_DS_MAX(8) * ROCM_DS_SETS(4) = 32 size_t.
#define ROCM_DS_MAX 8
#define ROCM_DS_SETS 4
typedef struct { size_t v[ROCM_DS_MAX * ROCM_DS_SETS]; } DimsStrides;

__device__ bool is_contiguous(
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    size_t acc = 1;
    for (unsigned int d = 0; d < num_dims; d++) {
        unsigned int dim_idx = num_dims - 1 - d;
        if (dims[dim_idx] > 1 && acc != strides[dim_idx]) {
            return false;
        }
        acc *= dims[dim_idx];
    }
    return true;
}

__device__ unsigned int get_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    unsigned int tmp_i = idx;
    for (int d = num_dims - 1; d >= 0; d--) {
        unsigned int i_dim = tmp_i % dims[d];
        strided_i += i_dim * strides[d];
        tmp_i /= dims[d];
    }
    return strided_i;
}

#define AFFINE_OP(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const DimsStrides ds, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const TYPENAME mul, \
    const TYPENAME add \
) { \
    const size_t *dims = ds.v; \
    const size_t *strides = ds.v + ROCM_DS_MAX; \
    if (num_dims == 0 || is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = inp[i] * mul + add; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int src_i = get_strided_index(i, num_dims, dims, strides); \
            out[i] = inp[src_i] * mul + add; \
        } \
    } \
}

AFFINE_OP(float, affine_f32)
AFFINE_OP(double, affine_f64)
AFFINE_OP(uint8_t, affine_u8)
AFFINE_OP(uint32_t, affine_u32)
AFFINE_OP(int64_t, affine_i64)

// 16-bit float variants: compute in float, cast back.
#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define AFFINE_OP_F(TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const DimsStrides ds, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const TYPENAME mul, \
    const TYPENAME add \
) { \
    float mulf = (float)mul; \
    float addf = (float)add; \
    const size_t *dims = ds.v; \
    const size_t *strides = ds.v + ROCM_DS_MAX; \
    if (num_dims == 0 || is_contiguous(num_dims, dims, strides)) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            out[i] = (TYPENAME)((float)inp[i] * mulf + addf); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int src_i = get_strided_index(i, num_dims, dims, strides); \
            out[i] = (TYPENAME)((float)inp[src_i] * mulf + addf); \
        } \
    } \
}

AFFINE_OP_F(__half, affine_f16)
AFFINE_OP_F(hip_bfloat16, affine_bf16)
#endif
