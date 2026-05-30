// ROCm/HIP ternary (where_cond) kernels — compiled at runtime.
#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>

__device__ unsigned int wc_strided_index(
    unsigned int idx,
    const size_t num_dims,
    const size_t *dims,
    const size_t *strides
) {
    unsigned int strided_i = 0;
    for (int d = num_dims - 1; d >= 0; d--) {
        strided_i += (idx % dims[d]) * strides[d];
        idx /= dims[d];
    }
    return strided_i;
}

// out[i] = cond[i] != 0 ? t[i] : f[i]   (cond/t/f each independently strided)
// layout buffer = [dims (num_dims), cond_strides, t_strides, f_strides]
#define WHERE_OP(TYPENAME, ID_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const size_t *dims_and_strides, \
    const ID_TYPENAME *cond, \
    const TYPENAME *t, \
    const TYPENAME *f, \
    TYPENAME *out \
) { \
    const size_t *dims = dims_and_strides; \
    const size_t *cond_s = dims_and_strides + 1 * num_dims; \
    const size_t *t_s = dims_and_strides + 2 * num_dims; \
    const size_t *f_s = dims_and_strides + 3 * num_dims; \
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
        unsigned int ci = wc_strided_index(i, num_dims, dims, cond_s); \
        unsigned int ti = wc_strided_index(i, num_dims, dims, t_s); \
        unsigned int fi = wc_strided_index(i, num_dims, dims, f_s); \
        out[i] = (cond[ci] != 0) ? t[ti] : f[fi]; \
    } \
}

WHERE_OP(float,    uint8_t,  where_u8_f32)
WHERE_OP(double,   uint8_t,  where_u8_f64)
WHERE_OP(uint8_t,  uint8_t,  where_u8_u8)
WHERE_OP(uint32_t, uint8_t,  where_u8_u32)
WHERE_OP(int64_t,  uint8_t,  where_u8_i64)

WHERE_OP(float,    uint32_t, where_u32_f32)
WHERE_OP(double,   uint32_t, where_u32_f64)
WHERE_OP(uint8_t,  uint32_t, where_u32_u8)
WHERE_OP(uint32_t, uint32_t, where_u32_u32)
WHERE_OP(int64_t,  uint32_t, where_u32_i64)

WHERE_OP(float,    int64_t,  where_i64_f32)
WHERE_OP(double,   int64_t,  where_i64_f64)
WHERE_OP(uint8_t,  int64_t,  where_i64_u8)
WHERE_OP(uint32_t, int64_t,  where_i64_u32)
WHERE_OP(int64_t,  int64_t,  where_i64_i64)

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

WHERE_OP(__half,       uint8_t,  where_u8_f16)
WHERE_OP(__half,       uint32_t, where_u32_f16)
WHERE_OP(__half,       int64_t,  where_i64_f16)

WHERE_OP(hip_bfloat16, uint8_t,  where_u8_bf16)
WHERE_OP(hip_bfloat16, uint32_t, where_u32_bf16)
WHERE_OP(hip_bfloat16, int64_t,  where_i64_bf16)
#endif
