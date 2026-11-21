// ROCm/HIP binary operation kernels
// These are embedded as strings and compiled at runtime

#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>

// Inline dims/strides metadata passed BY VALUE (no device buffer -> hipGraph-capture
// clean). Must match Rust `DimsStrides`: ROCM_DS_MAX(8) * ROCM_DS_SETS(4) = 32 size_t,
// layout [dims, lhs_strides, rhs_strides, _].
#define ROCM_DS_MAX 8
#define ROCM_DS_SETS 4
typedef struct { size_t v[ROCM_DS_MAX * ROCM_DS_SETS]; } DimsStrides;

// Helper to check if tensor is contiguous
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

// Binary operation macro - generates kernel for in-place operation
#define BINARY_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const DimsStrides ds, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *out \
) { \
    const size_t *dims = ds.v; \
    const size_t *lhs_strides = ds.v + 1 * ROCM_DS_MAX; \
    const size_t *rhs_strides = ds.v + 2 * ROCM_DS_MAX; \
    bool lhs_cont = num_dims == 0 || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = num_dims == 0 || is_contiguous(num_dims, dims, rhs_strides); \
    if (lhs_cont && rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = lhs[i]; \
            TYPENAME y = rhs[i]; \
            out[i] = FUNC; \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = lhs[i]; \
            TYPENAME y = rhs[rhs_i]; \
            out[i] = FUNC; \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = lhs[lhs_i]; \
            TYPENAME y = rhs[i]; \
            out[i] = FUNC; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = lhs[lhs_i]; \
            TYPENAME y = rhs[rhs_i]; \
            out[i] = FUNC; \
        } \
    } \
}

// badd kernels
BINARY_OP(float, badd_f32, x + y)
BINARY_OP(double, badd_f64, x + y)
BINARY_OP(uint8_t, badd_u8, x + y)
BINARY_OP(uint32_t, badd_u32, x + y)
BINARY_OP(int64_t, badd_i64, x + y)

// bdiv kernels
BINARY_OP(float, bdiv_f32, x / y)
BINARY_OP(double, bdiv_f64, x / y)
BINARY_OP(uint8_t, bdiv_u8, x / y)
BINARY_OP(uint32_t, bdiv_u32, x / y)
BINARY_OP(int64_t, bdiv_i64, x / y)

// bmul kernels
BINARY_OP(float, bmul_f32, x * y)
BINARY_OP(double, bmul_f64, x * y)
BINARY_OP(uint8_t, bmul_u8, x * y)
BINARY_OP(uint32_t, bmul_u32, x * y)
BINARY_OP(int64_t, bmul_i64, x * y)

// bsub kernels
BINARY_OP(float, bsub_f32, x - y)
BINARY_OP(double, bsub_f64, x - y)
BINARY_OP(uint8_t, bsub_u8, x - y)
BINARY_OP(uint32_t, bsub_u32, x - y)
BINARY_OP(int64_t, bsub_i64, x - y)

// bminimum kernels
BINARY_OP(float, bminimum_f32, (x < y ? x : y))
BINARY_OP(double, bminimum_f64, (x < y ? x : y))
BINARY_OP(uint8_t, bminimum_u8, (x < y ? x : y))
BINARY_OP(uint32_t, bminimum_u32, (x < y ? x : y))
BINARY_OP(int64_t, bminimum_i64, (x < y ? x : y))

// bmaximum kernels
BINARY_OP(float, bmaximum_f32, (x > y ? x : y))
BINARY_OP(double, bmaximum_f64, (x > y ? x : y))
BINARY_OP(uint8_t, bmaximum_u8, (x > y ? x : y))
BINARY_OP(uint32_t, bmaximum_u32, (x > y ? x : y))
BINARY_OP(int64_t, bmaximum_i64, (x > y ? x : y))

// Fused SwiGLU: out = silu(x) * y = (x / (1 + exp(-x))) * y. One launch instead of silu + mul.
BINARY_OP(float, silu_mul_f32, ((x / (1.0f + expf(-x))) * y))
BINARY_OP(double, silu_mul_f64, ((x / (1.0 + exp(-x))) * y))

// 16-bit float variants. hip_bfloat16/__half lack reliable arithmetic
// operators, so load+compute in float and cast the result back.
#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define BINARY_OP_F(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const DimsStrides ds, \
    const TYPENAME *lhs, \
    const TYPENAME *rhs, \
    TYPENAME *out \
) { \
    const size_t *dims = ds.v; \
    const size_t *lhs_strides = ds.v + 1 * ROCM_DS_MAX; \
    const size_t *rhs_strides = ds.v + 2 * ROCM_DS_MAX; \
    bool lhs_cont = num_dims == 0 || is_contiguous(num_dims, dims, lhs_strides); \
    bool rhs_cont = num_dims == 0 || is_contiguous(num_dims, dims, rhs_strides); \
    if (lhs_cont && rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            float x = (float)lhs[i]; \
            float y = (float)rhs[i]; \
            out[i] = (TYPENAME)(FUNC); \
        } \
    } else if (lhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            float x = (float)lhs[i]; \
            float y = (float)rhs[rhs_i]; \
            out[i] = (TYPENAME)(FUNC); \
        } \
    } else if (rhs_cont) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                lhs_i += i_dim * lhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            float x = (float)lhs[lhs_i]; \
            float y = (float)rhs[i]; \
            out[i] = (TYPENAME)(FUNC); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int lhs_i = 0; \
            unsigned int rhs_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                lhs_i += i_dim * lhs_strides[d]; \
                rhs_i += i_dim * rhs_strides[d]; \
                tmp_i /= dims[d]; \
            } \
            float x = (float)lhs[lhs_i]; \
            float y = (float)rhs[rhs_i]; \
            out[i] = (TYPENAME)(FUNC); \
        } \
    } \
}

BINARY_OP_F(__half, badd_f16, x + y)
BINARY_OP_F(__half, bdiv_f16, x / y)
BINARY_OP_F(__half, bmul_f16, x * y)
BINARY_OP_F(__half, bsub_f16, x - y)
BINARY_OP_F(__half, bminimum_f16, (x < y ? x : y))
BINARY_OP_F(__half, bmaximum_f16, (x > y ? x : y))

BINARY_OP_F(hip_bfloat16, badd_bf16, x + y)
BINARY_OP_F(hip_bfloat16, bdiv_bf16, x / y)
BINARY_OP_F(hip_bfloat16, bmul_bf16, x * y)
BINARY_OP_F(hip_bfloat16, bsub_bf16, x - y)
BINARY_OP_F(hip_bfloat16, bminimum_bf16, (x < y ? x : y))
BINARY_OP_F(hip_bfloat16, bmaximum_bf16, (x > y ? x : y))

// Fused SwiGLU (x, y already loaded as float in BINARY_OP_F; silu accum in f32).
BINARY_OP_F(__half, silu_mul_f16, ((x / (1.0f + expf(-x))) * y))
BINARY_OP_F(hip_bfloat16, silu_mul_bf16, ((x / (1.0f + expf(-x))) * y))
#endif
