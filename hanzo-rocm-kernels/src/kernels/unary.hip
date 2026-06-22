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

// Inline dims/strides metadata passed BY VALUE in the kernel param block (no device
// buffer, no host->device copy -> hipGraph-capture clean). Must match the Rust
// `DimsStrides` (rocm_backend/mod.rs): ROCM_DS_MAX(8) * ROCM_DS_SETS(4) = 32 size_t.
// Layout: [dims, strides_0, strides_1, strides_2], each block ROCM_DS_MAX wide.
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

#define UNARY_OP(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const DimsStrides ds, \
    const TYPENAME *input, \
    TYPENAME *out \
) { \
    const size_t *dims = ds.v; \
    const size_t *strides = ds.v + ROCM_DS_MAX; \
    bool contiguous = num_dims == 0 || is_contiguous(num_dims, dims, strides); \
    if (contiguous) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = input[i]; \
            out[i] = FUNC; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int src_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                src_i += i_dim * strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = input[src_i]; \
            out[i] = FUNC; \
        } \
    } \
}

#define UNARY_OP_SCALAR(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const DimsStrides ds, \
    const TYPENAME *input, \
    TYPENAME *out, \
    TYPENAME scalar \
) { \
    const size_t *dims = ds.v; \
    const size_t *strides = ds.v + ROCM_DS_MAX; \
    bool contiguous = num_dims == 0 || is_contiguous(num_dims, dims, strides); \
    if (contiguous) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            TYPENAME x = input[i]; \
            out[i] = FUNC; \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int src_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                src_i += i_dim * strides[d]; \
                tmp_i /= dims[d]; \
            } \
            TYPENAME x = input[src_i]; \
            out[i] = FUNC; \
        } \
    } \
}

UNARY_OP(float, ucopy_f32, x)
UNARY_OP(double, ucopy_f64, x)
UNARY_OP(uint8_t, ucopy_u8, x)
UNARY_OP(uint32_t, ucopy_u32, x)
UNARY_OP(int64_t, ucopy_i64, x)

UNARY_OP(float, urelu_f32, (x > 0.0f ? x : 0.0f))
UNARY_OP(double, urelu_f64, (x > 0.0 ? x : 0.0))

UNARY_OP(float, usigmoid_f32, (1.0f / (1.0f + expf(-x))))
UNARY_OP(double, usigmoid_f64, (1.0 / (1.0 + exp(-x))))

UNARY_OP(float, utan_f32, tanhf(x))
UNARY_OP(double, utan_f64, tanh(x))

UNARY_OP(float, uexp_f32, expf(x))
UNARY_OP(double, uexp_f64, exp(x))

UNARY_OP(float, ulog_f32, logf(x))
UNARY_OP(double, ulog_f64, log(x))

UNARY_OP(float, usin_f32, sinf(x))
UNARY_OP(double, usin_f64, sin(x))

UNARY_OP(float, ucos_f32, cosf(x))
UNARY_OP(double, ucos_f64, cos(x))

UNARY_OP(float, usqrt_f32, sqrtf(x))
UNARY_OP(double, usqrt_f64, sqrt(x))

UNARY_OP(float, uabs_f32, fabsf(x))
UNARY_OP(double, uabs_f64, fabs(x))
UNARY_OP(int64_t, uabs_i64, (x < 0 ? -x : x))

UNARY_OP(float, uneg_f32, -x)
UNARY_OP(double, uneg_f64, -x)
UNARY_OP(int64_t, uneg_i64, -x)

UNARY_OP(float, urecip_f32, (1.0f / x))
UNARY_OP(double, urecip_f64, (1.0 / x))

UNARY_OP(float, ufloor_f32, floorf(x))
UNARY_OP(double, ufloor_f64, floor(x))

UNARY_OP(float, uceil_f32, ceilf(x))
UNARY_OP(double, uceil_f64, ceil(x))

UNARY_OP(float, uround_f32, roundf(x))
UNARY_OP(double, uround_f64, round(x))

UNARY_OP(float, usqr_f32, (x * x))
UNARY_OP(double, usqr_f64, (x * x))

UNARY_OP(float, usign_f32, (x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f)))
UNARY_OP(double, usign_f64, (x > 0.0 ? 1.0 : (x < 0.0 ? -1.0 : 0.0)))

UNARY_OP(float, ugelu_f32, (0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)))))
UNARY_OP(double, ugelu_f64, (0.5 * x * (1.0 + tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))))

UNARY_OP(float, ugelu_erf_f32, (0.5f * x * (1.0f + erff(x * 0.7071067811865475f))))
UNARY_OP(double, ugelu_erf_f64, (0.5 * x * (1.0 + erf(x * 0.7071067811865475))))

UNARY_OP(float, usilu_f32, (x / (1.0f + expf(-x))))
UNARY_OP(double, usilu_f64, (x / (1.0 + exp(-x))))

UNARY_OP(float, uerf_f32, erff(x))
UNARY_OP(double, uerf_f64, erf(x))

UNARY_OP_SCALAR(float, uelu_f32, (x > 0.0f ? x : scalar * (expf(x) - 1.0f)))
UNARY_OP_SCALAR(double, uelu_f64, (x > 0.0 ? x : scalar * (exp(x) - 1.0)))

UNARY_OP_SCALAR(float, upowf_f32, powf(x, scalar))
UNARY_OP_SCALAR(double, upowf_f64, pow(x, scalar))

// 16-bit float variants: load/compute in float, cast result back.
#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

#define UNARY_OP_F(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const DimsStrides ds, \
    const TYPENAME *input, \
    TYPENAME *out \
) { \
    const size_t *dims = ds.v; \
    const size_t *strides = ds.v + ROCM_DS_MAX; \
    bool contiguous = num_dims == 0 || is_contiguous(num_dims, dims, strides); \
    if (contiguous) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            float x = (float)input[i]; \
            out[i] = (TYPENAME)(FUNC); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int src_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                src_i += i_dim * strides[d]; \
                tmp_i /= dims[d]; \
            } \
            float x = (float)input[src_i]; \
            out[i] = (TYPENAME)(FUNC); \
        } \
    } \
}

#define UNARY_OP_SCALAR_F(TYPENAME, FN_NAME, FUNC) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const size_t num_dims, \
    const DimsStrides ds, \
    const TYPENAME *input, \
    TYPENAME *out, \
    TYPENAME scalar_in \
) { \
    const size_t *dims = ds.v; \
    const size_t *strides = ds.v + ROCM_DS_MAX; \
    float scalar = (float)scalar_in; \
    bool contiguous = num_dims == 0 || is_contiguous(num_dims, dims, strides); \
    if (contiguous) { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            float x = (float)input[i]; \
            out[i] = (TYPENAME)(FUNC); \
        } \
    } else { \
        for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) { \
            unsigned int tmp_i = i; \
            unsigned int src_i = 0; \
            for (int d = num_dims - 1; d >= 0; d--) { \
                unsigned int i_dim = tmp_i % dims[d]; \
                src_i += i_dim * strides[d]; \
                tmp_i /= dims[d]; \
            } \
            float x = (float)input[src_i]; \
            out[i] = (TYPENAME)(FUNC); \
        } \
    } \
}

#define UNARY_ALL_F(T, SUF) \
UNARY_OP_F(T, ucopy_##SUF, x) \
UNARY_OP_F(T, urelu_##SUF, (x > 0.0f ? x : 0.0f)) \
UNARY_OP_F(T, usigmoid_##SUF, (1.0f / (1.0f + expf(-x)))) \
UNARY_OP_F(T, utan_##SUF, tanhf(x)) \
UNARY_OP_F(T, uexp_##SUF, expf(x)) \
UNARY_OP_F(T, ulog_##SUF, logf(x)) \
UNARY_OP_F(T, usin_##SUF, sinf(x)) \
UNARY_OP_F(T, ucos_##SUF, cosf(x)) \
UNARY_OP_F(T, usqrt_##SUF, sqrtf(x)) \
UNARY_OP_F(T, uabs_##SUF, fabsf(x)) \
UNARY_OP_F(T, uneg_##SUF, -x) \
UNARY_OP_F(T, urecip_##SUF, (1.0f / x)) \
UNARY_OP_F(T, ufloor_##SUF, floorf(x)) \
UNARY_OP_F(T, uceil_##SUF, ceilf(x)) \
UNARY_OP_F(T, uround_##SUF, roundf(x)) \
UNARY_OP_F(T, usqr_##SUF, (x * x)) \
UNARY_OP_F(T, usign_##SUF, (x > 0.0f ? 1.0f : (x < 0.0f ? -1.0f : 0.0f))) \
UNARY_OP_F(T, ugelu_##SUF, (0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x))))) \
UNARY_OP_F(T, ugelu_erf_##SUF, (0.5f * x * (1.0f + erff(x * 0.7071067811865475f)))) \
UNARY_OP_F(T, usilu_##SUF, (x / (1.0f + expf(-x)))) \
UNARY_OP_F(T, uerf_##SUF, erff(x)) \
UNARY_OP_SCALAR_F(T, uelu_##SUF, (x > 0.0f ? x : scalar * (expf(x) - 1.0f))) \
UNARY_OP_SCALAR_F(T, upowf_##SUF, powf(x, scalar))

UNARY_ALL_F(__half, f16)
UNARY_ALL_F(hip_bfloat16, bf16)
#endif
