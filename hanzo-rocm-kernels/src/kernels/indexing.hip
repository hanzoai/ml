#ifndef __HIPCC__
#define __device__
#define __global__
#define __forceinline__
#else
#include <hip/hip_runtime.h>
#endif

#include <stddef.h>
#include <stdint.h>

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
    for (int d = num_dims - 1; d >= 0; d--) {
        strided_i += (idx % dims[d]) * strides[d];
        idx /= dims[d];
    }
    return strided_i;
}

template <typename T>
__host__ __device__
constexpr T max_value();

template <>
__host__ __device__
constexpr int64_t max_value<int64_t>() {
    return 0x7FFFFFFFFFFFFFFFLL;
}

template <>
__host__ __device__
constexpr uint32_t max_value<uint32_t>() {
    return 0xFFFFFFFFu;
}

template <>
__host__ __device__
constexpr uint8_t max_value<uint8_t>() {
    return 0xFFu;
}

template<typename T, typename I>
__device__ void index_select(
    const size_t numel,
    const size_t num_dims,
    const size_t *info,
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size
) {
    const size_t *dims = info;
    const size_t *strides = info + num_dims;
    bool b = is_contiguous(num_dims, dims, strides);
    for (unsigned int dst_i = blockIdx.x * blockDim.x + threadIdx.x; dst_i < numel; dst_i += blockDim.x * gridDim.x) {
          unsigned int left_i = dst_i / (ids_dim_size * right_size);
          unsigned int id_i = dst_i / right_size % ids_dim_size;
          unsigned int right_i = dst_i % right_size;
          if (ids[id_i] == max_value<I>()) {
            out[dst_i] = static_cast<T>(0);
          } else {
            unsigned int src_i = left_i * (src_dim_size * right_size) + ids[id_i] * right_size + right_i;
            unsigned strided_i = b ? src_i : get_strided_index(src_i, num_dims, dims, strides);
            out[dst_i] = inp[strided_i];
          }
    }
}

#define IS_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t ids_dim_size, \
    const size_t right_size \
) { index_select(numel, num_dims, info, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size); }

IS_OP(float, int64_t, is_i64_f32)
IS_OP(double, int64_t, is_i64_f64)
IS_OP(uint8_t, int64_t, is_i64_u8)
IS_OP(uint32_t, int64_t, is_i64_u32)
IS_OP(int64_t, int64_t, is_i64_i64)

IS_OP(float, uint32_t, is_u32_f32)
IS_OP(double, uint32_t, is_u32_f64)
IS_OP(uint8_t, uint32_t, is_u32_u8)
IS_OP(uint32_t, uint32_t, is_u32_u32)
IS_OP(int64_t, uint32_t, is_u32_i64)

IS_OP(float, uint8_t, is_u8_f32)
IS_OP(double, uint8_t, is_u8_f64)
IS_OP(uint8_t, uint8_t, is_u8_u8)
IS_OP(uint32_t, uint8_t, is_u8_u32)
IS_OP(int64_t, uint8_t, is_u8_i64)

// Contiguous index_select: identical math to index_select but with a contiguous
// source (b == true), so it never dereferences `info` (dims/strides). This lets
// the host skip the H2D clone_htod of the shape-metadata buffer, which is the op
// that trips hipGraph capture (HIP 906) on the embedding/RoPE gather hot path.
// Signature matches IS_OP so the same launcher works with a null `info` pointer.
template <typename T, typename I>
__device__ void index_select_contig(
    const size_t numel,
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size
) {
    for (unsigned int dst_i = blockIdx.x * blockDim.x + threadIdx.x; dst_i < numel; dst_i += blockDim.x * gridDim.x) {
          unsigned int left_i = dst_i / (ids_dim_size * right_size);
          unsigned int id_i = dst_i / right_size % ids_dim_size;
          unsigned int right_i = dst_i % right_size;
          if (ids[id_i] == max_value<I>()) {
            out[dst_i] = static_cast<T>(0);
          } else {
            unsigned int src_i = left_i * (src_dim_size * right_size) + ids[id_i] * right_size + right_i;
            out[dst_i] = inp[src_i];
          }
    }
}

#define ISC_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME(  \
    const size_t numel,  \
    const size_t num_dims, \
    const size_t *info, \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t ids_dim_size, \
    const size_t right_size \
) { (void)num_dims; (void)info; index_select_contig(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size); }

ISC_OP(float, int64_t, isc_i64_f32)
ISC_OP(double, int64_t, isc_i64_f64)
ISC_OP(uint8_t, int64_t, isc_i64_u8)
ISC_OP(uint32_t, int64_t, isc_i64_u32)
ISC_OP(int64_t, int64_t, isc_i64_i64)

ISC_OP(float, uint32_t, isc_u32_f32)
ISC_OP(double, uint32_t, isc_u32_f64)
ISC_OP(uint8_t, uint32_t, isc_u32_u8)
ISC_OP(uint32_t, uint32_t, isc_u32_u32)
ISC_OP(int64_t, uint32_t, isc_u32_i64)

ISC_OP(float, uint8_t, isc_u8_f32)
ISC_OP(double, uint8_t, isc_u8_f64)
ISC_OP(uint8_t, uint8_t, isc_u8_u8)
ISC_OP(uint32_t, uint8_t, isc_u8_u32)
ISC_OP(int64_t, uint8_t, isc_u8_i64)

// gather: ids has the same shape as out; each ids[i] replaces the `dim`
// coordinate when reading from inp. Requires contiguous inp + ids.
template <typename T, typename I>
__device__ void gather(
    const size_t numel,
    const I *ids,
    const T *inp,
    T *out,
    const size_t left_size,
    const size_t src_dim_size,
    const size_t ids_dim_size,
    const size_t right_size
) {
    for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel; i += blockDim.x * gridDim.x) {
        size_t post = i % right_size;
        size_t pre = i / (ids_dim_size * right_size);
        size_t j = (size_t)ids[i];
        size_t src_i = (pre * src_dim_size + j) * right_size + post;
        out[i] = inp[src_i];
    }
}

#define GATHER_OP(TYPENAME, INDEX_TYPENAME, FN_NAME) \
extern "C" __global__ void FN_NAME( \
    const size_t numel, \
    const INDEX_TYPENAME *ids, \
    const TYPENAME *inp, \
    TYPENAME *out, \
    const size_t left_size, \
    const size_t src_dim_size, \
    const size_t ids_dim_size, \
    const size_t right_size \
) { gather(numel, ids, inp, out, left_size, src_dim_size, ids_dim_size, right_size); }

GATHER_OP(float, int64_t, gather_i64_f32)
GATHER_OP(double, int64_t, gather_i64_f64)
GATHER_OP(uint8_t, int64_t, gather_i64_u8)
GATHER_OP(uint32_t, int64_t, gather_i64_u32)
GATHER_OP(int64_t, int64_t, gather_i64_i64)

GATHER_OP(float, uint32_t, gather_u32_f32)
GATHER_OP(double, uint32_t, gather_u32_f64)
GATHER_OP(uint8_t, uint32_t, gather_u32_u8)
GATHER_OP(uint32_t, uint32_t, gather_u32_u32)
GATHER_OP(int64_t, uint32_t, gather_u32_i64)

GATHER_OP(float, uint8_t, gather_u8_f32)
GATHER_OP(double, uint8_t, gather_u8_f64)
GATHER_OP(uint8_t, uint8_t, gather_u8_u8)
GATHER_OP(uint32_t, uint8_t, gather_u8_u32)
GATHER_OP(int64_t, uint8_t, gather_u8_i64)

#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bfloat16.h>

IS_OP(hip_bfloat16, int64_t, is_i64_bf16)
IS_OP(hip_bfloat16, uint32_t, is_u32_bf16)
IS_OP(hip_bfloat16, uint8_t, is_u8_bf16)

IS_OP(__half, int64_t, is_i64_f16)
IS_OP(__half, uint32_t, is_u32_f16)
IS_OP(__half, uint8_t, is_u8_f16)

ISC_OP(hip_bfloat16, int64_t, isc_i64_bf16)
ISC_OP(hip_bfloat16, uint32_t, isc_u32_bf16)
ISC_OP(hip_bfloat16, uint8_t, isc_u8_bf16)

ISC_OP(__half, int64_t, isc_i64_f16)
ISC_OP(__half, uint32_t, isc_u32_f16)
ISC_OP(__half, uint8_t, isc_u8_f16)

GATHER_OP(hip_bfloat16, int64_t, gather_i64_bf16)
GATHER_OP(hip_bfloat16, uint32_t, gather_u32_bf16)
GATHER_OP(hip_bfloat16, uint8_t, gather_u8_bf16)

GATHER_OP(__half, int64_t, gather_i64_f16)
GATHER_OP(__half, uint32_t, gather_u32_f16)
GATHER_OP(__half, uint8_t, gather_u8_f16)
#endif
