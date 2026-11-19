#include <cuda_runtime.h>
#include <cuda_bf16.h>
typedef unsigned int uint;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long int uint64;
typedef signed char int8;
typedef signed short int16;
typedef signed int int32;
typedef signed long long int int64;

struct info_st {
    uint32 static_meta[14];
};


extern "C" __global__ void __launch_bounds__(64)gdn_gating_io_f_ (
    const __nv_bfloat16* __restrict__ buffer_0, const __nv_bfloat16* __restrict__ buffer_1, const float* __restrict__ buffer_2, const float* __restrict__ buffer_3, __nv_bfloat16* buffer_4, __nv_bfloat16* buffer_5, const uint32* __restrict__ buffer_6, const __grid_constant__ info_st info
) {
uint3 absoluteIdx = make_uint3(
    blockIdx.x * blockDim.x + threadIdx.x,
    blockIdx.y * blockDim.y + threadIdx.y,
    blockIdx.z * blockDim.z + threadIdx.z
);
uint32 idxGlobal = (
                absoluteIdx.z * gridDim.x * blockDim.x * gridDim.y * blockDim.y)
                + (absoluteIdx.y * gridDim.x * blockDim.x)
                + absoluteIdx.x;
float l_mut_11;
const uint32 l_0 = buffer_6[uint32(0)];
const bool l_1 = idxGlobal < l_0;
if (l_1) {
const uint32 l_2 = buffer_6[uint32(1)];
const uint32 l_3 = idxGlobal % l_2;
const __nv_bfloat16 l_4 = buffer_0[idxGlobal];
const float l_5 = float(l_4);
const __nv_bfloat16 l_6 = buffer_1[idxGlobal];
const float l_7 = float(l_6);
const float l_8 = buffer_3[l_3];
const float l_9 = l_7 + l_8;
const bool l_10 = l_9 > float(20.0);
if (l_10) {
l_mut_11 = l_9;
} else {
const float l_12 = exp(l_9);
const float l_13 = float(1.0) + l_12;
const float l_14 = log(l_13);
l_mut_11 = l_14;
}
const float l_15 = -l_5;
const float l_16 = exp(l_15);
const float l_17 = float(1.0) + l_16;
const float l_18 = float(1.0) / l_17;
const float l_19 = buffer_2[l_3];
const float l_20 = exp(l_19);
const float l_21 = -l_20;
const float l_22 = l_21 * l_mut_11;
const __nv_bfloat16 l_23 = __nv_bfloat16(l_18);
buffer_4[idxGlobal] = l_23;
const __nv_bfloat16 l_24 = __nv_bfloat16(l_22);
buffer_5[idxGlobal] = l_24;
}

}
