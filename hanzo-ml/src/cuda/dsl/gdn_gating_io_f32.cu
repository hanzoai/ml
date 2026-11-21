#include <cuda_runtime.h>
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


extern "C" __global__ void __launch_bounds__(64)gdn_gating_io_f_f32 (
    const float* __restrict__ buffer_0, const float* __restrict__ buffer_1, const float* __restrict__ buffer_2, const float* __restrict__ buffer_3, float* buffer_4, float* buffer_5, const uint32* __restrict__ buffer_6, const __grid_constant__ info_st info
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
float l_mut_9;
const uint32 l_0 = buffer_6[uint32(0)];
const bool l_1 = idxGlobal < l_0;
if (l_1) {
const uint32 l_2 = buffer_6[uint32(1)];
const uint32 l_3 = idxGlobal % l_2;
const float l_4 = buffer_0[idxGlobal];
const float l_5 = buffer_1[idxGlobal];
const float l_6 = buffer_3[l_3];
const float l_7 = l_5 + l_6;
const bool l_8 = l_7 > float(20.0);
if (l_8) {
l_mut_9 = l_7;
} else {
const float l_10 = exp(l_7);
const float l_11 = float(1.0) + l_10;
const float l_12 = log(l_11);
l_mut_9 = l_12;
}
const float l_13 = -l_4;
const float l_14 = exp(l_13);
const float l_15 = float(1.0) + l_14;
const float l_16 = float(1.0) / l_15;
const float l_17 = buffer_2[l_3];
const float l_18 = exp(l_17);
const float l_19 = -l_18;
const float l_20 = l_19 * l_mut_9;
buffer_4[idxGlobal] = l_16;
buffer_5[idxGlobal] = l_20;
}

}
