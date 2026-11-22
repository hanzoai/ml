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
    uint32 static_meta[10];
};


extern "C" __global__ void __launch_bounds__(64)gdn_conv1d_step_f_f32 (
    const float* __restrict__ buffer_0, const float* __restrict__ buffer_1, float* buffer_2, float* buffer_3, const uint32* __restrict__ buffer_4, const __grid_constant__ info_st info
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
float l_mut_7;
const uint32 l_0 = buffer_4[uint32(0)];
const uint32 l_1 = buffer_4[uint32(1)];
const uint32 l_2 = l_0 * l_1;
const bool l_3 = idxGlobal < l_2;
if (l_3) {
const uint32 l_4 = idxGlobal % l_1;
const uint32 l_5 = idxGlobal * uint32(4);
const uint32 l_6 = l_4 * uint32(4);
l_mut_7 = float(0.0);
const uint32 l_8 = l_5 + uint32(0);
const uint32 l_9 = l_8 + uint32(1);
const float l_10 = buffer_2[l_9];
const uint32 l_11 = l_5 + uint32(0);
buffer_2[l_11] = l_10;
const uint32 l_12 = l_6 + uint32(0);
const float l_13 = buffer_1[l_12];
const float l_14 = l_10 * l_13;
l_mut_7 = l_mut_7 + l_14;
const uint32 l_15 = l_5 + uint32(1);
const uint32 l_16 = l_15 + uint32(1);
const float l_17 = buffer_2[l_16];
const uint32 l_18 = l_5 + uint32(1);
buffer_2[l_18] = l_17;
const uint32 l_19 = l_6 + uint32(1);
const float l_20 = buffer_1[l_19];
const float l_21 = l_17 * l_20;
l_mut_7 = l_mut_7 + l_21;
const uint32 l_22 = l_5 + uint32(2);
const uint32 l_23 = l_22 + uint32(1);
const float l_24 = buffer_2[l_23];
const uint32 l_25 = l_5 + uint32(2);
buffer_2[l_25] = l_24;
const uint32 l_26 = l_6 + uint32(2);
const float l_27 = buffer_1[l_26];
const float l_28 = l_24 * l_27;
l_mut_7 = l_mut_7 + l_28;
const float l_29 = buffer_0[idxGlobal];
const uint32 l_30 = l_5 + uint32(3);
buffer_2[l_30] = l_29;
const uint32 l_31 = l_6 + uint32(3);
const float l_32 = buffer_1[l_31];
const float l_33 = l_29 * l_32;
l_mut_7 = l_mut_7 + l_33;
const float l_34 = -l_mut_7;
const float l_35 = exp(l_34);
const float l_36 = float(1.0) + l_35;
const float l_37 = l_mut_7 / l_36;
buffer_3[idxGlobal] = l_37;
}

}
