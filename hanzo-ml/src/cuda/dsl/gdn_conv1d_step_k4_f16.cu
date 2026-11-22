#include <cuda_runtime.h>
#include <cuda_fp16.h>
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


extern "C" __global__ void __launch_bounds__(64)gdn_conv1d_step_f_ (
    const __half* __restrict__ buffer_0, const __half* __restrict__ buffer_1, __half* buffer_2, __half* buffer_3, const uint32* __restrict__ buffer_4, const __grid_constant__ info_st info
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
const __half l_10 = buffer_2[l_9];
const uint32 l_11 = l_5 + uint32(0);
buffer_2[l_11] = l_10;
const float l_12 = float(l_10);
const uint32 l_13 = l_6 + uint32(0);
const __half l_14 = buffer_1[l_13];
const float l_15 = float(l_14);
const float l_16 = l_12 * l_15;
l_mut_7 = l_mut_7 + l_16;
const uint32 l_17 = l_5 + uint32(1);
const uint32 l_18 = l_17 + uint32(1);
const __half l_19 = buffer_2[l_18];
const uint32 l_20 = l_5 + uint32(1);
buffer_2[l_20] = l_19;
const float l_21 = float(l_19);
const uint32 l_22 = l_6 + uint32(1);
const __half l_23 = buffer_1[l_22];
const float l_24 = float(l_23);
const float l_25 = l_21 * l_24;
l_mut_7 = l_mut_7 + l_25;
const uint32 l_26 = l_5 + uint32(2);
const uint32 l_27 = l_26 + uint32(1);
const __half l_28 = buffer_2[l_27];
const uint32 l_29 = l_5 + uint32(2);
buffer_2[l_29] = l_28;
const float l_30 = float(l_28);
const uint32 l_31 = l_6 + uint32(2);
const __half l_32 = buffer_1[l_31];
const float l_33 = float(l_32);
const float l_34 = l_30 * l_33;
l_mut_7 = l_mut_7 + l_34;
const __half l_35 = buffer_0[idxGlobal];
const uint32 l_36 = l_5 + uint32(3);
buffer_2[l_36] = l_35;
const float l_37 = float(l_35);
const uint32 l_38 = l_6 + uint32(3);
const __half l_39 = buffer_1[l_38];
const float l_40 = float(l_39);
const float l_41 = l_37 * l_40;
l_mut_7 = l_mut_7 + l_41;
const float l_42 = -l_mut_7;
const float l_43 = exp(l_42);
const float l_44 = float(1.0) + l_43;
const float l_45 = l_mut_7 / l_44;
const __half l_46 = __half(l_45);
buffer_3[idxGlobal] = l_46;
}

}
