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
    uint32 static_meta[6];
};


extern "C" __global__ void __launch_bounds__(32)flash_attn_split_kv_reduce_f_ (
    const float* __restrict__ buffer_0, __nv_bfloat16* buffer_1, const uint32* __restrict__ buffer_2, const __grid_constant__ info_st info
) {
uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
uint32 blockIdxGlobal = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x;
float l_arr_7[2];

float l_mut_5;
float l_mut_6;
float l_mut_20;
const uint32 l_0 = buffer_2[uint32(2)];
const uint32 l_1 = blockIdxGlobal + l_0;
const uint32 l_2 = buffer_2[uint32(0)];
const uint32 l_3 = buffer_2[uint32(1)];
const bool l_4 = l_1 < l_2;
if (l_4) {
l_mut_5 = float(-3.3999999521443642e38);
l_mut_6 = float(0.0);

for (uint32 l_mut_8 = uint32(0); l_mut_8 < uint32(2); ++l_mut_8) {
l_arr_7[l_mut_8] = float(0.0);
}
const uint32 l_9 = l_1 * l_3;
const uint32 l_10 = l_9 * uint32(66);

for (uint32 l_mut_11 = uint32(0); l_mut_11 < l_3; ++l_mut_11) {
const uint32 l_12 = l_mut_11 * uint32(66);
const uint32 l_13 = l_10 + l_12;
const uint32 l_14 = l_13 + uint32(64);
const float l_15 = buffer_0[l_14];
const uint32 l_16 = l_13 + uint32(64);
const uint32 l_17 = l_16 + uint32(1);
const float l_18 = buffer_0[l_17];
const bool l_19 = l_18 > float(0.0);
if (l_19) {
l_mut_20 = l_mut_5;
const bool l_21 = l_15 > l_mut_20;
if (l_21) {
l_mut_20 = l_15;
}
const float l_22 = l_mut_5 - l_mut_20;
const float l_23 = exp(l_22);
const float l_24 = l_15 - l_mut_20;
const float l_25 = exp(l_24);
const float l_26 = l_mut_6 * l_23;
const float l_27 = l_18 * l_25;
const float l_28 = l_26 + l_27;
l_mut_6 = l_28;

for (uint32 l_mut_29 = uint32(0); l_mut_29 < uint32(2); ++l_mut_29) {
const uint32 l_30 = l_mut_29 * uint32(32);
const uint32 l_31 = l_30 + threadIdxGlobal;
const float l_32 = l_arr_7[l_mut_29];
const float l_33 = l_32 * l_23;
const uint32 l_34 = l_13 + l_31;
const float l_35 = buffer_0[l_34];
const float l_36 = l_35 * l_25;
const float l_37 = l_33 + l_36;
l_arr_7[l_mut_29] = l_37;
}
l_mut_5 = l_mut_20;
}
}
l_mut_20 = float(0.0);
const bool l_38 = l_mut_6 > float(0.0);
if (l_38) {
const float l_39 = float(1.0) / l_mut_6;
l_mut_20 = l_39;
}
const uint32 l_40 = l_1 * uint32(64);

for (uint32 l_mut_41 = uint32(0); l_mut_41 < uint32(2); ++l_mut_41) {
const uint32 l_42 = l_mut_41 * uint32(32);
const uint32 l_43 = l_42 + threadIdxGlobal;
const uint32 l_44 = l_40 + l_43;
const float l_45 = l_arr_7[l_mut_41];
const float l_46 = l_45 * l_mut_20;
const __nv_bfloat16 l_47 = __nv_bfloat16(l_46);
buffer_1[l_44] = l_47;
}
}

}
