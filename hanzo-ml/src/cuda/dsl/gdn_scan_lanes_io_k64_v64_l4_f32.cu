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
    uint32 static_meta[16];
};


extern "C" __global__ void __launch_bounds__(64)gdn_scan_lanes_io_f_f32 (
    const float* __restrict__ buffer_0, const float* __restrict__ buffer_1, const float* __restrict__ buffer_2, const float* __restrict__ buffer_3, const float* __restrict__ buffer_4, float* buffer_5, float* buffer_6, const uint32* __restrict__ buffer_7, const __grid_constant__ info_st info
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
uint32 warpSizeChecked = min(warpSize, blockDim.x * blockDim.y * blockDim.z);
float l_arr_20[16];

uint32 l_mut_6;
float l_mut_37;
const uint32 l_0 = buffer_7[uint32(0)];
const uint32 l_1 = buffer_7[uint32(1)];
const uint32 l_2 = l_0 * uint32(64);
const uint32 l_3 = idxGlobal / uint32(4);
const uint32 l_4 = idxGlobal % uint32(4);
const bool l_5 = l_3 < l_2;
l_mut_6 = l_3;
const bool l_7 = !l_5;
if (l_7) {
l_mut_6 = uint32(0);
}
const uint32 l_8 = l_mut_6 / uint32(64);
const uint32 l_9 = l_mut_6 % uint32(64);
const uint32 l_10 = l_8 * l_1;
const uint32 l_11 = l_10 * uint32(64);
const uint32 l_12 = l_4 * uint32(16);
const uint32 l_13 = l_11 + l_12;
const uint32 l_14 = l_8 * l_1;
const uint32 l_15 = l_14 * uint32(64);
const uint32 l_16 = l_15 + l_9;
const uint32 l_17 = l_8 * l_1;
const uint32 l_18 = l_8 * uint32(64);
const uint32 l_19 = l_18 * uint32(64);

for (uint32 l_mut_21 = uint32(0); l_mut_21 < uint32(16); ++l_mut_21) {
const uint32 l_22 = l_4 * uint32(16);
const uint32 l_23 = l_22 + l_mut_21;
const uint32 l_24 = l_23 * uint32(64);
const uint32 l_25 = l_19 + l_24;
const uint32 l_26 = l_25 + l_9;
const float l_27 = buffer_5[l_26];
l_arr_20[l_mut_21] = l_27;
}

for (uint32 l_mut_28 = uint32(0); l_mut_28 < l_1; ++l_mut_28) {
const uint32 l_29 = l_17 + l_mut_28;
const float l_30 = buffer_3[l_29];
const float l_31 = exp(l_30);
const uint32 l_32 = l_17 + l_mut_28;
const float l_33 = buffer_4[l_32];
const uint32 l_34 = l_mut_28 * uint32(64);
const uint32 l_35 = l_16 + l_34;
const float l_36 = buffer_2[l_35];
l_mut_37 = float(0.0);

for (uint32 l_mut_38 = uint32(0); l_mut_38 < uint32(16); ++l_mut_38) {
const float l_39 = l_arr_20[l_mut_38];
const float l_40 = l_39 * l_31;
l_arr_20[l_mut_38] = l_40;
const uint32 l_41 = l_mut_28 * uint32(64);
const uint32 l_42 = l_13 + l_41;
const uint32 l_43 = l_42 + l_mut_38;
const float l_44 = buffer_1[l_43];
const float l_45 = l_40 * l_44;
l_mut_37 = l_mut_37 + l_45;
}
l_mut_6 = uint32(1);
while (true) {
const bool l_46 = l_mut_6 < uint32(4);
const bool l_47 = !l_46;
if (l_47) {
break;}
const float l_48 = { __shfl_xor_sync(-1, l_mut_37, l_mut_6) };
l_mut_37 = l_mut_37 + l_48;
l_mut_6 = l_mut_6 * uint32(2);
}
const float l_49 = l_36 - l_mut_37;
const float l_50 = l_49 * l_33;
l_mut_37 = float(0.0);

for (uint32 l_mut_51 = uint32(0); l_mut_51 < uint32(16); ++l_mut_51) {
const uint32 l_52 = l_mut_28 * uint32(64);
const uint32 l_53 = l_13 + l_52;
const uint32 l_54 = l_53 + l_mut_51;
const float l_55 = buffer_1[l_54];
const float l_56 = l_arr_20[l_mut_51];
const float l_57 = l_55 * l_50;
const float l_58 = l_56 + l_57;
l_arr_20[l_mut_51] = l_58;
const uint32 l_59 = l_mut_28 * uint32(64);
const uint32 l_60 = l_13 + l_59;
const uint32 l_61 = l_60 + l_mut_51;
const float l_62 = buffer_0[l_61];
const float l_63 = l_58 * l_62;
l_mut_37 = l_mut_37 + l_63;
}
l_mut_6 = uint32(1);
while (true) {
const bool l_64 = l_mut_6 < uint32(4);
const bool l_65 = !l_64;
if (l_65) {
break;}
const float l_66 = { __shfl_xor_sync(-1, l_mut_37, l_mut_6) };
l_mut_37 = l_mut_37 + l_66;
l_mut_6 = l_mut_6 * uint32(2);
}
const bool l_67 = l_4 == uint32(0);
const bool l_68 = l_5 && l_67;
if (l_68) {
const uint32 l_69 = l_mut_28 * uint32(64);
const uint32 l_70 = l_16 + l_69;
buffer_6[l_70] = l_mut_37;
}
}
if (l_5) {

for (uint32 l_mut_71 = uint32(0); l_mut_71 < uint32(16); ++l_mut_71) {
const uint32 l_72 = l_4 * uint32(16);
const uint32 l_73 = l_72 + l_mut_71;
const uint32 l_74 = l_73 * uint32(64);
const uint32 l_75 = l_19 + l_74;
const uint32 l_76 = l_75 + l_9;
const float l_77 = l_arr_20[l_mut_71];
buffer_5[l_76] = l_77;
}
}

}
