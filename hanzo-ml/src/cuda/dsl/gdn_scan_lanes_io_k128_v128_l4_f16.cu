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
    uint32 static_meta[16];
};


extern "C" __global__ void __launch_bounds__(64)gdn_scan_lanes_io_f_ (
    const __half* __restrict__ buffer_0, const __half* __restrict__ buffer_1, const __half* __restrict__ buffer_2, const __half* __restrict__ buffer_3, const __half* __restrict__ buffer_4, float* buffer_5, __half* buffer_6, const uint32* __restrict__ buffer_7, const __grid_constant__ info_st info
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
float l_arr_20[32];

uint32 l_mut_6;
float l_mut_40;
const uint32 l_0 = buffer_7[uint32(0)];
const uint32 l_1 = buffer_7[uint32(1)];
const uint32 l_2 = l_0 * uint32(128);
const uint32 l_3 = idxGlobal / uint32(4);
const uint32 l_4 = idxGlobal % uint32(4);
const bool l_5 = l_3 < l_2;
l_mut_6 = l_3;
const bool l_7 = !l_5;
if (l_7) {
l_mut_6 = uint32(0);
}
const uint32 l_8 = l_mut_6 / uint32(128);
const uint32 l_9 = l_mut_6 % uint32(128);
const uint32 l_10 = l_8 * l_1;
const uint32 l_11 = l_10 * uint32(128);
const uint32 l_12 = l_4 * uint32(32);
const uint32 l_13 = l_11 + l_12;
const uint32 l_14 = l_8 * l_1;
const uint32 l_15 = l_14 * uint32(128);
const uint32 l_16 = l_15 + l_9;
const uint32 l_17 = l_8 * l_1;
const uint32 l_18 = l_8 * uint32(128);
const uint32 l_19 = l_18 * uint32(128);

for (uint32 l_mut_21 = uint32(0); l_mut_21 < uint32(32); ++l_mut_21) {
const uint32 l_22 = l_4 * uint32(32);
const uint32 l_23 = l_22 + l_mut_21;
const uint32 l_24 = l_23 * uint32(128);
const uint32 l_25 = l_19 + l_24;
const uint32 l_26 = l_25 + l_9;
const float l_27 = buffer_5[l_26];
l_arr_20[l_mut_21] = l_27;
}

for (uint32 l_mut_28 = uint32(0); l_mut_28 < l_1; ++l_mut_28) {
const uint32 l_29 = l_17 + l_mut_28;
const __half l_30 = buffer_3[l_29];
const float l_31 = float(l_30);
const float l_32 = exp(l_31);
const uint32 l_33 = l_17 + l_mut_28;
const __half l_34 = buffer_4[l_33];
const float l_35 = float(l_34);
const uint32 l_36 = l_mut_28 * uint32(128);
const uint32 l_37 = l_16 + l_36;
const __half l_38 = buffer_2[l_37];
const float l_39 = float(l_38);
l_mut_40 = float(0.0);

for (uint32 l_mut_41 = uint32(0); l_mut_41 < uint32(32); ++l_mut_41) {
const float l_42 = l_arr_20[l_mut_41];
const float l_43 = l_42 * l_32;
l_arr_20[l_mut_41] = l_43;
const uint32 l_44 = l_mut_28 * uint32(128);
const uint32 l_45 = l_13 + l_44;
const uint32 l_46 = l_45 + l_mut_41;
const __half l_47 = buffer_1[l_46];
const float l_48 = float(l_47);
const float l_49 = l_43 * l_48;
l_mut_40 = l_mut_40 + l_49;
}
l_mut_6 = uint32(1);
while (true) {
const bool l_50 = l_mut_6 < uint32(4);
const bool l_51 = !l_50;
if (l_51) {
break;}
const float l_52 = { __shfl_xor_sync(-1, l_mut_40, l_mut_6) };
l_mut_40 = l_mut_40 + l_52;
l_mut_6 = l_mut_6 * uint32(2);
}
const float l_53 = l_39 - l_mut_40;
const float l_54 = l_53 * l_35;
l_mut_40 = float(0.0);

for (uint32 l_mut_55 = uint32(0); l_mut_55 < uint32(32); ++l_mut_55) {
const uint32 l_56 = l_mut_28 * uint32(128);
const uint32 l_57 = l_13 + l_56;
const uint32 l_58 = l_57 + l_mut_55;
const __half l_59 = buffer_1[l_58];
const float l_60 = float(l_59);
const float l_61 = l_arr_20[l_mut_55];
const float l_62 = l_60 * l_54;
const float l_63 = l_61 + l_62;
l_arr_20[l_mut_55] = l_63;
const uint32 l_64 = l_mut_28 * uint32(128);
const uint32 l_65 = l_13 + l_64;
const uint32 l_66 = l_65 + l_mut_55;
const __half l_67 = buffer_0[l_66];
const float l_68 = float(l_67);
const float l_69 = l_63 * l_68;
l_mut_40 = l_mut_40 + l_69;
}
l_mut_6 = uint32(1);
while (true) {
const bool l_70 = l_mut_6 < uint32(4);
const bool l_71 = !l_70;
if (l_71) {
break;}
const float l_72 = { __shfl_xor_sync(-1, l_mut_40, l_mut_6) };
l_mut_40 = l_mut_40 + l_72;
l_mut_6 = l_mut_6 * uint32(2);
}
const bool l_73 = l_4 == uint32(0);
const bool l_74 = l_5 && l_73;
if (l_74) {
const uint32 l_75 = l_mut_28 * uint32(128);
const uint32 l_76 = l_16 + l_75;
const __half l_77 = __half(l_mut_40);
buffer_6[l_76] = l_77;
}
}
if (l_5) {

for (uint32 l_mut_78 = uint32(0); l_mut_78 < uint32(32); ++l_mut_78) {
const uint32 l_79 = l_4 * uint32(32);
const uint32 l_80 = l_79 + l_mut_78;
const uint32 l_81 = l_80 * uint32(128);
const uint32 l_82 = l_19 + l_81;
const uint32 l_83 = l_82 + l_9;
const float l_84 = l_arr_20[l_mut_78];
buffer_5[l_83] = l_84;
}
}

}
