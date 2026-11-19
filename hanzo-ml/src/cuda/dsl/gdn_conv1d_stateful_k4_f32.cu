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
    uint32 static_meta[12];
};


extern "C" __global__ void __launch_bounds__(64)gdn_conv1d_stateful_f_f32 (
    const float* __restrict__ buffer_0, const float* __restrict__ buffer_1, const float* __restrict__ buffer_2, float* buffer_3, float* buffer_4, const uint32* __restrict__ buffer_5, const __grid_constant__ info_st info
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
float l_mut_13;
float l_mut_14;
const uint32 l_0 = buffer_5[uint32(0)];
const uint32 l_1 = buffer_5[uint32(1)];
const uint32 l_2 = buffer_5[uint32(2)];
const uint32 l_3 = buffer_5[uint32(3)];
const uint32 l_4 = l_0 * l_1;
const uint32 l_5 = l_4 * l_2;
const bool l_6 = idxGlobal < l_5;
if (l_6) {
const uint32 l_7 = idxGlobal % l_2;
const uint32 l_8 = idxGlobal / l_2;
const uint32 l_9 = l_8 % l_1;
const uint32 l_10 = l_8 * l_2;
const uint32 l_11 = l_8 * uint32(4);
const uint32 l_12 = l_9 * uint32(4);
l_mut_13 = float(0.0);
l_mut_14 = float(0.0);
const uint32 l_15 = l_7 + uint32(0);
const bool l_16 = l_15 >= uint32(3);
if (l_16) {
const uint32 l_17 = l_10 + l_7;
const uint32 l_18 = l_17 + uint32(0);
const uint32 l_19 = l_18 - uint32(3);
const float l_20 = buffer_0[l_19];
l_mut_14 = l_20;
} else {
const bool l_21 = l_3 != uint32(0);
if (l_21) {
const uint32 l_22 = l_11 + uint32(1);
const uint32 l_23 = l_22 + l_7;
const uint32 l_24 = l_23 + uint32(0);
const float l_25 = buffer_2[l_24];
l_mut_14 = l_25;
}
}
const uint32 l_26 = l_12 + uint32(0);
const float l_27 = buffer_1[l_26];
const float l_28 = l_mut_14 * l_27;
l_mut_13 = l_mut_13 + l_28;
l_mut_14 = float(0.0);
const uint32 l_29 = l_7 + uint32(1);
const bool l_30 = l_29 >= uint32(3);
if (l_30) {
const uint32 l_31 = l_10 + l_7;
const uint32 l_32 = l_31 + uint32(1);
const uint32 l_33 = l_32 - uint32(3);
const float l_34 = buffer_0[l_33];
l_mut_14 = l_34;
} else {
const bool l_35 = l_3 != uint32(0);
if (l_35) {
const uint32 l_36 = l_11 + uint32(1);
const uint32 l_37 = l_36 + l_7;
const uint32 l_38 = l_37 + uint32(1);
const float l_39 = buffer_2[l_38];
l_mut_14 = l_39;
}
}
const uint32 l_40 = l_12 + uint32(1);
const float l_41 = buffer_1[l_40];
const float l_42 = l_mut_14 * l_41;
l_mut_13 = l_mut_13 + l_42;
l_mut_14 = float(0.0);
const uint32 l_43 = l_7 + uint32(2);
const bool l_44 = l_43 >= uint32(3);
if (l_44) {
const uint32 l_45 = l_10 + l_7;
const uint32 l_46 = l_45 + uint32(2);
const uint32 l_47 = l_46 - uint32(3);
const float l_48 = buffer_0[l_47];
l_mut_14 = l_48;
} else {
const bool l_49 = l_3 != uint32(0);
if (l_49) {
const uint32 l_50 = l_11 + uint32(1);
const uint32 l_51 = l_50 + l_7;
const uint32 l_52 = l_51 + uint32(2);
const float l_53 = buffer_2[l_52];
l_mut_14 = l_53;
}
}
const uint32 l_54 = l_12 + uint32(2);
const float l_55 = buffer_1[l_54];
const float l_56 = l_mut_14 * l_55;
l_mut_13 = l_mut_13 + l_56;
l_mut_14 = float(0.0);
const uint32 l_57 = l_7 + uint32(3);
const bool l_58 = l_57 >= uint32(3);
if (l_58) {
const uint32 l_59 = l_10 + l_7;
const uint32 l_60 = l_59 + uint32(3);
const uint32 l_61 = l_60 - uint32(3);
const float l_62 = buffer_0[l_61];
l_mut_14 = l_62;
} else {
const bool l_63 = l_3 != uint32(0);
if (l_63) {
const uint32 l_64 = l_11 + uint32(1);
const uint32 l_65 = l_64 + l_7;
const uint32 l_66 = l_65 + uint32(3);
const float l_67 = buffer_2[l_66];
l_mut_14 = l_67;
}
}
const uint32 l_68 = l_12 + uint32(3);
const float l_69 = buffer_1[l_68];
const float l_70 = l_mut_14 * l_69;
l_mut_13 = l_mut_13 + l_70;
const uint32 l_71 = l_10 + l_7;
const float l_72 = -l_mut_13;
const float l_73 = exp(l_72);
const float l_74 = float(1.0) + l_73;
const float l_75 = l_mut_13 / l_74;
buffer_4[l_71] = l_75;
const uint32 l_76 = l_7 + uint32(1);
const bool l_77 = l_76 == l_2;
if (l_77) {
const uint32 l_78 = l_2 + uint32(0);
const bool l_79 = l_78 >= uint32(4);
if (l_79) {
const uint32 l_80 = l_10 + l_2;
const uint32 l_81 = l_80 + uint32(0);
const uint32 l_82 = l_81 - uint32(4);
const float l_83 = buffer_0[l_82];
l_mut_14 = l_83;
} else {
const bool l_84 = l_3 != uint32(0);
if (l_84) {
const uint32 l_85 = l_11 + l_2;
const uint32 l_86 = l_85 + uint32(0);
const float l_87 = buffer_2[l_86];
l_mut_13 = l_87;
} else {
l_mut_13 = float(0.0);
}
l_mut_14 = l_mut_13;
}
const uint32 l_88 = l_11 + uint32(0);
buffer_3[l_88] = l_mut_14;
const uint32 l_89 = l_2 + uint32(1);
const bool l_90 = l_89 >= uint32(4);
if (l_90) {
const uint32 l_91 = l_10 + l_2;
const uint32 l_92 = l_91 + uint32(1);
const uint32 l_93 = l_92 - uint32(4);
const float l_94 = buffer_0[l_93];
l_mut_14 = l_94;
} else {
const bool l_95 = l_3 != uint32(0);
if (l_95) {
const uint32 l_96 = l_11 + l_2;
const uint32 l_97 = l_96 + uint32(1);
const float l_98 = buffer_2[l_97];
l_mut_13 = l_98;
} else {
l_mut_13 = float(0.0);
}
l_mut_14 = l_mut_13;
}
const uint32 l_99 = l_11 + uint32(1);
buffer_3[l_99] = l_mut_14;
const uint32 l_100 = l_2 + uint32(2);
const bool l_101 = l_100 >= uint32(4);
if (l_101) {
const uint32 l_102 = l_10 + l_2;
const uint32 l_103 = l_102 + uint32(2);
const uint32 l_104 = l_103 - uint32(4);
const float l_105 = buffer_0[l_104];
l_mut_14 = l_105;
} else {
const bool l_106 = l_3 != uint32(0);
if (l_106) {
const uint32 l_107 = l_11 + l_2;
const uint32 l_108 = l_107 + uint32(2);
const float l_109 = buffer_2[l_108];
l_mut_13 = l_109;
} else {
l_mut_13 = float(0.0);
}
l_mut_14 = l_mut_13;
}
const uint32 l_110 = l_11 + uint32(2);
buffer_3[l_110] = l_mut_14;
const uint32 l_111 = l_2 + uint32(3);
const bool l_112 = l_111 >= uint32(4);
if (l_112) {
const uint32 l_113 = l_10 + l_2;
const uint32 l_114 = l_113 + uint32(3);
const uint32 l_115 = l_114 - uint32(4);
const float l_116 = buffer_0[l_115];
l_mut_14 = l_116;
} else {
const bool l_117 = l_3 != uint32(0);
if (l_117) {
const uint32 l_118 = l_11 + l_2;
const uint32 l_119 = l_118 + uint32(3);
const float l_120 = buffer_2[l_119];
l_mut_13 = l_120;
} else {
l_mut_13 = float(0.0);
}
l_mut_14 = l_mut_13;
}
const uint32 l_121 = l_11 + uint32(3);
buffer_3[l_121] = l_mut_14;
}
}

}
