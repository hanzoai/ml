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
    uint32 static_meta[12];
};


extern "C" __global__ void __launch_bounds__(64)gdn_conv1d_stateful_f_ (
    const __nv_bfloat16* __restrict__ buffer_0, const __nv_bfloat16* __restrict__ buffer_1, const __nv_bfloat16* __restrict__ buffer_2, __nv_bfloat16* buffer_3, __nv_bfloat16* buffer_4, const uint32* __restrict__ buffer_5, const __grid_constant__ info_st info
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
__nv_bfloat16 l_mut_97;
__nv_bfloat16 l_mut_102;
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
const __nv_bfloat16 l_20 = buffer_0[l_19];
const float l_21 = float(l_20);
l_mut_14 = l_21;
} else {
const bool l_22 = l_3 != uint32(0);
if (l_22) {
const uint32 l_23 = l_11 + uint32(1);
const uint32 l_24 = l_23 + l_7;
const uint32 l_25 = l_24 + uint32(0);
const __nv_bfloat16 l_26 = buffer_2[l_25];
const float l_27 = float(l_26);
l_mut_14 = l_27;
}
}
const uint32 l_28 = l_12 + uint32(0);
const __nv_bfloat16 l_29 = buffer_1[l_28];
const float l_30 = float(l_29);
const float l_31 = l_mut_14 * l_30;
l_mut_13 = l_mut_13 + l_31;
l_mut_14 = float(0.0);
const uint32 l_32 = l_7 + uint32(1);
const bool l_33 = l_32 >= uint32(3);
if (l_33) {
const uint32 l_34 = l_10 + l_7;
const uint32 l_35 = l_34 + uint32(1);
const uint32 l_36 = l_35 - uint32(3);
const __nv_bfloat16 l_37 = buffer_0[l_36];
const float l_38 = float(l_37);
l_mut_14 = l_38;
} else {
const bool l_39 = l_3 != uint32(0);
if (l_39) {
const uint32 l_40 = l_11 + uint32(1);
const uint32 l_41 = l_40 + l_7;
const uint32 l_42 = l_41 + uint32(1);
const __nv_bfloat16 l_43 = buffer_2[l_42];
const float l_44 = float(l_43);
l_mut_14 = l_44;
}
}
const uint32 l_45 = l_12 + uint32(1);
const __nv_bfloat16 l_46 = buffer_1[l_45];
const float l_47 = float(l_46);
const float l_48 = l_mut_14 * l_47;
l_mut_13 = l_mut_13 + l_48;
l_mut_14 = float(0.0);
const uint32 l_49 = l_7 + uint32(2);
const bool l_50 = l_49 >= uint32(3);
if (l_50) {
const uint32 l_51 = l_10 + l_7;
const uint32 l_52 = l_51 + uint32(2);
const uint32 l_53 = l_52 - uint32(3);
const __nv_bfloat16 l_54 = buffer_0[l_53];
const float l_55 = float(l_54);
l_mut_14 = l_55;
} else {
const bool l_56 = l_3 != uint32(0);
if (l_56) {
const uint32 l_57 = l_11 + uint32(1);
const uint32 l_58 = l_57 + l_7;
const uint32 l_59 = l_58 + uint32(2);
const __nv_bfloat16 l_60 = buffer_2[l_59];
const float l_61 = float(l_60);
l_mut_14 = l_61;
}
}
const uint32 l_62 = l_12 + uint32(2);
const __nv_bfloat16 l_63 = buffer_1[l_62];
const float l_64 = float(l_63);
const float l_65 = l_mut_14 * l_64;
l_mut_13 = l_mut_13 + l_65;
l_mut_14 = float(0.0);
const uint32 l_66 = l_7 + uint32(3);
const bool l_67 = l_66 >= uint32(3);
if (l_67) {
const uint32 l_68 = l_10 + l_7;
const uint32 l_69 = l_68 + uint32(3);
const uint32 l_70 = l_69 - uint32(3);
const __nv_bfloat16 l_71 = buffer_0[l_70];
const float l_72 = float(l_71);
l_mut_14 = l_72;
} else {
const bool l_73 = l_3 != uint32(0);
if (l_73) {
const uint32 l_74 = l_11 + uint32(1);
const uint32 l_75 = l_74 + l_7;
const uint32 l_76 = l_75 + uint32(3);
const __nv_bfloat16 l_77 = buffer_2[l_76];
const float l_78 = float(l_77);
l_mut_14 = l_78;
}
}
const uint32 l_79 = l_12 + uint32(3);
const __nv_bfloat16 l_80 = buffer_1[l_79];
const float l_81 = float(l_80);
const float l_82 = l_mut_14 * l_81;
l_mut_13 = l_mut_13 + l_82;
const uint32 l_83 = l_10 + l_7;
const float l_84 = -l_mut_13;
const float l_85 = exp(l_84);
const float l_86 = float(1.0) + l_85;
const float l_87 = l_mut_13 / l_86;
const __nv_bfloat16 l_88 = __nv_bfloat16(l_87);
buffer_4[l_83] = l_88;
const uint32 l_89 = l_7 + uint32(1);
const bool l_90 = l_89 == l_2;
if (l_90) {
const uint32 l_91 = l_2 + uint32(0);
const bool l_92 = l_91 >= uint32(4);
if (l_92) {
const uint32 l_93 = l_10 + l_2;
const uint32 l_94 = l_93 + uint32(0);
const uint32 l_95 = l_94 - uint32(4);
const __nv_bfloat16 l_96 = buffer_0[l_95];
l_mut_97 = l_96;
} else {
const bool l_98 = l_3 != uint32(0);
if (l_98) {
const uint32 l_99 = l_11 + l_2;
const uint32 l_100 = l_99 + uint32(0);
const __nv_bfloat16 l_101 = buffer_2[l_100];
l_mut_102 = l_101;
} else {
l_mut_102 = __nv_bfloat16(0.0);
}
l_mut_97 = l_mut_102;
}
const uint32 l_103 = l_11 + uint32(0);
buffer_3[l_103] = l_mut_97;
const uint32 l_104 = l_2 + uint32(1);
const bool l_105 = l_104 >= uint32(4);
if (l_105) {
const uint32 l_106 = l_10 + l_2;
const uint32 l_107 = l_106 + uint32(1);
const uint32 l_108 = l_107 - uint32(4);
const __nv_bfloat16 l_109 = buffer_0[l_108];
l_mut_102 = l_109;
} else {
const bool l_110 = l_3 != uint32(0);
if (l_110) {
const uint32 l_111 = l_11 + l_2;
const uint32 l_112 = l_111 + uint32(1);
const __nv_bfloat16 l_113 = buffer_2[l_112];
l_mut_97 = l_113;
} else {
l_mut_97 = __nv_bfloat16(0.0);
}
l_mut_102 = l_mut_97;
}
const uint32 l_114 = l_11 + uint32(1);
buffer_3[l_114] = l_mut_102;
const uint32 l_115 = l_2 + uint32(2);
const bool l_116 = l_115 >= uint32(4);
if (l_116) {
const uint32 l_117 = l_10 + l_2;
const uint32 l_118 = l_117 + uint32(2);
const uint32 l_119 = l_118 - uint32(4);
const __nv_bfloat16 l_120 = buffer_0[l_119];
l_mut_102 = l_120;
} else {
const bool l_121 = l_3 != uint32(0);
if (l_121) {
const uint32 l_122 = l_11 + l_2;
const uint32 l_123 = l_122 + uint32(2);
const __nv_bfloat16 l_124 = buffer_2[l_123];
l_mut_97 = l_124;
} else {
l_mut_97 = __nv_bfloat16(0.0);
}
l_mut_102 = l_mut_97;
}
const uint32 l_125 = l_11 + uint32(2);
buffer_3[l_125] = l_mut_102;
const uint32 l_126 = l_2 + uint32(3);
const bool l_127 = l_126 >= uint32(4);
if (l_127) {
const uint32 l_128 = l_10 + l_2;
const uint32 l_129 = l_128 + uint32(3);
const uint32 l_130 = l_129 - uint32(4);
const __nv_bfloat16 l_131 = buffer_0[l_130];
l_mut_102 = l_131;
} else {
const bool l_132 = l_3 != uint32(0);
if (l_132) {
const uint32 l_133 = l_11 + l_2;
const uint32 l_134 = l_133 + uint32(3);
const __nv_bfloat16 l_135 = buffer_2[l_134];
l_mut_97 = l_135;
} else {
l_mut_97 = __nv_bfloat16(0.0);
}
l_mut_102 = l_mut_97;
}
const uint32 l_136 = l_11 + uint32(3);
buffer_3[l_136] = l_mut_102;
}
}

}
