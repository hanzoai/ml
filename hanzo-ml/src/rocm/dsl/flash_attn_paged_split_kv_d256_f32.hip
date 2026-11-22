#include <hip/hip_runtime.h>
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
    uint32 static_meta[18];
};


extern "C" __global__ void __launch_bounds__(32)flash_attn_paged_split_kv_f_f32 (
    const float* __restrict__ buffer_0, const float* __restrict__ buffer_1, const float* __restrict__ buffer_2, const uint32* __restrict__ buffer_3, const uint32* __restrict__ buffer_4, float* buffer_5, float* buffer_6, const float* __restrict__ buffer_7, const uint32* __restrict__ buffer_8, const info_st *__restrict__ info_ptr
) {
uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
uint32 blockIdxGlobal = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x;
uint32 warpSizeChecked = min(warpSize, blockDim.x * blockDim.y * blockDim.z);
const info_st &info = *info_ptr;
const uint32 *dynamic_meta = reinterpret_cast<const uint32 *>(
    reinterpret_cast<const char *>(info_ptr) + sizeof(info_st));

float l_arr_51[8];

float l_arr_52[8];

uint32 l_mut_24;
uint32 l_mut_32;
uint32 l_mut_48;
float l_mut_58;
float l_mut_59;
float l_mut_61;
float l_mut_73;
float l_mut_87;
const uint32 l_0 = buffer_8[uint32(10)];
const uint32 l_1 = blockIdxGlobal + l_0;
const float l_2 = buffer_7[uint32(0)];
const float l_3 = buffer_7[uint32(1)];
const uint32 l_4 = buffer_8[uint32(0)];
const uint32 l_5 = buffer_8[uint32(1)];
const uint32 l_6 = buffer_8[uint32(2)];
const uint32 l_7 = buffer_8[uint32(3)];
const uint32 l_8 = buffer_8[uint32(4)];
const uint32 l_9 = buffer_8[uint32(5)];
const uint32 l_10 = buffer_8[uint32(6)];
const uint32 l_11 = buffer_8[uint32(7)];
const uint32 l_12 = buffer_8[uint32(8)];
const uint32 l_13 = buffer_8[uint32(9)];
const uint32 l_14 = buffer_8[uint32(11)];
const uint32 l_15 = l_1 / l_14;
const uint32 l_16 = l_1 % l_14;
const uint32 l_17 = l_15 % l_4;
const uint32 l_18 = l_15 / l_4;
const uint32 l_19 = l_18 % l_5;
const uint32 l_20 = l_18 / l_5;
const uint32 l_21 = l_5 / l_6;
const uint32 l_22 = l_19 / l_21;
const uint32 l_23 = buffer_4[l_20];
l_mut_24 = uint32(0);
const bool l_25 = l_7 == uint32(2);
if (l_25) {
l_mut_24 = l_13;
} else {
const bool l_26 = l_23 >= l_4;
if (l_26) {
const uint32 l_27 = l_23 - l_4;
l_mut_24 = l_27;
}
}
const uint32 l_28 = l_mut_24 + l_17;
l_mut_24 = l_23;
const bool l_29 = l_7 != uint32(0);
if (l_29) {
const uint32 l_30 = l_28 + uint32(1);
const bool l_31 = l_30 < l_mut_24;
if (l_31) {
l_mut_24 = l_30;
}
}
l_mut_32 = uint32(0);
const bool l_33 = l_12 > uint32(0);
const uint32 l_34 = l_28 + uint32(1);
const bool l_35 = l_34 > l_12;
const bool l_36 = l_33 && l_35;
if (l_36) {
const uint32 l_37 = l_28 + uint32(1);
const uint32 l_38 = l_37 - l_12;
l_mut_32 = l_38;
}
const bool l_39 = l_mut_32 > l_mut_24;
if (l_39) {
l_mut_32 = l_mut_24;
}
const uint32 l_40 = l_mut_24 - l_mut_32;
const uint32 l_41 = l_40 + l_14;
const uint32 l_42 = l_41 - uint32(1);
const uint32 l_43 = l_42 / l_14;
const uint32 l_44 = l_16 * l_43;
const uint32 l_45 = l_mut_32 + l_44;
l_mut_32 = l_45;
const bool l_46 = l_mut_32 > l_mut_24;
if (l_46) {
l_mut_32 = l_mut_24;
}
const uint32 l_47 = l_mut_32 + l_43;
l_mut_48 = l_47;
const bool l_49 = l_mut_48 > l_mut_24;
if (l_49) {
l_mut_48 = l_mut_24;
}
const uint32 l_50 = l_15 * uint32(256);

for (uint32 l_mut_53 = uint32(0); l_mut_53 < uint32(8); ++l_mut_53) {
const uint32 l_54 = l_mut_53 * uint32(32);
const uint32 l_55 = l_54 + threadIdxGlobal;
const uint32 l_56 = l_50 + l_55;
const float l_57 = buffer_0[l_56];
l_arr_51[l_mut_53] = l_57;
l_arr_52[l_mut_53] = float(0.0);
}
l_mut_58 = float(-3.3999999521443642e38);
l_mut_59 = float(0.0);

for (uint32 l_mut_60 = l_mut_32; l_mut_60 < l_mut_48; ++l_mut_60) {
l_mut_61 = float(0.0);

for (uint32 l_mut_62 = uint32(0); l_mut_62 < uint32(8); ++l_mut_62) {
const uint32 l_63 = l_mut_62 * uint32(32);
const uint32 l_64 = l_63 + threadIdxGlobal;
const bool l_65 = l_10 == uint32(2);
if (l_65) {
const uint32 l_66 = l_20 * l_6;
const uint32 l_67 = l_66 + l_22;
const uint32 l_68 = l_67 * l_8;
const uint32 l_69 = l_68 + l_mut_60;
const uint32 l_70 = l_69 * uint32(256);
const uint32 l_71 = l_70 + l_64;
const float l_72 = buffer_1[l_71];
l_mut_73 = l_72;
} else {
const uint32 l_74 = l_20 * l_9;
const uint32 l_75 = l_mut_60 / l_8;
const uint32 l_76 = l_74 + l_75;
const uint32 l_77 = buffer_3[l_76];
const uint32 l_78 = l_mut_60 % l_8;
const bool l_79 = l_10 == uint32(0);
if (l_79) {
const uint32 l_80 = l_77 * l_6;
const uint32 l_81 = l_80 + l_22;
const uint32 l_82 = l_81 * l_8;
const uint32 l_83 = l_82 + l_78;
const uint32 l_84 = l_83 * uint32(256);
const uint32 l_85 = l_84 + l_64;
const float l_86 = buffer_1[l_85];
l_mut_87 = l_86;
} else {
const uint32 l_88 = uint32(256) / l_11;
const uint32 l_89 = l_77 * l_6;
const uint32 l_90 = l_89 + l_22;
const uint32 l_91 = l_90 * l_88;
const uint32 l_92 = l_64 / l_11;
const uint32 l_93 = l_91 + l_92;
const uint32 l_94 = l_93 * l_8;
const uint32 l_95 = l_94 + l_78;
const uint32 l_96 = l_95 * l_11;
const uint32 l_97 = l_64 % l_11;
const uint32 l_98 = l_96 + l_97;
const float l_99 = buffer_1[l_98];
l_mut_87 = l_99;
}
l_mut_73 = l_mut_87;
}
const float l_100 = l_arr_51[l_mut_62];
const float l_101 = l_100 * l_mut_73;
l_mut_61 = l_mut_61 + l_101;
}
l_mut_87 = float(0.0);
l_mut_73 = l_mut_61;
l_mut_24 = uint32(1);
while (true) {
const bool l_102 = l_mut_24 < uint32(32);
const bool l_103 = !l_102;
if (l_103) {
break;}
const float l_104 = { __shfl_xor(l_mut_73, l_mut_24) };
l_mut_73 = l_mut_73 + l_104;
l_mut_24 = l_mut_24 * uint32(2);
}
const float l_105 = l_mut_73 * l_2;
l_mut_87 = l_105;
const bool l_106 = l_3 > float(0.0);
if (l_106) {
const float l_107 = l_mut_87 / l_3;
const float l_108 = tanh(l_107);
const float l_109 = l_3 * l_108;
l_mut_87 = l_109;
}
l_mut_73 = l_mut_58;
const bool l_110 = l_mut_87 > l_mut_73;
if (l_110) {
l_mut_73 = l_mut_87;
}
const float l_111 = l_mut_58 - l_mut_73;
const float l_112 = exp(l_111);
const float l_113 = l_mut_87 - l_mut_73;
const float l_114 = exp(l_113);
const float l_115 = l_mut_59 * l_112;
const float l_116 = l_115 + l_114;
l_mut_59 = l_116;
l_mut_58 = l_mut_73;

for (uint32 l_mut_117 = uint32(0); l_mut_117 < uint32(8); ++l_mut_117) {
const uint32 l_118 = l_mut_117 * uint32(32);
const uint32 l_119 = l_118 + threadIdxGlobal;
const bool l_120 = l_10 == uint32(2);
if (l_120) {
const uint32 l_121 = l_20 * l_6;
const uint32 l_122 = l_121 + l_22;
const uint32 l_123 = l_122 * l_8;
const uint32 l_124 = l_123 + l_mut_60;
const uint32 l_125 = l_124 * uint32(256);
const uint32 l_126 = l_125 + l_119;
const float l_127 = buffer_2[l_126];
l_mut_87 = l_127;
} else {
const uint32 l_128 = l_20 * l_9;
const uint32 l_129 = l_mut_60 / l_8;
const uint32 l_130 = l_128 + l_129;
const uint32 l_131 = buffer_3[l_130];
const uint32 l_132 = l_mut_60 % l_8;
const bool l_133 = l_10 == uint32(0);
if (l_133) {
const uint32 l_134 = l_131 * l_6;
const uint32 l_135 = l_134 + l_22;
const uint32 l_136 = l_135 * l_8;
const uint32 l_137 = l_136 + l_132;
const uint32 l_138 = l_137 * uint32(256);
const uint32 l_139 = l_138 + l_119;
const float l_140 = buffer_2[l_139];
l_mut_73 = l_140;
} else {
const uint32 l_141 = l_131 * l_6;
const uint32 l_142 = l_141 + l_22;
const uint32 l_143 = l_142 * uint32(256);
const uint32 l_144 = l_143 + l_119;
const uint32 l_145 = l_144 * l_8;
const uint32 l_146 = l_145 + l_132;
const float l_147 = buffer_2[l_146];
l_mut_73 = l_147;
}
l_mut_87 = l_mut_73;
}
const float l_148 = l_arr_52[l_mut_117];
const float l_149 = l_148 * l_112;
const float l_150 = l_114 * l_mut_87;
const float l_151 = l_149 + l_150;
l_arr_52[l_mut_117] = l_151;
}
}
const bool l_152 = l_14 == uint32(1);
if (l_152) {
l_mut_87 = float(0.0);
const bool l_153 = l_mut_59 > float(0.0);
if (l_153) {
const float l_154 = float(1.0) / l_mut_59;
l_mut_87 = l_154;
}

for (uint32 l_mut_155 = uint32(0); l_mut_155 < uint32(8); ++l_mut_155) {
const uint32 l_156 = l_mut_155 * uint32(32);
const uint32 l_157 = l_156 + threadIdxGlobal;
const uint32 l_158 = l_50 + l_157;
const float l_159 = l_arr_52[l_mut_155];
const float l_160 = l_159 * l_mut_87;
buffer_5[l_158] = l_160;
}
} else {
const uint32 l_161 = l_15 * l_14;
const uint32 l_162 = l_161 + l_16;
const uint32 l_163 = l_162 * uint32(258);

for (uint32 l_mut_164 = uint32(0); l_mut_164 < uint32(8); ++l_mut_164) {
const uint32 l_165 = l_mut_164 * uint32(32);
const uint32 l_166 = l_165 + threadIdxGlobal;
const uint32 l_167 = l_163 + l_166;
const float l_168 = l_arr_52[l_mut_164];
buffer_6[l_167] = l_168;
}
const bool l_169 = threadIdxGlobal == uint32(0);
if (l_169) {
const uint32 l_170 = l_163 + uint32(256);
buffer_6[l_170] = l_mut_58;
const uint32 l_171 = l_163 + uint32(256);
const uint32 l_172 = l_171 + uint32(1);
buffer_6[l_172] = l_mut_59;
}
}

}
