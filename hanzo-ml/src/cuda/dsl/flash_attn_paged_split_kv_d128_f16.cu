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
    uint32 static_meta[18];
};


extern "C" __global__ void __launch_bounds__(32)flash_attn_paged_split_kv_f_ (
    const __half* __restrict__ buffer_0, const __half* __restrict__ buffer_1, const __half* __restrict__ buffer_2, const uint32* __restrict__ buffer_3, const uint32* __restrict__ buffer_4, __half* buffer_5, float* buffer_6, const float* __restrict__ buffer_7, const uint32* __restrict__ buffer_8, const __grid_constant__ info_st info
) {
uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
uint32 blockIdxGlobal = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x;
uint32 warpSizeChecked = min(warpSize, blockDim.x * blockDim.y * blockDim.z);
float l_arr_51[4];

float l_arr_52[4];

uint32 l_mut_24;
uint32 l_mut_32;
uint32 l_mut_48;
float l_mut_59;
float l_mut_60;
float l_mut_62;
__half l_mut_74;
__half l_mut_88;
float l_mut_104;
float l_mut_105;
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
const uint32 l_50 = l_15 * uint32(128);

for (uint32 l_mut_53 = uint32(0); l_mut_53 < uint32(4); ++l_mut_53) {
const uint32 l_54 = l_mut_53 * uint32(32);
const uint32 l_55 = l_54 + threadIdxGlobal;
const uint32 l_56 = l_50 + l_55;
const __half l_57 = buffer_0[l_56];
const float l_58 = float(l_57);
l_arr_51[l_mut_53] = l_58;
l_arr_52[l_mut_53] = float(0.0);
}
l_mut_59 = float(-3.3999999521443642e38);
l_mut_60 = float(0.0);

for (uint32 l_mut_61 = l_mut_32; l_mut_61 < l_mut_48; ++l_mut_61) {
l_mut_62 = float(0.0);

for (uint32 l_mut_63 = uint32(0); l_mut_63 < uint32(4); ++l_mut_63) {
const uint32 l_64 = l_mut_63 * uint32(32);
const uint32 l_65 = l_64 + threadIdxGlobal;
const bool l_66 = l_10 == uint32(2);
if (l_66) {
const uint32 l_67 = l_20 * l_6;
const uint32 l_68 = l_67 + l_22;
const uint32 l_69 = l_68 * l_8;
const uint32 l_70 = l_69 + l_mut_61;
const uint32 l_71 = l_70 * uint32(128);
const uint32 l_72 = l_71 + l_65;
const __half l_73 = buffer_1[l_72];
l_mut_74 = l_73;
} else {
const uint32 l_75 = l_20 * l_9;
const uint32 l_76 = l_mut_61 / l_8;
const uint32 l_77 = l_75 + l_76;
const uint32 l_78 = buffer_3[l_77];
const uint32 l_79 = l_mut_61 % l_8;
const bool l_80 = l_10 == uint32(0);
if (l_80) {
const uint32 l_81 = l_78 * l_6;
const uint32 l_82 = l_81 + l_22;
const uint32 l_83 = l_82 * l_8;
const uint32 l_84 = l_83 + l_79;
const uint32 l_85 = l_84 * uint32(128);
const uint32 l_86 = l_85 + l_65;
const __half l_87 = buffer_1[l_86];
l_mut_88 = l_87;
} else {
const uint32 l_89 = uint32(128) / l_11;
const uint32 l_90 = l_78 * l_6;
const uint32 l_91 = l_90 + l_22;
const uint32 l_92 = l_91 * l_89;
const uint32 l_93 = l_65 / l_11;
const uint32 l_94 = l_92 + l_93;
const uint32 l_95 = l_94 * l_8;
const uint32 l_96 = l_95 + l_79;
const uint32 l_97 = l_96 * l_11;
const uint32 l_98 = l_65 % l_11;
const uint32 l_99 = l_97 + l_98;
const __half l_100 = buffer_1[l_99];
l_mut_88 = l_100;
}
l_mut_74 = l_mut_88;
}
const float l_101 = float(l_mut_74);
const float l_102 = l_arr_51[l_mut_63];
const float l_103 = l_102 * l_101;
l_mut_62 = l_mut_62 + l_103;
}
l_mut_104 = float(0.0);
l_mut_105 = l_mut_62;
l_mut_24 = uint32(1);
while (true) {
const bool l_106 = l_mut_24 < uint32(32);
const bool l_107 = !l_106;
if (l_107) {
break;}
const float l_108 = { __shfl_xor_sync(-1, l_mut_105, l_mut_24) };
l_mut_105 = l_mut_105 + l_108;
l_mut_24 = l_mut_24 * uint32(2);
}
const float l_109 = l_mut_105 * l_2;
l_mut_104 = l_109;
const bool l_110 = l_3 > float(0.0);
if (l_110) {
const float l_111 = l_mut_104 / l_3;
const float l_112 = tanh(l_111);
const float l_113 = l_3 * l_112;
l_mut_104 = l_113;
}
l_mut_105 = l_mut_59;
const bool l_114 = l_mut_104 > l_mut_105;
if (l_114) {
l_mut_105 = l_mut_104;
}
const float l_115 = l_mut_59 - l_mut_105;
const float l_116 = exp(l_115);
const float l_117 = l_mut_104 - l_mut_105;
const float l_118 = exp(l_117);
const float l_119 = l_mut_60 * l_116;
const float l_120 = l_119 + l_118;
l_mut_60 = l_120;
l_mut_59 = l_mut_105;

for (uint32 l_mut_121 = uint32(0); l_mut_121 < uint32(4); ++l_mut_121) {
const uint32 l_122 = l_mut_121 * uint32(32);
const uint32 l_123 = l_122 + threadIdxGlobal;
const bool l_124 = l_10 == uint32(2);
if (l_124) {
const uint32 l_125 = l_20 * l_6;
const uint32 l_126 = l_125 + l_22;
const uint32 l_127 = l_126 * l_8;
const uint32 l_128 = l_127 + l_mut_61;
const uint32 l_129 = l_128 * uint32(128);
const uint32 l_130 = l_129 + l_123;
const __half l_131 = buffer_2[l_130];
l_mut_88 = l_131;
} else {
const uint32 l_132 = l_20 * l_9;
const uint32 l_133 = l_mut_61 / l_8;
const uint32 l_134 = l_132 + l_133;
const uint32 l_135 = buffer_3[l_134];
const uint32 l_136 = l_mut_61 % l_8;
const bool l_137 = l_10 == uint32(0);
if (l_137) {
const uint32 l_138 = l_135 * l_6;
const uint32 l_139 = l_138 + l_22;
const uint32 l_140 = l_139 * l_8;
const uint32 l_141 = l_140 + l_136;
const uint32 l_142 = l_141 * uint32(128);
const uint32 l_143 = l_142 + l_123;
const __half l_144 = buffer_2[l_143];
l_mut_74 = l_144;
} else {
const uint32 l_145 = l_135 * l_6;
const uint32 l_146 = l_145 + l_22;
const uint32 l_147 = l_146 * uint32(128);
const uint32 l_148 = l_147 + l_123;
const uint32 l_149 = l_148 * l_8;
const uint32 l_150 = l_149 + l_136;
const __half l_151 = buffer_2[l_150];
l_mut_74 = l_151;
}
l_mut_88 = l_mut_74;
}
const float l_152 = float(l_mut_88);
const float l_153 = l_arr_52[l_mut_121];
const float l_154 = l_153 * l_116;
const float l_155 = l_118 * l_152;
const float l_156 = l_154 + l_155;
l_arr_52[l_mut_121] = l_156;
}
}
const bool l_157 = l_14 == uint32(1);
if (l_157) {
l_mut_105 = float(0.0);
const bool l_158 = l_mut_60 > float(0.0);
if (l_158) {
const float l_159 = float(1.0) / l_mut_60;
l_mut_105 = l_159;
}

for (uint32 l_mut_160 = uint32(0); l_mut_160 < uint32(4); ++l_mut_160) {
const uint32 l_161 = l_mut_160 * uint32(32);
const uint32 l_162 = l_161 + threadIdxGlobal;
const uint32 l_163 = l_50 + l_162;
const float l_164 = l_arr_52[l_mut_160];
const float l_165 = l_164 * l_mut_105;
const __half l_166 = __half(l_165);
buffer_5[l_163] = l_166;
}
} else {
const uint32 l_167 = l_15 * l_14;
const uint32 l_168 = l_167 + l_16;
const uint32 l_169 = l_168 * uint32(130);

for (uint32 l_mut_170 = uint32(0); l_mut_170 < uint32(4); ++l_mut_170) {
const uint32 l_171 = l_mut_170 * uint32(32);
const uint32 l_172 = l_171 + threadIdxGlobal;
const uint32 l_173 = l_169 + l_172;
const float l_174 = l_arr_52[l_mut_170];
buffer_6[l_173] = l_174;
}
const bool l_175 = threadIdxGlobal == uint32(0);
if (l_175) {
const uint32 l_176 = l_169 + uint32(128);
buffer_6[l_176] = l_mut_59;
const uint32 l_177 = l_169 + uint32(128);
const uint32 l_178 = l_177 + uint32(1);
buffer_6[l_178] = l_mut_60;
}
}

}
