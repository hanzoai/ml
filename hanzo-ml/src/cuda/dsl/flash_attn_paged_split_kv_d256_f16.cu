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
float l_arr_55[8];

float l_arr_56[8];

uint32 l_mut_28;
uint32 l_mut_36;
uint32 l_mut_52;
float l_mut_63;
float l_mut_64;
float l_mut_73;
float l_mut_82;
float l_mut_83;
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
const uint32 l_15 = l_5 * l_4;
const uint32 l_16 = l_15 * l_14;
const uint32 l_17 = l_1 / l_16;
const uint32 l_18 = l_1 % l_16;
const uint32 l_19 = l_18 / l_15;
const uint32 l_20 = l_18 % l_15;
const uint32 l_21 = l_17 * l_15;
const uint32 l_22 = l_21 + l_20;
const uint32 l_23 = l_20 % l_4;
const uint32 l_24 = l_20 / l_4;
const uint32 l_25 = l_5 / l_6;
const uint32 l_26 = l_24 / l_25;
const uint32 l_27 = buffer_4[l_17];
l_mut_28 = uint32(0);
const bool l_29 = l_7 == uint32(2);
if (l_29) {
l_mut_28 = l_13;
} else {
const bool l_30 = l_27 >= l_4;
if (l_30) {
const uint32 l_31 = l_27 - l_4;
l_mut_28 = l_31;
}
}
const uint32 l_32 = l_mut_28 + l_23;
l_mut_28 = l_27;
const bool l_33 = l_7 != uint32(0);
if (l_33) {
const uint32 l_34 = l_32 + uint32(1);
const bool l_35 = l_34 < l_mut_28;
if (l_35) {
l_mut_28 = l_34;
}
}
l_mut_36 = uint32(0);
const bool l_37 = l_12 > uint32(0);
const uint32 l_38 = l_32 + uint32(1);
const bool l_39 = l_38 > l_12;
const bool l_40 = l_37 && l_39;
if (l_40) {
const uint32 l_41 = l_32 + uint32(1);
const uint32 l_42 = l_41 - l_12;
l_mut_36 = l_42;
}
const bool l_43 = l_mut_36 > l_mut_28;
if (l_43) {
l_mut_36 = l_mut_28;
}
const uint32 l_44 = l_mut_28 - l_mut_36;
const uint32 l_45 = l_44 + l_14;
const uint32 l_46 = l_45 - uint32(1);
const uint32 l_47 = l_46 / l_14;
const uint32 l_48 = l_19 * l_47;
const uint32 l_49 = l_mut_36 + l_48;
l_mut_36 = l_49;
const bool l_50 = l_mut_36 > l_mut_28;
if (l_50) {
l_mut_36 = l_mut_28;
}
const uint32 l_51 = l_mut_36 + l_47;
l_mut_52 = l_51;
const bool l_53 = l_mut_52 > l_mut_28;
if (l_53) {
l_mut_52 = l_mut_28;
}
const uint32 l_54 = l_22 * uint32(256);

for (uint32 l_mut_57 = uint32(0); l_mut_57 < uint32(8); ++l_mut_57) {
const uint32 l_58 = l_mut_57 * uint32(32);
const uint32 l_59 = l_58 + threadIdxGlobal;
const uint32 l_60 = l_54 + l_59;
const __half l_61 = buffer_0[l_60];
const float l_62 = float(l_61);
l_arr_55[l_mut_57] = l_62;
l_arr_56[l_mut_57] = float(0.0);
}
l_mut_63 = float(-3.3999999521443642e38);
l_mut_64 = float(0.0);
const bool l_65 = l_10 == uint32(2);
if (l_65) {
const uint32 l_66 = l_17 * l_6;
const uint32 l_67 = l_66 + l_26;
const uint32 l_68 = l_67 * l_8;
const uint32 l_69 = l_68 * uint32(256);

for (uint32 l_mut_70 = l_mut_36; l_mut_70 < l_mut_52; ++l_mut_70) {
const uint32 l_71 = l_mut_70 * uint32(256);
const uint32 l_72 = l_69 + l_71;
l_mut_73 = float(0.0);

for (uint32 l_mut_74 = uint32(0); l_mut_74 < uint32(8); ++l_mut_74) {
const uint32 l_75 = l_mut_74 * uint32(32);
const uint32 l_76 = l_75 + threadIdxGlobal;
const float l_77 = l_arr_55[l_mut_74];
const uint32 l_78 = l_72 + l_76;
const __half l_79 = buffer_1[l_78];
const float l_80 = float(l_79);
const float l_81 = l_77 * l_80;
l_mut_73 = l_mut_73 + l_81;
}
l_mut_82 = float(0.0);
l_mut_83 = l_mut_73;
l_mut_28 = uint32(1);
while (true) {
const bool l_84 = l_mut_28 < uint32(32);
const bool l_85 = !l_84;
if (l_85) {
break;}
const float l_86 = { __shfl_xor_sync(-1, l_mut_83, l_mut_28) };
l_mut_83 = l_mut_83 + l_86;
l_mut_28 = l_mut_28 * uint32(2);
}
const float l_87 = l_mut_83 * l_2;
l_mut_82 = l_87;
const bool l_88 = l_3 > float(0.0);
if (l_88) {
const float l_89 = l_mut_82 / l_3;
const float l_90 = tanh(l_89);
const float l_91 = l_3 * l_90;
l_mut_82 = l_91;
}
l_mut_83 = l_mut_63;
const bool l_92 = l_mut_82 > l_mut_83;
if (l_92) {
l_mut_83 = l_mut_82;
}
const float l_93 = l_mut_63 - l_mut_83;
const float l_94 = exp(l_93);
const float l_95 = l_mut_82 - l_mut_83;
const float l_96 = exp(l_95);
const float l_97 = l_mut_64 * l_94;
const float l_98 = l_97 + l_96;
l_mut_64 = l_98;
l_mut_63 = l_mut_83;

for (uint32 l_mut_99 = uint32(0); l_mut_99 < uint32(8); ++l_mut_99) {
const uint32 l_100 = l_mut_99 * uint32(32);
const uint32 l_101 = l_100 + threadIdxGlobal;
const float l_102 = l_arr_56[l_mut_99];
const float l_103 = l_102 * l_94;
const uint32 l_104 = l_72 + l_101;
const __half l_105 = buffer_2[l_104];
const float l_106 = float(l_105);
const float l_107 = l_96 * l_106;
const float l_108 = l_103 + l_107;
l_arr_56[l_mut_99] = l_108;
}
}
} else {
const uint32 l_109 = l_17 * l_9;
const uint32 l_110 = uint32(256) / l_11;

for (uint32 l_mut_111 = l_mut_36; l_mut_111 < l_mut_52; ++l_mut_111) {
const uint32 l_112 = l_mut_111 / l_8;
const uint32 l_113 = l_109 + l_112;
const uint32 l_114 = buffer_3[l_113];
const uint32 l_115 = l_mut_111 % l_8;
const uint32 l_116 = l_114 * l_6;
const uint32 l_117 = l_116 + l_26;
l_mut_83 = float(0.0);
const bool l_118 = l_10 == uint32(0);
if (l_118) {
const uint32 l_119 = l_117 * l_8;
const uint32 l_120 = l_119 + l_115;
const uint32 l_121 = l_120 * uint32(256);

for (uint32 l_mut_122 = uint32(0); l_mut_122 < uint32(8); ++l_mut_122) {
const uint32 l_123 = l_mut_122 * uint32(32);
const uint32 l_124 = l_123 + threadIdxGlobal;
const float l_125 = l_arr_55[l_mut_122];
const uint32 l_126 = l_121 + l_124;
const __half l_127 = buffer_1[l_126];
const float l_128 = float(l_127);
const float l_129 = l_125 * l_128;
l_mut_83 = l_mut_83 + l_129;
}
} else {
const uint32 l_130 = l_117 * l_110;

for (uint32 l_mut_131 = uint32(0); l_mut_131 < uint32(8); ++l_mut_131) {
const uint32 l_132 = l_mut_131 * uint32(32);
const uint32 l_133 = l_132 + threadIdxGlobal;
const uint32 l_134 = l_133 / l_11;
const uint32 l_135 = l_130 + l_134;
const uint32 l_136 = l_135 * l_8;
const uint32 l_137 = l_136 + l_115;
const uint32 l_138 = l_137 * l_11;
const uint32 l_139 = l_133 % l_11;
const uint32 l_140 = l_138 + l_139;
const float l_141 = l_arr_55[l_mut_131];
const __half l_142 = buffer_1[l_140];
const float l_143 = float(l_142);
const float l_144 = l_141 * l_143;
l_mut_83 = l_mut_83 + l_144;
}
}
l_mut_82 = float(0.0);
l_mut_73 = l_mut_83;
l_mut_28 = uint32(1);
while (true) {
const bool l_145 = l_mut_28 < uint32(32);
const bool l_146 = !l_145;
if (l_146) {
break;}
const float l_147 = { __shfl_xor_sync(-1, l_mut_73, l_mut_28) };
l_mut_73 = l_mut_73 + l_147;
l_mut_28 = l_mut_28 * uint32(2);
}
const float l_148 = l_mut_73 * l_2;
l_mut_82 = l_148;
const bool l_149 = l_3 > float(0.0);
if (l_149) {
const float l_150 = l_mut_82 / l_3;
const float l_151 = tanh(l_150);
const float l_152 = l_3 * l_151;
l_mut_82 = l_152;
}
l_mut_73 = l_mut_63;
const bool l_153 = l_mut_82 > l_mut_73;
if (l_153) {
l_mut_73 = l_mut_82;
}
const float l_154 = l_mut_63 - l_mut_73;
const float l_155 = exp(l_154);
const float l_156 = l_mut_82 - l_mut_73;
const float l_157 = exp(l_156);
const float l_158 = l_mut_64 * l_155;
const float l_159 = l_158 + l_157;
l_mut_64 = l_159;
l_mut_63 = l_mut_73;
const bool l_160 = l_10 == uint32(0);
if (l_160) {
const uint32 l_161 = l_117 * l_8;
const uint32 l_162 = l_161 + l_115;
const uint32 l_163 = l_162 * uint32(256);

for (uint32 l_mut_164 = uint32(0); l_mut_164 < uint32(8); ++l_mut_164) {
const uint32 l_165 = l_mut_164 * uint32(32);
const uint32 l_166 = l_165 + threadIdxGlobal;
const float l_167 = l_arr_56[l_mut_164];
const float l_168 = l_167 * l_155;
const uint32 l_169 = l_163 + l_166;
const __half l_170 = buffer_2[l_169];
const float l_171 = float(l_170);
const float l_172 = l_157 * l_171;
const float l_173 = l_168 + l_172;
l_arr_56[l_mut_164] = l_173;
}
} else {
const uint32 l_174 = l_117 * uint32(256);

for (uint32 l_mut_175 = uint32(0); l_mut_175 < uint32(8); ++l_mut_175) {
const uint32 l_176 = l_mut_175 * uint32(32);
const uint32 l_177 = l_176 + threadIdxGlobal;
const uint32 l_178 = l_174 + l_177;
const uint32 l_179 = l_178 * l_8;
const uint32 l_180 = l_179 + l_115;
const float l_181 = l_arr_56[l_mut_175];
const float l_182 = l_181 * l_155;
const __half l_183 = buffer_2[l_180];
const float l_184 = float(l_183);
const float l_185 = l_157 * l_184;
const float l_186 = l_182 + l_185;
l_arr_56[l_mut_175] = l_186;
}
}
}
}
const bool l_187 = l_14 == uint32(1);
if (l_187) {
l_mut_83 = float(0.0);
const bool l_188 = l_mut_64 > float(0.0);
if (l_188) {
const float l_189 = float(1.0) / l_mut_64;
l_mut_83 = l_189;
}

for (uint32 l_mut_190 = uint32(0); l_mut_190 < uint32(8); ++l_mut_190) {
const uint32 l_191 = l_mut_190 * uint32(32);
const uint32 l_192 = l_191 + threadIdxGlobal;
const uint32 l_193 = l_54 + l_192;
const float l_194 = l_arr_56[l_mut_190];
const float l_195 = l_194 * l_mut_83;
const __half l_196 = __half(l_195);
buffer_5[l_193] = l_196;
}
} else {
const uint32 l_197 = l_22 * l_14;
const uint32 l_198 = l_197 + l_19;
const uint32 l_199 = l_198 * uint32(258);

for (uint32 l_mut_200 = uint32(0); l_mut_200 < uint32(8); ++l_mut_200) {
const uint32 l_201 = l_mut_200 * uint32(32);
const uint32 l_202 = l_201 + threadIdxGlobal;
const uint32 l_203 = l_199 + l_202;
const float l_204 = l_arr_56[l_mut_200];
buffer_6[l_203] = l_204;
}
const bool l_205 = threadIdxGlobal == uint32(0);
if (l_205) {
const uint32 l_206 = l_199 + uint32(256);
buffer_6[l_206] = l_mut_63;
const uint32 l_207 = l_199 + uint32(256);
const uint32 l_208 = l_207 + uint32(1);
buffer_6[l_208] = l_mut_64;
}
}

}
