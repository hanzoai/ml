#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[18];
};

/// Metal has a weird numerical behaviour with tanh for inputs over 43.0
inline float safe_tanh_scalar(float x) {
    if (x > 43.0) {
        return 1.0;
    } else {
        return tanh(x);
    }
}
inline float safe_tanh(float x) {
    return safe_tanh_scalar(x);
}

[[kernel]]
void flash_attn_paged_split_kv_f_(
    const device bfloat * buffer_0 [[buffer(0)]],
    const device bfloat * buffer_1 [[buffer(1)]],
    const device bfloat * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    device bfloat * buffer_5 [[buffer(5)]],
    device float * buffer_6 [[buffer(6)]],
    const device float * buffer_7 [[buffer(7)]],
    const device uint * buffer_8 [[buffer(8)]],
    constant info_st& info [[buffer(9)]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]]
) {
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
uint threads_per_simdgroup_checked = min(simd_size, threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z);
float l_arr_55[2];

float l_arr_56[2];

uint l_mut_28;
uint l_mut_36;
uint l_mut_52;
float l_mut_63;
float l_mut_64;
float l_mut_73;
float l_mut_82;
float l_mut_83;
const uint l_0 = buffer_8[uint(10)];
const uint l_1 = threadgroup_index_in_grid + l_0;
const float l_2 = buffer_7[uint(0)];
const float l_3 = buffer_7[uint(1)];
const uint l_4 = buffer_8[uint(0)];
const uint l_5 = buffer_8[uint(1)];
const uint l_6 = buffer_8[uint(2)];
const uint l_7 = buffer_8[uint(3)];
const uint l_8 = buffer_8[uint(4)];
const uint l_9 = buffer_8[uint(5)];
const uint l_10 = buffer_8[uint(6)];
const uint l_11 = buffer_8[uint(7)];
const uint l_12 = buffer_8[uint(8)];
const uint l_13 = buffer_8[uint(9)];
const uint l_14 = buffer_8[uint(11)];
const uint l_15 = l_5 * l_4;
const uint l_16 = l_15 * l_14;
const uint l_17 = l_1 / l_16;
const uint l_18 = l_1 % l_16;
const uint l_19 = l_18 / l_15;
const uint l_20 = l_18 % l_15;
const uint l_21 = l_17 * l_15;
const uint l_22 = l_21 + l_20;
const uint l_23 = l_20 % l_4;
const uint l_24 = l_20 / l_4;
const uint l_25 = l_5 / l_6;
const uint l_26 = l_24 / l_25;
const uint l_27 = buffer_4[l_17];
l_mut_28 = uint(0);
const bool l_29 = l_7 == uint(2);
if (l_29) {
l_mut_28 = l_13;
} else {
const bool l_30 = l_27 >= l_4;
if (l_30) {
const uint l_31 = l_27 - l_4;
l_mut_28 = l_31;
}
}
const uint l_32 = l_mut_28 + l_23;
l_mut_28 = l_27;
const bool l_33 = l_7 != uint(0);
if (l_33) {
const uint l_34 = l_32 + uint(1);
const bool l_35 = l_34 < l_mut_28;
if (l_35) {
l_mut_28 = l_34;
}
}
l_mut_36 = uint(0);
const bool l_37 = l_12 > uint(0);
const uint l_38 = l_32 + uint(1);
const bool l_39 = l_38 > l_12;
const bool l_40 = l_37 && l_39;
if (l_40) {
const uint l_41 = l_32 + uint(1);
const uint l_42 = l_41 - l_12;
l_mut_36 = l_42;
}
const bool l_43 = l_mut_36 > l_mut_28;
if (l_43) {
l_mut_36 = l_mut_28;
}
const uint l_44 = l_mut_28 - l_mut_36;
const uint l_45 = l_44 + l_14;
const uint l_46 = l_45 - uint(1);
const uint l_47 = l_46 / l_14;
const uint l_48 = l_19 * l_47;
const uint l_49 = l_mut_36 + l_48;
l_mut_36 = l_49;
const bool l_50 = l_mut_36 > l_mut_28;
if (l_50) {
l_mut_36 = l_mut_28;
}
const uint l_51 = l_mut_36 + l_47;
l_mut_52 = l_51;
const bool l_53 = l_mut_52 > l_mut_28;
if (l_53) {
l_mut_52 = l_mut_28;
}
const uint l_54 = l_22 * uint(64);

for (uint l_mut_57 = uint(0); l_mut_57 < uint(2); ++l_mut_57) {
const uint l_58 = l_mut_57 * uint(32);
const uint l_59 = l_58 + thread_index_in_threadgroup;
const uint l_60 = l_54 + l_59;
const bfloat l_61 = buffer_0[l_60];
const float l_62 = float(l_61);
l_arr_55[l_mut_57] = l_62;
l_arr_56[l_mut_57] = float(0.0);
}
l_mut_63 = float(-3.3999999521443642e38);
l_mut_64 = float(0.0);
const bool l_65 = l_10 == uint(2);
if (l_65) {
const uint l_66 = l_17 * l_6;
const uint l_67 = l_66 + l_26;
const uint l_68 = l_67 * l_8;
const uint l_69 = l_68 * uint(64);

for (uint l_mut_70 = l_mut_36; l_mut_70 < l_mut_52; ++l_mut_70) {
const uint l_71 = l_mut_70 * uint(64);
const uint l_72 = l_69 + l_71;
l_mut_73 = float(0.0);

for (uint l_mut_74 = uint(0); l_mut_74 < uint(2); ++l_mut_74) {
const uint l_75 = l_mut_74 * uint(32);
const uint l_76 = l_75 + thread_index_in_threadgroup;
const float l_77 = l_arr_55[l_mut_74];
const uint l_78 = l_72 + l_76;
const bfloat l_79 = buffer_1[l_78];
const float l_80 = float(l_79);
const float l_81 = l_77 * l_80;
l_mut_73 = l_mut_73 + l_81;
}
l_mut_82 = float(0.0);
l_mut_83 = l_mut_73;
l_mut_28 = uint(1);
while (true) {
const bool l_84 = l_mut_28 < uint(32);
const bool l_85 = !l_84;
if (l_85) {
break;}
const float l_86 = { simd_shuffle_xor(l_mut_83, l_mut_28) };
l_mut_83 = l_mut_83 + l_86;
l_mut_28 = l_mut_28 * uint(2);
}
const float l_87 = l_mut_83 * l_2;
l_mut_82 = l_87;
const bool l_88 = l_3 > float(0.0);
if (l_88) {
const float l_89 = l_mut_82 / l_3;
const float l_90 = safe_tanh_scalar(l_89);
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

for (uint l_mut_99 = uint(0); l_mut_99 < uint(2); ++l_mut_99) {
const uint l_100 = l_mut_99 * uint(32);
const uint l_101 = l_100 + thread_index_in_threadgroup;
const float l_102 = l_arr_56[l_mut_99];
const float l_103 = l_102 * l_94;
const uint l_104 = l_72 + l_101;
const bfloat l_105 = buffer_2[l_104];
const float l_106 = float(l_105);
const float l_107 = l_96 * l_106;
const float l_108 = l_103 + l_107;
l_arr_56[l_mut_99] = l_108;
}
}
} else {
const uint l_109 = l_17 * l_9;
const uint l_110 = uint(64) / l_11;

for (uint l_mut_111 = l_mut_36; l_mut_111 < l_mut_52; ++l_mut_111) {
const uint l_112 = l_mut_111 / l_8;
const uint l_113 = l_109 + l_112;
const uint l_114 = buffer_3[l_113];
const uint l_115 = l_mut_111 % l_8;
const uint l_116 = l_114 * l_6;
const uint l_117 = l_116 + l_26;
l_mut_83 = float(0.0);
const bool l_118 = l_10 == uint(0);
if (l_118) {
const uint l_119 = l_117 * l_8;
const uint l_120 = l_119 + l_115;
const uint l_121 = l_120 * uint(64);

for (uint l_mut_122 = uint(0); l_mut_122 < uint(2); ++l_mut_122) {
const uint l_123 = l_mut_122 * uint(32);
const uint l_124 = l_123 + thread_index_in_threadgroup;
const float l_125 = l_arr_55[l_mut_122];
const uint l_126 = l_121 + l_124;
const bfloat l_127 = buffer_1[l_126];
const float l_128 = float(l_127);
const float l_129 = l_125 * l_128;
l_mut_83 = l_mut_83 + l_129;
}
} else {
const uint l_130 = l_117 * l_110;

for (uint l_mut_131 = uint(0); l_mut_131 < uint(2); ++l_mut_131) {
const uint l_132 = l_mut_131 * uint(32);
const uint l_133 = l_132 + thread_index_in_threadgroup;
const uint l_134 = l_133 / l_11;
const uint l_135 = l_130 + l_134;
const uint l_136 = l_135 * l_8;
const uint l_137 = l_136 + l_115;
const uint l_138 = l_137 * l_11;
const uint l_139 = l_133 % l_11;
const uint l_140 = l_138 + l_139;
const float l_141 = l_arr_55[l_mut_131];
const bfloat l_142 = buffer_1[l_140];
const float l_143 = float(l_142);
const float l_144 = l_141 * l_143;
l_mut_83 = l_mut_83 + l_144;
}
}
l_mut_82 = float(0.0);
l_mut_73 = l_mut_83;
l_mut_28 = uint(1);
while (true) {
const bool l_145 = l_mut_28 < uint(32);
const bool l_146 = !l_145;
if (l_146) {
break;}
const float l_147 = { simd_shuffle_xor(l_mut_73, l_mut_28) };
l_mut_73 = l_mut_73 + l_147;
l_mut_28 = l_mut_28 * uint(2);
}
const float l_148 = l_mut_73 * l_2;
l_mut_82 = l_148;
const bool l_149 = l_3 > float(0.0);
if (l_149) {
const float l_150 = l_mut_82 / l_3;
const float l_151 = safe_tanh_scalar(l_150);
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
const bool l_160 = l_10 == uint(0);
if (l_160) {
const uint l_161 = l_117 * l_8;
const uint l_162 = l_161 + l_115;
const uint l_163 = l_162 * uint(64);

for (uint l_mut_164 = uint(0); l_mut_164 < uint(2); ++l_mut_164) {
const uint l_165 = l_mut_164 * uint(32);
const uint l_166 = l_165 + thread_index_in_threadgroup;
const float l_167 = l_arr_56[l_mut_164];
const float l_168 = l_167 * l_155;
const uint l_169 = l_163 + l_166;
const bfloat l_170 = buffer_2[l_169];
const float l_171 = float(l_170);
const float l_172 = l_157 * l_171;
const float l_173 = l_168 + l_172;
l_arr_56[l_mut_164] = l_173;
}
} else {
const uint l_174 = l_117 * uint(64);

for (uint l_mut_175 = uint(0); l_mut_175 < uint(2); ++l_mut_175) {
const uint l_176 = l_mut_175 * uint(32);
const uint l_177 = l_176 + thread_index_in_threadgroup;
const uint l_178 = l_174 + l_177;
const uint l_179 = l_178 * l_8;
const uint l_180 = l_179 + l_115;
const float l_181 = l_arr_56[l_mut_175];
const float l_182 = l_181 * l_155;
const bfloat l_183 = buffer_2[l_180];
const float l_184 = float(l_183);
const float l_185 = l_157 * l_184;
const float l_186 = l_182 + l_185;
l_arr_56[l_mut_175] = l_186;
}
}
}
}
const bool l_187 = l_14 == uint(1);
if (l_187) {
l_mut_83 = float(0.0);
const bool l_188 = l_mut_64 > float(0.0);
if (l_188) {
const float l_189 = float(1.0) / l_mut_64;
l_mut_83 = l_189;
}

for (uint l_mut_190 = uint(0); l_mut_190 < uint(2); ++l_mut_190) {
const uint l_191 = l_mut_190 * uint(32);
const uint l_192 = l_191 + thread_index_in_threadgroup;
const uint l_193 = l_54 + l_192;
const float l_194 = l_arr_56[l_mut_190];
const float l_195 = l_194 * l_mut_83;
const bfloat l_196 = bfloat(l_195);
buffer_5[l_193] = l_196;
}
} else {
const uint l_197 = l_22 * l_14;
const uint l_198 = l_197 + l_19;
const uint l_199 = l_198 * uint(66);

for (uint l_mut_200 = uint(0); l_mut_200 < uint(2); ++l_mut_200) {
const uint l_201 = l_mut_200 * uint(32);
const uint l_202 = l_201 + thread_index_in_threadgroup;
const uint l_203 = l_199 + l_202;
const float l_204 = l_arr_56[l_mut_200];
buffer_6[l_203] = l_204;
}
const bool l_205 = thread_index_in_threadgroup == uint(0);
if (l_205) {
const uint l_206 = l_199 + uint(64);
buffer_6[l_206] = l_mut_63;
const uint l_207 = l_199 + uint(64);
const uint l_208 = l_207 + uint(1);
buffer_6[l_208] = l_mut_64;
}
}

}
