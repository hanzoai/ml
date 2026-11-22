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
void flash_attn_paged_split_kv_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    const device float * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    device float * buffer_5 [[buffer(5)]],
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
float l_arr_55[8];

float l_arr_56[8];

uint l_mut_28;
uint l_mut_36;
uint l_mut_52;
float l_mut_62;
float l_mut_63;
float l_mut_72;
float l_mut_80;
float l_mut_81;
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
const uint l_54 = l_22 * uint(256);

for (uint l_mut_57 = uint(0); l_mut_57 < uint(8); ++l_mut_57) {
const uint l_58 = l_mut_57 * uint(32);
const uint l_59 = l_58 + thread_index_in_threadgroup;
const uint l_60 = l_54 + l_59;
const float l_61 = buffer_0[l_60];
l_arr_55[l_mut_57] = l_61;
l_arr_56[l_mut_57] = float(0.0);
}
l_mut_62 = float(-3.3999999521443642e38);
l_mut_63 = float(0.0);
const bool l_64 = l_10 == uint(2);
if (l_64) {
const uint l_65 = l_17 * l_6;
const uint l_66 = l_65 + l_26;
const uint l_67 = l_66 * l_8;
const uint l_68 = l_67 * uint(256);

for (uint l_mut_69 = l_mut_36; l_mut_69 < l_mut_52; ++l_mut_69) {
const uint l_70 = l_mut_69 * uint(256);
const uint l_71 = l_68 + l_70;
l_mut_72 = float(0.0);

for (uint l_mut_73 = uint(0); l_mut_73 < uint(8); ++l_mut_73) {
const uint l_74 = l_mut_73 * uint(32);
const uint l_75 = l_74 + thread_index_in_threadgroup;
const float l_76 = l_arr_55[l_mut_73];
const uint l_77 = l_71 + l_75;
const float l_78 = buffer_1[l_77];
const float l_79 = l_76 * l_78;
l_mut_72 = l_mut_72 + l_79;
}
l_mut_80 = float(0.0);
l_mut_81 = l_mut_72;
l_mut_28 = uint(1);
while (true) {
const bool l_82 = l_mut_28 < uint(32);
const bool l_83 = !l_82;
if (l_83) {
break;}
const float l_84 = { simd_shuffle_xor(l_mut_81, l_mut_28) };
l_mut_81 = l_mut_81 + l_84;
l_mut_28 = l_mut_28 * uint(2);
}
const float l_85 = l_mut_81 * l_2;
l_mut_80 = l_85;
const bool l_86 = l_3 > float(0.0);
if (l_86) {
const float l_87 = l_mut_80 / l_3;
const float l_88 = safe_tanh_scalar(l_87);
const float l_89 = l_3 * l_88;
l_mut_80 = l_89;
}
l_mut_81 = l_mut_62;
const bool l_90 = l_mut_80 > l_mut_81;
if (l_90) {
l_mut_81 = l_mut_80;
}
const float l_91 = l_mut_62 - l_mut_81;
const float l_92 = exp(l_91);
const float l_93 = l_mut_80 - l_mut_81;
const float l_94 = exp(l_93);
const float l_95 = l_mut_63 * l_92;
const float l_96 = l_95 + l_94;
l_mut_63 = l_96;
l_mut_62 = l_mut_81;

for (uint l_mut_97 = uint(0); l_mut_97 < uint(8); ++l_mut_97) {
const uint l_98 = l_mut_97 * uint(32);
const uint l_99 = l_98 + thread_index_in_threadgroup;
const float l_100 = l_arr_56[l_mut_97];
const float l_101 = l_100 * l_92;
const uint l_102 = l_71 + l_99;
const float l_103 = buffer_2[l_102];
const float l_104 = l_94 * l_103;
const float l_105 = l_101 + l_104;
l_arr_56[l_mut_97] = l_105;
}
}
} else {
const uint l_106 = l_17 * l_9;
const uint l_107 = uint(256) / l_11;

for (uint l_mut_108 = l_mut_36; l_mut_108 < l_mut_52; ++l_mut_108) {
const uint l_109 = l_mut_108 / l_8;
const uint l_110 = l_106 + l_109;
const uint l_111 = buffer_3[l_110];
const uint l_112 = l_mut_108 % l_8;
const uint l_113 = l_111 * l_6;
const uint l_114 = l_113 + l_26;
l_mut_81 = float(0.0);
const bool l_115 = l_10 == uint(0);
if (l_115) {
const uint l_116 = l_114 * l_8;
const uint l_117 = l_116 + l_112;
const uint l_118 = l_117 * uint(256);

for (uint l_mut_119 = uint(0); l_mut_119 < uint(8); ++l_mut_119) {
const uint l_120 = l_mut_119 * uint(32);
const uint l_121 = l_120 + thread_index_in_threadgroup;
const float l_122 = l_arr_55[l_mut_119];
const uint l_123 = l_118 + l_121;
const float l_124 = buffer_1[l_123];
const float l_125 = l_122 * l_124;
l_mut_81 = l_mut_81 + l_125;
}
} else {
const uint l_126 = l_114 * l_107;

for (uint l_mut_127 = uint(0); l_mut_127 < uint(8); ++l_mut_127) {
const uint l_128 = l_mut_127 * uint(32);
const uint l_129 = l_128 + thread_index_in_threadgroup;
const uint l_130 = l_129 / l_11;
const uint l_131 = l_126 + l_130;
const uint l_132 = l_131 * l_8;
const uint l_133 = l_132 + l_112;
const uint l_134 = l_133 * l_11;
const uint l_135 = l_129 % l_11;
const uint l_136 = l_134 + l_135;
const float l_137 = l_arr_55[l_mut_127];
const float l_138 = buffer_1[l_136];
const float l_139 = l_137 * l_138;
l_mut_81 = l_mut_81 + l_139;
}
}
l_mut_80 = float(0.0);
l_mut_72 = l_mut_81;
l_mut_28 = uint(1);
while (true) {
const bool l_140 = l_mut_28 < uint(32);
const bool l_141 = !l_140;
if (l_141) {
break;}
const float l_142 = { simd_shuffle_xor(l_mut_72, l_mut_28) };
l_mut_72 = l_mut_72 + l_142;
l_mut_28 = l_mut_28 * uint(2);
}
const float l_143 = l_mut_72 * l_2;
l_mut_80 = l_143;
const bool l_144 = l_3 > float(0.0);
if (l_144) {
const float l_145 = l_mut_80 / l_3;
const float l_146 = safe_tanh_scalar(l_145);
const float l_147 = l_3 * l_146;
l_mut_80 = l_147;
}
l_mut_72 = l_mut_62;
const bool l_148 = l_mut_80 > l_mut_72;
if (l_148) {
l_mut_72 = l_mut_80;
}
const float l_149 = l_mut_62 - l_mut_72;
const float l_150 = exp(l_149);
const float l_151 = l_mut_80 - l_mut_72;
const float l_152 = exp(l_151);
const float l_153 = l_mut_63 * l_150;
const float l_154 = l_153 + l_152;
l_mut_63 = l_154;
l_mut_62 = l_mut_72;
const bool l_155 = l_10 == uint(0);
if (l_155) {
const uint l_156 = l_114 * l_8;
const uint l_157 = l_156 + l_112;
const uint l_158 = l_157 * uint(256);

for (uint l_mut_159 = uint(0); l_mut_159 < uint(8); ++l_mut_159) {
const uint l_160 = l_mut_159 * uint(32);
const uint l_161 = l_160 + thread_index_in_threadgroup;
const float l_162 = l_arr_56[l_mut_159];
const float l_163 = l_162 * l_150;
const uint l_164 = l_158 + l_161;
const float l_165 = buffer_2[l_164];
const float l_166 = l_152 * l_165;
const float l_167 = l_163 + l_166;
l_arr_56[l_mut_159] = l_167;
}
} else {
const uint l_168 = l_114 * uint(256);

for (uint l_mut_169 = uint(0); l_mut_169 < uint(8); ++l_mut_169) {
const uint l_170 = l_mut_169 * uint(32);
const uint l_171 = l_170 + thread_index_in_threadgroup;
const uint l_172 = l_168 + l_171;
const uint l_173 = l_172 * l_8;
const uint l_174 = l_173 + l_112;
const float l_175 = l_arr_56[l_mut_169];
const float l_176 = l_175 * l_150;
const float l_177 = buffer_2[l_174];
const float l_178 = l_152 * l_177;
const float l_179 = l_176 + l_178;
l_arr_56[l_mut_169] = l_179;
}
}
}
}
const bool l_180 = l_14 == uint(1);
if (l_180) {
l_mut_81 = float(0.0);
const bool l_181 = l_mut_63 > float(0.0);
if (l_181) {
const float l_182 = float(1.0) / l_mut_63;
l_mut_81 = l_182;
}

for (uint l_mut_183 = uint(0); l_mut_183 < uint(8); ++l_mut_183) {
const uint l_184 = l_mut_183 * uint(32);
const uint l_185 = l_184 + thread_index_in_threadgroup;
const uint l_186 = l_54 + l_185;
const float l_187 = l_arr_56[l_mut_183];
const float l_188 = l_187 * l_mut_81;
buffer_5[l_186] = l_188;
}
} else {
const uint l_189 = l_22 * l_14;
const uint l_190 = l_189 + l_19;
const uint l_191 = l_190 * uint(258);

for (uint l_mut_192 = uint(0); l_mut_192 < uint(8); ++l_mut_192) {
const uint l_193 = l_mut_192 * uint(32);
const uint l_194 = l_193 + thread_index_in_threadgroup;
const uint l_195 = l_191 + l_194;
const float l_196 = l_arr_56[l_mut_192];
buffer_6[l_195] = l_196;
}
const bool l_197 = thread_index_in_threadgroup == uint(0);
if (l_197) {
const uint l_198 = l_191 + uint(256);
buffer_6[l_198] = l_mut_62;
const uint l_199 = l_191 + uint(256);
const uint l_200 = l_199 + uint(1);
buffer_6[l_200] = l_mut_63;
}
}

}
