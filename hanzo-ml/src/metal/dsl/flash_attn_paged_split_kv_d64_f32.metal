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
float l_arr_51[2];

float l_arr_52[2];

uint l_mut_24;
uint l_mut_32;
uint l_mut_48;
float l_mut_58;
float l_mut_59;
float l_mut_61;
float l_mut_73;
float l_mut_87;
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
const uint l_15 = l_1 / l_14;
const uint l_16 = l_1 % l_14;
const uint l_17 = l_15 % l_4;
const uint l_18 = l_15 / l_4;
const uint l_19 = l_18 % l_5;
const uint l_20 = l_18 / l_5;
const uint l_21 = l_5 / l_6;
const uint l_22 = l_19 / l_21;
const uint l_23 = buffer_4[l_20];
l_mut_24 = uint(0);
const bool l_25 = l_7 == uint(2);
if (l_25) {
l_mut_24 = l_13;
} else {
const bool l_26 = l_23 >= l_4;
if (l_26) {
const uint l_27 = l_23 - l_4;
l_mut_24 = l_27;
}
}
const uint l_28 = l_mut_24 + l_17;
l_mut_24 = l_23;
const bool l_29 = l_7 != uint(0);
if (l_29) {
const uint l_30 = l_28 + uint(1);
const bool l_31 = l_30 < l_mut_24;
if (l_31) {
l_mut_24 = l_30;
}
}
l_mut_32 = uint(0);
const bool l_33 = l_12 > uint(0);
const uint l_34 = l_28 + uint(1);
const bool l_35 = l_34 > l_12;
const bool l_36 = l_33 && l_35;
if (l_36) {
const uint l_37 = l_28 + uint(1);
const uint l_38 = l_37 - l_12;
l_mut_32 = l_38;
}
const bool l_39 = l_mut_32 > l_mut_24;
if (l_39) {
l_mut_32 = l_mut_24;
}
const uint l_40 = l_mut_24 - l_mut_32;
const uint l_41 = l_40 + l_14;
const uint l_42 = l_41 - uint(1);
const uint l_43 = l_42 / l_14;
const uint l_44 = l_16 * l_43;
const uint l_45 = l_mut_32 + l_44;
l_mut_32 = l_45;
const bool l_46 = l_mut_32 > l_mut_24;
if (l_46) {
l_mut_32 = l_mut_24;
}
const uint l_47 = l_mut_32 + l_43;
l_mut_48 = l_47;
const bool l_49 = l_mut_48 > l_mut_24;
if (l_49) {
l_mut_48 = l_mut_24;
}
const uint l_50 = l_15 * uint(64);

for (uint l_mut_53 = uint(0); l_mut_53 < uint(2); ++l_mut_53) {
const uint l_54 = l_mut_53 * uint(32);
const uint l_55 = l_54 + thread_index_in_threadgroup;
const uint l_56 = l_50 + l_55;
const float l_57 = buffer_0[l_56];
l_arr_51[l_mut_53] = l_57;
l_arr_52[l_mut_53] = float(0.0);
}
l_mut_58 = float(-3.3999999521443642e38);
l_mut_59 = float(0.0);

for (uint l_mut_60 = l_mut_32; l_mut_60 < l_mut_48; ++l_mut_60) {
l_mut_61 = float(0.0);

for (uint l_mut_62 = uint(0); l_mut_62 < uint(2); ++l_mut_62) {
const uint l_63 = l_mut_62 * uint(32);
const uint l_64 = l_63 + thread_index_in_threadgroup;
const bool l_65 = l_10 == uint(2);
if (l_65) {
const uint l_66 = l_20 * l_6;
const uint l_67 = l_66 + l_22;
const uint l_68 = l_67 * l_8;
const uint l_69 = l_68 + l_mut_60;
const uint l_70 = l_69 * uint(64);
const uint l_71 = l_70 + l_64;
const float l_72 = buffer_1[l_71];
l_mut_73 = l_72;
} else {
const uint l_74 = l_20 * l_9;
const uint l_75 = l_mut_60 / l_8;
const uint l_76 = l_74 + l_75;
const uint l_77 = buffer_3[l_76];
const uint l_78 = l_mut_60 % l_8;
const bool l_79 = l_10 == uint(0);
if (l_79) {
const uint l_80 = l_77 * l_6;
const uint l_81 = l_80 + l_22;
const uint l_82 = l_81 * l_8;
const uint l_83 = l_82 + l_78;
const uint l_84 = l_83 * uint(64);
const uint l_85 = l_84 + l_64;
const float l_86 = buffer_1[l_85];
l_mut_87 = l_86;
} else {
const uint l_88 = uint(64) / l_11;
const uint l_89 = l_77 * l_6;
const uint l_90 = l_89 + l_22;
const uint l_91 = l_90 * l_88;
const uint l_92 = l_64 / l_11;
const uint l_93 = l_91 + l_92;
const uint l_94 = l_93 * l_8;
const uint l_95 = l_94 + l_78;
const uint l_96 = l_95 * l_11;
const uint l_97 = l_64 % l_11;
const uint l_98 = l_96 + l_97;
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
l_mut_24 = uint(1);
while (true) {
const bool l_102 = l_mut_24 < uint(32);
const bool l_103 = !l_102;
if (l_103) {
break;}
const float l_104 = { simd_shuffle_xor(l_mut_73, l_mut_24) };
l_mut_73 = l_mut_73 + l_104;
l_mut_24 = l_mut_24 * uint(2);
}
const float l_105 = l_mut_73 * l_2;
l_mut_87 = l_105;
const bool l_106 = l_3 > float(0.0);
if (l_106) {
const float l_107 = l_mut_87 / l_3;
const float l_108 = safe_tanh_scalar(l_107);
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

for (uint l_mut_117 = uint(0); l_mut_117 < uint(2); ++l_mut_117) {
const uint l_118 = l_mut_117 * uint(32);
const uint l_119 = l_118 + thread_index_in_threadgroup;
const bool l_120 = l_10 == uint(2);
if (l_120) {
const uint l_121 = l_20 * l_6;
const uint l_122 = l_121 + l_22;
const uint l_123 = l_122 * l_8;
const uint l_124 = l_123 + l_mut_60;
const uint l_125 = l_124 * uint(64);
const uint l_126 = l_125 + l_119;
const float l_127 = buffer_2[l_126];
l_mut_87 = l_127;
} else {
const uint l_128 = l_20 * l_9;
const uint l_129 = l_mut_60 / l_8;
const uint l_130 = l_128 + l_129;
const uint l_131 = buffer_3[l_130];
const uint l_132 = l_mut_60 % l_8;
const bool l_133 = l_10 == uint(0);
if (l_133) {
const uint l_134 = l_131 * l_6;
const uint l_135 = l_134 + l_22;
const uint l_136 = l_135 * l_8;
const uint l_137 = l_136 + l_132;
const uint l_138 = l_137 * uint(64);
const uint l_139 = l_138 + l_119;
const float l_140 = buffer_2[l_139];
l_mut_73 = l_140;
} else {
const uint l_141 = l_131 * l_6;
const uint l_142 = l_141 + l_22;
const uint l_143 = l_142 * uint(64);
const uint l_144 = l_143 + l_119;
const uint l_145 = l_144 * l_8;
const uint l_146 = l_145 + l_132;
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
const bool l_152 = l_14 == uint(1);
if (l_152) {
l_mut_87 = float(0.0);
const bool l_153 = l_mut_59 > float(0.0);
if (l_153) {
const float l_154 = float(1.0) / l_mut_59;
l_mut_87 = l_154;
}

for (uint l_mut_155 = uint(0); l_mut_155 < uint(2); ++l_mut_155) {
const uint l_156 = l_mut_155 * uint(32);
const uint l_157 = l_156 + thread_index_in_threadgroup;
const uint l_158 = l_50 + l_157;
const float l_159 = l_arr_52[l_mut_155];
const float l_160 = l_159 * l_mut_87;
buffer_5[l_158] = l_160;
}
} else {
const uint l_161 = l_15 * l_14;
const uint l_162 = l_161 + l_16;
const uint l_163 = l_162 * uint(66);

for (uint l_mut_164 = uint(0); l_mut_164 < uint(2); ++l_mut_164) {
const uint l_165 = l_mut_164 * uint(32);
const uint l_166 = l_165 + thread_index_in_threadgroup;
const uint l_167 = l_163 + l_166;
const float l_168 = l_arr_52[l_mut_164];
buffer_6[l_167] = l_168;
}
const bool l_169 = thread_index_in_threadgroup == uint(0);
if (l_169) {
const uint l_170 = l_163 + uint(64);
buffer_6[l_170] = l_mut_58;
const uint l_171 = l_163 + uint(64);
const uint l_172 = l_171 + uint(1);
buffer_6[l_172] = l_mut_59;
}
}

}
