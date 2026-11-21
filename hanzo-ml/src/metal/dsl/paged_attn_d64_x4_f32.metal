#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[16];
};

[[kernel]]
void paged_attn_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    const device float * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    device float * buffer_5 [[buffer(5)]],
    const device float * buffer_6 [[buffer(6)]],
    const device uint * buffer_7 [[buffer(7)]],
    constant info_st& info [[buffer(8)]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]]
) {
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
uint threads_per_simdgroup_checked = min(simd_size, threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z);
threadgroup uchar dynamic_shared_mem[1056];
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_122 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_123 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[16]);
// Shared array size: 256, 1024 bytes
threadgroup float* shared_memory_124 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[32]);
float l_arr_26[2];

float l_arr_27[2];

uint l_mut_14;
float l_mut_36;
float l_mut_37;
uint l_mut_39;
float l_mut_53;
float l_mut_84;
const uint l_0 = thread_index_in_threadgroup / uint(32);
const uint l_1 = thread_index_in_threadgroup % uint(32);
const uint l_2 = buffer_7[uint(0)];
const uint l_3 = buffer_7[uint(1)];
const uint l_4 = buffer_7[uint(2)];
const uint l_5 = buffer_7[uint(3)];
const uint l_6 = buffer_7[uint(4)];
const float l_7 = buffer_6[uint(0)];
const float l_8 = buffer_6[uint(1)];
const uint l_9 = threadgroup_index_in_grid / l_2;
const uint l_10 = threadgroup_index_in_grid % l_2;
const uint l_11 = l_2 / l_3;
const uint l_12 = l_10 / l_11;
const uint l_13 = buffer_4[l_9];
l_mut_14 = uint(0);
const bool l_15 = l_5 > uint(0);
const bool l_16 = l_13 > l_5;
const bool l_17 = l_15 && l_16;
if (l_17) {
const uint l_18 = l_13 - l_5;
l_mut_14 = l_18;
}
const uint l_19 = l_mut_14 / l_6;
const uint l_20 = l_13 + l_6;
const uint l_21 = l_20 - uint(1);
const uint l_22 = l_21 / l_6;
const uint l_23 = l_9 * l_2;
const uint l_24 = l_23 + l_10;
const uint l_25 = l_24 * uint(64);
const uint l_28 = l_1 * uint(2);
const uint l_29 = l_25 + l_28;
const uint l_30 = l_29 + uint(0);
const float l_31 = buffer_0[l_30];
l_arr_26[uint(0)] = l_31;
l_arr_27[uint(0)] = float(0.0);
const uint l_32 = l_1 * uint(2);
const uint l_33 = l_25 + l_32;
const uint l_34 = l_33 + uint(1);
const float l_35 = buffer_0[l_34];
l_arr_26[uint(1)] = l_35;
l_arr_27[uint(1)] = float(0.0);
l_mut_36 = float(-3.3999999521443642e38);
l_mut_37 = float(0.0);
const uint l_38 = l_19 + l_0;
l_mut_39 = l_38;
while (true) {
const bool l_40 = l_mut_39 < l_22;
const bool l_41 = !l_40;
if (l_41) {
break;}
const uint l_42 = l_9 * l_4;
const uint l_43 = l_42 + l_mut_39;
const uint l_44 = buffer_3[l_43];
const uint l_45 = l_44 * l_3;
const uint l_46 = l_45 + l_12;

for (uint l_mut_47 = uint(0); l_mut_47 < l_6; ++l_mut_47) {
const uint l_48 = l_mut_39 * l_6;
const uint l_49 = l_48 + l_mut_47;
const bool l_50 = l_49 >= l_mut_14;
const bool l_51 = l_49 < l_13;
const bool l_52 = l_50 && l_51;
if (l_52) {
l_mut_53 = float(0.0);
const uint l_54 = l_1 * uint(2);
const uint l_55 = l_54 + uint(0);
const uint l_56 = l_55 / uint(4);
const uint l_57 = l_55 % uint(4);
const uint l_58 = l_46 * uint(16);
const uint l_59 = l_58 + l_56;
const uint l_60 = l_59 * l_6;
const uint l_61 = l_60 + l_mut_47;
const uint l_62 = l_61 * uint(4);
const uint l_63 = l_62 + l_57;
const float l_64 = l_arr_26[uint(0)];
const float l_65 = buffer_1[l_63];
const float l_66 = l_64 * l_65;
l_mut_53 = l_mut_53 + l_66;
const uint l_67 = l_1 * uint(2);
const uint l_68 = l_67 + uint(1);
const uint l_69 = l_68 / uint(4);
const uint l_70 = l_68 % uint(4);
const uint l_71 = l_46 * uint(16);
const uint l_72 = l_71 + l_69;
const uint l_73 = l_72 * l_6;
const uint l_74 = l_73 + l_mut_47;
const uint l_75 = l_74 * uint(4);
const uint l_76 = l_75 + l_70;
const float l_77 = l_arr_26[uint(1)];
const float l_78 = buffer_1[l_76];
const float l_79 = l_77 * l_78;
l_mut_53 = l_mut_53 + l_79;
const float l_80 = float {simd_sum(l_mut_53)
};
const float l_81 = l_80 * l_7;
l_mut_53 = l_81;
const bool l_82 = l_8 > float(0.0);
if (l_82) {
const float l_83 = l_mut_53 / l_8;
l_mut_84 = l_83;
const bool l_85 = l_83 > float(10.0);
if (l_85) {
l_mut_84 = float(10.0);
} else {
const bool l_86 = l_83 < float(-10.0);
if (l_86) {
l_mut_84 = float(-10.0);
}
}
const float l_87 = float(2.0) * l_mut_84;
const float l_88 = exp(l_87);
const float l_89 = l_88 - float(1.0);
const float l_90 = l_88 + float(1.0);
const float l_91 = l_89 / l_90;
const float l_92 = l_8 * l_91;
l_mut_53 = l_92;
}
l_mut_84 = l_mut_36;
const bool l_93 = l_mut_53 > l_mut_36;
if (l_93) {
l_mut_84 = l_mut_53;
}
const float l_94 = l_mut_36 - l_mut_84;
const float l_95 = exp(l_94);
const float l_96 = l_mut_53 - l_mut_84;
const float l_97 = exp(l_96);
const float l_98 = l_mut_37 * l_95;
const float l_99 = l_98 + l_97;
l_mut_37 = l_99;
l_mut_36 = l_mut_84;
const uint l_100 = l_1 * uint(2);
const uint l_101 = l_100 + uint(0);
const uint l_102 = l_46 * uint(64);
const uint l_103 = l_102 + l_101;
const uint l_104 = l_103 * l_6;
const uint l_105 = l_104 + l_mut_47;
const float l_106 = l_arr_27[uint(0)];
const float l_107 = l_106 * l_95;
const float l_108 = buffer_2[l_105];
const float l_109 = l_97 * l_108;
const float l_110 = l_107 + l_109;
l_arr_27[uint(0)] = l_110;
const uint l_111 = l_1 * uint(2);
const uint l_112 = l_111 + uint(1);
const uint l_113 = l_46 * uint(64);
const uint l_114 = l_113 + l_112;
const uint l_115 = l_114 * l_6;
const uint l_116 = l_115 + l_mut_47;
const float l_117 = l_arr_27[uint(1)];
const float l_118 = l_117 * l_95;
const float l_119 = buffer_2[l_116];
const float l_120 = l_97 * l_119;
const float l_121 = l_118 + l_120;
l_arr_27[uint(1)] = l_121;
}
}
l_mut_39 = l_mut_39 + uint(4);
}
const bool l_125 = l_1 == uint(0);
if (l_125) {
shared_memory_122[l_0] = l_mut_36;
shared_memory_123[l_0] = l_mut_37;
}
const uint l_126 = l_0 * uint(64);
const uint l_127 = l_1 * uint(2);
const uint l_128 = l_126 + l_127;
const uint l_129 = l_128 + uint(0);
const float l_130 = l_arr_27[uint(0)];
shared_memory_124[l_129] = l_130;
const uint l_131 = l_0 * uint(64);
const uint l_132 = l_1 * uint(2);
const uint l_133 = l_131 + l_132;
const uint l_134 = l_133 + uint(1);
const float l_135 = l_arr_27[uint(1)];
shared_memory_124[l_134] = l_135;
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_136 = l_0 == uint(0);
if (l_136) {
l_mut_84 = float(-3.3999999521443642e38);

for (uint l_mut_137 = uint(0); l_mut_137 < uint(4); ++l_mut_137) {
const float l_138 = shared_memory_123[l_mut_137];
const bool l_139 = l_138 > float(0.0);
const float l_140 = shared_memory_122[l_mut_137];
const bool l_141 = l_140 > l_mut_84;
const bool l_142 = l_139 && l_141;
if (l_142) {
const float l_143 = shared_memory_122[l_mut_137];
l_mut_84 = l_143;
}
}
l_mut_53 = float(0.0);

for (uint l_mut_144 = uint(0); l_mut_144 < uint(4); ++l_mut_144) {
const float l_145 = shared_memory_123[l_mut_144];
const bool l_146 = l_145 > float(0.0);
if (l_146) {
const float l_147 = shared_memory_123[l_mut_144];
const float l_148 = shared_memory_122[l_mut_144];
const float l_149 = l_148 - l_mut_84;
const float l_150 = exp(l_149);
const float l_151 = l_147 * l_150;
l_mut_53 = l_mut_53 + l_151;
}
}
l_mut_37 = float(0.0);
const bool l_152 = l_mut_53 > float(0.0);
if (l_152) {
const float l_153 = float(1.0) / l_mut_53;
l_mut_37 = l_153;
}
const uint l_154 = l_1 * uint(2);
const uint l_155 = l_154 + uint(0);
l_mut_53 = float(0.0);

for (uint l_mut_156 = uint(0); l_mut_156 < uint(4); ++l_mut_156) {
const float l_157 = shared_memory_123[l_mut_156];
const bool l_158 = l_157 > float(0.0);
if (l_158) {
const uint l_159 = l_mut_156 * uint(64);
const uint l_160 = l_159 + l_155;
const float l_161 = shared_memory_124[l_160];
const float l_162 = shared_memory_122[l_mut_156];
const float l_163 = l_162 - l_mut_84;
const float l_164 = exp(l_163);
const float l_165 = l_161 * l_164;
l_mut_53 = l_mut_53 + l_165;
}
}
const uint l_166 = l_25 + l_155;
const float l_167 = l_mut_53 * l_mut_37;
buffer_5[l_166] = l_167;
const uint l_168 = l_1 * uint(2);
const uint l_169 = l_168 + uint(1);
l_mut_53 = float(0.0);

for (uint l_mut_170 = uint(0); l_mut_170 < uint(4); ++l_mut_170) {
const float l_171 = shared_memory_123[l_mut_170];
const bool l_172 = l_171 > float(0.0);
if (l_172) {
const uint l_173 = l_mut_170 * uint(64);
const uint l_174 = l_173 + l_169;
const float l_175 = shared_memory_124[l_174];
const float l_176 = shared_memory_122[l_mut_170];
const float l_177 = l_176 - l_mut_84;
const float l_178 = exp(l_177);
const float l_179 = l_175 * l_178;
l_mut_53 = l_mut_53 + l_179;
}
}
const uint l_180 = l_25 + l_169;
const float l_181 = l_mut_53 * l_mut_37;
buffer_5[l_180] = l_181;
}

}
