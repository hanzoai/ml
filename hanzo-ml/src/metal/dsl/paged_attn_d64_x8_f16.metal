#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[16];
};

[[kernel]]
void paged_attn_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device half * buffer_1 [[buffer(1)]],
    const device half * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    device half * buffer_5 [[buffer(5)]],
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
threadgroup float* shared_memory_128 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_129 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[16]);
// Shared array size: 256, 1024 bytes
threadgroup float* shared_memory_130 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[32]);
float l_arr_26[2];

float l_arr_27[2];

uint l_mut_14;
float l_mut_38;
float l_mut_39;
uint l_mut_41;
float l_mut_55;
float l_mut_88;
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
const half l_31 = buffer_0[l_30];
const float l_32 = float(l_31);
l_arr_26[uint(0)] = l_32;
l_arr_27[uint(0)] = float(0.0);
const uint l_33 = l_1 * uint(2);
const uint l_34 = l_25 + l_33;
const uint l_35 = l_34 + uint(1);
const half l_36 = buffer_0[l_35];
const float l_37 = float(l_36);
l_arr_26[uint(1)] = l_37;
l_arr_27[uint(1)] = float(0.0);
l_mut_38 = float(-3.3999999521443642e38);
l_mut_39 = float(0.0);
const uint l_40 = l_19 + l_0;
l_mut_41 = l_40;
while (true) {
const bool l_42 = l_mut_41 < l_22;
const bool l_43 = !l_42;
if (l_43) {
break;}
const uint l_44 = l_9 * l_4;
const uint l_45 = l_44 + l_mut_41;
const uint l_46 = buffer_3[l_45];
const uint l_47 = l_46 * l_3;
const uint l_48 = l_47 + l_12;

for (uint l_mut_49 = uint(0); l_mut_49 < l_6; ++l_mut_49) {
const uint l_50 = l_mut_41 * l_6;
const uint l_51 = l_50 + l_mut_49;
const bool l_52 = l_51 >= l_mut_14;
const bool l_53 = l_51 < l_13;
const bool l_54 = l_52 && l_53;
if (l_54) {
l_mut_55 = float(0.0);
const uint l_56 = l_1 * uint(2);
const uint l_57 = l_56 + uint(0);
const uint l_58 = l_57 / uint(8);
const uint l_59 = l_57 % uint(8);
const uint l_60 = l_48 * uint(8);
const uint l_61 = l_60 + l_58;
const uint l_62 = l_61 * l_6;
const uint l_63 = l_62 + l_mut_49;
const uint l_64 = l_63 * uint(8);
const uint l_65 = l_64 + l_59;
const float l_66 = l_arr_26[uint(0)];
const half l_67 = buffer_1[l_65];
const float l_68 = float(l_67);
const float l_69 = l_66 * l_68;
l_mut_55 = l_mut_55 + l_69;
const uint l_70 = l_1 * uint(2);
const uint l_71 = l_70 + uint(1);
const uint l_72 = l_71 / uint(8);
const uint l_73 = l_71 % uint(8);
const uint l_74 = l_48 * uint(8);
const uint l_75 = l_74 + l_72;
const uint l_76 = l_75 * l_6;
const uint l_77 = l_76 + l_mut_49;
const uint l_78 = l_77 * uint(8);
const uint l_79 = l_78 + l_73;
const float l_80 = l_arr_26[uint(1)];
const half l_81 = buffer_1[l_79];
const float l_82 = float(l_81);
const float l_83 = l_80 * l_82;
l_mut_55 = l_mut_55 + l_83;
const float l_84 = float {simd_sum(l_mut_55)
};
const float l_85 = l_84 * l_7;
l_mut_55 = l_85;
const bool l_86 = l_8 > float(0.0);
if (l_86) {
const float l_87 = l_mut_55 / l_8;
l_mut_88 = l_87;
const bool l_89 = l_87 > float(10.0);
if (l_89) {
l_mut_88 = float(10.0);
} else {
const bool l_90 = l_87 < float(-10.0);
if (l_90) {
l_mut_88 = float(-10.0);
}
}
const float l_91 = float(2.0) * l_mut_88;
const float l_92 = exp(l_91);
const float l_93 = l_92 - float(1.0);
const float l_94 = l_92 + float(1.0);
const float l_95 = l_93 / l_94;
const float l_96 = l_8 * l_95;
l_mut_55 = l_96;
}
l_mut_88 = l_mut_38;
const bool l_97 = l_mut_55 > l_mut_38;
if (l_97) {
l_mut_88 = l_mut_55;
}
const float l_98 = l_mut_38 - l_mut_88;
const float l_99 = exp(l_98);
const float l_100 = l_mut_55 - l_mut_88;
const float l_101 = exp(l_100);
const float l_102 = l_mut_39 * l_99;
const float l_103 = l_102 + l_101;
l_mut_39 = l_103;
l_mut_38 = l_mut_88;
const uint l_104 = l_1 * uint(2);
const uint l_105 = l_104 + uint(0);
const uint l_106 = l_48 * uint(64);
const uint l_107 = l_106 + l_105;
const uint l_108 = l_107 * l_6;
const uint l_109 = l_108 + l_mut_49;
const float l_110 = l_arr_27[uint(0)];
const float l_111 = l_110 * l_99;
const half l_112 = buffer_2[l_109];
const float l_113 = float(l_112);
const float l_114 = l_101 * l_113;
const float l_115 = l_111 + l_114;
l_arr_27[uint(0)] = l_115;
const uint l_116 = l_1 * uint(2);
const uint l_117 = l_116 + uint(1);
const uint l_118 = l_48 * uint(64);
const uint l_119 = l_118 + l_117;
const uint l_120 = l_119 * l_6;
const uint l_121 = l_120 + l_mut_49;
const float l_122 = l_arr_27[uint(1)];
const float l_123 = l_122 * l_99;
const half l_124 = buffer_2[l_121];
const float l_125 = float(l_124);
const float l_126 = l_101 * l_125;
const float l_127 = l_123 + l_126;
l_arr_27[uint(1)] = l_127;
}
}
l_mut_41 = l_mut_41 + uint(4);
}
const bool l_131 = l_1 == uint(0);
if (l_131) {
shared_memory_128[l_0] = l_mut_38;
shared_memory_129[l_0] = l_mut_39;
}
const uint l_132 = l_0 * uint(64);
const uint l_133 = l_1 * uint(2);
const uint l_134 = l_132 + l_133;
const uint l_135 = l_134 + uint(0);
const float l_136 = l_arr_27[uint(0)];
shared_memory_130[l_135] = l_136;
const uint l_137 = l_0 * uint(64);
const uint l_138 = l_1 * uint(2);
const uint l_139 = l_137 + l_138;
const uint l_140 = l_139 + uint(1);
const float l_141 = l_arr_27[uint(1)];
shared_memory_130[l_140] = l_141;
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_142 = l_0 == uint(0);
if (l_142) {
l_mut_88 = float(-3.3999999521443642e38);

for (uint l_mut_143 = uint(0); l_mut_143 < uint(4); ++l_mut_143) {
const float l_144 = shared_memory_129[l_mut_143];
const bool l_145 = l_144 > float(0.0);
const float l_146 = shared_memory_128[l_mut_143];
const bool l_147 = l_146 > l_mut_88;
const bool l_148 = l_145 && l_147;
if (l_148) {
const float l_149 = shared_memory_128[l_mut_143];
l_mut_88 = l_149;
}
}
l_mut_55 = float(0.0);

for (uint l_mut_150 = uint(0); l_mut_150 < uint(4); ++l_mut_150) {
const float l_151 = shared_memory_129[l_mut_150];
const bool l_152 = l_151 > float(0.0);
if (l_152) {
const float l_153 = shared_memory_129[l_mut_150];
const float l_154 = shared_memory_128[l_mut_150];
const float l_155 = l_154 - l_mut_88;
const float l_156 = exp(l_155);
const float l_157 = l_153 * l_156;
l_mut_55 = l_mut_55 + l_157;
}
}
l_mut_39 = float(0.0);
const bool l_158 = l_mut_55 > float(0.0);
if (l_158) {
const float l_159 = float(1.0) / l_mut_55;
l_mut_39 = l_159;
}
const uint l_160 = l_1 * uint(2);
const uint l_161 = l_160 + uint(0);
l_mut_55 = float(0.0);

for (uint l_mut_162 = uint(0); l_mut_162 < uint(4); ++l_mut_162) {
const float l_163 = shared_memory_129[l_mut_162];
const bool l_164 = l_163 > float(0.0);
if (l_164) {
const uint l_165 = l_mut_162 * uint(64);
const uint l_166 = l_165 + l_161;
const float l_167 = shared_memory_130[l_166];
const float l_168 = shared_memory_128[l_mut_162];
const float l_169 = l_168 - l_mut_88;
const float l_170 = exp(l_169);
const float l_171 = l_167 * l_170;
l_mut_55 = l_mut_55 + l_171;
}
}
const uint l_172 = l_25 + l_161;
const float l_173 = l_mut_55 * l_mut_39;
const half l_174 = half(l_173);
buffer_5[l_172] = l_174;
const uint l_175 = l_1 * uint(2);
const uint l_176 = l_175 + uint(1);
l_mut_55 = float(0.0);

for (uint l_mut_177 = uint(0); l_mut_177 < uint(4); ++l_mut_177) {
const float l_178 = shared_memory_129[l_mut_177];
const bool l_179 = l_178 > float(0.0);
if (l_179) {
const uint l_180 = l_mut_177 * uint(64);
const uint l_181 = l_180 + l_176;
const float l_182 = shared_memory_130[l_181];
const float l_183 = shared_memory_128[l_mut_177];
const float l_184 = l_183 - l_mut_88;
const float l_185 = exp(l_184);
const float l_186 = l_182 * l_185;
l_mut_55 = l_mut_55 + l_186;
}
}
const uint l_187 = l_25 + l_176;
const float l_188 = l_mut_55 * l_mut_39;
const half l_189 = half(l_188);
buffer_5[l_187] = l_189;
}

}
