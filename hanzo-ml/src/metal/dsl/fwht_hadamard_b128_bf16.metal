#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void fwht_hadamard_f_(
    const device bfloat * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device bfloat * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    constant info_st& info [[buffer(4)]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]]
) {
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
threadgroup uchar dynamic_shared_mem[512];
// Shared array size: 128, 512 bytes
threadgroup float* shared_memory_9 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
uint l_mut_6;
float l_mut_15;
float l_mut_20;
float l_mut_25;
float l_mut_30;
const uint l_0 = buffer_3[uint(0)];
const bool l_1 = threadgroup_index_in_grid < l_0;
if (l_1) {
const uint l_2 = buffer_3[uint(1)];
const uint l_3 = buffer_3[uint(2)];
const uint l_4 = buffer_3[uint(3)];
const uint l_5 = threadgroup_index_in_grid * uint(128);
l_mut_6 = uint(0);
const bool l_7 = l_3 != uint(0);
if (l_7) {
const uint l_8 = l_5 % l_2;
l_mut_6 = l_8;
}
const uint l_10 = thread_index_in_threadgroup + uint(0);
const uint l_11 = l_10 * uint(4);
const uint l_12 = l_5 + l_11;
const bfloat l_13 = buffer_0[l_12];
const float l_14 = float(l_13);
l_mut_15 = l_14;
const uint l_16 = l_5 + l_11;
const uint l_17 = l_16 + uint(1);
const bfloat l_18 = buffer_0[l_17];
const float l_19 = float(l_18);
l_mut_20 = l_19;
const uint l_21 = l_5 + l_11;
const uint l_22 = l_21 + uint(2);
const bfloat l_23 = buffer_0[l_22];
const float l_24 = float(l_23);
l_mut_25 = l_24;
const uint l_26 = l_5 + l_11;
const uint l_27 = l_26 + uint(3);
const bfloat l_28 = buffer_0[l_27];
const float l_29 = float(l_28);
l_mut_30 = l_29;
const bool l_31 = l_3 != uint(0);
const bool l_32 = l_4 == uint(0);
const bool l_33 = l_31 && l_32;
if (l_33) {
const uint l_34 = l_mut_6 + l_11;
const float l_35 = buffer_1[l_34];
l_mut_15 = l_mut_15 * l_35;
const uint l_36 = l_mut_6 + l_11;
const uint l_37 = l_36 + uint(1);
const float l_38 = buffer_1[l_37];
l_mut_20 = l_mut_20 * l_38;
const uint l_39 = l_mut_6 + l_11;
const uint l_40 = l_39 + uint(2);
const float l_41 = buffer_1[l_40];
l_mut_25 = l_mut_25 * l_41;
const uint l_42 = l_mut_6 + l_11;
const uint l_43 = l_42 + uint(3);
const float l_44 = buffer_1[l_43];
l_mut_30 = l_mut_30 * l_44;
}
const float l_45 = l_mut_15 + l_mut_20;
const float l_46 = l_mut_15 - l_mut_20;
const float l_47 = l_mut_25 + l_mut_30;
const float l_48 = l_mut_25 - l_mut_30;
const float l_49 = l_45 + l_47;
shared_memory_9[l_11] = l_49;
const uint l_50 = l_11 + uint(1);
const float l_51 = l_46 + l_48;
shared_memory_9[l_50] = l_51;
const uint l_52 = l_11 + uint(2);
const float l_53 = l_45 - l_47;
shared_memory_9[l_52] = l_53;
const uint l_54 = l_11 + uint(3);
const float l_55 = l_46 - l_48;
shared_memory_9[l_54] = l_55;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_56 = thread_index_in_threadgroup + uint(0);
const uint l_57 = l_56 >> uint(2);
const uint l_58 = l_56 & uint(3);
const uint l_59 = l_57 * uint(8);
const uint l_60 = l_59 + l_58;
const uint l_61 = l_60 + uint(4);
const float l_62 = shared_memory_9[l_60];
const float l_63 = shared_memory_9[l_61];
const float l_64 = l_62 + l_63;
shared_memory_9[l_60] = l_64;
const float l_65 = l_62 - l_63;
shared_memory_9[l_61] = l_65;
const uint l_66 = thread_index_in_threadgroup + uint(32);
const uint l_67 = l_66 >> uint(2);
const uint l_68 = l_66 & uint(3);
const uint l_69 = l_67 * uint(8);
const uint l_70 = l_69 + l_68;
const uint l_71 = l_70 + uint(4);
const float l_72 = shared_memory_9[l_70];
const float l_73 = shared_memory_9[l_71];
const float l_74 = l_72 + l_73;
shared_memory_9[l_70] = l_74;
const float l_75 = l_72 - l_73;
shared_memory_9[l_71] = l_75;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_76 = thread_index_in_threadgroup + uint(0);
const uint l_77 = l_76 >> uint(3);
const uint l_78 = l_76 & uint(7);
const uint l_79 = l_77 * uint(16);
const uint l_80 = l_79 + l_78;
const uint l_81 = l_80 + uint(8);
const float l_82 = shared_memory_9[l_80];
const float l_83 = shared_memory_9[l_81];
const float l_84 = l_82 + l_83;
shared_memory_9[l_80] = l_84;
const float l_85 = l_82 - l_83;
shared_memory_9[l_81] = l_85;
const uint l_86 = thread_index_in_threadgroup + uint(32);
const uint l_87 = l_86 >> uint(3);
const uint l_88 = l_86 & uint(7);
const uint l_89 = l_87 * uint(16);
const uint l_90 = l_89 + l_88;
const uint l_91 = l_90 + uint(8);
const float l_92 = shared_memory_9[l_90];
const float l_93 = shared_memory_9[l_91];
const float l_94 = l_92 + l_93;
shared_memory_9[l_90] = l_94;
const float l_95 = l_92 - l_93;
shared_memory_9[l_91] = l_95;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_96 = thread_index_in_threadgroup + uint(0);
const uint l_97 = l_96 >> uint(4);
const uint l_98 = l_96 & uint(15);
const uint l_99 = l_97 * uint(32);
const uint l_100 = l_99 + l_98;
const uint l_101 = l_100 + uint(16);
const float l_102 = shared_memory_9[l_100];
const float l_103 = shared_memory_9[l_101];
const float l_104 = l_102 + l_103;
shared_memory_9[l_100] = l_104;
const float l_105 = l_102 - l_103;
shared_memory_9[l_101] = l_105;
const uint l_106 = thread_index_in_threadgroup + uint(32);
const uint l_107 = l_106 >> uint(4);
const uint l_108 = l_106 & uint(15);
const uint l_109 = l_107 * uint(32);
const uint l_110 = l_109 + l_108;
const uint l_111 = l_110 + uint(16);
const float l_112 = shared_memory_9[l_110];
const float l_113 = shared_memory_9[l_111];
const float l_114 = l_112 + l_113;
shared_memory_9[l_110] = l_114;
const float l_115 = l_112 - l_113;
shared_memory_9[l_111] = l_115;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_116 = thread_index_in_threadgroup + uint(0);
const uint l_117 = l_116 >> uint(5);
const uint l_118 = l_116 & uint(31);
const uint l_119 = l_117 * uint(64);
const uint l_120 = l_119 + l_118;
const uint l_121 = l_120 + uint(32);
const float l_122 = shared_memory_9[l_120];
const float l_123 = shared_memory_9[l_121];
const float l_124 = l_122 + l_123;
shared_memory_9[l_120] = l_124;
const float l_125 = l_122 - l_123;
shared_memory_9[l_121] = l_125;
const uint l_126 = thread_index_in_threadgroup + uint(32);
const uint l_127 = l_126 >> uint(5);
const uint l_128 = l_126 & uint(31);
const uint l_129 = l_127 * uint(64);
const uint l_130 = l_129 + l_128;
const uint l_131 = l_130 + uint(32);
const float l_132 = shared_memory_9[l_130];
const float l_133 = shared_memory_9[l_131];
const float l_134 = l_132 + l_133;
shared_memory_9[l_130] = l_134;
const float l_135 = l_132 - l_133;
shared_memory_9[l_131] = l_135;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_136 = thread_index_in_threadgroup + uint(0);
const uint l_137 = l_136 >> uint(6);
const uint l_138 = l_136 & uint(63);
const uint l_139 = l_137 * uint(128);
const uint l_140 = l_139 + l_138;
const uint l_141 = l_140 + uint(64);
const float l_142 = shared_memory_9[l_140];
const float l_143 = shared_memory_9[l_141];
const float l_144 = l_142 + l_143;
shared_memory_9[l_140] = l_144;
const float l_145 = l_142 - l_143;
shared_memory_9[l_141] = l_145;
const uint l_146 = thread_index_in_threadgroup + uint(32);
const uint l_147 = l_146 >> uint(6);
const uint l_148 = l_146 & uint(63);
const uint l_149 = l_147 * uint(128);
const uint l_150 = l_149 + l_148;
const uint l_151 = l_150 + uint(64);
const float l_152 = shared_memory_9[l_150];
const float l_153 = shared_memory_9[l_151];
const float l_154 = l_152 + l_153;
shared_memory_9[l_150] = l_154;
const float l_155 = l_152 - l_153;
shared_memory_9[l_151] = l_155;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_156 = thread_index_in_threadgroup + uint(0);
const uint l_157 = l_156 * uint(4);
const float l_158 = shared_memory_9[l_157];
const float l_159 = l_158 * float(0.0883883461356163);
l_mut_30 = l_159;
const uint l_160 = l_157 + uint(1);
const float l_161 = shared_memory_9[l_160];
const float l_162 = l_161 * float(0.0883883461356163);
l_mut_25 = l_162;
const uint l_163 = l_157 + uint(2);
const float l_164 = shared_memory_9[l_163];
const float l_165 = l_164 * float(0.0883883461356163);
l_mut_20 = l_165;
const uint l_166 = l_157 + uint(3);
const float l_167 = shared_memory_9[l_166];
const float l_168 = l_167 * float(0.0883883461356163);
l_mut_15 = l_168;
const bool l_169 = l_3 != uint(0);
const bool l_170 = l_4 != uint(0);
const bool l_171 = l_169 && l_170;
if (l_171) {
const uint l_172 = l_mut_6 + l_157;
const float l_173 = buffer_1[l_172];
l_mut_30 = l_mut_30 * l_173;
const uint l_174 = l_mut_6 + l_157;
const uint l_175 = l_174 + uint(1);
const float l_176 = buffer_1[l_175];
l_mut_25 = l_mut_25 * l_176;
const uint l_177 = l_mut_6 + l_157;
const uint l_178 = l_177 + uint(2);
const float l_179 = buffer_1[l_178];
l_mut_20 = l_mut_20 * l_179;
const uint l_180 = l_mut_6 + l_157;
const uint l_181 = l_180 + uint(3);
const float l_182 = buffer_1[l_181];
l_mut_15 = l_mut_15 * l_182;
}
const uint l_183 = l_5 + l_157;
const bfloat l_184 = bfloat(l_mut_30);
buffer_2[l_183] = l_184;
const uint l_185 = l_5 + l_157;
const uint l_186 = l_185 + uint(1);
const bfloat l_187 = bfloat(l_mut_25);
buffer_2[l_186] = l_187;
const uint l_188 = l_5 + l_157;
const uint l_189 = l_188 + uint(2);
const bfloat l_190 = bfloat(l_mut_20);
buffer_2[l_189] = l_190;
const uint l_191 = l_5 + l_157;
const uint l_192 = l_191 + uint(3);
const bfloat l_193 = bfloat(l_mut_15);
buffer_2[l_192] = l_193;
}

}
