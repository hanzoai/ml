#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[12];
};

[[kernel]]
void flash_attn_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    const device float * buffer_2 [[buffer(2)]],
    device float * buffer_3 [[buffer(3)]],
    const device float * buffer_4 [[buffer(4)]],
    const device uint * buffer_5 [[buffer(5)]],
    constant info_st& info [[buffer(6)]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
threadgroup uchar dynamic_shared_mem[16064];
// Shared array size: 1024, 2048 bytes
threadgroup half* shared_memory_29 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[0]);
// Shared array size: 256, 1024 bytes
threadgroup float* shared_memory_30 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[2048]);
// Shared array size: 1024, 4096 bytes
threadgroup float* shared_memory_31 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[3072]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_32 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[7168]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_33 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[7232]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_34 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[7296]);
// Shared array size: 1024, 2048 bytes
threadgroup half* shared_memory_66 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[7360]);
// Shared array size: 256, 512 bytes
threadgroup half* shared_memory_154 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[9408]);
// Shared array size: 1024, 2048 bytes
threadgroup half* shared_memory_155 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[9920]);
// Shared array size: 1024, 4096 bytes
threadgroup float* shared_memory_156 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[11968]);
float l_mut_51;
uint l_mut_63;
uint l_mut_64;
float l_mut_139;
const uint l_0 = buffer_5[uint(8)];
const uint l_1 = threadgroup_index_in_grid + l_0;
const float l_2 = buffer_4[uint(0)];
const uint l_3 = buffer_5[uint(0)];
const uint l_4 = buffer_5[uint(1)];
const uint l_5 = buffer_5[uint(2)];
const uint l_6 = buffer_5[uint(3)];
const uint l_7 = buffer_5[uint(4)];
const uint l_8 = buffer_5[uint(5)];
const uint l_9 = buffer_5[uint(6)];
const uint l_10 = buffer_5[uint(7)];
const uint l_11 = l_3 + uint(16);
const uint l_12 = l_11 - uint(1);
const uint l_13 = l_12 / uint(16);
const uint l_14 = l_5 * l_13;
const uint l_15 = l_1 / l_14;
const uint l_16 = l_15 * l_14;
const uint l_17 = l_1 - l_16;
const uint l_18 = l_17 / l_13;
const uint l_19 = l_17 % l_13;
const uint l_20 = l_5 / l_6;
const uint l_21 = l_18 / l_20;
const uint l_22 = l_15 * l_5;
const uint l_23 = l_22 + l_18;
const uint l_24 = l_23 * l_3;
const uint l_25 = l_24 * uint(64);
const uint l_26 = l_15 * l_8;
const uint l_27 = l_21 * l_9;
const uint l_28 = l_26 + l_27;

for (uint l_mut_35 = uint(0); l_mut_35 < uint(32); ++l_mut_35) {
const uint l_36 = thread_index_in_threadgroup * uint(32);
const uint l_37 = l_36 + l_mut_35;
shared_memory_31[l_37] = float(0.0);
}
const bool l_38 = thread_index_in_threadgroup < uint(16);
if (l_38) {
shared_memory_32[thread_index_in_threadgroup] = float(-3.3999999521443642e38);
shared_memory_33[thread_index_in_threadgroup] = float(0.0);
}

for (uint l_mut_39 = uint(0); l_mut_39 < uint(32); ++l_mut_39) {
const uint l_40 = thread_index_in_threadgroup * uint(32);
const uint l_41 = l_40 + l_mut_39;
const uint l_42 = l_41 / uint(64);
const uint l_43 = l_41 % uint(64);
const uint l_44 = l_19 * uint(16);
const uint l_45 = l_44 + l_42;
const bool l_46 = l_45 < l_3;
if (l_46) {
const uint l_47 = l_45 * uint(64);
const uint l_48 = l_25 + l_47;
const uint l_49 = l_48 + l_43;
const float l_50 = buffer_0[l_49];
l_mut_51 = l_50;
} else {
l_mut_51 = float(0.0);
}
const half l_52 = half(l_mut_51);
shared_memory_29[l_41] = l_52;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_53 = l_4 + uint(16);
const uint l_54 = l_53 - uint(1);
const uint l_55 = l_54 / uint(16);
const bool l_56 = l_7 == uint(1);
if (l_56) {
const uint l_57 = l_19 * uint(16);
const uint l_58 = l_57 + uint(16);
const uint l_59 = l_58 - uint(1);
const uint l_60 = l_59 / uint(16);
const uint l_61 = l_60 + uint(1);
const bool l_62 = l_61 < l_55;
if (l_62) {
l_mut_63 = l_61;
} else {
l_mut_63 = l_55;
}
l_mut_64 = l_mut_63;
} else {
l_mut_64 = l_55;
}

for (uint l_mut_65 = uint(0); l_mut_65 < l_mut_64; ++l_mut_65) {

for (uint l_mut_67 = uint(0); l_mut_67 < uint(32); ++l_mut_67) {
const uint l_68 = thread_index_in_threadgroup * uint(32);
const uint l_69 = l_68 + l_mut_67;
const uint l_70 = l_69 / uint(64);
const uint l_71 = l_69 % uint(64);
const uint l_72 = l_mut_65 * uint(16);
const uint l_73 = l_72 + l_70;
const bool l_74 = l_73 < l_4;
if (l_74) {
const uint l_75 = l_73 * l_10;
const uint l_76 = l_28 + l_75;
const uint l_77 = l_76 + l_71;
const float l_78 = buffer_1[l_77];
l_mut_51 = l_78;
} else {
l_mut_51 = float(0.0);
}
const half l_79 = half(l_mut_51);
shared_memory_66[l_69] = l_79;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_80 = uint(0); l_mut_80 < uint(2); ++l_mut_80) {

for (uint l_mut_81 = uint(0); l_mut_81 < uint(2); ++l_mut_81) {
simdgroup_float8x8 frag_acc_82;
frag_acc_82 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_83 = uint(0); l_mut_83 < uint(8); ++l_mut_83) {
simdgroup_half8x8 frag_a_92;
simdgroup_half8x8 frag_b_101;
const uint l_84 = l_mut_80 * uint(8);
const uint l_85 = l_84 * uint(64);
const uint l_86 = l_mut_83 * uint(8);
const uint l_87 = l_85 + l_86;
const uint l_88 = 1024;
const uint l_89 = l_88 - uint(0);
const uint l_90 = uint(1024) - l_87;
const uint l_91 = l_87 + uint(0);
simdgroup_load(frag_a_92, shared_memory_29 + l_91, uint(64), 0, false);
const uint l_93 = l_mut_81 * uint(8);
const uint l_94 = l_93 * uint(64);
const uint l_95 = l_mut_83 * uint(8);
const uint l_96 = l_94 + l_95;
const uint l_97 = 1024;
const uint l_98 = l_97 - uint(0);
const uint l_99 = uint(1024) - l_96;
const uint l_100 = l_96 + uint(0);
simdgroup_load(frag_b_101, shared_memory_66 + l_100, uint(64), 0, true);
simdgroup_multiply_accumulate(frag_acc_82, frag_a_92, frag_b_101, frag_acc_82);
}
const uint l_102 = l_mut_80 * uint(8);
const uint l_103 = l_102 * uint(16);
const uint l_104 = l_mut_81 * uint(8);
const uint l_105 = l_103 + l_104;
const uint l_106 = 256;
const uint l_107 = l_106 - uint(0);
const uint l_108 = uint(256) - l_105;
const uint l_109 = l_105 + uint(0);
simdgroup_store(frag_acc_82, shared_memory_30 + l_109, uint(16));
simdgroup_barrier(mem_flags::mem_none);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_110 = uint(0); l_mut_110 < uint(8); ++l_mut_110) {
const uint l_111 = thread_index_in_threadgroup * uint(8);
const uint l_112 = l_111 + l_mut_110;
const uint l_113 = l_112 / uint(16);
const uint l_114 = l_112 % uint(16);
const uint l_115 = l_19 * uint(16);
const uint l_116 = l_115 + l_113;
const uint l_117 = l_mut_65 * uint(16);
const uint l_118 = l_117 + l_114;
const bool l_119 = l_7 == uint(1);
const bool l_120 = l_118 > l_116;
const bool l_121 = l_119 && l_120;
const bool l_122 = l_116 < l_3;
const bool l_123 = l_118 < l_4;
const bool l_124 = l_122 && l_123;
const bool l_125 = !l_121;
const bool l_126 = l_124 && l_125;
if (l_126) {
const float l_127 = shared_memory_30[l_112];
const float l_128 = l_127 * l_2;
shared_memory_30[l_112] = l_128;
} else {
shared_memory_30[l_112] = float(-3.3999999521443642e38);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_129 = thread_index_in_threadgroup < uint(16);
if (l_129) {
const uint l_130 = l_19 * uint(16);
const uint l_131 = l_130 + thread_index_in_threadgroup;
const bool l_132 = l_131 < l_3;
if (l_132) {
l_mut_51 = float(-3.3999999521443642e38);

for (uint l_mut_133 = uint(0); l_mut_133 < uint(16); ++l_mut_133) {
const uint l_134 = thread_index_in_threadgroup * uint(16);
const uint l_135 = l_134 + l_mut_133;
const float l_136 = shared_memory_30[l_135];
const bool l_137 = l_136 > l_mut_51;
if (l_137) {
l_mut_51 = l_136;
}
}
const float l_138 = shared_memory_32[thread_index_in_threadgroup];
l_mut_139 = l_138;
const bool l_140 = l_mut_51 > l_mut_139;
if (l_140) {
l_mut_139 = l_mut_51;
}
const float l_141 = l_138 - l_mut_139;
const float l_142 = exp(l_141);
l_mut_51 = float(0.0);

for (uint l_mut_143 = uint(0); l_mut_143 < uint(16); ++l_mut_143) {
const uint l_144 = thread_index_in_threadgroup * uint(16);
const uint l_145 = l_144 + l_mut_143;
const float l_146 = shared_memory_30[l_145];
const float l_147 = l_146 - l_mut_139;
const float l_148 = exp(l_147);
const uint l_149 = thread_index_in_threadgroup * uint(16);
const uint l_150 = l_149 + l_mut_143;
shared_memory_30[l_150] = l_148;
l_mut_51 = l_mut_51 + l_148;
}
const float l_151 = shared_memory_33[thread_index_in_threadgroup];
const float l_152 = l_151 * l_142;
const float l_153 = l_152 + l_mut_51;
shared_memory_33[thread_index_in_threadgroup] = l_153;
shared_memory_32[thread_index_in_threadgroup] = l_mut_139;
shared_memory_34[thread_index_in_threadgroup] = l_142;
} else {
shared_memory_34[thread_index_in_threadgroup] = float(1.0);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_157 = uint(0); l_mut_157 < uint(8); ++l_mut_157) {
const uint l_158 = thread_index_in_threadgroup * uint(8);
const uint l_159 = l_158 + l_mut_157;
const uint l_160 = l_159 / uint(16);
const uint l_161 = l_19 * uint(16);
const uint l_162 = l_161 + l_160;
const bool l_163 = l_162 < l_3;
if (l_163) {
const float l_164 = shared_memory_30[l_159];
l_mut_139 = l_164;
} else {
l_mut_139 = float(0.0);
}
const half l_165 = half(l_mut_139);
shared_memory_154[l_159] = l_165;
}

for (uint l_mut_166 = uint(0); l_mut_166 < uint(32); ++l_mut_166) {
const uint l_167 = thread_index_in_threadgroup * uint(32);
const uint l_168 = l_167 + l_mut_166;
const uint l_169 = l_168 / uint(64);
const uint l_170 = l_168 % uint(64);
const uint l_171 = l_mut_65 * uint(16);
const uint l_172 = l_171 + l_169;
const bool l_173 = l_172 < l_4;
if (l_173) {
const uint l_174 = l_172 * l_10;
const uint l_175 = l_28 + l_174;
const uint l_176 = l_175 + l_170;
const float l_177 = buffer_2[l_176];
l_mut_139 = l_177;
} else {
l_mut_139 = float(0.0);
}
const half l_178 = half(l_mut_139);
shared_memory_155[l_168] = l_178;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_179 = uint(0); l_mut_179 < uint(2); ++l_mut_179) {

for (uint l_mut_180 = uint(0); l_mut_180 < uint(8); ++l_mut_180) {
simdgroup_float8x8 frag_acc_181;
frag_acc_181 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_182 = uint(0); l_mut_182 < uint(2); ++l_mut_182) {
simdgroup_half8x8 frag_a_191;
simdgroup_half8x8 frag_b_200;
const uint l_183 = l_mut_179 * uint(8);
const uint l_184 = l_183 * uint(16);
const uint l_185 = l_mut_182 * uint(8);
const uint l_186 = l_184 + l_185;
const uint l_187 = 256;
const uint l_188 = l_187 - uint(0);
const uint l_189 = uint(256) - l_186;
const uint l_190 = l_186 + uint(0);
simdgroup_load(frag_a_191, shared_memory_154 + l_190, uint(16), 0, false);
const uint l_192 = l_mut_182 * uint(8);
const uint l_193 = l_192 * uint(64);
const uint l_194 = l_mut_180 * uint(8);
const uint l_195 = l_193 + l_194;
const uint l_196 = 1024;
const uint l_197 = l_196 - uint(0);
const uint l_198 = uint(1024) - l_195;
const uint l_199 = l_195 + uint(0);
simdgroup_load(frag_b_200, shared_memory_155 + l_199, uint(64), 0, false);
simdgroup_multiply_accumulate(frag_acc_181, frag_a_191, frag_b_200, frag_acc_181);
}
const uint l_201 = l_mut_179 * uint(8);
const uint l_202 = l_201 * uint(64);
const uint l_203 = l_mut_180 * uint(8);
const uint l_204 = l_202 + l_203;
const uint l_205 = 1024;
const uint l_206 = l_205 - uint(0);
const uint l_207 = uint(1024) - l_204;
const uint l_208 = l_204 + uint(0);
simdgroup_store(frag_acc_181, shared_memory_156 + l_208, uint(64));
simdgroup_barrier(mem_flags::mem_none);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_209 = uint(0); l_mut_209 < uint(32); ++l_mut_209) {
const uint l_210 = thread_index_in_threadgroup * uint(32);
const uint l_211 = l_210 + l_mut_209;
const uint l_212 = l_211 / uint(64);
const float l_213 = shared_memory_31[l_211];
const float l_214 = shared_memory_34[l_212];
const float l_215 = l_213 * l_214;
const float l_216 = shared_memory_156[l_211];
const float l_217 = l_215 + l_216;
shared_memory_31[l_211] = l_217;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint l_mut_218 = uint(0); l_mut_218 < uint(32); ++l_mut_218) {
const uint l_219 = thread_index_in_threadgroup * uint(32);
const uint l_220 = l_219 + l_mut_218;
const uint l_221 = l_220 / uint(64);
const uint l_222 = l_220 % uint(64);
const uint l_223 = l_19 * uint(16);
const uint l_224 = l_223 + l_221;
const bool l_225 = l_224 < l_3;
if (l_225) {
const uint l_226 = l_224 * uint(64);
const uint l_227 = l_25 + l_226;
const uint l_228 = l_227 + l_222;
const float l_229 = shared_memory_31[l_220];
const float l_230 = shared_memory_33[l_221];
const float l_231 = l_229 / l_230;
buffer_3[l_228] = l_231;
}
}

}
