#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[12];
};

[[kernel]]
void flash_attn_f_(
    const device bfloat * buffer_0 [[buffer(0)]],
    const device bfloat * buffer_1 [[buffer(1)]],
    const device bfloat * buffer_2 [[buffer(2)]],
    device bfloat * buffer_3 [[buffer(3)]],
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
threadgroup uchar dynamic_shared_mem[29152];
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_32 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[0]);
// Shared array size: 64, 256 bytes
threadgroup float* shared_memory_33 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[4096]);
// Shared array size: 2048, 8192 bytes
threadgroup float* shared_memory_34 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[4352]);
// Shared array size: 8, 32 bytes
threadgroup float* shared_memory_35 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[12544]);
// Shared array size: 8, 32 bytes
threadgroup float* shared_memory_36 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[12576]);
// Shared array size: 8, 32 bytes
threadgroup float* shared_memory_37 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[12608]);
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_74 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[12640]);
// Shared array size: 64, 128 bytes
threadgroup half* shared_memory_166 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[16736]);
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_167 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[16864]);
// Shared array size: 2048, 8192 bytes
threadgroup float* shared_memory_168 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[20960]);
uint l_mut_11;
float l_mut_49;
uint l_mut_60;
uint l_mut_64;
float l_mut_151;
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
l_mut_11 = uint(0);
const bool l_12 = l_4 > l_3;
if (l_12) {
const uint l_13 = l_4 - l_3;
l_mut_11 = l_13;
}
const uint l_14 = l_3 + uint(8);
const uint l_15 = l_14 - uint(1);
const uint l_16 = l_15 / uint(8);
const uint l_17 = l_5 * l_16;
const uint l_18 = l_1 / l_17;
const uint l_19 = l_18 * l_17;
const uint l_20 = l_1 - l_19;
const uint l_21 = l_20 / l_16;
const uint l_22 = l_20 % l_16;
const uint l_23 = l_5 / l_6;
const uint l_24 = l_21 / l_23;
const uint l_25 = l_18 * l_5;
const uint l_26 = l_25 + l_21;
const uint l_27 = l_26 * l_3;
const uint l_28 = l_27 * uint(256);
const uint l_29 = l_18 * l_8;
const uint l_30 = l_24 * l_9;
const uint l_31 = l_29 + l_30;

for (uint l_mut_38 = uint(0); l_mut_38 < uint(64); ++l_mut_38) {
const uint l_39 = thread_index_in_threadgroup * uint(64);
const uint l_40 = l_39 + l_mut_38;
shared_memory_34[l_40] = float(0.0);
}
const bool l_41 = thread_index_in_threadgroup < uint(8);
if (l_41) {
shared_memory_35[thread_index_in_threadgroup] = float(-3.3999999521443642e38);
shared_memory_36[thread_index_in_threadgroup] = float(0.0);
}

for (uint l_mut_42 = uint(0); l_mut_42 < uint(64); ++l_mut_42) {
const uint l_43 = thread_index_in_threadgroup * uint(64);
const uint l_44 = l_43 + l_mut_42;
const uint l_45 = l_44 / uint(256);
const uint l_46 = l_44 % uint(256);
const uint l_47 = l_22 * uint(8);
const uint l_48 = l_47 + l_45;
l_mut_49 = float(0.0);
const bool l_50 = l_48 < l_3;
if (l_50) {
const uint l_51 = l_48 * uint(256);
const uint l_52 = l_28 + l_51;
const uint l_53 = l_52 + l_46;
const bfloat l_54 = buffer_0[l_53];
const float l_55 = float(l_54);
l_mut_49 = l_55;
}
const half l_56 = half(l_mut_49);
shared_memory_32[l_44] = l_56;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_57 = l_4 + uint(8);
const uint l_58 = l_57 - uint(1);
const uint l_59 = l_58 / uint(8);
l_mut_60 = l_59;
const bool l_61 = l_7 == uint(1);
const bool l_62 = l_7 == uint(2);
const bool l_63 = l_61 || l_62;
if (l_63) {
l_mut_64 = uint(0);
const bool l_65 = l_7 == uint(2);
if (l_65) {
l_mut_64 = l_mut_11;
}
const uint l_66 = l_22 * uint(8);
const uint l_67 = l_mut_64 + l_66;
const uint l_68 = l_67 + uint(8);
const uint l_69 = l_68 - uint(1);
const uint l_70 = l_69 / uint(8);
const uint l_71 = l_70 + uint(1);
const bool l_72 = l_71 < l_59;
if (l_72) {
l_mut_60 = l_71;
}
}

for (uint l_mut_73 = uint(0); l_mut_73 < l_mut_60; ++l_mut_73) {
simdgroup_float8x8 frag_acc_89;

for (uint l_mut_75 = uint(0); l_mut_75 < uint(64); ++l_mut_75) {
const uint l_76 = thread_index_in_threadgroup * uint(64);
const uint l_77 = l_76 + l_mut_75;
const uint l_78 = l_77 / uint(256);
const uint l_79 = l_77 % uint(256);
const uint l_80 = l_mut_73 * uint(8);
const uint l_81 = l_80 + l_78;
l_mut_49 = float(0.0);
const bool l_82 = l_81 < l_4;
if (l_82) {
const uint l_83 = l_81 * l_10;
const uint l_84 = l_31 + l_83;
const uint l_85 = l_84 + l_79;
const bfloat l_86 = buffer_1[l_85];
const float l_87 = float(l_86);
l_mut_49 = l_87;
}
const half l_88 = half(l_mut_49);
shared_memory_74[l_77] = l_88;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
frag_acc_89 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_90 = uint(0); l_mut_90 < uint(32); ++l_mut_90) {
simdgroup_half8x8 frag_a_99;
simdgroup_half8x8 frag_b_108;
const uint l_91 = uint(0) * uint(8);
const uint l_92 = l_91 * uint(256);
const uint l_93 = l_mut_90 * uint(8);
const uint l_94 = l_92 + l_93;
const uint l_95 = 2048;
const uint l_96 = l_95 - uint(0);
const uint l_97 = uint(2048) - l_94;
const uint l_98 = l_94 + uint(0);
simdgroup_load(frag_a_99, shared_memory_32 + l_98, uint(256), 0, false);
const uint l_100 = uint(0) * uint(8);
const uint l_101 = l_100 * uint(256);
const uint l_102 = l_mut_90 * uint(8);
const uint l_103 = l_101 + l_102;
const uint l_104 = 2048;
const uint l_105 = l_104 - uint(0);
const uint l_106 = uint(2048) - l_103;
const uint l_107 = l_103 + uint(0);
simdgroup_load(frag_b_108, shared_memory_74 + l_107, uint(256), 0, true);
simdgroup_multiply_accumulate(frag_acc_89, frag_a_99, frag_b_108, frag_acc_89);
}
const uint l_109 = uint(0) * uint(8);
const uint l_110 = l_109 * uint(8);
const uint l_111 = uint(0) * uint(8);
const uint l_112 = l_110 + l_111;
const uint l_113 = 64;
const uint l_114 = l_113 - uint(0);
const uint l_115 = uint(64) - l_112;
const uint l_116 = l_112 + uint(0);
simdgroup_store(frag_acc_89, shared_memory_33 + l_116, uint(8));
simdgroup_barrier(mem_flags::mem_none);
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_117 = uint(0); l_mut_117 < uint(2); ++l_mut_117) {
const uint l_118 = thread_index_in_threadgroup * uint(2);
const uint l_119 = l_118 + l_mut_117;
const uint l_120 = l_119 / uint(8);
const uint l_121 = l_119 % uint(8);
const uint l_122 = l_22 * uint(8);
const uint l_123 = l_122 + l_120;
const uint l_124 = l_mut_73 * uint(8);
const uint l_125 = l_124 + l_121;
const bool l_126 = l_7 == uint(1);
const bool l_127 = l_125 > l_123;
const bool l_128 = l_126 && l_127;
const bool l_129 = l_7 == uint(2);
const uint l_130 = l_mut_11 + l_123;
const bool l_131 = l_125 > l_130;
const bool l_132 = l_129 && l_131;
const bool l_133 = l_128 || l_132;
const bool l_134 = l_123 < l_3;
const bool l_135 = l_125 < l_4;
const bool l_136 = l_134 && l_135;
const bool l_137 = !l_133;
const bool l_138 = l_136 && l_137;
if (l_138) {
const float l_139 = shared_memory_33[l_119];
const float l_140 = l_139 * l_2;
shared_memory_33[l_119] = l_140;
} else {
shared_memory_33[l_119] = float(-3.3999999521443642e38);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_141 = thread_index_in_threadgroup < uint(8);
if (l_141) {
const uint l_142 = l_22 * uint(8);
const uint l_143 = l_142 + thread_index_in_threadgroup;
const bool l_144 = l_143 < l_3;
if (l_144) {
l_mut_49 = float(-3.3999999521443642e38);

for (uint l_mut_145 = uint(0); l_mut_145 < uint(8); ++l_mut_145) {
const uint l_146 = thread_index_in_threadgroup * uint(8);
const uint l_147 = l_146 + l_mut_145;
const float l_148 = shared_memory_33[l_147];
const bool l_149 = l_148 > l_mut_49;
if (l_149) {
l_mut_49 = l_148;
}
}
const float l_150 = shared_memory_35[thread_index_in_threadgroup];
l_mut_151 = l_150;
const bool l_152 = l_mut_49 > l_mut_151;
if (l_152) {
l_mut_151 = l_mut_49;
}
const float l_153 = l_150 - l_mut_151;
const float l_154 = exp(l_153);
l_mut_49 = float(0.0);

for (uint l_mut_155 = uint(0); l_mut_155 < uint(8); ++l_mut_155) {
const uint l_156 = thread_index_in_threadgroup * uint(8);
const uint l_157 = l_156 + l_mut_155;
const float l_158 = shared_memory_33[l_157];
const float l_159 = l_158 - l_mut_151;
const float l_160 = exp(l_159);
const uint l_161 = thread_index_in_threadgroup * uint(8);
const uint l_162 = l_161 + l_mut_155;
shared_memory_33[l_162] = l_160;
l_mut_49 = l_mut_49 + l_160;
}
const float l_163 = shared_memory_36[thread_index_in_threadgroup];
const float l_164 = l_163 * l_154;
const float l_165 = l_164 + l_mut_49;
shared_memory_36[thread_index_in_threadgroup] = l_165;
shared_memory_35[thread_index_in_threadgroup] = l_mut_151;
shared_memory_37[thread_index_in_threadgroup] = l_154;
} else {
shared_memory_37[thread_index_in_threadgroup] = float(1.0);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_169 = uint(0); l_mut_169 < uint(2); ++l_mut_169) {
const uint l_170 = thread_index_in_threadgroup * uint(2);
const uint l_171 = l_170 + l_mut_169;
const uint l_172 = l_171 / uint(8);
const uint l_173 = l_22 * uint(8);
const uint l_174 = l_173 + l_172;
l_mut_151 = float(0.0);
const bool l_175 = l_174 < l_3;
if (l_175) {
const float l_176 = shared_memory_33[l_171];
l_mut_151 = l_176;
}
const half l_177 = half(l_mut_151);
shared_memory_166[l_171] = l_177;
}

for (uint l_mut_178 = uint(0); l_mut_178 < uint(64); ++l_mut_178) {
const uint l_179 = thread_index_in_threadgroup * uint(64);
const uint l_180 = l_179 + l_mut_178;
const uint l_181 = l_180 / uint(256);
const uint l_182 = l_180 % uint(256);
const uint l_183 = l_mut_73 * uint(8);
const uint l_184 = l_183 + l_181;
l_mut_151 = float(0.0);
const bool l_185 = l_184 < l_4;
if (l_185) {
const uint l_186 = l_184 * l_10;
const uint l_187 = l_31 + l_186;
const uint l_188 = l_187 + l_182;
const bfloat l_189 = buffer_2[l_188];
const float l_190 = float(l_189);
l_mut_151 = l_190;
}
const half l_191 = half(l_mut_151);
shared_memory_167[l_180] = l_191;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_192 = uint(0); l_mut_192 < uint(32); ++l_mut_192) {
simdgroup_float8x8 frag_acc_193;
simdgroup_half8x8 frag_a_202;
simdgroup_half8x8 frag_b_211;
frag_acc_193 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));
const uint l_194 = uint(0) * uint(8);
const uint l_195 = l_194 * uint(8);
const uint l_196 = uint(0) * uint(8);
const uint l_197 = l_195 + l_196;
const uint l_198 = 64;
const uint l_199 = l_198 - uint(0);
const uint l_200 = uint(64) - l_197;
const uint l_201 = l_197 + uint(0);
simdgroup_load(frag_a_202, shared_memory_166 + l_201, uint(8), 0, false);
const uint l_203 = uint(0) * uint(8);
const uint l_204 = l_203 * uint(256);
const uint l_205 = l_mut_192 * uint(8);
const uint l_206 = l_204 + l_205;
const uint l_207 = 2048;
const uint l_208 = l_207 - uint(0);
const uint l_209 = uint(2048) - l_206;
const uint l_210 = l_206 + uint(0);
simdgroup_load(frag_b_211, shared_memory_167 + l_210, uint(256), 0, false);
simdgroup_multiply_accumulate(frag_acc_193, frag_a_202, frag_b_211, frag_acc_193);
const uint l_212 = uint(0) * uint(8);
const uint l_213 = l_212 * uint(256);
const uint l_214 = l_mut_192 * uint(8);
const uint l_215 = l_213 + l_214;
const uint l_216 = 2048;
const uint l_217 = l_216 - uint(0);
const uint l_218 = uint(2048) - l_215;
const uint l_219 = l_215 + uint(0);
simdgroup_store(frag_acc_193, shared_memory_168 + l_219, uint(256));
simdgroup_barrier(mem_flags::mem_none);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_220 = uint(0); l_mut_220 < uint(64); ++l_mut_220) {
const uint l_221 = thread_index_in_threadgroup * uint(64);
const uint l_222 = l_221 + l_mut_220;
const uint l_223 = l_222 / uint(256);
const float l_224 = shared_memory_34[l_222];
const float l_225 = shared_memory_37[l_223];
const float l_226 = l_224 * l_225;
const float l_227 = shared_memory_168[l_222];
const float l_228 = l_226 + l_227;
shared_memory_34[l_222] = l_228;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint l_mut_229 = uint(0); l_mut_229 < uint(64); ++l_mut_229) {
const uint l_230 = thread_index_in_threadgroup * uint(64);
const uint l_231 = l_230 + l_mut_229;
const uint l_232 = l_231 / uint(256);
const uint l_233 = l_231 % uint(256);
const uint l_234 = l_22 * uint(8);
const uint l_235 = l_234 + l_232;
const bool l_236 = l_235 < l_3;
if (l_236) {
const uint l_237 = l_235 * uint(256);
const uint l_238 = l_28 + l_237;
const uint l_239 = l_238 + l_233;
const float l_240 = shared_memory_34[l_231];
const float l_241 = shared_memory_36[l_232];
const float l_242 = l_240 / l_241;
const bfloat l_243 = bfloat(l_242);
buffer_3[l_239] = l_243;
}
}

}
