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
threadgroup uchar dynamic_shared_mem[30400];
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_32 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[0]);
// Shared array size: 256, 1024 bytes
threadgroup float* shared_memory_33 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[4096]);
// Shared array size: 2048, 8192 bytes
threadgroup float* shared_memory_34 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[5120]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_35 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[13312]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_36 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[13376]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_37 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[13440]);
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_74 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[13504]);
// Shared array size: 256, 512 bytes
threadgroup half* shared_memory_168 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[17600]);
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_169 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[18112]);
// Shared array size: 2048, 8192 bytes
threadgroup float* shared_memory_170 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[22208]);
uint l_mut_11;
float l_mut_49;
uint l_mut_60;
uint l_mut_64;
float l_mut_153;
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
const uint l_14 = l_3 + uint(16);
const uint l_15 = l_14 - uint(1);
const uint l_16 = l_15 / uint(16);
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
const uint l_28 = l_27 * uint(128);
const uint l_29 = l_18 * l_8;
const uint l_30 = l_24 * l_9;
const uint l_31 = l_29 + l_30;

for (uint l_mut_38 = uint(0); l_mut_38 < uint(64); ++l_mut_38) {
const uint l_39 = thread_index_in_threadgroup * uint(64);
const uint l_40 = l_39 + l_mut_38;
shared_memory_34[l_40] = float(0.0);
}
const bool l_41 = thread_index_in_threadgroup < uint(16);
if (l_41) {
shared_memory_35[thread_index_in_threadgroup] = float(-3.3999999521443642e38);
shared_memory_36[thread_index_in_threadgroup] = float(0.0);
}

for (uint l_mut_42 = uint(0); l_mut_42 < uint(64); ++l_mut_42) {
const uint l_43 = thread_index_in_threadgroup * uint(64);
const uint l_44 = l_43 + l_mut_42;
const uint l_45 = l_44 / uint(128);
const uint l_46 = l_44 % uint(128);
const uint l_47 = l_22 * uint(16);
const uint l_48 = l_47 + l_45;
l_mut_49 = float(0.0);
const bool l_50 = l_48 < l_3;
if (l_50) {
const uint l_51 = l_48 * uint(128);
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
const uint l_57 = l_4 + uint(16);
const uint l_58 = l_57 - uint(1);
const uint l_59 = l_58 / uint(16);
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
const uint l_66 = l_22 * uint(16);
const uint l_67 = l_mut_64 + l_66;
const uint l_68 = l_67 + uint(16);
const uint l_69 = l_68 - uint(1);
const uint l_70 = l_69 / uint(16);
const uint l_71 = l_70 + uint(1);
const bool l_72 = l_71 < l_59;
if (l_72) {
l_mut_60 = l_71;
}
}

for (uint l_mut_73 = uint(0); l_mut_73 < l_mut_60; ++l_mut_73) {

for (uint l_mut_75 = uint(0); l_mut_75 < uint(64); ++l_mut_75) {
const uint l_76 = thread_index_in_threadgroup * uint(64);
const uint l_77 = l_76 + l_mut_75;
const uint l_78 = l_77 / uint(128);
const uint l_79 = l_77 % uint(128);
const uint l_80 = l_mut_73 * uint(16);
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

for (uint l_mut_89 = uint(0); l_mut_89 < uint(2); ++l_mut_89) {

for (uint l_mut_90 = uint(0); l_mut_90 < uint(2); ++l_mut_90) {
simdgroup_float8x8 frag_acc_91;
frag_acc_91 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_92 = uint(0); l_mut_92 < uint(16); ++l_mut_92) {
simdgroup_half8x8 frag_a_101;
simdgroup_half8x8 frag_b_110;
const uint l_93 = l_mut_89 * uint(8);
const uint l_94 = l_93 * uint(128);
const uint l_95 = l_mut_92 * uint(8);
const uint l_96 = l_94 + l_95;
const uint l_97 = 2048;
const uint l_98 = l_97 - uint(0);
const uint l_99 = uint(2048) - l_96;
const uint l_100 = l_96 + uint(0);
simdgroup_load(frag_a_101, shared_memory_32 + l_100, uint(128), 0, false);
const uint l_102 = l_mut_90 * uint(8);
const uint l_103 = l_102 * uint(128);
const uint l_104 = l_mut_92 * uint(8);
const uint l_105 = l_103 + l_104;
const uint l_106 = 2048;
const uint l_107 = l_106 - uint(0);
const uint l_108 = uint(2048) - l_105;
const uint l_109 = l_105 + uint(0);
simdgroup_load(frag_b_110, shared_memory_74 + l_109, uint(128), 0, true);
simdgroup_multiply_accumulate(frag_acc_91, frag_a_101, frag_b_110, frag_acc_91);
}
const uint l_111 = l_mut_89 * uint(8);
const uint l_112 = l_111 * uint(16);
const uint l_113 = l_mut_90 * uint(8);
const uint l_114 = l_112 + l_113;
const uint l_115 = 256;
const uint l_116 = l_115 - uint(0);
const uint l_117 = uint(256) - l_114;
const uint l_118 = l_114 + uint(0);
simdgroup_store(frag_acc_91, shared_memory_33 + l_118, uint(16));
simdgroup_barrier(mem_flags::mem_none);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_119 = uint(0); l_mut_119 < uint(8); ++l_mut_119) {
const uint l_120 = thread_index_in_threadgroup * uint(8);
const uint l_121 = l_120 + l_mut_119;
const uint l_122 = l_121 / uint(16);
const uint l_123 = l_121 % uint(16);
const uint l_124 = l_22 * uint(16);
const uint l_125 = l_124 + l_122;
const uint l_126 = l_mut_73 * uint(16);
const uint l_127 = l_126 + l_123;
const bool l_128 = l_7 == uint(1);
const bool l_129 = l_127 > l_125;
const bool l_130 = l_128 && l_129;
const bool l_131 = l_7 == uint(2);
const uint l_132 = l_mut_11 + l_125;
const bool l_133 = l_127 > l_132;
const bool l_134 = l_131 && l_133;
const bool l_135 = l_130 || l_134;
const bool l_136 = l_125 < l_3;
const bool l_137 = l_127 < l_4;
const bool l_138 = l_136 && l_137;
const bool l_139 = !l_135;
const bool l_140 = l_138 && l_139;
if (l_140) {
const float l_141 = shared_memory_33[l_121];
const float l_142 = l_141 * l_2;
shared_memory_33[l_121] = l_142;
} else {
shared_memory_33[l_121] = float(-3.3999999521443642e38);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_143 = thread_index_in_threadgroup < uint(16);
if (l_143) {
const uint l_144 = l_22 * uint(16);
const uint l_145 = l_144 + thread_index_in_threadgroup;
const bool l_146 = l_145 < l_3;
if (l_146) {
l_mut_49 = float(-3.3999999521443642e38);

for (uint l_mut_147 = uint(0); l_mut_147 < uint(16); ++l_mut_147) {
const uint l_148 = thread_index_in_threadgroup * uint(16);
const uint l_149 = l_148 + l_mut_147;
const float l_150 = shared_memory_33[l_149];
const bool l_151 = l_150 > l_mut_49;
if (l_151) {
l_mut_49 = l_150;
}
}
const float l_152 = shared_memory_35[thread_index_in_threadgroup];
l_mut_153 = l_152;
const bool l_154 = l_mut_49 > l_mut_153;
if (l_154) {
l_mut_153 = l_mut_49;
}
const float l_155 = l_152 - l_mut_153;
const float l_156 = exp(l_155);
l_mut_49 = float(0.0);

for (uint l_mut_157 = uint(0); l_mut_157 < uint(16); ++l_mut_157) {
const uint l_158 = thread_index_in_threadgroup * uint(16);
const uint l_159 = l_158 + l_mut_157;
const float l_160 = shared_memory_33[l_159];
const float l_161 = l_160 - l_mut_153;
const float l_162 = exp(l_161);
const uint l_163 = thread_index_in_threadgroup * uint(16);
const uint l_164 = l_163 + l_mut_157;
shared_memory_33[l_164] = l_162;
l_mut_49 = l_mut_49 + l_162;
}
const float l_165 = shared_memory_36[thread_index_in_threadgroup];
const float l_166 = l_165 * l_156;
const float l_167 = l_166 + l_mut_49;
shared_memory_36[thread_index_in_threadgroup] = l_167;
shared_memory_35[thread_index_in_threadgroup] = l_mut_153;
shared_memory_37[thread_index_in_threadgroup] = l_156;
} else {
shared_memory_37[thread_index_in_threadgroup] = float(1.0);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_171 = uint(0); l_mut_171 < uint(8); ++l_mut_171) {
const uint l_172 = thread_index_in_threadgroup * uint(8);
const uint l_173 = l_172 + l_mut_171;
const uint l_174 = l_173 / uint(16);
const uint l_175 = l_22 * uint(16);
const uint l_176 = l_175 + l_174;
l_mut_153 = float(0.0);
const bool l_177 = l_176 < l_3;
if (l_177) {
const float l_178 = shared_memory_33[l_173];
l_mut_153 = l_178;
}
const half l_179 = half(l_mut_153);
shared_memory_168[l_173] = l_179;
}

for (uint l_mut_180 = uint(0); l_mut_180 < uint(64); ++l_mut_180) {
const uint l_181 = thread_index_in_threadgroup * uint(64);
const uint l_182 = l_181 + l_mut_180;
const uint l_183 = l_182 / uint(128);
const uint l_184 = l_182 % uint(128);
const uint l_185 = l_mut_73 * uint(16);
const uint l_186 = l_185 + l_183;
l_mut_153 = float(0.0);
const bool l_187 = l_186 < l_4;
if (l_187) {
const uint l_188 = l_186 * l_10;
const uint l_189 = l_31 + l_188;
const uint l_190 = l_189 + l_184;
const bfloat l_191 = buffer_2[l_190];
const float l_192 = float(l_191);
l_mut_153 = l_192;
}
const half l_193 = half(l_mut_153);
shared_memory_169[l_182] = l_193;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_194 = uint(0); l_mut_194 < uint(2); ++l_mut_194) {

for (uint l_mut_195 = uint(0); l_mut_195 < uint(16); ++l_mut_195) {
simdgroup_float8x8 frag_acc_196;
frag_acc_196 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_197 = uint(0); l_mut_197 < uint(2); ++l_mut_197) {
simdgroup_half8x8 frag_a_206;
simdgroup_half8x8 frag_b_215;
const uint l_198 = l_mut_194 * uint(8);
const uint l_199 = l_198 * uint(16);
const uint l_200 = l_mut_197 * uint(8);
const uint l_201 = l_199 + l_200;
const uint l_202 = 256;
const uint l_203 = l_202 - uint(0);
const uint l_204 = uint(256) - l_201;
const uint l_205 = l_201 + uint(0);
simdgroup_load(frag_a_206, shared_memory_168 + l_205, uint(16), 0, false);
const uint l_207 = l_mut_197 * uint(8);
const uint l_208 = l_207 * uint(128);
const uint l_209 = l_mut_195 * uint(8);
const uint l_210 = l_208 + l_209;
const uint l_211 = 2048;
const uint l_212 = l_211 - uint(0);
const uint l_213 = uint(2048) - l_210;
const uint l_214 = l_210 + uint(0);
simdgroup_load(frag_b_215, shared_memory_169 + l_214, uint(128), 0, false);
simdgroup_multiply_accumulate(frag_acc_196, frag_a_206, frag_b_215, frag_acc_196);
}
const uint l_216 = l_mut_194 * uint(8);
const uint l_217 = l_216 * uint(128);
const uint l_218 = l_mut_195 * uint(8);
const uint l_219 = l_217 + l_218;
const uint l_220 = 2048;
const uint l_221 = l_220 - uint(0);
const uint l_222 = uint(2048) - l_219;
const uint l_223 = l_219 + uint(0);
simdgroup_store(frag_acc_196, shared_memory_170 + l_223, uint(128));
simdgroup_barrier(mem_flags::mem_none);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_224 = uint(0); l_mut_224 < uint(64); ++l_mut_224) {
const uint l_225 = thread_index_in_threadgroup * uint(64);
const uint l_226 = l_225 + l_mut_224;
const uint l_227 = l_226 / uint(128);
const float l_228 = shared_memory_34[l_226];
const float l_229 = shared_memory_37[l_227];
const float l_230 = l_228 * l_229;
const float l_231 = shared_memory_170[l_226];
const float l_232 = l_230 + l_231;
shared_memory_34[l_226] = l_232;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint l_mut_233 = uint(0); l_mut_233 < uint(64); ++l_mut_233) {
const uint l_234 = thread_index_in_threadgroup * uint(64);
const uint l_235 = l_234 + l_mut_233;
const uint l_236 = l_235 / uint(128);
const uint l_237 = l_235 % uint(128);
const uint l_238 = l_22 * uint(16);
const uint l_239 = l_238 + l_236;
const bool l_240 = l_239 < l_3;
if (l_240) {
const uint l_241 = l_239 * uint(128);
const uint l_242 = l_28 + l_241;
const uint l_243 = l_242 + l_237;
const float l_244 = shared_memory_34[l_235];
const float l_245 = shared_memory_36[l_236];
const float l_246 = l_244 / l_245;
const bfloat l_247 = bfloat(l_246);
buffer_3[l_243] = l_247;
}
}

}
