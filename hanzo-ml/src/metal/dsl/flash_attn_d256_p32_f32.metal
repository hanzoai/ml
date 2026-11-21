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
threadgroup half* shared_memory_73 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[12640]);
// Shared array size: 64, 128 bytes
threadgroup half* shared_memory_164 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[16736]);
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_165 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[16864]);
// Shared array size: 2048, 8192 bytes
threadgroup float* shared_memory_166 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[20960]);
uint l_mut_11;
float l_mut_49;
uint l_mut_59;
uint l_mut_63;
float l_mut_149;
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
const float l_54 = buffer_0[l_53];
l_mut_49 = l_54;
}
const half l_55 = half(l_mut_49);
shared_memory_32[l_44] = l_55;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_56 = l_4 + uint(8);
const uint l_57 = l_56 - uint(1);
const uint l_58 = l_57 / uint(8);
l_mut_59 = l_58;
const bool l_60 = l_7 == uint(1);
const bool l_61 = l_7 == uint(2);
const bool l_62 = l_60 || l_61;
if (l_62) {
l_mut_63 = uint(0);
const bool l_64 = l_7 == uint(2);
if (l_64) {
l_mut_63 = l_mut_11;
}
const uint l_65 = l_22 * uint(8);
const uint l_66 = l_mut_63 + l_65;
const uint l_67 = l_66 + uint(8);
const uint l_68 = l_67 - uint(1);
const uint l_69 = l_68 / uint(8);
const uint l_70 = l_69 + uint(1);
const bool l_71 = l_70 < l_58;
if (l_71) {
l_mut_59 = l_70;
}
}

for (uint l_mut_72 = uint(0); l_mut_72 < l_mut_59; ++l_mut_72) {
simdgroup_float8x8 frag_acc_87;

for (uint l_mut_74 = uint(0); l_mut_74 < uint(64); ++l_mut_74) {
const uint l_75 = thread_index_in_threadgroup * uint(64);
const uint l_76 = l_75 + l_mut_74;
const uint l_77 = l_76 / uint(256);
const uint l_78 = l_76 % uint(256);
const uint l_79 = l_mut_72 * uint(8);
const uint l_80 = l_79 + l_77;
l_mut_49 = float(0.0);
const bool l_81 = l_80 < l_4;
if (l_81) {
const uint l_82 = l_80 * l_10;
const uint l_83 = l_31 + l_82;
const uint l_84 = l_83 + l_78;
const float l_85 = buffer_1[l_84];
l_mut_49 = l_85;
}
const half l_86 = half(l_mut_49);
shared_memory_73[l_76] = l_86;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
frag_acc_87 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_88 = uint(0); l_mut_88 < uint(32); ++l_mut_88) {
simdgroup_half8x8 frag_a_97;
simdgroup_half8x8 frag_b_106;
const uint l_89 = uint(0) * uint(8);
const uint l_90 = l_89 * uint(256);
const uint l_91 = l_mut_88 * uint(8);
const uint l_92 = l_90 + l_91;
const uint l_93 = 2048;
const uint l_94 = l_93 - uint(0);
const uint l_95 = uint(2048) - l_92;
const uint l_96 = l_92 + uint(0);
simdgroup_load(frag_a_97, shared_memory_32 + l_96, uint(256), 0, false);
const uint l_98 = uint(0) * uint(8);
const uint l_99 = l_98 * uint(256);
const uint l_100 = l_mut_88 * uint(8);
const uint l_101 = l_99 + l_100;
const uint l_102 = 2048;
const uint l_103 = l_102 - uint(0);
const uint l_104 = uint(2048) - l_101;
const uint l_105 = l_101 + uint(0);
simdgroup_load(frag_b_106, shared_memory_73 + l_105, uint(256), 0, true);
simdgroup_multiply_accumulate(frag_acc_87, frag_a_97, frag_b_106, frag_acc_87);
}
const uint l_107 = uint(0) * uint(8);
const uint l_108 = l_107 * uint(8);
const uint l_109 = uint(0) * uint(8);
const uint l_110 = l_108 + l_109;
const uint l_111 = 64;
const uint l_112 = l_111 - uint(0);
const uint l_113 = uint(64) - l_110;
const uint l_114 = l_110 + uint(0);
simdgroup_store(frag_acc_87, shared_memory_33 + l_114, uint(8));
simdgroup_barrier(mem_flags::mem_none);
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_115 = uint(0); l_mut_115 < uint(2); ++l_mut_115) {
const uint l_116 = thread_index_in_threadgroup * uint(2);
const uint l_117 = l_116 + l_mut_115;
const uint l_118 = l_117 / uint(8);
const uint l_119 = l_117 % uint(8);
const uint l_120 = l_22 * uint(8);
const uint l_121 = l_120 + l_118;
const uint l_122 = l_mut_72 * uint(8);
const uint l_123 = l_122 + l_119;
const bool l_124 = l_7 == uint(1);
const bool l_125 = l_123 > l_121;
const bool l_126 = l_124 && l_125;
const bool l_127 = l_7 == uint(2);
const uint l_128 = l_mut_11 + l_121;
const bool l_129 = l_123 > l_128;
const bool l_130 = l_127 && l_129;
const bool l_131 = l_126 || l_130;
const bool l_132 = l_121 < l_3;
const bool l_133 = l_123 < l_4;
const bool l_134 = l_132 && l_133;
const bool l_135 = !l_131;
const bool l_136 = l_134 && l_135;
if (l_136) {
const float l_137 = shared_memory_33[l_117];
const float l_138 = l_137 * l_2;
shared_memory_33[l_117] = l_138;
} else {
shared_memory_33[l_117] = float(-3.3999999521443642e38);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_139 = thread_index_in_threadgroup < uint(8);
if (l_139) {
const uint l_140 = l_22 * uint(8);
const uint l_141 = l_140 + thread_index_in_threadgroup;
const bool l_142 = l_141 < l_3;
if (l_142) {
l_mut_49 = float(-3.3999999521443642e38);

for (uint l_mut_143 = uint(0); l_mut_143 < uint(8); ++l_mut_143) {
const uint l_144 = thread_index_in_threadgroup * uint(8);
const uint l_145 = l_144 + l_mut_143;
const float l_146 = shared_memory_33[l_145];
const bool l_147 = l_146 > l_mut_49;
if (l_147) {
l_mut_49 = l_146;
}
}
const float l_148 = shared_memory_35[thread_index_in_threadgroup];
l_mut_149 = l_148;
const bool l_150 = l_mut_49 > l_mut_149;
if (l_150) {
l_mut_149 = l_mut_49;
}
const float l_151 = l_148 - l_mut_149;
const float l_152 = exp(l_151);
l_mut_49 = float(0.0);

for (uint l_mut_153 = uint(0); l_mut_153 < uint(8); ++l_mut_153) {
const uint l_154 = thread_index_in_threadgroup * uint(8);
const uint l_155 = l_154 + l_mut_153;
const float l_156 = shared_memory_33[l_155];
const float l_157 = l_156 - l_mut_149;
const float l_158 = exp(l_157);
const uint l_159 = thread_index_in_threadgroup * uint(8);
const uint l_160 = l_159 + l_mut_153;
shared_memory_33[l_160] = l_158;
l_mut_49 = l_mut_49 + l_158;
}
const float l_161 = shared_memory_36[thread_index_in_threadgroup];
const float l_162 = l_161 * l_152;
const float l_163 = l_162 + l_mut_49;
shared_memory_36[thread_index_in_threadgroup] = l_163;
shared_memory_35[thread_index_in_threadgroup] = l_mut_149;
shared_memory_37[thread_index_in_threadgroup] = l_152;
} else {
shared_memory_37[thread_index_in_threadgroup] = float(1.0);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_167 = uint(0); l_mut_167 < uint(2); ++l_mut_167) {
const uint l_168 = thread_index_in_threadgroup * uint(2);
const uint l_169 = l_168 + l_mut_167;
const uint l_170 = l_169 / uint(8);
const uint l_171 = l_22 * uint(8);
const uint l_172 = l_171 + l_170;
l_mut_149 = float(0.0);
const bool l_173 = l_172 < l_3;
if (l_173) {
const float l_174 = shared_memory_33[l_169];
l_mut_149 = l_174;
}
const half l_175 = half(l_mut_149);
shared_memory_164[l_169] = l_175;
}

for (uint l_mut_176 = uint(0); l_mut_176 < uint(64); ++l_mut_176) {
const uint l_177 = thread_index_in_threadgroup * uint(64);
const uint l_178 = l_177 + l_mut_176;
const uint l_179 = l_178 / uint(256);
const uint l_180 = l_178 % uint(256);
const uint l_181 = l_mut_72 * uint(8);
const uint l_182 = l_181 + l_179;
l_mut_149 = float(0.0);
const bool l_183 = l_182 < l_4;
if (l_183) {
const uint l_184 = l_182 * l_10;
const uint l_185 = l_31 + l_184;
const uint l_186 = l_185 + l_180;
const float l_187 = buffer_2[l_186];
l_mut_149 = l_187;
}
const half l_188 = half(l_mut_149);
shared_memory_165[l_178] = l_188;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_189 = uint(0); l_mut_189 < uint(32); ++l_mut_189) {
simdgroup_float8x8 frag_acc_190;
simdgroup_half8x8 frag_a_199;
simdgroup_half8x8 frag_b_208;
frag_acc_190 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));
const uint l_191 = uint(0) * uint(8);
const uint l_192 = l_191 * uint(8);
const uint l_193 = uint(0) * uint(8);
const uint l_194 = l_192 + l_193;
const uint l_195 = 64;
const uint l_196 = l_195 - uint(0);
const uint l_197 = uint(64) - l_194;
const uint l_198 = l_194 + uint(0);
simdgroup_load(frag_a_199, shared_memory_164 + l_198, uint(8), 0, false);
const uint l_200 = uint(0) * uint(8);
const uint l_201 = l_200 * uint(256);
const uint l_202 = l_mut_189 * uint(8);
const uint l_203 = l_201 + l_202;
const uint l_204 = 2048;
const uint l_205 = l_204 - uint(0);
const uint l_206 = uint(2048) - l_203;
const uint l_207 = l_203 + uint(0);
simdgroup_load(frag_b_208, shared_memory_165 + l_207, uint(256), 0, false);
simdgroup_multiply_accumulate(frag_acc_190, frag_a_199, frag_b_208, frag_acc_190);
const uint l_209 = uint(0) * uint(8);
const uint l_210 = l_209 * uint(256);
const uint l_211 = l_mut_189 * uint(8);
const uint l_212 = l_210 + l_211;
const uint l_213 = 2048;
const uint l_214 = l_213 - uint(0);
const uint l_215 = uint(2048) - l_212;
const uint l_216 = l_212 + uint(0);
simdgroup_store(frag_acc_190, shared_memory_166 + l_216, uint(256));
simdgroup_barrier(mem_flags::mem_none);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_217 = uint(0); l_mut_217 < uint(64); ++l_mut_217) {
const uint l_218 = thread_index_in_threadgroup * uint(64);
const uint l_219 = l_218 + l_mut_217;
const uint l_220 = l_219 / uint(256);
const float l_221 = shared_memory_34[l_219];
const float l_222 = shared_memory_37[l_220];
const float l_223 = l_221 * l_222;
const float l_224 = shared_memory_166[l_219];
const float l_225 = l_223 + l_224;
shared_memory_34[l_219] = l_225;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint l_mut_226 = uint(0); l_mut_226 < uint(64); ++l_mut_226) {
const uint l_227 = thread_index_in_threadgroup * uint(64);
const uint l_228 = l_227 + l_mut_226;
const uint l_229 = l_228 / uint(256);
const uint l_230 = l_228 % uint(256);
const uint l_231 = l_22 * uint(8);
const uint l_232 = l_231 + l_229;
const bool l_233 = l_232 < l_3;
if (l_233) {
const uint l_234 = l_232 * uint(256);
const uint l_235 = l_28 + l_234;
const uint l_236 = l_235 + l_230;
const float l_237 = shared_memory_34[l_228];
const float l_238 = shared_memory_36[l_229];
const float l_239 = l_237 / l_238;
buffer_3[l_236] = l_239;
}
}

}
