#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[16];
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
void flash_attn_paged_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device half * buffer_1 [[buffer(1)]],
    const device half * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    device half * buffer_5 [[buffer(5)]],
    const device float * buffer_6 [[buffer(6)]],
    const device uint * buffer_7 [[buffer(7)]],
    constant info_st& info [[buffer(8)]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint thread_index_in_threadgroup [[thread_index_in_threadgroup]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]],
    uint simd_lane_id [[thread_index_in_simdgroup]],
    uint simd_size [[threads_per_simdgroup]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]]
) {
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
threadgroup uchar dynamic_shared_mem[12992];
// Shared array size: 1024, 2048 bytes
threadgroup half* shared_memory_34 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[0]);
// Shared array size: 256, 1024 bytes
threadgroup float* shared_memory_35 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[2048]);
// Shared array size: 1024, 4096 bytes
threadgroup float* shared_memory_36 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[3072]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_37 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[7168]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_38 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[7232]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_39 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[7296]);
// Shared array size: 1024, 2048 bytes
threadgroup half* shared_memory_80 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[7360]);
// Shared array size: 256, 512 bytes
threadgroup half* shared_memory_209 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[9408]);
// Shared array size: 1024, 2048 bytes
threadgroup half* shared_memory_210 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[9920]);
// Shared array size: 256, 1024 bytes
threadgroup float* shared_memory_211 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[11968]);
uint l_mut_26;
float l_mut_51;
uint l_mut_62;
uint l_mut_71;
half l_mut_97;
half l_mut_111;
float l_mut_194;
const uint l_0 = buffer_7[uint(10)];
const uint l_1 = threadgroup_index_in_grid + l_0;
const float l_2 = buffer_6[uint(0)];
const float l_3 = buffer_6[uint(1)];
const uint l_4 = buffer_7[uint(0)];
const uint l_5 = buffer_7[uint(1)];
const uint l_6 = buffer_7[uint(2)];
const uint l_7 = buffer_7[uint(3)];
const uint l_8 = buffer_7[uint(4)];
const uint l_9 = buffer_7[uint(5)];
const uint l_10 = buffer_7[uint(6)];
const uint l_11 = buffer_7[uint(7)];
const uint l_12 = buffer_7[uint(8)];
const uint l_13 = buffer_7[uint(9)];
const uint l_14 = l_4 + uint(16);
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
const uint l_25 = buffer_4[l_18];
l_mut_26 = uint(0);
const bool l_27 = l_7 == uint(2);
if (l_27) {
l_mut_26 = l_13;
} else {
const bool l_28 = l_25 >= l_4;
if (l_28) {
const uint l_29 = l_25 - l_4;
l_mut_26 = l_29;
}
}
const uint l_30 = l_18 * l_5;
const uint l_31 = l_30 + l_21;
const uint l_32 = l_31 * l_4;
const uint l_33 = l_32 * uint(64);

for (uint l_mut_40 = uint(0); l_mut_40 < uint(32); ++l_mut_40) {
const uint l_41 = thread_index_in_threadgroup * uint(32);
const uint l_42 = l_41 + l_mut_40;
shared_memory_36[l_42] = float(0.0);
}
const bool l_43 = thread_index_in_threadgroup < uint(16);
if (l_43) {
shared_memory_37[thread_index_in_threadgroup] = float(-3.3999999521443642e38);
shared_memory_38[thread_index_in_threadgroup] = float(0.0);
}

for (uint l_mut_44 = uint(0); l_mut_44 < uint(32); ++l_mut_44) {
const uint l_45 = thread_index_in_threadgroup * uint(32);
const uint l_46 = l_45 + l_mut_44;
const uint l_47 = l_46 / uint(64);
const uint l_48 = l_46 % uint(64);
const uint l_49 = l_22 * uint(16);
const uint l_50 = l_49 + l_47;
l_mut_51 = float(0.0);
const bool l_52 = l_50 < l_4;
if (l_52) {
const uint l_53 = l_50 * uint(64);
const uint l_54 = l_33 + l_53;
const uint l_55 = l_54 + l_48;
const half l_56 = buffer_0[l_55];
const float l_57 = float(l_56);
l_mut_51 = l_57;
}
const half l_58 = half(l_mut_51);
shared_memory_34[l_46] = l_58;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_59 = l_25 + uint(16);
const uint l_60 = l_59 - uint(1);
const uint l_61 = l_60 / uint(16);
l_mut_62 = l_61;
const bool l_63 = l_7 != uint(0);
if (l_63) {
const uint l_64 = l_22 * uint(16);
const uint l_65 = l_mut_26 + l_64;
const uint l_66 = l_65 + uint(16);
const uint l_67 = l_66 - uint(1);
const uint l_68 = l_67 / uint(16);
const uint l_69 = l_68 + uint(1);
const bool l_70 = l_69 < l_mut_62;
if (l_70) {
l_mut_62 = l_69;
}
}
l_mut_71 = uint(0);
const bool l_72 = l_12 > uint(0);
if (l_72) {
const uint l_73 = l_22 * uint(16);
const uint l_74 = l_mut_26 + l_73;
const bool l_75 = l_74 >= l_12;
if (l_75) {
const uint l_76 = l_74 - l_12;
const uint l_77 = l_76 + uint(1);
const uint l_78 = l_77 / uint(16);
l_mut_71 = l_78;
}
}

for (uint l_mut_79 = l_mut_71; l_mut_79 < l_mut_62; ++l_mut_79) {

for (uint l_mut_81 = uint(0); l_mut_81 < uint(32); ++l_mut_81) {
const uint l_82 = thread_index_in_threadgroup * uint(32);
const uint l_83 = l_82 + l_mut_81;
const uint l_84 = l_83 / uint(64);
const uint l_85 = l_83 % uint(64);
const uint l_86 = l_mut_79 * uint(16);
const uint l_87 = l_86 + l_84;
l_mut_51 = float(0.0);
const bool l_88 = l_87 < l_25;
if (l_88) {
const bool l_89 = l_10 == uint(2);
if (l_89) {
const uint l_90 = l_18 * l_6;
const uint l_91 = l_90 + l_24;
const uint l_92 = l_91 * l_8;
const uint l_93 = l_92 + l_87;
const uint l_94 = l_93 * uint(64);
const uint l_95 = l_94 + l_85;
const half l_96 = buffer_1[l_95];
l_mut_97 = l_96;
} else {
const uint l_98 = l_18 * l_9;
const uint l_99 = l_87 / l_8;
const uint l_100 = l_98 + l_99;
const uint l_101 = buffer_3[l_100];
const uint l_102 = l_87 % l_8;
const bool l_103 = l_10 == uint(0);
if (l_103) {
const uint l_104 = l_101 * l_6;
const uint l_105 = l_104 + l_24;
const uint l_106 = l_105 * l_8;
const uint l_107 = l_106 + l_102;
const uint l_108 = l_107 * uint(64);
const uint l_109 = l_108 + l_85;
const half l_110 = buffer_1[l_109];
l_mut_111 = l_110;
} else {
const uint l_112 = uint(64) / l_11;
const uint l_113 = l_101 * l_6;
const uint l_114 = l_113 + l_24;
const uint l_115 = l_114 * l_112;
const uint l_116 = l_85 / l_11;
const uint l_117 = l_115 + l_116;
const uint l_118 = l_117 * l_8;
const uint l_119 = l_118 + l_102;
const uint l_120 = l_119 * l_11;
const uint l_121 = l_85 % l_11;
const uint l_122 = l_120 + l_121;
const half l_123 = buffer_1[l_122];
l_mut_111 = l_123;
}
l_mut_97 = l_mut_111;
}
const float l_124 = float(l_mut_97);
l_mut_51 = l_124;
}
const half l_125 = half(l_mut_51);
shared_memory_80[l_83] = l_125;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_126 = uint(0); l_mut_126 < uint(2); ++l_mut_126) {

for (uint l_mut_127 = uint(0); l_mut_127 < uint(2); ++l_mut_127) {
simdgroup_float8x8 frag_acc_128;
frag_acc_128 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_129 = uint(0); l_mut_129 < uint(8); ++l_mut_129) {
simdgroup_half8x8 frag_a_138;
simdgroup_half8x8 frag_b_147;
const uint l_130 = l_mut_126 * uint(8);
const uint l_131 = l_130 * uint(64);
const uint l_132 = l_mut_129 * uint(8);
const uint l_133 = l_131 + l_132;
const uint l_134 = 1024;
const uint l_135 = l_134 - uint(0);
const uint l_136 = uint(1024) - l_133;
const uint l_137 = l_133 + uint(0);
simdgroup_load(frag_a_138, shared_memory_34 + l_137, uint(64), 0, false);
const uint l_139 = l_mut_127 * uint(8);
const uint l_140 = l_139 * uint(64);
const uint l_141 = l_mut_129 * uint(8);
const uint l_142 = l_140 + l_141;
const uint l_143 = 1024;
const uint l_144 = l_143 - uint(0);
const uint l_145 = uint(1024) - l_142;
const uint l_146 = l_142 + uint(0);
simdgroup_load(frag_b_147, shared_memory_80 + l_146, uint(64), 0, true);
simdgroup_multiply_accumulate(frag_acc_128, frag_a_138, frag_b_147, frag_acc_128);
}
const uint l_148 = l_mut_126 * uint(8);
const uint l_149 = l_148 * uint(16);
const uint l_150 = l_mut_127 * uint(8);
const uint l_151 = l_149 + l_150;
const uint l_152 = 256;
const uint l_153 = l_152 - uint(0);
const uint l_154 = uint(256) - l_151;
const uint l_155 = l_151 + uint(0);
simdgroup_store(frag_acc_128, shared_memory_35 + l_155, uint(16));
simdgroup_barrier(mem_flags::mem_none);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_156 = uint(0); l_mut_156 < uint(8); ++l_mut_156) {
const uint l_157 = thread_index_in_threadgroup * uint(8);
const uint l_158 = l_157 + l_mut_156;
const uint l_159 = l_158 / uint(16);
const uint l_160 = l_158 % uint(16);
const uint l_161 = l_22 * uint(16);
const uint l_162 = l_161 + l_159;
const uint l_163 = l_mut_79 * uint(16);
const uint l_164 = l_163 + l_160;
const uint l_165 = l_mut_26 + l_162;
const bool l_166 = l_7 == uint(0);
const bool l_167 = l_164 <= l_165;
const bool l_168 = l_166 || l_167;
const bool l_169 = l_12 == uint(0);
const uint l_170 = l_164 + l_12;
const bool l_171 = l_170 > l_165;
const bool l_172 = l_169 || l_171;
const bool l_173 = l_162 < l_4;
const bool l_174 = l_164 < l_25;
const bool l_175 = l_173 && l_174;
const bool l_176 = l_175 && l_168;
const bool l_177 = l_176 && l_172;
if (l_177) {
const float l_178 = shared_memory_35[l_158];
const float l_179 = l_178 * l_2;
l_mut_51 = l_179;
const bool l_180 = l_3 > float(0.0);
if (l_180) {
const float l_181 = l_mut_51 / l_3;
const float l_182 = safe_tanh_scalar(l_181);
const float l_183 = l_3 * l_182;
l_mut_51 = l_183;
}
shared_memory_35[l_158] = l_mut_51;
} else {
shared_memory_35[l_158] = float(-3.3999999521443642e38);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_184 = thread_index_in_threadgroup < uint(16);
if (l_184) {
const uint l_185 = l_22 * uint(16);
const uint l_186 = l_185 + thread_index_in_threadgroup;
const bool l_187 = l_186 < l_4;
if (l_187) {
l_mut_51 = float(-3.3999999521443642e38);

for (uint l_mut_188 = uint(0); l_mut_188 < uint(16); ++l_mut_188) {
const uint l_189 = thread_index_in_threadgroup * uint(16);
const uint l_190 = l_189 + l_mut_188;
const float l_191 = shared_memory_35[l_190];
const bool l_192 = l_191 > l_mut_51;
if (l_192) {
l_mut_51 = l_191;
}
}
const float l_193 = shared_memory_37[thread_index_in_threadgroup];
l_mut_194 = l_193;
const bool l_195 = l_mut_51 > l_mut_194;
if (l_195) {
l_mut_194 = l_mut_51;
}
const float l_196 = l_193 - l_mut_194;
const float l_197 = exp(l_196);
l_mut_51 = float(0.0);

for (uint l_mut_198 = uint(0); l_mut_198 < uint(16); ++l_mut_198) {
const uint l_199 = thread_index_in_threadgroup * uint(16);
const uint l_200 = l_199 + l_mut_198;
const float l_201 = shared_memory_35[l_200];
const float l_202 = l_201 - l_mut_194;
const float l_203 = exp(l_202);
const uint l_204 = thread_index_in_threadgroup * uint(16);
const uint l_205 = l_204 + l_mut_198;
shared_memory_35[l_205] = l_203;
l_mut_51 = l_mut_51 + l_203;
}
const float l_206 = shared_memory_38[thread_index_in_threadgroup];
const float l_207 = l_206 * l_197;
const float l_208 = l_207 + l_mut_51;
shared_memory_38[thread_index_in_threadgroup] = l_208;
shared_memory_37[thread_index_in_threadgroup] = l_mut_194;
shared_memory_39[thread_index_in_threadgroup] = l_197;
} else {
shared_memory_39[thread_index_in_threadgroup] = float(1.0);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_212 = uint(0); l_mut_212 < uint(8); ++l_mut_212) {
const uint l_213 = thread_index_in_threadgroup * uint(8);
const uint l_214 = l_213 + l_mut_212;
const uint l_215 = l_214 / uint(16);
const uint l_216 = l_22 * uint(16);
const uint l_217 = l_216 + l_215;
l_mut_194 = float(0.0);
const bool l_218 = l_217 < l_4;
if (l_218) {
const float l_219 = shared_memory_35[l_214];
l_mut_194 = l_219;
}
const half l_220 = half(l_mut_194);
shared_memory_209[l_214] = l_220;
}

for (uint l_mut_221 = uint(0); l_mut_221 < uint(32); ++l_mut_221) {
const uint l_222 = thread_index_in_threadgroup * uint(32);
const uint l_223 = l_222 + l_mut_221;
const uint l_224 = l_223 / uint(64);
const uint l_225 = l_223 % uint(64);
const uint l_226 = l_mut_79 * uint(16);
const uint l_227 = l_226 + l_224;
l_mut_194 = float(0.0);
const bool l_228 = l_227 < l_25;
if (l_228) {
const bool l_229 = l_10 == uint(2);
if (l_229) {
const uint l_230 = l_18 * l_6;
const uint l_231 = l_230 + l_24;
const uint l_232 = l_231 * l_8;
const uint l_233 = l_232 + l_227;
const uint l_234 = l_233 * uint(64);
const uint l_235 = l_234 + l_225;
const half l_236 = buffer_2[l_235];
l_mut_111 = l_236;
} else {
const uint l_237 = l_18 * l_9;
const uint l_238 = l_227 / l_8;
const uint l_239 = l_237 + l_238;
const uint l_240 = buffer_3[l_239];
const uint l_241 = l_227 % l_8;
const bool l_242 = l_10 == uint(0);
if (l_242) {
const uint l_243 = l_240 * l_6;
const uint l_244 = l_243 + l_24;
const uint l_245 = l_244 * l_8;
const uint l_246 = l_245 + l_241;
const uint l_247 = l_246 * uint(64);
const uint l_248 = l_247 + l_225;
const half l_249 = buffer_2[l_248];
l_mut_97 = l_249;
} else {
const uint l_250 = l_240 * l_6;
const uint l_251 = l_250 + l_24;
const uint l_252 = l_251 * uint(64);
const uint l_253 = l_252 + l_225;
const uint l_254 = l_253 * l_8;
const uint l_255 = l_254 + l_241;
const half l_256 = buffer_2[l_255];
l_mut_97 = l_256;
}
l_mut_111 = l_mut_97;
}
const float l_257 = float(l_mut_111);
l_mut_194 = l_257;
}
const half l_258 = half(l_mut_194);
shared_memory_210[l_223] = l_258;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_259 = uint(0); l_mut_259 < uint(4); ++l_mut_259) {

for (uint l_mut_260 = uint(0); l_mut_260 < uint(2); ++l_mut_260) {

for (uint l_mut_261 = uint(0); l_mut_261 < uint(2); ++l_mut_261) {
simdgroup_float8x8 frag_acc_262;
frag_acc_262 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_263 = uint(0); l_mut_263 < uint(2); ++l_mut_263) {
simdgroup_half8x8 frag_a_272;
simdgroup_half8x8 frag_b_283;
const uint l_264 = l_mut_260 * uint(8);
const uint l_265 = l_264 * uint(16);
const uint l_266 = l_mut_263 * uint(8);
const uint l_267 = l_265 + l_266;
const uint l_268 = 256;
const uint l_269 = l_268 - uint(0);
const uint l_270 = uint(256) - l_267;
const uint l_271 = l_267 + uint(0);
simdgroup_load(frag_a_272, shared_memory_209 + l_271, uint(16), 0, false);
const uint l_273 = l_mut_263 * uint(8);
const uint l_274 = l_273 * uint(64);
const uint l_275 = l_mut_259 * uint(16);
const uint l_276 = l_274 + l_275;
const uint l_277 = l_mut_261 * uint(8);
const uint l_278 = l_276 + l_277;
const uint l_279 = 1024;
const uint l_280 = l_279 - uint(0);
const uint l_281 = uint(1024) - l_278;
const uint l_282 = l_278 + uint(0);
simdgroup_load(frag_b_283, shared_memory_210 + l_282, uint(64), 0, false);
simdgroup_multiply_accumulate(frag_acc_262, frag_a_272, frag_b_283, frag_acc_262);
}
const uint l_284 = l_mut_260 * uint(8);
const uint l_285 = l_284 * uint(16);
const uint l_286 = l_mut_261 * uint(8);
const uint l_287 = l_285 + l_286;
const uint l_288 = 256;
const uint l_289 = l_288 - uint(0);
const uint l_290 = uint(256) - l_287;
const uint l_291 = l_287 + uint(0);
simdgroup_store(frag_acc_262, shared_memory_211 + l_291, uint(16));
simdgroup_barrier(mem_flags::mem_none);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_292 = uint(0); l_mut_292 < uint(8); ++l_mut_292) {
const uint l_293 = thread_index_in_threadgroup * uint(8);
const uint l_294 = l_293 + l_mut_292;
const uint l_295 = l_294 / uint(16);
const uint l_296 = l_294 % uint(16);
const uint l_297 = l_295 * uint(64);
const uint l_298 = l_mut_259 * uint(16);
const uint l_299 = l_297 + l_298;
const uint l_300 = l_299 + l_296;
const float l_301 = shared_memory_36[l_300];
const float l_302 = shared_memory_39[l_295];
const float l_303 = l_301 * l_302;
const float l_304 = shared_memory_211[l_294];
const float l_305 = l_303 + l_304;
shared_memory_36[l_300] = l_305;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint l_mut_306 = uint(0); l_mut_306 < uint(32); ++l_mut_306) {
const uint l_307 = thread_index_in_threadgroup * uint(32);
const uint l_308 = l_307 + l_mut_306;
const uint l_309 = l_308 / uint(64);
const uint l_310 = l_308 % uint(64);
const uint l_311 = l_22 * uint(16);
const uint l_312 = l_311 + l_309;
const bool l_313 = l_312 < l_4;
if (l_313) {
l_mut_194 = float(1.0);
const float l_314 = shared_memory_38[l_309];
const bool l_315 = l_314 > float(0.0);
if (l_315) {
const float l_316 = shared_memory_38[l_309];
l_mut_194 = l_316;
}
const uint l_317 = l_312 * uint(64);
const uint l_318 = l_33 + l_317;
const uint l_319 = l_318 + l_310;
const float l_320 = shared_memory_36[l_308];
const float l_321 = l_320 / l_mut_194;
const half l_322 = half(l_321);
buffer_5[l_319] = l_322;
}
}

}
