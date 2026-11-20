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
void flash_attn_paged_f_f32(
    const device float * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    const device float * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    device float * buffer_5 [[buffer(5)]],
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
threadgroup uchar dynamic_shared_mem[23232];
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_34 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[0]);
// Shared array size: 256, 1024 bytes
threadgroup float* shared_memory_35 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[4096]);
// Shared array size: 2048, 8192 bytes
threadgroup float* shared_memory_36 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[5120]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_37 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[13312]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_38 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[13376]);
// Shared array size: 16, 64 bytes
threadgroup float* shared_memory_39 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[13440]);
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_79 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[13504]);
// Shared array size: 256, 512 bytes
threadgroup half* shared_memory_206 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[17600]);
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_207 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[18112]);
// Shared array size: 256, 1024 bytes
threadgroup float* shared_memory_208 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[22208]);
uint l_mut_26;
float l_mut_51;
uint l_mut_61;
uint l_mut_70;
float l_mut_96;
float l_mut_110;
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
const uint l_33 = l_32 * uint(128);

for (uint l_mut_40 = uint(0); l_mut_40 < uint(64); ++l_mut_40) {
const uint l_41 = thread_index_in_threadgroup * uint(64);
const uint l_42 = l_41 + l_mut_40;
shared_memory_36[l_42] = float(0.0);
}
const bool l_43 = thread_index_in_threadgroup < uint(16);
if (l_43) {
shared_memory_37[thread_index_in_threadgroup] = float(-3.3999999521443642e38);
shared_memory_38[thread_index_in_threadgroup] = float(0.0);
}

for (uint l_mut_44 = uint(0); l_mut_44 < uint(64); ++l_mut_44) {
const uint l_45 = thread_index_in_threadgroup * uint(64);
const uint l_46 = l_45 + l_mut_44;
const uint l_47 = l_46 / uint(128);
const uint l_48 = l_46 % uint(128);
const uint l_49 = l_22 * uint(16);
const uint l_50 = l_49 + l_47;
l_mut_51 = float(0.0);
const bool l_52 = l_50 < l_4;
if (l_52) {
const uint l_53 = l_50 * uint(128);
const uint l_54 = l_33 + l_53;
const uint l_55 = l_54 + l_48;
const float l_56 = buffer_0[l_55];
l_mut_51 = l_56;
}
const half l_57 = half(l_mut_51);
shared_memory_34[l_46] = l_57;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_58 = l_25 + uint(16);
const uint l_59 = l_58 - uint(1);
const uint l_60 = l_59 / uint(16);
l_mut_61 = l_60;
const bool l_62 = l_7 != uint(0);
if (l_62) {
const uint l_63 = l_22 * uint(16);
const uint l_64 = l_mut_26 + l_63;
const uint l_65 = l_64 + uint(16);
const uint l_66 = l_65 - uint(1);
const uint l_67 = l_66 / uint(16);
const uint l_68 = l_67 + uint(1);
const bool l_69 = l_68 < l_mut_61;
if (l_69) {
l_mut_61 = l_68;
}
}
l_mut_70 = uint(0);
const bool l_71 = l_12 > uint(0);
if (l_71) {
const uint l_72 = l_22 * uint(16);
const uint l_73 = l_mut_26 + l_72;
const bool l_74 = l_73 >= l_12;
if (l_74) {
const uint l_75 = l_73 - l_12;
const uint l_76 = l_75 + uint(1);
const uint l_77 = l_76 / uint(16);
l_mut_70 = l_77;
}
}

for (uint l_mut_78 = l_mut_70; l_mut_78 < l_mut_61; ++l_mut_78) {

for (uint l_mut_80 = uint(0); l_mut_80 < uint(64); ++l_mut_80) {
const uint l_81 = thread_index_in_threadgroup * uint(64);
const uint l_82 = l_81 + l_mut_80;
const uint l_83 = l_82 / uint(128);
const uint l_84 = l_82 % uint(128);
const uint l_85 = l_mut_78 * uint(16);
const uint l_86 = l_85 + l_83;
l_mut_51 = float(0.0);
const bool l_87 = l_86 < l_25;
if (l_87) {
const bool l_88 = l_10 == uint(2);
if (l_88) {
const uint l_89 = l_18 * l_6;
const uint l_90 = l_89 + l_24;
const uint l_91 = l_90 * l_8;
const uint l_92 = l_91 + l_86;
const uint l_93 = l_92 * uint(128);
const uint l_94 = l_93 + l_84;
const float l_95 = buffer_1[l_94];
l_mut_96 = l_95;
} else {
const uint l_97 = l_18 * l_9;
const uint l_98 = l_86 / l_8;
const uint l_99 = l_97 + l_98;
const uint l_100 = buffer_3[l_99];
const uint l_101 = l_86 % l_8;
const bool l_102 = l_10 == uint(0);
if (l_102) {
const uint l_103 = l_100 * l_6;
const uint l_104 = l_103 + l_24;
const uint l_105 = l_104 * l_8;
const uint l_106 = l_105 + l_101;
const uint l_107 = l_106 * uint(128);
const uint l_108 = l_107 + l_84;
const float l_109 = buffer_1[l_108];
l_mut_110 = l_109;
} else {
const uint l_111 = uint(128) / l_11;
const uint l_112 = l_100 * l_6;
const uint l_113 = l_112 + l_24;
const uint l_114 = l_113 * l_111;
const uint l_115 = l_84 / l_11;
const uint l_116 = l_114 + l_115;
const uint l_117 = l_116 * l_8;
const uint l_118 = l_117 + l_101;
const uint l_119 = l_118 * l_11;
const uint l_120 = l_84 % l_11;
const uint l_121 = l_119 + l_120;
const float l_122 = buffer_1[l_121];
l_mut_110 = l_122;
}
l_mut_96 = l_mut_110;
}
l_mut_51 = l_mut_96;
}
const half l_123 = half(l_mut_51);
shared_memory_79[l_82] = l_123;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_124 = uint(0); l_mut_124 < uint(2); ++l_mut_124) {

for (uint l_mut_125 = uint(0); l_mut_125 < uint(2); ++l_mut_125) {
simdgroup_float8x8 frag_acc_126;
frag_acc_126 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_127 = uint(0); l_mut_127 < uint(16); ++l_mut_127) {
simdgroup_half8x8 frag_a_136;
simdgroup_half8x8 frag_b_145;
const uint l_128 = l_mut_124 * uint(8);
const uint l_129 = l_128 * uint(128);
const uint l_130 = l_mut_127 * uint(8);
const uint l_131 = l_129 + l_130;
const uint l_132 = 2048;
const uint l_133 = l_132 - uint(0);
const uint l_134 = uint(2048) - l_131;
const uint l_135 = l_131 + uint(0);
simdgroup_load(frag_a_136, shared_memory_34 + l_135, uint(128), 0, false);
const uint l_137 = l_mut_125 * uint(8);
const uint l_138 = l_137 * uint(128);
const uint l_139 = l_mut_127 * uint(8);
const uint l_140 = l_138 + l_139;
const uint l_141 = 2048;
const uint l_142 = l_141 - uint(0);
const uint l_143 = uint(2048) - l_140;
const uint l_144 = l_140 + uint(0);
simdgroup_load(frag_b_145, shared_memory_79 + l_144, uint(128), 0, true);
simdgroup_multiply_accumulate(frag_acc_126, frag_a_136, frag_b_145, frag_acc_126);
}
const uint l_146 = l_mut_124 * uint(8);
const uint l_147 = l_146 * uint(16);
const uint l_148 = l_mut_125 * uint(8);
const uint l_149 = l_147 + l_148;
const uint l_150 = 256;
const uint l_151 = l_150 - uint(0);
const uint l_152 = uint(256) - l_149;
const uint l_153 = l_149 + uint(0);
simdgroup_store(frag_acc_126, shared_memory_35 + l_153, uint(16));
simdgroup_barrier(mem_flags::mem_none);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_154 = uint(0); l_mut_154 < uint(8); ++l_mut_154) {
const uint l_155 = thread_index_in_threadgroup * uint(8);
const uint l_156 = l_155 + l_mut_154;
const uint l_157 = l_156 / uint(16);
const uint l_158 = l_156 % uint(16);
const uint l_159 = l_22 * uint(16);
const uint l_160 = l_159 + l_157;
const uint l_161 = l_mut_78 * uint(16);
const uint l_162 = l_161 + l_158;
const uint l_163 = l_mut_26 + l_160;
const bool l_164 = l_7 == uint(0);
const bool l_165 = l_162 <= l_163;
const bool l_166 = l_164 || l_165;
const bool l_167 = l_12 == uint(0);
const uint l_168 = l_162 + l_12;
const bool l_169 = l_168 > l_163;
const bool l_170 = l_167 || l_169;
const bool l_171 = l_160 < l_4;
const bool l_172 = l_162 < l_25;
const bool l_173 = l_171 && l_172;
const bool l_174 = l_173 && l_166;
const bool l_175 = l_174 && l_170;
if (l_175) {
const float l_176 = shared_memory_35[l_156];
const float l_177 = l_176 * l_2;
l_mut_110 = l_177;
const bool l_178 = l_3 > float(0.0);
if (l_178) {
const float l_179 = l_mut_110 / l_3;
const float l_180 = safe_tanh_scalar(l_179);
const float l_181 = l_3 * l_180;
l_mut_110 = l_181;
}
shared_memory_35[l_156] = l_mut_110;
} else {
shared_memory_35[l_156] = float(-3.3999999521443642e38);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_182 = thread_index_in_threadgroup < uint(16);
if (l_182) {
const uint l_183 = l_22 * uint(16);
const uint l_184 = l_183 + thread_index_in_threadgroup;
const bool l_185 = l_184 < l_4;
if (l_185) {
l_mut_110 = float(-3.3999999521443642e38);

for (uint l_mut_186 = uint(0); l_mut_186 < uint(16); ++l_mut_186) {
const uint l_187 = thread_index_in_threadgroup * uint(16);
const uint l_188 = l_187 + l_mut_186;
const float l_189 = shared_memory_35[l_188];
const bool l_190 = l_189 > l_mut_110;
if (l_190) {
l_mut_110 = l_189;
}
}
const float l_191 = shared_memory_37[thread_index_in_threadgroup];
l_mut_96 = l_191;
const bool l_192 = l_mut_110 > l_mut_96;
if (l_192) {
l_mut_96 = l_mut_110;
}
const float l_193 = l_191 - l_mut_96;
const float l_194 = exp(l_193);
l_mut_110 = float(0.0);

for (uint l_mut_195 = uint(0); l_mut_195 < uint(16); ++l_mut_195) {
const uint l_196 = thread_index_in_threadgroup * uint(16);
const uint l_197 = l_196 + l_mut_195;
const float l_198 = shared_memory_35[l_197];
const float l_199 = l_198 - l_mut_96;
const float l_200 = exp(l_199);
const uint l_201 = thread_index_in_threadgroup * uint(16);
const uint l_202 = l_201 + l_mut_195;
shared_memory_35[l_202] = l_200;
l_mut_110 = l_mut_110 + l_200;
}
const float l_203 = shared_memory_38[thread_index_in_threadgroup];
const float l_204 = l_203 * l_194;
const float l_205 = l_204 + l_mut_110;
shared_memory_38[thread_index_in_threadgroup] = l_205;
shared_memory_37[thread_index_in_threadgroup] = l_mut_96;
shared_memory_39[thread_index_in_threadgroup] = l_194;
} else {
shared_memory_39[thread_index_in_threadgroup] = float(1.0);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_209 = uint(0); l_mut_209 < uint(8); ++l_mut_209) {
const uint l_210 = thread_index_in_threadgroup * uint(8);
const uint l_211 = l_210 + l_mut_209;
const uint l_212 = l_211 / uint(16);
const uint l_213 = l_22 * uint(16);
const uint l_214 = l_213 + l_212;
l_mut_110 = float(0.0);
const bool l_215 = l_214 < l_4;
if (l_215) {
const float l_216 = shared_memory_35[l_211];
l_mut_110 = l_216;
}
const half l_217 = half(l_mut_110);
shared_memory_206[l_211] = l_217;
}

for (uint l_mut_218 = uint(0); l_mut_218 < uint(64); ++l_mut_218) {
const uint l_219 = thread_index_in_threadgroup * uint(64);
const uint l_220 = l_219 + l_mut_218;
const uint l_221 = l_220 / uint(128);
const uint l_222 = l_220 % uint(128);
const uint l_223 = l_mut_78 * uint(16);
const uint l_224 = l_223 + l_221;
l_mut_110 = float(0.0);
const bool l_225 = l_224 < l_25;
if (l_225) {
const bool l_226 = l_10 == uint(2);
if (l_226) {
const uint l_227 = l_18 * l_6;
const uint l_228 = l_227 + l_24;
const uint l_229 = l_228 * l_8;
const uint l_230 = l_229 + l_224;
const uint l_231 = l_230 * uint(128);
const uint l_232 = l_231 + l_222;
const float l_233 = buffer_2[l_232];
l_mut_96 = l_233;
} else {
const uint l_234 = l_18 * l_9;
const uint l_235 = l_224 / l_8;
const uint l_236 = l_234 + l_235;
const uint l_237 = buffer_3[l_236];
const uint l_238 = l_224 % l_8;
const bool l_239 = l_10 == uint(0);
if (l_239) {
const uint l_240 = l_237 * l_6;
const uint l_241 = l_240 + l_24;
const uint l_242 = l_241 * l_8;
const uint l_243 = l_242 + l_238;
const uint l_244 = l_243 * uint(128);
const uint l_245 = l_244 + l_222;
const float l_246 = buffer_2[l_245];
l_mut_51 = l_246;
} else {
const uint l_247 = l_237 * l_6;
const uint l_248 = l_247 + l_24;
const uint l_249 = l_248 * uint(128);
const uint l_250 = l_249 + l_222;
const uint l_251 = l_250 * l_8;
const uint l_252 = l_251 + l_238;
const float l_253 = buffer_2[l_252];
l_mut_51 = l_253;
}
l_mut_96 = l_mut_51;
}
l_mut_110 = l_mut_96;
}
const half l_254 = half(l_mut_110);
shared_memory_207[l_220] = l_254;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_255 = uint(0); l_mut_255 < uint(8); ++l_mut_255) {

for (uint l_mut_256 = uint(0); l_mut_256 < uint(2); ++l_mut_256) {

for (uint l_mut_257 = uint(0); l_mut_257 < uint(2); ++l_mut_257) {
simdgroup_float8x8 frag_acc_258;
frag_acc_258 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_259 = uint(0); l_mut_259 < uint(2); ++l_mut_259) {
simdgroup_half8x8 frag_a_268;
simdgroup_half8x8 frag_b_279;
const uint l_260 = l_mut_256 * uint(8);
const uint l_261 = l_260 * uint(16);
const uint l_262 = l_mut_259 * uint(8);
const uint l_263 = l_261 + l_262;
const uint l_264 = 256;
const uint l_265 = l_264 - uint(0);
const uint l_266 = uint(256) - l_263;
const uint l_267 = l_263 + uint(0);
simdgroup_load(frag_a_268, shared_memory_206 + l_267, uint(16), 0, false);
const uint l_269 = l_mut_259 * uint(8);
const uint l_270 = l_269 * uint(128);
const uint l_271 = l_mut_255 * uint(16);
const uint l_272 = l_270 + l_271;
const uint l_273 = l_mut_257 * uint(8);
const uint l_274 = l_272 + l_273;
const uint l_275 = 2048;
const uint l_276 = l_275 - uint(0);
const uint l_277 = uint(2048) - l_274;
const uint l_278 = l_274 + uint(0);
simdgroup_load(frag_b_279, shared_memory_207 + l_278, uint(128), 0, false);
simdgroup_multiply_accumulate(frag_acc_258, frag_a_268, frag_b_279, frag_acc_258);
}
const uint l_280 = l_mut_256 * uint(8);
const uint l_281 = l_280 * uint(16);
const uint l_282 = l_mut_257 * uint(8);
const uint l_283 = l_281 + l_282;
const uint l_284 = 256;
const uint l_285 = l_284 - uint(0);
const uint l_286 = uint(256) - l_283;
const uint l_287 = l_283 + uint(0);
simdgroup_store(frag_acc_258, shared_memory_208 + l_287, uint(16));
simdgroup_barrier(mem_flags::mem_none);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_288 = uint(0); l_mut_288 < uint(8); ++l_mut_288) {
const uint l_289 = thread_index_in_threadgroup * uint(8);
const uint l_290 = l_289 + l_mut_288;
const uint l_291 = l_290 / uint(16);
const uint l_292 = l_290 % uint(16);
const uint l_293 = l_291 * uint(128);
const uint l_294 = l_mut_255 * uint(16);
const uint l_295 = l_293 + l_294;
const uint l_296 = l_295 + l_292;
const float l_297 = shared_memory_36[l_296];
const float l_298 = shared_memory_39[l_291];
const float l_299 = l_297 * l_298;
const float l_300 = shared_memory_208[l_290];
const float l_301 = l_299 + l_300;
shared_memory_36[l_296] = l_301;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint l_mut_302 = uint(0); l_mut_302 < uint(64); ++l_mut_302) {
const uint l_303 = thread_index_in_threadgroup * uint(64);
const uint l_304 = l_303 + l_mut_302;
const uint l_305 = l_304 / uint(128);
const uint l_306 = l_304 % uint(128);
const uint l_307 = l_22 * uint(16);
const uint l_308 = l_307 + l_305;
const bool l_309 = l_308 < l_4;
if (l_309) {
l_mut_110 = float(1.0);
const float l_310 = shared_memory_38[l_305];
const bool l_311 = l_310 > float(0.0);
if (l_311) {
const float l_312 = shared_memory_38[l_305];
l_mut_110 = l_312;
}
const uint l_313 = l_308 * uint(128);
const uint l_314 = l_33 + l_313;
const uint l_315 = l_314 + l_306;
const float l_316 = shared_memory_36[l_304];
const float l_317 = l_316 / l_mut_110;
buffer_5[l_315] = l_317;
}
}

}
