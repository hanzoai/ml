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
threadgroup uchar dynamic_shared_mem[4128];
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_290 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_291 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[16]);
// Shared array size: 1024, 4096 bytes
threadgroup float* shared_memory_292 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[32]);
float l_arr_26[8];

float l_arr_27[8];

uint l_mut_14;
float l_mut_60;
float l_mut_61;
uint l_mut_63;
float l_mut_77;
float l_mut_186;
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
const uint l_25 = l_24 * uint(256);
const uint l_28 = l_1 * uint(8);
const uint l_29 = l_25 + l_28;
const uint l_30 = l_29 + uint(0);
const float l_31 = buffer_0[l_30];
l_arr_26[uint(0)] = l_31;
l_arr_27[uint(0)] = float(0.0);
const uint l_32 = l_1 * uint(8);
const uint l_33 = l_25 + l_32;
const uint l_34 = l_33 + uint(1);
const float l_35 = buffer_0[l_34];
l_arr_26[uint(1)] = l_35;
l_arr_27[uint(1)] = float(0.0);
const uint l_36 = l_1 * uint(8);
const uint l_37 = l_25 + l_36;
const uint l_38 = l_37 + uint(2);
const float l_39 = buffer_0[l_38];
l_arr_26[uint(2)] = l_39;
l_arr_27[uint(2)] = float(0.0);
const uint l_40 = l_1 * uint(8);
const uint l_41 = l_25 + l_40;
const uint l_42 = l_41 + uint(3);
const float l_43 = buffer_0[l_42];
l_arr_26[uint(3)] = l_43;
l_arr_27[uint(3)] = float(0.0);
const uint l_44 = l_1 * uint(8);
const uint l_45 = l_25 + l_44;
const uint l_46 = l_45 + uint(4);
const float l_47 = buffer_0[l_46];
l_arr_26[uint(4)] = l_47;
l_arr_27[uint(4)] = float(0.0);
const uint l_48 = l_1 * uint(8);
const uint l_49 = l_25 + l_48;
const uint l_50 = l_49 + uint(5);
const float l_51 = buffer_0[l_50];
l_arr_26[uint(5)] = l_51;
l_arr_27[uint(5)] = float(0.0);
const uint l_52 = l_1 * uint(8);
const uint l_53 = l_25 + l_52;
const uint l_54 = l_53 + uint(6);
const float l_55 = buffer_0[l_54];
l_arr_26[uint(6)] = l_55;
l_arr_27[uint(6)] = float(0.0);
const uint l_56 = l_1 * uint(8);
const uint l_57 = l_25 + l_56;
const uint l_58 = l_57 + uint(7);
const float l_59 = buffer_0[l_58];
l_arr_26[uint(7)] = l_59;
l_arr_27[uint(7)] = float(0.0);
l_mut_60 = float(-3.3999999521443642e38);
l_mut_61 = float(0.0);
const uint l_62 = l_19 + l_0;
l_mut_63 = l_62;
while (true) {
const bool l_64 = l_mut_63 < l_22;
const bool l_65 = !l_64;
if (l_65) {
break;}
const uint l_66 = l_9 * l_4;
const uint l_67 = l_66 + l_mut_63;
const uint l_68 = buffer_3[l_67];
const uint l_69 = l_68 * l_3;
const uint l_70 = l_69 + l_12;

for (uint l_mut_71 = uint(0); l_mut_71 < l_6; ++l_mut_71) {
const uint l_72 = l_mut_63 * l_6;
const uint l_73 = l_72 + l_mut_71;
const bool l_74 = l_73 >= l_mut_14;
const bool l_75 = l_73 < l_13;
const bool l_76 = l_74 && l_75;
if (l_76) {
l_mut_77 = float(0.0);
const uint l_78 = l_1 * uint(8);
const uint l_79 = l_78 + uint(0);
const uint l_80 = l_79 / uint(4);
const uint l_81 = l_79 % uint(4);
const uint l_82 = l_70 * uint(64);
const uint l_83 = l_82 + l_80;
const uint l_84 = l_83 * l_6;
const uint l_85 = l_84 + l_mut_71;
const uint l_86 = l_85 * uint(4);
const uint l_87 = l_86 + l_81;
const float l_88 = l_arr_26[uint(0)];
const float l_89 = buffer_1[l_87];
const float l_90 = l_88 * l_89;
l_mut_77 = l_mut_77 + l_90;
const uint l_91 = l_1 * uint(8);
const uint l_92 = l_91 + uint(1);
const uint l_93 = l_92 / uint(4);
const uint l_94 = l_92 % uint(4);
const uint l_95 = l_70 * uint(64);
const uint l_96 = l_95 + l_93;
const uint l_97 = l_96 * l_6;
const uint l_98 = l_97 + l_mut_71;
const uint l_99 = l_98 * uint(4);
const uint l_100 = l_99 + l_94;
const float l_101 = l_arr_26[uint(1)];
const float l_102 = buffer_1[l_100];
const float l_103 = l_101 * l_102;
l_mut_77 = l_mut_77 + l_103;
const uint l_104 = l_1 * uint(8);
const uint l_105 = l_104 + uint(2);
const uint l_106 = l_105 / uint(4);
const uint l_107 = l_105 % uint(4);
const uint l_108 = l_70 * uint(64);
const uint l_109 = l_108 + l_106;
const uint l_110 = l_109 * l_6;
const uint l_111 = l_110 + l_mut_71;
const uint l_112 = l_111 * uint(4);
const uint l_113 = l_112 + l_107;
const float l_114 = l_arr_26[uint(2)];
const float l_115 = buffer_1[l_113];
const float l_116 = l_114 * l_115;
l_mut_77 = l_mut_77 + l_116;
const uint l_117 = l_1 * uint(8);
const uint l_118 = l_117 + uint(3);
const uint l_119 = l_118 / uint(4);
const uint l_120 = l_118 % uint(4);
const uint l_121 = l_70 * uint(64);
const uint l_122 = l_121 + l_119;
const uint l_123 = l_122 * l_6;
const uint l_124 = l_123 + l_mut_71;
const uint l_125 = l_124 * uint(4);
const uint l_126 = l_125 + l_120;
const float l_127 = l_arr_26[uint(3)];
const float l_128 = buffer_1[l_126];
const float l_129 = l_127 * l_128;
l_mut_77 = l_mut_77 + l_129;
const uint l_130 = l_1 * uint(8);
const uint l_131 = l_130 + uint(4);
const uint l_132 = l_131 / uint(4);
const uint l_133 = l_131 % uint(4);
const uint l_134 = l_70 * uint(64);
const uint l_135 = l_134 + l_132;
const uint l_136 = l_135 * l_6;
const uint l_137 = l_136 + l_mut_71;
const uint l_138 = l_137 * uint(4);
const uint l_139 = l_138 + l_133;
const float l_140 = l_arr_26[uint(4)];
const float l_141 = buffer_1[l_139];
const float l_142 = l_140 * l_141;
l_mut_77 = l_mut_77 + l_142;
const uint l_143 = l_1 * uint(8);
const uint l_144 = l_143 + uint(5);
const uint l_145 = l_144 / uint(4);
const uint l_146 = l_144 % uint(4);
const uint l_147 = l_70 * uint(64);
const uint l_148 = l_147 + l_145;
const uint l_149 = l_148 * l_6;
const uint l_150 = l_149 + l_mut_71;
const uint l_151 = l_150 * uint(4);
const uint l_152 = l_151 + l_146;
const float l_153 = l_arr_26[uint(5)];
const float l_154 = buffer_1[l_152];
const float l_155 = l_153 * l_154;
l_mut_77 = l_mut_77 + l_155;
const uint l_156 = l_1 * uint(8);
const uint l_157 = l_156 + uint(6);
const uint l_158 = l_157 / uint(4);
const uint l_159 = l_157 % uint(4);
const uint l_160 = l_70 * uint(64);
const uint l_161 = l_160 + l_158;
const uint l_162 = l_161 * l_6;
const uint l_163 = l_162 + l_mut_71;
const uint l_164 = l_163 * uint(4);
const uint l_165 = l_164 + l_159;
const float l_166 = l_arr_26[uint(6)];
const float l_167 = buffer_1[l_165];
const float l_168 = l_166 * l_167;
l_mut_77 = l_mut_77 + l_168;
const uint l_169 = l_1 * uint(8);
const uint l_170 = l_169 + uint(7);
const uint l_171 = l_170 / uint(4);
const uint l_172 = l_170 % uint(4);
const uint l_173 = l_70 * uint(64);
const uint l_174 = l_173 + l_171;
const uint l_175 = l_174 * l_6;
const uint l_176 = l_175 + l_mut_71;
const uint l_177 = l_176 * uint(4);
const uint l_178 = l_177 + l_172;
const float l_179 = l_arr_26[uint(7)];
const float l_180 = buffer_1[l_178];
const float l_181 = l_179 * l_180;
l_mut_77 = l_mut_77 + l_181;
const float l_182 = float {simd_sum(l_mut_77)
};
const float l_183 = l_182 * l_7;
l_mut_77 = l_183;
const bool l_184 = l_8 > float(0.0);
if (l_184) {
const float l_185 = l_mut_77 / l_8;
l_mut_186 = l_185;
const bool l_187 = l_185 > float(10.0);
if (l_187) {
l_mut_186 = float(10.0);
} else {
const bool l_188 = l_185 < float(-10.0);
if (l_188) {
l_mut_186 = float(-10.0);
}
}
const float l_189 = float(2.0) * l_mut_186;
const float l_190 = exp(l_189);
const float l_191 = l_190 - float(1.0);
const float l_192 = l_190 + float(1.0);
const float l_193 = l_191 / l_192;
const float l_194 = l_8 * l_193;
l_mut_77 = l_194;
}
l_mut_186 = l_mut_60;
const bool l_195 = l_mut_77 > l_mut_60;
if (l_195) {
l_mut_186 = l_mut_77;
}
const float l_196 = l_mut_60 - l_mut_186;
const float l_197 = exp(l_196);
const float l_198 = l_mut_77 - l_mut_186;
const float l_199 = exp(l_198);
const float l_200 = l_mut_61 * l_197;
const float l_201 = l_200 + l_199;
l_mut_61 = l_201;
l_mut_60 = l_mut_186;
const uint l_202 = l_1 * uint(8);
const uint l_203 = l_202 + uint(0);
const uint l_204 = l_70 * uint(256);
const uint l_205 = l_204 + l_203;
const uint l_206 = l_205 * l_6;
const uint l_207 = l_206 + l_mut_71;
const float l_208 = l_arr_27[uint(0)];
const float l_209 = l_208 * l_197;
const float l_210 = buffer_2[l_207];
const float l_211 = l_199 * l_210;
const float l_212 = l_209 + l_211;
l_arr_27[uint(0)] = l_212;
const uint l_213 = l_1 * uint(8);
const uint l_214 = l_213 + uint(1);
const uint l_215 = l_70 * uint(256);
const uint l_216 = l_215 + l_214;
const uint l_217 = l_216 * l_6;
const uint l_218 = l_217 + l_mut_71;
const float l_219 = l_arr_27[uint(1)];
const float l_220 = l_219 * l_197;
const float l_221 = buffer_2[l_218];
const float l_222 = l_199 * l_221;
const float l_223 = l_220 + l_222;
l_arr_27[uint(1)] = l_223;
const uint l_224 = l_1 * uint(8);
const uint l_225 = l_224 + uint(2);
const uint l_226 = l_70 * uint(256);
const uint l_227 = l_226 + l_225;
const uint l_228 = l_227 * l_6;
const uint l_229 = l_228 + l_mut_71;
const float l_230 = l_arr_27[uint(2)];
const float l_231 = l_230 * l_197;
const float l_232 = buffer_2[l_229];
const float l_233 = l_199 * l_232;
const float l_234 = l_231 + l_233;
l_arr_27[uint(2)] = l_234;
const uint l_235 = l_1 * uint(8);
const uint l_236 = l_235 + uint(3);
const uint l_237 = l_70 * uint(256);
const uint l_238 = l_237 + l_236;
const uint l_239 = l_238 * l_6;
const uint l_240 = l_239 + l_mut_71;
const float l_241 = l_arr_27[uint(3)];
const float l_242 = l_241 * l_197;
const float l_243 = buffer_2[l_240];
const float l_244 = l_199 * l_243;
const float l_245 = l_242 + l_244;
l_arr_27[uint(3)] = l_245;
const uint l_246 = l_1 * uint(8);
const uint l_247 = l_246 + uint(4);
const uint l_248 = l_70 * uint(256);
const uint l_249 = l_248 + l_247;
const uint l_250 = l_249 * l_6;
const uint l_251 = l_250 + l_mut_71;
const float l_252 = l_arr_27[uint(4)];
const float l_253 = l_252 * l_197;
const float l_254 = buffer_2[l_251];
const float l_255 = l_199 * l_254;
const float l_256 = l_253 + l_255;
l_arr_27[uint(4)] = l_256;
const uint l_257 = l_1 * uint(8);
const uint l_258 = l_257 + uint(5);
const uint l_259 = l_70 * uint(256);
const uint l_260 = l_259 + l_258;
const uint l_261 = l_260 * l_6;
const uint l_262 = l_261 + l_mut_71;
const float l_263 = l_arr_27[uint(5)];
const float l_264 = l_263 * l_197;
const float l_265 = buffer_2[l_262];
const float l_266 = l_199 * l_265;
const float l_267 = l_264 + l_266;
l_arr_27[uint(5)] = l_267;
const uint l_268 = l_1 * uint(8);
const uint l_269 = l_268 + uint(6);
const uint l_270 = l_70 * uint(256);
const uint l_271 = l_270 + l_269;
const uint l_272 = l_271 * l_6;
const uint l_273 = l_272 + l_mut_71;
const float l_274 = l_arr_27[uint(6)];
const float l_275 = l_274 * l_197;
const float l_276 = buffer_2[l_273];
const float l_277 = l_199 * l_276;
const float l_278 = l_275 + l_277;
l_arr_27[uint(6)] = l_278;
const uint l_279 = l_1 * uint(8);
const uint l_280 = l_279 + uint(7);
const uint l_281 = l_70 * uint(256);
const uint l_282 = l_281 + l_280;
const uint l_283 = l_282 * l_6;
const uint l_284 = l_283 + l_mut_71;
const float l_285 = l_arr_27[uint(7)];
const float l_286 = l_285 * l_197;
const float l_287 = buffer_2[l_284];
const float l_288 = l_199 * l_287;
const float l_289 = l_286 + l_288;
l_arr_27[uint(7)] = l_289;
}
}
l_mut_63 = l_mut_63 + uint(4);
}
const bool l_293 = l_1 == uint(0);
if (l_293) {
shared_memory_290[l_0] = l_mut_60;
shared_memory_291[l_0] = l_mut_61;
}
const uint l_294 = l_0 * uint(256);
const uint l_295 = l_1 * uint(8);
const uint l_296 = l_294 + l_295;
const uint l_297 = l_296 + uint(0);
const float l_298 = l_arr_27[uint(0)];
shared_memory_292[l_297] = l_298;
const uint l_299 = l_0 * uint(256);
const uint l_300 = l_1 * uint(8);
const uint l_301 = l_299 + l_300;
const uint l_302 = l_301 + uint(1);
const float l_303 = l_arr_27[uint(1)];
shared_memory_292[l_302] = l_303;
const uint l_304 = l_0 * uint(256);
const uint l_305 = l_1 * uint(8);
const uint l_306 = l_304 + l_305;
const uint l_307 = l_306 + uint(2);
const float l_308 = l_arr_27[uint(2)];
shared_memory_292[l_307] = l_308;
const uint l_309 = l_0 * uint(256);
const uint l_310 = l_1 * uint(8);
const uint l_311 = l_309 + l_310;
const uint l_312 = l_311 + uint(3);
const float l_313 = l_arr_27[uint(3)];
shared_memory_292[l_312] = l_313;
const uint l_314 = l_0 * uint(256);
const uint l_315 = l_1 * uint(8);
const uint l_316 = l_314 + l_315;
const uint l_317 = l_316 + uint(4);
const float l_318 = l_arr_27[uint(4)];
shared_memory_292[l_317] = l_318;
const uint l_319 = l_0 * uint(256);
const uint l_320 = l_1 * uint(8);
const uint l_321 = l_319 + l_320;
const uint l_322 = l_321 + uint(5);
const float l_323 = l_arr_27[uint(5)];
shared_memory_292[l_322] = l_323;
const uint l_324 = l_0 * uint(256);
const uint l_325 = l_1 * uint(8);
const uint l_326 = l_324 + l_325;
const uint l_327 = l_326 + uint(6);
const float l_328 = l_arr_27[uint(6)];
shared_memory_292[l_327] = l_328;
const uint l_329 = l_0 * uint(256);
const uint l_330 = l_1 * uint(8);
const uint l_331 = l_329 + l_330;
const uint l_332 = l_331 + uint(7);
const float l_333 = l_arr_27[uint(7)];
shared_memory_292[l_332] = l_333;
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_334 = l_0 == uint(0);
if (l_334) {
l_mut_186 = float(-3.3999999521443642e38);

for (uint l_mut_335 = uint(0); l_mut_335 < uint(4); ++l_mut_335) {
const float l_336 = shared_memory_291[l_mut_335];
const bool l_337 = l_336 > float(0.0);
const float l_338 = shared_memory_290[l_mut_335];
const bool l_339 = l_338 > l_mut_186;
const bool l_340 = l_337 && l_339;
if (l_340) {
const float l_341 = shared_memory_290[l_mut_335];
l_mut_186 = l_341;
}
}
l_mut_77 = float(0.0);

for (uint l_mut_342 = uint(0); l_mut_342 < uint(4); ++l_mut_342) {
const float l_343 = shared_memory_291[l_mut_342];
const bool l_344 = l_343 > float(0.0);
if (l_344) {
const float l_345 = shared_memory_291[l_mut_342];
const float l_346 = shared_memory_290[l_mut_342];
const float l_347 = l_346 - l_mut_186;
const float l_348 = exp(l_347);
const float l_349 = l_345 * l_348;
l_mut_77 = l_mut_77 + l_349;
}
}
l_mut_61 = float(0.0);
const bool l_350 = l_mut_77 > float(0.0);
if (l_350) {
const float l_351 = float(1.0) / l_mut_77;
l_mut_61 = l_351;
}
const uint l_352 = l_1 * uint(8);
const uint l_353 = l_352 + uint(0);
l_mut_77 = float(0.0);

for (uint l_mut_354 = uint(0); l_mut_354 < uint(4); ++l_mut_354) {
const float l_355 = shared_memory_291[l_mut_354];
const bool l_356 = l_355 > float(0.0);
if (l_356) {
const uint l_357 = l_mut_354 * uint(256);
const uint l_358 = l_357 + l_353;
const float l_359 = shared_memory_292[l_358];
const float l_360 = shared_memory_290[l_mut_354];
const float l_361 = l_360 - l_mut_186;
const float l_362 = exp(l_361);
const float l_363 = l_359 * l_362;
l_mut_77 = l_mut_77 + l_363;
}
}
const uint l_364 = l_25 + l_353;
const float l_365 = l_mut_77 * l_mut_61;
buffer_5[l_364] = l_365;
const uint l_366 = l_1 * uint(8);
const uint l_367 = l_366 + uint(1);
l_mut_77 = float(0.0);

for (uint l_mut_368 = uint(0); l_mut_368 < uint(4); ++l_mut_368) {
const float l_369 = shared_memory_291[l_mut_368];
const bool l_370 = l_369 > float(0.0);
if (l_370) {
const uint l_371 = l_mut_368 * uint(256);
const uint l_372 = l_371 + l_367;
const float l_373 = shared_memory_292[l_372];
const float l_374 = shared_memory_290[l_mut_368];
const float l_375 = l_374 - l_mut_186;
const float l_376 = exp(l_375);
const float l_377 = l_373 * l_376;
l_mut_77 = l_mut_77 + l_377;
}
}
const uint l_378 = l_25 + l_367;
const float l_379 = l_mut_77 * l_mut_61;
buffer_5[l_378] = l_379;
const uint l_380 = l_1 * uint(8);
const uint l_381 = l_380 + uint(2);
l_mut_77 = float(0.0);

for (uint l_mut_382 = uint(0); l_mut_382 < uint(4); ++l_mut_382) {
const float l_383 = shared_memory_291[l_mut_382];
const bool l_384 = l_383 > float(0.0);
if (l_384) {
const uint l_385 = l_mut_382 * uint(256);
const uint l_386 = l_385 + l_381;
const float l_387 = shared_memory_292[l_386];
const float l_388 = shared_memory_290[l_mut_382];
const float l_389 = l_388 - l_mut_186;
const float l_390 = exp(l_389);
const float l_391 = l_387 * l_390;
l_mut_77 = l_mut_77 + l_391;
}
}
const uint l_392 = l_25 + l_381;
const float l_393 = l_mut_77 * l_mut_61;
buffer_5[l_392] = l_393;
const uint l_394 = l_1 * uint(8);
const uint l_395 = l_394 + uint(3);
l_mut_77 = float(0.0);

for (uint l_mut_396 = uint(0); l_mut_396 < uint(4); ++l_mut_396) {
const float l_397 = shared_memory_291[l_mut_396];
const bool l_398 = l_397 > float(0.0);
if (l_398) {
const uint l_399 = l_mut_396 * uint(256);
const uint l_400 = l_399 + l_395;
const float l_401 = shared_memory_292[l_400];
const float l_402 = shared_memory_290[l_mut_396];
const float l_403 = l_402 - l_mut_186;
const float l_404 = exp(l_403);
const float l_405 = l_401 * l_404;
l_mut_77 = l_mut_77 + l_405;
}
}
const uint l_406 = l_25 + l_395;
const float l_407 = l_mut_77 * l_mut_61;
buffer_5[l_406] = l_407;
const uint l_408 = l_1 * uint(8);
const uint l_409 = l_408 + uint(4);
l_mut_77 = float(0.0);

for (uint l_mut_410 = uint(0); l_mut_410 < uint(4); ++l_mut_410) {
const float l_411 = shared_memory_291[l_mut_410];
const bool l_412 = l_411 > float(0.0);
if (l_412) {
const uint l_413 = l_mut_410 * uint(256);
const uint l_414 = l_413 + l_409;
const float l_415 = shared_memory_292[l_414];
const float l_416 = shared_memory_290[l_mut_410];
const float l_417 = l_416 - l_mut_186;
const float l_418 = exp(l_417);
const float l_419 = l_415 * l_418;
l_mut_77 = l_mut_77 + l_419;
}
}
const uint l_420 = l_25 + l_409;
const float l_421 = l_mut_77 * l_mut_61;
buffer_5[l_420] = l_421;
const uint l_422 = l_1 * uint(8);
const uint l_423 = l_422 + uint(5);
l_mut_77 = float(0.0);

for (uint l_mut_424 = uint(0); l_mut_424 < uint(4); ++l_mut_424) {
const float l_425 = shared_memory_291[l_mut_424];
const bool l_426 = l_425 > float(0.0);
if (l_426) {
const uint l_427 = l_mut_424 * uint(256);
const uint l_428 = l_427 + l_423;
const float l_429 = shared_memory_292[l_428];
const float l_430 = shared_memory_290[l_mut_424];
const float l_431 = l_430 - l_mut_186;
const float l_432 = exp(l_431);
const float l_433 = l_429 * l_432;
l_mut_77 = l_mut_77 + l_433;
}
}
const uint l_434 = l_25 + l_423;
const float l_435 = l_mut_77 * l_mut_61;
buffer_5[l_434] = l_435;
const uint l_436 = l_1 * uint(8);
const uint l_437 = l_436 + uint(6);
l_mut_77 = float(0.0);

for (uint l_mut_438 = uint(0); l_mut_438 < uint(4); ++l_mut_438) {
const float l_439 = shared_memory_291[l_mut_438];
const bool l_440 = l_439 > float(0.0);
if (l_440) {
const uint l_441 = l_mut_438 * uint(256);
const uint l_442 = l_441 + l_437;
const float l_443 = shared_memory_292[l_442];
const float l_444 = shared_memory_290[l_mut_438];
const float l_445 = l_444 - l_mut_186;
const float l_446 = exp(l_445);
const float l_447 = l_443 * l_446;
l_mut_77 = l_mut_77 + l_447;
}
}
const uint l_448 = l_25 + l_437;
const float l_449 = l_mut_77 * l_mut_61;
buffer_5[l_448] = l_449;
const uint l_450 = l_1 * uint(8);
const uint l_451 = l_450 + uint(7);
l_mut_77 = float(0.0);

for (uint l_mut_452 = uint(0); l_mut_452 < uint(4); ++l_mut_452) {
const float l_453 = shared_memory_291[l_mut_452];
const bool l_454 = l_453 > float(0.0);
if (l_454) {
const uint l_455 = l_mut_452 * uint(256);
const uint l_456 = l_455 + l_451;
const float l_457 = shared_memory_292[l_456];
const float l_458 = shared_memory_290[l_mut_452];
const float l_459 = l_458 - l_mut_186;
const float l_460 = exp(l_459);
const float l_461 = l_457 * l_460;
l_mut_77 = l_mut_77 + l_461;
}
}
const uint l_462 = l_25 + l_451;
const float l_463 = l_mut_77 * l_mut_61;
buffer_5[l_462] = l_463;
}

}
