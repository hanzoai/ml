#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[16];
};

[[kernel]]
void paged_attn_f_(
    const device bfloat * buffer_0 [[buffer(0)]],
    const device bfloat * buffer_1 [[buffer(1)]],
    const device bfloat * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    device bfloat * buffer_5 [[buffer(5)]],
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
threadgroup float* shared_memory_314 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_315 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[16]);
// Shared array size: 1024, 4096 bytes
threadgroup float* shared_memory_316 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[32]);
float l_arr_26[8];

float l_arr_27[8];

uint l_mut_14;
float l_mut_68;
float l_mut_69;
uint l_mut_71;
float l_mut_85;
float l_mut_202;
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
const bfloat l_31 = buffer_0[l_30];
const float l_32 = float(l_31);
l_arr_26[uint(0)] = l_32;
l_arr_27[uint(0)] = float(0.0);
const uint l_33 = l_1 * uint(8);
const uint l_34 = l_25 + l_33;
const uint l_35 = l_34 + uint(1);
const bfloat l_36 = buffer_0[l_35];
const float l_37 = float(l_36);
l_arr_26[uint(1)] = l_37;
l_arr_27[uint(1)] = float(0.0);
const uint l_38 = l_1 * uint(8);
const uint l_39 = l_25 + l_38;
const uint l_40 = l_39 + uint(2);
const bfloat l_41 = buffer_0[l_40];
const float l_42 = float(l_41);
l_arr_26[uint(2)] = l_42;
l_arr_27[uint(2)] = float(0.0);
const uint l_43 = l_1 * uint(8);
const uint l_44 = l_25 + l_43;
const uint l_45 = l_44 + uint(3);
const bfloat l_46 = buffer_0[l_45];
const float l_47 = float(l_46);
l_arr_26[uint(3)] = l_47;
l_arr_27[uint(3)] = float(0.0);
const uint l_48 = l_1 * uint(8);
const uint l_49 = l_25 + l_48;
const uint l_50 = l_49 + uint(4);
const bfloat l_51 = buffer_0[l_50];
const float l_52 = float(l_51);
l_arr_26[uint(4)] = l_52;
l_arr_27[uint(4)] = float(0.0);
const uint l_53 = l_1 * uint(8);
const uint l_54 = l_25 + l_53;
const uint l_55 = l_54 + uint(5);
const bfloat l_56 = buffer_0[l_55];
const float l_57 = float(l_56);
l_arr_26[uint(5)] = l_57;
l_arr_27[uint(5)] = float(0.0);
const uint l_58 = l_1 * uint(8);
const uint l_59 = l_25 + l_58;
const uint l_60 = l_59 + uint(6);
const bfloat l_61 = buffer_0[l_60];
const float l_62 = float(l_61);
l_arr_26[uint(6)] = l_62;
l_arr_27[uint(6)] = float(0.0);
const uint l_63 = l_1 * uint(8);
const uint l_64 = l_25 + l_63;
const uint l_65 = l_64 + uint(7);
const bfloat l_66 = buffer_0[l_65];
const float l_67 = float(l_66);
l_arr_26[uint(7)] = l_67;
l_arr_27[uint(7)] = float(0.0);
l_mut_68 = float(-3.3999999521443642e38);
l_mut_69 = float(0.0);
const uint l_70 = l_19 + l_0;
l_mut_71 = l_70;
while (true) {
const bool l_72 = l_mut_71 < l_22;
const bool l_73 = !l_72;
if (l_73) {
break;}
const uint l_74 = l_9 * l_4;
const uint l_75 = l_74 + l_mut_71;
const uint l_76 = buffer_3[l_75];
const uint l_77 = l_76 * l_3;
const uint l_78 = l_77 + l_12;

for (uint l_mut_79 = uint(0); l_mut_79 < l_6; ++l_mut_79) {
const uint l_80 = l_mut_71 * l_6;
const uint l_81 = l_80 + l_mut_79;
const bool l_82 = l_81 >= l_mut_14;
const bool l_83 = l_81 < l_13;
const bool l_84 = l_82 && l_83;
if (l_84) {
l_mut_85 = float(0.0);
const uint l_86 = l_1 * uint(8);
const uint l_87 = l_86 + uint(0);
const uint l_88 = l_87 / uint(8);
const uint l_89 = l_87 % uint(8);
const uint l_90 = l_78 * uint(32);
const uint l_91 = l_90 + l_88;
const uint l_92 = l_91 * l_6;
const uint l_93 = l_92 + l_mut_79;
const uint l_94 = l_93 * uint(8);
const uint l_95 = l_94 + l_89;
const float l_96 = l_arr_26[uint(0)];
const bfloat l_97 = buffer_1[l_95];
const float l_98 = float(l_97);
const float l_99 = l_96 * l_98;
l_mut_85 = l_mut_85 + l_99;
const uint l_100 = l_1 * uint(8);
const uint l_101 = l_100 + uint(1);
const uint l_102 = l_101 / uint(8);
const uint l_103 = l_101 % uint(8);
const uint l_104 = l_78 * uint(32);
const uint l_105 = l_104 + l_102;
const uint l_106 = l_105 * l_6;
const uint l_107 = l_106 + l_mut_79;
const uint l_108 = l_107 * uint(8);
const uint l_109 = l_108 + l_103;
const float l_110 = l_arr_26[uint(1)];
const bfloat l_111 = buffer_1[l_109];
const float l_112 = float(l_111);
const float l_113 = l_110 * l_112;
l_mut_85 = l_mut_85 + l_113;
const uint l_114 = l_1 * uint(8);
const uint l_115 = l_114 + uint(2);
const uint l_116 = l_115 / uint(8);
const uint l_117 = l_115 % uint(8);
const uint l_118 = l_78 * uint(32);
const uint l_119 = l_118 + l_116;
const uint l_120 = l_119 * l_6;
const uint l_121 = l_120 + l_mut_79;
const uint l_122 = l_121 * uint(8);
const uint l_123 = l_122 + l_117;
const float l_124 = l_arr_26[uint(2)];
const bfloat l_125 = buffer_1[l_123];
const float l_126 = float(l_125);
const float l_127 = l_124 * l_126;
l_mut_85 = l_mut_85 + l_127;
const uint l_128 = l_1 * uint(8);
const uint l_129 = l_128 + uint(3);
const uint l_130 = l_129 / uint(8);
const uint l_131 = l_129 % uint(8);
const uint l_132 = l_78 * uint(32);
const uint l_133 = l_132 + l_130;
const uint l_134 = l_133 * l_6;
const uint l_135 = l_134 + l_mut_79;
const uint l_136 = l_135 * uint(8);
const uint l_137 = l_136 + l_131;
const float l_138 = l_arr_26[uint(3)];
const bfloat l_139 = buffer_1[l_137];
const float l_140 = float(l_139);
const float l_141 = l_138 * l_140;
l_mut_85 = l_mut_85 + l_141;
const uint l_142 = l_1 * uint(8);
const uint l_143 = l_142 + uint(4);
const uint l_144 = l_143 / uint(8);
const uint l_145 = l_143 % uint(8);
const uint l_146 = l_78 * uint(32);
const uint l_147 = l_146 + l_144;
const uint l_148 = l_147 * l_6;
const uint l_149 = l_148 + l_mut_79;
const uint l_150 = l_149 * uint(8);
const uint l_151 = l_150 + l_145;
const float l_152 = l_arr_26[uint(4)];
const bfloat l_153 = buffer_1[l_151];
const float l_154 = float(l_153);
const float l_155 = l_152 * l_154;
l_mut_85 = l_mut_85 + l_155;
const uint l_156 = l_1 * uint(8);
const uint l_157 = l_156 + uint(5);
const uint l_158 = l_157 / uint(8);
const uint l_159 = l_157 % uint(8);
const uint l_160 = l_78 * uint(32);
const uint l_161 = l_160 + l_158;
const uint l_162 = l_161 * l_6;
const uint l_163 = l_162 + l_mut_79;
const uint l_164 = l_163 * uint(8);
const uint l_165 = l_164 + l_159;
const float l_166 = l_arr_26[uint(5)];
const bfloat l_167 = buffer_1[l_165];
const float l_168 = float(l_167);
const float l_169 = l_166 * l_168;
l_mut_85 = l_mut_85 + l_169;
const uint l_170 = l_1 * uint(8);
const uint l_171 = l_170 + uint(6);
const uint l_172 = l_171 / uint(8);
const uint l_173 = l_171 % uint(8);
const uint l_174 = l_78 * uint(32);
const uint l_175 = l_174 + l_172;
const uint l_176 = l_175 * l_6;
const uint l_177 = l_176 + l_mut_79;
const uint l_178 = l_177 * uint(8);
const uint l_179 = l_178 + l_173;
const float l_180 = l_arr_26[uint(6)];
const bfloat l_181 = buffer_1[l_179];
const float l_182 = float(l_181);
const float l_183 = l_180 * l_182;
l_mut_85 = l_mut_85 + l_183;
const uint l_184 = l_1 * uint(8);
const uint l_185 = l_184 + uint(7);
const uint l_186 = l_185 / uint(8);
const uint l_187 = l_185 % uint(8);
const uint l_188 = l_78 * uint(32);
const uint l_189 = l_188 + l_186;
const uint l_190 = l_189 * l_6;
const uint l_191 = l_190 + l_mut_79;
const uint l_192 = l_191 * uint(8);
const uint l_193 = l_192 + l_187;
const float l_194 = l_arr_26[uint(7)];
const bfloat l_195 = buffer_1[l_193];
const float l_196 = float(l_195);
const float l_197 = l_194 * l_196;
l_mut_85 = l_mut_85 + l_197;
const float l_198 = float {simd_sum(l_mut_85)
};
const float l_199 = l_198 * l_7;
l_mut_85 = l_199;
const bool l_200 = l_8 > float(0.0);
if (l_200) {
const float l_201 = l_mut_85 / l_8;
l_mut_202 = l_201;
const bool l_203 = l_201 > float(10.0);
if (l_203) {
l_mut_202 = float(10.0);
} else {
const bool l_204 = l_201 < float(-10.0);
if (l_204) {
l_mut_202 = float(-10.0);
}
}
const float l_205 = float(2.0) * l_mut_202;
const float l_206 = exp(l_205);
const float l_207 = l_206 - float(1.0);
const float l_208 = l_206 + float(1.0);
const float l_209 = l_207 / l_208;
const float l_210 = l_8 * l_209;
l_mut_85 = l_210;
}
l_mut_202 = l_mut_68;
const bool l_211 = l_mut_85 > l_mut_68;
if (l_211) {
l_mut_202 = l_mut_85;
}
const float l_212 = l_mut_68 - l_mut_202;
const float l_213 = exp(l_212);
const float l_214 = l_mut_85 - l_mut_202;
const float l_215 = exp(l_214);
const float l_216 = l_mut_69 * l_213;
const float l_217 = l_216 + l_215;
l_mut_69 = l_217;
l_mut_68 = l_mut_202;
const uint l_218 = l_1 * uint(8);
const uint l_219 = l_218 + uint(0);
const uint l_220 = l_78 * uint(256);
const uint l_221 = l_220 + l_219;
const uint l_222 = l_221 * l_6;
const uint l_223 = l_222 + l_mut_79;
const float l_224 = l_arr_27[uint(0)];
const float l_225 = l_224 * l_213;
const bfloat l_226 = buffer_2[l_223];
const float l_227 = float(l_226);
const float l_228 = l_215 * l_227;
const float l_229 = l_225 + l_228;
l_arr_27[uint(0)] = l_229;
const uint l_230 = l_1 * uint(8);
const uint l_231 = l_230 + uint(1);
const uint l_232 = l_78 * uint(256);
const uint l_233 = l_232 + l_231;
const uint l_234 = l_233 * l_6;
const uint l_235 = l_234 + l_mut_79;
const float l_236 = l_arr_27[uint(1)];
const float l_237 = l_236 * l_213;
const bfloat l_238 = buffer_2[l_235];
const float l_239 = float(l_238);
const float l_240 = l_215 * l_239;
const float l_241 = l_237 + l_240;
l_arr_27[uint(1)] = l_241;
const uint l_242 = l_1 * uint(8);
const uint l_243 = l_242 + uint(2);
const uint l_244 = l_78 * uint(256);
const uint l_245 = l_244 + l_243;
const uint l_246 = l_245 * l_6;
const uint l_247 = l_246 + l_mut_79;
const float l_248 = l_arr_27[uint(2)];
const float l_249 = l_248 * l_213;
const bfloat l_250 = buffer_2[l_247];
const float l_251 = float(l_250);
const float l_252 = l_215 * l_251;
const float l_253 = l_249 + l_252;
l_arr_27[uint(2)] = l_253;
const uint l_254 = l_1 * uint(8);
const uint l_255 = l_254 + uint(3);
const uint l_256 = l_78 * uint(256);
const uint l_257 = l_256 + l_255;
const uint l_258 = l_257 * l_6;
const uint l_259 = l_258 + l_mut_79;
const float l_260 = l_arr_27[uint(3)];
const float l_261 = l_260 * l_213;
const bfloat l_262 = buffer_2[l_259];
const float l_263 = float(l_262);
const float l_264 = l_215 * l_263;
const float l_265 = l_261 + l_264;
l_arr_27[uint(3)] = l_265;
const uint l_266 = l_1 * uint(8);
const uint l_267 = l_266 + uint(4);
const uint l_268 = l_78 * uint(256);
const uint l_269 = l_268 + l_267;
const uint l_270 = l_269 * l_6;
const uint l_271 = l_270 + l_mut_79;
const float l_272 = l_arr_27[uint(4)];
const float l_273 = l_272 * l_213;
const bfloat l_274 = buffer_2[l_271];
const float l_275 = float(l_274);
const float l_276 = l_215 * l_275;
const float l_277 = l_273 + l_276;
l_arr_27[uint(4)] = l_277;
const uint l_278 = l_1 * uint(8);
const uint l_279 = l_278 + uint(5);
const uint l_280 = l_78 * uint(256);
const uint l_281 = l_280 + l_279;
const uint l_282 = l_281 * l_6;
const uint l_283 = l_282 + l_mut_79;
const float l_284 = l_arr_27[uint(5)];
const float l_285 = l_284 * l_213;
const bfloat l_286 = buffer_2[l_283];
const float l_287 = float(l_286);
const float l_288 = l_215 * l_287;
const float l_289 = l_285 + l_288;
l_arr_27[uint(5)] = l_289;
const uint l_290 = l_1 * uint(8);
const uint l_291 = l_290 + uint(6);
const uint l_292 = l_78 * uint(256);
const uint l_293 = l_292 + l_291;
const uint l_294 = l_293 * l_6;
const uint l_295 = l_294 + l_mut_79;
const float l_296 = l_arr_27[uint(6)];
const float l_297 = l_296 * l_213;
const bfloat l_298 = buffer_2[l_295];
const float l_299 = float(l_298);
const float l_300 = l_215 * l_299;
const float l_301 = l_297 + l_300;
l_arr_27[uint(6)] = l_301;
const uint l_302 = l_1 * uint(8);
const uint l_303 = l_302 + uint(7);
const uint l_304 = l_78 * uint(256);
const uint l_305 = l_304 + l_303;
const uint l_306 = l_305 * l_6;
const uint l_307 = l_306 + l_mut_79;
const float l_308 = l_arr_27[uint(7)];
const float l_309 = l_308 * l_213;
const bfloat l_310 = buffer_2[l_307];
const float l_311 = float(l_310);
const float l_312 = l_215 * l_311;
const float l_313 = l_309 + l_312;
l_arr_27[uint(7)] = l_313;
}
}
l_mut_71 = l_mut_71 + uint(4);
}
const bool l_317 = l_1 == uint(0);
if (l_317) {
shared_memory_314[l_0] = l_mut_68;
shared_memory_315[l_0] = l_mut_69;
}
const uint l_318 = l_0 * uint(256);
const uint l_319 = l_1 * uint(8);
const uint l_320 = l_318 + l_319;
const uint l_321 = l_320 + uint(0);
const float l_322 = l_arr_27[uint(0)];
shared_memory_316[l_321] = l_322;
const uint l_323 = l_0 * uint(256);
const uint l_324 = l_1 * uint(8);
const uint l_325 = l_323 + l_324;
const uint l_326 = l_325 + uint(1);
const float l_327 = l_arr_27[uint(1)];
shared_memory_316[l_326] = l_327;
const uint l_328 = l_0 * uint(256);
const uint l_329 = l_1 * uint(8);
const uint l_330 = l_328 + l_329;
const uint l_331 = l_330 + uint(2);
const float l_332 = l_arr_27[uint(2)];
shared_memory_316[l_331] = l_332;
const uint l_333 = l_0 * uint(256);
const uint l_334 = l_1 * uint(8);
const uint l_335 = l_333 + l_334;
const uint l_336 = l_335 + uint(3);
const float l_337 = l_arr_27[uint(3)];
shared_memory_316[l_336] = l_337;
const uint l_338 = l_0 * uint(256);
const uint l_339 = l_1 * uint(8);
const uint l_340 = l_338 + l_339;
const uint l_341 = l_340 + uint(4);
const float l_342 = l_arr_27[uint(4)];
shared_memory_316[l_341] = l_342;
const uint l_343 = l_0 * uint(256);
const uint l_344 = l_1 * uint(8);
const uint l_345 = l_343 + l_344;
const uint l_346 = l_345 + uint(5);
const float l_347 = l_arr_27[uint(5)];
shared_memory_316[l_346] = l_347;
const uint l_348 = l_0 * uint(256);
const uint l_349 = l_1 * uint(8);
const uint l_350 = l_348 + l_349;
const uint l_351 = l_350 + uint(6);
const float l_352 = l_arr_27[uint(6)];
shared_memory_316[l_351] = l_352;
const uint l_353 = l_0 * uint(256);
const uint l_354 = l_1 * uint(8);
const uint l_355 = l_353 + l_354;
const uint l_356 = l_355 + uint(7);
const float l_357 = l_arr_27[uint(7)];
shared_memory_316[l_356] = l_357;
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_358 = l_0 == uint(0);
if (l_358) {
l_mut_202 = float(-3.3999999521443642e38);

for (uint l_mut_359 = uint(0); l_mut_359 < uint(4); ++l_mut_359) {
const float l_360 = shared_memory_315[l_mut_359];
const bool l_361 = l_360 > float(0.0);
const float l_362 = shared_memory_314[l_mut_359];
const bool l_363 = l_362 > l_mut_202;
const bool l_364 = l_361 && l_363;
if (l_364) {
const float l_365 = shared_memory_314[l_mut_359];
l_mut_202 = l_365;
}
}
l_mut_85 = float(0.0);

for (uint l_mut_366 = uint(0); l_mut_366 < uint(4); ++l_mut_366) {
const float l_367 = shared_memory_315[l_mut_366];
const bool l_368 = l_367 > float(0.0);
if (l_368) {
const float l_369 = shared_memory_315[l_mut_366];
const float l_370 = shared_memory_314[l_mut_366];
const float l_371 = l_370 - l_mut_202;
const float l_372 = exp(l_371);
const float l_373 = l_369 * l_372;
l_mut_85 = l_mut_85 + l_373;
}
}
l_mut_69 = float(0.0);
const bool l_374 = l_mut_85 > float(0.0);
if (l_374) {
const float l_375 = float(1.0) / l_mut_85;
l_mut_69 = l_375;
}
const uint l_376 = l_1 * uint(8);
const uint l_377 = l_376 + uint(0);
l_mut_85 = float(0.0);

for (uint l_mut_378 = uint(0); l_mut_378 < uint(4); ++l_mut_378) {
const float l_379 = shared_memory_315[l_mut_378];
const bool l_380 = l_379 > float(0.0);
if (l_380) {
const uint l_381 = l_mut_378 * uint(256);
const uint l_382 = l_381 + l_377;
const float l_383 = shared_memory_316[l_382];
const float l_384 = shared_memory_314[l_mut_378];
const float l_385 = l_384 - l_mut_202;
const float l_386 = exp(l_385);
const float l_387 = l_383 * l_386;
l_mut_85 = l_mut_85 + l_387;
}
}
const uint l_388 = l_25 + l_377;
const float l_389 = l_mut_85 * l_mut_69;
const bfloat l_390 = bfloat(l_389);
buffer_5[l_388] = l_390;
const uint l_391 = l_1 * uint(8);
const uint l_392 = l_391 + uint(1);
l_mut_85 = float(0.0);

for (uint l_mut_393 = uint(0); l_mut_393 < uint(4); ++l_mut_393) {
const float l_394 = shared_memory_315[l_mut_393];
const bool l_395 = l_394 > float(0.0);
if (l_395) {
const uint l_396 = l_mut_393 * uint(256);
const uint l_397 = l_396 + l_392;
const float l_398 = shared_memory_316[l_397];
const float l_399 = shared_memory_314[l_mut_393];
const float l_400 = l_399 - l_mut_202;
const float l_401 = exp(l_400);
const float l_402 = l_398 * l_401;
l_mut_85 = l_mut_85 + l_402;
}
}
const uint l_403 = l_25 + l_392;
const float l_404 = l_mut_85 * l_mut_69;
const bfloat l_405 = bfloat(l_404);
buffer_5[l_403] = l_405;
const uint l_406 = l_1 * uint(8);
const uint l_407 = l_406 + uint(2);
l_mut_85 = float(0.0);

for (uint l_mut_408 = uint(0); l_mut_408 < uint(4); ++l_mut_408) {
const float l_409 = shared_memory_315[l_mut_408];
const bool l_410 = l_409 > float(0.0);
if (l_410) {
const uint l_411 = l_mut_408 * uint(256);
const uint l_412 = l_411 + l_407;
const float l_413 = shared_memory_316[l_412];
const float l_414 = shared_memory_314[l_mut_408];
const float l_415 = l_414 - l_mut_202;
const float l_416 = exp(l_415);
const float l_417 = l_413 * l_416;
l_mut_85 = l_mut_85 + l_417;
}
}
const uint l_418 = l_25 + l_407;
const float l_419 = l_mut_85 * l_mut_69;
const bfloat l_420 = bfloat(l_419);
buffer_5[l_418] = l_420;
const uint l_421 = l_1 * uint(8);
const uint l_422 = l_421 + uint(3);
l_mut_85 = float(0.0);

for (uint l_mut_423 = uint(0); l_mut_423 < uint(4); ++l_mut_423) {
const float l_424 = shared_memory_315[l_mut_423];
const bool l_425 = l_424 > float(0.0);
if (l_425) {
const uint l_426 = l_mut_423 * uint(256);
const uint l_427 = l_426 + l_422;
const float l_428 = shared_memory_316[l_427];
const float l_429 = shared_memory_314[l_mut_423];
const float l_430 = l_429 - l_mut_202;
const float l_431 = exp(l_430);
const float l_432 = l_428 * l_431;
l_mut_85 = l_mut_85 + l_432;
}
}
const uint l_433 = l_25 + l_422;
const float l_434 = l_mut_85 * l_mut_69;
const bfloat l_435 = bfloat(l_434);
buffer_5[l_433] = l_435;
const uint l_436 = l_1 * uint(8);
const uint l_437 = l_436 + uint(4);
l_mut_85 = float(0.0);

for (uint l_mut_438 = uint(0); l_mut_438 < uint(4); ++l_mut_438) {
const float l_439 = shared_memory_315[l_mut_438];
const bool l_440 = l_439 > float(0.0);
if (l_440) {
const uint l_441 = l_mut_438 * uint(256);
const uint l_442 = l_441 + l_437;
const float l_443 = shared_memory_316[l_442];
const float l_444 = shared_memory_314[l_mut_438];
const float l_445 = l_444 - l_mut_202;
const float l_446 = exp(l_445);
const float l_447 = l_443 * l_446;
l_mut_85 = l_mut_85 + l_447;
}
}
const uint l_448 = l_25 + l_437;
const float l_449 = l_mut_85 * l_mut_69;
const bfloat l_450 = bfloat(l_449);
buffer_5[l_448] = l_450;
const uint l_451 = l_1 * uint(8);
const uint l_452 = l_451 + uint(5);
l_mut_85 = float(0.0);

for (uint l_mut_453 = uint(0); l_mut_453 < uint(4); ++l_mut_453) {
const float l_454 = shared_memory_315[l_mut_453];
const bool l_455 = l_454 > float(0.0);
if (l_455) {
const uint l_456 = l_mut_453 * uint(256);
const uint l_457 = l_456 + l_452;
const float l_458 = shared_memory_316[l_457];
const float l_459 = shared_memory_314[l_mut_453];
const float l_460 = l_459 - l_mut_202;
const float l_461 = exp(l_460);
const float l_462 = l_458 * l_461;
l_mut_85 = l_mut_85 + l_462;
}
}
const uint l_463 = l_25 + l_452;
const float l_464 = l_mut_85 * l_mut_69;
const bfloat l_465 = bfloat(l_464);
buffer_5[l_463] = l_465;
const uint l_466 = l_1 * uint(8);
const uint l_467 = l_466 + uint(6);
l_mut_85 = float(0.0);

for (uint l_mut_468 = uint(0); l_mut_468 < uint(4); ++l_mut_468) {
const float l_469 = shared_memory_315[l_mut_468];
const bool l_470 = l_469 > float(0.0);
if (l_470) {
const uint l_471 = l_mut_468 * uint(256);
const uint l_472 = l_471 + l_467;
const float l_473 = shared_memory_316[l_472];
const float l_474 = shared_memory_314[l_mut_468];
const float l_475 = l_474 - l_mut_202;
const float l_476 = exp(l_475);
const float l_477 = l_473 * l_476;
l_mut_85 = l_mut_85 + l_477;
}
}
const uint l_478 = l_25 + l_467;
const float l_479 = l_mut_85 * l_mut_69;
const bfloat l_480 = bfloat(l_479);
buffer_5[l_478] = l_480;
const uint l_481 = l_1 * uint(8);
const uint l_482 = l_481 + uint(7);
l_mut_85 = float(0.0);

for (uint l_mut_483 = uint(0); l_mut_483 < uint(4); ++l_mut_483) {
const float l_484 = shared_memory_315[l_mut_483];
const bool l_485 = l_484 > float(0.0);
if (l_485) {
const uint l_486 = l_mut_483 * uint(256);
const uint l_487 = l_486 + l_482;
const float l_488 = shared_memory_316[l_487];
const float l_489 = shared_memory_314[l_mut_483];
const float l_490 = l_489 - l_mut_202;
const float l_491 = exp(l_490);
const float l_492 = l_488 * l_491;
l_mut_85 = l_mut_85 + l_492;
}
}
const uint l_493 = l_25 + l_482;
const float l_494 = l_mut_85 * l_mut_69;
const bfloat l_495 = bfloat(l_494);
buffer_5[l_493] = l_495;
}

}
