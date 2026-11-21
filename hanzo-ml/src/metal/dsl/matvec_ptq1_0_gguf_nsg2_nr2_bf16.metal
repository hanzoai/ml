#include <metal_stdlib>
using namespace metal;

struct alignas(4) half_2 {
    half i_0;
    half i_1;
};

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void matvec_ptq1_0_gguf_f_(
    const device uint * buffer_0 [[buffer(0)]],
    const device bfloat * buffer_1 [[buffer(1)]],
    device bfloat * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    constant info_st& info [[buffer(4)]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]]
) {
uint threadgroup_index_in_grid = (threadgroup_pos_in_grid.z * threadgroups_per_grid.y * threadgroups_per_grid.x) + (threadgroup_pos_in_grid.y * threadgroups_per_grid.x) + threadgroup_pos_in_grid.x;
uint threads_per_simdgroup_checked = min(simd_size, threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z);
float l_arr_14[2];

uint l_arr_24[2];

float l_mut_9;
uint l_mut_16;
uint l_mut_26;
uint l_mut_43;
uint l_mut_44;
const uint l_0 = buffer_3[uint(0)];
const uint l_1 = buffer_3[uint(1)];
const uint l_2 = l_0 / uint(128);
const uint l_3 = threadgroup_index_in_grid * uint(2);
const uint l_4 = l_3 + thread_pos_in_threadgroup.y;
const uint l_5 = l_4 * uint(2);
const uint l_6 = thread_pos_in_threadgroup.x / uint(8);
const uint l_7 = thread_pos_in_threadgroup.x % uint(8);
const uint l_8 = l_7 / uint(2);
l_mut_9 = float(1.0);
const bool l_10 = l_8 == uint(1);
if (l_10) {
l_mut_9 = float(3.0);
} else {
const bool l_11 = l_8 == uint(2);
if (l_11) {
l_mut_9 = float(9.0);
} else {
const bool l_12 = l_8 == uint(3);
if (l_12) {
l_mut_9 = float(27.0);
}
}
}
const float l_13 = float(3.0) * l_mut_9;
l_arr_14[uint(0)] = float(0.0);
l_arr_14[uint(1)] = float(0.0);
const uint l_15 = uint(7) * l_2;
l_mut_16 = uint(0);
const bool l_17 = l_5 < l_1;
if (l_17) {
const uint l_18 = l_1 - uint(1);
const uint l_19 = l_18 - l_5;
l_mut_16 = l_19;
}
const uint l_20 = l_mut_16 * l_15;
const uint l_21 = l_5 * l_2;
const uint l_22 = l_21 + l_6;
const uint l_23 = uint(7) * l_22;
const uint l_25 = uint(0) * l_15;
l_mut_26 = l_25;
const bool l_27 = uint(0) > l_mut_16;
if (l_27) {
l_mut_26 = l_20;
}
l_arr_24[uint(0)] = l_mut_26;
const uint l_28 = uint(1) * l_15;
l_mut_26 = l_28;
const bool l_29 = uint(1) > l_mut_16;
if (l_29) {
l_mut_26 = l_20;
}
l_arr_24[uint(1)] = l_mut_26;
const uint l_30 = uint(2) * l_7;
const uint l_31 = l_7 / uint(2);
const uint l_32 = l_7 % uint(2);
const uint l_33 = uint(16) * l_32;
const uint l_34 = l_33 + uint(8);
const uint l_35 = l_7 / uint(4);
const uint l_36 = uint(4) + l_35;
const uint l_37 = l_7 % uint(4);
const uint l_38 = uint(8) * l_37;
const uint l_39 = l_7 % uint(2);
const uint l_40 = uint(8) * l_39;
const bool l_41 = l_5 < l_1;
if (l_41) {
l_mut_26 = l_6;
const uint l_42 = l_6 * uint(128);
l_mut_43 = l_42;
l_mut_44 = l_23;
while (true) {
const bool l_45 = l_mut_26 < l_2;
const bool l_46 = !l_45;
if (l_46) {
break;}
const uint l_47 = l_mut_43 + l_30;
const bfloat l_48 = buffer_1[l_47];
const float l_49 = float(l_48);
const uint l_50 = l_mut_43 + uint(16);
const uint l_51 = l_50 + l_30;
const bfloat l_52 = buffer_1[l_51];
const float l_53 = float(l_52);
const uint l_54 = l_mut_43 + uint(32);
const uint l_55 = l_54 + l_30;
const bfloat l_56 = buffer_1[l_55];
const float l_57 = float(l_56);
const uint l_58 = l_mut_43 + uint(48);
const uint l_59 = l_58 + l_30;
const bfloat l_60 = buffer_1[l_59];
const float l_61 = float(l_60);
const uint l_62 = l_mut_43 + uint(64);
const uint l_63 = l_62 + l_30;
const bfloat l_64 = buffer_1[l_63];
const float l_65 = float(l_64);
const uint l_66 = l_mut_43 + l_30;
const uint l_67 = l_66 + uint(1);
const bfloat l_68 = buffer_1[l_67];
const float l_69 = float(l_68);
const uint l_70 = l_mut_43 + uint(16);
const uint l_71 = l_70 + l_30;
const uint l_72 = l_71 + uint(1);
const bfloat l_73 = buffer_1[l_72];
const float l_74 = float(l_73);
const uint l_75 = l_mut_43 + uint(32);
const uint l_76 = l_75 + l_30;
const uint l_77 = l_76 + uint(1);
const bfloat l_78 = buffer_1[l_77];
const float l_79 = float(l_78);
const uint l_80 = l_mut_43 + uint(48);
const uint l_81 = l_80 + l_30;
const uint l_82 = l_81 + uint(1);
const bfloat l_83 = buffer_1[l_82];
const float l_84 = float(l_83);
const uint l_85 = l_mut_43 + uint(64);
const uint l_86 = l_85 + l_30;
const uint l_87 = l_86 + uint(1);
const bfloat l_88 = buffer_1[l_87];
const float l_89 = float(l_88);
const uint l_90 = l_mut_43 + uint(80);
const uint l_91 = l_90 + l_7;
const bfloat l_92 = buffer_1[l_91];
const float l_93 = float(l_92);
const uint l_94 = l_mut_43 + uint(88);
const uint l_95 = l_94 + l_7;
const bfloat l_96 = buffer_1[l_95];
const float l_97 = float(l_96);
const uint l_98 = l_mut_43 + uint(96);
const uint l_99 = l_98 + l_7;
const bfloat l_100 = buffer_1[l_99];
const float l_101 = float(l_100);
const uint l_102 = l_mut_43 + uint(104);
const uint l_103 = l_102 + l_7;
const bfloat l_104 = buffer_1[l_103];
const float l_105 = float(l_104);
const uint l_106 = l_mut_43 + uint(112);
const uint l_107 = l_106 + l_7;
const bfloat l_108 = buffer_1[l_107];
const float l_109 = float(l_108);
const uint l_110 = l_mut_43 + uint(120);
const uint l_111 = l_110 + l_7;
const bfloat l_112 = buffer_1[l_111];
const float l_113 = float(l_112);
const float l_114 = l_49 + l_53;
const float l_115 = l_114 + l_57;
const float l_116 = l_115 + l_61;
const float l_117 = l_116 + l_65;
const float l_118 = l_69 + l_74;
const float l_119 = l_118 + l_79;
const float l_120 = l_119 + l_84;
const float l_121 = l_120 + l_89;
const float l_122 = l_117 + l_121;
const float l_123 = l_93 + l_97;
const float l_124 = l_123 + l_101;
const float l_125 = l_124 + l_105;
const float l_126 = l_125 + l_109;
const float l_127 = l_122 + l_126;
const float l_128 = l_127 + l_113;
const float l_129 = float(3.0) * l_53;
const float l_130 = l_49 - l_129;
const float l_131 = float(3.0) * l_57;
const float l_132 = l_53 - l_131;
const float l_133 = float(3.0) * l_61;
const float l_134 = l_57 - l_133;
const float l_135 = float(3.0) * l_65;
const float l_136 = l_61 - l_135;
const float l_137 = float(3.0) * l_74;
const float l_138 = l_69 - l_137;
const float l_139 = float(3.0) * l_79;
const float l_140 = l_74 - l_139;
const float l_141 = float(3.0) * l_84;
const float l_142 = l_79 - l_141;
const float l_143 = float(3.0) * l_89;
const float l_144 = l_84 - l_143;
const float l_145 = float(3.0) * l_97;
const float l_146 = l_93 - l_145;
const float l_147 = float(3.0) * l_101;
const float l_148 = l_97 - l_147;
const float l_149 = float(3.0) * l_105;
const float l_150 = l_101 - l_149;
const float l_151 = float(3.0) * l_109;
const float l_152 = l_105 - l_151;
const uint l_153 = l_arr_24[uint(0)];
const uint l_154 = l_mut_44 + l_153;
const uint l_155 = l_154 + l_31;
const uint l_156 = buffer_0[l_155];
const uint l_157 = l_156 >> l_33;
const uint l_158 = l_157 & uint(255);
const uint l_159 = l_156 >> l_34;
const uint l_160 = l_159 & uint(255);
const uint l_161 = l_154 + l_36;
const uint l_162 = buffer_0[l_161];
const uint l_163 = l_162 >> l_38;
const uint l_164 = l_163 & uint(255);
const uint l_165 = l_154 + uint(6);
const uint l_166 = buffer_0[l_165];
const uint l_167 = l_166 >> l_40;
const uint l_168 = l_167 & uint(255);
const uint l_169 = l_166 >> uint(16);
const half_2 l_170 = reinterpret_cast<thread half_2 const&>(l_169);
const half l_171 = l_170.i_0;
const float l_172 = float(l_171);
const float l_173 = float(l_168);
const float l_174 = l_173 * float(0.00390625);
const float l_175 = l_13 * l_174;
const float l_176 = floor(l_175);
const float l_177 = l_mut_9 * l_174;
const float l_178 = floor(l_177);
const float l_179 = float(3.0) * l_178;
const float l_180 = l_176 - l_179;
const float l_181 = float(l_158);
const float l_182 = l_181 * float(0.00390625);
const float l_183 = float(3.0) * l_182;
const float l_184 = floor(l_183);
const float l_185 = l_184 * l_130;
const float l_186 = float(9.0) * l_182;
const float l_187 = floor(l_186);
const float l_188 = l_187 * l_132;
const float l_189 = l_185 + l_188;
const float l_190 = float(27.0) * l_182;
const float l_191 = floor(l_190);
const float l_192 = l_191 * l_134;
const float l_193 = l_189 + l_192;
const float l_194 = float(81.0) * l_182;
const float l_195 = floor(l_194);
const float l_196 = l_195 * l_136;
const float l_197 = l_193 + l_196;
const float l_198 = float(243.0) * l_182;
const float l_199 = floor(l_198);
const float l_200 = l_199 * l_65;
const float l_201 = l_197 + l_200;
const float l_202 = float(l_160);
const float l_203 = l_202 * float(0.00390625);
const float l_204 = float(3.0) * l_203;
const float l_205 = floor(l_204);
const float l_206 = l_205 * l_138;
const float l_207 = float(9.0) * l_203;
const float l_208 = floor(l_207);
const float l_209 = l_208 * l_140;
const float l_210 = l_206 + l_209;
const float l_211 = float(27.0) * l_203;
const float l_212 = floor(l_211);
const float l_213 = l_212 * l_142;
const float l_214 = l_210 + l_213;
const float l_215 = float(81.0) * l_203;
const float l_216 = floor(l_215);
const float l_217 = l_216 * l_144;
const float l_218 = l_214 + l_217;
const float l_219 = float(243.0) * l_203;
const float l_220 = floor(l_219);
const float l_221 = l_220 * l_89;
const float l_222 = l_218 + l_221;
const float l_223 = l_201 + l_222;
const float l_224 = float(l_164);
const float l_225 = l_224 * float(0.00390625);
const float l_226 = float(3.0) * l_225;
const float l_227 = floor(l_226);
const float l_228 = l_227 * l_146;
const float l_229 = float(9.0) * l_225;
const float l_230 = floor(l_229);
const float l_231 = l_230 * l_148;
const float l_232 = l_228 + l_231;
const float l_233 = float(27.0) * l_225;
const float l_234 = floor(l_233);
const float l_235 = l_234 * l_150;
const float l_236 = l_232 + l_235;
const float l_237 = float(81.0) * l_225;
const float l_238 = floor(l_237);
const float l_239 = l_238 * l_152;
const float l_240 = l_236 + l_239;
const float l_241 = float(243.0) * l_225;
const float l_242 = floor(l_241);
const float l_243 = l_242 * l_109;
const float l_244 = l_240 + l_243;
const float l_245 = l_223 + l_244;
const float l_246 = l_180 * l_113;
const float l_247 = l_245 + l_246;
const float l_248 = l_247 - l_128;
const float l_249 = l_172 * l_248;
const float l_250 = l_arr_14[uint(0)];
const float l_251 = l_250 + l_249;
l_arr_14[uint(0)] = l_251;
const uint l_252 = l_arr_24[uint(1)];
const uint l_253 = l_mut_44 + l_252;
const uint l_254 = l_253 + l_31;
const uint l_255 = buffer_0[l_254];
const uint l_256 = l_255 >> l_33;
const uint l_257 = l_256 & uint(255);
const uint l_258 = l_255 >> l_34;
const uint l_259 = l_258 & uint(255);
const uint l_260 = l_253 + l_36;
const uint l_261 = buffer_0[l_260];
const uint l_262 = l_261 >> l_38;
const uint l_263 = l_262 & uint(255);
const uint l_264 = l_253 + uint(6);
const uint l_265 = buffer_0[l_264];
const uint l_266 = l_265 >> l_40;
const uint l_267 = l_266 & uint(255);
const uint l_268 = l_265 >> uint(16);
const half_2 l_269 = reinterpret_cast<thread half_2 const&>(l_268);
const half l_270 = l_269.i_0;
const float l_271 = float(l_270);
const float l_272 = float(l_267);
const float l_273 = l_272 * float(0.00390625);
const float l_274 = l_13 * l_273;
const float l_275 = floor(l_274);
const float l_276 = l_mut_9 * l_273;
const float l_277 = floor(l_276);
const float l_278 = float(3.0) * l_277;
const float l_279 = l_275 - l_278;
const float l_280 = float(l_257);
const float l_281 = l_280 * float(0.00390625);
const float l_282 = float(3.0) * l_281;
const float l_283 = floor(l_282);
const float l_284 = l_283 * l_130;
const float l_285 = float(9.0) * l_281;
const float l_286 = floor(l_285);
const float l_287 = l_286 * l_132;
const float l_288 = l_284 + l_287;
const float l_289 = float(27.0) * l_281;
const float l_290 = floor(l_289);
const float l_291 = l_290 * l_134;
const float l_292 = l_288 + l_291;
const float l_293 = float(81.0) * l_281;
const float l_294 = floor(l_293);
const float l_295 = l_294 * l_136;
const float l_296 = l_292 + l_295;
const float l_297 = float(243.0) * l_281;
const float l_298 = floor(l_297);
const float l_299 = l_298 * l_65;
const float l_300 = l_296 + l_299;
const float l_301 = float(l_259);
const float l_302 = l_301 * float(0.00390625);
const float l_303 = float(3.0) * l_302;
const float l_304 = floor(l_303);
const float l_305 = l_304 * l_138;
const float l_306 = float(9.0) * l_302;
const float l_307 = floor(l_306);
const float l_308 = l_307 * l_140;
const float l_309 = l_305 + l_308;
const float l_310 = float(27.0) * l_302;
const float l_311 = floor(l_310);
const float l_312 = l_311 * l_142;
const float l_313 = l_309 + l_312;
const float l_314 = float(81.0) * l_302;
const float l_315 = floor(l_314);
const float l_316 = l_315 * l_144;
const float l_317 = l_313 + l_316;
const float l_318 = float(243.0) * l_302;
const float l_319 = floor(l_318);
const float l_320 = l_319 * l_89;
const float l_321 = l_317 + l_320;
const float l_322 = l_300 + l_321;
const float l_323 = float(l_263);
const float l_324 = l_323 * float(0.00390625);
const float l_325 = float(3.0) * l_324;
const float l_326 = floor(l_325);
const float l_327 = l_326 * l_146;
const float l_328 = float(9.0) * l_324;
const float l_329 = floor(l_328);
const float l_330 = l_329 * l_148;
const float l_331 = l_327 + l_330;
const float l_332 = float(27.0) * l_324;
const float l_333 = floor(l_332);
const float l_334 = l_333 * l_150;
const float l_335 = l_331 + l_334;
const float l_336 = float(81.0) * l_324;
const float l_337 = floor(l_336);
const float l_338 = l_337 * l_152;
const float l_339 = l_335 + l_338;
const float l_340 = float(243.0) * l_324;
const float l_341 = floor(l_340);
const float l_342 = l_341 * l_109;
const float l_343 = l_339 + l_342;
const float l_344 = l_322 + l_343;
const float l_345 = l_279 * l_113;
const float l_346 = l_344 + l_345;
const float l_347 = l_346 - l_128;
const float l_348 = l_271 * l_347;
const float l_349 = l_arr_14[uint(1)];
const float l_350 = l_349 + l_348;
l_arr_14[uint(1)] = l_350;
l_mut_26 = l_mut_26 + uint(4);
l_mut_43 = l_mut_43 + uint(512);
l_mut_44 = l_mut_44 + uint(28);
}
}
const float l_351 = l_arr_14[uint(0)];
const float l_352 = float {simd_sum(l_351)
};
const bool l_353 = thread_pos_in_threadgroup.x == uint(0);
const uint l_354 = l_5 + uint(0);
const bool l_355 = l_354 < l_1;
const bool l_356 = l_353 && l_355;
if (l_356) {
const uint l_357 = l_5 + uint(0);
const bfloat l_358 = bfloat(l_352);
buffer_2[l_357] = l_358;
}
const float l_359 = l_arr_14[uint(1)];
const float l_360 = float {simd_sum(l_359)
};
const bool l_361 = thread_pos_in_threadgroup.x == uint(0);
const uint l_362 = l_5 + uint(1);
const bool l_363 = l_362 < l_1;
const bool l_364 = l_361 && l_363;
if (l_364) {
const uint l_365 = l_5 + uint(1);
const bfloat l_366 = bfloat(l_360);
buffer_2[l_365] = l_366;
}

}
