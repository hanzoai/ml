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
void matvec_ptq1_0_gguf_f_f32(
    const device uint * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device float * buffer_2 [[buffer(2)]],
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
const uint l_3 = threadgroup_index_in_grid * uint(4);
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
const float l_48 = buffer_1[l_47];
const uint l_49 = l_mut_43 + uint(16);
const uint l_50 = l_49 + l_30;
const float l_51 = buffer_1[l_50];
const uint l_52 = l_mut_43 + uint(32);
const uint l_53 = l_52 + l_30;
const float l_54 = buffer_1[l_53];
const uint l_55 = l_mut_43 + uint(48);
const uint l_56 = l_55 + l_30;
const float l_57 = buffer_1[l_56];
const uint l_58 = l_mut_43 + uint(64);
const uint l_59 = l_58 + l_30;
const float l_60 = buffer_1[l_59];
const uint l_61 = l_mut_43 + l_30;
const uint l_62 = l_61 + uint(1);
const float l_63 = buffer_1[l_62];
const uint l_64 = l_mut_43 + uint(16);
const uint l_65 = l_64 + l_30;
const uint l_66 = l_65 + uint(1);
const float l_67 = buffer_1[l_66];
const uint l_68 = l_mut_43 + uint(32);
const uint l_69 = l_68 + l_30;
const uint l_70 = l_69 + uint(1);
const float l_71 = buffer_1[l_70];
const uint l_72 = l_mut_43 + uint(48);
const uint l_73 = l_72 + l_30;
const uint l_74 = l_73 + uint(1);
const float l_75 = buffer_1[l_74];
const uint l_76 = l_mut_43 + uint(64);
const uint l_77 = l_76 + l_30;
const uint l_78 = l_77 + uint(1);
const float l_79 = buffer_1[l_78];
const uint l_80 = l_mut_43 + uint(80);
const uint l_81 = l_80 + l_7;
const float l_82 = buffer_1[l_81];
const uint l_83 = l_mut_43 + uint(88);
const uint l_84 = l_83 + l_7;
const float l_85 = buffer_1[l_84];
const uint l_86 = l_mut_43 + uint(96);
const uint l_87 = l_86 + l_7;
const float l_88 = buffer_1[l_87];
const uint l_89 = l_mut_43 + uint(104);
const uint l_90 = l_89 + l_7;
const float l_91 = buffer_1[l_90];
const uint l_92 = l_mut_43 + uint(112);
const uint l_93 = l_92 + l_7;
const float l_94 = buffer_1[l_93];
const uint l_95 = l_mut_43 + uint(120);
const uint l_96 = l_95 + l_7;
const float l_97 = buffer_1[l_96];
const float l_98 = l_48 + l_51;
const float l_99 = l_98 + l_54;
const float l_100 = l_99 + l_57;
const float l_101 = l_100 + l_60;
const float l_102 = l_63 + l_67;
const float l_103 = l_102 + l_71;
const float l_104 = l_103 + l_75;
const float l_105 = l_104 + l_79;
const float l_106 = l_101 + l_105;
const float l_107 = l_82 + l_85;
const float l_108 = l_107 + l_88;
const float l_109 = l_108 + l_91;
const float l_110 = l_109 + l_94;
const float l_111 = l_106 + l_110;
const float l_112 = l_111 + l_97;
const float l_113 = float(3.0) * l_51;
const float l_114 = l_48 - l_113;
const float l_115 = float(3.0) * l_54;
const float l_116 = l_51 - l_115;
const float l_117 = float(3.0) * l_57;
const float l_118 = l_54 - l_117;
const float l_119 = float(3.0) * l_60;
const float l_120 = l_57 - l_119;
const float l_121 = float(3.0) * l_67;
const float l_122 = l_63 - l_121;
const float l_123 = float(3.0) * l_71;
const float l_124 = l_67 - l_123;
const float l_125 = float(3.0) * l_75;
const float l_126 = l_71 - l_125;
const float l_127 = float(3.0) * l_79;
const float l_128 = l_75 - l_127;
const float l_129 = float(3.0) * l_85;
const float l_130 = l_82 - l_129;
const float l_131 = float(3.0) * l_88;
const float l_132 = l_85 - l_131;
const float l_133 = float(3.0) * l_91;
const float l_134 = l_88 - l_133;
const float l_135 = float(3.0) * l_94;
const float l_136 = l_91 - l_135;
const uint l_137 = l_arr_24[uint(0)];
const uint l_138 = l_mut_44 + l_137;
const uint l_139 = l_138 + l_31;
const uint l_140 = buffer_0[l_139];
const uint l_141 = l_140 >> l_33;
const uint l_142 = l_141 & uint(255);
const uint l_143 = l_140 >> l_34;
const uint l_144 = l_143 & uint(255);
const uint l_145 = l_138 + l_36;
const uint l_146 = buffer_0[l_145];
const uint l_147 = l_146 >> l_38;
const uint l_148 = l_147 & uint(255);
const uint l_149 = l_138 + uint(6);
const uint l_150 = buffer_0[l_149];
const uint l_151 = l_150 >> l_40;
const uint l_152 = l_151 & uint(255);
const uint l_153 = l_150 >> uint(16);
const half_2 l_154 = reinterpret_cast<thread half_2 const&>(l_153);
const half l_155 = l_154.i_0;
const float l_156 = float(l_155);
const float l_157 = float(l_152);
const float l_158 = l_157 * float(0.00390625);
const float l_159 = l_13 * l_158;
const float l_160 = floor(l_159);
const float l_161 = l_mut_9 * l_158;
const float l_162 = floor(l_161);
const float l_163 = float(3.0) * l_162;
const float l_164 = l_160 - l_163;
const float l_165 = float(l_142);
const float l_166 = l_165 * float(0.00390625);
const float l_167 = float(3.0) * l_166;
const float l_168 = floor(l_167);
const float l_169 = l_168 * l_114;
const float l_170 = float(9.0) * l_166;
const float l_171 = floor(l_170);
const float l_172 = l_171 * l_116;
const float l_173 = l_169 + l_172;
const float l_174 = float(27.0) * l_166;
const float l_175 = floor(l_174);
const float l_176 = l_175 * l_118;
const float l_177 = l_173 + l_176;
const float l_178 = float(81.0) * l_166;
const float l_179 = floor(l_178);
const float l_180 = l_179 * l_120;
const float l_181 = l_177 + l_180;
const float l_182 = float(243.0) * l_166;
const float l_183 = floor(l_182);
const float l_184 = l_183 * l_60;
const float l_185 = l_181 + l_184;
const float l_186 = float(l_144);
const float l_187 = l_186 * float(0.00390625);
const float l_188 = float(3.0) * l_187;
const float l_189 = floor(l_188);
const float l_190 = l_189 * l_122;
const float l_191 = float(9.0) * l_187;
const float l_192 = floor(l_191);
const float l_193 = l_192 * l_124;
const float l_194 = l_190 + l_193;
const float l_195 = float(27.0) * l_187;
const float l_196 = floor(l_195);
const float l_197 = l_196 * l_126;
const float l_198 = l_194 + l_197;
const float l_199 = float(81.0) * l_187;
const float l_200 = floor(l_199);
const float l_201 = l_200 * l_128;
const float l_202 = l_198 + l_201;
const float l_203 = float(243.0) * l_187;
const float l_204 = floor(l_203);
const float l_205 = l_204 * l_79;
const float l_206 = l_202 + l_205;
const float l_207 = l_185 + l_206;
const float l_208 = float(l_148);
const float l_209 = l_208 * float(0.00390625);
const float l_210 = float(3.0) * l_209;
const float l_211 = floor(l_210);
const float l_212 = l_211 * l_130;
const float l_213 = float(9.0) * l_209;
const float l_214 = floor(l_213);
const float l_215 = l_214 * l_132;
const float l_216 = l_212 + l_215;
const float l_217 = float(27.0) * l_209;
const float l_218 = floor(l_217);
const float l_219 = l_218 * l_134;
const float l_220 = l_216 + l_219;
const float l_221 = float(81.0) * l_209;
const float l_222 = floor(l_221);
const float l_223 = l_222 * l_136;
const float l_224 = l_220 + l_223;
const float l_225 = float(243.0) * l_209;
const float l_226 = floor(l_225);
const float l_227 = l_226 * l_94;
const float l_228 = l_224 + l_227;
const float l_229 = l_207 + l_228;
const float l_230 = l_164 * l_97;
const float l_231 = l_229 + l_230;
const float l_232 = l_231 - l_112;
const float l_233 = l_156 * l_232;
const float l_234 = l_arr_14[uint(0)];
const float l_235 = l_234 + l_233;
l_arr_14[uint(0)] = l_235;
const uint l_236 = l_arr_24[uint(1)];
const uint l_237 = l_mut_44 + l_236;
const uint l_238 = l_237 + l_31;
const uint l_239 = buffer_0[l_238];
const uint l_240 = l_239 >> l_33;
const uint l_241 = l_240 & uint(255);
const uint l_242 = l_239 >> l_34;
const uint l_243 = l_242 & uint(255);
const uint l_244 = l_237 + l_36;
const uint l_245 = buffer_0[l_244];
const uint l_246 = l_245 >> l_38;
const uint l_247 = l_246 & uint(255);
const uint l_248 = l_237 + uint(6);
const uint l_249 = buffer_0[l_248];
const uint l_250 = l_249 >> l_40;
const uint l_251 = l_250 & uint(255);
const uint l_252 = l_249 >> uint(16);
const half_2 l_253 = reinterpret_cast<thread half_2 const&>(l_252);
const half l_254 = l_253.i_0;
const float l_255 = float(l_254);
const float l_256 = float(l_251);
const float l_257 = l_256 * float(0.00390625);
const float l_258 = l_13 * l_257;
const float l_259 = floor(l_258);
const float l_260 = l_mut_9 * l_257;
const float l_261 = floor(l_260);
const float l_262 = float(3.0) * l_261;
const float l_263 = l_259 - l_262;
const float l_264 = float(l_241);
const float l_265 = l_264 * float(0.00390625);
const float l_266 = float(3.0) * l_265;
const float l_267 = floor(l_266);
const float l_268 = l_267 * l_114;
const float l_269 = float(9.0) * l_265;
const float l_270 = floor(l_269);
const float l_271 = l_270 * l_116;
const float l_272 = l_268 + l_271;
const float l_273 = float(27.0) * l_265;
const float l_274 = floor(l_273);
const float l_275 = l_274 * l_118;
const float l_276 = l_272 + l_275;
const float l_277 = float(81.0) * l_265;
const float l_278 = floor(l_277);
const float l_279 = l_278 * l_120;
const float l_280 = l_276 + l_279;
const float l_281 = float(243.0) * l_265;
const float l_282 = floor(l_281);
const float l_283 = l_282 * l_60;
const float l_284 = l_280 + l_283;
const float l_285 = float(l_243);
const float l_286 = l_285 * float(0.00390625);
const float l_287 = float(3.0) * l_286;
const float l_288 = floor(l_287);
const float l_289 = l_288 * l_122;
const float l_290 = float(9.0) * l_286;
const float l_291 = floor(l_290);
const float l_292 = l_291 * l_124;
const float l_293 = l_289 + l_292;
const float l_294 = float(27.0) * l_286;
const float l_295 = floor(l_294);
const float l_296 = l_295 * l_126;
const float l_297 = l_293 + l_296;
const float l_298 = float(81.0) * l_286;
const float l_299 = floor(l_298);
const float l_300 = l_299 * l_128;
const float l_301 = l_297 + l_300;
const float l_302 = float(243.0) * l_286;
const float l_303 = floor(l_302);
const float l_304 = l_303 * l_79;
const float l_305 = l_301 + l_304;
const float l_306 = l_284 + l_305;
const float l_307 = float(l_247);
const float l_308 = l_307 * float(0.00390625);
const float l_309 = float(3.0) * l_308;
const float l_310 = floor(l_309);
const float l_311 = l_310 * l_130;
const float l_312 = float(9.0) * l_308;
const float l_313 = floor(l_312);
const float l_314 = l_313 * l_132;
const float l_315 = l_311 + l_314;
const float l_316 = float(27.0) * l_308;
const float l_317 = floor(l_316);
const float l_318 = l_317 * l_134;
const float l_319 = l_315 + l_318;
const float l_320 = float(81.0) * l_308;
const float l_321 = floor(l_320);
const float l_322 = l_321 * l_136;
const float l_323 = l_319 + l_322;
const float l_324 = float(243.0) * l_308;
const float l_325 = floor(l_324);
const float l_326 = l_325 * l_94;
const float l_327 = l_323 + l_326;
const float l_328 = l_306 + l_327;
const float l_329 = l_263 * l_97;
const float l_330 = l_328 + l_329;
const float l_331 = l_330 - l_112;
const float l_332 = l_255 * l_331;
const float l_333 = l_arr_14[uint(1)];
const float l_334 = l_333 + l_332;
l_arr_14[uint(1)] = l_334;
l_mut_26 = l_mut_26 + uint(4);
l_mut_43 = l_mut_43 + uint(512);
l_mut_44 = l_mut_44 + uint(28);
}
}
const float l_335 = l_arr_14[uint(0)];
const float l_336 = float {simd_sum(l_335)
};
const bool l_337 = thread_pos_in_threadgroup.x == uint(0);
const uint l_338 = l_5 + uint(0);
const bool l_339 = l_338 < l_1;
const bool l_340 = l_337 && l_339;
if (l_340) {
const uint l_341 = l_5 + uint(0);
buffer_2[l_341] = l_336;
}
const float l_342 = l_arr_14[uint(1)];
const float l_343 = float {simd_sum(l_342)
};
const bool l_344 = thread_pos_in_threadgroup.x == uint(0);
const uint l_345 = l_5 + uint(1);
const bool l_346 = l_345 < l_1;
const bool l_347 = l_344 && l_346;
if (l_347) {
const uint l_348 = l_5 + uint(1);
buffer_2[l_348] = l_343;
}

}
