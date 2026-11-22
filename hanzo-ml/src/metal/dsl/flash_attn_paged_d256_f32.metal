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
threadgroup uchar dynamic_shared_mem[30048];
// Shared array size: 2048, 4096 bytes
threadgroup half* shared_memory_34 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[0]);
// Shared array size: 128, 512 bytes
threadgroup float* shared_memory_35 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[4096]);
// Shared array size: 2048, 8192 bytes
threadgroup float* shared_memory_36 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[4608]);
// Shared array size: 8, 32 bytes
threadgroup float* shared_memory_37 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[12800]);
// Shared array size: 8, 32 bytes
threadgroup float* shared_memory_38 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[12832]);
// Shared array size: 8, 32 bytes
threadgroup float* shared_memory_39 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[12864]);
// Shared array size: 4096, 8192 bytes
threadgroup half* shared_memory_79 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[12896]);
// Shared array size: 128, 256 bytes
threadgroup half* shared_memory_205 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[21088]);
// Shared array size: 4096, 8192 bytes
threadgroup half* shared_memory_206 = reinterpret_cast<threadgroup half*>(&dynamic_shared_mem[21344]);
// Shared array size: 128, 512 bytes
threadgroup float* shared_memory_207 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[29536]);
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
const uint l_14 = l_4 + uint(8);
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
const uint l_33 = l_32 * uint(256);

for (uint l_mut_40 = uint(0); l_mut_40 < uint(64); ++l_mut_40) {
const uint l_41 = thread_index_in_threadgroup * uint(64);
const uint l_42 = l_41 + l_mut_40;
shared_memory_36[l_42] = float(0.0);
}
const bool l_43 = thread_index_in_threadgroup < uint(8);
if (l_43) {
shared_memory_37[thread_index_in_threadgroup] = float(-3.3999999521443642e38);
shared_memory_38[thread_index_in_threadgroup] = float(0.0);
}

for (uint l_mut_44 = uint(0); l_mut_44 < uint(64); ++l_mut_44) {
const uint l_45 = thread_index_in_threadgroup * uint(64);
const uint l_46 = l_45 + l_mut_44;
const uint l_47 = l_46 / uint(256);
const uint l_48 = l_46 % uint(256);
const uint l_49 = l_22 * uint(8);
const uint l_50 = l_49 + l_47;
l_mut_51 = float(0.0);
const bool l_52 = l_50 < l_4;
if (l_52) {
const uint l_53 = l_50 * uint(256);
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
const uint l_63 = l_22 * uint(8);
const uint l_64 = l_mut_26 + l_63;
const uint l_65 = l_64 + uint(8);
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
const uint l_72 = l_22 * uint(8);
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

for (uint l_mut_80 = uint(0); l_mut_80 < uint(128); ++l_mut_80) {
const uint l_81 = thread_index_in_threadgroup * uint(128);
const uint l_82 = l_81 + l_mut_80;
const uint l_83 = l_82 / uint(256);
const uint l_84 = l_82 % uint(256);
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
const uint l_93 = l_92 * uint(256);
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
const uint l_107 = l_106 * uint(256);
const uint l_108 = l_107 + l_84;
const float l_109 = buffer_1[l_108];
l_mut_110 = l_109;
} else {
const uint l_111 = uint(256) / l_11;
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
simdgroup_float8x8 frag_acc_125;
frag_acc_125 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_126 = uint(0); l_mut_126 < uint(32); ++l_mut_126) {
simdgroup_half8x8 frag_a_135;
simdgroup_half8x8 frag_b_144;
const uint l_127 = uint(0) * uint(8);
const uint l_128 = l_127 * uint(256);
const uint l_129 = l_mut_126 * uint(8);
const uint l_130 = l_128 + l_129;
const uint l_131 = 2048;
const uint l_132 = l_131 - uint(0);
const uint l_133 = uint(2048) - l_130;
const uint l_134 = l_130 + uint(0);
simdgroup_load(frag_a_135, shared_memory_34 + l_134, uint(256), 0, false);
const uint l_136 = l_mut_124 * uint(8);
const uint l_137 = l_136 * uint(256);
const uint l_138 = l_mut_126 * uint(8);
const uint l_139 = l_137 + l_138;
const uint l_140 = 4096;
const uint l_141 = l_140 - uint(0);
const uint l_142 = uint(4096) - l_139;
const uint l_143 = l_139 + uint(0);
simdgroup_load(frag_b_144, shared_memory_79 + l_143, uint(256), 0, true);
simdgroup_multiply_accumulate(frag_acc_125, frag_a_135, frag_b_144, frag_acc_125);
}
const uint l_145 = uint(0) * uint(8);
const uint l_146 = l_145 * uint(16);
const uint l_147 = l_mut_124 * uint(8);
const uint l_148 = l_146 + l_147;
const uint l_149 = 128;
const uint l_150 = l_149 - uint(0);
const uint l_151 = uint(128) - l_148;
const uint l_152 = l_148 + uint(0);
simdgroup_store(frag_acc_125, shared_memory_35 + l_152, uint(16));
simdgroup_barrier(mem_flags::mem_none);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_153 = uint(0); l_mut_153 < uint(4); ++l_mut_153) {
const uint l_154 = thread_index_in_threadgroup * uint(4);
const uint l_155 = l_154 + l_mut_153;
const uint l_156 = l_155 / uint(16);
const uint l_157 = l_155 % uint(16);
const uint l_158 = l_22 * uint(8);
const uint l_159 = l_158 + l_156;
const uint l_160 = l_mut_78 * uint(16);
const uint l_161 = l_160 + l_157;
const uint l_162 = l_mut_26 + l_159;
const bool l_163 = l_7 == uint(0);
const bool l_164 = l_161 <= l_162;
const bool l_165 = l_163 || l_164;
const bool l_166 = l_12 == uint(0);
const uint l_167 = l_161 + l_12;
const bool l_168 = l_167 > l_162;
const bool l_169 = l_166 || l_168;
const bool l_170 = l_159 < l_4;
const bool l_171 = l_161 < l_25;
const bool l_172 = l_170 && l_171;
const bool l_173 = l_172 && l_165;
const bool l_174 = l_173 && l_169;
if (l_174) {
const float l_175 = shared_memory_35[l_155];
const float l_176 = l_175 * l_2;
l_mut_110 = l_176;
const bool l_177 = l_3 > float(0.0);
if (l_177) {
const float l_178 = l_mut_110 / l_3;
const float l_179 = safe_tanh_scalar(l_178);
const float l_180 = l_3 * l_179;
l_mut_110 = l_180;
}
shared_memory_35[l_155] = l_mut_110;
} else {
shared_memory_35[l_155] = float(-3.3999999521443642e38);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_181 = thread_index_in_threadgroup < uint(8);
if (l_181) {
const uint l_182 = l_22 * uint(8);
const uint l_183 = l_182 + thread_index_in_threadgroup;
const bool l_184 = l_183 < l_4;
if (l_184) {
l_mut_110 = float(-3.3999999521443642e38);

for (uint l_mut_185 = uint(0); l_mut_185 < uint(16); ++l_mut_185) {
const uint l_186 = thread_index_in_threadgroup * uint(16);
const uint l_187 = l_186 + l_mut_185;
const float l_188 = shared_memory_35[l_187];
const bool l_189 = l_188 > l_mut_110;
if (l_189) {
l_mut_110 = l_188;
}
}
const float l_190 = shared_memory_37[thread_index_in_threadgroup];
l_mut_96 = l_190;
const bool l_191 = l_mut_110 > l_mut_96;
if (l_191) {
l_mut_96 = l_mut_110;
}
const float l_192 = l_190 - l_mut_96;
const float l_193 = exp(l_192);
l_mut_110 = float(0.0);

for (uint l_mut_194 = uint(0); l_mut_194 < uint(16); ++l_mut_194) {
const uint l_195 = thread_index_in_threadgroup * uint(16);
const uint l_196 = l_195 + l_mut_194;
const float l_197 = shared_memory_35[l_196];
const float l_198 = l_197 - l_mut_96;
const float l_199 = exp(l_198);
const uint l_200 = thread_index_in_threadgroup * uint(16);
const uint l_201 = l_200 + l_mut_194;
shared_memory_35[l_201] = l_199;
l_mut_110 = l_mut_110 + l_199;
}
const float l_202 = shared_memory_38[thread_index_in_threadgroup];
const float l_203 = l_202 * l_193;
const float l_204 = l_203 + l_mut_110;
shared_memory_38[thread_index_in_threadgroup] = l_204;
shared_memory_37[thread_index_in_threadgroup] = l_mut_96;
shared_memory_39[thread_index_in_threadgroup] = l_193;
} else {
shared_memory_39[thread_index_in_threadgroup] = float(1.0);
}
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_208 = uint(0); l_mut_208 < uint(4); ++l_mut_208) {
const uint l_209 = thread_index_in_threadgroup * uint(4);
const uint l_210 = l_209 + l_mut_208;
const uint l_211 = l_210 / uint(16);
const uint l_212 = l_22 * uint(8);
const uint l_213 = l_212 + l_211;
l_mut_110 = float(0.0);
const bool l_214 = l_213 < l_4;
if (l_214) {
const float l_215 = shared_memory_35[l_210];
l_mut_110 = l_215;
}
const half l_216 = half(l_mut_110);
shared_memory_205[l_210] = l_216;
}

for (uint l_mut_217 = uint(0); l_mut_217 < uint(128); ++l_mut_217) {
const uint l_218 = thread_index_in_threadgroup * uint(128);
const uint l_219 = l_218 + l_mut_217;
const uint l_220 = l_219 / uint(256);
const uint l_221 = l_219 % uint(256);
const uint l_222 = l_mut_78 * uint(16);
const uint l_223 = l_222 + l_220;
l_mut_110 = float(0.0);
const bool l_224 = l_223 < l_25;
if (l_224) {
const bool l_225 = l_10 == uint(2);
if (l_225) {
const uint l_226 = l_18 * l_6;
const uint l_227 = l_226 + l_24;
const uint l_228 = l_227 * l_8;
const uint l_229 = l_228 + l_223;
const uint l_230 = l_229 * uint(256);
const uint l_231 = l_230 + l_221;
const float l_232 = buffer_2[l_231];
l_mut_96 = l_232;
} else {
const uint l_233 = l_18 * l_9;
const uint l_234 = l_223 / l_8;
const uint l_235 = l_233 + l_234;
const uint l_236 = buffer_3[l_235];
const uint l_237 = l_223 % l_8;
const bool l_238 = l_10 == uint(0);
if (l_238) {
const uint l_239 = l_236 * l_6;
const uint l_240 = l_239 + l_24;
const uint l_241 = l_240 * l_8;
const uint l_242 = l_241 + l_237;
const uint l_243 = l_242 * uint(256);
const uint l_244 = l_243 + l_221;
const float l_245 = buffer_2[l_244];
l_mut_51 = l_245;
} else {
const uint l_246 = l_236 * l_6;
const uint l_247 = l_246 + l_24;
const uint l_248 = l_247 * uint(256);
const uint l_249 = l_248 + l_221;
const uint l_250 = l_249 * l_8;
const uint l_251 = l_250 + l_237;
const float l_252 = buffer_2[l_251];
l_mut_51 = l_252;
}
l_mut_96 = l_mut_51;
}
l_mut_110 = l_mut_96;
}
const half l_253 = half(l_mut_110);
shared_memory_206[l_219] = l_253;
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_254 = uint(0); l_mut_254 < uint(16); ++l_mut_254) {

for (uint l_mut_255 = uint(0); l_mut_255 < uint(2); ++l_mut_255) {
simdgroup_float8x8 frag_acc_256;
frag_acc_256 = make_filled_simdgroup_matrix<float, 8, 8>(float(0.0));

for (uint l_mut_257 = uint(0); l_mut_257 < uint(2); ++l_mut_257) {
simdgroup_half8x8 frag_a_266;
simdgroup_half8x8 frag_b_277;
const uint l_258 = uint(0) * uint(8);
const uint l_259 = l_258 * uint(16);
const uint l_260 = l_mut_257 * uint(8);
const uint l_261 = l_259 + l_260;
const uint l_262 = 128;
const uint l_263 = l_262 - uint(0);
const uint l_264 = uint(128) - l_261;
const uint l_265 = l_261 + uint(0);
simdgroup_load(frag_a_266, shared_memory_205 + l_265, uint(16), 0, false);
const uint l_267 = l_mut_257 * uint(8);
const uint l_268 = l_267 * uint(256);
const uint l_269 = l_mut_254 * uint(16);
const uint l_270 = l_268 + l_269;
const uint l_271 = l_mut_255 * uint(8);
const uint l_272 = l_270 + l_271;
const uint l_273 = 4096;
const uint l_274 = l_273 - uint(0);
const uint l_275 = uint(4096) - l_272;
const uint l_276 = l_272 + uint(0);
simdgroup_load(frag_b_277, shared_memory_206 + l_276, uint(256), 0, false);
simdgroup_multiply_accumulate(frag_acc_256, frag_a_266, frag_b_277, frag_acc_256);
}
const uint l_278 = uint(0) * uint(8);
const uint l_279 = l_278 * uint(16);
const uint l_280 = l_mut_255 * uint(8);
const uint l_281 = l_279 + l_280;
const uint l_282 = 128;
const uint l_283 = l_282 - uint(0);
const uint l_284 = uint(128) - l_281;
const uint l_285 = l_281 + uint(0);
simdgroup_store(frag_acc_256, shared_memory_207 + l_285, uint(16));
simdgroup_barrier(mem_flags::mem_none);
}
threadgroup_barrier(mem_flags::mem_threadgroup);

for (uint l_mut_286 = uint(0); l_mut_286 < uint(4); ++l_mut_286) {
const uint l_287 = thread_index_in_threadgroup * uint(4);
const uint l_288 = l_287 + l_mut_286;
const uint l_289 = l_288 / uint(16);
const uint l_290 = l_288 % uint(16);
const uint l_291 = l_289 * uint(256);
const uint l_292 = l_mut_254 * uint(16);
const uint l_293 = l_291 + l_292;
const uint l_294 = l_293 + l_290;
const float l_295 = shared_memory_36[l_294];
const float l_296 = shared_memory_39[l_289];
const float l_297 = l_295 * l_296;
const float l_298 = shared_memory_207[l_288];
const float l_299 = l_297 + l_298;
shared_memory_36[l_294] = l_299;
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}

for (uint l_mut_300 = uint(0); l_mut_300 < uint(64); ++l_mut_300) {
const uint l_301 = thread_index_in_threadgroup * uint(64);
const uint l_302 = l_301 + l_mut_300;
const uint l_303 = l_302 / uint(256);
const uint l_304 = l_302 % uint(256);
const uint l_305 = l_22 * uint(8);
const uint l_306 = l_305 + l_303;
const bool l_307 = l_306 < l_4;
if (l_307) {
l_mut_110 = float(1.0);
const float l_308 = shared_memory_38[l_303];
const bool l_309 = l_308 > float(0.0);
if (l_309) {
const float l_310 = shared_memory_38[l_303];
l_mut_110 = l_310;
}
const uint l_311 = l_306 * uint(256);
const uint l_312 = l_33 + l_311;
const uint l_313 = l_312 + l_304;
const float l_314 = shared_memory_36[l_302];
const float l_315 = l_314 / l_mut_110;
buffer_5[l_313] = l_315;
}
}

}
