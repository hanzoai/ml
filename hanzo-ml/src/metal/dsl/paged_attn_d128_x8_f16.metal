#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[16];
};

[[kernel]]
void paged_attn_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device half * buffer_1 [[buffer(1)]],
    const device half * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    const device uint * buffer_4 [[buffer(4)]],
    device half * buffer_5 [[buffer(5)]],
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
threadgroup uchar dynamic_shared_mem[2080];
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_190 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_191 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[16]);
// Shared array size: 512, 2048 bytes
threadgroup float* shared_memory_192 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[32]);
float l_arr_26[4];

float l_arr_27[4];

uint l_mut_14;
float l_mut_48;
float l_mut_49;
uint l_mut_51;
float l_mut_65;
float l_mut_126;
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
const uint l_25 = l_24 * uint(128);
const uint l_28 = l_1 * uint(4);
const uint l_29 = l_25 + l_28;
const uint l_30 = l_29 + uint(0);
const half l_31 = buffer_0[l_30];
const float l_32 = float(l_31);
l_arr_26[uint(0)] = l_32;
l_arr_27[uint(0)] = float(0.0);
const uint l_33 = l_1 * uint(4);
const uint l_34 = l_25 + l_33;
const uint l_35 = l_34 + uint(1);
const half l_36 = buffer_0[l_35];
const float l_37 = float(l_36);
l_arr_26[uint(1)] = l_37;
l_arr_27[uint(1)] = float(0.0);
const uint l_38 = l_1 * uint(4);
const uint l_39 = l_25 + l_38;
const uint l_40 = l_39 + uint(2);
const half l_41 = buffer_0[l_40];
const float l_42 = float(l_41);
l_arr_26[uint(2)] = l_42;
l_arr_27[uint(2)] = float(0.0);
const uint l_43 = l_1 * uint(4);
const uint l_44 = l_25 + l_43;
const uint l_45 = l_44 + uint(3);
const half l_46 = buffer_0[l_45];
const float l_47 = float(l_46);
l_arr_26[uint(3)] = l_47;
l_arr_27[uint(3)] = float(0.0);
l_mut_48 = float(-3.3999999521443642e38);
l_mut_49 = float(0.0);
const uint l_50 = l_19 + l_0;
l_mut_51 = l_50;
while (true) {
const bool l_52 = l_mut_51 < l_22;
const bool l_53 = !l_52;
if (l_53) {
break;}
const uint l_54 = l_9 * l_4;
const uint l_55 = l_54 + l_mut_51;
const uint l_56 = buffer_3[l_55];
const uint l_57 = l_56 * l_3;
const uint l_58 = l_57 + l_12;

for (uint l_mut_59 = uint(0); l_mut_59 < l_6; ++l_mut_59) {
const uint l_60 = l_mut_51 * l_6;
const uint l_61 = l_60 + l_mut_59;
const bool l_62 = l_61 >= l_mut_14;
const bool l_63 = l_61 < l_13;
const bool l_64 = l_62 && l_63;
if (l_64) {
l_mut_65 = float(0.0);
const uint l_66 = l_1 * uint(4);
const uint l_67 = l_66 + uint(0);
const uint l_68 = l_67 / uint(8);
const uint l_69 = l_67 % uint(8);
const uint l_70 = l_58 * uint(16);
const uint l_71 = l_70 + l_68;
const uint l_72 = l_71 * l_6;
const uint l_73 = l_72 + l_mut_59;
const uint l_74 = l_73 * uint(8);
const uint l_75 = l_74 + l_69;
const float l_76 = l_arr_26[uint(0)];
const half l_77 = buffer_1[l_75];
const float l_78 = float(l_77);
const float l_79 = l_76 * l_78;
l_mut_65 = l_mut_65 + l_79;
const uint l_80 = l_1 * uint(4);
const uint l_81 = l_80 + uint(1);
const uint l_82 = l_81 / uint(8);
const uint l_83 = l_81 % uint(8);
const uint l_84 = l_58 * uint(16);
const uint l_85 = l_84 + l_82;
const uint l_86 = l_85 * l_6;
const uint l_87 = l_86 + l_mut_59;
const uint l_88 = l_87 * uint(8);
const uint l_89 = l_88 + l_83;
const float l_90 = l_arr_26[uint(1)];
const half l_91 = buffer_1[l_89];
const float l_92 = float(l_91);
const float l_93 = l_90 * l_92;
l_mut_65 = l_mut_65 + l_93;
const uint l_94 = l_1 * uint(4);
const uint l_95 = l_94 + uint(2);
const uint l_96 = l_95 / uint(8);
const uint l_97 = l_95 % uint(8);
const uint l_98 = l_58 * uint(16);
const uint l_99 = l_98 + l_96;
const uint l_100 = l_99 * l_6;
const uint l_101 = l_100 + l_mut_59;
const uint l_102 = l_101 * uint(8);
const uint l_103 = l_102 + l_97;
const float l_104 = l_arr_26[uint(2)];
const half l_105 = buffer_1[l_103];
const float l_106 = float(l_105);
const float l_107 = l_104 * l_106;
l_mut_65 = l_mut_65 + l_107;
const uint l_108 = l_1 * uint(4);
const uint l_109 = l_108 + uint(3);
const uint l_110 = l_109 / uint(8);
const uint l_111 = l_109 % uint(8);
const uint l_112 = l_58 * uint(16);
const uint l_113 = l_112 + l_110;
const uint l_114 = l_113 * l_6;
const uint l_115 = l_114 + l_mut_59;
const uint l_116 = l_115 * uint(8);
const uint l_117 = l_116 + l_111;
const float l_118 = l_arr_26[uint(3)];
const half l_119 = buffer_1[l_117];
const float l_120 = float(l_119);
const float l_121 = l_118 * l_120;
l_mut_65 = l_mut_65 + l_121;
const float l_122 = float {simd_sum(l_mut_65)
};
const float l_123 = l_122 * l_7;
l_mut_65 = l_123;
const bool l_124 = l_8 > float(0.0);
if (l_124) {
const float l_125 = l_mut_65 / l_8;
l_mut_126 = l_125;
const bool l_127 = l_125 > float(10.0);
if (l_127) {
l_mut_126 = float(10.0);
} else {
const bool l_128 = l_125 < float(-10.0);
if (l_128) {
l_mut_126 = float(-10.0);
}
}
const float l_129 = float(2.0) * l_mut_126;
const float l_130 = exp(l_129);
const float l_131 = l_130 - float(1.0);
const float l_132 = l_130 + float(1.0);
const float l_133 = l_131 / l_132;
const float l_134 = l_8 * l_133;
l_mut_65 = l_134;
}
l_mut_126 = l_mut_48;
const bool l_135 = l_mut_65 > l_mut_48;
if (l_135) {
l_mut_126 = l_mut_65;
}
const float l_136 = l_mut_48 - l_mut_126;
const float l_137 = exp(l_136);
const float l_138 = l_mut_65 - l_mut_126;
const float l_139 = exp(l_138);
const float l_140 = l_mut_49 * l_137;
const float l_141 = l_140 + l_139;
l_mut_49 = l_141;
l_mut_48 = l_mut_126;
const uint l_142 = l_1 * uint(4);
const uint l_143 = l_142 + uint(0);
const uint l_144 = l_58 * uint(128);
const uint l_145 = l_144 + l_143;
const uint l_146 = l_145 * l_6;
const uint l_147 = l_146 + l_mut_59;
const float l_148 = l_arr_27[uint(0)];
const float l_149 = l_148 * l_137;
const half l_150 = buffer_2[l_147];
const float l_151 = float(l_150);
const float l_152 = l_139 * l_151;
const float l_153 = l_149 + l_152;
l_arr_27[uint(0)] = l_153;
const uint l_154 = l_1 * uint(4);
const uint l_155 = l_154 + uint(1);
const uint l_156 = l_58 * uint(128);
const uint l_157 = l_156 + l_155;
const uint l_158 = l_157 * l_6;
const uint l_159 = l_158 + l_mut_59;
const float l_160 = l_arr_27[uint(1)];
const float l_161 = l_160 * l_137;
const half l_162 = buffer_2[l_159];
const float l_163 = float(l_162);
const float l_164 = l_139 * l_163;
const float l_165 = l_161 + l_164;
l_arr_27[uint(1)] = l_165;
const uint l_166 = l_1 * uint(4);
const uint l_167 = l_166 + uint(2);
const uint l_168 = l_58 * uint(128);
const uint l_169 = l_168 + l_167;
const uint l_170 = l_169 * l_6;
const uint l_171 = l_170 + l_mut_59;
const float l_172 = l_arr_27[uint(2)];
const float l_173 = l_172 * l_137;
const half l_174 = buffer_2[l_171];
const float l_175 = float(l_174);
const float l_176 = l_139 * l_175;
const float l_177 = l_173 + l_176;
l_arr_27[uint(2)] = l_177;
const uint l_178 = l_1 * uint(4);
const uint l_179 = l_178 + uint(3);
const uint l_180 = l_58 * uint(128);
const uint l_181 = l_180 + l_179;
const uint l_182 = l_181 * l_6;
const uint l_183 = l_182 + l_mut_59;
const float l_184 = l_arr_27[uint(3)];
const float l_185 = l_184 * l_137;
const half l_186 = buffer_2[l_183];
const float l_187 = float(l_186);
const float l_188 = l_139 * l_187;
const float l_189 = l_185 + l_188;
l_arr_27[uint(3)] = l_189;
}
}
l_mut_51 = l_mut_51 + uint(4);
}
const bool l_193 = l_1 == uint(0);
if (l_193) {
shared_memory_190[l_0] = l_mut_48;
shared_memory_191[l_0] = l_mut_49;
}
const uint l_194 = l_0 * uint(128);
const uint l_195 = l_1 * uint(4);
const uint l_196 = l_194 + l_195;
const uint l_197 = l_196 + uint(0);
const float l_198 = l_arr_27[uint(0)];
shared_memory_192[l_197] = l_198;
const uint l_199 = l_0 * uint(128);
const uint l_200 = l_1 * uint(4);
const uint l_201 = l_199 + l_200;
const uint l_202 = l_201 + uint(1);
const float l_203 = l_arr_27[uint(1)];
shared_memory_192[l_202] = l_203;
const uint l_204 = l_0 * uint(128);
const uint l_205 = l_1 * uint(4);
const uint l_206 = l_204 + l_205;
const uint l_207 = l_206 + uint(2);
const float l_208 = l_arr_27[uint(2)];
shared_memory_192[l_207] = l_208;
const uint l_209 = l_0 * uint(128);
const uint l_210 = l_1 * uint(4);
const uint l_211 = l_209 + l_210;
const uint l_212 = l_211 + uint(3);
const float l_213 = l_arr_27[uint(3)];
shared_memory_192[l_212] = l_213;
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_214 = l_0 == uint(0);
if (l_214) {
l_mut_126 = float(-3.3999999521443642e38);

for (uint l_mut_215 = uint(0); l_mut_215 < uint(4); ++l_mut_215) {
const float l_216 = shared_memory_191[l_mut_215];
const bool l_217 = l_216 > float(0.0);
const float l_218 = shared_memory_190[l_mut_215];
const bool l_219 = l_218 > l_mut_126;
const bool l_220 = l_217 && l_219;
if (l_220) {
const float l_221 = shared_memory_190[l_mut_215];
l_mut_126 = l_221;
}
}
l_mut_65 = float(0.0);

for (uint l_mut_222 = uint(0); l_mut_222 < uint(4); ++l_mut_222) {
const float l_223 = shared_memory_191[l_mut_222];
const bool l_224 = l_223 > float(0.0);
if (l_224) {
const float l_225 = shared_memory_191[l_mut_222];
const float l_226 = shared_memory_190[l_mut_222];
const float l_227 = l_226 - l_mut_126;
const float l_228 = exp(l_227);
const float l_229 = l_225 * l_228;
l_mut_65 = l_mut_65 + l_229;
}
}
l_mut_49 = float(0.0);
const bool l_230 = l_mut_65 > float(0.0);
if (l_230) {
const float l_231 = float(1.0) / l_mut_65;
l_mut_49 = l_231;
}
const uint l_232 = l_1 * uint(4);
const uint l_233 = l_232 + uint(0);
l_mut_65 = float(0.0);

for (uint l_mut_234 = uint(0); l_mut_234 < uint(4); ++l_mut_234) {
const float l_235 = shared_memory_191[l_mut_234];
const bool l_236 = l_235 > float(0.0);
if (l_236) {
const uint l_237 = l_mut_234 * uint(128);
const uint l_238 = l_237 + l_233;
const float l_239 = shared_memory_192[l_238];
const float l_240 = shared_memory_190[l_mut_234];
const float l_241 = l_240 - l_mut_126;
const float l_242 = exp(l_241);
const float l_243 = l_239 * l_242;
l_mut_65 = l_mut_65 + l_243;
}
}
const uint l_244 = l_25 + l_233;
const float l_245 = l_mut_65 * l_mut_49;
const half l_246 = half(l_245);
buffer_5[l_244] = l_246;
const uint l_247 = l_1 * uint(4);
const uint l_248 = l_247 + uint(1);
l_mut_65 = float(0.0);

for (uint l_mut_249 = uint(0); l_mut_249 < uint(4); ++l_mut_249) {
const float l_250 = shared_memory_191[l_mut_249];
const bool l_251 = l_250 > float(0.0);
if (l_251) {
const uint l_252 = l_mut_249 * uint(128);
const uint l_253 = l_252 + l_248;
const float l_254 = shared_memory_192[l_253];
const float l_255 = shared_memory_190[l_mut_249];
const float l_256 = l_255 - l_mut_126;
const float l_257 = exp(l_256);
const float l_258 = l_254 * l_257;
l_mut_65 = l_mut_65 + l_258;
}
}
const uint l_259 = l_25 + l_248;
const float l_260 = l_mut_65 * l_mut_49;
const half l_261 = half(l_260);
buffer_5[l_259] = l_261;
const uint l_262 = l_1 * uint(4);
const uint l_263 = l_262 + uint(2);
l_mut_65 = float(0.0);

for (uint l_mut_264 = uint(0); l_mut_264 < uint(4); ++l_mut_264) {
const float l_265 = shared_memory_191[l_mut_264];
const bool l_266 = l_265 > float(0.0);
if (l_266) {
const uint l_267 = l_mut_264 * uint(128);
const uint l_268 = l_267 + l_263;
const float l_269 = shared_memory_192[l_268];
const float l_270 = shared_memory_190[l_mut_264];
const float l_271 = l_270 - l_mut_126;
const float l_272 = exp(l_271);
const float l_273 = l_269 * l_272;
l_mut_65 = l_mut_65 + l_273;
}
}
const uint l_274 = l_25 + l_263;
const float l_275 = l_mut_65 * l_mut_49;
const half l_276 = half(l_275);
buffer_5[l_274] = l_276;
const uint l_277 = l_1 * uint(4);
const uint l_278 = l_277 + uint(3);
l_mut_65 = float(0.0);

for (uint l_mut_279 = uint(0); l_mut_279 < uint(4); ++l_mut_279) {
const float l_280 = shared_memory_191[l_mut_279];
const bool l_281 = l_280 > float(0.0);
if (l_281) {
const uint l_282 = l_mut_279 * uint(128);
const uint l_283 = l_282 + l_278;
const float l_284 = shared_memory_192[l_283];
const float l_285 = shared_memory_190[l_mut_279];
const float l_286 = l_285 - l_mut_126;
const float l_287 = exp(l_286);
const float l_288 = l_284 * l_287;
l_mut_65 = l_mut_65 + l_288;
}
}
const uint l_289 = l_25 + l_278;
const float l_290 = l_mut_65 * l_mut_49;
const half l_291 = half(l_290);
buffer_5[l_289] = l_291;
}

}
