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
threadgroup uchar dynamic_shared_mem[2080];
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_178 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
// Shared array size: 4, 16 bytes
threadgroup float* shared_memory_179 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[16]);
// Shared array size: 512, 2048 bytes
threadgroup float* shared_memory_180 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[32]);
float l_arr_26[4];

float l_arr_27[4];

uint l_mut_14;
float l_mut_44;
float l_mut_45;
uint l_mut_47;
float l_mut_61;
float l_mut_118;
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
const float l_31 = buffer_0[l_30];
l_arr_26[uint(0)] = l_31;
l_arr_27[uint(0)] = float(0.0);
const uint l_32 = l_1 * uint(4);
const uint l_33 = l_25 + l_32;
const uint l_34 = l_33 + uint(1);
const float l_35 = buffer_0[l_34];
l_arr_26[uint(1)] = l_35;
l_arr_27[uint(1)] = float(0.0);
const uint l_36 = l_1 * uint(4);
const uint l_37 = l_25 + l_36;
const uint l_38 = l_37 + uint(2);
const float l_39 = buffer_0[l_38];
l_arr_26[uint(2)] = l_39;
l_arr_27[uint(2)] = float(0.0);
const uint l_40 = l_1 * uint(4);
const uint l_41 = l_25 + l_40;
const uint l_42 = l_41 + uint(3);
const float l_43 = buffer_0[l_42];
l_arr_26[uint(3)] = l_43;
l_arr_27[uint(3)] = float(0.0);
l_mut_44 = float(-3.3999999521443642e38);
l_mut_45 = float(0.0);
const uint l_46 = l_19 + l_0;
l_mut_47 = l_46;
while (true) {
const bool l_48 = l_mut_47 < l_22;
const bool l_49 = !l_48;
if (l_49) {
break;}
const uint l_50 = l_9 * l_4;
const uint l_51 = l_50 + l_mut_47;
const uint l_52 = buffer_3[l_51];
const uint l_53 = l_52 * l_3;
const uint l_54 = l_53 + l_12;

for (uint l_mut_55 = uint(0); l_mut_55 < l_6; ++l_mut_55) {
const uint l_56 = l_mut_47 * l_6;
const uint l_57 = l_56 + l_mut_55;
const bool l_58 = l_57 >= l_mut_14;
const bool l_59 = l_57 < l_13;
const bool l_60 = l_58 && l_59;
if (l_60) {
l_mut_61 = float(0.0);
const uint l_62 = l_1 * uint(4);
const uint l_63 = l_62 + uint(0);
const uint l_64 = l_63 / uint(4);
const uint l_65 = l_63 % uint(4);
const uint l_66 = l_54 * uint(32);
const uint l_67 = l_66 + l_64;
const uint l_68 = l_67 * l_6;
const uint l_69 = l_68 + l_mut_55;
const uint l_70 = l_69 * uint(4);
const uint l_71 = l_70 + l_65;
const float l_72 = l_arr_26[uint(0)];
const float l_73 = buffer_1[l_71];
const float l_74 = l_72 * l_73;
l_mut_61 = l_mut_61 + l_74;
const uint l_75 = l_1 * uint(4);
const uint l_76 = l_75 + uint(1);
const uint l_77 = l_76 / uint(4);
const uint l_78 = l_76 % uint(4);
const uint l_79 = l_54 * uint(32);
const uint l_80 = l_79 + l_77;
const uint l_81 = l_80 * l_6;
const uint l_82 = l_81 + l_mut_55;
const uint l_83 = l_82 * uint(4);
const uint l_84 = l_83 + l_78;
const float l_85 = l_arr_26[uint(1)];
const float l_86 = buffer_1[l_84];
const float l_87 = l_85 * l_86;
l_mut_61 = l_mut_61 + l_87;
const uint l_88 = l_1 * uint(4);
const uint l_89 = l_88 + uint(2);
const uint l_90 = l_89 / uint(4);
const uint l_91 = l_89 % uint(4);
const uint l_92 = l_54 * uint(32);
const uint l_93 = l_92 + l_90;
const uint l_94 = l_93 * l_6;
const uint l_95 = l_94 + l_mut_55;
const uint l_96 = l_95 * uint(4);
const uint l_97 = l_96 + l_91;
const float l_98 = l_arr_26[uint(2)];
const float l_99 = buffer_1[l_97];
const float l_100 = l_98 * l_99;
l_mut_61 = l_mut_61 + l_100;
const uint l_101 = l_1 * uint(4);
const uint l_102 = l_101 + uint(3);
const uint l_103 = l_102 / uint(4);
const uint l_104 = l_102 % uint(4);
const uint l_105 = l_54 * uint(32);
const uint l_106 = l_105 + l_103;
const uint l_107 = l_106 * l_6;
const uint l_108 = l_107 + l_mut_55;
const uint l_109 = l_108 * uint(4);
const uint l_110 = l_109 + l_104;
const float l_111 = l_arr_26[uint(3)];
const float l_112 = buffer_1[l_110];
const float l_113 = l_111 * l_112;
l_mut_61 = l_mut_61 + l_113;
const float l_114 = float {simd_sum(l_mut_61)
};
const float l_115 = l_114 * l_7;
l_mut_61 = l_115;
const bool l_116 = l_8 > float(0.0);
if (l_116) {
const float l_117 = l_mut_61 / l_8;
l_mut_118 = l_117;
const bool l_119 = l_117 > float(10.0);
if (l_119) {
l_mut_118 = float(10.0);
} else {
const bool l_120 = l_117 < float(-10.0);
if (l_120) {
l_mut_118 = float(-10.0);
}
}
const float l_121 = float(2.0) * l_mut_118;
const float l_122 = exp(l_121);
const float l_123 = l_122 - float(1.0);
const float l_124 = l_122 + float(1.0);
const float l_125 = l_123 / l_124;
const float l_126 = l_8 * l_125;
l_mut_61 = l_126;
}
l_mut_118 = l_mut_44;
const bool l_127 = l_mut_61 > l_mut_44;
if (l_127) {
l_mut_118 = l_mut_61;
}
const float l_128 = l_mut_44 - l_mut_118;
const float l_129 = exp(l_128);
const float l_130 = l_mut_61 - l_mut_118;
const float l_131 = exp(l_130);
const float l_132 = l_mut_45 * l_129;
const float l_133 = l_132 + l_131;
l_mut_45 = l_133;
l_mut_44 = l_mut_118;
const uint l_134 = l_1 * uint(4);
const uint l_135 = l_134 + uint(0);
const uint l_136 = l_54 * uint(128);
const uint l_137 = l_136 + l_135;
const uint l_138 = l_137 * l_6;
const uint l_139 = l_138 + l_mut_55;
const float l_140 = l_arr_27[uint(0)];
const float l_141 = l_140 * l_129;
const float l_142 = buffer_2[l_139];
const float l_143 = l_131 * l_142;
const float l_144 = l_141 + l_143;
l_arr_27[uint(0)] = l_144;
const uint l_145 = l_1 * uint(4);
const uint l_146 = l_145 + uint(1);
const uint l_147 = l_54 * uint(128);
const uint l_148 = l_147 + l_146;
const uint l_149 = l_148 * l_6;
const uint l_150 = l_149 + l_mut_55;
const float l_151 = l_arr_27[uint(1)];
const float l_152 = l_151 * l_129;
const float l_153 = buffer_2[l_150];
const float l_154 = l_131 * l_153;
const float l_155 = l_152 + l_154;
l_arr_27[uint(1)] = l_155;
const uint l_156 = l_1 * uint(4);
const uint l_157 = l_156 + uint(2);
const uint l_158 = l_54 * uint(128);
const uint l_159 = l_158 + l_157;
const uint l_160 = l_159 * l_6;
const uint l_161 = l_160 + l_mut_55;
const float l_162 = l_arr_27[uint(2)];
const float l_163 = l_162 * l_129;
const float l_164 = buffer_2[l_161];
const float l_165 = l_131 * l_164;
const float l_166 = l_163 + l_165;
l_arr_27[uint(2)] = l_166;
const uint l_167 = l_1 * uint(4);
const uint l_168 = l_167 + uint(3);
const uint l_169 = l_54 * uint(128);
const uint l_170 = l_169 + l_168;
const uint l_171 = l_170 * l_6;
const uint l_172 = l_171 + l_mut_55;
const float l_173 = l_arr_27[uint(3)];
const float l_174 = l_173 * l_129;
const float l_175 = buffer_2[l_172];
const float l_176 = l_131 * l_175;
const float l_177 = l_174 + l_176;
l_arr_27[uint(3)] = l_177;
}
}
l_mut_47 = l_mut_47 + uint(4);
}
const bool l_181 = l_1 == uint(0);
if (l_181) {
shared_memory_178[l_0] = l_mut_44;
shared_memory_179[l_0] = l_mut_45;
}
const uint l_182 = l_0 * uint(128);
const uint l_183 = l_1 * uint(4);
const uint l_184 = l_182 + l_183;
const uint l_185 = l_184 + uint(0);
const float l_186 = l_arr_27[uint(0)];
shared_memory_180[l_185] = l_186;
const uint l_187 = l_0 * uint(128);
const uint l_188 = l_1 * uint(4);
const uint l_189 = l_187 + l_188;
const uint l_190 = l_189 + uint(1);
const float l_191 = l_arr_27[uint(1)];
shared_memory_180[l_190] = l_191;
const uint l_192 = l_0 * uint(128);
const uint l_193 = l_1 * uint(4);
const uint l_194 = l_192 + l_193;
const uint l_195 = l_194 + uint(2);
const float l_196 = l_arr_27[uint(2)];
shared_memory_180[l_195] = l_196;
const uint l_197 = l_0 * uint(128);
const uint l_198 = l_1 * uint(4);
const uint l_199 = l_197 + l_198;
const uint l_200 = l_199 + uint(3);
const float l_201 = l_arr_27[uint(3)];
shared_memory_180[l_200] = l_201;
threadgroup_barrier(mem_flags::mem_threadgroup);
const bool l_202 = l_0 == uint(0);
if (l_202) {
l_mut_118 = float(-3.3999999521443642e38);

for (uint l_mut_203 = uint(0); l_mut_203 < uint(4); ++l_mut_203) {
const float l_204 = shared_memory_179[l_mut_203];
const bool l_205 = l_204 > float(0.0);
const float l_206 = shared_memory_178[l_mut_203];
const bool l_207 = l_206 > l_mut_118;
const bool l_208 = l_205 && l_207;
if (l_208) {
const float l_209 = shared_memory_178[l_mut_203];
l_mut_118 = l_209;
}
}
l_mut_61 = float(0.0);

for (uint l_mut_210 = uint(0); l_mut_210 < uint(4); ++l_mut_210) {
const float l_211 = shared_memory_179[l_mut_210];
const bool l_212 = l_211 > float(0.0);
if (l_212) {
const float l_213 = shared_memory_179[l_mut_210];
const float l_214 = shared_memory_178[l_mut_210];
const float l_215 = l_214 - l_mut_118;
const float l_216 = exp(l_215);
const float l_217 = l_213 * l_216;
l_mut_61 = l_mut_61 + l_217;
}
}
l_mut_45 = float(0.0);
const bool l_218 = l_mut_61 > float(0.0);
if (l_218) {
const float l_219 = float(1.0) / l_mut_61;
l_mut_45 = l_219;
}
const uint l_220 = l_1 * uint(4);
const uint l_221 = l_220 + uint(0);
l_mut_61 = float(0.0);

for (uint l_mut_222 = uint(0); l_mut_222 < uint(4); ++l_mut_222) {
const float l_223 = shared_memory_179[l_mut_222];
const bool l_224 = l_223 > float(0.0);
if (l_224) {
const uint l_225 = l_mut_222 * uint(128);
const uint l_226 = l_225 + l_221;
const float l_227 = shared_memory_180[l_226];
const float l_228 = shared_memory_178[l_mut_222];
const float l_229 = l_228 - l_mut_118;
const float l_230 = exp(l_229);
const float l_231 = l_227 * l_230;
l_mut_61 = l_mut_61 + l_231;
}
}
const uint l_232 = l_25 + l_221;
const float l_233 = l_mut_61 * l_mut_45;
buffer_5[l_232] = l_233;
const uint l_234 = l_1 * uint(4);
const uint l_235 = l_234 + uint(1);
l_mut_61 = float(0.0);

for (uint l_mut_236 = uint(0); l_mut_236 < uint(4); ++l_mut_236) {
const float l_237 = shared_memory_179[l_mut_236];
const bool l_238 = l_237 > float(0.0);
if (l_238) {
const uint l_239 = l_mut_236 * uint(128);
const uint l_240 = l_239 + l_235;
const float l_241 = shared_memory_180[l_240];
const float l_242 = shared_memory_178[l_mut_236];
const float l_243 = l_242 - l_mut_118;
const float l_244 = exp(l_243);
const float l_245 = l_241 * l_244;
l_mut_61 = l_mut_61 + l_245;
}
}
const uint l_246 = l_25 + l_235;
const float l_247 = l_mut_61 * l_mut_45;
buffer_5[l_246] = l_247;
const uint l_248 = l_1 * uint(4);
const uint l_249 = l_248 + uint(2);
l_mut_61 = float(0.0);

for (uint l_mut_250 = uint(0); l_mut_250 < uint(4); ++l_mut_250) {
const float l_251 = shared_memory_179[l_mut_250];
const bool l_252 = l_251 > float(0.0);
if (l_252) {
const uint l_253 = l_mut_250 * uint(128);
const uint l_254 = l_253 + l_249;
const float l_255 = shared_memory_180[l_254];
const float l_256 = shared_memory_178[l_mut_250];
const float l_257 = l_256 - l_mut_118;
const float l_258 = exp(l_257);
const float l_259 = l_255 * l_258;
l_mut_61 = l_mut_61 + l_259;
}
}
const uint l_260 = l_25 + l_249;
const float l_261 = l_mut_61 * l_mut_45;
buffer_5[l_260] = l_261;
const uint l_262 = l_1 * uint(4);
const uint l_263 = l_262 + uint(3);
l_mut_61 = float(0.0);

for (uint l_mut_264 = uint(0); l_mut_264 < uint(4); ++l_mut_264) {
const float l_265 = shared_memory_179[l_mut_264];
const bool l_266 = l_265 > float(0.0);
if (l_266) {
const uint l_267 = l_mut_264 * uint(128);
const uint l_268 = l_267 + l_263;
const float l_269 = shared_memory_180[l_268];
const float l_270 = shared_memory_178[l_mut_264];
const float l_271 = l_270 - l_mut_118;
const float l_272 = exp(l_271);
const float l_273 = l_269 * l_272;
l_mut_61 = l_mut_61 + l_273;
}
}
const uint l_274 = l_25 + l_263;
const float l_275 = l_mut_61 * l_mut_45;
buffer_5[l_274] = l_275;
}

}
