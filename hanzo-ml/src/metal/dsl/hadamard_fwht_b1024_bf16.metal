#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void hadamard_fwht_f_(
    const device bfloat * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device bfloat * buffer_2 [[buffer(2)]],
    const device uint * buffer_3 [[buffer(3)]],
    constant info_st& info [[buffer(4)]],
    uint3 threads_per_threadgroup [[threads_per_threadgroup]],
    uint3 threadgroups_per_grid [[threadgroups_per_grid]],
    uint3 thread_pos_in_threadgroup [[thread_position_in_threadgroup]],
    uint3 threadgroup_pos_in_grid [[threadgroup_position_in_grid]],
    uint simd_size [[threads_per_simdgroup]]
) {
uint threads_per_simdgroup_checked = min(simd_size, threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z);
threadgroup uchar dynamic_shared_mem[4096];
// Shared array size: 1024, 4096 bytes
threadgroup float* shared_memory_4 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
float l_arr_5[4];

float l_arr_6[4];

uint l_mut_7;
float l_mut_17;
float l_mut_25;
const uint l_0 = buffer_3[uint(1)];
const uint l_1 = buffer_3[uint(2)];
const uint l_2 = buffer_3[uint(3)];
const uint l_3 = buffer_3[uint(4)];
l_mut_7 = threadgroup_pos_in_grid.x;
while (true) {
const bool l_8 = l_mut_7 < l_3;
const bool l_9 = !l_8;
if (l_9) {
break;}
const uint l_10 = l_mut_7 * uint(1024);
const uint l_11 = l_mut_7 % l_0;
const uint l_12 = l_11 * uint(1024);
const uint l_13 = uint(0) + thread_pos_in_threadgroup.x;
const uint l_14 = l_10 + l_13;
const bfloat l_15 = buffer_0[l_14];
const float l_16 = float(l_15);
l_mut_17 = l_16;
const bool l_18 = l_2 == uint(0);
const bool l_19 = l_1 != uint(0);
const bool l_20 = l_18 && l_19;
if (l_20) {
const uint l_21 = l_12 + l_13;
const float l_22 = buffer_1[l_21];
l_mut_17 = l_mut_17 * l_22;
}
const float l_23 = { simd_shuffle_xor(l_mut_17, uint(1)) };
const float l_24 = l_mut_17 + l_23;
l_mut_25 = l_24;
const uint l_26 = thread_pos_in_threadgroup.x & uint(1);
const bool l_27 = l_26 != uint(0);
if (l_27) {
const float l_28 = l_23 - l_mut_17;
l_mut_25 = l_28;
}
l_mut_17 = l_mut_25;
const float l_29 = { simd_shuffle_xor(l_mut_17, uint(2)) };
const float l_30 = l_mut_17 + l_29;
l_mut_25 = l_30;
const uint l_31 = thread_pos_in_threadgroup.x & uint(2);
const bool l_32 = l_31 != uint(0);
if (l_32) {
const float l_33 = l_29 - l_mut_17;
l_mut_25 = l_33;
}
l_mut_17 = l_mut_25;
const float l_34 = { simd_shuffle_xor(l_mut_17, uint(4)) };
const float l_35 = l_mut_17 + l_34;
l_mut_25 = l_35;
const uint l_36 = thread_pos_in_threadgroup.x & uint(4);
const bool l_37 = l_36 != uint(0);
if (l_37) {
const float l_38 = l_34 - l_mut_17;
l_mut_25 = l_38;
}
l_mut_17 = l_mut_25;
const float l_39 = { simd_shuffle_xor(l_mut_17, uint(8)) };
const float l_40 = l_mut_17 + l_39;
l_mut_25 = l_40;
const uint l_41 = thread_pos_in_threadgroup.x & uint(8);
const bool l_42 = l_41 != uint(0);
if (l_42) {
const float l_43 = l_39 - l_mut_17;
l_mut_25 = l_43;
}
l_mut_17 = l_mut_25;
const float l_44 = { simd_shuffle_xor(l_mut_17, uint(16)) };
const float l_45 = l_mut_17 + l_44;
l_mut_25 = l_45;
const uint l_46 = thread_pos_in_threadgroup.x & uint(16);
const bool l_47 = l_46 != uint(0);
if (l_47) {
const float l_48 = l_44 - l_mut_17;
l_mut_25 = l_48;
}
l_mut_17 = l_mut_25;
l_arr_5[uint(0)] = l_mut_17;
const uint l_49 = uint(256) + thread_pos_in_threadgroup.x;
const uint l_50 = l_10 + l_49;
const bfloat l_51 = buffer_0[l_50];
const float l_52 = float(l_51);
l_mut_25 = l_52;
const bool l_53 = l_2 == uint(0);
const bool l_54 = l_1 != uint(0);
const bool l_55 = l_53 && l_54;
if (l_55) {
const uint l_56 = l_12 + l_49;
const float l_57 = buffer_1[l_56];
l_mut_25 = l_mut_25 * l_57;
}
const float l_58 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_59 = l_mut_25 + l_58;
l_mut_17 = l_59;
const uint l_60 = thread_pos_in_threadgroup.x & uint(1);
const bool l_61 = l_60 != uint(0);
if (l_61) {
const float l_62 = l_58 - l_mut_25;
l_mut_17 = l_62;
}
l_mut_25 = l_mut_17;
const float l_63 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_64 = l_mut_25 + l_63;
l_mut_17 = l_64;
const uint l_65 = thread_pos_in_threadgroup.x & uint(2);
const bool l_66 = l_65 != uint(0);
if (l_66) {
const float l_67 = l_63 - l_mut_25;
l_mut_17 = l_67;
}
l_mut_25 = l_mut_17;
const float l_68 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_69 = l_mut_25 + l_68;
l_mut_17 = l_69;
const uint l_70 = thread_pos_in_threadgroup.x & uint(4);
const bool l_71 = l_70 != uint(0);
if (l_71) {
const float l_72 = l_68 - l_mut_25;
l_mut_17 = l_72;
}
l_mut_25 = l_mut_17;
const float l_73 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_74 = l_mut_25 + l_73;
l_mut_17 = l_74;
const uint l_75 = thread_pos_in_threadgroup.x & uint(8);
const bool l_76 = l_75 != uint(0);
if (l_76) {
const float l_77 = l_73 - l_mut_25;
l_mut_17 = l_77;
}
l_mut_25 = l_mut_17;
const float l_78 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_79 = l_mut_25 + l_78;
l_mut_17 = l_79;
const uint l_80 = thread_pos_in_threadgroup.x & uint(16);
const bool l_81 = l_80 != uint(0);
if (l_81) {
const float l_82 = l_78 - l_mut_25;
l_mut_17 = l_82;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(1)] = l_mut_25;
const uint l_83 = uint(512) + thread_pos_in_threadgroup.x;
const uint l_84 = l_10 + l_83;
const bfloat l_85 = buffer_0[l_84];
const float l_86 = float(l_85);
l_mut_25 = l_86;
const bool l_87 = l_2 == uint(0);
const bool l_88 = l_1 != uint(0);
const bool l_89 = l_87 && l_88;
if (l_89) {
const uint l_90 = l_12 + l_83;
const float l_91 = buffer_1[l_90];
l_mut_25 = l_mut_25 * l_91;
}
const float l_92 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_93 = l_mut_25 + l_92;
l_mut_17 = l_93;
const uint l_94 = thread_pos_in_threadgroup.x & uint(1);
const bool l_95 = l_94 != uint(0);
if (l_95) {
const float l_96 = l_92 - l_mut_25;
l_mut_17 = l_96;
}
l_mut_25 = l_mut_17;
const float l_97 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_98 = l_mut_25 + l_97;
l_mut_17 = l_98;
const uint l_99 = thread_pos_in_threadgroup.x & uint(2);
const bool l_100 = l_99 != uint(0);
if (l_100) {
const float l_101 = l_97 - l_mut_25;
l_mut_17 = l_101;
}
l_mut_25 = l_mut_17;
const float l_102 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_103 = l_mut_25 + l_102;
l_mut_17 = l_103;
const uint l_104 = thread_pos_in_threadgroup.x & uint(4);
const bool l_105 = l_104 != uint(0);
if (l_105) {
const float l_106 = l_102 - l_mut_25;
l_mut_17 = l_106;
}
l_mut_25 = l_mut_17;
const float l_107 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_108 = l_mut_25 + l_107;
l_mut_17 = l_108;
const uint l_109 = thread_pos_in_threadgroup.x & uint(8);
const bool l_110 = l_109 != uint(0);
if (l_110) {
const float l_111 = l_107 - l_mut_25;
l_mut_17 = l_111;
}
l_mut_25 = l_mut_17;
const float l_112 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_113 = l_mut_25 + l_112;
l_mut_17 = l_113;
const uint l_114 = thread_pos_in_threadgroup.x & uint(16);
const bool l_115 = l_114 != uint(0);
if (l_115) {
const float l_116 = l_112 - l_mut_25;
l_mut_17 = l_116;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(2)] = l_mut_25;
const uint l_117 = uint(768) + thread_pos_in_threadgroup.x;
const uint l_118 = l_10 + l_117;
const bfloat l_119 = buffer_0[l_118];
const float l_120 = float(l_119);
l_mut_25 = l_120;
const bool l_121 = l_2 == uint(0);
const bool l_122 = l_1 != uint(0);
const bool l_123 = l_121 && l_122;
if (l_123) {
const uint l_124 = l_12 + l_117;
const float l_125 = buffer_1[l_124];
l_mut_25 = l_mut_25 * l_125;
}
const float l_126 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_127 = l_mut_25 + l_126;
l_mut_17 = l_127;
const uint l_128 = thread_pos_in_threadgroup.x & uint(1);
const bool l_129 = l_128 != uint(0);
if (l_129) {
const float l_130 = l_126 - l_mut_25;
l_mut_17 = l_130;
}
l_mut_25 = l_mut_17;
const float l_131 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_132 = l_mut_25 + l_131;
l_mut_17 = l_132;
const uint l_133 = thread_pos_in_threadgroup.x & uint(2);
const bool l_134 = l_133 != uint(0);
if (l_134) {
const float l_135 = l_131 - l_mut_25;
l_mut_17 = l_135;
}
l_mut_25 = l_mut_17;
const float l_136 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_137 = l_mut_25 + l_136;
l_mut_17 = l_137;
const uint l_138 = thread_pos_in_threadgroup.x & uint(4);
const bool l_139 = l_138 != uint(0);
if (l_139) {
const float l_140 = l_136 - l_mut_25;
l_mut_17 = l_140;
}
l_mut_25 = l_mut_17;
const float l_141 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_142 = l_mut_25 + l_141;
l_mut_17 = l_142;
const uint l_143 = thread_pos_in_threadgroup.x & uint(8);
const bool l_144 = l_143 != uint(0);
if (l_144) {
const float l_145 = l_141 - l_mut_25;
l_mut_17 = l_145;
}
l_mut_25 = l_mut_17;
const float l_146 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_147 = l_mut_25 + l_146;
l_mut_17 = l_147;
const uint l_148 = thread_pos_in_threadgroup.x & uint(16);
const bool l_149 = l_148 != uint(0);
if (l_149) {
const float l_150 = l_146 - l_mut_25;
l_mut_17 = l_150;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(3)] = l_mut_25;
const uint l_151 = uint(0) + thread_pos_in_threadgroup.x;
const float l_152 = l_arr_5[uint(0)];
shared_memory_4[l_151] = l_152;
const uint l_153 = uint(256) + thread_pos_in_threadgroup.x;
const float l_154 = l_arr_5[uint(1)];
shared_memory_4[l_153] = l_154;
const uint l_155 = uint(512) + thread_pos_in_threadgroup.x;
const float l_156 = l_arr_5[uint(2)];
shared_memory_4[l_155] = l_156;
const uint l_157 = uint(768) + thread_pos_in_threadgroup.x;
const float l_158 = l_arr_5[uint(3)];
shared_memory_4[l_157] = l_158;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_159 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_160 = thread_pos_in_threadgroup.x & uint(32);
const bool l_161 = l_160 != uint(0);
const float l_162 = l_arr_5[uint(0)];
const uint l_163 = uint(0) + l_159;
const float l_164 = shared_memory_4[l_163];
const float l_165 = l_162 + l_164;
l_mut_25 = l_165;
if (l_161) {
const float l_166 = l_164 - l_162;
l_mut_25 = l_166;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_167 = l_arr_5[uint(1)];
const uint l_168 = uint(256) + l_159;
const float l_169 = shared_memory_4[l_168];
const float l_170 = l_167 + l_169;
l_mut_25 = l_170;
if (l_161) {
const float l_171 = l_169 - l_167;
l_mut_25 = l_171;
}
l_arr_6[uint(1)] = l_mut_25;
const float l_172 = l_arr_5[uint(2)];
const uint l_173 = uint(512) + l_159;
const float l_174 = shared_memory_4[l_173];
const float l_175 = l_172 + l_174;
l_mut_25 = l_175;
if (l_161) {
const float l_176 = l_174 - l_172;
l_mut_25 = l_176;
}
l_arr_6[uint(2)] = l_mut_25;
const float l_177 = l_arr_5[uint(3)];
const uint l_178 = uint(768) + l_159;
const float l_179 = shared_memory_4[l_178];
const float l_180 = l_177 + l_179;
l_mut_25 = l_180;
if (l_161) {
const float l_181 = l_179 - l_177;
l_mut_25 = l_181;
}
l_arr_6[uint(3)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_182 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_182;
const uint l_183 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_183] = l_182;
const float l_184 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_184;
const uint l_185 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_185] = l_184;
const float l_186 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_186;
const uint l_187 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_187] = l_186;
const float l_188 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_188;
const uint l_189 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_189] = l_188;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_190 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_191 = thread_pos_in_threadgroup.x & uint(64);
const bool l_192 = l_191 != uint(0);
const float l_193 = l_arr_5[uint(0)];
const uint l_194 = uint(0) + l_190;
const float l_195 = shared_memory_4[l_194];
const float l_196 = l_193 + l_195;
l_mut_25 = l_196;
if (l_192) {
const float l_197 = l_195 - l_193;
l_mut_25 = l_197;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_198 = l_arr_5[uint(1)];
const uint l_199 = uint(256) + l_190;
const float l_200 = shared_memory_4[l_199];
const float l_201 = l_198 + l_200;
l_mut_25 = l_201;
if (l_192) {
const float l_202 = l_200 - l_198;
l_mut_25 = l_202;
}
l_arr_6[uint(1)] = l_mut_25;
const float l_203 = l_arr_5[uint(2)];
const uint l_204 = uint(512) + l_190;
const float l_205 = shared_memory_4[l_204];
const float l_206 = l_203 + l_205;
l_mut_25 = l_206;
if (l_192) {
const float l_207 = l_205 - l_203;
l_mut_25 = l_207;
}
l_arr_6[uint(2)] = l_mut_25;
const float l_208 = l_arr_5[uint(3)];
const uint l_209 = uint(768) + l_190;
const float l_210 = shared_memory_4[l_209];
const float l_211 = l_208 + l_210;
l_mut_25 = l_211;
if (l_192) {
const float l_212 = l_210 - l_208;
l_mut_25 = l_212;
}
l_arr_6[uint(3)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_213 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_213;
const uint l_214 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_214] = l_213;
const float l_215 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_215;
const uint l_216 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_216] = l_215;
const float l_217 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_217;
const uint l_218 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_218] = l_217;
const float l_219 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_219;
const uint l_220 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_220] = l_219;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_221 = thread_pos_in_threadgroup.x ^ uint(128);
const uint l_222 = thread_pos_in_threadgroup.x & uint(128);
const bool l_223 = l_222 != uint(0);
const float l_224 = l_arr_5[uint(0)];
const uint l_225 = uint(0) + l_221;
const float l_226 = shared_memory_4[l_225];
const float l_227 = l_224 + l_226;
l_mut_25 = l_227;
if (l_223) {
const float l_228 = l_226 - l_224;
l_mut_25 = l_228;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_229 = l_arr_5[uint(1)];
const uint l_230 = uint(256) + l_221;
const float l_231 = shared_memory_4[l_230];
const float l_232 = l_229 + l_231;
l_mut_25 = l_232;
if (l_223) {
const float l_233 = l_231 - l_229;
l_mut_25 = l_233;
}
l_arr_6[uint(1)] = l_mut_25;
const float l_234 = l_arr_5[uint(2)];
const uint l_235 = uint(512) + l_221;
const float l_236 = shared_memory_4[l_235];
const float l_237 = l_234 + l_236;
l_mut_25 = l_237;
if (l_223) {
const float l_238 = l_236 - l_234;
l_mut_25 = l_238;
}
l_arr_6[uint(2)] = l_mut_25;
const float l_239 = l_arr_5[uint(3)];
const uint l_240 = uint(768) + l_221;
const float l_241 = shared_memory_4[l_240];
const float l_242 = l_239 + l_241;
l_mut_25 = l_242;
if (l_223) {
const float l_243 = l_241 - l_239;
l_mut_25 = l_243;
}
l_arr_6[uint(3)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_244 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_244;
const float l_245 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_245;
const float l_246 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_246;
const float l_247 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_247;
const float l_248 = l_arr_5[uint(0)];
const float l_249 = l_arr_5[uint(1)];
const float l_250 = l_248 + l_249;
l_arr_5[uint(0)] = l_250;
const float l_251 = l_248 - l_249;
l_arr_5[uint(1)] = l_251;
const float l_252 = l_arr_5[uint(2)];
const float l_253 = l_arr_5[uint(3)];
const float l_254 = l_252 + l_253;
l_arr_5[uint(2)] = l_254;
const float l_255 = l_252 - l_253;
l_arr_5[uint(3)] = l_255;
const float l_256 = l_arr_5[uint(0)];
const float l_257 = l_arr_5[uint(2)];
const float l_258 = l_256 + l_257;
l_arr_5[uint(0)] = l_258;
const float l_259 = l_256 - l_257;
l_arr_5[uint(2)] = l_259;
const float l_260 = l_arr_5[uint(1)];
const float l_261 = l_arr_5[uint(3)];
const float l_262 = l_260 + l_261;
l_arr_5[uint(1)] = l_262;
const float l_263 = l_260 - l_261;
l_arr_5[uint(3)] = l_263;
const uint l_264 = uint(0) + thread_pos_in_threadgroup.x;
const float l_265 = l_arr_5[uint(0)];
const float l_266 = l_265 * float(0.03125);
l_mut_25 = l_266;
const bool l_267 = l_2 != uint(0);
const bool l_268 = l_1 != uint(0);
const bool l_269 = l_267 && l_268;
if (l_269) {
const uint l_270 = l_12 + l_264;
const float l_271 = buffer_1[l_270];
l_mut_25 = l_mut_25 * l_271;
}
const uint l_272 = l_10 + l_264;
const bfloat l_273 = bfloat(l_mut_25);
buffer_2[l_272] = l_273;
const uint l_274 = uint(256) + thread_pos_in_threadgroup.x;
const float l_275 = l_arr_5[uint(1)];
const float l_276 = l_275 * float(0.03125);
l_mut_25 = l_276;
const bool l_277 = l_2 != uint(0);
const bool l_278 = l_1 != uint(0);
const bool l_279 = l_277 && l_278;
if (l_279) {
const uint l_280 = l_12 + l_274;
const float l_281 = buffer_1[l_280];
l_mut_25 = l_mut_25 * l_281;
}
const uint l_282 = l_10 + l_274;
const bfloat l_283 = bfloat(l_mut_25);
buffer_2[l_282] = l_283;
const uint l_284 = uint(512) + thread_pos_in_threadgroup.x;
const float l_285 = l_arr_5[uint(2)];
const float l_286 = l_285 * float(0.03125);
l_mut_25 = l_286;
const bool l_287 = l_2 != uint(0);
const bool l_288 = l_1 != uint(0);
const bool l_289 = l_287 && l_288;
if (l_289) {
const uint l_290 = l_12 + l_284;
const float l_291 = buffer_1[l_290];
l_mut_25 = l_mut_25 * l_291;
}
const uint l_292 = l_10 + l_284;
const bfloat l_293 = bfloat(l_mut_25);
buffer_2[l_292] = l_293;
const uint l_294 = uint(768) + thread_pos_in_threadgroup.x;
const float l_295 = l_arr_5[uint(3)];
const float l_296 = l_295 * float(0.03125);
l_mut_25 = l_296;
const bool l_297 = l_2 != uint(0);
const bool l_298 = l_1 != uint(0);
const bool l_299 = l_297 && l_298;
if (l_299) {
const uint l_300 = l_12 + l_294;
const float l_301 = buffer_1[l_300];
l_mut_25 = l_mut_25 * l_301;
}
const uint l_302 = l_10 + l_294;
const bfloat l_303 = bfloat(l_mut_25);
buffer_2[l_302] = l_303;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
