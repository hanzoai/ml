#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void hadamard_fwht_f_f32(
    const device float * buffer_0 [[buffer(0)]],
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
uint threads_per_simdgroup_checked = min(simd_size, threads_per_threadgroup.x * threads_per_threadgroup.y * threads_per_threadgroup.z);
threadgroup uchar dynamic_shared_mem[4096];
// Shared array size: 1024, 4096 bytes
threadgroup float* shared_memory_4 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
float l_arr_5[4];

float l_arr_6[4];

uint l_mut_7;
float l_mut_16;
float l_mut_24;
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
const float l_15 = buffer_0[l_14];
l_mut_16 = l_15;
const bool l_17 = l_2 == uint(0);
const bool l_18 = l_1 != uint(0);
const bool l_19 = l_17 && l_18;
if (l_19) {
const uint l_20 = l_12 + l_13;
const float l_21 = buffer_1[l_20];
l_mut_16 = l_mut_16 * l_21;
}
const float l_22 = { simd_shuffle_xor(l_mut_16, uint(1)) };
const float l_23 = l_mut_16 + l_22;
l_mut_24 = l_23;
const uint l_25 = thread_pos_in_threadgroup.x & uint(1);
const bool l_26 = l_25 != uint(0);
if (l_26) {
const float l_27 = l_22 - l_mut_16;
l_mut_24 = l_27;
}
l_mut_16 = l_mut_24;
const float l_28 = { simd_shuffle_xor(l_mut_16, uint(2)) };
const float l_29 = l_mut_16 + l_28;
l_mut_24 = l_29;
const uint l_30 = thread_pos_in_threadgroup.x & uint(2);
const bool l_31 = l_30 != uint(0);
if (l_31) {
const float l_32 = l_28 - l_mut_16;
l_mut_24 = l_32;
}
l_mut_16 = l_mut_24;
const float l_33 = { simd_shuffle_xor(l_mut_16, uint(4)) };
const float l_34 = l_mut_16 + l_33;
l_mut_24 = l_34;
const uint l_35 = thread_pos_in_threadgroup.x & uint(4);
const bool l_36 = l_35 != uint(0);
if (l_36) {
const float l_37 = l_33 - l_mut_16;
l_mut_24 = l_37;
}
l_mut_16 = l_mut_24;
const float l_38 = { simd_shuffle_xor(l_mut_16, uint(8)) };
const float l_39 = l_mut_16 + l_38;
l_mut_24 = l_39;
const uint l_40 = thread_pos_in_threadgroup.x & uint(8);
const bool l_41 = l_40 != uint(0);
if (l_41) {
const float l_42 = l_38 - l_mut_16;
l_mut_24 = l_42;
}
l_mut_16 = l_mut_24;
const float l_43 = { simd_shuffle_xor(l_mut_16, uint(16)) };
const float l_44 = l_mut_16 + l_43;
l_mut_24 = l_44;
const uint l_45 = thread_pos_in_threadgroup.x & uint(16);
const bool l_46 = l_45 != uint(0);
if (l_46) {
const float l_47 = l_43 - l_mut_16;
l_mut_24 = l_47;
}
l_mut_16 = l_mut_24;
l_arr_5[uint(0)] = l_mut_16;
const uint l_48 = uint(256) + thread_pos_in_threadgroup.x;
const uint l_49 = l_10 + l_48;
const float l_50 = buffer_0[l_49];
l_mut_24 = l_50;
const bool l_51 = l_2 == uint(0);
const bool l_52 = l_1 != uint(0);
const bool l_53 = l_51 && l_52;
if (l_53) {
const uint l_54 = l_12 + l_48;
const float l_55 = buffer_1[l_54];
l_mut_24 = l_mut_24 * l_55;
}
const float l_56 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_57 = l_mut_24 + l_56;
l_mut_16 = l_57;
const uint l_58 = thread_pos_in_threadgroup.x & uint(1);
const bool l_59 = l_58 != uint(0);
if (l_59) {
const float l_60 = l_56 - l_mut_24;
l_mut_16 = l_60;
}
l_mut_24 = l_mut_16;
const float l_61 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_62 = l_mut_24 + l_61;
l_mut_16 = l_62;
const uint l_63 = thread_pos_in_threadgroup.x & uint(2);
const bool l_64 = l_63 != uint(0);
if (l_64) {
const float l_65 = l_61 - l_mut_24;
l_mut_16 = l_65;
}
l_mut_24 = l_mut_16;
const float l_66 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_67 = l_mut_24 + l_66;
l_mut_16 = l_67;
const uint l_68 = thread_pos_in_threadgroup.x & uint(4);
const bool l_69 = l_68 != uint(0);
if (l_69) {
const float l_70 = l_66 - l_mut_24;
l_mut_16 = l_70;
}
l_mut_24 = l_mut_16;
const float l_71 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_72 = l_mut_24 + l_71;
l_mut_16 = l_72;
const uint l_73 = thread_pos_in_threadgroup.x & uint(8);
const bool l_74 = l_73 != uint(0);
if (l_74) {
const float l_75 = l_71 - l_mut_24;
l_mut_16 = l_75;
}
l_mut_24 = l_mut_16;
const float l_76 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_77 = l_mut_24 + l_76;
l_mut_16 = l_77;
const uint l_78 = thread_pos_in_threadgroup.x & uint(16);
const bool l_79 = l_78 != uint(0);
if (l_79) {
const float l_80 = l_76 - l_mut_24;
l_mut_16 = l_80;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(1)] = l_mut_24;
const uint l_81 = uint(512) + thread_pos_in_threadgroup.x;
const uint l_82 = l_10 + l_81;
const float l_83 = buffer_0[l_82];
l_mut_24 = l_83;
const bool l_84 = l_2 == uint(0);
const bool l_85 = l_1 != uint(0);
const bool l_86 = l_84 && l_85;
if (l_86) {
const uint l_87 = l_12 + l_81;
const float l_88 = buffer_1[l_87];
l_mut_24 = l_mut_24 * l_88;
}
const float l_89 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_90 = l_mut_24 + l_89;
l_mut_16 = l_90;
const uint l_91 = thread_pos_in_threadgroup.x & uint(1);
const bool l_92 = l_91 != uint(0);
if (l_92) {
const float l_93 = l_89 - l_mut_24;
l_mut_16 = l_93;
}
l_mut_24 = l_mut_16;
const float l_94 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_95 = l_mut_24 + l_94;
l_mut_16 = l_95;
const uint l_96 = thread_pos_in_threadgroup.x & uint(2);
const bool l_97 = l_96 != uint(0);
if (l_97) {
const float l_98 = l_94 - l_mut_24;
l_mut_16 = l_98;
}
l_mut_24 = l_mut_16;
const float l_99 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_100 = l_mut_24 + l_99;
l_mut_16 = l_100;
const uint l_101 = thread_pos_in_threadgroup.x & uint(4);
const bool l_102 = l_101 != uint(0);
if (l_102) {
const float l_103 = l_99 - l_mut_24;
l_mut_16 = l_103;
}
l_mut_24 = l_mut_16;
const float l_104 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_105 = l_mut_24 + l_104;
l_mut_16 = l_105;
const uint l_106 = thread_pos_in_threadgroup.x & uint(8);
const bool l_107 = l_106 != uint(0);
if (l_107) {
const float l_108 = l_104 - l_mut_24;
l_mut_16 = l_108;
}
l_mut_24 = l_mut_16;
const float l_109 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_110 = l_mut_24 + l_109;
l_mut_16 = l_110;
const uint l_111 = thread_pos_in_threadgroup.x & uint(16);
const bool l_112 = l_111 != uint(0);
if (l_112) {
const float l_113 = l_109 - l_mut_24;
l_mut_16 = l_113;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(2)] = l_mut_24;
const uint l_114 = uint(768) + thread_pos_in_threadgroup.x;
const uint l_115 = l_10 + l_114;
const float l_116 = buffer_0[l_115];
l_mut_24 = l_116;
const bool l_117 = l_2 == uint(0);
const bool l_118 = l_1 != uint(0);
const bool l_119 = l_117 && l_118;
if (l_119) {
const uint l_120 = l_12 + l_114;
const float l_121 = buffer_1[l_120];
l_mut_24 = l_mut_24 * l_121;
}
const float l_122 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_123 = l_mut_24 + l_122;
l_mut_16 = l_123;
const uint l_124 = thread_pos_in_threadgroup.x & uint(1);
const bool l_125 = l_124 != uint(0);
if (l_125) {
const float l_126 = l_122 - l_mut_24;
l_mut_16 = l_126;
}
l_mut_24 = l_mut_16;
const float l_127 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_128 = l_mut_24 + l_127;
l_mut_16 = l_128;
const uint l_129 = thread_pos_in_threadgroup.x & uint(2);
const bool l_130 = l_129 != uint(0);
if (l_130) {
const float l_131 = l_127 - l_mut_24;
l_mut_16 = l_131;
}
l_mut_24 = l_mut_16;
const float l_132 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_133 = l_mut_24 + l_132;
l_mut_16 = l_133;
const uint l_134 = thread_pos_in_threadgroup.x & uint(4);
const bool l_135 = l_134 != uint(0);
if (l_135) {
const float l_136 = l_132 - l_mut_24;
l_mut_16 = l_136;
}
l_mut_24 = l_mut_16;
const float l_137 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_138 = l_mut_24 + l_137;
l_mut_16 = l_138;
const uint l_139 = thread_pos_in_threadgroup.x & uint(8);
const bool l_140 = l_139 != uint(0);
if (l_140) {
const float l_141 = l_137 - l_mut_24;
l_mut_16 = l_141;
}
l_mut_24 = l_mut_16;
const float l_142 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_143 = l_mut_24 + l_142;
l_mut_16 = l_143;
const uint l_144 = thread_pos_in_threadgroup.x & uint(16);
const bool l_145 = l_144 != uint(0);
if (l_145) {
const float l_146 = l_142 - l_mut_24;
l_mut_16 = l_146;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(3)] = l_mut_24;
const uint l_147 = uint(0) + thread_pos_in_threadgroup.x;
const float l_148 = l_arr_5[uint(0)];
shared_memory_4[l_147] = l_148;
const uint l_149 = uint(256) + thread_pos_in_threadgroup.x;
const float l_150 = l_arr_5[uint(1)];
shared_memory_4[l_149] = l_150;
const uint l_151 = uint(512) + thread_pos_in_threadgroup.x;
const float l_152 = l_arr_5[uint(2)];
shared_memory_4[l_151] = l_152;
const uint l_153 = uint(768) + thread_pos_in_threadgroup.x;
const float l_154 = l_arr_5[uint(3)];
shared_memory_4[l_153] = l_154;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_155 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_156 = thread_pos_in_threadgroup.x & uint(32);
const bool l_157 = l_156 != uint(0);
const float l_158 = l_arr_5[uint(0)];
const uint l_159 = uint(0) + l_155;
const float l_160 = shared_memory_4[l_159];
const float l_161 = l_158 + l_160;
l_mut_24 = l_161;
if (l_157) {
const float l_162 = l_160 - l_158;
l_mut_24 = l_162;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_163 = l_arr_5[uint(1)];
const uint l_164 = uint(256) + l_155;
const float l_165 = shared_memory_4[l_164];
const float l_166 = l_163 + l_165;
l_mut_24 = l_166;
if (l_157) {
const float l_167 = l_165 - l_163;
l_mut_24 = l_167;
}
l_arr_6[uint(1)] = l_mut_24;
const float l_168 = l_arr_5[uint(2)];
const uint l_169 = uint(512) + l_155;
const float l_170 = shared_memory_4[l_169];
const float l_171 = l_168 + l_170;
l_mut_24 = l_171;
if (l_157) {
const float l_172 = l_170 - l_168;
l_mut_24 = l_172;
}
l_arr_6[uint(2)] = l_mut_24;
const float l_173 = l_arr_5[uint(3)];
const uint l_174 = uint(768) + l_155;
const float l_175 = shared_memory_4[l_174];
const float l_176 = l_173 + l_175;
l_mut_24 = l_176;
if (l_157) {
const float l_177 = l_175 - l_173;
l_mut_24 = l_177;
}
l_arr_6[uint(3)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_178 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_178;
const uint l_179 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_179] = l_178;
const float l_180 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_180;
const uint l_181 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_181] = l_180;
const float l_182 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_182;
const uint l_183 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_183] = l_182;
const float l_184 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_184;
const uint l_185 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_185] = l_184;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_186 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_187 = thread_pos_in_threadgroup.x & uint(64);
const bool l_188 = l_187 != uint(0);
const float l_189 = l_arr_5[uint(0)];
const uint l_190 = uint(0) + l_186;
const float l_191 = shared_memory_4[l_190];
const float l_192 = l_189 + l_191;
l_mut_24 = l_192;
if (l_188) {
const float l_193 = l_191 - l_189;
l_mut_24 = l_193;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_194 = l_arr_5[uint(1)];
const uint l_195 = uint(256) + l_186;
const float l_196 = shared_memory_4[l_195];
const float l_197 = l_194 + l_196;
l_mut_24 = l_197;
if (l_188) {
const float l_198 = l_196 - l_194;
l_mut_24 = l_198;
}
l_arr_6[uint(1)] = l_mut_24;
const float l_199 = l_arr_5[uint(2)];
const uint l_200 = uint(512) + l_186;
const float l_201 = shared_memory_4[l_200];
const float l_202 = l_199 + l_201;
l_mut_24 = l_202;
if (l_188) {
const float l_203 = l_201 - l_199;
l_mut_24 = l_203;
}
l_arr_6[uint(2)] = l_mut_24;
const float l_204 = l_arr_5[uint(3)];
const uint l_205 = uint(768) + l_186;
const float l_206 = shared_memory_4[l_205];
const float l_207 = l_204 + l_206;
l_mut_24 = l_207;
if (l_188) {
const float l_208 = l_206 - l_204;
l_mut_24 = l_208;
}
l_arr_6[uint(3)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_209 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_209;
const uint l_210 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_210] = l_209;
const float l_211 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_211;
const uint l_212 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_212] = l_211;
const float l_213 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_213;
const uint l_214 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_214] = l_213;
const float l_215 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_215;
const uint l_216 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_216] = l_215;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_217 = thread_pos_in_threadgroup.x ^ uint(128);
const uint l_218 = thread_pos_in_threadgroup.x & uint(128);
const bool l_219 = l_218 != uint(0);
const float l_220 = l_arr_5[uint(0)];
const uint l_221 = uint(0) + l_217;
const float l_222 = shared_memory_4[l_221];
const float l_223 = l_220 + l_222;
l_mut_24 = l_223;
if (l_219) {
const float l_224 = l_222 - l_220;
l_mut_24 = l_224;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_225 = l_arr_5[uint(1)];
const uint l_226 = uint(256) + l_217;
const float l_227 = shared_memory_4[l_226];
const float l_228 = l_225 + l_227;
l_mut_24 = l_228;
if (l_219) {
const float l_229 = l_227 - l_225;
l_mut_24 = l_229;
}
l_arr_6[uint(1)] = l_mut_24;
const float l_230 = l_arr_5[uint(2)];
const uint l_231 = uint(512) + l_217;
const float l_232 = shared_memory_4[l_231];
const float l_233 = l_230 + l_232;
l_mut_24 = l_233;
if (l_219) {
const float l_234 = l_232 - l_230;
l_mut_24 = l_234;
}
l_arr_6[uint(2)] = l_mut_24;
const float l_235 = l_arr_5[uint(3)];
const uint l_236 = uint(768) + l_217;
const float l_237 = shared_memory_4[l_236];
const float l_238 = l_235 + l_237;
l_mut_24 = l_238;
if (l_219) {
const float l_239 = l_237 - l_235;
l_mut_24 = l_239;
}
l_arr_6[uint(3)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_240 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_240;
const float l_241 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_241;
const float l_242 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_242;
const float l_243 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_243;
const float l_244 = l_arr_5[uint(0)];
const float l_245 = l_arr_5[uint(1)];
const float l_246 = l_244 + l_245;
l_arr_5[uint(0)] = l_246;
const float l_247 = l_244 - l_245;
l_arr_5[uint(1)] = l_247;
const float l_248 = l_arr_5[uint(2)];
const float l_249 = l_arr_5[uint(3)];
const float l_250 = l_248 + l_249;
l_arr_5[uint(2)] = l_250;
const float l_251 = l_248 - l_249;
l_arr_5[uint(3)] = l_251;
const float l_252 = l_arr_5[uint(0)];
const float l_253 = l_arr_5[uint(2)];
const float l_254 = l_252 + l_253;
l_arr_5[uint(0)] = l_254;
const float l_255 = l_252 - l_253;
l_arr_5[uint(2)] = l_255;
const float l_256 = l_arr_5[uint(1)];
const float l_257 = l_arr_5[uint(3)];
const float l_258 = l_256 + l_257;
l_arr_5[uint(1)] = l_258;
const float l_259 = l_256 - l_257;
l_arr_5[uint(3)] = l_259;
const uint l_260 = uint(0) + thread_pos_in_threadgroup.x;
const float l_261 = l_arr_5[uint(0)];
const float l_262 = l_261 * float(0.03125);
l_mut_24 = l_262;
const bool l_263 = l_2 != uint(0);
const bool l_264 = l_1 != uint(0);
const bool l_265 = l_263 && l_264;
if (l_265) {
const uint l_266 = l_12 + l_260;
const float l_267 = buffer_1[l_266];
l_mut_24 = l_mut_24 * l_267;
}
const uint l_268 = l_10 + l_260;
buffer_2[l_268] = l_mut_24;
const uint l_269 = uint(256) + thread_pos_in_threadgroup.x;
const float l_270 = l_arr_5[uint(1)];
const float l_271 = l_270 * float(0.03125);
l_mut_24 = l_271;
const bool l_272 = l_2 != uint(0);
const bool l_273 = l_1 != uint(0);
const bool l_274 = l_272 && l_273;
if (l_274) {
const uint l_275 = l_12 + l_269;
const float l_276 = buffer_1[l_275];
l_mut_24 = l_mut_24 * l_276;
}
const uint l_277 = l_10 + l_269;
buffer_2[l_277] = l_mut_24;
const uint l_278 = uint(512) + thread_pos_in_threadgroup.x;
const float l_279 = l_arr_5[uint(2)];
const float l_280 = l_279 * float(0.03125);
l_mut_24 = l_280;
const bool l_281 = l_2 != uint(0);
const bool l_282 = l_1 != uint(0);
const bool l_283 = l_281 && l_282;
if (l_283) {
const uint l_284 = l_12 + l_278;
const float l_285 = buffer_1[l_284];
l_mut_24 = l_mut_24 * l_285;
}
const uint l_286 = l_10 + l_278;
buffer_2[l_286] = l_mut_24;
const uint l_287 = uint(768) + thread_pos_in_threadgroup.x;
const float l_288 = l_arr_5[uint(3)];
const float l_289 = l_288 * float(0.03125);
l_mut_24 = l_289;
const bool l_290 = l_2 != uint(0);
const bool l_291 = l_1 != uint(0);
const bool l_292 = l_290 && l_291;
if (l_292) {
const uint l_293 = l_12 + l_287;
const float l_294 = buffer_1[l_293];
l_mut_24 = l_mut_24 * l_294;
}
const uint l_295 = l_10 + l_287;
buffer_2[l_295] = l_mut_24;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
