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
threadgroup uchar dynamic_shared_mem[16384];
// Shared array size: 4096, 16384 bytes
threadgroup float* shared_memory_4 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
float l_arr_5[16];

float l_arr_6[16];

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
const uint l_10 = l_mut_7 * uint(4096);
const uint l_11 = l_mut_7 % l_0;
const uint l_12 = l_11 * uint(4096);
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
const uint l_147 = uint(1024) + thread_pos_in_threadgroup.x;
const uint l_148 = l_10 + l_147;
const float l_149 = buffer_0[l_148];
l_mut_24 = l_149;
const bool l_150 = l_2 == uint(0);
const bool l_151 = l_1 != uint(0);
const bool l_152 = l_150 && l_151;
if (l_152) {
const uint l_153 = l_12 + l_147;
const float l_154 = buffer_1[l_153];
l_mut_24 = l_mut_24 * l_154;
}
const float l_155 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_156 = l_mut_24 + l_155;
l_mut_16 = l_156;
const uint l_157 = thread_pos_in_threadgroup.x & uint(1);
const bool l_158 = l_157 != uint(0);
if (l_158) {
const float l_159 = l_155 - l_mut_24;
l_mut_16 = l_159;
}
l_mut_24 = l_mut_16;
const float l_160 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_161 = l_mut_24 + l_160;
l_mut_16 = l_161;
const uint l_162 = thread_pos_in_threadgroup.x & uint(2);
const bool l_163 = l_162 != uint(0);
if (l_163) {
const float l_164 = l_160 - l_mut_24;
l_mut_16 = l_164;
}
l_mut_24 = l_mut_16;
const float l_165 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_166 = l_mut_24 + l_165;
l_mut_16 = l_166;
const uint l_167 = thread_pos_in_threadgroup.x & uint(4);
const bool l_168 = l_167 != uint(0);
if (l_168) {
const float l_169 = l_165 - l_mut_24;
l_mut_16 = l_169;
}
l_mut_24 = l_mut_16;
const float l_170 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_171 = l_mut_24 + l_170;
l_mut_16 = l_171;
const uint l_172 = thread_pos_in_threadgroup.x & uint(8);
const bool l_173 = l_172 != uint(0);
if (l_173) {
const float l_174 = l_170 - l_mut_24;
l_mut_16 = l_174;
}
l_mut_24 = l_mut_16;
const float l_175 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_176 = l_mut_24 + l_175;
l_mut_16 = l_176;
const uint l_177 = thread_pos_in_threadgroup.x & uint(16);
const bool l_178 = l_177 != uint(0);
if (l_178) {
const float l_179 = l_175 - l_mut_24;
l_mut_16 = l_179;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(4)] = l_mut_24;
const uint l_180 = uint(1280) + thread_pos_in_threadgroup.x;
const uint l_181 = l_10 + l_180;
const float l_182 = buffer_0[l_181];
l_mut_24 = l_182;
const bool l_183 = l_2 == uint(0);
const bool l_184 = l_1 != uint(0);
const bool l_185 = l_183 && l_184;
if (l_185) {
const uint l_186 = l_12 + l_180;
const float l_187 = buffer_1[l_186];
l_mut_24 = l_mut_24 * l_187;
}
const float l_188 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_189 = l_mut_24 + l_188;
l_mut_16 = l_189;
const uint l_190 = thread_pos_in_threadgroup.x & uint(1);
const bool l_191 = l_190 != uint(0);
if (l_191) {
const float l_192 = l_188 - l_mut_24;
l_mut_16 = l_192;
}
l_mut_24 = l_mut_16;
const float l_193 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_194 = l_mut_24 + l_193;
l_mut_16 = l_194;
const uint l_195 = thread_pos_in_threadgroup.x & uint(2);
const bool l_196 = l_195 != uint(0);
if (l_196) {
const float l_197 = l_193 - l_mut_24;
l_mut_16 = l_197;
}
l_mut_24 = l_mut_16;
const float l_198 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_199 = l_mut_24 + l_198;
l_mut_16 = l_199;
const uint l_200 = thread_pos_in_threadgroup.x & uint(4);
const bool l_201 = l_200 != uint(0);
if (l_201) {
const float l_202 = l_198 - l_mut_24;
l_mut_16 = l_202;
}
l_mut_24 = l_mut_16;
const float l_203 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_204 = l_mut_24 + l_203;
l_mut_16 = l_204;
const uint l_205 = thread_pos_in_threadgroup.x & uint(8);
const bool l_206 = l_205 != uint(0);
if (l_206) {
const float l_207 = l_203 - l_mut_24;
l_mut_16 = l_207;
}
l_mut_24 = l_mut_16;
const float l_208 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_209 = l_mut_24 + l_208;
l_mut_16 = l_209;
const uint l_210 = thread_pos_in_threadgroup.x & uint(16);
const bool l_211 = l_210 != uint(0);
if (l_211) {
const float l_212 = l_208 - l_mut_24;
l_mut_16 = l_212;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(5)] = l_mut_24;
const uint l_213 = uint(1536) + thread_pos_in_threadgroup.x;
const uint l_214 = l_10 + l_213;
const float l_215 = buffer_0[l_214];
l_mut_24 = l_215;
const bool l_216 = l_2 == uint(0);
const bool l_217 = l_1 != uint(0);
const bool l_218 = l_216 && l_217;
if (l_218) {
const uint l_219 = l_12 + l_213;
const float l_220 = buffer_1[l_219];
l_mut_24 = l_mut_24 * l_220;
}
const float l_221 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_222 = l_mut_24 + l_221;
l_mut_16 = l_222;
const uint l_223 = thread_pos_in_threadgroup.x & uint(1);
const bool l_224 = l_223 != uint(0);
if (l_224) {
const float l_225 = l_221 - l_mut_24;
l_mut_16 = l_225;
}
l_mut_24 = l_mut_16;
const float l_226 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_227 = l_mut_24 + l_226;
l_mut_16 = l_227;
const uint l_228 = thread_pos_in_threadgroup.x & uint(2);
const bool l_229 = l_228 != uint(0);
if (l_229) {
const float l_230 = l_226 - l_mut_24;
l_mut_16 = l_230;
}
l_mut_24 = l_mut_16;
const float l_231 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_232 = l_mut_24 + l_231;
l_mut_16 = l_232;
const uint l_233 = thread_pos_in_threadgroup.x & uint(4);
const bool l_234 = l_233 != uint(0);
if (l_234) {
const float l_235 = l_231 - l_mut_24;
l_mut_16 = l_235;
}
l_mut_24 = l_mut_16;
const float l_236 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_237 = l_mut_24 + l_236;
l_mut_16 = l_237;
const uint l_238 = thread_pos_in_threadgroup.x & uint(8);
const bool l_239 = l_238 != uint(0);
if (l_239) {
const float l_240 = l_236 - l_mut_24;
l_mut_16 = l_240;
}
l_mut_24 = l_mut_16;
const float l_241 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_242 = l_mut_24 + l_241;
l_mut_16 = l_242;
const uint l_243 = thread_pos_in_threadgroup.x & uint(16);
const bool l_244 = l_243 != uint(0);
if (l_244) {
const float l_245 = l_241 - l_mut_24;
l_mut_16 = l_245;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(6)] = l_mut_24;
const uint l_246 = uint(1792) + thread_pos_in_threadgroup.x;
const uint l_247 = l_10 + l_246;
const float l_248 = buffer_0[l_247];
l_mut_24 = l_248;
const bool l_249 = l_2 == uint(0);
const bool l_250 = l_1 != uint(0);
const bool l_251 = l_249 && l_250;
if (l_251) {
const uint l_252 = l_12 + l_246;
const float l_253 = buffer_1[l_252];
l_mut_24 = l_mut_24 * l_253;
}
const float l_254 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_255 = l_mut_24 + l_254;
l_mut_16 = l_255;
const uint l_256 = thread_pos_in_threadgroup.x & uint(1);
const bool l_257 = l_256 != uint(0);
if (l_257) {
const float l_258 = l_254 - l_mut_24;
l_mut_16 = l_258;
}
l_mut_24 = l_mut_16;
const float l_259 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_260 = l_mut_24 + l_259;
l_mut_16 = l_260;
const uint l_261 = thread_pos_in_threadgroup.x & uint(2);
const bool l_262 = l_261 != uint(0);
if (l_262) {
const float l_263 = l_259 - l_mut_24;
l_mut_16 = l_263;
}
l_mut_24 = l_mut_16;
const float l_264 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_265 = l_mut_24 + l_264;
l_mut_16 = l_265;
const uint l_266 = thread_pos_in_threadgroup.x & uint(4);
const bool l_267 = l_266 != uint(0);
if (l_267) {
const float l_268 = l_264 - l_mut_24;
l_mut_16 = l_268;
}
l_mut_24 = l_mut_16;
const float l_269 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_270 = l_mut_24 + l_269;
l_mut_16 = l_270;
const uint l_271 = thread_pos_in_threadgroup.x & uint(8);
const bool l_272 = l_271 != uint(0);
if (l_272) {
const float l_273 = l_269 - l_mut_24;
l_mut_16 = l_273;
}
l_mut_24 = l_mut_16;
const float l_274 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_275 = l_mut_24 + l_274;
l_mut_16 = l_275;
const uint l_276 = thread_pos_in_threadgroup.x & uint(16);
const bool l_277 = l_276 != uint(0);
if (l_277) {
const float l_278 = l_274 - l_mut_24;
l_mut_16 = l_278;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(7)] = l_mut_24;
const uint l_279 = uint(2048) + thread_pos_in_threadgroup.x;
const uint l_280 = l_10 + l_279;
const float l_281 = buffer_0[l_280];
l_mut_24 = l_281;
const bool l_282 = l_2 == uint(0);
const bool l_283 = l_1 != uint(0);
const bool l_284 = l_282 && l_283;
if (l_284) {
const uint l_285 = l_12 + l_279;
const float l_286 = buffer_1[l_285];
l_mut_24 = l_mut_24 * l_286;
}
const float l_287 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_288 = l_mut_24 + l_287;
l_mut_16 = l_288;
const uint l_289 = thread_pos_in_threadgroup.x & uint(1);
const bool l_290 = l_289 != uint(0);
if (l_290) {
const float l_291 = l_287 - l_mut_24;
l_mut_16 = l_291;
}
l_mut_24 = l_mut_16;
const float l_292 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_293 = l_mut_24 + l_292;
l_mut_16 = l_293;
const uint l_294 = thread_pos_in_threadgroup.x & uint(2);
const bool l_295 = l_294 != uint(0);
if (l_295) {
const float l_296 = l_292 - l_mut_24;
l_mut_16 = l_296;
}
l_mut_24 = l_mut_16;
const float l_297 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_298 = l_mut_24 + l_297;
l_mut_16 = l_298;
const uint l_299 = thread_pos_in_threadgroup.x & uint(4);
const bool l_300 = l_299 != uint(0);
if (l_300) {
const float l_301 = l_297 - l_mut_24;
l_mut_16 = l_301;
}
l_mut_24 = l_mut_16;
const float l_302 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_303 = l_mut_24 + l_302;
l_mut_16 = l_303;
const uint l_304 = thread_pos_in_threadgroup.x & uint(8);
const bool l_305 = l_304 != uint(0);
if (l_305) {
const float l_306 = l_302 - l_mut_24;
l_mut_16 = l_306;
}
l_mut_24 = l_mut_16;
const float l_307 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_308 = l_mut_24 + l_307;
l_mut_16 = l_308;
const uint l_309 = thread_pos_in_threadgroup.x & uint(16);
const bool l_310 = l_309 != uint(0);
if (l_310) {
const float l_311 = l_307 - l_mut_24;
l_mut_16 = l_311;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(8)] = l_mut_24;
const uint l_312 = uint(2304) + thread_pos_in_threadgroup.x;
const uint l_313 = l_10 + l_312;
const float l_314 = buffer_0[l_313];
l_mut_24 = l_314;
const bool l_315 = l_2 == uint(0);
const bool l_316 = l_1 != uint(0);
const bool l_317 = l_315 && l_316;
if (l_317) {
const uint l_318 = l_12 + l_312;
const float l_319 = buffer_1[l_318];
l_mut_24 = l_mut_24 * l_319;
}
const float l_320 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_321 = l_mut_24 + l_320;
l_mut_16 = l_321;
const uint l_322 = thread_pos_in_threadgroup.x & uint(1);
const bool l_323 = l_322 != uint(0);
if (l_323) {
const float l_324 = l_320 - l_mut_24;
l_mut_16 = l_324;
}
l_mut_24 = l_mut_16;
const float l_325 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_326 = l_mut_24 + l_325;
l_mut_16 = l_326;
const uint l_327 = thread_pos_in_threadgroup.x & uint(2);
const bool l_328 = l_327 != uint(0);
if (l_328) {
const float l_329 = l_325 - l_mut_24;
l_mut_16 = l_329;
}
l_mut_24 = l_mut_16;
const float l_330 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_331 = l_mut_24 + l_330;
l_mut_16 = l_331;
const uint l_332 = thread_pos_in_threadgroup.x & uint(4);
const bool l_333 = l_332 != uint(0);
if (l_333) {
const float l_334 = l_330 - l_mut_24;
l_mut_16 = l_334;
}
l_mut_24 = l_mut_16;
const float l_335 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_336 = l_mut_24 + l_335;
l_mut_16 = l_336;
const uint l_337 = thread_pos_in_threadgroup.x & uint(8);
const bool l_338 = l_337 != uint(0);
if (l_338) {
const float l_339 = l_335 - l_mut_24;
l_mut_16 = l_339;
}
l_mut_24 = l_mut_16;
const float l_340 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_341 = l_mut_24 + l_340;
l_mut_16 = l_341;
const uint l_342 = thread_pos_in_threadgroup.x & uint(16);
const bool l_343 = l_342 != uint(0);
if (l_343) {
const float l_344 = l_340 - l_mut_24;
l_mut_16 = l_344;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(9)] = l_mut_24;
const uint l_345 = uint(2560) + thread_pos_in_threadgroup.x;
const uint l_346 = l_10 + l_345;
const float l_347 = buffer_0[l_346];
l_mut_24 = l_347;
const bool l_348 = l_2 == uint(0);
const bool l_349 = l_1 != uint(0);
const bool l_350 = l_348 && l_349;
if (l_350) {
const uint l_351 = l_12 + l_345;
const float l_352 = buffer_1[l_351];
l_mut_24 = l_mut_24 * l_352;
}
const float l_353 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_354 = l_mut_24 + l_353;
l_mut_16 = l_354;
const uint l_355 = thread_pos_in_threadgroup.x & uint(1);
const bool l_356 = l_355 != uint(0);
if (l_356) {
const float l_357 = l_353 - l_mut_24;
l_mut_16 = l_357;
}
l_mut_24 = l_mut_16;
const float l_358 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_359 = l_mut_24 + l_358;
l_mut_16 = l_359;
const uint l_360 = thread_pos_in_threadgroup.x & uint(2);
const bool l_361 = l_360 != uint(0);
if (l_361) {
const float l_362 = l_358 - l_mut_24;
l_mut_16 = l_362;
}
l_mut_24 = l_mut_16;
const float l_363 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_364 = l_mut_24 + l_363;
l_mut_16 = l_364;
const uint l_365 = thread_pos_in_threadgroup.x & uint(4);
const bool l_366 = l_365 != uint(0);
if (l_366) {
const float l_367 = l_363 - l_mut_24;
l_mut_16 = l_367;
}
l_mut_24 = l_mut_16;
const float l_368 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_369 = l_mut_24 + l_368;
l_mut_16 = l_369;
const uint l_370 = thread_pos_in_threadgroup.x & uint(8);
const bool l_371 = l_370 != uint(0);
if (l_371) {
const float l_372 = l_368 - l_mut_24;
l_mut_16 = l_372;
}
l_mut_24 = l_mut_16;
const float l_373 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_374 = l_mut_24 + l_373;
l_mut_16 = l_374;
const uint l_375 = thread_pos_in_threadgroup.x & uint(16);
const bool l_376 = l_375 != uint(0);
if (l_376) {
const float l_377 = l_373 - l_mut_24;
l_mut_16 = l_377;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(10)] = l_mut_24;
const uint l_378 = uint(2816) + thread_pos_in_threadgroup.x;
const uint l_379 = l_10 + l_378;
const float l_380 = buffer_0[l_379];
l_mut_24 = l_380;
const bool l_381 = l_2 == uint(0);
const bool l_382 = l_1 != uint(0);
const bool l_383 = l_381 && l_382;
if (l_383) {
const uint l_384 = l_12 + l_378;
const float l_385 = buffer_1[l_384];
l_mut_24 = l_mut_24 * l_385;
}
const float l_386 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_387 = l_mut_24 + l_386;
l_mut_16 = l_387;
const uint l_388 = thread_pos_in_threadgroup.x & uint(1);
const bool l_389 = l_388 != uint(0);
if (l_389) {
const float l_390 = l_386 - l_mut_24;
l_mut_16 = l_390;
}
l_mut_24 = l_mut_16;
const float l_391 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_392 = l_mut_24 + l_391;
l_mut_16 = l_392;
const uint l_393 = thread_pos_in_threadgroup.x & uint(2);
const bool l_394 = l_393 != uint(0);
if (l_394) {
const float l_395 = l_391 - l_mut_24;
l_mut_16 = l_395;
}
l_mut_24 = l_mut_16;
const float l_396 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_397 = l_mut_24 + l_396;
l_mut_16 = l_397;
const uint l_398 = thread_pos_in_threadgroup.x & uint(4);
const bool l_399 = l_398 != uint(0);
if (l_399) {
const float l_400 = l_396 - l_mut_24;
l_mut_16 = l_400;
}
l_mut_24 = l_mut_16;
const float l_401 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_402 = l_mut_24 + l_401;
l_mut_16 = l_402;
const uint l_403 = thread_pos_in_threadgroup.x & uint(8);
const bool l_404 = l_403 != uint(0);
if (l_404) {
const float l_405 = l_401 - l_mut_24;
l_mut_16 = l_405;
}
l_mut_24 = l_mut_16;
const float l_406 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_407 = l_mut_24 + l_406;
l_mut_16 = l_407;
const uint l_408 = thread_pos_in_threadgroup.x & uint(16);
const bool l_409 = l_408 != uint(0);
if (l_409) {
const float l_410 = l_406 - l_mut_24;
l_mut_16 = l_410;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(11)] = l_mut_24;
const uint l_411 = uint(3072) + thread_pos_in_threadgroup.x;
const uint l_412 = l_10 + l_411;
const float l_413 = buffer_0[l_412];
l_mut_24 = l_413;
const bool l_414 = l_2 == uint(0);
const bool l_415 = l_1 != uint(0);
const bool l_416 = l_414 && l_415;
if (l_416) {
const uint l_417 = l_12 + l_411;
const float l_418 = buffer_1[l_417];
l_mut_24 = l_mut_24 * l_418;
}
const float l_419 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_420 = l_mut_24 + l_419;
l_mut_16 = l_420;
const uint l_421 = thread_pos_in_threadgroup.x & uint(1);
const bool l_422 = l_421 != uint(0);
if (l_422) {
const float l_423 = l_419 - l_mut_24;
l_mut_16 = l_423;
}
l_mut_24 = l_mut_16;
const float l_424 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_425 = l_mut_24 + l_424;
l_mut_16 = l_425;
const uint l_426 = thread_pos_in_threadgroup.x & uint(2);
const bool l_427 = l_426 != uint(0);
if (l_427) {
const float l_428 = l_424 - l_mut_24;
l_mut_16 = l_428;
}
l_mut_24 = l_mut_16;
const float l_429 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_430 = l_mut_24 + l_429;
l_mut_16 = l_430;
const uint l_431 = thread_pos_in_threadgroup.x & uint(4);
const bool l_432 = l_431 != uint(0);
if (l_432) {
const float l_433 = l_429 - l_mut_24;
l_mut_16 = l_433;
}
l_mut_24 = l_mut_16;
const float l_434 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_435 = l_mut_24 + l_434;
l_mut_16 = l_435;
const uint l_436 = thread_pos_in_threadgroup.x & uint(8);
const bool l_437 = l_436 != uint(0);
if (l_437) {
const float l_438 = l_434 - l_mut_24;
l_mut_16 = l_438;
}
l_mut_24 = l_mut_16;
const float l_439 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_440 = l_mut_24 + l_439;
l_mut_16 = l_440;
const uint l_441 = thread_pos_in_threadgroup.x & uint(16);
const bool l_442 = l_441 != uint(0);
if (l_442) {
const float l_443 = l_439 - l_mut_24;
l_mut_16 = l_443;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(12)] = l_mut_24;
const uint l_444 = uint(3328) + thread_pos_in_threadgroup.x;
const uint l_445 = l_10 + l_444;
const float l_446 = buffer_0[l_445];
l_mut_24 = l_446;
const bool l_447 = l_2 == uint(0);
const bool l_448 = l_1 != uint(0);
const bool l_449 = l_447 && l_448;
if (l_449) {
const uint l_450 = l_12 + l_444;
const float l_451 = buffer_1[l_450];
l_mut_24 = l_mut_24 * l_451;
}
const float l_452 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_453 = l_mut_24 + l_452;
l_mut_16 = l_453;
const uint l_454 = thread_pos_in_threadgroup.x & uint(1);
const bool l_455 = l_454 != uint(0);
if (l_455) {
const float l_456 = l_452 - l_mut_24;
l_mut_16 = l_456;
}
l_mut_24 = l_mut_16;
const float l_457 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_458 = l_mut_24 + l_457;
l_mut_16 = l_458;
const uint l_459 = thread_pos_in_threadgroup.x & uint(2);
const bool l_460 = l_459 != uint(0);
if (l_460) {
const float l_461 = l_457 - l_mut_24;
l_mut_16 = l_461;
}
l_mut_24 = l_mut_16;
const float l_462 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_463 = l_mut_24 + l_462;
l_mut_16 = l_463;
const uint l_464 = thread_pos_in_threadgroup.x & uint(4);
const bool l_465 = l_464 != uint(0);
if (l_465) {
const float l_466 = l_462 - l_mut_24;
l_mut_16 = l_466;
}
l_mut_24 = l_mut_16;
const float l_467 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_468 = l_mut_24 + l_467;
l_mut_16 = l_468;
const uint l_469 = thread_pos_in_threadgroup.x & uint(8);
const bool l_470 = l_469 != uint(0);
if (l_470) {
const float l_471 = l_467 - l_mut_24;
l_mut_16 = l_471;
}
l_mut_24 = l_mut_16;
const float l_472 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_473 = l_mut_24 + l_472;
l_mut_16 = l_473;
const uint l_474 = thread_pos_in_threadgroup.x & uint(16);
const bool l_475 = l_474 != uint(0);
if (l_475) {
const float l_476 = l_472 - l_mut_24;
l_mut_16 = l_476;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(13)] = l_mut_24;
const uint l_477 = uint(3584) + thread_pos_in_threadgroup.x;
const uint l_478 = l_10 + l_477;
const float l_479 = buffer_0[l_478];
l_mut_24 = l_479;
const bool l_480 = l_2 == uint(0);
const bool l_481 = l_1 != uint(0);
const bool l_482 = l_480 && l_481;
if (l_482) {
const uint l_483 = l_12 + l_477;
const float l_484 = buffer_1[l_483];
l_mut_24 = l_mut_24 * l_484;
}
const float l_485 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_486 = l_mut_24 + l_485;
l_mut_16 = l_486;
const uint l_487 = thread_pos_in_threadgroup.x & uint(1);
const bool l_488 = l_487 != uint(0);
if (l_488) {
const float l_489 = l_485 - l_mut_24;
l_mut_16 = l_489;
}
l_mut_24 = l_mut_16;
const float l_490 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_491 = l_mut_24 + l_490;
l_mut_16 = l_491;
const uint l_492 = thread_pos_in_threadgroup.x & uint(2);
const bool l_493 = l_492 != uint(0);
if (l_493) {
const float l_494 = l_490 - l_mut_24;
l_mut_16 = l_494;
}
l_mut_24 = l_mut_16;
const float l_495 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_496 = l_mut_24 + l_495;
l_mut_16 = l_496;
const uint l_497 = thread_pos_in_threadgroup.x & uint(4);
const bool l_498 = l_497 != uint(0);
if (l_498) {
const float l_499 = l_495 - l_mut_24;
l_mut_16 = l_499;
}
l_mut_24 = l_mut_16;
const float l_500 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_501 = l_mut_24 + l_500;
l_mut_16 = l_501;
const uint l_502 = thread_pos_in_threadgroup.x & uint(8);
const bool l_503 = l_502 != uint(0);
if (l_503) {
const float l_504 = l_500 - l_mut_24;
l_mut_16 = l_504;
}
l_mut_24 = l_mut_16;
const float l_505 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_506 = l_mut_24 + l_505;
l_mut_16 = l_506;
const uint l_507 = thread_pos_in_threadgroup.x & uint(16);
const bool l_508 = l_507 != uint(0);
if (l_508) {
const float l_509 = l_505 - l_mut_24;
l_mut_16 = l_509;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(14)] = l_mut_24;
const uint l_510 = uint(3840) + thread_pos_in_threadgroup.x;
const uint l_511 = l_10 + l_510;
const float l_512 = buffer_0[l_511];
l_mut_24 = l_512;
const bool l_513 = l_2 == uint(0);
const bool l_514 = l_1 != uint(0);
const bool l_515 = l_513 && l_514;
if (l_515) {
const uint l_516 = l_12 + l_510;
const float l_517 = buffer_1[l_516];
l_mut_24 = l_mut_24 * l_517;
}
const float l_518 = { simd_shuffle_xor(l_mut_24, uint(1)) };
const float l_519 = l_mut_24 + l_518;
l_mut_16 = l_519;
const uint l_520 = thread_pos_in_threadgroup.x & uint(1);
const bool l_521 = l_520 != uint(0);
if (l_521) {
const float l_522 = l_518 - l_mut_24;
l_mut_16 = l_522;
}
l_mut_24 = l_mut_16;
const float l_523 = { simd_shuffle_xor(l_mut_24, uint(2)) };
const float l_524 = l_mut_24 + l_523;
l_mut_16 = l_524;
const uint l_525 = thread_pos_in_threadgroup.x & uint(2);
const bool l_526 = l_525 != uint(0);
if (l_526) {
const float l_527 = l_523 - l_mut_24;
l_mut_16 = l_527;
}
l_mut_24 = l_mut_16;
const float l_528 = { simd_shuffle_xor(l_mut_24, uint(4)) };
const float l_529 = l_mut_24 + l_528;
l_mut_16 = l_529;
const uint l_530 = thread_pos_in_threadgroup.x & uint(4);
const bool l_531 = l_530 != uint(0);
if (l_531) {
const float l_532 = l_528 - l_mut_24;
l_mut_16 = l_532;
}
l_mut_24 = l_mut_16;
const float l_533 = { simd_shuffle_xor(l_mut_24, uint(8)) };
const float l_534 = l_mut_24 + l_533;
l_mut_16 = l_534;
const uint l_535 = thread_pos_in_threadgroup.x & uint(8);
const bool l_536 = l_535 != uint(0);
if (l_536) {
const float l_537 = l_533 - l_mut_24;
l_mut_16 = l_537;
}
l_mut_24 = l_mut_16;
const float l_538 = { simd_shuffle_xor(l_mut_24, uint(16)) };
const float l_539 = l_mut_24 + l_538;
l_mut_16 = l_539;
const uint l_540 = thread_pos_in_threadgroup.x & uint(16);
const bool l_541 = l_540 != uint(0);
if (l_541) {
const float l_542 = l_538 - l_mut_24;
l_mut_16 = l_542;
}
l_mut_24 = l_mut_16;
l_arr_5[uint(15)] = l_mut_24;
const uint l_543 = uint(0) + thread_pos_in_threadgroup.x;
const float l_544 = l_arr_5[uint(0)];
shared_memory_4[l_543] = l_544;
const uint l_545 = uint(256) + thread_pos_in_threadgroup.x;
const float l_546 = l_arr_5[uint(1)];
shared_memory_4[l_545] = l_546;
const uint l_547 = uint(512) + thread_pos_in_threadgroup.x;
const float l_548 = l_arr_5[uint(2)];
shared_memory_4[l_547] = l_548;
const uint l_549 = uint(768) + thread_pos_in_threadgroup.x;
const float l_550 = l_arr_5[uint(3)];
shared_memory_4[l_549] = l_550;
const uint l_551 = uint(1024) + thread_pos_in_threadgroup.x;
const float l_552 = l_arr_5[uint(4)];
shared_memory_4[l_551] = l_552;
const uint l_553 = uint(1280) + thread_pos_in_threadgroup.x;
const float l_554 = l_arr_5[uint(5)];
shared_memory_4[l_553] = l_554;
const uint l_555 = uint(1536) + thread_pos_in_threadgroup.x;
const float l_556 = l_arr_5[uint(6)];
shared_memory_4[l_555] = l_556;
const uint l_557 = uint(1792) + thread_pos_in_threadgroup.x;
const float l_558 = l_arr_5[uint(7)];
shared_memory_4[l_557] = l_558;
const uint l_559 = uint(2048) + thread_pos_in_threadgroup.x;
const float l_560 = l_arr_5[uint(8)];
shared_memory_4[l_559] = l_560;
const uint l_561 = uint(2304) + thread_pos_in_threadgroup.x;
const float l_562 = l_arr_5[uint(9)];
shared_memory_4[l_561] = l_562;
const uint l_563 = uint(2560) + thread_pos_in_threadgroup.x;
const float l_564 = l_arr_5[uint(10)];
shared_memory_4[l_563] = l_564;
const uint l_565 = uint(2816) + thread_pos_in_threadgroup.x;
const float l_566 = l_arr_5[uint(11)];
shared_memory_4[l_565] = l_566;
const uint l_567 = uint(3072) + thread_pos_in_threadgroup.x;
const float l_568 = l_arr_5[uint(12)];
shared_memory_4[l_567] = l_568;
const uint l_569 = uint(3328) + thread_pos_in_threadgroup.x;
const float l_570 = l_arr_5[uint(13)];
shared_memory_4[l_569] = l_570;
const uint l_571 = uint(3584) + thread_pos_in_threadgroup.x;
const float l_572 = l_arr_5[uint(14)];
shared_memory_4[l_571] = l_572;
const uint l_573 = uint(3840) + thread_pos_in_threadgroup.x;
const float l_574 = l_arr_5[uint(15)];
shared_memory_4[l_573] = l_574;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_575 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_576 = thread_pos_in_threadgroup.x & uint(32);
const bool l_577 = l_576 != uint(0);
const float l_578 = l_arr_5[uint(0)];
const uint l_579 = uint(0) + l_575;
const float l_580 = shared_memory_4[l_579];
const float l_581 = l_578 + l_580;
l_mut_24 = l_581;
if (l_577) {
const float l_582 = l_580 - l_578;
l_mut_24 = l_582;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_583 = l_arr_5[uint(1)];
const uint l_584 = uint(256) + l_575;
const float l_585 = shared_memory_4[l_584];
const float l_586 = l_583 + l_585;
l_mut_24 = l_586;
if (l_577) {
const float l_587 = l_585 - l_583;
l_mut_24 = l_587;
}
l_arr_6[uint(1)] = l_mut_24;
const float l_588 = l_arr_5[uint(2)];
const uint l_589 = uint(512) + l_575;
const float l_590 = shared_memory_4[l_589];
const float l_591 = l_588 + l_590;
l_mut_24 = l_591;
if (l_577) {
const float l_592 = l_590 - l_588;
l_mut_24 = l_592;
}
l_arr_6[uint(2)] = l_mut_24;
const float l_593 = l_arr_5[uint(3)];
const uint l_594 = uint(768) + l_575;
const float l_595 = shared_memory_4[l_594];
const float l_596 = l_593 + l_595;
l_mut_24 = l_596;
if (l_577) {
const float l_597 = l_595 - l_593;
l_mut_24 = l_597;
}
l_arr_6[uint(3)] = l_mut_24;
const float l_598 = l_arr_5[uint(4)];
const uint l_599 = uint(1024) + l_575;
const float l_600 = shared_memory_4[l_599];
const float l_601 = l_598 + l_600;
l_mut_24 = l_601;
if (l_577) {
const float l_602 = l_600 - l_598;
l_mut_24 = l_602;
}
l_arr_6[uint(4)] = l_mut_24;
const float l_603 = l_arr_5[uint(5)];
const uint l_604 = uint(1280) + l_575;
const float l_605 = shared_memory_4[l_604];
const float l_606 = l_603 + l_605;
l_mut_24 = l_606;
if (l_577) {
const float l_607 = l_605 - l_603;
l_mut_24 = l_607;
}
l_arr_6[uint(5)] = l_mut_24;
const float l_608 = l_arr_5[uint(6)];
const uint l_609 = uint(1536) + l_575;
const float l_610 = shared_memory_4[l_609];
const float l_611 = l_608 + l_610;
l_mut_24 = l_611;
if (l_577) {
const float l_612 = l_610 - l_608;
l_mut_24 = l_612;
}
l_arr_6[uint(6)] = l_mut_24;
const float l_613 = l_arr_5[uint(7)];
const uint l_614 = uint(1792) + l_575;
const float l_615 = shared_memory_4[l_614];
const float l_616 = l_613 + l_615;
l_mut_24 = l_616;
if (l_577) {
const float l_617 = l_615 - l_613;
l_mut_24 = l_617;
}
l_arr_6[uint(7)] = l_mut_24;
const float l_618 = l_arr_5[uint(8)];
const uint l_619 = uint(2048) + l_575;
const float l_620 = shared_memory_4[l_619];
const float l_621 = l_618 + l_620;
l_mut_24 = l_621;
if (l_577) {
const float l_622 = l_620 - l_618;
l_mut_24 = l_622;
}
l_arr_6[uint(8)] = l_mut_24;
const float l_623 = l_arr_5[uint(9)];
const uint l_624 = uint(2304) + l_575;
const float l_625 = shared_memory_4[l_624];
const float l_626 = l_623 + l_625;
l_mut_24 = l_626;
if (l_577) {
const float l_627 = l_625 - l_623;
l_mut_24 = l_627;
}
l_arr_6[uint(9)] = l_mut_24;
const float l_628 = l_arr_5[uint(10)];
const uint l_629 = uint(2560) + l_575;
const float l_630 = shared_memory_4[l_629];
const float l_631 = l_628 + l_630;
l_mut_24 = l_631;
if (l_577) {
const float l_632 = l_630 - l_628;
l_mut_24 = l_632;
}
l_arr_6[uint(10)] = l_mut_24;
const float l_633 = l_arr_5[uint(11)];
const uint l_634 = uint(2816) + l_575;
const float l_635 = shared_memory_4[l_634];
const float l_636 = l_633 + l_635;
l_mut_24 = l_636;
if (l_577) {
const float l_637 = l_635 - l_633;
l_mut_24 = l_637;
}
l_arr_6[uint(11)] = l_mut_24;
const float l_638 = l_arr_5[uint(12)];
const uint l_639 = uint(3072) + l_575;
const float l_640 = shared_memory_4[l_639];
const float l_641 = l_638 + l_640;
l_mut_24 = l_641;
if (l_577) {
const float l_642 = l_640 - l_638;
l_mut_24 = l_642;
}
l_arr_6[uint(12)] = l_mut_24;
const float l_643 = l_arr_5[uint(13)];
const uint l_644 = uint(3328) + l_575;
const float l_645 = shared_memory_4[l_644];
const float l_646 = l_643 + l_645;
l_mut_24 = l_646;
if (l_577) {
const float l_647 = l_645 - l_643;
l_mut_24 = l_647;
}
l_arr_6[uint(13)] = l_mut_24;
const float l_648 = l_arr_5[uint(14)];
const uint l_649 = uint(3584) + l_575;
const float l_650 = shared_memory_4[l_649];
const float l_651 = l_648 + l_650;
l_mut_24 = l_651;
if (l_577) {
const float l_652 = l_650 - l_648;
l_mut_24 = l_652;
}
l_arr_6[uint(14)] = l_mut_24;
const float l_653 = l_arr_5[uint(15)];
const uint l_654 = uint(3840) + l_575;
const float l_655 = shared_memory_4[l_654];
const float l_656 = l_653 + l_655;
l_mut_24 = l_656;
if (l_577) {
const float l_657 = l_655 - l_653;
l_mut_24 = l_657;
}
l_arr_6[uint(15)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_658 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_658;
const uint l_659 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_659] = l_658;
const float l_660 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_660;
const uint l_661 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_661] = l_660;
const float l_662 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_662;
const uint l_663 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_663] = l_662;
const float l_664 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_664;
const uint l_665 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_665] = l_664;
const float l_666 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_666;
const uint l_667 = uint(1024) + thread_pos_in_threadgroup.x;
shared_memory_4[l_667] = l_666;
const float l_668 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_668;
const uint l_669 = uint(1280) + thread_pos_in_threadgroup.x;
shared_memory_4[l_669] = l_668;
const float l_670 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_670;
const uint l_671 = uint(1536) + thread_pos_in_threadgroup.x;
shared_memory_4[l_671] = l_670;
const float l_672 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_672;
const uint l_673 = uint(1792) + thread_pos_in_threadgroup.x;
shared_memory_4[l_673] = l_672;
const float l_674 = l_arr_6[uint(8)];
l_arr_5[uint(8)] = l_674;
const uint l_675 = uint(2048) + thread_pos_in_threadgroup.x;
shared_memory_4[l_675] = l_674;
const float l_676 = l_arr_6[uint(9)];
l_arr_5[uint(9)] = l_676;
const uint l_677 = uint(2304) + thread_pos_in_threadgroup.x;
shared_memory_4[l_677] = l_676;
const float l_678 = l_arr_6[uint(10)];
l_arr_5[uint(10)] = l_678;
const uint l_679 = uint(2560) + thread_pos_in_threadgroup.x;
shared_memory_4[l_679] = l_678;
const float l_680 = l_arr_6[uint(11)];
l_arr_5[uint(11)] = l_680;
const uint l_681 = uint(2816) + thread_pos_in_threadgroup.x;
shared_memory_4[l_681] = l_680;
const float l_682 = l_arr_6[uint(12)];
l_arr_5[uint(12)] = l_682;
const uint l_683 = uint(3072) + thread_pos_in_threadgroup.x;
shared_memory_4[l_683] = l_682;
const float l_684 = l_arr_6[uint(13)];
l_arr_5[uint(13)] = l_684;
const uint l_685 = uint(3328) + thread_pos_in_threadgroup.x;
shared_memory_4[l_685] = l_684;
const float l_686 = l_arr_6[uint(14)];
l_arr_5[uint(14)] = l_686;
const uint l_687 = uint(3584) + thread_pos_in_threadgroup.x;
shared_memory_4[l_687] = l_686;
const float l_688 = l_arr_6[uint(15)];
l_arr_5[uint(15)] = l_688;
const uint l_689 = uint(3840) + thread_pos_in_threadgroup.x;
shared_memory_4[l_689] = l_688;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_690 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_691 = thread_pos_in_threadgroup.x & uint(64);
const bool l_692 = l_691 != uint(0);
const float l_693 = l_arr_5[uint(0)];
const uint l_694 = uint(0) + l_690;
const float l_695 = shared_memory_4[l_694];
const float l_696 = l_693 + l_695;
l_mut_24 = l_696;
if (l_692) {
const float l_697 = l_695 - l_693;
l_mut_24 = l_697;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_698 = l_arr_5[uint(1)];
const uint l_699 = uint(256) + l_690;
const float l_700 = shared_memory_4[l_699];
const float l_701 = l_698 + l_700;
l_mut_24 = l_701;
if (l_692) {
const float l_702 = l_700 - l_698;
l_mut_24 = l_702;
}
l_arr_6[uint(1)] = l_mut_24;
const float l_703 = l_arr_5[uint(2)];
const uint l_704 = uint(512) + l_690;
const float l_705 = shared_memory_4[l_704];
const float l_706 = l_703 + l_705;
l_mut_24 = l_706;
if (l_692) {
const float l_707 = l_705 - l_703;
l_mut_24 = l_707;
}
l_arr_6[uint(2)] = l_mut_24;
const float l_708 = l_arr_5[uint(3)];
const uint l_709 = uint(768) + l_690;
const float l_710 = shared_memory_4[l_709];
const float l_711 = l_708 + l_710;
l_mut_24 = l_711;
if (l_692) {
const float l_712 = l_710 - l_708;
l_mut_24 = l_712;
}
l_arr_6[uint(3)] = l_mut_24;
const float l_713 = l_arr_5[uint(4)];
const uint l_714 = uint(1024) + l_690;
const float l_715 = shared_memory_4[l_714];
const float l_716 = l_713 + l_715;
l_mut_24 = l_716;
if (l_692) {
const float l_717 = l_715 - l_713;
l_mut_24 = l_717;
}
l_arr_6[uint(4)] = l_mut_24;
const float l_718 = l_arr_5[uint(5)];
const uint l_719 = uint(1280) + l_690;
const float l_720 = shared_memory_4[l_719];
const float l_721 = l_718 + l_720;
l_mut_24 = l_721;
if (l_692) {
const float l_722 = l_720 - l_718;
l_mut_24 = l_722;
}
l_arr_6[uint(5)] = l_mut_24;
const float l_723 = l_arr_5[uint(6)];
const uint l_724 = uint(1536) + l_690;
const float l_725 = shared_memory_4[l_724];
const float l_726 = l_723 + l_725;
l_mut_24 = l_726;
if (l_692) {
const float l_727 = l_725 - l_723;
l_mut_24 = l_727;
}
l_arr_6[uint(6)] = l_mut_24;
const float l_728 = l_arr_5[uint(7)];
const uint l_729 = uint(1792) + l_690;
const float l_730 = shared_memory_4[l_729];
const float l_731 = l_728 + l_730;
l_mut_24 = l_731;
if (l_692) {
const float l_732 = l_730 - l_728;
l_mut_24 = l_732;
}
l_arr_6[uint(7)] = l_mut_24;
const float l_733 = l_arr_5[uint(8)];
const uint l_734 = uint(2048) + l_690;
const float l_735 = shared_memory_4[l_734];
const float l_736 = l_733 + l_735;
l_mut_24 = l_736;
if (l_692) {
const float l_737 = l_735 - l_733;
l_mut_24 = l_737;
}
l_arr_6[uint(8)] = l_mut_24;
const float l_738 = l_arr_5[uint(9)];
const uint l_739 = uint(2304) + l_690;
const float l_740 = shared_memory_4[l_739];
const float l_741 = l_738 + l_740;
l_mut_24 = l_741;
if (l_692) {
const float l_742 = l_740 - l_738;
l_mut_24 = l_742;
}
l_arr_6[uint(9)] = l_mut_24;
const float l_743 = l_arr_5[uint(10)];
const uint l_744 = uint(2560) + l_690;
const float l_745 = shared_memory_4[l_744];
const float l_746 = l_743 + l_745;
l_mut_24 = l_746;
if (l_692) {
const float l_747 = l_745 - l_743;
l_mut_24 = l_747;
}
l_arr_6[uint(10)] = l_mut_24;
const float l_748 = l_arr_5[uint(11)];
const uint l_749 = uint(2816) + l_690;
const float l_750 = shared_memory_4[l_749];
const float l_751 = l_748 + l_750;
l_mut_24 = l_751;
if (l_692) {
const float l_752 = l_750 - l_748;
l_mut_24 = l_752;
}
l_arr_6[uint(11)] = l_mut_24;
const float l_753 = l_arr_5[uint(12)];
const uint l_754 = uint(3072) + l_690;
const float l_755 = shared_memory_4[l_754];
const float l_756 = l_753 + l_755;
l_mut_24 = l_756;
if (l_692) {
const float l_757 = l_755 - l_753;
l_mut_24 = l_757;
}
l_arr_6[uint(12)] = l_mut_24;
const float l_758 = l_arr_5[uint(13)];
const uint l_759 = uint(3328) + l_690;
const float l_760 = shared_memory_4[l_759];
const float l_761 = l_758 + l_760;
l_mut_24 = l_761;
if (l_692) {
const float l_762 = l_760 - l_758;
l_mut_24 = l_762;
}
l_arr_6[uint(13)] = l_mut_24;
const float l_763 = l_arr_5[uint(14)];
const uint l_764 = uint(3584) + l_690;
const float l_765 = shared_memory_4[l_764];
const float l_766 = l_763 + l_765;
l_mut_24 = l_766;
if (l_692) {
const float l_767 = l_765 - l_763;
l_mut_24 = l_767;
}
l_arr_6[uint(14)] = l_mut_24;
const float l_768 = l_arr_5[uint(15)];
const uint l_769 = uint(3840) + l_690;
const float l_770 = shared_memory_4[l_769];
const float l_771 = l_768 + l_770;
l_mut_24 = l_771;
if (l_692) {
const float l_772 = l_770 - l_768;
l_mut_24 = l_772;
}
l_arr_6[uint(15)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_773 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_773;
const uint l_774 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_774] = l_773;
const float l_775 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_775;
const uint l_776 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_776] = l_775;
const float l_777 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_777;
const uint l_778 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_778] = l_777;
const float l_779 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_779;
const uint l_780 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_780] = l_779;
const float l_781 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_781;
const uint l_782 = uint(1024) + thread_pos_in_threadgroup.x;
shared_memory_4[l_782] = l_781;
const float l_783 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_783;
const uint l_784 = uint(1280) + thread_pos_in_threadgroup.x;
shared_memory_4[l_784] = l_783;
const float l_785 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_785;
const uint l_786 = uint(1536) + thread_pos_in_threadgroup.x;
shared_memory_4[l_786] = l_785;
const float l_787 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_787;
const uint l_788 = uint(1792) + thread_pos_in_threadgroup.x;
shared_memory_4[l_788] = l_787;
const float l_789 = l_arr_6[uint(8)];
l_arr_5[uint(8)] = l_789;
const uint l_790 = uint(2048) + thread_pos_in_threadgroup.x;
shared_memory_4[l_790] = l_789;
const float l_791 = l_arr_6[uint(9)];
l_arr_5[uint(9)] = l_791;
const uint l_792 = uint(2304) + thread_pos_in_threadgroup.x;
shared_memory_4[l_792] = l_791;
const float l_793 = l_arr_6[uint(10)];
l_arr_5[uint(10)] = l_793;
const uint l_794 = uint(2560) + thread_pos_in_threadgroup.x;
shared_memory_4[l_794] = l_793;
const float l_795 = l_arr_6[uint(11)];
l_arr_5[uint(11)] = l_795;
const uint l_796 = uint(2816) + thread_pos_in_threadgroup.x;
shared_memory_4[l_796] = l_795;
const float l_797 = l_arr_6[uint(12)];
l_arr_5[uint(12)] = l_797;
const uint l_798 = uint(3072) + thread_pos_in_threadgroup.x;
shared_memory_4[l_798] = l_797;
const float l_799 = l_arr_6[uint(13)];
l_arr_5[uint(13)] = l_799;
const uint l_800 = uint(3328) + thread_pos_in_threadgroup.x;
shared_memory_4[l_800] = l_799;
const float l_801 = l_arr_6[uint(14)];
l_arr_5[uint(14)] = l_801;
const uint l_802 = uint(3584) + thread_pos_in_threadgroup.x;
shared_memory_4[l_802] = l_801;
const float l_803 = l_arr_6[uint(15)];
l_arr_5[uint(15)] = l_803;
const uint l_804 = uint(3840) + thread_pos_in_threadgroup.x;
shared_memory_4[l_804] = l_803;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_805 = thread_pos_in_threadgroup.x ^ uint(128);
const uint l_806 = thread_pos_in_threadgroup.x & uint(128);
const bool l_807 = l_806 != uint(0);
const float l_808 = l_arr_5[uint(0)];
const uint l_809 = uint(0) + l_805;
const float l_810 = shared_memory_4[l_809];
const float l_811 = l_808 + l_810;
l_mut_24 = l_811;
if (l_807) {
const float l_812 = l_810 - l_808;
l_mut_24 = l_812;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_813 = l_arr_5[uint(1)];
const uint l_814 = uint(256) + l_805;
const float l_815 = shared_memory_4[l_814];
const float l_816 = l_813 + l_815;
l_mut_24 = l_816;
if (l_807) {
const float l_817 = l_815 - l_813;
l_mut_24 = l_817;
}
l_arr_6[uint(1)] = l_mut_24;
const float l_818 = l_arr_5[uint(2)];
const uint l_819 = uint(512) + l_805;
const float l_820 = shared_memory_4[l_819];
const float l_821 = l_818 + l_820;
l_mut_24 = l_821;
if (l_807) {
const float l_822 = l_820 - l_818;
l_mut_24 = l_822;
}
l_arr_6[uint(2)] = l_mut_24;
const float l_823 = l_arr_5[uint(3)];
const uint l_824 = uint(768) + l_805;
const float l_825 = shared_memory_4[l_824];
const float l_826 = l_823 + l_825;
l_mut_24 = l_826;
if (l_807) {
const float l_827 = l_825 - l_823;
l_mut_24 = l_827;
}
l_arr_6[uint(3)] = l_mut_24;
const float l_828 = l_arr_5[uint(4)];
const uint l_829 = uint(1024) + l_805;
const float l_830 = shared_memory_4[l_829];
const float l_831 = l_828 + l_830;
l_mut_24 = l_831;
if (l_807) {
const float l_832 = l_830 - l_828;
l_mut_24 = l_832;
}
l_arr_6[uint(4)] = l_mut_24;
const float l_833 = l_arr_5[uint(5)];
const uint l_834 = uint(1280) + l_805;
const float l_835 = shared_memory_4[l_834];
const float l_836 = l_833 + l_835;
l_mut_24 = l_836;
if (l_807) {
const float l_837 = l_835 - l_833;
l_mut_24 = l_837;
}
l_arr_6[uint(5)] = l_mut_24;
const float l_838 = l_arr_5[uint(6)];
const uint l_839 = uint(1536) + l_805;
const float l_840 = shared_memory_4[l_839];
const float l_841 = l_838 + l_840;
l_mut_24 = l_841;
if (l_807) {
const float l_842 = l_840 - l_838;
l_mut_24 = l_842;
}
l_arr_6[uint(6)] = l_mut_24;
const float l_843 = l_arr_5[uint(7)];
const uint l_844 = uint(1792) + l_805;
const float l_845 = shared_memory_4[l_844];
const float l_846 = l_843 + l_845;
l_mut_24 = l_846;
if (l_807) {
const float l_847 = l_845 - l_843;
l_mut_24 = l_847;
}
l_arr_6[uint(7)] = l_mut_24;
const float l_848 = l_arr_5[uint(8)];
const uint l_849 = uint(2048) + l_805;
const float l_850 = shared_memory_4[l_849];
const float l_851 = l_848 + l_850;
l_mut_24 = l_851;
if (l_807) {
const float l_852 = l_850 - l_848;
l_mut_24 = l_852;
}
l_arr_6[uint(8)] = l_mut_24;
const float l_853 = l_arr_5[uint(9)];
const uint l_854 = uint(2304) + l_805;
const float l_855 = shared_memory_4[l_854];
const float l_856 = l_853 + l_855;
l_mut_24 = l_856;
if (l_807) {
const float l_857 = l_855 - l_853;
l_mut_24 = l_857;
}
l_arr_6[uint(9)] = l_mut_24;
const float l_858 = l_arr_5[uint(10)];
const uint l_859 = uint(2560) + l_805;
const float l_860 = shared_memory_4[l_859];
const float l_861 = l_858 + l_860;
l_mut_24 = l_861;
if (l_807) {
const float l_862 = l_860 - l_858;
l_mut_24 = l_862;
}
l_arr_6[uint(10)] = l_mut_24;
const float l_863 = l_arr_5[uint(11)];
const uint l_864 = uint(2816) + l_805;
const float l_865 = shared_memory_4[l_864];
const float l_866 = l_863 + l_865;
l_mut_24 = l_866;
if (l_807) {
const float l_867 = l_865 - l_863;
l_mut_24 = l_867;
}
l_arr_6[uint(11)] = l_mut_24;
const float l_868 = l_arr_5[uint(12)];
const uint l_869 = uint(3072) + l_805;
const float l_870 = shared_memory_4[l_869];
const float l_871 = l_868 + l_870;
l_mut_24 = l_871;
if (l_807) {
const float l_872 = l_870 - l_868;
l_mut_24 = l_872;
}
l_arr_6[uint(12)] = l_mut_24;
const float l_873 = l_arr_5[uint(13)];
const uint l_874 = uint(3328) + l_805;
const float l_875 = shared_memory_4[l_874];
const float l_876 = l_873 + l_875;
l_mut_24 = l_876;
if (l_807) {
const float l_877 = l_875 - l_873;
l_mut_24 = l_877;
}
l_arr_6[uint(13)] = l_mut_24;
const float l_878 = l_arr_5[uint(14)];
const uint l_879 = uint(3584) + l_805;
const float l_880 = shared_memory_4[l_879];
const float l_881 = l_878 + l_880;
l_mut_24 = l_881;
if (l_807) {
const float l_882 = l_880 - l_878;
l_mut_24 = l_882;
}
l_arr_6[uint(14)] = l_mut_24;
const float l_883 = l_arr_5[uint(15)];
const uint l_884 = uint(3840) + l_805;
const float l_885 = shared_memory_4[l_884];
const float l_886 = l_883 + l_885;
l_mut_24 = l_886;
if (l_807) {
const float l_887 = l_885 - l_883;
l_mut_24 = l_887;
}
l_arr_6[uint(15)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_888 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_888;
const float l_889 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_889;
const float l_890 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_890;
const float l_891 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_891;
const float l_892 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_892;
const float l_893 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_893;
const float l_894 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_894;
const float l_895 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_895;
const float l_896 = l_arr_6[uint(8)];
l_arr_5[uint(8)] = l_896;
const float l_897 = l_arr_6[uint(9)];
l_arr_5[uint(9)] = l_897;
const float l_898 = l_arr_6[uint(10)];
l_arr_5[uint(10)] = l_898;
const float l_899 = l_arr_6[uint(11)];
l_arr_5[uint(11)] = l_899;
const float l_900 = l_arr_6[uint(12)];
l_arr_5[uint(12)] = l_900;
const float l_901 = l_arr_6[uint(13)];
l_arr_5[uint(13)] = l_901;
const float l_902 = l_arr_6[uint(14)];
l_arr_5[uint(14)] = l_902;
const float l_903 = l_arr_6[uint(15)];
l_arr_5[uint(15)] = l_903;
const float l_904 = l_arr_5[uint(0)];
const float l_905 = l_arr_5[uint(1)];
const float l_906 = l_904 + l_905;
l_arr_5[uint(0)] = l_906;
const float l_907 = l_904 - l_905;
l_arr_5[uint(1)] = l_907;
const float l_908 = l_arr_5[uint(2)];
const float l_909 = l_arr_5[uint(3)];
const float l_910 = l_908 + l_909;
l_arr_5[uint(2)] = l_910;
const float l_911 = l_908 - l_909;
l_arr_5[uint(3)] = l_911;
const float l_912 = l_arr_5[uint(4)];
const float l_913 = l_arr_5[uint(5)];
const float l_914 = l_912 + l_913;
l_arr_5[uint(4)] = l_914;
const float l_915 = l_912 - l_913;
l_arr_5[uint(5)] = l_915;
const float l_916 = l_arr_5[uint(6)];
const float l_917 = l_arr_5[uint(7)];
const float l_918 = l_916 + l_917;
l_arr_5[uint(6)] = l_918;
const float l_919 = l_916 - l_917;
l_arr_5[uint(7)] = l_919;
const float l_920 = l_arr_5[uint(8)];
const float l_921 = l_arr_5[uint(9)];
const float l_922 = l_920 + l_921;
l_arr_5[uint(8)] = l_922;
const float l_923 = l_920 - l_921;
l_arr_5[uint(9)] = l_923;
const float l_924 = l_arr_5[uint(10)];
const float l_925 = l_arr_5[uint(11)];
const float l_926 = l_924 + l_925;
l_arr_5[uint(10)] = l_926;
const float l_927 = l_924 - l_925;
l_arr_5[uint(11)] = l_927;
const float l_928 = l_arr_5[uint(12)];
const float l_929 = l_arr_5[uint(13)];
const float l_930 = l_928 + l_929;
l_arr_5[uint(12)] = l_930;
const float l_931 = l_928 - l_929;
l_arr_5[uint(13)] = l_931;
const float l_932 = l_arr_5[uint(14)];
const float l_933 = l_arr_5[uint(15)];
const float l_934 = l_932 + l_933;
l_arr_5[uint(14)] = l_934;
const float l_935 = l_932 - l_933;
l_arr_5[uint(15)] = l_935;
const float l_936 = l_arr_5[uint(0)];
const float l_937 = l_arr_5[uint(2)];
const float l_938 = l_936 + l_937;
l_arr_5[uint(0)] = l_938;
const float l_939 = l_936 - l_937;
l_arr_5[uint(2)] = l_939;
const float l_940 = l_arr_5[uint(1)];
const float l_941 = l_arr_5[uint(3)];
const float l_942 = l_940 + l_941;
l_arr_5[uint(1)] = l_942;
const float l_943 = l_940 - l_941;
l_arr_5[uint(3)] = l_943;
const float l_944 = l_arr_5[uint(4)];
const float l_945 = l_arr_5[uint(6)];
const float l_946 = l_944 + l_945;
l_arr_5[uint(4)] = l_946;
const float l_947 = l_944 - l_945;
l_arr_5[uint(6)] = l_947;
const float l_948 = l_arr_5[uint(5)];
const float l_949 = l_arr_5[uint(7)];
const float l_950 = l_948 + l_949;
l_arr_5[uint(5)] = l_950;
const float l_951 = l_948 - l_949;
l_arr_5[uint(7)] = l_951;
const float l_952 = l_arr_5[uint(8)];
const float l_953 = l_arr_5[uint(10)];
const float l_954 = l_952 + l_953;
l_arr_5[uint(8)] = l_954;
const float l_955 = l_952 - l_953;
l_arr_5[uint(10)] = l_955;
const float l_956 = l_arr_5[uint(9)];
const float l_957 = l_arr_5[uint(11)];
const float l_958 = l_956 + l_957;
l_arr_5[uint(9)] = l_958;
const float l_959 = l_956 - l_957;
l_arr_5[uint(11)] = l_959;
const float l_960 = l_arr_5[uint(12)];
const float l_961 = l_arr_5[uint(14)];
const float l_962 = l_960 + l_961;
l_arr_5[uint(12)] = l_962;
const float l_963 = l_960 - l_961;
l_arr_5[uint(14)] = l_963;
const float l_964 = l_arr_5[uint(13)];
const float l_965 = l_arr_5[uint(15)];
const float l_966 = l_964 + l_965;
l_arr_5[uint(13)] = l_966;
const float l_967 = l_964 - l_965;
l_arr_5[uint(15)] = l_967;
const float l_968 = l_arr_5[uint(0)];
const float l_969 = l_arr_5[uint(4)];
const float l_970 = l_968 + l_969;
l_arr_5[uint(0)] = l_970;
const float l_971 = l_968 - l_969;
l_arr_5[uint(4)] = l_971;
const float l_972 = l_arr_5[uint(1)];
const float l_973 = l_arr_5[uint(5)];
const float l_974 = l_972 + l_973;
l_arr_5[uint(1)] = l_974;
const float l_975 = l_972 - l_973;
l_arr_5[uint(5)] = l_975;
const float l_976 = l_arr_5[uint(2)];
const float l_977 = l_arr_5[uint(6)];
const float l_978 = l_976 + l_977;
l_arr_5[uint(2)] = l_978;
const float l_979 = l_976 - l_977;
l_arr_5[uint(6)] = l_979;
const float l_980 = l_arr_5[uint(3)];
const float l_981 = l_arr_5[uint(7)];
const float l_982 = l_980 + l_981;
l_arr_5[uint(3)] = l_982;
const float l_983 = l_980 - l_981;
l_arr_5[uint(7)] = l_983;
const float l_984 = l_arr_5[uint(8)];
const float l_985 = l_arr_5[uint(12)];
const float l_986 = l_984 + l_985;
l_arr_5[uint(8)] = l_986;
const float l_987 = l_984 - l_985;
l_arr_5[uint(12)] = l_987;
const float l_988 = l_arr_5[uint(9)];
const float l_989 = l_arr_5[uint(13)];
const float l_990 = l_988 + l_989;
l_arr_5[uint(9)] = l_990;
const float l_991 = l_988 - l_989;
l_arr_5[uint(13)] = l_991;
const float l_992 = l_arr_5[uint(10)];
const float l_993 = l_arr_5[uint(14)];
const float l_994 = l_992 + l_993;
l_arr_5[uint(10)] = l_994;
const float l_995 = l_992 - l_993;
l_arr_5[uint(14)] = l_995;
const float l_996 = l_arr_5[uint(11)];
const float l_997 = l_arr_5[uint(15)];
const float l_998 = l_996 + l_997;
l_arr_5[uint(11)] = l_998;
const float l_999 = l_996 - l_997;
l_arr_5[uint(15)] = l_999;
const float l_1000 = l_arr_5[uint(0)];
const float l_1001 = l_arr_5[uint(8)];
const float l_1002 = l_1000 + l_1001;
l_arr_5[uint(0)] = l_1002;
const float l_1003 = l_1000 - l_1001;
l_arr_5[uint(8)] = l_1003;
const float l_1004 = l_arr_5[uint(1)];
const float l_1005 = l_arr_5[uint(9)];
const float l_1006 = l_1004 + l_1005;
l_arr_5[uint(1)] = l_1006;
const float l_1007 = l_1004 - l_1005;
l_arr_5[uint(9)] = l_1007;
const float l_1008 = l_arr_5[uint(2)];
const float l_1009 = l_arr_5[uint(10)];
const float l_1010 = l_1008 + l_1009;
l_arr_5[uint(2)] = l_1010;
const float l_1011 = l_1008 - l_1009;
l_arr_5[uint(10)] = l_1011;
const float l_1012 = l_arr_5[uint(3)];
const float l_1013 = l_arr_5[uint(11)];
const float l_1014 = l_1012 + l_1013;
l_arr_5[uint(3)] = l_1014;
const float l_1015 = l_1012 - l_1013;
l_arr_5[uint(11)] = l_1015;
const float l_1016 = l_arr_5[uint(4)];
const float l_1017 = l_arr_5[uint(12)];
const float l_1018 = l_1016 + l_1017;
l_arr_5[uint(4)] = l_1018;
const float l_1019 = l_1016 - l_1017;
l_arr_5[uint(12)] = l_1019;
const float l_1020 = l_arr_5[uint(5)];
const float l_1021 = l_arr_5[uint(13)];
const float l_1022 = l_1020 + l_1021;
l_arr_5[uint(5)] = l_1022;
const float l_1023 = l_1020 - l_1021;
l_arr_5[uint(13)] = l_1023;
const float l_1024 = l_arr_5[uint(6)];
const float l_1025 = l_arr_5[uint(14)];
const float l_1026 = l_1024 + l_1025;
l_arr_5[uint(6)] = l_1026;
const float l_1027 = l_1024 - l_1025;
l_arr_5[uint(14)] = l_1027;
const float l_1028 = l_arr_5[uint(7)];
const float l_1029 = l_arr_5[uint(15)];
const float l_1030 = l_1028 + l_1029;
l_arr_5[uint(7)] = l_1030;
const float l_1031 = l_1028 - l_1029;
l_arr_5[uint(15)] = l_1031;
const uint l_1032 = uint(0) + thread_pos_in_threadgroup.x;
const float l_1033 = l_arr_5[uint(0)];
const float l_1034 = l_1033 * float(0.015625);
l_mut_24 = l_1034;
const bool l_1035 = l_2 != uint(0);
const bool l_1036 = l_1 != uint(0);
const bool l_1037 = l_1035 && l_1036;
if (l_1037) {
const uint l_1038 = l_12 + l_1032;
const float l_1039 = buffer_1[l_1038];
l_mut_24 = l_mut_24 * l_1039;
}
const uint l_1040 = l_10 + l_1032;
buffer_2[l_1040] = l_mut_24;
const uint l_1041 = uint(256) + thread_pos_in_threadgroup.x;
const float l_1042 = l_arr_5[uint(1)];
const float l_1043 = l_1042 * float(0.015625);
l_mut_24 = l_1043;
const bool l_1044 = l_2 != uint(0);
const bool l_1045 = l_1 != uint(0);
const bool l_1046 = l_1044 && l_1045;
if (l_1046) {
const uint l_1047 = l_12 + l_1041;
const float l_1048 = buffer_1[l_1047];
l_mut_24 = l_mut_24 * l_1048;
}
const uint l_1049 = l_10 + l_1041;
buffer_2[l_1049] = l_mut_24;
const uint l_1050 = uint(512) + thread_pos_in_threadgroup.x;
const float l_1051 = l_arr_5[uint(2)];
const float l_1052 = l_1051 * float(0.015625);
l_mut_24 = l_1052;
const bool l_1053 = l_2 != uint(0);
const bool l_1054 = l_1 != uint(0);
const bool l_1055 = l_1053 && l_1054;
if (l_1055) {
const uint l_1056 = l_12 + l_1050;
const float l_1057 = buffer_1[l_1056];
l_mut_24 = l_mut_24 * l_1057;
}
const uint l_1058 = l_10 + l_1050;
buffer_2[l_1058] = l_mut_24;
const uint l_1059 = uint(768) + thread_pos_in_threadgroup.x;
const float l_1060 = l_arr_5[uint(3)];
const float l_1061 = l_1060 * float(0.015625);
l_mut_24 = l_1061;
const bool l_1062 = l_2 != uint(0);
const bool l_1063 = l_1 != uint(0);
const bool l_1064 = l_1062 && l_1063;
if (l_1064) {
const uint l_1065 = l_12 + l_1059;
const float l_1066 = buffer_1[l_1065];
l_mut_24 = l_mut_24 * l_1066;
}
const uint l_1067 = l_10 + l_1059;
buffer_2[l_1067] = l_mut_24;
const uint l_1068 = uint(1024) + thread_pos_in_threadgroup.x;
const float l_1069 = l_arr_5[uint(4)];
const float l_1070 = l_1069 * float(0.015625);
l_mut_24 = l_1070;
const bool l_1071 = l_2 != uint(0);
const bool l_1072 = l_1 != uint(0);
const bool l_1073 = l_1071 && l_1072;
if (l_1073) {
const uint l_1074 = l_12 + l_1068;
const float l_1075 = buffer_1[l_1074];
l_mut_24 = l_mut_24 * l_1075;
}
const uint l_1076 = l_10 + l_1068;
buffer_2[l_1076] = l_mut_24;
const uint l_1077 = uint(1280) + thread_pos_in_threadgroup.x;
const float l_1078 = l_arr_5[uint(5)];
const float l_1079 = l_1078 * float(0.015625);
l_mut_24 = l_1079;
const bool l_1080 = l_2 != uint(0);
const bool l_1081 = l_1 != uint(0);
const bool l_1082 = l_1080 && l_1081;
if (l_1082) {
const uint l_1083 = l_12 + l_1077;
const float l_1084 = buffer_1[l_1083];
l_mut_24 = l_mut_24 * l_1084;
}
const uint l_1085 = l_10 + l_1077;
buffer_2[l_1085] = l_mut_24;
const uint l_1086 = uint(1536) + thread_pos_in_threadgroup.x;
const float l_1087 = l_arr_5[uint(6)];
const float l_1088 = l_1087 * float(0.015625);
l_mut_24 = l_1088;
const bool l_1089 = l_2 != uint(0);
const bool l_1090 = l_1 != uint(0);
const bool l_1091 = l_1089 && l_1090;
if (l_1091) {
const uint l_1092 = l_12 + l_1086;
const float l_1093 = buffer_1[l_1092];
l_mut_24 = l_mut_24 * l_1093;
}
const uint l_1094 = l_10 + l_1086;
buffer_2[l_1094] = l_mut_24;
const uint l_1095 = uint(1792) + thread_pos_in_threadgroup.x;
const float l_1096 = l_arr_5[uint(7)];
const float l_1097 = l_1096 * float(0.015625);
l_mut_24 = l_1097;
const bool l_1098 = l_2 != uint(0);
const bool l_1099 = l_1 != uint(0);
const bool l_1100 = l_1098 && l_1099;
if (l_1100) {
const uint l_1101 = l_12 + l_1095;
const float l_1102 = buffer_1[l_1101];
l_mut_24 = l_mut_24 * l_1102;
}
const uint l_1103 = l_10 + l_1095;
buffer_2[l_1103] = l_mut_24;
const uint l_1104 = uint(2048) + thread_pos_in_threadgroup.x;
const float l_1105 = l_arr_5[uint(8)];
const float l_1106 = l_1105 * float(0.015625);
l_mut_24 = l_1106;
const bool l_1107 = l_2 != uint(0);
const bool l_1108 = l_1 != uint(0);
const bool l_1109 = l_1107 && l_1108;
if (l_1109) {
const uint l_1110 = l_12 + l_1104;
const float l_1111 = buffer_1[l_1110];
l_mut_24 = l_mut_24 * l_1111;
}
const uint l_1112 = l_10 + l_1104;
buffer_2[l_1112] = l_mut_24;
const uint l_1113 = uint(2304) + thread_pos_in_threadgroup.x;
const float l_1114 = l_arr_5[uint(9)];
const float l_1115 = l_1114 * float(0.015625);
l_mut_24 = l_1115;
const bool l_1116 = l_2 != uint(0);
const bool l_1117 = l_1 != uint(0);
const bool l_1118 = l_1116 && l_1117;
if (l_1118) {
const uint l_1119 = l_12 + l_1113;
const float l_1120 = buffer_1[l_1119];
l_mut_24 = l_mut_24 * l_1120;
}
const uint l_1121 = l_10 + l_1113;
buffer_2[l_1121] = l_mut_24;
const uint l_1122 = uint(2560) + thread_pos_in_threadgroup.x;
const float l_1123 = l_arr_5[uint(10)];
const float l_1124 = l_1123 * float(0.015625);
l_mut_24 = l_1124;
const bool l_1125 = l_2 != uint(0);
const bool l_1126 = l_1 != uint(0);
const bool l_1127 = l_1125 && l_1126;
if (l_1127) {
const uint l_1128 = l_12 + l_1122;
const float l_1129 = buffer_1[l_1128];
l_mut_24 = l_mut_24 * l_1129;
}
const uint l_1130 = l_10 + l_1122;
buffer_2[l_1130] = l_mut_24;
const uint l_1131 = uint(2816) + thread_pos_in_threadgroup.x;
const float l_1132 = l_arr_5[uint(11)];
const float l_1133 = l_1132 * float(0.015625);
l_mut_24 = l_1133;
const bool l_1134 = l_2 != uint(0);
const bool l_1135 = l_1 != uint(0);
const bool l_1136 = l_1134 && l_1135;
if (l_1136) {
const uint l_1137 = l_12 + l_1131;
const float l_1138 = buffer_1[l_1137];
l_mut_24 = l_mut_24 * l_1138;
}
const uint l_1139 = l_10 + l_1131;
buffer_2[l_1139] = l_mut_24;
const uint l_1140 = uint(3072) + thread_pos_in_threadgroup.x;
const float l_1141 = l_arr_5[uint(12)];
const float l_1142 = l_1141 * float(0.015625);
l_mut_24 = l_1142;
const bool l_1143 = l_2 != uint(0);
const bool l_1144 = l_1 != uint(0);
const bool l_1145 = l_1143 && l_1144;
if (l_1145) {
const uint l_1146 = l_12 + l_1140;
const float l_1147 = buffer_1[l_1146];
l_mut_24 = l_mut_24 * l_1147;
}
const uint l_1148 = l_10 + l_1140;
buffer_2[l_1148] = l_mut_24;
const uint l_1149 = uint(3328) + thread_pos_in_threadgroup.x;
const float l_1150 = l_arr_5[uint(13)];
const float l_1151 = l_1150 * float(0.015625);
l_mut_24 = l_1151;
const bool l_1152 = l_2 != uint(0);
const bool l_1153 = l_1 != uint(0);
const bool l_1154 = l_1152 && l_1153;
if (l_1154) {
const uint l_1155 = l_12 + l_1149;
const float l_1156 = buffer_1[l_1155];
l_mut_24 = l_mut_24 * l_1156;
}
const uint l_1157 = l_10 + l_1149;
buffer_2[l_1157] = l_mut_24;
const uint l_1158 = uint(3584) + thread_pos_in_threadgroup.x;
const float l_1159 = l_arr_5[uint(14)];
const float l_1160 = l_1159 * float(0.015625);
l_mut_24 = l_1160;
const bool l_1161 = l_2 != uint(0);
const bool l_1162 = l_1 != uint(0);
const bool l_1163 = l_1161 && l_1162;
if (l_1163) {
const uint l_1164 = l_12 + l_1158;
const float l_1165 = buffer_1[l_1164];
l_mut_24 = l_mut_24 * l_1165;
}
const uint l_1166 = l_10 + l_1158;
buffer_2[l_1166] = l_mut_24;
const uint l_1167 = uint(3840) + thread_pos_in_threadgroup.x;
const float l_1168 = l_arr_5[uint(15)];
const float l_1169 = l_1168 * float(0.015625);
l_mut_24 = l_1169;
const bool l_1170 = l_2 != uint(0);
const bool l_1171 = l_1 != uint(0);
const bool l_1172 = l_1170 && l_1171;
if (l_1172) {
const uint l_1173 = l_12 + l_1167;
const float l_1174 = buffer_1[l_1173];
l_mut_24 = l_mut_24 * l_1174;
}
const uint l_1175 = l_10 + l_1167;
buffer_2[l_1175] = l_mut_24;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
