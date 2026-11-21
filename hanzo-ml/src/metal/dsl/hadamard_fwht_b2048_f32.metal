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
threadgroup uchar dynamic_shared_mem[8192];
// Shared array size: 2048, 8192 bytes
threadgroup float* shared_memory_4 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
float l_arr_5[8];

float l_arr_6[8];

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
const uint l_10 = l_mut_7 * uint(2048);
const uint l_11 = l_mut_7 % l_0;
const uint l_12 = l_11 * uint(2048);
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
const uint l_279 = uint(0) + thread_pos_in_threadgroup.x;
const float l_280 = l_arr_5[uint(0)];
shared_memory_4[l_279] = l_280;
const uint l_281 = uint(256) + thread_pos_in_threadgroup.x;
const float l_282 = l_arr_5[uint(1)];
shared_memory_4[l_281] = l_282;
const uint l_283 = uint(512) + thread_pos_in_threadgroup.x;
const float l_284 = l_arr_5[uint(2)];
shared_memory_4[l_283] = l_284;
const uint l_285 = uint(768) + thread_pos_in_threadgroup.x;
const float l_286 = l_arr_5[uint(3)];
shared_memory_4[l_285] = l_286;
const uint l_287 = uint(1024) + thread_pos_in_threadgroup.x;
const float l_288 = l_arr_5[uint(4)];
shared_memory_4[l_287] = l_288;
const uint l_289 = uint(1280) + thread_pos_in_threadgroup.x;
const float l_290 = l_arr_5[uint(5)];
shared_memory_4[l_289] = l_290;
const uint l_291 = uint(1536) + thread_pos_in_threadgroup.x;
const float l_292 = l_arr_5[uint(6)];
shared_memory_4[l_291] = l_292;
const uint l_293 = uint(1792) + thread_pos_in_threadgroup.x;
const float l_294 = l_arr_5[uint(7)];
shared_memory_4[l_293] = l_294;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_295 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_296 = thread_pos_in_threadgroup.x & uint(32);
const bool l_297 = l_296 != uint(0);
const float l_298 = l_arr_5[uint(0)];
const uint l_299 = uint(0) + l_295;
const float l_300 = shared_memory_4[l_299];
const float l_301 = l_298 + l_300;
l_mut_24 = l_301;
if (l_297) {
const float l_302 = l_300 - l_298;
l_mut_24 = l_302;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_303 = l_arr_5[uint(1)];
const uint l_304 = uint(256) + l_295;
const float l_305 = shared_memory_4[l_304];
const float l_306 = l_303 + l_305;
l_mut_24 = l_306;
if (l_297) {
const float l_307 = l_305 - l_303;
l_mut_24 = l_307;
}
l_arr_6[uint(1)] = l_mut_24;
const float l_308 = l_arr_5[uint(2)];
const uint l_309 = uint(512) + l_295;
const float l_310 = shared_memory_4[l_309];
const float l_311 = l_308 + l_310;
l_mut_24 = l_311;
if (l_297) {
const float l_312 = l_310 - l_308;
l_mut_24 = l_312;
}
l_arr_6[uint(2)] = l_mut_24;
const float l_313 = l_arr_5[uint(3)];
const uint l_314 = uint(768) + l_295;
const float l_315 = shared_memory_4[l_314];
const float l_316 = l_313 + l_315;
l_mut_24 = l_316;
if (l_297) {
const float l_317 = l_315 - l_313;
l_mut_24 = l_317;
}
l_arr_6[uint(3)] = l_mut_24;
const float l_318 = l_arr_5[uint(4)];
const uint l_319 = uint(1024) + l_295;
const float l_320 = shared_memory_4[l_319];
const float l_321 = l_318 + l_320;
l_mut_24 = l_321;
if (l_297) {
const float l_322 = l_320 - l_318;
l_mut_24 = l_322;
}
l_arr_6[uint(4)] = l_mut_24;
const float l_323 = l_arr_5[uint(5)];
const uint l_324 = uint(1280) + l_295;
const float l_325 = shared_memory_4[l_324];
const float l_326 = l_323 + l_325;
l_mut_24 = l_326;
if (l_297) {
const float l_327 = l_325 - l_323;
l_mut_24 = l_327;
}
l_arr_6[uint(5)] = l_mut_24;
const float l_328 = l_arr_5[uint(6)];
const uint l_329 = uint(1536) + l_295;
const float l_330 = shared_memory_4[l_329];
const float l_331 = l_328 + l_330;
l_mut_24 = l_331;
if (l_297) {
const float l_332 = l_330 - l_328;
l_mut_24 = l_332;
}
l_arr_6[uint(6)] = l_mut_24;
const float l_333 = l_arr_5[uint(7)];
const uint l_334 = uint(1792) + l_295;
const float l_335 = shared_memory_4[l_334];
const float l_336 = l_333 + l_335;
l_mut_24 = l_336;
if (l_297) {
const float l_337 = l_335 - l_333;
l_mut_24 = l_337;
}
l_arr_6[uint(7)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_338 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_338;
const uint l_339 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_339] = l_338;
const float l_340 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_340;
const uint l_341 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_341] = l_340;
const float l_342 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_342;
const uint l_343 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_343] = l_342;
const float l_344 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_344;
const uint l_345 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_345] = l_344;
const float l_346 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_346;
const uint l_347 = uint(1024) + thread_pos_in_threadgroup.x;
shared_memory_4[l_347] = l_346;
const float l_348 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_348;
const uint l_349 = uint(1280) + thread_pos_in_threadgroup.x;
shared_memory_4[l_349] = l_348;
const float l_350 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_350;
const uint l_351 = uint(1536) + thread_pos_in_threadgroup.x;
shared_memory_4[l_351] = l_350;
const float l_352 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_352;
const uint l_353 = uint(1792) + thread_pos_in_threadgroup.x;
shared_memory_4[l_353] = l_352;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_354 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_355 = thread_pos_in_threadgroup.x & uint(64);
const bool l_356 = l_355 != uint(0);
const float l_357 = l_arr_5[uint(0)];
const uint l_358 = uint(0) + l_354;
const float l_359 = shared_memory_4[l_358];
const float l_360 = l_357 + l_359;
l_mut_24 = l_360;
if (l_356) {
const float l_361 = l_359 - l_357;
l_mut_24 = l_361;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_362 = l_arr_5[uint(1)];
const uint l_363 = uint(256) + l_354;
const float l_364 = shared_memory_4[l_363];
const float l_365 = l_362 + l_364;
l_mut_24 = l_365;
if (l_356) {
const float l_366 = l_364 - l_362;
l_mut_24 = l_366;
}
l_arr_6[uint(1)] = l_mut_24;
const float l_367 = l_arr_5[uint(2)];
const uint l_368 = uint(512) + l_354;
const float l_369 = shared_memory_4[l_368];
const float l_370 = l_367 + l_369;
l_mut_24 = l_370;
if (l_356) {
const float l_371 = l_369 - l_367;
l_mut_24 = l_371;
}
l_arr_6[uint(2)] = l_mut_24;
const float l_372 = l_arr_5[uint(3)];
const uint l_373 = uint(768) + l_354;
const float l_374 = shared_memory_4[l_373];
const float l_375 = l_372 + l_374;
l_mut_24 = l_375;
if (l_356) {
const float l_376 = l_374 - l_372;
l_mut_24 = l_376;
}
l_arr_6[uint(3)] = l_mut_24;
const float l_377 = l_arr_5[uint(4)];
const uint l_378 = uint(1024) + l_354;
const float l_379 = shared_memory_4[l_378];
const float l_380 = l_377 + l_379;
l_mut_24 = l_380;
if (l_356) {
const float l_381 = l_379 - l_377;
l_mut_24 = l_381;
}
l_arr_6[uint(4)] = l_mut_24;
const float l_382 = l_arr_5[uint(5)];
const uint l_383 = uint(1280) + l_354;
const float l_384 = shared_memory_4[l_383];
const float l_385 = l_382 + l_384;
l_mut_24 = l_385;
if (l_356) {
const float l_386 = l_384 - l_382;
l_mut_24 = l_386;
}
l_arr_6[uint(5)] = l_mut_24;
const float l_387 = l_arr_5[uint(6)];
const uint l_388 = uint(1536) + l_354;
const float l_389 = shared_memory_4[l_388];
const float l_390 = l_387 + l_389;
l_mut_24 = l_390;
if (l_356) {
const float l_391 = l_389 - l_387;
l_mut_24 = l_391;
}
l_arr_6[uint(6)] = l_mut_24;
const float l_392 = l_arr_5[uint(7)];
const uint l_393 = uint(1792) + l_354;
const float l_394 = shared_memory_4[l_393];
const float l_395 = l_392 + l_394;
l_mut_24 = l_395;
if (l_356) {
const float l_396 = l_394 - l_392;
l_mut_24 = l_396;
}
l_arr_6[uint(7)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_397 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_397;
const uint l_398 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_398] = l_397;
const float l_399 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_399;
const uint l_400 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_400] = l_399;
const float l_401 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_401;
const uint l_402 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_402] = l_401;
const float l_403 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_403;
const uint l_404 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_404] = l_403;
const float l_405 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_405;
const uint l_406 = uint(1024) + thread_pos_in_threadgroup.x;
shared_memory_4[l_406] = l_405;
const float l_407 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_407;
const uint l_408 = uint(1280) + thread_pos_in_threadgroup.x;
shared_memory_4[l_408] = l_407;
const float l_409 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_409;
const uint l_410 = uint(1536) + thread_pos_in_threadgroup.x;
shared_memory_4[l_410] = l_409;
const float l_411 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_411;
const uint l_412 = uint(1792) + thread_pos_in_threadgroup.x;
shared_memory_4[l_412] = l_411;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_413 = thread_pos_in_threadgroup.x ^ uint(128);
const uint l_414 = thread_pos_in_threadgroup.x & uint(128);
const bool l_415 = l_414 != uint(0);
const float l_416 = l_arr_5[uint(0)];
const uint l_417 = uint(0) + l_413;
const float l_418 = shared_memory_4[l_417];
const float l_419 = l_416 + l_418;
l_mut_24 = l_419;
if (l_415) {
const float l_420 = l_418 - l_416;
l_mut_24 = l_420;
}
l_arr_6[uint(0)] = l_mut_24;
const float l_421 = l_arr_5[uint(1)];
const uint l_422 = uint(256) + l_413;
const float l_423 = shared_memory_4[l_422];
const float l_424 = l_421 + l_423;
l_mut_24 = l_424;
if (l_415) {
const float l_425 = l_423 - l_421;
l_mut_24 = l_425;
}
l_arr_6[uint(1)] = l_mut_24;
const float l_426 = l_arr_5[uint(2)];
const uint l_427 = uint(512) + l_413;
const float l_428 = shared_memory_4[l_427];
const float l_429 = l_426 + l_428;
l_mut_24 = l_429;
if (l_415) {
const float l_430 = l_428 - l_426;
l_mut_24 = l_430;
}
l_arr_6[uint(2)] = l_mut_24;
const float l_431 = l_arr_5[uint(3)];
const uint l_432 = uint(768) + l_413;
const float l_433 = shared_memory_4[l_432];
const float l_434 = l_431 + l_433;
l_mut_24 = l_434;
if (l_415) {
const float l_435 = l_433 - l_431;
l_mut_24 = l_435;
}
l_arr_6[uint(3)] = l_mut_24;
const float l_436 = l_arr_5[uint(4)];
const uint l_437 = uint(1024) + l_413;
const float l_438 = shared_memory_4[l_437];
const float l_439 = l_436 + l_438;
l_mut_24 = l_439;
if (l_415) {
const float l_440 = l_438 - l_436;
l_mut_24 = l_440;
}
l_arr_6[uint(4)] = l_mut_24;
const float l_441 = l_arr_5[uint(5)];
const uint l_442 = uint(1280) + l_413;
const float l_443 = shared_memory_4[l_442];
const float l_444 = l_441 + l_443;
l_mut_24 = l_444;
if (l_415) {
const float l_445 = l_443 - l_441;
l_mut_24 = l_445;
}
l_arr_6[uint(5)] = l_mut_24;
const float l_446 = l_arr_5[uint(6)];
const uint l_447 = uint(1536) + l_413;
const float l_448 = shared_memory_4[l_447];
const float l_449 = l_446 + l_448;
l_mut_24 = l_449;
if (l_415) {
const float l_450 = l_448 - l_446;
l_mut_24 = l_450;
}
l_arr_6[uint(6)] = l_mut_24;
const float l_451 = l_arr_5[uint(7)];
const uint l_452 = uint(1792) + l_413;
const float l_453 = shared_memory_4[l_452];
const float l_454 = l_451 + l_453;
l_mut_24 = l_454;
if (l_415) {
const float l_455 = l_453 - l_451;
l_mut_24 = l_455;
}
l_arr_6[uint(7)] = l_mut_24;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_456 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_456;
const float l_457 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_457;
const float l_458 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_458;
const float l_459 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_459;
const float l_460 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_460;
const float l_461 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_461;
const float l_462 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_462;
const float l_463 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_463;
const float l_464 = l_arr_5[uint(0)];
const float l_465 = l_arr_5[uint(1)];
const float l_466 = l_464 + l_465;
l_arr_5[uint(0)] = l_466;
const float l_467 = l_464 - l_465;
l_arr_5[uint(1)] = l_467;
const float l_468 = l_arr_5[uint(2)];
const float l_469 = l_arr_5[uint(3)];
const float l_470 = l_468 + l_469;
l_arr_5[uint(2)] = l_470;
const float l_471 = l_468 - l_469;
l_arr_5[uint(3)] = l_471;
const float l_472 = l_arr_5[uint(4)];
const float l_473 = l_arr_5[uint(5)];
const float l_474 = l_472 + l_473;
l_arr_5[uint(4)] = l_474;
const float l_475 = l_472 - l_473;
l_arr_5[uint(5)] = l_475;
const float l_476 = l_arr_5[uint(6)];
const float l_477 = l_arr_5[uint(7)];
const float l_478 = l_476 + l_477;
l_arr_5[uint(6)] = l_478;
const float l_479 = l_476 - l_477;
l_arr_5[uint(7)] = l_479;
const float l_480 = l_arr_5[uint(0)];
const float l_481 = l_arr_5[uint(2)];
const float l_482 = l_480 + l_481;
l_arr_5[uint(0)] = l_482;
const float l_483 = l_480 - l_481;
l_arr_5[uint(2)] = l_483;
const float l_484 = l_arr_5[uint(1)];
const float l_485 = l_arr_5[uint(3)];
const float l_486 = l_484 + l_485;
l_arr_5[uint(1)] = l_486;
const float l_487 = l_484 - l_485;
l_arr_5[uint(3)] = l_487;
const float l_488 = l_arr_5[uint(4)];
const float l_489 = l_arr_5[uint(6)];
const float l_490 = l_488 + l_489;
l_arr_5[uint(4)] = l_490;
const float l_491 = l_488 - l_489;
l_arr_5[uint(6)] = l_491;
const float l_492 = l_arr_5[uint(5)];
const float l_493 = l_arr_5[uint(7)];
const float l_494 = l_492 + l_493;
l_arr_5[uint(5)] = l_494;
const float l_495 = l_492 - l_493;
l_arr_5[uint(7)] = l_495;
const float l_496 = l_arr_5[uint(0)];
const float l_497 = l_arr_5[uint(4)];
const float l_498 = l_496 + l_497;
l_arr_5[uint(0)] = l_498;
const float l_499 = l_496 - l_497;
l_arr_5[uint(4)] = l_499;
const float l_500 = l_arr_5[uint(1)];
const float l_501 = l_arr_5[uint(5)];
const float l_502 = l_500 + l_501;
l_arr_5[uint(1)] = l_502;
const float l_503 = l_500 - l_501;
l_arr_5[uint(5)] = l_503;
const float l_504 = l_arr_5[uint(2)];
const float l_505 = l_arr_5[uint(6)];
const float l_506 = l_504 + l_505;
l_arr_5[uint(2)] = l_506;
const float l_507 = l_504 - l_505;
l_arr_5[uint(6)] = l_507;
const float l_508 = l_arr_5[uint(3)];
const float l_509 = l_arr_5[uint(7)];
const float l_510 = l_508 + l_509;
l_arr_5[uint(3)] = l_510;
const float l_511 = l_508 - l_509;
l_arr_5[uint(7)] = l_511;
const uint l_512 = uint(0) + thread_pos_in_threadgroup.x;
const float l_513 = l_arr_5[uint(0)];
const float l_514 = l_513 * float(0.022097086533904076);
l_mut_24 = l_514;
const bool l_515 = l_2 != uint(0);
const bool l_516 = l_1 != uint(0);
const bool l_517 = l_515 && l_516;
if (l_517) {
const uint l_518 = l_12 + l_512;
const float l_519 = buffer_1[l_518];
l_mut_24 = l_mut_24 * l_519;
}
const uint l_520 = l_10 + l_512;
buffer_2[l_520] = l_mut_24;
const uint l_521 = uint(256) + thread_pos_in_threadgroup.x;
const float l_522 = l_arr_5[uint(1)];
const float l_523 = l_522 * float(0.022097086533904076);
l_mut_24 = l_523;
const bool l_524 = l_2 != uint(0);
const bool l_525 = l_1 != uint(0);
const bool l_526 = l_524 && l_525;
if (l_526) {
const uint l_527 = l_12 + l_521;
const float l_528 = buffer_1[l_527];
l_mut_24 = l_mut_24 * l_528;
}
const uint l_529 = l_10 + l_521;
buffer_2[l_529] = l_mut_24;
const uint l_530 = uint(512) + thread_pos_in_threadgroup.x;
const float l_531 = l_arr_5[uint(2)];
const float l_532 = l_531 * float(0.022097086533904076);
l_mut_24 = l_532;
const bool l_533 = l_2 != uint(0);
const bool l_534 = l_1 != uint(0);
const bool l_535 = l_533 && l_534;
if (l_535) {
const uint l_536 = l_12 + l_530;
const float l_537 = buffer_1[l_536];
l_mut_24 = l_mut_24 * l_537;
}
const uint l_538 = l_10 + l_530;
buffer_2[l_538] = l_mut_24;
const uint l_539 = uint(768) + thread_pos_in_threadgroup.x;
const float l_540 = l_arr_5[uint(3)];
const float l_541 = l_540 * float(0.022097086533904076);
l_mut_24 = l_541;
const bool l_542 = l_2 != uint(0);
const bool l_543 = l_1 != uint(0);
const bool l_544 = l_542 && l_543;
if (l_544) {
const uint l_545 = l_12 + l_539;
const float l_546 = buffer_1[l_545];
l_mut_24 = l_mut_24 * l_546;
}
const uint l_547 = l_10 + l_539;
buffer_2[l_547] = l_mut_24;
const uint l_548 = uint(1024) + thread_pos_in_threadgroup.x;
const float l_549 = l_arr_5[uint(4)];
const float l_550 = l_549 * float(0.022097086533904076);
l_mut_24 = l_550;
const bool l_551 = l_2 != uint(0);
const bool l_552 = l_1 != uint(0);
const bool l_553 = l_551 && l_552;
if (l_553) {
const uint l_554 = l_12 + l_548;
const float l_555 = buffer_1[l_554];
l_mut_24 = l_mut_24 * l_555;
}
const uint l_556 = l_10 + l_548;
buffer_2[l_556] = l_mut_24;
const uint l_557 = uint(1280) + thread_pos_in_threadgroup.x;
const float l_558 = l_arr_5[uint(5)];
const float l_559 = l_558 * float(0.022097086533904076);
l_mut_24 = l_559;
const bool l_560 = l_2 != uint(0);
const bool l_561 = l_1 != uint(0);
const bool l_562 = l_560 && l_561;
if (l_562) {
const uint l_563 = l_12 + l_557;
const float l_564 = buffer_1[l_563];
l_mut_24 = l_mut_24 * l_564;
}
const uint l_565 = l_10 + l_557;
buffer_2[l_565] = l_mut_24;
const uint l_566 = uint(1536) + thread_pos_in_threadgroup.x;
const float l_567 = l_arr_5[uint(6)];
const float l_568 = l_567 * float(0.022097086533904076);
l_mut_24 = l_568;
const bool l_569 = l_2 != uint(0);
const bool l_570 = l_1 != uint(0);
const bool l_571 = l_569 && l_570;
if (l_571) {
const uint l_572 = l_12 + l_566;
const float l_573 = buffer_1[l_572];
l_mut_24 = l_mut_24 * l_573;
}
const uint l_574 = l_10 + l_566;
buffer_2[l_574] = l_mut_24;
const uint l_575 = uint(1792) + thread_pos_in_threadgroup.x;
const float l_576 = l_arr_5[uint(7)];
const float l_577 = l_576 * float(0.022097086533904076);
l_mut_24 = l_577;
const bool l_578 = l_2 != uint(0);
const bool l_579 = l_1 != uint(0);
const bool l_580 = l_578 && l_579;
if (l_580) {
const uint l_581 = l_12 + l_575;
const float l_582 = buffer_1[l_581];
l_mut_24 = l_mut_24 * l_582;
}
const uint l_583 = l_10 + l_575;
buffer_2[l_583] = l_mut_24;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
