#include <metal_stdlib>
using namespace metal;

struct info_st {
    uint static_meta[8];
};

[[kernel]]
void hadamard_fwht_f_(
    const device half * buffer_0 [[buffer(0)]],
    const device float * buffer_1 [[buffer(1)]],
    device half * buffer_2 [[buffer(2)]],
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
const uint l_10 = l_mut_7 * uint(2048);
const uint l_11 = l_mut_7 % l_0;
const uint l_12 = l_11 * uint(2048);
const uint l_13 = uint(0) + thread_pos_in_threadgroup.x;
const uint l_14 = l_10 + l_13;
const half l_15 = buffer_0[l_14];
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
const half l_51 = buffer_0[l_50];
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
const half l_85 = buffer_0[l_84];
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
const half l_119 = buffer_0[l_118];
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
const uint l_151 = uint(1024) + thread_pos_in_threadgroup.x;
const uint l_152 = l_10 + l_151;
const half l_153 = buffer_0[l_152];
const float l_154 = float(l_153);
l_mut_25 = l_154;
const bool l_155 = l_2 == uint(0);
const bool l_156 = l_1 != uint(0);
const bool l_157 = l_155 && l_156;
if (l_157) {
const uint l_158 = l_12 + l_151;
const float l_159 = buffer_1[l_158];
l_mut_25 = l_mut_25 * l_159;
}
const float l_160 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_161 = l_mut_25 + l_160;
l_mut_17 = l_161;
const uint l_162 = thread_pos_in_threadgroup.x & uint(1);
const bool l_163 = l_162 != uint(0);
if (l_163) {
const float l_164 = l_160 - l_mut_25;
l_mut_17 = l_164;
}
l_mut_25 = l_mut_17;
const float l_165 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_166 = l_mut_25 + l_165;
l_mut_17 = l_166;
const uint l_167 = thread_pos_in_threadgroup.x & uint(2);
const bool l_168 = l_167 != uint(0);
if (l_168) {
const float l_169 = l_165 - l_mut_25;
l_mut_17 = l_169;
}
l_mut_25 = l_mut_17;
const float l_170 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_171 = l_mut_25 + l_170;
l_mut_17 = l_171;
const uint l_172 = thread_pos_in_threadgroup.x & uint(4);
const bool l_173 = l_172 != uint(0);
if (l_173) {
const float l_174 = l_170 - l_mut_25;
l_mut_17 = l_174;
}
l_mut_25 = l_mut_17;
const float l_175 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_176 = l_mut_25 + l_175;
l_mut_17 = l_176;
const uint l_177 = thread_pos_in_threadgroup.x & uint(8);
const bool l_178 = l_177 != uint(0);
if (l_178) {
const float l_179 = l_175 - l_mut_25;
l_mut_17 = l_179;
}
l_mut_25 = l_mut_17;
const float l_180 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_181 = l_mut_25 + l_180;
l_mut_17 = l_181;
const uint l_182 = thread_pos_in_threadgroup.x & uint(16);
const bool l_183 = l_182 != uint(0);
if (l_183) {
const float l_184 = l_180 - l_mut_25;
l_mut_17 = l_184;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(4)] = l_mut_25;
const uint l_185 = uint(1280) + thread_pos_in_threadgroup.x;
const uint l_186 = l_10 + l_185;
const half l_187 = buffer_0[l_186];
const float l_188 = float(l_187);
l_mut_25 = l_188;
const bool l_189 = l_2 == uint(0);
const bool l_190 = l_1 != uint(0);
const bool l_191 = l_189 && l_190;
if (l_191) {
const uint l_192 = l_12 + l_185;
const float l_193 = buffer_1[l_192];
l_mut_25 = l_mut_25 * l_193;
}
const float l_194 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_195 = l_mut_25 + l_194;
l_mut_17 = l_195;
const uint l_196 = thread_pos_in_threadgroup.x & uint(1);
const bool l_197 = l_196 != uint(0);
if (l_197) {
const float l_198 = l_194 - l_mut_25;
l_mut_17 = l_198;
}
l_mut_25 = l_mut_17;
const float l_199 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_200 = l_mut_25 + l_199;
l_mut_17 = l_200;
const uint l_201 = thread_pos_in_threadgroup.x & uint(2);
const bool l_202 = l_201 != uint(0);
if (l_202) {
const float l_203 = l_199 - l_mut_25;
l_mut_17 = l_203;
}
l_mut_25 = l_mut_17;
const float l_204 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_205 = l_mut_25 + l_204;
l_mut_17 = l_205;
const uint l_206 = thread_pos_in_threadgroup.x & uint(4);
const bool l_207 = l_206 != uint(0);
if (l_207) {
const float l_208 = l_204 - l_mut_25;
l_mut_17 = l_208;
}
l_mut_25 = l_mut_17;
const float l_209 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_210 = l_mut_25 + l_209;
l_mut_17 = l_210;
const uint l_211 = thread_pos_in_threadgroup.x & uint(8);
const bool l_212 = l_211 != uint(0);
if (l_212) {
const float l_213 = l_209 - l_mut_25;
l_mut_17 = l_213;
}
l_mut_25 = l_mut_17;
const float l_214 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_215 = l_mut_25 + l_214;
l_mut_17 = l_215;
const uint l_216 = thread_pos_in_threadgroup.x & uint(16);
const bool l_217 = l_216 != uint(0);
if (l_217) {
const float l_218 = l_214 - l_mut_25;
l_mut_17 = l_218;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(5)] = l_mut_25;
const uint l_219 = uint(1536) + thread_pos_in_threadgroup.x;
const uint l_220 = l_10 + l_219;
const half l_221 = buffer_0[l_220];
const float l_222 = float(l_221);
l_mut_25 = l_222;
const bool l_223 = l_2 == uint(0);
const bool l_224 = l_1 != uint(0);
const bool l_225 = l_223 && l_224;
if (l_225) {
const uint l_226 = l_12 + l_219;
const float l_227 = buffer_1[l_226];
l_mut_25 = l_mut_25 * l_227;
}
const float l_228 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_229 = l_mut_25 + l_228;
l_mut_17 = l_229;
const uint l_230 = thread_pos_in_threadgroup.x & uint(1);
const bool l_231 = l_230 != uint(0);
if (l_231) {
const float l_232 = l_228 - l_mut_25;
l_mut_17 = l_232;
}
l_mut_25 = l_mut_17;
const float l_233 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_234 = l_mut_25 + l_233;
l_mut_17 = l_234;
const uint l_235 = thread_pos_in_threadgroup.x & uint(2);
const bool l_236 = l_235 != uint(0);
if (l_236) {
const float l_237 = l_233 - l_mut_25;
l_mut_17 = l_237;
}
l_mut_25 = l_mut_17;
const float l_238 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_239 = l_mut_25 + l_238;
l_mut_17 = l_239;
const uint l_240 = thread_pos_in_threadgroup.x & uint(4);
const bool l_241 = l_240 != uint(0);
if (l_241) {
const float l_242 = l_238 - l_mut_25;
l_mut_17 = l_242;
}
l_mut_25 = l_mut_17;
const float l_243 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_244 = l_mut_25 + l_243;
l_mut_17 = l_244;
const uint l_245 = thread_pos_in_threadgroup.x & uint(8);
const bool l_246 = l_245 != uint(0);
if (l_246) {
const float l_247 = l_243 - l_mut_25;
l_mut_17 = l_247;
}
l_mut_25 = l_mut_17;
const float l_248 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_249 = l_mut_25 + l_248;
l_mut_17 = l_249;
const uint l_250 = thread_pos_in_threadgroup.x & uint(16);
const bool l_251 = l_250 != uint(0);
if (l_251) {
const float l_252 = l_248 - l_mut_25;
l_mut_17 = l_252;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(6)] = l_mut_25;
const uint l_253 = uint(1792) + thread_pos_in_threadgroup.x;
const uint l_254 = l_10 + l_253;
const half l_255 = buffer_0[l_254];
const float l_256 = float(l_255);
l_mut_25 = l_256;
const bool l_257 = l_2 == uint(0);
const bool l_258 = l_1 != uint(0);
const bool l_259 = l_257 && l_258;
if (l_259) {
const uint l_260 = l_12 + l_253;
const float l_261 = buffer_1[l_260];
l_mut_25 = l_mut_25 * l_261;
}
const float l_262 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_263 = l_mut_25 + l_262;
l_mut_17 = l_263;
const uint l_264 = thread_pos_in_threadgroup.x & uint(1);
const bool l_265 = l_264 != uint(0);
if (l_265) {
const float l_266 = l_262 - l_mut_25;
l_mut_17 = l_266;
}
l_mut_25 = l_mut_17;
const float l_267 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_268 = l_mut_25 + l_267;
l_mut_17 = l_268;
const uint l_269 = thread_pos_in_threadgroup.x & uint(2);
const bool l_270 = l_269 != uint(0);
if (l_270) {
const float l_271 = l_267 - l_mut_25;
l_mut_17 = l_271;
}
l_mut_25 = l_mut_17;
const float l_272 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_273 = l_mut_25 + l_272;
l_mut_17 = l_273;
const uint l_274 = thread_pos_in_threadgroup.x & uint(4);
const bool l_275 = l_274 != uint(0);
if (l_275) {
const float l_276 = l_272 - l_mut_25;
l_mut_17 = l_276;
}
l_mut_25 = l_mut_17;
const float l_277 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_278 = l_mut_25 + l_277;
l_mut_17 = l_278;
const uint l_279 = thread_pos_in_threadgroup.x & uint(8);
const bool l_280 = l_279 != uint(0);
if (l_280) {
const float l_281 = l_277 - l_mut_25;
l_mut_17 = l_281;
}
l_mut_25 = l_mut_17;
const float l_282 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_283 = l_mut_25 + l_282;
l_mut_17 = l_283;
const uint l_284 = thread_pos_in_threadgroup.x & uint(16);
const bool l_285 = l_284 != uint(0);
if (l_285) {
const float l_286 = l_282 - l_mut_25;
l_mut_17 = l_286;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(7)] = l_mut_25;
const uint l_287 = uint(0) + thread_pos_in_threadgroup.x;
const float l_288 = l_arr_5[uint(0)];
shared_memory_4[l_287] = l_288;
const uint l_289 = uint(256) + thread_pos_in_threadgroup.x;
const float l_290 = l_arr_5[uint(1)];
shared_memory_4[l_289] = l_290;
const uint l_291 = uint(512) + thread_pos_in_threadgroup.x;
const float l_292 = l_arr_5[uint(2)];
shared_memory_4[l_291] = l_292;
const uint l_293 = uint(768) + thread_pos_in_threadgroup.x;
const float l_294 = l_arr_5[uint(3)];
shared_memory_4[l_293] = l_294;
const uint l_295 = uint(1024) + thread_pos_in_threadgroup.x;
const float l_296 = l_arr_5[uint(4)];
shared_memory_4[l_295] = l_296;
const uint l_297 = uint(1280) + thread_pos_in_threadgroup.x;
const float l_298 = l_arr_5[uint(5)];
shared_memory_4[l_297] = l_298;
const uint l_299 = uint(1536) + thread_pos_in_threadgroup.x;
const float l_300 = l_arr_5[uint(6)];
shared_memory_4[l_299] = l_300;
const uint l_301 = uint(1792) + thread_pos_in_threadgroup.x;
const float l_302 = l_arr_5[uint(7)];
shared_memory_4[l_301] = l_302;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_303 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_304 = thread_pos_in_threadgroup.x & uint(32);
const bool l_305 = l_304 != uint(0);
const float l_306 = l_arr_5[uint(0)];
const uint l_307 = uint(0) + l_303;
const float l_308 = shared_memory_4[l_307];
const float l_309 = l_306 + l_308;
l_mut_25 = l_309;
if (l_305) {
const float l_310 = l_308 - l_306;
l_mut_25 = l_310;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_311 = l_arr_5[uint(1)];
const uint l_312 = uint(256) + l_303;
const float l_313 = shared_memory_4[l_312];
const float l_314 = l_311 + l_313;
l_mut_25 = l_314;
if (l_305) {
const float l_315 = l_313 - l_311;
l_mut_25 = l_315;
}
l_arr_6[uint(1)] = l_mut_25;
const float l_316 = l_arr_5[uint(2)];
const uint l_317 = uint(512) + l_303;
const float l_318 = shared_memory_4[l_317];
const float l_319 = l_316 + l_318;
l_mut_25 = l_319;
if (l_305) {
const float l_320 = l_318 - l_316;
l_mut_25 = l_320;
}
l_arr_6[uint(2)] = l_mut_25;
const float l_321 = l_arr_5[uint(3)];
const uint l_322 = uint(768) + l_303;
const float l_323 = shared_memory_4[l_322];
const float l_324 = l_321 + l_323;
l_mut_25 = l_324;
if (l_305) {
const float l_325 = l_323 - l_321;
l_mut_25 = l_325;
}
l_arr_6[uint(3)] = l_mut_25;
const float l_326 = l_arr_5[uint(4)];
const uint l_327 = uint(1024) + l_303;
const float l_328 = shared_memory_4[l_327];
const float l_329 = l_326 + l_328;
l_mut_25 = l_329;
if (l_305) {
const float l_330 = l_328 - l_326;
l_mut_25 = l_330;
}
l_arr_6[uint(4)] = l_mut_25;
const float l_331 = l_arr_5[uint(5)];
const uint l_332 = uint(1280) + l_303;
const float l_333 = shared_memory_4[l_332];
const float l_334 = l_331 + l_333;
l_mut_25 = l_334;
if (l_305) {
const float l_335 = l_333 - l_331;
l_mut_25 = l_335;
}
l_arr_6[uint(5)] = l_mut_25;
const float l_336 = l_arr_5[uint(6)];
const uint l_337 = uint(1536) + l_303;
const float l_338 = shared_memory_4[l_337];
const float l_339 = l_336 + l_338;
l_mut_25 = l_339;
if (l_305) {
const float l_340 = l_338 - l_336;
l_mut_25 = l_340;
}
l_arr_6[uint(6)] = l_mut_25;
const float l_341 = l_arr_5[uint(7)];
const uint l_342 = uint(1792) + l_303;
const float l_343 = shared_memory_4[l_342];
const float l_344 = l_341 + l_343;
l_mut_25 = l_344;
if (l_305) {
const float l_345 = l_343 - l_341;
l_mut_25 = l_345;
}
l_arr_6[uint(7)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_346 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_346;
const uint l_347 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_347] = l_346;
const float l_348 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_348;
const uint l_349 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_349] = l_348;
const float l_350 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_350;
const uint l_351 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_351] = l_350;
const float l_352 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_352;
const uint l_353 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_353] = l_352;
const float l_354 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_354;
const uint l_355 = uint(1024) + thread_pos_in_threadgroup.x;
shared_memory_4[l_355] = l_354;
const float l_356 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_356;
const uint l_357 = uint(1280) + thread_pos_in_threadgroup.x;
shared_memory_4[l_357] = l_356;
const float l_358 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_358;
const uint l_359 = uint(1536) + thread_pos_in_threadgroup.x;
shared_memory_4[l_359] = l_358;
const float l_360 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_360;
const uint l_361 = uint(1792) + thread_pos_in_threadgroup.x;
shared_memory_4[l_361] = l_360;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_362 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_363 = thread_pos_in_threadgroup.x & uint(64);
const bool l_364 = l_363 != uint(0);
const float l_365 = l_arr_5[uint(0)];
const uint l_366 = uint(0) + l_362;
const float l_367 = shared_memory_4[l_366];
const float l_368 = l_365 + l_367;
l_mut_25 = l_368;
if (l_364) {
const float l_369 = l_367 - l_365;
l_mut_25 = l_369;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_370 = l_arr_5[uint(1)];
const uint l_371 = uint(256) + l_362;
const float l_372 = shared_memory_4[l_371];
const float l_373 = l_370 + l_372;
l_mut_25 = l_373;
if (l_364) {
const float l_374 = l_372 - l_370;
l_mut_25 = l_374;
}
l_arr_6[uint(1)] = l_mut_25;
const float l_375 = l_arr_5[uint(2)];
const uint l_376 = uint(512) + l_362;
const float l_377 = shared_memory_4[l_376];
const float l_378 = l_375 + l_377;
l_mut_25 = l_378;
if (l_364) {
const float l_379 = l_377 - l_375;
l_mut_25 = l_379;
}
l_arr_6[uint(2)] = l_mut_25;
const float l_380 = l_arr_5[uint(3)];
const uint l_381 = uint(768) + l_362;
const float l_382 = shared_memory_4[l_381];
const float l_383 = l_380 + l_382;
l_mut_25 = l_383;
if (l_364) {
const float l_384 = l_382 - l_380;
l_mut_25 = l_384;
}
l_arr_6[uint(3)] = l_mut_25;
const float l_385 = l_arr_5[uint(4)];
const uint l_386 = uint(1024) + l_362;
const float l_387 = shared_memory_4[l_386];
const float l_388 = l_385 + l_387;
l_mut_25 = l_388;
if (l_364) {
const float l_389 = l_387 - l_385;
l_mut_25 = l_389;
}
l_arr_6[uint(4)] = l_mut_25;
const float l_390 = l_arr_5[uint(5)];
const uint l_391 = uint(1280) + l_362;
const float l_392 = shared_memory_4[l_391];
const float l_393 = l_390 + l_392;
l_mut_25 = l_393;
if (l_364) {
const float l_394 = l_392 - l_390;
l_mut_25 = l_394;
}
l_arr_6[uint(5)] = l_mut_25;
const float l_395 = l_arr_5[uint(6)];
const uint l_396 = uint(1536) + l_362;
const float l_397 = shared_memory_4[l_396];
const float l_398 = l_395 + l_397;
l_mut_25 = l_398;
if (l_364) {
const float l_399 = l_397 - l_395;
l_mut_25 = l_399;
}
l_arr_6[uint(6)] = l_mut_25;
const float l_400 = l_arr_5[uint(7)];
const uint l_401 = uint(1792) + l_362;
const float l_402 = shared_memory_4[l_401];
const float l_403 = l_400 + l_402;
l_mut_25 = l_403;
if (l_364) {
const float l_404 = l_402 - l_400;
l_mut_25 = l_404;
}
l_arr_6[uint(7)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_405 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_405;
const uint l_406 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_406] = l_405;
const float l_407 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_407;
const uint l_408 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_408] = l_407;
const float l_409 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_409;
const uint l_410 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_410] = l_409;
const float l_411 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_411;
const uint l_412 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_412] = l_411;
const float l_413 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_413;
const uint l_414 = uint(1024) + thread_pos_in_threadgroup.x;
shared_memory_4[l_414] = l_413;
const float l_415 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_415;
const uint l_416 = uint(1280) + thread_pos_in_threadgroup.x;
shared_memory_4[l_416] = l_415;
const float l_417 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_417;
const uint l_418 = uint(1536) + thread_pos_in_threadgroup.x;
shared_memory_4[l_418] = l_417;
const float l_419 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_419;
const uint l_420 = uint(1792) + thread_pos_in_threadgroup.x;
shared_memory_4[l_420] = l_419;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_421 = thread_pos_in_threadgroup.x ^ uint(128);
const uint l_422 = thread_pos_in_threadgroup.x & uint(128);
const bool l_423 = l_422 != uint(0);
const float l_424 = l_arr_5[uint(0)];
const uint l_425 = uint(0) + l_421;
const float l_426 = shared_memory_4[l_425];
const float l_427 = l_424 + l_426;
l_mut_25 = l_427;
if (l_423) {
const float l_428 = l_426 - l_424;
l_mut_25 = l_428;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_429 = l_arr_5[uint(1)];
const uint l_430 = uint(256) + l_421;
const float l_431 = shared_memory_4[l_430];
const float l_432 = l_429 + l_431;
l_mut_25 = l_432;
if (l_423) {
const float l_433 = l_431 - l_429;
l_mut_25 = l_433;
}
l_arr_6[uint(1)] = l_mut_25;
const float l_434 = l_arr_5[uint(2)];
const uint l_435 = uint(512) + l_421;
const float l_436 = shared_memory_4[l_435];
const float l_437 = l_434 + l_436;
l_mut_25 = l_437;
if (l_423) {
const float l_438 = l_436 - l_434;
l_mut_25 = l_438;
}
l_arr_6[uint(2)] = l_mut_25;
const float l_439 = l_arr_5[uint(3)];
const uint l_440 = uint(768) + l_421;
const float l_441 = shared_memory_4[l_440];
const float l_442 = l_439 + l_441;
l_mut_25 = l_442;
if (l_423) {
const float l_443 = l_441 - l_439;
l_mut_25 = l_443;
}
l_arr_6[uint(3)] = l_mut_25;
const float l_444 = l_arr_5[uint(4)];
const uint l_445 = uint(1024) + l_421;
const float l_446 = shared_memory_4[l_445];
const float l_447 = l_444 + l_446;
l_mut_25 = l_447;
if (l_423) {
const float l_448 = l_446 - l_444;
l_mut_25 = l_448;
}
l_arr_6[uint(4)] = l_mut_25;
const float l_449 = l_arr_5[uint(5)];
const uint l_450 = uint(1280) + l_421;
const float l_451 = shared_memory_4[l_450];
const float l_452 = l_449 + l_451;
l_mut_25 = l_452;
if (l_423) {
const float l_453 = l_451 - l_449;
l_mut_25 = l_453;
}
l_arr_6[uint(5)] = l_mut_25;
const float l_454 = l_arr_5[uint(6)];
const uint l_455 = uint(1536) + l_421;
const float l_456 = shared_memory_4[l_455];
const float l_457 = l_454 + l_456;
l_mut_25 = l_457;
if (l_423) {
const float l_458 = l_456 - l_454;
l_mut_25 = l_458;
}
l_arr_6[uint(6)] = l_mut_25;
const float l_459 = l_arr_5[uint(7)];
const uint l_460 = uint(1792) + l_421;
const float l_461 = shared_memory_4[l_460];
const float l_462 = l_459 + l_461;
l_mut_25 = l_462;
if (l_423) {
const float l_463 = l_461 - l_459;
l_mut_25 = l_463;
}
l_arr_6[uint(7)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_464 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_464;
const float l_465 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_465;
const float l_466 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_466;
const float l_467 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_467;
const float l_468 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_468;
const float l_469 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_469;
const float l_470 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_470;
const float l_471 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_471;
const float l_472 = l_arr_5[uint(0)];
const float l_473 = l_arr_5[uint(1)];
const float l_474 = l_472 + l_473;
l_arr_5[uint(0)] = l_474;
const float l_475 = l_472 - l_473;
l_arr_5[uint(1)] = l_475;
const float l_476 = l_arr_5[uint(2)];
const float l_477 = l_arr_5[uint(3)];
const float l_478 = l_476 + l_477;
l_arr_5[uint(2)] = l_478;
const float l_479 = l_476 - l_477;
l_arr_5[uint(3)] = l_479;
const float l_480 = l_arr_5[uint(4)];
const float l_481 = l_arr_5[uint(5)];
const float l_482 = l_480 + l_481;
l_arr_5[uint(4)] = l_482;
const float l_483 = l_480 - l_481;
l_arr_5[uint(5)] = l_483;
const float l_484 = l_arr_5[uint(6)];
const float l_485 = l_arr_5[uint(7)];
const float l_486 = l_484 + l_485;
l_arr_5[uint(6)] = l_486;
const float l_487 = l_484 - l_485;
l_arr_5[uint(7)] = l_487;
const float l_488 = l_arr_5[uint(0)];
const float l_489 = l_arr_5[uint(2)];
const float l_490 = l_488 + l_489;
l_arr_5[uint(0)] = l_490;
const float l_491 = l_488 - l_489;
l_arr_5[uint(2)] = l_491;
const float l_492 = l_arr_5[uint(1)];
const float l_493 = l_arr_5[uint(3)];
const float l_494 = l_492 + l_493;
l_arr_5[uint(1)] = l_494;
const float l_495 = l_492 - l_493;
l_arr_5[uint(3)] = l_495;
const float l_496 = l_arr_5[uint(4)];
const float l_497 = l_arr_5[uint(6)];
const float l_498 = l_496 + l_497;
l_arr_5[uint(4)] = l_498;
const float l_499 = l_496 - l_497;
l_arr_5[uint(6)] = l_499;
const float l_500 = l_arr_5[uint(5)];
const float l_501 = l_arr_5[uint(7)];
const float l_502 = l_500 + l_501;
l_arr_5[uint(5)] = l_502;
const float l_503 = l_500 - l_501;
l_arr_5[uint(7)] = l_503;
const float l_504 = l_arr_5[uint(0)];
const float l_505 = l_arr_5[uint(4)];
const float l_506 = l_504 + l_505;
l_arr_5[uint(0)] = l_506;
const float l_507 = l_504 - l_505;
l_arr_5[uint(4)] = l_507;
const float l_508 = l_arr_5[uint(1)];
const float l_509 = l_arr_5[uint(5)];
const float l_510 = l_508 + l_509;
l_arr_5[uint(1)] = l_510;
const float l_511 = l_508 - l_509;
l_arr_5[uint(5)] = l_511;
const float l_512 = l_arr_5[uint(2)];
const float l_513 = l_arr_5[uint(6)];
const float l_514 = l_512 + l_513;
l_arr_5[uint(2)] = l_514;
const float l_515 = l_512 - l_513;
l_arr_5[uint(6)] = l_515;
const float l_516 = l_arr_5[uint(3)];
const float l_517 = l_arr_5[uint(7)];
const float l_518 = l_516 + l_517;
l_arr_5[uint(3)] = l_518;
const float l_519 = l_516 - l_517;
l_arr_5[uint(7)] = l_519;
const uint l_520 = uint(0) + thread_pos_in_threadgroup.x;
const float l_521 = l_arr_5[uint(0)];
const float l_522 = l_521 * float(0.022097086533904076);
l_mut_25 = l_522;
const bool l_523 = l_2 != uint(0);
const bool l_524 = l_1 != uint(0);
const bool l_525 = l_523 && l_524;
if (l_525) {
const uint l_526 = l_12 + l_520;
const float l_527 = buffer_1[l_526];
l_mut_25 = l_mut_25 * l_527;
}
const uint l_528 = l_10 + l_520;
const half l_529 = half(l_mut_25);
buffer_2[l_528] = l_529;
const uint l_530 = uint(256) + thread_pos_in_threadgroup.x;
const float l_531 = l_arr_5[uint(1)];
const float l_532 = l_531 * float(0.022097086533904076);
l_mut_25 = l_532;
const bool l_533 = l_2 != uint(0);
const bool l_534 = l_1 != uint(0);
const bool l_535 = l_533 && l_534;
if (l_535) {
const uint l_536 = l_12 + l_530;
const float l_537 = buffer_1[l_536];
l_mut_25 = l_mut_25 * l_537;
}
const uint l_538 = l_10 + l_530;
const half l_539 = half(l_mut_25);
buffer_2[l_538] = l_539;
const uint l_540 = uint(512) + thread_pos_in_threadgroup.x;
const float l_541 = l_arr_5[uint(2)];
const float l_542 = l_541 * float(0.022097086533904076);
l_mut_25 = l_542;
const bool l_543 = l_2 != uint(0);
const bool l_544 = l_1 != uint(0);
const bool l_545 = l_543 && l_544;
if (l_545) {
const uint l_546 = l_12 + l_540;
const float l_547 = buffer_1[l_546];
l_mut_25 = l_mut_25 * l_547;
}
const uint l_548 = l_10 + l_540;
const half l_549 = half(l_mut_25);
buffer_2[l_548] = l_549;
const uint l_550 = uint(768) + thread_pos_in_threadgroup.x;
const float l_551 = l_arr_5[uint(3)];
const float l_552 = l_551 * float(0.022097086533904076);
l_mut_25 = l_552;
const bool l_553 = l_2 != uint(0);
const bool l_554 = l_1 != uint(0);
const bool l_555 = l_553 && l_554;
if (l_555) {
const uint l_556 = l_12 + l_550;
const float l_557 = buffer_1[l_556];
l_mut_25 = l_mut_25 * l_557;
}
const uint l_558 = l_10 + l_550;
const half l_559 = half(l_mut_25);
buffer_2[l_558] = l_559;
const uint l_560 = uint(1024) + thread_pos_in_threadgroup.x;
const float l_561 = l_arr_5[uint(4)];
const float l_562 = l_561 * float(0.022097086533904076);
l_mut_25 = l_562;
const bool l_563 = l_2 != uint(0);
const bool l_564 = l_1 != uint(0);
const bool l_565 = l_563 && l_564;
if (l_565) {
const uint l_566 = l_12 + l_560;
const float l_567 = buffer_1[l_566];
l_mut_25 = l_mut_25 * l_567;
}
const uint l_568 = l_10 + l_560;
const half l_569 = half(l_mut_25);
buffer_2[l_568] = l_569;
const uint l_570 = uint(1280) + thread_pos_in_threadgroup.x;
const float l_571 = l_arr_5[uint(5)];
const float l_572 = l_571 * float(0.022097086533904076);
l_mut_25 = l_572;
const bool l_573 = l_2 != uint(0);
const bool l_574 = l_1 != uint(0);
const bool l_575 = l_573 && l_574;
if (l_575) {
const uint l_576 = l_12 + l_570;
const float l_577 = buffer_1[l_576];
l_mut_25 = l_mut_25 * l_577;
}
const uint l_578 = l_10 + l_570;
const half l_579 = half(l_mut_25);
buffer_2[l_578] = l_579;
const uint l_580 = uint(1536) + thread_pos_in_threadgroup.x;
const float l_581 = l_arr_5[uint(6)];
const float l_582 = l_581 * float(0.022097086533904076);
l_mut_25 = l_582;
const bool l_583 = l_2 != uint(0);
const bool l_584 = l_1 != uint(0);
const bool l_585 = l_583 && l_584;
if (l_585) {
const uint l_586 = l_12 + l_580;
const float l_587 = buffer_1[l_586];
l_mut_25 = l_mut_25 * l_587;
}
const uint l_588 = l_10 + l_580;
const half l_589 = half(l_mut_25);
buffer_2[l_588] = l_589;
const uint l_590 = uint(1792) + thread_pos_in_threadgroup.x;
const float l_591 = l_arr_5[uint(7)];
const float l_592 = l_591 * float(0.022097086533904076);
l_mut_25 = l_592;
const bool l_593 = l_2 != uint(0);
const bool l_594 = l_1 != uint(0);
const bool l_595 = l_593 && l_594;
if (l_595) {
const uint l_596 = l_12 + l_590;
const float l_597 = buffer_1[l_596];
l_mut_25 = l_mut_25 * l_597;
}
const uint l_598 = l_10 + l_590;
const half l_599 = half(l_mut_25);
buffer_2[l_598] = l_599;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
