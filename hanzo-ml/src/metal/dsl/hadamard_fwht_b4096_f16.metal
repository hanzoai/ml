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
threadgroup uchar dynamic_shared_mem[16384];
// Shared array size: 4096, 16384 bytes
threadgroup float* shared_memory_4 = reinterpret_cast<threadgroup float*>(&dynamic_shared_mem[0]);
float l_arr_5[16];

float l_arr_6[16];

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
const uint l_10 = l_mut_7 * uint(4096);
const uint l_11 = l_mut_7 % l_0;
const uint l_12 = l_11 * uint(4096);
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
const uint l_287 = uint(2048) + thread_pos_in_threadgroup.x;
const uint l_288 = l_10 + l_287;
const half l_289 = buffer_0[l_288];
const float l_290 = float(l_289);
l_mut_25 = l_290;
const bool l_291 = l_2 == uint(0);
const bool l_292 = l_1 != uint(0);
const bool l_293 = l_291 && l_292;
if (l_293) {
const uint l_294 = l_12 + l_287;
const float l_295 = buffer_1[l_294];
l_mut_25 = l_mut_25 * l_295;
}
const float l_296 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_297 = l_mut_25 + l_296;
l_mut_17 = l_297;
const uint l_298 = thread_pos_in_threadgroup.x & uint(1);
const bool l_299 = l_298 != uint(0);
if (l_299) {
const float l_300 = l_296 - l_mut_25;
l_mut_17 = l_300;
}
l_mut_25 = l_mut_17;
const float l_301 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_302 = l_mut_25 + l_301;
l_mut_17 = l_302;
const uint l_303 = thread_pos_in_threadgroup.x & uint(2);
const bool l_304 = l_303 != uint(0);
if (l_304) {
const float l_305 = l_301 - l_mut_25;
l_mut_17 = l_305;
}
l_mut_25 = l_mut_17;
const float l_306 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_307 = l_mut_25 + l_306;
l_mut_17 = l_307;
const uint l_308 = thread_pos_in_threadgroup.x & uint(4);
const bool l_309 = l_308 != uint(0);
if (l_309) {
const float l_310 = l_306 - l_mut_25;
l_mut_17 = l_310;
}
l_mut_25 = l_mut_17;
const float l_311 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_312 = l_mut_25 + l_311;
l_mut_17 = l_312;
const uint l_313 = thread_pos_in_threadgroup.x & uint(8);
const bool l_314 = l_313 != uint(0);
if (l_314) {
const float l_315 = l_311 - l_mut_25;
l_mut_17 = l_315;
}
l_mut_25 = l_mut_17;
const float l_316 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_317 = l_mut_25 + l_316;
l_mut_17 = l_317;
const uint l_318 = thread_pos_in_threadgroup.x & uint(16);
const bool l_319 = l_318 != uint(0);
if (l_319) {
const float l_320 = l_316 - l_mut_25;
l_mut_17 = l_320;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(8)] = l_mut_25;
const uint l_321 = uint(2304) + thread_pos_in_threadgroup.x;
const uint l_322 = l_10 + l_321;
const half l_323 = buffer_0[l_322];
const float l_324 = float(l_323);
l_mut_25 = l_324;
const bool l_325 = l_2 == uint(0);
const bool l_326 = l_1 != uint(0);
const bool l_327 = l_325 && l_326;
if (l_327) {
const uint l_328 = l_12 + l_321;
const float l_329 = buffer_1[l_328];
l_mut_25 = l_mut_25 * l_329;
}
const float l_330 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_331 = l_mut_25 + l_330;
l_mut_17 = l_331;
const uint l_332 = thread_pos_in_threadgroup.x & uint(1);
const bool l_333 = l_332 != uint(0);
if (l_333) {
const float l_334 = l_330 - l_mut_25;
l_mut_17 = l_334;
}
l_mut_25 = l_mut_17;
const float l_335 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_336 = l_mut_25 + l_335;
l_mut_17 = l_336;
const uint l_337 = thread_pos_in_threadgroup.x & uint(2);
const bool l_338 = l_337 != uint(0);
if (l_338) {
const float l_339 = l_335 - l_mut_25;
l_mut_17 = l_339;
}
l_mut_25 = l_mut_17;
const float l_340 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_341 = l_mut_25 + l_340;
l_mut_17 = l_341;
const uint l_342 = thread_pos_in_threadgroup.x & uint(4);
const bool l_343 = l_342 != uint(0);
if (l_343) {
const float l_344 = l_340 - l_mut_25;
l_mut_17 = l_344;
}
l_mut_25 = l_mut_17;
const float l_345 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_346 = l_mut_25 + l_345;
l_mut_17 = l_346;
const uint l_347 = thread_pos_in_threadgroup.x & uint(8);
const bool l_348 = l_347 != uint(0);
if (l_348) {
const float l_349 = l_345 - l_mut_25;
l_mut_17 = l_349;
}
l_mut_25 = l_mut_17;
const float l_350 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_351 = l_mut_25 + l_350;
l_mut_17 = l_351;
const uint l_352 = thread_pos_in_threadgroup.x & uint(16);
const bool l_353 = l_352 != uint(0);
if (l_353) {
const float l_354 = l_350 - l_mut_25;
l_mut_17 = l_354;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(9)] = l_mut_25;
const uint l_355 = uint(2560) + thread_pos_in_threadgroup.x;
const uint l_356 = l_10 + l_355;
const half l_357 = buffer_0[l_356];
const float l_358 = float(l_357);
l_mut_25 = l_358;
const bool l_359 = l_2 == uint(0);
const bool l_360 = l_1 != uint(0);
const bool l_361 = l_359 && l_360;
if (l_361) {
const uint l_362 = l_12 + l_355;
const float l_363 = buffer_1[l_362];
l_mut_25 = l_mut_25 * l_363;
}
const float l_364 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_365 = l_mut_25 + l_364;
l_mut_17 = l_365;
const uint l_366 = thread_pos_in_threadgroup.x & uint(1);
const bool l_367 = l_366 != uint(0);
if (l_367) {
const float l_368 = l_364 - l_mut_25;
l_mut_17 = l_368;
}
l_mut_25 = l_mut_17;
const float l_369 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_370 = l_mut_25 + l_369;
l_mut_17 = l_370;
const uint l_371 = thread_pos_in_threadgroup.x & uint(2);
const bool l_372 = l_371 != uint(0);
if (l_372) {
const float l_373 = l_369 - l_mut_25;
l_mut_17 = l_373;
}
l_mut_25 = l_mut_17;
const float l_374 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_375 = l_mut_25 + l_374;
l_mut_17 = l_375;
const uint l_376 = thread_pos_in_threadgroup.x & uint(4);
const bool l_377 = l_376 != uint(0);
if (l_377) {
const float l_378 = l_374 - l_mut_25;
l_mut_17 = l_378;
}
l_mut_25 = l_mut_17;
const float l_379 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_380 = l_mut_25 + l_379;
l_mut_17 = l_380;
const uint l_381 = thread_pos_in_threadgroup.x & uint(8);
const bool l_382 = l_381 != uint(0);
if (l_382) {
const float l_383 = l_379 - l_mut_25;
l_mut_17 = l_383;
}
l_mut_25 = l_mut_17;
const float l_384 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_385 = l_mut_25 + l_384;
l_mut_17 = l_385;
const uint l_386 = thread_pos_in_threadgroup.x & uint(16);
const bool l_387 = l_386 != uint(0);
if (l_387) {
const float l_388 = l_384 - l_mut_25;
l_mut_17 = l_388;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(10)] = l_mut_25;
const uint l_389 = uint(2816) + thread_pos_in_threadgroup.x;
const uint l_390 = l_10 + l_389;
const half l_391 = buffer_0[l_390];
const float l_392 = float(l_391);
l_mut_25 = l_392;
const bool l_393 = l_2 == uint(0);
const bool l_394 = l_1 != uint(0);
const bool l_395 = l_393 && l_394;
if (l_395) {
const uint l_396 = l_12 + l_389;
const float l_397 = buffer_1[l_396];
l_mut_25 = l_mut_25 * l_397;
}
const float l_398 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_399 = l_mut_25 + l_398;
l_mut_17 = l_399;
const uint l_400 = thread_pos_in_threadgroup.x & uint(1);
const bool l_401 = l_400 != uint(0);
if (l_401) {
const float l_402 = l_398 - l_mut_25;
l_mut_17 = l_402;
}
l_mut_25 = l_mut_17;
const float l_403 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_404 = l_mut_25 + l_403;
l_mut_17 = l_404;
const uint l_405 = thread_pos_in_threadgroup.x & uint(2);
const bool l_406 = l_405 != uint(0);
if (l_406) {
const float l_407 = l_403 - l_mut_25;
l_mut_17 = l_407;
}
l_mut_25 = l_mut_17;
const float l_408 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_409 = l_mut_25 + l_408;
l_mut_17 = l_409;
const uint l_410 = thread_pos_in_threadgroup.x & uint(4);
const bool l_411 = l_410 != uint(0);
if (l_411) {
const float l_412 = l_408 - l_mut_25;
l_mut_17 = l_412;
}
l_mut_25 = l_mut_17;
const float l_413 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_414 = l_mut_25 + l_413;
l_mut_17 = l_414;
const uint l_415 = thread_pos_in_threadgroup.x & uint(8);
const bool l_416 = l_415 != uint(0);
if (l_416) {
const float l_417 = l_413 - l_mut_25;
l_mut_17 = l_417;
}
l_mut_25 = l_mut_17;
const float l_418 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_419 = l_mut_25 + l_418;
l_mut_17 = l_419;
const uint l_420 = thread_pos_in_threadgroup.x & uint(16);
const bool l_421 = l_420 != uint(0);
if (l_421) {
const float l_422 = l_418 - l_mut_25;
l_mut_17 = l_422;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(11)] = l_mut_25;
const uint l_423 = uint(3072) + thread_pos_in_threadgroup.x;
const uint l_424 = l_10 + l_423;
const half l_425 = buffer_0[l_424];
const float l_426 = float(l_425);
l_mut_25 = l_426;
const bool l_427 = l_2 == uint(0);
const bool l_428 = l_1 != uint(0);
const bool l_429 = l_427 && l_428;
if (l_429) {
const uint l_430 = l_12 + l_423;
const float l_431 = buffer_1[l_430];
l_mut_25 = l_mut_25 * l_431;
}
const float l_432 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_433 = l_mut_25 + l_432;
l_mut_17 = l_433;
const uint l_434 = thread_pos_in_threadgroup.x & uint(1);
const bool l_435 = l_434 != uint(0);
if (l_435) {
const float l_436 = l_432 - l_mut_25;
l_mut_17 = l_436;
}
l_mut_25 = l_mut_17;
const float l_437 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_438 = l_mut_25 + l_437;
l_mut_17 = l_438;
const uint l_439 = thread_pos_in_threadgroup.x & uint(2);
const bool l_440 = l_439 != uint(0);
if (l_440) {
const float l_441 = l_437 - l_mut_25;
l_mut_17 = l_441;
}
l_mut_25 = l_mut_17;
const float l_442 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_443 = l_mut_25 + l_442;
l_mut_17 = l_443;
const uint l_444 = thread_pos_in_threadgroup.x & uint(4);
const bool l_445 = l_444 != uint(0);
if (l_445) {
const float l_446 = l_442 - l_mut_25;
l_mut_17 = l_446;
}
l_mut_25 = l_mut_17;
const float l_447 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_448 = l_mut_25 + l_447;
l_mut_17 = l_448;
const uint l_449 = thread_pos_in_threadgroup.x & uint(8);
const bool l_450 = l_449 != uint(0);
if (l_450) {
const float l_451 = l_447 - l_mut_25;
l_mut_17 = l_451;
}
l_mut_25 = l_mut_17;
const float l_452 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_453 = l_mut_25 + l_452;
l_mut_17 = l_453;
const uint l_454 = thread_pos_in_threadgroup.x & uint(16);
const bool l_455 = l_454 != uint(0);
if (l_455) {
const float l_456 = l_452 - l_mut_25;
l_mut_17 = l_456;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(12)] = l_mut_25;
const uint l_457 = uint(3328) + thread_pos_in_threadgroup.x;
const uint l_458 = l_10 + l_457;
const half l_459 = buffer_0[l_458];
const float l_460 = float(l_459);
l_mut_25 = l_460;
const bool l_461 = l_2 == uint(0);
const bool l_462 = l_1 != uint(0);
const bool l_463 = l_461 && l_462;
if (l_463) {
const uint l_464 = l_12 + l_457;
const float l_465 = buffer_1[l_464];
l_mut_25 = l_mut_25 * l_465;
}
const float l_466 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_467 = l_mut_25 + l_466;
l_mut_17 = l_467;
const uint l_468 = thread_pos_in_threadgroup.x & uint(1);
const bool l_469 = l_468 != uint(0);
if (l_469) {
const float l_470 = l_466 - l_mut_25;
l_mut_17 = l_470;
}
l_mut_25 = l_mut_17;
const float l_471 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_472 = l_mut_25 + l_471;
l_mut_17 = l_472;
const uint l_473 = thread_pos_in_threadgroup.x & uint(2);
const bool l_474 = l_473 != uint(0);
if (l_474) {
const float l_475 = l_471 - l_mut_25;
l_mut_17 = l_475;
}
l_mut_25 = l_mut_17;
const float l_476 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_477 = l_mut_25 + l_476;
l_mut_17 = l_477;
const uint l_478 = thread_pos_in_threadgroup.x & uint(4);
const bool l_479 = l_478 != uint(0);
if (l_479) {
const float l_480 = l_476 - l_mut_25;
l_mut_17 = l_480;
}
l_mut_25 = l_mut_17;
const float l_481 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_482 = l_mut_25 + l_481;
l_mut_17 = l_482;
const uint l_483 = thread_pos_in_threadgroup.x & uint(8);
const bool l_484 = l_483 != uint(0);
if (l_484) {
const float l_485 = l_481 - l_mut_25;
l_mut_17 = l_485;
}
l_mut_25 = l_mut_17;
const float l_486 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_487 = l_mut_25 + l_486;
l_mut_17 = l_487;
const uint l_488 = thread_pos_in_threadgroup.x & uint(16);
const bool l_489 = l_488 != uint(0);
if (l_489) {
const float l_490 = l_486 - l_mut_25;
l_mut_17 = l_490;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(13)] = l_mut_25;
const uint l_491 = uint(3584) + thread_pos_in_threadgroup.x;
const uint l_492 = l_10 + l_491;
const half l_493 = buffer_0[l_492];
const float l_494 = float(l_493);
l_mut_25 = l_494;
const bool l_495 = l_2 == uint(0);
const bool l_496 = l_1 != uint(0);
const bool l_497 = l_495 && l_496;
if (l_497) {
const uint l_498 = l_12 + l_491;
const float l_499 = buffer_1[l_498];
l_mut_25 = l_mut_25 * l_499;
}
const float l_500 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_501 = l_mut_25 + l_500;
l_mut_17 = l_501;
const uint l_502 = thread_pos_in_threadgroup.x & uint(1);
const bool l_503 = l_502 != uint(0);
if (l_503) {
const float l_504 = l_500 - l_mut_25;
l_mut_17 = l_504;
}
l_mut_25 = l_mut_17;
const float l_505 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_506 = l_mut_25 + l_505;
l_mut_17 = l_506;
const uint l_507 = thread_pos_in_threadgroup.x & uint(2);
const bool l_508 = l_507 != uint(0);
if (l_508) {
const float l_509 = l_505 - l_mut_25;
l_mut_17 = l_509;
}
l_mut_25 = l_mut_17;
const float l_510 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_511 = l_mut_25 + l_510;
l_mut_17 = l_511;
const uint l_512 = thread_pos_in_threadgroup.x & uint(4);
const bool l_513 = l_512 != uint(0);
if (l_513) {
const float l_514 = l_510 - l_mut_25;
l_mut_17 = l_514;
}
l_mut_25 = l_mut_17;
const float l_515 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_516 = l_mut_25 + l_515;
l_mut_17 = l_516;
const uint l_517 = thread_pos_in_threadgroup.x & uint(8);
const bool l_518 = l_517 != uint(0);
if (l_518) {
const float l_519 = l_515 - l_mut_25;
l_mut_17 = l_519;
}
l_mut_25 = l_mut_17;
const float l_520 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_521 = l_mut_25 + l_520;
l_mut_17 = l_521;
const uint l_522 = thread_pos_in_threadgroup.x & uint(16);
const bool l_523 = l_522 != uint(0);
if (l_523) {
const float l_524 = l_520 - l_mut_25;
l_mut_17 = l_524;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(14)] = l_mut_25;
const uint l_525 = uint(3840) + thread_pos_in_threadgroup.x;
const uint l_526 = l_10 + l_525;
const half l_527 = buffer_0[l_526];
const float l_528 = float(l_527);
l_mut_25 = l_528;
const bool l_529 = l_2 == uint(0);
const bool l_530 = l_1 != uint(0);
const bool l_531 = l_529 && l_530;
if (l_531) {
const uint l_532 = l_12 + l_525;
const float l_533 = buffer_1[l_532];
l_mut_25 = l_mut_25 * l_533;
}
const float l_534 = { simd_shuffle_xor(l_mut_25, uint(1)) };
const float l_535 = l_mut_25 + l_534;
l_mut_17 = l_535;
const uint l_536 = thread_pos_in_threadgroup.x & uint(1);
const bool l_537 = l_536 != uint(0);
if (l_537) {
const float l_538 = l_534 - l_mut_25;
l_mut_17 = l_538;
}
l_mut_25 = l_mut_17;
const float l_539 = { simd_shuffle_xor(l_mut_25, uint(2)) };
const float l_540 = l_mut_25 + l_539;
l_mut_17 = l_540;
const uint l_541 = thread_pos_in_threadgroup.x & uint(2);
const bool l_542 = l_541 != uint(0);
if (l_542) {
const float l_543 = l_539 - l_mut_25;
l_mut_17 = l_543;
}
l_mut_25 = l_mut_17;
const float l_544 = { simd_shuffle_xor(l_mut_25, uint(4)) };
const float l_545 = l_mut_25 + l_544;
l_mut_17 = l_545;
const uint l_546 = thread_pos_in_threadgroup.x & uint(4);
const bool l_547 = l_546 != uint(0);
if (l_547) {
const float l_548 = l_544 - l_mut_25;
l_mut_17 = l_548;
}
l_mut_25 = l_mut_17;
const float l_549 = { simd_shuffle_xor(l_mut_25, uint(8)) };
const float l_550 = l_mut_25 + l_549;
l_mut_17 = l_550;
const uint l_551 = thread_pos_in_threadgroup.x & uint(8);
const bool l_552 = l_551 != uint(0);
if (l_552) {
const float l_553 = l_549 - l_mut_25;
l_mut_17 = l_553;
}
l_mut_25 = l_mut_17;
const float l_554 = { simd_shuffle_xor(l_mut_25, uint(16)) };
const float l_555 = l_mut_25 + l_554;
l_mut_17 = l_555;
const uint l_556 = thread_pos_in_threadgroup.x & uint(16);
const bool l_557 = l_556 != uint(0);
if (l_557) {
const float l_558 = l_554 - l_mut_25;
l_mut_17 = l_558;
}
l_mut_25 = l_mut_17;
l_arr_5[uint(15)] = l_mut_25;
const uint l_559 = uint(0) + thread_pos_in_threadgroup.x;
const float l_560 = l_arr_5[uint(0)];
shared_memory_4[l_559] = l_560;
const uint l_561 = uint(256) + thread_pos_in_threadgroup.x;
const float l_562 = l_arr_5[uint(1)];
shared_memory_4[l_561] = l_562;
const uint l_563 = uint(512) + thread_pos_in_threadgroup.x;
const float l_564 = l_arr_5[uint(2)];
shared_memory_4[l_563] = l_564;
const uint l_565 = uint(768) + thread_pos_in_threadgroup.x;
const float l_566 = l_arr_5[uint(3)];
shared_memory_4[l_565] = l_566;
const uint l_567 = uint(1024) + thread_pos_in_threadgroup.x;
const float l_568 = l_arr_5[uint(4)];
shared_memory_4[l_567] = l_568;
const uint l_569 = uint(1280) + thread_pos_in_threadgroup.x;
const float l_570 = l_arr_5[uint(5)];
shared_memory_4[l_569] = l_570;
const uint l_571 = uint(1536) + thread_pos_in_threadgroup.x;
const float l_572 = l_arr_5[uint(6)];
shared_memory_4[l_571] = l_572;
const uint l_573 = uint(1792) + thread_pos_in_threadgroup.x;
const float l_574 = l_arr_5[uint(7)];
shared_memory_4[l_573] = l_574;
const uint l_575 = uint(2048) + thread_pos_in_threadgroup.x;
const float l_576 = l_arr_5[uint(8)];
shared_memory_4[l_575] = l_576;
const uint l_577 = uint(2304) + thread_pos_in_threadgroup.x;
const float l_578 = l_arr_5[uint(9)];
shared_memory_4[l_577] = l_578;
const uint l_579 = uint(2560) + thread_pos_in_threadgroup.x;
const float l_580 = l_arr_5[uint(10)];
shared_memory_4[l_579] = l_580;
const uint l_581 = uint(2816) + thread_pos_in_threadgroup.x;
const float l_582 = l_arr_5[uint(11)];
shared_memory_4[l_581] = l_582;
const uint l_583 = uint(3072) + thread_pos_in_threadgroup.x;
const float l_584 = l_arr_5[uint(12)];
shared_memory_4[l_583] = l_584;
const uint l_585 = uint(3328) + thread_pos_in_threadgroup.x;
const float l_586 = l_arr_5[uint(13)];
shared_memory_4[l_585] = l_586;
const uint l_587 = uint(3584) + thread_pos_in_threadgroup.x;
const float l_588 = l_arr_5[uint(14)];
shared_memory_4[l_587] = l_588;
const uint l_589 = uint(3840) + thread_pos_in_threadgroup.x;
const float l_590 = l_arr_5[uint(15)];
shared_memory_4[l_589] = l_590;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_591 = thread_pos_in_threadgroup.x ^ uint(32);
const uint l_592 = thread_pos_in_threadgroup.x & uint(32);
const bool l_593 = l_592 != uint(0);
const float l_594 = l_arr_5[uint(0)];
const uint l_595 = uint(0) + l_591;
const float l_596 = shared_memory_4[l_595];
const float l_597 = l_594 + l_596;
l_mut_25 = l_597;
if (l_593) {
const float l_598 = l_596 - l_594;
l_mut_25 = l_598;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_599 = l_arr_5[uint(1)];
const uint l_600 = uint(256) + l_591;
const float l_601 = shared_memory_4[l_600];
const float l_602 = l_599 + l_601;
l_mut_25 = l_602;
if (l_593) {
const float l_603 = l_601 - l_599;
l_mut_25 = l_603;
}
l_arr_6[uint(1)] = l_mut_25;
const float l_604 = l_arr_5[uint(2)];
const uint l_605 = uint(512) + l_591;
const float l_606 = shared_memory_4[l_605];
const float l_607 = l_604 + l_606;
l_mut_25 = l_607;
if (l_593) {
const float l_608 = l_606 - l_604;
l_mut_25 = l_608;
}
l_arr_6[uint(2)] = l_mut_25;
const float l_609 = l_arr_5[uint(3)];
const uint l_610 = uint(768) + l_591;
const float l_611 = shared_memory_4[l_610];
const float l_612 = l_609 + l_611;
l_mut_25 = l_612;
if (l_593) {
const float l_613 = l_611 - l_609;
l_mut_25 = l_613;
}
l_arr_6[uint(3)] = l_mut_25;
const float l_614 = l_arr_5[uint(4)];
const uint l_615 = uint(1024) + l_591;
const float l_616 = shared_memory_4[l_615];
const float l_617 = l_614 + l_616;
l_mut_25 = l_617;
if (l_593) {
const float l_618 = l_616 - l_614;
l_mut_25 = l_618;
}
l_arr_6[uint(4)] = l_mut_25;
const float l_619 = l_arr_5[uint(5)];
const uint l_620 = uint(1280) + l_591;
const float l_621 = shared_memory_4[l_620];
const float l_622 = l_619 + l_621;
l_mut_25 = l_622;
if (l_593) {
const float l_623 = l_621 - l_619;
l_mut_25 = l_623;
}
l_arr_6[uint(5)] = l_mut_25;
const float l_624 = l_arr_5[uint(6)];
const uint l_625 = uint(1536) + l_591;
const float l_626 = shared_memory_4[l_625];
const float l_627 = l_624 + l_626;
l_mut_25 = l_627;
if (l_593) {
const float l_628 = l_626 - l_624;
l_mut_25 = l_628;
}
l_arr_6[uint(6)] = l_mut_25;
const float l_629 = l_arr_5[uint(7)];
const uint l_630 = uint(1792) + l_591;
const float l_631 = shared_memory_4[l_630];
const float l_632 = l_629 + l_631;
l_mut_25 = l_632;
if (l_593) {
const float l_633 = l_631 - l_629;
l_mut_25 = l_633;
}
l_arr_6[uint(7)] = l_mut_25;
const float l_634 = l_arr_5[uint(8)];
const uint l_635 = uint(2048) + l_591;
const float l_636 = shared_memory_4[l_635];
const float l_637 = l_634 + l_636;
l_mut_25 = l_637;
if (l_593) {
const float l_638 = l_636 - l_634;
l_mut_25 = l_638;
}
l_arr_6[uint(8)] = l_mut_25;
const float l_639 = l_arr_5[uint(9)];
const uint l_640 = uint(2304) + l_591;
const float l_641 = shared_memory_4[l_640];
const float l_642 = l_639 + l_641;
l_mut_25 = l_642;
if (l_593) {
const float l_643 = l_641 - l_639;
l_mut_25 = l_643;
}
l_arr_6[uint(9)] = l_mut_25;
const float l_644 = l_arr_5[uint(10)];
const uint l_645 = uint(2560) + l_591;
const float l_646 = shared_memory_4[l_645];
const float l_647 = l_644 + l_646;
l_mut_25 = l_647;
if (l_593) {
const float l_648 = l_646 - l_644;
l_mut_25 = l_648;
}
l_arr_6[uint(10)] = l_mut_25;
const float l_649 = l_arr_5[uint(11)];
const uint l_650 = uint(2816) + l_591;
const float l_651 = shared_memory_4[l_650];
const float l_652 = l_649 + l_651;
l_mut_25 = l_652;
if (l_593) {
const float l_653 = l_651 - l_649;
l_mut_25 = l_653;
}
l_arr_6[uint(11)] = l_mut_25;
const float l_654 = l_arr_5[uint(12)];
const uint l_655 = uint(3072) + l_591;
const float l_656 = shared_memory_4[l_655];
const float l_657 = l_654 + l_656;
l_mut_25 = l_657;
if (l_593) {
const float l_658 = l_656 - l_654;
l_mut_25 = l_658;
}
l_arr_6[uint(12)] = l_mut_25;
const float l_659 = l_arr_5[uint(13)];
const uint l_660 = uint(3328) + l_591;
const float l_661 = shared_memory_4[l_660];
const float l_662 = l_659 + l_661;
l_mut_25 = l_662;
if (l_593) {
const float l_663 = l_661 - l_659;
l_mut_25 = l_663;
}
l_arr_6[uint(13)] = l_mut_25;
const float l_664 = l_arr_5[uint(14)];
const uint l_665 = uint(3584) + l_591;
const float l_666 = shared_memory_4[l_665];
const float l_667 = l_664 + l_666;
l_mut_25 = l_667;
if (l_593) {
const float l_668 = l_666 - l_664;
l_mut_25 = l_668;
}
l_arr_6[uint(14)] = l_mut_25;
const float l_669 = l_arr_5[uint(15)];
const uint l_670 = uint(3840) + l_591;
const float l_671 = shared_memory_4[l_670];
const float l_672 = l_669 + l_671;
l_mut_25 = l_672;
if (l_593) {
const float l_673 = l_671 - l_669;
l_mut_25 = l_673;
}
l_arr_6[uint(15)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_674 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_674;
const uint l_675 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_675] = l_674;
const float l_676 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_676;
const uint l_677 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_677] = l_676;
const float l_678 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_678;
const uint l_679 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_679] = l_678;
const float l_680 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_680;
const uint l_681 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_681] = l_680;
const float l_682 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_682;
const uint l_683 = uint(1024) + thread_pos_in_threadgroup.x;
shared_memory_4[l_683] = l_682;
const float l_684 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_684;
const uint l_685 = uint(1280) + thread_pos_in_threadgroup.x;
shared_memory_4[l_685] = l_684;
const float l_686 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_686;
const uint l_687 = uint(1536) + thread_pos_in_threadgroup.x;
shared_memory_4[l_687] = l_686;
const float l_688 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_688;
const uint l_689 = uint(1792) + thread_pos_in_threadgroup.x;
shared_memory_4[l_689] = l_688;
const float l_690 = l_arr_6[uint(8)];
l_arr_5[uint(8)] = l_690;
const uint l_691 = uint(2048) + thread_pos_in_threadgroup.x;
shared_memory_4[l_691] = l_690;
const float l_692 = l_arr_6[uint(9)];
l_arr_5[uint(9)] = l_692;
const uint l_693 = uint(2304) + thread_pos_in_threadgroup.x;
shared_memory_4[l_693] = l_692;
const float l_694 = l_arr_6[uint(10)];
l_arr_5[uint(10)] = l_694;
const uint l_695 = uint(2560) + thread_pos_in_threadgroup.x;
shared_memory_4[l_695] = l_694;
const float l_696 = l_arr_6[uint(11)];
l_arr_5[uint(11)] = l_696;
const uint l_697 = uint(2816) + thread_pos_in_threadgroup.x;
shared_memory_4[l_697] = l_696;
const float l_698 = l_arr_6[uint(12)];
l_arr_5[uint(12)] = l_698;
const uint l_699 = uint(3072) + thread_pos_in_threadgroup.x;
shared_memory_4[l_699] = l_698;
const float l_700 = l_arr_6[uint(13)];
l_arr_5[uint(13)] = l_700;
const uint l_701 = uint(3328) + thread_pos_in_threadgroup.x;
shared_memory_4[l_701] = l_700;
const float l_702 = l_arr_6[uint(14)];
l_arr_5[uint(14)] = l_702;
const uint l_703 = uint(3584) + thread_pos_in_threadgroup.x;
shared_memory_4[l_703] = l_702;
const float l_704 = l_arr_6[uint(15)];
l_arr_5[uint(15)] = l_704;
const uint l_705 = uint(3840) + thread_pos_in_threadgroup.x;
shared_memory_4[l_705] = l_704;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_706 = thread_pos_in_threadgroup.x ^ uint(64);
const uint l_707 = thread_pos_in_threadgroup.x & uint(64);
const bool l_708 = l_707 != uint(0);
const float l_709 = l_arr_5[uint(0)];
const uint l_710 = uint(0) + l_706;
const float l_711 = shared_memory_4[l_710];
const float l_712 = l_709 + l_711;
l_mut_25 = l_712;
if (l_708) {
const float l_713 = l_711 - l_709;
l_mut_25 = l_713;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_714 = l_arr_5[uint(1)];
const uint l_715 = uint(256) + l_706;
const float l_716 = shared_memory_4[l_715];
const float l_717 = l_714 + l_716;
l_mut_25 = l_717;
if (l_708) {
const float l_718 = l_716 - l_714;
l_mut_25 = l_718;
}
l_arr_6[uint(1)] = l_mut_25;
const float l_719 = l_arr_5[uint(2)];
const uint l_720 = uint(512) + l_706;
const float l_721 = shared_memory_4[l_720];
const float l_722 = l_719 + l_721;
l_mut_25 = l_722;
if (l_708) {
const float l_723 = l_721 - l_719;
l_mut_25 = l_723;
}
l_arr_6[uint(2)] = l_mut_25;
const float l_724 = l_arr_5[uint(3)];
const uint l_725 = uint(768) + l_706;
const float l_726 = shared_memory_4[l_725];
const float l_727 = l_724 + l_726;
l_mut_25 = l_727;
if (l_708) {
const float l_728 = l_726 - l_724;
l_mut_25 = l_728;
}
l_arr_6[uint(3)] = l_mut_25;
const float l_729 = l_arr_5[uint(4)];
const uint l_730 = uint(1024) + l_706;
const float l_731 = shared_memory_4[l_730];
const float l_732 = l_729 + l_731;
l_mut_25 = l_732;
if (l_708) {
const float l_733 = l_731 - l_729;
l_mut_25 = l_733;
}
l_arr_6[uint(4)] = l_mut_25;
const float l_734 = l_arr_5[uint(5)];
const uint l_735 = uint(1280) + l_706;
const float l_736 = shared_memory_4[l_735];
const float l_737 = l_734 + l_736;
l_mut_25 = l_737;
if (l_708) {
const float l_738 = l_736 - l_734;
l_mut_25 = l_738;
}
l_arr_6[uint(5)] = l_mut_25;
const float l_739 = l_arr_5[uint(6)];
const uint l_740 = uint(1536) + l_706;
const float l_741 = shared_memory_4[l_740];
const float l_742 = l_739 + l_741;
l_mut_25 = l_742;
if (l_708) {
const float l_743 = l_741 - l_739;
l_mut_25 = l_743;
}
l_arr_6[uint(6)] = l_mut_25;
const float l_744 = l_arr_5[uint(7)];
const uint l_745 = uint(1792) + l_706;
const float l_746 = shared_memory_4[l_745];
const float l_747 = l_744 + l_746;
l_mut_25 = l_747;
if (l_708) {
const float l_748 = l_746 - l_744;
l_mut_25 = l_748;
}
l_arr_6[uint(7)] = l_mut_25;
const float l_749 = l_arr_5[uint(8)];
const uint l_750 = uint(2048) + l_706;
const float l_751 = shared_memory_4[l_750];
const float l_752 = l_749 + l_751;
l_mut_25 = l_752;
if (l_708) {
const float l_753 = l_751 - l_749;
l_mut_25 = l_753;
}
l_arr_6[uint(8)] = l_mut_25;
const float l_754 = l_arr_5[uint(9)];
const uint l_755 = uint(2304) + l_706;
const float l_756 = shared_memory_4[l_755];
const float l_757 = l_754 + l_756;
l_mut_25 = l_757;
if (l_708) {
const float l_758 = l_756 - l_754;
l_mut_25 = l_758;
}
l_arr_6[uint(9)] = l_mut_25;
const float l_759 = l_arr_5[uint(10)];
const uint l_760 = uint(2560) + l_706;
const float l_761 = shared_memory_4[l_760];
const float l_762 = l_759 + l_761;
l_mut_25 = l_762;
if (l_708) {
const float l_763 = l_761 - l_759;
l_mut_25 = l_763;
}
l_arr_6[uint(10)] = l_mut_25;
const float l_764 = l_arr_5[uint(11)];
const uint l_765 = uint(2816) + l_706;
const float l_766 = shared_memory_4[l_765];
const float l_767 = l_764 + l_766;
l_mut_25 = l_767;
if (l_708) {
const float l_768 = l_766 - l_764;
l_mut_25 = l_768;
}
l_arr_6[uint(11)] = l_mut_25;
const float l_769 = l_arr_5[uint(12)];
const uint l_770 = uint(3072) + l_706;
const float l_771 = shared_memory_4[l_770];
const float l_772 = l_769 + l_771;
l_mut_25 = l_772;
if (l_708) {
const float l_773 = l_771 - l_769;
l_mut_25 = l_773;
}
l_arr_6[uint(12)] = l_mut_25;
const float l_774 = l_arr_5[uint(13)];
const uint l_775 = uint(3328) + l_706;
const float l_776 = shared_memory_4[l_775];
const float l_777 = l_774 + l_776;
l_mut_25 = l_777;
if (l_708) {
const float l_778 = l_776 - l_774;
l_mut_25 = l_778;
}
l_arr_6[uint(13)] = l_mut_25;
const float l_779 = l_arr_5[uint(14)];
const uint l_780 = uint(3584) + l_706;
const float l_781 = shared_memory_4[l_780];
const float l_782 = l_779 + l_781;
l_mut_25 = l_782;
if (l_708) {
const float l_783 = l_781 - l_779;
l_mut_25 = l_783;
}
l_arr_6[uint(14)] = l_mut_25;
const float l_784 = l_arr_5[uint(15)];
const uint l_785 = uint(3840) + l_706;
const float l_786 = shared_memory_4[l_785];
const float l_787 = l_784 + l_786;
l_mut_25 = l_787;
if (l_708) {
const float l_788 = l_786 - l_784;
l_mut_25 = l_788;
}
l_arr_6[uint(15)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_789 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_789;
const uint l_790 = uint(0) + thread_pos_in_threadgroup.x;
shared_memory_4[l_790] = l_789;
const float l_791 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_791;
const uint l_792 = uint(256) + thread_pos_in_threadgroup.x;
shared_memory_4[l_792] = l_791;
const float l_793 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_793;
const uint l_794 = uint(512) + thread_pos_in_threadgroup.x;
shared_memory_4[l_794] = l_793;
const float l_795 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_795;
const uint l_796 = uint(768) + thread_pos_in_threadgroup.x;
shared_memory_4[l_796] = l_795;
const float l_797 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_797;
const uint l_798 = uint(1024) + thread_pos_in_threadgroup.x;
shared_memory_4[l_798] = l_797;
const float l_799 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_799;
const uint l_800 = uint(1280) + thread_pos_in_threadgroup.x;
shared_memory_4[l_800] = l_799;
const float l_801 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_801;
const uint l_802 = uint(1536) + thread_pos_in_threadgroup.x;
shared_memory_4[l_802] = l_801;
const float l_803 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_803;
const uint l_804 = uint(1792) + thread_pos_in_threadgroup.x;
shared_memory_4[l_804] = l_803;
const float l_805 = l_arr_6[uint(8)];
l_arr_5[uint(8)] = l_805;
const uint l_806 = uint(2048) + thread_pos_in_threadgroup.x;
shared_memory_4[l_806] = l_805;
const float l_807 = l_arr_6[uint(9)];
l_arr_5[uint(9)] = l_807;
const uint l_808 = uint(2304) + thread_pos_in_threadgroup.x;
shared_memory_4[l_808] = l_807;
const float l_809 = l_arr_6[uint(10)];
l_arr_5[uint(10)] = l_809;
const uint l_810 = uint(2560) + thread_pos_in_threadgroup.x;
shared_memory_4[l_810] = l_809;
const float l_811 = l_arr_6[uint(11)];
l_arr_5[uint(11)] = l_811;
const uint l_812 = uint(2816) + thread_pos_in_threadgroup.x;
shared_memory_4[l_812] = l_811;
const float l_813 = l_arr_6[uint(12)];
l_arr_5[uint(12)] = l_813;
const uint l_814 = uint(3072) + thread_pos_in_threadgroup.x;
shared_memory_4[l_814] = l_813;
const float l_815 = l_arr_6[uint(13)];
l_arr_5[uint(13)] = l_815;
const uint l_816 = uint(3328) + thread_pos_in_threadgroup.x;
shared_memory_4[l_816] = l_815;
const float l_817 = l_arr_6[uint(14)];
l_arr_5[uint(14)] = l_817;
const uint l_818 = uint(3584) + thread_pos_in_threadgroup.x;
shared_memory_4[l_818] = l_817;
const float l_819 = l_arr_6[uint(15)];
l_arr_5[uint(15)] = l_819;
const uint l_820 = uint(3840) + thread_pos_in_threadgroup.x;
shared_memory_4[l_820] = l_819;
threadgroup_barrier(mem_flags::mem_threadgroup);
const uint l_821 = thread_pos_in_threadgroup.x ^ uint(128);
const uint l_822 = thread_pos_in_threadgroup.x & uint(128);
const bool l_823 = l_822 != uint(0);
const float l_824 = l_arr_5[uint(0)];
const uint l_825 = uint(0) + l_821;
const float l_826 = shared_memory_4[l_825];
const float l_827 = l_824 + l_826;
l_mut_25 = l_827;
if (l_823) {
const float l_828 = l_826 - l_824;
l_mut_25 = l_828;
}
l_arr_6[uint(0)] = l_mut_25;
const float l_829 = l_arr_5[uint(1)];
const uint l_830 = uint(256) + l_821;
const float l_831 = shared_memory_4[l_830];
const float l_832 = l_829 + l_831;
l_mut_25 = l_832;
if (l_823) {
const float l_833 = l_831 - l_829;
l_mut_25 = l_833;
}
l_arr_6[uint(1)] = l_mut_25;
const float l_834 = l_arr_5[uint(2)];
const uint l_835 = uint(512) + l_821;
const float l_836 = shared_memory_4[l_835];
const float l_837 = l_834 + l_836;
l_mut_25 = l_837;
if (l_823) {
const float l_838 = l_836 - l_834;
l_mut_25 = l_838;
}
l_arr_6[uint(2)] = l_mut_25;
const float l_839 = l_arr_5[uint(3)];
const uint l_840 = uint(768) + l_821;
const float l_841 = shared_memory_4[l_840];
const float l_842 = l_839 + l_841;
l_mut_25 = l_842;
if (l_823) {
const float l_843 = l_841 - l_839;
l_mut_25 = l_843;
}
l_arr_6[uint(3)] = l_mut_25;
const float l_844 = l_arr_5[uint(4)];
const uint l_845 = uint(1024) + l_821;
const float l_846 = shared_memory_4[l_845];
const float l_847 = l_844 + l_846;
l_mut_25 = l_847;
if (l_823) {
const float l_848 = l_846 - l_844;
l_mut_25 = l_848;
}
l_arr_6[uint(4)] = l_mut_25;
const float l_849 = l_arr_5[uint(5)];
const uint l_850 = uint(1280) + l_821;
const float l_851 = shared_memory_4[l_850];
const float l_852 = l_849 + l_851;
l_mut_25 = l_852;
if (l_823) {
const float l_853 = l_851 - l_849;
l_mut_25 = l_853;
}
l_arr_6[uint(5)] = l_mut_25;
const float l_854 = l_arr_5[uint(6)];
const uint l_855 = uint(1536) + l_821;
const float l_856 = shared_memory_4[l_855];
const float l_857 = l_854 + l_856;
l_mut_25 = l_857;
if (l_823) {
const float l_858 = l_856 - l_854;
l_mut_25 = l_858;
}
l_arr_6[uint(6)] = l_mut_25;
const float l_859 = l_arr_5[uint(7)];
const uint l_860 = uint(1792) + l_821;
const float l_861 = shared_memory_4[l_860];
const float l_862 = l_859 + l_861;
l_mut_25 = l_862;
if (l_823) {
const float l_863 = l_861 - l_859;
l_mut_25 = l_863;
}
l_arr_6[uint(7)] = l_mut_25;
const float l_864 = l_arr_5[uint(8)];
const uint l_865 = uint(2048) + l_821;
const float l_866 = shared_memory_4[l_865];
const float l_867 = l_864 + l_866;
l_mut_25 = l_867;
if (l_823) {
const float l_868 = l_866 - l_864;
l_mut_25 = l_868;
}
l_arr_6[uint(8)] = l_mut_25;
const float l_869 = l_arr_5[uint(9)];
const uint l_870 = uint(2304) + l_821;
const float l_871 = shared_memory_4[l_870];
const float l_872 = l_869 + l_871;
l_mut_25 = l_872;
if (l_823) {
const float l_873 = l_871 - l_869;
l_mut_25 = l_873;
}
l_arr_6[uint(9)] = l_mut_25;
const float l_874 = l_arr_5[uint(10)];
const uint l_875 = uint(2560) + l_821;
const float l_876 = shared_memory_4[l_875];
const float l_877 = l_874 + l_876;
l_mut_25 = l_877;
if (l_823) {
const float l_878 = l_876 - l_874;
l_mut_25 = l_878;
}
l_arr_6[uint(10)] = l_mut_25;
const float l_879 = l_arr_5[uint(11)];
const uint l_880 = uint(2816) + l_821;
const float l_881 = shared_memory_4[l_880];
const float l_882 = l_879 + l_881;
l_mut_25 = l_882;
if (l_823) {
const float l_883 = l_881 - l_879;
l_mut_25 = l_883;
}
l_arr_6[uint(11)] = l_mut_25;
const float l_884 = l_arr_5[uint(12)];
const uint l_885 = uint(3072) + l_821;
const float l_886 = shared_memory_4[l_885];
const float l_887 = l_884 + l_886;
l_mut_25 = l_887;
if (l_823) {
const float l_888 = l_886 - l_884;
l_mut_25 = l_888;
}
l_arr_6[uint(12)] = l_mut_25;
const float l_889 = l_arr_5[uint(13)];
const uint l_890 = uint(3328) + l_821;
const float l_891 = shared_memory_4[l_890];
const float l_892 = l_889 + l_891;
l_mut_25 = l_892;
if (l_823) {
const float l_893 = l_891 - l_889;
l_mut_25 = l_893;
}
l_arr_6[uint(13)] = l_mut_25;
const float l_894 = l_arr_5[uint(14)];
const uint l_895 = uint(3584) + l_821;
const float l_896 = shared_memory_4[l_895];
const float l_897 = l_894 + l_896;
l_mut_25 = l_897;
if (l_823) {
const float l_898 = l_896 - l_894;
l_mut_25 = l_898;
}
l_arr_6[uint(14)] = l_mut_25;
const float l_899 = l_arr_5[uint(15)];
const uint l_900 = uint(3840) + l_821;
const float l_901 = shared_memory_4[l_900];
const float l_902 = l_899 + l_901;
l_mut_25 = l_902;
if (l_823) {
const float l_903 = l_901 - l_899;
l_mut_25 = l_903;
}
l_arr_6[uint(15)] = l_mut_25;
threadgroup_barrier(mem_flags::mem_threadgroup);
const float l_904 = l_arr_6[uint(0)];
l_arr_5[uint(0)] = l_904;
const float l_905 = l_arr_6[uint(1)];
l_arr_5[uint(1)] = l_905;
const float l_906 = l_arr_6[uint(2)];
l_arr_5[uint(2)] = l_906;
const float l_907 = l_arr_6[uint(3)];
l_arr_5[uint(3)] = l_907;
const float l_908 = l_arr_6[uint(4)];
l_arr_5[uint(4)] = l_908;
const float l_909 = l_arr_6[uint(5)];
l_arr_5[uint(5)] = l_909;
const float l_910 = l_arr_6[uint(6)];
l_arr_5[uint(6)] = l_910;
const float l_911 = l_arr_6[uint(7)];
l_arr_5[uint(7)] = l_911;
const float l_912 = l_arr_6[uint(8)];
l_arr_5[uint(8)] = l_912;
const float l_913 = l_arr_6[uint(9)];
l_arr_5[uint(9)] = l_913;
const float l_914 = l_arr_6[uint(10)];
l_arr_5[uint(10)] = l_914;
const float l_915 = l_arr_6[uint(11)];
l_arr_5[uint(11)] = l_915;
const float l_916 = l_arr_6[uint(12)];
l_arr_5[uint(12)] = l_916;
const float l_917 = l_arr_6[uint(13)];
l_arr_5[uint(13)] = l_917;
const float l_918 = l_arr_6[uint(14)];
l_arr_5[uint(14)] = l_918;
const float l_919 = l_arr_6[uint(15)];
l_arr_5[uint(15)] = l_919;
const float l_920 = l_arr_5[uint(0)];
const float l_921 = l_arr_5[uint(1)];
const float l_922 = l_920 + l_921;
l_arr_5[uint(0)] = l_922;
const float l_923 = l_920 - l_921;
l_arr_5[uint(1)] = l_923;
const float l_924 = l_arr_5[uint(2)];
const float l_925 = l_arr_5[uint(3)];
const float l_926 = l_924 + l_925;
l_arr_5[uint(2)] = l_926;
const float l_927 = l_924 - l_925;
l_arr_5[uint(3)] = l_927;
const float l_928 = l_arr_5[uint(4)];
const float l_929 = l_arr_5[uint(5)];
const float l_930 = l_928 + l_929;
l_arr_5[uint(4)] = l_930;
const float l_931 = l_928 - l_929;
l_arr_5[uint(5)] = l_931;
const float l_932 = l_arr_5[uint(6)];
const float l_933 = l_arr_5[uint(7)];
const float l_934 = l_932 + l_933;
l_arr_5[uint(6)] = l_934;
const float l_935 = l_932 - l_933;
l_arr_5[uint(7)] = l_935;
const float l_936 = l_arr_5[uint(8)];
const float l_937 = l_arr_5[uint(9)];
const float l_938 = l_936 + l_937;
l_arr_5[uint(8)] = l_938;
const float l_939 = l_936 - l_937;
l_arr_5[uint(9)] = l_939;
const float l_940 = l_arr_5[uint(10)];
const float l_941 = l_arr_5[uint(11)];
const float l_942 = l_940 + l_941;
l_arr_5[uint(10)] = l_942;
const float l_943 = l_940 - l_941;
l_arr_5[uint(11)] = l_943;
const float l_944 = l_arr_5[uint(12)];
const float l_945 = l_arr_5[uint(13)];
const float l_946 = l_944 + l_945;
l_arr_5[uint(12)] = l_946;
const float l_947 = l_944 - l_945;
l_arr_5[uint(13)] = l_947;
const float l_948 = l_arr_5[uint(14)];
const float l_949 = l_arr_5[uint(15)];
const float l_950 = l_948 + l_949;
l_arr_5[uint(14)] = l_950;
const float l_951 = l_948 - l_949;
l_arr_5[uint(15)] = l_951;
const float l_952 = l_arr_5[uint(0)];
const float l_953 = l_arr_5[uint(2)];
const float l_954 = l_952 + l_953;
l_arr_5[uint(0)] = l_954;
const float l_955 = l_952 - l_953;
l_arr_5[uint(2)] = l_955;
const float l_956 = l_arr_5[uint(1)];
const float l_957 = l_arr_5[uint(3)];
const float l_958 = l_956 + l_957;
l_arr_5[uint(1)] = l_958;
const float l_959 = l_956 - l_957;
l_arr_5[uint(3)] = l_959;
const float l_960 = l_arr_5[uint(4)];
const float l_961 = l_arr_5[uint(6)];
const float l_962 = l_960 + l_961;
l_arr_5[uint(4)] = l_962;
const float l_963 = l_960 - l_961;
l_arr_5[uint(6)] = l_963;
const float l_964 = l_arr_5[uint(5)];
const float l_965 = l_arr_5[uint(7)];
const float l_966 = l_964 + l_965;
l_arr_5[uint(5)] = l_966;
const float l_967 = l_964 - l_965;
l_arr_5[uint(7)] = l_967;
const float l_968 = l_arr_5[uint(8)];
const float l_969 = l_arr_5[uint(10)];
const float l_970 = l_968 + l_969;
l_arr_5[uint(8)] = l_970;
const float l_971 = l_968 - l_969;
l_arr_5[uint(10)] = l_971;
const float l_972 = l_arr_5[uint(9)];
const float l_973 = l_arr_5[uint(11)];
const float l_974 = l_972 + l_973;
l_arr_5[uint(9)] = l_974;
const float l_975 = l_972 - l_973;
l_arr_5[uint(11)] = l_975;
const float l_976 = l_arr_5[uint(12)];
const float l_977 = l_arr_5[uint(14)];
const float l_978 = l_976 + l_977;
l_arr_5[uint(12)] = l_978;
const float l_979 = l_976 - l_977;
l_arr_5[uint(14)] = l_979;
const float l_980 = l_arr_5[uint(13)];
const float l_981 = l_arr_5[uint(15)];
const float l_982 = l_980 + l_981;
l_arr_5[uint(13)] = l_982;
const float l_983 = l_980 - l_981;
l_arr_5[uint(15)] = l_983;
const float l_984 = l_arr_5[uint(0)];
const float l_985 = l_arr_5[uint(4)];
const float l_986 = l_984 + l_985;
l_arr_5[uint(0)] = l_986;
const float l_987 = l_984 - l_985;
l_arr_5[uint(4)] = l_987;
const float l_988 = l_arr_5[uint(1)];
const float l_989 = l_arr_5[uint(5)];
const float l_990 = l_988 + l_989;
l_arr_5[uint(1)] = l_990;
const float l_991 = l_988 - l_989;
l_arr_5[uint(5)] = l_991;
const float l_992 = l_arr_5[uint(2)];
const float l_993 = l_arr_5[uint(6)];
const float l_994 = l_992 + l_993;
l_arr_5[uint(2)] = l_994;
const float l_995 = l_992 - l_993;
l_arr_5[uint(6)] = l_995;
const float l_996 = l_arr_5[uint(3)];
const float l_997 = l_arr_5[uint(7)];
const float l_998 = l_996 + l_997;
l_arr_5[uint(3)] = l_998;
const float l_999 = l_996 - l_997;
l_arr_5[uint(7)] = l_999;
const float l_1000 = l_arr_5[uint(8)];
const float l_1001 = l_arr_5[uint(12)];
const float l_1002 = l_1000 + l_1001;
l_arr_5[uint(8)] = l_1002;
const float l_1003 = l_1000 - l_1001;
l_arr_5[uint(12)] = l_1003;
const float l_1004 = l_arr_5[uint(9)];
const float l_1005 = l_arr_5[uint(13)];
const float l_1006 = l_1004 + l_1005;
l_arr_5[uint(9)] = l_1006;
const float l_1007 = l_1004 - l_1005;
l_arr_5[uint(13)] = l_1007;
const float l_1008 = l_arr_5[uint(10)];
const float l_1009 = l_arr_5[uint(14)];
const float l_1010 = l_1008 + l_1009;
l_arr_5[uint(10)] = l_1010;
const float l_1011 = l_1008 - l_1009;
l_arr_5[uint(14)] = l_1011;
const float l_1012 = l_arr_5[uint(11)];
const float l_1013 = l_arr_5[uint(15)];
const float l_1014 = l_1012 + l_1013;
l_arr_5[uint(11)] = l_1014;
const float l_1015 = l_1012 - l_1013;
l_arr_5[uint(15)] = l_1015;
const float l_1016 = l_arr_5[uint(0)];
const float l_1017 = l_arr_5[uint(8)];
const float l_1018 = l_1016 + l_1017;
l_arr_5[uint(0)] = l_1018;
const float l_1019 = l_1016 - l_1017;
l_arr_5[uint(8)] = l_1019;
const float l_1020 = l_arr_5[uint(1)];
const float l_1021 = l_arr_5[uint(9)];
const float l_1022 = l_1020 + l_1021;
l_arr_5[uint(1)] = l_1022;
const float l_1023 = l_1020 - l_1021;
l_arr_5[uint(9)] = l_1023;
const float l_1024 = l_arr_5[uint(2)];
const float l_1025 = l_arr_5[uint(10)];
const float l_1026 = l_1024 + l_1025;
l_arr_5[uint(2)] = l_1026;
const float l_1027 = l_1024 - l_1025;
l_arr_5[uint(10)] = l_1027;
const float l_1028 = l_arr_5[uint(3)];
const float l_1029 = l_arr_5[uint(11)];
const float l_1030 = l_1028 + l_1029;
l_arr_5[uint(3)] = l_1030;
const float l_1031 = l_1028 - l_1029;
l_arr_5[uint(11)] = l_1031;
const float l_1032 = l_arr_5[uint(4)];
const float l_1033 = l_arr_5[uint(12)];
const float l_1034 = l_1032 + l_1033;
l_arr_5[uint(4)] = l_1034;
const float l_1035 = l_1032 - l_1033;
l_arr_5[uint(12)] = l_1035;
const float l_1036 = l_arr_5[uint(5)];
const float l_1037 = l_arr_5[uint(13)];
const float l_1038 = l_1036 + l_1037;
l_arr_5[uint(5)] = l_1038;
const float l_1039 = l_1036 - l_1037;
l_arr_5[uint(13)] = l_1039;
const float l_1040 = l_arr_5[uint(6)];
const float l_1041 = l_arr_5[uint(14)];
const float l_1042 = l_1040 + l_1041;
l_arr_5[uint(6)] = l_1042;
const float l_1043 = l_1040 - l_1041;
l_arr_5[uint(14)] = l_1043;
const float l_1044 = l_arr_5[uint(7)];
const float l_1045 = l_arr_5[uint(15)];
const float l_1046 = l_1044 + l_1045;
l_arr_5[uint(7)] = l_1046;
const float l_1047 = l_1044 - l_1045;
l_arr_5[uint(15)] = l_1047;
const uint l_1048 = uint(0) + thread_pos_in_threadgroup.x;
const float l_1049 = l_arr_5[uint(0)];
const float l_1050 = l_1049 * float(0.015625);
l_mut_25 = l_1050;
const bool l_1051 = l_2 != uint(0);
const bool l_1052 = l_1 != uint(0);
const bool l_1053 = l_1051 && l_1052;
if (l_1053) {
const uint l_1054 = l_12 + l_1048;
const float l_1055 = buffer_1[l_1054];
l_mut_25 = l_mut_25 * l_1055;
}
const uint l_1056 = l_10 + l_1048;
const half l_1057 = half(l_mut_25);
buffer_2[l_1056] = l_1057;
const uint l_1058 = uint(256) + thread_pos_in_threadgroup.x;
const float l_1059 = l_arr_5[uint(1)];
const float l_1060 = l_1059 * float(0.015625);
l_mut_25 = l_1060;
const bool l_1061 = l_2 != uint(0);
const bool l_1062 = l_1 != uint(0);
const bool l_1063 = l_1061 && l_1062;
if (l_1063) {
const uint l_1064 = l_12 + l_1058;
const float l_1065 = buffer_1[l_1064];
l_mut_25 = l_mut_25 * l_1065;
}
const uint l_1066 = l_10 + l_1058;
const half l_1067 = half(l_mut_25);
buffer_2[l_1066] = l_1067;
const uint l_1068 = uint(512) + thread_pos_in_threadgroup.x;
const float l_1069 = l_arr_5[uint(2)];
const float l_1070 = l_1069 * float(0.015625);
l_mut_25 = l_1070;
const bool l_1071 = l_2 != uint(0);
const bool l_1072 = l_1 != uint(0);
const bool l_1073 = l_1071 && l_1072;
if (l_1073) {
const uint l_1074 = l_12 + l_1068;
const float l_1075 = buffer_1[l_1074];
l_mut_25 = l_mut_25 * l_1075;
}
const uint l_1076 = l_10 + l_1068;
const half l_1077 = half(l_mut_25);
buffer_2[l_1076] = l_1077;
const uint l_1078 = uint(768) + thread_pos_in_threadgroup.x;
const float l_1079 = l_arr_5[uint(3)];
const float l_1080 = l_1079 * float(0.015625);
l_mut_25 = l_1080;
const bool l_1081 = l_2 != uint(0);
const bool l_1082 = l_1 != uint(0);
const bool l_1083 = l_1081 && l_1082;
if (l_1083) {
const uint l_1084 = l_12 + l_1078;
const float l_1085 = buffer_1[l_1084];
l_mut_25 = l_mut_25 * l_1085;
}
const uint l_1086 = l_10 + l_1078;
const half l_1087 = half(l_mut_25);
buffer_2[l_1086] = l_1087;
const uint l_1088 = uint(1024) + thread_pos_in_threadgroup.x;
const float l_1089 = l_arr_5[uint(4)];
const float l_1090 = l_1089 * float(0.015625);
l_mut_25 = l_1090;
const bool l_1091 = l_2 != uint(0);
const bool l_1092 = l_1 != uint(0);
const bool l_1093 = l_1091 && l_1092;
if (l_1093) {
const uint l_1094 = l_12 + l_1088;
const float l_1095 = buffer_1[l_1094];
l_mut_25 = l_mut_25 * l_1095;
}
const uint l_1096 = l_10 + l_1088;
const half l_1097 = half(l_mut_25);
buffer_2[l_1096] = l_1097;
const uint l_1098 = uint(1280) + thread_pos_in_threadgroup.x;
const float l_1099 = l_arr_5[uint(5)];
const float l_1100 = l_1099 * float(0.015625);
l_mut_25 = l_1100;
const bool l_1101 = l_2 != uint(0);
const bool l_1102 = l_1 != uint(0);
const bool l_1103 = l_1101 && l_1102;
if (l_1103) {
const uint l_1104 = l_12 + l_1098;
const float l_1105 = buffer_1[l_1104];
l_mut_25 = l_mut_25 * l_1105;
}
const uint l_1106 = l_10 + l_1098;
const half l_1107 = half(l_mut_25);
buffer_2[l_1106] = l_1107;
const uint l_1108 = uint(1536) + thread_pos_in_threadgroup.x;
const float l_1109 = l_arr_5[uint(6)];
const float l_1110 = l_1109 * float(0.015625);
l_mut_25 = l_1110;
const bool l_1111 = l_2 != uint(0);
const bool l_1112 = l_1 != uint(0);
const bool l_1113 = l_1111 && l_1112;
if (l_1113) {
const uint l_1114 = l_12 + l_1108;
const float l_1115 = buffer_1[l_1114];
l_mut_25 = l_mut_25 * l_1115;
}
const uint l_1116 = l_10 + l_1108;
const half l_1117 = half(l_mut_25);
buffer_2[l_1116] = l_1117;
const uint l_1118 = uint(1792) + thread_pos_in_threadgroup.x;
const float l_1119 = l_arr_5[uint(7)];
const float l_1120 = l_1119 * float(0.015625);
l_mut_25 = l_1120;
const bool l_1121 = l_2 != uint(0);
const bool l_1122 = l_1 != uint(0);
const bool l_1123 = l_1121 && l_1122;
if (l_1123) {
const uint l_1124 = l_12 + l_1118;
const float l_1125 = buffer_1[l_1124];
l_mut_25 = l_mut_25 * l_1125;
}
const uint l_1126 = l_10 + l_1118;
const half l_1127 = half(l_mut_25);
buffer_2[l_1126] = l_1127;
const uint l_1128 = uint(2048) + thread_pos_in_threadgroup.x;
const float l_1129 = l_arr_5[uint(8)];
const float l_1130 = l_1129 * float(0.015625);
l_mut_25 = l_1130;
const bool l_1131 = l_2 != uint(0);
const bool l_1132 = l_1 != uint(0);
const bool l_1133 = l_1131 && l_1132;
if (l_1133) {
const uint l_1134 = l_12 + l_1128;
const float l_1135 = buffer_1[l_1134];
l_mut_25 = l_mut_25 * l_1135;
}
const uint l_1136 = l_10 + l_1128;
const half l_1137 = half(l_mut_25);
buffer_2[l_1136] = l_1137;
const uint l_1138 = uint(2304) + thread_pos_in_threadgroup.x;
const float l_1139 = l_arr_5[uint(9)];
const float l_1140 = l_1139 * float(0.015625);
l_mut_25 = l_1140;
const bool l_1141 = l_2 != uint(0);
const bool l_1142 = l_1 != uint(0);
const bool l_1143 = l_1141 && l_1142;
if (l_1143) {
const uint l_1144 = l_12 + l_1138;
const float l_1145 = buffer_1[l_1144];
l_mut_25 = l_mut_25 * l_1145;
}
const uint l_1146 = l_10 + l_1138;
const half l_1147 = half(l_mut_25);
buffer_2[l_1146] = l_1147;
const uint l_1148 = uint(2560) + thread_pos_in_threadgroup.x;
const float l_1149 = l_arr_5[uint(10)];
const float l_1150 = l_1149 * float(0.015625);
l_mut_25 = l_1150;
const bool l_1151 = l_2 != uint(0);
const bool l_1152 = l_1 != uint(0);
const bool l_1153 = l_1151 && l_1152;
if (l_1153) {
const uint l_1154 = l_12 + l_1148;
const float l_1155 = buffer_1[l_1154];
l_mut_25 = l_mut_25 * l_1155;
}
const uint l_1156 = l_10 + l_1148;
const half l_1157 = half(l_mut_25);
buffer_2[l_1156] = l_1157;
const uint l_1158 = uint(2816) + thread_pos_in_threadgroup.x;
const float l_1159 = l_arr_5[uint(11)];
const float l_1160 = l_1159 * float(0.015625);
l_mut_25 = l_1160;
const bool l_1161 = l_2 != uint(0);
const bool l_1162 = l_1 != uint(0);
const bool l_1163 = l_1161 && l_1162;
if (l_1163) {
const uint l_1164 = l_12 + l_1158;
const float l_1165 = buffer_1[l_1164];
l_mut_25 = l_mut_25 * l_1165;
}
const uint l_1166 = l_10 + l_1158;
const half l_1167 = half(l_mut_25);
buffer_2[l_1166] = l_1167;
const uint l_1168 = uint(3072) + thread_pos_in_threadgroup.x;
const float l_1169 = l_arr_5[uint(12)];
const float l_1170 = l_1169 * float(0.015625);
l_mut_25 = l_1170;
const bool l_1171 = l_2 != uint(0);
const bool l_1172 = l_1 != uint(0);
const bool l_1173 = l_1171 && l_1172;
if (l_1173) {
const uint l_1174 = l_12 + l_1168;
const float l_1175 = buffer_1[l_1174];
l_mut_25 = l_mut_25 * l_1175;
}
const uint l_1176 = l_10 + l_1168;
const half l_1177 = half(l_mut_25);
buffer_2[l_1176] = l_1177;
const uint l_1178 = uint(3328) + thread_pos_in_threadgroup.x;
const float l_1179 = l_arr_5[uint(13)];
const float l_1180 = l_1179 * float(0.015625);
l_mut_25 = l_1180;
const bool l_1181 = l_2 != uint(0);
const bool l_1182 = l_1 != uint(0);
const bool l_1183 = l_1181 && l_1182;
if (l_1183) {
const uint l_1184 = l_12 + l_1178;
const float l_1185 = buffer_1[l_1184];
l_mut_25 = l_mut_25 * l_1185;
}
const uint l_1186 = l_10 + l_1178;
const half l_1187 = half(l_mut_25);
buffer_2[l_1186] = l_1187;
const uint l_1188 = uint(3584) + thread_pos_in_threadgroup.x;
const float l_1189 = l_arr_5[uint(14)];
const float l_1190 = l_1189 * float(0.015625);
l_mut_25 = l_1190;
const bool l_1191 = l_2 != uint(0);
const bool l_1192 = l_1 != uint(0);
const bool l_1193 = l_1191 && l_1192;
if (l_1193) {
const uint l_1194 = l_12 + l_1188;
const float l_1195 = buffer_1[l_1194];
l_mut_25 = l_mut_25 * l_1195;
}
const uint l_1196 = l_10 + l_1188;
const half l_1197 = half(l_mut_25);
buffer_2[l_1196] = l_1197;
const uint l_1198 = uint(3840) + thread_pos_in_threadgroup.x;
const float l_1199 = l_arr_5[uint(15)];
const float l_1200 = l_1199 * float(0.015625);
l_mut_25 = l_1200;
const bool l_1201 = l_2 != uint(0);
const bool l_1202 = l_1 != uint(0);
const bool l_1203 = l_1201 && l_1202;
if (l_1203) {
const uint l_1204 = l_12 + l_1198;
const float l_1205 = buffer_1[l_1204];
l_mut_25 = l_mut_25 * l_1205;
}
const uint l_1206 = l_10 + l_1198;
const half l_1207 = half(l_mut_25);
buffer_2[l_1206] = l_1207;
l_mut_7 = l_mut_7 + threadgroups_per_grid.x;
}

}
