#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
typedef unsigned int uint;
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long int uint64;
typedef signed char int8;
typedef signed short int16;
typedef signed int int32;
typedef signed long long int int64;

struct __align__(16) float_4 {
    float i_0;
    float i_1;
    float i_2;
    float i_3;
};

struct __align__(4) __half_2 {
    __half i_0;
    __half i_1;
};

struct info_st {
    uint32 static_meta[8];
};


extern "C" __global__ void __launch_bounds__(128)matmul_pq2_0_gguf_f_f32 (
    const uint32* __restrict__ buffer_0, const float_4* __restrict__ buffer_1, float* buffer_2, const uint32* __restrict__ buffer_3, const __grid_constant__ info_st info
) {
uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
extern __shared__ __align__(4) uint8 dynamic_shared_mem[];
// Shared array size: 512, 2048 bytes
float *shared_memory_38 = reinterpret_cast<float*>(&dynamic_shared_mem[0]);
// Shared array size: 2560, 5120 bytes
__half *shared_memory_41 = reinterpret_cast<__half*>(&dynamic_shared_mem[2048]);
// Shared array size: 2560, 5120 bytes
__half *shared_memory_42 = reinterpret_cast<__half*>(&dynamic_shared_mem[7168]);
__half l_arr_39[16];

__half l_arr_40[16];

nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_48;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_49;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_50;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_51;
uint32 l_mut_16;
uint32 l_mut_23;
__half l_mut_54;
uint32 l_mut_55;
const uint32 l_0 = threadIdxGlobal / uint32(32);
const uint32 l_1 = l_0 / uint32(2);
const uint32 l_2 = l_0 % uint32(2);
const uint32 l_3 = buffer_3[uint32(0)];
const uint32 l_4 = buffer_3[uint32(1)];
const uint32 l_5 = buffer_3[uint32(2)];
const uint32 l_6 = blockIdx.x * uint32(64);
const uint32 l_7 = blockIdx.y * uint32(64);
const uint32 l_8 = l_5 / uint32(128);
const uint32 l_9 = l_5 / uint32(4);
const uint32 l_10 = threadIdxGlobal % uint32(4);
const uint32 l_11 = l_10 * uint32(8);
const uint32 l_12 = threadIdxGlobal / uint32(4);
const uint32 l_13 = l_6 + l_12;
const bool l_14 = l_13 < l_3;
if (l_14) {
const uint32 l_15 = l_6 + l_12;
l_mut_16 = l_15;
} else {
const uint32 l_17 = l_3 - uint32(1);
l_mut_16 = l_17;
}
const uint32 l_18 = l_6 + uint32(32);
const uint32 l_19 = l_18 + l_12;
const bool l_20 = l_19 < l_3;
if (l_20) {
const uint32 l_21 = l_6 + uint32(32);
const uint32 l_22 = l_21 + l_12;
l_mut_23 = l_22;
} else {
const uint32 l_24 = l_3 - uint32(1);
l_mut_23 = l_24;
}
const uint32 l_25 = l_mut_16 * l_9;
const uint32 l_26 = l_10 * uint32(2);
const uint32 l_27 = l_25 + l_26;
const uint32 l_28 = l_mut_23 * l_9;
const uint32 l_29 = l_10 * uint32(2);
const uint32 l_30 = l_28 + l_29;
const uint32 l_31 = threadIdxGlobal / uint32(2);
const uint32 l_32 = threadIdxGlobal % uint32(2);
const uint32 l_33 = l_7 + l_31;
const bool l_34 = l_33 < l_4;
if (l_34) {
const uint32 l_35 = l_7 + l_31;
l_mut_23 = l_35;
} else {
const uint32 l_36 = l_4 - uint32(1);
l_mut_23 = l_36;
}
const uint32 l_37 = l_mut_23 * l_8;
const uint32 l_43 = l_12 * uint32(40);
const uint32 l_44 = l_43 + l_11;
const uint32 l_45 = l_31 * uint32(40);
const uint32 l_46 = l_32 * uint32(16);
const uint32 l_47 = l_45 + l_46;
nvcuda::wmma::fill_fragment(frag_acc_48, float(0.0));
nvcuda::wmma::fill_fragment(frag_acc_49, float(0.0));
nvcuda::wmma::fill_fragment(frag_acc_50, float(0.0));
nvcuda::wmma::fill_fragment(frag_acc_51, float(0.0));

for (uint32 l_mut_52 = uint32(0); l_mut_52 < l_8; ++l_mut_52) {
const __half l_53 = __half(float(0.0));
l_mut_54 = l_53;
l_mut_23 = uint32(0);
l_mut_16 = uint32(0);
l_mut_55 = uint32(0);
const bool l_56 = threadIdxGlobal < uint32(128);
const bool l_57 = bool(true) || l_56;
if (l_57) {
const uint32 l_58 = l_37 + l_mut_52;
const uint32 l_59 = l_58 / uint32(2);
const uint32 l_60 = l_58 % uint32(2);
const uint32 l_61 = uint32(17) * l_59;
const uint32 l_62 = uint32(8) * l_60;
const uint32 l_63 = l_61 + l_62;
const uint32 l_64 = buffer_0[l_63];
const uint32 l_65 = uint32(16) * l_60;
const uint32 l_66 = l_64 >> l_65;
const uint32 l_67 = l_66 & uint32(65535);
const __half_2 l_68 = reinterpret_cast<__half_2 const&>(l_67);
const __half l_69 = l_68.i_0;
l_mut_54 = l_69;
const uint32 l_70 = uint32(17) * l_59;
const uint32 l_71 = uint32(9) * l_60;
const uint32 l_72 = l_70 + l_71;
l_mut_23 = l_72;
const uint32 l_73 = uint32(1) - l_60;
const uint32 l_74 = uint32(16) * l_73;
l_mut_16 = l_74;
const uint32 l_75 = uint32(1) - l_60;
l_mut_55 = l_75;
}

for (uint32 l_mut_76 = uint32(0); l_mut_76 < uint32(4); ++l_mut_76) {
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_282;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_288;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_294;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_300;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_306;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_312;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_318;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_324;
const uint32 l_77 = l_mut_52 * uint32(4);
const uint32 l_78 = l_77 + l_mut_76;
const uint32 l_79 = l_78 * uint32(8);
const uint32 l_80 = uint32(2) * l_mut_76;
const uint32 l_81 = l_80 + l_32;
const uint32 l_82 = l_27 + l_79;
const float_4 l_83 = buffer_1[l_82];
const uint32 l_84 = l_82 + uint32(1);
const float_4 l_85 = buffer_1[l_84];
const float l_86 = l_83.i_0;
const __half l_87 = __half(l_86);
l_arr_39[uint32(0)] = l_87;
const float l_88 = l_85.i_0;
const __half l_89 = __half(l_88);
l_arr_39[uint32(4)] = l_89;
const float l_90 = l_83.i_1;
const __half l_91 = __half(l_90);
l_arr_39[uint32(1)] = l_91;
const float l_92 = l_85.i_1;
const __half l_93 = __half(l_92);
l_arr_39[uint32(5)] = l_93;
const float l_94 = l_83.i_2;
const __half l_95 = __half(l_94);
l_arr_39[uint32(2)] = l_95;
const float l_96 = l_85.i_2;
const __half l_97 = __half(l_96);
l_arr_39[uint32(6)] = l_97;
const float l_98 = l_83.i_3;
const __half l_99 = __half(l_98);
l_arr_39[uint32(3)] = l_99;
const float l_100 = l_85.i_3;
const __half l_101 = __half(l_100);
l_arr_39[uint32(7)] = l_101;
const uint32 l_102 = l_30 + l_79;
const float_4 l_103 = buffer_1[l_102];
const uint32 l_104 = l_102 + uint32(1);
const float_4 l_105 = buffer_1[l_104];
const float l_106 = l_103.i_0;
const __half l_107 = __half(l_106);
l_arr_39[uint32(8)] = l_107;
const float l_108 = l_105.i_0;
const __half l_109 = __half(l_108);
l_arr_39[uint32(12)] = l_109;
const float l_110 = l_103.i_1;
const __half l_111 = __half(l_110);
l_arr_39[uint32(9)] = l_111;
const float l_112 = l_105.i_1;
const __half l_113 = __half(l_112);
l_arr_39[uint32(13)] = l_113;
const float l_114 = l_103.i_2;
const __half l_115 = __half(l_114);
l_arr_39[uint32(10)] = l_115;
const float l_116 = l_105.i_2;
const __half l_117 = __half(l_116);
l_arr_39[uint32(14)] = l_117;
const float l_118 = l_103.i_3;
const __half l_119 = __half(l_118);
l_arr_39[uint32(11)] = l_119;
const float l_120 = l_105.i_3;
const __half l_121 = __half(l_120);
l_arr_39[uint32(15)] = l_121;
const uint32 l_122 = l_mut_23 + l_81;
const uint32 l_123 = buffer_0[l_122];
const uint32 l_124 = l_mut_23 + l_81;
const uint32 l_125 = l_124 + l_mut_55;
const uint32 l_126 = buffer_0[l_125];
const uint32 l_127 = l_123 >> l_mut_16;
const uint32 l_128 = l_126 << uint32(15);
const uint32 l_129 = uint32(17) - l_mut_16;
const uint32 l_130 = l_128 << l_129;
const uint32 l_131 = l_127 | l_130;
const uint32 l_132 = l_131 >> uint32(0);
const uint32 l_133 = l_132 & uint32(3);
const __half l_134 = __half(l_133);
const __half l_135 = l_134 * l_mut_54;
const __half l_136 = l_135 - l_mut_54;
l_arr_40[uint32(0)] = l_136;
const uint32 l_137 = l_131 >> uint32(2);
const uint32 l_138 = l_137 & uint32(3);
const __half l_139 = __half(l_138);
const __half l_140 = l_139 * l_mut_54;
const __half l_141 = l_140 - l_mut_54;
l_arr_40[uint32(1)] = l_141;
const uint32 l_142 = l_131 >> uint32(4);
const uint32 l_143 = l_142 & uint32(3);
const __half l_144 = __half(l_143);
const __half l_145 = l_144 * l_mut_54;
const __half l_146 = l_145 - l_mut_54;
l_arr_40[uint32(2)] = l_146;
const uint32 l_147 = l_131 >> uint32(6);
const uint32 l_148 = l_147 & uint32(3);
const __half l_149 = __half(l_148);
const __half l_150 = l_149 * l_mut_54;
const __half l_151 = l_150 - l_mut_54;
l_arr_40[uint32(3)] = l_151;
const uint32 l_152 = l_131 >> uint32(8);
const uint32 l_153 = l_152 & uint32(3);
const __half l_154 = __half(l_153);
const __half l_155 = l_154 * l_mut_54;
const __half l_156 = l_155 - l_mut_54;
l_arr_40[uint32(4)] = l_156;
const uint32 l_157 = l_131 >> uint32(10);
const uint32 l_158 = l_157 & uint32(3);
const __half l_159 = __half(l_158);
const __half l_160 = l_159 * l_mut_54;
const __half l_161 = l_160 - l_mut_54;
l_arr_40[uint32(5)] = l_161;
const uint32 l_162 = l_131 >> uint32(12);
const uint32 l_163 = l_162 & uint32(3);
const __half l_164 = __half(l_163);
const __half l_165 = l_164 * l_mut_54;
const __half l_166 = l_165 - l_mut_54;
l_arr_40[uint32(6)] = l_166;
const uint32 l_167 = l_131 >> uint32(14);
const uint32 l_168 = l_167 & uint32(3);
const __half l_169 = __half(l_168);
const __half l_170 = l_169 * l_mut_54;
const __half l_171 = l_170 - l_mut_54;
l_arr_40[uint32(7)] = l_171;
const uint32 l_172 = l_131 >> uint32(16);
const uint32 l_173 = l_172 & uint32(3);
const __half l_174 = __half(l_173);
const __half l_175 = l_174 * l_mut_54;
const __half l_176 = l_175 - l_mut_54;
l_arr_40[uint32(8)] = l_176;
const uint32 l_177 = l_131 >> uint32(18);
const uint32 l_178 = l_177 & uint32(3);
const __half l_179 = __half(l_178);
const __half l_180 = l_179 * l_mut_54;
const __half l_181 = l_180 - l_mut_54;
l_arr_40[uint32(9)] = l_181;
const uint32 l_182 = l_131 >> uint32(20);
const uint32 l_183 = l_182 & uint32(3);
const __half l_184 = __half(l_183);
const __half l_185 = l_184 * l_mut_54;
const __half l_186 = l_185 - l_mut_54;
l_arr_40[uint32(10)] = l_186;
const uint32 l_187 = l_131 >> uint32(22);
const uint32 l_188 = l_187 & uint32(3);
const __half l_189 = __half(l_188);
const __half l_190 = l_189 * l_mut_54;
const __half l_191 = l_190 - l_mut_54;
l_arr_40[uint32(11)] = l_191;
const uint32 l_192 = l_131 >> uint32(24);
const uint32 l_193 = l_192 & uint32(3);
const __half l_194 = __half(l_193);
const __half l_195 = l_194 * l_mut_54;
const __half l_196 = l_195 - l_mut_54;
l_arr_40[uint32(12)] = l_196;
const uint32 l_197 = l_131 >> uint32(26);
const uint32 l_198 = l_197 & uint32(3);
const __half l_199 = __half(l_198);
const __half l_200 = l_199 * l_mut_54;
const __half l_201 = l_200 - l_mut_54;
l_arr_40[uint32(13)] = l_201;
const uint32 l_202 = l_131 >> uint32(28);
const uint32 l_203 = l_202 & uint32(3);
const __half l_204 = __half(l_203);
const __half l_205 = l_204 * l_mut_54;
const __half l_206 = l_205 - l_mut_54;
l_arr_40[uint32(14)] = l_206;
const uint32 l_207 = l_131 >> uint32(30);
const uint32 l_208 = l_207 & uint32(3);
const __half l_209 = __half(l_208);
const __half l_210 = l_209 * l_mut_54;
const __half l_211 = l_210 - l_mut_54;
l_arr_40[uint32(15)] = l_211;
__syncthreads();

const uint32 l_212 = l_44 + uint32(0);
const __half l_213 = l_arr_39[uint32(0)];
shared_memory_41[l_212] = l_213;
const uint32 l_214 = l_44 + uint32(1);
const __half l_215 = l_arr_39[uint32(1)];
shared_memory_41[l_214] = l_215;
const uint32 l_216 = l_44 + uint32(2);
const __half l_217 = l_arr_39[uint32(2)];
shared_memory_41[l_216] = l_217;
const uint32 l_218 = l_44 + uint32(3);
const __half l_219 = l_arr_39[uint32(3)];
shared_memory_41[l_218] = l_219;
const uint32 l_220 = l_44 + uint32(4);
const __half l_221 = l_arr_39[uint32(4)];
shared_memory_41[l_220] = l_221;
const uint32 l_222 = l_44 + uint32(5);
const __half l_223 = l_arr_39[uint32(5)];
shared_memory_41[l_222] = l_223;
const uint32 l_224 = l_44 + uint32(6);
const __half l_225 = l_arr_39[uint32(6)];
shared_memory_41[l_224] = l_225;
const uint32 l_226 = l_44 + uint32(7);
const __half l_227 = l_arr_39[uint32(7)];
shared_memory_41[l_226] = l_227;
const uint32 l_228 = l_44 + uint32(1280);
const uint32 l_229 = l_228 + uint32(0);
const __half l_230 = l_arr_39[uint32(8)];
shared_memory_41[l_229] = l_230;
const uint32 l_231 = l_228 + uint32(1);
const __half l_232 = l_arr_39[uint32(9)];
shared_memory_41[l_231] = l_232;
const uint32 l_233 = l_228 + uint32(2);
const __half l_234 = l_arr_39[uint32(10)];
shared_memory_41[l_233] = l_234;
const uint32 l_235 = l_228 + uint32(3);
const __half l_236 = l_arr_39[uint32(11)];
shared_memory_41[l_235] = l_236;
const uint32 l_237 = l_228 + uint32(4);
const __half l_238 = l_arr_39[uint32(12)];
shared_memory_41[l_237] = l_238;
const uint32 l_239 = l_228 + uint32(5);
const __half l_240 = l_arr_39[uint32(13)];
shared_memory_41[l_239] = l_240;
const uint32 l_241 = l_228 + uint32(6);
const __half l_242 = l_arr_39[uint32(14)];
shared_memory_41[l_241] = l_242;
const uint32 l_243 = l_228 + uint32(7);
const __half l_244 = l_arr_39[uint32(15)];
shared_memory_41[l_243] = l_244;
const uint32 l_245 = l_47 + uint32(0);
const __half l_246 = l_arr_40[uint32(0)];
shared_memory_42[l_245] = l_246;
const uint32 l_247 = l_47 + uint32(1);
const __half l_248 = l_arr_40[uint32(1)];
shared_memory_42[l_247] = l_248;
const uint32 l_249 = l_47 + uint32(2);
const __half l_250 = l_arr_40[uint32(2)];
shared_memory_42[l_249] = l_250;
const uint32 l_251 = l_47 + uint32(3);
const __half l_252 = l_arr_40[uint32(3)];
shared_memory_42[l_251] = l_252;
const uint32 l_253 = l_47 + uint32(4);
const __half l_254 = l_arr_40[uint32(4)];
shared_memory_42[l_253] = l_254;
const uint32 l_255 = l_47 + uint32(5);
const __half l_256 = l_arr_40[uint32(5)];
shared_memory_42[l_255] = l_256;
const uint32 l_257 = l_47 + uint32(6);
const __half l_258 = l_arr_40[uint32(6)];
shared_memory_42[l_257] = l_258;
const uint32 l_259 = l_47 + uint32(7);
const __half l_260 = l_arr_40[uint32(7)];
shared_memory_42[l_259] = l_260;
const uint32 l_261 = l_47 + uint32(8);
const __half l_262 = l_arr_40[uint32(8)];
shared_memory_42[l_261] = l_262;
const uint32 l_263 = l_47 + uint32(9);
const __half l_264 = l_arr_40[uint32(9)];
shared_memory_42[l_263] = l_264;
const uint32 l_265 = l_47 + uint32(10);
const __half l_266 = l_arr_40[uint32(10)];
shared_memory_42[l_265] = l_266;
const uint32 l_267 = l_47 + uint32(11);
const __half l_268 = l_arr_40[uint32(11)];
shared_memory_42[l_267] = l_268;
const uint32 l_269 = l_47 + uint32(12);
const __half l_270 = l_arr_40[uint32(12)];
shared_memory_42[l_269] = l_270;
const uint32 l_271 = l_47 + uint32(13);
const __half l_272 = l_arr_40[uint32(13)];
shared_memory_42[l_271] = l_272;
const uint32 l_273 = l_47 + uint32(14);
const __half l_274 = l_arr_40[uint32(14)];
shared_memory_42[l_273] = l_274;
const uint32 l_275 = l_47 + uint32(15);
const __half l_276 = l_arr_40[uint32(15)];
shared_memory_42[l_275] = l_276;
__syncthreads();

const uint32 l_277 = l_2 * uint32(32);
const uint32 l_278 = l_277 + uint32(0);
const uint32 l_279 = l_278 * uint32(40);
const uint32 l_280 = l_279 + uint32(0);
const uint32 l_281 = uint32(2560) - l_280;
nvcuda::wmma::load_matrix_sync(frag_b_282, shared_memory_42 + l_280, uint32(40));
const uint32 l_283 = l_2 * uint32(32);
const uint32 l_284 = l_283 + uint32(16);
const uint32 l_285 = l_284 * uint32(40);
const uint32 l_286 = l_285 + uint32(0);
const uint32 l_287 = uint32(2560) - l_286;
nvcuda::wmma::load_matrix_sync(frag_b_288, shared_memory_42 + l_286, uint32(40));
const uint32 l_289 = l_1 * uint32(32);
const uint32 l_290 = l_289 + uint32(0);
const uint32 l_291 = l_290 * uint32(40);
const uint32 l_292 = l_291 + uint32(0);
const uint32 l_293 = uint32(2560) - l_292;
nvcuda::wmma::load_matrix_sync(frag_a_294, shared_memory_41 + l_292, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_294, frag_b_282, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_294, frag_b_288, frag_acc_49);
const uint32 l_295 = l_1 * uint32(32);
const uint32 l_296 = l_295 + uint32(16);
const uint32 l_297 = l_296 * uint32(40);
const uint32 l_298 = l_297 + uint32(0);
const uint32 l_299 = uint32(2560) - l_298;
nvcuda::wmma::load_matrix_sync(frag_a_300, shared_memory_41 + l_298, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_50, frag_a_300, frag_b_282, frag_acc_50);
nvcuda::wmma::mma_sync(frag_acc_51, frag_a_300, frag_b_288, frag_acc_51);
const uint32 l_301 = l_2 * uint32(32);
const uint32 l_302 = l_301 + uint32(0);
const uint32 l_303 = l_302 * uint32(40);
const uint32 l_304 = l_303 + uint32(16);
const uint32 l_305 = uint32(2560) - l_304;
nvcuda::wmma::load_matrix_sync(frag_b_306, shared_memory_42 + l_304, uint32(40));
const uint32 l_307 = l_2 * uint32(32);
const uint32 l_308 = l_307 + uint32(16);
const uint32 l_309 = l_308 * uint32(40);
const uint32 l_310 = l_309 + uint32(16);
const uint32 l_311 = uint32(2560) - l_310;
nvcuda::wmma::load_matrix_sync(frag_b_312, shared_memory_42 + l_310, uint32(40));
const uint32 l_313 = l_1 * uint32(32);
const uint32 l_314 = l_313 + uint32(0);
const uint32 l_315 = l_314 * uint32(40);
const uint32 l_316 = l_315 + uint32(16);
const uint32 l_317 = uint32(2560) - l_316;
nvcuda::wmma::load_matrix_sync(frag_a_318, shared_memory_41 + l_316, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_318, frag_b_306, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_318, frag_b_312, frag_acc_49);
const uint32 l_319 = l_1 * uint32(32);
const uint32 l_320 = l_319 + uint32(16);
const uint32 l_321 = l_320 * uint32(40);
const uint32 l_322 = l_321 + uint32(16);
const uint32 l_323 = uint32(2560) - l_322;
nvcuda::wmma::load_matrix_sync(frag_a_324, shared_memory_41 + l_322, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_50, frag_a_324, frag_b_306, frag_acc_50);
nvcuda::wmma::mma_sync(frag_acc_51, frag_a_324, frag_b_312, frag_acc_51);
}
}
const bool l_325 = l_0 == uint32(0);
if (l_325) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_326 = l_6 + uint32(0);
const uint32 l_327 = l_326 + uint32(0);
const uint32 l_328 = l_7 + uint32(0);
__syncthreads();

const uint32 l_329 = threadIdxGlobal + uint32(0);
const uint32 l_330 = l_329 / uint32(32);
const uint32 l_331 = l_327 + l_330;
const uint32 l_332 = l_329 % uint32(32);
const uint32 l_333 = l_328 + l_332;
const bool l_334 = l_331 < l_3;
const bool l_335 = l_333 < l_4;
const bool l_336 = l_334 && l_335;
if (l_336) {
const uint32 l_337 = l_331 * l_4;
const uint32 l_338 = l_337 + l_333;
const float l_339 = shared_memory_38[l_329];
buffer_2[l_338] = l_339;
}
const uint32 l_340 = threadIdxGlobal + uint32(128);
const uint32 l_341 = l_340 / uint32(32);
const uint32 l_342 = l_327 + l_341;
const uint32 l_343 = l_340 % uint32(32);
const uint32 l_344 = l_328 + l_343;
const bool l_345 = l_342 < l_3;
const bool l_346 = l_344 < l_4;
const bool l_347 = l_345 && l_346;
if (l_347) {
const uint32 l_348 = l_342 * l_4;
const uint32 l_349 = l_348 + l_344;
const float l_350 = shared_memory_38[l_340];
buffer_2[l_349] = l_350;
}
const uint32 l_351 = threadIdxGlobal + uint32(256);
const uint32 l_352 = l_351 / uint32(32);
const uint32 l_353 = l_327 + l_352;
const uint32 l_354 = l_351 % uint32(32);
const uint32 l_355 = l_328 + l_354;
const bool l_356 = l_353 < l_3;
const bool l_357 = l_355 < l_4;
const bool l_358 = l_356 && l_357;
if (l_358) {
const uint32 l_359 = l_353 * l_4;
const uint32 l_360 = l_359 + l_355;
const float l_361 = shared_memory_38[l_351];
buffer_2[l_360] = l_361;
}
const uint32 l_362 = threadIdxGlobal + uint32(384);
const uint32 l_363 = l_362 / uint32(32);
const uint32 l_364 = l_327 + l_363;
const uint32 l_365 = l_362 % uint32(32);
const uint32 l_366 = l_328 + l_365;
const bool l_367 = l_364 < l_3;
const bool l_368 = l_366 < l_4;
const bool l_369 = l_367 && l_368;
if (l_369) {
const uint32 l_370 = l_364 * l_4;
const uint32 l_371 = l_370 + l_366;
const float l_372 = shared_memory_38[l_362];
buffer_2[l_371] = l_372;
}
__syncthreads();

const bool l_373 = l_0 == uint32(0);
if (l_373) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_374 = l_6 + uint32(0);
const uint32 l_375 = l_374 + uint32(16);
const uint32 l_376 = l_7 + uint32(0);
__syncthreads();

const uint32 l_377 = threadIdxGlobal + uint32(0);
const uint32 l_378 = l_377 / uint32(32);
const uint32 l_379 = l_375 + l_378;
const uint32 l_380 = l_377 % uint32(32);
const uint32 l_381 = l_376 + l_380;
const bool l_382 = l_379 < l_3;
const bool l_383 = l_381 < l_4;
const bool l_384 = l_382 && l_383;
if (l_384) {
const uint32 l_385 = l_379 * l_4;
const uint32 l_386 = l_385 + l_381;
const float l_387 = shared_memory_38[l_377];
buffer_2[l_386] = l_387;
}
const uint32 l_388 = threadIdxGlobal + uint32(128);
const uint32 l_389 = l_388 / uint32(32);
const uint32 l_390 = l_375 + l_389;
const uint32 l_391 = l_388 % uint32(32);
const uint32 l_392 = l_376 + l_391;
const bool l_393 = l_390 < l_3;
const bool l_394 = l_392 < l_4;
const bool l_395 = l_393 && l_394;
if (l_395) {
const uint32 l_396 = l_390 * l_4;
const uint32 l_397 = l_396 + l_392;
const float l_398 = shared_memory_38[l_388];
buffer_2[l_397] = l_398;
}
const uint32 l_399 = threadIdxGlobal + uint32(256);
const uint32 l_400 = l_399 / uint32(32);
const uint32 l_401 = l_375 + l_400;
const uint32 l_402 = l_399 % uint32(32);
const uint32 l_403 = l_376 + l_402;
const bool l_404 = l_401 < l_3;
const bool l_405 = l_403 < l_4;
const bool l_406 = l_404 && l_405;
if (l_406) {
const uint32 l_407 = l_401 * l_4;
const uint32 l_408 = l_407 + l_403;
const float l_409 = shared_memory_38[l_399];
buffer_2[l_408] = l_409;
}
const uint32 l_410 = threadIdxGlobal + uint32(384);
const uint32 l_411 = l_410 / uint32(32);
const uint32 l_412 = l_375 + l_411;
const uint32 l_413 = l_410 % uint32(32);
const uint32 l_414 = l_376 + l_413;
const bool l_415 = l_412 < l_3;
const bool l_416 = l_414 < l_4;
const bool l_417 = l_415 && l_416;
if (l_417) {
const uint32 l_418 = l_412 * l_4;
const uint32 l_419 = l_418 + l_414;
const float l_420 = shared_memory_38[l_410];
buffer_2[l_419] = l_420;
}
__syncthreads();

const bool l_421 = l_0 == uint32(1);
if (l_421) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_422 = l_6 + uint32(0);
const uint32 l_423 = l_422 + uint32(0);
const uint32 l_424 = l_7 + uint32(32);
__syncthreads();

const uint32 l_425 = threadIdxGlobal + uint32(0);
const uint32 l_426 = l_425 / uint32(32);
const uint32 l_427 = l_423 + l_426;
const uint32 l_428 = l_425 % uint32(32);
const uint32 l_429 = l_424 + l_428;
const bool l_430 = l_427 < l_3;
const bool l_431 = l_429 < l_4;
const bool l_432 = l_430 && l_431;
if (l_432) {
const uint32 l_433 = l_427 * l_4;
const uint32 l_434 = l_433 + l_429;
const float l_435 = shared_memory_38[l_425];
buffer_2[l_434] = l_435;
}
const uint32 l_436 = threadIdxGlobal + uint32(128);
const uint32 l_437 = l_436 / uint32(32);
const uint32 l_438 = l_423 + l_437;
const uint32 l_439 = l_436 % uint32(32);
const uint32 l_440 = l_424 + l_439;
const bool l_441 = l_438 < l_3;
const bool l_442 = l_440 < l_4;
const bool l_443 = l_441 && l_442;
if (l_443) {
const uint32 l_444 = l_438 * l_4;
const uint32 l_445 = l_444 + l_440;
const float l_446 = shared_memory_38[l_436];
buffer_2[l_445] = l_446;
}
const uint32 l_447 = threadIdxGlobal + uint32(256);
const uint32 l_448 = l_447 / uint32(32);
const uint32 l_449 = l_423 + l_448;
const uint32 l_450 = l_447 % uint32(32);
const uint32 l_451 = l_424 + l_450;
const bool l_452 = l_449 < l_3;
const bool l_453 = l_451 < l_4;
const bool l_454 = l_452 && l_453;
if (l_454) {
const uint32 l_455 = l_449 * l_4;
const uint32 l_456 = l_455 + l_451;
const float l_457 = shared_memory_38[l_447];
buffer_2[l_456] = l_457;
}
const uint32 l_458 = threadIdxGlobal + uint32(384);
const uint32 l_459 = l_458 / uint32(32);
const uint32 l_460 = l_423 + l_459;
const uint32 l_461 = l_458 % uint32(32);
const uint32 l_462 = l_424 + l_461;
const bool l_463 = l_460 < l_3;
const bool l_464 = l_462 < l_4;
const bool l_465 = l_463 && l_464;
if (l_465) {
const uint32 l_466 = l_460 * l_4;
const uint32 l_467 = l_466 + l_462;
const float l_468 = shared_memory_38[l_458];
buffer_2[l_467] = l_468;
}
__syncthreads();

const bool l_469 = l_0 == uint32(1);
if (l_469) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_470 = l_6 + uint32(0);
const uint32 l_471 = l_470 + uint32(16);
const uint32 l_472 = l_7 + uint32(32);
__syncthreads();

const uint32 l_473 = threadIdxGlobal + uint32(0);
const uint32 l_474 = l_473 / uint32(32);
const uint32 l_475 = l_471 + l_474;
const uint32 l_476 = l_473 % uint32(32);
const uint32 l_477 = l_472 + l_476;
const bool l_478 = l_475 < l_3;
const bool l_479 = l_477 < l_4;
const bool l_480 = l_478 && l_479;
if (l_480) {
const uint32 l_481 = l_475 * l_4;
const uint32 l_482 = l_481 + l_477;
const float l_483 = shared_memory_38[l_473];
buffer_2[l_482] = l_483;
}
const uint32 l_484 = threadIdxGlobal + uint32(128);
const uint32 l_485 = l_484 / uint32(32);
const uint32 l_486 = l_471 + l_485;
const uint32 l_487 = l_484 % uint32(32);
const uint32 l_488 = l_472 + l_487;
const bool l_489 = l_486 < l_3;
const bool l_490 = l_488 < l_4;
const bool l_491 = l_489 && l_490;
if (l_491) {
const uint32 l_492 = l_486 * l_4;
const uint32 l_493 = l_492 + l_488;
const float l_494 = shared_memory_38[l_484];
buffer_2[l_493] = l_494;
}
const uint32 l_495 = threadIdxGlobal + uint32(256);
const uint32 l_496 = l_495 / uint32(32);
const uint32 l_497 = l_471 + l_496;
const uint32 l_498 = l_495 % uint32(32);
const uint32 l_499 = l_472 + l_498;
const bool l_500 = l_497 < l_3;
const bool l_501 = l_499 < l_4;
const bool l_502 = l_500 && l_501;
if (l_502) {
const uint32 l_503 = l_497 * l_4;
const uint32 l_504 = l_503 + l_499;
const float l_505 = shared_memory_38[l_495];
buffer_2[l_504] = l_505;
}
const uint32 l_506 = threadIdxGlobal + uint32(384);
const uint32 l_507 = l_506 / uint32(32);
const uint32 l_508 = l_471 + l_507;
const uint32 l_509 = l_506 % uint32(32);
const uint32 l_510 = l_472 + l_509;
const bool l_511 = l_508 < l_3;
const bool l_512 = l_510 < l_4;
const bool l_513 = l_511 && l_512;
if (l_513) {
const uint32 l_514 = l_508 * l_4;
const uint32 l_515 = l_514 + l_510;
const float l_516 = shared_memory_38[l_506];
buffer_2[l_515] = l_516;
}
__syncthreads();

const bool l_517 = l_0 == uint32(2);
if (l_517) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_518 = l_6 + uint32(32);
const uint32 l_519 = l_518 + uint32(0);
const uint32 l_520 = l_7 + uint32(0);
__syncthreads();

const uint32 l_521 = threadIdxGlobal + uint32(0);
const uint32 l_522 = l_521 / uint32(32);
const uint32 l_523 = l_519 + l_522;
const uint32 l_524 = l_521 % uint32(32);
const uint32 l_525 = l_520 + l_524;
const bool l_526 = l_523 < l_3;
const bool l_527 = l_525 < l_4;
const bool l_528 = l_526 && l_527;
if (l_528) {
const uint32 l_529 = l_523 * l_4;
const uint32 l_530 = l_529 + l_525;
const float l_531 = shared_memory_38[l_521];
buffer_2[l_530] = l_531;
}
const uint32 l_532 = threadIdxGlobal + uint32(128);
const uint32 l_533 = l_532 / uint32(32);
const uint32 l_534 = l_519 + l_533;
const uint32 l_535 = l_532 % uint32(32);
const uint32 l_536 = l_520 + l_535;
const bool l_537 = l_534 < l_3;
const bool l_538 = l_536 < l_4;
const bool l_539 = l_537 && l_538;
if (l_539) {
const uint32 l_540 = l_534 * l_4;
const uint32 l_541 = l_540 + l_536;
const float l_542 = shared_memory_38[l_532];
buffer_2[l_541] = l_542;
}
const uint32 l_543 = threadIdxGlobal + uint32(256);
const uint32 l_544 = l_543 / uint32(32);
const uint32 l_545 = l_519 + l_544;
const uint32 l_546 = l_543 % uint32(32);
const uint32 l_547 = l_520 + l_546;
const bool l_548 = l_545 < l_3;
const bool l_549 = l_547 < l_4;
const bool l_550 = l_548 && l_549;
if (l_550) {
const uint32 l_551 = l_545 * l_4;
const uint32 l_552 = l_551 + l_547;
const float l_553 = shared_memory_38[l_543];
buffer_2[l_552] = l_553;
}
const uint32 l_554 = threadIdxGlobal + uint32(384);
const uint32 l_555 = l_554 / uint32(32);
const uint32 l_556 = l_519 + l_555;
const uint32 l_557 = l_554 % uint32(32);
const uint32 l_558 = l_520 + l_557;
const bool l_559 = l_556 < l_3;
const bool l_560 = l_558 < l_4;
const bool l_561 = l_559 && l_560;
if (l_561) {
const uint32 l_562 = l_556 * l_4;
const uint32 l_563 = l_562 + l_558;
const float l_564 = shared_memory_38[l_554];
buffer_2[l_563] = l_564;
}
__syncthreads();

const bool l_565 = l_0 == uint32(2);
if (l_565) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_566 = l_6 + uint32(32);
const uint32 l_567 = l_566 + uint32(16);
const uint32 l_568 = l_7 + uint32(0);
__syncthreads();

const uint32 l_569 = threadIdxGlobal + uint32(0);
const uint32 l_570 = l_569 / uint32(32);
const uint32 l_571 = l_567 + l_570;
const uint32 l_572 = l_569 % uint32(32);
const uint32 l_573 = l_568 + l_572;
const bool l_574 = l_571 < l_3;
const bool l_575 = l_573 < l_4;
const bool l_576 = l_574 && l_575;
if (l_576) {
const uint32 l_577 = l_571 * l_4;
const uint32 l_578 = l_577 + l_573;
const float l_579 = shared_memory_38[l_569];
buffer_2[l_578] = l_579;
}
const uint32 l_580 = threadIdxGlobal + uint32(128);
const uint32 l_581 = l_580 / uint32(32);
const uint32 l_582 = l_567 + l_581;
const uint32 l_583 = l_580 % uint32(32);
const uint32 l_584 = l_568 + l_583;
const bool l_585 = l_582 < l_3;
const bool l_586 = l_584 < l_4;
const bool l_587 = l_585 && l_586;
if (l_587) {
const uint32 l_588 = l_582 * l_4;
const uint32 l_589 = l_588 + l_584;
const float l_590 = shared_memory_38[l_580];
buffer_2[l_589] = l_590;
}
const uint32 l_591 = threadIdxGlobal + uint32(256);
const uint32 l_592 = l_591 / uint32(32);
const uint32 l_593 = l_567 + l_592;
const uint32 l_594 = l_591 % uint32(32);
const uint32 l_595 = l_568 + l_594;
const bool l_596 = l_593 < l_3;
const bool l_597 = l_595 < l_4;
const bool l_598 = l_596 && l_597;
if (l_598) {
const uint32 l_599 = l_593 * l_4;
const uint32 l_600 = l_599 + l_595;
const float l_601 = shared_memory_38[l_591];
buffer_2[l_600] = l_601;
}
const uint32 l_602 = threadIdxGlobal + uint32(384);
const uint32 l_603 = l_602 / uint32(32);
const uint32 l_604 = l_567 + l_603;
const uint32 l_605 = l_602 % uint32(32);
const uint32 l_606 = l_568 + l_605;
const bool l_607 = l_604 < l_3;
const bool l_608 = l_606 < l_4;
const bool l_609 = l_607 && l_608;
if (l_609) {
const uint32 l_610 = l_604 * l_4;
const uint32 l_611 = l_610 + l_606;
const float l_612 = shared_memory_38[l_602];
buffer_2[l_611] = l_612;
}
__syncthreads();

const bool l_613 = l_0 == uint32(3);
if (l_613) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_614 = l_6 + uint32(32);
const uint32 l_615 = l_614 + uint32(0);
const uint32 l_616 = l_7 + uint32(32);
__syncthreads();

const uint32 l_617 = threadIdxGlobal + uint32(0);
const uint32 l_618 = l_617 / uint32(32);
const uint32 l_619 = l_615 + l_618;
const uint32 l_620 = l_617 % uint32(32);
const uint32 l_621 = l_616 + l_620;
const bool l_622 = l_619 < l_3;
const bool l_623 = l_621 < l_4;
const bool l_624 = l_622 && l_623;
if (l_624) {
const uint32 l_625 = l_619 * l_4;
const uint32 l_626 = l_625 + l_621;
const float l_627 = shared_memory_38[l_617];
buffer_2[l_626] = l_627;
}
const uint32 l_628 = threadIdxGlobal + uint32(128);
const uint32 l_629 = l_628 / uint32(32);
const uint32 l_630 = l_615 + l_629;
const uint32 l_631 = l_628 % uint32(32);
const uint32 l_632 = l_616 + l_631;
const bool l_633 = l_630 < l_3;
const bool l_634 = l_632 < l_4;
const bool l_635 = l_633 && l_634;
if (l_635) {
const uint32 l_636 = l_630 * l_4;
const uint32 l_637 = l_636 + l_632;
const float l_638 = shared_memory_38[l_628];
buffer_2[l_637] = l_638;
}
const uint32 l_639 = threadIdxGlobal + uint32(256);
const uint32 l_640 = l_639 / uint32(32);
const uint32 l_641 = l_615 + l_640;
const uint32 l_642 = l_639 % uint32(32);
const uint32 l_643 = l_616 + l_642;
const bool l_644 = l_641 < l_3;
const bool l_645 = l_643 < l_4;
const bool l_646 = l_644 && l_645;
if (l_646) {
const uint32 l_647 = l_641 * l_4;
const uint32 l_648 = l_647 + l_643;
const float l_649 = shared_memory_38[l_639];
buffer_2[l_648] = l_649;
}
const uint32 l_650 = threadIdxGlobal + uint32(384);
const uint32 l_651 = l_650 / uint32(32);
const uint32 l_652 = l_615 + l_651;
const uint32 l_653 = l_650 % uint32(32);
const uint32 l_654 = l_616 + l_653;
const bool l_655 = l_652 < l_3;
const bool l_656 = l_654 < l_4;
const bool l_657 = l_655 && l_656;
if (l_657) {
const uint32 l_658 = l_652 * l_4;
const uint32 l_659 = l_658 + l_654;
const float l_660 = shared_memory_38[l_650];
buffer_2[l_659] = l_660;
}
__syncthreads();

const bool l_661 = l_0 == uint32(3);
if (l_661) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_662 = l_6 + uint32(32);
const uint32 l_663 = l_662 + uint32(16);
const uint32 l_664 = l_7 + uint32(32);
__syncthreads();

const uint32 l_665 = threadIdxGlobal + uint32(0);
const uint32 l_666 = l_665 / uint32(32);
const uint32 l_667 = l_663 + l_666;
const uint32 l_668 = l_665 % uint32(32);
const uint32 l_669 = l_664 + l_668;
const bool l_670 = l_667 < l_3;
const bool l_671 = l_669 < l_4;
const bool l_672 = l_670 && l_671;
if (l_672) {
const uint32 l_673 = l_667 * l_4;
const uint32 l_674 = l_673 + l_669;
const float l_675 = shared_memory_38[l_665];
buffer_2[l_674] = l_675;
}
const uint32 l_676 = threadIdxGlobal + uint32(128);
const uint32 l_677 = l_676 / uint32(32);
const uint32 l_678 = l_663 + l_677;
const uint32 l_679 = l_676 % uint32(32);
const uint32 l_680 = l_664 + l_679;
const bool l_681 = l_678 < l_3;
const bool l_682 = l_680 < l_4;
const bool l_683 = l_681 && l_682;
if (l_683) {
const uint32 l_684 = l_678 * l_4;
const uint32 l_685 = l_684 + l_680;
const float l_686 = shared_memory_38[l_676];
buffer_2[l_685] = l_686;
}
const uint32 l_687 = threadIdxGlobal + uint32(256);
const uint32 l_688 = l_687 / uint32(32);
const uint32 l_689 = l_663 + l_688;
const uint32 l_690 = l_687 % uint32(32);
const uint32 l_691 = l_664 + l_690;
const bool l_692 = l_689 < l_3;
const bool l_693 = l_691 < l_4;
const bool l_694 = l_692 && l_693;
if (l_694) {
const uint32 l_695 = l_689 * l_4;
const uint32 l_696 = l_695 + l_691;
const float l_697 = shared_memory_38[l_687];
buffer_2[l_696] = l_697;
}
const uint32 l_698 = threadIdxGlobal + uint32(384);
const uint32 l_699 = l_698 / uint32(32);
const uint32 l_700 = l_663 + l_699;
const uint32 l_701 = l_698 % uint32(32);
const uint32 l_702 = l_664 + l_701;
const bool l_703 = l_700 < l_3;
const bool l_704 = l_702 < l_4;
const bool l_705 = l_703 && l_704;
if (l_705) {
const uint32 l_706 = l_700 * l_4;
const uint32 l_707 = l_706 + l_702;
const float l_708 = shared_memory_38[l_698];
buffer_2[l_707] = l_708;
}
__syncthreads();


}
