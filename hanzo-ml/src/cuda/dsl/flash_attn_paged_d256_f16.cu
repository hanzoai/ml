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

struct info_st {
    uint32 static_meta[16];
};


extern "C" __global__ void __launch_bounds__(32)flash_attn_paged_f_ (
    const __half* __restrict__ buffer_0, const __half* __restrict__ buffer_1, const __half* __restrict__ buffer_2, const uint32* __restrict__ buffer_3, const uint32* __restrict__ buffer_4, __half* buffer_5, const float* __restrict__ buffer_6, const uint32* __restrict__ buffer_7, const __grid_constant__ info_st info
) {
uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
uint32 blockIdxGlobal = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x;
extern __shared__ __align__(4) uint8 dynamic_shared_mem[];
// Shared array size: 4096, 8192 bytes
__half *shared_memory_34 = reinterpret_cast<__half*>(&dynamic_shared_mem[0]);
// Shared array size: 256, 1024 bytes
float *shared_memory_35 = reinterpret_cast<float*>(&dynamic_shared_mem[8192]);
// Shared array size: 4096, 16384 bytes
float *shared_memory_36 = reinterpret_cast<float*>(&dynamic_shared_mem[9216]);
// Shared array size: 16, 64 bytes
float *shared_memory_37 = reinterpret_cast<float*>(&dynamic_shared_mem[25600]);
// Shared array size: 16, 64 bytes
float *shared_memory_38 = reinterpret_cast<float*>(&dynamic_shared_mem[25664]);
// Shared array size: 16, 64 bytes
float *shared_memory_39 = reinterpret_cast<float*>(&dynamic_shared_mem[25728]);
// Shared array size: 4096, 8192 bytes
__half *shared_memory_80 = reinterpret_cast<__half*>(&dynamic_shared_mem[25792]);
// Shared array size: 256, 512 bytes
__half *shared_memory_201 = reinterpret_cast<__half*>(&dynamic_shared_mem[33984]);
// Shared array size: 4096, 8192 bytes
__half *shared_memory_202 = reinterpret_cast<__half*>(&dynamic_shared_mem[34496]);
// Shared array size: 256, 1024 bytes
float *shared_memory_203 = reinterpret_cast<float*>(&dynamic_shared_mem[42688]);
uint32 l_mut_26;
float l_mut_51;
uint32 l_mut_62;
uint32 l_mut_71;
__half l_mut_97;
__half l_mut_111;
float l_mut_186;
const uint32 l_0 = buffer_7[uint32(10)];
const uint32 l_1 = blockIdxGlobal + l_0;
const float l_2 = buffer_6[uint32(0)];
const float l_3 = buffer_6[uint32(1)];
const uint32 l_4 = buffer_7[uint32(0)];
const uint32 l_5 = buffer_7[uint32(1)];
const uint32 l_6 = buffer_7[uint32(2)];
const uint32 l_7 = buffer_7[uint32(3)];
const uint32 l_8 = buffer_7[uint32(4)];
const uint32 l_9 = buffer_7[uint32(5)];
const uint32 l_10 = buffer_7[uint32(6)];
const uint32 l_11 = buffer_7[uint32(7)];
const uint32 l_12 = buffer_7[uint32(8)];
const uint32 l_13 = buffer_7[uint32(9)];
const uint32 l_14 = l_4 + uint32(16);
const uint32 l_15 = l_14 - uint32(1);
const uint32 l_16 = l_15 / uint32(16);
const uint32 l_17 = l_5 * l_16;
const uint32 l_18 = l_1 / l_17;
const uint32 l_19 = l_18 * l_17;
const uint32 l_20 = l_1 - l_19;
const uint32 l_21 = l_20 / l_16;
const uint32 l_22 = l_20 % l_16;
const uint32 l_23 = l_5 / l_6;
const uint32 l_24 = l_21 / l_23;
const uint32 l_25 = buffer_4[l_18];
l_mut_26 = uint32(0);
const bool l_27 = l_7 == uint32(2);
if (l_27) {
l_mut_26 = l_13;
} else {
const bool l_28 = l_25 >= l_4;
if (l_28) {
const uint32 l_29 = l_25 - l_4;
l_mut_26 = l_29;
}
}
const uint32 l_30 = l_18 * l_5;
const uint32 l_31 = l_30 + l_21;
const uint32 l_32 = l_31 * l_4;
const uint32 l_33 = l_32 * uint32(256);

for (uint32 l_mut_40 = uint32(0); l_mut_40 < uint32(128); ++l_mut_40) {
const uint32 l_41 = threadIdxGlobal * uint32(128);
const uint32 l_42 = l_41 + l_mut_40;
shared_memory_36[l_42] = float(0.0);
}
const bool l_43 = threadIdxGlobal < uint32(16);
if (l_43) {
shared_memory_37[threadIdxGlobal] = float(-3.3999999521443642e38);
shared_memory_38[threadIdxGlobal] = float(0.0);
}

for (uint32 l_mut_44 = uint32(0); l_mut_44 < uint32(128); ++l_mut_44) {
const uint32 l_45 = threadIdxGlobal * uint32(128);
const uint32 l_46 = l_45 + l_mut_44;
const uint32 l_47 = l_46 / uint32(256);
const uint32 l_48 = l_46 % uint32(256);
const uint32 l_49 = l_22 * uint32(16);
const uint32 l_50 = l_49 + l_47;
l_mut_51 = float(0.0);
const bool l_52 = l_50 < l_4;
if (l_52) {
const uint32 l_53 = l_50 * uint32(256);
const uint32 l_54 = l_33 + l_53;
const uint32 l_55 = l_54 + l_48;
const __half l_56 = buffer_0[l_55];
const float l_57 = float(l_56);
l_mut_51 = l_57;
}
const __half l_58 = __half(l_mut_51);
shared_memory_34[l_46] = l_58;
}
__syncthreads();

const uint32 l_59 = l_25 + uint32(16);
const uint32 l_60 = l_59 - uint32(1);
const uint32 l_61 = l_60 / uint32(16);
l_mut_62 = l_61;
const bool l_63 = l_7 != uint32(0);
if (l_63) {
const uint32 l_64 = l_22 * uint32(16);
const uint32 l_65 = l_mut_26 + l_64;
const uint32 l_66 = l_65 + uint32(16);
const uint32 l_67 = l_66 - uint32(1);
const uint32 l_68 = l_67 / uint32(16);
const uint32 l_69 = l_68 + uint32(1);
const bool l_70 = l_69 < l_mut_62;
if (l_70) {
l_mut_62 = l_69;
}
}
l_mut_71 = uint32(0);
const bool l_72 = l_12 > uint32(0);
if (l_72) {
const uint32 l_73 = l_22 * uint32(16);
const uint32 l_74 = l_mut_26 + l_73;
const bool l_75 = l_74 >= l_12;
if (l_75) {
const uint32 l_76 = l_74 - l_12;
const uint32 l_77 = l_76 + uint32(1);
const uint32 l_78 = l_77 / uint32(16);
l_mut_71 = l_78;
}
}

for (uint32 l_mut_79 = l_mut_71; l_mut_79 < l_mut_62; ++l_mut_79) {
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_126;

for (uint32 l_mut_81 = uint32(0); l_mut_81 < uint32(128); ++l_mut_81) {
const uint32 l_82 = threadIdxGlobal * uint32(128);
const uint32 l_83 = l_82 + l_mut_81;
const uint32 l_84 = l_83 / uint32(256);
const uint32 l_85 = l_83 % uint32(256);
const uint32 l_86 = l_mut_79 * uint32(16);
const uint32 l_87 = l_86 + l_84;
l_mut_51 = float(0.0);
const bool l_88 = l_87 < l_25;
if (l_88) {
const bool l_89 = l_10 == uint32(2);
if (l_89) {
const uint32 l_90 = l_18 * l_6;
const uint32 l_91 = l_90 + l_24;
const uint32 l_92 = l_91 * l_8;
const uint32 l_93 = l_92 + l_87;
const uint32 l_94 = l_93 * uint32(256);
const uint32 l_95 = l_94 + l_85;
const __half l_96 = buffer_1[l_95];
l_mut_97 = l_96;
} else {
const uint32 l_98 = l_18 * l_9;
const uint32 l_99 = l_87 / l_8;
const uint32 l_100 = l_98 + l_99;
const uint32 l_101 = buffer_3[l_100];
const uint32 l_102 = l_87 % l_8;
const bool l_103 = l_10 == uint32(0);
if (l_103) {
const uint32 l_104 = l_101 * l_6;
const uint32 l_105 = l_104 + l_24;
const uint32 l_106 = l_105 * l_8;
const uint32 l_107 = l_106 + l_102;
const uint32 l_108 = l_107 * uint32(256);
const uint32 l_109 = l_108 + l_85;
const __half l_110 = buffer_1[l_109];
l_mut_111 = l_110;
} else {
const uint32 l_112 = uint32(256) / l_11;
const uint32 l_113 = l_101 * l_6;
const uint32 l_114 = l_113 + l_24;
const uint32 l_115 = l_114 * l_112;
const uint32 l_116 = l_85 / l_11;
const uint32 l_117 = l_115 + l_116;
const uint32 l_118 = l_117 * l_8;
const uint32 l_119 = l_118 + l_102;
const uint32 l_120 = l_119 * l_11;
const uint32 l_121 = l_85 % l_11;
const uint32 l_122 = l_120 + l_121;
const __half l_123 = buffer_1[l_122];
l_mut_111 = l_123;
}
l_mut_97 = l_mut_111;
}
const float l_124 = float(l_mut_97);
l_mut_51 = l_124;
}
const __half l_125 = __half(l_mut_51);
shared_memory_80[l_83] = l_125;
}
__syncthreads();

nvcuda::wmma::fill_fragment(frag_acc_126, float(0.0));

for (uint32 l_mut_127 = uint32(0); l_mut_127 < uint32(16); ++l_mut_127) {
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_133;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_142;
const uint32 l_128 = l_mut_127 * uint32(16);
const uint32 l_129 = 4096;
const uint32 l_130 = l_129 - uint32(0);
const uint32 l_131 = uint32(4096) - l_128;
const uint32 l_132 = l_128 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_a_133, shared_memory_34 + l_132, uint32(256));
const uint32 l_134 = uint32(0) * uint32(16);
const uint32 l_135 = l_134 * uint32(256);
const uint32 l_136 = l_mut_127 * uint32(16);
const uint32 l_137 = l_135 + l_136;
const uint32 l_138 = 4096;
const uint32 l_139 = l_138 - uint32(0);
const uint32 l_140 = uint32(4096) - l_137;
const uint32 l_141 = l_137 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_b_142, shared_memory_80 + l_141, uint32(256));
nvcuda::wmma::mma_sync(frag_acc_126, frag_a_133, frag_b_142, frag_acc_126);
}
const uint32 l_143 = uint32(0) * uint32(16);
const uint32 l_144 = 256;
const uint32 l_145 = l_144 - uint32(0);
const uint32 l_146 = uint32(256) - l_143;
const uint32 l_147 = l_143 + uint32(0);
nvcuda::wmma::store_matrix_sync(shared_memory_35 + l_147, frag_acc_126, uint32(16), nvcuda::wmma::mem_row_major);
__syncthreads();


for (uint32 l_mut_148 = uint32(0); l_mut_148 < uint32(8); ++l_mut_148) {
const uint32 l_149 = threadIdxGlobal * uint32(8);
const uint32 l_150 = l_149 + l_mut_148;
const uint32 l_151 = l_150 / uint32(16);
const uint32 l_152 = l_150 % uint32(16);
const uint32 l_153 = l_22 * uint32(16);
const uint32 l_154 = l_153 + l_151;
const uint32 l_155 = l_mut_79 * uint32(16);
const uint32 l_156 = l_155 + l_152;
const uint32 l_157 = l_mut_26 + l_154;
const bool l_158 = l_7 == uint32(0);
const bool l_159 = l_156 <= l_157;
const bool l_160 = l_158 || l_159;
const bool l_161 = l_12 == uint32(0);
const uint32 l_162 = l_156 + l_12;
const bool l_163 = l_162 > l_157;
const bool l_164 = l_161 || l_163;
const bool l_165 = l_154 < l_4;
const bool l_166 = l_156 < l_25;
const bool l_167 = l_165 && l_166;
const bool l_168 = l_167 && l_160;
const bool l_169 = l_168 && l_164;
if (l_169) {
const float l_170 = shared_memory_35[l_150];
const float l_171 = l_170 * l_2;
l_mut_51 = l_171;
const bool l_172 = l_3 > float(0.0);
if (l_172) {
const float l_173 = l_mut_51 / l_3;
const float l_174 = tanh(l_173);
const float l_175 = l_3 * l_174;
l_mut_51 = l_175;
}
shared_memory_35[l_150] = l_mut_51;
} else {
shared_memory_35[l_150] = float(-3.3999999521443642e38);
}
}
__syncthreads();

const bool l_176 = threadIdxGlobal < uint32(16);
if (l_176) {
const uint32 l_177 = l_22 * uint32(16);
const uint32 l_178 = l_177 + threadIdxGlobal;
const bool l_179 = l_178 < l_4;
if (l_179) {
l_mut_51 = float(-3.3999999521443642e38);

for (uint32 l_mut_180 = uint32(0); l_mut_180 < uint32(16); ++l_mut_180) {
const uint32 l_181 = threadIdxGlobal * uint32(16);
const uint32 l_182 = l_181 + l_mut_180;
const float l_183 = shared_memory_35[l_182];
const bool l_184 = l_183 > l_mut_51;
if (l_184) {
l_mut_51 = l_183;
}
}
const float l_185 = shared_memory_37[threadIdxGlobal];
l_mut_186 = l_185;
const bool l_187 = l_mut_51 > l_mut_186;
if (l_187) {
l_mut_186 = l_mut_51;
}
const float l_188 = l_185 - l_mut_186;
const float l_189 = exp(l_188);
l_mut_51 = float(0.0);

for (uint32 l_mut_190 = uint32(0); l_mut_190 < uint32(16); ++l_mut_190) {
const uint32 l_191 = threadIdxGlobal * uint32(16);
const uint32 l_192 = l_191 + l_mut_190;
const float l_193 = shared_memory_35[l_192];
const float l_194 = l_193 - l_mut_186;
const float l_195 = exp(l_194);
const uint32 l_196 = threadIdxGlobal * uint32(16);
const uint32 l_197 = l_196 + l_mut_190;
shared_memory_35[l_197] = l_195;
l_mut_51 = l_mut_51 + l_195;
}
const float l_198 = shared_memory_38[threadIdxGlobal];
const float l_199 = l_198 * l_189;
const float l_200 = l_199 + l_mut_51;
shared_memory_38[threadIdxGlobal] = l_200;
shared_memory_37[threadIdxGlobal] = l_mut_186;
shared_memory_39[threadIdxGlobal] = l_189;
} else {
shared_memory_39[threadIdxGlobal] = float(1.0);
}
}
__syncthreads();


for (uint32 l_mut_204 = uint32(0); l_mut_204 < uint32(8); ++l_mut_204) {
const uint32 l_205 = threadIdxGlobal * uint32(8);
const uint32 l_206 = l_205 + l_mut_204;
const uint32 l_207 = l_206 / uint32(16);
const uint32 l_208 = l_22 * uint32(16);
const uint32 l_209 = l_208 + l_207;
l_mut_186 = float(0.0);
const bool l_210 = l_209 < l_4;
if (l_210) {
const float l_211 = shared_memory_35[l_206];
l_mut_186 = l_211;
}
const __half l_212 = __half(l_mut_186);
shared_memory_201[l_206] = l_212;
}

for (uint32 l_mut_213 = uint32(0); l_mut_213 < uint32(128); ++l_mut_213) {
const uint32 l_214 = threadIdxGlobal * uint32(128);
const uint32 l_215 = l_214 + l_mut_213;
const uint32 l_216 = l_215 / uint32(256);
const uint32 l_217 = l_215 % uint32(256);
const uint32 l_218 = l_mut_79 * uint32(16);
const uint32 l_219 = l_218 + l_216;
l_mut_186 = float(0.0);
const bool l_220 = l_219 < l_25;
if (l_220) {
const bool l_221 = l_10 == uint32(2);
if (l_221) {
const uint32 l_222 = l_18 * l_6;
const uint32 l_223 = l_222 + l_24;
const uint32 l_224 = l_223 * l_8;
const uint32 l_225 = l_224 + l_219;
const uint32 l_226 = l_225 * uint32(256);
const uint32 l_227 = l_226 + l_217;
const __half l_228 = buffer_2[l_227];
l_mut_111 = l_228;
} else {
const uint32 l_229 = l_18 * l_9;
const uint32 l_230 = l_219 / l_8;
const uint32 l_231 = l_229 + l_230;
const uint32 l_232 = buffer_3[l_231];
const uint32 l_233 = l_219 % l_8;
const bool l_234 = l_10 == uint32(0);
if (l_234) {
const uint32 l_235 = l_232 * l_6;
const uint32 l_236 = l_235 + l_24;
const uint32 l_237 = l_236 * l_8;
const uint32 l_238 = l_237 + l_233;
const uint32 l_239 = l_238 * uint32(256);
const uint32 l_240 = l_239 + l_217;
const __half l_241 = buffer_2[l_240];
l_mut_97 = l_241;
} else {
const uint32 l_242 = l_232 * l_6;
const uint32 l_243 = l_242 + l_24;
const uint32 l_244 = l_243 * uint32(256);
const uint32 l_245 = l_244 + l_217;
const uint32 l_246 = l_245 * l_8;
const uint32 l_247 = l_246 + l_233;
const __half l_248 = buffer_2[l_247];
l_mut_97 = l_248;
}
l_mut_111 = l_mut_97;
}
const float l_249 = float(l_mut_111);
l_mut_186 = l_249;
}
const __half l_250 = __half(l_mut_186);
shared_memory_202[l_215] = l_250;
}
__syncthreads();


for (uint32 l_mut_251 = uint32(0); l_mut_251 < uint32(16); ++l_mut_251) {
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_252;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_258;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_b_267;
nvcuda::wmma::fill_fragment(frag_acc_252, float(0.0));
const uint32 l_253 = uint32(0) * uint32(16);
const uint32 l_254 = 256;
const uint32 l_255 = l_254 - uint32(0);
const uint32 l_256 = uint32(256) - l_253;
const uint32 l_257 = l_253 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_a_258, shared_memory_201 + l_257, uint32(16));
const uint32 l_259 = uint32(0) * uint32(16);
const uint32 l_260 = l_259 * uint32(256);
const uint32 l_261 = l_mut_251 * uint32(16);
const uint32 l_262 = l_260 + l_261;
const uint32 l_263 = 4096;
const uint32 l_264 = l_263 - uint32(0);
const uint32 l_265 = uint32(4096) - l_262;
const uint32 l_266 = l_262 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_b_267, shared_memory_202 + l_266, uint32(256));
nvcuda::wmma::mma_sync(frag_acc_252, frag_a_258, frag_b_267, frag_acc_252);
const uint32 l_268 = 256;
const uint32 l_269 = l_268 - uint32(0);
nvcuda::wmma::store_matrix_sync(shared_memory_203 + uint32(0), frag_acc_252, uint32(16), nvcuda::wmma::mem_row_major);
__syncthreads();


for (uint32 l_mut_270 = uint32(0); l_mut_270 < uint32(8); ++l_mut_270) {
const uint32 l_271 = threadIdxGlobal * uint32(8);
const uint32 l_272 = l_271 + l_mut_270;
const uint32 l_273 = l_272 / uint32(16);
const uint32 l_274 = l_272 % uint32(16);
const uint32 l_275 = l_273 * uint32(256);
const uint32 l_276 = l_mut_251 * uint32(16);
const uint32 l_277 = l_275 + l_276;
const uint32 l_278 = l_277 + l_274;
const float l_279 = shared_memory_36[l_278];
const float l_280 = shared_memory_39[l_273];
const float l_281 = l_279 * l_280;
const float l_282 = shared_memory_203[l_272];
const float l_283 = l_281 + l_282;
shared_memory_36[l_278] = l_283;
}
__syncthreads();

}
__syncthreads();

}

for (uint32 l_mut_284 = uint32(0); l_mut_284 < uint32(128); ++l_mut_284) {
const uint32 l_285 = threadIdxGlobal * uint32(128);
const uint32 l_286 = l_285 + l_mut_284;
const uint32 l_287 = l_286 / uint32(256);
const uint32 l_288 = l_286 % uint32(256);
const uint32 l_289 = l_22 * uint32(16);
const uint32 l_290 = l_289 + l_287;
const bool l_291 = l_290 < l_4;
if (l_291) {
l_mut_186 = float(1.0);
const float l_292 = shared_memory_38[l_287];
const bool l_293 = l_292 > float(0.0);
if (l_293) {
const float l_294 = shared_memory_38[l_287];
l_mut_186 = l_294;
}
const uint32 l_295 = l_290 * uint32(256);
const uint32 l_296 = l_33 + l_295;
const uint32 l_297 = l_296 + l_288;
const float l_298 = shared_memory_36[l_286];
const float l_299 = l_298 / l_mut_186;
const __half l_300 = __half(l_299);
buffer_5[l_297] = l_300;
}
}

}
