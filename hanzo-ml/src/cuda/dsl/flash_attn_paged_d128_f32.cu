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


extern "C" __global__ void __launch_bounds__(32)flash_attn_paged_f_f32 (
    const float* __restrict__ buffer_0, const float* __restrict__ buffer_1, const float* __restrict__ buffer_2, const uint32* __restrict__ buffer_3, const uint32* __restrict__ buffer_4, float* buffer_5, const float* __restrict__ buffer_6, const uint32* __restrict__ buffer_7, const __grid_constant__ info_st info
) {
uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * (blockDim.x * blockDim.y);
uint32 blockIdxGlobal = (blockIdx.z * gridDim.y * gridDim.x) + (blockIdx.y * gridDim.x) + blockIdx.x;
extern __shared__ __align__(4) uint8 dynamic_shared_mem[];
// Shared array size: 2048, 4096 bytes
__half *shared_memory_34 = reinterpret_cast<__half*>(&dynamic_shared_mem[0]);
// Shared array size: 256, 1024 bytes
float *shared_memory_35 = reinterpret_cast<float*>(&dynamic_shared_mem[4096]);
// Shared array size: 2048, 8192 bytes
float *shared_memory_36 = reinterpret_cast<float*>(&dynamic_shared_mem[5120]);
// Shared array size: 16, 64 bytes
float *shared_memory_37 = reinterpret_cast<float*>(&dynamic_shared_mem[13312]);
// Shared array size: 16, 64 bytes
float *shared_memory_38 = reinterpret_cast<float*>(&dynamic_shared_mem[13376]);
// Shared array size: 16, 64 bytes
float *shared_memory_39 = reinterpret_cast<float*>(&dynamic_shared_mem[13440]);
// Shared array size: 512, 1024 bytes
__half *shared_memory_79 = reinterpret_cast<__half*>(&dynamic_shared_mem[13504]);
// Shared array size: 256, 512 bytes
__half *shared_memory_201 = reinterpret_cast<__half*>(&dynamic_shared_mem[14528]);
// Shared array size: 512, 1024 bytes
__half *shared_memory_202 = reinterpret_cast<__half*>(&dynamic_shared_mem[15040]);
// Shared array size: 256, 1024 bytes
float *shared_memory_203 = reinterpret_cast<float*>(&dynamic_shared_mem[16064]);
uint32 l_mut_26;
float l_mut_51;
uint32 l_mut_61;
uint32 l_mut_70;
float l_mut_102;
float l_mut_116;
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
const uint32 l_33 = l_32 * uint32(128);

for (uint32 l_mut_40 = uint32(0); l_mut_40 < uint32(64); ++l_mut_40) {
const uint32 l_41 = threadIdxGlobal * uint32(64);
const uint32 l_42 = l_41 + l_mut_40;
shared_memory_36[l_42] = float(0.0);
}
const bool l_43 = threadIdxGlobal < uint32(16);
if (l_43) {
shared_memory_37[threadIdxGlobal] = float(-3.3999999521443642e38);
shared_memory_38[threadIdxGlobal] = float(0.0);
}

for (uint32 l_mut_44 = uint32(0); l_mut_44 < uint32(64); ++l_mut_44) {
const uint32 l_45 = threadIdxGlobal * uint32(64);
const uint32 l_46 = l_45 + l_mut_44;
const uint32 l_47 = l_46 / uint32(128);
const uint32 l_48 = l_46 % uint32(128);
const uint32 l_49 = l_22 * uint32(16);
const uint32 l_50 = l_49 + l_47;
l_mut_51 = float(0.0);
const bool l_52 = l_50 < l_4;
if (l_52) {
const uint32 l_53 = l_50 * uint32(128);
const uint32 l_54 = l_33 + l_53;
const uint32 l_55 = l_54 + l_48;
const float l_56 = buffer_0[l_55];
l_mut_51 = l_56;
}
const __half l_57 = __half(l_mut_51);
shared_memory_34[l_46] = l_57;
}
__syncthreads();

const uint32 l_58 = l_25 + uint32(16);
const uint32 l_59 = l_58 - uint32(1);
const uint32 l_60 = l_59 / uint32(16);
l_mut_61 = l_60;
const bool l_62 = l_7 != uint32(0);
if (l_62) {
const uint32 l_63 = l_22 * uint32(16);
const uint32 l_64 = l_mut_26 + l_63;
const uint32 l_65 = l_64 + uint32(16);
const uint32 l_66 = l_65 - uint32(1);
const uint32 l_67 = l_66 / uint32(16);
const uint32 l_68 = l_67 + uint32(1);
const bool l_69 = l_68 < l_mut_61;
if (l_69) {
l_mut_61 = l_68;
}
}
l_mut_70 = uint32(0);
const bool l_71 = l_12 > uint32(0);
if (l_71) {
const uint32 l_72 = l_22 * uint32(16);
const uint32 l_73 = l_mut_26 + l_72;
const bool l_74 = l_73 >= l_12;
if (l_74) {
const uint32 l_75 = l_73 - l_12;
const uint32 l_76 = l_75 + uint32(1);
const uint32 l_77 = l_76 / uint32(16);
l_mut_70 = l_77;
}
}

for (uint32 l_mut_78 = l_mut_70; l_mut_78 < l_mut_61; ++l_mut_78) {
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_80;
nvcuda::wmma::fill_fragment(frag_acc_80, float(0.0));

for (uint32 l_mut_81 = uint32(0); l_mut_81 < uint32(4); ++l_mut_81) {
const uint32 l_82 = l_mut_81 * uint32(32);

for (uint32 l_mut_83 = uint32(0); l_mut_83 < uint32(16); ++l_mut_83) {
const uint32 l_84 = threadIdxGlobal * uint32(16);
const uint32 l_85 = l_84 + l_mut_83;
const uint32 l_86 = l_85 / uint32(32);
const uint32 l_87 = l_85 % uint32(32);
const uint32 l_88 = l_mut_78 * uint32(16);
const uint32 l_89 = uint32(0) * uint32(16);
const uint32 l_90 = l_88 + l_89;
const uint32 l_91 = l_90 + l_86;
l_mut_51 = float(0.0);
const bool l_92 = l_91 < l_25;
if (l_92) {
const uint32 l_93 = l_82 + l_87;
const bool l_94 = l_10 == uint32(2);
if (l_94) {
const uint32 l_95 = l_18 * l_6;
const uint32 l_96 = l_95 + l_24;
const uint32 l_97 = l_96 * l_8;
const uint32 l_98 = l_97 + l_91;
const uint32 l_99 = l_98 * uint32(128);
const uint32 l_100 = l_99 + l_93;
const float l_101 = buffer_1[l_100];
l_mut_102 = l_101;
} else {
const uint32 l_103 = l_18 * l_9;
const uint32 l_104 = l_91 / l_8;
const uint32 l_105 = l_103 + l_104;
const uint32 l_106 = buffer_3[l_105];
const uint32 l_107 = l_91 % l_8;
const bool l_108 = l_10 == uint32(0);
if (l_108) {
const uint32 l_109 = l_106 * l_6;
const uint32 l_110 = l_109 + l_24;
const uint32 l_111 = l_110 * l_8;
const uint32 l_112 = l_111 + l_107;
const uint32 l_113 = l_112 * uint32(128);
const uint32 l_114 = l_113 + l_93;
const float l_115 = buffer_1[l_114];
l_mut_116 = l_115;
} else {
const uint32 l_117 = uint32(128) / l_11;
const uint32 l_118 = l_106 * l_6;
const uint32 l_119 = l_118 + l_24;
const uint32 l_120 = l_119 * l_117;
const uint32 l_121 = l_93 / l_11;
const uint32 l_122 = l_120 + l_121;
const uint32 l_123 = l_122 * l_8;
const uint32 l_124 = l_123 + l_107;
const uint32 l_125 = l_124 * l_11;
const uint32 l_126 = l_93 % l_11;
const uint32 l_127 = l_125 + l_126;
const float l_128 = buffer_1[l_127];
l_mut_116 = l_128;
}
l_mut_102 = l_mut_116;
}
l_mut_51 = l_mut_102;
}
const __half l_129 = __half(l_mut_51);
shared_memory_79[l_85] = l_129;
}
__syncthreads();


for (uint32 l_mut_130 = uint32(0); l_mut_130 < uint32(2); ++l_mut_130) {
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_137;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_143;
const uint32 l_131 = l_mut_130 * uint32(16);
const uint32 l_132 = l_82 + l_131;
const uint32 l_133 = 2048;
const uint32 l_134 = l_133 - uint32(0);
const uint32 l_135 = uint32(2048) - l_132;
const uint32 l_136 = l_132 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_a_137, shared_memory_34 + l_136, uint32(128));
const uint32 l_138 = l_mut_130 * uint32(16);
const uint32 l_139 = 512;
const uint32 l_140 = l_139 - uint32(0);
const uint32 l_141 = uint32(512) - l_138;
const uint32 l_142 = l_138 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_b_143, shared_memory_79 + l_142, uint32(32));
nvcuda::wmma::mma_sync(frag_acc_80, frag_a_137, frag_b_143, frag_acc_80);
}
__syncthreads();

}
const uint32 l_144 = uint32(0) * uint32(16);
const uint32 l_145 = 256;
const uint32 l_146 = l_145 - uint32(0);
const uint32 l_147 = uint32(256) - l_144;
const uint32 l_148 = l_144 + uint32(0);
nvcuda::wmma::store_matrix_sync(shared_memory_35 + l_148, frag_acc_80, uint32(16), nvcuda::wmma::mem_row_major);
__syncthreads();


for (uint32 l_mut_149 = uint32(0); l_mut_149 < uint32(8); ++l_mut_149) {
const uint32 l_150 = threadIdxGlobal * uint32(8);
const uint32 l_151 = l_150 + l_mut_149;
const uint32 l_152 = l_151 / uint32(16);
const uint32 l_153 = l_151 % uint32(16);
const uint32 l_154 = l_22 * uint32(16);
const uint32 l_155 = l_154 + l_152;
const uint32 l_156 = l_mut_78 * uint32(16);
const uint32 l_157 = l_156 + l_153;
const uint32 l_158 = l_mut_26 + l_155;
const bool l_159 = l_7 == uint32(0);
const bool l_160 = l_157 <= l_158;
const bool l_161 = l_159 || l_160;
const bool l_162 = l_12 == uint32(0);
const uint32 l_163 = l_157 + l_12;
const bool l_164 = l_163 > l_158;
const bool l_165 = l_162 || l_164;
const bool l_166 = l_155 < l_4;
const bool l_167 = l_157 < l_25;
const bool l_168 = l_166 && l_167;
const bool l_169 = l_168 && l_161;
const bool l_170 = l_169 && l_165;
if (l_170) {
const float l_171 = shared_memory_35[l_151];
const float l_172 = l_171 * l_2;
l_mut_116 = l_172;
const bool l_173 = l_3 > float(0.0);
if (l_173) {
const float l_174 = l_mut_116 / l_3;
const float l_175 = tanh(l_174);
const float l_176 = l_3 * l_175;
l_mut_116 = l_176;
}
shared_memory_35[l_151] = l_mut_116;
} else {
shared_memory_35[l_151] = float(-3.3999999521443642e38);
}
}
__syncthreads();

const bool l_177 = threadIdxGlobal < uint32(16);
if (l_177) {
const uint32 l_178 = l_22 * uint32(16);
const uint32 l_179 = l_178 + threadIdxGlobal;
const bool l_180 = l_179 < l_4;
if (l_180) {
l_mut_116 = float(-3.3999999521443642e38);

for (uint32 l_mut_181 = uint32(0); l_mut_181 < uint32(16); ++l_mut_181) {
const uint32 l_182 = threadIdxGlobal * uint32(16);
const uint32 l_183 = l_182 + l_mut_181;
const float l_184 = shared_memory_35[l_183];
const bool l_185 = l_184 > l_mut_116;
if (l_185) {
l_mut_116 = l_184;
}
}
const float l_186 = shared_memory_37[threadIdxGlobal];
l_mut_102 = l_186;
const bool l_187 = l_mut_116 > l_mut_102;
if (l_187) {
l_mut_102 = l_mut_116;
}
const float l_188 = l_186 - l_mut_102;
const float l_189 = exp(l_188);
l_mut_116 = float(0.0);

for (uint32 l_mut_190 = uint32(0); l_mut_190 < uint32(16); ++l_mut_190) {
const uint32 l_191 = threadIdxGlobal * uint32(16);
const uint32 l_192 = l_191 + l_mut_190;
const float l_193 = shared_memory_35[l_192];
const float l_194 = l_193 - l_mut_102;
const float l_195 = exp(l_194);
const uint32 l_196 = threadIdxGlobal * uint32(16);
const uint32 l_197 = l_196 + l_mut_190;
shared_memory_35[l_197] = l_195;
l_mut_116 = l_mut_116 + l_195;
}
const float l_198 = shared_memory_38[threadIdxGlobal];
const float l_199 = l_198 * l_189;
const float l_200 = l_199 + l_mut_116;
shared_memory_38[threadIdxGlobal] = l_200;
shared_memory_37[threadIdxGlobal] = l_mut_102;
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
l_mut_116 = float(0.0);
const bool l_210 = l_209 < l_4;
if (l_210) {
const float l_211 = shared_memory_35[l_206];
l_mut_116 = l_211;
}
const __half l_212 = __half(l_mut_116);
shared_memory_201[l_206] = l_212;
}
__syncthreads();


for (uint32 l_mut_213 = uint32(0); l_mut_213 < uint32(4); ++l_mut_213) {
const uint32 l_214 = l_mut_213 * uint32(32);

for (uint32 l_mut_215 = uint32(0); l_mut_215 < uint32(16); ++l_mut_215) {
const uint32 l_216 = threadIdxGlobal * uint32(16);
const uint32 l_217 = l_216 + l_mut_215;
const uint32 l_218 = l_217 / uint32(32);
const uint32 l_219 = l_217 % uint32(32);
const uint32 l_220 = l_mut_78 * uint32(16);
const uint32 l_221 = l_220 + l_218;
l_mut_116 = float(0.0);
const bool l_222 = l_221 < l_25;
if (l_222) {
const uint32 l_223 = l_214 + l_219;
const bool l_224 = l_10 == uint32(2);
if (l_224) {
const uint32 l_225 = l_18 * l_6;
const uint32 l_226 = l_225 + l_24;
const uint32 l_227 = l_226 * l_8;
const uint32 l_228 = l_227 + l_221;
const uint32 l_229 = l_228 * uint32(128);
const uint32 l_230 = l_229 + l_223;
const float l_231 = buffer_2[l_230];
l_mut_102 = l_231;
} else {
const uint32 l_232 = l_18 * l_9;
const uint32 l_233 = l_221 / l_8;
const uint32 l_234 = l_232 + l_233;
const uint32 l_235 = buffer_3[l_234];
const uint32 l_236 = l_221 % l_8;
const bool l_237 = l_10 == uint32(0);
if (l_237) {
const uint32 l_238 = l_235 * l_6;
const uint32 l_239 = l_238 + l_24;
const uint32 l_240 = l_239 * l_8;
const uint32 l_241 = l_240 + l_236;
const uint32 l_242 = l_241 * uint32(128);
const uint32 l_243 = l_242 + l_223;
const float l_244 = buffer_2[l_243];
l_mut_51 = l_244;
} else {
const uint32 l_245 = l_235 * l_6;
const uint32 l_246 = l_245 + l_24;
const uint32 l_247 = l_246 * uint32(128);
const uint32 l_248 = l_247 + l_223;
const uint32 l_249 = l_248 * l_8;
const uint32 l_250 = l_249 + l_236;
const float l_251 = buffer_2[l_250];
l_mut_51 = l_251;
}
l_mut_102 = l_mut_51;
}
l_mut_116 = l_mut_102;
}
const __half l_252 = __half(l_mut_116);
shared_memory_202[l_217] = l_252;
}
__syncthreads();


for (uint32 l_mut_253 = uint32(0); l_mut_253 < uint32(2); ++l_mut_253) {
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_254;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_260;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_b_269;
nvcuda::wmma::fill_fragment(frag_acc_254, float(0.0));
const uint32 l_255 = uint32(0) * uint32(16);
const uint32 l_256 = 256;
const uint32 l_257 = l_256 - uint32(0);
const uint32 l_258 = uint32(256) - l_255;
const uint32 l_259 = l_255 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_a_260, shared_memory_201 + l_259, uint32(16));
const uint32 l_261 = uint32(0) * uint32(16);
const uint32 l_262 = l_261 * uint32(32);
const uint32 l_263 = l_mut_253 * uint32(16);
const uint32 l_264 = l_262 + l_263;
const uint32 l_265 = 512;
const uint32 l_266 = l_265 - uint32(0);
const uint32 l_267 = uint32(512) - l_264;
const uint32 l_268 = l_264 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_b_269, shared_memory_202 + l_268, uint32(32));
nvcuda::wmma::mma_sync(frag_acc_254, frag_a_260, frag_b_269, frag_acc_254);
const uint32 l_270 = 256;
const uint32 l_271 = l_270 - uint32(0);
nvcuda::wmma::store_matrix_sync(shared_memory_203 + uint32(0), frag_acc_254, uint32(16), nvcuda::wmma::mem_row_major);
__syncthreads();


for (uint32 l_mut_272 = uint32(0); l_mut_272 < uint32(8); ++l_mut_272) {
const uint32 l_273 = threadIdxGlobal * uint32(8);
const uint32 l_274 = l_273 + l_mut_272;
const uint32 l_275 = l_274 / uint32(16);
const uint32 l_276 = l_274 % uint32(16);
const uint32 l_277 = l_275 * uint32(128);
const uint32 l_278 = l_277 + l_214;
const uint32 l_279 = l_mut_253 * uint32(16);
const uint32 l_280 = l_278 + l_279;
const uint32 l_281 = l_280 + l_276;
const float l_282 = shared_memory_36[l_281];
const float l_283 = shared_memory_39[l_275];
const float l_284 = l_282 * l_283;
const float l_285 = shared_memory_203[l_274];
const float l_286 = l_284 + l_285;
shared_memory_36[l_281] = l_286;
}
__syncthreads();

}
}
__syncthreads();

}

for (uint32 l_mut_287 = uint32(0); l_mut_287 < uint32(64); ++l_mut_287) {
const uint32 l_288 = threadIdxGlobal * uint32(64);
const uint32 l_289 = l_288 + l_mut_287;
const uint32 l_290 = l_289 / uint32(128);
const uint32 l_291 = l_289 % uint32(128);
const uint32 l_292 = l_22 * uint32(16);
const uint32 l_293 = l_292 + l_290;
const bool l_294 = l_293 < l_4;
if (l_294) {
l_mut_116 = float(1.0);
const float l_295 = shared_memory_38[l_290];
const bool l_296 = l_295 > float(0.0);
if (l_296) {
const float l_297 = shared_memory_38[l_290];
l_mut_116 = l_297;
}
const uint32 l_298 = l_293 * uint32(128);
const uint32 l_299 = l_33 + l_298;
const uint32 l_300 = l_299 + l_291;
const float l_301 = shared_memory_36[l_289];
const float l_302 = l_301 / l_mut_116;
buffer_5[l_300] = l_302;
}
}

}
