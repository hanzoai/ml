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
// Shared array size: 2048, 4096 bytes
__half *shared_memory_79 = reinterpret_cast<__half*>(&dynamic_shared_mem[13504]);
// Shared array size: 256, 512 bytes
__half *shared_memory_198 = reinterpret_cast<__half*>(&dynamic_shared_mem[17600]);
// Shared array size: 2048, 4096 bytes
__half *shared_memory_199 = reinterpret_cast<__half*>(&dynamic_shared_mem[18112]);
// Shared array size: 256, 1024 bytes
float *shared_memory_200 = reinterpret_cast<float*>(&dynamic_shared_mem[22208]);
uint32 l_mut_26;
float l_mut_51;
uint32 l_mut_61;
uint32 l_mut_70;
float l_mut_96;
float l_mut_110;
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
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_124;

for (uint32 l_mut_80 = uint32(0); l_mut_80 < uint32(64); ++l_mut_80) {
const uint32 l_81 = threadIdxGlobal * uint32(64);
const uint32 l_82 = l_81 + l_mut_80;
const uint32 l_83 = l_82 / uint32(128);
const uint32 l_84 = l_82 % uint32(128);
const uint32 l_85 = l_mut_78 * uint32(16);
const uint32 l_86 = l_85 + l_83;
l_mut_51 = float(0.0);
const bool l_87 = l_86 < l_25;
if (l_87) {
const bool l_88 = l_10 == uint32(2);
if (l_88) {
const uint32 l_89 = l_18 * l_6;
const uint32 l_90 = l_89 + l_24;
const uint32 l_91 = l_90 * l_8;
const uint32 l_92 = l_91 + l_86;
const uint32 l_93 = l_92 * uint32(128);
const uint32 l_94 = l_93 + l_84;
const float l_95 = buffer_1[l_94];
l_mut_96 = l_95;
} else {
const uint32 l_97 = l_18 * l_9;
const uint32 l_98 = l_86 / l_8;
const uint32 l_99 = l_97 + l_98;
const uint32 l_100 = buffer_3[l_99];
const uint32 l_101 = l_86 % l_8;
const bool l_102 = l_10 == uint32(0);
if (l_102) {
const uint32 l_103 = l_100 * l_6;
const uint32 l_104 = l_103 + l_24;
const uint32 l_105 = l_104 * l_8;
const uint32 l_106 = l_105 + l_101;
const uint32 l_107 = l_106 * uint32(128);
const uint32 l_108 = l_107 + l_84;
const float l_109 = buffer_1[l_108];
l_mut_110 = l_109;
} else {
const uint32 l_111 = uint32(128) / l_11;
const uint32 l_112 = l_100 * l_6;
const uint32 l_113 = l_112 + l_24;
const uint32 l_114 = l_113 * l_111;
const uint32 l_115 = l_84 / l_11;
const uint32 l_116 = l_114 + l_115;
const uint32 l_117 = l_116 * l_8;
const uint32 l_118 = l_117 + l_101;
const uint32 l_119 = l_118 * l_11;
const uint32 l_120 = l_84 % l_11;
const uint32 l_121 = l_119 + l_120;
const float l_122 = buffer_1[l_121];
l_mut_110 = l_122;
}
l_mut_96 = l_mut_110;
}
l_mut_51 = l_mut_96;
}
const __half l_123 = __half(l_mut_51);
shared_memory_79[l_82] = l_123;
}
__syncthreads();

nvcuda::wmma::fill_fragment(frag_acc_124, float(0.0));

for (uint32 l_mut_125 = uint32(0); l_mut_125 < uint32(8); ++l_mut_125) {
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_131;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_140;
const uint32 l_126 = l_mut_125 * uint32(16);
const uint32 l_127 = 2048;
const uint32 l_128 = l_127 - uint32(0);
const uint32 l_129 = uint32(2048) - l_126;
const uint32 l_130 = l_126 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_a_131, shared_memory_34 + l_130, uint32(128));
const uint32 l_132 = uint32(0) * uint32(16);
const uint32 l_133 = l_132 * uint32(128);
const uint32 l_134 = l_mut_125 * uint32(16);
const uint32 l_135 = l_133 + l_134;
const uint32 l_136 = 2048;
const uint32 l_137 = l_136 - uint32(0);
const uint32 l_138 = uint32(2048) - l_135;
const uint32 l_139 = l_135 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_b_140, shared_memory_79 + l_139, uint32(128));
nvcuda::wmma::mma_sync(frag_acc_124, frag_a_131, frag_b_140, frag_acc_124);
}
const uint32 l_141 = uint32(0) * uint32(16);
const uint32 l_142 = 256;
const uint32 l_143 = l_142 - uint32(0);
const uint32 l_144 = uint32(256) - l_141;
const uint32 l_145 = l_141 + uint32(0);
nvcuda::wmma::store_matrix_sync(shared_memory_35 + l_145, frag_acc_124, uint32(16), nvcuda::wmma::mem_row_major);
__syncthreads();


for (uint32 l_mut_146 = uint32(0); l_mut_146 < uint32(8); ++l_mut_146) {
const uint32 l_147 = threadIdxGlobal * uint32(8);
const uint32 l_148 = l_147 + l_mut_146;
const uint32 l_149 = l_148 / uint32(16);
const uint32 l_150 = l_148 % uint32(16);
const uint32 l_151 = l_22 * uint32(16);
const uint32 l_152 = l_151 + l_149;
const uint32 l_153 = l_mut_78 * uint32(16);
const uint32 l_154 = l_153 + l_150;
const uint32 l_155 = l_mut_26 + l_152;
const bool l_156 = l_7 == uint32(0);
const bool l_157 = l_154 <= l_155;
const bool l_158 = l_156 || l_157;
const bool l_159 = l_12 == uint32(0);
const uint32 l_160 = l_154 + l_12;
const bool l_161 = l_160 > l_155;
const bool l_162 = l_159 || l_161;
const bool l_163 = l_152 < l_4;
const bool l_164 = l_154 < l_25;
const bool l_165 = l_163 && l_164;
const bool l_166 = l_165 && l_158;
const bool l_167 = l_166 && l_162;
if (l_167) {
const float l_168 = shared_memory_35[l_148];
const float l_169 = l_168 * l_2;
l_mut_110 = l_169;
const bool l_170 = l_3 > float(0.0);
if (l_170) {
const float l_171 = l_mut_110 / l_3;
const float l_172 = tanh(l_171);
const float l_173 = l_3 * l_172;
l_mut_110 = l_173;
}
shared_memory_35[l_148] = l_mut_110;
} else {
shared_memory_35[l_148] = float(-3.3999999521443642e38);
}
}
__syncthreads();

const bool l_174 = threadIdxGlobal < uint32(16);
if (l_174) {
const uint32 l_175 = l_22 * uint32(16);
const uint32 l_176 = l_175 + threadIdxGlobal;
const bool l_177 = l_176 < l_4;
if (l_177) {
l_mut_110 = float(-3.3999999521443642e38);

for (uint32 l_mut_178 = uint32(0); l_mut_178 < uint32(16); ++l_mut_178) {
const uint32 l_179 = threadIdxGlobal * uint32(16);
const uint32 l_180 = l_179 + l_mut_178;
const float l_181 = shared_memory_35[l_180];
const bool l_182 = l_181 > l_mut_110;
if (l_182) {
l_mut_110 = l_181;
}
}
const float l_183 = shared_memory_37[threadIdxGlobal];
l_mut_96 = l_183;
const bool l_184 = l_mut_110 > l_mut_96;
if (l_184) {
l_mut_96 = l_mut_110;
}
const float l_185 = l_183 - l_mut_96;
const float l_186 = exp(l_185);
l_mut_110 = float(0.0);

for (uint32 l_mut_187 = uint32(0); l_mut_187 < uint32(16); ++l_mut_187) {
const uint32 l_188 = threadIdxGlobal * uint32(16);
const uint32 l_189 = l_188 + l_mut_187;
const float l_190 = shared_memory_35[l_189];
const float l_191 = l_190 - l_mut_96;
const float l_192 = exp(l_191);
const uint32 l_193 = threadIdxGlobal * uint32(16);
const uint32 l_194 = l_193 + l_mut_187;
shared_memory_35[l_194] = l_192;
l_mut_110 = l_mut_110 + l_192;
}
const float l_195 = shared_memory_38[threadIdxGlobal];
const float l_196 = l_195 * l_186;
const float l_197 = l_196 + l_mut_110;
shared_memory_38[threadIdxGlobal] = l_197;
shared_memory_37[threadIdxGlobal] = l_mut_96;
shared_memory_39[threadIdxGlobal] = l_186;
} else {
shared_memory_39[threadIdxGlobal] = float(1.0);
}
}
__syncthreads();


for (uint32 l_mut_201 = uint32(0); l_mut_201 < uint32(8); ++l_mut_201) {
const uint32 l_202 = threadIdxGlobal * uint32(8);
const uint32 l_203 = l_202 + l_mut_201;
const uint32 l_204 = l_203 / uint32(16);
const uint32 l_205 = l_22 * uint32(16);
const uint32 l_206 = l_205 + l_204;
l_mut_110 = float(0.0);
const bool l_207 = l_206 < l_4;
if (l_207) {
const float l_208 = shared_memory_35[l_203];
l_mut_110 = l_208;
}
const __half l_209 = __half(l_mut_110);
shared_memory_198[l_203] = l_209;
}

for (uint32 l_mut_210 = uint32(0); l_mut_210 < uint32(64); ++l_mut_210) {
const uint32 l_211 = threadIdxGlobal * uint32(64);
const uint32 l_212 = l_211 + l_mut_210;
const uint32 l_213 = l_212 / uint32(128);
const uint32 l_214 = l_212 % uint32(128);
const uint32 l_215 = l_mut_78 * uint32(16);
const uint32 l_216 = l_215 + l_213;
l_mut_110 = float(0.0);
const bool l_217 = l_216 < l_25;
if (l_217) {
const bool l_218 = l_10 == uint32(2);
if (l_218) {
const uint32 l_219 = l_18 * l_6;
const uint32 l_220 = l_219 + l_24;
const uint32 l_221 = l_220 * l_8;
const uint32 l_222 = l_221 + l_216;
const uint32 l_223 = l_222 * uint32(128);
const uint32 l_224 = l_223 + l_214;
const float l_225 = buffer_2[l_224];
l_mut_96 = l_225;
} else {
const uint32 l_226 = l_18 * l_9;
const uint32 l_227 = l_216 / l_8;
const uint32 l_228 = l_226 + l_227;
const uint32 l_229 = buffer_3[l_228];
const uint32 l_230 = l_216 % l_8;
const bool l_231 = l_10 == uint32(0);
if (l_231) {
const uint32 l_232 = l_229 * l_6;
const uint32 l_233 = l_232 + l_24;
const uint32 l_234 = l_233 * l_8;
const uint32 l_235 = l_234 + l_230;
const uint32 l_236 = l_235 * uint32(128);
const uint32 l_237 = l_236 + l_214;
const float l_238 = buffer_2[l_237];
l_mut_51 = l_238;
} else {
const uint32 l_239 = l_229 * l_6;
const uint32 l_240 = l_239 + l_24;
const uint32 l_241 = l_240 * uint32(128);
const uint32 l_242 = l_241 + l_214;
const uint32 l_243 = l_242 * l_8;
const uint32 l_244 = l_243 + l_230;
const float l_245 = buffer_2[l_244];
l_mut_51 = l_245;
}
l_mut_96 = l_mut_51;
}
l_mut_110 = l_mut_96;
}
const __half l_246 = __half(l_mut_110);
shared_memory_199[l_212] = l_246;
}
__syncthreads();


for (uint32 l_mut_247 = uint32(0); l_mut_247 < uint32(8); ++l_mut_247) {
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_248;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_254;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_b_263;
nvcuda::wmma::fill_fragment(frag_acc_248, float(0.0));
const uint32 l_249 = uint32(0) * uint32(16);
const uint32 l_250 = 256;
const uint32 l_251 = l_250 - uint32(0);
const uint32 l_252 = uint32(256) - l_249;
const uint32 l_253 = l_249 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_a_254, shared_memory_198 + l_253, uint32(16));
const uint32 l_255 = uint32(0) * uint32(16);
const uint32 l_256 = l_255 * uint32(128);
const uint32 l_257 = l_mut_247 * uint32(16);
const uint32 l_258 = l_256 + l_257;
const uint32 l_259 = 2048;
const uint32 l_260 = l_259 - uint32(0);
const uint32 l_261 = uint32(2048) - l_258;
const uint32 l_262 = l_258 + uint32(0);
nvcuda::wmma::load_matrix_sync(frag_b_263, shared_memory_199 + l_262, uint32(128));
nvcuda::wmma::mma_sync(frag_acc_248, frag_a_254, frag_b_263, frag_acc_248);
const uint32 l_264 = 256;
const uint32 l_265 = l_264 - uint32(0);
nvcuda::wmma::store_matrix_sync(shared_memory_200 + uint32(0), frag_acc_248, uint32(16), nvcuda::wmma::mem_row_major);
__syncthreads();


for (uint32 l_mut_266 = uint32(0); l_mut_266 < uint32(8); ++l_mut_266) {
const uint32 l_267 = threadIdxGlobal * uint32(8);
const uint32 l_268 = l_267 + l_mut_266;
const uint32 l_269 = l_268 / uint32(16);
const uint32 l_270 = l_268 % uint32(16);
const uint32 l_271 = l_269 * uint32(128);
const uint32 l_272 = l_mut_247 * uint32(16);
const uint32 l_273 = l_271 + l_272;
const uint32 l_274 = l_273 + l_270;
const float l_275 = shared_memory_36[l_274];
const float l_276 = shared_memory_39[l_269];
const float l_277 = l_275 * l_276;
const float l_278 = shared_memory_200[l_268];
const float l_279 = l_277 + l_278;
shared_memory_36[l_274] = l_279;
}
__syncthreads();

}
__syncthreads();

}

for (uint32 l_mut_280 = uint32(0); l_mut_280 < uint32(64); ++l_mut_280) {
const uint32 l_281 = threadIdxGlobal * uint32(64);
const uint32 l_282 = l_281 + l_mut_280;
const uint32 l_283 = l_282 / uint32(128);
const uint32 l_284 = l_282 % uint32(128);
const uint32 l_285 = l_22 * uint32(16);
const uint32 l_286 = l_285 + l_283;
const bool l_287 = l_286 < l_4;
if (l_287) {
l_mut_110 = float(1.0);
const float l_288 = shared_memory_38[l_283];
const bool l_289 = l_288 > float(0.0);
if (l_289) {
const float l_290 = shared_memory_38[l_283];
l_mut_110 = l_290;
}
const uint32 l_291 = l_286 * uint32(128);
const uint32 l_292 = l_33 + l_291;
const uint32 l_293 = l_292 + l_284;
const float l_294 = shared_memory_36[l_282];
const float l_295 = l_294 / l_mut_110;
buffer_5[l_293] = l_295;
}
}

}
