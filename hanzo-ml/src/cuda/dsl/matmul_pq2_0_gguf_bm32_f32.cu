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
// Shared array size: 1280, 2560 bytes
__half *shared_memory_41 = reinterpret_cast<__half*>(&dynamic_shared_mem[2048]);
// Shared array size: 2560, 5120 bytes
__half *shared_memory_42 = reinterpret_cast<__half*>(&dynamic_shared_mem[4608]);
__half l_arr_39[8];

__half l_arr_40[16];

nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_48;
nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float> frag_acc_49;
uint32 l_mut_16;
uint32 l_mut_23;
__half l_mut_52;
uint32 l_mut_53;
const uint32 l_0 = threadIdxGlobal / uint32(32);
const uint32 l_1 = l_0 / uint32(2);
const uint32 l_2 = l_0 % uint32(2);
const uint32 l_3 = buffer_3[uint32(0)];
const uint32 l_4 = buffer_3[uint32(1)];
const uint32 l_5 = buffer_3[uint32(2)];
const uint32 l_6 = blockIdx.x * uint32(32);
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

for (uint32 l_mut_50 = uint32(0); l_mut_50 < l_8; ++l_mut_50) {
const __half l_51 = __half(float(0.0));
l_mut_52 = l_51;
l_mut_23 = uint32(0);
l_mut_16 = uint32(0);
l_mut_53 = uint32(0);
const bool l_54 = threadIdxGlobal < uint32(128);
const bool l_55 = bool(true) || l_54;
if (l_55) {
const uint32 l_56 = l_37 + l_mut_50;
const uint32 l_57 = l_56 / uint32(2);
const uint32 l_58 = l_56 % uint32(2);
const uint32 l_59 = uint32(17) * l_57;
const uint32 l_60 = uint32(8) * l_58;
const uint32 l_61 = l_59 + l_60;
const uint32 l_62 = buffer_0[l_61];
const uint32 l_63 = uint32(16) * l_58;
const uint32 l_64 = l_62 >> l_63;
const uint32 l_65 = l_64 & uint32(65535);
const __half_2 l_66 = reinterpret_cast<__half_2 const&>(l_65);
const __half l_67 = l_66.i_0;
l_mut_52 = l_67;
const uint32 l_68 = uint32(17) * l_57;
const uint32 l_69 = uint32(9) * l_58;
const uint32 l_70 = l_68 + l_69;
l_mut_23 = l_70;
const uint32 l_71 = uint32(1) - l_58;
const uint32 l_72 = uint32(16) * l_71;
l_mut_16 = l_72;
const uint32 l_73 = uint32(1) - l_58;
l_mut_53 = l_73;
}

for (uint32 l_mut_74 = uint32(0); l_mut_74 < uint32(4); ++l_mut_74) {
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_243;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_249;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_255;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_261;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_267;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_273;
const uint32 l_75 = l_mut_50 * uint32(4);
const uint32 l_76 = l_75 + l_mut_74;
const uint32 l_77 = l_76 * uint32(8);
const uint32 l_78 = uint32(2) * l_mut_74;
const uint32 l_79 = l_78 + l_32;
const uint32 l_80 = l_27 + l_77;
const float_4 l_81 = buffer_1[l_80];
const uint32 l_82 = l_80 + uint32(1);
const float_4 l_83 = buffer_1[l_82];
const float l_84 = l_81.i_0;
const __half l_85 = __half(l_84);
l_arr_39[uint32(0)] = l_85;
const float l_86 = l_83.i_0;
const __half l_87 = __half(l_86);
l_arr_39[uint32(4)] = l_87;
const float l_88 = l_81.i_1;
const __half l_89 = __half(l_88);
l_arr_39[uint32(1)] = l_89;
const float l_90 = l_83.i_1;
const __half l_91 = __half(l_90);
l_arr_39[uint32(5)] = l_91;
const float l_92 = l_81.i_2;
const __half l_93 = __half(l_92);
l_arr_39[uint32(2)] = l_93;
const float l_94 = l_83.i_2;
const __half l_95 = __half(l_94);
l_arr_39[uint32(6)] = l_95;
const float l_96 = l_81.i_3;
const __half l_97 = __half(l_96);
l_arr_39[uint32(3)] = l_97;
const float l_98 = l_83.i_3;
const __half l_99 = __half(l_98);
l_arr_39[uint32(7)] = l_99;
const uint32 l_100 = l_mut_23 + l_79;
const uint32 l_101 = buffer_0[l_100];
const uint32 l_102 = l_mut_23 + l_79;
const uint32 l_103 = l_102 + l_mut_53;
const uint32 l_104 = buffer_0[l_103];
const uint32 l_105 = l_101 >> l_mut_16;
const uint32 l_106 = l_104 << uint32(15);
const uint32 l_107 = uint32(17) - l_mut_16;
const uint32 l_108 = l_106 << l_107;
const uint32 l_109 = l_105 | l_108;
const uint32 l_110 = l_109 >> uint32(0);
const uint32 l_111 = l_110 & uint32(3);
const __half l_112 = __half(l_111);
const __half l_113 = l_112 * l_mut_52;
const __half l_114 = l_113 - l_mut_52;
l_arr_40[uint32(0)] = l_114;
const uint32 l_115 = l_109 >> uint32(2);
const uint32 l_116 = l_115 & uint32(3);
const __half l_117 = __half(l_116);
const __half l_118 = l_117 * l_mut_52;
const __half l_119 = l_118 - l_mut_52;
l_arr_40[uint32(1)] = l_119;
const uint32 l_120 = l_109 >> uint32(4);
const uint32 l_121 = l_120 & uint32(3);
const __half l_122 = __half(l_121);
const __half l_123 = l_122 * l_mut_52;
const __half l_124 = l_123 - l_mut_52;
l_arr_40[uint32(2)] = l_124;
const uint32 l_125 = l_109 >> uint32(6);
const uint32 l_126 = l_125 & uint32(3);
const __half l_127 = __half(l_126);
const __half l_128 = l_127 * l_mut_52;
const __half l_129 = l_128 - l_mut_52;
l_arr_40[uint32(3)] = l_129;
const uint32 l_130 = l_109 >> uint32(8);
const uint32 l_131 = l_130 & uint32(3);
const __half l_132 = __half(l_131);
const __half l_133 = l_132 * l_mut_52;
const __half l_134 = l_133 - l_mut_52;
l_arr_40[uint32(4)] = l_134;
const uint32 l_135 = l_109 >> uint32(10);
const uint32 l_136 = l_135 & uint32(3);
const __half l_137 = __half(l_136);
const __half l_138 = l_137 * l_mut_52;
const __half l_139 = l_138 - l_mut_52;
l_arr_40[uint32(5)] = l_139;
const uint32 l_140 = l_109 >> uint32(12);
const uint32 l_141 = l_140 & uint32(3);
const __half l_142 = __half(l_141);
const __half l_143 = l_142 * l_mut_52;
const __half l_144 = l_143 - l_mut_52;
l_arr_40[uint32(6)] = l_144;
const uint32 l_145 = l_109 >> uint32(14);
const uint32 l_146 = l_145 & uint32(3);
const __half l_147 = __half(l_146);
const __half l_148 = l_147 * l_mut_52;
const __half l_149 = l_148 - l_mut_52;
l_arr_40[uint32(7)] = l_149;
const uint32 l_150 = l_109 >> uint32(16);
const uint32 l_151 = l_150 & uint32(3);
const __half l_152 = __half(l_151);
const __half l_153 = l_152 * l_mut_52;
const __half l_154 = l_153 - l_mut_52;
l_arr_40[uint32(8)] = l_154;
const uint32 l_155 = l_109 >> uint32(18);
const uint32 l_156 = l_155 & uint32(3);
const __half l_157 = __half(l_156);
const __half l_158 = l_157 * l_mut_52;
const __half l_159 = l_158 - l_mut_52;
l_arr_40[uint32(9)] = l_159;
const uint32 l_160 = l_109 >> uint32(20);
const uint32 l_161 = l_160 & uint32(3);
const __half l_162 = __half(l_161);
const __half l_163 = l_162 * l_mut_52;
const __half l_164 = l_163 - l_mut_52;
l_arr_40[uint32(10)] = l_164;
const uint32 l_165 = l_109 >> uint32(22);
const uint32 l_166 = l_165 & uint32(3);
const __half l_167 = __half(l_166);
const __half l_168 = l_167 * l_mut_52;
const __half l_169 = l_168 - l_mut_52;
l_arr_40[uint32(11)] = l_169;
const uint32 l_170 = l_109 >> uint32(24);
const uint32 l_171 = l_170 & uint32(3);
const __half l_172 = __half(l_171);
const __half l_173 = l_172 * l_mut_52;
const __half l_174 = l_173 - l_mut_52;
l_arr_40[uint32(12)] = l_174;
const uint32 l_175 = l_109 >> uint32(26);
const uint32 l_176 = l_175 & uint32(3);
const __half l_177 = __half(l_176);
const __half l_178 = l_177 * l_mut_52;
const __half l_179 = l_178 - l_mut_52;
l_arr_40[uint32(13)] = l_179;
const uint32 l_180 = l_109 >> uint32(28);
const uint32 l_181 = l_180 & uint32(3);
const __half l_182 = __half(l_181);
const __half l_183 = l_182 * l_mut_52;
const __half l_184 = l_183 - l_mut_52;
l_arr_40[uint32(14)] = l_184;
const uint32 l_185 = l_109 >> uint32(30);
const uint32 l_186 = l_185 & uint32(3);
const __half l_187 = __half(l_186);
const __half l_188 = l_187 * l_mut_52;
const __half l_189 = l_188 - l_mut_52;
l_arr_40[uint32(15)] = l_189;
__syncthreads();

const uint32 l_190 = l_44 + uint32(0);
const __half l_191 = l_arr_39[uint32(0)];
shared_memory_41[l_190] = l_191;
const uint32 l_192 = l_44 + uint32(1);
const __half l_193 = l_arr_39[uint32(1)];
shared_memory_41[l_192] = l_193;
const uint32 l_194 = l_44 + uint32(2);
const __half l_195 = l_arr_39[uint32(2)];
shared_memory_41[l_194] = l_195;
const uint32 l_196 = l_44 + uint32(3);
const __half l_197 = l_arr_39[uint32(3)];
shared_memory_41[l_196] = l_197;
const uint32 l_198 = l_44 + uint32(4);
const __half l_199 = l_arr_39[uint32(4)];
shared_memory_41[l_198] = l_199;
const uint32 l_200 = l_44 + uint32(5);
const __half l_201 = l_arr_39[uint32(5)];
shared_memory_41[l_200] = l_201;
const uint32 l_202 = l_44 + uint32(6);
const __half l_203 = l_arr_39[uint32(6)];
shared_memory_41[l_202] = l_203;
const uint32 l_204 = l_44 + uint32(7);
const __half l_205 = l_arr_39[uint32(7)];
shared_memory_41[l_204] = l_205;
const uint32 l_206 = l_47 + uint32(0);
const __half l_207 = l_arr_40[uint32(0)];
shared_memory_42[l_206] = l_207;
const uint32 l_208 = l_47 + uint32(1);
const __half l_209 = l_arr_40[uint32(1)];
shared_memory_42[l_208] = l_209;
const uint32 l_210 = l_47 + uint32(2);
const __half l_211 = l_arr_40[uint32(2)];
shared_memory_42[l_210] = l_211;
const uint32 l_212 = l_47 + uint32(3);
const __half l_213 = l_arr_40[uint32(3)];
shared_memory_42[l_212] = l_213;
const uint32 l_214 = l_47 + uint32(4);
const __half l_215 = l_arr_40[uint32(4)];
shared_memory_42[l_214] = l_215;
const uint32 l_216 = l_47 + uint32(5);
const __half l_217 = l_arr_40[uint32(5)];
shared_memory_42[l_216] = l_217;
const uint32 l_218 = l_47 + uint32(6);
const __half l_219 = l_arr_40[uint32(6)];
shared_memory_42[l_218] = l_219;
const uint32 l_220 = l_47 + uint32(7);
const __half l_221 = l_arr_40[uint32(7)];
shared_memory_42[l_220] = l_221;
const uint32 l_222 = l_47 + uint32(8);
const __half l_223 = l_arr_40[uint32(8)];
shared_memory_42[l_222] = l_223;
const uint32 l_224 = l_47 + uint32(9);
const __half l_225 = l_arr_40[uint32(9)];
shared_memory_42[l_224] = l_225;
const uint32 l_226 = l_47 + uint32(10);
const __half l_227 = l_arr_40[uint32(10)];
shared_memory_42[l_226] = l_227;
const uint32 l_228 = l_47 + uint32(11);
const __half l_229 = l_arr_40[uint32(11)];
shared_memory_42[l_228] = l_229;
const uint32 l_230 = l_47 + uint32(12);
const __half l_231 = l_arr_40[uint32(12)];
shared_memory_42[l_230] = l_231;
const uint32 l_232 = l_47 + uint32(13);
const __half l_233 = l_arr_40[uint32(13)];
shared_memory_42[l_232] = l_233;
const uint32 l_234 = l_47 + uint32(14);
const __half l_235 = l_arr_40[uint32(14)];
shared_memory_42[l_234] = l_235;
const uint32 l_236 = l_47 + uint32(15);
const __half l_237 = l_arr_40[uint32(15)];
shared_memory_42[l_236] = l_237;
__syncthreads();

const uint32 l_238 = l_2 * uint32(32);
const uint32 l_239 = l_238 + uint32(0);
const uint32 l_240 = l_239 * uint32(40);
const uint32 l_241 = l_240 + uint32(0);
const uint32 l_242 = uint32(2560) - l_241;
nvcuda::wmma::load_matrix_sync(frag_b_243, shared_memory_42 + l_241, uint32(40));
const uint32 l_244 = l_2 * uint32(32);
const uint32 l_245 = l_244 + uint32(16);
const uint32 l_246 = l_245 * uint32(40);
const uint32 l_247 = l_246 + uint32(0);
const uint32 l_248 = uint32(2560) - l_247;
nvcuda::wmma::load_matrix_sync(frag_b_249, shared_memory_42 + l_247, uint32(40));
const uint32 l_250 = l_1 * uint32(16);
const uint32 l_251 = l_250 + uint32(0);
const uint32 l_252 = l_251 * uint32(40);
const uint32 l_253 = l_252 + uint32(0);
const uint32 l_254 = uint32(1280) - l_253;
nvcuda::wmma::load_matrix_sync(frag_a_255, shared_memory_41 + l_253, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_255, frag_b_243, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_255, frag_b_249, frag_acc_49);
const uint32 l_256 = l_2 * uint32(32);
const uint32 l_257 = l_256 + uint32(0);
const uint32 l_258 = l_257 * uint32(40);
const uint32 l_259 = l_258 + uint32(16);
const uint32 l_260 = uint32(2560) - l_259;
nvcuda::wmma::load_matrix_sync(frag_b_261, shared_memory_42 + l_259, uint32(40));
const uint32 l_262 = l_2 * uint32(32);
const uint32 l_263 = l_262 + uint32(16);
const uint32 l_264 = l_263 * uint32(40);
const uint32 l_265 = l_264 + uint32(16);
const uint32 l_266 = uint32(2560) - l_265;
nvcuda::wmma::load_matrix_sync(frag_b_267, shared_memory_42 + l_265, uint32(40));
const uint32 l_268 = l_1 * uint32(16);
const uint32 l_269 = l_268 + uint32(0);
const uint32 l_270 = l_269 * uint32(40);
const uint32 l_271 = l_270 + uint32(16);
const uint32 l_272 = uint32(1280) - l_271;
nvcuda::wmma::load_matrix_sync(frag_a_273, shared_memory_41 + l_271, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_273, frag_b_261, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_273, frag_b_267, frag_acc_49);
}
}
const bool l_274 = l_0 == uint32(0);
if (l_274) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_275 = l_6 + uint32(0);
const uint32 l_276 = l_275 + uint32(0);
const uint32 l_277 = l_7 + uint32(0);
__syncthreads();

const uint32 l_278 = threadIdxGlobal + uint32(0);
const uint32 l_279 = l_278 / uint32(32);
const uint32 l_280 = l_276 + l_279;
const uint32 l_281 = l_278 % uint32(32);
const uint32 l_282 = l_277 + l_281;
const bool l_283 = l_280 < l_3;
const bool l_284 = l_282 < l_4;
const bool l_285 = l_283 && l_284;
if (l_285) {
const uint32 l_286 = l_280 * l_4;
const uint32 l_287 = l_286 + l_282;
const float l_288 = shared_memory_38[l_278];
buffer_2[l_287] = l_288;
}
const uint32 l_289 = threadIdxGlobal + uint32(128);
const uint32 l_290 = l_289 / uint32(32);
const uint32 l_291 = l_276 + l_290;
const uint32 l_292 = l_289 % uint32(32);
const uint32 l_293 = l_277 + l_292;
const bool l_294 = l_291 < l_3;
const bool l_295 = l_293 < l_4;
const bool l_296 = l_294 && l_295;
if (l_296) {
const uint32 l_297 = l_291 * l_4;
const uint32 l_298 = l_297 + l_293;
const float l_299 = shared_memory_38[l_289];
buffer_2[l_298] = l_299;
}
const uint32 l_300 = threadIdxGlobal + uint32(256);
const uint32 l_301 = l_300 / uint32(32);
const uint32 l_302 = l_276 + l_301;
const uint32 l_303 = l_300 % uint32(32);
const uint32 l_304 = l_277 + l_303;
const bool l_305 = l_302 < l_3;
const bool l_306 = l_304 < l_4;
const bool l_307 = l_305 && l_306;
if (l_307) {
const uint32 l_308 = l_302 * l_4;
const uint32 l_309 = l_308 + l_304;
const float l_310 = shared_memory_38[l_300];
buffer_2[l_309] = l_310;
}
const uint32 l_311 = threadIdxGlobal + uint32(384);
const uint32 l_312 = l_311 / uint32(32);
const uint32 l_313 = l_276 + l_312;
const uint32 l_314 = l_311 % uint32(32);
const uint32 l_315 = l_277 + l_314;
const bool l_316 = l_313 < l_3;
const bool l_317 = l_315 < l_4;
const bool l_318 = l_316 && l_317;
if (l_318) {
const uint32 l_319 = l_313 * l_4;
const uint32 l_320 = l_319 + l_315;
const float l_321 = shared_memory_38[l_311];
buffer_2[l_320] = l_321;
}
__syncthreads();

const bool l_322 = l_0 == uint32(1);
if (l_322) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_323 = l_6 + uint32(0);
const uint32 l_324 = l_323 + uint32(0);
const uint32 l_325 = l_7 + uint32(32);
__syncthreads();

const uint32 l_326 = threadIdxGlobal + uint32(0);
const uint32 l_327 = l_326 / uint32(32);
const uint32 l_328 = l_324 + l_327;
const uint32 l_329 = l_326 % uint32(32);
const uint32 l_330 = l_325 + l_329;
const bool l_331 = l_328 < l_3;
const bool l_332 = l_330 < l_4;
const bool l_333 = l_331 && l_332;
if (l_333) {
const uint32 l_334 = l_328 * l_4;
const uint32 l_335 = l_334 + l_330;
const float l_336 = shared_memory_38[l_326];
buffer_2[l_335] = l_336;
}
const uint32 l_337 = threadIdxGlobal + uint32(128);
const uint32 l_338 = l_337 / uint32(32);
const uint32 l_339 = l_324 + l_338;
const uint32 l_340 = l_337 % uint32(32);
const uint32 l_341 = l_325 + l_340;
const bool l_342 = l_339 < l_3;
const bool l_343 = l_341 < l_4;
const bool l_344 = l_342 && l_343;
if (l_344) {
const uint32 l_345 = l_339 * l_4;
const uint32 l_346 = l_345 + l_341;
const float l_347 = shared_memory_38[l_337];
buffer_2[l_346] = l_347;
}
const uint32 l_348 = threadIdxGlobal + uint32(256);
const uint32 l_349 = l_348 / uint32(32);
const uint32 l_350 = l_324 + l_349;
const uint32 l_351 = l_348 % uint32(32);
const uint32 l_352 = l_325 + l_351;
const bool l_353 = l_350 < l_3;
const bool l_354 = l_352 < l_4;
const bool l_355 = l_353 && l_354;
if (l_355) {
const uint32 l_356 = l_350 * l_4;
const uint32 l_357 = l_356 + l_352;
const float l_358 = shared_memory_38[l_348];
buffer_2[l_357] = l_358;
}
const uint32 l_359 = threadIdxGlobal + uint32(384);
const uint32 l_360 = l_359 / uint32(32);
const uint32 l_361 = l_324 + l_360;
const uint32 l_362 = l_359 % uint32(32);
const uint32 l_363 = l_325 + l_362;
const bool l_364 = l_361 < l_3;
const bool l_365 = l_363 < l_4;
const bool l_366 = l_364 && l_365;
if (l_366) {
const uint32 l_367 = l_361 * l_4;
const uint32 l_368 = l_367 + l_363;
const float l_369 = shared_memory_38[l_359];
buffer_2[l_368] = l_369;
}
__syncthreads();

const bool l_370 = l_0 == uint32(2);
if (l_370) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_371 = l_6 + uint32(16);
const uint32 l_372 = l_371 + uint32(0);
const uint32 l_373 = l_7 + uint32(0);
__syncthreads();

const uint32 l_374 = threadIdxGlobal + uint32(0);
const uint32 l_375 = l_374 / uint32(32);
const uint32 l_376 = l_372 + l_375;
const uint32 l_377 = l_374 % uint32(32);
const uint32 l_378 = l_373 + l_377;
const bool l_379 = l_376 < l_3;
const bool l_380 = l_378 < l_4;
const bool l_381 = l_379 && l_380;
if (l_381) {
const uint32 l_382 = l_376 * l_4;
const uint32 l_383 = l_382 + l_378;
const float l_384 = shared_memory_38[l_374];
buffer_2[l_383] = l_384;
}
const uint32 l_385 = threadIdxGlobal + uint32(128);
const uint32 l_386 = l_385 / uint32(32);
const uint32 l_387 = l_372 + l_386;
const uint32 l_388 = l_385 % uint32(32);
const uint32 l_389 = l_373 + l_388;
const bool l_390 = l_387 < l_3;
const bool l_391 = l_389 < l_4;
const bool l_392 = l_390 && l_391;
if (l_392) {
const uint32 l_393 = l_387 * l_4;
const uint32 l_394 = l_393 + l_389;
const float l_395 = shared_memory_38[l_385];
buffer_2[l_394] = l_395;
}
const uint32 l_396 = threadIdxGlobal + uint32(256);
const uint32 l_397 = l_396 / uint32(32);
const uint32 l_398 = l_372 + l_397;
const uint32 l_399 = l_396 % uint32(32);
const uint32 l_400 = l_373 + l_399;
const bool l_401 = l_398 < l_3;
const bool l_402 = l_400 < l_4;
const bool l_403 = l_401 && l_402;
if (l_403) {
const uint32 l_404 = l_398 * l_4;
const uint32 l_405 = l_404 + l_400;
const float l_406 = shared_memory_38[l_396];
buffer_2[l_405] = l_406;
}
const uint32 l_407 = threadIdxGlobal + uint32(384);
const uint32 l_408 = l_407 / uint32(32);
const uint32 l_409 = l_372 + l_408;
const uint32 l_410 = l_407 % uint32(32);
const uint32 l_411 = l_373 + l_410;
const bool l_412 = l_409 < l_3;
const bool l_413 = l_411 < l_4;
const bool l_414 = l_412 && l_413;
if (l_414) {
const uint32 l_415 = l_409 * l_4;
const uint32 l_416 = l_415 + l_411;
const float l_417 = shared_memory_38[l_407];
buffer_2[l_416] = l_417;
}
__syncthreads();

const bool l_418 = l_0 == uint32(3);
if (l_418) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_419 = l_6 + uint32(16);
const uint32 l_420 = l_419 + uint32(0);
const uint32 l_421 = l_7 + uint32(32);
__syncthreads();

const uint32 l_422 = threadIdxGlobal + uint32(0);
const uint32 l_423 = l_422 / uint32(32);
const uint32 l_424 = l_420 + l_423;
const uint32 l_425 = l_422 % uint32(32);
const uint32 l_426 = l_421 + l_425;
const bool l_427 = l_424 < l_3;
const bool l_428 = l_426 < l_4;
const bool l_429 = l_427 && l_428;
if (l_429) {
const uint32 l_430 = l_424 * l_4;
const uint32 l_431 = l_430 + l_426;
const float l_432 = shared_memory_38[l_422];
buffer_2[l_431] = l_432;
}
const uint32 l_433 = threadIdxGlobal + uint32(128);
const uint32 l_434 = l_433 / uint32(32);
const uint32 l_435 = l_420 + l_434;
const uint32 l_436 = l_433 % uint32(32);
const uint32 l_437 = l_421 + l_436;
const bool l_438 = l_435 < l_3;
const bool l_439 = l_437 < l_4;
const bool l_440 = l_438 && l_439;
if (l_440) {
const uint32 l_441 = l_435 * l_4;
const uint32 l_442 = l_441 + l_437;
const float l_443 = shared_memory_38[l_433];
buffer_2[l_442] = l_443;
}
const uint32 l_444 = threadIdxGlobal + uint32(256);
const uint32 l_445 = l_444 / uint32(32);
const uint32 l_446 = l_420 + l_445;
const uint32 l_447 = l_444 % uint32(32);
const uint32 l_448 = l_421 + l_447;
const bool l_449 = l_446 < l_3;
const bool l_450 = l_448 < l_4;
const bool l_451 = l_449 && l_450;
if (l_451) {
const uint32 l_452 = l_446 * l_4;
const uint32 l_453 = l_452 + l_448;
const float l_454 = shared_memory_38[l_444];
buffer_2[l_453] = l_454;
}
const uint32 l_455 = threadIdxGlobal + uint32(384);
const uint32 l_456 = l_455 / uint32(32);
const uint32 l_457 = l_420 + l_456;
const uint32 l_458 = l_455 % uint32(32);
const uint32 l_459 = l_421 + l_458;
const bool l_460 = l_457 < l_3;
const bool l_461 = l_459 < l_4;
const bool l_462 = l_460 && l_461;
if (l_462) {
const uint32 l_463 = l_457 * l_4;
const uint32 l_464 = l_463 + l_459;
const float l_465 = shared_memory_38[l_455];
buffer_2[l_464] = l_465;
}
__syncthreads();


}
