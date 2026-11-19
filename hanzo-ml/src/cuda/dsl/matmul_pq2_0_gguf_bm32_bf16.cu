#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

struct __align__(4) __half_2 {
    __half i_0;
    __half i_1;
};

struct __align__(8) __nv_bfloat162_2 {
    __nv_bfloat162 i_0;
    __nv_bfloat162 i_1;
};

struct __align__(8) __nv_bfloat16_4 {
    __nv_bfloat16 i_0;
    __nv_bfloat16 i_1;
    __nv_bfloat16 i_2;
    __nv_bfloat16 i_3;
};

struct info_st {
    uint32 static_meta[8];
};


extern "C" __global__ void __launch_bounds__(128)matmul_pq2_0_gguf_f_ (
    const uint32* __restrict__ buffer_0, const __nv_bfloat16_4* __restrict__ buffer_1, __nv_bfloat16* buffer_2, const uint32* __restrict__ buffer_3, const __grid_constant__ info_st info
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
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_251;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_257;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_263;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_269;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_275;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_281;
const uint32 l_75 = l_mut_50 * uint32(4);
const uint32 l_76 = l_75 + l_mut_74;
const uint32 l_77 = l_76 * uint32(8);
const uint32 l_78 = uint32(2) * l_mut_74;
const uint32 l_79 = l_78 + l_32;
const uint32 l_80 = l_27 + l_77;
const __nv_bfloat16_4 l_81 = buffer_1[l_80];
const uint32 l_82 = l_80 + uint32(1);
const __nv_bfloat16_4 l_83 = buffer_1[l_82];
const __nv_bfloat16 l_84 = l_81.i_0;
const float l_85 = float(l_84);
const __half l_86 = __half(l_85);
l_arr_39[uint32(0)] = l_86;
const __nv_bfloat16 l_87 = l_83.i_0;
const float l_88 = float(l_87);
const __half l_89 = __half(l_88);
l_arr_39[uint32(4)] = l_89;
const __nv_bfloat16 l_90 = l_81.i_1;
const float l_91 = float(l_90);
const __half l_92 = __half(l_91);
l_arr_39[uint32(1)] = l_92;
const __nv_bfloat16 l_93 = l_83.i_1;
const float l_94 = float(l_93);
const __half l_95 = __half(l_94);
l_arr_39[uint32(5)] = l_95;
const __nv_bfloat16 l_96 = l_81.i_2;
const float l_97 = float(l_96);
const __half l_98 = __half(l_97);
l_arr_39[uint32(2)] = l_98;
const __nv_bfloat16 l_99 = l_83.i_2;
const float l_100 = float(l_99);
const __half l_101 = __half(l_100);
l_arr_39[uint32(6)] = l_101;
const __nv_bfloat16 l_102 = l_81.i_3;
const float l_103 = float(l_102);
const __half l_104 = __half(l_103);
l_arr_39[uint32(3)] = l_104;
const __nv_bfloat16 l_105 = l_83.i_3;
const float l_106 = float(l_105);
const __half l_107 = __half(l_106);
l_arr_39[uint32(7)] = l_107;
const uint32 l_108 = l_mut_23 + l_79;
const uint32 l_109 = buffer_0[l_108];
const uint32 l_110 = l_mut_23 + l_79;
const uint32 l_111 = l_110 + l_mut_53;
const uint32 l_112 = buffer_0[l_111];
const uint32 l_113 = l_109 >> l_mut_16;
const uint32 l_114 = l_112 << uint32(15);
const uint32 l_115 = uint32(17) - l_mut_16;
const uint32 l_116 = l_114 << l_115;
const uint32 l_117 = l_113 | l_116;
const uint32 l_118 = l_117 >> uint32(0);
const uint32 l_119 = l_118 & uint32(3);
const __half l_120 = __half(l_119);
const __half l_121 = l_120 * l_mut_52;
const __half l_122 = l_121 - l_mut_52;
l_arr_40[uint32(0)] = l_122;
const uint32 l_123 = l_117 >> uint32(2);
const uint32 l_124 = l_123 & uint32(3);
const __half l_125 = __half(l_124);
const __half l_126 = l_125 * l_mut_52;
const __half l_127 = l_126 - l_mut_52;
l_arr_40[uint32(1)] = l_127;
const uint32 l_128 = l_117 >> uint32(4);
const uint32 l_129 = l_128 & uint32(3);
const __half l_130 = __half(l_129);
const __half l_131 = l_130 * l_mut_52;
const __half l_132 = l_131 - l_mut_52;
l_arr_40[uint32(2)] = l_132;
const uint32 l_133 = l_117 >> uint32(6);
const uint32 l_134 = l_133 & uint32(3);
const __half l_135 = __half(l_134);
const __half l_136 = l_135 * l_mut_52;
const __half l_137 = l_136 - l_mut_52;
l_arr_40[uint32(3)] = l_137;
const uint32 l_138 = l_117 >> uint32(8);
const uint32 l_139 = l_138 & uint32(3);
const __half l_140 = __half(l_139);
const __half l_141 = l_140 * l_mut_52;
const __half l_142 = l_141 - l_mut_52;
l_arr_40[uint32(4)] = l_142;
const uint32 l_143 = l_117 >> uint32(10);
const uint32 l_144 = l_143 & uint32(3);
const __half l_145 = __half(l_144);
const __half l_146 = l_145 * l_mut_52;
const __half l_147 = l_146 - l_mut_52;
l_arr_40[uint32(5)] = l_147;
const uint32 l_148 = l_117 >> uint32(12);
const uint32 l_149 = l_148 & uint32(3);
const __half l_150 = __half(l_149);
const __half l_151 = l_150 * l_mut_52;
const __half l_152 = l_151 - l_mut_52;
l_arr_40[uint32(6)] = l_152;
const uint32 l_153 = l_117 >> uint32(14);
const uint32 l_154 = l_153 & uint32(3);
const __half l_155 = __half(l_154);
const __half l_156 = l_155 * l_mut_52;
const __half l_157 = l_156 - l_mut_52;
l_arr_40[uint32(7)] = l_157;
const uint32 l_158 = l_117 >> uint32(16);
const uint32 l_159 = l_158 & uint32(3);
const __half l_160 = __half(l_159);
const __half l_161 = l_160 * l_mut_52;
const __half l_162 = l_161 - l_mut_52;
l_arr_40[uint32(8)] = l_162;
const uint32 l_163 = l_117 >> uint32(18);
const uint32 l_164 = l_163 & uint32(3);
const __half l_165 = __half(l_164);
const __half l_166 = l_165 * l_mut_52;
const __half l_167 = l_166 - l_mut_52;
l_arr_40[uint32(9)] = l_167;
const uint32 l_168 = l_117 >> uint32(20);
const uint32 l_169 = l_168 & uint32(3);
const __half l_170 = __half(l_169);
const __half l_171 = l_170 * l_mut_52;
const __half l_172 = l_171 - l_mut_52;
l_arr_40[uint32(10)] = l_172;
const uint32 l_173 = l_117 >> uint32(22);
const uint32 l_174 = l_173 & uint32(3);
const __half l_175 = __half(l_174);
const __half l_176 = l_175 * l_mut_52;
const __half l_177 = l_176 - l_mut_52;
l_arr_40[uint32(11)] = l_177;
const uint32 l_178 = l_117 >> uint32(24);
const uint32 l_179 = l_178 & uint32(3);
const __half l_180 = __half(l_179);
const __half l_181 = l_180 * l_mut_52;
const __half l_182 = l_181 - l_mut_52;
l_arr_40[uint32(12)] = l_182;
const uint32 l_183 = l_117 >> uint32(26);
const uint32 l_184 = l_183 & uint32(3);
const __half l_185 = __half(l_184);
const __half l_186 = l_185 * l_mut_52;
const __half l_187 = l_186 - l_mut_52;
l_arr_40[uint32(13)] = l_187;
const uint32 l_188 = l_117 >> uint32(28);
const uint32 l_189 = l_188 & uint32(3);
const __half l_190 = __half(l_189);
const __half l_191 = l_190 * l_mut_52;
const __half l_192 = l_191 - l_mut_52;
l_arr_40[uint32(14)] = l_192;
const uint32 l_193 = l_117 >> uint32(30);
const uint32 l_194 = l_193 & uint32(3);
const __half l_195 = __half(l_194);
const __half l_196 = l_195 * l_mut_52;
const __half l_197 = l_196 - l_mut_52;
l_arr_40[uint32(15)] = l_197;
__syncthreads();

const uint32 l_198 = l_44 + uint32(0);
const __half l_199 = l_arr_39[uint32(0)];
shared_memory_41[l_198] = l_199;
const uint32 l_200 = l_44 + uint32(1);
const __half l_201 = l_arr_39[uint32(1)];
shared_memory_41[l_200] = l_201;
const uint32 l_202 = l_44 + uint32(2);
const __half l_203 = l_arr_39[uint32(2)];
shared_memory_41[l_202] = l_203;
const uint32 l_204 = l_44 + uint32(3);
const __half l_205 = l_arr_39[uint32(3)];
shared_memory_41[l_204] = l_205;
const uint32 l_206 = l_44 + uint32(4);
const __half l_207 = l_arr_39[uint32(4)];
shared_memory_41[l_206] = l_207;
const uint32 l_208 = l_44 + uint32(5);
const __half l_209 = l_arr_39[uint32(5)];
shared_memory_41[l_208] = l_209;
const uint32 l_210 = l_44 + uint32(6);
const __half l_211 = l_arr_39[uint32(6)];
shared_memory_41[l_210] = l_211;
const uint32 l_212 = l_44 + uint32(7);
const __half l_213 = l_arr_39[uint32(7)];
shared_memory_41[l_212] = l_213;
const uint32 l_214 = l_47 + uint32(0);
const __half l_215 = l_arr_40[uint32(0)];
shared_memory_42[l_214] = l_215;
const uint32 l_216 = l_47 + uint32(1);
const __half l_217 = l_arr_40[uint32(1)];
shared_memory_42[l_216] = l_217;
const uint32 l_218 = l_47 + uint32(2);
const __half l_219 = l_arr_40[uint32(2)];
shared_memory_42[l_218] = l_219;
const uint32 l_220 = l_47 + uint32(3);
const __half l_221 = l_arr_40[uint32(3)];
shared_memory_42[l_220] = l_221;
const uint32 l_222 = l_47 + uint32(4);
const __half l_223 = l_arr_40[uint32(4)];
shared_memory_42[l_222] = l_223;
const uint32 l_224 = l_47 + uint32(5);
const __half l_225 = l_arr_40[uint32(5)];
shared_memory_42[l_224] = l_225;
const uint32 l_226 = l_47 + uint32(6);
const __half l_227 = l_arr_40[uint32(6)];
shared_memory_42[l_226] = l_227;
const uint32 l_228 = l_47 + uint32(7);
const __half l_229 = l_arr_40[uint32(7)];
shared_memory_42[l_228] = l_229;
const uint32 l_230 = l_47 + uint32(8);
const __half l_231 = l_arr_40[uint32(8)];
shared_memory_42[l_230] = l_231;
const uint32 l_232 = l_47 + uint32(9);
const __half l_233 = l_arr_40[uint32(9)];
shared_memory_42[l_232] = l_233;
const uint32 l_234 = l_47 + uint32(10);
const __half l_235 = l_arr_40[uint32(10)];
shared_memory_42[l_234] = l_235;
const uint32 l_236 = l_47 + uint32(11);
const __half l_237 = l_arr_40[uint32(11)];
shared_memory_42[l_236] = l_237;
const uint32 l_238 = l_47 + uint32(12);
const __half l_239 = l_arr_40[uint32(12)];
shared_memory_42[l_238] = l_239;
const uint32 l_240 = l_47 + uint32(13);
const __half l_241 = l_arr_40[uint32(13)];
shared_memory_42[l_240] = l_241;
const uint32 l_242 = l_47 + uint32(14);
const __half l_243 = l_arr_40[uint32(14)];
shared_memory_42[l_242] = l_243;
const uint32 l_244 = l_47 + uint32(15);
const __half l_245 = l_arr_40[uint32(15)];
shared_memory_42[l_244] = l_245;
__syncthreads();

const uint32 l_246 = l_2 * uint32(32);
const uint32 l_247 = l_246 + uint32(0);
const uint32 l_248 = l_247 * uint32(40);
const uint32 l_249 = l_248 + uint32(0);
const uint32 l_250 = uint32(2560) - l_249;
nvcuda::wmma::load_matrix_sync(frag_b_251, shared_memory_42 + l_249, uint32(40));
const uint32 l_252 = l_2 * uint32(32);
const uint32 l_253 = l_252 + uint32(16);
const uint32 l_254 = l_253 * uint32(40);
const uint32 l_255 = l_254 + uint32(0);
const uint32 l_256 = uint32(2560) - l_255;
nvcuda::wmma::load_matrix_sync(frag_b_257, shared_memory_42 + l_255, uint32(40));
const uint32 l_258 = l_1 * uint32(16);
const uint32 l_259 = l_258 + uint32(0);
const uint32 l_260 = l_259 * uint32(40);
const uint32 l_261 = l_260 + uint32(0);
const uint32 l_262 = uint32(1280) - l_261;
nvcuda::wmma::load_matrix_sync(frag_a_263, shared_memory_41 + l_261, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_263, frag_b_251, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_263, frag_b_257, frag_acc_49);
const uint32 l_264 = l_2 * uint32(32);
const uint32 l_265 = l_264 + uint32(0);
const uint32 l_266 = l_265 * uint32(40);
const uint32 l_267 = l_266 + uint32(16);
const uint32 l_268 = uint32(2560) - l_267;
nvcuda::wmma::load_matrix_sync(frag_b_269, shared_memory_42 + l_267, uint32(40));
const uint32 l_270 = l_2 * uint32(32);
const uint32 l_271 = l_270 + uint32(16);
const uint32 l_272 = l_271 * uint32(40);
const uint32 l_273 = l_272 + uint32(16);
const uint32 l_274 = uint32(2560) - l_273;
nvcuda::wmma::load_matrix_sync(frag_b_275, shared_memory_42 + l_273, uint32(40));
const uint32 l_276 = l_1 * uint32(16);
const uint32 l_277 = l_276 + uint32(0);
const uint32 l_278 = l_277 * uint32(40);
const uint32 l_279 = l_278 + uint32(16);
const uint32 l_280 = uint32(1280) - l_279;
nvcuda::wmma::load_matrix_sync(frag_a_281, shared_memory_41 + l_279, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_281, frag_b_269, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_281, frag_b_275, frag_acc_49);
}
}
const bool l_282 = l_0 == uint32(0);
if (l_282) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_283 = l_6 + uint32(0);
const uint32 l_284 = l_283 + uint32(0);
const uint32 l_285 = l_7 + uint32(0);
__syncthreads();

const uint32 l_286 = threadIdxGlobal + uint32(0);
const uint32 l_287 = l_286 / uint32(32);
const uint32 l_288 = l_284 + l_287;
const uint32 l_289 = l_286 % uint32(32);
const uint32 l_290 = l_285 + l_289;
const bool l_291 = l_288 < l_3;
const bool l_292 = l_290 < l_4;
const bool l_293 = l_291 && l_292;
if (l_293) {
const uint32 l_294 = l_288 * l_4;
const uint32 l_295 = l_294 + l_290;
const float l_296 = shared_memory_38[l_286];
const __nv_bfloat16 l_297 = __nv_bfloat16(l_296);
buffer_2[l_295] = l_297;
}
const uint32 l_298 = threadIdxGlobal + uint32(128);
const uint32 l_299 = l_298 / uint32(32);
const uint32 l_300 = l_284 + l_299;
const uint32 l_301 = l_298 % uint32(32);
const uint32 l_302 = l_285 + l_301;
const bool l_303 = l_300 < l_3;
const bool l_304 = l_302 < l_4;
const bool l_305 = l_303 && l_304;
if (l_305) {
const uint32 l_306 = l_300 * l_4;
const uint32 l_307 = l_306 + l_302;
const float l_308 = shared_memory_38[l_298];
const __nv_bfloat16 l_309 = __nv_bfloat16(l_308);
buffer_2[l_307] = l_309;
}
const uint32 l_310 = threadIdxGlobal + uint32(256);
const uint32 l_311 = l_310 / uint32(32);
const uint32 l_312 = l_284 + l_311;
const uint32 l_313 = l_310 % uint32(32);
const uint32 l_314 = l_285 + l_313;
const bool l_315 = l_312 < l_3;
const bool l_316 = l_314 < l_4;
const bool l_317 = l_315 && l_316;
if (l_317) {
const uint32 l_318 = l_312 * l_4;
const uint32 l_319 = l_318 + l_314;
const float l_320 = shared_memory_38[l_310];
const __nv_bfloat16 l_321 = __nv_bfloat16(l_320);
buffer_2[l_319] = l_321;
}
const uint32 l_322 = threadIdxGlobal + uint32(384);
const uint32 l_323 = l_322 / uint32(32);
const uint32 l_324 = l_284 + l_323;
const uint32 l_325 = l_322 % uint32(32);
const uint32 l_326 = l_285 + l_325;
const bool l_327 = l_324 < l_3;
const bool l_328 = l_326 < l_4;
const bool l_329 = l_327 && l_328;
if (l_329) {
const uint32 l_330 = l_324 * l_4;
const uint32 l_331 = l_330 + l_326;
const float l_332 = shared_memory_38[l_322];
const __nv_bfloat16 l_333 = __nv_bfloat16(l_332);
buffer_2[l_331] = l_333;
}
__syncthreads();

const bool l_334 = l_0 == uint32(1);
if (l_334) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_335 = l_6 + uint32(0);
const uint32 l_336 = l_335 + uint32(0);
const uint32 l_337 = l_7 + uint32(32);
__syncthreads();

const uint32 l_338 = threadIdxGlobal + uint32(0);
const uint32 l_339 = l_338 / uint32(32);
const uint32 l_340 = l_336 + l_339;
const uint32 l_341 = l_338 % uint32(32);
const uint32 l_342 = l_337 + l_341;
const bool l_343 = l_340 < l_3;
const bool l_344 = l_342 < l_4;
const bool l_345 = l_343 && l_344;
if (l_345) {
const uint32 l_346 = l_340 * l_4;
const uint32 l_347 = l_346 + l_342;
const float l_348 = shared_memory_38[l_338];
const __nv_bfloat16 l_349 = __nv_bfloat16(l_348);
buffer_2[l_347] = l_349;
}
const uint32 l_350 = threadIdxGlobal + uint32(128);
const uint32 l_351 = l_350 / uint32(32);
const uint32 l_352 = l_336 + l_351;
const uint32 l_353 = l_350 % uint32(32);
const uint32 l_354 = l_337 + l_353;
const bool l_355 = l_352 < l_3;
const bool l_356 = l_354 < l_4;
const bool l_357 = l_355 && l_356;
if (l_357) {
const uint32 l_358 = l_352 * l_4;
const uint32 l_359 = l_358 + l_354;
const float l_360 = shared_memory_38[l_350];
const __nv_bfloat16 l_361 = __nv_bfloat16(l_360);
buffer_2[l_359] = l_361;
}
const uint32 l_362 = threadIdxGlobal + uint32(256);
const uint32 l_363 = l_362 / uint32(32);
const uint32 l_364 = l_336 + l_363;
const uint32 l_365 = l_362 % uint32(32);
const uint32 l_366 = l_337 + l_365;
const bool l_367 = l_364 < l_3;
const bool l_368 = l_366 < l_4;
const bool l_369 = l_367 && l_368;
if (l_369) {
const uint32 l_370 = l_364 * l_4;
const uint32 l_371 = l_370 + l_366;
const float l_372 = shared_memory_38[l_362];
const __nv_bfloat16 l_373 = __nv_bfloat16(l_372);
buffer_2[l_371] = l_373;
}
const uint32 l_374 = threadIdxGlobal + uint32(384);
const uint32 l_375 = l_374 / uint32(32);
const uint32 l_376 = l_336 + l_375;
const uint32 l_377 = l_374 % uint32(32);
const uint32 l_378 = l_337 + l_377;
const bool l_379 = l_376 < l_3;
const bool l_380 = l_378 < l_4;
const bool l_381 = l_379 && l_380;
if (l_381) {
const uint32 l_382 = l_376 * l_4;
const uint32 l_383 = l_382 + l_378;
const float l_384 = shared_memory_38[l_374];
const __nv_bfloat16 l_385 = __nv_bfloat16(l_384);
buffer_2[l_383] = l_385;
}
__syncthreads();

const bool l_386 = l_0 == uint32(2);
if (l_386) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_387 = l_6 + uint32(16);
const uint32 l_388 = l_387 + uint32(0);
const uint32 l_389 = l_7 + uint32(0);
__syncthreads();

const uint32 l_390 = threadIdxGlobal + uint32(0);
const uint32 l_391 = l_390 / uint32(32);
const uint32 l_392 = l_388 + l_391;
const uint32 l_393 = l_390 % uint32(32);
const uint32 l_394 = l_389 + l_393;
const bool l_395 = l_392 < l_3;
const bool l_396 = l_394 < l_4;
const bool l_397 = l_395 && l_396;
if (l_397) {
const uint32 l_398 = l_392 * l_4;
const uint32 l_399 = l_398 + l_394;
const float l_400 = shared_memory_38[l_390];
const __nv_bfloat16 l_401 = __nv_bfloat16(l_400);
buffer_2[l_399] = l_401;
}
const uint32 l_402 = threadIdxGlobal + uint32(128);
const uint32 l_403 = l_402 / uint32(32);
const uint32 l_404 = l_388 + l_403;
const uint32 l_405 = l_402 % uint32(32);
const uint32 l_406 = l_389 + l_405;
const bool l_407 = l_404 < l_3;
const bool l_408 = l_406 < l_4;
const bool l_409 = l_407 && l_408;
if (l_409) {
const uint32 l_410 = l_404 * l_4;
const uint32 l_411 = l_410 + l_406;
const float l_412 = shared_memory_38[l_402];
const __nv_bfloat16 l_413 = __nv_bfloat16(l_412);
buffer_2[l_411] = l_413;
}
const uint32 l_414 = threadIdxGlobal + uint32(256);
const uint32 l_415 = l_414 / uint32(32);
const uint32 l_416 = l_388 + l_415;
const uint32 l_417 = l_414 % uint32(32);
const uint32 l_418 = l_389 + l_417;
const bool l_419 = l_416 < l_3;
const bool l_420 = l_418 < l_4;
const bool l_421 = l_419 && l_420;
if (l_421) {
const uint32 l_422 = l_416 * l_4;
const uint32 l_423 = l_422 + l_418;
const float l_424 = shared_memory_38[l_414];
const __nv_bfloat16 l_425 = __nv_bfloat16(l_424);
buffer_2[l_423] = l_425;
}
const uint32 l_426 = threadIdxGlobal + uint32(384);
const uint32 l_427 = l_426 / uint32(32);
const uint32 l_428 = l_388 + l_427;
const uint32 l_429 = l_426 % uint32(32);
const uint32 l_430 = l_389 + l_429;
const bool l_431 = l_428 < l_3;
const bool l_432 = l_430 < l_4;
const bool l_433 = l_431 && l_432;
if (l_433) {
const uint32 l_434 = l_428 * l_4;
const uint32 l_435 = l_434 + l_430;
const float l_436 = shared_memory_38[l_426];
const __nv_bfloat16 l_437 = __nv_bfloat16(l_436);
buffer_2[l_435] = l_437;
}
__syncthreads();

const bool l_438 = l_0 == uint32(3);
if (l_438) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_439 = l_6 + uint32(16);
const uint32 l_440 = l_439 + uint32(0);
const uint32 l_441 = l_7 + uint32(32);
__syncthreads();

const uint32 l_442 = threadIdxGlobal + uint32(0);
const uint32 l_443 = l_442 / uint32(32);
const uint32 l_444 = l_440 + l_443;
const uint32 l_445 = l_442 % uint32(32);
const uint32 l_446 = l_441 + l_445;
const bool l_447 = l_444 < l_3;
const bool l_448 = l_446 < l_4;
const bool l_449 = l_447 && l_448;
if (l_449) {
const uint32 l_450 = l_444 * l_4;
const uint32 l_451 = l_450 + l_446;
const float l_452 = shared_memory_38[l_442];
const __nv_bfloat16 l_453 = __nv_bfloat16(l_452);
buffer_2[l_451] = l_453;
}
const uint32 l_454 = threadIdxGlobal + uint32(128);
const uint32 l_455 = l_454 / uint32(32);
const uint32 l_456 = l_440 + l_455;
const uint32 l_457 = l_454 % uint32(32);
const uint32 l_458 = l_441 + l_457;
const bool l_459 = l_456 < l_3;
const bool l_460 = l_458 < l_4;
const bool l_461 = l_459 && l_460;
if (l_461) {
const uint32 l_462 = l_456 * l_4;
const uint32 l_463 = l_462 + l_458;
const float l_464 = shared_memory_38[l_454];
const __nv_bfloat16 l_465 = __nv_bfloat16(l_464);
buffer_2[l_463] = l_465;
}
const uint32 l_466 = threadIdxGlobal + uint32(256);
const uint32 l_467 = l_466 / uint32(32);
const uint32 l_468 = l_440 + l_467;
const uint32 l_469 = l_466 % uint32(32);
const uint32 l_470 = l_441 + l_469;
const bool l_471 = l_468 < l_3;
const bool l_472 = l_470 < l_4;
const bool l_473 = l_471 && l_472;
if (l_473) {
const uint32 l_474 = l_468 * l_4;
const uint32 l_475 = l_474 + l_470;
const float l_476 = shared_memory_38[l_466];
const __nv_bfloat16 l_477 = __nv_bfloat16(l_476);
buffer_2[l_475] = l_477;
}
const uint32 l_478 = threadIdxGlobal + uint32(384);
const uint32 l_479 = l_478 / uint32(32);
const uint32 l_480 = l_440 + l_479;
const uint32 l_481 = l_478 % uint32(32);
const uint32 l_482 = l_441 + l_481;
const bool l_483 = l_480 < l_3;
const bool l_484 = l_482 < l_4;
const bool l_485 = l_483 && l_484;
if (l_485) {
const uint32 l_486 = l_480 * l_4;
const uint32 l_487 = l_486 + l_482;
const float l_488 = shared_memory_38[l_478];
const __nv_bfloat16 l_489 = __nv_bfloat16(l_488);
buffer_2[l_487] = l_489;
}
__syncthreads();


}
