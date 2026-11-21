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


extern "C" __global__ void __launch_bounds__(128)matmul_ptq1_0_gguf_f_f32 (
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
uint32 l_mut_53;
uint32 l_mut_54;
uint32 l_mut_55;
uint32 l_mut_56;
uint32 l_mut_57;
__half l_mut_59;
uint32 l_mut_128;
uint32 l_mut_278;
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
l_mut_23 = uint32(0);
l_mut_16 = uint32(0);
l_mut_53 = uint32(0);
l_mut_54 = uint32(0);
l_mut_55 = uint32(0);
l_mut_56 = uint32(0);
l_mut_57 = uint32(0);
const __half l_58 = __half(float(0.0));
l_mut_59 = l_58;
const bool l_60 = threadIdxGlobal < uint32(128);
const bool l_61 = bool(true) || l_60;
if (l_61) {
const uint32 l_62 = l_37 + l_mut_52;
const uint32 l_63 = uint32(7) * l_62;
const uint32 l_64 = buffer_0[l_63];
l_mut_23 = l_64;
const uint32 l_65 = l_63 + uint32(1);
const uint32 l_66 = buffer_0[l_65];
l_mut_16 = l_66;
const uint32 l_67 = l_63 + uint32(2);
const uint32 l_68 = buffer_0[l_67];
l_mut_53 = l_68;
const uint32 l_69 = l_63 + uint32(3);
const uint32 l_70 = buffer_0[l_69];
l_mut_54 = l_70;
const uint32 l_71 = l_63 + uint32(4);
const uint32 l_72 = buffer_0[l_71];
l_mut_55 = l_72;
const uint32 l_73 = l_63 + uint32(5);
const uint32 l_74 = buffer_0[l_73];
l_mut_56 = l_74;
const uint32 l_75 = l_63 + uint32(6);
const uint32 l_76 = buffer_0[l_75];
l_mut_57 = l_76;
const uint32 l_77 = l_mut_57 >> uint32(16);
const uint32 l_78 = l_77 & uint32(65535);
const __half_2 l_79 = reinterpret_cast<__half_2 const&>(l_78);
const __half l_80 = l_79.i_0;
l_mut_59 = l_80;
}

for (uint32 l_mut_81 = uint32(0); l_mut_81 < uint32(4); ++l_mut_81) {
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_609;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_615;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_621;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_627;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_633;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_639;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_645;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_651;
const uint32 l_82 = l_mut_52 * uint32(4);
const uint32 l_83 = l_82 + l_mut_81;
const uint32 l_84 = l_83 * uint32(8);
const uint32 l_85 = uint32(2) * l_mut_81;
const uint32 l_86 = l_85 + l_32;
const uint32 l_87 = l_27 + l_84;
const float_4 l_88 = buffer_1[l_87];
const uint32 l_89 = l_87 + uint32(1);
const float_4 l_90 = buffer_1[l_89];
const float l_91 = l_88.i_0;
const __half l_92 = __half(l_91);
l_arr_39[uint32(0)] = l_92;
const float l_93 = l_90.i_0;
const __half l_94 = __half(l_93);
l_arr_39[uint32(4)] = l_94;
const float l_95 = l_88.i_1;
const __half l_96 = __half(l_95);
l_arr_39[uint32(1)] = l_96;
const float l_97 = l_90.i_1;
const __half l_98 = __half(l_97);
l_arr_39[uint32(5)] = l_98;
const float l_99 = l_88.i_2;
const __half l_100 = __half(l_99);
l_arr_39[uint32(2)] = l_100;
const float l_101 = l_90.i_2;
const __half l_102 = __half(l_101);
l_arr_39[uint32(6)] = l_102;
const float l_103 = l_88.i_3;
const __half l_104 = __half(l_103);
l_arr_39[uint32(3)] = l_104;
const float l_105 = l_90.i_3;
const __half l_106 = __half(l_105);
l_arr_39[uint32(7)] = l_106;
const uint32 l_107 = l_30 + l_84;
const float_4 l_108 = buffer_1[l_107];
const uint32 l_109 = l_107 + uint32(1);
const float_4 l_110 = buffer_1[l_109];
const float l_111 = l_108.i_0;
const __half l_112 = __half(l_111);
l_arr_39[uint32(8)] = l_112;
const float l_113 = l_110.i_0;
const __half l_114 = __half(l_113);
l_arr_39[uint32(12)] = l_114;
const float l_115 = l_108.i_1;
const __half l_116 = __half(l_115);
l_arr_39[uint32(9)] = l_116;
const float l_117 = l_110.i_1;
const __half l_118 = __half(l_117);
l_arr_39[uint32(13)] = l_118;
const float l_119 = l_108.i_2;
const __half l_120 = __half(l_119);
l_arr_39[uint32(10)] = l_120;
const float l_121 = l_110.i_2;
const __half l_122 = __half(l_121);
l_arr_39[uint32(14)] = l_122;
const float l_123 = l_108.i_3;
const __half l_124 = __half(l_123);
l_arr_39[uint32(11)] = l_124;
const float l_125 = l_110.i_3;
const __half l_126 = __half(l_125);
l_arr_39[uint32(15)] = l_126;
const bool l_127 = l_86 < uint32(5);
if (l_127) {
l_mut_128 = uint32(1);
const bool l_129 = l_86 == uint32(1);
if (l_129) {
l_mut_128 = uint32(3);
} else {
const bool l_130 = l_86 == uint32(2);
if (l_130) {
l_mut_128 = uint32(9);
} else {
const bool l_131 = l_86 == uint32(3);
if (l_131) {
l_mut_128 = uint32(27);
} else {
const bool l_132 = l_86 == uint32(4);
if (l_132) {
l_mut_128 = uint32(81);
}
}
}
}
const uint32 l_133 = l_mut_23 >> uint32(0);
const uint32 l_134 = l_133 & uint32(255);
const uint32 l_135 = l_134 * l_mut_128;
const uint32 l_136 = l_135 & uint32(255);
const uint32 l_137 = l_136 * uint32(3);
const uint32 l_138 = l_137 >> uint32(8);
const __half l_139 = __half(l_138);
const __half l_140 = l_139 * l_mut_59;
const __half l_141 = l_140 - l_mut_59;
l_arr_40[uint32(0)] = l_141;
const uint32 l_142 = l_mut_16 >> uint32(0);
const uint32 l_143 = l_142 & uint32(255);
const uint32 l_144 = l_143 * l_mut_128;
const uint32 l_145 = l_144 & uint32(255);
const uint32 l_146 = l_145 * uint32(3);
const uint32 l_147 = l_146 >> uint32(8);
const __half l_148 = __half(l_147);
const __half l_149 = l_148 * l_mut_59;
const __half l_150 = l_149 - l_mut_59;
l_arr_40[uint32(4)] = l_150;
const uint32 l_151 = l_mut_53 >> uint32(0);
const uint32 l_152 = l_151 & uint32(255);
const uint32 l_153 = l_152 * l_mut_128;
const uint32 l_154 = l_153 & uint32(255);
const uint32 l_155 = l_154 * uint32(3);
const uint32 l_156 = l_155 >> uint32(8);
const __half l_157 = __half(l_156);
const __half l_158 = l_157 * l_mut_59;
const __half l_159 = l_158 - l_mut_59;
l_arr_40[uint32(8)] = l_159;
const uint32 l_160 = l_mut_54 >> uint32(0);
const uint32 l_161 = l_160 & uint32(255);
const uint32 l_162 = l_161 * l_mut_128;
const uint32 l_163 = l_162 & uint32(255);
const uint32 l_164 = l_163 * uint32(3);
const uint32 l_165 = l_164 >> uint32(8);
const __half l_166 = __half(l_165);
const __half l_167 = l_166 * l_mut_59;
const __half l_168 = l_167 - l_mut_59;
l_arr_40[uint32(12)] = l_168;
const uint32 l_169 = l_mut_23 >> uint32(8);
const uint32 l_170 = l_169 & uint32(255);
const uint32 l_171 = l_170 * l_mut_128;
const uint32 l_172 = l_171 & uint32(255);
const uint32 l_173 = l_172 * uint32(3);
const uint32 l_174 = l_173 >> uint32(8);
const __half l_175 = __half(l_174);
const __half l_176 = l_175 * l_mut_59;
const __half l_177 = l_176 - l_mut_59;
l_arr_40[uint32(1)] = l_177;
const uint32 l_178 = l_mut_16 >> uint32(8);
const uint32 l_179 = l_178 & uint32(255);
const uint32 l_180 = l_179 * l_mut_128;
const uint32 l_181 = l_180 & uint32(255);
const uint32 l_182 = l_181 * uint32(3);
const uint32 l_183 = l_182 >> uint32(8);
const __half l_184 = __half(l_183);
const __half l_185 = l_184 * l_mut_59;
const __half l_186 = l_185 - l_mut_59;
l_arr_40[uint32(5)] = l_186;
const uint32 l_187 = l_mut_53 >> uint32(8);
const uint32 l_188 = l_187 & uint32(255);
const uint32 l_189 = l_188 * l_mut_128;
const uint32 l_190 = l_189 & uint32(255);
const uint32 l_191 = l_190 * uint32(3);
const uint32 l_192 = l_191 >> uint32(8);
const __half l_193 = __half(l_192);
const __half l_194 = l_193 * l_mut_59;
const __half l_195 = l_194 - l_mut_59;
l_arr_40[uint32(9)] = l_195;
const uint32 l_196 = l_mut_54 >> uint32(8);
const uint32 l_197 = l_196 & uint32(255);
const uint32 l_198 = l_197 * l_mut_128;
const uint32 l_199 = l_198 & uint32(255);
const uint32 l_200 = l_199 * uint32(3);
const uint32 l_201 = l_200 >> uint32(8);
const __half l_202 = __half(l_201);
const __half l_203 = l_202 * l_mut_59;
const __half l_204 = l_203 - l_mut_59;
l_arr_40[uint32(13)] = l_204;
const uint32 l_205 = l_mut_23 >> uint32(16);
const uint32 l_206 = l_205 & uint32(255);
const uint32 l_207 = l_206 * l_mut_128;
const uint32 l_208 = l_207 & uint32(255);
const uint32 l_209 = l_208 * uint32(3);
const uint32 l_210 = l_209 >> uint32(8);
const __half l_211 = __half(l_210);
const __half l_212 = l_211 * l_mut_59;
const __half l_213 = l_212 - l_mut_59;
l_arr_40[uint32(2)] = l_213;
const uint32 l_214 = l_mut_16 >> uint32(16);
const uint32 l_215 = l_214 & uint32(255);
const uint32 l_216 = l_215 * l_mut_128;
const uint32 l_217 = l_216 & uint32(255);
const uint32 l_218 = l_217 * uint32(3);
const uint32 l_219 = l_218 >> uint32(8);
const __half l_220 = __half(l_219);
const __half l_221 = l_220 * l_mut_59;
const __half l_222 = l_221 - l_mut_59;
l_arr_40[uint32(6)] = l_222;
const uint32 l_223 = l_mut_53 >> uint32(16);
const uint32 l_224 = l_223 & uint32(255);
const uint32 l_225 = l_224 * l_mut_128;
const uint32 l_226 = l_225 & uint32(255);
const uint32 l_227 = l_226 * uint32(3);
const uint32 l_228 = l_227 >> uint32(8);
const __half l_229 = __half(l_228);
const __half l_230 = l_229 * l_mut_59;
const __half l_231 = l_230 - l_mut_59;
l_arr_40[uint32(10)] = l_231;
const uint32 l_232 = l_mut_54 >> uint32(16);
const uint32 l_233 = l_232 & uint32(255);
const uint32 l_234 = l_233 * l_mut_128;
const uint32 l_235 = l_234 & uint32(255);
const uint32 l_236 = l_235 * uint32(3);
const uint32 l_237 = l_236 >> uint32(8);
const __half l_238 = __half(l_237);
const __half l_239 = l_238 * l_mut_59;
const __half l_240 = l_239 - l_mut_59;
l_arr_40[uint32(14)] = l_240;
const uint32 l_241 = l_mut_23 >> uint32(24);
const uint32 l_242 = l_241 & uint32(255);
const uint32 l_243 = l_242 * l_mut_128;
const uint32 l_244 = l_243 & uint32(255);
const uint32 l_245 = l_244 * uint32(3);
const uint32 l_246 = l_245 >> uint32(8);
const __half l_247 = __half(l_246);
const __half l_248 = l_247 * l_mut_59;
const __half l_249 = l_248 - l_mut_59;
l_arr_40[uint32(3)] = l_249;
const uint32 l_250 = l_mut_16 >> uint32(24);
const uint32 l_251 = l_250 & uint32(255);
const uint32 l_252 = l_251 * l_mut_128;
const uint32 l_253 = l_252 & uint32(255);
const uint32 l_254 = l_253 * uint32(3);
const uint32 l_255 = l_254 >> uint32(8);
const __half l_256 = __half(l_255);
const __half l_257 = l_256 * l_mut_59;
const __half l_258 = l_257 - l_mut_59;
l_arr_40[uint32(7)] = l_258;
const uint32 l_259 = l_mut_53 >> uint32(24);
const uint32 l_260 = l_259 & uint32(255);
const uint32 l_261 = l_260 * l_mut_128;
const uint32 l_262 = l_261 & uint32(255);
const uint32 l_263 = l_262 * uint32(3);
const uint32 l_264 = l_263 >> uint32(8);
const __half l_265 = __half(l_264);
const __half l_266 = l_265 * l_mut_59;
const __half l_267 = l_266 - l_mut_59;
l_arr_40[uint32(11)] = l_267;
const uint32 l_268 = l_mut_54 >> uint32(24);
const uint32 l_269 = l_268 & uint32(255);
const uint32 l_270 = l_269 * l_mut_128;
const uint32 l_271 = l_270 & uint32(255);
const uint32 l_272 = l_271 * uint32(3);
const uint32 l_273 = l_272 >> uint32(8);
const __half l_274 = __half(l_273);
const __half l_275 = l_274 * l_mut_59;
const __half l_276 = l_275 - l_mut_59;
l_arr_40[uint32(15)] = l_276;
} else {
const bool l_277 = l_86 < uint32(7);
if (l_277) {
l_mut_128 = uint32(1);
l_mut_278 = uint32(3);
const bool l_279 = l_86 == uint32(6);
if (l_279) {
l_mut_128 = uint32(9);
l_mut_278 = uint32(27);
}
const uint32 l_280 = l_mut_55 >> uint32(0);
const uint32 l_281 = l_280 & uint32(255);
const uint32 l_282 = l_mut_56 >> uint32(0);
const uint32 l_283 = l_282 & uint32(255);
const uint32 l_284 = l_281 * l_mut_128;
const uint32 l_285 = l_284 & uint32(255);
const uint32 l_286 = l_285 * uint32(3);
const uint32 l_287 = l_286 >> uint32(8);
const __half l_288 = __half(l_287);
const __half l_289 = l_288 * l_mut_59;
const __half l_290 = l_289 - l_mut_59;
l_arr_40[uint32(0)] = l_290;
const uint32 l_291 = l_283 * l_mut_128;
const uint32 l_292 = l_291 & uint32(255);
const uint32 l_293 = l_292 * uint32(3);
const uint32 l_294 = l_293 >> uint32(8);
const __half l_295 = __half(l_294);
const __half l_296 = l_295 * l_mut_59;
const __half l_297 = l_296 - l_mut_59;
l_arr_40[uint32(4)] = l_297;
const uint32 l_298 = l_281 * l_mut_278;
const uint32 l_299 = l_298 & uint32(255);
const uint32 l_300 = l_299 * uint32(3);
const uint32 l_301 = l_300 >> uint32(8);
const __half l_302 = __half(l_301);
const __half l_303 = l_302 * l_mut_59;
const __half l_304 = l_303 - l_mut_59;
l_arr_40[uint32(8)] = l_304;
const uint32 l_305 = l_283 * l_mut_278;
const uint32 l_306 = l_305 & uint32(255);
const uint32 l_307 = l_306 * uint32(3);
const uint32 l_308 = l_307 >> uint32(8);
const __half l_309 = __half(l_308);
const __half l_310 = l_309 * l_mut_59;
const __half l_311 = l_310 - l_mut_59;
l_arr_40[uint32(12)] = l_311;
const uint32 l_312 = l_mut_55 >> uint32(8);
const uint32 l_313 = l_312 & uint32(255);
const uint32 l_314 = l_mut_56 >> uint32(8);
const uint32 l_315 = l_314 & uint32(255);
const uint32 l_316 = l_313 * l_mut_128;
const uint32 l_317 = l_316 & uint32(255);
const uint32 l_318 = l_317 * uint32(3);
const uint32 l_319 = l_318 >> uint32(8);
const __half l_320 = __half(l_319);
const __half l_321 = l_320 * l_mut_59;
const __half l_322 = l_321 - l_mut_59;
l_arr_40[uint32(1)] = l_322;
const uint32 l_323 = l_315 * l_mut_128;
const uint32 l_324 = l_323 & uint32(255);
const uint32 l_325 = l_324 * uint32(3);
const uint32 l_326 = l_325 >> uint32(8);
const __half l_327 = __half(l_326);
const __half l_328 = l_327 * l_mut_59;
const __half l_329 = l_328 - l_mut_59;
l_arr_40[uint32(5)] = l_329;
const uint32 l_330 = l_313 * l_mut_278;
const uint32 l_331 = l_330 & uint32(255);
const uint32 l_332 = l_331 * uint32(3);
const uint32 l_333 = l_332 >> uint32(8);
const __half l_334 = __half(l_333);
const __half l_335 = l_334 * l_mut_59;
const __half l_336 = l_335 - l_mut_59;
l_arr_40[uint32(9)] = l_336;
const uint32 l_337 = l_315 * l_mut_278;
const uint32 l_338 = l_337 & uint32(255);
const uint32 l_339 = l_338 * uint32(3);
const uint32 l_340 = l_339 >> uint32(8);
const __half l_341 = __half(l_340);
const __half l_342 = l_341 * l_mut_59;
const __half l_343 = l_342 - l_mut_59;
l_arr_40[uint32(13)] = l_343;
const uint32 l_344 = l_mut_55 >> uint32(16);
const uint32 l_345 = l_344 & uint32(255);
const uint32 l_346 = l_mut_56 >> uint32(16);
const uint32 l_347 = l_346 & uint32(255);
const uint32 l_348 = l_345 * l_mut_128;
const uint32 l_349 = l_348 & uint32(255);
const uint32 l_350 = l_349 * uint32(3);
const uint32 l_351 = l_350 >> uint32(8);
const __half l_352 = __half(l_351);
const __half l_353 = l_352 * l_mut_59;
const __half l_354 = l_353 - l_mut_59;
l_arr_40[uint32(2)] = l_354;
const uint32 l_355 = l_347 * l_mut_128;
const uint32 l_356 = l_355 & uint32(255);
const uint32 l_357 = l_356 * uint32(3);
const uint32 l_358 = l_357 >> uint32(8);
const __half l_359 = __half(l_358);
const __half l_360 = l_359 * l_mut_59;
const __half l_361 = l_360 - l_mut_59;
l_arr_40[uint32(6)] = l_361;
const uint32 l_362 = l_345 * l_mut_278;
const uint32 l_363 = l_362 & uint32(255);
const uint32 l_364 = l_363 * uint32(3);
const uint32 l_365 = l_364 >> uint32(8);
const __half l_366 = __half(l_365);
const __half l_367 = l_366 * l_mut_59;
const __half l_368 = l_367 - l_mut_59;
l_arr_40[uint32(10)] = l_368;
const uint32 l_369 = l_347 * l_mut_278;
const uint32 l_370 = l_369 & uint32(255);
const uint32 l_371 = l_370 * uint32(3);
const uint32 l_372 = l_371 >> uint32(8);
const __half l_373 = __half(l_372);
const __half l_374 = l_373 * l_mut_59;
const __half l_375 = l_374 - l_mut_59;
l_arr_40[uint32(14)] = l_375;
const uint32 l_376 = l_mut_55 >> uint32(24);
const uint32 l_377 = l_376 & uint32(255);
const uint32 l_378 = l_mut_56 >> uint32(24);
const uint32 l_379 = l_378 & uint32(255);
const uint32 l_380 = l_377 * l_mut_128;
const uint32 l_381 = l_380 & uint32(255);
const uint32 l_382 = l_381 * uint32(3);
const uint32 l_383 = l_382 >> uint32(8);
const __half l_384 = __half(l_383);
const __half l_385 = l_384 * l_mut_59;
const __half l_386 = l_385 - l_mut_59;
l_arr_40[uint32(3)] = l_386;
const uint32 l_387 = l_379 * l_mut_128;
const uint32 l_388 = l_387 & uint32(255);
const uint32 l_389 = l_388 * uint32(3);
const uint32 l_390 = l_389 >> uint32(8);
const __half l_391 = __half(l_390);
const __half l_392 = l_391 * l_mut_59;
const __half l_393 = l_392 - l_mut_59;
l_arr_40[uint32(7)] = l_393;
const uint32 l_394 = l_377 * l_mut_278;
const uint32 l_395 = l_394 & uint32(255);
const uint32 l_396 = l_395 * uint32(3);
const uint32 l_397 = l_396 >> uint32(8);
const __half l_398 = __half(l_397);
const __half l_399 = l_398 * l_mut_59;
const __half l_400 = l_399 - l_mut_59;
l_arr_40[uint32(11)] = l_400;
const uint32 l_401 = l_379 * l_mut_278;
const uint32 l_402 = l_401 & uint32(255);
const uint32 l_403 = l_402 * uint32(3);
const uint32 l_404 = l_403 >> uint32(8);
const __half l_405 = __half(l_404);
const __half l_406 = l_405 * l_mut_59;
const __half l_407 = l_406 - l_mut_59;
l_arr_40[uint32(15)] = l_407;
} else {
const uint32 l_408 = l_mut_57 & uint32(255);
const uint32 l_409 = l_mut_57 >> uint32(8);
const uint32 l_410 = l_409 & uint32(255);
l_mut_278 = uint32(1);
const uint32 l_411 = l_mut_55 >> uint32(0);
const uint32 l_412 = l_411 & uint32(255);
const uint32 l_413 = l_412 * uint32(81);
const uint32 l_414 = l_413 & uint32(255);
const uint32 l_415 = l_414 * uint32(3);
const uint32 l_416 = l_415 >> uint32(8);
const __half l_417 = __half(l_416);
const __half l_418 = l_417 * l_mut_59;
const __half l_419 = l_418 - l_mut_59;
l_arr_40[uint32(0)] = l_419;
const uint32 l_420 = l_mut_56 >> uint32(0);
const uint32 l_421 = l_420 & uint32(255);
const uint32 l_422 = l_421 * uint32(81);
const uint32 l_423 = l_422 & uint32(255);
const uint32 l_424 = l_423 * uint32(3);
const uint32 l_425 = l_424 >> uint32(8);
const __half l_426 = __half(l_425);
const __half l_427 = l_426 * l_mut_59;
const __half l_428 = l_427 - l_mut_59;
l_arr_40[uint32(4)] = l_428;
const uint32 l_429 = l_408 * l_mut_278;
const uint32 l_430 = l_429 & uint32(255);
const uint32 l_431 = l_430 * uint32(3);
const uint32 l_432 = l_431 >> uint32(8);
const __half l_433 = __half(l_432);
const __half l_434 = l_433 * l_mut_59;
const __half l_435 = l_434 - l_mut_59;
l_arr_40[uint32(8)] = l_435;
const uint32 l_436 = l_410 * l_mut_278;
const uint32 l_437 = l_436 & uint32(255);
const uint32 l_438 = l_437 * uint32(3);
const uint32 l_439 = l_438 >> uint32(8);
const __half l_440 = __half(l_439);
const __half l_441 = l_440 * l_mut_59;
const __half l_442 = l_441 - l_mut_59;
l_arr_40[uint32(9)] = l_442;
l_mut_278 = l_mut_278 * uint32(3);
const uint32 l_443 = l_mut_55 >> uint32(8);
const uint32 l_444 = l_443 & uint32(255);
const uint32 l_445 = l_444 * uint32(81);
const uint32 l_446 = l_445 & uint32(255);
const uint32 l_447 = l_446 * uint32(3);
const uint32 l_448 = l_447 >> uint32(8);
const __half l_449 = __half(l_448);
const __half l_450 = l_449 * l_mut_59;
const __half l_451 = l_450 - l_mut_59;
l_arr_40[uint32(1)] = l_451;
const uint32 l_452 = l_mut_56 >> uint32(8);
const uint32 l_453 = l_452 & uint32(255);
const uint32 l_454 = l_453 * uint32(81);
const uint32 l_455 = l_454 & uint32(255);
const uint32 l_456 = l_455 * uint32(3);
const uint32 l_457 = l_456 >> uint32(8);
const __half l_458 = __half(l_457);
const __half l_459 = l_458 * l_mut_59;
const __half l_460 = l_459 - l_mut_59;
l_arr_40[uint32(5)] = l_460;
const uint32 l_461 = l_408 * l_mut_278;
const uint32 l_462 = l_461 & uint32(255);
const uint32 l_463 = l_462 * uint32(3);
const uint32 l_464 = l_463 >> uint32(8);
const __half l_465 = __half(l_464);
const __half l_466 = l_465 * l_mut_59;
const __half l_467 = l_466 - l_mut_59;
l_arr_40[uint32(10)] = l_467;
const uint32 l_468 = l_410 * l_mut_278;
const uint32 l_469 = l_468 & uint32(255);
const uint32 l_470 = l_469 * uint32(3);
const uint32 l_471 = l_470 >> uint32(8);
const __half l_472 = __half(l_471);
const __half l_473 = l_472 * l_mut_59;
const __half l_474 = l_473 - l_mut_59;
l_arr_40[uint32(11)] = l_474;
l_mut_278 = l_mut_278 * uint32(3);
const uint32 l_475 = l_mut_55 >> uint32(16);
const uint32 l_476 = l_475 & uint32(255);
const uint32 l_477 = l_476 * uint32(81);
const uint32 l_478 = l_477 & uint32(255);
const uint32 l_479 = l_478 * uint32(3);
const uint32 l_480 = l_479 >> uint32(8);
const __half l_481 = __half(l_480);
const __half l_482 = l_481 * l_mut_59;
const __half l_483 = l_482 - l_mut_59;
l_arr_40[uint32(2)] = l_483;
const uint32 l_484 = l_mut_56 >> uint32(16);
const uint32 l_485 = l_484 & uint32(255);
const uint32 l_486 = l_485 * uint32(81);
const uint32 l_487 = l_486 & uint32(255);
const uint32 l_488 = l_487 * uint32(3);
const uint32 l_489 = l_488 >> uint32(8);
const __half l_490 = __half(l_489);
const __half l_491 = l_490 * l_mut_59;
const __half l_492 = l_491 - l_mut_59;
l_arr_40[uint32(6)] = l_492;
const uint32 l_493 = l_408 * l_mut_278;
const uint32 l_494 = l_493 & uint32(255);
const uint32 l_495 = l_494 * uint32(3);
const uint32 l_496 = l_495 >> uint32(8);
const __half l_497 = __half(l_496);
const __half l_498 = l_497 * l_mut_59;
const __half l_499 = l_498 - l_mut_59;
l_arr_40[uint32(12)] = l_499;
const uint32 l_500 = l_410 * l_mut_278;
const uint32 l_501 = l_500 & uint32(255);
const uint32 l_502 = l_501 * uint32(3);
const uint32 l_503 = l_502 >> uint32(8);
const __half l_504 = __half(l_503);
const __half l_505 = l_504 * l_mut_59;
const __half l_506 = l_505 - l_mut_59;
l_arr_40[uint32(13)] = l_506;
l_mut_278 = l_mut_278 * uint32(3);
const uint32 l_507 = l_mut_55 >> uint32(24);
const uint32 l_508 = l_507 & uint32(255);
const uint32 l_509 = l_508 * uint32(81);
const uint32 l_510 = l_509 & uint32(255);
const uint32 l_511 = l_510 * uint32(3);
const uint32 l_512 = l_511 >> uint32(8);
const __half l_513 = __half(l_512);
const __half l_514 = l_513 * l_mut_59;
const __half l_515 = l_514 - l_mut_59;
l_arr_40[uint32(3)] = l_515;
const uint32 l_516 = l_mut_56 >> uint32(24);
const uint32 l_517 = l_516 & uint32(255);
const uint32 l_518 = l_517 * uint32(81);
const uint32 l_519 = l_518 & uint32(255);
const uint32 l_520 = l_519 * uint32(3);
const uint32 l_521 = l_520 >> uint32(8);
const __half l_522 = __half(l_521);
const __half l_523 = l_522 * l_mut_59;
const __half l_524 = l_523 - l_mut_59;
l_arr_40[uint32(7)] = l_524;
const uint32 l_525 = l_408 * l_mut_278;
const uint32 l_526 = l_525 & uint32(255);
const uint32 l_527 = l_526 * uint32(3);
const uint32 l_528 = l_527 >> uint32(8);
const __half l_529 = __half(l_528);
const __half l_530 = l_529 * l_mut_59;
const __half l_531 = l_530 - l_mut_59;
l_arr_40[uint32(14)] = l_531;
const uint32 l_532 = l_410 * l_mut_278;
const uint32 l_533 = l_532 & uint32(255);
const uint32 l_534 = l_533 * uint32(3);
const uint32 l_535 = l_534 >> uint32(8);
const __half l_536 = __half(l_535);
const __half l_537 = l_536 * l_mut_59;
const __half l_538 = l_537 - l_mut_59;
l_arr_40[uint32(15)] = l_538;
l_mut_278 = l_mut_278 * uint32(3);
}
}
__syncthreads();

const uint32 l_539 = l_44 + uint32(0);
const __half l_540 = l_arr_39[uint32(0)];
shared_memory_41[l_539] = l_540;
const uint32 l_541 = l_44 + uint32(1);
const __half l_542 = l_arr_39[uint32(1)];
shared_memory_41[l_541] = l_542;
const uint32 l_543 = l_44 + uint32(2);
const __half l_544 = l_arr_39[uint32(2)];
shared_memory_41[l_543] = l_544;
const uint32 l_545 = l_44 + uint32(3);
const __half l_546 = l_arr_39[uint32(3)];
shared_memory_41[l_545] = l_546;
const uint32 l_547 = l_44 + uint32(4);
const __half l_548 = l_arr_39[uint32(4)];
shared_memory_41[l_547] = l_548;
const uint32 l_549 = l_44 + uint32(5);
const __half l_550 = l_arr_39[uint32(5)];
shared_memory_41[l_549] = l_550;
const uint32 l_551 = l_44 + uint32(6);
const __half l_552 = l_arr_39[uint32(6)];
shared_memory_41[l_551] = l_552;
const uint32 l_553 = l_44 + uint32(7);
const __half l_554 = l_arr_39[uint32(7)];
shared_memory_41[l_553] = l_554;
const uint32 l_555 = l_44 + uint32(1280);
const uint32 l_556 = l_555 + uint32(0);
const __half l_557 = l_arr_39[uint32(8)];
shared_memory_41[l_556] = l_557;
const uint32 l_558 = l_555 + uint32(1);
const __half l_559 = l_arr_39[uint32(9)];
shared_memory_41[l_558] = l_559;
const uint32 l_560 = l_555 + uint32(2);
const __half l_561 = l_arr_39[uint32(10)];
shared_memory_41[l_560] = l_561;
const uint32 l_562 = l_555 + uint32(3);
const __half l_563 = l_arr_39[uint32(11)];
shared_memory_41[l_562] = l_563;
const uint32 l_564 = l_555 + uint32(4);
const __half l_565 = l_arr_39[uint32(12)];
shared_memory_41[l_564] = l_565;
const uint32 l_566 = l_555 + uint32(5);
const __half l_567 = l_arr_39[uint32(13)];
shared_memory_41[l_566] = l_567;
const uint32 l_568 = l_555 + uint32(6);
const __half l_569 = l_arr_39[uint32(14)];
shared_memory_41[l_568] = l_569;
const uint32 l_570 = l_555 + uint32(7);
const __half l_571 = l_arr_39[uint32(15)];
shared_memory_41[l_570] = l_571;
const uint32 l_572 = l_47 + uint32(0);
const __half l_573 = l_arr_40[uint32(0)];
shared_memory_42[l_572] = l_573;
const uint32 l_574 = l_47 + uint32(1);
const __half l_575 = l_arr_40[uint32(1)];
shared_memory_42[l_574] = l_575;
const uint32 l_576 = l_47 + uint32(2);
const __half l_577 = l_arr_40[uint32(2)];
shared_memory_42[l_576] = l_577;
const uint32 l_578 = l_47 + uint32(3);
const __half l_579 = l_arr_40[uint32(3)];
shared_memory_42[l_578] = l_579;
const uint32 l_580 = l_47 + uint32(4);
const __half l_581 = l_arr_40[uint32(4)];
shared_memory_42[l_580] = l_581;
const uint32 l_582 = l_47 + uint32(5);
const __half l_583 = l_arr_40[uint32(5)];
shared_memory_42[l_582] = l_583;
const uint32 l_584 = l_47 + uint32(6);
const __half l_585 = l_arr_40[uint32(6)];
shared_memory_42[l_584] = l_585;
const uint32 l_586 = l_47 + uint32(7);
const __half l_587 = l_arr_40[uint32(7)];
shared_memory_42[l_586] = l_587;
const uint32 l_588 = l_47 + uint32(8);
const __half l_589 = l_arr_40[uint32(8)];
shared_memory_42[l_588] = l_589;
const uint32 l_590 = l_47 + uint32(9);
const __half l_591 = l_arr_40[uint32(9)];
shared_memory_42[l_590] = l_591;
const uint32 l_592 = l_47 + uint32(10);
const __half l_593 = l_arr_40[uint32(10)];
shared_memory_42[l_592] = l_593;
const uint32 l_594 = l_47 + uint32(11);
const __half l_595 = l_arr_40[uint32(11)];
shared_memory_42[l_594] = l_595;
const uint32 l_596 = l_47 + uint32(12);
const __half l_597 = l_arr_40[uint32(12)];
shared_memory_42[l_596] = l_597;
const uint32 l_598 = l_47 + uint32(13);
const __half l_599 = l_arr_40[uint32(13)];
shared_memory_42[l_598] = l_599;
const uint32 l_600 = l_47 + uint32(14);
const __half l_601 = l_arr_40[uint32(14)];
shared_memory_42[l_600] = l_601;
const uint32 l_602 = l_47 + uint32(15);
const __half l_603 = l_arr_40[uint32(15)];
shared_memory_42[l_602] = l_603;
__syncthreads();

const uint32 l_604 = l_2 * uint32(32);
const uint32 l_605 = l_604 + uint32(0);
const uint32 l_606 = l_605 * uint32(40);
const uint32 l_607 = l_606 + uint32(0);
const uint32 l_608 = uint32(2560) - l_607;
nvcuda::wmma::load_matrix_sync(frag_b_609, shared_memory_42 + l_607, uint32(40));
const uint32 l_610 = l_2 * uint32(32);
const uint32 l_611 = l_610 + uint32(16);
const uint32 l_612 = l_611 * uint32(40);
const uint32 l_613 = l_612 + uint32(0);
const uint32 l_614 = uint32(2560) - l_613;
nvcuda::wmma::load_matrix_sync(frag_b_615, shared_memory_42 + l_613, uint32(40));
const uint32 l_616 = l_1 * uint32(32);
const uint32 l_617 = l_616 + uint32(0);
const uint32 l_618 = l_617 * uint32(40);
const uint32 l_619 = l_618 + uint32(0);
const uint32 l_620 = uint32(2560) - l_619;
nvcuda::wmma::load_matrix_sync(frag_a_621, shared_memory_41 + l_619, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_621, frag_b_609, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_621, frag_b_615, frag_acc_49);
const uint32 l_622 = l_1 * uint32(32);
const uint32 l_623 = l_622 + uint32(16);
const uint32 l_624 = l_623 * uint32(40);
const uint32 l_625 = l_624 + uint32(0);
const uint32 l_626 = uint32(2560) - l_625;
nvcuda::wmma::load_matrix_sync(frag_a_627, shared_memory_41 + l_625, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_50, frag_a_627, frag_b_609, frag_acc_50);
nvcuda::wmma::mma_sync(frag_acc_51, frag_a_627, frag_b_615, frag_acc_51);
const uint32 l_628 = l_2 * uint32(32);
const uint32 l_629 = l_628 + uint32(0);
const uint32 l_630 = l_629 * uint32(40);
const uint32 l_631 = l_630 + uint32(16);
const uint32 l_632 = uint32(2560) - l_631;
nvcuda::wmma::load_matrix_sync(frag_b_633, shared_memory_42 + l_631, uint32(40));
const uint32 l_634 = l_2 * uint32(32);
const uint32 l_635 = l_634 + uint32(16);
const uint32 l_636 = l_635 * uint32(40);
const uint32 l_637 = l_636 + uint32(16);
const uint32 l_638 = uint32(2560) - l_637;
nvcuda::wmma::load_matrix_sync(frag_b_639, shared_memory_42 + l_637, uint32(40));
const uint32 l_640 = l_1 * uint32(32);
const uint32 l_641 = l_640 + uint32(0);
const uint32 l_642 = l_641 * uint32(40);
const uint32 l_643 = l_642 + uint32(16);
const uint32 l_644 = uint32(2560) - l_643;
nvcuda::wmma::load_matrix_sync(frag_a_645, shared_memory_41 + l_643, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_645, frag_b_633, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_645, frag_b_639, frag_acc_49);
const uint32 l_646 = l_1 * uint32(32);
const uint32 l_647 = l_646 + uint32(16);
const uint32 l_648 = l_647 * uint32(40);
const uint32 l_649 = l_648 + uint32(16);
const uint32 l_650 = uint32(2560) - l_649;
nvcuda::wmma::load_matrix_sync(frag_a_651, shared_memory_41 + l_649, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_50, frag_a_651, frag_b_633, frag_acc_50);
nvcuda::wmma::mma_sync(frag_acc_51, frag_a_651, frag_b_639, frag_acc_51);
}
}
const bool l_652 = l_0 == uint32(0);
if (l_652) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_653 = l_6 + uint32(0);
const uint32 l_654 = l_653 + uint32(0);
const uint32 l_655 = l_7 + uint32(0);
__syncthreads();

const uint32 l_656 = threadIdxGlobal + uint32(0);
const uint32 l_657 = l_656 / uint32(32);
const uint32 l_658 = l_654 + l_657;
const uint32 l_659 = l_656 % uint32(32);
const uint32 l_660 = l_655 + l_659;
const bool l_661 = l_658 < l_3;
const bool l_662 = l_660 < l_4;
const bool l_663 = l_661 && l_662;
if (l_663) {
const uint32 l_664 = l_658 * l_4;
const uint32 l_665 = l_664 + l_660;
const float l_666 = shared_memory_38[l_656];
buffer_2[l_665] = l_666;
}
const uint32 l_667 = threadIdxGlobal + uint32(128);
const uint32 l_668 = l_667 / uint32(32);
const uint32 l_669 = l_654 + l_668;
const uint32 l_670 = l_667 % uint32(32);
const uint32 l_671 = l_655 + l_670;
const bool l_672 = l_669 < l_3;
const bool l_673 = l_671 < l_4;
const bool l_674 = l_672 && l_673;
if (l_674) {
const uint32 l_675 = l_669 * l_4;
const uint32 l_676 = l_675 + l_671;
const float l_677 = shared_memory_38[l_667];
buffer_2[l_676] = l_677;
}
const uint32 l_678 = threadIdxGlobal + uint32(256);
const uint32 l_679 = l_678 / uint32(32);
const uint32 l_680 = l_654 + l_679;
const uint32 l_681 = l_678 % uint32(32);
const uint32 l_682 = l_655 + l_681;
const bool l_683 = l_680 < l_3;
const bool l_684 = l_682 < l_4;
const bool l_685 = l_683 && l_684;
if (l_685) {
const uint32 l_686 = l_680 * l_4;
const uint32 l_687 = l_686 + l_682;
const float l_688 = shared_memory_38[l_678];
buffer_2[l_687] = l_688;
}
const uint32 l_689 = threadIdxGlobal + uint32(384);
const uint32 l_690 = l_689 / uint32(32);
const uint32 l_691 = l_654 + l_690;
const uint32 l_692 = l_689 % uint32(32);
const uint32 l_693 = l_655 + l_692;
const bool l_694 = l_691 < l_3;
const bool l_695 = l_693 < l_4;
const bool l_696 = l_694 && l_695;
if (l_696) {
const uint32 l_697 = l_691 * l_4;
const uint32 l_698 = l_697 + l_693;
const float l_699 = shared_memory_38[l_689];
buffer_2[l_698] = l_699;
}
__syncthreads();

const bool l_700 = l_0 == uint32(0);
if (l_700) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_701 = l_6 + uint32(0);
const uint32 l_702 = l_701 + uint32(16);
const uint32 l_703 = l_7 + uint32(0);
__syncthreads();

const uint32 l_704 = threadIdxGlobal + uint32(0);
const uint32 l_705 = l_704 / uint32(32);
const uint32 l_706 = l_702 + l_705;
const uint32 l_707 = l_704 % uint32(32);
const uint32 l_708 = l_703 + l_707;
const bool l_709 = l_706 < l_3;
const bool l_710 = l_708 < l_4;
const bool l_711 = l_709 && l_710;
if (l_711) {
const uint32 l_712 = l_706 * l_4;
const uint32 l_713 = l_712 + l_708;
const float l_714 = shared_memory_38[l_704];
buffer_2[l_713] = l_714;
}
const uint32 l_715 = threadIdxGlobal + uint32(128);
const uint32 l_716 = l_715 / uint32(32);
const uint32 l_717 = l_702 + l_716;
const uint32 l_718 = l_715 % uint32(32);
const uint32 l_719 = l_703 + l_718;
const bool l_720 = l_717 < l_3;
const bool l_721 = l_719 < l_4;
const bool l_722 = l_720 && l_721;
if (l_722) {
const uint32 l_723 = l_717 * l_4;
const uint32 l_724 = l_723 + l_719;
const float l_725 = shared_memory_38[l_715];
buffer_2[l_724] = l_725;
}
const uint32 l_726 = threadIdxGlobal + uint32(256);
const uint32 l_727 = l_726 / uint32(32);
const uint32 l_728 = l_702 + l_727;
const uint32 l_729 = l_726 % uint32(32);
const uint32 l_730 = l_703 + l_729;
const bool l_731 = l_728 < l_3;
const bool l_732 = l_730 < l_4;
const bool l_733 = l_731 && l_732;
if (l_733) {
const uint32 l_734 = l_728 * l_4;
const uint32 l_735 = l_734 + l_730;
const float l_736 = shared_memory_38[l_726];
buffer_2[l_735] = l_736;
}
const uint32 l_737 = threadIdxGlobal + uint32(384);
const uint32 l_738 = l_737 / uint32(32);
const uint32 l_739 = l_702 + l_738;
const uint32 l_740 = l_737 % uint32(32);
const uint32 l_741 = l_703 + l_740;
const bool l_742 = l_739 < l_3;
const bool l_743 = l_741 < l_4;
const bool l_744 = l_742 && l_743;
if (l_744) {
const uint32 l_745 = l_739 * l_4;
const uint32 l_746 = l_745 + l_741;
const float l_747 = shared_memory_38[l_737];
buffer_2[l_746] = l_747;
}
__syncthreads();

const bool l_748 = l_0 == uint32(1);
if (l_748) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_749 = l_6 + uint32(0);
const uint32 l_750 = l_749 + uint32(0);
const uint32 l_751 = l_7 + uint32(32);
__syncthreads();

const uint32 l_752 = threadIdxGlobal + uint32(0);
const uint32 l_753 = l_752 / uint32(32);
const uint32 l_754 = l_750 + l_753;
const uint32 l_755 = l_752 % uint32(32);
const uint32 l_756 = l_751 + l_755;
const bool l_757 = l_754 < l_3;
const bool l_758 = l_756 < l_4;
const bool l_759 = l_757 && l_758;
if (l_759) {
const uint32 l_760 = l_754 * l_4;
const uint32 l_761 = l_760 + l_756;
const float l_762 = shared_memory_38[l_752];
buffer_2[l_761] = l_762;
}
const uint32 l_763 = threadIdxGlobal + uint32(128);
const uint32 l_764 = l_763 / uint32(32);
const uint32 l_765 = l_750 + l_764;
const uint32 l_766 = l_763 % uint32(32);
const uint32 l_767 = l_751 + l_766;
const bool l_768 = l_765 < l_3;
const bool l_769 = l_767 < l_4;
const bool l_770 = l_768 && l_769;
if (l_770) {
const uint32 l_771 = l_765 * l_4;
const uint32 l_772 = l_771 + l_767;
const float l_773 = shared_memory_38[l_763];
buffer_2[l_772] = l_773;
}
const uint32 l_774 = threadIdxGlobal + uint32(256);
const uint32 l_775 = l_774 / uint32(32);
const uint32 l_776 = l_750 + l_775;
const uint32 l_777 = l_774 % uint32(32);
const uint32 l_778 = l_751 + l_777;
const bool l_779 = l_776 < l_3;
const bool l_780 = l_778 < l_4;
const bool l_781 = l_779 && l_780;
if (l_781) {
const uint32 l_782 = l_776 * l_4;
const uint32 l_783 = l_782 + l_778;
const float l_784 = shared_memory_38[l_774];
buffer_2[l_783] = l_784;
}
const uint32 l_785 = threadIdxGlobal + uint32(384);
const uint32 l_786 = l_785 / uint32(32);
const uint32 l_787 = l_750 + l_786;
const uint32 l_788 = l_785 % uint32(32);
const uint32 l_789 = l_751 + l_788;
const bool l_790 = l_787 < l_3;
const bool l_791 = l_789 < l_4;
const bool l_792 = l_790 && l_791;
if (l_792) {
const uint32 l_793 = l_787 * l_4;
const uint32 l_794 = l_793 + l_789;
const float l_795 = shared_memory_38[l_785];
buffer_2[l_794] = l_795;
}
__syncthreads();

const bool l_796 = l_0 == uint32(1);
if (l_796) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_797 = l_6 + uint32(0);
const uint32 l_798 = l_797 + uint32(16);
const uint32 l_799 = l_7 + uint32(32);
__syncthreads();

const uint32 l_800 = threadIdxGlobal + uint32(0);
const uint32 l_801 = l_800 / uint32(32);
const uint32 l_802 = l_798 + l_801;
const uint32 l_803 = l_800 % uint32(32);
const uint32 l_804 = l_799 + l_803;
const bool l_805 = l_802 < l_3;
const bool l_806 = l_804 < l_4;
const bool l_807 = l_805 && l_806;
if (l_807) {
const uint32 l_808 = l_802 * l_4;
const uint32 l_809 = l_808 + l_804;
const float l_810 = shared_memory_38[l_800];
buffer_2[l_809] = l_810;
}
const uint32 l_811 = threadIdxGlobal + uint32(128);
const uint32 l_812 = l_811 / uint32(32);
const uint32 l_813 = l_798 + l_812;
const uint32 l_814 = l_811 % uint32(32);
const uint32 l_815 = l_799 + l_814;
const bool l_816 = l_813 < l_3;
const bool l_817 = l_815 < l_4;
const bool l_818 = l_816 && l_817;
if (l_818) {
const uint32 l_819 = l_813 * l_4;
const uint32 l_820 = l_819 + l_815;
const float l_821 = shared_memory_38[l_811];
buffer_2[l_820] = l_821;
}
const uint32 l_822 = threadIdxGlobal + uint32(256);
const uint32 l_823 = l_822 / uint32(32);
const uint32 l_824 = l_798 + l_823;
const uint32 l_825 = l_822 % uint32(32);
const uint32 l_826 = l_799 + l_825;
const bool l_827 = l_824 < l_3;
const bool l_828 = l_826 < l_4;
const bool l_829 = l_827 && l_828;
if (l_829) {
const uint32 l_830 = l_824 * l_4;
const uint32 l_831 = l_830 + l_826;
const float l_832 = shared_memory_38[l_822];
buffer_2[l_831] = l_832;
}
const uint32 l_833 = threadIdxGlobal + uint32(384);
const uint32 l_834 = l_833 / uint32(32);
const uint32 l_835 = l_798 + l_834;
const uint32 l_836 = l_833 % uint32(32);
const uint32 l_837 = l_799 + l_836;
const bool l_838 = l_835 < l_3;
const bool l_839 = l_837 < l_4;
const bool l_840 = l_838 && l_839;
if (l_840) {
const uint32 l_841 = l_835 * l_4;
const uint32 l_842 = l_841 + l_837;
const float l_843 = shared_memory_38[l_833];
buffer_2[l_842] = l_843;
}
__syncthreads();

const bool l_844 = l_0 == uint32(2);
if (l_844) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_845 = l_6 + uint32(32);
const uint32 l_846 = l_845 + uint32(0);
const uint32 l_847 = l_7 + uint32(0);
__syncthreads();

const uint32 l_848 = threadIdxGlobal + uint32(0);
const uint32 l_849 = l_848 / uint32(32);
const uint32 l_850 = l_846 + l_849;
const uint32 l_851 = l_848 % uint32(32);
const uint32 l_852 = l_847 + l_851;
const bool l_853 = l_850 < l_3;
const bool l_854 = l_852 < l_4;
const bool l_855 = l_853 && l_854;
if (l_855) {
const uint32 l_856 = l_850 * l_4;
const uint32 l_857 = l_856 + l_852;
const float l_858 = shared_memory_38[l_848];
buffer_2[l_857] = l_858;
}
const uint32 l_859 = threadIdxGlobal + uint32(128);
const uint32 l_860 = l_859 / uint32(32);
const uint32 l_861 = l_846 + l_860;
const uint32 l_862 = l_859 % uint32(32);
const uint32 l_863 = l_847 + l_862;
const bool l_864 = l_861 < l_3;
const bool l_865 = l_863 < l_4;
const bool l_866 = l_864 && l_865;
if (l_866) {
const uint32 l_867 = l_861 * l_4;
const uint32 l_868 = l_867 + l_863;
const float l_869 = shared_memory_38[l_859];
buffer_2[l_868] = l_869;
}
const uint32 l_870 = threadIdxGlobal + uint32(256);
const uint32 l_871 = l_870 / uint32(32);
const uint32 l_872 = l_846 + l_871;
const uint32 l_873 = l_870 % uint32(32);
const uint32 l_874 = l_847 + l_873;
const bool l_875 = l_872 < l_3;
const bool l_876 = l_874 < l_4;
const bool l_877 = l_875 && l_876;
if (l_877) {
const uint32 l_878 = l_872 * l_4;
const uint32 l_879 = l_878 + l_874;
const float l_880 = shared_memory_38[l_870];
buffer_2[l_879] = l_880;
}
const uint32 l_881 = threadIdxGlobal + uint32(384);
const uint32 l_882 = l_881 / uint32(32);
const uint32 l_883 = l_846 + l_882;
const uint32 l_884 = l_881 % uint32(32);
const uint32 l_885 = l_847 + l_884;
const bool l_886 = l_883 < l_3;
const bool l_887 = l_885 < l_4;
const bool l_888 = l_886 && l_887;
if (l_888) {
const uint32 l_889 = l_883 * l_4;
const uint32 l_890 = l_889 + l_885;
const float l_891 = shared_memory_38[l_881];
buffer_2[l_890] = l_891;
}
__syncthreads();

const bool l_892 = l_0 == uint32(2);
if (l_892) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_893 = l_6 + uint32(32);
const uint32 l_894 = l_893 + uint32(16);
const uint32 l_895 = l_7 + uint32(0);
__syncthreads();

const uint32 l_896 = threadIdxGlobal + uint32(0);
const uint32 l_897 = l_896 / uint32(32);
const uint32 l_898 = l_894 + l_897;
const uint32 l_899 = l_896 % uint32(32);
const uint32 l_900 = l_895 + l_899;
const bool l_901 = l_898 < l_3;
const bool l_902 = l_900 < l_4;
const bool l_903 = l_901 && l_902;
if (l_903) {
const uint32 l_904 = l_898 * l_4;
const uint32 l_905 = l_904 + l_900;
const float l_906 = shared_memory_38[l_896];
buffer_2[l_905] = l_906;
}
const uint32 l_907 = threadIdxGlobal + uint32(128);
const uint32 l_908 = l_907 / uint32(32);
const uint32 l_909 = l_894 + l_908;
const uint32 l_910 = l_907 % uint32(32);
const uint32 l_911 = l_895 + l_910;
const bool l_912 = l_909 < l_3;
const bool l_913 = l_911 < l_4;
const bool l_914 = l_912 && l_913;
if (l_914) {
const uint32 l_915 = l_909 * l_4;
const uint32 l_916 = l_915 + l_911;
const float l_917 = shared_memory_38[l_907];
buffer_2[l_916] = l_917;
}
const uint32 l_918 = threadIdxGlobal + uint32(256);
const uint32 l_919 = l_918 / uint32(32);
const uint32 l_920 = l_894 + l_919;
const uint32 l_921 = l_918 % uint32(32);
const uint32 l_922 = l_895 + l_921;
const bool l_923 = l_920 < l_3;
const bool l_924 = l_922 < l_4;
const bool l_925 = l_923 && l_924;
if (l_925) {
const uint32 l_926 = l_920 * l_4;
const uint32 l_927 = l_926 + l_922;
const float l_928 = shared_memory_38[l_918];
buffer_2[l_927] = l_928;
}
const uint32 l_929 = threadIdxGlobal + uint32(384);
const uint32 l_930 = l_929 / uint32(32);
const uint32 l_931 = l_894 + l_930;
const uint32 l_932 = l_929 % uint32(32);
const uint32 l_933 = l_895 + l_932;
const bool l_934 = l_931 < l_3;
const bool l_935 = l_933 < l_4;
const bool l_936 = l_934 && l_935;
if (l_936) {
const uint32 l_937 = l_931 * l_4;
const uint32 l_938 = l_937 + l_933;
const float l_939 = shared_memory_38[l_929];
buffer_2[l_938] = l_939;
}
__syncthreads();

const bool l_940 = l_0 == uint32(3);
if (l_940) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_941 = l_6 + uint32(32);
const uint32 l_942 = l_941 + uint32(0);
const uint32 l_943 = l_7 + uint32(32);
__syncthreads();

const uint32 l_944 = threadIdxGlobal + uint32(0);
const uint32 l_945 = l_944 / uint32(32);
const uint32 l_946 = l_942 + l_945;
const uint32 l_947 = l_944 % uint32(32);
const uint32 l_948 = l_943 + l_947;
const bool l_949 = l_946 < l_3;
const bool l_950 = l_948 < l_4;
const bool l_951 = l_949 && l_950;
if (l_951) {
const uint32 l_952 = l_946 * l_4;
const uint32 l_953 = l_952 + l_948;
const float l_954 = shared_memory_38[l_944];
buffer_2[l_953] = l_954;
}
const uint32 l_955 = threadIdxGlobal + uint32(128);
const uint32 l_956 = l_955 / uint32(32);
const uint32 l_957 = l_942 + l_956;
const uint32 l_958 = l_955 % uint32(32);
const uint32 l_959 = l_943 + l_958;
const bool l_960 = l_957 < l_3;
const bool l_961 = l_959 < l_4;
const bool l_962 = l_960 && l_961;
if (l_962) {
const uint32 l_963 = l_957 * l_4;
const uint32 l_964 = l_963 + l_959;
const float l_965 = shared_memory_38[l_955];
buffer_2[l_964] = l_965;
}
const uint32 l_966 = threadIdxGlobal + uint32(256);
const uint32 l_967 = l_966 / uint32(32);
const uint32 l_968 = l_942 + l_967;
const uint32 l_969 = l_966 % uint32(32);
const uint32 l_970 = l_943 + l_969;
const bool l_971 = l_968 < l_3;
const bool l_972 = l_970 < l_4;
const bool l_973 = l_971 && l_972;
if (l_973) {
const uint32 l_974 = l_968 * l_4;
const uint32 l_975 = l_974 + l_970;
const float l_976 = shared_memory_38[l_966];
buffer_2[l_975] = l_976;
}
const uint32 l_977 = threadIdxGlobal + uint32(384);
const uint32 l_978 = l_977 / uint32(32);
const uint32 l_979 = l_942 + l_978;
const uint32 l_980 = l_977 % uint32(32);
const uint32 l_981 = l_943 + l_980;
const bool l_982 = l_979 < l_3;
const bool l_983 = l_981 < l_4;
const bool l_984 = l_982 && l_983;
if (l_984) {
const uint32 l_985 = l_979 * l_4;
const uint32 l_986 = l_985 + l_981;
const float l_987 = shared_memory_38[l_977];
buffer_2[l_986] = l_987;
}
__syncthreads();

const bool l_988 = l_0 == uint32(3);
if (l_988) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_989 = l_6 + uint32(32);
const uint32 l_990 = l_989 + uint32(16);
const uint32 l_991 = l_7 + uint32(32);
__syncthreads();

const uint32 l_992 = threadIdxGlobal + uint32(0);
const uint32 l_993 = l_992 / uint32(32);
const uint32 l_994 = l_990 + l_993;
const uint32 l_995 = l_992 % uint32(32);
const uint32 l_996 = l_991 + l_995;
const bool l_997 = l_994 < l_3;
const bool l_998 = l_996 < l_4;
const bool l_999 = l_997 && l_998;
if (l_999) {
const uint32 l_1000 = l_994 * l_4;
const uint32 l_1001 = l_1000 + l_996;
const float l_1002 = shared_memory_38[l_992];
buffer_2[l_1001] = l_1002;
}
const uint32 l_1003 = threadIdxGlobal + uint32(128);
const uint32 l_1004 = l_1003 / uint32(32);
const uint32 l_1005 = l_990 + l_1004;
const uint32 l_1006 = l_1003 % uint32(32);
const uint32 l_1007 = l_991 + l_1006;
const bool l_1008 = l_1005 < l_3;
const bool l_1009 = l_1007 < l_4;
const bool l_1010 = l_1008 && l_1009;
if (l_1010) {
const uint32 l_1011 = l_1005 * l_4;
const uint32 l_1012 = l_1011 + l_1007;
const float l_1013 = shared_memory_38[l_1003];
buffer_2[l_1012] = l_1013;
}
const uint32 l_1014 = threadIdxGlobal + uint32(256);
const uint32 l_1015 = l_1014 / uint32(32);
const uint32 l_1016 = l_990 + l_1015;
const uint32 l_1017 = l_1014 % uint32(32);
const uint32 l_1018 = l_991 + l_1017;
const bool l_1019 = l_1016 < l_3;
const bool l_1020 = l_1018 < l_4;
const bool l_1021 = l_1019 && l_1020;
if (l_1021) {
const uint32 l_1022 = l_1016 * l_4;
const uint32 l_1023 = l_1022 + l_1018;
const float l_1024 = shared_memory_38[l_1014];
buffer_2[l_1023] = l_1024;
}
const uint32 l_1025 = threadIdxGlobal + uint32(384);
const uint32 l_1026 = l_1025 / uint32(32);
const uint32 l_1027 = l_990 + l_1026;
const uint32 l_1028 = l_1025 % uint32(32);
const uint32 l_1029 = l_991 + l_1028;
const bool l_1030 = l_1027 < l_3;
const bool l_1031 = l_1029 < l_4;
const bool l_1032 = l_1030 && l_1031;
if (l_1032) {
const uint32 l_1033 = l_1027 * l_4;
const uint32 l_1034 = l_1033 + l_1029;
const float l_1035 = shared_memory_38[l_1025];
buffer_2[l_1034] = l_1035;
}
__syncthreads();


}
