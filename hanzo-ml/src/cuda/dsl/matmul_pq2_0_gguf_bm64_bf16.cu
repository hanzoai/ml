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
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_298;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_304;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_310;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_316;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_322;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16, __half, nvcuda::wmma::col_major> frag_b_328;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_334;
nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16, __half, nvcuda::wmma::row_major> frag_a_340;
const uint32 l_77 = l_mut_52 * uint32(4);
const uint32 l_78 = l_77 + l_mut_76;
const uint32 l_79 = l_78 * uint32(8);
const uint32 l_80 = uint32(2) * l_mut_76;
const uint32 l_81 = l_80 + l_32;
const uint32 l_82 = l_27 + l_79;
const __nv_bfloat16_4 l_83 = buffer_1[l_82];
const uint32 l_84 = l_82 + uint32(1);
const __nv_bfloat16_4 l_85 = buffer_1[l_84];
const __nv_bfloat16 l_86 = l_83.i_0;
const float l_87 = float(l_86);
const __half l_88 = __half(l_87);
l_arr_39[uint32(0)] = l_88;
const __nv_bfloat16 l_89 = l_85.i_0;
const float l_90 = float(l_89);
const __half l_91 = __half(l_90);
l_arr_39[uint32(4)] = l_91;
const __nv_bfloat16 l_92 = l_83.i_1;
const float l_93 = float(l_92);
const __half l_94 = __half(l_93);
l_arr_39[uint32(1)] = l_94;
const __nv_bfloat16 l_95 = l_85.i_1;
const float l_96 = float(l_95);
const __half l_97 = __half(l_96);
l_arr_39[uint32(5)] = l_97;
const __nv_bfloat16 l_98 = l_83.i_2;
const float l_99 = float(l_98);
const __half l_100 = __half(l_99);
l_arr_39[uint32(2)] = l_100;
const __nv_bfloat16 l_101 = l_85.i_2;
const float l_102 = float(l_101);
const __half l_103 = __half(l_102);
l_arr_39[uint32(6)] = l_103;
const __nv_bfloat16 l_104 = l_83.i_3;
const float l_105 = float(l_104);
const __half l_106 = __half(l_105);
l_arr_39[uint32(3)] = l_106;
const __nv_bfloat16 l_107 = l_85.i_3;
const float l_108 = float(l_107);
const __half l_109 = __half(l_108);
l_arr_39[uint32(7)] = l_109;
const uint32 l_110 = l_30 + l_79;
const __nv_bfloat16_4 l_111 = buffer_1[l_110];
const uint32 l_112 = l_110 + uint32(1);
const __nv_bfloat16_4 l_113 = buffer_1[l_112];
const __nv_bfloat16 l_114 = l_111.i_0;
const float l_115 = float(l_114);
const __half l_116 = __half(l_115);
l_arr_39[uint32(8)] = l_116;
const __nv_bfloat16 l_117 = l_113.i_0;
const float l_118 = float(l_117);
const __half l_119 = __half(l_118);
l_arr_39[uint32(12)] = l_119;
const __nv_bfloat16 l_120 = l_111.i_1;
const float l_121 = float(l_120);
const __half l_122 = __half(l_121);
l_arr_39[uint32(9)] = l_122;
const __nv_bfloat16 l_123 = l_113.i_1;
const float l_124 = float(l_123);
const __half l_125 = __half(l_124);
l_arr_39[uint32(13)] = l_125;
const __nv_bfloat16 l_126 = l_111.i_2;
const float l_127 = float(l_126);
const __half l_128 = __half(l_127);
l_arr_39[uint32(10)] = l_128;
const __nv_bfloat16 l_129 = l_113.i_2;
const float l_130 = float(l_129);
const __half l_131 = __half(l_130);
l_arr_39[uint32(14)] = l_131;
const __nv_bfloat16 l_132 = l_111.i_3;
const float l_133 = float(l_132);
const __half l_134 = __half(l_133);
l_arr_39[uint32(11)] = l_134;
const __nv_bfloat16 l_135 = l_113.i_3;
const float l_136 = float(l_135);
const __half l_137 = __half(l_136);
l_arr_39[uint32(15)] = l_137;
const uint32 l_138 = l_mut_23 + l_81;
const uint32 l_139 = buffer_0[l_138];
const uint32 l_140 = l_mut_23 + l_81;
const uint32 l_141 = l_140 + l_mut_55;
const uint32 l_142 = buffer_0[l_141];
const uint32 l_143 = l_139 >> l_mut_16;
const uint32 l_144 = l_142 << uint32(15);
const uint32 l_145 = uint32(17) - l_mut_16;
const uint32 l_146 = l_144 << l_145;
const uint32 l_147 = l_143 | l_146;
const uint32 l_148 = l_147 >> uint32(0);
const uint32 l_149 = l_148 & uint32(3);
const __half l_150 = __half(l_149);
const __half l_151 = l_150 * l_mut_54;
const __half l_152 = l_151 - l_mut_54;
l_arr_40[uint32(0)] = l_152;
const uint32 l_153 = l_147 >> uint32(2);
const uint32 l_154 = l_153 & uint32(3);
const __half l_155 = __half(l_154);
const __half l_156 = l_155 * l_mut_54;
const __half l_157 = l_156 - l_mut_54;
l_arr_40[uint32(1)] = l_157;
const uint32 l_158 = l_147 >> uint32(4);
const uint32 l_159 = l_158 & uint32(3);
const __half l_160 = __half(l_159);
const __half l_161 = l_160 * l_mut_54;
const __half l_162 = l_161 - l_mut_54;
l_arr_40[uint32(2)] = l_162;
const uint32 l_163 = l_147 >> uint32(6);
const uint32 l_164 = l_163 & uint32(3);
const __half l_165 = __half(l_164);
const __half l_166 = l_165 * l_mut_54;
const __half l_167 = l_166 - l_mut_54;
l_arr_40[uint32(3)] = l_167;
const uint32 l_168 = l_147 >> uint32(8);
const uint32 l_169 = l_168 & uint32(3);
const __half l_170 = __half(l_169);
const __half l_171 = l_170 * l_mut_54;
const __half l_172 = l_171 - l_mut_54;
l_arr_40[uint32(4)] = l_172;
const uint32 l_173 = l_147 >> uint32(10);
const uint32 l_174 = l_173 & uint32(3);
const __half l_175 = __half(l_174);
const __half l_176 = l_175 * l_mut_54;
const __half l_177 = l_176 - l_mut_54;
l_arr_40[uint32(5)] = l_177;
const uint32 l_178 = l_147 >> uint32(12);
const uint32 l_179 = l_178 & uint32(3);
const __half l_180 = __half(l_179);
const __half l_181 = l_180 * l_mut_54;
const __half l_182 = l_181 - l_mut_54;
l_arr_40[uint32(6)] = l_182;
const uint32 l_183 = l_147 >> uint32(14);
const uint32 l_184 = l_183 & uint32(3);
const __half l_185 = __half(l_184);
const __half l_186 = l_185 * l_mut_54;
const __half l_187 = l_186 - l_mut_54;
l_arr_40[uint32(7)] = l_187;
const uint32 l_188 = l_147 >> uint32(16);
const uint32 l_189 = l_188 & uint32(3);
const __half l_190 = __half(l_189);
const __half l_191 = l_190 * l_mut_54;
const __half l_192 = l_191 - l_mut_54;
l_arr_40[uint32(8)] = l_192;
const uint32 l_193 = l_147 >> uint32(18);
const uint32 l_194 = l_193 & uint32(3);
const __half l_195 = __half(l_194);
const __half l_196 = l_195 * l_mut_54;
const __half l_197 = l_196 - l_mut_54;
l_arr_40[uint32(9)] = l_197;
const uint32 l_198 = l_147 >> uint32(20);
const uint32 l_199 = l_198 & uint32(3);
const __half l_200 = __half(l_199);
const __half l_201 = l_200 * l_mut_54;
const __half l_202 = l_201 - l_mut_54;
l_arr_40[uint32(10)] = l_202;
const uint32 l_203 = l_147 >> uint32(22);
const uint32 l_204 = l_203 & uint32(3);
const __half l_205 = __half(l_204);
const __half l_206 = l_205 * l_mut_54;
const __half l_207 = l_206 - l_mut_54;
l_arr_40[uint32(11)] = l_207;
const uint32 l_208 = l_147 >> uint32(24);
const uint32 l_209 = l_208 & uint32(3);
const __half l_210 = __half(l_209);
const __half l_211 = l_210 * l_mut_54;
const __half l_212 = l_211 - l_mut_54;
l_arr_40[uint32(12)] = l_212;
const uint32 l_213 = l_147 >> uint32(26);
const uint32 l_214 = l_213 & uint32(3);
const __half l_215 = __half(l_214);
const __half l_216 = l_215 * l_mut_54;
const __half l_217 = l_216 - l_mut_54;
l_arr_40[uint32(13)] = l_217;
const uint32 l_218 = l_147 >> uint32(28);
const uint32 l_219 = l_218 & uint32(3);
const __half l_220 = __half(l_219);
const __half l_221 = l_220 * l_mut_54;
const __half l_222 = l_221 - l_mut_54;
l_arr_40[uint32(14)] = l_222;
const uint32 l_223 = l_147 >> uint32(30);
const uint32 l_224 = l_223 & uint32(3);
const __half l_225 = __half(l_224);
const __half l_226 = l_225 * l_mut_54;
const __half l_227 = l_226 - l_mut_54;
l_arr_40[uint32(15)] = l_227;
__syncthreads();

const uint32 l_228 = l_44 + uint32(0);
const __half l_229 = l_arr_39[uint32(0)];
shared_memory_41[l_228] = l_229;
const uint32 l_230 = l_44 + uint32(1);
const __half l_231 = l_arr_39[uint32(1)];
shared_memory_41[l_230] = l_231;
const uint32 l_232 = l_44 + uint32(2);
const __half l_233 = l_arr_39[uint32(2)];
shared_memory_41[l_232] = l_233;
const uint32 l_234 = l_44 + uint32(3);
const __half l_235 = l_arr_39[uint32(3)];
shared_memory_41[l_234] = l_235;
const uint32 l_236 = l_44 + uint32(4);
const __half l_237 = l_arr_39[uint32(4)];
shared_memory_41[l_236] = l_237;
const uint32 l_238 = l_44 + uint32(5);
const __half l_239 = l_arr_39[uint32(5)];
shared_memory_41[l_238] = l_239;
const uint32 l_240 = l_44 + uint32(6);
const __half l_241 = l_arr_39[uint32(6)];
shared_memory_41[l_240] = l_241;
const uint32 l_242 = l_44 + uint32(7);
const __half l_243 = l_arr_39[uint32(7)];
shared_memory_41[l_242] = l_243;
const uint32 l_244 = l_44 + uint32(1280);
const uint32 l_245 = l_244 + uint32(0);
const __half l_246 = l_arr_39[uint32(8)];
shared_memory_41[l_245] = l_246;
const uint32 l_247 = l_244 + uint32(1);
const __half l_248 = l_arr_39[uint32(9)];
shared_memory_41[l_247] = l_248;
const uint32 l_249 = l_244 + uint32(2);
const __half l_250 = l_arr_39[uint32(10)];
shared_memory_41[l_249] = l_250;
const uint32 l_251 = l_244 + uint32(3);
const __half l_252 = l_arr_39[uint32(11)];
shared_memory_41[l_251] = l_252;
const uint32 l_253 = l_244 + uint32(4);
const __half l_254 = l_arr_39[uint32(12)];
shared_memory_41[l_253] = l_254;
const uint32 l_255 = l_244 + uint32(5);
const __half l_256 = l_arr_39[uint32(13)];
shared_memory_41[l_255] = l_256;
const uint32 l_257 = l_244 + uint32(6);
const __half l_258 = l_arr_39[uint32(14)];
shared_memory_41[l_257] = l_258;
const uint32 l_259 = l_244 + uint32(7);
const __half l_260 = l_arr_39[uint32(15)];
shared_memory_41[l_259] = l_260;
const uint32 l_261 = l_47 + uint32(0);
const __half l_262 = l_arr_40[uint32(0)];
shared_memory_42[l_261] = l_262;
const uint32 l_263 = l_47 + uint32(1);
const __half l_264 = l_arr_40[uint32(1)];
shared_memory_42[l_263] = l_264;
const uint32 l_265 = l_47 + uint32(2);
const __half l_266 = l_arr_40[uint32(2)];
shared_memory_42[l_265] = l_266;
const uint32 l_267 = l_47 + uint32(3);
const __half l_268 = l_arr_40[uint32(3)];
shared_memory_42[l_267] = l_268;
const uint32 l_269 = l_47 + uint32(4);
const __half l_270 = l_arr_40[uint32(4)];
shared_memory_42[l_269] = l_270;
const uint32 l_271 = l_47 + uint32(5);
const __half l_272 = l_arr_40[uint32(5)];
shared_memory_42[l_271] = l_272;
const uint32 l_273 = l_47 + uint32(6);
const __half l_274 = l_arr_40[uint32(6)];
shared_memory_42[l_273] = l_274;
const uint32 l_275 = l_47 + uint32(7);
const __half l_276 = l_arr_40[uint32(7)];
shared_memory_42[l_275] = l_276;
const uint32 l_277 = l_47 + uint32(8);
const __half l_278 = l_arr_40[uint32(8)];
shared_memory_42[l_277] = l_278;
const uint32 l_279 = l_47 + uint32(9);
const __half l_280 = l_arr_40[uint32(9)];
shared_memory_42[l_279] = l_280;
const uint32 l_281 = l_47 + uint32(10);
const __half l_282 = l_arr_40[uint32(10)];
shared_memory_42[l_281] = l_282;
const uint32 l_283 = l_47 + uint32(11);
const __half l_284 = l_arr_40[uint32(11)];
shared_memory_42[l_283] = l_284;
const uint32 l_285 = l_47 + uint32(12);
const __half l_286 = l_arr_40[uint32(12)];
shared_memory_42[l_285] = l_286;
const uint32 l_287 = l_47 + uint32(13);
const __half l_288 = l_arr_40[uint32(13)];
shared_memory_42[l_287] = l_288;
const uint32 l_289 = l_47 + uint32(14);
const __half l_290 = l_arr_40[uint32(14)];
shared_memory_42[l_289] = l_290;
const uint32 l_291 = l_47 + uint32(15);
const __half l_292 = l_arr_40[uint32(15)];
shared_memory_42[l_291] = l_292;
__syncthreads();

const uint32 l_293 = l_2 * uint32(32);
const uint32 l_294 = l_293 + uint32(0);
const uint32 l_295 = l_294 * uint32(40);
const uint32 l_296 = l_295 + uint32(0);
const uint32 l_297 = uint32(2560) - l_296;
nvcuda::wmma::load_matrix_sync(frag_b_298, shared_memory_42 + l_296, uint32(40));
const uint32 l_299 = l_2 * uint32(32);
const uint32 l_300 = l_299 + uint32(16);
const uint32 l_301 = l_300 * uint32(40);
const uint32 l_302 = l_301 + uint32(0);
const uint32 l_303 = uint32(2560) - l_302;
nvcuda::wmma::load_matrix_sync(frag_b_304, shared_memory_42 + l_302, uint32(40));
const uint32 l_305 = l_1 * uint32(32);
const uint32 l_306 = l_305 + uint32(0);
const uint32 l_307 = l_306 * uint32(40);
const uint32 l_308 = l_307 + uint32(0);
const uint32 l_309 = uint32(2560) - l_308;
nvcuda::wmma::load_matrix_sync(frag_a_310, shared_memory_41 + l_308, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_310, frag_b_298, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_310, frag_b_304, frag_acc_49);
const uint32 l_311 = l_1 * uint32(32);
const uint32 l_312 = l_311 + uint32(16);
const uint32 l_313 = l_312 * uint32(40);
const uint32 l_314 = l_313 + uint32(0);
const uint32 l_315 = uint32(2560) - l_314;
nvcuda::wmma::load_matrix_sync(frag_a_316, shared_memory_41 + l_314, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_50, frag_a_316, frag_b_298, frag_acc_50);
nvcuda::wmma::mma_sync(frag_acc_51, frag_a_316, frag_b_304, frag_acc_51);
const uint32 l_317 = l_2 * uint32(32);
const uint32 l_318 = l_317 + uint32(0);
const uint32 l_319 = l_318 * uint32(40);
const uint32 l_320 = l_319 + uint32(16);
const uint32 l_321 = uint32(2560) - l_320;
nvcuda::wmma::load_matrix_sync(frag_b_322, shared_memory_42 + l_320, uint32(40));
const uint32 l_323 = l_2 * uint32(32);
const uint32 l_324 = l_323 + uint32(16);
const uint32 l_325 = l_324 * uint32(40);
const uint32 l_326 = l_325 + uint32(16);
const uint32 l_327 = uint32(2560) - l_326;
nvcuda::wmma::load_matrix_sync(frag_b_328, shared_memory_42 + l_326, uint32(40));
const uint32 l_329 = l_1 * uint32(32);
const uint32 l_330 = l_329 + uint32(0);
const uint32 l_331 = l_330 * uint32(40);
const uint32 l_332 = l_331 + uint32(16);
const uint32 l_333 = uint32(2560) - l_332;
nvcuda::wmma::load_matrix_sync(frag_a_334, shared_memory_41 + l_332, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_48, frag_a_334, frag_b_322, frag_acc_48);
nvcuda::wmma::mma_sync(frag_acc_49, frag_a_334, frag_b_328, frag_acc_49);
const uint32 l_335 = l_1 * uint32(32);
const uint32 l_336 = l_335 + uint32(16);
const uint32 l_337 = l_336 * uint32(40);
const uint32 l_338 = l_337 + uint32(16);
const uint32 l_339 = uint32(2560) - l_338;
nvcuda::wmma::load_matrix_sync(frag_a_340, shared_memory_41 + l_338, uint32(40));
nvcuda::wmma::mma_sync(frag_acc_50, frag_a_340, frag_b_322, frag_acc_50);
nvcuda::wmma::mma_sync(frag_acc_51, frag_a_340, frag_b_328, frag_acc_51);
}
}
const bool l_341 = l_0 == uint32(0);
if (l_341) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_342 = l_6 + uint32(0);
const uint32 l_343 = l_342 + uint32(0);
const uint32 l_344 = l_7 + uint32(0);
__syncthreads();

const uint32 l_345 = threadIdxGlobal + uint32(0);
const uint32 l_346 = l_345 / uint32(32);
const uint32 l_347 = l_343 + l_346;
const uint32 l_348 = l_345 % uint32(32);
const uint32 l_349 = l_344 + l_348;
const bool l_350 = l_347 < l_3;
const bool l_351 = l_349 < l_4;
const bool l_352 = l_350 && l_351;
if (l_352) {
const uint32 l_353 = l_347 * l_4;
const uint32 l_354 = l_353 + l_349;
const float l_355 = shared_memory_38[l_345];
const __nv_bfloat16 l_356 = __nv_bfloat16(l_355);
buffer_2[l_354] = l_356;
}
const uint32 l_357 = threadIdxGlobal + uint32(128);
const uint32 l_358 = l_357 / uint32(32);
const uint32 l_359 = l_343 + l_358;
const uint32 l_360 = l_357 % uint32(32);
const uint32 l_361 = l_344 + l_360;
const bool l_362 = l_359 < l_3;
const bool l_363 = l_361 < l_4;
const bool l_364 = l_362 && l_363;
if (l_364) {
const uint32 l_365 = l_359 * l_4;
const uint32 l_366 = l_365 + l_361;
const float l_367 = shared_memory_38[l_357];
const __nv_bfloat16 l_368 = __nv_bfloat16(l_367);
buffer_2[l_366] = l_368;
}
const uint32 l_369 = threadIdxGlobal + uint32(256);
const uint32 l_370 = l_369 / uint32(32);
const uint32 l_371 = l_343 + l_370;
const uint32 l_372 = l_369 % uint32(32);
const uint32 l_373 = l_344 + l_372;
const bool l_374 = l_371 < l_3;
const bool l_375 = l_373 < l_4;
const bool l_376 = l_374 && l_375;
if (l_376) {
const uint32 l_377 = l_371 * l_4;
const uint32 l_378 = l_377 + l_373;
const float l_379 = shared_memory_38[l_369];
const __nv_bfloat16 l_380 = __nv_bfloat16(l_379);
buffer_2[l_378] = l_380;
}
const uint32 l_381 = threadIdxGlobal + uint32(384);
const uint32 l_382 = l_381 / uint32(32);
const uint32 l_383 = l_343 + l_382;
const uint32 l_384 = l_381 % uint32(32);
const uint32 l_385 = l_344 + l_384;
const bool l_386 = l_383 < l_3;
const bool l_387 = l_385 < l_4;
const bool l_388 = l_386 && l_387;
if (l_388) {
const uint32 l_389 = l_383 * l_4;
const uint32 l_390 = l_389 + l_385;
const float l_391 = shared_memory_38[l_381];
const __nv_bfloat16 l_392 = __nv_bfloat16(l_391);
buffer_2[l_390] = l_392;
}
__syncthreads();

const bool l_393 = l_0 == uint32(0);
if (l_393) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_394 = l_6 + uint32(0);
const uint32 l_395 = l_394 + uint32(16);
const uint32 l_396 = l_7 + uint32(0);
__syncthreads();

const uint32 l_397 = threadIdxGlobal + uint32(0);
const uint32 l_398 = l_397 / uint32(32);
const uint32 l_399 = l_395 + l_398;
const uint32 l_400 = l_397 % uint32(32);
const uint32 l_401 = l_396 + l_400;
const bool l_402 = l_399 < l_3;
const bool l_403 = l_401 < l_4;
const bool l_404 = l_402 && l_403;
if (l_404) {
const uint32 l_405 = l_399 * l_4;
const uint32 l_406 = l_405 + l_401;
const float l_407 = shared_memory_38[l_397];
const __nv_bfloat16 l_408 = __nv_bfloat16(l_407);
buffer_2[l_406] = l_408;
}
const uint32 l_409 = threadIdxGlobal + uint32(128);
const uint32 l_410 = l_409 / uint32(32);
const uint32 l_411 = l_395 + l_410;
const uint32 l_412 = l_409 % uint32(32);
const uint32 l_413 = l_396 + l_412;
const bool l_414 = l_411 < l_3;
const bool l_415 = l_413 < l_4;
const bool l_416 = l_414 && l_415;
if (l_416) {
const uint32 l_417 = l_411 * l_4;
const uint32 l_418 = l_417 + l_413;
const float l_419 = shared_memory_38[l_409];
const __nv_bfloat16 l_420 = __nv_bfloat16(l_419);
buffer_2[l_418] = l_420;
}
const uint32 l_421 = threadIdxGlobal + uint32(256);
const uint32 l_422 = l_421 / uint32(32);
const uint32 l_423 = l_395 + l_422;
const uint32 l_424 = l_421 % uint32(32);
const uint32 l_425 = l_396 + l_424;
const bool l_426 = l_423 < l_3;
const bool l_427 = l_425 < l_4;
const bool l_428 = l_426 && l_427;
if (l_428) {
const uint32 l_429 = l_423 * l_4;
const uint32 l_430 = l_429 + l_425;
const float l_431 = shared_memory_38[l_421];
const __nv_bfloat16 l_432 = __nv_bfloat16(l_431);
buffer_2[l_430] = l_432;
}
const uint32 l_433 = threadIdxGlobal + uint32(384);
const uint32 l_434 = l_433 / uint32(32);
const uint32 l_435 = l_395 + l_434;
const uint32 l_436 = l_433 % uint32(32);
const uint32 l_437 = l_396 + l_436;
const bool l_438 = l_435 < l_3;
const bool l_439 = l_437 < l_4;
const bool l_440 = l_438 && l_439;
if (l_440) {
const uint32 l_441 = l_435 * l_4;
const uint32 l_442 = l_441 + l_437;
const float l_443 = shared_memory_38[l_433];
const __nv_bfloat16 l_444 = __nv_bfloat16(l_443);
buffer_2[l_442] = l_444;
}
__syncthreads();

const bool l_445 = l_0 == uint32(1);
if (l_445) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_446 = l_6 + uint32(0);
const uint32 l_447 = l_446 + uint32(0);
const uint32 l_448 = l_7 + uint32(32);
__syncthreads();

const uint32 l_449 = threadIdxGlobal + uint32(0);
const uint32 l_450 = l_449 / uint32(32);
const uint32 l_451 = l_447 + l_450;
const uint32 l_452 = l_449 % uint32(32);
const uint32 l_453 = l_448 + l_452;
const bool l_454 = l_451 < l_3;
const bool l_455 = l_453 < l_4;
const bool l_456 = l_454 && l_455;
if (l_456) {
const uint32 l_457 = l_451 * l_4;
const uint32 l_458 = l_457 + l_453;
const float l_459 = shared_memory_38[l_449];
const __nv_bfloat16 l_460 = __nv_bfloat16(l_459);
buffer_2[l_458] = l_460;
}
const uint32 l_461 = threadIdxGlobal + uint32(128);
const uint32 l_462 = l_461 / uint32(32);
const uint32 l_463 = l_447 + l_462;
const uint32 l_464 = l_461 % uint32(32);
const uint32 l_465 = l_448 + l_464;
const bool l_466 = l_463 < l_3;
const bool l_467 = l_465 < l_4;
const bool l_468 = l_466 && l_467;
if (l_468) {
const uint32 l_469 = l_463 * l_4;
const uint32 l_470 = l_469 + l_465;
const float l_471 = shared_memory_38[l_461];
const __nv_bfloat16 l_472 = __nv_bfloat16(l_471);
buffer_2[l_470] = l_472;
}
const uint32 l_473 = threadIdxGlobal + uint32(256);
const uint32 l_474 = l_473 / uint32(32);
const uint32 l_475 = l_447 + l_474;
const uint32 l_476 = l_473 % uint32(32);
const uint32 l_477 = l_448 + l_476;
const bool l_478 = l_475 < l_3;
const bool l_479 = l_477 < l_4;
const bool l_480 = l_478 && l_479;
if (l_480) {
const uint32 l_481 = l_475 * l_4;
const uint32 l_482 = l_481 + l_477;
const float l_483 = shared_memory_38[l_473];
const __nv_bfloat16 l_484 = __nv_bfloat16(l_483);
buffer_2[l_482] = l_484;
}
const uint32 l_485 = threadIdxGlobal + uint32(384);
const uint32 l_486 = l_485 / uint32(32);
const uint32 l_487 = l_447 + l_486;
const uint32 l_488 = l_485 % uint32(32);
const uint32 l_489 = l_448 + l_488;
const bool l_490 = l_487 < l_3;
const bool l_491 = l_489 < l_4;
const bool l_492 = l_490 && l_491;
if (l_492) {
const uint32 l_493 = l_487 * l_4;
const uint32 l_494 = l_493 + l_489;
const float l_495 = shared_memory_38[l_485];
const __nv_bfloat16 l_496 = __nv_bfloat16(l_495);
buffer_2[l_494] = l_496;
}
__syncthreads();

const bool l_497 = l_0 == uint32(1);
if (l_497) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_498 = l_6 + uint32(0);
const uint32 l_499 = l_498 + uint32(16);
const uint32 l_500 = l_7 + uint32(32);
__syncthreads();

const uint32 l_501 = threadIdxGlobal + uint32(0);
const uint32 l_502 = l_501 / uint32(32);
const uint32 l_503 = l_499 + l_502;
const uint32 l_504 = l_501 % uint32(32);
const uint32 l_505 = l_500 + l_504;
const bool l_506 = l_503 < l_3;
const bool l_507 = l_505 < l_4;
const bool l_508 = l_506 && l_507;
if (l_508) {
const uint32 l_509 = l_503 * l_4;
const uint32 l_510 = l_509 + l_505;
const float l_511 = shared_memory_38[l_501];
const __nv_bfloat16 l_512 = __nv_bfloat16(l_511);
buffer_2[l_510] = l_512;
}
const uint32 l_513 = threadIdxGlobal + uint32(128);
const uint32 l_514 = l_513 / uint32(32);
const uint32 l_515 = l_499 + l_514;
const uint32 l_516 = l_513 % uint32(32);
const uint32 l_517 = l_500 + l_516;
const bool l_518 = l_515 < l_3;
const bool l_519 = l_517 < l_4;
const bool l_520 = l_518 && l_519;
if (l_520) {
const uint32 l_521 = l_515 * l_4;
const uint32 l_522 = l_521 + l_517;
const float l_523 = shared_memory_38[l_513];
const __nv_bfloat16 l_524 = __nv_bfloat16(l_523);
buffer_2[l_522] = l_524;
}
const uint32 l_525 = threadIdxGlobal + uint32(256);
const uint32 l_526 = l_525 / uint32(32);
const uint32 l_527 = l_499 + l_526;
const uint32 l_528 = l_525 % uint32(32);
const uint32 l_529 = l_500 + l_528;
const bool l_530 = l_527 < l_3;
const bool l_531 = l_529 < l_4;
const bool l_532 = l_530 && l_531;
if (l_532) {
const uint32 l_533 = l_527 * l_4;
const uint32 l_534 = l_533 + l_529;
const float l_535 = shared_memory_38[l_525];
const __nv_bfloat16 l_536 = __nv_bfloat16(l_535);
buffer_2[l_534] = l_536;
}
const uint32 l_537 = threadIdxGlobal + uint32(384);
const uint32 l_538 = l_537 / uint32(32);
const uint32 l_539 = l_499 + l_538;
const uint32 l_540 = l_537 % uint32(32);
const uint32 l_541 = l_500 + l_540;
const bool l_542 = l_539 < l_3;
const bool l_543 = l_541 < l_4;
const bool l_544 = l_542 && l_543;
if (l_544) {
const uint32 l_545 = l_539 * l_4;
const uint32 l_546 = l_545 + l_541;
const float l_547 = shared_memory_38[l_537];
const __nv_bfloat16 l_548 = __nv_bfloat16(l_547);
buffer_2[l_546] = l_548;
}
__syncthreads();

const bool l_549 = l_0 == uint32(2);
if (l_549) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_550 = l_6 + uint32(32);
const uint32 l_551 = l_550 + uint32(0);
const uint32 l_552 = l_7 + uint32(0);
__syncthreads();

const uint32 l_553 = threadIdxGlobal + uint32(0);
const uint32 l_554 = l_553 / uint32(32);
const uint32 l_555 = l_551 + l_554;
const uint32 l_556 = l_553 % uint32(32);
const uint32 l_557 = l_552 + l_556;
const bool l_558 = l_555 < l_3;
const bool l_559 = l_557 < l_4;
const bool l_560 = l_558 && l_559;
if (l_560) {
const uint32 l_561 = l_555 * l_4;
const uint32 l_562 = l_561 + l_557;
const float l_563 = shared_memory_38[l_553];
const __nv_bfloat16 l_564 = __nv_bfloat16(l_563);
buffer_2[l_562] = l_564;
}
const uint32 l_565 = threadIdxGlobal + uint32(128);
const uint32 l_566 = l_565 / uint32(32);
const uint32 l_567 = l_551 + l_566;
const uint32 l_568 = l_565 % uint32(32);
const uint32 l_569 = l_552 + l_568;
const bool l_570 = l_567 < l_3;
const bool l_571 = l_569 < l_4;
const bool l_572 = l_570 && l_571;
if (l_572) {
const uint32 l_573 = l_567 * l_4;
const uint32 l_574 = l_573 + l_569;
const float l_575 = shared_memory_38[l_565];
const __nv_bfloat16 l_576 = __nv_bfloat16(l_575);
buffer_2[l_574] = l_576;
}
const uint32 l_577 = threadIdxGlobal + uint32(256);
const uint32 l_578 = l_577 / uint32(32);
const uint32 l_579 = l_551 + l_578;
const uint32 l_580 = l_577 % uint32(32);
const uint32 l_581 = l_552 + l_580;
const bool l_582 = l_579 < l_3;
const bool l_583 = l_581 < l_4;
const bool l_584 = l_582 && l_583;
if (l_584) {
const uint32 l_585 = l_579 * l_4;
const uint32 l_586 = l_585 + l_581;
const float l_587 = shared_memory_38[l_577];
const __nv_bfloat16 l_588 = __nv_bfloat16(l_587);
buffer_2[l_586] = l_588;
}
const uint32 l_589 = threadIdxGlobal + uint32(384);
const uint32 l_590 = l_589 / uint32(32);
const uint32 l_591 = l_551 + l_590;
const uint32 l_592 = l_589 % uint32(32);
const uint32 l_593 = l_552 + l_592;
const bool l_594 = l_591 < l_3;
const bool l_595 = l_593 < l_4;
const bool l_596 = l_594 && l_595;
if (l_596) {
const uint32 l_597 = l_591 * l_4;
const uint32 l_598 = l_597 + l_593;
const float l_599 = shared_memory_38[l_589];
const __nv_bfloat16 l_600 = __nv_bfloat16(l_599);
buffer_2[l_598] = l_600;
}
__syncthreads();

const bool l_601 = l_0 == uint32(2);
if (l_601) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_602 = l_6 + uint32(32);
const uint32 l_603 = l_602 + uint32(16);
const uint32 l_604 = l_7 + uint32(0);
__syncthreads();

const uint32 l_605 = threadIdxGlobal + uint32(0);
const uint32 l_606 = l_605 / uint32(32);
const uint32 l_607 = l_603 + l_606;
const uint32 l_608 = l_605 % uint32(32);
const uint32 l_609 = l_604 + l_608;
const bool l_610 = l_607 < l_3;
const bool l_611 = l_609 < l_4;
const bool l_612 = l_610 && l_611;
if (l_612) {
const uint32 l_613 = l_607 * l_4;
const uint32 l_614 = l_613 + l_609;
const float l_615 = shared_memory_38[l_605];
const __nv_bfloat16 l_616 = __nv_bfloat16(l_615);
buffer_2[l_614] = l_616;
}
const uint32 l_617 = threadIdxGlobal + uint32(128);
const uint32 l_618 = l_617 / uint32(32);
const uint32 l_619 = l_603 + l_618;
const uint32 l_620 = l_617 % uint32(32);
const uint32 l_621 = l_604 + l_620;
const bool l_622 = l_619 < l_3;
const bool l_623 = l_621 < l_4;
const bool l_624 = l_622 && l_623;
if (l_624) {
const uint32 l_625 = l_619 * l_4;
const uint32 l_626 = l_625 + l_621;
const float l_627 = shared_memory_38[l_617];
const __nv_bfloat16 l_628 = __nv_bfloat16(l_627);
buffer_2[l_626] = l_628;
}
const uint32 l_629 = threadIdxGlobal + uint32(256);
const uint32 l_630 = l_629 / uint32(32);
const uint32 l_631 = l_603 + l_630;
const uint32 l_632 = l_629 % uint32(32);
const uint32 l_633 = l_604 + l_632;
const bool l_634 = l_631 < l_3;
const bool l_635 = l_633 < l_4;
const bool l_636 = l_634 && l_635;
if (l_636) {
const uint32 l_637 = l_631 * l_4;
const uint32 l_638 = l_637 + l_633;
const float l_639 = shared_memory_38[l_629];
const __nv_bfloat16 l_640 = __nv_bfloat16(l_639);
buffer_2[l_638] = l_640;
}
const uint32 l_641 = threadIdxGlobal + uint32(384);
const uint32 l_642 = l_641 / uint32(32);
const uint32 l_643 = l_603 + l_642;
const uint32 l_644 = l_641 % uint32(32);
const uint32 l_645 = l_604 + l_644;
const bool l_646 = l_643 < l_3;
const bool l_647 = l_645 < l_4;
const bool l_648 = l_646 && l_647;
if (l_648) {
const uint32 l_649 = l_643 * l_4;
const uint32 l_650 = l_649 + l_645;
const float l_651 = shared_memory_38[l_641];
const __nv_bfloat16 l_652 = __nv_bfloat16(l_651);
buffer_2[l_650] = l_652;
}
__syncthreads();

const bool l_653 = l_0 == uint32(3);
if (l_653) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_48, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_49, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_654 = l_6 + uint32(32);
const uint32 l_655 = l_654 + uint32(0);
const uint32 l_656 = l_7 + uint32(32);
__syncthreads();

const uint32 l_657 = threadIdxGlobal + uint32(0);
const uint32 l_658 = l_657 / uint32(32);
const uint32 l_659 = l_655 + l_658;
const uint32 l_660 = l_657 % uint32(32);
const uint32 l_661 = l_656 + l_660;
const bool l_662 = l_659 < l_3;
const bool l_663 = l_661 < l_4;
const bool l_664 = l_662 && l_663;
if (l_664) {
const uint32 l_665 = l_659 * l_4;
const uint32 l_666 = l_665 + l_661;
const float l_667 = shared_memory_38[l_657];
const __nv_bfloat16 l_668 = __nv_bfloat16(l_667);
buffer_2[l_666] = l_668;
}
const uint32 l_669 = threadIdxGlobal + uint32(128);
const uint32 l_670 = l_669 / uint32(32);
const uint32 l_671 = l_655 + l_670;
const uint32 l_672 = l_669 % uint32(32);
const uint32 l_673 = l_656 + l_672;
const bool l_674 = l_671 < l_3;
const bool l_675 = l_673 < l_4;
const bool l_676 = l_674 && l_675;
if (l_676) {
const uint32 l_677 = l_671 * l_4;
const uint32 l_678 = l_677 + l_673;
const float l_679 = shared_memory_38[l_669];
const __nv_bfloat16 l_680 = __nv_bfloat16(l_679);
buffer_2[l_678] = l_680;
}
const uint32 l_681 = threadIdxGlobal + uint32(256);
const uint32 l_682 = l_681 / uint32(32);
const uint32 l_683 = l_655 + l_682;
const uint32 l_684 = l_681 % uint32(32);
const uint32 l_685 = l_656 + l_684;
const bool l_686 = l_683 < l_3;
const bool l_687 = l_685 < l_4;
const bool l_688 = l_686 && l_687;
if (l_688) {
const uint32 l_689 = l_683 * l_4;
const uint32 l_690 = l_689 + l_685;
const float l_691 = shared_memory_38[l_681];
const __nv_bfloat16 l_692 = __nv_bfloat16(l_691);
buffer_2[l_690] = l_692;
}
const uint32 l_693 = threadIdxGlobal + uint32(384);
const uint32 l_694 = l_693 / uint32(32);
const uint32 l_695 = l_655 + l_694;
const uint32 l_696 = l_693 % uint32(32);
const uint32 l_697 = l_656 + l_696;
const bool l_698 = l_695 < l_3;
const bool l_699 = l_697 < l_4;
const bool l_700 = l_698 && l_699;
if (l_700) {
const uint32 l_701 = l_695 * l_4;
const uint32 l_702 = l_701 + l_697;
const float l_703 = shared_memory_38[l_693];
const __nv_bfloat16 l_704 = __nv_bfloat16(l_703);
buffer_2[l_702] = l_704;
}
__syncthreads();

const bool l_705 = l_0 == uint32(3);
if (l_705) {
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(0), frag_acc_50, uint32(32), nvcuda::wmma::mem_row_major);
nvcuda::wmma::store_matrix_sync(shared_memory_38 + uint32(16), frag_acc_51, uint32(32), nvcuda::wmma::mem_row_major);
}
const uint32 l_706 = l_6 + uint32(32);
const uint32 l_707 = l_706 + uint32(16);
const uint32 l_708 = l_7 + uint32(32);
__syncthreads();

const uint32 l_709 = threadIdxGlobal + uint32(0);
const uint32 l_710 = l_709 / uint32(32);
const uint32 l_711 = l_707 + l_710;
const uint32 l_712 = l_709 % uint32(32);
const uint32 l_713 = l_708 + l_712;
const bool l_714 = l_711 < l_3;
const bool l_715 = l_713 < l_4;
const bool l_716 = l_714 && l_715;
if (l_716) {
const uint32 l_717 = l_711 * l_4;
const uint32 l_718 = l_717 + l_713;
const float l_719 = shared_memory_38[l_709];
const __nv_bfloat16 l_720 = __nv_bfloat16(l_719);
buffer_2[l_718] = l_720;
}
const uint32 l_721 = threadIdxGlobal + uint32(128);
const uint32 l_722 = l_721 / uint32(32);
const uint32 l_723 = l_707 + l_722;
const uint32 l_724 = l_721 % uint32(32);
const uint32 l_725 = l_708 + l_724;
const bool l_726 = l_723 < l_3;
const bool l_727 = l_725 < l_4;
const bool l_728 = l_726 && l_727;
if (l_728) {
const uint32 l_729 = l_723 * l_4;
const uint32 l_730 = l_729 + l_725;
const float l_731 = shared_memory_38[l_721];
const __nv_bfloat16 l_732 = __nv_bfloat16(l_731);
buffer_2[l_730] = l_732;
}
const uint32 l_733 = threadIdxGlobal + uint32(256);
const uint32 l_734 = l_733 / uint32(32);
const uint32 l_735 = l_707 + l_734;
const uint32 l_736 = l_733 % uint32(32);
const uint32 l_737 = l_708 + l_736;
const bool l_738 = l_735 < l_3;
const bool l_739 = l_737 < l_4;
const bool l_740 = l_738 && l_739;
if (l_740) {
const uint32 l_741 = l_735 * l_4;
const uint32 l_742 = l_741 + l_737;
const float l_743 = shared_memory_38[l_733];
const __nv_bfloat16 l_744 = __nv_bfloat16(l_743);
buffer_2[l_742] = l_744;
}
const uint32 l_745 = threadIdxGlobal + uint32(384);
const uint32 l_746 = l_745 / uint32(32);
const uint32 l_747 = l_707 + l_746;
const uint32 l_748 = l_745 % uint32(32);
const uint32 l_749 = l_708 + l_748;
const bool l_750 = l_747 < l_3;
const bool l_751 = l_749 < l_4;
const bool l_752 = l_750 && l_751;
if (l_752) {
const uint32 l_753 = l_747 * l_4;
const uint32 l_754 = l_753 + l_749;
const float l_755 = shared_memory_38[l_745];
const __nv_bfloat16 l_756 = __nv_bfloat16(l_755);
buffer_2[l_754] = l_756;
}
__syncthreads();


}
