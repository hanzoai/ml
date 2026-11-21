#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
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

struct __align__(8) __hip_bfloat16_4 {
  __hip_bfloat16 i_0;
  __hip_bfloat16 i_1;
  __hip_bfloat16 i_2;
  __hip_bfloat16 i_3;
};

struct info_st {
  uint32 static_meta[8];
};

typedef __bf16 bhalf8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bhalf16_t __attribute__((ext_vector_type(16)));
typedef _Float16 half8_t __attribute__((ext_vector_type(8)));
typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float float8_t __attribute__((ext_vector_type(8)));

// Fill the fragment.
__device__ void wmma_fill_float_c_16x16x16_(float8_t &frag, float value) {
#pragma unroll
  for (uint i = 0; i < 8; ++i) {
    frag[i] = value;
  }
}

// Load the fragment.
__device__ void wmma_load___half_b_16x16x16_col_(half16_t &frag,
                                                 const __half *value_ptr,
                                                 const uint stride) {
  uint wmmaLane = uint(threadIdx.x % 16);

#pragma unroll
  for (uint i = 0; i < 16; ++i) {
    const uint index = i + wmmaLane * stride;
    frag[i * 1] = value_ptr[index];
  }
}

// Load the fragment.
__device__ void wmma_load___half_a_16x16x16_row_(half16_t &frag,
                                                 const __half *value_ptr,
                                                 const uint stride) {
  uint wmmaLane = uint(threadIdx.x % 16);

#pragma unroll
  for (uint i = 0; i < 16; ++i) {
    const uint index = i + wmmaLane * stride;
    frag[i * 1] = value_ptr[index];
  }
}

// Execute wmma.
__device__ void wmma_execute_16x16x16___half_float(const half16_t &frag_a,
                                                   const half16_t &frag_b,
                                                   const float8_t &frag_c,
                                                   float8_t &frag_d) {
  frag_d = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(frag_a, frag_b, frag_c);
}

// Store the fragment.
__device__ void wmma_store_float_c_16x16x16__row(float8_t &frag,
                                                 float *output_ptr,
                                                 uint stride) {
  uint wmmaLane = uint(threadIdx.x % 16);

#pragma unroll
  for (uint elemIdx = 0; elemIdx < uint(8); ++elemIdx) {
    const uint rowIdx = elemIdx * uint(2) + (threadIdx.x % uint(32)) / uint(16);
    output_ptr[wmmaLane + rowIdx * stride] = frag[elemIdx];
  }
}

extern "C" __global__ void __launch_bounds__(128)
    matmul_ptq1_0_gguf_f_(const uint32 *__restrict__ buffer_0,
                          const __hip_bfloat16_4 *__restrict__ buffer_1,
                          __hip_bfloat16 *buffer_2,
                          const uint32 *__restrict__ buffer_3,
                          const info_st *__restrict__ info_ptr) {
  uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x +
                           threadIdx.z * (blockDim.x * blockDim.y);
  extern __shared__ __align__(4) uchar dynamic_shared_mem[];
  const info_st &info = *info_ptr;
  const uint32 *dynamic_meta = reinterpret_cast<const uint32 *>(
      reinterpret_cast<const char *>(info_ptr) + sizeof(info_st));

  // Shared array size: 512, 2048 bytes
  float *shared_memory_38 = reinterpret_cast<float *>(&dynamic_shared_mem[0]);
  // Shared array size: 2560, 5120 bytes
  __half *shared_memory_41 =
      reinterpret_cast<__half *>(&dynamic_shared_mem[2048]);
  // Shared array size: 2560, 5120 bytes
  __half *shared_memory_42 =
      reinterpret_cast<__half *>(&dynamic_shared_mem[7168]);
  __half l_arr_39[16];

  __half l_arr_40[16];

  float8_t frag_acc_48;
  float8_t frag_acc_49;
  float8_t frag_acc_50;
  float8_t frag_acc_51;
  uint32 l_mut_16;
  uint32 l_mut_23;
  uint32 l_mut_53;
  uint32 l_mut_54;
  uint32 l_mut_55;
  uint32 l_mut_56;
  uint32 l_mut_57;
  __half l_mut_59;
  uint32 l_mut_144;
  uint32 l_mut_294;
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
  wmma_fill_float_c_16x16x16_(frag_acc_48, float(0.0));
  wmma_fill_float_c_16x16x16_(frag_acc_49, float(0.0));
  wmma_fill_float_c_16x16x16_(frag_acc_50, float(0.0));
  wmma_fill_float_c_16x16x16_(frag_acc_51, float(0.0));

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
      const __half_2 l_79 = reinterpret_cast<__half_2 const &>(l_78);
      const __half l_80 = l_79.i_0;
      l_mut_59 = l_80;
    }

    for (uint32 l_mut_81 = uint32(0); l_mut_81 < uint32(4); ++l_mut_81) {
      half16_t frag_b_625;
      half16_t frag_b_631;
      half16_t frag_a_637;
      half16_t frag_a_643;
      half16_t frag_b_649;
      half16_t frag_b_655;
      half16_t frag_a_661;
      half16_t frag_a_667;
      const uint32 l_82 = l_mut_52 * uint32(4);
      const uint32 l_83 = l_82 + l_mut_81;
      const uint32 l_84 = l_83 * uint32(8);
      const uint32 l_85 = uint32(2) * l_mut_81;
      const uint32 l_86 = l_85 + l_32;
      const uint32 l_87 = l_27 + l_84;
      const __hip_bfloat16_4 l_88 = buffer_1[l_87];
      const uint32 l_89 = l_87 + uint32(1);
      const __hip_bfloat16_4 l_90 = buffer_1[l_89];
      const __hip_bfloat16 l_91 = l_88.i_0;
      const float l_92 = float(l_91);
      const __half l_93 = __half(l_92);
      l_arr_39[uint32(0)] = l_93;
      const __hip_bfloat16 l_94 = l_90.i_0;
      const float l_95 = float(l_94);
      const __half l_96 = __half(l_95);
      l_arr_39[uint32(4)] = l_96;
      const __hip_bfloat16 l_97 = l_88.i_1;
      const float l_98 = float(l_97);
      const __half l_99 = __half(l_98);
      l_arr_39[uint32(1)] = l_99;
      const __hip_bfloat16 l_100 = l_90.i_1;
      const float l_101 = float(l_100);
      const __half l_102 = __half(l_101);
      l_arr_39[uint32(5)] = l_102;
      const __hip_bfloat16 l_103 = l_88.i_2;
      const float l_104 = float(l_103);
      const __half l_105 = __half(l_104);
      l_arr_39[uint32(2)] = l_105;
      const __hip_bfloat16 l_106 = l_90.i_2;
      const float l_107 = float(l_106);
      const __half l_108 = __half(l_107);
      l_arr_39[uint32(6)] = l_108;
      const __hip_bfloat16 l_109 = l_88.i_3;
      const float l_110 = float(l_109);
      const __half l_111 = __half(l_110);
      l_arr_39[uint32(3)] = l_111;
      const __hip_bfloat16 l_112 = l_90.i_3;
      const float l_113 = float(l_112);
      const __half l_114 = __half(l_113);
      l_arr_39[uint32(7)] = l_114;
      const uint32 l_115 = l_30 + l_84;
      const __hip_bfloat16_4 l_116 = buffer_1[l_115];
      const uint32 l_117 = l_115 + uint32(1);
      const __hip_bfloat16_4 l_118 = buffer_1[l_117];
      const __hip_bfloat16 l_119 = l_116.i_0;
      const float l_120 = float(l_119);
      const __half l_121 = __half(l_120);
      l_arr_39[uint32(8)] = l_121;
      const __hip_bfloat16 l_122 = l_118.i_0;
      const float l_123 = float(l_122);
      const __half l_124 = __half(l_123);
      l_arr_39[uint32(12)] = l_124;
      const __hip_bfloat16 l_125 = l_116.i_1;
      const float l_126 = float(l_125);
      const __half l_127 = __half(l_126);
      l_arr_39[uint32(9)] = l_127;
      const __hip_bfloat16 l_128 = l_118.i_1;
      const float l_129 = float(l_128);
      const __half l_130 = __half(l_129);
      l_arr_39[uint32(13)] = l_130;
      const __hip_bfloat16 l_131 = l_116.i_2;
      const float l_132 = float(l_131);
      const __half l_133 = __half(l_132);
      l_arr_39[uint32(10)] = l_133;
      const __hip_bfloat16 l_134 = l_118.i_2;
      const float l_135 = float(l_134);
      const __half l_136 = __half(l_135);
      l_arr_39[uint32(14)] = l_136;
      const __hip_bfloat16 l_137 = l_116.i_3;
      const float l_138 = float(l_137);
      const __half l_139 = __half(l_138);
      l_arr_39[uint32(11)] = l_139;
      const __hip_bfloat16 l_140 = l_118.i_3;
      const float l_141 = float(l_140);
      const __half l_142 = __half(l_141);
      l_arr_39[uint32(15)] = l_142;
      const bool l_143 = l_86 < uint32(5);
      if (l_143) {
        l_mut_144 = uint32(1);
        const bool l_145 = l_86 == uint32(1);
        if (l_145) {
          l_mut_144 = uint32(3);
        } else {
          const bool l_146 = l_86 == uint32(2);
          if (l_146) {
            l_mut_144 = uint32(9);
          } else {
            const bool l_147 = l_86 == uint32(3);
            if (l_147) {
              l_mut_144 = uint32(27);
            } else {
              const bool l_148 = l_86 == uint32(4);
              if (l_148) {
                l_mut_144 = uint32(81);
              }
            }
          }
        }
        const uint32 l_149 = l_mut_23 >> uint32(0);
        const uint32 l_150 = l_149 & uint32(255);
        const uint32 l_151 = l_150 * l_mut_144;
        const uint32 l_152 = l_151 & uint32(255);
        const uint32 l_153 = l_152 * uint32(3);
        const uint32 l_154 = l_153 >> uint32(8);
        const __half l_155 = __half(l_154);
        const __half l_156 = l_155 * l_mut_59;
        const __half l_157 = l_156 - l_mut_59;
        l_arr_40[uint32(0)] = l_157;
        const uint32 l_158 = l_mut_16 >> uint32(0);
        const uint32 l_159 = l_158 & uint32(255);
        const uint32 l_160 = l_159 * l_mut_144;
        const uint32 l_161 = l_160 & uint32(255);
        const uint32 l_162 = l_161 * uint32(3);
        const uint32 l_163 = l_162 >> uint32(8);
        const __half l_164 = __half(l_163);
        const __half l_165 = l_164 * l_mut_59;
        const __half l_166 = l_165 - l_mut_59;
        l_arr_40[uint32(4)] = l_166;
        const uint32 l_167 = l_mut_53 >> uint32(0);
        const uint32 l_168 = l_167 & uint32(255);
        const uint32 l_169 = l_168 * l_mut_144;
        const uint32 l_170 = l_169 & uint32(255);
        const uint32 l_171 = l_170 * uint32(3);
        const uint32 l_172 = l_171 >> uint32(8);
        const __half l_173 = __half(l_172);
        const __half l_174 = l_173 * l_mut_59;
        const __half l_175 = l_174 - l_mut_59;
        l_arr_40[uint32(8)] = l_175;
        const uint32 l_176 = l_mut_54 >> uint32(0);
        const uint32 l_177 = l_176 & uint32(255);
        const uint32 l_178 = l_177 * l_mut_144;
        const uint32 l_179 = l_178 & uint32(255);
        const uint32 l_180 = l_179 * uint32(3);
        const uint32 l_181 = l_180 >> uint32(8);
        const __half l_182 = __half(l_181);
        const __half l_183 = l_182 * l_mut_59;
        const __half l_184 = l_183 - l_mut_59;
        l_arr_40[uint32(12)] = l_184;
        const uint32 l_185 = l_mut_23 >> uint32(8);
        const uint32 l_186 = l_185 & uint32(255);
        const uint32 l_187 = l_186 * l_mut_144;
        const uint32 l_188 = l_187 & uint32(255);
        const uint32 l_189 = l_188 * uint32(3);
        const uint32 l_190 = l_189 >> uint32(8);
        const __half l_191 = __half(l_190);
        const __half l_192 = l_191 * l_mut_59;
        const __half l_193 = l_192 - l_mut_59;
        l_arr_40[uint32(1)] = l_193;
        const uint32 l_194 = l_mut_16 >> uint32(8);
        const uint32 l_195 = l_194 & uint32(255);
        const uint32 l_196 = l_195 * l_mut_144;
        const uint32 l_197 = l_196 & uint32(255);
        const uint32 l_198 = l_197 * uint32(3);
        const uint32 l_199 = l_198 >> uint32(8);
        const __half l_200 = __half(l_199);
        const __half l_201 = l_200 * l_mut_59;
        const __half l_202 = l_201 - l_mut_59;
        l_arr_40[uint32(5)] = l_202;
        const uint32 l_203 = l_mut_53 >> uint32(8);
        const uint32 l_204 = l_203 & uint32(255);
        const uint32 l_205 = l_204 * l_mut_144;
        const uint32 l_206 = l_205 & uint32(255);
        const uint32 l_207 = l_206 * uint32(3);
        const uint32 l_208 = l_207 >> uint32(8);
        const __half l_209 = __half(l_208);
        const __half l_210 = l_209 * l_mut_59;
        const __half l_211 = l_210 - l_mut_59;
        l_arr_40[uint32(9)] = l_211;
        const uint32 l_212 = l_mut_54 >> uint32(8);
        const uint32 l_213 = l_212 & uint32(255);
        const uint32 l_214 = l_213 * l_mut_144;
        const uint32 l_215 = l_214 & uint32(255);
        const uint32 l_216 = l_215 * uint32(3);
        const uint32 l_217 = l_216 >> uint32(8);
        const __half l_218 = __half(l_217);
        const __half l_219 = l_218 * l_mut_59;
        const __half l_220 = l_219 - l_mut_59;
        l_arr_40[uint32(13)] = l_220;
        const uint32 l_221 = l_mut_23 >> uint32(16);
        const uint32 l_222 = l_221 & uint32(255);
        const uint32 l_223 = l_222 * l_mut_144;
        const uint32 l_224 = l_223 & uint32(255);
        const uint32 l_225 = l_224 * uint32(3);
        const uint32 l_226 = l_225 >> uint32(8);
        const __half l_227 = __half(l_226);
        const __half l_228 = l_227 * l_mut_59;
        const __half l_229 = l_228 - l_mut_59;
        l_arr_40[uint32(2)] = l_229;
        const uint32 l_230 = l_mut_16 >> uint32(16);
        const uint32 l_231 = l_230 & uint32(255);
        const uint32 l_232 = l_231 * l_mut_144;
        const uint32 l_233 = l_232 & uint32(255);
        const uint32 l_234 = l_233 * uint32(3);
        const uint32 l_235 = l_234 >> uint32(8);
        const __half l_236 = __half(l_235);
        const __half l_237 = l_236 * l_mut_59;
        const __half l_238 = l_237 - l_mut_59;
        l_arr_40[uint32(6)] = l_238;
        const uint32 l_239 = l_mut_53 >> uint32(16);
        const uint32 l_240 = l_239 & uint32(255);
        const uint32 l_241 = l_240 * l_mut_144;
        const uint32 l_242 = l_241 & uint32(255);
        const uint32 l_243 = l_242 * uint32(3);
        const uint32 l_244 = l_243 >> uint32(8);
        const __half l_245 = __half(l_244);
        const __half l_246 = l_245 * l_mut_59;
        const __half l_247 = l_246 - l_mut_59;
        l_arr_40[uint32(10)] = l_247;
        const uint32 l_248 = l_mut_54 >> uint32(16);
        const uint32 l_249 = l_248 & uint32(255);
        const uint32 l_250 = l_249 * l_mut_144;
        const uint32 l_251 = l_250 & uint32(255);
        const uint32 l_252 = l_251 * uint32(3);
        const uint32 l_253 = l_252 >> uint32(8);
        const __half l_254 = __half(l_253);
        const __half l_255 = l_254 * l_mut_59;
        const __half l_256 = l_255 - l_mut_59;
        l_arr_40[uint32(14)] = l_256;
        const uint32 l_257 = l_mut_23 >> uint32(24);
        const uint32 l_258 = l_257 & uint32(255);
        const uint32 l_259 = l_258 * l_mut_144;
        const uint32 l_260 = l_259 & uint32(255);
        const uint32 l_261 = l_260 * uint32(3);
        const uint32 l_262 = l_261 >> uint32(8);
        const __half l_263 = __half(l_262);
        const __half l_264 = l_263 * l_mut_59;
        const __half l_265 = l_264 - l_mut_59;
        l_arr_40[uint32(3)] = l_265;
        const uint32 l_266 = l_mut_16 >> uint32(24);
        const uint32 l_267 = l_266 & uint32(255);
        const uint32 l_268 = l_267 * l_mut_144;
        const uint32 l_269 = l_268 & uint32(255);
        const uint32 l_270 = l_269 * uint32(3);
        const uint32 l_271 = l_270 >> uint32(8);
        const __half l_272 = __half(l_271);
        const __half l_273 = l_272 * l_mut_59;
        const __half l_274 = l_273 - l_mut_59;
        l_arr_40[uint32(7)] = l_274;
        const uint32 l_275 = l_mut_53 >> uint32(24);
        const uint32 l_276 = l_275 & uint32(255);
        const uint32 l_277 = l_276 * l_mut_144;
        const uint32 l_278 = l_277 & uint32(255);
        const uint32 l_279 = l_278 * uint32(3);
        const uint32 l_280 = l_279 >> uint32(8);
        const __half l_281 = __half(l_280);
        const __half l_282 = l_281 * l_mut_59;
        const __half l_283 = l_282 - l_mut_59;
        l_arr_40[uint32(11)] = l_283;
        const uint32 l_284 = l_mut_54 >> uint32(24);
        const uint32 l_285 = l_284 & uint32(255);
        const uint32 l_286 = l_285 * l_mut_144;
        const uint32 l_287 = l_286 & uint32(255);
        const uint32 l_288 = l_287 * uint32(3);
        const uint32 l_289 = l_288 >> uint32(8);
        const __half l_290 = __half(l_289);
        const __half l_291 = l_290 * l_mut_59;
        const __half l_292 = l_291 - l_mut_59;
        l_arr_40[uint32(15)] = l_292;
      } else {
        const bool l_293 = l_86 < uint32(7);
        if (l_293) {
          l_mut_144 = uint32(1);
          l_mut_294 = uint32(3);
          const bool l_295 = l_86 == uint32(6);
          if (l_295) {
            l_mut_144 = uint32(9);
            l_mut_294 = uint32(27);
          }
          const uint32 l_296 = l_mut_55 >> uint32(0);
          const uint32 l_297 = l_296 & uint32(255);
          const uint32 l_298 = l_mut_56 >> uint32(0);
          const uint32 l_299 = l_298 & uint32(255);
          const uint32 l_300 = l_297 * l_mut_144;
          const uint32 l_301 = l_300 & uint32(255);
          const uint32 l_302 = l_301 * uint32(3);
          const uint32 l_303 = l_302 >> uint32(8);
          const __half l_304 = __half(l_303);
          const __half l_305 = l_304 * l_mut_59;
          const __half l_306 = l_305 - l_mut_59;
          l_arr_40[uint32(0)] = l_306;
          const uint32 l_307 = l_299 * l_mut_144;
          const uint32 l_308 = l_307 & uint32(255);
          const uint32 l_309 = l_308 * uint32(3);
          const uint32 l_310 = l_309 >> uint32(8);
          const __half l_311 = __half(l_310);
          const __half l_312 = l_311 * l_mut_59;
          const __half l_313 = l_312 - l_mut_59;
          l_arr_40[uint32(4)] = l_313;
          const uint32 l_314 = l_297 * l_mut_294;
          const uint32 l_315 = l_314 & uint32(255);
          const uint32 l_316 = l_315 * uint32(3);
          const uint32 l_317 = l_316 >> uint32(8);
          const __half l_318 = __half(l_317);
          const __half l_319 = l_318 * l_mut_59;
          const __half l_320 = l_319 - l_mut_59;
          l_arr_40[uint32(8)] = l_320;
          const uint32 l_321 = l_299 * l_mut_294;
          const uint32 l_322 = l_321 & uint32(255);
          const uint32 l_323 = l_322 * uint32(3);
          const uint32 l_324 = l_323 >> uint32(8);
          const __half l_325 = __half(l_324);
          const __half l_326 = l_325 * l_mut_59;
          const __half l_327 = l_326 - l_mut_59;
          l_arr_40[uint32(12)] = l_327;
          const uint32 l_328 = l_mut_55 >> uint32(8);
          const uint32 l_329 = l_328 & uint32(255);
          const uint32 l_330 = l_mut_56 >> uint32(8);
          const uint32 l_331 = l_330 & uint32(255);
          const uint32 l_332 = l_329 * l_mut_144;
          const uint32 l_333 = l_332 & uint32(255);
          const uint32 l_334 = l_333 * uint32(3);
          const uint32 l_335 = l_334 >> uint32(8);
          const __half l_336 = __half(l_335);
          const __half l_337 = l_336 * l_mut_59;
          const __half l_338 = l_337 - l_mut_59;
          l_arr_40[uint32(1)] = l_338;
          const uint32 l_339 = l_331 * l_mut_144;
          const uint32 l_340 = l_339 & uint32(255);
          const uint32 l_341 = l_340 * uint32(3);
          const uint32 l_342 = l_341 >> uint32(8);
          const __half l_343 = __half(l_342);
          const __half l_344 = l_343 * l_mut_59;
          const __half l_345 = l_344 - l_mut_59;
          l_arr_40[uint32(5)] = l_345;
          const uint32 l_346 = l_329 * l_mut_294;
          const uint32 l_347 = l_346 & uint32(255);
          const uint32 l_348 = l_347 * uint32(3);
          const uint32 l_349 = l_348 >> uint32(8);
          const __half l_350 = __half(l_349);
          const __half l_351 = l_350 * l_mut_59;
          const __half l_352 = l_351 - l_mut_59;
          l_arr_40[uint32(9)] = l_352;
          const uint32 l_353 = l_331 * l_mut_294;
          const uint32 l_354 = l_353 & uint32(255);
          const uint32 l_355 = l_354 * uint32(3);
          const uint32 l_356 = l_355 >> uint32(8);
          const __half l_357 = __half(l_356);
          const __half l_358 = l_357 * l_mut_59;
          const __half l_359 = l_358 - l_mut_59;
          l_arr_40[uint32(13)] = l_359;
          const uint32 l_360 = l_mut_55 >> uint32(16);
          const uint32 l_361 = l_360 & uint32(255);
          const uint32 l_362 = l_mut_56 >> uint32(16);
          const uint32 l_363 = l_362 & uint32(255);
          const uint32 l_364 = l_361 * l_mut_144;
          const uint32 l_365 = l_364 & uint32(255);
          const uint32 l_366 = l_365 * uint32(3);
          const uint32 l_367 = l_366 >> uint32(8);
          const __half l_368 = __half(l_367);
          const __half l_369 = l_368 * l_mut_59;
          const __half l_370 = l_369 - l_mut_59;
          l_arr_40[uint32(2)] = l_370;
          const uint32 l_371 = l_363 * l_mut_144;
          const uint32 l_372 = l_371 & uint32(255);
          const uint32 l_373 = l_372 * uint32(3);
          const uint32 l_374 = l_373 >> uint32(8);
          const __half l_375 = __half(l_374);
          const __half l_376 = l_375 * l_mut_59;
          const __half l_377 = l_376 - l_mut_59;
          l_arr_40[uint32(6)] = l_377;
          const uint32 l_378 = l_361 * l_mut_294;
          const uint32 l_379 = l_378 & uint32(255);
          const uint32 l_380 = l_379 * uint32(3);
          const uint32 l_381 = l_380 >> uint32(8);
          const __half l_382 = __half(l_381);
          const __half l_383 = l_382 * l_mut_59;
          const __half l_384 = l_383 - l_mut_59;
          l_arr_40[uint32(10)] = l_384;
          const uint32 l_385 = l_363 * l_mut_294;
          const uint32 l_386 = l_385 & uint32(255);
          const uint32 l_387 = l_386 * uint32(3);
          const uint32 l_388 = l_387 >> uint32(8);
          const __half l_389 = __half(l_388);
          const __half l_390 = l_389 * l_mut_59;
          const __half l_391 = l_390 - l_mut_59;
          l_arr_40[uint32(14)] = l_391;
          const uint32 l_392 = l_mut_55 >> uint32(24);
          const uint32 l_393 = l_392 & uint32(255);
          const uint32 l_394 = l_mut_56 >> uint32(24);
          const uint32 l_395 = l_394 & uint32(255);
          const uint32 l_396 = l_393 * l_mut_144;
          const uint32 l_397 = l_396 & uint32(255);
          const uint32 l_398 = l_397 * uint32(3);
          const uint32 l_399 = l_398 >> uint32(8);
          const __half l_400 = __half(l_399);
          const __half l_401 = l_400 * l_mut_59;
          const __half l_402 = l_401 - l_mut_59;
          l_arr_40[uint32(3)] = l_402;
          const uint32 l_403 = l_395 * l_mut_144;
          const uint32 l_404 = l_403 & uint32(255);
          const uint32 l_405 = l_404 * uint32(3);
          const uint32 l_406 = l_405 >> uint32(8);
          const __half l_407 = __half(l_406);
          const __half l_408 = l_407 * l_mut_59;
          const __half l_409 = l_408 - l_mut_59;
          l_arr_40[uint32(7)] = l_409;
          const uint32 l_410 = l_393 * l_mut_294;
          const uint32 l_411 = l_410 & uint32(255);
          const uint32 l_412 = l_411 * uint32(3);
          const uint32 l_413 = l_412 >> uint32(8);
          const __half l_414 = __half(l_413);
          const __half l_415 = l_414 * l_mut_59;
          const __half l_416 = l_415 - l_mut_59;
          l_arr_40[uint32(11)] = l_416;
          const uint32 l_417 = l_395 * l_mut_294;
          const uint32 l_418 = l_417 & uint32(255);
          const uint32 l_419 = l_418 * uint32(3);
          const uint32 l_420 = l_419 >> uint32(8);
          const __half l_421 = __half(l_420);
          const __half l_422 = l_421 * l_mut_59;
          const __half l_423 = l_422 - l_mut_59;
          l_arr_40[uint32(15)] = l_423;
        } else {
          const uint32 l_424 = l_mut_57 & uint32(255);
          const uint32 l_425 = l_mut_57 >> uint32(8);
          const uint32 l_426 = l_425 & uint32(255);
          l_mut_294 = uint32(1);
          const uint32 l_427 = l_mut_55 >> uint32(0);
          const uint32 l_428 = l_427 & uint32(255);
          const uint32 l_429 = l_428 * uint32(81);
          const uint32 l_430 = l_429 & uint32(255);
          const uint32 l_431 = l_430 * uint32(3);
          const uint32 l_432 = l_431 >> uint32(8);
          const __half l_433 = __half(l_432);
          const __half l_434 = l_433 * l_mut_59;
          const __half l_435 = l_434 - l_mut_59;
          l_arr_40[uint32(0)] = l_435;
          const uint32 l_436 = l_mut_56 >> uint32(0);
          const uint32 l_437 = l_436 & uint32(255);
          const uint32 l_438 = l_437 * uint32(81);
          const uint32 l_439 = l_438 & uint32(255);
          const uint32 l_440 = l_439 * uint32(3);
          const uint32 l_441 = l_440 >> uint32(8);
          const __half l_442 = __half(l_441);
          const __half l_443 = l_442 * l_mut_59;
          const __half l_444 = l_443 - l_mut_59;
          l_arr_40[uint32(4)] = l_444;
          const uint32 l_445 = l_424 * l_mut_294;
          const uint32 l_446 = l_445 & uint32(255);
          const uint32 l_447 = l_446 * uint32(3);
          const uint32 l_448 = l_447 >> uint32(8);
          const __half l_449 = __half(l_448);
          const __half l_450 = l_449 * l_mut_59;
          const __half l_451 = l_450 - l_mut_59;
          l_arr_40[uint32(8)] = l_451;
          const uint32 l_452 = l_426 * l_mut_294;
          const uint32 l_453 = l_452 & uint32(255);
          const uint32 l_454 = l_453 * uint32(3);
          const uint32 l_455 = l_454 >> uint32(8);
          const __half l_456 = __half(l_455);
          const __half l_457 = l_456 * l_mut_59;
          const __half l_458 = l_457 - l_mut_59;
          l_arr_40[uint32(9)] = l_458;
          l_mut_294 = l_mut_294 * uint32(3);
          const uint32 l_459 = l_mut_55 >> uint32(8);
          const uint32 l_460 = l_459 & uint32(255);
          const uint32 l_461 = l_460 * uint32(81);
          const uint32 l_462 = l_461 & uint32(255);
          const uint32 l_463 = l_462 * uint32(3);
          const uint32 l_464 = l_463 >> uint32(8);
          const __half l_465 = __half(l_464);
          const __half l_466 = l_465 * l_mut_59;
          const __half l_467 = l_466 - l_mut_59;
          l_arr_40[uint32(1)] = l_467;
          const uint32 l_468 = l_mut_56 >> uint32(8);
          const uint32 l_469 = l_468 & uint32(255);
          const uint32 l_470 = l_469 * uint32(81);
          const uint32 l_471 = l_470 & uint32(255);
          const uint32 l_472 = l_471 * uint32(3);
          const uint32 l_473 = l_472 >> uint32(8);
          const __half l_474 = __half(l_473);
          const __half l_475 = l_474 * l_mut_59;
          const __half l_476 = l_475 - l_mut_59;
          l_arr_40[uint32(5)] = l_476;
          const uint32 l_477 = l_424 * l_mut_294;
          const uint32 l_478 = l_477 & uint32(255);
          const uint32 l_479 = l_478 * uint32(3);
          const uint32 l_480 = l_479 >> uint32(8);
          const __half l_481 = __half(l_480);
          const __half l_482 = l_481 * l_mut_59;
          const __half l_483 = l_482 - l_mut_59;
          l_arr_40[uint32(10)] = l_483;
          const uint32 l_484 = l_426 * l_mut_294;
          const uint32 l_485 = l_484 & uint32(255);
          const uint32 l_486 = l_485 * uint32(3);
          const uint32 l_487 = l_486 >> uint32(8);
          const __half l_488 = __half(l_487);
          const __half l_489 = l_488 * l_mut_59;
          const __half l_490 = l_489 - l_mut_59;
          l_arr_40[uint32(11)] = l_490;
          l_mut_294 = l_mut_294 * uint32(3);
          const uint32 l_491 = l_mut_55 >> uint32(16);
          const uint32 l_492 = l_491 & uint32(255);
          const uint32 l_493 = l_492 * uint32(81);
          const uint32 l_494 = l_493 & uint32(255);
          const uint32 l_495 = l_494 * uint32(3);
          const uint32 l_496 = l_495 >> uint32(8);
          const __half l_497 = __half(l_496);
          const __half l_498 = l_497 * l_mut_59;
          const __half l_499 = l_498 - l_mut_59;
          l_arr_40[uint32(2)] = l_499;
          const uint32 l_500 = l_mut_56 >> uint32(16);
          const uint32 l_501 = l_500 & uint32(255);
          const uint32 l_502 = l_501 * uint32(81);
          const uint32 l_503 = l_502 & uint32(255);
          const uint32 l_504 = l_503 * uint32(3);
          const uint32 l_505 = l_504 >> uint32(8);
          const __half l_506 = __half(l_505);
          const __half l_507 = l_506 * l_mut_59;
          const __half l_508 = l_507 - l_mut_59;
          l_arr_40[uint32(6)] = l_508;
          const uint32 l_509 = l_424 * l_mut_294;
          const uint32 l_510 = l_509 & uint32(255);
          const uint32 l_511 = l_510 * uint32(3);
          const uint32 l_512 = l_511 >> uint32(8);
          const __half l_513 = __half(l_512);
          const __half l_514 = l_513 * l_mut_59;
          const __half l_515 = l_514 - l_mut_59;
          l_arr_40[uint32(12)] = l_515;
          const uint32 l_516 = l_426 * l_mut_294;
          const uint32 l_517 = l_516 & uint32(255);
          const uint32 l_518 = l_517 * uint32(3);
          const uint32 l_519 = l_518 >> uint32(8);
          const __half l_520 = __half(l_519);
          const __half l_521 = l_520 * l_mut_59;
          const __half l_522 = l_521 - l_mut_59;
          l_arr_40[uint32(13)] = l_522;
          l_mut_294 = l_mut_294 * uint32(3);
          const uint32 l_523 = l_mut_55 >> uint32(24);
          const uint32 l_524 = l_523 & uint32(255);
          const uint32 l_525 = l_524 * uint32(81);
          const uint32 l_526 = l_525 & uint32(255);
          const uint32 l_527 = l_526 * uint32(3);
          const uint32 l_528 = l_527 >> uint32(8);
          const __half l_529 = __half(l_528);
          const __half l_530 = l_529 * l_mut_59;
          const __half l_531 = l_530 - l_mut_59;
          l_arr_40[uint32(3)] = l_531;
          const uint32 l_532 = l_mut_56 >> uint32(24);
          const uint32 l_533 = l_532 & uint32(255);
          const uint32 l_534 = l_533 * uint32(81);
          const uint32 l_535 = l_534 & uint32(255);
          const uint32 l_536 = l_535 * uint32(3);
          const uint32 l_537 = l_536 >> uint32(8);
          const __half l_538 = __half(l_537);
          const __half l_539 = l_538 * l_mut_59;
          const __half l_540 = l_539 - l_mut_59;
          l_arr_40[uint32(7)] = l_540;
          const uint32 l_541 = l_424 * l_mut_294;
          const uint32 l_542 = l_541 & uint32(255);
          const uint32 l_543 = l_542 * uint32(3);
          const uint32 l_544 = l_543 >> uint32(8);
          const __half l_545 = __half(l_544);
          const __half l_546 = l_545 * l_mut_59;
          const __half l_547 = l_546 - l_mut_59;
          l_arr_40[uint32(14)] = l_547;
          const uint32 l_548 = l_426 * l_mut_294;
          const uint32 l_549 = l_548 & uint32(255);
          const uint32 l_550 = l_549 * uint32(3);
          const uint32 l_551 = l_550 >> uint32(8);
          const __half l_552 = __half(l_551);
          const __half l_553 = l_552 * l_mut_59;
          const __half l_554 = l_553 - l_mut_59;
          l_arr_40[uint32(15)] = l_554;
          l_mut_294 = l_mut_294 * uint32(3);
        }
      }
      __syncthreads();

      const uint32 l_555 = l_44 + uint32(0);
      const __half l_556 = l_arr_39[uint32(0)];
      shared_memory_41[l_555] = l_556;
      const uint32 l_557 = l_44 + uint32(1);
      const __half l_558 = l_arr_39[uint32(1)];
      shared_memory_41[l_557] = l_558;
      const uint32 l_559 = l_44 + uint32(2);
      const __half l_560 = l_arr_39[uint32(2)];
      shared_memory_41[l_559] = l_560;
      const uint32 l_561 = l_44 + uint32(3);
      const __half l_562 = l_arr_39[uint32(3)];
      shared_memory_41[l_561] = l_562;
      const uint32 l_563 = l_44 + uint32(4);
      const __half l_564 = l_arr_39[uint32(4)];
      shared_memory_41[l_563] = l_564;
      const uint32 l_565 = l_44 + uint32(5);
      const __half l_566 = l_arr_39[uint32(5)];
      shared_memory_41[l_565] = l_566;
      const uint32 l_567 = l_44 + uint32(6);
      const __half l_568 = l_arr_39[uint32(6)];
      shared_memory_41[l_567] = l_568;
      const uint32 l_569 = l_44 + uint32(7);
      const __half l_570 = l_arr_39[uint32(7)];
      shared_memory_41[l_569] = l_570;
      const uint32 l_571 = l_44 + uint32(1280);
      const uint32 l_572 = l_571 + uint32(0);
      const __half l_573 = l_arr_39[uint32(8)];
      shared_memory_41[l_572] = l_573;
      const uint32 l_574 = l_571 + uint32(1);
      const __half l_575 = l_arr_39[uint32(9)];
      shared_memory_41[l_574] = l_575;
      const uint32 l_576 = l_571 + uint32(2);
      const __half l_577 = l_arr_39[uint32(10)];
      shared_memory_41[l_576] = l_577;
      const uint32 l_578 = l_571 + uint32(3);
      const __half l_579 = l_arr_39[uint32(11)];
      shared_memory_41[l_578] = l_579;
      const uint32 l_580 = l_571 + uint32(4);
      const __half l_581 = l_arr_39[uint32(12)];
      shared_memory_41[l_580] = l_581;
      const uint32 l_582 = l_571 + uint32(5);
      const __half l_583 = l_arr_39[uint32(13)];
      shared_memory_41[l_582] = l_583;
      const uint32 l_584 = l_571 + uint32(6);
      const __half l_585 = l_arr_39[uint32(14)];
      shared_memory_41[l_584] = l_585;
      const uint32 l_586 = l_571 + uint32(7);
      const __half l_587 = l_arr_39[uint32(15)];
      shared_memory_41[l_586] = l_587;
      const uint32 l_588 = l_47 + uint32(0);
      const __half l_589 = l_arr_40[uint32(0)];
      shared_memory_42[l_588] = l_589;
      const uint32 l_590 = l_47 + uint32(1);
      const __half l_591 = l_arr_40[uint32(1)];
      shared_memory_42[l_590] = l_591;
      const uint32 l_592 = l_47 + uint32(2);
      const __half l_593 = l_arr_40[uint32(2)];
      shared_memory_42[l_592] = l_593;
      const uint32 l_594 = l_47 + uint32(3);
      const __half l_595 = l_arr_40[uint32(3)];
      shared_memory_42[l_594] = l_595;
      const uint32 l_596 = l_47 + uint32(4);
      const __half l_597 = l_arr_40[uint32(4)];
      shared_memory_42[l_596] = l_597;
      const uint32 l_598 = l_47 + uint32(5);
      const __half l_599 = l_arr_40[uint32(5)];
      shared_memory_42[l_598] = l_599;
      const uint32 l_600 = l_47 + uint32(6);
      const __half l_601 = l_arr_40[uint32(6)];
      shared_memory_42[l_600] = l_601;
      const uint32 l_602 = l_47 + uint32(7);
      const __half l_603 = l_arr_40[uint32(7)];
      shared_memory_42[l_602] = l_603;
      const uint32 l_604 = l_47 + uint32(8);
      const __half l_605 = l_arr_40[uint32(8)];
      shared_memory_42[l_604] = l_605;
      const uint32 l_606 = l_47 + uint32(9);
      const __half l_607 = l_arr_40[uint32(9)];
      shared_memory_42[l_606] = l_607;
      const uint32 l_608 = l_47 + uint32(10);
      const __half l_609 = l_arr_40[uint32(10)];
      shared_memory_42[l_608] = l_609;
      const uint32 l_610 = l_47 + uint32(11);
      const __half l_611 = l_arr_40[uint32(11)];
      shared_memory_42[l_610] = l_611;
      const uint32 l_612 = l_47 + uint32(12);
      const __half l_613 = l_arr_40[uint32(12)];
      shared_memory_42[l_612] = l_613;
      const uint32 l_614 = l_47 + uint32(13);
      const __half l_615 = l_arr_40[uint32(13)];
      shared_memory_42[l_614] = l_615;
      const uint32 l_616 = l_47 + uint32(14);
      const __half l_617 = l_arr_40[uint32(14)];
      shared_memory_42[l_616] = l_617;
      const uint32 l_618 = l_47 + uint32(15);
      const __half l_619 = l_arr_40[uint32(15)];
      shared_memory_42[l_618] = l_619;
      __syncthreads();

      const uint32 l_620 = l_2 * uint32(32);
      const uint32 l_621 = l_620 + uint32(0);
      const uint32 l_622 = l_621 * uint32(40);
      const uint32 l_623 = l_622 + uint32(0);
      const uint32 l_624 = uint32(2560) - l_623;
      __half *_tmp_0 = shared_memory_42 + l_623;
      wmma_load___half_b_16x16x16_col_(frag_b_625, _tmp_0, uint32(40));
      const uint32 l_626 = l_2 * uint32(32);
      const uint32 l_627 = l_626 + uint32(16);
      const uint32 l_628 = l_627 * uint32(40);
      const uint32 l_629 = l_628 + uint32(0);
      const uint32 l_630 = uint32(2560) - l_629;
      __half *_tmp_1 = shared_memory_42 + l_629;
      wmma_load___half_b_16x16x16_col_(frag_b_631, _tmp_1, uint32(40));
      const uint32 l_632 = l_1 * uint32(32);
      const uint32 l_633 = l_632 + uint32(0);
      const uint32 l_634 = l_633 * uint32(40);
      const uint32 l_635 = l_634 + uint32(0);
      const uint32 l_636 = uint32(2560) - l_635;
      __half *_tmp_2 = shared_memory_41 + l_635;
      wmma_load___half_a_16x16x16_row_(frag_a_637, _tmp_2, uint32(40));
      wmma_execute_16x16x16___half_float(frag_a_637, frag_b_625, frag_acc_48,
                                         frag_acc_48);
      wmma_execute_16x16x16___half_float(frag_a_637, frag_b_631, frag_acc_49,
                                         frag_acc_49);
      const uint32 l_638 = l_1 * uint32(32);
      const uint32 l_639 = l_638 + uint32(16);
      const uint32 l_640 = l_639 * uint32(40);
      const uint32 l_641 = l_640 + uint32(0);
      const uint32 l_642 = uint32(2560) - l_641;
      __half *_tmp_3 = shared_memory_41 + l_641;
      wmma_load___half_a_16x16x16_row_(frag_a_643, _tmp_3, uint32(40));
      wmma_execute_16x16x16___half_float(frag_a_643, frag_b_625, frag_acc_50,
                                         frag_acc_50);
      wmma_execute_16x16x16___half_float(frag_a_643, frag_b_631, frag_acc_51,
                                         frag_acc_51);
      const uint32 l_644 = l_2 * uint32(32);
      const uint32 l_645 = l_644 + uint32(0);
      const uint32 l_646 = l_645 * uint32(40);
      const uint32 l_647 = l_646 + uint32(16);
      const uint32 l_648 = uint32(2560) - l_647;
      __half *_tmp_4 = shared_memory_42 + l_647;
      wmma_load___half_b_16x16x16_col_(frag_b_649, _tmp_4, uint32(40));
      const uint32 l_650 = l_2 * uint32(32);
      const uint32 l_651 = l_650 + uint32(16);
      const uint32 l_652 = l_651 * uint32(40);
      const uint32 l_653 = l_652 + uint32(16);
      const uint32 l_654 = uint32(2560) - l_653;
      __half *_tmp_5 = shared_memory_42 + l_653;
      wmma_load___half_b_16x16x16_col_(frag_b_655, _tmp_5, uint32(40));
      const uint32 l_656 = l_1 * uint32(32);
      const uint32 l_657 = l_656 + uint32(0);
      const uint32 l_658 = l_657 * uint32(40);
      const uint32 l_659 = l_658 + uint32(16);
      const uint32 l_660 = uint32(2560) - l_659;
      __half *_tmp_6 = shared_memory_41 + l_659;
      wmma_load___half_a_16x16x16_row_(frag_a_661, _tmp_6, uint32(40));
      wmma_execute_16x16x16___half_float(frag_a_661, frag_b_649, frag_acc_48,
                                         frag_acc_48);
      wmma_execute_16x16x16___half_float(frag_a_661, frag_b_655, frag_acc_49,
                                         frag_acc_49);
      const uint32 l_662 = l_1 * uint32(32);
      const uint32 l_663 = l_662 + uint32(16);
      const uint32 l_664 = l_663 * uint32(40);
      const uint32 l_665 = l_664 + uint32(16);
      const uint32 l_666 = uint32(2560) - l_665;
      __half *_tmp_7 = shared_memory_41 + l_665;
      wmma_load___half_a_16x16x16_row_(frag_a_667, _tmp_7, uint32(40));
      wmma_execute_16x16x16___half_float(frag_a_667, frag_b_649, frag_acc_50,
                                         frag_acc_50);
      wmma_execute_16x16x16___half_float(frag_a_667, frag_b_655, frag_acc_51,
                                         frag_acc_51);
    }
  }
  const bool l_668 = l_0 == uint32(0);
  if (l_668) {
    float *_tmp_8 = shared_memory_38 + uint32(0);
    wmma_store_float_c_16x16x16__row(frag_acc_48, _tmp_8, uint32(32));
    float *_tmp_9 = shared_memory_38 + uint32(16);
    wmma_store_float_c_16x16x16__row(frag_acc_49, _tmp_9, uint32(32));
  }
  const uint32 l_669 = l_6 + uint32(0);
  const uint32 l_670 = l_669 + uint32(0);
  const uint32 l_671 = l_7 + uint32(0);
  __syncthreads();

  const uint32 l_672 = threadIdxGlobal + uint32(0);
  const uint32 l_673 = l_672 / uint32(32);
  const uint32 l_674 = l_670 + l_673;
  const uint32 l_675 = l_672 % uint32(32);
  const uint32 l_676 = l_671 + l_675;
  const bool l_677 = l_674 < l_3;
  const bool l_678 = l_676 < l_4;
  const bool l_679 = l_677 && l_678;
  if (l_679) {
    const uint32 l_680 = l_674 * l_4;
    const uint32 l_681 = l_680 + l_676;
    const float l_682 = shared_memory_38[l_672];
    const __hip_bfloat16 l_683 = __hip_bfloat16(l_682);
    buffer_2[l_681] = l_683;
  }
  const uint32 l_684 = threadIdxGlobal + uint32(128);
  const uint32 l_685 = l_684 / uint32(32);
  const uint32 l_686 = l_670 + l_685;
  const uint32 l_687 = l_684 % uint32(32);
  const uint32 l_688 = l_671 + l_687;
  const bool l_689 = l_686 < l_3;
  const bool l_690 = l_688 < l_4;
  const bool l_691 = l_689 && l_690;
  if (l_691) {
    const uint32 l_692 = l_686 * l_4;
    const uint32 l_693 = l_692 + l_688;
    const float l_694 = shared_memory_38[l_684];
    const __hip_bfloat16 l_695 = __hip_bfloat16(l_694);
    buffer_2[l_693] = l_695;
  }
  const uint32 l_696 = threadIdxGlobal + uint32(256);
  const uint32 l_697 = l_696 / uint32(32);
  const uint32 l_698 = l_670 + l_697;
  const uint32 l_699 = l_696 % uint32(32);
  const uint32 l_700 = l_671 + l_699;
  const bool l_701 = l_698 < l_3;
  const bool l_702 = l_700 < l_4;
  const bool l_703 = l_701 && l_702;
  if (l_703) {
    const uint32 l_704 = l_698 * l_4;
    const uint32 l_705 = l_704 + l_700;
    const float l_706 = shared_memory_38[l_696];
    const __hip_bfloat16 l_707 = __hip_bfloat16(l_706);
    buffer_2[l_705] = l_707;
  }
  const uint32 l_708 = threadIdxGlobal + uint32(384);
  const uint32 l_709 = l_708 / uint32(32);
  const uint32 l_710 = l_670 + l_709;
  const uint32 l_711 = l_708 % uint32(32);
  const uint32 l_712 = l_671 + l_711;
  const bool l_713 = l_710 < l_3;
  const bool l_714 = l_712 < l_4;
  const bool l_715 = l_713 && l_714;
  if (l_715) {
    const uint32 l_716 = l_710 * l_4;
    const uint32 l_717 = l_716 + l_712;
    const float l_718 = shared_memory_38[l_708];
    const __hip_bfloat16 l_719 = __hip_bfloat16(l_718);
    buffer_2[l_717] = l_719;
  }
  __syncthreads();

  const bool l_720 = l_0 == uint32(0);
  if (l_720) {
    float *_tmp_10 = shared_memory_38 + uint32(0);
    wmma_store_float_c_16x16x16__row(frag_acc_50, _tmp_10, uint32(32));
    float *_tmp_11 = shared_memory_38 + uint32(16);
    wmma_store_float_c_16x16x16__row(frag_acc_51, _tmp_11, uint32(32));
  }
  const uint32 l_721 = l_6 + uint32(0);
  const uint32 l_722 = l_721 + uint32(16);
  const uint32 l_723 = l_7 + uint32(0);
  __syncthreads();

  const uint32 l_724 = threadIdxGlobal + uint32(0);
  const uint32 l_725 = l_724 / uint32(32);
  const uint32 l_726 = l_722 + l_725;
  const uint32 l_727 = l_724 % uint32(32);
  const uint32 l_728 = l_723 + l_727;
  const bool l_729 = l_726 < l_3;
  const bool l_730 = l_728 < l_4;
  const bool l_731 = l_729 && l_730;
  if (l_731) {
    const uint32 l_732 = l_726 * l_4;
    const uint32 l_733 = l_732 + l_728;
    const float l_734 = shared_memory_38[l_724];
    const __hip_bfloat16 l_735 = __hip_bfloat16(l_734);
    buffer_2[l_733] = l_735;
  }
  const uint32 l_736 = threadIdxGlobal + uint32(128);
  const uint32 l_737 = l_736 / uint32(32);
  const uint32 l_738 = l_722 + l_737;
  const uint32 l_739 = l_736 % uint32(32);
  const uint32 l_740 = l_723 + l_739;
  const bool l_741 = l_738 < l_3;
  const bool l_742 = l_740 < l_4;
  const bool l_743 = l_741 && l_742;
  if (l_743) {
    const uint32 l_744 = l_738 * l_4;
    const uint32 l_745 = l_744 + l_740;
    const float l_746 = shared_memory_38[l_736];
    const __hip_bfloat16 l_747 = __hip_bfloat16(l_746);
    buffer_2[l_745] = l_747;
  }
  const uint32 l_748 = threadIdxGlobal + uint32(256);
  const uint32 l_749 = l_748 / uint32(32);
  const uint32 l_750 = l_722 + l_749;
  const uint32 l_751 = l_748 % uint32(32);
  const uint32 l_752 = l_723 + l_751;
  const bool l_753 = l_750 < l_3;
  const bool l_754 = l_752 < l_4;
  const bool l_755 = l_753 && l_754;
  if (l_755) {
    const uint32 l_756 = l_750 * l_4;
    const uint32 l_757 = l_756 + l_752;
    const float l_758 = shared_memory_38[l_748];
    const __hip_bfloat16 l_759 = __hip_bfloat16(l_758);
    buffer_2[l_757] = l_759;
  }
  const uint32 l_760 = threadIdxGlobal + uint32(384);
  const uint32 l_761 = l_760 / uint32(32);
  const uint32 l_762 = l_722 + l_761;
  const uint32 l_763 = l_760 % uint32(32);
  const uint32 l_764 = l_723 + l_763;
  const bool l_765 = l_762 < l_3;
  const bool l_766 = l_764 < l_4;
  const bool l_767 = l_765 && l_766;
  if (l_767) {
    const uint32 l_768 = l_762 * l_4;
    const uint32 l_769 = l_768 + l_764;
    const float l_770 = shared_memory_38[l_760];
    const __hip_bfloat16 l_771 = __hip_bfloat16(l_770);
    buffer_2[l_769] = l_771;
  }
  __syncthreads();

  const bool l_772 = l_0 == uint32(1);
  if (l_772) {
    float *_tmp_12 = shared_memory_38 + uint32(0);
    wmma_store_float_c_16x16x16__row(frag_acc_48, _tmp_12, uint32(32));
    float *_tmp_13 = shared_memory_38 + uint32(16);
    wmma_store_float_c_16x16x16__row(frag_acc_49, _tmp_13, uint32(32));
  }
  const uint32 l_773 = l_6 + uint32(0);
  const uint32 l_774 = l_773 + uint32(0);
  const uint32 l_775 = l_7 + uint32(32);
  __syncthreads();

  const uint32 l_776 = threadIdxGlobal + uint32(0);
  const uint32 l_777 = l_776 / uint32(32);
  const uint32 l_778 = l_774 + l_777;
  const uint32 l_779 = l_776 % uint32(32);
  const uint32 l_780 = l_775 + l_779;
  const bool l_781 = l_778 < l_3;
  const bool l_782 = l_780 < l_4;
  const bool l_783 = l_781 && l_782;
  if (l_783) {
    const uint32 l_784 = l_778 * l_4;
    const uint32 l_785 = l_784 + l_780;
    const float l_786 = shared_memory_38[l_776];
    const __hip_bfloat16 l_787 = __hip_bfloat16(l_786);
    buffer_2[l_785] = l_787;
  }
  const uint32 l_788 = threadIdxGlobal + uint32(128);
  const uint32 l_789 = l_788 / uint32(32);
  const uint32 l_790 = l_774 + l_789;
  const uint32 l_791 = l_788 % uint32(32);
  const uint32 l_792 = l_775 + l_791;
  const bool l_793 = l_790 < l_3;
  const bool l_794 = l_792 < l_4;
  const bool l_795 = l_793 && l_794;
  if (l_795) {
    const uint32 l_796 = l_790 * l_4;
    const uint32 l_797 = l_796 + l_792;
    const float l_798 = shared_memory_38[l_788];
    const __hip_bfloat16 l_799 = __hip_bfloat16(l_798);
    buffer_2[l_797] = l_799;
  }
  const uint32 l_800 = threadIdxGlobal + uint32(256);
  const uint32 l_801 = l_800 / uint32(32);
  const uint32 l_802 = l_774 + l_801;
  const uint32 l_803 = l_800 % uint32(32);
  const uint32 l_804 = l_775 + l_803;
  const bool l_805 = l_802 < l_3;
  const bool l_806 = l_804 < l_4;
  const bool l_807 = l_805 && l_806;
  if (l_807) {
    const uint32 l_808 = l_802 * l_4;
    const uint32 l_809 = l_808 + l_804;
    const float l_810 = shared_memory_38[l_800];
    const __hip_bfloat16 l_811 = __hip_bfloat16(l_810);
    buffer_2[l_809] = l_811;
  }
  const uint32 l_812 = threadIdxGlobal + uint32(384);
  const uint32 l_813 = l_812 / uint32(32);
  const uint32 l_814 = l_774 + l_813;
  const uint32 l_815 = l_812 % uint32(32);
  const uint32 l_816 = l_775 + l_815;
  const bool l_817 = l_814 < l_3;
  const bool l_818 = l_816 < l_4;
  const bool l_819 = l_817 && l_818;
  if (l_819) {
    const uint32 l_820 = l_814 * l_4;
    const uint32 l_821 = l_820 + l_816;
    const float l_822 = shared_memory_38[l_812];
    const __hip_bfloat16 l_823 = __hip_bfloat16(l_822);
    buffer_2[l_821] = l_823;
  }
  __syncthreads();

  const bool l_824 = l_0 == uint32(1);
  if (l_824) {
    float *_tmp_14 = shared_memory_38 + uint32(0);
    wmma_store_float_c_16x16x16__row(frag_acc_50, _tmp_14, uint32(32));
    float *_tmp_15 = shared_memory_38 + uint32(16);
    wmma_store_float_c_16x16x16__row(frag_acc_51, _tmp_15, uint32(32));
  }
  const uint32 l_825 = l_6 + uint32(0);
  const uint32 l_826 = l_825 + uint32(16);
  const uint32 l_827 = l_7 + uint32(32);
  __syncthreads();

  const uint32 l_828 = threadIdxGlobal + uint32(0);
  const uint32 l_829 = l_828 / uint32(32);
  const uint32 l_830 = l_826 + l_829;
  const uint32 l_831 = l_828 % uint32(32);
  const uint32 l_832 = l_827 + l_831;
  const bool l_833 = l_830 < l_3;
  const bool l_834 = l_832 < l_4;
  const bool l_835 = l_833 && l_834;
  if (l_835) {
    const uint32 l_836 = l_830 * l_4;
    const uint32 l_837 = l_836 + l_832;
    const float l_838 = shared_memory_38[l_828];
    const __hip_bfloat16 l_839 = __hip_bfloat16(l_838);
    buffer_2[l_837] = l_839;
  }
  const uint32 l_840 = threadIdxGlobal + uint32(128);
  const uint32 l_841 = l_840 / uint32(32);
  const uint32 l_842 = l_826 + l_841;
  const uint32 l_843 = l_840 % uint32(32);
  const uint32 l_844 = l_827 + l_843;
  const bool l_845 = l_842 < l_3;
  const bool l_846 = l_844 < l_4;
  const bool l_847 = l_845 && l_846;
  if (l_847) {
    const uint32 l_848 = l_842 * l_4;
    const uint32 l_849 = l_848 + l_844;
    const float l_850 = shared_memory_38[l_840];
    const __hip_bfloat16 l_851 = __hip_bfloat16(l_850);
    buffer_2[l_849] = l_851;
  }
  const uint32 l_852 = threadIdxGlobal + uint32(256);
  const uint32 l_853 = l_852 / uint32(32);
  const uint32 l_854 = l_826 + l_853;
  const uint32 l_855 = l_852 % uint32(32);
  const uint32 l_856 = l_827 + l_855;
  const bool l_857 = l_854 < l_3;
  const bool l_858 = l_856 < l_4;
  const bool l_859 = l_857 && l_858;
  if (l_859) {
    const uint32 l_860 = l_854 * l_4;
    const uint32 l_861 = l_860 + l_856;
    const float l_862 = shared_memory_38[l_852];
    const __hip_bfloat16 l_863 = __hip_bfloat16(l_862);
    buffer_2[l_861] = l_863;
  }
  const uint32 l_864 = threadIdxGlobal + uint32(384);
  const uint32 l_865 = l_864 / uint32(32);
  const uint32 l_866 = l_826 + l_865;
  const uint32 l_867 = l_864 % uint32(32);
  const uint32 l_868 = l_827 + l_867;
  const bool l_869 = l_866 < l_3;
  const bool l_870 = l_868 < l_4;
  const bool l_871 = l_869 && l_870;
  if (l_871) {
    const uint32 l_872 = l_866 * l_4;
    const uint32 l_873 = l_872 + l_868;
    const float l_874 = shared_memory_38[l_864];
    const __hip_bfloat16 l_875 = __hip_bfloat16(l_874);
    buffer_2[l_873] = l_875;
  }
  __syncthreads();

  const bool l_876 = l_0 == uint32(2);
  if (l_876) {
    float *_tmp_16 = shared_memory_38 + uint32(0);
    wmma_store_float_c_16x16x16__row(frag_acc_48, _tmp_16, uint32(32));
    float *_tmp_17 = shared_memory_38 + uint32(16);
    wmma_store_float_c_16x16x16__row(frag_acc_49, _tmp_17, uint32(32));
  }
  const uint32 l_877 = l_6 + uint32(32);
  const uint32 l_878 = l_877 + uint32(0);
  const uint32 l_879 = l_7 + uint32(0);
  __syncthreads();

  const uint32 l_880 = threadIdxGlobal + uint32(0);
  const uint32 l_881 = l_880 / uint32(32);
  const uint32 l_882 = l_878 + l_881;
  const uint32 l_883 = l_880 % uint32(32);
  const uint32 l_884 = l_879 + l_883;
  const bool l_885 = l_882 < l_3;
  const bool l_886 = l_884 < l_4;
  const bool l_887 = l_885 && l_886;
  if (l_887) {
    const uint32 l_888 = l_882 * l_4;
    const uint32 l_889 = l_888 + l_884;
    const float l_890 = shared_memory_38[l_880];
    const __hip_bfloat16 l_891 = __hip_bfloat16(l_890);
    buffer_2[l_889] = l_891;
  }
  const uint32 l_892 = threadIdxGlobal + uint32(128);
  const uint32 l_893 = l_892 / uint32(32);
  const uint32 l_894 = l_878 + l_893;
  const uint32 l_895 = l_892 % uint32(32);
  const uint32 l_896 = l_879 + l_895;
  const bool l_897 = l_894 < l_3;
  const bool l_898 = l_896 < l_4;
  const bool l_899 = l_897 && l_898;
  if (l_899) {
    const uint32 l_900 = l_894 * l_4;
    const uint32 l_901 = l_900 + l_896;
    const float l_902 = shared_memory_38[l_892];
    const __hip_bfloat16 l_903 = __hip_bfloat16(l_902);
    buffer_2[l_901] = l_903;
  }
  const uint32 l_904 = threadIdxGlobal + uint32(256);
  const uint32 l_905 = l_904 / uint32(32);
  const uint32 l_906 = l_878 + l_905;
  const uint32 l_907 = l_904 % uint32(32);
  const uint32 l_908 = l_879 + l_907;
  const bool l_909 = l_906 < l_3;
  const bool l_910 = l_908 < l_4;
  const bool l_911 = l_909 && l_910;
  if (l_911) {
    const uint32 l_912 = l_906 * l_4;
    const uint32 l_913 = l_912 + l_908;
    const float l_914 = shared_memory_38[l_904];
    const __hip_bfloat16 l_915 = __hip_bfloat16(l_914);
    buffer_2[l_913] = l_915;
  }
  const uint32 l_916 = threadIdxGlobal + uint32(384);
  const uint32 l_917 = l_916 / uint32(32);
  const uint32 l_918 = l_878 + l_917;
  const uint32 l_919 = l_916 % uint32(32);
  const uint32 l_920 = l_879 + l_919;
  const bool l_921 = l_918 < l_3;
  const bool l_922 = l_920 < l_4;
  const bool l_923 = l_921 && l_922;
  if (l_923) {
    const uint32 l_924 = l_918 * l_4;
    const uint32 l_925 = l_924 + l_920;
    const float l_926 = shared_memory_38[l_916];
    const __hip_bfloat16 l_927 = __hip_bfloat16(l_926);
    buffer_2[l_925] = l_927;
  }
  __syncthreads();

  const bool l_928 = l_0 == uint32(2);
  if (l_928) {
    float *_tmp_18 = shared_memory_38 + uint32(0);
    wmma_store_float_c_16x16x16__row(frag_acc_50, _tmp_18, uint32(32));
    float *_tmp_19 = shared_memory_38 + uint32(16);
    wmma_store_float_c_16x16x16__row(frag_acc_51, _tmp_19, uint32(32));
  }
  const uint32 l_929 = l_6 + uint32(32);
  const uint32 l_930 = l_929 + uint32(16);
  const uint32 l_931 = l_7 + uint32(0);
  __syncthreads();

  const uint32 l_932 = threadIdxGlobal + uint32(0);
  const uint32 l_933 = l_932 / uint32(32);
  const uint32 l_934 = l_930 + l_933;
  const uint32 l_935 = l_932 % uint32(32);
  const uint32 l_936 = l_931 + l_935;
  const bool l_937 = l_934 < l_3;
  const bool l_938 = l_936 < l_4;
  const bool l_939 = l_937 && l_938;
  if (l_939) {
    const uint32 l_940 = l_934 * l_4;
    const uint32 l_941 = l_940 + l_936;
    const float l_942 = shared_memory_38[l_932];
    const __hip_bfloat16 l_943 = __hip_bfloat16(l_942);
    buffer_2[l_941] = l_943;
  }
  const uint32 l_944 = threadIdxGlobal + uint32(128);
  const uint32 l_945 = l_944 / uint32(32);
  const uint32 l_946 = l_930 + l_945;
  const uint32 l_947 = l_944 % uint32(32);
  const uint32 l_948 = l_931 + l_947;
  const bool l_949 = l_946 < l_3;
  const bool l_950 = l_948 < l_4;
  const bool l_951 = l_949 && l_950;
  if (l_951) {
    const uint32 l_952 = l_946 * l_4;
    const uint32 l_953 = l_952 + l_948;
    const float l_954 = shared_memory_38[l_944];
    const __hip_bfloat16 l_955 = __hip_bfloat16(l_954);
    buffer_2[l_953] = l_955;
  }
  const uint32 l_956 = threadIdxGlobal + uint32(256);
  const uint32 l_957 = l_956 / uint32(32);
  const uint32 l_958 = l_930 + l_957;
  const uint32 l_959 = l_956 % uint32(32);
  const uint32 l_960 = l_931 + l_959;
  const bool l_961 = l_958 < l_3;
  const bool l_962 = l_960 < l_4;
  const bool l_963 = l_961 && l_962;
  if (l_963) {
    const uint32 l_964 = l_958 * l_4;
    const uint32 l_965 = l_964 + l_960;
    const float l_966 = shared_memory_38[l_956];
    const __hip_bfloat16 l_967 = __hip_bfloat16(l_966);
    buffer_2[l_965] = l_967;
  }
  const uint32 l_968 = threadIdxGlobal + uint32(384);
  const uint32 l_969 = l_968 / uint32(32);
  const uint32 l_970 = l_930 + l_969;
  const uint32 l_971 = l_968 % uint32(32);
  const uint32 l_972 = l_931 + l_971;
  const bool l_973 = l_970 < l_3;
  const bool l_974 = l_972 < l_4;
  const bool l_975 = l_973 && l_974;
  if (l_975) {
    const uint32 l_976 = l_970 * l_4;
    const uint32 l_977 = l_976 + l_972;
    const float l_978 = shared_memory_38[l_968];
    const __hip_bfloat16 l_979 = __hip_bfloat16(l_978);
    buffer_2[l_977] = l_979;
  }
  __syncthreads();

  const bool l_980 = l_0 == uint32(3);
  if (l_980) {
    float *_tmp_20 = shared_memory_38 + uint32(0);
    wmma_store_float_c_16x16x16__row(frag_acc_48, _tmp_20, uint32(32));
    float *_tmp_21 = shared_memory_38 + uint32(16);
    wmma_store_float_c_16x16x16__row(frag_acc_49, _tmp_21, uint32(32));
  }
  const uint32 l_981 = l_6 + uint32(32);
  const uint32 l_982 = l_981 + uint32(0);
  const uint32 l_983 = l_7 + uint32(32);
  __syncthreads();

  const uint32 l_984 = threadIdxGlobal + uint32(0);
  const uint32 l_985 = l_984 / uint32(32);
  const uint32 l_986 = l_982 + l_985;
  const uint32 l_987 = l_984 % uint32(32);
  const uint32 l_988 = l_983 + l_987;
  const bool l_989 = l_986 < l_3;
  const bool l_990 = l_988 < l_4;
  const bool l_991 = l_989 && l_990;
  if (l_991) {
    const uint32 l_992 = l_986 * l_4;
    const uint32 l_993 = l_992 + l_988;
    const float l_994 = shared_memory_38[l_984];
    const __hip_bfloat16 l_995 = __hip_bfloat16(l_994);
    buffer_2[l_993] = l_995;
  }
  const uint32 l_996 = threadIdxGlobal + uint32(128);
  const uint32 l_997 = l_996 / uint32(32);
  const uint32 l_998 = l_982 + l_997;
  const uint32 l_999 = l_996 % uint32(32);
  const uint32 l_1000 = l_983 + l_999;
  const bool l_1001 = l_998 < l_3;
  const bool l_1002 = l_1000 < l_4;
  const bool l_1003 = l_1001 && l_1002;
  if (l_1003) {
    const uint32 l_1004 = l_998 * l_4;
    const uint32 l_1005 = l_1004 + l_1000;
    const float l_1006 = shared_memory_38[l_996];
    const __hip_bfloat16 l_1007 = __hip_bfloat16(l_1006);
    buffer_2[l_1005] = l_1007;
  }
  const uint32 l_1008 = threadIdxGlobal + uint32(256);
  const uint32 l_1009 = l_1008 / uint32(32);
  const uint32 l_1010 = l_982 + l_1009;
  const uint32 l_1011 = l_1008 % uint32(32);
  const uint32 l_1012 = l_983 + l_1011;
  const bool l_1013 = l_1010 < l_3;
  const bool l_1014 = l_1012 < l_4;
  const bool l_1015 = l_1013 && l_1014;
  if (l_1015) {
    const uint32 l_1016 = l_1010 * l_4;
    const uint32 l_1017 = l_1016 + l_1012;
    const float l_1018 = shared_memory_38[l_1008];
    const __hip_bfloat16 l_1019 = __hip_bfloat16(l_1018);
    buffer_2[l_1017] = l_1019;
  }
  const uint32 l_1020 = threadIdxGlobal + uint32(384);
  const uint32 l_1021 = l_1020 / uint32(32);
  const uint32 l_1022 = l_982 + l_1021;
  const uint32 l_1023 = l_1020 % uint32(32);
  const uint32 l_1024 = l_983 + l_1023;
  const bool l_1025 = l_1022 < l_3;
  const bool l_1026 = l_1024 < l_4;
  const bool l_1027 = l_1025 && l_1026;
  if (l_1027) {
    const uint32 l_1028 = l_1022 * l_4;
    const uint32 l_1029 = l_1028 + l_1024;
    const float l_1030 = shared_memory_38[l_1020];
    const __hip_bfloat16 l_1031 = __hip_bfloat16(l_1030);
    buffer_2[l_1029] = l_1031;
  }
  __syncthreads();

  const bool l_1032 = l_0 == uint32(3);
  if (l_1032) {
    float *_tmp_22 = shared_memory_38 + uint32(0);
    wmma_store_float_c_16x16x16__row(frag_acc_50, _tmp_22, uint32(32));
    float *_tmp_23 = shared_memory_38 + uint32(16);
    wmma_store_float_c_16x16x16__row(frag_acc_51, _tmp_23, uint32(32));
  }
  const uint32 l_1033 = l_6 + uint32(32);
  const uint32 l_1034 = l_1033 + uint32(16);
  const uint32 l_1035 = l_7 + uint32(32);
  __syncthreads();

  const uint32 l_1036 = threadIdxGlobal + uint32(0);
  const uint32 l_1037 = l_1036 / uint32(32);
  const uint32 l_1038 = l_1034 + l_1037;
  const uint32 l_1039 = l_1036 % uint32(32);
  const uint32 l_1040 = l_1035 + l_1039;
  const bool l_1041 = l_1038 < l_3;
  const bool l_1042 = l_1040 < l_4;
  const bool l_1043 = l_1041 && l_1042;
  if (l_1043) {
    const uint32 l_1044 = l_1038 * l_4;
    const uint32 l_1045 = l_1044 + l_1040;
    const float l_1046 = shared_memory_38[l_1036];
    const __hip_bfloat16 l_1047 = __hip_bfloat16(l_1046);
    buffer_2[l_1045] = l_1047;
  }
  const uint32 l_1048 = threadIdxGlobal + uint32(128);
  const uint32 l_1049 = l_1048 / uint32(32);
  const uint32 l_1050 = l_1034 + l_1049;
  const uint32 l_1051 = l_1048 % uint32(32);
  const uint32 l_1052 = l_1035 + l_1051;
  const bool l_1053 = l_1050 < l_3;
  const bool l_1054 = l_1052 < l_4;
  const bool l_1055 = l_1053 && l_1054;
  if (l_1055) {
    const uint32 l_1056 = l_1050 * l_4;
    const uint32 l_1057 = l_1056 + l_1052;
    const float l_1058 = shared_memory_38[l_1048];
    const __hip_bfloat16 l_1059 = __hip_bfloat16(l_1058);
    buffer_2[l_1057] = l_1059;
  }
  const uint32 l_1060 = threadIdxGlobal + uint32(256);
  const uint32 l_1061 = l_1060 / uint32(32);
  const uint32 l_1062 = l_1034 + l_1061;
  const uint32 l_1063 = l_1060 % uint32(32);
  const uint32 l_1064 = l_1035 + l_1063;
  const bool l_1065 = l_1062 < l_3;
  const bool l_1066 = l_1064 < l_4;
  const bool l_1067 = l_1065 && l_1066;
  if (l_1067) {
    const uint32 l_1068 = l_1062 * l_4;
    const uint32 l_1069 = l_1068 + l_1064;
    const float l_1070 = shared_memory_38[l_1060];
    const __hip_bfloat16 l_1071 = __hip_bfloat16(l_1070);
    buffer_2[l_1069] = l_1071;
  }
  const uint32 l_1072 = threadIdxGlobal + uint32(384);
  const uint32 l_1073 = l_1072 / uint32(32);
  const uint32 l_1074 = l_1034 + l_1073;
  const uint32 l_1075 = l_1072 % uint32(32);
  const uint32 l_1076 = l_1035 + l_1075;
  const bool l_1077 = l_1074 < l_3;
  const bool l_1078 = l_1076 < l_4;
  const bool l_1079 = l_1077 && l_1078;
  if (l_1079) {
    const uint32 l_1080 = l_1074 * l_4;
    const uint32 l_1081 = l_1080 + l_1076;
    const float l_1082 = shared_memory_38[l_1072];
    const __hip_bfloat16 l_1083 = __hip_bfloat16(l_1082);
    buffer_2[l_1081] = l_1083;
  }
  __syncthreads();
}
