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

struct info_st {
  uint32 static_meta[16];
};
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

// Load the fragment.
__device__ void wmma_load___half_b_16x16x16_row_(half16_t &frag,
                                                 const __half *value_ptr,
                                                 const uint stride) {
  uint wmmaLane = uint(threadIdx.x % 16);

#pragma unroll
  for (uint i = 0; i < 16; ++i) {
    const uint index = i * stride + wmmaLane;
    frag[i * 1] = value_ptr[index];
  }
}

extern "C" __global__ void __launch_bounds__(32) flash_attn_paged_f_(
    const __half *__restrict__ buffer_0, const __half *__restrict__ buffer_1,
    const __half *__restrict__ buffer_2, const uint32 *__restrict__ buffer_3,
    const uint32 *__restrict__ buffer_4, __half *buffer_5,
    const float *__restrict__ buffer_6, const uint32 *__restrict__ buffer_7,
    const info_st *__restrict__ info_ptr) {
  uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x +
                           threadIdx.z * (blockDim.x * blockDim.y);
  uint32 blockIdxGlobal = (blockIdx.z * gridDim.y * gridDim.x) +
                          (blockIdx.y * gridDim.x) + blockIdx.x;
  extern __shared__ __align__(4) uchar dynamic_shared_mem[];
  const info_st &info = *info_ptr;
  const uint32 *dynamic_meta = reinterpret_cast<const uint32 *>(
      reinterpret_cast<const char *>(info_ptr) + sizeof(info_st));

  // Shared array size: 1024, 2048 bytes
  __half *shared_memory_34 = reinterpret_cast<__half *>(&dynamic_shared_mem[0]);
  // Shared array size: 256, 1024 bytes
  float *shared_memory_35 =
      reinterpret_cast<float *>(&dynamic_shared_mem[2048]);
  // Shared array size: 1024, 4096 bytes
  float *shared_memory_36 =
      reinterpret_cast<float *>(&dynamic_shared_mem[3072]);
  // Shared array size: 16, 64 bytes
  float *shared_memory_37 =
      reinterpret_cast<float *>(&dynamic_shared_mem[7168]);
  // Shared array size: 16, 64 bytes
  float *shared_memory_38 =
      reinterpret_cast<float *>(&dynamic_shared_mem[7232]);
  // Shared array size: 16, 64 bytes
  float *shared_memory_39 =
      reinterpret_cast<float *>(&dynamic_shared_mem[7296]);
  // Shared array size: 512, 1024 bytes
  __half *shared_memory_80 =
      reinterpret_cast<__half *>(&dynamic_shared_mem[7360]);
  // Shared array size: 256, 512 bytes
  __half *shared_memory_204 =
      reinterpret_cast<__half *>(&dynamic_shared_mem[8384]);
  // Shared array size: 512, 1024 bytes
  __half *shared_memory_205 =
      reinterpret_cast<__half *>(&dynamic_shared_mem[8896]);
  // Shared array size: 256, 1024 bytes
  float *shared_memory_206 =
      reinterpret_cast<float *>(&dynamic_shared_mem[9920]);
  uint32 l_mut_26;
  float l_mut_51;
  uint32 l_mut_62;
  uint32 l_mut_71;
  __half l_mut_103;
  __half l_mut_117;
  float l_mut_189;
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
  const uint32 l_33 = l_32 * uint32(64);

  for (uint32 l_mut_40 = uint32(0); l_mut_40 < uint32(32); ++l_mut_40) {
    const uint32 l_41 = threadIdxGlobal * uint32(32);
    const uint32 l_42 = l_41 + l_mut_40;
    shared_memory_36[l_42] = float(0.0);
  }
  const bool l_43 = threadIdxGlobal < uint32(16);
  if (l_43) {
    shared_memory_37[threadIdxGlobal] = float(-3.3999999521443642e38);
    shared_memory_38[threadIdxGlobal] = float(0.0);
  }

  for (uint32 l_mut_44 = uint32(0); l_mut_44 < uint32(32); ++l_mut_44) {
    const uint32 l_45 = threadIdxGlobal * uint32(32);
    const uint32 l_46 = l_45 + l_mut_44;
    const uint32 l_47 = l_46 / uint32(64);
    const uint32 l_48 = l_46 % uint32(64);
    const uint32 l_49 = l_22 * uint32(16);
    const uint32 l_50 = l_49 + l_47;
    l_mut_51 = float(0.0);
    const bool l_52 = l_50 < l_4;
    if (l_52) {
      const uint32 l_53 = l_50 * uint32(64);
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
    float8_t frag_acc_81;
    wmma_fill_float_c_16x16x16_(frag_acc_81, float(0.0));

    for (uint32 l_mut_82 = uint32(0); l_mut_82 < uint32(2); ++l_mut_82) {
      const uint32 l_83 = l_mut_82 * uint32(32);

      for (uint32 l_mut_84 = uint32(0); l_mut_84 < uint32(16); ++l_mut_84) {
        const uint32 l_85 = threadIdxGlobal * uint32(16);
        const uint32 l_86 = l_85 + l_mut_84;
        const uint32 l_87 = l_86 / uint32(32);
        const uint32 l_88 = l_86 % uint32(32);
        const uint32 l_89 = l_mut_79 * uint32(16);
        const uint32 l_90 = uint32(0) * uint32(16);
        const uint32 l_91 = l_89 + l_90;
        const uint32 l_92 = l_91 + l_87;
        l_mut_51 = float(0.0);
        const bool l_93 = l_92 < l_25;
        if (l_93) {
          const uint32 l_94 = l_83 + l_88;
          const bool l_95 = l_10 == uint32(2);
          if (l_95) {
            const uint32 l_96 = l_18 * l_6;
            const uint32 l_97 = l_96 + l_24;
            const uint32 l_98 = l_97 * l_8;
            const uint32 l_99 = l_98 + l_92;
            const uint32 l_100 = l_99 * uint32(64);
            const uint32 l_101 = l_100 + l_94;
            const __half l_102 = buffer_1[l_101];
            l_mut_103 = l_102;
          } else {
            const uint32 l_104 = l_18 * l_9;
            const uint32 l_105 = l_92 / l_8;
            const uint32 l_106 = l_104 + l_105;
            const uint32 l_107 = buffer_3[l_106];
            const uint32 l_108 = l_92 % l_8;
            const bool l_109 = l_10 == uint32(0);
            if (l_109) {
              const uint32 l_110 = l_107 * l_6;
              const uint32 l_111 = l_110 + l_24;
              const uint32 l_112 = l_111 * l_8;
              const uint32 l_113 = l_112 + l_108;
              const uint32 l_114 = l_113 * uint32(64);
              const uint32 l_115 = l_114 + l_94;
              const __half l_116 = buffer_1[l_115];
              l_mut_117 = l_116;
            } else {
              const uint32 l_118 = uint32(64) / l_11;
              const uint32 l_119 = l_107 * l_6;
              const uint32 l_120 = l_119 + l_24;
              const uint32 l_121 = l_120 * l_118;
              const uint32 l_122 = l_94 / l_11;
              const uint32 l_123 = l_121 + l_122;
              const uint32 l_124 = l_123 * l_8;
              const uint32 l_125 = l_124 + l_108;
              const uint32 l_126 = l_125 * l_11;
              const uint32 l_127 = l_94 % l_11;
              const uint32 l_128 = l_126 + l_127;
              const __half l_129 = buffer_1[l_128];
              l_mut_117 = l_129;
            }
            l_mut_103 = l_mut_117;
          }
          const float l_130 = float(l_mut_103);
          l_mut_51 = l_130;
        }
        const __half l_131 = __half(l_mut_51);
        shared_memory_80[l_86] = l_131;
      }
      __syncthreads();

      for (uint32 l_mut_132 = uint32(0); l_mut_132 < uint32(2); ++l_mut_132) {
        half16_t frag_a_139;
        half16_t frag_b_145;
        const uint32 l_133 = l_mut_132 * uint32(16);
        const uint32 l_134 = l_83 + l_133;
        const uint32 l_135 = 1024;
        const uint32 l_136 = l_135 - uint32(0);
        const uint32 l_137 = uint32(1024) - l_134;
        const uint32 l_138 = l_134 + uint32(0);
        __half *_tmp_0 = shared_memory_34 + l_138;
        wmma_load___half_a_16x16x16_row_(frag_a_139, _tmp_0, uint32(64));
        const uint32 l_140 = l_mut_132 * uint32(16);
        const uint32 l_141 = 512;
        const uint32 l_142 = l_141 - uint32(0);
        const uint32 l_143 = uint32(512) - l_140;
        const uint32 l_144 = l_140 + uint32(0);
        __half *_tmp_1 = shared_memory_80 + l_144;
        wmma_load___half_b_16x16x16_col_(frag_b_145, _tmp_1, uint32(32));
        wmma_execute_16x16x16___half_float(frag_a_139, frag_b_145, frag_acc_81,
                                           frag_acc_81);
      }
      __syncthreads();
    }
    const uint32 l_146 = uint32(0) * uint32(16);
    const uint32 l_147 = 256;
    const uint32 l_148 = l_147 - uint32(0);
    const uint32 l_149 = uint32(256) - l_146;
    const uint32 l_150 = l_146 + uint32(0);
    float *_tmp_2 = shared_memory_35 + l_150;
    wmma_store_float_c_16x16x16__row(frag_acc_81, _tmp_2, uint32(16));
    __syncthreads();

    for (uint32 l_mut_151 = uint32(0); l_mut_151 < uint32(8); ++l_mut_151) {
      const uint32 l_152 = threadIdxGlobal * uint32(8);
      const uint32 l_153 = l_152 + l_mut_151;
      const uint32 l_154 = l_153 / uint32(16);
      const uint32 l_155 = l_153 % uint32(16);
      const uint32 l_156 = l_22 * uint32(16);
      const uint32 l_157 = l_156 + l_154;
      const uint32 l_158 = l_mut_79 * uint32(16);
      const uint32 l_159 = l_158 + l_155;
      const uint32 l_160 = l_mut_26 + l_157;
      const bool l_161 = l_7 == uint32(0);
      const bool l_162 = l_159 <= l_160;
      const bool l_163 = l_161 || l_162;
      const bool l_164 = l_12 == uint32(0);
      const uint32 l_165 = l_159 + l_12;
      const bool l_166 = l_165 > l_160;
      const bool l_167 = l_164 || l_166;
      const bool l_168 = l_157 < l_4;
      const bool l_169 = l_159 < l_25;
      const bool l_170 = l_168 && l_169;
      const bool l_171 = l_170 && l_163;
      const bool l_172 = l_171 && l_167;
      if (l_172) {
        const float l_173 = shared_memory_35[l_153];
        const float l_174 = l_173 * l_2;
        l_mut_51 = l_174;
        const bool l_175 = l_3 > float(0.0);
        if (l_175) {
          const float l_176 = l_mut_51 / l_3;
          const float l_177 = tanh(l_176);
          const float l_178 = l_3 * l_177;
          l_mut_51 = l_178;
        }
        shared_memory_35[l_153] = l_mut_51;
      } else {
        shared_memory_35[l_153] = float(-3.3999999521443642e38);
      }
    }
    __syncthreads();

    const bool l_179 = threadIdxGlobal < uint32(16);
    if (l_179) {
      const uint32 l_180 = l_22 * uint32(16);
      const uint32 l_181 = l_180 + threadIdxGlobal;
      const bool l_182 = l_181 < l_4;
      if (l_182) {
        l_mut_51 = float(-3.3999999521443642e38);

        for (uint32 l_mut_183 = uint32(0); l_mut_183 < uint32(16);
             ++l_mut_183) {
          const uint32 l_184 = threadIdxGlobal * uint32(16);
          const uint32 l_185 = l_184 + l_mut_183;
          const float l_186 = shared_memory_35[l_185];
          const bool l_187 = l_186 > l_mut_51;
          if (l_187) {
            l_mut_51 = l_186;
          }
        }
        const float l_188 = shared_memory_37[threadIdxGlobal];
        l_mut_189 = l_188;
        const bool l_190 = l_mut_51 > l_mut_189;
        if (l_190) {
          l_mut_189 = l_mut_51;
        }
        const float l_191 = l_188 - l_mut_189;
        const float l_192 = exp(l_191);
        l_mut_51 = float(0.0);

        for (uint32 l_mut_193 = uint32(0); l_mut_193 < uint32(16);
             ++l_mut_193) {
          const uint32 l_194 = threadIdxGlobal * uint32(16);
          const uint32 l_195 = l_194 + l_mut_193;
          const float l_196 = shared_memory_35[l_195];
          const float l_197 = l_196 - l_mut_189;
          const float l_198 = exp(l_197);
          const uint32 l_199 = threadIdxGlobal * uint32(16);
          const uint32 l_200 = l_199 + l_mut_193;
          shared_memory_35[l_200] = l_198;
          l_mut_51 = l_mut_51 + l_198;
        }
        const float l_201 = shared_memory_38[threadIdxGlobal];
        const float l_202 = l_201 * l_192;
        const float l_203 = l_202 + l_mut_51;
        shared_memory_38[threadIdxGlobal] = l_203;
        shared_memory_37[threadIdxGlobal] = l_mut_189;
        shared_memory_39[threadIdxGlobal] = l_192;
      } else {
        shared_memory_39[threadIdxGlobal] = float(1.0);
      }
    }
    __syncthreads();

    for (uint32 l_mut_207 = uint32(0); l_mut_207 < uint32(8); ++l_mut_207) {
      const uint32 l_208 = threadIdxGlobal * uint32(8);
      const uint32 l_209 = l_208 + l_mut_207;
      const uint32 l_210 = l_209 / uint32(16);
      const uint32 l_211 = l_22 * uint32(16);
      const uint32 l_212 = l_211 + l_210;
      l_mut_189 = float(0.0);
      const bool l_213 = l_212 < l_4;
      if (l_213) {
        const float l_214 = shared_memory_35[l_209];
        l_mut_189 = l_214;
      }
      const __half l_215 = __half(l_mut_189);
      shared_memory_204[l_209] = l_215;
    }
    __syncthreads();

    for (uint32 l_mut_216 = uint32(0); l_mut_216 < uint32(2); ++l_mut_216) {
      const uint32 l_217 = l_mut_216 * uint32(32);

      for (uint32 l_mut_218 = uint32(0); l_mut_218 < uint32(16); ++l_mut_218) {
        const uint32 l_219 = threadIdxGlobal * uint32(16);
        const uint32 l_220 = l_219 + l_mut_218;
        const uint32 l_221 = l_220 / uint32(32);
        const uint32 l_222 = l_220 % uint32(32);
        const uint32 l_223 = l_mut_79 * uint32(16);
        const uint32 l_224 = l_223 + l_221;
        l_mut_189 = float(0.0);
        const bool l_225 = l_224 < l_25;
        if (l_225) {
          const uint32 l_226 = l_217 + l_222;
          const bool l_227 = l_10 == uint32(2);
          if (l_227) {
            const uint32 l_228 = l_18 * l_6;
            const uint32 l_229 = l_228 + l_24;
            const uint32 l_230 = l_229 * l_8;
            const uint32 l_231 = l_230 + l_224;
            const uint32 l_232 = l_231 * uint32(64);
            const uint32 l_233 = l_232 + l_226;
            const __half l_234 = buffer_2[l_233];
            l_mut_117 = l_234;
          } else {
            const uint32 l_235 = l_18 * l_9;
            const uint32 l_236 = l_224 / l_8;
            const uint32 l_237 = l_235 + l_236;
            const uint32 l_238 = buffer_3[l_237];
            const uint32 l_239 = l_224 % l_8;
            const bool l_240 = l_10 == uint32(0);
            if (l_240) {
              const uint32 l_241 = l_238 * l_6;
              const uint32 l_242 = l_241 + l_24;
              const uint32 l_243 = l_242 * l_8;
              const uint32 l_244 = l_243 + l_239;
              const uint32 l_245 = l_244 * uint32(64);
              const uint32 l_246 = l_245 + l_226;
              const __half l_247 = buffer_2[l_246];
              l_mut_103 = l_247;
            } else {
              const uint32 l_248 = l_238 * l_6;
              const uint32 l_249 = l_248 + l_24;
              const uint32 l_250 = l_249 * uint32(64);
              const uint32 l_251 = l_250 + l_226;
              const uint32 l_252 = l_251 * l_8;
              const uint32 l_253 = l_252 + l_239;
              const __half l_254 = buffer_2[l_253];
              l_mut_103 = l_254;
            }
            l_mut_117 = l_mut_103;
          }
          const float l_255 = float(l_mut_117);
          l_mut_189 = l_255;
        }
        const __half l_256 = __half(l_mut_189);
        shared_memory_205[l_220] = l_256;
      }
      __syncthreads();

      for (uint32 l_mut_257 = uint32(0); l_mut_257 < uint32(2); ++l_mut_257) {
        float8_t frag_acc_258;
        half16_t frag_a_264;
        half16_t frag_b_273;
        wmma_fill_float_c_16x16x16_(frag_acc_258, float(0.0));
        const uint32 l_259 = uint32(0) * uint32(16);
        const uint32 l_260 = 256;
        const uint32 l_261 = l_260 - uint32(0);
        const uint32 l_262 = uint32(256) - l_259;
        const uint32 l_263 = l_259 + uint32(0);
        __half *_tmp_3 = shared_memory_204 + l_263;
        wmma_load___half_a_16x16x16_row_(frag_a_264, _tmp_3, uint32(16));
        const uint32 l_265 = uint32(0) * uint32(16);
        const uint32 l_266 = l_265 * uint32(32);
        const uint32 l_267 = l_mut_257 * uint32(16);
        const uint32 l_268 = l_266 + l_267;
        const uint32 l_269 = 512;
        const uint32 l_270 = l_269 - uint32(0);
        const uint32 l_271 = uint32(512) - l_268;
        const uint32 l_272 = l_268 + uint32(0);
        __half *_tmp_4 = shared_memory_205 + l_272;
        wmma_load___half_b_16x16x16_row_(frag_b_273, _tmp_4, uint32(32));
        wmma_execute_16x16x16___half_float(frag_a_264, frag_b_273, frag_acc_258,
                                           frag_acc_258);
        const uint32 l_274 = 256;
        const uint32 l_275 = l_274 - uint32(0);
        float *_tmp_5 = shared_memory_206 + uint32(0);
        wmma_store_float_c_16x16x16__row(frag_acc_258, _tmp_5, uint32(16));
        __syncthreads();

        for (uint32 l_mut_276 = uint32(0); l_mut_276 < uint32(8); ++l_mut_276) {
          const uint32 l_277 = threadIdxGlobal * uint32(8);
          const uint32 l_278 = l_277 + l_mut_276;
          const uint32 l_279 = l_278 / uint32(16);
          const uint32 l_280 = l_278 % uint32(16);
          const uint32 l_281 = l_279 * uint32(64);
          const uint32 l_282 = l_281 + l_217;
          const uint32 l_283 = l_mut_257 * uint32(16);
          const uint32 l_284 = l_282 + l_283;
          const uint32 l_285 = l_284 + l_280;
          const float l_286 = shared_memory_36[l_285];
          const float l_287 = shared_memory_39[l_279];
          const float l_288 = l_286 * l_287;
          const float l_289 = shared_memory_206[l_278];
          const float l_290 = l_288 + l_289;
          shared_memory_36[l_285] = l_290;
        }
        __syncthreads();
      }
    }
    __syncthreads();
  }

  for (uint32 l_mut_291 = uint32(0); l_mut_291 < uint32(32); ++l_mut_291) {
    const uint32 l_292 = threadIdxGlobal * uint32(32);
    const uint32 l_293 = l_292 + l_mut_291;
    const uint32 l_294 = l_293 / uint32(64);
    const uint32 l_295 = l_293 % uint32(64);
    const uint32 l_296 = l_22 * uint32(16);
    const uint32 l_297 = l_296 + l_294;
    const bool l_298 = l_297 < l_4;
    if (l_298) {
      l_mut_189 = float(1.0);
      const float l_299 = shared_memory_38[l_294];
      const bool l_300 = l_299 > float(0.0);
      if (l_300) {
        const float l_301 = shared_memory_38[l_294];
        l_mut_189 = l_301;
      }
      const uint32 l_302 = l_297 * uint32(64);
      const uint32 l_303 = l_33 + l_302;
      const uint32 l_304 = l_303 + l_295;
      const float l_305 = shared_memory_36[l_293];
      const float l_306 = l_305 / l_mut_189;
      const __half l_307 = __half(l_306);
      buffer_5[l_304] = l_307;
    }
  }
}
