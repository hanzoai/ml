#include <hip/hip_bf16.h>
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
  uint32 static_meta[8];
};

extern "C" __global__ void __launch_bounds__(16)
    fwht_hadamard_f_(const __hip_bfloat16 *__restrict__ buffer_0,
                     const float *__restrict__ buffer_1,
                     __hip_bfloat16 *buffer_2,
                     const uint32 *__restrict__ buffer_3,
                     const info_st *__restrict__ info_ptr) {
  uint32 threadIdxGlobal = threadIdx.x + threadIdx.y * blockDim.x +
                           threadIdx.z * (blockDim.x * blockDim.y);
  uint32 blockIdxGlobal = (blockIdx.z * gridDim.y * gridDim.x) +
                          (blockIdx.y * gridDim.x) + blockIdx.x;
  extern __shared__ __align__(4) uchar dynamic_shared_mem[];
  const info_st &info = *info_ptr;
  const uint32 *dynamic_meta = reinterpret_cast<const uint32 *>(
      reinterpret_cast<const char *>(info_ptr) + sizeof(info_st));

  // Shared array size: 64, 256 bytes
  float *shared_memory_9 = reinterpret_cast<float *>(&dynamic_shared_mem[0]);
  uint32 l_mut_6;
  float l_mut_15;
  float l_mut_20;
  float l_mut_25;
  float l_mut_30;
  const uint32 l_0 = buffer_3[uint32(0)];
  const bool l_1 = blockIdxGlobal < l_0;
  if (l_1) {
    const uint32 l_2 = buffer_3[uint32(1)];
    const uint32 l_3 = buffer_3[uint32(2)];
    const uint32 l_4 = buffer_3[uint32(3)];
    const uint32 l_5 = blockIdxGlobal * uint32(64);
    l_mut_6 = uint32(0);
    const bool l_7 = l_3 != uint32(0);
    if (l_7) {
      const uint32 l_8 = l_5 % l_2;
      l_mut_6 = l_8;
    }
    const uint32 l_10 = threadIdxGlobal + uint32(0);
    const uint32 l_11 = l_10 * uint32(4);
    const uint32 l_12 = l_5 + l_11;
    const __hip_bfloat16 l_13 = buffer_0[l_12];
    const float l_14 = float(l_13);
    l_mut_15 = l_14;
    const uint32 l_16 = l_5 + l_11;
    const uint32 l_17 = l_16 + uint32(1);
    const __hip_bfloat16 l_18 = buffer_0[l_17];
    const float l_19 = float(l_18);
    l_mut_20 = l_19;
    const uint32 l_21 = l_5 + l_11;
    const uint32 l_22 = l_21 + uint32(2);
    const __hip_bfloat16 l_23 = buffer_0[l_22];
    const float l_24 = float(l_23);
    l_mut_25 = l_24;
    const uint32 l_26 = l_5 + l_11;
    const uint32 l_27 = l_26 + uint32(3);
    const __hip_bfloat16 l_28 = buffer_0[l_27];
    const float l_29 = float(l_28);
    l_mut_30 = l_29;
    const bool l_31 = l_3 != uint32(0);
    const bool l_32 = l_4 == uint32(0);
    const bool l_33 = l_31 && l_32;
    if (l_33) {
      const uint32 l_34 = l_mut_6 + l_11;
      const float l_35 = buffer_1[l_34];
      l_mut_15 = l_mut_15 * l_35;
      const uint32 l_36 = l_mut_6 + l_11;
      const uint32 l_37 = l_36 + uint32(1);
      const float l_38 = buffer_1[l_37];
      l_mut_20 = l_mut_20 * l_38;
      const uint32 l_39 = l_mut_6 + l_11;
      const uint32 l_40 = l_39 + uint32(2);
      const float l_41 = buffer_1[l_40];
      l_mut_25 = l_mut_25 * l_41;
      const uint32 l_42 = l_mut_6 + l_11;
      const uint32 l_43 = l_42 + uint32(3);
      const float l_44 = buffer_1[l_43];
      l_mut_30 = l_mut_30 * l_44;
    }
    const float l_45 = l_mut_15 + l_mut_20;
    const float l_46 = l_mut_15 - l_mut_20;
    const float l_47 = l_mut_25 + l_mut_30;
    const float l_48 = l_mut_25 - l_mut_30;
    const float l_49 = l_45 + l_47;
    shared_memory_9[l_11] = l_49;
    const uint32 l_50 = l_11 + uint32(1);
    const float l_51 = l_46 + l_48;
    shared_memory_9[l_50] = l_51;
    const uint32 l_52 = l_11 + uint32(2);
    const float l_53 = l_45 - l_47;
    shared_memory_9[l_52] = l_53;
    const uint32 l_54 = l_11 + uint32(3);
    const float l_55 = l_46 - l_48;
    shared_memory_9[l_54] = l_55;
    __syncthreads();

    const uint32 l_56 = threadIdxGlobal + uint32(0);
    const uint32 l_57 = l_56 >> uint32(2);
    const uint32 l_58 = l_56 & uint32(3);
    const uint32 l_59 = l_57 * uint32(8);
    const uint32 l_60 = l_59 + l_58;
    const uint32 l_61 = l_60 + uint32(4);
    const float l_62 = shared_memory_9[l_60];
    const float l_63 = shared_memory_9[l_61];
    const float l_64 = l_62 + l_63;
    shared_memory_9[l_60] = l_64;
    const float l_65 = l_62 - l_63;
    shared_memory_9[l_61] = l_65;
    const uint32 l_66 = threadIdxGlobal + uint32(16);
    const uint32 l_67 = l_66 >> uint32(2);
    const uint32 l_68 = l_66 & uint32(3);
    const uint32 l_69 = l_67 * uint32(8);
    const uint32 l_70 = l_69 + l_68;
    const uint32 l_71 = l_70 + uint32(4);
    const float l_72 = shared_memory_9[l_70];
    const float l_73 = shared_memory_9[l_71];
    const float l_74 = l_72 + l_73;
    shared_memory_9[l_70] = l_74;
    const float l_75 = l_72 - l_73;
    shared_memory_9[l_71] = l_75;
    __syncthreads();

    const uint32 l_76 = threadIdxGlobal + uint32(0);
    const uint32 l_77 = l_76 >> uint32(3);
    const uint32 l_78 = l_76 & uint32(7);
    const uint32 l_79 = l_77 * uint32(16);
    const uint32 l_80 = l_79 + l_78;
    const uint32 l_81 = l_80 + uint32(8);
    const float l_82 = shared_memory_9[l_80];
    const float l_83 = shared_memory_9[l_81];
    const float l_84 = l_82 + l_83;
    shared_memory_9[l_80] = l_84;
    const float l_85 = l_82 - l_83;
    shared_memory_9[l_81] = l_85;
    const uint32 l_86 = threadIdxGlobal + uint32(16);
    const uint32 l_87 = l_86 >> uint32(3);
    const uint32 l_88 = l_86 & uint32(7);
    const uint32 l_89 = l_87 * uint32(16);
    const uint32 l_90 = l_89 + l_88;
    const uint32 l_91 = l_90 + uint32(8);
    const float l_92 = shared_memory_9[l_90];
    const float l_93 = shared_memory_9[l_91];
    const float l_94 = l_92 + l_93;
    shared_memory_9[l_90] = l_94;
    const float l_95 = l_92 - l_93;
    shared_memory_9[l_91] = l_95;
    __syncthreads();

    const uint32 l_96 = threadIdxGlobal + uint32(0);
    const uint32 l_97 = l_96 >> uint32(4);
    const uint32 l_98 = l_96 & uint32(15);
    const uint32 l_99 = l_97 * uint32(32);
    const uint32 l_100 = l_99 + l_98;
    const uint32 l_101 = l_100 + uint32(16);
    const float l_102 = shared_memory_9[l_100];
    const float l_103 = shared_memory_9[l_101];
    const float l_104 = l_102 + l_103;
    shared_memory_9[l_100] = l_104;
    const float l_105 = l_102 - l_103;
    shared_memory_9[l_101] = l_105;
    const uint32 l_106 = threadIdxGlobal + uint32(16);
    const uint32 l_107 = l_106 >> uint32(4);
    const uint32 l_108 = l_106 & uint32(15);
    const uint32 l_109 = l_107 * uint32(32);
    const uint32 l_110 = l_109 + l_108;
    const uint32 l_111 = l_110 + uint32(16);
    const float l_112 = shared_memory_9[l_110];
    const float l_113 = shared_memory_9[l_111];
    const float l_114 = l_112 + l_113;
    shared_memory_9[l_110] = l_114;
    const float l_115 = l_112 - l_113;
    shared_memory_9[l_111] = l_115;
    __syncthreads();

    const uint32 l_116 = threadIdxGlobal + uint32(0);
    const uint32 l_117 = l_116 >> uint32(5);
    const uint32 l_118 = l_116 & uint32(31);
    const uint32 l_119 = l_117 * uint32(64);
    const uint32 l_120 = l_119 + l_118;
    const uint32 l_121 = l_120 + uint32(32);
    const float l_122 = shared_memory_9[l_120];
    const float l_123 = shared_memory_9[l_121];
    const float l_124 = l_122 + l_123;
    shared_memory_9[l_120] = l_124;
    const float l_125 = l_122 - l_123;
    shared_memory_9[l_121] = l_125;
    const uint32 l_126 = threadIdxGlobal + uint32(16);
    const uint32 l_127 = l_126 >> uint32(5);
    const uint32 l_128 = l_126 & uint32(31);
    const uint32 l_129 = l_127 * uint32(64);
    const uint32 l_130 = l_129 + l_128;
    const uint32 l_131 = l_130 + uint32(32);
    const float l_132 = shared_memory_9[l_130];
    const float l_133 = shared_memory_9[l_131];
    const float l_134 = l_132 + l_133;
    shared_memory_9[l_130] = l_134;
    const float l_135 = l_132 - l_133;
    shared_memory_9[l_131] = l_135;
    __syncthreads();

    const uint32 l_136 = threadIdxGlobal + uint32(0);
    const uint32 l_137 = l_136 * uint32(4);
    const float l_138 = shared_memory_9[l_137];
    const float l_139 = l_138 * float(0.125);
    l_mut_30 = l_139;
    const uint32 l_140 = l_137 + uint32(1);
    const float l_141 = shared_memory_9[l_140];
    const float l_142 = l_141 * float(0.125);
    l_mut_25 = l_142;
    const uint32 l_143 = l_137 + uint32(2);
    const float l_144 = shared_memory_9[l_143];
    const float l_145 = l_144 * float(0.125);
    l_mut_20 = l_145;
    const uint32 l_146 = l_137 + uint32(3);
    const float l_147 = shared_memory_9[l_146];
    const float l_148 = l_147 * float(0.125);
    l_mut_15 = l_148;
    const bool l_149 = l_3 != uint32(0);
    const bool l_150 = l_4 != uint32(0);
    const bool l_151 = l_149 && l_150;
    if (l_151) {
      const uint32 l_152 = l_mut_6 + l_137;
      const float l_153 = buffer_1[l_152];
      l_mut_30 = l_mut_30 * l_153;
      const uint32 l_154 = l_mut_6 + l_137;
      const uint32 l_155 = l_154 + uint32(1);
      const float l_156 = buffer_1[l_155];
      l_mut_25 = l_mut_25 * l_156;
      const uint32 l_157 = l_mut_6 + l_137;
      const uint32 l_158 = l_157 + uint32(2);
      const float l_159 = buffer_1[l_158];
      l_mut_20 = l_mut_20 * l_159;
      const uint32 l_160 = l_mut_6 + l_137;
      const uint32 l_161 = l_160 + uint32(3);
      const float l_162 = buffer_1[l_161];
      l_mut_15 = l_mut_15 * l_162;
    }
    const uint32 l_163 = l_5 + l_137;
    const __hip_bfloat16 l_164 = __hip_bfloat16(l_mut_30);
    buffer_2[l_163] = l_164;
    const uint32 l_165 = l_5 + l_137;
    const uint32 l_166 = l_165 + uint32(1);
    const __hip_bfloat16 l_167 = __hip_bfloat16(l_mut_25);
    buffer_2[l_166] = l_167;
    const uint32 l_168 = l_5 + l_137;
    const uint32 l_169 = l_168 + uint32(2);
    const __hip_bfloat16 l_170 = __hip_bfloat16(l_mut_20);
    buffer_2[l_169] = l_170;
    const uint32 l_171 = l_5 + l_137;
    const uint32 l_172 = l_171 + uint32(3);
    const __hip_bfloat16 l_173 = __hip_bfloat16(l_mut_15);
    buffer_2[l_172] = l_173;
  }
}
